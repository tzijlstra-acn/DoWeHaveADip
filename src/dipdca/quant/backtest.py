"""Core backtest engine: DCA, wait-for-dip, and tiered-dip strategies."""

from __future__ import annotations

from datetime import date

import pandas as pd

from dipdca.models import SimulationParams, StrategyResult
from dipdca.quant.contributions import build_contribution_schedule
from dipdca.quant.drawdown import drawdown as compute_drawdown
from dipdca.quant.metrics import (
    cagr_from_wealth,
    portfolio_returns,
    sharpe_ratio,
    sortino_ratio,
    time_in_market_pct,
)
from dipdca.quant.xirr import xirr


def _apply_cost(
    amount: float,
    fixed_fee: float,
    pct_fee: float,
    slippage: float,
) -> tuple[float, float]:
    """Return (amount_invested_net, total_cost)."""
    cost = fixed_fee + amount * pct_fee + amount * slippage
    net = max(0.0, amount - cost)
    return net, cost


def _build_ledger_template(price_data: pd.DataFrame) -> pd.DataFrame:
    """Create an empty ledger DataFrame aligned to trading days."""
    idx = price_data.index
    return pd.DataFrame(
        {
            "price": price_data["adj_close"],
            "cash": 0.0,
            "units": 0.0,
            "market_value": 0.0,
            "total_wealth": 0.0,
            "deployed": 0.0,
            "fees": 0.0,
            "interest": 0.0,
            "dd": 0.0,
        },
        index=idx,
    )


def run_dca(
    price_data: pd.DataFrame,
    params: SimulationParams,
    cash_rate_series: pd.Series | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """DCA strategy: deploy every contribution on the next available trading day.

    Critical rules:
    - Contribution arrives on payday, invests on next trading day.
    - No drawdown logic — always deploy.
    - Interest accrues on uninvested cash at overnight rate.

    Args:
        price_data: DataFrame with DatetimeIndex and adj_close column.
        params: Simulation parameters.
        cash_rate_series: Daily annualized cash interest rate (optional).

    Returns:
        Tuple of (StrategyResult, day-by-day ledger DataFrame).
    """
    prices = price_data["adj_close"].copy()
    trading_days = prices.index

    schedule = build_contribution_schedule(
        params.start_date,
        params.end_date,
        params.monthly_contribution,
        params.payday,
        trading_days,
    )

    # Map invest_date → amount to contribute
    invest_map: dict[pd.Timestamp, float] = {}
    for _, row in schedule.iterrows():
        invest_date = row["invest_date"]
        invest_map[invest_date] = invest_map.get(invest_date, 0.0) + row["amount"]

    # Daily default cash rate
    if cash_rate_series is None:
        cash_rate = params.cash_rate_override or 0.0
        cash_rate_series = pd.Series(cash_rate, index=trading_days)
    else:
        cash_rate_series = cash_rate_series.reindex(trading_days, method="ffill").fillna(0.0)

    ledger = _build_ledger_template(price_data)
    cash = params.initial_investment
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []

    if params.initial_investment > 0:
        # Deploy initial investment on first trading day
        first_day = trading_days[0]
        p = prices.iloc[0]
        net, cost = _apply_cost(cash, params.fixed_fee, params.pct_fee, params.slippage)
        units = net / p if p > 0 else 0.0
        cash = 0.0
        total_fees += cost
        n_deployments += 1
        cash_flows.append((first_day.date(), -params.initial_investment))

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        # Accrue interest on uninvested cash
        if prev_dt is not None and cash > 0:
            days_elapsed = (dt - prev_dt).days
            r = cash_rate_series.iloc[i - 1]
            if r > -1 and days_elapsed > 0:
                new_cash = cash * (1 + r) ** (days_elapsed / 365)
                earned = new_cash - cash
                total_interest += earned
                cash = new_cash
                ledger.at[dt, "interest"] = earned

        # Receive contribution
        if dt in invest_map:
            contrib = invest_map[dt]
            cash += contrib
            cash_flows.append((dt.date(), -contrib))

            # DCA: deploy immediately
            p = prices.iloc[i]
            if p > 0 and cash > 0:
                net, cost = _apply_cost(cash, params.fixed_fee, params.pct_fee, params.slippage)
                units += net / p
                total_fees += cost
                n_deployments += 1
                ledger.at[dt, "deployed"] = cash
                ledger.at[dt, "fees"] = cost
                cash = 0.0

        market_val = units * prices.iloc[i]
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val
        prev_dt = dt

    # XIRR: add terminal receipt
    final_wealth = float(ledger["total_wealth"].iloc[-1])
    if cash_flows and final_wealth > 0:
        cash_flows.append((trading_days[-1].date(), final_wealth))

    total_contributions = schedule["amount"].sum() + params.initial_investment
    dd_series = compute_drawdown(ledger["total_wealth"])
    ledger["dd"] = dd_series

    returns = portfolio_returns(ledger["total_wealth"])
    years = (
        pd.Timestamp(params.end_date) - pd.Timestamp(params.start_date)
    ).days / 365.25

    result = StrategyResult(
        strategy_name="DCA",
        ending_wealth=final_wealth,
        total_contributions=total_contributions,
        ending_cash=float(ledger["cash"].iloc[-1]),
        ending_market_value=float(ledger["market_value"].iloc[-1]),
        pnl=final_wealth - total_contributions,
        xirr=xirr(cash_flows),
        cagr=cagr_from_wealth(total_contributions, final_wealth, years),
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        max_drawdown=float(dd_series.min()),
        time_in_market_pct=time_in_market_pct(ledger["units"]),
        n_deployments=n_deployments,
        total_fees=total_fees,
        total_cash_interest=total_interest,
    )
    return result, ledger


def run_wait_for_dip(
    price_data: pd.DataFrame,
    params: SimulationParams,
    cash_rate_series: pd.Series | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """Wait-for-dip strategy: hold contributions as cash until drawdown threshold met.

    Critical lookahead rules:
    - Signal uses PRIOR day's completed close (index t-1).
    - Trade executes at NEXT available close (index t).
    - max_wait_months: Force-deploy if cash held > N months without threshold hit.

    Args:
        price_data: DataFrame with DatetimeIndex and adj_close column.
        params: Simulation parameters.
        cash_rate_series: Daily annualized cash interest rate (optional).

    Returns:
        Tuple of (StrategyResult, ledger DataFrame).
    """
    prices = price_data["adj_close"].copy()
    trading_days = prices.index

    schedule = build_contribution_schedule(
        params.start_date,
        params.end_date,
        params.monthly_contribution,
        params.payday,
        trading_days,
    )

    # Map invest_date → amount
    invest_map: dict[pd.Timestamp, float] = {}
    first_invest_date: dict[pd.Timestamp, pd.Timestamp] = {}  # invest_date → payday
    for _, row in schedule.iterrows():
        invest_date = row["invest_date"]
        invest_map[invest_date] = invest_map.get(invest_date, 0.0) + row["amount"]
        if invest_date not in first_invest_date:
            first_invest_date[invest_date] = row["contribution_date"]

    if cash_rate_series is None:
        cash_rate = params.cash_rate_override or 0.0
        cash_rate_series = pd.Series(cash_rate, index=trading_days)
    else:
        cash_rate_series = cash_rate_series.reindex(trading_days, method="ffill").fillna(0.0)

    # Pre-compute drawdown series on the full price history
    # Signal is prices.iloc[t-1] → act on prices.iloc[t]
    dd_full = compute_drawdown(prices)

    ledger = _build_ledger_template(price_data)
    cash = params.initial_investment
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []
    cash_accumulation_date: pd.Timestamp | None = None  # when cash first started accumulating
    # Pending spread deployments: list of (target_date, amount) sorted by date
    pending_spreads: list[tuple[pd.Timestamp, float]] = []

    if params.initial_investment > 0:
        p = prices.iloc[0]
        net, cost = _apply_cost(cash, params.fixed_fee, params.pct_fee, params.slippage)
        units = net / p if p > 0 else 0.0
        cash = 0.0
        total_fees += cost
        n_deployments += 1
        cash_flows.append((trading_days[0].date(), -params.initial_investment))

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        # Accrue interest on cash
        if prev_dt is not None and cash > 0:
            days_elapsed = (dt - prev_dt).days
            r = cash_rate_series.iloc[i - 1]
            if r > -1 and days_elapsed > 0:
                new_cash = cash * (1 + r) ** (days_elapsed / 365)
                earned = new_cash - cash
                total_interest += earned
                cash = new_cash
                ledger.at[dt, "interest"] = earned

        # Receive contribution
        if dt in invest_map:
            contrib = invest_map[dt]
            if cash == 0:
                cash_accumulation_date = dt
            cash += contrib
            cash_flows.append((dt.date(), -contrib))

        # Check deploy signal: use PRIOR day's drawdown (t-1 signal → t execution)
        p = prices.iloc[i]

        # Process any pending spread deployments due on or before this day
        if pending_spreads and p > 0:
            remaining_spreads: list[tuple[pd.Timestamp, float]] = []
            spread_deployed_today = 0.0
            for target_dt, chunk in pending_spreads:
                if target_dt <= dt:
                    actual_chunk = min(chunk, cash)
                    if actual_chunk > 0:
                        net, cost = _apply_cost(actual_chunk, params.fixed_fee, params.pct_fee, params.slippage)
                        units += net / p
                        total_fees += cost
                        n_deployments += 1
                        cash -= actual_chunk
                        spread_deployed_today += actual_chunk
                        ledger.at[dt, "fees"] = cost
                else:
                    remaining_spreads.append((target_dt, chunk))
            pending_spreads = remaining_spreads
            if spread_deployed_today > 0:
                existing_deployed = float(ledger.at[dt, "deployed"]) if ledger.at[dt, "deployed"] != 0 else 0.0
                ledger.at[dt, "deployed"] = existing_deployed + spread_deployed_today

        if cash > 0 and i > 0:
            signal_dd = dd_full.iloc[i - 1]  # No lookahead: yesterday's drawdown
            should_deploy = signal_dd <= params.dip_threshold

            # Force deploy if waited too long
            if not should_deploy and cash_accumulation_date is not None:
                months_waited = (dt - cash_accumulation_date).days / 30.44
                if months_waited >= params.max_wait_months:
                    should_deploy = True

            if should_deploy and p > 0:
                buffer = params.cash_buffer_months * params.monthly_contribution
                deployable = max(0.0, cash - buffer)
                amount_to_deploy = deployable * params.deployment_pct

                if amount_to_deploy > 0:
                    if params.deploy_spread_months <= 1:
                        # Lump sum: deploy immediately
                        net, cost = _apply_cost(
                            amount_to_deploy, params.fixed_fee, params.pct_fee, params.slippage
                        )
                        units += net / p
                        total_fees += cost
                        n_deployments += 1
                        ledger.at[dt, "deployed"] = amount_to_deploy
                        ledger.at[dt, "fees"] = cost
                        cash -= amount_to_deploy
                    else:
                        # Spread deployment over N months
                        chunk = amount_to_deploy / params.deploy_spread_months
                        for m in range(params.deploy_spread_months):
                            target = pd.Timestamp(dt) + pd.DateOffset(months=m)
                            pending_spreads.append((pd.Timestamp(target), chunk))
                        pending_spreads.sort(key=lambda x: x[0])

                cash_accumulation_date = None  # Reset; remaining cash starts a new accumulation

        market_val = units * p
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val
        prev_dt = dt

    final_wealth = float(ledger["total_wealth"].iloc[-1])
    if cash_flows and final_wealth > 0:
        cash_flows.append((trading_days[-1].date(), final_wealth))

    total_contributions = schedule["amount"].sum() + params.initial_investment
    dd_series = compute_drawdown(ledger["total_wealth"])
    ledger["dd"] = dd_series

    returns = portfolio_returns(ledger["total_wealth"])
    years = (
        pd.Timestamp(params.end_date) - pd.Timestamp(params.start_date)
    ).days / 365.25

    result = StrategyResult(
        strategy_name="Wait-for-Dip",
        ending_wealth=final_wealth,
        total_contributions=total_contributions,
        ending_cash=float(ledger["cash"].iloc[-1]),
        ending_market_value=float(ledger["market_value"].iloc[-1]),
        pnl=final_wealth - total_contributions,
        xirr=xirr(cash_flows),
        cagr=cagr_from_wealth(total_contributions, final_wealth, years),
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        max_drawdown=float(dd_series.min()),
        time_in_market_pct=time_in_market_pct(ledger["units"]),
        n_deployments=n_deployments,
        total_fees=total_fees,
        total_cash_interest=total_interest,
    )
    return result, ledger


def run_tiered_dip(
    price_data: pd.DataFrame,
    params: SimulationParams,
    tiers: list[dict],
    cash_rate_series: pd.Series | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """Tiered dip strategy: deploy different fractions at different drawdown levels.

    tiers example:
        [
            {"threshold": -0.05, "fraction": 0.25},
            {"threshold": -0.10, "fraction": 0.50},
            {"threshold": -0.20, "fraction": 1.00},
        ]
    Tiers reset only after a new all-time high.

    Args:
        price_data: DataFrame with adj_close.
        params: Simulation parameters.
        tiers: List of dicts with 'threshold' and 'fraction' keys.
        cash_rate_series: Daily cash rate series (optional).

    Returns:
        Tuple of (StrategyResult, ledger DataFrame).
    """
    prices = price_data["adj_close"].copy()
    trading_days = prices.index

    # Sort tiers from shallowest to deepest
    tiers_sorted = sorted(tiers, key=lambda t: t["threshold"], reverse=True)

    schedule = build_contribution_schedule(
        params.start_date,
        params.end_date,
        params.monthly_contribution,
        params.payday,
        trading_days,
    )

    invest_map: dict[pd.Timestamp, float] = {}
    for _, row in schedule.iterrows():
        invest_date = row["invest_date"]
        invest_map[invest_date] = invest_map.get(invest_date, 0.0) + row["amount"]

    if cash_rate_series is None:
        cash_rate = params.cash_rate_override or 0.0
        cash_rate_series = pd.Series(cash_rate, index=trading_days)
    else:
        cash_rate_series = cash_rate_series.reindex(trading_days, method="ffill").fillna(0.0)

    dd_full = compute_drawdown(prices)
    peak = prices.expanding().max()

    ledger = _build_ledger_template(price_data)
    cash = params.initial_investment
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []
    tiers_triggered: set[int] = set()  # indices of tiers already triggered in this episode

    if params.initial_investment > 0:
        p = prices.iloc[0]
        net, cost = _apply_cost(cash, params.fixed_fee, params.pct_fee, params.slippage)
        units = net / p if p > 0 else 0.0
        cash = 0.0
        total_fees += cost
        n_deployments += 1
        cash_flows.append((trading_days[0].date(), -params.initial_investment))

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        if prev_dt is not None and cash > 0:
            days_elapsed = (dt - prev_dt).days
            r = cash_rate_series.iloc[i - 1]
            if r > -1 and days_elapsed > 0:
                new_cash = cash * (1 + r) ** (days_elapsed / 365)
                earned = new_cash - cash
                total_interest += earned
                cash = new_cash
                ledger.at[dt, "interest"] = earned

        if dt in invest_map:
            contrib = invest_map[dt]
            cash += contrib
            cash_flows.append((dt.date(), -contrib))

        p = prices.iloc[i]
        if i > 0:
            signal_dd = dd_full.iloc[i - 1]

            # Reset tiers after new all-time high
            if i > 0 and prices.iloc[i - 1] >= peak.iloc[i - 1]:
                tiers_triggered = set()

            # Check each tier
            if cash > 0:
                deployed_this_step = 0.0
                for j, tier in enumerate(tiers_sorted):
                    if j in tiers_triggered:
                        continue
                    if signal_dd <= tier["threshold"]:
                        fraction = tier["fraction"]
                        deploy_amount = min(cash, cash * fraction) if j == len(tiers_sorted) - 1 else cash * fraction
                        if deploy_amount > 0 and p > 0:
                            net, cost = _apply_cost(
                                deploy_amount, params.fixed_fee, params.pct_fee, params.slippage
                            )
                            units += net / p
                            total_fees += cost
                            n_deployments += 1
                            cash -= deploy_amount
                            deployed_this_step += deploy_amount
                            tiers_triggered.add(j)

                if deployed_this_step > 0:
                    ledger.at[dt, "deployed"] = deployed_this_step
                    ledger.at[dt, "fees"] = total_fees

        market_val = units * p
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val
        prev_dt = dt

    final_wealth = float(ledger["total_wealth"].iloc[-1])
    if cash_flows and final_wealth > 0:
        cash_flows.append((trading_days[-1].date(), final_wealth))

    total_contributions = schedule["amount"].sum() + params.initial_investment
    dd_series = compute_drawdown(ledger["total_wealth"])
    ledger["dd"] = dd_series

    returns = portfolio_returns(ledger["total_wealth"])
    years = (
        pd.Timestamp(params.end_date) - pd.Timestamp(params.start_date)
    ).days / 365.25

    result = StrategyResult(
        strategy_name="Tiered-Dip",
        ending_wealth=final_wealth,
        total_contributions=total_contributions,
        ending_cash=float(ledger["cash"].iloc[-1]),
        ending_market_value=float(ledger["market_value"].iloc[-1]),
        pnl=final_wealth - total_contributions,
        xirr=xirr(cash_flows),
        cagr=cagr_from_wealth(total_contributions, final_wealth, years),
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        max_drawdown=float(dd_series.min()),
        time_in_market_pct=time_in_market_pct(ledger["units"]),
        n_deployments=n_deployments,
        total_fees=total_fees,
        total_cash_interest=total_interest,
    )
    return result, ledger
