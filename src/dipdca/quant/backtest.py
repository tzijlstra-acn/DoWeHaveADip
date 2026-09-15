"""Core backtest engine: DCA, wait-for-dip, and tiered-dip strategies.

Event ordering (daily close sequence)
--------------------------------------
1. Accrue cash interest since the previous timestamp.
2. Receive scheduled external contributions (positive external_flow).
3. Execute DCA purchases (for DCA) or check dip signal (for wait/tiered).
   - Dip signal uses PRIOR day's close (i-1); execution at current close (i).
   - Threshold triggers once per drawdown episode (re-arms after recovery).
4. Value the portfolio at the current close.
5. Update the running peak / drawdown from the completed close.
   (Next iteration uses this as the signal for the following day.)

Capital model
-------------
- initial_investment : deployed at the first legal execution price in ALL strategies.
- initial_cash_reserve: deployed immediately in DCA; held as cash in dip strategies
  until the dip threshold is crossed or max_wait_months expires.
- monthly_contribution : recurring external flows; same for all compared strategies.

Episode state machine (wait-for-dip)
-------------------------------------
- episode_armed = True  : strategy is ready to trigger on a threshold crossing.
- On crossing (signal_dd <= threshold, armed): deploy, disarm (episode_armed = False).
- On recovery (signal_dd > threshold while disarmed): re-arm (episode_armed = True).
- Force-deploy (max_wait_months): fires regardless of episode state; re-arms after.

Tiered deployment episode cash basis
--------------------------------------
- When the first tier of an episode triggers, record episode_cash_basis = cash.
- Each tier's notional is fraction * episode_cash_basis (capped at available cash).
- This ensures fractions sum to 100% of the episode opening balance, not ~70%.
- Tiers reset (and episode_cash_basis clears) after a new all-time high.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from dipdca.models import DeploymentTier, MarketDefinition, SimulationParams, StrategyResult
from dipdca.quant.contributions import build_contribution_schedule
from dipdca.quant.drawdown import drawdown as compute_drawdown
from dipdca.quant.metrics import (
    build_nav,
    cagr_from_wealth,
    flow_adjusted_returns,
    nav_max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    time_in_market_pct,
    twr_cagr,
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


def _ledger_float(ledger: pd.DataFrame, idx: object, col: str) -> float:
    """Read a scalar from ledger.at[] as float — isolates pandas-stubs arg-type suppression."""
    return float(ledger.at[idx, col])  # type: ignore[arg-type]


def _build_ledger_template(price_data: pd.DataFrame) -> pd.DataFrame:
    """Create an empty ledger DataFrame aligned to trading days.

    Columns
    -------
    price          : asset adjusted-close price.
    external_flow  : net external capital flow (positive = contribution in).
    cash           : uninvested cash balance after all events.
    units          : asset units held.
    market_value   : units * price.
    total_wealth   : cash + market_value (invariant: no internal move changes this
                     except fees; only external_flow and price changes do).
    deployed       : notional amount moved from cash to asset this day.
    fees           : transaction costs incurred this day (marginal, not cumulative).
    interest       : cash interest earned this day.
    signal_dd      : drawdown signal used for decisions (prior day's drawdown).
    dd             : running drawdown of total_wealth (populated at end of run).
    """
    idx = price_data.index
    return pd.DataFrame(
        {
            "price": price_data["adj_close"],
            "external_flow": 0.0,
            "cash": 0.0,
            "units": 0.0,
            "market_value": 0.0,
            "total_wealth": 0.0,
            "deployed": 0.0,
            "fees": 0.0,
            "interest": 0.0,
            "signal_dd": 0.0,
            "dd": 0.0,
            # Benchmark audit columns (populated by run_dip_deployment when benchmark_data provided)
            "benchmark_close": float("nan"),
            "benchmark_ath": float("nan"),
            "benchmark_drawdown": float("nan"),
        },
        index=idx,
    )


def _build_strategy_result(
    strategy_name: str,
    ledger: pd.DataFrame,
    cash_flows: list[tuple[date, float]],
    total_contributions: float,
    total_fees: float,
    total_interest: float,
    n_deployments: int,
    years: float,
) -> StrategyResult:
    """Compute all metrics and assemble a StrategyResult from a completed ledger."""
    final_wealth = float(ledger["total_wealth"].iloc[-1])

    # XIRR: add terminal liquidation receipt
    xirr_flows = list(cash_flows)
    if xirr_flows and final_wealth > 0:
        last_date = ledger.index[-1].date()
        xirr_flows.append((last_date, final_wealth))

    # Flow-adjusted returns (deposit-corrected)
    ext_flows = ledger["external_flow"]
    fa_returns = flow_adjusted_returns(ledger["total_wealth"], ext_flows)
    nav = build_nav(fa_returns) if len(fa_returns) > 0 else pd.Series([], dtype=float)

    # Drawdown from raw wealth (legacy, deposit-contaminated)
    dd_series = compute_drawdown(ledger["total_wealth"])
    ledger["dd"] = dd_series

    return StrategyResult(
        strategy_name=strategy_name,
        ending_wealth=final_wealth,
        total_contributions=total_contributions,
        ending_cash=float(ledger["cash"].iloc[-1]),
        ending_market_value=float(ledger["market_value"].iloc[-1]),
        pnl=final_wealth - total_contributions,
        xirr=xirr(xirr_flows),
        cagr=cagr_from_wealth(total_contributions, final_wealth, years),
        twr=twr_cagr(fa_returns, years),
        sharpe=sharpe_ratio(fa_returns),
        sortino=sortino_ratio(fa_returns),
        max_drawdown=float(dd_series.min()),
        nav_mdd=nav_max_drawdown(nav) if len(nav) > 0 else None,
        time_in_market_pct=time_in_market_pct(ledger["units"]),
        n_deployments=n_deployments,
        total_fees=total_fees,
        total_cash_interest=total_interest,
    )


def run_savings_only(
    price_data: pd.DataFrame,
    params: SimulationParams,
) -> tuple[StrategyResult, pd.DataFrame]:
    """Savings-account-only strategy: all contributions stay in cash earning savings rate.

    No asset purchases are ever made. This is a benchmark showing pure cash accumulation.
    The same contribution schedule as DCA and dip strategies is used.
    """
    ledger = _build_ledger_template(price_data)
    trading_days = list(ledger.index)
    if not trading_days:
        raise ValueError("No trading days in price data")

    schedule = build_contribution_schedule(
        params.start_date,
        params.end_date,
        params.monthly_contribution,
        params.payday,
        pd.DatetimeIndex(trading_days),
    )
    invest_map: dict[pd.Timestamp, float] = {}
    for _, row in schedule.iterrows():
        invest_date = row["invest_date"]
        invest_map[invest_date] = invest_map.get(invest_date, 0.0) + row["amount"]

    cash_rate = params.cash_rate_override if params.cash_rate_override is not None else 0.0

    # All capital (including initial_investment) stays in cash
    cash = params.initial_cash_reserve + params.initial_investment
    total_interest = 0.0
    cash_flows: list[tuple[date, float]] = []

    # Record day-1 external flows
    if cash > 0:
        ledger.at[trading_days[0], "external_flow"] = cash
        cash_flows.append((trading_days[0].date(), -cash))

    prev_dt: pd.Timestamp | None = None
    for _i, dt in enumerate(trading_days):
        # 1. Accrue interest
        if prev_dt is not None and cash > 0 and cash_rate > 0:
            days_elapsed = (dt - prev_dt).days
            interest = cash * ((1.0 + cash_rate) ** (days_elapsed / 365.25) - 1.0)
            cash += interest
            total_interest += interest
            ledger.at[dt, "interest"] = interest

        # 2. Receive contributions
        contrib = invest_map.get(dt, 0.0)
        if contrib > 0:
            cash += contrib
            ledger.at[dt, "external_flow"] = _ledger_float(ledger, dt, "external_flow") + contrib
            cash_flows.append((dt.date(), -contrib))

        # 3. No purchases — record ledger row
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = 0.0
        ledger.at[dt, "market_value"] = 0.0
        ledger.at[dt, "total_wealth"] = cash
        ledger.at[dt, "deployed"] = 0.0
        ledger.at[dt, "fees"] = 0.0

        prev_dt = dt

    total_contributions = schedule["amount"].sum() + params.initial_investment + params.initial_cash_reserve
    years = (trading_days[-1] - trading_days[0]).days / 365.25

    result = _build_strategy_result(
        strategy_name="Savings account",
        ledger=ledger,
        cash_flows=cash_flows,
        total_contributions=total_contributions,
        total_fees=0.0,
        total_interest=total_interest,
        n_deployments=0,
        years=years,
    )
    return result, ledger


def run_dip_deployment(
    price_data: pd.DataFrame,
    params: SimulationParams,
    tiers: list[DeploymentTier],
    benchmark_data: pd.DataFrame | None = None,
    market_def: MarketDefinition | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """Dip-deployment strategy with cumulative deployment targets.

    .. deprecated::
        Use :func:`run_ath_deployment` for new code.  This function is kept for
        backward compatibility with existing tests.  Known limitations:
        - Bug 1: When ``benchmark_data`` is ``None``, silently falls back to ETF
          prices for ATH / drawdown computation.
        - Bug 2: ``bm_running_high`` is initialised to 0 and seeded from the first
          loop observation instead of from the full benchmark history.
        - Bug 4: When the benchmark gaps across multiple tiers in one close, each
          tier schedules a separate order (over-deploys).
        :func:`run_ath_deployment` fixes all three issues.

    Monthly savings accumulate in a savings account (earning cash_rate) until
    a configured drawdown threshold is crossed. At each threshold, a configured
    CUMULATIVE fraction of the eligible saved capital is deployed.

    Signal source (ATH and drawdown):
        When benchmark_data is provided, the drawdown signal is computed from the
        benchmark index (e.g. Nasdaq-100 Index for a QQQ investment). This is the
        correct model — the index decides WHEN to buy. When benchmark_data is None,
        falls back to price_data (backward compatible).

    Execution source:
        Always uses price_data["adj_close"] — the investable instrument price.

    eligible_capital = current_cash + principal_deployed_in_current_episode

    incremental_trade = min(
        cash,
        max(0, tier.cumulative_deployment_fraction * eligible_capital
               - principal_deployed_in_episode)
    )

    Thresholds trigger ONCE per drawdown episode (defined as ending at a new ATH).
    Signal is detected at close t; execution happens at close t+1 (no look-ahead bias).

    Args:
        price_data: Investable instrument prices (ETF). Used for execution only.
        params: Simulation parameters.
        tiers: Cumulative deployment tier schedule.
        benchmark_data: Optional reference index data. When provided, its adj_close
            column is used for ATH and drawdown signal computation. Must be at least
            as long as price_data; aligned to instrument calendar via forward-fill.
        market_def: Optional metadata about the benchmark/instrument pairing.
            Used only for audit ledger columns; does not change signal logic.
    """
    if not tiers:
        raise ValueError("At least one DeploymentTier is required")
    DeploymentTier.validate_schedule(tiers)

    ledger = _build_ledger_template(price_data)
    trading_days = list(ledger.index)
    if not trading_days:
        raise ValueError("No trading days in price data")

    # Align benchmark series to instrument trading calendar.
    # When benchmark_data is provided, forward-fill its closes to the instrument
    # dates so every instrument date has a benchmark observation. This preserves
    # the "signal at t, execute at t+1" guarantee even when the benchmark has
    # different missing dates (e.g. US holiday vs European holiday).
    if benchmark_data is not None and "adj_close" in benchmark_data.columns:
        bm_aligned: pd.Series = (
            benchmark_data["adj_close"]
            .reindex(ledger.index, method="ffill")
            .ffill()
        )
    else:
        bm_aligned = price_data["adj_close"]

    schedule = build_contribution_schedule(
        params.start_date,
        params.end_date,
        params.monthly_contribution,
        params.payday,
        pd.DatetimeIndex(trading_days),
    )
    invest_map: dict[pd.Timestamp, float] = {}
    for _, row in schedule.iterrows():
        invest_date = row["invest_date"]
        invest_map[invest_date] = invest_map.get(invest_date, 0.0) + row["amount"]

    cash_rate = params.cash_rate_override if params.cash_rate_override is not None else 0.0

    cash = params.initial_cash_reserve
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []

    # Deploy initial_investment on day 1 (same as DCA)
    if params.initial_investment > 0 and trading_days:
        p0 = float(price_data["adj_close"].iloc[0])
        if p0 > 0:
            net, cost = _apply_cost(params.initial_investment, params.fixed_fee, params.pct_fee, params.slippage)
            units = net / p0
            total_fees += cost
            n_deployments += 1
        ledger.at[trading_days[0], "external_flow"] = (
            _ledger_float(ledger, trading_days[0], "external_flow") + params.initial_investment
        )
        ledger.at[trading_days[0], "deployed"] = params.initial_investment
        cash_flows.append((trading_days[0].date(), -params.initial_investment))

    if params.initial_cash_reserve > 0:
        ledger.at[trading_days[0], "external_flow"] = (
            _ledger_float(ledger, trading_days[0], "external_flow") + params.initial_cash_reserve
        )
        cash_flows.append((trading_days[0].date(), -params.initial_cash_reserve))

    # Episode state
    triggered_tiers: set[int] = set()
    principal_deployed_in_episode = 0.0
    prev_drawdown = 0.0
    bm_running_high = 0.0  # benchmark ATH — drives signal; NOT instrument price
    # Pending: (execute_on_or_after_dt, tier_index, amount)
    pending_executions: list[tuple[pd.Timestamp, int, float]] = []

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        # Instrument price (execution only)
        p = _ledger_float(ledger, dt, "price")
        # Benchmark close (signal only)
        bm_close = float(bm_aligned.iloc[i])

        # 1. Accrue interest on cash
        if prev_dt is not None and cash > 0 and cash_rate > 0:
            days_elapsed = (dt - prev_dt).days
            interest = cash * ((1.0 + cash_rate) ** (days_elapsed / 365.25) - 1.0)
            cash += interest
            total_interest += interest
            ledger.at[dt, "interest"] = interest

        # 2. Receive contributions
        contrib = invest_map.get(dt, 0.0)
        if contrib > 0:
            cash += contrib
            ledger.at[dt, "external_flow"] = _ledger_float(ledger, dt, "external_flow") + contrib
            cash_flows.append((dt.date(), -contrib))

        # 3. Execute pending trades (signalled at t-1, executed at t using instrument price)
        deployed_today = 0.0
        fees_today = 0.0
        for exec_dt, tier_idx, amount in list(pending_executions):
            if dt >= exec_dt and p > 0:
                actual = min(amount, cash)
                if actual > 0:
                    net, cost = _apply_cost(actual, params.fixed_fee, params.pct_fee, params.slippage)
                    units += net / p
                    cash -= actual
                    total_fees += cost
                    n_deployments += 1
                    deployed_today += actual
                    fees_today += cost
                    principal_deployed_in_episode += actual
                pending_executions.remove((exec_dt, tier_idx, amount))

        if deployed_today > 0:
            ledger.at[dt, "deployed"] = _ledger_float(ledger, dt, "deployed") + deployed_today
            ledger.at[dt, "fees"] = _ledger_float(ledger, dt, "fees") + fees_today

        # 4. Value portfolio using instrument price
        market_val = units * p
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val

        # 5. Update benchmark ATH and drawdown (benchmark series, not instrument)
        if bm_close >= bm_running_high or bm_running_high == 0.0:
            bm_running_high = bm_close
            if triggered_tiers:  # new benchmark ATH after episode — rearm
                triggered_tiers = set()
                principal_deployed_in_episode = 0.0
            current_dd = 0.0
        else:
            current_dd = bm_close / bm_running_high - 1.0

        # Write benchmark audit columns
        ledger.at[dt, "benchmark_close"] = bm_close
        ledger.at[dt, "benchmark_ath"] = bm_running_high
        ledger.at[dt, "benchmark_drawdown"] = current_dd

        # 6. Detect threshold crossings using benchmark drawdown
        #    (signal at close t, execute at close t+1 using instrument price)
        if i > 0:
            for j, tier in enumerate(tiers):
                if j in triggered_tiers:
                    continue
                # Crossing: yesterday benchmark above threshold, today at or below
                crossed = prev_drawdown > tier.drawdown_threshold >= current_dd
                if crossed:
                    eligible = cash + principal_deployed_in_episode
                    target_total = tier.cumulative_deployment_fraction * eligible
                    incremental = min(cash, max(0.0, target_total - principal_deployed_in_episode))
                    if incremental > 0 and i + 1 < len(trading_days):
                        pending_executions.append((trading_days[i + 1], j, incremental))
                    triggered_tiers.add(j)
                    ledger.at[dt, "signal_dd"] = current_dd

        prev_drawdown = current_dd
        prev_dt = dt

    total_contributions = schedule["amount"].sum() + params.initial_investment + params.initial_cash_reserve
    years = (trading_days[-1] - trading_days[0]).days / 365.25

    result = _build_strategy_result(
        strategy_name="Dip deployment",
        ledger=ledger,
        cash_flows=cash_flows,
        total_contributions=total_contributions,
        total_fees=total_fees,
        total_interest=total_interest,
        n_deployments=n_deployments,
        years=years,
    )
    return result, ledger


def run_ath_deployment(
    instrument_data: pd.DataFrame,
    params: SimulationParams,
    tiers: list[DeploymentTier],
    benchmark_data: pd.DataFrame | None,
    initial_ath: float | None = None,
    market_def: MarketDefinition | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """ATH-based dip deployment strategy.

    Invariants vs run_dip_deployment
    ---------------------------------
    1. ``benchmark_data`` is **required** — raises ``ValueError`` if ``None``.
    2. ATH is seeded BEFORE the loop from ``benchmark_data["adj_close"].max()``
       (or from the explicit ``initial_ath`` argument), so a simulation that starts
       mid-bear-market computes drawdowns from the true historical peak.
    3. Episode resets **only** when the benchmark closes at or above the previous ATH.
       Recovering above a threshold does NOT rearm a tier.
    4. When the benchmark gaps across multiple tiers in one close, the engine finds
       ALL newly-crossed tiers, picks the **deepest** cumulative target, and schedules
       **one** incremental order — preventing the over-deploy bug.
    5. There is **no** ``max_wait_months`` force-deployment. Cash is deployed only on
       threshold crossings.

    Signal source
    -------------
    ``benchmark_data["adj_close"]`` drives ATH, drawdown, and threshold crossings.
    ``instrument_data["adj_close"]`` is used only for execution price and valuation.

    ATH seeding
    -----------
    ``initial_ath = benchmark_data["adj_close"].max()`` by default.
    Pass ``initial_ath`` explicitly when you have a pre-simulation historical ATH
    that is NOT present in the benchmark data slice you are passing.

    Args:
        instrument_data: ETF / investable instrument prices (execution + valuation).
        params: Simulation parameters.
        tiers: Cumulative deployment tier schedule (validated ascending fractions,
               descending thresholds).
        benchmark_data: Required reference index. Must have an ``adj_close`` column.
            Raises ``ValueError`` when ``None``.
        initial_ath: Optional explicit ATH seed. If ``None``, computed as
            ``float(benchmark_data["adj_close"].max())``.
        market_def: Optional metadata (not used in signal logic; for audit only).

    Returns:
        Tuple of (StrategyResult, day-by-day ledger DataFrame).
    """
    if benchmark_data is None:
        raise ValueError(
            "benchmark_data is required for run_ath_deployment. "
            "Pass the benchmark index DataFrame (e.g. ^NDX for QQQ). "
            "If the asset has no configured index_symbol, use run_dip_deployment instead."
        )
    if "adj_close" not in benchmark_data.columns:
        raise ValueError("benchmark_data must have an 'adj_close' column")
    if len(benchmark_data) == 0:
        raise ValueError("benchmark_data must not be empty")
    if not tiers:
        raise ValueError("At least one DeploymentTier is required")
    DeploymentTier.validate_schedule(tiers)

    ledger = _build_ledger_template(instrument_data)
    trading_days = list(ledger.index)
    if not trading_days:
        raise ValueError("No trading days in instrument_data")

    # Align benchmark to instrument trading calendar (forward-fill across holidays /
    # different missing-date conventions).
    bm_aligned: pd.Series = (
        benchmark_data["adj_close"]
        .reindex(ledger.index, method="ffill")
        .ffill()
    )

    # -----------------------------------------------------------------------
    # Seed ATH BEFORE the loop — Bug 2 fix.
    # We use the maximum of the ENTIRE benchmark series passed in (which the
    # caller should include pre-simulation history).  The in-loop logic then
    # updates bm_running_high whenever the benchmark closes at a new high.
    # -----------------------------------------------------------------------
    if initial_ath is not None:
        bm_running_high = float(initial_ath)
    else:
        bm_running_high = float(benchmark_data["adj_close"].max())

    schedule = build_contribution_schedule(
        params.start_date,
        params.end_date,
        params.monthly_contribution,
        params.payday,
        pd.DatetimeIndex(trading_days),
    )
    invest_map: dict[pd.Timestamp, float] = {}
    for _, row in schedule.iterrows():
        invest_date = row["invest_date"]
        invest_map[invest_date] = invest_map.get(invest_date, 0.0) + row["amount"]

    cash_rate = params.cash_rate_override if params.cash_rate_override is not None else 0.0

    cash = params.initial_cash_reserve
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []

    # Deploy initial_investment on day 1 (identical to DCA / run_dip_deployment)
    if params.initial_investment > 0 and trading_days:
        p0 = float(instrument_data["adj_close"].iloc[0])
        if p0 > 0:
            net, cost = _apply_cost(
                params.initial_investment, params.fixed_fee, params.pct_fee, params.slippage
            )
            units = net / p0
            total_fees += cost
            n_deployments += 1
        ledger.at[trading_days[0], "external_flow"] = (
            _ledger_float(ledger, trading_days[0], "external_flow") + params.initial_investment
        )
        ledger.at[trading_days[0], "deployed"] = params.initial_investment
        cash_flows.append((trading_days[0].date(), -params.initial_investment))

    if params.initial_cash_reserve > 0:
        ledger.at[trading_days[0], "external_flow"] = (
            _ledger_float(ledger, trading_days[0], "external_flow") + params.initial_cash_reserve
        )
        cash_flows.append((trading_days[0].date(), -params.initial_cash_reserve))

    # Episode state
    triggered_tiers: set[int] = set()
    principal_deployed_in_episode = 0.0
    prev_drawdown = 0.0
    # Pending: (execute_on_or_after_dt, tier_index, amount)
    pending_executions: list[tuple[pd.Timestamp, int, float]] = []

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        # Instrument price (execution and valuation only)
        p = _ledger_float(ledger, dt, "price")
        # Benchmark close (signal only)
        bm_close = float(bm_aligned.iloc[i])

        # 1. Accrue interest on cash
        if prev_dt is not None and cash > 0 and cash_rate > 0:
            days_elapsed = (dt - prev_dt).days
            interest = cash * ((1.0 + cash_rate) ** (days_elapsed / 365.25) - 1.0)
            cash += interest
            total_interest += interest
            ledger.at[dt, "interest"] = interest

        # 2. Receive contributions
        contrib = invest_map.get(dt, 0.0)
        if contrib > 0:
            cash += contrib
            ledger.at[dt, "external_flow"] = (
                _ledger_float(ledger, dt, "external_flow") + contrib
            )
            cash_flows.append((dt.date(), -contrib))

        # 3. Execute pending trades (signalled at t-1, executed at t using instrument price)
        deployed_today = 0.0
        fees_today = 0.0
        for exec_dt, tier_idx, amount in list(pending_executions):
            if dt >= exec_dt and p > 0:
                actual = min(amount, cash)
                if actual > 0:
                    net, cost = _apply_cost(actual, params.fixed_fee, params.pct_fee, params.slippage)
                    units += net / p
                    cash -= actual
                    total_fees += cost
                    n_deployments += 1
                    deployed_today += actual
                    fees_today += cost
                    principal_deployed_in_episode += actual
                pending_executions.remove((exec_dt, tier_idx, amount))

        if deployed_today > 0:
            ledger.at[dt, "deployed"] = (
                _ledger_float(ledger, dt, "deployed") + deployed_today
            )
            ledger.at[dt, "fees"] = (
                _ledger_float(ledger, dt, "fees") + fees_today
            )

        # 4. Value portfolio using instrument price
        market_val = units * p
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val

        # 5. Update benchmark ATH and drawdown.
        #    Episode resets ONLY when benchmark closes at or above the previous ATH.
        #    Recovering above a threshold does NOT rearm a tier (Bug 3 fix).
        if bm_close >= bm_running_high:
            # New ATH: reset episode state
            bm_running_high = bm_close
            if triggered_tiers:
                triggered_tiers = set()
                principal_deployed_in_episode = 0.0
            current_dd = 0.0
        else:
            current_dd = bm_close / bm_running_high - 1.0

        # Write benchmark audit columns
        ledger.at[dt, "benchmark_close"] = bm_close
        ledger.at[dt, "benchmark_ath"] = bm_running_high
        ledger.at[dt, "benchmark_drawdown"] = current_dd

        # 6. Detect threshold crossings using benchmark drawdown.
        #    Multi-tier gap fix (Bug 4): collect ALL newly-crossed tiers, pick the
        #    DEEPEST cumulative target, schedule ONE incremental order.
        #
        #    Unlike run_dip_deployment, we allow i=0 to signal a crossing.  When
        #    initial_ath is seeded from pre-simulation history, prev_drawdown=0.0
        #    (the "before simulation" reference) and current_dd may already be below
        #    a threshold on the very first day.  Detecting that crossing on day 0
        #    schedules execution at day 1 — no look-ahead bias.
        if True:
            crossed_tiers = [
                (j, tier)
                for j, tier in enumerate(tiers)
                if j not in triggered_tiers
                and prev_drawdown > tier.drawdown_threshold >= current_dd
            ]
            if crossed_tiers:
                # Mark ALL crossed tiers as triggered (they are all "consumed")
                for j, _tier in crossed_tiers:
                    triggered_tiers.add(j)

                # Use ONLY the deepest (most negative threshold) cumulative target
                deepest_j, deepest_tier = min(
                    crossed_tiers, key=lambda jt: jt[1].drawdown_threshold
                )
                cumulative_target = deepest_tier.cumulative_deployment_fraction
                eligible = cash + principal_deployed_in_episode
                target_total = cumulative_target * eligible

                # Account for already-pending (not yet executed) orders
                pending_total = sum(amt for _, _, amt in pending_executions)
                incremental = min(
                    max(0.0, cash - pending_total),
                    max(0.0, target_total - principal_deployed_in_episode - pending_total),
                )
                if incremental > 0 and i + 1 < len(trading_days):
                    pending_executions.append((trading_days[i + 1], deepest_j, incremental))
                ledger.at[dt, "signal_dd"] = current_dd

        prev_drawdown = current_dd
        prev_dt = dt

    total_contributions = (
        schedule["amount"].sum() + params.initial_investment + params.initial_cash_reserve
    )
    years = (trading_days[-1] - trading_days[0]).days / 365.25

    result = _build_strategy_result(
        strategy_name="ATH Dip Deployment",
        ledger=ledger,
        cash_flows=cash_flows,
        total_contributions=total_contributions,
        total_fees=total_fees,
        total_interest=total_interest,
        n_deployments=n_deployments,
        years=years,
    )
    return result, ledger


def run_dca(
    price_data: pd.DataFrame,
    params: SimulationParams,
    cash_rate_series: pd.Series | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """DCA strategy: deploy every contribution on the next available trading day.

    Capital model:
    - initial_investment + initial_cash_reserve both deploy on day 1.
    - Each monthly contribution deploys immediately on its invest_date.
    - No drawdown logic — always deploy.

    Args:
        price_data: DataFrame with DatetimeIndex and adj_close column.
        params: Simulation parameters.
        cash_rate_series: Daily annualized cash interest rate (optional).

    Returns:
        Tuple of (StrategyResult, day-by-day ledger DataFrame).
    """
    prices = price_data["adj_close"].copy()
    trading_days: pd.DatetimeIndex = pd.DatetimeIndex(prices.index)

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

    ledger = _build_ledger_template(price_data)
    # Both initial_investment and initial_cash_reserve deploy on day 1 in DCA
    starting_capital = params.initial_investment + params.initial_cash_reserve
    cash = starting_capital
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []

    if starting_capital > 0:
        first_day = trading_days[0]
        p = prices.iloc[0]
        net, cost = _apply_cost(cash, params.fixed_fee, params.pct_fee, params.slippage)
        units = net / p if p > 0 else 0.0
        cash = 0.0
        total_fees += cost
        n_deployments += 1
        ledger.at[first_day, "external_flow"] = starting_capital
        ledger.at[first_day, "deployed"] = starting_capital
        ledger.at[first_day, "fees"] = cost
        if params.initial_investment > 0:
            cash_flows.append((first_day.date(), -params.initial_investment))
        if params.initial_cash_reserve > 0:
            cash_flows.append((first_day.date(), -params.initial_cash_reserve))

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        # 1. Accrue interest on uninvested cash
        if prev_dt is not None and cash > 0:
            days_elapsed = (dt - prev_dt).days
            r = cash_rate_series.iloc[i - 1]
            if r > -1 and days_elapsed > 0:
                new_cash = cash * (1 + r) ** (days_elapsed / 365.25)
                earned = new_cash - cash
                total_interest += earned
                cash = new_cash
                ledger.at[dt, "interest"] = earned

        # 2. Receive contribution and deploy immediately
        if dt in invest_map:
            contrib = invest_map[dt]
            cash += contrib
            cash_flows.append((dt.date(), -contrib))
            ledger.at[dt, "external_flow"] = contrib

            p = prices.iloc[i]
            if p > 0 and cash > 0:
                net, cost = _apply_cost(cash, params.fixed_fee, params.pct_fee, params.slippage)
                units += net / p
                total_fees += cost
                n_deployments += 1
                ledger.at[dt, "deployed"] = cash
                ledger.at[dt, "fees"] = cost
                cash = 0.0

        # 4. Value portfolio
        market_val = units * prices.iloc[i]
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val
        prev_dt = dt

    total_contributions = (
        schedule["amount"].sum() + params.initial_investment + params.initial_cash_reserve
    )
    years = (pd.Timestamp(params.end_date) - pd.Timestamp(params.start_date)).days / 365.25

    result = _build_strategy_result(
        "DCA", ledger, cash_flows, total_contributions, total_fees, total_interest, n_deployments, years
    )
    return result, ledger


def run_wait_for_dip(
    price_data: pd.DataFrame,
    params: SimulationParams,
    cash_rate_series: pd.Series | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """Wait-for-dip strategy: hold contributions as cash until drawdown threshold met.

    .. deprecated::
        Use :func:`run_ath_deployment` for new UI code.  Kept for backward
        compatibility with existing tests.  Known limitation (Bug 3): the episode
        re-arms when drawdown recovers above the threshold (``signal_dd > threshold``).
        The correct rule is that only a new benchmark ATH resets the episode.

    Episode state machine:
    - episode_armed=True: strategy will trigger on the next threshold crossing.
    - On crossing: deploy, disarm (episode_armed=False).
    - On recovery above threshold: re-arm (episode_armed=True).
    - max_wait_months force-deploy: fires regardless of episode_armed state.

    Capital model:
    - initial_investment deploys on day 1 (same as DCA).
    - initial_cash_reserve is held in cash, subject to dip strategy rules.

    Lookahead rules:
    - Signal uses prior close drawdown (index i-1).
    - Execution at current close (index i).

    Args:
        price_data: DataFrame with DatetimeIndex and adj_close column.
        params: Simulation parameters.
        cash_rate_series: Daily annualized cash interest rate (optional).

    Returns:
        Tuple of (StrategyResult, ledger DataFrame).
    """
    prices = price_data["adj_close"].copy()
    trading_days: pd.DatetimeIndex = pd.DatetimeIndex(prices.index)

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

    # Pre-compute drawdown on full price history: signal at i-1, execute at i
    dd_full = compute_drawdown(prices)

    ledger = _build_ledger_template(price_data)
    # initial_cash_reserve stays in cash waiting for the dip signal
    cash = params.initial_cash_reserve
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []
    cash_accumulation_date: pd.Timestamp | None = None
    pending_spreads: list[tuple[pd.Timestamp, float]] = []

    # Episode state machine
    episode_armed = True  # ready to trigger; disarmed after first trigger in episode

    # initial_cash_reserve: external inflow on day 1
    if params.initial_cash_reserve > 0:
        cash_flows.append((trading_days[0].date(), -params.initial_cash_reserve))
        ledger.at[trading_days[0], "external_flow"] = params.initial_cash_reserve
        cash_accumulation_date = trading_days[0]

    # initial_investment: deploy on day 1 (identical to DCA)
    if params.initial_investment > 0:
        p = prices.iloc[0]
        net, cost = _apply_cost(params.initial_investment, params.fixed_fee, params.pct_fee, params.slippage)
        units = net / p if p > 0 else 0.0
        total_fees += cost
        n_deployments += 1
        cash_flows.append((trading_days[0].date(), -params.initial_investment))
        ledger.at[trading_days[0], "external_flow"] = (
            _ledger_float(ledger, trading_days[0], "external_flow") + params.initial_investment
        )
        ledger.at[trading_days[0], "deployed"] = params.initial_investment
        ledger.at[trading_days[0], "fees"] = cost

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        # 1. Accrue interest on cash
        if prev_dt is not None and cash > 0:
            days_elapsed = (dt - prev_dt).days
            r = cash_rate_series.iloc[i - 1]
            if r > -1 and days_elapsed > 0:
                new_cash = cash * (1 + r) ** (days_elapsed / 365.25)
                earned = new_cash - cash
                total_interest += earned
                cash = new_cash
                ledger.at[dt, "interest"] = earned

        # 2. Receive contribution (held in cash)
        if dt in invest_map:
            contrib = invest_map[dt]
            if cash == 0:
                cash_accumulation_date = dt
            cash += contrib
            cash_flows.append((dt.date(), -contrib))
            ledger.at[dt, "external_flow"] = contrib

        # 3. Execute pending spread deployments
        p = prices.iloc[i]
        if pending_spreads and p > 0:
            remaining_spreads: list[tuple[pd.Timestamp, float]] = []
            spread_deployed_today = 0.0
            fees_spread_today = 0.0
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
                        fees_spread_today += cost
                else:
                    remaining_spreads.append((target_dt, chunk))
            pending_spreads = remaining_spreads
            if spread_deployed_today > 0:
                ledger.at[dt, "deployed"] = _ledger_float(ledger, dt, "deployed") + spread_deployed_today
                ledger.at[dt, "fees"] = _ledger_float(ledger, dt, "fees") + fees_spread_today

        # 3b. Check dip signal (t-1 signal → t execution)
        if cash > 0 and i > 0:
            signal_dd = float(dd_full.iloc[i - 1])
            ledger.at[dt, "signal_dd"] = signal_dd

            # Re-arm when market recovers above threshold
            if not episode_armed and signal_dd > params.dip_threshold:
                episode_armed = True

            # Threshold trigger: only when armed
            should_deploy = signal_dd <= params.dip_threshold and episode_armed

            # Force-deploy: when cash has waited too long (overrides episode state)
            force_deploy = False
            if not should_deploy and cash_accumulation_date is not None:
                months_waited = (dt - cash_accumulation_date).days / 30.44
                if months_waited >= params.max_wait_months:
                    force_deploy = True

            if (should_deploy or force_deploy) and p > 0:
                buffer = params.cash_buffer_months * params.monthly_contribution
                deployable = max(0.0, cash - buffer)
                amount_to_deploy = deployable * params.deployment_pct

                if amount_to_deploy > 0:
                    if params.deploy_spread_months <= 1:
                        net, cost = _apply_cost(
                            amount_to_deploy, params.fixed_fee, params.pct_fee, params.slippage
                        )
                        units += net / p
                        total_fees += cost
                        n_deployments += 1
                        ledger.at[dt, "deployed"] = _ledger_float(ledger, dt, "deployed") + amount_to_deploy
                        ledger.at[dt, "fees"] = _ledger_float(ledger, dt, "fees") + cost
                        cash -= amount_to_deploy
                    else:
                        chunk = amount_to_deploy / params.deploy_spread_months
                        for m in range(params.deploy_spread_months):
                            target = pd.Timestamp(dt) + pd.DateOffset(months=m)
                            pending_spreads.append((pd.Timestamp(target), chunk))
                        pending_spreads.sort(key=lambda x: x[0])

                if should_deploy:
                    episode_armed = False  # disarm: this episode has triggered
                # Reset accumulation timer; remaining cash age starts fresh
                cash_accumulation_date = None

        # 4. Value portfolio
        market_val = units * p
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val
        prev_dt = dt

    total_contributions = (
        schedule["amount"].sum() + params.initial_investment + params.initial_cash_reserve
    )
    years = (pd.Timestamp(params.end_date) - pd.Timestamp(params.start_date)).days / 365.25

    result = _build_strategy_result(
        "Wait-for-Dip", ledger, cash_flows, total_contributions, total_fees, total_interest, n_deployments, years
    )
    return result, ledger


def run_tiered_dip(
    price_data: pd.DataFrame,
    params: SimulationParams,
    tiers: list[dict],
    cash_rate_series: pd.Series | None = None,
) -> tuple[StrategyResult, pd.DataFrame]:
    """Tiered dip strategy: deploy fixed fractions of the episode cash basis at each tier.

    .. deprecated::
        Use :func:`run_ath_deployment` for new UI code.  Kept for backward
        compatibility with existing tests.  Known limitation: uses the investable
        instrument price (not a benchmark index) for ATH detection.

    Episode cash basis semantics:
    - When the first tier of a drawdown episode triggers, record episode_cash_basis = cash.
    - Each tier's notional = fraction * episode_cash_basis (capped at available cash).
    - Fractions are portions of the OPENING episode balance, so 33%+33%+34% = 100%.
    - Tiers reset (and episode_cash_basis clears) after a new all-time high.

    tiers example:
        [
            {"threshold": -0.05, "fraction": 0.33},
            {"threshold": -0.10, "fraction": 0.33},
            {"threshold": -0.20, "fraction": 0.34},
        ]

    Args:
        price_data: DataFrame with adj_close.
        params: Simulation parameters.
        tiers: List of dicts with 'threshold' (negative) and 'fraction' keys.
        cash_rate_series: Daily cash rate series (optional).

    Returns:
        Tuple of (StrategyResult, ledger DataFrame).
    """
    prices = price_data["adj_close"].copy()
    trading_days: pd.DatetimeIndex = pd.DatetimeIndex(prices.index)

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
    cash = params.initial_cash_reserve
    units = 0.0
    total_fees = 0.0
    total_interest = 0.0
    n_deployments = 0
    cash_flows: list[tuple[date, float]] = []
    tiers_triggered: set[int] = set()
    episode_cash_basis: float | None = None  # set when first tier of episode fires

    if params.initial_cash_reserve > 0:
        cash_flows.append((trading_days[0].date(), -params.initial_cash_reserve))
        ledger.at[trading_days[0], "external_flow"] = params.initial_cash_reserve

    if params.initial_investment > 0:
        p = prices.iloc[0]
        net, cost = _apply_cost(params.initial_investment, params.fixed_fee, params.pct_fee, params.slippage)
        units = net / p if p > 0 else 0.0
        total_fees += cost
        n_deployments += 1
        cash_flows.append((trading_days[0].date(), -params.initial_investment))
        ledger.at[trading_days[0], "external_flow"] = (
            _ledger_float(ledger, trading_days[0], "external_flow") + params.initial_investment
        )
        ledger.at[trading_days[0], "deployed"] = params.initial_investment
        ledger.at[trading_days[0], "fees"] = cost

    prev_dt: pd.Timestamp | None = None
    for i, dt in enumerate(trading_days):
        # 1. Accrue interest
        if prev_dt is not None and cash > 0:
            days_elapsed = (dt - prev_dt).days
            r = cash_rate_series.iloc[i - 1]
            if r > -1 and days_elapsed > 0:
                new_cash = cash * (1 + r) ** (days_elapsed / 365.25)
                earned = new_cash - cash
                total_interest += earned
                cash = new_cash
                ledger.at[dt, "interest"] = earned

        # 2. Receive contribution
        if dt in invest_map:
            contrib = invest_map[dt]
            cash += contrib
            cash_flows.append((dt.date(), -contrib))
            ledger.at[dt, "external_flow"] = contrib

        # 3. Check tier signals (i-1 signal → i execution)
        p = prices.iloc[i]
        if i > 0:
            signal_dd = float(dd_full.iloc[i - 1])
            ledger.at[dt, "signal_dd"] = signal_dd

            # Reset tiers and episode basis after new all-time high
            if prices.iloc[i - 1] >= peak.iloc[i - 1]:
                tiers_triggered = set()
                episode_cash_basis = None

            if cash > 0:
                deployed_this_step = 0.0
                fees_this_step = 0.0
                for j, tier in enumerate(tiers_sorted):
                    if j in tiers_triggered:
                        continue
                    if signal_dd <= tier["threshold"]:
                        # Record episode cash basis on first trigger
                        if episode_cash_basis is None:
                            episode_cash_basis = cash

                        fraction = tier["fraction"]
                        # Notional based on fixed episode opening balance
                        notional = episode_cash_basis * fraction
                        # Cap at actually available cash (for later tiers)
                        deploy_amount = min(notional, cash)

                        if deploy_amount > 0 and p > 0:
                            net, cost = _apply_cost(
                                deploy_amount, params.fixed_fee, params.pct_fee, params.slippage
                            )
                            units += net / p
                            total_fees += cost
                            fees_this_step += cost
                            n_deployments += 1
                            cash -= deploy_amount
                            deployed_this_step += deploy_amount
                            tiers_triggered.add(j)

                if deployed_this_step > 0:
                    ledger.at[dt, "deployed"] = _ledger_float(ledger, dt, "deployed") + deployed_this_step
                    ledger.at[dt, "fees"] = _ledger_float(ledger, dt, "fees") + fees_this_step

        # 4. Value portfolio
        market_val = units * p
        ledger.at[dt, "cash"] = cash
        ledger.at[dt, "units"] = units
        ledger.at[dt, "market_value"] = market_val
        ledger.at[dt, "total_wealth"] = cash + market_val
        prev_dt = dt

    total_contributions = (
        schedule["amount"].sum() + params.initial_investment + params.initial_cash_reserve
    )
    years = (pd.Timestamp(params.end_date) - pd.Timestamp(params.start_date)).days / 365.25

    result = _build_strategy_result(
        "Tiered-Dip", ledger, cash_flows, total_contributions, total_fees, total_interest, n_deployments, years
    )
    return result, ledger
