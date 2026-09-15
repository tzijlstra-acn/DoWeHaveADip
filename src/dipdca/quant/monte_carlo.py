"""
Monte Carlo simulation for dip-buying strategies.

Two simulation modes:

1. PARAMETER SWEEP — grid search over (threshold, deploy_pct) pairs.
   For each pair, run the backtest across ALL rolling windows of W years
   in the historical dataset. Record win rate vs DCA.

2. CONDITIONAL PATH BOOTSTRAP — given we're currently at drawdown X%,
   sample historical continuation paths from that same drawdown level.
   Shows the distribution of outcomes depending on how much you deploy now.

No Streamlit imports — this module is safe to import in tests and CLI contexts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dipdca.models import SimulationParams
from dipdca.quant.backtest import run_dca, run_wait_for_dip
from dipdca.quant.drawdown import drawdown as compute_drawdown


@dataclass
class SweepResult:
    threshold: float
    deploy_pct: float
    win_rate: float               # fraction of windows where dip beat DCA
    median_outperformance: float  # median (dip_wealth / dca_wealth - 1)
    p10_outperformance: float     # 10th percentile — worst case
    p90_outperformance: float     # 90th percentile — best case
    n_windows: int                # number of rolling windows tested
    median_deployments: float     # median number of dip triggers per window


@dataclass
class PathSimulation:
    deploy_pct: float
    p5_wealth: np.ndarray    # 5th percentile wealth path
    p25_wealth: np.ndarray   # 25th percentile
    p50_wealth: np.ndarray   # median
    p75_wealth: np.ndarray   # 75th percentile
    p95_wealth: np.ndarray   # 95th percentile
    prob_beats_dca: float    # probability of beating DCA at end of horizon
    horizon_months: int


def run_parameter_sweep(
    prices: pd.Series,
    base_params: SimulationParams,
    thresholds: list[float] | None = None,
    deploy_pcts: list[float] | None = None,
    window_years: int = 10,
    step_months: int = 12,
    points_per_year: int = 252,
) -> list[SweepResult]:
    """Grid search over threshold x deploy_pct.

    For each combination:
    - Slide a window of `window_years` across the full price history
    - Step by `step_months` between windows
    - Run DCA and wait-for-dip backtest in each window
    - Record whether dip beat DCA (by ending wealth)

    Args:
        prices: Price series (daily or monthly — set points_per_year accordingly).
        base_params: Base simulation parameters (threshold and deploy_pct overridden).
        thresholds: Drawdown thresholds to test (negative fractions).
        deploy_pcts: Deployment percentages to test (0-1 fractions).
        window_years: Window length in years.
        step_months: Step between windows in months.
        points_per_year: Data points per calendar year (252 for daily, 12 for monthly).

    Returns:
        List of SweepResult, one per (threshold, deploy_pct) pair with enough data.
    """
    if thresholds is None:
        thresholds = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40]
    if deploy_pcts is None:
        deploy_pcts = [0.25, 0.50, 0.75, 1.00]

    points_per_month = max(1, points_per_year // 12)
    window_size = window_years * points_per_year
    step_size = step_months * points_per_month

    results = []

    for threshold in thresholds:
        for deploy_pct in deploy_pcts:
            outperformances: list[float] = []
            deployments_list: list[float] = []

            # Slide windows across full history
            i = 0
            while i + window_size <= len(prices):
                window = prices.iloc[i : i + window_size]

                w_start = window.index[0].date()
                w_end = window.index[-1].date()

                if w_end <= w_start:
                    i += step_size
                    continue

                # Build params for this window
                try:
                    params = SimulationParams(
                        monthly_contribution=base_params.monthly_contribution,
                        payday=base_params.payday,
                        initial_investment=base_params.initial_investment,
                        start_date=w_start,
                        end_date=w_end,
                        dip_threshold=threshold,
                        max_wait_months=base_params.max_wait_months,
                        fixed_fee=base_params.fixed_fee,
                        pct_fee=base_params.pct_fee,
                        deployment_pct=deploy_pct,
                        cash_buffer_months=base_params.cash_buffer_months,
                        deploy_spread_months=base_params.deploy_spread_months,
                    )
                except Exception:
                    i += step_size
                    continue

                # Build price DataFrame with adj_close column for backtest
                price_df = window.to_frame("adj_close")

                try:
                    dca_result, _ = run_dca(price_df, params)
                    dip_result, _ = run_wait_for_dip(price_df, params)

                    if dca_result.ending_wealth > 0:
                        outperformance = dip_result.ending_wealth / dca_result.ending_wealth - 1
                        outperformances.append(outperformance)
                        deployments_list.append(float(dip_result.n_deployments))
                except Exception:
                    pass  # skip windows with insufficient data

                i += step_size

            if len(outperformances) < 3:
                continue  # not enough windows for meaningful stats

            arr = np.array(outperformances)
            results.append(
                SweepResult(
                    threshold=threshold,
                    deploy_pct=deploy_pct,
                    win_rate=float(np.mean(arr > 0)),
                    median_outperformance=float(np.median(arr)),
                    p10_outperformance=float(np.percentile(arr, 10)),
                    p90_outperformance=float(np.percentile(arr, 90)),
                    n_windows=len(arr),
                    median_deployments=float(np.median(deployments_list)),
                )
            )

    return results


def sweep_to_dataframe(results: list[SweepResult]) -> pd.DataFrame:
    """Convert sweep results to a flat DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "threshold": r.threshold,
                "deploy_pct": r.deploy_pct,
                "win_rate": r.win_rate,
                "median_outperformance": r.median_outperformance,
                "p10": r.p10_outperformance,
                "p90": r.p90_outperformance,
                "n_windows": r.n_windows,
                "median_deployments": r.median_deployments,
            }
        )
    return pd.DataFrame(rows)


def win_rate_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sweep DataFrame to threshold (rows) x deploy_pct (cols), values = win_rate.

    Thresholds formatted as percentages (e.g., "-10%"), deploy_pcts as (e.g., "50%").
    """
    pivot = df.pivot(index="threshold", columns="deploy_pct", values="win_rate")
    pivot.index = [f"{t:.0%}" for t in pivot.index]
    pivot.columns = [f"{d:.0%}" for d in pivot.columns]
    return pivot


def outperformance_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Same pivot but for median outperformance."""
    pivot = df.pivot(index="threshold", columns="deploy_pct", values="median_outperformance")
    pivot.index = [f"{t:.0%}" for t in pivot.index]
    pivot.columns = [f"{d:.0%}" for d in pivot.columns]
    return pivot


def conditional_path_bootstrap(
    prices: pd.Series,
    current_drawdown: float,
    deploy_pcts: list[float],
    monthly_contribution: float,
    cash_accumulated: float,
    horizon_months: int = 24,
    n_simulations: int = 500,
    cash_rate: float = 0.02,
) -> list[PathSimulation]:
    """Bootstrap historical continuation paths given current drawdown level.

    Given that we're currently at `current_drawdown` (e.g., -0.15), sample historical
    continuation paths from similar drawdown levels to show the distribution of outcomes.

    Algorithm:
    1. Compute full drawdown series from historical prices.
    2. Find all dates where drawdown was within ±5% of current_drawdown.
    3. For each such date, extract the next horizon_months of price returns.
    4. For each deploy_pct: simulate N paths using random continuation samples.
    5. Return percentile bands across simulations.

    Args:
        prices: Full price history (daily or monthly Series).
        current_drawdown: Current drawdown level (negative, e.g., -0.15).
        deploy_pcts: List of deployment fractions to test.
        monthly_contribution: Monthly cash contribution amount.
        cash_accumulated: Total cash currently sitting uninvested.
        horizon_months: Simulation horizon in months.
        n_simulations: Number of bootstrap draws per deploy_pct.
        cash_rate: Annual interest rate on uninvested cash.

    Returns:
        List of PathSimulation, one per deploy_pct. Empty list if insufficient history.
    """
    dd_series = compute_drawdown(prices)
    monthly_prices = prices.resample("ME").last().dropna()

    # Find entry points within ±5% of current drawdown
    tolerance = 0.05
    entry_mask = (dd_series <= current_drawdown + tolerance) & (
        dd_series >= current_drawdown - tolerance
    )
    entry_dates = dd_series[entry_mask].index

    if len(entry_dates) < 5:
        # Widen tolerance if too few samples
        entry_mask2 = (dd_series <= current_drawdown + 0.10) & (
            dd_series >= current_drawdown - 0.10
        )
        entry_dates = dd_series[entry_mask2].index

    if len(entry_dates) < 3:
        return []

    # Extract continuation return paths (horizon_months long)
    continuation_paths: list[np.ndarray] = []
    for entry in entry_dates:
        try:
            entry_price = float(monthly_prices.asof(entry))  # type: ignore[arg-type]
            if entry_price <= 0 or np.isnan(entry_price):
                continue
            future_idx = monthly_prices.index[monthly_prices.index > entry][:horizon_months]
            if len(future_idx) >= int(horizon_months * 0.8):  # at least 80% of horizon
                future_prices = monthly_prices.loc[future_idx]
                returns = (future_prices / entry_price).values  # cumulative return factors
                continuation_paths.append(returns)
        except Exception:
            continue

    if not continuation_paths:
        return []

    monthly_cash_factor = (1 + cash_rate) ** (1 / 12)

    results: list[PathSimulation] = []

    for deploy_pct in deploy_pcts:
        sim_terminal_wealths: list[float] = []
        sim_wealth_paths: list[list[float]] = []
        sim_dca_terminals: list[float] = []

        for _ in range(n_simulations):
            # Sample a continuation path
            path = continuation_paths[np.random.randint(len(continuation_paths))]
            n_months = len(path)

            # --- Dip strategy ---
            deployed = cash_accumulated * deploy_pct
            cash_remaining = cash_accumulated * (1 - deploy_pct)

            wealth_path: list[float] = []
            for m in range(n_months):
                return_factor = float(path[m])  # cumulative return from entry
                # Monthly contribution goes to cash
                cash_remaining = cash_remaining * monthly_cash_factor + monthly_contribution
                # Portfolio value: initially deployed amount * return
                portfolio = deployed * return_factor
                total_wealth = portfolio + cash_remaining
                wealth_path.append(total_wealth)

            sim_wealth_paths.append(wealth_path)
            sim_terminal_wealths.append(wealth_path[-1] if wealth_path else 0.0)

            # --- DCA baseline (invest monthly, cash_accumulated stays in cash) ---
            dca_invested = 0.0
            dca_cash = cash_accumulated
            for m in range(n_months):
                return_factor = float(path[m])
                dca_invested += monthly_contribution  # simplification: invest at start of each month
                dca_terminal = dca_invested * return_factor + dca_cash * (
                    monthly_cash_factor ** (m + 1)
                )
            sim_dca_terminals.append(dca_terminal)

        # Build wealth matrix for percentile computation
        # Paths may differ in length — pad shorter ones with last value
        max_len = max(len(p) for p in sim_wealth_paths) if sim_wealth_paths else horizon_months
        padded = []
        for wp in sim_wealth_paths:
            if len(wp) < max_len:
                wp = wp + [wp[-1]] * (max_len - len(wp))
            padded.append(wp[:max_len])

        wealth_matrix = np.array(padded)  # shape: (n_simulations, n_months)

        terminal_arr = np.array(sim_terminal_wealths)
        dca_arr = np.array(sim_dca_terminals)
        prob_beats_dca = float(np.mean(terminal_arr > dca_arr))

        results.append(
            PathSimulation(
                deploy_pct=deploy_pct,
                p5_wealth=np.percentile(wealth_matrix, 5, axis=0),
                p25_wealth=np.percentile(wealth_matrix, 25, axis=0),
                p50_wealth=np.percentile(wealth_matrix, 50, axis=0),
                p75_wealth=np.percentile(wealth_matrix, 75, axis=0),
                p95_wealth=np.percentile(wealth_matrix, 95, axis=0),
                prob_beats_dca=prob_beats_dca,
                horizon_months=max_len,
            )
        )

    return results
