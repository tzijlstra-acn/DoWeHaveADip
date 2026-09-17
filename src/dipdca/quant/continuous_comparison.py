"""Continuous DCA vs dip-accumulate comparison.

Models the behaviour of a saver who receives a fixed monthly contribution
throughout the evaluation window, comparing two strategies:

DCA
    Buy at every month-end payday, regardless of market conditions.

Dip accumulate
    Accumulate savings in cash each month; deploy all accumulated cash the
    first time the benchmark falls at or below the threshold from its running
    ATH.  Execution is on the next bar (T+1).  After deployment, any further
    monthly contributions are invested at each payday (DCA mode for the tail).

Both strategies invest the same total amount over the window — so the
comparison is purely about price: did waiting for the dip buy more units?

Payday convention
-----------------
A bar is treated as a payday (monthly contribution) if it is the last bar in
its calendar month that falls on or after the 25th day of the month.  This
mimics the ``payday=25`` parameter used by the simulation engine and avoids
treating mid-month bars (like early-March execution prices) as contribution
days.  If a month has no bar on or after the 25th, no contribution is received
in that month.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ContinuousComparisonResult:
    """Outcome of a DCA vs dip-accumulate comparison.

    Fractions are expressed as decimals; multiply by 100 for display.
    """

    dca_total_units: float
    """Total units purchased by the DCA strategy."""

    dip_total_units: float
    """Total units purchased by the dip strategy."""

    dip_deployed_at_signal: float
    """Cash deployed (all accumulated savings) at the threshold-triggered execution."""

    dca_effective_price: float
    """Average purchase price for DCA (total_cash / dca_total_units)."""

    dip_execution_price: float
    """Instrument price at which the bulk dip deployment executed."""

    unit_advantage: float
    """dip_total_units / dca_total_units - 1 (positive = dip bought more units)."""


def _build_payday_index(index: pd.DatetimeIndex, payday: int = 25) -> set:
    """Return the set of dates that are DCA payday bars.

    A payday bar is the LAST bar in each calendar month whose day-of-month is
    >= payday.  If a month has no bar on or after ``payday``, it contributes
    no payday bar (no contribution received that month).
    """
    month_periods = index.to_period("M")
    payday_dates: set = set()
    for period in month_periods.unique():
        mask = (month_periods == period) & (index.day >= payday)
        candidates = index[mask]
        if len(candidates) > 0:
            payday_dates.add(candidates[-1])
    return payday_dates


def run_continuous_comparison(
    instrument: pd.DataFrame,
    benchmark: pd.DataFrame,
    monthly_contribution: float,
    threshold: float,
    initial_ath: float | None = None,
) -> ContinuousComparisonResult | None:
    """Run a continuous DCA vs dip-accumulate comparison over the full window.

    Both strategies start from the beginning of the instrument series.

    Event ordering per bar
    ----------------------
    1. Execute any pending dip deployment at today's instrument price (T+1 fill).
    2. Check benchmark for threshold crossing (signal for next bar's execution).
       When the signal fires, the pending deployment equals the cash accumulated
       *before* this bar's payday contribution — preserving T+1 execution
       semantics and avoiding inclusion of the same-bar contribution.
    3. Process payday: receive monthly contribution.
       - DCA: buy immediately at today's price.
       - Dip (pre-trigger): add to accumulated cash.
       - Dip (post-trigger): buy immediately at today's price (DCA mode).

    Args:
        instrument: DataFrame with an ``adj_close`` column (execution prices).
        benchmark: DataFrame with an ``adj_close`` column (signal detection).
        monthly_contribution: Fixed monthly savings amount.
        threshold: Drawdown level that triggers the dip deployment (negative
            fraction, e.g. ``-0.15`` for a 15% drawdown).
        initial_ath: Seed the running ATH from pre-window history.  When
            ``None``, the first bar of the series is used as the initial ATH.

    Returns:
        ``ContinuousComparisonResult`` when the threshold is crossed at least
        once and a T+1 execution bar exists.  ``None`` otherwise.
    """
    bm = benchmark["adj_close"].dropna()
    inst = instrument["adj_close"].dropna()

    # Align to shared dates
    common = bm.index.intersection(inst.index)
    if len(common) < 2:
        return None
    bm = bm.reindex(common)
    inst = inst.reindex(common)

    # Identify payday bars (last bar per month with day >= 25)
    payday_dates = _build_payday_index(pd.DatetimeIndex(bm.index))

    # Running ATH starts from the seeded value or first bar
    peak = float(initial_ath) if initial_ath is not None else float(bm.iloc[0])

    # --- Strategy state ---
    dca_units: float = 0.0
    dca_total_paid: float = 0.0

    dip_units: float = 0.0
    dip_accumulated_cash: float = 0.0
    dip_deployed_at_signal: float = 0.0
    dip_execution_price: float = float("nan")
    dip_pending_deployment: float = 0.0   # cash queued for next-bar execution
    dip_triggered: bool = False

    for dt, bm_val in bm.items():
        bm_price = float(bm_val)
        inst_price = float(inst.at[dt])  # type: ignore[arg-type]

        # --- Step 1: Execute pending dip deployment (placed on prior bar) ---
        if dip_pending_deployment > 0.0:
            units_bought = dip_pending_deployment / inst_price
            dip_units += units_bought
            dip_execution_price = inst_price
            dip_deployed_at_signal = dip_pending_deployment
            dip_pending_deployment = 0.0

        # --- Update running ATH ---
        if bm_price > peak:
            peak = bm_price

        # --- Step 2: Check threshold (BEFORE payday contribution) ---
        # This ensures the pending deployment excludes the current bar's contribution,
        # matching T+1 execution semantics from the backtest engine.
        if not dip_triggered and peak > 0.0:
            dd = bm_price / peak - 1.0
            if dd <= threshold:
                dip_triggered = True
                # Queue current accumulated cash (no current-bar contribution yet)
                dip_pending_deployment = dip_accumulated_cash
                dip_accumulated_cash = 0.0

        # --- Step 3: Payday — receive monthly contribution ---
        if dt in payday_dates:
            # DCA: always buy at today's close
            dca_units += monthly_contribution / inst_price
            dca_total_paid += monthly_contribution

            # Dip strategy
            if not dip_triggered:
                # Still accumulating — add to cash reserve
                dip_accumulated_cash += monthly_contribution
            else:
                # Post-trigger: invest immediately (DCA mode for tail contributions)
                dip_units += monthly_contribution / inst_price

    # If threshold was never crossed, or pending order was never executed
    if not dip_triggered:
        return None
    if dip_pending_deployment > 0.0:
        # Triggered on last bar — no T+1 execution bar exists
        return None

    # Compute summary statistics
    dca_effective_price = dca_total_paid / dca_units if dca_units > 0 else float("nan")
    unit_advantage = (
        dip_units / dca_units - 1.0 if dca_units > 0 else float("nan")
    )

    return ContinuousComparisonResult(
        dca_total_units=dca_units,
        dip_total_units=dip_units,
        dip_deployed_at_signal=dip_deployed_at_signal,
        dca_effective_price=dca_effective_price,
        dip_execution_price=dip_execution_price,
        unit_advantage=unit_advantage,
    )
