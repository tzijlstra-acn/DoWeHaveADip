"""Walk-forward validation across rolling windows."""

from __future__ import annotations

import pandas as pd

from dipdca.models import SimulationParams
from dipdca.quant.backtest import run_dca, run_wait_for_dip


def walk_forward_comparison(
    price_data: pd.DataFrame,
    base_params: SimulationParams,
    window_years: int = 3,
    step_months: int = 6,
) -> pd.DataFrame:
    """Run rolling window backtest comparing DCA vs wait-for-dip.

    Args:
        price_data: Full price history with DatetimeIndex and adj_close.
        base_params: Base simulation parameters (start/end overridden per window).
        window_years: Duration of each window.
        step_months: Step between window starts.

    Returns:
        DataFrame with columns: window_start, window_end,
            dca_xirr, dip_xirr, dca_cagr, dip_cagr, winner.
    """
    records = []
    start = pd.Timestamp(base_params.start_date)
    end = pd.Timestamp(base_params.end_date)
    window_delta = pd.DateOffset(years=window_years)
    step_delta = pd.DateOffset(months=step_months)

    window_start = start
    while window_start + window_delta <= end:
        window_end = window_start + window_delta
        window_prices = price_data.loc[window_start:window_end]
        if len(window_prices) < 30:
            window_start += step_delta
            continue

        params = base_params.model_copy(
            update={
                "start_date": window_start.date(),
                "end_date": window_end.date(),
            }
        )

        try:
            dca_result, _ = run_dca(window_prices, params)
            dip_result, _ = run_wait_for_dip(window_prices, params)
            records.append(
                {
                    "window_start": window_start.date(),
                    "window_end": window_end.date(),
                    "dca_xirr": dca_result.xirr,
                    "dip_xirr": dip_result.xirr,
                    "dca_cagr": dca_result.cagr,
                    "dip_cagr": dip_result.cagr,
                    "dca_ending_wealth": dca_result.ending_wealth,
                    "dip_ending_wealth": dip_result.ending_wealth,
                    "winner": "DCA"
                    if dca_result.ending_wealth >= dip_result.ending_wealth
                    else "Dip",
                }
            )
        except Exception:
            pass

        window_start += step_delta

    return pd.DataFrame(records)
