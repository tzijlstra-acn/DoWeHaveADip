"""Total return index construction and validation."""

from __future__ import annotations

import pandas as pd


def normalize_total_return(
    adj_close: pd.Series,
    fx: pd.Series,
    t0: pd.Timestamp,
) -> pd.Series:
    """Construct total return index in base currency, rebased to 1.0 at t0.

    TR_base(t) = [AdjClose(t) * FX(t)] / [AdjClose(t0) * FX(t0)]

    Args:
        adj_close: Adjusted close prices (total return in native currency).
        fx: FX rate series (base_per_asset), aligned to adj_close.
        t0: Reference date. Result equals 1.0 at t0.

    Returns:
        Total return index in base currency, value = 1.0 at t0.

    Raises:
        ValueError: If t0 not in adj_close index.
    """
    if t0 not in adj_close.index:
        # Find closest available date
        pos = adj_close.index.searchsorted(t0)
        if pos >= len(adj_close.index):
            raise ValueError(f"t0={t0} is beyond the price series range")
        t0 = adj_close.index[pos]

    fx_aligned = fx.reindex(adj_close.index, method="ffill")

    combined = adj_close * fx_aligned
    ref_value = combined.loc[t0]
    if ref_value == 0:
        raise ValueError(f"Reference value at t0={t0} is zero — cannot normalize")

    return combined / ref_value


def validate_adjusted_data(df: pd.DataFrame) -> bool:
    """Return True if adj_close is present and differs from close (i.e. dividend-adjusted).

    Returns False if:
    - adj_close column not present
    - adj_close is identical to close (no adjustment applied)
    - adj_close is all NaN
    """
    if "adj_close" not in df.columns:
        return False
    if df["adj_close"].isna().all():
        return False
    if "close" in df.columns:
        # If all values are equal, adjustment was not applied
        diff = (df["adj_close"] - df["close"]).abs()
        if diff.max() < 1e-8:
            return False
    return True


def compute_period_return(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Compute total return between two dates as a fraction (e.g. 0.15 = 15%).

    Uses adj_close values directly or a total return index.
    """
    s = series.loc[start:end]
    if len(s) < 2:
        return 0.0
    return float(s.iloc[-1] / s.iloc[0] - 1)


def cagr(total_return: float, years: float) -> float:
    """Annualize a total return over a given number of years."""
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1
