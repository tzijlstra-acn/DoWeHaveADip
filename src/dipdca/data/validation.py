"""Data quality validation utilities."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from dipdca.models import DataHealthStatus

logger = logging.getLogger(__name__)

STALE_THRESHOLD_DAYS = 5
MAX_GAP_DAYS = 10  # More than this many consecutive missing trading days is suspicious


def check_stale(last_date: date, threshold_days: int = STALE_THRESHOLD_DAYS) -> bool:
    """Return True if last_date is more than threshold_days before today."""
    return (date.today() - last_date).days > threshold_days


def check_large_gaps(df: pd.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> list[dict]:
    """Find large gaps in the price series.

    Returns list of dicts with start_date, end_date, gap_days.
    """
    if len(df) < 2:
        return []

    gaps = []
    idx = df.index.sort_values()
    for i in range(1, len(idx)):
        gap = (idx[i] - idx[i - 1]).days
        if gap > max_gap_days:
            gaps.append(
                {
                    "start_date": idx[i - 1].date(),
                    "end_date": idx[i].date(),
                    "gap_days": gap,
                }
            )
    return gaps


def check_suspicious_returns(df: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Flag single-day returns larger than threshold (likely data errors).

    Args:
        df: DataFrame with adj_close column.
        threshold: Daily return fraction to flag (default 50%).

    Returns:
        List of dicts with date, return.
    """
    if "adj_close" not in df.columns or len(df) < 2:
        return []

    returns = df["adj_close"].pct_change().dropna()
    suspicious = returns[returns.abs() > threshold]
    return [
        {"date": str(dt.date()), "return": float(val)} for dt, val in suspicious.items()
    ]


def build_health_status(
    symbol: str,
    df: pd.DataFrame | None,
    source: str = "unknown",
    error: str | None = None,
) -> DataHealthStatus:
    """Build a DataHealthStatus from a price DataFrame."""
    if df is None or len(df) == 0:
        return DataHealthStatus(symbol=symbol, source=source, error=error or "No data")

    last_date = df.index[-1].date()
    has_adj = "adj_close" in df.columns and not df["adj_close"].isna().all()

    return DataHealthStatus(
        symbol=symbol,
        last_date=last_date,
        n_rows=len(df),
        has_adj_close=has_adj,
        is_stale=check_stale(last_date),
        source=source,
        error=error,
    )
