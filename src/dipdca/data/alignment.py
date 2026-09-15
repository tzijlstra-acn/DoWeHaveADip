"""Time-series alignment utilities for multi-asset comparisons."""

from __future__ import annotations

import pandas as pd


def align_to_common_index(
    frames: dict[str, pd.DataFrame],
    method: str = "ffill",
    dropna: bool = True,
) -> dict[str, pd.DataFrame]:
    """Align multiple price DataFrames to a common DatetimeIndex.

    Args:
        frames: Dict of {name: DataFrame with DatetimeIndex}.
        method: Fill method for missing dates ('ffill', 'bfill', None).
        dropna: If True, drop dates where any series has NaN after fill.

    Returns:
        Dict of DataFrames aligned to the common intersection.
    """
    if not frames:
        return {}

    # Find common date range
    common_idx = None
    for df in frames.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    aligned = {}
    for name, df in frames.items():
        reindexed = df.reindex(common_idx, method=method)
        if dropna:
            reindexed = reindexed.dropna(subset=["adj_close"])
        aligned[name] = reindexed

    return aligned


def find_common_start(
    frames: dict[str, pd.DataFrame],
    min_overlap_days: int = 252,
) -> pd.Timestamp | None:
    """Find the latest start date across all frames with enough overlap.

    Returns None if overlap < min_overlap_days.
    """
    starts = [df.index.min() for df in frames.values()]
    ends = [df.index.max() for df in frames.values()]
    common_start = max(starts)
    common_end = min(ends)
    overlap = (common_end - common_start).days
    if overlap < min_overlap_days:
        return None
    return common_start


def reindex_to_business_days(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Reindex DataFrame to business days and forward-fill gaps."""
    bday_idx = pd.bdate_range(start, end)
    return df.reindex(bday_idx, method=fill_method)
