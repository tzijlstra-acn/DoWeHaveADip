"""Drawdown calculations and labelling."""

from __future__ import annotations

import pandas as pd

# Sorted from shallowest to deepest for lookup order
DRAWDOWN_LABELS: dict[tuple[float, float], str] = {
    (-0.05, 0.0): "Barely a dip",
    (-0.10, -0.05): "Snack-size salsa",
    (-0.20, -0.10): "Proper nacho dip",
    (-0.35, -0.20): "Bear-market guacamole",
    (-1.0, -0.35): "Financial Mariana Trench",
}


def running_peak(series: pd.Series) -> pd.Series:
    """Expanding maximum — running peak."""
    return series.expanding().max()


def drawdown(series: pd.Series) -> pd.Series:
    """drawdown(t) = series(t) / running_peak(t) - 1.

    Returns 0.0 at new highs, negative values in drawdown.
    """
    peak = running_peak(series)
    return series / peak - 1


def max_drawdown(series: pd.Series) -> float:
    """Return maximum (deepest) drawdown as a negative fraction."""
    dd = drawdown(series)
    return float(dd.min())


def drawdown_label(dd: float) -> str:
    """Return a humorous label for the given drawdown level."""
    for (low, high), label in DRAWDOWN_LABELS.items():
        if low <= dd <= high:
            return label
    # Below all ranges
    return "Financial Mariana Trench"


def drawdown_episodes(dd_series: pd.Series, threshold: float = -0.05) -> pd.DataFrame:
    """Identify discrete drawdown episodes below threshold.

    Returns DataFrame with columns: start, trough_date, end, trough_dd, duration_days.
    """
    in_episode = dd_series <= threshold
    records = []
    episode_start: pd.Timestamp | None = None
    trough_date: pd.Timestamp | None = None
    trough_val = 0.0

    for dt_raw, val in dd_series.items():
        dt = pd.Timestamp(dt_raw)  # type: ignore[arg-type]
        if in_episode[dt]:
            if episode_start is None:
                episode_start = dt
                trough_date = dt
                trough_val = val
            elif val < trough_val:
                trough_val = val
                trough_date = dt
        elif episode_start is not None:
            records.append(
                {
                    "start": episode_start,
                    "trough_date": trough_date,
                    "end": dt,
                    "trough_dd": trough_val,
                    "duration_days": (dt - episode_start).days,
                }
            )
            episode_start = None
            trough_date = None
            trough_val = 0.0

    # Close open episode
    if episode_start is not None:
        last_dt = dd_series.index[-1]
        records.append(
            {
                "start": episode_start,
                "trough_date": trough_date,
                "end": last_dt,
                "trough_dd": trough_val,
                "duration_days": (last_dt - episode_start).days,
            }
        )

    return pd.DataFrame(records)
