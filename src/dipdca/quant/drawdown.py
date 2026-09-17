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


def seeded_drawdown(series: pd.Series, initial_ath: float) -> pd.Series:
    """Drawdown relative to a pre-seeded all-time high.

    Like :func:`drawdown`, but the running peak starts at ``initial_ath``
    rather than the first value in ``series``.  Use this when the benchmark
    history before the evaluation window is known: callers should pass the
    ATH from that pre-window period so that the displayed drawdown is
    consistent with the engine's own calculation (which also seeds the ATH).

    Args:
        series: Price series (or benchmark close series), sorted ascending.
        initial_ath: The all-time high from before the series window.  When
            the series contains values exceeding ``initial_ath``, the peak
            updates to those values.

    Returns:
        Drawdown series with the same index as ``series``.  Values are ≤ 0;
        0.0 means a new all-time high relative to the seeded peak.
    """
    seeded = pd.concat(
        [pd.Series([initial_ath], index=[series.index[0] - pd.Timedelta(days=1)]), series]
    )
    seeded_peak = seeded.expanding().max()
    # Drop the synthetic seed row so the result aligns with the original index
    peak_aligned = seeded_peak.iloc[1:]
    peak_aligned.index = series.index
    return series / peak_aligned - 1


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
