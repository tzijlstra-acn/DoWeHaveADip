"""Drawdown episode analysis."""

from __future__ import annotations

import pandas as pd

from dipdca.quant.drawdown import drawdown, drawdown_episodes


def load_named_episodes() -> list[dict]:
    """Load the curated named episode catalog from data/episodes.yaml."""
    from pathlib import Path

    import yaml

    episodes_path = Path(__file__).parent.parent.parent.parent / "data" / "episodes.yaml"
    with open(episodes_path) as f:
        return yaml.safe_load(f)["episodes"]


def analyze_episodes(
    price_series: pd.Series,
    threshold: float = -0.05,
) -> pd.DataFrame:
    """Identify and analyze drawdown episodes.

    Args:
        price_series: Price or total return index (DatetimeIndex).
        threshold: Drawdown level to define an episode start.

    Returns:
        DataFrame of episodes with start, trough, end, depth, duration.
    """
    dd = drawdown(price_series)
    episodes = drawdown_episodes(dd, threshold=threshold)
    return episodes


def dip_opportunity_count(
    price_series: pd.Series,
    threshold: float = -0.05,
) -> int:
    """Count distinct dip opportunities (episodes crossing threshold)."""
    episodes = analyze_episodes(price_series, threshold)
    return len(episodes)


def dip_recovery_stats(episodes: pd.DataFrame) -> dict:
    """Summarize episode recovery statistics.

    Returns:
        Dict with mean/median duration_days, mean/median trough_dd, count.
    """
    if episodes.empty:
        return {"count": 0}
    return {
        "count": len(episodes),
        "mean_duration_days": float(episodes["duration_days"].mean()),
        "median_duration_days": float(episodes["duration_days"].median()),
        "mean_trough_dd": float(episodes["trough_dd"].mean()),
        "median_trough_dd": float(episodes["trough_dd"].median()),
        "deepest_dd": float(episodes["trough_dd"].min()),
        "longest_duration_days": int(episodes["duration_days"].max()),
    }
