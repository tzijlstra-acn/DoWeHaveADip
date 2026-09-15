"""Tests for drawdown episode analysis."""

from __future__ import annotations

import pandas as pd

from dipdca.quant.episodes import analyze_episodes, dip_opportunity_count, dip_recovery_stats


def make_series(prices: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.Series(prices, index=idx)


class TestAnalyzeEpisodes:
    def test_no_dip_flat_series(self):
        s = make_series([100.0] * 50)
        episodes = analyze_episodes(s, threshold=-0.05)
        assert len(episodes) == 0

    def test_single_episode(self):
        prices = [100.0] * 20 + [90.0] * 10 + [105.0] * 20
        s = make_series(prices)
        episodes = analyze_episodes(s, threshold=-0.05)
        assert len(episodes) >= 1
        assert episodes["trough_dd"].min() <= -0.09

    def test_multiple_episodes(self):
        # Two separate dips
        prices = (
            [100.0] * 10
            + [90.0] * 10  # dip 1
            + [110.0] * 10  # recovery
            + [95.0] * 10  # dip 2
            + [115.0] * 10  # recovery
        )
        s = make_series(prices)
        episodes = analyze_episodes(s, threshold=-0.05)
        assert len(episodes) >= 2

    def test_episode_columns(self):
        prices = [100.0] * 20 + [80.0] * 20 + [105.0] * 10
        s = make_series(prices)
        episodes = analyze_episodes(s, threshold=-0.05)
        if len(episodes) > 0:
            required = {"start", "trough_date", "end", "trough_dd", "duration_days"}
            assert required.issubset(set(episodes.columns))


class TestDipOpportunityCount:
    def test_rising_market_no_dips(self):
        s = make_series([100.0 + i for i in range(100)])
        count = dip_opportunity_count(s, threshold=-0.05)
        assert count == 0

    def test_crash_counts_one(self):
        prices = [100.0] * 20 + [85.0] * 20 + [110.0] * 10
        s = make_series(prices)
        count = dip_opportunity_count(s, threshold=-0.05)
        assert count >= 1


class TestDipRecoveryStats:
    def test_empty_episodes_returns_count_zero(self):
        result = dip_recovery_stats(pd.DataFrame())
        assert result["count"] == 0

    def test_single_episode_stats(self):
        prices = [100.0] * 20 + [80.0] * 30 + [105.0] * 10
        s = make_series(prices)
        episodes = analyze_episodes(s, threshold=-0.05)
        stats = dip_recovery_stats(episodes)
        assert stats["count"] >= 1
        assert stats["deepest_dd"] < 0
        assert stats["mean_duration_days"] > 0
