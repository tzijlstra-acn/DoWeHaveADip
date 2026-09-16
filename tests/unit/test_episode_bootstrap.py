"""Tests for the episode-level paired bootstrap.

Key invariants:
- With all episodes where policy beats DCA: win_rate = 1.0, CI lower > 0.5.
- With all episodes where DCA beats policy: win_rate = 0.0, CI upper < 0.5.
- Median relative return and percentiles are computed on the episode-level rels.
- Returns None when fewer than 2 eligible episodes exist.
- bootstrap_all_thresholds skips thresholds with insufficient data.
- results_to_dataframe produces a row per threshold.
- CI width shrinks with more episodes (stochastic, checked directionally).
- Reproducible under the same seed; different seeds may differ.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dipdca.quant.ath_episodes import ATHEpisode, EpisodeStudy, PolicyOutcome
from dipdca.quant.episode_bootstrap import (
    bootstrap_all_thresholds,
    bootstrap_win_rate,
    results_to_dataframe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_episode(ath_date: str = "2020-01-01") -> ATHEpisode:
    return ATHEpisode(
        ath_date=pd.Timestamp(ath_date),
        ath_level=100.0,
        trough_date=pd.Timestamp("2020-06-01"),
        trough_drawdown=-0.20,
        recovery_date=pd.Timestamp("2021-01-01"),
        first_crossings={-0.15: pd.Timestamp("2020-03-01")},
    )


def make_study(
    ath_date: str,
    dip_wealth: float,
    dca_wealth: float,
    threshold: float = -0.15,
    horizon_label: str = "12m",
    policy: str = "ATH all-in",
) -> EpisodeStudy:
    contributions = 10_000.0
    outcomes = [
        PolicyOutcome(
            policy=policy,
            horizon_label=horizon_label,
            ending_wealth=dip_wealth,
            total_contributions=contributions,
            n_deployments=1,
            ending_cash=0.0,
        ),
        PolicyOutcome(
            policy="DCA",
            horizon_label=horizon_label,
            ending_wealth=dca_wealth,
            total_contributions=contributions,
            n_deployments=12,
            ending_cash=0.0,
        ),
    ]
    return EpisodeStudy(
        episode=make_episode(ath_date),
        threshold=threshold,
        signal_date=pd.Timestamp(ath_date),
        outcomes=outcomes,
    )


def all_dip_wins(n: int = 8) -> list[EpisodeStudy]:
    dates = [f"20{20+i:02d}-01-01" for i in range(n)]
    return [make_study(d, dip_wealth=12_000.0, dca_wealth=10_000.0) for d in dates]


def all_dca_wins(n: int = 8) -> list[EpisodeStudy]:
    dates = [f"20{20+i:02d}-01-01" for i in range(n)]
    return [make_study(d, dip_wealth=9_000.0, dca_wealth=10_000.0) for d in dates]


def mixed(n_dip: int = 5, n_dca: int = 3) -> list[EpisodeStudy]:
    dip_dates = [f"20{20+i:02d}-01-01" for i in range(n_dip)]
    dca_dates = [f"20{20+n_dip+i:02d}-01-01" for i in range(n_dca)]
    studies = [make_study(d, dip_wealth=12_000.0, dca_wealth=10_000.0) for d in dip_dates]
    studies += [make_study(d, dip_wealth=9_000.0, dca_wealth=10_000.0) for d in dca_dates]
    return studies


# ---------------------------------------------------------------------------
# Win rate point estimates
# ---------------------------------------------------------------------------

class TestWinRatePointEstimate:
    def test_all_dip_wins_gives_win_rate_1(self):
        result = bootstrap_win_rate(all_dip_wins(), threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.win_rate == pytest.approx(1.0)

    def test_all_dca_wins_gives_win_rate_0(self):
        result = bootstrap_win_rate(all_dca_wins(), threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.win_rate == pytest.approx(0.0)

    def test_mixed_win_rate_is_fraction_of_dip_wins(self):
        result = bootstrap_win_rate(mixed(5, 3), threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.win_rate == pytest.approx(5 / 8)

    def test_n_episodes_matches_eligible_count(self):
        studies = all_dip_wins(n=6)
        result = bootstrap_win_rate(studies, threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.n_episodes == 6

    def test_returns_none_with_one_episode(self):
        studies = [make_study("2020-01-01", 12_000.0, 10_000.0)]
        result = bootstrap_win_rate(studies, threshold=-0.15, horizon_label="12m")
        assert result is None

    def test_returns_none_with_no_episodes(self):
        result = bootstrap_win_rate([], threshold=-0.15, horizon_label="12m")
        assert result is None

    def test_threshold_filter_excludes_wrong_threshold(self):
        # Studies are all at -0.15; querying -0.25 should find nothing
        result = bootstrap_win_rate(all_dip_wins(8), threshold=-0.25, horizon_label="12m")
        assert result is None


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

class TestConfidenceIntervals:
    def test_all_wins_ci_lower_above_half(self):
        result = bootstrap_win_rate(all_dip_wins(8), threshold=-0.15, horizon_label="12m", n_boot=500)
        assert result is not None
        assert result.ci_lower > 0.5

    def test_all_losses_ci_upper_below_half(self):
        result = bootstrap_win_rate(all_dca_wins(8), threshold=-0.15, horizon_label="12m", n_boot=500)
        assert result is not None
        assert result.ci_upper < 0.5

    def test_ci_contains_win_rate(self):
        result = bootstrap_win_rate(mixed(5, 3), threshold=-0.15, horizon_label="12m", n_boot=500)
        assert result is not None
        assert result.ci_lower <= result.win_rate <= result.ci_upper

    def test_ci_level_stored_in_result(self):
        result = bootstrap_win_rate(all_dip_wins(), threshold=-0.15, horizon_label="12m", ci_level=0.90)
        assert result is not None
        assert result.ci_level == pytest.approx(0.90)

    def test_more_episodes_gives_narrower_ci(self):
        small = bootstrap_win_rate(mixed(4, 2), threshold=-0.15, horizon_label="12m", n_boot=1000, seed=0)
        large = bootstrap_win_rate(mixed(16, 8), threshold=-0.15, horizon_label="12m", n_boot=1000, seed=0)
        assert small is not None and large is not None
        small_width = small.ci_upper - small.ci_lower
        large_width = large.ci_upper - large.ci_lower
        assert large_width < small_width


# ---------------------------------------------------------------------------
# Median and percentiles
# ---------------------------------------------------------------------------

class TestMedianAndPercentiles:
    def test_all_wins_median_vs_dca_positive(self):
        result = bootstrap_win_rate(all_dip_wins(), threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.median_vs_dca == pytest.approx(0.20)  # 12000/10000 - 1

    def test_all_losses_median_vs_dca_negative(self):
        result = bootstrap_win_rate(all_dca_wins(), threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.median_vs_dca == pytest.approx(-0.10)  # 9000/10000 - 1

    def test_p10_le_median_le_p90(self):
        result = bootstrap_win_rate(mixed(5, 3), threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.p10_vs_dca <= result.median_vs_dca
        assert result.median_vs_dca <= result.p90_vs_dca

    def test_p25_le_p75(self):
        result = bootstrap_win_rate(mixed(5, 3), threshold=-0.15, horizon_label="12m")
        assert result is not None
        assert result.p25_vs_dca <= result.p75_vs_dca


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        studies = mixed(5, 3)
        r1 = bootstrap_win_rate(studies, -0.15, "12m", seed=42)
        r2 = bootstrap_win_rate(studies, -0.15, "12m", seed=42)
        assert r1 is not None and r2 is not None
        assert r1.ci_lower == pytest.approx(r2.ci_lower)
        assert r1.ci_upper == pytest.approx(r2.ci_upper)

    def test_different_seeds_may_differ(self):
        studies = mixed(4, 4)
        r1 = bootstrap_win_rate(studies, -0.15, "12m", n_boot=100, seed=0)
        r2 = bootstrap_win_rate(studies, -0.15, "12m", n_boot=100, seed=99)
        assert r1 is not None and r2 is not None
        # Win rate is the same (point estimate); CI may differ
        assert r1.win_rate == pytest.approx(r2.win_rate)


# ---------------------------------------------------------------------------
# Multi-threshold convenience function
# ---------------------------------------------------------------------------

class TestBootstrapAllThresholds:
    def test_returns_results_for_populated_thresholds(self):
        studies_15 = all_dip_wins(6)
        studies_25 = [make_study(f"20{30+i:02d}-01-01", 12_000, 10_000, threshold=-0.25)
                      for i in range(4)]
        all_studies = studies_15 + studies_25
        results = bootstrap_all_thresholds(
            all_studies, thresholds=(-0.15, -0.25), horizon_label="12m"
        )
        assert len(results) == 2
        assert {r.threshold for r in results} == {-0.15, -0.25}

    def test_skips_thresholds_with_insufficient_data(self):
        studies = all_dip_wins(6)  # only threshold -0.15
        results = bootstrap_all_thresholds(
            studies, thresholds=(-0.15, -0.25), horizon_label="12m"
        )
        assert len(results) == 1
        assert results[0].threshold == pytest.approx(-0.15)

    def test_empty_studies_returns_empty_list(self):
        results = bootstrap_all_thresholds([], thresholds=(-0.15, -0.25), horizon_label="12m")
        assert results == []


# ---------------------------------------------------------------------------
# results_to_dataframe
# ---------------------------------------------------------------------------

class TestResultsToDataframe:
    def test_one_row_per_result(self):
        results = bootstrap_all_thresholds(
            all_dip_wins(6), thresholds=(-0.15,), horizon_label="12m"
        )
        df = results_to_dataframe(results)
        assert len(df) == 1

    def test_columns_include_win_rate_and_ci(self):
        results = bootstrap_all_thresholds(
            all_dip_wins(6), thresholds=(-0.15,), horizon_label="12m"
        )
        df = results_to_dataframe(results)
        assert "Win rate" in df.columns
        assert "Episodes" in df.columns
        assert "Median vs DCA" in df.columns

    def test_empty_results_gives_empty_dataframe(self):
        df = results_to_dataframe([])
        assert len(df) == 0
