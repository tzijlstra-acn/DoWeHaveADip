"""ATH-episode identification and event study.

Central invariants:
- an episode is anchored to ONE all-time high and closes only when that level is
  recovered, so the anchor never drifts;
- each threshold records only its FIRST crossing, so adjacent crash days cannot
  be counted as separate scenarios;
- an episode still open at the end of the data is censored, not a recovery.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dipdca.models import DeploymentTier
from dipdca.quant.ath_episodes import (
    episodes_to_dataframe,
    find_ath_episodes,
    run_event_study,
    study_episode,
    summarise_threshold,
)


def series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx)


def frame(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


THRESHOLDS = (-0.10, -0.20, -0.30)


class TestEpisodeIdentification:
    def test_single_v_shaped_episode(self):
        # 100 -> 70 (-30%) -> back above 100
        vals = [100.0] + [70.0] * 10 + [105.0] * 5
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert len(eps) == 1
        ep = eps[0]
        assert ep.ath_level == 100.0
        assert abs(ep.trough_drawdown - (-0.30)) < 1e-9
        assert not ep.is_censored

    def test_adjacent_crash_days_are_one_episode_not_many(self):
        """A long crash is a single observation, however many days it spans."""
        vals = [100.0] + [70.0] * 200 + [105.0]
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert len(eps) == 1

    def test_first_crossing_only_is_recorded(self):
        """Re-crossing the same level inside one episode adds no new crossing."""
        # Dips below -20%, partially recovers to -5%, dips below -20% again.
        vals = [100.0] + [75.0] * 5 + [95.0] * 5 + [75.0] * 5 + [105.0]
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert len(eps) == 1
        crossings = eps[0].first_crossings
        # -20% recorded exactly once, at its first occurrence.
        assert -0.20 in crossings
        assert crossings[-0.20] == series(vals).index[1]

    def test_two_separate_episodes_when_ath_recovered_between(self):
        vals = (
            [100.0] + [70.0] * 5 + [105.0] * 3      # episode 1, recovers
            + [70.0] * 5 + [120.0] * 3              # episode 2, recovers
        )
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert len(eps) == 2
        assert eps[0].ath_level == 100.0
        assert eps[1].ath_level == 105.0
        assert all(not e.is_censored for e in eps)

    def test_anchor_does_not_drift_to_an_interior_lower_high(self):
        """A rally that fails to reclaim the anchor must not re-anchor."""
        # 100 -> 70 -> 90 (interior high, still below 100) -> 60 -> recover
        vals = [100.0] + [70.0] * 3 + [90.0] * 3 + [60.0] * 3 + [101.0]
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert len(eps) == 1
        ep = eps[0]
        assert ep.ath_level == 100.0
        # Trough measured against the anchor, not the interior high.
        assert abs(ep.trough_drawdown - (-0.40)) < 1e-9

    def test_unrecovered_episode_is_censored(self):
        vals = [100.0] + [70.0] * 10
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert len(eps) == 1
        assert eps[0].is_censored
        assert eps[0].recovery_date is None
        assert eps[0].days_to_recovery is None

    def test_shallow_dip_below_min_depth_is_not_an_episode(self):
        """A 5% wobble is not a decision point when the shallowest tier is 10%."""
        vals = [100.0] + [95.0] * 5 + [105.0]
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert eps == []

    def test_monotonically_rising_series_has_no_episodes(self):
        vals = [100.0 + i for i in range(50)]

        assert find_ath_episodes(series(vals), thresholds=THRESHOLDS) == []

    def test_empty_series_returns_no_episodes(self):
        assert find_ath_episodes(pd.Series(dtype=float), thresholds=THRESHOLDS) == []

    def test_deeper_thresholds_recorded_only_when_reached(self):
        # Bottoms at -25%: crosses -10% and -20%, never -30%.
        vals = [100.0] + [75.0] * 5 + [105.0]
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        crossings = eps[0].first_crossings
        assert -0.10 in crossings
        assert -0.20 in crossings
        assert -0.30 not in crossings

    def test_recovery_requires_reaching_the_anchor_not_a_threshold(self):
        """Climbing back above the threshold does not end the episode."""
        # Dips to -30%, recovers to -5% (above every threshold), never reclaims 100.
        vals = [100.0] + [70.0] * 5 + [95.0] * 20
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)

        assert len(eps) == 1
        assert eps[0].is_censored


class TestEpisodeMetadata:
    def test_days_to_crossing_measured_from_anchor(self):
        vals = [100.0] + [70.0] * 5 + [105.0]
        ep = find_ath_episodes(series(vals), thresholds=THRESHOLDS)[0]

        days = ep.days_to_crossing(-0.20)
        assert days is not None and days >= 0

    def test_days_to_crossing_is_none_when_not_crossed(self):
        vals = [100.0] + [75.0] * 5 + [105.0]
        ep = find_ath_episodes(series(vals), thresholds=THRESHOLDS)[0]

        assert ep.days_to_crossing(-0.30) is None

    def test_episodes_to_dataframe_shape(self):
        vals = [100.0] + [70.0] * 5 + [105.0] * 3 + [70.0] * 5 + [120.0]
        eps = find_ath_episodes(series(vals), thresholds=THRESHOLDS)
        df = episodes_to_dataframe(eps)

        assert len(df) == len(eps)
        assert "Trough drawdown" in df.columns
        assert "Recovered" in df.columns


class TestStudyEpisode:
    def test_returns_none_when_threshold_not_crossed(self):
        vals = [100.0] + [75.0] * 30 + [105.0]
        ep = find_ath_episodes(series(vals), thresholds=THRESHOLDS)[0]

        study = study_episode(
            episode=ep,
            threshold=-0.30,          # never reached
            instrument=frame(vals),
            benchmark=frame(vals),
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.10, 1.00)],
            horizon_months=(12,),
        )

        assert study is None

    def test_all_policies_receive_identical_contributions(self):
        """Paired comparison: same external cash flows for every policy."""
        vals = [100.0] + [70.0] * 60 + [105.0] * 20
        ep = find_ath_episodes(series(vals), thresholds=THRESHOLDS)[0]

        study = study_episode(
            episode=ep,
            threshold=-0.20,
            instrument=frame(vals),
            benchmark=frame(vals),
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.10, 0.5), DeploymentTier(-0.20, 1.0)],
            horizon_months=(3,),
        )

        assert study is not None
        by_horizon: dict[str, set[float]] = {}
        for o in study.outcomes:
            by_horizon.setdefault(o.horizon_label, set()).add(o.total_contributions)
        for label, totals in by_horizon.items():
            assert len(totals) == 1, f"contributions diverged at {label}: {totals}"

    def test_study_reports_every_policy(self):
        vals = [100.0] + [70.0] * 60 + [105.0] * 20
        ep = find_ath_episodes(series(vals), thresholds=THRESHOLDS)[0]

        study = study_episode(
            episode=ep,
            threshold=-0.20,
            instrument=frame(vals),
            benchmark=frame(vals),
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.20, 1.00)],
            horizon_months=(3,),
        )

        assert study is not None
        policies = {o.policy for o in study.outcomes}
        assert policies == {"Savings only", "DCA", "ATH all-in", "ATH tiered"}

    def test_savings_only_never_deploys(self):
        vals = [100.0] + [70.0] * 60 + [105.0] * 20
        ep = find_ath_episodes(series(vals), thresholds=THRESHOLDS)[0]

        study = study_episode(
            episode=ep,
            threshold=-0.20,
            instrument=frame(vals),
            benchmark=frame(vals),
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.20, 1.00)],
            horizon_months=(3,),
        )

        assert study is not None
        savings = [o for o in study.outcomes if o.policy == "Savings only"]
        assert savings and all(o.n_deployments == 0 for o in savings)

    def test_deploying_into_a_recovered_episode_beats_savings(self):
        """V-shaped recovery above the anchor: buying the dip must win."""
        vals = [100.0] + [70.0] * 40 + [130.0] * 40
        ep = find_ath_episodes(series(vals), thresholds=THRESHOLDS)[0]

        study = study_episode(
            episode=ep,
            threshold=-0.20,
            instrument=frame(vals),
            benchmark=frame(vals),
            monthly_contribution=1.0,          # isolate the reserve decision
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.20, 1.00)],
            horizon_months=(6,),
        )

        assert study is not None
        all_in = study.wealth("ATH all-in", "6m")
        savings = study.wealth("Savings only", "6m")
        assert all_in is not None and savings is not None
        assert all_in > savings


class TestEventStudy:
    def test_event_study_produces_one_record_per_crossed_threshold(self):
        # Bottoms at -25%: crosses -10% and -20%, stops short of -30%.
        vals = [100.0] + [75.0] * 60 + [105.0] * 40
        inst = frame(vals)

        studies = run_event_study(
            instrument=inst,
            benchmark=inst,
            tiers=[DeploymentTier(-0.10, 0.5), DeploymentTier(-0.20, 1.0)],
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            thresholds=THRESHOLDS,
            horizon_months=(12,),
        )

        # -30% is never reached, so only -10% and -20% yield studies.
        assert {s.threshold for s in studies} == {-0.10, -0.20}

    def test_event_study_requires_adj_close(self):
        bad = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=2))

        with pytest.raises(ValueError, match="adj_close"):
            run_event_study(
                instrument=bad,
                benchmark=bad,
                tiers=[DeploymentTier(-0.10, 1.00)],
            )

    def test_baseline_cache_does_not_change_results(self):
        """The savings/DCA memo must be a pure speed-up, not a behaviour change.

        Guards against the cache leaking one threshold's baseline into another.
        """
        vals = [100.0] + [70.0] * 60 + [130.0] * 60
        inst = frame(vals)
        kw = dict(
            instrument=inst,
            benchmark=inst,
            tiers=[DeploymentTier(-0.10, 0.5), DeploymentTier(-0.20, 1.0)],
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            thresholds=THRESHOLDS,
            horizon_months=(12,),
        )

        cached = run_event_study(**kw)

        # Recompute each study with no cache at all.
        episodes = find_ath_episodes(inst["adj_close"], thresholds=THRESHOLDS)
        uncached = []
        for ep in episodes:
            for th in THRESHOLDS:
                s = study_episode(
                    episode=ep,
                    threshold=th,
                    instrument=inst,
                    benchmark=inst,
                    monthly_contribution=1000.0,
                    opening_reserve=10_000.0,
                    tiers=kw["tiers"],
                    horizon_months=(12,),
                    baseline_cache=None,
                )
                if s is not None:
                    uncached.append(s)

        assert len(cached) == len(uncached)
        for a, b in zip(cached, uncached, strict=True):
            assert a.threshold == b.threshold
            wa = {(o.policy, o.horizon_label): o.ending_wealth for o in a.outcomes}
            wb = {(o.policy, o.horizon_label): o.ending_wealth for o in b.outcomes}
            assert wa == wb

    def test_never_firing_policy_is_reported_as_not_traded(self):
        """A threshold the episode never reaches must not look like a skilful win.

        Savings-only beats DCA in any falling market, so a policy that never fires
        can post a large "ahead of DCA" figure while doing nothing. The
        pct_episodes_deployed column is what separates the two.
        """
        # Bottoms at -25%, so a -20% tier fires but the study also covers -10%.
        vals = [100.0] + [75.0] * 60 + [105.0] * 30
        inst = frame(vals)

        studies = run_event_study(
            instrument=inst,
            benchmark=inst,
            tiers=[DeploymentTier(-0.20, 1.00)],
            monthly_contribution=1.0,
            opening_reserve=10_000.0,
            thresholds=(-0.20,),
            horizon_months=(6,),
        )
        fired = summarise_threshold(studies, -0.20, "6m", "ATH all-in")

        assert fired["pct_episodes_deployed"] == 1.0
        # Having deployed, little cash should remain idle.
        assert fired["median_undeployed_cash"] is not None
        assert fired["median_undeployed_cash"] < 0.5

    def test_savings_only_policy_reports_no_trades_and_all_cash_idle(self):
        vals = [100.0] + [75.0] * 60 + [105.0] * 30
        inst = frame(vals)

        studies = run_event_study(
            instrument=inst,
            benchmark=inst,
            tiers=[DeploymentTier(-0.20, 1.00)],
            monthly_contribution=1.0,
            opening_reserve=10_000.0,
            thresholds=(-0.20,),
            horizon_months=(6,),
        )
        summary = summarise_threshold(studies, -0.20, "6m", "Savings only")

        assert summary["pct_episodes_deployed"] == 0.0
        assert summary["median_undeployed_cash"] == pytest.approx(1.0, abs=1e-6)

    def test_summary_reports_zero_episodes_cleanly(self):
        summary = summarise_threshold([], -0.20, "12m")

        assert summary["episodes"] == 0
        assert summary["median_vs_dca"] is None

    def test_summary_counts_episodes_and_percentiles(self):
        vals = [100.0] + [70.0] * 60 + [130.0] * 60
        inst = frame(vals)

        studies = run_event_study(
            instrument=inst,
            benchmark=inst,
            tiers=[DeploymentTier(-0.20, 1.00)],
            monthly_contribution=1.0,
            opening_reserve=10_000.0,
            thresholds=(-0.20,),
            horizon_months=(6,),
        )
        summary = summarise_threshold(studies, -0.20, "6m")

        assert summary["episodes"] == 1
        assert summary["median_vs_dca"] is not None
        assert summary["pct_ahead_of_dca"] is not None
