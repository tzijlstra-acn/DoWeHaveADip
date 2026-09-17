"""Phase 2 regression tests — documents confirmed defects and verifies fixes.

Tests 1-4:   Pending order bugs in study_episode with anchor="execution"
Tests 5-6:   Continuous saver model (new run_continuous_comparison API)
Tests 7-8:   Nasdaq 2020 arithmetic validation
Test 9:      Per-episode threshold selector helper
Test 10:     Winning episodes function
Test 11:     Bootstrap executed-only filter
Test 12:     Failed episodes reported in run_event_study
Test 13:     Displayed drawdown equals engine drawdown
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from dipdca.models import DeploymentTier, SimulationParams
from dipdca.quant.ath_episodes import (
    ATHEpisode,
    EpisodeStudy,
    PolicyOutcome,
    find_ath_episodes,
    run_event_study,
    study_episode,
)
from dipdca.quant.drawdown import drawdown

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bm_frame(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


def bm_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, name="adj_close")


def _make_episode_study(
    n_deployments: int,
    rel_return: float,
    ath_date_str: str = "2020-01-01",
) -> EpisodeStudy:
    """Build a minimal EpisodeStudy with specified deployment count and relative return.

    DCA wealth is fixed at 1000. ATH all-in wealth = 1000 * (1 + rel_return).
    """
    dca_wealth = 1000.0
    all_in_wealth = dca_wealth * (1.0 + rel_return)

    ath_date = pd.Timestamp(ath_date_str)
    signal_date = ath_date + pd.Timedelta(days=5)

    episode = ATHEpisode(
        ath_date=ath_date,
        ath_level=100.0,
        trough_date=signal_date,
        trough_drawdown=-0.16,
        recovery_date=None,
        first_crossings={-0.15: signal_date},
    )

    outcomes = [
        PolicyOutcome(
            policy="DCA",
            horizon_label="12m",
            ending_wealth=dca_wealth,
            total_contributions=1000.0,
            n_deployments=0,
            ending_cash=0.0,
        ),
        PolicyOutcome(
            policy="ATH all-in",
            horizon_label="12m",
            ending_wealth=all_in_wealth,
            total_contributions=1000.0,
            n_deployments=n_deployments,
            ending_cash=0.0,
        ),
    ]

    return EpisodeStudy(
        episode=episode,
        threshold=-0.15,
        signal_date=signal_date,
        outcomes=outcomes,
    )


# ===========================================================================
# Tests 1-4: Pending order bugs in study_episode with anchor="execution"
# ===========================================================================


class TestExecutionAnchorPendingOrder:
    """anchor='execution' must capture the threshold crossing from the signal date.

    Bug: study_episode starts the simulation at the execution date (T+1). If the
    benchmark has rebounded above the threshold between T and T+1, no pending order
    fires, and n_deployments=0 even though the threshold was clearly crossed.

    Fix: when anchor='execution', sim_start=signal_date (so the crossing is captured)
    but anchor_date=execution_date (for measurement windows).
    """

    def _make_series_with_dip(
        self,
        n_days: int = 200,
        ath_days: int = 20,
        dip_value: float = 84.0,
        rebound_value: float = 95.0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build a price series: ATH for ath_days, dips to dip_value on day ath_days,
        then rebounds to rebound_value for the rest.

        Returns (DataFrame with adj_close, Series for find_ath_episodes).
        """
        values = [100.0] * ath_days + [dip_value] + [rebound_value] * (n_days - ath_days - 1)
        idx = pd.bdate_range(start="2020-01-02", periods=n_days)
        df = pd.DataFrame({"adj_close": values, "close": values}, index=idx)
        series = pd.Series(values, index=idx, name="adj_close")
        return df, series

    def test_execution_anchor_carries_pending_order_from_signal(self):
        """Threshold crossing at day T must still deploy at day T+1 execution.

        Setup:
        - Days 0-19: benchmark at 100 (ATH)
        - Day 20:    benchmark drops to 84 (drawdown = -16%, crosses -15%)
        - Days 21+:  benchmark at 95 (rebounded above -15% threshold but below ATH)

        With anchor='execution', sim_start must be signal_date (day 20) so the
        pending order fires. Previously sim_start=execution_date (day 21) which
        saw bm=95 = -5% drawdown → no order fires → n_deployments=0 (WRONG).
        """
        df, series = self._make_series_with_dip(
            n_days=200, ath_days=20, dip_value=84.0, rebound_value=95.0
        )

        episodes = find_ath_episodes(series, thresholds=(-0.15,))
        assert episodes, "Expected at least one episode with -15% crossing"
        ep = episodes[0]
        assert ep.crossed(-0.15), "Episode must have crossed -15%"

        tiers = [DeploymentTier(-0.15, 1.00)]
        result = study_episode(
            episode=ep,
            threshold=-0.15,
            instrument=df,
            benchmark=df,
            monthly_contribution=1000.0,
            opening_reserve=12_000.0,
            tiers=tiers,
            horizon_months=(6,),
            anchor="execution",
        )

        assert result is not None, "study_episode returned None — no study produced"

        ath_outcome = next(
            (o for o in result.outcomes if o.policy == "ATH all-in"),
            None,
        )
        assert ath_outcome is not None, "No 'ATH all-in' outcome found"

        assert ath_outcome.n_deployments == 1, (
            f"Expected n_deployments=1 (pending order from signal day must carry to "
            f"execution day), got {ath_outcome.n_deployments}. "
            "Bug: sim_start was set to execution_date, missing the threshold crossing."
        )

    def test_rebound_after_signal_does_not_cancel_execution(self):
        """Even if the benchmark rebounds to a new ATH on execution day, the order fires.

        Setup:
        - Days 0-19: benchmark at 100 (ATH)
        - Day 20:    benchmark drops to 84 (crosses -15%)
        - Days 21+:  benchmark at 102 (new ATH — episode closes, but first episode
                     must still have deployed on day 21)
        """
        values = [100.0] * 20 + [84.0] + [102.0] * 179
        n_days = len(values)
        idx = pd.bdate_range(start="2020-01-02", periods=n_days)
        df = pd.DataFrame({"adj_close": values, "close": values}, index=idx)
        series = pd.Series(values, index=idx, name="adj_close")

        episodes = find_ath_episodes(series, thresholds=(-0.15,))
        assert episodes, "Expected at least one episode"
        ep = episodes[0]
        assert ep.crossed(-0.15), "Episode must have crossed -15%"

        tiers = [DeploymentTier(-0.15, 1.00)]
        result = study_episode(
            episode=ep,
            threshold=-0.15,
            instrument=df,
            benchmark=df,
            monthly_contribution=1000.0,
            opening_reserve=12_000.0,
            tiers=tiers,
            horizon_months=(6,),
            anchor="execution",
        )

        assert result is not None, "study_episode returned None"

        ath_outcomes = [o for o in result.outcomes if o.policy == "ATH all-in"]
        assert ath_outcomes, "No 'ATH all-in' outcomes found"

        total_deployments = sum(o.n_deployments for o in ath_outcomes)
        assert total_deployments >= 1, (
            f"Expected at least 1 deployment across all horizons, got {total_deployments}. "
            "The rebound to new ATH on execution day must not cancel the pending order."
        )

    def test_all_in_crossing_episode_has_exactly_one_deployment(self):
        """run_event_study with anchor='execution' — each crossed episode deploys once.

        For a dataset with a clear -15% crossing followed by slow rebound
        (benchmark stays below ATH for 100+ days), the ATH all-in strategy
        must show n_deployments=1 for each episode that crossed the threshold.
        """
        # 20 days ATH, 1 day dip to 84, 180 days at 95 (never recovers ATH during window)
        values = [100.0] * 20 + [84.0] + [95.0] * 179
        n_days = len(values)
        idx = pd.bdate_range(start="2020-01-02", periods=n_days)
        df = pd.DataFrame({"adj_close": values, "close": values}, index=idx)

        tiers = [DeploymentTier(-0.15, 1.00)]
        studies = run_event_study(
            instrument=df,
            benchmark=df,
            tiers=tiers,
            monthly_contribution=1000.0,
            opening_reserve=12_000.0,
            thresholds=(-0.15,),
            horizon_months=(6,),
            anchor="execution",
        ).studies

        assert studies, "Expected at least one study"

        for s in studies:
            for o in s.outcomes:
                if o.policy == "ATH all-in":
                    assert o.n_deployments == 1, (
                        f"Episode at {s.episode.ath_date.date()}: expected n_deployments=1 "
                        f"for ATH all-in at {o.horizon_label}, got {o.n_deployments}."
                    )

    def test_execution_anchor_does_not_move_trade_to_second_crossing(self):
        """First crossing at T should deploy at T+1, within a 1-month window.

        Setup:
        - Days 0-9:   benchmark at 100 (ATH)
        - Day 10:     benchmark drops to 84 (first -15% crossing, signal day T)
        - Days 11-50: benchmark at 90 (above -15%=85, but below ATH=100, ~40 trading days)
        - Day 51+:    benchmark drops to 82 (second crossing below -15%)

        Horizon: 1 month from anchor_date (day 11). The 1-month window ends around
        day 32, which is BEFORE the second crossing at day 51.

        With anchor='execution' and sim_start=signal_date (fix):
        - Pending order fires at day 11 (T+1) → n_deployments=1 within 1-month window

        With anchor='execution' and sim_start=execution_date (bug):
        - Simulation starts at day 11, bm=90 > -15% → no pending order
        - Second crossing at day 51 is OUTSIDE the 1-month window
        - → n_deployments=0 in the 1-month window (WRONG)
        """
        # 10 ATH days + 1 signal day + 40 rebound days + 149 second-crossing days = 200
        values = (
            [100.0] * 10       # days 0-9: ATH
            + [84.0]           # day 10: first crossing (-16%)
            + [90.0] * 40      # days 11-50: rebound above -15% but below ATH
            + [82.0] * 149     # days 51+: second crossing below -15%
        )
        n_days = len(values)
        idx = pd.bdate_range(start="2020-01-02", periods=n_days)
        df = pd.DataFrame({"adj_close": values, "close": values}, index=idx)
        series = pd.Series(values, index=idx, name="adj_close")

        episodes = find_ath_episodes(series, thresholds=(-0.15,))
        assert episodes, "Expected at least one episode"
        ep = episodes[0]
        assert ep.crossed(-0.15), "Episode must have crossed -15%"

        tiers = [DeploymentTier(-0.15, 1.00)]
        result = study_episode(
            episode=ep,
            threshold=-0.15,
            instrument=df,
            benchmark=df,
            monthly_contribution=1000.0,
            opening_reserve=12_000.0,
            tiers=tiers,
            horizon_months=(1,),  # 1 month: ends before second crossing at day 51
            anchor="execution",
        )

        assert result is not None, "study_episode returned None"

        # With the fix (sim_start=signal_date), the pending order fires at T+1
        # and n_deployments=1 within the 1-month window (which ends ~day 32).
        # With the bug (sim_start=execution_date=day11), bm=90 > threshold, no order
        # fires, and the second crossing at day 51 is outside the 1-month window
        # → n_deployments=0.
        ath_outcomes = [o for o in result.outcomes if o.policy == "ATH all-in"]
        assert ath_outcomes, "No 'ATH all-in' outcomes found"

        for o in ath_outcomes:
            assert o.n_deployments == 1, (
                f"Expected n_deployments=1 at horizon {o.horizon_label} "
                f"(deploy at first execution T+1), got {o.n_deployments}. "
                "Bug: sim_start=execution_date sees bm=90 (above -15%), "
                "waits, second crossing at day 51 is outside the 1-month window "
                "→ n_deployments=0. Fix: set sim_start=signal_date."
            )


# ===========================================================================
# Tests 5-6: Continuous saver model
# ===========================================================================


class TestContinuousComparison:
    """Tests for the new run_continuous_comparison() function.

    These FAIL today with ImportError. After fix: create
    src/dipdca/quant/continuous_comparison.py.
    """

    def _make_monthly_prices(
        self,
        month_end_prices: list[float],
        start_year: int = 2020,
        start_month: int = 1,
    ) -> pd.DataFrame:
        """Build a simple price DataFrame with one bar per month."""
        dates = []
        for i, _ in enumerate(month_end_prices):
            m = (start_month - 1 + i) % 12 + 1
            y = start_year + (start_month - 1 + i) // 12
            # Use the last day of the month as the date
            import calendar
            last_day = calendar.monthrange(y, m)[1]
            dates.append(pd.Timestamp(f"{y}-{m:02d}-{last_day}"))
        values = month_end_prices
        df = pd.DataFrame(
            {"adj_close": values, "close": values},
            index=pd.DatetimeIndex(dates),
        )
        return df

    def test_continuous_dca_includes_months_before_threshold(self):
        """DCA buys every month-end; total units > 4 * (1000/100).

        Setup (6 months):
        - Month 1-4: price = 100 (DCA buys here)
        - Month 5:   price = 84  (threshold crossing -16%)
        - Month 6:   price = 90  (rebound)

        DCA: buys at 100, 100, 100, 100, 84, 90
        Total DCA units > 4 * (1000/100) = 40 (it also buys at 84 and 90)
        """
        from dipdca.quant.continuous_comparison import run_continuous_comparison

        prices = [100.0, 100.0, 100.0, 100.0, 84.0, 90.0]
        df = self._make_monthly_prices(prices)

        result = run_continuous_comparison(
            instrument=df,
            benchmark=df,
            monthly_contribution=1000.0,
            threshold=-0.15,
        )

        assert result is not None, (
            "run_continuous_comparison returned None — threshold must be crossed in this dataset"
        )

        min_expected_units = 4 * (1000.0 / 100.0)
        assert result.dca_total_units > min_expected_units, (
            f"Expected dca_total_units > {min_expected_units} (4 months at 100), "
            f"got {result.dca_total_units}. DCA must also buy in months 5 and 6."
        )

    def test_continuous_dip_policy_accumulates_same_monthly_flows(self):
        """Dip strategy accumulates 4 months of savings then deploys at threshold.

        Setup: same 6-month series.
        Dip: accumulate 1000/month for 4 months, deploy 4000 at threshold in month 5.
        Assert: result.dip_deployed_at_signal == 4000
        """
        from dipdca.quant.continuous_comparison import run_continuous_comparison

        prices = [100.0, 100.0, 100.0, 100.0, 84.0, 90.0]
        df = self._make_monthly_prices(prices)

        result = run_continuous_comparison(
            instrument=df,
            benchmark=df,
            monthly_contribution=1000.0,
            threshold=-0.15,
        )

        assert result is not None, "run_continuous_comparison returned None"

        assert result.dip_deployed_at_signal == pytest.approx(4000.0, rel=0.01), (
            f"Expected dip_deployed_at_signal=4000 (4 months of 1000 accumulated), "
            f"got {result.dip_deployed_at_signal}."
        )


# ===========================================================================
# Tests 7-8: Nasdaq 2020 arithmetic validation
# ===========================================================================


class TestNasdaq2020Arithmetic:
    """Validate the Nasdaq-100 2020 crash arithmetic from the audit.

    These use a synthetic price series matching the specific dates and prices
    cited in the audit report. Both FAIL today with ImportError.
    """

    def _make_nasdaq_fixture(self) -> pd.DataFrame:
        """Build the minimal Nasdaq-100 fixture from the audit."""
        # Key dates and prices from the audit
        dates_prices = [
            ("2020-01-31", 8991.51),   # month-end Jan: DCA buys here
            ("2020-02-19", 9718.73),   # ATH (mid-month — DCA doesn't buy)
            ("2020-02-28", 8461.83),   # month-end Feb: DCA buys here
            ("2020-03-09", 7948.03),   # first -15% crossing below ATH 9718.73
            ("2020-03-10", 8372.27),   # execution price for -15% threshold
        ]
        dates = pd.DatetimeIndex([d for d, _ in dates_prices])
        prices = [p for _, p in dates_prices]
        df = pd.DataFrame({"adj_close": prices, "close": prices}, index=dates)
        return df

    def _make_nasdaq_fixture_25pct(self) -> pd.DataFrame:
        """Nasdaq fixture with additional -25% crossing."""
        dates_prices = [
            ("2020-01-31", 8991.51),   # month-end Jan: DCA buys
            ("2020-02-19", 9718.73),   # ATH
            ("2020-02-28", 8461.83),   # month-end Feb: DCA buys
            ("2020-03-09", 7948.03),   # first -15% crossing
            ("2020-03-10", 8372.27),   # execution at -15%
            ("2020-03-12", 7263.65),   # first -25% crossing
            ("2020-03-13", 7995.26),   # execution at -25%
        ]
        dates = pd.DatetimeIndex([d for d, _ in dates_prices])
        prices = [p for _, p in dates_prices]
        df = pd.DataFrame({"adj_close": prices, "close": prices}, index=dates)
        return df

    def test_nasdaq_2020_15_percent_case_buys_more_units_than_dca(self):
        """Dip at -15% buys ~4% more units than month-end DCA.

        Audit arithmetic:
        - DCA units: 1000/8991.51 + 1000/8461.83 = ~0.22939
        - Dip units: 2000/8372.27 = ~0.23889
        - Unit advantage: ~4.14%
        """
        from dipdca.quant.continuous_comparison import run_continuous_comparison

        df = self._make_nasdaq_fixture()

        result = run_continuous_comparison(
            instrument=df,
            benchmark=df,
            monthly_contribution=1000.0,
            threshold=-0.15,
            initial_ath=9718.73,
        )

        assert result is not None, (
            "run_continuous_comparison returned None — -15% threshold must be crossed"
        )

        unit_advantage = result.dip_total_units / result.dca_total_units - 1.0
        assert unit_advantage > 0.04, (
            f"Expected unit advantage > 4% (audit shows ~4.14%), "
            f"got {unit_advantage:.2%}. "
            f"DCA units: {result.dca_total_units:.5f}, "
            f"Dip units: {result.dip_total_units:.5f}"
        )

    def test_nasdaq_2020_25_percent_case_buys_more_units_than_dca(self):
        """Dip at -25% buys ~9% more units than month-end DCA.

        Audit arithmetic:
        - DCA units: 1000/8991.51 + 1000/8461.83 = ~0.22939
        - Dip units: 2000/7995.26 = ~0.25016
        - Unit advantage: ~9.05%
        """
        from dipdca.quant.continuous_comparison import run_continuous_comparison

        df = self._make_nasdaq_fixture_25pct()

        result = run_continuous_comparison(
            instrument=df,
            benchmark=df,
            monthly_contribution=1000.0,
            threshold=-0.25,
            initial_ath=9718.73,
        )

        assert result is not None, (
            "run_continuous_comparison returned None — -25% threshold must be crossed"
        )

        unit_advantage = result.dip_total_units / result.dca_total_units - 1.0
        assert unit_advantage > 0.09, (
            f"Expected unit advantage > 9% (audit shows ~9.05%), "
            f"got {unit_advantage:.2%}. "
            f"DCA units: {result.dca_total_units:.5f}, "
            f"Dip units: {result.dip_total_units:.5f}"
        )


# ===========================================================================
# Test 9: Per-episode threshold selector
# ===========================================================================


class TestEpisodesForThreshold:
    """Test the new episodes_for_threshold() helper.

    FAILS today with ImportError. After fix: add to ath_episodes.py.
    """

    def test_episode_detail_can_show_every_selected_threshold(self):
        """episodes_for_threshold(studies, threshold) returns only matching studies."""
        from dipdca.quant.ath_episodes import episodes_for_threshold

        # Build a small set of studies with two different thresholds
        studies_15 = [_make_episode_study(1, 0.05, f"2020-0{i+1}-01") for i in range(3)]
        studies_20 = [_make_episode_study(1, 0.03, f"2021-0{i+1}-01") for i in range(2)]

        # Patch the threshold on studies_20 to -0.20
        patched_20 = []
        for s in studies_20:
            ep20 = ATHEpisode(
                ath_date=s.episode.ath_date,
                ath_level=100.0,
                trough_date=s.episode.trough_date,
                trough_drawdown=-0.21,
                recovery_date=None,
                first_crossings={-0.20: s.episode.trough_date},
            )
            patched_20.append(EpisodeStudy(
                episode=ep20,
                threshold=-0.20,
                signal_date=s.signal_date,
                outcomes=s.outcomes,
            ))

        all_studies = studies_15 + patched_20

        result_15 = episodes_for_threshold(all_studies, -0.15)
        result_20 = episodes_for_threshold(all_studies, -0.20)

        assert len(result_15) == 3, (
            f"Expected 3 studies for -15%, got {len(result_15)}"
        )
        assert len(result_20) == 2, (
            f"Expected 2 studies for -20%, got {len(result_20)}"
        )
        assert all(s.threshold == -0.15 for s in result_15)
        assert all(s.threshold == -0.20 for s in result_20)


# ===========================================================================
# Test 10: Winning episodes function
# ===========================================================================


class TestWinningEpisodes:
    """Test the new winning_episodes() helper.

    FAILS today with ImportError. After fix: add to ath_episodes.py.
    """

    def test_winning_episode_table_contains_positive_executed_cases(self):
        """winning_episodes() returns only episodes where policy beat DCA AND deployed.

        Setup:
        - 3 episodes with rel_return > 0 and n_deployments=1 (winners)
        - 1 episode with rel_return < 0 and n_deployments=1 (loser)
        - 1 episode with rel_return > 0 but n_deployments=0 (undeployed — exclude)

        winning_episodes must return exactly 3 results.
        """
        from dipdca.quant.ath_episodes import winning_episodes

        winning = [_make_episode_study(1, 0.05, f"2020-0{i+1}-01") for i in range(3)]
        loser = [_make_episode_study(1, -0.03, "2020-04-01")]
        undeployed = [_make_episode_study(0, 0.05, "2020-05-01")]

        all_studies = winning + loser + undeployed

        result = winning_episodes(all_studies, threshold=-0.15, horizon_label="12m")

        assert len(result) >= 3, (
            f"Expected at least 3 winning episodes, got {len(result)}. "
            "Loser and undeployed episodes must be excluded."
        )
        # Undeployed should not appear
        assert len(result) <= 3, (
            f"Expected exactly 3 (not 5), got {len(result)}. "
            "Loser (rel_return<0) and undeployed (n_deployments=0) must be excluded."
        )


# ===========================================================================
# Test 11: Bootstrap executed-only filter
# ===========================================================================


class TestBootstrapExecutedOnly:
    """Bootstrap with executed_only=True must exclude zero-deployment episodes.

    FAILS today: executed_only parameter doesn't exist in _filter_eligible.
    """

    def test_bootstrap_executed_only_excludes_zero_deployment(self):
        """executed_only=True excludes episodes where ATH all-in never deployed."""
        from dipdca.quant.episode_bootstrap import bootstrap_win_rate

        # 1 episode with n_deployments=0 (strategy never fired)
        zero_dep = _make_episode_study(0, 0.05, "2019-01-01")

        # 5 episodes that deployed and outperformed DCA
        deployed = [_make_episode_study(1, 0.05, f"2020-0{i+1}-01") for i in range(5)]

        all_studies = [zero_dep] + deployed

        result = bootstrap_win_rate(
            all_studies,
            threshold=-0.15,
            horizon_label="12m",
            policy="ATH all-in",
            executed_only=True,
            seed=42,
        )

        assert result is not None, "bootstrap_win_rate returned None — need ≥2 eligible episodes"

        assert result.n_episodes == 5, (
            f"Expected n_episodes=5 (zero-deployment excluded), got {result.n_episodes}. "
            "executed_only=True must filter out episodes with n_deployments=0."
        )


# ===========================================================================
# Test 12: Failed episodes reported
# ===========================================================================


class TestFailedEpisodesReported:
    """run_event_study must report failed episodes, not silently discard them.

    FAILS today: run_event_study returns list[EpisodeStudy] with no failures field.
    After fix: returns EventStudyResult(studies, failures).
    """

    def test_failed_episode_is_reported_not_discarded(self):
        """When study_episode raises RuntimeError, the failure is returned in result.failures."""
        from dipdca.quant.ath_episodes import EventStudyResult

        # 100 days: 10 ATH, 1 dip to 84, rest at 95
        values = [100.0] * 10 + [84.0] + [95.0] * 89
        n_days = len(values)
        idx = pd.bdate_range(start="2020-01-02", periods=n_days)
        df = pd.DataFrame({"adj_close": values, "close": values}, index=idx)

        call_count = {"n": 0}
        original_study_episode = __import__(
            "dipdca.quant.ath_episodes", fromlist=["study_episode"]
        ).study_episode

        def patched_study_episode(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated study failure")
            return original_study_episode(*args, **kwargs)

        tiers = [DeploymentTier(-0.15, 1.00)]

        with patch(
            "dipdca.quant.ath_episodes.study_episode",
            side_effect=patched_study_episode,
        ):
            result = run_event_study(
                instrument=df,
                benchmark=df,
                tiers=tiers,
                monthly_contribution=1000.0,
                opening_reserve=12_000.0,
                thresholds=(-0.15,),
                horizon_months=(6,),
            )

        # After fix: result must be an EventStudyResult with .studies and .failures
        assert isinstance(result, EventStudyResult), (
            f"Expected EventStudyResult, got {type(result).__name__}. "
            "run_event_study must return EventStudyResult(studies, failures)."
        )
        assert hasattr(result, "studies"), "EventStudyResult must have 'studies' attribute"
        assert hasattr(result, "failures"), "EventStudyResult must have 'failures' attribute"
        assert len(result.failures) >= 1, (
            f"Expected at least 1 failure (from patched study_episode), "
            f"got {len(result.failures)}."
        )


# ===========================================================================
# Test 13: Displayed drawdown equals engine drawdown
# ===========================================================================


class TestDisplayedDrawdownMatchesEngine:
    """The displayed drawdown must match the engine's drawdown calculation.

    Bug: when a seeded initial_ath is higher than anything in the series window,
    the engine uses initial_ath for drawdown (correct), but the displayed drawdown
    uses drawdown(series) which computes its own running peak — giving a different
    (wrong) value.

    Fix: add seeded_drawdown(series, initial_ath) to drawdown.py and use it in
    the display layer. This test verifies that seeded_drawdown agrees with the
    engine's drawdown calculation.
    """

    def test_displayed_drawdown_equals_engine_drawdown(self):
        """seeded_drawdown(series, initial_ath) must match the engine's drawdown.

        Setup:
        - Series: 200 days, monotonically increasing from 90 to ~91.99
        - Seeded ATH = 110.0 (higher than anything in the series)

        Raw drawdown(series).iloc[-1] = 0.0 (series is at all-time-high at end,
        but only within the window — doesn't know about the seeded 110 ATH).

        Engine drawdown: last_price / 110.0 - 1 ≈ -16.4% (using seeded ATH).

        seeded_drawdown(series, 110.0).iloc[-1] must agree with the engine.

        FAILS before fix: seeded_drawdown doesn't exist in drawdown.py.
        PASSES after fix: seeded_drawdown is implemented and gives the correct result.
        """
        from dipdca.quant.backtest import run_ath_deployment
        from dipdca.quant.drawdown import seeded_drawdown  # FAILS before fix

        n = 200
        idx = pd.bdate_range(start="2020-01-02", periods=n)
        # Monotonically increasing: 90.00, 90.01, ..., ~91.99
        price_values = [90.0 + i * 0.01 for i in range(n)]
        series = pd.Series(price_values, index=idx)
        df = pd.DataFrame({"adj_close": price_values, "close": price_values}, index=idx)

        # Seeded ATH from pre-window history: higher than anything in the series
        initial_ath = 110.0

        tiers = [DeploymentTier(-0.15, 1.00)]
        params = SimulationParams(
            monthly_contribution=1000.0,
            payday=25,
            contribution_timing="month_end",
            initial_investment=0.0,
            initial_cash_reserve=12_000.0,
            start_date=idx[0].date(),
            end_date=idx[-1].date(),
            dip_threshold=-0.15,
            fixed_fee=0.0,
            pct_fee=0.0,
            slippage=0.0,
            cash_rate_override=None,
        )

        _, ledger = run_ath_deployment(
            instrument_data=df,
            params=params,
            tiers=tiers,
            benchmark_data=df,
            initial_ath=initial_ath,
        )

        # Engine drawdown: uses seeded ATH=110
        last_price = float(ledger["price"].iloc[-1])
        engine_dd = last_price / initial_ath - 1.0

        # Raw drawdown (the bug): ignores seeded ATH
        raw_dd = float(drawdown(series).iloc[-1])
        assert raw_dd != pytest.approx(engine_dd, abs=0.001), (
            f"Raw drawdown ({raw_dd:.4f}) unexpectedly equals engine drawdown "
            f"({engine_dd:.4f}) — test setup may be wrong. "
            "The series must be monotonically increasing with seeded ATH above it."
        )

        # Fixed drawdown: seeded_drawdown knows about initial_ath
        fixed_dd = float(seeded_drawdown(series, initial_ath).iloc[-1])

        assert fixed_dd == pytest.approx(engine_dd, abs=0.001), (
            f"seeded_drawdown(..., {initial_ath}).iloc[-1] = {fixed_dd:.4f} "
            f"!= engine drawdown {engine_dd:.4f}. "
            "seeded_drawdown must use initial_ath as the starting peak so the "
            "display agrees with the engine's drawdown calculation."
        )
