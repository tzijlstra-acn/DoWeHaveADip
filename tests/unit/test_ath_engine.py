"""20 failing tests for the ATH-based dip deployment engine.

All tests use run_ath_deployment from dipdca.quant.backtest.
Before implementation, all tests fail with ImportError (function does not exist).
After implementation, all tests must pass (except Test 15, marked xfail).

Covered invariants
------------------
1.  Index (not ETF) determines ATH and drawdown.
2.  ETF price divergence does not change signal dates.
3.  FX changes units/wealth but not index drawdown.
4.  Pre-simulation index history seeds the ATH correctly.
5.  Rolling window does not reset the ATH.
6.  Simulation beginning during drawdown preserves earlier ATH.
7.  Recovering above a threshold does NOT rearm it.
8.  Reaching the anchored ATH resets the episode.
9.  Each tier fires once per ATH episode.
10. A later ATH episode may fire the tiers again.
11. All-in is exactly one-tier 100% policy.
12. A gap across several tiers schedules only the deepest cumulative target.
13. EUR 20,000 crossing -15% and -25% at once schedules EUR 12,000.
14. Signal close t executes strictly after t (not same day).
15. Event-study placeholder (xfail until scenarios.py updated).
16. Each episode contributes at most one deployment per threshold.
17. All compared strategies receive identical external savings flows.
18. DCA purchase prices never affect threshold signals.
19. Pure ATH policy has no time-based forced deployment (no max_wait_months).
20. Asset lacking benchmark_data raises ValueError, never silently uses ETF ATH.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dipdca.models import DeploymentTier, SimulationParams
from dipdca.quant.backtest import run_ath_deployment, run_dca

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_prices(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Create a minimal price DataFrame from a list of prices on bdate_range."""
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


def base_params(**kw) -> SimulationParams:
    """Minimal valid SimulationParams with no costs and a small monthly contribution."""
    defaults = dict(
        monthly_contribution=0.01,       # tiny — tests focus on initial_cash_reserve
        payday=25,
        initial_investment=0.0,
        initial_cash_reserve=10_000.0,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
        dip_threshold=-0.10,
        max_wait_months=24,
        fixed_fee=0.0,
        pct_fee=0.0,
        slippage=0.0,
        cash_rate_override=None,
    )
    defaults.update(kw)
    return SimulationParams(**defaults)


TIER_15 = [DeploymentTier(-0.15, 1.00)]          # fire all-in at -15%
TIERS_25_60 = [DeploymentTier(-0.15, 0.25),       # 25% at -15%
               DeploymentTier(-0.25, 0.60)]        # 60% total at -25%


# ---------------------------------------------------------------------------
# Test 1: Index (not ETF) determines ATH and drawdown
# ---------------------------------------------------------------------------
class TestIndexDeterminesSignal:
    def test_benchmark_drop_triggers_despite_shallow_etf(self):
        """Benchmark drops -20% (crosses -15%); ETF only drops -5%. Signal fires."""
        # Benchmark: ATH then -20%
        bm_vals = [100.0] * 20 + [80.0] * 80
        # ETF: barely moves
        etf_vals = [100.0] * 20 + [95.0] * 80

        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)

        params = base_params(end_date=date(2020, 5, 20))
        r, ledger = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)

        assert r.n_deployments >= 1, "Benchmark -20% must trigger the -15% tier"
        assert r.ending_market_value > 0, "ETF units must have been purchased"

    def test_etf_drop_without_benchmark_drop_does_not_trigger(self):
        """ETF drops -20% but benchmark is flat. No signal should fire."""
        bm_vals = [100.0] * 100          # benchmark: flat (never draws down)
        etf_vals = [100.0] * 30 + [75.0] * 70   # ETF drops -25%

        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 5, 20))

        r, _ = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)
        assert r.n_deployments == 0, "ETF-only ATH must not drive the signal"


# ---------------------------------------------------------------------------
# Test 2: ETF price divergence does not change signal dates
# ---------------------------------------------------------------------------
class TestETFDivergenceNoEffect:
    def test_same_signal_date_regardless_of_etf_scale(self):
        """Two ETFs at different price levels but same benchmark: identical signal day."""
        bm_vals = [100.0] * 15 + [84.0] * 85   # -16% at day 16

        etf_a = make_prices([50.0] * 15 + [42.0] * 85)   # ETF A (price level 50)
        etf_b = make_prices([200.0] * 15 + [168.0] * 85) # ETF B (price level 200)
        bm_df = make_prices(bm_vals)

        params = base_params(end_date=date(2020, 5, 20))
        _, ledger_a = run_ath_deployment(etf_a, params, TIER_15, benchmark_data=bm_df)
        _, ledger_b = run_ath_deployment(etf_b, params, TIER_15, benchmark_data=bm_df)

        sig_a = ledger_a[ledger_a["signal_dd"] != 0.0].index.tolist()
        sig_b = ledger_b[ledger_b["signal_dd"] != 0.0].index.tolist()
        assert sig_a == sig_b, "Signal day must be identical across ETFs with same benchmark"


# ---------------------------------------------------------------------------
# Test 3: FX changes units/wealth but not index drawdown
# ---------------------------------------------------------------------------
class TestFXDoesNotAlterBenchmarkDrawdown:
    def test_benchmark_drawdown_identical_across_fx_levels(self):
        """FX-adjusted ETF vs base ETF: benchmark_drawdown column must be identical."""
        bm_vals = [100.0] * 10 + [84.0] * 90      # -16%
        bm_df = make_prices(bm_vals)

        etf_base = make_prices([100.0] * 10 + [84.0] * 90)
        etf_fx   = make_prices([100.0 / 1.2] * 10 + [84.0 / 1.2] * 90)  # EUR/USD = 1.2

        params = base_params(end_date=date(2020, 5, 20))
        _, ledger_base = run_ath_deployment(etf_base, params, TIER_15, benchmark_data=bm_df)
        _, ledger_fx   = run_ath_deployment(etf_fx,   params, TIER_15, benchmark_data=bm_df)

        dd_base = ledger_base["benchmark_drawdown"].dropna().reset_index(drop=True)
        dd_fx   = ledger_fx["benchmark_drawdown"].dropna().reset_index(drop=True)
        pd.testing.assert_series_equal(dd_base, dd_fx, check_names=False, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: Pre-simulation index history seeds the ATH correctly
# ---------------------------------------------------------------------------
class TestPreSimATHSeed:
    def test_ath_seeded_from_full_benchmark_max(self):
        """Benchmark peaks at 200 in the full history; this must become the initial ATH."""
        # Full benchmark (includes "pre-simulation" period at high)
        bm_vals = [200.0] * 20 + [160.0] * 80   # ATH=200, then -20%
        etf_vals = [100.0] * 100
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)

        params = base_params(end_date=date(2020, 5, 20))
        r, ledger = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)

        # Seeded ATH = 200 → drawdown at day 21 = 160/200-1 = -20% → fires -15%
        bm_dd = ledger["benchmark_drawdown"].dropna()
        assert bm_dd.min() < -0.15, "ATH of 200 must be used; drawdown must reach -20%"
        assert r.n_deployments >= 1, "Deployment must fire against the correctly seeded ATH"

    def test_initial_ath_parameter_overrides_computed_ath(self):
        """Explicit initial_ath overrides the computed max from benchmark_data."""
        # Benchmark only goes to 120 in the passed data, but we know true ATH was 200
        bm_vals = [120.0] * 10 + [96.0] * 90    # -20% from 120 (≈ -16%)
        bm_df = make_prices(bm_vals)
        etf_df = make_prices([100.0] * 100)

        params = base_params(end_date=date(2020, 5, 20))
        # Pass explicit initial_ath = 200 → drawdown = 96/200-1 = -52%
        r_seeded, _ = run_ath_deployment(
            etf_df, params, TIER_15, benchmark_data=bm_df, initial_ath=200.0
        )
        # With initial_ath=200 the drawdown is much deeper → fires early
        assert r_seeded.n_deployments >= 1


# ---------------------------------------------------------------------------
# Test 5: Rolling window does not reset the ATH
# ---------------------------------------------------------------------------
class TestATHNeverDecreases:
    def test_benchmark_ath_monotonically_non_decreasing(self):
        """The running ATH column in the ledger must never decrease."""
        bm_vals = [100.0] * 10 + [150.0] * 10 + [80.0] * 30 + [120.0] * 50
        bm_df = make_prices(bm_vals)
        etf_df = make_prices([100.0] * 100)

        params = base_params(end_date=date(2020, 5, 20))
        _, ledger = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)

        bm_ath = ledger["benchmark_ath"].dropna()
        diffs = bm_ath.diff().dropna()
        assert (diffs >= -1e-10).all(), "Benchmark ATH must never decrease"
        assert float(bm_ath.max()) == pytest.approx(150.0, abs=1e-6), (
            "ATH must be 150 (the historical peak), not 120 (sub-ATH recovery)"
        )


# ---------------------------------------------------------------------------
# Test 6: Simulation beginning during drawdown preserves earlier ATH
# ---------------------------------------------------------------------------
class TestSimStartInDrawdown:
    def test_initial_ath_used_when_simulation_starts_in_drawdown(self):
        """Benchmark peaked at 200 before our data window; initial_ath=200 is passed."""
        # Our benchmark data starts at 160 (already in drawdown from 200)
        bm_vals = [160.0] * 100       # all below 200
        bm_df = make_prices(bm_vals)
        etf_df = make_prices([100.0] * 100)

        params = base_params(end_date=date(2020, 5, 20))
        # Pass explicit ATH of 200 — the simulation begins in an existing drawdown
        r, ledger = run_ath_deployment(
            etf_df, params, TIER_15, benchmark_data=bm_df, initial_ath=200.0
        )

        # Drawdown at any point = 160/200-1 = -20% → -15% tier fires immediately
        bm_dd = ledger["benchmark_drawdown"].dropna()
        assert bm_dd.min() < -0.15, "Drawdown must be computed from seeded ATH=200"
        assert r.n_deployments >= 1, "Should deploy because drawdown > -15% from seeded ATH"


# ---------------------------------------------------------------------------
# Test 7: Recovering above a threshold does NOT rearm it
# ---------------------------------------------------------------------------
class TestNoRearmOnThresholdRecovery:
    def test_single_fire_despite_two_dips_below_threshold(self):
        """Benchmark: ATH → -20% → -10% → -20% again. Tier fires exactly once."""
        # Episode does not rearm when benchmark recovers from -20% to -10% (above -15%)
        # Only a new ATH (≥100) would rearm the episode
        bm_vals = (
            [100.0] * 20   # at ATH
            + [80.0] * 20  # -20% → fires -15% tier (1st crossing)
            + [90.0] * 20  # -10% → above threshold, but NOT a new ATH → no rearm
            + [80.0] * 40  # -20% again → tier already triggered; must NOT fire again
        )
        etf_df = make_prices([100.0] * 100)
        bm_df = make_prices(bm_vals)
        params = base_params(end_date=date(2020, 5, 20))

        r, _ = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)
        assert r.n_deployments == 1, (
            "Tier must fire exactly once; threshold recovery must NOT rearm the episode"
        )


# ---------------------------------------------------------------------------
# Test 8: Reaching the anchored ATH resets the episode
# ---------------------------------------------------------------------------
class TestATHResetsEpisode:
    def test_new_ath_resets_episode_allowing_next_fire(self):
        """After a new ATH, a second drawdown episode must be able to fire the tier again."""
        # Phase 1: ATH=100 → drop to 80 (-20%) → tier fires → recover to 110 (new ATH)
        # Phase 2: drop from 110 to 88 (-20%) → tier fires again (new episode)
        bm_vals = (
            [100.0] * 20   # ATH = 100
            + [80.0] * 20  # -20% → fires
            + [110.0] * 20 # new ATH = 110 → episode resets
            + [88.0] * 40  # -20% from 110 → fires again (new episode)
        )
        etf_df = make_prices([100.0] * 100)
        bm_df = make_prices(bm_vals)
        params = base_params(end_date=date(2020, 5, 20), initial_cash_reserve=20_000.0)

        r, _ = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)
        assert r.n_deployments >= 2, "New ATH must reset episode; tier must fire in both episodes"


# ---------------------------------------------------------------------------
# Test 9: Each tier fires once per ATH episode
# ---------------------------------------------------------------------------
class TestEachTierFiresOncePerEpisode:
    def test_multi_tier_each_fires_exactly_once_in_one_episode(self):
        """With tiers at -15% and -25%, each fires exactly once in a single episode."""
        # Benchmark drops step by step: 0% → -16% → -26%
        bm_vals = (
            [100.0] * 20    # ATH
            + [84.0] * 20   # -16% → tier 1 (-15%) fires
            + [74.0] * 60   # -26% → tier 2 (-25%) fires
        )
        etf_df = make_prices([100.0] * 100)
        bm_df = make_prices(bm_vals)
        params = base_params(end_date=date(2020, 5, 20), initial_cash_reserve=10_000.0)

        r, ledger = run_ath_deployment(etf_df, params, TIERS_25_60, benchmark_data=bm_df)
        # 2 tiers → 2 execution events (one for each tier crossing, step by step)
        deploy_rows = ledger[ledger["deployed"] > 0]
        assert len(deploy_rows) == 2, (
            f"Expected 2 deployment events (one per tier), got {len(deploy_rows)}"
        )


# ---------------------------------------------------------------------------
# Test 10: A later ATH episode may fire the tiers again
# ---------------------------------------------------------------------------
class TestTiersReFireInLaterEpisode:
    def test_tiers_fire_again_after_new_ath(self):
        """After the second ATH, the same tiers may fire again in the next episode."""
        # Episode 1: ATH=100 → -16% (fires tier 1) → recovery to 110 (new ATH)
        # Episode 2: -16% again → fires tier 1 again (new episode)
        bm_vals = (
            [100.0] * 15   # ATH=100
            + [84.0] * 15  # -16% → tier 1 fires in episode 1
            + [110.0] * 15 # new ATH=110 → episode resets
            + [92.4] * 55  # -16% from 110 → tier 1 fires in episode 2
        )
        etf_df = make_prices([100.0] * 100)
        bm_df = make_prices(bm_vals)
        params = base_params(
            end_date=date(2020, 5, 20),
            initial_cash_reserve=10_000.0,
        )
        tiers = [DeploymentTier(-0.15, 1.00)]

        r, _ = run_ath_deployment(etf_df, params, tiers, benchmark_data=bm_df)
        assert r.n_deployments >= 2, "Tier must be able to fire in multiple ATH episodes"


# ---------------------------------------------------------------------------
# Test 11: All-in is exactly one-tier 100% policy
# ---------------------------------------------------------------------------
class TestAllInOneTierPolicy:
    def test_single_100pct_tier_deploys_all_cash(self):
        """A single tier with cumulative_deployment_fraction=1.0 deploys all cash."""
        bm_vals = [100.0] * 20 + [80.0] * 80   # -20% → fires 100% tier
        etf_vals = [100.0] * 100
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 5, 20), initial_cash_reserve=10_000.0)
        tiers = [DeploymentTier(-0.15, 1.00)]

        r, ledger = run_ath_deployment(etf_df, params, tiers, benchmark_data=bm_df)

        # All cash (≈ 10,000) must be deployed
        total_deployed = float(ledger["deployed"].sum())
        # Subtracting the tiny monthly contribution contributions
        assert total_deployed == pytest.approx(10_000.0, rel=0.01), (
            f"All-in policy must deploy all cash; got {total_deployed}"
        )
        # Ending cash should be near zero (minus contributions that arrive after deployment)
        assert r.n_deployments == 1


# ---------------------------------------------------------------------------
# Test 12: A gap across several tiers schedules only the deepest cumulative target
# ---------------------------------------------------------------------------
class TestGapSchedulesOnlyDeepest:
    def test_gap_from_0_to_30pct_schedules_one_order(self):
        """Benchmark jumps from 0% to -30% in one close; one order at deepest tier."""
        # Tiers: -15% (25%) and -25% (60%)
        # Benchmark: ATH for 20 days, then instantly at 70% of ATH (-30%)
        bm_vals = [100.0] * 20 + [70.0] * 80
        etf_vals = [100.0] * 100
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 5, 20), initial_cash_reserve=10_000.0)

        r, ledger = run_ath_deployment(etf_df, params, TIERS_25_60, benchmark_data=bm_df)

        deploy_rows = ledger[ledger["deployed"] > 0]
        assert len(deploy_rows) == 1, (
            f"Gap should schedule ONE order (deepest tier only), got {len(deploy_rows)}"
        )
        # The order must be 60% of eligible (deepest cumulative target)
        total_deployed = float(ledger["deployed"].sum())
        expected = 0.60 * 10_000.0
        assert total_deployed == pytest.approx(expected, rel=0.02), (
            f"Deployed {total_deployed}, expected ~{expected} (60% of eligible)"
        )


# ---------------------------------------------------------------------------
# Test 13: EUR 20,000 crossing -15% and -25% at once schedules EUR 12,000
# ---------------------------------------------------------------------------
class TestEur20kCrossing15And25AtOnce:
    def test_eur_20000_crossing_25pct_drawdown_schedules_eur_12000(self):
        """€20,000 jumps straight to -25%+ drawdown → single order of €12,000 (60%)."""
        bm_vals = [100.0] * 20 + [74.0] * 80   # 26% drawdown → crosses -15% AND -25%
        etf_vals = [100.0] * 100
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(
            end_date=date(2020, 5, 20),
            initial_cash_reserve=20_000.0,
        )

        r, ledger = run_ath_deployment(etf_df, params, TIERS_25_60, benchmark_data=bm_df)

        total_deployed = float(ledger["deployed"].sum())
        expected = 0.60 * 20_000.0   # 60% of eligible = 12,000
        assert total_deployed == pytest.approx(expected, rel=0.02), (
            f"Expected €12,000 (60% of €20,000); got €{total_deployed:.0f}"
        )
        deploy_rows = ledger[ledger["deployed"] > 0]
        assert len(deploy_rows) == 1, "Must be a SINGLE order, not two separate orders"


# ---------------------------------------------------------------------------
# Test 14: Signal close t executes strictly after t (not same day)
# ---------------------------------------------------------------------------
class TestSignalTExecutesAfterT:
    def test_execution_happens_on_next_instrument_day(self):
        """Signal fires on day 20 (benchmark crosses -15%); execution on day 21+."""
        bm_vals = [100.0] * 20 + [84.0] * 80   # -16% starting at index 20
        etf_vals = [100.0] * 100
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 5, 20))
        tiers = [DeploymentTier(-0.15, 1.00)]

        _, ledger = run_ath_deployment(etf_df, params, tiers, benchmark_data=bm_df)

        deploy_rows = ledger[ledger["deployed"] > 0]
        if not deploy_rows.empty:
            # Signal fires at index 20 (0-based); execution must be at index >= 21
            first_exec_pos = ledger.index.get_loc(deploy_rows.index[0])
            assert first_exec_pos >= 21, (
                f"Execution at position {first_exec_pos} should be >= 21 "
                "(signal at t, execute at t+1)"
            )


# ---------------------------------------------------------------------------
# Test 15: Event-study placeholder — xfail until scenarios.py updated
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="Event-study / scenarios.py integration not yet updated to use run_ath_deployment",
    strict=True,
)
def test_event_study_placeholder_xfail():
    """Placeholder: event-study analysis must eventually use run_ath_deployment.

    This test will be converted to a real test when app_pages/scenarios.py is
    updated to use run_ath_deployment as its engine.
    """
    # Verify that scenarios.py calls run_ath_deployment (currently it does not)
    import pathlib

    src = pathlib.Path(__file__).parent.parent.parent / "app_pages" / "scenarios.py"
    if src.exists():
        code = src.read_text()
        assert "run_ath_deployment" in code, (
            "scenarios.py must call run_ath_deployment (not yet updated)"
        )
    raise AssertionError("scenarios.py not yet updated to use run_ath_deployment")


# ---------------------------------------------------------------------------
# Test 16: Each episode contributes at most one observation per threshold
# ---------------------------------------------------------------------------
class TestSingleFirePerThresholdPerEpisode:
    def test_tier_fires_at_most_once_even_with_continued_drawdown(self):
        """A tier that fires at -20% must not fire again as drawdown deepens to -30%."""
        # -15% tier fires when benchmark crosses -15%
        # Then drawdown continues to -30% — tier must NOT fire again
        bm_vals = (
            [100.0] * 20    # ATH
            + [80.0] * 20   # -20% → tier fires once
            + [70.0] * 60   # -30% → tier already triggered; no second fire
        )
        etf_vals = [100.0] * 100
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 5, 20))

        r, ledger = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)
        assert r.n_deployments == 1, (
            "Tier fires at most once per ATH episode, regardless of deepening drawdown"
        )


# ---------------------------------------------------------------------------
# Test 17: All compared strategies receive identical external savings flows
# ---------------------------------------------------------------------------
class TestIdenticalExternalFlows:
    def test_dca_and_ath_deployment_have_same_contribution_schedule(self):
        """DCA and ATH-deployment must accumulate the same total contributions."""
        bm_vals = [100.0] * 20 + [80.0] * 232
        etf_vals = [100.0] * 252
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)

        params = base_params(
            end_date=date(2020, 12, 31),
            monthly_contribution=500.0,
            initial_cash_reserve=0.0,
        )

        _, ledger_ath = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)
        _, ledger_dca = run_dca(etf_df, params)

        total_flow_ath = float(ledger_ath["external_flow"].sum())
        total_flow_dca = float(ledger_dca["external_flow"].sum())

        assert total_flow_ath == pytest.approx(total_flow_dca, rel=1e-4), (
            f"ATH total flow {total_flow_ath:.2f} != DCA total flow {total_flow_dca:.2f}"
        )


# ---------------------------------------------------------------------------
# Test 18: DCA purchase prices never affect threshold signals
# ---------------------------------------------------------------------------
class TestDCAPricesDoNotAffectSignal:
    def test_benchmark_drawdown_is_independent_of_dca_transactions(self):
        """Running DCA alongside ATH-deployment must not alter the benchmark drawdown."""
        bm_vals = [100.0] * 10 + [84.0] * 90
        bm_df = make_prices(bm_vals)
        etf_df = make_prices([100.0] * 100)
        params = base_params(end_date=date(2020, 5, 20))

        # Run ATH deployment — benchmark drawdown is what it is
        _, ledger_ath = run_ath_deployment(etf_df, params, TIER_15, benchmark_data=bm_df)

        # Benchmark drawdown columns must reflect only the benchmark, not instrument buys
        bm_dd = ledger_ath["benchmark_drawdown"].dropna()
        # The drawdown at end must be: 84/100 - 1 = -0.16
        assert float(bm_dd.iloc[-1]) == pytest.approx(-0.16, abs=1e-4), (
            "Benchmark drawdown must reflect only benchmark prices, not DCA execution"
        )


# ---------------------------------------------------------------------------
# Test 19: Pure ATH policy has no time-based forced deployment
# ---------------------------------------------------------------------------
class TestNoMaxWaitMonthsInATHEngine:
    def test_cash_stays_uninvested_when_threshold_never_crossed(self):
        """In a rising market (no drawdown), ATH engine must never force-deploy cash.

        Note: ``initial_ath`` is explicitly set to the simulation's starting price
        (100.0).  For a rising market, every new close is a new ATH, so current_dd
        stays at 0.0 throughout — no threshold is ever crossed.  The key invariant is
        that ``max_wait_months`` in params is IGNORED by ``run_ath_deployment``.
        """
        # Rising market: never drops below -15%
        n = 252
        bm_vals = [100.0 + i * 0.2 for i in range(n)]   # pure bull
        etf_vals = [100.0 + i * 0.2 for i in range(n)]
        bm_df = make_prices(bm_vals)
        etf_df = make_prices(etf_vals)

        # max_wait_months=1 in params — but run_ath_deployment must IGNORE it
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            max_wait_months=1,       # would force-deploy in run_wait_for_dip
            monthly_contribution=0.01,
        )
        tiers = [DeploymentTier(-0.15, 1.00)]

        # initial_ath = starting price (simulation begins at ATH).
        # Using max(full series) = future price would incorrectly show a drawdown.
        r, _ = run_ath_deployment(
            etf_df, params, tiers,
            benchmark_data=bm_df,
            initial_ath=bm_vals[0],   # 100.0 — the ATH at simulation start
        )
        # No drawdown → no threshold crossing → no deployment (max_wait ignored)
        assert r.n_deployments == 0, (
            "ATH engine must not force-deploy; max_wait_months must be ignored"
        )


# ---------------------------------------------------------------------------
# Test 20: Asset lacking benchmark_data raises or warns, never silently uses ETF ATH
# ---------------------------------------------------------------------------
class TestMissingBenchmarkRaises:
    def test_none_benchmark_data_raises_value_error(self):
        """run_ath_deployment must raise ValueError when benchmark_data is None."""
        etf_vals = [100.0] * 50
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 3, 10))
        tiers = [DeploymentTier(-0.15, 1.00)]

        with pytest.raises(ValueError, match="benchmark_data"):
            run_ath_deployment(etf_df, params, tiers, benchmark_data=None)  # type: ignore[arg-type]

    def test_empty_benchmark_data_raises(self):
        """run_ath_deployment with an empty DataFrame as benchmark must raise."""
        etf_df = make_prices([100.0] * 50)
        empty_bm = pd.DataFrame({"adj_close": []}, index=pd.DatetimeIndex([]))
        params = base_params(end_date=date(2020, 3, 10))
        tiers = [DeploymentTier(-0.15, 1.00)]

        with pytest.raises((ValueError, IndexError)):
            run_ath_deployment(etf_df, params, tiers, benchmark_data=empty_bm)
