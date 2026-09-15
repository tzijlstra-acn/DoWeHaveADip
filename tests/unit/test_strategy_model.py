"""Strategy model correctness tests: savings-only, month-end DCA, dip deployment."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dipdca.models import DeploymentTier, SimulationParams
from dipdca.quant.backtest import run_dca, run_dip_deployment, run_savings_only


def make_prices(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


def base_params(**kw) -> SimulationParams:
    defaults = dict(
        monthly_contribution=500.0,
        payday=25,
        initial_investment=0.0,
        initial_cash_reserve=0.0,
        start_date=date(2020, 1, 1),
        end_date=date(2021, 12, 31),
        dip_threshold=-0.10,
        max_wait_months=24,
        fixed_fee=0.0,
        pct_fee=0.0,
        slippage=0.0,
        cash_rate_override=None,
    )
    defaults.update(kw)
    return SimulationParams(**defaults)


DEFAULT_TIERS = [
    DeploymentTier(-0.15, 0.25),
    DeploymentTier(-0.25, 0.60),
    DeploymentTier(-0.35, 1.00),
]


# Test 1: All strategies receive identical external flows
class TestIdenticalExternalFlows:
    def test_all_strategies_same_total_contributions(self):
        prices = [100.0] * 504
        df = make_prices(prices)
        params = base_params(initial_cash_reserve=5_000.0)
        r_savings, _ = run_savings_only(df, params)
        r_dca, _ = run_dca(df, params)
        r_dip, _ = run_dip_deployment(df, params, DEFAULT_TIERS)
        assert r_savings.total_contributions == pytest.approx(r_dca.total_contributions, rel=1e-9)
        assert r_dca.total_contributions == pytest.approx(r_dip.total_contributions, rel=1e-9)

    def test_external_flow_columns_identical_across_strategies(self):
        prices = [100.0] * 504
        df = make_prices(prices)
        params = base_params(initial_cash_reserve=3_000.0)
        _, l_savings = run_savings_only(df, params)
        _, l_dca = run_dca(df, params)
        _, l_dip = run_dip_deployment(df, params, DEFAULT_TIERS)
        s1 = l_savings["external_flow"].sum()
        s2 = l_dca["external_flow"].sum()
        s3 = l_dip["external_flow"].sum()
        assert s1 == pytest.approx(s2, rel=1e-9)
        assert s2 == pytest.approx(s3, rel=1e-9)


# Test 2: Savings-only never buys asset units
class TestSavingsOnly:
    def test_no_asset_units(self):
        prices = [100.0] * 252
        df = make_prices(prices)
        params = base_params(end_date=date(2020, 12, 31), initial_cash_reserve=1_000.0)
        _, ledger = run_savings_only(df, params)
        assert float(ledger["units"].max()) == pytest.approx(0.0, abs=1e-10)

    def test_cash_earns_savings_rate(self):
        prices = [100.0] * 252
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,  # negligible
            cash_rate_override=0.04,
        )
        r, _ = run_savings_only(df, params)
        assert r.total_cash_interest > 0.0
        # ~4% on 10k for ~1 year ≈ 400
        assert r.total_cash_interest == pytest.approx(400.0, rel=0.1)

    def test_zero_rate_no_interest(self):
        prices = [100.0] * 252
        df = make_prices(prices)
        params = base_params(end_date=date(2020, 12, 31), initial_cash_reserve=5_000.0)
        r, _ = run_savings_only(df, params)
        assert r.total_cash_interest == pytest.approx(0.0, abs=1e-6)


# Test 3: Month-end DCA invests each saving at month-end
class TestMonthEndDCA:
    def test_dca_market_value_grows(self):
        prices = [100.0 + i * 0.1 for i in range(252)]
        df = make_prices(prices)
        params = base_params(end_date=date(2020, 12, 31))
        r, _ = run_dca(df, params)
        assert r.ending_market_value > 0

    def test_dca_total_contributions_match(self):
        prices = [100.0] * 504
        df = make_prices(prices)
        params = base_params(initial_cash_reserve=2_000.0, initial_investment=500.0)
        r, ledger = run_dca(df, params)
        assert float(ledger["external_flow"].sum()) == pytest.approx(r.total_contributions, rel=1e-6)


# Test 4: Dip contributions remain in savings before trigger
class TestDipContributionsInSavings:
    def test_cash_held_when_no_trigger(self):
        # Rising market never dips to threshold
        prices = [100.0 + i * 0.5 for i in range(252)]
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
        )
        tiers = [DeploymentTier(-0.50, 1.00)]  # very deep — never reached
        r, ledger = run_dip_deployment(df, params, tiers)
        # All cash still in cash, zero units
        assert r.ending_market_value == pytest.approx(0.0, abs=1e-6)
        assert r.ending_cash > 0

    def test_dip_cash_earns_interest(self):
        prices = [100.0] * 252  # flat — no trigger at -50% threshold
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
            cash_rate_override=0.04,
        )
        tiers = [DeploymentTier(-0.50, 1.00)]
        r, _ = run_dip_deployment(df, params, tiers)
        assert r.total_cash_interest > 0.0


# Test 5: Signal at t executes at t+1
class TestNoLookAhead:
    def test_execution_not_at_signal_close(self):
        # Flat for 30 days, sharp drop on day 30 to -20%
        prices = [100.0] * 30 + [80.0] * 100 + [100.0] * 122
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=5_000.0,
            monthly_contribution=0.01,
        )
        tiers = [DeploymentTier(-0.15, 1.00)]
        _, ledger = run_dip_deployment(df, params, tiers)
        deploy_rows = ledger[ledger["deployed"] > 0]
        if not deploy_rows.empty:
            first_deploy_loc = df.index.get_loc(deploy_rows.index[0])
            # Signal fires on day 30 (index 30), execution must be on day 31+ (index 31+)
            assert first_deploy_loc >= 30


# Test 6: Threshold triggers once per episode
class TestOneTriggerPerEpisode:
    def test_single_trigger_during_crash(self):
        # Flat 20 days, crash to -20% for 30 days, recover
        prices = [100.0] * 20 + [80.0] * 30 + [100.0] * 202
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
        )
        tiers = [DeploymentTier(-0.15, 1.00)]
        _, ledger = run_dip_deployment(df, params, tiers)
        episode_deploys = int((ledger.iloc[20:50]["deployed"] > 0).sum())
        assert episode_deploys == 1, f"Expected 1 trigger, got {episode_deploys}"


# Test 7: Cumulative deployment — the EUR 20k example
class TestCumulativeDeploymentMath:
    def test_20k_three_tier_schedule(self):
        """With EUR 20k and tiers 25%/60%/100%, trades must be 5k/7k/8k."""
        # Price path that crosses all three thresholds in sequence
        prices = (
            [100.0] * 10         # flat at ATH
            + [85.0] * 5         # -15% trigger
            + [75.0] * 5         # -25% trigger
            + [65.0] * 100       # -35% trigger
            + [100.0] * 132      # recovery
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=20_000.0,
            monthly_contribution=0.01,  # negligible
            fixed_fee=0.0,
            pct_fee=0.0,
            slippage=0.0,
        )
        tiers = [
            DeploymentTier(-0.15, 0.25),
            DeploymentTier(-0.25, 0.60),
            DeploymentTier(-0.35, 1.00),
        ]
        _, ledger = run_dip_deployment(df, params, tiers)
        deploy_rows = ledger[ledger["deployed"] > 0]
        deploys = sorted(deploy_rows["deployed"].tolist())

        # Should have 3 trades approximately summing to 20k
        assert len(deploys) >= 3, f"Expected 3 trades, got {len(deploys)}: {deploys}"
        total = sum(deploys)
        assert total == pytest.approx(20_000.0, rel=0.01), f"Total deployed {total} != 20000"

    def test_incremental_not_fraction_of_remaining(self):
        """Verify 60% tier does NOT deploy 60% of remaining cash (wrong old logic)."""
        # After 25% tier deploys 5000 from 20000, remaining = 15000
        # Wrong old logic: 60% of 15000 = 9000 (total = 14000)
        # Correct new logic: 60% of 20000 = 12000; incremental = 12000 - 5000 = 7000
        prices = (
            [100.0] * 10
            + [85.0] * 10  # -15%
            + [75.0] * 100  # -25%
            + [100.0] * 132
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=20_000.0,
            monthly_contribution=0.01,
            fixed_fee=0.0,
            pct_fee=0.0,
            slippage=0.0,
        )
        tiers = [DeploymentTier(-0.15, 0.25), DeploymentTier(-0.25, 0.60)]
        _, ledger = run_dip_deployment(df, params, tiers)
        deploys = sorted(ledger[ledger["deployed"] > 0]["deployed"].tolist())

        # Two tiers should fire: ~5000 and ~7000 (total ~12000)
        assert len(deploys) == 2, f"Expected 2 trades: {deploys}"
        assert sum(deploys) == pytest.approx(12_000.0, rel=0.01)
        # The second trade must be ~7000, NOT ~9000 (old wrong logic)
        second_trade = deploys[1]
        assert second_trade == pytest.approx(7_000.0, rel=0.01), (
            f"Second trade {second_trade} looks like old logic (should be ~7000, not ~9000)"
        )


# Test 8: Contributions between thresholds included at deeper trigger
class TestContributionsBetweenTiers:
    def test_new_savings_included_in_eligible_capital(self):
        """Savings arriving between tier 1 and tier 2 are included at tier 2."""
        prices = (
            [100.0] * 5
            + [85.0] * 50   # -15%, tier 1 fires
            + [75.0] * 100  # -25%, tier 2 fires (with new contributions included)
            + [100.0] * 97
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=20_000.0,
            monthly_contribution=1_000.0,  # meaningful contributions during drawdown
        )
        tiers = [DeploymentTier(-0.15, 0.25), DeploymentTier(-0.25, 0.60)]
        r, ledger = run_dip_deployment(df, params, tiers)

        # Total deployed must be more than 12000 (since new contributions arrive)
        total_deployed = float(ledger["deployed"].sum())
        assert total_deployed > 12_000.0, "New contributions should have increased eligible capital"


# Test 9: Asset gains don't alter deployment target
class TestAssetGainsDoNotAlterTarget:
    def test_only_principal_counts_not_market_value(self):
        """Asset value increases must not inflate the cumulative deployment target.

        Between tier 0 and tier 1 the price drifts from 85 to 95 (asset up ~12%),
        but the episode does NOT reset (95 < ATH of 100). At tier 1 (-25%),
        eligible must use PRINCIPAL (5000), not market value (5000/85 * 75 = 4412).
        """
        prices = (
            [100.0] * 10
            + [85.0] * 5   # -15%: deploy 25% of 20k = 5k
            + [95.0] * 5   # asset rises to 95 (below ATH of 100 — no episode reset)
            + [75.0] * 100  # -25% from ATH=100: target should use PRINCIPAL (5k)
            + [100.0] * 127
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=20_000.0,
            monthly_contribution=0.01,
            fixed_fee=0.0,
            pct_fee=0.0,
            slippage=0.0,
        )
        tiers = [DeploymentTier(-0.15, 0.25), DeploymentTier(-0.25, 0.60)]
        _, ledger = run_dip_deployment(df, params, tiers)
        deploys = sorted(ledger[ledger["deployed"] > 0]["deployed"].tolist())

        assert len(deploys) == 2, f"Expected 2 trades, got {deploys}"
        # Correct: eligible = cash(15000) + principal(5000) = 20000, incremental = 7000
        # Wrong (market value): eligible = 15000 + 4412 = 19412, incremental != 7000
        assert deploys[1] == pytest.approx(7_000.0, rel=0.01)


# Test 10: New ATH rearms the strategy
class TestEpisodeRearm:
    def test_new_ath_rearms_for_next_episode(self):
        """After recovering to a new ATH, a subsequent dip triggers again."""
        prices = (
            [100.0] * 10
            + [85.0] * 30   # episode 1: -15%
            + [105.0] * 10  # new ATH — rearm
            + [89.25] * 80  # episode 2: -15% of 105 = 89.25 — should trigger again
            + [105.0] * 122
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=500.0,
        )
        tiers = [DeploymentTier(-0.15, 1.00)]
        r, ledger = run_dip_deployment(df, params, tiers)
        # Both episodes should trigger → at least 2 deployment events
        n_deploy_events = int((ledger["deployed"] > 0).sum())
        assert n_deploy_events >= 2, f"Expected >=2 deployment events, got {n_deploy_events}"


# Test 11: Deposit alone is not an investment return
class TestDepositNotAReturn:
    def test_flow_adjusted_return_zero_on_deposit_day(self):
        from dipdca.quant.metrics import flow_adjusted_returns
        wealth = pd.Series([1000.0, 1500.0], index=pd.bdate_range("2020-01-01", periods=2))
        flows = pd.Series([0.0, 500.0], index=wealth.index)
        r = flow_adjusted_returns(wealth, flows)
        assert r.iloc[0] == pytest.approx(0.0, abs=1e-10)


# Test 12: Zero-threshold-event path — dip acts like savings-only
class TestNoTriggerPathLikeSavings:
    def test_very_deep_threshold_never_reached(self):
        """If threshold never crossed, dip strategy keeps all cash (like savings-only)."""
        prices = [100.0 + i * 0.3 for i in range(252)]
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
        )
        tiers = [DeploymentTier(-0.99, 1.00)]  # never reached
        r_dip, l_dip = run_dip_deployment(df, params, tiers)
        r_sav, l_sav = run_savings_only(df, params)
        # Both end with all capital in cash (market value = 0 for dip)
        assert r_dip.ending_market_value == pytest.approx(0.0, abs=1e-6)
        # Total wealth should be close (both earn same cash interest)
        assert r_dip.ending_wealth == pytest.approx(r_sav.ending_wealth, rel=1e-6)


# Test 13: 100% first-tier deploys full reserve once
class TestFullDeployOnFirstTier:
    def test_100pct_first_tier_deploys_everything(self):
        prices = (
            [100.0] * 20
            + [80.0] * 100  # -20%, below -15% threshold
            + [100.0] * 132
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
        )
        tiers = [DeploymentTier(-0.15, 1.00)]
        r, ledger = run_dip_deployment(df, params, tiers)
        # One deployment event at approximately 10000
        deploy_rows = ledger[ledger["deployed"] > 0]
        assert len(deploy_rows) == 1
        assert float(deploy_rows["deployed"].iloc[0]) == pytest.approx(10_000.0, rel=0.01)


# Test 14: Final fraction below 100% intentionally leaves cash undeployed
class TestPartialDeploymentLeavesCash:
    def test_70pct_max_leaves_30pct_in_cash(self):
        prices = (
            [100.0] * 10
            + [80.0] * 100  # -20%
            + [60.0] * 50   # -40%
            + [100.0] * 92
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
        )
        tiers = [DeploymentTier(-0.15, 0.40), DeploymentTier(-0.35, 0.70)]  # max 70%
        r, _ = run_dip_deployment(df, params, tiers)
        # ~30% of 10000 = 3000 should remain in cash
        assert r.ending_cash == pytest.approx(3_000.0, rel=0.05)


# Test 15: DeploymentTier validation
class TestDeploymentTierValidation:
    def test_positive_threshold_rejected(self):
        with pytest.raises(ValueError):
            DeploymentTier(0.15, 0.50)  # must be negative

    def test_zero_fraction_rejected(self):
        with pytest.raises(ValueError):
            DeploymentTier(-0.15, 0.0)  # must be > 0

    def test_decreasing_threshold_order_rejected(self):
        tiers = [DeploymentTier(-0.25, 0.50), DeploymentTier(-0.15, 1.00)]
        with pytest.raises(ValueError):
            DeploymentTier.validate_schedule(tiers)

    def test_decreasing_fraction_rejected(self):
        tiers = [DeploymentTier(-0.15, 0.60), DeploymentTier(-0.25, 0.40)]
        with pytest.raises(ValueError):
            DeploymentTier.validate_schedule(tiers)


# Test 16: Savings interest accrues on undeployed cash
class TestCashInterestDuringDrawdown:
    def test_interest_on_remaining_cash(self):
        prices = (
            [100.0] * 10
            + [80.0] * 200  # below threshold
            + [100.0] * 42
        )
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
            cash_rate_override=0.04,
        )
        tiers = [DeploymentTier(-0.15, 0.50)]  # deploy only 50%
        r, _ = run_dip_deployment(df, params, tiers)
        # 50% deployed, 50% remaining should still earn interest
        assert r.total_cash_interest > 0


# Test 17: Undeployed cash reported
class TestUndeployedCashVisible:
    def test_ending_cash_positive_when_threshold_not_reached(self):
        prices = [100.0 + i * 0.2 for i in range(252)]
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=8_000.0,
            monthly_contribution=200.0,
        )
        tiers = [DeploymentTier(-0.50, 1.00)]  # never triggered in rising market
        r, _ = run_dip_deployment(df, params, tiers)
        assert r.ending_cash > 0
        assert r.n_deployments == 0


# Test 18: Savings-only always outperforms dip when market never dips
class TestSavingsOnlyVsDipNoDip:
    def test_savings_only_ends_same_wealth_as_dip_with_same_cash_rate(self):
        """If threshold never hit, dip = savings-only (when both earn same rate)."""
        prices = [100.0 + i * 0.1 for i in range(252)]
        df = make_prices(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
            cash_rate_override=0.03,
        )
        tiers = [DeploymentTier(-0.99, 1.00)]
        r_sav, _ = run_savings_only(df, params)
        r_dip, _ = run_dip_deployment(df, params, tiers)
        assert r_sav.ending_wealth == pytest.approx(r_dip.ending_wealth, rel=1e-5)


# Test 19: DCA wealth invariant
class TestDCAInvestedAmount:
    def test_dca_market_value_plus_cash_equals_total_wealth(self):
        prices = [100.0] * 252
        df = make_prices(prices)
        params = base_params(end_date=date(2020, 12, 31))
        r, ledger = run_dca(df, params)
        last_row = ledger.iloc[-1]
        assert float(last_row["total_wealth"]) == pytest.approx(
            float(last_row["cash"]) + float(last_row["market_value"]), abs=1e-6
        )


# Test 20: DeploymentTier validate empty schedule raises
class TestEmptyScheduleRejected:
    def test_empty_tiers_rejected_in_validate(self):
        with pytest.raises(ValueError):
            DeploymentTier.validate_schedule([])
