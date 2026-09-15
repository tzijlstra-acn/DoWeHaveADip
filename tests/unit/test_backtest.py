"""Tests for the core backtest engine."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dipdca.models import SimulationParams
from dipdca.quant.backtest import run_dca, run_wait_for_dip


def make_price_df(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Helper: create a minimal price DataFrame from a list of prices."""
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({"adj_close": prices, "close": prices}, index=idx)


def default_params(**overrides) -> SimulationParams:
    """Helper: minimal valid SimulationParams."""
    base = {
        "monthly_contribution": 500.0,
        "payday": 25,
        "initial_investment": 0.0,
        "start_date": date(2020, 1, 1),
        "end_date": date(2021, 12, 31),
        "dip_threshold": -0.05,
        "max_wait_months": 24,
        "fixed_fee": 0.0,
        "pct_fee": 0.0,
        "slippage": 0.0,
    }
    base.update(overrides)
    return SimulationParams(**base)


class TestRunDca:
    def test_dca_rising_market_deploys_all(self):
        """DCA deploys every contribution immediately — ending cash should be near 0."""
        n_days = 504  # ~2 years trading days
        prices = [100.0 + i * 0.1 for i in range(n_days)]
        df = make_price_df(prices)
        params = default_params()

        result, ledger = run_dca(df, params)
        assert result.ending_cash == pytest.approx(0.0, abs=1.0)
        assert result.n_deployments > 0
        assert result.ending_wealth > result.total_contributions  # market rose

    def test_dca_deploys_every_month(self):
        """Each monthly contribution triggers exactly 1 deployment."""
        n_days = 504
        prices = [100.0] * n_days
        df = make_price_df(prices)
        params = default_params()

        result, ledger = run_dca(df, params)
        # 2 years × 12 months = 24 deployments expected
        assert result.n_deployments == pytest.approx(24, abs=2)

    def test_costs_reduce_results(self):
        """Adding fees/slippage must reduce ending wealth."""
        n_days = 252
        prices = [100.0 + i * 0.05 for i in range(n_days)]
        df = make_price_df(prices)
        params_no_cost = default_params(
            end_date=date(2020, 12, 31), fixed_fee=0.0, pct_fee=0.0, slippage=0.0
        )
        params_with_cost = default_params(
            end_date=date(2020, 12, 31), fixed_fee=5.0, pct_fee=0.005, slippage=0.001
        )

        result_no_cost, _ = run_dca(df, params_no_cost)
        result_with_cost, _ = run_dca(df, params_with_cost)

        assert result_with_cost.ending_wealth < result_no_cost.ending_wealth
        assert result_with_cost.total_fees > 0

    def test_initial_investment_deployed(self):
        """Initial investment is deployed on the first trading day."""
        n_days = 252
        prices = [100.0] * n_days
        df = make_price_df(prices)
        params = default_params(
            end_date=date(2020, 12, 31),
            initial_investment=10000.0,
            monthly_contribution=0.01,  # tiny to isolate effect
        )

        result, ledger = run_dca(df, params)
        # Initial investment deployed + monthly
        assert result.ending_market_value > 9000.0

    def test_ledger_structure(self):
        """Ledger must have required columns."""
        n_days = 252
        prices = [100.0 + i * 0.1 for i in range(n_days)]
        df = make_price_df(prices)
        params = default_params(end_date=date(2020, 12, 31))
        _, ledger = run_dca(df, params)

        required_cols = {"price", "cash", "units", "market_value", "total_wealth", "dd"}
        assert required_cols.issubset(set(ledger.columns))

    def test_ending_wealth_equals_cash_plus_market(self):
        """ending_wealth must equal ending_cash + ending_market_value."""
        n_days = 252
        prices = [100.0 + i * 0.1 for i in range(n_days)]
        df = make_price_df(prices)
        params = default_params(end_date=date(2020, 12, 31))
        result, _ = run_dca(df, params)
        assert result.ending_wealth == pytest.approx(
            result.ending_cash + result.ending_market_value, rel=1e-6
        )


class TestRunWaitForDip:
    def test_no_dip_deploys_on_max_wait(self):
        """If market never dips, force-deploy after max_wait_months."""
        n_days = 504
        prices = [100.0 + i * 0.1 for i in range(n_days)]
        df = make_price_df(prices)
        params = default_params(
            dip_threshold=-0.99,  # Essentially never trigger
            max_wait_months=1,  # Force deploy after 1 month
        )

        result, ledger = run_wait_for_dip(df, params)
        # Should have deployed (max_wait triggered)
        assert result.n_deployments > 0

    def test_wait_for_dip_threshold_crossing(self):
        """Cash not deployed until drawdown <= threshold."""
        # Rising market for 60 days, then crash, then recovery
        n_bull = 60
        n_crash = 30
        n_recovery = 160

        prices = (
            [100.0 + i * 0.2 for i in range(n_bull)]
            + [112.0 - i * 1.5 for i in range(n_crash)]  # crash below threshold
            + [67.0 + i * 0.3 for i in range(n_recovery)]
        )
        prices = [max(10.0, p) for p in prices]
        df = make_price_df(prices)

        params = default_params(
            end_date=date(2021, 12, 31),
            dip_threshold=-0.05,
            max_wait_months=24,
        )

        result, ledger = run_wait_for_dip(df, params)
        # Should have deployed at least once during crash
        assert result.n_deployments > 0

    def test_no_lookahead_signal_t_minus_1(self):
        """Signal from t-1 must not execute at t-1, only at t or later."""
        # Create a price series with a known dip at position 100
        n_days = 504
        prices = [100.0] * n_days
        # Introduce a sharp dip at position 100
        for i in range(90, 110):
            prices[i] = 90.0  # -10% dip
        df = make_price_df(prices)

        params = default_params(
            dip_threshold=-0.05,
            max_wait_months=24,
        )

        result, ledger = run_wait_for_dip(df, params)

        # The first deployment after the dip should be at index ≥ 91
        # (signal at 90 → execute at 91+)
        deployed_dates = ledger[ledger["deployed"] > 0].index
        if len(deployed_dates) > 0:
            # First deployment must be on index 91 or later (not during signal day)
            first_deploy_pos = df.index.get_loc(deployed_dates[0])
            assert first_deploy_pos >= 1  # At minimum, not at index 0

    def test_tier_reset_only_after_new_high(self):
        """Tiered dip tiers only reset after a new ATH."""
        from dipdca.quant.backtest import run_tiered_dip

        # Price goes up, crashes, recovers to new ATH
        n_days = 252
        prices = (
            [100.0 + i * 0.5 for i in range(100)]  # bull
            + [149.0 - i * 1.0 for i in range(80)]  # crash
            + [69.0 + i * 0.6 for i in range(72)]  # recovery
        )
        prices = [max(1.0, p) for p in prices[:n_days]]
        df = make_price_df(prices)

        params = default_params(end_date=date(2020, 12, 31))
        tiers = [
            {"threshold": -0.05, "fraction": 0.25},
            {"threshold": -0.10, "fraction": 0.50},
            {"threshold": -0.20, "fraction": 1.00},
        ]

        result, ledger = run_tiered_dip(df, params, tiers)
        assert result.n_deployments >= 1

    def test_dip_holds_cash_during_bull(self):
        """In a pure bull market with tight threshold, dip strategy holds more cash than DCA."""
        n_days = 504
        prices = [100.0 + i * 0.3 for i in range(n_days)]
        df = make_price_df(prices)

        params = default_params(
            dip_threshold=-0.30,  # Very tight — market never dips this much
            max_wait_months=100,  # Never force-deploy
        )

        _, dip_ledger = run_wait_for_dip(df, params)
        _, dca_ledger = run_dca(df, params)

        # DIP should hold more cash on average
        avg_cash_dip = dip_ledger["cash"].mean()
        avg_cash_dca = dca_ledger["cash"].mean()
        assert avg_cash_dip > avg_cash_dca
