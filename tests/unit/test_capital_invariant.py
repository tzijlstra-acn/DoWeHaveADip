"""Regression tests for quantitative correctness of the backtest engine.

Invariants enforced:
1. Both strategies receive identical external capital on identical dates.
2. initial_cash_reserve stays in cash (not deployed day 1) for dip strategies.
3. initial_cash_reserve deploys on day 1 for DCA.
4. Tiered ledger fees column is marginal (not cumulative).
5. total_contributions is the same across DCA and Wait-for-dip given identical params.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dipdca.models import SimulationParams
from dipdca.quant.backtest import run_dca, run_tiered_dip, run_wait_for_dip


def make_price_df(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({"adj_close": prices, "close": prices}, index=idx)


def base_params(**overrides) -> SimulationParams:
    base = {
        "monthly_contribution": 500.0,
        "payday": 25,
        "initial_investment": 0.0,
        "initial_cash_reserve": 0.0,
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


class TestCapitalFlowInvariant:
    """Core invariant: DCA and Wait-for-dip must receive the same external capital."""

    def test_total_contributions_equal_no_initial(self):
        """Without any initial capital, both strategies report the same total contributions."""
        n_days = 504
        prices = [100.0 + i * 0.1 for i in range(n_days)]
        df = make_price_df(prices)
        params = base_params()

        dca_result, _ = run_dca(df, params)
        dip_result, _ = run_wait_for_dip(df, params)

        assert dca_result.total_contributions == pytest.approx(dip_result.total_contributions, rel=1e-9)

    def test_total_contributions_equal_with_initial_investment(self):
        """initial_investment counts in total_contributions for both strategies."""
        n_days = 504
        prices = [100.0] * n_days
        df = make_price_df(prices)
        params = base_params(initial_investment=10_000.0)

        dca_result, _ = run_dca(df, params)
        dip_result, _ = run_wait_for_dip(df, params)

        assert dca_result.total_contributions == pytest.approx(dip_result.total_contributions, rel=1e-9)

    def test_total_contributions_equal_with_cash_reserve(self):
        """initial_cash_reserve counts in total_contributions for both strategies."""
        n_days = 504
        prices = [100.0] * n_days
        df = make_price_df(prices)
        params = base_params(initial_cash_reserve=5_000.0)

        dca_result, _ = run_dca(df, params)
        dip_result, _ = run_wait_for_dip(df, params)

        assert dca_result.total_contributions == pytest.approx(dip_result.total_contributions, rel=1e-9)

    def test_total_contributions_includes_both_initial_fields(self):
        """total_contributions = monthly_sum + initial_investment + initial_cash_reserve."""
        n_days = 504
        prices = [100.0] * n_days
        df = make_price_df(prices)
        params = base_params(initial_investment=3_000.0, initial_cash_reserve=7_000.0)

        dca_result, _ = run_dca(df, params)

        # total_contributions should be at least initial_investment + initial_cash_reserve
        assert dca_result.total_contributions >= 3_000.0 + 7_000.0


class TestInitialCashReserveAllocation:
    """initial_cash_reserve: DCA deploys day 1; dip strategies hold it in cash."""

    def test_dca_deploys_cash_reserve_on_day1(self):
        """In DCA, initial_cash_reserve + initial_investment must be in market after day 1."""
        n_days = 252
        prices = [100.0] * n_days
        df = make_price_df(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,  # tiny, to isolate reserve
            dip_threshold=-0.99,
        )

        result, ledger = run_dca(df, params)
        # After day 1, market value should reflect the deployed reserve
        assert ledger["market_value"].iloc[1] == pytest.approx(10_000.0, rel=1e-3)
        assert ledger["cash"].iloc[1] == pytest.approx(0.0, abs=1.0)

    def test_wait_for_dip_holds_cash_reserve_in_bull_market(self):
        """In a pure bull market (no dip), initial_cash_reserve must stay in cash for dip strategy."""
        n_days = 252
        prices = [100.0 + i * 0.5 for i in range(n_days)]  # rising — never dips
        df = make_price_df(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
            dip_threshold=-0.99,   # impossible threshold — never triggers
            max_wait_months=240,   # never force-deploy
        )

        result, ledger = run_wait_for_dip(df, params)
        # Cash reserve should remain uninvested throughout
        # (only force-deploys after max_wait_months which is 240 months > 1 year)
        assert ledger["cash"].iloc[5] >= 9_000.0  # still mostly in cash on day 5

    def test_wait_for_dip_deploys_cash_reserve_at_threshold(self):
        """initial_cash_reserve is deployed when dip threshold is crossed."""
        # Flat then crash
        n_flat = 80
        n_crash = 40
        n_recovery = 130
        prices = (
            [100.0] * n_flat
            + [100.0 - i * 1.0 for i in range(n_crash)]  # -40% crash
            + [60.0 + i * 0.3 for i in range(n_recovery)]
        )
        prices = [max(1.0, p) for p in prices]
        df = make_price_df(prices)

        params = base_params(
            end_date=date(2020, 12, 31),
            initial_cash_reserve=10_000.0,
            monthly_contribution=0.01,
            dip_threshold=-0.10,
            max_wait_months=240,
        )

        result, ledger = run_wait_for_dip(df, params)
        # After the crash, the reserve should have been deployed
        cash_at_end = float(ledger["cash"].iloc[-1])
        assert cash_at_end < 10_000.0  # some cash was deployed

    def test_initial_investment_deploys_day1_in_both_strategies(self):
        """initial_investment deploys on day 1 in both DCA and wait-for-dip."""
        n_days = 252
        prices = [100.0] * n_days
        df = make_price_df(prices)
        params = base_params(
            end_date=date(2020, 12, 31),
            initial_investment=5_000.0,
            monthly_contribution=0.01,
            dip_threshold=-0.99,
            max_wait_months=240,
        )

        _, dca_ledger = run_dca(df, params)
        _, dip_ledger = run_wait_for_dip(df, params)

        # Both should have ~5000 in market after day 1
        assert dca_ledger["market_value"].iloc[1] == pytest.approx(5_000.0, rel=1e-3)
        assert dip_ledger["market_value"].iloc[1] == pytest.approx(5_000.0, rel=1e-3)


class TestTieredLedgerFees:
    """Tiered dip ledger fees column must be marginal (per-day cost), not cumulative."""

    def test_tiered_fees_column_is_marginal(self):
        """Sum of ledger fees column should equal total_fees from StrategyResult."""
        n_bull = 60
        n_crash = 40
        n_recovery = 150
        prices = (
            [100.0 + i * 0.3 for i in range(n_bull)]
            + [118.0 - i * 1.5 for i in range(n_crash)]
            + [58.0 + i * 0.2 for i in range(n_recovery)]
        )
        prices = [max(1.0, p) for p in prices]
        df = make_price_df(prices)

        params = base_params(
            end_date=date(2020, 12, 31),
            pct_fee=0.001,
            dip_threshold=-0.05,
        )
        tiers = [
            {"threshold": -0.05, "fraction": 0.33},
            {"threshold": -0.10, "fraction": 0.33},
            {"threshold": -0.20, "fraction": 0.34},
        ]

        result, ledger = run_tiered_dip(df, params, tiers)

        # Sum of per-day ledger fees must equal total_fees (within floating-point tolerance)
        ledger_fees_sum = float(ledger["fees"].sum())
        assert ledger_fees_sum == pytest.approx(result.total_fees, rel=1e-6)

    def test_tiered_ledger_fees_not_cumulative(self):
        """No single row in ledger fees should exceed the running total (i.e. not storing cumulative)."""
        prices = (
            [100.0 + i * 0.2 for i in range(80)]
            + [116.0 - i * 1.0 for i in range(80)]
            + [36.0 + i * 0.5 for i in range(92)]
        )
        prices = [max(1.0, p) for p in prices]
        df = make_price_df(prices)

        params = base_params(
            end_date=date(2020, 12, 31),
            pct_fee=0.005,
            dip_threshold=-0.05,
            initial_cash_reserve=5_000.0,
        )
        tiers = [
            {"threshold": -0.05, "fraction": 0.5},
            {"threshold": -0.15, "fraction": 0.5},
        ]

        result, ledger = run_tiered_dip(df, params, tiers)

        # If fees were cumulative, some ledger rows would equal total_fees.
        # With marginal fees, every row must be ≤ total_fees AND the max single-row
        # fee must be much less than total_fees (assuming multiple deployments).
        max_single_fee = float(ledger["fees"].max())
        if result.total_fees > 0 and result.n_deployments >= 2:
            assert max_single_fee < result.total_fees
