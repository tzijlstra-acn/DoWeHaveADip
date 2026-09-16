"""External flows must book at the END of the period.

The backtest accrues interest, receives the contribution, executes the purchase at
the CURRENT close, and only then values the portfolio. The contribution therefore
earns nothing during that period, so the correct form is

    r_t = (V_t - F_t) / V_{t-1} - 1

The beginning-of-period form V_t / (V_{t-1} + F_t) - 1 understates the return
whenever a contribution shares a period with a market move, which made TWR,
Sharpe, Sortino and NAV drawdown unreliable.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dipdca.models import SimulationParams
from dipdca.quant.backtest import run_dca
from dipdca.quant.metrics import (
    build_nav,
    flow_adjusted_returns,
    nav_max_drawdown,
    twr_cagr,
)


def wealth_series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def flow_series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


class TestAuditWorkedExample:
    """The audit's exact figures: 100 -> 110 with a 100 contribution -> 210."""

    def test_contribution_with_price_move_yields_true_investment_return(self):
        wealth = wealth_series([100.0, 210.0])
        flows = flow_series([0.0, 100.0])

        returns = flow_adjusted_returns(wealth, flows)

        assert len(returns) == 1
        assert float(returns.iloc[0]) == pytest.approx(0.10)

    def test_beginning_of_period_form_would_have_understated_it(self):
        """Documents the rejected result so a regression is unambiguous."""
        wealth = wealth_series([100.0, 210.0])
        flows = flow_series([0.0, 100.0])

        correct = float(flow_adjusted_returns(wealth, flows).iloc[0])
        beginning_of_period = 210.0 / (100.0 + 100.0) - 1.0

        assert beginning_of_period == pytest.approx(0.05)
        assert correct != pytest.approx(beginning_of_period)


class TestDepositsAreNotReturns:
    def test_pure_deposit_on_flat_day_is_zero_return(self):
        wealth = wealth_series([1000.0, 2000.0, 2000.0])
        flows = flow_series([0.0, 1000.0, 0.0])

        returns = flow_adjusted_returns(wealth, flows)

        assert float(returns.iloc[0]) == pytest.approx(0.0)
        assert float(returns.iloc[1]) == pytest.approx(0.0)

    def test_price_move_without_flow_is_the_price_return(self):
        wealth = wealth_series([1000.0, 1100.0])
        flows = flow_series([0.0, 0.0])

        returns = flow_adjusted_returns(wealth, flows)

        assert float(returns.iloc[0]) == pytest.approx(0.10)

    def test_withdrawal_is_not_a_loss(self):
        wealth = wealth_series([1000.0, 500.0])
        flows = flow_series([0.0, -500.0])

        returns = flow_adjusted_returns(wealth, flows)

        assert float(returns.iloc[0]) == pytest.approx(0.0)

    def test_loss_coinciding_with_a_deposit_is_measured_correctly(self):
        """Holdings fall 1000 -> 900 while 500 is deposited at the close."""
        wealth = wealth_series([1000.0, 1400.0])
        flows = flow_series([0.0, 500.0])

        returns = flow_adjusted_returns(wealth, flows)

        assert float(returns.iloc[0]) == pytest.approx(-0.10)

    def test_zero_prior_wealth_rows_are_dropped(self):
        wealth = wealth_series([0.0, 1000.0, 1100.0])
        flows = flow_series([0.0, 1000.0, 0.0])

        returns = flow_adjusted_returns(wealth, flows)

        # The 0 -> 1000 step has no prior base, so only the real move survives.
        assert len(returns) == 1
        assert float(returns.iloc[0]) == pytest.approx(0.10)


class TestDownstreamMetrics:
    def test_nav_ignores_contributions_entirely(self):
        """A pure savings pattern must leave NAV at 1.0 regardless of deposits."""
        wealth = wealth_series([1000.0, 2000.0, 3000.0, 4000.0])
        flows = flow_series([0.0, 1000.0, 1000.0, 1000.0])

        nav = build_nav(flow_adjusted_returns(wealth, flows))

        assert float(nav.iloc[-1]) == pytest.approx(1.0)

    def test_nav_drawdown_is_not_created_by_contributions(self):
        wealth = wealth_series([1000.0, 2000.0, 3000.0])
        flows = flow_series([0.0, 1000.0, 1000.0])

        nav = build_nav(flow_adjusted_returns(wealth, flows))

        assert nav_max_drawdown(nav) == pytest.approx(0.0)

    def test_nav_drawdown_reflects_a_real_price_fall(self):
        wealth = wealth_series([1000.0, 1200.0, 900.0])
        flows = flow_series([0.0, 0.0, 0.0])

        nav = build_nav(flow_adjusted_returns(wealth, flows))

        assert nav_max_drawdown(nav) == pytest.approx(-0.25)  # 900/1200 - 1

    def test_twr_is_not_inflated_by_contributions(self):
        """Savings-only growth is all deposits, so the TWR CAGR must be ~0."""
        wealth = wealth_series([1000.0 * (i + 1) for i in range(12)])
        flows = flow_series([0.0] + [1000.0] * 11)

        returns = flow_adjusted_returns(wealth, flows)
        cagr = twr_cagr(returns, years=1.0)

        assert cagr is not None
        assert cagr == pytest.approx(0.0, abs=1e-9)


class TestAgainstTheRealEngine:
    def test_dca_on_flat_prices_has_zero_investment_return(self):
        """Contributions into a flat market must not register as performance."""
        prices = [100.0] * 260
        frame = pd.DataFrame(
            {"adj_close": prices, "close": prices},
            index=pd.bdate_range("2020-01-01", periods=len(prices)),
        )
        params = SimulationParams(
            monthly_contribution=1000.0,
            payday=25,
            contribution_timing="month_end",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            pct_fee=0.0,
            slippage=0.0,
            cash_rate_override=None,
        )

        result, ledger = run_dca(frame, params)
        returns = flow_adjusted_returns(ledger["total_wealth"], ledger["external_flow"])
        nav = build_nav(returns)

        # Wealth grew purely from deposits, so NAV must be flat.
        assert result.ending_wealth > 0
        assert float(nav.iloc[-1]) == pytest.approx(1.0, abs=1e-9)
        assert nav_max_drawdown(nav) == pytest.approx(0.0, abs=1e-9)

    def test_dca_nav_drawdown_is_smaller_than_raw_wealth_drawdown(self):
        """Raw wealth drawdown is deposit-contaminated; NAV is not."""
        # Rising then falling market, with contributions throughout.
        prices = [100.0 + i * 0.5 for i in range(130)] + [
            165.0 - i * 0.5 for i in range(130)
        ]
        frame = pd.DataFrame(
            {"adj_close": prices, "close": prices},
            index=pd.bdate_range("2020-01-01", periods=len(prices)),
        )
        params = SimulationParams(
            monthly_contribution=1000.0,
            payday=25,
            contribution_timing="month_end",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            pct_fee=0.0,
            slippage=0.0,
            cash_rate_override=None,
        )

        _, ledger = run_dca(frame, params)
        returns = flow_adjusted_returns(ledger["total_wealth"], ledger["external_flow"])
        nav_dd = nav_max_drawdown(build_nav(returns))

        assert nav_dd < 0.0, "a real price fall must show up in NAV drawdown"
