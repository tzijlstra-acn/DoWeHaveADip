"""Tests for portfolio metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dipdca.quant.metrics import (
    cagr_from_wealth,
    portfolio_returns,
    sharpe_ratio,
    sortino_ratio,
    time_in_market_pct,
)


class TestSharpeRatio:
    def test_positive_sharpe(self):
        returns = pd.Series([0.001] * 252)  # Constant positive return
        result = sharpe_ratio(returns)
        assert result is not None
        assert result > 0

    def test_zero_std_returns_none(self):
        returns = pd.Series([0.0] * 252)
        result = sharpe_ratio(returns)
        assert result is None

    def test_negative_mean_negative_sharpe(self):
        returns = pd.Series([-0.001] * 252)
        result = sharpe_ratio(returns)
        assert result is not None
        assert result < 0

    def test_short_series(self):
        returns = pd.Series([0.01])
        result = sharpe_ratio(returns)
        assert result is None


class TestSortinoRatio:
    def test_no_negative_returns_none(self):
        returns = pd.Series([0.001] * 100)
        result = sortino_ratio(returns)
        assert result is None

    def test_mixed_returns(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        result = sortino_ratio(returns)
        assert result is not None  # Should compute without error


class TestPortfolioReturns:
    def test_steady_returns(self):
        wealth = pd.Series([100.0, 110.0, 121.0])
        returns = portfolio_returns(wealth)
        assert returns.iloc[0] == pytest.approx(0.10)
        assert returns.iloc[1] == pytest.approx(0.10)

    def test_flat_wealth_zero_returns(self):
        wealth = pd.Series([100.0] * 5)
        returns = portfolio_returns(wealth)
        assert (returns == 0.0).all()


class TestTimeInMarketPct:
    def test_always_invested(self):
        units = pd.Series([10.0] * 100)
        assert time_in_market_pct(units) == pytest.approx(1.0)

    def test_never_invested(self):
        units = pd.Series([0.0] * 100)
        assert time_in_market_pct(units) == pytest.approx(0.0)

    def test_half_invested(self):
        units = pd.Series([0.0] * 50 + [5.0] * 50)
        assert time_in_market_pct(units) == pytest.approx(0.5)

    def test_empty_series(self):
        units = pd.Series([], dtype=float)
        assert time_in_market_pct(units) == pytest.approx(0.0)


class TestCagrFromWealth:
    def test_doubling_in_7_years(self):
        result = cagr_from_wealth(1000.0, 2000.0, 7.0)
        assert result is not None
        assert result == pytest.approx((2.0) ** (1 / 7) - 1, rel=1e-6)

    def test_no_change(self):
        result = cagr_from_wealth(1000.0, 1000.0, 5.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_invalid_years_returns_none(self):
        assert cagr_from_wealth(1000.0, 2000.0, 0) is None

    def test_invalid_start_returns_none(self):
        assert cagr_from_wealth(0.0, 2000.0, 5.0) is None
