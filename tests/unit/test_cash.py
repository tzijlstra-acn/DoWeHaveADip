"""Tests for cash interest accrual."""

from __future__ import annotations

import pandas as pd
import pytest

from dipdca.quant.cash import accrue_interest, accrue_over_period


class TestAccrueInterest:
    def test_positive_rate(self):
        result = accrue_interest(1000.0, 0.03, 365)
        assert result == pytest.approx(1030.0, rel=1e-6)

    def test_zero_rate(self):
        assert accrue_interest(1000.0, 0.0, 180) == pytest.approx(1000.0)

    def test_negative_valid_rate(self):
        result = accrue_interest(1000.0, -0.005, 365)
        assert result == pytest.approx(995.0, rel=1e-4)

    def test_invalid_rate_minus_one(self):
        with pytest.raises(ValueError, match="annual_rate"):
            accrue_interest(1000.0, -1.0, 365)

    def test_invalid_rate_below_minus_one(self):
        with pytest.raises(ValueError, match="annual_rate"):
            accrue_interest(1000.0, -1.5, 365)

    def test_zero_days(self):
        result = accrue_interest(1000.0, 0.05, 0)
        assert result == pytest.approx(1000.0)

    def test_compounding(self):
        """Test that compounding exceeds simple interest."""
        result = accrue_interest(1000.0, 0.10, 730)
        # Compound: 1000 * 1.10^2 = 1210
        assert result == pytest.approx(1000.0 * 1.10**2, rel=1e-6)

    def test_high_rate(self):
        result = accrue_interest(100.0, 0.50, 365)
        assert result == pytest.approx(150.0, rel=1e-6)


class TestAccrueOverPeriod:
    def test_zero_cash_no_interest(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        cash = pd.Series(0.0, index=idx)
        rates = pd.Series(0.05, index=idx)
        interest = accrue_over_period(cash, rates)
        assert interest.sum() == pytest.approx(0.0)

    def test_positive_rate_earns_interest(self):
        idx = pd.date_range("2020-01-01", periods=252, freq="B")
        cash = pd.Series(10000.0, index=idx)
        rates = pd.Series(0.04, index=idx)
        interest = accrue_over_period(cash, rates)
        # 252 business days span ~350 calendar days, not 365.
        # At 4% p.a. on 10000, interest ≈ 10000 * 0.04 * (350/365) ≈ 384
        assert interest.sum() > 0
        assert 300.0 < interest.sum() < 450.0

    def test_empty_series(self):
        idx = pd.DatetimeIndex([])
        cash = pd.Series(dtype=float, index=idx)
        rates = pd.Series(dtype=float, index=idx)
        result = accrue_over_period(cash, rates)
        assert len(result) == 0
