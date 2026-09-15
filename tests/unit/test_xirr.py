"""Tests for XIRR calculation."""

from __future__ import annotations

from datetime import date

import pytest

from dipdca.quant.xirr import xirr


class TestXirr:
    def test_simple_known_case(self):
        """$1000 invested, $1100 returned after exactly 1 year → XIRR ≈ 10%.

        Note: 2020 is a leap year (366 days), so XIRR ≈ 1.1^(365/366) - 1 ≈ 0.09971.
        """
        flows = [(date(2020, 1, 1), -1000), (date(2021, 1, 1), 1100)]
        result = xirr(flows)
        assert result is not None
        # Allow 1% relative tolerance to handle leap year (366-day year)
        assert result == pytest.approx(0.10, rel=0.01)

    def test_zero_return(self):
        """$1000 invested, $1000 returned after 1 year → XIRR ≈ 0%."""
        flows = [(date(2020, 1, 1), -1000), (date(2021, 1, 1), 1000)]
        result = xirr(flows)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_loss(self):
        """$1000 invested, $800 returned → negative XIRR."""
        flows = [(date(2020, 1, 1), -1000), (date(2021, 1, 1), 800)]
        result = xirr(flows)
        assert result is not None
        assert result < 0

    def test_all_positive_returns_none(self):
        """All positive flows — no investment — should return None."""
        flows = [(date(2020, 1, 1), 100), (date(2021, 1, 1), 200)]
        result = xirr(flows)
        assert result is None

    def test_all_negative_returns_none(self):
        """All negative flows — no receipt — should return None."""
        flows = [(date(2020, 1, 1), -100), (date(2021, 1, 1), -200)]
        result = xirr(flows)
        assert result is None

    def test_empty_returns_none(self):
        """Empty flows should return None."""
        result = xirr([])
        assert result is None

    def test_multiple_flows(self):
        """Monthly investments with terminal receipt."""
        from datetime import timedelta

        start = date(2020, 1, 1)
        flows = [(start + timedelta(days=30 * i), -100) for i in range(12)]
        # Total invested: 1200. Get back 1300 at end.
        flows.append((start + timedelta(days=365), 1300))
        result = xirr(flows)
        assert result is not None
        assert result > 0  # Should be positive return

    def test_short_term_high_return(self):
        """Double your money in ~6 months (182 days).

        Annualized: 2^(365/182) - 1 ≈ 3.02 (302% p.a.)
        """
        flows = [(date(2020, 1, 1), -1000), (date(2020, 7, 1), 2000)]
        result = xirr(flows)
        assert result is not None
        assert result > 2.5  # annualized 2x in 6 months ≈ 300% p.a.

    def test_sorted_regardless_of_input_order(self):
        """XIRR should be order-independent."""
        flows_a = [(date(2020, 1, 1), -1000), (date(2021, 1, 1), 1100)]
        flows_b = [(date(2021, 1, 1), 1100), (date(2020, 1, 1), -1000)]
        result_a = xirr(flows_a)
        result_b = xirr(flows_b)
        assert result_a == pytest.approx(result_b, rel=1e-6)
