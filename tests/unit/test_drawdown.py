"""Tests for drawdown calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from dipdca.quant.drawdown import drawdown, drawdown_label, running_peak


class TestRunningPeak:
    def test_flat_series(self):
        s = pd.Series([100.0] * 5)
        peak = running_peak(s)
        assert (peak == 100.0).all()

    def test_rising_series(self):
        s = pd.Series([100.0, 110.0, 120.0, 130.0])
        peak = running_peak(s)
        expected = pd.Series([100.0, 110.0, 120.0, 130.0])
        pd.testing.assert_series_equal(peak, expected)

    def test_falling_series(self):
        s = pd.Series([130.0, 120.0, 110.0, 100.0])
        peak = running_peak(s)
        assert (peak == 130.0).all()


class TestDrawdown:
    def test_flat_series(self):
        s = pd.Series([100.0] * 10)
        dd = drawdown(s)
        assert (dd == 0.0).all()

    def test_rising_series(self):
        s = pd.Series([100.0, 110.0, 120.0])
        dd = drawdown(s)
        assert (dd == 0.0).all()

    def test_known_peak_trough(self):
        s = pd.Series([100.0, 120.0, 90.0, 110.0])
        dd = drawdown(s)
        assert dd.iloc[0] == pytest.approx(0.0)
        assert dd.iloc[1] == pytest.approx(0.0)
        assert dd.iloc[2] == pytest.approx(-0.25)  # 90/120 - 1
        assert dd.iloc[3] == pytest.approx(110 / 120 - 1)  # -1/12

    def test_single_point(self):
        s = pd.Series([50.0])
        dd = drawdown(s)
        assert dd.iloc[0] == pytest.approx(0.0)

    def test_never_positive(self):
        """Drawdown is always <= 0."""
        import numpy as np

        rng = np.random.default_rng(42)
        s = pd.Series(rng.lognormal(0, 0.1, 100))
        dd = drawdown(s)
        assert (dd <= 1e-10).all()

    def test_full_recovery(self):
        """After full recovery to ATH, drawdown returns to 0."""
        s = pd.Series([100.0, 80.0, 60.0, 100.0, 110.0])
        dd = drawdown(s)
        assert dd.iloc[3] == pytest.approx(0.0)
        assert dd.iloc[4] == pytest.approx(0.0)


class TestDrawdownLabel:
    def test_no_dip(self):
        assert drawdown_label(0.0) == "Barely a dip"

    def test_small_dip(self):
        assert drawdown_label(-0.03) == "Barely a dip"

    def test_snack_size(self):
        assert drawdown_label(-0.07) == "Snack-size salsa"

    def test_proper_nacho(self):
        assert drawdown_label(-0.15) == "Proper nacho dip"

    def test_bear_market(self):
        assert drawdown_label(-0.25) == "Bear-market guacamole"

    def test_mariana_trench(self):
        assert drawdown_label(-0.50) == "Financial Mariana Trench"

    def test_extreme(self):
        assert drawdown_label(-0.99) == "Financial Mariana Trench"
