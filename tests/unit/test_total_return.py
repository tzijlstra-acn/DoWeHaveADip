"""Tests for total return calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from dipdca.quant.total_return import normalize_total_return, validate_adjusted_data


class TestNormalizeTotalReturn:
    def test_rebases_to_one(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        adj = pd.Series([100.0] * 10, index=idx)
        fx = pd.Series([1.0] * 10, index=idx)
        result = normalize_total_return(adj, fx, idx[0])
        assert result.iloc[0] == pytest.approx(1.0)

    def test_rising_price(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        adj = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=idx)
        fx = pd.Series([1.0] * 5, index=idx)
        result = normalize_total_return(adj, fx, idx[0])
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[-1] == pytest.approx(1.4)

    def test_fx_impact(self):
        """Depreciating asset currency reduces TR in base currency."""
        idx = pd.date_range("2020-01-01", periods=2, freq="B")
        adj = pd.Series([100.0, 100.0], index=idx)
        # FX moves from 1.0 to 0.9 (10% depreciation)
        fx = pd.Series([1.0, 0.9], index=idx)
        result = normalize_total_return(adj, fx, idx[0])
        assert result.iloc[1] == pytest.approx(0.9)


class TestValidateAdjustedData:
    def test_no_adj_close_column_returns_false(self):
        df = pd.DataFrame({"close": [100.0, 101.0]})
        assert validate_adjusted_data(df) is False

    def test_adj_equals_close_returns_false(self):
        df = pd.DataFrame({"adj_close": [100.0, 101.0], "close": [100.0, 101.0]})
        assert validate_adjusted_data(df) is False

    def test_adj_differs_from_close_returns_true(self):
        df = pd.DataFrame({"adj_close": [98.0, 101.0], "close": [100.0, 102.0]})
        assert validate_adjusted_data(df) is True

    def test_all_nan_adj_close_returns_false(self):

        df = pd.DataFrame({"adj_close": [float("nan"), float("nan")]})
        assert validate_adjusted_data(df) is False

    def test_no_close_column_still_valid(self):
        """adj_close present without close column should return True (no comparison possible)."""
        df = pd.DataFrame({"adj_close": [98.0, 101.5, 99.0]})
        assert validate_adjusted_data(df) is True
