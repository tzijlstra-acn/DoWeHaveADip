"""Tests for FX conversion utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dipdca.quant.fx import fx_base_per_asset, log_return_decomposition, value_in_base


class TestFxBasePerAsset:
    def test_eur_base_usd_asset(self):
        """EUR base, USD asset: q_EUR=1, q_USD=1.08 → EUR/USD=1/1.08."""
        result = fx_base_per_asset(q_base=1.0, q_asset=1.08)
        assert result == pytest.approx(1 / 1.08, rel=1e-9)

    def test_same_currency(self):
        """Same currency: rate should be 1."""
        result = fx_base_per_asset(q_base=1.0, q_asset=1.0)
        assert result == pytest.approx(1.0)

    def test_chf_base_usd_asset(self):
        """CHF base (q_CHF=0.93), USD asset (q_USD=1.08)."""
        result = fx_base_per_asset(q_base=0.93, q_asset=1.08)
        assert result == pytest.approx(0.93 / 1.08, rel=1e-9)

    def test_zero_asset_raises(self):
        with pytest.raises(ValueError):
            fx_base_per_asset(q_base=1.0, q_asset=0.0)


class TestValueInBase:
    def test_usd_price_to_eur(self):
        """$100 asset → EUR value."""
        # q_EUR = 1, q_USD = 1.08 → EUR per USD = 1/1.08
        result = value_in_base(100.0, q_base=1.0, q_asset=1.08)
        assert result == pytest.approx(100.0 / 1.08, rel=1e-9)

    def test_same_currency_unchanged(self):
        result = value_in_base(250.0, q_base=1.0, q_asset=1.0)
        assert result == pytest.approx(250.0)


class TestLogReturnDecomposition:
    def test_same_returns_zero_fx(self):
        """If TR_base == TR_native (no FX effect), FX log returns should be ~0."""
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        tr = pd.Series(np.cumprod(1 + np.array([0.01] * 10)), index=idx)
        native_ret, fx_ret = log_return_decomposition(tr, tr)
        assert fx_ret.abs().max() == pytest.approx(0.0, abs=1e-10)

    def test_returns_sum_to_total(self):
        """native + fx log returns should equal total log return."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        tr_native = pd.Series(np.cumprod(1 + rng.normal(0.001, 0.01, 50)), index=idx)
        fx_moves = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.005, 50)), index=idx)
        tr_base = tr_native * fx_moves

        native_ret, fx_ret = log_return_decomposition(tr_base, tr_native)
        total_log = np.log(tr_base).diff().dropna()
        combined = native_ret + fx_ret

        pd.testing.assert_series_equal(
            combined.round(10), total_log.loc[combined.index].round(10)
        )
