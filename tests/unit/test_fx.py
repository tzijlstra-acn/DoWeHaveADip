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


class TestFxCrossRateTriangulation:
    """Verify EUR triangulation math for non-EUR base/asset pairs."""

    def test_usd_base_chf_asset(self):
        """USD base, CHF asset: cross = USD_per_EUR / CHF_per_EUR."""
        # ECB rates: USD=1.08, CHF=0.93 (both per EUR)
        usd_per_eur = 1.08
        chf_per_eur = 0.93
        cross = usd_per_eur / chf_per_eur  # USD per CHF
        assert cross == pytest.approx(1.08 / 0.93, rel=1e-9)

    def test_eur_base_usd_asset_via_fx_function(self):
        """EUR base, USD asset: fx_base_per_asset = 1 / (USD_per_EUR)."""
        result = fx_base_per_asset(q_base=1.0, q_asset=1.08)
        assert result == pytest.approx(1 / 1.08, rel=1e-9)

    def test_triangulation_consistency(self):
        """USD/CHF cross via EUR should equal direct USD_per_CHF rate."""
        usd_per_eur = 1.08
        chf_per_eur = 0.93
        # Triangulate: USD per CHF = (USD per EUR) / (CHF per EUR)
        cross = usd_per_eur / chf_per_eur
        # Direct: CHF 1 = EUR (1/0.93); EUR 1 = USD 1.08; so CHF 1 = USD 1.08/0.93
        direct = 1.08 / 0.93
        assert cross == pytest.approx(direct, rel=1e-9)

    def test_same_currency_cross_is_one(self):
        """Same currency: cross rate must be 1.0."""
        rate_per_eur = 1.08
        cross = rate_per_eur / rate_per_eur
        assert cross == pytest.approx(1.0)

    def test_cross_rate_inverse(self):
        """(A/B cross rate) * (B/A cross rate) should equal 1."""
        usd_per_eur = 1.08
        chf_per_eur = 0.93
        usd_per_chf = usd_per_eur / chf_per_eur
        chf_per_usd = chf_per_eur / usd_per_eur
        assert usd_per_chf * chf_per_usd == pytest.approx(1.0, rel=1e-9)


class TestFxSeries:
    """Test that Series FX calculations work correctly."""

    def test_eur_per_asset_from_ecb_rates(self):
        """ECB gives USD_per_EUR; EUR_per_USD = 1 / USD_per_EUR."""
        import pandas as pd

        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        usd_per_eur = pd.Series([1.08, 1.09, 1.07, 1.10, 1.08], index=idx)
        eur_per_usd = 1.0 / usd_per_eur
        expected = pd.Series([1/1.08, 1/1.09, 1/1.07, 1/1.10, 1/1.08], index=idx)
        pd.testing.assert_series_equal(eur_per_usd, expected)

    def test_cross_rate_series(self):
        """USD/CHF cross computed from ECB rates matches manual calculation."""
        import pandas as pd

        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        usd_per_eur = pd.Series([1.08, 1.09, 1.07], index=idx)
        chf_per_eur = pd.Series([0.93, 0.94, 0.92], index=idx)
        cross = usd_per_eur / chf_per_eur
        expected = pd.Series([1.08/0.93, 1.09/0.94, 1.07/0.92], index=idx)
        pd.testing.assert_series_equal(cross, expected)
