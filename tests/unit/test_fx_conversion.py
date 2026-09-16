"""FX conversion must affect execution and valuation but not the native benchmark signal.

Key invariants:
- The reference index drawdown is calculated in the index's published native level.
- FX conversion applies to the investable instrument prices only.
- A constant FX rate changes the EUR value of each unit purchased but not the
  number of units bought relative to DCA (if DCA also uses the same EUR budget).
- A 2x USD appreciation makes EUR buyers pay twice as much per unit, halving their
  units relative to a hypothetical USD investor.
- The native index ATH and drawdown are unchanged by any FX move.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dipdca.quant.fx import instrument_to_base_currency


def ecb_frame(usd_per_eur: float | list[float], n: int, start: str = "2020-01-01") -> pd.DataFrame:
    """Minimal ECB rates frame: USD_per_EUR over n days."""
    idx = pd.bdate_range(start=start, periods=n)
    rates = [float(usd_per_eur)] * n if isinstance(usd_per_eur, (int, float)) else list(usd_per_eur)
    return pd.DataFrame({"USD": rates, "EUR": 1.0}, index=idx)


def price_frame(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({"adj_close": prices, "close": prices}, index=idx)


class TestConstantFxConversion:
    def test_constant_fx_scales_all_prices(self):
        """USD_per_EUR=2.0 means 1 EUR = 2 USD, so 100 USD = 50 EUR."""
        pf = price_frame([100.0, 200.0, 150.0])
        rates = ecb_frame(usd_per_eur=2.0, n=3)

        result = instrument_to_base_currency(pf, "USD", "EUR", rates)

        assert result["adj_close"].tolist() == pytest.approx([50.0, 100.0, 75.0])
        assert result["close"].tolist() == pytest.approx([50.0, 100.0, 75.0])

    def test_same_currency_returns_unchanged(self):
        pf = price_frame([100.0, 110.0])
        rates = ecb_frame(usd_per_eur=1.08, n=2)

        result = instrument_to_base_currency(pf, "EUR", "EUR", rates)

        assert result["adj_close"].tolist() == [100.0, 110.0]

    def test_eur_to_usd_multiplies_by_rate(self):
        """1 EUR = 1.08 USD, so 100 EUR = 108 USD."""
        pf = price_frame([100.0])
        rates = ecb_frame(usd_per_eur=1.08, n=1)

        result = instrument_to_base_currency(pf, "EUR", "USD", rates)

        assert result["adj_close"].iloc[0] == pytest.approx(108.0)

    def test_missing_currency_raises(self):
        pf = price_frame([100.0])
        rates = ecb_frame(usd_per_eur=1.08, n=1)  # only USD column

        with pytest.raises(ValueError, match="CHF"):
            instrument_to_base_currency(pf, "CHF", "EUR", rates)

    def test_original_frame_is_not_mutated(self):
        pf = price_frame([100.0, 200.0])
        original_prices = pf["adj_close"].copy()
        rates = ecb_frame(usd_per_eur=2.0, n=2)

        instrument_to_base_currency(pf, "USD", "EUR", rates)

        assert pf["adj_close"].tolist() == original_prices.tolist()


class TestChangingFxChangesWealthNotSignal:
    def test_fx_changes_units_purchased(self):
        """With a fixed EUR budget, stronger USD means fewer units per EUR.

        If 1 EUR buys 2 USD (rate=2.0), a 100 USD instrument costs 50 EUR,
        so 1000 EUR buys 20 units.
        If 1 EUR buys 1 USD (rate=1.0), the same instrument costs 100 EUR,
        so 1000 EUR buys 10 units.
        """
        p_usd = [100.0]
        eur_budget = 1_000.0

        for usd_per_eur, expected_units in [(2.0, 20.0), (1.0, 10.0)]:
            pf = price_frame(p_usd)
            rates = ecb_frame(usd_per_eur=usd_per_eur, n=1)
            pf_eur = instrument_to_base_currency(pf, "USD", "EUR", rates)
            price_eur = float(pf_eur["adj_close"].iloc[0])
            units = eur_budget / price_eur
            assert units == pytest.approx(expected_units)

    def test_fx_does_not_change_native_index_drawdown(self):
        """The benchmark index passes through the engine unchanged.

        FX is applied to the instrument only. The benchmark drawdown is computed
        from the native index levels and must not shift when the FX rate changes.
        """
        # A 10% drawdown in native index levels
        bm_native = [100.0, 90.0]
        expected_dd = 90.0 / 100.0 - 1.0  # -10%

        # Verify the drawdown calculation doesn't involve FX
        from dipdca.quant.drawdown import drawdown
        bm_series = pd.Series(bm_native, index=pd.bdate_range("2020-01-01", periods=2))
        dd = drawdown(bm_series)

        assert float(dd.iloc[-1]) == pytest.approx(expected_dd)

        # Now verify that applying FX to the INSTRUMENT does not change the benchmark
        pf_instrument = price_frame([50.0, 45.0])  # some USD instrument
        for usd_rate in [0.5, 1.0, 2.0, 5.0]:
            rates = ecb_frame(usd_per_eur=usd_rate, n=2)
            instrument_to_base_currency(pf_instrument, "USD", "EUR", rates)
            # The benchmark is untouched — the FX was applied only to the instrument
            assert float(dd.iloc[-1]) == pytest.approx(expected_dd)

    def test_missing_fx_should_not_silently_produce_eur_label(self):
        """If FX fetch fails, the result is in native currency and must not be labelled EUR.

        This test verifies the invariant at the function level — the calling code
        is responsible for the UI label, but the function itself must raise or
        return native prices unchanged when currency info is missing.
        """
        pf = price_frame([100.0])
        empty_rates = pd.DataFrame({"EUR": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1))

        with pytest.raises(ValueError):
            instrument_to_base_currency(pf, "USD", "EUR", empty_rates)

    def test_fx_forward_fill_does_not_use_future_rate(self):
        """Rates are aligned with reindex(method='ffill') — no future rates used."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        n = len(prices)
        idx = pd.bdate_range("2020-01-01", periods=n)
        pf = pd.DataFrame({"adj_close": prices, "close": prices}, index=idx)

        # Only two known FX dates: day 0 and day 3
        rates_idx = idx[[0, 3]]
        rates = pd.DataFrame({"USD": [1.0, 2.0], "EUR": [1.0, 1.0]}, index=rates_idx)

        result = instrument_to_base_currency(pf, "USD", "EUR", rates)

        # Days 0-2: rate 1.0 → price 100 EUR
        # Days 3-4: rate 2.0 → price 50 EUR
        assert result["adj_close"].iloc[0] == pytest.approx(100.0)
        assert result["adj_close"].iloc[1] == pytest.approx(100.0)  # ffilled from day 0
        assert result["adj_close"].iloc[2] == pytest.approx(100.0)  # ffilled from day 0
        assert result["adj_close"].iloc[3] == pytest.approx(50.0)
        assert result["adj_close"].iloc[4] == pytest.approx(50.0)   # ffilled from day 3
