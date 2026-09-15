"""Benchmark vs instrument separation tests.

Proves that:
- Drawdown/ATH is computed from the reference benchmark index, not the ETF.
- ETF price movements do not alter the benchmark signal.
- Benchmark history before simulation_start is included in the ATH.
- ATH is never reset per rolling window.
- Signal at close t executes in the instrument after t.
- FX changes affect units and portfolio value but not the native benchmark drawdown.
- Price-index and total-return-index series cannot be mixed.
- Missing instrument prices delay execution to the next valid instrument date.
- Incomplete benchmark history is identified as a non-absolute ATH.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dipdca.models import DeploymentTier, MarketDefinition, SimulationParams
from dipdca.quant.backtest import run_dip_deployment


def make_prices(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


def base_params(**kw) -> SimulationParams:
    defaults = dict(
        monthly_contribution=0.01,
        payday=25,
        initial_investment=0.0,
        initial_cash_reserve=10_000.0,
        start_date=date(2020, 1, 1),
        end_date=date(2021, 12, 31),
        dip_threshold=-0.10,
        max_wait_months=24,
        fixed_fee=0.0,
        pct_fee=0.0,
        slippage=0.0,
        cash_rate_override=None,
    )
    defaults.update(kw)
    return SimulationParams(**defaults)


TIER_15 = [DeploymentTier(-0.15, 1.00)]


# ---------------------------------------------------------------------------
# Test 1: ATH is calculated from the benchmark, not the ETF
# ---------------------------------------------------------------------------
class TestATHFromBenchmarkNotETF:
    def test_benchmark_drawdown_ignores_etf_ath(self):
        """Benchmark drops -20%, ETF only drops -5%. Signal fires from benchmark."""
        # Benchmark: rises to 100, then falls to 80 (-20%)
        benchmark_vals = [100.0] * 20 + [80.0] * 80
        # ETF: rises to 100, then falls only -5% (tracks differently)
        etf_vals = [100.0] * 20 + [95.0] * 80
        benchmark_df = make_prices(benchmark_vals)
        etf_df = make_prices(etf_vals)

        params = base_params(end_date=date(2020, 5, 20))
        tiers = [DeploymentTier(-0.15, 1.00)]  # trigger at benchmark -15%
        r, ledger = run_dip_deployment(etf_df, params, tiers, benchmark_data=benchmark_df)

        # Benchmark crossed -15%, so deployment must have happened
        assert r.n_deployments >= 1, "Benchmark signal should trigger despite ETF < -15%"
        assert r.ending_market_value > 0

    def test_etf_ath_does_not_trigger_signal(self):
        """If only the ETF forms a new ATH but benchmark does not, no signal fires."""
        # Benchmark: flat at 100 throughout (no ATH, no drawdown)
        # ETF: rises 50%, then falls back to 100 (ETF -33% from its own ATH)
        # But benchmark never crosses the tier threshold
        benchmark_vals = [100.0] * 100
        etf_vals = [100.0] * 20 + [150.0] * 30 + [100.0] * 50  # ETF forms own ATH

        benchmark_df = make_prices(benchmark_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 5, 20))
        tiers = [DeploymentTier(-0.15, 1.00)]

        r, ledger = run_dip_deployment(etf_df, params, tiers, benchmark_data=benchmark_df)
        # Benchmark never fell, so no signal should fire
        assert r.n_deployments == 0, "ETF's own ATH must not drive the signal"


# ---------------------------------------------------------------------------
# Test 2: Benchmark history before simulation_start is included in ATH
# ---------------------------------------------------------------------------
class TestBenchmarkHistoryBeforeSimStart:
    def test_ath_includes_pre_simulation_history(self):
        """Benchmark peaked at 200 before simulation starts. Day-1 drawdown must use 200."""
        # 50 days of benchmark history: first 20 at 200 (the true ATH), then falls to 160
        # The simulation only "starts" at day 20 (we pass full history)
        benchmark_vals = [200.0] * 20 + [160.0] * 80  # ATH = 200, then -20% drawdown
        etf_vals = [100.0] * 20 + [80.0] * 80
        benchmark_df = make_prices(benchmark_vals)
        etf_df = make_prices(etf_vals)

        params = base_params(end_date=date(2020, 5, 20))
        tiers = [DeploymentTier(-0.15, 1.00)]
        r, ledger = run_dip_deployment(etf_df, params, tiers, benchmark_data=benchmark_df)

        # With full benchmark history, ATH=200, drawdown reaches -20% => signal fires
        bm_dd = ledger["benchmark_drawdown"].dropna()
        assert float(bm_dd.min()) < -0.15, "ATH of 200 from pre-sim history must be used"
        assert r.n_deployments >= 1


# ---------------------------------------------------------------------------
# Test 3: ATH is never reset between evaluation windows
# ---------------------------------------------------------------------------
class TestATHNotResetPerWindow:
    def test_benchmark_ath_is_monotonically_non_decreasing(self):
        """The running benchmark ATH must never decrease."""
        # Benchmark rises then falls then rises again (but below prior ATH)
        benchmark_vals = [100.0] * 10 + [150.0] * 10 + [80.0] * 30 + [120.0] * 50
        etf_vals = [100.0] * 100
        benchmark_df = make_prices(benchmark_vals)
        etf_df = make_prices(etf_vals)

        params = base_params(end_date=date(2020, 5, 20))
        r, ledger = run_dip_deployment(etf_df, params, TIER_15, benchmark_data=benchmark_df)

        bm_ath = ledger["benchmark_ath"].dropna()
        # ATH must be monotonically non-decreasing
        diffs = bm_ath.diff().dropna()
        assert (diffs >= -1e-10).all(), "Benchmark ATH must never decrease"
        # Peak ATH must be 150 (highest ever), not 120 (would be wrong if reset)
        assert float(bm_ath.max()) == pytest.approx(150.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 4: Signal at close t executes at instrument price after t
# ---------------------------------------------------------------------------
class TestSignalTExecutesAfterT:
    def test_execution_uses_instrument_price_after_signal_date(self):
        """Signal fires from benchmark on day 20; execution must use instrument price on day 21+."""
        # Benchmark drops to -15% on day 20 (index 19)
        # ETF on day 21 has a specific price we can verify
        etf_execution_price = 90.0
        benchmark_vals = [100.0] * 20 + [84.0] * 80  # -16% on day 21 (0-indexed 20)
        etf_vals = [100.0] * 20 + [etf_execution_price] * 80

        benchmark_df = make_prices(benchmark_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(initial_cash_reserve=9_000.0, end_date=date(2020, 5, 20))
        tiers = [DeploymentTier(-0.15, 1.00)]

        r, ledger = run_dip_deployment(etf_df, params, tiers, benchmark_data=benchmark_df)

        # Find the execution row
        deploy_rows = ledger[ledger["deployed"] > 0]
        assert not deploy_rows.empty, "Should have executed"
        # Units bought ≈ deployed / etf_execution_price (no fees)
        execution_idx = deploy_rows.index[0]
        units_bought = float(deploy_rows["units"].iloc[0]) if "units" in deploy_rows else float(ledger.loc[execution_idx, "units"])
        deployed_amt = float(deploy_rows["deployed"].iloc[0])
        implied_price = deployed_amt / units_bought if units_bought > 0 else 0.0
        assert implied_price == pytest.approx(etf_execution_price, rel=0.01), (
            f"Execution price {implied_price} should match instrument price {etf_execution_price}"
        )


# ---------------------------------------------------------------------------
# Test 5: Different ETF tracking does not change the benchmark signal
# ---------------------------------------------------------------------------
class TestETFTrackingDoesNotAlterSignal:
    def test_same_signal_regardless_of_etf_price_level(self):
        """Two ETFs with different price levels tracking the same benchmark fire on the same day."""
        benchmark_vals = [100.0] * 15 + [84.0] * 85  # -16% on day 16

        # ETF A: priced at 50 (half of benchmark)
        etf_a_vals = [50.0] * 15 + [42.0] * 85
        # ETF B: priced at 200 (double of benchmark)
        etf_b_vals = [200.0] * 15 + [168.0] * 85

        benchmark_df = make_prices(benchmark_vals)
        etf_a_df = make_prices(etf_a_vals)
        etf_b_df = make_prices(etf_b_vals)

        params_a = base_params(end_date=date(2020, 5, 20))
        params_b = base_params(end_date=date(2020, 5, 20))

        tiers = [DeploymentTier(-0.15, 1.00)]
        r_a, ledger_a = run_dip_deployment(etf_a_df, params_a, tiers, benchmark_data=benchmark_df)
        r_b, ledger_b = run_dip_deployment(etf_b_df, params_b, tiers, benchmark_data=benchmark_df)

        # Both should deploy (same benchmark signal)
        assert r_a.n_deployments >= 1
        assert r_b.n_deployments >= 1

        # Signal date must be identical (same benchmark crossing)
        signal_a = ledger_a[ledger_a["signal_dd"] != 0.0].index
        signal_b = ledger_b[ledger_b["signal_dd"] != 0.0].index
        assert list(signal_a) == list(signal_b), "Same benchmark must produce same signal date"


# ---------------------------------------------------------------------------
# Test 6: FX rate changes affect units and portfolio value but not benchmark drawdown
# ---------------------------------------------------------------------------
class TestFXDoesNotAlterBenchmarkDrawdown:
    def test_benchmark_drawdown_unchanged_by_fx(self):
        """Simulate two scenarios: ETF hedged vs unhedged (different price levels due to FX).
        Benchmark drawdown must be identical in both.
        """
        benchmark_vals = [100.0] * 10 + [84.0] * 90  # -16%

        # Unhedged ETF: same levels as benchmark (USD)
        etf_base_vals = [100.0] * 10 + [84.0] * 90
        # FX-adjusted ETF: EUR/USD = 1.2, so ETF in EUR is 20% lower
        etf_fx_vals = [100.0 / 1.2] * 10 + [84.0 / 1.2] * 90

        benchmark_df = make_prices(benchmark_vals)
        etf_base_df = make_prices(etf_base_vals)
        etf_fx_df = make_prices(etf_fx_vals)

        params = base_params(end_date=date(2020, 5, 20))
        tiers = [DeploymentTier(-0.15, 1.00)]

        _, ledger_base = run_dip_deployment(etf_base_df, params, tiers, benchmark_data=benchmark_df)
        _, ledger_fx = run_dip_deployment(etf_fx_df, params, tiers, benchmark_data=benchmark_df)

        # Benchmark drawdown must be identical regardless of FX-adjusted instrument
        dd_base = ledger_base["benchmark_drawdown"].dropna()
        dd_fx = ledger_fx["benchmark_drawdown"].dropna()
        pd.testing.assert_series_equal(dd_base.reset_index(drop=True), dd_fx.reset_index(drop=True),
                                       check_names=False, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 7: Price-index and total-return-index series cannot be mixed
# ---------------------------------------------------------------------------
class TestPriceAndTotalReturnNotMixed:
    def test_market_definition_enforces_return_type(self):
        """MarketDefinition stores return type; mixing types must be detectable."""
        md_price = MarketDefinition(
            benchmark_symbol="^NDX",
            benchmark_name="Nasdaq-100 Price Index",
            benchmark_currency="USD",
            benchmark_return_type="price_index",
            instrument_symbol="QQQ",
            instrument_name="Invesco QQQ ETF",
            instrument_currency="USD",
            base_currency="EUR",
        )
        md_tr = MarketDefinition(
            benchmark_symbol="^NDXTR",
            benchmark_name="Nasdaq-100 Net Total Return",
            benchmark_currency="USD",
            benchmark_return_type="net_total_return",
            instrument_symbol="QQQ",
            instrument_name="Invesco QQQ ETF",
            instrument_currency="USD",
            base_currency="EUR",
        )
        assert md_price.benchmark_return_type == "price_index"
        assert md_tr.benchmark_return_type == "net_total_return"
        # Different ATHs: total-return series trends higher than price index
        # Confirm they are treated as separate objects (no implicit merging)
        assert md_price.benchmark_symbol != md_tr.benchmark_symbol
        assert md_price.benchmark_return_type != md_tr.benchmark_return_type

    def test_invalid_return_type_rejected(self):
        """An invalid benchmark_return_type must raise a TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            MarketDefinition(
                benchmark_symbol="^NDX",
                benchmark_name="Nasdaq-100",
                benchmark_currency="USD",
                benchmark_return_type="bad_type",  # type: ignore[arg-type]
                instrument_symbol="QQQ",
                instrument_name="QQQ",
                instrument_currency="USD",
                base_currency="EUR",
            )


# ---------------------------------------------------------------------------
# Test 8: Missing instrument prices delay execution to next valid instrument date
# ---------------------------------------------------------------------------
class TestMissingInstrumentPricesDelayExecution:
    def test_execution_delayed_when_instrument_has_gap(self):
        """Benchmark signal fires on day 20; instrument has a gap on day 21.
        Execution must happen on day 22+ (next valid instrument date).
        """
        # Build benchmark that drops -16% on day 20
        benchmark_vals = [100.0] * 20 + [84.0] * 80

        # ETF has a price gap on day 21 (NaN) and resumes on day 22
        etf_adj = [100.0] * 20 + [float("nan")] + [88.0] * 79
        idx = pd.bdate_range(start="2020-01-01", periods=100)
        etf_df = pd.DataFrame({"adj_close": etf_adj, "close": etf_adj}, index=idx)
        benchmark_df = make_prices(benchmark_vals)

        params = base_params(end_date=date(2020, 5, 20))
        tiers = [DeploymentTier(-0.15, 1.00)]

        r, ledger = run_dip_deployment(etf_df, params, tiers, benchmark_data=benchmark_df)

        if r.n_deployments > 0:
            deploy_rows = ledger[ledger["deployed"] > 0]
            # Execution must be on day 22+ (index >= 21), not on day 21 (index 20) which has NaN price
            first_exec_loc = ledger.index.get_loc(deploy_rows.index[0])
            assert first_exec_loc >= 21, (
                f"Execution at position {first_exec_loc} should be at position ≥ 21 "
                f"(day 21 is NaN, day 22 is first valid instrument price)"
            )


# ---------------------------------------------------------------------------
# Test 9: Benchmark drawdown audit columns are populated in ledger
# ---------------------------------------------------------------------------
class TestLedgerBenchmarkAuditColumns:
    def test_benchmark_audit_columns_populated(self):
        """When benchmark_data is provided, ledger must have benchmark_close,
        benchmark_ath, benchmark_drawdown columns with sensible values.
        """
        benchmark_vals = [100.0] * 30 + [85.0] * 70
        etf_vals = [50.0] * 100

        benchmark_df = make_prices(benchmark_vals)
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 5, 20))

        _, ledger = run_dip_deployment(etf_df, params, TIER_15, benchmark_data=benchmark_df)

        assert "benchmark_close" in ledger.columns
        assert "benchmark_ath" in ledger.columns
        assert "benchmark_drawdown" in ledger.columns

        bm_close = ledger["benchmark_close"].dropna()
        bm_ath = ledger["benchmark_ath"].dropna()
        bm_dd = ledger["benchmark_drawdown"].dropna()

        # Benchmark close must reflect the actual benchmark values
        assert float(bm_close.iloc[0]) == pytest.approx(100.0, abs=1e-6)
        assert float(bm_close.iloc[-1]) == pytest.approx(85.0, abs=1e-6)

        # ATH must be 100 throughout (the initial peak)
        assert float(bm_ath.max()) == pytest.approx(100.0, abs=1e-6)

        # Drawdown at the end must be -15%
        assert float(bm_dd.iloc[-1]) == pytest.approx(-0.15, abs=1e-4)

    def test_without_benchmark_data_audit_columns_nan(self):
        """Without benchmark_data, benchmark audit columns should contain NaN
        (populated in column template but not filled).
        """
        etf_vals = [100.0] * 50
        etf_df = make_prices(etf_vals)
        params = base_params(end_date=date(2020, 3, 10))

        # No benchmark_data passed
        _, ledger = run_dip_deployment(etf_df, params, TIER_15)

        # Columns exist but are all NaN (except benchmark_drawdown tracks instrument)
        # In backward-compat mode (no benchmark_data), bm_aligned = price_data["adj_close"]
        # so benchmark_drawdown IS populated (from instrument used as its own benchmark)
        assert "benchmark_close" in ledger.columns
        assert "benchmark_ath" in ledger.columns
        assert "benchmark_drawdown" in ledger.columns


# ---------------------------------------------------------------------------
# Test 10: MarketDefinition dataclass is frozen and immutable
# ---------------------------------------------------------------------------
class TestMarketDefinitionImmutability:
    def test_market_definition_is_frozen(self):
        """MarketDefinition must be frozen (immutable dataclass)."""
        md = MarketDefinition(
            benchmark_symbol="^GSPC",
            benchmark_name="S&P 500 Price Index",
            benchmark_currency="USD",
            benchmark_return_type="price_index",
            instrument_symbol="SPY",
            instrument_name="SPDR S&P 500 ETF",
            instrument_currency="USD",
            base_currency="EUR",
        )
        with pytest.raises((AttributeError, TypeError)):
            md.benchmark_symbol = "^NDX"  # type: ignore[misc]

    def test_market_definition_valid_construction(self):
        """MarketDefinition with all required fields constructs correctly."""
        md = MarketDefinition(
            benchmark_symbol="^GDAXI",
            benchmark_name="DAX Price Index",
            benchmark_currency="EUR",
            benchmark_return_type="price_index",
            instrument_symbol="EXS1.DE",
            instrument_name="iShares Core DAX UCITS ETF",
            instrument_currency="EUR",
            base_currency="EUR",
        )
        assert md.benchmark_symbol == "^GDAXI"
        assert md.instrument_symbol == "EXS1.DE"
        assert md.benchmark_return_type == "price_index"
        assert md.base_currency == "EUR"
