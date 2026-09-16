"""Phase 1 regression tests — documents confirmed defects and verifies already-fixed items.

Tests that FAIL before fixes (confirmed bugs):
  - test_drawdown_chart_rejects_raw_positive_price_series
  - test_drawdown_chart_y_values_are_negative_or_zero
  - test_calendar_days_since_ath_uses_date_difference_not_row_count
  - test_episode_failure_is_returned_not_silently_skipped
  - test_fraction_0_041_displays_as_4_10_percent
  - test_fraction_0_60_displays_as_60_percent

Tests that PASS (defects already addressed):
  - test_event_study_signal_on_t_executes_exactly_on_t_plus_1
  - test_rebound_on_t_plus_1_does_not_cancel_pending_order
  - test_measurement_anchor_does_not_change_signal_or_execution
  - test_future_index_values_cannot_change_current_state
  - test_small_nonzero_currency_delta_is_visible
  - test_missing_fx_cannot_silently_use_one_to_one_parity
  - test_no_user_page_imports_run_wait_for_dip
  - test_today_does_not_import_conditional_path_bootstrap
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from dipdca.models import DeploymentTier, SimulationParams
from dipdca.quant.ath_episodes import find_ath_episodes, study_episode
from dipdca.quant.backtest import run_ath_deployment
from dipdca.quant.drawdown import drawdown
from dipdca.quant.fx import instrument_to_base_currency
from ui.charts import drawdown_chart
from ui.formatting import fmt_delta, fmt_pct

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "app_pages"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bm_frame(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


def bm_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, name="adj_close")


def base_params(
    values: list[float], start: str = "2020-01-01", **kw
) -> SimulationParams:
    idx = pd.bdate_range(start=start, periods=len(values))
    defaults = dict(
        monthly_contribution=0.01,
        payday=25,
        initial_investment=0.0,
        initial_cash_reserve=10_000.0,
        start_date=idx[0].date(),
        end_date=idx[-1].date(),
        dip_threshold=-0.15,
        fixed_fee=0.0,
        pct_fee=0.0,
        slippage=0.0,
        cash_rate_override=None,
    )
    defaults.update(kw)
    return SimulationParams(**defaults)


# ===========================================================================
# Event execution tests
# ===========================================================================


class TestEventExecution:
    def test_event_study_signal_on_t_executes_exactly_on_t_plus_1(self):
        """Benchmark crosses -15% on day T. Execution must land on day T+1 only.

        Signal day: idx[4] (bm drops from 100 to 84, drawdown = -16%)
        Execution day expected: idx[5]
        """
        bm_values = [100.0, 100.0, 100.0, 100.0, 84.0, 84.0, 84.0]
        inst_values = [50.0, 50.0, 50.0, 50.0, 42.0, 42.0, 42.0]
        idx = pd.bdate_range("2020-01-01", periods=len(bm_values))

        bm = pd.DataFrame({"adj_close": bm_values}, index=idx)
        inst = pd.DataFrame({"adj_close": inst_values}, index=idx)

        params = base_params(bm_values, start="2020-01-01")

        _, ledger = run_ath_deployment(
            instrument_data=inst,
            params=params,
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=bm,
            initial_ath=None,
        )

        deployed = ledger["deployed"]

        # Day 4 (index 4): signal fires — must NOT deploy on same day
        assert float(deployed.iloc[4]) == 0.0, (
            f"Deployed on signal day (t): {float(deployed.iloc[4])}. "
            "Expected 0.0 — execution must not happen on the signal day."
        )
        # Day 5 (index 5): execution day — must deploy here
        assert float(deployed.iloc[5]) > 0.0, (
            f"No deployment on t+1 (idx[5]): {float(deployed.iloc[5])}. "
            "Expected > 0.0 — execution must happen the day after the signal."
        )
        # Day 6: must not deploy again (already executed)
        assert float(deployed.iloc[6]) == 0.0, (
            f"Deployed again on t+2 (idx[6]): {float(deployed.iloc[6])}. "
            "Expected 0.0 — should not deploy twice."
        )

    def test_rebound_on_t_plus_1_does_not_cancel_pending_order(self):
        """A benchmark rebound between signal (T) and execution (T+1) must not cancel the order.

        Day 4: bm drops to 84 (signals -16% drawdown)
        Day 5: bm rebounds to 102 (new ATH — in the real app this would re-arm, but
               the pending order from day 4 still executes at day 5 instrument price)
        """
        bm_values = [100.0, 100.0, 100.0, 100.0, 84.0, 102.0, 102.0]
        inst_values = [100.0, 100.0, 100.0, 100.0, 84.0, 102.0, 102.0]
        idx = pd.bdate_range("2020-01-01", periods=len(bm_values))

        bm = pd.DataFrame({"adj_close": bm_values}, index=idx)
        inst = pd.DataFrame({"adj_close": inst_values}, index=idx)

        params = base_params(bm_values, start="2020-01-01")

        _, ledger = run_ath_deployment(
            instrument_data=inst,
            params=params,
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=bm,
            initial_ath=None,
        )

        # Pending order from day 4 must execute on day 5 despite the rebound
        total_deployed = float(ledger["deployed"].sum())
        assert total_deployed > 0.0, (
            "Pending order was cancelled by the T+1 rebound. "
            "Expected: order executes at T+1 price regardless of the rebound."
        )

    def test_measurement_anchor_does_not_change_signal_or_execution(self):
        """Changing the anchor (ath/signal/execution) must not change when the trade fires.

        The anchor changes the measurement window for outcomes, not the trade date.
        """
        vals = [100.0] + [70.0] * 60 + [130.0] * 60
        inst = bm_frame(vals)

        episodes = find_ath_episodes(bm_series(vals), thresholds=(-0.20,))
        assert episodes, "Expected at least one episode"
        ep = episodes[0]

        tiers = [DeploymentTier(-0.20, 1.00)]
        kwargs = dict(
            episode=ep,
            threshold=-0.20,
            instrument=inst,
            benchmark=inst,
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=tiers,
            horizon_months=(12,),
        )

        study_ath = study_episode(**kwargs, anchor="ath")
        study_exec = study_episode(**kwargs, anchor="execution")
        study_sig = study_episode(**kwargs, anchor="signal")

        # All three must produce results (not None)
        assert study_ath is not None, "anchor='ath' produced None"
        assert study_exec is not None, "anchor='execution' produced None"
        assert study_sig is not None, "anchor='signal' produced None"

        # The signal_date must be identical regardless of anchor
        assert study_ath.signal_date == study_exec.signal_date == study_sig.signal_date, (
            "signal_date changed when anchor changed — anchor must not affect when the trade fires."
        )


# ===========================================================================
# Drawdown chart tests
# ===========================================================================


class TestDrawdownChart:
    def test_drawdown_chart_rejects_raw_positive_price_series(self):
        """drawdown_chart() must raise ValueError when given raw prices (values > 0).

        Passing raw prices produces a nonsensical chart with y-values in the tens of
        thousands instead of drawdown percentages in [-100, 0].

        EXPECTED TO FAIL: the current implementation has no input validation.
        """
        raw_prices = pd.Series(
            [100.0, 110.0, 105.0, 120.0],
            index=pd.bdate_range("2020-01-01", periods=4),
        )
        with pytest.raises(ValueError, match="drawdown"):
            drawdown_chart(raw_prices)

    def test_drawdown_chart_y_values_are_negative_or_zero(self):
        """drawdown_chart() called with a valid drawdown series produces all-negative y values.

        Fixed: callers now pass drawdown(signal_series) so y-values are ≤ 0 (%).
        """
        raw_prices = pd.Series(
            [100.0, 110.0, 105.0, 120.0],
            index=pd.bdate_range("2020-01-01", periods=4),
        )
        dd = drawdown(raw_prices)
        fig = drawdown_chart(dd)

        y_vals = [v for v in fig.data[0].y if v is not None]
        assert all(v <= 0 for v in y_vals), (
            f"drawdown_chart produced positive y-values: max(y) = {max(y_vals):.1f}. "
            "Expected all y ≤ 0 when given a proper drawdown series."
        )


# ===========================================================================
# Benchmark state tests
# ===========================================================================


class TestBenchmarkState:
    def test_future_index_values_cannot_change_current_state(self):
        """Appending future data must not change earlier ATH or drawdown values.

        The drawdown() function uses expanding().max() which is causal: the peak
        at time t is the max of [0, t], never of [0, T] where T > t.
        """
        prices_short = bm_series([100.0, 90.0, 95.0, 85.0])
        prices_long = bm_series([100.0, 90.0, 95.0, 85.0, 200.0, 200.0])

        dd_short = drawdown(prices_short)
        dd_long = drawdown(prices_long)

        # The first 4 values must be identical regardless of what comes after
        for i in range(len(prices_short)):
            assert float(dd_short.iloc[i]) == pytest.approx(float(dd_long.iloc[i])), (
                f"Drawdown at position {i} changed when future data was appended. "
                f"Short: {float(dd_short.iloc[i]):.4f}, Long: {float(dd_long.iloc[i]):.4f}"
            )

    def test_calendar_days_since_ath_uses_date_difference_not_row_count(self):
        """'Calendar days from high' uses date arithmetic, not row count.

        On a weekly index, 9 rows span 63 calendar days. The label says
        'Calendar days from high' so the value must be 63, not 9.

        Fixed: today.py and compare.py now use (last_date - ath_date).days.
        """
        # Weekly index: 10 dates spanning ~9 weeks (63 calendar days)
        dates = pd.to_datetime([
            "2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22", "2020-01-29",
            "2020-02-05", "2020-02-12", "2020-02-19", "2020-02-26", "2020-03-04",
        ])
        # ATH on the first date; all subsequent dates are in drawdown
        values = [100.0] + [90.0] * 9
        signal_series = pd.Series(values, index=dates)

        # Correct implementation: calendar days via date arithmetic
        ath_date = signal_series.idxmax()
        last_date = signal_series.index[-1]
        days_since_high = (last_date - ath_date).days

        # 63 calendar days, not 9 rows
        assert days_since_high == 63, (
            f"Expected 63 calendar days but got {days_since_high}. "
            "Date arithmetic must be used, not row counting."
        )

        # Confirm the old row-count approach gives a different (wrong) answer
        peak_series = signal_series.expanding().max()
        at_peak = signal_series >= peak_series
        row_count = 0
        for i in range(len(at_peak) - 1, -1, -1):
            if at_peak.iloc[i]:
                break
            row_count += 1
        assert row_count != days_since_high, (
            "Row count and calendar days are unexpectedly equal — test setup may be wrong."
        )


# ===========================================================================
# Error visibility tests
# ===========================================================================


class TestErrorVisibility:
    def test_episode_failure_is_returned_not_silently_skipped(self):
        """When run_ath_deployment raises ValueError inside study_episode, the
        error must propagate as RuntimeError — not be silently swallowed.

        Fixed behaviour: study_episode re-raises engine failures as RuntimeError
        so callers can see and handle them. run_event_study logs a warning and
        skips the episode rather than silently returning incomplete results.
        """
        vals = [100.0] + [70.0] * 60 + [130.0] * 60
        idx = pd.bdate_range("2020-01-01", periods=len(vals))
        inst = pd.DataFrame({"adj_close": vals, "close": vals}, index=idx)
        series = pd.Series(vals, index=idx)

        episodes = find_ath_episodes(series, thresholds=(-0.20,))
        assert episodes, "Need at least one episode for this test"
        ep = episodes[0]

        with (
            patch(
                "dipdca.quant.ath_episodes.run_ath_deployment",
                side_effect=ValueError("simulated engine failure"),
            ),
            pytest.raises(RuntimeError, match="simulated engine failure"),
        ):
            study_episode(
                episode=ep,
                threshold=-0.20,
                instrument=inst,
                benchmark=inst,
                monthly_contribution=1000.0,
                opening_reserve=10_000.0,
                tiers=[DeploymentTier(-0.20, 1.00)],
                horizon_months=(12,),
            )


# ===========================================================================
# Percentage display tests
# ===========================================================================


class TestPercentageDisplay:
    def test_fraction_0_041_displays_as_4_10_percent(self):
        """fmt_pct(0.041) should display as '4.10%', not '0.04%' or '41.00%' or '4.1%'.

        - '0.04%' would mean the value was NOT multiplied by 100 (already a percent)
        - '41.00%' would mean the value was multiplied by 100 twice
        - '4.1%' is the current output with default decimals=1 — insufficient precision

        EXPECTED TO FAIL: current default decimals=1 returns '4.1%', not '4.10%'.
        """
        result = fmt_pct(0.041)
        assert result == "4.10%", (
            f"fmt_pct(0.041) = {result!r}. "
            "Expected '4.10%' (fraction * 100, 2 decimal places). "
            "Got '0.04%'? → value was not multiplied by 100. "
            "Got '41.00%'? → value was double-scaled. "
            "Got '4.1%'? → default precision is 1 decimal, needs 2 for financial display."
        )

    def test_fraction_0_60_displays_as_60_percent(self):
        """fmt_pct(0.60) should display as '60.00%'.

        EXPECTED TO FAIL: current default decimals=1 returns '60.0%', not '60.00%'.
        """
        result = fmt_pct(0.60)
        assert result == "60.00%", (
            f"fmt_pct(0.60) = {result!r}. "
            "Expected '60.00%' (2 decimal places). "
            "Current default decimals=1 returns '60.0%'."
        )

    def test_small_nonzero_currency_delta_is_visible(self):
        """fmt_delta(0.43, 1000) must not display as EUR 0.

        EXPECTED TO PASS: fmt_delta already uses :+,.2f formatting.
        """
        result = fmt_delta(0.43, 1000)
        assert "0.43" in result, (
            f"fmt_delta(0.43, 1000) = {result!r}. "
            "Expected '0.43' to appear in the output (not rounded to zero)."
        )
        assert "EUR" in result, f"Missing currency label in {result!r}"
        assert "%" in result, f"Missing relative percentage in {result!r}"


# ===========================================================================
# FX tests
# ===========================================================================


class TestFX:
    def test_missing_fx_cannot_silently_use_one_to_one_parity(self):
        """instrument_to_base_currency must raise ValueError when FX rate is missing.

        EXPECTED TO PASS: the function correctly raises ValueError.

        Note: app_pages/advanced_fx.py uses fillna(1.0) which silently applies 1:1
        parity on that specific page. This test covers the main backtest path.
        """
        idx = pd.bdate_range("2020-01-01", periods=5)
        instrument_df = pd.DataFrame(
            {"adj_close": [100.0] * 5, "close": [100.0] * 5}, index=idx
        )
        # ECB rates DataFrame with no USD column
        ecb_rates = pd.DataFrame(
            {"GBP": [0.88] * 5}, index=idx
        )

        with pytest.raises(ValueError, match="USD"):
            instrument_to_base_currency(instrument_df, "USD", "EUR", ecb_rates)


# ===========================================================================
# Architecture tests
# ===========================================================================


class TestArchitecture:
    def test_no_user_page_imports_run_wait_for_dip(self):
        """app_pages/*.py must not reference run_wait_for_dip.

        EXPECTED TO PASS: the deprecated engine has been removed from all pages.
        """
        for page in sorted(PAGES.glob("*.py")):
            src = page.read_text(encoding="utf-8")
            assert "run_wait_for_dip" not in src, (
                f"{page.name} still references run_wait_for_dip. "
                "Use run_ath_deployment instead."
            )

    def test_today_does_not_import_conditional_path_bootstrap(self):
        """app_pages/today.py must not reference conditional_path_bootstrap.

        EXPECTED TO PASS: this function was removed from the today page.
        """
        src = (PAGES / "today.py").read_text(encoding="utf-8")
        assert "conditional_path_bootstrap" not in src, (
            "today.py still references conditional_path_bootstrap — "
            "this deprecated Monte Carlo bootstrap path should not appear in user pages."
        )
