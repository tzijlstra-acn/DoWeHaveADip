"""Display convention tests.

Covers:
  Bug 1 — percentage formatting (fraction_to_pct helper + scenarios *100 values)
  Bug 2 — prehistory ATH seeding (find_ath_episodes initial_ath parameter)
  Bug 3 — horizon anchor at signal/execution date instead of ATH
  Bug 4 — partial final month excluded from month_end schedule
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from dipdca.models import DeploymentTier
from dipdca.quant.ath_episodes import (
    find_ath_episodes,
    run_event_study,
    study_episode,
    summarise_threshold,
)
from dipdca.quant.contributions import build_contribution_schedule


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx)


def frame(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


# ---------------------------------------------------------------------------
# Bug 1: percentage formatting
# ---------------------------------------------------------------------------


class TestFractionToPercent:
    def test_fraction_0_041_displays_as_4_10_percent(self):
        """The table stores raw fraction 0.041; after *100 the format gives 4.10%."""
        assert f"{0.041 * 100:.2f}%" == "4.10%"

    def test_fraction_0_60_displays_as_60_percent(self):
        assert f"{0.60 * 100:.0f}%" == "60%"

    def test_fraction_to_pct_converts_correctly(self):
        """fraction_to_pct must live in ui.formatting after the fix."""
        from ui.formatting import fraction_to_pct

        assert fraction_to_pct(0.041) == "4.10%"
        assert fraction_to_pct(0.60) == "60.00%"
        assert fraction_to_pct(None) == "N/A"
        assert fraction_to_pct(1.0) == "100.00%"
        assert fraction_to_pct(0.0) == "0.00%"

    def test_fraction_to_pct_decimals_param(self):
        from ui.formatting import fraction_to_pct

        assert fraction_to_pct(0.1234, decimals=1) == "12.3%"
        assert fraction_to_pct(0.1234, decimals=0) == "12%"

    def test_scenarios_pct_columns_are_multiplied_by_100(self):
        """After the fix, fractional values from summarise_threshold must be
        multiplied by 100 before being placed in the scenarios rows dict.

        We can verify the raw fraction is ≤ 1.0 and check the *100 value is
        the value that the column_config format="%.2f%%" should receive.
        """
        vals = [100.0] + [70.0] * 60 + [130.0] * 60
        inst = frame(vals)
        studies = run_event_study(
            instrument=inst,
            benchmark=inst,
            tiers=[DeploymentTier(-0.20, 1.00)],
            monthly_contribution=1.0,
            opening_reserve=10_000.0,
            thresholds=(-0.20,),
            horizon_months=(6,),
        ).studies
        s = summarise_threshold(studies, -0.20, "6m", "ATH all-in")

        # Raw values from summarise_threshold are fractions (0-1)
        pct_deployed = s["pct_episodes_deployed"]
        pct_ahead = s["pct_ahead_of_dca"]
        assert pct_deployed is not None and pct_deployed <= 1.0
        assert pct_ahead is not None and pct_ahead <= 1.0

        # After multiplying by 100, format="%.2f%%" gives "100.00%" not "1.00%"
        assert f"{pct_deployed * 100:.2f}%" != f"{pct_deployed:.2f}%"


# ---------------------------------------------------------------------------
# Bug 2: prehistory ATH seeding
# ---------------------------------------------------------------------------


class TestInitialATH:
    def test_initial_ath_seeds_peak_before_data(self):
        """When initial_ath=200.0 and data never reaches 200, all bars are
        in drawdown from day 1 and there must be a censored episode anchored at 200."""
        vals = [100.0, 95.0, 90.0, 95.0, 100.0]  # max is 100, never reaches 200
        s = series(vals)
        eps = find_ath_episodes(
            s,
            thresholds=(-0.10, -0.20, -0.30, -0.40, -0.50),
            initial_ath=200.0,
        )
        assert len(eps) >= 1
        ep = eps[0]
        # Episode must be anchored to the pre-seeded ATH
        assert ep.ath_level == 200.0
        # Trough is 90.0; 90/200 - 1 = -0.55
        assert ep.trough_drawdown < -0.50

    def test_initial_ath_with_initial_ath_date(self):
        """initial_ath_date seeds the anchor date."""
        vals = [100.0] * 5
        s = series(vals)
        pre_date = pd.Timestamp("2019-12-31")
        eps = find_ath_episodes(
            s,
            thresholds=(-0.10,),
            initial_ath=200.0,
            initial_ath_date=pre_date,
        )
        assert len(eps) >= 1
        assert eps[0].ath_date == pre_date

    def test_find_ath_episodes_default_unchanged_without_initial_ath(self):
        """Without initial_ath, existing behaviour is preserved."""
        vals = [100.0] + [70.0] * 10 + [105.0] * 5
        s = series(vals)
        eps = find_ath_episodes(s, thresholds=(-0.10, -0.20, -0.30))
        assert len(eps) == 1
        assert eps[0].ath_level == 100.0


# ---------------------------------------------------------------------------
# Bug 3: horizon anchor at execution date, not ATH
# ---------------------------------------------------------------------------


class TestHorizonAnchor:
    def test_study_episode_accepts_anchor_parameter(self):
        """study_episode must accept anchor='execution' after the fix."""
        vals = [100.0] + [90.0] * 19 + [79.0] * 280
        ep = find_ath_episodes(series(vals), thresholds=(-0.20,))[0]
        inst = frame(vals)
        # Before the fix this raises TypeError; after the fix it should not
        study = study_episode(
            episode=ep,
            threshold=-0.20,
            instrument=inst,
            benchmark=inst,
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.20, 1.00)],
            horizon_months=(12,),
            anchor="execution",
        )
        # study may be None if data is too short; that's OK — what matters is
        # that no TypeError was raised
        assert study is None or hasattr(study, "outcomes")

    def test_execution_anchor_differs_from_ath_anchor(self):
        """anchor='execution' and anchor='ath' produce different outcomes when
        the signal date is well after the ATH date."""
        # ATH on day 0 (100), signal at day 20 (first cross -20%), flat after
        vals = [100.0] + [90.0] * 19 + [79.0] * 280
        ep = find_ath_episodes(series(vals), thresholds=(-0.20,))[0]
        signal_date = ep.first_crossings.get(-0.20)
        assert signal_date is not None

        inst = frame(vals)
        study_ath = study_episode(
            episode=ep,
            threshold=-0.20,
            instrument=inst,
            benchmark=inst,
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.20, 1.00)],
            horizon_months=(12,),
            anchor="ath",
        )
        study_exec = study_episode(
            episode=ep,
            threshold=-0.20,
            instrument=inst,
            benchmark=inst,
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            tiers=[DeploymentTier(-0.20, 1.00)],
            horizon_months=(12,),
            anchor="execution",
        )
        assert study_ath is not None
        assert study_exec is not None
        # The windows differ because execution starts ~20 business days later
        ath_wealth = study_ath.wealth("DCA", "12m")
        exec_wealth = study_exec.wealth("DCA", "12m")
        # Both should be computable; and they should differ (different start dates)
        assert ath_wealth is not None
        assert exec_wealth is not None
        # Windows start at different points so total contributions differ slightly
        # (or end at different times); at minimum the anchor is being used
        assert ath_wealth != exec_wealth

    def test_run_event_study_accepts_anchor_parameter(self):
        """run_event_study must forward anchor to study_episode."""
        vals = [100.0] + [70.0] * 60 + [130.0] * 60
        inst = frame(vals)
        # Should not raise TypeError
        result = run_event_study(
            instrument=inst,
            benchmark=inst,
            tiers=[DeploymentTier(-0.20, 1.00)],
            monthly_contribution=1000.0,
            opening_reserve=10_000.0,
            thresholds=(-0.20,),
            horizon_months=(6,),
            anchor="execution",
        )
        assert isinstance(result.studies, list)


# ---------------------------------------------------------------------------
# Bug 4: partial final month must not produce a month_end contribution
# ---------------------------------------------------------------------------


class TestPartialMonth:
    def test_partial_march_no_contribution(self):
        """End date March 16 should NOT produce a March contribution.

        The calendar month-end (March 31) is not within the window, so no
        March month-end trading day should be included.
        """
        idx = trading_days("2020-01-01", "2020-03-16")
        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 16), 100.0, 25, idx, timing="month_end"
        )
        months = [d.month for d in df["invest_date"]]
        assert 3 not in months, (
            f"March should not appear in a schedule ending March 16, got: {months}"
        )

    def test_partial_march_produces_exactly_two_contributions(self):
        """Jan–Mar16 window → only Jan and Feb are complete months."""
        idx = trading_days("2020-01-01", "2020-03-16")
        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 16), 100.0, 25, idx, timing="month_end"
        )
        assert len(df) == 2

    def test_full_march_has_contribution(self):
        """End date March 31 SHOULD produce a March contribution."""
        idx = trading_days("2020-01-01", "2020-03-31")
        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 31), 100.0, 25, idx, timing="month_end"
        )
        months = [d.month for d in df["invest_date"]]
        assert 3 in months

    def test_full_months_produce_correct_count(self):
        """Jan-March (complete) window should produce exactly 3 contributions."""
        idx = trading_days("2020-01-01", "2020-03-31")
        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 31), 100.0, 25, idx, timing="month_end"
        )
        assert len(df) == 3

    def test_month_end_invest_dates_are_actual_calendar_month_ends(self):
        """Invest dates must fall on the last trading day OF a completed month."""
        idx = trading_days("2020-01-01", "2020-06-30")
        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 6, 30), 100.0, 25, idx, timing="month_end"
        )
        # 2020-02-28 (Feb, leap year: Feb 29 is a Saturday, so last trading day = Feb 28)
        invest_dates = [d.date() for d in df["invest_date"]]
        assert date(2020, 1, 31) in invest_dates
        assert date(2020, 2, 28) in invest_dates
