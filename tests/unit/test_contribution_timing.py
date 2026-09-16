"""Contribution timing conventions.

The DCA-versus-timing comparison is specified against "invest the saved amount at
the end of every month", so ``month_end`` must be the default and must resolve to
a real trading close rather than a calendar date.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dipdca.models import SimulationParams
from dipdca.quant.contributions import (
    DEFAULT_CONTRIBUTION_TIMING,
    build_contribution_schedule,
    monthly_boundary_trading_days,
)


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


class TestDefaults:
    def test_module_default_is_month_end(self):
        assert DEFAULT_CONTRIBUTION_TIMING == "month_end"

    def test_simulation_params_default_is_month_end(self):
        params = SimulationParams(
            monthly_contribution=500.0,
            payday=25,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

        assert params.contribution_timing == "month_end"


class TestMonthEnd:
    def test_end_of_month_dca_uses_last_valid_trading_close(self):
        """2020-01-31 is a Friday; 2020-02-29 is a Saturday -> expect Feb 28."""
        idx = trading_days("2020-01-01", "2020-03-31")

        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 31), 1000.0, 25, idx, timing="month_end"
        )
        got = [d.date() for d in df["invest_date"]]

        assert got == [date(2020, 1, 31), date(2020, 2, 28), date(2020, 3, 31)]

    def test_invest_dates_are_always_real_trading_days(self):
        idx = trading_days("2021-01-01", "2021-12-31")

        df = build_contribution_schedule(
            date(2021, 1, 1), date(2021, 12, 31), 500.0, 25, idx, timing="month_end"
        )

        assert len(df) == 12
        for d in df["invest_date"]:
            assert d in idx

    def test_contribution_and_invest_date_coincide(self):
        """No arrival-then-invest lag under a month-boundary convention."""
        idx = trading_days("2020-01-01", "2020-06-30")

        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 6, 30), 100.0, 25, idx, timing="month_end"
        )

        assert list(df["contribution_date"]) == list(df["invest_date"])

    def test_month_end_excludes_partial_final_month(self):
        """A partial final month (end date March 16) must NOT produce a March
        contribution.  Only calendar-complete months are included — the March
        month-end (March 31) is outside the window so March is skipped."""
        idx = trading_days("2020-01-01", "2020-03-16")

        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 16), 100.0, 25, idx, timing="month_end"
        )
        got = [d.date() for d in df["invest_date"]]

        # Only January and February are complete months in this window
        assert len(got) == 2
        assert date(2020, 1, 31) in got
        assert date(2020, 2, 28) in got
        assert all(d <= date(2020, 3, 16) for d in got)


class TestMonthStart:
    def test_month_start_uses_first_valid_trading_close(self):
        """2020-03-01 is a Sunday -> expect Mar 2."""
        idx = trading_days("2020-01-01", "2020-03-31")

        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 31), 1000.0, 25, idx, timing="month_start"
        )
        got = [d.date() for d in df["invest_date"]]

        assert got == [date(2020, 1, 1), date(2020, 2, 3), date(2020, 3, 2)]


class TestFixedDay:
    def test_fixed_day_preserves_payday_behaviour(self):
        """2020-01-25 is a Saturday -> invests on Monday 2020-01-27."""
        idx = trading_days("2020-01-01", "2020-03-31")

        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 3, 31), 1000.0, 25, idx, timing="fixed_day"
        )
        got = [d.date() for d in df["invest_date"]]

        assert got[0] == date(2020, 1, 27)
        assert date(2020, 2, 25) in got

    def test_fixed_day_arrival_can_precede_investment(self):
        idx = trading_days("2020-01-01", "2020-02-29")

        df = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 2, 29), 1000.0, 25, idx, timing="fixed_day"
        )
        first = df.iloc[0]

        assert first["contribution_date"].date() == date(2020, 1, 25)
        assert first["invest_date"].date() == date(2020, 1, 27)


class TestValidation:
    def test_unknown_timing_raises(self):
        idx = trading_days("2020-01-01", "2020-03-31")

        with pytest.raises(ValueError, match="timing must be one of"):
            build_contribution_schedule(
                date(2020, 1, 1), date(2020, 3, 31), 100.0, 25, idx, timing="quarterly"
            )

    def test_empty_trading_index_yields_no_contributions(self):
        df = build_contribution_schedule(
            date(2020, 1, 1),
            date(2020, 3, 31),
            100.0,
            25,
            pd.DatetimeIndex([]),
            timing="month_end",
        )

        assert df.empty

    def test_boundary_helper_skips_months_outside_window(self):
        idx = trading_days("2020-02-01", "2020-02-29")

        picked = monthly_boundary_trading_days(
            date(2020, 2, 1), date(2020, 2, 29), idx, which="last"
        )

        assert len(picked) == 1
        assert picked[0].date() == date(2020, 2, 28)


class TestTimingChangesResults:
    def test_month_end_and_fixed_day_produce_different_schedules(self):
        """Guards against the timing argument being silently ignored."""
        idx = trading_days("2020-01-01", "2020-12-31")
        kw = dict(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            monthly_amount=1000.0,
            payday=25,
            trading_index=idx,
        )

        me = build_contribution_schedule(**kw, timing="month_end")
        fd = build_contribution_schedule(**kw, timing="fixed_day")

        assert list(me["invest_date"]) != list(fd["invest_date"])
        # Same number of contributions, same total capital deployed.
        assert len(me) == len(fd) == 12
        assert me["amount"].sum() == fd["amount"].sum()
