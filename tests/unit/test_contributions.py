"""Tests for contribution schedule generation."""

from __future__ import annotations

from datetime import date

import pandas as pd

from dipdca.quant.contributions import (
    build_contribution_schedule,
    generate_contribution_dates,
    next_trading_day,
)


class TestGenerateContributionDates:
    def test_basic_monthly(self):
        dates = generate_contribution_dates(date(2020, 1, 1), date(2020, 12, 31), payday=25)
        assert len(dates) == 12
        assert all(d.day == 25 for d in dates)

    def test_partial_year(self):
        dates = generate_contribution_dates(date(2020, 6, 1), date(2020, 9, 30), payday=15)
        # June 15, July 15, Aug 15, Sep 15
        assert len(dates) == 4

    def test_payday_clamped_to_28(self):
        dates = generate_contribution_dates(date(2020, 1, 1), date(2020, 3, 31), payday=31)
        assert all(d.day == 28 for d in dates)

    def test_start_after_payday(self):
        """If start is after payday in that month, first date is next month."""
        dates = generate_contribution_dates(date(2020, 1, 26), date(2020, 3, 31), payday=25)
        assert dates[0] >= pd.Timestamp("2020-02-25")

    def test_empty_range(self):
        dates = generate_contribution_dates(date(2020, 1, 1), date(2020, 1, 5), payday=25)
        assert len(dates) == 0


class TestNextTradingDay:
    def setup_method(self):
        self.trading_idx = pd.bdate_range("2020-01-01", "2020-01-31")

    def test_on_trading_day(self):
        """If dt is a trading day, return it."""
        result = next_trading_day(pd.Timestamp("2020-01-02"), self.trading_idx)
        assert result == pd.Timestamp("2020-01-02")

    def test_on_weekend_returns_monday(self):
        """Saturday → next Monday."""
        result = next_trading_day(pd.Timestamp("2020-01-04"), self.trading_idx)
        assert result == pd.Timestamp("2020-01-06")

    def test_beyond_range_returns_none(self):
        result = next_trading_day(pd.Timestamp("2020-02-01"), self.trading_idx)
        assert result is None


class TestBuildContributionSchedule:
    def test_basic(self):
        trading_idx = pd.bdate_range("2020-01-01", "2020-12-31")
        schedule = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 12, 31), 500.0, 25, trading_idx
        )
        assert "contribution_date" in schedule.columns
        assert "invest_date" in schedule.columns
        assert "amount" in schedule.columns
        assert (schedule["amount"] == 500.0).all()
        assert len(schedule) == 12

    def test_invest_date_on_or_after_contribution(self):
        trading_idx = pd.bdate_range("2020-01-01", "2020-12-31")
        schedule = build_contribution_schedule(
            date(2020, 1, 1), date(2020, 12, 31), 1000.0, 25, trading_idx
        )
        for _, row in schedule.iterrows():
            assert row["invest_date"] >= row["contribution_date"]
