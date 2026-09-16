"""Monthly contribution schedule generation.

Three timing conventions are supported:

``month_end``
    Invest at the last valid trading close of each month. This is the default and
    the convention the DCA-versus-timing comparison is specified against — "invest
    the saved amount at the end of every month".
``fixed_day``
    Money arrives on a fixed day of month (a salary date) and is invested at the
    next valid trading close.
``month_start``
    Invest at the first valid trading close of each month.

``month_end`` and ``month_start`` are resolved by selecting directly from the
instrument's trading calendar, so they can never land on a non-trading day.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd

ContributionTiming = Literal["month_end", "fixed_day", "month_start"]

DEFAULT_CONTRIBUTION_TIMING: ContributionTiming = "month_end"


def generate_contribution_dates(
    start_date: date,
    end_date: date,
    payday: int = 25,
) -> pd.DatetimeIndex:
    """Generate all payday dates in [start_date, end_date].

    Payday is clamped to 28 to avoid month-end edge cases.

    Args:
        start_date: First possible contribution date.
        end_date: Last possible contribution date.
        payday: Day of month contributions arrive (1-28).

    Returns:
        DatetimeIndex of contribution dates.
    """
    payday = min(payday, 28)
    dates = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Move to first payday on or after start_date
    y, m = current.year, current.month
    candidate = pd.Timestamp(y, m, payday)
    if candidate < current:
        # Move to next month
        candidate = (
            pd.Timestamp(y + 1, 1, payday) if m == 12 else pd.Timestamp(y, m + 1, payday)
        )

    while candidate <= end:
        dates.append(candidate)
        # Advance by one month
        ny = candidate.year
        nm = candidate.month + 1
        if nm > 12:
            nm = 1
            ny += 1
        candidate = pd.Timestamp(ny, nm, payday)

    return pd.DatetimeIndex(dates)


def next_trading_day(dt: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return the first trading day on or after dt.

    Args:
        dt: Target date.
        trading_index: Sorted DatetimeIndex of trading days.

    Returns:
        Timestamp of next trading day, or None if none found.
    """
    pos = trading_index.searchsorted(dt)
    if pos >= len(trading_index):
        return None
    return trading_index[int(pos)]


def monthly_boundary_trading_days(
    start_date: date,
    end_date: date,
    trading_index: pd.DatetimeIndex,
    which: Literal["last", "first"],
) -> pd.DatetimeIndex:
    """Select the last (or first) trading day of each *complete* month in the window.

    For ``which="last"``: generates calendar month-ends (``freq="ME"``) within
    ``[start_date, end_date]``, then maps each to the nearest trading day on or
    before it.  A partial final month (e.g. ending March 16) is therefore
    excluded, because March 31 falls outside the window.

    For ``which="first"``: generates calendar month-starts (``freq="MS"``), then
    maps each to the nearest trading day on or after it.

    The result is always a real trading day from ``trading_index``.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if which == "last":
        # Calendar month-ends strictly within the window
        month_ends = pd.date_range(start=start_ts, end=end_ts, freq="ME")
        result = []
        for me in month_ends:
            eligible = trading_index[trading_index <= me]
            if len(eligible) > 0 and eligible[-1] >= start_ts:
                result.append(pd.Timestamp(eligible[-1]))
        return pd.DatetimeIndex(sorted(set(result)))

    # which == "first"
    month_starts = pd.date_range(start=start_ts, end=end_ts, freq="MS")
    result = []
    for ms in month_starts:
        eligible = trading_index[trading_index >= ms]
        if len(eligible) > 0 and eligible[0] <= end_ts:
            result.append(pd.Timestamp(eligible[0]))
    return pd.DatetimeIndex(sorted(set(result)))


def build_contribution_schedule(
    start_date: date,
    end_date: date,
    monthly_amount: float,
    payday: int,
    trading_index: pd.DatetimeIndex,
    timing: ContributionTiming = DEFAULT_CONTRIBUTION_TIMING,
) -> pd.DataFrame:
    """Build a DataFrame of contribution events with their investment dates.

    Args:
        start_date: First possible contribution date.
        end_date: Last possible contribution date.
        monthly_amount: Amount contributed each month.
        payday: Day of month money arrives. Used only when ``timing`` is
            ``"fixed_day"``.
        trading_index: Sorted DatetimeIndex of instrument trading days.
        timing: Contribution convention. Defaults to ``"month_end"``.

    Returns:
        DataFrame with columns ``contribution_date``, ``invest_date``, ``amount``.
        For the month-boundary conventions the contribution and investment dates
        are the same trading day; for ``fixed_day`` the investment date is the
        next trading day on or after the arrival date.
    """
    if timing in ("month_end", "month_start"):
        invest_dates = monthly_boundary_trading_days(
            start_date,
            end_date,
            trading_index,
            which="last" if timing == "month_end" else "first",
        )
        df = pd.DataFrame(
            {
                "contribution_date": invest_dates,
                "invest_date": invest_dates,
                "amount": monthly_amount,
            }
        )
        return df.dropna(subset=["invest_date"])

    if timing != "fixed_day":
        raise ValueError(
            f"timing must be one of 'month_end', 'fixed_day', 'month_start'; got {timing!r}"
        )

    paydates = generate_contribution_dates(start_date, end_date, payday)
    records = []
    for pd_date in paydates:
        invest_date = next_trading_day(pd_date, trading_index)
        records.append(
            {
                "contribution_date": pd_date,
                "invest_date": invest_date,
                "amount": monthly_amount,
            }
        )
    df = pd.DataFrame(records)
    df = df.dropna(subset=["invest_date"])
    return df
