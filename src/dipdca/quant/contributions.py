"""Monthly contribution schedule generation."""

from __future__ import annotations

from datetime import date

import pandas as pd


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


def build_contribution_schedule(
    start_date: date,
    end_date: date,
    monthly_amount: float,
    payday: int,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a DataFrame of contribution events with their investment dates.

    Returns DataFrame with columns:
        contribution_date: When money arrives.
        invest_date: Next trading day on/after contribution_date.
        amount: Amount contributed.
    """
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
