"""Cash balance and interest accrual utilities."""

from __future__ import annotations

import pandas as pd


def accrue_interest(balance: float, annual_rate: float, days: int) -> float:
    """Compound interest: balance * (1 + r) ** (d / 365).

    Args:
        balance: Starting cash balance.
        annual_rate: Annual interest rate as decimal (e.g. 0.03 = 3%).
        days: Number of calendar days to accrue over.

    Returns:
        New balance after accrual.

    Raises:
        ValueError: If annual_rate <= -1 (undefined compounding).
    """
    if annual_rate <= -1:
        raise ValueError(f"annual_rate must be > -1, got {annual_rate}")
    if days < 0:
        raise ValueError(f"days must be >= 0, got {days}")
    return balance * (1 + annual_rate) ** (days / 365)


def accrue_over_period(
    cash_series: pd.Series,
    rate_series: pd.Series,
) -> pd.Series:
    """Day-by-day compounding using elapsed calendar days.

    Args:
        cash_series: Daily cash balances (DatetimeIndex).
        rate_series: Monthly annual rates, forward-filled to daily (DatetimeIndex).

    Returns:
        Series of interest earned each day.
    """
    if len(cash_series) == 0:
        return pd.Series(dtype=float)

    # Align rate_series to cash_series index
    rate_aligned = rate_series.reindex(cash_series.index, method="ffill").fillna(0.0)

    interest = pd.Series(0.0, index=cash_series.index)
    for i in range(1, len(cash_series)):
        prev_dt = cash_series.index[i - 1]
        curr_dt = cash_series.index[i]
        days = (curr_dt - prev_dt).days
        r = rate_aligned.iloc[i - 1]
        prev_balance = cash_series.iloc[i - 1]
        if prev_balance > 0 and days > 0:
            new_balance = accrue_interest(prev_balance, r, days)
            interest.iloc[i] = new_balance - prev_balance

    return interest


def total_interest_earned(
    cash_series: pd.Series,
    rate_series: pd.Series,
) -> float:
    """Sum of all interest earned over the period."""
    return float(accrue_over_period(cash_series, rate_series).sum())
