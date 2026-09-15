"""Rolling window statistics for walk-forward analysis."""

from __future__ import annotations

import pandas as pd


def rolling_cagr(
    wealth_series: pd.Series,
    window_years: int = 3,
) -> pd.Series:
    """Compute rolling CAGR over a window of N years.

    Args:
        wealth_series: Portfolio wealth (DatetimeIndex).
        window_years: Lookback window in years.

    Returns:
        Series of rolling CAGR values.
    """
    window_days = window_years * 252  # approximate trading days
    result = pd.Series(index=wealth_series.index, dtype=float)
    for i in range(window_days, len(wealth_series)):
        start_val = wealth_series.iloc[i - window_days]
        end_val = wealth_series.iloc[i]
        if start_val > 0:
            result.iloc[i] = (end_val / start_val) ** (1 / window_years) - 1
    return result


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_daily: float = 0.0,
) -> pd.Series:
    """Rolling Sharpe ratio.

    Args:
        returns: Daily returns series.
        window: Rolling window in trading days.
        risk_free_daily: Daily risk-free rate.

    Returns:
        Rolling Sharpe series, annualized.
    """

    excess = returns - risk_free_daily
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std(ddof=1)
    return (roll_mean / roll_std) * (252**0.5)


def rolling_max_drawdown(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling maximum drawdown over a window."""
    result = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)):
        window_slice = series.iloc[i - window : i + 1]
        peak = window_slice.expanding().max()
        dd = window_slice / peak - 1
        result.iloc[i] = float(dd.min())
    return result
