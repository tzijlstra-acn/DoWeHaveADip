"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float | None:
    """Compute Sharpe ratio from daily returns.

    Args:
        returns: Daily returns as fractions (e.g. 0.01 = 1%).
        risk_free_rate: Annual risk-free rate.
        annualize: If True, annualize using sqrt(252).

    Returns:
        Sharpe ratio, or None if std is zero.
    """
    if len(returns) < 2:
        return None
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if std == 0:
        return None
    ratio = excess.mean() / std
    if annualize:
        ratio *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(ratio)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float | None:
    """Compute Sortino ratio (downside deviation only).

    Args:
        returns: Daily returns.
        risk_free_rate: Annual risk-free rate.
        annualize: If True, annualize using sqrt(252).

    Returns:
        Sortino ratio, or None if downside std is zero.
    """
    if len(returns) < 2:
        return None
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return None
    ratio = excess.mean() / downside_std
    if annualize:
        ratio *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(ratio)


def portfolio_returns(wealth_series: pd.Series) -> pd.Series:
    """Compute daily percentage returns from a wealth series."""
    return wealth_series.pct_change().dropna()


def time_in_market_pct(units_held: pd.Series) -> float:
    """Fraction of days where portfolio held > 0 units of the ETF."""
    if len(units_held) == 0:
        return 0.0
    return float((units_held > 0).mean())


def cagr_from_wealth(
    start_wealth: float,
    end_wealth: float,
    years: float,
) -> float | None:
    """Compound Annual Growth Rate from start to end wealth."""
    if years <= 0 or start_wealth <= 0:
        return None
    return float((end_wealth / start_wealth) ** (1 / years) - 1)
