"""Portfolio performance metrics — flow-adjusted and raw."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Flow-adjusted returns (deposit-corrected)
# ---------------------------------------------------------------------------


def flow_adjusted_returns(
    wealth: pd.Series,
    external_flows: pd.Series,
) -> pd.Series:
    """Beginning-of-period flow-adjusted daily returns.

    Convention: external flows book at the BEGINNING of the period (i.e. a
    contribution arrives and is invested before the close is recorded).

        r_t = V_t / (V_{t-1} + F_t) - 1

    A pure deposit on a flat-price day produces r_t = 0.
    A pure market gain with no deposit produces the correct price return.

    Args:
        wealth: Daily total portfolio wealth series (cash + market value).
        external_flows: Signed external flows aligned to the same index.
                        Positive = contribution into the portfolio.
                        Negative = withdrawal from the portfolio.
                        Zero = no external flow that day.

    Returns:
        Series of flow-adjusted daily returns (first day dropped; NaN rows dropped).
    """
    flows = external_flows.reindex(wealth.index).fillna(0.0)
    prev_wealth = wealth.shift(1)
    denominator = prev_wealth + flows
    returns = (wealth / denominator - 1).where(denominator > 0)
    return returns.dropna()


def build_nav(flow_adj_returns: pd.Series, nav_start: float = 1.0) -> pd.Series:
    """Unitized NAV from a series of flow-adjusted returns.

    nav_0 = nav_start
    nav_t = nav_{t-1} * (1 + r_t)
    """
    return nav_start * (1.0 + flow_adj_returns).cumprod()


def nav_max_drawdown(nav: pd.Series) -> float:
    """Maximum drawdown of a unitized NAV series (always ≤ 0)."""
    if len(nav) == 0:
        return 0.0
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def twr_cagr(flow_adj_returns: pd.Series, years: float) -> float | None:
    """Time-weighted CAGR from flow-adjusted daily returns.

    TWR = product of (1 + r_t) over all days.
    CAGR = TWR^(1/years) - 1.
    """
    if years <= 0 or len(flow_adj_returns) == 0:
        return None
    total_return = float((1.0 + flow_adj_returns).prod())  # type: ignore[arg-type]
    if total_return <= 0:
        return None
    return float(total_return ** (1.0 / years) - 1.0)


# ---------------------------------------------------------------------------
# Risk metrics (require flow-adjusted returns)
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float | None:
    """Annualized Sharpe ratio from daily returns.

    Risk-free rate is converted to a daily effective rate before subtraction:
        daily_rf = (1 + annual_rf)^(1/252) - 1

    Args:
        returns: Daily returns (flow-adjusted or raw).
        risk_free_rate: Annual effective risk-free rate.
        annualize: If True, multiply by sqrt(252).

    Returns:
        Sharpe ratio, or None if std is zero or series too short.
    """
    if len(returns) < 2:
        return None
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
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
    """Annualized Sortino ratio (downside deviation only).

    Args:
        returns: Daily returns (flow-adjusted or raw).
        risk_free_rate: Annual effective risk-free rate.
        annualize: If True, multiply by sqrt(252).

    Returns:
        Sortino ratio, or None if no negative excess returns or series too short.
    """
    if len(returns) < 2:
        return None
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
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


# ---------------------------------------------------------------------------
# Legacy / convenience helpers
# ---------------------------------------------------------------------------


def portfolio_returns(wealth_series: pd.Series) -> pd.Series:
    """Compute simple daily percentage returns from a wealth series.

    WARNING: When the wealth series includes external capital flows (contributions),
    those flows inflate the returns. Use flow_adjusted_returns() instead for
    Sharpe, Sortino, volatility, and drawdown calculations.

    Retained for backward compatibility and for pure price-series analysis
    (e.g. drawdown charts on the raw ETF price series with no contributions).
    """
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
    """Simple CAGR from a start wealth to end wealth over a number of years.

    NOTE: When used with total_contributions as start_wealth, this is NOT a true
    time-weighted CAGR — it ignores the timing of contributions. Use twr_cagr()
    for a proper time-weighted measure. This function is retained as a simple
    money-weighted approximation for display purposes.
    """
    if years <= 0 or start_wealth <= 0:
        return None
    return float((end_wealth / start_wealth) ** (1.0 / years) - 1.0)
