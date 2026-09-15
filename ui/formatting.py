"""Number and date formatting helpers."""

from __future__ import annotations

from datetime import date


def fmt_currency(value: float, currency: str = "EUR", decimals: int = 0) -> str:
    """Format a number as currency."""
    return f"{currency} {value:,.{decimals}f}"


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def fmt_ratio(value: float | None, decimals: int = 2) -> str:
    """Format a ratio (Sharpe, Sortino) to given decimal places."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_date(dt: date | None) -> str:
    """Format a date as ISO string."""
    if dt is None:
        return "N/A"
    return str(dt)


def fmt_large_number(value: float) -> str:
    """Format large numbers with K/M suffix."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"
