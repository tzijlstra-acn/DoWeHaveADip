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


# Relative tolerance below which two outcomes are called equal. Chosen as a
# relative figure rather than a flat euro amount so it stays meaningful whether
# the portfolio is EUR 1,000 or EUR 1,000,000.
TIE_TOLERANCE = 1e-6


def fmt_delta(value: float, baseline: float, currency: str = "EUR") -> str:
    """Format a difference with both absolute and relative precision.

    A whole-euro format hides any difference below EUR 1 as "0", so cents and the
    relative figure are both shown.
    """
    rel = value / baseline if baseline else 0.0
    return f"{currency} {value:+,.2f} ({rel:+.3%})"


def compare_outcomes(
    candidate: float,
    baseline: float,
    candidate_label: str,
    baseline_label: str,
    tolerance: float = TIE_TOLERANCE,
) -> tuple[str, float]:
    """Describe a comparison as one of three states, never as a signed zero.

    ``candidate >= baseline`` would report an exact tie as the baseline winning
    "by EUR 0". Returns ``(verdict, difference)``.
    """
    diff = candidate - baseline
    scale = abs(baseline) if baseline else 1.0
    if abs(diff) / scale < tolerance:
        return "Effectively equal", diff
    if diff > 0:
        return f"{candidate_label} ahead", diff
    return f"{baseline_label} ahead", diff
