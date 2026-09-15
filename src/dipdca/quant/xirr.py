"""Extended Internal Rate of Return (XIRR) calculation."""

from __future__ import annotations

import contextlib
from datetime import date

from scipy.optimize import brentq


def xirr(
    cash_flows: list[tuple[date, float]],
    guess: float = 0.1,
) -> float | None:
    """Newton-Raphson / brentq XIRR.

    Args:
        cash_flows: List of (date, amount) tuples.
                    Investments are negative, receipts are positive.
        guess: Initial guess for the rate (default 10%).

    Returns:
        Annualized IRR as a decimal (e.g. 0.10 = 10%), or None if no solution.

    Raises:
        ValueError: If cash flows are all positive or all negative (no solution possible).
    """
    if not cash_flows:
        return None

    # Sort by date
    flows = sorted(cash_flows, key=lambda x: x[0])
    amounts = [f[1] for f in flows]

    has_negative = any(a < 0 for a in amounts)
    has_positive = any(a > 0 for a in amounts)
    if not has_negative or not has_positive:
        return None

    t0 = flows[0][0]
    days = [(f[0] - t0).days for f in flows]

    def npv(rate: float) -> float:
        return sum(a / (1 + rate) ** (d / 365) for a, d in zip(amounts, days, strict=False))

    # Search in a bounded range
    try:
        result = brentq(npv, -0.9999, 100.0, maxiter=1000)
    except ValueError:
        return None
    except Exception:
        return None

    return float(result)


def simple_twr(wealth_series: pd.Series, contributions: pd.Series) -> float:  # noqa: F821
    """Time-weighted return — placeholder for future implementation."""
    raise NotImplementedError("TWR not yet implemented")


# Avoid importing pandas at module level for xirr purity
with contextlib.suppress(ImportError):
    import pandas as pd  # noqa: F401
