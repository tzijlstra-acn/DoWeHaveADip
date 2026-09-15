"""Break-even analysis: when does waiting for a dip beat DCA?

The break-even condition
------------------------
DCA buys units at prices P_1, P_2, ..., P_n with contribution amounts C_1, ..., C_n.

    U_DCA = sum(C_i / P_i)   (total units bought)

If the wait strategy has deployable cash K at execution price P_T (net of fees):

    U_WAIT = K / P_T

The wait strategy acquires MORE units when:

    K / P_T > U_DCA
    P_T < K / U_DCA

The break-even execution price is:

    P_break_even = K / U_DCA

This equals the contribution-weighted harmonic mean of DCA purchase prices
(when K = total_contributions and fees = 0).

A market being 15% below its peak does NOT automatically mean waiting wins.
The market might have risen 50% first, so a 15% drawdown still leaves P_T
well above P_break_even.
"""

from __future__ import annotations


def dca_effective_price(
    contributions: list[tuple[float, float]],
) -> float:
    """Contribution-weighted harmonic mean DCA purchase price.

    Args:
        contributions: List of (amount_net_of_fees, price) pairs for each DCA trade.

    Returns:
        Effective average price per unit = total_cash / total_units.
        Returns 0.0 if no valid contributions.
    """
    total_cash = sum(c for c, _ in contributions)
    total_units = sum(c / p for c, p in contributions if p > 0)
    if total_units <= 0:
        return 0.0
    return total_cash / total_units


def dca_total_units(contributions: list[tuple[float, float]]) -> float:
    """Total units acquired by DCA across all contribution trades.

    Args:
        contributions: List of (amount_net_of_fees, price) pairs.

    Returns:
        Total units held.
    """
    return sum(c / p for c, p in contributions if p > 0)


def break_even_execution_price(
    deployable_cash: float,
    dca_units: float,
) -> float:
    """Execution price at which wait strategy acquires exactly as many units as DCA.

    Wait beats DCA when:   execution_price < break_even_execution_price
    Wait loses to DCA when: execution_price > break_even_execution_price

    Without fees and with equal total contributions, this equals the
    contribution-weighted harmonic mean of DCA purchase prices.

    Args:
        deployable_cash: Cash available to the wait strategy at execution (net of fees).
        dca_units: Total units acquired by the DCA strategy over the same period.

    Returns:
        Break-even execution price, or 0.0 if dca_units is zero.
    """
    if dca_units <= 0:
        return 0.0
    return deployable_cash / dca_units


def unit_advantage_pct(wait_units: float, dca_units: float) -> float | None:
    """Percentage unit advantage of wait over DCA.

    Positive means wait acquired more units (wait wins on total units).
    Negative means DCA acquired more units.

    Args:
        wait_units: Units acquired by wait strategy.
        dca_units: Units acquired by DCA strategy.

    Returns:
        (wait_units / dca_units - 1) as a fraction, or None if dca_units is zero.
    """
    if dca_units <= 0:
        return None
    return wait_units / dca_units - 1.0


def contributions_from_ledger(ledger, price_col: str = "price", deployed_col: str = "deployed") -> list[tuple[float, float]]:
    """Extract (deployed_amount, price) pairs from a backtest ledger.

    Args:
        ledger: Ledger DataFrame from run_dca() or run_wait_for_dip().
        price_col: Column with asset price.
        deployed_col: Column with notional deployed (0 = no trade).

    Returns:
        List of (amount, price) pairs for days where deployed > 0.
    """
    deployed = ledger[ledger[deployed_col] > 0]
    return [(float(row[deployed_col]), float(row[price_col])) for _, row in deployed.iterrows()]
