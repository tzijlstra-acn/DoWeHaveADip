"""Exit strategy simulation (target return, time-based hold, trailing stop).

Exit rules must be evaluated on investment performance, never on raw account
wealth. Raw ``total_wealth`` contains external contributions, which corrupts every
rule built on it:

- a deposit alone can satisfy a target-return threshold;
- a deposit raises the running peak, so a trailing stop measures the wrong
  drawdown;
- with the default zero initial investment, opening wealth is zero, so
  ``target = 0 * (1 + r)`` is zero and the rule fires on day one.

So decisions here are taken on a **flow-adjusted unitized NAV**: contributions
change the unit count, not the unit price. A pure deposit on a flat-price day
leaves NAV unchanged.

Two further requirements for a fair comparison:

- the holding clock starts at the **first deployment**, not the first ledger row,
  since there is nothing to hold before any money is invested;
- after an exit the proceeds still exist. They are held at the savings rate to a
  **common terminal date** shared by every policy, so exit and never-sell are
  valued on the same day.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

from dipdca.quant.metrics import build_nav, flow_adjusted_returns


class ExitType(StrEnum):
    TARGET_RETURN = "target_return"
    TIME_BASED = "time_based"
    TRAILING_STOP = "trailing_stop"
    NONE = "none"


def ledger_nav(ledger: pd.DataFrame) -> pd.Series:
    """Flow-adjusted unitized NAV for a backtest ledger, based at 1.0.

    Requires ``total_wealth`` and ``external_flow``. Deposits are removed from the
    return series, so the NAV reflects investment performance alone.
    """
    if "total_wealth" not in ledger.columns:
        raise ValueError("ledger must have a 'total_wealth' column")
    flows = (
        ledger["external_flow"]
        if "external_flow" in ledger.columns
        else pd.Series(0.0, index=ledger.index)
    )
    returns = flow_adjusted_returns(ledger["total_wealth"], flows)
    if len(returns) == 0:
        return pd.Series(1.0, index=ledger.index[:1])
    nav = build_nav(returns, nav_start=1.0)
    # Re-attach a 1.0 base on the first ledger date for a complete series.
    first = ledger.index[0]
    if first not in nav.index:
        nav = pd.concat([pd.Series([1.0], index=[first]), nav])
    return nav.sort_index()


def first_deployment_date(ledger: pd.DataFrame) -> pd.Timestamp | None:
    """Date of the first actual purchase, or None if nothing was ever bought."""
    if "deployed" in ledger.columns:
        deployed = ledger[ledger["deployed"] > 0]
        if not deployed.empty:
            return pd.Timestamp(deployed.index[0])
    if "units" in ledger.columns:
        held = ledger[ledger["units"] > 0]
        if not held.empty:
            return pd.Timestamp(held.index[0])
    return None


def _accrue_cash(amount: float, days: int, annual_rate: float) -> float:
    """Grow cash at an annual effective rate over a number of days."""
    if days <= 0 or annual_rate <= 0:
        return amount
    return amount * (1.0 + annual_rate) ** (days / 365.25)


def simulate_exit(
    ledger: pd.DataFrame,
    exit_type: ExitType,
    target_return: float = 0.5,
    hold_years: int = 5,
    trailing_stop: float = -0.15,
    cash_rate: float = 0.0,
    terminal_date: pd.Timestamp | None = None,
) -> dict:
    """Apply an exit rule to a backtest ledger.

    Args:
        ledger: Day-by-day ledger with ``total_wealth`` and ``external_flow``.
        exit_type: Which exit trigger to apply.
        target_return: NAV gain that triggers a target-return exit (0.5 = +50%).
        hold_years: Years to hold from the FIRST DEPLOYMENT for a time-based exit.
        trailing_stop: NAV drawdown that triggers a stop (-0.15 = -15%).
        cash_rate: Annual rate earned on proceeds between exit and terminal date.
        terminal_date: Common valuation date for every policy. Defaults to the
            last ledger date.

    Returns:
        Dict with keys:
            ``exit_date``          date the rule fired, or None if it never did
            ``exit_wealth``        wealth at the moment of exit
            ``terminal_wealth``    wealth at ``terminal_date`` (proceeds accrued
                                   at ``cash_rate`` after an exit)
            ``terminal_date``      the common valuation date
            ``triggered``          whether the rule actually fired
            ``investment_return``  NAV-based return over the holding period
            ``hold_days``          days from first deployment to exit/terminal
            ``first_deployment``   date of the first purchase, or None
    """
    if len(ledger) == 0:
        return {
            "exit_date": None,
            "exit_wealth": 0.0,
            "terminal_wealth": 0.0,
            "terminal_date": None,
            "triggered": False,
            "investment_return": 0.0,
            "hold_days": 0,
            "first_deployment": None,
        }

    wealth = ledger["total_wealth"]
    last_date = pd.Timestamp(ledger.index[-1])
    terminal = pd.Timestamp(terminal_date) if terminal_date is not None else last_date

    nav = ledger_nav(ledger)
    entry = first_deployment_date(ledger)

    def result(
        exit_date: pd.Timestamp | None,
        triggered: bool,
    ) -> dict:
        if exit_date is None or not triggered:
            # Held to the end: value on the terminal date itself.
            valued_on = terminal if terminal in wealth.index else last_date
            terminal_wealth = float(wealth.loc[valued_on])
            exit_wealth = terminal_wealth
            effective_exit = None
        else:
            exit_wealth = float(wealth.loc[exit_date])
            # Proceeds persist: hold at the savings rate to the common date.
            terminal_wealth = _accrue_cash(
                exit_wealth, (terminal - exit_date).days, cash_rate
            )
            effective_exit = exit_date

        if entry is None:
            hold_days = 0
            inv_return = 0.0
        else:
            end_for_hold = effective_exit or (
                terminal if terminal in wealth.index else last_date
            )
            hold_days = max(0, (end_for_hold - entry).days)
            inv_return = _nav_return(nav, entry, end_for_hold)

        return {
            "exit_date": effective_exit,
            "exit_wealth": exit_wealth,
            "terminal_wealth": terminal_wealth,
            "terminal_date": terminal,
            "triggered": triggered,
            "investment_return": inv_return,
            "hold_days": hold_days,
            "first_deployment": entry,
        }

    if exit_type == ExitType.NONE or entry is None:
        # Nothing was ever bought, so no exit rule can meaningfully fire.
        return result(None, triggered=False)

    # Only the invested period is eligible for an exit.
    nav_held = nav.loc[nav.index >= entry]
    if len(nav_held) == 0:
        return result(None, triggered=False)

    if exit_type == ExitType.TARGET_RETURN:
        entry_nav = float(nav_held.iloc[0])
        target_nav = entry_nav * (1.0 + target_return)
        hit = nav_held[nav_held >= target_nav]
        return result(
            pd.Timestamp(hit.index[0]) if len(hit) else None, triggered=len(hit) > 0
        )

    if exit_type == ExitType.TIME_BASED:
        # Clock runs from the first deployment, not the first ledger row.
        due = entry + pd.DateOffset(years=hold_years)
        eligible = nav_held.index[nav_held.index >= due]
        if len(eligible) == 0:
            # The hold period extends past the data: the rule never fires.
            return result(None, triggered=False)
        return result(pd.Timestamp(eligible[0]), triggered=True)

    if exit_type == ExitType.TRAILING_STOP:
        # Drawdown of NAV, so contributions cannot move the peak.
        dd = nav_held / nav_held.cummax() - 1.0
        hit = dd[dd <= trailing_stop]
        return result(
            pd.Timestamp(hit.index[0]) if len(hit) else None, triggered=len(hit) > 0
        )

    return result(None, triggered=False)


def _nav_return(nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """NAV-based return between two dates, using the nearest available points."""
    window = nav.loc[(nav.index >= start) & (nav.index <= end)]
    if len(window) < 2:
        return 0.0
    first = float(window.iloc[0])
    last = float(window.iloc[-1])
    if first <= 0:
        return 0.0
    return last / first - 1.0
