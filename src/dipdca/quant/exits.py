"""Exit strategy simulation (target-price, time-based, trailing stop)."""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class ExitType(StrEnum):
    TARGET_RETURN = "target_return"
    TIME_BASED = "time_based"
    TRAILING_STOP = "trailing_stop"
    NONE = "none"


def simulate_exit(
    ledger: pd.DataFrame,
    exit_type: ExitType,
    target_return: float = 0.5,
    hold_years: int = 5,
    trailing_stop: float = -0.15,
) -> dict:
    """Simulate an exit strategy on a backtest ledger.

    Args:
        ledger: Day-by-day backtest ledger with total_wealth column.
        exit_type: Which exit trigger to use.
        target_return: Target total return to exit at (e.g. 0.5 = 50%).
        hold_years: Hold period for time-based exit.
        trailing_stop: Trailing drawdown stop (e.g. -0.15 = -15%).

    Returns:
        Dict with exit_date, exit_wealth, total_return, hold_days.
    """
    wealth = ledger["total_wealth"]
    if len(wealth) == 0:
        return {"exit_date": None, "exit_wealth": 0.0, "total_return": 0.0, "hold_days": 0}

    start_wealth = wealth.iloc[0]

    if exit_type == ExitType.NONE:
        return {
            "exit_date": wealth.index[-1],
            "exit_wealth": float(wealth.iloc[-1]),
            "total_return": float(wealth.iloc[-1] / start_wealth - 1) if start_wealth > 0 else 0.0,
            "hold_days": (wealth.index[-1] - wealth.index[0]).days,
        }

    if exit_type == ExitType.TARGET_RETURN:
        target_wealth = start_wealth * (1 + target_return)
        triggered = wealth[wealth >= target_wealth]
        if len(triggered) > 0:
            exit_date = triggered.index[0]
            exit_wealth = float(triggered.iloc[0])
        else:
            exit_date = wealth.index[-1]
            exit_wealth = float(wealth.iloc[-1])

    elif exit_type == ExitType.TIME_BASED:
        exit_date_target = wealth.index[0] + pd.DateOffset(years=hold_years)
        pos = wealth.index.searchsorted(exit_date_target)
        pos = min(pos, len(wealth) - 1)
        exit_date = wealth.index[pos]
        exit_wealth = float(wealth.iloc[pos])

    elif exit_type == ExitType.TRAILING_STOP:
        running_peak = wealth.expanding().max()
        dd = wealth / running_peak - 1
        triggered = dd[dd <= trailing_stop]
        if len(triggered) > 0:
            exit_date = triggered.index[0]
            exit_wealth = float(wealth.loc[exit_date])
        else:
            exit_date = wealth.index[-1]
            exit_wealth = float(wealth.iloc[-1])

    else:
        exit_date = wealth.index[-1]
        exit_wealth = float(wealth.iloc[-1])

    return {
        "exit_date": exit_date,
        "exit_wealth": exit_wealth,
        "total_return": float(exit_wealth / start_wealth - 1) if start_wealth > 0 else 0.0,
        "hold_days": (exit_date - wealth.index[0]).days,
    }
