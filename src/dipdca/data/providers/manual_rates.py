"""Manual / override cash rate provider."""

from __future__ import annotations

from datetime import date

import pandas as pd


class ManualRatesProvider:
    """Return a user-specified flat annual cash rate."""

    def __init__(self, annual_rate: float = 0.0) -> None:
        self.annual_rate = annual_rate

    def get_rate_series(self, start: date, end: date) -> pd.Series:
        """Return a constant daily rate series.

        Args:
            start: Start date.
            end: End date.

        Returns:
            Daily Series with constant value = annual_rate.
        """
        idx = pd.date_range(start, end, freq="D")
        return pd.Series(self.annual_rate, index=idx, name="manual_rate")
