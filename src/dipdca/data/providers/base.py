"""Provider protocol for live market data sources."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from dipdca.data.market_models import PriceDataResult


class MarketDataProvider(Protocol):
    """Protocol that all market data providers must satisfy."""

    def get_history(self, symbol: str, start: date, end: date) -> PriceDataResult:
        """Fetch adjusted price history for symbol in [start, end].

        Returns:
            PriceDataResult with a sorted DatetimeIndex and adj_close column.

        Raises:
            LiveDataUnavailable: If the source cannot be reached or returns bad data.
            StaleLiveData: If the most recent observation is too old.
        """
        ...

    def get_latest(self, symbol: str) -> PriceDataResult:
        """Fetch the latest available close price.

        Raises:
            LiveDataUnavailable: If the source cannot be reached.
        """
        ...
