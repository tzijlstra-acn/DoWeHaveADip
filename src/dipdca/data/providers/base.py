"""Abstract base class for data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from dipdca.models import PriceData


class BaseProvider(ABC):
    """Base interface for market data providers."""

    @abstractmethod
    def get_price_data(self, symbol: str, start: date, end: date) -> PriceData:
        """Fetch OHLCV price data for a symbol.

        Args:
            symbol: Ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            PriceData with adj_close column and DatetimeIndex.
        """
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> tuple[float, date]:
        """Return (latest_close, as_of_date)."""
        ...

    def is_available(self) -> bool:
        """Check if provider can reach its data source."""
        return True
