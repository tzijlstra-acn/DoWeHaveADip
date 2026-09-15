"""Centralized live market data service with TTL caching at the UI boundary.

Pages should import and use `get_market_data_service()` instead of instantiating
providers directly. This module is the single place where caching is applied to
market data; provider modules themselves must not use @st.cache_data.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from functools import lru_cache

import streamlit as st

from dipdca import settings
from dipdca.data.errors import (
    InvalidProviderResponse,
    LiveDataUnavailable,
    StaleLiveData,
)
from dipdca.data.market_models import DataFreshness, PriceDataResult
from dipdca.data.providers.yahoo import YahooProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached fetch functions — cache key is (provider_name, symbol, start, end)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=settings.MARKET_HISTORY_TTL_SECONDS, persist=False, show_spinner=False)
def _cached_history(
    provider_name: str, symbol: str, start: date, end: date
) -> PriceDataResult:
    """Fetch and cache adjusted price history. TTL = MARKET_HISTORY_TTL_SECONDS (6h)."""
    logger.debug("Cache miss — fetching history %s %s %s–%s", provider_name, symbol, start, end)
    provider = _build_provider(provider_name)
    return provider.get_history(symbol, start, end)


@st.cache_data(ttl=settings.MARKET_QUOTE_TTL_SECONDS, persist=False, show_spinner=False)
def _cached_latest(provider_name: str, symbol: str) -> PriceDataResult:
    """Fetch and cache the latest quote. TTL = MARKET_QUOTE_TTL_SECONDS (5min)."""
    logger.debug("Cache miss — fetching latest %s %s", provider_name, symbol)
    provider = _build_provider(provider_name)
    return provider.get_latest(symbol)


def _build_provider(provider_name: str) -> _ProviderAdapter:
    if provider_name == "yahoo":
        return _ProviderAdapter(YahooProvider())
    raise LiveDataUnavailable(f"Unknown provider: {provider_name!r}")


# ---------------------------------------------------------------------------
# Internal adapter — wraps existing YahooProvider and produces PriceDataResult
# ---------------------------------------------------------------------------


class _ProviderAdapter:
    """Wraps the legacy YahooProvider to produce PriceDataResult with DataFreshness."""

    def __init__(self, provider: YahooProvider) -> None:
        self._p = provider

    def get_history(self, symbol: str, start: date, end: date) -> PriceDataResult:
        retrieved_at = datetime.now(tz=UTC)
        try:
            price_data = self._p.get_price_data(symbol, start, end)
        except (RuntimeError, ValueError) as exc:
            raise LiveDataUnavailable(
                f"Failed to fetch history for {symbol}: {exc}"
            ) from exc

        df = price_data.df
        if df is None or len(df) == 0:
            raise InvalidProviderResponse(f"Empty price series for {symbol}")

        last_obs = df.index[-1]
        observed_at = datetime.fromtimestamp(
            last_obs.timestamp(), tz=UTC
        ) if hasattr(last_obs, "timestamp") else datetime.combine(
            last_obs.date(), datetime.min.time(), tzinfo=UTC
        )

        age_minutes = (retrieved_at - observed_at).total_seconds() / 60
        if age_minutes > settings.MARKET_DATA_MAX_AGE_MINUTES:
            raise StaleLiveData(
                f"Last observation for {symbol} is {age_minutes:.0f} minutes old "
                f"(limit: {settings.MARKET_DATA_MAX_AGE_MINUTES})"
            )

        status = _freshness_status(age_minutes)
        freshness = DataFreshness(
            provider=settings.MARKET_DATA_PROVIDER,
            symbol=symbol,
            observed_at=observed_at,
            retrieved_at=retrieved_at,
            status=status,
            label=_freshness_label(status),
        )
        return PriceDataResult(frame=df, freshness=freshness)

    def get_latest(self, symbol: str) -> PriceDataResult:
        retrieved_at = datetime.now(tz=UTC)
        try:
            price, as_of = self._p.get_latest_price(symbol)
        except (RuntimeError, ValueError) as exc:
            raise LiveDataUnavailable(
                f"Failed to fetch latest price for {symbol}: {exc}"
            ) from exc

        observed_at = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
        age_minutes = (retrieved_at - observed_at).total_seconds() / 60

        if age_minutes > settings.MARKET_DATA_MAX_AGE_MINUTES:
            raise StaleLiveData(
                f"Latest quote for {symbol} is {age_minutes:.0f} minutes old"
            )

        import pandas as pd

        frame = pd.DataFrame({"adj_close": [price]}, index=pd.DatetimeIndex([observed_at]))
        status = _freshness_status(age_minutes)
        freshness = DataFreshness(
            provider=settings.MARKET_DATA_PROVIDER,
            symbol=symbol,
            observed_at=observed_at,
            retrieved_at=retrieved_at,
            status=status,
            label=_freshness_label(status),
        )
        return PriceDataResult(frame=frame, freshness=freshness)


def _freshness_status(age_minutes: float) -> str:
    if age_minutes < 30:
        return "fresh"
    if age_minutes < 1440:
        return "delayed"
    return "closed_market"


def _freshness_label(status: str) -> str:
    labels = {
        "fresh": "Live source — latest available close",
        "delayed": "15-minute delayed data",
        "closed_market": "Most recent close (market closed)",
    }
    return labels.get(status, "Live source")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class MarketDataService:
    """Public facade for fetching live market data with TTL caching."""

    def __init__(self, provider_name: str = settings.MARKET_DATA_PROVIDER) -> None:
        self._provider_name = provider_name

    def get_history(self, symbol: str, start: date, end: date) -> PriceDataResult:
        """Return cached adjusted price history for symbol in [start, end]."""
        return _cached_history(self._provider_name, symbol, start, end)

    def get_latest(self, symbol: str) -> PriceDataResult:
        """Return cached latest quote for symbol."""
        return _cached_latest(self._provider_name, symbol)

    @property
    def provider_name(self) -> str:
        return self._provider_name


@lru_cache(maxsize=1)
def get_market_data_service() -> MarketDataService:
    """Return the singleton MarketDataService configured from settings."""
    return MarketDataService(settings.MARKET_DATA_PROVIDER)
