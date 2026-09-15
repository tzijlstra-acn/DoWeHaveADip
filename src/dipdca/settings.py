"""Runtime configuration read from environment variables or st.secrets."""

from __future__ import annotations

import os


def _int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _str(key: str, default: str) -> str:
    return os.environ.get(key, default)


MARKET_DATA_PROVIDER: str = _str("MARKET_DATA_PROVIDER", "yahoo")
MARKET_DATA_TIMEOUT_SECONDS: int = _int("MARKET_DATA_TIMEOUT_SECONDS", 15)
MARKET_QUOTE_TTL_SECONDS: int = _int("MARKET_QUOTE_TTL_SECONDS", 300)
MARKET_HISTORY_TTL_SECONDS: int = _int("MARKET_HISTORY_TTL_SECONDS", 21600)
MARKET_DATA_MAX_AGE_MINUTES: int = _int("MARKET_DATA_MAX_AGE_MINUTES", 1440)
