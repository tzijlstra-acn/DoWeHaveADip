"""DuckDB-backed cache for price and FX data."""

from __future__ import annotations

import logging
from datetime import date, datetime

import duckdb
import pandas as pd

from dipdca.config import CACHE_DIR

logger = logging.getLogger(__name__)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "prices.duckdb"


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the cache database."""
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_cache (
            symbol VARCHAR,
            dt DATE,
            adj_close DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            source VARCHAR,
            fetched_at TIMESTAMP,
            PRIMARY KEY (symbol, dt)
        )
        """
    )
    return conn


def read_cached(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    """Read cached price data for a symbol, or return None if not cached."""
    try:
        conn = _get_connection()
        df = conn.execute(
            """
            SELECT dt, adj_close, close, volume
            FROM price_cache
            WHERE symbol = ? AND dt >= ? AND dt <= ?
            ORDER BY dt
            """,
            [symbol, start, end],
        ).df()
        conn.close()
        if len(df) == 0:
            return None
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.set_index("dt")
        df.index.name = None
        return df
    except Exception as exc:
        logger.warning("Cache read failed for %s: %s", symbol, exc)
        return None


def write_cache(symbol: str, df: pd.DataFrame, source: str) -> None:
    """Write price data to the cache."""
    try:
        conn = _get_connection()
        now = datetime.utcnow()
        for dt, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO price_cache
                    (symbol, dt, adj_close, close, volume, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    symbol,
                    dt.date() if hasattr(dt, "date") else dt,
                    float(row.get("adj_close", float("nan"))),
                    float(row.get("close", float("nan"))),
                    float(row.get("volume", float("nan"))),
                    source,
                    now,
                ],
            )
        conn.close()
    except Exception as exc:
        logger.warning("Cache write failed for %s: %s", symbol, exc)


def cache_is_fresh(symbol: str, end: date, max_age_days: int = 1) -> bool:
    """Check if cache has data for symbol up to end date within max_age_days."""
    try:
        conn = _get_connection()
        result = conn.execute(
            """
            SELECT MAX(dt) as last_dt
            FROM price_cache
            WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()
        conn.close()
        if result is None or result[0] is None:
            return False
        last_dt = result[0]
        if hasattr(last_dt, "date"):
            last_dt = last_dt.date()
        age = (date.today() - last_dt).days
        return age <= max_age_days and last_dt >= end
    except Exception:
        return False
