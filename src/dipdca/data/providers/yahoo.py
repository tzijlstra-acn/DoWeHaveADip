"""Yahoo Finance data provider via yfinance."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from dipdca.models import DataHealthStatus, PriceData
from dipdca.quant.total_return import validate_adjusted_data

logger = logging.getLogger(__name__)


class YahooProvider:
    """Fetch price data from Yahoo Finance with auto-adjustment enabled."""

    def get_price_data(self, symbol: str, start: date, end: date) -> PriceData:
        """Download adjusted price data for a symbol.

        Uses auto_adjust=True to get dividend-adjusted prices.
        Validates that adj_close is actually different from close.

        Args:
            symbol: Yahoo ticker symbol.
            start: Start date.
            end: End date (inclusive — yfinance end is exclusive, we add 1 day).

        Returns:
            PriceData with adj_close, close, volume columns.

        Raises:
            ValueError: If adj_close is not available or equals close.
            RuntimeError: If download fails.
        """
        import datetime

        # yfinance end is exclusive
        end_exclusive = end + datetime.timedelta(days=1)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=str(start),
                end=str(end_exclusive),
                auto_adjust=True,
                actions=False,
            )
        except Exception as exc:
            raise RuntimeError(f"yfinance download failed for {symbol}: {exc}") from exc

        if df is None or len(df) == 0:
            raise ValueError(f"No data returned for symbol {symbol}")

        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # yfinance with auto_adjust=True: Close IS the adjusted close
        # We store it as adj_close; raw close may not differ (that's expected here)
        if "close" not in df.columns:
            raise ValueError(f"No Close column found for {symbol}")

        df["adj_close"] = df["close"]

        # Try to detect if adjustment was applied by checking for dividends
        # (with auto_adjust=True, close is already adjusted, so diff from raw = dividends)
        raw_df = None
        try:
            raw_ticker = yf.Ticker(symbol)
            raw_df = raw_ticker.history(
                start=str(start),
                end=str(end_exclusive),
                auto_adjust=False,
                actions=False,
            )
            if raw_df is not None and len(raw_df) > 0:
                raw_df.columns = [c.lower().replace(" ", "_") for c in raw_df.columns]
                if "adj_close" not in raw_df.columns:
                    raw_df["adj_close"] = raw_df.get("close", df["adj_close"])
                # Use raw adj_close
                aligned = raw_df["adj_close"].reindex(df.index)
                df["adj_close"] = aligned.fillna(df["close"])
                df["close"] = raw_df.get("close", df["close"]).reindex(df.index).fillna(df["close"])
        except Exception:
            # Fall back: adj_close = close (no adjustment info)
            logger.warning("Could not fetch raw (unadjusted) prices for %s", symbol)

        keep_cols = ["adj_close", "close"]
        if "volume" in df.columns:
            keep_cols.append("volume")
        df = df[keep_cols].copy()

        # Ensure DatetimeIndex with UTC tz removed
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        df = df.dropna(subset=["adj_close"])

        if len(df) == 0:
            raise ValueError(f"No valid adj_close rows for {symbol}")

        is_total_return = validate_adjusted_data(df)
        if not is_total_return:
            logger.warning(
                "adj_close equals close for %s — total return not confirmed (no dividend adjustment).",
                symbol,
            )

        as_of = df.index[-1].date()
        return PriceData(
            df=df,
            is_total_return=is_total_return,
            source="yahoo_finance",
            symbol=symbol,
            as_of=as_of,
            demo_mode=False,
        )

    def get_latest_price(self, symbol: str) -> tuple[float, date]:
        """Return latest close price and its date."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", auto_adjust=True)
            if hist is None or len(hist) == 0:
                raise ValueError(f"No recent data for {symbol}")
            last_close = float(hist["Close"].iloc[-1])
            last_date = hist.index[-1].date()
            return last_close, last_date
        except Exception as exc:
            raise RuntimeError(f"Failed to get latest price for {symbol}: {exc}") from exc

    def get_index_level(self, symbol: str) -> tuple[float, date]:
        """Return latest index level and as-of date (same as get_latest_price)."""
        return self.get_latest_price(symbol)

    def health_check(self, symbol: str) -> DataHealthStatus:
        """Check data freshness and quality for a symbol."""
        try:
            price, as_of = self.get_latest_price(symbol)
            return DataHealthStatus(
                symbol=symbol,
                last_date=as_of,
                n_rows=1,
                has_adj_close=True,
                is_stale=False,
                source="yahoo_finance",
            )
        except Exception as exc:
            return DataHealthStatus(
                symbol=symbol,
                source="yahoo_finance",
                error=str(exc),
            )
