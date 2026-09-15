"""
Longest-possible historical price series from Yahoo Finance.

Tickers with deep history:
  ^GSPC  — S&P 500, daily from 1927-12-30
  ^DJI   — Dow Jones, daily from 1928-10-01
  ^IXIC  — NASDAQ Composite, from 1971
  ^STOXX50E — Euro Stoxx 50, from 1986
  ^GDAXI — DAX, from 1987
  ^AEX   — AEX Amsterdam, from 1983

Returns monthly resampled (last business day of month) total return index.
Monthly resampling: less noise, better for long-horizon analysis.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

DEEP_HISTORY_TICKERS = {
    "S&P 500 (1928-)": "^GSPC",
    "Dow Jones (1928-)": "^DJI",
    "NASDAQ (1971-)": "^IXIC",
    "Euro Stoxx 50 (1987-)": "^STOXX50E",
    "DAX (1988-)": "^GDAXI",
    "AEX Amsterdam (1983-)": "^AEX",
}


@st.cache_data(ttl=86400)  # cache 24h — deep history doesn't change
def fetch_deep_history(ticker: str = "^GSPC") -> pd.Series:
    """Fetch full price history for ticker, return as monthly total-return index (base=100).

    Uses Adj Close (auto_adjust=True) to capture dividend reinvestment.

    Args:
        ticker: Yahoo Finance ticker symbol.

    Returns:
        pd.Series with DatetimeIndex (month-end), name = ticker, base = 100.
    """
    raw = yf.download(ticker, start="1920-01-01", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Use Close (after auto_adjust this is adjusted close)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].squeeze().dropna()
    else:
        prices = raw["Close"].squeeze().dropna()

    # Resample to monthly (last trading day of month)
    monthly = prices.resample("ME").last().dropna()

    # Rebase to 100
    tr_index = monthly / monthly.iloc[0] * 100
    tr_index.name = ticker
    return tr_index


@st.cache_data(ttl=86400)
def fetch_deep_history_daily(ticker: str = "^GSPC") -> pd.Series:
    """Fetch full daily price history for ticker.

    Returns raw adjusted close series (not rebased), for use with parameter sweep.

    Args:
        ticker: Yahoo Finance ticker symbol.

    Returns:
        pd.Series with daily DatetimeIndex, name = ticker.
    """
    raw = yf.download(ticker, start="1920-01-01", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].squeeze().dropna()
    else:
        prices = raw["Close"].squeeze().dropna()

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.name = ticker
    return prices.dropna()
