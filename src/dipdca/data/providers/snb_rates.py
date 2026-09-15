"""Swiss National Bank policy rate provider (cash rate for CHF zone)."""

from __future__ import annotations

import logging
from datetime import date

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

SNB_RATE_URL = "https://data.snb.ch/api/cube/snbsnbzisratep/data/csv/en"


class SnbRatesProvider:
    """Fetch SNB sight deposit rate (CHF cash rate)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def get_policy_rate(self, start: date, end: date) -> pd.Series:
        """Return SNB sight deposit rate as daily series (annualized decimal).

        Args:
            start: Start date.
            end: End date.

        Returns:
            Daily rate series, forward-filled.
        """
        try:
            response = httpx.get(SNB_RATE_URL, timeout=self.timeout)
            response.raise_for_status()
            from io import StringIO

            df = pd.read_csv(StringIO(response.text), sep=";", skiprows=3)
        except Exception as exc:
            raise RuntimeError(f"SNB rate fetch failed: {exc}") from exc

        # SNB CSV format: semicolon-separated, first col = date (YYYY-MM-DD), last col = value
        try:
            df.columns = [c.strip().lower() for c in df.columns]
            # Drop empty columns
            df = df.dropna(axis=1, how="all")
            df = df[[c for c in df.columns if not df[c].astype(str).str.strip().eq("").all()]]
            # First column is the date, last numeric column is the rate
            date_col = df.columns[0]
            numeric_cols = [c for c in df.columns[1:] if pd.to_numeric(df[c], errors="coerce").notna().any()]
            if not numeric_cols:
                raise RuntimeError("No numeric column found in SNB CSV")
            val_col = numeric_cols[-1]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)
            rate_series = pd.to_numeric(df[val_col], errors="coerce").dropna() / 100.0
        except Exception as exc:
            raise RuntimeError(f"Could not parse SNB rate CSV: {exc}") from exc

        daily_idx = pd.date_range(start, end, freq="D")
        rate_daily = rate_series.reindex(daily_idx, method="ffill")
        return rate_daily.fillna(0.0)
