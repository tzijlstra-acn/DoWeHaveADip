"""Swiss National Bank policy rate provider (cash rate for CHF zone).

SNB Data Portal API v2 (migrated from /api/cube/snbsnbzisratep/ which returned 404):
  Endpoint: https://data.snb.ch/api/cube/snbgwdzid/data/csv/en
  Series D0=LZ is the SNB policy rate (Leitzinssatz), in percent annualized.
  Format: BOM + semicolon-separated CSV with header rows, then Date;D0;Value rows.
"""

from __future__ import annotations

import logging
from datetime import date
from io import StringIO

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# SNB Data Portal — interest rates cube (snbgwdzid replaces deprecated snbsnbzisratep)
SNB_RATE_URL = "https://data.snb.ch/api/cube/snbgwdzid/data/csv/en"
# LZ = Leitzinssatz (SNB policy rate), SARON = overnight rate, ENG = special rate
SNB_POLICY_SERIES = "LZ"


class SnbRatesProvider:
    """Fetch SNB sight deposit rate (CHF cash rate)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def get_policy_rate(self, start: date, end: date) -> pd.Series:
        """Return SNB policy rate (LZ) as daily series (annualized decimal).

        Args:
            start: Start date.
            end: End date.

        Returns:
            Daily rate series, forward-filled. Returns 0.0 where data is missing.
        """
        try:
            response = httpx.get(SNB_RATE_URL, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"SNB rate fetch failed: {exc}") from exc

        try:
            # Strip BOM and find the data section (after the two header rows)
            text = response.text.lstrip("﻿")
            lines = text.splitlines()

            # Locate the row containing "Date" to find data start
            data_start = next(
                (i for i, ln in enumerate(lines) if ln.startswith('"Date"')),
                None,
            )
            if data_start is None:
                raise RuntimeError("Could not locate data header in SNB CSV response")

            data_text = "\n".join(lines[data_start:])
            df = pd.read_csv(
                StringIO(data_text),
                sep=";",
                quotechar='"',
                names=["date", "series", "value"],
                header=0,
            )

            # Filter to the policy rate series
            lz = df[df["series"] == SNB_POLICY_SERIES].copy()
            if lz.empty:
                raise RuntimeError(f"Series '{SNB_POLICY_SERIES}' not found in SNB response")

            lz["date"] = pd.to_datetime(lz["date"], errors="coerce")
            lz = lz.dropna(subset=["date"]).set_index("date")
            rate_series = pd.to_numeric(lz["value"], errors="coerce").dropna() / 100.0

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Could not parse SNB rate CSV: {exc}") from exc

        # Filter to requested window and forward-fill on a daily index
        rate_series = rate_series.loc[str(start):str(end)]
        daily_idx = pd.date_range(start, end, freq="D")
        rate_daily = rate_series.reindex(daily_idx, method="ffill")
        return rate_daily.fillna(0.0)
