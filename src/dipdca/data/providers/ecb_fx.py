"""ECB Foreign Exchange Reference Rates provider."""

from __future__ import annotations

import logging
from datetime import date
from io import StringIO

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://data-api.ecb.europa.eu/service/data"
DATASET = "EXR"
SERIES_KEY_TEMPLATE = "D.{currency}.EUR.SP00.A"


class EcbFxProvider:
    """Fetch ECB FX reference rates (units per EUR) via REST API."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def get_rates(
        self,
        currencies: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return DataFrame of ECB FX rates (units of currency per EUR).

        Args:
            currencies: List of ISO 4217 currency codes (e.g. ['USD', 'CHF']).
            start: Start date.
            end: End date.

        Returns:
            DataFrame with DatetimeIndex, one column per currency.
            EUR is always 1.0 by convention.

        Raises:
            RuntimeError: If the ECB API call fails.
        """
        frames = {}
        for currency in currencies:
            if currency.upper() == "EUR":
                continue
            try:
                df = self._fetch_single(currency, start, end)
                frames[currency] = df
            except Exception as exc:
                logger.warning("ECB FX fetch failed for %s: %s", currency, exc)

        if not frames:
            # Return empty frame with EUR only
            idx = pd.date_range(start, end, freq="B")
            return pd.DataFrame({"EUR": 1.0}, index=idx)

        # Combine all currencies
        combined = pd.concat(frames.values(), axis=1, keys=frames.keys())
        combined.columns = list(frames.keys())
        combined["EUR"] = 1.0
        combined = combined.ffill()
        return combined

    def _fetch_single(self, currency: str, start: date, end: date) -> pd.Series:
        """Fetch a single currency's FX rate from ECB."""
        series_key = SERIES_KEY_TEMPLATE.format(currency=currency.upper())
        url = f"{BASE_URL}/{DATASET}/{series_key}"
        params = {
            "startPeriod": str(start),
            "endPeriod": str(end),
            "format": "csvdata",
        }

        response = httpx.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        return self._parse_csv(response.text, currency)

    def _parse_csv(self, csv_text: str, currency: str) -> pd.Series:
        """Parse ECB CSV response into a date-indexed Series."""
        df = pd.read_csv(StringIO(csv_text))

        # ECB CSV has TIME_PERIOD and OBS_VALUE columns
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            raise ValueError(f"Unexpected ECB CSV format for {currency}: {list(df.columns)}")

        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        df = df.set_index("TIME_PERIOD")
        series = pd.to_numeric(df["OBS_VALUE"], errors="coerce").dropna()
        series.name = currency
        return series
