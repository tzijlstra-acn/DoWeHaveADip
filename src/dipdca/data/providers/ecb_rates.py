"""ECB deposit facility rate provider (cash rate for EUR zone)."""

from __future__ import annotations

import logging
from datetime import date
from io import StringIO

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://data-api.ecb.europa.eu/service/data"


class EcbRatesProvider:
    """Fetch ECB deposit facility (overnight) rates."""

    SERIES_KEY = "FM.B.U2.EUR.BN.J.1.%2B0.R.DFR...."

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def get_deposit_rate(self, start: date, end: date) -> pd.Series:
        """Return ECB deposit facility rate as daily series (annualized decimal).

        Args:
            start: Start date.
            end: End date.

        Returns:
            Daily Series with annualized rate (e.g. 0.04 = 4%), forward-filled.

        Raises:
            RuntimeError: If API call fails.
        """
        url = f"{BASE_URL}/FM/{self.SERIES_KEY}"
        params = {
            "startPeriod": str(start),
            "endPeriod": str(end),
            "format": "csvdata",
        }

        try:
            response = httpx.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
        except Exception as exc:
            raise RuntimeError(f"ECB rate fetch failed: {exc}") from exc

        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            raise RuntimeError(f"Unexpected ECB rate CSV format: {list(df.columns)}")

        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        df = df.set_index("TIME_PERIOD")
        rate_series = pd.to_numeric(df["OBS_VALUE"], errors="coerce").dropna() / 100.0

        # Reindex to daily and forward-fill
        daily_idx = pd.date_range(start, end, freq="D")
        rate_daily = rate_series.reindex(daily_idx, method="ffill")
        return rate_daily.fillna(0.0)

    def get_current_rate(self) -> float:
        """Return most recent ECB deposit rate."""

        today = date.today()
        start = date(today.year - 1, today.month, today.day)
        rates = self.get_deposit_rate(start, today)
        return float(rates.iloc[-1])
