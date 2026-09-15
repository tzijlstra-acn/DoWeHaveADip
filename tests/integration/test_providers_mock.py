"""Integration tests for data providers using mocks."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestYahooProviderMock:
    def _make_mock_history(self, n: int = 252, adj_differs: bool = True) -> pd.DataFrame:
        """Create a mock yfinance history DataFrame."""
        idx = pd.bdate_range("2020-01-01", periods=n)
        close = [100.0 + i * 0.1 for i in range(n)]
        adj = [v * 1.001 for v in close] if adj_differs else close
        return pd.DataFrame(
            {
                "Close": close,
                "Adj Close": adj,
                "Volume": [1_000_000] * n,
            },
            index=pd.DatetimeIndex([t.replace(tzinfo=None) for t in idx]),
        )

    def test_yahoo_validates_adj_close_present(self):
        """Provider should accept data when adj_close is present and differs from close."""
        from dipdca.data.providers.yahoo import YahooProvider

        mock_hist = self._make_mock_history(adj_differs=True)

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_hist
            mock_ticker_cls.return_value = mock_ticker

            provider = YahooProvider()
            result = provider.get_price_data("QQQ", date(2020, 1, 1), date(2020, 12, 31))

        assert result is not None
        assert "adj_close" in result.df.columns
        assert len(result.df) > 0

    def test_yahoo_fails_gracefully_on_empty_data(self):
        """Empty DataFrame from yfinance should raise ValueError."""
        from dipdca.data.providers.yahoo import YahooProvider

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_cls.return_value = mock_ticker

            provider = YahooProvider()
            with pytest.raises((ValueError, RuntimeError)):
                provider.get_price_data("INVALID", date(2020, 1, 1), date(2020, 12, 31))

    def test_yahoo_fails_gracefully_on_exception(self):
        """yfinance exception should be wrapped in RuntimeError."""
        from dipdca.data.providers.yahoo import YahooProvider

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.side_effect = Exception("Network error")
            mock_ticker_cls.return_value = mock_ticker

            provider = YahooProvider()
            with pytest.raises((ValueError, RuntimeError)):
                provider.get_price_data("QQQ", date(2020, 1, 1), date(2020, 12, 31))


class TestEcbFxProviderMock:
    ECB_SAMPLE_CSV = """KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2020-01-02,1.1213
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2020-01-03,1.1198
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2020-01-06,1.1156
"""

    def test_ecb_fx_parses_csv_correctly(self):
        """ECB provider should parse the TIME_PERIOD/OBS_VALUE CSV format."""
        from dipdca.data.providers.ecb_fx import EcbFxProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self.ECB_SAMPLE_CSV
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            provider = EcbFxProvider()
            result = provider.get_rates(["USD"], date(2020, 1, 1), date(2020, 1, 10))

        assert "USD" in result.columns
        assert len(result) > 0
        # Check values are numeric and in plausible range
        assert (result["USD"] > 0.5).all()
        assert (result["USD"] < 5.0).all()

    def test_ecb_fx_eur_always_one(self):
        """EUR column should always be 1.0."""
        from dipdca.data.providers.ecb_fx import EcbFxProvider

        mock_response = MagicMock()
        mock_response.text = self.ECB_SAMPLE_CSV
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            provider = EcbFxProvider()
            result = provider.get_rates(["USD"], date(2020, 1, 1), date(2020, 1, 10))

        assert "EUR" in result.columns
        assert (result["EUR"] == 1.0).all()

    def test_ecb_fx_handles_http_error(self):
        """HTTP error should not crash — should return empty/EUR-only frame."""
        from dipdca.data.providers.ecb_fx import EcbFxProvider

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            provider = EcbFxProvider()
            result = provider.get_rates(["USD"], date(2020, 1, 1), date(2020, 1, 10))

        # Should return something (possibly just EUR=1.0)
        assert isinstance(result, pd.DataFrame)


class TestDataStaleness:
    def test_stale_data_warning(self):
        """DataHealthStatus.is_stale should be True if data is old."""
        from datetime import timedelta

        from dipdca.data.validation import check_stale

        old_date = date.today() - timedelta(days=10)
        assert check_stale(old_date, threshold_days=5) is True

    def test_fresh_data_not_stale(self):
        from dipdca.data.validation import check_stale

        fresh_date = date.today()
        assert check_stale(fresh_date, threshold_days=5) is False
