"""FX conversion utilities using ECB convention."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ECB convention: q_currency = units of currency per 1 EUR
# q_EUR = 1.0 always
# Example: q_USD = 1.08 means 1 EUR = 1.08 USD


def fx_base_per_asset(q_base: float, q_asset: float) -> float:
    """Compute FX rate: units of base currency per unit of asset currency.

    Args:
        q_base: ECB quote rate for base currency (units per EUR).
        q_asset: ECB quote rate for asset currency (units per EUR).

    Returns:
        FX rate: base per asset = q_base / q_asset.
        E.g. EUR base, USD asset: EUR/USD = 1.0 / 1.08 ≈ 0.926 EUR per USD.
    """
    if q_asset == 0:
        raise ValueError("q_asset cannot be zero")
    return q_base / q_asset


def value_in_base(asset_price: float, q_base: float, q_asset: float) -> float:
    """Convert asset price to base currency.

    Args:
        asset_price: Price in asset's native currency.
        q_base: ECB quote for base currency.
        q_asset: ECB quote for asset currency.

    Returns:
        Asset price expressed in base currency.
    """
    return asset_price * fx_base_per_asset(q_base, q_asset)


def log_return_decomposition(
    tr_base: pd.Series, tr_native: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Decompose total return in base into native return and FX return.

    Uses log returns: ln(TR_base) = ln(TR_native) + ln(FX_ratio)

    Args:
        tr_base: Total return index in base currency (rebased to 1.0).
        tr_native: Total return index in native currency (rebased to 1.0).

    Returns:
        Tuple of (native_log_returns, fx_log_returns) as daily series.
    """
    log_base = np.log(tr_base).diff().dropna()
    log_native = np.log(tr_native).diff().dropna()

    # Align on common index
    common = log_base.index.intersection(log_native.index)
    log_base = log_base.loc[common]
    log_native = log_native.loc[common]

    fx_log_returns = log_base - log_native
    return log_native, fx_log_returns


def rebase_to_common_currency(
    price_series: pd.Series,
    fx_series: pd.Series,
    t0: pd.Timestamp,
) -> pd.Series:
    """Convert price_series (in native currency) to base using fx_series.

    Args:
        price_series: Prices in asset native currency (DatetimeIndex).
        fx_series: FX rates base_per_asset (DatetimeIndex), aligned to price_series.
        t0: Reference date for normalization.

    Returns:
        Price series expressed in base currency.
    """
    fx_aligned = fx_series.reindex(price_series.index, method="ffill")
    return price_series * fx_aligned


def instrument_to_base_currency(
    instrument_df: pd.DataFrame,
    from_currency: str,
    to_currency: str,
    ecb_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Convert all price/value columns of an instrument DataFrame to ``to_currency``.

    ECB convention: ``ecb_rates[currency]`` = units of that currency per 1 EUR.

    For EUR base and USD instrument:
        EUR_per_USD = 1 / USD_per_EUR = 1 / ecb_rates["USD"]
        price_EUR   = price_USD * EUR_per_USD

    The benchmark index is NOT passed through this function — its drawdown
    must be computed in the index's own published native level.

    Columns converted: ``adj_close``, ``close``, and any others that are
    numeric and appear to be price columns (open, high, low).

    Args:
        instrument_df: DataFrame with DatetimeIndex and price columns.
        from_currency: The instrument's native quote currency (e.g. ``"USD"``).
        to_currency: Target base currency (e.g. ``"EUR"``).
        ecb_rates: DataFrame of ECB rates (units of currency per EUR),
                   as returned by ``EcbFxProvider.get_rates()``.

    Returns:
        New DataFrame with prices expressed in ``to_currency``.
        Original is unchanged.
    """
    from_upper = from_currency.upper()
    to_upper = to_currency.upper()
    if from_upper == to_upper:
        return instrument_df.copy()

    if from_upper == "EUR":
        # EUR → other: multiply by target units per EUR
        if to_upper not in ecb_rates.columns:
            raise ValueError(
                f"ECB rates do not contain {to_upper}. "
                f"Available: {list(ecb_rates.columns)}"
            )
        rate_series = ecb_rates[to_upper].reindex(instrument_df.index, method="ffill")
        multiplier = rate_series
    elif to_upper == "EUR":
        # other → EUR: divide by source units per EUR (= multiply by reciprocal)
        if from_upper not in ecb_rates.columns:
            raise ValueError(
                f"ECB rates do not contain {from_upper}. "
                f"Available: {list(ecb_rates.columns)}"
            )
        rate_series = ecb_rates[from_upper].reindex(instrument_df.index, method="ffill")
        # 1 unit of from_currency = 1/rate EUR
        multiplier = 1.0 / rate_series
    else:
        # Cross rate: from → EUR → to
        if from_upper not in ecb_rates.columns or to_upper not in ecb_rates.columns:
            raise ValueError(
                f"ECB rates missing {from_upper} or {to_upper}. "
                f"Available: {list(ecb_rates.columns)}"
            )
        from_rate = ecb_rates[from_upper].reindex(instrument_df.index, method="ffill")
        to_rate = ecb_rates[to_upper].reindex(instrument_df.index, method="ffill")
        multiplier = to_rate / from_rate

    price_cols = [c for c in ("adj_close", "close", "open", "high", "low")
                  if c in instrument_df.columns]
    result = instrument_df.copy()
    for col in price_cols:
        result[col] = instrument_df[col] * multiplier
    return result
