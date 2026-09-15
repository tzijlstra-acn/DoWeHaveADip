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
