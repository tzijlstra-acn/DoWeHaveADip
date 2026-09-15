"""Generate synthetic fixture data for offline demo mode.

Run with: uv run python data/fixtures/generate.py

Generates deterministic, reproducible synthetic price series.
NO real market data is used or committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

FIXTURES_DIR = Path(__file__).parent
RNG = np.random.default_rng(42)  # Fixed seed for reproducibility

TRADING_DAYS_PER_YEAR = 252
YEARS = 5
N_DAYS = TRADING_DAYS_PER_YEAR * YEARS
START_DATE = pd.Timestamp("2018-01-02")


def business_dates(n: int, start: pd.Timestamp = START_DATE) -> pd.DatetimeIndex:
    """Generate N business days starting from start."""
    return pd.bdate_range(start=start, periods=n)


def gbm_prices(
    mu: float,
    sigma: float,
    s0: float = 100.0,
    n: int = N_DAYS,
    seed_extra: int = 0,
) -> np.ndarray:
    """Geometric Brownian Motion price path."""
    rng = np.random.default_rng(42 + seed_extra)
    dt = 1 / TRADING_DAYS_PER_YEAR
    returns = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    prices = s0 * np.exp(np.cumsum(returns))
    return np.concatenate([[s0], prices[:-1]])


def make_price_df(prices: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a price DataFrame with adj_close, close, volume columns."""
    # Add slight noise to close vs adj_close to simulate dividend adjustment
    rng = np.random.default_rng(99)
    # adj_close reflects dividend reinvestment — slightly higher than close over time
    adj_factor = np.cumprod(1 + rng.uniform(0, 0.0002, len(prices)))
    close = prices.copy()
    adj_close = prices * adj_factor

    volume = rng.integers(100_000, 10_000_000, len(prices)).astype(float)

    return pd.DataFrame(
        {
            "adj_close": adj_close,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def generate_bull_market() -> pd.DataFrame:
    """5-year steady bull market: ~12% CAGR, 15% vol."""
    idx = business_dates(N_DAYS)
    prices = gbm_prices(mu=0.12, sigma=0.15, seed_extra=1)
    return make_price_df(prices, idx)


def generate_crash_and_recovery() -> pd.DataFrame:
    """Bull market → 40% crash → full recovery."""
    n_bull = N_DAYS // 3
    n_crash = 60  # ~3 months of selling
    n_recovery = N_DAYS - n_bull - n_crash

    idx = business_dates(N_DAYS)

    rng = np.random.default_rng(43)
    dt = 1 / TRADING_DAYS_PER_YEAR

    # Bull phase
    bull = np.exp(
        np.cumsum(rng.normal((0.10 - 0.5 * 0.14**2) * dt, 0.14 * np.sqrt(dt), n_bull))
    )
    s_peak = bull[-1]

    # Crash phase: rapid decline to ~60% of peak
    crash_target = s_peak * 0.60
    crash = np.linspace(s_peak, crash_target, n_crash)
    crash += rng.normal(0, crash_target * 0.01, n_crash)

    # Recovery phase
    s_trough = crash[-1]
    recovery = s_trough * np.exp(
        np.cumsum(rng.normal((0.18 - 0.5 * 0.20**2) * dt, 0.20 * np.sqrt(dt), n_recovery))
    )

    # Scale bull to start at 100
    bull_scaled = bull / bull[0] * 100.0
    # Concatenate: bull, crash, recovery — truncate/pad to exactly N_DAYS
    raw = np.concatenate([bull_scaled, crash, recovery])
    prices = raw[:N_DAYS]
    if len(prices) < N_DAYS:
        prices = np.pad(prices, (0, N_DAYS - len(prices)), mode="edge")
    return make_price_df(prices, idx)


def generate_sideways() -> pd.DataFrame:
    """5-year sideways market: ~2% CAGR with 20% vol."""
    idx = business_dates(N_DAYS)
    prices = gbm_prices(mu=0.02, sigma=0.20, seed_extra=2)
    return make_price_df(prices, idx)


def generate_demo_fx() -> pd.DataFrame:
    """Stable FX rates: EUR=1.0, USD≈1.08±0.05, CHF≈0.93±0.03."""
    idx = business_dates(N_DAYS)
    rng = np.random.default_rng(77)

    n = len(idx)
    # Slow-moving FX with mean reversion
    usd_shock = rng.normal(0, 0.003, n)
    chf_shock = rng.normal(0, 0.002, n)

    usd = np.zeros(n)
    chf = np.zeros(n)
    usd[0] = 1.08
    chf[0] = 0.93

    for i in range(1, n):
        usd[i] = usd[i - 1] * 0.999 + 1.08 * 0.001 + usd_shock[i]
        chf[i] = chf[i - 1] * 0.999 + 0.93 * 0.001 + chf_shock[i]

    # Clamp to realistic ranges
    usd = np.clip(usd, 0.90, 1.30)
    chf = np.clip(chf, 0.85, 1.05)

    return pd.DataFrame(
        {
            "EUR": 1.0,
            "USD": usd,
            "CHF": chf,
        },
        index=idx,
    )


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic fixtures...")

    bull = generate_bull_market()
    bull.to_parquet(FIXTURES_DIR / "bull_market.parquet")
    print(f"  bull_market.parquet: {len(bull)} rows, close range [{bull['adj_close'].min():.2f}, {bull['adj_close'].max():.2f}]")

    crash = generate_crash_and_recovery()
    crash.to_parquet(FIXTURES_DIR / "crash_and_recovery.parquet")
    print(f"  crash_and_recovery.parquet: {len(crash)} rows")

    sideways = generate_sideways()
    sideways.to_parquet(FIXTURES_DIR / "sideways.parquet")
    print(f"  sideways.parquet: {len(sideways)} rows")

    fx = generate_demo_fx()
    fx.to_parquet(FIXTURES_DIR / "demo_fx.parquet")
    print(f"  demo_fx.parquet: {len(fx)} rows, USD range [{fx['USD'].min():.3f}, {fx['USD'].max():.3f}]")

    print("Done. All fixtures are synthetic — no real market data.")


if __name__ == "__main__":
    main()
