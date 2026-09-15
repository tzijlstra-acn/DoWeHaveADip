"""Block bootstrap for statistical confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_returns(
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    """Block bootstrap resampling of return series.

    Preserves autocorrelation structure by sampling contiguous blocks.

    Args:
        returns: Daily returns series.
        n_simulations: Number of bootstrap samples.
        block_size: Size of each block in trading days.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of shape (len(returns), n_simulations).
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    vals = returns.values

    result = np.empty((n, n_simulations))
    for sim in range(n_simulations):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        resampled = np.concatenate([vals[s : s + block_size] for s in starts])[:n]
        result[:, sim] = resampled

    return pd.DataFrame(result, index=returns.index)


def bootstrap_metric(
    metric_fn,
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 21,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict:
    """Compute a metric's bootstrap confidence interval.

    Args:
        metric_fn: Function taking a pd.Series of returns, returning a float.
        returns: Daily returns series.
        n_simulations: Bootstrap iterations.
        block_size: Block size for block bootstrap.
        seed: Random seed.
        ci_level: Confidence level (e.g. 0.95 for 95% CI).

    Returns:
        Dict with 'mean', 'lower', 'upper', 'std'.
    """
    boot = block_bootstrap_returns(returns, n_simulations, block_size, seed)
    metrics = [metric_fn(boot.iloc[:, i]) for i in range(n_simulations)]
    metrics = [m for m in metrics if m is not None and not np.isnan(m)]
    if not metrics:
        return {"mean": None, "lower": None, "upper": None, "std": None}
    alpha = (1 - ci_level) / 2
    return {
        "mean": float(np.mean(metrics)),
        "lower": float(np.quantile(metrics, alpha)),
        "upper": float(np.quantile(metrics, 1 - alpha)),
        "std": float(np.std(metrics)),
    }
