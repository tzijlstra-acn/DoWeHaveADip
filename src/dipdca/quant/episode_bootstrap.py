"""Paired bootstrap over independent ATH episodes.

With ~8 independent episodes, a naive point estimate of the win rate has huge
sampling uncertainty: 7/8 could easily be 6/8 or 8/8 by chance. This module
bootstraps episode-level paired comparisons to produce honest confidence
intervals on win rates and median-vs-DCA outcomes.

Design
------
- The comparison is always *paired within an episode* — every policy in a
  bootstrap resample faces the same market path, contributions, and timing.
- Resampling is at the episode level, not the day level, so the temporal
  autocorrelation inside each episode is preserved.
- Only episodes that actually generated a trade (deployed > 0) and have a
  valid DCA outcome at the requested horizon are included in the pool.

Usage
-----
    from dipdca.quant.episode_bootstrap import bootstrap_win_rate

    studies = run_event_study(...)
    result = bootstrap_win_rate(
        studies,
        threshold=-0.15,
        horizon_label="12m",
        policy="ATH all-in",
    )
    print(f"Win rate: {result.win_rate:.0%}  95% CI [{result.ci_lower:.0%}, {result.ci_upper:.0%}]")
    print(f"Based on {result.n_episodes} independent episodes, {result.n_boot} bootstrap draws")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dipdca.quant.ath_episodes import EpisodeStudy

DEFAULT_N_BOOT = 2_000
DEFAULT_CI_LEVEL = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap estimate of policy vs DCA outcomes across ATH episodes.

    All fractions (win_rate, ci_lower, ci_upper, median_vs_dca, etc.) are
    expressed as decimals — multiply by 100 for display.
    """

    threshold: float
    horizon_label: str
    policy: str

    n_episodes: int
    n_boot: int
    ci_level: float

    # Win-rate bootstrap
    win_rate: float
    ci_lower: float
    ci_upper: float

    # Median relative return bootstrap (policy ending wealth / DCA ending wealth - 1)
    median_vs_dca: float
    median_ci_lower: float
    median_ci_upper: float

    # Raw percentiles of the point-estimate distribution (for display)
    p10_vs_dca: float
    p25_vs_dca: float
    p75_vs_dca: float
    p90_vs_dca: float


def _filter_eligible(
    studies: list[EpisodeStudy],
    threshold: float,
    horizon_label: str,
    policy: str,
) -> list[EpisodeStudy]:
    """Return only studies that have paired, non-None outcomes for policy and DCA."""
    eligible = []
    for s in studies:
        if s.threshold != threshold:
            continue
        dca_w = s.wealth("DCA", horizon_label)
        pol_w = s.wealth(policy, horizon_label)
        if dca_w is None or pol_w is None or dca_w <= 0:
            continue
        eligible.append(s)
    return eligible


def _relative_returns(studies: list[EpisodeStudy], policy: str, horizon_label: str) -> np.ndarray:
    """Array of (policy / DCA - 1) for a list of eligible studies."""
    return np.array([s.relative_to_dca(policy, horizon_label) for s in studies], dtype=float)


def bootstrap_win_rate(
    studies: list[EpisodeStudy],
    threshold: float,
    horizon_label: str,
    policy: str = "ATH all-in",
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int | None = 42,
) -> BootstrapResult | None:
    """Bootstrap confidence interval on the win rate of ``policy`` vs DCA.

    Returns ``None`` when fewer than 2 eligible episodes are available (CI
    is undefined with 0 or 1 observations).

    Args:
        studies: Output of ``run_event_study``.
        threshold: The drawdown threshold to analyse (e.g. ``-0.15``).
        horizon_label: Horizon label (e.g. ``"12m"`` or ``"recovery"``).
        policy: Policy label to compare against DCA.
        n_boot: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95 → 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``BootstrapResult`` or ``None`` if the pool is too small.
    """
    eligible = _filter_eligible(studies, threshold, horizon_label, policy)
    n = len(eligible)
    if n < 2:
        return None

    rels = _relative_returns(eligible, policy, horizon_label)
    rng = np.random.default_rng(seed)

    # Bootstrap: resample n episodes with replacement, compute win rate and median
    boot_win_rates = np.empty(n_boot, dtype=float)
    boot_medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = rels[idx]
        boot_win_rates[i] = float(np.mean(sample > 0))
        boot_medians[i] = float(np.median(sample))

    alpha = 1.0 - ci_level
    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0

    # Point estimates on the full eligible set
    point_win_rate = float(np.mean(rels > 0))
    point_median = float(np.median(rels))

    return BootstrapResult(
        threshold=threshold,
        horizon_label=horizon_label,
        policy=policy,
        n_episodes=n,
        n_boot=n_boot,
        ci_level=ci_level,
        win_rate=point_win_rate,
        ci_lower=float(np.quantile(boot_win_rates, lo)),
        ci_upper=float(np.quantile(boot_win_rates, hi)),
        median_vs_dca=point_median,
        median_ci_lower=float(np.quantile(boot_medians, lo)),
        median_ci_upper=float(np.quantile(boot_medians, hi)),
        p10_vs_dca=float(np.quantile(rels, 0.10)),
        p25_vs_dca=float(np.quantile(rels, 0.25)),
        p75_vs_dca=float(np.quantile(rels, 0.75)),
        p90_vs_dca=float(np.quantile(rels, 0.90)),
    )


def bootstrap_all_thresholds(
    studies: list[EpisodeStudy],
    thresholds: tuple[float, ...],
    horizon_label: str,
    policy: str = "ATH all-in",
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int | None = 42,
) -> list[BootstrapResult]:
    """Run bootstrap win-rate estimation across every threshold.

    Skips thresholds with fewer than 2 eligible episodes.
    """
    results = []
    for th in thresholds:
        r = bootstrap_win_rate(
            studies=studies,
            threshold=th,
            horizon_label=horizon_label,
            policy=policy,
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed,
        )
        if r is not None:
            results.append(r)
    return results


def results_to_dataframe(results: list[BootstrapResult]) -> pd.DataFrame:
    """Convert a list of bootstrap results to a display-ready DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Threshold": f"{r.threshold:.0%}",
            "Episodes": r.n_episodes,
            "Win rate": r.win_rate,
            f"CI lower ({r.ci_level:.0%})": r.ci_lower,
            f"CI upper ({r.ci_level:.0%})": r.ci_upper,
            "Median vs DCA": r.median_vs_dca,
            "P10 vs DCA": r.p10_vs_dca,
            "P90 vs DCA": r.p90_vs_dca,
        })
    return pd.DataFrame(rows)
