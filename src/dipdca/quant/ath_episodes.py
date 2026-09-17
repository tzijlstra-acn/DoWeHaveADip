"""ATH-episode event study.

An aggregate backtest over one long window answers the wrong question: a few
winning crashes disappear inside years of cash drag. The question is conditional —
*given the index has just fallen X% below its previous all-time high, what happened
next?* — so the unit of observation must be the drawdown episode, not the calendar.

Episode definition
------------------
An episode is anchored to a single all-time high and stays open until the
benchmark closes at or above **that** level again. Therefore:

- the anchor ATH never moves during the episode;
- each threshold records only its **first** crossing, so adjacent crash days
  cannot be counted as separate scenarios;
- an episode still open when the data ends is *censored*, not discarded, and is
  reported separately rather than being treated as a recovery.

All policies within an episode receive an identical external contribution series
and see the same market path, so the comparison is paired.

Signal and execution stay separated throughout: the benchmark index generates
crossings, and the investable instrument supplies execution prices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from dipdca.models import DeploymentTier, SimulationParams
from dipdca.quant.backtest import run_ath_deployment, run_dca, run_savings_only

logger = logging.getLogger(__name__)

# Thresholds reported by default (negative fractions).
DEFAULT_THRESHOLDS: tuple[float, ...] = (-0.10, -0.15, -0.20, -0.25, -0.30, -0.35)

# Horizons reported after the first crossing.
DEFAULT_HORIZON_MONTHS: tuple[int, ...] = (12, 36, 60)


@dataclass(frozen=True)
class ATHEpisode:
    """One drawdown episode anchored to a single all-time high."""

    ath_date: pd.Timestamp
    ath_level: float
    trough_date: pd.Timestamp
    trough_drawdown: float
    recovery_date: pd.Timestamp | None
    first_crossings: dict[float, pd.Timestamp] = field(default_factory=dict)

    @property
    def is_censored(self) -> bool:
        """True when the anchor ATH was never recovered within the data."""
        return self.recovery_date is None

    @property
    def days_to_recovery(self) -> int | None:
        if self.recovery_date is None:
            return None
        return int((self.recovery_date - self.ath_date).days)

    def crossed(self, threshold: float) -> bool:
        return threshold in self.first_crossings

    def days_to_crossing(self, threshold: float) -> int | None:
        dt = self.first_crossings.get(threshold)
        if dt is None:
            return None
        return int((dt - self.ath_date).days)


def find_ath_episodes(
    benchmark: pd.Series,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    min_depth: float | None = None,
    initial_ath: float | None = None,
    initial_ath_date: pd.Timestamp | None = None,
) -> list[ATHEpisode]:
    """Identify independent ATH episodes in a benchmark close series.

    Args:
        benchmark: Benchmark index closes, indexed by date and sorted ascending.
        thresholds: Drawdown levels whose first crossing should be recorded.
        min_depth: Discard episodes shallower than this (negative fraction).
            Defaults to the shallowest supplied threshold, so an episode is only
            reported once it is deep enough to be a decision point.
        initial_ath: Seed the opening all-time high from pre-window history.
            When provided, the episode detection starts with this as the peak
            rather than the first value in the series.  Use this to avoid the
            first bar of the evaluation window being treated as a new ATH.
        initial_ath_date: Date corresponding to ``initial_ath``.  Ignored when
            ``initial_ath`` is ``None``.

    Returns:
        Episodes in chronological order. A trailing censored episode is included
        with ``recovery_date=None``.
    """
    series = benchmark.dropna()
    if len(series) == 0:
        return []

    if min_depth is None:
        min_depth = max(thresholds) if thresholds else -0.10

    episodes: list[ATHEpisode] = []

    if initial_ath is not None:
        peak = float(initial_ath)
        peak_date = (
            pd.Timestamp(initial_ath_date)
            if initial_ath_date is not None
            else pd.Timestamp(series.index[0]) - pd.Timedelta(days=1)
        )
    else:
        peak = float(series.iloc[0])
        peak_date = pd.Timestamp(series.index[0])

    open_episode = False
    anchor_level = peak
    anchor_date = peak_date
    trough_dd = 0.0
    trough_date = peak_date
    crossings: dict[float, pd.Timestamp] = {}

    def close_episode(recovery: pd.Timestamp | None) -> None:
        if trough_dd <= min_depth:
            episodes.append(
                ATHEpisode(
                    ath_date=anchor_date,
                    ath_level=anchor_level,
                    trough_date=trough_date,
                    trough_drawdown=trough_dd,
                    recovery_date=recovery,
                    first_crossings=dict(crossings),
                )
            )

    for raw_dt, raw_level in series.items():
        dt = pd.Timestamp(raw_dt)  # type: ignore[arg-type]
        level = float(raw_level)

        if level >= peak:
            # Recovering the anchor closes the episode; a new ATH is established.
            if open_episode:
                close_episode(recovery=dt)
                open_episode = False
            peak = level
            peak_date = dt
            continue

        dd = level / peak - 1.0

        if not open_episode:
            open_episode = True
            anchor_level = peak
            anchor_date = peak_date
            trough_dd = dd
            trough_date = dt
            crossings = {}
        elif dd < trough_dd:
            trough_dd = dd
            trough_date = dt

        for th in thresholds:
            if th not in crossings and dd <= th:
                crossings[th] = dt

    if open_episode:
        close_episode(recovery=None)

    return episodes


@dataclass(frozen=True)
class PolicyOutcome:
    """Ending wealth for one policy measured at one horizon."""

    policy: str
    horizon_label: str
    ending_wealth: float
    total_contributions: float
    n_deployments: int
    ending_cash: float


@dataclass(frozen=True)
class EpisodeStudy:
    """Outcomes for every policy in a single episode at a single threshold."""

    episode: ATHEpisode
    threshold: float
    signal_date: pd.Timestamp
    outcomes: list[PolicyOutcome]

    def wealth(self, policy: str, horizon_label: str) -> float | None:
        for o in self.outcomes:
            if o.policy == policy and o.horizon_label == horizon_label:
                return o.ending_wealth
        return None

    def relative_to_dca(self, policy: str, horizon_label: str) -> float | None:
        """Ending wealth relative to month-end DCA, as a fraction."""
        dca = self.wealth("DCA", horizon_label)
        mine = self.wealth(policy, horizon_label)
        if dca is None or mine is None or dca <= 0:
            return None
        return mine / dca - 1.0


@dataclass(frozen=True)
class FailedEpisode:
    """A threshold study that raised an exception during run_event_study."""

    episode: ATHEpisode
    threshold: float
    reason: str


@dataclass(frozen=True)
class EventStudyResult:
    """Return value from run_event_study.

    Separates successful studies from episodes that raised exceptions, so that
    partial failures are visible to callers rather than being silently dropped.
    """

    studies: list[EpisodeStudy]
    failures: list[FailedEpisode]


def _window_end(
    anchor: pd.Timestamp,
    months: int,
    trading_index: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Last trading day at or before anchor + `months`."""
    target = anchor + pd.DateOffset(months=months)
    eligible = trading_index[trading_index <= target]
    if len(eligible) == 0:
        return None
    last = pd.Timestamp(eligible[-1])
    return last if last > anchor else None


def _outcome(policy: str, label: str, result: object) -> PolicyOutcome:
    return PolicyOutcome(
        policy=policy,
        horizon_label=label,
        ending_wealth=float(result.ending_wealth),      # type: ignore[attr-defined]
        total_contributions=float(result.total_contributions),  # type: ignore[attr-defined]
        n_deployments=int(result.n_deployments),        # type: ignore[attr-defined]
        ending_cash=float(result.ending_cash),          # type: ignore[attr-defined]
    )


def episode_windows(
    episode: ATHEpisode,
    trading_index: pd.DatetimeIndex,
    horizon_months: tuple[int, ...],
    anchor_date: pd.Timestamp | None = None,
) -> list[tuple[str, pd.Timestamp]]:
    """Measurement windows for an episode: fixed horizons plus ATH recovery.

    Args:
        episode: The episode to build windows for.
        trading_index: Sorted trading-day index of the instrument.
        horizon_months: Fixed calendar-month horizons to compute.
        anchor_date: Start of the measurement window.  Defaults to the episode
            ATH date when ``None``.
    """
    anchor = anchor_date if anchor_date is not None else episode.ath_date
    out: list[tuple[str, pd.Timestamp]] = []
    for m in horizon_months:
        end = _window_end(anchor, m, trading_index)
        if end is not None:
            out.append((f"{m}m", end))
    if episode.recovery_date is not None:
        out.append(("recovery", episode.recovery_date))
    return out


def study_episode(
    episode: ATHEpisode,
    threshold: float,
    instrument: pd.DataFrame,
    benchmark: pd.DataFrame,
    monthly_contribution: float,
    opening_reserve: float,
    tiers: list[DeploymentTier],
    horizon_months: tuple[int, ...] = DEFAULT_HORIZON_MONTHS,
    cash_rate: float | None = None,
    fixed_fee: float = 0.0,
    pct_fee: float = 0.0,
    slippage: float = 0.0,
    baseline_cache: dict[tuple[pd.Timestamp, pd.Timestamp], list[PolicyOutcome]]
    | None = None,
    anchor: str = "ath",
) -> EpisodeStudy | None:
    """Compare policies within one episode at one threshold.

    The evaluation window opens at the episode's anchor ATH (``anchor="ath"``)
    or at the signal / execution date (``anchor="signal"`` or
    ``anchor="execution"``).

    Args:
        anchor: Where to anchor the measurement windows.
            ``"ath"``       — start at the episode's all-time-high date (default).
            ``"signal"``    — start at the first crossing of ``threshold``.
            ``"execution"`` — start at the first trading day *after* the signal.

    ``baseline_cache`` memoises the threshold-independent policies (savings-only
    and DCA) per window, since they would otherwise be recomputed for every
    threshold.

    Returns ``None`` when the episode never crossed the threshold.
    """
    signal_date = episode.first_crossings.get(threshold)
    if signal_date is None:
        return None

    ath_date = episode.ath_date
    trading_idx = pd.DatetimeIndex(instrument.index)

    # Resolve the anchor date (for horizon-window measurement) and sim_start
    # (the date from which the simulation data is sliced, which must include
    # the threshold-crossing so that the pending order can carry to T+1).
    if anchor == "ath":
        anchor_date = ath_date
        sim_start = ath_date
    elif anchor == "signal":
        anchor_date = signal_date
        sim_start = signal_date
    elif anchor == "execution":
        # Horizon windows are measured from the first trading day after the signal.
        pos = trading_idx.searchsorted(signal_date, side="right")
        if pos >= len(trading_idx):
            return None
        anchor_date = pd.Timestamp(trading_idx[int(pos)])
        # Simulation must start at signal_date so the pending order placed at
        # the crossing carries through to the execution day (T+1).  Starting at
        # anchor_date (T+1) loses the crossing event when the benchmark has
        # already rebounded above the threshold by then.
        sim_start = signal_date
    else:
        raise ValueError(f"anchor must be 'ath', 'signal', or 'execution'; got {anchor!r}")

    windows = episode_windows(
        episode, trading_idx, horizon_months, anchor_date=anchor_date
    )
    if not windows:
        return None

    all_in = [DeploymentTier(threshold, 1.00)]
    outcomes: list[PolicyOutcome] = []

    for label, window_end in windows:
        inst_window = instrument.loc[sim_start:window_end]
        bm_window = benchmark.loc[sim_start:window_end]
        if len(inst_window) < 2 or len(bm_window) < 2:
            continue

        params = SimulationParams(
            monthly_contribution=monthly_contribution,
            payday=25,
            contribution_timing="month_end",
            initial_investment=0.0,
            initial_cash_reserve=opening_reserve,
            start_date=sim_start.date(),
            end_date=window_end.date(),
            dip_threshold=threshold,
            fixed_fee=fixed_fee,
            pct_fee=pct_fee,
            slippage=slippage,
            cash_rate_override=cash_rate,
        )

        key = (sim_start, window_end)
        cached = baseline_cache.get(key) if baseline_cache is not None else None
        try:
            if cached is None:
                baselines = [
                    _outcome("Savings only", label, run_savings_only(inst_window, params)[0]),
                    _outcome("DCA", label, run_dca(inst_window, params)[0]),
                ]
                if baseline_cache is not None:
                    baseline_cache[key] = baselines
            else:
                # Re-label the cached result for this horizon.
                baselines = [
                    PolicyOutcome(
                        policy=o.policy,
                        horizon_label=label,
                        ending_wealth=o.ending_wealth,
                        total_contributions=o.total_contributions,
                        n_deployments=o.n_deployments,
                        ending_cash=o.ending_cash,
                    )
                    for o in cached
                ]

            outcomes.extend(baselines)
            outcomes.append(
                _outcome("ATH all-in", label, run_ath_deployment(
                    instrument_data=inst_window,
                    params=params,
                    tiers=all_in,
                    benchmark_data=bm_window,
                    initial_ath=episode.ath_level,
                )[0])
            )
            outcomes.append(
                _outcome("ATH tiered", label, run_ath_deployment(
                    instrument_data=inst_window,
                    params=params,
                    tiers=tiers,
                    benchmark_data=bm_window,
                    initial_ath=episode.ath_level,
                )[0])
            )
        except (ValueError, KeyError) as exc:
            raise RuntimeError(
                f"study_episode failed for episode at {episode.ath_date.date()}, "
                f"threshold={threshold}, horizon={label}: {exc}"
            ) from exc

    if not outcomes:
        return None

    return EpisodeStudy(
        episode=episode,
        threshold=threshold,
        signal_date=pd.Timestamp(signal_date),
        outcomes=outcomes,
    )


def run_event_study(
    instrument: pd.DataFrame,
    benchmark: pd.DataFrame,
    tiers: list[DeploymentTier],
    monthly_contribution: float = 1000.0,
    opening_reserve: float = 12_000.0,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    horizon_months: tuple[int, ...] = DEFAULT_HORIZON_MONTHS,
    cash_rate: float | None = None,
    fixed_fee: float = 0.0,
    pct_fee: float = 0.0,
    slippage: float = 0.0,
    anchor: str = "ath",
    initial_ath: float | None = None,
    initial_ath_date: pd.Timestamp | None = None,
) -> EventStudyResult:
    """Run the full event study across every episode and threshold.

    Args:
        anchor: Horizon measurement anchor — ``"ath"``, ``"signal"``, or
            ``"execution"``.  See :func:`study_episode` for details.
        initial_ath: Seed the opening all-time high from pre-window history.
            Passed through to :func:`find_ath_episodes`.
        initial_ath_date: Date for ``initial_ath``.

    Returns:
        ``EventStudyResult`` with ``.studies`` (successful episodes) and
        ``.failures`` (episodes that raised an exception).  Callers must access
        ``.studies`` — the return value is no longer a bare ``list``.
    """
    if "adj_close" not in benchmark.columns:
        raise ValueError("benchmark must have an 'adj_close' column")
    if "adj_close" not in instrument.columns:
        raise ValueError("instrument must have an 'adj_close' column")

    episodes = find_ath_episodes(
        benchmark["adj_close"],
        thresholds=thresholds,
        initial_ath=initial_ath,
        initial_ath_date=initial_ath_date,
    )

    # Savings-only and DCA do not depend on the threshold, so they are computed
    # once per (episode, window) and shared across thresholds.
    baseline_cache: dict[tuple[pd.Timestamp, pd.Timestamp], list[PolicyOutcome]] = {}

    studies: list[EpisodeStudy] = []
    failures: list[FailedEpisode] = []
    for ep in episodes:
        for th in thresholds:
            try:
                study = study_episode(
                    episode=ep,
                    threshold=th,
                    instrument=instrument,
                    benchmark=benchmark,
                    monthly_contribution=monthly_contribution,
                    opening_reserve=opening_reserve,
                    tiers=tiers,
                    horizon_months=horizon_months,
                    cash_rate=cash_rate,
                    fixed_fee=fixed_fee,
                    pct_fee=pct_fee,
                    slippage=slippage,
                    baseline_cache=baseline_cache,
                    anchor=anchor,
                )
                if study is not None:
                    studies.append(study)
            except RuntimeError as exc:
                logger.warning("Skipping episode (threshold=%s): %s", th, exc)
                failures.append(FailedEpisode(episode=ep, threshold=th, reason=str(exc)))
    return EventStudyResult(studies=studies, failures=failures)


def summarise_threshold(
    studies: list[EpisodeStudy],
    threshold: float,
    horizon_label: str,
    policy: str = "ATH all-in",
) -> dict[str, float | int | None]:
    """Aggregate one threshold's episodes into the reported statistics.

    Percentiles are reported rather than a mean: the distribution is skewed and a
    single average across episodes hides exactly the cases of interest.

    ``pct_episodes_deployed`` is essential for reading the rest. A threshold deeper
    than the horizon reaches may never fire, in which case the policy is simply
    savings-only — and savings-only beats DCA in any falling market. Without this
    column, "won by buying the dip well" is indistinguishable from "won by never
    buying at all".
    """
    matching = [s for s in studies if s.threshold == threshold]
    rel = [
        r
        for s in matching
        if (r := s.relative_to_dca(policy, horizon_label)) is not None
    ]

    recovered = [s for s in matching if not s.episode.is_censored]
    recovery_days = [
        d for s in recovered if (d := s.episode.days_to_recovery) is not None
    ]

    tiered_vs_all_in = []
    for s in matching:
        a = s.wealth("ATH all-in", horizon_label)
        t = s.wealth("ATH tiered", horizon_label)
        if a is not None and t is not None and a > 0:
            tiered_vs_all_in.append(t / a - 1.0)

    # Did the policy actually trade within the measurement window?
    deployed_flags: list[bool] = []
    cash_fracs: list[float] = []
    for s in matching:
        for o in s.outcomes:
            if o.policy != policy or o.horizon_label != horizon_label:
                continue
            deployed_flags.append(o.n_deployments > 0)
            if o.ending_wealth > 0:
                cash_fracs.append(o.ending_cash / o.ending_wealth)

    base = {
        "threshold": threshold,
        "episodes": len(matching),
        "censored": len(matching) - len(recovered),
        "pct_episodes_deployed": (
            float(pd.Series(deployed_flags).mean()) if deployed_flags else None
        ),
        "median_undeployed_cash": (
            float(pd.Series(cash_fracs).median()) if cash_fracs else None
        ),
        "median_days_to_recovery": (
            float(pd.Series(recovery_days).median()) if recovery_days else None
        ),
    }

    if not rel:
        return {
            **base,
            "median_vs_dca": None,
            "p10_vs_dca": None,
            "p90_vs_dca": None,
            "pct_ahead_of_dca": None,
            "pct_tiered_beat_all_in": None,
            "worst_vs_dca": None,
        }

    rel_s = pd.Series(rel)
    tiered_s = pd.Series(tiered_vs_all_in) if tiered_vs_all_in else None

    return {
        **base,
        "median_vs_dca": float(rel_s.median()),
        "p10_vs_dca": float(rel_s.quantile(0.10)),
        "p90_vs_dca": float(rel_s.quantile(0.90)),
        "pct_ahead_of_dca": float((rel_s > 0).mean()),
        "pct_tiered_beat_all_in": (
            float((tiered_s > 0).mean()) if tiered_s is not None else None
        ),
        "worst_vs_dca": float(rel_s.min()),
    }


def episodes_to_dataframe(episodes: list[ATHEpisode]) -> pd.DataFrame:
    """Tabulate episodes for display."""
    rows = []
    for ep in episodes:
        rows.append(
            {
                "ATH date": ep.ath_date.date(),
                "ATH level": ep.ath_level,
                "Trough date": ep.trough_date.date(),
                "Trough drawdown": ep.trough_drawdown,
                "Recovery date": (
                    ep.recovery_date.date() if ep.recovery_date is not None else None
                ),
                "Days to recovery": ep.days_to_recovery,
                "Recovered": not ep.is_censored,
                "Thresholds crossed": len(ep.first_crossings),
            }
        )
    return pd.DataFrame(rows)


def summary_to_dataframe(
    studies: list[EpisodeStudy],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    horizon_label: str = "12m",
    policy: str = "ATH all-in",
) -> pd.DataFrame:
    """Tabulate per-threshold summary statistics for display."""
    rows = [
        summarise_threshold(studies, th, horizon_label, policy) for th in thresholds
    ]
    return pd.DataFrame([r for r in rows if r["episodes"]])


# ---------------------------------------------------------------------------
# Helper selectors
# ---------------------------------------------------------------------------


def episodes_for_threshold(
    studies: list[EpisodeStudy],
    threshold: float,
) -> list[EpisodeStudy]:
    """Return only the episodes that were run at the given threshold level."""
    return [s for s in studies if s.threshold == threshold]


def winning_episodes(
    studies: list[EpisodeStudy],
    threshold: float,
    horizon_label: str,
    policy: str = "ATH all-in",
) -> list[EpisodeStudy]:
    """Episodes where *policy* outperformed DCA AND actually deployed capital.

    An episode is included only when:

    - ``relative_to_dca(policy, horizon_label) > 0`` (policy beat DCA), AND
    - ``n_deployments > 0`` for the matching outcome (the strategy executed a
      trade; not winning by sitting in cash).

    Args:
        studies: Full output of ``run_event_study``.
        threshold: Drawdown threshold to filter on.
        horizon_label: Horizon label (e.g. ``"12m"``).
        policy: Policy to evaluate against DCA.

    Returns:
        Subset of studies that are genuine policy wins.
    """
    result: list[EpisodeStudy] = []
    for s in episodes_for_threshold(studies, threshold):
        rel = s.relative_to_dca(policy, horizon_label)
        if rel is None or rel <= 0:
            continue
        n_dep = next(
            (
                o.n_deployments
                for o in s.outcomes
                if o.policy == policy and o.horizon_label == horizon_label
            ),
            0,
        )
        if n_dep > 0:
            result.append(s)
    return result
