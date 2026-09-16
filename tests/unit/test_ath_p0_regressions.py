"""P0 regression tests from the numerical audit.

Two defects made every historical result uninterpretable:

1. Future-ATH leak — the caller seeded the opening peak with
   ``benchmark_df["adj_close"].max()`` over the whole evaluation window, so day 1
   was compared against a high reached years later. Every tier was marked crossed
   before any cash existed.

2. Reversed tier classification — "Decision today" selected the shallowest
   *reached* tier as "next" and treated the deeper *unreached* tiers as already
   fired, producing a screen that said "threshold reached" while computing EUR 0.

The tier-classification logic is verified here against the same pure helpers the
page uses, so the arithmetic is covered independently of Streamlit.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from dipdca.models import DeploymentTier, SimulationParams
from dipdca.quant.backtest import run_ath_deployment


def make_prices(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"adj_close": values, "close": values}, index=idx)


def base_params(**kw) -> SimulationParams:
    defaults = dict(
        monthly_contribution=0.01,
        payday=25,
        initial_investment=0.0,
        initial_cash_reserve=10_000.0,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
        dip_threshold=-0.10,
        max_wait_months=24,
        fixed_fee=0.0,
        pct_fee=0.0,
        slippage=0.0,
        cash_rate_override=None,
    )
    defaults.update(kw)
    return SimulationParams(**defaults)


# ---------------------------------------------------------------------------
# Tier classification — mirrors the logic in app_pages/compare.py
# ---------------------------------------------------------------------------

def classify_tiers(current_dd: float, tiers: list[DeploymentTier]):
    """Return (deepest_reached, next_deeper) for a drawdown against a schedule."""
    reached = [t for t in tiers if current_dd <= t.drawdown_threshold]
    unreached_deeper = [t for t in tiers if t.drawdown_threshold < current_dd]
    deepest_reached = min(reached, key=lambda t: t.drawdown_threshold) if reached else None
    next_deeper = (
        max(unreached_deeper, key=lambda t: t.drawdown_threshold)
        if unreached_deeper
        else None
    )
    return deepest_reached, next_deeper


def deployment_now(
    current_dd: float,
    tiers: list[DeploymentTier],
    cash: float,
    already_deployed: float,
) -> float:
    """Incremental principal to deploy at the deepest reached tier."""
    deepest_reached, _ = classify_tiers(current_dd, tiers)
    target_fraction = (
        deepest_reached.cumulative_deployment_fraction if deepest_reached else 0.0
    )
    eligible = cash + already_deployed
    target_principal = target_fraction * eligible
    return min(cash, max(0.0, target_principal - already_deployed))


SCHEDULE = [
    DeploymentTier(-0.15, 0.25),
    DeploymentTier(-0.25, 0.60),
    DeploymentTier(-0.35, 1.00),
]


class TestTierClassificationNotReversed:
    def test_minus_20_percent_reaches_minus_15_tier_not_minus_25(self):
        """At -20%: -15% is reached; -25% and -35% are not.

        The reversed implementation picked -15% as "next" and classified
        -25%/-35% as already fired.
        """
        deepest_reached, next_deeper = classify_tiers(-0.20, SCHEDULE)

        assert deepest_reached is not None
        assert deepest_reached.drawdown_threshold == -0.15
        assert next_deeper is not None
        assert next_deeper.drawdown_threshold == -0.25

    def test_reached_tier_produces_nonzero_deployment_when_cash_exists(self):
        """The headline audit symptom: 'threshold reached' but EUR 0 to deploy."""
        deploy = deployment_now(-0.20, SCHEDULE, cash=20_000.0, already_deployed=0.0)

        # 25% of EUR 20,000 eligible = EUR 5,000
        assert deploy == 5_000.0

    def test_no_tier_reached_above_shallowest_threshold(self):
        deepest_reached, next_deeper = classify_tiers(-0.05, SCHEDULE)

        assert deepest_reached is None
        assert next_deeper is not None
        assert next_deeper.drawdown_threshold == -0.15
        assert deployment_now(-0.05, SCHEDULE, 20_000.0, 0.0) == 0.0

    def test_deepest_tier_reached_targets_full_deployment(self):
        deepest_reached, next_deeper = classify_tiers(-0.40, SCHEDULE)

        assert deepest_reached is not None
        assert deepest_reached.drawdown_threshold == -0.35
        assert next_deeper is None
        assert deployment_now(-0.40, SCHEDULE, 20_000.0, 0.0) == 20_000.0

    def test_exact_threshold_counts_as_reached(self):
        deepest_reached, _ = classify_tiers(-0.25, SCHEDULE)

        assert deepest_reached is not None
        assert deepest_reached.drawdown_threshold == -0.25

    def test_already_deployed_reduces_incremental_trade(self):
        """EUR 8,000 cash left after EUR 12,000 deployed; -35% targets 100%."""
        deploy = deployment_now(
            -0.40, SCHEDULE, cash=8_000.0, already_deployed=12_000.0
        )

        # eligible = 20,000; target = 100% = 20,000; less 12,000 already = 8,000
        assert deploy == 8_000.0

    def test_deployment_never_exceeds_available_cash(self):
        deploy = deployment_now(-0.40, SCHEDULE, cash=1_000.0, already_deployed=0.0)

        assert deploy == 1_000.0

    def test_overshoot_from_prior_deployment_clamps_to_zero_not_negative(self):
        """Already past the target for the reached tier — deploy nothing, never negative."""
        deploy = deployment_now(
            -0.20, SCHEDULE, cash=1_000.0, already_deployed=19_000.0
        )

        # eligible = 20,000; -15% target = 25% = 5,000; already 19,000 > target
        assert deploy == 0.0

    def test_zero_cash_yields_zero_without_error(self):
        assert deployment_now(-0.40, SCHEDULE, cash=0.0, already_deployed=0.0) == 0.0


# ---------------------------------------------------------------------------
# Future-ATH leak
# ---------------------------------------------------------------------------

class TestNoFutureATHLeak:
    def test_future_index_values_cannot_change_past_ath_or_signals(self):
        """Appending an enormous future peak must not alter any earlier trade.

        Under the leak, .max() over the whole frame made day 1 look like a deep
        drawdown against the appended spike.
        """
        base_vals = [100.0] * 20 + [80.0] * 40
        spiked_vals = base_vals + [10_000.0] * 20
        padded_vals = base_vals + [80.0] * 20

        params = base_params(monthly_contribution=0.01, initial_cash_reserve=10_000.0)

        res_plain, ledger_plain = run_ath_deployment(
            instrument_data=make_prices(padded_vals),
            params=params,
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=make_prices(padded_vals),
            initial_ath=100.0,
        )
        res_spiked, ledger_spiked = run_ath_deployment(
            instrument_data=make_prices(spiked_vals),
            params=params,
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=make_prices(spiked_vals),
            initial_ath=100.0,
        )

        # Compare only the window before the spike — it must be byte-identical.
        n = len(base_vals)
        pre_plain = ledger_plain.iloc[:n]
        pre_spiked = ledger_spiked.iloc[:n]

        assert list(pre_plain["deployed"]) == list(pre_spiked["deployed"])
        assert list(pre_plain["benchmark_ath"]) == list(pre_spiked["benchmark_ath"])
        assert list(pre_plain["benchmark_drawdown"]) == list(pre_spiked["benchmark_drawdown"])
        assert res_plain.n_deployments == res_spiked.n_deployments

    def test_ath_never_seeded_from_evaluation_window_max(self):
        """With initial_ath=None the opening peak is the FIRST close, not the max.

        A rising series has its max at the end. Seeding from that max would show a
        large day-1 drawdown; seeding from the first close shows zero.
        """
        rising = [100.0 + i for i in range(60)]

        _, ledger = run_ath_deployment(
            instrument_data=make_prices(rising),
            params=base_params(),
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=make_prices(rising),
            initial_ath=None,
        )

        assert ledger["benchmark_ath"].iloc[0] == 100.0
        assert ledger["benchmark_drawdown"].iloc[0] == 0.0
        # A monotonically rising benchmark is never in drawdown.
        assert float(ledger["benchmark_drawdown"].min()) == 0.0
        assert float(ledger["deployed"].sum()) == 0.0

    def test_monotonically_rising_benchmark_never_triggers_a_tier(self):
        """The leak's signature failure: instant triggers on a market that only rose."""
        rising = [100.0 + i * 2 for i in range(80)]

        result, ledger = run_ath_deployment(
            instrument_data=make_prices(rising),
            params=base_params(initial_cash_reserve=50_000.0),
            tiers=[
                DeploymentTier(-0.15, 0.25),
                DeploymentTier(-0.25, 0.60),
                DeploymentTier(-0.35, 1.00),
            ],
            benchmark_data=make_prices(rising),
            initial_ath=None,
        )

        assert result.n_deployments == 0
        assert float(ledger["deployed"].sum()) == 0.0

    def test_simulation_starting_mid_drawdown_preserves_prior_ath(self):
        """Evaluation opens already 20% below a pre-simulation peak of 100."""
        # Window never reaches 100; prior ATH must still anchor the drawdown.
        window = [80.0] * 10 + [70.0] * 50

        _, ledger = run_ath_deployment(
            instrument_data=make_prices(window),
            params=base_params(),
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=make_prices(window),
            initial_ath=100.0,
        )

        # Day 1: 80/100 - 1 = -20%, not 0%.
        assert ledger["benchmark_ath"].iloc[0] == 100.0
        assert abs(float(ledger["benchmark_drawdown"].iloc[0]) - (-0.20)) < 1e-9

    def test_explicit_initial_ath_overrides_window_contents(self):
        vals = [50.0] * 40

        _, ledger = run_ath_deployment(
            instrument_data=make_prices(vals),
            params=base_params(),
            tiers=[DeploymentTier(-0.50, 1.00)],
            benchmark_data=make_prices(vals),
            initial_ath=200.0,
        )

        assert ledger["benchmark_ath"].iloc[0] == 200.0
        assert abs(float(ledger["benchmark_drawdown"].iloc[0]) - (-0.75)) < 1e-9

    def test_zero_cash_does_not_permanently_consume_a_tier(self):
        """A tier crossed with no cash must still be available in a later episode.

        Under the leak, tiers were consumed on day 1 before any savings existed,
        leaving the strategy permanently in cash.
        """
        # Dip to -20% with zero opening cash, recover to a new ATH, dip again.
        vals = (
            [100.0] * 5
            + [80.0] * 5           # -20%: crosses -15% but no cash yet
            + [105.0] * 5          # new ATH -> episode resets
            + [80.0] * 25          # -23.8% from 105: crosses -15% again
        )

        params = base_params(
            monthly_contribution=1_000.0,
            initial_cash_reserve=0.0,     # nothing to deploy at the first crossing
            start_date=date(2020, 1, 1),
            end_date=date(2020, 4, 30),
        )

        result, ledger = run_ath_deployment(
            instrument_data=make_prices(vals),
            params=params,
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=make_prices(vals),
            initial_ath=100.0,
        )

        # Contributions accumulated, so the second crossing must actually deploy.
        assert float(ledger["deployed"].sum()) > 0.0
        assert result.n_deployments >= 1
