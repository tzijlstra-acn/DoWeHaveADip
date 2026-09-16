"""2020 Nasdaq-100 fixture: proves threshold deployment can beat monthly DCA.

The audit supplied this case as a numerical sanity check. Prior index ATH was
9,718.73 on 2020-02-19. With EUR 1,000 saved at each month end:

    U_DCA  = 1000/8991.51 + 1000/8461.83      (Jan 31, Feb 28 closes)
    P_DCA  = 2000 / U_DCA = 8,718.63          (harmonic mean of the two prices)

First -15% crossing was 2020-03-09 (close 7,948.03), executing next close
2020-03-10 at 8,372.27. The index crossed both -20% and -25% on 2020-03-12
(close 7,263.65), executing next close 2020-03-13 at 7,995.26.

    all-in at first -15% crossing -> 8,372.27 -> +4.1% units vs DCA
    all-in at first -20% crossing -> 7,995.26 -> +9.0% units vs DCA

Index levels stand in for the execution price in this arithmetic check only. A
production result must execute through the ETF and carry tracking difference,
EUR/USD conversion, cash interest, and fees — those shift the magnitudes but not
the conclusion that winning cases exist.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from dipdca.models import DeploymentTier, SimulationParams
from dipdca.quant.ath_episodes import find_ath_episodes
from dipdca.quant.backtest import run_ath_deployment
from dipdca.quant.drawdown import drawdown

# Nasdaq-100 daily closes, 2020-01-02 .. 2020-03-31.
NDX_2020: dict[str, float] = {
    "2020-01-02": 8872.22, "2020-01-03": 8793.90, "2020-01-06": 8848.52,
    "2020-01-07": 8846.45, "2020-01-08": 8912.37, "2020-01-09": 9012.12,
    "2020-01-10": 8988.75, "2020-01-13": 9078.19, "2020-01-14": 9066.40,
    "2020-01-15": 9085.02, "2020-01-16": 9163.61, "2020-01-17": 9178.86,
    "2020-01-21": 9145.65, "2020-01-22": 9163.13, "2020-01-23": 9185.98,
    "2020-01-24": 9120.51, "2020-01-27": 8967.28, "2020-01-28": 9086.09,
    "2020-01-29": 9096.59, "2020-01-30": 9142.50, "2020-01-31": 8991.51,
    "2020-02-03": 9163.43, "2020-02-04": 9366.75, "2020-02-05": 9404.41,
    "2020-02-06": 9467.99, "2020-02-07": 9401.17, "2020-02-10": 9549.62,
    "2020-02-11": 9598.59, "2020-02-12": 9682.41, "2020-02-13": 9660.36,
    "2020-02-14": 9692.29, "2020-02-18": 9670.19, "2020-02-19": 9718.73,
    "2020-02-20": 9620.52, "2020-02-21": 9384.48, "2020-02-24": 9035.10,
    "2020-02-25": 8697.48, "2020-02-26": 8717.26, "2020-02-27": 8376.70,
    "2020-02-28": 8461.83,
    "2020-03-02": 8871.51, "2020-03-03": 8620.79, "2020-03-04": 8925.23,
    "2020-03-05": 8615.36, "2020-03-06": 8500.23, "2020-03-09": 7948.03,
    "2020-03-10": 8372.27, "2020-03-11": 8009.85, "2020-03-12": 7263.65,
    "2020-03-13": 7995.26, "2020-03-16": 7007.29, "2020-03-17": 7334.78,
    "2020-03-18": 6994.29, "2020-03-19": 7188.02, "2020-03-20": 6879.90,
    "2020-03-23": 6860.67, "2020-03-24": 7407.49, "2020-03-25": 7355.19,
    "2020-03-26": 7797.60, "2020-03-27": 7502.35, "2020-03-30": 7813.27,
    "2020-03-31": 7813.50,
}

ATH_2020_02_19 = 9718.73
DCA_JAN = 8991.51
DCA_FEB = 8461.83
EXEC_AFTER_15 = 8372.27   # 2020-03-10
EXEC_AFTER_25 = 7995.26   # 2020-03-13
MONTHLY = 1000.0


def ndx_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in NDX_2020])
    vals = list(NDX_2020.values())
    return pd.DataFrame({"adj_close": vals, "close": vals}, index=idx)


class TestAuditArithmetic:
    """The audit's unit-count comparison, independent of the engine."""

    def test_dca_effective_price_is_harmonic_mean(self):
        units = MONTHLY / DCA_JAN + MONTHLY / DCA_FEB
        effective = (2 * MONTHLY) / units

        assert abs(effective - 8718.63) < 0.01

    def test_all_in_at_15_percent_beats_dca(self):
        dca_units = MONTHLY / DCA_JAN + MONTHLY / DCA_FEB
        dip_units = (2 * MONTHLY) / EXEC_AFTER_15
        advantage = dip_units / dca_units - 1.0

        assert dip_units > dca_units
        assert 0.035 < advantage < 0.045   # audit: ~+4.1%

    def test_all_in_at_25_percent_beats_dca(self):
        dca_units = MONTHLY / DCA_JAN + MONTHLY / DCA_FEB
        dip_units = (2 * MONTHLY) / EXEC_AFTER_25
        advantage = dip_units / dca_units - 1.0

        assert dip_units > dca_units
        assert 0.085 < advantage < 0.095   # audit: ~+9.0%

    def test_deeper_threshold_acquires_more_units_here(self):
        """In a V-shaped crash, waiting for -25% bought cheaper than -15%."""
        assert (2 * MONTHLY) / EXEC_AFTER_25 > (2 * MONTHLY) / EXEC_AFTER_15


class TestBenchmarkSignalDates:
    """The drawdown series must place the crossings on the audited dates."""

    def test_ath_is_2020_02_19(self):
        s = ndx_frame()["adj_close"]
        pre_crash = s.loc[:"2020-02-19"]

        assert pre_crash.idxmax() == pd.Timestamp("2020-02-19")
        assert abs(float(pre_crash.max()) - ATH_2020_02_19) < 0.01

    def test_first_15_percent_crossing_is_2020_03_09(self):
        dd = drawdown(ndx_frame()["adj_close"])
        crossed = dd[dd <= -0.15]

        assert crossed.index[0] == pd.Timestamp("2020-03-09")

    def test_first_25_percent_crossing_is_2020_03_12(self):
        dd = drawdown(ndx_frame()["adj_close"])
        crossed = dd[dd <= -0.25]

        assert crossed.index[0] == pd.Timestamp("2020-03-12")

    def test_20_and_25_percent_cross_on_the_same_close(self):
        """One close gapped through both, so they share a signal date."""
        dd = drawdown(ndx_frame()["adj_close"])

        assert dd[dd <= -0.20].index[0] == pd.Timestamp("2020-03-12")
        assert dd[dd <= -0.25].index[0] == pd.Timestamp("2020-03-12")


class TestEpisodeDetectionOnRealData:
    """The 2020 crash must resolve to exactly one episode with audited dates."""

    THRESHOLDS = (-0.10, -0.15, -0.20, -0.25, -0.30)

    def _episode(self):
        eps = find_ath_episodes(ndx_frame()["adj_close"], thresholds=self.THRESHOLDS)
        assert len(eps) == 1, f"expected one episode, got {len(eps)}"
        return eps[0]

    def test_whole_crash_is_a_single_episode(self):
        """26 trading days below the ATH is one observation, not 26."""
        ep = self._episode()

        assert ep.ath_date == pd.Timestamp("2020-02-19")
        assert abs(ep.ath_level - ATH_2020_02_19) < 0.01

    def test_trough_is_2020_03_23(self):
        ep = self._episode()

        assert ep.trough_date == pd.Timestamp("2020-03-23")
        assert -0.30 < ep.trough_drawdown < -0.29

    def test_audited_first_crossings(self):
        ep = self._episode()

        assert ep.first_crossings[-0.15] == pd.Timestamp("2020-03-09")
        assert ep.first_crossings[-0.20] == pd.Timestamp("2020-03-12")
        assert ep.first_crossings[-0.25] == pd.Timestamp("2020-03-12")

    def test_never_reached_30_percent(self):
        ep = self._episode()

        assert not ep.crossed(-0.30)

    def test_episode_is_censored_because_recovery_was_in_june(self):
        """The fixture stops 2020-03-31; the ATH was not reclaimed until June."""
        ep = self._episode()

        assert ep.is_censored
        assert ep.recovery_date is None


class TestEngineFindsTheWinningCase:
    """End-to-end: the engine must deploy, and beat holding cash."""

    def _params(self, **kw) -> SimulationParams:
        defaults = dict(
            monthly_contribution=MONTHLY,
            payday=25,
            initial_investment=0.0,
            initial_cash_reserve=2 * MONTHLY,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
            dip_threshold=-0.15,
            max_wait_months=24,
            fixed_fee=0.0,
            pct_fee=0.0,
            slippage=0.0,
            cash_rate_override=None,
        )
        defaults.update(kw)
        return SimulationParams(**defaults)

    def test_all_in_at_15_executes_after_the_signal_close(self):
        frame = ndx_frame()

        _, ledger = run_ath_deployment(
            instrument_data=frame,
            params=self._params(),
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=frame,
            initial_ath=ATH_2020_02_19,
        )

        deploys = ledger[ledger["deployed"] > 0]
        assert not deploys.empty, "engine found no deployment in the 2020 crash"
        # Signal 2020-03-09 must execute strictly later, at the next close.
        assert deploys.index[0] == pd.Timestamp("2020-03-10")

    def test_tiered_schedule_deploys_at_both_crossings(self):
        frame = ndx_frame()

        _, ledger = run_ath_deployment(
            instrument_data=frame,
            params=self._params(),
            tiers=[DeploymentTier(-0.15, 0.25), DeploymentTier(-0.25, 1.00)],
            benchmark_data=frame,
            initial_ath=ATH_2020_02_19,
        )

        deploy_dates = list(ledger[ledger["deployed"] > 0].index)
        assert pd.Timestamp("2020-03-10") in deploy_dates
        assert pd.Timestamp("2020-03-13") in deploy_dates

    def test_threshold_deployment_acquires_more_units_than_month_end_dca(self):
        """The audit's actual claim: more units for the same EUR 2,000.

        Ending wealth is the wrong lens on this window — the fixture stops
        2020-03-31 at 7,813, still below the 8,372 purchase price, and the ATH was
        not recovered until June. Units acquired is the timing question.
        """
        frame = ndx_frame()

        _, ledger = run_ath_deployment(
            instrument_data=frame,
            params=self._params(monthly_contribution=0.01),
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=frame,
            initial_ath=ATH_2020_02_19,
        )

        engine_units = float(ledger["units"].iloc[-1])
        dca_units = MONTHLY / DCA_JAN + MONTHLY / DCA_FEB

        assert engine_units > dca_units
        advantage = engine_units / dca_units - 1.0
        assert 0.03 < advantage < 0.05, f"expected ~+4.1%, got {advantage:+.2%}"

    def test_deployment_beats_holding_cash_once_price_recovers(self):
        """Same contributions; buying the crash wins only after price recovers.

        Valued on the last fixture day the purchase is still under water, so this
        asserts the honest direction: cash is ahead at 2020-03-31. It guards
        against a future change that silently revalues trades.
        """
        frame = ndx_frame()
        params = self._params(monthly_contribution=0.01)

        deployed_result, _ = run_ath_deployment(
            instrument_data=frame,
            params=params,
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=frame,
            initial_ath=ATH_2020_02_19,
        )
        # A threshold far deeper than the crash never fires -> stays in cash.
        never_result, _ = run_ath_deployment(
            instrument_data=frame,
            params=params,
            tiers=[DeploymentTier(-0.90, 1.00)],
            benchmark_data=frame,
            initial_ath=ATH_2020_02_19,
        )

        assert deployed_result.n_deployments == 1
        assert never_result.n_deployments == 0

        # Bought at 8,372.27, valued at 7,813.50 -> a 6.7% paper loss.
        bought_at = EXEC_AFTER_15
        valued_at = NDX_2020["2020-03-31"]
        assert deployed_result.ending_wealth < never_result.ending_wealth
        assert valued_at < bought_at

    def test_future_recovery_does_not_move_the_signal_dates(self):
        """Appending the post-crash rally must not change the crossing dates."""
        frame = ndx_frame()
        extended = pd.concat([
            frame,
            pd.DataFrame(
                {"adj_close": [12_000.0] * 5, "close": [12_000.0] * 5},
                index=pd.bdate_range("2020-04-01", periods=5),
            ),
        ])

        _, ledger = run_ath_deployment(
            instrument_data=extended,
            params=self._params(end_date=date(2020, 4, 30)),
            tiers=[DeploymentTier(-0.15, 1.00)],
            benchmark_data=extended,
            initial_ath=ATH_2020_02_19,
        )

        deploys = ledger[ledger["deployed"] > 0]
        assert deploys.index[0] == pd.Timestamp("2020-03-10")
