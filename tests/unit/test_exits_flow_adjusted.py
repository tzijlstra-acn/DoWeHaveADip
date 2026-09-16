"""Exit rules must key off investment performance, not raw account wealth.

The audited defects, each covered below:

- a deposit satisfied a target-return rule;
- zero starting wealth (the default) made the target zero, firing on day one;
- the holding clock ran from the first ledger row, not the deployment date;
- a trailing stop measured drawdown on raw wealth, so deposits moved the peak;
- exit and never-sell were valued on different dates;
- a rule that never fired was indistinguishable from never-sell.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dipdca.quant.exits import (
    ExitType,
    first_deployment_date,
    ledger_nav,
    simulate_exit,
)


def make_ledger(
    prices: list[float],
    flows: list[float] | None = None,
    units: list[float] | None = None,
    start: str = "2020-01-01",
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a minimal ledger consistent with the real engine's invariants.

    Cash is spent at the price on the day units change, so a later price move
    revalues the holding instead of retroactively altering what was paid.
    """
    n = len(prices)
    idx = index if index is not None else pd.bdate_range(start=start, periods=n)
    flows = flows if flows is not None else [0.0] * n
    units = units if units is not None else [0.0] * n

    cash: list[float] = []
    deployed: list[float] = []
    running = 0.0
    prev_units = 0.0
    for i in range(n):
        running += flows[i]
        traded = units[i] - prev_units
        running -= traded * prices[i]      # buy spends cash, sell returns it
        deployed.append(max(0.0, traded) * prices[i])
        prev_units = units[i]
        cash.append(running)

    market_value = [units[i] * prices[i] for i in range(n)]
    total = [cash[i] + market_value[i] for i in range(n)]

    return pd.DataFrame(
        {
            "price": prices,
            "external_flow": flows,
            "cash": cash,
            "units": units,
            "market_value": market_value,
            "total_wealth": total,
            "deployed": deployed,
            "fees": 0.0,
            "interest": 0.0,
        },
        index=idx,
    )


class TestDepositsDoNotDriveExits:
    def test_deposit_does_not_trigger_target_return_exit(self):
        """Flat price, wealth doubles purely from a deposit -> no +50% exit."""
        prices = [100.0] * 10
        # Fully invested at 100 from day 0, then a large deposit on day 5.
        flows = [1000.0] + [0.0] * 4 + [1000.0] + [0.0] * 4
        units = [10.0] * 10

        ledger = make_ledger(prices, flows, units)
        res = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=0.5)

        assert not res["triggered"], "a deposit must not satisfy a return target"
        assert res["exit_date"] is None

    def test_price_gain_does_trigger_target_return_exit(self):
        """Same shape, but the gain is real: +60% price move fires a +50% rule."""
        prices = [100.0] * 5 + [160.0] * 5
        flows = [1000.0] + [0.0] * 9
        units = [10.0] * 10

        ledger = make_ledger(prices, flows, units)
        res = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=0.5)

        assert res["triggered"]
        assert res["exit_date"] == ledger.index[5]

    def test_zero_starting_wealth_does_not_trigger_immediate_exit(self):
        """The default initial_investment is 0, which previously fired on day one."""
        prices = [100.0] * 20
        # Nothing invested until day 10.
        flows = [0.0] * 10 + [1000.0] + [0.0] * 9
        units = [0.0] * 10 + [10.0] * 10

        ledger = make_ledger(prices, flows, units)
        res = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=0.2)

        assert not res["triggered"]
        assert res["exit_date"] is None

    def test_deposit_does_not_move_the_trailing_stop_peak(self):
        """Flat price with a big deposit must not register any NAV drawdown."""
        prices = [100.0] * 10
        flows = [1000.0] + [0.0] * 4 + [5000.0] + [0.0] * 4
        units = [10.0] * 10

        ledger = make_ledger(prices, flows, units)
        res = simulate_exit(ledger, ExitType.TRAILING_STOP, trailing_stop=-0.10)

        assert not res["triggered"]

    def test_real_price_drop_triggers_the_trailing_stop(self):
        prices = [100.0] * 3 + [120.0] * 3 + [100.0] * 4   # -16.7% from peak
        flows = [1000.0] + [0.0] * 9
        units = [10.0] * 10

        ledger = make_ledger(prices, flows, units)
        res = simulate_exit(ledger, ExitType.TRAILING_STOP, trailing_stop=-0.10)

        assert res["triggered"]
        assert res["exit_date"] == ledger.index[6]


class TestHoldingClock:
    def test_clock_starts_at_first_deployment_not_backtest_start(self):
        """36-month hold from a deployment 12 months in must not fire at month 36."""
        # 40 months of month-ends; first purchase at index 12.
        n = 40
        idx = pd.date_range("2020-01-31", periods=n, freq="ME")
        ledger = make_ledger(
            prices=[100.0] * n,
            flows=[0.0] * 12 + [1000.0] + [0.0] * (n - 13),
            units=[0.0] * 12 + [10.0] * (n - 12),
            index=idx,
        )

        entry = first_deployment_date(ledger)
        assert entry == idx[12]

        res = simulate_exit(ledger, ExitType.TIME_BASED, hold_years=2)

        assert res["triggered"]
        # Two years after the DEPLOYMENT (idx[12]), not after idx[0].
        assert res["exit_date"] >= idx[12] + pd.DateOffset(years=2)
        assert res["first_deployment"] == idx[12]

    def test_time_based_rule_does_not_fire_when_hold_exceeds_data(self):
        prices = [100.0] * 30
        flows = [1000.0] + [0.0] * 29
        units = [10.0] * 30

        ledger = make_ledger(prices, flows, units)
        res = simulate_exit(ledger, ExitType.TIME_BASED, hold_years=10)

        assert not res["triggered"]
        assert res["exit_date"] is None

    def test_hold_days_measured_from_deployment(self):
        prices = [100.0] * 20
        flows = [0.0] * 5 + [1000.0] + [0.0] * 14
        units = [0.0] * 5 + [10.0] * 15

        ledger = make_ledger(prices, flows, units)
        res = simulate_exit(ledger, ExitType.NONE)

        expected = (ledger.index[-1] - ledger.index[5]).days
        assert res["hold_days"] == expected


class TestCommonTerminalDate:
    def test_exit_and_never_sell_are_valued_on_same_terminal_date(self):
        prices = [100.0] * 3 + [200.0] + [50.0] * 6
        flows = [1000.0] + [0.0] * 9
        units = [10.0] * 10
        ledger = make_ledger(prices, flows, units)

        exited = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=0.5)
        never = simulate_exit(ledger, ExitType.NONE)

        assert exited["triggered"]
        assert exited["terminal_date"] == never["terminal_date"] == ledger.index[-1]

    def test_proceeds_persist_after_exit_rather_than_vanishing(self):
        """Exiting before a crash must be rewarded at the common terminal date."""
        prices = [100.0] * 3 + [200.0] + [50.0] * 6
        flows = [1000.0] + [0.0] * 9
        units = [10.0] * 10
        ledger = make_ledger(prices, flows, units)

        exited = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=0.5)
        never = simulate_exit(ledger, ExitType.NONE)

        # Sold at 200, held as cash; never-sell rode it down to 50.
        assert exited["terminal_wealth"] > never["terminal_wealth"]

    def test_proceeds_accrue_the_savings_rate_after_exit(self):
        prices = [100.0] * 3 + [200.0] + [200.0] * 6
        flows = [1000.0] + [0.0] * 9
        units = [10.0] * 10
        ledger = make_ledger(prices, flows, units)

        no_rate = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=0.5, cash_rate=0.0)
        with_rate = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=0.5, cash_rate=0.05)

        assert with_rate["terminal_wealth"] > no_rate["terminal_wealth"]

    def test_explicit_terminal_date_is_honoured(self):
        prices = [100.0] * 10
        flows = [1000.0] + [0.0] * 9
        units = [10.0] * 10
        ledger = make_ledger(prices, flows, units)

        res = simulate_exit(ledger, ExitType.NONE, terminal_date=ledger.index[5])

        assert res["terminal_date"] == ledger.index[5]


class TestNotTriggeredIsDistinguishable:
    def test_untriggered_rule_is_flagged_not_silently_zero(self):
        """'Rule not triggered' must be reported, not shown as a zero delta."""
        prices = [100.0] * 10
        flows = [1000.0] + [0.0] * 9
        units = [10.0] * 10
        ledger = make_ledger(prices, flows, units)

        res = simulate_exit(ledger, ExitType.TARGET_RETURN, target_return=5.0)

        assert res["triggered"] is False
        assert res["exit_date"] is None

    def test_never_bought_reports_no_deployment(self):
        prices = [100.0] * 10
        ledger = make_ledger(prices, [1000.0] + [0.0] * 9, [0.0] * 10)

        res = simulate_exit(ledger, ExitType.TRAILING_STOP, trailing_stop=-0.05)

        assert res["first_deployment"] is None
        assert not res["triggered"]
        assert res["hold_days"] == 0

    def test_empty_ledger_is_handled(self):
        res = simulate_exit(pd.DataFrame(), ExitType.TARGET_RETURN)

        assert not res["triggered"]
        assert res["terminal_wealth"] == 0.0


class TestNav:
    def test_pure_deposit_leaves_nav_flat(self):
        prices = [100.0] * 6
        flows = [1000.0, 0.0, 500.0, 0.0, 500.0, 0.0]
        units = [10.0] * 6

        nav = ledger_nav(make_ledger(prices, flows, units))

        assert nav.max() == pytest.approx(nav.min(), rel=1e-9)

    def test_nav_tracks_price_when_no_flows(self):
        prices = [100.0, 110.0, 121.0]
        ledger = make_ledger(prices, [1000.0, 0.0, 0.0], [10.0] * 3)

        nav = ledger_nav(ledger)

        assert float(nav.iloc[-1]) == pytest.approx(1.21, rel=1e-6)

    def test_ledger_without_total_wealth_raises(self):
        bad = pd.DataFrame({"price": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1))

        with pytest.raises(ValueError, match="total_wealth"):
            ledger_nav(bad)
