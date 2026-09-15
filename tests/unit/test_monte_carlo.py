"""Unit tests for dipdca.quant.monte_carlo."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from dipdca.models import SimulationParams
from dipdca.quant.monte_carlo import (
    PathSimulation,
    SweepResult,
    conditional_path_bootstrap,
    outperformance_pivot,
    run_parameter_sweep,
    sweep_to_dataframe,
    win_rate_pivot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n: int = 50, start: str = "2010-01-04", trend: float = 0.0003) -> pd.Series:
    """Create a synthetic daily price series with slight upward drift."""
    dates = pd.bdate_range(start=start, periods=n)
    prices = 100.0 * np.cumprod(1 + trend + np.random.default_rng(42).normal(0, 0.01, n))
    return pd.Series(prices, index=dates, name="test")


def _make_sweep_results() -> list[SweepResult]:
    return [
        SweepResult(
            threshold=-0.10,
            deploy_pct=0.50,
            win_rate=0.55,
            median_outperformance=0.02,
            p10_outperformance=-0.05,
            p90_outperformance=0.09,
            n_windows=30,
            median_deployments=3.0,
        ),
        SweepResult(
            threshold=-0.10,
            deploy_pct=1.00,
            win_rate=0.60,
            median_outperformance=0.04,
            p10_outperformance=-0.03,
            p90_outperformance=0.12,
            n_windows=30,
            median_deployments=3.0,
        ),
        SweepResult(
            threshold=-0.20,
            deploy_pct=0.50,
            win_rate=0.48,
            median_outperformance=-0.01,
            p10_outperformance=-0.10,
            p90_outperformance=0.08,
            n_windows=20,
            median_deployments=1.5,
        ),
        SweepResult(
            threshold=-0.20,
            deploy_pct=1.00,
            win_rate=0.52,
            median_outperformance=0.01,
            p10_outperformance=-0.08,
            p90_outperformance=0.11,
            n_windows=20,
            median_deployments=1.5,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: sweep_to_dataframe
# ---------------------------------------------------------------------------


def test_sweep_to_dataframe_shape():
    results = _make_sweep_results()
    df = sweep_to_dataframe(results)
    assert len(df) == 4
    assert set(df.columns) == {
        "threshold",
        "deploy_pct",
        "win_rate",
        "median_outperformance",
        "p10",
        "p90",
        "n_windows",
        "median_deployments",
    }


def test_sweep_to_dataframe_values():
    results = _make_sweep_results()
    df = sweep_to_dataframe(results)
    # Check first row matches first result
    row = df[df["threshold"] == -0.10].iloc[0]
    assert row["deploy_pct"] in [0.50, 1.00]
    assert 0.0 <= row["win_rate"] <= 1.0


def test_sweep_to_dataframe_empty():
    df = sweep_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: win_rate_pivot
# ---------------------------------------------------------------------------


def test_win_rate_pivot_shape():
    results = _make_sweep_results()
    df = sweep_to_dataframe(results)
    pivot = win_rate_pivot(df)

    # 2 thresholds (-10%, -20%), 2 deploy_pcts (50%, 100%)
    assert pivot.shape == (2, 2)


def test_win_rate_pivot_index_format():
    results = _make_sweep_results()
    df = sweep_to_dataframe(results)
    pivot = win_rate_pivot(df)

    # Index should be formatted as percentage strings
    for idx in pivot.index:
        assert "%" in idx, f"Expected % in index label, got: {idx}"

    # Columns should be formatted as percentage strings
    for col in pivot.columns:
        assert "%" in col, f"Expected % in column label, got: {col}"


def test_win_rate_pivot_values_in_range():
    results = _make_sweep_results()
    df = sweep_to_dataframe(results)
    pivot = win_rate_pivot(df)

    assert pivot.values.min() >= 0.0
    assert pivot.values.max() <= 1.0


def test_outperformance_pivot_shape():
    results = _make_sweep_results()
    df = sweep_to_dataframe(results)
    pivot = outperformance_pivot(df)
    assert pivot.shape == (2, 2)


def test_outperformance_pivot_index_format():
    results = _make_sweep_results()
    df = sweep_to_dataframe(results)
    pivot = outperformance_pivot(df)
    for idx in pivot.index:
        assert "%" in idx


# ---------------------------------------------------------------------------
# Smoke tests: run_parameter_sweep
# ---------------------------------------------------------------------------


def _base_params(start: date = date(2010, 1, 1), end: date = date(2010, 3, 31)) -> SimulationParams:
    return SimulationParams(
        monthly_contribution=500.0,
        payday=15,
        start_date=start,
        end_date=end,
    )


def test_run_parameter_sweep_returns_list():
    """Smoke test: function returns a list without crashing on tiny series."""
    prices = _make_prices(n=50)
    params = _base_params()

    results = run_parameter_sweep(
        prices=prices,
        base_params=params,
        thresholds=[-0.05],
        deploy_pcts=[1.0],
        window_years=1,
        step_months=3,
    )
    assert isinstance(results, list)


def test_run_parameter_sweep_too_short_returns_empty():
    """With only 50 data points and 1-year (252-point) window, should return []."""
    prices = _make_prices(n=50)
    params = _base_params()

    results = run_parameter_sweep(
        prices=prices,
        base_params=params,
        thresholds=[-0.10],
        deploy_pcts=[1.0],
        window_years=1,
        step_months=12,
        points_per_year=252,
    )
    assert isinstance(results, list)
    # 50 < 252, so no windows can be created
    assert len(results) == 0


def test_run_parameter_sweep_monthly_data():
    """Smoke test: monthly data with points_per_year=12 should produce windows."""
    # 5 years of monthly data = 60 points; use 2-year window
    dates = pd.period_range("2000-01", periods=60, freq="M").to_timestamp(how="end")
    prices = pd.Series(
        100.0 * np.cumprod(1 + np.random.default_rng(7).normal(0.005, 0.04, 60)),
        index=dates,
        name="monthly_test",
    )
    params = SimulationParams(
        monthly_contribution=500.0,
        payday=15,
        start_date=date(2000, 1, 1),
        end_date=date(2004, 12, 31),
    )

    results = run_parameter_sweep(
        prices=prices,
        base_params=params,
        thresholds=[-0.10],
        deploy_pcts=[1.0],
        window_years=2,
        step_months=6,
        points_per_year=12,
    )
    assert isinstance(results, list)
    # May have results or not — just verify no exceptions and correct type


def test_run_parameter_sweep_sweep_result_fields():
    """With enough data, sweep results have correct field types."""
    # Create 300 daily points (~1.2 years) and use a 6-month window
    prices = _make_prices(n=300, start="2015-01-01")
    start = prices.index[0].date()
    end = prices.index[-1].date()
    params = SimulationParams(
        monthly_contribution=500.0,
        payday=15,
        start_date=start,
        end_date=end,
    )

    results = run_parameter_sweep(
        prices=prices,
        base_params=params,
        thresholds=[-0.05],
        deploy_pcts=[0.50, 1.00],
        window_years=1,  # 252 data points needed
        step_months=1,
        points_per_year=252,
    )

    for r in results:
        assert isinstance(r, SweepResult)
        assert 0.0 <= r.win_rate <= 1.0
        assert r.n_windows >= 3
        assert r.median_deployments >= 0


# ---------------------------------------------------------------------------
# Smoke tests: conditional_path_bootstrap
# ---------------------------------------------------------------------------


def test_conditional_path_bootstrap_empty_on_no_match():
    """Returns empty list when no historical entries match the drawdown level."""
    # Prices that only go up — no drawdowns
    dates = pd.bdate_range("2010-01-01", periods=300)
    prices = pd.Series(np.linspace(100, 200, 300), index=dates)

    result = conditional_path_bootstrap(
        prices=prices,
        current_drawdown=-0.50,  # -50% — won't exist in monotonically rising series
        deploy_pcts=[0.50, 1.00],
        monthly_contribution=500.0,
        cash_accumulated=6000.0,
        horizon_months=12,
        n_simulations=20,
    )
    assert isinstance(result, list)
    assert len(result) == 0


def test_conditional_path_bootstrap_returns_path_simulations():
    """Smoke test: returns PathSimulation objects with correct shapes."""
    rng = np.random.default_rng(99)
    # Create a series with genuine drawdowns
    returns = rng.normal(0.005, 0.06, 200)
    # Force a -20% drawdown period in the middle
    returns[80:95] = -0.04  # 15 months of -4%/month
    dates = pd.period_range("2005-01", periods=200, freq="M").to_timestamp(how="end")
    prices = pd.Series(
        100.0 * np.cumprod(1 + returns),
        index=dates,
    )

    result = conditional_path_bootstrap(
        prices=prices,
        current_drawdown=-0.15,
        deploy_pcts=[0.50, 1.00],
        monthly_contribution=500.0,
        cash_accumulated=6000.0,
        horizon_months=12,
        n_simulations=50,
    )

    assert isinstance(result, list)
    for sim in result:
        assert isinstance(sim, PathSimulation)
        assert 0.0 <= sim.deploy_pct <= 1.0
        assert 0.0 <= sim.prob_beats_dca <= 1.0
        assert len(sim.p50_wealth) > 0
        assert len(sim.p25_wealth) == len(sim.p50_wealth)
        assert len(sim.p75_wealth) == len(sim.p50_wealth)
