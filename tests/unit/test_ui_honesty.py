"""The UI must not display figures the selected engine does not produce.

Audited display defects:

- controls for max-wait, emergency buffer and deployment spreading were shown on
  a page driven by the ATH engine, which consumes none of them, so changing them
  produced identical results;
- the deposit-contaminated raw-wealth drawdown was displayed while the
  flow-adjusted NAV drawdown was computed and ignored;
- slippage defaulted to ~0.1% and was never surfaced, so setting the visible fee
  to zero did not make trading free.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import ui.components as components
from dipdca.quant.backtest import run_ath_deployment, run_dca, run_savings_only

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "app_pages"

# Parameters belonging to the deprecated wait-for-dip engine.
WAIT_ENGINE_ONLY = ("max_wait_months", "cash_buffer_months", "deploy_spread_months")

# Pages whose strategies are all ATH-engine based.
ATH_PAGES = ("compare.py", "scenarios.py", "advanced_exit.py")


class TestDeadControlsAreNotExposed:
    @pytest.mark.parametrize("param", WAIT_ENGINE_ONLY)
    def test_ath_engine_ignores_wait_engine_params(self, param: str):
        """Pins the premise: these genuinely do nothing in the ATH engine."""
        for fn in (run_ath_deployment, run_dca, run_savings_only):
            assert f"params.{param}" not in inspect.getsource(fn), (
                f"{fn.__name__} now reads params.{param}; the UI claim that this "
                f"control is inert would no longer hold."
            )

    @pytest.mark.parametrize("page", ATH_PAGES)
    def test_page_does_not_offer_inert_controls(self, page: str):
        src = (PAGES / page).read_text(encoding="utf-8")
        # Bare mentions in comments are fine; an assignment from a widget is not.
        for param in WAIT_ENGINE_ONLY:
            assert f"{param}=" not in src.replace(f"# {param}=", ""), (
                f"{page} passes {param}, which the ATH engine ignores."
            )


def body_without_docstring(fn) -> str:
    """Source of a function with its docstring removed.

    The docstrings here legitimately name the rejected field when explaining why
    it is rejected, so prose must not be mistaken for usage.
    """
    src = inspect.getsource(fn)
    doc = inspect.getdoc(fn)
    if not doc:
        return src
    for line in doc.splitlines():
        src = src.replace(line, "")
    return src


class TestDrawdownIsFlowAdjusted:
    def test_metrics_row_shows_nav_drawdown_not_raw_wealth(self):
        src = body_without_docstring(components.strategy_metrics)

        assert "nav_mdd" in src
        assert "result.max_drawdown" not in src

    def test_comparison_cards_show_nav_drawdown(self):
        src = body_without_docstring(components.strategy_comparison_cards)

        assert "nav_mdd" in src
        assert "max_drawdown" not in src


class TestTieHandling:
    def test_comparison_cards_do_not_use_a_ge_winner_test(self):
        """`dca >= dip` reported an exact tie as DCA winning by zero."""
        src = inspect.getsource(components.strategy_comparison_cards)

        assert "ending_wealth >= " not in src
        assert "compare_outcomes" in src


class TestAssumptionsAreSurfaced:
    @pytest.mark.parametrize("page", ATH_PAGES)
    def test_page_exposes_slippage(self, page: str):
        """Slippage was a hidden ~0.1% cost added to every trade."""
        src = (PAGES / page).read_text(encoding="utf-8")

        assert "slippage" in src, f"{page} does not surface slippage"

    @pytest.mark.parametrize("page", ATH_PAGES)
    def test_page_states_its_cost_assumptions(self, page: str):
        src = (PAGES / page).read_text(encoding="utf-8")

        assert "Assumptions:" in src, f"{page} has no assumptions summary"

    @pytest.mark.parametrize("page", ("compare.py", "advanced_exit.py"))
    def test_page_pins_month_end_contributions(self, page: str):
        src = (PAGES / page).read_text(encoding="utf-8")

        assert "month_end" in src, f"{page} does not pin contribution timing"

    def test_event_study_pins_month_end_on_behalf_of_scenarios_page(self):
        """scenarios.py delegates params to the event study, which must pin it."""
        from dipdca.quant import ath_episodes

        src = inspect.getsource(ath_episodes.study_episode)

        assert 'contribution_timing="month_end"' in src
