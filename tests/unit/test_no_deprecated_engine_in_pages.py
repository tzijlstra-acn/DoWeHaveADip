"""Pages must not drive decisions through the deprecated strategy engine.

``run_wait_for_dip`` contradicts the ATH model in ways that change results:

- it uses the investable ETF, not the reference index, for the drawdown signal;
- it computes the peak inside each sliced rolling window, resetting it at the
  window start;
- it rearms after recovering above a *threshold* rather than after recovering the
  previous *all-time high*;
- it can force deployment after a maximum waiting period.

``run_parameter_sweep`` inherits all of the above because it calls that engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PAGES = Path(__file__).resolve().parents[2] / "app_pages"

DEPRECATED = ("run_wait_for_dip", "run_parameter_sweep")

# Pages still awaiting migration. Each entry is a known gap, not an exemption.
NOT_YET_MIGRATED = {"advanced_exit.py"}


def page_files() -> list[Path]:
    return sorted(PAGES.glob("*.py"))


def test_pages_directory_is_discoverable():
    assert page_files(), f"no page files found under {PAGES}"


@pytest.mark.parametrize("page", page_files(), ids=lambda p: p.name)
def test_page_does_not_use_deprecated_engine(page: Path):
    if page.name in NOT_YET_MIGRATED:
        pytest.xfail(f"{page.name} still calls the deprecated engine (tracked)")

    src = page.read_text(encoding="utf-8")
    found = [sym for sym in DEPRECATED if sym in src]

    assert not found, (
        f"{page.name} references {found}. Use run_ath_deployment, or the "
        f"ath_episodes event study, so the reference index drives the signal."
    )


def test_scenarios_page_uses_the_event_study():
    """The historical-scenarios page must be built on ATH episodes."""
    src = (PAGES / "scenarios.py").read_text(encoding="utf-8")

    assert "run_event_study" in src
    assert "find_ath_episodes" in src
    for sym in DEPRECATED:
        assert sym not in src


def test_scenarios_page_loads_the_benchmark_index():
    """Signal must come from index_symbol, not the ETF, when one is configured."""
    src = (PAGES / "scenarios.py").read_text(encoding="utf-8")

    assert "index_symbol" in src


def test_migration_tracker_only_lists_real_pages():
    """Keeps NOT_YET_MIGRATED from silently exempting a deleted or renamed file."""
    names = {p.name for p in page_files()}

    stale = NOT_YET_MIGRATED - names
    assert not stale, f"NOT_YET_MIGRATED lists non-existent pages: {stale}"
