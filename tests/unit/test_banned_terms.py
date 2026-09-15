"""Enforce that banned terms are absent from all UI and page files.

Banned terms include mascot names that were removed in the product redesign,
and "Monte Carlo" which must not appear in user-visible copy (it may still
appear in quant module internals and this test explicitly excludes those).
"""

from __future__ import annotations

import re
from pathlib import Path

# Terms banned from all files under PAGE_DIRS
BANNED: list[str] = [
    "Monthly Machine",
    "Cash Goblin",
    "Dip Buffet",
    "Market Arcade",
]

# "Monte Carlo" is only banned in user-facing page/ui files, not quant internals
BANNED_IN_UI_ONLY: list[str] = [
    "Monte Carlo",
]

# Directories to check for all banned terms
PAGE_DIRS = [
    "app_pages",
    "ui",
]

# Directories to additionally check for UI-only banned terms
UI_DIRS = [
    "app_pages",
    "ui",
]

# File extensions to scan
EXTENSIONS = {".py"}

REPO_ROOT = Path(__file__).parent.parent.parent


def _find_files(dirs: list[str]) -> list[Path]:
    files = []
    for d in dirs:
        base = REPO_ROOT / d
        if base.exists():
            files.extend(f for f in base.rglob("*") if f.suffix in EXTENSIONS)
    return files


def _grep(files: list[Path], term: str) -> list[str]:
    matches = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if term in line:
                matches.append(f"{f.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    return matches


def test_no_mascot_names_in_ui() -> None:
    """Mascot strategy names must not appear in app_pages/ or ui/."""
    files = _find_files(PAGE_DIRS)
    for banned in BANNED:
        matches = _grep(files, banned)
        assert not matches, (
            f"Found banned term '{banned}' in UI/page files:\n"
            + "\n".join(matches)
        )


def test_no_monte_carlo_in_page_copy() -> None:
    """'Monte Carlo' must not appear in user-facing page copy (app_pages/ and ui/)."""
    files = _find_files(UI_DIRS)
    for banned in BANNED_IN_UI_ONLY:
        matches = _grep(files, banned)
        assert not matches, (
            f"Found '{banned}' in UI/page files — use 'historical scenarios' in user copy:\n"
            + "\n".join(matches)
        )


def test_no_emoji_in_page_titles() -> None:
    """page_header() calls and st.set_page_config page_title must not contain emoji."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002600-\U000027BF"  # misc symbols
        "\U0001F900-\U0001F9FF"  # supplemental
        "]+",
        flags=re.UNICODE,
    )
    files = _find_files(PAGE_DIRS)
    violations = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ("page_header(" in line or "page_title=" in line) and emoji_pattern.search(line):
                violations.append(f"{f.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not violations, (
        "Found emoji in page_header/page_title calls:\n" + "\n".join(violations)
    )
