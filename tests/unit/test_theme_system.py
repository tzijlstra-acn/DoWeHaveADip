"""Design system correctness tests — dark theme, contrast, brand rules."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

# Ensure project root is importable so ui.* modules can be found
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

ROOT = _ROOT
TOKENS_PATH = ROOT / "ui" / "design_tokens.py"
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"
APP_PAGES_DIR = ROOT / "app_pages"
UI_DIR = ROOT / "ui"

# ---------------------------------------------------------------------------
# Contrast helpers
# ---------------------------------------------------------------------------


def _relative_luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast(fg: str, bg: str) -> float:
    l1, l2 = _relative_luminance(fg), _relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Import tokens at module level (path is set above)
# ---------------------------------------------------------------------------

from ui.design_tokens import (  # noqa: E402
    ACCENT_CYAN,
    BG_CANVAS,
    BG_SURFACE,
    NEGATIVE,
    POSITIVE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
)

# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestConfigMatchesTokens:
    def test_primary_color(self) -> None:
        cfg = _load_config()
        assert cfg["theme"]["primaryColor"].upper() == ACCENT_CYAN.upper()

    def test_bg_color(self) -> None:
        cfg = _load_config()
        assert cfg["theme"]["backgroundColor"].upper() == BG_CANVAS.upper()

    def test_text_color(self) -> None:
        cfg = _load_config()
        assert cfg["theme"]["textColor"].upper() == TEXT_PRIMARY.upper()


class TestContrast:
    def test_text_primary_on_canvas(self) -> None:
        assert _contrast(TEXT_PRIMARY, BG_CANVAS) >= 4.5

    def test_text_primary_on_surface(self) -> None:
        assert _contrast(TEXT_PRIMARY, BG_SURFACE) >= 4.5

    def test_text_secondary_on_surface(self) -> None:
        assert _contrast(TEXT_SECONDARY, BG_SURFACE) >= 4.5

    def test_positive_on_surface(self) -> None:
        assert _contrast(POSITIVE, BG_SURFACE) >= 3.0

    def test_negative_on_surface(self) -> None:
        assert _contrast(NEGATIVE, BG_SURFACE) >= 3.0

    def test_warning_on_surface(self) -> None:
        assert _contrast(WARNING, BG_SURFACE) >= 3.0

    def test_accent_cyan_on_surface(self) -> None:
        assert _contrast(ACCENT_CYAN, BG_SURFACE) >= 3.0


class TestNoBannedTerms:
    def _scan(self, dirs: list, term: str) -> str | None:
        for d in dirs:
            for p in Path(d).rglob("*.py"):
                if term in p.read_text(encoding="utf-8", errors="ignore"):
                    return str(p)
        return None

    def test_no_class_of_one(self) -> None:
        found = self._scan([APP_PAGES_DIR, UI_DIR], "Class of One")
        assert found is None, f"Found 'Class of One' in {found}"

    def test_no_futurebrand_reference(self) -> None:
        found = self._scan([APP_PAGES_DIR, UI_DIR], "futurebrand")
        assert found is None

    def test_no_world_athletics_reference(self) -> None:
        found = self._scan([APP_PAGES_DIR, UI_DIR], "World Athletics")
        assert found is None


class TestCSSContents:
    def _get_global_css(self) -> str:
        from ui.theme import GLOBAL_CSS  # noqa: PLC0415

        return GLOBAL_CSS

    def test_reduced_motion_present(self) -> None:
        assert "prefers-reduced-motion" in self._get_global_css()

    def test_focus_visible_present(self) -> None:
        css = self._get_global_css()
        assert ":focus" in css or "focus-visible" in css

    def test_no_white_background_in_css(self) -> None:
        css = self._get_global_css()
        assert "#FFFFFF" not in css
        assert "#F9FAFB" not in css

    def test_no_dark_text_hardcode_in_css(self) -> None:
        css = self._get_global_css()
        assert "#111827" not in css


class TestComponentsDarkTheme:
    def test_page_header_no_light_colors(self) -> None:
        from unittest.mock import patch

        import streamlit as st

        calls: list = []
        with patch.object(st, "markdown", side_effect=lambda x, **kw: calls.append(x)):
            from ui.components import page_header

            page_header("Test Title", "Test subtitle")
        assert calls, "page_header must call st.markdown"
        output = " ".join(str(c) for c in calls)
        assert "#FFFFFF" not in output
        assert "#F9FAFB" not in output
        assert "#111827" not in output


class TestSVGMotif:
    def test_motif_has_aria_hidden(self) -> None:
        from ui.components import brand_motif

        svg = brand_motif()
        assert 'aria-hidden="true"' in svg

    def test_motif_no_text_element(self) -> None:
        from ui.components import brand_motif

        svg = brand_motif()
        assert "<text" not in svg.lower()


class TestImports:
    def test_ui_modules_importable(self) -> None:
        """All ui modules must be importable cleanly."""
        import importlib

        for mod in ("ui.design_tokens", "ui.theme", "ui.components", "ui.charts"):
            importlib.import_module(mod)
