"""Centralized design tokens — dark-first DIP SIGNAL brand system."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Surfaces (dark-first)
# ---------------------------------------------------------------------------

BG_CANVAS = "#07070A"
BG_SURFACE = "#111218"
BG_SURFACE_RAISED = "#1A1C24"
BG_SURFACE_ACTIVE = "#232631"

BORDER = "#343746"
BORDER_STRONG = "#505465"

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

TEXT_PRIMARY = "#F7F7F2"
TEXT_SECONDARY = "#C3C6D1"
TEXT_MUTED = "#959AAA"
TEXT_DISABLED = "#727786"

# ---------------------------------------------------------------------------
# Accents
# ---------------------------------------------------------------------------

ACCENT_CYAN = "#42E8FF"
ACCENT_MAGENTA = "#FF4DC4"
ACCENT_LIME = "#D8FF4F"
ACCENT_ORANGE = "#FF914D"

# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------

POSITIVE = "#37D39A"
NEGATIVE = "#FF5F73"
WARNING = "#FFB84A"
NEUTRAL = "#A7ACB9"

# ---------------------------------------------------------------------------
# Brand gradient (decoration only — never place body text on this)
# ---------------------------------------------------------------------------

BRAND_GRADIENT = "linear-gradient(115deg,#42E8FF 0%,#6576FF 30%,#FF4DC4 65%,#D8FF4F 100%)"

# ---------------------------------------------------------------------------
# Strategy colors
# ---------------------------------------------------------------------------

STRATEGY_COLORS: dict[str, str] = {
    "DCA":               NEUTRAL,
    "Invest monthly":    NEUTRAL,
    "Wait for dip":      ACCENT_CYAN,
    "Tiered":            ACCENT_MAGENTA,
    "Tiered deployment": ACCENT_MAGENTA,
    "Invest now":        ACCENT_ORANGE,
}
STRATEGY_DASH: dict[str, str] = {
    "DCA":               "solid",
    "Invest monthly":    "solid",
    "Wait for dip":      "dash",
    "Tiered":            "dashdot",
    "Tiered deployment": "dashdot",
    "Invest now":        "dot",
}
CHART_COLORS = [ACCENT_CYAN, NEUTRAL, ACCENT_MAGENTA, ACCENT_ORANGE, ACCENT_LIME, POSITIVE, NEGATIVE]

# ---------------------------------------------------------------------------
# Backward-compat aliases (pages still import these names)
# ---------------------------------------------------------------------------

PRIMARY   = ACCENT_CYAN
SECONDARY = TEXT_SECONDARY
BG_PAGE   = BG_CANVAS
BG_CARD   = BG_SURFACE
BG_SUBTLE = BG_SURFACE_RAISED

# ---------------------------------------------------------------------------
# Spacing & radius
# ---------------------------------------------------------------------------

RADIUS_SM = "4px"
RADIUS_MD = "8px"
RADIUS_LG = "12px"
