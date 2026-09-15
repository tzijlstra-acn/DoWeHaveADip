"""Centralized design tokens — light-first color, border, and spacing system.

All UI files should import from here rather than hardcoding hex values.
Charts import CHART_COLORS and STRATEGY_COLORS; CSS uses the string constants.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Brand colors
# ---------------------------------------------------------------------------

PRIMARY   = "#1A56DB"  # blue — CTAs, active tabs, primary actions
SECONDARY = "#374151"  # dark gray — secondary text, subdued elements

# ---------------------------------------------------------------------------
# Semantic colors
# ---------------------------------------------------------------------------

POSITIVE = "#057A55"  # green — gains, beats DCA, good outcomes
NEGATIVE = "#E02424"  # red — losses, drawdowns, below DCA
NEUTRAL  = "#6B7280"  # gray — idle, DCA baseline, no signal
WARNING  = "#C27803"  # amber — approaching dip threshold

# ---------------------------------------------------------------------------
# Surface colors (light-first)
# ---------------------------------------------------------------------------

BG_PAGE   = "#F9FAFB"  # page background
BG_CARD   = "#FFFFFF"  # card / metric container background
BG_SUBTLE = "#F3F4F6"  # sidebar, table alternates, subtle separators

# ---------------------------------------------------------------------------
# Borders
# ---------------------------------------------------------------------------

BORDER       = "#E5E7EB"  # default card/container border
BORDER_STRONG = "#D1D5DB"  # emphasized dividers

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

TEXT_PRIMARY   = "#111827"  # headings, primary labels
TEXT_SECONDARY = "#6B7280"  # captions, metadata
TEXT_DISABLED  = "#9CA3AF"  # placeholder, disabled state

# ---------------------------------------------------------------------------
# Chart palette (ordered — cycle through for multi-series charts)
# ---------------------------------------------------------------------------

CHART_COLORS = ["#1A56DB", "#057A55", "#C27803", "#6B7280", "#E02424"]

# ---------------------------------------------------------------------------
# Strategy colors (plain names — no mascot aliases)
# ---------------------------------------------------------------------------

STRATEGY_COLORS: dict[str, str] = {
    "DCA":              NEUTRAL,   # gray — monthly DCA baseline
    "Invest monthly":   NEUTRAL,
    "Invest now":       PRIMARY,   # blue — lump-sum / deploy cash now
    "Wait for dip":     WARNING,   # amber — wait-for-threshold strategy
    "Tiered":           POSITIVE,  # green — tiered deployment
    "Tiered deployment": POSITIVE,
}

# ---------------------------------------------------------------------------
# Border-radius scale (referenced in inline style strings)
# ---------------------------------------------------------------------------

RADIUS_SM = "6px"
RADIUS_MD = "10px"
RADIUS_LG = "16px"
