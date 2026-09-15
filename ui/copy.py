"""UI copy strings and labels — plain language, no mascot names."""

from __future__ import annotations

APP_TITLE   = "Do we have a dip?"
APP_TAGLINE = "Compare monthly investing, waiting for a dip, and investing now — using real market data."
APP_ICON    = ":material/trending_down:"

# ---------------------------------------------------------------------------
# Strategy labels (no mascot names)
# ---------------------------------------------------------------------------

LABEL_DCA        = "Invest monthly (DCA)"
LABEL_WAIT       = "Wait for a dip"
LABEL_INVEST_NOW = "Invest now"
LABEL_TIERED     = "Tiered deployment"

# ---------------------------------------------------------------------------
# Page descriptions
# ---------------------------------------------------------------------------

PAGE_DESCRIPTIONS = {
    "Today":                 "Is this historically a dip? What have similar levels led to?",
    "Compare choices":       "Head-to-head comparison: monthly DCA vs waiting vs investing now.",
    "Historical scenarios":  "How have similar market levels played out historically?",
    "How it works":          "Methodology, data sources, formulas, and assumptions.",
    "Market overview":       "Browse all tracked assets and their current drawdown status.",
    "Exit strategies":       "Model deterministic exit rules vs never selling.",
    "Currency breakdown":    "How FX movements affected your returns on foreign assets.",
    "Interest rates":        "Official ECB and SNB policy rates as an opportunity-cost baseline.",
}

# ---------------------------------------------------------------------------
# Drawdown descriptions (plain language)
# ---------------------------------------------------------------------------

DRAWDOWN_LABELS: dict[tuple[float, float], str] = {
    (-0.05, 0.0):   "Minimal decline",
    (-0.10, -0.05): "Minor pullback",
    (-0.20, -0.10): "Moderate drawdown",
    (-0.40, -0.20): "Significant drawdown",
    (-1.00, -0.40): "Severe drawdown",
}


def drawdown_label(dd: float) -> str:
    """Return a plain-language label for a drawdown fraction (negative)."""
    for (lo, hi), label in DRAWDOWN_LABELS.items():
        if lo <= dd < hi:
            return label
    return "Severe drawdown"


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER_SHORT = (
    "Educational and informational purposes only. "
    "Not investment advice. Past performance does not guarantee future results."
)
