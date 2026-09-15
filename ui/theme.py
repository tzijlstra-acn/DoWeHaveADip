"""Chart layout helpers and global CSS — light-first theme.

Color constants are re-exported from ui.design_tokens for backward compatibility
with pages that do `from ui.theme import GREEN, ORANGE, RED`.
"""

from __future__ import annotations

import plotly.graph_objects as go

from ui.design_tokens import (
    BG_CARD,
    BG_PAGE,
    BORDER,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    PRIMARY,
    TEXT_PRIMARY,
    WARNING,
)

# ---------------------------------------------------------------------------
# Backward-compatibility aliases — pages import these by name
# ---------------------------------------------------------------------------

ORANGE = WARNING    # was #F47920 arcade orange → now amber warning
GREEN  = POSITIVE   # was #00C896 → now #057A55
RED    = NEGATIVE   # was #FF4B6B → now #E02424
BLUE   = PRIMARY    # was #4C9BE8 → now #1A56DB
GRAY   = NEUTRAL    # unchanged #6B7280
GOLD   = "#C27803"  # kept for tiered strategy (maps to WARNING)
NAVY   = "#1F2144"  # kept (unused in new theme but imported by some pages)
SILVER = "#6B7280"
BRONZE = "#9CA3AF"
PURPLE = "#7C3AED"
BG_HIGHLIGHT = "#F3F4F6"  # was dark highlight — now subtle light bg

# ---------------------------------------------------------------------------
# Chart layout — light theme
# ---------------------------------------------------------------------------

CHART_LAYOUT: dict = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_PAGE,
    font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=12),
    xaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        zerolinecolor=BORDER,
    ),
    yaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        zerolinecolor=BORDER,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        borderwidth=1,
        orientation="h",
        y=-0.18,
    ),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=50, b=60),
)

# ---------------------------------------------------------------------------
# Global CSS — injected on every page via st.markdown(GLOBAL_CSS, ...)
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #6B7280 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
}
.stMarkdown {
    font-size: 0.9rem;
}
[data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid #E5E7EB;
}
[data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: #6B7280;
    font-weight: 500;
}
[aria-selected="true"] {
    background: #EFF6FF !important;
    color: #1A56DB !important;
    border-bottom: 2px solid #1A56DB !important;
}
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    background: #FFFFFF;
}
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
}
.stButton > button {
    background: #1A56DB;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #1446B0;
}
hr {
    border-color: #E5E7EB;
}
</style>
"""


def apply_chart_layout(fig: go.Figure, title: str = "", subtitle: str = "") -> None:
    """Apply standard light chart layout to a Plotly figure in-place."""
    layout = dict(CHART_LAYOUT)
    if title:
        layout["title"] = dict(
            text=f"<b>{title}</b>" + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
            x=0.02,
            font=dict(size=15, color=TEXT_PRIMARY),
        )
    fig.update_layout(**layout)
