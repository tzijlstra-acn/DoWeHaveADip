"""Chart layout helpers and global CSS — dark-first DIP SIGNAL theme.

Color constants are re-exported from ui.design_tokens for backward compatibility
with pages that do `from ui.theme import GREEN, ORANGE, RED`.
"""

from __future__ import annotations

import plotly.graph_objects as go

from ui.design_tokens import (
    ACCENT_CYAN,
    ACCENT_MAGENTA,
    ACCENT_ORANGE,
    BG_SURFACE,
    BG_SURFACE_ACTIVE,
    BG_SURFACE_RAISED,
    BORDER,
    BORDER_STRONG,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    RADIUS_MD,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
)

# ---------------------------------------------------------------------------
# Backward-compatibility aliases — pages import these by name
# ---------------------------------------------------------------------------

ORANGE = ACCENT_ORANGE
GREEN  = POSITIVE
RED    = NEGATIVE
BLUE   = ACCENT_CYAN
GRAY   = NEUTRAL
GOLD   = WARNING
NAVY   = BG_SURFACE_RAISED
SILVER = TEXT_SECONDARY
BRONZE = TEXT_MUTED
PURPLE = ACCENT_MAGENTA
BG_HIGHLIGHT = BG_SURFACE_ACTIVE

# ---------------------------------------------------------------------------
# Chart layout — dark theme
# ---------------------------------------------------------------------------

CHART_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_SURFACE,
    font=dict(color=TEXT_PRIMARY, family="'Inter','Segoe UI',sans-serif", size=13),
    xaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER_STRONG,
        zerolinecolor=BORDER_STRONG,
        tickfont=dict(color=TEXT_SECONDARY, size=12),
    ),
    yaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER_STRONG,
        zerolinecolor=BORDER_STRONG,
        tickfont=dict(color=TEXT_SECONDARY, size=12),
    ),
    legend=dict(
        bgcolor=BG_SURFACE_RAISED,
        bordercolor=BORDER,
        borderwidth=1,
        orientation="h",
        y=-0.20,
        font=dict(color=TEXT_PRIMARY),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=BG_SURFACE_RAISED,
        bordercolor=BORDER_STRONG,
        font=dict(color=TEXT_PRIMARY, size=13),
    ),
    margin=dict(l=10, r=10, t=50, b=70),
)

# ---------------------------------------------------------------------------
# Global CSS — injected on every page via st.markdown(GLOBAL_CSS, ...)
# ---------------------------------------------------------------------------

GLOBAL_CSS = f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

[data-testid="metric-container"] {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_MD};
    padding: 16px 20px;
}}
[data-testid="metric-container"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {TEXT_PRIMARY} !important;
    font-variant-numeric: tabular-nums;
}}
.stMarkdown {{
    font-size: 0.9rem;
}}
[data-baseweb="tab-list"] {{
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid {BORDER};
}}
[data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: {TEXT_SECONDARY};
    font-weight: 500;
}}
[aria-selected="true"] {{
    background: {BG_SURFACE_ACTIVE} !important;
    color: {ACCENT_CYAN} !important;
    border-bottom: 2px solid {ACCENT_CYAN} !important;
}}
[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS_MD};
    background: {BG_SURFACE};
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
.stButton > button {{
    background: {ACCENT_CYAN};
    color: #07070A;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: background 0.2s;
}}
.stButton > button:hover {{
    background: #2ED4EA;
}}
hr {{
    border-color: {BORDER};
}}
*:focus-visible {{
    outline: 2px solid {ACCENT_CYAN};
    outline-offset: 2px;
}}
@media (prefers-reduced-motion: reduce) {{
    * {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }}
}}
</style>
"""


def apply_chart_layout(fig: go.Figure, title: str = "", subtitle: str = "") -> None:
    """Apply standard dark chart layout to a Plotly figure in-place."""
    layout = dict(CHART_LAYOUT)
    if title:
        layout["title"] = dict(
            text=f"<b>{title}</b>" + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
            x=0.02,
            font=dict(size=15, color=TEXT_PRIMARY),
        )
    fig.update_layout(**layout)
