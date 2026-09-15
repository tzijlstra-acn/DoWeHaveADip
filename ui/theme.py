"""Visual design tokens and chart layout helpers."""

from __future__ import annotations

import plotly.graph_objects as go

# --- Color palette ---
ORANGE = "#F47920"
NAVY = "#1F2144"
GREEN = "#00C896"
RED = "#FF4B6B"
GOLD = "#FFD700"
SILVER = "#C0C0C0"
BRONZE = "#CD7F32"
BLUE = "#4C9BE8"
PURPLE = "#9B59B6"
GRAY = "#6B7280"
BG_CARD = "#1E2130"
BG_HIGHLIGHT = "#252840"

# Strategy identity colors
STRATEGY_COLORS: dict[str, str] = {
    "Monthly Machine": GREEN,
    "DCA": GREEN,
    "Cash Goblin": ORANGE,
    "Wait-for-Dip": ORANGE,
    "Dip Buffet": GOLD,
    "Tiered-Dip": GOLD,
}

# Chart-level layout defaults (dark theme, transparent background)
CHART_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="Inter, sans-serif", size=12),
    xaxis=dict(
        gridcolor="#2D3047",
        linecolor="#3D4066",
        zerolinecolor="#3D4066",
    ),
    yaxis=dict(
        gridcolor="#2D3047",
        linecolor="#3D4066",
        zerolinecolor="#3D4066",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#3D4066",
        borderwidth=1,
        orientation="h",
        y=-0.18,
    ),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=50, b=60),
)


GLOBAL_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="metric-container"] {
    background: #1E2130;
    border: 1px solid #2D3047;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #9CA3AF !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #FAFAFA !important;
}
.stMarkdown {
    font-size: 0.9rem;
}
[data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid #2D3047;
}
[data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: #9CA3AF;
    font-weight: 500;
}
[aria-selected="true"] {
    background: #252840 !important;
    color: #F47920 !important;
    border-bottom: 2px solid #F47920 !important;
}
[data-testid="stSidebar"] {
    background: #13151F;
    border-right: 1px solid #2D3047;
}
[data-testid="stSidebar"] label {
    color: #D1D5DB;
    font-size: 0.85rem;
}
[data-testid="stExpander"] {
    border: 1px solid #2D3047;
    border-radius: 10px;
    background: #1A1D27;
}
[data-testid="stDataFrame"] {
    border: 1px solid #2D3047;
    border-radius: 8px;
}
[data-baseweb="select"] {
    background: #1E2130;
}
.stButton > button {
    background: #F47920;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #D96810;
}
hr {
    border-color: #2D3047;
}
</style>
"""


def apply_chart_layout(fig: go.Figure, title: str = "", subtitle: str = "") -> None:
    """Apply standard dark chart layout to a Plotly figure in-place."""
    layout = dict(CHART_LAYOUT)
    if title:
        layout["title"] = dict(
            text=f"<b>{title}</b>" + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
            x=0.02,
            font=dict(size=15, color="#FAFAFA"),
        )
    fig.update_layout(**layout)
