"""Plotly chart factory functions — dark-first DIP SIGNAL theme."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.design_tokens import (
    ACCENT_CYAN,
    ACCENT_MAGENTA,
    ACCENT_ORANGE,
    BG_SURFACE,
    BORDER,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
)
from ui.theme import apply_chart_layout

COLORS = {
    "dca": NEUTRAL,
    "dip": ACCENT_CYAN,
    "tiered": ACCENT_MAGENTA,
    "cash": ACCENT_CYAN,
    "neutral": NEUTRAL,
    "positive": POSITIVE,
    "negative": NEGATIVE,
}


# ---------------------------------------------------------------------------
# Existing charts (dark-themed)
# ---------------------------------------------------------------------------


def total_return_chart(
    series_dict: dict[str, pd.Series],
    title: str = "Indexed Total Return (base = 100)",
    currency: str = "EUR",
    as_of: date | None = None,
    data_type: str = "Total Return (Adj.)",
) -> go.Figure:
    """Multi-line indexed total return chart rebased to 100."""
    fig = go.Figure()
    color_list = [ACCENT_CYAN, ACCENT_ORANGE, WARNING, NEUTRAL, NEGATIVE, POSITIVE]

    for i, (name, series) in enumerate(series_dict.items()):
        s = series.dropna()
        if len(s) == 0:
            continue
        rebased = s / s.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased.values,
                name=name,
                line={"color": color_list[i % len(color_list)], "width": 2},
                hovertemplate=f"<b>{name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.1f}}<extra></extra>",
            )
        )

    annotation_text = f"Type: {data_type} | Currency: {currency}"
    if as_of:
        annotation_text += f" | As of: {as_of}"

    apply_chart_layout(fig, title=title)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Indexed Value (start = 100)",
        annotations=[
            {
                "text": annotation_text,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "font": {"size": 10, "color": TEXT_MUTED},
            }
        ],
    )
    return fig


def wealth_comparison_chart(
    dca_ledger: pd.DataFrame,
    dip_ledger: pd.DataFrame,
    currency: str = "EUR",
    as_of: date | None = None,
) -> go.Figure:
    """Dual-strategy wealth curve comparison chart."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dca_ledger.index,
            y=dca_ledger["total_wealth"],
            name="Invest monthly (DCA)",
            line={"color": COLORS["dca"], "width": 2},
            hovertemplate="<b>DCA</b><br>Date: %{x|%Y-%m-%d}<br>Wealth: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dip_ledger.index,
            y=dip_ledger["total_wealth"],
            name="Wait for a dip",
            line={"color": COLORS["dip"], "width": 2},
            hovertemplate="<b>Wait-for-Dip</b><br>Date: %{x|%Y-%m-%d}<br>Wealth: %{y:,.0f}<extra></extra>",
        )
    )

    # Shade area between curves
    fig.add_trace(
        go.Scatter(
            x=list(dca_ledger.index) + list(dip_ledger.index[::-1]),
            y=list(dca_ledger["total_wealth"]) + list(dip_ledger["total_wealth"][::-1]),
            fill="toself",
            fillcolor="rgba(66,232,255,0.06)",
            line={"color": "rgba(255,255,255,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    annotation_text = f"Type: Total Return (Adj.) | Currency: {currency}"
    if as_of:
        annotation_text += f" | As of: {as_of}"

    apply_chart_layout(fig, title=f"Portfolio Wealth: DCA vs Wait-for-Dip ({currency})")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"Wealth ({currency})",
        annotations=[
            {
                "text": annotation_text,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "font": {"size": 10, "color": TEXT_MUTED},
            }
        ],
    )
    return fig


def drawdown_chart(
    series: pd.Series,
    title: str = "Drawdown",
    threshold: float | None = None,
) -> go.Figure:
    """Drawdown waterfall chart with optional threshold line.

    Args:
        series: Drawdown series — values must be ≤ 0. Pass ``drawdown(price_series)``
                rather than the raw price series.
    """
    if series.max() > 0:
        raise ValueError(
            "drawdown_chart() received a series with positive values — expected a drawdown "
            "series (all values ≤ 0). Pass drawdown(price_series) instead of raw prices."
        )
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values * 100,
            fill="tozeroy",
            fillcolor="rgba(255,95,115,0.20)",
            line={"color": NEGATIVE, "width": 1},
            name="Drawdown",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>DD: %{y:.1f}%<extra></extra>",
        )
    )

    if threshold is not None:
        fig.add_hline(
            y=threshold * 100,
            line_dash="dash",
            line_color=ACCENT_ORANGE,
            annotation_text=f"Threshold ({threshold:.0%})",
            annotation_font_color=ACCENT_ORANGE,
        )

    apply_chart_layout(fig, title=title)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
    )
    return fig


# ---------------------------------------------------------------------------
# New chart functions
# ---------------------------------------------------------------------------


def add_dip_highlights(
    fig: go.Figure,
    drawdown_series: pd.Series,
    threshold: float = -0.10,
    row: int = 1,
    col: int = 1,
) -> None:
    """Add shaded vrect regions for each dip episode below threshold.

    The most recent episode gets a brighter highlight + annotation.
    """
    triggered = drawdown_series <= threshold
    episodes: list[tuple] = []
    in_episode = False
    start = None

    for dt, val in triggered.items():
        if val and not in_episode:
            in_episode = True
            start = dt
        elif not val and in_episode:
            in_episode = False
            episodes.append((start, dt))
    if in_episode and start is not None:
        episodes.append((start, drawdown_series.index[-1]))

    for i, (ep_start, ep_end) in enumerate(episodes):
        is_last = i == len(episodes) - 1
        fig.add_vrect(
            x0=ep_start,
            x1=ep_end,
            fillcolor="rgba(255,95,115,0.22)" if is_last else "rgba(255,95,115,0.10)",
            layer="below",
            line_width=0,
            row=row,
            col=col,
        )

    if episodes:
        last_start, last_end = episodes[-1]
        ep_dd = drawdown_series[last_start:last_end].min()
        fig.add_annotation(
            x=last_end,
            y=1.04,
            xref="x",
            yref="paper",
            text=f"Last dip: {ep_dd:.1%}",
            showarrow=False,
            font=dict(color=NEGATIVE, size=11),
            bgcolor="rgba(255,95,115,0.2)",
            bordercolor=NEGATIVE,
            borderwidth=1,
            borderpad=4,
        )


def plot_strategy_wealth(
    dca_ledger: pd.DataFrame,
    dip_ledger: pd.DataFrame,
    base_currency: str = "EUR",
    tiered_ledger: pd.DataFrame | None = None,
) -> go.Figure:
    """Dual-panel: wealth curves (top) + drawdown (bottom) with dip highlights."""
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Portfolio Value", "Drawdown (%)"],
    )

    # Top panel: wealth curves
    # DCA: NEUTRAL solid
    fig.add_trace(
        go.Scatter(
            x=dca_ledger.index,
            y=dca_ledger["total_wealth"],
            name="Invest monthly (DCA)",
            line=dict(color=NEUTRAL, width=2, dash="solid"),
            hovertemplate="<b>DCA</b><br>%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Dip: ACCENT_CYAN dash
    fig.add_trace(
        go.Scatter(
            x=dip_ledger.index,
            y=dip_ledger["total_wealth"],
            name="Wait for a dip",
            line=dict(color=ACCENT_CYAN, width=2, dash="dash"),
            hovertemplate="<b>Dip</b><br>%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if tiered_ledger is not None:
        # Tiered: ACCENT_MAGENTA dashdot
        fig.add_trace(
            go.Scatter(
                x=tiered_ledger.index,
                y=tiered_ledger["total_wealth"],
                name="Tiered deployment",
                line=dict(color=ACCENT_MAGENTA, width=2, dash="dashdot"),
                hovertemplate="<b>Tiered</b><br>%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Bottom panel: DCA drawdown
    fig.add_trace(
        go.Scatter(
            x=dca_ledger.index,
            y=dca_ledger["dd"] * 100,
            fill="tozeroy",
            fillcolor="rgba(255,95,115,0.18)",
            line=dict(color=NEGATIVE, width=1),
            name="DCA Drawdown",
            showlegend=False,
            hovertemplate="DD: %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Dip highlights on top panel using dip strategy drawdown
    add_dip_highlights(fig, dip_ledger["dd"], threshold=-0.05, row=1, col=1)

    apply_chart_layout(
        fig,
        title=f"Strategy portfolio value ({base_currency})",
        subtitle="Shaded = dip episodes | DCA vs Wait-for-dip",
    )
    fig.update_yaxes(title_text=f"Wealth ({base_currency})", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", ticksuffix="%", row=2, col=1)
    fig.update_layout(height=550)
    return fig


def plot_drawdown_gauge(current_dd: float, threshold: float) -> go.Figure:
    """Plotly gauge showing current drawdown vs threshold."""
    gauge_val = abs(current_dd) * 100  # positive number for gauge
    threshold_abs = abs(threshold) * 100

    color = POSITIVE if current_dd > threshold else NEGATIVE

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            delta={"reference": threshold_abs, "increasing": {"color": NEGATIVE}, "decreasing": {"color": POSITIVE}},
            number={"suffix": "% from peak", "font": {"color": color, "size": 18}},
            gauge={
                "axis": {"range": [0, 50], "ticksuffix": "%", "tickfont": {"color": TEXT_PRIMARY}},
                "bar": {"color": color},
                "bgcolor": BG_SURFACE,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, threshold_abs], "color": "rgba(55,211,154,0.12)"},
                    {"range": [threshold_abs, 50], "color": "rgba(255,95,115,0.12)"},
                ],
                "threshold": {
                    "line": {"color": ACCENT_ORANGE, "width": 3},
                    "thickness": 0.8,
                    "value": threshold_abs,
                },
            },
        )
    )

    apply_chart_layout(fig, title="Current Drawdown", subtitle=f"Threshold: {threshold:.0%}")
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_forward_returns_box(episodes_df: pd.DataFrame) -> go.Figure:
    """Box + strip plot of forward returns at multiple horizons."""
    if episodes_df is None or episodes_df.empty:
        fig = go.Figure()
        apply_chart_layout(fig, title="Forward Returns (no episodes found)")
        return fig

    horizon_cols = [c for c in episodes_df.columns if c.startswith("fwd_")]
    if not horizon_cols:
        fig = go.Figure()
        apply_chart_layout(fig, title="Forward Returns (no horizon data)")
        return fig

    fig = go.Figure()
    colors_cycle = [ACCENT_ORANGE, WARNING, POSITIVE, ACCENT_CYAN, NEGATIVE]

    for i, col in enumerate(horizon_cols):
        label = col.replace("fwd_", "").replace("_", " ")
        vals = episodes_df[col].dropna() * 100
        if len(vals) == 0:
            continue
        color = colors_cycle[i % len(colors_cycle)]
        fig.add_trace(
            go.Box(
                y=vals,
                name=label,
                marker_color=color,
                boxmean=True,
                hovertemplate=f"<b>{label}</b><br>Return: %{{y:.1f}}%<extra></extra>",
            )
        )

    apply_chart_layout(fig, title="Forward Returns After Dip Entry", subtitle="Annualised returns by horizon")
    fig.update_layout(
        yaxis_title="Return (%)",
        yaxis_ticksuffix="%",
        showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, annotation_text="Breakeven")
    return fig


def plot_rolling_win_rate(rolling_df: pd.DataFrame) -> go.Figure:
    """Heatmap: threshold (rows) vs horizon (cols), color = % windows where dip beat DCA."""
    if rolling_df is None or rolling_df.empty:
        fig = go.Figure()
        apply_chart_layout(fig, title="Rolling Win Rate (no data)")
        return fig

    pivot = rolling_df.copy()

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values * 100,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale=[[0, NEGATIVE], [0.5, NEUTRAL], [1, POSITIVE]],
            zmid=50,
            colorbar=dict(
                title=dict(text="Win Rate (%)", font=dict(color=TEXT_PRIMARY)),
                ticksuffix="%",
                tickfont=dict(color=TEXT_PRIMARY),
            ),
            hovertemplate="Threshold: %{y}<br>Horizon: %{x}<br>Win rate: %{z:.1f}%<extra></extra>",
        )
    )

    apply_chart_layout(
        fig,
        title="Rolling Win Rate: Dip vs DCA",
        subtitle="% of rolling windows where dip strategy won",
    )
    fig.update_layout(
        xaxis_title="Horizon",
        yaxis_title="Dip Threshold",
    )
    return fig


def plot_fx_decomposition(
    tr_native: pd.Series,
    fx_contribution: pd.Series,
    tr_base: pd.Series,
    base_currency: str = "EUR",
) -> go.Figure:
    """Stacked area: native return + FX contribution = base-currency total return."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=tr_native.index,
            y=(tr_native - 1) * 100,
            name="Native Return",
            fill="tozeroy",
            fillcolor="rgba(66,232,255,0.15)",
            line=dict(color=ACCENT_CYAN, width=1.5),
            hovertemplate="Native: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fx_contribution.index,
            y=fx_contribution * 100,
            name="FX Contribution",
            fill="tozeroy",
            fillcolor="rgba(255,145,77,0.18)",
            line=dict(color=ACCENT_ORANGE, width=1.5),
            hovertemplate="FX: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tr_base.index,
            y=(tr_base - 1) * 100,
            name=f"Total ({base_currency})",
            line=dict(color=POSITIVE, width=2, dash="dot"),
            hovertemplate=f"Total ({base_currency}): %{{y:.2f}}%<extra></extra>",
        )
    )

    apply_chart_layout(
        fig,
        title=f"FX Return Decomposition (base = {base_currency})",
        subtitle="Native return + FX contribution = total base-currency return",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        yaxis_ticksuffix="%",
    )
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    return fig


def add_named_episode_labels(
    fig: go.Figure,
    episodes: list[dict],
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
    row: int = 1,
    col: int = 1,
) -> None:
    """Add vertical shaded regions + short label annotations for named historical episodes.

    Only shows episodes that overlap with [date_start, date_end].
    Labels appear as small rotated text at the top of each shaded region.
    """
    category_colors = {
        "bubble": "rgba(155,89,182,0.12)",
        "financial": "rgba(255,95,115,0.14)",
        "macro": "rgba(255,145,77,0.10)",
        "geopolitical": "rgba(66,232,255,0.10)",
    }

    for ep in episodes:
        ep_start = pd.Timestamp(ep["start"])
        ep_end = pd.Timestamp(ep["end"])

        # Filter to visible date range
        if date_end is not None and ep_start > date_end:
            continue
        if date_start is not None and ep_end < date_start:
            continue

        color = category_colors.get(ep.get("category", "macro"), "rgba(128,128,128,0.10)")
        border_color = color.replace("0.10", "0.4").replace("0.12", "0.4").replace("0.14", "0.4")

        fig.add_vrect(
            x0=ep_start,
            x1=ep_end,
            fillcolor=color,
            layer="below",
            line_width=0.5,
            line_color=border_color,
        )

        # Label at the top of the shaded region
        mid_date = ep_start + (ep_end - ep_start) / 2
        fig.add_annotation(
            x=mid_date,
            y=1.0,
            xref="x",
            yref="paper",
            text=ep["label"],
            showarrow=False,
            font=dict(size=9, color=TEXT_MUTED),
            textangle=-45,
            xanchor="left",
            yanchor="bottom",
        )


def plot_savings_rates(rates_df: pd.DataFrame) -> go.Figure:
    """Line chart of deposit/policy rates over time."""
    if rates_df is None or rates_df.empty:
        fig = go.Figure()
        apply_chart_layout(fig, title="Savings Rates (no data)")
        return fig

    fig = go.Figure()
    color_map = {
        "DE": ACCENT_ORANGE,
        "NL": WARNING,
        "CH": ACCENT_CYAN,
        "ECB": POSITIVE,
        "SNB": ACCENT_CYAN,
    }
    color_list = [ACCENT_ORANGE, POSITIVE, ACCENT_CYAN, WARNING, NEGATIVE]

    for i, col in enumerate(rates_df.columns):
        key = str(col).upper()
        color = color_map.get(key, color_list[i % len(color_list)])
        vals = rates_df[col].dropna()
        if len(vals) == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=vals.index,
                y=vals.values * 100,
                name=str(col),
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Rate: %{{y:.2f}}%<extra></extra>",
            )
        )

        # Annotate latest observation
        if len(vals) > 0:
            latest_val = float(vals.iloc[-1]) * 100
            fig.add_annotation(
                x=vals.index[-1],
                y=latest_val,
                text=f" {latest_val:.2f}%",
                showarrow=False,
                font=dict(color=color, size=11),
                xanchor="left",
            )

    apply_chart_layout(fig, title="Official Deposit / Policy Rates", subtitle="Monthly frequency")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Rate (% p.a.)",
        yaxis_ticksuffix="%",
    )
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string to an rgba() CSS string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_sweep_heatmap(
    pivot_df: pd.DataFrame,
    title: str = "Win Rate vs DCA",
    subtitle: str = "% of rolling windows where dip strategy beats monthly DCA",
    value_fmt: str = ".0%",
    zmid: float = 50,
    colorscale: str | list | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Heatmap: threshold (rows) x deploy fraction (cols).

    Green = beats DCA more often. Red = rarely beats DCA.
    Annotates each cell with the formatted value.

    Args:
        pivot_df: DataFrame with thresholds as index, deploy_pcts as columns.
        title: Chart title.
        subtitle: Chart subtitle.
        value_fmt: Python format spec for cell annotations.
        zmid: Midpoint of the colorscale (default 50 for win-rate 0-100 scale).
        colorscale: Plotly colorscale override.
        zmin: Minimum value for colorscale.
        zmax: Maximum value for colorscale.
    """
    z_vals = pivot_df.values.astype(float)

    text_vals = [
        [f"{v:{value_fmt}}" if not np.isnan(v) else "" for v in row] for row in z_vals
    ]

    # Scale z to 0-100 for percentage metrics
    z_display = z_vals * 100 if value_fmt.endswith("%") else z_vals
    zmid_display = zmid * 100 if value_fmt.endswith("%") else zmid

    default_colorscale = [[0, NEGATIVE], [0.4, NEUTRAL], [0.6, NEUTRAL], [1, POSITIVE]]
    cs = colorscale if colorscale is not None else default_colorscale

    heatmap_kwargs: dict = dict(
        z=z_display,
        x=list(pivot_df.columns),
        y=list(pivot_df.index),
        colorscale=cs,
        zmid=zmid_display,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=12, color=TEXT_PRIMARY),
        colorbar=dict(
            title=dict(text=title, font=dict(color=TEXT_PRIMARY)),
            tickfont=dict(color=TEXT_PRIMARY),
        ),
        hovertemplate="Threshold: %{y}<br>Deploy: %{x}<br>Value: %{z:.1f}<extra></extra>",
    )
    if zmin is not None:
        heatmap_kwargs["zmin"] = zmin * 100 if value_fmt.endswith("%") else zmin
    if zmax is not None:
        heatmap_kwargs["zmax"] = zmax * 100 if value_fmt.endswith("%") else zmax

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))

    apply_chart_layout(fig, title=title, subtitle=subtitle)
    fig.update_layout(
        xaxis_title="Deploy fraction",
        yaxis_title="Dip threshold",
        height=400,
    )
    return fig


def plot_fan_chart(
    sim_results: list,  # list[PathSimulation]
    horizon_months: int,
    monthly_contribution: float = 0.0,
    currency: str = "EUR",
) -> go.Figure:
    """Fan chart showing wealth distribution for multiple deploy fractions.

    One fan per deploy_pct (semi-transparent band from p25 to p75 + p50 line).

    Args:
        sim_results: List of PathSimulation objects.
        horizon_months: Simulation horizon in months (for axis vlines).
        monthly_contribution: Monthly contribution (unused in chart, kept for signature).
        currency: Currency label for y-axis.
    """
    from ui.design_tokens import CHART_COLORS

    fig = go.Figure()

    for i, sim in enumerate(sim_results):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        label = f"Deploy {sim.deploy_pct:.0%}"
        n = len(sim.p50_wealth)
        x = list(range(1, n + 1))

        fill_color = _hex_to_rgba(color, 0.15) if color.startswith("#") else color

        # Shaded band (p25 to p75)
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(sim.p75_wealth) + list(sim.p25_wealth[::-1]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Median line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sim.p50_wealth,
                name=f"{label} (P(beats DCA): {sim.prob_beats_dca:.0%})",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{label}</b><br>Month %{{x}}<br>Median wealth: %{{y:,.0f}}<extra></extra>"
                ),
            )
        )

    apply_chart_layout(
        fig,
        title=f"Conditional Path Simulation ({currency})",
        subtitle="Fan = P25-P75 | Line = median | Based on historical continuation paths",
    )
    fig.update_layout(
        xaxis_title="Months from now",
        yaxis_title=f"Total Wealth ({currency})",
        height=450,
    )

    if horizon_months >= 12:
        fig.add_vline(x=12, line_dash="dot", line_color=NEUTRAL, annotation_text="1Y")
    if horizon_months >= 24:
        fig.add_vline(x=24, line_dash="dot", line_color=NEUTRAL, annotation_text="2Y")
    if horizon_months >= 36:
        fig.add_vline(x=36, line_dash="dot", line_color=NEUTRAL, annotation_text="3Y")

    return fig


def plot_threshold_wealth_bar(results: list[dict], baseline_wealth: float, currency: str = "EUR") -> go.Figure:
    """Bar chart: ending wealth per threshold vs DCA baseline."""
    if not results:
        fig = go.Figure()
        apply_chart_layout(fig, title="Threshold sweep (no results)")
        return fig

    thresholds = [f"{r['threshold']:.0%}" for r in results]
    wealths = [r["ending_wealth"] for r in results]
    colors = [POSITIVE if w >= baseline_wealth else NEGATIVE for w in wealths]

    fig = go.Figure(
        go.Bar(
            x=thresholds,
            y=wealths,
            marker_color=colors,
            hovertemplate="Threshold: %{x}<br>Ending Wealth: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=baseline_wealth,
        line_dash="dash",
        line_color=POSITIVE,
        annotation_text=f"DCA baseline ({baseline_wealth:,.0f})",
        annotation_font_color=POSITIVE,
    )

    apply_chart_layout(
        fig,
        title=f"Ending Wealth by Dip Threshold ({currency})",
        subtitle="Green = beats DCA | Red = DCA wins",
    )
    fig.update_layout(
        xaxis_title="Dip Threshold",
        yaxis_title=f"Ending Wealth ({currency})",
        bargap=0.2,
    )
    return fig
