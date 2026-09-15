"""Reusable Streamlit UI components — dark-first DIP SIGNAL brand system."""

from __future__ import annotations

import datetime
import html
import traceback

import streamlit as st

from dipdca.data.errors import LiveDataUnavailable
from dipdca.data.market_models import DataFreshness
from dipdca.models import StrategyResult
from ui.design_tokens import (
    ACCENT_CYAN,
    BG_SURFACE,
    BG_SURFACE_RAISED,
    BORDER,
    BRAND_GRADIENT,
    NEGATIVE,
    POSITIVE,
    RADIUS_LG,
    RADIUS_MD,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
)
from ui.formatting import fmt_pct


def demo_banner() -> None:
    """Full-width warning banner for demo mode."""
    st.warning(
        "**DEMO DATA** — No live market data available. Showing synthetic fixtures only. "
        "Results are illustrative, not a representation of any real market.",
        icon=":material/warning:",
    )


def live_data_error(error: LiveDataUnavailable, context: str = "") -> None:
    """Render a live-data unavailable notice with Retry and collapsed technical details."""
    location = f" ({context})" if context else ""
    st.error(
        f"**Live market data is unavailable{location}.** "
        "No backup dataset is being shown. "
        "Check your connection and retry, or return later.",
        icon=":material/error:",
    )
    col_retry, col_space = st.columns([1, 4])
    with col_retry:
        if st.button("Retry", key=f"retry_{context}"):
            st.cache_data.clear()
            st.rerun()
    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}", language="text")


def freshness_caption(freshness: DataFreshness) -> None:
    """Render a one-line data freshness caption."""
    st.caption(freshness.caption())


def metric_card(
    label: str,
    value: str,
    delta: str = "",
    help: str = "",
    color: str = ACCENT_CYAN,
) -> None:
    """Render a styled metric using st.metric."""
    st.metric(label=label, value=value, delta=delta or None, help=help or None)


def strategy_metrics(result: StrategyResult, currency: str = "EUR") -> None:
    """Display a row of metric cards for a StrategyResult."""
    cols = st.columns(5)
    with cols[0]:
        st.metric("Ending Wealth", f"{currency} {result.ending_wealth:,.0f}")
    with cols[1]:
        xirr_val = f"{result.xirr:.1%}" if result.xirr is not None else "N/A"
        st.metric("XIRR", xirr_val)
    with cols[2]:
        st.metric("Max Drawdown", f"{result.max_drawdown:.1%}")
    with cols[3]:
        st.metric("Time in Market", f"{result.time_in_market_pct:.1%}")
    with cols[4]:
        st.metric("Deployments", str(result.n_deployments))


def strategy_comparison_cards(
    dca_result: StrategyResult,
    dip_result: StrategyResult,
    currency: str = "EUR",
) -> None:
    """Side-by-side comparison cards for DCA vs Wait-for-dip."""
    dca_wins = dca_result.ending_wealth >= dip_result.ending_wealth
    diff = abs(dca_result.ending_wealth - dip_result.ending_wealth)
    winner_label = "Invest monthly came out ahead" if dca_wins else "Wait-for-dip came out ahead"

    col1, col_mid, col2 = st.columns([2, 1, 2])

    with col1:
        border_color = POSITIVE if dca_wins else BORDER
        st.markdown(
            f"""
            <div style="border:2px solid {border_color};border-radius:{RADIUS_LG};
                        padding:16px;text-align:center;background:{BG_SURFACE};">
                <h3 style="margin:0;color:{border_color}">Invest monthly (DCA)</h3>
                <p style="font-size:2em;margin:8px 0;font-weight:bold;
                          color:{TEXT_PRIMARY}">{currency} {dca_result.ending_wealth:,.0f}</p>
                <p style="margin:4px 0;color:{TEXT_SECONDARY}">Ann. return: {fmt_pct(dca_result.xirr)}</p>
                <p style="margin:4px 0;color:{TEXT_SECONDARY}">Max DD: {fmt_pct(dca_result.max_drawdown)}</p>
                <p style="margin:4px 0;color:{TEXT_SECONDARY}">In market: {fmt_pct(dca_result.time_in_market_pct)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mid:
        st.markdown(
            f"""
            <div style="text-align:center;padding:20px 0;">
                <p style="font-size:0.85em;color:{TEXT_PRIMARY};font-weight:bold;
                          margin:8px 0">{winner_label}</p>
                <p style="font-size:0.8em;color:{TEXT_SECONDARY}">by {currency} {diff:,.0f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        border_color = POSITIVE if not dca_wins else BORDER
        st.markdown(
            f"""
            <div style="border:2px solid {border_color};border-radius:{RADIUS_LG};
                        padding:16px;text-align:center;background:{BG_SURFACE};">
                <h3 style="margin:0;color:{border_color}">Wait for a dip</h3>
                <p style="font-size:2em;margin:8px 0;font-weight:bold;
                          color:{TEXT_PRIMARY}">{currency} {dip_result.ending_wealth:,.0f}</p>
                <p style="margin:4px 0;color:{TEXT_SECONDARY}">Ann. return: {fmt_pct(dip_result.xirr)}</p>
                <p style="margin:4px 0;color:{TEXT_SECONDARY}">Max DD: {fmt_pct(dip_result.max_drawdown)}</p>
                <p style="margin:4px 0;color:{TEXT_SECONDARY}">In market: {fmt_pct(dip_result.time_in_market_pct)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def dip_badge(drawdown_pct: float, label: str, threshold: float) -> None:
    """Colored badge showing current dip status."""
    if drawdown_pct <= threshold:
        color = NEGATIVE
        status = "DIP TRIGGERED"
    elif drawdown_pct <= threshold * 0.5:
        color = WARNING
        status = "APPROACHING"
    else:
        color = POSITIVE
        status = "IDLE"

    safe_label = html.escape(label)
    st.markdown(
        f"""
        <div style="border:1px solid {color};border-radius:{RADIUS_MD};padding:10px 14px;
                    background:{BG_SURFACE_RAISED};display:inline-block;margin:4px;">
            <span style="color:{color};font-weight:bold">{status}</span>
            <br><span style="color:{TEXT_PRIMARY};font-size:1.1em">{safe_label}</span>
            <br><span style="color:{TEXT_MUTED};font-size:0.85em">{drawdown_pct:.1%} from peak</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_card(
    status_label: str,
    drawdown_pct: float,
    threshold_pct: float,
    days_since_high: int,
    distance_to_threshold_pct: float,
    status_color: str,
) -> None:
    """Hero signal card for the Today page."""
    safe_status = html.escape(status_label)
    dd_display = f"{drawdown_pct:.1%}"
    dist_display = f"{distance_to_threshold_pct:.1%}"

    st.markdown(
        f"""
        <div style="background:{BG_SURFACE};border:1px solid {status_color};
                    border-radius:{RADIUS_LG};padding:28px 32px;margin:16px 0 24px 0;">
            <div style="display:inline-block;background:{status_color}22;
                        border:1px solid {status_color};border-radius:4px;
                        padding:2px 10px;margin-bottom:12px;">
                <span style="color:{status_color};font-size:0.75rem;font-weight:700;
                             text-transform:uppercase;letter-spacing:0.1em">{safe_status}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px;">
                <div>
                    <p style="margin:0;color:{TEXT_MUTED};font-size:0.7rem;
                              text-transform:uppercase;letter-spacing:0.08em">Current drawdown</p>
                    <p style="margin:4px 0 0;color:{status_color};font-size:2rem;
                              font-weight:800;font-variant-numeric:tabular-nums">{dd_display}</p>
                </div>
                <div>
                    <p style="margin:0;color:{TEXT_MUTED};font-size:0.7rem;
                              text-transform:uppercase;letter-spacing:0.08em">Days from high</p>
                    <p style="margin:4px 0 0;color:{TEXT_PRIMARY};font-size:2rem;
                              font-weight:800;font-variant-numeric:tabular-nums">{days_since_high:,}</p>
                </div>
                <div>
                    <p style="margin:0;color:{TEXT_MUTED};font-size:0.7rem;
                              text-transform:uppercase;letter-spacing:0.08em">Threshold</p>
                    <p style="margin:4px 0 0;color:{TEXT_PRIMARY};font-size:1.4rem;
                              font-weight:700;font-variant-numeric:tabular-nums">{threshold_pct:.1%}</p>
                </div>
                <div>
                    <p style="margin:0;color:{TEXT_MUTED};font-size:0.7rem;
                              text-transform:uppercase;letter-spacing:0.08em">Distance to threshold</p>
                    <p style="margin:4px 0 0;color:{TEXT_PRIMARY};font-size:1.4rem;
                              font-weight:700;font-variant-numeric:tabular-nums">{dist_display}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def conclusion_banner(text: str, sub: str = "", color: str = "") -> None:
    """Full-width conclusion strip with brand left border."""
    if not color:
        color = ACCENT_CYAN
    safe_text = html.escape(text)
    safe_sub = html.escape(sub) if sub else ""
    sub_html = (
        f'<p style="margin:4px 0 0;color:{TEXT_SECONDARY};font-size:0.9rem">{safe_sub}</p>'
        if safe_sub
        else ""
    )
    st.markdown(
        f"""
        <div style="background:{BG_SURFACE_RAISED};border-left:4px solid {color};
                    border-radius:0 {RADIUS_MD} {RADIUS_MD} 0;
                    padding:14px 20px;margin:16px 0 24px 0">
            <p style="margin:0;color:{TEXT_PRIMARY};font-size:1em;
                      font-weight:700;text-transform:uppercase;
                      letter-spacing:0.05em">{safe_text}</p>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def brand_motif() -> str:
    """Return the DIP SIGNAL SVG motif as an HTML string."""
    return """<svg aria-hidden="true" focusable="false" width="72" height="28"
                   viewBox="0 0 72 28" fill="none"
                   style="display:inline-block;vertical-align:middle;margin-right:8px">
        <polyline points="0,14 22,24 36,4 72,2"
                  stroke="#FF4DC4" stroke-width="1" opacity="0.3" stroke-linecap="round"/>
        <polyline points="0,16 22,26 36,6 72,4"
                  stroke="#42E8FF" stroke-width="1" opacity="0.5" stroke-linecap="round"/>
        <polyline points="0,18 22,28 36,8 72,6"
                  stroke="#42E8FF" stroke-width="2.5" stroke-linecap="round"/>
    </svg>"""


def sidebar_simulation_params() -> dict:
    """Render sidebar inputs and return a dict of param values."""
    st.sidebar.header("Simulation Parameters")

    monthly = st.sidebar.number_input(
        "Monthly Contribution (EUR)", min_value=10, max_value=100_000, value=500, step=50
    )
    payday = st.sidebar.slider("Payday (day of month)", min_value=1, max_value=28, value=25)
    initial = st.sidebar.number_input(
        "Initial Investment (EUR)", min_value=0, max_value=1_000_000, value=0, step=100
    )

    st.sidebar.subheader("Strategy Settings")
    dip_threshold = st.sidebar.slider(
        "Dip Threshold (%)", min_value=-50, max_value=-1, value=-5, step=1
    ) / 100.0

    max_wait = st.sidebar.slider("Max Wait (months)", min_value=1, max_value=60, value=24)

    st.sidebar.subheader("Costs")
    fixed_fee = st.sidebar.number_input("Fixed Fee (EUR)", min_value=0.0, value=0.0, step=0.5)
    pct_fee = st.sidebar.slider("% Fee (bps)", min_value=0, max_value=100, value=10) / 10_000.0

    st.sidebar.subheader("Date Range")
    start = st.sidebar.date_input("Start Date", value=datetime.date(2015, 1, 1))
    end = st.sidebar.date_input("End Date", value=datetime.date(2024, 12, 31))

    return {
        "monthly_contribution": float(monthly),
        "payday": payday,
        "initial_investment": float(initial),
        "dip_threshold": dip_threshold,
        "max_wait_months": max_wait,
        "fixed_fee": fixed_fee,
        "pct_fee": pct_fee,
        "start_date": start,
        "end_date": end,
    }


def data_source_caption(
    source: str,
    as_of: object,
    is_total_return: bool,
    currency: str,
    freshness: DataFreshness | None = None,
) -> None:
    """Render a small caption with data provenance info."""
    if freshness is not None:
        st.caption(freshness.caption())
    else:
        tr_label = "Total Return (Adj.)" if is_total_return else "Price Only"
        st.caption(
            f"Data: {source} | Type: {tr_label} | Currency: {currency} | As of: {as_of}"
        )


def evidence_label(n: int) -> str:
    """Return evidence-quality label based on sample size."""
    if n < 10:
        return "Insufficient history"
    if n < 20:
        return f"Small sample (n={n})"
    return f"Statistically suggestive (n={n})"


def page_header(title: str, subtitle: str, icon: str = "") -> None:
    """Render a styled page header with title and subtitle.

    The icon parameter is accepted for backward compatibility but not rendered
    in HTML — Material Symbols strings don't work in raw HTML. The nav sidebar
    already shows the page icon.
    """
    safe_title = html.escape(title)
    safe_sub = html.escape(subtitle)
    st.markdown(
        f"""
        <div style="margin-bottom:24px;padding-bottom:16px;
                    border-bottom:1px solid {BORDER};position:relative">
            <div style="height:3px;background:{BRAND_GRADIENT};
                        border-radius:2px;margin-bottom:12px"></div>
            <h1 style="font-size:clamp(1.6rem,4vw,2.8rem);font-weight:800;
                       color:{TEXT_PRIMARY};margin:0;letter-spacing:-0.02em;
                       font-family:'Arial Narrow','Roboto Condensed',Arial,sans-serif">
                {safe_title}
            </h1>
            <p style="color:{TEXT_MUTED};margin:6px 0 0 0;font-size:0.9rem">{safe_sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_size_badge(n: int) -> None:
    """Display a colored sample-size badge."""
    if n < 10:
        color = NEGATIVE
        label = f"n={n} — insufficient"
    elif n < 20:
        color = WARNING
        label = f"n={n} — small sample"
    else:
        color = POSITIVE
        label = f"n={n} — suggestive"

    st.markdown(
        f'<span style="background:{color}22;border:1px solid {color};'
        f'border-radius:4px;padding:2px 8px;color:{color};font-size:0.85em">'
        f"Sample size: {label}</span>",
        unsafe_allow_html=True,
    )
