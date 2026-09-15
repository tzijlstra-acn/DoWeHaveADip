"""Reusable Streamlit UI components."""

from __future__ import annotations

import datetime

import streamlit as st

from dipdca.models import StrategyResult
from ui.formatting import fmt_pct


def demo_banner() -> None:
    """Full-width orange warning banner for demo mode."""
    st.warning(
        "**DEMO DATA** — No live market data available. Showing synthetic fixtures only. "
        "Results are illustrative, not a representation of any real market.",
        icon="⚠️",
    )


def metric_card(
    label: str,
    value: str,
    delta: str = "",
    help: str = "",
    color: str = "#F47920",
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
    """Side-by-side comparison hero cards for DCA vs Dip."""
    dca_wins = dca_result.ending_wealth >= dip_result.ending_wealth
    winner_label = "Monthly Machine wins!" if dca_wins else "Cash Goblin wins!"
    diff = abs(dca_result.ending_wealth - dip_result.ending_wealth)

    col1, col_mid, col2 = st.columns([2, 1, 2])

    with col1:
        color = "#00C896" if dca_wins else "#6B7280"
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:10px; padding:16px; text-align:center;">
                <h3 style="margin:0;color:{color}">🤖 Monthly Machine</h3>
                <p style="font-size:2em;margin:8px 0;font-weight:bold">{currency} {dca_result.ending_wealth:,.0f}</p>
                <p style="margin:4px 0;color:#aaa">XIRR: {fmt_pct(dca_result.xirr)}</p>
                <p style="margin:4px 0;color:#aaa">Max DD: {fmt_pct(dca_result.max_drawdown)}</p>
                <p style="margin:4px 0;color:#aaa">In market: {fmt_pct(dca_result.time_in_market_pct)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mid:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px 0;">
                <p style="font-size:1.5em;margin:0">⚔️</p>
                <p style="font-size:0.85em;color:#F47920;font-weight:bold;margin:8px 0">{winner_label}</p>
                <p style="font-size:0.8em;color:#aaa">by {currency} {diff:,.0f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        color = "#F47920" if not dca_wins else "#6B7280"
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:10px; padding:16px; text-align:center;">
                <h3 style="margin:0;color:{color}">💰 Cash Goblin</h3>
                <p style="font-size:2em;margin:8px 0;font-weight:bold">{currency} {dip_result.ending_wealth:,.0f}</p>
                <p style="margin:4px 0;color:#aaa">XIRR: {fmt_pct(dip_result.xirr)}</p>
                <p style="margin:4px 0;color:#aaa">Max DD: {fmt_pct(dip_result.max_drawdown)}</p>
                <p style="margin:4px 0;color:#aaa">In market: {fmt_pct(dip_result.time_in_market_pct)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def dip_badge(drawdown_pct: float, label: str, threshold: float) -> None:
    """Colored badge showing current dip status."""
    if drawdown_pct <= threshold:
        color = "#FF4B6B"
        bg = "rgba(255,75,107,0.1)"
        status = "DIP TRIGGERED"
        icon = "🔴"
    elif drawdown_pct <= threshold * 0.5:
        color = "#F47920"
        bg = "rgba(244,121,32,0.1)"
        status = "APPROACHING"
        icon = "🟠"
    else:
        color = "#00C896"
        bg = "rgba(0,200,150,0.1)"
        status = "IDLE"
        icon = "🟢"

    st.markdown(
        f"""
        <div style="border:1px solid {color}; border-radius:8px; padding:10px 14px;
                    background:{bg}; display:inline-block; margin:4px;">
            <span style="color:{color};font-weight:bold">{icon} {status}</span>
            <br><span style="color:#FAFAFA;font-size:1.1em">{label}</span>
            <br><span style="color:#aaa;font-size:0.85em">{drawdown_pct:.1%} from peak</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    source: str, as_of: object, is_total_return: bool, currency: str
) -> None:
    """Render a small caption with data provenance info."""
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
    """Render a styled page header with title and subtitle."""
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f"""
        <div style="margin-bottom:24px; padding-bottom:16px; border-bottom:1px solid #2D3047">
            <h1 style="font-size:2em; font-weight:800; color:#FAFAFA; margin:0">
                {prefix}{title}
            </h1>
            <p style="color:#9CA3AF; margin:6px 0 0 0; font-size:0.95em">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_size_badge(n: int) -> None:
    """Display a colored sample-size badge."""
    if n < 10:
        color = "#FF4B6B"
        label = f"n={n} — insufficient"
    elif n < 20:
        color = "#F47920"
        label = f"n={n} — small sample"
    else:
        color = "#00C896"
        label = f"n={n} — suggestive"

    st.markdown(
        f'<span style="background:{color}22; border:1px solid {color}; '
        f'border-radius:4px; padding:2px 8px; color:{color}; font-size:0.85em">'
        f"Sample size: {label}</span>",
        unsafe_allow_html=True,
    )
