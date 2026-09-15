"""Dip, DCA & Chill — main Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.quant.drawdown import drawdown, drawdown_episodes, drawdown_label  # noqa: E402
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.set_page_config(
    page_title="Dip, DCA & Chill",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Health check endpoint support
if "health" in st.query_params:
    st.json({"status": "ok", "version": "0.1.0"})
    st.stop()

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1A1D27 0%, #252840 50%, #1F2144 100%);
        border: 1px solid #2D3047;
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 24px;
        text-align: center;
    ">
        <h1 style="font-size:2.8em; font-weight:900; color:#FAFAFA; margin:0; letter-spacing:-0.02em">
            📉 Dip, DCA &amp; Chill
        </h1>
        <p style="font-size:1.2em; color:#9CA3AF; margin:12px 0 0 0">
            Monthly Machine versus Cash Goblin — settled by historical data, not vibes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Strategy explanation cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div style="background:#1E2130; border:1px solid #00C896; border-radius:12px; padding:20px; min-height:160px">
            <p style="font-size:1.5em; margin:0">🤖</p>
            <h3 style="color:#00C896; margin:8px 0 4px">Monthly Machine</h3>
            <p style="color:#9CA3AF; font-size:0.9em; margin:0">Invests every month, like clockwork, regardless of market mood. Boring. Reliable. Backtested.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div style="background:#1E2130; border:1px solid #F47920; border-radius:12px; padding:20px; min-height:160px">
            <p style="font-size:1.5em; margin:0">💰</p>
            <h3 style="color:#F47920; margin:8px 0 4px">Cash Goblin</h3>
            <p style="color:#9CA3AF; font-size:0.9em; margin:0">Hoards cash, waits for the perfect dip, deploys with maximum drama. Feels brilliant. Sometimes is.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div style="background:#1E2130; border:1px solid #FFD700; border-radius:12px; padding:20px; min-height:160px">
            <p style="font-size:1.5em; margin:0">🍽️</p>
            <h3 style="color:#FFD700; margin:8px 0 4px">Dip Buffet</h3>
            <p style="color:#9CA3AF; font-size:0.9em; margin:0">Tiered deployment at multiple drawdown depths. For the indecisive optimiser.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------------------
# Live drawdown status card (SPY)
# ---------------------------------------------------------------------------
from datetime import date  # noqa: E402

from dipdca.data.providers.yahoo import YahooProvider  # noqa: E402

with st.spinner("Fetching SPY market data..."):
    try:
        provider = YahooProvider()
        _end = date.today()
        _start = date(_end.year - 1, _end.month, _end.day)
        price_data = provider.get_price_data("SPY", _start, _end)
        adj = price_data.df["adj_close"].dropna()
        dd_series = drawdown(adj)
        current_dd = float(dd_series.iloc[-1])
        label = drawdown_label(current_dd)
        as_of = price_data.as_of
    except Exception as exc:
        st.error(f"Could not load market data. Check your internet connection. ({exc})")
        st.stop()

color = "#00C896" if current_dd > -0.05 else ("#F47920" if current_dd > -0.20 else "#FF4B6B")

st.markdown(
    f"""
    <div style="border:2px solid {color}; border-radius:12px; padding:20px; text-align:center;
                background:rgba(30,33,48,0.8); max-width:400px; margin:0 auto 24px auto">
        <p style="margin:0;color:#aaa;font-size:0.85em">Live · SPY (S&amp;P 500)</p>
        <h2 style="margin:8px 0;color:{color}">{fmt_pct(current_dd)}</h2>
        <p style="margin:0;color:{color};font-weight:bold">{label}</p>
        <p style="margin:4px 0;color:#6B7280;font-size:0.8em">from all-time high · as of {as_of}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fear & Greed widget
try:
    from dipdca.data.providers.fear_greed import (  # noqa: E402
        fetch_current_fear_greed,
        score_to_color,
    )

    fg = fetch_current_fear_greed()
    if fg:
        fg_color = score_to_color(fg.value)
        st.markdown(
            f"""
            <div style="background:#1E2130; border:1px solid {fg_color}; border-radius:12px;
                        padding:20px; text-align:center; margin:0 auto 16px auto; max-width:400px">
                <p style="color:#9CA3AF; font-size:0.8em; margin:0; text-transform:uppercase;
                           letter-spacing:0.1em">
                    CNN Fear &amp; Greed Index
                </p>
                <p style="font-size:3em; font-weight:900; color:{fg_color}; margin:8px 0; line-height:1">
                    {fg.value:.0f}
                </p>
                <p style="color:{fg_color}; font-weight:700; margin:0; font-size:1em">{fg.label}</p>
                <p style="color:#4B5563; font-size:0.75em; margin:4px 0 0">Live · {fg.as_of}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        fg_manual = st.sidebar.slider(
            "Fear & Greed (manual)",
            0,
            100,
            50,
            help="Enter the current CNN Fear & Greed Index reading manually if live fetch fails",
        )
except Exception:
    pass  # F&G widget is optional — never break the homepage

st.divider()

# ---------------------------------------------------------------------------
# Recent dip episodes (live data)
# ---------------------------------------------------------------------------
if len(dd_series) > 10:
    st.markdown(
        """
        <h3 style="color:#FAFAFA; font-weight:700; margin-bottom:12px">
            Recent Drawdown Episodes (≤ −5%)
        </h3>
        """,
        unsafe_allow_html=True,
    )
    episodes_df = drawdown_episodes(dd_series, threshold=-0.05)

    if episodes_df.empty:
        st.info("No dip episodes at −5% or deeper in the selected data range.")
    else:
        # Show last 5, most recent first
        display_df = episodes_df.tail(5).iloc[::-1].copy()
        display_df["trough_dd"] = display_df["trough_dd"].map(lambda x: f"{x:.1%}")
        display_df["duration_days"] = display_df["duration_days"].astype(int)
        display_df = display_df.rename(
            columns={
                "start": "Episode Start",
                "trough_date": "Trough Date",
                "end": "Recovery / End",
                "trough_dd": "Max Drawdown",
                "duration_days": "Duration (days)",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Navigation cards
# ---------------------------------------------------------------------------
st.markdown(
    '<h3 style="color:#FAFAFA; font-weight:700; margin-bottom:16px">Explore the Lab</h3>',
    unsafe_allow_html=True,
)

pages_info = [
    (
        "🕹️ Market Arcade",
        "Live index levels, drawdowns, and total-return comparison charts",
        "pages/1_Market_Arcade.py",
        "#2D3047",
    ),
    (
        "📉 Dip-O-Meter",
        "Is this historically a dip? Forward-return analysis, no crystal ball",
        "pages/2_Dip_O_Meter.py",
        "#2D3047",
    ),
    (
        "⚔️ Strategy Lab",
        "Monthly Machine vs Cash Goblin — head-to-head backtests",
        "pages/3_Strategy_Lab.py",
        "#F47920",
    ),
    (
        "🔬 Threshold Lab",
        "Which threshold wins historically? Rolling windows, heatmaps",
        "pages/4_Threshold_Lab.py",
        "#2D3047",
    ),
    (
        "🚪 Exit Lab",
        "When to sell? Deterministic exit rules vs never selling",
        "pages/5_Exit_Lab.py",
        "#2D3047",
    ),
    (
        "💱 Currency Check",
        "How much of your return was the asset vs the exchange rate?",
        "pages/6_Currency_Reality_Check.py",
        "#2D3047",
    ),
    (
        "🏦 Savings Rates",
        "What does your cash actually earn? Official ECB / SNB rates",
        "pages/7_Savings_Rates.py",
        "#2D3047",
    ),
    (
        "⚕️ Data Health",
        "Formulas, sources, freshness, cache status",
        "pages/8_Methodology_and_Data_Health.py",
        "#2D3047",
    ),
]

cols = st.columns(4)
for i, (title, desc, page, accent) in enumerate(pages_info):
    with cols[i % 4]:
        st.markdown(
            f"""
            <div style="background:#1E2130; border:1px solid {accent};
                        border-radius:10px; padding:14px; margin-bottom:12px; min-height:90px">
                <p style="margin:0; font-weight:700; color:#FAFAFA; font-size:0.95em">{title}</p>
                <p style="margin:4px 0 0 0; color:#9CA3AF; font-size:0.8em">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(page, label=f"Open {title.split(' ', 1)[-1]}")

st.divider()
st.caption(
    "**Disclaimer**: Educational and informational purposes only. Not investment advice. "
    "Past performance does not guarantee future results. See DISCLAIMER.md."
)
