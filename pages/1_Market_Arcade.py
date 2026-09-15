"""Market Arcade — browse all assets and their current drawdown status."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.config import load_assets_config  # noqa: E402
from dipdca.quant.drawdown import drawdown, drawdown_episodes, drawdown_label  # noqa: E402
from dipdca.quant.episodes import load_named_episodes  # noqa: E402
from ui.charts import add_dip_highlights, add_named_episode_labels, total_return_chart  # noqa: E402
from ui.components import data_source_caption, page_header  # noqa: E402
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.set_page_config(page_title="Market Arcade", page_icon="🕹️", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("Market Arcade", "Browse assets and see their current drawdown status.", "🕹️")


def load_live_prices(assets: list[dict]) -> dict[str, pd.DataFrame]:
    """Load live Yahoo Finance data for all assets."""
    from datetime import date

    from dipdca.data.providers.yahoo import YahooProvider

    provider = YahooProvider()
    end = date.today()
    start = date(end.year - 3, end.month, end.day)

    live_data = {}
    for asset in assets:
        symbol = asset["etf_symbol"]
        try:
            price_data = provider.get_price_data(symbol, start, end)
            live_data[asset["display_name"]] = price_data.df
        except Exception:
            pass  # Skip unavailable assets; warn below
    return live_data


# Load data
assets = load_assets_config()

with st.spinner("Fetching market data from Yahoo Finance..."):
    price_frames = load_live_prices(assets)

if not price_frames:
    st.error("Could not load any market data. Check your internet connection.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: dip threshold for highlights
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
highlight_threshold = (
    st.sidebar.slider("Dip highlight threshold (%)", min_value=-40, max_value=-1, value=-10, step=1)
    / 100.0
)

# ---------------------------------------------------------------------------
# Current Drawdown Status Banner
# ---------------------------------------------------------------------------
st.markdown(
    '<h3 style="color:#FAFAFA; font-weight:700; margin-bottom:4px">Current Drawdown Status</h3>',
    unsafe_allow_html=True,
)

cols = st.columns(min(len(price_frames), 3))
for i, (name, df) in enumerate(price_frames.items()):
    col = cols[i % 3]
    with col:
        adj = df["adj_close"].dropna()
        if len(adj) == 0:
            continue
        dd_series = drawdown(adj)
        current_dd = float(dd_series.iloc[-1])
        label = drawdown_label(current_dd)
        last_price = float(adj.iloc[-1])
        days_since_high = 0
        peak_series = adj.expanding().max()
        for j in range(len(adj) - 1, -1, -1):
            if adj.iloc[j] >= peak_series.iloc[j]:
                break
            days_since_high += 1

        color = "#00C896" if current_dd > -0.05 else ("#F47920" if current_dd > -0.20 else "#FF4B6B")
        triggered = current_dd <= highlight_threshold
        trigger_str = f"🔴 DIP TRIGGERED ({highlight_threshold:.0%})" if triggered else "🟢 No trigger"
        trigger_color = "#FF4B6B" if triggered else "#00C896"

        st.markdown(
            f"""
            <div style="background:#1E2130; border:2px solid {color}; border-radius:12px;
                        padding:16px; margin-bottom:10px">
                <h4 style="margin:0; color:#FAFAFA">{name}</h4>
                <p style="margin:6px 0; font-size:1.8em; color:{color}; font-weight:800; line-height:1">{fmt_pct(current_dd)}</p>
                <p style="margin:0; color:#9CA3AF; font-size:0.85em">{label}</p>
                <p style="margin:4px 0; color:#6B7280; font-size:0.8em">Last: {last_price:.2f} | {days_since_high}d from ATH</p>
                <span style="background:{trigger_color}22; border:1px solid {trigger_color};
                             border-radius:4px; padding:2px 8px; color:{trigger_color};
                             font-size:0.8em; font-weight:600">{trigger_str}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Total Return Chart with dip highlights
# ---------------------------------------------------------------------------
st.markdown(
    '<h3 style="color:#FAFAFA; font-weight:700; margin:20px 0 8px 0">Indexed Total Return (base = 100)</h3>',
    unsafe_allow_html=True,
)

series_dict = {}
for name, df in price_frames.items():
    adj = df["adj_close"].dropna()
    if len(adj) > 0:
        series_dict[name] = adj

if series_dict:
    fig = total_return_chart(
        series_dict,
        title="Indexed Total Return (base = 100)",
        currency="EUR/USD",
        data_type="Total Return (Adj.)",
    )

    # Add dip highlights using the first available series' drawdown
    first_name = list(series_dict.keys())[0]
    first_series = series_dict[first_name].dropna()
    if len(first_series) > 5:
        dd_for_highlights = drawdown(first_series)
        add_dip_highlights(fig, dd_for_highlights, threshold=highlight_threshold)

    # Add named historical episode overlays
    try:
        named_eps = load_named_episodes()
        start_ts = first_series.index[0] if len(first_series) > 0 else None
        end_ts = first_series.index[-1] if len(first_series) > 0 else None
        add_named_episode_labels(fig, named_eps, date_start=start_ts, date_end=end_ts)
    except Exception:
        pass  # Named episodes are optional — don't break the chart

    st.plotly_chart(fig, use_container_width=True)

    as_of = list(price_frames.values())[0].index[-1].date()
    data_source_caption(
        source="Yahoo Finance",
        as_of=as_of,
        is_total_return=True,
        currency="various",
    )

    # Historical episodes catalog expander
    try:
        named_eps_catalog = load_named_episodes()
        with st.expander("Historical Named Episodes Catalog"):
            category_color_map = {
                "bubble": "#9B59B6",
                "financial": "#FF4B6B",
                "macro": "#F47920",
                "geopolitical": "#4C9BE8",
            }
            for ep in named_eps_catalog:
                cat = ep.get("category", "macro")
                color = category_color_map.get(cat, "#9CA3AF")
                st.markdown(
                    f"""
                    <div style="border-left:3px solid {color}; padding:8px 12px; margin-bottom:8px;
                                background:#1E2130; border-radius:0 6px 6px 0">
                        <b style="color:{color}">{ep['label']}</b>
                        <span style="color:#6B7280; font-size:0.85em; margin-left:8px">
                            {ep['start']} → {ep['end']}
                        </span>
                        <span style="color:#FF4B6B; font-weight:700; margin-left:8px">
                            {ep['max_drawdown']:.0%}
                        </span>
                        <br>
                        <span style="color:#9CA3AF; font-size:0.85em">{ep['note']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Recent Episodes table — styled with HTML rows
# ---------------------------------------------------------------------------
st.markdown(
    '<h3 style="color:#FAFAFA; font-weight:700; margin:20px 0 8px 0">Recent Drawdown Episodes</h3>',
    unsafe_allow_html=True,
)

first_df = list(price_frames.values())[0]
first_adj = first_df["adj_close"].dropna()
if len(first_adj) > 10:
    dd_eps = drawdown(first_adj)
    episodes_df = drawdown_episodes(dd_eps, threshold=highlight_threshold)

    if episodes_df.empty:
        st.info(f"No episodes at {fmt_pct(highlight_threshold)} or deeper in the history shown.")
    else:
        # Show last 5, most recent first
        recent = episodes_df.tail(5).iloc[::-1].copy()

        # Mark most recent episode (open episode = end == last date)
        last_date = dd_eps.index[-1]

        display_eps = recent.copy()
        display_eps["trough_dd"] = display_eps["trough_dd"].map(lambda x: f"{x:.1%}")
        display_eps["duration_days"] = display_eps["duration_days"].astype(int)
        display_eps["start"] = display_eps["start"].dt.date
        display_eps["trough_date"] = display_eps["trough_date"].dt.date
        display_eps["end"] = display_eps["end"].dt.date
        display_eps.columns = ["Start", "Trough Date", "Recovery / End", "Max Drawdown", "Duration (days)"]
        st.dataframe(display_eps, use_container_width=True, hide_index=True)

        # Most recent episode callout
        last_ep = episodes_df.iloc[-1]
        last_dd = last_ep["trough_dd"]
        last_dur = last_ep["duration_days"]
        is_ongoing = last_ep["end"] == last_date
        color = "#FF4B6B" if is_ongoing else "#00C896"
        status = "🔴 ONGOING" if is_ongoing else "✓ Recovered"
        st.markdown(
            f"""
            <div style="background:#1E2130; border:1px solid {color}; border-radius:10px; padding:14px;
                        margin-top:8px">
                <span style="color:{color}; font-weight:700; font-size:0.95em">{status} — Most Recent Episode</span><br>
                <span style="color:#FAFAFA">
                    Start: {last_ep["start"].date()} |
                    Max drawdown: <b>{last_dd:.1%}</b> |
                    Duration: {last_dur} days
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Asset Universe table
# ---------------------------------------------------------------------------
st.markdown(
    '<h3 style="color:#FAFAFA; font-weight:700; margin:20px 0 8px 0">Asset Universe</h3>',
    unsafe_allow_html=True,
)
asset_rows = [
    {
        "ID": a["id"],
        "Name": a["display_name"],
        "Index": a.get("index_symbol") or "—",
        "ETF": a["etf_symbol"],
        "Currency": a["quote_currency"],
        "Hedged": "Yes" if a.get("is_hedged") else "No",
        "Notes": a.get("notes", ""),
    }
    for a in assets
]
st.dataframe(pd.DataFrame(asset_rows), use_container_width=True, hide_index=True)
