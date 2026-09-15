"""Market overview — browse all assets and their current drawdown status."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


from dipdca.config import load_assets_config  # noqa: E402
from dipdca.data.errors import LiveDataUnavailable  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from dipdca.quant.drawdown import drawdown, drawdown_episodes, drawdown_label  # noqa: E402
from dipdca.quant.episodes import load_named_episodes  # noqa: E402
from ui.charts import add_dip_highlights, add_named_episode_labels, total_return_chart  # noqa: E402
from ui.components import data_source_caption, page_header  # noqa: E402
from ui.design_tokens import (  # noqa: E402
    ACCENT_CYAN,
    BG_SURFACE_RAISED,
    NEGATIVE,
    POSITIVE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
)
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("Market overview", "Browse assets and see their current drawdown status.")


def load_live_prices(assets: list[dict]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load live market data for all assets. Returns (data_dict, error_list)."""
    from datetime import date

    svc = get_market_data_service()
    end = date.today()
    start = (end - pd.DateOffset(years=3)).date()  # leap-day safe

    live_data: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for asset in assets:
        symbol = asset["etf_symbol"]
        name = asset["display_name"]
        try:
            result = svc.get_history(symbol, start, end)
            live_data[name] = result.frame
        except LiveDataUnavailable as exc:
            errors.append(f"{name} ({symbol}): {exc}")
    return live_data, errors


# Load data
assets = load_assets_config()

with st.spinner("Fetching market data from Yahoo Finance..."):
    price_frames, load_errors = load_live_prices(assets)

if load_errors:
    with st.expander(f"{len(load_errors)} asset(s) could not be loaded"):
        for err in load_errors:
            st.caption(f"- {err}")

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
st.subheader("Current Drawdown Status")

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

        color = POSITIVE if current_dd > -0.05 else (WARNING if current_dd > -0.20 else NEGATIVE)
        triggered = current_dd <= highlight_threshold
        trigger_str = f"DIP TRIGGERED ({highlight_threshold:.0%})" if triggered else "No trigger"
        trigger_color = NEGATIVE if triggered else POSITIVE

        st.markdown(
            f"""
            <div style="background:{BG_SURFACE_RAISED}; border:2px solid {color}; border-radius:12px;
                        padding:16px; margin-bottom:10px">
                <h4 style="margin:0; color:{TEXT_PRIMARY}">{name}</h4>
                <p style="margin:6px 0; font-size:1.8em; color:{color}; font-weight:800; line-height:1;
                          font-variant-numeric:tabular-nums">{fmt_pct(current_dd)}</p>
                <p style="margin:0; color:{TEXT_MUTED}; font-size:0.85em">{label}</p>
                <p style="margin:4px 0; color:{TEXT_SECONDARY}; font-size:0.8em">Last: {last_price:.2f} | {days_since_high} calendar days from high</p>
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
st.subheader("Indexed Total Return (base = 100)")

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

    st.plotly_chart(fig, width="stretch")

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
                "bubble": "#A855F7",   # purple — no old arcade alias
                "financial": NEGATIVE,
                "macro": WARNING,
                "geopolitical": ACCENT_CYAN,
            }
            for ep in named_eps_catalog:
                cat = ep.get("category", "macro")
                color = category_color_map.get(cat, TEXT_MUTED)
                st.markdown(
                    f"""
                    <div style="border-left:3px solid {color}; padding:8px 12px; margin-bottom:8px;
                                background:{BG_SURFACE_RAISED}; border-radius:0 6px 6px 0">
                        <b style="color:{color}">{ep['label']}</b>
                        <span style="color:{TEXT_SECONDARY}; font-size:0.85em; margin-left:8px">
                            {ep['start']} → {ep['end']}
                        </span>
                        <span style="color:{NEGATIVE}; font-weight:700; margin-left:8px">
                            {ep['max_drawdown']:.0%}
                        </span>
                        <br>
                        <span style="color:{TEXT_MUTED}; font-size:0.85em">{ep['note']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Recent Episodes table — styled with HTML rows
# ---------------------------------------------------------------------------
st.subheader("Recent Drawdown Episodes")

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
        st.dataframe(display_eps, width="stretch", hide_index=True)

        # Most recent episode callout
        last_ep = episodes_df.iloc[-1]
        last_dd = last_ep["trough_dd"]
        last_dur = last_ep["duration_days"]
        is_ongoing = last_ep["end"] == last_date
        color = NEGATIVE if is_ongoing else POSITIVE
        status = "ONGOING" if is_ongoing else "Recovered"
        st.markdown(
            f"""
            <div style="background:{BG_SURFACE_RAISED}; border:1px solid {color}; border-radius:10px;
                        padding:14px; margin-top:8px">
                <span style="color:{color}; font-weight:700; font-size:0.95em">{status} — Most Recent Episode</span><br>
                <span style="color:{TEXT_PRIMARY}">
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
st.subheader("Asset Universe")
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
st.dataframe(pd.DataFrame(asset_rows), width="stretch", hide_index=True)
