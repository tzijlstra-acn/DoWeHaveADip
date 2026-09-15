"""Today — current drawdown status and historical context."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.config import load_assets_config  # noqa: E402
from dipdca.data.errors import LiveDataUnavailable  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from dipdca.quant.drawdown import drawdown, drawdown_episodes  # noqa: E402
from dipdca.quant.monte_carlo import conditional_path_bootstrap  # noqa: E402
from ui.charts import drawdown_chart, plot_fan_chart  # noqa: E402
from ui.components import (  # noqa: E402
    freshness_caption,
    live_data_error,
    page_header,
    signal_card,
)
from ui.copy import DISCLAIMER_SHORT  # noqa: E402
from ui.design_tokens import (  # noqa: E402
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    WARNING,
)
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "DO WE HAVE A DIP?",
    "Market pressure, measured. Current drawdown status and what history says about similar periods.",
)

# ---------------------------------------------------------------------------
# Inputs — all in main column (no sidebar)
# ---------------------------------------------------------------------------
_assets_cfg = load_assets_config()
# Map display_name → full asset config dict
_asset_map = {a["display_name"]: a for a in _assets_cfg}
_asset_names = list(_asset_map.keys())
_default_idx = next((i for i, n in enumerate(_asset_names) if "S&P 500" in n), 0)

col_asset, col_monthly, col_cash = st.columns([3, 2, 2])
with col_asset:
    selected_name = st.selectbox("Market or ETF", _asset_names, index=_default_idx)
with col_monthly:
    monthly_contribution = st.number_input(
        "Monthly investment (EUR)", min_value=0, max_value=100_000, value=500, step=50
    )
with col_cash:
    cash_available = st.number_input(
        "Cash currently waiting (EUR)", min_value=0, max_value=1_000_000, value=0, step=100
    )

with st.expander("Advanced assumptions"):
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        history_start = st.date_input("History start", value=date(2005, 1, 1))
    with col_a2:
        dip_threshold = st.slider("Dip threshold (%)", min_value=-50, max_value=-1, value=-10, step=1) / 100.0

asset_cfg = _asset_map[selected_name]
etf_symbol = asset_cfg["etf_symbol"]
index_symbol = asset_cfg.get("index_symbol")
history_end = date.today()

# ---------------------------------------------------------------------------
# Load market data — ETF for valuation, benchmark index for ATH/drawdown signal
# ---------------------------------------------------------------------------
svc = get_market_data_service()

with st.spinner(f"Loading {selected_name} data..."):
    try:
        # Always load the investable instrument (ETF)
        _inst_result = svc.get_history(etf_symbol, history_start, history_end)
        price_df = _inst_result.frame
        price_series = price_df["adj_close"].dropna()

        # Load benchmark index for ATH / drawdown display when available
        if index_symbol is not None:
            _bm_result = svc.get_history(index_symbol, history_start, history_end)
            bm_series = _bm_result.frame["adj_close"].dropna()
            signal_series = bm_series
            signal_label = index_symbol
            freshness_caption(_bm_result.freshness)
        else:
            signal_series = price_series
            signal_label = f"{etf_symbol} (ETF proxy — no index configured)"
            freshness_caption(_inst_result.freshness)
            st.caption(
                f"No benchmark index configured for {selected_name}. "
                "Using ETF price as a proxy for the ATH/drawdown signal. "
                "Results may differ from the underlying index."
            )

    except LiveDataUnavailable as exc:
        live_data_error(exc, context=selected_name)
        st.stop()

if len(signal_series) < 30:
    st.error("Not enough data for analysis. Try an earlier history start date.")
    st.stop()

dd_series = drawdown(signal_series)
current_dd = float(dd_series.iloc[-1])

# Benchmark ATH date
ath_date = signal_series.idxmax()

# Days since last all-time high (from the benchmark/index series)
peak_series = signal_series.expanding().max()
at_peak = signal_series >= peak_series
days_since_high = 0
for i in range(len(at_peak) - 1, -1, -1):
    if at_peak.iloc[i]:
        break
    days_since_high += 1

# Historical dip frequency at this level (±5%)
tol = 0.05
similar_mask = (dd_series <= current_dd + tol) & (dd_series >= current_dd - tol)
dip_freq = float(similar_mask.mean())

# Average recovery time from similar levels
episodes_df = drawdown_episodes(dd_series, threshold=min(current_dd - tol, -0.05))
avg_recovery_days = (
    float(episodes_df["duration_days"].mean()) if not episodes_df.empty else None
)

# ---------------------------------------------------------------------------
# Hero result
# ---------------------------------------------------------------------------
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if abs(current_dd) < 0.001:
    hero_color = POSITIVE
    direction_text = "at its previous high"
    dd_display = "0.0%"
elif current_dd <= dip_threshold:
    hero_color = NEGATIVE
    direction_text = f"below its previous high — dip threshold reached ({dip_threshold:.0%})"
    dd_display = fmt_pct(current_dd)
elif current_dd <= dip_threshold * 0.5:
    hero_color = WARNING
    direction_text = "below its previous high — approaching threshold"
    dd_display = fmt_pct(current_dd)
else:
    hero_color = NEUTRAL
    direction_text = "below its previous high"
    dd_display = fmt_pct(current_dd)

if abs(current_dd) < 0.001:
    status_label = "AT THE HIGH"
elif current_dd <= dip_threshold:
    status_label = "THRESHOLD HIT" if current_dd > dip_threshold * 1.5 else "DEEP DRAWDOWN"
elif current_dd <= dip_threshold * 0.5:
    status_label = "APPROACHING"
else:
    status_label = "NO SIGNAL YET"

distance = current_dd - dip_threshold  # positive = still above threshold

signal_card(
    status_label=status_label,
    drawdown_pct=current_dd,
    threshold_pct=dip_threshold,
    days_since_high=days_since_high,
    distance_to_threshold_pct=distance,
    status_color=hero_color,
)

# Benchmark source attribution
ath_date_str = str(ath_date.date()) if hasattr(ath_date, "date") else str(ath_date)
st.caption(
    f"Signal source: **{signal_label}** — "
    f"All-time high: **{signal_series.max():,.2f}** on **{ath_date_str}**"
)

# ---------------------------------------------------------------------------
# 3 secondary metrics
# ---------------------------------------------------------------------------
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Calendar days from high", f"{days_since_high:,}")
with m2:
    st.metric(
        "Historical frequency at this level",
        f"{dip_freq:.1%}",
        help="Fraction of trading days in history where the drawdown was within 5% of today's level",
    )
with m3:
    if avg_recovery_days is not None:
        st.metric(
            "Avg recovery time from similar dips",
            f"{avg_recovery_days:.0f} days",
            help="Average duration of historical dip episodes at or below this level",
        )
    else:
        st.metric("Avg recovery time", "No similar episodes")

st.divider()

# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------
st.subheader("Drawdown history")
# Use benchmark/index series for the drawdown chart (the true signal source)
fig_dd = drawdown_chart(signal_series, title=f"{selected_name} — Drawdown from High ({signal_label})")
st.plotly_chart(fig_dd, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Historical context — fan chart (conditional path bootstrap)
# ---------------------------------------------------------------------------
st.subheader("What happened next in similar periods?")
st.caption(
    f"Historical periods where {selected_name} was within 5% of today's level "
    f"({current_dd:.1%}). Shows the distribution of 12-month outcomes. "
    "This is historical data — not a prediction."
)

with st.spinner("Finding similar historical periods..."):
    try:
        monthly_prices = price_series.resample("ME").last().dropna()
        sims = conditional_path_bootstrap(
            prices=monthly_prices,
            current_drawdown=current_dd,
            deploy_pcts=[0.0, 0.5, 1.0] if cash_available > 0 else [0.0],
            monthly_contribution=float(monthly_contribution),
            cash_accumulated=float(cash_available),
            horizon_months=12,
            n_simulations=200,
            seed=42,
        )
    except Exception:
        sims = []

if not sims:
    st.info(
        "Not enough similar historical periods found for a distribution chart. "
        "Try a wider date range or a different asset."
    )
else:
    deploy_labels = {0.0: "Invest monthly only", 0.5: "Deploy 50% of cash now", 1.0: "Deploy all cash now"}
    for sim in sims:
        label_key = sim.deploy_pct
        chart_label = deploy_labels.get(label_key, f"Deploy {sim.deploy_pct:.0%} of cash")
        n_periods = len([d for d in dd_series.index if abs(float(dd_series.loc[d]) - current_dd) <= 0.05])
        st.caption(f"Based on {n_periods} historical periods — {chart_label}")
        fig_fan = plot_fan_chart([sim], currency="EUR", horizon_months=12)
        st.plotly_chart(fig_fan, width="stretch")

# ---------------------------------------------------------------------------
# CTA buttons
# ---------------------------------------------------------------------------
st.divider()
st.markdown("**Want to go deeper?**")
cta1, cta2, cta3 = st.columns(3)
with cta1:
    st.page_link("app_pages/compare.py", label="Compare all strategies", icon=":material/compare_arrows:")
with cta2:
    st.page_link("app_pages/scenarios.py", label="Full historical scenarios", icon=":material/history:")
with cta3:
    st.page_link("app_pages/about.py", label="How it works", icon=":material/info:")

st.divider()
st.caption(DISCLAIMER_SHORT)
