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
from ui.components import freshness_caption, live_data_error  # noqa: E402
from ui.copy import DISCLAIMER_SHORT, drawdown_label  # noqa: E402
from ui.design_tokens import NEGATIVE, NEUTRAL, POSITIVE, WARNING  # noqa: E402
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inputs — all in main column (no sidebar)
# ---------------------------------------------------------------------------
_assets_cfg = load_assets_config()
_asset_options = {a["display_name"]: a["etf_symbol"] for a in _assets_cfg}
_asset_names = list(_asset_options.keys())
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

symbol = _asset_options[selected_name]
history_end = date.today()

# ---------------------------------------------------------------------------
# Load market data
# ---------------------------------------------------------------------------
with st.spinner(f"Loading {selected_name} data..."):
    try:
        _result = get_market_data_service().get_history(symbol, history_start, history_end)
        price_df = _result.frame
        price_series = price_df["adj_close"].dropna()
    except LiveDataUnavailable as exc:
        live_data_error(exc, context=selected_name)
        st.stop()

freshness_caption(_result.freshness)

if len(price_series) < 30:
    st.error("Not enough data for analysis. Try an earlier history start date.")
    st.stop()

dd_series = drawdown(price_series)
current_dd = float(dd_series.iloc[-1])

# Days since last all-time high
peak_series = price_series.expanding().max()
at_peak = price_series >= peak_series
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

label = drawdown_label(current_dd)

st.markdown(
    f"""
    <div style="background:#FFFFFF; border:1px solid {hero_color};
                border-radius:12px; padding:28px 32px; margin:16px 0 24px 0;">
        <p style="margin:0; color:#6B7280; font-size:0.9em">{selected_name} ({symbol})</p>
        <h2 style="margin:8px 0; font-size:2.2em; font-weight:800; color:{hero_color}">
            {dd_display}
        </h2>
        <p style="margin:0; color:#374151; font-size:1.05em">
            {direction_text}
        </p>
        <p style="margin:6px 0 0; color:#9CA3AF; font-size:0.85em">{label}</p>
    </div>
    """,
    unsafe_allow_html=True,
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
fig_dd = drawdown_chart(price_series, title=f"{selected_name} — Drawdown from High")
st.plotly_chart(fig_dd, use_container_width=True)

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
        fig_fan = plot_fan_chart(sim, currency="EUR", horizon_months=12)
        st.plotly_chart(fig_fan, use_container_width=True)

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
