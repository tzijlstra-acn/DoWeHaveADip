"""Dip-O-Meter — is this historically a dip?"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


from dipdca.quant.drawdown import drawdown, drawdown_episodes, drawdown_label  # noqa: E402
from dipdca.quant.monte_carlo import PathSimulation, conditional_path_bootstrap  # noqa: E402
from ui.charts import plot_drawdown_gauge, plot_fan_chart, plot_forward_returns_box  # noqa: E402
from ui.components import (  # noqa: E402
    data_source_caption,
    page_header,
    sample_size_badge,
)
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS, GREEN, ORANGE, RED  # noqa: E402

st.set_page_config(page_title="Dip-O-Meter | Dip, DCA & Chill", page_icon="📉", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "Dip-O-Meter",
    "Is this historically a dip? Let the data decide. No crystal ball provided.",
    "📉",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

threshold = (
    st.sidebar.slider("Dip threshold (%)", min_value=-50, max_value=-1, value=-10, step=1)
    / 100.0
)

symbol = st.sidebar.text_input("Ticker", value="SPY")

start_default = datetime.date(2005, 1, 1)
end_default = datetime.date.today()
start_date = st.sidebar.date_input("History start", value=start_default)
end_date = st.sidebar.date_input("History end", value=end_default)

horizons = [63, 126, 252, 504, 756]  # ~3m, 6m, 1y, 2y, 3y
horizon_labels = {63: "3 months", 126: "6 months", 252: "1 year", 504: "2 years", 756: "3 years"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
from dipdca.data.errors import LiveDataUnavailable  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from ui.components import freshness_caption, live_data_error  # noqa: E402

with st.spinner(f"Fetching market data for {symbol}..."):
    try:
        _result = get_market_data_service().get_history(symbol, start_date, end_date)
        price_df = _result.frame
        data_source = f"Yahoo Finance ({symbol})"
        as_of_date = _result.freshness.observed_at.date()
        freshness_caption(_result.freshness)
    except LiveDataUnavailable as exc:
        live_data_error(exc, context=symbol)
        st.stop()

price_series = price_df["adj_close"].dropna()
price_series = price_series.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

if len(price_series) < 30:
    st.error("Not enough data for analysis. Try a wider date range.")
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

# Rolling volatility
ret = price_series.pct_change().dropna()
vol_20d = float(ret.tail(20).std() * (252**0.5)) if len(ret) >= 20 else None
vol_60d = float(ret.tail(60).std() * (252**0.5)) if len(ret) >= 60 else None

# ---------------------------------------------------------------------------
# Current status — gauge prominent at full width
# ---------------------------------------------------------------------------
st.markdown(
    '<h3 style="color:#FAFAFA; font-weight:700; margin-bottom:8px">Current Status</h3>',
    unsafe_allow_html=True,
)

label = drawdown_label(current_dd)
triggered = current_dd <= threshold
trigger_color = RED if triggered else GREEN
trigger_text = "RULE TRIGGERED" if triggered else "Rule not triggered"
trigger_icon = "🔴" if triggered else "🟢"

# Full-width gauge
fig_gauge = plot_drawdown_gauge(current_dd, threshold)
fig_gauge.update_layout(height=320)
st.plotly_chart(fig_gauge, use_container_width=True)

# Big dip label badge below gauge
badge_bg = "rgba(255,75,107,0.15)" if triggered else "rgba(0,200,150,0.1)"
st.markdown(
    f"""
    <div style="text-align:center; margin: -8px 0 20px 0">
        <span style="font-size:1.6em; font-weight:800; color:{trigger_color}">
            {trigger_icon} {label.upper()}
        </span>
        <br>
        <span style="display:inline-block; background:{badge_bg}; border:1px solid {trigger_color};
                     border-radius:20px; padding:4px 16px; color:{trigger_color};
                     font-weight:600; font-size:0.9em; margin-top:6px">
            {trigger_text} · threshold {fmt_pct(threshold)}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metrics row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "Current Drawdown",
        fmt_pct(current_dd),
        help="Distance from the running all-time high.",
    )
with c2:
    st.metric("Days Since ATH", str(days_since_high), help="Calendar days since the last all-time high.")
with c3:
    st.metric(
        "20d Realised Vol",
        f"{vol_20d:.1%}" if vol_20d else "N/A",
        help="Annualised 20-day realised volatility.",
    )
with c4:
    st.metric(
        "60d Realised Vol",
        f"{vol_60d:.1%}" if vol_60d else "N/A",
        help="Annualised 60-day realised volatility.",
    )

# ---------------------------------------------------------------------------
# Historical episode analysis
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    '<h3 style="color:#FAFAFA; font-weight:700; margin-bottom:4px">Historical Episode Analysis</h3>',
    unsafe_allow_html=True,
)
st.caption(
    "Comparable episodes = past periods where drawdown crossed the selected threshold. "
    "Forward returns are computed from the episode start date. "
    "**No crystal ball: this describes history, not tomorrow.**"
)

episodes = drawdown_episodes(dd_series, threshold=threshold)
n_episodes = len(episodes)

col_ep1, col_ep2 = st.columns([1, 2])

with col_ep1:
    sample_size_badge(n_episodes)
    st.write("")

    if episodes.empty:
        st.info(f"No episodes at {fmt_pct(threshold)} threshold in this history.")
    else:
        # Episode recovery stats
        avg_duration = episodes["duration_days"].mean()
        avg_trough = episodes["trough_dd"].mean()
        deepest = episodes["trough_dd"].min()

        st.metric("Episodes found", str(n_episodes))
        st.metric("Avg duration", f"{avg_duration:.0f} days")
        st.metric("Avg max drawdown", fmt_pct(avg_trough))
        st.metric("Deepest ever", fmt_pct(deepest))

with col_ep2:
    if not episodes.empty:
        # Compute forward returns for each episode
        fwd_records = []
        for _, ep in episodes.iterrows():
            rec: dict = {
                "start": ep["start"],
                "trough_date": ep["trough_date"],
                "trough_dd": ep["trough_dd"],
                "duration_days": ep["duration_days"],
            }
            entry_date = ep["start"]
            for h in horizons:
                future_date = entry_date + pd.Timedelta(days=h)
                window = price_series.loc[entry_date:future_date]
                if len(window) >= 2:
                    fwd_ret = float(window.iloc[-1] / window.iloc[0] - 1)
                else:
                    fwd_ret = np.nan
                rec[f"fwd_{horizon_labels[h].replace(' ', '_')}"] = fwd_ret
            fwd_records.append(rec)

        fwd_df = pd.DataFrame(fwd_records)
        fig_fwd = plot_forward_returns_box(fwd_df)
        st.plotly_chart(fig_fwd, use_container_width=True)

# ---------------------------------------------------------------------------
# Probability of beating cash
# ---------------------------------------------------------------------------
if not episodes.empty and "fwd_1_year" in fwd_df.columns:
    st.divider()
    st.markdown(
        '<h3 style="color:#FAFAFA; font-weight:700; margin-bottom:8px">Probability of Beating Cash (1-year horizon)</h3>',
        unsafe_allow_html=True,
    )

    fwd_1y = fwd_df["fwd_1_year"].dropna()
    cash_rate = st.slider(
        "Cash rate to beat (%/year)", min_value=0.0, max_value=10.0, value=3.5, step=0.25
    ) / 100.0

    if len(fwd_1y) > 0:
        p_beat = float((fwd_1y > cash_rate).mean())
        n_obs = len(fwd_1y)
        sample_size_badge(n_obs)

        col_prob, col_ci = st.columns(2)
        with col_prob:
            color = GREEN if p_beat > 0.6 else (ORANGE if p_beat > 0.4 else RED)
            st.markdown(
                f"""
                <div style="background:#1E2130; border:2px solid {color}; border-radius:12px;
                            padding:20px; text-align:center; margin-top:8px">
                    <p style="margin:0;color:#9CA3AF;font-size:0.85em">Historically beat cash ({cash_rate:.1%}/yr)</p>
                    <h2 style="margin:8px 0;color:{color}; font-size:2.5em; font-weight:800">{p_beat:.0%}</h2>
                    <p style="margin:0;color:#6B7280;font-size:0.8em">of 1-year windows after dip entry</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_ci:
            # Simple Wilson CI
            if n_obs >= 5:
                from scipy import stats as scipy_stats

                ci = scipy_stats.proportion_confint(
                    round(p_beat * n_obs), n_obs, alpha=0.1, method="wilson"
                )
                st.metric("90% CI lower", f"{ci[0]:.0%}")
                st.metric("90% CI upper", f"{ci[1]:.0%}")
                st.caption(
                    "Wilson confidence interval. Wide CI = small sample, interpret cautiously."
                )
            else:
                st.info("Too few episodes for a confidence interval.")

# ---------------------------------------------------------------------------
# Episodes table
# ---------------------------------------------------------------------------
if not episodes.empty:
    with st.expander("All Episodes Table"):
        display = episodes.copy()
        display["trough_dd"] = display["trough_dd"].map(lambda x: f"{x:.1%}")
        display["duration_days"] = display["duration_days"].astype(int)
        display["start"] = display["start"].dt.date
        display["trough_date"] = display["trough_date"].dt.date
        display["end"] = display["end"].dt.date
        display.columns = ["Start", "Trough Date", "Recovery / End", "Max Drawdown", "Duration (days)"]
        st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Conditional path simulation — "What happens if you deploy now?"
# ---------------------------------------------------------------------------
if not episodes.empty:
    st.divider()
    with st.expander("What happens if you deploy now?", expanded=False):
        st.markdown(
            f"""
            <div style="background:#1E2130; border:1px solid #2D3047; border-radius:10px;
                        padding:12px; margin-bottom:12px">
                <b style="color:#F47920">Current drawdown: {current_dd:.1%}</b>
                <span style="color:#9CA3AF"> — Bootstrap question:
                given we're at this drawdown level, what have historical continuation paths
                looked like depending on how much cash you deploy now?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_cp1, col_cp2, col_cp3 = st.columns(3)
        with col_cp1:
            cp_monthly = st.number_input(
                "Monthly contribution (EUR)",
                min_value=100.0,
                max_value=50_000.0,
                value=1000.0,
                step=100.0,
                key="dip_cp_monthly",
            )
        with col_cp2:
            cp_cash = st.number_input(
                "Accumulated cash (EUR)",
                min_value=100.0,
                max_value=500_000.0,
                value=cp_monthly * 12,
                step=500.0,
                key="dip_cp_cash",
            )
        with col_cp3:
            cp_horizon = st.slider(
                "Horizon (months)",
                min_value=6,
                max_value=36,
                value=24,
                step=6,
                key="dip_cp_horizon",
            )

        cp_deploy_opts = st.multiselect(
            "Deploy fractions to compare",
            options=["25%", "50%", "75%", "100%"],
            default=["25%", "50%", "75%", "100%"],
            key="dip_cp_deploys",
        )
        cp_deploy_pcts = (
            [float(d.replace("%", "")) / 100.0 for d in cp_deploy_opts]
            or [0.25, 0.50, 0.75, 1.00]
        )

        _cp_fp = hashlib.sha256(
            json.dumps(
                {
                    "symbol": symbol, "start": str(start_date), "end": str(end_date),
                    "threshold": threshold, "monthly": cp_monthly, "cash": round(cp_cash),
                    "horizon": cp_horizon, "deploys": sorted(cp_deploy_pcts),
                    "current_dd": round(current_dd, 4),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]

        if st.button("Run conditional path bootstrap", key="dip_run_cp"):
            with st.spinner(
                f"Bootstrapping paths from {current_dd:.0%} historical entries..."
            ):
                cp_sims = conditional_path_bootstrap(
                    prices=price_series,
                    current_drawdown=current_dd,
                    deploy_pcts=cp_deploy_pcts,
                    monthly_contribution=cp_monthly,
                    cash_accumulated=cp_cash,
                    horizon_months=cp_horizon,
                    n_simulations=500,
                )
            st.session_state["dip_cp_sims"] = {
                "fp": _cp_fp,
                "data": [
                    {
                        "deploy_pct": s.deploy_pct,
                        "p5_wealth": s.p5_wealth.tolist(),
                        "p25_wealth": s.p25_wealth.tolist(),
                        "p50_wealth": s.p50_wealth.tolist(),
                        "p75_wealth": s.p75_wealth.tolist(),
                        "p95_wealth": s.p95_wealth.tolist(),
                        "prob_beats_dca": s.prob_beats_dca,
                        "horizon_months": s.horizon_months,
                    }
                    for s in cp_sims
                ],
            }

        _stored_cp = st.session_state.get("dip_cp_sims")
        if _stored_cp and isinstance(_stored_cp, dict) and _stored_cp.get("fp") == _cp_fp:
            cp_dicts = _stored_cp["data"]
        else:
            if _stored_cp:
                st.info("Parameters changed — click Run to update results.")
            cp_dicts = []
        if cp_dicts:
            cp_path_sims = [
                PathSimulation(
                    deploy_pct=d["deploy_pct"],
                    p5_wealth=np.array(d["p5_wealth"]),
                    p25_wealth=np.array(d["p25_wealth"]),
                    p50_wealth=np.array(d["p50_wealth"]),
                    p75_wealth=np.array(d["p75_wealth"]),
                    p95_wealth=np.array(d["p95_wealth"]),
                    prob_beats_dca=d["prob_beats_dca"],
                    horizon_months=d["horizon_months"],
                )
                for d in cp_dicts
            ]

            fig_cp_fan = plot_fan_chart(
                cp_path_sims,
                horizon_months=cp_horizon,
                monthly_contribution=cp_monthly,
                currency="EUR",
            )
            st.plotly_chart(fig_cp_fan, use_container_width=True)

            # Summary table
            summary_cp = []
            for s in cp_path_sims:
                summary_cp.append(
                    {
                        "Deploy %": f"{s.deploy_pct:.0%}",
                        "P(beats DCA)": f"{s.prob_beats_dca:.0%}",
                        "Median Wealth": f"EUR {s.p50_wealth[-1]:,.0f}",
                        "P5 (Worst)": f"EUR {s.p5_wealth[-1]:,.0f}",
                        "P95 (Best)": f"EUR {s.p95_wealth[-1]:,.0f}",
                    }
                )
            st.dataframe(
                pd.DataFrame(summary_cp), use_container_width=True, hide_index=True
            )

            st.caption(
                f"Based on {len(episodes)} historical entry points at "
                f"drawdown levels near {current_dd:.0%}. "
                "500 bootstrap paths per deploy fraction. Not a forecast."
            )

# ---------------------------------------------------------------------------
# No crystal ball disclaimer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="border:1px solid #3D4066; border-radius:10px; padding:16px;
                background:rgba(30,33,48,0.6);">
        <b style="color:#F47920">🔮 No Crystal Ball</b><br>
        <span style="color:#9CA3AF">
        This analysis describes what happened <em>after</em> past drawdowns of similar depth.
        It does not predict future performance. Markets can and do behave differently from
        historical base rates — especially during structurally novel regimes. Historical
        frequency is not probability. Use this tool to inform your thinking, not to time markets.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

data_source_caption(
    source=data_source,
    as_of=as_of_date,
    is_total_return=True,
    currency="various",
)
