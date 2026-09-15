"""Compare choices — head-to-head backtest: DCA vs Wait-for-dip vs Tiered."""

from __future__ import annotations

import datetime
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
from dipdca.models import SimulationParams  # noqa: E402
from dipdca.quant.backtest import run_dca, run_tiered_dip, run_wait_for_dip  # noqa: E402
from dipdca.quant.episodes import load_named_episodes  # noqa: E402
from ui.charts import add_named_episode_labels, drawdown_chart, plot_strategy_wealth  # noqa: E402
from ui.components import (  # noqa: E402
    freshness_caption,
    live_data_error,
    page_header,
    strategy_comparison_cards,
    strategy_metrics,
)
from ui.copy import DISCLAIMER_SHORT, LABEL_DCA, LABEL_TIERED, LABEL_WAIT  # noqa: E402
from ui.design_tokens import NEUTRAL, POSITIVE, WARNING  # noqa: E402
from ui.formatting import fmt_currency  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "Compare choices",
    "Head-to-head comparison: invest monthly vs wait for a dip vs invest now. "
    "Uses real historical data — not a prediction.",
)

# ---------------------------------------------------------------------------
# Inputs — 4 visible, rest in expander
# ---------------------------------------------------------------------------
_assets_cfg = load_assets_config()
_asset_options = {a["display_name"]: a["etf_symbol"] for a in _assets_cfg}
_asset_names = list(_asset_options.keys())
_default_idx = next((i for i, n in enumerate(_asset_names) if "Nasdaq-100" in n), 0)

col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 2])
with col_a:
    selected_name = st.selectbox("Market or ETF", _asset_names, index=_default_idx)
with col_b:
    monthly_contribution = st.number_input("Monthly amount (EUR)", min_value=10, max_value=100_000, value=500, step=50)
with col_c:
    cash_available = st.number_input("Cash to deploy (EUR)", min_value=0, max_value=1_000_000, value=0, step=100)
with col_d:
    initial_investment = st.number_input("Initial lump sum (EUR)", min_value=0, max_value=1_000_000, value=0, step=100)

symbol = _asset_options[selected_name]

with st.expander("Advanced assumptions"):
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        dip_threshold = st.slider("Dip threshold (%)", min_value=-50, max_value=-1, value=-10, step=1) / 100.0
        max_wait_months = st.slider("Max wait (months)", min_value=1, max_value=60, value=24)
    with adv2:
        deployment_pct = st.slider("Deploy % of cash at dip", 10, 100, 100, step=10) / 100.0
        cash_buffer = st.slider("Emergency buffer (months)", 0, 24, 0)
    with adv3:
        start_date = st.date_input("History start", value=datetime.date(2015, 1, 1))
        end_date = st.date_input("History end", value=datetime.date(2024, 12, 31))
        fixed_fee = st.number_input("Fixed fee (EUR)", min_value=0.0, value=0.0, step=0.5)
        pct_fee = st.slider("% fee (bps)", min_value=0, max_value=100, value=10) / 10_000.0

# ---------------------------------------------------------------------------
# Load market data
# ---------------------------------------------------------------------------
with st.spinner(f"Loading {selected_name} data..."):
    try:
        _result = get_market_data_service().get_history(symbol, start_date, end_date)
        price_df = _result.frame
    except LiveDataUnavailable as exc:
        live_data_error(exc, context=selected_name)
        st.stop()

freshness_caption(_result.freshness)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
price_df = price_df.loc[start_ts:end_ts]

if len(price_df) < 10:
    st.error("Not enough data in the selected date range. Try a wider range.")
    st.stop()

# ---------------------------------------------------------------------------
# Build SimulationParams
# ---------------------------------------------------------------------------
try:
    params = SimulationParams(
        monthly_contribution=float(monthly_contribution),
        payday=25,
        initial_investment=float(initial_investment),
        start_date=start_date,
        end_date=end_date,
        dip_threshold=dip_threshold,
        max_wait_months=max_wait_months,
        fixed_fee=fixed_fee,
        pct_fee=pct_fee,
        deployment_pct=deployment_pct,
        cash_buffer_months=cash_buffer,
        deploy_spread_months=1,
    )
except Exception as exc:
    st.error(f"Invalid parameters: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_main, tab_tiered = st.tabs([f"{LABEL_DCA} vs {LABEL_WAIT}", LABEL_TIERED])

# ---------------------------------------------------------------------------
# Tab 1: DCA vs Wait-for-Dip
# ---------------------------------------------------------------------------
with tab_main:
    with st.spinner("Running backtests..."):
        try:
            dca_result, dca_ledger = run_dca(price_df, params)
            dip_result, dip_ledger = run_wait_for_dip(price_df, params)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    # Neutral conclusion
    diff = dca_result.ending_wealth - dip_result.ending_wealth
    if abs(diff) < 1:
        conclusion = "Both strategies ended at essentially the same wealth over this period."
        conclusion_color = NEUTRAL
    elif diff > 0:
        conclusion = (
            f"Invest monthly came out ahead by {fmt_currency(abs(diff))} in this period. "
            "Past outcomes do not predict future results."
        )
        conclusion_color = POSITIVE
    else:
        conclusion = (
            f"Wait for a dip came out ahead by {fmt_currency(abs(diff))} in this period. "
            "Past outcomes do not predict future results."
        )
        conclusion_color = WARNING

    st.markdown(
        f"""
        <div style="background:#F9FAFB; border-left:4px solid {conclusion_color};
                    border-radius:0 8px 8px 0; padding:14px 20px; margin:16px 0 24px 0">
            <p style="margin:0; color:#111827; font-size:1em">{conclusion}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Comparison cards (no mascots — updated in ui/components.py)
    strategy_comparison_cards(dca_result, dip_result)
    st.write("")

    # Metrics rows
    st.markdown(f"**{LABEL_DCA}**")
    strategy_metrics(dca_result)

    st.markdown(f"**{LABEL_WAIT}** (threshold: {dip_threshold:.0%})")
    strategy_metrics(dip_result)

    st.divider()

    # Wealth chart
    st.subheader("Wealth over time")
    fig_wealth = plot_strategy_wealth(dca_ledger, dip_ledger, base_currency="EUR")
    try:
        named_eps = load_named_episodes()
        add_named_episode_labels(fig_wealth, named_eps, date_start=dca_ledger.index[0], date_end=dca_ledger.index[-1])
    except Exception:
        pass
    st.plotly_chart(fig_wealth, use_container_width=True)

    # Drawdown chart
    with st.expander("Drawdown history"):
        price_series = price_df["adj_close"].dropna()
        fig_dd = drawdown_chart(price_series, title=f"{selected_name} — Drawdown from High")
        st.plotly_chart(fig_dd, use_container_width=True)

    # Deployment events
    deploy_dates = dip_ledger[dip_ledger["deployed"] > 0]
    if not deploy_dates.empty:
        with st.expander(f"Deployment events ({len(deploy_dates)} total)"):
            disp = deploy_dates[["deployed", "fees"]].copy()
            disp.index = disp.index.date
            disp = disp.rename(columns={"deployed": "Deployed (EUR)", "fees": "Fees (EUR)"})
            st.dataframe(disp, use_container_width=True)

    # Savings summary
    total_saved = float(monthly_contribution) * len(
        pd.date_range(start_date, end_date, freq="ME")
    )
    with st.expander("Savings vs investing summary"):
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Total contributions", fmt_currency(total_saved))
        with s2:
            peak_uninvested = float(dip_ledger["cash"].max())
            st.metric("Peak uninvested cash", fmt_currency(peak_uninvested))
        with s3:
            cash_days = float((dip_ledger["cash"] > 0).sum())
            total_days = float(len(dip_ledger))
            avg_cash_drag = (cash_days / total_days) * (total_days / 21)
            st.metric("Avg cash drag", f"{avg_cash_drag:.1f} months")

# ---------------------------------------------------------------------------
# Tab 2: Tiered deployment
# ---------------------------------------------------------------------------
with tab_tiered:
    st.markdown(
        "Tiered deployment invests a fixed fraction of accumulated cash at each dip level. "
        "Configure tiers below."
    )

    st.subheader("Configure tiers")
    tier_cols = st.columns(3)
    tiers: list[tuple[float, float]] = []
    default_tiers = [(-0.10, 0.33), (-0.20, 0.33), (-0.30, 0.34)]
    for i, col in enumerate(tier_cols):
        with col:
            t_thresh = col.slider(
                f"Tier {i+1} threshold (%)", min_value=-50, max_value=-1,
                value=int(default_tiers[i][0] * 100), step=1, key=f"tier_thresh_{i}"
            ) / 100.0
            t_pct = col.slider(
                f"Tier {i+1} deploy %", min_value=1, max_value=100,
                value=int(default_tiers[i][1] * 100), step=1, key=f"tier_pct_{i}"
            ) / 100.0
            tiers.append((t_thresh, t_pct))

    if st.button("Run tiered backtest", type="primary"):
        with st.spinner("Running tiered backtest..."):
            try:
                tiered_result, tiered_ledger = run_tiered_dip(price_df, params, tiers=tiers)
                dca_result2, dca_ledger2 = run_dca(price_df, params)
            except Exception as exc:
                st.error(f"Tiered backtest failed: {exc}")
                st.stop()

        diff2 = tiered_result.ending_wealth - dca_result2.ending_wealth
        conclusion2 = (
            f"Tiered deployment ended {'ahead' if diff2 > 0 else 'behind'} by "
            f"{fmt_currency(abs(diff2))} compared to monthly DCA."
        )
        st.info(conclusion2)

        st.markdown(f"**{LABEL_TIERED}**")
        strategy_metrics(tiered_result)
        st.markdown(f"**{LABEL_DCA} (baseline)**")
        strategy_metrics(dca_result2)

        fig_t = plot_strategy_wealth(dca_ledger2, tiered_ledger, base_currency="EUR")
        st.plotly_chart(fig_t, use_container_width=True)

st.divider()
st.caption(DISCLAIMER_SHORT)
