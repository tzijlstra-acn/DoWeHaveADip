"""Strategy Lab — head-to-head DCA vs Wait-for-Dip vs Dip Buffet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.config import load_assets_config  # noqa: E402
from dipdca.models import SimulationParams  # noqa: E402
from dipdca.quant.backtest import run_dca, run_tiered_dip, run_wait_for_dip  # noqa: E402
from dipdca.quant.episodes import load_named_episodes  # noqa: E402
from ui.charts import add_named_episode_labels, drawdown_chart, plot_strategy_wealth  # noqa: E402
from ui.components import (  # noqa: E402
    data_source_caption,
    page_header,
    sidebar_simulation_params,
    strategy_comparison_cards,
    strategy_metrics,
)
from ui.formatting import fmt_currency, fmt_pct, fmt_ratio  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.set_page_config(page_title="Strategy Lab", page_icon="⚔️", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "Strategy Lab",
    "Head-to-head: Monthly Machine (DCA) vs Cash Goblin (Wait-for-Dip) vs Dip Buffet (Tiered)",
    "⚔️",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
param_dict = sidebar_simulation_params()

st.sidebar.subheader("Savings Mechanics")
deployment_pct = (
    st.sidebar.slider("Deploy % of cash at dip (%)", 10, 100, 100, step=10) / 100.0
)
cash_buffer = st.sidebar.slider("Emergency buffer (months)", 0, 24, 0)
deploy_spread_raw = st.sidebar.select_slider(
    "Deploy schedule",
    ["Lump sum", "Over 3 months", "Over 6 months"],
    value="Lump sum",
)
spread_map = {"Lump sum": 1, "Over 3 months": 3, "Over 6 months": 6}
deploy_spread = spread_map[deploy_spread_raw]

st.sidebar.subheader("Conviction Signal")
use_fg = st.sidebar.checkbox(
    "Apply Fear & Greed multiplier",
    value=False,
    help="Adjusts deployment fraction based on market sentiment",
)

fg_value = 50
fg = None
if use_fg:
    try:
        from dipdca.data.providers.fear_greed import (  # noqa: E402
            fetch_current_fear_greed,
            score_to_color,
            score_to_deployment_multiplier,
        )

        fg = fetch_current_fear_greed()
    except Exception:
        fg = None

    if fg is not None:
        fg_value = fg.value
        fg_color = score_to_color(fg.value)
        st.sidebar.markdown(f"**Live F&G: {fg.value:.0f} — {fg.label}**")
    else:
        fg_value = st.sidebar.slider("F&G Index (manual)", 0, 100, 50)
        score_to_deployment_multiplier = lambda s: (  # noqa: E731
            2.0 if s <= 15 else 1.5 if s <= 25 else 1.0 if s <= 55 else 0.75 if s <= 75 else 0.5
        )

    multiplier = score_to_deployment_multiplier(fg_value)
    effective_deploy_pct = min(1.0, deployment_pct * multiplier)
    st.sidebar.caption(
        f"Conviction multiplier: {multiplier:.1f}x → effective deploy: {effective_deploy_pct:.0%}"
    )
else:
    effective_deploy_pct = deployment_pct

st.sidebar.subheader("Asset / Ticker")
_assets = load_assets_config()
_asset_options = {a["display_name"]: a["etf_symbol"] for a in _assets}
_asset_names = list(_asset_options.keys())
_default_idx = _asset_names.index("Nasdaq-100") if "Nasdaq-100" in _asset_names else 0
_selected_name = st.sidebar.selectbox("Select Asset", _asset_names, index=_default_idx)
symbol = _asset_options[_selected_name]

# ---------------------------------------------------------------------------
# Load price data
# ---------------------------------------------------------------------------
from dipdca.data.providers.yahoo import YahooProvider  # noqa: E402

with st.spinner(f"Fetching market data for {symbol}..."):
    try:
        provider = YahooProvider()
        price_data = provider.get_price_data(symbol, param_dict["start_date"], param_dict["end_date"])
        price_df = price_data.df
        data_source = f"Yahoo Finance ({symbol})"
        as_of_date = price_data.as_of
    except Exception as exc:
        st.error(f"Could not load market data for {symbol}: {exc}")
        st.stop()

start_ts = pd.Timestamp(param_dict["start_date"])
end_ts = pd.Timestamp(param_dict["end_date"])
price_df = price_df.loc[start_ts:end_ts]

if len(price_df) < 10:
    st.error("Not enough data in the selected date range. Try a wider range.")
    st.stop()

# ---------------------------------------------------------------------------
# Build SimulationParams
# ---------------------------------------------------------------------------
try:
    params = SimulationParams(
        monthly_contribution=param_dict["monthly_contribution"],
        payday=param_dict["payday"],
        initial_investment=param_dict["initial_investment"],
        start_date=param_dict["start_date"],
        end_date=param_dict["end_date"],
        dip_threshold=param_dict["dip_threshold"],
        max_wait_months=param_dict["max_wait_months"],
        fixed_fee=param_dict["fixed_fee"],
        pct_fee=param_dict["pct_fee"],
        deployment_pct=effective_deploy_pct,
        cash_buffer_months=cash_buffer,
        deploy_spread_months=deploy_spread,
    )
except Exception as exc:
    st.error(f"Invalid parameters: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dca_dip, tab_tiered = st.tabs(["⚔️ DCA vs Cash Goblin", "🍽️ Dip Buffet (Tiered)"])

# ---------------------------------------------------------------------------
# Tab 1: DCA vs Wait-for-Dip
# ---------------------------------------------------------------------------
with tab_dca_dip:
    # F&G callout banner
    if use_fg and fg_value < 30:
        st.info(
            f"Extreme Fear detected (F&G: {fg_value:.0f}). "
            f"Conviction multiplier boosted to {multiplier:.1f}x. "
            "The spreadsheet is getting excited."
        )

    with st.spinner("Running backtests..."):
        try:
            dca_result, dca_ledger = run_dca(price_df, params)
            dip_result, dip_ledger = run_wait_for_dip(price_df, params)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    # Savings vs Investing summary
    total_saved = param_dict["monthly_contribution"] * len(
        pd.date_range(param_dict["start_date"], param_dict["end_date"], freq="ME")
    )
    total_invested_dip = float(dip_ledger["market_value"].mean())
    cash_days = float((dip_ledger["cash"] > 0).sum())
    total_days = float(len(dip_ledger))
    avg_cash_drag_months = (cash_days / total_days) * (total_days / 21)
    peak_uninvested = float(dip_ledger["cash"].max())

    with st.expander("Savings vs Investing Summary"):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Saved (contributions)", fmt_currency(total_saved))
        with s2:
            st.metric("Avg Invested (market value)", fmt_currency(total_invested_dip))
        with s3:
            st.metric("Avg Cash Drag", f"{avg_cash_drag_months:.1f} months")
        with s4:
            st.metric("Peak Uninvested Cash", fmt_currency(peak_uninvested))
        st.caption(
            f"Deploy schedule: {deploy_spread_raw} | Buffer: {cash_buffer} months | "
            f"Deploy fraction: {effective_deploy_pct:.0%}"
        )

    # Hero comparison cards
    strategy_comparison_cards(dca_result, dip_result)
    st.write("")

    # Dramatic winner announcement
    if dca_result.ending_wealth >= dip_result.ending_wealth:
        diff = dca_result.ending_wealth - dip_result.ending_wealth
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,rgba(0,200,150,0.15) 0%,rgba(0,200,150,0.05) 100%);
                        border:2px solid #00C896; border-radius:14px; padding:20px; text-align:center;
                        margin:16px 0">
                <p style="font-size:2em; margin:0">🤖</p>
                <h2 style="color:#00C896; margin:4px 0; font-size:1.8em; font-weight:900">
                    Monthly Machine wins!
                </h2>
                <p style="color:#9CA3AF; margin:4px 0">
                    By <b style="color:#00C896">{fmt_currency(diff)}</b> in ending wealth.
                    Boring wins again — but past performance does not guarantee future results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        diff = dip_result.ending_wealth - dca_result.ending_wealth
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,rgba(244,121,32,0.15) 0%,rgba(244,121,32,0.05) 100%);
                        border:2px solid #F47920; border-radius:14px; padding:20px; text-align:center;
                        margin:16px 0">
                <p style="font-size:2em; margin:0">💰</p>
                <h2 style="color:#F47920; margin:4px 0; font-size:1.8em; font-weight:900">
                    Cash Goblin wins!
                </h2>
                <p style="color:#9CA3AF; margin:4px 0">
                    By <b style="color:#F47920">{fmt_currency(diff)}</b> in ending wealth.
                    The patient hoarder prevailed — in this particular window.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Strategy metric rows
    st.markdown(
        '<h4 style="color:#00C896; margin:16px 0 8px 0">🤖 Monthly Machine (DCA)</h4>',
        unsafe_allow_html=True,
    )
    strategy_metrics(dca_result)

    st.markdown(
        '<h4 style="color:#F47920; margin:16px 0 8px 0">💰 Cash Goblin (Wait-for-Dip)</h4>',
        unsafe_allow_html=True,
    )
    strategy_metrics(dip_result)

    # Wealth chart with dip highlights
    st.markdown(
        '<h4 style="color:#FAFAFA; font-weight:700; margin:20px 0 8px 0">Wealth Curves + Dip Highlights</h4>',
        unsafe_allow_html=True,
    )
    fig_wealth = plot_strategy_wealth(dca_ledger, dip_ledger, base_currency="EUR")

    # Add named historical episode overlays
    try:
        named_eps = load_named_episodes()
        start_ts = dca_ledger.index[0]
        end_ts = dca_ledger.index[-1]
        add_named_episode_labels(fig_wealth, named_eps, date_start=start_ts, date_end=end_ts)
    except Exception:
        pass

    st.plotly_chart(fig_wealth, use_container_width=True)

    # Historical episodes catalog
    try:
        named_eps_catalog = load_named_episodes()
        with st.expander("Historical Named Episodes"):
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
                        <span style="color:#9CA3AF; font-size:0.8em; margin-left:8px">
                            Recovery: {ep.get('recovery', 'N/A')}
                        </span>
                        <br>
                        <span style="color:#9CA3AF; font-size:0.85em">{ep['note']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    # Deployment markers
    deploy_dates = dip_ledger[dip_ledger["deployed"] > 0]
    if not deploy_dates.empty:
        with st.expander(f"Deployment events ({len(deploy_dates)} total)"):
            disp = deploy_dates[["deployed", "fees"]].copy()
            disp.index = disp.index.date
            disp.columns = ["Amount Deployed (EUR)", "Fees (EUR)"]
            st.dataframe(disp, use_container_width=True)

    # Drawdown charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<h4 style="color:#FAFAFA; font-weight:700">DCA Portfolio Drawdown</h4>',
            unsafe_allow_html=True,
        )
        fig_dd_dca = drawdown_chart(dca_ledger["dd"], title="DCA Drawdown", threshold=params.dip_threshold)
        st.plotly_chart(fig_dd_dca, use_container_width=True)
    with col2:
        st.markdown(
            '<h4 style="color:#FAFAFA; font-weight:700">Dip Strategy Drawdown</h4>',
            unsafe_allow_html=True,
        )
        fig_dd_dip = drawdown_chart(
            dip_ledger["dd"], title="Dip Strategy Drawdown", threshold=params.dip_threshold
        )
        st.plotly_chart(fig_dd_dip, use_container_width=True)

    # Full metrics table
    st.markdown(
        '<h4 style="color:#FAFAFA; font-weight:700; margin:20px 0 8px 0">Full Metrics Comparison</h4>',
        unsafe_allow_html=True,
    )
    metrics_data = {
        "Metric": [
            "Ending Wealth",
            "Total Contributions",
            "P&L",
            "XIRR",
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Time in Market",
            "Deployments",
            "Total Fees",
            "Cash Interest Earned",
        ],
        "Monthly Machine (DCA)": [
            fmt_currency(dca_result.ending_wealth),
            fmt_currency(dca_result.total_contributions),
            fmt_currency(dca_result.pnl),
            fmt_pct(dca_result.xirr),
            fmt_pct(dca_result.cagr),
            fmt_ratio(dca_result.sharpe),
            fmt_ratio(dca_result.sortino),
            fmt_pct(dca_result.max_drawdown),
            fmt_pct(dca_result.time_in_market_pct),
            str(dca_result.n_deployments),
            fmt_currency(dca_result.total_fees),
            fmt_currency(dca_result.total_cash_interest),
        ],
        "Cash Goblin (Wait-for-Dip)": [
            fmt_currency(dip_result.ending_wealth),
            fmt_currency(dip_result.total_contributions),
            fmt_currency(dip_result.pnl),
            fmt_pct(dip_result.xirr),
            fmt_pct(dip_result.cagr),
            fmt_ratio(dip_result.sharpe),
            fmt_ratio(dip_result.sortino),
            fmt_pct(dip_result.max_drawdown),
            fmt_pct(dip_result.time_in_market_pct),
            str(dip_result.n_deployments),
            fmt_currency(dip_result.total_fees),
            fmt_currency(dip_result.total_cash_interest),
        ],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # Downloads
    st.markdown(
        '<h4 style="color:#FAFAFA; font-weight:700; margin:20px 0 8px 0">Download Ledgers</h4>',
        unsafe_allow_html=True,
    )
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download DCA Ledger (CSV)", data=dca_ledger.to_csv(), file_name="dca_ledger.csv", mime="text/csv"
        )
    with col_dl2:
        st.download_button(
            "Download Dip Ledger (CSV)", data=dip_ledger.to_csv(), file_name="dip_ledger.csv", mime="text/csv"
        )

    data_source_caption(source=data_source, as_of=as_of_date, is_total_return=True, currency="EUR")

# ---------------------------------------------------------------------------
# Tab 2: Tiered Dip Buffet
# ---------------------------------------------------------------------------
with tab_tiered:
    st.markdown(
        '<h3 style="color:#FFD700; font-weight:700; margin-bottom:4px">🍽️ Dip Buffet — Configure Your Tiers</h3>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Deploy different fractions of your cash at different drawdown depths. "
        "Tiers reset after a new all-time high. Deeper dip = bigger deployment."
    )

    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        st.markdown("**Tier 1 (Shallow dip)**")
        tier1_thresh = st.slider("Threshold T1 (%)", -30, -1, -5, key="t1_thresh") / 100.0
        tier1_frac = st.slider("Fraction of cash T1 (%)", 5, 100, 25, key="t1_frac") / 100.0

    with col_t2:
        st.markdown("**Tier 2 (Medium dip)**")
        tier2_thresh = st.slider("Threshold T2 (%)", -40, -5, -10, key="t2_thresh") / 100.0
        tier2_frac = st.slider("Fraction of cash T2 (%)", 5, 100, 50, key="t2_frac") / 100.0

    with col_t3:
        st.markdown("**Tier 3 (Deep dip)**")
        tier3_thresh = st.slider("Threshold T3 (%)", -60, -10, -20, key="t3_thresh") / 100.0
        tier3_frac = st.slider("Fraction of cash T3 (%)", 5, 100, 100, key="t3_frac") / 100.0

    tiers = [
        {"threshold": tier1_thresh, "fraction": tier1_frac},
        {"threshold": tier2_thresh, "fraction": tier2_frac},
        {"threshold": tier3_thresh, "fraction": tier3_frac},
    ]

    with st.spinner("Running Dip Buffet backtest..."):
        try:
            tiered_result, tiered_ledger = run_tiered_dip(price_df, params, tiers)
            dca_result_t, dca_ledger_t = run_dca(price_df, params)
        except Exception as exc:
            st.error(f"Tiered backtest failed: {exc}")
            st.stop()

    # Dramatic comparison
    diff_vs_dca = tiered_result.ending_wealth - dca_result_t.ending_wealth
    winner = "Dip Buffet" if diff_vs_dca > 0 else "Monthly Machine"
    winner_color = "#FFD700" if diff_vs_dca > 0 else "#00C896"
    st.markdown(
        f"""
        <div style="background:#1E2130; border:2px solid {winner_color}; border-radius:12px;
                    padding:16px; text-align:center; margin-bottom:16px">
            <h3 style="color:{winner_color}; margin:0; font-weight:900">{winner} wins</h3>
            <p style="color:#9CA3AF; margin:4px 0">
                by <b style="color:{winner_color}">{fmt_currency(abs(diff_vs_dca))}</b> ending wealth
                (DCA: {fmt_currency(dca_result_t.ending_wealth)} | Buffet: {fmt_currency(tiered_result.ending_wealth)})
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    with col_m1:
        st.metric("Ending Wealth", fmt_currency(tiered_result.ending_wealth))
    with col_m2:
        st.metric("XIRR", fmt_pct(tiered_result.xirr))
    with col_m3:
        st.metric("Max Drawdown", fmt_pct(tiered_result.max_drawdown))
    with col_m4:
        st.metric("Time in Market", fmt_pct(tiered_result.time_in_market_pct))
    with col_m5:
        st.metric("Deployments", str(tiered_result.n_deployments))
    with col_m6:
        st.metric("Cash Interest", fmt_currency(tiered_result.total_cash_interest))

    # Wealth chart
    st.markdown(
        '<h4 style="color:#FAFAFA; font-weight:700; margin:16px 0 8px 0">Wealth + Cash Curves</h4>',
        unsafe_allow_html=True,
    )
    import plotly.graph_objects as go

    from ui.theme import BLUE, GOLD, GREEN, apply_chart_layout

    fig_tiered = go.Figure()
    fig_tiered.add_trace(
        go.Scatter(
            x=dca_ledger_t.index,
            y=dca_ledger_t["total_wealth"],
            name="Monthly Machine (DCA)",
            line=dict(color=GREEN, width=2),
        )
    )
    fig_tiered.add_trace(
        go.Scatter(
            x=tiered_ledger.index,
            y=tiered_ledger["total_wealth"],
            name="Dip Buffet (Tiered)",
            line=dict(color=GOLD, width=2),
        )
    )
    fig_tiered.add_trace(
        go.Scatter(
            x=tiered_ledger.index,
            y=tiered_ledger["cash"],
            name="Cash balance (Buffet)",
            line=dict(color=BLUE, width=1.5, dash="dot"),
            opacity=0.7,
        )
    )

    # Mark deployment events
    deploy_tiered = tiered_ledger[tiered_ledger["deployed"] > 0]
    if not deploy_tiered.empty:
        fig_tiered.add_trace(
            go.Scatter(
                x=deploy_tiered.index,
                y=deploy_tiered["total_wealth"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=GOLD),
                name="Deployments",
                hovertemplate="Deployed: %{customdata:,.0f}<extra></extra>",
                customdata=deploy_tiered["deployed"].values,
            )
        )

    apply_chart_layout(fig_tiered, title="Dip Buffet vs Monthly Machine (EUR)")
    fig_tiered.update_layout(xaxis_title="Date", yaxis_title="Wealth (EUR)", height=450)
    st.plotly_chart(fig_tiered, use_container_width=True)

    st.caption(
        "Triangles mark deployment events. Dotted line = uninvested cash in Dip Buffet. "
        "Congratulations, you discovered the past. 🎉"
    )

    data_source_caption(source=data_source, as_of=as_of_date, is_total_return=True, currency="EUR")
