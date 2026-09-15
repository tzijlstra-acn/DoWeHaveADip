"""Exit Laboratory — model exit strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.models import SimulationParams  # noqa: E402
from dipdca.quant.backtest import run_dca, run_wait_for_dip  # noqa: E402
from dipdca.quant.exits import ExitType, simulate_exit  # noqa: E402
from ui.charts import drawdown_chart  # noqa: E402
from ui.components import (  # noqa: E402
    data_source_caption,
    page_header,
    sidebar_simulation_params,
)
from ui.formatting import fmt_currency, fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS, GREEN, ORANGE, RED, apply_chart_layout  # noqa: E402

st.set_page_config(page_title="Exit Lab", page_icon="🚪", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "Exit Lab",
    "When should you sell? Model deterministic exit rules vs the never-sell benchmark. Every rule has a cost.",
    "🚪",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
param_dict = sidebar_simulation_params()

st.sidebar.subheader("Ticker")
symbol = st.sidebar.text_input("ETF Symbol", value="SPY")

st.sidebar.subheader("Exit Rule")
exit_type_label = st.sidebar.selectbox(
    "Exit Rule",
    ["Never Sell", "Target Return", "Time-Based Hold", "Trailing Stop"],
)

exit_type_map = {
    "Never Sell": ExitType.NONE,
    "Target Return": ExitType.TARGET_RETURN,
    "Time-Based Hold": ExitType.TIME_BASED,
    "Trailing Stop": ExitType.TRAILING_STOP,
}
exit_type = exit_type_map[exit_type_label]

target_return = 0.5
hold_years = 5
trailing_stop = -0.15

if exit_type == ExitType.TARGET_RETURN:
    target_return = (
        st.sidebar.slider("Target gain (%)", min_value=10, max_value=300, value=50, step=5) / 100.0
    )
elif exit_type == ExitType.TIME_BASED:
    hold_years = st.sidebar.slider("Hold period (years)", min_value=1, max_value=30, value=5)
elif exit_type == ExitType.TRAILING_STOP:
    trailing_stop = (
        st.sidebar.slider("Trailing stop (%)", min_value=-50, max_value=-5, value=-15, step=1) / 100.0
    )

# ---------------------------------------------------------------------------
# Load data
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

if len(price_df) < 30:
    st.error("Not enough data. Try a wider date range.")
    st.stop()

# ---------------------------------------------------------------------------
# Build params and run
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
    )
except Exception as exc:
    st.error(f"Invalid parameters: {exc}")
    st.stop()

with st.spinner("Running backtests..."):
    try:
        dca_result, dca_ledger = run_dca(price_df, params)
        dip_result, dip_ledger = run_wait_for_dip(price_df, params)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

# Simulate exit on dip ledger
exit_result_dca = simulate_exit(
    dca_ledger,
    ExitType.NONE,
    target_return=target_return,
    hold_years=hold_years,
    trailing_stop=trailing_stop,
)
exit_result_dip_never = simulate_exit(dip_ledger, ExitType.NONE)
exit_result_dip = simulate_exit(
    dip_ledger,
    exit_type,
    target_return=target_return,
    hold_years=hold_years,
    trailing_stop=trailing_stop,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
st.subheader(f"Exit Rule: {exit_type_label}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Monthly Machine (never sell)**")
    st.metric("Ending Wealth", fmt_currency(exit_result_dca["exit_wealth"]))
    st.metric("Total Return", fmt_pct(exit_result_dca["total_return"]))
    st.metric("Hold Period", f"{exit_result_dca['hold_days']} days")
    exit_dca_date = exit_result_dca["exit_date"]
    if exit_dca_date:
        st.caption(f"Exit date: {exit_dca_date}")

with col2:
    st.markdown("**Cash Goblin (never sell)**")
    st.metric("Ending Wealth", fmt_currency(exit_result_dip_never["exit_wealth"]))
    st.metric("Total Return", fmt_pct(exit_result_dip_never["total_return"]))
    st.metric("Hold Period", f"{exit_result_dip_never['hold_days']} days")

with col3:
    st.markdown(f"**Cash Goblin + {exit_type_label}**")
    delta_vs_never = exit_result_dip["exit_wealth"] - exit_result_dip_never["exit_wealth"]
    delta_str = f"{delta_vs_never:+,.0f} vs never sell"
    color = GREEN if delta_vs_never >= 0 else RED
    st.metric(
        "Exit Wealth",
        fmt_currency(exit_result_dip["exit_wealth"]),
        delta=delta_str,
    )
    st.metric("Total Return", fmt_pct(exit_result_dip["total_return"]))
    st.metric("Hold Period", f"{exit_result_dip['hold_days']} days")
    exit_date = exit_result_dip.get("exit_date")
    if exit_date and exit_date != dip_ledger.index[-1]:
        st.caption(f"Exit triggered: {exit_date}")
    else:
        st.caption("Exit not triggered — held to end")

# ---------------------------------------------------------------------------
# Wealth chart with exit marker
# ---------------------------------------------------------------------------
st.subheader("Wealth Curves")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=dca_ledger.index,
        y=dca_ledger["total_wealth"],
        name="Monthly Machine (DCA)",
        line=dict(color=GREEN, width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=dip_ledger.index,
        y=dip_ledger["total_wealth"],
        name="Cash Goblin (never sell)",
        line=dict(color=ORANGE, width=2),
    )
)

# Mark exit point if applicable
if exit_type != ExitType.NONE and exit_result_dip.get("exit_date"):
    ed = exit_result_dip["exit_date"]
    ew = exit_result_dip["exit_wealth"]
    fig.add_trace(
        go.Scatter(
            x=[ed],
            y=[ew],
            mode="markers",
            marker=dict(symbol="x", size=14, color=RED, line=dict(width=2)),
            name=f"Exit: {exit_type_label}",
        )
    )
    fig.add_vline(
        x=ed,
        line_dash="dash",
        line_color=RED,
        annotation_text=f"Exit ({exit_type_label})",
        annotation_font_color=RED,
    )

apply_chart_layout(fig, title="Wealth Curves with Exit Markers (EUR)")
fig.update_layout(xaxis_title="Date", yaxis_title="Wealth (EUR)", height=420)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Time in market comparison
# ---------------------------------------------------------------------------
st.subheader("Time in Market")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.metric("DCA — Time in Market", fmt_pct(dca_result.time_in_market_pct))
    st.metric("DCA — Turnover events", str(dca_result.n_deployments))
with col_t2:
    st.metric("Dip — Time in Market", fmt_pct(dip_result.time_in_market_pct))
    st.metric("Dip — Turnover events", str(dip_result.n_deployments))

# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
st.subheader("Portfolio Drawdown")
col_dd1, col_dd2 = st.columns(2)
with col_dd1:
    fig_dd1 = drawdown_chart(dca_ledger["dd"], title="DCA Drawdown", threshold=params.dip_threshold)
    st.plotly_chart(fig_dd1, use_container_width=True)
with col_dd2:
    fig_dd2 = drawdown_chart(dip_ledger["dd"], title="Dip Drawdown", threshold=params.dip_threshold)
    st.plotly_chart(fig_dd2, use_container_width=True)

# ---------------------------------------------------------------------------
# Exit rule descriptions
# ---------------------------------------------------------------------------
with st.expander("Exit Rule Definitions"):
    st.markdown(
        """
        | Rule | Trigger | Notes |
        |------|---------|-------|
        | **Never Sell** | End of period | Hold-forever benchmark |
        | **Target Return** | Portfolio gain ≥ X% from start | Locks in profits, misses further upside |
        | **Time-Based Hold** | Sell after N years | Predictable, ignores market conditions |
        | **Trailing Stop** | Portfolio falls Y% from its peak-since-entry | Protects gains, may exit on volatility |

        All rules fire **once** — after the exit, the portfolio is marked as liquidated.
        Tax and re-investment effects are not modelled here.
        """
    )

data_source_caption(source=data_source, as_of=as_of_date, is_total_return=True, currency="EUR")
