"""Exit strategies — deterministic exit rules against the never-sell benchmark.

Rebuilt on the ATH-deployment engine and flow-adjusted NAV.

- The drawdown signal comes from the reference index, not the ETF price, so entry
  timing follows the same rule as the rest of the app.
- Exit rules are evaluated on a unitized NAV, so a contribution can neither
  satisfy a return target nor move a trailing-stop peak.
- The holding clock starts at the first deployment, not the first chart date.
- Every policy is valued on one common terminal date; proceeds from an exit are
  held at the savings rate rather than disappearing.
- Deltas are shown against monthly DCA, savings-only, and the same entry with no
  exit, since "better" depends on which of those you mean.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.config import load_assets_config  # noqa: E402
from dipdca.data.errors import LiveDataUnavailable  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from dipdca.models import DeploymentTier, SimulationParams  # noqa: E402
from dipdca.quant.backtest import (  # noqa: E402
    run_ath_deployment,
    run_dca,
    run_savings_only,
)
from dipdca.quant.exits import ExitType, ledger_nav, simulate_exit  # noqa: E402
from ui.components import (  # noqa: E402
    data_source_caption,
    freshness_caption,
    live_data_error,
    page_header,
)
from ui.formatting import compare_outcomes, fmt_currency, fmt_delta  # noqa: E402
from ui.theme import GLOBAL_CSS, GREEN, ORANGE, RED, apply_chart_layout  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "EXIT STRATEGIES",
    "When should you sell? Exit rules are measured on investment performance, "
    "not on account balance, and every policy is valued on the same date.",
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
_assets_cfg = load_assets_config()
_asset_map = {a["display_name"]: a for a in _assets_cfg}
_asset_names = list(_asset_map.keys())
_default_idx = next((i for i, n in enumerate(_asset_names) if "S&P 500" in n), 0)

col_a, col_b, col_c = st.columns([3, 2, 2])
with col_a:
    selected_name = st.selectbox("Market or ETF", _asset_names, index=_default_idx)
with col_b:
    monthly_contribution = st.number_input(
        "Monthly savings (EUR)", min_value=10, max_value=100_000, value=500, step=50
    )
with col_c:
    cash_available = st.number_input(
        "Cash waiting for a dip (EUR)", min_value=0, max_value=1_000_000,
        value=10_000, step=500,
    )

asset_cfg = _asset_map[selected_name]
etf_symbol = asset_cfg["etf_symbol"]
index_symbol = asset_cfg.get("index_symbol")
has_benchmark = index_symbol is not None

st.subheader("Exit rule")
exit_col1, exit_col2 = st.columns([2, 3])
with exit_col1:
    exit_type_label = st.selectbox(
        "Rule",
        ["Never sell", "Target return", "Time-based hold", "Trailing stop"],
    )

exit_type_map = {
    "Never sell": ExitType.NONE,
    "Target return": ExitType.TARGET_RETURN,
    "Time-based hold": ExitType.TIME_BASED,
    "Trailing stop": ExitType.TRAILING_STOP,
}
exit_type = exit_type_map[exit_type_label]

target_return, hold_years, trailing_stop = 0.5, 5, -0.15
with exit_col2:
    if exit_type == ExitType.TARGET_RETURN:
        target_return = st.slider(
            "Target investment gain (%)", min_value=10, max_value=300, value=50, step=5,
            help="Measured on unitized NAV, so contributions cannot satisfy it.",
        ) / 100.0
    elif exit_type == ExitType.TIME_BASED:
        hold_years = st.slider(
            "Hold period (years)", min_value=1, max_value=30, value=5,
            help="Counted from the first deployment, not from the chart start.",
        )
    elif exit_type == ExitType.TRAILING_STOP:
        trailing_stop = st.slider(
            "Trailing stop (%)", min_value=-50, max_value=-5, value=-15, step=1,
            help="Measured on unitized NAV, so deposits cannot move the peak.",
        ) / 100.0
    else:
        st.caption("The never-sell benchmark holds to the terminal date.")

with st.expander("Advanced assumptions"):
    adv1, adv2 = st.columns(2)
    with adv1:
        start_date = st.date_input("History start", value=datetime.date(2015, 1, 1))
        end_date = st.date_input("History end", value=datetime.date(2024, 12, 31))
        cash_rate_pct = st.slider(
            "Savings rate on cash (% p.a.)", min_value=0.0, max_value=8.0,
            value=2.0, step=0.25,
            help="Earned on cash waiting for a dip and on proceeds after an exit.",
        )
    with adv2:
        fixed_fee = st.number_input("Fixed fee (EUR)", min_value=0.0, value=0.0, step=0.5)
        pct_fee = st.slider("% fee (bps)", min_value=0, max_value=100, value=10) / 10_000.0
        slippage = st.slider(
            "Slippage (bps)", min_value=0, max_value=50, value=10,
            help="Applied per trade in addition to the fee above.",
        ) / 10_000.0
        entry_threshold = st.slider(
            "Entry threshold (% below index ATH)",
            min_value=-50, max_value=-5, value=-15, step=1,
        ) / 100.0

st.caption(
    f"Assumptions: entry at {entry_threshold:.0%} below the "
    f"{'index' if has_benchmark else 'ETF-proxy'} all-time high, "
    f"{pct_fee * 10_000:.0f} bps fee + {slippage * 10_000:.0f} bps slippage per trade, "
    f"{fmt_currency(fixed_fee, decimals=2)} fixed fee, "
    f"{cash_rate_pct:.2f}% p.a. on cash, contributions at each month-end close."
)

if not has_benchmark:
    st.warning(
        f"**{selected_name}** has no configured benchmark index, so the ETF price is "
        "used as its own drawdown signal. Results are ETF-proxy mode."
    )

# ---------------------------------------------------------------------------
# Load data — ATH seeded from history strictly before the evaluation window
# ---------------------------------------------------------------------------
LOOKBACK_YEARS = 20
history_start = start_date - datetime.timedelta(days=365 * LOOKBACK_YEARS)
start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

svc = get_market_data_service()
with st.spinner(f"Loading {selected_name} data..."):
    try:
        _inst = svc.get_history(etf_symbol, history_start, end_date)
        instrument_full = _inst.frame
        if has_benchmark:
            _bm = svc.get_history(index_symbol, history_start, end_date)
            benchmark_full = _bm.frame
            freshness_caption(_bm.freshness)
        else:
            benchmark_full = instrument_full
            freshness_caption(_inst.freshness)
        data_source = f"Yahoo Finance ({etf_symbol})"
        as_of_date = _inst.freshness.observed_at.date()
    except LiveDataUnavailable as exc:
        live_data_error(exc, context=selected_name)
        st.stop()

price_df = instrument_full.loc[start_ts:end_ts]
benchmark_df = benchmark_full.loc[start_ts:end_ts]

_pre = benchmark_full.loc[benchmark_full.index < start_ts]
initial_ath = float(_pre["adj_close"].max()) if not _pre.empty else None

if len(price_df) < 30:
    st.error("Not enough data in the selected range. Try a wider range.")
    st.stop()

if initial_ath is not None:
    st.caption(f"Opening ATH seeded from pre-window history: **{initial_ath:,.2f}**")

# ---------------------------------------------------------------------------
# Run strategies
# ---------------------------------------------------------------------------
try:
    params = SimulationParams(
        monthly_contribution=float(monthly_contribution),
        payday=25,
        contribution_timing="month_end",
        initial_investment=0.0,
        initial_cash_reserve=float(cash_available),
        start_date=start_date,
        end_date=end_date,
        dip_threshold=entry_threshold,
        fixed_fee=fixed_fee,
        pct_fee=pct_fee,
        slippage=slippage,
        cash_rate_override=cash_rate_pct / 100.0 if cash_rate_pct > 0 else None,
    )
except Exception as exc:
    st.error(f"Invalid parameters: {exc}")
    st.stop()

with st.spinner("Running backtests..."):
    try:
        dca_result, dca_ledger = run_dca(price_df, params)
        savings_result, savings_ledger = run_savings_only(price_df, params)
        dip_result, dip_ledger = run_ath_deployment(
            instrument_data=price_df,
            params=params,
            tiers=[DeploymentTier(entry_threshold, 1.00)],
            benchmark_data=benchmark_df,
            initial_ath=initial_ath,
        )
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

# One terminal date for every policy.
terminal_date = pd.Timestamp(price_df.index[-1])
cash_rate = cash_rate_pct / 100.0

exit_kw = dict(
    target_return=target_return,
    hold_years=hold_years,
    trailing_stop=trailing_stop,
    cash_rate=cash_rate,
    terminal_date=terminal_date,
)
dip_no_exit = simulate_exit(dip_ledger, ExitType.NONE, **exit_kw)
dip_with_exit = simulate_exit(dip_ledger, exit_type, **exit_kw)
dca_no_exit = simulate_exit(dca_ledger, ExitType.NONE, **exit_kw)
savings_hold = simulate_exit(savings_ledger, ExitType.NONE, **exit_kw)

# ---------------------------------------------------------------------------
# Trigger status
# ---------------------------------------------------------------------------
st.divider()
st.subheader(f"Result — {exit_type_label}")

if dip_with_exit["first_deployment"] is None:
    st.warning(
        f"The entry threshold of {entry_threshold:.0%} below the all-time high was "
        "never reached in this window, so nothing was ever bought and no exit rule "
        "could apply. Try a shallower threshold or a wider date range."
    )
elif exit_type == ExitType.NONE:
    st.info("Never-sell benchmark — held to the terminal date.")
elif dip_with_exit["triggered"]:
    st.success(
        f"**Rule triggered on {dip_with_exit['exit_date'].date()}** — "
        f"{dip_with_exit['hold_days']:,} days after the first purchase on "
        f"{dip_with_exit['first_deployment'].date()}. Proceeds then held at "
        f"{cash_rate_pct:.2f}% p.a. to {terminal_date.date()}."
    )
else:
    st.warning(
        "**Rule not triggered.** The condition was never met, so this is identical "
        "to never selling. A zero difference below is the absence of a trade, not a "
        "judgement about the rule."
    )

# ---------------------------------------------------------------------------
# The three comparisons the delta could mean
# ---------------------------------------------------------------------------
st.markdown(f"**All figures valued on {terminal_date.date()}**")

strategy_wealth = dip_with_exit["terminal_wealth"]
c1, c2, c3 = st.columns(3)

with c1:
    verdict, diff = compare_outcomes(
        strategy_wealth, dca_no_exit["terminal_wealth"],
        "Dip + exit", "Monthly DCA",
    )
    st.metric(
        "vs monthly DCA",
        fmt_currency(dca_no_exit["terminal_wealth"]),
        delta=fmt_delta(diff, dca_no_exit["terminal_wealth"]),
    )
    st.caption(verdict)

with c2:
    verdict, diff = compare_outcomes(
        strategy_wealth, savings_hold["terminal_wealth"],
        "Dip + exit", "Savings only",
    )
    st.metric(
        "vs savings only",
        fmt_currency(savings_hold["terminal_wealth"]),
        delta=fmt_delta(diff, savings_hold["terminal_wealth"]),
    )
    st.caption(verdict)

with c3:
    verdict, diff = compare_outcomes(
        strategy_wealth, dip_no_exit["terminal_wealth"],
        "Dip + exit", "Same entry, no exit",
    )
    st.metric(
        "vs same entry, no exit",
        fmt_currency(dip_no_exit["terminal_wealth"]),
        delta=fmt_delta(diff, dip_no_exit["terminal_wealth"]),
    )
    st.caption(verdict)

st.caption(
    "Each card shows the comparison baseline, with the exit strategy's difference "
    "against it. Differences carry cents and a relative figure, so a small but real "
    "gap is never displayed as zero."
)

# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------
st.divider()
detail = pd.DataFrame(
    [
        {
            "Policy": "Monthly DCA (never sell)",
            "Terminal wealth": dca_no_exit["terminal_wealth"],
            "Investment return (NAV)": dca_no_exit["investment_return"],
            "Exit date": "—",
            "Hold days": dca_no_exit["hold_days"],
        },
        {
            "Policy": "Savings only",
            "Terminal wealth": savings_hold["terminal_wealth"],
            "Investment return (NAV)": savings_hold["investment_return"],
            "Exit date": "—",
            "Hold days": savings_hold["hold_days"],
        },
        {
            "Policy": f"Dip entry at {entry_threshold:.0%} (never sell)",
            "Terminal wealth": dip_no_exit["terminal_wealth"],
            "Investment return (NAV)": dip_no_exit["investment_return"],
            "Exit date": "—",
            "Hold days": dip_no_exit["hold_days"],
        },
        {
            "Policy": f"Dip entry + {exit_type_label}",
            "Terminal wealth": dip_with_exit["terminal_wealth"],
            "Investment return (NAV)": dip_with_exit["investment_return"],
            "Exit date": (
                str(dip_with_exit["exit_date"].date())
                if dip_with_exit["exit_date"] is not None
                else "Not triggered"
            ),
            "Hold days": dip_with_exit["hold_days"],
        },
    ]
)
st.dataframe(
    detail,
    width="stretch",
    hide_index=True,
    column_config={
        "Terminal wealth": st.column_config.NumberColumn(format="%.2f"),
        "Investment return (NAV)": st.column_config.NumberColumn(format="%.2f%%"),
    },
)
st.caption(
    "Investment return is measured on flow-adjusted NAV over the holding period, "
    "so it excludes the effect of contributions."
)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
st.subheader("Wealth over time")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dca_ledger.index, y=dca_ledger["total_wealth"],
    name="Monthly DCA", line=dict(color=GREEN, width=2),
))
fig.add_trace(go.Scatter(
    x=dip_ledger.index, y=dip_ledger["total_wealth"],
    name=f"Dip entry at {entry_threshold:.0%}", line=dict(color=ORANGE, width=2),
))
if dip_with_exit["triggered"] and dip_with_exit["exit_date"] is not None:
    ed = dip_with_exit["exit_date"]
    fig.add_trace(go.Scatter(
        x=[ed], y=[dip_with_exit["exit_wealth"]],
        mode="markers", name=f"Exit: {exit_type_label}",
        marker=dict(symbol="x", size=14, color=RED, line=dict(width=2)),
    ))
    fig.add_vline(
        x=ed, line_dash="dash", line_color=RED,
        annotation_text=f"Exit ({exit_type_label})", annotation_font_color=RED,
    )
apply_chart_layout(fig, title="Wealth curves (EUR)")
fig.update_layout(xaxis_title="Date", yaxis_title="Wealth (EUR)", height=420)
st.plotly_chart(fig, width="stretch")

st.subheader("Investment performance (flow-adjusted NAV)")
st.caption(
    "Exit rules are evaluated on these curves, not on the wealth curves above. "
    "Contributions change the unit count, not the unit price, so a deposit moves "
    "wealth without moving NAV."
)
fig_nav = go.Figure()
for label, ledger, colour in (
    ("Monthly DCA", dca_ledger, GREEN),
    (f"Dip entry at {entry_threshold:.0%}", dip_ledger, ORANGE),
):
    nav = ledger_nav(ledger)
    fig_nav.add_trace(go.Scatter(
        x=nav.index, y=nav.values, name=label, line=dict(color=colour, width=2)
    ))
if exit_type == ExitType.TRAILING_STOP:
    fig_nav.add_hline(
        y=0.0, line_width=0,
        annotation_text=f"Stop triggers {trailing_stop:.0%} below the NAV peak",
    )
apply_chart_layout(fig_nav, title="Unitized NAV (base 1.0)")
fig_nav.update_layout(xaxis_title="Date", yaxis_title="NAV", height=360)
st.plotly_chart(fig_nav, width="stretch")

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
with st.expander("Exit rule definitions"):
    st.markdown(
        """
        | Rule | Trigger | Measured on |
        |------|---------|-------------|
        | **Never sell** | Terminal date | — |
        | **Target return** | NAV gain reaches the target | Unitized NAV since first deployment |
        | **Time-based hold** | N years after the first deployment | Calendar, from the purchase date |
        | **Trailing stop** | NAV falls X% from its peak since entry | Unitized NAV since first deployment |

        Each rule fires at most once. After an exit the proceeds are held at the
        savings rate to the common terminal date, so exit and never-sell are always
        compared on the same day.

        Tax is not modelled. A real exit in a taxable account would realise gains.
        """
    )

data_source_caption(source=data_source, as_of=as_of_date, is_total_return=True, currency="EUR")
