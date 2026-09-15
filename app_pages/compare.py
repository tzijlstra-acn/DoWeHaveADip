"""Compare choices — ATH-based dip deployment vs DCA vs savings.

Tab 1: Decision today — current benchmark drawdown, tier state, amount to deploy.
Tab 2: Historical ATH episodes — backtest DCA vs ATH-deployment vs savings.
"""

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
from dipdca.models import DeploymentTier, SimulationParams  # noqa: E402
from dipdca.quant.backtest import (  # noqa: E402
    run_ath_deployment,
    run_dca,
    run_savings_only,
)
from dipdca.quant.drawdown import drawdown  # noqa: E402
from ui.charts import drawdown_chart, plot_strategy_wealth  # noqa: E402
from ui.components import (  # noqa: E402
    conclusion_banner,
    freshness_caption,
    live_data_error,
    page_header,
    strategy_comparison_cards,
    strategy_metrics,
)
from ui.copy import DISCLAIMER_SHORT, LABEL_DCA  # noqa: E402
from ui.design_tokens import NEUTRAL, POSITIVE, WARNING  # noqa: E402
from ui.formatting import fmt_currency, fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "SETTLE THE TIMING DEBATE",
    "Monthly DCA versus cash-on-dip — identical external cash flows, same market.",
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
_assets_cfg = load_assets_config()
# Build a dict of display_name → full asset config dict
_asset_map = {a["display_name"]: a for a in _assets_cfg}
_asset_names = list(_asset_map.keys())
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

asset_cfg = _asset_map[selected_name]
etf_symbol = asset_cfg["etf_symbol"]
index_symbol = asset_cfg.get("index_symbol")

with st.expander("Advanced assumptions"):
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        max_wait_months = st.slider("Max wait (months, DCA baseline only)", min_value=1, max_value=60, value=24)
    with adv2:
        cash_buffer = st.slider("Emergency buffer (months)", 0, 24, 0)
    with adv3:
        start_date = st.date_input("History start", value=datetime.date(2015, 1, 1))
        end_date = st.date_input("History end", value=datetime.date(2024, 12, 31))
        fixed_fee = st.number_input("Fixed fee (EUR)", min_value=0.0, value=0.0, step=0.5)
        pct_fee = st.slider("% fee (bps)", min_value=0, max_value=100, value=10) / 10_000.0
        cash_rate_pct = st.slider(
            "Savings rate on waiting cash (% p.a.)",
            min_value=0.0, max_value=8.0, value=2.5, step=0.25,
            help="Annual interest rate earned on cash held by the ATH-deployment strategy.",
        )
        cash_rate_override = cash_rate_pct / 100.0

# ---------------------------------------------------------------------------
# Warn early if no benchmark index configured
# ---------------------------------------------------------------------------
has_benchmark = index_symbol is not None

if not has_benchmark:
    st.warning(
        f"**{selected_name}** has no configured benchmark index (`index_symbol` is null in "
        "assets.yaml). ATH-based analysis is unavailable — the ETF price will be used as "
        "its own benchmark, which may be misleading. Results are labelled as ETF-proxy mode."
    )

# ---------------------------------------------------------------------------
# Load market data — ETF (instrument) and benchmark index separately
# ---------------------------------------------------------------------------
svc = get_market_data_service()

with st.spinner(f"Loading {selected_name} data..."):
    try:
        # Always load the investable instrument (ETF)
        _inst_result = svc.get_history(etf_symbol, start_date, end_date)
        price_df = _inst_result.frame

        # Load benchmark index for ATH / drawdown signal when available
        if has_benchmark:
            _bm_result = svc.get_history(index_symbol, start_date, end_date)
            benchmark_df: pd.DataFrame | None = _bm_result.frame
            initial_ath = float(benchmark_df["adj_close"].max())
            freshness_caption(_bm_result.freshness)
        else:
            benchmark_df = None
            initial_ath = None
            freshness_caption(_inst_result.freshness)

    except LiveDataUnavailable as exc:
        live_data_error(exc, context=selected_name)
        st.stop()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
price_df = price_df.loc[start_ts:end_ts]
if benchmark_df is not None:
    benchmark_df = benchmark_df.loc[start_ts:end_ts]

if len(price_df) < 10:
    st.error("Not enough data in the selected date range. Try a wider range.")
    st.stop()

# ---------------------------------------------------------------------------
# Derived benchmark signal series for "Decision today"
# ---------------------------------------------------------------------------
signal_df = benchmark_df if benchmark_df is not None else price_df
signal_series = signal_df["adj_close"].dropna()
dd_series = drawdown(signal_series)
current_dd = float(dd_series.iloc[-1])
signal_label = index_symbol if has_benchmark else f"{etf_symbol} (ETF proxy)"

# ---------------------------------------------------------------------------
# Build SimulationParams
# ---------------------------------------------------------------------------
try:
    params = SimulationParams(
        monthly_contribution=float(monthly_contribution),
        payday=25,
        initial_investment=float(initial_investment),
        initial_cash_reserve=float(cash_available),
        start_date=start_date,
        end_date=end_date,
        dip_threshold=-0.10,      # kept for run_dca compatibility; not used by ATH engine
        max_wait_months=max_wait_months,
        fixed_fee=fixed_fee,
        pct_fee=pct_fee,
        deployment_pct=1.0,
        cash_buffer_months=cash_buffer,
        deploy_spread_months=1,
        cash_rate_override=cash_rate_override if cash_rate_override > 0 else None,
    )
except Exception as exc:
    st.error(f"Invalid parameters: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Default tier schedule
# ---------------------------------------------------------------------------
default_tiers_df = pd.DataFrame({
    "Drawdown threshold (%)": [-15, -25, -35],
    "Total savings deployed by this level (%)": [25, 60, 100],
})

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_today, tab_history = st.tabs(["Decision today", "Historical ATH episodes"])

# ===========================================================================
# Tab 1: Decision today
# ===========================================================================
with tab_today:
    st.subheader("Current market signal")

    # Current benchmark state
    peak_series = signal_series.expanding().max()
    at_peak = signal_series >= peak_series
    days_since_high = 0
    for _i in range(len(at_peak) - 1, -1, -1):
        if at_peak.iloc[_i]:
            break
        days_since_high += 1

    # ATH date
    ath_date = signal_series.idxmax()
    ath_val = float(signal_series.max())

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(
            label=f"Current drawdown ({signal_label})",
            value=fmt_pct(current_dd),
            help="Drawdown of the benchmark index from its all-time high.",
        )
    with col_m2:
        st.metric(
            label="All-time high date",
            value=str(ath_date.date()) if hasattr(ath_date, "date") else str(ath_date),
            help=f"Benchmark ATH was {ath_val:,.2f}",
        )
    with col_m3:
        st.metric("Calendar days from ATH", f"{days_since_high:,}")

    # Tier configuration
    st.subheader("Configure your deployment tiers")
    tiers_df = st.data_editor(
        default_tiers_df,
        num_rows="dynamic",
        width="stretch",
        key="today_tiers",
        column_config={
            "Drawdown threshold (%)": st.column_config.NumberColumn(
                "Drawdown threshold (%)", min_value=-99, max_value=-1, step=1
            ),
            "Total savings deployed by this level (%)": st.column_config.NumberColumn(
                "Total savings deployed by this level (%)", min_value=1, max_value=100, step=1
            ),
        },
    )
    st.caption(
        "**Cumulative fractions.** At −15%: deploy 25% of your waiting cash.  "
        "At −25%: deploy a further 35% (60% total).  At −35%: deploy the remaining 40% (100% total)."
    )

    # Parse tiers and compute next-tier state
    try:
        today_tiers = sorted(
            [
                DeploymentTier(
                    drawdown_threshold=float(row["Drawdown threshold (%)"]) / 100.0,
                    cumulative_deployment_fraction=float(
                        row["Total savings deployed by this level (%)"]
                    ) / 100.0,
                )
                for _, row in tiers_df.iterrows()
            ],
            key=lambda t: t.drawdown_threshold,
            reverse=True,
        )
        DeploymentTier.validate_schedule(today_tiers)
        tiers_valid = True
    except Exception as exc:
        st.error(f"Invalid tier configuration: {exc}")
        tiers_valid = False
        today_tiers = []

    if tiers_valid and today_tiers and cash_available > 0:
        # Find the next tier that would fire
        next_tier = next(
            (t for t in today_tiers if current_dd <= t.drawdown_threshold), None
        )
        already_fired = [t for t in today_tiers if current_dd > t.drawdown_threshold]

        st.divider()
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            if next_tier is None:
                st.success("All configured tiers have been triggered by the current drawdown.")
            else:
                dist = current_dd - next_tier.drawdown_threshold  # negative = still to go
                if current_dd <= next_tier.drawdown_threshold:
                    st.error(
                        f"**{next_tier.drawdown_threshold:.0%} tier would trigger NOW** "
                        f"(drawdown {current_dd:.1%} ≤ threshold {next_tier.drawdown_threshold:.0%})"
                    )
                else:
                    st.info(
                        f"Next tier: **{next_tier.drawdown_threshold:.0%}** — "
                        f"needs {abs(dist):.1%} more drawdown from here"
                    )

        with col_s2:
            eligible = float(cash_available)
            # Account for already-triggered fractions (cumulative)
            already_deployed_fraction = (
                max((t.cumulative_deployment_fraction for t in already_fired), default=0.0)
            )
            if next_tier is not None:
                incremental_fraction = (
                    next_tier.cumulative_deployment_fraction - already_deployed_fraction
                )
                amount_next = max(0.0, incremental_fraction * eligible)
                st.metric(
                    "Amount at next tier",
                    fmt_currency(amount_next),
                    help=f"Incremental deployment if {next_tier.drawdown_threshold:.0%} tier fires.",
                )
            else:
                st.metric("Amount at next tier", "—")

    # Drawdown chart
    st.divider()
    st.subheader("Drawdown history")
    fig_dd = drawdown_chart(signal_series, title=f"{selected_name} — Drawdown from High")
    st.plotly_chart(fig_dd, width="stretch")

# ===========================================================================
# Tab 2: Historical ATH episodes
# ===========================================================================
with tab_history:
    st.markdown(
        "Compare **DCA** (always invest) vs **ATH dip deployment** (hold cash until a "
        "benchmark drawdown threshold is crossed) over the selected history. "
        "Both strategies receive identical monthly savings flows."
    )

    if not has_benchmark:
        st.warning(
            f"No benchmark index for {selected_name} — running in ETF-proxy mode. "
            "Drawdown signals are derived from the ETF price itself, which may differ "
            "from the underlying index."
        )

    st.subheader("Configure deployment schedule")
    hist_tiers_df = st.data_editor(
        default_tiers_df,
        num_rows="dynamic",
        width="stretch",
        key="hist_tiers",
        column_config={
            "Drawdown threshold (%)": st.column_config.NumberColumn(
                "Drawdown threshold (%)", min_value=-99, max_value=-1, step=1
            ),
            "Total savings deployed by this level (%)": st.column_config.NumberColumn(
                "Total savings deployed by this level (%)", min_value=1, max_value=100, step=1
            ),
        },
    )
    st.caption(
        "**Example:** With €20,000 saved and the 25%/60%/100% schedule:  \n"
        "• At −15% drawdown → invest €5,000 (25% of €20,000)  \n"
        "• At −25% drawdown → invest €7,000 more (total €12,000 = 60%)  \n"
        "• At −35% drawdown → invest €8,000 more (total €20,000 = 100%)"
    )

    if st.button("Run historical comparison", type="primary"):
        # Validate tier config
        try:
            hist_tiers = sorted(
                [
                    DeploymentTier(
                        drawdown_threshold=float(row["Drawdown threshold (%)"]) / 100.0,
                        cumulative_deployment_fraction=float(
                            row["Total savings deployed by this level (%)"]
                        ) / 100.0,
                    )
                    for _, row in hist_tiers_df.iterrows()
                ],
                key=lambda t: t.drawdown_threshold,
                reverse=True,
            )
            DeploymentTier.validate_schedule(hist_tiers)
        except Exception as exc:
            st.error(f"Invalid tier configuration: {exc}")
            st.stop()

        with st.spinner("Running backtests..."):
            try:
                # ATH deployment — uses benchmark index for signal (or ETF if no index)
                if has_benchmark:
                    ath_result, ath_ledger = run_ath_deployment(
                        instrument_data=price_df,
                        params=params,
                        tiers=hist_tiers,
                        benchmark_data=benchmark_df,
                        initial_ath=initial_ath,
                    )
                else:
                    # ETF-proxy mode: use ETF as its own benchmark
                    ath_result, ath_ledger = run_ath_deployment(
                        instrument_data=price_df,
                        params=params,
                        tiers=hist_tiers,
                        benchmark_data=price_df,
                        initial_ath=float(price_df["adj_close"].max()),
                    )

                # Baseline comparisons
                dca_result, dca_ledger = run_dca(price_df, params)
                savings_result, _ = run_savings_only(price_df, params)

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.stop()

        # Conclusion
        diff = ath_result.ending_wealth - dca_result.ending_wealth
        if abs(diff) < 1:
            conclusion = "Both strategies ended at essentially the same wealth over this period."
            conclusion_color = NEUTRAL
        elif diff > 0:
            conclusion = (
                f"ATH dip deployment came out ahead by {fmt_currency(abs(diff))} in this period. "
                "Past outcomes do not predict future results."
            )
            conclusion_color = POSITIVE
        else:
            conclusion = (
                f"Monthly DCA came out ahead by {fmt_currency(abs(diff))} in this period. "
                "Past outcomes do not predict future results."
            )
            conclusion_color = WARNING

        conclusion_banner(conclusion, color=conclusion_color)

        # Comparison cards
        strategy_comparison_cards(dca_result, ath_result)
        st.write("")

        # Metrics
        st.markdown(f"**{LABEL_DCA}**")
        strategy_metrics(dca_result)

        st.markdown("**ATH Dip Deployment**")
        strategy_metrics(ath_result)

        st.markdown("**Savings account** (cash only — no investment)")
        strategy_metrics(savings_result)

        st.divider()

        # Wealth chart
        st.subheader("Wealth over time")
        fig_wealth = plot_strategy_wealth(dca_ledger, ath_ledger, base_currency="EUR")
        st.plotly_chart(fig_wealth, width="stretch")

        # Drawdown chart (benchmark or ETF)
        with st.expander("Drawdown history (signal source)"):
            fig_dd2 = drawdown_chart(
                signal_series,
                title=f"{selected_name} — Drawdown from High ({signal_label})",
            )
            st.plotly_chart(fig_dd2, width="stretch")

        # Deployment events
        deploy_dates = ath_ledger[ath_ledger["deployed"] > 0]
        if not deploy_dates.empty:
            with st.expander(f"ATH deployment events ({len(deploy_dates)} total)"):
                disp = deploy_dates[["deployed", "fees", "benchmark_drawdown"]].copy()
                disp.index = disp.index.date
                disp = disp.rename(columns={
                    "deployed": "Deployed (EUR)",
                    "fees": "Fees (EUR)",
                    "benchmark_drawdown": "Benchmark drawdown at signal",
                })
                st.dataframe(disp, width="stretch")

        # Savings summary
        with st.expander("Savings vs investing summary"):
            s1, s2, s3 = st.columns(3)
            with s1:
                total_saved = float(monthly_contribution) * len(
                    pd.date_range(start_date, end_date, freq="ME")
                )
                st.metric("Total contributions", fmt_currency(total_saved))
            with s2:
                peak_uninvested = float(ath_ledger["cash"].max())
                st.metric("Peak uninvested cash (ATH strategy)", fmt_currency(peak_uninvested))
            with s3:
                wealth = ath_ledger["total_wealth"].replace(0, float("nan"))
                avg_cash_weight = float((ath_ledger["cash"] / wealth).mean())
                st.metric(
                    "Avg cash weight (ATH strategy)",
                    f"{avg_cash_weight:.1%}",
                    help="Average fraction of portfolio held in cash waiting for a dip.",
                )

st.divider()
st.caption(DISCLAIMER_SHORT)
