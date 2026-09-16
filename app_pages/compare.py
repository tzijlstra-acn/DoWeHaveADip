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
from dipdca.data.providers.ecb_fx import EcbFxProvider  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from dipdca.models import DeploymentTier, SimulationParams  # noqa: E402
from dipdca.quant.backtest import (  # noqa: E402
    run_ath_deployment,
    run_dca,
    run_savings_only,
)
from dipdca.quant.drawdown import drawdown  # noqa: E402
from dipdca.quant.fx import instrument_to_base_currency  # noqa: E402
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
from ui.formatting import (  # noqa: E402
    compare_outcomes,
    fmt_currency,
    fmt_delta,
    fmt_pct,
)
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
    # Only controls the ATH engine actually consumes are exposed here. Max-wait,
    # emergency buffer and deployment spreading belong to the deprecated
    # wait-for-dip engine; showing them would imply they change these results.
    adv1, adv2 = st.columns(2)
    with adv1:
        start_date = st.date_input("History start", value=datetime.date(2015, 1, 1))
        end_date = st.date_input("History end", value=datetime.date(2024, 12, 31))
        cash_rate_pct = st.slider(
            "Savings rate on waiting cash (% p.a.)",
            min_value=0.0, max_value=8.0, value=2.5, step=0.25,
            help="Annual interest rate earned on cash held by the ATH-deployment strategy.",
        )
        cash_rate_override = cash_rate_pct / 100.0
    with adv2:
        fixed_fee = st.number_input("Fixed fee (EUR)", min_value=0.0, value=0.0, step=0.5)
        pct_fee = st.slider("% fee (bps)", min_value=0, max_value=100, value=10) / 10_000.0
        slippage = st.slider(
            "Slippage (bps)", min_value=0, max_value=50, value=10,
            help="Applied per trade in addition to the fee. Previously a hidden "
                 "0.1% default, so setting the fee to zero did not make trading free.",
        ) / 10_000.0

st.caption(
    f"Assumptions: {pct_fee * 10_000:.0f} bps fee + {slippage * 10_000:.0f} bps "
    f"slippage per trade, {fmt_currency(fixed_fee, decimals=2)} fixed fee, "
    f"{cash_rate_pct:.2f}% p.a. on waiting cash, contributions invested at the "
    "last trading close of each month."
)

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

# The opening ATH must come from history strictly BEFORE simulation_start.
# Fetching only the evaluation window and taking .max() leaks future data:
# day 1 would be compared against a peak reached years later.
BENCHMARK_LOOKBACK_YEARS = 20
history_start = start_date - datetime.timedelta(days=365 * BENCHMARK_LOOKBACK_YEARS)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

with st.spinner(f"Loading {selected_name} data..."):
    try:
        # Always load the investable instrument (ETF)
        _inst_result = svc.get_history(etf_symbol, history_start, end_date)
        instrument_full = _inst_result.frame

        # Load benchmark index for ATH / drawdown signal when available
        if has_benchmark:
            _bm_result = svc.get_history(index_symbol, history_start, end_date)
            benchmark_full: pd.DataFrame | None = _bm_result.frame
            freshness_caption(_bm_result.freshness)
        else:
            benchmark_full = None
            freshness_caption(_inst_result.freshness)

    except LiveDataUnavailable as exc:
        live_data_error(exc, context=selected_name)
        st.stop()

# ---------------------------------------------------------------------------
# FX conversion — instrument prices to base currency (EUR) when needed.
# The benchmark index is NEVER converted; its drawdown stays in the native
# published index level.
# ---------------------------------------------------------------------------
_quote_currency = asset_cfg.get("quote_currency", "EUR").upper()
_base_currency = "EUR"
_fx_label = ""
if _quote_currency != _base_currency:
    _fx_label = f" (converted {_quote_currency}→{_base_currency} via ECB rates)"
    try:
        _ecb = EcbFxProvider()
        _ecb_rates = _ecb.get_rates([_quote_currency], history_start, end_date)
        instrument_full = instrument_to_base_currency(
            instrument_full, _quote_currency, _base_currency, _ecb_rates
        )
        st.caption(
            f"Instrument prices converted from {_quote_currency} to {_base_currency} "
            "using ECB reference rates. "
            "The benchmark index drawdown remains in its native index level."
        )
    except Exception as _fx_exc:
        st.warning(
            f"Could not fetch ECB FX rates for {_quote_currency}→{_base_currency}: {_fx_exc}. "
            f"Results are in the native instrument currency ({_quote_currency}), "
            "not EUR. Do not treat them as EUR wealth."
        )
        _fx_label = f" ⚠ native {_quote_currency} (FX unavailable)"

price_df = instrument_full.loc[start_ts:end_ts]

# Seed the ATH from pre-simulation history only. The engine must never see
# the evaluation window's own maximum as its opening peak.
_ath_source = benchmark_full if benchmark_full is not None else instrument_full
_pre_start = _ath_source.loc[_ath_source.index < start_ts]
initial_ath: float | None = (
    float(_pre_start["adj_close"].max()) if not _pre_start.empty else None
)
benchmark_history_first = _ath_source.index.min() if len(_ath_source) else None
ath_is_complete = not _pre_start.empty

benchmark_df = benchmark_full.loc[start_ts:end_ts] if benchmark_full is not None else None

if len(price_df) < 10:
    st.error("Not enough data in the selected date range. Try a wider range.")
    st.stop()

if not ath_is_complete:
    st.info(
        f"No benchmark history available before {start_date}. The opening all-time high "
        "will be established from the first day of the evaluation window, so early "
        "drawdowns are measured against the highest close observed since data begins."
    )
elif benchmark_history_first is not None:
    st.caption(
        f"Opening ATH seeded from history before {start_date}: "
        f"**{initial_ath:,.2f}** "
        f"(benchmark data begins {benchmark_history_first.date()})"
    )

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
        contribution_timing="month_end",
        dip_threshold=-0.10,      # required by the model; unused by the ATH engine
        fixed_fee=fixed_fee,
        pct_fee=pct_fee,
        slippage=slippage,
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
    ath_date = signal_series.idxmax()
    days_since_high = (signal_series.index[-1] - ath_date).days
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

    if tiers_valid and today_tiers:
        # Tier classification.
        #   reached          : drawdown is at or deeper than the tier threshold
        #   unreached_deeper : tier threshold is deeper than the current drawdown
        # At -20% with tiers -15/-25/-35:
        #   reached = [-15], unreached_deeper = [-25, -35]
        # Thresholds are negative, so "deeper" means more negative.
        reached = [t for t in today_tiers if current_dd <= t.drawdown_threshold]
        unreached_deeper = [t for t in today_tiers if t.drawdown_threshold < current_dd]

        deepest_reached = (
            min(reached, key=lambda t: t.drawdown_threshold) if reached else None
        )
        next_deeper = (
            max(unreached_deeper, key=lambda t: t.drawdown_threshold)
            if unreached_deeper
            else None
        )

        st.divider()
        st.markdown("**Where you stand in the tier schedule**")

        # The current drawdown alone cannot reveal whether the investor actually
        # executed the shallower tiers, so episode state must be supplied.
        already_deployed = st.number_input(
            "Principal already deployed during this ATH episode (EUR)",
            min_value=0.0,
            max_value=10_000_000.0,
            value=0.0,
            step=100.0,
            help=(
                "How much you have already invested since the benchmark last set an "
                "all-time high. Required to compute what is still owed at the tier "
                "you have reached — the drawdown alone cannot tell us."
            ),
        )

        eligible_capital = float(cash_available) + already_deployed

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            if deepest_reached is not None:
                st.error(
                    f"**{deepest_reached.drawdown_threshold:.0%} tier is reached** — "
                    f"drawdown {current_dd:.1%} is at or below "
                    f"{deepest_reached.drawdown_threshold:.0%}. Cumulative target: "
                    f"**{deepest_reached.cumulative_deployment_fraction:.0%}** of eligible capital."
                )
            else:
                st.success(
                    "No tier reached yet. The benchmark has not fallen far enough to "
                    "trigger the shallowest threshold."
                )

            if next_deeper is not None:
                gap = next_deeper.drawdown_threshold - current_dd
                st.info(
                    f"Next deeper tier: **{next_deeper.drawdown_threshold:.0%}** "
                    f"(target {next_deeper.cumulative_deployment_fraction:.0%}) — "
                    f"needs a further **{abs(gap):.1%}** decline from here."
                )
            else:
                st.caption("All configured tiers have been reached at this drawdown.")

        with col_s2:
            target_fraction = (
                deepest_reached.cumulative_deployment_fraction if deepest_reached else 0.0
            )
            target_principal = target_fraction * eligible_capital
            deploy_now = min(
                float(cash_available),
                max(0.0, target_principal - already_deployed),
            )

            st.metric(
                "Deploy now",
                fmt_currency(deploy_now),
                help=(
                    f"Target {target_fraction:.0%} of {fmt_currency(eligible_capital)} "
                    f"eligible capital = {fmt_currency(target_principal)}, "
                    f"less {fmt_currency(already_deployed)} already deployed, "
                    f"capped at {fmt_currency(float(cash_available))} available cash."
                ),
            )

            if next_deeper is not None:
                next_target = next_deeper.cumulative_deployment_fraction * eligible_capital
                held_back = max(0.0, float(cash_available) - deploy_now)
                reserve_for_next = min(
                    held_back, max(0.0, next_target - target_principal)
                )
                st.metric(
                    f"Reserved for {next_deeper.drawdown_threshold:.0%}",
                    fmt_currency(reserve_for_next),
                    help="Cash held back for the next deeper tier if the decline continues.",
                )

            if cash_available <= 0:
                st.caption(
                    "No cash entered above, so nothing can deploy. Enter your waiting "
                    "cash in the **Cash to deploy** field to size the trade."
                )

    # Drawdown chart
    st.divider()
    st.subheader("Drawdown history")
    fig_dd = drawdown_chart(dd_series, title=f"{selected_name} — Drawdown from High")
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
                # ATH deployment — uses benchmark index for signal (or ETF if no index).
                # initial_ath is seeded from pre-simulation history in both branches;
                # passing the evaluation window's own max would leak future data.
                ath_result, ath_ledger = run_ath_deployment(
                    instrument_data=price_df,
                    params=params,
                    tiers=hist_tiers,
                    benchmark_data=benchmark_df if has_benchmark else price_df,
                    initial_ath=initial_ath,
                )

                # Baseline comparisons
                dca_result, dca_ledger = run_dca(price_df, params)
                savings_result, _ = run_savings_only(price_df, params)

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.stop()

        # Conclusion. A flat EUR 1 tie band hid real differences on small
        # portfolios and called genuine gaps ties; the tolerance is relative and
        # the figure carries cents plus a percentage.
        verdict, diff = compare_outcomes(
            ath_result.ending_wealth,
            dca_result.ending_wealth,
            "ATH dip deployment",
            "Monthly DCA",
        )
        if verdict == "Effectively equal":
            conclusion = (
                "Both strategies ended at effectively the same wealth over this "
                f"period — a difference of {fmt_delta(diff, dca_result.ending_wealth)}."
            )
            conclusion_color = NEUTRAL
        else:
            conclusion = (
                f"{verdict.replace(' ahead', '')} came out ahead by "
                f"{fmt_delta(abs(diff), dca_result.ending_wealth)} in this period. "
                "Past outcomes do not predict future results."
            )
            conclusion_color = POSITIVE if diff > 0 else WARNING

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
                dd_series,
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
