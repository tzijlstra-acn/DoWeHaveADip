"""Historical scenarios — ATH-episode event study.

Answers the conditional question: given the benchmark index has just fallen X%
below its previous all-time high, what happened next?

The unit of observation is the drawdown episode, not the calendar. Each episode is
anchored to one all-time high, records only the FIRST crossing of each threshold,
and closes only when that same high is recovered. Adjacent crash days therefore
cannot be counted as separate scenarios, and a few winning crashes cannot be
averaged away inside years of cash drag.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _make_fp(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


from dipdca.config import load_assets_config  # noqa: E402
from dipdca.data.errors import LiveDataUnavailable  # noqa: E402
from dipdca.data.providers.ecb_fx import EcbFxProvider  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from dipdca.models import DeploymentTier  # noqa: E402
from dipdca.quant.ath_episodes import (  # noqa: E402
    episodes_to_dataframe,
    find_ath_episodes,
    run_event_study,
    summarise_threshold,
)
from dipdca.quant.drawdown import drawdown  # noqa: E402
from dipdca.quant.episode_bootstrap import bootstrap_win_rate  # noqa: E402
from dipdca.quant.fx import instrument_to_base_currency  # noqa: E402
from ui.components import freshness_caption, live_data_error, page_header  # noqa: E402
from ui.copy import DISCLAIMER_SHORT  # noqa: E402
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "HISTORICAL SCENARIOS",
    "Independent all-time-high episodes — what actually happened after each "
    "threshold was first crossed. Real data, not a prediction.",
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
        "Monthly savings (EUR)", min_value=10, max_value=100_000, value=1000, step=50
    )
with col_c:
    opening_reserve = st.number_input(
        "Savings already accumulated at each ATH (EUR)",
        min_value=0, max_value=1_000_000, value=12_000, step=1_000,
        help="Cash on hand when each episode begins — the capital the timing "
             "decision is made about.",
    )

asset_cfg = _asset_map[selected_name]
etf_symbol = asset_cfg["etf_symbol"]
index_symbol = asset_cfg.get("index_symbol")
has_benchmark = index_symbol is not None

with st.expander("Advanced settings"):
    adv1, adv2 = st.columns(2)
    with adv1:
        start_date = st.date_input("History start", value=datetime.date(2000, 1, 1))
        end_date = st.date_input("History end", value=datetime.date.today())
        cash_rate_pct = st.slider(
            "Savings rate on waiting cash (% p.a.)",
            min_value=0.0, max_value=8.0, value=2.0, step=0.25,
        )
    with adv2:
        threshold_options = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40]
        selected_thresholds_raw = st.multiselect(
            "Thresholds to study",
            options=[f"{t:.0%}" for t in threshold_options],
            default=["-10%", "-15%", "-20%", "-25%", "-30%"],
        )
        thresholds = tuple(
            sorted(
                (float(t.replace("%", "")) / 100.0 for t in selected_thresholds_raw),
                reverse=True,
            )
        ) or (-0.10, -0.15, -0.20, -0.25, -0.30)

        horizon_raw = st.multiselect(
            "Horizons after the ATH",
            options=["12m", "36m", "60m"],
            default=["12m", "36m"],
        )
        horizon_months = tuple(
            sorted(int(h.replace("m", "")) for h in horizon_raw)
        ) or (12,)
        pct_fee = st.slider("% fee (bps)", min_value=0, max_value=100, value=10) / 10_000.0
        slippage = st.slider("Slippage (bps)", min_value=0, max_value=50, value=10) / 10_000.0

st.caption(
    f"Assumptions: {pct_fee * 10_000:.0f} bps fee and {slippage * 10_000:.0f} bps "
    f"slippage per trade, {cash_rate_pct:.2f}% p.a. on waiting cash, "
    "contributions invested at the last trading close of each month."
)

if not has_benchmark:
    st.warning(
        f"**{selected_name}** has no configured benchmark index (`index_symbol` is "
        "null in assets.yaml), so the ETF price is used as its own drawdown signal. "
        "Results are ETF-proxy mode and may differ from the underlying index."
    )

# ---------------------------------------------------------------------------
# Load market data — benchmark drives the signal, ETF supplies execution prices
# ---------------------------------------------------------------------------
# Load 20 years before start_date so that the opening ATH is a genuine
# pre-window high, not the first bar of the evaluation window.
BENCHMARK_LOOKBACK_YEARS = 20
_history_start = start_date - datetime.timedelta(days=365 * BENCHMARK_LOOKBACK_YEARS)
_start_ts = pd.Timestamp(start_date)

with st.spinner(f"Loading {selected_name} data..."):
    try:
        _inst_result = get_market_data_service().get_history(etf_symbol, start_date, end_date)
        instrument_df = _inst_result.frame

        if has_benchmark:
            _bm_full_result = get_market_data_service().get_history(
                index_symbol, _history_start, end_date
            )
            _bm_full = _bm_full_result.frame
            freshness_caption(_bm_full_result.freshness)
            # Pre-window history: compute the ATH before the evaluation window
            _pre_start = _bm_full.loc[_bm_full.index < _start_ts]
            if len(_pre_start) > 0:
                _initial_ath = float(_pre_start["adj_close"].max())
                _initial_ath_date = pd.Timestamp(_pre_start["adj_close"].idxmax())
            else:
                _initial_ath = None
                _initial_ath_date = None
            # Evaluation window only
            benchmark_df = _bm_full.loc[_bm_full.index >= _start_ts]
        else:
            benchmark_df = instrument_df
            _initial_ath = None
            _initial_ath_date = None
            freshness_caption(_inst_result.freshness)
    except LiveDataUnavailable as exc:
        live_data_error(exc, context=selected_name)
        st.stop()

# FX conversion: instrument execution prices → EUR; benchmark signal unchanged.
_quote_currency = asset_cfg.get("quote_currency", "EUR").upper()
if _quote_currency != "EUR":
    try:
        _ecb = EcbFxProvider()
        _ecb_rates = _ecb.get_rates([_quote_currency], start_date, end_date)
        instrument_df = instrument_to_base_currency(
            instrument_df, _quote_currency, "EUR", _ecb_rates
        )
        st.caption(
            f"Instrument prices converted from {_quote_currency} to EUR "
            "using ECB reference rates. The benchmark drawdown stays in its native index level."
        )
    except Exception as _fx_exc:
        st.warning(
            f"Could not fetch ECB FX rates for {_quote_currency}→EUR: {_fx_exc}. "
            f"Results are in the native instrument currency ({_quote_currency}), not EUR."
        )

if len(instrument_df) < 60 or len(benchmark_df) < 60:
    st.error("Not enough history for episode analysis. Try an earlier start date.")
    st.stop()

signal_label = index_symbol if has_benchmark else f"{etf_symbol} (ETF proxy)"
bm_series = benchmark_df["adj_close"].dropna()
current_dd = float(drawdown(bm_series).iloc[-1])

# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
episodes = find_ath_episodes(
    bm_series,
    thresholds=thresholds,
    initial_ath=_initial_ath,
    initial_ath_date=_initial_ath_date,
)
recovered = [e for e in episodes if not e.is_censored]
censored = [e for e in episodes if e.is_censored]

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(f"Current drawdown ({signal_label})", fmt_pct(current_dd))
with m2:
    st.metric("Independent episodes", f"{len(episodes)}")
with m3:
    st.metric("Recovered", f"{len(recovered)}")
with m4:
    st.metric(
        "Still open",
        f"{len(censored)}",
        help="Episodes whose all-time high was never reclaimed within the data. "
             "Reported separately — not counted as recoveries.",
    )

if not episodes:
    st.info(
        f"No episode in this window reached {max(thresholds):.0%}. "
        "Try an earlier start date or a shallower threshold."
    )
    st.stop()

tab_summary, tab_episodes = st.tabs(["Threshold summary", "Episode log"])

# ---------------------------------------------------------------------------
# Tab 1: per-threshold event study
# ---------------------------------------------------------------------------
with tab_summary:
    st.markdown(
        "For each threshold, every independent episode that crossed it is one "
        "observation. Percentiles are shown rather than an average, because the "
        "distribution is skewed and a single mean hides the cases of interest."
    )

    default_tiers_df = pd.DataFrame({
        "Drawdown threshold (%)": [-15, -25, -35],
        "Total savings deployed by this level (%)": [25, 60, 100],
    })
    st.subheader("Tiered schedule (for the 'ATH tiered' policy)")
    tiers_df = st.data_editor(
        default_tiers_df,
        num_rows="dynamic",
        width="stretch",
        key="scenario_tiers",
        column_config={
            "Drawdown threshold (%)": st.column_config.NumberColumn(
                "Drawdown threshold (%)", min_value=-99, max_value=-1, step=1
            ),
            "Total savings deployed by this level (%)": st.column_config.NumberColumn(
                "Total savings deployed by this level (%)", min_value=1, max_value=100, step=1
            ),
        },
    )

    policy = st.radio(
        "Policy to compare against month-end DCA",
        options=["ATH all-in", "ATH tiered", "Savings only"],
        horizontal=True,
    )
    horizon_label = st.radio(
        "Measured at",
        options=[f"{m}m" for m in horizon_months] + ["recovery"],
        horizontal=True,
        help="'recovery' measures at the date the anchor all-time high was "
             "reclaimed, and is available only for recovered episodes.",
    )

    _fp = _make_fp({
        "etf": etf_symbol, "index": index_symbol,
        "start": str(start_date), "end": str(end_date),
        "monthly": int(monthly_contribution), "reserve": int(opening_reserve),
        "thresholds": list(thresholds), "horizons": list(horizon_months),
        "tiers": tiers_df.to_dict(), "rate": cash_rate_pct,
        "fee": pct_fee, "slip": slippage,
    })
    _stored = st.session_state.get("scenarios_event_study")
    if _stored and isinstance(_stored, dict) and _stored.get("fp") == _fp:
        studies = _stored["data"]
    else:
        if _stored:
            st.info("Parameters changed — run the study again to update results.")
        studies = []

    if st.button("Run episode event study", type="primary"):
        try:
            tiers = sorted(
                [
                    DeploymentTier(
                        drawdown_threshold=float(r["Drawdown threshold (%)"]) / 100.0,
                        cumulative_deployment_fraction=float(
                            r["Total savings deployed by this level (%)"]
                        ) / 100.0,
                    )
                    for _, r in tiers_df.iterrows()
                ],
                key=lambda t: t.drawdown_threshold,
                reverse=True,
            )
            DeploymentTier.validate_schedule(tiers)
        except Exception as exc:
            st.error(f"Invalid tier configuration: {exc}")
            st.stop()

        with st.spinner(
            f"Studying {len(episodes)} episodes across {len(thresholds)} thresholds..."
        ):
            try:
                studies = run_event_study(
                    instrument=instrument_df,
                    benchmark=benchmark_df,
                    tiers=tiers,
                    monthly_contribution=float(monthly_contribution),
                    opening_reserve=float(opening_reserve),
                    thresholds=thresholds,
                    horizon_months=horizon_months,
                    cash_rate=cash_rate_pct / 100.0 if cash_rate_pct > 0 else None,
                    pct_fee=pct_fee,
                    slippage=slippage,
                    anchor="execution",
                    initial_ath=_initial_ath,
                    initial_ath_date=_initial_ath_date,
                )
                st.session_state["scenarios_event_study"] = {"fp": _fp, "data": studies}
            except Exception as exc:
                st.error(f"Event study failed: {exc}")
                st.stop()

    if studies:
        rows = []
        for th in thresholds:
            s = summarise_threshold(studies, th, horizon_label, policy)
            if not s["episodes"]:
                continue
            def _pct(v):
                return v * 100 if v is not None else None

            # Bootstrap CI on win rate (None → <2 eligible episodes)
            boot = bootstrap_win_rate(
                studies, threshold=th, horizon_label=horizon_label, policy=policy,
                n_boot=1_000, seed=42,
            )
            ci_str = (
                f"[{boot.ci_lower:.0%}, {boot.ci_upper:.0%}]"
                if boot is not None else "n/a"
            )

            rows.append({
                "Threshold": f"{th:.0%}",
                "Episodes": s["episodes"],
                "Still open": s["censored"],
                "% actually traded": _pct(s["pct_episodes_deployed"]),
                "% ahead of DCA": _pct(s["pct_ahead_of_dca"]),
                "95% CI on win rate": ci_str,
                "Median vs DCA": _pct(s["median_vs_dca"]),
                "P10 vs DCA": _pct(s["p10_vs_dca"]),
                "P90 vs DCA": _pct(s["p90_vs_dca"]),
                "Worst vs DCA": _pct(s["worst_vs_dca"]),
                "Cash left idle": _pct(s["median_undeployed_cash"]),
                "% tiered beat all-in": _pct(s["pct_tiered_beat_all_in"]),
                "Median days to recovery": s["median_days_to_recovery"],
            })

        if not rows:
            st.info(
                f"No episode produced a result at the **{horizon_label}** horizon. "
                "Shorter horizons or a wider date range will yield more observations."
            )
        else:
            summary_df = pd.DataFrame(rows)
            st.subheader(f"{policy} vs month-end DCA — measured at {horizon_label}")
            st.dataframe(
                summary_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "% actually traded": st.column_config.NumberColumn(format="%.0f%%"),
                    "% ahead of DCA": st.column_config.NumberColumn(format="%.0f%%"),
                    "Median vs DCA": st.column_config.NumberColumn(format="%.2f%%"),
                    "P10 vs DCA": st.column_config.NumberColumn(format="%.2f%%"),
                    "P90 vs DCA": st.column_config.NumberColumn(format="%.2f%%"),
                    "Worst vs DCA": st.column_config.NumberColumn(format="%.2f%%"),
                    "Cash left idle": st.column_config.NumberColumn(format="%.0f%%"),
                    "% tiered beat all-in": st.column_config.NumberColumn(format="%.0f%%"),
                },
            )
            st.warning(
                "**Read '% actually traded' first.** A threshold deeper than the "
                "horizon reaches may never fire, and a policy that never fires is "
                "just a savings account — which beats DCA in any falling market. "
                "Where that column is low, the result reflects *avoiding* the market "
                "rather than *buying* it well."
            )
            st.caption(
                "Values are fractions relative to month-end DCA over the same window "
                "with identical contributions. 'Still open' counts episodes whose "
                "all-time high was never reclaimed. 'Cash left idle' is the median "
                "share of ending wealth still uninvested. A small episode count means "
                "the percentiles are indicative only."
            )

            with st.expander("Per-episode detail"):
                detail = []
                for s in studies:
                    if s.threshold != thresholds[0]:
                        continue
                    rel = s.relative_to_dca(policy, horizon_label)
                    detail.append({
                        "ATH date": s.episode.ath_date.date(),
                        "Signal date": s.signal_date.date(),
                        "Trough": s.episode.trough_drawdown,
                        "Recovered": not s.episode.is_censored,
                        f"{policy} vs DCA": rel,
                    })
                if detail:
                    st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
                    st.caption(f"Shown for the {thresholds[0]:.0%} threshold.")

# ---------------------------------------------------------------------------
# Tab 2: episode log
# ---------------------------------------------------------------------------
with tab_episodes:
    st.markdown(
        f"Every independent episode found in **{signal_label}**. Each row is one "
        "all-time high, the drawdown that followed, and whether that high was "
        "reclaimed."
    )
    ep_df = episodes_to_dataframe(episodes)
    st.dataframe(
        ep_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Trough drawdown": st.column_config.NumberColumn(format="%.2f%%"),
            "ATH level": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.subheader("First crossings per episode")
    cross_rows = []
    for ep in episodes:
        row: dict[str, object] = {"ATH date": ep.ath_date.date()}
        for th in thresholds:
            dt = ep.first_crossings.get(th)
            row[f"{th:.0%}"] = dt.date() if dt is not None else None
        cross_rows.append(row)
    st.dataframe(pd.DataFrame(cross_rows), width="stretch", hide_index=True)
    st.caption(
        "Only the first crossing of each threshold within an episode is recorded, "
        "so a long crash contributes one observation rather than one per day."
    )

st.divider()
st.caption(DISCLAIMER_SHORT)
