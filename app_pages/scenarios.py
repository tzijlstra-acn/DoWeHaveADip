"""Historical scenarios — parameter sweep and conditional path bootstrap."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _make_fp(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


from dipdca.config import load_assets_config  # noqa: E402
from dipdca.data.errors import LiveDataUnavailable  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from dipdca.models import SimulationParams  # noqa: E402
from dipdca.quant.monte_carlo import (  # noqa: E402
    conditional_path_bootstrap,
    outperformance_pivot,
    run_parameter_sweep,
    sweep_to_dataframe,
    win_rate_pivot,
)
from ui.charts import plot_fan_chart, plot_sweep_heatmap  # noqa: E402
from ui.components import freshness_caption, live_data_error, page_header  # noqa: E402
from ui.copy import DISCLAIMER_SHORT  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "HISTORICAL SCENARIOS",
    "Past market states with a similar drawdown — what happened next. "
    "Real data, not a prediction.",
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
_assets_cfg = load_assets_config()
_asset_options = {a["display_name"]: a["etf_symbol"] for a in _assets_cfg}
_asset_names = list(_asset_options.keys())
_default_idx = next((i for i, n in enumerate(_asset_names) if "S&P 500" in n), 0)

col_a, col_b, col_c = st.columns([3, 2, 2])
with col_a:
    selected_name = st.selectbox("Market or ETF", _asset_names, index=_default_idx)
with col_b:
    monthly_contribution = st.number_input("Monthly amount (EUR)", min_value=10, max_value=100_000, value=500, step=50)
with col_c:
    cash_available = st.number_input("Cash to deploy (EUR)", min_value=0, max_value=1_000_000, value=0, step=100)

symbol = _asset_options[selected_name]

with st.expander("Advanced settings"):
    adv1, adv2 = st.columns(2)
    with adv1:
        start_date = st.date_input("History start", value=datetime.date(2005, 1, 1))
        end_date = st.date_input("History end", value=datetime.date.today())
        monthly_sim = st.number_input("Monthly contribution in simulation (EUR)", min_value=0, value=int(monthly_contribution), step=50)
    with adv2:
        cp_horizon = st.select_slider("Horizon (months)", options=[6, 12, 18, 24, 36], value=12)
        cp_n_sims = st.slider("Number of draws", min_value=50, max_value=500, value=200, step=50)
        cp_deploy_pcts_raw = st.multiselect(
            "Deploy fractions to show",
            options=["0%", "25%", "50%", "75%", "100%"],
            default=["0%", "50%", "100%"],
        )
        cp_deploy_pcts = [float(d.replace("%", "")) / 100.0 for d in cp_deploy_pcts_raw] or [0.0, 0.5, 1.0]
        window_years = st.select_slider("Rolling window size (years)", options=[5, 10, 15, 20], value=10)

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
price_series = price_df["adj_close"].dropna()

if len(price_series) < 60:
    st.error("Not enough history for scenario analysis. Try an earlier start date.")
    st.stop()

from dipdca.quant.drawdown import drawdown  # noqa: E402

dd_series = drawdown(price_series)
current_dd = float(dd_series.iloc[-1])

st.info(
    f"**{selected_name}** is currently {current_dd:.1%} from its previous high. "
    f"Scenarios below are based on {len(price_series):,} historical trading days "
    f"({start_date} – {end_date})."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_paths, tab_sweep = st.tabs(["Historical scenarios", "Threshold sweep"])

# ---------------------------------------------------------------------------
# Tab 1: Conditional path bootstrap
# ---------------------------------------------------------------------------
with tab_paths:
    st.markdown(
        "The chart below shows the distribution of historical outcomes when "
        f"**{selected_name}** was at a similar drawdown level ({current_dd:.1%} ± 5%). "
        f"Outcomes are shown over a **{cp_horizon}-month horizon**."
    )

    _cp_fp_params = {
        "symbol": symbol, "start": str(start_date), "end": str(end_date),
        "current_dd": round(current_dd, 4), "monthly": int(monthly_sim),
        "cash": int(cash_available), "horizon": cp_horizon,
        "n_sims": cp_n_sims, "deploys": sorted(cp_deploy_pcts),
    }
    _cp_fp = _make_fp(_cp_fp_params)

    _stored_cp = st.session_state.get("scenarios_cp_sims")
    if _stored_cp and isinstance(_stored_cp, dict) and _stored_cp.get("fp") == _cp_fp:
        cp_sims = _stored_cp["data"]
    else:
        if _stored_cp:
            st.info("Parameters changed — click Run to update results.")
        cp_sims = []

    if st.button("Run historical scenario analysis", type="primary"):
        with st.spinner(f"Finding historical periods similar to {current_dd:.1%} drawdown..."):
            monthly_prices = price_series.resample("ME").last().dropna()
            cp_sims = conditional_path_bootstrap(
                prices=monthly_prices,
                current_drawdown=current_dd,
                deploy_pcts=cp_deploy_pcts,
                monthly_contribution=float(monthly_sim),
                cash_accumulated=float(cash_available),
                horizon_months=cp_horizon,
                n_simulations=cp_n_sims,
                seed=42,
            )
            st.session_state["scenarios_cp_sims"] = {"fp": _cp_fp, "data": cp_sims}

    if cp_sims:
        # Count matching periods
        tol = 0.05
        n_periods = int((
            (dd_series <= current_dd + tol) & (dd_series >= current_dd - tol)
        ).sum())
        st.caption(f"Based on {n_periods} historical periods with similar drawdown.")

        deploy_labels = {
            0.0: "Invest monthly, no lump sum",
            0.25: "Deploy 25% of cash",
            0.5: "Deploy 50% of cash",
            0.75: "Deploy 75% of cash",
            1.0: "Deploy all cash",
        }

        for sim in cp_sims:
            dlabel = deploy_labels.get(sim.deploy_pct, f"Deploy {sim.deploy_pct:.0%}")
            st.subheader(dlabel)
            cols_m = st.columns(3)
            with cols_m[0]:
                st.metric("Worst case (P5)", f"EUR {sim.p5_wealth[-1]:,.0f}",
                          help="5th percentile outcome at end of horizon — 1-in-20 bad draw")
            with cols_m[1]:
                st.metric("Typical (P50)", f"EUR {sim.p50_wealth[-1]:,.0f}")
            with cols_m[2]:
                st.metric("Best case (P95)", f"EUR {sim.p95_wealth[-1]:,.0f}",
                          help="95th percentile outcome at end of horizon — 1-in-20 good draw")
            st.metric(
                "Beat monthly DCA",
                f"{sim.prob_beats_dca:.0%}",
                help="Fraction of historical draws where this strategy ended ahead of monthly DCA",
            )
            fig_fan = plot_fan_chart([sim], currency="EUR", horizon_months=cp_horizon)
            st.plotly_chart(fig_fan, use_container_width=True)

    elif not cp_sims and st.session_state.get("scenarios_cp_sims"):
        st.info(
            "Not enough historical periods found at this drawdown level. "
            "Try a wider date range or a larger tolerance."
        )

# ---------------------------------------------------------------------------
# Tab 2: Threshold sweep
# ---------------------------------------------------------------------------
with tab_sweep:
    st.markdown(
        "Grid search over dip thresholds and deployment fractions. "
        "For each combination, shows the historical win rate vs monthly DCA "
        "across all rolling windows in the dataset."
    )

    st.subheader("Threshold grid")
    threshold_options = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40]
    selected_thresholds_raw = st.multiselect(
        "Dip thresholds",
        options=[f"{t:.0%}" for t in threshold_options],
        default=["-5%", "-10%", "-15%", "-20%", "-25%"],
    )
    thresholds = [float(t.replace("%", "")) / 100.0 for t in selected_thresholds_raw] or threshold_options[:5]

    deploy_options = [0.25, 0.50, 0.75, 1.00]
    selected_deploys_raw = st.multiselect(
        "Deploy fractions",
        options=[f"{d:.0%}" for d in deploy_options],
        default=["25%", "50%", "75%", "100%"],
    )
    deploy_pcts_sweep = [float(d.replace("%", "")) / 100.0 for d in selected_deploys_raw] or deploy_options

    _sweep_fp_params = {
        "symbol": symbol, "start": str(start_date), "end": str(end_date),
        "thresholds": sorted(thresholds), "deploys": sorted(deploy_pcts_sweep),
        "window_years": window_years, "monthly": int(monthly_contribution),
    }
    _sweep_fp = _make_fp(_sweep_fp_params)
    _stored_sweep = st.session_state.get("scenarios_sweep_cache")
    if _stored_sweep and isinstance(_stored_sweep, dict) and _stored_sweep.get("fp") == _sweep_fp:
        sweep_results = _stored_sweep["data"]
    else:
        if _stored_sweep:
            st.info("Parameters changed — click Run to update the sweep.")
        sweep_results = []

    if st.button("Run threshold sweep", type="primary", key="run_sweep"):
        try:
            base_params = SimulationParams(
                monthly_contribution=float(monthly_contribution),
                payday=25,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            st.error(f"Invalid parameters: {exc}")
            st.stop()

        with st.spinner("Running threshold sweep across rolling windows..."):
            sweep_results = run_parameter_sweep(
                prices=price_series,
                base_params=base_params,
                thresholds=thresholds,
                deploy_pcts=deploy_pcts_sweep,
                window_years=window_years,
                step_months=12,
                points_per_year=252,
            )
            st.session_state["scenarios_sweep_cache"] = {"fp": _sweep_fp, "data": sweep_results}

    if sweep_results:
        sweep_df = sweep_to_dataframe(sweep_results)
        st.subheader("Win rate vs monthly DCA")
        st.caption(
            "Fraction of rolling windows where the dip strategy ended ahead of DCA. "
            f"Window size: {window_years} years."
        )
        wr_pivot = win_rate_pivot(sweep_df)
        fig_wr = plot_sweep_heatmap(
            wr_pivot,
            title="Win rate vs DCA (fraction of windows)",
            colorscale="RdYlGn",
            zmin=0.0,
            zmax=1.0,
        )
        st.plotly_chart(fig_wr, use_container_width=True)

        st.subheader("Median outperformance vs DCA")
        op_pivot = outperformance_pivot(sweep_df)
        fig_op = plot_sweep_heatmap(
            op_pivot,
            title="Median outperformance vs DCA",
            colorscale="RdYlGn",
        )
        st.plotly_chart(fig_op, use_container_width=True)

        with st.expander("Raw sweep data"):
            st.dataframe(sweep_df, use_container_width=True)

st.divider()
st.caption(DISCLAIMER_SHORT)
