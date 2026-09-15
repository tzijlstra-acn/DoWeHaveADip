"""Threshold Laboratory — Monte Carlo parameter sweep + conditional path simulation."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.data.providers.deep_history import (  # noqa: E402
    DEEP_HISTORY_TICKERS,
    fetch_deep_history,
)
from dipdca.data.providers.yahoo import YahooProvider  # noqa: E402
from dipdca.models import SimulationParams  # noqa: E402
from dipdca.quant.monte_carlo import (  # noqa: E402
    PathSimulation,
    SweepResult,
    conditional_path_bootstrap,
    outperformance_pivot,
    run_parameter_sweep,
    sweep_to_dataframe,
    win_rate_pivot,
)
from ui.charts import plot_fan_chart, plot_sweep_heatmap  # noqa: E402
from ui.components import data_source_caption, page_header, sidebar_simulation_params  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.set_page_config(page_title="Monte Carlo Lab", page_icon="🎲", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "Monte Carlo Lab",
    "Parameter sweep + conditional path bootstrap — because the future is uncertain.",
    "🎲",
)

# ---------------------------------------------------------------------------
# Sidebar — global params
# ---------------------------------------------------------------------------
param_dict = sidebar_simulation_params()

st.sidebar.subheader("Ticker")
symbol = st.sidebar.text_input("ETF Symbol", value="SPY")

st.sidebar.subheader("Monte Carlo Settings")
window_years = st.sidebar.select_slider(
    "Rolling window size",
    options=[5, 10, 15, 20],
    value=10,
    help="Window length for parameter sweep",
)
step_months = st.sidebar.slider(
    "Step between windows (months)",
    min_value=3,
    max_value=24,
    value=12,
    step=3,
)

st.sidebar.subheader("Threshold Grid")
threshold_options = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40]
selected_thresholds = st.sidebar.multiselect(
    "Dip thresholds",
    options=[f"{t:.0%}" for t in threshold_options],
    default=["-5%", "-10%", "-15%", "-20%", "-25%"],
)
thresholds = [float(t.replace("%", "")) / 100.0 for t in selected_thresholds] or threshold_options[:5]

st.sidebar.subheader("Deploy % Grid")
deploy_options = [0.25, 0.50, 0.75, 1.00]
selected_deploys = st.sidebar.multiselect(
    "Deploy fractions",
    options=[f"{d:.0%}" for d in deploy_options],
    default=["25%", "50%", "75%", "100%"],
)
deploy_pcts = [float(d.replace("%", "")) / 100.0 for d in selected_deploys] or deploy_options

# ---------------------------------------------------------------------------
# Load ETF data (shared across Tab 1 and Tab 2)
# ---------------------------------------------------------------------------
with st.spinner(f"Fetching market data for {symbol}..."):
    try:
        provider = YahooProvider()
        price_data = provider.get_price_data(
            symbol, param_dict["start_date"], param_dict["end_date"]
        )
        price_df = price_data.df
        data_source = f"Yahoo Finance ({symbol})"
        as_of_date = price_data.as_of
    except Exception as exc:
        st.error(f"Could not load market data for {symbol}: {exc}")
        st.stop()

start_ts = pd.Timestamp(param_dict["start_date"])
end_ts = pd.Timestamp(param_dict["end_date"])
price_df = price_df.loc[start_ts:end_ts]
price_series = price_df["adj_close"].dropna()

if len(price_series) < 60:
    st.error("Not enough data. Try a wider date range.")
    st.stop()

# Base params for sweep
base_params = SimulationParams(
    monthly_contribution=param_dict["monthly_contribution"],
    payday=param_dict["payday"],
    initial_investment=param_dict["initial_investment"],
    start_date=param_dict["start_date"],
    end_date=param_dict["end_date"],
    dip_threshold=-0.10,
    max_wait_months=param_dict["max_wait_months"],
    fixed_fee=param_dict["fixed_fee"],
    pct_fee=param_dict["pct_fee"],
)


# ---------------------------------------------------------------------------
# Cached sweep function (ETF daily data)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_parameter_sweep(
    price_bytes: bytes,
    monthly: float,
    payday: int,
    initial: float,
    start: object,
    end: object,
    max_wait: int,
    fixed_fee: float,
    pct_fee: float,
    thresholds_t: tuple,
    deploy_pcts_t: tuple,
    window_years_v: int,
    step_months_v: int,
    points_per_year: int = 252,
) -> list[dict]:
    """Run parameter sweep and return serialisable results."""
    df = pd.read_parquet(io.BytesIO(price_bytes))
    prices = df["adj_close"].squeeze()

    base = SimulationParams(
        monthly_contribution=monthly,
        payday=payday,
        initial_investment=initial,
        start_date=start,
        end_date=end,
        dip_threshold=-0.10,
        max_wait_months=max_wait,
        fixed_fee=fixed_fee,
        pct_fee=pct_fee,
    )

    sweep_results = run_parameter_sweep(
        prices=prices,
        base_params=base,
        thresholds=list(thresholds_t),
        deploy_pcts=list(deploy_pcts_t),
        window_years=window_years_v,
        step_months=step_months_v,
        points_per_year=points_per_year,
    )

    # Convert dataclasses to dicts for caching
    return [
        {
            "threshold": r.threshold,
            "deploy_pct": r.deploy_pct,
            "win_rate": r.win_rate,
            "median_outperformance": r.median_outperformance,
            "p10_outperformance": r.p10_outperformance,
            "p90_outperformance": r.p90_outperformance,
            "n_windows": r.n_windows,
            "median_deployments": r.median_deployments,
        }
        for r in sweep_results
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def cached_path_bootstrap(
    price_bytes: bytes,
    current_dd: float,
    deploy_pcts_t: tuple,
    monthly_contribution: float,
    cash_accumulated: float,
    horizon_months: int,
    n_simulations: int,
) -> list[dict]:
    """Run conditional path bootstrap and return serialisable results."""
    df = pd.read_parquet(io.BytesIO(price_bytes))
    prices = df["adj_close"].squeeze()

    sims = conditional_path_bootstrap(
        prices=prices,
        current_drawdown=current_dd,
        deploy_pcts=list(deploy_pcts_t),
        monthly_contribution=monthly_contribution,
        cash_accumulated=cash_accumulated,
        horizon_months=horizon_months,
        n_simulations=n_simulations,
    )

    return [
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
        for s in sims
    ]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_sweep, tab_path, tab_deep = st.tabs(
    ["Parameter Sweep", "Conditional Path", "Deep History"]
)

# ============================================================
# Tab 1: Parameter Sweep
# ============================================================
with tab_sweep:
    st.warning(
        "**Congratulations, you discovered the past!** "
        "Any threshold that 'wins' in this dataset is a product of that specific history. "
        "This is in-sample optimisation — the future may and probably will disagree.",
        icon="🎉",
    )

    n_combos = len(thresholds) * len(deploy_pcts)
    run_sweep_btn = st.button(
        f"Run Parameter Sweep ({n_combos} combinations x rolling windows)",
        key="run_sweep_etf",
    )

    if run_sweep_btn or "sweep_results_cache" in st.session_state:
        buf = io.BytesIO()
        price_df.to_parquet(buf)

        with st.spinner(
            f"Running parameter sweep over {window_years}Y rolling windows... "
            f"This may take 1-2 minutes."
        ):
            raw_results = cached_parameter_sweep(
                price_bytes=buf.getvalue(),
                monthly=param_dict["monthly_contribution"],
                payday=param_dict["payday"],
                initial=param_dict["initial_investment"],
                start=param_dict["start_date"],
                end=param_dict["end_date"],
                max_wait=param_dict["max_wait_months"],
                fixed_fee=param_dict["fixed_fee"],
                pct_fee=param_dict["pct_fee"],
                thresholds_t=tuple(thresholds),
                deploy_pcts_t=tuple(deploy_pcts),
                window_years_v=window_years,
                step_months_v=step_months,
            )
            st.session_state["sweep_results_cache"] = raw_results

        results_dicts = st.session_state.get("sweep_results_cache", [])

        if not results_dicts:
            st.info(
                "No sweep results — the date range may be too short for the selected window size. "
                f"Need at least {window_years} years of data."
            )
        else:
            # Reconstruct SweepResult dataclasses
            sweep_results = [
                SweepResult(
                    threshold=d["threshold"],
                    deploy_pct=d["deploy_pct"],
                    win_rate=d["win_rate"],
                    median_outperformance=d["median_outperformance"],
                    p10_outperformance=d["p10_outperformance"],
                    p90_outperformance=d["p90_outperformance"],
                    n_windows=d["n_windows"],
                    median_deployments=d["median_deployments"],
                )
                for d in results_dicts
            ]

            sweep_df = sweep_to_dataframe(sweep_results)

            # Win-rate heatmap
            st.markdown(
                '<h4 style="color:#FAFAFA; font-weight:700; margin:16px 0 8px 0">'
                "Win Rate Heatmap</h4>",
                unsafe_allow_html=True,
            )
            wr_pivot = win_rate_pivot(sweep_df)
            fig_wr = plot_sweep_heatmap(
                wr_pivot,
                title="Win Rate vs DCA",
                subtitle=f"% of {window_years}Y rolling windows where dip strategy beats monthly DCA",
                value_fmt=".0%",
                zmid=50,
            )
            st.plotly_chart(fig_wr, use_container_width=True)

            # Outperformance heatmap
            st.markdown(
                '<h4 style="color:#FAFAFA; font-weight:700; margin:16px 0 8px 0">'
                "Median Outperformance Heatmap</h4>",
                unsafe_allow_html=True,
            )
            op_pivot = outperformance_pivot(sweep_df)
            fig_op = plot_sweep_heatmap(
                op_pivot,
                title="Median Outperformance vs DCA",
                subtitle="Median (dip_wealth / dca_wealth - 1) across rolling windows",
                value_fmt=".1%",
                zmid=0,
            )
            st.plotly_chart(fig_op, use_container_width=True)

            # Best combo callout
            best = sweep_df.sort_values("win_rate", ascending=False).iloc[0]
            st.success(
                f"**Best combo:** Deploy **{best['deploy_pct']:.0%}** at "
                f"**{best['threshold']:.0%}** threshold beats DCA in "
                f"**{best['win_rate']:.0%}** of {best['n_windows']:.0f} "
                f"rolling {window_years}-year windows. "
                f"(Median outperformance: {best['median_outperformance']:.1%})"
            )

            # Full results table
            with st.expander("Full sweep results table (sorted by win rate)"):
                display_df = sweep_df.copy()
                display_df["threshold"] = display_df["threshold"].map(lambda x: f"{x:.0%}")
                display_df["deploy_pct"] = display_df["deploy_pct"].map(lambda x: f"{x:.0%}")
                display_df["win_rate"] = display_df["win_rate"].map(lambda x: f"{x:.1%}")
                display_df["median_outperformance"] = display_df["median_outperformance"].map(
                    lambda x: f"{x:.2%}"
                )
                display_df["p10"] = display_df["p10"].map(lambda x: f"{x:.2%}")
                display_df["p90"] = display_df["p90"].map(lambda x: f"{x:.2%}")
                display_df["median_deployments"] = display_df["median_deployments"].map(
                    lambda x: f"{x:.1f}"
                )
                display_df = display_df.sort_values("win_rate", ascending=False)
                display_df.columns = [
                    "Threshold",
                    "Deploy %",
                    "Win Rate",
                    "Median Outperf.",
                    "P10 (Worst)",
                    "P90 (Best)",
                    "N Windows",
                    "Median Deployments",
                ]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    data_source_caption(source=data_source, as_of=as_of_date, is_total_return=True, currency="EUR")

# ============================================================
# Tab 2: Conditional Path
# ============================================================
with tab_path:
    st.markdown(
        """
        <div style="background:#1E2130; border:1px solid #2D3047; border-radius:10px;
                    padding:16px; margin-bottom:16px">
            <b style="color:#F47920; font-size:1.1em">The Core Uncertainty</b><br>
            <span style="color:#9CA3AF">
            You're sitting at a -10% drawdown. Deploy 25%, 50%, 75%, or 100%?<br>
            The honest answer: <b style="color:#FAFAFA">you cannot know</b> whether this
            recovers from here or drops to -40%. This tool shows the historical distribution
            of outcomes for each choice — without lookahead.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_path1, col_path2 = st.columns(2)
    with col_path1:
        current_dd_input = st.slider(
            "Current drawdown (%)",
            min_value=-50,
            max_value=0,
            value=-10,
            step=1,
            help="What drawdown level are you currently at?",
        )
        current_dd = current_dd_input / 100.0

        cash_accumulated = st.number_input(
            "Accumulated cash (EUR)",
            min_value=100.0,
            max_value=500_000.0,
            value=float(param_dict["monthly_contribution"] * 12),
            step=500.0,
            help="Total cash sitting uninvested right now",
        )

    with col_path2:
        horizon_months = st.slider(
            "Simulation horizon (months)",
            min_value=6,
            max_value=36,
            value=24,
            step=6,
        )
        n_sims = st.select_slider(
            "Simulation paths",
            options=[100, 250, 500, 1000],
            value=500,
        )
        deploy_path_options = st.multiselect(
            "Deploy fractions to compare",
            options=["25%", "50%", "75%", "100%"],
            default=["25%", "50%", "75%", "100%"],
            key="path_deploys",
        )
        path_deploy_pcts = (
            [float(d.replace("%", "")) / 100.0 for d in deploy_path_options]
            or [0.25, 0.50, 0.75, 1.00]
        )

    run_path_btn = st.button("Run Conditional Path Bootstrap", key="run_path")

    if run_path_btn or "path_results_cache" in st.session_state:
        buf2 = io.BytesIO()
        price_df.to_parquet(buf2)

        with st.spinner(
            f"Bootstrapping {n_sims} paths per deploy fraction from historical "
            f"{current_dd:.0%} drawdown entries..."
        ):
            raw_path_results = cached_path_bootstrap(
                price_bytes=buf2.getvalue(),
                current_dd=current_dd,
                deploy_pcts_t=tuple(sorted(path_deploy_pcts)),
                monthly_contribution=param_dict["monthly_contribution"],
                cash_accumulated=cash_accumulated,
                horizon_months=horizon_months,
                n_simulations=n_sims,
            )
            st.session_state["path_results_cache"] = raw_path_results

        path_dicts = st.session_state.get("path_results_cache", [])

        if not path_dicts:
            st.warning(
                f"No historical entry points found near {current_dd:.0%} drawdown. "
                "Try adjusting the drawdown level or using a longer history."
            )
        else:
            # Reconstruct PathSimulation objects
            path_sims = [
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
                for d in path_dicts
            ]

            # Fan chart
            fig_fan = plot_fan_chart(
                path_sims,
                horizon_months=horizon_months,
                monthly_contribution=param_dict["monthly_contribution"],
                currency="EUR",
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            # Summary table
            summary_rows = []
            for s in path_sims:
                summary_rows.append(
                    {
                        "Deploy %": f"{s.deploy_pct:.0%}",
                        "P(beats DCA)": f"{s.prob_beats_dca:.0%}",
                        "Median Wealth": f"EUR {s.p50_wealth[-1]:,.0f}",
                        "P10 (Worst)": f"EUR {s.p5_wealth[-1]:,.0f}",
                        "P90 (Best)": f"EUR {s.p95_wealth[-1]:,.0f}",
                    }
                )
            st.dataframe(
                pd.DataFrame(summary_rows), use_container_width=True, hide_index=True
            )

            # Key insight: best risk-adjusted outcome
            best_path = max(path_sims, key=lambda s: s.prob_beats_dca)
            conservative = min(path_sims, key=lambda s: abs(s.p5_wealth[-1] - s.p50_wealth[-1]))

            st.markdown(
                f"""
                <div style="background:#1E2130; border:1px solid #2D3047; border-radius:10px;
                            padding:16px; margin-top:8px">
                    <b style="color:#00C896">Key insight:</b>
                    <span style="color:#9CA3AF">
                    Deploying <b style="color:#FAFAFA">{best_path.deploy_pct:.0%}</b> has the
                    highest probability of beating DCA at this drawdown level
                    ({best_path.prob_beats_dca:.0%}). The most conservative option
                    (<b style="color:#FAFAFA">{conservative.deploy_pct:.0%}</b>) has the
                    tightest outcome range — less regret either way.
                    Based on historical paths from {len(path_dicts)} deploy scenarios
                    with {n_sims} simulations each.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style="border:1px solid #3D4066; border-radius:10px; padding:12px;
                    background:rgba(30,33,48,0.6); margin-top:16px">
            <b style="color:#F47920">No lookahead guarantee</b>
            <span style="color:#9CA3AF"> — paths are sampled from past continuations at
            similar drawdown depths. Historical base rates are not forecasts.
            Each resampled path is one possible future, not the predicted one.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_source_caption(source=data_source, as_of=as_of_date, is_total_return=True, currency="EUR")

# ============================================================
# Tab 3: Deep History
# ============================================================
with tab_deep:
    st.markdown(
        '<h4 style="color:#FAFAFA; font-weight:700; margin-bottom:8px">Deep History Analysis</h4>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Run parameter sweep on 90+ years of market history. "
        "More data = more robust base rates, but regime changes mean older data is less relevant."
    )

    dh_ticker_name = st.selectbox(
        "Historical index",
        options=list(DEEP_HISTORY_TICKERS.keys()),
        index=0,
    )
    dh_ticker = DEEP_HISTORY_TICKERS[dh_ticker_name]

    col_dh1, col_dh2 = st.columns(2)
    with col_dh1:
        dh_window_years = st.select_slider(
            "Window size (years)",
            options=[5, 10, 15, 20],
            value=10,
            key="dh_window",
        )
    with col_dh2:
        dh_step_months = st.slider(
            "Step (months)",
            min_value=6,
            max_value=24,
            value=12,
            key="dh_step",
        )

    with st.spinner(f"Fetching deep history for {dh_ticker}..."):
        try:
            dh_monthly = fetch_deep_history(dh_ticker)
        except Exception as exc:
            st.error(f"Could not fetch deep history: {exc}")
            dh_monthly = None

    if dh_monthly is not None:
        st.markdown(
            f"**{dh_ticker_name}**: {len(dh_monthly)} monthly observations "
            f"({dh_monthly.index[0].year} – {dh_monthly.index[-1].year})"
        )

        # Total return chart
        import plotly.graph_objects as go

        from ui.theme import GREEN as _GREEN
        from ui.theme import apply_chart_layout as _apply_layout

        fig_dh = go.Figure()
        fig_dh.add_trace(
            go.Scatter(
                x=dh_monthly.index,
                y=dh_monthly.values,
                name=dh_ticker_name,
                line=dict(color=_GREEN, width=1.5),
                hovertemplate="Date: %{x|%Y-%m}<br>Index: %{y:.1f}<extra></extra>",
            )
        )
        _apply_layout(
            fig_dh,
            title=f"{dh_ticker_name} — Total Return Index (base = 100)",
            subtitle="Monthly resampled, adjusted for dividends",
        )
        fig_dh.update_layout(
            xaxis_title="Date",
            yaxis_title="Index (base = 100)",
            height=350,
            yaxis_type="log",
        )
        st.plotly_chart(fig_dh, use_container_width=True)

        # Deep history parameter sweep
        run_dh_sweep_btn = st.button(
            "Run Parameter Sweep on Deep History",
            key="run_dh_sweep",
        )

        @st.cache_data(ttl=86400, show_spinner=False)
        def cached_dh_sweep(
            dh_ticker_key: str,
            monthly: float,
            payday: int,
            window_years_v: int,
            step_months_v: int,
            thresholds_t: tuple,
            deploy_pcts_t: tuple,
        ) -> list[dict]:
            """Run parameter sweep on deep history monthly data."""
            series = fetch_deep_history(dh_ticker_key)
            base = SimulationParams(
                monthly_contribution=monthly,
                payday=payday,
                initial_investment=0.0,
                start_date=series.index[0].date(),
                end_date=series.index[-1].date(),
                dip_threshold=-0.10,
                max_wait_months=24,
            )
            sweep_results = run_parameter_sweep(
                prices=series,
                base_params=base,
                thresholds=list(thresholds_t),
                deploy_pcts=list(deploy_pcts_t),
                window_years=window_years_v,
                step_months=step_months_v,
                points_per_year=12,  # monthly data
            )
            return [
                {
                    "threshold": r.threshold,
                    "deploy_pct": r.deploy_pct,
                    "win_rate": r.win_rate,
                    "median_outperformance": r.median_outperformance,
                    "p10_outperformance": r.p10_outperformance,
                    "p90_outperformance": r.p90_outperformance,
                    "n_windows": r.n_windows,
                    "median_deployments": r.median_deployments,
                }
                for r in sweep_results
            ]

        if run_dh_sweep_btn:
            with st.spinner(
                f"Running parameter sweep on {dh_ticker_name} ({len(dh_monthly)} months)..."
            ):
                dh_sweep_raw = cached_dh_sweep(
                    dh_ticker_key=dh_ticker,
                    monthly=param_dict["monthly_contribution"],
                    payday=param_dict["payday"],
                    window_years_v=dh_window_years,
                    step_months_v=dh_step_months,
                    thresholds_t=tuple(thresholds),
                    deploy_pcts_t=tuple(deploy_pcts),
                )
            st.session_state["dh_sweep_cache"] = dh_sweep_raw

        dh_results = st.session_state.get("dh_sweep_cache", [])

        if dh_results:
            dh_sweep_df = sweep_to_dataframe(
                [
                    SweepResult(
                        threshold=d["threshold"],
                        deploy_pct=d["deploy_pct"],
                        win_rate=d["win_rate"],
                        median_outperformance=d["median_outperformance"],
                        p10_outperformance=d["p10_outperformance"],
                        p90_outperformance=d["p90_outperformance"],
                        n_windows=d["n_windows"],
                        median_deployments=d["median_deployments"],
                    )
                    for d in dh_results
                ]
            )

            st.markdown(
                f"**Based on {dh_monthly.index[0].year}–{dh_monthly.index[-1].year} data, "
                f"here's what worked:**"
            )

            wr_pivot_dh = win_rate_pivot(dh_sweep_df)
            fig_wr_dh = plot_sweep_heatmap(
                wr_pivot_dh,
                title=f"Win Rate vs DCA — {dh_ticker_name}",
                subtitle=f"% of {dh_window_years}Y windows where dip beats DCA ({len(dh_monthly)} months of history)",
                value_fmt=".0%",
                zmid=50,
            )
            st.plotly_chart(fig_wr_dh, use_container_width=True)

            if len(dh_sweep_df) > 0:
                best_dh = dh_sweep_df.sort_values("win_rate", ascending=False).iloc[0]
                st.info(
                    f"**{dh_ticker_name} ({dh_monthly.index[0].year}-):** "
                    f"Deploying **{best_dh['deploy_pct']:.0%}** at "
                    f"**{best_dh['threshold']:.0%}** beats DCA in "
                    f"**{best_dh['win_rate']:.0%}** of {best_dh['n_windows']:.0f} "
                    f"{dh_window_years}-year windows. "
                    f"(Median outperformance: {best_dh['median_outperformance']:.1%})"
                )

    st.markdown(
        """
        <div style="border:1px solid #3D4066; border-radius:10px; padding:12px;
                    background:rgba(30,33,48,0.6); margin-top:16px">
            <b style="color:#F47920">Survivorship bias warning</b>
            <span style="color:#9CA3AF"> — deep history only exists for indices that
            <em>survived</em>. Markets that failed (Nikkei 1989, Weimar, etc.) are absent.
            Long-term win rates are upward-biased.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
