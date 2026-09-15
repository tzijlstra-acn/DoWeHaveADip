"""Savings Rates — compare cash yields across NL / DE / CH."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ui.charts import plot_savings_rates  # noqa: E402
from ui.components import data_source_caption, page_header  # noqa: E402
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.set_page_config(page_title="Savings Rates", page_icon="🏦", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "Interest Rates",
    "Official ECB household deposit rates and SNB policy rate — not retail savings account rates.",
    "🏦",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

start_date = st.sidebar.date_input("History start", value=datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("History end", value=datetime.date.today())

manual_override = st.sidebar.checkbox("Add manual rate override", value=False)
manual_rate = 0.035
manual_label = "My bank"
if manual_override:
    manual_rate = (
        st.sidebar.slider("Your actual rate (%/year)", min_value=0.0, max_value=10.0, value=3.5, step=0.25)
        / 100.0
    )
    manual_label = st.sidebar.text_input("Label", value="My bank rate")

# ---------------------------------------------------------------------------
# Load rates — live only (ECB + SNB)
# ---------------------------------------------------------------------------
from io import StringIO  # noqa: E402

import httpx  # noqa: E402

from dipdca.data.providers.snb_rates import SnbRatesProvider  # noqa: E402

rates_dict: dict[str, pd.Series] = {}

with st.spinner("Fetching interest rates from ECB and SNB..."):
    # ECB MIR (MFI Interest Rates) — household overnight deposits
    for country_code, flow_ref, key in [
        ("DE", "MIR", "M.DE.B.L22.A.R.A.2250.EUR.N"),
        ("NL", "MIR", "M.NL.B.L22.A.R.A.2250.EUR.N"),
    ]:
        url = f"https://data-api.ecb.europa.eu/service/data/{flow_ref}/{key}"
        params = {
            "startPeriod": str(start_date),
            "endPeriod": str(end_date),
            "format": "csvdata",
        }
        try:
            resp = httpx.get(url, params=params, timeout=20.0)
            resp.raise_for_status()
            df_ecb = pd.read_csv(StringIO(resp.text))
            if "TIME_PERIOD" in df_ecb.columns and "OBS_VALUE" in df_ecb.columns:
                df_ecb["TIME_PERIOD"] = pd.to_datetime(df_ecb["TIME_PERIOD"])
                s = pd.to_numeric(df_ecb.set_index("TIME_PERIOD")["OBS_VALUE"], errors="coerce").dropna()
                rates_dict[f"{country_code} (ECB MIR overnight)"] = s / 100.0
        except Exception as exc:
            st.warning(f"ECB MIR {country_code} fetch failed: {exc}")

    # SNB policy rate
    try:
        snb = SnbRatesProvider()
        snb_series = snb.get_policy_rate(start_date, end_date)
        snb_monthly = snb_series.resample("ME").last()
        rates_dict["CH (SNB sight deposit rate — policy, not retail)"] = snb_monthly
    except Exception as exc:
        st.warning(f"SNB fetch failed: {exc}")

if not rates_dict:
    st.error("Could not load any savings rate data. Check your internet connection.")
    st.stop()

# Add manual rate overlay
if manual_override:
    idx_m = pd.date_range(start_date, end_date, freq="ME")
    rates_dict[manual_label] = pd.Series(manual_rate, index=idx_m)

if not rates_dict:
    st.error("No rate data available.")
    st.stop()

rates_df = pd.DataFrame(rates_dict)

# ---------------------------------------------------------------------------
# Latest rates row
# ---------------------------------------------------------------------------
st.subheader("Current / Latest Observed Rates")
cols = st.columns(min(len(rates_dict), 4))
for i, (name, series) in enumerate(rates_dict.items()):
    s = series.dropna()
    if len(s) == 0:
        continue
    latest_rate = float(s.iloc[-1])
    latest_date = s.index[-1].date()
    col = cols[i % len(cols)]
    with col:
        short_name = name.split(" (")[0]
        color = "#00C896" if latest_rate > 0.02 else ("#F47920" if latest_rate > 0.005 else "#FF4B6B")
        st.markdown(
            f"""
            <div style="border:1px solid {color}; border-radius:8px; padding:12px; text-align:center;">
                <p style="margin:0;color:#aaa;font-size:0.8em">{name}</p>
                <h2 style="margin:6px 0;color:{color}">{latest_rate:.2%}</h2>
                <p style="margin:0;color:#888;font-size:0.75em">as of {latest_date} (monthly)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ---------------------------------------------------------------------------
# Rate history chart
# ---------------------------------------------------------------------------
st.subheader("Rate History")
fig = plot_savings_rates(rates_df)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Break-even analysis
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Break-Even Dip Size")
st.caption(
    "How deep does the dip need to be to justify holding cash instead of investing immediately? "
    "This is a rough back-of-envelope calculation — not a precise recommendation."
)

col_be1, col_be2 = st.columns(2)
with col_be1:
    cash_rate_be = (
        st.slider("Your cash rate (%/year)", min_value=0.0, max_value=10.0, value=3.5, step=0.25) / 100.0
    )
    wait_months = st.slider("Expected wait time (months)", min_value=1, max_value=60, value=12)

with col_be2:
    expected_market_return = (
        st.slider("Expected annual market return (%)", min_value=1, max_value=20, value=8, step=1) / 100.0
    )

    # Break-even dip needed:
    # (1 + r_market)^(wait/12) = (1 + r_cash)^(wait/12) * (1 - dip)
    # dip = 1 - (1 + r_cash)^(wait/12) / (1 + r_market)^(wait/12)
    t_years = wait_months / 12.0
    break_even_dip = 1.0 - ((1 + cash_rate_be) ** t_years) / ((1 + expected_market_return) ** t_years)

    st.metric(
        "Break-even dip needed",
        fmt_pct(break_even_dip),
        help=(
            "The minimum price drop needed so that the cash interest earned during the wait "
            "compensates for the market gains missed while waiting."
        ),
    )
    st.caption(
        f"Cash earns {cash_rate_be:.2%}/yr for {wait_months} months. "
        f"Market assumed to return {expected_market_return:.0%}/yr. "
        "If dip > this level, Cash Goblin strategy has a theoretical advantage."
    )

# ---------------------------------------------------------------------------
# Disclaimer notes
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    **Notes:**
    - ECB MIR rates are official averages for EU banks. Your bank's rate may be higher or lower.
    - SNB policy rate is the Swiss National Bank sight deposit rate — retail savings rates often differ.
    - Rates are monthly frequency (latest observation forward-filled to daily in backtests).
    - Rate data from ECB is available at `data-api.ecb.europa.eu`.
    - Negative rates (Switzerland, 2014–2022) meant banks charged depositors — unusual but historical.
    """
)

data_source_caption(
    source="ECB MIR / SNB",
    as_of=rates_df.index[-1].date() if len(rates_df) > 0 else None,
    is_total_return=False,
    currency="EUR / CHF",
)
