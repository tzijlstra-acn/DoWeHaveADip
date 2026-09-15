"""Currency Reality Check — how FX affected your returns."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dipdca.config import load_assets_config  # noqa: E402
from ui.charts import plot_fx_decomposition  # noqa: E402
from ui.components import data_source_caption, page_header  # noqa: E402
from ui.formatting import fmt_pct  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "Currency Reality Check",
    "How much of your return was the asset vs the exchange rate? Log-return decomposition: Native + FX = Total.",
)

st.latex(r"r_{base}(t) = r_{native}(t) + r_{FX}(t)")
st.caption(
    "Where r = log return. Native = log-return of adj_close in asset currency. "
    "FX = log-return of (base_currency / asset_currency) rate. "
    "Sum = log-return of adj_close in base currency."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
base_currency = st.sidebar.selectbox("Your base currency", ["EUR", "CHF", "USD"], index=0)

start_date = st.sidebar.date_input("Start", value=datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End", value=datetime.date.today())

# ---------------------------------------------------------------------------
# Load FX data (ECB)
# ---------------------------------------------------------------------------
from dipdca.data.providers.ecb_fx import EcbFxProvider  # noqa: E402

with st.spinner("Fetching FX rates from ECB..."):
    try:
        fx_provider = EcbFxProvider()
        currencies = ["USD", "CHF", "GBP"]
        fx_df = fx_provider.get_rates(currencies, start_date, end_date)
    except Exception as exc:
        st.error(f"Could not load FX rates from ECB: {exc}")
        st.stop()

# Align FX to date range
fx_df = fx_df.loc[pd.Timestamp(start_date): pd.Timestamp(end_date)]
if fx_df.empty:
    st.error("No FX data available for the selected period.")
    st.stop()

# ---------------------------------------------------------------------------
# Load asset price data
# ---------------------------------------------------------------------------
assets = load_assets_config()

# Filter to assets whose quote currency differs from base currency
relevant_assets = [a for a in assets if a["quote_currency"] != base_currency]

st.subheader(f"Select Asset (base currency: {base_currency})")
asset_options = {a["display_name"]: a for a in relevant_assets}

if not asset_options:
    st.info(f"All assets are already quoted in {base_currency}. No FX conversion needed.")
    st.stop()

selected_name = st.selectbox("Asset", list(asset_options.keys()))
selected_asset = asset_options[selected_name]
asset_currency = selected_asset["quote_currency"]

st.info(
    f"**{selected_name}** is quoted in **{asset_currency}**. "
    f"You are investing in **{base_currency}**. "
    f"The FX pair is {asset_currency}/{base_currency}."
)

# Load price data
from dipdca.data.errors import LiveDataUnavailable  # noqa: E402
from dipdca.data.service import get_market_data_service  # noqa: E402
from ui.components import freshness_caption, live_data_error  # noqa: E402

symbol = selected_asset.get("etf_symbol") or selected_asset.get("index_symbol")
if not symbol:
    st.error(f"No ticker symbol configured for {selected_name}.")
    st.stop()

with st.spinner(f"Fetching price data for {symbol}..."):
    try:
        _result = get_market_data_service().get_history(symbol, start_date, end_date)
        price_df = _result.frame
        data_source = f"Yahoo Finance ({symbol})"
        freshness_caption(_result.freshness)
    except LiveDataUnavailable as exc:
        live_data_error(exc, context=symbol)
        st.stop()

price_df = price_df.loc[pd.Timestamp(start_date): pd.Timestamp(end_date)]
if len(price_df) < 10:
    st.error("Not enough price data. Try a wider date range.")
    st.stop()

adj = price_df["adj_close"].dropna()

# ---------------------------------------------------------------------------
# FX decomposition
# ---------------------------------------------------------------------------

# Get FX rate: how many base currency units per 1 unit of asset currency
# ECB rates give units of foreign per EUR, so to convert asset → EUR:
# price_EUR = price_USD / (USD_per_EUR)
# If base is EUR and asset is USD: fx_rate = 1 / (USD_per_EUR) = EUR_per_USD

# Build fx_rate series: (base_currency) per (asset_currency)
if asset_currency == base_currency:
    fx_rate = pd.Series(1.0, index=adj.index)
elif base_currency == "EUR" and asset_currency in fx_df.columns:
    # ECB gives USD_per_EUR, so EUR_per_USD = 1 / USD_per_EUR
    eur_per_asset = 1.0 / fx_df[asset_currency]
    fx_rate = eur_per_asset.reindex(adj.index, method="ffill").bfill()
elif asset_currency == "EUR" and base_currency in fx_df.columns:
    # fx_df gives base_per_EUR
    fx_rate = fx_df[base_currency].reindex(adj.index, method="ffill").bfill()
elif base_currency in fx_df.columns and asset_currency in fx_df.columns:
    # Triangulate through EUR: base_per_asset = base_per_eur / asset_per_eur
    base_per_eur = fx_df[base_currency]
    asset_per_eur = fx_df[asset_currency]
    cross = base_per_eur / asset_per_eur
    fx_rate = cross.reindex(adj.index, method="ffill").bfill()
else:
    st.error(
        f"FX rate for {asset_currency}/{base_currency} is not available from the live ECB source. "
        "No results will be shown until the data source can be reached."
    )
    st.stop()

# Align
adj_aligned = adj.copy()
fx_aligned = fx_rate.reindex(adj_aligned.index, method="ffill").fillna(1.0)

# Compute total return indices
tr_native = adj_aligned / adj_aligned.iloc[0]  # native currency, rebased to 1
tr_base_raw = adj_aligned * fx_aligned
tr_base = tr_base_raw / tr_base_raw.iloc[0]  # base currency, rebased to 1

# FX contribution: tr_base = tr_native * fx_contribution_index
fx_index = fx_aligned / fx_aligned.iloc[0]
fx_contribution = fx_index - 1  # cumulative FX return

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    native_total = float(tr_native.iloc[-1] - 1)
    st.metric(f"Native Return ({asset_currency})", fmt_pct(native_total))
with col_meta2:
    fx_total = float(fx_contribution.iloc[-1])
    st.metric(f"FX Contribution ({asset_currency} → {base_currency})", fmt_pct(fx_total))
with col_meta3:
    base_total = float(tr_base.iloc[-1] - 1)
    st.metric(f"Total Return ({base_currency})", fmt_pct(base_total))

# Verify decomposition (log-return check)
log_native = float(np.log(tr_native.iloc[-1]))
log_fx = float(np.log(fx_aligned.iloc[-1] / fx_aligned.iloc[0]))
log_base = float(np.log(tr_base.iloc[-1]))
residual = abs(log_native + log_fx - log_base)

if residual > 1e-6:
    st.warning(f"Log-return decomposition residual: {residual:.6f} (rounding may cause small differences).")

# Chart
fig = plot_fx_decomposition(tr_native, fx_contribution, tr_base, base_currency=base_currency)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Hedged vs unhedged note
# ---------------------------------------------------------------------------
is_hedged = selected_asset.get("is_hedged", False)
if is_hedged:
    st.success(
        f"**{selected_name}** is a currency-hedged product. "
        "The actual FX impact on this ETF should be near zero. "
        "The chart above shows the unhedged theoretical impact."
    )
else:
    st.info(
        f"**{selected_name}** is **not currency-hedged**. "
        "The FX contribution shown above is a real component of your return. "
        f"A weak {asset_currency} vs {base_currency} erodes returns; "
        f"a strong {asset_currency} amplifies them."
    )

# ---------------------------------------------------------------------------
# Formula reminder
# ---------------------------------------------------------------------------
with st.expander("Formula details"):
    st.latex(
        r"""
        r_{base}(t) = \ln\!\left(\frac{P_{native}(t) \cdot FX(t)}{P_{native}(t_0) \cdot FX(t_0)}\right)
                    = r_{native}(t) + r_{FX}(t)
        """
    )
    st.markdown(
        f"""
        - **Native return**: log-return of adj_close in {asset_currency}
        - **FX return**: log-return of ({base_currency} per {asset_currency}) exchange rate
        - **Total**: their sum = log-return of your position in {base_currency}

        Source: FX rates from ECB (live).
        """
    )

data_source_caption(
    source=data_source,
    as_of=price_df.index[-1].date() if len(price_df) > 0 else None,
    is_total_return=True,
    currency=f"{asset_currency} → {base_currency}",
)
