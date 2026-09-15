"""How it works — methodology, data sources, and assumptions."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ui.components import page_header  # noqa: E402
from ui.copy import DISCLAIMER_SHORT  # noqa: E402
from ui.theme import GLOBAL_CSS  # noqa: E402

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header(
    "How it works",
    "Methodology, data sources, formulas, and assumptions behind the numbers.",
)

# ---------------------------------------------------------------------------
# What the app does
# ---------------------------------------------------------------------------
st.header("What this app does")
st.markdown(
    """
    **Do we have a dip?** compares three investment strategies using real historical market data:

    - **Invest monthly (DCA)** — invest a fixed amount on a fixed day every month, regardless of market conditions.
    - **Wait for a dip** — hold cash until the market drops by a chosen percentage, then invest.
    - **Tiered deployment** — invest gradually across multiple drawdown levels.

    The app runs these strategies against actual historical prices and shows what the outcomes would have been.
    **It does not predict the future.**
    """
)

# ---------------------------------------------------------------------------
# Historical scenarios engine
# ---------------------------------------------------------------------------
st.header("Historical scenarios")
st.markdown(
    """
    The "Historical scenarios" page uses a **conditional path bootstrap**:

    1. We compute the full drawdown history for the chosen asset.
    2. We find all historical dates where the drawdown was within ±5% of today's level.
    3. For each matching date, we extract the next N months of price returns.
    4. We sample these historical continuation paths to build a distribution of outcomes.

    This uses only real historical returns — it does not generate synthetic random paths.
    The distribution shows what actually happened in past periods with a similar drawdown,
    weighted equally.
    """
)

# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------
st.header("Core formulas")

st.subheader("Drawdown")
st.latex(r"DD(t) = \frac{P(t)}{\max_{s \leq t} P(s)} - 1")
st.caption("Signal uses prior completed close (t-1). Trade executes at t.")

st.subheader("Total return index")
st.latex(r"TR_{base}(t) = \frac{AdjClose(t) \cdot FX(t)}{AdjClose(t_0) \cdot FX(t_0)}")
st.caption(
    "AdjClose: Yahoo Finance auto_adjust=True (dividend-reinvested). "
    "FX: ECB reference rates."
)

st.subheader("Annualized return (XIRR)")
st.latex(r"\sum_{i} \frac{CF_i}{(1+r)^{d_i/365}} = 0")
st.caption("Solved via scipy brentq. Investments are negative flows; terminal wealth is positive.")

st.subheader("Cash interest accrual")
st.latex(r"B(t) = B(t_0) \cdot (1 + r_{annual})^{d/365}")
st.caption("Day-by-day compounding using calendar days elapsed.")

# ---------------------------------------------------------------------------
# No-lookahead rules
# ---------------------------------------------------------------------------
st.header("No-lookahead rules")
st.markdown(
    """
    To prevent look-ahead bias in all backtests:

    1. **Signal**: Drawdown is computed using the **prior completed close** (index t-1).
    2. **Execution**: Trade executes at the **next available close** (index t).
    3. **Contributions**: Cash arrives on payday, invested on the **next trading day**.
    4. **Tiers**: Tiers reset only after a **new all-time high** is confirmed.
    """
)

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
st.header("Data sources")
st.markdown(
    """
    | Source | Data | Notes |
    |--------|------|-------|
    | Yahoo Finance (yfinance) | ETF/index prices | Auto-adjusted for dividends and splits |
    | ECB Data Portal | EUR FX rates | Reference rates, not interbank rates |
    | ECB Data Portal | ECB deposit facility rate | Policy rate — not a retail savings rate |
    | SNB | CHF policy rate | Sight deposit rate — policy rate |

    **Live data only.** This app does not serve synthetic, locally stored, or demo market data.
    All market data is fetched from the sources above on demand.
    If a source is unavailable, the app shows an error with a Retry button — it does not show stale data.
    """
)

st.subheader("Data freshness")
st.markdown(
    """
    Every page shows a freshness indicator for the data it displays:

    - **Live** — fetched within the last 30 minutes
    - **Delayed** — data is older than 30 minutes but less than one trading day
    - **Market closed** — most recent close (normal outside trading hours)

    Historical price data is cached for up to 6 hours. Current quotes are cached for 5 minutes.
    """
)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.divider()
st.header("Disclaimer")
st.markdown(
    f"""
    {DISCLAIMER_SHORT}

    This tool is for educational and informational purposes only.
    It does not constitute investment advice, a recommendation to buy or sell any security,
    or a prediction of future performance.

    See [DISCLAIMER.md](https://github.com/tzijlstra-acn/DoWeHaveADip/blob/master/DISCLAIMER.md)
    for full terms.
    """
)
