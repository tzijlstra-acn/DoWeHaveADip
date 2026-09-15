"""Do we have a dip? — Streamlit navigation shell."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ui.icons import (  # noqa: E402
    ICON_COMPARE,
    ICON_EXIT,
    ICON_FX,
    ICON_HISTORY,
    ICON_INFO,
    ICON_MARKET,
    ICON_RATES,
    ICON_TODAY,
)

st.set_page_config(
    page_title="Do we have a dip?",
    page_icon=":material/trending_down:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Health check endpoint
if "health" in st.query_params:
    st.json({"status": "ok", "version": "0.2.0"})
    st.stop()

main_pages = [
    st.Page("app_pages/today.py",     title="Today",                icon=ICON_TODAY),
    st.Page("app_pages/compare.py",   title="Compare choices",      icon=ICON_COMPARE),
    st.Page("app_pages/scenarios.py", title="Historical scenarios",  icon=ICON_HISTORY),
    st.Page("app_pages/about.py",     title="How it works",         icon=ICON_INFO),
]

advanced_pages = [
    st.Page("app_pages/advanced_market.py", title="Market overview",    icon=ICON_MARKET),
    st.Page("app_pages/advanced_exit.py",   title="Exit strategies",    icon=ICON_EXIT),
    st.Page("app_pages/advanced_fx.py",     title="Currency breakdown", icon=ICON_FX),
    st.Page("app_pages/advanced_rates.py",  title="Interest rates",     icon=ICON_RATES),
]

pg = st.navigation({"": main_pages, "Advanced": advanced_pages})
pg.run()
