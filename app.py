"""Streamlit entry point: global page config and tab routing. No business logic lives here."""

import streamlit as st

from tabs.watchlist_tab import render_watchlist_tab
from tabs.analysis_tab import render_analysis_tab
from tabs.portfolio_tab import render_portfolio_tab

st.set_page_config(
    page_title="Stock Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_watchlist, tab_analysis, tab_portfolio = st.tabs(
    ["📋 Watchlist", "🔍 Analysis", "💼 Portfolio"]
)

with tab_watchlist:
    render_watchlist_tab()

with tab_analysis:
    render_analysis_tab()

with tab_portfolio:
    render_portfolio_tab()
