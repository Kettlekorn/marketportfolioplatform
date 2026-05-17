"""Analysis tab: full 8-section analysis card for any ticker."""

from datetime import datetime

import streamlit as st

from analytics.meridian import compute_meridian_score
from analytics.risk import compute_risk_metrics
from components.cards import metric_card, score_badge, section_header
from components.charts import factor_bar_chart, plotly_price_chart
from data.insider_data import get_insider_summary
from data.market_data import (
    get_company_name,
    get_current_price,
    get_fundamentals,
    get_news,
    get_price_history,
    get_recommendations,
    get_ticker_info,
)

_PERIOD_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _v(val, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a value or return '—' when None / NaN."""
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _pct(val) -> str:
    return _v(val, ".1%") if val is not None else "—"


def _fmt_price(val) -> str:
    return f"${val:,.2f}" if val is not None else "—"


def _delta_str(chg: float | None) -> str | None:
    if chg is None:
        return None
    return f"{chg:+.2f}%"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(ticker: str) -> None:
    info = get_ticker_info(ticker)
    name = info.get("shortName") or info.get("longName") or ticker
    price, chg = get_current_price(ticker)

    col_title, col_badge = st.columns([3, 2])
    with col_title:
        st.markdown(f"## {ticker} &nbsp; <span style='font-size:1rem;color:#aaa'>{name}</span>",
                    unsafe_allow_html=True)
        price_str = _fmt_price(price)
        delta_str = _delta_str(chg)
        color = "green" if (chg or 0) >= 0 else "red"
        delta_md = f" &nbsp; :{color}[{delta_str}]" if delta_str else ""
        st.markdown(f"<span style='font-size:1.6rem; font-weight:700;'>{price_str}</span>{delta_md}",
                    unsafe_allow_html=True)


def _render_meridian(ticker: str) -> None:
    section_header("Meridian Signal")
    with st.spinner("Computing Meridian score…"):
        ms = compute_meridian_score(ticker)

    score_badge(ms["signal"], ms["composite"], ms["n_factors"])
    st.plotly_chart(factor_bar_chart(ms["factors"]), use_container_width=True)


def _render_risk(ticker: str) -> None:
    section_header("Risk Metrics (1Y)")
    with st.spinner("Computing risk metrics…"):
        rm = compute_risk_metrics(ticker)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Volatility (Ann.)", _pct(rm.get("volatility")))
    c2.metric("Beta vs SPY", _v(rm.get("beta"), ".2f"))
    c3.metric("Sharpe", _v(rm.get("sharpe"), ".2f"))
    c4.metric("Sortino", _v(rm.get("sortino"), ".2f"))
    c5.metric("Max Drawdown", _pct(rm.get("max_drawdown")))


def _render_price_chart(ticker: str) -> None:
    section_header("Price Chart")
    period_label = st.radio(
        "Period", list(_PERIOD_MAP.keys()),
        index=3, horizontal=True,
        key=f"an_period_{ticker}",
        label_visibility="collapsed",
    )
    period = _PERIOD_MAP[period_label]
    hist = get_price_history(ticker, period=period)
    if hist.empty:
        st.warning("Price history unavailable.")
    else:
        st.plotly_chart(plotly_price_chart(hist, ticker), use_container_width=True)


def _render_fundamentals(ticker: str) -> None:
    section_header("Fundamentals")
    fund = get_fundamentals(ticker)

    cols = st.columns(6)
    labels = ["P/E (TTM)", "Forward P/E", "EPS (TTM)", "Rev. Growth", "Profit Margin", "Debt/Equity"]
    values = [
        _v(fund.get("pe_ratio"), ".1f"),
        _v(fund.get("forward_pe"), ".1f"),
        _v(fund.get("eps"), ".2f", ""),
        _pct(fund.get("revenue_growth")),
        _pct(fund.get("profit_margin")),
        _v(fund.get("debt_to_equity"), ".1f"),
    ]
    for col, lbl, val in zip(cols, labels, values):
        col.metric(lbl, val)


def _render_news(ticker: str) -> None:
    section_header("Recent News")
    news = get_news(ticker, n=5)
    if not news:
        st.info("No recent news available.")
        return

    for item in news:
        title     = item.get("title") or "—"
        publisher = item.get("publisher") or ""
        link      = item.get("link") or ""
        pub_raw   = item.get("published")

        # Format date
        date_str = ""
        if pub_raw:
            try:
                if isinstance(pub_raw, (int, float)):
                    date_str = datetime.utcfromtimestamp(pub_raw).strftime("%b %d, %Y")
                else:
                    date_str = str(pub_raw)[:10]
            except Exception:
                date_str = ""

        meta = " · ".join(filter(None, [publisher, date_str]))
        if link:
            st.markdown(f"**[{title}]({link})**  \n<small style='color:#888'>{meta}</small>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"**{title}**  \n<small style='color:#888'>{meta}</small>",
                        unsafe_allow_html=True)
        st.divider()


def _render_recommendations(ticker: str) -> None:
    section_header("Analyst Ratings")
    rec = get_recommendations(ticker)
    if not rec:
        st.info("No analyst ratings available.")
        return

    strong_buy  = rec.get("strongBuy") or 0
    buy         = rec.get("buy") or 0
    hold        = rec.get("hold") or 0
    sell        = rec.get("sell") or 0
    strong_sell = rec.get("strongSell") or 0
    target      = rec.get("targetMeanPrice")

    total = strong_buy + buy + hold + sell + strong_sell

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Strong Buy", strong_buy if total else "—")
    c2.metric("Buy", buy if total else "—")
    c3.metric("Hold", hold if total else "—")
    c4.metric("Sell", sell if total else "—")
    c5.metric("Strong Sell", strong_sell if total else "—")
    c6.metric("Mean Target", _fmt_price(target))


def _render_insider(ticker: str) -> None:
    section_header("Insider Activity (Last 90 Days)")
    summary = get_insider_summary(ticker, days=90)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buys", summary["buys"] or "—")
    c2.metric("Sells", summary["sells"] or "—")
    c3.metric("Net Shares",
              f"{summary['net_shares']:+,.0f}" if summary["net_shares"] != 0 else "—")
    c4.metric("Net Ratio",
              f"{summary['net_ratio']:+.2f}" if summary["buys"] + summary["sells"] > 0 else "—")

    txns = summary.get("transactions", [])
    if txns:
        import pandas as pd
        df = pd.DataFrame(txns)
        # Clean up display
        rename = {}
        for col in df.columns:
            if "date" in col.lower():
                rename[col] = "Date"
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
        if rename:
            df = df.rename(columns=rename)
        # Drop URL column if present
        df = df.drop(columns=[c for c in ("URL", "url") if c in df.columns], errors="ignore")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No insider transactions found in the last 90 days.")


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_analysis_tab():
    st.subheader("Analysis")

    # ── Ticker input ───────────────────────────────────────────────────────
    with st.form("an_form"):
        c1, c2 = st.columns([5, 1])
        with c1:
            ticker_input = st.text_input(
                "Ticker",
                placeholder="Enter a ticker symbol, e.g. AAPL",
                label_visibility="collapsed",
            )
        with c2:
            analyze = st.form_submit_button("Analyze", use_container_width=True)

    if analyze and ticker_input.strip():
        st.session_state["an_ticker"] = ticker_input.strip().upper()

    ticker = st.session_state.get("an_ticker", "")
    if not ticker:
        st.info("Enter a ticker above and click Analyze to generate a full analysis card.")
        return

    # ── Analysis card ──────────────────────────────────────────────────────
    # 1 — Header
    _render_header(ticker)
    st.markdown("")

    # 2 — Meridian signal + factor bar chart
    _render_meridian(ticker)

    # 3 — Risk metrics row
    _render_risk(ticker)

    # 4 — Price chart with period toggle
    _render_price_chart(ticker)

    # 5 — Fundamentals table
    _render_fundamentals(ticker)

    # 6 — Recent news
    _render_news(ticker)

    # 7 — Analyst ratings
    _render_recommendations(ticker)

    # 8 — Insider activity
    _render_insider(ticker)
