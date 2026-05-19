"""yfinance wrappers for prices, fundamentals, info, news, and recommendations.

All market data calls must go through this module — no yfinance imports elsewhere.
"""

import pandas as pd
import yfinance as yf
import streamlit as st


# ---------------------------------------------------------------------------
# Core info / price
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_ticker_info(ticker: str) -> dict:
    """Return yfinance .info dict. Returns {} on any failure."""
    try:
        data = yf.Ticker(ticker).info
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=600)
def get_company_name(ticker: str) -> str:
    """Return short company name, falling back to ticker on failure."""
    info = get_ticker_info(ticker)
    return info.get("shortName") or info.get("longName") or ticker.upper()


@st.cache_data(ttl=600)
def get_current_price(ticker: str) -> tuple[float | None, float | None]:
    """Return (current_price, day_change_pct). Both None on failure."""
    try:
        info = get_ticker_info(ticker)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
        pct = (price - prev) / prev * 100 if price is not None and prev is not None and prev != 0 else None
        return price, pct
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Return auto-adjusted OHLCV DataFrame for a preset period string.

    period accepts yfinance values: 1d 5d 1mo 3mo 6mo 1y 2y 5y ytd max
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_price_history_range(
    ticker: str, start: str, end: str | None = None
) -> pd.DataFrame:
    """Return auto-adjusted OHLCV DataFrame between start and end (YYYY-MM-DD)."""
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_fundamentals(ticker: str) -> dict:
    """Return key fundamental metrics. Any missing field is None."""
    info = get_ticker_info(ticker)
    return {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity") / 100 if info.get("debtToEquity") is not None else None,
    }


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_news(ticker: str, n: int = 5) -> list[dict]:
    """Return up to n recent news items as a list of dicts.

    Each dict: title, publisher, link, published (str or int timestamp).
    Handles both the legacy flat format and the nested 'content' format
    introduced in yfinance 0.2.54+.
    """
    try:
        raw = yf.Ticker(ticker).news or []
        results = []
        for item in raw[:n]:
            c = item.get("content") or {}
            if c:
                provider = c.get("provider") or {}
                canon = c.get("canonicalUrl") or {}
                results.append({
                    "title": c.get("title", ""),
                    "publisher": provider.get("displayName", "") if isinstance(provider, dict) else "",
                    "link": canon.get("url", "") if isinstance(canon, dict) else "",
                    "published": c.get("pubDate", ""),
                })
            else:
                results.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": item.get("providerPublishTime", ""),
                })
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Analyst recommendations
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_recommendations(ticker: str) -> dict:
    """Return analyst ratings summary: strongBuy/buy/hold/sell/strongSell + mean target.

    Returns {} if data is unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        info = get_ticker_info(ticker)
        target = info.get("targetMeanPrice")

        rec = t.recommendations
        if rec is not None and not rec.empty and "period" in rec.columns:
            mask = rec["period"] == "0m"
            row = rec[mask].iloc[0] if mask.any() else rec.iloc[-1]
            return {
                "strongBuy": int(row.get("strongBuy", 0)),
                "buy": int(row.get("buy", 0)),
                "hold": int(row.get("hold", 0)),
                "sell": int(row.get("sell", 0)),
                "strongSell": int(row.get("strongSell", 0)),
                "targetMeanPrice": target,
            }
        return {"targetMeanPrice": target}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Insider transactions (raw)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_insider_transactions_raw(ticker: str) -> pd.DataFrame:
    """Return raw insider_transactions DataFrame from yfinance.

    Returns empty DataFrame on any failure. Processing lives in insider_data.py.
    """
    try:
        df = yf.Ticker(ticker).insider_transactions
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Meridian factor inputs
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_short_interest(ticker: str) -> float | None:
    """Return shortPercentOfFloat (0–1 scale). None if unavailable."""
    info = get_ticker_info(ticker)
    return info.get("shortPercentOfFloat")


@st.cache_data(ttl=600)
def get_institutional_ownership(ticker: str) -> float | None:
    """Return institutionPercentHeld (0–1 scale). None if unavailable."""
    info = get_ticker_info(ticker)
    return info.get("institutionsPercentHeld") or info.get("heldPercentInstitutions")


@st.cache_data(ttl=600)
def get_roe(ticker: str) -> float | None:
    """Return returnOnEquity. None if unavailable."""
    info = get_ticker_info(ticker)
    return info.get("returnOnEquity")


@st.cache_data(ttl=600)
def get_revenue_growth(ticker: str) -> float | None:
    """Return revenueGrowth YoY. None if unavailable."""
    info = get_ticker_info(ticker)
    return info.get("revenueGrowth")


@st.cache_data(ttl=600)
def get_analyst_target_change(ticker: str) -> float | None:
    """Estimate analyst target-price change % vs current price as a proxy for estimate revisions."""
    try:
        info = get_ticker_info(ticker)
        target = info.get("targetMeanPrice")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if target and price:
            return (target - price) / price
        return None
    except Exception:
        return None
