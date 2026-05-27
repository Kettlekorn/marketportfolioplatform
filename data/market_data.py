"""yfinance wrappers for prices, fundamentals, info, news, and recommendations.

All market data calls must go through this module — no yfinance imports elsewhere.
"""

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
import streamlit as st

_EDGAR_HEADERS = {"User-Agent": "StockResearchPlatform brandonjackwu9@gmail.com"}


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
        fast = yf.Ticker(ticker).fast_info
        price = fast.last_price
        prev = fast.previous_close
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

@st.cache_data(ttl=3600)
def _get_annual_revenue_growth(ticker: str) -> float | None:
    """Calculate YoY revenue growth from the two most recent annual income statements."""
    try:
        fin = yf.Ticker(ticker).financials  # columns = fiscal year dates, rows = line items
        if fin is None or fin.empty:
            return None
        rev_row = None
        for label in ("Total Revenue", "Revenue"):
            if label in fin.index:
                rev_row = fin.loc[label]
                break
        if rev_row is None or len(rev_row) < 2:
            return None
        rev_row = rev_row.sort_index(ascending=False)
        recent, prior = float(rev_row.iloc[0]), float(rev_row.iloc[1])
        if prior == 0:
            return None
        return (recent - prior) / abs(prior)
    except Exception:
        return None


@st.cache_data(ttl=600)
def get_fundamentals(ticker: str) -> dict:
    """Return key fundamental metrics. Any missing field is None."""
    info = get_ticker_info(ticker)
    return {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "revenue_growth": _get_annual_revenue_growth(ticker),
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
# Insider transactions — SEC EDGAR Form 4
# ---------------------------------------------------------------------------

_FORM4_CODES = {
    "P": "Purchase",
    "S": "Sale",
    "A": "Award/Grant",
    "M": "Option Exercise",
    "F": "Tax Withholding",
    "D": "Disposition",
    "G": "Gift",
}


@st.cache_data(ttl=86400, show_spinner=False)
def _edgar_cik(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_EDGAR_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def _xml_text(el, path: str) -> str | None:
    node = el.find(path)
    return node.text.strip() if node is not None and node.text else None


def _xml_float(el, path: str) -> float | None:
    node = el.find(path)
    if node is not None and node.text:
        try:
            return float(node.text.strip())
        except ValueError:
            pass
    return None


def _parse_form4_xml(xml_bytes: bytes, filing_date: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    owner  = _xml_text(root, ".//rptOwnerName") or "Unknown"
    title  = (_xml_text(root, ".//officerTitle")
              or ("Director" if _xml_text(root, ".//isDirector") == "1" else "Other"))

    rows = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code   = _xml_text(txn, ".//transactionCode") or "J"
        shares = _xml_float(txn, ".//transactionShares/value")
        price  = _xml_float(txn, ".//transactionPricePerShare/value")
        date   = _xml_text(txn, ".//transactionDate/value") or filing_date
        if shares is None:
            continue
        rows.append({
            "Date":        date,
            "Insider":     owner,
            "Position":    title,
            "Transaction": _FORM4_CODES.get(code, f"Other ({code})"),
            "Code":        code,
            "Shares":      abs(shares),
            "Value":       round(abs(shares) * price, 2) if price else None,
        })
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def get_insider_transactions_raw(ticker: str) -> pd.DataFrame:
    """Fetch Form 4 insider transactions directly from SEC EDGAR."""
    cik = _edgar_cik(ticker)
    if not cik:
        return pd.DataFrame()

    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_EDGAR_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})
    except Exception:
        return pd.DataFrame()

    forms    = recent.get("form", [])
    dates    = recent.get("filingDate", [])
    accs     = recent.get("accessionNumber", [])
    docs     = recent.get("primaryDocument", [])
    cutoff   = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    cik_int  = int(cik)

    all_rows, count = [], 0
    for form, date, acc, doc in zip(forms, dates, accs, docs):
        if form != "4" or date < cutoff:
            continue
        if count >= 60:
            break
        acc_nodash = acc.replace("-", "")
        # primaryDocument may have an xsl viewer prefix (e.g. xslF345X06/ownership.xml)
        # — strip it to get the raw XML path
        raw_doc = doc.split("/")[-1] if "/" in doc else doc
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{raw_doc}"
        try:
            xml_resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=10)
            xml_resp.raise_for_status()
            all_rows.extend(_parse_form4_xml(xml_resp.content, date))
            time.sleep(0.05)
        except Exception:
            continue
        count += 1

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


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
    """Return annual YoY revenue growth from income statements. None if unavailable."""
    return _get_annual_revenue_growth(ticker)


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
