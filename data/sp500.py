"""S&P 500 constituent list and batch factor rankings."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf
import streamlit as st
from bs4 import BeautifulSoup

# Factors supported without EDGAR calls
BATCHABLE_FACTORS = {
    "momentum", "value", "quality", "growth",
    "estimate_revisions", "short_interest", "institutional",
}


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_meta() -> pd.DataFrame:
    """Ticker, name, GICS sector from Wikipedia. Cached 24 h."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        rows = table.find_all("tr")
        data = []
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) >= 3:
                data.append({
                    "ticker": cols[0].get_text(strip=True).replace(".", "-"),
                    "name":   cols[1].get_text(strip=True),
                    "sector": cols[2].get_text(strip=True),
                })
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(columns=["ticker", "name", "sector"])


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_bulk_info() -> dict:
    """Fetch yfinance .info for all S&P 500 tickers via 20 threads. First load ~60-90 s. Cached 24 h."""
    meta = get_sp500_meta()
    if meta.empty:
        return {}

    tickers = meta["ticker"].tolist()

    def _get(t: str) -> tuple[str, dict]:
        try:
            info = yf.Ticker(t).info
            return t, {
                "forwardPE":           info.get("forwardPE"),
                "returnOnEquity":      info.get("returnOnEquity"),
                "shortPercentOfFloat": info.get("shortPercentOfFloat"),
                "institutionsPct":     info.get("institutionsPercentHeld") or info.get("heldPercentInstitutions"),
                "targetMeanPrice":     info.get("targetMeanPrice"),
                "currentPrice":        info.get("currentPrice") or info.get("regularMarketPrice"),
                "revenueGrowth":       info.get("revenueGrowth"),
            }
        except Exception:
            return t, {}

    with ThreadPoolExecutor(max_workers=20) as ex:
        return dict(ex.map(_get, tickers))


@st.cache_data(ttl=86400, show_spinner=False)
def get_momentum_rankings() -> pd.DataFrame:
    """12-1 month momentum for all S&P 500 via one batch price download. Cached 24 h."""
    meta = get_sp500_meta()
    if meta.empty:
        return pd.DataFrame()

    tickers = meta["ticker"].tolist()
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False, threads=True)
    prices = (
        raw["Close"]
        if isinstance(raw.columns, pd.MultiIndex)
        else raw[["Close"]].rename(columns={"Close": tickers[0]})
    )

    lookup = meta.set_index("ticker").to_dict("index")
    rows = []
    for t in tickers:
        if t not in prices.columns:
            continue
        col = prices[t].dropna()
        if len(col) < 22:
            continue
        m = lookup.get(t, {})
        rows.append({
            "ticker": t,
            "name":   m.get("name", ""),
            "sector": m.get("sector", ""),
            "value":  float(col.iloc[-21] / col.iloc[0] - 1),
        })

    df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def build_factor_rankings(factor_key: str, bulk_info: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Compute S&P 500 rankings for a factor from pre-fetched bulk info. Not cached — data already is."""
    lookup = meta.set_index("ticker").to_dict("index")

    def _extract(info: dict) -> float | None:
        if factor_key == "value":
            fpe = info.get("forwardPE")
            return 1 / fpe if fpe and fpe > 0 else None
        if factor_key == "quality":
            return info.get("returnOnEquity")
        if factor_key == "growth":
            return info.get("revenueGrowth")
        if factor_key == "estimate_revisions":
            tp, cp = info.get("targetMeanPrice"), info.get("currentPrice")
            return (tp - cp) / cp if tp and cp and cp > 0 else None
        if factor_key == "short_interest":
            s = info.get("shortPercentOfFloat")
            return -s if s is not None else None
        if factor_key == "institutional":
            return info.get("institutionsPct")
        return None

    rows = []
    for t, info in bulk_info.items():
        val = _extract(info)
        if val is None:
            continue
        m = lookup.get(t, {})
        rows.append({"ticker": t, "name": m.get("name", ""), "sector": m.get("sector", ""), "value": val})

    df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
