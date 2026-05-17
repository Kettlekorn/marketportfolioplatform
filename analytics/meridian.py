"""Meridian-style 8-factor composite score → LONG / SHORT / NEUTRAL signal.

Factors:
  momentum          12-1 month price return
  value             1 / forward P/E
  quality           return on equity (ROE)
  growth            revenue YoY growth
  estimate_revisions analyst mean-target vs current price (proxy)
  short_interest    negative of short % of float (high short = bearish)
  insider           net insider buy ratio last 90 days
  institutional     institutional ownership %

Each raw value is normalised to a z-score using domain-calibrated centre/scale
parameters, then clipped to [-3, 3]. The composite is the unweighted mean of
all available z-scores. Missing factors are dropped from the average.
"""

import numpy as np

from data.market_data import (
    get_price_history,
    get_ticker_info,
    get_short_interest,
    get_institutional_ownership,
    get_roe,
    get_revenue_growth,
    get_analyst_target_change,
)
from data.insider_data import get_insider_summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _z(value: float | None, center: float, scale: float, clip: float = 3.0) -> float | None:
    """Normalise value to a z-score, clipped to ±clip. Returns None if value is None."""
    if value is None or scale == 0:
        return None
    try:
        z = (float(value) - center) / scale
        return float(max(-clip, min(clip, z)))
    except (TypeError, ValueError):
        return None


def _factor(label: str, raw, z_score) -> dict:
    available = raw is not None and z_score is not None
    return {"label": label, "raw": raw, "z_score": z_score, "available": available}


# ---------------------------------------------------------------------------
# Individual factor computations
# ---------------------------------------------------------------------------

def _momentum(ticker: str) -> dict:
    """12-1 month return: price 1 month ago / price 12 months ago - 1."""
    hist = get_price_history(ticker, period="1y")
    raw = None
    if not hist.empty and len(hist) >= 22:
        prices = hist["Close"]
        raw = float(prices.iloc[-21] / prices.iloc[0] - 1)
    return _factor("Momentum", raw, _z(raw, center=0.08, scale=0.30))


def _value(ticker: str) -> dict:
    """1 / forward P/E — higher means cheaper (more bullish)."""
    info = get_ticker_info(ticker)
    fpe = info.get("forwardPE")
    raw = None
    if fpe and fpe > 0:
        raw = 1.0 / float(fpe)
    # Market centre ≈ PE 20 → 1/PE = 0.05; scale = 0.025 (≈ 10 PE points)
    return _factor("Value (1/Fwd PE)", raw, _z(raw, center=0.05, scale=0.025))


def _quality(ticker: str) -> dict:
    """Return on equity."""
    raw = get_roe(ticker)
    if raw is not None:
        raw = float(raw)
    # 15 % ROE ≈ market neutral; scale = 0.20 (wide to handle outliers like AAPL)
    return _factor("Quality (ROE)", raw, _z(raw, center=0.15, scale=0.20))


def _growth(ticker: str) -> dict:
    """Revenue YoY growth."""
    raw = get_revenue_growth(ticker)
    if raw is not None:
        raw = float(raw)
    # 5 % = mature growth neutral; 15 % above = 1 std dev
    return _factor("Growth (Rev YoY)", raw, _z(raw, center=0.05, scale=0.15))


def _estimate_revisions(ticker: str) -> dict:
    """Analyst mean-target premium over current price (proxy for estimate revisions)."""
    raw = get_analyst_target_change(ticker)
    if raw is not None:
        raw = float(raw)
    # 0 % premium = neutral; ±15 % = ±1 std dev
    return _factor("Estimate Revisions", raw, _z(raw, center=0.0, scale=0.15))


def _short_interest(ticker: str) -> dict:
    """Negated short float % — high short = bearish."""
    short_pct = get_short_interest(ticker)
    raw = None
    if short_pct is not None:
        raw = -float(short_pct)   # negate: less short → more positive
    # 3 % short = neutral (-0.03); scale = 0.05
    return _factor("Short Interest", raw, _z(raw, center=-0.03, scale=0.05))


def _insider(ticker: str) -> dict:
    """Net insider buy ratio over last 90 days."""
    summary = get_insider_summary(ticker, days=90)
    raw = summary.get("net_ratio")  # (buys - sells) / (buys + sells)
    if raw is not None:
        raw = float(raw)
    # 0 = neutral (equal buys and sells); scale = 0.40
    return _factor("Insider Activity", raw, _z(raw, center=0.0, scale=0.40))


def _institutional(ticker: str) -> dict:
    """Institutional ownership fraction."""
    raw = get_institutional_ownership(ticker)
    if raw is not None:
        raw = float(raw)
    # 60 % ownership = neutral; scale = 0.15
    return _factor("Institutional Own.", raw, _z(raw, center=0.60, scale=0.15))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FACTOR_FUNCS = [
    ("momentum", _momentum),
    ("value", _value),
    ("quality", _quality),
    ("growth", _growth),
    ("estimate_revisions", _estimate_revisions),
    ("short_interest", _short_interest),
    ("insider", _insider),
    ("institutional", _institutional),
]


def compute_meridian_score(ticker: str) -> dict:
    """Compute the 8-factor Meridian composite for *ticker*.

    Returns:
        factors     dict[str, dict]   – per-factor label/raw/z_score/available
        composite   float             – mean z-score across available factors
        signal      str               – "LONG" | "SHORT" | "NEUTRAL"
        n_factors   int               – how many factors were available
    """
    factors: dict[str, dict] = {}

    for key, fn in FACTOR_FUNCS:
        try:
            factors[key] = fn(ticker)
        except Exception:
            factors[key] = _factor(key.replace("_", " ").title(), None, None)

    available_z = [
        f["z_score"] for f in factors.values()
        if f["available"] and f["z_score"] is not None
    ]

    composite = float(np.mean(available_z)) if available_z else 0.0

    if composite >= 0.5:
        signal = "LONG"
    elif composite <= -0.5:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    return {
        "factors": factors,
        "composite": composite,
        "signal": signal,
        "n_factors": len(available_z),
    }
