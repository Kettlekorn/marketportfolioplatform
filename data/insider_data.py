"""Insider transaction processing. Raw data sourced from SEC EDGAR via market_data."""

import pandas as pd

from data.market_data import get_insider_transactions_raw


def get_insider_summary(ticker: str, days: int | None = 90) -> dict:
    """Return insider buy/sell summary over the last ``days`` days (None = all time).

    Only open-market purchases (Code P) and sales (Code S) count toward
    buy/sell sentiment. Awards, option exercises, and tax withholding are
    included in the transaction table but excluded from the counts.

    Result keys:
        buys         int   – open-market purchase transactions
        sells        int   – open-market sale transactions
        buy_shares   float – total shares purchased
        sell_shares  float – total shares sold
        net_shares   float – buy_shares - sell_shares
        net_ratio    float – (buy_shares - sell_shares) / total_shares, or 0
        transactions list  – all rows as list[dict] (capped at 50)
    """
    empty = {
        "buys": 0, "sells": 0,
        "buy_shares": 0.0, "sell_shares": 0.0,
        "net_shares": 0.0, "net_ratio": 0.0,
        "transactions": [],
    }

    df = get_insider_transactions_raw(ticker)
    if df.empty or "Date" not in df.columns:
        return empty

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    if days is not None:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        recent = df[df["Date"] >= cutoff].copy()
    else:
        recent = df.copy()
    if recent.empty:
        return empty

    shares  = pd.to_numeric(recent["Shares"], errors="coerce").fillna(0).abs()
    is_buy  = recent["Code"] == "P"
    is_sell = recent["Code"] == "S"

    buys        = int(is_buy.sum())
    sells       = int(is_sell.sum())
    buy_shares  = float(shares[is_buy].sum())
    sell_shares = float(shares[is_sell].sum())
    net_shares  = buy_shares - sell_shares
    total_shares = buy_shares + sell_shares
    net_ratio    = (buy_shares - sell_shares) / total_shares if total_shares > 0 else 0.0

    recent_display = recent.copy()
    recent_display["Date"] = recent_display["Date"].dt.strftime("%Y-%m-%d")
    display_cols = [c for c in ["Insider", "Position", "Transaction", "Shares", "Value", "Date"]
                    if c in recent_display.columns]
    transactions = recent_display[display_cols].head(50).to_dict("records")

    return {
        "buys":        buys,
        "sells":       sells,
        "buy_shares":  buy_shares,
        "sell_shares": sell_shares,
        "net_shares":  net_shares,
        "net_ratio":   net_ratio,
        "transactions": transactions,
    }
