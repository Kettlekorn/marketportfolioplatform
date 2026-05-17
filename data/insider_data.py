"""Insider transaction processing. Raw data sourced from market_data; no yfinance here."""

import pandas as pd

from data.market_data import get_insider_transactions_raw


def get_insider_summary(ticker: str, days: int = 90) -> dict:
    """Return insider buy/sell summary over the last ``days`` days.

    Result keys:
        buys         int   – number of buy transactions
        sells        int   – number of sell transactions
        buy_shares   float – total shares purchased
        sell_shares  float – total shares sold
        net_shares   float – buy_shares - sell_shares
        net_ratio    float – (buys - sells) / (buys + sells), or 0 if none
        transactions list  – raw rows as list[dict] (capped at 20 rows)
    """
    empty = {
        "buys": 0, "sells": 0,
        "buy_shares": 0.0, "sell_shares": 0.0,
        "net_shares": 0.0, "net_ratio": 0.0,
        "transactions": [],
    }

    df = get_insider_transactions_raw(ticker)
    if df.empty:
        return empty

    df = df.copy().reset_index()

    # ── locate date column ────────────────────────────────────────────────
    date_col = next(
        (c for c in df.columns if "date" in c.lower()),
        None,
    )
    if date_col is None:
        # Try the reset index column (yfinance sometimes puts date as index)
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
                if parsed.notna().sum() > len(df) * 0.5:
                    date_col = c
                    break
            except Exception:
                continue

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        recent = df[df[date_col] >= cutoff].copy()
    else:
        recent = df.copy()

    if recent.empty:
        return empty

    # ── locate transaction-type and shares columns ────────────────────────
    txn_col = next(
        (c for c in recent.columns if c.lower() in ("transaction", "type", "text")),
        None,
    )
    shares_col = next(
        (c for c in recent.columns if c.lower() == "shares"),
        None,
    )

    buys = sells = 0
    buy_shares = sell_shares = 0.0

    if txn_col and shares_col:
        txn_text = recent[txn_col].astype(str)
        is_buy = txn_text.str.contains(r"Buy|Purchase|Acquisition", case=False, na=False)
        is_sell = txn_text.str.contains(r"Sale|Sell|Sold|Disposition", case=False, na=False)

        shares = pd.to_numeric(recent[shares_col], errors="coerce").fillna(0).abs()

        buys = int(is_buy.sum())
        sells = int(is_sell.sum())
        buy_shares = float(shares[is_buy].sum())
        sell_shares = float(shares[is_sell].sum())

    net_shares = buy_shares - sell_shares
    total_txns = buys + sells
    net_ratio = (buys - sells) / total_txns if total_txns > 0 else 0.0

    # ── build display rows ────────────────────────────────────────────────
    display_cols = [
        c for c in ["Insider", "Position", "Transaction", "Text", "Shares", "Value", date_col]
        if c and c in recent.columns
    ]
    transactions = [
        {col: row.get(col) for col in display_cols}
        for _, row in recent.head(20).iterrows()
    ]

    return {
        "buys": buys,
        "sells": sells,
        "buy_shares": buy_shares,
        "sell_shares": sell_shares,
        "net_shares": net_shares,
        "net_ratio": net_ratio,
        "transactions": transactions,
    }
