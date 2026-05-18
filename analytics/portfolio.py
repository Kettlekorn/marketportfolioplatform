"""Portfolio analytics: P/L per position, allocation, sector breakdown, vs-benchmark series.

All functions accept portfolio_rows as returned by data.storage.get_portfolio().
No database calls are made here — the tab layer handles storage queries.
"""

import numpy as np
import pandas as pd

from config import BENCHMARK_TICKER, RISK_FREE_RATE
from data.market_data import get_current_price, get_ticker_info, get_price_history_range
from analytics.risk import (
    annualized_volatility,
    beta_vs_benchmark,
    sharpe_ratio,
    compute_risk_metrics,
)


# ---------------------------------------------------------------------------
# Position enrichment
# ---------------------------------------------------------------------------

def compute_positions(portfolio_rows: list[dict]) -> list[dict]:
    """Aggregate all lots per ticker into one enriched position dict.

    Multiple lots for the same ticker are combined:
        total_shares  = sum of shares across all lots
        avg_cost      = weighted average cost  (Σ price_bought*shares / Σ shares)
        current_value = current_price * total_shares
        pl_dollar     = (current_price - avg_cost) * total_shares
        pl_pct        = (current_price - avg_cost) / avg_cost

    Output dict keys:
        ticker, shares, avg_cost, current_price, current_value,
        pl_dollar, pl_pct, sector, weight_pct, lots (raw list of lot dicts)
    """
    if not portfolio_rows:
        return []

    # Group lots by ticker (preserving insertion order via dict)
    from collections import defaultdict
    lots_by_ticker: dict[str, list[dict]] = defaultdict(list)
    for row in portfolio_rows:
        lots_by_ticker[row["ticker"]].append(row)

    enriched = []
    for ticker, lots in lots_by_ticker.items():
        total_shares = sum(lot.get("shares", 1.0) for lot in lots)
        avg_cost = (
            sum(lot["price_bought"] * lot.get("shares", 1.0) for lot in lots) / total_shares
            if total_shares else 0.0
        )

        current_price, _ = get_current_price(ticker)
        if current_price is None:
            current_price = avg_cost  # fall back to cost basis

        info = get_ticker_info(ticker)
        sector = info.get("sector") or "Unknown"

        current_value = current_price * total_shares
        pl_dollar     = (current_price - avg_cost) * total_shares
        pl_pct        = (current_price - avg_cost) / avg_cost if avg_cost else 0.0

        enriched.append({
            "ticker":        ticker,
            "shares":        total_shares,
            "avg_cost":      avg_cost,
            "current_price": current_price,
            "current_value": current_value,
            "pl_dollar":     pl_dollar,
            "pl_pct":        pl_pct,
            "sector":        sector,
            "lots":          lots,   # raw lot rows, used by portfolio tab for lot detail
        })

    total_value = sum(p["current_value"] for p in enriched)
    for pos in enriched:
        pos["weight_pct"] = pos["current_value"] / total_value if total_value > 0 else 0.0

    return enriched


def aggregate_totals(positions: list[dict]) -> dict:
    """Compute portfolio-level cost basis, current value, and P/L from enriched positions."""
    if not positions:
        return {"cost_basis": 0.0, "current_value": 0.0, "pl_dollar": 0.0, "pl_pct": 0.0}

    # cost_basis = avg_cost * shares  (already aggregated per ticker in compute_positions)
    cost_basis    = sum(p["avg_cost"] * p["shares"] for p in positions)
    current_value = sum(p["current_value"] for p in positions)
    pl_dollar     = current_value - cost_basis
    pl_pct        = pl_dollar / cost_basis if cost_basis else 0.0

    return {
        "cost_basis":    cost_basis,
        "current_value": current_value,
        "pl_dollar":     pl_dollar,
        "pl_pct":        pl_pct,
    }


# ---------------------------------------------------------------------------
# Portfolio vs benchmark return series
# ---------------------------------------------------------------------------

def portfolio_vs_benchmark(
    portfolio_rows: list[dict],
    benchmark: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Build equal-weight cumulative return series rebased to 100.

    Each ticker contributes returns only from its own purchase date, so
    stocks bought recently don't retroactively inflate the portfolio history.

    Returns a DataFrame with columns "Portfolio" and the benchmark ticker.
    Returns empty DataFrame if price data cannot be fetched.
    """
    if not portfolio_rows:
        return pd.DataFrame()

    earliest = min(r["date_bought"] for r in portfolio_rows)
    start = earliest.strftime("%Y-%m-%d")

    # Collect daily returns per ticker, each starting from its own purchase date
    returns_by_ticker: dict[str, pd.Series] = {}
    for row in portfolio_rows:
        ticker = row["ticker"]
        ticker_start = row["date_bought"].strftime("%Y-%m-%d")
        hist = get_price_history_range(ticker, start=ticker_start)
        if not hist.empty:
            returns_by_ticker[ticker] = hist["Close"].pct_change()

    if not returns_by_ticker:
        return pd.DataFrame()

    # Equal-weight mean of whichever tickers are active each day (skipna=True)
    # Tickers not yet purchased have NaN returns and are automatically excluded.
    df_returns = pd.DataFrame(returns_by_ticker)
    port_daily = df_returns.mean(axis=1)

    # Drop the leading NaN row (first trading day has no prior close for any ticker)
    port_daily = port_daily.dropna()
    if port_daily.empty:
        return pd.DataFrame()

    portfolio_line = (1 + port_daily).cumprod() * 100
    result = pd.DataFrame({"Portfolio": portfolio_line})

    # Benchmark — fetch from the overall start so the chart spans the same window
    bench_hist = get_price_history_range(benchmark, start=start)
    if not bench_hist.empty:
        bench = bench_hist["Close"].reindex(result.index, method="ffill")
        first_valid = bench.first_valid_index()
        if first_valid is not None:
            result[benchmark] = bench / bench.loc[first_valid] * 100

    return result


# ---------------------------------------------------------------------------
# Aggregated risk metrics
# ---------------------------------------------------------------------------

def aggregate_risk_metrics(portfolio_rows: list[dict]) -> dict:
    """Compute weighted portfolio beta, weighted volatility, and portfolio-level Sharpe.

    Beta and volatility are value-weighted averages across positions.
    Sharpe is computed directly from the portfolio return series.
    """
    if not portfolio_rows:
        return {"portfolio_beta": None, "portfolio_volatility": None, "portfolio_sharpe": None}

    positions = compute_positions(portfolio_rows)
    total_value = sum(p["current_value"] for p in positions)

    w_beta = 0.0
    w_vol = 0.0
    beta_weight_sum = 0.0
    vol_weight_sum = 0.0

    for pos in positions:
        w = pos["current_value"] / total_value if total_value > 0 else 0.0
        metrics = compute_risk_metrics(pos["ticker"])

        if metrics.get("beta") is not None:
            w_beta += w * metrics["beta"]
            beta_weight_sum += w

        if metrics.get("volatility") is not None:
            w_vol += w * metrics["volatility"]
            vol_weight_sum += w

    portfolio_beta = w_beta / beta_weight_sum if beta_weight_sum > 0 else None
    portfolio_vol = w_vol / vol_weight_sum if vol_weight_sum > 0 else None

    # Sharpe from actual portfolio return series
    portfolio_sharpe = None
    vs_spy = portfolio_vs_benchmark(portfolio_rows)
    if not vs_spy.empty and "Portfolio" in vs_spy.columns:
        port_prices = vs_spy["Portfolio"]
        port_returns = port_prices.pct_change().dropna()
        if len(port_returns) >= 20:
            portfolio_sharpe = sharpe_ratio(port_returns, RISK_FREE_RATE)

    return {
        "portfolio_beta": portfolio_beta,
        "portfolio_volatility": portfolio_vol,
        "portfolio_sharpe": portfolio_sharpe,
    }


# ---------------------------------------------------------------------------
# Rolling volatility series
# ---------------------------------------------------------------------------

def portfolio_rolling_volatility(
    portfolio_rows: list[dict],
    window: int = 21,
    benchmark: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Rolling annualised volatility for the combined equal-weight portfolio and benchmark.

    Uses the same price series as portfolio_vs_benchmark so the two charts are consistent.
    Returns a DataFrame with columns "Portfolio" and the benchmark ticker.
    """
    vs = portfolio_vs_benchmark(portfolio_rows, benchmark)
    if vs.empty:
        return pd.DataFrame()

    result: dict[str, pd.Series] = {}
    if "Portfolio" in vs.columns:
        result["Portfolio"] = vs["Portfolio"].pct_change().rolling(window).std() * np.sqrt(252)
    if benchmark in vs.columns:
        result[benchmark] = vs[benchmark].pct_change().rolling(window).std() * np.sqrt(252)

    return pd.DataFrame(result).dropna(how="all")


def individual_rolling_volatilities(
    portfolio_rows: list[dict],
    window: int = 21,
    benchmark: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Rolling annualised volatility per ticker plus the benchmark.

    Each ticker's series starts from the portfolio's earliest purchase date.
    Returns a DataFrame with one column per ticker plus the benchmark.
    """
    if not portfolio_rows:
        return pd.DataFrame()

    earliest = min(r["date_bought"] for r in portfolio_rows)
    start = earliest.strftime("%Y-%m-%d")

    series: dict[str, pd.Series] = {}
    for row in portfolio_rows:
        ticker = row["ticker"]
        hist = get_price_history_range(ticker, start=start)
        if not hist.empty:
            series[ticker] = (
                hist["Close"].pct_change().rolling(window).std() * np.sqrt(252)
            )

    bench_hist = get_price_history_range(benchmark, start=start)
    if not bench_hist.empty:
        series[benchmark] = (
            bench_hist["Close"].pct_change().rolling(window).std() * np.sqrt(252)
        )

    if not series:
        return pd.DataFrame()

    return pd.DataFrame(series).dropna(how="all")
