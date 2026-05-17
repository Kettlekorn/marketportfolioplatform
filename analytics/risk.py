"""Risk metrics: annualised volatility, beta vs benchmark, Sharpe, Sortino, max drawdown."""

import numpy as np
import pandas as pd

from config import BENCHMARK_TICKER, RISK_FREE_RATE
from data.market_data import get_price_history


# ---------------------------------------------------------------------------
# Pure metric functions (operate on pd.Series of daily returns / prices)
# ---------------------------------------------------------------------------

def annualized_volatility(returns: pd.Series) -> float | None:
    if returns.empty or len(returns) < 5:
        return None
    return float(returns.std() * np.sqrt(252))


def beta_vs_benchmark(
    returns: pd.Series, benchmark_returns: pd.Series
) -> float | None:
    """Covariance-based beta; requires at least 30 overlapping observations."""
    r, b = returns.align(benchmark_returns, join="inner")
    mask = r.notna() & b.notna()
    r, b = r[mask], b[mask]
    if len(r) < 30:
        return None
    cov = np.cov(r.values, b.values)
    bench_var = cov[1, 1]
    return float(cov[0, 1] / bench_var) if bench_var != 0 else None


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE
) -> float | None:
    if returns.empty or len(returns) < 5:
        return None
    daily_rf = risk_free_rate / 252
    std = returns.std()
    if std == 0:
        return None
    excess_mean = (returns - daily_rf).mean()
    return float(excess_mean / std * np.sqrt(252))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE
) -> float | None:
    if returns.empty or len(returns) < 5:
        return None
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty:
        return None
    downside_std = float(np.sqrt((downside ** 2).mean()))
    if downside_std == 0:
        return None
    return float(excess.mean() / downside_std * np.sqrt(252))


def max_drawdown(prices: pd.Series) -> float | None:
    """Return maximum peak-to-trough drawdown as a negative fraction."""
    if prices.empty or len(prices) < 2:
        return None
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# Convenience wrapper: fetch data + compute all metrics in one call
# ---------------------------------------------------------------------------

def compute_risk_metrics(ticker: str, period: str = "1y") -> dict:
    """Fetch 1-year price history and return all risk metrics as a dict.

    Missing metrics are None. Keys: volatility, beta, sharpe, sortino, max_drawdown.
    """
    hist = get_price_history(ticker, period=period)
    spy_hist = get_price_history(BENCHMARK_TICKER, period=period)

    if hist.empty:
        return {
            "volatility": None,
            "beta": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
        }

    prices = hist["Close"]
    returns = prices.pct_change().dropna()

    spy_returns = (
        spy_hist["Close"].pct_change().dropna() if not spy_hist.empty else pd.Series(dtype=float)
    )

    return {
        "volatility": annualized_volatility(returns),
        "beta": beta_vs_benchmark(returns, spy_returns) if not spy_returns.empty else None,
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(prices),
    }
