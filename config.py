"""Application-wide constants: database path, benchmark ticker, risk-free rate."""

from pathlib import Path

_BASE_DIR = Path(__file__).parent

DB_PATH = f"sqlite:///{(_BASE_DIR / 'data' / 'app.db').as_posix()}"
BENCHMARK_TICKER = "VOO"
RISK_FREE_RATE = 0.05  # 5% annual, used for Sharpe / Sortino
