"""Regime detection via Hidden Markov Model with causal forward filtering.

No look-ahead bias: regime labels at time t depend only on observations 0..t.
"""

import warnings
warnings.filterwarnings("ignore", message=".*not converging.*", module="hmmlearn")
warnings.filterwarnings("ignore", message=".*Model is not converging.*")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from hmmlearn import hmm
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from arch import arch_model

FEATURE_COLS = ["log_return", "realized_vol", "volume_ratio", "hl_range"]

_REGIME_LABELS = ["Low Vol", "Medium Vol", "High Vol", "Very High Vol", "Extreme Vol", "Ultra High Vol", "Max Vol"]
_REGIME_COLORS = ["#00BFFF", "#FFD700", "#FF4500", "#9B59B6", "#2ECC71", "#E74C3C", "#F39C12"]


def _regime_palette(n: int) -> tuple[list[str], list[str]]:
    return _REGIME_LABELS[:n], _REGIME_COLORS[:n]


# ---------------------------------------------------------------------------
# Data & features
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["log_return"], inplace=True)

    scaled = df["log_return"] * 100
    try:
        res = arch_model(scaled, vol="Garch", p=1, o=1, q=1, dist="normal").fit(disp="off")
        df["realized_vol"] = res.conditional_volatility / 100
    except Exception as e:
        raise ValueError(
            f"GARCH model failed to fit — try a longer date range or a different ticker. Detail: {e}"
        ) from e

    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# HMM training
# ---------------------------------------------------------------------------

def _bic_score(model: hmm.GaussianHMM, X: np.ndarray) -> float:
    n, f = X.shape
    k = model.n_components
    # diag covariance: k*f variance params (not k*f*(f+1)/2 which is for full matrices)
    n_params = (k - 1) + k * (k - 1) + k * f + k * f
    return -2 * model.score(X) * n + n_params * np.log(n)


def _train_best_hmm(
    X: np.ndarray, n_override: int | None = None
) -> tuple[hmm.GaussianHMM | None, int | None, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    candidates = [n_override] if n_override else range(3, 8)
    best_model, best_bic, best_n = None, np.inf, None

    for n in candidates:
        best_run, best_run_ll = None, -np.inf
        for seed in range(5):
            try:
                m = hmm.GaussianHMM(
                    n_components=n, covariance_type="diag",
                    n_iter=500, random_state=seed, tol=1e-4,
                )
                m.fit(Xs)
                ll = m.score(Xs)
                if ll > best_run_ll:
                    best_run_ll, best_run = ll, m
            except Exception:
                continue
        if best_run is None:
            continue
        b = _bic_score(best_run, Xs)
        if b < best_bic:
            best_bic, best_model, best_n = b, best_run, n

    return best_model, best_n, scaler


# ---------------------------------------------------------------------------
# Forward algorithm (causal, no look-ahead)
# ---------------------------------------------------------------------------

def _forward_filter(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Return normalised state probabilities shape (T, K) using only past observations."""
    T = len(X)
    K = model.n_components
    log_emit = np.column_stack([
        multivariate_normal.logpdf(X, mean=model.means_[k], cov=np.diag(model.covars_[k]))
        for k in range(model.n_components)
    ])
    log_trans = np.log(model.transmat_ + 1e-300)
    log_init = np.log(model.startprob_ + 1e-300)

    log_alpha = np.empty((T, K))
    log_alpha[0] = log_init + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = (
            np.logaddexp.reduce(log_alpha[t - 1][:, None] + log_trans, axis=0)
            + log_emit[t]
        )

    alpha = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
    alpha /= alpha.sum(axis=1, keepdims=True)
    return alpha


# ---------------------------------------------------------------------------
# Stability filter
# ---------------------------------------------------------------------------

def _stability_filter(
    raw: list[int], persist: int = 3, window: int = 20, threshold: int = 4
) -> tuple[list[int], list[bool]]:
    n = len(raw)
    confirmed = [raw[0]] * n
    run = 1
    for i in range(1, n):
        run = run + 1 if raw[i] == raw[i - 1] else 1
        confirmed[i] = raw[i] if run >= persist else confirmed[i - 1]

    uncertain = [False] * n
    for i in range(1, n):
        look = min(i, window)
        seg = confirmed[i - look:i]
        transitions = sum(seg[j] != seg[j - 1] for j in range(1, look))
        uncertain[i] = transitions > threshold

    return confirmed, uncertain


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def compute_regimes(
    ticker: str, start: str, end: str, n_override: int | None = None
) -> dict | None:
    """Fetch data, fit HMM, return regime results dict (or None on failure)."""
    raw = _fetch_ohlcv(ticker, start, end)
    if raw is None or raw.empty:
        return None

    df = _engineer_features(raw)
    if len(df) < 60:
        return None

    X = df[FEATURE_COLS].values
    model, n, scaler = _train_best_hmm(X, n_override)
    if model is None:
        return None

    Xs = scaler.transform(X)
    probs = _forward_filter(model, Xs)
    raw_states = np.argmax(probs, axis=1).tolist()
    confidence = probs.max(axis=1).tolist()

    labels, colors = _regime_palette(n)
    order = np.argsort(model.means_[:, 1])
    state_to_label = {int(order[i]): labels[i] for i in range(n)}
    state_to_color = {int(order[i]): colors[i] for i in range(n)}
    color_map = {labels[i]: colors[i] for i in range(n)}

    confirmed, uncertain = _stability_filter(raw_states)
    reg_labels = [state_to_label[s] for s in confirmed]
    reg_colors = [state_to_color[s] for s in confirmed]

    log_rets = df["log_return"].tolist()
    real_vols = df["realized_vol"].tolist()
    vol_rats = df["volume_ratio"].tolist()

    stats: dict[str, dict] = {}
    for lbl, col in color_map.items():
        idx = [i for i, l in enumerate(reg_labels) if l == lbl]
        if not idx:
            continue
        stats[lbl] = {
            "mean_return": float(np.mean([log_rets[i] for i in idx])) * 100,
            "mean_vol": float(np.mean([real_vols[i] for i in idx])) * 100,
            "mean_vol_ratio": float(np.mean([vol_rats[i] for i in idx])),
            "pct_time": len(idx) / len(reg_labels) * 100,
            "color": col,
        }

    return {
        "ticker":      ticker,
        "dates":       df.index.strftime("%Y-%m-%d").tolist(),
        "opens":       df["Open"].squeeze().tolist(),
        "highs":       df["High"].squeeze().tolist(),
        "lows":        df["Low"].squeeze().tolist(),
        "closes":      df["Close"].squeeze().tolist(),
        "volumes":     df["Volume"].squeeze().tolist(),
        "daily_returns": (df["Close"].pct_change() * 100).fillna(0).round(4).tolist(),
        "realized_vols": [round(v * 100, 4) for v in real_vols],
        "reg_labels":  reg_labels,
        "reg_colors":  reg_colors,
        "confidence":  confidence,
        "stats":       stats,
        "n_regimes":   len(stats),
        "cur_label":   reg_labels[-1],
        "cur_conf":    confidence[-1] * 100,
        "stability":   "Uncertain" if uncertain[-1] else "Stable",
        "color_map":   color_map,
    }
