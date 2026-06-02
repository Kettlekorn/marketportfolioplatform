"""Analysis tab: full 8-section analysis card for any ticker."""

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.meridian import compute_meridian_score
from analytics.regime import compute_regimes
from analytics.risk import compute_risk_metrics
from components.cards import score_badge, section_header
from components.charts import factor_bar_chart, plotly_price_chart
from data.insider_data import get_insider_summary
from data.market_data import (
    get_current_price,
    get_fundamentals,
    get_news,
    get_price_history,
    get_recommendations,
    get_ticker_info,
)
from data.sp500 import get_momentum_rankings, get_sp500_bulk_info, get_sp500_meta, build_factor_rankings

_PERIOD_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _v(val, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a value or return '—' when None / NaN."""
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _pct(val) -> str:
    return _v(val, ".1%") if val is not None else "—"


def _fmt_price(val) -> str:
    return f"${val:,.2f}" if val is not None else "—"


def _delta_str(chg: float | None) -> str | None:
    if chg is None:
        return None
    return f"{chg:+.2f}%"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(ticker: str) -> None:
    info = get_ticker_info(ticker)
    name = info.get("shortName") or info.get("longName") or ticker
    price, chg = get_current_price(ticker)

    col_title, col_badge = st.columns([3, 2])
    with col_title:
        yf_url = f"https://finance.yahoo.com/quote/{ticker}"
        st.markdown(
            f"## <a href='{yf_url}' target='_blank' style='color:inherit;text-decoration:none;'>{ticker}</a>"
            f" &nbsp; <span style='font-size:1rem;color:#aaa'>{name}</span>",
            unsafe_allow_html=True,
        )
        price_str = _fmt_price(price)
        delta_str = _delta_str(chg)
        color = "green" if (chg or 0) >= 0 else "red"
        delta_md = f" &nbsp; :{color}[{delta_str}]" if delta_str else ""
        st.markdown(f"<span style='font-size:1.6rem; font-weight:700;'>{price_str}</span>{delta_md}",
                    unsafe_allow_html=True)


def _render_meridian(ticker: str) -> None:
    section_header("Meridian Signal")
    with st.spinner("Computing Meridian score…"):
        ms = compute_meridian_score(ticker)

    score_badge(ms["signal"], ms["composite"], ms["n_factors"], ms["n_total"])

    event = st.plotly_chart(
        factor_bar_chart(ms["factors"]),
        use_container_width=True,
        on_select="rerun",
        key=f"factor_chart_{ticker}",
    )

    clicked = None
    if event and event.selection and event.selection.points:
        clicked = event.selection.points[0].get("customdata")

    if clicked:
        _render_factor_ranking(clicked)


_FACTOR_LABELS = {
    "momentum": "Momentum",
    "value": "Value (1/Fwd PE)",
    "quality": "Quality (ROE)",
    "growth": "Growth (Rev YoY)",
    "estimate_revisions": "Estimate Revisions",
    "short_interest": "Short Interest",
    "insider": "Insider Activity",
    "institutional": "Institutional Own.",
}


_FACTOR_FMT = {
    "momentum":          lambda x: f"{x:+.1%}",
    "value":             lambda x: f"{x:.4f}",
    "quality":           lambda x: f"{x:+.1%}",
    "growth":            lambda x: f"{x:+.1%}",
    "estimate_revisions":lambda x: f"{x:+.1%}",
    "short_interest":    lambda x: f"{-x:.1%} short",
    "institutional":     lambda x: f"{x:.1%}",
}

_FACTOR_COL = {
    "momentum":           "12-1M Return",
    "value":              "1/Fwd PE",
    "quality":            "ROE",
    "growth":             "Rev Growth",
    "estimate_revisions": "Target Upside",
    "short_interest":     "Short %",
    "institutional":      "Inst. Owned",
}


def _render_factor_ranking(factor_key: str) -> None:
    label = _FACTOR_LABELS.get(factor_key, factor_key)
    st.markdown(f"#### S&P 500 — {label} Ranking")

    if factor_key == "insider":
        st.info("Insider Activity ranking requires SEC EDGAR queries for 500 companies — not available.")
        return

    if factor_key == "momentum":
        with st.spinner("Loading momentum data… (cached after first load)"):
            df = get_momentum_rankings()
    else:
        with st.spinner("Loading S&P 500 fundamental data… first load takes ~60-90 s, cached 24 h"):
            bulk = get_sp500_bulk_info()
            meta = get_sp500_meta()
        df = build_factor_rankings(factor_key, bulk, meta)

    if df.empty:
        st.warning("Could not load S&P 500 data.")
        return

    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sector = st.selectbox("Filter by sector", sectors, key="sp500_sector_filter")

    filtered = df if sector == "All" else df[df["sector"] == sector]
    top25 = filtered.head(25).copy()
    fmt = _FACTOR_FMT.get(factor_key, lambda x: f"{x:.3f}")
    top25["value"] = top25["value"].map(fmt)
    top25 = top25[["rank", "ticker", "name", "sector", "value"]]
    top25.columns = ["Rank", "Ticker", "Company", "Sector", _FACTOR_COL.get(factor_key, "Score")]

    st.dataframe(top25, use_container_width=True, hide_index=True)


_RISK_PERIOD_MAP = {"1M": "1mo", "1Y": "1y", "Max": "max"}


def _render_risk(ticker: str) -> None:
    section_header("Risk Metrics")
    period_label = st.radio(
        "Risk period", list(_RISK_PERIOD_MAP.keys()),
        index=1, horizontal=True,
        key=f"risk_period_{ticker}",
        label_visibility="collapsed",
    )
    period = _RISK_PERIOD_MAP[period_label]
    with st.spinner("Computing risk metrics…"):
        rm = compute_risk_metrics(ticker, period=period)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Volatility (Ann.)", _pct(rm.get("volatility")))
    c2.metric("Beta vs SPY", _v(rm.get("beta"), ".2f"))
    c3.metric("Sharpe", _v(rm.get("sharpe"), ".2f"))
    c4.metric("Sortino", _v(rm.get("sortino"), ".2f"))
    c5.metric("Max Drawdown", _pct(rm.get("max_drawdown")))


def _render_price_chart(ticker: str) -> None:
    section_header("Price Chart")
    period_label = st.radio(
        "Period", list(_PERIOD_MAP.keys()),
        index=3, horizontal=True,
        key=f"an_period_{ticker}",
        label_visibility="collapsed",
    )
    period = _PERIOD_MAP[period_label]
    hist = get_price_history(ticker, period=period)
    if hist.empty:
        st.warning("Price history unavailable.")
    else:
        st.plotly_chart(plotly_price_chart(hist, ticker), use_container_width=True)


def _render_fundamentals(ticker: str) -> None:
    section_header("Fundamentals", f"https://finance.yahoo.com/quote/{ticker}/financials")
    fund = get_fundamentals(ticker)

    cols = st.columns(6)
    labels = ["P/E (TTM)", "Forward P/E", "EPS (TTM)", "Rev. Growth", "Profit Margin", "Debt/Equity"]
    values = [
        _v(fund.get("pe_ratio"), ".1f"),
        _v(fund.get("forward_pe"), ".1f"),
        _v(fund.get("eps"), ".2f", ""),
        _pct(fund.get("revenue_growth")),
        _pct(fund.get("profit_margin")),
        _v(fund.get("debt_to_equity"), ".1f"),
    ]
    for col, lbl, val in zip(cols, labels, values):
        col.metric(lbl, val)


def _render_news(ticker: str) -> None:
    section_header("Recent News")
    news = get_news(ticker, n=5)
    if news is None:
        st.warning("News temporarily unavailable — Yahoo Finance may be rate-limiting. Refresh to retry.")
        return
    if not news:
        st.info("No recent news available.")
        return

    for item in news:
        title     = item.get("title") or "—"
        publisher = item.get("publisher") or ""
        link      = item.get("link") or ""
        pub_raw   = item.get("published")

        # Format date
        date_str = ""
        if pub_raw:
            try:
                if isinstance(pub_raw, (int, float)):
                    date_str = datetime.utcfromtimestamp(pub_raw).strftime("%b %d, %Y")
                else:
                    date_str = str(pub_raw)[:10]
            except Exception:
                date_str = ""

        meta = " · ".join(filter(None, [publisher, date_str]))
        if link:
            st.markdown(f"**[{title}]({link})**  \n<small style='color:#888'>{meta}</small>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"**{title}**  \n<small style='color:#888'>{meta}</small>",
                        unsafe_allow_html=True)
        st.divider()


def _render_recommendations(ticker: str) -> None:
    section_header("Analyst Ratings", f"https://finance.yahoo.com/quote/{ticker}/analysis")
    rec = get_recommendations(ticker)
    if rec is None:
        st.warning("Ratings temporarily unavailable — Yahoo Finance may be rate-limiting. Refresh to retry.")
        return
    if not rec:
        st.info("No analyst ratings available.")
        return

    strong_buy  = rec.get("strongBuy") or 0
    buy         = rec.get("buy") or 0
    hold        = rec.get("hold") or 0
    sell        = rec.get("sell") or 0
    strong_sell = rec.get("strongSell") or 0
    target      = rec.get("targetMeanPrice")

    total = strong_buy + buy + hold + sell + strong_sell

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Strong Buy", strong_buy if total else "—")
    c2.metric("Buy", buy if total else "—")
    c3.metric("Hold", hold if total else "—")
    c4.metric("Sell", sell if total else "—")
    c5.metric("Strong Sell", strong_sell if total else "—")
    c6.metric("Mean Target", _fmt_price(target))


_INSIDER_PERIOD_MAP = {"30D": 30, "1Y": 365, "13M (Max)": None}


def _render_insider(ticker: str) -> None:
    section_header("Insider Activity", f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4")
    period_label = st.radio(
        "Insider period", list(_INSIDER_PERIOD_MAP.keys()),
        index=0, horizontal=True,
        key=f"insider_period_{ticker}",
        label_visibility="collapsed",
    )
    days = _INSIDER_PERIOD_MAP[period_label]
    summary = get_insider_summary(ticker, days=days)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buys", summary["buys"])
    c2.metric("Sells", summary["sells"])
    c3.metric("Net Shares",
              f"{summary['net_shares']:+,.0f}" if summary["net_shares"] != 0 else "—")
    c4.metric("Net Ratio",
              f"{summary['net_ratio']:+.2f}" if summary["buys"] + summary["sells"] > 0 else "—")

    txns = summary.get("transactions", [])
    if txns:
        df = pd.DataFrame(txns)
        rename = {}
        for col in df.columns:
            if "date" in col.lower():
                rename[col] = "Date"
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
        if rename:
            df = df.rename(columns=rename)
        df = df.drop(columns=[c for c in ("URL", "url") if c in df.columns], errors="ignore")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        no_data_msg = f"No insider transactions found in the last {period_label}."
        st.info(no_data_msg)


# ---------------------------------------------------------------------------
# Regime section
# ---------------------------------------------------------------------------

_ACCENT = "#00BFFF"


def _hex_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _render_regime(ticker: str) -> None:
    section_header("Market Regime (HMM)")

    today = datetime.today()
    default_end = today.strftime("%Y-%m-%d")
    default_start = (today - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    with st.expander("Regime Controls", expanded=False):
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input(
                "Start date",
                value=datetime.strptime(default_start, "%Y-%m-%d"),
                key=f"reg_start_{ticker}",
            )
        with col_e:
            end_date = st.date_input(
                "End date",
                value=datetime.strptime(default_end, "%Y-%m-%d"),
                key=f"reg_end_{ticker}",
            )

        use_auto = st.toggle("Auto BIC selection", value=True, key=f"reg_auto_{ticker}")
        n_slider = st.slider(
            "Number of regimes",
            min_value=3, max_value=7, value=3,
            key=f"reg_n_{ticker}",
            disabled=use_auto,
        )
        run_btn = st.button("Run Regime Analysis", key=f"reg_run_{ticker}", type="primary")

    n_override = None if use_auto else int(n_slider)

    if run_btn:
        try:
            with st.spinner("Fitting GARCH + HMM… this may take 20–40 seconds."):
                result = compute_regimes(ticker, str(start_date), str(end_date), n_override)
            st.session_state[f"regime_data_{ticker}"] = result
            st.session_state[f"regime_ran_{ticker}"] = True
        except ValueError as e:
            st.error(str(e))

    if not st.session_state.get(f"regime_ran_{ticker}"):
        st.info("Expand controls above and click **Run Regime Analysis** to detect market regimes.")
        return

    data = st.session_state.get(f"regime_data_{ticker}")
    if data is None:
        st.error("Regime analysis failed — not enough data or model did not converge.")
        return

    dates          = data["dates"]
    opens          = data.get("opens", [])
    highs          = data.get("highs", [])
    lows           = data.get("lows", [])
    closes         = data["closes"]
    volumes        = data.get("volumes", [])
    daily_returns  = data.get("daily_returns", [])
    realized_vols  = data.get("realized_vols", [])
    reg_labels     = data["reg_labels"]
    confidence     = data["confidence"]
    stats          = data["stats"]
    color_map      = data["color_map"]
    cur_label      = data["cur_label"]
    cur_conf       = data["cur_conf"]
    stability      = data["stability"]
    n_regimes      = data["n_regimes"]

    # ── Current regime summary row ──────────────────────────────────────────
    badge_col = color_map.get(cur_label, _ACCENT)
    text_col = "#000" if badge_col in (_ACCENT, "#FFD700") else "#fff"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div style='font-size:0.8em;color:#888'>Current Regime</div>"
        f"<div style='padding:4px 10px;background:{badge_col};color:{text_col};"
        f"border-radius:4px;font-weight:bold;display:inline-block'>{cur_label}</div>",
        unsafe_allow_html=True,
    )
    c2.metric("Confidence", f"{cur_conf:.1f}%")
    c3.metric("Stability", stability)
    c4.metric("Regimes Found", n_regimes)

    # ── Price chart with regime bands ───────────────────────────────────────
    fig = go.Figure()

    # Extract spans for both vrects and clickable markers
    spans = []
    i = 0
    while i < len(reg_labels):
        lbl = reg_labels[i]
        j = i + 1
        while j < len(reg_labels) and reg_labels[j] == lbl:
            j += 1
        end_idx = min(j, len(dates) - 1)
        spans.append({
            "label": lbl,
            "start": dates[i],
            "end":   dates[end_idx],
            "mid":   dates[(i + end_idx) // 2],
            "color": color_map.get(lbl, "#ffffff"),
        })
        fig.add_vrect(
            x0=dates[i],
            x1=dates[end_idx],
            fillcolor=_hex_rgba(color_map.get(lbl, "#ffffff"), 0.13),
            layer="below",
            line_width=0,
        )
        i = j

    max_close = max(closes) if closes else 1
    min_close = min(closes) if closes else 0
    bar_base   = max_close * 1.005   # thin bar sits just above the price line
    bar_height = max_close * 0.012   # bar thickness

    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        line=dict(color="white", width=1.5),
        name="Price",
        hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
    ))

    # Clickable thin bars — one per regime span — drawn above the price line
    bar_x, bar_widths, bar_colors, bar_cd = [], [], [], []
    for s in spans:
        s_dt = datetime.strptime(s["start"], "%Y-%m-%d")
        e_dt = datetime.strptime(s["end"],   "%Y-%m-%d")
        width_ms = max((e_dt - s_dt).total_seconds() * 1000, 86_400_000)
        bar_x.append(s["mid"])
        bar_widths.append(width_ms)
        bar_colors.append(s["color"])
        bar_cd.append([s["start"], s["end"], s["label"]])

    fig.add_trace(go.Bar(
        x=bar_x,
        y=[bar_height] * len(spans),
        base=[bar_base] * len(spans),
        width=bar_widths,
        marker_color=bar_colors,
        marker_line_width=0,
        customdata=bar_cd,
        showlegend=False,
        name="",
        opacity=0.9,
        hovertemplate=(
            "<b>%{customdata[2]}</b><br>"
            "%{customdata[0]} → %{customdata[1]}<br>"
            "<i>Click to select range</i><extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=420,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(
            showgrid=True, gridcolor="#1e1e2e", zeroline=False, title="Price ($)",
            range=[min_close * 0.97, max_close * 1.022],
        ),
        margin=dict(l=60, r=20, t=10, b=40),
        hovermode="closest",
        barmode="overlay",
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key=f"regime_chart_{ticker}",
    )

    # Clicking a regime bar seeds the date inputs — iterate to find bar customdata
    if event and event.selection and event.selection.points:
        for pt in event.selection.points:
            cd = pt.get("customdata")
            if cd and len(cd) >= 3:
                st.session_state[f"exp_start_{ticker}"] = datetime.strptime(cd[0], "%Y-%m-%d").date()
                st.session_state[f"exp_end_{ticker}"]   = datetime.strptime(cd[1], "%Y-%m-%d").date()
                break

    # ── Export panel — always visible, date range freely adjustable ──────────
    st.markdown("**Export Range** — click a regime bar above to pre-fill, or set dates manually")
    ec1, ec2, ec3 = st.columns([2, 2, 1])
    with ec1:
        exp_start = st.date_input(
            "Start date",
            value=datetime.strptime(dates[0], "%Y-%m-%d"),
            key=f"exp_start_{ticker}",
        )
    with ec2:
        exp_end = st.date_input(
            "End date",
            value=datetime.strptime(dates[-1], "%Y-%m-%d"),
            key=f"exp_end_{ticker}",
        )

    # Build export DataFrame — richer columns, filtered to selected date range
    base_len = len(dates)
    exp_df = pd.DataFrame({
        "date":               dates,
        "open":               opens  if len(opens)  == base_len else [None] * base_len,
        "high":               highs  if len(highs)  == base_len else [None] * base_len,
        "low":                lows   if len(lows)   == base_len else [None] * base_len,
        "close":              closes,
        "volume":             volumes       if len(volumes)       == base_len else [None] * base_len,
        "daily_return_pct":   daily_returns if len(daily_returns) == base_len else [None] * base_len,
        "realized_vol_pct":   realized_vols if len(realized_vols) == base_len else [None] * base_len,
        "regime":             reg_labels,
        "regime_confidence":  [round(c * 100, 1) for c in confidence],
    })
    exp_df = exp_df[
        (exp_df["date"] >= str(exp_start)) & (exp_df["date"] <= str(exp_end))
    ]

    with ec3:
        st.download_button(
            "Export CSV",
            data=exp_df.to_csv(index=False),
            file_name=f"{ticker}_regime_{exp_start}_{exp_end}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"exp_btn_{ticker}",
        )

    # ── Per-regime stats cards ──────────────────────────────────────────────
    cols = st.columns(max(1, len(stats)))
    for col, (lbl, s) in zip(cols, stats.items()):
        with col:
            st.markdown(
                f"<div style='border-left:4px solid {s['color']};padding:8px 12px;"
                f"background:#0d0d1a;border-radius:4px;margin-bottom:4px'>"
                f"<span style='color:{s['color']};font-weight:bold'>{lbl}</span><br>"
                f"<small>Return: {s['mean_return']:+.3f}%<br>"
                f"Vol: {s['mean_vol']:.3f}%<br>"
                f"Vol Ratio: {s['mean_vol_ratio']:.2f}×<br>"
                f"Time: {s['pct_time']:.1f}%</small></div>",
                unsafe_allow_html=True,
            )

    # ── Confidence timeline ─────────────────────────────────────────────────
    conf_fig = go.Figure()
    conf_fig.add_trace(go.Scatter(
        x=dates,
        y=[c * 100 for c in confidence],
        mode="lines",
        fill="tozeroy",
        fillcolor=_hex_rgba(_ACCENT, 0.25),
        line=dict(color=_ACCENT, width=1.2),
        name="Confidence %",
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
    ))
    conf_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=160,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2e", zeroline=False,
                   title="Conf %", range=[0, 105]),
        margin=dict(l=60, r=20, t=6, b=40),
    )
    st.plotly_chart(conf_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_analysis_tab():
    st.subheader("Analysis")

    # ── Ticker input ───────────────────────────────────────────────────────
    with st.form("an_form"):
        c1, c2 = st.columns([5, 1])
        with c1:
            ticker_input = st.text_input(
                "Ticker",
                placeholder="Enter a ticker symbol, e.g. AAPL",
                label_visibility="collapsed",
            )
        with c2:
            analyze = st.form_submit_button("Analyze", use_container_width=True)

    if analyze and ticker_input.strip():
        st.session_state["an_ticker"] = ticker_input.strip().upper()

    ticker = st.session_state.get("an_ticker", "")
    if not ticker:
        st.info("Enter a ticker above and click Analyze to generate a full analysis card.")
        return

    # ── Analysis card ──────────────────────────────────────────────────────
    # 1 — Header
    _render_header(ticker)
    st.markdown("")

    # 2 — Meridian signal + factor bar chart
    _render_meridian(ticker)

    # 3 — Risk metrics row
    _render_risk(ticker)

    # 4 — Price chart with period toggle
    _render_price_chart(ticker)

    # 5 — Market regime detection
    _render_regime(ticker)

    # 6 — Fundamentals table
    _render_fundamentals(ticker)

    # 7 — Recent news
    _render_news(ticker)

    # 8 — Analyst ratings
    _render_recommendations(ticker)

    # 9 — Insider activity
    _render_insider(ticker)
