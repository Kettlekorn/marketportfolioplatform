"""Portfolio tab: P/L KPIs, per-position table, allocation charts, SPY comparison, risk."""

import pandas as pd
import streamlit as st

from analytics.portfolio import (
    aggregate_risk_metrics,
    aggregate_totals,
    compute_positions,
    individual_rolling_volatilities,
    portfolio_rolling_volatility,
    portfolio_vs_benchmark,
)
from components.cards import section_header
from components.charts import allocation_pie, benchmark_compare, rolling_volatility_chart, sector_bar
from config import BENCHMARK_TICKER
from data.storage import get_portfolio


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_price(v: float | None) -> str:
    return f"${v:,.2f}" if v is not None else "—"


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.1%}" if v is not None else "—"


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_portfolio_tab():
    st.subheader("Portfolio")

    portfolio_rows = get_portfolio()
    if not portfolio_rows:
        st.info(
            "No positions yet. Add tickers on the **Watchlist** tab, check the row, "
            "fill in a price and date, then click **Save to Portfolio**."
        )
        return

    # ── Compute all analytics ──────────────────────────────────────────────
    with st.spinner("Loading positions…"):
        positions = compute_positions(portfolio_rows)

    totals = aggregate_totals(positions)

    # ── KPI row ────────────────────────────────────────────────────────────
    section_header("Summary")

    k1, k2, k3, k4 = st.columns(4)
    pl_color = "normal" if totals["pl_dollar"] >= 0 else "inverse"
    k1.metric("Cost Basis",    _fmt_price(totals["cost_basis"]))
    k2.metric("Current Value", _fmt_price(totals["current_value"]))
    k3.metric(
        "Total P/L",
        _fmt_price(totals["pl_dollar"]),
        delta=_fmt_pct(totals["pl_pct"]),
        delta_color=pl_color,
    )
    k4.metric("Return", _fmt_pct(totals["pl_pct"]))

    # ── Positions table ────────────────────────────────────────────────────
    section_header("Positions")

    table_rows = [
        {
            "Ticker":   pos["ticker"],
            "Shares":   pos["shares"],
            "Avg Cost": pos["avg_cost"],
            "Price":    pos["current_price"],
            "P/L $":    pos["pl_dollar"],
            "P/L %":    pos["pl_pct"] * 100,
            "Weight %": pos["weight_pct"] * 100,
            "Sector":   pos["sector"],
        }
        for pos in positions
    ]
    df = pd.DataFrame(table_rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Shares":   st.column_config.NumberColumn("Shares",       format="%.4g"),
            "Avg Cost": st.column_config.NumberColumn("Avg Cost",     format="$%.2f"),
            "Price":    st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "P/L $":    st.column_config.NumberColumn("P/L $",        format="%.2f"),
            "P/L %":    st.column_config.NumberColumn("P/L %",        format="%.2f%%"),
            "Weight %": st.column_config.NumberColumn("Weight",       format="%.1f%%"),
        },
    )

    # ── Per-ticker lot detail ──────────────────────────────────────────────
    with st.expander("Lot detail (all individual purchases)"):
        for pos in positions:
            st.markdown(f"**{pos['ticker']}** — {pos['shares']:g} shares total")
            lot_rows = [
                {
                    "Date":          str(lot["date_bought"]),
                    "Shares":        lot.get("shares", 1.0),
                    "Price Paid ($)": lot["price_bought"],
                    "Cost Basis ($)": lot["price_bought"] * lot.get("shares", 1.0),
                }
                for lot in pos.get("lots", [])
            ]
            if lot_rows:
                st.dataframe(
                    pd.DataFrame(lot_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Shares":         st.column_config.NumberColumn(format="%.4g"),
                        "Price Paid ($)": st.column_config.NumberColumn(format="$%.2f"),
                        "Cost Basis ($)": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )

    # ── Allocation pie + sector breakdown ──────────────────────────────────
    section_header("Allocation")

    pie_col, bar_col = st.columns(2)

    with pie_col:
        tickers = [p["ticker"]    for p in positions]
        weights = [p["weight_pct"] for p in positions]
        st.plotly_chart(
            allocation_pie(tickers, weights, title="By Ticker"),
            use_container_width=True,
        )

    with bar_col:
        sector_map: dict[str, float] = {}
        for pos in positions:
            s = pos["sector"]
            sector_map[s] = sector_map.get(s, 0.0) + pos["weight_pct"]
        st.plotly_chart(sector_bar(sector_map), use_container_width=True)

    # ── Portfolio vs SPY ───────────────────────────────────────────────────
    section_header(f"Performance vs {BENCHMARK_TICKER}")

    with st.spinner("Building return series…"):
        vs_spy = portfolio_vs_benchmark(portfolio_rows)

    if vs_spy.empty:
        st.warning("Not enough price history to build a comparison chart.")
    else:
        # Show period return summary above the chart
        if "Portfolio" in vs_spy.columns:
            port_ret = (vs_spy["Portfolio"].iloc[-1] / 100) - 1
            bench_col = BENCHMARK_TICKER if BENCHMARK_TICKER in vs_spy.columns else None
            bench_ret = ((vs_spy[bench_col].iloc[-1] / 100) - 1) if bench_col else None

            s1, s2, _ = st.columns([1, 1, 4])
            s1.metric("Portfolio (since earliest buy)", _fmt_pct(port_ret))
            if bench_ret is not None:
                s2.metric(f"{BENCHMARK_TICKER} (same period)", _fmt_pct(bench_ret))

        st.plotly_chart(
            benchmark_compare(vs_spy, benchmark=BENCHMARK_TICKER),
            use_container_width=True,
        )

    # ── Rolling volatility ─────────────────────────────────────────────────
    section_header("Rolling Volatility")

    vol_left, vol_right = st.columns([3, 2])
    with vol_left:
        combined = st.checkbox(
            "Combined portfolio volatility",
            value=True,
            key="pt_combined_vol",
            help="Checked → single portfolio line vs VOO. "
                 "Unchecked → each ticker gets its own volatility curve.",
        )
    with vol_right:
        window_label = st.radio(
            "Window",
            ["21-day (1M)", "63-day (3M)"],
            horizontal=True,
            key="pt_vol_window",
            label_visibility="collapsed",
        )
    vol_window = 21 if window_label.startswith("21") else 63

    with st.spinner("Computing rolling volatility…"):
        vol_df = (
            portfolio_rolling_volatility(portfolio_rows, window=vol_window)
            if combined
            else individual_rolling_volatilities(portfolio_rows, window=vol_window)
        )

    if vol_df.empty:
        st.warning(
            f"Not enough price history to plot {vol_window}-day rolling volatility "
            f"(need at least {vol_window + 1} trading days)."
        )
    else:
        st.plotly_chart(
            rolling_volatility_chart(vol_df, window=vol_window, benchmark=BENCHMARK_TICKER),
            use_container_width=True,
        )

    # ── Aggregated risk ────────────────────────────────────────────────────
    section_header("Portfolio Risk")

    with st.spinner("Computing risk metrics…"):
        risk = aggregate_risk_metrics(portfolio_rows)

    beta = risk.get("portfolio_beta")
    vol  = risk.get("portfolio_volatility")
    shr  = risk.get("portfolio_sharpe")

    r1, r2, r3 = st.columns(3)
    r1.metric("Weighted Beta",        f"{beta:.2f}" if beta is not None else "—")
    r2.metric("Portfolio Volatility", f"{vol:.1%}"  if vol  is not None else "—")
    r3.metric("Portfolio Sharpe",     f"{shr:.2f}"  if shr  is not None else "—")
