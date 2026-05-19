"""Watchlist tab: live search, add/remove tickers, inline portfolio save."""

import streamlit as st
from datetime import date

from data.storage import (
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    add_portfolio_lot,
    remove_portfolio_lot,
    remove_from_portfolio,
    is_in_portfolio,
    get_portfolio,
    get_lots_for_ticker,
)
from data.market_data import get_company_name, get_current_price, get_ticker_info


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_price(v: float | None) -> str:
    return f"${v:,.2f}" if v is not None else "—"


def _colored_pct(v: float | None) -> str:
    if v is None:
        return "—"
    color = "green" if v >= 0 else "red"
    return f":{color}[{v:+.2f}%]"


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_watchlist_tab():
    st.subheader("Watchlist")

    # ── Add ticker form (Enter key also submits) ───────────────────────────
    with st.form("wl_add_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            new_ticker = st.text_input(
                "ticker",
                placeholder="Add a ticker symbol, e.g. AAPL",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("＋ Add", use_container_width=True)

    if submitted:
        t = new_ticker.strip().upper()
        if not t:
            st.warning("Please enter a ticker symbol.")
        else:
            info = get_ticker_info(t)
            if not (info.get("shortName") or info.get("longName") or info.get("symbol")):
                st.warning(f"Could not find market data for **{t}** — adding anyway. Check the symbol if prices show —.")
            added = add_to_watchlist(t)
            if not added:
                st.warning(f"**{t}** is already in your watchlist.")
            else:
                st.success(f"Added **{t}** to watchlist.")
                st.rerun()

    # ── Live search filter ─────────────────────────────────────────────────
    st.text_input(
        "search",
        placeholder="🔍  Filter by ticker or company name…",
        key="wl_search",
        label_visibility="collapsed",
    )
    search_q = st.session_state.get("wl_search", "").strip().lower()

    # ── Load and enrich ────────────────────────────────────────────────────
    wl = get_watchlist()
    if not wl:
        st.info("Your watchlist is empty. Add tickers using the input above.")
        return

    with st.spinner("Fetching prices…"):
        rows: list[dict] = []
        for item in wl:
            tk = item["ticker"]
            price, chg = get_current_price(tk)
            rows.append(dict(ticker=tk, name=get_company_name(tk), price=price, day_pct=chg))

    # ── Apply filter ───────────────────────────────────────────────────────
    if search_q:
        rows = [r for r in rows if search_q in r["ticker"].lower() or search_q in r["name"].lower()]

    if not rows:
        st.info("No items match your filter.")
        return

    # Pre-load portfolio for input pre-fill
    port_map: dict[str, dict] = {p["ticker"]: p for p in get_portfolio()}

    # ── Table header ───────────────────────────────────────────────────────
    COLS = [0.35, 1.1, 3.6, 1.5, 1.5, 0.45]
    hdr = st.columns(COLS)
    for col, lbl in zip(hdr, ["", "Ticker", "Company", "Last Price", "Day %", ""]):
        col.markdown(f"<small><b>{lbl}</b></small>", unsafe_allow_html=True)
    st.divider()

    # ── Render each row ────────────────────────────────────────────────────
    for row in rows:
        ticker   = row["ticker"]
        ck_key   = f"wl_ck_{ticker}"
        prev_key = f"wl_prev_{ticker}"

        prev_checked = st.session_state.get(prev_key, False)

        # Restore checkbox if the user cancelled an "uncheck while in portfolio" dialog
        restore_key = f"wl_restore_{ticker}"
        if st.session_state.pop(restore_key, False):
            st.session_state[ck_key] = True

        # Main row
        c = st.columns(COLS)
        with c[0]:
            checked = st.checkbox("", key=ck_key, label_visibility="collapsed")
        with c[1]:
            st.markdown(f"**{ticker}**")
        with c[2]:
            st.write(row["name"])
        with c[3]:
            st.write(_fmt_price(row["price"]))
        with c[4]:
            st.markdown(_colored_pct(row["day_pct"]))
        with c[5]:
            if st.button("✕", key=f"wl_rm_{ticker}", help=f"Remove {ticker} from watchlist"):
                st.session_state[f"wl_crm_{ticker}"] = True
                st.rerun()

        # Detect uncheck-while-in-portfolio transition
        if prev_checked and not checked and is_in_portfolio(ticker):
            st.session_state[f"wl_cup_{ticker}"] = True

        st.session_state[prev_key] = checked  # save for next render

        # ── Confirm: remove from watchlist ─────────────────────────────────
        if st.session_state.get(f"wl_crm_{ticker}"):
            extra = " This will also remove it from your portfolio." if is_in_portfolio(ticker) else ""
            st.warning(f"Remove **{ticker}** from your watchlist?{extra}")
            y_col, n_col, _ = st.columns([1.3, 1.3, 5])
            with y_col:
                if st.button("Yes, remove", key=f"wl_crm_y_{ticker}", type="primary"):
                    remove_from_watchlist(ticker)
                    for k in (ck_key, prev_key, f"wl_crm_{ticker}"):
                        st.session_state.pop(k, None)
                    st.rerun()
            with n_col:
                if st.button("Cancel", key=f"wl_crm_n_{ticker}"):
                    st.session_state.pop(f"wl_crm_{ticker}", None)
                    st.rerun()

        # ── Confirm: remove from portfolio on uncheck ──────────────────────
        if st.session_state.get(f"wl_cup_{ticker}"):
            st.warning(f"Remove **{ticker}** from your portfolio?")
            y_col, n_col, _ = st.columns([1.6, 1.8, 4])
            with y_col:
                if st.button("Yes, remove", key=f"wl_cup_y_{ticker}", type="primary"):
                    remove_from_portfolio(ticker)
                    st.session_state.pop(f"wl_cup_{ticker}", None)
                    st.rerun()
            with n_col:
                if st.button("Keep in portfolio", key=f"wl_cup_n_{ticker}"):
                    st.session_state[restore_key] = True
                    st.session_state[prev_key] = True
                    st.session_state.pop(f"wl_cup_{ticker}", None)
                    st.rerun()

        # ── Inline portfolio inputs (revealed when checkbox is checked) ─────
        if checked:
            default_price = float(row["price"]) if row["price"] is not None else 0.01

            with st.container():
                ic = st.columns([0.35, 1.5, 2.0, 2.0, 2.5, 0.5])
                with ic[1]:
                    sh = st.number_input(
                        "Shares",
                        min_value=0.0001,
                        value=1.0,
                        format="%.4f",
                        key=f"wl_sh_{ticker}",
                    )
                with ic[2]:
                    pb = st.number_input(
                        "Price bought ($)",
                        min_value=0.01,
                        value=default_price,
                        format="%.2f",
                        key=f"wl_pb_{ticker}",
                    )
                with ic[3]:
                    db = st.date_input(
                        "Date bought",
                        value=date.today(),
                        max_value=date.today(),
                        key=f"wl_db_{ticker}",
                    )
                with ic[4]:
                    if st.button("＋ Add Lot", key=f"wl_sv_{ticker}", use_container_width=True):
                        add_portfolio_lot(ticker, sh, pb, db)
                        st.success(
                            f"Added **{sh:g} × {ticker}** at **{_fmt_price(pb)}** on {db}."
                        )

            # ── Existing lots for this ticker ─────────────────────────────
            lots = get_lots_for_ticker(ticker)
            if lots:
                st.markdown(
                    f"<small style='color:#aaa'>Existing lots for {ticker}</small>",
                    unsafe_allow_html=True,
                )
                for lot in lots:
                    lc = st.columns([0.35, 1.5, 2.0, 2.0, 2.5, 0.5])
                    lc[1].caption(f"{lot['shares']:g} sh")
                    lc[2].caption(f"@ {_fmt_price(lot['price_bought'])}")
                    lc[3].caption(str(lot["date_bought"]))
                    with lc[5]:
                        if st.button("✕", key=f"wl_rl_{lot['id']}", help="Remove this lot"):
                            remove_portfolio_lot(lot["id"])
                            st.rerun()

        st.divider()
