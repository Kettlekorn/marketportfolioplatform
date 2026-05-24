"""Reusable UI components: score_badge(), section_header()."""

import streamlit as st


def score_badge(signal: str, composite: float, n_factors: int, n_total: int = 8) -> None:
    """Render a large Meridian signal badge with composite score."""
    colors = {"LONG": "#00C853", "SHORT": "#FF1744", "NEUTRAL": "#FFD600"}
    bg = colors.get(signal, "#888")
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:{bg};
            color:#0E1117;
            font-weight:700;
            font-size:1.1rem;
            padding:0.35rem 1rem;
            border-radius:6px;
            letter-spacing:0.08em;
        ">{signal}</div>
        <span style="color:#aaa; font-size:0.85rem; margin-left:0.7rem;">
            composite&nbsp;{composite:+.2f} &nbsp;·&nbsp; {n_factors}/{n_total} factors
        </span>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, source_url: str | None = None) -> None:
    """Render a subtle section divider with a title and optional source link icon."""
    link_html = (
        f"<a href='{source_url}' target='_blank' style='"
        f"color:#7C5CFF; font-size:0.7em; opacity:0.45; margin-left:5px; "
        f"vertical-align:super; text-decoration:none;' title='View source'>&#x2197;</a>"
        if source_url else ""
    )
    st.markdown(
        f"<p style='color:#7C5CFF; font-weight:600; margin-bottom:0.2rem; "
        f"font-size:0.9rem; text-transform:uppercase; letter-spacing:0.07em;'>"
        f"{title}{link_html}</p>",
        unsafe_allow_html=True,
    )
    st.divider()
