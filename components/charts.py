"""Reusable Plotly chart components: price chart, allocation pie, benchmark comparison."""

import pandas as pd
import plotly.graph_objects as go

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

_PRIMARY = "#7C5CFF"
_GREEN   = "#00C853"
_RED     = "#FF1744"
_ACCENT  = "#40C4FF"


def plotly_price_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + volume chart for a price-history DataFrame."""
    if hist.empty or "Close" not in hist.columns:
        return go.Figure()

    fig = go.Figure()

    has_ohlc = all(c in hist.columns for c in ("Open", "High", "Low", "Close"))
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name=ticker,
                increasing_line_color=_GREEN,
                decreasing_line_color=_RED,
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist["Close"],
                mode="lines", name=ticker,
                line=dict(color=_PRIMARY, width=1.8),
            )
        )

    # Volume bars on a secondary y-axis
    if "Volume" in hist.columns:
        colors = [
            _GREEN if (hist["Close"].iloc[i] >= hist["Open"].iloc[i] if has_ohlc else True)
            else _RED
            for i in range(len(hist))
        ]
        fig.add_trace(
            go.Bar(
                x=hist.index, y=hist["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.35,
                yaxis="y2",
                showlegend=False,
            )
        )
        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                showticklabels=False,
                range=[0, hist["Volume"].max() * 5],
            )
        )

    fig.update_layout(
        **_DARK,
        height=380,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    return fig


def factor_bar_chart(factors: dict) -> go.Figure:
    """Horizontal bar chart of Meridian factor z-scores."""
    labels, zscores, colors = [], [], []
    for f in factors.values():
        if not f["available"]:
            labels.append(f["label"])
            zscores.append(0)
            colors.append("#555")
        else:
            labels.append(f["label"])
            z = f["z_score"]
            zscores.append(z)
            colors.append(_GREEN if z >= 0 else _RED)

    fig = go.Figure(
        go.Bar(
            x=zscores,
            y=labels,
            orientation="h",
            marker_color=colors,
            customdata=list(factors.keys()),
            text=[f"{z:+.2f}" if f["available"] else "—" for z, f in zip(zscores, factors.values())],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.update_layout(
        **_DARK,
        height=300,
        xaxis=dict(range=[-3.5, 3.5], zeroline=False, title="z-score"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=60, t=10, b=30),
    )
    return fig


def allocation_pie(labels: list[str], values: list[float], title: str = "") -> go.Figure:
    """Donut chart for portfolio allocation."""
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        **_DARK,
        height=350,
        showlegend=True,
        legend=dict(orientation="v", x=1.0),
        margin=dict(l=0, r=80, t=30, b=0),
        title=dict(text=title, font=dict(size=13)),
    )
    return fig


def benchmark_compare(vs_df: pd.DataFrame, benchmark: str = "SPY") -> go.Figure:
    """Line chart comparing portfolio cumulative return vs a benchmark, rebased to 100."""
    fig = go.Figure()

    if "Portfolio" in vs_df.columns:
        fig.add_trace(
            go.Scatter(
                x=vs_df.index, y=vs_df["Portfolio"],
                mode="lines", name="Portfolio",
                line=dict(color=_PRIMARY, width=2),
            )
        )
    if benchmark in vs_df.columns:
        fig.add_trace(
            go.Scatter(
                x=vs_df.index, y=vs_df[benchmark],
                mode="lines", name=benchmark,
                line=dict(color=_ACCENT, width=1.5, dash="dot"),
            )
        )

    fig.add_hline(y=100, line_color="#555", line_width=1)
    fig.update_layout(
        **_DARK,
        height=350,
        xaxis=dict(title=""),
        yaxis=dict(title="Rebased (100 = start)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    return fig


def rolling_volatility_chart(
    vol_df: pd.DataFrame, window: int, benchmark: str
) -> go.Figure:
    """Line chart of rolling annualised volatility for multiple series.

    The benchmark is rendered as a dashed line; each other series gets a
    distinct colour from a fixed palette.
    """
    PALETTE = [_PRIMARY, "#00C853", "#FF1744", "#FFD600", "#FF6D00", "#E040FB"]

    fig = go.Figure()
    color_idx = 0

    for col in vol_df.columns:
        is_bench = col == benchmark
        color = _ACCENT if is_bench else PALETTE[color_idx % len(PALETTE)]
        if not is_bench:
            color_idx += 1

        fig.add_trace(
            go.Scatter(
                x=vol_df.index,
                y=vol_df[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=1.8, dash="dot" if is_bench else "solid"),
                hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>",
            )
        )

    fig.update_layout(
        **_DARK,
        height=350,
        xaxis=dict(title=""),
        yaxis=dict(title=f"{window}-day Rolling Volatility (Ann.)", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    return fig


def sector_bar(sectors: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of sector weights."""
    sorted_items = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
    labels = [s for s, _ in sorted_items]
    vals   = [v for _, v in sorted_items]

    fig = go.Figure(
        go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=_PRIMARY,
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_DARK,
        height=max(200, 40 * len(labels)),
        xaxis=dict(tickformat=".0%", range=[0, max(vals) * 1.3]),
        margin=dict(l=0, r=60, t=10, b=0),
    )
    return fig
