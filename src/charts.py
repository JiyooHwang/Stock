"""Plotly 차트 — 캔들 + 엘리엇 파동 라벨링."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .elliott_wave import Pivot, WaveAnalysis


def candle_with_waves(
    df: pd.DataFrame,
    pivots: list[Pivot] | None = None,
    analysis: WaveAnalysis | None = None,
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
            increasing_line_color="#d24f45",
            decreasing_line_color="#1f77b4",
        )
    )

    if pivots:
        fig.add_trace(
            go.Scatter(
                x=[p.date for p in pivots],
                y=[p.price for p in pivots],
                mode="lines+markers",
                line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dot"),
                marker=dict(size=6, color="#888"),
                name="ZigZag",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
            )
        )

    if analysis and analysis.waves:
        for label, (a, b) in analysis.waves.items():
            fig.add_trace(
                go.Scatter(
                    x=[a.date, b.date],
                    y=[a.price, b.price],
                    mode="lines+text",
                    line=dict(color="#ff6f00", width=2),
                    text=[None, label],
                    textposition="top center",
                    textfont=dict(size=14, color="#ff6f00"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if analysis.targets:
            last = analysis.pivots[-1]
            for label, price in analysis.targets.items():
                fig.add_hline(
                    y=price,
                    line=dict(color="rgba(50,150,50,0.5)", width=1, dash="dash"),
                    annotation_text=f"{label}: {price:,.0f}",
                    annotation_position="right",
                )
                fig.add_trace(
                    go.Scatter(
                        x=[last.date],
                        y=[price],
                        mode="markers",
                        marker=dict(symbol="triangle-right", size=10, color="green"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=120, t=50, b=20),
        template="plotly_white",
    )
    return fig
