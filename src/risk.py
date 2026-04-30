"""리스크 관리 — ATR 기반 손절가와 포지션 사이징."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskPlan:
    atr: float
    stop_loss: float
    stop_loss_pct: float
    suggested_qty: int
    risk_amount: float


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR. 14일 기본."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def plan_risk(
    df: pd.DataFrame,
    entry_price: float,
    capital: float,
    risk_pct: float = 2.0,
    atr_mult: float = 2.0,
    period: int = 14,
) -> RiskPlan:
    """ATR 기반 손절가 + 포지션 사이즈.

    - 손절가 = 진입가 - atr_mult * ATR
    - 종목당 리스크 = 자본 * risk_pct/100
    - 수량 = 종목당 리스크 / (진입가 - 손절가)
    """
    a = atr(df, period=period)
    stop = entry_price - atr_mult * a
    risk_amount = capital * (risk_pct / 100.0)
    per_share_risk = max(entry_price - stop, 1e-9)
    qty = int(risk_amount // per_share_risk) if per_share_risk > 0 else 0
    return RiskPlan(
        atr=a,
        stop_loss=stop,
        stop_loss_pct=(stop / entry_price - 1) * 100,
        suggested_qty=qty,
        risk_amount=risk_amount,
    )
