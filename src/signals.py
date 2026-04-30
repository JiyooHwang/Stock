"""종목 점수판 — 추세 / 모멘텀 / 변동성 / 밸류에이션 점수와 종합 신호.

각 카테고리는 0~100 점수로 환산하고, 가중평균으로 종합 점수를 낸다.

- 추세(Trend): 종가의 200·50일 이평선 대비 위치. 학계 검증 강함.
- 모멘텀(Momentum): 12개월 수익률(직전 1개월 제외 변형 가능). Jegadeesh-Titman.
- 변동성(Volatility): 연환산 변동성. 낮을수록 점수 ↑ (역지표).
- 밸류에이션(Valuation): PER/PBR. 낮을수록 점수 ↑ (역지표).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pykrx import stock


@dataclass
class Score:
    trend: float
    momentum: float
    volatility: float
    valuation: float
    composite: float
    signal: str  # 매수 / 관망 / 매도
    detail: dict[str, float]


def explain_score(s: "Score") -> dict[str, str]:
    """각 점수가 왜 그렇게 나왔는지 일반인용 한글 설명."""
    d = s.detail
    out: dict[str, str] = {}

    # 추세
    price = d.get("trend.price")
    ma200 = d.get("trend.MA200")
    ma50 = d.get("trend.MA50")
    gap = d.get("trend.gap%")
    if price is not None and ma200 is not None:
        above200 = price > ma200
        golden = ma50 is not None and ma50 > ma200
        parts = []
        parts.append("📈 200일선 위에 있음 (상승 추세)" if above200 else "📉 200일선 아래 (하락 추세)")
        if ma50 is not None:
            parts.append("골든크로스 (50일선이 200일선 위)" if golden else "데드크로스 (50일선이 200일선 아래)")
        if gap is not None:
            parts.append(f"200일선 대비 {gap:+.1f}% 이격")
        out["추세"] = " · ".join(parts)
    else:
        out["추세"] = "데이터 부족"

    # 모멘텀
    r12 = d.get("mom.12m%")
    r6 = d.get("mom.6m%")
    r3 = d.get("mom.3m%")
    if r12 is not None:
        if r12 > 30:
            tag = "🚀 매우 강한 상승"
        elif r12 > 0:
            tag = "↗️ 완만한 상승"
        elif r12 > -20:
            tag = "↘️ 약한 하락"
        else:
            tag = "⛔ 큰 하락"
        out["모멘텀"] = f"{tag} · 12개월 {r12:+.1f}% / 6개월 {r6:+.1f}% / 3개월 {r3:+.1f}%"
    else:
        out["모멘텀"] = "데이터 부족"

    # 변동성
    vol = d.get("vol.ann_vol%")
    if vol is not None:
        if vol < 25:
            tag = "🟢 안정적"
        elif vol < 40:
            tag = "🟡 보통"
        else:
            tag = "🔴 변동성 큼"
        out["변동성"] = f"{tag} · 연 변동성 {vol:.1f}% (낮을수록 안전)"
    else:
        out["변동성"] = "데이터 부족"

    # 밸류에이션
    per = d.get("val.PER")
    pbr = d.get("val.PBR")
    div = d.get("val.DIV%")
    if per is not None:
        if per <= 0:
            tag = "⚠️ 적자(PER 음수) — 평가 보류"
        elif per < 10:
            tag = "💰 저평가 영역"
        elif per < 20:
            tag = "🟡 적정 평가"
        else:
            tag = "💸 고평가"
        parts = [tag, f"PER {per:.1f}"]
        if pbr:
            parts.append(f"PBR {pbr:.2f}")
        if div:
            parts.append(f"배당수익률 {div:.2f}%")
        out["밸류에이션"] = " · ".join(parts)
    else:
        out["밸류에이션"] = "재무 데이터 없음 (중립 처리)"

    # 종합
    if s.composite >= 70:
        out["종합"] = "🟢 매수 — 추세·모멘텀이 양호하고 큰 위험 신호 없음"
    elif s.composite >= 40:
        out["종합"] = "🟡 관망 — 일부 지표는 좋지만 다른 지표가 발목을 잡음"
    else:
        out["종합"] = "🔴 매도 — 추세 약화 또는 고평가/고변동성 신호"
    return out


WEIGHTS = {"trend": 0.35, "momentum": 0.30, "volatility": 0.15, "valuation": 0.20}


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def trend_score(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    if len(df) < 220:
        return 50.0, {}
    close = df["Close"]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    last = float(close.iloc[-1])
    score = 50.0
    score += 25 if last > ma200 else -25
    score += 15 if ma50 > ma200 else -15
    # 200MA 대비 이격도(+10% 이내면 가산, -10% 이하면 감점)
    gap = (last - ma200) / ma200
    score += _clip(gap * 100, -10, 10)
    return _clip(score), {"price": last, "MA50": float(ma50), "MA200": float(ma200), "gap%": gap * 100}


def momentum_score(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    close = df["Close"]
    if len(close) < 252:
        return 50.0, {}
    ret_12m = float(close.iloc[-1] / close.iloc[-252] - 1)
    ret_6m = float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else 0.0
    ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0.0
    # 12m 모멘텀을 ±50% 범위로 0~100 정규화 (50%면 만점)
    base = 50 + (ret_12m / 0.50) * 50
    base += (ret_6m / 0.30) * 10
    base += (ret_3m / 0.20) * 5
    return _clip(base), {"12m%": ret_12m * 100, "6m%": ret_6m * 100, "3m%": ret_3m * 100}


def volatility_score(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    close = df["Close"]
    if len(close) < 60:
        return 50.0, {}
    daily = close.pct_change().dropna()
    ann_vol = float(daily.std() * np.sqrt(252))
    # 연 변동성 20% 이하면 만점, 60% 이상이면 0점 선형 보간
    score = 100 - ((ann_vol - 0.20) / 0.40) * 100
    return _clip(score), {"ann_vol%": ann_vol * 100}


def valuation_score(ticker: str) -> tuple[float, dict[str, float]]:
    """pykrx fundamental: PER/PBR/DIV. 적자/결측은 중립."""
    end = datetime.now()
    start = end - timedelta(days=10)
    try:
        df = stock.get_market_fundamental(
            start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker
        )
    except Exception:
        return 50.0, {}
    if df is None or df.empty:
        return 50.0, {}
    last = df.iloc[-1]
    per = float(last.get("PER", 0) or 0)
    pbr = float(last.get("PBR", 0) or 0)
    div = float(last.get("DIV", 0) or 0)
    score = 50.0
    if per > 0:
        # PER 8 이하 만점, 30 이상 0점
        score += _clip(50 - (per - 8) * (50 / 22), -50, 50) * 0.5
    if pbr > 0:
        # PBR 0.8 이하 만점, 3 이상 0점
        score += _clip(50 - (pbr - 0.8) * (50 / 2.2), -50, 50) * 0.3
    # 배당수익률 가산
    score += _clip(div * 5, 0, 20) * 0.2
    return _clip(score), {"PER": per, "PBR": pbr, "DIV%": div}


def score_ticker(df: pd.DataFrame, ticker: str) -> Score:
    t, t_d = trend_score(df)
    m, m_d = momentum_score(df)
    v, v_d = volatility_score(df)
    val, val_d = valuation_score(ticker)
    composite = (
        WEIGHTS["trend"] * t
        + WEIGHTS["momentum"] * m
        + WEIGHTS["volatility"] * v
        + WEIGHTS["valuation"] * val
    )
    if composite >= 70:
        signal = "매수"
    elif composite >= 40:
        signal = "관망"
    else:
        signal = "매도"
    detail: dict[str, float] = {}
    detail.update({f"trend.{k}": v for k, v in t_d.items()})
    detail.update({f"mom.{k}": v for k, v in m_d.items()})
    detail.update({f"vol.{k}": v for k, v in v_d.items()})
    detail.update({f"val.{k}": v for k, v in val_d.items()})
    return Score(
        trend=t, momentum=m, volatility=v, valuation=val,
        composite=composite, signal=signal, detail=detail,
    )
