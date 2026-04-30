"""엘리엇 파동 검출 및 예측.

전체 흐름:
1) ZigZag 알고리즘으로 가격 시계열의 주요 스윙(고/저점)을 추출.
2) 마지막 9개의 스윙으로 5-임펄스 + ABC 조정 패턴을 후보로 만든다.
3) 엘리엇 3대 규칙으로 검증하고 피보나치 비율로 적합도(score)를 매긴다.
4) 가장 최근의 진행 중인 파동을 추정하고, 다음 파동 목표가를 피보나치 확장으로 계산한다.

이 구현은 결정론적 휴리스틱이며 100%의 정답을 보장하지 않는다 — 참고용 신호 도구.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Pivot:
    idx: int
    date: pd.Timestamp
    price: float
    kind: str  # "H" or "L"


@dataclass
class WaveAnalysis:
    pivots: list[Pivot]
    direction: str  # "up" (상승 5파) 또는 "down"
    waves: dict[str, tuple[Pivot, Pivot]] = field(default_factory=dict)
    rule_violations: list[str] = field(default_factory=list)
    fib_scores: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    next_wave: str | None = None
    targets: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def zigzag(df: pd.DataFrame, pct: float = 0.05) -> list[Pivot]:
    """퍼센트 기반 ZigZag. High/Low 컬럼을 사용한다.

    pct: 추세 반전으로 인정할 최소 변동률 (기본 5%).
    """
    if df.empty:
        return []

    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    dates = df.index

    pivots: list[Pivot] = []
    # 첫 점은 시가의 평균으로 잡는 대신 첫 캔들 종가 인근
    direction = 0  # 0=미정, 1=상승추적, -1=하락추적
    last_pivot_idx = 0
    last_pivot_price = float(df["Close"].iloc[0])
    extreme_idx = 0
    extreme_price = last_pivot_price

    for i in range(1, len(df)):
        h, l = highs[i], lows[i]
        if direction >= 0:
            if h > extreme_price:
                extreme_price = h
                extreme_idx = i
            if l < extreme_price * (1 - pct):
                # 반전: 직전 극값을 H 피벗으로 확정
                pivots.append(
                    Pivot(extreme_idx, dates[extreme_idx], float(extreme_price), "H")
                )
                last_pivot_idx, last_pivot_price = extreme_idx, extreme_price
                direction = -1
                extreme_price, extreme_idx = l, i
                continue
        if direction <= 0:
            if l < extreme_price:
                extreme_price = l
                extreme_idx = i
            if h > extreme_price * (1 + pct):
                pivots.append(
                    Pivot(extreme_idx, dates[extreme_idx], float(extreme_price), "L")
                )
                last_pivot_idx, last_pivot_price = extreme_idx, extreme_price
                direction = 1
                extreme_price, extreme_idx = h, i

    # 마지막 진행 중인 극값도 임시 피벗으로 추가
    kind = "H" if direction >= 0 else "L"
    if not pivots or pivots[-1].idx != extreme_idx:
        pivots.append(Pivot(extreme_idx, dates[extreme_idx], float(extreme_price), kind))

    # H/L 교차 정리
    cleaned: list[Pivot] = []
    for p in pivots:
        if cleaned and cleaned[-1].kind == p.kind:
            # 같은 종류가 연달아 나오면 더 극단값을 채택
            if (p.kind == "H" and p.price > cleaned[-1].price) or (
                p.kind == "L" and p.price < cleaned[-1].price
            ):
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned


def _fib_score(actual: float, ideal: float, tol: float = 0.15) -> float:
    """실제값이 이상적 비율에 얼마나 가까운지 0~1 점수."""
    if ideal == 0:
        return 0.0
    err = abs(actual - ideal) / ideal
    return max(0.0, 1.0 - err / tol)


def analyze_impulse(pivots: list[Pivot], direction: str = "up") -> WaveAnalysis | None:
    """마지막 9개 피벗을 5파(임펄스) + ABC(조정)으로 라벨링 시도.

    상승(up)이면 시작점은 L 이어야 한다: L H L H L H L H L
        wave1: P0(L)->P1(H), wave2: P1->P2(L), wave3: P2->P3(H), ...
        wave A: P5->P6(L), wave B: P6->P7(H), wave C: P7->P8(L)
    """
    if len(pivots) < 6:  # 최소 1~5파만이라도
        return None

    # 마지막 9개(없으면 가능한 만큼)
    seq = pivots[-9:] if len(pivots) >= 9 else pivots[-len(pivots) :]
    expected_first_kind = "L" if direction == "up" else "H"
    # 시작 정렬: 첫 피벗이 기대 종류가 아니면 한 칸 밀기
    if seq[0].kind != expected_first_kind:
        if len(pivots) > len(seq):
            start = len(pivots) - len(seq) - 1
            seq = pivots[start : start + 9] if len(pivots) - start >= 1 else seq
        else:
            seq = seq[1:]
    if len(seq) < 6 or seq[0].kind != expected_first_kind:
        return None

    sign = 1 if direction == "up" else -1

    def length(a: Pivot, b: Pivot) -> float:
        return (b.price - a.price) * sign

    analysis = WaveAnalysis(pivots=seq, direction=direction)

    # 임펄스 1~5
    if len(seq) >= 6:
        w1 = (seq[0], seq[1])
        w2 = (seq[1], seq[2])
        w3 = (seq[2], seq[3])
        w4 = (seq[3], seq[4])
        w5 = (seq[4], seq[5])
        analysis.waves.update({"1": w1, "2": w2, "3": w3, "4": w4, "5": w5})

        l1, l2, l3, l4, l5 = (length(*w) for w in (w1, w2, w3, w4, w5))

        # 규칙 1: w2는 w1을 100% 초과 되돌릴 수 없다
        if -l2 >= l1:  # l2는 음수(되돌림). |l2| >= l1 이면 위반
            analysis.rule_violations.append("Rule1: 2파가 1파를 100% 이상 되돌림")
        # 규칙 2: w3는 1,3,5 중 가장 짧지 않다
        if l3 < l1 and l3 < l5:
            analysis.rule_violations.append("Rule2: 3파가 1·3·5 중 가장 짧음")
        # 규칙 3: w4는 w1의 영역과 겹치지 않는다 (현물 기준)
        end_w1 = w1[1].price
        end_w4 = w4[1].price
        if direction == "up" and end_w4 <= end_w1:
            analysis.rule_violations.append("Rule3: 4파가 1파 고점 이하로 하락")
        if direction == "down" and end_w4 >= end_w1:
            analysis.rule_violations.append("Rule3: 4파가 1파 저점 이상으로 상승")

        # 피보나치 적합도
        analysis.fib_scores["w2_retr"] = _fib_score(-l2 / l1 if l1 else 0, 0.618, tol=0.25)
        analysis.fib_scores["w3_ext"] = _fib_score(l3 / l1 if l1 else 0, 1.618, tol=0.35)
        analysis.fib_scores["w4_retr"] = _fib_score(-l4 / l3 if l3 else 0, 0.382, tol=0.25)

    # 조정 ABC 6,7,8
    if len(seq) >= 9:
        wA = (seq[5], seq[6])
        wB = (seq[6], seq[7])
        wC = (seq[7], seq[8])
        analysis.waves.update({"A": wA, "B": wB, "C": wC})
        lA = length(wA[0], wA[1]) * -1  # A는 추세 반대 방향이므로 양수화
        lB = length(wB[0], wB[1])
        lC = length(wC[0], wC[1]) * -1
        if lA > 0:
            analysis.fib_scores["B_retr"] = _fib_score(lB / lA if lA else 0, 0.5, tol=0.3)
            analysis.fib_scores["C_vs_A"] = _fib_score(lC / lA if lA else 0, 1.0, tol=0.4)

    # 종합 점수
    rule_penalty = 0.3 * len(analysis.rule_violations)
    fib_avg = float(np.mean(list(analysis.fib_scores.values()))) if analysis.fib_scores else 0.0
    analysis.score = max(0.0, fib_avg - rule_penalty)

    # 다음 파동 추정 + 목표가
    _project_targets(analysis, direction, sign)
    return analysis


def _project_targets(analysis: WaveAnalysis, direction: str, sign: int) -> None:
    waves = analysis.waves
    last_pivot = analysis.pivots[-1]
    fibs_pos = (1.0, 1.382, 1.618, 2.0, 2.618)

    def length(a: Pivot, b: Pivot) -> float:
        return (b.price - a.price) * sign

    if "5" not in waves:
        # 1~4까지 있는 상태에서 다음 파동
        if "4" in waves:
            w1 = waves["1"]
            w3 = waves.get("3")
            base = waves["4"][1].price
            l1 = length(*w1)
            l3 = length(*w3) if w3 else l1
            # 5파 목표: 0.618*(1파+3파), 1.0*(1파+3파)
            for r in (0.618, 1.0, 1.618):
                analysis.targets[f"5파 목표 (×{r})"] = base + sign * (l1 + l3) * r
            analysis.next_wave = "5"
        elif "3" in waves:
            base = waves["3"][1].price
            l3 = length(*waves["3"])
            for r in (0.382, 0.5, 0.618):
                analysis.targets[f"4파 되돌림 ({int(r*100)}%)"] = base - sign * l3 * r
            analysis.next_wave = "4"
        elif "2" in waves:
            base = waves["2"][1].price
            l1 = length(*waves["1"])
            for r in (1.618, 2.0, 2.618):
                analysis.targets[f"3파 목표 (×{r})"] = base + sign * l1 * r
            analysis.next_wave = "3"
        elif "1" in waves:
            base = waves["1"][1].price
            l1 = length(*waves["1"])
            for r in (0.382, 0.5, 0.618):
                analysis.targets[f"2파 되돌림 ({int(r*100)}%)"] = base - sign * l1 * r
            analysis.next_wave = "2"
        return

    # 5파까지 완성 → ABC 조정 시작 추정
    if "C" not in waves:
        if "A" not in waves:
            base = waves["5"][1].price
            l_total = length(waves["1"][0], waves["5"][1])
            for r in (0.382, 0.5, 0.618):
                analysis.targets[f"A파 되돌림 ({int(r*100)}%)"] = base - sign * l_total * r
            analysis.next_wave = "A"
        elif "B" not in waves:
            base = waves["A"][1].price
            lA = -length(*waves["A"])  # A의 절대 길이
            for r in (0.382, 0.5, 0.618):
                analysis.targets[f"B파 반등 ({int(r*100)}%)"] = base + sign * lA * r
            analysis.next_wave = "B"
        else:
            base = waves["B"][1].price
            lA = -length(*waves["A"])
            for r in (1.0, 1.382, 1.618):
                analysis.targets[f"C파 목표 (×{r})"] = base - sign * lA * r
            analysis.next_wave = "C"
    else:
        # ABC 완료 → 새로운 추세 1파 시작 가능성
        analysis.next_wave = "신규 1파"
        base = last_pivot.price
        l_total = abs(length(waves["1"][0], waves["5"][1]))
        for r in (0.382, 0.618, 1.0):
            analysis.targets[f"신규 추세 목표 ({int(r*100)}%)"] = base + sign * l_total * r


def best_analysis(df: pd.DataFrame, pct_candidates: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12)) -> tuple[WaveAnalysis | None, list[Pivot]]:
    """여러 ZigZag 임계값을 시도해 가장 점수 높은 분석을 반환."""
    best: WaveAnalysis | None = None
    best_pivots: list[Pivot] = []
    for pct in pct_candidates:
        pivots = zigzag(df, pct=pct)
        if len(pivots) < 4:
            continue
        for direction in ("up", "down"):
            a = analyze_impulse(pivots, direction=direction)
            if a is None:
                continue
            a.notes.append(f"ZigZag {int(pct*100)}% / 방향={direction}")
            if best is None or a.score > best.score:
                best = a
                best_pivots = pivots
    return best, best_pivots
