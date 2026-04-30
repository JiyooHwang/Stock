"""규칙 기반 백테스트.

규칙(기본): 종가 > 200일 이평선 AND 12개월 모멘텀 > 0 일 때 long, 아니면 현금.
일별 시그널 → 다음 거래일 시가에 진입한다고 가정(시그널 lag=1).
거래비용은 매매당 0.1% (수수료+세금 단순 가정).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series       # 전략 자산 곡선 (1.0 시작)
    benchmark: pd.Series    # 매수후보유 곡선
    trades: pd.DataFrame    # 거래 기록
    metrics: dict[str, float]


def _metrics(curve: pd.Series, daily_rf: float = 0.02 / 252) -> dict[str, float]:
    if len(curve) < 2:
        return {}
    rets = curve.pct_change().dropna()
    days = (curve.index[-1] - curve.index[0]).days or 1
    cagr = curve.iloc[-1] ** (365.25 / days) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = ((rets.mean() - daily_rf) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0
    drawdown = curve / curve.cummax() - 1
    mdd = drawdown.min()
    return {
        "총수익률(%)": (curve.iloc[-1] - 1) * 100,
        "CAGR(%)": cagr * 100,
        "변동성(%)": vol * 100,
        "Sharpe": sharpe,
        "MDD(%)": mdd * 100,
    }


def run_backtest(
    df: pd.DataFrame,
    ma_window: int = 200,
    momentum_window: int = 252,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """단일 종목 백테스트. df는 OHLCV(인덱스=날짜)."""
    if len(df) < max(ma_window, momentum_window) + 5:
        raise ValueError("백테스트에 충분한 데이터가 없습니다.")

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float).reindex(close.index).ffill()
    ma = close.rolling(ma_window).mean()
    mom = close / close.shift(momentum_window) - 1

    signal = ((close > ma) & (mom > 0)).astype(int)
    # 시그널 lag=1: 오늘 시그널 → 내일 시가 진입
    pos = signal.shift(1).fillna(0)

    # 일별 수익률: 포지션 1이면 시가-종가가 아닌 close-to-close 단순 적용 (lag로 신호 누수 방지)
    daily_ret = close.pct_change().fillna(0) * pos

    # 거래비용 차감 (포지션 변경시)
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    cost = turnover * (cost_bps / 10000.0)
    net_ret = daily_ret - cost

    equity = (1 + net_ret).cumprod()
    benchmark = close / close.iloc[0]

    # 거래 기록 추출
    trades_list = []
    in_pos = False
    entry_date = entry_price = None
    for d, p in pos.items():
        if p == 1 and not in_pos:
            in_pos = True
            entry_date, entry_price = d, float(close.loc[d])
        elif p == 0 and in_pos:
            exit_price = float(close.loc[d])
            trades_list.append({
                "진입일": entry_date,
                "청산일": d,
                "진입가": entry_price,
                "청산가": exit_price,
                "수익률(%)": (exit_price / entry_price - 1) * 100,
                "보유일": (d - entry_date).days,
            })
            in_pos = False
    if in_pos:
        trades_list.append({
            "진입일": entry_date,
            "청산일": close.index[-1],
            "진입가": entry_price,
            "청산가": float(close.iloc[-1]),
            "수익률(%)": (close.iloc[-1] / entry_price - 1) * 100,
            "보유일": (close.index[-1] - entry_date).days,
        })
    trades = pd.DataFrame(trades_list)

    m_strat = _metrics(equity)
    m_bench = _metrics(benchmark)
    metrics = {f"전략 {k}": v for k, v in m_strat.items()}
    metrics.update({f"매수후보유 {k}": v for k, v in m_bench.items()})
    if not trades.empty:
        wins = (trades["수익률(%)"] > 0).sum()
        metrics["거래 횟수"] = float(len(trades))
        metrics["승률(%)"] = float(wins / len(trades) * 100)
        metrics["평균 수익률(%)"] = float(trades["수익률(%)"].mean())

    return BacktestResult(equity=equity, benchmark=benchmark, trades=trades, metrics=metrics)


def run_walkforward(
    df: pd.DataFrame,
    train_years: int = 3,
    test_years: int = 1,
    ma_window: int = 200,
    momentum_window: int = 252,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """시기별 분할 백테스트 — 시간 구간을 나눠 룰의 일관성을 본다.

    이 룰은 학습 파라미터가 없으므로 엄밀한 워크포워드는 아니다. 단,
    시기마다 성과가 어떻게 다른지(특정 시기에만 작동하는지) 진단할 수 있다.

    각 슬라이스의 성과를 매수후보유 대비 비교해 행으로 반환한다.
    """
    if len(df) < (train_years + test_years) * 252:
        return pd.DataFrame()

    splits: list[dict] = []
    start_date = df.index[0]
    end_date = df.index[-1]
    cur_test_start = start_date + pd.DateOffset(years=train_years)

    while cur_test_start + pd.DateOffset(years=test_years) <= end_date:
        test_end = cur_test_start + pd.DateOffset(years=test_years)
        # MA/모멘텀 lookback 위해 전체 history를 넣고, test 구간만 잘라낸다
        sub = df.loc[:test_end]
        if len(sub) < max(ma_window, momentum_window) + 5:
            cur_test_start += pd.DateOffset(years=test_years)
            continue
        try:
            res = run_backtest(sub, ma_window=ma_window, momentum_window=momentum_window, cost_bps=cost_bps)
        except ValueError:
            cur_test_start += pd.DateOffset(years=test_years)
            continue

        eq_slice = res.equity.loc[cur_test_start:test_end]
        bench_slice = res.benchmark.loc[cur_test_start:test_end]
        if len(eq_slice) < 5 or len(bench_slice) < 5:
            cur_test_start += pd.DateOffset(years=test_years)
            continue

        eq_norm = eq_slice / eq_slice.iloc[0]
        bench_norm = bench_slice / bench_slice.iloc[0]
        strat_ret = (eq_norm.iloc[-1] - 1) * 100
        bench_ret = (bench_norm.iloc[-1] - 1) * 100

        splits.append(
            {
                "기간": f"{cur_test_start:%Y-%m} ~ {test_end:%Y-%m}",
                "전략(%)": round(strat_ret, 2),
                "매수후보유(%)": round(bench_ret, 2),
                "초과수익(%)": round(strat_ret - bench_ret, 2),
                "전략 우세": "✅" if strat_ret > bench_ret else "❌",
            }
        )
        cur_test_start += pd.DateOffset(years=test_years)

    return pd.DataFrame(splits)
