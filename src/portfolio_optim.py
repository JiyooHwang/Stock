"""포트폴리오 비중 최적화.

- equal_weight: 동일 비중 (벤치마크용)
- risk_parity: 1/변동성 가중 (역변동성). 단순하지만 견고하다.
- min_variance: 공분산 행렬 기반 최소분산 (선택)

수익률은 일별 단순수익률(pct_change)을 사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """가격 시계열(컬럼=종목, 인덱스=날짜)을 일별 수익률로."""
    return price_df.pct_change().dropna(how="all")


def equal_weight(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series([1.0 / n] * n, index=tickers)


def risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """1/연환산변동성 비중. 변동성 큰 종목은 비중 ↓."""
    if returns.empty:
        return pd.Series(dtype=float)
    vols = returns.std() * np.sqrt(252)
    vols = vols.replace(0, np.nan).dropna()
    if vols.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / vols
    return inv / inv.sum()


def min_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """최소분산 포트폴리오. w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1).

    공분산 비특이일 때만 동작. 음수 비중 허용(공매도). 결과가 부적절하면
    risk_parity로 폴백.
    """
    if returns.empty or returns.shape[1] < 2:
        return pd.Series(dtype=float)
    cov = returns.cov().values
    try:
        inv = np.linalg.pinv(cov)
        ones = np.ones(cov.shape[0])
        w = inv @ ones
        w = w / w.sum()
        if np.any(np.isnan(w)):
            raise ValueError("nan in weights")
        return pd.Series(w, index=returns.columns)
    except Exception:
        return risk_parity_weights(returns)


def portfolio_metrics(returns: pd.DataFrame, weights: pd.Series, rf: float = 0.02) -> dict[str, float]:
    """포트폴리오 핵심 지표 (CAGR / 변동성 / Sharpe / MDD)."""
    if returns.empty or weights.empty:
        return {}
    aligned = returns[weights.index]
    port_ret = (aligned * weights).sum(axis=1)
    if port_ret.empty:
        return {}
    cum = (1 + port_ret).cumprod()
    days = (returns.index[-1] - returns.index[0]).days or 1
    cagr = cum.iloc[-1] ** (365.25 / days) - 1
    vol = port_ret.std() * np.sqrt(252)
    sharpe = ((port_ret.mean() - rf / 252) / port_ret.std()) * np.sqrt(252) if port_ret.std() > 0 else 0.0
    mdd = (cum / cum.cummax() - 1).min()
    return {
        "CAGR(%)": cagr * 100,
        "변동성(%)": vol * 100,
        "Sharpe": sharpe,
        "MDD(%)": mdd * 100,
        "총수익률(%)": (cum.iloc[-1] - 1) * 100,
    }
