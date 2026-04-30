"""KRX 시세 데이터 로더. pykrx를 사용해 OHLCV를 가져오고 디스크 캐시를 둔다."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def get_ticker_name(ticker: str) -> str:
    try:
        name = stock.get_market_ticker_name(ticker)
        return name or ticker
    except Exception:
        return ticker


def search_ticker(query: str) -> list[tuple[str, str]]:
    """종목명/코드로 검색. (ticker, name) 리스트를 반환."""
    query = query.strip()
    if not query:
        return []

    if query.isdigit() and len(query) == 6:
        name = get_ticker_name(query)
        if name and name != query:
            return [(query, name)]
        return []

    today = _fmt(datetime.now())
    results: list[tuple[str, str]] = []
    for market in ("KOSPI", "KOSDAQ"):
        try:
            tickers = stock.get_market_ticker_list(today, market=market)
        except Exception:
            continue
        for t in tickers:
            name = stock.get_market_ticker_name(t)
            if name and query in name:
                results.append((t, name))
    return results


def load_ohlcv(ticker: str, years: int = 10, use_cache: bool = True) -> pd.DataFrame:
    """지난 N년치 일봉 OHLCV. index=Date, columns=[Open, High, Low, Close, Volume]."""
    end = datetime.now()
    start = end - timedelta(days=int(365.25 * years) + 5)
    cache_path = CACHE_DIR / f"{ticker}_{years}y.parquet"

    if use_cache and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime) < timedelta(hours=12):
            try:
                df = pd.read_parquet(cache_path)
                df.index = pd.to_datetime(df.index)
                return df
            except Exception:
                pass

    df = stock.get_market_ohlcv(_fmt(start), _fmt(end), ticker)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
        }
    )
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    try:
        df.to_parquet(cache_path)
    except Exception:
        pass

    return df


def get_latest_price(ticker: str) -> float | None:
    """최신 종가. 캐시된 데이터가 있으면 그것을 우선 사용."""
    end = datetime.now()
    start = end - timedelta(days=10)
    try:
        df = stock.get_market_ohlcv(_fmt(start), _fmt(end), ticker)
        if df is not None and not df.empty:
            return float(df["종가"].iloc[-1])
    except Exception:
        pass
    return None
