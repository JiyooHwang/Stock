"""벤치마크(KOSPI / KOSDAQ / KOSPI200) 시세 + 종목과의 상대 성과."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "index"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INDICES: dict[str, str] = {
    "KOSPI": "1001",
    "KOSDAQ": "2001",
    "KOSPI200": "1028",
    "KRX300": "5042",
}


def load_index_ohlcv(index_code: str, years: int = 3, use_cache: bool = True) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=int(365.25 * years) + 5)
    cache_path = CACHE_DIR / f"{index_code}_{years}y.parquet"
    if use_cache and cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(hours=12):
            try:
                df = pd.read_parquet(cache_path)
                df.index = pd.to_datetime(df.index)
                return df
            except Exception:
                pass
    try:
        df = stock.get_index_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), index_code)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}
    )
    df.index = pd.to_datetime(df.index)
    try:
        df.to_parquet(cache_path)
    except Exception:
        pass
    return df


def detect_market(ticker: str) -> str | None:
    """KOSPI / KOSDAQ 둘 중 어디 상장종목인지."""
    today = datetime.now().strftime("%Y%m%d")
    for market in ("KOSPI", "KOSDAQ"):
        try:
            tickers = stock.get_market_ticker_list(today, market=market)
            if ticker in tickers:
                return market
        except Exception:
            continue
    return None


def relative_performance(
    stock_df: pd.DataFrame, index_df: pd.DataFrame
) -> pd.DataFrame:
    """공통 기간으로 정렬하고, 시작일=100 으로 정규화한 두 시계열."""
    if stock_df.empty or index_df.empty:
        return pd.DataFrame()
    join = pd.concat(
        [stock_df["Close"].rename("Stock"), index_df["Close"].rename("Index")],
        axis=1,
    ).dropna()
    if join.empty:
        return pd.DataFrame()
    norm = (join / join.iloc[0]) * 100.0
    norm["초과수익(%)"] = norm["Stock"] - norm["Index"]
    return norm


def perf_summary(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> dict[str, float]:
    df = relative_performance(stock_df, index_df)
    if df.empty:
        return {}
    return {
        "종목 수익률(%)": df["Stock"].iloc[-1] - 100,
        "지수 수익률(%)": df["Index"].iloc[-1] - 100,
        "초과수익(%)": df["초과수익(%)"].iloc[-1],
    }
