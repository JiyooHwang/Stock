"""KRX 시세 데이터 로더. pykrx를 사용해 OHLCV를 가져오고 디스크 캐시를 둔다."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NAME_MAP_PATH = CACHE_DIR / "ticker_name_map.json"
NAME_MAP_TTL = timedelta(days=1)


def _fmt(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _build_name_map() -> dict[str, str]:
    """KOSPI + KOSDAQ 의 ticker→name 풀 매핑.

    pykrx 가 제공하는 bulk API 가 있으면 사용해 한 번에 받고,
    없으면 종목 리스트 + 개별 이름 조회로 폴백한다.
    """
    today = _fmt(datetime.now())
    out: dict[str, str] = {}

    # 우선 bulk API 시도 (pykrx 신버전)
    bulk_fn = getattr(stock, "get_market_ticker_and_name", None)
    if callable(bulk_fn):
        for market in ("KOSPI", "KOSDAQ"):
            try:
                series = bulk_fn(today, market=market)
                if series is None:
                    continue
                if isinstance(series, pd.Series):
                    for t, n in series.items():
                        if t and n:
                            out[str(t)] = str(n)
                elif isinstance(series, dict):
                    for t, n in series.items():
                        if t and n:
                            out[str(t)] = str(n)
            except Exception:
                continue

    # 부족하면 개별 조회로 보강
    if len(out) < 100:
        for market in ("KOSPI", "KOSDAQ"):
            try:
                tickers = stock.get_market_ticker_list(today, market=market)
            except Exception:
                continue
            for t in tickers:
                if t in out:
                    continue
                try:
                    n = stock.get_market_ticker_name(t)
                except Exception:
                    n = None
                if n:
                    out[t] = n
    return out


def _load_name_map(force_refresh: bool = False) -> dict[str, str]:
    """디스크 캐시된 ticker→name 매핑. 24시간 TTL."""
    if not force_refresh and NAME_MAP_PATH.exists():
        age = datetime.now() - datetime.fromtimestamp(NAME_MAP_PATH.stat().st_mtime)
        if age < NAME_MAP_TTL:
            try:
                data = json.loads(NAME_MAP_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return data
            except Exception:
                pass

    fresh = _build_name_map()
    if fresh:
        try:
            NAME_MAP_PATH.write_text(
                json.dumps(fresh, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        return fresh

    # 신규 빌드 실패 시 만료 캐시라도 사용
    if NAME_MAP_PATH.exists():
        try:
            data = json.loads(NAME_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass
    return {}


def get_ticker_name(ticker: str) -> str:
    cached = _load_name_map().get(ticker)
    if cached:
        return cached
    try:
        name = stock.get_market_ticker_name(ticker)
        return name or ticker
    except Exception:
        return ticker


def search_ticker(query: str) -> list[tuple[str, str]]:
    """종목명/코드로 검색. (ticker, name) 리스트를 반환.

    - 6자리 숫자: 정확 일치 코드 조회
    - 그 외: 캐시된 매핑에서 부분일치 (대소문자/공백 무시)
    """
    query = query.strip()
    if not query:
        return []

    if query.isdigit() and len(query) == 6:
        name = get_ticker_name(query)
        if name and name != query:
            return [(query, name)]
        return []

    name_map = _load_name_map()
    if not name_map:
        return []

    needle = query.replace(" ", "").lower()
    exact: list[tuple[str, str]] = []
    starts: list[tuple[str, str]] = []
    contains: list[tuple[str, str]] = []
    for t, n in name_map.items():
        hay = n.replace(" ", "").lower()
        if hay == needle:
            exact.append((t, n))
        elif hay.startswith(needle):
            starts.append((t, n))
        elif needle in hay:
            contains.append((t, n))
    return exact + sorted(starts, key=lambda x: x[1]) + sorted(contains, key=lambda x: x[1])


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
