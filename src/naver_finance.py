"""네이버 금융(Naver Finance) 재무 데이터 스크래퍼.

종목 메인 페이지의 "투자정보" 사이드바와 "기업실적분석" 표에서
PER/PBR/EPS/BPS/ROE/부채비율/영업이익률/배당수익률을 추출한다.

pykrx는 PER/PBR/DIV/EPS/BPS만 제공하므로, 네이버 데이터를 우선 시도하고
실패 시(차단·구조 변경·종목 미상장) 호출 측에서 pykrx로 폴백한다.

데이터는 1일 단위로 디스크 캐시한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "naver"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class Fundamentals:
    per: float | None = None
    pbr: float | None = None
    eps: float | None = None
    bps: float | None = None
    forward_per: float | None = None
    dividend_yield: float | None = None  # %
    market_cap_eok: float | None = None  # 시가총액(억원)
    roe: float | None = None  # %
    debt_ratio: float | None = None  # 부채비율 %
    op_margin: float | None = None  # 영업이익률 %
    net_margin: float | None = None  # 순이익률 %
    fiscal_year: str | None = None  # 기업실적 기준 회계연도
    source: str = "naver"


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.json"


def _read_cache(ticker: str, ttl_hours: int = 24) -> Fundamentals | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    if age > timedelta(hours=ttl_hours):
        return None
    try:
        return Fundamentals(**json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None


def _write_cache(ticker: str, f: Fundamentals) -> None:
    try:
        _cache_path(ticker).write_text(
            json.dumps(asdict(f), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    s = re.sub(r"[,\s%원배]", "", text)
    s = s.replace("N/A", "").replace("-", "" if s.strip("-") == "" else s)
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_market_cap(text: str | None) -> float | None:
    """'462조 4,200억원' 형태를 억원 단위 float로."""
    if not text:
        return None
    text = text.replace(",", "").replace(" ", "")
    eok = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)조", text)
    if m:
        eok += float(m.group(1)) * 10000
    m = re.search(r"(\d+(?:\.\d+)?)억", text)
    if m:
        eok += float(m.group(1))
    return eok or None


def fetch_naver_fundamentals(ticker: str, use_cache: bool = True) -> Fundamentals | None:
    """네이버 금융에서 재무 지표 수집. 실패 시 None."""
    if use_cache:
        cached = _read_cache(ticker)
        if cached:
            return cached

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=8)
        r.raise_for_status()
    except Exception:
        return None

    r.encoding = "euc-kr"
    soup = BeautifulSoup(r.text, "html.parser")
    f = Fundamentals()

    # 1) 투자정보 사이드바: PER, EPS, PBR, BPS, 배당수익률, 시가총액
    aside = soup.select_one("div.aside_invest_info") or soup
    for th in aside.find_all("th"):
        label = th.get_text(strip=True).replace("\n", "")
        td = th.find_next("td") or th.find_next("em")
        if td is None:
            continue
        val_text = td.get_text(" ", strip=True)
        if "시가총액" in label and "순위" not in label and f.market_cap_eok is None:
            f.market_cap_eok = _parse_market_cap(val_text)
        elif label.startswith("PER") and "추정" not in label and f.per is None:
            # "16.50배 l EPS 4,500원" 같은 텍스트
            m = re.search(r"(-?\d[\d.,]*)\s*배", val_text)
            if m:
                f.per = _to_float(m.group(1))
            m = re.search(r"EPS\s*(-?\d[\d.,]*)", val_text)
            if m:
                f.eps = _to_float(m.group(1))
        elif "추정PER" in label or "선행PER" in label:
            m = re.search(r"(-?\d[\d.,]*)\s*배", val_text)
            if m:
                f.forward_per = _to_float(m.group(1))
        elif label.startswith("PBR") and f.pbr is None:
            m = re.search(r"(-?\d[\d.,]*)\s*배", val_text)
            if m:
                f.pbr = _to_float(m.group(1))
            m = re.search(r"BPS\s*(-?\d[\d.,]*)", val_text)
            if m:
                f.bps = _to_float(m.group(1))
        elif "배당수익률" in label and f.dividend_yield is None:
            m = re.search(r"(-?\d[\d.,]*)\s*%", val_text)
            if m:
                f.dividend_yield = _to_float(m.group(1))

    # 2) 기업실적분석 테이블에서 ROE / 부채비율 / 영업이익률 / 순이익률
    try:
        table = soup.find("table", {"summary": re.compile(r"기업실적분석|시세")})
        # 좀 더 안전하게: 클래스로 찾기
        if table is None:
            table = soup.select_one("table.tb_type1.tb_num.tb_type1_ifrs") or \
                    soup.select_one("table.tb_type1.tb_num")
        if table is not None:
            tables = pd.read_html(str(table))
            if tables:
                df = tables[0]
                df.columns = [
                    " ".join([str(x) for x in c]) if isinstance(c, tuple) else str(c)
                    for c in df.columns
                ]
                first_col = df.columns[0]
                # 마지막 "실적(E 아닌)" 컬럼을 찾는다 — 없으면 마지막 컬럼
                result_cols = [c for c in df.columns[1:] if "(E)" not in c and "예상" not in c]
                target_col = result_cols[-1] if result_cols else df.columns[-1]
                f.fiscal_year = target_col

                def _row_value(keyword: str) -> float | None:
                    mask = df[first_col].astype(str).str.replace(" ", "").str.contains(
                        keyword.replace(" ", ""), regex=False, na=False
                    )
                    sub = df[mask]
                    if sub.empty:
                        return None
                    return _to_float(str(sub.iloc[0][target_col]))

                if f.roe is None:
                    f.roe = _row_value("ROE")
                if f.debt_ratio is None:
                    f.debt_ratio = _row_value("부채비율")
                if f.op_margin is None:
                    f.op_margin = _row_value("영업이익률")
                if f.net_margin is None:
                    f.net_margin = _row_value("순이익률")
    except Exception:
        pass

    # 모든 값이 None이면 실패로 간주
    if all(
        v is None
        for v in (f.per, f.pbr, f.eps, f.bps, f.roe, f.dividend_yield)
    ):
        return None

    _write_cache(ticker, f)
    return f
