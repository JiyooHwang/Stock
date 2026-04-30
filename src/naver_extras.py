"""네이버 금융 부가 데이터 — 컨센서스 목표주가, 뉴스 헤드라인.

스크래핑이라 페이지 구조가 바뀌면 깨질 수 있다 → 호출 측에서 폴백 필요.
캐시는 짧게(컨센서스 6시간, 뉴스 30분) 유지한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "naver_extras"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class Consensus:
    target_price: float | None = None  # 목표주가
    opinion: str | None = None  # 투자의견 (예: 매수, 강력매수)
    opinion_score: float | None = None  # 4점 만점
    upside_pct: float | None = None  # 현재가 대비 (계산값)


def _fresh(path: Path, hours: int) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=hours)


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    s = re.sub(r"[,\s원배%]", "", text)
    if s in ("", "-", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_consensus(ticker: str, current_price: float | None = None) -> Consensus | None:
    """네이버 종목 메인 페이지에서 컨센서스 목표가/투자의견을 추출.

    페이지 우측 사이드바 'aside_invest_info' 안에 '목표주가'와
    '투자의견' 텍스트가 노출되는 종목이 있고, 없는 종목도 있다.
    """
    cache_path = CACHE_DIR / f"consensus_{ticker}.json"
    if _fresh(cache_path, hours=6):
        try:
            return Consensus(**json.loads(cache_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    try:
        r = requests.get(
            f"https://finance.naver.com/item/coinfo.naver?code={ticker}",
            headers={"User-Agent": USER_AGENT},
            timeout=8,
        )
        r.raise_for_status()
    except Exception:
        return None
    r.encoding = "euc-kr"
    soup = BeautifulSoup(r.text, "html.parser")
    c = Consensus()

    # 컨센서스 영역: 클래스 'cmp_comp' 또는 '투자의견' 텍스트 근처
    text_blob = soup.get_text(" ", strip=True)

    # 목표주가
    m = re.search(r"목표주가\s*([\d,]+)", text_blob)
    if m:
        c.target_price = _to_float(m.group(1))

    # 투자의견 점수 (예: "투자의견 4.00 매수")
    m = re.search(r"투자의견\s*([\d.]+)\s*([가-힣]+)?", text_blob)
    if m:
        c.opinion_score = _to_float(m.group(1))
        if m.group(2):
            c.opinion = m.group(2)

    if not (c.target_price or c.opinion_score):
        return None

    if c.target_price and current_price and current_price > 0:
        c.upside_pct = (c.target_price / current_price - 1) * 100

    try:
        cache_path.write_text(json.dumps(asdict(c), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return c


def fetch_news(ticker: str, max_items: int = 15):
    """종목 관련 최근 뉴스 헤드라인 + 링크.

    페이지: https://finance.naver.com/item/news_news.naver?code=...
    """
    import pandas as pd

    cache_path = CACHE_DIR / f"news_{ticker}.json"
    if _fresh(cache_path, hours=0.5):
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return pd.DataFrame(data)
        except Exception:
            pass

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return pd.DataFrame()

    url = f"https://finance.naver.com/item/news_news.naver?code={ticker}&page=1&sm=title_entity_id.basic&clusterId="
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=8)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()
    r.encoding = "euc-kr"
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    for tr in soup.select("tr"):
        a = tr.select_one("td.title a")
        if not a:
            continue
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if href and not href.startswith("http"):
            href = "https://finance.naver.com" + href
        info = tr.select_one("td.info")
        date = tr.select_one("td.date")
        rows.append(
            {
                "제목": title,
                "언론사": info.get_text(strip=True) if info else "",
                "일시": date.get_text(strip=True) if date else "",
                "URL": href,
            }
        )
        if len(rows) >= max_items:
            break

    if not rows:
        return pd.DataFrame()

    try:
        cache_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return pd.DataFrame(rows)
