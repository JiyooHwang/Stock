"""DART (전자공시시스템) 공식 API 클라이언트.

OPEN DART (https://opendart.fss.or.kr) 무료 API. 사용자가 가입 후
인증키를 발급받아 환경변수 `OPEN_DART_KEY` 또는 Streamlit secrets,
또는 앱 사이드바에서 입력하면 활성화된다. 키가 없으면 빈 결과 반환.

기능:
- ticker(6자리) → corp_code 매핑 (DART의 corpCode.xml.zip 캐시)
- 최근 N일 공시 목록
- 단일회사 주요 재무지표 (매출액·영업이익·당기순이익) — 분기/연간

corpCode.xml은 약 5MB라 7일 캐시한다.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

DART_BASE = "https://opendart.fss.or.kr/api"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "dart"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CORP_MAP_PATH = CACHE_DIR / "corp_map.json"


def _get_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("OPEN_DART_KEY")
    if env:
        return env
    try:
        import streamlit as st
        return st.secrets.get("OPEN_DART_KEY")
    except Exception:
        return None


def diagnose(ticker: str, api_key: str | None = None) -> dict:
    """공시 조회 실패 시 어디서 막혔는지 단계별로 진단."""
    info: dict = {
        "key_found": False,
        "key_source": "없음",
        "key_preview": None,
        "corp_map_size": None,
        "corp_code": None,
        "api_status": None,
        "api_message": None,
        "filings_count": None,
        "error": None,
    }
    if api_key:
        key = api_key
        info["key_source"] = "사이드바 입력"
    elif os.environ.get("OPEN_DART_KEY"):
        key = os.environ.get("OPEN_DART_KEY")
        info["key_source"] = "환경변수 OPEN_DART_KEY"
    else:
        try:
            import streamlit as st
            key = st.secrets.get("OPEN_DART_KEY")
            if key:
                info["key_source"] = "Streamlit Cloud Secrets"
        except Exception:
            key = None

    if not key:
        info["error"] = "API 키를 찾지 못했습니다. Secrets 또는 사이드바를 확인하세요."
        return info

    info["key_found"] = True
    info["key_preview"] = f"{key[:4]}...{key[-4:]} (길이 {len(key)})"

    try:
        cmap = _load_corp_map(key)
        info["corp_map_size"] = len(cmap)
    except Exception as e:
        info["error"] = f"corpCode.xml 로드 실패: {e}"
        return info

    if not cmap:
        info["error"] = (
            "corpCode.xml 다운로드 실패. 키가 잘못됐거나 DART 서버 일시 장애 가능. "
            "키 앞뒤 공백/따옴표 포함 여부를 확인하세요."
        )
        return info

    info["corp_code"] = cmap.get(ticker)
    if not info["corp_code"]:
        info["error"] = f"ticker {ticker} 이(가) corpCode 매핑에 없습니다 (상장폐지/종목코드 오타 가능)."
        return info

    try:
        import requests
        end = datetime.now()
        start = end - timedelta(days=90)
        r = requests.get(
            f"{DART_BASE}/list.json",
            params={
                "crtfc_key": key,
                "corp_code": info["corp_code"],
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_count": 100,
            },
            timeout=10,
        )
        data = r.json()
        info["api_status"] = data.get("status")
        info["api_message"] = data.get("message")
        info["filings_count"] = len(data.get("list", []))
    except Exception as e:
        info["error"] = f"list.json 호출 실패: {e}"
    return info


def _load_corp_map(api_key: str, refresh_days: int = 7) -> dict[str, str]:
    """ticker → corp_code 매핑. 디스크 캐시 7일."""
    if CORP_MAP_PATH.exists():
        age = datetime.now() - datetime.fromtimestamp(CORP_MAP_PATH.stat().st_mtime)
        if age < timedelta(days=refresh_days):
            try:
                return json.loads(CORP_MAP_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

    try:
        import requests
        r = requests.get(f"{DART_BASE}/corpCode.xml", params={"crtfc_key": api_key}, timeout=20)
        r.raise_for_status()
    except Exception:
        return {}

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        xml_text = z.read(z.namelist()[0]).decode("utf-8")
    except Exception:
        return {}

    out: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
        for child in root.findall("list"):
            stock_code = (child.findtext("stock_code") or "").strip()
            corp_code = (child.findtext("corp_code") or "").strip()
            if stock_code and corp_code:
                out[stock_code] = corp_code
    except Exception:
        return {}

    try:
        CORP_MAP_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return out


def get_corp_code(ticker: str, api_key: str | None = None) -> str | None:
    key = _get_key(api_key)
    if not key:
        return None
    return _load_corp_map(key).get(ticker)


def list_disclosures(ticker: str, days: int = 60, api_key: str | None = None) -> pd.DataFrame:
    """최근 N일 공시 목록."""
    key = _get_key(api_key)
    if not key:
        return pd.DataFrame()
    corp = get_corp_code(ticker, api_key=key)
    if not corp:
        return pd.DataFrame()
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        import requests
        r = requests.get(
            f"{DART_BASE}/list.json",
            params={
                "crtfc_key": key,
                "corp_code": corp,
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_count": 100,
            },
            timeout=10,
        )
        data = r.json()
    except Exception:
        return pd.DataFrame()
    if data.get("status") != "000":
        return pd.DataFrame()
    rows = data.get("list", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = ["rcept_dt", "report_nm", "rcept_no", "rm"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.columns = ["접수일", "보고서명", "rcept_no", "비고"][: len(df.columns)]
    df["접수일"] = pd.to_datetime(df["접수일"])
    df["URL"] = df["rcept_no"].apply(
        lambda x: f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={x}"
    )
    return df.drop(columns=["rcept_no"])


def get_quarterly_financials(
    ticker: str, year: int, quarter: int, api_key: str | None = None
) -> pd.DataFrame:
    """단일회사 주요 재무지표. quarter: 1=1분기, 2=반기, 3=3분기, 4=사업보고서."""
    key = _get_key(api_key)
    if not key:
        return pd.DataFrame()
    corp = get_corp_code(ticker, api_key=key)
    if not corp:
        return pd.DataFrame()
    reprt_code = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}.get(quarter)
    if not reprt_code:
        return pd.DataFrame()
    try:
        import requests
        r = requests.get(
            f"{DART_BASE}/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": key,
                "corp_code": corp,
                "bsns_year": str(year),
                "reprt_code": reprt_code,
                "fs_div": "CFS",  # 연결재무제표
            },
            timeout=10,
        )
        data = r.json()
    except Exception:
        return pd.DataFrame()
    if data.get("status") != "000":
        return pd.DataFrame()
    rows = data.get("list", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df
