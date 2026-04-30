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
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

DART_BASE = "https://opendart.fss.or.kr/api"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "dart"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CORP_MAP_PATH = CACHE_DIR / "corp_map.json"
CORP_ZIP_PATH = CACHE_DIR / "CorpCode.zip"

# DART 서버는 한국에서 호스팅되어 해외(예: Streamlit Cloud)에서 첫 연결이 느릴 수 있다.
# 또한 corpCode.xml.zip 이 ~5MB 라 read 도 오래 걸린다.
_CONNECT_TIMEOUT = 60
_READ_TIMEOUT_SMALL = 30
_READ_TIMEOUT_LARGE = 180

_UA = (
    "Mozilla/5.0 (compatible; KRX-Portfolio/1.0; +https://github.com/jiyoohwang/stock)"
)


def _http_get(url: str, *, params: dict | None = None, timeout: tuple[int, int],
              retries: int = 3, stream: bool = False):
    """requests.get with retries on connection/timeout errors."""
    import requests
    headers = {"User-Agent": _UA, "Accept": "*/*"}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return requests.get(
                url, params=params, timeout=timeout, headers=headers, stream=stream
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


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

    # 캐시된 corp_map을 먼저 보고, 비었으면 직접 fetch해서 응답 검사
    cmap: dict[str, str] = {}
    if CORP_MAP_PATH.exists():
        try:
            cmap = json.loads(CORP_MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not cmap:
        # 디스크에 zip 캐시가 있다면 그것부터 시도
        if CORP_ZIP_PATH.exists():
            try:
                cmap = _parse_corp_zip(CORP_ZIP_PATH.read_bytes())
                if cmap:
                    CORP_MAP_PATH.write_text(
                        json.dumps(cmap, ensure_ascii=False), encoding="utf-8"
                    )
            except Exception:
                pass

    if not cmap:
        # 직접 fetch해서 DART가 무슨 응답을 주는지 확인
        try:
            r = _http_get(
                f"{DART_BASE}/corpCode.xml",
                params={"crtfc_key": key},
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_LARGE),
                retries=3,
            )
            info["corp_xml_http"] = r.status_code
            info["corp_xml_content_type"] = r.headers.get("Content-Type", "")
            info["corp_xml_size_bytes"] = len(r.content)
            # DART는 키 오류 시 JSON 응답을 200으로 준다
            text_preview = r.text[:200] if isinstance(r.text, str) else ""
            if text_preview.strip().startswith("{"):
                try:
                    data = r.json()
                    info["error"] = (
                        f"DART corpCode.xml 응답 status={data.get('status')}: "
                        f"{data.get('message')}"
                    )
                    return info
                except Exception:
                    pass
            if r.status_code != 200:
                info["error"] = f"corpCode.xml HTTP {r.status_code}"
                return info
            # zip 파싱 시도
            try:
                cmap = _parse_corp_zip(r.content)
                CORP_ZIP_PATH.write_bytes(r.content)
                CORP_MAP_PATH.write_text(json.dumps(cmap, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                info["error"] = f"corpCode.xml 파싱 실패: {e} (응답 미리보기: {text_preview[:100]})"
                return info
        except Exception as e:
            info["error"] = (
                f"corpCode.xml 네트워크 오류 (3회 재시도 후 실패): {e}. "
                "DART 서버는 한국에 위치해 해외 호스팅(Streamlit Cloud 등)에서 "
                "연결이 막히거나 매우 느릴 수 있습니다. 잠시 뒤 다시 시도하거나 "
                "로컬 환경에서 한 번 실행해 corp_map 캐시를 만들어 두세요."
            )
            return info

    info["corp_map_size"] = len(cmap)
    if not cmap:
        info["error"] = "corp_map이 비어 있습니다."
        return info

    info["corp_code"] = cmap.get(ticker)
    if not info["corp_code"]:
        info["error"] = f"ticker {ticker} 이(가) corpCode 매핑에 없습니다 (상장폐지/종목코드 오타 가능)."
        return info

    try:
        end = datetime.now()
        start = end - timedelta(days=90)
        r = _http_get(
            f"{DART_BASE}/list.json",
            params={
                "crtfc_key": key,
                "corp_code": info["corp_code"],
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_count": 100,
            },
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_SMALL),
            retries=3,
        )
        data = r.json()
        info["api_status"] = data.get("status")
        info["api_message"] = data.get("message")
        info["filings_count"] = len(data.get("list", []))
    except Exception as e:
        info["error"] = f"list.json 호출 실패: {e}"
    return info


def _parse_corp_zip(content: bytes) -> dict[str, str]:
    z = zipfile.ZipFile(io.BytesIO(content))
    xml_text = z.read(z.namelist()[0]).decode("utf-8")
    root = ET.fromstring(xml_text)
    out: dict[str, str] = {}
    for child in root.findall("list"):
        stock_code = (child.findtext("stock_code") or "").strip()
        corp_code = (child.findtext("corp_code") or "").strip()
        if stock_code and corp_code:
            out[stock_code] = corp_code
    return out


def _parse_corp_xml_text(xml_text: str) -> dict[str, str]:
    root = ET.fromstring(xml_text)
    out: dict[str, str] = {}
    for child in root.findall("list"):
        sc = (child.findtext("stock_code") or "").strip()
        cc = (child.findtext("corp_code") or "").strip()
        if sc and cc:
            out[sc] = cc
    return out


def install_corp_map_from_bytes(data: bytes) -> tuple[int, str]:
    """사용자가 업로드한 파일(zip / json / xml) 바이트를 캐시에 설치.

    파일 시그니처(매직바이트) 로 형식을 자동 판별 — 확장자 무관.

    Returns: (등록된 매핑 수, 입력 형식 라벨).
    Raises: ValueError 파싱 실패 시.
    """
    if not data:
        raise ValueError("빈 파일입니다.")

    # 1) zip 시그니처 PK\x03\x04
    if data[:2] == b"PK":
        try:
            cmap = _parse_corp_zip(data)
        except Exception as e:
            raise ValueError(f"zip 파싱 실패: {e}") from e
        if not cmap:
            raise ValueError("zip 파싱은 됐지만 매핑이 비어 있습니다.")
        CORP_ZIP_PATH.write_bytes(data)
        CORP_MAP_PATH.write_text(json.dumps(cmap, ensure_ascii=False), encoding="utf-8")
        return len(cmap), "CorpCode.zip"

    # 텍스트로 디코드 (UTF-8 → CP949 → latin-1 폴백)
    text: str | None = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("파일 인코딩을 인식할 수 없습니다.")

    stripped = text.lstrip()

    # 2) JSON 시도
    if stripped.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 실패: {e}") from e
        if not isinstance(obj, dict) or not obj:
            raise ValueError("corp_map.json 이 비어 있거나 형식이 잘못되었습니다.")
        cleaned: dict[str, str] = {}
        for k, v in obj.items():
            ks = str(k).strip()
            vs = str(v).strip()
            if ks and vs:
                cleaned[ks] = vs
        if not cleaned:
            raise ValueError("corp_map.json 에 유효한 매핑이 없습니다.")
        CORP_MAP_PATH.write_text(
            json.dumps(cleaned, ensure_ascii=False), encoding="utf-8"
        )
        return len(cleaned), "corp_map.json"

    # 3) XML 시도 (DART 가 키 오류 시 JSON 본문을 줄 때도 있어 한 번 더 체크)
    if stripped.startswith("<"):
        # DART 에러 응답이 XML 형식일 수도 있음
        if "<status>" in text and "<message>" in text:
            try:
                root = ET.fromstring(text)
                status = (root.findtext("status") or "").strip()
                message = (root.findtext("message") or "").strip()
                if status and status != "000":
                    raise ValueError(
                        f"DART API 에러 응답이 업로드됐습니다 (status={status}: {message}). "
                        "키나 권한을 확인하고 다시 받으세요."
                    )
            except ET.ParseError:
                pass
        try:
            cmap = _parse_corp_xml_text(text)
        except ET.ParseError as e:
            raise ValueError(f"XML 파싱 실패: {e}") from e
        if not cmap:
            raise ValueError("XML 파싱은 됐지만 매핑이 비어 있습니다.")
        CORP_MAP_PATH.write_text(json.dumps(cmap, ensure_ascii=False), encoding="utf-8")
        return len(cmap), "CorpCode.xml"

    raise ValueError(
        "파일 형식을 인식할 수 없습니다. zip / json / xml 중 하나여야 합니다. "
        f"(앞 16바이트: {data[:16]!r})"
    )


def corp_map_status() -> dict:
    """현재 캐시 상태."""
    info: dict = {"size": 0, "age_hours": None, "path": str(CORP_MAP_PATH)}
    if CORP_MAP_PATH.exists():
        try:
            cmap = json.loads(CORP_MAP_PATH.read_text(encoding="utf-8"))
            info["size"] = len(cmap)
        except Exception:
            pass
        age = datetime.now() - datetime.fromtimestamp(CORP_MAP_PATH.stat().st_mtime)
        info["age_hours"] = round(age.total_seconds() / 3600, 1)
    return info


def _load_corp_map(api_key: str, refresh_days: int = 7) -> dict[str, str]:
    """ticker → corp_code 매핑. 디스크 캐시 7일. 네트워크 실패 시 만료 캐시도 사용."""
    cached: dict[str, str] = {}
    cache_age_days: float | None = None
    if CORP_MAP_PATH.exists():
        cache_age_days = (
            datetime.now() - datetime.fromtimestamp(CORP_MAP_PATH.stat().st_mtime)
        ).total_seconds() / 86400
        try:
            cached = json.loads(CORP_MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        if cache_age_days < refresh_days and cached:
            return cached

    # zip 파일도 캐시 (파싱은 성공했지만 JSON 쓰기 실패했을 때 등 복구용)
    if not cached and CORP_ZIP_PATH.exists():
        try:
            cached = _parse_corp_zip(CORP_ZIP_PATH.read_bytes())
            if cached:
                CORP_MAP_PATH.write_text(
                    json.dumps(cached, ensure_ascii=False), encoding="utf-8"
                )
                return cached
        except Exception:
            pass

    try:
        r = _http_get(
            f"{DART_BASE}/corpCode.xml",
            params={"crtfc_key": api_key},
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_LARGE),
            retries=3,
        )
        r.raise_for_status()
    except Exception:
        # 네트워크 실패 시 만료된 캐시라도 있으면 반환
        return cached

    try:
        out = _parse_corp_zip(r.content)
    except Exception:
        return cached

    if not out:
        return cached

    try:
        CORP_ZIP_PATH.write_bytes(r.content)
    except Exception:
        pass
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
        r = _http_get(
            f"{DART_BASE}/list.json",
            params={
                "crtfc_key": key,
                "corp_code": corp,
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_count": 100,
            },
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_SMALL),
            retries=3,
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
        r = _http_get(
            f"{DART_BASE}/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": key,
                "corp_code": corp,
                "bsns_year": str(year),
                "reprt_code": reprt_code,
                "fs_div": "CFS",  # 연결재무제표
            },
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT_SMALL),
            retries=3,
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
