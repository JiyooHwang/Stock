"""DART CorpCode.zip 다운로드 + corp_map.json 생성 (로컬 실행용).

Streamlit Cloud 등에서 `opendart.fss.or.kr` 로 직접 연결이 막힐 때 사용.
한국 네트워크가 닿는 PC 에서 한 번 실행하면 `corp_map.json` 이 생성된다.
이 파일을 앱 사이드바의 `📦 DART corp_map 업로드` 에 올리면 끝.

사용법:
    OPEN_DART_KEY=발급받은키 python tools/build_corp_map.py
    # 또는
    python tools/build_corp_map.py 발급받은키
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

OUT_DIR = Path(__file__).resolve().parent.parent
JSON_OUT = OUT_DIR / "corp_map.json"
ZIP_OUT = OUT_DIR / "CorpCode.zip"


def main() -> int:
    key = os.environ.get("OPEN_DART_KEY") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not key:
        print("ERROR: API 키가 필요합니다. OPEN_DART_KEY 환경변수 또는 첫 번째 인자.", file=sys.stderr)
        return 1

    print("→ corpCode.xml.zip 다운로드 중...")
    r = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": key},
        timeout=120,
    )
    r.raise_for_status()
    if r.headers.get("Content-Type", "").startswith("application/json"):
        print(f"ERROR: API 응답: {r.text[:200]}", file=sys.stderr)
        return 2

    ZIP_OUT.write_bytes(r.content)
    print(f"  저장: {ZIP_OUT} ({len(r.content):,} bytes)")

    z = zipfile.ZipFile(io.BytesIO(r.content))
    xml_text = z.read(z.namelist()[0]).decode("utf-8")
    root = ET.fromstring(xml_text)

    cmap: dict[str, str] = {}
    for child in root.findall("list"):
        sc = (child.findtext("stock_code") or "").strip()
        cc = (child.findtext("corp_code") or "").strip()
        if sc and cc:
            cmap[sc] = cc

    JSON_OUT.write_text(json.dumps(cmap, ensure_ascii=False), encoding="utf-8")
    print(f"  저장: {JSON_OUT} ({len(cmap):,}개 매핑)")
    print()
    print("✅ 완료. 둘 중 하나를 앱 사이드바에 업로드하세요:")
    print(f"   - {ZIP_OUT}  (zip 그대로)")
    print(f"   - {JSON_OUT} (가벼움)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
