"""보유 주식(포트폴리오) 자료구조와 직렬화.

웹 앱은 브라우저 세션(`st.session_state`)에 데이터를 보관하고
import/export로 영구 저장한다 — 인스턴스를 공유하는 다른 방문자에게
데이터가 새지 않게 하기 위함.

`load_portfolio` / `save_portfolio`(파일 IO)는 로컬 CLI/스크립트용으로 남겨둔다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PORTFOLIO_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio.json"
PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Holding:
    ticker: str
    name: str
    quantity: int
    avg_price: float
    memo: str = ""


def _load_raw() -> list[dict]:
    if not PORTFOLIO_PATH.exists():
        return []
    try:
        return json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_portfolio() -> list[Holding]:
    return [Holding(**row) for row in _load_raw()]


def save_portfolio(holdings: list[Holding]) -> None:
    payload = [asdict(h) for h in holdings]
    PORTFOLIO_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def upsert_holding(holdings: list[Holding], new: Holding) -> list[Holding]:
    """같은 종목이 있으면 평단가/수량을 가중평균으로 합치고, 없으면 추가."""
    for i, h in enumerate(holdings):
        if h.ticker == new.ticker:
            total_qty = h.quantity + new.quantity
            if total_qty <= 0:
                holdings.pop(i)
                return holdings
            avg = (h.quantity * h.avg_price + new.quantity * new.avg_price) / total_qty
            holdings[i] = Holding(
                ticker=h.ticker,
                name=new.name or h.name,
                quantity=total_qty,
                avg_price=round(avg, 2),
                memo=new.memo or h.memo,
            )
            return holdings
    holdings.append(new)
    return holdings


def remove_holding(holdings: list[Holding], ticker: str) -> list[Holding]:
    return [h for h in holdings if h.ticker != ticker]


def serialize(holdings: list[Holding]) -> str:
    """JSON 문자열로 직렬화 (파일 다운로드용)."""
    return json.dumps([asdict(h) for h in holdings], ensure_ascii=False, indent=2)


def deserialize(text: str | bytes) -> list[Holding]:
    """JSON 문자열/바이트를 Holding 리스트로 역직렬화."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("올바른 포트폴리오 JSON이 아닙니다 (배열이어야 합니다).")
    out: list[Holding] = []
    for row in raw:
        # 알 수 없는 필드는 무시
        kwargs = {k: row[k] for k in ("ticker", "name", "quantity", "avg_price", "memo") if k in row}
        out.append(Holding(**kwargs))
    return out
