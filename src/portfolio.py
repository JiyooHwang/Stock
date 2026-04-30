"""보유 주식(포트폴리오) 관리 — JSON 파일 기반."""
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
