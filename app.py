"""한국 주식 포트폴리오 + 엘리엇 파동 예측 웹앱.

실행:
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data_loader as dl
from src.charts import candle_with_waves
from src.elliott_wave import best_analysis
from src.portfolio import (
    Holding,
    load_portfolio,
    remove_holding,
    save_portfolio,
    upsert_holding,
)

st.set_page_config(page_title="KRX 포트폴리오 & 엘리엇 파동", layout="wide")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    return dl.load_ohlcv(ticker, years=years)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_price(ticker: str) -> float | None:
    return dl.get_latest_price(ticker)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_name(ticker: str) -> str:
    return dl.get_ticker_name(ticker)


def _resolve_ticker(query: str) -> tuple[str, str] | None:
    query = query.strip()
    if not query:
        return None
    if query.isdigit() and len(query) == 6:
        return query, cached_name(query)
    matches = dl.search_ticker(query)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    pick = st.selectbox(
        "여러 종목이 검색되었습니다. 선택하세요.",
        matches,
        format_func=lambda x: f"{x[1]} ({x[0]})",
        key=f"pick_{query}",
    )
    return pick


def page_portfolio() -> None:
    st.header("내 포트폴리오")
    holdings = load_portfolio()

    with st.expander("종목 추가 / 매수", expanded=not holdings):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
        with col1:
            q = st.text_input("종목명 또는 6자리 코드", key="add_q")
        with col2:
            qty = st.number_input("수량", min_value=1, step=1, value=1, key="add_qty")
        with col3:
            price = st.number_input("평단가(원)", min_value=0.0, step=100.0, value=0.0, key="add_price")
        with col4:
            memo = st.text_input("메모(선택)", key="add_memo")

        if st.button("추가/합산", type="primary"):
            resolved = _resolve_ticker(q)
            if not resolved:
                st.error("종목을 찾을 수 없습니다.")
            elif price <= 0:
                st.error("평단가를 입력하세요.")
            else:
                ticker, name = resolved
                holdings = upsert_holding(
                    holdings,
                    Holding(ticker=ticker, name=name, quantity=int(qty), avg_price=float(price), memo=memo),
                )
                save_portfolio(holdings)
                st.success(f"{name}({ticker}) {qty}주 @ {price:,.0f}원 추가되었습니다.")
                st.rerun()

    if not holdings:
        st.info("아직 보유 종목이 없습니다. 위에서 추가하세요.")
        return

    rows = []
    total_cost = 0.0
    total_value = 0.0
    for h in holdings:
        cur = cached_price(h.ticker)
        cost = h.quantity * h.avg_price
        value = h.quantity * cur if cur else 0.0
        pl = value - cost if cur else 0.0
        pl_pct = (pl / cost * 100) if cost else 0.0
        rows.append(
            {
                "종목": f"{h.name} ({h.ticker})",
                "수량": h.quantity,
                "평단가": h.avg_price,
                "현재가": cur or 0,
                "평가금액": value,
                "매입금액": cost,
                "손익": pl,
                "손익률(%)": round(pl_pct, 2),
                "메모": h.memo,
            }
        )
        total_cost += cost
        total_value += value

    df = pd.DataFrame(rows)
    total_pl = total_value - total_cost
    total_pct = (total_pl / total_cost * 100) if total_cost else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 매입금액", f"{total_cost:,.0f}원")
    m2.metric("총 평가금액", f"{total_value:,.0f}원")
    m3.metric("평가손익", f"{total_pl:,.0f}원", f"{total_pct:.2f}%")
    m4.metric("보유 종목 수", f"{len(holdings)}개")

    st.dataframe(
        df.style.format(
            {
                "평단가": "{:,.0f}",
                "현재가": "{:,.0f}",
                "평가금액": "{:,.0f}",
                "매입금액": "{:,.0f}",
                "손익": "{:,.0f}",
                "손익률(%)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("종목 삭제")
    pick = st.selectbox(
        "삭제할 종목 선택",
        holdings,
        format_func=lambda h: f"{h.name} ({h.ticker})",
        key="del_pick",
    )
    if st.button("삭제", type="secondary"):
        holdings = remove_holding(holdings, pick.ticker)
        save_portfolio(holdings)
        st.success("삭제되었습니다.")
        st.rerun()


def page_wave() -> None:
    st.header("엘리엇 파동 분석")
    st.caption("⚠️ 휴리스틱 기반 보조 지표입니다 — 투자 판단의 절대적 근거가 될 수 없습니다.")

    holdings = load_portfolio()
    options: list[tuple[str, str]] = [(h.ticker, h.name) for h in holdings]

    col1, col2 = st.columns([3, 1])
    with col1:
        if options:
            mode = st.radio("종목 선택 방식", ["보유 종목에서", "직접 입력"], horizontal=True)
        else:
            mode = "직접 입력"
        if mode == "보유 종목에서" and options:
            pick = st.selectbox(
                "분석할 종목",
                options,
                format_func=lambda x: f"{x[1]} ({x[0]})",
            )
            ticker, name = pick
        else:
            q = st.text_input("종목명 또는 6자리 코드", value="삼성전자")
            resolved = _resolve_ticker(q) if q else None
            if not resolved:
                st.stop()
            ticker, name = resolved
    with col2:
        years = st.slider("분석 기간(년)", min_value=1, max_value=15, value=10)

    with st.spinner(f"{name} 데이터 로딩 중..."):
        df = cached_ohlcv(ticker, years)

    if df.empty:
        st.error("데이터를 가져올 수 없습니다.")
        return

    st.caption(f"기간: {df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}  ·  {len(df)} 거래일")

    with st.spinner("엘리엇 파동 검출 중..."):
        analysis, pivots = best_analysis(df)

    fig = candle_with_waves(df, pivots=pivots, analysis=analysis, title=f"{name} ({ticker})")
    st.plotly_chart(fig, use_container_width=True)

    if analysis is None:
        st.warning("뚜렷한 엘리엇 파동 패턴을 찾지 못했습니다. 기간이나 종목을 바꿔 시도해 보세요.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("판정 요약")
        st.markdown(f"- **방향**: {'상승 임펄스' if analysis.direction == 'up' else '하락 임펄스'}")
        st.markdown(f"- **검출된 파동**: {', '.join(analysis.waves.keys())}")
        st.markdown(f"- **다음 예상 파동**: `{analysis.next_wave or '미정'}`")
        st.markdown(f"- **종합 점수**: {analysis.score:.2f} / 1.00")
        for note in analysis.notes:
            st.caption(note)
        if analysis.rule_violations:
            st.error("⚠️ 엘리엇 규칙 위반:\n" + "\n".join(f"- {v}" for v in analysis.rule_violations))
        else:
            st.success("✅ 엘리엇 3대 규칙 통과")

    with c2:
        st.subheader("피보나치 적합도")
        if analysis.fib_scores:
            st.dataframe(
                pd.DataFrame(
                    [{"비율": k, "점수(0~1)": round(v, 3)} for k, v in analysis.fib_scores.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("다음 파동 목표가")
    if analysis.targets:
        last_close = float(df["Close"].iloc[-1])
        rows = []
        for label, price in analysis.targets.items():
            diff = price - last_close
            pct = diff / last_close * 100
            rows.append(
                {
                    "구분": label,
                    "목표가": f"{price:,.0f}",
                    "현재가 대비": f"{diff:+,.0f}",
                    "변동률(%)": f"{pct:+.2f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("목표가를 산출할 수 없습니다.")


def main() -> None:
    st.sidebar.title("📈 KRX 도구")
    page = st.sidebar.radio("메뉴", ["내 포트폴리오", "엘리엇 파동 분석"])
    st.sidebar.divider()
    st.sidebar.caption(
        "데이터: pykrx (KRX)\n\n"
        "포트폴리오는 `data/portfolio.json` 에 저장됩니다."
    )
    if page == "내 포트폴리오":
        page_portfolio()
    else:
        page_wave()


if __name__ == "__main__":
    main()
