"""한국 주식 포트폴리오 + 엘리엇 파동 예측 웹앱.

실행:
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_loader as dl
from src.backtest import run_backtest
from src.charts import candle_with_waves
from src.elliott_wave import best_analysis
from src.portfolio import (
    Holding,
    load_portfolio,
    remove_holding,
    save_portfolio,
    upsert_holding,
)
from src.risk import plan_risk
from src.signals import score_ticker

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


def page_scorecard() -> None:
    st.header("종목 점수판")
    st.caption(
        "추세(35%) · 모멘텀(30%) · 밸류에이션(20%) · 변동성(15%) 가중평균. "
        "0~100 종합점수, 70+ 매수 / 40~70 관망 / 40- 매도."
    )

    holdings = load_portfolio()
    extra = st.text_input("추가로 점수 매길 종목(쉼표로 구분, 코드 또는 종목명)", value="")

    targets: list[tuple[str, str]] = [(h.ticker, h.name) for h in holdings]
    for q in [s.strip() for s in extra.split(",") if s.strip()]:
        resolved = _resolve_ticker(q)
        if resolved and resolved not in targets:
            targets.append(resolved)

    if not targets:
        st.info("보유 종목이 없거나 입력이 없습니다.")
        return

    rows = []
    progress = st.progress(0.0, text="점수 계산 중...")
    for i, (ticker, name) in enumerate(targets, 1):
        df = cached_ohlcv(ticker, 3)
        if df.empty or len(df) < 220:
            rows.append({"종목": f"{name} ({ticker})", "신호": "데이터 부족"})
            progress.progress(i / len(targets))
            continue
        s = score_ticker(df, ticker)
        rows.append(
            {
                "종목": f"{name} ({ticker})",
                "추세": round(s.trend, 1),
                "모멘텀": round(s.momentum, 1),
                "변동성": round(s.volatility, 1),
                "밸류에이션": round(s.valuation, 1),
                "종합": round(s.composite, 1),
                "신호": s.signal,
                "12m수익(%)": round(s.detail.get("mom.12m%", 0), 1),
                "PER": round(s.detail.get("val.PER", 0), 1),
                "연변동성(%)": round(s.detail.get("vol.ann_vol%", 0), 1),
            }
        )
        progress.progress(i / len(targets))
    progress.empty()

    df_score = pd.DataFrame(rows).sort_values(by="종합", ascending=False, na_position="last")

    def _sig_style(val):
        if val == "매수":
            return "background-color: #d4edda; color: #155724; font-weight: bold;"
        if val == "매도":
            return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
        if val == "관망":
            return "background-color: #fff3cd; color: #856404;"
        return ""

    st.dataframe(
        df_score.style.map(_sig_style, subset=["신호"]),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "💡 종합 신호는 단일 절대 답이 아닙니다. 추세·모멘텀이 함께 양호할 때 가장 신뢰할 수 있고, "
        "밸류에이션은 성장주에서 과도하게 낮게 나올 수 있습니다."
    )


def page_backtest() -> None:
    st.header("백테스트 — 추세 + 모멘텀")
    st.caption("규칙: 종가 > 200일 이평선 AND 12개월 모멘텀 > 0 일 때 long, 아니면 현금. 시그널 lag=1, 거래비용 0.1%.")

    holdings = load_portfolio()
    options: list[tuple[str, str]] = [(h.ticker, h.name) for h in holdings]

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        if options:
            mode = st.radio("종목 선택", ["보유 종목", "직접 입력"], horizontal=True, key="bt_mode")
        else:
            mode = "직접 입력"
        if mode == "보유 종목" and options:
            pick = st.selectbox("종목", options, format_func=lambda x: f"{x[1]} ({x[0]})", key="bt_pick")
            ticker, name = pick
        else:
            q = st.text_input("종목명 또는 6자리 코드", value="삼성전자", key="bt_q")
            resolved = _resolve_ticker(q) if q else None
            if not resolved:
                st.stop()
            ticker, name = resolved
    with c2:
        years = st.slider("기간(년)", 2, 15, 10, key="bt_years")
    with c3:
        ma_window = st.number_input("이평선", min_value=20, max_value=300, value=200, step=10, key="bt_ma")
    with c4:
        mom_window = st.number_input("모멘텀(거래일)", min_value=20, max_value=500, value=252, step=20, key="bt_mom")

    df = cached_ohlcv(ticker, years)
    if df.empty:
        st.error("데이터 없음")
        return

    try:
        res = run_backtest(df, ma_window=int(ma_window), momentum_window=int(mom_window))
    except ValueError as e:
        st.error(str(e))
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity.values, name="전략", line=dict(color="#d24f45", width=2)))
    fig.add_trace(go.Scatter(x=res.benchmark.index, y=res.benchmark.values, name="매수후보유", line=dict(color="#1f77b4", width=2)))
    fig.update_layout(
        title=f"{name} ({ticker}) — 자산 곡선",
        height=450, template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="누적 배수(시작=1)",
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(4)
    metrics = res.metrics
    for i, key in enumerate(["전략 CAGR(%)", "전략 MDD(%)", "전략 Sharpe", "승률(%)"]):
        if key in metrics:
            cols[i].metric(key, f"{metrics[key]:.2f}")

    st.subheader("성과 비교")
    rows = [
        {"지표": k.replace("전략 ", ""),
         "전략": metrics.get(f"전략 {k.replace('전략 ', '')}"),
         "매수후보유": metrics.get(f"매수후보유 {k.replace('전략 ', '')}")}
        for k in ["전략 총수익률(%)", "전략 CAGR(%)", "전략 변동성(%)", "전략 Sharpe", "전략 MDD(%)"]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if not res.trades.empty:
        st.subheader(f"거래 기록 ({len(res.trades)}건)")
        st.dataframe(
            res.trades.style.format({"진입가": "{:,.0f}", "청산가": "{:,.0f}", "수익률(%)": "{:+.2f}"}),
            hide_index=True, use_container_width=True,
        )


def page_risk() -> None:
    st.header("리스크 관리 — ATR 손절가 & 포지션 사이징")
    st.caption(
        "ATR(평균진폭) 기반 손절가 = 진입가 − ATR배수 × ATR. "
        "포지션 크기는 한 트레이드에서 잃을 금액(자본 × 리스크%)을 손절폭으로 나눠 산출."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        capital = st.number_input("총 자본(원)", min_value=0, value=10_000_000, step=500_000)
    with c2:
        risk_pct = st.number_input("거래당 리스크(%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    with c3:
        atr_mult = st.number_input("ATR 배수", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    holdings = load_portfolio()
    rows = []
    if holdings:
        st.subheader("보유 종목 손절 플랜")
        for h in holdings:
            df = cached_ohlcv(h.ticker, 1)
            cur = cached_price(h.ticker)
            if df.empty or cur is None:
                continue
            plan = plan_risk(df, entry_price=h.avg_price, capital=capital, risk_pct=risk_pct, atr_mult=atr_mult)
            cur_stop = plan_risk(df, entry_price=cur, capital=capital, risk_pct=risk_pct, atr_mult=atr_mult)
            rows.append(
                {
                    "종목": f"{h.name} ({h.ticker})",
                    "보유수량": h.quantity,
                    "평단가": h.avg_price,
                    "현재가": cur,
                    "ATR(14)": round(plan.atr, 1),
                    "평단기준 손절가": round(plan.stop_loss),
                    "평단 손절률(%)": round(plan.stop_loss_pct, 2),
                    "현재가기준 손절가": round(cur_stop.stop_loss),
                    "권장수량": cur_stop.suggested_qty,
                }
            )
        if rows:
            st.dataframe(
                pd.DataFrame(rows).style.format(
                    {
                        "평단가": "{:,.0f}", "현재가": "{:,.0f}",
                        "평단기준 손절가": "{:,.0f}", "현재가기준 손절가": "{:,.0f}",
                    }
                ),
                hide_index=True, use_container_width=True,
            )

    st.divider()
    st.subheader("새 진입 시뮬레이션")
    cc1, cc2 = st.columns([3, 1])
    with cc1:
        q = st.text_input("종목명 또는 코드", value="삼성전자", key="risk_q")
    with cc2:
        entry = st.number_input("예상 진입가(원)", min_value=0.0, value=0.0, step=100.0, key="risk_entry")

    resolved = _resolve_ticker(q) if q else None
    if resolved:
        ticker, name = resolved
        df = cached_ohlcv(ticker, 1)
        if df.empty:
            st.error("데이터 없음")
            return
        if entry <= 0:
            entry = float(df["Close"].iloc[-1])
            st.caption(f"진입가 미입력 → 최신 종가 {entry:,.0f}원 사용")
        plan = plan_risk(df, entry_price=entry, capital=capital, risk_pct=risk_pct, atr_mult=atr_mult)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ATR(14)", f"{plan.atr:,.1f}")
        m2.metric("손절가", f"{plan.stop_loss:,.0f}원", f"{plan.stop_loss_pct:.2f}%")
        m3.metric("권장 수량", f"{plan.suggested_qty:,}주")
        m4.metric("최대 손실액", f"{plan.risk_amount:,.0f}원")
        st.caption(
            f"{name}({ticker}) — 진입가 {entry:,.0f} / 손절가 {plan.stop_loss:,.0f} 도달 시 "
            f"약 {plan.suggested_qty * (entry - plan.stop_loss):,.0f}원 손실 (자본의 {risk_pct}%)."
        )


def main() -> None:
    st.sidebar.title("📈 KRX 도구")
    page = st.sidebar.radio(
        "메뉴",
        ["내 포트폴리오", "종목 점수판", "백테스트", "리스크 관리", "엘리엇 파동 분석"],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "데이터: pykrx (KRX)\n\n"
        "포트폴리오는 `data/portfolio.json` 에 저장됩니다."
    )
    {
        "내 포트폴리오": page_portfolio,
        "종목 점수판": page_scorecard,
        "백테스트": page_backtest,
        "리스크 관리": page_risk,
        "엘리엇 파동 분석": page_wave,
    }[page]()


if __name__ == "__main__":
    main()
