"""한국 주식 포트폴리오 + 엘리엇 파동 예측 웹앱.

실행:
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_loader as dl
from src import dart, naver_extras
from src.backtest import run_backtest, run_walkforward
from src.benchmark import INDICES, detect_market, load_index_ohlcv, perf_summary, relative_performance
from src.charts import candle_with_waves
from src.elliott_wave import best_analysis
from src.portfolio import (
    Holding,
    deserialize,
    remove_holding,
    serialize,
    upsert_holding,
)
from src.portfolio_optim import (
    daily_returns,
    equal_weight,
    portfolio_metrics,
    risk_parity_weights,
)
from src.risk import plan_risk
from src.signals import explain_score, score_ticker

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


def _holdings() -> list[Holding]:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []
    return st.session_state.portfolio


def _set_holdings(h: list[Holding]) -> None:
    st.session_state.portfolio = h


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
    st.caption(
        "🔒 포트폴리오는 **본인 브라우저 세션에만** 저장되며 다른 방문자에게 공유되지 않습니다. "
        "탭을 닫으면 사라지므로 영구 보관이 필요하면 아래에서 **다운로드** 받아 두세요."
    )

    with st.expander("📂 가져오기 / 내보내기", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            uploaded = st.file_uploader(
                "포트폴리오 JSON 파일 가져오기", type=["json"], key="upload_pf",
                help="이전에 다운로드한 portfolio.json 을 올리세요. 기존 세션 데이터는 덮어씁니다.",
            )
            if uploaded is not None:
                try:
                    new_holdings = deserialize(uploaded.read())
                    _set_holdings(new_holdings)
                    st.success(f"{len(new_holdings)}개 종목을 불러왔습니다.")
                except Exception as e:
                    st.error(f"파일을 읽지 못했습니다: {e}")
        with c2:
            current = _holdings()
            if current:
                st.download_button(
                    "💾 현재 포트폴리오 다운로드 (JSON)",
                    data=serialize(current),
                    file_name="portfolio.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.button("💾 다운로드 (보유 종목 없음)", disabled=True, use_container_width=True)
            if st.button("🗑️ 세션 초기화", use_container_width=True):
                _set_holdings([])
                st.rerun()

    holdings = _holdings()

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
                _set_holdings(holdings)
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
        _set_holdings(holdings)
        st.success("삭제되었습니다.")
        st.rerun()


def page_wave() -> None:
    st.header("엘리엇 파동 분석")
    st.caption("⚠️ 휴리스틱 기반 보조 지표입니다 — 투자 판단의 절대적 근거가 될 수 없습니다.")

    holdings = _holdings()
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
    st.markdown(
        "각 종목을 **4가지 관점**에서 0\\~100점으로 평가하고 가중평균해서 종합점수를 냅니다.  \n"
        "종합 **70점 이상은 🟢 매수**, **40\\~70점은 🟡 관망**, **40점 미만은 🔴 매도** 신호입니다."
    )

    with st.expander("📖 점수판 사용법 — 처음이라면 꼭 읽어보세요", expanded=False):
        st.markdown(
            """
**1. 추세 (35% · 가장 중요)**
- 주가가 **200일 평균선 위**에 있나요? 위면 ↑ 점수, 아래면 ↓ 점수
- **50일선 > 200일선** (골든크로스) 이면 추가 가산점
- 200일선과 너무 멀어지면 과열, 너무 가까우면 추세 약화
- *왜 중요?* 학계에서 가장 잘 검증된 신호. 추세를 거스르지 마세요.

**2. 모멘텀 (30%)**
- **최근 12개월 수익률**이 핵심. 6개월·3개월 수익률도 본다
- 많이 오른 종목은 단기간 더 오를 가능성이 통계적으로 높음 (관성효과, Jegadeesh-Titman)
- 12개월 +30% 이상이면 강한 상승, -20% 이하면 큰 하락

**3. 밸류에이션 + 퀄리티 (20%)**
- **밸류 (60%)**: PER이 낮을수록 싸다 (8 이하 만점, 30 이상 0점). PBR·배당수익률도 본다.
- **선행 PER 우선**: 네이버에서 컨센서스 추정 PER이 있으면 그것을 사용 (현재 PER보다 미래 가치 반영)
- **퀄리티 (40%)** — 네이버 데이터가 있을 때만 가산:
  - **ROE**: 자본 대비 얼마나 이익을 내는지. 15% 이상이면 우량
  - **부채비율**: 100% 이하면 안정, 200% 초과면 위험
  - **영업이익률**: 본업으로 얼마나 남기는지. 15%↑ 우수, 적자면 감점
- 데이터: **네이버 금융 우선 → 차단/구조변경 시 pykrx 폴백 (1일 캐시)**
- ⚠️ 적자기업·성장주는 PER이 의미 없거나 비정상적으로 보일 수 있음

**4. 변동성 (15%) — "얼마나 출렁이나"**
- 연환산 변동성이 **20% 이하면 안정적**, 60% 이상이면 위험
- 변동성이 크면 수익도 크지만 손실도 클 수 있어요
- 큰 손실을 피하려면 변동성이 너무 높은 종목은 피하는 게 안전

---

**🔍 신호의 한계 — 꼭 알아두세요**
- 단일 점수가 매수/매도를 결정하는 게 아닙니다. **각 항목 점수 분포를 보세요.**
- "종합 75점이지만 변동성 0점" 이면 진입 타이밍을 짧게 잡거나 비중을 줄이는 게 안전
- "종합 60점이지만 추세 100점" 이면 추세를 따라가도 됨 (모멘텀이 약하지만 추세는 강함)
- 재무 데이터가 없으면 밸류에이션은 50점(중립)으로 처리됩니다
            """
        )

    holdings = _holdings()
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
    score_cache: dict[str, object] = {}
    progress = st.progress(0.0, text="점수 계산 중...")
    for i, (ticker, name) in enumerate(targets, 1):
        df = cached_ohlcv(ticker, 3)
        if df.empty or len(df) < 220:
            rows.append({"종목": f"{name} ({ticker})", "신호": "데이터 부족"})
            progress.progress(i / len(targets))
            continue
        s = score_ticker(df, ticker)
        score_cache[ticker] = s
        cur_p = cached_price(ticker)
        consensus = naver_extras.fetch_consensus(ticker, current_price=cur_p)
        rows.append(
            {
                "종목": f"{name} ({ticker})",
                "_ticker": ticker,
                "추세": round(s.trend, 1),
                "모멘텀": round(s.momentum, 1),
                "밸류에이션": round(s.valuation, 1),
                "변동성": round(s.volatility, 1),
                "종합": round(s.composite, 1),
                "신호": s.signal,
                "12개월 수익(%)": round(s.detail.get("mom.12m%", 0), 1),
                "PER": round(s.detail.get("val.PER", 0), 1),
                "연 변동성(%)": round(s.detail.get("vol.ann_vol%", 0), 1),
                "목표가": round(consensus.target_price, 0) if consensus and consensus.target_price else None,
                "상승여력(%)": round(consensus.upside_pct, 1) if consensus and consensus.upside_pct is not None else None,
            }
        )
        progress.progress(i / len(targets))
    progress.empty()

    df_score = pd.DataFrame(rows).sort_values(by="종합", ascending=False, na_position="last")
    display_df = df_score.drop(columns=[c for c in ["_ticker"] if c in df_score.columns])

    def _sig_style(val):
        if val == "매수":
            return "background-color: #d4edda; color: #155724; font-weight: bold;"
        if val == "매도":
            return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
        if val == "관망":
            return "background-color: #fff3cd; color: #856404;"
        return ""

    fmt = {
        "추세": "{:.1f}", "모멘텀": "{:.1f}", "밸류에이션": "{:.1f}",
        "변동성": "{:.1f}", "종합": "{:.1f}",
        "12개월 수익(%)": "{:+.1f}", "PER": "{:.1f}", "연 변동성(%)": "{:.1f}",
        "목표가": "{:,.0f}", "상승여력(%)": "{:+.1f}",
    }
    st.dataframe(
        display_df.style.map(_sig_style, subset=["신호"]).format(fmt),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("🔎 종목별 자세히 보기")
    for _, row in df_score.iterrows():
        ticker = row.get("_ticker")
        if not ticker or ticker not in score_cache:
            continue
        s = score_cache[ticker]
        signal_emoji = {"매수": "🟢", "관망": "🟡", "매도": "🔴"}.get(s.signal, "⚪")
        with st.expander(
            f"{signal_emoji} {row['종목']}  —  종합 {s.composite:.1f}점 · {s.signal}",
            expanded=False,
        ):
            explain = explain_score(s)
            cols = st.columns(4)
            cols[0].metric("추세", f"{s.trend:.0f}점")
            cols[1].metric("모멘텀", f"{s.momentum:.0f}점")
            cols[2].metric("밸류에이션", f"{s.valuation:.0f}점")
            cols[3].metric("변동성", f"{s.volatility:.0f}점")
            st.markdown(f"**추세**: {explain['추세']}")
            st.markdown(f"**모멘텀**: {explain['모멘텀']}")
            st.markdown(f"**밸류에이션**: {explain['밸류에이션']}")
            if "퀄리티" in explain:
                st.markdown(f"**퀄리티**: {explain['퀄리티']}")
            st.markdown(f"**변동성**: {explain['변동성']}")
            st.info(explain["종합"])

    st.caption(
        "💡 종합 신호는 단일 절대 답이 아닙니다. 추세·모멘텀이 함께 양호할 때 가장 신뢰할 수 있고, "
        "밸류에이션은 성장주에서 과도하게 낮게 나올 수 있습니다. "
        "재무 데이터가 없는 종목은 밸류에이션 50점(중립)으로 처리됩니다."
    )


def page_backtest() -> None:
    st.header("백테스트 — 추세 + 모멘텀")
    st.caption("규칙: 종가 > 200일 이평선 AND 12개월 모멘텀 > 0 일 때 long, 아니면 현금. 시그널 lag=1, 거래비용 0.1%.")

    holdings = _holdings()
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

    st.divider()
    st.subheader("⏱️ 시기별 분할 백테스트 (워크포워드)")
    st.caption(
        "전체 기간을 여러 시간대로 쪼개서 룰의 일관성을 본다. "
        "특정 시기에만 작동하는 룰은 과적합 의심 신호. "
        "*우리 룰은 학습 파라미터가 없으므로 엄밀한 워크포워드는 아니지만, 시기별 안정성 진단에 유용.*"
    )
    cwf1, cwf2 = st.columns(2)
    with cwf1:
        wf_train = st.number_input("학습기간(년) — 룰 적용 컨텍스트", min_value=1, max_value=8, value=3, key="wf_train")
    with cwf2:
        wf_test = st.number_input("검증기간(년) — 한 슬라이스", min_value=1, max_value=5, value=1, key="wf_test")

    wf_df = run_walkforward(
        df,
        train_years=int(wf_train),
        test_years=int(wf_test),
        ma_window=int(ma_window),
        momentum_window=int(mom_window),
    )
    if wf_df.empty:
        st.info("백테스트에 충분한 데이터가 없습니다 (기간을 줄여 보세요).")
    else:
        st.dataframe(wf_df, hide_index=True, use_container_width=True)
        wins = (wf_df["전략(%)"] > wf_df["매수후보유(%)"]).sum()
        st.caption(
            f"전체 {len(wf_df)}개 시기 중 **{wins}개**에서 전략이 매수후보유를 이겼습니다 "
            f"(평균 초과수익 {wf_df['초과수익(%)'].mean():+.2f}%)."
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

    holdings = _holdings()
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


def page_market() -> None:
    st.header("시장 비교 & 포트폴리오 자산배분")
    st.caption(
        "보유 종목을 KOSPI/KOSDAQ 등 지수와 비교하고, 변동성 기반 권장 비중(리스크 패리티)을 계산합니다."
    )

    holdings = _holdings()

    tab1, tab2 = st.tabs(["📈 종목 vs 지수", "⚖️ 리스크 패리티 비중"])

    with tab1:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            options = [(h.ticker, h.name) for h in holdings]
            if options:
                pick = st.selectbox("종목", options, format_func=lambda x: f"{x[1]} ({x[0]})", key="mk_pick")
                ticker, name = pick
            else:
                q = st.text_input("종목명 또는 6자리 코드", value="삼성전자", key="mk_q")
                resolved = _resolve_ticker(q) if q else None
                if not resolved:
                    st.stop()
                ticker, name = resolved
        with col2:
            years = st.slider("기간(년)", 1, 10, 3, key="mk_years")
        with col3:
            market_hint = detect_market(ticker)
            default_idx = 0 if market_hint == "KOSPI" else 1 if market_hint == "KOSDAQ" else 0
            index_label = st.selectbox("벤치마크", list(INDICES.keys()), index=default_idx, key="mk_idx")

        df_stock = cached_ohlcv(ticker, years)
        df_index = load_index_ohlcv(INDICES[index_label], years=years)
        if df_stock.empty or df_index.empty:
            st.error("데이터 부족")
            return

        rel = relative_performance(df_stock, df_index)
        if rel.empty:
            st.error("공통 기간이 없습니다.")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rel.index, y=rel["Stock"], name=name, line=dict(color="#d24f45", width=2)))
        fig.add_trace(go.Scatter(x=rel.index, y=rel["Index"], name=index_label, line=dict(color="#1f77b4", width=2)))
        fig.update_layout(
            title=f"{name} vs {index_label} (시작 = 100)",
            height=450, template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title="누적 (시작=100)",
        )
        st.plotly_chart(fig, use_container_width=True)

        s = perf_summary(df_stock, df_index)
        if s:
            m1, m2, m3 = st.columns(3)
            m1.metric("종목 수익률", f"{s['종목 수익률(%)']:+.2f}%")
            m2.metric(f"{index_label} 수익률", f"{s['지수 수익률(%)']:+.2f}%")
            m3.metric("초과수익", f"{s['초과수익(%)']:+.2f}%")
            if s["초과수익(%)"] > 0:
                st.success(f"📈 {name}이(가) {index_label}을(를) {s['초과수익(%)']:.1f}%p 앞섰습니다.")
            else:
                st.warning(f"📉 {name}이(가) {index_label}을(를) {abs(s['초과수익(%)']):.1f}%p 뒤졌습니다.")

    with tab2:
        st.markdown(
            "**리스크 패리티**: 변동성이 큰 종목엔 작은 비중을, 안정적인 종목엔 큰 비중을 줘서 "
            "전체 위험을 분산시키는 방법입니다. (월가 헤지펀드의 표준 전략 중 하나)"
        )
        if len(holdings) < 2:
            st.info("리스크 패리티는 종목이 2개 이상일 때 의미가 있습니다. 보유 종목을 추가하세요.")
            return

        years = st.slider("계산 기간(년)", 1, 5, 2, key="rp_years")
        with st.spinner("종목별 시계열 로딩..."):
            price_dict = {}
            for h in holdings:
                d = cached_ohlcv(h.ticker, years)
                if not d.empty:
                    price_dict[h.ticker] = d["Close"]

        if len(price_dict) < 2:
            st.error("계산 가능한 종목이 부족합니다.")
            return

        prices = pd.DataFrame(price_dict).dropna()
        rets = daily_returns(prices)
        rp_w = risk_parity_weights(rets)
        eq_w = equal_weight(list(rp_w.index))
        cur_value: dict[str, float] = {}
        total_val = 0.0
        for h in holdings:
            if h.ticker in rp_w.index:
                cur = cached_price(h.ticker) or h.avg_price
                v = h.quantity * cur
                cur_value[h.ticker] = v
                total_val += v
        cur_w = pd.Series({t: v / total_val for t, v in cur_value.items()}) if total_val else pd.Series()

        name_map = {h.ticker: h.name for h in holdings}
        rows = []
        for t in rp_w.index:
            ann_vol = rets[t].std() * (252 ** 0.5) * 100
            rows.append(
                {
                    "종목": f"{name_map.get(t, t)} ({t})",
                    "연 변동성(%)": round(ann_vol, 2),
                    "현재 비중(%)": round(cur_w.get(t, 0) * 100, 2) if not cur_w.empty else None,
                    "동일가중(%)": round(eq_w[t] * 100, 2),
                    "리스크패리티(%)": round(rp_w[t] * 100, 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.subheader("백테스트 비교 (지난 N년)")
        rp_metrics = portfolio_metrics(rets, rp_w)
        eq_metrics = portfolio_metrics(rets, eq_w)
        cur_metrics = portfolio_metrics(rets, cur_w) if not cur_w.empty else {}
        compare_rows = []
        for k in ("총수익률(%)", "CAGR(%)", "변동성(%)", "Sharpe", "MDD(%)"):
            compare_rows.append(
                {
                    "지표": k,
                    "현재 비중": cur_metrics.get(k),
                    "동일가중": eq_metrics.get(k),
                    "리스크패리티": rp_metrics.get(k),
                }
            )
        st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)
        st.caption(
            "리스크패리티 비중은 권장값일 뿐입니다. "
            "변동성 기반이라 적은 종목·짧은 기간엔 안정성이 떨어질 수 있어요."
        )


def page_filings() -> None:
    st.header("공시 & 뉴스")
    st.caption(
        "DART 공시(공식)와 네이버 뉴스 헤드라인. "
        "DART는 [opendart.fss.or.kr](https://opendart.fss.or.kr)에서 무료 API 키 발급 후 사이드바에 입력하세요."
    )

    holdings = _holdings()
    options = [(h.ticker, h.name) for h in holdings]

    col1, col2 = st.columns([4, 1])
    with col1:
        if options:
            mode = st.radio("종목 선택", ["보유 종목", "직접 입력"], horizontal=True, key="fl_mode")
        else:
            mode = "직접 입력"
        if mode == "보유 종목" and options:
            pick = st.selectbox("종목", options, format_func=lambda x: f"{x[1]} ({x[0]})", key="fl_pick")
            ticker, name = pick
        else:
            q = st.text_input("종목명 또는 6자리 코드", value="삼성전자", key="fl_q")
            resolved = _resolve_ticker(q) if q else None
            if not resolved:
                st.stop()
            ticker, name = resolved
    with col2:
        days = st.slider("공시 조회기간(일)", 7, 180, 60, key="fl_days")

    api_key = st.session_state.get("dart_api_key") or None

    tab_dart, tab_news, tab_target = st.tabs(["📑 DART 공시", "📰 뉴스", "🎯 컨센서스 목표가"])

    with tab_dart:
        if not api_key and not dart._get_key():
            st.warning(
                "DART API 키가 설정되지 않았습니다. "
                "사이드바 'DART 키' 입력란에 키를 넣거나 Streamlit Cloud Secrets에 "
                "`OPEN_DART_KEY` 를 설정하세요."
            )
        else:
            with st.spinner(f"DART 공시 조회 중 ({name})..."):
                df_filings = dart.list_disclosures(ticker, days=int(days), api_key=api_key)
            if df_filings.empty:
                st.info("최근 공시가 없거나 조회에 실패했습니다.")
                with st.expander("🔬 진단 정보 보기 (어디서 막혔는지 확인)", expanded=True):
                    with st.spinner("진단 중..."):
                        diag = dart.diagnose(ticker, api_key=api_key)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**키 발견**: {'✅' if diag['key_found'] else '❌'}")
                        st.markdown(f"**키 출처**: `{diag['key_source']}`")
                        if diag["key_preview"]:
                            st.markdown(f"**키 미리보기**: `{diag['key_preview']}`")
                        st.markdown(
                            f"**corp_map 크기**: "
                            f"{diag['corp_map_size'] if diag['corp_map_size'] is not None else '-'}"
                        )
                    with c2:
                        st.markdown(f"**corp_code 매핑**: `{diag['corp_code'] or '없음'}`")
                        st.markdown(f"**API status**: `{diag['api_status'] or '-'}`")
                        st.markdown(f"**API message**: {diag['api_message'] or '-'}")
                        st.markdown(f"**조회 건수**: {diag['filings_count'] if diag['filings_count'] is not None else '-'}")
                    if "corp_xml_http" in diag:
                        st.markdown(
                            f"**corpCode.xml 응답**: HTTP `{diag['corp_xml_http']}` · "
                            f"`{diag.get('corp_xml_content_type', '')}` · "
                            f"`{diag.get('corp_xml_size_bytes', 0):,}` bytes"
                        )
                    if diag["error"]:
                        st.error(diag["error"])
                        if "네트워크" in str(diag["error"]) or "timed out" in str(diag["error"]).lower():
                            st.info(
                                "💡 **우회 방법**: 사이드바 → "
                                "`📦 DART corp_map 업로드` 에서 로컬에서 받은 "
                                "`CorpCode.zip` 또는 `corp_map.json` 을 업로드하세요. "
                                "이후 공시 조회는 네트워크 없이 캐시로 동작합니다."
                            )
                    elif diag.get("api_status") == "013":
                        st.warning("DART status 013 = 조회된 데이터가 없음. 종목/기간을 바꿔 보세요.")
                    elif diag.get("api_status") and diag["api_status"] != "000":
                        st.warning(
                            f"DART status {diag['api_status']}: {diag['api_message']}. "
                            "키 오류(010/011), 한도초과(020) 등이 의심됩니다."
                        )
            else:
                df_show = df_filings.copy()
                df_show["접수일"] = df_show["접수일"].dt.strftime("%Y-%m-%d")
                st.dataframe(
                    df_show,
                    column_config={
                        "URL": st.column_config.LinkColumn("바로가기", display_text="🔗 열기"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(f"총 {len(df_filings)}건 — 최근 {days}일 기준")

    with tab_news:
        with st.spinner("뉴스 헤드라인 로딩..."):
            df_news = naver_extras.fetch_news(ticker, max_items=20)
        if df_news.empty:
            st.info("뉴스를 가져오지 못했습니다 (네이버 차단 또는 데이터 없음).")
        else:
            df_show = df_news.copy()
            st.dataframe(
                df_show,
                column_config={
                    "URL": st.column_config.LinkColumn("기사", display_text="🔗 읽기"),
                },
                hide_index=True,
                use_container_width=True,
            )

    with tab_target:
        cur = cached_price(ticker)
        c = naver_extras.fetch_consensus(ticker, current_price=cur)
        if c is None:
            st.info("이 종목의 컨센서스 정보를 찾지 못했습니다 (소형주는 누락될 수 있습니다).")
        else:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("현재가", f"{cur:,.0f}원" if cur else "-")
            mc2.metric("목표주가", f"{c.target_price:,.0f}원" if c.target_price else "-")
            mc3.metric(
                "상승여력",
                f"{c.upside_pct:+.2f}%" if c.upside_pct is not None else "-",
            )
            mc4.metric("투자의견", f"{c.opinion or '-'} ({c.opinion_score:.2f})" if c.opinion_score else "-")
            st.caption("출처: 네이버 금융 컨센서스 (애널리스트 추정치 평균)")


def main() -> None:
    st.sidebar.title("📈 KRX 도구")
    page = st.sidebar.radio(
        "메뉴",
        [
            "내 포트폴리오",
            "종목 점수판",
            "시장 비교",
            "백테스트",
            "리스크 관리",
            "공시 & 뉴스",
            "엘리엇 파동 분석",
        ],
    )
    st.sidebar.divider()
    with st.sidebar.expander("⚙️ DART API 키 (선택)"):
        st.markdown(
            "[opendart.fss.or.kr](https://opendart.fss.or.kr) 에서 무료 가입 후 발급. "
            "설정하면 공시 페이지가 활성화됩니다. "
            "키는 본인 세션에만 저장되고 서버에 영구 저장되지 않습니다."
        )
        key_in = st.text_input("DART 인증키", value=st.session_state.get("dart_api_key", ""), type="password")
        if key_in:
            st.session_state.dart_api_key = key_in

    with st.sidebar.expander("📦 DART corp_map 업로드 (네트워크 차단 시)"):
        status = dart.corp_map_status()
        if status["size"]:
            st.success(
                f"✅ 캐시됨: {status['size']:,}개 매핑 "
                f"(업데이트 {status['age_hours']:.1f}시간 전)"
            )
        else:
            st.warning("❌ corp_map 캐시 없음 — 공시 조회 실패 시 아래에서 업로드하세요.")
        st.caption(
            "Streamlit Cloud 등에서 `opendart.fss.or.kr` 에 직접 연결이 막힐 때, "
            "**로컬 PC** 에서 한 번 받아 업로드하면 이후 모든 조회가 캐시로 처리됩니다.\n\n"
            "**받는 법 (택1)**:\n"
            "1. 브라우저로 `https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key=본인키` 열어서 다운로드 → "
            "받아진 `CorpCode.zip` 그대로 업로드\n"
            "2. 저장소의 `tools/build_corp_map.py` 실행 → 생성된 `corp_map.json` 업로드"
        )
        up = st.file_uploader(
            "CorpCode.zip / corp_map.json / corpCode (확장자 무관)",
            type=None,
            key="corp_map_upload",
            help="브라우저가 확장자 없이 'corpCode' 로 저장한 파일도 그대로 올리면 됩니다.",
        )
        if up is not None:
            try:
                n, kind = dart.install_corp_map_from_bytes(up.read())
                st.success(f"✅ {kind} 업로드 완료: {n:,}개 매핑 등록")
                st.rerun()
            except Exception as e:
                st.error(f"업로드 실패: {e}")
    st.sidebar.divider()
    st.sidebar.caption(
        "데이터: 네이버 금융 / DART / pykrx (KRX)\n\n"
        "🔒 포트폴리오는 본인 브라우저 세션에만 저장됩니다.\n"
        "탭을 닫으면 사라지므로 \"가져오기/내보내기\"로 백업하세요."
    )
    {
        "내 포트폴리오": page_portfolio,
        "종목 점수판": page_scorecard,
        "시장 비교": page_market,
        "백테스트": page_backtest,
        "리스크 관리": page_risk,
        "공시 & 뉴스": page_filings,
        "엘리엇 파동 분석": page_wave,
    }[page]()


if __name__ == "__main__":
    main()
