# KRX 포트폴리오 & 엘리엇 파동 예측

> 👉 **앱 바로 열기**: <https://jiyoohwang-stock.streamlit.app> *(아래 "배포하기" 1회 진행 후 동작)*

한국 주식(KRX) 보유 종목을 관리하고, 추세·모멘텀·밸류에이션·변동성 점수와 엘리엇 파동
피보나치 목표가까지 한 화면에서 보여 주는 Streamlit 웹앱.

> ⚠️ 본 도구는 결정론적 휴리스틱 기반 보조 신호이며, 투자 판단의 근거가 될 수 없습니다.

## 🚀 배포하기 (1회 · 3분)

1. <https://share.streamlit.io> 직접 접속 → **Continue with GitHub** 로그인
2. GitHub OAuth 권한 요청이 뜨면 **Authorize streamlit** 클릭
3. **Create app** → **Deploy a public app from GitHub** 선택
4. 폼 입력 후 **Deploy**:
   - Repository: `JiyooHwang/Stock`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: `jiyoohwang-stock` (README 첫 줄 링크와 맞추려면 이 슬러그 사용)
5. 1~2분 후 `https://<슬러그>.streamlit.app` 에서 앱이 뜹니다
6. 이후 `main` 에 푸시하면 **자동 재배포**

> 슬러그를 다르게 정했다면 README 첫 줄의 URL을 그것으로 수정하세요.

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속. 같은 네트워크의 친구와 공유하려면
`streamlit run app.py --server.address 0.0.0.0` 후 본인 PC IP로 접속.

## 기능

- **포트폴리오 관리**: 종목 추가·합산(가중평단)·삭제, 실시간 평가손익
- **종목 점수판**: 추세(35%) · 모멘텀(30%) · 밸류에이션(20%) · 변동성(15%) 가중 종합 신호 (매수/관망/매도)
- **백테스트**: "200일 이평선 위 + 12개월 모멘텀 > 0" 규칙으로 매수후보유 대비 성과 비교 (CAGR / Sharpe / MDD / 승률)
- **리스크 관리**: ATR(14) 기반 손절가 + 거래당 리스크 % 기반 권장 수량 산출
- **엘리엇 파동 분석**:
  - ZigZag(여러 임계값 자동 탐색)로 스윙 포인트 추출
  - 마지막 9개 피벗을 5-임펄스 + ABC 조정으로 라벨링
  - 엘리엇 3대 규칙(2파 100% 되돌림 금지 / 3파 최단 금지 / 4파 1파 영역 침범 금지) 검증
  - 피보나치 비율로 적합도 평가, 종합 점수 산출
  - 다음 파동의 피보나치 확장/되돌림 목표가 제시
- **인터랙티브 캔들 차트**(Plotly) + 파동 라벨 + 목표가 라인

## 데이터 소스

- [pykrx](https://github.com/sharebook-kr/pykrx) — KRX KOSPI/KOSDAQ 일봉 OHLCV
- 시세는 6시간 캐시(`data/cache/`), 포트폴리오 현재가는 30분 캐시
- NXT(넥스트레이드) 단독 시세는 미지원 — KRX 통합 시세를 사용합니다

## 포트폴리오 프라이버시

포트폴리오는 **본인 브라우저 세션에만** 저장됩니다 (`st.session_state`).
- ✅ 친구에게 앱 URL을 보내도 친구는 자기 종목만 보입니다 (서로 격리)
- ⚠️ 탭을 닫으면 사라집니다 → "가져오기/내보내기" 패널에서 **JSON 다운로드**로 백업
- 다시 접속할 때 그 JSON을 업로드하면 복원됩니다

## 폴더 구조

```
.
├── app.py                  Streamlit 진입점
├── .streamlit/
│   └── config.toml         테마/서버 설정
├── runtime.txt             Python 버전 핀 (Streamlit Cloud용)
├── src/
│   ├── data_loader.py      pykrx 시세 로더 + 캐시
│   ├── elliott_wave.py     ZigZag + 파동 라벨링 + 피보나치 목표가
│   ├── signals.py          추세/모멘텀/변동성/밸류에이션 점수
│   ├── backtest.py         이평선+모멘텀 룰 백테스트
│   ├── risk.py             ATR 손절가 + 포지션 사이징
│   ├── portfolio.py        보유 종목 JSON 영속화
│   └── charts.py           Plotly 차트
├── data/
│   ├── portfolio.json      (gitignore) 보유 종목
│   └── cache/              (gitignore) 시세 캐시
└── requirements.txt
```
