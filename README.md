# KRX 포트폴리오 & 엘리엇 파동 예측

한국 주식(KRX) 보유 종목을 관리하고, 지난 N년치 일봉 데이터로 **엘리엇 파동**을 검출하여
다음 파동의 **피보나치 목표가**를 계산해 주는 Streamlit 웹앱.

> ⚠️ 본 도구는 결정론적 휴리스틱 기반 보조 신호이며, 투자 판단의 근거가 될 수 없습니다.

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

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

## 친구와 공유

1. **간단**: 같은 네트워크라면 `streamlit run app.py --server.address 0.0.0.0` 후 본인 PC IP로 접속
2. **추천**: 이 저장소를 GitHub에 두고 [Streamlit Community Cloud](https://streamlit.io/cloud)에 무료 배포
   - 단, 무료 플랜은 포트폴리오가 호스트와 함께 휘발됩니다 — 친구와 데이터 공유가 필요하면
     `data/portfolio.json` 을 GitHub에 올리거나 별도 DB(Supabase 등)로 마이그레이션 필요

## 데이터 소스

- [pykrx](https://github.com/sharebook-kr/pykrx) — KRX KOSPI/KOSDAQ 일봉 OHLCV
- 시세는 6시간 캐시(`data/cache/`), 포트폴리오 현재가는 30분 캐시
- NXT(넥스트레이드) 단독 시세는 미지원 — KRX 통합 시세를 사용합니다

## 폴더 구조

```
.
├── app.py                  Streamlit 진입점
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
