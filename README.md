# Coinview

Binance 현물·무기한 선물 데이터를 기반으로 코인을 분석하는 Python/Vercel 애플리케이션입니다. 백엔드 API와 반응형 HTML·CSS·JavaScript UI가 `api/index.py`에 통합되어 있습니다.

## 주요 기능

- 한글 코인명, 별칭, `BTC`, `ETHUSDT` 형식 자동 해석 및 자동완성
- Binance OHLCV·24시간 시세·오픈인터레스트 기반 24시간 시장 분석
- RSI, MACD, 이동평균, 볼린저밴드, ATR, ADX, OBV 등 기술지표
- 캔들 패턴과 삼각수렴·쐐기·헤드앤숄더·이중천장/바닥 등 차트 패턴 오버레이
- 복합 점수, 상승·하락 가능성, 신호 신뢰도와 데이터 불확실성 표시
- 레버리지 권고, 청산 위험, 분할 진입, 손절·목표 가격 및 시나리오 분석
- 해킹·규제·프로토콜 장애·유동성 뉴스 이벤트 위험 점수
- 스마트컨트랙트·DeFi·레이어2·밈·AI 등 유사 코인 그룹 상대 모멘텀
- 주요 코인·24시간 상승/하락·거래량 스크리너 및 공포·탐욕 지수
- 브라우저 로컬 저장소 기반 USDT 목표가·24시간 급등락 알림
- 모바일 사이드바와 반응형 카드·차트 UI

> 모든 확률·목표가·레버리지 값은 관측 데이터 기반 참고 지표이며 투자 수익이나 방향을 보장하지 않습니다.

## 데이터 기준

| 항목 | 소스 |
|---|---|
| OHLCV·24시간 시세 | Binance 공개 REST API |
| 무기한 선물 오픈인터레스트 | Binance Futures 공개 REST API |
| 시장 심리 | Alternative.me Fear & Greed Index |
| 뉴스 | Google News RSS |
| 차트 | Lightweight Charts 4.1.3 |

API 키 없이 핵심 기능을 사용할 수 있습니다. 예측 학습 로그는 기본적으로 꺼져 있으며, 명시적으로 `COINVIEW_LEARNING_ENABLED=1`을 설정한 경우에만 기록됩니다.

## API

| 경로 | 설명 |
|---|---|
| `GET /` | Coinview UI |
| `GET /api/coin?ticker=BTC&period=1mo` | 코인 상세 분석 |
| `GET /api/resolve?q=비트코인` | 코인 심볼 해석 |
| `GET /api/suggestions?q=eth` | 자동완성 |
| `GET /api/crypto/overview` | 주요 코인·등락·거래량·심리 |
| `GET /api/price?ticker=BTCUSDT` | 24시간 현재가 폴링 |
| `GET /api/peer-outlook?ticker=ETHUSDT` | 유사 코인 상대 전망 |
| `GET /api/alert/quote?symbols=BTCUSDT,ETHUSDT` | 알림용 일괄 시세 |
| `GET /api/cron` | 코인 시장 캐시 예열 |

`/api/stock`은 기존 배포 URL 호환을 위해 `/api/coin`의 별칭으로만 남아 있습니다.

## 로컬 실행

```powershell
python -m pip install -r requirements.txt
python dev_server.py
```

브라우저에서 [http://localhost:3000](http://localhost:3000)을 엽니다.

## 테스트

```powershell
python -m pytest -q
python -m py_compile api/index.py market_briefing/pattern_engine.py
```

테스트는 코인 심볼 해석·자동완성·가격 정밀도·예측 계약·라우팅·HTML/JavaScript 핵심 계약을 포함합니다. 실데이터 스모크 테스트는 Binance 네트워크가 가능한 환경에서 별도로 실행합니다.

## 프로젝트 구조

```text
Coinview/
├─ api/
│  └─ index.py
├─ market_briefing/
│  ├─ pattern_engine.py
│  └─ ...
├─ tests/
├─ dev_server.py
├─ requirements.txt
└─ vercel.json
```

## 배포

`vercel.json`은 Python 서버리스 함수에 60초·1024MB를 할당하고 `market_briefing/**`를 포함합니다. 매시간 `/api/cron`을 호출해 Binance 코인 개요 캐시를 예열합니다.
