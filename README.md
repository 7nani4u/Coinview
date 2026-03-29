# CoinOracle

CoinOracle는 Binance 공개 REST API를 이용해 암호화폐 가격·거래량·펀딩비·기술지표를 분석하는 **규칙 기반 시그널 시스템**입니다. 이 프로젝트는 확률적 참고 정보를 제공하며, 특정 수익이나 방향을 보장하는 자동매매 시스템이 아닙니다.

## 1. 정보 무결성 기준
- 기존의 "AI 예측" 표현은 `규칙 기반 분석`, `시그널 엔진`, `참고용 예측 시나리오`로 수정했습니다.
- `xgb_forecast`라는 이름은 하위 호환을 위해 유지하지만, 실제 구현은 XGBoost가 아니라 **모멘텀 시뮬레이션**입니다.
- `/api/stock`은 더 이상 권장하지 않으며 `/api/coin` 사용을 안내합니다.

## 2. 현재 구조
- `public/index.html`: 프런트엔드 UI
- `api/index.py`: 데이터 수집, 지표 계산, 시그널·레버리지·검증 라우팅
- `api/config.py`: 상수·타임아웃·유효 범위 설정
- `api/validators.py`: 입력 검증
- `api/backtesting.py`: 검증 지표 계산
- `api/health.py`, `api/validation.py`: 상태 확인 및 로직 검증 엔드포인트

## 3. 주요 엔드포인트
- `/api/coin?ticker=BTC&interval=1d&limit=365`
- `/api/screener?sort_by=volume&sort_order=desc`
- `/api/leverage?symbol=BTCUSDT`
- `/api/sentiment?market=CRYPTO`
- `/api/validation?ticker=BTCUSDT&interval=1d&window=180&horizon=7`
- `/api/health`
- `/api/cron`

## 4. Vercel 최적화안 반영
### cron
- `vercel.json`에서 `/api/cron`을 **15분 주기(`*/15 * * * *`)**로 실행하도록 조정했습니다.
- cron은 `sentiment`, `screener`, `BTC/ETH/SOL` 대표 심볼의 일봉 데이터를 미리 워밍합니다.

### cache
- 외부 Redis/KV는 도입하지 않고, **짧은 TTL 메모리 캐시 + Vercel CDN 캐시 제어 헤더** 조합으로 유지했습니다.
- `/api/coin`: `s-maxage=30`
- `/api/screener`: `s-maxage=300`
- `/api/validation`: `s-maxage=300`
- `/api/health`: `no-store`

### timeout
- `vercel.json`의 Python 함수 `maxDuration`을 15초로 명시했습니다.
- 네트워크 요청 기본 타임아웃은 6초, 스크리너용 배치 호출은 8초 기준으로 설계했습니다.

### batch 호출
- 스크리너는 심볼별로 `ticker/24hr`와 `premiumIndex`를 반복 호출하지 않습니다.
- `24hr ticker 전체 1회`, `funding 전체 1회`, `klines는 상위 심볼만 병렬 호출` 구조로 재설계했습니다.
- 병렬 워커는 4개로 제한해 Vercel 실행 시간과 Binance rate-limit 위험을 낮췄습니다.

## 5. 백테스트 설계안 반영
`/api/validation` 엔드포인트는 현재 forecast/score/leverage 로직을 다음 방식으로 검증합니다.

### forecast 검증
- 방법: rolling walk-forward
- 입력: `window`, `horizon`
- 지표:
  - 방향 정확도(direction accuracy)
  - MAPE
- 대상:
  - Holt-Winters 예측
  - 모멘텀 시뮬레이션 예측

### score 검증
- 규칙:
  - `score >= 60` → long 시그널
  - `score <= 40` → short 시그널
- 검증값:
  - hit rate
  - 평균/중앙값 미래 수익률

### leverage 검증
- 규칙:
  - 시점별 추천 레버리지와 `stop_loss_pct` 계산
  - horizon 시점 실제 절대 수익률이 손절폭을 초과했는지 집계
- 검증값:
  - stop breach rate
  - 평균/중앙값 추천 레버리지

### 해석 주의
- 이 검증은 **휴리스틱 로직 품질 확인용**입니다.
- 슬리피지, 수수료, 체결 실패, 포지션 동시 보유, 레버리지 청산 메커니즘은 별도 리스크 모델로 추가 검증해야 합니다.

## 6. 보안 점검표 및 반영 내용
### CSP
- `unsafe-eval` 제거
- `object-src 'none'`, `base-uri 'self'`, `frame-ancestors 'none'` 추가
- 현재 UI가 인라인 스크립트를 사용하므로 `script-src`의 `unsafe-inline`은 유지했습니다. 완전 제거를 원하면 프런트엔드 스크립트를 외부 파일로 분리해야 합니다.

### XSS
- 프런트엔드에서 API 응답 문자열을 `sanitizePayload()`로 HTML escape 처리합니다.
- 뉴스 링크는 `safeUrl()`로 `http/https`만 허용합니다.
- 외부 링크에는 `rel="noopener noreferrer"`를 추가했습니다.

### 입력 검증
- `ticker`, `interval`, `window`, `horizon`, `limit`에 대한 정규식 및 범위 검증을 중앙화했습니다.
- `/api/coin`, `/api/stock`, `/api/validation` 모두 동일 검증 규칙을 사용합니다.

### 에러 노출
- 사용자에게는 일반화된 에러 메시지를 반환하고, 세부 네트워크 예외는 서버 로그에 남기도록 정리했습니다.
- `/api/health`에서 런타임·캐시 메타 정보만 노출하고 내부 스택트레이스는 반환하지 않습니다.

## 7. 로컬 실행
```bash
pip install -r requirements.txt
python api/index.py
# 또는 python dev_server.py
```
브라우저에서 `http://localhost:8000` 또는 `http://localhost:3000` 접속

## 8. 테스트 및 품질 게이트
- `tests/`에 입력 검증 및 backtesting 요약 함수에 대한 단위 테스트를 추가했습니다.
- GitHub Actions CI는 다음을 수행합니다.
  - 의존성 설치
  - `unittest` 실행
  - `compileall`로 API 모듈 컴파일 검증

## 9. 한계
- 메모리 캐시는 서버리스 인스턴스 간 공유되지 않습니다.
- 현재 예측 로직은 통계적 참고치이며, 학습형 ML 모델이 아닙니다.
- Binance 공개 엔드포인트 가용성과 지역별 차단 여부에 따라 응답 품질이 달라질 수 있습니다.
