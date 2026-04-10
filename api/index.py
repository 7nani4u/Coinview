# -*- coding: utf-8 -*-
"""
CoinOracle - Binance 기반 암호화폐 규칙 기반 분석 시스템
===================================================
[구조 설명]
- StockOracle(주식 분석)을 암호화폐 전용으로 완전 변환
- Binance REST API 기반 데이터 수집 (yfinance → Binance API)
- 현물(Spot) + 선물(Futures) 데이터 통합 지원
- 레버리지 자동 예측 기능 추가 (ATR, 변동성, 펀딩비 기반)
- 24시간 거래 구조 반영 (주말/공휴일 없음)
- 단일 Python 파일이 HTML 프론트엔드 + 모든 /api/* 엔드포인트 처리

[변환 내역 요약]
- yfinance → Binance REST API (무인증, 공개 엔드포인트)
- KRX/US 시장 → Spot/Futures 시장 구분
- 종목 코드 → BTCUSDT 형식 심볼
- 재무 스크리너(PER/ROE/PBR) → 온체인/시장 스크리너(거래량/펀딩비/변동성)
- 리스크 계산 → 레버리지 예측 + 포지션 크기 조절 로직 추가
- 예측 모델 → 24시간 기반 Holt-Winters + 모멘텀 시뮬레이션

[Binance API 구조]
- 현물 OHLCV  : GET https://api.binance.com/api/v3/klines
- 현물 24h    : GET https://api.binance.com/api/v3/ticker/24hr
- 선물 펀딩비 : GET https://fapi.binance.com/fapi/v1/premiumIndex
- 선물 OI     : GET https://fapi.binance.com/fapi/v1/openInterest
- 시총(CoinGecko): GET https://api.coingecko.com/api/v3/coins/markets
"""

import json
import os
import sys
import time
import datetime
import concurrent.futures
import math
import traceback
import warnings
import functools
import tempfile
import re
from typing import Optional, Dict, Any, List, Tuple
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

# ── /tmp 강제 사용 (Vercel은 /tmp 외 쓰기 금지) ─────────────────────────────
if os.name == 'nt':
    TMP_DIR = tempfile.gettempdir()
else:
    TMP_DIR = "/tmp"

os.environ.setdefault("TMPDIR", TMP_DIR)
os.environ.setdefault("HOME", TMP_DIR)
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(TMP_DIR, "cache"))

warnings.filterwarnings("ignore")

# ── 의존성 ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import requests

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from googlenewsdecoder import gnewsdecoder
    GNEWSDECODER_AVAILABLE = True
except ImportError:
    GNEWSDECODER_AVAILABLE = False

from api import config
from api.backtesting import (
    calc_directional_accuracy,
    calc_mape,
    safe_pct_change,
    summarize_leverage_outcomes,
    summarize_signal_outcomes,
)
from api.cache_backend import cache_meta, ttl_cache
from api.validators import (
    normalize_interval,
    normalize_limit,
    normalize_validation_horizon,
    normalize_validation_window,
    validate_interval,
    validate_ticker,
)

# =============================================================================
# 캐시 / 설정 / 검증 모듈
# =============================================================================

# =============================================================================
# ── 코인 심볼 매핑 (한국어/영어 별칭 → Binance USDT 심볼)
# =============================================================================
COIN_ALIASES = config.COIN_ALIASES

# 스크리너용 주요 코인 유니버스
COIN_UNIVERSE = config.COIN_UNIVERSE

def resolve_symbol(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    사용자 입력 → Binance 심볼 변환
    Returns: (binance_symbol, display_name)
    """
    q = q.strip().upper()
    if not q:
        return None, None

    # 별칭 매핑 (먼저 확인하여 한글 등 매핑)
    q_orig = q
    q_lower = q.lower()
    for alias, sym in COIN_ALIASES.items():
        if alias.upper() == q_orig or alias.lower() == q_lower:
            return sym, sym.replace("USDT", "")

    # 직접 USDT 심볼 입력 (예: BTCUSDT)
    if q.endswith("USDT") and len(q) > 4:
        return q, q.replace("USDT", "")
        
    # 직접 BUSD, USDC 등 다른 페어 입력시 그대로 반환
    if q.endswith("BUSD") or q.endswith("USDC"):
        return q, q

    # 순수 심볼 입력 (예: BTC → BTCUSDT)
    # 이미 영문+숫자 조합인 경우 기본적으로 USDT를 붙임
    candidate = q + "USDT"
    return candidate, q

# =============================================================================
# ── Binance API 데이터 수집 모듈
# =============================================================================

BINANCE_BASES = config.BINANCE_BASES
BINANCE_FAPI = config.BINANCE_FAPI
COINGECKO_BASE = config.COINGECKO_BASE
_HEADERS = config.REQUEST_HEADERS
REQUEST_TIMEOUT = config.DEFAULT_TIMEOUT

def _get(url: str, params: dict = None, timeout: int = REQUEST_TIMEOUT) -> Any:
    """공통 GET 요청 (오류 시 None 반환)"""
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        print(f"[API Timeout] GET {url} params={params} timed out")
    except requests.exceptions.HTTPError as e:
        print(f"[API HTTPError] GET {url} params={params} failed: {e}")
    except requests.exceptions.RequestException as e:
        print(f"[API RequestError] GET {url} params={params} failed: {e}")
    except ValueError as e:
        print(f"[API JSONError] GET {url} params={params} failed: {e}")
    return None

def _get_binance(endpoint: str, params: dict = None, timeout: int = REQUEST_TIMEOUT) -> Any:
    """Binance 현물 API 요청 (여러 엔드포인트 폴백 지원)"""
    for base in BINANCE_BASES:
        url = f"{base}{endpoint}"
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 429, 451):
                print(f"[API Warning] {url} returned {r.status_code}, trying next...")
                continue
            print(f"[API Warning] {url} returned {r.status_code}, trying next...")
        except requests.exceptions.Timeout:
            print(f"[API Timeout] GET {url} params={params} timed out")
        except requests.exceptions.RequestException as e:
            print(f"[API Warning] GET {url} failed: {e}, trying next...")
        except ValueError:
            print(f"[API Error] {url} returned invalid JSON")
    print(f"[API Error] All Binance base endpoints failed for {endpoint}")
    return None

@ttl_cache(60)
def fetch_coin_klines(symbol: str, interval: str = "1d", limit: int = 365) -> Optional[pd.DataFrame]:
    """
    Binance 현물 OHLCV 수집
    interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    data = _get_binance("/api/v3/klines",
                        {"symbol": symbol, "interval": interval, "limit": limit})
    if not data:
        return None
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume","quote_volume"]:
        df[col] = pd.to_numeric(df[col])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.rename(columns={
        "open_time": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
        "quote_volume": "QuoteVolume"
    })
    return df

@ttl_cache(30)
def fetch_ticker_24h(symbol: str) -> Optional[Dict]:
    """Binance 24시간 티커 (현재가, 변동률, 거래량)"""
    return _get_binance("/api/v3/ticker/24hr", {"symbol": symbol})

@ttl_cache(30)
def fetch_all_tickers_24h() -> Optional[List[Dict]]:
    """Binance 전체 24시간 티커"""
    return _get_binance("/api/v3/ticker/24hr")

@ttl_cache(30)
def fetch_funding_rate(symbol: str) -> Optional[float]:
    """
    Binance 선물 펀딩비 조회
    선물 심볼 형식과 동일 (BTCUSDT)
    """
    data = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    if data and "lastFundingRate" in data:
        return float(data["lastFundingRate"]) * 100  # 퍼센트 변환
    return None

@ttl_cache(30)
def fetch_all_funding_rates() -> Dict[str, float]:
    """Binance 전체 선물 펀딩비"""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex")
    if not data:
        return {}
    result = {}
    for item in data:
        sym = item.get("symbol", "")
        rate = item.get("lastFundingRate")
        if sym and rate is not None:
            result[sym] = float(rate) * 100
    return result

@ttl_cache(30)
def fetch_open_interest(symbol: str) -> Optional[float]:
    """Binance 선물 미결제약정(Open Interest)"""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/openInterest", {"symbol": symbol})
    if data and "openInterest" in data:
        return float(data["openInterest"])
    return None

@ttl_cache(300)
def fetch_coingecko_markets(limit: int = 100) -> List[Dict]:
    """CoinGecko 시가총액 상위 코인 목록"""
    data = _get(f"{COINGECKO_BASE}/coins/markets", {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false"
    })
    return data or []

@ttl_cache(60)
def fetch_coin_news(symbol_base: str) -> List[Dict]:
    """코인 관련 뉴스 (Google RSS)"""
    news = []
    if not FEEDPARSER_AVAILABLE:
        return news
    try:
        q = f"{symbol_base} cryptocurrency"
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
        entries = feedparser.parse(url).entries[:5]
        
        def process_entry(e):
            link = e.link
            if GNEWSDECODER_AVAILABLE:
                try:
                    res = gnewsdecoder(link)
                    if res.get("status") and res.get("decoded_url"):
                        link = res["decoded_url"]
                except Exception:
                    pass
            return {
                "title": e.title,
                "link": link,
                "publisher": getattr(e, "source", type("", (), {"title": "Google News"})()).title,
                "published": getattr(e, "published", ""),
            }
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            news = list(executor.map(process_entry, entries))
    except Exception:
        pass
    return news

# =============================================================================
# ── 기술적 지표 계산 (암호화폐 특화)
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    기술적 지표 계산
    [변경] 주식용 MA5/MA120 → 코인용 MA7/MA200 추가
    [변경] 24시간 거래 반영 (주말 갭 없음)
    [추가] VWAP, Stochastic RSI
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # 이동평균 (코인 시장 표준: 7, 25, 50, 99, 200)
    for w in [7, 25, 50, 99, 200]:
        df[f"MA{w}"] = c.rolling(w).mean()
    # 기존 UI 호환용 별칭
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()

    # EMA
    df["EMA12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA26"] = c.ewm(span=26, adjust=False).mean()

    # RSI (14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    # 볼린저 밴드 (20, 2σ)
    df["BB_Middle"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]  # 변동성 지표

    # ATR (14) - 코인 레버리지 계산의 핵심
    hl  = h - l
    hc  = (h - c.shift()).abs()
    lc  = (l - c.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR"] / c * 100  # ATR을 가격 대비 퍼센트로 표현

    # 스토캐스틱 (14, 3)
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    denom  = (high14 - low14).replace(0, np.nan)
    df["%K"] = (c - low14) / denom * 100
    df["%D"] = df["%K"].rolling(3).mean()

    # VWAP (20일 롤링) - 코인 트레이딩의 핵심 지표
    typical_price = (h + l + c) / 3
    df["VWAP"] = (typical_price * v).rolling(20).sum() / v.rolling(20).sum()

    # 30일 변동성 (연율화) - 레버리지 계산에 사용
    df["Volatility_30d"] = c.pct_change().rolling(30).std() * np.sqrt(365) * 100

    # OBV (On-Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    return df

@ttl_cache(60)
def fetch_coin_data(symbol: str, interval: str = "1d", limit: int = 365):
    """
    코인 데이터 통합 수집 (기존 fetch_stock_data 대체)
    - Binance OHLCV + 지표 계산
    - 24h 티커 (현재가, 변동률)
    - 선물 펀딩비 (있는 경우)
    - 뉴스
    """
    symbol = symbol.upper()
    # resolve_symbol에서 이미 USDT를 붙여서 주지만 혹시 몰라 방어코드 유지
    if not symbol.endswith("USDT") and not symbol.endswith("BUSD") and not symbol.endswith("USDC"):
        symbol = symbol + "USDT"

    # OHLCV 수집
    df = fetch_coin_klines(symbol, interval=interval, limit=limit)
    if df is None:
        print(f"[Data Error] fetch_coin_klines returned None for {symbol} ({interval})")
        return None, None, symbol
    if df.empty:
        print(f"[Data Error] fetch_coin_klines returned empty DataFrame for {symbol} ({interval})")
        return None, None, symbol

    # 지표 계산
    df = add_indicators(df)
    df = df.dropna(subset=["Close", "MA20", "RSI"])
    
    if df.empty:
        print(f"[Data Error] DataFrame became empty after dropna for {symbol} (not enough data for indicators)")
        return None, None, symbol

    # 24h 티커
    ticker = fetch_ticker_24h(symbol)

    # 선물 펀딩비
    funding_rate = fetch_funding_rate(symbol)

    # 뉴스
    base = symbol.replace("USDT", "")
    news = fetch_coin_news(base)

    # 날짜 포맷
    df2 = df.copy()
    df2["Date"] = df2["Date"].dt.strftime("%Y-%m-%d %H:%M")
    d = df2.where(pd.notna(df2), other=None).to_dict(orient="list")

    # 24h 정보 추가
    if ticker:
        d["ticker_24h"] = {
            "price":      float(ticker.get("lastPrice", 0)),
            "change_pct": float(ticker.get("priceChangePercent", 0)),
            "high_24h":   float(ticker.get("highPrice", 0)),
            "low_24h":    float(ticker.get("lowPrice", 0)),
            "volume_24h": float(ticker.get("quoteVolume", 0)),
            "count":      int(ticker.get("count", 0)),
        }

    d["funding_rate"] = funding_rate

    return d, news, symbol

# =============================================================================
# ── 레버리지 예측 엔진 (핵심 신규 기능)
# =============================================================================

def calc_leverage_recommendation(
    price: float,
    atr: float,
    atr_pct: float,
    volatility_30d: float,
    funding_rate: Optional[float],
    rsi: float,
    volume_ratio: float,
    macd: float = 0.0,
    macd_signal: float = 0.0,
    score: int = 50,
    bb_upper: float = 0.0,
    bb_lower: float = 0.0,
    ema12: float = 0.0,
    ema26: float = 0.0
) -> Dict:
    """
    적정 레버리지 자동 계산 엔진
    ─────────────────────────────────────────────────────────────────
    [알고리즘]
    1. 기본 레버리지 = 20 / ATR_Pct  (ATR이 클수록 레버리지 낮춤)
    2. 변동성 조정   = 30일 연율 변동성 기반 추가 감소
    3. 펀딩비 패널티 = |펀딩비| > 0.05% 시 레버리지 감소
    4. RSI 극단값   = RSI < 20 또는 > 80 시 추가 감소 (반전 위험)
    5. 거래량 이상  = 거래량 급증 시 변동성 위험 반영

    [출력]
    - recommended_leverage: 최종 추천 레버리지 (1~20x)
    - risk_grade: Low / Medium / High / Extreme
    - max_leverage: 절대 초과 금지 레버리지
    - position_size_pct: 권장 포지션 크기 (자산 대비 %)
    - stop_loss_pct: 권장 손절 퍼센트
    - take_profit_pct: 권장 익절 퍼센트
    ─────────────────────────────────────────────────────────────────
    """
    if not atr_pct or atr_pct <= 0:
        atr_pct = 2.0

    # ── Step 1: ATR 기반 기본 레버리지 ──────────────────────────────
    # 공식: base_lev = 20 / ATR_Pct
    # ATR_Pct 2% → 10x, 4% → 5x, 1% → 20x
    base_lev = min(20.0, 20.0 / max(atr_pct, 0.5))

    # ── Step 2: 30일 변동성 조정 ────────────────────────────────────
    # 연율 변동성 80% 이상 → 고변동성 코인 (알트코인 특성)
    vol_factor = 1.0
    if volatility_30d > 150:
        vol_factor = 0.4   # 극고변동성 (밈코인 등)
    elif volatility_30d > 100:
        vol_factor = 0.6
    elif volatility_30d > 80:
        vol_factor = 0.75
    elif volatility_30d > 60:
        vol_factor = 0.85
    elif volatility_30d < 30:
        vol_factor = 1.1   # 저변동성 (BTC, ETH 등)

    # ── Step 3: 펀딩비 패널티 ───────────────────────────────────────
    # 펀딩비 절대값이 크면 시장 쏠림 → 반전 위험 증가
    funding_factor = 1.0
    if funding_rate is not None:
        abs_fr = abs(funding_rate)
        if abs_fr > 0.1:
            funding_factor = 0.5   # 극단적 쏠림
        elif abs_fr > 0.05:
            funding_factor = 0.7
        elif abs_fr > 0.03:
            funding_factor = 0.85

    # ── Step 4: RSI 극단값 조정 ─────────────────────────────────────
    rsi_factor = 1.0
    if rsi < 20 or rsi > 80:
        rsi_factor = 0.7   # 극단적 과매도/과매수 → 반전 위험
    elif rsi < 25 or rsi > 75:
        rsi_factor = 0.85

    # ── Step 5: 거래량 급증 조정 ────────────────────────────────────
    vol_spike_factor = 1.0
    if volume_ratio > 3.0:
        vol_spike_factor = 0.8   # 거래량 3배 이상 급증 → 변동성 위험
    elif volume_ratio > 2.0:
        vol_spike_factor = 0.9

    # ── 최종 레버리지 계산 ──────────────────────────────────────────
    final_lev = base_lev * vol_factor * funding_factor * rsi_factor * vol_spike_factor
    
    # ── 위험도 등급 및 레버리지 세부 조정 (요청사항 반영) ───────────────────
    # 기존에는 최종 레버리지 값에 따라 등급을 나눴으나, 
    # 이제는 산출된 추천 레버리지(final_lev)와 시장 변동성(volatility_30d)을 종합하여 등급을 부여하고
    # 각 등급에 맞는 지정된 범위 내에서 레버리지를 세밀하게 결정합니다.
    
    # 1. 등급 판별 기준 설정 (변동성과 산출된 레버리지 종합)
    if volatility_30d < 40 and final_lev >= 10:
        risk_grade = "Low"
        risk_color = "#3fb950"
        risk_desc  = "안전: 변동성이 낮고 추세가 안정적"
        # Low 등급: 10 ~ 20 (final_lev가 20을 넘지 않도록 제한)
        recommended = int(max(10, min(20, round(final_lev))))
        
    elif volatility_30d < 70 and final_lev >= 5:
        risk_grade = "Medium"
        risk_color = "#d29922"
        risk_desc  = "주의: 일반적인 시장 상태"
        # Medium 등급: 5 ~ 9
        # final_lev 값을 5~9 사이로 매핑
        recommended = int(max(5, min(9, round(final_lev))))
        
    elif volatility_30d < 120 or final_lev >= 3:
        risk_grade = "High"
        risk_color = "#f85149"
        risk_desc  = "위험: 변동성이 높음, 레버리지 축소 권장"
        # High 등급: 3 ~ 4
        recommended = int(max(3, min(4, round(final_lev))))
        
    else:
        risk_grade = "Extreme"
        risk_color = "#ff0000"
        risk_desc  = "극도 위험: 극심한 변동성, 현물 거래 권장"
        # Extreme 등급: 1 ~ 2
        recommended = int(max(1, min(2, round(final_lev))))

    # ── 포지션 방향(Long/Short) 예측 (볼린저 밴드, EMA 추가 반영) ─────────────────────────────────
    is_ema_bullish = ema12 > ema26 if ema12 and ema26 else False
    is_macd_bullish = macd > macd_signal
    is_overbought = rsi > 70 or (bb_upper > 0 and price >= bb_upper)
    is_oversold = rsi < 30 or (bb_lower > 0 and price <= bb_lower)

    if score >= 65:
        if is_overbought:
            position = "Neutral"
            position_desc = "강한 상승이나 단기 과열 (조정 주의)"
        else:
            position = "Long"
            position_desc = "강한 상승 추세 (AI 및 지표 우수)"
    elif score <= 35:
        if is_oversold:
            position = "Neutral"
            position_desc = "강한 하락이나 단기 투매 (반등 주의)"
        else:
            position = "Short"
            position_desc = "강한 하락 추세 (AI 및 지표 약세)"
    else:
        # 종합 점수 중립 구간 (35 ~ 65): EMA, MACD, BB를 통한 세밀한 모멘텀 판단
        if is_ema_bullish and is_macd_bullish and rsi > 50:
            if bb_upper > 0 and price >= bb_upper:
                position = "Short"
                position_desc = "볼린저 상단 저항 (단기 하락 전환 예상)"
            else:
                position = "Long"
                position_desc = "단기 상승 모멘텀 (EMA 정배열 및 MACD 호조)"
        elif not is_ema_bullish and not is_macd_bullish and rsi < 50:
            if bb_lower > 0 and price <= bb_lower:
                position = "Long"
                position_desc = "볼린저 하단 지지 (단기 반등 예상)"
            else:
                position = "Short"
                position_desc = "단기 하락 모멘텀 (EMA 역배열 및 MACD 약세)"
        else:
            position = "Neutral"
            position_desc = "추세 불분명 (관망 권장)"

    # ── 포지션 크기 및 손절/익절 계산 ──────────────────────────────
    # 켈리 기준 변형: 포지션 크기 = 10% / 레버리지 (보수적)
    position_size_pct = round(min(10.0, 10.0 / max(recommended, 1)), 1)

    # 손절: ATR × 1.5 (레버리지 적용 전 가격 기준)
    stop_loss_pct  = round(atr_pct * 1.5, 2)
    # 익절: 손절의 2배 (최소 RR 1:2)
    take_profit_pct = round(stop_loss_pct * 2.0, 2)

    # 레버리지별 실질 손실 계산
    lev_stop_loss_pct = round(stop_loss_pct * recommended, 1)

    # ── Long / Short 2가지 진입 전략 (Stop Limit용) 추가 ───────────────────
    # Long 진입 (상향 돌파)
    long_stop = price + (atr * 0.5)
    long_limit = long_stop * 1.001
    long_sl = price - (atr * 1.5)
    long_tp = price * (1 + take_profit_pct / 100)
    
    # Short 진입 (하향 돌파)
    short_stop = price - (atr * 0.5)
    short_limit = short_stop * 0.999
    short_sl = price + (atr * 1.5)
    short_tp = price * (1 - take_profit_pct / 100)
    
    trading_signals = {
        "long": {
            "stop": round(long_stop, 6),
            "limit": round(long_limit, 6),
            "sl": round(long_sl, 6),
            "tp": round(long_tp, 6),
            "leverage": recommended
        },
        "short": {
            "stop": round(short_stop, 6),
            "limit": round(short_limit, 6),
            "sl": round(short_sl, 6),
            "tp": round(short_tp, 6),
            "leverage": recommended
        }
    }

    return {
        "position":             position,
        "position_desc":        position_desc,
        "recommended_leverage": recommended,
        "max_leverage":         min(recommended * 2, 20),
        "risk_grade":           risk_grade,
        "risk_color":           risk_color,
        "risk_desc":            risk_desc,
        "position_size_pct":    position_size_pct,
        "stop_loss_pct":        stop_loss_pct,
        "take_profit_pct":      take_profit_pct,
        "lev_stop_loss_pct":    lev_stop_loss_pct,
        "trading_signals":      trading_signals,
        # 계산 근거 (UI 표시용)
        "factors": {
            "base_leverage":    round(base_lev, 2),
            "vol_factor":       round(vol_factor, 2),
            "funding_factor":   round(funding_factor, 2),
            "rsi_factor":       round(rsi_factor, 2),
            "vol_spike_factor": round(vol_spike_factor, 2),
            "atr_pct":          round(atr_pct, 2),
            "volatility_30d":   round(volatility_30d, 1) if volatility_30d else None,
            "funding_rate":     round(funding_rate, 4) if funding_rate else None,
        }
    }

# =============================================================================
# ── 리스크 관리 (기존 calc_risk 확장)
# =============================================================================

def calc_risk(price: float, atr: float, leverage_info: Dict = None) -> Dict:
    """
    리스크 시나리오 계산 (레버리지 정보 통합)
    [변경] 주식용 고정 배수 → 코인 ATR 기반 동적 계산
    [추가] 레버리지 적용 시 실질 손익 표시
    """
    if not atr or np.isnan(atr):
        atr = price * 0.02

    lev = leverage_info.get("recommended_leverage", 3) if leverage_info else 3

    return {
        "conservative": {
            "label":     "보수적 (현물)",
            "target":    round(price + atr * 1.5, 6),
            "stop":      round(price - atr, 6),
            "ratio":     "1:1.5",
            "desc":      "현물 보유 / 리스크 최소화",
            "icon":      "🛡️",
            "lev":       "1x",
            "lev_gain":  round(atr * 1.5 / price * 100, 2),
            "lev_loss":  round(atr / price * 100, 2),
        },
        "balanced": {
            "label":     f"중립적 ({lev}x 레버리지)",
            "target":    round(price + atr * 2.5, 6),
            "stop":      round(price - atr * 1.5, 6),
            "ratio":     "1:1.67",
            "desc":      "스윙 트레이딩 / 권장 레버리지",
            "icon":      "⚖️",
            "lev":       f"{lev}x",
            "lev_gain":  round(atr * 2.5 / price * 100 * lev, 2),
            "lev_loss":  round(atr * 1.5 / price * 100 * lev, 2),
        },
        "aggressive": {
            "label":     "공격적 (5x 레버리지)",
            "target":    round(price + atr * 4, 6),
            "stop":      round(price - atr * 2, 6),
            "ratio":     "1:2.0",
            "desc":      "추세 추종 / 고위험 고수익",
            "icon":      "🚀",
            "lev":       "5x",
            "lev_gain":  round(atr * 4 / price * 100 * 5, 2),
            "lev_loss":  round(atr * 2 / price * 100 * 5, 2),
        },
    }

# =============================================================================
# ── 분석 엔진 (기존 구조 유지, 코인 특화 수정)
# =============================================================================

class ChartPatternAnalyzer:
    """기하학적 차트 패턴 분석 (기존 로직 유지)"""
    def __init__(self, df):
        self.df     = df
        self.closes = np.array(df["Close"].values, dtype=float)
        self.highs  = np.array(df["High"].values,  dtype=float)
        self.lows   = np.array(df["Low"].values,   dtype=float)

    def find_local_extrema(self, order=5):
        peaks, troughs = [], []
        for i in range(order, len(self.highs) - order):
            window = self.highs[i-order: i+order+1]
            if self.highs[i] == np.max(window) and self.highs[i] != self.highs[i-1]:
                peaks.append(i)
        for i in range(order, len(self.lows) - order):
            window = self.lows[i-order: i+order+1]
            if self.lows[i] == np.min(window) and self.lows[i] != self.lows[i-1]:
                troughs.append(i)
        return np.array(peaks), np.array(troughs)

    def detect_patterns(self):
        patterns = []
        try:
            peaks, troughs = self.find_local_extrema(order=5)
            if len(peaks) < 3 or len(troughs) < 3:
                return patterns
            last_peaks   = peaks[-3:]
            last_troughs = troughs[-3:]

            def get_slope(x, y):
                if len(x) < 2: return 0
                return np.polyfit(x, y, 1)[0]

            slope_upper = get_slope(last_peaks,   self.highs[last_peaks])
            slope_lower = get_slope(last_troughs, self.lows[last_troughs])

            if slope_upper < 0 and slope_lower > 0:
                patterns.append({"name":"대칭 삼각형 (Symmetrical Triangle)","signal":"중립/변동성 축소","desc":"곧 큰 방향성이 나올 것입니다."})
            elif slope_upper < 0 and abs(slope_lower) < 0.05:
                patterns.append({"name":"하락 삼각형 (Descending Triangle)","signal":"매도 (하락형)","desc":"지지선 붕괴 위험이 있습니다."})
            elif abs(slope_upper) < 0.05 and slope_lower > 0:
                patterns.append({"name":"상승 삼각형 (Ascending Triangle)","signal":"매수 (상승형)","desc":"저항선 돌파 시도가 예상됩니다."})
            if slope_upper < 0 and slope_lower < 0 and slope_lower < slope_upper:
                patterns.append({"name":"하락 쐐기형 (Falling Wedge)","signal":"매수 (반전)","desc":"하락세가 약화되고 반등할 가능성이 큽니다."})
            if slope_upper > 0 and slope_lower > 0 and slope_lower > slope_upper:
                patterns.append({"name":"상승 쐐기형 (Rising Wedge)","signal":"매도 (반전)","desc":"상승세가 약화되고 하락할 가능성이 큽니다."})
            if len(last_peaks) >= 2:
                if abs(self.highs[last_peaks[-1]] - self.highs[last_peaks[-2]]) / (self.highs[last_peaks[-1]] or 1) < 0.02:
                    patterns.append({"name":"이중 천장 (Double Top)","signal":"매도","desc":"고점 돌파 실패, 하락 전환 가능성."})
            if len(troughs) >= 2:
                if abs(self.lows[last_troughs[-1]] - self.lows[last_troughs[-2]]) / (self.lows[last_troughs[-1]] or 1) < 0.02:
                    patterns.append({"name":"이중 바닥 (Double Bottom)","signal":"매수","desc":"바닥 지지 성공, 상승 전환 가능성."})
        except Exception:
            pass
        return patterns


def detect_patterns(dd: Dict) -> List[Dict]:
    """캔들스틱 패턴 감지 (기존 로직 유지)"""
    patterns = []
    try:
        o = [float(x) for x in dd.get("Open",  []) if x is not None]
        h = [float(x) for x in dd.get("High",  []) if x is not None]
        l = [float(x) for x in dd.get("Low",   []) if x is not None]
        c = [float(x) for x in dd.get("Close", []) if x is not None]
        if len(c) < 3: return []
        o1,h1,l1,c1 = o[-1],h[-1],l[-1],c[-1]
        o2,h2,l2,c2 = o[-2],h[-2],l[-2],c[-2]
        o3,h3,l3,c3 = o[-3],h[-3],l[-3],c[-3]
        body1 = abs(c1-o1); rng1 = h1-l1 or 0.001
        body2 = abs(c2-o2); rng2 = h2-l2 or 0.001
        up_sh1 = h1-max(c1,o1); lo_sh1 = min(c1,o1)-l1
        bull1 = c1>=o1; bull2 = c2>=o2; bull3 = c3>=o3

        if body1/rng1 < 0.1:
            patterns.append({"name":"✖️ Doji","desc":"도지","direction":"중립","conf":100})
        if lo_sh1 >= body1*2 and up_sh1 <= body1*0.5 and body1>0 and c2>c1:
            patterns.append({"name":"🔨 Hammer","desc":"해머 (반등 신호)","direction":"상승","conf":100})
        if up_sh1 >= body1*2 and lo_sh1 <= body1*0.5 and body1>0 and c2<c1:
            patterns.append({"name":"⭐ Shooting Star","desc":"유성형 (하락 신호)","direction":"하락","conf":100})
        if bull1 and not bull2 and o1<=c2 and c1>=o2 and body1>body2:
            patterns.append({"name":"🫂 Bullish Engulfing","desc":"상승 포용형","direction":"상승","conf":100})
        if not bull1 and bull2 and o1>=c2 and c1<=o2 and body1>body2:
            patterns.append({"name":"🫂 Bearish Engulfing","desc":"하락 포용형","direction":"하락","conf":100})
        if bull1 and not bull2 and o1>c2 and c1<o2 and body1<body2*0.5:
            patterns.append({"name":"🤰 Bullish Harami","desc":"상승 하라미","direction":"상승","conf":100})
        if body1/rng1 > 0.9 and body1>0:
            patterns.append({"name":"📏 Marubozu","desc":f"마루보즈({'상승' if bull1 else '하락'})","direction":"상승" if bull1 else "하락","conf":100})
        if bull3 and body2/rng2 < 0.3 and bull1 and c1 > (o3+c3)/2 and c3 > o3:
            patterns.append({"name":"🌆 Evening Star","desc":"이브닝스타 (하락 반전)","direction":"하락","conf":100})
        if not bull3 and body2/rng2 < 0.3 and bull1 and c1 < (o3+c3)/2 and c3 < o3:
            patterns.append({"name":"🌅 Morning Star","desc":"모닝스타 (상승 반전)","direction":"상승","conf":100})
    except Exception:
        pass
    return patterns


def analyze_score(dd: Dict):
    """
    AI 분석 점수 계산 (기존 구조 유지, 코인 특화 수정)
    [변경] 주식 재무 지표 → 코인 온체인/시장 지표
    [추가] 펀딩비 분석 스텝, VWAP 분석 스텝
    """
    closes = dd.get("Close", [])
    if len(closes) < 20:
        return 50, [], [], []

    def v(k):
        a = dd.get(k, [])
        val = a[-1] if a else None
        return float(val) if val is not None else 0.0

    close   = v("Close")
    ma20    = v("MA20")
    ma60    = v("MA60")
    rsi     = v("RSI")
    macd    = v("MACD")
    sig     = v("Signal_Line")
    bb_u    = v("BB_Upper")
    bb_l    = v("BB_Lower")
    vwap    = v("VWAP")
    atr_pct = v("ATR_Pct")
    vols    = dd.get("Volume", [])
    cur_vol = float(vols[-1]) if vols else 0
    avg_vol = float(np.mean([x for x in vols[-20:] if x])) if vols else 1
    opn     = v("Open")
    funding = dd.get("funding_rate")

    score, steps = 50, []

    # ── 1. 추세 분석 (MA) ──────────────────────────────────────────
    ts, msg = 0, ""
    if close > ma20:
        ts += 10
        if close > ma60:
            ts += 10
            if ma20 > ma60: ts += 10; msg = "단기/장기 이동평균 정배열 → 강한 상승 추세"
            else: msg = "장기 이평선 위 → 상승 기조"
        else: msg = "20일 이평선 위 → 단기 상승 시도"
    else:
        ts -= 10
        if close < ma60:
            ts -= 10
            if ma20 < ma60: ts -= 10; msg = "역배열 → 하락 압력 강함"
            else: msg = "장기 이평선 아래 → 하락 추세 우려"
        else: msg = "20일 이평선 하회 → 조정 중"
    score += ts
    steps.append({"step": "1. 추세 분석 (MA)", "result": msg, "score": ts})

    # ── 2. 모멘텀 (RSI/MACD) ──────────────────────────────────────
    ms, msgs = 0, []
    if rsi > 70:   ms -= 5;  msgs.append(f"RSI {rsi:.1f} 과매수 — 단기 조정 주의")
    elif rsi < 30: ms += 10; msgs.append(f"RSI {rsi:.1f} 과매도 → 반등 기대")
    else:          msgs.append(f"RSI {rsi:.1f} 중립")
    if macd > sig: ms += 10; msgs.append("MACD 골든크로스 → 상승 신호")
    else:          ms -= 10; msgs.append("MACD 데드크로스 → 하락 신호")
    score += ms
    steps.append({"step": "2. 모멘텀 (RSI/MACD)", "result": " | ".join(msgs), "score": ms})

    # ── 3. 거래량/볼린저 밴드 ─────────────────────────────────────
    vs, vmsgs = 0, []
    if close > bb_u * 0.98:   vs += 5;  vmsgs.append("볼린저 상단 터치 — 과매수 경계")
    elif close < bb_l * 1.02: vs -= 5;  vmsgs.append("볼린저 하단 터치 — 반등 가능")
    if avg_vol > 0 and cur_vol > avg_vol * 1.5:
        if close > opn: vs += 10; vmsgs.append("거래량 급증 + 상승 → 신뢰도 높음")
        else:           vs -= 10; vmsgs.append("거래량 급증 + 하락 → 매도 압력")
    else: vmsgs.append(f"거래량 평이 ({cur_vol/avg_vol:.1f}x)")
    score += vs
    steps.append({"step": "3. 거래량/볼린저 밴드", "result": " | ".join(vmsgs), "score": vs})

    # ── 4. VWAP 분석 (코인 특화) ──────────────────────────────────
    ws, wmsgs = 0, []
    if vwap > 0:
        vwap_diff = (close - vwap) / vwap * 100
        if vwap_diff > 3:
            ws -= 5; wmsgs.append(f"VWAP 대비 +{vwap_diff:.1f}% — 단기 과열")
        elif vwap_diff < -3:
            ws += 5; wmsgs.append(f"VWAP 대비 {vwap_diff:.1f}% — 저평가 구간")
        else:
            wmsgs.append(f"VWAP 근접 ({vwap_diff:+.1f}%) — 균형 상태")
    else:
        wmsgs.append("VWAP 데이터 없음")
    score += ws
    steps.append({"step": "4. VWAP 분석", "result": " | ".join(wmsgs), "score": ws})

    # ── 5. 펀딩비 분석 (선물 코인 특화) ──────────────────────────
    fs, fmsgs = 0, []
    if funding is not None:
        abs_fr = abs(funding)
        if abs_fr > 0.1:
            fs -= 10; fmsgs.append(f"펀딩비 {funding:+.4f}% — 극단적 쏠림, 반전 위험 높음")
        elif abs_fr > 0.05:
            fs -= 5;  fmsgs.append(f"펀딩비 {funding:+.4f}% — 과도한 레버리지 경고")
        elif funding > 0.01:
            fs -= 3;  fmsgs.append(f"펀딩비 {funding:+.4f}% — 롱 우세 (약한 경고)")
        elif funding < -0.01:
            fs += 3;  fmsgs.append(f"펀딩비 {funding:+.4f}% — 숏 우세 (반등 기대)")
        else:
            fmsgs.append(f"펀딩비 {funding:+.4f}% — 중립")
    else:
        fmsgs.append("선물 데이터 없음 (현물 전용)")
    score += fs
    steps.append({"step": "5. 펀딩비 (선물 심리)", "result": " | ".join(fmsgs), "score": fs})

    # ── 6. 캔들 패턴 ──────────────────────────────────────────────
    patterns = detect_patterns(dd)
    ps, pmsgs = 0, []
    if patterns:
        bull = sum(1 for p in patterns if p["direction"] == "상승")
        bear = sum(1 for p in patterns if p["direction"] == "하락")
        if bull > bear:   ps += 10; pmsgs.append(f"상승 패턴 {bull}개")
        elif bear > bull: ps -= 10; pmsgs.append(f"하락 패턴 {bear}개")
        else:             pmsgs.append(f"패턴 혼재 {len(patterns)}개")
    else: pmsgs.append("특이 패턴 없음")
    score += ps
    steps.append({"step": "6. 캔들 패턴", "result": " | ".join(pmsgs), "score": ps})

    # ── 기하학적 패턴 ─────────────────────────────────────────────
    geo_patterns = []
    try:
        df = pd.DataFrame({k: dd[k] for k in ["Open","High","Low","Close"] if k in dd})
        if not df.empty and len(df) > 20:
            geo_patterns = ChartPatternAnalyzer(df).detect_patterns()
    except Exception:
        pass

    return max(0, min(100, score)), steps, patterns, geo_patterns

# =============================================================================
# ── 예측 모델 (기존 구조 유지, 24시간 기반으로 수정)
# =============================================================================

def holt_winters_forecast(dd: Dict, days: int = 30):
    """
    Holt-Winters 이중 지수평활 예측
    [변경] 주말 스킵 로직 제거 (코인은 24/7 거래)
    """
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        dates  = dd.get("Date", [])
        if len(closes) < 30: return None

        alpha, beta = 0.8, 0.2
        level = closes[0]
        trend = closes[1] - closes[0]
        for i in range(1, len(closes)):
            last_level = level
            level = alpha * closes[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend

        forecast, future_dates = [], []
        try:
            last_d = datetime.datetime.strptime(dates[-1][:10], "%Y-%m-%d") if dates else datetime.datetime.now()
        except Exception:
            last_d = datetime.datetime.now()

        std = np.std(closes[-30:]) if len(closes) >= 30 else 0
        d = last_d
        for i in range(1, days + 1):
            d += datetime.timedelta(days=1)
            # [변경] 코인은 주말 없음 — while d.weekday() >= 5 제거
            future_dates.append(d.strftime("%Y-%m-%d"))
            yhat = level + i * trend
            forecast.append(yhat)

        return {
            "dates":      future_dates,
            "yhat":       [round(float(f), 6) for f in forecast],
            "yhat_upper": [round(float(f) + 1.96 * std, 6) for f in forecast],
            "yhat_lower": [round(float(f) - 1.96 * std, 6) for f in forecast],
        }
    except Exception:
        return linear_forecast(dd, days)


def linear_forecast(dd: Dict, days: int):
    """선형 회귀 예측 (fallback)"""
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        dates  = dd.get("Date", [])
        if len(closes) < 20: return None

        y = np.array(closes)
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(y), len(y) + days)
        preds = slope * future_x + intercept

        try:
            last_d = datetime.datetime.strptime(dates[-1][:10], "%Y-%m-%d") if dates else datetime.datetime.now()
        except Exception:
            last_d = datetime.datetime.now()

        fds, d = [], last_d
        for _ in range(days):
            d += datetime.timedelta(days=1)
            fds.append(d.strftime("%Y-%m-%d"))

        return {
            "dates":      fds,
            "yhat":       [round(float(p), 6) for p in preds],
            "yhat_upper": [round(float(p) * 1.05, 6) for p in preds],
            "yhat_lower": [round(float(p) * 0.95, 6) for p in preds],
        }
    except Exception:
        return None


def momentum_sim_forecast(dd: Dict, days: int = 30):
    """
    모멘텀 기반 시뮬레이션 예측
    [변경] 코인 고변동성 반영 — decay 0.95 → 0.90
    """
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        if len(closes) < 60: return None

        short_ma = np.mean(closes[-5:])
        long_ma  = np.mean(closes[-20:])
        momentum_strength = (short_ma - long_ma) / long_ma

        last_price    = closes[-1]
        preds         = []
        decay         = 0.90  # [변경] 코인 고변동성 반영 (0.95 → 0.90)
        current_price = last_price
        drift         = momentum_strength * last_price * 0.1

        for _ in range(days):
            drift         *= decay
            current_price += drift
            preds.append(round(current_price, 6))

        return preds
    except Exception:
        return None


def xgb_forecast(dd: Dict, days: int = 30):
    """하위 호환용 별칭. 실제 로직은 모멘텀 시뮬레이션이다."""
    return momentum_sim_forecast(dd, days)

# =============================================================================
# ── 코인 스크리너 (기존 주식 스크리너 완전 대체)
# =============================================================================

SCREENER_INTERVALS = config.SCREENER_INTERVALS

def _categorize_coin(base: str) -> str:
    """간단한 코인 카테고리 분류"""
    base = base.upper()
    if base in ["BTC", "ETH"]:
        return "메이저 (Major)"
    if base in ["BNB", "SOL", "ADA", "XRP", "DOT", "AVAX", "LINK", "MATIC", "POL", "TRX"]:
        return "알트 대장 (Large Alt)"
    if base in ["DOGE", "SHIB", "PEPE", "BONK", "FLOKI", "PENGU", "TRUMP"]:
        return "밈 (Meme)"
    if base in ["UNI", "AAVE", "CRV", "RUNE", "MKR", "LDO", "SNX"]:
        return "디파이 (DeFi)"
    if base in ["OP", "ARB", "MATIC", "POL", "MANTLE"]:
        return "L2 (레이어2)"
    if base in ["SUI", "APT", "SEI", "TIA", "INJ", "TAO"]:
        return "신흥 알트 (New Alt)"
    return "기타 (Others)"

def fetch_coin_screener_item(symbol: str, prefetched_ticker: Optional[Dict] = None, funding_rate: Optional[float] = None) -> Optional[Dict]:
    """단일 코인 스크리너 데이터 수집"""
    try:
        ticker = prefetched_ticker or fetch_ticker_24h(symbol)
        if not ticker:
            return None

        price = float(ticker.get("lastPrice", 0))
        change_pct = float(ticker.get("priceChangePercent", 0))
        volume_24h = float(ticker.get("quoteVolume", 0))
        if price <= 0 or volume_24h <= 0:
            return None

        df = fetch_coin_klines(symbol, interval="1d", limit=60)
        if df is None or len(df) < 20:
            return None
        df = add_indicators(df)
        df = df.dropna(subset=["Close", "RSI", "ATR"])
        if df.empty:
            return None

        rsi = float(df["RSI"].iloc[-1])
        atr = float(df["ATR"].iloc[-1])
        atr_pct = float(df["ATR_Pct"].iloc[-1])
        ma20 = float(df["MA20"].iloc[-1])
        vol_30d = float(df["Volatility_30d"].iloc[-1]) if "Volatility_30d" in df.columns and not pd.isna(df["Volatility_30d"].iloc[-1]) else 0
        macd = float(df["MACD"].iloc[-1]) if "MACD" in df.columns and not pd.isna(df["MACD"].iloc[-1]) else 0.0
        macd_signal = float(df["Signal_Line"].iloc[-1]) if "Signal_Line" in df.columns and not pd.isna(df["Signal_Line"].iloc[-1]) else 0.0
        bb_upper = float(df["BB_Upper"].iloc[-1]) if "BB_Upper" in df.columns and not pd.isna(df["BB_Upper"].iloc[-1]) else 0.0
        bb_lower = float(df["BB_Lower"].iloc[-1]) if "BB_Lower" in df.columns and not pd.isna(df["BB_Lower"].iloc[-1]) else 0.0
        ema12 = float(df["EMA12"].iloc[-1]) if "EMA12" in df.columns and not pd.isna(df["EMA12"].iloc[-1]) else 0.0
        ema26 = float(df["EMA26"].iloc[-1]) if "EMA26" in df.columns and not pd.isna(df["EMA26"].iloc[-1]) else 0.0
        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = float(df["Volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0
        funding = funding_rate
        # 밈코인 등 1000이 붙는 선물 심볼로 인해 못 찾은 경우 개별 호출 생략 (속도 및 에러 방지)

        lev_info = calc_leverage_recommendation(
            price=price, atr=atr, atr_pct=atr_pct,
            volatility_30d=vol_30d, funding_rate=funding,
            rsi=rsi, volume_ratio=vol_ratio,
            macd=macd, macd_signal=macd_signal, score=50,
            bb_upper=bb_upper, bb_lower=bb_lower,
            ema12=ema12, ema26=ema26
        )

        signal = "중립"
        if rsi < 25:
            signal = "적극 매수"
        elif rsi < 35 and price > ma20 * 0.98:
            signal = "매수"
        elif rsi > 75:
            signal = "적극 매도"
        elif rsi > 70:
            signal = "매도"

        base = symbol.replace("USDT", "")
        return {
            "symbol": symbol,
            "name": base,
            "ticker": base,
            "price": (f"${price:,.2f}" if price >= 1000 else f"${price:.4f}" if price >= 1 else f"${price:.6f}" if price >= 0.001 else f"${price:.8f}"),
            "price_val": price,
            "change": change_pct,
            "volume_24h": volume_24h,
            "volume_ratio": round(vol_ratio, 2),
            "rsi": round(rsi, 1),
            "atr_pct": round(atr_pct, 2),
            "volatility": round(vol_30d, 1),
            "funding_rate": round(funding, 4) if funding is not None else None,
            "leverage_rec": lev_info["recommended_leverage"],
            "risk_grade": lev_info["risk_grade"],
            "signal": signal,
            "category": _categorize_coin(base),
        }
    except Exception as e:
        print(f"[Screener Item Error] {symbol}: {e}")
        return None


@ttl_cache(config.CACHE_TTL_SCREENER)
def fetch_coin_screener(sort_by: str = "volume", sort_order: str = "desc") -> Dict:
    """배치 호출 중심 코인 스크리너"""
    ticker_list = fetch_all_tickers_24h() or []
    funding_map = fetch_all_funding_rates()
    ticker_map = {item.get("symbol"): item for item in ticker_list if item.get("symbol") in COIN_UNIVERSE}

    ranked_symbols = sorted(
        [sym for sym in COIN_UNIVERSE if sym in ticker_map],
        key=lambda sym: float(ticker_map[sym].get("quoteVolume", 0) or 0),
        reverse=True,
    )[: config.SCREENER_TOP_N]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.SCREENER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                fetch_coin_screener_item, 
                sym, 
                ticker_map.get(sym), 
                funding_map.get(sym) or funding_map.get(f"1000{sym}")
            ): sym 
            for sym in ranked_symbols
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    sort_map = {
        "name": "name",
        "price": "price_val",
        "change": "change",
        "volume": "volume_24h",
        "rsi": "rsi",
        "leverage": "leverage_rec",
    }
    sort_field = sort_map.get(sort_by, "volume_24h")
    results.sort(key=lambda x: (x.get(sort_field) or 0), reverse=(sort_order != "asc"))

    return {
        "data": results,
        "total": len(results),
        "sort_by": sort_by,
        "sort_order": sort_order,
        "batch_strategy": {
            "tickers_24h": "all-symbol single request",
            "funding_rates": "all-symbol single request",
            "klines": f"top {len(ranked_symbols)} symbols only",
            "workers": config.SCREENER_MAX_WORKERS,
        },
        "filter_conditions": {
            "거래소": "Binance (현물/선물)",
            "호출 최적화": "24h ticker 1회 + funding 1회 + 심볼별 klines",
            "정렬": sort_by,
            "리스크": "ATR/변동성 기반 자동 계산",
            "데이터": "Binance REST API (실시간)",
        },
    }

# =============================================================================
# ── 시장 심리 (기존 fetch_sentiment 대체)
# =============================================================================

@ttl_cache(300)
def fetch_market_sentiment(market: str = "CRYPTO") -> Dict:
    """
    암호화폐 시장 심리 지수
    [대체] KRX/US 지수 → BTC 도미넌스 + Fear & Greed 지수
    """
    try:
        # BTC 24h 데이터
        btc = fetch_ticker_24h("BTCUSDT")
        if btc:
            price  = float(btc.get("lastPrice", 0))
            change = float(btc.get("priceChangePercent", 0))

            # 심리 판단 (BTC 기준)
            if change > 5:
                sentiment = "극도의 탐욕"
            elif change > 2:
                sentiment = "탐욕"
            elif change > -2:
                sentiment = "중립"
            elif change > -5:
                sentiment = "공포"
            else:
                sentiment = "극도의 공포"

            return {
                "name":      "Bitcoin (BTC)",
                "value":     round(price, 2),
                "change":    round(change, 2),
                "sentiment": sentiment,
            }
    except Exception:
        pass

    return {
        "name":      "Bitcoin (BTC)",
        "value":     0,
        "change":    0,
        "sentiment": "조회 실패",
    }

# =============================================================================
# ── 검증 / 라우팅
# =============================================================================

def build_validation_report(symbol: str, interval: str = "1d", window: int = 180, horizon: int = 7) -> Dict:
    raw = fetch_coin_klines(symbol, interval=interval, limit=min(window + horizon + 120, config.MAX_LIMIT))
    if raw is None or len(raw) < max(90, horizon + 60):
        return {"error": "검증에 필요한 캔들 데이터가 부족합니다."}

    df = add_indicators(raw.copy()).reset_index(drop=True)
    start_idx = max(60, len(df) - window)
    hw_pred, hw_actual, hw_base = [], [], []
    mm_pred, mm_actual, mm_base = [], [], []
    score_outcomes = []
    leverage_outcomes = []

    for end_idx in range(start_idx, len(df) - horizon):
        hist = df.iloc[: end_idx + 1].copy()
        future_price = float(df["Close"].iloc[end_idx + horizon])
        base_price = float(hist["Close"].iloc[-1])
        dd = hist.to_dict(orient="list")

        hw = holt_winters_forecast(dd, days=horizon)
        if hw and hw.get("yhat") and len(hw["yhat"]) >= horizon:
            hw_pred.append(float(hw["yhat"][horizon - 1]))
            hw_actual.append(future_price)
            hw_base.append(base_price)

        mm = momentum_sim_forecast(dd, days=horizon)
        if mm and len(mm) >= horizon:
            mm_pred.append(float(mm[horizon - 1]))
            mm_actual.append(future_price)
            mm_base.append(base_price)

        score, _, _, _ = analyze_score(dd)
        future_ret = safe_pct_change(base_price, future_price)
        if score >= 60:
            score_outcomes.append({"future_return_pct": future_ret})
        elif score <= 40:
            score_outcomes.append({"future_return_pct": -future_ret})

        row = hist.iloc[-1]
        vols = hist["Volume"].tail(20)
        avg_vol = float(vols.mean()) if len(vols) else 1.0
        lev = calc_leverage_recommendation(
            price=base_price,
            atr=float(row.get("ATR", 0) or 0),
            atr_pct=float(row.get("ATR_Pct", 0) or 0),
            volatility_30d=float(row.get("Volatility_30d", 0) or 0),
            funding_rate=fetch_funding_rate(symbol),
            rsi=float(row.get("RSI", 50) or 50),
            volume_ratio=float(row.get("Volume", 0) or 0) / avg_vol if avg_vol > 0 else 1.0,
            macd=float(row.get("MACD", 0) or 0),
            macd_signal=float(row.get("Signal_Line", 0) or 0),
            score=score,
            bb_upper=float(row.get("BB_Upper", 0) or 0),
            bb_lower=float(row.get("BB_Lower", 0) or 0),
            ema12=float(row.get("EMA12", 0) or 0),
            ema26=float(row.get("EMA26", 0) or 0),
        )
        leverage_outcomes.append({
            "future_return_pct": future_ret,
            "realized_abs_return_pct": abs(future_ret),
            "stop_loss_pct": lev.get("stop_loss_pct", 0),
            "recommended_leverage": lev.get("recommended_leverage", 1),
        })

    return {
        "symbol": symbol,
        "interval": interval,
        "window": window,
        "horizon": horizon,
        "methodology": {
            "forecast": "rolling walk-forward; horizon 시점의 종가 방향성과 MAPE를 확인",
            "score": "score>=60은 long, score<=40은 short로 간주하여 horizon 수익률 평가",
            "leverage": "추천 손절폭(stop_loss_pct) 대비 실제 절대수익률 초과율을 추적",
        },
        "forecast_validation": {
            "holt_winters": {
                "samples": len(hw_pred),
                "directional_accuracy": calc_directional_accuracy(hw_pred, hw_actual, hw_base),
                "mape": calc_mape(hw_pred, hw_actual),
            },
            "momentum_sim": {
                "samples": len(mm_pred),
                "directional_accuracy": calc_directional_accuracy(mm_pred, mm_actual, mm_base),
                "mape": calc_mape(mm_pred, mm_actual),
            },
        },
        "score_validation": summarize_signal_outcomes(score_outcomes),
        "leverage_validation": summarize_leverage_outcomes(leverage_outcomes),
        "notes": [
            "현재 검증은 규칙 기반 휴리스틱 품질 측정용이며 실거래 성능을 보장하지 않음",
            "슬리피지·수수료·체결 실패는 보수적으로 별도 반영 필요",
        ],
    }

def route(path: str, params: Dict) -> Optional[Dict]:
    if path == "/api/stock":
        return {"error": "Deprecated endpoint. Use /api/coin", "replacement": "/api/coin", "status": "deprecated"}

    if path == "/api/coin":
        raw = params.get("ticker", "BTCUSDT")
        interval = normalize_interval(params.get("interval", config.DEFAULT_INTERVAL))
        limit = normalize_limit(params.get("limit", str(config.DEFAULT_LIMIT)))

        symbol, display = resolve_symbol(raw)
        if not symbol:
            return {"error": f"'{raw}' 코인을 찾을 수 없습니다."}

        dd, news, sym = fetch_coin_data(symbol, interval=interval, limit=limit)
        if dd is None:
            return {"error": f"데이터 조회 실패: 데이터 없음: {sym}"}

        closes = dd.get("Close", [])
        last_ohlcv = float(closes[-1]) if closes else 0
        prev = float(closes[-2]) if len(closes) > 1 else last_ohlcv
        ticker_24h = dd.get("ticker_24h", {})
        if ticker_24h and "price" in ticker_24h:
            last = float(ticker_24h["price"])
            pct = float(ticker_24h.get("change_pct", (last - prev) / prev * 100 if prev else 0))
        else:
            last = last_ohlcv
            pct = (last - prev) / prev * 100 if prev else 0

        def v(k):
            a = dd.get(k, [])
            val = a[-1] if a else None
            return float(val) if val is not None else 0.0

        rsi = v("RSI")
        atr = v("ATR")
        atr_pct = v("ATR_Pct")
        vol_30d = v("Volatility_30d")
        macd = v("MACD")
        macd_signal = v("Signal_Line")
        bb_upper = v("BB_Upper")
        bb_lower = v("BB_Lower")
        ema12 = v("EMA12")
        ema26 = v("EMA26")
        vols = dd.get("Volume", [])
        cur_vol = float(vols[-1]) if vols else 0
        avg_vol = float(np.mean([x for x in vols[-20:] if x])) if vols else 1
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        funding = dd.get("funding_rate")

        score, steps, patterns, geo_patterns = analyze_score(dd)
        for gp in geo_patterns:
            direction = "상승" if gp.get("signal") == "매수" else "하락" if gp.get("signal") == "매도" else "중립"
            patterns.append({"name": gp.get("name"), "desc": gp.get("desc"), "direction": direction, "conf": 100})

        forecast = holt_winters_forecast(dd)
        momentum = momentum_sim_forecast(dd)
        leverage_info = calc_leverage_recommendation(
            price=last, atr=atr, atr_pct=atr_pct,
            volatility_30d=vol_30d, funding_rate=funding,
            rsi=rsi, volume_ratio=vol_ratio,
            macd=macd, macd_signal=macd_signal, score=score,
            bb_upper=bb_upper, bb_lower=bb_lower, ema12=ema12, ema26=ema26
        )
        risk = calc_risk(last, atr, leverage_info)

        return {
            "symbol": sym,
            "company": display or sym.replace("USDT", ""),
            "market": "CRYPTO",
            "analysis_engine": "rules-based signal engine",
            "forecast_model_note": "예측 값은 규칙 기반 참고치이며 확정적 투자 신호가 아님",
            "last_close": round(last, 8),
            "prev_close": round(prev, 8),
            "pct_change": round(pct, 2),
            "rsi": round(rsi, 1),
            "volume": int(cur_vol),
            "volume_ratio": round(vol_ratio, 2),
            "atr": round(atr, 8),
            "atr_pct": round(atr_pct, 2),
            "volatility_30d": round(vol_30d, 1),
            "funding_rate": round(funding, 4) if funding is not None else None,
            "ticker_24h": ticker_24h,
            "score": score,
            "analysis_steps": steps,
            "candlestick_patterns": patterns,
            "chart_data": {
                "dates": dd.get("Date", []),
                "open": dd.get("Open", []),
                "high": dd.get("High", []),
                "low": dd.get("Low", []),
                "close": dd.get("Close", []),
                "volume": dd.get("Volume", []),
                "ma20": dd.get("MA20", []),
                "ma60": dd.get("MA60", []),
                "bb_upper": dd.get("BB_Upper", []),
                "bb_lower": dd.get("BB_Lower", []),
                "rsi": dd.get("RSI", []),
                "macd": dd.get("MACD", []),
                "signal_line": dd.get("Signal_Line", []),
                "vwap": dd.get("VWAP", []),
            },
            "forecast": forecast,
            "momentum_forecast": momentum,
            "xgb_forecast": momentum,
            "risk_scenarios": risk,
            "leverage_info": leverage_info,
            "news": news or [],
        }

    if path == "/api/screener":
        sort_by = params.get("sort_by", "volume")
        sort_order = params.get("sort_order", "desc")
        if sort_by not in config.SCREENER_VALID_SORT:
            sort_by = "volume"
        if sort_order not in config.SCREENER_VALID_ORDER:
            sort_order = "desc"
        return fetch_coin_screener(sort_by=sort_by, sort_order=sort_order)

    if path == "/api/sentiment":
        return fetch_market_sentiment(params.get("market", "CRYPTO"))

    if path == "/api/resolve":
        q = params.get("q", "")
        sym, display = resolve_symbol(q)
        return {"symbol": sym, "display": display} if sym else {"error": f"'{q}' 미발견"}

    if path == "/api/leverage":
        sym_raw = params.get("symbol", "BTCUSDT")
        symbol, _ = resolve_symbol(sym_raw)
        if not symbol:
            return {"error": "심볼 없음"}
        dd, _, _ = fetch_coin_data(symbol, interval="1d", limit=60)
        if not dd:
            return {"error": "데이터 없음"}
        def v(k):
            a = dd.get(k, [])
            val = a[-1] if a else None
            return float(val) if val is not None else 0.0
        closes = dd.get("Close", [])
        last = float(closes[-1]) if closes else 0
        vols = dd.get("Volume", [])
        cur_vol = float(vols[-1]) if vols else 0
        avg_vol = float(np.mean([x for x in vols[-20:] if x])) if vols else 1
        lev = calc_leverage_recommendation(price=last, atr=v("ATR"), atr_pct=v("ATR_Pct"), volatility_30d=v("Volatility_30d"), funding_rate=dd.get("funding_rate"), rsi=v("RSI"), volume_ratio=cur_vol/avg_vol if avg_vol > 0 else 1)
        return {"symbol": symbol, "leverage_info": lev}

    if path == "/api/validation":
        raw = params.get("ticker", params.get("symbol", "BTCUSDT"))
        interval = normalize_interval(params.get("interval", config.DEFAULT_INTERVAL))
        window = normalize_validation_window(params.get("window", str(config.VALIDATION_DEFAULT_WINDOW)))
        horizon = normalize_validation_horizon(params.get("horizon", str(config.VALIDATION_DEFAULT_HORIZON)))
        symbol, _ = resolve_symbol(raw)
        if not symbol:
            return {"error": "심볼 없음"}
        return build_validation_report(symbol=symbol, interval=interval, window=window, horizon=horizon)

    if path == "/api/health":
        return {
            "status": "ok",
            "service": config.APP_NAME,
            "version": config.APP_VERSION,
            "runtime": {"python": sys.version.split()[0], "serverless": bool(os.getenv("VERCEL"))},
            "cache": cache_meta(),
            "features": {"screener_batch": True, "validation_endpoint": True, "frontend": "public/index.html"},
        }

    if path == "/api/cron":
        warmed = []
        fetch_market_sentiment("CRYPTO")
        warmed.append("/api/sentiment?market=CRYPTO")
        fetch_coin_screener()
        warmed.append("/api/screener")
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            fetch_coin_data(symbol, interval="1d", limit=120)
            warmed.append(f"/api/coin?ticker={symbol}&interval=1d&limit=120")
        return {"status": "ok", "message": "warm paths executed", "warmed": warmed, "cache": cache_meta(), "cron_strategy": "15분 주기 캐시 워밍 + 짧은 TTL 병행"}

    return None

# =============================================================================
# ── HTML 프론트엔드 (기존 UI 구조 유지, 코인 특화 수정)
# =============================================================================
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoinOracle — Binance 규칙 기반 분석 시스템</title>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI','Noto Sans KR',sans-serif;display:flex;height:100vh;overflow:hidden}

/* 사이드바 */
#sidebar{width:260px;background:#161b22;border-right:1px solid #30363d;display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto}
.sb-header{padding:16px;border-bottom:1px solid #30363d}
.sb-header h1{font-size:15px;font-weight:700;display:flex;align-items:center;gap:6px}
.sb-header p{font-size:11px;color:#8b949e;margin-top:4px}
.sb-section{padding:14px;border-bottom:1px solid #30363d}
.sb-label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;display:block}
.mkt-btns{display:flex;gap:6px}
.mkt-btn{flex:1;padding:8px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500;transition:all .15s}
.mkt-btn.active{background:#1f6feb;color:#fff}
.mkt-btn:not(.active){background:#21262d;color:#8b949e}
.mkt-btn:not(.active):hover{background:#30363d;color:#e6edf3}
input,select{width:100%;background:#21262d;border:1px solid #30363d;border-radius:8px;padding:9px 12px;color:#e6edf3;font-size:13px;outline:none;transition:border-color .15s}
input:focus,select:focus{border-color:#1f6feb}
input::placeholder{color:#484f58}
#analyze-btn{width:100%;background:#1f6feb;color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:600;cursor:pointer;transition:background .15s;margin-top:4px}
#analyze-btn:hover{background:#388bfd}
#analyze-btn:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.sentiment-card{background:#21262d;border-radius:10px;padding:12px}
.sent-name{font-size:11px;color:#8b949e}
.sent-val{font-size:20px;font-weight:700;margin:3px 0}
.sent-chg{font-size:12px;font-weight:500}
.sent-badge{display:inline-block;margin-top:6px;font-size:10px;padding:2px 8px;border-radius:20px;background:#21262d;border:1px solid #30363d;color:#8b949e}
.sb-footer{padding:12px;margin-top:auto;border-top:1px solid #30363d}
.sb-footer p{font-size:10px;color:#484f58;line-height:1.5}

/* 메인 */
#main{flex:1;overflow-y:auto;background:#0d1117;padding:24px}

/* 로딩/빈 상태 */
.center-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;text-align:center}
.center-state .icon{font-size:56px}
.center-state h2{font-size:22px;font-weight:700}
.center-state p{color:#8b949e;font-size:14px;line-height:1.6;max-width:380px}
.sample-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;justify-content:center}
.sample-tag{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:6px 14px;font-size:13px;color:#8b949e;cursor:pointer}
.sample-tag:hover{background:#21262d;color:#e6edf3}
.spinner{width:40px;height:40px;border:4px solid #21262d;border-top-color:#1f6feb;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* 메트릭 카드 */
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}
.m-label{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.m-value{font-size:22px;font-weight:700}
.m-sub{font-size:12px;font-weight:500;margin-top:2px}
.rise{color:#3fb950}
.fall{color:#f85149}

/* 탭 */
.tabs{display:flex;gap:6px;border-bottom:1px solid #21262d;margin-bottom:16px;padding-bottom:2px}
.tab-btn{padding:7px 14px;border-radius:8px;border:none;background:none;color:#8b949e;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s}
.tab-btn.active{background:#1f6feb;color:#fff}
.tab-btn:not(.active):hover{background:#21262d;color:#e6edf3}

/* 카드 */
.card{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:18px;margin-bottom:14px}
.card-title{font-size:13px;font-weight:600;color:#8b949e;margin-bottom:14px;text-transform:uppercase;letter-spacing:.05em}

/* 헤더 */
.page-header{margin-bottom:20px}
.page-header h2{font-size:22px;font-weight:700;display:flex;align-items:center;gap:8px}
.ticker-badge{font-size:12px;font-weight:400;color:#8b949e;background:#21262d;padding:2px 8px;border-radius:6px;margin-left:4px}
.page-header p{font-size:12px;color:#484f58;margin-top:4px}

/* 코인 정보 그리드 (기존 펀더멘털 대체) */
.coin-info-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.fund-item{background:#21262d;border-radius:10px;padding:12px}
.fund-label{font-size:11px;color:#8b949e;margin-bottom:4px}
.fund-val{font-size:14px;font-weight:600}

/* 스코어 */
.score-wrap{display:flex;align-items:flex-end;gap:8px;margin-bottom:10px}
.score-num{font-size:52px;font-weight:800;line-height:1}
.score-bar-bg{background:#21262d;border-radius:6px;height:10px;overflow:hidden}
.score-bar-fill{height:10px;border-radius:6px;transition:width .6s ease}

/* 분석 스텝 */
.step-item{background:#21262d;border-radius:10px;padding:14px;margin-bottom:8px}
.step-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.step-title{font-size:13px;font-weight:600}
.step-score{font-size:12px;font-weight:700;padding:2px 8px;border-radius:12px}
.step-score.pos{background:#0d2d1a;color:#3fb950}
.step-score.neg{background:#2d0d0d;color:#f85149}
.step-score.neu{background:#21262d;color:#8b949e}
.step-result{font-size:13px;color:#8b949e;line-height:1.5}

/* 패턴 */
.pattern-item{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:8px;margin-bottom:6px;font-size:13px}
.pattern-bull{background:#0d2d1a;border:1px solid #1a4730}
.pattern-bear{background:#2d0d0d;border:1px solid #4d1515}
.pattern-neu{background:#21262d;border:1px solid #30363d}

/* 차트 */
#price-chart,#rsi-chart,#macd-chart{width:100%;border-radius:8px;overflow:hidden}

/* 레버리지 카드 (신규) */
.lev-card{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:18px;margin-bottom:14px}
.lev-main{display:flex;align-items:center;gap:20px;margin-bottom:16px}
.lev-num{font-size:64px;font-weight:900;line-height:1}
.lev-info{flex:1}
.lev-grade{font-size:14px;font-weight:700;padding:4px 12px;border-radius:20px;display:inline-block;margin-bottom:6px}
.lev-grade.Low{background:#0d2d1a;color:#3fb950}
.lev-grade.Medium{background:#2d200a;color:#d29922}
.lev-grade.High{background:#2d0d0d;color:#f85149}
.lev-grade.Extreme{background:#1a0000;color:#ff4444}
.lev-desc{font-size:13px;color:#8b949e;line-height:1.5}
.lev-factors{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.lev-factor{background:#21262d;border-radius:8px;padding:10px;text-align:center}
.lev-factor-label{font-size:10px;color:#8b949e;margin-bottom:4px}
.lev-factor-val{font-size:14px;font-weight:700}
.lev-warning{background:#2d0d0d;border:1px solid #4d1515;border-radius:10px;padding:12px;margin-top:12px;font-size:12px;color:#f85149;line-height:1.6}

/* 리스크 카드 */
.risk-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.risk-card{border-radius:12px;padding:16px;border:1px solid transparent}
.risk-card.conservative{background:#0a2d1a;border-color:#1a4730}
.risk-card.balanced{background:#2d200a;border-color:#4d3615}
.risk-card.aggressive{background:#2d0d0d;border-color:#4d1515}
.risk-icon{font-size:22px;margin-bottom:6px}
.risk-name{font-size:14px;font-weight:600;margin-bottom:4px}
.risk-desc{font-size:11px;color:#8b949e;margin-bottom:12px}
.risk-row{display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px}
.risk-lbl{color:#8b949e}
.risk-tgt{color:#3fb950;font-weight:700}
.risk-stp{color:#f85149;font-weight:700}
.risk-ratio{text-align:right;font-size:11px;color:#484f58;margin-top:8px;border-top:1px solid #30363d;padding-top:8px}

/* 뉴스 */
.news-item{display:flex;gap:10px;padding:10px 0;border-bottom:1px solid #21262d}
.news-item:last-child{border-bottom:none}
.news-dot{color:#388bfd;margin-top:2px;flex-shrink:0}
.news-a{color:#8b949e;font-size:13px;text-decoration:none;line-height:1.5}
.news-a:hover{color:#e6edf3;text-decoration:underline}
.news-meta{font-size:11px;color:#484f58;margin-top:3px}

/* 스크리너 */
.screener-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px}
.screener-table{width:100%;border-collapse:collapse;font-size:13px}
.screener-table th{padding:10px 14px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #30363d;cursor:pointer;white-space:nowrap}
.screener-table th:hover{color:#e6edf3}
.screener-table td{padding:12px 14px;border-bottom:1px solid #21262d;vertical-align:middle}
.screener-table tr:hover td{background:#161b22}
.ticker-name{font-weight:600}
.ticker-code{font-size:11px;color:#484f58;margin-top:2px}
.cat-badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#21262d;color:#8b949e}
.signal-badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600}
.sig-buy-strong{background:#0d2d1a;color:#3fb950}
.sig-buy{background:#0d2020;color:#238636}
.sig-neu{background:#2d2206;color:#d29922}
.sig-sell{background:#2d0d0d;color:#f85149}
.lev-badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:700}
.lev-low{background:#0d2d1a;color:#3fb950}
.lev-medium{background:#2d200a;color:#d29922}
.lev-high{background:#2d0d0d;color:#f85149}
.lev-extreme{background:#1a0000;color:#ff4444}

/* 반응형 */
@media(max-width:900px){
  .metrics-grid{grid-template-columns:repeat(2,1fr)}
  .risk-grid{grid-template-columns:1fr}
  .coin-info-grid{grid-template-columns:repeat(2,1fr)}
  .lev-factors{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:640px){
  #sidebar{width:220px}
  .metrics-grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>

<!-- ── 사이드바 ── -->
<div id="sidebar">
  <div class="sb-header">
    <h1>🪙 CoinOracle</h1>
    <p>Binance 규칙 기반 암호화폐 분석 시스템</p>
  </div>

  <div class="sb-section">
    <span class="sb-label">메뉴</span>
    <div style="display:flex;flex-direction:column;gap:4px">
      <button class="mkt-btn active" style="text-align:left;padding:10px 12px" id="nav-analysis" onclick="showPage('analysis')">🔍 코인 상세 분석</button>
      <button class="mkt-btn" style="text-align:left;padding:10px 12px" id="nav-screener" onclick="showPage('screener')">📋 코인 스크리너</button>
    </div>
  </div>

  <div class="sb-section">
    <span class="sb-label">🌍 BTC 시장 심리</span>
    <div id="sentiment-widget" class="sentiment-card">
      <div class="sent-name">로딩 중...</div>
    </div>
  </div>

  <div class="sb-section" id="analysis-controls">
    <span class="sb-label">시간 단위 (인터벌)</span>
    <div class="mkt-btns" style="margin-bottom:12px;flex-wrap:wrap;gap:4px">
      <button class="mkt-btn" id="iv-15m" onclick="setInterval('15m')">15분</button>
      <button class="mkt-btn" id="iv-1h"  onclick="setInterval('1h')">1시간</button>
      <button class="mkt-btn" id="iv-4h"  onclick="setInterval('4h')">4시간</button>
      <button class="mkt-btn active" id="iv-1d"  onclick="setInterval('1d')">일봉</button>
      <button class="mkt-btn" id="iv-1w"  onclick="setInterval('1w')">주봉</button>
    </div>
    <span class="sb-label">코인명 / 심볼</span>
    <input type="text" id="ticker-input" value="BTC" placeholder="예: BTC, ETH, 비트코인, SOLUSDT"
           style="margin-bottom:10px" onkeydown="if(event.key==='Enter')analyze()">
    <span class="sb-label">데이터 기간 (캔들 수)</span>
    <select id="period-select" style="margin-bottom:12px">
      <option value="90">90개</option>
      <option value="180">180개</option>
      <option value="365" selected>365개</option>
      <option value="500">500개</option>
    </select>
    <button id="analyze-btn" onclick="analyze()">🔍 분석 시작</button>
  </div>

  <div class="sb-footer">
    <p>⚠️ 본 시스템은 참고용이며, 투자 결정의 책임은 본인에게 있습니다.<br>레버리지 거래는 원금 손실 위험이 있습니다.</p>
  </div>
</div>

<!-- ── 메인 ── -->
<div id="main">
  <!-- 분석 페이지 -->
  <div id="page-analysis">
    <div id="state-empty" class="center-state">
      <div class="icon">🪙</div>
      <h2>CoinOracle — Binance 규칙 기반 분석</h2>
      <p>왼쪽 패널에서 코인명 또는 심볼을 입력하고<br><strong style="color:#388bfd">분석 시작</strong> 버튼을 누르세요.</p>
      <div class="sample-tags">
        <span class="sample-tag" onclick="quickSearch('BTC')">₿ BTC</span>
        <span class="sample-tag" onclick="quickSearch('ETH')">Ξ ETH</span>
        <span class="sample-tag" onclick="quickSearch('SOL')">◎ SOL</span>
        <span class="sample-tag" onclick="quickSearch('BNB')">⬡ BNB</span>
        <span class="sample-tag" onclick="quickSearch('XRP')">✕ XRP</span>
        <span class="sample-tag" onclick="quickSearch('DOGE')">Ð DOGE</span>
      </div>
    </div>
    <div id="state-loading" class="center-state" style="display:none">
      <div class="spinner"></div>
      <p style="color:#8b949e">Binance 데이터 수집 중...<br><span style="font-size:12px;color:#484f58">규칙 기반 엔진이 기술적 지표를 계산하고 있습니다</span></p>
    </div>
    <div id="state-error" class="center-state" style="display:none">
      <div class="icon">⚠️</div>
      <p id="error-msg" style="color:#f85149"></p>
    </div>

    <div id="state-result" style="display:none">
      <div class="page-header">
        <h2 id="r-title"></h2>
        <p id="r-subtitle"></p>
      </div>

      <!-- 메트릭 카드 -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="m-label">현재가 (USDT)</div>
          <div class="m-value" id="r-price"></div>
          <div class="m-sub" id="r-pct"></div>
        </div>
        <div class="metric-card">
          <div class="m-label">RSI (14)</div>
          <div class="m-value" id="r-rsi"></div>
          <div class="m-sub" id="r-rsi-label"></div>
        </div>
        <div class="metric-card">
          <div class="m-label">24h 거래량 (코인)</div>
          <div class="m-value" style="font-size:18px" id="r-vol"></div>
          <div class="m-sub" id="r-vol-ratio"></div>
        </div>
        <div class="metric-card">
          <div class="m-label">ATR / 변동성</div>
          <div class="m-value" style="font-size:18px" id="r-atr"></div>
          <div class="m-sub" id="r-atr-pct"></div>
        </div>
      </div>

      <!-- 코인 정보 (기존 펀더멘털 대체) -->
      <div class="card" id="r-coin-info">
        <div class="card-title">📊 코인 시장 정보</div>
        <div class="coin-info-grid">
          <div class="fund-item">
            <div class="fund-label">24h 고가</div>
            <div class="fund-val" id="f-high24h">-</div>
          </div>
          <div class="fund-item">
            <div class="fund-label">24h 저가</div>
            <div class="fund-val" id="f-low24h">-</div>
          </div>
          <div class="fund-item">
            <div class="fund-label">24h 거래대금 (USDT)</div>
            <div class="fund-val" id="f-volume-usdt">-</div>
          </div>
          <div class="fund-item">
            <div class="fund-label">선물 펀딩비</div>
            <div class="fund-val" id="f-funding">-</div>
          </div>
          <div class="fund-item">
            <div class="fund-label">30일 변동성 (연율)</div>
            <div class="fund-val" id="f-vol30d">-</div>
          </div>
          <div class="fund-item">
            <div class="fund-label">거래량 비율 (vs 20MA)</div>
            <div class="fund-val" id="f-volratio">-</div>
          </div>
          <div class="fund-item">
            <div class="fund-label">ATR (%)</div>
            <div class="fund-val" id="f-atrpct">-</div>
          </div>
          <div class="fund-item">
            <div class="fund-label">24h 체결 건수</div>
            <div class="fund-val" id="f-trades">-</div>
          </div>
        </div>
      </div>

      <!-- 탭 -->
      <div class="tabs">
        <button class="tab-btn active" id="tab-chart"    onclick="switchTab('chart')">📈 차트</button>
        <button class="tab-btn"        id="tab-ai"       onclick="switchTab('ai')">🧭 시그널 진단</button>
        <button class="tab-btn"        id="tab-leverage" onclick="switchTab('leverage')">⚡ 레버리지</button>
        <button class="tab-btn"        id="tab-forecast" onclick="switchTab('forecast')">🔮 예측</button>
        <button class="tab-btn"        id="tab-news"     onclick="switchTab('news')">📰 뉴스</button>
      </div>

      <!-- 차트 탭 -->
      <div id="tab-content-chart">
        <div class="card">
          <div class="card-title">가격 차트 (캔들스틱)</div>
          <div id="price-chart" style="height:320px"></div>
        </div>
        <div class="card">
          <div class="card-title">RSI (14)</div>
          <div id="rsi-chart" style="height:160px"></div>
        </div>
        <div class="card">
          <div class="card-title">MACD</div>
          <div id="macd-chart" style="height:160px"></div>
        </div>
      </div>

      <!-- AI 진단 탭 -->
      <div id="tab-content-ai" style="display:none">
        <div class="card">
          <div class="card-title">🧭 종합 점수</div>
          <div class="score-wrap">
            <div class="score-num" id="ai-score">-</div>
            <div style="flex:1">
              <div class="score-bar-bg"><div class="score-bar-fill" id="ai-score-bar" style="width:0%"></div></div>
              <div style="font-size:13px;color:#8b949e;margin-top:6px" id="ai-score-desc"></div>
            </div>
          </div>
          <div id="steps-list"></div>
        </div>
        <div class="card">
          <div class="card-title">🕯️ 캔들 패턴 감지</div>
          <div id="patterns-list"></div>
        </div>
      </div>

      <!-- 레버리지 탭 (신규) -->
      <div id="tab-content-leverage" style="display:none">
        
        <!-- 매수 타점 및 예측 근거 (신규) -->
        <div id="lev-buy-section"></div>

        <div class="lev-card">
          <div class="card-title">⚡ 레버리지 예측 엔진</div>
          <div class="lev-main">
            <div>
              <div style="font-size:12px;color:#8b949e;margin-bottom:4px">추천 포지션</div>
              <div class="lev-num" id="lev-pos" style="font-size:42px">-</div>
            </div>
            <div>
              <div style="font-size:12px;color:#8b949e;margin-bottom:4px">추천 레버리지</div>
              <div class="lev-num" id="lev-num" style="color:#1f6feb">-x</div>
            </div>
            <div class="lev-info">
              <div class="lev-grade" id="lev-grade">-</div>
              <div class="lev-desc" id="lev-desc"></div>
              <div class="lev-desc" id="lev-pos-desc" style="margin-top:4px;color:#8b949e;font-size:12px"></div>
            </div>
          </div>
          <div class="lev-factors" id="lev-factors"></div>
          <div class="lev-warning" id="lev-warning"></div>
        </div>

        <div class="card">
          <div class="card-title">🛡️ 리스크 시나리오</div>
          <div class="risk-grid" id="risk-grid"></div>
        </div>
      </div>

      <!-- 예측 탭 -->
      <div id="tab-content-forecast" style="display:none">
        <div class="card">
          <div class="card-title">🔮 참고 시나리오 예측 (30일)</div>
          <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px" id="forecast-summary"></div>
        </div>
      </div>

      <!-- 뉴스 탭 -->
      <div id="tab-content-news" style="display:none">
        <div class="card">
          <div class="card-title">📰 관련 뉴스</div>
          <div id="news-list"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- 스크리너 페이지 -->
  <div id="page-screener" style="display:none">
    <div class="screener-header">
      <div>
        <h2 style="font-size:20px;font-weight:700;margin-bottom:6px">📋 코인 스크리너</h2>
        <p style="font-size:12px;color:#484f58">Binance 주요 코인 기술적 분석 스크리너 | 실시간 데이터</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <span style="font-size:12px;color:#484f58" id="scrn-count"></span>
        <button onclick="loadScreener()" style="background:#21262d;border:1px solid #30363d;color:#8b949e;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:12px">🔄 새로고침</button>
      </div>
    </div>

    <!-- 필터 조건 표시 -->
    <div class="card" style="margin-bottom:16px" id="scrn-filter-card">
      <div class="card-title">🔍 스크리너 기준</div>
      <div id="scrn-filter-conds" style="display:flex;flex-wrap:wrap;gap:8px;font-size:12px;color:#8b949e"></div>
    </div>

    <div class="card" style="padding:0;overflow:hidden">
      <table class="screener-table">
        <thead>
          <tr>
            <th>#</th>
            <th onclick="sortScreener('name')">코인</th>
            <th onclick="sortScreener('price')" style="text-align:right">현재가</th>
            <th onclick="sortScreener('change')" style="text-align:right">24h 변동</th>
            <th>카테고리</th>
            <th onclick="sortScreener('volume')" style="text-align:right">24h 거래량</th>
            <th onclick="sortScreener('rsi')" style="text-align:center">RSI</th>
            <th onclick="sortScreener('leverage')" style="text-align:center">추천 레버리지</th>
            <th style="text-align:center">신호</th>
          </tr>
        </thead>
        <tbody id="scrn-tbody">
          <tr><td colspan="9" style="text-align:center;padding:40px;color:#8b949e">로딩 중...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
// ── 상태 ──
let currentData = null;
let currentInterval = '1d';
let screenerData = [];
let scrnSort = {key:'volume', dir:'desc'};
let chartInstances = {};

// ── 페이지 전환 ──
function showPage(page) {
  document.getElementById('page-analysis').style.display = page === 'analysis' ? 'block' : 'none';
  document.getElementById('page-screener').style.display = page === 'screener' ? 'block' : 'none';
  document.getElementById('analysis-controls').style.display = page === 'analysis' ? 'block' : 'none';
  document.getElementById('nav-analysis').classList.toggle('active', page === 'analysis');
  document.getElementById('nav-screener').classList.toggle('active', page === 'screener');
  if (page === 'screener' && screenerData.length === 0) loadScreener();
}

// ── 인터벌 선택 ──
function setInterval(iv) {
  currentInterval = iv;
  ['15m','1h','4h','1d','1w'].forEach(i => {
    document.getElementById('iv-' + i).classList.toggle('active', i === iv);
  });
}

function quickSearch(name) {
  document.getElementById('ticker-input').value = name;
  analyze();
}

// ── BTC 시장 심리 ──
async function loadSentiment() {
  const w = document.getElementById('sentiment-widget');
  w.innerHTML = '<div class="sent-name">로딩 중...</div>';
  try {
    const r = await fetch('/api/sentiment?market=CRYPTO');
    const d = await r.json();
    if (d.error) { w.innerHTML = '<div class="sent-name" style="color:#484f58">조회 실패</div>'; return; }
    const isUp = d.change >= 0;
    const clr = isUp ? '#3fb950' : '#f85149';
    w.innerHTML = `
      <div class="sent-name">${d.name}</div>
      <div class="sent-val">$${d.value.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
      <div class="sent-chg" style="color:${clr}">${isUp?'▲':'▼'} ${Math.abs(d.change).toFixed(2)}%</div>
      <span class="sent-badge">${d.sentiment}</span>`;
  } catch(e) { w.innerHTML = '<div class="sent-name" style="color:#484f58">조회 실패</div>'; }
}

// ── 분석 ──
async function analyze() {
  const ticker = document.getElementById('ticker-input').value.trim();
  const limit  = document.getElementById('period-select').value;
  if (!ticker) return;
  setState('loading');
  document.getElementById('analyze-btn').disabled = true;
  destroyCharts();
  try {
    const r = await fetch(`/api/coin?ticker=${encodeURIComponent(ticker)}&interval=${currentInterval}&limit=${limit}`);
    const d = await r.json();
    if (d.error) { setState('error'); document.getElementById('error-msg').textContent = d.error; return; }
    currentData = d;
    renderResult(d);
    setState('result');
  } catch(e) {
    setState('error');
    document.getElementById('error-msg').textContent = 'API 서버 오류: ' + e.message;
  } finally {
    document.getElementById('analyze-btn').disabled = false;
  }
}

function setState(s) {
  ['empty','loading','error','result'].forEach(n => {
    const el = document.getElementById('state-' + n);
    if (el) el.style.display = n === s ? (s === 'result' ? 'block' : 'flex') : 'none';
  });
}

// ── 가격 포맷 (코인 소수점 처리) ──
function fmtPrice(v) {
  if (v >= 1000)   return '$' + v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  if (v >= 1)      return '$' + v.toFixed(4);
  if (v >= 0.001)  return '$' + v.toFixed(6);
  return '$' + v.toFixed(8);
}

function fmtUsd(v) {
  return Number(v || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtVol(v) {
  if (v >= 1e9) return (v/1e9).toFixed(2) + 'B';
  if (v >= 1e6) return (v/1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v/1e3).toFixed(2) + 'K';
  return v.toFixed(0);
}

// ── 렌더링 ──
function renderResult(d) {
  const up  = d.pct_change >= 0;
  const clr = up ? '#3fb950' : '#f85149';

  document.getElementById('r-title').innerHTML =
    `${d.company || d.symbol} <span class="ticker-badge">${d.symbol}</span>`;
  document.getElementById('r-subtitle').textContent =
    `기준: ${new Date().toLocaleString('ko-KR')} | 거래소: Binance | 인터벌: ${currentInterval}`;
  document.getElementById('r-price').textContent = fmtPrice(d.last_close);
  document.getElementById('r-pct').innerHTML = `<span style="color:${clr}">${up?'▲':'▼'} ${Math.abs(d.pct_change).toFixed(2)}%</span>`;

  const rsi = d.rsi;
  const rsiClr = rsi > 70 ? '#f85149' : rsi < 30 ? '#3fb950' : '#e6edf3';
  document.getElementById('r-rsi').innerHTML = `<span style="color:${rsiClr}">${rsi.toFixed(1)}</span>`;
  document.getElementById('r-rsi-label').innerHTML = `<span style="color:${rsiClr}">${rsi>70?'과매수':rsi<30?'과매도':'중립'}</span>`;
  document.getElementById('r-vol').textContent = fmtVol(d.volume);
  document.getElementById('r-vol-ratio').innerHTML = `<span style="color:${d.volume_ratio>1.5?'#3fb950':'#8b949e'}">${d.volume_ratio.toFixed(2)}x (20MA 대비)</span>`;
  document.getElementById('r-atr').textContent = fmtPrice(d.atr);
  document.getElementById('r-atr-pct').innerHTML = `<span style="color:#d29922">±${d.atr_pct.toFixed(2)}% / 일</span>`;

  // 코인 정보
  const t24 = d.ticker_24h || {};
  document.getElementById('f-high24h').textContent   = t24.high_24h  ? fmtPrice(t24.high_24h)  : '-';
  document.getElementById('f-low24h').textContent    = t24.low_24h   ? fmtPrice(t24.low_24h)   : '-';
  document.getElementById('f-volume-usdt').textContent = t24.volume_24h ? '$' + fmtVol(t24.volume_24h) : '-';
  const fr = d.funding_rate;
  document.getElementById('f-funding').innerHTML = fr != null
    ? `<span style="color:${Math.abs(fr)>0.05?'#f85149':fr>0?'#d29922':'#3fb950'}">${fr>0?'+':''}${fr.toFixed(4)}%</span>`
    : '<span style="color:#484f58">선물 없음</span>';
  document.getElementById('f-vol30d').innerHTML = d.volatility_30d
    ? `<span style="color:${d.volatility_30d>100?'#f85149':d.volatility_30d>60?'#d29922':'#3fb950'}">${d.volatility_30d.toFixed(1)}%</span>`
    : '-';
  document.getElementById('f-volratio').innerHTML = `<span style="color:${d.volume_ratio>1.5?'#3fb950':'#8b949e'}">${d.volume_ratio.toFixed(2)}x</span>`;
  document.getElementById('f-atrpct').innerHTML = `<span style="color:#d29922">${d.atr_pct.toFixed(2)}%</span>`;
  document.getElementById('f-trades').textContent = t24.count ? t24.count.toLocaleString() : '-';

  renderAI(d);
  renderLeverage(d);
  renderForecast(d);
  renderNews(d);
  switchTab('chart');
  setTimeout(() => renderCharts(d), 50);
}

function renderAI(d) {
  const s = d.score;
  const sClr = s >= 70 ? '#3fb950' : s >= 40 ? '#d29922' : '#f85149';
  document.getElementById('ai-score').innerHTML = `<span style="color:${sClr}">${s}</span>`;
  const bar = document.getElementById('ai-score-bar');
  bar.style.width = s + '%'; bar.style.background = sClr;
  document.getElementById('ai-score-desc').textContent =
    s >= 70 ? '✅ 상승 우위 — 매수 신호 강함'
    : s >= 40 ? '⚖️ 중립 — 추세 확인 필요'
    : '⚠️ 하락 우위 — 리스크 주의';

  const stepsList = document.getElementById('steps-list');
  stepsList.innerHTML = d.analysis_steps.map(st => {
    const sc = st.score;
    const cls = sc > 0 ? 'pos' : sc < 0 ? 'neg' : 'neu';
    const label = sc > 0 ? '+' + sc : sc;
    return `<div class="step-item">
      <div class="step-header">
        <span class="step-title">${st.step}</span>
        <span class="step-score ${cls}">${label}점</span>
      </div>
      <div class="step-result">${st.result}</div>
    </div>`;
  }).join('');

  const patList = document.getElementById('patterns-list');
  if (!d.candlestick_patterns || d.candlestick_patterns.length === 0) {
    patList.innerHTML = '<p style="font-size:13px;color:#484f58">특이한 캔들 패턴이 감지되지 않았습니다.</p>';
  } else {
    patList.innerHTML = d.candlestick_patterns.map(p => {
      const cls = p.direction === '상승' ? 'pattern-bull' : p.direction === '하락' ? 'pattern-bear' : 'pattern-neu';
      const icon = p.direction === '상승' ? '📈' : p.direction === '하락' ? '📉' : '➖';
      return `<div class="pattern-item ${cls}">
        <span>${icon} <strong>${p.name}</strong></span>
        <span style="font-size:12px;color:#8b949e">${p.desc}</span>
      </div>`;
    }).join('');
  }
}

// ── 레버리지 렌더링 (신규) ──
function renderLeverage(d) {
  const lev  = d.leverage_info;
  const risk = d.risk_scenarios;
  if (!lev) return;

  // 레버리지 숫자
  const levEl = document.getElementById('lev-num');
  levEl.textContent = lev.recommended_leverage + 'x';
  levEl.style.color = lev.risk_color;

  // 포지션 방향
  const posEl = document.getElementById('lev-pos');
  if(posEl) {
    posEl.textContent = lev.position || '-';
    posEl.style.color = lev.position === 'Long' ? '#3fb950' : (lev.position === 'Short' ? '#f85149' : '#8b949e');
  }
  const posDescEl = document.getElementById('lev-pos-desc');
  if(posDescEl) {
    posDescEl.textContent = lev.position_desc ? `방향성: ${lev.position_desc}` : '';
  }

  // 위험도 등급
  const gradeEl = document.getElementById('lev-grade');
  gradeEl.textContent = lev.risk_grade;
  gradeEl.className = 'lev-grade ' + lev.risk_grade;

  document.getElementById('lev-desc').textContent = lev.risk_desc;

  // 계산 근거 팩터
  const f = lev.factors || {};
  document.getElementById('lev-factors').innerHTML = `
    <div class="lev-factor">
      <div class="lev-factor-label">ATR (%)</div>
      <div class="lev-factor-val" style="color:#d29922">${f.atr_pct || '-'}%</div>
    </div>
    <div class="lev-factor">
      <div class="lev-factor-label">30일 변동성</div>
      <div class="lev-factor-val" style="color:${(f.volatility_30d||0)>80?'#f85149':'#d29922'}">${f.volatility_30d || '-'}%</div>
    </div>
    <div class="lev-factor">
      <div class="lev-factor-label">펀딩비</div>
      <div class="lev-factor-val" style="color:${Math.abs(f.funding_rate||0)>0.05?'#f85149':'#3fb950'}">${f.funding_rate != null ? f.funding_rate + '%' : 'N/A'}</div>
    </div>
    <div class="lev-factor">
      <div class="lev-factor-label">기본 레버리지</div>
      <div class="lev-factor-val">${f.base_leverage || '-'}x</div>
    </div>
    <div class="lev-factor">
      <div class="lev-factor-label">변동성 계수</div>
      <div class="lev-factor-val">${f.vol_factor || '-'}</div>
    </div>
    <div class="lev-factor">
      <div class="lev-factor-label">RSI 계수</div>
      <div class="lev-factor-val">${f.rsi_factor || '-'}</div>
    </div>`;

  // 경고 메시지
  document.getElementById('lev-warning').innerHTML = `
    ⚠️ <strong>레버리지 사용 주의사항</strong><br>
    • 최대 허용 레버리지: <strong>${lev.max_leverage}x</strong> (절대 초과 금지)<br>
    • 권장 포지션 크기: 총 자산의 <strong>${lev.position_size_pct}%</strong> 이하<br>
    • 레버리지 ${lev.recommended_leverage}x 적용 시 손절 손실: <strong>${lev.lev_stop_loss_pct}%</strong><br>
    • 고변동성 구간에서는 레버리지를 즉시 축소하세요.`;

  // 매수 타점 및 예측 근거 렌더링 (신규 UI)
  const levBuySection = document.getElementById('lev-buy-section');
  if (levBuySection) {
    const last = d.last_close || 0;
    const atr = d.atr || 0;
    const aggR = [last - 0.8 * atr, last - 0.2 * atr];
    const recR = [last - 1.5 * atr, last - 0.8 * atr];

    // 데이터 추출
    const cd = d.chart_data || {};
    const bbl = cd.bb_lower && cd.bb_lower.length > 0 ? cd.bb_lower[cd.bb_lower.length - 1] : 0;
    const ma20 = cd.ma20 && cd.ma20.length > 0 ? cd.ma20[cd.ma20.length - 1] : 0;
    const ma60 = cd.ma60 && cd.ma60.length > 0 ? cd.ma60[cd.ma60.length - 1] : 0;
    const low30d = cd.low && cd.low.length >= 30 ? Math.min(...cd.low.slice(-30)) : (last * 0.9);
    
    // AI 시그널 매수 개수
    const steps = d.analysis_steps || [];
    const buySignals = steps.filter(s => s.score > 0).length;
    const totalSignals = steps.length;

    levBuySection.innerHTML = `
      <div class="buy-price-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
        <div class="buy-card aggressive" style="border-radius:12px;padding:16px;border:1px solid #4d3615;background:#2d200a;">
          <div class="buy-label" style="font-size:11px;color:#f97316;margin-bottom:8px;font-weight:600;display:flex;align-items:center;gap:6px;">⚡ 공격적 매수</div>
          <div class="buy-price-val" style="color:#f97316;font-size:20px;font-weight:800;margin-bottom:12px;">${fmtPrice(aggR[0])} ~ ${fmtPrice(aggR[1])}</div>
          <div class="buy-basis-box" style="font-size:12px;color:#8b949e;line-height:1.5;border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">현재가 대비 단기 눌림 구간<br>ATR 0.5배 기반 · 빠른 진입</div>
        </div>
        <div class="buy-card recommended" style="border-radius:12px;padding:16px;border:1px solid #1a4730;background:#0d2d1a;">
          <div class="buy-label" style="font-size:11px;color:#3fb950;margin-bottom:8px;font-weight:600;display:flex;align-items:center;gap:6px;">✅ 추천 매수 구간</div>
          <div class="buy-price-val" style="color:#3fb950;font-size:20px;font-weight:800;margin-bottom:12px;">${fmtPrice(recR[0])} ~ ${fmtPrice(recR[1])}</div>
          <div class="buy-basis-box" style="font-size:12px;color:#8b949e;line-height:1.5;border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">ATR 기반 최적 진입 구간<br>분할 매수 권장</div>
        </div>
      </div>
      <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:16px;">
        <div style="font-size:13px;color:#e6edf3;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:6px;">📋 예측 근거</div>
        <div style="font-size:13px;color:#8b949e;line-height:1.8;padding-left:4px;">
          • 볼린저 하단 지지선 근접 (≈${fmtPrice(bbl)})<br>
          • 단기 생명선(MA20) 지지 (≈${fmtPrice(ma20)})<br>
          • 중기 추세선(MA60) 지지 (≈${fmtPrice(ma60)})<br>
          • 일간 변동성(ATR) 기반 구간 분할 (ATR ≈${fmtPrice(atr)})<br>
          • 최근 30일 핵심 매물대/저점 지지구간 (≈${fmtPrice(low30d)})<br>
          • AI 종합 진단 점수 (${d.score}점) 및 시그널(${buySignals}/${totalSignals} 매수) 반영
        </div>
      </div>
    `;
  }

  window.currentTradingSignals = lev.trading_signals;
  window.currentRiskData = {
    price: d.last_close,
    position: lev.position,
    recommended_leverage: lev.recommended_leverage,
    max_leverage: lev.max_leverage,
    stop_loss_pct: lev.stop_loss_pct,
    take_profit_pct: lev.take_profit_pct,
    atr: d.atr,
    position_size_pct: lev.position_size_pct,
    risk_grade: lev.risk_grade,
    volatility: lev.factors ? lev.factors.volatility_30d : 50
  };
  updateTradingSignals();
}

function getLeverageBucket(leverage) {
  if (leverage >= 8) return 'high';
  if (leverage >= 4) return 'medium';
  return 'low';
}

function getPositionContext(position) {
  if (position === 'Short' || position === 'Sell') {
    return {
      label: 'Short',
      color: '#f85149',
      biasText: '하락 추세 대응',
      adverseMove: '상승 반등',
      favourableMove: '하락 가속'
    };
  }
  if (position === 'Long' || position === 'Buy') {
    return {
      label: 'Long',
      color: '#3fb950',
      biasText: '상승 추세 추종',
      adverseMove: '하락 조정',
      favourableMove: '상승 확장'
    };
  }
  return {
    label: 'Neutral',
    color: '#8b949e',
    biasText: '방향성 대기',
    adverseMove: '양방향 급변동',
    favourableMove: '확정 추세 대기'
  };
}

function getScenarioConfigs(position, recommendedLev, maxLev) {
  const bucket = getLeverageBucket(recommendedLev);
  const isShort = position === 'Short' || position === 'Sell';
  const isLong = position === 'Long' || position === 'Buy';
  const notes = {
    low: isShort
      ? ['완만한 하락 구간에서 반등 리스크를 최소화하는 운용', '권장 레버리지 중심의 기본 공매도 운용', '하락 가속을 노리되 반등 시 빠른 축소가 필요한 운용']
      : isLong
        ? ['완만한 상승 구간에서 조정 폭을 작게 가져가는 운용', '권장 레버리지 중심의 기본 추세 추종 운용', '상승 연장을 노리되 조정 시 손절 관리가 필요한 운용']
        : ['방향성 확인 전 자본 노출을 낮추는 대기 운용', '중립 구간에서 이벤트 대응용 기본 운용', '돌파 대응을 염두에 둔 고위험 대기 운용'],
    medium: isShort
      ? ['중간 레버리지 공매도에서 급반등을 방어하는 운용', '추천 레버리지 기준으로 추세 하락을 추종하는 운용', '추세 하락이 이어질 때 수익 극대화를 노리는 운용']
      : isLong
        ? ['중간 레버리지 매수에서 변동성 확장을 방어하는 운용', '추천 레버리지 기준으로 추세 상승을 추종하는 운용', '추세 상승이 이어질 때 수익 극대화를 노리는 운용']
        : ['중립 구간에서 추세 확인 전 노출을 제한하는 운용', '중립 구간에서 조건부 진입을 준비하는 운용', '방향성 확정 전 단기 이벤트 대응 운용'],
    high: isShort
      ? ['고레버리지 공매도에서 반등 폭을 최소화하기 위한 방어 운용', '추천 레버리지 기반의 고위험 추세 추종 운용', '최대 레버리지에 가깝게 하락 가속을 노리는 운용']
      : isLong
        ? ['고레버리지 매수에서 급락 리스크를 우선 제어하는 운용', '추천 레버리지 기반의 고위험 추세 추종 운용', '최대 레버리지에 가깝게 상승 연장을 노리는 운용']
        : ['고변동성 중립 구간에서 최소 노출로 대기하는 운용', '중립 상태에서 신호 확인 전 단기 대응 운용', '방향성 확정 전 이벤트성 변동성 대응 운용']
  };
  const profileByBucket = {
    low: {
      levFactors: [0.8, 1.0, 1.35],
      capitalRatios: [0.02, 0.045, 0.07],
      stopFactors: [0.85, 1.0, 1.15]
    },
    medium: {
      levFactors: [0.75, 1.0, 1.3],
      capitalRatios: [0.02, 0.05, 0.08],
      stopFactors: [0.9, 1.05, 1.2]
    },
    high: {
      levFactors: [0.65, 1.0, 1.15],
      capitalRatios: [0.015, 0.035, 0.06],
      stopFactors: [0.95, 1.1, 1.3]
    }
  };
  const profile = profileByBucket[bucket];
  return [
    {
      key: 'conservative',
      name: '안전 (Safe)',
      icon: '🛡️',
      leverage: Math.max(1, Math.min(maxLev, Math.round(recommendedLev * profile.levFactors[0]))),
      capitalRatio: profile.capitalRatios[0],
      stopFactor: profile.stopFactors[0],
      desc: notes[bucket][0]
    },
    {
      key: 'balanced',
      name: '중간 (Balanced)',
      icon: '⚖️',
      leverage: Math.max(1, Math.min(maxLev, Math.round(recommendedLev * profile.levFactors[1]))),
      capitalRatio: profile.capitalRatios[1],
      stopFactor: profile.stopFactors[1],
      desc: notes[bucket][1]
    },
    {
      key: 'aggressive',
      name: '공격 (Aggressive)',
      icon: '🚀',
      leverage: Math.max(1, Math.min(maxLev, Math.round(recommendedLev * profile.levFactors[2]))),
      capitalRatio: profile.capitalRatios[2],
      stopFactor: profile.stopFactors[2],
      desc: notes[bucket][2]
    }
  ];
}

function getVolatilityRiskLabel(volatility, leverageRatio) {
  const score = (volatility || 0) + (leverageRatio * 100);
  if (score >= 150) return '높음';
  if (score >= 95) return '보통';
  return '낮음';
}

function calcLiquidationDisplay(entryPrice, position, leverage, volatility) {
  const buffer = volatility >= 100 ? 0.84 : volatility >= 70 ? 0.89 : 0.93;
  const longPrice = entryPrice * (1 - buffer / leverage);
  const shortPrice = entryPrice * (1 + buffer / leverage);
  if (position === 'Neutral') {
    return `${fmtPrice(longPrice)} / ${fmtPrice(shortPrice)}`;
  }
  return position === 'Short' ? fmtPrice(shortPrice) : fmtPrice(longPrice);
}

function formatLossRange(minAmount, maxAmount, minPct, maxPct) {
  return `-${fmtUsd(minAmount)} ~ -${fmtUsd(maxAmount)} (${minPct.toFixed(1)}% ~ ${maxPct.toFixed(1)}%)`;
}

function updateTradingSignals() {
  const ts = window.currentTradingSignals;
  const gridEl = document.getElementById('trading-signals-grid');
  if (!ts || !gridEl) {
    updateRiskScenario();
    return;
  }

  const investInput = document.getElementById('invest-amount');
  const investAmt = parseFloat(investInput ? investInput.value : 0) || 0;

  const safeAmt = investAmt * 0.02;
  const midAmt = investAmt * 0.05;
  const aggAmt = investAmt * 0.10;

  const fmtAmt = (val) => {
    if (val === 0) return '0.00';
    return val.toFixed(2);
  };

  const renderCard = (type, data, title, color) => {
    const tpPrice = data.tp || 0;
    const coinQtySafe = data.limit > 0 ? (safeAmt * data.leverage / data.limit).toFixed(4) : '0';
    const coinQtyMid = data.limit > 0 ? (midAmt * data.leverage / data.limit).toFixed(4) : '0';
    const coinQtyAgg = data.limit > 0 ? (aggAmt * data.leverage / data.limit).toFixed(4) : '0';

    return `
      <div style="background:#21262d; border:1px solid #30363d; border-radius:8px; padding:16px;">
        <div style="font-size:16px; font-weight:700; color:${color}; margin-bottom:12px; border-bottom:1px solid #30363d; padding-bottom:8px;">
          ${title}
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; margin-bottom:12px;">
          <div style="background:#0d1117; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">Stop (트리거)</div>
            <div style="font-size:14px; font-weight:600; color:#c9d1d9;">${fmtPrice(data.stop)}</div>
          </div>
          <div style="background:#0d1117; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">Limit (주문가)</div>
            <div style="font-size:14px; font-weight:600; color:#c9d1d9;">${fmtPrice(data.limit)}</div>
          </div>
          <div style="background:#0d1117; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">Stop Loss</div>
            <div style="font-size:14px; font-weight:600; color:#f85149;">${fmtPrice(data.sl)}</div>
          </div>
          <div style="background:#0d1117; padding:10px; border-radius:6px; text-align:center; border:1px solid #3fb95033;">
            <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">Take Profit (목표가)</div>
            <div style="font-size:14px; font-weight:600; color:#3fb950;">${fmtPrice(tpPrice)}</div>
          </div>
        </div>
        <div style="font-size:13px; font-weight:600; color:#c9d1d9; margin-bottom:8px;">
          💰 Amount (주문 규모) - 레버리지 ${data.leverage}x
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;">
          <div style="background:#161b22; border:1px solid #3fb950; padding:8px; border-radius:6px; text-align:center;">
            <div style="font-size:11px; color:#3fb950; margin-bottom:2px;">안전형 (마진 $${fmtAmt(safeAmt)})</div>
            <div style="font-size:14px; font-weight:600; color:#fff;">$${fmtAmt(safeAmt * data.leverage)}</div>
            <div style="font-size:10px; color:#8b949e; margin-top:2px;">≈ ${coinQtySafe} 개</div>
          </div>
          <div style="background:#161b22; border:1px solid #d29922; padding:8px; border-radius:6px; text-align:center;">
            <div style="font-size:11px; color:#d29922; margin-bottom:2px;">중간형 (마진 $${fmtAmt(midAmt)})</div>
            <div style="font-size:14px; font-weight:600; color:#fff;">$${fmtAmt(midAmt * data.leverage)}</div>
            <div style="font-size:10px; color:#8b949e; margin-top:2px;">≈ ${coinQtyMid} 개</div>
          </div>
          <div style="background:#161b22; border:1px solid #f85149; padding:8px; border-radius:6px; text-align:center;">
            <div style="font-size:11px; color:#f85149; margin-bottom:2px;">공격형 (마진 $${fmtAmt(aggAmt)})</div>
            <div style="font-size:14px; font-weight:600; color:#fff;">$${fmtAmt(aggAmt * data.leverage)}</div>
            <div style="font-size:10px; color:#8b949e; margin-top:2px;">≈ ${coinQtyAgg} 개</div>
          </div>
        </div>
      </div>
    `;
  };

  let html = '';
  if (ts.long) {
    html += renderCard('long', ts.long, '📈 Long (상승장 진입)', '#3fb950');
  }
  if (ts.short) {
    html += renderCard('short', ts.short, '📉 Short (하락장 진입)', '#f85149');
  }

  // 매도 타이밍 요약 추가 (한 눈에 보이도록)
  if (window.currentSellPrediction) {
    const sp = window.currentSellPrediction;
    html += `
      <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-top:4px;">
        <div style="font-size:14px; font-weight:700; color:#e6edf3; margin-bottom:12px; display:flex; align-items:center; gap:6px;">
          ⏱️ 스마트 매도/청산 타이밍 예측
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
          <div style="background:#0d1117; padding:12px; border-radius:8px; text-align:center;">
            <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">예상 매도 날짜</div>
            <div style="font-size:15px; font-weight:700; color:#c9d1d9;">${sp.sellDate}</div>
          </div>
          <div style="background:#0d1117; padding:12px; border-radius:8px; text-align:center;">
            <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">예상 매도 시간</div>
            <div style="font-size:15px; font-weight:700; color:#c9d1d9;">${sp.sellTime}</div>
          </div>
        </div>
        <div style="font-size:12px; color:#8b949e; margin-top:10px; text-align:center;">
          * 예측 탭에서 더 자세한 매도 근거를 확인할 수 있습니다.
        </div>
      </div>
    `;
  }

  gridEl.innerHTML = html;
  updateRiskScenario();
}

function updateRiskScenario() {
  const riskData = window.currentRiskData;
  if (!riskData) return;

  const investInput = document.getElementById('invest-amount');
  const investAmt = parseFloat(investInput ? investInput.value : 1000) || 1000;
  const recommendedLev = riskData.recommended_leverage || 1;
  const maxLev = riskData.max_leverage || 2;
  const volatility = riskData.volatility || 0;
  const positionInfo = getPositionContext(riskData.position);
  const scenarios = getScenarioConfigs(positionInfo.label, recommendedLev, maxLev);

  const isLong = positionInfo.label === 'Long' || positionInfo.label === 'Buy';
  const isShort = positionInfo.label === 'Short' || positionInfo.label === 'Sell';
  
  const currentPrice = riskData.price || 0;
  // ATR 기반 기본 손절 폭 계산 (stop_loss_pct 미제공 또는 0인 경우 ATR로 대체)
  const atrPct = (riskData.atr && currentPrice) ? (riskData.atr / currentPrice * 100) : 2.0;
  const baseStopLossPct = (riskData.stop_loss_pct > 0) ? riskData.stop_loss_pct : (atrPct * 1.5);
  const baseTakeProfitPct = (riskData.take_profit_pct > 0) ? riskData.take_profit_pct : (baseStopLossPct * 2);

  // 포지션에 따른 동적 라벨 및 색상
  const profitLabel = isLong ? '예상 수익 (Long)' : isShort ? '예상 수익 (Short)' : '예상 수익 (대기)';
  const profitColor = isLong ? '#3fb950' : isShort ? '#f85149' : '#8b949e';

  const riskHtml = scenarios.map((scenario) => {
    const leverageRatio = maxLev > 0 ? scenario.leverage / maxLev : 0;
    const riskLevel = getVolatilityRiskLabel(volatility, leverageRatio);

    // 시나리오별 손절 폭 (stopFactor 적용으로 시나리오 간 차별화)
    const scenarioStopLossPct = baseStopLossPct * (scenario.stopFactor || 1.0);
    // 익절 폭 = 손절 폭 × 2 (최소 RR 1:2 유지)
    const scenarioTakeProfitPct = scenarioStopLossPct * 2;
    // 시나리오별 레버리지가 적용된 예상 수익률(%)
    const scenarioExpectedProfitPct = scenarioTakeProfitPct * scenario.leverage;

    // 현재가 기반 목표가/손절가 계산
    let takeProfitPrice = 0;
    let stopLossPrice = 0;

    if (isLong) {
        takeProfitPrice = currentPrice * (1 + scenarioTakeProfitPct / 100);
        stopLossPrice   = currentPrice * (1 - scenarioStopLossPct / 100);
    } else if (isShort) {
        takeProfitPrice = currentPrice * (1 - scenarioTakeProfitPct / 100);
        stopLossPrice   = currentPrice * (1 + scenarioStopLossPct / 100);
    } else {
        // Neutral: 상단 목표 / 하단 손절 동시 표시 (방향 미확정)
        takeProfitPrice = currentPrice * (1 + scenarioTakeProfitPct / 100);
        stopLossPrice   = currentPrice * (1 - scenarioStopLossPct / 100);
    }

    return `
      <div class="risk-card ${scenario.key}">
        <div class="risk-icon">${scenario.icon}</div>
        <div class="risk-name">${scenario.name}</div>
        <div class="risk-desc" style="min-height: 36px;">${scenario.desc}</div>
        
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px dashed #30363d;">
          <div class="risk-row" style="margin-bottom: 12px;">
            <span class="risk-lbl">운용 레버리지</span>
            <span style="color:#388bfd;font-weight:700;font-size:14px;">${scenario.leverage}x</span>
          </div>
          
          <div class="risk-row" style="margin-bottom: 12px;">
            <span class="risk-lbl">목표가</span>
            <span style="color:#3fb950;font-weight:700;font-size:14px;">${fmtPrice(takeProfitPrice)}</span>
          </div>

          <div class="risk-row" style="margin-bottom: 12px;">
            <span class="risk-lbl">손절가</span>
            <span style="color:#f85149;font-weight:700;font-size:14px;">${fmtPrice(stopLossPrice)}</span>
          </div>
          
          <div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
            <span class="risk-lbl" style="font-size: 13px;">예상 수익률</span>
            <span style="color:#3fb950; font-weight:800; font-size: 16px;">+${scenarioExpectedProfitPct.toFixed(1)}%</span>
          </div>
        </div>
        
        <div class="risk-ratio" style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; text-align: center;">
          변동성 위험 수준: <span style="color:#c9d1d9; font-weight:600;">${riskLevel}</span>
        </div>
      </div>
    `;
  }).join('');
  document.getElementById('risk-grid').innerHTML = riskHtml;
}

function parsePredictionDateTime(rawValue) {
  if (!rawValue) {
    return { date: '-', time: '23:59' };
  }
  if (rawValue.includes('T')) {
    const parts = rawValue.split('T');
    return { date: parts[0], time: (parts[1] || '23:59').substring(0, 5) || '23:59' };
  }
  if (rawValue.includes(' ')) {
    const parts = rawValue.split(' ');
    return { date: parts[0], time: (parts[1] || '23:59').substring(0, 5) || '23:59' };
  }
  return { date: rawValue, time: '23:59' };
}

function buildSellPrediction(d, fc, xgb, last) {
  const hwSeries = Array.isArray(fc.yhat) ? fc.yhat : [];
  const momentumSeries = Array.isArray(xgb) ? xgb : [];
  const dates = Array.isArray(fc.dates) ? fc.dates : [];
  const combinedSeries = hwSeries.map((value, idx) => {
    const momentum = momentumSeries[idx];
    return momentum != null ? value * 0.6 + momentum * 0.4 : value;
  });
  const combinedLast = combinedSeries.length ? combinedSeries[combinedSeries.length - 1] : last;
  const ma20Series = d.chart_data && Array.isArray(d.chart_data.ma20) ? d.chart_data.ma20 : [];
  const ma60Series = d.chart_data && Array.isArray(d.chart_data.ma60) ? d.chart_data.ma60 : [];
  const ma20 = ma20Series.length ? ma20Series[ma20Series.length - 1] : last;
  const ma60 = ma60Series.length ? ma60Series[ma60Series.length - 1] : last;
  const volatility = d.volatility_30d || 0;
  const rsi = d.rsi || 50;
  const atr = d.atr || (last * 0.03);
  const trendUp = combinedLast >= last;
  const movingAverageBias = ma20 >= ma60 ? 1 : -1;
  const rsiBias = rsi >= 68 ? -0.5 : rsi <= 38 ? 0.5 : 0;
  const trendScore = (trendUp ? 1 : -1) + movingAverageBias + rsiBias;
  const atrFactor = volatility >= 100 ? 1.0 : trendScore >= 1.5 ? 1.8 : trendScore >= 0.5 ? 1.4 : 1.1;
  const targetPrice = trendUp ? last + (atr * atrFactor) : last - (atr * Math.max(0.8, atrFactor - 0.2));
  let triggerIndex = combinedSeries.findIndex((value) => trendUp ? value >= targetPrice : value <= targetPrice);
  let triggerType = triggerIndex >= 0 ? 'target' : '';
  if (triggerIndex < 0 && combinedSeries.length >= 3) {
    triggerIndex = combinedSeries.findIndex((value, idx, arr) => {
      if (idx < 2) return false;
      if (trendUp) {
        return value < arr[idx - 1] && arr[idx - 1] >= arr[idx - 2];
      }
      return value > arr[idx - 1] && arr[idx - 1] <= arr[idx - 2];
    });
    if (triggerIndex >= 0) {
      triggerType = 'trend_shift';
    }
  }
  if (triggerIndex < 0) {
    triggerIndex = Math.max(0, combinedSeries.length - 1);
    triggerType = 'horizon';
  }
  const triggerPoint = parsePredictionDateTime(dates[triggerIndex] || dates[dates.length - 1] || '');
  let sellReason = '';
  if (triggerType === 'target') {
    sellReason = trendUp
      ? `앙상블 예측 가격이 목표가 ${fmtPrice(targetPrice)}에 도달하는 구간입니다. MA20 우위와 추세 상승이 유지되더라도 RSI 과열 또는 변동성 확대가 보이면 분할 매도를 우선합니다.`
      : `앙상블 예측 가격이 방어 가격 ${fmtPrice(targetPrice)}를 이탈하는 구간입니다. 하락 추세 지속과 변동성 확대로 추가 낙폭 가능성이 있어 보수적 매도를 우선합니다.`;
  } else if (triggerType === 'trend_shift') {
    sellReason = trendUp
      ? `목표가 도달 전 예측 곡선의 기울기가 둔화되는 시점입니다. 상승 추세가 이어져도 힘이 약해지는 신호로 해석해 부분 매도를 고려합니다.`
      : `예측 곡선의 하락 기울기가 완화되는 시점입니다. 급락 이후 반등 가능성이 커지는 구간이므로 단기 포지션 청산 또는 비중 축소를 고려합니다.`;
  } else {
    sellReason = trendUp
      ? `예측 기간 안에 즉시 목표가를 돌파하지 않아 마지막 예측 시점을 기본 매도 관찰 시점으로 둡니다. 추세는 유지되지만 변동성에 따라 분할 청산이 적절합니다.`
      : `예측 기간 안에 방어 가격을 명확히 이탈하지 않아 마지막 예측 시점을 기준으로 반등 여부를 재평가합니다. 추세 약화 전까지는 보수적으로 대응합니다.`;
  }
  const criteria = `${momentumSeries.length ? '앙상블(60% HW + 40% 모멘텀)' : 'Holt-Winters'} · MA20/MA60 추세 · RSI(${rsi.toFixed(1)}) · 30일 변동성(${(volatility || 0).toFixed(1)}%)`;
  return {
    sellDate: triggerPoint.date,
    sellTime: triggerPoint.time,
    sellReason,
    targetPrice,
    criteria
  };
}

function renderForecast(d) {
  const fc  = d.forecast;
  const xgb = d.xgb_forecast;
  const last = d.last_close;
  const sumEl = document.getElementById('forecast-summary');

  if (!fc || !fc.dates || fc.dates.length === 0) {
    sumEl.innerHTML = '<p style="color:#484f58;font-size:13px">예측 데이터 부족 (최소 30개 캔들 필요)</p>';
    return;
  }

  const hwF  = fc.yhat[fc.yhat.length - 1];
  const xgbF = xgb ? xgb[xgb.length - 1] : null;
  const ens  = xgbF != null ? hwF * 0.6 + xgbF * 0.4 : hwF;
  const ensChg = (ens - last) / last * 100;
  const ensUp  = ensChg >= 0;
  const clr    = ensUp ? '#3fb950' : '#f85149';
  const sellPrediction = buildSellPrediction(d, fc, xgb, last);

  sumEl.innerHTML = `
    <div style="background:#21262d;border-radius:10px;padding:12px;text-align:center;flex:1;min-width:120px">
      <div style="font-size:11px;color:#8b949e;margin-bottom:4px">HW 예측 (30일)</div>
      <div style="font-size:15px;font-weight:700;color:#388bfd">${fmtPrice(hwF)}</div>
    </div>
    ${xgbF != null ? `<div style="background:#21262d;border-radius:10px;padding:12px;text-align:center;flex:1;min-width:120px">
      <div style="font-size:11px;color:#8b949e;margin-bottom:4px">모멘텀 시뮬 (30일)</div>
      <div style="font-size:15px;font-weight:700;color:#d29922">${fmtPrice(xgbF)}</div>
    </div>` : ''}
    <div style="background:#0d1f4f;border:1px solid #1f4f8e;border-radius:10px;padding:12px;text-align:center;flex:1;min-width:120px">
      <div style="font-size:11px;color:#8b949e;margin-bottom:4px">📌 참고 시나리오</div>
      <div style="font-size:17px;font-weight:800;color:#fff">${fmtPrice(ens)}</div>
      <div style="font-size:12px;color:${clr}">${ensUp?'▲':'▼'} ${Math.abs(ensChg).toFixed(2)}%</div>
    </div>
    
    <div style="width:100%; background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; margin-top:8px;">
      <div style="font-size:14px; font-weight:700; color:#e6edf3; margin-bottom:12px; display:flex; align-items:center; gap:6px;">
        ⏱️ 스마트 매도 타이밍 예측
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px;">
        <div style="background:#0d1117; padding:12px; border-radius:8px;">
          <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">예상 매도 날짜</div>
          <div style="font-size:16px; font-weight:700; color:#c9d1d9;">${sellPrediction.sellDate}</div>
        </div>
        <div style="background:#0d1117; padding:12px; border-radius:8px;">
          <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">예상 매도 시간</div>
          <div style="font-size:16px; font-weight:700; color:#c9d1d9;">${sellPrediction.sellTime}</div>
        </div>
      </div>
      <div style="background:#21262d; padding:12px; border-radius:8px;">
        <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">매도 근거 (예측 로직)</div>
        <div style="font-size:13px; color:#e6edf3; line-height:1.5;">
          ${sellPrediction.sellReason}<br>
          <span style="color:#8b949e; font-size:11px; margin-top:4px; display:block;">
            * 기준: ${sellPrediction.criteria} · 목표 가격 ${fmtPrice(sellPrediction.targetPrice)}
          </span>
        </div>
      </div>
    </div>`;
}

function renderNews(d) {
  const newsList = document.getElementById('news-list');
  if (!d.news || d.news.length === 0) {
    newsList.innerHTML = '<p style="font-size:13px;color:#484f58">관련 뉴스를 찾을 수 없습니다.</p>';
    return;
  }
  newsList.innerHTML = d.news.map(n => `
    <div class="news-item">
      <div class="news-dot">●</div>
      <div>
        <a class="news-a" href="${n.link}" target="_blank">${n.title}</a>
        <div class="news-meta">${n.publisher || ''} ${n.published ? '· ' + n.published.substring(0,16) : ''}</div>
      </div>
    </div>`).join('');
}

// ── 탭 전환 ──
function switchTab(tab) {
  ['chart','ai','leverage','forecast','news'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('active', t === tab);
    document.getElementById('tab-content-' + t).style.display = t === tab ? 'block' : 'none';
  });
  if (tab === 'chart' && currentData) setTimeout(() => renderCharts(currentData), 50);
}

// ── 차트 렌더링 ──
function destroyCharts() {
  Object.values(chartInstances).forEach(c => { try { c.remove(); } catch(e){} });
  chartInstances = {};
}

function renderCharts(d) {
  destroyCharts();
  const cd = d.chart_data || {};

  const dates  = cd.dates  || [];
  const opens  = cd.open   || [];
  const highs  = cd.high   || [];
  const lows   = cd.low    || [];
  const closes = cd.close  || [];
  const vols   = cd.volume || [];
  const ma20   = cd.ma20   || [];
  const ma60   = cd.ma60   || [];
  const bbu    = cd.bb_upper || [];
  const bbl    = cd.bb_lower || [];
  const rsis   = cd.rsi    || [];
  const macds  = cd.macd   || [];
  const sigs   = cd.signal_line || [];
  const vwaps  = cd.vwap   || [];

  function toTime(s) {
    try { return Math.floor(new Date(s).getTime() / 1000); }
    catch(e) { return 0; }
  }

  // ── 가격 차트 ──
  const priceEl = document.getElementById('price-chart');
  if (priceEl && dates.length > 0) {
    const chart = LightweightCharts.createChart(priceEl, {
      layout:     {background:{color:'#161b22'}, textColor:'#8b949e'},
      grid:       {vertLines:{color:'#21262d'}, horzLines:{color:'#21262d'}},
      crosshair:  {mode: LightweightCharts.CrosshairMode.Normal},
      rightPriceScale: {borderColor:'#30363d'},
      timeScale:  {borderColor:'#30363d', timeVisible:true},
      height: 320,
    });
    chartInstances['price'] = chart;

    // 캔들스틱
    const candleSeries = chart.addCandlestickSeries({
      upColor:'#3fb950', downColor:'#f85149',
      borderUpColor:'#3fb950', borderDownColor:'#f85149',
      wickUpColor:'#3fb950', wickDownColor:'#f85149',
    });
    const candleData = dates.map((dt, i) => ({
      time: toTime(dt), open: opens[i], high: highs[i], low: lows[i], close: closes[i]
    })).filter(x => x.time > 0 && x.open != null);
    candleSeries.setData(candleData);

    // MA20
    if (ma20.length > 0) {
      const ma20s = chart.addLineSeries({color:'#388bfd', lineWidth:1, title:'MA20'});
      ma20s.setData(dates.map((dt,i) => ({time:toTime(dt), value:ma20[i]})).filter(x=>x.value!=null&&x.time>0));
    }
    // MA60
    if (ma60.length > 0) {
      const ma60s = chart.addLineSeries({color:'#d29922', lineWidth:1, title:'MA60'});
      ma60s.setData(dates.map((dt,i) => ({time:toTime(dt), value:ma60[i]})).filter(x=>x.value!=null&&x.time>0));
    }
    // BB Upper
    if (bbu.length > 0) {
      const bbus = chart.addLineSeries({color:'rgba(100,100,255,0.4)', lineWidth:1, lineStyle:2, title:'BB Upper'});
      bbus.setData(dates.map((dt,i) => ({time:toTime(dt), value:bbu[i]})).filter(x=>x.value!=null&&x.time>0));
    }
    // BB Lower
    if (bbl.length > 0) {
      const bbls = chart.addLineSeries({color:'rgba(100,100,255,0.4)', lineWidth:1, lineStyle:2, title:'BB Lower'});
      bbls.setData(dates.map((dt,i) => ({time:toTime(dt), value:bbl[i]})).filter(x=>x.value!=null&&x.time>0));
    }
    // VWAP
    if (vwaps.length > 0) {
      const vwapS = chart.addLineSeries({color:'rgba(255,200,0,0.7)', lineWidth:1, lineStyle:3, title:'VWAP'});
      vwapS.setData(dates.map((dt,i) => ({time:toTime(dt), value:vwaps[i]})).filter(x=>x.value!=null&&x.time>0));
    }
    chart.timeScale().fitContent();
  }

  // ── RSI 차트 ──
  const rsiEl = document.getElementById('rsi-chart');
  if (rsiEl && rsis.length > 0) {
    const chart = LightweightCharts.createChart(rsiEl, {
      layout:  {background:{color:'#161b22'}, textColor:'#8b949e'},
      grid:    {vertLines:{color:'#21262d'}, horzLines:{color:'#21262d'}},
      rightPriceScale: {borderColor:'#30363d'},
      timeScale: {borderColor:'#30363d', timeVisible:true},
      height: 160,
    });
    chartInstances['rsi'] = chart;
    const rsiS = chart.addLineSeries({color:'#a371f7', lineWidth:2, title:'RSI'});
    rsiS.setData(dates.map((dt,i) => ({time:toTime(dt), value:rsis[i]})).filter(x=>x.value!=null&&x.time>0));
    // 과매수/과매도 라인
    const ob = chart.addLineSeries({color:'rgba(248,81,73,0.5)', lineWidth:1, lineStyle:2});
    ob.setData(dates.map((dt) => ({time:toTime(dt), value:70})).filter(x=>x.time>0));
    const os = chart.addLineSeries({color:'rgba(63,185,80,0.5)', lineWidth:1, lineStyle:2});
    os.setData(dates.map((dt) => ({time:toTime(dt), value:30})).filter(x=>x.time>0));
    chart.timeScale().fitContent();
  }

  // ── MACD 차트 ──
  const macdEl = document.getElementById('macd-chart');
  if (macdEl && macds.length > 0) {
    const chart = LightweightCharts.createChart(macdEl, {
      layout:  {background:{color:'#161b22'}, textColor:'#8b949e'},
      grid:    {vertLines:{color:'#21262d'}, horzLines:{color:'#21262d'}},
      rightPriceScale: {borderColor:'#30363d'},
      timeScale: {borderColor:'#30363d', timeVisible:true},
      height: 160,
    });
    chartInstances['macd'] = chart;
    const macdS = chart.addLineSeries({color:'#388bfd', lineWidth:2, title:'MACD'});
    macdS.setData(dates.map((dt,i) => ({time:toTime(dt), value:macds[i]})).filter(x=>x.value!=null&&x.time>0));
    const sigS = chart.addLineSeries({color:'#f85149', lineWidth:1, title:'Signal'});
    sigS.setData(dates.map((dt,i) => ({time:toTime(dt), value:sigs[i]})).filter(x=>x.value!=null&&x.time>0));
    chart.timeScale().fitContent();
  }
}

// ── 스크리너 ──
async function loadScreener(sortBy, sortOrder) {
  const tbody = document.getElementById('scrn-tbody');
  tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:40px;color:#8b949e"><div class="spinner" style="margin:0 auto"></div></td></tr>';
  try {
    const sb = sortBy    || scrnSort.key;
    const so = sortOrder || scrnSort.dir;
    const r  = await fetch(`/api/screener?sort_by=${sb}&sort_order=${so}`);
    const d  = await r.json();
    screenerData = d.data || [];
    document.getElementById('scrn-count').textContent = `총 ${screenerData.length}개 코인`;

    // 필터 조건 표시
    const fc = d.filter_conditions || {};
    document.getElementById('scrn-filter-conds').innerHTML = Object.entries(fc).map(([k,v]) =>
      `<span style="background:#21262d;border:1px solid #30363d;border-radius:6px;padding:3px 8px">${k}: ${v}</span>`
    ).join('');

    renderScreener();
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:40px;color:#f85149">로딩 실패: ${e.message}</td></tr>`;
  }
}

function sortScreener(key) {
  if (scrnSort.key === key) scrnSort.dir = scrnSort.dir === 'desc' ? 'asc' : 'desc';
  else { scrnSort.key = key; scrnSort.dir = 'desc'; }
  loadScreener(scrnSort.key, scrnSort.dir);
}

function renderScreener() {
  const tbody = document.getElementById('scrn-tbody');
  if (!screenerData.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:40px;color:#8b949e">데이터 없음</td></tr>';
    return;
  }

  tbody.innerHTML = screenerData.map((s, idx) => {
    const up  = s.change >= 0;
    const clr = up ? '#3fb950' : '#f85149';
    const rsiClr = s.rsi > 70 ? '#f85149' : s.rsi < 30 ? '#3fb950' : '#8b949e';
    const levCls = s.risk_grade === 'Low' ? 'lev-low' : s.risk_grade === 'Medium' ? 'lev-medium' : s.risk_grade === 'High' ? 'lev-high' : 'lev-extreme';
    const rawSig = s.signal || '중립';
    const sigCls = rawSig.includes('적극 매수') ? 'sig-buy-strong' : rawSig === '매수' ? 'sig-buy' : rawSig === '중립' ? 'sig-neu' : 'sig-sell';

    return '<tr onclick="quickSearch(\'' + s.symbol + '\')" style="cursor:pointer">'
      + '<td style="color:#484f58">' + (idx+1) + '</td>'
      + '<td><div class="ticker-name">' + s.name + '</div><div class="ticker-code">' + s.symbol + '</div></td>'
      + '<td style="text-align:right;font-weight:600">' + (s.price || '-') + '</td>'
      + '<td style="text-align:right;font-weight:700;color:' + clr + '">' + (up?'▲':'▼') + ' ' + Math.abs(s.change||0).toFixed(2) + '%</td>'
      + '<td><span class="cat-badge">' + (s.category||'-') + '</span></td>'
      + '<td style="text-align:right;color:#8b949e;font-size:12px">$' + fmtVol(s.volume_24h||0) + '</td>'
      + '<td style="text-align:center;color:' + rsiClr + ';font-weight:700">' + (s.rsi||'-') + '</td>'
      + '<td style="text-align:center"><span class="lev-badge ' + levCls + '">' + (s.leverage_rec||'-') + 'x</span></td>'
      + '<td style="text-align:center"><span class="signal-badge ' + sigCls + '">' + rawSig + '</span></td>'
      + '</tr>';
  }).join('');
}

// ── 초기화 ──
loadSentiment();
</script>
</body>
</html>"""


# =============================================================================
# Vercel Handler (기존 구조 유지)
# =============================================================================
def replace_nan_with_none(obj):
    if isinstance(obj, (pd.Series, pd.Index)):
        return replace_nan_with_none(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return replace_nan_with_none(obj.to_dict(orient='list'))
    if isinstance(obj, np.ndarray):
        return replace_nan_with_none(obj.tolist())
    if isinstance(obj, list):
        return [replace_nan_with_none(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    return obj

VALID_INTERVALS = config.VALID_INTERVALS

def _send(handler_self, data: Any, status: int = 200, content_type: str = "application/json"):
    path = getattr(handler_self, 'path', '/').split('?')[0].rstrip('/') or '/'
    if content_type == "application/json":
        data = replace_nan_with_none(data)
        if isinstance(data, dict) and "api_version" not in data:
            data["api_version"] = config.APP_VERSION
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    else:
        body = data if isinstance(data, bytes) else data.encode("utf-8")

    handler_self.send_response(status)
    handler_self.send_header("Content-Type", content_type + "; charset=utf-8")
    handler_self.send_header("Content-Length", str(len(body)))
    handler_self.send_header("Access-Control-Allow-Origin", "*")
    handler_self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    handler_self.send_header("Access-Control-Allow-Headers", "Content-Type")

    cache_control = "no-store, no-cache, must-revalidate, proxy-revalidate"
    if status == 200:
        if path in ("/api/screener",):
            cache_control = "public, s-maxage=300, stale-while-revalidate=900"
        elif path in ("/api/coin",):
            cache_control = "public, s-maxage=30, stale-while-revalidate=120"
        elif path in ("/api/validation",):
            cache_control = "public, s-maxage=300, stale-while-revalidate=600"
        elif path in ("/api/health",):
            cache_control = "no-store"
        elif path in ("/", "/index.html"):
            cache_control = "public, s-maxage=3600"
        elif path.startswith("/api/"):
            cache_control = "public, s-maxage=30, stale-while-revalidate=120"

    handler_self.send_header("Cache-Control", cache_control)
    handler_self.send_header("X-Content-Type-Options", "nosniff")
    handler_self.send_header("X-Frame-Options", "DENY")
    handler_self.send_header("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
    handler_self.send_header(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://api.binance.com https://api1.binance.com https://api2.binance.com https://api3.binance.com https://data-api.binance.vision https://fapi.binance.com https://api.coingecko.com https://news.google.com; object-src 'none'; base-uri 'self'; frame-ancestors 'none';"
    )
    handler_self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
    handler_self.end_headers()
    handler_self.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[CoinOracle] {fmt % args}")

    def do_OPTIONS(self):
        _send(self, {})

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            path   = parsed.path.rstrip("/") or "/"

            # 입력 검증
            if path in ("/api/coin", "/api/stock", "/api/validation"):
                ok, ticker_or_error = validate_ticker(params.get("ticker", params.get("symbol", "")))
                if not ok:
                    _send(self, {"error": ticker_or_error}, 400)
                    return
                ok, interval_or_error = validate_interval(params.get("interval", config.DEFAULT_INTERVAL))
                if not ok:
                    _send(self, {"error": interval_or_error}, 400)
                    return

            # API 라우팅 처리
            if path.startswith("/api/"):
                result = route(path, params)
                if result is None:
                    _send(self, {"error": "Not Found"}, 404)
                else:
                    _send(self, result)
            else:
                # 루트 또는 HTML 요청은 public/index.html 우선 사용
                try:
                    public_index = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
                    with open(public_index, "r", encoding="utf-8") as f:
                        _send(self, f.read(), 200, "text/html")
                except Exception:
                    _send(self, HTML, 200, "text/html")
        except Exception as e:
            print(f"Server Error: {str(e)}\n{traceback.format_exc()}")
            _send(self, {"error": "Internal Server Error"}, 500)
