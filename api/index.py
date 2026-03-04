# -*- coding: utf-8 -*-
# SignalCoin API - Vercel 공식 FastAPI 배포 방식
# 참고: https://github.com/vercel/examples/tree/main/python/fastapi
#
# ★ Vercel 핵심 규칙 ★
#   - 파일 위치: api/index.py  (반드시 api/ 폴더 안에 위치)
#   - 진입점: app = FastAPI(...)  ← Vercel이 이 변수를 자동으로 인식
#   - vercel.json의 functions 패턴: "api/**/*.py" 로 지정

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests

# ─────────────────────────────────────────────
# 1. FastAPI 앱 초기화  ← Vercel이 이 변수를 진입점으로 인식
# ─────────────────────────────────────────────
app = FastAPI(
    title="SignalCoin API",
    description="암호화폐 기술적 분석 및 시장 데이터 API (Vercel Serverless Edition)",
    version="4.0.0",
)

# ─────────────────────────────────────────────
# 2. Pydantic 응답 모델
# ─────────────────────────────────────────────
class FearGreedResponse(BaseModel):
    value: int = Field(..., description="공포-탐욕 지수 (0~100)")
    classification: str = Field(..., description="상태 분류 (예: Extreme Fear)")
    timestamp: str

class CoinInfo(BaseModel):
    id: str
    symbol: str
    name: str

class IndicatorResponse(BaseModel):
    ticker: str
    rsi: Optional[float] = Field(None, description="RSI (14)")
    macd: Optional[float] = Field(None, description="MACD 값")
    macd_signal: Optional[float] = Field(None, description="MACD 시그널")
    stoch_k: Optional[float] = Field(None, description="Stochastic %K (14)")
    bb_upper: Optional[float] = Field(None, description="볼린저 밴드 상단")
    bb_lower: Optional[float] = Field(None, description="볼린저 밴드 하단")
    ema20: Optional[float] = Field(None, description="EMA 20")
    ema50: Optional[float] = Field(None, description="EMA 50")
    current_price: Optional[float]

class SignalResponse(BaseModel):
    ticker: str
    signal: str = Field(..., description="STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL")
    total_score: float = Field(..., description="종합 점수 (0~100)")
    confidence: float = Field(..., description="신뢰도 (0~100)")
    rsi_score: float
    trend_score: float

class BacktestResponse(BaseModel):
    ticker: str
    period_days: int
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    initial_capital: float = 1000.0
    final_capital: float

class TradingMetricsResponse(BaseModel):
    ticker: str
    return_1w_pct: float
    return_1m_pct: float
    return_3m_pct: float
    buy_sell_ratio: float
    sentiment: str
    last_update: str

# ─────────────────────────────────────────────
# 3. 내부 유틸리티 함수 (원본 app.py 로직 이식)
# ─────────────────────────────────────────────

def _fetch_ohlcv(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    """원본 load_crypto_data 경량 버전"""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"'{ticker}' 데이터를 찾을 수 없습니다.")
    # MultiIndex 컬럼 평탄화 (yfinance 0.2+ 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df.rename(columns={date_col: "Date"}, inplace=True)
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.tz_localize(None)
    return df


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, MACD, Stochastic, Bollinger Bands, EMA 계산"""
    close = df["Close"]

    # RSI (Wilder's smoothing)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Stochastic %K
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = (close - low14) * 100 / (high14 - low14 + 1e-8)

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20

    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    return df


def _calc_signal_score(df: pd.DataFrame) -> dict:
    """RSI + EMA 기반 매수/매도 신호 점수 계산"""
    df = _calc_indicators(df)
    latest = df.iloc[-1]

    rsi = latest["RSI"] if pd.notna(latest["RSI"]) else 50
    rsi_score = 50
    if rsi < 30:   rsi_score = 80
    elif rsi < 40: rsi_score = 65
    elif rsi > 70: rsi_score = 20
    elif rsi > 60: rsi_score = 35

    close = latest["Close"]
    ema20 = latest["EMA20"]
    ema50 = latest["EMA50"]
    trend_score = 50
    if close > ema20 > ema50:   trend_score = 80
    elif close > ema20:         trend_score = 60
    elif close < ema20 < ema50: trend_score = 20
    elif close < ema20:         trend_score = 40

    total = rsi_score * 0.5 + trend_score * 0.5

    if total >= 70:   signal = "STRONG_BUY"
    elif total >= 58: signal = "BUY"
    elif total <= 30: signal = "STRONG_SELL"
    elif total <= 42: signal = "SELL"
    else:             signal = "NEUTRAL"

    return {
        "signal": signal,
        "total_score": round(total, 2),
        "confidence": round(abs(total - 50) * 2, 2),
        "rsi_score": round(rsi_score, 2),
        "trend_score": round(trend_score, 2),
    }


def _calc_backtest(df: pd.DataFrame) -> dict:
    """Buy-and-Hold 백테스트"""
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 2:
        raise HTTPException(status_code=400, detail="백테스트를 위한 데이터가 부족합니다.")

    cum = (1 + returns).cumprod()
    total_return = (cum.iloc[-1] - 1) * 100
    excess = returns - (0.02 / 252)
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0.0
    running_max = cum.cummax()
    max_dd = ((cum - running_max) / running_max).min() * 100
    win_rate = (returns > 0).sum() / len(returns) * 100
    final_capital = 1000.0 * cum.iloc[-1]

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "final_capital": round(final_capital, 2),
    }


def _calc_trading_metrics(ticker: str) -> dict:
    """기간별 수익률 및 시장 심리 지표"""
    df = _fetch_ohlcv(ticker, 100, "1d")
    close = df["Close"]
    current = close.iloc[-1]

    def ret(n):
        return ((current - close.iloc[-n]) / close.iloc[-n] * 100) if len(close) >= n else 0.0

    vol_recent = df["Volume"].iloc[-5:].mean()
    vol_avg = df["Volume"].mean()
    buy_ratio = min(100, max(0, 50 + (vol_recent / (vol_avg + 1e-8) - 1) * 50))
    sentiment = "BULLISH" if buy_ratio > 60 else ("BEARISH" if buy_ratio < 40 else "NEUTRAL")

    return {
        "return_1w_pct": round(ret(7), 2),
        "return_1m_pct": round(ret(30), 2),
        "return_3m_pct": round(ret(90), 2),
        "buy_sell_ratio": round(buy_ratio, 2),
        "sentiment": sentiment,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ─────────────────────────────────────────────
# 4. API 엔드포인트
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Status"])
def root():
    """API 상태 확인 및 엔드포인트 목록 안내"""
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SignalCoin API</title>
        <style>
            body { font-family: -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0;
                   max-width: 820px; margin: 60px auto; padding: 0 20px; }
            h1 { color: #00d4aa; }
            h2 { color: #888; font-size: 0.9rem; font-weight: normal; margin-top: -10px; }
            .card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
                    padding: 16px 20px; margin: 10px 0; }
            .method { background: #00d4aa; color: #000; padding: 2px 8px; border-radius: 4px;
                      font-size: 0.8rem; font-weight: bold; margin-right: 8px; }
            a { color: #00d4aa; text-decoration: none; }
            a:hover { text-decoration: underline; }
            code { background: #2a2a2a; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }
            p { margin: 6px 0 0; color: #aaa; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <h1>📊 SignalCoin API v4.0</h1>
        <h2>암호화폐 기술적 분석 API — Vercel Serverless Edition</h2>

        <div class="card">
            <span class="method">GET</span>
            <a href="/api/v2/market/fear-and-greed">/api/v2/market/fear-and-greed</a>
            <p>공포-탐욕 지수 조회</p>
        </div>
        <div class="card">
            <span class="method">GET</span>
            <a href="/api/v2/coins/list">/api/v2/coins/list</a>
            <p>CoinGecko 시가총액 상위 코인 목록 조회</p>
        </div>
        <div class="card">
            <span class="method">GET</span>
            <code>/api/v2/market/price/{ticker}</code>
            <p>OHLCV 가격 데이터 조회 · 예: <a href="/api/v2/market/price/BTC-USD">/api/v2/market/price/BTC-USD</a></p>
        </div>
        <div class="card">
            <span class="method">GET</span>
            <code>/api/v2/analysis/indicators/{ticker}</code>
            <p>RSI, MACD, 볼린저밴드 등 기술적 지표 · 예:
               <a href="/api/v2/analysis/indicators/BTC-USD">/api/v2/analysis/indicators/BTC-USD</a></p>
        </div>
        <div class="card">
            <span class="method">GET</span>
            <code>/api/v2/analysis/signal/{ticker}</code>
            <p>매수/매도 신호 점수 · 예:
               <a href="/api/v2/analysis/signal/BTC-USD">/api/v2/analysis/signal/BTC-USD</a></p>
        </div>
        <div class="card">
            <span class="method">GET</span>
            <code>/api/v2/analysis/metrics/{ticker}</code>
            <p>기간별 수익률 및 시장 심리 · 예:
               <a href="/api/v2/analysis/metrics/BTC-USD">/api/v2/analysis/metrics/BTC-USD</a></p>
        </div>
        <div class="card">
            <span class="method">GET</span>
            <code>/api/v2/portfolio/backtest/{ticker}</code>
            <p>Buy-and-Hold 백테스트 · 예:
               <a href="/api/v2/portfolio/backtest/BTC-USD">/api/v2/portfolio/backtest/BTC-USD</a></p>
        </div>
        <div class="card">
            <span class="method">GET</span>
            <a href="/docs">/docs</a> &nbsp;|&nbsp; <a href="/redoc">/redoc</a>
            <p>Swagger / ReDoc 자동 생성 API 문서</p>
        </div>
    </body>
    </html>
    """


@app.get("/api/v2/market/fear-and-greed",
         response_model=FearGreedResponse, tags=["Market Data"])
def get_fear_and_greed():
    """Alternative.me 공포-탐욕 지수 조회"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        d = r.json()["data"][0]
        return FearGreedResponse(
            value=int(d["value"]),
            classification=d["value_classification"],
            timestamp=datetime.fromtimestamp(int(d["timestamp"])).strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Fear & Greed API 호출 실패: {e}")


@app.get("/api/v2/coins/list",
         response_model=List[CoinInfo], tags=["Market Data"])
def get_coin_list(limit: int = Query(50, ge=1, le=250)):
    """CoinGecko 시가총액 상위 코인 목록 조회"""
    try:
        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI()
        markets = cg.get_coins_markets(
            vs_currency="usd", order="market_cap_desc",
            per_page=limit, page=1
        )
        return [
            CoinInfo(id=c["id"], symbol=str(c.get("symbol", "")).upper(), name=c["name"])
            for c in markets if c.get("id")
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"CoinGecko API 호출 실패: {e}")


@app.get("/api/v2/market/price/{ticker}", tags=["Market Data"])
def get_price(
    ticker: str,
    days: int = Query(90, ge=1, le=730),
    interval: str = Query("1d", description="1m/5m/15m/1h/1d/1wk")
):
    """OHLCV 가격 데이터 조회 (yfinance). ticker 예: BTC-USD, ETH-USD"""
    df = _fetch_ohlcv(ticker, days, interval)
    df["Date"] = df["Date"].astype(str)
    return {
        "ticker": ticker,
        "interval": interval,
        "count": len(df),
        "data": df[["Date", "Open", "High", "Low", "Close", "Volume"]].round(4).to_dict(orient="records")
    }


@app.get("/api/v2/analysis/indicators/{ticker}",
         response_model=IndicatorResponse, tags=["Technical Analysis"])
def get_indicators(ticker: str, days: int = Query(120, ge=50, le=730)):
    """RSI, MACD, 볼린저밴드, EMA 등 기술적 지표 최신 값 반환"""
    df = _fetch_ohlcv(ticker, days, "1d")
    df = _calc_indicators(df)
    latest = df.iloc[-1]

    def safe(col):
        v = latest.get(col)
        return round(float(v), 4) if v is not None and pd.notna(v) else None

    return IndicatorResponse(
        ticker=ticker,
        rsi=safe("RSI"),
        macd=safe("MACD"),
        macd_signal=safe("MACD_Signal"),
        stoch_k=safe("Stoch_K"),
        bb_upper=safe("BB_upper"),
        bb_lower=safe("BB_lower"),
        ema20=safe("EMA20"),
        ema50=safe("EMA50"),
        current_price=safe("Close"),
    )


@app.get("/api/v2/analysis/signal/{ticker}",
         response_model=SignalResponse, tags=["Technical Analysis"])
def get_signal(ticker: str, days: int = Query(120, ge=50, le=730)):
    """RSI + EMA 기반 매수/매도 신호 점수 반환"""
    df = _fetch_ohlcv(ticker, days, "1d")
    if len(df) < 50:
        raise HTTPException(status_code=400,
                            detail="신호 계산을 위해 최소 50일치 데이터가 필요합니다.")
    result = _calc_signal_score(df)
    return SignalResponse(ticker=ticker, **result)


@app.get("/api/v2/analysis/metrics/{ticker}",
         response_model=TradingMetricsResponse, tags=["Technical Analysis"])
def get_trading_metrics(ticker: str):
    """기간별 수익률 및 시장 심리 지표 반환"""
    metrics = _calc_trading_metrics(ticker)
    return TradingMetricsResponse(ticker=ticker, **metrics)


@app.get("/api/v2/portfolio/backtest/{ticker}",
         response_model=BacktestResponse, tags=["Portfolio"])
def run_backtest(ticker: str, days: int = Query(365, ge=30, le=1825)):
    """Buy-and-Hold 백테스트"""
    df = _fetch_ohlcv(ticker, days, "1d")
    result = _calc_backtest(df)
    return BacktestResponse(ticker=ticker, period_days=days, **result)
