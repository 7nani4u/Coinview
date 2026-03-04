# -*- coding: utf-8 -*-
# Vercel 호환 FastAPI 변환 버전 (Manus AI, 2026-03-05)
# 원본 Streamlit 앱(app.py)의 핵심 로직을 Vercel 서버리스 환경에 맞게 재구성

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from scipy import stats
import statsmodels.api as sm
from pycoingecko import CoinGeckoAPI

# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title="SignalCoin API (Vercel Edition)",
    description="Streamlit 앱의 핵심 기능을 Vercel 서버리스 환경으로 마이그레이션한 API",
    version="2.0.0",
)

# --- Pydantic 데이터 모델 정의 ---
class FearGreedIndex(BaseModel):
    value: int = Field(..., description="공포-탐욕 지수 (0-100)")
    value_classification: str = Field(..., description="지수 상태 (예: 'Extreme Fear')")

class PriceData(BaseModel):
    timestamp: int
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int

class IndicatorResult(BaseModel):
    rsi: Optional[float] = Field(None, description="RSI 값")
    macd: Optional[float] = Field(None, description="MACD 값")
    macd_signal: Optional[float] = Field(None, description="MACD 시그널 라인")
    stoch_k: Optional[float] = Field(None, description="Stochastic %K")

class SignalScore(BaseModel):
    total_score: float
    signal: str
    confidence: float

class PortfolioBacktest(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

# --- Vercel 호환으로 변환된 핵심 함수 ---

# 원본: get_all_coins_from_coingecko (685행)
@app.get("/api/v2/coins/list", tags=["Market Data"])
def get_coin_list(limit: int = Query(100, ge=1, le=250)):
    """CoinGecko에서 시가총액 순위 코인 목록을 가져옵니다."""
    try:
        cg = CoinGeckoAPI()
        markets = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=limit, page=1)
        return [{
            "id": coin.get('id'), 
            "symbol": str(coin.get('symbol','')).upper(),
            "name": coin.get('name')
        } for coin in markets if coin.get('id')]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"CoinGecko API 호출 실패: {e}")

# 원본: load_crypto_data (2211행)
def fetch_price_data_util(ticker: str, days: int, interval: str) -> pd.DataFrame:
    """yfinance로부터 가격 데이터를 가져오는 유틸리티 함수"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"{ticker}에 대한 데이터를 찾을 수 없습니다.")
        df.reset_index(inplace=True)
        # yfinance가 Datetime을 tz-aware로 반환할 경우를 대비해 tz-naive로 변환
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
             df['Date'] = df['Date'].dt.tz_localize(None)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance 데이터 로딩 실패: {e}")

# 원본: calculate_indicators_wilders (2433행)
def calculate_indicators_util(df: pd.DataFrame) -> pd.DataFrame:
    """경량 기술적 지표를 계산하는 유틸리티 함수"""
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).sum() / df['Close'].diff().lt(0).abs().rolling(14).sum()))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - low14) * 100 / (high14 - low14 + 1e-8)
    return df

# --- API 엔드포인트 ---

@app.get("/", tags=["Status"])
def read_root():
    return {"status": "ok", "message": "SignalCoin API v2에 오신 것을 환영합니다."}

# 원본: get_fear_and_greed (app.py에 없었으나, 이전 코드에서 추가)
@app.get("/api/v2/market/fear-and-greed", response_model=FearGreedIndex, tags=["Market Data"])
def get_fear_and_greed_index():
    """Alternative.me의 공포-탐욕 지수를 가져옵니다."""
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        response.raise_for_status()
        data = response.json()["data"][0]
        return FearGreedIndex(value=int(data["value"]), value_classification=data["value_classification"])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Alternative.me API 호출 실패: {e}")

@app.get("/api/v2/market/price/{ticker}", response_model=List[PriceData], tags=["Market Data"])
def get_price_data(ticker: str = "BTC-USD", days: int = 90, interval: str = "1d"):
    """특정 티커의 OHLCV 데이터를 가져옵니다."""
    df = fetch_price_data_util(ticker, days, interval)
    df.rename(columns={'Date': 'timestamp'}, inplace=True)
    df["timestamp"] = df["timestamp"].apply(lambda x: int(x.timestamp()))
    return df.to_dict(orient="records")

# 원본: calculate_indicators_wilders (2433행) + 개별 지표 함수들
@app.get("/api/v2/analysis/indicators/{ticker}", response_model=IndicatorResult, tags=["Technical Analysis"])
def get_technical_indicators(ticker: str = "BTC-USD", days: int = 90):
    """RSI, MACD, Stochastic 등 기본 기술적 지표의 최신 값을 계산합니다."""
    df = fetch_price_data_util(ticker, days + 30, "1d") # 계산을 위해 추가 데이터 로드
    df = calculate_indicators_util(df)
    latest = df.iloc[-1]
    return IndicatorResult(
        rsi=round(latest['RSI'], 2) if pd.notna(latest['RSI']) else None,
        macd=round(latest['MACD'], 4) if pd.notna(latest['MACD']) else None,
        macd_signal=round(latest['MACD_Signal'], 4) if pd.notna(latest['MACD_Signal']) else None,
        stoch_k=round(latest['Stoch_K'], 2) if pd.notna(latest['Stoch_K']) else None,
    )

# 원본: calculate_signal_score (430행)
@app.get("/api/v2/analysis/signal-score/{ticker}", response_model=SignalScore, tags=["Technical Analysis"])
def get_signal_score(ticker: str = "BTC-USD"):
    """RSI와 EMA를 기반으로 단순화된 매수/매도 신호 점수를 계산합니다."""
    df = fetch_price_data_util(ticker, 100, "1d")
    if len(df) < 50:
        raise HTTPException(status_code=400, detail="신호 계산을 위해 최소 50일치 데이터가 필요합니다.")
    
    df = calculate_indicators_util(df)
    latest = df.iloc[-1]
    
    rsi = latest['RSI']
    pattern_score = 50
    if rsi < 30: pattern_score = 80
    elif rsi < 40: pattern_score = 60
    elif rsi > 70: pattern_score = 20
    elif rsi > 60: pattern_score = 40

    close = latest['Close']
    ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    trend_score = 50
    if close > ema20 > ema50: trend_score = 80
    elif close > ema20: trend_score = 60
    elif close < ema20 < ema50: trend_score = 20
    elif close < ema20: trend_score = 40

    total_score = (pattern_score * 0.5 + trend_score * 0.5)
    signal = "NEUTRAL"
    if total_score >= 70: signal = 'STRONG_BUY'
    elif total_score >= 55: signal = 'BUY'
    elif total_score <= 30: signal = 'STRONG_SELL'
    elif total_score <= 45: signal = 'SELL'

    return SignalScore(
        total_score=round(total_score, 2),
        signal=signal,
        confidence=abs(total_score - 50) * 2
    )

# 원본: backtest_portfolio_simple (1910행)
@app.get("/api/v2/portfolio/backtest/{ticker}", response_model=PortfolioBacktest, tags=["Portfolio"])
def run_simple_backtest(ticker: str = "BTC-USD", days: int = 365):
    """단순 'Buy and Hold' 전략에 대한 백테스트를 실행합니다."""
    df = fetch_price_data_util(ticker, days, "1d")
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="백테스트를 위해 최소 2일치 데이터가 필요합니다.")

    returns = df['Close'].pct_change().dropna()
    if len(returns) < 2:
        return PortfolioBacktest(total_return=0, sharpe_ratio=0, max_drawdown=0, win_rate=0)

    cumulative_returns = (1 + returns).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100

    daily_rf = 0.02 / 252
    excess_returns = returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0

    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    win_rate = (returns > 0).sum() / len(returns) * 100

    return PortfolioBacktest(
        total_return=round(total_return, 2),
        sharpe_ratio=round(sharpe_ratio, 2),
        max_drawdown=round(max_drawdown, 2),
        win_rate=round(win_rate, 2)
    )

# 로컬 테스트: uvicorn api.index:app --reload
