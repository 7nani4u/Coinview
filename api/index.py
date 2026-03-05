# -*- coding: utf-8 -*-
# SignalCoin API v5.0 - SPA Edition
# 원본 Streamlit 앱의 백엔드 로직을 FastAPI로 완전 재구현

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. FastAPI 앱 초기화 및 정적 파일 마운트
# ─────────────────────────────────────────────
app = FastAPI(
    title="SignalCoin API for SPA",
    description="Streamlit 앱의 모든 백엔드 기능을 제공하는 API",
    version="5.0.0",
)

# ─────────────────────────────────────────────
# 2. Pydantic 모델 정의 (API 응답 구조)
# ─────────────────────────────────────────────
class FearGreedResponse(BaseModel):
    value: int
    classification: str

class CoinInfo(BaseModel):
    id: str
    symbol: str
    name: str

class IndicatorData(BaseModel):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None

class SignalData(BaseModel):
    signal: str
    total_score: float
    confidence: float
    rsi_score: float
    trend_score: float

class BacktestResult(BaseModel):
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    final_capital: float

class TradingMetrics(BaseModel):
    returns: Dict[str, float]
    buy_sell_ratio: float
    sentiment: str

class FullAnalysisResponse(BaseModel):
    ticker: str
    current_price: float
    indicators: IndicatorData
    signal: SignalData
    backtest: BacktestResult
    metrics: TradingMetrics
    chart_data: List[Dict]

# ─────────────────────────────────────────────
# 3. 핵심 백엔드 로직 (원본 app.py 이식 및 최적화)
# ─────────────────────────────────────────────

@app.get("/api/fear-greed", response_model=FearGreedResponse)
def get_fear_greed_index():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        r.raise_for_status()
        data = r.json()["data"][0]
        return {"value": int(data["value"]), "classification": data["value_classification"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Fear & Greed API 호출 실패: {e}")

@app.get("/api/coins", response_model=List[CoinInfo])
def get_coin_list(limit: int = 100):
    try:
        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI()
        markets = cg.get_coins_markets(vs_currency="usd", order="market_cap_desc", per_page=limit, page=1)
        return [CoinInfo(id=c["id"], symbol=str(c.get("symbol", "")).upper(), name=c["name"]) for c in markets if c.get("id")]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"CoinGecko API 호출 실패: {e}")

@app.get("/api/analyze", response_model=FullAnalysisResponse)
def get_full_analysis(
    ticker: str = Query("BTC-USD", description="yfinance 티커 (예: BTC-USD)"),
    days: int = Query(365, ge=90, le=1825, description="분석 기간(일)")
):
    try:
        # 1. 데이터 로드
        df = _fetch_ohlcv(ticker, days)

        # 2. 지표 계산
        df_indicators = _calculate_all_indicators(df.copy())
        latest_indicators = df_indicators.iloc[-1]

        # 3. 신호 생성
        signal_result = _calculate_signal_score(df_indicators)

        # 4. 백테스트
        backtest_result = _run_simple_backtest(df)

        # 5. 트레이딩 메트릭
        metrics_result = _calculate_trading_metrics(df)

        # 6. 차트 데이터 준비
        df_chart = df_indicators.reset_index()
        df_chart["Date"] = df_chart["Date"].dt.strftime("%Y-%m-%d")
        chart_data = df_chart.to_dict(orient="records")

        return FullAnalysisResponse(
            ticker=ticker,
            current_price=df["Close"].iloc[-1],
            indicators=IndicatorData(**latest_indicators.to_dict()),
            signal=SignalData(**signal_result),
            backtest=BacktestResult(**backtest_result),
            metrics=TradingMetrics(**metrics_result),
            chart_data=chart_data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {e}")

# ─────────────────────────────────────────────
# 4. 내부 헬퍼 함수
# ─────────────────────────────────────────────

def _fetch_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"데이터 없음: '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    # EMA
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    return df.dropna()

def _calculate_signal_score(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    rsi_score = (100 - latest["rsi"]) * 0.8 if latest["rsi"] > 50 else latest["rsi"] * 1.2
    trend_score = 50
    if latest["Close"] > latest["ema20"] > latest["ema50"]: trend_score = 85
    elif latest["Close"] > latest["ema20"]: trend_score = 65
    elif latest["Close"] < latest["ema20"] < latest["ema50"]: trend_score = 15
    elif latest["Close"] < latest["ema20"]: trend_score = 35
    total = np.clip(rsi_score * 0.5 + trend_score * 0.5, 0, 100)
    signal = "STRONG_BUY" if total >= 75 else "BUY" if total >= 60 else "STRONG_SELL" if total <= 25 else "SELL" if total <= 40 else "NEUTRAL"
    return {
        "signal": signal, "total_score": total, "confidence": abs(total - 50) * 2,
        "rsi_score": rsi_score, "trend_score": trend_score
    }

def _run_simple_backtest(df: pd.DataFrame) -> dict:
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 2: return {"total_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0, "win_rate_pct": 0, "final_capital": 1000}
    cum_returns = (1 + returns).cumprod()
    total_return = (cum_returns.iloc[-1] - 1) * 100
    sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
    running_max = cum_returns.cummax()
    max_dd = ((cum_returns - running_max) / running_max).min() * 100
    return {
        "total_return_pct": total_return, "sharpe_ratio": sharpe, "max_drawdown_pct": max_dd,
        "win_rate_pct": (returns > 0).sum() / len(returns) * 100, "final_capital": 1000 * cum_returns.iloc[-1]
    }

def _calculate_trading_metrics(df: pd.DataFrame) -> dict:
    close = df["Close"]
    returns = {p: ((close.iloc[-1] - close.iloc[-d]) / close.iloc[-d] * 100) if len(close) > d else 0 for p, d in {"1w": 7, "1m": 30, "3m": 90}.items()}
    buy_ratio = np.clip(50 + (df["Volume"].iloc[-5:].mean() / (df["Volume"].mean() + 1e-8) - 1) * 50, 0, 100)
    sentiment = "BULLISH" if buy_ratio > 60 else "BEARISH" if buy_ratio < 40 else "NEUTRAL"
    return {"returns": returns, "buy_sell_ratio": buy_ratio, "sentiment": sentiment}

# ─────────────────────────────────────────────
# 5. 정적 파일 서빙 (프론트엔드 UI)
# ─────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all(full_path: str):
    return FileResponse("static/index.html")
