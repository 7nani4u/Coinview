# -*- coding: utf-8 -*-
# SignalCoin API v6.0 - 500 오류 완전 수정판
#
# 수정 사항:
# [FIX-1] StaticFiles 마운트 제거 → HTML은 public/ 폴더에서 Vercel이 직접 서빙
# [FIX-2] IndicatorData(**latest.to_dict()) → 필요한 컬럼만 명시적으로 추출
# [FIX-3] chart_data NaN 값을 None으로 변환하여 JSON 직렬화 안정화
# [FIX-4] current_price float() 명시적 변환으로 numpy 타입 직렬화 보장

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import warnings
import math

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. FastAPI 앱 초기화
# [FIX-1] StaticFiles 마운트 완전 제거
#   - 이전 코드: app.mount("/", StaticFiles(directory="static"))
#   - 문제: Vercel에서 api/index.py 실행 시 CWD=/var/task 이므로
#           "static" 상대경로가 /var/task/static을 찾지만 존재하지 않아
#           RuntimeError → 500 FUNCTION_INVOCATION_FAILED 발생
#   - 해결: HTML 파일을 public/ 폴더에 두고 vercel.json의 정적 파일 서빙으로 처리
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SignalCoin API",
    description="암호화폐 AI 분석 API",
    version="6.0.0",
)

# CORS 설정 (프론트엔드에서 API 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# 2. Pydantic 모델
# ─────────────────────────────────────────────────────────────
class FearGreedResponse(BaseModel):
    value: int
    classification: str

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
    chart_data: List[Dict[str, Any]]

# ─────────────────────────────────────────────────────────────
# 3. API 엔드포인트
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "6.0.0"}

@app.get("/api/fear-greed", response_model=FearGreedResponse)
def get_fear_greed_index():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        r.raise_for_status()
        data = r.json()["data"][0]
        return {"value": int(data["value"]), "classification": data["value_classification"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Fear & Greed API 호출 실패: {str(e)}")

@app.get("/api/analyze", response_model=FullAnalysisResponse)
def get_full_analysis(
    ticker: str = Query("BTC-USD", description="yfinance 티커 (예: BTC-USD)"),
    days: int = Query(365, ge=90, le=1825, description="분석 기간(일)")
):
    try:
        # 1. 데이터 로드
        df = _fetch_ohlcv(ticker, days)

        # 2. 지표 계산
        df_ind = _calculate_all_indicators(df.copy())

        # 3. 신호 생성
        signal_result = _calculate_signal_score(df_ind)

        # 4. 백테스트
        backtest_result = _run_simple_backtest(df)

        # 5. 트레이딩 메트릭
        metrics_result = _calculate_trading_metrics(df)

        # 6. 차트 데이터 준비
        # [FIX-3] NaN → None 변환으로 JSON 직렬화 안정화
        df_chart = df_ind.reset_index()
        if "Date" in df_chart.columns:
            df_chart["Date"] = df_chart["Date"].dt.strftime("%Y-%m-%d")
        elif "Datetime" in df_chart.columns:
            df_chart["Date"] = df_chart["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
            df_chart.drop(columns=["Datetime"], inplace=True)
        chart_data = _sanitize_chart_data(df_chart)

        # [FIX-2] IndicatorData: 필요한 컬럼만 명시적으로 추출
        latest = df_ind.iloc[-1]
        indicators = IndicatorData(
            rsi=_safe_float(latest.get("rsi")),
            macd=_safe_float(latest.get("macd")),
            macd_signal=_safe_float(latest.get("macd_signal")),
            bb_upper=_safe_float(latest.get("bb_upper")),
            bb_lower=_safe_float(latest.get("bb_lower")),
            ema20=_safe_float(latest.get("ema20")),
            ema50=_safe_float(latest.get("ema50")),
        )

        # [FIX-4] current_price: float() 명시적 변환
        current_price = float(df["Close"].iloc[-1])

        return FullAnalysisResponse(
            ticker=ticker,
            current_price=current_price,
            indicators=indicators,
            signal=SignalData(**signal_result),
            backtest=BacktestResult(**backtest_result),
            metrics=TradingMetrics(**metrics_result),
            chart_data=chart_data,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 오류: {str(e)}")

# ─────────────────────────────────────────────────────────────
# 4. 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """numpy/pandas 값을 안전하게 Python float으로 변환. NaN은 None으로."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None

def _sanitize_chart_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame을 JSON 직렬화 가능한 딕셔너리 리스트로 변환. NaN → None."""
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            record[col] = _safe_float(val) if isinstance(val, (float, np.floating)) else val
        records.append(record)
    return records

def _fetch_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"데이터 없음: '{ticker}'. 올바른 yfinance 티커인지 확인하세요.")
    # MultiIndex 컬럼 평탄화 (yfinance 최신 버전 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()  # Series 보장
    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    # Bollinger Bands (20, 2σ)
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
    rsi = float(latest["rsi"])
    close = float(latest["Close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])

    rsi_score = (100 - rsi) * 0.8 if rsi > 50 else rsi * 1.2

    trend_score = 50.0
    if close > ema20 > ema50:
        trend_score = 85.0
    elif close > ema20:
        trend_score = 65.0
    elif close < ema20 < ema50:
        trend_score = 15.0
    elif close < ema20:
        trend_score = 35.0

    total = float(np.clip(rsi_score * 0.5 + trend_score * 0.5, 0, 100))

    if total >= 75:
        signal = "STRONG_BUY"
    elif total >= 60:
        signal = "BUY"
    elif total <= 25:
        signal = "STRONG_SELL"
    elif total <= 40:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "total_score": total,
        "confidence": float(abs(total - 50) * 2),
        "rsi_score": float(rsi_score),
        "trend_score": float(trend_score),
    }

def _run_simple_backtest(df: pd.DataFrame) -> dict:
    returns = df["Close"].squeeze().pct_change().dropna()
    if len(returns) < 2:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "final_capital": 1000.0}
    cum = (1 + returns).cumprod()
    total_return = float((cum.iloc[-1] - 1) * 100)
    sharpe = float((returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252))
    running_max = cum.cummax()
    max_dd = float(((cum - running_max) / (running_max + 1e-10)).min() * 100)
    win_rate = float((returns > 0).sum() / len(returns) * 100)
    final_cap = float(1000 * cum.iloc[-1])
    return {
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "final_capital": final_cap,
    }

def _calculate_trading_metrics(df: pd.DataFrame) -> dict:
    close = df["Close"].squeeze()
    n = len(close)
    returns = {}
    for period, days in {"1w": 7, "1m": 30, "3m": 90}.items():
        if n > days:
            returns[period] = float((close.iloc[-1] - close.iloc[-days]) / close.iloc[-days] * 100)
        else:
            returns[period] = 0.0

    vol_recent = float(df["Volume"].squeeze().iloc[-5:].mean())
    vol_avg = float(df["Volume"].squeeze().mean())
    buy_ratio = float(np.clip(50 + (vol_recent / (vol_avg + 1e-10) - 1) * 50, 0, 100))

    sentiment = "BULLISH" if buy_ratio > 60 else "BEARISH" if buy_ratio < 40 else "NEUTRAL"
    return {"returns": returns, "buy_sell_ratio": buy_ratio, "sentiment": sentiment}
