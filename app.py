# -*- coding: utf-8 -*-
"""
코인 AI 예측 시스템 - v2.2.0 (AI Prediction + Position Recommendation)
- TA-Lib 기반 61개 캔들스틱 패턴 지원
- 매도 시점 예측 기능 (언제 팔아야 하는지)
- 적응형 지표 계산
- 직접 입력 코인 지원
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import streamlit as st
from scipy import stats
import os
import logging
import requests
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit



from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit


# 앙상블 모델 imports
import warnings
warnings.filterwarnings('ignore')

# 딥러닝 모델
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되지 않았습니다. 딥러닝 모델을 사용할 수 없습니다.")
    # 더미 클래스 정의 (import 오류 방지)
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class ModuleList:
            pass
        class GRU:
            pass
        class LSTM:
            pass
    class torch:
        @staticmethod
        def relu(x):
            return x
        @staticmethod
        def zeros(*args, **kwargs):
            return None
        class optim:
            class Adam:
                pass
        class utils:
            class data:
                class Dataset:
                    pass
                class DataLoader:
                    pass

# 트리 기반 모델
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost가 설치되지 않았습니다.")
    # 더미 모듈
    class xgb:
        class XGBRegressor:
            pass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM이 설치되지 않았습니다.")
    # 더미 모듈
    class lgb:
        class LGBMRegressor:
            pass

# 시계열 모델
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet이 설치되지 않았습니다.")
    # 더미 클래스
    class Prophet:
        pass

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Keep-Alive 모듈 (선택적)
try:
    from keep_alive import keep_alive
    # Keep-alive 서버 시작 (백그라운드)
    keep_alive()
except ImportError:
    # keep_alive.py 파일이 없으면 무시
    pass
except Exception as e:
    # Keep-alive 실패 시에도 앱은 정상 실행
    print(f"ℹ️  Keep-alive 비활성화: {e}")

# TA-Lib 선택적 임포트
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # 경고 메시지는 메인 UI에서 표시 (st 함수는 import 시점에 호출 불가)

# ────────────────────────────────────────────────────────────────────────
# 1) Streamlit 페이지 설정
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="코인 AI 예측 시스템 v2.1",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────────────
# 2) CSS 스타일
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { padding: 1rem; }
    
    .section-title {
        font-size: 32px;
        font-weight: bold;
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 3px solid #3498DB;
        color: #2C3E50;
    }
    
    .stMetric {
        background-color: #F8F9FA;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .progress-step {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 8px;
        background: #ECF0F1;
        color: #7F8C8D;
        font-size: 14px;
    }
    
    .progress-step.active {
        background: #3498DB;
        color: white;
        font-weight: bold;
    }
    
    .pattern-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .pattern-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 12px;
    }
    
    .exit-card {
        background: linear-gradient(135deg, #F093FB 0%, #F5576C 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .exit-title {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# 3) 상수
# ────────────────────────────────────────────────────────────────────────
CRYPTO_MAP = {
    "비트코인 (BTC)": "BTCUSDT",
    "이더리움 (ETH)": "ETHUSDT",
    "리플 (XRP)": "XRPUSDT",
    "도지코인 (DOGE)": "DOGEUSDT",
    "에이다 (ADA)": "ADAUSDT",
    "솔라나 (SOL)": "SOLUSDT"
}


# ════════════════════════════════════════════════════════════════════════════
# Phase 3: 실거래 시뮬레이션 - 거래소 프리셋
# ════════════════════════════════════════════════════════════════════════════

# 거래소별 수수료 프리셋
EXCHANGE_PRESETS = {
    '바이낸스 선물': {
        'maker_fee': 0.0002,  # 0.02%
        'taker_fee': 0.0004,  # 0.04%
        'max_leverage': 125,
        'slippage_rate': 0.0005,  # 0.05% (기본 슬리피지)
        'funding_rate': 0.0001,  # 8시간당 0.01% (평균)
    },
    '바이비트 선물': {
        'maker_fee': 0.0002,  # 0.02%
        'taker_fee': 0.0006,  # 0.06%
        'max_leverage': 100,
        'slippage_rate': 0.0006,  # 0.06%
        'funding_rate': 0.0001,
    },
    '사용자 정의': {
        'maker_fee': 0.0002,
        'taker_fee': 0.0004,
        'max_leverage': 50,
        'slippage_rate': 0.0005,
        'funding_rate': 0.0001,
    }
}


def calculate_effective_leverage(nominal_leverage, stop_loss_pct, volatility):
    """
    유효 레버리지 계산
    
    Parameters:
    -----------
    nominal_leverage : float
        명목 레버리지
    stop_loss_pct : float
        손절 비율 (%)
    volatility : float
        변동성 (표준편차)
    
    Returns:
    --------
    dict : 유효 레버리지 정보
        - 'effective': 유효 레버리지
        - 'risk_adjusted': 리스크 조정 레버리지
        - 'liquidation_distance': 청산 거리 (%)
    """
    # 청산 거리 계산
    liquidation_distance = 100 / nominal_leverage  # %
    
    # 유효 레버리지 (손절 고려)
    effective = min(nominal_leverage, 100 / stop_loss_pct)
    
    # 리스크 조정 레버리지 (변동성 고려)
    risk_adjusted = effective * (1 - min(volatility / 10, 0.5))
    
    return {
        'effective': effective,
        'risk_adjusted': risk_adjusted,
        'liquidation_distance': liquidation_distance
    }


def calculate_expected_fill_price(entry_price, side, slippage_rate, market_impact=0.0):
    """
    기대 체결가 계산 (슬리피지 반영)
    
    Parameters:
    -----------
    entry_price : float
        진입 가격
    side : str
        'long' or 'short'
    slippage_rate : float
        슬리피지 비율 (0.0005 = 0.05%)
    market_impact : float
        시장 충격 비율 (큰 주문일 경우 추가)
    
    Returns:
    --------
    dict : 체결 정보
        - 'expected_price': 기대 체결가
        - 'slippage_amount': 슬리피지 금액
        - 'slippage_pct': 슬리피지 퍼센트
    """
    total_slippage = slippage_rate + market_impact
    
    if side == 'long':
        # 매수: 가격이 올라가므로 불리
        expected_price = entry_price * (1 + total_slippage)
        slippage_amount = expected_price - entry_price
    else:  # short
        # 매도: 가격이 내려가므로 불리
        expected_price = entry_price * (1 - total_slippage)
        slippage_amount = entry_price - expected_price
    
    slippage_pct = (slippage_amount / entry_price) * 100
    
    return {
        'expected_price': expected_price,
        'slippage_amount': slippage_amount,
        'slippage_pct': slippage_pct
    }


def calculate_trading_costs(position_size, entry_price, exit_price, 
                           leverage, exchange_preset, holding_hours=24):
    """
    총 거래 비용 계산 (수수료 + 슬리피지 + 펀딩 비용)
    
    Parameters:
    -----------
    position_size : float
        포지션 크기 (코인 수량)
    entry_price : float
        진입가
    exit_price : float
        청산가
    leverage : float
        레버리지
    exchange_preset : dict
        거래소 프리셋
    holding_hours : int
        보유 시간 (시간)
    
    Returns:
    --------
    dict : 비용 내역
    """
    position_value = position_size * entry_price
    
    # 1. 진입 수수료 (Taker)
    entry_fee = position_value * exchange_preset['taker_fee']
    
    # 2. 진입 슬리피지
    entry_slip = calculate_expected_fill_price(
        entry_price, 'long', exchange_preset['slippage_rate']
    )
    entry_slippage_cost = position_size * entry_slip['slippage_amount']
    
    # 3. 청산 수수료 (Taker)
    exit_value = position_size * exit_price
    exit_fee = exit_value * exchange_preset['taker_fee']
    
    # 4. 청산 슬리피지
    exit_slip = calculate_expected_fill_price(
        exit_price, 'short', exchange_preset['slippage_rate']
    )
    exit_slippage_cost = position_size * exit_slip['slippage_amount']
    
    # 5. 펀딩 비용 (8시간마다)
    funding_periods = holding_hours / 8
    funding_cost = position_value * exchange_preset['funding_rate'] * funding_periods
    
    total_cost = entry_fee + exit_fee + entry_slippage_cost + exit_slippage_cost + funding_cost
    
    return {
        'entry_fee': entry_fee,
        'exit_fee': exit_fee,
        'entry_slippage': entry_slippage_cost,
        'exit_slippage': exit_slippage_cost,
        'funding_cost': funding_cost,
        'total_cost': total_cost,
        'cost_pct': (total_cost / position_value) * 100
    }


def calibrate_talib_pattern_confidence(pattern_value, candle_range, volatility):
    """
    TA-Lib 패턴 신뢰도 교정
    
    Parameters:
    -----------
    pattern_value : int
        TA-Lib 패턴 값 (-100 ~ 100)
    candle_range : float
        봉 길이 (고가 - 저가)
    volatility : float
        변동성 (표준편차)
    
    Returns:
    --------
    float : 교정된 신뢰도 (0 ~ 100)
    """
    # 1. 기본 신뢰도 (절댓값 변환)
    base_confidence = abs(pattern_value)
    
    # 2. 봉 길이 정규화 (긴 봉일수록 신뢰도 증가)
    avg_candle_range = volatility * 2  # 평균 봉 길이 추정
    if avg_candle_range > 0:
        range_factor = min(candle_range / avg_candle_range, 2.0)  # 최대 2배
    else:
        range_factor = 1.0
    
    # 3. 변동성 대비 스케일링
    # 고변동성 시장: 패턴 신뢰도 감소
    volatility_factor = 1 / (1 + volatility / 10)
    
    # 4. 최종 신뢰도
    calibrated = base_confidence * range_factor * volatility_factor
    
    return min(calibrated, 100.0)



MAX_LEVERAGE_MAP = {
    "BTCUSDT": 125,
    "ETHUSDT": 100,
    "XRPUSDT": 75,
    "DOGEUSDT": 50,
    "ADAUSDT": 75,
    "SOLUSDT": 50
}

RESOLUTION_MAP = {
    "1분봉 (1m)": "1m",
    "5분봉 (5m)": "5m",
    "1시간봉 (1h)": "1h",
    "1일봉 (1d)": "1d"
}

# ────────────────────────────────────────────────────────────────────────
# 4) 데이터 로드
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_crypto_data(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
    interval: str = '1d'
) -> pd.DataFrame:
    """암호화폐 데이터 로드"""
    df = pd.DataFrame()
    yf_ticker = symbol[:-4] + "-USD"
    
    days_diff = (end - start).days
    
    interval_limits = {
        '1m': 7,
        '5m': 60,
        '1h': 730,
        '1d': 99999
    }
    
    max_days = interval_limits.get(interval, 99999)
    
    if days_diff > max_days:
        start = end - datetime.timedelta(days=max_days)
    
    try:
        ticker = yf.Ticker(yf_ticker)
        
        if days_diff <= 7:
            period = '7d'
        elif days_diff <= 30:
            period = '1mo'
        elif days_diff <= 90:
            period = '3mo'
        elif days_diff <= 180:
            period = '6mo'
        elif days_diff <= 365:
            period = '1y'
        elif days_diff <= 730:
            period = '2y'
        else:
            period = 'max'
        
        df_hist = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
        
        if df_hist is not None and not df_hist.empty:
            df_hist = df_hist[(df_hist.index.date >= start) & (df_hist.index.date <= end)]
            
            if not df_hist.empty:
                df = df_hist.copy()
                if 'Volume' in df.columns:
                    df = df[df['Volume'] > 0].copy()
                if not df.empty:
                    return df
    except Exception as e:
        pass
    
    if df.empty:
        try:
            ticker = yf.Ticker(yf_ticker)
            df_hist = ticker.history(
                start=start,
                end=end + datetime.timedelta(days=1),
                interval=interval,
                auto_adjust=True,
                actions=False
            )
            if df_hist is not None and not df_hist.empty:
                df = df_hist.copy()
                if 'Volume' in df.columns:
                    df = df[df['Volume'] > 0].copy()
                if not df.empty:
                    return df
        except Exception as e:
            pass

    if df.empty:
        try:
            df_max = yf.download(
                yf_ticker,
                period=period if days_diff <= 730 else '2y',
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
                actions=False
            )
            if df_max is not None and not df_max.empty:
                df = df_max.copy()
                if 'Volume' in df.columns:
                    df = df[df['Volume'] > 0].copy()
        except Exception as e:
            pass

    if df is not None and not df.empty:
        return df
    
    return pd.DataFrame()


def calculate_indicators_wilders(df: pd.DataFrame) -> pd.DataFrame:
    """적응형 지표 계산"""
    df = df.copy()
    data_len = len(df)
    
    df['일일수익률'] = df['Close'].pct_change()

    window_12 = min(12, max(3, data_len // 10))
    window_14 = min(14, max(3, data_len // 8))
    window_20 = min(20, max(5, data_len // 6))
    window_26 = min(26, max(5, data_len // 5))
    window_30 = min(30, max(5, data_len // 4))
    window_50 = min(50, max(10, data_len // 3))
    window_200 = min(200, max(20, data_len // 2))
    
    if data_len >= window_50:
        df['MA50'] = df['Close'].rolling(window=window_50).mean()
        df['EMA50'] = df['Close'].ewm(span=window_50, adjust=False).mean()
    else:
        df['MA50'] = df['Close'].rolling(window=max(3, data_len // 3)).mean()
        df['EMA50'] = df['Close'].ewm(span=max(3, data_len // 3), adjust=False).mean()
    
    if data_len >= window_200:
        df['EMA200'] = df['Close'].ewm(span=window_200, adjust=False).mean()
    else:
        df['EMA200'] = df['Close'].ewm(span=max(10, data_len // 2), adjust=False).mean()
    
    df['EMA12'] = df['Close'].ewm(span=window_12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=window_26, adjust=False).mean()

    # RSI (Wilder's Smoothing)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    period = window_14
    alpha = 1.0 / period
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    if data_len > period:
        for i in range(period, len(df)):
            avg_gain.iloc[i] = alpha * gain.iloc[i] + (1 - alpha) * avg_gain.iloc[i - 1]
            avg_loss.iloc[i] = alpha * loss.iloc[i] + (1 - alpha) * avg_loss.iloc[i - 1]
    
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI14'] = 100 - (100 / (1 + rs))

    # ATR (Wilder's Smoothing)
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['TR'] = true_range
    
    atr = true_range.rolling(window=period).mean()
    if data_len > period:
        for i in range(period, len(df)):
            atr.iloc[i] = alpha * true_range.iloc[i] + (1 - alpha) * atr.iloc[i - 1]
    
    df['ATR14'] = atr
    df['Volatility30d'] = df['일일수익률'].rolling(window=window_30).std()

    # Stochastic
    df['StochK14'] = 0.0
    if data_len >= window_14:
        low14 = df['Low'].rolling(window=window_14).min()
        high14 = df['High'].rolling(window=window_14).max()
        df['StochK14'] = (df['Close'] - low14) / (high14 - low14 + 1e-8) * 100

    # MFI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['MF'] = typical_price * df['Volume']
    # 조건부 할당 (pandas 호환성 개선)
    price_change = df['Close'].diff()
    df['PosMF'] = df['MF'].copy()
    df.loc[price_change <= 0, 'PosMF'] = 0
    df['NegMF'] = df['MF'].copy()
    df.loc[price_change >= 0, 'NegMF'] = 0
    roll_pos = df['PosMF'].rolling(window=window_14).sum()
    roll_neg = df['NegMF'].rolling(window=window_14).sum()
    df['MFI14'] = 100 - (100 / (1 + roll_pos / (roll_neg + 1e-8)))

    # VWAP
    df['PV'] = df['Close'] * df['Volume']
    df['Cum_PV'] = df['PV'].cumsum()
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / (df['Cum_Vol'] + 1e-8)

    df['Vol_MA20'] = df['Volume'].rolling(window=window_20).mean()

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # EMA 교차
    df['Cross_Signal'] = 0
    ema50 = df['EMA50']
    ema200 = df['EMA200']
    cond_up = (ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))
    cond_down = (ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))
    df.loc[cond_up, 'Cross_Signal'] = 1
    df.loc[cond_down, 'Cross_Signal'] = -1

    essential_cols = ['Close', 'High', 'Low', 'Volume', '일일수익률']
    df_clean = df.dropna(subset=essential_cols)
    
    optional_cols = ['RSI14', 'ATR14', 'StochK14', 'MFI14', 'MACD', 'MACD_Signal']
    for col in optional_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    
    return df_clean


def detect_candlestick_patterns_basic(df: pd.DataFrame) -> list:
    """기본 3개 패턴 감지 (TA-Lib 없을 때)"""
    patterns = []
    
    if len(df) < 3:
        return []
    
    df_sorted = df.sort_index()
    
    for i in range(2, len(df_sorted)):
        o1, c1, h1, l1 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i - 2]
        date1 = df_sorted.index[i - 2]
        
        o2, c2, h2, l2 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i - 1]
        date2 = df_sorted.index[i - 1]
        
        o3, c3, h3, l3 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i]
        date3 = df_sorted.index[i]

        # Three White Soldiers
        if (c1 > o1) and (c2 > o2) and (c3 > o3) and (c2 > c1) and (c3 > c2):
            patterns.append({
                'name': '⚪ Three White Soldiers',
                'category': '3-캔들',
                'date': date3,
                'conf': 100.0,
                'desc': '세 개의 연속 양봉',
                'impact': '강력한 상승 신호',
                'direction': '상승'
            })

        # Morning Star
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        range2 = (h2 - l2) if (h2 - l2) != 0 else 1e-8
        if (c1 < o1) and (body2 < range2 * 0.3) and (c2 < c1) and (o2 < c1) and \
           (c3 > o3) and (c3 > (o1 + c1) / 2):
            conf = min((body3 / (h3 - l3 + 1e-8)) * 100, 100.0)
            patterns.append({
                'name': '🌅 Morning Star',
                'category': '3-캔들',
                'date': date3,
                'conf': round(conf, 2),
                'desc': '하락 후 반전 신호',
                'impact': '상승 전환 가능성',
                'direction': '상승'
            })

        # Doji
        if abs(o3 - c3) <= (h3 - l3) * 0.1:
            patterns.append({
                'name': '✖️ Doji',
                'category': '단일',
                'date': date3,
                'conf': 100.0,
                'desc': '매수/매도 균형',
                'impact': '추세 전환 가능성',
                'direction': '중립'
            })

    # 같은 패턴명은 최신 1개만
    unique_patterns = {}
    for pattern in reversed(patterns):
        pattern_name = pattern['name']
        if pattern_name not in unique_patterns:
            unique_patterns[pattern_name] = pattern
    
    result = list(unique_patterns.values())
    result.sort(key=lambda x: x['date'], reverse=True)
    
    return result[:10]


def detect_candlestick_patterns_talib(df: pd.DataFrame) -> list:
    """TA-Lib 기반 61개 패턴 감지"""
    patterns = []
    
    if len(df) < 5:  # 최소 5개 필요 (일부 패턴이 5봉 요구)
        return []
    
    df_sorted = df.sort_index()
    open_prices = df_sorted['Open'].values
    high_prices = df_sorted['High'].values
    low_prices = df_sorted['Low'].values
    close_prices = df_sorted['Close'].values
    
    # TA-Lib 패턴 정의 (58개 + 기존 3개 = 61개)
    pattern_functions = {
        # 단일(1-캔들) - 15개
        'CDLBELTHOLD': ('🔨 Belt Hold', '벨트 홀드', '단일'),
        'CDLCLOSINGMARUBOZU': ('📊 Closing Marubozu', '종가 마루보즈', '단일'),
        'CDLMARUBOZU': ('📏 Marubozu', '마루보즈', '단일'),
        'CDLLONGLINE': ('📐 Long Line', '장대봉', '단일'),
        'CDLSHORTLINE': ('📌 Short Line', '단봉', '단일'),
        'CDLSPINNINGTOP': ('🌪️ Spinning Top', '팽이형', '단일'),
        'CDLHIGHWAVE': ('🌊 High Wave', '높은 파동형', '단일'),
        'CDLHAMMER': ('🔨 Hammer', '해머', '단일'),
        'CDLHANGINGMAN': ('👤 Hanging Man', '교수형', '단일'),
        'CDLINVERTEDHAMMER': ('🔧 Inverted Hammer', '역망치', '단일'),
        'CDLSHOOTINGSTAR': ('⭐ Shooting Star', '유성형', '단일'),
        'CDLRICKSHAWMAN': ('🚶 Rickshaw Man', '릭샤맨', '단일'),
        'CDLTAKURI': ('🎣 Takuri', '타쿠리', '단일'),
        'CDLKICKING': ('👟 Kicking', '킥킹', '단일'),
        'CDLKICKINGBYLENGTH': ('👢 Kicking by Length', '킥킹(길이 기준)', '단일'),
        
        # 2-캔들 - 12개
        'CDLENGULFING': ('🫂 Engulfing', '포용형', '2-캔들'),
        'CDLHARAMI': ('🤰 Harami', '하라미', '2-캔들'),
        'CDLHARAMICROSS': ('➕ Harami Cross', '하라미 크로스', '2-캔들'),
        'CDLPIERCING': ('🎯 Piercing', '관통형', '2-캔들'),
        'CDLDARKCLOUDCOVER': ('☁️ Dark Cloud Cover', '암운형', '2-캔들'),
        'CDLCOUNTERATTACK': ('⚔️ Counterattack', '반격선', '2-캔들'),
        'CDLONNECK': ('🦢 On Neck', '온넥', '2-캔들'),
        'CDLINNECK': ('🦆 In Neck', '인넥', '2-캔들'),
        'CDLTHRUSTING': ('🗡️ Thrusting', '스러스팅', '2-캔들'),
        'CDLSEPARATINGLINES': ('↔️ Separating Lines', '세퍼레이팅 라인', '2-캔들'),
        'CDLMATCHINGLOW': ('🎯 Matching Low', '매칭 로우', '2-캔들'),
        'CDLHOMINGPIGEON': ('🕊️ Homing Pigeon', '호밍 피전', '2-캔들'),
        
        # 3-캔들 - 11개
        'CDL2CROWS': ('🐦 Two Crows', '투 크로우즈', '3-캔들'),
        'CDL3INSIDE': ('📦 Three Inside', '삼내부', '3-캔들'),
        'CDL3OUTSIDE': ('📤 Three Outside', '삼외부', '3-캔들'),
        'CDL3LINESTRIKE': ('⚡ Three Line Strike', '쓰리 라인 스트라이크', '3-캔들'),
        'CDL3BLACKCROWS': ('🐦‍⬛ Three Black Crows', '세 검은 까마귀', '3-캔들'),
        'CDLIDENTICAL3CROWS': ('🦅 Identical Three Crows', '동일 삼까마귀', '3-캔들'),
        'CDLUNIQUE3RIVER': ('🏞️ Unique 3 River', '유니크 쓰리 리버', '3-캔들'),
        'CDL3STARSINSOUTH': ('⭐ Three Stars in South', '남쪽의 세 별', '3-캔들'),
        'CDLUPSIDEGAP2CROWS': ('📈 Upside Gap Two Crows', '업사이드 갭 투 크로우즈', '3-캔들'),
        'CDLEVENINGSTAR': ('🌆 Evening Star', '석별형', '3-캔들'),
        'CDLTRISTAR': ('✨ Tristar', '트리스타', '3-캔들'),
        
        # 갭/지속/복합 - 9개
        'CDLBREAKAWAY': ('🚀 Breakaway', '브레이크어웨이', '복합'),
        'CDLRISEFALL3METHODS': ('📊 Rising/Falling 3 Methods', '상승하락 삼법', '복합'),
        'CDLMATHOLD': ('🤝 Mat Hold', '매트 홀드', '복합'),
        'CDLTASUKIGAP': ('📏 Tasuki Gap', '타스키 갭', '복합'),
        'CDLGAPSIDESIDEWHITE': ('⬜ Gap Side-by-Side White', '갭 사이드바이사이드', '복합'),
        'CDLXSIDEGAP3METHODS': ('📈 Gap Three Methods', '갭 쓰리 메서즈', '복합'),
        'CDLABANDONEDBABY': ('👶 Abandoned Baby', '어밴던드 베이비', '복합'),
        'CDLCONCEALBABYSWALL': ('🐦 Concealing Baby Swallow', '컨실링 베이비', '복합'),
        'CDLLADDERBOTTOM': ('🪜 Ladder Bottom', '래더 바텀', '복합'),
        
        # 특수 - 5개
        'CDLADVANCEBLOCK': ('🚧 Advance Block', '전진 봉쇄', '특수'),
        'CDLSTALLEDPATTERN': ('⏸️ Stalled Pattern', '정체 패턴', '특수'),
        'CDLSTICKSANDWICH': ('🥪 Stick Sandwich', '스틱 샌드위치', '특수'),
        'CDLHIKKAKE': ('🎣 Hikkake', '힛카케', '특수'),
        'CDLHIKKAKEMOD': ('🎯 Modified Hikkake', '수정 힛카케', '특수'),
        
        # 기존 3개 (TA-Lib에도 있지만 명시적으로 추가)
        'CDL3WHITESOLDIERS': ('⚪ Three White Soldiers', '세 개의 연속 양봉', '3-캔들'),
        'CDLMORNINGSTAR': ('🌅 Morning Star', '하락 후 반전 신호', '3-캔들'),
        'CDLDOJI': ('✖️ Doji', '매수/매도 균형', '단일'),
    }
    
    # 각 패턴 감지
    for func_name, (emoji_name, korean_name, category) in pattern_functions.items():
        try:
            if not hasattr(talib, func_name):
                continue
                
            pattern_func = getattr(talib, func_name)
            result = pattern_func(open_prices, high_prices, low_prices, close_prices)
            
            # 패턴 발생 지점 찾기
            for i, value in enumerate(result):
                if value != 0:  # 0이 아니면 패턴 발생
                    # 신뢰도 변환: -100~100 → 0~100%
                    confidence = abs(value)
                    
                    # 방향 판단
                    if value > 0:
                        direction = '상승'
                        impact = '상승 신호'
                    elif value < 0:
                        direction = '하락'
                        impact = '하락 신호'
                    else:
                        direction = '중립'
                        impact = '추세 전환 가능성'
                    
                    patterns.append({
                        'name': emoji_name,
                        'category': category,
                        'date': df_sorted.index[i],
                        'conf': confidence,
                        'desc': korean_name,
                        'impact': impact,
                        'direction': direction
                    })
        except Exception as e:
            continue
    
    # 같은 패턴명은 최신 1개만
    unique_patterns = {}
    for pattern in reversed(patterns):
        pattern_name = pattern['name']
        if pattern_name not in unique_patterns:
            unique_patterns[pattern_name] = pattern
    
    result = list(unique_patterns.values())
    result.sort(key=lambda x: x['date'], reverse=True)
    
    return result[:10]  # 최대 10개


def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    """캔들스틱 패턴 감지 (TA-Lib 있으면 61개, 없으면 3개)"""
    if TALIB_AVAILABLE:
        return detect_candlestick_patterns_talib(df)
    else:
        return detect_candlestick_patterns_basic(df)


def calculate_exit_strategy(df: pd.DataFrame, entry_price: float, atr: float, 
                            investment_amount: float, leverage: float) -> dict:
    """
    매도 시점 예측
    - 보수적/중립/공격적 시나리오 제공
    - ATR 기반 동적 손절/익절
    - 추세 전환 신호 감지
    """
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI14'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    
    # 추세 판단
    trend = 'bullish' if ema50 > ema200 else 'bearish'
    
    # 3가지 시나리오
    scenarios = {}
    
    # 1. 보수적 (빠른 익절, 손절)
    scenarios['conservative'] = {
        'name': '🛡️ 보수적 전략',
        'take_profit': entry_price + (atr * 1.5),
        'stop_loss': entry_price - (atr * 1.0),
        'holding_period': '1-3일',
        'description': '빠른 수익 실현, 리스크 최소화',
        'rr_ratio': 1.5,
        'exit_signals': [
            'RSI > 70 (과매수)',
            'EMA50 하향 돌파',
            '목표 수익률 5% 도달'
        ]
    }
    
    # 2. 중립적 (균형잡힌 접근)
    scenarios['neutral'] = {
        'name': '⚖️ 중립적 전략',
        'take_profit': entry_price + (atr * 2.5),
        'stop_loss': entry_price - (atr * 1.5),
        'holding_period': '3-7일',
        'description': '리스크-수익 균형',
        'rr_ratio': 1.67,
        'exit_signals': [
            'RSI > 75 (강한 과매수)',
            'EMA50/200 데드크로스',
            '목표 수익률 10% 도달'
        ]
    }
    
    # 3. 공격적 (큰 수익 추구)
    scenarios['aggressive'] = {
        'name': '🚀 공격적 전략',
        'take_profit': entry_price + (atr * 4.0),
        'stop_loss': entry_price - (atr * 2.0),
        'holding_period': '7-14일',
        'description': '큰 수익 추구, 높은 리스크',
        'rr_ratio': 2.0,
        'exit_signals': [
            'RSI > 80 (극심한 과매수)',
            '주요 저항선 도달',
            '목표 수익률 20% 도달'
        ]
    }
    
    # 추세 기반 조정
    if trend == 'bearish':
        # 하락 추세에서는 더 보수적으로
        for scenario in scenarios.values():
            scenario['take_profit'] *= 0.8
            scenario['stop_loss'] *= 1.2
    
    # 현재 상태 평가
    current_status = {
        'current_price': current_price,
        'entry_price': entry_price,
        'unrealized_pnl': (current_price - entry_price) / entry_price * 100,
        'rsi_status': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral',
        'trend': trend,
        'recommendation': None
    }
    
    # 즉시 매도 권장 조건
    if rsi > 80 and current_status['unrealized_pnl'] > 10:
        current_status['recommendation'] = '⚠️ 즉시 매도 고려 (극심한 과매수 + 높은 수익)'
    elif trend == 'bearish' and current_status['unrealized_pnl'] < -5:
        current_status['recommendation'] = '⚠️ 손절 고려 (하락 추세 + 손실 확대)'
    elif current_status['unrealized_pnl'] > 20:
        current_status['recommendation'] = '✅ 부분 익절 고려 (높은 수익 달성)'
    else:
        current_status['recommendation'] = '⏳ 홀딩 유지'
    
    return {
        'scenarios': scenarios,
        'current_status': current_status,
        'atr': atr,
        'trend': trend
    }


# 기타 함수들 (기존 유지)

# ────────────────────────────────────────────────────────────────────────
# [추가됨] AI 예측 및 포지션 추천 함수 (v2.2.0)
# ────────────────────────────────────────────────────────────────────────

def predict_trend_with_ai(df: pd.DataFrame, current_price: float, 
                          ema_short: float, ema_long: float, 
                          rsi: float, macd: float, macd_signal: float) -> dict:
    """
    AI 기반 단기 추세 예측
    
    Returns:
        dict: {
            'trend': 'bullish' | 'bearish' | 'neutral',
            'confidence': float (0-100),
            'reasoning': str,
            'signal_strength': float (0-100)
        }
    """
    signals = []
    weights = []
    reasons = []
    
    # 1. 이동평균 분석 (가중치: 30%)
    if ema_short > ema_long:
        ma_diff_pct = ((ema_short - ema_long) / ema_long) * 100
        if ma_diff_pct > 2:
            signals.append(1.0)
            reasons.append(f"골든크로스 강세 (차이 {ma_diff_pct:.1f}%)")
        elif ma_diff_pct > 0.5:
            signals.append(0.7)
            reasons.append(f"골든크로스 ({ma_diff_pct:.1f}%)")
        else:
            signals.append(0.5)
            reasons.append("단기 이평선 우위")
        weights.append(0.30)
    else:
        ma_diff_pct = ((ema_long - ema_short) / ema_long) * 100
        if ma_diff_pct > 2:
            signals.append(-1.0)
            reasons.append(f"데드크로스 약세 (차이 {ma_diff_pct:.1f}%)")
        elif ma_diff_pct > 0.5:
            signals.append(-0.7)
            reasons.append(f"데드크로스 ({ma_diff_pct:.1f}%)")
        else:
            signals.append(-0.5)
            reasons.append("장기 이평선 우위")
        weights.append(0.30)
    
    # 2. RSI 분석 (가중치: 25%)
    if rsi > 70:
        signals.append(-0.8)
        reasons.append(f"과매수 영역 (RSI {rsi:.1f})")
        weights.append(0.25)
    elif rsi > 60:
        signals.append(0.3)
        reasons.append(f"강세 영역 (RSI {rsi:.1f})")
        weights.append(0.25)
    elif rsi < 30:
        signals.append(0.8)
        reasons.append(f"과매도 영역 (RSI {rsi:.1f})")
        weights.append(0.25)
    elif rsi < 40:
        signals.append(-0.3)
        reasons.append(f"약세 영역 (RSI {rsi:.1f})")
        weights.append(0.25)
    else:
        signals.append(0.0)
        reasons.append(f"중립 영역 (RSI {rsi:.1f})")
        weights.append(0.25)
    
    # 3. MACD 분석 (가중치: 25%)
    macd_diff = macd - macd_signal
    if macd_diff > 0:
        if macd > 0:
            signals.append(0.9)
            reasons.append("MACD 상승 모멘텀")
        else:
            signals.append(0.5)
            reasons.append("MACD 반전 신호")
        weights.append(0.25)
    else:
        if macd < 0:
            signals.append(-0.9)
            reasons.append("MACD 하락 모멘텀")
        else:
            signals.append(-0.5)
            reasons.append("MACD 약화 신호")
        weights.append(0.25)
    
    # 4. 거래량 분석 (가중치: 20%)
    if 'Volume' in df.columns and len(df) > 20:
        recent_volume = df['Volume'].iloc[-5:].mean()
        avg_volume = df['Volume'].iloc[-20:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 1.5:
            current_trend = 1 if ema_short > ema_long else -1
            signals.append(current_trend * 0.7)
            reasons.append(f"거래량 급증 ({volume_ratio:.1f}배)")
            weights.append(0.20)
        elif volume_ratio > 1.2:
            current_trend = 1 if ema_short > ema_long else -1
            signals.append(current_trend * 0.4)
            reasons.append(f"거래량 증가 ({volume_ratio:.1f}배)")
            weights.append(0.20)
        elif volume_ratio < 0.7:
            signals.append(0.0)
            reasons.append(f"거래량 감소 ({volume_ratio:.1f}배)")
            weights.append(0.20)
        else:
            signals.append(0.0)
            reasons.append("거래량 평균 수준")
            weights.append(0.20)
    
    # 가중 평균 계산
    weighted_signal = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
    
    # 추세 판단
    if weighted_signal > 0.3:
        trend = 'bullish'
        trend_kr = '상승'
    elif weighted_signal < -0.3:
        trend = 'bearish'
        trend_kr = '하락'
    else:
        trend = 'neutral'
        trend_kr = '보합'
    
    # 신뢰도 계산
    confidence = min(abs(weighted_signal) * 100, 100)
    signal_strength = (weighted_signal + 1) * 50
    
    top_reasons = reasons[:2] if len(reasons) >= 2 else reasons
    reasoning = " + ".join(top_reasons)
    
    return {
        'trend': trend,
        'trend_kr': trend_kr,
        'confidence': round(confidence, 1),
        'reasoning': reasoning,
        'signal_strength': round(signal_strength, 1),
        'weighted_signal': weighted_signal
    }


def recommend_position(ai_prediction: dict, current_price: float, 
                      stop_loss: float, take_profit: float,
                      volatility: float) -> dict:
    """
    AI 예측을 기반으로 포지션 추천
    """
    trend = ai_prediction['trend']
    confidence = ai_prediction['confidence']
    
    # 포지션 결정
    if trend == 'bullish' and confidence > 50:
        position = 'LONG'
        position_kr = '롱 포지션'
        probability = 50 + (confidence * 0.5)
        base_reason = ai_prediction['reasoning']
    elif trend == 'bearish' and confidence > 50:
        position = 'SHORT'
        position_kr = '숏 포지션'
        probability = 50 + (confidence * 0.5)
        base_reason = ai_prediction['reasoning']
    else:
        position = 'NEUTRAL'
        position_kr = '관망'
        probability = 50
        base_reason = "명확한 추세 없음"
    
    # 리스크 레벨
    if volatility > 0.05:
        risk_level = 'HIGH'
        risk_kr = '높음'
        risk_adjustment = -5
    elif volatility > 0.03:
        risk_level = 'MEDIUM'
        risk_kr = '중간'
        risk_adjustment = 0
    else:
        risk_level = 'LOW'
        risk_kr = '낮음'
        risk_adjustment = +5
    
    final_probability = min(max(probability + risk_adjustment, 45), 85)
    
    if position == 'NEUTRAL':
        reasoning = f"{base_reason}, 시장 변동성 {risk_kr}"
        recommendation_text = "현재 데이터 기준, **관망(보류)** 를 권장합니다."
    else:
        reasoning = f"{base_reason}, 시장 변동성 {risk_kr}"
        recommendation_text = f"현재 데이터 기준, **{position_kr}** 이 우세(약 **{final_probability:.0f}%**) 로 판단됩니다."
    
    # 손익비
    if position == 'LONG':
        potential_profit = ((take_profit - current_price) / current_price) * 100
        potential_loss = ((current_price - stop_loss) / current_price) * 100
    elif position == 'SHORT':
        potential_profit = ((current_price - stop_loss) / current_price) * 100
        potential_loss = ((take_profit - current_price) / current_price) * 100
    else:
        potential_profit = 0
        potential_loss = 0
    
    rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
    
    return {
        'position': position,
        'position_kr': position_kr,
        'probability': round(final_probability, 1),
        'reasoning': reasoning,
        'risk_level': risk_level,
        'risk_kr': risk_kr,
        'recommendation_text': recommendation_text,
        'potential_profit_pct': round(potential_profit, 2),
        'potential_loss_pct': round(potential_loss, 2),
        'risk_reward_ratio': round(rr_ratio, 2)
    }


def render_ai_prediction(ai_prediction: dict, current_price: float):
    """[추가됨] 🤖 AI 예측 결과 섹션"""
    st.markdown("<div class='section-title'>🤖 AI 예측 결과</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 단기 추세 예측")
        trend_emoji = {'bullish': '📈', 'bearish': '📉', 'neutral': '➡️'}
        trend_color = {'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}
        trend = ai_prediction['trend']
        st.markdown(f"<h2 style='color:{trend_color[trend]};'>{trend_emoji[trend]} {ai_prediction['trend_kr']}</h2>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 예측 신뢰도")
        confidence = ai_prediction['confidence']
        if confidence >= 70:
            bar_color = 'green'
            conf_text = '높음'
        elif confidence >= 50:
            bar_color = 'orange'
            conf_text = '중간'
        else:
            bar_color = 'red'
            conf_text = '낮음'
        st.markdown(f"""
        <div style='background-color: #f0f0f0; border-radius: 10px; padding: 10px;'>
            <div style='background-color: {bar_color}; width: {confidence}%; height: 30px; 
                        border-radius: 5px; text-align: center; line-height: 30px; color: white; font-weight: bold;'>
                {confidence:.1f}%
            </div>
            <p style='text-align: center; margin-top: 5px; color: #666;'>{conf_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 💡 예측 근거")
        st.info(ai_prediction['reasoning'])
    
    signal_strength = ai_prediction['signal_strength']
    if signal_strength > 60:
        gauge_color = 'linear-gradient(90deg, #28a745 0%, #218838 100%)'
        gauge_text = '강한 매수 신호'
    elif signal_strength > 55:
        gauge_color = 'linear-gradient(90deg, #90EE90 0%, #28a745 100%)'
        gauge_text = '매수 신호'
    elif signal_strength >= 45:
        gauge_color = 'linear-gradient(90deg, #FFA500 0%, #FF8C00 100%)'
        gauge_text = '중립'
    elif signal_strength >= 40:
        gauge_color = 'linear-gradient(90deg, #FF6347 0%, #FFA500 100%)'
        gauge_text = '매도 신호'
    else:
        gauge_color = 'linear-gradient(90deg, #dc3545 0%, #c82333 100%)'
        gauge_text = '강한 매도 신호'
    
    st.markdown(f"""
    <div style='background-color: #e0e0e0; border-radius: 20px; height: 40px; position: relative; overflow: hidden; margin-top: 20px;'>
        <div style='background: {gauge_color}; width: {signal_strength}%; height: 100%; border-radius: 20px;'></div>
        <div style='position: absolute; top: 0; left: 0; right: 0; text-align: center; 
                    line-height: 40px; color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
            {gauge_text} ({signal_strength:.0f}/100)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


def render_position_recommendation(position_rec: dict):
    """[추가됨] 포지션 추천 (매매 전략 내부)"""
    st.markdown("#### 🎯 포지션 추천")
    position = position_rec['position']
    if position == 'LONG':
        bg_color = '#d4edda'
        border_color = '#28a745'
        icon = '📈'
    elif position == 'SHORT':
        bg_color = '#f8d7da'
        border_color = '#dc3545'
        icon = '📉'
    else:
        bg_color = '#fff3cd'
        border_color = '#ffc107'
        icon = '⏸️'
    
    st.markdown(f"""
    <div style='background-color: {bg_color}; border-left: 5px solid {border_color}; 
                padding: 20px; border-radius: 10px; margin: 10px 0;'>
        <h3 style='margin: 0; color: {border_color};'>{icon} {position_rec['recommendation_text']}</h3>
        <p style='margin: 10px 0 0 0; color: #666;'>
            <strong>추천 이유:</strong> {position_rec['reasoning']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="포지션", value=position_rec['position_kr'])
    with col2:
        st.metric(label="확률", value=f"{position_rec['probability']:.0f}%")
    with col3:
        st.metric(label="리스크", value=position_rec['risk_kr'])
    with col4:
        if position != 'NEUTRAL':
            st.metric(label="손익비", value=f"{position_rec['risk_reward_ratio']:.2f}")
    
    if position != 'NEUTRAL':
        st.markdown("##### 💰 예상 손익")
        col_profit, col_loss = st.columns(2)
        with col_profit:
            st.success(f"**목표 수익:** +{position_rec['potential_profit_pct']:.2f}%")
        with col_loss:
            st.error(f"**최대 손실:** -{position_rec['potential_loss_pct']:.2f}%")
    
    st.warning("""
    ⚠️ **주의사항**  
    - 이 추천은 과거 데이터 기반 확률적 예측이며, 투자 권유가 아닙니다.  
    - 실제 투자 시 본인의 리스크 허용 범위 내에서 결정하시기 바랍니다.  
    - 시장 상황은 실시간으로 변하므로, 지속적인 모니터링이 필요합니다.
    """)


def calculate_optimized_leverage(investment_amount: float, volatility: float, 
                                 atr_ratio: float, confidence: float, max_leverage: int, 
                                 crypto_name: str = "BTC") -> dict:
    """
    레버리지 최적화 (v2.3.0)
    
    Returns:
        dict: {
            'recommended': 권장 레버리지,
            'maximum': 최대 레버리지,
            'risk_level': 리스크 레벨
        }
    """
    base_leverage = 10
    
    # [추가됨] v2.3.0: 코인별 리스크 팩터 (변동성 기반)
    crypto_risk_factors = {
        'BTC': 1.0,   # 비트코인: 기준 (가장 안정적)
        'ETH': 1.1,   # 이더리움: 약간 높은 변동성
        'BNB': 1.2,   # 바이낸스: 중간 변동성
        'XRP': 1.3,   # 리플: 높은 변동성
        'ADA': 1.3,   # 카르다노: 높은 변동성
        'SOL': 1.4,   # 솔라나: 매우 높은 변동성
        'DOGE': 1.5,  # 도지코인: 극심한 변동성
        'MATIC': 1.4, # 폴리곤: 매우 높은 변동성
        'DOT': 1.3,   # 폴카닷: 높은 변동성
        'AVAX': 1.4,  # 아발란체: 매우 높은 변동성
    }
    
    # 코인 이름에서 기호 추출 (예: "Bitcoin (BTC)" -> "BTC")
    crypto_symbol = crypto_name
    for symbol in crypto_risk_factors.keys():
        if symbol in crypto_name.upper():
            crypto_symbol = symbol
            break
    
    crypto_factor = crypto_risk_factors.get(crypto_symbol, 1.2)  # 기본값: 중간 리스크
    
    # [추가됨] v2.3.0: 투자 금액에 따른 세분화된 팩터
    # 대량 투자자 → 리스크 감수 능력 높음 → 레버리지 여유
    # 소액 투자자 → 손실 회복 어려움 → 보수적 레버리지
    if investment_amount >= 50000:
        investment_factor = 1.3  # 초대량 투자 → 레버리지 여유
    elif investment_amount >= 20000:
        investment_factor = 1.2  # 대량 투자 → 레버리지 여유
    elif investment_amount >= 10000:
        investment_factor = 1.1  # 상당 투자 → 약간 여유
    elif investment_amount >= 5000:
        investment_factor = 1.0  # 기준
    elif investment_amount >= 2000:
        investment_factor = 0.9  # 중간 → 약간 신중
    elif investment_amount >= 1000:
        investment_factor = 0.8  # 소액 → 신중
    else:
        investment_factor = 0.6  # 극소액 → 매우 보수적
    
    # 변동성 팩터
    if volatility < 0.02:
        volatility_factor = 1.5  # 낮은 변동성 → 레버리지 여유
    elif volatility < 0.05:
        volatility_factor = 1.2  # 중간 변동성
    elif volatility < 0.10:
        volatility_factor = 0.9  # 높은 변동성
    else:
        volatility_factor = 0.7  # 극심한 변동성 → 보수적
    
    confidence_factor = confidence / 100.0
    atr_factor = 1.0 / (atr_ratio + 0.5)
    
    # [추가됨] v2.3.0: 권장 레버리지 계산 (보수적)
    recommended_leverage = base_leverage * investment_factor * volatility_factor * confidence_factor * atr_factor / crypto_factor
    recommended_leverage = max(1.0, min(recommended_leverage, float(max_leverage) * 0.7))  # 최대 레버리지의 70%까지
    recommended_leverage = round(recommended_leverage, 1)
    
    # [추가됨] v2.3.0: 최대 레버리지 계산 (공격적)
    maximum_leverage = recommended_leverage * 1.5  # 권장의 1.5배
    maximum_leverage = max(recommended_leverage + 1, min(maximum_leverage, float(max_leverage)))
    maximum_leverage = round(maximum_leverage, 1)
    
    # [추가됨] v2.3.0: 리스크 레벨 판단
    risk_score = crypto_factor * volatility * 100
    if risk_score < 3:
        risk_level = "중간"
    elif risk_score < 6:
        risk_level = "중간"
    else:
        risk_level = "중간"
    
    return {
        'recommended': recommended_leverage,
        'maximum': maximum_leverage,
        'risk_level': risk_level
    }





# ════════════════════════════════════════════════════════════════════════════
# Phase 2: 워크-포워드 검증 (Walk-Forward Validation)
# ════════════════════════════════════════════════════════════════════════════

def walk_forward_validation(data, n_splits=5, forecast_horizon=3, seasonal_period=7, seasonal_type='add'):
    """
    워크-포워드 검증 (시계열 교차 검증)
    
    Parameters:
    -----------
    data : pd.Series
        시계열 데이터
    n_splits : int
        분할 개수 (기본 5)
    forecast_horizon : int
        예측 기간 (기본 3)
    seasonal_period : int
        계절성 주기
    seasonal_type : str
        계절성 타입 ('add' or 'mul')
    
    Returns:
    --------
    dict : 검증 결과
        - 'scores': 각 폴드별 점수 리스트
        - 'mean_score': 평균 점수
        - 'std_score': 표준편차
        - 'direction_accuracy': 방향 정확도 리스트
        - 'mean_direction': 평균 방향 정확도
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    mase_scores = []
    direction_accuracies = []
    
    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # 테스트 데이터가 예측 기간보다 작으면 스킵
        if len(test_data) < forecast_horizon:
            continue
        
        # 최신 500개 데이터만 사용 (성능 최적화)
        if len(train_data) > 500:
            train_data = train_data[-500:]
        
        try:
            # 계절성 모델 시도
            if seasonal_period and len(train_data) >= 2 * seasonal_period:
                model = ExponentialSmoothing(
                    train_data,
                    seasonal_periods=seasonal_period,
                    trend='add',
                    seasonal=seasonal_type,
                    initialization_method="estimated"
                )
                fitted = model.fit()
            else:
                # 비계절 모델
                model = SimpleExpSmoothing(train_data, initialization_method="estimated")
                fitted = model.fit()
            
            # 예측
            forecast = fitted.forecast(steps=forecast_horizon)
            actual = test_data.iloc[:forecast_horizon]
            
            # MASE 계산
            naive_errors = np.abs(np.diff(train_data))
            scale = np.mean(naive_errors)
            
            if scale > 0:
                errors = np.abs(actual.values - forecast.values)
                mase = np.mean(errors) / scale
                mase_scores.append(mase)
            
            # 방향 정확도 계산
            if len(actual) > 1:
                actual_direction = (actual.values[1:] > actual.values[:-1]).astype(int)
                forecast_direction = (forecast.values[1:] > forecast.values[:-1]).astype(int)
                direction_acc = np.mean(actual_direction == forecast_direction) * 100
                direction_accuracies.append(direction_acc)
        
        except Exception as e:
            continue
    
    if not mase_scores:
        return None
    
    return {
        'scores': mase_scores,
        'mean_score': np.mean(mase_scores),
        'std_score': np.std(mase_scores),
        'direction_accuracy': direction_accuracies,
        'mean_direction': np.mean(direction_accuracies) if direction_accuracies else 0.0
    }


def calculate_brier_score(actual_direction, predicted_probs):
    """
    Brier Score 계산 (확률 예측 정확도)
    
    Parameters:
    -----------
    actual_direction : array-like
        실제 방향 (0: 하락, 1: 상승)
    predicted_probs : array-like
        예측 확률 (0~1 사이)
    
    Returns:
    --------
    float : Brier Score (낮을수록 좋음, 0~1)
    """
    try:
        score = brier_score_loss(actual_direction, predicted_probs)
        return score
    except Exception as e:
        return None


def calculate_log_loss_score(actual_direction, predicted_probs):
    """
    Log Loss 계산 (확률 예측 손실)
    
    Parameters:
    -----------
    actual_direction : array-like
        실제 방향 (0: 하락, 1: 상승)
    predicted_probs : array-like
        예측 확률 (0~1 사이)
    
    Returns:
    --------
    float : Log Loss (낮을수록 좋음)
    """
    try:
        # 확률을 0.01~0.99로 클리핑 (log(0) 방지)
        predicted_probs = np.clip(predicted_probs, 0.01, 0.99)
        score = log_loss(actual_direction, predicted_probs)
        return score
    except Exception as e:
        return None


def calculate_direction_metrics(actual, predicted):
    """
    방향 예측 메트릭 종합 계산
    
    Parameters:
    -----------
    actual : array-like
        실제 가격 시계열
    predicted : array-like
        예측 가격 시계열
    
    Returns:
    --------
    dict : 방향 메트릭
        - 'direction_accuracy': 방향 정확도 (%)
        - 'brier_score': Brier Score
        - 'log_loss': Log Loss
        - 'up_accuracy': 상승 방향 정확도 (%)
        - 'down_accuracy': 하락 방향 정확도 (%)
    """
    if len(actual) < 2 or len(predicted) < 2:
        return None
    
    # 방향 계산 (1: 상승, 0: 하락)
    actual_direction = (actual[1:] > actual[:-1]).astype(int)
    predicted_direction = (predicted[1:] > predicted[:-1]).astype(int)
    
    # 방향 정확도
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # 확률 계산 (예측값의 변화율을 시그모이드 변환)
    predicted_change_rate = (predicted[1:] - predicted[:-1]) / (predicted[:-1] + 1e-10)
    predicted_probs = 1 / (1 + np.exp(-predicted_change_rate * 10))  # 시그모이드
    
    # Brier Score & Log Loss
    brier = calculate_brier_score(actual_direction, predicted_probs)
    logloss = calculate_log_loss_score(actual_direction, predicted_probs)
    
    # 상승/하락 별 정확도
    up_mask = (actual_direction == 1)
    down_mask = (actual_direction == 0)
    
    up_accuracy = (
        np.mean(predicted_direction[up_mask] == 1) * 100 
        if np.sum(up_mask) > 0 else 0.0
    )
    down_accuracy = (
        np.mean(predicted_direction[down_mask] == 0) * 100 
        if np.sum(down_mask) > 0 else 0.0
    )
    
    return {
        'direction_accuracy': direction_accuracy,
        'brier_score': brier,
        'log_loss': logloss,
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy
    }




# ════════════════════════════════════════════════════════════════════════════
# v2.5.0: 앙상블 모델 구현
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# 1. N-BEATS 모델 (Neural Basis Expansion Analysis for Time Series)
# ────────────────────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """N-BEATS의 기본 블록"""
    def __init__(self, input_size, theta_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        # Backcast and Forecast heads
        self.backcast_fc = nn.Linear(hidden_size, input_size)
        self.forecast_fc = nn.Linear(hidden_size, theta_size)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        
        backcast = self.backcast_fc(h)
        forecast = self.forecast_fc(h)
        
        return backcast, forecast


class NBeatsModel(nn.Module):
    """N-BEATS 전체 모델"""
    def __init__(self, input_size, forecast_size, num_blocks=3, hidden_size=256):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, forecast_size, hidden_size)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        residuals = x
        forecast = torch.zeros(x.size(0), self.blocks[0].forecast_fc.out_features).to(x.device)
        
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        
        return forecast


def train_nbeats(data, forecast_days=3, lookback=180, epochs=50):
    """N-BEATS 모델 학습"""
    if not TORCH_AVAILABLE:
        return None, None
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    # 학습 데이터 생성
    X, y = [], []
    for i in range(lookback, len(scaled_data) - forecast_days):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i:i+forecast_days])
    
    if len(X) < 10:
        return None, scaler
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # 모델 초기화
    model = NBeatsModel(lookback, forecast_days, num_blocks=3, hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model, scaler


def predict_nbeats(model, scaler, last_sequence, forecast_days=3):
    """N-BEATS 예측"""
    if model is None or not TORCH_AVAILABLE:
        return None
    
    with torch.no_grad():
        last_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        X = torch.FloatTensor(last_scaled).unsqueeze(0)
        forecast = model(X).numpy().flatten()
        forecast_original = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
    
    return forecast_original


# ────────────────────────────────────────────────────────────────────────────
# 2. TFT (Temporal Fusion Transformer) - 간소화 버전
# ────────────────────────────────────────────────────────────────────────────

class SimpleTFT(nn.Module):
    """간소화된 TFT 모델 (핵심 어텐션 메커니즘)"""
    def __init__(self, input_size, hidden_size=128, num_heads=4, forecast_size=3):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        embedded = self.embedding(x)
        
        # Self-attention
        attn_out, _ = self.attention(embedded, embedded, embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(attn_out)
        
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        
        # Forecast
        forecast = self.fc(last_output)
        
        return forecast


def train_tft(data, features_df, forecast_days=3, lookback=90, epochs=50):
    """TFT 모델 학습 (다변량)"""
    if not TORCH_AVAILABLE:
        return None, None
    
    # 가격 + 지표 결합
    combined_data = features_df[['Close', 'RSI14', 'MACD', 'Volume']].iloc[-len(data):].values
    
    # 정규화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    # 학습 데이터 생성
    X, y = [], []
    for i in range(lookback, len(scaled_data) - forecast_days):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i:i+forecast_days, 0])  # Close만 예측
    
    if len(X) < 10:
        return None, scaler
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # 모델 초기화
    model = SimpleTFT(input_size=combined_data.shape[1], hidden_size=64, 
                      num_heads=4, forecast_size=forecast_days)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model, scaler


def predict_tft(model, scaler, last_sequence, forecast_days=3):
    """TFT 예측"""
    if model is None or not TORCH_AVAILABLE:
        return None
    
    with torch.no_grad():
        last_scaled = scaler.transform(last_sequence)
        X = torch.FloatTensor(last_scaled).unsqueeze(0)
        forecast = model(X).numpy().flatten()
        
        # 역변환 (Close만)
        dummy = np.zeros((forecast.shape[0], scaler.n_features_in_))
        dummy[:, 0] = forecast
        forecast_original = scaler.inverse_transform(dummy)[:, 0]
    
    return forecast_original


# ────────────────────────────────────────────────────────────────────────────
# 3. XGBoost
# ────────────────────────────────────────────────────────────────────────────

def train_xgboost(data, features_df, forecast_days=3, lookback=60):
    """XGBoost 모델 학습"""
    if not XGBOOST_AVAILABLE:
        return None, None
    
    # 특징 선택 (기술적 지표)
    feature_cols = ['RSI14', 'MACD', 'StochK14', 'MFI14', 'ATR14']
    X_features = features_df[feature_cols].iloc[-len(data):].values
    
    # 시계열 특징 추가 (과거 가격)
    X, y = [], []
    for i in range(lookback, len(data) - forecast_days):
        past_prices = data.iloc[i-lookback:i].values
        past_features = X_features[i-lookback:i].mean(axis=0)  # 평균 지표
        X.append(np.concatenate([past_prices[-10:], past_features]))  # 최근 10개 가격 + 지표
        y.append(data.iloc[i+forecast_days-1])  # forecast_days 후 가격
    
    if len(X) < 10:
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    # XGBoost 학습
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    return model, (lookback, feature_cols)


def predict_xgboost(model, metadata, data, features_df, forecast_days=3):
    """XGBoost 예측"""
    if model is None or not XGBOOST_AVAILABLE:
        return None
    
    lookback, feature_cols = metadata
    X_features = features_df[feature_cols].values
    
    # 마지막 시퀀스
    past_prices = data.iloc[-lookback:].values
    past_features = X_features[-lookback:].mean(axis=0)
    X_pred = np.concatenate([past_prices[-10:], past_features]).reshape(1, -1)
    
    forecast = model.predict(X_pred)
    
    # 반복 예측 (단일 스텝씩)
    forecasts = [forecast[0]]
    for _ in range(1, forecast_days):
        # 이전 예측을 포함하여 다음 예측
        new_prices = np.append(past_prices[1-10:], forecasts[-1])
        X_pred = np.concatenate([new_prices, past_features]).reshape(1, -1)
        forecasts.append(model.predict(X_pred)[0])
    
    return np.array(forecasts)


# ────────────────────────────────────────────────────────────────────────────
# 4. GRU (Gated Recurrent Unit)
# ────────────────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    """GRU 모델"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, forecast_size=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, forecast_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def train_gru(data, forecast_days=3, lookback=120, epochs=50):
    """GRU 모델 학습"""
    if not TORCH_AVAILABLE:
        return None, None
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # 학습 데이터 생성
    X, y = [], []
    for i in range(lookback, len(scaled_data) - forecast_days):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i:i+forecast_days].flatten())
    
    if len(X) < 10:
        return None, scaler
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # 모델 초기화
    model = GRUModel(input_size=1, hidden_size=64, num_layers=2, forecast_size=forecast_days)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model, scaler


def predict_gru(model, scaler, last_sequence, forecast_days=3):
    """GRU 예측"""
    if model is None or not TORCH_AVAILABLE:
        return None
    
    with torch.no_grad():
        last_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        X = torch.FloatTensor(last_scaled).unsqueeze(0)
        forecast = model(X).numpy().flatten()
        forecast_original = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
    
    return forecast_original


# ────────────────────────────────────────────────────────────────────────────
# 5. LightGBM
# ────────────────────────────────────────────────────────────────────────────

def train_lightgbm(data, features_df, forecast_days=3, lookback=60):
    """LightGBM 모델 학습"""
    if not LIGHTGBM_AVAILABLE:
        return None, None
    
    # 특징 선택
    feature_cols = ['RSI14', 'MACD', 'StochK14', 'MFI14', 'ATR14', 'BB_upper', 'BB_lower']
    X_features = features_df[feature_cols].iloc[-len(data):].values
    
    # 학습 데이터 생성
    X, y = [], []
    for i in range(lookback, len(data) - forecast_days):
        past_prices = data.iloc[i-lookback:i].values
        past_features = X_features[i-lookback:i].mean(axis=0)
        X.append(np.concatenate([past_prices[-10:], past_features]))
        y.append(data.iloc[i+forecast_days-1])
    
    if len(X) < 10:
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    # LightGBM 학습
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    
    return model, (lookback, feature_cols)


def predict_lightgbm(model, metadata, data, features_df, forecast_days=3):
    """LightGBM 예측"""
    if model is None or not LIGHTGBM_AVAILABLE:
        return None
    
    lookback, feature_cols = metadata
    X_features = features_df[feature_cols].values
    
    # 마지막 시퀀스
    past_prices = data.iloc[-lookback:].values
    past_features = X_features[-lookback:].mean(axis=0)
    X_pred = np.concatenate([past_prices[-10:], past_features]).reshape(1, -1)
    
    forecasts = [model.predict(X_pred)[0]]
    for _ in range(1, forecast_days):
        new_prices = np.append(past_prices[1-10:], forecasts[-1])
        X_pred = np.concatenate([new_prices, past_features]).reshape(1, -1)
        forecasts.append(model.predict(X_pred)[0])
    
    return np.array(forecasts)


# ────────────────────────────────────────────────────────────────────────────
# 6. Prophet (간단한 래퍼)
# ────────────────────────────────────────────────────────────────────────────

def train_prophet(data, forecast_days=3):
    """Prophet 모델 학습"""
    if not PROPHET_AVAILABLE:
        return None
    
    # Prophet 형식으로 변환
    df_prophet = pd.DataFrame({
        'ds': data.index,
        'y': data.values
    })
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df_prophet)
    
    return model


def predict_prophet(model, forecast_days=3):
    """Prophet 예측"""
    if model is None or not PROPHET_AVAILABLE:
        return None
    
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    return forecast['yhat'].iloc[-forecast_days:].values


# ────────────────────────────────────────────────────────────────────────────
# 7. LSTM (기존보다 강화된 버전)
# ────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """LSTM 모델"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, forecast_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, forecast_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(data, forecast_days=3, lookback=120, epochs=50):
    """LSTM 모델 학습"""
    if not TORCH_AVAILABLE:
        return None, None
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # 학습 데이터 생성
    X, y = [], []
    for i in range(lookback, len(scaled_data) - forecast_days):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i:i+forecast_days].flatten())
    
    if len(X) < 10:
        return None, scaler
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # 모델 초기화
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, forecast_size=forecast_days)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model, scaler


def predict_lstm(model, scaler, last_sequence, forecast_days=3):
    """LSTM 예측"""
    if model is None or not TORCH_AVAILABLE:
        return None
    
    with torch.no_grad():
        last_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        X = torch.FloatTensor(last_scaled).unsqueeze(0)
        forecast = model(X).numpy().flatten()
        forecast_original = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
    
    return forecast_original


# ────────────────────────────────────────────────────────────────────────────
# 8. 앙상블 조합 및 자동 선택
# ────────────────────────────────────────────────────────────────────────────

def get_ensemble_config(interval):
    """
    시간 프레임에 따른 앙상블 모델 자동 선택
    
    Parameters:
    -----------
    interval : str
        시간 프레임 ('1m', '5m', '1h', '1d')
    
    Returns:
    --------
    dict : 앙상블 설정
        - 'models': 사용할 모델 리스트
        - 'weights': 각 모델의 가중치
        - 'lookback': 학습 윈도우 크기
        - 'epochs': 학습 에폭 수
    """
    if interval in ['1m', '5m']:
        # 초단타 트레이딩: N-BEATS + TFT + XGBoost
        return {
            'models': ['nbeats', 'tft', 'xgboost'],
            'weights': [0.40, 0.35, 0.25],
            'lookback': {'nbeats': 180, 'tft': 90, 'xgboost': 60},
            'epochs': 30,
            'description': '초단타 트레이딩 (N-BEATS 40% + TFT 35% + XGBoost 25%)'
        }
    elif interval == '1h':
        # 단기 트레이딩 상단: N-BEATS + TFT + XGBoost (시간봉도 빠른 편)
        return {
            'models': ['nbeats', 'tft', 'xgboost'],
            'weights': [0.40, 0.35, 0.25],
            'lookback': {'nbeats': 240, 'tft': 120, 'xgboost': 90},
            'epochs': 40,
            'description': '시간봉 트레이딩 (N-BEATS 40% + TFT 35% + XGBoost 25%)'
        }
    elif interval == '1d':
        # 단기 트레이딩: LightGBM + GRU + Prophet
        return {
            'models': ['gru', 'lightgbm', 'prophet'],
            'weights': [0.40, 0.35, 0.25],
            'lookback': {'gru': 120, 'lightgbm': 60, 'prophet': None},
            'epochs': 50,
            'description': '일봉 트레이딩 (GRU 40% + LightGBM 35% + Prophet 25%)'
        }
    else:
        # 중기 트레이딩 (주봉 이상): XGBoost + LSTM + Holt-Winters
        return {
            'models': ['lstm', 'xgboost', 'holtwinters'],
            'weights': [0.45, 0.30, 0.25],
            'lookback': {'lstm': 150, 'xgboost': 90, 'holtwinters': None},
            'epochs': 50,
            'description': '중기 트레이딩 (LSTM 45% + XGBoost 30% + Holt-Winters 25%)'
        }


def train_ensemble_models(data, features_df, interval, forecast_days=3):
    """
    앙상블 모델 학습
    
    Parameters:
    -----------
    data : pd.Series
        가격 데이터
    features_df : pd.DataFrame
        기술적 지표 데이터
    interval : str
        시간 프레임
    forecast_days : int
        예측 일수
    
    Returns:
    --------
    dict : 학습된 모델들과 메타데이터
    """
    config = get_ensemble_config(interval)
    models = {}
    
    st.info(f"🤖 앙상블 모델 선택: {config['description']}")
    
    progress_bar = st.progress(0)
    total_models = len(config['models'])
    
    for idx, model_name in enumerate(config['models']):
        try:
            lookback = config['lookback'].get(model_name, 90)
            epochs = config['epochs']
            
            st.text(f"학습 중: {model_name.upper()} ({idx+1}/{total_models})")
            
            if model_name == 'nbeats':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_nbeats(data, forecast_days, lookback, epochs)
                    models['nbeats'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'tft':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_tft(data, features_df, forecast_days, lookback, epochs)
                    models['tft'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: XGBoost 미설치")
                    models[model_name] = None
                else:
                    model, metadata = train_xgboost(data, features_df, forecast_days, lookback)
                    models['xgboost'] = {'model': model, 'metadata': metadata}
            
            elif model_name == 'gru':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_gru(data, forecast_days, lookback, epochs)
                    models['gru'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 부0가: LightGBM 미설치")
                    models[model_name] = None
                else:
                    model, metadata = train_lightgbm(data, features_df, forecast_days, lookback)
                    models['lightgbm'] = {'model': model, 'metadata': metadata}
            
            elif model_name == 'prophet':
                if not PROPHET_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 부0가: Prophet 미설치")
                    models[model_name] = None
                else:
                    model = train_prophet(data, forecast_days)
                    models['prophet'] = {'model': model}
            
            elif model_name == 'lstm':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 부0가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_lstm(data, forecast_days, lookback, epochs)
                    models['lstm'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'holtwinters':
                # Holt-Winters는 기존 함수 재사용
                hw_model, seasonality_info, window_size = fit_hw_model_robust(data, max_window=500)
                models['holtwinters'] = {'model': hw_model, 'seasonality': seasonality_info}
            
            progress_bar.progress((idx + 1) / total_models)
        
        except Exception as e:
            st.warning(f"⚠️ {model_name} 학습 실패: {e}")
            models[model_name] = None
    
    progress_bar.empty()
    
    return models, config


def predict_ensemble(models, config, data, features_df, forecast_days=3):
    """
    앙상블 예측
    
    Parameters:
    -----------
    models : dict
        학습된 모델들
    config : dict
        앙상블 설정
    data : pd.Series
        가격 데이터
    features_df : pd.DataFrame
        기술적 지표
    forecast_days : int
        예측 일수
    
    Returns:
    --------
    np.array : 앙상블 예측값
    dict : 각 모델별 예측값
    """
    predictions = {}
    weights = {}
    
    for model_name, weight in zip(config['models'], config['weights']):
        if models.get(model_name) is None:
            continue
        
        try:
            model_data = models[model_name]
            lookback = config['lookback'].get(model_name, 90)
            
            if model_name == 'nbeats':
                pred = predict_nbeats(
                    model_data['model'], 
                    model_data['scaler'], 
                    data.iloc[-lookback:].values, 
                    forecast_days
                )
            
            elif model_name == 'tft':
                last_features = features_df[['Close', 'RSI14', 'MACD', 'Volume']].iloc[-lookback:].values
                pred = predict_tft(
                    model_data['model'], 
                    model_data['scaler'], 
                    last_features, 
                    forecast_days
                )
            
            elif model_name == 'xgboost':
                pred = predict_xgboost(
                    model_data['model'], 
                    model_data['metadata'], 
                    data, 
                    features_df, 
                    forecast_days
                )
            
            elif model_name == 'gru':
                pred = predict_gru(
                    model_data['model'], 
                    model_data['scaler'], 
                    data.iloc[-lookback:].values, 
                    forecast_days
                )
            
            elif model_name == 'lightgbm':
                pred = predict_lightgbm(
                    model_data['model'], 
                    model_data['metadata'], 
                    data, 
                    features_df, 
                    forecast_days
                )
            
            elif model_name == 'prophet':
                pred = predict_prophet(model_data['model'], forecast_days)
            
            elif model_name == 'lstm':
                pred = predict_lstm(
                    model_data['model'], 
                    model_data['scaler'], 
                    data.iloc[-lookback:].values, 
                    forecast_days
                )
            
            elif model_name == 'holtwinters':
                hw_forecast = model_data['model'].forecast(steps=forecast_days)
                pred = hw_forecast.values
            
            if pred is not None and len(pred) == forecast_days:
                predictions[model_name] = pred
                weights[model_name] = weight
        
        except Exception as e:
            st.warning(f"⚠️ {model_name} 예측 실패: {e}")
    
    # 가중 평균
    if not predictions:
        return None, {}
    
    # 가중치 정규화
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # 앙상블 예측
    ensemble_forecast = np.zeros(forecast_days)
    for model_name, pred in predictions.items():
        ensemble_forecast += pred * normalized_weights[model_name]
    
    return ensemble_forecast, predictions


def detect_seasonality_auto(series: pd.Series, max_period: int = 30) -> tuple:
    """
    자동 계절성 감지 (v2.4.0)
    
    Returns:
        tuple: (has_seasonality: bool, optimal_period: int, seasonality_type: str)
    """
    from scipy import stats
    
    if len(series) < 20:
        return False, None, None
    
    # 1. ACF 기반 계절성 감지
    try:
        from statsmodels.tsa.stattools import acf
        acf_values = acf(series, nlags=min(max_period, len(series) // 2), fft=True)
        
        # ACF 피크 찾기 (첫 번째 지연 제외)
        peaks = []
        for i in range(2, len(acf_values)):
            if acf_values[i] > 0.3:  # 임계값
                peaks.append((i, acf_values[i]))
        
        if peaks:
            # 가장 강한 피크 선택
            optimal_period = max(peaks, key=lambda x: x[1])[0]
            has_seasonality = True
        else:
            has_seasonality = False
            optimal_period = None
    except:
        has_seasonality = False
        optimal_period = None
    
    # 2. 변동성 기반 계절성 타입 결정
    if has_seasonality:
        volatility = series.pct_change().std()
        if volatility > 0.05:
            seasonality_type = 'mul'  # 변동성 높으면 multiplicative
        else:
            seasonality_type = 'add'  # 변동성 낮으면 additive
    else:
        seasonality_type = None
    
    return has_seasonality, optimal_period, seasonality_type


def fit_hw_model_robust(series: pd.Series, max_window: int = 500) -> tuple:
    """
    강건한 Holt-Winters 모델 학습 (v2.4.0)
    
    Features:
    - 자동 계절성 감지
    - 최신 윈도우 제한 (성능 개선)
    - 폴백 전략
    
    Returns:
        tuple: (model, seasonality_info: dict, training_window_size: int)
    """
    import statsmodels.api as sm
    
    # 최신 데이터만 사용 (성능 개선)
    if len(series) > max_window:
        series_windowed = series.iloc[-max_window:]
        original_index = series.index
        window_used = max_window
    else:
        series_windowed = series
        original_index = series.index
        window_used = len(series)
    
    # 계절성 자동 감지
    has_seasonality, optimal_period, seasonality_type = detect_seasonality_auto(
        series_windowed, 
        max_period=min(30, len(series_windowed) // 3)
    )
    
    seasonality_info = {
        'detected': has_seasonality,
        'period': optimal_period,
        'type': seasonality_type,
        'window_size': window_used
    }
    
    # 모델 학습 (계층적 폴백)
    model = None
    error_log = []
    
    # Try 1: 감지된 계절성 사용
    if has_seasonality and optimal_period and optimal_period >= 2:
        try:
            model = sm.tsa.ExponentialSmoothing(
                series_windowed,
                trend='add',
                seasonal=seasonality_type,
                seasonal_periods=optimal_period,
                initialization_method="estimated"
            ).fit(optimized=True)
            seasonality_info['model_type'] = f'{seasonality_type}_seasonal'
            return model, seasonality_info, window_used
        except Exception as e:
            error_log.append(f"Seasonal model failed: {str(e)[:50]}")
    
    # Try 2: 단순 계절성 (period=7)
    if len(series_windowed) >= 14:
        try:
            model = sm.tsa.ExponentialSmoothing(
                series_windowed,
                trend='add',
                seasonal='add',
                seasonal_periods=7,
                initialization_method="estimated"
            ).fit(optimized=True)
            seasonality_info['model_type'] = 'default_seasonal'
            seasonality_info['period'] = 7
            return model, seasonality_info, window_used
        except Exception as e:
            error_log.append(f"Default seasonal failed: {str(e)[:50]}")
    
    # Try 3: 비계절 모델 (폴백)
    try:
        model = sm.tsa.ExponentialSmoothing(
            series_windowed,
            trend='add',
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        seasonality_info['model_type'] = 'non_seasonal'
        seasonality_info['detected'] = False
        return model, seasonality_info, window_used
    except Exception as e:
        error_log.append(f"Non-seasonal failed: {str(e)[:50]}")
    
    # 모든 시도 실패
    raise ValueError(f"All model fitting attempts failed: {'; '.join(error_log)}")


def forecast_with_offset_scaling(model, steps: int, last_actual_value: float, 
                                  recent_trend: float) -> np.ndarray:
    """
    오프셋 스케일링 예측 (v2.4.0)
    
    절대가격 대신 차분(difference) 기반으로 예측하여 하락 추세 보정
    
    Args:
        model: 학습된 HW 모델
        steps: 예측 스텝 수
        last_actual_value: 마지막 실제 가격
        recent_trend: 최근 추세 (이동평균 기울기)
    
    Returns:
        np.ndarray: 보정된 예측값
    """
    # 모델 예측 (차분 공간)
    raw_forecast = model.forecast(steps=steps)
    
    # 오프셋 보정
    # 1. 첫 예측값과 실제값의 차이 계산
    offset = last_actual_value - raw_forecast.iloc[0]
    
    # 2. 추세 보정 (최근 추세 반영)
    trend_correction = np.linspace(0, recent_trend * steps, steps)
    
    # 3. 최종 예측값 = 원본 예측 + 오프셋 + 추세 보정
    corrected_forecast = raw_forecast + offset + trend_correction
    
    return corrected_forecast




def perform_timeseries_cv(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """TimeSeriesSplit 검증"""
    if len(df) < n_splits * 10:
        return pd.DataFrame({
            'Fold': [1],
            'Accuracy': ['N/A (데이터 부족)'],
            'MASE': ['N/A'],
            'Mean_Error': ['N/A'],
            'Train_Size': [len(df)],
            'Test_Size': [0]
        })
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    close_values = df['Close'].values
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(close_values), 1):
        train_data = close_values[train_idx]
        test_data = close_values[test_idx]
        
        if len(train_data) < 20:
            results.append({
                'Fold': fold,
                'Accuracy': 'N/A',
                'MASE': 'N/A',
                'Mean_Error': 'N/A',
                'Train_Size': len(train_data),
                'Test_Size': len(test_data)
            })
            continue
        
        try:
            seasonal_periods = min(7, len(train_data) // 3)
            
            if seasonal_periods >= 2:
                hw_model = sm.tsa.ExponentialSmoothing(
                    train_data,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated"
                ).fit(optimized=True)
            else:
                hw_model = sm.tsa.ExponentialSmoothing(
                    train_data,
                    trend='add',
                    seasonal=None,
                    initialization_method="estimated"
                ).fit(optimized=True)
            
            forecast = hw_model.forecast(steps=len(test_data))
            
            if len(test_data) > 1:
                actual_direction = np.sign(np.diff(test_data))
                pred_direction = np.sign(np.diff(forecast))
                accuracy = (actual_direction == pred_direction).mean() * 100
            else:
                accuracy = 0.0
            
            mase = calculate_mase(test_data[1:], forecast[1:], train_data)
            mean_error = np.abs(test_data - forecast).mean()
            
            results.append({
                'Fold': fold,
                'Accuracy': f"{accuracy:.2f}%",
                'MASE': f"{mase:.4f}",
                'Mean_Error': f"${mean_error:.2f}",
                'Train_Size': len(train_data),
                'Test_Size': len(test_data)
            })
        except Exception as e:
            results.append({
                'Fold': fold,
                'Accuracy': 'N/A',
                'MASE': 'N/A',
                'Mean_Error': 'N/A',
                'Train_Size': len(train_data),
                'Test_Size': len(test_data)
            })
    
    return pd.DataFrame(results)


def calculate_mase(actual, forecast, train_data):
    """MASE 계산"""
    try:
        mae = np.mean(np.abs(actual - forecast))
        naive_mae = np.mean(np.abs(np.diff(train_data)))
        if naive_mae == 0:
            return 999.0
        mase = mae / naive_mae
        return mase
    except:
        return 999.0


def calculate_rr_ratio(entry_price: float, take_profit: float, stop_loss: float) -> float:
    """Risk-Reward Ratio 계산"""
    reward = abs(take_profit - entry_price)
    risk = abs(entry_price - stop_loss)
    
    if risk == 0:
        return 999.0
    
    return reward / risk


def render_progress_bar(step: int, total: int = 6):
    """진행 상태"""
    steps = ['데이터 로드', '지표 계산', 'AI 학습', '패턴 분석', '검증', '결과 생성']
    progress_html = '<div style="margin: 20px 0;">'
    for i, step_name in enumerate(steps[:total], 1):
        if i <= step:
            progress_html += f'<span class="progress-step active">{i}. {step_name}</span>'
        else:
            progress_html += f'<span class="progress-step">{i}. {step_name}</span>'
    progress_html += '</div>'
    return progress_html


def render_data_summary(df: pd.DataFrame, selected_crypto: str, interval_name: str):
    """데이터 요약"""
    st.markdown("<div class='section-title'>📊 데이터 개요</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    daily_change = df['일일수익률'].iloc[-1] * 100
    avg_volume = df['Volume'].mean()
    total_periods = len(df)
    
    with col1:
        st.metric(
            label=f"현재가 (USD)",
            value=f"${current_price:,.2f}",
            delta=f"{daily_change:+.2f}%"
        )
    
    with col2:
        period_text = f"{total_periods} 기간"
        st.metric(
            label=f"분석 기간 ({interval_name})",
            value=period_text
        )
    
    with col3:
        st.metric(
            label="평균 거래량",
            value=f"{avg_volume:,.0f}"
        )
    
    with col4:
        volatility = df['Volatility30d'].iloc[-1] * 100
        st.metric(
            label="변동성 (30기간)",
            value=f"{volatility:.2f}%"
        )


def render_ai_forecast(future_df: pd.DataFrame, hw_confidence: float):
    """AI 예측"""
    st.markdown("<div class='section-title'>🤖 AI 예측 (Holt-Winters Seasonal)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df['예측 종가'],
            mode='lines+markers',
            name='예측 종가',
            line=dict(color='#667EEA', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="향후 30일 예측",
            xaxis_title="날짜",
            yaxis_title="예측 가격 (USD)",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 예측 요약")
        st.metric(
            label="30일 후 예상가",
            value=f"${future_df['예측 종가'].iloc[-1]:,.2f}",
            delta=f"{((future_df['예측 종가'].iloc[-1] / future_df['예측 종가'].iloc[0]) - 1) * 100:+.2f}%"
        )
        
        st.metric(
            label="모델 신뢰도",
            value=f"{hw_confidence:.1f}%"
        )
        
        predicted_change = ((future_df['예측 종가'].iloc[-1] / future_df['예측 종가'].iloc[0]) - 1) * 100
        
        if predicted_change > 5:
            st.success("🚀 강한 상승 예상")
        elif predicted_change > 0:
            st.info("📈 소폭 상승 예상")
        elif predicted_change > -5:
            st.warning("📉 소폭 하락 예상")
        else:
            st.error("⚠️ 강한 하락 예상")


def render_patterns(patterns: list):
    """패턴 분석 (개선된 레이아웃)"""
    st.markdown("<div class='section-title'>🕯️ 캔들스틱 패턴</div>", unsafe_allow_html=True)
    
    if not patterns:
        st.info("최근 주요 패턴이 감지되지 않았습니다.")
        return
    
    # 패턴 카테고리별 분류
    categories = {}
    for pattern in patterns:
        cat = pattern.get('category', '기타')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(pattern)
    
    # 카테고리별 통계
    st.markdown(f"**총 {len(patterns)}개 패턴 감지** | 카테고리: {', '.join([f'{k}({len(v)})' for k, v in categories.items()])}")
    
    for pattern in patterns:
        with st.container():
            date_str = pattern['date'].strftime('%Y-%m-%d %H:%M') if hasattr(pattern['date'], 'strftime') else str(pattern['date'])
            
            st.markdown(f"""
                <div class='pattern-card'>
                    <div class='pattern-title'>{pattern['name']} [{pattern.get('category', '기타')}]</div>
                    <table style='width: 100%; color: white; border-collapse: collapse;'>
                        <tr>
                            <td style='width: 50%; padding: 8px 0;'>📅 발생일: {date_str}</td>
                            <td style='width: 50%; padding: 8px 0;'>📝 설명: {pattern['desc']}</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px 0;'>🎯 신뢰도: {pattern['conf']}%</td>
                            <td style='padding: 8px 0;'>💡 영향: {pattern['impact']}</td>
                        </tr>
                    </table>
                </div>
            """, unsafe_allow_html=True)


def render_exit_strategy(exit_strategy: dict, entry_price: float, investment_amount: float, leverage: float):
    """매도 전략 (신규)"""
    st.markdown("<div class='section-title'>💰 매도 시점 예측 (언제 팔아야 하는가?)</div>", unsafe_allow_html=True)
    
    current_status = exit_strategy['current_status']
    scenarios = exit_strategy['scenarios']
    
    # 현재 상태
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="진입가",
            value=f"${entry_price:,.2f}"
        )
    
    with col2:
        st.metric(
            label="현재가",
            value=f"${current_status['current_price']:,.2f}",
            delta=f"{current_status['unrealized_pnl']:+.2f}%"
        )
    
    with col3:
        rsi_color = "🔴" if current_status['rsi_status'] == 'overbought' else "🟢" if current_status['rsi_status'] == 'oversold' else "⚪"
        st.metric(
            label="RSI 상태",
            value=f"{rsi_color} {current_status['rsi_status'].upper()}"
        )
    
    with col4:
        trend_color = "📈" if current_status['trend'] == 'bullish' else "📉"
        st.metric(
            label="추세",
            value=f"{trend_color} {current_status['trend'].upper()}"
        )
    
    # 권장사항
    if current_status['recommendation']:
        if '즉시' in current_status['recommendation']:
            st.error(current_status['recommendation'])
        elif '고려' in current_status['recommendation']:
            st.warning(current_status['recommendation'])
        else:
            st.info(current_status['recommendation'])
    
    st.markdown("---")
    
    # 3가지 시나리오
    st.markdown("### 🎯 매도 시나리오")
    
    for scenario_key, scenario in scenarios.items():
        with st.container():
            profit_pct = ((scenario['take_profit'] - entry_price) / entry_price) * 100
            loss_pct = ((entry_price - scenario['stop_loss']) / entry_price) * 100
            
            profit_amount = investment_amount * leverage * (profit_pct / 100)
            loss_amount = investment_amount * leverage * (loss_pct / 100)
            
            st.markdown(f"""
                <div class='exit-card'>
                    <div class='exit-title'>{scenario['name']}</div>
                    <table style='width: 100%; color: white; border-collapse: collapse;'>
                        <tr>
                            <td style='width: 33%; padding: 8px 0;'>🎯 익절가: ${scenario['take_profit']:,.2f} (+{profit_pct:.2f}%)</td>
                            <td style='width: 33%; padding: 8px 0;'>🛑 손절가: ${scenario['stop_loss']:,.2f} (-{loss_pct:.2f}%)</td>
                            <td style='width: 34%; padding: 8px 0;'>⏱️ 보유기간: {scenario['holding_period']}</td>
                        </tr>
                        <tr>
                            <td style='padding: 8px 0;'>💵 목표 수익: ${profit_amount:,.2f}</td>
                            <td style='padding: 8px 0;'>💸 최대 손실: ${loss_amount:,.2f}</td>
                            <td style='padding: 8px 0;'>📊 RR Ratio: {scenario['rr_ratio']:.2f}</td>
                        </tr>
                        <tr>
                            <td colspan='3' style='padding: 8px 0;'>📝 {scenario['description']}</td>
                        </tr>
                    </table>
                    <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.3);'>
                        <strong>매도 신호:</strong><br/>
                        {'<br/>'.join(['• ' + signal for signal in scenario['exit_signals']])}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_validation_results(cv_results: pd.DataFrame):
    """모델 검증"""
    st.markdown("<div class='section-title'>✅ 모델 검증 (TimeSeriesSplit)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.dataframe(
            cv_results,
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("### 📊 검증 지표 설명")
        st.markdown("""
        - **Accuracy**: 방향성 예측 정확도
        - **MASE**: 예측 오차 (1.0 미만이 우수)
        - **Mean_Error**: 평균 절대 오차
        - **Train/Test Size**: 학습/테스트 데이터 크기
        """)
        
        try:
            accuracies = []
            for acc in cv_results['Accuracy']:
                if isinstance(acc, str) and '%' in acc:
                    accuracies.append(float(acc.replace('%', '')))
            
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                st.metric(
                    label="평균 방향성 정확도",
                    value=f"{avg_accuracy:.2f}%"
                )
        except:
            pass



# ────────────────────────────────────────────────────────────────────────
# [추가됨] v2.3.1: 개별 차트 생성 함수
# ────────────────────────────────────────────────────────────────────────

def create_candlestick_chart(df: pd.DataFrame, symbol: str):
    """캔들스틱 차트 생성"""
    fig = go.Figure()
    
    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='가격',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
    )
    
    # EMA50
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA50'],
            name='EMA50',
            line=dict(color='orange', width=2)
        )
    )
    
    # EMA200
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA200'],
            name='EMA200',
            line=dict(color='purple', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} 가격 차트',
        xaxis_title='날짜',
        yaxis_title='가격 (USD)',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_volume_chart(df: pd.DataFrame):
    """거래량 차트 생성"""
    fig = go.Figure()
    
    # 거래량 막대 (상승/하락에 따라 색상 구분)
    colors = ['#26a69a' if close >= open_ else '#ef5350' 
              for close, open_ in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='거래량',
            marker_color=colors
        )
    )
    
    # 거래량 이동평균 (20일)
    volume_ma20 = df['Volume'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=volume_ma20,
            name='거래량 MA20',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.update_layout(
        title='거래량 차트',
        xaxis_title='날짜',
        yaxis_title='거래량',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    return fig


def create_rsi_chart(df: pd.DataFrame):
    """RSI 차트 생성"""
    fig = go.Figure()
    
    # RSI 라인
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI14'],
            name='RSI (14)',
            line=dict(color='blue', width=2)
        )
    )
    
    # 과매수/과매도 라인
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="과매수 (70)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="과매도 (30)")
    
    # 배경 색상 영역
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title='RSI (Relative Strength Index)',
        xaxis_title='날짜',
        yaxis_title='RSI',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_macd_chart(df: pd.DataFrame):
    """MACD 차트 생성"""
    fig = go.Figure()
    
    # MACD 라인
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            line=dict(color='blue', width=2)
        )
    )
    
    # Signal 라인
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            name='Signal',
            line=dict(color='red', width=2)
        )
    )
    
    # Histogram
    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['MACD_Hist'],
            name='Histogram',
            marker_color=colors
        )
    )
    
    # 0선
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        xaxis_title='날짜',
        yaxis_title='MACD',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    return fig


def render_trading_strategy(current_price: float, leverage_info: dict, entry_price: float,
                           stop_loss: float, take_profit: float, position_size: float,
                           rr_ratio: float, investment_amount: float):
    """매매 전략 (v2.3.0: 레버리지 표시 개선)"""
    st.markdown("<div class='section-title'>🎯 매매 전략</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📍 진입 설정")
        # [수정됨] v2.3.0: 권장/최대 레버리지 분리 표시
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
            <p style='margin: 0; font-size: 14px; color: #666;'>⚙️ 레버리지 최적화</p>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <div>
                    <p style='margin: 0; font-size: 12px; color: #888;'>권장 레버리지</p>
                    <p style='margin: 0; font-size: 24px; font-weight: bold; color: #1f77b4;'>{leverage_info['recommended']}배</p>
                </div>
                <div>
                    <p style='margin: 0; font-size: 12px; color: #888;'>최대 레버리지</p>
                    <p style='margin: 0; font-size: 24px; font-weight: bold; color: #ff7f0e;'>{leverage_info['maximum']}배</p>
                </div>
            </div>
            <p style='margin: 5px 0 0 0; font-size: 11px; color: #888; text-align: center;'>
                리스크 레벨: <strong>{leverage_info['risk_level']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.metric(label="진입가", value=f"${entry_price:,.2f}")
        st.metric(label="포지션 크기", value=f"{position_size:.4f} 코인")
    
    with col2:
        st.markdown("### 🛑 리스크 관리")
        st.metric(label="손절가", value=f"${stop_loss:,.2f}")
        st.metric(label="목표가", value=f"${take_profit:,.2f}")
        st.metric(label="RR Ratio", value=f"{rr_ratio:.2f}")
    
    with col3:
        st.markdown("### 💰 예상 손익")
        expected_profit = position_size * (take_profit - entry_price)
        expected_loss = position_size * (entry_price - stop_loss)
        
        st.metric(
            label="목표 수익",
            value=f"${expected_profit:,.2f}",
            delta=f"{(expected_profit / investment_amount) * 100:.2f}%"
        )
        st.metric(
            label="최대 손실",
            value=f"-${expected_loss:,.2f}",
            delta=f"-{(expected_loss / investment_amount) * 100:.2f}%"
        )
    
    if rr_ratio >= 3:
        st.success(f"✅ 우수한 RR Ratio ({rr_ratio:.2f}) - 리스크 대비 높은 수익 가능")
    elif rr_ratio >= 2:
        st.info(f"📊 적정한 RR Ratio ({rr_ratio:.2f}) - 균형잡힌 전략")
    else:
        st.warning(f"⚠️ 낮은 RR Ratio ({rr_ratio:.2f}) - 리스크 대비 수익이 작음")


def render_technical_indicators(df: pd.DataFrame):
    """기술적 지표"""
    st.markdown("<div class='section-title'>📊 기술적 지표</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi = df['RSI14'].iloc[-1]
        rsi_signal = "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
        st.metric(label="RSI (14)", value=f"{rsi:.2f}", delta=rsi_signal)
    
    with col2:
        stoch = df['StochK14'].iloc[-1]
        stoch_signal = "과매수" if stoch > 80 else "과매도" if stoch < 20 else "중립"
        st.metric(label="Stochastic (14)", value=f"{stoch:.2f}", delta=stoch_signal)
    
    with col3:
        mfi = df['MFI14'].iloc[-1]
        mfi_signal = "과매수" if mfi > 80 else "과매도" if mfi < 20 else "중립"
        st.metric(label="MFI (14)", value=f"{mfi:.2f}", delta=mfi_signal)
    
    with col4:
        macd_hist = df['MACD_Hist'].iloc[-1]
        macd_signal = "상승" if macd_hist > 0 else "하락"
        st.metric(label="MACD Histogram", value=f"{macd_hist:.2f}", delta=macd_signal)


# ────────────────────────────────────────────────────────────────────────
# 메인 UI
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🚀 설정")
    st.markdown("---")
    
    # TA-Lib 상태 표시
    if TALIB_AVAILABLE:
        st.success("✅ TA-Lib 사용 가능 (61개 패턴)")
    else:
        st.warning("⚠️ TA-Lib 미설치 (기본 3개 패턴)")
    
    st.markdown("## 1️⃣ 분해능 선택")
    resolution_choice = st.selectbox(
        "📈 시간 프레임",
        list(RESOLUTION_MAP.keys()),
        index=3,
        help="짧은 기간일수록 최신 데이터만 제공됩니다"
    )
    interval = RESOLUTION_MAP[resolution_choice]
    interval_name = resolution_choice
    
    interval_info = {
        '1m': '⏱️ 1분봉: 최근 **7일**만 지원 (초단타 매매용)',
        '5m': '⏱️ 5분봉: 최근 **60일**만 지원 (단타 매매용)',
        '1h': '⏱️ 1시간봉: 최근 **2년**만 지원 (스윙 트레이딩용)',
        '1d': '⏱️ 1일봉: **전체 기간** 지원 (중장기 투자용)'
    }
    
    st.info(interval_info.get(interval, ''))
    
    st.markdown("---")
    st.markdown("## 2️⃣ 코인 선택")
    
    coin_input_method = st.radio(
        "🔧 입력 방식",
        ["목록에서 선택", "직접 입력"],
        horizontal=True
    )
    
    if coin_input_method == "목록에서 선택":
        crypto_choice = st.selectbox(
            "💎 암호화폐",
            list(CRYPTO_MAP.keys())
        )
        selected_crypto = CRYPTO_MAP[crypto_choice]
    else:
        custom_symbol = st.text_input(
            "💎 코인 심볼 입력",
            value="BTCUSDT",
            help="예: BTCUSDT, ETHUSDT, BNBUSDT 등 (USDT 페어만 지원)"
        ).upper().strip()
        
        if not custom_symbol.endswith("USDT"):
            st.warning("⚠️ USDT 페어만 지원됩니다. 심볼 끝에 'USDT'를 추가해주세요.")
            custom_symbol = custom_symbol + "USDT" if custom_symbol else "BTCUSDT"
        
        selected_crypto = custom_symbol
        st.info(f"선택된 코인: **{selected_crypto}** ({selected_crypto[:-4]}-USD)")
    
    st.markdown("---")
    st.markdown("## 3️⃣ 분석 기간")
    
    period_choice = st.radio(
        "📅 기간 설정",
        ["자동 (분해능에 최적화)", "수동 설정"],
        help="자동 모드는 분해능별 제한을 자동으로 적용합니다"
    )
    
    if period_choice == "자동 (분해능에 최적화)":
        today = datetime.date.today()
        
        interval_periods = {
            '1m': 7,
            '5m': 60,
            '1h': 730,
            '1d': 365 * 5
        }
        
        days_back = interval_periods.get(interval, 180)
        START = today - datetime.timedelta(days=days_back)
        
        listing_dates = {
            "BTCUSDT": datetime.date(2017, 8, 17),
            "ETHUSDT": datetime.date(2017, 8, 17),
            "XRPUSDT": datetime.date(2018, 5, 14),
            "DOGEUSDT": datetime.date(2021, 5, 6),
            "ADAUSDT": datetime.date(2018, 4, 17),
            "SOLUSDT": datetime.date(2021, 8, 11)
        }
        
        listing_date = listing_dates.get(selected_crypto, START)
        
        if START < listing_date:
            START = listing_date
        
        END = today
        st.info(f"📅 분석 기간: {START} ~ {END} ({(END - START).days}일)")
    else:
        col_s, col_e = st.columns(2)
        with col_s:
            START = st.date_input(
                "시작일",
                value=datetime.date.today() - datetime.timedelta(days=180)
            )
        with col_e:
            END = st.date_input(
                "종료일",
                value=datetime.date.today()
            )
        
        if START >= END:
            st.error("시작일은 종료일 이전이어야 합니다.")
            st.stop()
    
    st.markdown("---")
    st.markdown("## 4️⃣ 투자 설정")
    
    
    # 예측 일수 설정
    forecast_days = st.slider(
        "🔮 예측 기간",
        min_value=1,
        max_value=30,
        value=3,
        step=1,
        help="몇 일 후의 가격을 예측할지 선택하세요"
    )
    st.session_state['forecast_days'] = forecast_days

    investment_amount = st.number_input(
        "💰 투자 금액 (USDT)",
        min_value=1.0,
        value=1000.0,
        step=50.0
    )
    
    risk_per_trade_pct = st.slider(
        "⚠️ 리스크 비율 (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="한 거래당 최대 손실 허용 퍼센트"
    ) / 100.0
    
    stop_loss_k = st.number_input(
        "🛑 손절 배수 (σ 기준)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.5
    )
    
    default_max_lev = MAX_LEVERAGE_MAP.get(selected_crypto, 50)
    leverage_ceiling = st.number_input(
        "📊 최대 레버리지",
        min_value=1,
        max_value=500,
        value=int(default_max_lev),
        step=1
    )
    
    st.markdown("---")
    bt = st.button("🚀 분석 시작", type="primary", use_container_width=True)

# 메인 로직
if bt:
    try:
        progress_placeholder = st.empty()
        status_text = st.empty()
        
        progress_placeholder.markdown(render_progress_bar(1, 6), unsafe_allow_html=True)
        status_text.info(f"🔍 데이터를 가져오는 중... (분해능: {interval_name})")
        
        raw_df = load_crypto_data(selected_crypto, START, END, interval)
        
        if raw_df.empty:
            yf_ticker = selected_crypto[:-4] + "-USD"
            st.error(f"❌ {yf_ticker} 데이터를 불러올 수 없습니다.")
            st.warning(f"""
            **가능한 원인**:
            - 선택한 기간({START} ~ {END})에 데이터가 없습니다
            - 분해능({interval_name})이 해당 기간을 지원하지 않습니다
            - yfinance API 일시적 오류
            
            **해결 방법**:
            1. 더 최근 기간 선택
            2. 분해능을 1일봉으로 변경
            3. 다른 코인 선택
            4. 잠시 후 다시 시도
            """)
            
            if st.button("🔄 캐시 초기화 후 재시도"):
                st.cache_data.clear()
                st.rerun()
            st.stop()
        
        min_required = 20
        if len(raw_df) < min_required:
            st.error(f"❌ 최소 {min_required} 기간 이상의 데이터가 필요합니다. (현재: {len(raw_df)})")
            st.warning("""
            **해결 방법**:
            1. 더 긴 기간 선택
            2. 다른 분해능 선택 (1일봉 권장)
            3. 다른 코인 선택
            """)
            st.stop()
        
        progress_placeholder.markdown(render_progress_bar(2, 6), unsafe_allow_html=True)
        status_text.info("📊 적응형 지표를 계산하는 중...")
        
        df = calculate_indicators_wilders(raw_df)
        
        if df.empty:
            st.error("❌ 지표 계산 후 유효한 데이터가 없습니다.")
            st.warning(f"""
            **문제 분석**:
            - 원본 데이터: {len(raw_df)}개
            - 지표 계산 후: {len(df)}개 (모두 NaN 제거됨)
            
            **해결 방법**:
            1. 더 긴 기간 선택 (최소 50개 이상 권장)
            2. 1일봉 선택 (더 많은 데이터 확보)
            """)
            st.stop()
        
        if len(df) < 10:
            st.warning(f"""
            ⚠️ 유효한 데이터가 매우 적습니다 ({len(df)}개).
            
            분석은 진행되지만 정확도가 낮을 수 있습니다.
            더 긴 기간을 선택하시는 것을 권장합니다.
            """)
        
        progress_placeholder.markdown(render_progress_bar(3, 6), unsafe_allow_html=True)
        status_text.info("🤖 앙상블 모델을 학습하는 중...")
        
        close_series = df['Close']
        
        if len(close_series) < 10:
            st.error("❌ 모델 학습에 필요한 최소 데이터가 부족합니다.")
            st.stop()
        
        # ═══════════════════════════════════════════════════════════════════
        # v2.5.0: 앙상블 모델 학습 (시간 프레임 기반 자동 선택)
        # ═══════════════════════════════════════════════════════════════════
        try:
            # 앙상블 모델 학습
            ensemble_models, ensemble_config = train_ensemble_models(
                data=close_series,
                features_df=df,
                interval=interval,
                forecast_days=forecast_days
            )
            
            if not ensemble_models:
                st.error("❌ 앙상블 모델 학습에 실패했습니다.")
                st.stop()
            
            st.success(f"✅ 앙상블 모델 학습 완료: {ensemble_config['description']}")
            
        except Exception as e:
            st.error(f"❌ 앙상블 모델 학습 중 오류: {e}")
            import traceback
            st.text(traceback.format_exc())
            st.stop()
        
        close_series = df['Close']
        
        if len(close_series) < 10:
            st.error("❌ 모델 학습에 필요한 최소 데이터가 부족합니다.")
            st.stop()
        
        # [개선됨] v2.4.0: 강건한 모델 학습 및 자동 계절성 감지
        try:
            hw_model, seasonality_info, window_size = fit_hw_model_robust(
                close_series, 
                max_window=500  # 최신 500개 데이터만 사용 (성능 개선)
            )
            
            # 계절성 정보 표시
            if seasonality_info['detected']:
                st.info(f"✅ 계절성 감지: 주기 {seasonality_info['period']}, "
                       f"타입 {seasonality_info['type']}, "
                       f"학습 데이터: {window_size}개")
            else:
                st.info(f"ℹ️ 비계절 모델 사용 (학습 데이터: {window_size}개)")
        
        except Exception as e:
            st.error(f"❌ 모델 학습 실패: {str(e)}")
            st.warning("""
            **해결 방법**:
            1. 더 긴 기간 선택
            2. 1일봉으로 변경
            3. 다른 코인 선택
            """)
            st.stop()
        
        pred_in_sample = hw_model.fittedvalues
        
        # [개선됨] v2.4.0: 오프셋 스케일링 예측
        forecast_steps = min(30, len(close_series) // 2)
        last_actual_value = close_series.iloc[-1]
        
        # 최근 추세 계산 (최근 20개 데이터의 선형 회귀 기울기)
        recent_window = min(20, len(close_series))
        recent_prices = close_series.iloc[-recent_window:].values
        recent_trend = np.polyfit(range(recent_window), recent_prices, 1)[0]
        
        future_forecast = forecast_with_offset_scaling(
            hw_model, 
            forecast_steps, 
            last_actual_value, 
            recent_trend
        )
        
        last_date = df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_steps)]
        future_df = pd.DataFrame({'예측 종가': future_forecast.values}, index=future_dates)
        
        progress_placeholder.markdown(render_progress_bar(4, 6), unsafe_allow_html=True)
        status_text.info("🕯️ 패턴을 분석하는 중...")
        
        patterns = detect_candlestick_patterns(df)
        
        progress_placeholder.markdown(render_progress_bar(5, 6), unsafe_allow_html=True)
        status_text.info("✅ 모델을 검증하는 중...")
        
        cv_results = perform_timeseries_cv(df, n_splits=min(5, len(df) // 20))
        
        progress_placeholder.markdown(render_progress_bar(6, 6), unsafe_allow_html=True)
        status_text.info("🎯 매매 전략을 생성하는 중...")
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR14'].iloc[-1]
        volatility = df['Volatility30d'].iloc[-1]
        atr_ratio = atr / current_price if current_price != 0 else 0.01
        
        hw_confidence = 75.0
        
        # [수정됨] v2.3.0: 레버리지 정보를 딕셔너리로 받음
        leverage_info = calculate_optimized_leverage(
            investment_amount=investment_amount,
            volatility=volatility,
            atr_ratio=atr_ratio,
            confidence=hw_confidence,
            max_leverage=leverage_ceiling,
            crypto_name=selected_crypto  # [추가됨] 코인 이름 전달
        )
        
        entry_price = current_price
        stop_loss = entry_price - (atr * stop_loss_k)
        take_profit = entry_price + (atr * stop_loss_k * 2)
        
        # [수정됨] v2.3.0: 권장 레버리지 사용
        risk_amount = investment_amount * risk_per_trade_pct
        position_size = (risk_amount * leverage_info['recommended']) / (entry_price - stop_loss)
        
        rr_ratio = calculate_rr_ratio(entry_price, take_profit, stop_loss)
        
        # 매도 전략 계산
        exit_strategy = calculate_exit_strategy(df, entry_price, atr, investment_amount, leverage_info['recommended'])
        
        progress_placeholder.empty()
        status_text.empty()
        
        st.success("✅ 분석이 완료되었습니다!")
        
        # 결과 출력
        render_data_summary(df, selected_crypto, interval_name)
        render_ai_forecast(future_df, hw_confidence)
        render_patterns(patterns)
        render_technical_indicators(df)
        render_validation_results(cv_results)
        # [추가됨] v2.2.1: AI 예측에 필요한 변수 추출
        ema_short = df['EMA50'].iloc[-1]
        ema_long = df['EMA200'].iloc[-1]
        rsi = df['RSI14'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        
        # [추가됨] AI 예측 실행
        ai_prediction = predict_trend_with_ai(
            df=df,
            current_price=current_price,
            ema_short=ema_short,
            ema_long=ema_long,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal
        )
        
        # [추가됨] AI 예측 결과 렌더링 (데이터 분석 결과 다음)
        render_ai_prediction(ai_prediction, current_price)
        
        # [추가됨] 포지션 추천 계산
        position_recommendation = recommend_position(
            ai_prediction=ai_prediction,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volatility=volatility
        )
        
        render_trading_strategy(current_price, leverage_info, entry_price,
                               stop_loss, take_profit, position_size,
                               rr_ratio, investment_amount)
        
        # [추가됨] 포지션 추천 렌더링 (매매 전략 직후)
        render_position_recommendation(position_recommendation)
        
        # 매도 전략 (신규)
        render_exit_strategy(exit_strategy, entry_price, investment_amount, leverage_info['recommended'])
        
        # 가격 차트
        st.markdown("---")
        st.markdown("### 📈 차트")
        
        tab1, tab2, tab3, tab4 = st.tabs(["💹 캔들스틱", "📊 거래량", "🔵 RSI", "📉 MACD"])
        
        with tab1:
            fig = create_candlestick_chart(df, selected_crypto)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_volume_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = create_rsi_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            fig = create_macd_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 오류가 발생했습니다: {str(e)}")
        st.warning("""
        **일반적인 해결 방법**:
        1. 캐시 초기화 후 재시도
        2. 더 긴 기간 선택 (최소 30일 이상)
        3. 1일봉으로 변경
        4. 다른 코인 선택
        5. 잠시 후 다시 시도
        """)
        
        if st.button("🔄 캐시 초기화"):
            st.cache_data.clear()
            st.rerun()
        
        with st.expander("🔍 상세 오류 정보 (개발자용)"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())
