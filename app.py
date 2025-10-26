# -*- coding: utf-8 -*-
"""
코인 AI 예측 시스템 - v2.0.3
- 짧은 기간 데이터 완벽 지원
- 적응형 지표 계산 (데이터 길이에 따라 자동 조정)
- ExponentialSmoothing 빈 데이터 방지
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import streamlit as st
import os
import logging
import requests
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit

# ────────────────────────────────────────────────────────────────────────
# 1) Streamlit 페이지 설정
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="코인 AI 예측 시스템 v2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────────────
# 2) 개선된 반응형 CSS
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
    
    .metric-label {
        font-size: 18px;
        font-weight: 600;
        color: #34495E;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #2C3E50;
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
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 16px 32px;
        border: none;
        border-radius: 12px;
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px rgba(102, 126, 234, 0.4);
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
    
    .pattern-detail {
        font-size: 16px;
        margin: 8px 0;
        opacity: 0.95;
    }
    
    @media (max-width: 768px) {
        .section-title {
            font-size: 24px;
        }
        
        .stButton>button {
            font-size: 16px;
            padding: 12px 24px;
        }
        
        .pattern-title {
            font-size: 20px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# 3) 상수 및 설정
# ────────────────────────────────────────────────────────────────────────
CRYPTO_MAP = {
    "비트코인 (BTC)": "BTCUSDT",
    "이더리움 (ETH)": "ETHUSDT",
    "리플 (XRP)": "XRPUSDT",
    "도지코인 (DOGE)": "DOGEUSDT",
    "에이다 (ADA)": "ADAUSDT",
    "솔라나 (SOL)": "SOLUSDT"
}

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
# 4) 데이터 로드 함수 (개선된 버전)
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_crypto_data(
    symbol: str,
    start: datetime.date,
    end: datetime.date,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    암호화폐 데이터 로드 (yfinance 사용)
    - period 파라미터 우선 사용 (API 제한 자동 준수)
    - 3단계 fallback 메커니즘
    """
    df = pd.DataFrame()
    yf_ticker = symbol[:-4] + "-USD"
    
    # 기간 계산
    days_diff = (end - start).days
    
    # 분해능별 제한 적용
    interval_limits = {
        '1m': 7,
        '5m': 60,
        '1h': 730,
        '1d': 99999
    }
    
    max_days = interval_limits.get(interval, 99999)
    
    # 기간이 제한을 초과하면 자동으로 조정
    if days_diff > max_days:
        start = end - datetime.timedelta(days=max_days)
    
    # 방법 1: period 파라미터 사용 (더 안정적)
    try:
        ticker = yf.Ticker(yf_ticker)
        
        # 기간 계산
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
        
        # ✅ period 방식으로 먼저 시도 (더 안정적)
        df_hist = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
        
        if df_hist is not None and not df_hist.empty:
            # start/end 범위로 필터링
            df_hist = df_hist[(df_hist.index.date >= start) & (df_hist.index.date <= end)]
            
            if not df_hist.empty:
                df = df_hist.copy()
                if 'Volume' in df.columns:
                    df = df[df['Volume'] > 0].copy()
                if not df.empty:
                    return df
    except Exception as e:
        pass
    
    # 방법 2: start/end 파라미터 사용 (fallback)
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

    # 방법 3: yf.download() 사용 (최종 fallback)
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

    # 최종 검증 및 반환
    if df is not None and not df.empty:
        return df
    
    # 빈 DataFrame 반환 (캐싱되지 않음)
    return pd.DataFrame()


def calculate_indicators_wilders(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 적응형 지표 계산 (데이터 길이에 따라 자동 조정)
    - 짧은 기간에서도 안전하게 작동
    - 필수 지표만 NaN 제거, 선택적 지표는 유지
    """
    df = df.copy()
    data_len = len(df)
    
    # 일일 수익률 (필수)
    df['일일수익률'] = df['Close'].pct_change()

    # ═══════════════════════════════════════════════════════════
    # 적응형 윈도우 크기 설정
    # ═══════════════════════════════════════════════════════════
    # 기본 윈도우 크기
    window_12 = min(12, max(3, data_len // 10))
    window_14 = min(14, max(3, data_len // 8))
    window_20 = min(20, max(5, data_len // 6))
    window_26 = min(26, max(5, data_len // 5))
    window_30 = min(30, max(5, data_len // 4))
    window_50 = min(50, max(10, data_len // 3))
    window_200 = min(200, max(20, data_len // 2))
    
    # 이동평균 (적응형)
    if data_len >= window_50:
        df['MA50'] = df['Close'].rolling(window=window_50).mean()
        df['EMA50'] = df['Close'].ewm(span=window_50, adjust=False).mean()
    else:
        df['MA50'] = df['Close'].rolling(window=max(3, data_len // 3)).mean()
        df['EMA50'] = df['Close'].ewm(span=max(3, data_len // 3), adjust=False).mean()
    
    if data_len >= window_200:
        df['EMA200'] = df['Close'].ewm(span=window_200, adjust=False).mean()
    else:
        # 짧은 기간에서는 사용 가능한 최대 윈도우 사용
        df['EMA200'] = df['Close'].ewm(span=max(10, data_len // 2), adjust=False).mean()
    
    df['EMA12'] = df['Close'].ewm(span=window_12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=window_26, adjust=False).mean()

    # ═══════════════════════════════════════════════════════════
    # RSI (Wilder's Smoothing)
    # ═══════════════════════════════════════════════════════════
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's EMA: alpha = 1/period
    period = window_14
    alpha = 1.0 / period
    
    # 첫 번째 평균은 단순 평균
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Wilder's Smoothing 적용
    if data_len > period:
        for i in range(period, len(df)):
            avg_gain.iloc[i] = alpha * gain.iloc[i] + (1 - alpha) * avg_gain.iloc[i - 1]
            avg_loss.iloc[i] = alpha * loss.iloc[i] + (1 - alpha) * avg_loss.iloc[i - 1]
    
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI14'] = 100 - (100 / (1 + rs))

    # ═══════════════════════════════════════════════════════════
    # ATR (Wilder's Smoothing)
    # ═══════════════════════════════════════════════════════════
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['TR'] = true_range
    
    # Wilder's Smoothing for ATR
    atr = true_range.rolling(window=period).mean()
    if data_len > period:
        for i in range(period, len(df)):
            atr.iloc[i] = alpha * true_range.iloc[i] + (1 - alpha) * atr.iloc[i - 1]
    
    df['ATR14'] = atr
    df['Volatility30d'] = df['일일수익률'].rolling(window=window_30).std()

    # Stochastic (적응형)
    df['StochK14'] = 0.0
    if data_len >= window_14:
        low14 = df['Low'].rolling(window=window_14).min()
        high14 = df['High'].rolling(window=window_14).max()
        df['StochK14'] = (df['Close'] - low14) / (high14 - low14 + 1e-8) * 100

    # ═══════════════════════════════════════════════════════════
    # MFI (Typical Price 기반)
    # ═══════════════════════════════════════════════════════════
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['MF'] = typical_price * df['Volume']
    df['PosMF'] = df['MF'].where(df['Close'] > df['Close'].shift(1), 0)
    df['NegMF'] = df['MF'].where(df['Close'] < df['Close'].shift(1), 0)
    roll_pos = df['PosMF'].rolling(window=window_14).sum()
    roll_neg = df['NegMF'].rolling(window=window_14).sum()
    df['MFI14'] = 100 - (100 / (1 + roll_pos / (roll_neg + 1e-8)))

    # VWAP
    df['PV'] = df['Close'] * df['Volume']
    df['Cum_PV'] = df['PV'].cumsum()
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / (df['Cum_Vol'] + 1e-8)

    # 거래량 이동평균 (적응형)
    df['Vol_MA20'] = df['Volume'].rolling(window=window_20).mean()

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # EMA 교차 시그널
    df['Cross_Signal'] = 0
    ema50 = df['EMA50']
    ema200 = df['EMA200']
    cond_up = (ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))
    cond_down = (ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))
    df.loc[cond_up, 'Cross_Signal'] = 1
    df.loc[cond_down, 'Cross_Signal'] = -1

    # ✅ 핵심 수정: 필수 컬럼만 NaN 체크
    essential_cols = ['Close', 'High', 'Low', 'Volume', '일일수익률']
    df_clean = df.dropna(subset=essential_cols)
    
    # 선택적 지표는 0으로 채움 (계산 실패해도 분석 진행)
    optional_cols = ['RSI14', 'ATR14', 'StochK14', 'MFI14', 'MACD', 'MACD_Signal']
    for col in optional_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    
    return df_clean


def generate_targets(entry_price: float, num_targets: int, direction: str = 'down'):
    """목표가 생성"""
    targets = []
    for i in range(1, num_targets + 1):
        pct = i / (num_targets + 1)
        if direction == 'down':
            targets.append(entry_price * (1 - pct * 0.02))
        else:
            targets.append(entry_price * (1 + pct * 0.02))
    return targets


def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    """
    캔들스틱 패턴 감지 (발생일 포함)
    - Three White Soldiers
    - Morning Star
    - Doji
    """
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
                'date': date3,
                'conf': 100.0,
                'desc': '매수/매도 균형',
                'impact': '추세 전환 가능성',
                'direction': '중립'
            })

    return patterns[-5:] if patterns else []


def calculate_optimized_leverage(
    investment_amount: float,
    volatility: float,
    atr_ratio: float,
    confidence: float,
    max_leverage: int
) -> float:
    """
    투자금액 반영 레버리지 최적화
    - 투자 금액이 높을수록 보수적
    - ATR 낮고 신뢰도 높으면 점진적 증가
    """
    # 기본 레버리지
    base_leverage = 10
    
    # 투자금액 조정 (높을수록 감소)
    if investment_amount >= 10000:
        investment_factor = 0.6
    elif investment_amount >= 5000:
        investment_factor = 0.8
    elif investment_amount >= 1000:
        investment_factor = 1.0
    else:
        investment_factor = 1.2
    
    # 변동성 조정 (낮을수록 증가)
    if volatility < 0.02:
        volatility_factor = 1.5
    elif volatility < 0.05:
        volatility_factor = 1.2
    else:
        volatility_factor = 0.8
    
    # 신뢰도 조정
    confidence_factor = confidence / 100.0
    
    # ATR 조정 (낮을수록 증가)
    atr_factor = 1.0 / (atr_ratio + 0.5)
    
    # 최종 레버리지 계산
    optimal_leverage = base_leverage * investment_factor * volatility_factor * confidence_factor * atr_factor
    
    # 범위 제한
    optimal_leverage = max(1.0, min(optimal_leverage, float(max_leverage)))
    
    return round(optimal_leverage, 2)


def perform_timeseries_cv(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    TimeSeriesSplit을 사용한 모델 검증
    - 방향성 정확도
    - MASE (Mean Absolute Scaled Error)
    """
    if len(df) < n_splits * 10:
        # 데이터가 너무 짧으면 검증 생략
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
        
        # 학습 데이터가 너무 짧으면 스킵
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
        
        # Holt-Winters 모델 학습
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
            
            # 예측
            forecast = hw_model.forecast(steps=len(test_data))
            
            # 방향성 정확도
            if len(test_data) > 1:
                actual_direction = np.sign(np.diff(test_data))
                pred_direction = np.sign(np.diff(forecast))
                accuracy = (actual_direction == pred_direction).mean() * 100
            else:
                accuracy = 0.0
            
            # MASE
            mase = calculate_mase(test_data[1:], forecast[1:], train_data)
            
            # 평균 예측 오차
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
    """
    MASE (Mean Absolute Scaled Error) 계산
    - 1.0 미만: 예측이 naive 방법보다 우수
    - 1.0: naive 방법과 동일
    - 1.0 초과: naive 방법보다 열등
    """
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
    """
    Risk-Reward Ratio 계산
    RR = (목표 이익) / (최대 손실)
    """
    reward = abs(take_profit - entry_price)
    risk = abs(entry_price - stop_loss)
    
    if risk == 0:
        return 999.0
    
    return reward / risk


# ────────────────────────────────────────────────────────────────────────
# 5) 렌더링 함수 (모듈화)
# ────────────────────────────────────────────────────────────────────────

def render_progress_bar(step: int, total: int = 6):
    """진행 상태 표시"""
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
    """데이터 요약 섹션"""
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
    """AI 예측 섹션"""
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
    """패턴 분석 섹션"""
    st.markdown("<div class='section-title'>🕯️ 캔들스틱 패턴</div>", unsafe_allow_html=True)
    
    if not patterns:
        st.info("최근 주요 패턴이 감지되지 않았습니다.")
        return
    
    for pattern in patterns:
        with st.container():
            st.markdown(f"""
                <div class='pattern-card'>
                    <div class='pattern-title'>{pattern['name']}</div>
                    <div class='pattern-detail'>📅 발생일: {pattern['date'].strftime('%Y-%m-%d %H:%M') if hasattr(pattern['date'], 'strftime') else pattern['date']}</div>
                    <div class='pattern-detail'>🎯 신뢰도: {pattern['conf']}%</div>
                    <div class='pattern-detail'>📝 설명: {pattern['desc']}</div>
                    <div class='pattern-detail'>💡 영향: {pattern['impact']}</div>
                    <div class='pattern-detail'>🔄 방향: {pattern['direction']}</div>
                </div>
            """, unsafe_allow_html=True)


def render_validation_results(cv_results: pd.DataFrame):
    """모델 검증 결과 섹션"""
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
        
        # 평균 정확도 계산 (N/A 제외)
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


def render_trading_strategy(
    current_price: float,
    optimized_leverage: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    position_size: float,
    rr_ratio: float,
    investment_amount: float
):
    """매매 전략 섹션"""
    st.markdown("<div class='section-title'>🎯 매매 전략</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📍 진입 설정")
        st.metric(label="최적 레버리지", value=f"{optimized_leverage}x")
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
    
    # RR Ratio 평가
    if rr_ratio >= 3:
        st.success(f"✅ 우수한 RR Ratio ({rr_ratio:.2f}) - 리스크 대비 높은 수익 가능")
    elif rr_ratio >= 2:
        st.info(f"📊 적정한 RR Ratio ({rr_ratio:.2f}) - 균형잡힌 전략")
    else:
        st.warning(f"⚠️ 낮은 RR Ratio ({rr_ratio:.2f}) - 리스크 대비 수익이 작음")


def render_technical_indicators(df: pd.DataFrame):
    """기술적 지표 섹션"""
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
# 6) 메인 UI
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🚀 설정")
    st.markdown("---")
    
    st.markdown("## 1️⃣ 분해능 선택")
    resolution_choice = st.selectbox(
        "📈 시간 프레임",
        list(RESOLUTION_MAP.keys()),
        index=3,
        help="짧은 기간일수록 최신 데이터만 제공됩니다"
    )
    interval = RESOLUTION_MAP[resolution_choice]
    interval_name = resolution_choice
    
    # 분해능별 안내 메시지
    interval_info = {
        '1m': '⏱️ 1분봉: 최근 **7일**만 지원 (초단타 매매용)',
        '5m': '⏱️ 5분봉: 최근 **60일**만 지원 (단타 매매용)',
        '1h': '⏱️ 1시간봉: 최근 **2년**만 지원 (스윙 트레이딩용)',
        '1d': '⏱️ 1일봉: **전체 기간** 지원 (중장기 투자용)'
    }
    
    st.info(interval_info.get(interval, ''))
    
    st.markdown("---")
    st.markdown("## 2️⃣ 코인 선택")
    
    crypto_choice = st.selectbox(
        "💎 암호화폐",
        list(CRYPTO_MAP.keys())
    )
    selected_crypto = CRYPTO_MAP[crypto_choice]
    
    st.markdown("---")
    st.markdown("## 3️⃣ 분석 기간")
    
    period_choice = st.radio(
        "📅 기간 설정",
        ["자동 (분해능에 최적화)", "수동 설정"],
        help="자동 모드는 분해능별 제한을 자동으로 적용합니다"
    )
    
    if period_choice == "자동 (분해능에 최적화)":
        today = datetime.date.today()
        
        # 분해능별 자동 기간 설정
        interval_periods = {
            '1m': 7,
            '5m': 60,
            '1h': 730,
            '1d': 365 * 5  # 5년
        }
        
        days_back = interval_periods.get(interval, 180)
        START = today - datetime.timedelta(days=days_back)
        
        # 상장일 확인 (선택적)
        listing_dates = {
            "BTCUSDT": datetime.date(2017, 8, 17),
            "ETHUSDT": datetime.date(2017, 8, 17),
            "XRPUSDT": datetime.date(2018, 5, 14),
            "DOGEUSDT": datetime.date(2021, 5, 6),
            "ADAUSDT": datetime.date(2018, 4, 17),
            "SOLUSDT": datetime.date(2021, 8, 11)
        }
        
        listing_date = listing_dates.get(selected_crypto, START)
        
        # 상장일 이후만 선택
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

# ────────────────────────────────────────────────────────────────────────
# 7) 메인 로직
# ────────────────────────────────────────────────────────────────────────
if bt:
    try:
        progress_placeholder = st.empty()
        status_text = st.empty()
        
        # Step 1: 데이터 로드
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
            1. 더 최근 기간 선택 (최근 30일 권장)
            2. 분해능을 1일봉으로 변경
            3. 다른 코인 선택
            4. 잠시 후 다시 시도
            """)
            
            # 캐시 초기화 버튼 추가
            if st.button("🔄 캐시 초기화 후 재시도"):
                st.cache_data.clear()
                st.rerun()
            st.stop()
        
        # ✅ 최소 데이터 요구사항 완화
        min_required = 20  # 모든 분해능에서 최소 20개만 요구
        if len(raw_df) < min_required:
            st.error(f"❌ 최소 {min_required} 기간 이상의 데이터가 필요합니다. (현재: {len(raw_df)})")
            st.warning("""
            **해결 방법**:
            1. 더 긴 기간 선택
            2. 다른 분해능 선택 (1일봉 권장)
            3. 다른 코인 선택
            """)
            st.stop()
        
        # Step 2: 지표 계산 (Wilder's Method - 적응형)
        progress_placeholder.markdown(render_progress_bar(2, 6), unsafe_allow_html=True)
        status_text.info("📊 적응형 지표를 계산하는 중...")
        
        df = calculate_indicators_wilders(raw_df)
        
        # ✅ 지표 계산 후 데이터 검증
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
        
        # Step 3: AI 모델 학습 (Seasonal 추가)
        progress_placeholder.markdown(render_progress_bar(3, 6), unsafe_allow_html=True)
        status_text.info("🤖 Holt-Winters Seasonal 모델을 학습하는 중...")
        
        # ✅ 모델 학습 전 데이터 검증
        close_series = df['Close']
        
        if len(close_series) < 10:
            st.error("❌ 모델 학습에 필요한 최소 데이터가 부족합니다.")
            st.stop()
        
        # ✅ 적응형 seasonal_periods 설정
        seasonal_periods = max(2, min(7, len(close_series) // 3))
        
        try:
            if seasonal_periods >= 2 and len(close_series) >= seasonal_periods * 2:
                hw_model = sm.tsa.ExponentialSmoothing(
                    close_series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated"
                ).fit(optimized=True)
            else:
                hw_model = sm.tsa.ExponentialSmoothing(
                    close_series,
                    trend='add',
                    seasonal=None,
                    initialization_method="estimated"
                ).fit(optimized=True)
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
        
        # ✅ 예측 기간 조정
        forecast_steps = min(30, len(close_series) // 2)
        future_forecast = hw_model.forecast(steps=forecast_steps)
        
        last_date = df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_steps)]
        future_df = pd.DataFrame({'예측 종가': future_forecast.values}, index=future_dates)
        
        # Step 4: 패턴 분석
        progress_placeholder.markdown(render_progress_bar(4, 6), unsafe_allow_html=True)
        status_text.info("🕯️ 패턴을 분석하는 중...")
        
        patterns = detect_candlestick_patterns(df)
        
        # Step 5: 모델 검증 (TimeSeriesSplit)
        progress_placeholder.markdown(render_progress_bar(5, 6), unsafe_allow_html=True)
        status_text.info("✅ 모델을 검증하는 중...")
        
        cv_results = perform_timeseries_cv(df, n_splits=min(5, len(df) // 20))
        
        # Step 6: 매매 전략 수립
        progress_placeholder.markdown(render_progress_bar(6, 6), unsafe_allow_html=True)
        status_text.info("🎯 매매 전략을 생성하는 중...")
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR14'].iloc[-1]
        volatility = df['Volatility30d'].iloc[-1]
        atr_ratio = atr / current_price if current_price != 0 else 0.01
        
        # 신뢰도 계산
        hw_confidence = 75.0  # 기본값
        
        # 레버리지 최적화
        optimized_leverage = calculate_optimized_leverage(
            investment_amount=investment_amount,
            volatility=volatility,
            atr_ratio=atr_ratio,
            confidence=hw_confidence,
            max_leverage=leverage_ceiling
        )
        
        # 진입가, 손절가, 목표가 계산
        entry_price = current_price
        stop_loss = entry_price - (atr * stop_loss_k)
        take_profit = entry_price + (atr * stop_loss_k * 2)
        
        # 포지션 크기 계산
        risk_amount = investment_amount * risk_per_trade_pct
        position_size = (risk_amount * optimized_leverage) / (entry_price - stop_loss)
        
        # RR Ratio 계산
        rr_ratio = calculate_rr_ratio(entry_price, take_profit, stop_loss)
        
        # 진행 상태 제거
        progress_placeholder.empty()
        status_text.empty()
        
        # ═══════════════════════════════════════════════════════════
        # 결과 출력
        # ═══════════════════════════════════════════════════════════
        st.success("✅ 분석이 완료되었습니다!")
        
        # 1. 데이터 요약
        render_data_summary(df, selected_crypto, interval_name)
        
        # 2. AI 예측
        render_ai_forecast(future_df, hw_confidence)
        
        # 3. 캔들스틱 패턴
        render_patterns(patterns)
        
        # 4. 기술적 지표
        render_technical_indicators(df)
        
        # 5. 모델 검증
        render_validation_results(cv_results)
        
        # 6. 매매 전략
        render_trading_strategy(
            current_price=current_price,
            optimized_leverage=optimized_leverage,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            rr_ratio=rr_ratio,
            investment_amount=investment_amount
        )
        
        # 7. 가격 차트
        st.markdown("<div class='section-title'>📈 가격 차트</div>", unsafe_allow_html=True)
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('가격', 'RSI', 'MACD'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 가격 차트
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='가격'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA50'],
                name='EMA50',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA200'],
                name='EMA200',
                line=dict(color='purple', width=2)
            ),
            row=1, col=1
        )
        
        # RSI 차트
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI14'],
                name='RSI',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD 차트
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Hist'],
                name='Histogram',
                marker_color='gray'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=900,
            template="plotly_white",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="날짜", row=3, col=1)
        fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
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
        
        # 디버그 정보 (선택적)
        with st.expander("🔍 상세 오류 정보 (개발자용)"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())
