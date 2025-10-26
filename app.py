# -*- coding: utf-8 -*-
"""
코인 AI 예측 시스템 - v2.0.0
- AI 트레이딩 분석 및 전략 요약 프롬프트 반영
- 분해능 선택 (1m, 5m, 1h, 1d)
- Wilder's Smoothing 방식 지표
- Holt-Winters Seasonal 모델
- TimeSeriesSplit 검증
- 투자금액 반영 레버리지 최적화
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
    
    @media (max-width: 600px) {
        .section-title { font-size: 22px; margin-top: 24px; }
        .dataframe { font-size: 11px; overflow-x: auto; display: block; }
        .stColumn { width: 100% !important; margin-bottom: 1rem; }
        [data-testid="stMetricValue"] { font-size: 20px; }
    }
    
    @media (min-width: 601px) and (max-width: 1024px) {
        .section-title { font-size: 28px; }
    }
    
    @media (min-width: 1200px) {
        .section-title { font-size: 36px; }
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .external-links {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .external-links a {
        padding: 10px 20px;
        background-color: #3498DB;
        color: white !important;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .external-links a:hover {
        background-color: #2980B9;
    }
    
    .pattern-card {
        background-color: #F8F9FA;
        border-left: 4px solid #3498DB;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
    
    .alert-box { padding: 15px; border-radius: 8px; margin: 10px 0; }
    .alert-success { background-color: #D4EDDA; border-left: 4px solid #28A745; color: #155724; }
    .alert-warning { background-color: #FFF3CD; border-left: 4px solid #FFC107; color: #856404; }
    .alert-danger { background-color: #F8D7DA; border-left: 4px solid #DC3545; color: #721C24; }
    
    .progress-step {
        display: inline-block;
        padding: 5px 15px;
        margin: 5px;
        background-color: #E8F4F8;
        border-radius: 20px;
        font-size: 14px;
    }
    
    .progress-step.active {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# 3) 상수 정의
# ────────────────────────────────────────────────────────────────────────
MAX_LEVERAGE_MAP = {
    'BTCUSDT': 125, 'ETHUSDT': 75, 'BNBUSDT': 50, 'DOGEUSDT': 50,
    'LTCUSDT': 50, 'AVAXUSDT': 50, 'IMXUSDT': 25, 'SOLUSDT': 50,
    'XRPUSDT': 50, 'ADAUSDT': 50,
}

LISTING_DATE_MAP = {
    'BTCUSDT': datetime.date(2017, 9, 2), 'ETHUSDT': datetime.date(2017, 8, 7),
    'BNBUSDT': datetime.date(2017, 7, 25), 'DOGEUSDT': datetime.date(2019, 4, 6),
    'LTCUSDT': datetime.date(2017, 6, 12), 'AVAXUSDT': datetime.date(2020, 7, 22),
    'IMXUSDT': datetime.date(2021, 6, 15), 'SOLUSDT': datetime.date(2020, 4, 10),
    'XRPUSDT': datetime.date(2018, 5, 14), 'ADAUSDT': datetime.date(2018, 4, 17),
}

POPULAR_CRYPTOS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LTC', 'IMX']

COLORS = {
    'bullish': '#00C853', 'bearish': '#FF1744', 'neutral': '#FFC107',
    'primary': '#2196F3', 'secondary': '#9C27B0', 'background': '#F5F5F5',
}

# 분해능 맵핑
INTERVAL_MAP = {
    '1분봉': '1m',
    '5분봉': '5m',
    '1시간봉': '1h',
    '1일봉': '1d'
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ────────────────────────────────────────────────────────────────────────
# 4) 헬퍼 함수 모음
# ────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_listing_date(symbol: str) -> datetime.date:
    """코인 상장일 조회"""
    if symbol in LISTING_DATE_MAP:
        return LISTING_DATE_MAP[symbol]
    try:
        yf_symbol = symbol[:-4] + "-USD"
        ticker = yf.Ticker(yf_symbol)
        df_full = ticker.history(period="max", interval="1d")
        if df_full is None or df_full.empty:
            return datetime.date.today()
        return df_full.index.min().date()
    except Exception:
        return datetime.date.today()


@st.cache_data(ttl=3600, show_spinner=False)
def load_crypto_data(symbol: str, start: datetime.date, end: datetime.date, interval: str = '1d') -> pd.DataFrame:
    """암호화폐 데이터 로드 (분해능 지원) - 개선된 버전 v2"""
    yf_ticker = symbol[:-4] + "-USD"
    df = pd.DataFrame()
    
    # ✅ 기간 검증 및 자동 조정
    days_diff = (end - start).days
    
    # yfinance API 제한 확인
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
    Wilder's Smoothing 방식으로 지표 계산
    - RSI: Wilder's EMA 방식
    - ATR: Wilder's Smoothing
    - MFI: Typical Price 기반
    """
    df = df.copy()
    
    # 일일 수익률
    df['일일수익률'] = df['Close'].pct_change()

    # 이동평균
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # ═══════════════════════════════════════════════════════════
    # RSI (Wilder's Smoothing)
    # ═══════════════════════════════════════════════════════════
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's EMA: alpha = 1/period
    period = 14
    alpha = 1.0 / period
    
    # 첫 번째 평균은 단순 평균
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Wilder's Smoothing 적용
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
    for i in range(period, len(df)):
        atr.iloc[i] = alpha * true_range.iloc[i] + (1 - alpha) * atr.iloc[i - 1]
    
    df['ATR14'] = atr
    df['Volatility30d'] = df['일일수익률'].rolling(window=30).std()

    # Stochastic
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['StochK14'] = (df['Close'] - low14) / (high14 - low14 + 1e-8) * 100

    # ═══════════════════════════════════════════════════════════
    # MFI (Typical Price 기반)
    # ═══════════════════════════════════════════════════════════
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['MF'] = typical_price * df['Volume']
    df['PosMF'] = df['MF'].where(df['Close'] > df['Close'].shift(1), 0)
    df['NegMF'] = df['MF'].where(df['Close'] < df['Close'].shift(1), 0)
    roll_pos = df['PosMF'].rolling(window=14).sum()
    roll_neg = df['NegMF'].rolling(window=14).sum()
    df['MFI14'] = 100 - (100 / (1 + roll_pos / (roll_neg + 1e-8)))

    # VWAP
    df['PV'] = df['Close'] * df['Volume']
    df['Cum_PV'] = df['PV'].cumsum()
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / (df['Cum_Vol'] + 1e-8)

    # 거래량 이동평균
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

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

    return df.dropna()


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
    캔들스틱 패턴 감지 (최근 발생일 포함)
    출력: 패턴명 | 발생일자 | 방향성
    """
    patterns = []
    df_sorted = df.sort_index(ascending=True)

    # EMA 교차 패턴
    ema50 = df['EMA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    ema50_prev = df['EMA50'].iloc[-2]
    ema200_prev = df['EMA200'].iloc[-2]

    if ema50 > ema200 and ema50_prev <= ema200_prev:
        patterns.append({
            'name': '🌟 골든 크로스',
            'date': df.index[-1],
            'conf': 90.0,
            'desc': 'EMA50이 EMA200을 상향 돌파',
            'impact': '장기 상승 추세 전환',
            'direction': '상승'
        })
    elif ema50 < ema200 and ema50_prev >= ema200_prev:
        patterns.append({
            'name': '💀 데드 크로스',
            'date': df.index[-1],
            'conf': 85.0,
            'desc': 'EMA50이 EMA200을 하향 돌파',
            'impact': '장기 하락 추세 전환',
            'direction': '하락'
        })

    # 캔들스틱 패턴 감지
    for i in range(2, min(len(df_sorted), 100)):
        o1, c1, h1, l1 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i - 2]
        o2, c2, h2, l2 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i - 1]
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
    
    # 최종 레버리지
    optimal_leverage = base_leverage * investment_factor * volatility_factor * confidence_factor
    
    # 최대값 제한
    return round(min(optimal_leverage, max_leverage), 2)


def calculate_mase(y_true, y_pred, y_train):
    """MASE (Mean Absolute Scaled Error) 계산"""
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / (d + 1e-10)


def timeseries_cv_validation(df: pd.DataFrame, n_splits: int = 5):
    """
    TimeSeriesSplit 기반 외부표본 검증
    반환: validation_results (DataFrame)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    close_values = df['Close'].values
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(close_values), 1):
        train_data = close_values[train_idx]
        test_data = close_values[test_idx]
        
        # Holt-Winters 모델 학습
        try:
            hw_model = sm.tsa.ExponentialSmoothing(
                train_data,
                trend='add',
                seasonal='add',
                seasonal_periods=min(7, len(train_data) // 2),
                initialization_method="estimated"
            ).fit(optimized=True)
            
            # 예측
            forecast = hw_model.forecast(steps=len(test_data))
            
            # 방향성 정확도
            actual_direction = np.sign(np.diff(test_data))
            pred_direction = np.sign(np.diff(forecast))
            accuracy = (actual_direction == pred_direction).mean() * 100
            
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
            value=f"{avg_volume/1e6:.1f}M"
        )
    
    with col4:
        high_52w = df['High'].tail(min(252, len(df))).max()
        low_52w = df['Low'].tail(min(252, len(df))).min()
        st.metric(
            label="최고/최저",
            value=f"${high_52w:,.0f}",
            delta=f"최저: ${low_52w:,.0f}"
        )


def render_price_chart(df: pd.DataFrame, future_df: pd.DataFrame, pred_in_sample: pd.Series, selected_crypto: str):
    """가격 차트 섹션"""
    st.markdown("<div class='section-title'>📈 가격 차트 및 예측</div>", unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('가격 및 이동평균', 'RSI (14일, Wilder)', '거래량'),
        row_heights=[0.5, 0.25, 0.25]
    )

    # 가격
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['Close'],
            name='실제 가격',
            line=dict(color=COLORS['primary'], width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MA50'],
            name='MA50',
            line=dict(color='orange', width=1, dash='dash')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['EMA200'],
            name='EMA200',
            line=dict(color='red', width=1, dash='dot')
        ),
        row=1, col=1
    )

    # In-sample 예측
    fig.add_trace(
        go.Scatter(
            x=pred_in_sample.index, y=pred_in_sample.values,
            name='AI 학습 예측',
            line=dict(color='purple', width=1, dash='dash'),
            opacity=0.7
        ),
        row=1, col=1
    )

    # 미래 예측
    fig.add_trace(
        go.Scatter(
            x=future_df.index, y=future_df['예측 종가'],
            name='30일 예측',
            line=dict(color=COLORS['bullish'], width=2),
            mode='lines+markers'
        ),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['RSI14'],
            name='RSI',
            line=dict(color=COLORS['secondary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(156, 39, 176, 0.1)'
        ),
        row=2, col=1
    )

    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # 거래량
    colors_volume = [COLORS['bullish'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                     else COLORS['bearish'] for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['Volume'],
            name='거래량',
            marker_color=colors_volume,
            opacity=0.7
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis3=dict(rangeslider=dict(visible=True, thickness=0.05)),
        template='plotly_white'
    )

    fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="거래량", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})


def render_indicators_tabs(df: pd.DataFrame):
    """기술적 지표 탭"""
    st.markdown("<div class='section-title'>📊 기술적 지표 분석 (Wilder's Method)</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📉 변동성 (ATR)", "📊 모멘텀", "💹 거래량", "🔄 MACD"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            latest_atr = df['ATR14'].iloc[-1]
            prev_atr = df['ATR14'].iloc[-2]
            delta_atr = latest_atr - prev_atr
            
            st.metric(
                label="ATR (14일, Wilder's Smoothing)",
                value=f"{latest_atr:.2f}",
                delta=f"{delta_atr:+.2f}"
            )
            
            if latest_atr > prev_atr:
                st.markdown("🔺 변동성이 증가하고 있습니다. 위험도 높음.")
            else:
                st.markdown("🔻 변동성이 감소하고 있습니다. 안정화 단계.")
        
        with col2:
            volatility = df['Volatility30d'].iloc[-1] * 100
            st.metric(
                label="30일 변동성 (σ)",
                value=f"{volatility:.2f}%"
            )
            st.markdown("※ Wilder's Smoothing은 급격한 변화를 완화하여 더 안정적인 ATR을 제공합니다.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            last_rsi = df['RSI14'].iloc[-1]
            st.metric("RSI (14, Wilder)", f"{last_rsi:.2f}")
            
            if last_rsi < 30:
                st.markdown("<div class='alert-box alert-success'>과매도 구간 - 반등 가능성</div>", unsafe_allow_html=True)
            elif last_rsi > 70:
                st.markdown("<div class='alert-box alert-danger'>과매수 구간 - 조정 가능성</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-box alert-warning'>중립 구간</div>", unsafe_allow_html=True)
        
        with col2:
            last_stoch = df['StochK14'].iloc[-1]
            st.metric("Stochastic %K", f"{last_stoch:.2f}")
            
            if last_stoch < 20:
                st.markdown("🔽 과매도")
            elif last_stoch > 80:
                st.markdown("🔼 과매수")
            else:
                st.markdown("➖ 중립")
    
    with tab3:
        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Vol_MA20'].iloc[-1]
        vol_ratio = (current_vol / avg_vol - 1) * 100 if avg_vol > 0 else 0
        
        st.metric(
            label="현재 거래량",
            value=f"{current_vol/1e6:.2f}M",
            delta=f"{vol_ratio:+.1f}% vs 20일 평균"
        )
        
        if vol_ratio > 50:
            st.markdown("📈 **비정상적으로 높은 거래량** - 강한 추세 변화 가능성")
        elif vol_ratio < -30:
            st.markdown("📉 **낮은 거래량** - 횡보 가능성")
        else:
            st.markdown("📊 정상 거래량 범위")
    
    with tab4:
        last_macd = df['MACD'].iloc[-1]
        last_signal = df['MACD_Signal'].iloc[-1]
        last_hist = df['MACD_Hist'].iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("MACD", f"{last_macd:.2f}")
            st.metric("Signal", f"{last_signal:.2f}")
        
        with col2:
            st.metric("Histogram", f"{last_hist:.2f}")
            
            if last_hist > 0 and df['MACD_Hist'].iloc[-2] <= 0:
                st.markdown("🟢 **강세 크로스오버** - 매수 신호")
            elif last_hist < 0 and df['MACD_Hist'].iloc[-2] >= 0:
                st.markdown("🔴 **약세 크로스오버** - 매도 신호")
            else:
                st.markdown("현재 추세 유지 중")


def render_pattern_analysis(patterns: list):
    """패턴 분석 섹션"""
    st.markdown("<div class='section-title'>🕯️ 캔들스틱 패턴 감지 (최근 발생일 포함)</div>", unsafe_allow_html=True)
    
    if not patterns:
        st.info("최근 감지된 패턴이 없습니다.")
        return
    
    for pattern in patterns:
        pattern_html = f"""
        <div class='pattern-card'>
            <h4>{pattern['name']}</h4>
            <p><strong>발생일자:</strong> {pattern['date'].strftime('%Y-%m-%d')}</p>
            <p><strong>신뢰도:</strong> {pattern['conf']:.1f}%</p>
            <p><strong>설명:</strong> {pattern['desc']}</p>
            <p><strong>영향:</strong> {pattern['impact']}</p>
            <p><strong>방향성:</strong> <span style="color:{'green' if pattern['direction']=='상승' else 'red' if pattern['direction']=='하락' else 'gray'};font-weight:bold;">{pattern['direction']}</span></p>
        </div>
        """
        st.markdown(pattern_html, unsafe_allow_html=True)


def render_validation_results(validation_df: pd.DataFrame):
    """검증 결과 섹션"""
    st.markdown("<div class='section-title'>✅ TimeSeriesSplit 외부표본 검증</div>", unsafe_allow_html=True)
    
    st.markdown("""
    **검증 방법**: TimeSeriesSplit을 사용하여 시계열 데이터를 순차적으로 분할하고, 각 Fold에서 모델 성능을 평가합니다.
    - **방향성 정확도**: 예측 방향(상승/하락)과 실제 방향의 일치율
    - **MASE**: Mean Absolute Scaled Error (낮을수록 좋음, <1이면 naive 예측보다 우수)
    """)
    
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    
    # 평균 정확도 계산
    try:
        accuracies = [float(acc.strip('%')) for acc in validation_df['Accuracy'] if acc != 'N/A']
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            if avg_accuracy >= 60:
                st.success(f"✅ 평균 방향성 정확도: {avg_accuracy:.2f}% (양호)")
            elif avg_accuracy >= 50:
                st.warning(f"⚠️ 평균 방향성 정확도: {avg_accuracy:.2f}% (보통)")
            else:
                st.error(f"❌ 평균 방향성 정확도: {avg_accuracy:.2f}% (낮음)")
    except:
        pass


def render_leverage_optimization(
    selected_crypto: str,
    investment_amount: float,
    entry_price: float,
    direction: str,
    confidence: float,
    volatility: float,
    atr_ratio: float
):
    """레버리지 최적화 섹션"""
    st.markdown("<div class='section-title'>⚖️ 코인별 레버리지 최적화 (투자금액 반영)</div>", unsafe_allow_html=True)
    
    max_leverage = MAX_LEVERAGE_MAP.get(selected_crypto, 50)
    
    optimal_leverage = calculate_optimized_leverage(
        investment_amount=investment_amount,
        volatility=volatility,
        atr_ratio=atr_ratio,
        confidence=confidence,
        max_leverage=max_leverage
    )
    
    # 예상 수익률 계산
    expected_return_pct = abs((entry_price * 1.02 - entry_price) / entry_price) * 100 * optimal_leverage
    
    st.markdown(f"""
    ### 최적화 결과
    
    | 항목 | 값 |
    |------|-----|
    | **코인명** | {selected_crypto[:-4]} |
    | **방향성** | {direction} |
    | **제안 레버리지** | {optimal_leverage}x |
    | **최대 허용 레버리지** | {max_leverage}x |
    | **투자금액** | ${investment_amount:,.2f} USDT |
    | **예상 수익률** | {expected_return_pct:.2f}% (2% 가격 변동 시) |
    | **신뢰도** | {confidence:.1f}% |
    | **변동성 (σ)** | {volatility*100:.2f}% |
    
    ### 근거 지표
    - **투자금액 조정**: {'높음 → 보수적' if investment_amount >= 5000 else '보통' if investment_amount >= 1000 else '낮음 → 공격적'}
    - **변동성 조정**: {'낮음 → 레버리지 증가' if volatility < 0.03 else '보통' if volatility < 0.05 else '높음 → 레버리지 감소'}
    - **신뢰도 조정**: {confidence:.1f}% 반영
    """)
    
    if optimal_leverage < max_leverage * 0.3:
        st.warning("⚠️ 현재 시장 조건에서는 낮은 레버리지가 권장됩니다.")
    elif optimal_leverage > max_leverage * 0.7:
        st.info("💡 높은 신뢰도로 적극적인 레버리지가 적용되었습니다.")


def render_forecast_history(hw_model, df: pd.DataFrame):
    """단기 추세 예측"""
    st.markdown("<div class='section-title'>📈 단기 추세 예측 (6단계)</div>", unsafe_allow_html=True)
    
    try:
        now = datetime.datetime.now()
        minute = (now.minute // 5) * 5
        base_time = now.replace(minute=minute, second=0, microsecond=0)
        forecast_steps = 6
        future_dates_5m = [base_time + datetime.timedelta(minutes=5 * (i + 1)) for i in range(forecast_steps)]
        
        hw_forecast = hw_model.forecast(steps=forecast_steps)
        actual_steps = len(hw_forecast)
        
        if actual_steps < forecast_steps:
            st.warning(f"⚠️ 예측 단계가 {forecast_steps}에서 {actual_steps}로 조정되었습니다.")
            future_dates_5m = future_dates_5m[:actual_steps]
        
        last_close = df['Close'].iloc[-1]
        
        time_list = []
        price_list = []
        change_list = []
        comment_list = []
        
        for i in range(actual_steps):
            time_list.append(future_dates_5m[i].strftime('%H:%M'))
            curr_val = hw_forecast.values[i]
            price_list.append(f"${curr_val:.2f}")
            
            prev_val = last_close if i == 0 else hw_forecast.values[i - 1]
            change_pct = ((curr_val - prev_val) / prev_val) * 100 if prev_val != 0 else 0
            change_list.append(f"{change_pct:+.2f}%")
            
            if curr_val < prev_val:
                comment = "🔻 하락 가능성 증가" if abs(change_pct) > 0.5 else "📉 완만한 하락"
            elif curr_val > prev_val:
                comment = "🔺 상승 가능성 증가" if abs(change_pct) > 0.5 else "📈 완만한 상승"
            else:
                comment = "➖ 횡보 예상"
            comment_list.append(comment)
        
        hist_df = pd.DataFrame({
            'Time': time_list,
            '예측가': price_list,
            '변동률': change_list,
            '코멘트': comment_list
        })
        
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ 예측 히스토리 생성 중 오류 발생: {str(e)}")


def render_position_summary(
    position_signal: str,
    entry_price: float,
    stop_loss_price: float,
    targets: list,
    recommended_leverage: float,
    position_qty: float,
    investment_amount: float,
    rate_win: float,
    learned_patterns: int
):
    """포지션 요약"""
    st.markdown("<div class='section-title'>💖 AI 매매 전략 요약</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 포지션 정보")
        st.markdown(f"""
        - **포지션**: {position_signal}
        - **진입가**: ${entry_price:.2f}
        - **손절가**: ${stop_loss_price:.2f}
        - **수량**: {position_qty:.4f}
        - **투자액**: ${investment_amount:.2f}
        - **레버리지**: {recommended_leverage}x
        """)
    
    with col2:
        st.markdown("### 🎯 목표가")
        for i, target in enumerate(targets, 1):
            pct_profit = abs((target - entry_price) / entry_price * 100)
            st.markdown(f"- **목표가 {i}**: ${target:.2f} ({pct_profit:.2f}%)")
    
    st.markdown("---")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("AI 승률", f"{rate_win}%")
    
    with col4:
        st.metric("학습 패턴 수", f"{learned_patterns:,}개")
    
    with col5:
        risk_reward = abs((targets[-1] - entry_price) / (entry_price - stop_loss_price)) if abs(entry_price - stop_loss_price) > 0 else 0
        st.metric("리스크/보상 비율", f"1:{risk_reward:.2f}")


def render_external_links(selected_crypto: str):
    """외부 링크"""
    st.markdown("<div class='section-title'>🔗 외부 차트 링크</div>", unsafe_allow_html=True)
    
    tv_url = f"https://www.tradingview.com/symbols/{selected_crypto}/"
    yf_url = f"https://finance.yahoo.com/quote/{selected_crypto[:-4]}-USD"
    
    links_html = f"""
    <div class='external-links'>
        <a href="{tv_url}" target="_blank">📊 TradingView에서 보기</a>
        <a href="{yf_url}" target="_blank">📈 Yahoo Finance에서 보기</a>
    </div>
    """
    st.markdown(links_html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────
# 6) 사이드바 (입력)
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🪙 코인 AI 예측 v2.0")
    st.markdown("---")
    
    with st.expander("❓ v2.0 새로운 기능", expanded=False):
        st.markdown("""
        ### 🆕 주요 업데이트
        
        ✅ **분해능 선택**: 1분, 5분, 1시간, 1일봉  
        ✅ **Wilder's Smoothing**: RSI/ATR 정확도 향상  
        ✅ **Seasonal 모델**: Holt-Winters 계절성 추가  
        ✅ **TimeSeriesSplit 검증**: 외부표본 성능 평가  
        ✅ **레버리지 최적화**: 투자금액 반영 자동 계산  
        ✅ **패턴 발생일**: 최근 발생일 명시  
        
        ### 📖 사용 방법
        1. 분해능 선택 (새로 추가!)
        2. 코인 선택
        3. 투자 설정
        4. 분석 시작
        """)
    
    st.markdown("## 1️⃣ 분해능 설정")
    interval_name = st.selectbox(
        "📊 분해능 선택",
        options=list(INTERVAL_MAP.keys()),
        index=3,  # 기본값: 1일봉
        help="분석할 시간 단위를 선택하세요"
    )
    interval = INTERVAL_MAP[interval_name]
    
    st.info(f"선택된 분해능: **{interval_name}** ({interval})")
    
    st.markdown("---")
    st.markdown("## 2️⃣ 코인 선택")
    
    selection_mode = st.radio(
        "선택 방식",
        options=["인기 코인", "직접 입력"],
        horizontal=True
    )
    
    if selection_mode == "인기 코인":
        base_symbol = st.selectbox(
            "인기 코인 선택",
            options=POPULAR_CRYPTOS,
            index=0
        )
    else:
        base_symbol = st.text_input(
            "코인 심볼 입력",
            value="BTC",
            help="예: BTC, ETH, DOGE"
        ).strip().upper()
    
    if not base_symbol:
        st.warning("코인 심볼을 입력해주세요.")
        st.stop()
    
    selected_crypto = base_symbol + "USDT" if not base_symbol.endswith("USDT") else base_symbol
    
    # 빠른 데이터 검증 (yfinance만 확인)
    yf_ticker_symbol = selected_crypto[:-4] + "-USD"
    
    with st.spinner(f"'{yf_ticker_symbol}' 데이터 확인 중..."):
        try:
            yf_ticker = yf.Ticker(yf_ticker_symbol)
            df_test = yf_ticker.history(period="5d")
            
            if df_test is None or df_test.empty:
                st.error(f"❌ '{yf_ticker_symbol}' 데이터를 찾을 수 없습니다. 올바른 코인 심볼인지 확인해주세요.")
                st.info("💡 지원되는 코인: BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LTC 등")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ 데이터 조회 실패: {str(e)}")
            st.info("💡 네트워크 연결을 확인하거나 다른 코인을 선택해주세요.")
            st.stop()
    
    st.success(f"✅ {selected_crypto} ({yf_ticker_symbol}) 선택됨")
    
    st.markdown("---")
    st.markdown("## 3️⃣ 분석 기간")
    
    mode = st.radio(
        "기간 선택 방식",
        options=["자동(상장일→오늘)", "직접 선택"],
        index=0,
        horizontal=True
    )
    
    if mode == "자동(상장일→오늘)":
        listing_date = get_listing_date(selected_crypto)
        today = datetime.date.today()
        
        # ✅ 분해능별 최대 기간 제한 적용
        if interval == '1m':
            max_days = 7
            START = max(listing_date, today - datetime.timedelta(days=max_days))
            st.warning(f"⚠️ 1분봉은 최근 {max_days}일 데이터만 제공됩니다 (yfinance API 제한)")
        elif interval == '5m':
            max_days = 60
            START = max(listing_date, today - datetime.timedelta(days=max_days))
            st.warning(f"⚠️ 5분봉은 최근 {max_days}일 데이터만 제공됩니다 (yfinance API 제한)")
        elif interval == '1h':
            max_days = 730
            START = max(listing_date, today - datetime.timedelta(days=max_days))
            st.info(f"ℹ️ 1시간봉은 최근 {max_days}일(2년) 데이터만 제공됩니다")
        else:  # 1d
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
        
        min_required = 100 if interval == '1d' else 50
        if len(raw_df) < min_required:
            st.error(f"❌ 최소 {min_required} 기간 이상의 데이터가 필요합니다. (현재: {len(raw_df)})")
            st.stop()
        
        # Step 2: 지표 계산 (Wilder's Method)
        progress_placeholder.markdown(render_progress_bar(2, 6), unsafe_allow_html=True)
        status_text.info("📊 Wilder's Smoothing 방식으로 지표를 계산하는 중...")
        
        df = calculate_indicators_wilders(raw_df)
        
        # Step 3: AI 모델 학습 (Seasonal 추가)
        progress_placeholder.markdown(render_progress_bar(3, 6), unsafe_allow_html=True)
        status_text.info("🤖 Holt-Winters Seasonal 모델을 학습하는 중...")
        
        close_series = df['Close']
        seasonal_periods = min(7, len(close_series) // 2) if len(close_series) > 14 else None
        
        if seasonal_periods:
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
        
        pred_in_sample = hw_model.fittedvalues
        future_forecast = hw_model.forecast(steps=30)
        
        last_date = df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(30)]
        future_df = pd.DataFrame({'예측 종가': future_forecast.values}, index=future_dates)
        
        # Step 4: 패턴 분석
        progress_placeholder.markdown(render_progress_bar(4, 6), unsafe_allow_html=True)
        status_text.info("🕯️ 패턴을 분석하는 중...")
        
        patterns = detect_candlestick_patterns(df)
        
        # Step 5: TimeSeriesSplit 검증
        progress_placeholder.markdown(render_progress_bar(5, 6), unsafe_allow_html=True)
        status_text.info("✅ TimeSeriesSplit 검증을 수행하는 중...")
        
        validation_df = timeseries_cv_validation(df, n_splits=5)
        
        # Step 6: 결과 계산
        progress_placeholder.markdown(render_progress_bar(6, 6), unsafe_allow_html=True)
        status_text.info("📈 최종 결과를 생성하는 중...")
        
        entry_price = raw_df['Close'].iloc[-1]
        far_price = future_df['예측 종가'].iloc[-1]
        
        max_loss_amount = investment_amount * risk_per_trade_pct
        stop_loss_pct = df['Volatility30d'].iloc[-1] * stop_loss_k
        per_coin_risk = entry_price * stop_loss_pct if entry_price > 0 else 0
        
        if per_coin_risk > 0:
            position_qty = max_loss_amount / per_coin_risk
        else:
            position_qty = 0.0
        
        notional_value = entry_price * position_qty
        recommended_leverage = (notional_value / investment_amount) if investment_amount > 0 else 1.0
        
        max_allowed = MAX_LEVERAGE_MAP.get(selected_crypto, leverage_ceiling)
        ultimate_ceiling = min(max_allowed, leverage_ceiling)
        
        # 레버리지 최적화
        confidence = 70.0  # 기본값
        volatility = df['Volatility30d'].iloc[-1]
        atr_ratio = df['ATR14'].iloc[-1] / entry_price if entry_price > 0 else 0
        
        recommended_leverage = calculate_optimized_leverage(
            investment_amount=investment_amount,
            volatility=volatility,
            atr_ratio=atr_ratio,
            confidence=confidence,
            max_leverage=ultimate_ceiling
        )
        
        pct_change = abs(far_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        if pct_change >= 0.05:
            num_targets = 5
        elif pct_change >= 0.02:
            num_targets = 3
        else:
            num_targets = 1
        
        if far_price > entry_price:
            direction = 'up'
            position_signal = "📈 매수 / 롱"
            stop_loss_price = entry_price * (1 - stop_loss_pct)
        else:
            direction = 'down'
            position_signal = "📉 매도 / 숏"
            stop_loss_price = entry_price * (1 + stop_loss_pct)
        
        targets = generate_targets(entry_price, num_targets, direction=direction)
        
        # AI 승률
        all_close = df['Close'].values
        all_pred = pred_in_sample.values
        correct_count = 0
        total_count = len(all_pred) - 1
        
        for i in range(1, len(all_pred)):
            actual_dir = 1 if all_close[i] > all_close[i - 1] else -1
            pred_dir = 1 if all_pred[i] > all_pred[i - 1] else -1
            if actual_dir == pred_dir:
                correct_count += 1
        
        rate_win = round((correct_count / total_count * 100.0) if total_count > 0 else 0.0, 2)
        learned_patterns = len(all_pred)
        
        progress_placeholder.empty()
        status_text.empty()
        
        # ════════════════════════════════════════════════════════════════
        # 결과 렌더링
        # ════════════════════════════════════════════════════════════════
        
        st.balloons()
        st.success("✅ 분석이 완료되었습니다!")
        
        # 1. 데이터 요약
        render_data_summary(df, selected_crypto, interval_name)
        
        # 2. 가격 차트
        render_price_chart(df, future_df, pred_in_sample, selected_crypto)
        
        # 3. 기술적 지표
        render_indicators_tabs(df)
        
        # 4. 패턴 분석
        render_pattern_analysis(patterns)
        
        # 5. TimeSeriesSplit 검증
        render_validation_results(validation_df)
        
        # 6. 레버리지 최적화
        render_leverage_optimization(
            selected_crypto=selected_crypto,
            investment_amount=investment_amount,
            entry_price=entry_price,
            direction=direction,
            confidence=confidence,
            volatility=volatility,
            atr_ratio=atr_ratio
        )
        
        # 7. 단기 예측
        render_forecast_history(hw_model, df)
        
        # 8. 포지션 요약
        render_position_summary(
            position_signal,
            entry_price,
            stop_loss_price,
            targets,
            recommended_leverage,
            position_qty,
            investment_amount,
            rate_win,
            learned_patterns
        )
        
        # 9. 외부 링크
        render_external_links(selected_crypto)
        
        # 면책 조항
        st.markdown("---")
        st.markdown("""
        <div style='background-color:#FFF3CD; padding:15px; border-radius:8px; border-left:4px solid #FFC107;'>
            <strong>⚠️ 면책 조항</strong><br>
            본 시스템은 교육 목적으로 제공되며, 투자 조언이 아닙니다. 
            암호화폐 투자는 높은 위험을 수반하므로 신중하게 결정하시기 바랍니다.
            모든 투자 결정과 손실은 투자자 본인의 책임입니다.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ 오류가 발생했습니다: {str(e)}")
        st.exception(e)
else:
    # 초기 화면
    st.markdown("""
    <div style='text-align:center; padding:50px;'>
        <h1>🪙 코인 AI 예측 시스템 v2.0</h1>
        <p style='font-size:18px; color:#666;'>
            <strong>🆕 새로운 기능:</strong> 분해능 선택, Wilder's Smoothing, Seasonal 모델, 
            TimeSeriesSplit 검증, 레버리지 최적화
        </p>
        <p style='font-size:18px; color:#666;'>
            왼쪽 사이드바에서 설정을 완료하고<br>
            <strong>🚀 분석 시작</strong> 버튼을 클릭하세요!
        </p>
        <br>
        <img src='https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/chart-line.svg' width='100'>
    </div>
    """, unsafe_allow_html=True)
