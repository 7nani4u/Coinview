# -*- coding: utf-8 -*-
"""
코인 AI 예측 시스템 - 개선 버전
- 반응형 디자인 강화
- Plotly 인터랙티브 차트
- 모듈화된 구조
- 개선된 UI/UX 흐름
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

# ────────────────────────────────────────────────────────────────────────
# 1) Streamlit 페이지 설정
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="코인 AI 예측 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────────────
# 2) 개선된 반응형 CSS
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* 전역 설정 */
    .main {
        padding: 1rem;
    }
    
    /* 섹션 제목 - 반응형 */
    .section-title {
        font-size: 32px;
        font-weight: bold;
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 3px solid #3498DB;
        color: #2C3E50;
    }
    
    /* 모바일 최적화 (600px 이하) */
    @media (max-width: 600px) {
        .section-title {
            font-size: 22px;
            margin-top: 24px;
        }
        
        /* 테이블 스크롤 개선 */
        .dataframe {
            font-size: 11px;
            overflow-x: auto;
            display: block;
        }
        
        /* 컬럼 스택 */
        .stColumn {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        /* 메트릭 크기 조정 */
        [data-testid="stMetricValue"] {
            font-size: 20px;
        }
    }
    
    /* 태블릿 최적화 (601px ~ 1024px) */
    @media (min-width: 601px) and (max-width: 1024px) {
        .section-title {
            font-size: 28px;
        }
    }
    
    /* 데스크톱 최적화 (1200px 이상) */
    @media (min-width: 1200px) {
        .section-title {
            font-size: 36px;
        }
    }
    
    /* 카드 스타일 */
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
    
    /* 외부 링크 */
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
    
    /* 패턴 카드 */
    .pattern-card {
        background-color: #F8F9FA;
        border-left: 4px solid #3498DB;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
    
    /* 알림 박스 */
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .alert-success {
        background-color: #D4EDDA;
        border-left: 4px solid #28A745;
        color: #155724;
    }
    
    .alert-warning {
        background-color: #FFF3CD;
        border-left: 4px solid #FFC107;
        color: #856404;
    }
    
    .alert-danger {
        background-color: #F8D7DA;
        border-left: 4px solid #DC3545;
        color: #721C24;
    }
    
    /* 진행률 표시 */
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
    'BTCUSDT': 125,
    'ETHUSDT': 75,
    'BNBUSDT': 50,
    'DOGEUSDT': 50,
    'LTCUSDT': 50,
    'AVAXUSDT': 50,
    'IMXUSDT': 25,
    'SOLUSDT': 50,
    'XRPUSDT': 50,
    'ADAUSDT': 50,
}

LISTING_DATE_MAP = {
    'BTCUSDT': datetime.date(2017, 9, 2),
    'ETHUSDT': datetime.date(2017, 8, 7),
    'BNBUSDT': datetime.date(2017, 7, 25),
    'DOGEUSDT': datetime.date(2019, 4, 6),
    'LTCUSDT': datetime.date(2017, 6, 12),
    'AVAXUSDT': datetime.date(2020, 7, 22),
    'IMXUSDT': datetime.date(2021, 6, 15),
    'SOLUSDT': datetime.date(2020, 4, 10),
    'XRPUSDT': datetime.date(2018, 5, 14),
    'ADAUSDT': datetime.date(2018, 4, 17),
}

# 인기 코인 리스트
POPULAR_CRYPTOS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LTC', 'IMX']

# 색상 팔레트 (고대비)
COLORS = {
    'bullish': '#00C853',
    'bearish': '#FF1744',
    'neutral': '#FFC107',
    'primary': '#2196F3',
    'secondary': '#9C27B0',
    'background': '#F5F5F5',
}

# TensorFlow 경고 억제
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


@st.cache_data(ttl=86400)
def load_crypto_data(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """암호화폐 데이터 로드"""
    yf_ticker = symbol[:-4] + "-USD"
    df = pd.DataFrame()
    
    try:
        ticker = yf.Ticker(yf_ticker)
        df_hist = ticker.history(
            start=start,
            end=end + datetime.timedelta(days=1),
            interval="1d"
        )
        if df_hist is not None and not df_hist.empty:
            df = df_hist.copy()
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df_max = yf.download(
                yf_ticker,
                start=start,
                end=end + datetime.timedelta(days=1),
                interval="1d",
                progress=False,
                threads=False
            )
            if df_max is not None and not df_max.empty:
                df = df_max.copy()
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Volume > 0 필터링
    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0].copy()
    
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산"""
    df = df.copy()
    
    # 일일 수익률
    df['일일수익률'] = df['Close'].pct_change()

    # 이동평균
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # 변동성
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['TR'] = true_range
    df['ATR14'] = df['TR'].rolling(window=14).mean()
    df['Volatility30d'] = df['일일수익률'].rolling(window=30).std()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['RSI14'] = 100 - (100 / (1 + rs))

    # Stochastic
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['StochK14'] = (df['Close'] - low14) / (high14 - low14 + 1e-8) * 100

    # MFI
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
    """캔들스틱 패턴 감지"""
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
            'impact': '장기 상승 추세 전환 가능성'
        })
    elif ema50 < ema200 and ema50_prev >= ema200_prev:
        patterns.append({
            'name': '💀 데드 크로스',
            'date': df.index[-1],
            'conf': 85.0,
            'desc': 'EMA50이 EMA200을 하향 돌파',
            'impact': '장기 하락 추세 전환 가능성'
        })

    # 캔들스틱 패턴 감지
    for i in range(2, min(len(df_sorted), 100)):  # 최근 100개만 검사
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
                'impact': '강력한 상승 신호'
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
                'impact': '상승 전환 가능성'
            })

        # Doji
        if abs(o3 - c3) <= (h3 - l3) * 0.1:
            patterns.append({
                'name': '✖️ Doji',
                'date': date3,
                'conf': 100.0,
                'desc': '매수/매도 균형',
                'impact': '추세 전환 가능성'
            })

    # 최근 5개만 반환
    return patterns[-5:] if patterns else []


# ────────────────────────────────────────────────────────────────────────
# 5) 렌더링 함수 (모듈화)
# ────────────────────────────────────────────────────────────────────────

def render_progress_bar(step: int, total: int = 5):
    """진행 상태 표시"""
    steps = ['데이터 로드', '지표 계산', 'AI 학습', '패턴 분석', '결과 생성']
    progress_html = '<div style="margin: 20px 0;">'
    for i, step_name in enumerate(steps[:total], 1):
        if i <= step:
            progress_html += f'<span class="progress-step active">{i}. {step_name}</span>'
        else:
            progress_html += f'<span class="progress-step">{i}. {step_name}</span>'
    progress_html += '</div>'
    return progress_html


def render_data_summary(df: pd.DataFrame, selected_crypto: str):
    """데이터 요약 섹션"""
    st.markdown("<div class='section-title'>📊 데이터 개요</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    daily_change = df['일일수익률'].iloc[-1] * 100
    avg_volume = df['Volume'].mean()
    total_days = len(df)
    
    with col1:
        st.metric(
            label="현재가 (USD)",
            value=f"${current_price:,.2f}",
            delta=f"{daily_change:+.2f}%"
        )
    
    with col2:
        high_52w = df['High'].tail(252).max()
        low_52w = df['Low'].tail(252).min()
        st.metric(
            label="52주 최고/최저",
            value=f"${high_52w:,.2f}",
            delta=f"최저: ${low_52w:,.2f}"
        )
    
    with col3:
        st.metric(
            label="평균 거래량",
            value=f"{avg_volume/1e6:.1f}M"
        )
    
    with col4:
        st.metric(
            label="분석 기간",
            value=f"{total_days}일"
        )


def render_price_chart(df: pd.DataFrame, future_df: pd.DataFrame, pred_in_sample: pd.Series, selected_crypto: str):
    """가격 차트 섹션 (Plotly 인터랙티브)"""
    st.markdown("<div class='section-title'>📈 가격 차트 및 예측</div>", unsafe_allow_html=True)
    
    # 서브플롯 생성: 가격, RSI, 거래량
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('가격 및 이동평균', 'RSI (14일)', '거래량'),
        row_heights=[0.5, 0.25, 0.25],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    # 1. 가격 차트
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['Close'],
            name='실제 가격',
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate='%{y:,.2f}<extra></extra>'
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

    # 2. RSI
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

    # RSI 기준선
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # 3. 거래량
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

    # 레이아웃 업데이트
    fig.update_layout(
        height=900,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis3=dict(rangeslider=dict(visible=True, thickness=0.05)),
        template='plotly_white'
    )

    # Y축 레이블
    fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="거래량", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})


def render_indicators_tabs(df: pd.DataFrame):
    """기술적 지표 탭"""
    st.markdown("<div class='section-title'>📊 기술적 지표 분석</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📉 변동성", "📊 모멘텀", "💹 거래량", "🔄 MACD"])
    
    with tab1:
        # ATR
        col1, col2 = st.columns(2)
        with col1:
            latest_atr = df['ATR14'].iloc[-1]
            prev_atr = df['ATR14'].iloc[-2]
            delta_atr = latest_atr - prev_atr
            
            st.metric(
                label="ATR (14일)",
                value=f"{latest_atr:.2f}",
                delta=f"{delta_atr:+.2f}"
            )
            
            if latest_atr > prev_atr:
                st.markdown("🔺 변동성이 증가하고 있습니다. 위험도가 높아졌습니다.")
            else:
                st.markdown("🔻 변동성이 감소하고 있습니다. 안정화 단계입니다.")
        
        with col2:
            volatility = df['Volatility30d'].iloc[-1] * 100
            st.metric(
                label="30일 변동성 (σ)",
                value=f"{volatility:.2f}%"
            )
            st.markdown("※ 높을수록 가격 변동이 큽니다.")
    
    with tab2:
        # RSI & Stochastic
        col1, col2 = st.columns(2)
        
        with col1:
            last_rsi = df['RSI14'].iloc[-1]
            st.metric("RSI (14)", f"{last_rsi:.2f}")
            
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
        # 거래량 분석
        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Vol_MA20'].iloc[-1]
        vol_ratio = (current_vol / avg_vol - 1) * 100
        
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
        # MACD
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


def render_support_resistance(df: pd.DataFrame, entry_price: float):
    """지지/저항 섹션 (Plotly)"""
    st.markdown("<div class='section-title'>🛡️ 지지 및 저항 레벨</div>", unsafe_allow_html=True)
    
    fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    high_price = df['High'].max()
    low_price = df['Low'].min()
    diff = high_price - low_price if high_price != low_price else 1e-8
    
    fib_levels = []
    for ratio in fib_ratios:
        level_price = high_price - diff * ratio
        fib_levels.append({'ratio': ratio, 'price': level_price})
    
    # Plotly 차트
    fig = go.Figure()
    
    # 현재가 표시
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[entry_price, entry_price],
        mode='lines+text',
        name='현재가',
        line=dict(color='blue', width=3),
        text=['', f'현재가: ${entry_price:.2f}'],
        textposition='middle right'
    ))
    
    # 피보나치 레벨
    colors_fib = ['#C62828', '#E53935', '#F57C00', '#FBC02D', '#7CB342', '#388E3C', '#1976D2']
    for lvl, color in zip(fib_levels, colors_fib):
        fig.add_hline(
            y=lvl['price'],
            line_dash="dash",
            line_color=color,
            annotation_text=f"Fib {lvl['ratio']*100:.1f}%: ${lvl['price']:.2f}",
            annotation_position="right",
            opacity=0.7
        )
    
    fig.update_layout(
        height=400,
        yaxis_title="가격 (USD)",
        xaxis=dict(visible=False),
        showlegend=False,
        margin=dict(l=10, r=150, t=30, b=10),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 레벨 테이블
    fib_df = pd.DataFrame(fib_levels)
    fib_df['ratio'] = fib_df['ratio'].apply(lambda x: f"{x*100:.1f}%")
    fib_df['price'] = fib_df['price'].apply(lambda x: f"${x:.2f}")
    fib_df.columns = ['비율', '가격']
    
    st.dataframe(fib_df, use_container_width=True, hide_index=True)


def render_pattern_analysis(patterns: list):
    """패턴 분석 섹션"""
    st.markdown("<div class='section-title'>🕯️ 캔들스틱 패턴 감지</div>", unsafe_allow_html=True)
    
    if not patterns:
        st.info("최근 감지된 패턴이 없습니다.")
        return
    
    for pattern in patterns:
        pattern_html = f"""
        <div class='pattern-card'>
            <h4>{pattern['name']}</h4>
            <p><strong>날짜:</strong> {pattern['date'].strftime('%Y-%m-%d')}</p>
            <p><strong>신뢰도:</strong> {pattern['conf']:.1f}%</p>
            <p><strong>설명:</strong> {pattern['desc']}</p>
            <p><strong>영향:</strong> {pattern['impact']}</p>
        </div>
        """
        st.markdown(pattern_html, unsafe_allow_html=True)


def render_ai_prediction_basis(df: pd.DataFrame, selected_crypto: str, entry_price: float, far_price: float):
    """AI 예측 근거"""
    st.markdown("<div class='section-title'>🤖 AI 예측 근거</div>", unsafe_allow_html=True)
    
    last_ma50 = df['MA50'].iloc[-1]
    last_rsi = df['RSI14'].iloc[-1]
    last_stoch = df['StochK14'].iloc[-1]
    last_macd = df['MACD'].iloc[-1]
    prev_macd = df['MACD'].iloc[-2]
    last_mfi = df['MFI14'].iloc[-1]
    
    price_trend = "하락세" if entry_price < last_ma50 else "상승세"
    price_trend_colored = (
        f"<span style='color:{COLORS['bearish']};font-weight:bold;'>하락세</span>" if price_trend == "하락세"
        else f"<span style='color:{COLORS['bullish']};font-weight:bold;'>상승세</span>"
    )
    
    macd_trend = "감소세" if last_macd < prev_macd else "증가세"
    macd_trend_colored = f"<span style='color:{COLORS['primary']};font-weight:bold;'>{macd_trend}</span>"
    
    # 모멘텀 분석
    if last_rsi < 30 and last_stoch < 20:
        momentum_desc = f"모멘텀 지표 RSI({last_rsi:.1f})와 스토캐스틱({last_stoch:.1f})이 과매도 상태입니다."
        future_trend = f"<span style='color:{COLORS['bullish']};font-weight:bold;'>반등</span> 가능성이 있습니다."
    elif last_rsi > 70 and last_stoch > 80:
        momentum_desc = f"모멘텀 지표 RSI({last_rsi:.1f})와 스토캐스틱({last_stoch:.1f})이 과매수 상태입니다."
        future_trend = f"<span style='color:{COLORS['bearish']};font-weight:bold;'>조정</span> 가능성이 있습니다."
    else:
        momentum_desc = f"모멘텀 지표 RSI({last_rsi:.1f})와 스토캐스틱({last_stoch:.1f})이 중립 영역입니다."
        future_trend = f"<span style='color:{COLORS['neutral']};font-weight:bold;'>횡보</span> 가능성이 있습니다."
    
    ai_reason = "<br>".join([
        f"📊 현재 {selected_crypto[:-4]} 가격은 <strong>${entry_price:.2f}</strong>로 {price_trend_colored}이며, MA50 대비 {price_trend}를 보입니다.",
        f"📈 MACD는 {macd_trend_colored}를 보이며 {'하락' if macd_trend=='감소세' else '상승'} 추세를 형성 중입니다.",
        f"💹 {momentum_desc}",
        f"🔮 따라서 향후 {future_trend}",
        f"💰 MFI는 {last_mfi:.1f}로 {'자금 유입이 활발' if last_mfi > 50 else '자금 유출 중'}합니다."
    ])
    
    st.markdown(f"<div style='line-height:1.8; font-size:16px; padding:15px; background-color:#F8F9FA; border-radius:8px;'>{ai_reason}</div>", unsafe_allow_html=True)


def render_forecast_history(hw_model, df: pd.DataFrame):
    """추세 예측 히스토리 (5분 간격)"""
    st.markdown("<div class='section-title'>📈 단기 추세 예측</div>", unsafe_allow_html=True)
    
    now = datetime.datetime.now()
    minute = (now.minute // 5) * 5
    base_time = now.replace(minute=minute, second=0, microsecond=0)
    future_dates_5m = [base_time + datetime.timedelta(minutes=5 * (i + 1)) for i in range(6)]
    
    hw_forecast_6 = hw_model.forecast(steps=6)
    hist_df_5m = pd.DataFrame({
        'Time': [d.strftime('%H:%M') for d in future_dates_5m],
        '예측가': [f"${v:.2f}" for v in hw_forecast_6.values],
        '변동률': [''] * 6,
        '코멘트': [''] * 6
    })
    
    last_close = df['Close'].iloc[-1]
    comments_list = []
    
    for i in range(len(hw_forecast_6)):
        if i == 0:
            prev_val = last_close
        else:
            prev_val = hw_forecast_6.values[i - 1]
        
        curr_val = hw_forecast_6.values[i]
        change_pct = ((curr_val - prev_val) / prev_val) * 100
        hist_df_5m.loc[i, '변동률'] = f"{change_pct:+.2f}%"
        
        if curr_val < prev_val:
            if abs(change_pct) > 0.5:
                comments_list.append("🔻 하락 가능성 증가")
            else:
                comments_list.append("📉 완만한 하락")
        elif curr_val > prev_val:
            if abs(change_pct) > 0.5:
                comments_list.append("🔺 상승 가능성 증가")
            else:
                comments_list.append("📈 완만한 상승")
        else:
            comments_list.append("➖ 횡보 예상")
    
    hist_df_5m['코멘트'] = comments_list
    
    st.dataframe(hist_df_5m, use_container_width=True, hide_index=True)


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
        risk_reward = abs((targets[-1] - entry_price) / (entry_price - stop_loss_price))
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
    st.markdown("# 🪙 코인 AI 예측")
    st.markdown("---")
    
    # 사용 가이드
    with st.expander("❓ 사용 방법", expanded=False):
        st.markdown("""
        ### 📖 이용 가이드
        
        1. **코인 선택**: 인기 코인 또는 직접 입력
        2. **기간 설정**: 자동(상장일부터) 또는 직접 선택
        3. **리스크 설정**:
           - 투자 금액
           - 리스크 비율 (1-2% 권장)
           - 손절 배수
        4. **분석 시작** 버튼 클릭
        
        #### 💡 초보자 팁
        - 리스크: 1%, 손절: 2배 권장
        - 변동성 높은 코인: 레버리지 낮게
        - 항상 손절가를 준수하세요
        """)
    
    st.markdown("## 1️⃣ 코인 선택")
    
    # 인기 코인 또는 직접 입력
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
    
    # 유효성 검사
    tv_url_test = f"https://www.tradingview.com/symbols/{selected_crypto}/"
    try:
        tv_resp = requests.get(tv_url_test, timeout=5)
    except Exception:
        tv_resp = None
    
    if tv_resp is None or tv_resp.status_code != 200:
        st.error(f"❌ '{selected_crypto}' 페이지를 찾을 수 없습니다.")
        st.stop()
    
    yf_ticker_symbol = selected_crypto[:-4] + "-USD"
    try:
        yf_ticker = yf.Ticker(yf_ticker_symbol)
        df_test = yf_ticker.history(period="1d")
        if df_test is None or df_test.empty:
            raise ValueError
    except Exception:
        st.error(f"❌ '{yf_ticker_symbol}' 데이터를 찾을 수 없습니다.")
        st.stop()
    
    st.success(f"✅ {selected_crypto} 선택됨")
    
    st.markdown("---")
    st.markdown("## 2️⃣ 분석 기간")
    
    mode = st.radio(
        "기간 선택 방식",
        options=["자동(상장일→오늘)", "직접 선택"],
        index=0,
        horizontal=True
    )
    
    if mode == "자동(상장일→오늘)":
        listing_date = get_listing_date(selected_crypto)
        today = datetime.date.today()
        START = listing_date
        END = today
        st.info(f"📅 {START} ~ {END}")
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
    st.markdown("## 3️⃣ 투자 설정")
    
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
        # 진행 상태 표시
        progress_placeholder = st.empty()
        status_text = st.empty()
        
        # Step 1: 데이터 로드
        progress_placeholder.markdown(render_progress_bar(1), unsafe_allow_html=True)
        status_text.info("🔍 데이터를 가져오는 중...")
        
        raw_df = load_crypto_data(selected_crypto, START, END)
        
        if raw_df.empty:
            st.error(f"❌ {selected_crypto} 데이터가 없습니다.")
            st.stop()
        
        if len(raw_df) < 100:
            st.error(f"❌ 최소 100일 이상의 데이터가 필요합니다. (현재: {len(raw_df)}일)")
            st.stop()
        
        # Step 2: 지표 계산
        progress_placeholder.markdown(render_progress_bar(2), unsafe_allow_html=True)
        status_text.info("📊 기술적 지표를 계산하는 중...")
        
        df = calculate_indicators(raw_df)
        
        # Step 3: AI 모델 학습
        progress_placeholder.markdown(render_progress_bar(3), unsafe_allow_html=True)
        status_text.info("🤖 AI 모델을 학습하는 중...")
        
        close_series = df['Close']
        hw_model = sm.tsa.ExponentialSmoothing(
            close_series,
            trend='add',
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        
        pred_in_sample = hw_model.fittedvalues
        future_forecast = hw_model.forecast(steps=30)
        
        last_date = df.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i + 1) for i in range(30)]
        future_df = pd.DataFrame({'예측 종가': future_forecast.values}, index=future_dates)
        
        # Step 4: 패턴 분석
        progress_placeholder.markdown(render_progress_bar(4), unsafe_allow_html=True)
        status_text.info("🕯️ 패턴을 분석하는 중...")
        
        patterns = detect_candlestick_patterns(df)
        
        # Step 5: 결과 계산
        progress_placeholder.markdown(render_progress_bar(5), unsafe_allow_html=True)
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
        recommended_leverage = round(max(1.0, min(recommended_leverage, ultimate_ceiling)), 2)
        
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
        
        # AI 승률 계산
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
        
        # 진행 상태 제거
        progress_placeholder.empty()
        status_text.empty()
        
        # ════════════════════════════════════════════════════════════════
        # 결과 렌더링 (개선된 순서)
        # ════════════════════════════════════════════════════════════════
        
        st.balloons()
        st.success("✅ 분석이 완료되었습니다!")
        
        # 1. 데이터 요약
        render_data_summary(df, selected_crypto)
        
        # 2. 가격 차트 (가장 먼저!)
        render_price_chart(df, future_df, pred_in_sample, selected_crypto)
        
        # 3. 기술적 지표 탭
        render_indicators_tabs(df)
        
        # 4. 지지/저항
        render_support_resistance(df, entry_price)
        
        # 5. 패턴 분석
        render_pattern_analysis(patterns)
        
        # 6. AI 예측 근거
        render_ai_prediction_basis(df, selected_crypto, entry_price, far_price)
        
        # 7. 단기 예측 히스토리
        render_forecast_history(hw_model, df)
        
        # 8. 포지션 요약 (최종 결론)
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
        <h1>🪙 코인 AI 예측 시스템</h1>
        <p style='font-size:18px; color:#666;'>
            왼쪽 사이드바에서 설정을 완료하고<br>
            <strong>🚀 분석 시작</strong> 버튼을 클릭하세요!
        </p>
        <br>
        <img src='https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/chart-line.svg' width='100'>
    </div>
    """, unsafe_allow_html=True)
