"""
AI 기반 암호화폐 투자 전략 분석 시스템 v2.2.1 (긴급 버그 수정)

주요 변경사항:
- [수정됨] v2.2.1: ema_short, ema_long 변수 명시적 추출 추가
- [추가됨] v2.2.0: AI 예측 결과 섹션
- [추가됨] v2.2.0: 포지션 추천 기능
- [기존] v2.1.2: Keep-Alive 기능
- [기존] v2.1.1: 레버리지 로직 수정
- [기존] v2.1.0: 61개 캔들스틱 패턴 + 매도 전략
- [기존] v2.0.3: IndexError 완벽 해결 + UI 개선
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TA-Lib 선택적 import
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    st.warning("⚠️ TA-Lib가 설치되지 않았습니다. 기본 3개 패턴만 사용됩니다.")


# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="AI 암호화폐 투자 전략 v2.2.1",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== Keep-Alive 서버 ====================
def start_keep_alive():
    """Flask 백그라운드 서버 시작 (Streamlit Cloud 슬립 모드 방지)"""
    try:
        from keep_alive import keep_alive
        keep_alive()
        st.sidebar.success("✅ Keep-Alive 서버 실행 중")
    except ImportError:
        st.sidebar.info("ℹ️ keep_alive.py가 없습니다. 로컬 환경에서는 필요하지 않습니다.")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Keep-Alive 서버 시작 실패: {str(e)}")


# ==================== 데이터 로드 함수 ====================
@st.cache_data(ttl=300, show_spinner=False)
def load_crypto_data(symbol, resolution='1h', days=90):
    """
    암호화폐 데이터 로드 (yfinance API 제한 대응)
    
    Args:
        symbol: 암호화폐 심볼 (예: 'BTC-USD')
        resolution: 분해능 ('1h', '4h', '1d' 등)
        days: 조회 일수
    
    Returns:
        pandas.DataFrame: OHLCV 데이터
    """
    try:
        # yfinance API 제한 대응
        max_days_map = {
            '1m': 7,
            '5m': 60,
            '15m': 60,
            '30m': 60,
            '1h': 730,  # yfinance 1시간봉 최대 730일
            '4h': 730,
            '1d': 3650,
            '1wk': 3650
        }
        
        max_allowed_days = max_days_map.get(resolution, 730)
        actual_days = min(days, max_allowed_days)
        
        if actual_days < days:
            st.warning(f"⚠️ {resolution} 분해능은 최대 {max_allowed_days}일까지만 지원됩니다. {actual_days}일로 조정됩니다.")
        
        # period 파라미터 사용 (start/end보다 안정적)
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{actual_days}d", interval=resolution)
        
        if data.empty:
            st.error(f"❌ {symbol} 데이터를 가져올 수 없습니다.")
            return pd.DataFrame()
        
        # 컬럼명 표준화
        data.columns = [col.capitalize() for col in data.columns]
        
        # 필수 컬럼 검증
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"❌ 필수 컬럼 누락: {missing_cols}")
            return pd.DataFrame()
        
        # 결측치 처리
        data = data.dropna()
        
        # 인덱스 초기화 (타임존 제거)
        data.index = pd.to_datetime(data.index).tz_localize(None)
        
        return data
    
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류: {str(e)}")
        return pd.DataFrame()


# ==================== 기술적 지표 계산 ====================
def calculate_indicators(data, ema_short_period=50, ema_long_period=200):
    """
    기술적 지표 계산 (적응형 윈도우 적용)
    
    Args:
        data: OHLCV 데이터프레임
        ema_short_period: 단기 EMA 기간 (기본 50)
        ema_long_period: 장기 EMA 기간 (기본 200)
    
    Returns:
        pandas.DataFrame: 지표가 추가된 데이터프레임
    """
    if data.empty or len(data) < 14:  # RSI 최소 요구 길이
        st.error("❌ 데이터가 부족합니다. 최소 14개 이상의 데이터가 필요합니다.")
        return data
    
    df = data.copy()
    
    # 적응형 윈도우 크기 조정
    actual_short = min(ema_short_period, len(df) - 1)
    actual_long = min(ema_long_period, len(df) - 1)
    
    if actual_short < ema_short_period:
        st.warning(f"⚠️ 데이터 길이가 부족하여 EMA50 기간을 {actual_short}로 조정합니다.")
    if actual_long < ema_long_period:
        st.warning(f"⚠️ 데이터 길이가 부족하여 EMA200 기간을 {actual_long}로 조정합니다.")
    
    # EMA 계산
    df['EMA50'] = df['Close'].ewm(span=actual_short, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=actual_long, adjust=False).mean()
    
    # RSI 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD 계산
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # 볼린저 밴드
    sma_20 = df['Close'].rolling(window=20, min_periods=1).mean()
    std_20 = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Middle'] = sma_20
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    
    # ATR 계산 (매도 전략용)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14, min_periods=1).mean()
    
    # 거래량 이동평균
    df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    
    # 선택적 NaN 제거 (충분한 데이터가 있을 때만)
    if len(df) > 50:
        df = df.dropna()
    
    return df


# ==================== AI 예측 함수 ====================
def predict_trend_with_ai(data):
    """
    [추가됨] v2.2.0: 4개 지표 기반 AI 예측 알고리즘
    
    가중치 시스템:
    - 이동평균 분석: 30%
    - RSI 분석: 25%
    - MACD 분석: 25%
    - 거래량 분석: 20%
    
    Args:
        data: 지표가 계산된 데이터프레임
    
    Returns:
        dict: {
            'predicted_trend': 'bullish'|'bearish'|'neutral',
            'confidence': 0-100,
            'reasoning': {
                'ema_analysis': str,
                'rsi_analysis': str,
                'macd_analysis': str,
                'volume_analysis': str
            }
        }
    """
    if data.empty or len(data) < 2:
        return {
            'predicted_trend': 'neutral',
            'confidence': 0,
            'reasoning': {
                'ema_analysis': '데이터 부족',
                'rsi_analysis': '데이터 부족',
                'macd_analysis': '데이터 부족',
                'volume_analysis': '데이터 부족'
            }
        }
    
    # 최신 데이터 추출
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    
    signals = []
    reasoning = {}
    
    # 1. 이동평균 분석 (30%)
    ema50 = latest['EMA50']
    ema200 = latest['EMA200']
    close = latest['Close']
    
    if ema50 > ema200:
        if close > ema50:
            signals.append(1.0)  # 강한 상승
            reasoning['ema_analysis'] = "골든크로스 형성 중 (EMA50 > EMA200)"
        else:
            signals.append(0.5)  # 약한 상승
            reasoning['ema_analysis'] = "골든크로스이나 현재가가 EMA50 아래"
    else:
        if close < ema50:
            signals.append(-1.0)  # 강한 하락
            reasoning['ema_analysis'] = "데드크로스 형성 중 (EMA50 < EMA200)"
        else:
            signals.append(-0.5)  # 약한 하락
            reasoning['ema_analysis'] = "데드크로스이나 현재가가 EMA50 위"
    
    # 2. RSI 분석 (25%)
    rsi = latest['RSI']
    if rsi > 70:
        signals.append(-0.8)  # 과매수
        reasoning['rsi_analysis'] = f"과매수 영역 (RSI: {rsi:.1f})"
    elif rsi < 30:
        signals.append(0.8)  # 과매도
        reasoning['rsi_analysis'] = f"과매도 영역 (RSI: {rsi:.1f})"
    elif 45 <= rsi <= 55:
        signals.append(0.0)  # 중립
        reasoning['rsi_analysis'] = f"중립 영역 (RSI: {rsi:.1f})"
    elif rsi > 55:
        signals.append(0.4)  # 약한 상승
        reasoning['rsi_analysis'] = f"상승 추세 (RSI: {rsi:.1f})"
    else:
        signals.append(-0.4)  # 약한 하락
        reasoning['rsi_analysis'] = f"하락 추세 (RSI: {rsi:.1f})"
    
    # 3. MACD 분석 (25%)
    macd = latest['MACD']
    signal_line = latest['Signal']
    macd_hist = latest['MACD_Hist']
    prev_macd_hist = prev['MACD_Hist']
    
    if macd > signal_line and macd_hist > 0:
        if macd_hist > prev_macd_hist:
            signals.append(1.0)  # 강한 상승
            reasoning['macd_analysis'] = "양수 전환 및 히스토그램 증가 (상승 모멘텀)"
        else:
            signals.append(0.5)  # 약한 상승
            reasoning['macd_analysis'] = "양수이나 히스토그램 감소 (모멘텀 약화)"
    elif macd < signal_line and macd_hist < 0:
        if macd_hist < prev_macd_hist:
            signals.append(-1.0)  # 강한 하락
            reasoning['macd_analysis'] = "음수 전환 및 히스토그램 감소 (하락 모멘텀)"
        else:
            signals.append(-0.5)  # 약한 하락
            reasoning['macd_analysis'] = "음수이나 히스토그램 증가 (하락 둔화)"
    else:
        signals.append(0.0)  # 중립
        reasoning['macd_analysis'] = "신호선 근처 (중립)"
    
    # 4. 거래량 분석 (20%)
    volume = latest['Volume']
    volume_ma = latest['Volume_MA']
    
    volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
    price_change = (close - prev['Close']) / prev['Close'] if prev['Close'] > 0 else 0
    
    if volume_ratio > 1.5 and price_change > 0:
        signals.append(0.8)  # 거래량 급증 + 상승
        reasoning['volume_analysis'] = f"거래량 급증 (+{(volume_ratio-1)*100:.0f}%) 및 가격 상승"
    elif volume_ratio > 1.5 and price_change < 0:
        signals.append(-0.8)  # 거래량 급증 + 하락
        reasoning['volume_analysis'] = f"거래량 급증 (+{(volume_ratio-1)*100:.0f}%) 및 가격 하락"
    elif volume_ratio < 0.7:
        signals.append(0.0)  # 거래량 감소 (관심 부족)
        reasoning['volume_analysis'] = f"거래량 감소 (-{(1-volume_ratio)*100:.0f}%)"
    else:
        signals.append(0.0)  # 평균적 거래량
        reasoning['volume_analysis'] = "평균 거래량 수준"
    
    # 가중 평균 계산
    weights = [0.30, 0.25, 0.25, 0.20]  # 이동평균, RSI, MACD, 거래량
    weighted_signal = sum(s * w for s, w in zip(signals, weights))
    
    # 추세 판단
    if weighted_signal > 0.3:
        trend = 'bullish'
    elif weighted_signal < -0.3:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    # 신뢰도 계산 (0-100)
    confidence = min(abs(weighted_signal) * 100, 100)
    
    return {
        'predicted_trend': trend,
        'confidence': round(confidence, 1),
        'reasoning': reasoning
    }


def recommend_position(ai_prediction, data):
    """
    [추가됨] v2.2.0: 확률 기반 포지션 추천
    
    Args:
        ai_prediction: predict_trend_with_ai() 결과
        data: 지표가 계산된 데이터프레임
    
    Returns:
        dict: {
            'position': 'long'|'short'|'hold',
            'probability': 45-85,
            'reason': str
        }
    """
    trend = ai_prediction['predicted_trend']
    confidence = ai_prediction['confidence']
    reasoning = ai_prediction['reasoning']
    
    # 신뢰도가 낮으면 관망
    if confidence < 60:
        return {
            'position': 'hold',
            'probability': 50,
            'reason': f"예측 신뢰도가 낮아 관망을 권장합니다 (신뢰도: {confidence}%)"
        }
    
    # 기본 확률 계산
    base_probability = 50 + (confidence * 0.5)  # 50-100%
    
    # 변동성 조정
    if not data.empty and 'ATR' in data.columns:
        latest_atr = data['ATR'].iloc[-1]
        atr_mean = data['ATR'].mean()
        volatility_ratio = latest_atr / atr_mean if atr_mean > 0 else 1.0
        
        if volatility_ratio > 1.5:
            adjustment = -5  # 높은 변동성 → 확률 감소
        elif volatility_ratio < 0.7:
            adjustment = 5   # 낮은 변동성 → 확률 증가
        else:
            adjustment = 0
    else:
        adjustment = 0
    
    final_probability = base_probability + adjustment
    final_probability = max(45, min(85, final_probability))  # 45-85% 범위로 제한
    
    # 포지션 결정
    if trend == 'bullish':
        position = 'long'
        # 가장 강한 시그널 찾기
        main_reason = max(reasoning.items(), key=lambda x: len(x[1]))[1]
    elif trend == 'bearish':
        position = 'short'
        main_reason = max(reasoning.items(), key=lambda x: len(x[1]))[1]
    else:
        position = 'hold'
        main_reason = "중립 구간으로 진입 시점이 명확하지 않습니다"
    
    return {
        'position': position,
        'probability': round(final_probability, 0),
        'reason': main_reason
    }


# ==================== AI 예측 결과 UI ====================
def render_ai_prediction(ai_prediction):
    """
    [추가됨] v2.2.0: AI 예측 결과 섹션 렌더링
    
    Args:
        ai_prediction: predict_trend_with_ai() 결과
    """
    st.markdown("### 🤖 AI 예측 결과")
    
    trend = ai_prediction['predicted_trend']
    confidence = ai_prediction['confidence']
    reasoning = ai_prediction['reasoning']
    
    # 추세 이모지 매핑
    trend_emoji = {
        'bullish': '🟢',
        'bearish': '🔴',
        'neutral': '⚪'
    }
    
    trend_text = {
        'bullish': '상승 (Bullish)',
        'bearish': '하락 (Bearish)',
        'neutral': '보합 (Neutral)'
    }
    
    # 예측 결과 표시
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📊 단기 추세 예측")
        st.markdown(f"**🔹 예측**: {trend_emoji[trend]} {trend_text[trend]}")
        st.markdown(f"**🔹 신뢰도**: {confidence}%")
    
    with col2:
        st.markdown("#### 📝 예측 근거")
        for key, value in reasoning.items():
            label = {
                'ema_analysis': '이동평균',
                'rsi_analysis': 'RSI',
                'macd_analysis': 'MACD',
                'volume_analysis': '거래량'
            }.get(key, key)
            st.markdown(f"• **{label}**: {value}")
    
    # 면책 조항
    st.info("⚠️ **면책 조항**: 이 예측은 과거 데이터 기반 분석이며, 실제 시장 결과를 보장하지 않습니다.")


def render_position_recommendation(position_rec):
    """
    [추가됨] v2.2.0: 포지션 추천 UI 렌더링
    
    Args:
        position_rec: recommend_position() 결과
    """
    position = position_rec['position']
    probability = position_rec['probability']
    reason = position_rec['reason']
    
    st.markdown("#### 📍 포지션 추천")
    
    # 포지션별 스타일
    if position == 'long':
        position_text = "**롱 포지션**"
        emoji = "📈"
        color = "green"
    elif position == 'short':
        position_text = "**숏 포지션**"
        emoji = "📉"
        color = "red"
    else:
        position_text = "**관망(보류)**"
        emoji = "⏸️"
        color = "gray"
    
    # 추천 표시
    if position == 'hold':
        st.warning(f"{emoji} 현재 데이터 기준, **{position_text}** 상태입니다.")
    else:
        st.success(f"{emoji} 현재 데이터 기준, **{position_text}**이 우세(약 **{probability}%**)로 판단됩니다.")
    
    # 추천 이유
    st.markdown(f"**💡 추천 이유**")
    st.markdown(f"• {reason}")
    
    # 주의사항
    st.caption("⚠️ **주의사항**")
    st.caption("• 이 추천은 확률적 분석 결과입니다")
    st.caption("• 시장 상황에 따라 실시간 변동 가능합니다")
    st.caption("• 반드시 손절매 설정 후 진입하세요")


# ==================== 캔들스틱 패턴 분석 ====================
def detect_candlestick_patterns(data):
    """
    61개 캔들스틱 패턴 감지 (TA-Lib 기반)
    
    Args:
        data: OHLCV 데이터프레임
    
    Returns:
        list: [{'name': str, 'signal': int, 'confidence': int, 'category': str}]
    """
    if not TALIB_AVAILABLE:
        # TA-Lib 미설치 시 기본 3개 패턴만 사용
        return detect_basic_patterns(data)
    
    patterns = []
    
    # 61개 TA-Lib 패턴 함수 목록
    pattern_functions = {
        # 반전 패턴 (Reversal)
        'CDL2CROWS': ('Two Crows', 'reversal'),
        'CDL3BLACKCROWS': ('Three Black Crows', 'reversal'),
        'CDL3INSIDE': ('Three Inside Up/Down', 'reversal'),
        'CDL3LINESTRIKE': ('Three Line Strike', 'reversal'),
        'CDL3OUTSIDE': ('Three Outside Up/Down', 'reversal'),
        'CDL3STARSINSOUTH': ('Three Stars In The South', 'reversal'),
        'CDL3WHITESOLDIERS': ('Three Advancing White Soldiers', 'reversal'),
        'CDLABANDONEDBABY': ('Abandoned Baby', 'reversal'),
        'CDLDARKCLOUDCOVER': ('Dark Cloud Cover', 'reversal'),
        'CDLDOJI': ('Doji', 'reversal'),
        'CDLDOJISTAR': ('Doji Star', 'reversal'),
        'CDLDRAGONFLYDOJI': ('Dragonfly Doji', 'reversal'),
        'CDLENGULFING': ('Engulfing Pattern', 'reversal'),
        'CDLEVENINGDOJISTAR': ('Evening Doji Star', 'reversal'),
        'CDLEVENINGSTAR': ('Evening Star', 'reversal'),
        'CDLGRAVESTONEDOJI': ('Gravestone Doji', 'reversal'),
        'CDLHAMMER': ('Hammer', 'reversal'),
        'CDLHANGINGMAN': ('Hanging Man', 'reversal'),
        'CDLHARAMI': ('Harami Pattern', 'reversal'),
        'CDLHARAMICROSS': ('Harami Cross Pattern', 'reversal'),
        'CDLINVERTEDHAMMER': ('Inverted Hammer', 'reversal'),
        'CDLKICKING': ('Kicking', 'reversal'),
        'CDLKICKINGBYLENGTH': ('Kicking (by length)', 'reversal'),
        'CDLLADDERBOTTOM': ('Ladder Bottom', 'reversal'),
        'CDLMORNINGDOJISTAR': ('Morning Doji Star', 'reversal'),
        'CDLMORNINGSTAR': ('Morning Star', 'reversal'),
        'CDLPIERCING': ('Piercing Pattern', 'reversal'),
        'CDLSHOOTINGSTAR': ('Shooting Star', 'reversal'),
        'CDLSTICKSANDWICH': ('Stick Sandwich', 'reversal'),
        'CDLTAKURI': ('Takuri (Dragonfly Doji with very long lower shadow)', 'reversal'),
        'CDLTASUKIGAP': ('Tasuki Gap', 'reversal'),
        'CDLUNIQUE3RIVER': ('Unique 3 River', 'reversal'),
        
        # 지속 패턴 (Continuation)
        'CDLADVANCEBLOCK': ('Advance Block', 'continuation'),
        'CDLBREAKAWAY': ('Breakaway', 'continuation'),
        'CDLCLOSINGMARUBOZU': ('Closing Marubozu', 'continuation'),
        'CDLCONCEALBABYSWALL': ('Concealing Baby Swallow', 'continuation'),
        'CDLCOUNTERATTACK': ('Counterattack', 'continuation'),
        'CDLGAPSIDESIDEWHITE': ('Up/Down-gap side-by-side white lines', 'continuation'),
        'CDLHIGHWAVE': ('High-Wave Candle', 'continuation'),
        'CDLHIKKAKE': ('Hikkake Pattern', 'continuation'),
        'CDLHIKKAKEMOD': ('Modified Hikkake Pattern', 'continuation'),
        'CDLHOMINGPIGEON': ('Homing Pigeon', 'continuation'),
        'CDLIDENTICAL3CROWS': ('Identical Three Crows', 'continuation'),
        'CDLINNECK': ('In-Neck Pattern', 'continuation'),
        'CDLLONGLEGGEDDOJI': ('Long Legged Doji', 'continuation'),
        'CDLLONGLINE': ('Long Line Candle', 'continuation'),
        'CDLMARUBOZU': ('Marubozu', 'continuation'),
        'CDLMATCHINGLOW': ('Matching Low', 'continuation'),
        'CDLMATHOLD': ('Mat Hold', 'continuation'),
        'CDLONNECK': ('On-Neck Pattern', 'continuation'),
        'CDLRISEFALL3METHODS': ('Rising/Falling Three Methods', 'continuation'),
        'CDLSEPARATINGLINES': ('Separating Lines', 'continuation'),
        'CDLSHORTLINE': ('Short Line Candle', 'continuation'),
        'CDLSPINNINGTOP': ('Spinning Top', 'continuation'),
        'CDLSTALLEDPATTERN': ('Stalled Pattern', 'continuation'),
        'CDLTHRUSTING': ('Thrusting Pattern', 'continuation'),
        'CDLTRISTAR': ('Tristar Pattern', 'continuation'),
        'CDLUPSIDEGAP2CROWS': ('Upside Gap Two Crows', 'continuation'),
        'CDLXSIDEGAP3METHODS': ('Upside/Downside Gap Three Methods', 'continuation'),
        
        # 중립 패턴 (Neutral)
        'CDLBELTHOLD': ('Belt-hold', 'neutral'),
        'CDLRISEFALL3METHODS': ('Rising/Falling Three Methods', 'neutral'),
        'CDLRICKSHAWMAN': ('Rickshaw Man', 'neutral')
    }
    
    open_prices = data['Open'].values
    high_prices = data['High'].values
    low_prices = data['Low'].values
    close_prices = data['Close'].values
    
    for func_name, (pattern_name, category) in pattern_functions.items():
        try:
            pattern_func = getattr(talib, func_name)
            result = pattern_func(open_prices, high_prices, low_prices, close_prices)
            
            # 최근 3개 데이터에서 패턴 확인
            recent_signals = result[-3:]
            if np.any(recent_signals != 0):
                signal_value = int(recent_signals[recent_signals != 0][-1])
                confidence = abs(signal_value)
                
                patterns.append({
                    'name': pattern_name,
                    'signal': signal_value,
                    'confidence': confidence,
                    'category': category
                })
        except Exception:
            continue
    
    return patterns


def detect_basic_patterns(data):
    """
    기본 3개 패턴 감지 (TA-Lib 미설치 시)
    
    Args:
        data: OHLCV 데이터프레임
    
    Returns:
        list: [{'name': str, 'signal': int, 'confidence': int, 'category': str}]
    """
    if len(data) < 3:
        return []
    
    patterns = []
    latest = data.iloc[-1]
    prev1 = data.iloc[-2]
    prev2 = data.iloc[-3] if len(data) >= 3 else prev1
    
    # 1. Hammer (망치형)
    body = abs(latest['Close'] - latest['Open'])
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
    
    if lower_shadow > body * 2 and upper_shadow < body * 0.3:
        patterns.append({
            'name': 'Hammer',
            'signal': 100,
            'confidence': 80,
            'category': 'reversal'
        })
    
    # 2. Engulfing (포용형)
    if prev1['Close'] < prev1['Open'] and latest['Close'] > latest['Open']:
        if latest['Open'] < prev1['Close'] and latest['Close'] > prev1['Open']:
            patterns.append({
                'name': 'Bullish Engulfing',
                'signal': 100,
                'confidence': 90,
                'category': 'reversal'
            })
    
    # 3. Doji (도지)
    if abs(latest['Close'] - latest['Open']) / (latest['High'] - latest['Low']) < 0.1:
        patterns.append({
            'name': 'Doji',
            'signal': 0,
            'confidence': 70,
            'category': 'reversal'
        })
    
    return patterns


# ==================== 매도 시점 예측 ====================
def predict_sell_points(data, entry_price=None):
    """
    3가지 시나리오 기반 매도 시점 예측
    
    Args:
        data: 지표가 계산된 데이터프레임
        entry_price: 진입 가격 (None이면 현재가 사용)
    
    Returns:
        dict: {
            'conservative': {...},
            'neutral': {...},
            'aggressive': {...}
        }
    """
    if data.empty or 'ATR' not in data.columns:
        return {}
    
    latest = data.iloc[-1]
    current_price = latest['Close']
    atr = latest['ATR']
    
    if entry_price is None:
        entry_price = current_price
    
    scenarios = {}
    
    # 1. 보수적 시나리오 (단기 수익 확보)
    scenarios['conservative'] = {
        'name': '보수적 전략',
        'description': '빠른 수익 확보 (단기)',
        'take_profit': entry_price + (atr * 1.5),
        'stop_loss': entry_price - (atr * 1.0),
        'trailing_stop': atr * 0.5,
        'risk_reward_ratio': 1.5,
        'holding_period': '1-3일'
    }
    
    # 2. 중립적 시나리오 (균형)
    scenarios['neutral'] = {
        'name': '중립적 전략',
        'description': '리스크/수익 균형',
        'take_profit': entry_price + (atr * 2.5),
        'stop_loss': entry_price - (atr * 1.5),
        'trailing_stop': atr * 1.0,
        'risk_reward_ratio': 1.67,
        'holding_period': '3-7일'
    }
    
    # 3. 공격적 시나리오 (장기 수익)
    scenarios['aggressive'] = {
        'name': '공격적 전략',
        'description': '큰 수익 추구 (장기)',
        'take_profit': entry_price + (atr * 4.0),
        'stop_loss': entry_price - (atr * 2.0),
        'trailing_stop': atr * 1.5,
        'risk_reward_ratio': 2.0,
        'holding_period': '1-2주'
    }
    
    # 현재 상태 평가
    for scenario_name, scenario in scenarios.items():
        profit_pct = ((scenario['take_profit'] - entry_price) / entry_price) * 100
        loss_pct = ((entry_price - scenario['stop_loss']) / entry_price) * 100
        
        scenario['profit_pct'] = round(profit_pct, 2)
        scenario['loss_pct'] = round(loss_pct, 2)
        
        # 현재가 기준 상태
        if current_price >= scenario['take_profit']:
            scenario['status'] = '목표 달성'
            scenario['status_emoji'] = '✅'
        elif current_price <= scenario['stop_loss']:
            scenario['status'] = '손절 필요'
            scenario['status_emoji'] = '⛔'
        else:
            progress = (current_price - entry_price) / (scenario['take_profit'] - entry_price)
            scenario['status'] = f'진행중 ({progress*100:.1f}%)'
            scenario['status_emoji'] = '🔄'
    
    return scenarios


# ==================== 차트 생성 ====================
def create_candlestick_chart(data, symbol):
    """
    캔들스틱 차트 생성 (기술적 지표 포함)
    
    Args:
        data: 지표가 계산된 데이터프레임
        symbol: 암호화폐 심볼
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # EMA
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA50'],
        name='EMA50',
        line=dict(color='blue', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA200'],
        name='EMA200',
        line=dict(color='red', width=1.5)
    ))
    
    # 볼린저 밴드
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_Upper'],
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_Lower'],
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty',
        opacity=0.3
    ))
    
    # 레이아웃
    fig.update_layout(
        title=f'{symbol} 가격 차트',
        yaxis_title='가격 (USD)',
        xaxis_title='날짜',
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig


def create_volume_chart(data):
    """
    거래량 차트 생성
    
    Args:
        data: OHLCV 데이터프레임
    
    Returns:
        plotly.graph_objects.Figure
    """
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
              for i in range(len(data))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume_MA'],
        name='Volume MA',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title='거래량',
        yaxis_title='거래량',
        xaxis_title='날짜',
        template='plotly_dark',
        height=300,
        hovermode='x unified'
    )
    
    return fig


def create_rsi_chart(data):
    """
    RSI 차트 생성
    
    Args:
        data: 지표가 계산된 데이터프레임
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # 과매수/과매도 선
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도")
    
    fig.update_layout(
        title='RSI (Relative Strength Index)',
        yaxis_title='RSI',
        xaxis_title='날짜',
        template='plotly_dark',
        height=300,
        hovermode='x unified'
    )
    
    return fig


def create_macd_chart(data):
    """
    MACD 차트 생성
    
    Args:
        data: 지표가 계산된 데이터프레임
    
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MACD'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Signal'],
        name='Signal',
        line=dict(color='red', width=2)
    ))
    
    # 히스토그램
    colors = ['green' if val > 0 else 'red' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['MACD_Hist'],
        name='Histogram',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        yaxis_title='MACD',
        xaxis_title='날짜',
        template='plotly_dark',
        height=300,
        hovermode='x unified'
    )
    
    return fig


# ==================== 매매 전략 렌더링 ====================
def render_trading_strategy(data, ema_short, ema_long, selected_resolution, rsi, macd, signal):
    """
    매매 전략 섹션 렌더링
    
    Args:
        data: 지표가 계산된 데이터프레임
        ema_short: 단기 EMA 값
        ema_long: 장기 EMA 값
        selected_resolution: 선택된 분해능
        rsi: RSI 값
        macd: MACD 값
        signal: Signal 값
    """
    st.markdown("### 🎯 매매 전략")
    
    # 현재가 정보
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else latest_price
    price_change = ((latest_price - prev_price) / prev_price) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="현재가",
            value=f"${latest_price:,.2f}",
            delta=f"{price_change:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="RSI",
            value=f"{rsi:.1f}" if rsi is not None else "N/A"
        )
    
    with col3:
        st.metric(
            label="MACD",
            value=f"{macd:.2f}" if macd is not None else "N/A"
        )
    
    # RSI 기반 전략
    st.markdown("#### 📊 RSI 분석")
    if rsi is not None:
        if rsi > 70:
            st.warning("⚠️ **과매수 구간**: 매도를 고려할 시점입니다.")
        elif rsi < 30:
            st.success("✅ **과매도 구간**: 매수를 고려할 시점입니다.")
        else:
            st.info("ℹ️ **중립 구간**: 추가 신호를 기다리는 것이 좋습니다.")
    
    # MACD 기반 전략
    st.markdown("#### 📈 MACD 분석")
    if macd is not None and signal is not None:
        if macd > signal:
            st.success("✅ **상승 신호**: MACD가 시그널선 위에 있습니다.")
        else:
            st.warning("⚠️ **하락 신호**: MACD가 시그널선 아래에 있습니다.")
    
    # 이동평균 기반 전략
    st.markdown("#### 🔄 이동평균 분석")
    if ema_short is not None and ema_long is not None:
        if ema_short > ema_long:
            st.success("✅ **골든크로스**: 상승 추세가 예상됩니다.")
        else:
            st.warning("⚠️ **데드크로스**: 하락 추세가 예상됩니다.")


# ==================== 레버리지 최적화 ====================
def optimize_leverage(investment_amount, volatility, selected_resolution):
    """
    투자 금액, 변동성, 분해능 기반 레버리지 최적화
    
    [수정됨] v2.1.1: 투자 금액 팩터 반전 (소액 → 보수적, 대량 → 공격적)
    
    Args:
        investment_amount: 투자 금액 (USD)
        volatility: ATR 기반 변동성
        selected_resolution: 선택된 분해능
    
    Returns:
        dict: {
            'recommended_leverage': float,
            'max_leverage': float,
            'risk_level': str,
            'explanation': str
        }
    """
    # 1. 투자 금액 팩터 (수정됨)
    if investment_amount >= 10000:
        amount_factor = 1.2  # 대량 투자 → 여유
    elif investment_amount >= 5000:
        amount_factor = 1.0  # 기준
    elif investment_amount >= 1000:
        amount_factor = 0.8  # 신중
    else:
        amount_factor = 0.6  # 소액 투자 → 보수적
    
    # 2. 변동성 팩터
    if volatility > 0.05:
        volatility_factor = 0.7  # 높은 변동성 → 낮은 레버리지
    elif volatility > 0.03:
        volatility_factor = 1.0  # 중간 변동성
    else:
        volatility_factor = 1.3  # 낮은 변동성 → 높은 레버리지
    
    # 3. 분해능 팩터
    resolution_factors = {
        '1m': 0.5,
        '5m': 0.6,
        '15m': 0.7,
        '30m': 0.8,
        '1h': 1.0,
        '4h': 1.2,
        '1d': 1.5
    }
    resolution_factor = resolution_factors.get(selected_resolution, 1.0)
    
    # 최종 레버리지 계산
    base_leverage = 3.0
    recommended_leverage = base_leverage * amount_factor * volatility_factor * resolution_factor
    recommended_leverage = max(1.0, min(recommended_leverage, 10.0))  # 1-10배 제한
    
    # 최대 레버리지 (권장의 1.5배)
    max_leverage = min(recommended_leverage * 1.5, 15.0)
    
    # 리스크 레벨
    if recommended_leverage <= 2:
        risk_level = "낮음"
    elif recommended_leverage <= 5:
        risk_level = "중간"
    else:
        risk_level = "높음"
    
    # 설명 생성
    explanation = f"""
    **투자 금액**: ${investment_amount:,.0f} → 팩터 {amount_factor:.1f}x
    **변동성**: {volatility:.4f} → 팩터 {volatility_factor:.1f}x
    **분해능**: {selected_resolution} → 팩터 {resolution_factor:.1f}x
    
    **권장 레버리지**: {recommended_leverage:.1f}배
    **최대 레버리지**: {max_leverage:.1f}배
    **리스크 레벨**: {risk_level}
    """
    
    return {
        'recommended_leverage': round(recommended_leverage, 1),
        'max_leverage': round(max_leverage, 1),
        'risk_level': risk_level,
        'explanation': explanation
    }


# ==================== 메인 애플리케이션 ====================
def main():
    """메인 애플리케이션"""
    
    # Keep-Alive 서버 시작
    start_keep_alive()
    
    st.title("🤖 AI 기반 암호화폐 투자 전략 분석 v2.2.1")
    st.markdown("**긴급 버그 수정**: ema_short, ema_long 변수 오류 해결")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    
    # 코인 선택
    st.sidebar.subheader("1️⃣ 코인 선택")
    
    default_coins = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Ripple': 'XRP-USD',
        'Cardano': 'ADA-USD',
        'Solana': 'SOL-USD'
    }
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        coin_choice = st.selectbox(
            "코인 선택",
            list(default_coins.keys()),
            key='coin_select'
        )
    with col2:
        use_custom = st.checkbox("직접 입력", key='use_custom')
    
    if use_custom:
        custom_coin = st.sidebar.text_input(
            "심볼 입력 (예: DOGE)",
            value="",
            key='custom_coin'
        ).strip().upper()
        
        if custom_coin:
            if not custom_coin.endswith('-USD') and not custom_coin.endswith('USDT'):
                symbol = f"{custom_coin}-USD"
            else:
                symbol = custom_coin
        else:
            symbol = default_coins[coin_choice]
    else:
        symbol = default_coins[coin_choice]
    
    st.sidebar.info(f"📊 선택된 심볼: **{symbol}**")
    
    # 분해능 선택
    st.sidebar.subheader("2️⃣ 데이터 설정")
    resolution_options = {
        '1분': '1m',
        '5분': '5m',
        '15분': '15m',
        '30분': '30m',
        '1시간': '1h',
        '4시간': '4h',
        '1일': '1d'
    }
    
    selected_resolution_name = st.sidebar.selectbox(
        "분해능",
        list(resolution_options.keys()),
        index=4  # 기본값: 1시간
    )
    selected_resolution = resolution_options[selected_resolution_name]
    
    # 기간 선택
    days = st.sidebar.slider(
        "조회 기간 (일)",
        min_value=7,
        max_value=365,
        value=90,
        step=7
    )
    
    # 투자 금액 입력
    st.sidebar.subheader("3️⃣ 투자 설정")
    investment_amount = st.sidebar.number_input(
        "투자 금액 (USD)",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=100
    )
    
    # 고급 옵션
    with st.sidebar.expander("🔧 고급 설정"):
        show_patterns = st.checkbox("캔들스틱 패턴 분석", value=True)
        show_sell_strategy = st.checkbox("매도 전략 표시", value=True)
        show_leverage = st.checkbox("레버리지 최적화", value=True)
    
    # 데이터 로드 버튼
    if st.sidebar.button("🚀 분석 시작", type="primary"):
        with st.spinner(f"{symbol} 데이터를 로딩 중..."):
            data = load_crypto_data(symbol, selected_resolution, days)
        
        if data.empty:
            st.error("❌ 데이터를 가져올 수 없습니다. 심볼을 확인해주세요.")
            return
        
        with st.spinner("기술적 지표를 계산 중..."):
            data = calculate_indicators(data)
        
        if data.empty:
            st.error("❌ 지표 계산에 실패했습니다.")
            return
        
        # 데이터 분석 결과
        st.markdown("---")
        st.markdown("### 📊 데이터 분석 결과")
        
        col1, col2, col3, col4 = st.columns(4)
        
        latest = data.iloc[-1]
        
        with col1:
            st.metric("데이터 개수", f"{len(data):,}개")
        with col2:
            st.metric("시작일", data.index[0].strftime('%Y-%m-%d'))
        with col3:
            st.metric("종료일", data.index[-1].strftime('%Y-%m-%d'))
        with col4:
            volatility = latest['ATR'] / latest['Close'] if latest['Close'] > 0 else 0
            st.metric("변동성 (ATR%)", f"{volatility*100:.2f}%")
        
        # [수정됨] v2.2.1: EMA 변수 명시적 추출
        ema_short = data['EMA50'].iloc[-1] if 'EMA50' in data.columns and len(data) > 0 else None
        ema_long = data['EMA200'].iloc[-1] if 'EMA200' in data.columns and len(data) > 0 else None
        
        # [추가됨] v2.2.0: AI 예측 실행
        ai_prediction = predict_trend_with_ai(data)
        
        # [추가됨] v2.2.0: AI 예측 결과 표시
        st.markdown("---")
        render_ai_prediction(ai_prediction)
        
        # 📈 예측 요약 (기존 섹션)
        st.markdown("---")
        st.markdown("### 📈 예측 요약")
        
        current_price = latest['Close']
        rsi = latest['RSI']
        macd = latest['MACD']
        signal_line = latest['Signal']
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("#### 💰 가격 정보")
            st.markdown(f"**현재가**: ${current_price:,.2f}")
            st.markdown(f"**EMA50**: ${ema_short:,.2f}" if ema_short else "**EMA50**: N/A")
            st.markdown(f"**EMA200**: ${ema_long:,.2f}" if ema_long else "**EMA200**: N/A")
        
        with summary_col2:
            st.markdown("#### 📊 지표 상태")
            st.markdown(f"**RSI**: {rsi:.1f}")
            st.markdown(f"**MACD**: {macd:.2f}")
            st.markdown(f"**Signal**: {signal_line:.2f}")
        
        # 매매 전략 표시
        st.markdown("---")
        render_trading_strategy(
            data=data,
            ema_short=ema_short,  # ✅ 이제 정의됨
            ema_long=ema_long,    # ✅ 이제 정의됨
            selected_resolution=selected_resolution,
            rsi=rsi,
            macd=macd,
            signal=signal_line
        )
        
        # [추가됨] v2.2.0: 포지션 추천 표시
        position_rec = recommend_position(ai_prediction, data)
        render_position_recommendation(position_rec)
        
        # 차트 표시
        st.markdown("---")
        st.markdown("### 📈 차트")
        
        tab1, tab2, tab3, tab4 = st.tabs(["💹 캔들스틱", "📊 거래량", "🔵 RSI", "📉 MACD"])
        
        with tab1:
            fig = create_candlestick_chart(data, symbol)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_volume_chart(data)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = create_rsi_chart(data)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            fig = create_macd_chart(data)
            st.plotly_chart(fig, use_container_width=True)
        
        # 캔들스틱 패턴 분석
        if show_patterns:
            st.markdown("---")
            st.markdown("### 🎨 캔들스틱 패턴 분석")
            
            with st.spinner("패턴을 분석 중..."):
                patterns = detect_candlestick_patterns(data)
            
            if patterns:
                # 중복 제거
                unique_patterns = {}
                for p in patterns:
                    if p['name'] not in unique_patterns:
                        unique_patterns[p['name']] = p
                
                patterns = list(unique_patterns.values())
                
                # 카테고리별 분류
                reversal_patterns = [p for p in patterns if p['category'] == 'reversal']
                continuation_patterns = [p for p in patterns if p['category'] == 'continuation']
                neutral_patterns = [p for p in patterns if p['category'] == 'neutral']
                
                # 2열 레이아웃
                col1, col2 = st.columns(2)
                
                with col1:
                    if reversal_patterns:
                        st.markdown("#### 🔄 반전 패턴")
                        for p in reversal_patterns[:5]:  # 상위 5개
                            signal_text = "🟢 상승" if p['signal'] > 0 else "🔴 하락"
                            st.markdown(f"**{p['name']}** - {signal_text} (신뢰도: {p['confidence']}%)")
                
                with col2:
                    if continuation_patterns:
                        st.markdown("#### ➡️ 지속 패턴")
                        for p in continuation_patterns[:5]:
                            signal_text = "🟢 상승" if p['signal'] > 0 else "🔴 하락"
                            st.markdown(f"**{p['name']}** - {signal_text} (신뢰도: {p['confidence']}%)")
                
                if neutral_patterns:
                    st.markdown("#### ⚪ 중립 패턴")
                    for p in neutral_patterns:
                        st.markdown(f"**{p['name']}** (신뢰도: {p['confidence']}%)")
            else:
                st.info("ℹ️ 감지된 패턴이 없습니다.")
        
        # 매도 전략
        if show_sell_strategy:
            st.markdown("---")
            st.markdown("### 💰 매도 시점 예측")
            
            entry_price = st.number_input(
                "진입 가격 (USD)",
                min_value=0.01,
                value=float(current_price),
                step=0.01,
                key='entry_price'
            )
            
            scenarios = predict_sell_points(data, entry_price)
            
            if scenarios:
                tab1, tab2, tab3 = st.tabs(["🐢 보수적", "⚖️ 중립적", "🚀 공격적"])
                
                for tab, scenario_name in zip([tab1, tab2, tab3], ['conservative', 'neutral', 'aggressive']):
                    scenario = scenarios[scenario_name]
                    
                    with tab:
                        st.markdown(f"#### {scenario['name']}")
                        st.markdown(f"**설명**: {scenario['description']}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("목표가", f"${scenario['take_profit']:,.2f}", 
                                     f"+{scenario['profit_pct']}%")
                            st.metric("손절가", f"${scenario['stop_loss']:,.2f}", 
                                     f"-{scenario['loss_pct']}%")
                        
                        with col2:
                            st.metric("추적 손절", f"${scenario['trailing_stop']:,.2f}")
                            st.metric("리스크/수익 비율", f"{scenario['risk_reward_ratio']:.2f}")
                        
                        st.markdown(f"**보유 기간**: {scenario['holding_period']}")
                        st.markdown(f"**현재 상태**: {scenario['status_emoji']} {scenario['status']}")
        
        # 레버리지 최적화
        if show_leverage:
            st.markdown("---")
            st.markdown("### ⚙️ 레버리지 최적화")
            
            volatility = latest['ATR'] / latest['Close'] if latest['Close'] > 0 else 0
            leverage_result = optimize_leverage(investment_amount, volatility, selected_resolution)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("권장 레버리지", f"{leverage_result['recommended_leverage']}배")
            with col2:
                st.metric("최대 레버리지", f"{leverage_result['max_leverage']}배")
            with col3:
                st.metric("리스크 레벨", leverage_result['risk_level'])
            
            with st.expander("📊 상세 설명"):
                st.markdown(leverage_result['explanation'])
    
    # 사용 가이드
    with st.sidebar.expander("📖 사용 가이드"):
        st.markdown("""
        ### 사용 방법
        1. 코인 선택 또는 직접 입력
        2. 분해능 및 기간 설정
        3. 투자 금액 입력
        4. "🚀 분석 시작" 버튼 클릭
        
        ### 주요 기능
        - 🤖 AI 예측 결과 (v2.2.0)
        - 📍 포지션 추천 (v2.2.0)
        - 🎨 61개 캔들스틱 패턴
        - 💰 3가지 매도 전략
        - ⚙️ 레버리지 최적화
        - 📊 실시간 차트
        
        ### 버전 정보
        - v2.2.1: ema_short 버그 수정
        - v2.2.0: AI 예측 + 포지션 추천
        - v2.1.2: Keep-Alive 추가
        - v2.1.1: 레버리지 로직 수정
        """)


if __name__ == "__main__":
    main()
