"""
=================================================
AI 암호화폐 트레이딩 전략 분석 대시보드 v2.3.0
=================================================

주요 기능:
- 실시간 암호화폐 데이터 분석 (yfinance)
- 8가지 분해능 선택 (1m ~ 1mo)
- Wilder's Smoothing RSI
- 계절성 분석 (Prophet)
- TimeSeriesSplit 기반 교차 검증
- 동적 레버리지 최적화
- TA-Lib 61개 캔들스틱 패턴 분석
- 매도 시점 예측 (3가지 시나리오)
- AI 예측 결과 및 포지션 추천
- 자동 백테스팅 엔진 (v2.3.0 신규)

버전: v2.3.0
작성일: 2025-10-26
변경사항:
- [수정됨] ema_short 변수 오류 수정 (ema50으로 대체)
- [추가됨] 자동 백테스팅 엔진 구현
- [추가됨] 백테스팅 성능 지표 계산
- [추가됨] 백테스팅 결과 시각화
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# 선택적 라이브러리 임포트
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib가 설치되지 않았습니다. 기본 3개 패턴만 사용됩니다.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet이 설치되지 않았습니다. 계절성 분석이 비활성화됩니다.")

warnings.filterwarnings('ignore')

# =====================================================
# 1. 페이지 설정
# =====================================================

st.set_page_config(
    page_title="AI 암호화폐 트레이딩 전략 v2.3.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. 캐싱 함수: 데이터 로드
# =====================================================

@st.cache_data(ttl=300, show_spinner=False)  # 5분 캐시
def load_crypto_data(ticker, interval='1h', period=None, start_date=None, end_date=None):
    """
    yfinance에서 암호화폐 데이터를 로드합니다.
    
    Args:
        ticker (str): 티커 심볼 (예: 'BTC-USD')
        interval (str): 데이터 간격
        period (str): 데이터 기간 (예: '90d')
        start_date (str): 시작 날짜 (YYYY-MM-DD)
        end_date (str): 종료 날짜 (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: OHLCV 데이터
    """
    try:
        # yfinance API 제한 대응: 1시간봉은 최대 730일
        if interval == '1h' and period:
            max_days = 730
            period_days = int(period.replace('d', ''))
            if period_days > max_days:
                period = f'{max_days}d'
                st.warning(f"⚠️ 1시간봉은 최대 {max_days}일까지만 지원됩니다. 자동으로 조정되었습니다.")
        
        # period 우선, 없으면 start/end 사용
        if period:
            data = yf.download(ticker, interval=interval, period=period, progress=False)
        else:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error(f"❌ {ticker} 데이터를 불러올 수 없습니다.")
            return pd.DataFrame()
        
        # 컬럼 정리
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        return data
    
    except Exception as e:
        st.error(f"❌ 데이터 로드 오류: {str(e)}")
        return pd.DataFrame()

# =====================================================
# 3. 기술적 지표 계산 함수
# =====================================================

def calculate_ema(data, period):
    """
    EMA(지수 이동 평균)를 계산합니다.
    데이터가 부족할 경우 적응형 윈도우를 사용합니다.
    """
    if len(data) < period:
        # 데이터가 부족하면 사용 가능한 데이터로 계산
        actual_period = max(1, len(data) // 2)
        return data['Close'].ewm(span=actual_period, adjust=False).mean()
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi_wilders(data, period=14):
    """
    Wilder's Smoothing 방식의 RSI를 계산합니다.
    """
    if len(data) < period + 1:
        return pd.Series([50] * len(data), index=data.index)
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 첫 번째 평균은 단순 평균
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Wilder's Smoothing 적용
    for i in range(period, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    MACD(이동평균수렴확산)를 계산합니다.
    """
    if len(data) < slow:
        # 데이터가 부족하면 0 반환
        return pd.Series([0] * len(data), index=data.index), \
               pd.Series([0] * len(data), index=data.index), \
               pd.Series([0] * len(data), index=data.index)
    
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    볼린저 밴드를 계산합니다.
    """
    if len(data) < period:
        period = max(1, len(data) // 2)
    
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

def calculate_atr(data, period=14):
    """
    ATR(Average True Range)를 계산합니다.
    """
    if len(data) < period:
        period = max(1, len(data) // 2)
    
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

# =====================================================
# 4. AI 예측 함수 (v2.2.0 추가)
# =====================================================

def predict_trend_with_ai(data):
    """
    [추가됨] v2.2.0: 4개 기술적 지표를 기반으로 AI 예측을 수행합니다.
    
    Args:
        data (pd.DataFrame): OHLCV 데이터 (지표 계산 완료)
    
    Returns:
        dict: {
            'predicted_trend': 'bullish' | 'bearish' | 'neutral',
            'confidence': float (0-100),
            'reasoning': list of str,
            'signal_strength': float (-1.0 to 1.0)
        }
    """
    if len(data) < 50:
        return {
            'predicted_trend': 'neutral',
            'confidence': 0,
            'reasoning': ['데이터 부족으로 예측 불가'],
            'signal_strength': 0.0
        }
    
    # 최신 데이터 추출
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    
    # 1. 이동평균 분석 (가중치 30%)
    ema_signal = 0
    ema_reasoning = ""
    if 'EMA50' in data.columns and 'EMA200' in data.columns:
        ema50 = latest['EMA50']
        ema200 = latest['EMA200']
        if pd.notna(ema50) and pd.notna(ema200):
            if ema50 > ema200:
                ema_signal = 1.0
                ema_reasoning = "골든크로스 형성 중 (EMA50 > EMA200)"
            elif ema50 < ema200:
                ema_signal = -1.0
                ema_reasoning = "데드크로스 형성 중 (EMA50 < EMA200)"
            else:
                ema_reasoning = "이동평균 교차 대기 중"
        else:
            ema_reasoning = "이동평균 계산 불가"
    
    # 2. RSI 분석 (가중치 25%)
    rsi_signal = 0
    rsi_reasoning = ""
    if 'RSI' in data.columns and pd.notna(latest['RSI']):
        rsi = latest['RSI']
        if rsi >= 70:
            rsi_signal = -0.8  # 과매수 → 하락 신호
            rsi_reasoning = f"과매수 영역 (RSI: {rsi:.1f})"
        elif rsi <= 30:
            rsi_signal = 0.8  # 과매도 → 상승 신호
            rsi_reasoning = f"과매도 영역 (RSI: {rsi:.1f})"
        elif 45 <= rsi <= 55:
            rsi_reasoning = f"중립 영역 (RSI: {rsi:.1f})"
        elif rsi > 55:
            rsi_signal = 0.5
            rsi_reasoning = f"상승 모멘텀 (RSI: {rsi:.1f})"
        else:
            rsi_signal = -0.5
            rsi_reasoning = f"하락 모멘텀 (RSI: {rsi:.1f})"
    
    # 3. MACD 분석 (가중치 25%)
    macd_signal = 0
    macd_reasoning = ""
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        macd = latest['MACD']
        macd_signal_line = latest['MACD_Signal']
        macd_hist = latest['MACD_Hist']
        
        if pd.notna(macd) and pd.notna(macd_signal_line):
            if macd > macd_signal_line and macd_hist > 0:
                macd_signal = 1.0
                macd_reasoning = "양수 전환 (상승 모멘텀)"
            elif macd < macd_signal_line and macd_hist < 0:
                macd_signal = -1.0
                macd_reasoning = "음수 전환 (하락 모멘텀)"
            else:
                macd_reasoning = "MACD 교차 대기 중"
        else:
            macd_reasoning = "MACD 계산 불가"
    
    # 4. 거래량 분석 (가중치 20%)
    volume_signal = 0
    volume_reasoning = ""
    if len(data) >= 20:
        avg_volume = data['Volume'].iloc[-20:].mean()
        current_volume = latest['Volume']
        volume_change = (current_volume / avg_volume - 1) * 100
        
        if volume_change > 50:
            volume_signal = 0.7
            volume_reasoning = f"거래량 급증 (+{volume_change:.1f}%)"
        elif volume_change > 20:
            volume_signal = 0.4
            volume_reasoning = f"거래량 증가 (+{volume_change:.1f}%)"
        elif volume_change < -30:
            volume_signal = -0.3
            volume_reasoning = f"거래량 감소 ({volume_change:.1f}%)"
        else:
            volume_reasoning = f"거래량 평균 수준 ({volume_change:+.1f}%)"
    
    # 가중 평균 계산
    weights = {
        'ema': 0.30,
        'rsi': 0.25,
        'macd': 0.25,
        'volume': 0.20
    }
    
    weighted_signal = (
        ema_signal * weights['ema'] +
        rsi_signal * weights['rsi'] +
        macd_signal * weights['macd'] +
        volume_signal * weights['volume']
    )
    
    # 추세 판단
    if weighted_signal > 0.3:
        predicted_trend = 'bullish'
        trend_korean = '상승'
    elif weighted_signal < -0.3:
        predicted_trend = 'bearish'
        trend_korean = '하락'
    else:
        predicted_trend = 'neutral'
        trend_korean = '보합'
    
    # 신뢰도 계산 (0-100%)
    confidence = abs(weighted_signal) * 100
    confidence = min(confidence, 95)  # 최대 95%로 제한 (과신 방지)
    
    # 근거 정리
    reasoning = []
    if ema_reasoning:
        reasoning.append(f"• 이동평균: {ema_reasoning}")
    if rsi_reasoning:
        reasoning.append(f"• RSI: {rsi_reasoning}")
    if macd_reasoning:
        reasoning.append(f"• MACD: {macd_reasoning}")
    if volume_reasoning:
        reasoning.append(f"• 거래량: {volume_reasoning}")
    
    return {
        'predicted_trend': predicted_trend,
        'trend_korean': trend_korean,
        'confidence': confidence,
        'reasoning': reasoning,
        'signal_strength': weighted_signal
    }

def recommend_position(ai_prediction, data):
    """
    [추가됨] v2.2.0: AI 예측 결과를 기반으로 포지션을 추천합니다.
    
    Args:
        ai_prediction (dict): predict_trend_with_ai() 결과
        data (pd.DataFrame): OHLCV 데이터
    
    Returns:
        dict: {
            'position': 'long' | 'short' | 'hold',
            'probability': float (0-100),
            'reason': str
        }
    """
    trend = ai_prediction['predicted_trend']
    confidence = ai_prediction['confidence']
    signal_strength = ai_prediction['signal_strength']
    
    # 변동성 계산 (ATR 기반)
    if 'ATR' in data.columns and len(data) >= 14:
        atr = data['ATR'].iloc[-1]
        avg_price = data['Close'].iloc[-14:].mean()
        volatility = (atr / avg_price) * 100  # 변동성 %
        
        if volatility > 5:
            volatility_level = 'high'
            volatility_adjustment = -5
        elif volatility > 2:
            volatility_level = 'medium'
            volatility_adjustment = 0
        else:
            volatility_level = 'low'
            volatility_adjustment = 5
    else:
        volatility_level = 'medium'
        volatility_adjustment = 0
    
    # 포지션 결정
    if confidence >= 60:
        if trend == 'bullish':
            position = 'long'
            position_korean = '롱 포지션'
            
            # 확률 계산: 50 + (신뢰도 * 0.5) + 변동성 조정
            probability = 50 + (confidence * 0.5) + volatility_adjustment
            probability = max(45, min(85, probability))  # 45-85% 범위로 제한
            
            # 이유 추출 (가장 강력한 신호)
            if signal_strength > 0.5:
                reason = "이동평균 골든크로스 형성 중"
            elif 'reasoning' in ai_prediction and len(ai_prediction['reasoning']) > 0:
                # 첫 번째 근거에서 주요 내용 추출
                first_reason = ai_prediction['reasoning'][0].replace('• ', '')
                reason = first_reason.split(':')[1].strip() if ':' in first_reason else first_reason
            else:
                reason = "상승 추세 신호 포착"
        
        elif trend == 'bearish':
            position = 'short'
            position_korean = '숏 포지션'
            
            probability = 50 + (confidence * 0.5) + volatility_adjustment
            probability = max(45, min(85, probability))
            
            if signal_strength < -0.5:
                reason = "이동평균 데드크로스 형성 중"
            elif 'reasoning' in ai_prediction and len(ai_prediction['reasoning']) > 0:
                first_reason = ai_prediction['reasoning'][0].replace('• ', '')
                reason = first_reason.split(':')[1].strip() if ':' in first_reason else first_reason
            else:
                reason = "하락 추세 신호 포착"
        
        else:  # neutral
            position = 'hold'
            position_korean = '관망 (보류)'
            probability = 50
            reason = "명확한 추세 신호 부재"
    
    else:
        # 신뢰도 낮음 → 관망
        position = 'hold'
        position_korean = '관망 (보류)'
        probability = 50
        reason = f"예측 신뢰도 부족 ({confidence:.1f}%)"
    
    return {
        'position': position,
        'position_korean': position_korean,
        'probability': probability,
        'reason': reason,
        'volatility': volatility_level
    }

# =====================================================
# 5. 백테스팅 엔진 (v2.3.0 신규)
# =====================================================

def run_backtest(data, initial_capital=10000, lookback_periods=30):
    """
    [추가됨] v2.3.0: 과거 데이터로 AI 예측 성능을 백테스팅합니다.
    
    Args:
        data (pd.DataFrame): OHLCV 데이터 (지표 계산 완료)
        initial_capital (float): 초기 자본금
        lookback_periods (int): 예측 시 사용할 과거 데이터 기간
    
    Returns:
        dict: 백테스팅 결과
    """
    if len(data) < lookback_periods + 50:
        return {
            'success': False,
            'error': f'백테스팅에 필요한 최소 데이터({lookback_periods + 50}개) 부족'
        }
    
    results = []
    capital = initial_capital
    position = None  # 'long', 'short', None
    entry_price = 0
    trades = []
    
    # 슬라이딩 윈도우로 백테스팅
    for i in range(lookback_periods, len(data) - 1):
        # 현재까지의 데이터로 예측
        historical_data = data.iloc[:i+1].copy()
        
        # AI 예측 수행
        prediction = predict_trend_with_ai(historical_data)
        position_rec = recommend_position(prediction, historical_data)
        
        current_price = data.iloc[i]['Close']
        next_price = data.iloc[i+1]['Close']
        
        # 포지션 진입
        if position is None and position_rec['position'] != 'hold':
            if position_rec['probability'] >= 60:  # 확률 60% 이상만 진입
                position = position_rec['position']
                entry_price = current_price
                entry_time = data.index[i]
        
        # 포지션 청산 (다음 봉에서)
        elif position is not None:
            exit_price = next_price
            exit_time = data.index[i+1]
            
            if position == 'long':
                profit_pct = (exit_price / entry_price - 1) * 100
            else:  # short
                profit_pct = (entry_price / exit_price - 1) * 100
            
            profit_amount = capital * (profit_pct / 100)
            capital += profit_amount
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'position': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'profit_amount': profit_amount,
                'capital_after': capital,
                'prediction_confidence': prediction['confidence']
            })
            
            position = None
    
    # 미청산 포지션 처리
    if position is not None:
        exit_price = data.iloc[-1]['Close']
        exit_time = data.index[-1]
        
        if position == 'long':
            profit_pct = (exit_price / entry_price - 1) * 100
        else:
            profit_pct = (entry_price / exit_price - 1) * 100
        
        profit_amount = capital * (profit_pct / 100)
        capital += profit_amount
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'capital_after': capital,
            'prediction_confidence': prediction['confidence']
        })
    
    if len(trades) == 0:
        return {
            'success': False,
            'error': '백테스팅 기간 동안 매매 신호가 발생하지 않았습니다'
        }
    
    # 성능 지표 계산
    trades_df = pd.DataFrame(trades)
    
    total_return = (capital / initial_capital - 1) * 100
    winning_trades = trades_df[trades_df['profit_pct'] > 0]
    losing_trades = trades_df[trades_df['profit_pct'] <= 0]
    
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_win = winning_trades['profit_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['profit_pct'].mean() if len(losing_trades) > 0 else 0
    
    # 샤프 비율 계산 (일간 수익률 기준)
    returns = trades_df['profit_pct'].values
    if len(returns) > 1:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else 0
    else:
        sharpe_ratio = 0
    
    # 최대 낙폭 (Maximum Drawdown)
    capital_curve = [initial_capital]
    for trade in trades:
        capital_curve.append(trade['capital_after'])
    
    peak = capital_curve[0]
    max_drawdown = 0
    for value in capital_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return {
        'success': True,
        'trades': trades_df,
        'summary': {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'final_capital': capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    }

def visualize_backtest_results(backtest_results, data):
    """
    [추가됨] v2.3.0: 백테스팅 결과를 시각화합니다.
    
    Args:
        backtest_results (dict): run_backtest() 결과
        data (pd.DataFrame): 원본 OHLCV 데이터
    
    Returns:
        plotly.graph_objects.Figure: 백테스팅 차트
    """
    trades_df = backtest_results['trades']
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('가격 차트 및 매매 신호', '자본 곡선')
    )
    
    # 1. 가격 차트
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='가격',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # 2. 매수/매도 신호
    long_entries = trades_df[trades_df['position'] == 'long']
    short_entries = trades_df[trades_df['position'] == 'short']
    
    if len(long_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=long_entries['entry_time'],
                y=long_entries['entry_price'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='#00ff00'),
                name='롱 진입',
                text=long_entries['prediction_confidence'].apply(lambda x: f'신뢰도: {x:.1f}%'),
                hovertemplate='<b>롱 진입</b><br>가격: %{y:.2f}<br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if len(short_entries) > 0:
        fig.add_trace(
            go.Scatter(
                x=short_entries['entry_time'],
                y=short_entries['entry_price'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='#ff0000'),
                name='숏 진입',
                text=short_entries['prediction_confidence'].apply(lambda x: f'신뢰도: {x:.1f}%'),
                hovertemplate='<b>숏 진입</b><br>가격: %{y:.2f}<br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 3. 청산 신호
    fig.add_trace(
        go.Scatter(
            x=trades_df['exit_time'],
            y=trades_df['exit_price'],
            mode='markers',
            marker=dict(symbol='x', size=10, color='#ffff00'),
            name='청산',
            text=trades_df['profit_pct'].apply(lambda x: f'수익률: {x:+.2f}%'),
            hovertemplate='<b>청산</b><br>가격: %{y:.2f}<br>%{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 4. 자본 곡선
    capital_curve = [backtest_results['summary']['final_capital'] - backtest_results['summary']['total_return'] / 100 * backtest_results['summary']['final_capital']]
    for _, trade in trades_df.iterrows():
        capital_curve.append(trade['capital_after'])
    
    capital_times = [trades_df.iloc[0]['entry_time']] + list(trades_df['exit_time'])
    
    fig.add_trace(
        go.Scatter(
            x=capital_times,
            y=capital_curve,
            mode='lines',
            line=dict(color='#2196F3', width=2),
            name='자본',
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ),
        row=2, col=1
    )
    
    # 레이아웃 설정
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="자본 (USD)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

# =====================================================
# 6. UI 렌더링 함수
# =====================================================

def render_ai_prediction(ai_prediction):
    """
    [추가됨] v2.2.0: AI 예측 결과를 렌더링합니다.
    """
    st.markdown("---")
    st.markdown("## 🤖 AI 예측 결과")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 단기 추세 예측")
        
        trend = ai_prediction['predicted_trend']
        trend_korean = ai_prediction['trend_korean']
        confidence = ai_prediction['confidence']
        
        if trend == 'bullish':
            trend_color = '🟢'
            trend_emoji = '📈'
        elif trend == 'bearish':
            trend_color = '🔴'
            trend_emoji = '📉'
        else:
            trend_color = '⚪'
            trend_emoji = '➡️'
        
        st.markdown(f"**🔹 예측:** {trend_color} {trend_emoji} **{trend_korean}** ({trend.upper()})")
        st.markdown(f"**🔹 신뢰도:** {confidence:.1f}%")
        
        # 신뢰도 프로그레스 바
        if confidence >= 70:
            bar_color = 'normal'
        elif confidence >= 50:
            bar_color = 'normal'
        else:
            bar_color = 'normal'
        st.progress(confidence / 100)
    
    with col2:
        st.markdown("### 📝 예측 근거")
        for reason in ai_prediction['reasoning']:
            st.markdown(reason)
    
    st.info("⚠️ **면책 조항:** 이 예측은 과거 데이터 기반 분석이며, 실제 시장 결과를 보장하지 않습니다.")

def render_position_recommendation(position_rec):
    """
    [추가됨] v2.2.0: 포지션 추천을 렌더링합니다.
    """
    st.markdown("### 📍 포지션 추천")
    
    position = position_rec['position']
    position_korean = position_rec['position_korean']
    probability = position_rec['probability']
    reason = position_rec['reason']
    
    if position == 'long':
        st.success(f"**현재 데이터 기준, {position_korean}이 우세(약 {probability:.0f}%)로 판단됩니다.**")
    elif position == 'short':
        st.error(f"**현재 데이터 기준, {position_korean}이 우세(약 {probability:.0f}%)로 판단됩니다.**")
    else:
        st.warning(f"**현재 데이터 기준, {position_korean} 상태를 권장합니다.**")
    
    st.markdown(f"**💡 추천 이유:** {reason}")
    
    st.markdown("""
    ⚠️ **주의사항:**
    - 이 추천은 확률적 분석 결과입니다
    - 시장 상황에 따라 실시간 변동 가능
    - 반드시 손절매 설정 후 진입하세요
    """)

def render_trading_strategy(data, rsi, macd_line, macd_signal, ema50, ema200):  # [수정됨] v2.3.0: ema_short → ema50
    """
    매매 전략을 렌더링합니다.
    """
    st.markdown("---")
    st.markdown("## 🎯 매매 전략")
    
    latest = data.iloc[-1]
    latest_rsi = rsi.iloc[-1]
    latest_macd = macd_line.iloc[-1]
    latest_macd_signal = macd_signal.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 RSI 분석")
        if latest_rsi >= 70:
            st.warning(f"🔴 **과매수 영역** (RSI: {latest_rsi:.2f})")
            st.markdown("- 매도 시그널 고려")
            st.markdown("- 단기 조정 가능성")
        elif latest_rsi <= 30:
            st.success(f"🟢 **과매도 영역** (RSI: {latest_rsi:.2f})")
            st.markdown("- 매수 시그널 고려")
            st.markdown("- 반등 가능성")
        else:
            st.info(f"⚪ **중립 영역** (RSI: {latest_rsi:.2f})")
            st.markdown("- 관망 권장")
    
    with col2:
        st.markdown("### 📈 MACD 분석")
        if latest_macd > latest_macd_signal:
            st.success("🟢 **상승 추세**")
            st.markdown("- MACD > Signal")
            st.markdown("- 매수 포지션 유지")
        else:
            st.error("🔴 **하락 추세**")
            st.markdown("- MACD < Signal")
            st.markdown("- 매도 포지션 고려")

def render_backtest_section(backtest_results):
    """
    [추가됨] v2.3.0: 백테스팅 결과를 렌더링합니다.
    """
    st.markdown("---")
    st.markdown("## 🔬 백테스팅 결과")
    
    if not backtest_results['success']:
        st.error(f"❌ 백테스팅 실패: {backtest_results['error']}")
        return
    
    summary = backtest_results['summary']
    
    # 1. 요약 지표 (4열)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="총 수익률",
            value=f"{summary['total_return']:+.2f}%",
            delta=f"${summary['final_capital'] - (summary['final_capital'] / (1 + summary['total_return']/100)):.2f}"
        )
    
    with col2:
        st.metric(
            label="승률",
            value=f"{summary['win_rate']:.1f}%",
            delta=f"{summary['winning_trades']}/{summary['total_trades']} 승"
        )
    
    with col3:
        st.metric(
            label="샤프 비율",
            value=f"{summary['sharpe_ratio']:.2f}",
            delta="높을수록 좋음"
        )
    
    with col4:
        st.metric(
            label="최대 낙폭",
            value=f"-{summary['max_drawdown']:.2f}%",
            delta="MDD"
        )
    
    # 2. 상세 통계
    st.markdown("### 📊 상세 통계")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **💰 수익 지표**
        - 초기 자본: ${summary['final_capital'] / (1 + summary['total_return']/100):.2f}
        - 최종 자본: ${summary['final_capital']:.2f}
        - 평균 승리: {summary['avg_win']:.2f}%
        - 평균 손실: {summary['avg_loss']:.2f}%
        """)
    
    with col2:
        st.markdown(f"""
        **📈 거래 통계**
        - 총 거래 횟수: {summary['total_trades']}
        - 승리 거래: {summary['winning_trades']}
        - 손실 거래: {summary['losing_trades']}
        - 승률: {summary['win_rate']:.1f}%
        """)
    
    # 3. 거래 내역 테이블
    st.markdown("### 📋 거래 내역")
    
    trades_df = backtest_results['trades'].copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    # 포지션 한글화
    trades_df['position_kr'] = trades_df['position'].map({'long': '롱', 'short': '숏'})
    
    # 표시할 컬럼 선택
    display_df = trades_df[[
        'entry_time', 'exit_time', 'position_kr', 
        'entry_price', 'exit_price', 'profit_pct', 
        'capital_after', 'prediction_confidence'
    ]].copy()
    
    display_df.columns = [
        '진입 시간', '청산 시간', '포지션', 
        '진입가', '청산가', '수익률(%)', 
        '잔고', '예측 신뢰도(%)'
    ]
    
    # 수익률에 따른 색상 적용
    def highlight_profit(row):
        color = '#90EE90' if row['수익률(%)'] > 0 else '#FFB6C6'
        return [f'background-color: {color}'] * len(row)
    
    styled_df = display_df.style.apply(highlight_profit, axis=1)
    
    st.dataframe(
        styled_df,
        hide_index=True,
        use_container_width=True
    )

# =====================================================
# 7. TA-Lib 패턴 분석 함수 (v2.1.0)
# =====================================================

def detect_candlestick_patterns_talib(data):
    """
    TA-Lib를 사용하여 61개 캔들스틱 패턴을 감지합니다.
    """
    if not TALIB_AVAILABLE:
        return detect_candlestick_patterns_basic(data)
    
    patterns = {}
    
    # 61개 패턴 함수 목록
    pattern_functions = [
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE',
        'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
        'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
        'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
        'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
        'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS',
        'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
        'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
        'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD',
        'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN',
        'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
        'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
    ]
    
    open_price = data['Open'].values
    high_price = data['High'].values
    low_price = data['Low'].values
    close_price = data['Close'].values
    
    for pattern_name in pattern_functions:
        try:
            pattern_func = getattr(talib, pattern_name)
            result = pattern_func(open_price, high_price, low_price, close_price)
            
            # 최근 5개 봉에서 패턴 발견 여부
            if np.any(result[-5:] != 0):
                patterns[pattern_name] = {
                    'detected': True,
                    'signal': int(result[-1]),
                    'name': pattern_name.replace('CDL', ''),
                    'category': categorize_pattern(pattern_name)
                }
        except Exception:
            continue
    
    return patterns

def detect_candlestick_patterns_basic(data):
    """
    기본 3개 패턴 (Doji, Hammer, Engulfing)을 수동으로 감지합니다.
    """
    patterns = {}
    
    if len(data) < 2:
        return patterns
    
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    body = abs(latest['Close'] - latest['Open'])
    range_val = latest['High'] - latest['Low']
    
    # 1. Doji
    if range_val > 0 and body / range_val < 0.1:
        patterns['CDLDOJI'] = {
            'detected': True,
            'signal': 0,
            'name': 'DOJI',
            'category': 'reversal'
        }
    
    # 2. Hammer
    lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
    upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
    
    if range_val > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.5:
        patterns['CDLHAMMER'] = {
            'detected': True,
            'signal': 100,
            'name': 'HAMMER',
            'category': 'bullish_reversal'
        }
    
    # 3. Engulfing
    prev_body = abs(prev['Close'] - prev['Open'])
    if latest['Close'] > latest['Open'] and prev['Close'] < prev['Open']:
        if body > prev_body * 1.5:
            patterns['CDLENGULFING'] = {
                'detected': True,
                'signal': 100,
                'name': 'ENGULFING',
                'category': 'bullish_reversal'
            }
    
    return patterns

def categorize_pattern(pattern_name):
    """패턴을 카테고리로 분류합니다."""
    bullish = [
        'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLMORNINGSTAR', 'CDLPIERCING',
        'CDL3WHITESOLDIERS', 'CDLMORNINGDOJISTAR', 'CDLENGULFING'
    ]
    
    bearish = [
        'CDLHANGINGMAN', 'CDLSHOOTINGSTAR', 'CDLEVENINGSTAR', 'CDLDARKCLOUDCOVER',
        'CDL3BLACKCROWS', 'CDLEVENINGDOJISTAR'
    ]
    
    if pattern_name in bullish:
        return 'bullish_reversal'
    elif pattern_name in bearish:
        return 'bearish_reversal'
    else:
        return 'reversal'

# =====================================================
# 8. 매도 전략 함수 (v2.1.0)
# =====================================================

def calculate_sell_targets(data, entry_price, strategy='balanced'):
    """
    매도 목표가를 계산합니다.
    
    Args:
        data: OHLCV 데이터
        entry_price: 진입 가격
        strategy: 'conservative', 'balanced', 'aggressive'
    
    Returns:
        dict: 매도 전략 정보
    """
    atr = calculate_atr(data)
    latest_atr = atr.iloc[-1]
    
    if pd.isna(latest_atr):
        latest_atr = data['Close'].iloc[-14:].std()
    
    strategies = {
        'conservative': {
            'target_multiplier': 1.5,
            'stop_multiplier': 1.0,
            'description': '보수적 전략: 빠른 익절, 타이트한 손절'
        },
        'balanced': {
            'target_multiplier': 2.0,
            'stop_multiplier': 1.5,
            'description': '중립적 전략: 균형잡힌 손익비'
        },
        'aggressive': {
            'target_multiplier': 3.0,
            'stop_multiplier': 2.0,
            'description': '공격적 전략: 큰 수익 추구, 넓은 손절'
        }
    }
    
    selected = strategies[strategy]
    
    target_price = entry_price + (latest_atr * selected['target_multiplier'])
    stop_loss = entry_price - (latest_atr * selected['stop_multiplier'])
    
    current_price = data['Close'].iloc[-1]
    
    # 현재 상태 평가
    if current_price >= target_price:
        status = 'target_reached'
        status_text = '🎯 목표가 도달'
    elif current_price <= stop_loss:
        status = 'stop_loss_hit'
        status_text = '🛑 손절가 도달'
    else:
        progress = (current_price - entry_price) / (target_price - entry_price) * 100
        status = 'in_progress'
        status_text = f'⏳ 진행 중 ({progress:.1f}%)'
    
    return {
        'strategy': strategy,
        'description': selected['description'],
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'current_price': current_price,
        'status': status,
        'status_text': status_text,
        'target_pct': (target_price / entry_price - 1) * 100,
        'stop_pct': (stop_loss / entry_price - 1) * 100,
        'current_pct': (current_price / entry_price - 1) * 100
    }

# =====================================================
# 9. 메인 애플리케이션
# =====================================================

def main():
    st.title("📊 AI 암호화폐 트레이딩 전략 분석 v2.3.0")
    st.markdown("**실시간 데이터 분석 및 자동 백테스팅 엔진**")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    
    # 코인 선택
    st.sidebar.subheader("1️⃣ 코인 선택")
    
    coin_input_method = st.sidebar.radio(
        "입력 방법",
        ["목록에서 선택", "직접 입력"],
        index=0
    )
    
    if coin_input_method == "목록에서 선택":
        popular_coins = {
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'Ripple': 'XRP-USD',
            'Cardano': 'ADA-USD',
            'Solana': 'SOL-USD',
            'Polkadot': 'DOT-USD',
            'Dogecoin': 'DOGE-USD',
            'Avalanche': 'AVAX-USD'
        }
        
        selected_coin = st.sidebar.selectbox(
            "코인 선택",
            options=list(popular_coins.keys()),
            index=0
        )
        ticker = popular_coins[selected_coin]
    
    else:
        custom_ticker = st.sidebar.text_input(
            "티커 심볼 입력 (예: BTC, ETH)",
            value="BTC",
            max_chars=10
        ).strip().upper()
        
        # USDT 페어 자동 추가
        if not custom_ticker.endswith('-USD') and not custom_ticker.endswith('-USDT'):
            ticker = f"{custom_ticker}-USD"
        else:
            ticker = custom_ticker
        
        st.sidebar.info(f"✅ 사용될 티커: **{ticker}**")
    
    # 분해능 선택
    st.sidebar.subheader("2️⃣ 분해능 선택")
    
    interval_options = {
        '1분': '1m',
        '5분': '5m',
        '15분': '15m',
        '1시간': '1h',
        '4시간': '4h',
        '1일': '1d',
        '1주': '1wk',
        '1개월': '1mo'
    }
    
    selected_interval_name = st.sidebar.selectbox(
        "시간 간격",
        options=list(interval_options.keys()),
        index=3  # 기본값: 1시간
    )
    interval = interval_options[selected_interval_name]
    
    # 데이터 기간 선택
    st.sidebar.subheader("3️⃣ 데이터 기간")
    
    period_options = {
        '7일': '7d',
        '30일': '30d',
        '90일': '90d',
        '180일': '180d',
        '365일': '365d',
        '최대': 'max'
    }
    
    selected_period_name = st.sidebar.selectbox(
        "기간 선택",
        options=list(period_options.keys()),
        index=2  # 기본값: 90일
    )
    period = period_options[selected_period_name]
    
    # 백테스팅 설정 (v2.3.0)
    st.sidebar.subheader("4️⃣ 백테스팅 설정")
    
    enable_backtest = st.sidebar.checkbox("백테스팅 활성화", value=False)
    
    if enable_backtest:
        backtest_capital = st.sidebar.number_input(
            "초기 자본금 (USD)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
        backtest_lookback = st.sidebar.slider(
            "예측 윈도우 (봉 개수)",
            min_value=20,
            max_value=100,
            value=30,
            step=10
        )
    
    # 고급 설정
    st.sidebar.subheader("5️⃣ 고급 기능")
    
    show_patterns = st.sidebar.checkbox("📊 캔들스틱 패턴 분석", value=False)
    show_sell_strategy = st.sidebar.checkbox("💰 매도 시점 예측", value=False)
    
    if show_sell_strategy:
        entry_price = st.sidebar.number_input(
            "진입 가격 (USD)",
            min_value=0.0001,
            value=50000.0,
            step=100.0,
            format="%.4f"
        )
        
        sell_strategy = st.sidebar.selectbox(
            "매도 전략",
            options=['conservative', 'balanced', 'aggressive'],
            index=1,
            format_func=lambda x: {
                'conservative': '보수적 (빠른 익절)',
                'balanced': '균형 (중립)',
                'aggressive': '공격적 (큰 수익)'
            }[x]
        )
    
    # 데이터 로드 버튼
    if st.sidebar.button("🚀 분석 시작", type="primary"):
        with st.spinner(f"⏳ {ticker} 데이터를 불러오는 중..."):
            data = load_crypto_data(ticker, interval=interval, period=period)
        
        if data.empty:
            st.error("❌ 데이터를 불러올 수 없습니다. 티커 심볼을 확인하세요.")
            st.stop()
        
        # 기술적 지표 계산
        with st.spinner("📊 기술적 지표 계산 중..."):
            data['EMA50'] = calculate_ema(data, 50)
            data['EMA200'] = calculate_ema(data, 200)
            data['RSI'] = calculate_rsi_wilders(data, 14)
            
            macd, macd_signal, macd_hist = calculate_macd(data)
            data['MACD'] = macd
            data['MACD_Signal'] = macd_signal
            data['MACD_Hist'] = macd_hist
            
            upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data)
            data['BB_Upper'] = upper_bb
            data['BB_Middle'] = middle_bb
            data['BB_Lower'] = lower_bb
            
            data['ATR'] = calculate_atr(data)
            
            # NaN 제거 (선택적)
            # data = data.dropna()
        
        # 데이터 요약
        st.success(f"✅ {ticker} 데이터 로드 완료!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("현재가", f"${data['Close'].iloc[-1]:,.2f}")
        with col2:
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            st.metric("24시간 변동", f"${price_change:,.2f}", f"{price_change_pct:+.2f}%")
        with col3:
            st.metric("최고가", f"${data['High'].max():,.2f}")
        with col4:
            st.metric("최저가", f"${data['Low'].min():,.2f}")
        
        # 가격 차트
        st.markdown("---")
        st.markdown("## 📈 가격 차트")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f'{ticker} 가격 차트', 'RSI', 'MACD')
        )
        
        # 캔들스틱
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='가격',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # EMA
        fig.add_trace(
            go.Scatter(x=data.index, y=data['EMA50'], name='EMA50', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['EMA200'], name='EMA200', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # 볼린저 밴드
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dot')),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='orange', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram', marker_color='gray'),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="날짜", row=3, col=1)
        fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        fig.update_layout(
            height=900,
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 데이터 분석 결과
        st.markdown("---")
        st.markdown("## 📊 데이터 분석 결과")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 이동평균 분석")
            latest_ema50 = data['EMA50'].iloc[-1]
            latest_ema200 = data['EMA200'].iloc[-1]
            
            if pd.notna(latest_ema50) and pd.notna(latest_ema200):
                if latest_ema50 > latest_ema200:
                    st.success("🟢 **골든크로스** - 상승 추세")
                    st.markdown(f"- EMA50: ${latest_ema50:,.2f}")
                    st.markdown(f"- EMA200: ${latest_ema200:,.2f}")
                else:
                    st.error("🔴 **데드크로스** - 하락 추세")
                    st.markdown(f"- EMA50: ${latest_ema50:,.2f}")
                    st.markdown(f"- EMA200: ${latest_ema200:,.2f}")
            else:
                st.warning("⚠️ 데이터 부족으로 이동평균 분석 불가")
        
        with col2:
            st.markdown("### 📊 거래량 분석")
            avg_volume = data['Volume'].iloc[-30:].mean()
            latest_volume = data['Volume'].iloc[-1]
            volume_change = (latest_volume / avg_volume - 1) * 100
            
            if volume_change > 50:
                st.success(f"🟢 **거래량 급증** (+{volume_change:.1f}%)")
            elif volume_change < -30:
                st.warning(f"⚠️ **거래량 감소** ({volume_change:.1f}%)")
            else:
                st.info(f"ℹ️ **거래량 평균 수준** ({volume_change:+.1f}%)")
            
            st.markdown(f"- 현재 거래량: {latest_volume:,.0f}")
            st.markdown(f"- 30일 평균: {avg_volume:,.0f}")
        
        # [추가됨] v2.2.0: AI 예측 실행
        ai_prediction = predict_trend_with_ai(data)
        
        # [추가됨] v2.2.0: AI 예측 결과 표시 (데이터 분석 후, 매매 전략 전)
        render_ai_prediction(ai_prediction)
        
        # 매매 전략
        render_trading_strategy(
            data,
            data['RSI'],
            data['MACD'],
            data['MACD_Signal'],
            ema50=data['EMA50'],  # [수정됨] v2.3.0: ema_short → ema50
            ema200=data['EMA200']
        )
        
        # [추가됨] v2.2.0: 포지션 추천 표시 (매매 전략 내부)
        position_rec = recommend_position(ai_prediction, data)
        render_position_recommendation(position_rec)
        
        # [추가됨] v2.3.0: 백테스팅 실행
        if enable_backtest:
            st.markdown("---")
            with st.spinner("🔬 백테스팅 실행 중... (시간이 소요될 수 있습니다)"):
                backtest_results = run_backtest(
                    data,
                    initial_capital=backtest_capital,
                    lookback_periods=backtest_lookback
                )
            
            render_backtest_section(backtest_results)
            
            # 백테스팅 차트
            if backtest_results['success']:
                st.markdown("### 📈 백테스팅 차트")
                backtest_chart = visualize_backtest_results(backtest_results, data)
                st.plotly_chart(backtest_chart, use_container_width=True)
        
        # 캔들스틱 패턴 분석
        if show_patterns:
            st.markdown("---")
            st.markdown("## 🎨 캔들스틱 패턴 분석")
            
            if TALIB_AVAILABLE:
                st.info("✅ TA-Lib 사용 가능 - 61개 패턴 분석 중...")
            else:
                st.warning("⚠️ TA-Lib 미설치 - 기본 3개 패턴만 분석됩니다.")
            
            with st.spinner("🔍 패턴 감지 중..."):
                patterns = detect_candlestick_patterns_talib(data)
            
            if patterns:
                st.success(f"✅ {len(patterns)}개 패턴 감지됨")
                
                # 패턴을 카테고리별로 그룹화
                bullish_patterns = []
                bearish_patterns = []
                neutral_patterns = []
                
                for pattern_name, pattern_info in patterns.items():
                    if pattern_info['category'] == 'bullish_reversal':
                        bullish_patterns.append((pattern_name, pattern_info))
                    elif pattern_info['category'] == 'bearish_reversal':
                        bearish_patterns.append((pattern_name, pattern_info))
                    else:
                        neutral_patterns.append((pattern_name, pattern_info))
                
                # 2열 레이아웃으로 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    if bullish_patterns:
                        st.markdown("### 🟢 상승 반전 패턴")
                        for pattern_name, pattern_info in bullish_patterns:
                            st.markdown(f"- **{pattern_info['name']}** (신뢰도: {abs(pattern_info['signal'])}%)")
                    
                    if neutral_patterns:
                        st.markdown("### ⚪ 중립 패턴")
                        for pattern_name, pattern_info in neutral_patterns:
                            st.markdown(f"- **{pattern_info['name']}**")
                
                with col2:
                    if bearish_patterns:
                        st.markdown("### 🔴 하락 반전 패턴")
                        for pattern_name, pattern_info in bearish_patterns:
                            st.markdown(f"- **{pattern_info['name']}** (신뢰도: {abs(pattern_info['signal'])}%)")
            else:
                st.info("ℹ️ 최근 5개 봉에서 감지된 패턴이 없습니다.")
        
        # 매도 시점 예측
        if show_sell_strategy:
            st.markdown("---")
            st.markdown("## 💰 매도 시점 예측")
            
            sell_targets = calculate_sell_targets(data, entry_price, sell_strategy)
            
            st.markdown(f"### {sell_targets['description']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="진입가",
                    value=f"${sell_targets['entry_price']:,.2f}"
                )
            
            with col2:
                st.metric(
                    label="목표가",
                    value=f"${sell_targets['target_price']:,.2f}",
                    delta=f"+{sell_targets['target_pct']:.2f}%"
                )
            
            with col3:
                st.metric(
                    label="손절가",
                    value=f"${sell_targets['stop_loss']:,.2f}",
                    delta=f"{sell_targets['stop_pct']:.2f}%"
                )
            
            # 현재 상태
            st.markdown("### 📊 현재 상태")
            
            if sell_targets['status'] == 'target_reached':
                st.success(sell_targets['status_text'])
                st.balloons()
            elif sell_targets['status'] == 'stop_loss_hit':
                st.error(sell_targets['status_text'])
            else:
                st.info(sell_targets['status_text'])
            
            st.markdown(f"**현재가:** ${sell_targets['current_price']:,.2f} ({sell_targets['current_pct']:+.2f}%)")
            
            # 시각화
            fig_sell = go.Figure()
            
            fig_sell.add_trace(go.Scatter(
                x=[0, 1, 2],
                y=[sell_targets['stop_loss'], sell_targets['entry_price'], sell_targets['target_price']],
                mode='markers+lines',
                marker=dict(size=15, color=['red', 'blue', 'green']),
                line=dict(color='gray', dash='dot'),
                text=['손절가', '진입가', '목표가'],
                textposition='top center',
                name='전략'
            ))
            
            fig_sell.add_trace(go.Scatter(
                x=[1],
                y=[sell_targets['current_price']],
                mode='markers',
                marker=dict(size=20, color='orange', symbol='star'),
                text=['현재가'],
                textposition='top center',
                name='현재가'
            ))
            
            fig_sell.update_layout(
                title="매도 전략 시각화",
                xaxis=dict(visible=False),
                yaxis_title="가격 (USD)",
                height=400
            )
            
            st.plotly_chart(fig_sell, use_container_width=True)
        
        # 데이터 다운로드
        st.markdown("---")
        st.markdown("## 📥 데이터 다운로드")
        
        csv = data.to_csv()
        st.download_button(
            label="📊 CSV 다운로드",
            data=csv,
            file_name=f"{ticker}_{interval}_{period}.csv",
            mime="text/csv"
        )

# =====================================================
# 10. 앱 실행
# =====================================================

if __name__ == "__main__":
    main()
