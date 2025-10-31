# -*- coding: utf-8 -*-
"""
코인 AI 예측 시스템 - v2.9.0.4 (Typing Fixed) (Advanced Risk Management)
✨ 주요 기능:
- 시장 심리 지수 (Fear & Greed Index)
- 포트폴리오 분석 (선택한 코인)
- 앙상블 예측 (8개 모델)
- 적응형 지표 계산

🟢 v2.8.0 신규 기능:
1. Kelly Criterion: AI 신뢰도 기반 최적 Position Size
3. Monte Carlo 시뮬레이션: 확률적 손익 분석
4. Position Sizing 전략 비교: 4가지 전략 성과 비교
5. 포트폴리오 리스크 관리: 다중 포지션 통합 분석
6. 백테스팅 개선: 전략별 성과 측정

🔴 v2.7.2 수정 사항 (CRITICAL):
- Position Size 계산 로직 수정 (레버리지 오류 수정)
- Stop Loss 롱/숏 구분 추가
- 증거금 정보 표시 추가
- 0 나누기 보호 추가
- 가격 유효성 검증 추가

🔵 v2.8.1 최적화 (Optimization):

🟢 v2.9.0 글로벌 데이터 통합:
- Monte Carlo 시뮬레이션 제거 (단순 시뮬레이션 → 실제 데이터 기반)
- CryptoPanic API: 실시간 글로벌 뉴스 및 센티먼트 분석
- FRED API: 미국 CPI 경제 지표 실시간 연동
- 비트코인 도미넌스, 김치 프리미엄, 펀딩비 온체인 분석
- 종합 시장 스코어링: 뉴스+매크로+온체인 통합 (0-100점)
- Dead Code 제거: detect_candlestick_patterns_basic() 삭제
- 미사용 Validation 함수 제거 (4개)
- 미사용 imports 제거 (seaborn, BytesIO, sklearn validation)
- Risk Management 함수 논리적 순서로 재배치
- ML Models 카테고리별 그룹화
- 중복 주석 정리 (-269 라인, 5.0% 감소)
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
# v2.9.0.1: TimeSeriesSplit 복원 (사용됨)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss

# v2.6.0: 추가 분석 도구
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

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
    # PyTorch 미설치 시 로그 (UI에서 표시)
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
    # XGBoost 미설치 시 로그 (UI에서 표시)
    # 더미 모듈
    class xgb:
        class XGBRegressor:
            pass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    # LightGBM 미설치 시 로그 (UI에서 표시)
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
    # Prophet 미설치 시 로그 (UI에서 표시)
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
    pass  # 로그는 UI에서 표시

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
    page_title="코인 AI 예측 시스템 v2.9.0.4",
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


@st.cache_data(ttl=3600)
def get_all_binance_usdt_pairs():
    """
    바이낸스에서 거래 가능한 모든 USDT 페어 가져오기
    
    Returns:
    --------
    list : USDT 페어 리스트 [("비트코인 (BTC)", "BTCUSDT"), ...]
    """
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        usdt_pairs = []
        
        for symbol_info in data['symbols']:
            symbol = symbol_info['symbol']
            status = symbol_info['status']
            
            # USDT 페어이고 거래 가능한 것만 필터링
            if symbol.endswith('USDT') and status == 'TRADING':
                base_asset = symbol_info['baseAsset']
                
                # 한글 이름 매핑 (주요 코인)
                korean_names = {
                    'BTC': '비트코인',
                    'ETH': '이더리움',
                    'BNB': '바이낸스코인',
                    'XRP': '리플',
                    'ADA': '카다노',
                    'SOL': '솔라나',
                    'DOGE': '도지코인',
                    'DOT': '폴카닷',
                    'MATIC': '폴리곤',
                    'SHIB': '시바이누',
                    'AVAX': '아발란체',
                    'UNI': '유니스왑',
                    'LINK': '체인링크',
                    'ATOM': '코스모스',
                    'LTC': '라이트코인',
                    'ETC': '이더리움클래식',
                    'XLM': '스텔라루멘',
                    'NEAR': '니어프로토콜',
                    'APT': '앱토스',
                    'FIL': '파일코인',
                    'ARB': '아비트럼',
                    'OP': '옵티미즘',
                    'SUI': '수이',
                    'TRX': '트론',
                    'BCH': '비트코인캐시',
                    'ALGO': '알고랜드',
                    'VET': '비체인',
                    'ICP': '인터넷컴퓨터',
                    'FTM': '팬텀',
                    'XMR': '모네로',
                    'SAND': '샌드박스',
                    'MANA': '디센트럴랜드',
                    'AXS': '액시인피니티',
                    'THETA': '쎄타',
                    'XTZ': '테조스',
                    'AAVE': '에이브',
                    'GRT': '더그래프',
                    'EOS': '이오스',
                    'MKR': '메이커',
                    'RUNE': '토르체인',
                    'KSM': '쿠사마',
                    'CAKE': '팬케이크스왑',
                    'CRV': '커브',
                    'WAVES': '웨이브',
                    'ZEC': '지캐시',
                    'DASH': '대시',
                    'COMP': '컴파운드',
                    'YFI': '연파이낸스',
                    'SNX': '신세틱스',
                    'BAT': '베이직어텐션토큰',
                    'ENJ': '엔진코인',
                    'SUSHI': '스시스왑',
                    '1INCH': '원인치',
                    'CHZ': '칠리즈',
                    'HBAR': '헤데라',
                    'HOT': '홀로체인',
                    'ZIL': '질리카',
                    'ONT': '온톨로지',
                    'ICX': '아이콘',
                    'QNT': '퀀트',
                    'LRC': '루프링',
                    'CELO': '셀로',
                    'ANKR': '앵커',
                    'KAVA': '카바',
                    'BAND': '밴드프로토콜',
                    'SC': '시아코인',
                    'RVN': '레이븐코인',
                    'ZEN': '호라이즌',
                    'IOST': '아이오스트',
                    'CVC': '시빅',
                    'STORJ': '스토리지',
                    'DYDX': '디와이디엑스',
                    'GMX': '지엠엑스',
                    'LDO': '리도',
                    'BLUR': '블러',
                    'PEPE': '페페',
                    'FLOKI': '플로키',
                    'INJ': '인젝티브',
                    'STX': '스택스',
                    'IMX': '이뮤터블엑스',
                    'TIA': '셀레스티아',
                    'SEI': '세이',
                    'PYTH': '피스네트워크',
                    'JUP': '주피터',
                    'WIF': '도그위프햇',
                    'BONK': '봉크',
                    'STRK': '스타크넷',
                    'WLD': '월드코인',
                    'FET': '페치AI',
                    'AGIX': '싱귤래리티넷',
                    'RNDR': '렌더토큰',
                    'GRT': '더그래프',
                    'OCEAN': '오션프로토콜'
                }
                
                if base_asset in korean_names:
                    display_name = f"{korean_names[base_asset]} ({base_asset})"
                else:
                    display_name = base_asset
                
                usdt_pairs.append((display_name, symbol))
        
        # 심볼 알파벳 순서로 정렬
        usdt_pairs.sort(key=lambda x: x[1])
        
        return usdt_pairs
    
    except Exception as e:
        st.warning(f"⚠️ 바이낸스 API 오류: {e}")
        # 실패 시 기본 목록 반환
        return [
            ("비트코인 (BTC)", "BTCUSDT"),
            ("이더리움 (ETH)", "ETHUSDT"),
            ("리플 (XRP)", "XRPUSDT"),
            ("도지코인 (DOGE)", "DOGEUSDT"),
            ("카다노 (ADA)", "ADAUSDT"),
            ("솔라나 (SOL)", "SOLUSDT")
        ]

# ════════════════════════════════════════════════════════════════════════════
# v2.6.0: 고급 분석 기능
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_fear_greed_index(limit=30):
    """
    Fear & Greed Index 가져오기 (Alternative.me API)
    
    Returns:
    --------
    dict or None
        - 'current_value': 현재 값 (0-100)
        - 'current_classification': 분류
        - 'historical_data': DataFrame
    """
    try:
        url = f'https://api.alternative.me/fng/?limit={limit}'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'data' not in data:
            return None
        
        current = data['data'][0]
        current_value = int(current['value'])
        current_classification = current['value_classification']
        
        historical = []
        for item in data['data']:
            historical.append({
                'timestamp': datetime.datetime.fromtimestamp(int(item['timestamp'])),
                'value': int(item['value']),
                'classification': item['value_classification']
            })
        
        historical_df = pd.DataFrame(historical)
        
        return {
            'current_value': current_value,
            'current_classification': current_classification,
            'historical_data': historical_df
        }
    except Exception as e:
        return None


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Sharpe Ratio 계산
    
    Parameters:
    -----------
    returns : pd.Series
        수익률 데이터
    risk_free_rate : float
        연간 무위험 이자율 (기본 2%)
    
    Returns:
    --------
    float : Sharpe Ratio (연율화)
    """
    if len(returns) < 2:
        return 0.0
    
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe





def backtest_portfolio_simple(price_data_df, symbol_name):
    """
    단일 코인 포트폴리오 분석 (이미 다운로드된 데이터 사용)
    
    Parameters:
    -----------
    price_data_df : pd.DataFrame
        가격 데이터 (Close 컬럼 포함)
    symbol_name : str
        코인 심볼 (예: 'BTCUSDT')
    
    Returns:
    --------
    dict or None
        - 'total_return': 총 수익률
        - 'sharpe_ratio': Sharpe Ratio
        - 'max_drawdown': 최대 낙폭
        - 'win_rate': 승률
        - 'portfolio_value': 포트폴리오 가치 시계열
        - 'individual_returns': 각 코인별 수익률
    """
    try:
        # Close 가격 추출
        if 'Close' not in price_data_df.columns:
            return None
        
        prices = price_data_df['Close'].dropna()
        
        if len(prices) < 5:
            return None
        
        # 수익률 계산
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            return None
        
        # 누적 수익률
        cumulative_returns = (1 + returns).cumprod()
        portfolio_value = cumulative_returns * 1000  # 초기 투자 $1000
        
        # 성과 지표 계산
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe = calculate_sharpe_ratio(returns)
        
        # 최대 낙폭 (Maximum Drawdown)
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 승률 (양의 수익률 비율)
        win_rate = (returns > 0).sum() / len(returns) * 100
        
        # 코인별 수익률 (단일 코인)
        coin_short_name = symbol_name[:-4] if symbol_name.endswith('USDT') else symbol_name
        individual_returns = {
            coin_short_name: total_return
        }
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'portfolio_value': portfolio_value,
            'individual_returns': individual_returns
        }
    
    except Exception as e:
        return None



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
        # MultiIndex 컴럼 처리 (yf.download가 MultiIndex 반환하는 경우)
        if isinstance(df.columns, pd.MultiIndex):
            # 컴럼이 ('Close', 'BTC-USD') 형태인 경우 평탄화
            df.columns = df.columns.get_level_values(0)
        
        # 필수 컴럼 확인
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in df.columns for col in required_cols):
            return df
    
    return pd.DataFrame()


def add_candlestick_pattern_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    캠들스틱 패턴 특징 추가 (AI 모델 학습용)
    
    6개 특징 추가:
    - pattern_bullish: 상승 패턴 개수 (최근 20 캠들)
    - pattern_bearish: 하락 패턴 개수 (최근 20 캠들)
    - pattern_strength: 평균 신뢰도 (0-100)
    - pattern_momentum: (-1 to +1, 상승/하락 밸런스)
    - pattern_recency: 최근성 (0-1, 지수 감쇠)
    - pattern_diversity: 다양성 (0-1)
    """
    df = df.copy()
    
    # 초기화
    df['pattern_bullish'] = 0.0
    df['pattern_bearish'] = 0.0
    df['pattern_strength'] = 0.0
    df['pattern_momentum'] = 0.0
    df['pattern_recency'] = 0.0
    df['pattern_diversity'] = 0.0
    
    if not TALIB_AVAILABLE or len(df) < 5:
        return df
    
    try:
        # 패턴 방향 매핑 (간략화된 버전)
        bullish_patterns = [
            'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLMORNINGSTAR', 'CDL3WHITESOLDIERS',
            'CDLPIERCING', 'CDLMATCHINGLOW', 'CDLHOMINGPIGEON', 'CDLTAKURI',
            'CDLMATHOLD', 'CDLCONCEALBABYSWALL', 'CDLLADDERBOTTOM', 'CDLUNIQUE3RIVER',
            'CDL3STARSINSOUTH'
        ]
        
        bearish_patterns = [
            'CDLSHOOTINGSTAR', 'CDLHANGINGMAN', 'CDLEVENINGSTAR', 'CDL3BLACKCROWS',
            'CDLDARKCLOUDCOVER', 'CDLONNECK', 'CDLINNECK', 'CDLTHRUSTING',
            'CDL2CROWS', 'CDLIDENTICAL3CROWS', 'CDLADVANCEBLOCK', 'CDLSTALLEDPATTERN',
            'CDLUPSIDEGAP2CROWS'
        ]
        
        # 모든 패턴 감지 결과 저장
        all_pattern_results = {}
        
        for pattern_list, pattern_type in [(bullish_patterns, 'bullish'), (bearish_patterns, 'bearish')]:
            for pattern_name in pattern_list:
                if hasattr(talib, pattern_name):
                    try:
                        pattern_func = getattr(talib, pattern_name)
                        result = pattern_func(df['Open'].values, df['High'].values, 
                                            df['Low'].values, df['Close'].values)
                        all_pattern_results[pattern_name] = (result, pattern_type)
                    except:
                        continue
        
        # 각 행에 대해 윈도우 기반 특징 계산
        for i in range(window, len(df)):
            bullish_count = 0
            bearish_count = 0
            confidences = []
            unique_patterns = set()
            min_distance = window
            
            # 윈도우 범위
            start_idx = i - window
            end_idx = i + 1
            
            # 모든 패턴 결과 검사
            for pattern_name, (result, pattern_type) in all_pattern_results.items():
                for j in range(start_idx, end_idx):
                    if j < 0 or j >= len(result):
                        continue
                    
                    if result[j] != 0:
                        conf = abs(result[j])
                        confidences.append(conf)
                        unique_patterns.add(pattern_name)
                        
                        if pattern_type == 'bullish':
                            bullish_count += 1
                        else:
                            bearish_count += 1
                        
                        distance = i - j
                        if distance < min_distance:
                            min_distance = distance
            
            # 특징 값 설정
            total_patterns = bullish_count + bearish_count
            
            df.iloc[i, df.columns.get_loc('pattern_bullish')] = bullish_count
            df.iloc[i, df.columns.get_loc('pattern_bearish')] = bearish_count
            
            if confidences:
                df.iloc[i, df.columns.get_loc('pattern_strength')] = np.mean(confidences)
            
            if total_patterns > 0:
                # 모멘텀 (-1 to +1)
                momentum = (bullish_count - bearish_count) / (total_patterns + 1)
                df.iloc[i, df.columns.get_loc('pattern_momentum')] = momentum
                
                # 최근성 (지수 감쇠)
                recency = np.exp(-min_distance / 5)
                df.iloc[i, df.columns.get_loc('pattern_recency')] = recency
                
                # 다양성
                diversity = len(unique_patterns) / total_patterns
                df.iloc[i, df.columns.get_loc('pattern_diversity')] = diversity
    
    except Exception as e:
        pass
    
    return df


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

    # Bollinger Bands (추가)
    df['BB_middle'] = df['Close'].rolling(window=window_20).mean()
    df['BB_std'] = df['Close'].rolling(window=window_20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

    # EMA 교차
    df['Cross_Signal'] = 0
    ema50 = df['EMA50']
    ema200 = df['EMA200']
    cond_up = (ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))
    cond_down = (ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))
    df.loc[cond_up, 'Cross_Signal'] = 1
    df.loc[cond_down, 'Cross_Signal'] = -1

    # [v2.7.1 새로 추가] 캠들스틱 패턴 특징 추가
    df = add_candlestick_pattern_features(df)
    
    essential_cols = ['Close', 'High', 'Low', 'Volume', '일일수익률']
    df_clean = df.dropna(subset=essential_cols)
    
    optional_cols = ['RSI14', 'ATR14', 'StochK14', 'MFI14', 'MACD', 'MACD_Signal',
                     'pattern_bullish', 'pattern_bearish', 'pattern_strength',
                     'pattern_momentum', 'pattern_recency', 'pattern_diversity']
    for col in optional_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    
    return df_clean


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
                            investment_amount: float, leverage: float, interval: str = '1h') -> dict:
    """
    매도 시점 예측
    - 보수적/중립/공격적 시나리오 제공
    - ATR 기반 동적 손절/익절
    - 추세 전환 신호 감지
    - 시간 기반 예측 (분/시간/일 단위)
    """
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI14'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    
    # 추세 판단
    trend = 'bullish' if ema50 > ema200 else 'bearish'
    
    # 시간 간격별 분 단위 계산
    interval_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }
    minutes_per_candle = interval_minutes.get(interval, 60)
    
    # 3가지 시나리오
    scenarios = {}
    
    # 1. 보수적 (빠른 익절, 손절)
    scenarios['conservative'] = {
        'name': '🛡️ 보수적 전략',
        'take_profit': entry_price + (atr * 1.5),
        'stop_loss': entry_price - (atr * 1.0),
        'holding_period': '1-3일',
        'time_estimate_minutes': 24 * 60,  # 1일 기본값
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
        'time_estimate_minutes': 5 * 24 * 60,  # 5일 기본값
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
        'time_estimate_minutes': 10 * 24 * 60,  # 10일 기본값
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
    
    # 시간 예측 계산 (가격 변동률 기반)
    try:
        # 최근 24시간 가격 변동률 계산
        recent_prices = df['Close'].tail(min(24, len(df)))
        price_changes = recent_prices.pct_change().dropna()
        avg_change_per_period = price_changes.mean() if len(price_changes) > 0 else 0.001
        
        # 각 시나리오별 예측 시간 계산
        for scenario in scenarios.values():
            target_price = scenario['take_profit']
            price_diff_pct = (target_price - current_price) / current_price
            
            if avg_change_per_period > 0.0001:  # 영으로 나누기 방지
                periods_needed = abs(price_diff_pct / avg_change_per_period)
                minutes_needed = int(periods_needed * minutes_per_candle)
                
                # 최소/최대 제한
                minutes_needed = max(60, min(minutes_needed, 30 * 24 * 60))  # 1시간 ~ 30일
                scenario['time_estimate_minutes'] = minutes_needed
            
    except Exception as e:
        # 계산 실패 시 기본값 유지
        pass
    
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
    
    # 주의사항 삭제됨



# ==============================================================================
# RISK MANAGEMENT FUNCTIONS (Reordered for logical flow)
# ==============================================================================

def calculate_kelly_criterion(ai_confidence: float, rr_ratio: float, win_rate: float = None,
                              kelly_fraction: float = 0.5, max_position: float = 0.25) -> dict:
    """
    Kelly Criterion을 사용한 최적 Position Size 계산
    
    공식: Kelly = (b*p - q) / b
    - b = RR Ratio (승률)
    - p = 승리 확률 (AI 신뢰도 또는 백테스팅 결과)
    - q = 패배 확률 (1-p)
    """
    p = (ai_confidence / 100.0) if win_rate is None else win_rate
    p = max(0.01, min(0.99, p))
    q = 1.0 - p
    
    if rr_ratio <= 0:
        return {'kelly_full': 0.0, 'kelly_adjusted': 0.0, 'kelly_capped': 0.0,
                'position_pct': 0.0, 'recommendation': 'NO TRADE',
                'risk_category': '비정상', 'reason': 'RR Ratio가 0 이하입니다.',
                'win_rate_used': p, 'rr_ratio_used': rr_ratio, 'kelly_fraction_used': kelly_fraction}
    
    b = rr_ratio
    kelly_full = (b * p - q) / b
    
    if kelly_full <= 0:
        return {'kelly_full': kelly_full, 'kelly_adjusted': 0.0, 'kelly_capped': 0.0,
                'position_pct': 0.0, 'recommendation': 'NO TRADE',
                'risk_category': '기대값 음수',
                'reason': f'기대값이 음수입니다 (p={p:.1%}, b={b:.2f})',
                'win_rate_used': p, 'rr_ratio_used': b, 'kelly_fraction_used': kelly_fraction}
    
    kelly_adjusted = kelly_full * kelly_fraction
    kelly_capped = min(kelly_adjusted, max_position)
    
    if kelly_capped < 0.02:
        risk_category, recommendation = '거래 제외', 'SKIP'
        reason = '포지션 크기가 너무 작습니다 (2% 미만)'
    elif kelly_capped < 0.05:
        risk_category, recommendation = '매우 보수적', 'TRADE'
        reason = '보수적 포지션 (2-5%)'
    elif kelly_capped < 0.10:
        risk_category, recommendation = '중립적', 'TRADE'
        reason = '중립적 포지션 (5-10%)'
    elif kelly_capped < 0.15:
        risk_category, recommendation = '공격적', 'TRADE'
        reason = '공격적 포지션 (10-15%)'
    else:
        risk_category, recommendation = '매우 공격적', 'TRADE'
        reason = '매우 공격적 포지션 (15%+)'
    
    return {
        'kelly_full': round(kelly_full, 4),
        'kelly_adjusted': round(kelly_adjusted, 4),
        'kelly_capped': round(kelly_capped, 4),
        'position_pct': round(kelly_capped * 100, 2),
        'recommendation': recommendation,
        'risk_category': risk_category,
        'reason': reason,
        'win_rate_used': p,
        'rr_ratio_used': b,
        'kelly_fraction_used': kelly_fraction
    }



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
    
    # [수정됨] v2.9.0.2: 리스크 레벨 판단 로직 수정
    # 리스크 점수 = 코인 리스크 * 변동성 * 100
    risk_score = crypto_factor * volatility * 100
    
    # 리스크 레벨 분류 (세분화)
    if risk_score < 2:
        risk_level = "매우 낮음"  # 안정적 (BTC + 낮은 변동성)
    elif risk_score < 4:
        risk_level = "낮음"  # 보수적
    elif risk_score < 6:
        risk_level = "중간"  # 중립적
    elif risk_score < 8:
        risk_level = "높음"  # 공격적
    else:
        risk_level = "매우 높음"  # 매우 위험 (알트코인 + 높은 변동성)
    
    return {
        'recommended': recommended_leverage,
        'maximum': maximum_leverage,
        'risk_level': risk_level
    }




# ════════════════════════════════════════════════════════════════════════════
# v2.9.0: 실시간 글로벌 데이터 분석 함수들
# ════════════════════════════════════════════════════════════════════════════
# - CryptoPanic 뉴스 분석
# - FRED 경제 지표
# - 온체인 메트릭 (도미넌스, 김프, 펀딩비, 청산)
# - 종합 시장 분석
# ════════════════════════════════════════════════════════════════════════════
# v2.9.0.4: typing import 명시적 재선언 (Streamlit Cloud 호환성)
from typing import Dict, List, Optional, Tuple


def fetch_cryptopanic_news(
    currency: str = 'BTC',
    api_key: Optional[str] = None,
    limit: int = 20
) -> Dict:
    """
    CryptoPanic API를 통해 실시간 뉴스 수집
    
    Parameters:
    -----------
    currency : str
        암호화폐 심볼 (BTC, ETH 등)
    api_key : str, optional
        CryptoPanic API 키 (없으면 공개 데이터만)
    limit : int
        수집할 뉴스 개수
    
    Returns:
    --------
    dict : {
        'news': list of dict,
        'sentiment_score': float,
        'total_count': int,
        'bullish_count': int,
        'bearish_count': int,
        'neutral_count': int
    }
    """
    try:
        # API 엔드포인트
        base_url = "https://cryptopanic.com/api/v1/posts/"
        
        params = {
            'auth_token': api_key if api_key else 'free',
            'currencies': currency,
            'kind': 'news',  # news, media, blog
            'filter': 'rising',  # hot, rising, bullish, bearish
            'public': 'true'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])[:limit]
            
            # 센티먼트 분석
            sentiment_counts = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
            
            news_list = []
            for item in results:
                votes = item.get('votes', {})
                
                # 센티먼트 결정
                positive = votes.get('positive', 0)
                negative = votes.get('negative', 0)
                important = votes.get('important', 0)
                
                if positive > negative:
                    sentiment = 'positive'
                    sentiment_counts['positive'] += 1
                elif negative > positive:
                    sentiment = 'negative'
                    sentiment_counts['negative'] += 1
                else:
                    sentiment = 'neutral'
                    sentiment_counts['neutral'] += 1
                
                news_list.append({
                    'title': item.get('title', ''),
                    'published_at': item.get('published_at', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', {}).get('title', 'Unknown'),
                    'sentiment': sentiment,
                    'votes_positive': positive,
                    'votes_negative': negative,
                    'votes_important': important,
                    'currencies': item.get('currencies', [])
                })
            
            # 센티먼트 스코어 계산 (-1 ~ +1)
            total = sum(sentiment_counts.values())
            if total > 0:
                sentiment_score = (
                    (sentiment_counts['positive'] - sentiment_counts['negative']) / total
                )
            else:
                sentiment_score = 0.0
            
            return {
                'news': news_list,
                'sentiment_score': sentiment_score,
                'total_count': len(news_list),
                'bullish_count': sentiment_counts['positive'],
                'bearish_count': sentiment_counts['negative'],
                'neutral_count': sentiment_counts['neutral'],
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        else:
            return {
                'news': [],
                'sentiment_score': 0.0,
                'total_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'error': f'API Error: {response.status_code}',
                'status': 'error'
            }
    
    except Exception as e:
        return {
            'news': [],
            'sentiment_score': 0.0,
            'total_count': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'error': str(e),
            'status': 'error'
        }


def analyze_news_sentiment_advanced(news_data: Dict) -> Dict:
    """
    뉴스 센티먼트 고급 분석
    
    Returns:
    --------
    dict : {
        'overall_sentiment': str (Bullish/Bearish/Neutral),
        'confidence': float (0-1),
        'market_impact': str (High/Medium/Low),
        'key_topics': list,
        'recommendation': str
    }
    """
    if not news_data.get('news'):
        return {
            'overall_sentiment': 'Neutral',
            'confidence': 0.0,
            'market_impact': 'Low',
            'key_topics': [],
            'recommendation': 'No recent news data available'
        }
    
    sentiment_score = news_data['sentiment_score']
    total_votes = sum([
        n['votes_positive'] + n['votes_negative'] + n['votes_important']
        for n in news_data['news']
    ])
    
    # 전체 센티먼트 결정
    if sentiment_score > 0.3:
        overall = 'Bullish'
    elif sentiment_score < -0.3:
        overall = 'Bearish'
    else:
        overall = 'Neutral'
    
    # 신뢰도 계산
    confidence = min(abs(sentiment_score) + (total_votes / 1000), 1.0)
    
    # 시장 영향도
    if total_votes > 500 and abs(sentiment_score) > 0.5:
        impact = 'High'
    elif total_votes > 200 or abs(sentiment_score) > 0.3:
        impact = 'Medium'
    else:
        impact = 'Low'
    
    # 키워드 추출 (간단한 빈도 분석)
    all_titles = ' '.join([n['title'].lower() for n in news_data['news']])
    keywords = ['bitcoin', 'eth', 'regulation', 'sec', 'etf', 'trading', 
                'price', 'market', 'crypto', 'bullish', 'bearish']
    key_topics = [kw for kw in keywords if kw in all_titles][:5]
    
    # 추천 메시지
    if overall == 'Bullish' and confidence > 0.6:
        recommendation = "Strong positive market sentiment. Consider long positions."
    elif overall == 'Bearish' and confidence > 0.6:
        recommendation = "Negative market sentiment detected. Exercise caution."
    else:
        recommendation = "Mixed signals. Wait for clearer market direction."
    
    return {
        'overall_sentiment': overall,
        'confidence': confidence,
        'market_impact': impact,
        'key_topics': key_topics,
        'recommendation': recommendation
    }


# ==============================================================================
# 2. FRED ECONOMIC DATA
# ==============================================================================

def fetch_fred_economic_data(
    series_id: str = 'CPIAUCSL',  # CPI for All Urban Consumers
    api_key: Optional[str] = None,
    limit: int = 12
) -> Dict:
    """
    FRED (Federal Reserve Economic Data) API에서 경제 지표 수집
    
    Parameters:
    -----------
    series_id : str
        FRED 시리즈 ID
        - CPIAUCSL: Consumer Price Index
        - UNRATE: Unemployment Rate
        - DFF: Federal Funds Rate
    api_key : str
        FRED API 키
    limit : int
        데이터 포인트 개수
    
    Returns:
    --------
    dict : {
        'data': pandas.DataFrame,
        'latest_value': float,
        'change_mom': float (month-over-month),
        'change_yoy': float (year-over-year),
        'trend': str
    }
    """
    try:
        if not api_key:
            # API 키 없을 시 더미 데이터 반환
            return _get_fred_dummy_data(series_id)
        
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': limit
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            
            # DataFrame 생성
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df = df.sort_values('date')
            
            if len(df) == 0:
                return _get_fred_dummy_data(series_id)
            
            latest_value = df['value'].iloc[-1]
            
            # Month-over-Month 변화
            if len(df) >= 2:
                change_mom = ((latest_value / df['value'].iloc[-2]) - 1) * 100
            else:
                change_mom = 0.0
            
            # Year-over-Year 변화
            if len(df) >= 12:
                change_yoy = ((latest_value / df['value'].iloc[-12]) - 1) * 100
            else:
                change_yoy = 0.0
            
            # 트렌드 분석
            if change_mom > 0.2:
                trend = 'Rising'
            elif change_mom < -0.2:
                trend = 'Falling'
            else:
                trend = 'Stable'
            
            return {
                'data': df,
                'latest_value': latest_value,
                'change_mom': change_mom,
                'change_yoy': change_yoy,
                'trend': trend,
                'series_id': series_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        else:
            return _get_fred_dummy_data(series_id)
    
    except Exception as e:
        return _get_fred_dummy_data(series_id)


def _get_fred_dummy_data(series_id: str) -> Dict:
    """FRED API 실패 시 더미 데이터 반환"""
    # 최근 12개월 더미 데이터
    dates = pd.date_range(end=datetime.now(), periods=12, freq='MS')
    
    if 'CPI' in series_id:
        # CPI 더미 (약 3% 인플레이션)
        base = 300.0
        values = [base * (1.03 ** (i/12)) for i in range(12)]
    else:
        values = [100 + i * 0.5 for i in range(12)]
    
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    return {
        'data': df,
        'latest_value': values[-1],
        'change_mom': 0.3,
        'change_yoy': 3.2,
        'trend': 'Rising',
        'series_id': series_id,
        'timestamp': datetime.now().isoformat(),
        'status': 'dummy'
    }


# ==============================================================================
# 3. ONCHAIN METRICS
# ==============================================================================

def fetch_btc_dominance() -> Dict:
    """
    비트코인 도미넌스 (시가총액 점유율) 수집
    
    Returns:
    --------
    dict : {
        'dominance': float (percentage),
        'trend': str,
        'change_24h': float
    }
    """
    try:
        # CoinGecko API
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            market_data = data.get('data', {})
            
            dominance = market_data.get('market_cap_percentage', {}).get('btc', 0)
            
            # 간단한 트렌드 (실제로는 historical 데이터 필요)
            if dominance > 45:
                trend = 'Strong'
            elif dominance > 40:
                trend = 'Moderate'
            else:
                trend = 'Weak'
            
            return {
                'dominance': dominance,
                'trend': trend,
                'change_24h': 0.0,  # Historical data needed
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        else:
            return {'dominance': 0, 'trend': 'Unknown', 'change_24h': 0, 'status': 'error'}
    
    except Exception as e:
        return {'dominance': 0, 'trend': 'Unknown', 'change_24h': 0, 'error': str(e), 'status': 'error'}


def fetch_kimchi_premium(symbol: str = 'BTC') -> Dict:
    """
    김치 프리미엄 계산 (한국 거래소 vs 글로벌 거래소)
    
    Returns:
    --------
    dict : {
        'premium': float (percentage),
        'korea_price': float,
        'global_price': float,
        'signal': str
    }
    """
    try:
        # Upbit (한국) 가격
        upbit_url = f"https://api.upbit.com/v1/ticker?markets=KRW-{symbol}"
        upbit_response = requests.get(upbit_url, timeout=10)
        
        # Binance (글로벌) 가격
        binance_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        binance_response = requests.get(binance_url, timeout=10)
        
        # USD/KRW 환율 (고정값 또는 API에서 가져오기)
        usd_krw = 1320  # 대략적인 환율
        
        if upbit_response.status_code == 200 and binance_response.status_code == 200:
            upbit_data = upbit_response.json()[0]
            binance_data = binance_response.json()
            
            korea_price = upbit_data['trade_price']  # KRW
            global_price_usd = float(binance_data['price'])  # USD
            global_price_krw = global_price_usd * usd_krw
            
            # 프리미엄 계산
            premium = ((korea_price / global_price_krw) - 1) * 100
            
            # 시그널
            if premium > 5:
                signal = 'High Premium (Bullish KR Market)'
            elif premium < -5:
                signal = 'Negative Premium (Bearish KR Market)'
            else:
                signal = 'Normal Range'
            
            return {
                'premium': premium,
                'korea_price': korea_price,
                'global_price': global_price_krw,
                'usd_krw_rate': usd_krw,
                'signal': signal,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        else:
            return {'premium': 0, 'korea_price': 0, 'global_price': 0, 'signal': 'Unknown', 'status': 'error'}
    
    except Exception as e:
        return {'premium': 0, 'korea_price': 0, 'global_price': 0, 'signal': 'Unknown', 'error': str(e), 'status': 'error'}


def fetch_funding_rate(symbol: str = 'BTCUSDT') -> Dict:
    """
    Binance 선물 펀딩비 수집
    
    Returns:
    --------
    dict : {
        'funding_rate': float,
        'next_funding_time': str,
        'signal': str
    }
    """
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                latest = data[0]
                funding_rate = float(latest['fundingRate']) * 100  # Percentage
                
                # 시그널
                if funding_rate > 0.05:
                    signal = 'High Positive (Overleveraged Long)'
                elif funding_rate < -0.05:
                    signal = 'Negative (Short Dominance)'
                else:
                    signal = 'Neutral'
                
                return {
                    'funding_rate': funding_rate,
                    'funding_time': latest['fundingTime'],
                    'signal': signal,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
        
        return {'funding_rate': 0, 'funding_time': '', 'signal': 'Unknown', 'status': 'error'}
    
    except Exception as e:
        return {'funding_rate': 0, 'funding_time': '', 'signal': 'Unknown', 'error': str(e), 'status': 'error'}


def fetch_liquidation_data(symbol: str = 'BTCUSDT', period: str = '24h') -> Dict:
    """
    청산 데이터 수집 (Coinglass API 또는 추정)
    
    Returns:
    --------
    dict : {
        'total_liquidation': float,
        'long_liquidation': float,
        'short_liquidation': float,
        'signal': str
    }
    """
    try:
        # Coinglass API는 유료이므로 Binance 공개 데이터 활용
        # 실제로는 historical liquidation data가 필요
        
        # 더미 데이터 (실제 구현 시 API 연동 필요)
        return {
            'total_liquidation': 0,
            'long_liquidation': 0,
            'short_liquidation': 0,
            'signal': 'Data Unavailable (Premium API Required)',
            'timestamp': datetime.now().isoformat(),
            'status': 'dummy'
        }
    
    except Exception as e:
        return {
            'total_liquidation': 0,
            'long_liquidation': 0,
            'short_liquidation': 0,
            'signal': 'Error',
            'error': str(e),
            'status': 'error'
        }


# ==============================================================================
# 4. COMPREHENSIVE MARKET ANALYSIS
# ==============================================================================

def analyze_comprehensive_market(
    symbol: str,
    news_data: Dict,
    fred_data: Dict,
    dominance_data: Dict,
    kimchi_data: Dict,
    funding_data: Dict,
    current_price: float,
    ai_confidence: float
) -> Dict:
    """
    종합 시장 분석 - 모든 데이터 통합
    
    Returns:
    --------
    dict : {
        'overall_score': float (0-100),
        'recommendation': str (Strong Buy/Buy/Hold/Sell/Strong Sell),
        'confidence': float (0-1),
        'key_factors': list,
        'risk_level': str,
        'summary': str
    }
    """
    scores = []
    factors = []
    
    # 1. 뉴스 센티먼트 (30% 가중치)
    news_score = (news_data.get('sentiment_score', 0) + 1) * 50  # 0-100 scale
    scores.append(news_score * 0.3)
    if news_data.get('sentiment_score', 0) > 0.3:
        factors.append("✅ Positive News Sentiment")
    elif news_data.get('sentiment_score', 0) < -0.3:
        factors.append("⚠️ Negative News Sentiment")
    
    # 2. 경제 지표 (20% 가중치)
    if fred_data.get('trend') == 'Rising':
        macro_score = 30  # 인플레이션 상승은 crypto에 부정적일 수 있음
        factors.append("⚠️ Rising Inflation (Macro Risk)")
    else:
        macro_score = 70
        factors.append("✅ Stable Macro Environment")
    scores.append(macro_score * 0.2)
    
    # 3. 비트코인 도미넌스 (15% 가중치)
    dominance = dominance_data.get('dominance', 0)
    if dominance > 45:
        dom_score = 80 if 'BTC' in symbol else 40
        factors.append(f"{'✅' if 'BTC' in symbol else '⚠️'} BTC Dominance High ({dominance:.1f}%)")
    else:
        dom_score = 40 if 'BTC' in symbol else 80
        factors.append(f"{'⚠️' if 'BTC' in symbol else '✅'} BTC Dominance Low ({dominance:.1f}%)")
    scores.append(dom_score * 0.15)
    
    # 4. 김치 프리미엄 (10% 가중치)
    premium = kimchi_data.get('premium', 0)
    if premium > 3:
        kimchi_score = 75
        factors.append(f"✅ Kimchi Premium Positive (+{premium:.2f}%)")
    elif premium < -3:
        kimchi_score = 25
        factors.append(f"⚠️ Kimchi Premium Negative ({premium:.2f}%)")
    else:
        kimchi_score = 50
    scores.append(kimchi_score * 0.1)
    
    # 5. 펀딩비 (15% 가중치)
    funding = funding_data.get('funding_rate', 0)
    if funding > 0.1:
        funding_score = 30  # Over-leveraged long
        factors.append(f"⚠️ High Funding Rate (+{funding:.3f}%) - Overleveraged")
    elif funding < -0.05:
        funding_score = 70  # Short squeeze potential
        factors.append(f"✅ Negative Funding ({funding:.3f}%) - Short Squeeze Risk")
    else:
        funding_score = 60
        factors.append("✅ Balanced Funding Rate")
    scores.append(funding_score * 0.15)
    
    # 6. AI 신뢰도 (10% 가중치)
    ai_score = ai_confidence * 100
    scores.append(ai_score * 0.1)
    if ai_confidence > 0.7:
        factors.append(f"✅ High AI Confidence ({ai_confidence:.1%})")
    
    # 종합 점수
    overall_score = sum(scores)
    
    # 추천 결정
    if overall_score >= 75:
        recommendation = "Strong Buy"
        risk_level = "Low"
    elif overall_score >= 60:
        recommendation = "Buy"
        risk_level = "Medium"
    elif overall_score >= 40:
        recommendation = "Hold"
        risk_level = "Medium"
    elif overall_score >= 25:
        recommendation = "Sell"
        risk_level = "High"
    else:
        recommendation = "Strong Sell"
        risk_level = "Very High"
    
    # 신뢰도 계산
    confidence = (
        0.3 * (1 if news_data.get('status') == 'success' else 0) +
        0.2 * (1 if fred_data.get('status') in ['success', 'dummy'] else 0) +
        0.2 * (1 if dominance_data.get('status') == 'success' else 0) +
        0.15 * (1 if kimchi_data.get('status') == 'success' else 0) +
        0.15 * (1 if funding_data.get('status') == 'success' else 0)
    )
    
    summary = f"Comprehensive market analysis shows {recommendation} signal with {overall_score:.0f}/100 score."
    
    return {
        'overall_score': overall_score,
        'recommendation': recommendation,
        'confidence': confidence,
        'key_factors': factors,
        'risk_level': risk_level,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }




# ════════════════════════════════════════════════════════════════════════════
# v2.8.0: 고급 리스크 관리 함수들
# ════════════════════════════════════════════════════════════════════════════


def compare_position_sizing_strategies(investment_amount: float, entry_price: float,
                                      stop_loss: float, take_profit: float,
                                      ai_confidence: float, volatility: float,
                                      leverage: float = 1.0, rr_ratio: float = 2.0) -> dict:
    """
    여러 Position Sizing 전략 비교
    1. Fixed Fractional (2%)
    2. Kelly Criterion (Half Kelly)
    3. Volatility Adjusted
    4. AI Confidence Based
    """
    stop_loss_distance = abs(entry_price - stop_loss)
    
    # 1. Fixed Fractional
    fixed_risk_pct = 0.02
    fixed_position_size = (investment_amount * fixed_risk_pct) / stop_loss_distance
    
    # 2. Kelly Criterion
    kelly_result = calculate_kelly_criterion(ai_confidence, rr_ratio, kelly_fraction=0.5)
    kelly_risk_pct = kelly_result['kelly_capped']
    kelly_position_size = (investment_amount * kelly_risk_pct) / stop_loss_distance
    
    # 3. Volatility Adjusted
    vol_adjusted_risk_pct = max(0.005, min(0.05, 0.02 / (volatility * 50 + 1)))
    vol_adjusted_position_size = (investment_amount * vol_adjusted_risk_pct) / stop_loss_distance
    
    # 4. AI Confidence Based
    ai_risk_pct = max(0.01, min(0.05, (ai_confidence / 100) * 0.05))
    ai_position_size = (investment_amount * ai_risk_pct) / stop_loss_distance
    
    strategies = {
        'fixed_fractional': {
            'name': '고정 비율 (2%)',
            'risk_pct': fixed_risk_pct * 100,
            'position_size': fixed_position_size,
            'position_value': fixed_position_size * entry_price,
            'max_loss': fixed_position_size * stop_loss_distance,
            'max_profit': fixed_position_size * (take_profit - entry_price),
            'required_margin': (fixed_position_size * entry_price) / leverage
        },
        'kelly_criterion': {
            'name': 'Kelly Criterion (Half)',
            'risk_pct': kelly_risk_pct * 100,
            'position_size': kelly_position_size,
            'position_value': kelly_position_size * entry_price,
            'max_loss': kelly_position_size * stop_loss_distance,
            'max_profit': kelly_position_size * (take_profit - entry_price),
            'required_margin': (kelly_position_size * entry_price) / leverage
        },
        'volatility_adjusted': {
            'name': '변동성 조정',
            'risk_pct': vol_adjusted_risk_pct * 100,
            'position_size': vol_adjusted_position_size,
            'position_value': vol_adjusted_position_size * entry_price,
            'max_loss': vol_adjusted_position_size * stop_loss_distance,
            'max_profit': vol_adjusted_position_size * (take_profit - entry_price),
            'required_margin': (vol_adjusted_position_size * entry_price) / leverage
        },
        'ai_confidence_based': {
            'name': 'AI 신뢰도 기반',
            'risk_pct': ai_risk_pct * 100,
            'position_size': ai_position_size,
            'position_value': ai_position_size * entry_price,
            'max_loss': ai_position_size * stop_loss_distance,
            'max_profit': ai_position_size * (take_profit - entry_price),
            'required_margin': (ai_position_size * entry_price) / leverage
        }
    }
    
    recommended_strategy = 'kelly_criterion' if (kelly_result['recommendation'] == 'TRADE' and kelly_risk_pct >= 0.01) else 'fixed_fractional'
    
    return {
        'strategies': strategies,
        'recommended_strategy': recommended_strategy,
        'kelly_details': kelly_result
    }


# ════════════════════════════════════════════════════════════════════════════
# Phase 2: 워크-포워드 검증 (Walk-Forward Validation)
# ════════════════════════════════════════════════════════════════════════════


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
        # [안전성] 디바이스 문제 해결 - x와 같은 디바이스 사용
        forecast = torch.zeros(x.size(0), self.blocks[0].forecast_fc.out_features, dtype=x.dtype, device=x.device)
        
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        
        return forecast


def train_nbeats(data, forecast_days=3, lookback=180, epochs=50):
    """N-BEATS 모델 학습 (경량화 버전)"""
    if not TORCH_AVAILABLE:
        return None, None
    
    try:
        # [안전성] 데이터 길이 체크
        if len(data) < lookback + forecast_days + 20:
            return None, None
        
        # [최적화] lookback 축소 (메모리 절약)
        effective_lookback = min(lookback, 60)  # 최대 60일로 제한
        
        # 데이터 정규화
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        
        # 학습 데이터 생성
        X, y = [], []
        for i in range(effective_lookback, len(scaled_data) - forecast_days):
            X.append(scaled_data[i-effective_lookback:i])
            y.append(scaled_data[i:i+forecast_days])
        
        if len(X) < 20:  # 최소 20개 샘플 필요
            return None, None
        
        # [안전성] 텐서 변환 전 크기 체크
        if len(X) * effective_lookback > 100000:  # 메모리 제한
            # 최근 500개만 사용
            X = X[-500:]
            y = y[-500:]
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
    except Exception as e:
        # N-BEATS 데이터 준비 실패
        return None, None
    
    try:
        # [경량화] 작은 모델 사용
        model = NBeatsModel(
            input_size=effective_lookback, 
            forecast_size=forecast_days, 
            num_blocks=2,  # 3→2 블록
            hidden_size=64  # 128→64 차원
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # [최적화] 배치 학습
        batch_size = min(32, len(X))
        num_batches = len(X) // batch_size
        
        # [안전성] epochs 제한
        safe_epochs = min(epochs, 20)  # 최대 20 epoch
        
        model.train()
        for epoch in range(safe_epochs):
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model, scaler
        
    except Exception as e:
        # N-BEATS 학습 실패
        return None, None


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
    """TFT 모델 학습 (경량화 버전)"""
    if not TORCH_AVAILABLE:
        return None, None
    
    try:
        # [안전성] 데이터 길이 체크
        if len(data) < lookback + forecast_days + 20:
            return None, None
        
        # [최적화] lookback 축소
        effective_lookback = min(lookback, 60)
        
        # 가격 + 지표 결합
        combined_data = features_df[['Close', 'RSI14', 'MACD', 'Volume']].iloc[-len(data):].values
        
        # 정규화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        
        # 학습 데이터 생성
        X, y = [], []
        for i in range(effective_lookback, len(scaled_data) - forecast_days):
            X.append(scaled_data[i-effective_lookback:i])
            y.append(scaled_data[i:i+forecast_days, 0])  # Close만 예측
        
        if len(X) < 20:
            return None, None
        
        # [안전성] 메모리 제한
        if len(X) > 500:
            X = X[-500:]
            y = y[-500:]
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # [경량화] 작은 모델
        model = SimpleTFT(
            input_size=combined_data.shape[1], 
            hidden_size=32,  # 64→32
            num_heads=2,      # 4→2
            forecast_size=forecast_days
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # [최적화] 배치 학습
        batch_size = min(32, len(X))
        safe_epochs = min(epochs, 15)
        
        model.train()
        for epoch in range(safe_epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                if len(X_batch) == 0:
                    continue
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model, scaler
        
    except Exception as e:
        # TFT 학습 실패
        return None, None


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
    
    # 특징 선택 (기술적 지표 + 패턴 특징)
    feature_cols = ['RSI14', 'MACD', 'StochK14', 'MFI14', 'ATR14',
                    'pattern_bullish', 'pattern_bearish', 'pattern_strength',
                    'pattern_momentum', 'pattern_recency', 'pattern_diversity']
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
    """GRU 모델 (경량화)"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, forecast_size=3):
        super().__init__()
        # [안전성] dropout은 num_layers>1일 때만 사용
        if num_layers > 1:
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        else:
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def train_gru(data, forecast_days=3, lookback=120, epochs=50):
    """GRU 모델 학습 (경량화)"""
    if not TORCH_AVAILABLE:
        return None, None
    
    try:
        # [안전성] 데이터 체크
        if len(data) < lookback + forecast_days + 20:
            return None, None
        
        # [최적화] lookback 축소
        effective_lookback = min(lookback, 60)
        
        # 데이터 정규화
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        # 학습 데이터 생성
        X, y = [], []
        for i in range(effective_lookback, len(scaled_data) - forecast_days):
            X.append(scaled_data[i-effective_lookback:i])
            y.append(scaled_data[i:i+forecast_days].flatten())
        
        if len(X) < 20:
            return None, None
        
        # [안전성] 메모리 제한
        if len(X) > 500:
            X = X[-500:]
            y = y[-500:]
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # [경량화] 작은 모델
        model = GRUModel(
            input_size=1, 
            hidden_size=32,  # 64→32
            num_layers=1,    # 2→1
            forecast_size=forecast_days
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # [최적화] 배치 학습
        batch_size = min(32, len(X))
        safe_epochs = min(epochs, 15)
        
        model.train()
        for epoch in range(safe_epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                if len(X_batch) == 0:
                    continue
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model, scaler
        
    except Exception as e:
        # GRU 학습 실패
        return None, None


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
    
    # 특징 선택 (기술 지표 + 패턴 특징)
    feature_cols = ['RSI14', 'MACD', 'StochK14', 'MFI14', 'ATR14', 'BB_upper', 'BB_lower',
                    'pattern_bullish', 'pattern_bearish', 'pattern_strength',
                    'pattern_momentum', 'pattern_recency', 'pattern_diversity']
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
    
    # Prophet 형식으로 변환 (timezone 제거)
    df_prophet = pd.DataFrame({
        'ds': pd.to_datetime(data.index).tz_localize(None),
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
    """LSTM 모델 (경량화)"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, forecast_size=3):
        super().__init__()
        # [안전성] dropout은 num_layers>1일 때만 사용
        if num_layers > 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(data, forecast_days=3, lookback=120, epochs=50):
    """LSTM 모델 학습 (경량화)"""
    if not TORCH_AVAILABLE:
        return None, None
    
    try:
        # [안전성] 데이터 체크
        if len(data) < lookback + forecast_days + 20:
            return None, None
        
        # [최적화] lookback 축소
        effective_lookback = min(lookback, 60)
        
        # 데이터 정규화
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        # 학습 데이터 생성
        X, y = [], []
        for i in range(effective_lookback, len(scaled_data) - forecast_days):
            X.append(scaled_data[i-effective_lookback:i])
            y.append(scaled_data[i:i+forecast_days].flatten())
        
        if len(X) < 20:
            return None, None
        
        # [안전성] 메모리 제한
        if len(X) > 500:
            X = X[-500:]
            y = y[-500:]
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # [경량화] 작은 모델
        model = LSTMModel(
            input_size=1, 
            hidden_size=32,  # 128→32
            num_layers=1,    # 3→1
            forecast_size=forecast_days
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # [최적화] 배치 학습
        batch_size = min(32, len(X))
        safe_epochs = min(epochs, 15)
        
        model.train()
        for epoch in range(safe_epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                if len(X_batch) == 0:
                    continue
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model, scaler
        
    except Exception as e:
        # LSTM 학습 실패
        return None, None


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
            'lookback': {'nbeats': 60, 'tft': 60, 'xgboost': 60},  # [최적화] 축소
            'epochs': 20,  # [최적화] 30→20
            'description': '초단타 트레이딩 (N-BEATS 40% + TFT 35% + XGBoost 25%)'
        }
    elif interval == '1h':
        # 단기 트레이딩 상단: N-BEATS + TFT + XGBoost (시간봉도 빠른 편)
        return {
            'models': ['nbeats', 'tft', 'xgboost'],
            'weights': [0.40, 0.35, 0.25],
            'lookback': {'nbeats': 60, 'tft': 60, 'xgboost': 60},  # [최적화] 축소
            'epochs': 20,  # [최적화] 40→20
            'description': '시간봉 트레이딩 (N-BEATS 40% + TFT 35% + XGBoost 25%)'
        }
    elif interval == '1d':
        # 단기 트레이딩: LightGBM + GRU + Prophet
        return {
            'models': ['gru', 'lightgbm', 'prophet'],
            'weights': [0.40, 0.35, 0.25],
            'lookback': {'gru': 60, 'lightgbm': 60, 'prophet': None},  # [최적화] 축소
            'epochs': 20,  # [최적화] 50→20
            'description': '일봉 트레이딩 (GRU 40% + LightGBM 35% + Prophet 25%)'
        }
    else:
        # 중기 트레이딩 (주봉 이상): XGBoost + LSTM + Holt-Winters
        return {
            'models': ['lstm', 'xgboost', 'holtwinters'],
            'weights': [0.45, 0.30, 0.25],
            'lookback': {'lstm': 60, 'xgboost': 60, 'holtwinters': None},  # [최적화] 축소
            'epochs': 20,  # [최적화] 50→20
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
    status_text = st.empty()  # 동적 상태 텍스트용
    total_models = len(config['models'])
    
    for idx, model_name in enumerate(config['models']):
        try:
            lookback = config['lookback'].get(model_name, 90)
            epochs = config['epochs']
            
            status_text.text(f"🔄 학습 중: {model_name.upper()} ({idx+1}/{total_models}) - lookback={lookback}, epochs={epochs}")
            
            if model_name == 'nbeats':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_nbeats(data, forecast_days, lookback, epochs)
                    if model is None:
                        st.warning(f"⚠️ {model_name} 학습 실패 (데이터 부족 또는 오류)")
                        models[model_name] = None
                    else:
                        models['nbeats'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'tft':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_tft(data, features_df, forecast_days, lookback, epochs)
                    if model is None:
                        st.warning(f"⚠️ {model_name} 학습 실패")
                        models[model_name] = None
                    else:
                        models['tft'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: XGBoost 미설치")
                    models[model_name] = None
                else:
                    model, metadata = train_xgboost(data, features_df, forecast_days, lookback)
                    if model is None:
                        st.warning(f"⚠️ {model_name} 학습 실패")
                        models[model_name] = None
                    else:
                        models['xgboost'] = {'model': model, 'metadata': metadata}
            
            elif model_name == 'gru':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_gru(data, forecast_days, lookback, epochs)
                    if model is None:
                        st.warning(f"⚠️ {model_name} 학습 실패")
                        models[model_name] = None
                    else:
                        models['gru'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: LightGBM 미설치")
                    models[model_name] = None
                else:
                    model, metadata = train_lightgbm(data, features_df, forecast_days, lookback)
                    if model is None:
                        st.warning(f"⚠️ {model_name} 학습 실패")
                        models[model_name] = None
                    else:
                        models['lightgbm'] = {'model': model, 'metadata': metadata}
            
            elif model_name == 'prophet':
                if not PROPHET_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: Prophet 미설치")
                    models[model_name] = None
                else:
                    model = train_prophet(data, forecast_days)
                    if model is None:
                        st.warning(f"⚠️ {model_name} 학습 실패")
                        models[model_name] = None
                    else:
                        models['prophet'] = {'model': model}
            
            elif model_name == 'lstm':
                if not TORCH_AVAILABLE:
                    st.warning(f"⚠️ {model_name} 사용 불가: PyTorch 미설치")
                    models[model_name] = None
                else:
                    model, scaler = train_lstm(data, forecast_days, lookback, epochs)
                    if model is None:
                        st.warning(f"⚠️ {model_name} 학습 실패")
                        models[model_name] = None
                    else:
                        models['lstm'] = {'model': model, 'scaler': scaler}
            
            elif model_name == 'holtwinters':
                # Holt-Winters는 기존 함수 재사용
                hw_model, seasonality_info, window_size = fit_hw_model_robust(data, max_window=500)
                models['holtwinters'] = {'model': hw_model, 'seasonality': seasonality_info}
            
            progress_bar.progress((idx + 1) / total_models)
        
        except Exception as e:
            st.error(f"❌ {model_name.upper()} 학습 중 오류: {type(e).__name__}")
            st.exception(e)  # 전체 traceback 표시
            models[model_name] = None
    
    progress_bar.empty()
    status_text.empty()
    
    # 학습 결과 요약
    successful_models = [k for k, v in models.items() if v is not None]
    st.success(f"✅ 학습 완료: {len(successful_models)}/{total_models} 모델 성공")
    if successful_models:
        st.info(f"🎯 사용 가능 모델: {', '.join([m.upper() for m in successful_models])}")
    
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
    st.markdown("<div class='section-title'>💰 매도 시점 예측</div>", unsafe_allow_html=True)
    
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
        # RSI 상태 한글 번역
        rsi_korean = {
            'overbought': '과매수',
            'oversold': '과매도',
            'neutral': '중립'
        }
        rsi_status_kr = rsi_korean.get(current_status['rsi_status'], current_status['rsi_status'])
        rsi_color = "🔴" if current_status['rsi_status'] == 'overbought' else "🟢" if current_status['rsi_status'] == 'oversold' else "⚪"
        st.metric(
            label="RSI 상태",
            value=f"{rsi_color} {rsi_status_kr}"
        )
    
    with col4:
        # 추세 한글 번역
        trend_korean = {
            'bullish': '상승',
            'bearish': '하락',
            'neutral': '중립'
        }
        trend_kr = trend_korean.get(current_status['trend'], current_status['trend'])
        trend_color = "📈" if current_status['trend'] == 'bullish' else "📉"
        st.metric(
            label="추세",
            value=f"{trend_color} {trend_kr}"
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
            
            # 시간 예측 포맷팅
            time_minutes = scenario.get('time_estimate_minutes', 0)
            if time_minutes >= 1440:  # 1일 이상
                days = time_minutes // 1440
                hours = (time_minutes % 1440) // 60
                time_str = f"{days}일" if hours == 0 else f"{days}일 {hours}시간"
            elif time_minutes >= 60:  # 1시간 이상
                hours = time_minutes // 60
                minutes = time_minutes % 60
                time_str = f"{hours}시간" if minutes == 0 else f"{hours}시간 {minutes}분"
            else:  # 1시간 미만
                time_str = f"{time_minutes}분"
            
            st.markdown(f"""
                <div class='exit-card'>
                    <div class='exit-title'>{scenario['name']}</div>
                    <table style='width: 100%; color: white; border-collapse: collapse;'>
                        <tr>
                            <td style='width: 33%; padding: 8px 0;'>🎯 익절가: ${scenario['take_profit']:,.2f} (+{profit_pct:.2f}%)</td>
                            <td style='width: 33%; padding: 8px 0;'>🛑 손절가: ${scenario['stop_loss']:,.2f} (-{loss_pct:.2f}%)</td>
                            <td style='width: 34%; padding: 8px 0;'>⏰ 예측 시간: <strong style='color: #ffd700;'>{time_str} 후</strong></td>
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
    
    # [개선됨] v2.9.0.1: 초보자 친화적 증거금 정보 표시
    st.markdown("---")
    st.markdown("### 💳 거래 자금 정보")
    st.caption("📌 레버리지를 사용하면 적은 자금으로 큰 거래가 가능합니다")
    
    position_value = position_size * entry_price
    required_margin = position_value / leverage_info['recommended']
    margin_usage = (required_margin / investment_amount) * 100
    margin_saved = investment_amount - required_margin
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 실제 거래 금액",
            value=f"${position_value:,.2f}",
            help="레버리지를 사용하여 거래하는 총 금액입니다"
        )
    
    with col2:
        st.metric(
            label="💵 필요한 내 돈",
            value=f"${required_margin:,.2f}",
            delta=f"-{((margin_saved) / investment_amount * 100):.1f}% 절약",
            help=f"실제로 내가 내야 하는 돈입니다 ({leverage_info['recommended']}배 레버리지 사용)"
        )
    
    with col3:
        st.metric(
            label="📈 자금 사용률",
            value=f"{margin_usage:.1f}%",
            help="내 투자금 중에서 이번 거래에 쓰는 비율입니다"
        )
    
    with col4:
        st.metric(
            label="💰 남은 자금",
            value=f"${margin_saved:,.2f}",
            delta=f"+{(margin_saved / investment_amount * 100):.1f}%",
            help="다른 거래에 사용할 수 있는 남은 돈입니다"
        )
    
    # 초보자를 위한 쉬운 설명 추가
    with st.expander("💡 레버리지란? (초보자 가이드)"):
        st.markdown(f"""
        **레버리지는 '지렛대'라는 뜻입니다. 적은 돈으로 큰 거래를 하는 방법이에요!**
        
        🎯 **현재 예시:**
        - 실제 거래 금액: **${position_value:,.2f}**
        - 내가 내야 할 돈: **${required_margin:,.2f}**
        - 레버리지: **{leverage_info['recommended']}배**
        
        💡 **쉽게 말하면:**
        - ${required_margin:,.2f}만 있으면 ${position_value:,.2f}어치 거래를 할 수 있어요
        - 나머지 ${margin_saved:,.2f}는 다른 코인에 투자할 수 있어요
        
        ⚠️ **주의사항:**
        - 수익도 {leverage_info['recommended']}배가 되지만, **손실도 {leverage_info['recommended']}배**가 됩니다
        - 손실이 증거금을 넘으면 자동으로 청산(강제 종료)됩니다
        - 처음에는 낮은 레버리지(1-3배)로 시작하는 것을 권장합니다
        """)
    
    # [추가됨] v2.7.2: 리스크 검증 메시지
    st.markdown("---")
    actual_risk_pct = (expected_loss / investment_amount) * 100
    
    if actual_risk_pct > 5.0:
        st.error(f"🚨 경고: 실제 리스크가 {actual_risk_pct:.2f}%로 매우 높습니다. 포지션 크기를 줄이는 것을 권장합니다.")
    elif actual_risk_pct > 3.0:
        st.warning(f"⚠️ 주의: 실제 리스크가 {actual_risk_pct:.2f}%로 높습니다.")
    else:
        st.success(f"✅ 리스크 관리: 실제 리스크가 {actual_risk_pct:.2f}%로 적정 범위 내에 있습니다.")
    
    if rr_ratio >= 3:
        st.success(f"✅ 우수한 RR Ratio ({rr_ratio:.2f}) - 리스크 대비 높은 수익 가능")
    elif rr_ratio >= 2:
        st.info(f"📊 적정한 RR Ratio ({rr_ratio:.2f}) - 균형잡힌 전략")
    else:
        st.warning(f"⚠️ 낮은 RR Ratio ({rr_ratio:.2f}) - 리스크 대비 수익이 작음")


# ════════════════════════════════════════════════════════════════════════════
# v2.8.0: 고급 리스크 관리 UI 함수들
# ════════════════════════════════════════════════════════════════════════════

def render_kelly_analysis(kelly_result: dict, current_position_size: float, 
                         entry_price: float, investment_amount: float):
    """🎲 Kelly Criterion 분석 결과 표시"""
    st.markdown("<div class='section-title'>🎲 Kelly Criterion 분석</div>", unsafe_allow_html=True)
    
    # 기본 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Full Kelly",
            value=f"{kelly_result['kelly_full']:.2%}",
            help="이론적 최적 포지션 크기 (매우 공격적)"
        )
    
    with col2:
        st.metric(
            label="Half Kelly (권장)",
            value=f"{kelly_result['kelly_adjusted']:.2%}",
            help="안정적인 권장 크기 (Full Kelly의 50%)"
        )
    
    with col3:
        st.metric(
            label="최종 권장",
            value=f"{kelly_result['kelly_capped']:.2%}",
            help="최대치 제한 적용 후"
        )
    
    with col4:
        category_emoji = {
            '매우 보수적': '🛡️',
            '중립적': '⚖️',
            '공격적': '🚀',
            '매우 공격적': '🔥',
            '거래 제외': '⛔',
            '기대값 음수': '❌',
            '비정상': '⚠️'
        }
        emoji = category_emoji.get(kelly_result['risk_category'], '📊')
        st.metric(
            label="리스크 카테고리",
            value=f"{emoji} {kelly_result['risk_category']}"
        )
    
    # Kelly 결과 해석
    if kelly_result['recommendation'] == 'TRADE':
        kelly_position_value = investment_amount * kelly_result['kelly_capped']
        kelly_position_size = kelly_position_value / entry_price
        current_position_value = current_position_size * entry_price
        
        st.markdown("---")
        st.markdown("### 📈 Kelly vs 현재 전략 비교")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>
                <h4 style='color: #2e7d32; margin: 0 0 10px 0;'>🎯 Kelly Criterion 권장</h4>
                <p style='margin: 5px 0;'><strong>포지션 크기:</strong> {kelly_position_size:.6f} 코인</p>
                <p style='margin: 5px 0;'><strong>포지션 가치:</strong> ${kelly_position_value:,.2f}</p>
                <p style='margin: 5px 0;'><strong>리스크 비율:</strong> {kelly_result['position_pct']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px;'>
                <h4 style='color: #1565c0; margin: 0 0 10px 0;'>📊 현재 전략 (Fixed 2%)</h4>
                <p style='margin: 5px 0;'><strong>포지션 크기:</strong> {current_position_size:.6f} 코인</p>
                <p style='margin: 5px 0;'><strong>포지션 가치:</strong> ${current_position_value:,.2f}</p>
                <p style='margin: 5px 0;'><strong>리스크 비율:</strong> 2.00%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 차이 분석 (0 나누기 보호)
        if current_position_size > 0:
            diff_pct = ((kelly_position_size - current_position_size) / current_position_size) * 100
            if abs(diff_pct) > 10:
                if diff_pct > 0:
                    st.info(f"📈 Kelly Criterion은 현재보다 **{diff_pct:.1f}% 더 큰** 포지션을 권장합니다. (AI 신뢰도가 높고 RR Ratio가 좋음)")
                else:
                    st.warning(f"📉 Kelly Criterion은 현재보다 **{abs(diff_pct):.1f}% 더 작은** 포지션을 권장합니다. (AI 신뢰도가 낮거나 RR Ratio가 난조함)")
            else:
                st.success("✅ Kelly Criterion과 현재 전략이 유사합니다. (±10% 이내)")
        else:
            st.warning("⚠️ 현재 포지션 크기가 0이어서 비교할 수 없습니다.")
    
    else:
        st.error(f"❌ {kelly_result['reason']}")
        st.warning("⚠️ Kelly Criterion에 따르면 이 거래를 건너뛄는 것이 좋습니다.")
    
    # 상세 정보
    with st.expander("📖 Kelly Criterion 상세 정보"):
        # 안전하게 키 접근
        win_rate_used = kelly_result.get('win_rate_used', 0.5)
        rr_ratio_used = kelly_result.get('rr_ratio_used', 1.0)
        kelly_fraction_used = kelly_result.get('kelly_fraction_used', 0.5)
        
        st.markdown(f"""
        **입력 파라미터:**
        - 승률 (Win Rate): {win_rate_used:.1%}
        - RR Ratio: {rr_ratio_used:.2f}
        - Kelly Fraction: {kelly_fraction_used:.0%} (Half Kelly)
        
        **공식:**
        ```
        Kelly = (b*p - q) / b
        
        여기서:
        - b = RR Ratio (승률)
        - p = 승리 확률 (AI 신뢰도)
        - q = 패배 확률 (1 - p)
        ```
        
        **해석:**
        - Full Kelly는 이론적 최적값이지만 변동성이 큽니다.
        - Half Kelly (50%)는 안정적이면서도 좋은 성과를 냅니다. (권장)
        - Quarter Kelly (25%)는 보수적 접근입니다.
        """)


def render_strategy_comparison(comparison: dict, investment_amount: float):
    """🏆 Position Sizing 전략 비교"""
    st.markdown("<div class='section-title'>🏆 Position Sizing 전략 비교</div>", unsafe_allow_html=True)
    
    strategies = comparison['strategies']
    recommended = comparison['recommended_strategy']
    
    # 비교 표
    st.markdown("### 📊 4가지 전략 비교")
    
    data = []
    for key, strategy in strategies.items():
        is_recommended = (key == recommended)
        emoji = "⭐" if is_recommended else ""
        
        data.append({
            '전략': f"{emoji} {strategy['name']}",
            '리스크 비율': f"{strategy['risk_pct']:.2f}%",
            '포지션 크기': f"{strategy['position_size']:.6f}",
            '포지션 가치': f"${strategy['position_value']:,.0f}",
            '최대 손실': f"${strategy['max_loss']:,.0f}",
            '최대 수익': f"${strategy['max_profit']:,.0f}",
            '필요 증거금': f"${strategy['required_margin']:,.0f}"
        })
    
    import pandas as pd
    df_comparison = pd.DataFrame(data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # 권장 전략
    st.markdown("---")
    recommended_strategy = strategies[recommended]
    
    st.success(f"""
    ⭐ **권장 전략: {recommended_strategy['name']}**
    
    - 포지션 크기: **{recommended_strategy['position_size']:.6f} 코인**
    - 리스크 비율: **{recommended_strategy['risk_pct']:.2f}%**
    - 예상 손실: **${recommended_strategy['max_loss']:,.2f}**
    - 예상 수익: **${recommended_strategy['max_profit']:,.2f}**
    """)
    
    # 전략별 특징
    with st.expander("📚 전략별 특징"):
        st.markdown("""
        **1️⃣ 고정 비율 (Fixed Fractional 2%)**
        - 가장 단순하고 안정적
        - 모든 거래에 동일한 리스크 적용
        - 초보자에게 추천
        
        **2️⃣ Kelly Criterion (Half Kelly)**
        - 수학적 최적화 기반
        - AI 신뢰도와 RR Ratio를 고려
        - 승률이 높을 때 포지션 확대
        - 중급자 이상 추천
        
        **3️⃣ 변동성 조정 (Volatility Adjusted)**
        - 시장 변동성에 따라 자동 조정
        - 변동성 높을 때 포지션 축소
        - 리스크 회피형 트레이더에게 적합
        
        **4️⃣ AI 신뢰도 기반**
        - AI 예측 신뢰도를 직접 반영
        - 신뢰도 높을 때 공격적
        - AI 모델 성능을 신뢰하는 경우
        """)


def render_portfolio_backtest(price_data_df, symbol_name):
    """
    포트폴리오 분석 결과 렌더링 (이미 다운로드된 데이터 사용)
    """
    result = backtest_portfolio_simple(price_data_df, symbol_name)
    
    if result is None:
        st.warning("⚠️ 포트폴리오 분석을 위한 데이터가 부족합니다.")
        return
    
    # 성과 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        return_color = "normal" if result['total_return'] >= 0 else "inverse"
        st.metric(
            label="📈 총 수익률",
            value=f"{result['total_return']:.2f}%",
            delta=f"{result['total_return']:.2f}%",
            delta_color=return_color
        )
    
    with col2:
        st.metric(
            label="📊 Sharpe Ratio",
            value=f"{result['sharpe_ratio']:.3f}",
            help="리스크 조정 수익률 (높을수록 좋음)"
        )
    
    with col3:
        st.metric(
            label="📉 최대 낙폭",
            value=f"{result['max_drawdown']:.2f}%",
            delta=f"{result['max_drawdown']:.2f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="🎯 승률",
            value=f"{result['win_rate']:.1f}%",
            help="양의 수익률을 기록한 날의 비율"
        )
    
    # 포트폴리오 가치 추이 및 코인별 성과 섹션 삭제됨


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

# ═══════════════════════════════════════════════════════════
# 실시간 글로벌 데이터 UI 렌더링 함수 (v2.9.0)
# ═══════════════════════════════════════════════════════════

def render_news_analysis(news_analysis: Dict, news_data: Dict):
    """뉴스 분석 결과 렌더링"""
    st.markdown("### 📡 실시간 글로벌 뉴스 분석")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment = news_analysis['overall_sentiment']
        emoji = "🟢" if sentiment == 'Bullish' else ("🔴" if sentiment == 'Bearish' else "🟡")
        st.metric(
            label="전체 센티먼트",
            value=f"{emoji} {sentiment}",
            help="뉴스 전체의 시장 심리"
        )
    
    with col2:
        confidence = news_analysis['confidence']
        st.metric(
            label="신뢰도",
            value=f"{confidence:.1%}",
            help="센티먼트 분석의 신뢰도"
        )
    
    with col3:
        impact = news_analysis['market_impact']
        impact_emoji = {"High": "🔥", "Medium": "⚖️", "Low": "💤"}
        st.metric(
            label="시장 영향도",
            value=f"{impact_emoji.get(impact, '')} {impact}",
            help="뉴스가 시장에 미치는 영향의 크기"
        )
    
    # 주요 뉴스 표시
    if news_data.get('news'):
        st.markdown("#### 📰 최근 주요 뉴스 (Top 3)")
        for i, news in enumerate(news_data['news'][:3], 1):
            sentiment_emoji = {
                'positive': '👍',
                'negative': '👎',
                'neutral': '😐'
            }
            emoji = sentiment_emoji.get(news['sentiment'], '📰')
            st.markdown(f"{emoji} **[{news['title']}]({news['url']})**")
            st.caption(f"출처: {news['source']} | {news['published_at'][:10]}")
    
    st.markdown(f"**💡 추천:** {news_analysis['recommendation']}")


def render_economic_indicators(fred_data: Dict):
    """경제 지표 렌더링"""
    st.markdown("### 🤖 실시간 경제 지표 (FRED)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="최신 CPI",
            value=f"{fred_data['latest_value']:.2f}",
            help="미국 소비자물가지수 (Consumer Price Index)"
        )
    
    with col2:
        change_mom = fred_data['change_mom']
        color = "🔴" if change_mom > 0 else "🔵"
        st.metric(
            label="MoM 변화",
            value=f"{color} {change_mom:+.2f}%",
            help="전월 대비 변화율 (Month-over-Month)"
        )
    
    with col3:
        change_yoy = fred_data['change_yoy']
        color = "🔴" if change_yoy > 0 else "🔵"
        st.metric(
            label="YoY 변화",
            value=f"{color} {change_yoy:+.2f}%",
            help="전년 대비 변화율 (Year-over-Year)"
        )
    
    with col4:
        trend = fred_data['trend']
        trend_emoji = {"Rising": "📈", "Falling": "📉", "Stable": "➡️"}
        st.metric(
            label="트렌드",
            value=f"{trend_emoji.get(trend, '')} {trend}",
            help="현재 경제 지표 추세"
        )
    
    # 해석
    if trend == 'Rising':
        st.info("📊 인플레이션 상승 중 → 암호화폐 헤지 수요 증가 가능")
    elif trend == 'Falling':
        st.success("📊 인플레이션 하락 중 → 매크로 리스크 감소")
    else:
        st.info("📊 안정적인 경제 환경 유지")


def render_onchain_metrics(dominance_data: Dict, kimchi_data: Dict, funding_data: Dict):
    """온체인 메트릭 렌더링"""
    st.markdown("### 📊 온체인 메트릭스")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🪙 BTC 도미넌스")
        if dominance_data.get('status') == 'success':
            dominance = dominance_data['dominance']
            st.metric(
                label="시가총액 점유율",
                value=f"{dominance:.2f}%",
                help="전체 암호화폐 시가총액 중 비트코인 비율"
            )
            
            if dominance > 45:
                st.success("✅ BTC 강세 → 안정적 시장")
            elif dominance > 40:
                st.info("⚖️ 균형 상태")
            else:
                st.warning("⚠️ 알트코인 시즌 → 변동성 주의")
        else:
            st.error("❌ 데이터 수집 실패")
    
    with col2:
        st.markdown("#### 🇰🇷 김치 프리미엄")
        if kimchi_data.get('status') == 'success':
            premium = kimchi_data['premium']
            st.metric(
                label="한국 vs 글로벌",
                value=f"{premium:+.2f}%",
                help="한국 거래소와 글로벌 거래소의 가격 차이"
            )
            
            if premium > 3:
                st.success(f"✅ 긍정적 프리미엄 → 한국 투자 심리 좋음")
            elif premium < -3:
                st.error(f"⚠️ 네거티브 프리미엄 → 한국 투자 심리 악화")
            else:
                st.info("⚖️ 정상 범위")
        else:
            st.error("❌ 데이터 수집 실패")
    
    with col3:
        st.markdown("#### 💰 펀딩비 (Funding Rate)")
        if funding_data.get('status') == 'success':
            funding = funding_data['funding_rate']
            st.metric(
                label="선물 펀딩비",
                value=f"{funding:+.4f}%",
                help="선물 시장의 롱/숏 균형 지표"
            )
            
            if funding > 0.1:
                st.warning("⚠️ 롱 과열 → 청산 리스크")
            elif funding < -0.05:
                st.info("💡 숏 우세 → 숏 스퀴즈 가능")
            else:
                st.success("✅ 균형 잡힌 상태")
        else:
            st.error("❌ 데이터 수집 실패")


def render_comprehensive_analysis(analysis: Dict):
    """종합 분석 결과 렌더링"""
    st.markdown("### 🎯 종합 시장 분석")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = analysis['overall_score']
        st.metric(
            label="종합 점수",
            value=f"{score:.0f}/100",
            help="모든 지표를 종합한 시장 점수"
        )
        # 점수 바 표시
        bar_length = int(score / 10)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        st.text(bar)
    
    with col2:
        recommendation = analysis['recommendation']
        rec_emoji = {
            "Strong Buy": "💪",
            "Buy": "👍",
            "Hold": "🤝",
            "Sell": "👎",
            "Strong Sell": "🚨"
        }
        rec_color = {
            "Strong Buy": "success",
            "Buy": "info",
            "Hold": "warning",
            "Sell": "warning",
            "Strong Sell": "error"
        }
        
        st.metric(
            label="추천 등급",
            value=f"{rec_emoji.get(recommendation, '')} {recommendation}",
            help="종합 분석 기반 투자 추천"
        )
    
    with col3:
        risk_level = analysis['risk_level']
        risk_emoji = {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Very High": "🔴"
        }
        st.metric(
            label="리스크 레벨",
            value=f"{risk_emoji.get(risk_level, '')} {risk_level}",
            help="현재 시장의 리스크 수준"
        )
    
    st.markdown("# 🚀 설정")
    st.markdown("---")
    
    # v2.6.0: Fear & Greed Index
    st.markdown("### 😱 시장 심리")
    try:
        fg_data = get_fear_greed_index()
        if fg_data:
            current_value = fg_data['current_value']
            classification = fg_data['current_classification']
            
            # 한글 번역 맵
            korean_map = {
                'Extreme Fear': '극도의 공포',
                'Fear': '공포',
                'Neutral': '중립',
                'Greed': '탐욕',
                'Extreme Greed': '극도의 탐욕'
            }
            korean_classification = korean_map.get(classification, classification)
            
            color_map = {
                'Extreme Fear': '#e74c3c',
                'Fear': '#e67e22',
                'Neutral': '#f39c12',
                'Greed': '#2ecc71',
                'Extreme Greed': '#27ae60'
            }
            color = color_map.get(classification, 'gray')
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}aa, {color}); 
                        padding:20px; border-radius:15px; text-align:center; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom:20px;'>
                <h1 style='margin:0; color:white; font-size:48px;'>{current_value}</h1>
                <p style='margin:5px 0 0 0; color:white; font-size:18px; font-weight:bold;'>{korean_classification}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if current_value < 25:
                st.success("🟢 극도의 공포 → 매수 기회")
            elif current_value > 75:
                st.warning("🔴 극도의 탐욕 → 매도 고려")
        else:
            st.info("ℹ️ Fear & Greed 데이터 로딩 중...")
    except Exception as e:
        pass
    
    st.markdown("---")
    
    # TA-Lib 상태 표시
    if TALIB_AVAILABLE:
        st.success("✅ TA-Lib 사용 가능 (61개 패턴)")
    else:
        st.warning("⚠️ TA-Lib 미설치 (기본 3개 패턴)")
    
    st.markdown("## 1️⃣ 시간 선택")
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
        ["기본 목록", "전체 코인 검색 (바이낸스)", "직접 입력"],
        horizontal=True
    )
    
    if coin_input_method == "기본 목록":
        crypto_choice = st.selectbox(
            "💎 암호화폐",
            list(CRYPTO_MAP.keys())
        )
        selected_crypto = CRYPTO_MAP[crypto_choice]
    
    elif coin_input_method == "전체 코인 검색 (바이낸스)":
        # 바이낸스 모든 USDT 페어 가져오기
        with st.spinner("🔎 바이낸스에서 코인 목록을 불러오는 중..."):
            all_pairs = get_all_binance_usdt_pairs()
        
        # 검색 기능 추가
        search_query = st.text_input(
            "🔍 코인 검색",
            value="",
            placeholder="코인 이름 또는 심볼 입력 (예: BTC, 비트코인, SOL)"
        )
        
        # 검색 필터링
        if search_query:
            search_upper = search_query.upper()
            filtered_pairs = [
                pair for pair in all_pairs 
                if search_upper in pair[0].upper() or search_upper in pair[1].upper()
            ]
        else:
            filtered_pairs = all_pairs
        
        if filtered_pairs:
            # 매칭된 코인 수 표시
            st.caption(f"📊 총 {len(filtered_pairs)}개 코인 표시 중 (Binance USDT 페어)")
            
            # selectbox로 선택
            display_names = [pair[0] for pair in filtered_pairs]
            selected_display = st.selectbox(
                "💎 코인 선택",
                display_names,
                key="binance_coin_select"
            )
            
            # 선택된 코인의 심볼 찾기
            for pair in filtered_pairs:
                if pair[0] == selected_display:
                    selected_crypto = pair[1]
                    break
            
            st.success(f"✅ 선택됨: **{selected_crypto}**")
        else:
            st.warning("⚠️ 검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
            selected_crypto = "BTCUSDT"
    
    else:  # "직접 입력"
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
        ["자동", "수동 설정"],
        help="자동 모드는 분해능별 제한을 자동으로 적용합니다"
    )
    
    if period_choice == "자동":
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
        
        # [추가됨] v2.7.2: AI 예측 먼저 실행하여 포지션 타입 결정
        # (AI 예측 코드는 아래에서 실행되지만, 여기서는 임시로 LONG 가정)
        # 실제로는 AI 예측 후 다시 계산해야 함
        position_type = 'LONG'  # 기본값, AI 예측 후 업데이트
        
        # [수정됨] v2.7.2: 롱/숏 구분하여 Stop Loss & Take Profit 계산
        if position_type == 'LONG':
            stop_loss = entry_price - (atr * stop_loss_k)
            take_profit = entry_price + (atr * stop_loss_k * 2)
        else:  # SHORT
            stop_loss = entry_price + (atr * stop_loss_k)
            take_profit = entry_price - (atr * stop_loss_k * 2)
        
        # [추가됨] v2.7.2: 가격 유효성 검증
        if position_type == 'LONG':
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.95  # 5% 아래로 강제 조정
                st.warning("⚠️ Stop Loss가 진입가보다 높아 5% 아래로 조정되었습니다.")
            if take_profit <= entry_price:
                take_profit = entry_price * 1.10  # 10% 위로 강제 조정
        else:  # SHORT
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.05  # 5% 위로 강제 조정
                st.warning("⚠️ Stop Loss가 진입가보다 낮아 5% 위로 조정되었습니다.")
            if take_profit >= entry_price:
                take_profit = entry_price * 0.90  # 10% 아래로 강제 조정
        
        # [수정됨] v2.7.2: Position Size 계산 오류 수정 (CRITICAL FIX)
        # 기존: (risk_amount * leverage) / stop_loss_distance → 레버리지만큼 리스크 증폭 ❌
        # 수정: risk_amount / stop_loss_distance → 레버리지는 증거금에만 영향 ✓
        risk_amount = investment_amount * risk_per_trade_pct
        stop_loss_distance = abs(entry_price - stop_loss)
        
        # [추가됨] v2.7.2: 0 나누기 보호
        if stop_loss_distance < entry_price * 0.001:  # 0.1% 최소값
            stop_loss_distance = entry_price * 0.01  # 1%로 조정
            st.warning("⚠️ Stop Loss 거리가 너무 작아 1%로 조정되었습니다.")
        
        # 올바른 Position Size 공식 (Fixed Fractional Method)
        position_size = risk_amount / stop_loss_distance
        
        # [추가됨] v2.7.2: 필요 증거금 계산
        position_value = position_size * entry_price
        required_margin = position_value / leverage_info['recommended']
        
        # [추가됨] v2.7.2: 증거금 부족 체크
        if required_margin > investment_amount:
            st.error(f"❌ 증거금 부족: ${required_margin:,.2f} 필요 (보유: ${investment_amount:,.2f})")
            # 사용 가능한 최대 포지션으로 조정
            position_size = (investment_amount * leverage_info['recommended']) / entry_price
            position_value = position_size * entry_price
            required_margin = investment_amount
            st.info(f"→ 포지션 크기를 {position_size:.6f} 코인으로 조정합니다.")
        
        rr_ratio = calculate_rr_ratio(entry_price, take_profit, stop_loss)
        
        # 매도 전략 계산 (interval 파라미터 추가)
        exit_strategy = calculate_exit_strategy(df, entry_price, atr, investment_amount, leverage_info['recommended'], interval)
        
        progress_placeholder.empty()
        status_text.empty()
        
        st.success("✅ 분석이 완료되었습니다!")
        
        # 결과 출력
        render_data_summary(df, selected_crypto, interval_name)
        render_ai_forecast(future_df, hw_confidence)
        render_patterns(patterns)
        render_technical_indicators(df)
        # render_validation_results(cv_results)  # 삭제됨
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
        
        # [추가됨] v2.8.0: 고급 리스크 관리 분석
        st.markdown("---")
        st.markdown("<div class='section-title'>🛡️ 고급 리스크 관리 분석</div>", unsafe_allow_html=True)
        
        # 1. Kelly Criterion 분석
        kelly_result = calculate_kelly_criterion(
            ai_confidence=ai_prediction['confidence'],
            rr_ratio=rr_ratio,
            kelly_fraction=0.5  # Half Kelly
        )
        render_kelly_analysis(kelly_result, position_size, entry_price, investment_amount)
        
        st.markdown("---")
        highest_price = df['High'].tail(20).max()  # 최근 20개 캠들 중 최고가
        st.markdown('---')
        st.markdown('<div class="section-title">🌐 실시간 글로벌 시장 데이터</div>', unsafe_allow_html=True)
        
        # API 키 가져오기
        cryptopanic_key = None
        fred_key = None
        try:
            if hasattr(st, 'secrets'):
                cryptopanic_key = st.secrets.get('CRYPTOPANIC_API_KEY')
                fred_key = st.secrets.get('FRED_API_KEY')
        except: pass
        
        with st.spinner('📡 실시간 뉴스 수집...'):
            news_data = fetch_cryptopanic_news(selected_crypto.replace('-USDT',''), cryptopanic_key, 20)
            news_analysis = analyze_news_sentiment_advanced(news_data)
        render_news_analysis(news_data, news_analysis)
        st.markdown('---')
        
        with st.spinner('🌍 경제 지표 수집...'):
            fred_data = fetch_fred_economic_data('CPIAUCSL', fred_key, 12)
        render_economic_indicators(fred_data)
        st.markdown('---')
        
        with st.spinner('⛓️ 온체인 데이터 수집...'):
            dominance_data = fetch_btc_dominance()
            kimchi_data = fetch_kimchi_premium(selected_crypto.replace('-USDT',''))
            funding_data = fetch_funding_rate(selected_crypto)
        render_onchain_metrics(dominance_data, kimchi_data, funding_data)
        st.markdown('---')
        
        comprehensive = analyze_comprehensive_market(
            selected_crypto, news_data, fred_data, dominance_data,
            kimchi_data, funding_data, current_price, ai_prediction['confidence']/100.0
        )
        render_comprehensive_analysis(comprehensive)
        # [v2.9.0] Monte Carlo 시뮬레이션 완전 제거됨
        
        # 4. Position Sizing 전략 비교
        st.markdown("---")
        strategy_comparison = compare_position_sizing_strategies(
            investment_amount=investment_amount,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ai_confidence=ai_prediction['confidence'],
            volatility=volatility,
            leverage=leverage_info['recommended'],
            rr_ratio=rr_ratio
        )
        render_strategy_comparison(strategy_comparison, investment_amount)
        
        # 매도 전략 (신규)
        st.markdown("---")
        render_exit_strategy(exit_strategy, entry_price, investment_amount, leverage_info['recommended'])
        
        # v2.6.0: 포트폴리오 분석 (선택한 코인에 대해 자동 실행)
        st.markdown("---")
        st.markdown("<div class='section-title'>🎯 포트폴리오 분석 (선택 기간별 투자 성과 종합 분석)</div>", unsafe_allow_html=True)
        
        # 선택한 코인에 대해 포트폴리오 분석 자동 실행 (raw_df 사용)
        render_portfolio_backtest(raw_df, selected_crypto)
        
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


# ═══════════════════════════════════════════════════════════════
# v2.9.0: 실시간 데이터 UI 렌더링 함수들
# ═══════════════════════════════════════════════════════════════

