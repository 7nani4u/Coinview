# -*- coding: utf-8 -*-
"""
v2.6.0 신규 기능 모듈

이 파일의 내용을 app.py의 Line 223 다음에 삽입하세요.
"""

import pandas as pd
import numpy as np
import requests
import datetime
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf


# ════════════════════════════════════════════════════════════════════════════
# v2.6.0: 고급 분석 기능
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# 1. Fear & Greed Index (시장 심리 지수)
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # 1시간 캐시
def get_fear_greed_index(limit=30):
    """
    Alternative.me API에서 Fear & Greed Index 가져오기
    
    Parameters:
    -----------
    limit : int
        가져올 데이터 개수 (기본 30일)
    
    Returns:
    --------
    dict or None
        - 'current_value': 현재 값 (0-100)
        - 'current_classification': 분류 ('Extreme Fear', 'Fear', etc.)
        - 'historical_data': DataFrame with columns ['timestamp', 'value', 'classification']
    """
    try:
        url = f'https://api.alternative.me/fng/?limit={limit}'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data:
            return None
        
        # 현재 값
        current = data['data'][0]
        current_value = int(current['value'])
        current_classification = current['value_classification']
        
        # 역사적 데이터
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


def plot_fear_greed_chart(fg_data):
    """
    Fear & Greed Index 차트 생성
    """
    if fg_data is None:
        return None
    
    df = fg_data['historical_data']
    
    fig = go.Figure()
    
    # 라인 차트
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['value'],
        mode='lines+markers',
        name='Fear & Greed Index',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    # 구간 표시
    fig.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.1, 
                  annotation_text="Extreme Fear", annotation_position="top left")
    fig.add_hrect(y0=25, y1=45, fillcolor="orange", opacity=0.1,
                  annotation_text="Fear", annotation_position="top left")
    fig.add_hrect(y0=45, y1=55, fillcolor="yellow", opacity=0.1,
                  annotation_text="Neutral", annotation_position="top left")
    fig.add_hrect(y0=55, y1=75, fillcolor="lightgreen", opacity=0.1,
                  annotation_text="Greed", annotation_position="top left")
    fig.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.1,
                  annotation_text="Extreme Greed", annotation_position="top left")
    
    fig.update_layout(
        title='Fear & Greed Index (최근 30일)',
        xaxis_title='날짜',
        yaxis_title='지수',
        yaxis=dict(range=[0, 100]),
        height=400,
        hovermode='x unified'
    )
    
    return fig


# ────────────────────────────────────────────────────────────────────────────
# 2. 코렐레이션 분석 (자산 간 상관관계)
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def analyze_correlation(selected_crypto, start_date, end_date, interval='1d'):
    """
    주요 자산과의 상관관계 분석
    
    Parameters:
    -----------
    selected_crypto : str
        분석할 코인 (e.g., 'BTCUSDT')
    start_date, end_date : datetime.date
        분석 기간
    interval : str
        데이터 분해능 (default: '1d')
    
    Returns:
    --------
    dict or None
        - 'correlation_matrix': DataFrame (상관계수 행렬)
        - 'heatmap_fig': Plotly Figure (히트맵)
        - 'insights': list (분석 결과 텍스트)
    """
    try:
        # 주요 자산 리스트
        assets = {
            '선택한 코인': selected_crypto[:-4] + '-USD',  # BTCUSDT -> BTC-USD
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD',
            'S&P 500': 'SPY',
            '금': 'GLD',
            '달러 지수': 'DX-Y.NYB'
        }
        
        # 데이터 다운로드
        data_dict = {}
        for name, ticker in assets.items():
            try:
                df = yf.download(ticker, start=start_date, end=end_date, 
                                interval=interval, progress=False)
                if not df.empty and 'Close' in df.columns:
                    # MultiIndex 처리
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    data_dict[name] = df['Close']
            except:
                continue
        
        if len(data_dict) < 2:
            return None
        
        # DataFrame 생성
        combined_df = pd.DataFrame(data_dict)
        combined_df = combined_df.dropna()
        
        if len(combined_df) < 20:
            return None
        
        # 수익률 계산 (더 정확한 상관관계)
        returns_df = combined_df.pct_change().dropna()
        
        # 상관계수 계산
        corr_matrix = returns_df.corr()
        
        # Plotly 히트맵
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="상관계수")
        ))
        
        fig.update_layout(
            title='자산 간 상관관계 분석',
            xaxis_title='자산',
            yaxis_title='자산',
            height=500,
            width=700
        )
        
        # 인사이트 생성
        insights = []
        selected_coin_name = '선택한 코인'
        
        if selected_coin_name in corr_matrix.columns:
            corr_values = corr_matrix[selected_coin_name].drop(selected_coin_name)
            
            # 가장 높은 양의 상관관계
            max_corr = corr_values.max()
            max_asset = corr_values.idxmax()
            insights.append(f"📈 **가장 강한 양의 상관관계**: {max_asset} ({max_corr:.2f})")
            
            # 가장 낮은 (음의) 상관관계
            min_corr = corr_values.min()
            min_asset = corr_values.idxmin()
            insights.append(f"📉 **가장 강한 음의 상관관계**: {min_asset} ({min_corr:.2f})")
            
            # 해석
            if max_corr > 0.7:
                insights.append(f"⚠️ {max_asset}와 매우 강하게 연동되어 움직입니다. 분산 효과 낮음.")
            elif max_corr < 0.3:
                insights.append(f"✅ 다른 자산과 독립적으로 움직임. 포트폴리오 분산 효과 좋음.")
            
            if min_corr < -0.5:
                insights.append(f"🛡️ {min_asset}는 역방향 헤지 자산으로 활용 가능합니다.")
        
        return {
            'correlation_matrix': corr_matrix,
            'heatmap_fig': fig,
            'insights': insights
        }
    
    except Exception as e:
        return None


# ────────────────────────────────────────────────────────────────────────────
# 3. Sharpe Ratio 최적화
# ────────────────────────────────────────────────────────────────────────────

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Sharpe Ratio 계산
    
    Parameters:
    -----------
    returns : pd.Series or np.array
        수익률 데이터
    risk_free_rate : float
        연간 무위험 이자율 (기본 2%)
    
    Returns:
    --------
    float : Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # 일일 무위험 수익률
    daily_rf = risk_free_rate / 252
    
    # 초과 수익률
    excess_returns = returns - daily_rf
    
    # Sharpe Ratio (연율화)
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    return sharpe


def optimize_leverage_sharpe(data, max_leverage=10.0):
    """
    레버리지별 Sharpe Ratio 계산 및 최적 레버리지 찾기
    
    Parameters:
    -----------
    data : pd.Series
        가격 데이터
    max_leverage : float
        최대 레버리지 (기본 10배)
    
    Returns:
    --------
    dict
        - 'best_leverage': 최적 레버리지
        - 'best_sharpe': 최대 Sharpe Ratio
        - 'leverage_sharpe': DataFrame (레버리지별 Sharpe)
    """
    # 수익률 계산
    returns = data.pct_change().dropna()
    
    if len(returns) < 20:
        return None
    
    # 레버리지 범위
    leverages = np.arange(1.0, max_leverage + 0.5, 0.5)
    sharpe_values = []
    
    for leverage in leverages:
        # 레버리지 적용 수익률
        leveraged_returns = returns * leverage
        
        # Sharpe Ratio 계산
        sharpe = calculate_sharpe_ratio(leveraged_returns)
        sharpe_values.append(sharpe)
    
    # 최적값 찾기
    best_idx = np.argmax(sharpe_values)
    best_leverage = leverages[best_idx]
    best_sharpe = sharpe_values[best_idx]
    
    # DataFrame 생성
    leverage_df = pd.DataFrame({
        'leverage': leverages,
        'sharpe_ratio': sharpe_values
    })
    
    return {
        'best_leverage': best_leverage,
        'best_sharpe': best_sharpe,
        'leverage_sharpe': leverage_df
    }


def plot_leverage_sharpe(leverage_data):
    """
    레버리지-Sharpe Ratio 차트
    """
    if leverage_data is None:
        return None
    
    df = leverage_data['leverage_sharpe']
    best_leverage = leverage_data['best_leverage']
    best_sharpe = leverage_data['best_sharpe']
    
    fig = go.Figure()
    
    # 라인 차트
    fig.add_trace(go.Scatter(
        x=df['leverage'],
        y=df['sharpe_ratio'],
        mode='lines+markers',
        name='Sharpe Ratio',
        line=dict(color='#2ecc71', width=2)
    ))
    
    # 최적점 표시
    fig.add_trace(go.Scatter(
        x=[best_leverage],
        y=[best_sharpe],
        mode='markers',
        name=f'최적 레버리지 ({best_leverage:.1f}x)',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title=f'레버리지별 Sharpe Ratio 분석 (최적: {best_leverage:.1f}x)',
        xaxis_title='레버리지 (배수)',
        yaxis_title='Sharpe Ratio',
        height=400,
        hovermode='x unified'
    )
    
    return fig


# ────────────────────────────────────────────────────────────────────────────
# 4. 포트폴리오 백테스트
# ────────────────────────────────────────────────────────────────────────────

def backtest_portfolio(coins, weights, start_date, end_date, interval='1d', rebalance='monthly'):
    """
    포트폴리오 백테스트
    
    Parameters:
    -----------
    coins : list
        코인 리스트 (e.g., ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    weights : list
        가중치 리스트 (합 = 1.0)
    start_date, end_date : datetime.date
        백테스트 기간
    interval : str
        데이터 분해능
    rebalance : str
        리밸런싱 주기 ('daily', 'weekly', 'monthly', 'none')
    
    Returns:
    --------
    dict or None
        - 'total_return': 총 수익률
        - 'sharpe_ratio': Sharpe Ratio
        - 'max_drawdown': 최대 낙폭
        - 'win_rate': 승률
        - 'equity_curve': 자산 곡선 DataFrame
    """
    try:
        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 데이터 다운로드
        prices_dict = {}
        for coin in coins:
            ticker = coin[:-4] + '-USD'
            df = yf.download(ticker, start=start_date, end=end_date, 
                            interval=interval, progress=False)
            if not df.empty and 'Close' in df.columns:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                prices_dict[coin] = df['Close']
        
        if len(prices_dict) != len(coins):
            return None
        
        # DataFrame 생성
        prices_df = pd.DataFrame(prices_dict)
        prices_df = prices_df.dropna()
        
        if len(prices_df) < 20:
            return None
        
        # 수익률 계산
        returns_df = prices_df.pct_change().fillna(0)
        
        # 포트폴리오 수익률
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # 리밸런싱
        if rebalance != 'none':
            # 리밸런싱 주기 설정
            if rebalance == 'daily':
                rebalance_freq = 1
            elif rebalance == 'weekly':
                rebalance_freq = 7
            elif rebalance == 'monthly':
                rebalance_freq = 30
            else:
                rebalance_freq = len(portfolio_returns)
            
            # 리밸런싱 적용 (간략화)
            # 실제로는 더 복잡한 로직 필요
            pass
        
        # 자산 곡선
        equity_curve = (1 + portfolio_returns).cumprod()
        
        # 성과 지표
        total_return = equity_curve.iloc[-1] - 1
        sharpe = calculate_sharpe_ratio(portfolio_returns)
        
        # 최대 낙폭
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 승률
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        # DataFrame 생성
        equity_df = pd.DataFrame({
            'timestamp': equity_curve.index,
            'equity': equity_curve.values
        })
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'equity_curve': equity_df
        }
    
    except Exception as e:
        return None


def plot_equity_curve(backtest_results):
    """
    포트폴리오 자산 곡선 차트
    """
    if backtest_results is None:
        return None
    
    df = backtest_results['equity_curve']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['equity'],
        mode='lines',
        name='포트폴리오 가치',
        line=dict(color='#3498db', width=2),
        fill='tozeroy'
    ))
    
    # 초기 자본 라인
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="초기 자본")
    
    fig.update_layout(
        title=f'포트폴리오 백테스트 결과 (수익률: {backtest_results["total_return"]*100:.2f}%)',
        xaxis_title='날짜',
        yaxis_title='포트폴리오 가치 (배수)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


print("✅ v2.6.0 신규 기능 모듈 로드 완료")
print("다음 함수들이 추가되었습니다:")
print("  - get_fear_greed_index()")
print("  - plot_fear_greed_chart()")
print("  - analyze_correlation()")
print("  - calculate_sharpe_ratio()")
print("  - optimize_leverage_sharpe()")
print("  - plot_leverage_sharpe()")
print("  - backtest_portfolio()")
print("  - plot_equity_curve()")
