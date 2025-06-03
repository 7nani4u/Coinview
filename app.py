# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import streamlit as st
import streamlit.components.v1 as components  # HTML/CSS 임베드용
import os
import logging
import requests
import statsmodels.api as sm  # Holt-Winters 예측

# ────────────────────────────────────────────────────────────────────────
# 1) Streamlit 페이지 설정 (반드시 최상단에 위치)
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="코인 AI 예측 시스템",
    layout="wide"
)

# ────────────────────────────────────────────────────────────────────────
# 2) CSS: 단락 제목의 반응형 폰트 크기 정의 및 패턴 설명 간격 조정
# ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* 모든 단락 제목에 동일하게 적용할 클래스 */
    .section-title {
        font-size: 32px;
        font-weight: bold;
        margin-top: 24px;
        margin-bottom: 12px;
    }
    /* 화면 너비가 600px 이하(모바일)일 때 폰트 크기를 줄임 */
    @media (max-width: 600px) {
        .section-title {
            font-size: 24px;
        }
    }
    /* 화면 너비가 1200px 이상(데스크톱)일 때 폰트 크기를 더 키울 수 있음 */
    @media (min-width: 1200px) {
        .section-title {
            font-size: 36px;
        }
    }
    /* “캔들스틱 패턴 감지 및 해석” 내 텍스트 줄 간격 및 여백 최소화 */
    .pattern-compact {
        margin: 0;
        line-height: 1.4;
    }
    /* 외부 링크 스타일: 한 줄에 양쪽 정렬 */
    .external-links {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
        margin-bottom: 24px;
    }
    .external-links a {
        font-weight: bold;
        text-decoration: none;
        color: #3498DB;
        margin: 0 8px; /* 간격 조정 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────────────────
# 3) 상수 정의: 레버리지 맵, 상장일 맵 등
# ────────────────────────────────────────────────────────────────────────
max_leverage_map = {
    'BTCUSDT': 125,
    'ETHUSDT': 75,
    # 필요시 여기에 추가
}

listing_date_map = {
    'BTCUSDT': datetime.date(2017, 9, 2),
    'ETHUSDT': datetime.date(2017, 8, 7),
    'BNBUSDT': datetime.date(2017, 7, 25),
    'DOGEUSDT': datetime.date(2019, 4, 6),
    'LTCUSDT': datetime.date(2017, 6, 12),
    'AVAXUSDT': datetime.date(2020, 7, 22),
    'IMXUSDT': datetime.date(2021, 6, 15),
    # 필요시 추가
}

# TensorFlow 경고 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ────────────────────────────────────────────────────────────────────────
# 4) 헬퍼 함수 모음
# ────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_listing_date(symbol: str) -> datetime.date:
    """
    - listing_date_map에 값이 있으면 해당 상장일 반환
    - 없으면 yfinance에서 History(period="max")로 받아서
      DataFrame.index.min()을 상장일로 사용
    - 오류 시 오늘 날짜 반환
    """
    if symbol in listing_date_map:
        return listing_date_map[symbol]
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
    """
    symbol: 'XXXXUSDT' → 'XXXX-USD' 로 변환해서 yfinance 조회
    1) ticker.history() 로 먼저 시도
    2) 비어 있으면 yf.download()로 재시도
    3) Volume 컬럼이 있으면 Volume > 0 필터링
    """
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


def minmax_scale(arr: np.ndarray, data_min: float = None, data_max: float = None):
    """NumPy 기반 MinMax 정규화"""
    if data_min is None:
        data_min = np.nanmin(arr)
    if data_max is None:
        data_max = np.nanmax(arr)
    scaled = (arr - data_min) / (data_max - data_min + 1e-8)
    return scaled, data_min, data_max


def minmax_inverse(scaled: np.ndarray, data_min: float, data_max: float):
    """MinMax 정규화 복원"""
    return scaled * (data_max - data_min + 1e-8) + data_min


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    전달받은 DataFrame에 다음 지표들을 컬럼으로 추가:
    - MA50, EMA50, EMA200
    - 일일 수익률
    - 14일 ATR, 30일 변동성 (σ)
    - RSI(14), Stochastic %K(14)
    - MFI(14)
    - VWAP(당일 기준)
    - Volume 20일 이동평균
    - EMA50/EMA200 교차 시그널
    """
    df = df.copy()
    # (1) 일일 수익률
    df['일일수익률'] = df['Close'].pct_change()

    # (2) 이동평균(MA50), 지수이동평균(EMA50, EMA200)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # (3) 변동성: ATR(14) & 30일 표준편차 σ
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

    # (4) RSI(14)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['RSI14'] = 100 - (100 / (1 + rs))

    # (5) Stochastic %K (14)
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['StochK14'] = (df['Close'] - low14) / (high14 - low14 + 1e-8) * 100

    # (6) MFI(14)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['MF'] = typical_price * df['Volume']
    df['PosMF'] = df['MF'].where(df['Close'] > df['Close'].shift(1), 0)
    df['NegMF'] = df['MF'].where(df['Close'] < df['Close'].shift(1), 0)
    roll_pos = df['PosMF'].rolling(window=14).sum()
    roll_neg = df['NegMF'].rolling(window=14).sum()
    df['MFI14'] = 100 - (100 / (1 + roll_pos / (roll_neg + 1e-8)))

    # (7) VWAP (당일 기준)
    df['PV'] = df['Close'] * df['Volume']
    df['Cum_PV'] = df['PV'].cumsum()
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / (df['Cum_Vol'] + 1e-8)

    # (8) 거래량 20일 이동평균
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

    # (9) EMA50/EMA200 교차 시그널
    df['Cross_Signal'] = 0
    ema50 = df['EMA50']
    ema200 = df['EMA200']
    cond_up = (ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))
    cond_down = (ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))
    df.loc[cond_up, 'Cross_Signal'] = 1
    df.loc[cond_down, 'Cross_Signal'] = -1

    return df.dropna()


def generate_targets(entry_price: float, num_targets: int, direction: str = 'down'):
    """
    진입가를 기준으로 num_targets만큼 등비 형태로 목표가 생성
    direction == 'up' 이면 상승 목표가, 'down'이면 하락 목표가
    """
    targets = []
    for i in range(1, num_targets + 1):
        pct = i / (num_targets + 1)
        if direction == 'down':
            targets.append(entry_price * (1 - pct * 0.02))
        else:
            targets.append(entry_price * (1 + pct * 0.02))
    return targets


# ────────────────────────────────────────────────────────────────────────
# 5) Streamlit 사이드바: 사용자 입력
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 코인 AI 예측 시스템")

    # (1) 암호화폐 심볼 입력
    input_symbol = st.text_input(
        "🔍 암호화폐 심볼 입력 (예: BTC, ETH, DOGE 등)",
        value="",
        help="예: BTC → 자동으로 BTCUSDT로 변환"
    )
    if not input_symbol:
        st.warning("먼저 암호화폐 심볼을 입력해주세요.")
        st.stop()
    base_symbol = input_symbol.strip().upper()
    if not base_symbol.endswith("USDT"):
        selected_crypto = base_symbol + "USDT"
    else:
        selected_crypto = base_symbol

    # (2) TradingView 페이지 유효성 검사
    tv_url_test = f"https://www.tradingview.com/symbols/{selected_crypto}/"
    try:
        tv_resp = requests.get(tv_url_test, timeout=5)
    except Exception:
        tv_resp = None
    if tv_resp is None or tv_resp.status_code != 200:
        st.error(f"❌ TradingView에서 '{selected_crypto}' 페이지를 찾을 수 없습니다. 심볼을 다시 확인해 주세요.")
        st.stop()

    # (3) Yahoo Finance 유효성 검사
    yf_ticker_symbol = selected_crypto[:-4] + "-USD"
    try:
        yf_ticker = yf.Ticker(yf_ticker_symbol)
        df_test = yf_ticker.history(period="1d")
        if df_test is None or df_test.empty:
            raise ValueError
    except Exception:
        st.error(f"❌ Yahoo Finance에서 '{yf_ticker_symbol}' 데이터를 찾을 수 없습니다. 심볼을 다시 확인해 주세요.")
        st.stop()

    # (4) 기간 설정: 자동(상장일→오늘) or 커스텀
    mode = st.radio(
        "🔢 분석 기간 모드 선택",
        options=["자동(상장일 → 오늘)", "직접 선택"],
        index=0
    )
    if mode == "자동(상장일 → 오늘)":
        listing_date = get_listing_date(selected_crypto)
        today = datetime.date.today()
        START = listing_date
        END = today
        st.markdown(f"- **시작일(상장일)**: {START}  \n- **종료일**: {END}")
    else:
        col_s, col_e = st.columns(2)
        with col_s:
            START = st.date_input(
                "시작일 선택",
                value=datetime.date.today() - datetime.timedelta(days=180),
                help="yyyy-mm-dd 형식"
            )
        with col_e:
            END = st.date_input(
                "종료일 선택",
                value=datetime.date.today(),
                help="yyyy-mm-dd 형식"
            )
        if START >= END:
            st.error("❌ 시작일은 종료일 이전이어야 합니다.")
            st.stop()

    # (5) 투자·리스크 설정
    st.markdown("## 2) 투자 및 리스크 설정")
    investment_amount = st.number_input(
        "투자 금액 (USDT)",
        min_value=1.0,
        value=1000.0,
        step=10.0,
        help="해당 종목에 투입할 USDT 금액"
    )
    risk_per_trade_pct = st.slider(
        "리스크 비율 (%)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="한 거래 당 최대 손실 허용 퍼센트"
    ) / 100.0

    stop_loss_k = st.number_input(
        "손절 배수 (σ 기준)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.5,
        help="stop_loss_pct = 변동성(σ) × k"
    )

    # (6) 허용 레버리지 설정: 맵에 있으면 기본값, 없으면 경고 및 직접 입력
    default_max_lev = max_leverage_map.get(selected_crypto, None)
    if default_max_lev is None:
        st.warning("❗ 해당 코인의 최대 레버리지 정보가 없습니다. 직접 값을 입력해 주세요.")
        default_max_lev = 50  # 안내 메시지가 보여지는 동안 기본 렌더링용(사용자가 직접 바꿀 것을 권장)
    leverage_ceiling = st.number_input(
        "허용 최대 레버리지 (직접 설정)",
        min_value=1, max_value=500, value=int(default_max_lev), step=1,
        help="해당 종목에 허용할 최대 레버리지를 설정하세요"
    )

    bt = st.button("🚀 분석 시작", type="primary")

# ────────────────────────────────────────────────────────────────────────
# 6) 메인 로직: 버튼 클릭 시 실행
# ────────────────────────────────────────────────────────────────────────
if bt:
    try:
        # ────────────────────────────────────────────────────────────────────
        # 6-1) 데이터 불러오기 및 간단 오류 처리
        # ────────────────────────────────────────────────────────────────────
        with st.spinner("🔍 데이터 가져오는 중..."):
            raw_df = load_crypto_data(selected_crypto, START, END)
            if raw_df.empty:
                raise ValueError(f"{selected_crypto} 데이터가 없습니다. 심볼/기간을 확인해주세요.")
            if len(raw_df) < 100:
                raise ValueError(f"최소 100 거래일 이상의 데이터가 필요합니다. 현재: {len(raw_df)}일")

        # ────────────────────────────────────────────────────────────────────
        # 6-2) 지표 계산: 모든 보조지표를 컬럼으로 추가
        # ────────────────────────────────────────────────────────────────────
        df = calculate_indicators(raw_df)

        # ────────────────────────────────────────────────────────────────────
        # 6-3) Holt-Winters 모델 학습(In-sample) & 30일 예측 생성
        # ────────────────────────────────────────────────────────────────────
        close_series = df['Close']
        hw_model = sm.tsa.ExponentialSmoothing(
            close_series,
            trend='add',
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)

        # In-sample 예측값(학습 데이터용)
        pred_in_sample = hw_model.fittedvalues

        # 향후 30일 예측
        future_forecast = hw_model.forecast(steps=30)
        last_date = df.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i + 1) for i in range(30)]
        future_df = pd.DataFrame({'예측 종가': future_forecast.values}, index=future_dates)

        # ────────────────────────────────────────────────────────────────────
        # 6-4) 최종 진입/손절/목표가/레버리지 계산
        # ────────────────────────────────────────────────────────────────────
        entry_price = raw_df['Close'].iloc[-1]
        far_price = future_df['예측 종가'].iloc[-1]

        # 최대 손실 금액
        max_loss_amount = investment_amount * risk_per_trade_pct
        stop_loss_pct = df['Volatility30d'].iloc[-1] * stop_loss_k
        per_coin_risk = entry_price * stop_loss_pct if entry_price > 0 else 0

        if per_coin_risk > 0:
            position_qty = max_loss_amount / per_coin_risk
        else:
            position_qty = 0.0

        notional_value = entry_price * position_qty
        recommended_leverage = (notional_value / investment_amount) if investment_amount > 0 else 1.0

        # 실제 허용 상한 = 맵에 정의된 값 vs 사이드바에 입력한 값 중 작은 쪽
        max_allowed = max_leverage_map.get(selected_crypto, leverage_ceiling)
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
            position_signal = "매수 / 롱"
            stop_loss_price = entry_price * (1 - stop_loss_pct)
        else:
            direction = 'down'
            position_signal = "매도 / 숏"
            stop_loss_price = entry_price * (1 + stop_loss_pct)

        targets = generate_targets(entry_price, num_targets, direction=direction)
        primary_target = targets[-1]

        # In-sample 예측 정확도(방향성 기준)
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

        # ============================== [순서 변경 시작] ==============================
        # ────────────────────────────────────────────────────────────────────
        # 6-7) AI 예측 근거
        # ────────────────────────────────────────────────────────────────────
        last_ma50 = df['MA50'].iloc[-1]
        price_trend = "하락세" if entry_price < last_ma50 else "상승세"
        price_trend_colored = (
            f"<span style='color:#E74C3C;font-weight:bold;'>하락세</span>" if price_trend == "하락세"
            else f"<span style='color:#27AE60;font-weight:bold;'>상승세</span>"
        )
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        last_macd = df['MACD'].iloc[-1]
        prev_macd = df['MACD'].iloc[-2]
        macd_trend = "감소세" if last_macd < prev_macd else "증가세"
        macd_trend_colored = f"<span style='color:#3498DB;font-weight:bold;'>{macd_trend}</span>"
        last_rsi = df['RSI14'].iloc[-1]
        last_stoch = df['StochK14'].iloc[-1]
        if last_rsi < 30 and last_stoch < 20:
            momentum_desc = (
                f"모멘텀 지표인 <span style='color:#F1C40F;font-weight:bold;'>RSI({last_rsi:.2f})</span>와 "
                f"<span style='color:#F1C40F;font-weight:bold;'>스토캐스틱({last_stoch:.2f})</span>가 과매도 상태입니다."
            )
            future_trend = "이는 향후 <span style='color:#E74C3C;font-weight:bold;'>하락</span> 지속 가능성을 시사합니다."
        elif last_rsi > 70 and last_stoch > 80:
            momentum_desc = (
                f"모멘텀 지표인 <span style='color:#F1C40F;font-weight:bold;'>RSI({last_rsi:.2f})</span>와 "
                f"<span style='color:#F1C40F;font-weight:bold;'>스토캐스틱({last_stoch:.2f})</span>가 과매수 상태입니다."
            )
            future_trend = "이는 향후 <span style='color:#27AE60;font-weight:bold;'>반등</span> 가능성을 시사합니다."
        else:
            momentum_desc = (
                f"모멘텀 지표인 <span style='color:#F1C40F;font-weight:bold;'>RSI({last_rsi:.2f})</span>와 "
                f"<span style='color:#F1C40F;font-weight:bold;'>스토캐스틱({last_stoch:.2f})</span>가 중립 영역을 유지 중입니다."
            )
            future_trend = "이는 향후 <span style='color:#F1C40F;font-weight:bold;'>횡보</span> 가능성을 시사합니다."

        last_mfi = df['MFI14'].iloc[-1]
        mfi_desc = f"단기적으로는 MFI가 <span style='font-weight:bold;'>{last_mfi:.2f}</span>으로 긍정권이며, 횡보 가능성이 있습니다."

        ai_reason = "<br>".join([
            f"현재 {selected_crypto[:-4]} 가격은 <span style='font-weight:bold;'>{entry_price:.2f}</span>로 {price_trend_colored}이며, MA50 대비 {price_trend}를 보이고 있습니다.",
            f"MACD도 {macd_trend_colored}를 보이며 {'하락' if macd_trend=='감소세' else '상승'} 추세를 형성 중입니다.",
            momentum_desc,
            future_trend,
            mfi_desc
        ])
        st.markdown("<div class='section-title'>🤖 AI 예측 근거</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='line-height:1.6; font-size:16px;'>{ai_reason}</div>", unsafe_allow_html=True)

        # ────────────────────────────────────────────────────────────────────
        # 6-8) 주요 지표 요약
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>📊 주요 지표 요약</div>", unsafe_allow_html=True)
        if price_trend == "하락세":
            card1_bg, card1_icon, card1_text = "#C0392B", "📉", "현재 하락 추세"
        else:
            card1_bg, card1_icon, card1_text = "#27AE60", "📈", "현재 상승 추세"

        if last_rsi < 30 and last_stoch < 20:
            card2_bg, card2_icon, card2_text = "#F1C40F", "⚠️", "과매도 상태"
        elif last_rsi > 70 and last_stoch > 80:
            card2_bg, card2_icon, card2_text = "#F1C40F", "⚠️", "과매수 상태"
        else:
            card2_bg, card2_icon, card2_text = "#95A5A6", "➖", "모멘텀 중립"

        cross_signal = df['Cross_Signal'].iloc[-1]
        if cross_signal == 1:
            card3_bg, card3_icon, card3_text = "#2ECC71", "🔀", "EMA 골든 크로스"
        elif cross_signal == -1:
            card3_bg, card3_icon, card3_text = "#2ECC71", "🔀", "EMA 데드 크로스"
        else:
            if last_macd < prev_macd:
                card3_bg, card3_icon, card3_text = "#E74C3C", "📉", "MACD 감소세"
            else:
                card3_bg, card3_icon, card3_text = "#2ECC71", "📈", "MACD 증가세"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div style='background-color:{card1_bg};padding:12px;border-radius:5px;text-align:center;'>"
                f"<span style='font-size:24px;'>{card1_icon}</span><br>{card1_text}</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div style='background-color:{card2_bg};padding:12px;border-radius:5px;text-align:center;'>"
                f"<span style='font-size:24px;'>{card2_icon}</span><br>{card2_text}</div>",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"<div style='background-color:{card3_bg};padding:12px;border-radius:5px;text-align:center;'>"
                f"<span style='font-size:24px;'>{card3_icon}</span><br>{card3_text}</div>",
                unsafe_allow_html=True
            )

        # ────────────────────────────────────────────────────────────────────
        # 6-12) 추세 예측 히스토리 (5분 간격)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>📈 추세 예측 히스토리</div>", unsafe_allow_html=True)
        now = datetime.datetime.now()
        minute = (now.minute // 5) * 5
        base_time = now.replace(minute=minute, second=0, microsecond=0)
        future_dates_5m = [base_time + datetime.timedelta(minutes=5 * (i + 1)) for i in range(6)]
        hw_forecast_6 = hw_model.forecast(steps=6)
        hist_df_5m = pd.DataFrame({
            'Time_5m': future_dates_5m,
            'Pred_Close_5m': hw_forecast_6.values
        })
        hist_df_5m.set_index('Time_5m', inplace=True)

        comments_list = []
        for i in range(len(hist_df_5m)):
            if i == 0:
                if hw_forecast_6.values[0] < raw_df['Close'].iloc[-1]:
                    comments_list.append("하락 지속 가능성")
                elif hw_forecast_6.values[0] > raw_df['Close'].iloc[-1]:
                    comments_list.append("강세 지속 가능성")
                else:
                    comments_list.append("조정 가능성")
            else:
                prev = hw_forecast_6.values[i - 1]
                curr = hw_forecast_6.values[i]
                if curr < prev:
                    if (prev - curr) > (0.005 * prev):
                        comments_list.append("하락 가능성 증가")
                    else:
                        comments_list.append("하락 지속 가능성")
                elif curr > prev:
                    if (curr - prev) < (0.005 * prev):
                        comments_list.append("상승 조정 가능성")
                    else:
                        comments_list.append("강세 지속 가능성")
                else:
                    comments_list.append("조정 가능성")

        hist_df_5m['Comment'] = comments_list
        output_df = hist_df_5m.copy()
        output_df.index = output_df.index.strftime('%Y-%m-%d %H:%M')
        st.dataframe(
            output_df
            .reset_index()
            .rename(columns={'Time_5m': 'Time', 'Pred_Close_5m': 'Pred_Close'}),
            use_container_width=True
        )

        # ────────────────────────────────────────────────────────────────────
        # 6-9) 변동성 지표 (ATR)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>📉 변동성 지표 (ATR)</div>", unsafe_allow_html=True)
        latest_atr = df['ATR14'].iloc[-1]
        prev_atr = df['ATR14'].iloc[-2] if len(df['ATR14'].dropna()) > 1 else latest_atr
        if latest_atr > prev_atr:
            atr_symbol, atr_trend_text = "🔺", "🔍 ATR이 이전 대비 증가하여 변동성이 확대되었습니다."
        elif latest_atr < prev_atr:
            atr_symbol, atr_trend_text = "🔻", "🔍 ATR이 이전 대비 감소하여 변동성이 축소되었습니다."
        else:
            atr_symbol, atr_trend_text = "⏺️", "🔍 ATR이 이전과 비슷한 수준입니다."

        st.markdown(f"<div style='font-size:20px; font-weight:bold;'>{latest_atr:.2f} {atr_symbol}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px; color:#555;'>{atr_trend_text}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "※ ATR이 높으면 높은 변동성, 낮으면 낮은 변동성을 의미하며, StopLoss·Position Sizing 등 리스크 관리에 주로 활용됩니다。",
            unsafe_allow_html=True
        )

        # ────────────────────────────────────────────────────────────────────
        # 6-10) 지원 및 저항 지표 (원래 상태 유지)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>🛡️ 지원 및 저항 지표</div>", unsafe_allow_html=True)
        fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 1.0]
        high_price = df['High'].max()
        low_price = df['Low'].min()
        diff = high_price - low_price if high_price != low_price else 1e-8
        fib_levels = []
        for ratio in fib_ratios:
            level_price = high_price - diff * ratio
            fib_levels.append({'ratio': ratio, 'price': level_price})

        current_ratio = (high_price - entry_price) / diff
        closest_ratio = min(fib_ratios, key=lambda x: abs(current_ratio - x))

        html_fib = """
<div style="background-color:#111; padding:16px; border-radius:8px;">
  <div style="color:#FFFFFF; font-size:18px; font-weight:bold; margin-bottom:4px;">
    피보나치 되돌림
  </div>
  <div style="color:#AAAAAA; font-size:14px; margin-bottom:12px;">
    주요 지지/저항 레벨
  </div>
  <div style="position:relative; height:180px; border-left:1px solid #555; margin-left:40px;">
"""
        for lvl in fib_levels:
            top_pct = lvl['ratio'] * 100
            price_str = f"{lvl['price']:.2f}"
            perc_str = f"{int(lvl['ratio']*100)}%"
            html_fib += f"""
    <div style="position:absolute; top:{top_pct:.1f}%; left:-40px; width:100%; display:flex; align-items:center;">
      <div style="width:40px; text-align:right; color:#CCCCCC; font-size:12px;">
        {price_str}
      </div>
      <div style="flex:1; border-top:1px solid #555;"></div>
      <div style="width:60px; text-align:right; color:#CCCCCC; font-size:12px; margin-left:4px;">
        {perc_str}
      </div>
    </div>
"""
            if abs(lvl['ratio'] - closest_ratio) < 1e-6:
                html_fib += f"""
    <div style="position:absolute; top:{top_pct:.1f}%; left:40px; width:8px; height:8px;
                background-color:#E74C3C; border-radius:50%; transform:translate(-50%, -50%);"></div>
"""
        html_fib += """
  </div>
</div>
"""
        components.html(html_fib, height=280)

        # ────────────────────────────────────────────────────────────────────
        # 6-11) 캔들스틱 패턴 감지 및 해석 (EMA 교차 패턴 해석 추가)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>🕯️ 캔들스틱 패턴 감지 및 해석</div>", unsafe_allow_html=True)
        patterns = []
        df_sorted = df.sort_index(ascending=True)

        # EMA 교차 패턴 해석 추가
        ema50 = df['EMA50'].iloc[-1]
        ema200 = df['EMA200'].iloc[-1]
        ema50_prev = df['EMA50'].iloc[-2]
        ema200_prev = df['EMA200'].iloc[-2]

        # EMA 교차 패턴 분석
        if ema50 > ema200 and ema50_prev <= ema200_prev:
            patterns.append({
                'name': '골든 크로스',
                'date': df.index[-1],
                'conf': 90.0,  # 신뢰도
                'desc': 'EMA50이 EMA200을 상향 돌파하여 강력한 상승 신호',
                'impact': '장기 상승 추세 전환 가능성 높음'
            })
        elif ema50 < ema200 and ema50_prev >= ema200_prev:
            patterns.append({
                'name': '데드 크로스',
                'date': df.index[-1],
                'conf': 85.0,  # 신뢰도
                'desc': 'EMA50이 EMA200을 하향 돌파하여 강력한 하락 신호',
                'impact': '장기 하락 추세 전환 가능성 높음'
            })
        elif (ema50 - ema200) > (ema50_prev - ema200_prev):
            patterns.append({
                'name': 'EMA 확산',
                'date': df.index[-1],
                'conf': 75.0,
                'desc': 'EMA50과 EMA200 간 거리가 확대되는 상승 추세',
                'impact': '현재 추세 강화 가능성'
            })
        elif (ema50 - ema200) < (ema50_prev - ema200_prev):
            patterns.append({
                'name': 'EMA 수렴',
                'date': df.index[-1],
                'conf': 70.0,
                'desc': 'EMA50과 EMA200 간 거리가 좁아지는 추세 전환 신호',
                'impact': '현재 추세 약화 및 반전 가능성'
            })

        # 기존 캔들스틱 패턴 (Morning Star, Doji, Three White Soldiers) 탐지
        for i in range(2, len(df_sorted)):
            o1, c1, h1, l1 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i - 2]
            o2, c2, h2, l2 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i - 1]
            o3, c3, h3, l3 = df_sorted[['Open', 'Close', 'High', 'Low']].iloc[i]
            date3 = df_sorted.index[i]

            # Three White Soldiers
            if (c1 > o1) and (c2 > o2) and (c3 > o3) and (c2 > c1) and (c3 > c2):
                patterns.append({
                    'name': 'Three White Soldiers',
                    'date': date3,
                    'conf': 100.00,
                    'desc': '세 개의 연속 양봉으로 강력한 상승 신호입니다.'
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
                    'name': 'Morning Star',
                    'date': date3,
                    'conf': round(conf, 2),
                    'desc': '하락 후 작은 몸통 양봉→큰 몸통 양봉으로 반전 신호입니다.'
                })
            # Doji
            if abs(o3 - c3) <= (h3 - l3) * 0.1:
                patterns.append({
                    'name': 'Doji',
                    'date': date3,
                    'conf': 100.00,
                    'desc': '시가와 종가가 비슷한 십자형 캔들입니다. 매수/매도 세력 균형으로 추세 전환 가능성이 있습니다.'
                })

        if not patterns:
            st.write("해당 기간 내 캔들스틱 또는 EMA 교차 패턴이 감지되지 않았습니다.")
        else:
            patterns_sorted = sorted(patterns, key=lambda x: x['date'], reverse=True)
            latest_by_name = {}
            for pat in patterns_sorted:
                nm = pat['name']
                if nm not in latest_by_name:
                    latest_by_name[nm] = pat

            display_order = ['골든 크로스', '데드 크로스', 'EMA 확산', 'EMA 수렴',
                             'Morning Star', 'Doji', 'Three White Soldiers']
            for nm in display_order:
                if nm in latest_by_name:
                    pat = latest_by_name[nm]
                    date_str = pat['date'].strftime("%m-%d")
                    conf = pat.get('conf', None)
                    desc = pat['desc']
                    impact = pat.get('impact', None)
                    if conf is not None:
                        st.markdown(
                            f"<div class='pattern-compact'><strong>{nm}</strong>  –  {date_str}  (신뢰도 {conf:.2f}%)</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='pattern-compact'><strong>{nm}</strong>  –  {date_str}</div>",
                            unsafe_allow_html=True
                        )
                    st.markdown(
                        f"<div class='pattern-compact'>{desc}</div>",
                        unsafe_allow_html=True
                    )
                    if impact:
                        st.markdown(
                            f"<div class='pattern-compact'><span style='color:#F39C12;'>영향: {impact}</span></div>",
                            unsafe_allow_html=True
                        )
                    st.markdown("<br>", unsafe_allow_html=True)

        # ────────────────────────────────────────────────────────────────────
        # 6-13) AI 전략 기반 매수 포지션 요약 및 진입/목표가 추천 결과 (목표가 예측 시간 포함)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>💖 AI 전략 기반 매수 포지션 요약 및 진입/목표가 추천 결과</div>", unsafe_allow_html=True)
        st.markdown(f"""
1) **포지션 신호**: **{position_signal}**  
2) **현재가 (진입가)**: **{entry_price:.4f}** USDT  
3) **투자 금액**: **{investment_amount:,.2f}** USDT  
4) **포지션 수량**: **{position_qty:.4f}** 개  
5) **진입가 범위 (Entry Range)**: **{entry_price*(1 - df['Volatility30d'].iloc[-1]):.8f} – {entry_price*(1 + df['Volatility30d'].iloc[-1]):.8f}** USDT  
6) **손절가 (StopLoss)**: **{stop_loss_price:.4f}** USDT  
7) **최종 목표가 (Primary Target)**: **{primary_target:,.5f}** USDT  

➖ **목표가 목록** ➖  
""", unsafe_allow_html=True)

        # now_dt를 tz-aware로 변경: future_df.index의 timezone을 가져와 동기화
        now_dt = pd.Timestamp.now(tz=future_df.index.tz)

        for idx, tgt in enumerate(targets, start=1):
            # 방향에 따라 달성 예측 날짜 여부 판단
            if direction == 'up':
                cond = future_df['예측 종가'] >= tgt
            else:
                cond = future_df['예측 종가'] <= tgt

            if cond.any():
                # 첫 번째 달성 예상 날짜
                target_date = future_df[cond].index[0]
                delta = target_date - now_dt
            else:
                # 예측 범위(30일) 이후 달성 예상을 위해 마지막 예측 날짜와 현재 시점 차이 계산
                last_forecast_date = future_df.index[-1]
                delta = last_forecast_date - now_dt

            # delta를 월, 일, 시간, 분 단위로 분해
            total_minutes = delta.days * 24 * 60 + delta.seconds // 60
            months = total_minutes // (30 * 24 * 60)
            rem_minutes_after_months = total_minutes % (30 * 24 * 60)
            days = rem_minutes_after_months // (24 * 60)
            rem_minutes_after_days = rem_minutes_after_months % (24 * 60)
            hours = rem_minutes_after_days // 60
            minutes = rem_minutes_after_days % 60

            # 문자열 생성 (0 단위는 생략)
            parts = []
            if months > 0:
                parts.append(f"{months}개월")
            if days > 0:
                parts.append(f"{days}일")
            if hours > 0:
                parts.append(f"{hours}시간")
            if minutes > 0:
                parts.append(f"{minutes}분")
            if not parts:
                parts.append("0분")
            time_str = " ".join(parts)

            st.markdown(f"- 🎯 목표가 {idx}: **{tgt:.5f}** USDT (도달 예상: 약 {time_str})", unsafe_allow_html=True)

        st.markdown(f"""
➖ **AI 전략 (Strategy AI)** ➖  
- 🎰 **승률 (Rate Win)**: **{rate_win:.2f}%**  
- 🧠 **학습 데이터 포인트 수**: **{learned_patterns}개**  

➖ **추천 레버리지**: **{recommended_leverage:.2f}배** (허용 상한: **{ultimate_ceiling}배**)  
""", unsafe_allow_html=True)

        # ────────────────────────────────────────────────────────────────────
        # 6-5) 🔗 외부 링크 출력 (한 줄에 양쪽 정렬, 간격 최소화)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>🔗 외부 링크</div>", unsafe_allow_html=True)
        tv_url = f"https://www.tradingview.com/symbols/{selected_crypto}/"
        yf_url = f"https://finance.yahoo.com/quote/{yf_ticker_symbol}/"
        st.markdown(
            f"<div class='external-links'>"
            f"<a href='{tv_url}' target='_blank'>TradingView에서 보기 ▶</a>"
            f"<a href='{yf_url}' target='_blank'>Yahoo Finance에서 보기 ▶</a>"
            f"</div>",
            unsafe_allow_html=True
        )

        # ────────────────────────────────────────────────────────────────────
        # 6-6) 종가 & 예측 차트
        # ────────────────────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>📈 종가 & 예측 차트</div>", unsafe_allow_html=True)
        plot_df = pd.DataFrame({
            'Close': df['Close'],
            'MA50': df['MA50'],
            'EMA200': df['EMA200'],
        })
        pred_plot = pd.Series(pred_in_sample.values, index=df.index, name='HW_InSample')
        forecast_plot = pd.Series(future_forecast.values, index=future_dates, name='HW_Forecast')

        combined = pd.concat([plot_df, pred_plot, forecast_plot], axis=1)
        st.line_chart(combined, use_container_width=True)

        # ============================== [순서 변경 종료] ==============================

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.markdown("""
**문제 해결 안내**
1. 입력하신 심볼이 TradingView 및 Yahoo Finance에서 유효한지 확인하세요.  
2. `yfinance` 라이브러리를 최신 버전으로 유지 (`pip install --upgrade yfinance`).  
3. 최소 100 거래일 이상의 데이터가 필요합니다.  
4. Streamlit Cloud에서 실행 시 `statsmodels>=0.12.0` 설치 여부를 확인하세요.
""", unsafe_allow_html=True)
        st.stop()

else:
    # 버튼 클릭 전 첫 화면 안내문
    st.markdown("""
<div style='text-align:center'>
    <h1>💎 코인 AI 예측 시스템</h1>
    <p>사이드바에서 “암호화폐 심볼”과 “분석 기간”을 설정한 뒤, 투자/리스크 설정을 완료하고 『🚀 분석 시작』 버튼을 눌러주세요.</p>
</div>
""", unsafe_allow_html=True)
