import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import streamlit as st
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import requests

# ────────────────────────────────────────────────────────────────────────
# 1) 종목별 최대 레버리지 매핑 (Binance USDⓈ-M 선물)
#    → 필요하다면 여기에 특정 심볼별 최대 레버리지 값 지정
# ────────────────────────────────────────────────────────────────────────
max_leverage_map = {
    # 예시: 'BTCUSDT': 125, 'ETHUSDT': 75
}

# ────────────────────────────────────────────────────────────────────────
# 2) Binance 상장일 매핑 (수동 정의 + 기본값 2017-01-01)
#    → 필요하다면 주요 종목만 실제 상장일로 지정
# ────────────────────────────────────────────────────────────────────────
listing_date_map = {
    'BTCUSDT': datetime.date(2017, 9, 2),
    'ETHUSDT': datetime.date(2017, 8, 7),
    'BNBUSDT': datetime.date(2017, 7, 25),
    'DOGEUSDT': datetime.date(2019, 4, 6),
    'LTCUSDT': datetime.date(2017, 6, 12),
    'AVAXUSDT': datetime.date(2020, 7, 22),
    'IMXUSDT': datetime.date(2021, 6, 15),
    # 필요 시 여기에 다른 심볼과 상장일 추가
}

# TensorFlow 및 yfinance 경고 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

st.set_page_config(
    page_title="한글 AI 코인 예측 시스템 (시작일 & 시퀀스 자동)",
    layout="wide"
)

# ────────────────────────────────────────────────────────────────────────
# 3) 유틸 함수: 상장일 조회
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_listing_date(symbol: str) -> datetime.date:
    """
    listing_date_map에 있으면 해당 상장일 반환,
    없으면 yfinance에서 최대 기간 조회 후 최초 날짜 반환.
    오류 시 오늘 날짜 반환.
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

# ────────────────────────────────────────────────────────────────────────
# 4) 유틸 함수: 암호화폐 데이터 불러오기
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_crypto_data(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    1) symbol: 'XXXXUSDT' → 'XXXX-USD' 로 변환
    2) yf.Ticker.history 시도, 실패 시 yf.download() 재시도
    3) Volume이 0인 날 제거 후 반환
    """
    yf_ticker = symbol[:-4] + "-USD"
    df = pd.DataFrame()
    try:
        ticker = yf.Ticker(yf_ticker)
        df_hist = ticker.history(
            start=start, end=end + datetime.timedelta(days=1), interval="1d"
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
    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0].copy()
    return df

# ────────────────────────────────────────────────────────────────────────
# 5) 목표가 생성 함수 (동적 개수)
# ────────────────────────────────────────────────────────────────────────
def generate_targets(entry_price: float, num_targets: int, direction: str = 'down'):
    """
    entry_price: 진입가
    num_targets: 목표가 개수 (1~5)
    direction: 'down'=숏(하락), 'up'=롱(상승)
    pct = i/(num_targets+1) × 2%
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
# 6) 사이드바: 사용자 입력 (심볼 직접 입력)
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 💎 한글 AI 코인 예측 시스템")

    # 6-1) 사용자가 암호화폐 심볼을 직접 입력
    input_symbol = st.text_input(
        "🔍 암호화폐 심볼 입력 (예: BTC, DOGE 등)",
        value="",
        help="예: BTC, DOGE처럼 기초 심볼만 입력해도 USDT 페어로 자동 매핑됩니다."
    )

    if not input_symbol:
        st.warning("먼저 암호화폐 심볼을 입력해주세요.")
        st.stop()

    # 6-2) 입력값 처리: 소문자/대문자 구분 없이,
    # 입력값이 USDT로 끝나지 않으면 자동으로 USDT를 덧붙임
    base_symbol = input_symbol.strip().upper()
    if not base_symbol.endswith("USDT"):
        selected_crypto = base_symbol + "USDT"
    else:
        selected_crypto = base_symbol

    # 6-3) TradingView 페이지 존재 여부 확인
    tv_url_test = f"https://www.tradingview.com/symbols/{selected_crypto}/"
    try:
        tv_resp = requests.get(tv_url_test, timeout=5)
    except Exception:
        tv_resp = None

    if tv_resp is None or tv_resp.status_code != 200:
        st.error(f"❌ TradingView에서 '{selected_crypto}' 페이지를 찾을 수 없습니다.\n"
                 "입력하신 심볼을 다시 확인해 주세요.")
        st.stop()

    # 6-4) 상장일 조회 → START에 자동 할당
    listing_date = get_listing_date(selected_crypto)
    today = datetime.date.today()

    st.markdown("## 2) 분석 기간 설정")
    st.markdown(f"- **시작일**: {listing_date}        ")
    st.markdown(f"- **종료일**: {today}        ")
    START = listing_date
    END = today

    # 6-5) 모델 하이퍼파라미터
    #    과거 시퀀스 길이(N일)는 START/END를 기준으로 자동 계산
    #    3년(≈1095일) 초과 시 1095, 미만 시 상장 이후 전체 기간(일 수)
    total_days = (END - START).days
    if total_days <= 0:
        st.error("❌ 상장일 이후로 데이터가 없습니다. 심볼 또는 기간을 확인하세요.")
        st.stop()

    seq_auto = min(total_days, 1095)
    st.markdown(f"- **자동 설정된 과거 시퀀스 길이 (N일)**: {seq_auto}일 (상장일 기준 최대 3년 또는 보유 기간)")
    sequence_length = seq_auto

    gru_units = st.selectbox(
        "GRU 은닉 유닛 수",
        options=[32, 64, 128],
        index=1,
        help="GRU 레이어의 은닉 유닛 수"
    )
    dropout_rate = st.slider(
        "드롭아웃 비율",
        min_value=0.0, max_value=0.5, value=0.2, step=0.1,
        help="과적합 방지를 위한 드롭아웃 비율"
    )
    learning_rate = st.selectbox(
        "학습률 (Learning Rate)",
        options=[1e-4, 5e-4, 1e-3],
        index=2,
        help="Adam 옵티마이저의 학습률"
    )

    st.markdown("## 3) 투자 및 리스크 설정")
    investment_amount = st.number_input(
        "투자 금액 (USDT)",
        min_value=1.0,
        value=1000.0,
        step=10.0,
        help="해당 종목에 투입할 USDT 금액"
    )
    risk_per_trade_pct = st.slider(
        "리스크 비율 (계정 자산 대비 %)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="한 거래 당 최대 손실 허용 퍼센트 (0.5%~5%)"
    ) / 100.0

    stop_loss_k = st.number_input(
        "손절 배수 (σ 기준)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.5,
        help="stop_loss_pct = 변동성(σ) × k (예: k=2 → ±2σ)"
    )

    default_max_lev = max_leverage_map.get(selected_crypto, 50)
    leverage_ceiling = st.number_input(
        "허용 최대 레버리지",
        min_value=1, max_value=500, value=int(default_max_lev), step=1,
        help="해당 종목에 허용할 최대 레버리지"
    )

    bt = st.button("🚀 분석 시작", type="primary")

# ────────────────────────────────────────────────────────────────────────
# 7) 메인 로직: 예측 및 최적 레버리지 + 목표가 & 롱/숏 & AI 전략 & 진입가 범위
# ────────────────────────────────────────────────────────────────────────
if bt:
    try:
        # 7-1) 데이터 불러오기 및 유효성 검증
        with st.spinner("🔍 데이터 가져오는 중..."):
            raw_df = load_crypto_data(selected_crypto, START, END)
            if raw_df.empty:
                raise ValueError(f"{selected_crypto} 데이터가 없습니다. 심볼 또는 기간을 확인하세요.")

            # 최소 100거래일 이상 데이터가 필요
            if len(raw_df) < 100:
                raise ValueError(f"최소 100 거래일 이상 필요합니다. 현재: {len(raw_df)}일")

            # 시퀀스 길이(sequence_length)보다 데이터가 짧으면 오류
            if len(raw_df) < sequence_length + 1:
                raise ValueError(
                    f"데이터가 너무 짧습니다. 자동 설정된 시퀀스 길이를 줄이거나 기간을 늘려주세요.\n"
                    f"현재 데이터 길이: {len(raw_df)}일, 시퀀스 길이: {sequence_length}일"
                )

        # 7-2) 과거 변동성 계산 (일간 수익률 표준편차)
        with st.spinner("🔧 변동성 계산 중..."):
            raw_df['일일수익률'] = raw_df['Close'].pct_change()
            lookback_vol = min(30, len(raw_df) - 1)
            if lookback_vol > 0:
                recent_returns = raw_df['일일수익률'].dropna().iloc[-lookback_vol:]
                volatility_30d = recent_returns.std()
            else:
                volatility_30d = raw_df['일일수익률'].dropna().std()

        # 7-3) 피처 엔지니어링 및 학습 준비
        with st.spinner("🔧 데이터 전처리 중..."):
            df = raw_df.copy()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['수익률'] = df['Close'].pct_change()

            # BTC 상관계수 계산을 위한 BTC 데이터 로드
            btc_df = load_crypto_data('BTCUSDT', START, END)
            if btc_df.empty:
                raise ValueError("BTCUSDT 데이터 로드 실패: 상관계수 계산 불가")
            btc_df['BTC_수익률'] = btc_df['Close'].pct_change()
            df = df.join(btc_df['BTC_수익률'], how='inner')
            df['BTC_상관계수'] = df['수익률'].rolling(window=sequence_length).corr(df['BTC_수익률'])
            df = df.dropna()

            feature_cols = ['Close', 'Volume', 'MA50', '수익률', 'BTC_상관계수']
            data_features = df[feature_cols].values

            scaler_feat = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler_feat.fit_transform(data_features)

            scaler_close = MinMaxScaler(feature_range=(0, 1))
            scaled_close = scaler_close.fit_transform(df[['Close']].values)

            X, y = [], []
            for i in range(sequence_length, len(scaled_features)):
                X.append(scaled_features[i - sequence_length:i, :])
                y.append(scaled_features[i, 0])
            X = np.array(X)
            y = np.array(y)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

        # 7-4) 모델 구성 및 학습
        with st.spinner("🤖 모델 학습 중..."):
            tf.keras.backend.clear_session()
            model = Sequential([
                GRU(units=gru_units, return_sequences=True, input_shape=(sequence_length, len(feature_cols))),
                Dropout(rate=dropout_rate),
                GRU(units=gru_units // 2, return_sequences=False),
                Dropout(rate=dropout_rate),
                Dense(units=1, activation='linear')
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[es],
                verbose=0
            )

        # 7-5) 예측 및 스케일 복원
        with st.spinner("🔍 예측 결과 생성 중..."):
            train_pred_scaled = model.predict(X_train)
            test_pred_scaled = model.predict(X_test)

            train_pred_close = scaler_close.inverse_transform(
                np.concatenate([
                    train_pred_scaled,
                    np.zeros((train_pred_scaled.shape[0], len(feature_cols) - 1))
                ], axis=1)
            )[:, 0]
            y_train_close = scaler_close.inverse_transform(
                np.concatenate([
                    y_train.reshape(-1, 1),
                    np.zeros((y_train.reshape(-1, 1).shape[0], len(feature_cols) - 1))
                ], axis=1)
            )[:, 0]

            test_pred_close = scaler_close.inverse_transform(
                np.concatenate([
                    test_pred_scaled,
                    np.zeros((test_pred_scaled.shape[0], len(feature_cols) - 1))
                ], axis=1)
            )[:, 0]
            y_test_close = scaler_close.inverse_transform(
                np.concatenate([
                    y_test.reshape(-1, 1),
                    np.zeros((y_test.reshape(-1, 1).shape[0], len(feature_cols) - 1))
                ], axis=1)
            )[:, 0]

            all_dates = df.index[sequence_length:]
            plotdf = pd.DataFrame({
                '실제 종가': np.concatenate([y_train_close.flatten(), y_test_close.flatten()], axis=0),
                '학습 예측 종가': np.concatenate([
                    train_pred_close.flatten(),
                    np.full_like(test_pred_close.flatten(), np.nan)
                ], axis=0),
                '검증 예측 종가': np.concatenate([
                    np.full_like(train_pred_close.flatten(), np.nan),
                    test_pred_close.flatten()
                ], axis=0)
            }, index=all_dates)

        # 7-6) 향후 30일 종가 예측
        with st.spinner("🔮 향후 30일 예측 생성 중..."):
            last_close_scaled = scaled_close[-sequence_length:].reshape(1, sequence_length, 1)
            future_scaled = []

            for _ in range(30):
                dummy = np.zeros((1, sequence_length, len(feature_cols)))
                dummy[0, :, 0] = last_close_scaled.reshape(sequence_length)
                pred_scaled = model.predict(dummy)[0, 0]
                future_scaled.append(pred_scaled)
                last_close_scaled = np.append(
                    last_close_scaled[:, 1:, :],
                    [[[pred_scaled]]],
                    axis=1
                )

            future_prices = scaler_close.inverse_transform(
                np.array(future_scaled).reshape(-1, 1)
            ).flatten()
            last_date = df.index[-1]
            future_dates = [last_date + datetime.timedelta(days=i + 1) for i in range(30)]
            future_df = pd.DataFrame({'예측 종가': future_prices}, index=future_dates)

        # 7-7) 결과 출력: 과거 vs 예측, 학습 손실 등
        st.success("✅ 분석이 완료되었습니다!")

        st.subheader(f"🔎 {selected_crypto} 기본 분석")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**최근 5일 거래 현황 (종가, 거래량)**")
            st.dataframe(raw_df[['Close', 'Volume']].tail(), use_container_width=True)
        with col2:
            st.markdown("**데이터 통계 요약**")
            st.dataframe(df[['Close', 'Volume', 'MA50', '수익률', 'BTC_상관계수']].describe(), use_container_width=True)

        st.subheader("📊 과거 vs 예측 비교")
        st.line_chart(plotdf)

        st.subheader("🔮 향후 30일 종가 예측")
        col_pred, col_info = st.columns([2, 1])
        with col_pred:
            st.line_chart(future_df)
        with col_info:
            st.dataframe(future_df.style.format(precision=5), height=400, use_container_width=True)

        st.subheader("📈 모델 학습 손실 변화")
        history_df = pd.DataFrame({
            '학습 손실': history.history['loss'],
            '검증 손실': history.history['val_loss']
        })
        st.line_chart(history_df)

        # 7-8) 최적 레버리지, 동적 목표가 생성, 롱/숏 신호, AI 전략, 그리고 진입가 범위 계산
        entry_price = raw_df['Close'].iloc[-1]
        far_price = future_df['예측 종가'].iloc[-1]

        max_loss_amount = investment_amount * risk_per_trade_pct
        stop_loss_pct = volatility_30d * stop_loss_k
        per_coin_risk = entry_price * stop_loss_pct

        if per_coin_risk > 0:
            position_qty = max_loss_amount / per_coin_risk
        else:
            position_qty = 0.0

        notional_value = entry_price * position_qty
        if investment_amount > 0:
            recommended_leverage = notional_value / investment_amount
        else:
            recommended_leverage = 1.0

        max_allowed = max_leverage_map.get(selected_crypto, 50)
        ultimate_ceiling = min(max_allowed, leverage_ceiling)
        recommended_leverage = round(max(1.0, min(recommended_leverage, ultimate_ceiling)), 2)

        if entry_price > 0:
            pct_change = abs(far_price - entry_price) / entry_price
        else:
            pct_change = 0.0

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

        if direction == 'up':
            entry_low = entry_price * (1 - volatility_30d)
            entry_high = entry_price
        else:
            entry_low = entry_price
            entry_high = entry_price * (1 + volatility_30d)

        test_start_idx = sequence_length + train_size
        correct_count = 0
        total_count = len(y_test_close)
        for i in range(total_count):
            idx = test_start_idx + i
            prev_price = df['Close'].iloc[idx - 1]
            actual_price = df['Close'].iloc[idx]
            predicted_price = test_pred_close[i]
            actual_dir = 1 if actual_price > prev_price else -1
            pred_dir = 1 if predicted_price > prev_price else -1
            if actual_dir == pred_dir:
                correct_count += 1
        rate_win = round((correct_count / total_count * 100.0) if total_count > 0 else 0.0, 2)
        learned_patterns = len(X_train)

        # 7-9) 최종 출력: 재배치된 순서로 결과 출력
        st.subheader("💖 최적 레버리지 & 목표가 / 롱·숏 / AI 전략 / 진입가 범위 추천 결과")
        st.markdown(f"""
1) **포지션 신호**: {position_signal}  
2) **현재가 (진입가)**: {entry_price:.4f} USDT  
3) **투자 금액**: {investment_amount:,.2f} USDT  
4) **포지션 수량 (코인 수량)**: {position_qty:.4f}  
5) **진입가 범위 (Entry Range)**: {entry_low:.8f} – {entry_high:.8f} USDT  
6) **손절가 (StopLoss)**: {stop_loss_price:.4f} USDT  
7) **주요 목표가 (Primary Target)**: {primary_target:,.5f} USDT  

➖ **목표가 목록** ➖  
""", unsafe_allow_html=True)

        for idx, tgt in enumerate(targets, start=1):
            st.markdown(f"- 🎯 목표가 {idx} : {tgt:.5f} USDT")

        st.markdown(f"""

➖ **AI 전략 (Strategy AI)** ➖  
- 🎰 **승률 (Rate Win)**: {rate_win:.2f}%  
- 🧠 **학습 패턴 수 (Learned Pattern)**: {learned_patterns}개  

➖ **추천 레버리지**: {recommended_leverage:.2f}배 (허용 상한 {ultimate_ceiling}배)  
""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.markdown("""
**문제 해결 안내**  
1. 입력하신 심볼이 TradingView, Yahoo Finance에서 유효한지 확인하세요.  
2. `yfinance` 라이브러리를 최신 버전으로 유지하세요 (`pip install --upgrade yfinance`).  
3. 최소 100 거래일 이상 데이터가 필요합니다.  
4. 자동 설정된 시퀀스(N일)가 데이터보다 클 경우 오류가 발생하므로, 상장일 및 종가 데이터를 다시 확인하세요.
""", unsafe_allow_html=True)
        st.stop()

else:
    st.markdown("""
<div style='text-align:center'>
    <h1>💎 한글 AI 코인 예측 시스템 (시작일 & 시퀀스 자동)</h1>
    <p>사이드바에 “암호화폐 심볼”을 입력하면, TradingView & Yahoo Finance 페이지 유효성을 검사합니다.</p>
    <p>입력값이 USDT로 끝나지 않아도 자동으로 USDT가 붙습니다 (예: DOGE → DOGEUSDT).</p>
    <p>상장일은 Binance 상장일 매핑 혹은 yfinance 정보로 자동 설정됩니다.</p>
    <p>과거 시퀀스 길이(N일)는 “최대 3년(≈1095일) 또는 해당 코인 보유 기간” 중 작은 값으로 자동 계산됩니다.</p>
    <p>종료일(오늘) 이후, 투자 설정 등을 완료한 뒤 ‘🚀 분석 시작’ 버튼을 눌러 예측 결과를 확인하세요.</p>
</div>
""", unsafe_allow_html=True)

st.title("코인 예측 앱")
st.write("이제 Streamlit Community Cloud에 배포됩니다!")
