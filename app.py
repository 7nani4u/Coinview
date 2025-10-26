# -*- coding: utf-8 -*-
"""
Streamlit Crypto Forecast App — Revised (No hard sklearn dependency)
- 데이터 소스 토글: yfinance(일봉) / ccxt(거래소 OHLCV)
- 지표 정의 교정: RSI(Wilder), MFI(Typical Price), ATR(Wilder)
- 예측: Holt‑Winters(ETS) ExponentialSmoothing + 계절성(m=7 기본)
- 검증: TimeSeriesSplit(가능 시) 또는 경량 롤링‑오리진 분할(내장 구현)
  * 지표: 방향성 정확도(%) + MASE
- 리스크: ATR 기반 손절/목표가, 위험기반 포지션 크기 계산
- 시각화: Plotly 캔들 + 예측선 + 손절/목표가 라인
- 캐시/예외: st.cache_data, 네트워크/데이터 폴백 처리

면책: 교육/연구용. 실거래 금지. 체결·슬리피지·수수료·호가단위·최소주문금액·레버리지 상한 미반영.
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---- 선택 의존성(없어도 동작) ----
try:  # scikit-learn이 없을 수 있으므로 옵션 처리
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    TimeSeriesSplit = None  # type: ignore
    _HAS_SKLEARN = False

try:
    import ccxt  # type: ignore
    _HAS_CCXT = True
except Exception:
    _HAS_CCXT = False

import requests

# -------------------------------
# 유틸: 시간대 처리
# -------------------------------
_TZ = 'Asia/Seoul'

@st.cache_data(show_spinner=False)
def _tz():
    try:
        import pytz
        return pytz.timezone(_TZ)
    except Exception:
        return None


def _to_local(dt: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    tz = _tz()
    if isinstance(dt, pd.Series):
        idx = dt
    else:
        idx = pd.Series(dt)
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize('UTC')
    idx = idx.dt.tz_convert(_TZ if tz else 'UTC')
    return pd.DatetimeIndex(idx)


# -------------------------------
# 데이터 소스
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_yf(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Adj Close':'AdjClose', 'Volume':'Volume'
    })
    df.index = _to_local(df.index)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df = df[~df.index.duplicated(keep='last')]
    return df


@st.cache_data(show_spinner=False)
def fetch_ccxt(exchange_name: str, market: str, timeframe: str, lookback_days: int = 365*2) -> pd.DataFrame:
    if not _HAS_CCXT:
        return pd.DataFrame()
    ex_name = exchange_name.lower().strip()
    if not hasattr(ccxt, ex_name):
        return pd.DataFrame()
    ex = getattr(ccxt, ex_name)({'enableRateLimit': True})
    if not ex.has.get('fetchOHLCV', False):
        return pd.DataFrame()
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
    limit = None
    try:
        ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, since=since_ms, limit=limit)
    except Exception:
        return pd.DataFrame()
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('Date').drop(columns=['timestamp'])
    df.index = _to_local(df.index)
    df = df[['Open','High','Low','Close','Volume']].astype(float)
    df = df[~df.index.duplicated(keep='last')]
    return df


# -------------------------------
# 지표: Wilder 계열/표준 정의
# -------------------------------

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill')


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    rmf = tp * df['Volume']
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    mr = pos.rolling(period).sum() / (neg.rolling(period).sum().replace(0, np.nan))
    mfi_val = 100 - (100 / (1 + mr))
    return mfi_val.fillna(method='bfill')


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def macd(close: pd.Series, fast:int=12, slow:int=26, signal:int=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -------------------------------
# 경량 TimeSeriesSplit 대체 구현
# -------------------------------
class SimpleTimeSeriesSplit:
    """학습 구간이 확장되는(expanding window) 시계열 분할.
    sklearn.model_selection.TimeSeriesSplit과 비슷한 인덱스 쌍을 생성.
    test_size를 지정하지 않으면 roughly N//(n_splits+1).
    """
    def __init__(self, n_splits: int = 5, test_size: int | None = None, gap: int = 0):
        self.n_splits = int(max(1, n_splits))
        self.test_size = None if test_size is None else int(max(1, test_size))
        self.gap = int(max(0, gap))

    def split(self, X):
        n_samples = len(X)
        ts = self.test_size or max(1, n_samples // (self.n_splits + 1))
        n_folds = self.n_splits
        train_end = n_samples - ts * n_folds
        if train_end <= 0:
            train_end = ts
        for i in range(n_folds):
            start_test = train_end + i * ts
            end_test = start_test + ts
            if end_test > n_samples:
                break
            train_end_idx = max(0, start_test - self.gap)
            train_idx = np.arange(0, train_end_idx)
            test_idx = np.arange(start_test, end_test)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


# -------------------------------
# 예측/검증 지표
# -------------------------------

def mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    if len(y_true) < 2:
        return np.nan
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive if mae_naive != 0 else np.nan


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan
    s_true = np.sign(np.diff(y_true))
    s_pred = np.sign(np.diff(y_pred))
    mask = ~np.isnan(s_true) & ~np.isnan(s_pred)
    if not np.any(mask):
        return np.nan
    return float(np.mean(s_true[mask] == s_pred[mask]) * 100.0)


def rolling_backtest(series: pd.Series, n_splits:int=5, seasonal_periods:int=7) -> dict:
    series = series.dropna()
    if len(series) < max(30, seasonal_periods*3):
        return {"mase": np.nan, "dir_acc": np.nan}

    # sklearn이 있으면 사용, 없으면 경량 분할기로 대체
    if _HAS_SKLEARN and TimeSeriesSplit is not None:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = tscv.split(series)
    else:
        test_size = max(14, seasonal_periods)
        tscv = SimpleTimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        splits = tscv.split(series)

    mase_list, acc_list = [], []
    for tr_idx, te_idx in splits:
        train = series.iloc[tr_idx]
        test = series.iloc[te_idx]
        if len(train) < seasonal_periods*2 or len(test) < max(5, seasonal_periods//2):
            continue
        try:
            model = ExponentialSmoothing(
                train, trend='add', seasonal='add', seasonal_periods=seasonal_periods
            ).fit(optimized=True, use_brute=True)
            pred = model.forecast(len(test))
        except Exception:
            continue
        mase_list.append(mase(test.values, pred.values))
        acc_list.append(directional_accuracy(test.values, pred.values))
    return {
        "mase": float(np.nanmean(mase_list)) if len(mase_list)>0 else np.nan,
        "dir_acc": float(np.nanmean(acc_list)) if len(acc_list)>0 else np.nan,
    }


def fit_forecast(series: pd.Series, horizon:int=30, seasonal_periods:int=7) -> pd.Series:
    series = series.dropna()
    model = ExponentialSmoothing(
        series, trend='add', seasonal='add', seasonal_periods=seasonal_periods
    ).fit(optimized=True, use_brute=True)
    fcst = model.forecast(horizon)
    return pd.Series(fcst, index=pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq='D', tz=series.index.tz))


# -------------------------------
# 포지션/리스크 계산 (ATR 기반)
# -------------------------------

def risk_position(entry: float, stop: float, account: float, risk_pct: float) -> dict:
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return {"qty": 0.0, "notional": 0.0}
    risk_amt = max(0.0, account * max(0.0, risk_pct))
    qty = risk_amt / per_unit_risk
    notional = qty * entry
    return {"qty": float(qty), "notional": float(notional)}


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crypto Forecast (Revised)", layout="wide")

st.title("코인 가격 예측 · 검증 · 리스크 (개정판)")
with st.expander("가정/한계 및 의존성"):
    st.markdown("""
    - RSI(14, Wilder), MFI(14, Typical Price), ATR(14, Wilder) 사용
    - Holt‑Winters(추세/계절 add-add, m=선택). 구조적 레짐변화/이상치/거래소별 가격편차 미반영
    - 거래비용/슬리피지/체결/호가단위/최소주문금액/레버리지 상한/펀딩비용 미반영
    - yfinance 데이터는 지연·조정 가능성, ccxt는 거래소별 제한/간헐적 실패 가능
    - scikit‑learn 미설치 환경에서도 동작하도록 경량 TimeSeriesSplit 내장
    """)

with st.sidebar:
    st.header("데이터 & 기간")
    source_opts = ['yfinance'] + (['ccxt'] if _HAS_CCXT else [])
    data_source = st.radio("데이터 소스", source_opts, index=0, horizontal=False)

    if data_source == 'yfinance':
        symbol = st.text_input("심볼(yfinance)", value="BTC-USD")
        start_date = st.date_input("시작일", value=(datetime.now().date() - timedelta(days=365*2)))
        end_date = st.date_input("종료일", value=datetime.now().date())
        timeframe = '1d'
        ex_name, market = None, None
    else:
        ex_name = st.text_input("거래소(ccxt)", value="binance")
        market = st.text_input("마켓(ccxt)", value="BTC/USDT")
        timeframe = st.selectbox("시간프레임", options=['1d','12h','4h','1h'], index=0)
        lookback_days = st.slider("조회 기간(일)", min_value=180, max_value=1500, value=730, step=30)
        start_date = datetime.now().date() - timedelta(days=lookback_days)
        end_date = datetime.now().date()
        symbol = None

    st.header("예측 & 검증")
    horizon = st.slider("예측 일수", min_value=7, max_value=90, value=30, step=1)
    seasonal_periods = st.selectbox("계절성(periods)", options=[7, 14, 30], index=0, help="주간=7 가정")
    n_splits = st.slider("TSCV 분할 수", min_value=3, max_value=10, value=5)

    st.header("리스크 관리 (ATR)")
    k_sl = st.number_input("손절 배수(×ATR)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    k_targets = st.text_input("목표가 배수 목록(쉼표)", value="1,2,3")
    investment = st.number_input("가정 투자금(USD)", min_value=0.0, value=10000.0, step=100.0)
    risk_pct = st.number_input("거래당 위험 비율(%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0

    run_btn = st.button("분석 실행", type='primary', use_container_width=True)


# 데이터 로드 & 처리
if run_btn:
    if data_source == 'yfinance':
        df = fetch_yf(symbol, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.min.time()))
        data_label = symbol
    else:
        df = fetch_ccxt(ex_name, market, timeframe, lookback_days if 'lookback_days' in locals() else 365*2)
        data_label = f"{ex_name}:{market}:{timeframe}"

    if df is None or df.empty:
        st.error("데이터를 불러오지 못했습니다. 심볼/거래소/마켓/기간을 확인하십시오.")
        st.stop()

    # 지표 계산
    df = df.copy()
    df['RSI14'] = rsi_wilder(df['Close'], 14)
    df['MFI14'] = mfi(df, 14)
    df['ATR14'] = atr_wilder(df, 14)
    macd_line, signal_line, hist = macd(df['Close'])
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = macd_line, signal_line, hist

    # 검증(TSCV)
    metrics = rolling_backtest(df['Close'], n_splits=n_splits, seasonal_periods=seasonal_periods)

    # 최종 예측
    fcst = fit_forecast(df['Close'], horizon=horizon, seasonal_periods=seasonal_periods)
    last_close = float(df['Close'].iloc[-1])

    # 방향/손절/목표가
    direction = 'long' if fcst.iloc[-1] >= last_close else 'short'
    atr = float(df['ATR14'].iloc[-1])
    if np.isnan(atr) or atr == 0:
        st.warning("ATR 계산이 충분하지 않습니다. 기간을 늘려보십시오.")
        atr = float(np.nanmean(df['ATR14'].tail(20))) if not df['ATR14'].tail(20).isna().all() else 0.0

    entry = last_close
    if direction == 'long':
        stop = entry - k_sl * atr
        targets = [entry + float(x.strip()) * atr for x in k_targets.split(',') if x.strip()]
    else:
        stop = entry + k_sl * atr
        targets = [entry - float(x.strip()) * atr for x in k_targets.split(',') if x.strip()]

    pos = risk_position(entry, stop, investment, risk_pct)

    # ------------------- 시각화 -------------------
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'))

    # 예측선
    fig.add_trace(go.Scatter(x=fcst.index, y=fcst.values, mode='lines', name='예측(ETS-HW)'))

    # 손절/목표 라인
    fig.add_hline(y=stop, line_dash='dot', annotation_text=f"손절 {stop:.2f}")
    for i, t in enumerate(targets, start=1):
        fig.add_hline(y=t, line_dash='dash', annotation_text=f"T{i} {t:.2f}")

    fig.update_layout(height=650, xaxis_title='일자', yaxis_title='가격', legend_orientation='h')

    st.subheader("차트")
    st.plotly_chart(fig, use_container_width=True)

    # 지표 표
    with st.expander("지표 스냅샷(최근 5일)"):
        st.dataframe(df[['Close','RSI14','MFI14','ATR14','MACD','MACD_SIGNAL','MACD_HIST']].tail(5))

    # 성능/리스크 요약
    c1, c2, c3 = st.columns(3)
    c1.metric("방향성 정확도(외부표본, %)", f"{metrics['dir_acc']:.2f}" if metrics['dir_acc']==metrics['dir_acc'] else "N/A")
    c2.metric("MASE(↓)", f"{metrics['mase']:.4f}" if metrics['mase']==metrics['mase'] else "N/A")
    c3.metric("계절성(periods)", f"{seasonal_periods}")

    c4, c5, c6 = st.columns(3)
    c4.metric("현재 종가", f"{entry:,.2f}")
    c5.metric("손절가(ATR×{:.1f})".format(k_sl), f"{stop:,.2f}")
    c6.metric("예상 포지션 수량(위험기반)", f"{pos['qty']:.6f}")

    st.caption("* TSCV는 미래 데이터 누수를 방지하기 위한 시계열 전용 교차검증. * MASE는 나이브 모델 대비 오차비율.")

    # 결과 다운로드
    out = pd.DataFrame({'forecast': fcst})
    st.download_button(
        label="예측결과 CSV 다운로드",
        data=out.to_csv(index=True).encode('utf-8'),
        file_name=f"forecast_{data_label.replace(':','_')}.csv",
        mime='text/csv'
    )
else:
    st.info("좌측 설정을 입력한 뒤 '분석 실행'을 클릭하십시오.")
