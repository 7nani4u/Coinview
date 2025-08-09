# -*- coding: utf-8 -*-
"""
한글 AI 코인 예측 시스템 (Streamlit Cloud)
- ETS/Theta 자동 선택 + 예측구간(호환성: summary_frame/Theta prediction_intervals)
- 검증 수치 노출 개선(기본 비노출, 익스팬더에서 상세)
- XGBoost 레버리지: 라벨/피처 정제(sanitize) + 조용한 폴백
"""

import os
import logging
import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
import streamlit as st

# ────────────────────────────────────────────────────────────────────────
# Streamlit 기본 설정
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="한글 AI 코인 예측 (ETS/Theta + XGBoost 레버리지)",
    layout="wide"
)

# ────────────────────────────────────────────────────────────────────────
# keep_alive: 환경변수 ENABLE_KEEP_ALIVE=1일 때만 1회 기동
# ────────────────────────────────────────────────────────────────────────
ENABLE_KEEP_ALIVE = os.getenv("ENABLE_KEEP_ALIVE", "0") == "1"
if ENABLE_KEEP_ALIVE:
    if "keepalive_started" not in st.session_state:
        try:
            from keep_alive import keep_alive
            keep_alive()  # Flask 등 백그라운드 서버 1회 기동
            st.session_state["keepalive_started"] = True
        except Exception as e:
            st.warning(f"keep_alive 시작 실패: {e}")

# 런타임 경고 억제(선택)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ────────────────────────────────────────────────────────────────────────
# 1) 상한/상장일/코인 안전 캡(휴리스틱)
# ────────────────────────────────────────────────────────────────────────
max_leverage_map: Dict[str, int] = {
    # 예: 'BTCUSDT': 125, 'ETHUSDT': 75
}

listing_date_map: Dict[str, datetime.date] = {
    'BTCUSDT': datetime.date(2017, 9, 2),
    'ETHUSDT': datetime.date(2017, 8, 7),
    'BNBUSDT': datetime.date(2017, 7, 25),
    'DOGEUSDT': datetime.date(2019, 4, 6),
    'LTCUSDT': datetime.date(2017, 6, 12),
    'AVAXUSDT': datetime.date(2020, 7, 22),
    'IMXUSDT': datetime.date(2021, 6, 15),
}

coin_safe_cap_map: Dict[str, float] = {
    'BTCUSDT': 5.0, 'ETHUSDT': 7.0, 'BNBUSDT': 7.0, 'LTCUSDT': 6.0,
    'DOGEUSDT': 5.0, 'AVAXUSDT': 6.0, 'IMXUSDT': 5.0,
}

# ────────────────────────────────────────────────────────────────────────
# 2) 유틸/데이터/지표
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_listing_date(symbol: str) -> datetime.date:
    if symbol in listing_date_map:
        return listing_date_map[symbol]
    try:
        yf_symbol = symbol[:-4] + "-USD"
        df_full = yf.Ticker(yf_symbol).history(period="max", interval="1d")
        if df_full is None or df_full.empty:
            return datetime.date.today()
        idx = pd.to_datetime(df_full.index).tz_localize(None)
        return idx.min().date()
    except Exception:
        return datetime.date.today()

@st.cache_data(ttl=86400, show_spinner=False)
def load_crypto_data(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    yf_ticker = symbol[:-4] + "-USD"
    df = pd.DataFrame()

    def _cleanup(_df: pd.DataFrame) -> pd.DataFrame:
        if _df is None or _df.empty:
            return pd.DataFrame()
        _df = _df.copy()
        # 일 단위 고정 빈도 보장 → ETS 예측 안정화(정수 start/end와 궁합)
        _df.index = pd.to_datetime(_df.index).tz_localize(None)
        full_idx = pd.date_range(start=start, end=end, freq="D")
        _df = _df.reindex(full_idx)
        if "Close" in _df.columns:
            _df["Close"] = _df["Close"].ffill()
        if "Volume" in _df.columns:
            _df["Volume"] = _df["Volume"].fillna(0)
        for col in ["High", "Low", "Open"]:
            if col in _df.columns:
                _df[col] = _df[col].fillna(_df["Close"])
        return _df

    try:
        df_hist = yf.Ticker(yf_ticker).history(
            start=start, end=end + datetime.timedelta(days=1), interval="1d"
        )
        if df_hist is not None and not df_hist.empty:
            df = _cleanup(df_hist)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df_dl = yf.download(
                yf_ticker, start=start, end=end + datetime.timedelta(days=1),
                interval="1d", progress=False, threads=False
            )
            if df_dl is not None and not df_dl.empty:
                df = _cleanup(df_dl)
        except Exception:
            df = pd.DataFrame()
    return df if df is not None else pd.DataFrame()

def winsorize_returns(r: pd.Series, p: float = 0.01) -> pd.Series:
    if r.dropna().empty:
        return r
    lo, hi = r.quantile(p), r.quantile(1 - p)
    return r.clip(lower=lo, upper=hi)

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_bbands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower})

def find_pivot_points(series: pd.Series, left: int = 3, right: int = 3, mode: str = "low") -> List[Tuple[pd.Timestamp, float]]:
    pivots = []
    for i in range(left, len(series)-right):
        window = series.iloc[i-left:i+right+1]
        val = series.iloc[i]
        if mode == "low" and val == window.min():
            pivots.append((series.index[i], float(val)))
        if mode == "high" and val == window.max():
            pivots.append((series.index[i], float(val)))
    return pivots

def fit_downtrend_line(high_series: pd.Series, piv_k: int = 5):
    highs = find_pivot_points(high_series, left=3, right=3, mode="high")
    if len(highs) < 2:
        return np.nan, np.nan, pd.DataFrame()
    highs = highs[-piv_k:]
    x = np.arange(len(highs)); y = np.array([h[1] for h in highs], dtype=float)
    a, b = np.polyfit(x, y, 1)
    idx_dates = [h[0] for h in highs]
    line_df = pd.DataFrame({"날짜": idx_dates, "추세값": a*np.arange(len(highs)) + b}).set_index("날짜")
    return float(a), float(b), line_df

# ────────────────────────────────────────────────────────────────────────
# 3) 사이드바 입력
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 💎 한글 AI 코인 예측 시스템")
    base = st.text_input("🔍 암호화폐 심볼 (예: BTC, DOGE)", value="")
    if not base:
        st.warning("먼저 암호화폐 심볼을 입력해주세요.")
        st.stop()
    selected_crypto = base.strip().upper()
    if not selected_crypto.endswith("USDT"):
        selected_crypto += "USDT"

    listing_date = get_listing_date(selected_crypto)
    today = datetime.date.today()
    st.markdown("## 2) 분석 기간 설정")
    st.markdown(f"- **시작일**: {listing_date}")
    st.markdown(f"- **종료일**: {today}")

    START, END = listing_date, today
    if (END - START).days <= 0:
        st.error("❌ 상장일 이후 데이터가 없습니다.")
        st.stop()

    st.markdown("## 3) 예측 옵션")
    model_mode = st.selectbox("모델 선택",
                              ["자동(권장)", "ETS(Holt–Winters)", "Theta", "앙상블(ETS+Theta)"],
                              index=0)
    use_weekly_seasonality = st.checkbox("주간 계절성(7일) 고려", value=False)
    use_damped_trend = st.checkbox("Damped Trend(추세 완화) 사용", value=True)
    use_loglike = st.checkbox("로그(≈Box-Cox) 변환 사용(안정화)", value=True)
    horizon = st.slider("예측 기간(일)", 7, 60, 30, 1)

    st.markdown("## 4) 검증 속도")
    val_mode = st.selectbox("검증 모드", ["빠름", "정밀", "끄기"], index=0)

    st.markdown("## 5) 투자 및 리스크 설정")
    investment_amount = st.number_input("투자 금액 (USDT)", min_value=1.0, value=1000.0, step=10.0)
    risk_per_trade_pct = st.slider("리스크 비율 (%)", 0.5, 5.0, 2.0, 0.5) / 100.0
    stop_loss_k = st.number_input("손절 배수 (σ 기준)", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
    default_max_lev = max_leverage_map.get(selected_crypto, 50)
    leverage_ceiling = st.number_input("허용 최대 레버리지", 1, 500, int(default_max_lev), 1)

    bt = st.button("🚀 분석 시작", type="primary")

# ────────────────────────────────────────────────────────────────────────
# 4) 모델(ETS/Theta) 헬퍼
# ────────────────────────────────────────────────────────────────────────
def _maybe_log(x: pd.Series, enable: bool):
    if enable:
        eps = 1e-9
        return np.log(x + eps), ("log", eps)
    return x, None

def _inv_log(x: pd.Series, info: Any) -> pd.Series:
    if info and info[0] == "log":
        eps = info[1]
        return np.exp(x) - eps
    return x

def fit_ets(series: pd.Series, weekly: bool, damped: bool):
    seasonal = "add" if weekly else None
    seasonal_periods = 7 if weekly else None
    model = ETSModel(endog=series, error="add", trend="add",
                     damped_trend=damped, seasonal=seasonal,
                     seasonal_periods=seasonal_periods)
    res = model.fit()
    return {"model": model, "res": res}

def forecast_ets(res, steps: int, future_index: pd.DatetimeIndex, method: str = "exact", reps: int = 600):
    """
    정수 start/end + summary_frame 기반 예측구간 추출(버전 호환)
    """
    nobs = res.model.nobs
    pred = res.get_prediction(start=nobs, end=nobs + steps - 1,
                              method=method, simulate_repetitions=reps)  # 공식 API. 평균/구간 계산. :contentReference[oaicite:3]{index=3}
    try:
        sf = pred.summary_frame(alpha=0.05)  # mean, mean_ci_lower, mean_ci_upper 제공. :contentReference[oaicite:4]{index=4}
        mean = pd.Series(sf["mean"].values, index=future_index, name="예측 종가")
        ci = pd.DataFrame({"lower y": sf["mean_ci_lower"].values,
                           "upper y": sf["mean_ci_upper"].values}, index=future_index)
    except Exception:
        pm = np.asarray(pred.predicted_mean).ravel()
        mean = pd.Series(pm, index=future_index, name="예측 종가")
        ci = pd.DataFrame({"lower y": np.full_like(pm, np.nan, dtype=float),
                           "upper y": np.full_like(pm, np.nan, dtype=float)}, index=future_index)
    return mean, ci

def fit_theta(series: pd.Series, period: int = None):
    model = ThetaModel(series, period=period)
    res = model.fit()
    return {"model": model, "res": res}

def forecast_theta(res, steps: int, future_index: pd.DatetimeIndex):
    # Theta는 prediction_intervals로 하/상한 제공(공식). :contentReference[oaicite:5]{index=5}
    mean = res.forecast(steps=steps)
    pi = res.prediction_intervals(steps=steps, alpha=0.05)
    ci = pd.DataFrame({"lower y": pi["lower"], "upper y": pi["upper"]}, index=mean.index)
    mean.index = future_index; ci.index = future_index
    return mean.rename("예측 종가"), ci

def rolling_one_step_score(series: pd.Series, fit_func, window_days: int, stride: int) -> float:
    """
    속도 최적화: stride로 건너뛰며 평가, forecast(1)만 사용.
    유효 샘플이 없으면 NaN 반환(과거 1e9% 대신).
    """
    if len(series) < window_days + 10:
        window_days = min(len(series) - 10, 60)
    idx = series.index
    start_pos = len(series) - window_days
    errs = []
    for i in range(start_pos + 1, len(series), stride):
        train = series.iloc[:i]
        try:
            m = fit_func(train)
            yhat = float(m["res"].forecast(steps=1)[0])
        except Exception:
            yhat = np.nan
        y = float(series.iloc[i])
        if not np.isnan(yhat):
            errs.append(abs((y - yhat) / max(abs(y), 1e-8)))
    return float(np.mean(errs) * 100.0) if errs else np.nan

# ────────────────────────────────────────────────────────────────────────
# 5) XGBoost 레버리지(정량회귀) + 폴백
# ────────────────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def sanitize_xy(X: pd.DataFrame, y: np.ndarray, max_clip: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    X, y에서 NaN/Inf 제거 및 합리적 범위로 클리핑.
    X의 어느 열이라도 비유한 값이 있으면 해당 행 제거.
    y는 (1.0, max_clip] 범위로 제한.
    """
    if isinstance(y, pd.Series):
        y = y.values
    # 1) y 정리
    y = y.astype(np.float64)
    y = np.clip(y, 1.0, max_clip)
    mask_y = np.isfinite(y)
    X = X.loc[mask_y]
    y = y[mask_y]
    # 2) X 정리(행 단위)
    Xv = X.values.astype(np.float64)
    mask_x = np.isfinite(Xv).all(axis=1)
    X = X.loc[mask_x]
    y = y[mask_x]
    return X, y.astype(np.float32)

@st.cache_resource(show_spinner=False)
def train_xgb_leverage_model(train_X: pd.DataFrame, train_y: np.ndarray, alphas: np.ndarray):
    """
    XGBoost 2.0+ 정량회귀(QuantileDMatrix + reg:quantileerror).
    버전/환경상 불가하면 MSE 폴백.  :contentReference[oaicite:6]{index=6}
    """
    try:
        Xy = xgb.QuantileDMatrix(train_X.values.astype(np.float32), train_y.astype(np.float32))
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": alphas,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": 42,
        }
        booster = xgb.train(params, Xy, num_boost_round=160, early_stopping_rounds=20, evals=[(Xy, "Train")])
        return ("quantile", booster)
    except Exception:
        dtrain = xgb.DMatrix(train_X.values.astype(np.float32), label=train_y.astype(np.float32))
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": 42,
        }
        booster = xgb.train(params, dtrain, num_boost_round=160, early_stopping_rounds=20, evals=[(dtrain, "Train")])
        return ("mse", booster)

def build_leverage_training_set(df: pd.DataFrame,
                                risk_per_trade_pct: float,
                                exch_cap: float,
                                user_cap: float,
                                base_coin_cap: float) -> Tuple[pd.DataFrame, np.ndarray]:
    eps = 1e-4
    ret = df["Close"].pct_change()
    sigma14 = ret.rolling(14).std().shift(1)
    sigma30 = ret.rolling(30).std().shift(1)
    rsi14 = df["RSI14"].shift(1)
    volr = df["VOL_RATIO"].shift(1)
    corr30 = df["BTC_상관계수"].shift(1)
    bbw = ((df["bb_upper"] - df["bb_lower"]) / (df["Close"] + 1e-9)).shift(1)
    ma_slope = (df["MA20"].pct_change().shift(1)).fillna(0)

    fwd_low3 = df["Low"].shift(-1).rolling(window=3).min()
    mae3 = np.maximum(0.0, (df["Close"] - fwd_low3) / (df["Close"] + eps))
    max_cap = min(exch_cap, user_cap, base_coin_cap)
    y = (risk_per_trade_pct / (mae3 + eps)).clip(1.0, max_cap)

    X = pd.DataFrame({
        "sigma14": sigma14, "sigma30": sigma30,
        "rsi14": rsi14, "vol_ratio": volr,
        "btc_corr30": corr30,
        "bb_width": bbw, "ma20_slope": ma_slope,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    # 라벨을 X 인덱스에 맞춰 정렬 후, sanitize
    y = y.reindex(X.index).values
    X, y = sanitize_xy(X, y, max_clip=max_cap)
    return X, y

def predict_leverage(booster_info, cur_feat: pd.DataFrame, alphas: np.ndarray, caps: Tuple[float,float,float]) -> float:
    mode, booster = booster_info
    exch_cap, user_cap, coin_cap = caps
    if mode == "quantile":
        X_cur = xgb.QuantileDMatrix(cur_feat.values.astype(np.float32))
        scores = booster.inplace_predict(X_cur)  # (n, len(alphas))
        q10 = float(scores[-1, 0])
        q50 = float(scores[-1, 1]) if scores.shape[1] > 1 else q10
        rec = q10 if np.isfinite(q10) else q50
    else:
        dcur = xgb.DMatrix(cur_feat.values.astype(np.float32))
        pred = float(booster.inplace_predict(dcur)[-1])
        rec = pred * 0.7  # 보수적 보정
    return float(np.clip(rec, 1.0, min(exch_cap, user_cap, coin_cap)))

# ────────────────────────────────────────────────────────────────────────
# 6) 메인
# ────────────────────────────────────────────────────────────────────────
if bt:
    try:
        # 데이터/지표
        with st.spinner("🔍 데이터 가져오는 중..."):
            raw_df = load_crypto_data(selected_crypto, START, END)
            if raw_df.empty or "Close" not in raw_df.columns:
                raise ValueError(f"{selected_crypto} 데이터가 없습니다.")
            close = raw_df["Close"].astype(float)
            if close.dropna().shape[0] < 100:
                raise ValueError(f"최소 100 거래일 이상 필요합니다. 현재: {close.dropna().shape[0]}일")

        with st.spinner("🔧 지표 계산 중..."):
            df = raw_df.copy()
            df["일일수익률"] = winsorize_returns(df["Close"].pct_change(), p=0.01)
            lookback = min(30, max(2, len(df) - 1))
            volatility_30d = df["일일수익률"].dropna().iloc[-lookback:].std() if df["일일수익률"].dropna().shape[0] else 0.01
            if not np.isfinite(volatility_30d) or volatility_30d == 0:
                volatility_30d = 0.01

            df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
            df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
            bb = calc_bbands(df["Close"], window=20, num_std=2.0)
            df = df.join(bb)
            df["RSI14"] = calc_rsi(df["Close"], period=14)

            btc = load_crypto_data("BTCUSDT", START, END)
            if btc.empty or "Close" not in btc.columns:
                raise ValueError("BTCUSDT 데이터 로드 실패 (상관계수 계산 불가)")
            btc["BTC_수익률"] = btc["Close"].pct_change()
            df["수익률"] = df["Close"].pct_change()
            df = df.join(btc[["BTC_수익률"]], how="inner")
            df["BTC_상관계수"] = df["수익률"].rolling(window=30).corr(df["BTC_수익률"])
            df = df.dropna()

            df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=1).mean()
            df["VOL_RATIO"] = df["Volume"] / (df["VOL_MA20"] + 1e-9)

            slope, intercept, down_line = fit_downtrend_line(df["High"] if "High" in df.columns else df["Close"], piv_k=5)
            piv_lows = find_pivot_points(df["Low"] if "Low" in df.columns else df["Close"], left=3, right=3, mode="low")
            last_pivot_low = float(piv_lows[-1][1]) if len(piv_lows) else float(df["bb_lower"].iloc[-1])
            support_price = max(last_pivot_low, float(df["bb_lower"].iloc[-1]))

        # 시계열 예측(ETS/Theta)
        series_log, log_info = _maybe_log(df["Close"], enable=use_loglike)
        def _fit_ets_auto(s): return fit_ets(s, weekly=use_weekly_seasonality, damped=use_damped_trend)
        def _fit_ets_no_season(s): return fit_ets(s, weekly=False, damped=use_damped_trend)
        def _fit_theta_auto(s): return fit_theta(s, period=None)

        cand, cand_names = [], []
        if model_mode in ["자동(권장)", "ETS(Holt–Winters)", "앙상블(ETS+Theta)"]:
            cand.append(_fit_ets_auto); cand_names.append("ETS(옵션)")
        if model_mode in ["자동(권장)", "Theta", "앙상블(ETS+Theta)"]:
            cand.append(_fit_theta_auto); cand_names.append("Theta")
        if model_mode == "자동(권장)":
            cand.append(_fit_ets_no_season); cand_names.append("ETS(무계절)")

        # 검증 모드: 기본 비노출(상단 메시지엔 안 붙임)
        if val_mode == "끄기":
            scores = [np.nan] * len(cand_names)
        else:
            win = 60 if val_mode == "빠름" else 120
            step = 5 if val_mode == "빠름" else 1
            with st.spinner(f"🧪 모델 검증(원스텝, window={win}, stride={step}) 중..."):
                scores = [rolling_one_step_score(series_log, f, window_days=win, stride=step) for f in cand]

        with st.spinner("🤖 최종 모델 적합 및 예측 중..."):
            last_date = df.index[-1]
            future_idx = pd.date_range(last_date + datetime.timedelta(days=1), periods=horizon, freq="D")

            def _fit_and_forecast(func, name):
                if "Theta" in name:
                    res = func(series_log)["res"]
                    mean_log, ci = forecast_theta(res, steps=horizon, future_index=future_idx)
                    mean = _inv_log(mean_log, log_info)
                    ci = ci.copy()
                    ci["lower y"] = _inv_log(ci["lower y"], log_info)
                    ci["upper y"] = _inv_log(ci["upper y"], log_info)
                    return {"mean": mean, "ci": ci}
                else:
                    res = func(series_log)["res"]
                    mean_log, ci = forecast_ets(res, steps=horizon, future_index=future_idx, method="exact", reps=600)
                    mean = _inv_log(mean_log, log_info)
                    ci = pd.DataFrame({
                        "lower y": _inv_log(ci["lower y"], log_info),
                        "upper y": _inv_log(ci["upper y"], log_info)
                    }, index=future_idx)
                    return {"mean": mean, "ci": ci}

            if model_mode == "앙상블(ETS+Theta)":
                ets_out = _fit_and_forecast(_fit_ets_auto, "ETS(옵션)")
                th_out = _fit_and_forecast(_fit_theta_auto, "Theta")
                mean = (ets_out["mean"] + th_out["mean"]) / 2.0
                ci = pd.DataFrame({
                    "lower y": (ets_out["ci"]["lower y"] + th_out["ci"]["lower y"]) / 2.0,
                    "upper y": (ets_out["ci"]["upper y"] + th_out["ci"]["upper y"]) / 2.0
                }, index=future_idx)
                best_name = "앙상블(ETS+Theta)"
            elif model_mode == "ETS(Holt–Winters)":
                out = _fit_and_forecast(_fit_ets_auto, "ETS(옵션)")
                mean, ci = out["mean"], out["ci"]; best_name = "ETS(Holt–Winters)"
            elif model_mode == "Theta":
                out = _fit_and_forecast(_fit_theta_auto, "Theta")
                mean, ci = out["mean"], out["ci"]; best_name = "Theta"
            else:
                # 유효 점수만 고려해 베스트 선택, 모두 NaN이면 ETS(옵션)
                finite_scores = np.array([s for s in scores if np.isfinite(s)])
                if finite_scores.size == 0:
                    best_name = "ETS(옵션)"
                    out = _fit_and_forecast(_fit_ets_auto, best_name)
                else:
                    best_idx = int(np.nanargmin(scores))
                    best_name = cand_names[best_idx]
                    out = _fit_and_forecast(cand[best_idx], best_name)
                mean, ci = out["mean"], out["ci"]

        # in-sample 적합값(시각화용)
        with st.spinner("📊 과거 적합 곡선 계산 중..."):
            if "Theta" in best_name:
                res = _fit_theta_auto(series_log)["res"]; fitted_log = res.fittedvalues
            elif "무계절" in best_name:
                res = _fit_ets_no_season(series_log)["res"]; fitted_log = res.fittedvalues
            else:
                res = _fit_ets_auto(series_log)["res"]; fitted_log = res.fittedvalues
            fitted = _inv_log(pd.Series(fitted_log, index=series_log.index), log_info)

        # ✅ 상단 메시지: 요청하신 문구로 변경
        st.success(f"분석이 완료되었습니다 ! (선택 모델 = **{best_name}**)")

        # 필요 시에만 검증 수치 노출
        if val_mode != "끄기":
            with st.expander("검증 상세 보기", expanded=False):
                _rows = []
                for nm, sc in zip(cand_names, scores):
                    _rows.append({"모델": nm, "MAPE(%)": None if not np.isfinite(sc) else round(sc, 2)})
                st.dataframe(pd.DataFrame(_rows), use_container_width=True)

        # ────────────────────────────────────────────────────────────────
        # 🔧 XGBoost: 코인별 레버리지 추천 (정량회귀)
        # ────────────────────────────────────────────────────────────────
        entry_price = float(df["Close"].iloc[-1])
        far_price = float(mean.iloc[-1])
        exp_move = ( (ci["upper y"].iloc[0] - ci["lower y"].iloc[0]) / max(entry_price, 1e-8) / 2.0
                    if np.isfinite(ci["upper y"].iloc[0]) else 0.0 )
        sigma_eff = float(max(winsorize_returns(df["Close"].pct_change(),0.01).rolling(30).std().iloc[-1] or 0.01, exp_move or 0.0))
        exch_cap = max_leverage_map.get(selected_crypto, 50)
        user_cap = int(leverage_ceiling)
        base_cap = coin_safe_cap_map.get(selected_crypto, 6.0)
        vol_factor = float(np.clip(0.02 / max(sigma_eff, 1e-4), 0.4, 1.2))
        coin_cap = max(1.0, round(base_cap * vol_factor, 2))

        X_train, y_train = build_leverage_training_set(df.copy(), risk_per_trade_pct, exch_cap, user_cap, coin_cap)

        cur_feat = pd.DataFrame({
            "sigma14": [df["Close"].pct_change().rolling(14).std().iloc[-2]],
            "sigma30": [df["Close"].pct_change().rolling(30).std().iloc[-2]],
            "rsi14":   [df["RSI14"].iloc[-2]],
            "vol_ratio":[(df["Volume"].iloc[-2] / (df["VOL_MA20"].iloc[-2] + 1e-9))],
            "btc_corr30":[df["BTC_상관계수"].iloc[-2]],
            "bb_width":[((df["bb_upper"].iloc[-2]-df["bb_lower"].iloc[-2])/(df["Close"].iloc[-2]+1e-9))],
            "ma20_slope":[(df["MA20"].pct_change().iloc[-2])],
        }).replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)

        # 충분한 학습 데이터 + XGBoost 사용 가능 시에만 시도
        rec_lev = None
        if XGB_AVAILABLE and len(X_train) >= 200:
            try:
                alphas = np.array([0.10, 0.50, 0.90], dtype=np.float32)
                booster_info = train_xgb_leverage_model(X_train, y_train, alphas)
                rec_lev = predict_leverage(booster_info, cur_feat, alphas, (exch_cap, user_cap, coin_cap))
            except Exception:
                # 스택 트레이스 대신 안내만
                st.info("XGBoost 정량회귀 학습을 생략합니다: 학습 데이터에 결측/비정상 값이 감지되었습니다. (룰 기반 폴백 적용)")
                rec_lev = None

        # 폴백: 룰 기반
        if rec_lev is None or not np.isfinite(rec_lev):
            max_loss_amount = float(investment_amount) * float(risk_per_trade_pct)
            stop_loss_pct = max(float(sigma_eff) * float(stop_loss_k), 0.002)
            per_coin_risk = entry_price * stop_loss_pct if entry_price > 0 else 0.0
            position_qty = (max_loss_amount / per_coin_risk) if per_coin_risk > 0 else 0.0
            notional_value = entry_price * position_qty
            rec_lev = float(np.clip((notional_value / float(investment_amount)) if investment_amount > 0 else 1.0,
                                    1.0, min(exch_cap, user_cap, coin_cap)))

        direction = "up" if far_price > entry_price else "down"
        position_signal = "매수 / 롱" if direction == "up" else "매도 / 숏"

        # 손절/수량/목표가
        max_loss_amount = float(investment_amount) * float(risk_per_trade_pct)
        stop_loss_pct = max(float(sigma_eff) * float(stop_loss_k), 0.002)
        per_coin_risk = entry_price * stop_loss_pct if entry_price > 0 else 0.0
        position_qty = (max_loss_amount / per_coin_risk) if per_coin_risk > 0 else 0.0
        stop_loss_price = entry_price * (1 - stop_loss_pct) if direction == "up" else entry_price * (1 + stop_loss_pct)

        def gen_targets(entry: float, sigma: float, direction: str, k_list=(0.5,1.0,1.5,2.0,3.0)):
            return [(entry + entry*sigma*k) if direction == "up" else (entry - entry*sigma*k) for k in k_list]
        pct_change = abs(far_price - entry_price) / max(entry_price, 1e-8)
        ks = (0.5,1.0,1.5,2.0,3.0) if pct_change >= 0.05 else ((0.5,1.0,1.5) if pct_change >= 0.02 else (1.0,))
        targets = gen_targets(entry_price, sigma_eff, direction, ks)
        primary_target = targets[-1]
        entry_low = entry_price * (1 - sigma_eff) if direction == "up" else entry_price
        entry_high = entry_price if direction == "up" else entry_price * (1 + sigma_eff)

        # 출력
        st.subheader("💖 레버리지·목표가 / 롱·숏 / AI 전략 / 진입가 범위")
        st.markdown(f"""
1) **포지션 신호**: {position_signal}  
2) **현재가 (진입가)**: {entry_price:.4f} USDT  
3) **투자 금액**: {investment_amount:,.2f} USDT  
4) **포지션 수량 (코인 수량)**: {position_qty:.4f}  
5) **진입가 범위 (Entry Range)**: {entry_low:.8f} – {entry_high:.8f} USDT  
6) **손절가 (StopLoss)**: {stop_loss_price:.4f} USDT  
7) **주요 목표가 (Primary Target)**: {primary_target:,.5f} USDT  

➖ **추천 레버리지 (XGBoost·코인별 학습)**: {rec_lev:.2f}배  
""", unsafe_allow_html=True)
        for i, tgt in enumerate(targets, 1):
            st.markdown(f"- 🎯 목표가 {i} : {tgt:.5f} USDT")

        st.info("⚠️ 본 내용은 투자 조언이 아니며 연구/교육용 참고 자료입니다.")

    except Exception as e:
        st.error(f"❌ 오류: {str(e)}")
        st.markdown("""
**문제 해결 안내**  
1) 심볼을 BTC → **BTCUSDT**처럼 입력했는지 확인하세요.  
2) yfinance가 최신인지 점검하세요 (`pip install --upgrade yfinance`).  
3) 최소 100 거래일 이상 데이터가 필요합니다.  
4) `statsmodels`는 0.14+ 권장입니다(`get_prediction` + `summary_frame` 호환).  :contentReference[oaicite:7]{index=7}  
5) `xgboost`는 2.0.0+ 권장입니다(정량회귀 사용 시).  :contentReference[oaicite:8]{index=8}
""", unsafe_allow_html=True)

else:
    # 버튼 클릭 전 첫 화면 안내문
    st.markdown("""
<div style='text-align:center'>
    <h1>💎 코인 AI 예측 시스템</h1>
    <p>사이드바에서 “암호화폐 심볼”과 “분석 기간”을 설정한 뒤, 투자/리스크 설정을 완료하고 『🚀 분석 시작』 버튼을 눌러주세요.</p>
</div>
""", unsafe_allow_html=True)
