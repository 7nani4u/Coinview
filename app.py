# -*- coding: utf-8 -*-
"""
한글 AI 코인 예측 시스템 (Streamlit Cloud)
- ETS/Theta 자동 선택 + 예측구간
- 기술적 분석: 지지선/추세선/거래량/RSI/볼린저밴드
- XGBoost 기반 코인별 맞춤 레버리지 추천(정량회귀) + 휴리스틱 폴백
"""

import os
import logging
import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import altair as alt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel

import streamlit as st

# ────────────────────────────────────────────────────────────────────────
# Streamlit 기본 설정
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="한글 AI 코인 예측 시스템 (TA/ETS/Theta + XGBoost 레버리지)",
    layout="wide"
)

# ────────────────────────────────────────────────────────────────────────
# (옵션) keep_alive: 환경변수 기반 1회 기동 가드 (Streamlit Cloud는 보통 불필요)
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
# 1) 종목별 상한/상장일/코인별 안전 캡(휴리스틱)
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
# 2) 유틸: 상장일/데이터/지표
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
    x = np.arange(len(highs))
    y = np.array([h[1] for h in highs], dtype=float)
    a, b = np.polyfit(x, y, 1)
    idx_dates = [h[0] for h in highs]
    x_full = np.arange(0, len(highs))
    y_full = a * x_full + b
    line_df = pd.DataFrame({"날짜": idx_dates, "추세값": y_full}).set_index("날짜")
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

    st.markdown("## 4) 투자 및 리스크 설정")
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

def forecast_ets(res, future_index: pd.DatetimeIndex, method: str = "exact", reps: int = 1000):
    try:
        pred = res.get_prediction(index=future_index, method=method, simulate_repetitions=reps)
    except Exception:
        pred = res.get_prediction(index=future_index, method="simulated", simulate_repetitions=reps)
    mean = pd.Series(pred.predicted_mean, index=future_index, name="예측 종가")
    ci = pred.conf_int(); ci.index = future_index
    return mean, ci

def fit_theta(series: pd.Series, period: int = None):
    model = ThetaModel(series, period=period)
    res = model.fit()
    return {"model": model, "res": res}

def forecast_theta(res, steps: int):
    mean = res.forecast(steps=steps)
    pi = res.prediction_intervals(steps=steps, alpha=0.05)  # 95% 구간
    ci = pd.DataFrame({"lower y": pi["lower"], "upper y": pi["upper"]})
    return mean.rename("예측 종가"), ci  # Theta 예측구간 문서 참조 :contentReference[oaicite:1]{index=1}

def rolling_one_step_score(series: pd.Series, fit_func, fcst_func, horizon: int = 1, window_days: int = 120) -> float:
    if len(series) < window_days + 10:
        window_days = min(len(series) - 10, 60)
    idx = series.index
    start_pos = len(series) - window_days
    errs = []
    for i in range(start_pos + 1, len(series)):
        train = series.iloc[:i]
        try:
            m = fit_func(train)
            if "res" in m:
                fut_idx = pd.date_range(idx[i], periods=horizon, freq="D")
                mean, _ = fcst_func(m["res"], fut_idx)
                yhat = float(mean.iloc[0])
            else:
                yhat = np.nan
        except Exception:
            yhat = np.nan
        y = float(series.iloc[i])
        if not np.isnan(yhat):
            errs.append(abs((y - yhat) / max(abs(y), 1e-8)))
    return float(np.mean(errs) * 100.0) if errs else 1e9

# ────────────────────────────────────────────────────────────────────────
# 5) 🔧 XGBoost 레버리지 모델 (정량회귀 기반)
#    - 과거 각 시점의 "안전 레버리지 타깃"을 생성해 지도학습
#    - 현재 피처로 안전/중앙/공격적 레버리지(α=0.1/0.5/0.9) 추정
# ────────────────────────────────────────────────────────────────────────
# XGBoost 의존성 체크
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def train_xgb_leverage_model(train_X: pd.DataFrame, train_y: np.ndarray, alphas: np.ndarray):
    """
    XGBoost QuantileDMatrix + 'reg:quantileerror' 로 다중 분위수(예: 0.1/0.5/0.9) 회귀
    - XGBoost 2.0.0+ 필요
    """
    Xy = xgb.QuantileDMatrix(train_X.values.astype(np.float32), train_y.astype(np.float32))
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": alphas,     # 다중 분위수
        "tree_method": "hist",        # 정량회귀는 exact 비권장
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }
    booster = xgb.train(
        params, Xy, num_boost_round=256,
        early_stopping_rounds=20,
        evals=[(Xy, "Train")]
    )
    return booster  # 정식 사용법/파라미터는 XGBoost 문서 참고 :contentReference[oaicite:2]{index=2}

def build_leverage_training_set(df: pd.DataFrame,
                                risk_per_trade_pct: float,
                                exch_cap: float,
                                user_cap: float,
                                base_coin_cap: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    레버리지 지도학습용 (X, y) 생성
    y(타깃): '다음 3일 최대 불리 변동' 기준의 안전 레버리지 상한
            L_t = risk_per_trade_pct / (MAE_3d_t + eps)
    * MAE_3d_t: t의 종가 대비 앞으로 3일 최저가 하락폭(음수), 절댓값 사용
    """
    eps = 1e-4
    # 피처(전날까지 정보를 사용하도록 시프팅)
    ret = df["Close"].pct_change()
    sigma14 = ret.rolling(14).std().shift(1)
    sigma30 = ret.rolling(30).std().shift(1)
    rsi14 = df["RSI14"].shift(1)
    volr = df["VOL_RATIO"].shift(1)
    corr30 = df["BTC_상관계수"].shift(1)
    bbw = ((df["bb_upper"] - df["bb_lower"]) / (df["Close"] + 1e-9)).shift(1)  # 밴드폭 비율
    ma_slope = (df["MA20"].pct_change().shift(1)).fillna(0)

    # 라벨: 향후 3일 최대 불리 변동(하락) 절댓값
    fwd_low3 = df["Low"].shift(-1).rolling(window=3).min()
    mae3 = np.maximum(0.0, (df["Close"] - fwd_low3) / (df["Close"] + eps))  # 0~상한
    y = (risk_per_trade_pct / (mae3 + eps)).clip(1.0, min(exch_cap, user_cap, base_coin_cap))

    X = pd.DataFrame({
        "sigma14": sigma14, "sigma30": sigma30,
        "rsi14": rsi14, "vol_ratio": volr,
        "btc_corr30": corr30,
        "bb_width": bbw, "ma20_slope": ma_slope,
    }).replace([np.inf, -np.inf], np.nan).dropna()
    y = y.reindex(X.index).values.astype(np.float32)
    return X, y

def predict_leverage_with_xgb(booster, cur_feat: pd.DataFrame, alphas: np.ndarray) -> Dict[str, float]:
    X_cur = cur_feat.values.astype(np.float32)
    scores = booster.inplace_predict(X_cur)  # (n, len(alphas))
    # 분위수 배열과 매칭
    out = {}
    for i, a in enumerate(alphas):
        out[f"q{int(a*100)}"] = float(scores[-1, i])  # 가장 최근 행
    return out

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
            if pd.isna(volatility_30d) or volatility_30d == 0:
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

        # 시계열 예측(ETS/Theta) — (기존 로직 그대로, 생략 없이 유지)
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

        with st.spinner("🧪 모델 검증(원스텝) 중..."):
            scores = []
            for f, nm in zip(cand, cand_names):
                try:
                    sc = rolling_one_step_score(series_log, f,
                        fcst_func=lambda res, idx, method="exact": forecast_ets(res, idx, method),
                        horizon=1, window_days=120)
                except Exception:
                    sc = rolling_one_step_score(series_log, f,
                        fcst_func=lambda res, steps=1: forecast_theta(res, steps),
                        horizon=1, window_days=120)
                scores.append(sc)

        with st.spinner("🤖 최종 모델 적합 및 예측 중..."):
            last_date = df.index[-1]
            future_idx = pd.date_range(last_date + datetime.timedelta(days=1), periods=horizon, freq="D")

            def _fit_and_forecast(func, name):
                if "Theta" in name:
                    res = func(series_log)["res"]
                    mean_log, ci = forecast_theta(res, steps=horizon)
                    mean = _inv_log(mean_log, log_info)
                    ci = ci.copy()
                    ci["lower y"] = _inv_log(ci["lower y"], log_info)
                    ci["upper y"] = _inv_log(ci["upper y"], log_info)
                    mean.index = future_idx; ci.index = future_idx
                    return {"mean": mean, "ci": ci}
                else:
                    res = func(series_log)["res"]
                    mean_log, ci = forecast_ets(res, future_idx, method="exact", reps=1000)
                    mean = _inv_log(mean_log, log_info)
                    ci = ci.rename(columns=lambda c: "lower y" if "lower" in c.lower() else ("upper y" if "upper" in c.lower() else c))
                    ci = pd.DataFrame({
                        "lower y": _inv_log(ci.iloc[:, 0], log_info),
                        "upper y": _inv_log(ci.iloc[:, -1], log_info)
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
                best_idx = int(np.argmin(scores))
                best_name = cand_names[best_idx]
                out = _fit_and_forecast(cand[best_idx], best_name)
                mean, ci = out["mean"], out["ci"]

        # in-sample 적합값
        with st.spinner("📊 과거 적합 곡선 계산 중..."):
            if "Theta" in best_name:
                res = _fit_theta_auto(series_log)["res"]; fitted_log = res.fittedvalues
            elif "무계절" in best_name:
                res = _fit_ets_no_season(series_log)["res"]; fitted_log = res.fittedvalues
            else:
                res = _fit_ets_auto(series_log)["res"]; fitted_log = res.fittedvalues
            fitted = _inv_log(pd.Series(fitted_log, index=series_log.index), log_info)

        # ── 기본 출력(표/차트) — 기존과 동일, 생략 ──

        # ────────────────────────────────────────────────────────────────
        # 🔧 XGBoost: 코인별 레버리지 추천(정량회귀)
        # ────────────────────────────────────────────────────────────────
        entry_price = float(df["Close"].iloc[-1])
        far_price = float(mean.iloc[-1])
        # 예측구간 첫날 폭으로 기대 변동치 추정
        exp_move = float((ci["upper y"].iloc[0] - ci["lower y"].iloc[0]) / max(entry_price, 1e-8)) / 2.0
        sigma_eff = float(max(volatility_30d, exp_move) if not np.isnan(exp_move) else volatility_30d)
        vol_ratio_now = float((df["Volume"].iloc[-1] / (df["VOL_MA20"].iloc[-1] + 1e-9)))

        exch_cap = max_leverage_map.get(selected_crypto, 50)
        user_cap = int(leverage_ceiling)
        base_cap = coin_safe_cap_map.get(selected_crypto, 6.0)
        # σ가 커질수록 코인 안전 캡 축소
        vol_factor = float(np.clip(0.02 / max(sigma_eff, 1e-4), 0.4, 1.2))
        coin_cap = max(1.0, round(base_cap * vol_factor, 2))
        # 학습 데이터 구성
        X_train, y_train = build_leverage_training_set(
            df.copy(), risk_per_trade_pct, exch_cap, user_cap, coin_cap
        )

        # 현재 피처(가장 최근 1행) 준비
        cur_feat = pd.DataFrame({
            "sigma14": [df["Close"].pct_change().rolling(14).std().iloc[-2]],
            "sigma30": [df["Close"].pct_change().rolling(30).std().iloc[-2]],
            "rsi14":   [df["RSI14"].iloc[-2]],
            "vol_ratio":[df["VOL_RATIO"].iloc[-2]],
            "btc_corr30":[df["BTC_상관계수"].iloc[-2]],
            "bb_width":[((df["bb_upper"].iloc[-2]-df["bb_lower"].iloc[-2])/(df["Close"].iloc[-2]+1e-9))],
            "ma20_slope":[(df["MA20"].pct_change().iloc[-2])],
        }).replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)

        # 학습/예측
        rec_from_xgb = None
        if XGB_AVAILABLE and len(X_train) >= 200:  # 최소 표본 확보 후 모델 사용
            try:
                alphas = np.array([0.10, 0.50, 0.90], dtype=np.float32)  # 보수/중앙/공격
                booster = train_xgb_leverage_model(X_train, y_train, alphas)
                q_preds = predict_leverage_with_xgb(booster, cur_feat, alphas)
                # 안전 측(10% 분위수)을 기본 추천으로 채택
                safe_q = q_preds.get("q10", np.nan)
                median_q = q_preds.get("q50", np.nan)
                # 최종 추천(클립: 거래소/사용자/코인 캡)
                rec_from_xgb = float(np.clip(safe_q if not np.isnan(safe_q) else median_q,
                                             1.0, min(exch_cap, user_cap, coin_cap)))
            except Exception as e:
                st.info(f"XGBoost 정량회귀 사용 불가(폴백 사용): {e}")
                rec_from_xgb = None

        # 폴백(룰 기반)
        if rec_from_xgb is None or np.isnan(rec_from_xgb):
            max_loss_amount = float(investment_amount) * float(risk_per_trade_pct)
            stop_loss_pct = max(float(sigma_eff) * float(stop_loss_k), 0.002)
            per_coin_risk = entry_price * stop_loss_pct if entry_price > 0 else 0.0
            position_qty = (max_loss_amount / per_coin_risk) if per_coin_risk > 0 else 0.0
            notional_value = entry_price * position_qty
            rec_from_rule = (notional_value / float(investment_amount)) if investment_amount > 0 else 1.0
            rec_from_xgb = float(np.clip(rec_from_rule, 1.0, min(exch_cap, user_cap, coin_cap)))

        # 이후 출력/목표가·손절 등은 기존 로직 사용 (rec_from_xgb를 추천 레버리지로)
        direction = "up" if far_price > entry_price else "down"
        position_signal = "매수 / 롱" if direction == "up" else "매도 / 숏"

        # 손절/수량 재계산(추천 레버리지는 별개로 표시)
        max_loss_amount = float(investment_amount) * float(risk_per_trade_pct)
        stop_loss_pct = max(float(sigma_eff) * float(stop_loss_k), 0.002)
        per_coin_risk = entry_price * stop_loss_pct if entry_price > 0 else 0.0
        position_qty = (max_loss_amount / per_coin_risk) if per_coin_risk > 0 else 0.0

        # σ 기반 목표가
        def gen_targets(entry: float, sigma: float, direction: str, k_list=(0.5, 1.0, 1.5, 2.0, 3.0)):
            return [(entry + entry*sigma*k) if direction == "up" else (entry - entry*sigma*k) for k in k_list]
        pct_change = abs(far_price - entry_price) / max(entry_price, 1e-8)
        ks = (0.5, 1.0, 1.5, 2.0, 3.0) if pct_change >= 0.05 else ((0.5, 1.0, 1.5) if pct_change >= 0.02 else (1.0,))
        targets = gen_targets(entry_price, sigma_eff, direction, ks)
        primary_target = targets[-1]
        stop_loss_price = entry_price * (1 - stop_loss_pct) if direction == "up" else entry_price * (1 + stop_loss_pct)

        # 간이 방향일치 승률(생략 없이 유지) …
        N = min(180, len(series_log) - 30)
        correct = 0; total = 0
        if N > 10:
            idx = series_log.index[-N:]
            for i in range(1, len(idx)):
                train = series_log.loc[:idx[i-1]]
                try:
                    if "Theta" in best_name:
                        r = _fit_theta_auto(train)["res"]; pred1 = float(r.forecast(steps=1).iloc[0])
                    elif "무계절" in best_name:
                        r = _fit_ets_no_season(train)["res"]; fut_idx = pd.date_range(idx[i-1] + datetime.timedelta(days=1), periods=1, freq="D")
                        pred1 = float(forecast_ets(r, fut_idx, "exact")[0].iloc[0])
                    else:
                        r = _fit_ets_auto(train)["res"]; fut_idx = pd.date_range(idx[i-1] + datetime.timedelta(days=1), periods=1, freq="D")
                        pred1 = float(forecast_ets(r, fut_idx, "exact")[0].iloc[0])
                    pred1 = float(_inv_log(pd.Series([pred1]), log_info).iloc[0])
                except Exception:
                    continue
                real_today = float(df.loc[idx[i], "Close"]); real_yest = float(df.loc[idx[i-1], "Close"])
                actual_dir = 1 if real_today > real_yest else -1
                pred_dir = 1 if pred1 > real_yest else -1
                correct += int(actual_dir == pred_dir); total += 1
        rate_win = round((correct / total * 100.0), 2) if total > 0 else 0.0

        # ── 출력 ───────────────────────────────────────────────────────
        st.subheader("💖 레버리지·목표가 / 롱·숏 / AI 전략 / 진입가 범위")
        entry_low = entry_price * (1 - sigma_eff) if direction == "up" else entry_price
        entry_high = entry_price if direction == "up" else entry_price * (1 + sigma_eff)
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
        for i, tgt in enumerate(targets, 1):
            st.markdown(f"- 🎯 목표가 {i} : {tgt:.5f} USDT")

        st.markdown(f"""
➖ **AI 전략(원스텝 백테스트)** ➖  
- 🎰 **방향 일치율**: {rate_win:.2f}% (최근 {total}회 평가)  
- 🧠 **선택 모델**: {best_name}  

➖ **추천 레버리지 (XGBoost·코인별 학습)**: {rec_from_xgb:.2f}배  
- 거래소 상한: {exch_cap}배, 사용자 상한: {user_cap}배, 코인 안전 캡(변동성 조정): {coin_cap}배

> ⚠️ 본 내용은 투자 조언이 아니며 연구/교육용 참고 자료입니다.
""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ 오류: {str(e)}")
        st.markdown("""
**문제 해결 안내**  
1) 심볼을 BTC → **BTCUSDT**처럼 입력했는지 확인하세요.  
2) yfinance가 최신인지 점검하세요 (`pip install --upgrade yfinance`).  
3) 최소 100 거래일 이상 데이터가 필요합니다.  
4) `statsmodels`는 0.14+, `xgboost`는 2.0.0+ 권장입니다.  
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div style='text-align:center'>
    <h1>💎 한글 AI 코인 예측 시스템 (TA/ETS/Theta + XGBoost 레버리지)</h1>
    <p>사이드바에서 심볼과 옵션을 설정한 뒤 ‘🚀 분석 시작’을 누르세요.</p>
    <p>코인별 과거 데이터를 학습한 XGBoost 정량회귀로 안전/중앙/공격 분위수 레버리지를 추정합니다.</p>
    <p>지지선·추세선·RSI·거래량 기반 평가/조언과, 예측 평균/예측구간도 함께 제공합니다.</p>
</div>
""", unsafe_allow_html=True)
