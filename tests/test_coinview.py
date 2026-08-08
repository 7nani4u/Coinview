# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd
import pytest

import api.index as app


def _synthetic_market_data(points=320, base=30_000.0):
    x = np.arange(points, dtype=float)
    close = base + 18.0 * x + 420.0 * np.sin(x / 9.0)
    frame = pd.DataFrame(
        {
            "Open": close - 12.0,
            "High": close + 90.0,
            "Low": close - 90.0,
            "Close": close,
            "Volume": 1_000.0 + 120.0 * np.cos(x / 4.0),
        }
    )
    frame = app.add_indicators(frame)
    data = {
        column: [None if pd.isna(value) else float(value) for value in frame[column].tolist()]
        for column in frame.columns
    }
    data["Date"] = [f"2025-01-{index % 28 + 1:02d}" for index in range(points)]
    return data


@pytest.mark.parametrize(
    ("query", "symbol", "name"),
    [
        ("비트코인", "BTCUSDT", "BTC"),
        ("btc", "BTCUSDT", "BTC"),
        ("ETH/USDT", "ETHUSDT", "ETH"),
        ("솔라나", "SOLUSDT", "SOL"),
        ("doge", "DOGEUSDT", "DOGE"),
    ],
)
def test_resolve_ticker_is_crypto_only(query, symbol, name):
    assert app.resolve_ticker(query) == (symbol, "CRYPTO", name)


def test_coin_suggestions_support_korean_alias_and_symbol():
    korean = app.search_coin_suggestions("이더", 5)
    symbol = app.search_coin_suggestions("eth", 5)
    assert korean and korean[0]["ticker"] == "ETHUSDT"
    assert symbol and symbol[0]["ticker"] == "ETHUSDT"
    assert all(item["market"] == "CRYPTO" for item in korean + symbol)
    assert all(item["exchange"] == "Binance USDT" for item in korean + symbol)


@pytest.mark.parametrize(
    ("price", "digits"),
    [(65_000.0, 2), (1_917.1234, 2), (0.07044321, 6), (0.0000087654, 8)],
)
def test_crypto_price_precision(price, digits):
    assert app.get_round_digits(price, "CRYPTO") == digits


def test_crypto_event_risk_uses_crypto_categories():
    result = app.calc_event_risk(
        "BTCUSDT",
        "CRYPTO",
        [{"title": "Exchange suspends withdrawals after major hack and liquidity crisis"}],
    )
    assert result["score"] >= 28
    assert result["level"] == "high"
    assert result["days_to_earnings"] is None
    assert any("해킹" in reason or "유동성" in reason for reason in result["reasons"])


def test_crypto_signal_confidence_contract():
    result = app.build_crypto_signal_confidence(
        technical_score=72,
        ai_score=68,
        regime="BULL",
        news_items=[{"title": "Bitcoin adoption and ETF inflow rise"}],
        volatility_pct=3.2,
    )
    assert result["signal"] == "BUY"
    assert 0 <= result["confidence"] <= 100
    interval = result["confidence_interval"]
    assert interval["lower"] <= result["confidence"] <= interval["upper"]
    assert result["days_to_earnings"] is None
    assert result["macro_regime"]["regime"] == "Risk-On"


def test_prediction_trade_plan_reverses_entries_and_stop_for_long_short():
    dd = {
        "RSI": [55.0], "EMA12": [100.0], "EMA20": [100.0], "EMA50": [99.0],
        "MACD": [1.0], "ADX": [30.0],
    }
    leverage = {"recommended_leverage": 3, "risk_grade": "Medium", "factors": {}}
    long_plan = app.build_prediction(100.0, 4.0, 4.0, 70.0, leverage, dd, 65.0, 35.0)["trade_plan"]
    short_plan = app.build_prediction(100.0, 4.0, 4.0, 30.0, leverage, dd, 35.0, 65.0)["trade_plan"]

    assert long_plan["direction"] == "LONG"
    assert long_plan["second_entry"]["high"] < 100.0
    assert long_plan["stop_loss"] < long_plan["second_entry"]["low"]
    assert "매수 구간" in long_plan["first_label"]

    assert short_plan["direction"] == "SHORT"
    assert short_plan["second_entry"]["low"] > 100.0
    assert short_plan["stop_loss"] > short_plan["second_entry"]["high"]
    assert "숏" in short_plan["first_label"]


def test_detailed_entry_stop_is_outside_every_long_and_short_band():
    buy_price = {
        "aggressive_bands": [{"steps": [{"price": 98.0}, {"price": 96.0}]}],
        "recommended_bands": [{"steps": [{"price": 94.0}, {"price": 92.0}]}],
    }
    long_prediction = {
        "direction": "LONG", "stop_loss": 95.0,
        "trade_plan": {"direction": "LONG", "stop_loss": 95.0},
        "targets": [{"price": 106.0}], "risk": {},
    }
    short_prediction = {
        "direction": "SHORT", "stop_loss": 105.0,
        "trade_plan": {"direction": "SHORT", "stop_loss": 105.0},
        "targets": [{"price": 94.0}], "risk": {},
    }
    long_result = app.align_prediction_stop_with_detailed_entries(long_prediction, buy_price, 100.0, 2.0)
    short_result = app.align_prediction_stop_with_detailed_entries(short_prediction, buy_price, 100.0, 2.0)
    assert long_result["trade_plan"]["stop_loss"] < 92.0
    assert short_result["trade_plan"]["stop_loss"] > 108.0


def test_derivatives_contract_and_probability_adjustment(monkeypatch):
    def fake_binance(path, params=None, fapi=False, timeout=8):
        if path.endswith("premiumIndex"):
            return {"markPrice": "101", "indexPrice": "100", "lastFundingRate": "0.0008", "nextFundingTime": 123}
        if path.endswith("globalLongShortAccountRatio"):
            return [{"longShortRatio": "2", "longAccount": "0.67", "shortAccount": "0.33"}]
        if path.endswith("symbolAdlRisk"):
            return {"adlRisk": "HIGH"}
        return None

    app._CACHE.clear()
    monkeypatch.setattr(app, "_binance_get", fake_binance)
    result = app.fetch_crypto_derivatives("BTCUSDT")
    assert result["available"] is True
    assert result["funding_rate_pct"] == pytest.approx(0.08)
    assert result["long_account_pct"] == pytest.approx(67.0)
    assert result["adl_risk"] == "HIGH"
    assert result["probability_adjustment"] < 0


def test_english_news_translation_returns_korean(monkeypatch):
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return [[["비트코인 상승세가 확대되고 있습니다.", "Bitcoin rally expands"]], None, "en"]

    app._CACHE.clear()
    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: FakeResponse())
    translated = app._translate_to_korean("Bitcoin rally expands")
    assert "비트코인" in translated


def test_coin_route_full_contract_without_external_network(monkeypatch):
    market_data = _synthetic_market_data()
    monkeypatch.setattr(app, "fetch_coin_data", lambda *args, **kwargs: (market_data, [], "BTCUSDT"))
    monkeypatch.setattr(app, "fetch_open_interest", lambda *args, **kwargs: 12345.0)
    monkeypatch.setattr(app, "fetch_crypto_derivatives", lambda *args, **kwargs: {"available": False})
    monkeypatch.setattr(app, "check_market_regime", lambda *args, **kwargs: "BULL")

    result = app.route("/api/coin", {"ticker": "BTC", "period": "1y", "market": "CRYPTO"})

    assert "error" not in result
    assert result["market"] == "CRYPTO"
    assert result["symbol"] == "BTCUSDT"
    assert result["last_close"] > 0
    assert len(result["chart_data"]["close"]) == len(market_data["Close"])
    assert result["prediction"]
    assert result["prediction"]["trade_plan"]["direction"] in {"LONG", "SHORT", "NEUTRAL"}
    assert result["prediction_outlook"]["scenarios"]
    assert result["signal_confidence"]["days_to_earnings"] is None
    assert result["event_risk"]["days_to_earnings"] is None
    assert result["learning_adjustment"]["applied"] is False
    assert result["chart_data"]["pattern_overlay_options"]["interaction_mode"] == "hover_touch"


def test_backward_compatible_stock_path_is_coin_alias(monkeypatch):
    market_data = _synthetic_market_data()
    monkeypatch.setattr(app, "fetch_coin_data", lambda *args, **kwargs: (market_data, [], "ETHUSDT"))
    monkeypatch.setattr(app, "fetch_open_interest", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "fetch_crypto_derivatives", lambda *args, **kwargs: {"available": False})
    monkeypatch.setattr(app, "check_market_regime", lambda *args, **kwargs: "NEUTRAL")
    result = app.route("/api/stock", {"ticker": "ETH", "period": "1y"})
    assert result["market"] == "CRYPTO"
    assert result["symbol"] == "ETHUSDT"


def test_lightweight_endpoints(monkeypatch):
    monkeypatch.setattr(
        app,
        "fetch_ticker_24h",
        lambda symbol: {"price": 0.07044321, "change_pct": 2.5, "high_24h": 0.08, "low_24h": 0.06},
    )
    price = app.route("/api/price", {"ticker": "DOGE"})
    assert price["price"] == pytest.approx(0.070443, abs=1e-9)
    assert price["session_name"] == "24시간"

    monkeypatch.setattr(
        app,
        "fetch_tickers_batch",
        lambda symbols: {
            symbol: {"price": index + 1.0, "change_pct": index - 1.0, "volume": 100.0}
            for index, symbol in enumerate(symbols)
        },
    )
    quotes = app.route("/api/alert/quote", {"symbols": "BTCUSDT,ETHUSDT"})["quotes"]
    assert set(quotes) == {"BTCUSDT", "ETHUSDT"}


def test_crypto_peer_outlook_contract(monkeypatch):
    samples = {
        "ETH": (4.0, 10.0, 64.0),
        "SOL": (6.0, 15.0, 70.0),
        "ADA": (-1.0, 3.0, 48.0),
        "AVAX": (2.0, 8.0, 59.0),
        "SUI": (8.0, 20.0, 73.0),
        "APT": (1.0, 5.0, 55.0),
        "NEAR": (-2.0, 1.0, 45.0),
    }

    def fake_metrics(base):
        ret5, ret20, up = samples[base]
        return {
            "ticker": f"{base}USDT",
            "name": app.COIN_DISPLAY_NAMES.get(base, base),
            "return_5d": ret5,
            "return_20d": ret20,
            "rsi": 55.0,
            "above_ma20": ret20 >= 0,
            "trend": "상승" if up >= 57 else "하락" if up <= 43 else "혼조",
            "up_probability": up,
            "down_probability": 100.0 - up,
        }

    app._CACHE.clear()
    monkeypatch.setattr(app, "_crypto_peer_metrics", fake_metrics)
    result = app.build_crypto_peer_outlook("ETHUSDT", "이더리움")
    assert result["ok"] is True
    assert result["market"] == "CRYPTO"
    assert result["group_name"] == "스마트컨트랙트 플랫폼"
    assert result["selected"]["ticker"] == "ETHUSDT"
    assert result["peers"]


def test_html_exposes_coin_ui_and_new_interactions():
    html = app.HTML
    assert "<title>암호화폐 AI 레버리지 예측 시스템 (Binance)</title>" in html
    assert 'id="stock-search-wrap"' in html
    assert 'id="ticker-suggestions"' in html
    assert "🪙 유사 코인 전망" in html
    assert "코인 진단" in html
    assert "/api/coin?ticker=" in html
    assert "let currentMarket = 'CRYPTO';" in html
    assert "/api/price?ticker=" in html
    assert "market=CRYPTO" in html
    assert "function _escPrediction(value)" in html
    assert "function renderPullbackIntoForecast(d, isKrx)" in html
    assert "🤖 AI 종합 의견" not in html
    assert "📈 향후 가격 상승 가능 범위 (목표가 예측)" not in html
    assert "⚠️ <strong>레버리지 사용 주의</strong>" not in html
    assert "💡 AI 종합 진단 및 트레이딩 전략" not in html
    assert "⚡ 1차 매수 구간 (ATR 기반) · 소액 탐색" in html
    assert "📍 2차 매수 구간 · 주 진입" in html
    assert "🛡️ 리스크 관리 (ATR 기반)" in html
    assert "function _buildCryptoDetailedForecastView(source)" in html
    assert 'aria-label="밴드 ${b.band} 5단계 ${entryNoun} 가격"' in html
    assert "bpEl.innerHTML = (isCrypto ? '' : stratBanner + artyHtml)" in html
    assert "news-single-column" in html
    assert "classList.toggle('news-single-column', !isKrx)" in html
    assert 'id="page-portfolio"' in html
    assert "cv_portfolio_v1" in html
    assert "wss://stream.binance.com:9443" in html
    assert "wss://fstream.binance.com" in html
    assert "@forceOrder" in html
    assert "data-live-liquidation" in html
