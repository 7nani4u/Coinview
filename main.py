# -*- coding: utf-8 -*-
# SignalCoin - main.py
#
# Vercel 공식 FastAPI 배포 구조:
#   - 파일 위치: 프로젝트 루트의 main.py
#   - vercel.json: builds + routes 설정
#   - requirements.txt: 의존성
#   - HTML은 HTMLResponse로 직접 서빙
#
# 참고: https://vercel.com/docs/frameworks/backend/fastapi

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import warnings
import math

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# FastAPI 앱 초기화 (Vercel이 이 변수를 자동으로 찾음)
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SignalCoin",
    description="코인 AI 예측 시스템",
    version="9.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# HTML 프론트엔드 (인라인으로 포함 - Vercel 공식 방식)
# ─────────────────────────────────────────────────────────────
HTML_CONTENT = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>코인 AI 예측 시스템 v5.0</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Noto Sans KR', sans-serif;
      background: #F0F2F6;
      color: #2C3E50;
      display: flex;
      min-height: 100vh;
    }
    #sidebar {
      width: 320px;
      min-width: 280px;
      background: #FFFFFF;
      border-right: 1px solid #E0E0E0;
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
      height: 100vh;
      position: sticky;
      top: 0;
    }
    #sidebar h1 {
      font-size: 18px;
      font-weight: 700;
      color: #2C3E50;
      border-bottom: 3px solid #3498DB;
      padding-bottom: 8px;
    }
    .sidebar-section { display: flex; flex-direction: column; gap: 8px; }
    .sidebar-section h3 {
      font-size: 14px;
      font-weight: 700;
      color: #3498DB;
      margin-bottom: 2px;
    }
    .sidebar-section label { font-size: 13px; color: #555; }
    .sidebar-section select, .sidebar-section input[type="number"],
    .sidebar-section input[type="text"] {
      width: 100%;
      padding: 7px 10px;
      border: 1px solid #CCC;
      border-radius: 6px;
      font-size: 13px;
      background: #FAFAFA;
    }
    .sidebar-section input[type="range"] { width: 100%; }
    .radio-group { display: flex; gap: 10px; }
    .radio-group label { display: flex; align-items: center; gap: 4px; font-size: 13px; cursor: pointer; }
    .btn-analyze {
      background: #3498DB;
      color: #FFF;
      border: none;
      border-radius: 8px;
      padding: 12px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn-analyze:hover { background: #2980B9; }
    .btn-analyze:disabled { background: #95A5A6; cursor: not-allowed; }
    #fg-gauge { width: 100%; min-height: 180px; }
    .info-box {
      background: #EBF5FB;
      border-left: 4px solid #3498DB;
      padding: 8px 10px;
      border-radius: 4px;
      font-size: 12px;
      color: #2C3E50;
    }
    .warn-box {
      background: #FEF9E7;
      border-left: 4px solid #F39C12;
      padding: 8px 10px;
      border-radius: 4px;
      font-size: 12px;
      color: #7D6608;
    }
    .success-box {
      background: #EAFAF1;
      border-left: 4px solid #27AE60;
      padding: 8px 10px;
      border-radius: 4px;
      font-size: 12px;
      color: #1E8449;
    }
    hr { border: none; border-top: 1px solid #E0E0E0; }
    #main {
      flex: 1;
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .section-title {
      font-size: 22px;
      font-weight: 700;
      padding-bottom: 8px;
      border-bottom: 3px solid #3498DB;
      color: #2C3E50;
      margin-bottom: 12px;
    }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
    .metric-card {
      background: #FFF;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    .metric-card .metric-label { font-size: 12px; color: #7F8C8D; margin-bottom: 4px; }
    .metric-card .metric-value { font-size: 22px; font-weight: 700; }
    .metric-card .metric-delta { font-size: 12px; margin-top: 4px; }
    .delta-up { color: #27AE60; }
    .delta-down { color: #E74C3C; }
    .tab-bar { display: flex; gap: 4px; border-bottom: 2px solid #E0E0E0; margin-bottom: 16px; flex-wrap: wrap; }
    .tab-btn {
      padding: 8px 16px;
      border: none;
      background: transparent;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      color: #7F8C8D;
      border-bottom: 3px solid transparent;
      margin-bottom: -2px;
      transition: all 0.2s;
    }
    .tab-btn.active { color: #3498DB; border-bottom-color: #3498DB; font-weight: 700; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .signal-badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 999px;
      font-size: 14px;
      font-weight: 700;
      color: #FFF;
    }
    .signal-STRONG_BUY { background: #1ABC9C; }
    .signal-BUY { background: #27AE60; }
    .signal-NEUTRAL { background: #F39C12; }
    .signal-SELL { background: #E67E22; }
    .signal-STRONG_SELL { background: #E74C3C; }
    .progress-wrap { background: #ECF0F1; border-radius: 999px; height: 10px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 999px; background: #3498DB; transition: width 0.4s; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { background: #EBF5FB; padding: 8px 12px; text-align: left; font-weight: 600; }
    td { padding: 8px 12px; border-bottom: 1px solid #F0F0F0; }
    tr:hover td { background: #F8FBFF; }
    #loading-overlay {
      display: none;
      position: fixed; inset: 0;
      background: rgba(255,255,255,0.8);
      z-index: 9999;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 16px;
    }
    #loading-overlay.active { display: flex; }
    .spinner {
      width: 48px; height: 48px;
      border: 5px solid #E0E0E0;
      border-top-color: #3498DB;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #loading-text { font-size: 16px; font-weight: 600; color: #2C3E50; }
    @media (max-width: 768px) {
      body { flex-direction: column; }
      #sidebar { width: 100%; height: auto; position: static; }
      #main { padding: 12px; }
    }
  </style>
</head>
<body>

<div id="loading-overlay">
  <div class="spinner"></div>
  <div id="loading-text">분석 중...</div>
</div>

<aside id="sidebar">
  <h1>&#129689; 코인 AI 예측 시스템 v5.0</h1>

  <div class="sidebar-section">
    <h3>&#128561; 시장 심리 지수</h3>
    <div id="fg-gauge"></div>
    <div id="fg-message" class="info-box">시장 심리 데이터 로딩 중...</div>
  </div>
  <hr/>

  <div class="sidebar-section">
    <h3>1&#65039;&#8419; 시간 선택</h3>
    <label>&#128200; 시간 프레임</label>
    <select id="sel-interval">
      <option value="1d" selected>1일봉 (전체 기간)</option>
      <option value="1h">1시간봉 (최근 730일)</option>
      <option value="5m">5분봉 (최근 60일)</option>
      <option value="1m">1분봉 (최근 7일)</option>
    </select>
    <div id="interval-info" class="info-box">&#9201;&#65039; 1일봉: 전체 기간 지원 (중장기 투자용)</div>
  </div>
  <hr/>

  <div class="sidebar-section">
    <h3>2&#65039;&#8419; 코인 선택</h3>
    <div class="radio-group">
      <label><input type="radio" name="coin-method" value="basic" checked /> 기본 목록</label>
      <label><input type="radio" name="coin-method" value="custom" /> 직접 입력</label>
    </div>
    <div id="basic-list-wrap">
      <label>&#128142; 암호화폐</label>
      <select id="sel-coin">
        <option value="BTC-USD">Bitcoin (BTC)</option>
        <option value="ETH-USD">Ethereum (ETH)</option>
        <option value="BNB-USD">BNB (BNB)</option>
        <option value="XRP-USD">Ripple (XRP)</option>
        <option value="ADA-USD">Cardano (ADA)</option>
        <option value="SOL-USD">Solana (SOL)</option>
        <option value="DOGE-USD">Dogecoin (DOGE)</option>
        <option value="DOT-USD">Polkadot (DOT)</option>
        <option value="AVAX-USD">Avalanche (AVAX)</option>
        <option value="MATIC-USD">Polygon (MATIC)</option>
        <option value="LINK-USD">Chainlink (LINK)</option>
        <option value="UNI-USD">Uniswap (UNI)</option>
        <option value="LTC-USD">Litecoin (LTC)</option>
        <option value="ATOM-USD">Cosmos (ATOM)</option>
        <option value="ETC-USD">Ethereum Classic (ETC)</option>
        <option value="XLM-USD">Stellar (XLM)</option>
        <option value="TRX-USD">TRON (TRX)</option>
        <option value="NEAR-USD">NEAR Protocol (NEAR)</option>
        <option value="APT-USD">Aptos (APT)</option>
        <option value="SUI-USD">Sui (SUI)</option>
      </select>
    </div>
    <div id="custom-input-wrap" style="display:none;">
      <label>&#128269; 심볼 직접 입력</label>
      <input type="text" id="custom-ticker" placeholder="예: BTC-USD, ETH-USD, PEPE-USD" />
      <div class="info-box" style="margin-top:4px;">yfinance 형식으로 입력하세요 (예: BTC-USD)</div>
    </div>
  </div>
  <hr/>

  <div class="sidebar-section">
    <h3>3&#65039;&#8419; 분석 기간</h3>
    <label>&#128197; 기간 (일): <strong id="days-label">365</strong>일</label>
    <input type="range" id="range-days" min="90" max="1825" value="365" step="30"
           oninput="document.getElementById('days-label').textContent=this.value" />
    <div id="days-note" class="info-box" style="font-size:11px;">&#9432; 1분봉은 최대 7일, 5분봉은 최대 60일로 자동 제한됩니다.</div>
  </div>
  <hr/>

  <div class="sidebar-section">
    <h3>4&#65039;&#8419; 투자 설정</h3>
    <label>&#128302; 예측 기간 (일): <strong id="forecast-label">3</strong>일</label>
    <input type="range" id="range-forecast" min="1" max="30" value="3" step="1"
           oninput="document.getElementById('forecast-label').textContent=this.value" />
    <label>&#128176; 투자 금액 (USDT)</label>
    <input type="number" id="investment-amount" value="1000" min="1" step="50" />
    <label>&#9888;&#65039; 리스크 비율 (%): <strong id="risk-label">2.0</strong>%</label>
    <input type="range" id="range-risk" min="0.5" max="5.0" value="2.0" step="0.5"
           oninput="document.getElementById('risk-label').textContent=parseFloat(this.value).toFixed(1)" />
    <label>&#128721; 손절 배수 (sigma): <strong id="sl-label">2.0</strong></label>
    <input type="range" id="range-sl" min="1.0" max="3.0" value="2.0" step="0.5"
           oninput="document.getElementById('sl-label').textContent=parseFloat(this.value).toFixed(1)" />
    <label>&#128202; 최대 레버리지</label>
    <input type="number" id="max-leverage" value="50" min="1" max="500" step="1" />
  </div>
  <hr/>

  <button class="btn-analyze" id="btn-analyze" onclick="runAnalysis()">&#128269; 분석 시작</button>
  <div id="sidebar-status"></div>
</aside>

<main id="main">
  <div id="welcome-section">
    <div class="section-title">&#129689; 코인 AI 예측 시스템 v5.0</div>
    <div class="info-box" style="font-size:14px; padding:16px;">
      왼쪽 사이드바에서 코인과 설정을 선택한 후 <strong>&#128269; 분석 시작</strong> 버튼을 클릭하세요.
      <br/><br/>
      &#10003; 실시간 가격 데이터 (yfinance)<br/>
      &#10003; 기술적 지표: RSI, MACD, 볼린저 밴드, EMA<br/>
      &#10003; AI 신호 점수 시스템 (STRONG_BUY ~ STRONG_SELL)<br/>
      &#10003; 포트폴리오 백테스트 (샤프 비율, 최대 낙폭)<br/>
      &#10003; 기간별 수익률 분석 (1주/1개월/3개월)
    </div>
  </div>

  <div id="result-section" style="display:none;">
    <div class="section-title" id="result-title">&#128202; 분석 결과</div>
    <div class="metric-grid" id="summary-metrics"></div>

    <div>
      <div class="section-title">&#128202; 실시간 시장 분석</div>
      <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('signal-tab', this)">&#127919; 종합 신호</button>
        <button class="tab-btn" onclick="switchTab('returns-tab', this)">&#128200; 기간별 수익률</button>
      </div>
      <div id="signal-tab" class="tab-panel active"><div id="signal-content"></div></div>
      <div id="returns-tab" class="tab-panel"><div id="returns-content"></div></div>
    </div>

    <div>
      <div class="section-title">&#127919; 포트폴리오 분석</div>
      <div id="portfolio-content"></div>
    </div>

    <div>
      <div class="section-title">&#128208; 기술적 지표</div>
      <div id="indicators-content"></div>
    </div>

    <div>
      <div class="section-title">&#128200; 차트</div>
      <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('chart-candle', this)">&#128185; 캔들스틱</button>
        <button class="tab-btn" onclick="switchTab('chart-volume', this)">&#128202; 거래량</button>
        <button class="tab-btn" onclick="switchTab('chart-rsi', this)">&#128309; RSI</button>
        <button class="tab-btn" onclick="switchTab('chart-macd', this)">&#128201; MACD</button>
        <button class="tab-btn" onclick="switchTab('chart-bb', this)">&#128207; 볼린저밴드</button>
      </div>
      <div id="chart-candle" class="tab-panel active"><div id="plotly-candle" style="height:420px;"></div></div>
      <div id="chart-volume" class="tab-panel"><div id="plotly-volume" style="height:300px;"></div></div>
      <div id="chart-rsi" class="tab-panel"><div id="plotly-rsi" style="height:300px;"></div></div>
      <div id="chart-macd" class="tab-panel"><div id="plotly-macd" style="height:300px;"></div></div>
      <div id="chart-bb" class="tab-panel"><div id="plotly-bb" style="height:380px;"></div></div>
    </div>
  </div>
</main>

<script>
let currentData = null;

document.addEventListener("DOMContentLoaded", () => {
  loadFearGreed();
  setupCoinMethodToggle();
  setupIntervalInfo();
});

function setupCoinMethodToggle() {
  document.querySelectorAll('input[name="coin-method"]').forEach(r => {
    r.addEventListener("change", () => {
      const isCustom = r.value === "custom" && r.checked;
      document.getElementById("basic-list-wrap").style.display = isCustom ? "none" : "block";
      document.getElementById("custom-input-wrap").style.display = isCustom ? "block" : "none";
    });
  });
}

function setupIntervalInfo() {
  const infoMap = {
    "1m": "⏱️ 1분봉: 최근 7일만 지원 (초단타 매매용) - 기간은 자동으로 7일 제한",
    "5m": "⏱️ 5분봉: 최근 60일만 지원 (단타 매매용) - 기간은 자동으로 60일 제한",
    "1h": "⏱️ 1시간봉: 최근 730일만 지원 (스윙 트레이딩용)",
    "1d": "⏱️ 1일봉: 전체 기간 지원 (중장기 투자용)"
  };
  document.getElementById("sel-interval").addEventListener("change", function() {
    document.getElementById("interval-info").textContent = infoMap[this.value] || "";
  });
}

async function loadFearGreed() {
  try {
    const res = await fetch("/api/fear-greed");
    if (!res.ok) throw new Error("API 오류");
    const data = await res.json();
    renderFearGreedGauge(data.value, data.classification);
    const msgEl = document.getElementById("fg-message");
    if (data.value < 25) {
      msgEl.className = "success-box";
      msgEl.textContent = "극도의 공포 → 매수 기회";
    } else if (data.value > 75) {
      msgEl.className = "warn-box";
      msgEl.textContent = "극도의 탐욕 → 매도 고려";
    } else {
      msgEl.className = "info-box";
      msgEl.textContent = "중립 상태 → 추세 관찰";
    }
  } catch (e) {
    document.getElementById("fg-message").textContent = "시장 심리 데이터를 불러올 수 없습니다.";
  }
}

function renderFearGreedGauge(value, label) {
  const color = value < 25 ? "#E74C3C" : value < 45 ? "#E67E22" : value < 55 ? "#F1C40F" : value < 75 ? "#2ECC71" : "#1ABC9C";
  Plotly.newPlot("fg-gauge", [{
    type: "indicator", mode: "gauge+number+delta",
    value: value,
    title: { text: label, font: { size: 13 } },
    gauge: {
      axis: { range: [0, 100], tickwidth: 1 },
      bar: { color: color },
      steps: [
        { range: [0, 25], color: "#FADBD8" },
        { range: [25, 45], color: "#FAE5D3" },
        { range: [45, 55], color: "#FEF9E7" },
        { range: [55, 75], color: "#D5F5E3" },
        { range: [75, 100], color: "#D1F2EB" }
      ],
      threshold: { line: { color: "#2C3E50", width: 3 }, thickness: 0.75, value: value }
    }
  }], {
    margin: { t: 30, b: 10, l: 10, r: 10 },
    height: 180,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)"
  }, { displayModeBar: false });
}

async function runAnalysis() {
  const btn = document.getElementById("btn-analyze");
  btn.disabled = true;
  showLoading("데이터를 가져오는 중...");

  const method = document.querySelector('input[name="coin-method"]:checked').value;
  const ticker = method === "basic"
    ? document.getElementById("sel-coin").value
    : (document.getElementById("custom-ticker").value.trim() || "BTC-USD");
  const interval = document.getElementById("sel-interval").value;
  let days = parseInt(document.getElementById("range-days").value);

  // interval별 최대 일수 제한 (yfinance 제약)
  const maxDays = { "1m": 7, "5m": 60, "1h": 730, "1d": 1825 };
  if (days > (maxDays[interval] || 1825)) {
    days = maxDays[interval];
  }

  try {
    updateLoadingText("지표를 계산하는 중...");
    const url = "/api/analyze?ticker=" + encodeURIComponent(ticker)
              + "&days=" + days
              + "&interval=" + encodeURIComponent(interval);
    const res = await fetch(url);
    if (!res.ok) {
      let errMsg = "분석 실패";
      try { const err = await res.json(); errMsg = err.detail || errMsg; } catch(_) {}
      throw new Error(errMsg);
    }
    currentData = await res.json();
    updateLoadingText("차트를 렌더링하는 중...");
    renderResults(currentData);
    document.getElementById("welcome-section").style.display = "none";
    document.getElementById("result-section").style.display = "block";
  } catch (e) {
    alert("오류: " + e.message);
  } finally {
    hideLoading();
    btn.disabled = false;
  }
}

function renderResults(data) {
  document.getElementById("result-title").textContent = data.ticker + " 분석 결과";
  renderSummaryMetrics(data);
  renderSignalSection(data.signal);
  renderReturnsSection(data.metrics);
  renderPortfolioSection(data.backtest);
  renderIndicatorsTable(data.indicators, data.current_price);
  renderCharts(data.chart_data, data.ticker);
}

function renderSummaryMetrics(data) {
  const price = data.current_price;
  const rsi = data.indicators.rsi;
  const signal = data.signal.signal;
  const score = data.signal.total_score;
  const sharpe = data.backtest.sharpe_ratio;
  const totalRet = data.backtest.total_return_pct;
  const cards = [
    { label: "현재 가격", value: "$" + price.toLocaleString(undefined, {maximumFractionDigits:4}), delta: null, emoji: "&#128176;" },
    { label: "RSI (14)", value: rsi ? rsi.toFixed(1) : "N/A", delta: rsi > 70 ? "과매수" : rsi < 30 ? "과매도" : "중립", deltaClass: rsi > 70 ? "delta-down" : rsi < 30 ? "delta-up" : "", emoji: "&#128202;" },
    { label: "AI 신호", value: '<span class="signal-badge signal-' + signal + '">' + signal + '</span>', delta: "점수: " + score.toFixed(1), deltaClass: "", emoji: "&#129302;" },
    { label: "총 수익률", value: (totalRet >= 0 ? "+" : "") + totalRet.toFixed(1) + "%", delta: null, deltaClass: totalRet >= 0 ? "delta-up" : "delta-down", emoji: "&#128200;" },
    { label: "샤프 비율", value: sharpe.toFixed(2), delta: sharpe > 1 ? "우수" : sharpe > 0 ? "보통" : "불량", deltaClass: sharpe > 1 ? "delta-up" : sharpe > 0 ? "" : "delta-down", emoji: "&#9889;" },
    { label: "최대 낙폭", value: data.backtest.max_drawdown_pct.toFixed(1) + "%", delta: null, deltaClass: "delta-down", emoji: "&#128201;" },
  ];
  document.getElementById("summary-metrics").innerHTML = cards.map(c =>
    '<div class="metric-card">' +
      '<div class="metric-label">' + c.emoji + ' ' + c.label + '</div>' +
      '<div class="metric-value">' + c.value + '</div>' +
      (c.delta ? '<div class="metric-delta ' + c.deltaClass + '">' + c.delta + '</div>' : '') +
    '</div>'
  ).join("");
}

function renderSignalSection(signal) {
  const pct = signal.total_score;
  const signalColors = { STRONG_BUY: "#1ABC9C", BUY: "#27AE60", NEUTRAL: "#F39C12", SELL: "#E67E22", STRONG_SELL: "#E74C3C" };
  const color = signalColors[signal.signal] || "#95A5A6";
  document.getElementById("signal-content").innerHTML =
    '<div style="display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start;">' +
      '<div style="flex:2; min-width:200px;">' +
        '<h3 style="margin-bottom:8px;">종합 신호: <span class="signal-badge signal-' + signal.signal + '">' + signal.signal + '</span></h3>' +
        '<div class="progress-wrap" style="margin:8px 0;">' +
          '<div class="progress-fill" style="width:' + pct + '%; background:' + color + ';"></div>' +
        '</div>' +
        '<div style="font-size:14px; color:#555;">종합 점수: <strong>' + pct.toFixed(1) + ' / 100</strong></div>' +
        '<div style="font-size:13px; margin-top:4px; color:#777;">신뢰도: ' + signal.confidence.toFixed(0) + '%</div>' +
      '</div>' +
      '<div style="flex:1; min-width:160px;">' +
        '<h4 style="margin-bottom:8px;">세부 점수</h4>' +
        '<table><tr><td>RSI 점수</td><td><strong>' + signal.rsi_score.toFixed(1) + '</strong></td></tr>' +
        '<tr><td>추세 점수</td><td><strong>' + signal.trend_score.toFixed(1) + '</strong></td></tr></table>' +
      '</div>' +
    '</div>';
}

function renderReturnsSection(metrics) {
  const r = metrics.returns;
  function badge(val) {
    const up = val >= 0;
    return '<span style="font-size:1.4rem; font-weight:700;">' + (up ? "+" : "") + val.toFixed(2) + '%</span>' +
           '<span style="font-size:12px; color:' + (up ? "#27AE60" : "#E74C3C") + '; background:' + (up ? "#EAFAF1" : "#FDEDEC") + '; padding:3px 8px; border-radius:999px; margin-left:6px;">' + (up ? "상승" : "하락") + '</span>';
  }
  const sentimentColor = metrics.sentiment === "BULLISH" ? "#27AE60" : metrics.sentiment === "BEARISH" ? "#E74C3C" : "#F39C12";
  document.getElementById("returns-content").innerHTML =
    '<div class="metric-grid" style="margin-bottom:16px;">' +
      '<div class="metric-card"><div class="metric-label">1주일</div><div>' + badge(r["1w"] || 0) + '</div></div>' +
      '<div class="metric-card"><div class="metric-label">1개월</div><div>' + badge(r["1m"] || 0) + '</div></div>' +
      '<div class="metric-card"><div class="metric-label">3개월</div><div>' + badge(r["3m"] || 0) + '</div></div>' +
    '</div>' +
    '<div style="background:#FFF; border-radius:12px; padding:16px; box-shadow:0 2px 4px rgba(0,0,0,0.06);">' +
      '<h4 style="margin-bottom:8px;">예상 매매 비율</h4>' +
      '<div style="display:flex; gap:0; border-radius:8px; overflow:hidden; height:32px; margin-bottom:8px;">' +
        '<div style="background:#27AE60; width:' + metrics.buy_sell_ratio + '%; display:flex; align-items:center; justify-content:center; color:#FFF; font-size:13px; font-weight:700;">매수 ' + metrics.buy_sell_ratio.toFixed(0) + '%</div>' +
        '<div style="background:#E74C3C; width:' + (100 - metrics.buy_sell_ratio) + '%; display:flex; align-items:center; justify-content:center; color:#FFF; font-size:13px; font-weight:700;">매도 ' + (100 - metrics.buy_sell_ratio).toFixed(0) + '%</div>' +
      '</div>' +
      '<div style="font-size:13px;">시장 심리: <strong style="color:' + sentimentColor + ';">' + metrics.sentiment + '</strong></div>' +
    '</div>';
}

function renderPortfolioSection(bt) {
  document.getElementById("portfolio-content").innerHTML =
    '<div class="metric-grid">' +
      '<div class="metric-card">' +
        '<div class="metric-label">최종 자본 ($1,000 투자 시)</div>' +
        '<div class="metric-value">$' + bt.final_capital.toLocaleString(undefined, {maximumFractionDigits:0}) + '</div>' +
        '<div class="metric-delta ' + (bt.total_return_pct >= 0 ? "delta-up" : "delta-down") + '">' + (bt.total_return_pct >= 0 ? "+" : "") + bt.total_return_pct.toFixed(1) + '%</div>' +
      '</div>' +
      '<div class="metric-card">' +
        '<div class="metric-label">샤프 비율</div>' +
        '<div class="metric-value">' + bt.sharpe_ratio.toFixed(2) + '</div>' +
        '<div class="metric-delta">' + (bt.sharpe_ratio > 1 ? "우수" : bt.sharpe_ratio > 0 ? "보통" : "불량") + '</div>' +
      '</div>' +
      '<div class="metric-card">' +
        '<div class="metric-label">최대 낙폭 (MDD)</div>' +
        '<div class="metric-value delta-down">' + bt.max_drawdown_pct.toFixed(1) + '%</div>' +
      '</div>' +
      '<div class="metric-card">' +
        '<div class="metric-label">승률</div>' +
        '<div class="metric-value">' + bt.win_rate_pct.toFixed(1) + '%</div>' +
        '<div class="metric-delta">(일별 수익 기준)</div>' +
      '</div>' +
    '</div>';
}

function renderIndicatorsTable(ind, price) {
  const fmt = (v, decimals=4) => v != null ? v.toLocaleString(undefined, {maximumFractionDigits: decimals}) : "N/A";
  const rows = [
    ["현재 가격", "$" + fmt(price), ""],
    ["RSI (14)", ind.rsi != null ? ind.rsi.toFixed(2) : "N/A", ind.rsi > 70 ? "과매수" : ind.rsi < 30 ? "과매도 (매수 기회)" : "중립"],
    ["MACD", ind.macd != null ? ind.macd.toFixed(6) : "N/A", ind.macd != null && ind.macd_signal != null ? (ind.macd > ind.macd_signal ? "골든크로스" : "데드크로스") : ""],
    ["MACD Signal", ind.macd_signal != null ? ind.macd_signal.toFixed(6) : "N/A", ""],
    ["볼린저 상단", ind.bb_upper != null ? "$" + fmt(ind.bb_upper) : "N/A", price > ind.bb_upper ? "상단 돌파" : ""],
    ["볼린저 하단", ind.bb_lower != null ? "$" + fmt(ind.bb_lower) : "N/A", price < ind.bb_lower ? "하단 지지" : ""],
    ["EMA 20", ind.ema20 != null ? "$" + fmt(ind.ema20) : "N/A", price > ind.ema20 ? "상향" : "하향"],
    ["EMA 50", ind.ema50 != null ? "$" + fmt(ind.ema50) : "N/A", price > ind.ema50 ? "상향" : "하향"],
  ];
  document.getElementById("indicators-content").innerHTML =
    '<table><thead><tr><th>지표</th><th>값</th><th>해석</th></tr></thead><tbody>' +
    rows.map(r => '<tr><td>' + r[0] + '</td><td><strong>' + r[1] + '</strong></td><td style="color:#555;">' + r[2] + '</td></tr>').join("") +
    '</tbody></table>';
}

function renderCharts(chartData, ticker) {
  if (!chartData || chartData.length === 0) return;
  const dates = chartData.map(d => d.Date);
  const opens = chartData.map(d => d.Open);
  const highs = chartData.map(d => d.High);
  const lows = chartData.map(d => d.Low);
  const closes = chartData.map(d => d.Close);
  const volumes = chartData.map(d => d.Volume);
  const rsi = chartData.map(d => d.rsi);
  const macd = chartData.map(d => d.macd);
  const macdSig = chartData.map(d => d.macd_signal);
  const bbUpper = chartData.map(d => d.bb_upper);
  const bbLower = chartData.map(d => d.bb_lower);
  const ema20 = chartData.map(d => d.ema20);
  const ema50 = chartData.map(d => d.ema50);

  const layout_base = {
    paper_bgcolor: "#FFF", plot_bgcolor: "#FAFAFA",
    margin: { t: 30, b: 40, l: 60, r: 20 },
    xaxis: { showgrid: false },
    yaxis: { gridcolor: "#F0F0F0" },
    legend: { orientation: "h", y: -0.15 }
  };

  Plotly.newPlot("plotly-candle", [{
    type: "candlestick", x: dates, open: opens, high: highs, low: lows, close: closes,
    name: ticker,
    increasing: { line: { color: "#27AE60" } },
    decreasing: { line: { color: "#E74C3C" } }
  }, {
    type: "scatter", x: dates, y: ema20, name: "EMA20", line: { color: "#3498DB", width: 1.5 }
  }, {
    type: "scatter", x: dates, y: ema50, name: "EMA50", line: { color: "#E67E22", width: 1.5, dash: "dot" }
  }], Object.assign({}, layout_base, { title: ticker + " 캔들스틱 차트" }), { responsive: true });

  const volColors = closes.map((c, i) => i === 0 ? "#95A5A6" : c >= closes[i-1] ? "#27AE60" : "#E74C3C");
  Plotly.newPlot("plotly-volume", [{
    type: "bar", x: dates, y: volumes, name: "거래량",
    marker: { color: volColors }
  }], Object.assign({}, layout_base, { title: "거래량" }), { responsive: true });

  Plotly.newPlot("plotly-rsi", [{
    type: "scatter", x: dates, y: rsi, name: "RSI", line: { color: "#9B59B6", width: 2 }
  }, {
    type: "scatter", x: dates, y: Array(dates.length).fill(70), name: "과매수(70)", line: { color: "#E74C3C", dash: "dash", width: 1 }
  }, {
    type: "scatter", x: dates, y: Array(dates.length).fill(30), name: "과매도(30)", line: { color: "#27AE60", dash: "dash", width: 1 }
  }], Object.assign({}, layout_base, { title: "RSI (14)", yaxis: { gridcolor: "#F0F0F0", range: [0, 100] } }), { responsive: true });

  const macdHist = macd.map((v, i) => (v != null && macdSig[i] != null) ? v - macdSig[i] : null);
  Plotly.newPlot("plotly-macd", [{
    type: "bar", x: dates, y: macdHist, name: "MACD Histogram",
    marker: { color: macdHist.map(v => v == null ? "#95A5A6" : v >= 0 ? "#27AE60" : "#E74C3C") }
  }, {
    type: "scatter", x: dates, y: macd, name: "MACD", line: { color: "#3498DB", width: 2 }
  }, {
    type: "scatter", x: dates, y: macdSig, name: "Signal", line: { color: "#E67E22", width: 2 }
  }], Object.assign({}, layout_base, { title: "MACD" }), { responsive: true });

  Plotly.newPlot("plotly-bb", [{
    type: "scatter", x: dates, y: bbUpper, name: "BB Upper", line: { color: "#E74C3C", width: 1, dash: "dot" }, fill: "none"
  }, {
    type: "scatter", x: dates, y: bbLower, name: "BB Lower", line: { color: "#27AE60", width: 1, dash: "dot" },
    fill: "tonexty", fillcolor: "rgba(52,152,219,0.08)"
  }, {
    type: "scatter", x: dates, y: closes, name: "종가", line: { color: "#3498DB", width: 2 }
  }, {
    type: "scatter", x: dates, y: ema20, name: "EMA20", line: { color: "#E67E22", width: 1.5 }
  }], Object.assign({}, layout_base, { title: "볼린저 밴드" }), { responsive: true });
}

function switchTab(panelId, btn) {
  const bar = btn.closest(".tab-bar");
  bar.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  let el = bar.nextElementSibling;
  while (el && el.classList.contains("tab-panel")) {
    el.classList.remove("active");
    el = el.nextElementSibling;
  }
  document.getElementById(panelId).classList.add("active");
}

function showLoading(msg) {
  document.getElementById("loading-text").textContent = msg || "분석 중...";
  document.getElementById("loading-overlay").classList.add("active");
}
function updateLoadingText(msg) {
  document.getElementById("loading-text").textContent = msg;
}
function hideLoading() {
  document.getElementById("loading-overlay").classList.remove("active");
}
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
# Pydantic 모델
# ─────────────────────────────────────────────────────────────
class FearGreedResponse(BaseModel):
    value: int
    classification: str

class IndicatorData(BaseModel):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None

class SignalData(BaseModel):
    signal: str
    total_score: float
    confidence: float
    rsi_score: float
    trend_score: float

class BacktestResult(BaseModel):
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    final_capital: float

class TradingMetrics(BaseModel):
    returns: Dict[str, float]
    buy_sell_ratio: float
    sentiment: str

class FullAnalysisResponse(BaseModel):
    ticker: str
    current_price: float
    indicators: IndicatorData
    signal: SignalData
    backtest: BacktestResult
    metrics: TradingMetrics
    chart_data: List[Dict[str, Any]]

# ─────────────────────────────────────────────────────────────
# interval별 최대 허용 일수 (yfinance 제약)
# ─────────────────────────────────────────────────────────────
INTERVAL_MAX_DAYS: Dict[str, int] = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h":  730,
    "1d":  1825,
    "1wk": 1825,
    "1mo": 1825,
}

# interval별 연간 거래 캔들 수 (샤프 비율 annualization 계수)
ANNUALIZE_FACTOR: Dict[str, float] = {
    "1m":  525600,
    "2m":  262800,
    "5m":  105120,
    "15m": 35040,
    "30m": 17520,
    "60m": 8760,
    "1h":  8760,
    "1d":  365,
    "1wk": 52,
    "1mo": 12,
}

# ─────────────────────────────────────────────────────────────
# 라우트
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """메인 페이지 - HTML 직접 서빙"""
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "9.1.0"}

@app.get("/api/fear-greed", response_model=FearGreedResponse)
def get_fear_greed_index():
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=8,
            headers={"User-Agent": "SignalCoin/9.1.0"}
        )
        r.raise_for_status()
        data = r.json()["data"][0]
        return {"value": int(data["value"]), "classification": data["value_classification"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Fear & Greed API 호출 실패: {str(e)}")

@app.get("/api/analyze", response_model=FullAnalysisResponse)
def get_full_analysis(
    ticker: str = Query("BTC-USD", description="yfinance 티커 (예: BTC-USD)"),
    days: int = Query(365, ge=1, le=1825, description="분석 기간(일)"),
    interval: str = Query("1d", description="시간 프레임 (1m/5m/1h/1d 등)")
):
    # interval 유효성 검사
    allowed_intervals = {"1m","2m","5m","15m","30m","60m","1h","1d","1wk","1mo"}
    if interval not in allowed_intervals:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 interval: {interval}")

    # interval별 최대 일수 강제 적용
    max_days = INTERVAL_MAX_DAYS.get(interval, 1825)
    days = min(days, max_days)

    try:
        df = _fetch_ohlcv(ticker, days, interval)
        df_ind = _calculate_all_indicators(df.copy())
        ann_factor = ANNUALIZE_FACTOR.get(interval, 365)
        signal_result = _calculate_signal_score(df_ind)
        backtest_result = _run_simple_backtest(df, ann_factor)
        metrics_result = _calculate_trading_metrics(df)

        df_chart = df_ind.reset_index()

        # 날짜 컬럼 정규화 (일봉=Date, 분/시간봉=Datetime)
        if "Datetime" in df_chart.columns:
            df_chart["Date"] = pd.to_datetime(df_chart["Datetime"]).dt.strftime("%Y-%m-%d %H:%M")
            df_chart.drop(columns=["Datetime"], errors="ignore", inplace=True)
        elif "Date" in df_chart.columns:
            df_chart["Date"] = pd.to_datetime(df_chart["Date"]).dt.strftime("%Y-%m-%d")
        else:
            # 인덱스가 날짜인 경우 (reset_index 후 첫 번째 컬럼)
            first_col = df_chart.columns[0]
            df_chart["Date"] = pd.to_datetime(df_chart[first_col]).dt.strftime("%Y-%m-%d %H:%M")
            df_chart.drop(columns=[first_col], errors="ignore", inplace=True)

        # 필요한 컬럼만 선택 (불필요한 컬럼 제거로 응답 크기 최소화)
        keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume",
                     "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "ema20", "ema50"]
        df_chart = df_chart[[c for c in keep_cols if c in df_chart.columns]]
        chart_data = _sanitize_chart_data(df_chart)

        latest = df_ind.iloc[-1]
        indicators = IndicatorData(
            rsi=_safe_float(latest.get("rsi")),
            macd=_safe_float(latest.get("macd")),
            macd_signal=_safe_float(latest.get("macd_signal")),
            bb_upper=_safe_float(latest.get("bb_upper")),
            bb_lower=_safe_float(latest.get("bb_lower")),
            ema20=_safe_float(latest.get("ema20")),
            ema50=_safe_float(latest.get("ema50")),
        )

        # 현재 가격: Close 컬럼에서 안전하게 추출
        close_series = df["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        current_price = float(close_series.iloc[-1])

        return FullAnalysisResponse(
            ticker=ticker,
            current_price=current_price,
            indicators=indicators,
            signal=SignalData(**signal_result),
            backtest=BacktestResult(**backtest_result),
            metrics=TradingMetrics(**metrics_result),
            chart_data=chart_data,
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        detail = f"분석 오류: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

# ─────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """NaN/Inf/None을 None으로 안전하게 변환"""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None

def _sanitize_chart_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame을 JSON 직렬화 가능한 형태로 변환 (NaN → None)"""
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            if isinstance(val, (float, np.floating)):
                record[col] = _safe_float(val)
            elif isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.bool_,)):
                record[col] = bool(val)
            else:
                record[col] = val
        records.append(record)
    return records

def _fetch_ohlcv(ticker: str, days: int, interval: str = "1d") -> pd.DataFrame:
    """yfinance로 OHLCV 데이터 다운로드 및 컬럼 정규화"""
    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"yfinance 데이터 수신 실패: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"데이터 없음: '{ticker}' (interval={interval}). 올바른 yfinance 티커와 기간인지 확인하세요."
        )

    # MultiIndex 컬럼 평탄화 (yfinance 0.2.x 이상에서 발생)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 컬럼명 중복 제거 (동일 이름이 있을 경우)
    df = df.loc[:, ~df.columns.duplicated()]

    # 필수 컬럼 확인
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=500, detail=f"데이터 컬럼 누락: {missing}")

    return df

def _get_close_series(df: pd.DataFrame) -> pd.Series:
    """Close 컬럼을 1차원 Series로 안전하게 추출"""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()

def _calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산 (RSI, MACD, 볼린저밴드, EMA)"""
    close = _get_close_series(df)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드 (20, 2σ)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # EMA
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    # NaN 행 제거 (초기 계산 불가 구간)
    df = df.dropna(subset=["rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "ema20", "ema50"])
    return df

def _calculate_signal_score(df: pd.DataFrame) -> dict:
    """AI 종합 신호 점수 계산"""
    latest = df.iloc[-1]
    rsi = float(latest["rsi"])
    close = float(_get_close_series(df).iloc[-1])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])

    # RSI 기반 점수 (0~100)
    rsi_score = (100 - rsi) * 0.8 if rsi > 50 else rsi * 1.2

    # 추세 기반 점수 (EMA 배열 분석)
    trend_score = 50.0
    if close > ema20 > ema50:
        trend_score = 85.0   # 강한 상승 추세
    elif close > ema20:
        trend_score = 65.0   # 약한 상승 추세
    elif close < ema20 < ema50:
        trend_score = 15.0   # 강한 하락 추세
    elif close < ema20:
        trend_score = 35.0   # 약한 하락 추세

    total = float(np.clip(rsi_score * 0.5 + trend_score * 0.5, 0, 100))

    if total >= 75:
        signal = "STRONG_BUY"
    elif total >= 60:
        signal = "BUY"
    elif total <= 25:
        signal = "STRONG_SELL"
    elif total <= 40:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "total_score": total,
        "confidence": float(min(abs(total - 50) * 2, 100)),
        "rsi_score": float(rsi_score),
        "trend_score": float(trend_score),
    }

def _run_simple_backtest(df: pd.DataFrame, ann_factor: float = 365) -> dict:
    """Buy & Hold 전략 백테스트"""
    close = _get_close_series(df)
    returns = close.pct_change().dropna()

    if len(returns) < 2:
        return {
            "total_return_pct": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "final_capital": 1000.0
        }

    cum = (1 + returns).cumprod()
    total_return = float((cum.iloc[-1] - 1) * 100)
    sharpe = float((returns.mean() / (returns.std() + 1e-10)) * math.sqrt(ann_factor))
    running_max = cum.cummax()
    max_dd = float(((cum - running_max) / (running_max + 1e-10)).min() * 100)
    win_rate = float((returns > 0).sum() / len(returns) * 100)
    final_cap = float(1000 * cum.iloc[-1])

    return {
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "final_capital": final_cap,
    }

def _calculate_trading_metrics(df: pd.DataFrame) -> dict:
    """기간별 수익률 및 매매 비율 계산"""
    close = _get_close_series(df)
    n = len(close)
    returns = {}
    for period, pd_days in {"1w": 7, "1m": 30, "3m": 90}.items():
        if n > pd_days:
            base = float(close.iloc[-pd_days])
            curr = float(close.iloc[-1])
            returns[period] = float((curr - base) / base * 100) if base != 0 else 0.0
        else:
            returns[period] = 0.0

    vol_series = df["Volume"]
    if isinstance(vol_series, pd.DataFrame):
        vol_series = vol_series.iloc[:, 0]
    vol_series = vol_series.squeeze()

    vol_recent = float(vol_series.iloc[-5:].mean())
    vol_avg = float(vol_series.mean())
    buy_ratio = float(np.clip(50 + (vol_recent / (vol_avg + 1e-10) - 1) * 50, 0, 100))
    sentiment = "BULLISH" if buy_ratio > 60 else "BEARISH" if buy_ratio < 40 else "NEUTRAL"

    return {"returns": returns, "buy_sell_ratio": buy_ratio, "sentiment": sentiment}
