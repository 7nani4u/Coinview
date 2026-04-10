# -*- coding: utf-8 -*-
"""Central configuration for CoinOracle."""

APP_NAME = "CoinOracle"
APP_VERSION = "2.1.0"
APP_DESCRIPTION = "Binance 기반 암호화폐 규칙 기반 분석 및 시그널 시스템"
APP_TAGLINE = "검증 가능한 규칙 기반 분석 · 예측 값은 참고용"

BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]
BINANCE_FAPI = "https://fapi.binance.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
REQUEST_HEADERS = {"User-Agent": "CoinOracle/2.1 (+Vercel)"}
DEFAULT_TIMEOUT = 6
SCREENER_TIMEOUT = 8

CACHE_TTL_KLINES = 60
CACHE_TTL_TICKER = 30
CACHE_TTL_FUNDING = 30
CACHE_TTL_OI = 30
CACHE_TTL_COINGECKO = 300
CACHE_TTL_NEWS = 120
CACHE_TTL_COIN_DATA = 60
CACHE_TTL_SCREENER = 300
CACHE_TTL_SENTIMENT = 300
CACHE_TTL_VALIDATION = 300
CACHE_TTL_HEALTH = 30

SCREENER_MAX_WORKERS = 8
SCREENER_TOP_N = 30
SCREENER_VALID_SORT = {"name", "price", "change", "volume", "rsi", "leverage"}
SCREENER_VALID_ORDER = {"asc", "desc"}
SCREENER_INTERVALS = {"1d": 90, "4h": 120}

DEFAULT_INTERVAL = "1d"
DEFAULT_LIMIT = 365
MIN_LIMIT = 50
MAX_LIMIT = 1000
VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

VALIDATION_DEFAULT_WINDOW = 180
VALIDATION_MIN_WINDOW = 90
VALIDATION_MAX_WINDOW = 365
VALIDATION_DEFAULT_HORIZON = 7
VALIDATION_MIN_HORIZON = 1
VALIDATION_MAX_HORIZON = 30

COIN_ALIASES = {
    "비트코인": "BTCUSDT", "BTC": "BTCUSDT", "비트": "BTCUSDT",
    "이더리움": "ETHUSDT", "ETH": "ETHUSDT", "이더": "ETHUSDT",
    "바이낸스코인": "BNBUSDT", "BNB": "BNBUSDT",
    "리플": "XRPUSDT", "XRP": "XRPUSDT",
    "솔라나": "SOLUSDT", "SOL": "SOLUSDT",
    "에이다": "ADAUSDT", "ADA": "ADAUSDT", "카르다노": "ADAUSDT",
    "도지": "DOGEUSDT", "DOGE": "DOGEUSDT", "도지코인": "DOGEUSDT",
    "아발란체": "AVAXUSDT", "AVAX": "AVAXUSDT",
    "시바이누": "SHIBUSDT", "SHIB": "SHIBUSDT",
    "폴카닷": "DOTUSDT", "DOT": "DOTUSDT",
    "체인링크": "LINKUSDT", "LINK": "LINKUSDT",
    "유니스왑": "UNIUSDT", "UNI": "UNIUSDT",
    "아톰": "ATOMUSDT", "ATOM": "ATOMUSDT",
    "리테라코인": "LTCUSDT", "LTC": "LTCUSDT", "라이트코인": "LTCUSDT",
    "이더리움클래식": "ETCUSDT", "ETC": "ETCUSDT",
    "비트코인캐시": "BCHUSDT", "BCH": "BCHUSDT",
    "수이": "SUIUSDT", "SUI": "SUIUSDT",
    "아비트럼": "ARBUSDT", "ARB": "ARBUSDT",
    "옵티미즘": "OPUSDT", "OP": "OPUSDT",
    "매틱": "POLUSDT", "MATIC": "MATICUSDT", "폴리곤": "POLUSDT",
    "니어": "NEARUSDT", "NEAR": "NEARUSDT",
    "인젝티브": "INJUSDT", "INJ": "INJUSDT",
    "셀레스티아": "TIAUSDT", "TIA": "TIAUSDT",
    "아이오타": "IOTAUSDT", "IOTA": "IOTAUSDT",
    "앱토스": "APTUSDT", "APT": "APTUSDT",
    "필코인": "FILUSDT", "FIL": "FILUSDT",
    "샌드박스": "SANDUSDT", "SAND": "SANDUSDT",
    "디센트럴랜드": "MANAUSDT", "MANA": "MANAUSDT",
    "아이오에스티": "IOSTUSDT", "IOST": "IOSTUSDT",
    "트론": "TRXUSDT", "TRX": "TRXUSDT",
    "스텔라": "XLMUSDT", "XLM": "XLMUSDT",
    "에이브": "AAVEUSDT", "AAVE": "AAVEUSDT",
    "커브": "CRVUSDT", "CRV": "CRVUSDT",
    "렌더": "RENDERUSDT", "RENDER": "RENDERUSDT",
    "월드코인": "WLDUSDT", "WLD": "WLDUSDT",
    "페페": "PEPEUSDT", "PEPE": "PEPEUSDT",
    "봉크": "BONKUSDT", "BONK": "BONKUSDT",
    "플로키": "FLOKIUSDT", "FLOKI": "FLOKIUSDT",
    "비트텐서": "TAOUSDT", "TAO": "TAOUSDT",
    "에테나": "ENAUSDT", "ENA": "ENAUSDT",
    "제트로": "ZROUSDT", "ZRO": "ZROUSDT",
    "세이": "SEIUSDT", "SEI": "SEIUSDT",
    "칠리즈": "CHZUSDT", "CHZ": "CHZUSDT",
    "펜구": "PENGUUSDT", "PENGU": "PENGUUSDT",
    "트럼프": "TRUMPUSDT", "TRUMP": "TRUMPUSDT",
}

COIN_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT", "BCHUSDT",
    "SUIUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT", "INJUSDT",
    "TIAUSDT", "APTUSDT", "FILUSDT", "AAVEUSDT", "CRVUSDT",
    "RENDERUSDT", "WLDUSDT", "PEPEUSDT", "BONKUSDT", "TAOUSDT",
    "ENAUSDT", "SEIUSDT", "CHZUSDT", "TRXUSDT", "XLMUSDT",
    "SANDUSDT", "MANAUSDT", "MATICUSDT", "SHIBUSDT", "FLOKIUSDT",
]
