# -*- coding: utf-8 -*-
"""Input validation helpers."""

from __future__ import annotations

import re
from typing import Tuple

from api.config import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MIN_LIMIT,
    VALIDATION_DEFAULT_HORIZON,
    VALIDATION_DEFAULT_WINDOW,
    VALIDATION_MAX_HORIZON,
    VALIDATION_MAX_WINDOW,
    VALIDATION_MIN_HORIZON,
    VALIDATION_MIN_WINDOW,
    VALID_INTERVALS,
)

TICKER_RE = re.compile(r"^[a-zA-Z0-9가-힣.\-\s]+$")


def validate_ticker(value: str) -> Tuple[bool, str]:
    ticker = (value or "").strip()
    if not ticker:
        return True, ""
    if len(ticker) > 30 or not TICKER_RE.match(ticker):
        return False, "Invalid ticker format"
    return True, ticker


def validate_interval(value: str) -> Tuple[bool, str]:
    interval = (value or DEFAULT_INTERVAL).strip()
    if interval not in VALID_INTERVALS:
        return False, f"Invalid interval. Allowed: {', '.join(sorted(VALID_INTERVALS))}"
    return True, interval


def normalize_interval(value: str) -> str:
    interval = (value or DEFAULT_INTERVAL).strip()
    return interval if interval in VALID_INTERVALS else DEFAULT_INTERVAL


def clamp_int(value: str, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(lower, min(upper, parsed))


def normalize_limit(value: str) -> int:
    return clamp_int(value, DEFAULT_LIMIT, MIN_LIMIT, MAX_LIMIT)


def normalize_validation_window(value: str) -> int:
    return clamp_int(value, VALIDATION_DEFAULT_WINDOW, VALIDATION_MIN_WINDOW, VALIDATION_MAX_WINDOW)


def normalize_validation_horizon(value: str) -> int:
    return clamp_int(value, VALIDATION_DEFAULT_HORIZON, VALIDATION_MIN_HORIZON, VALIDATION_MAX_HORIZON)
