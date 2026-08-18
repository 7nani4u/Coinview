# -*- coding: utf-8 -*-
"""Backtesting helpers for forecast, score, and leverage validation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def safe_pct_change(current: float, future: float) -> float:
    if not current:
        return 0.0
    return (future - current) / current * 100.0


def calc_directional_accuracy(predicted: List[float], actual: List[float], baseline: List[float]) -> float | None:
    total = min(len(predicted), len(actual), len(baseline))
    if total <= 0:
        return None
    hits = 0
    for pred, act, base in zip(predicted[:total], actual[:total], baseline[:total]):
        pred_dir = 1 if pred >= base else -1
        act_dir = 1 if act >= base else -1
        hits += int(pred_dir == act_dir)
    return round(hits / total, 4)


def calc_mape(predicted: List[float], actual: List[float]) -> float | None:
    if not predicted or not actual:
        return None
    errors = []
    for pred, act in zip(predicted, actual):
        if act:
            errors.append(abs((act - pred) / act) * 100.0)
    if not errors:
        return None
    return round(float(np.mean(errors)), 4)


def summarize_signal_outcomes(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not outcomes:
        return {"samples": 0, "hit_rate": None, "avg_return_pct": None, "median_return_pct": None}
    returns = [item["future_return_pct"] for item in outcomes]
    wins = [item for item in outcomes if item["future_return_pct"] > 0]
    return {
        "samples": len(outcomes),
        "hit_rate": round(len(wins) / len(outcomes), 4),
        "avg_return_pct": round(float(np.mean(returns)), 4),
        "median_return_pct": round(float(np.median(returns)), 4),
    }


def summarize_leverage_outcomes(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not outcomes:
        return {"samples": 0, "stop_breach_rate": None, "avg_recommended_leverage": None, "median_recommended_leverage": None}
    breaches = [item for item in outcomes if item["realized_abs_return_pct"] > item["stop_loss_pct"]]
    leverages = [item["recommended_leverage"] for item in outcomes]
    return {
        "samples": len(outcomes),
        "stop_breach_rate": round(len(breaches) / len(outcomes), 4),
        "avg_recommended_leverage": round(float(np.mean(leverages)), 4),
        "median_recommended_leverage": round(float(np.median(leverages)), 4),
    }
