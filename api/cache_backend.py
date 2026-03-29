# -*- coding: utf-8 -*-
"""In-memory TTL cache backend for serverless-friendly short-lived caching."""

from __future__ import annotations

import functools
import time
from typing import Any, Dict, Tuple

_CACHE: Dict[str, Tuple[Any, float]] = {}


def _now() -> float:
    return time.time()


def _purge() -> None:
    now = _now()
    expired = [key for key, (_, expires_at) in _CACHE.items() if expires_at <= now]
    for key in expired:
        _CACHE.pop(key, None)


def ttl_cache(ttl: int):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _purge()
            key = f"{fn.__name__}|{args}|{sorted(kwargs.items())}"
            item = _CACHE.get(key)
            now = _now()
            if item and item[1] > now:
                return item[0]
            value = fn(*args, **kwargs)
            
            # 실패 응답(None, Exception을 포함한 dict, 에러 메시지 등)은 캐시하지 않음
            should_cache = True
            if value is None:
                should_cache = False
            elif isinstance(value, dict) and ("error" in value or "status" in value and value["status"] == "error"):
                should_cache = False
                
            if should_cache:
                _CACHE[key] = (value, now + ttl)
            return value
        return wrapper
    return deco


def cache_meta() -> Dict[str, Any]:
    _purge()
    now = _now()
    return {
        "backend": "memory_ttl",
        "entries": sum(1 for _, expires_at in _CACHE.values() if expires_at > now),
        "serverless_note": "인스턴스 간 공유되지 않음",
    }
