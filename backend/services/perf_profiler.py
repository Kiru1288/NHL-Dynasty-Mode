"""
Process-wide performance profiler for NHL Franchise Mode.

Behavior-preserving instrumentation only — records timings, never changes outcomes.
Enable detailed logs with NHL_PERF=1. Snapshot via GET /api/perf/snapshot.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

_LOCK = threading.Lock()
_ENABLED = os.environ.get("NHL_PERF", "1").strip().lower() not in ("0", "false", "off", "no")
_LOG_SLOW_MS = float(os.environ.get("NHL_PERF_SLOW_MS", "100") or 100)


class _Bucket:
    __slots__ = ("count", "total_ms", "max_ms", "min_ms", "last_ms")

    def __init__(self) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.min_ms = float("inf")
        self.last_ms = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms
        if ms < self.min_ms:
            self.min_ms = ms

    def as_dict(self) -> Dict[str, Any]:
        avg = (self.total_ms / self.count) if self.count else 0.0
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(avg, 3),
            "max_ms": round(self.max_ms, 3),
            "min_ms": round(self.min_ms if self.count else 0.0, 3),
            "last_ms": round(self.last_ms, 3),
        }


_BUCKETS: Dict[str, _Bucket] = {}
_RECENT: List[Dict[str, Any]] = []
_RECENT_LIMIT = 200
_STARTED_AT = time.time()


def is_enabled() -> bool:
    return _ENABLED


def record(name: str, duration_ms: float, *, meta: Optional[Dict[str, Any]] = None) -> None:
    if not _ENABLED:
        return
    ms = float(duration_ms)
    with _LOCK:
        bucket = _BUCKETS.get(name)
        if bucket is None:
            bucket = _Bucket()
            _BUCKETS[name] = bucket
        bucket.add(ms)
        entry = {
            "name": name,
            "ms": round(ms, 3),
            "t": time.time(),
        }
        if meta:
            entry["meta"] = meta
        _RECENT.append(entry)
        if len(_RECENT) > _RECENT_LIMIT:
            del _RECENT[: len(_RECENT) - _RECENT_LIMIT]
    if ms >= _LOG_SLOW_MS:
        try:
            import logging

            logging.getLogger("uvicorn.error").info(
                "[perf] SLOW %s %.1fms%s",
                name,
                ms,
                f" {meta}" if meta else "",
            )
        except Exception:
            pass


@contextmanager
def span(name: str, **meta: Any) -> Iterator[None]:
    if not _ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(name, (time.perf_counter() - t0) * 1000.0, meta=meta or None)


def timed(name: Optional[str] = None):
    """Decorator for sync functions."""

    def deco(fn):
        label = name or f"{fn.__module__}.{fn.__qualname__}"

        def wrapper(*args, **kwargs):
            if not _ENABLED:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                record(label, (time.perf_counter() - t0) * 1000.0)

        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper

    return deco


def snapshot(*, top_n: int = 40) -> Dict[str, Any]:
    with _LOCK:
        rows = [
            {"name": k, **v.as_dict()}
            for k, v in _BUCKETS.items()
        ]
        recent = list(_RECENT[-50:])
    rows.sort(key=lambda r: (-float(r.get("total_ms") or 0), -float(r.get("max_ms") or 0)))
    slow = [r for r in rows if float(r.get("max_ms") or 0) >= _LOG_SLOW_MS]
    return {
        "ok": True,
        "enabled": _ENABLED,
        "uptime_s": round(time.time() - _STARTED_AT, 1),
        "slow_threshold_ms": _LOG_SLOW_MS,
        "bucket_count": len(rows),
        "top_by_total_ms": rows[: max(1, int(top_n))],
        "slow_by_max_ms": sorted(slow, key=lambda r: -float(r.get("max_ms") or 0))[:top_n],
        "recent": recent,
    }


def reset() -> Dict[str, Any]:
    with _LOCK:
        _BUCKETS.clear()
        _RECENT.clear()
    return {"ok": True, "cleared": True}
