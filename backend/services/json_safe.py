"""JSON-safe payload sanitization for FastAPI / Pydantic serialization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


def json_safe(value: Any, *, depth: int = 0, _stack: Optional[Set[int]] = None) -> Any:
    """Strip callables / unknown objects so FastAPI can serialize responses.

    Shared object references (same dict in ``skaters`` and ``league_leaders``)
    are deep-copied. Only true circular references are collapsed to ``None``.
    """
    if depth > 16:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return None
    if callable(value):
        return None

    if _stack is None:
        _stack = set()

    oid = id(value)
    if isinstance(value, dict):
        if oid in _stack:
            return None
        _stack.add(oid)
        try:
            out: Dict[str, Any] = {}
            for k, v in value.items():
                if callable(v):
                    continue
                cleaned = json_safe(v, depth=depth + 1, _stack=_stack)
                # Keep False / 0 / "" — `cleaned is not None` alone drops False and
                # broke Stats Central (war_valid=false stripped → UI showed WAR 0.00).
                if cleaned is not None or v is None or v is False:
                    out[str(k)] = cleaned
            return out
        finally:
            _stack.discard(oid)

    if isinstance(value, (list, tuple, set)):
        if oid in _stack:
            return None
        _stack.add(oid)
        try:
            return [
                json_safe(v, depth=depth + 1, _stack=_stack)
                for v in value
                if not callable(v)
            ]
        finally:
            _stack.discard(oid)

    if hasattr(value, "__dict__"):
        if oid in _stack:
            return None
        _stack.add(oid)
        try:
            try:
                return json_safe(vars(value), depth=depth + 1, _stack=_stack)
            except Exception:
                try:
                    return str(value)
                except Exception:
                    return None
        finally:
            _stack.discard(oid)

    try:
        return str(value)
    except Exception:
        return None


def find_methods(value: Any, path: str = "root", depth: int = 0, _seen: Optional[Set[int]] = None) -> List[str]:
    """Debug helper: paths to callables nested in a payload."""
    hits: List[str] = []
    if depth > 14:
        return hits
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return hits
    if callable(value) and not isinstance(value, type):
        hits.append(f"{path} ({getattr(value, '__qualname__', type(value).__name__)})")
        return hits
    if _seen is None:
        _seen = set()
    oid = id(value)
    if oid in _seen:
        return hits
    _seen.add(oid)
    if isinstance(value, dict):
        for k, v in value.items():
            hits.extend(find_methods(v, f"{path}.{k}", depth + 1, _seen))
    elif isinstance(value, (list, tuple, set)):
        for i, v in enumerate(list(value)[:80]):
            hits.extend(find_methods(v, f"{path}[{i}]", depth + 1, _seen))
    return hits
