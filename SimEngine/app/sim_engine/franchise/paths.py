"""Ensure SimEngine root is on sys.path for `app.*` imports."""

from __future__ import annotations

import sys
from pathlib import Path

_SIMENGINE_ROOT: Path | None = None


def simengine_root() -> Path:
    global _SIMENGINE_ROOT
    if _SIMENGINE_ROOT is None:
        # .../SimEngine/app/sim_engine/franchise/paths.py -> SimEngine
        _SIMENGINE_ROOT = Path(__file__).resolve().parents[3]
    return _SIMENGINE_ROOT


def ensure_simengine_path() -> Path:
    root = simengine_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
