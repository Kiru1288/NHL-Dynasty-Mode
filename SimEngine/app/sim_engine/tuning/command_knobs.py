"""Franchise Command knobs: runtime multipliers set by the commissioner sliders.

The backend writes the active save's slider values here (``set_knobs``) whenever a
slider moves and at the start of every sim day. Engine code reads a knob with
``knob(name, default)``; nothing is stored on players, so a slider change is live on
the next tick and leaves no residue when it is moved back.

The scoring environment (``scoring_env``) is how progression keeps production fair:
70 points in a high-scoring league is worth less than 70 points in a dead-puck one.
"""
from __future__ import annotations

from typing import Any, Dict

KNOBS: Dict[str, float] = {}


def set_knobs(values: Dict[str, Any]) -> None:
    KNOBS.clear()
    for k, v in (values or {}).items():
        try:
            KNOBS[str(k)] = float(v)
        except (TypeError, ValueError):
            continue


def knob(name: str, default: float = 1.0) -> float:
    try:
        return float(KNOBS.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def scoring_env(is_defense: bool = False) -> float:
    """League scoring level versus the default (1.0 = default sliders).

    Set from the on-ice sliders (scoring pace, power play, penalties). When the
    "Normalize production" slider is off, raw points are judged as-is.
    """
    if knob("era_normalize", 1.0) < 0.5:
        return 1.0
    env = knob("scoring_env", 1.0)
    if is_defense:
        env *= knob("defense_scoring_env", 1.0)
    return max(0.6, min(1.6, env))
