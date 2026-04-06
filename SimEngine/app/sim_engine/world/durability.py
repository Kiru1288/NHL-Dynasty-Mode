# app/sim_engine/world/durability.py
"""Per-player durability [0,1]: injury resistance, fatigue buildup dampener, ages down."""

from __future__ import annotations

from typing import Any, Optional

KEY = "_world_durability"


def _age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    if ident is not None and hasattr(ident, "age"):
        return int(ident.age)
    return int(getattr(player, "age", 26))


def init_player_durability(player: Any) -> None:
    if getattr(player, KEY, None) is not None:
        return
    r = getattr(player, "ratings", None)
    base = 0.52
    if isinstance(r, dict):
        ph = float(r.get("ph_injury_resistance", 50)) / 99.0
        dur = float(r.get("ph_durability", 50)) / 99.0
        base = 0.45 + 0.38 * (0.5 * ph + 0.5 * dur)
    setattr(player, KEY, max(0.18, min(0.96, base)))


def get_durability(player: Any) -> float:
    init_player_durability(player)
    return float(getattr(player, KEY, 0.55))


def fatigue_buildup_multiplier(player: Any) -> float:
    """Higher durability => slower fatigue accumulation."""
    d = get_durability(player)
    return 1.15 - 0.35 * d


def injury_chance_multiplier(player: Any) -> float:
    """Higher durability => lower injury propensity."""
    d = get_durability(player)
    return 0.55 + 0.90 * (1.0 - d)


def apply_season_aging_durability(player: Any) -> None:
    """Small yearly decline for veterans."""
    init_player_durability(player)
    age = _age(player)
    if age < 30:
        return
    years = max(0, age - 29)
    penalty = min(0.14, 0.008 * years)
    v = float(getattr(player, KEY, 0.55)) - penalty
    setattr(player, KEY, max(0.15, min(0.98, v)))
