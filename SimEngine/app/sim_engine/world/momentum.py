# app/sim_engine/world/momentum.py
"""Team momentum [-1, 1]: streak-driven, blowout-sensitive, slow decay."""

from __future__ import annotations

from typing import Any, Optional

KEY = "_world_momentum"
STREAK_W = "_world_streak"  # signed int: wins positive, losses negative


def _get(team: Any) -> float:
    v = getattr(team, KEY, None)
    if v is None:
        return 0.0
    return max(-1.0, min(1.0, float(v)))


def _set(team: Any, v: float) -> None:
    setattr(team, KEY, max(-1.0, min(1.0, float(v))))


def get_team_momentum(team: Any) -> float:
    return _get(team)


def team_strength_modifier(team: Any) -> float:
    """~0.95 at min momentum, ~1.05 at max."""
    m = _get(team)
    return 1.0 + 0.05 * m


def init_team_momentum(team: Any) -> None:
    if getattr(team, KEY, None) is None:
        setattr(team, KEY, 0.0)
    if getattr(team, STREAK_W, None) is None:
        setattr(team, STREAK_W, 0)


def decay_momentum(team: Any, days: float = 1.0) -> None:
    """Gradual pull toward 0."""
    if days <= 0:
        return
    m = _get(team)
    decay = 0.04 * float(days)
    if m > 0:
        _set(team, max(0.0, m - decay))
    elif m < 0:
        _set(team, min(0.0, m + decay))


def update_momentum_after_game(team: Any, goals_for: int, goals_against: int, rng: Any) -> None:
    init_team_momentum(team)
    diff = int(goals_for) - int(goals_against)
    streak = int(getattr(team, STREAK_W, 0) or 0)
    blow = abs(diff) >= 3
    if diff > 0:
        streak = max(0, streak) + 1
        delta = 0.06 + 0.02 * min(5, streak) + (0.04 if blow else 0.0)
        delta *= 0.92 + 0.08 * rng.random()
    elif diff < 0:
        streak = min(0, streak) - 1
        delta = -(0.06 + 0.02 * min(5, -streak) + (0.04 if blow else 0.0))
        delta *= 0.92 + 0.08 * rng.random()
    else:
        streak = 0 if abs(streak) < 2 else int(streak * 0.5)
        delta = 0.0
    setattr(team, STREAK_W, streak)
    _set(team, _get(team) + delta)


def decay_all_teams(teams: Any, factor: float = 1.0) -> None:
    for t in teams or []:
        decay_momentum(t, factor)
