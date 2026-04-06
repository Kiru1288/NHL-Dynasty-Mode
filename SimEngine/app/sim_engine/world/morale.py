# app/sim_engine/world/morale.py
"""Player morale on 0–100 scale; bridges psych.morale [0,1] and game effects."""

from __future__ import annotations

from typing import Any

KEY = "_world_morale_100"


def sync_from_psych(player: Any) -> float:
    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        m01 = float(psych.morale)
        setattr(player, KEY, max(0.0, min(100.0, m01 * 100.0)))
        return float(getattr(player, KEY))
    v = getattr(player, KEY, None)
    if v is None:
        setattr(player, KEY, 52.0)
    return float(getattr(player, KEY, 52.0))


def get_morale_100(player: Any) -> float:
    return sync_from_psych(player)


def push_to_psych(player: Any, value_100: float) -> None:
    setattr(player, KEY, max(0.0, min(100.0, float(value_100))))
    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        psych.morale = max(0.0, min(1.0, float(value_100) / 100.0))


def performance_factor(player: Any) -> float:
    m = get_morale_100(player)
    # 50 -> 1.0 ; high -> +1.6% ; low -> -1.6%
    return max(0.982, min(1.018, 1.0 + 0.00032 * (m - 50.0)))


def development_penalty_factor(player: Any) -> float:
    """Sub-40 morale slightly hurts implicit growth hooks."""
    m = get_morale_100(player)
    if m >= 42.0:
        return 1.0
    return max(0.94, 0.98 - 0.001 * (42.0 - m))


def update_after_team_result(
    player: Any,
    team_won: bool,
    goal_diff_for_team: int,
    rng: Any,
    role_satisfaction_proxy: float = 0.5,
) -> None:
    m = get_morale_100(player)
    delta = 0.0
    if team_won:
        delta += 1.1 + 0.35 * max(0, min(4, goal_diff_for_team))
    else:
        delta -= 1.0 + 0.3 * max(0, min(4, -goal_diff_for_team))
    delta += (float(role_satisfaction_proxy) - 0.5) * 0.4
    delta *= 0.9 + 0.12 * rng.random()
    push_to_psych(player, m + delta)


def team_avg_morale(team: Any) -> float:
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    if not roster:
        return 50.0
    return sum(get_morale_100(p) for p in roster) / len(roster)


def team_morale_strength_factor(team: Any) -> float:
    avg = team_avg_morale(team)
    return max(0.985, min(1.015, 1.0 + 0.00025 * (avg - 50.0)))
