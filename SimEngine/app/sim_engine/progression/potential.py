# app/sim_engine/progression/potential.py
"""
Dynamic potential: breakout, stagnate, bust. Small adjustments only (+1, +2, -1, -2).
"""

from typing import Any


def _age(player: Any) -> int:
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        return int(identity.age)
    return int(getattr(player, "age", 26))


def _ovr(player: Any) -> float:
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            return float(ovr_fn())
        except Exception:
            pass
    return getattr(player, "ovr", 0.5)


def _morale(player: Any) -> float:
    psych = getattr(player, "psych", None)
    if psych is not None and hasattr(psych, "morale"):
        return float(psych.morale)
    return float(getattr(player, "morale", 0.5))


def _potential(player: Any) -> float:
    p = getattr(player, "potential", None)
    if p is not None:
        return float(p)
    return _ovr(player) * 1.05


def update_player_potential(player: Any, rng: Any) -> None:
    """
    Dynamic potential: breakout (young + high morale), stagnate, or bust (low morale + low performance).
    Only small changes: +1, +2, -1, -2 in 0-99 scale (or equivalent in 0-1 scale ~0.01, 0.02).
    """
    age = _age(player)
    ovr = _ovr(player)
    morale = _morale(player)
    potential = _potential(player)

    # Normalize potential to 0-1 if it's in 0-99 scale
    if potential > 2.0:
        potential /= 99.0
    if ovr > 2.0:
        ovr_norm = ovr / 99.0
    else:
        ovr_norm = ovr

    delta = 0.0  # in 0-1 scale: 0.01, 0.02, -0.01, -0.02
    arch = str(getattr(player, "_dev_archetype", "") or "").upper()
    bp = float(getattr(player, "_bust_pressure", 0.08) or 0.08)
    sm = float(getattr(player, "_steal_momentum", 0.06) or 0.06)
    nar_g = float(getattr(player, "_narrative_prog_growth_mult", 1.0) or 1.0)
    nar_d = float(getattr(player, "_narrative_decline_p_mult", 1.0) or 1.0)
    p_break = 0.12
    p_bust = 0.10
    if arch == "HIGH_VARIANCE":
        p_break += 0.035
        p_bust += 0.028
    elif arch == "SAFE_LOW_CEILING":
        p_break -= 0.022
        p_bust += 0.012
    elif arch == "LATE_BLOOMER" and 22 <= age <= 26:
        p_break += 0.04
    elif arch == "ELITE_CEILING_VOLATILE":
        p_break += 0.02
        p_bust += 0.035
    p_break *= max(0.72, min(1.28, nar_g))
    p_bust *= max(0.75, min(1.32, nar_d))
    if sm >= 0.5:
        p_break += 0.04 * (sm - 0.5)
    if bp >= 0.45:
        p_bust += 0.05 * (bp - 0.45)
    # Breakout: young, high morale
    if age < 24 and morale >= 0.6 and potential < 0.90 and rng.random() < min(0.22, p_break):
        delta = rng.choice([0.01, 0.02])
    # Early bust: prospect, low morale + underperforming
    elif age < 23 and morale < 0.4 and ovr_norm < 0.70 and rng.random() < min(0.22, p_bust):
        delta = rng.choice([-0.01, -0.02])
    elif age < 24 and bp > 0.5 and rng.random() < 0.06 + 0.14 * (bp - 0.5):
        delta = rng.choice([-0.01, -0.02])
    # Stagnate: mid-age, low morale
    elif 23 <= age <= 27 and morale < 0.45 and rng.random() < 0.06:
        delta = -0.01

    if delta == 0.0:
        return

    new_potential = potential + delta
    new_potential = max(0.20, min(0.98, new_potential))
    setattr(player, "potential", new_potential)
