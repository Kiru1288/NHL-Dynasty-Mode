# app/sim_engine/progression/aging_curves.py
"""
Age-based attribute scaling. Position-specific peak and decline phases.
Returns a modifier (positive = growth, negative = decline) for use by development/regression.
"""

from typing import Any

# Position enum check (avoid hard import)
def _is_goalie(player: Any) -> bool:
    pos = getattr(player, "position", None)
    if pos is None:
        return False
    return getattr(pos, "value", str(pos)) == "G"

def _is_defense(player: Any) -> bool:
    pos = getattr(player, "position", None)
    if pos is None:
        return False
    v = getattr(pos, "value", str(pos))
    return v == "D"

def _get_peak_and_decline(player: Any) -> tuple:
    """Return (peak_age, decline_start_age) by position."""
    if _is_goalie(player):
        return 29, 34
    if _is_defense(player):
        return 27, 32
    return 26, 31  # forwards


def get_age_modifier(player: Any, rng: Any) -> float:
    """
    Realistic hockey aging curve. Returns a modifier:
    - Positive (e.g. 0.08) = growth phase
    - Negative (e.g. -0.04) = decline phase
    All randomness must use rng (e.g. rng.random(), rng.uniform()).
    """
    age = int(getattr(player, "age", 27))
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        age = int(identity.age)
    peak_age, decline_start = _get_peak_and_decline(player)

    # Age phases (deterministic curve; optional tiny variance via rng if desired)
    if age <= 21:
        # 18-21: rapid development
        return 0.06 + rng.uniform(0.0, 0.04)
    if age <= 24:
        # 22-24: strong development
        return 0.04 + rng.uniform(0.0, 0.03)
    if age <= 27:
        # 25-27: peak growth
        return 0.02 + rng.uniform(-0.01, 0.02)
    if age <= 30:
        # 28-30: plateau
        return rng.uniform(-0.01, 0.01)
    if age <= 33:
        # 31-33: mild regression
        return -0.015 + rng.uniform(-0.01, 0.005)
    if age <= 36:
        # 34-36: moderate regression
        return -0.03 + rng.uniform(-0.015, 0.0)
    # 37+: severe regression
    return -0.05 + rng.uniform(-0.02, 0.0)
