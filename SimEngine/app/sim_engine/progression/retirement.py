# app/sim_engine/progression/retirement.py
"""
Career-ending logic: age thresholds (35/37/39/41+) and modifiers (OVR, injuries, cup wins).
"""

from typing import Any


def _age(player: Any) -> int:
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        return int(identity.age)
    return int(getattr(player, "age", 27))


def _ovr(player: Any) -> float:
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            return float(ovr_fn())
        except Exception:
            pass
    return getattr(player, "ovr", 0.5)


def _injury_history(player: Any) -> list:
    health = getattr(player, "health", None)
    if health is not None and hasattr(health, "injury_history"):
        h = getattr(health, "injury_history", [])
        return h if isinstance(h, list) else []
    return getattr(player, "injury_history", []) or []


def _cup_wins(player: Any) -> int:
    return int(getattr(player, "cup_wins", 0))


def should_player_retire(player: Any, rng: Any) -> bool:
    """
    Retirement by age: 35 small chance, 37 moderate, 39 high, 41+ near guaranteed.
    Modifiers: low OVR and injuries increase chance; cup wins reduce it.
    Returns True if the player should retire this year. Caller sets player.retired and removes from league.
    """
    age = _age(player)
    if age < 35:
        return False

    # Base chance by age
    if age >= 41:
        base = 0.85
    elif age >= 39:
        base = 0.55
    elif age >= 37:
        base = 0.25
    else:
        base = 0.08  # 35-36

    ovr = _ovr(player)
    if ovr > 1.0:
        ovr = ovr / 99.0
    # Low OVR increases retirement
    if ovr < 0.50:
        base += 0.15
    elif ovr < 0.60:
        base += 0.08
    elif ovr < 0.70:
        base += 0.03

    injuries = _injury_history(player)
    base += min(0.12, 0.02 * len(injuries))

    cup_wins = _cup_wins(player)
    base -= min(0.15, 0.05 * cup_wins)

    chance = max(0.02, min(0.98, base))
    return rng.random() < chance
