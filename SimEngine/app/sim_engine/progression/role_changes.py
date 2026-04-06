# app/sim_engine/progression/role_changes.py
"""
Lineup role transitions based on ovr, age, team strength, performance.
Roles: elite, top_line, top_4, middle_6, bottom_6, depth, prospect.
"""

from typing import Any

ROLES_FWD = ["elite", "top_line", "middle_6", "bottom_6", "depth", "prospect"]
ROLES_D = ["elite", "top_4", "middle_6", "bottom_6", "depth", "prospect"]
ROLES_G = ["elite", "starter", "backup", "depth", "prospect"]


def _ovr(player: Any) -> float:
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            return float(ovr_fn())
        except Exception:
            pass
    return getattr(player, "ovr", 0.5)


def _age(player: Any) -> int:
    identity = getattr(player, "identity", None)
    if identity is not None and hasattr(identity, "age"):
        return int(identity.age)
    return int(getattr(player, "age", 26))


def _is_goalie(player: Any) -> bool:
    pos = getattr(player, "position", None)
    if pos is None:
        return False
    return getattr(pos, "value", str(pos)) == "G"


def _is_defense(player: Any) -> bool:
    pos = getattr(player, "position", None)
    if pos is None:
        return False
    return getattr(pos, "value", str(pos)) == "D"


def update_player_role(player: Any) -> None:
    """
    Set player role from ovr, age, and (optional) team strength/performance.
    Uses ovr in 0-1 scale. Mutates player.role or equivalent.
    """
    ovr = _ovr(player)
    if ovr > 1.0:
        ovr = ovr / 99.0
    age = _age(player)

    if _is_goalie(player):
        if ovr >= 0.88:
            role = "elite"
        elif ovr >= 0.78:
            role = "starter"
        elif ovr >= 0.65:
            role = "backup"
        elif ovr >= 0.50:
            role = "depth"
        else:
            role = "prospect"
    elif _is_defense(player):
        if ovr >= 0.88:
            role = "elite"
        elif ovr >= 0.78:
            role = "top_4"
        elif ovr >= 0.65:
            role = "middle_6"
        elif ovr >= 0.55:
            role = "bottom_6"
        elif ovr >= 0.45:
            role = "depth"
        else:
            role = "prospect"
    else:
        if ovr >= 0.88:
            role = "elite"
        elif ovr >= 0.78:
            role = "top_line"
        elif ovr >= 0.65:
            role = "middle_6"
        elif ovr >= 0.55:
            role = "bottom_6"
        elif ovr >= 0.45:
            role = "depth"
        else:
            role = "prospect"

    if age <= 22 and ovr < 0.70:
        role = "prospect"

    setattr(player, "role", role)
