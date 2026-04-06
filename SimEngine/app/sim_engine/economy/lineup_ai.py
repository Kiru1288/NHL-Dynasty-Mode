"""
Lineup AI.

Selects best lineup using:
- OVR
- position fit
- morale (bonus)
- fatigue (penalty)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _pos(player: Any) -> str:
    p = getattr(player, "position", "")
    s = str(getattr(p, "value", p)).upper()
    if s in ("LW", "RW", "W"):
        return "W"
    if s == "C":
        return "C"
    if s == "D":
        return "D"
    if s == "G":
        return "G"
    return s


def _ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            return 0.0
    return _safe_float(getattr(player, "ovr", None), 0.0)


def _morale(player: Any) -> float:
    return _safe_float(getattr(getattr(player, "psych", None), "morale", None), 0.5)


def _fatigue(player: Any) -> float:
    return _safe_float(getattr(getattr(player, "health", None), "fatigue", None), 0.0)


def _score_player(player: Any, desired_pos: Optional[str] = None) -> float:
    base = _ovr(player)
    m = _morale(player)
    f = _fatigue(player)
    pos = _pos(player)
    pos_bonus = 0.0
    if desired_pos is not None and desired_pos:
        if desired_pos == pos:
            pos_bonus = 0.02
        elif desired_pos in ("C", "W") and pos in ("C", "W"):
            pos_bonus = 0.01
        else:
            pos_bonus = -0.03
    # morale is small; fatigue penalty is bigger
    return base + (m - 0.5) * 0.04 - f * 0.06 + pos_bonus


@dataclass
class LineupAI:
    """
    Produces a lineup dict. This does not mutate roster.
    Engine can store it on `team.current_lineup`.
    """

    def generate(self, team: Any) -> Dict[str, Any]:
        roster: List[Any] = list(getattr(team, "roster", None) or [])
        if not roster:
            return {"lines": [], "pairs": [], "starter_goalie": None}

        fwds = [p for p in roster if _pos(p) in ("C", "W")]
        defs = [p for p in roster if _pos(p) == "D"]
        gs = [p for p in roster if _pos(p) == "G"]

        # choose top 12 forwards and top 6 defense
        fwds_sorted = sorted(fwds, key=lambda p: _score_player(p), reverse=True)[:12]
        defs_sorted = sorted(defs, key=lambda p: _score_player(p, "D"), reverse=True)[:6]

        # lines: 4 lines of 3
        lines: List[List[str]] = []
        for i in range(0, min(12, len(fwds_sorted)), 3):
            trio = fwds_sorted[i : i + 3]
            if len(trio) == 3:
                lines.append([getattr(p, "name", getattr(p, "id", "Player")) for p in trio])

        # pairs: 3 pairs of 2
        pairs: List[List[str]] = []
        for i in range(0, min(6, len(defs_sorted)), 2):
            pair = defs_sorted[i : i + 2]
            if len(pair) == 2:
                pairs.append([getattr(p, "name", getattr(p, "id", "Player")) for p in pair])

        starter_goalie = None
        if gs:
            starter = max(gs, key=lambda p: _score_player(p, "G"))
            starter_goalie = getattr(starter, "name", getattr(starter, "id", "Goalie"))

        out = {"lines": lines, "pairs": pairs, "starter_goalie": starter_goalie}
        try:
            team.current_lineup = out
        except Exception:
            pass
        return out


_DEFAULT = LineupAI()


def generate_lineup(team: Any) -> Dict[str, Any]:
    return _DEFAULT.generate(team)

