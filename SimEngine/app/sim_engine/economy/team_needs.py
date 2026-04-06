"""
Team needs evaluation.

Outputs priorities in [0, 1] for key needs buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


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
    if s in ("C",):
        return "C"
    if s in ("D",):
        return "D"
    if s in ("G",):
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


def _bucket_priority(target: float, actual: float) -> float:
    # if actual << target => high need
    gap = target - actual
    # normalize gap: 0.00 -> 0, 0.20 -> 1 (rough)
    return _clamp(gap / 0.20, 0.0, 1.0)


@dataclass
class TeamNeeds:
    """
    Deterministic needs evaluation from current roster.
    """

    # Targets by role (OVR in [0,1] space)
    target_top_line_fwd: float = 0.74
    target_top4_def: float = 0.72
    target_goalie: float = 0.74
    target_depth_fwd: float = 0.62

    def evaluate(self, team: Any) -> Dict[str, float]:
        roster: List[Any] = list(getattr(team, "roster", None) or [])
        fwds = [p for p in roster if _pos(p) in ("C", "W")]
        defs = [p for p in roster if _pos(p) == "D"]
        gs = [p for p in roster if _pos(p) == "G"]

        fwds_sorted = sorted(fwds, key=_ovr, reverse=True)
        defs_sorted = sorted(defs, key=_ovr, reverse=True)
        gs_sorted = sorted(gs, key=_ovr, reverse=True)

        top_fwd_avg = sum((_ovr(p) for p in fwds_sorted[:3]), 0.0) / max(1, min(3, len(fwds_sorted)))
        top4_def_avg = sum((_ovr(p) for p in defs_sorted[:4]), 0.0) / max(1, min(4, len(defs_sorted)))
        goalie_ovr = _ovr(gs_sorted[0]) if gs_sorted else 0.0
        depth_fwd_avg = sum((_ovr(p) for p in fwds_sorted[6:12]), 0.0) / max(1, min(6, max(0, len(fwds_sorted) - 6)))

        needs = {
            "top_line_forward": _bucket_priority(self.target_top_line_fwd, top_fwd_avg),
            "top_4_defense": _bucket_priority(self.target_top4_def, top4_def_avg),
            "goalie": _bucket_priority(self.target_goalie, goalie_ovr),
            "depth_forward": _bucket_priority(self.target_depth_fwd, depth_fwd_avg),
        }

        # rebuilding heuristic: if overall average is poor, increase core needs
        avg_team = sum((_ovr(p) for p in roster), 0.0) / max(1, len(roster))
        if avg_team < 0.62:
            needs["top_line_forward"] = _clamp(needs["top_line_forward"] + 0.15)
            needs["top_4_defense"] = _clamp(needs["top_4_defense"] + 0.12)
            needs["goalie"] = _clamp(needs["goalie"] + 0.08)

        # persist onto team for other systems
        try:
            team.needs = needs
        except Exception:
            pass
        return needs


_DEFAULT = TeamNeeds()


def evaluate_team_needs(team: Any) -> Dict[str, float]:
    return _DEFAULT.evaluate(team)

