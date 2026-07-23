"""
Team needs evaluation.

Outputs priorities in [0, 1] for key needs buckets.
Injury-aware: boosts positional need when starters are sidelined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _pos(player: Any) -> str:
    ident = getattr(player, "identity", None)
    p = getattr(ident, "position", None) if ident else getattr(player, "position", "")
    s = str(getattr(p, "value", p)).upper()
    if s in ("LW", "RW", "W", "F"):
        return "W"
    if s in ("C",):
        return "C"
    if s in ("D", "LD", "RD"):
        return "D"
    if s in ("G",):
        return "G"
    return s


def _ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            v = float(fn())
        except Exception:
            return 0.0
    else:
        v = _safe_float(getattr(player, "ovr", None), 0.0)
    return v * 99.0 if v <= 1.5 else v


def _bucket_priority(target: float, actual: float) -> float:
    gap = target - actual
    return _clamp(gap / 0.20, 0.0, 1.0)


def _injury_games_out(player: Any) -> int:
    for key in (
        "injury_games_remaining",
        "games_out",
        "games_remaining",
        "injury_days_remaining",
    ):
        val = getattr(player, key, None)
        if val is not None:
            try:
                g = int(val)
                if g > 0:
                    return g
            except (TypeError, ValueError):
                continue
    health = getattr(player, "health", None)
    if health is not None:
        for key in ("injury_games_remaining", "games_out", "games_remaining"):
            val = getattr(health, key, None)
            if val is not None:
                try:
                    g = int(val)
                    if g > 0:
                        return g
                except (TypeError, ValueError):
                    continue
    return 0


def _injury_status(player: Any) -> str:
    for key in ("injury_status", "health_status", "status"):
        val = str(getattr(player, key, "") or "").strip().upper()
        if val:
            return val
    health = getattr(player, "health", None)
    if health is not None:
        for key in ("injury_status", "status"):
            val = str(getattr(health, key, "") or "").strip().upper()
            if val:
                return val
    return ""


def is_player_injured(player: Any) -> bool:
    if bool(getattr(player, "is_injured", False) or getattr(player, "injured", False)):
        return True
    if _injury_games_out(player) > 0:
        return True
    st = _injury_status(player)
    if st in ("INJURED", "OUT", "IR", "LTIR", "DAY_TO_DAY", "DTD"):
        return True
    if st in ("INJURED", "OUT"):
        return True
    return False


def _injury_need_boost(roster: List[Any]) -> Dict[str, float]:
    """Extra need [0,1] from current injuries on the roster."""
    boost = {
        "top_line_forward": 0.0,
        "top_4_defense": 0.0,
        "goalie": 0.0,
        "depth_forward": 0.0,
    }
    fwds = sorted([p for p in roster if _pos(p) in ("C", "W")], key=_ovr, reverse=True)
    defs = sorted([p for p in roster if _pos(p) == "D"], key=_ovr, reverse=True)
    gs = sorted([p for p in roster if _pos(p) == "G"], key=_ovr, reverse=True)

    injured_fwds = [p for p in fwds[:6] if is_player_injured(p)]
    if injured_fwds:
        sev = min(1.0, sum(min(1.0, _injury_games_out(p) / 20.0) for p in injured_fwds) / max(1, len(injured_fwds)))
        boost["top_line_forward"] = _clamp(0.18 + 0.42 * sev)
        if len(injured_fwds) >= 2:
            boost["depth_forward"] = _clamp(0.12 + 0.28 * sev)

    injured_defs = [p for p in defs[:4] if is_player_injured(p)]
    if injured_defs:
        sev = min(1.0, sum(min(1.0, _injury_games_out(p) / 25.0) for p in injured_defs) / max(1, len(injured_defs)))
        boost["top_4_defense"] = _clamp(0.15 + 0.40 * sev)

    injured_gs = [p for p in gs[:2] if is_player_injured(p)]
    healthy_gs = [p for p in gs if not is_player_injured(p)]
    if injured_gs:
        sev = min(1.0, max(min(1.0, _injury_games_out(p) / 15.0) for p in injured_gs))
        boost["goalie"] = _clamp(0.28 + 0.55 * sev)
        if len(healthy_gs) == 0:
            boost["goalie"] = _clamp(boost["goalie"] + 0.25)
    elif len(gs) >= 1 and len(healthy_gs) == 0:
        boost["goalie"] = 0.85

    return boost


@dataclass
class TeamNeeds:
    """
    Deterministic needs evaluation from current roster (+ injury context).
    """

    target_top_line_fwd: float = 0.74
    target_top4_def: float = 0.72
    target_goalie: float = 0.74
    target_depth_fwd: float = 0.62

    def evaluate(self, team: Any, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
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

        avg_team = sum((_ovr(p) for p in roster), 0.0) / max(1, len(roster))
        if avg_team < 0.62:
            needs["top_line_forward"] = _clamp(needs["top_line_forward"] + 0.15)
            needs["top_4_defense"] = _clamp(needs["top_4_defense"] + 0.12)
            needs["goalie"] = _clamp(needs["goalie"] + 0.08)

        injury_boost = _injury_need_boost(roster)
        for key, extra in injury_boost.items():
            if extra > 0:
                needs[key] = _clamp(max(needs.get(key, 0.0), extra))

        ctx = context or {}
        deadline_phase = _safe_float(ctx.get("deadline_phase"), 0.0)
        window = str(getattr(team, "gm_window", getattr(team, "window", "")) or "").lower()
        if deadline_phase > 0.35 and window == "contender":
            needs["top_line_forward"] = _clamp(needs["top_line_forward"] + 0.06 * deadline_phase)
            needs["top_4_defense"] = _clamp(needs["top_4_defense"] + 0.05 * deadline_phase)
            needs["goalie"] = _clamp(needs["goalie"] + 0.04 * deadline_phase)

        try:
            team.needs = needs
        except Exception:
            pass
        return needs


_DEFAULT = TeamNeeds()


def evaluate_team_needs(team: Any, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    return _DEFAULT.evaluate(team, context=context)
