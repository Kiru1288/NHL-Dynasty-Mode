"""
On-ice rules voted in by the Board of Governors.

The Board writes its combined modifiers onto ``league.governance_modifiers``
(backend/services/league_governance.py). This module turns the on-ice keys into the
numbers the game sim and the standings actually use, so a passed rule changes how
games are played and how points are awarded — not just fan mood.

Keys (all optional; missing = current NHL rules):
  scoring_pace   pct   goal rate (event + fast sim)
  penalty_rate   pct   penalties called per game (more power plays)
  pp_efficiency  pct   power-play scoring
  home_ice       pct   size of the home-ice edge
  ot_minutes     abs   regular-season overtime length (5 = NHL)
  ties_allowed   abs   1 = no shootout: games tied after OT stand as ties
  pts_reg_win    abs   points for a regulation win (2 = NHL, 3 = 3-2-1-0)
  pts_ot_win     abs   points for an OT / shootout win
  pts_otl        abs   points for an OT / shootout loss
  pts_tie        abs   points each for a tie (only with ties_allowed)
"""

from __future__ import annotations

from typing import Any, Dict

NHL_OT_MINUTES = 5.0
#: Share of NHL overtime games that reach a shootout with 5 minutes of 3-on-3.
NHL_SHOOTOUT_SHARE = 0.42


def _f(v: Any, d: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(d)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _mods(league: Any) -> Dict[str, Any]:
    mods = getattr(league, "governance_modifiers", None) if league is not None else None
    return mods if isinstance(mods, dict) else {}


def rules_for(league: Any) -> Dict[str, Any]:
    m = _mods(league)
    ties = _f(m.get("ties_allowed", 0), 0) >= 0.5
    reg_win = int(round(_clamp(_f(m.get("pts_reg_win", 2), 2), 1, 4)))
    ot_win = int(round(_clamp(_f(m.get("pts_ot_win", 2), 2), 1, 4)))
    otl = int(round(_clamp(_f(m.get("pts_otl", 1), 1), 0, 2)))
    return {
        "scoring_mult": _clamp(1.0 + _f(m.get("scoring_pace", 0), 0) / 100.0, 0.80, 1.30),
        "penalty_mult": _clamp(1.0 + _f(m.get("penalty_rate", 0), 0) / 100.0, 0.60, 1.60),
        "pp_mult": _clamp(1.0 + _f(m.get("pp_efficiency", 0), 0) / 100.0, 0.70, 1.45),
        "home_ice_mult": _clamp(1.0 + _f(m.get("home_ice", 0), 0) / 100.0, 0.30, 2.0),
        "ot_minutes": _clamp(_f(m.get("ot_minutes", NHL_OT_MINUTES), NHL_OT_MINUTES), 3.0, 20.0),
        "ties_allowed": ties,
        "pts_reg_win": reg_win,
        "pts_ot_win": min(ot_win, reg_win),
        "pts_otl": otl,
        "pts_tie": int(round(_clamp(_f(m.get("pts_tie", 1), 1), 0, 2))),
    }


def shootout_share(rules: Dict[str, Any]) -> float:
    """Chance an overtime game is still tied when OT ends (longer OT → fewer)."""
    mins = _f(rules.get("ot_minutes"), NHL_OT_MINUTES)
    return _clamp(NHL_SHOOTOUT_SHARE * (NHL_OT_MINUTES / max(1.0, mins)) ** 0.85, 0.06, 0.60)


def ot_attempt_scale(rules: Dict[str, Any]) -> float:
    """Scales overtime shot attempts in the event sim with OT length."""
    return _clamp(_f(rules.get("ot_minutes"), NHL_OT_MINUTES) / NHL_OT_MINUTES, 0.6, 4.0)


def light_goal_rate_mult(rules: Dict[str, Any]) -> float:
    """Fast-sim goal rate: scoring pace plus the knock-on of more/less power play time."""
    pen = float(rules.get("penalty_mult", 1.0))
    pp = float(rules.get("pp_mult", 1.0))
    return float(rules.get("scoring_mult", 1.0)) * (1.0 + (pen - 1.0) * 0.10) * (1.0 + (pp - 1.0) * 0.08)


def points_label(rules: Dict[str, Any]) -> str:
    base = f"{rules['pts_reg_win']}-{rules['pts_ot_win']}-{rules['pts_otl']}-0"
    return base + (" · ties stand" if rules.get("ties_allowed") else "")


#: Fan confidence (0-100) at which the home crowd adds nothing extra.
FAN_NEUTRAL = 60.0


def fan_home_ice_share(fan_confidence: Any) -> float:
    """Extra share of the normal home-ice edge from the crowd: +25% for a sold-out,
    electric building (fans ~90), down to -25% when the fan base has turned (~30)."""
    return _clamp((_f(fan_confidence, FAN_NEUTRAL) - FAN_NEUTRAL) / 100.0 * 0.8, -0.25, 0.25)


def team_fan_confidence(team: Any) -> float:
    """The club's current fan confidence (mirrored weekly from the franchise fan profile)."""
    return _f(getattr(team, "_fan_confidence", None), FAN_NEUTRAL)
