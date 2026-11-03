"""Team development personality: how hard a club pushes its prospects.

PATIENT    — lets prospects marinate in junior/AHL; only elite talent skips steps.
BALANCED   — NHL at 19-20 if clearly ready; 18-year-olds only when elite.
AGGRESSIVE — gives kids NHL jobs early (rebuilds, chaos GMs).

Deterministic per team: a stable hash of the team id sets a base lean, the club's
competitive window (gm_window / window / status) tilts it. Contenders have no
spots and protect the lineup (patient with kids); rebuilds hand out ice time.
Used by the CPU promotion pass (engine._run_prospect_promotion) and in-season
CPU call-ups (backend/services/cpu_roster_moves.py).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

PATIENT = "Patient"
BALANCED = "Balanced"
AGGRESSIVE = "Aggressive"

# Minimum age to promote a prospect who is NOT elite, and the potential (0-1
# draft-value scale) that lets an elite prospect jump the queue.
_RULES: Dict[str, Dict[str, float]] = {
    PATIENT: {"min_age": 20, "elite": 0.86, "score_mult": 0.82, "callup_margin": 1.5, "young_penalty": 2.0},
    BALANCED: {"min_age": 19, "elite": 0.81, "score_mult": 1.0, "callup_margin": 0.5, "young_penalty": 0.8},
    AGGRESSIVE: {"min_age": 18, "elite": 0.74, "score_mult": 1.12, "callup_margin": -0.5, "young_penalty": 0.0},
}

_BLURB = {
    PATIENT: "Lets prospects marinate in junior and the AHL; only elite talent skips a step.",
    BALANCED: "Promotes when a prospect is clearly ready; teenagers only if elite.",
    AGGRESSIVE: "Hands young players NHL jobs early and lives with the growing pains.",
}


def _tid(team: Any) -> str:
    return str(getattr(team, "team_id", None) or getattr(team, "id", "") or getattr(team, "name", "") or "")


def team_window(team: Any) -> str:
    for key in ("gm_window", "window"):
        w = str(getattr(team, key, "") or "").lower()
        if w in ("rebuild", "contender", "declining", "emerging"):
            return w
    st = str(getattr(team, "status", "") or getattr(team, "archetype", "") or "").lower()
    if "rebuild" in st or "tank" in st:
        return "rebuild"
    if "contend" in st or "win_now" in st:
        return "contender"
    return "emerging"


def team_dev_personality(team: Any) -> str:
    override = str(getattr(team, "dev_personality_override", "") or "")
    if override in _RULES:
        return override
    h = int(hashlib.md5(f"devp|{_tid(team)}".encode("utf-8")).hexdigest()[:8], 16)
    lean = (h % 1000) / 1000.0  # 0 = patient end, 1 = aggressive end
    window = team_window(team)
    arch = str(getattr(team, "archetype", "") or "").lower()
    if window == "contender":
        lean -= 0.22
    elif window == "rebuild":
        lean += 0.24
    elif window == "declining":
        lean += 0.06
    if "draft_and_develop" in arch or "development" in arch:
        lean -= 0.18
    if "chaos" in arch:
        lean += 0.15
    if lean < 0.36:
        return PATIENT
    if lean > 0.78:
        return AGGRESSIVE
    return BALANCED


def rules_for(team: Any) -> Dict[str, float]:
    return _RULES[team_dev_personality(team)]


def promotion_allowed(team: Any, age: int, potential: float, years_left: int = 0) -> bool:
    """Would this club move the prospect up to the NHL now?"""
    r = rules_for(team)
    try:
        age_i = int(age)
        pot = float(potential)
    except (TypeError, ValueError):
        return True
    if age_i >= 22:
        return True
    if pot >= r["elite"]:
        return True
    if age_i < int(r["min_age"]):
        return False
    # Patient clubs also wait out the development clock on non-elite kids.
    if team_dev_personality(team) == PATIENT and int(years_left or 0) > 1 and age_i < 21:
        return False
    return True


def dev_personality_payload(team: Any) -> Dict[str, Any]:
    label = team_dev_personality(team)
    return {"label": label, "window": team_window(team), "blurb": _BLURB[label]}
