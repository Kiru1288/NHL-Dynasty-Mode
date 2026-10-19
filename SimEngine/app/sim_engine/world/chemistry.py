# app/sim_engine/world/chemistry.py
"""Team chemistry [0,1]: roster stability, identity fit, winning."""

from __future__ import annotations

import hashlib
from typing import Any, List

KEY = "_world_chemistry"
ROSTER_SIG = "_world_roster_sig"


def init_chemistry(team: Any) -> None:
    if getattr(team, KEY, None) is None:
        setattr(team, KEY, 0.52)


def get_chemistry(team: Any) -> float:
    init_chemistry(team)
    return max(0.0, min(1.0, float(getattr(team, KEY, 0.52))))


def _roster_signature(team: Any) -> str:
    ids: List[str] = []
    for p in getattr(team, "roster", None) or []:
        if getattr(p, "retired", False):
            continue
        pid = getattr(p, "player_id", None) or getattr(getattr(p, "identity", None), "name", None) or str(id(p))
        ids.append(str(pid))
    ids.sort()
    raw = "|".join(ids).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]


def roster_stability_factor(team: Any) -> float:
    sig = _roster_signature(team)
    prev = getattr(team, ROSTER_SIG, None)
    setattr(team, ROSTER_SIG, sig)
    if prev is None or prev == sig:
        return 1.0
    return 0.92


CHURN_KEY = "_world_trade_churn"
ROSTER_IDS = "_world_roster_ids"


def _trade_newcomers(team: Any) -> int:
    """Players acquired by trade who weren't on the roster at the last check."""
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    ids = {
        str(getattr(p, "player_id", None) or getattr(p, "id", None) or id(p))
        for p in roster
    }
    prev = getattr(team, ROSTER_IDS, None)
    setattr(team, ROSTER_IDS, frozenset(ids))
    if not prev:
        return 0
    return sum(
        1 for p in roster
        if str(getattr(p, "player_id", None) or getattr(p, "id", None) or id(p)) not in prev
        and bool(getattr(p, "acquired_via_trade", False))
    )


def trade_churn(team: Any) -> float:
    return max(0.0, float(getattr(team, CHURN_KEY, 0.0) or 0.0))


def identity_alignment(team: Any) -> float:
    """Light boost when archetype matches contender/rebuild posture (proxy from attributes)."""
    arch = str(getattr(team, "archetype", "") or getattr(team, "identity_tag", "") or "").lower()
    competitive = float(getattr(getattr(team, "state", None), "competitive_score", 0.5) or 0.5)
    if "rebuild" in arch and competitive < 0.45:
        return 1.04
    if "win" in arch and competitive > 0.55:
        return 1.03
    return 1.0


def update_after_game(team: Any, won: bool, blowout: bool, rng: Any) -> None:
    init_chemistry(team)
    c = get_chemistry(team)
    stab = roster_stability_factor(team)
    equilibrium = 0.52
    delta = 0.0
    if won:
        delta += 0.008 + (0.005 if blowout else 0.0)
    else:
        delta -= 0.008 + (0.004 if blowout else 0.0)
    delta *= stab * identity_alignment(team)
    delta *= 0.92 + 0.10 * rng.random()
    # Trade churn: every new face takes a few weeks to fit in. Fades ~6%/game.
    churn = min(6.0, trade_churn(team) * 0.94 + _trade_newcomers(team))
    setattr(team, CHURN_KEY, round(churn, 3))
    delta -= 0.0025 * min(4.0, churn)
    reversion = (equilibrium - c) * 0.018
    setattr(team, KEY, max(0.08, min(0.96, c + delta + reversion)))


def team_strength_modifier(team: Any) -> float:
    # Prefer canonical room chemistry when present; otherwise fall back to world store.
    room = getattr(team, "_chemistry_cache", None)
    if isinstance(room, dict) and room.get("overall") is not None:
        c = max(0.0, min(1.0, float(room.get("overall", 50.0)) / 100.0))
    else:
        c = get_chemistry(team)
    return 1.0 + 0.04 * (c - 0.5) - 0.003 * min(4.0, trade_churn(team))


def chemistry_chaos_dampen(team: Any, chaos_noise: float) -> float:
    """Higher chemistry slightly reduces effective random variance."""
    c = get_chemistry(team)
    damp = 0.86 + 0.14 * c
    return chaos_noise * damp
