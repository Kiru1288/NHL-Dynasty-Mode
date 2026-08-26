"""Conduct / storyline OVR impact — eligibility + temporary readiness (not permanent talent wipes)."""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from app.sim_engine.entities.player import (  # noqa: WPS433
        DEFENSE_ATTRS,
        GOALIE_ATTRS,
        IQ_ATTRS,
        OFFENSE_ATTRS,
        PERSONALITY_ATTRS,
        PHYSICAL_ATTRS,
        SKATING_ATTRS,
    )
except Exception:  # pragma: no cover
    OFFENSE_ATTRS = ["off_shooting", "off_deking"]
    DEFENSE_ATTRS = ["def_stick_checking"]
    SKATING_ATTRS = ["skg_speed"]
    PHYSICAL_ATTRS = ["phy_strength"]
    IQ_ATTRS = ["iqm_awareness"]
    PERSONALITY_ATTRS = ["per_leadership"]
    GOALIE_ATTRS = ["g_reflexes"]

GAMES_KEY = "_world_conduct_games_remaining"
STORYLINE_ID_KEY = "_world_conduct_storyline_id"
SEVERITY_KEY = "_world_conduct_severity"
STATUS_KEY = "_world_conduct_status"
RESOLVED_KEY = "_world_conduct_resolved"
NUDGE_ID_KEY = "_storyline_ovr_nudge_id"
MODIFIERS_KEY = "_franchise_ovr_modifiers"

_POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "C": {"offense": 0.30, "defense": 0.20, "skating": 0.20, "physical": 0.10, "mental": 0.20},
    "LW": {"offense": 0.36, "defense": 0.14, "skating": 0.22, "physical": 0.12, "mental": 0.16},
    "RW": {"offense": 0.36, "defense": 0.14, "skating": 0.22, "physical": 0.12, "mental": 0.16},
    "F": {"offense": 0.34, "defense": 0.16, "skating": 0.22, "physical": 0.12, "mental": 0.16},
    "D": {"offense": 0.18, "defense": 0.36, "skating": 0.18, "physical": 0.14, "mental": 0.14},
    "LD": {"offense": 0.18, "defense": 0.36, "skating": 0.18, "physical": 0.14, "mental": 0.14},
    "RD": {"offense": 0.18, "defense": 0.36, "skating": 0.18, "physical": 0.14, "mental": 0.14},
    "G": {"technical": 0.48, "athletic": 0.20, "mental": 0.22, "puck": 0.10},
}


def _clamp_int(x: float, lo: int = 40, hi: int = 99) -> int:
    return int(max(lo, min(hi, round(x))))


def _player_pos(player: Any) -> str:
    ident = getattr(player, "identity", None)
    pos = getattr(ident, "position", None) if ident is not None else None
    raw = str(getattr(pos, "value", pos) if pos is not None else getattr(player, "position", "") or "F")
    p = raw.strip().upper()
    if p in ("W", "WING"):
        return "F"
    if p in ("LD", "RD", "D"):
        return p if p in ("LD", "RD") else "D"
    if p in ("C", "LW", "RW", "F", "G"):
        return p
    return "F"


def _rating_avg_for_keys(player: Any, keys: List[str]) -> Optional[float]:
    r = getattr(player, "ratings", None) or {}
    vals: List[float] = []
    for k in keys:
        try:
            v = float(r.get(k, 0) or 0)
        except (TypeError, ValueError):
            continue
        if v > 0:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _morale_adjustment(player: Any) -> float:
    psych = getattr(player, "psych", None)
    try:
        morale = float(getattr(psych, "morale", 0.5) or 0.5) * 100.0
    except (TypeError, ValueError):
        morale = 50.0
    if morale >= 90:
        return 1.0
    if morale >= 75:
        return 0.5
    if morale <= 25:
        return -1.2
    if morale <= 40:
        return -0.6
    return 0.0


def _compute_true_ovr_display(player: Any) -> int:
    """Match RosterScreen trueOverall — ratings-weighted blend, not raw sim ovr()."""
    explicit = _ovr_display(player)
    pos = _player_pos(player)
    weights = _POSITION_WEIGHTS.get(pos) or _POSITION_WEIGHTS.get("F") or {}

    if pos == "G":
        summary = {
            "technical": _rating_avg_for_keys(player, GOALIE_ATTRS),
            "athletic": _rating_avg_for_keys(player, GOALIE_ATTRS),
            "mental": _rating_avg_for_keys(player, IQ_ATTRS + PERSONALITY_ATTRS),
            "puck": _rating_avg_for_keys(player, GOALIE_ATTRS),
        }
    else:
        iq = _rating_avg_for_keys(player, IQ_ATTRS)
        per = _rating_avg_for_keys(player, PERSONALITY_ATTRS)
        mental = None
        if iq is not None and per is not None:
            mental = (iq + per) / 2.0
        elif iq is not None:
            mental = iq
        elif per is not None:
            mental = per
        summary = {
            "offense": _rating_avg_for_keys(player, OFFENSE_ATTRS),
            "defense": _rating_avg_for_keys(player, DEFENSE_ATTRS),
            "skating": _rating_avg_for_keys(player, SKATING_ATTRS),
            "physical": _rating_avg_for_keys(player, PHYSICAL_ATTRS),
            "mental": mental,
        }

    weighted = 0.0
    used = 0.0
    for key, wt in weights.items():
        val = summary.get(key)
        if val is not None and val > 0:
            weighted += float(val) * float(wt)
            used += float(wt)
    rating_based = (weighted / used) if used > 0 else None

    if rating_based is not None and explicit > 0:
        base = rating_based * 0.65 + float(explicit) * 0.35
    elif rating_based is not None:
        base = rating_based
    elif explicit > 0:
        base = float(explicit)
    else:
        base = 0.0

    true = base + _morale_adjustment(player)
    if true <= 0:
        return explicit
    return _clamp_int(true)


STORYLINE_ID_KEY = "_world_conduct_storyline_id"
SEVERITY_KEY = "_world_conduct_severity"
STATUS_KEY = "_world_conduct_status"
RESOLVED_KEY = "_world_conduct_resolved"
NUDGE_ID_KEY = "_storyline_ovr_nudge_id"
MODIFIERS_KEY = "_franchise_ovr_modifiers"


def _player_ovr01(player: Any) -> float:
    try:
        fn = getattr(player, "ovr", None)
        return float(fn()) if callable(fn) else float(fn or 0.5)
    except Exception:
        return 0.5


def _ovr_display(player: Any) -> int:
    return int(round(_player_ovr01(player) * 99.0))


def _ensure_modifiers(player: Any) -> List[Dict[str, Any]]:
    mods = getattr(player, MODIFIERS_KEY, None)
    if not isinstance(mods, list):
        mods = []
        setattr(player, MODIFIERS_KEY, mods)
    return mods


def _is_trade_rumor_modifier(mod: Dict[str, Any]) -> bool:
    src = str(mod.get("source") or "").lower()
    ctype = str(mod.get("cause_type") or "").upper()
    mtype = str(mod.get("modifier_type") or "").lower()
    if mtype == "trade_rumor":
        return True
    if src == "trade_rumor":
        return True
    return ctype in ("TRADE_REJECTED", "PLAYER_REPEATEDLY_SHOPPED", "TRADE_ATTEMPTED_BY_USER")


def get_player_ovr_modifiers(player: Any) -> List[Dict[str, Any]]:
    return list(_ensure_modifiers(player))


def sum_active_modifier_amount(player: Any) -> int:
    total = 0
    for m in _ensure_modifiers(player):
        if m.get("resolved"):
            continue
        try:
            total += int(m.get("amount") or 0)
        except (TypeError, ValueError):
            pass
    return int(total)


def get_base_ovr_display(player: Any) -> int:
    """Roster-aligned true OVR (ratings blend), not raw sim engine ovr()."""
    return _compute_true_ovr_display(player)


def get_effective_ovr_display(player: Any) -> int:
    return max(40, min(99, get_base_ovr_display(player) + sum_active_modifier_amount(player)))


def apply_permanent_ovr_delta(
    player: Any,
    amount: int,
    *,
    reason: str = "",
    storyline_id: str = "",
    cause_type: str = "",
) -> Dict[str, Any]:
    """
    Permanently nudge base ratings so displayed OVR moves by ~amount points.
    This is the real overall — not a temporary display overlay.
    """
    amt = int(amount)
    if amt == 0 or player is None:
        return {}
    sid = str(storyline_id or "")
    if sid and str(getattr(player, NUDGE_ID_KEY, "") or "") == sid:
        base = get_base_ovr_display(player)
        return {
            "overall_before": base,
            "overall_after": base,
            "overall_delta": 0,
            "base_overall": base,
            "effective_overall": get_effective_ovr_display(player),
            "already_active": True,
            "permanent": True,
        }

    ovr_before = get_base_ovr_display(player)
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict) or not ratings:
        return {}

    try:
        from app.sim_engine.progression.development import (  # noqa: WPS433
            allocate_growth_to_attributes,
            apply_attribute_deltas,
        )
        from app.sim_engine.entities.player import persist_recomputed_ovr  # noqa: WPS433

        phase = "SPIKE" if amt > 0 else "REGRESSION"
        # Slight overshoot so display OVR usually lands near the intended delta.
        budget_01 = (float(amt) / 99.0) * 1.22
        deltas = allocate_growth_to_attributes(
            player,
            budget_01,
            role=str(getattr(player, "role", "") or ""),
            archetype=getattr(player, "archetype", None),
            phase=phase,
        )
        apply_attribute_deltas(player, deltas)
        persist_recomputed_ovr(player)
        ovr_after = get_base_ovr_display(player)
        # If undershot, force flat bumps on a few skill keys.
        short = int(amt) - int(ovr_after - ovr_before)
        if short != 0:
            skill_keys = [
                k
                for k in ratings.keys()
                if str(k).lower()
                not in (
                    "dev_potential",
                    "dev_ceiling",
                    "potential",
                    "overall",
                    "ovr",
                    "true_potential",
                    "true_ceiling",
                )
            ]
            if skill_keys:
                step = 1 if short > 0 else -1
                for i in range(abs(short) * 3):
                    key = skill_keys[i % len(skill_keys)]
                    cur = float(ratings.get(key, 50) or 50)
                    ratings[key] = max(20.0, min(99.0, cur + step * 1.5))
                    if i % 3 == 2:
                        persist_recomputed_ovr(player)
                        ovr_after = get_base_ovr_display(player)
                        if (ovr_after - ovr_before) * (1 if amt > 0 else -1) >= abs(amt):
                            break
                persist_recomputed_ovr(player)
                ovr_after = get_base_ovr_display(player)
    except Exception:
        ovr_after = get_base_ovr_display(player)

    if sid:
        setattr(player, NUDGE_ID_KEY, sid)
    delta = int(ovr_after - ovr_before)
    return {
        "overall_before": int(ovr_before),
        "overall_after": int(ovr_after),
        "overall_delta": delta,
        "base_overall": int(ovr_after),
        "effective_overall": get_effective_ovr_display(player),
        "impact_reason": str(reason or ""),
        "already_active": False,
        "permanent": True,
        "cause_type": str(cause_type or ""),
    }


def apply_temporary_ovr_modifier(
    player: Any,
    *,
    source: str,
    amount: int,
    reason: str,
    duration_games: int = 10,
    storyline_id: str = "",
    cause_type: str = "",
    cause_event_id: str = "",
    modifier_type: str = "",
) -> Dict[str, Any]:
    """Add a temporary OVR modifier. Does not mutate base ratings."""
    amt = int(amount)
    if amt == 0:
        return {}
    sid = str(storyline_id or "")
    mtype = str(modifier_type or "").strip().lower()
    if not mtype and str(source or "").strip().lower() == "trade_rumor":
        mtype = "trade_rumor"
    mods = _ensure_modifiers(player)
    if sid:
        for m in mods:
            if str(m.get("storyline_id") or "") == sid and not m.get("resolved"):
                return {
                    "overall_before": get_base_ovr_display(player),
                    "overall_after": get_effective_ovr_display(player),
                    "overall_delta": 0,
                    "already_active": True,
                }
    # Only one active trade-rumor OVR modifier per player.
    if mtype == "trade_rumor":
        ovr_before = get_effective_ovr_display(player)
        for m in mods:
            if m.get("resolved") or not _is_trade_rumor_modifier(m):
                continue
            prev_amt = int(m.get("amount") or 0)
            prev_gr = int(m.get("games_remaining") or 0)
            prev_dur = int(m.get("duration_games") or 0)
            # Keep the stronger (more negative) penalty and only modestly refresh duration.
            if amt < prev_amt:
                m["amount"] = amt
            m["duration_games"] = max(prev_dur, int(duration_games))
            m["games_remaining"] = max(prev_gr, min(max(prev_gr + 2, 1), int(duration_games)))
            ovr_after = get_effective_ovr_display(player)
            return {
                "overall_before": int(ovr_before),
                "overall_after": int(ovr_after),
                "overall_delta": int(ovr_after - ovr_before),
                "base_overall": get_base_ovr_display(player),
                "effective_overall": int(ovr_after),
                "impact_reason": str(reason or ""),
                "already_active": True,
            }
    ovr_before = get_effective_ovr_display(player)
    entry = {
        "id": f"mod_{uuid.uuid4().hex[:10]}",
        "source": str(source or cause_type or "STORYLINE"),
        "cause_type": str(cause_type or source or ""),
        "cause_event_id": str(cause_event_id or ""),
        "modifier_type": mtype,
        "amount": amt,
        "reason": str(reason or ""),
        "duration_games": max(0, int(duration_games)),
        "games_remaining": max(0, int(duration_games)),
        "storyline_id": sid,
        "resolved": False,
        "created_date": "",
    }
    mods.append(entry)
    ovr_after = get_effective_ovr_display(player)
    delta = int(ovr_after - ovr_before)
    if sid:
        setattr(player, NUDGE_ID_KEY, sid)
    return {
        "overall_before": int(ovr_before),
        "overall_after": int(ovr_after),
        "overall_delta": delta,
        "base_overall": get_base_ovr_display(player),
        "effective_overall": int(ovr_after),
        "impact_reason": str(reason or ""),
        "modifier_id": entry["id"],
        "already_active": False,
    }


def tick_player_ovr_modifiers(player: Any) -> None:
    """Decrement modifier duration after a team game."""
    for m in _ensure_modifiers(player):
        if m.get("resolved"):
            continue
        gr = int(m.get("games_remaining") or 0)
        if gr <= 0:
            continue
        m["games_remaining"] = gr - 1
        if m["games_remaining"] <= 0:
            m["resolved"] = True


def resolve_modifiers_for_storyline(player: Any, storyline_id: str) -> int:
    """Mark modifiers tied to a storyline resolved. Returns count cleared."""
    sid = str(storyline_id or "")
    if not sid:
        return 0
    n = 0
    for m in _ensure_modifiers(player):
        if str(m.get("storyline_id") or "") == sid and not m.get("resolved"):
            m["resolved"] = True
            m["games_remaining"] = 0
            n += 1
    return n


def clear_trade_fallout_modifiers(player: Any) -> int:
    """Remove temporary OVR penalties from failed trade talks (successful trade clears fallout)."""
    n = 0
    for m in _ensure_modifiers(player):
        if m.get("resolved"):
            continue
        ct = str(m.get("cause_type") or m.get("source") or "").upper()
        reason = str(m.get("reason") or "").lower()
        is_trade_fallout = (
            ct in ("TRADE_REJECTED", "TRADE_ATTEMPTED_BY_USER", "PLAYER_REPEATEDLY_SHOPPED", "PLAYER_TRADED")
            or "TRADE" in ct
            or "trade" in reason
            or "tradehub" in reason
        )
        if is_trade_fallout:
            m["resolved"] = True
            m["games_remaining"] = 0
            n += 1
    return n


def conduct_spec(severity: str, rng: Optional[random.Random] = None) -> Tuple[int, float]:
    """Return (team_games_out, target_ovr_drop_points). Only major tier uses games out."""
    r = rng or random.Random()
    sev = str(severity or "minor").lower()
    if sev == "major":
        games = r.randint(12, 22)
        drop_pts = r.randint(18, 28)
    else:
        games = 0
        if sev == "moderate":
            drop_pts = r.randint(5, 10)
        else:
            drop_pts = r.randint(2, 5)
    return int(games), float(drop_pts)


def get_conduct_games_remaining(player: Any) -> int:
    return max(0, int(getattr(player, GAMES_KEY, 0) or 0))


def is_under_conduct_suspension(player: Any) -> bool:
    """True when player cannot dress due to leave/suspension (or legacy games remaining)."""
    if getattr(player, "_conduct_incident_id", None):
        return not bool(getattr(player, "_conduct_eligible_to_play", True))
    return get_conduct_games_remaining(player) > 0


def apply_conduct_suspension(
    player: Any,
    *,
    severity: str,
    storyline_id: str,
    cause_type: str = "LOW_CHARACTER_CONFLICT",
    cause_event_id: str = "",
    rng: Optional[random.Random] = None,
    host: Any = None,
    team_id: str = "",
    storyline_text: str = "",
    player_fame: float = 0.5,
) -> Dict[str, Any]:
    """Open a conduct incident (eligibility + soft readiness). No permanent talent wipe."""
    if is_under_conduct_suspension(player) and getattr(player, "_conduct_incident_id", None):
        gr = get_conduct_games_remaining(player)
        eff = get_effective_ovr_display(player)
        return {
            "games_remaining": gr,
            "overall_before": eff,
            "overall_after": eff,
            "overall_delta": 0,
            "base_overall": get_base_ovr_display(player),
            "effective_overall": eff,
            "already_active": True,
            "conduct_model": "state_machine",
        }

    from app.sim_engine.franchise.conduct_incidents import create_conduct_incident

    registry_host = host
    incident = create_conduct_incident(
        registry_host,
        player=player,
        team_id=str(team_id or ""),
        storyline_text=str(storyline_text or "Off-ice conduct matter under review"),
        severity=str(severity or "major"),
        storyline_id=str(storyline_id or ""),
        cause_event_id=str(cause_event_id or ""),
        player_fame=float(player_fame),
        rng=rng,
    )
    games = int(incident.get("games_remaining") or 0)
    eff = get_effective_ovr_display(player)
    base = get_base_ovr_display(player)
    return {
        "games_remaining": games,
        "games_initial": int(incident.get("games_initial") or games),
        "overall_before": base,
        "overall_after": eff,
        "overall_delta": int(eff - base),
        "base_overall": base,
        "effective_overall": eff,
        "already_active": False,
        "conduct_severity": str(severity or "major").lower(),
        "conduct_model": "state_machine",
        "incident_id": incident.get("incident_id"),
        "eligible_to_play": bool(incident.get("eligible_to_play")),
        "information_status": incident.get("information_status"),
        "legal_status": incident.get("legal_status"),
        "league_status": incident.get("league_status"),
        "team_status": incident.get("team_status"),
        "impact_reason": (
            "Player unavailable pending investigation / leave — physical attributes unchanged"
            if not incident.get("eligible_to_play")
            else "Under investigation — eligible but organizational backlash risk if dressed"
        ),
        "allegation_note": "Reports are allegations until an official ruling.",
        "cause_type": cause_type,
    }


def tick_conduct_games_missed(player: Any) -> bool:
    """Decrement suspension when player's team plays and player is sidelined."""
    gr = get_conduct_games_remaining(player)
    if gr <= 0:
        return False
    setattr(player, GAMES_KEY, gr - 1)
    tick_player_ovr_modifiers(player)
    return True


def resolve_conduct_if_cleared(player: Any) -> Optional[Dict[str, Any]]:
    """If suspension expired, resolve conduct modifiers and return metadata."""
    if get_conduct_games_remaining(player) > 0:
        return None
    if not getattr(player, STORYLINE_ID_KEY, None):
        return None
    if getattr(player, RESOLVED_KEY, False):
        return None

    sev = str(getattr(player, SEVERITY_KEY, "minor") or "minor")
    sid = str(getattr(player, STORYLINE_ID_KEY, "") or "")
    base = get_base_ovr_display(player)
    ovr_now = get_effective_ovr_display(player)

    resolve_modifiers_for_storyline(player, sid)
    # Lingering partial penalty after major conduct
    if sev == "major":
        apply_temporary_ovr_modifier(
            player,
            source="CONDUCT_RECOVERY",
            amount=-max(2, int((base - ovr_now) * 0.15)),
            reason="Partial recovery after conduct suspension",
            duration_games=20,
            storyline_id=f"{sid}:recovery",
            cause_type="CONDUCT_RESOLVED",
        )
    elif sev == "moderate":
        apply_temporary_ovr_modifier(
            player,
            source="CONDUCT_RECOVERY",
            amount=-2,
            reason="Distraction lingers after off-ice matter",
            duration_games=12,
            storyline_id=f"{sid}:recovery",
            cause_type="CONDUCT_RESOLVED",
        )

    ovr_restored = get_effective_ovr_display(player)
    setattr(player, RESOLVED_KEY, True)
    setattr(player, STATUS_KEY, "resolved")
    setattr(player, GAMES_KEY, 0)

    return {
        "storyline_id": sid,
        "conduct_severity": sev,
        "overall_before_penalty": base,
        "overall_after_return": int(ovr_restored),
        "base_overall": base,
        "effective_overall": int(ovr_restored),
        "status": "resolved",
        "resolution_summary": (
            f"Conduct suspension lifted. Effective OVR {ovr_restored} "
            f"(base {base}; temporary modifiers may still apply)."
        ),
    }


def build_conduct_storyline_fields(meta: Dict[str, Any], *, return_estimate: str = "", return_date: str = "") -> Dict[str, Any]:
    """Extra storyline/notification fields for conduct incidents (eligibility-first)."""
    gr = int(meta.get("games_remaining") or 0)
    eligible = bool(meta.get("eligible_to_play", gr <= 0))
    ret_est = return_estimate or (f"In {gr} games" if gr > 0 and not eligible else "")
    ovr_b = meta.get("overall_before")
    ovr_a = meta.get("overall_after")
    base = meta.get("base_overall")
    eff = meta.get("effective_overall") or ovr_a
    delta = meta.get("overall_delta")
    parts = []
    if not eligible and gr > 0:
        parts.append(f"Unavailable — projected return: {ret_est or f'{gr} games'}")
    elif not eligible:
        parts.append("Unavailable pending leave / league suspension")
    else:
        parts.append("Eligible to dress — organizational backlash risk if played")
    if delta is not None and int(delta) != 0:
        parts.append(f"Temporary readiness {base} → {eff} ({int(delta):+d}; talent base unchanged)")
    reason = str(meta.get("impact_reason") or "")
    if reason:
        parts.append(reason)
    allegation = str(meta.get("allegation_note") or "Reports are allegations until an official ruling.")
    return {
        "games_remaining": gr,
        "games_initial": int(meta.get("games_initial") or gr),
        "return_estimate": ret_est,
        "return_date": return_date,
        "overall_before": ovr_b,
        "overall_after": ovr_a,
        "overall_delta": delta,
        "base_overall": base,
        "effective_overall": eff,
        "player_overall": eff or ovr_a,
        "impact_reason": reason,
        "effect_summary": " · ".join(parts) if parts else "",
        "arc_status": "active",
        "category": "legal_trouble",
        "incident_id": meta.get("incident_id"),
        "eligible_to_play": eligible,
        "information_status": meta.get("information_status"),
        "legal_status": meta.get("legal_status"),
        "league_status": meta.get("league_status"),
        "team_status": meta.get("team_status"),
        "conduct_model": meta.get("conduct_model") or "state_machine",
        "allegation_note": allegation,
        "follow_up": (
            "Player cannot dress while suspended or on leave."
            if not eligible
            else "Dressing an investigated player can hit owner/fan/media/sponsor/revenue pressure."
        ),
    }


def apply_storyline_ovr_nudge(
    player: Any,
    *,
    tier: str = "minor",
    legal_severity: str = "",
    storyline_id: str = "",
    cause_type: str = "",
    cause_event_id: str = "",
    reason: str = "",
    rng: Optional[random.Random] = None,
    amount: Optional[int] = None,
) -> Dict[str, Any]:
    """Storyline beat — temporary readiness change (no permanent talent wipe)."""
    if is_under_conduct_suspension(player):
        return {}
    sid = str(storyline_id or "")
    if sid and str(getattr(player, NUDGE_ID_KEY, "") or "") == sid:
        eff = get_effective_ovr_display(player)
        base = get_base_ovr_display(player)
        return {
            "overall_before": base,
            "overall_after": base,
            "overall_delta": 0,
            "base_overall": base,
            "effective_overall": eff,
            "already_active": True,
            "permanent": False,
        }

    r = rng or random.Random()
    sev = str(legal_severity or tier or "minor").lower()

    if amount is not None:
        signed = int(amount)
    elif sev == "major":
        signed = -int(r.randint(4, 8))
    elif sev in ("moderate", "mid"):
        signed = -int(r.randint(2, 5))
    else:
        signed = -int(r.randint(1, 3))

    impact_reason = str(reason or "").strip()
    if not impact_reason:
        if signed >= 0:
            impact_reason = "Momentum / confidence surge lifting form"
        elif sev == "major":
            impact_reason = "Major controversy weighing on readiness"
        elif sev in ("moderate", "mid"):
            impact_reason = "Media pressure / locker-room distraction"
        else:
            impact_reason = "Off-ice distraction affecting form"

    src = str(cause_type or "STORYLINE_MORALE")
    duration = 18 if sev == "major" else 12 if sev in ("moderate", "mid") else 8
    meta = apply_temporary_ovr_modifier(
        player,
        source=src,
        amount=signed,
        reason=impact_reason,
        duration_games=duration,
        storyline_id=sid,
        cause_type=src,
        cause_event_id=str(cause_event_id or ""),
        modifier_type="storyline_readiness",
    )
    if not meta or meta.get("already_active"):
        return meta or {}
    meta.setdefault("games_remaining", duration if signed < 0 else 0)
    meta.setdefault("games_initial", duration if signed < 0 else 0)
    meta["impact_reason"] = impact_reason
    meta["cause_event_id"] = str(cause_event_id or "")
    meta["permanent"] = False
    meta["effect_summary"] = (
        f"Temporary readiness {meta.get('overall_before')} → {meta.get('overall_after')} "
        f"({meta.get('overall_delta'):+d}) — {impact_reason}"
        if meta.get("overall_delta")
        else impact_reason
    )
    meta["arc_status"] = "active"
    return meta


def build_impact_storyline_fields(
    meta: Dict[str, Any],
    *,
    return_estimate: str = "",
    return_date: str = "",
) -> Dict[str, Any]:
    """Unified impact payload for storylines / popups (with or without games out)."""
    if not meta:
        return {}
    gr = int(meta.get("games_remaining") or 0)
    if gr > 0:
        return build_conduct_storyline_fields(meta, return_estimate=return_estimate, return_date=return_date)
    ovr_b = meta.get("overall_before")
    ovr_a = meta.get("overall_after")
    base = meta.get("base_overall")
    eff = meta.get("effective_overall") or ovr_a
    delta = meta.get("overall_delta")
    reason = str(meta.get("impact_reason") or "")
    parts = []
    if delta is not None and int(delta) != 0:
        if ovr_b is not None and ovr_a is not None:
            parts.append(f"OVR {ovr_b} → {ovr_a} ({int(delta):+d})")
        elif base is not None:
            parts.append(f"OVR {base} → {eff} ({int(delta):+d})")
    if reason:
        parts.append(reason)
    return {
        "games_remaining": 0,
        "return_estimate": "",
        "return_date": "",
        "overall_before": ovr_b,
        "overall_after": ovr_a,
        "overall_delta": delta,
        "base_overall": base,
        "effective_overall": eff,
        "player_overall": ovr_b or base or eff or ovr_a,
        "impact_reason": reason,
        "effect_summary": meta.get("effect_summary") or (" · ".join(parts) if parts else ""),
        "arc_status": "active",
        "follow_up": "Player remains available — modifier expires with time or resolution.",
    }


def get_player_stat_allocation_modifiers(player: Any) -> Dict[str, float]:
    """
    Map active temporary readiness modifiers into engine stat-allocation fingerprints.

    Base effective OVR already flows through engine._gm_ovr_0_100; these deltas
    steer *how* readiness shows up (shots vs assists vs TOI), not raw talent.
    """
    ovr_delta = float(sum_active_modifier_amount(player))
    if ovr_delta == 0:
        return {}
    return {
        "readiness_ovr_delta": ovr_delta,
        "shot_involvement": ovr_delta * 0.004,
        "assist_involvement": ovr_delta * 0.003,
        "toi_readiness": ovr_delta * 0.003,
        "turnover_risk": max(0.0, -ovr_delta) * 0.003,
        "penalty_risk": max(0.0, -ovr_delta) * 0.0025,
    }


def serialize_ovr_modifiers_for_ui(player: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in _ensure_modifiers(player):
        if m.get("resolved"):
            continue
        out.append(
            {
                "source": str(m.get("source") or ""),
                "cause_type": str(m.get("cause_type") or ""),
                "amount": int(m.get("amount") or 0),
                "reason": str(m.get("reason") or ""),
                "games_remaining": int(m.get("games_remaining") or 0),
                "duration_games": int(m.get("duration_games") or 0),
            }
        )
    return out
