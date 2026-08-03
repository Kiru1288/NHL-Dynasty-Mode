"""
Conduct / legal incident state machine.

Allegation, team action, league action, and legal outcome stay separate.
Eligibility and organizational backlash replace the old permanent OVR wipe.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

# --- Status vocabularies -------------------------------------------------------

LIFECYCLE_STATUSES = (
    "reported",
    "under_investigation",
    "charged",
    "administrative_leave",
    "league_suspended",
    "team_suspended",
    "cleared",
    "disciplined",
    "resolved",
)

INFORMATION_STATUSES = (
    "unconfirmed",
    "reported",
    "confirmed_investigation",
    "charges_filed",
    "official_ruling",
)

LEGAL_STATUSES = (
    "none",
    "allegation",
    "investigation",
    "charged",
    "plea",
    "conviction",
    "acquittal",
    "withdrawn",
    "civil_settlement",
    "no_charges",
)

LEAGUE_STATUSES = (
    "none",
    "monitoring",
    "investigating",
    "suspended",
    "disciplined",
    "cleared",
)

TEAM_STATUSES = (
    "none",
    "monitoring",
    "administrative_leave",
    "team_suspended",
    "support_program",
    "cleared",
)

INCIDENT_FAMILIES = (
    "driving",
    "violence",
    "gambling",
    "financial",
    "substance",
    "harassment",
    "team_conduct",
    "public_disorder",
)

REGISTRY_KEY = "conduct_incidents"
PLAYER_INCIDENT_KEY = "_conduct_incident_id"
PLAYER_ELIGIBLE_KEY = "_conduct_eligible_to_play"
PLAYER_OVERRIDE_KEY = "_conduct_team_can_override"
PLAYER_BACKLASH_KEY = "_conduct_dress_backlash_risk"
TEAM_ORG_KEY = "_conduct_org_pressure"

# Soft readiness only — never erase physical talent.
READINESS_DROP_BY_SEVERITY = {
    "minor": (0, 1),
    "moderate": (1, 3),
    "major": (2, 6),
}

SUSPENSION_GAMES_BY_SEVERITY = {
    "minor": (0, 2),
    "moderate": (0, 8),
    "major": (8, 22),
}


def _safe_str(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default


def _rng(rng: Optional[random.Random] = None) -> random.Random:
    return rng if rng is not None else random.Random()


def infer_incident_family(text: str, *, legal_severity: str = "") -> str:
    t = str(text or "").lower()
    if any(k in t for k in ("dui", "impaired", "reckless driv", "speeding", "suspended license", "stunt-driving", "traffic")):
        return "driving"
    if any(k in t for k in ("betting", "gambling", "poker", "integrity")):
        return "gambling"
    if any(
        k in t
        for k in (
            "domestic",
            "assault",
            "bar fight",
            "violence",
            "weapons",
            "animal cruelty",
            "altercation",
            "confrontation",
        )
    ):
        return "violence"
    if any(k in t for k in ("sexual", "harassment", "stalking", "image-sharing", "non-consensual")):
        return "harassment"
    if any(k in t for k in ("drug", "prescription", "substance")):
        return "substance"
    if any(k in t for k in ("fraud", "tax", "crypto", "nft", "identity fraud", "scam")):
        return "financial"
    if any(k in t for k in ("hazing", "curfew", "coach", "locker room", "team suspension", "conduct violation")):
        return "team_conduct"
    if str(legal_severity or "").lower() == "major":
        return "violence"
    return "public_disorder"


def _get_registry(host: Any) -> Dict[str, Dict[str, Any]]:
    """host may be league or franchise session."""
    if host is None:
        return {}
    reg = getattr(host, REGISTRY_KEY, None)
    if not isinstance(reg, dict):
        reg = {}
        try:
            setattr(host, REGISTRY_KEY, reg)
        except Exception:
            pass
    return reg


def get_incident(host: Any, incident_id: str) -> Optional[Dict[str, Any]]:
    reg = _get_registry(host)
    row = reg.get(_safe_str(incident_id))
    return dict(row) if isinstance(row, dict) else None


def get_active_incident_for_player(host: Any, player_id: str) -> Optional[Dict[str, Any]]:
    pid = _safe_str(player_id)
    if not pid:
        return None
    reg = _get_registry(host)
    active = []
    for row in reg.values():
        if not isinstance(row, dict):
            continue
        if _safe_str(row.get("player_id")) != pid:
            continue
        st = _safe_str(row.get("status"))
        if st in ("resolved", "cleared") and not row.get("ongoing_backlash"):
            continue
        if st == "resolved":
            continue
        active.append(row)
    if not active:
        return None
    active.sort(key=lambda r: str(r.get("updated_at") or r.get("created_at") or ""), reverse=True)
    return dict(active[0])


def _append_history(incident: Dict[str, Any], entry: str, *, detail: Optional[Dict[str, Any]] = None) -> None:
    hist = incident.get("history_entries")
    if not isinstance(hist, list):
        hist = []
        incident["history_entries"] = hist
    hist.append(
        {
            "entry": str(entry),
            "detail": dict(detail or {}),
        }
    )


def _sync_player_flags(player: Any, incident: Dict[str, Any]) -> None:
    if player is None:
        return
    try:
        setattr(player, PLAYER_INCIDENT_KEY, _safe_str(incident.get("incident_id")))
        setattr(player, PLAYER_ELIGIBLE_KEY, bool(incident.get("eligible_to_play")))
        setattr(player, PLAYER_OVERRIDE_KEY, bool(incident.get("team_can_override")))
        setattr(player, PLAYER_BACKLASH_KEY, float(incident.get("dress_backlash_risk") or 0.0))
        # Keep legacy games key in sync for tickers / UI that still read it.
        setattr(player, "_world_conduct_games_remaining", int(incident.get("games_remaining") or 0))
        setattr(player, "_world_conduct_storyline_id", _safe_str(incident.get("storyline_id") or incident.get("incident_id")))
        setattr(player, "_world_conduct_severity", _safe_str(incident.get("severity") or "minor"))
        setattr(player, "_world_conduct_status", _safe_str(incident.get("status")))
        setattr(player, "_world_conduct_resolved", _safe_str(incident.get("status")) in ("resolved", "cleared"))
    except Exception:
        pass


def player_eligible_to_dress(player: Any, host: Any = None) -> bool:
    """False when league/team suspension or leave blocks dressing."""
    if player is None:
        return True
    incident_id = getattr(player, PLAYER_INCIDENT_KEY, None)
    if incident_id:
        return bool(getattr(player, PLAYER_ELIGIBLE_KEY, True))
    try:
        if int(getattr(player, "_world_conduct_games_remaining", 0) or 0) > 0:
            return False
    except Exception:
        pass
    if host is not None:
        pid = _safe_str(getattr(player, "id", None) or getattr(player, "player_id", None))
        inc = get_active_incident_for_player(host, pid)
        if inc is not None:
            return bool(inc.get("eligible_to_play"))
    return True


def recompute_eligibility(incident: Dict[str, Any]) -> None:
    """Derive eligible_to_play / team_can_override from separate status channels."""
    league = _safe_str(incident.get("league_status") or "none")
    team = _safe_str(incident.get("team_status") or "none")
    status = _safe_str(incident.get("status") or "reported")

    if league == "suspended" or status == "league_suspended":
        incident["eligible_to_play"] = False
        incident["team_can_override"] = False
        return
    if team in ("administrative_leave", "team_suspended") or status in ("administrative_leave", "team_suspended"):
        incident["eligible_to_play"] = False
        incident["team_can_override"] = True
        return
    if status in ("resolved", "cleared") and league in ("cleared", "none", "disciplined") and team in ("cleared", "none", "support_program"):
        # Disciplined but available after serving time.
        if int(incident.get("games_remaining") or 0) > 0 and league == "suspended":
            incident["eligible_to_play"] = False
            incident["team_can_override"] = False
            return
        incident["eligible_to_play"] = True
        incident["team_can_override"] = False
        return
    # Investigation / reported / charged without leave: eligible but backlash risk.
    incident["eligible_to_play"] = True
    incident["team_can_override"] = False


def _initial_dress_backlash_risk(
    *,
    severity: str,
    family: str,
    information_status: str,
    evidence_confidence: float,
    fame: float,
) -> float:
    sev = str(severity or "minor").lower()
    base = {"minor": 0.15, "moderate": 0.35, "major": 0.72}.get(sev, 0.35)
    if family in ("gambling", "harassment", "violence"):
        base += 0.12
    info = str(information_status or "")
    if info in ("unconfirmed", "reported"):
        base *= 0.55
    elif info == "confirmed_investigation":
        base *= 0.85
    elif info in ("charges_filed", "official_ruling"):
        base *= 1.15
    base *= 0.75 + 0.5 * max(0.0, min(1.0, float(evidence_confidence)))
    base *= 0.85 + 0.4 * max(0.0, min(1.0, float(fame)))
    return max(0.05, min(1.0, base))


def create_conduct_incident(
    host: Any,
    *,
    player: Any,
    team_id: str,
    storyline_text: str,
    severity: str = "moderate",
    storyline_id: str = "",
    cause_event_id: str = "",
    player_fame: float = 0.5,
    rng: Optional[random.Random] = None,
    auto_league_suspend_major: bool = True,
) -> Dict[str, Any]:
    """Open a new multi-channel conduct incident (allegation-first)."""
    r = _rng(rng)
    sev = str(severity or "moderate").lower()
    if sev not in ("minor", "moderate", "major"):
        sev = "moderate"
    family = infer_incident_family(storyline_text, legal_severity=sev)
    pid = _safe_str(getattr(player, "id", None) or getattr(player, "player_id", None))
    incident_id = f"conduct:{uuid.uuid4().hex[:12]}"

    # Start as allegation / report — never as established guilt.
    information_status = "reported"
    legal_status = "allegation"
    league_status = "monitoring"
    team_status = "monitoring"
    status = "reported"
    games = 0

    evidence = round(r.uniform(0.25, 0.55) if sev != "major" else r.uniform(0.35, 0.7), 3)

    if sev == "major" and auto_league_suspend_major and family in ("gambling", "harassment"):
        # Integrity / severe misconduct: league can suspend while facts develop.
        league_status = "suspended"
        status = "league_suspended"
        information_status = "confirmed_investigation"
        legal_status = "investigation"
        lo, hi = SUSPENSION_GAMES_BY_SEVERITY["major"]
        games = int(r.randint(lo, hi))
    elif sev == "major":
        # Default major path: investigation + optional admin leave (not guilt).
        status = "under_investigation"
        information_status = "confirmed_investigation"
        legal_status = "investigation"
        league_status = "investigating"
        team_status = "administrative_leave"
        # Leave uses team channel; still not a conviction.
        status = "administrative_leave"
        lo, hi = SUSPENSION_GAMES_BY_SEVERITY["major"]
        games = int(r.randint(max(4, lo // 2), hi))
    elif sev == "moderate":
        status = "under_investigation"
        information_status = "reported"
        legal_status = "investigation"
        league_status = "monitoring"
        team_status = "monitoring"
        if r.random() < 0.35:
            team_status = "administrative_leave"
            status = "administrative_leave"
            games = int(r.randint(1, 6))
    else:
        status = "reported"
        information_status = "reported"
        legal_status = "allegation"
        if r.random() < 0.2:
            games = 1

    incident: Dict[str, Any] = {
        "incident_id": incident_id,
        "player_id": pid,
        "team_id": _safe_str(team_id),
        "storyline_id": _safe_str(storyline_id) or incident_id,
        "cause_event_id": _safe_str(cause_event_id),
        "incident_family": family,
        "severity": sev,
        "status": status,
        "information_status": information_status,
        "legal_status": legal_status,
        "league_status": league_status,
        "team_status": team_status,
        "evidence_confidence": evidence,
        "games_remaining": int(games),
        "games_initial": int(games),
        "eligible_to_play": True,
        "team_can_override": False,
        "resolution": "",
        "history_entries": [],
        "storyline_text": str(storyline_text or ""),
        "player_fame": float(player_fame),
        "dress_backlash_risk": 0.0,
        "ongoing_backlash": False,
        "support_program": False,
        "statement_tone": "",
        "repeat_count": _prior_incident_count(host, pid),
        "created_at": "",
        "updated_at": "",
    }
    incident["dress_backlash_risk"] = _initial_dress_backlash_risk(
        severity=sev,
        family=family,
        information_status=information_status,
        evidence_confidence=evidence,
        fame=float(player_fame),
    )
    recompute_eligibility(incident)
    _append_history(
        incident,
        "Incident reported — allegation under review (not an established finding).",
        detail={"information_status": information_status, "legal_status": legal_status},
    )
    if team_status == "administrative_leave":
        _append_history(incident, "Team placed player on administrative leave pending facts.")
    if league_status == "suspended":
        _append_history(incident, "League suspended player pending investigation.")

    reg = _get_registry(host)
    reg[incident_id] = incident
    try:
        setattr(host, REGISTRY_KEY, reg)
    except Exception:
        pass

    # Soft readiness only (temporary).
    _apply_readiness_modifier(player, incident, rng=r)
    _sync_player_flags(player, incident)
    return dict(incident)


def _prior_incident_count(host: Any, player_id: str) -> int:
    pid = _safe_str(player_id)
    n = 0
    for row in _get_registry(host).values():
        if isinstance(row, dict) and _safe_str(row.get("player_id")) == pid:
            n += 1
    return n


def _apply_readiness_modifier(player: Any, incident: Dict[str, Any], *, rng: random.Random) -> None:
    if player is None:
        return
    try:
        from app.sim_engine.franchise.storyline_conduct import apply_temporary_ovr_modifier

        sev = str(incident.get("severity") or "minor")
        lo, hi = READINESS_DROP_BY_SEVERITY.get(sev, (1, 3))
        drop = int(rng.randint(lo, hi)) if hi > 0 else 0
        if drop <= 0:
            return
        apply_temporary_ovr_modifier(
            player,
            source="CONDUCT_READINESS",
            amount=-drop,
            reason="Media distraction / reduced readiness during conduct matter",
            duration_games=max(8, int(incident.get("games_remaining") or 8) + 6),
            storyline_id=str(incident.get("storyline_id") or incident.get("incident_id")),
            cause_type="CONDUCT_INCIDENT",
            modifier_type="conduct_readiness",
        )
    except Exception:
        pass


def apply_gm_conduct_choice(
    host: Any,
    *,
    incident_id: str,
    choice_id: str,
    player: Any = None,
    statement_tone: str = "",
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Apply a functional GM decision to an incident. Returns effects summary."""
    r = _rng(rng)
    reg = _get_registry(host)
    incident = reg.get(_safe_str(incident_id))
    if not isinstance(incident, dict):
        # Fallback: look up by storyline id.
        for row in reg.values():
            if isinstance(row, dict) and _safe_str(row.get("storyline_id")) == _safe_str(incident_id):
                incident = row
                break
    if not isinstance(incident, dict):
        return {"ok": False, "reason": "incident_not_found"}

    cid = str(choice_id or "").strip()
    effects: Dict[str, Any] = {"ok": True, "choice_id": cid, "incident_id": incident.get("incident_id")}
    org = _team_org_bucket(host, _safe_str(incident.get("team_id")))

    if cid in ("suspend_internally", "place_on_leave", "administrative_leave"):
        incident["team_status"] = "administrative_leave" if cid != "suspend_internally" else "team_suspended"
        incident["status"] = "administrative_leave" if cid != "suspend_internally" else "team_suspended"
        if int(incident.get("games_remaining") or 0) <= 0:
            lo, hi = SUSPENSION_GAMES_BY_SEVERITY.get(str(incident.get("severity") or "moderate"), (2, 8))
            incident["games_remaining"] = int(r.randint(max(1, lo), max(lo, hi)))
            incident["games_initial"] = int(incident["games_remaining"])
        incident["dress_backlash_risk"] = max(0.1, float(incident.get("dress_backlash_risk") or 0.3) * 0.65)
        org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.2) + 0.08)
        org["owner_confidence"] = min(1.0, float(org.get("owner_confidence") or 0.55) + 0.04)
        org["fan_approval"] = float(org.get("fan_approval") or 0.5) + (0.03 if str(incident.get("severity")) != "minor" else 0.01)
        _append_history(incident, "GM placed player on leave / internal suspension.")
        effects["effect_summary"] = "Player ineligible. Media escalation slowed; possible grievance risk."

    elif cid in ("wait_league", "await_league"):
        incident["league_status"] = "investigating" if incident.get("league_status") == "monitoring" else incident.get("league_status")
        incident["status"] = "under_investigation"
        incident["team_status"] = "monitoring"
        incident["dress_backlash_risk"] = min(1.0, float(incident.get("dress_backlash_risk") or 0.4) + 0.12)
        org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.2) + 0.14)
        org["owner_confidence"] = max(0.0, float(org.get("owner_confidence") or 0.55) - 0.05)
        _append_history(incident, "GM will await league investigation; player remains roster-eligible.")
        effects["effect_summary"] = "Player remains eligible. Uncertainty and media pressure rise."

    elif cid in ("continue_playing",):
        # Team may want to dress him — never clears a league suspension.
        if _safe_str(incident.get("league_status")) != "suspended":
            incident["team_status"] = "monitoring"
            if _safe_str(incident.get("status")) in ("administrative_leave", "team_suspended"):
                incident["status"] = "under_investigation"
        incident["dress_backlash_risk"] = min(1.0, float(incident.get("dress_backlash_risk") or 0.5) + 0.22)
        org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.2) + 0.2)
        org["owner_confidence"] = max(0.0, float(org.get("owner_confidence") or 0.55) - 0.1)
        org["sponsor_confidence"] = max(0.0, float(org.get("sponsor_confidence") or 0.55) - 0.12)
        _append_history(incident, "GM will continue dressing the player during the matter.")
        effects["effect_summary"] = "Talent available, but dressing him risks severe organizational backlash."
        if _safe_str(incident.get("league_status")) == "suspended":
            effects["effect_summary"] = "League suspension remains — player cannot dress under any circumstances."

    elif cid in ("support_program",):
        incident["support_program"] = True
        incident["team_status"] = "support_program"
        org["media_heat"] = max(0.0, float(org.get("media_heat") or 0.3) - 0.04)
        org["fan_approval"] = float(org.get("fan_approval") or 0.5) + 0.02
        # Support does not replace discipline if already suspended.
        _append_history(incident, "Player entered support / assistance program (does not replace discipline).")
        effects["effect_summary"] = "Rehabilitation path opened. Does not clear leave or league suspension."

    elif cid in ("trade_immediately", "explore_trade"):
        incident["trade_market_restricted"] = True
        org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.2) + 0.06)
        _append_history(incident, "Front office exploring restricted trade market.")
        effects["effect_summary"] = "Restricted trade market opened — many clubs will refuse."
        effects["trade_market_restricted"] = True

    elif cid in ("release_statement",):
        tone = str(statement_tone or "support_investigation").strip() or "support_investigation"
        incident["statement_tone"] = tone
        if tone in ("condemn_alleged", "announce_leave"):
            org["fan_approval"] = float(org.get("fan_approval") or 0.5) + 0.04
            org["media_heat"] = max(0.0, float(org.get("media_heat") or 0.3) - 0.03)
            incident["dress_backlash_risk"] = max(0.08, float(incident.get("dress_backlash_risk") or 0.4) * 0.9)
        elif tone in ("support_player",):
            org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.2) + 0.08)
            org["sponsor_confidence"] = max(0.0, float(org.get("sponsor_confidence") or 0.55) - 0.05)
        elif tone in ("decline_comment",):
            org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.2) + 0.05)
        else:  # support_investigation
            org["owner_confidence"] = min(1.0, float(org.get("owner_confidence") or 0.55) + 0.03)
            org["media_heat"] = max(0.0, float(org.get("media_heat") or 0.3) - 0.02)
        _append_history(incident, f"Team released statement ({tone}).")
        effects["effect_summary"] = f"Public statement issued ({tone})."
        effects["statement_tone"] = tone

    elif cid in ("do_nothing",):
        org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.2) + 0.18)
        org["owner_confidence"] = max(0.0, float(org.get("owner_confidence") or 0.55) - 0.12)
        org["sponsor_confidence"] = max(0.0, float(org.get("sponsor_confidence") or 0.55) - 0.08)
        incident["dress_backlash_risk"] = min(1.0, float(incident.get("dress_backlash_risk") or 0.4) + 0.15)
        _append_history(incident, "GM took no immediate action — pressure escalates.")
        effects["effect_summary"] = "Media heat and owner pressure escalate."

    else:
        effects["ok"] = False
        effects["reason"] = f"unknown_choice:{cid}"
        return effects

    recompute_eligibility(incident)
    reg[_safe_str(incident.get("incident_id"))] = incident
    if player is not None:
        _sync_player_flags(player, incident)
    effects["eligible_to_play"] = bool(incident.get("eligible_to_play"))
    effects["status"] = incident.get("status")
    effects["org"] = dict(org)
    return effects


def _team_org_bucket(host: Any, team_id: str) -> Dict[str, float]:
    store = getattr(host, TEAM_ORG_KEY, None)
    if not isinstance(store, dict):
        store = {}
        try:
            setattr(host, TEAM_ORG_KEY, store)
        except Exception:
            pass
    tid = _safe_str(team_id) or "_league"
    row = store.get(tid)
    if not isinstance(row, dict):
        row = {
            "owner_confidence": 0.62,
            "fan_approval": 0.55,
            "media_heat": 0.15,
            "sponsor_confidence": 0.6,
            "team_reputation": 0.58,
            "locker_room_trust": 0.55,
            "revenue_modifier": 1.0,
            "free_agent_attractiveness": 0.55,
        }
        store[tid] = row
    return row


def get_team_org_pressure(host: Any, team_id: str) -> Dict[str, float]:
    return dict(_team_org_bucket(host, team_id))


def get_team_revenue_modifier(host: Any, team_id: str) -> float:
    org = _team_org_bucket(host, team_id)
    try:
        return max(0.55, min(1.08, float(org.get("revenue_modifier") or 1.0)))
    except Exception:
        return 1.0


def apply_dress_backlash(
    host: Any,
    *,
    team_id: str,
    player: Any,
    incident: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Punish the org for dressing a controversial but eligible player."""
    pid = _safe_str(getattr(player, "id", None) or "")
    inc = incident or get_active_incident_for_player(host, pid)
    if not isinstance(inc, dict):
        return {}
    if not bool(inc.get("eligible_to_play")):
        return {}
    # No meaningful backlash once fully cleared/resolved without ongoing flag.
    if _safe_str(inc.get("status")) in ("resolved", "cleared") and not inc.get("ongoing_backlash"):
        return {}
    risk = float(inc.get("dress_backlash_risk") or 0.0)
    if risk < 0.08:
        return {}

    org = _team_org_bucket(host, team_id)
    hit = risk
    info = _safe_str(inc.get("information_status"))
    if info in ("charges_filed", "official_ruling"):
        hit *= 1.35
    elif info in ("unconfirmed", "reported"):
        hit *= 0.7

    org["media_heat"] = min(1.0, float(org.get("media_heat") or 0.15) + 0.12 * hit)
    org["owner_confidence"] = max(0.0, float(org.get("owner_confidence") or 0.6) - 0.1 * hit)
    org["fan_approval"] = max(0.0, float(org.get("fan_approval") or 0.55) - 0.09 * hit)
    org["sponsor_confidence"] = max(0.0, float(org.get("sponsor_confidence") or 0.6) - 0.11 * hit)
    org["team_reputation"] = max(0.0, float(org.get("team_reputation") or 0.58) - 0.08 * hit)
    org["locker_room_trust"] = max(0.0, float(org.get("locker_room_trust") or 0.55) - 0.05 * hit)
    org["free_agent_attractiveness"] = max(0.0, float(org.get("free_agent_attractiveness") or 0.55) - 0.07 * hit)
    org["revenue_modifier"] = max(0.55, float(org.get("revenue_modifier") or 1.0) - 0.06 * hit)

    _append_history(
        inc,
        "Team dressed player during active conduct matter — organizational backlash applied.",
        detail={"risk": round(hit, 3), "revenue_modifier": org["revenue_modifier"]},
    )
    reg = _get_registry(host)
    reg[_safe_str(inc.get("incident_id"))] = inc
    return {
        "incident_id": inc.get("incident_id"),
        "backlash": round(hit, 3),
        "org": dict(org),
    }


def tick_incident_games(host: Any, player: Any) -> Optional[Dict[str, Any]]:
    """Decrement games_remaining for the player's active incident after a team game."""
    if player is None:
        return None
    pid = _safe_str(getattr(player, "id", None) or "")
    inc = get_active_incident_for_player(host, pid)
    if not isinstance(inc, dict):
        # Legacy path.
        try:
            from app.sim_engine.franchise.storyline_conduct import tick_conduct_games_missed

            tick_conduct_games_missed(player)
        except Exception:
            pass
        return None

    gr = int(inc.get("games_remaining") or 0)
    if gr > 0:
        inc["games_remaining"] = gr - 1
    try:
        from app.sim_engine.franchise.storyline_conduct import tick_player_ovr_modifiers

        tick_player_ovr_modifiers(player)
    except Exception:
        pass

    cleared = None
    if int(inc.get("games_remaining") or 0) <= 0 and _safe_str(inc.get("status")) in (
        "league_suspended",
        "team_suspended",
        "administrative_leave",
        "disciplined",
    ):
        cleared = resolve_incident_availability(host, incident=inc, player=player)

    _sync_player_flags(player, inc)
    reg = _get_registry(host)
    reg[_safe_str(inc.get("incident_id"))] = inc
    return cleared


def resolve_incident_availability(
    host: Any,
    *,
    incident: Dict[str, Any],
    player: Any = None,
    legal_outcome: str = "",
    league_outcome: str = "",
) -> Dict[str, Any]:
    """After games served / ruling — restore eligibility without inventing guilt."""
    legal = str(legal_outcome or incident.get("legal_status") or "no_charges")
    if legal in ("allegation", "investigation"):
        legal = "no_charges"
    incident["legal_status"] = legal

    league = str(league_outcome or "")
    if not league:
        if str(incident.get("severity")) == "major" and legal in ("conviction", "plea"):
            league = "disciplined"
        elif legal in ("acquittal", "withdrawn", "no_charges"):
            league = "cleared"
        else:
            league = "cleared"
    incident["league_status"] = league
    incident["team_status"] = "cleared"
    incident["games_remaining"] = 0
    incident["information_status"] = "official_ruling"

    if league == "disciplined":
        incident["status"] = "disciplined"
        incident["resolution"] = f"League discipline applied; legal status={legal}."
        incident["ongoing_backlash"] = True
        incident["dress_backlash_risk"] = max(0.15, float(incident.get("dress_backlash_risk") or 0.3) * 0.55)
    else:
        incident["status"] = "cleared"
        incident["resolution"] = f"Cleared to return; legal status={legal}."
        incident["ongoing_backlash"] = bool(float(incident.get("dress_backlash_risk") or 0) > 0.35)
        incident["dress_backlash_risk"] = max(0.0, float(incident.get("dress_backlash_risk") or 0) * 0.35)

    recompute_eligibility(incident)
    _append_history(incident, incident["resolution"])
    if player is not None:
        _sync_player_flags(player, incident)
        # Soft lingering readiness — not permanent talent wipe.
        try:
            from app.sim_engine.franchise.storyline_conduct import apply_temporary_ovr_modifier

            if str(incident.get("severity")) == "major":
                apply_temporary_ovr_modifier(
                    player,
                    source="CONDUCT_RECOVERY",
                    amount=-2,
                    reason="Reintegration / rust after conduct absence",
                    duration_games=16,
                    storyline_id=f"{incident.get('storyline_id')}:recovery",
                    cause_type="CONDUCT_RESOLVED",
                    modifier_type="conduct_readiness",
                )
        except Exception:
            pass
    reg = _get_registry(host)
    reg[_safe_str(incident.get("incident_id"))] = incident
    return dict(incident)


def advance_investigation(
    host: Any,
    *,
    incident_id: str,
    player: Any = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Optional beat: allegation → investigation → charges/ruling."""
    r = _rng(rng)
    inc = get_incident(host, incident_id)
    if not isinstance(inc, dict):
        return {}
    info = _safe_str(inc.get("information_status"))
    legal = _safe_str(inc.get("legal_status"))
    if info == "reported":
        inc["information_status"] = "confirmed_investigation"
        inc["legal_status"] = "investigation"
        inc["status"] = "under_investigation"
        inc["league_status"] = "investigating"
        _append_history(inc, "Investigation confirmed.")
    elif info == "confirmed_investigation" and legal == "investigation":
        roll = r.random()
        conf = float(inc.get("evidence_confidence") or 0.4)
        if roll < 0.22 * conf:
            inc["information_status"] = "charges_filed"
            inc["legal_status"] = "charged"
            inc["status"] = "charged"
            _append_history(inc, "Charges filed — still not a conviction.")
            if str(inc.get("severity")) == "major" and inc.get("league_status") != "suspended":
                inc["league_status"] = "suspended"
                inc["status"] = "league_suspended"
                inc["eligible_to_play"] = False
                if int(inc.get("games_remaining") or 0) <= 0:
                    lo, hi = SUSPENSION_GAMES_BY_SEVERITY["major"]
                    inc["games_remaining"] = int(r.randint(lo, hi))
                _append_history(inc, "League suspension while charges pending.")
        elif roll < 0.55:
            inc["legal_status"] = "no_charges"
            inc["information_status"] = "official_ruling"
            resolve_incident_availability(host, incident=inc, player=player, legal_outcome="no_charges")
        else:
            _append_history(inc, "Investigation continues; no ruling yet.")
    recompute_eligibility(inc)
    if player is not None:
        _sync_player_flags(player, inc)
    reg = _get_registry(host)
    reg[_safe_str(inc.get("incident_id"))] = inc
    return dict(inc)


def legal_gm_choice_options() -> List[Dict[str, Any]]:
    return [
        {"id": "suspend_internally", "label": "Suspend player internally", "effect_summary": "Ineligible; slows media escalation."},
        {"id": "place_on_leave", "label": "Place on administrative leave", "effect_summary": "Ineligible pending facts."},
        {"id": "wait_league", "label": "Await league investigation", "effect_summary": "Remains eligible; pressure grows."},
        {"id": "continue_playing", "label": "Continue playing him", "effect_summary": "Talent available; severe backlash risk."},
        {"id": "support_program", "label": "Send to support program", "effect_summary": "Rehab path; does not replace discipline."},
        {"id": "explore_trade", "label": "Explore trade market", "effect_summary": "Restricted buyers only."},
        {"id": "release_statement", "label": "Release statement", "effect_summary": "Shapes public and player reaction."},
        {"id": "do_nothing", "label": "Do nothing", "effect_summary": "Media heat and owner pressure escalate."},
    ]


def serialize_incident_for_ui(incident: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "incident_id": incident.get("incident_id"),
        "player_id": incident.get("player_id"),
        "team_id": incident.get("team_id"),
        "incident_family": incident.get("incident_family"),
        "severity": incident.get("severity"),
        "status": incident.get("status"),
        "information_status": incident.get("information_status"),
        "legal_status": incident.get("legal_status"),
        "league_status": incident.get("league_status"),
        "team_status": incident.get("team_status"),
        "evidence_confidence": incident.get("evidence_confidence"),
        "games_remaining": incident.get("games_remaining"),
        "eligible_to_play": incident.get("eligible_to_play"),
        "team_can_override": incident.get("team_can_override"),
        "resolution": incident.get("resolution"),
        "dress_backlash_risk": incident.get("dress_backlash_risk"),
        "storyline_text": incident.get("storyline_text"),
        "history_entries": list(incident.get("history_entries") or [])[-8:],
        "allegation_note": "Public reports are allegations until an official ruling.",
    }
