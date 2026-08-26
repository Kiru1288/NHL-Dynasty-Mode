"""Persistent NHL player-agent business layer.

Five recognizable agents (~20% of skaters each), deterministic assignment,
GM relationship memory, and trade-demand / contract negotiation modifiers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

AGENT_IDS: Tuple[str, ...] = ("carter", "walsh", "kim", "rossi", "blake")

PLAYER_AGENTS: List[Dict[str, Any]] = [
    {
        "id": "carter",
        "name": "Allan Carter",
        "agency": "Carter Hockey Group",
        "style": "leaker",
        "style_label": "Aggressive Leaker",
        "leak_tendency": 0.74,
        "negotiation": "aggressive",
        "patience": 0.35,
        "crisis_timer_mult": 1.0,
        "ntc_pressure": 0.55,
        "aav_priority": 0.92,
        "term_priority": 0.42,
        "offer_sheet_interest": 0.48,
        "reconciliation_tendency": 0.22,
    },
    {
        "id": "walsh",
        "name": "Patricia Walsh",
        "agency": "Walsh Sports",
        "style": "discreet",
        "style_label": "Discreet / Patient",
        "leak_tendency": 0.09,
        "negotiation": "patient",
        "patience": 0.88,
        "crisis_timer_mult": 1.0,
        "ntc_pressure": 0.62,
        "aav_priority": 0.58,
        "term_priority": 0.86,
        "offer_sheet_interest": 0.12,
        "reconciliation_tendency": 0.78,
    },
    {
        "id": "kim",
        "name": "Daniel Kim",
        "agency": "Kim & Partners",
        "style": "leverage",
        "style_label": "Leverage / Competitive",
        "leak_tendency": 0.52,
        "negotiation": "competitive",
        "patience": 0.52,
        "crisis_timer_mult": 0.92,
        "ntc_pressure": 0.50,
        "aav_priority": 0.78,
        "term_priority": 0.55,
        "offer_sheet_interest": 0.64,
        "reconciliation_tendency": 0.40,
    },
    {
        "id": "rossi",
        "name": "Marco Rossi",
        "agency": "Northline Athletes",
        "style": "media_savvy",
        "style_label": "Media Savvy / Stable",
        "leak_tendency": 0.38,
        "negotiation": "stable",
        "patience": 0.68,
        "crisis_timer_mult": 0.96,
        "ntc_pressure": 0.58,
        "aav_priority": 0.66,
        "term_priority": 0.68,
        "offer_sheet_interest": 0.28,
        "reconciliation_tendency": 0.58,
    },
    {
        "id": "blake",
        "name": "Jordan Blake",
        "agency": "Blake Advisory",
        "style": "disruptor",
        "style_label": "Disruptor / Demanding",
        "leak_tendency": 0.61,
        "negotiation": "demanding",
        "patience": 0.18,
        "crisis_timer_mult": 0.55,
        "ntc_pressure": 0.88,
        "aav_priority": 0.95,
        "term_priority": 0.35,
        "offer_sheet_interest": 0.72,
        "reconciliation_tendency": 0.08,
    },
]

_AGENT_BY_ID: Dict[str, Dict[str, Any]] = {a["id"]: a for a in PLAYER_AGENTS}

_STYLE_LABELS = {
    "leaker": "Aggressive Leaker",
    "discreet": "Discreet / Patient",
    "leverage": "Leverage / Competitive",
    "media_savvy": "Media Savvy / Stable",
    "disruptor": "Disruptor / Demanding",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _player_id(player: Any) -> str:
    return str(getattr(player, "id", None) or getattr(player, "player_id", "") or "")


def ensure_agent_relationships(session: Any) -> Dict[str, Any]:
    rel = getattr(session, "agent_relationships", None)
    if not isinstance(rel, dict):
        rel = {}
        session.agent_relationships = rel
    return rel


def _default_gm_relationship(agent_id: str) -> Dict[str, Any]:
    return {
        "agent_id": agent_id,
        "agent_gm_trust": 0.55,
        "agent_respect": 0.50,
        "contracts_completed": 0,
        "negotiations_failed": 0,
        "clients_traded": 0,
        "promises_broken": 0,
        "public_conflicts": 0,
    }


def get_agent_gm_relationship(session: Any, agent_id: str) -> Dict[str, Any]:
    rel = ensure_agent_relationships(session)
    aid = str(agent_id or "")
    row = rel.get(aid)
    if not isinstance(row, dict):
        row = _default_gm_relationship(aid)
        rel[aid] = row
    return row


def assign_agent_id(player: Any) -> str:
    """Deterministic ~20% split across the five agents."""
    pid = _player_id(player)
    if not pid:
        return AGENT_IDS[0]
    digest = hashlib.sha256(pid.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % len(AGENT_IDS)
    return AGENT_IDS[bucket]


def ensure_player_agent(player: Any, session: Optional[Any] = None) -> Dict[str, Any]:
    stored = getattr(player, "agent_profile", None)
    if isinstance(stored, dict) and stored.get("id") in _AGENT_BY_ID:
        return dict(_AGENT_BY_ID[stored["id"]], **{k: v for k, v in stored.items() if k not in ("name", "agency")})

    aid = assign_agent_id(player)
    agent = dict(_AGENT_BY_ID[aid])
    profile = {
        "id": aid,
        "name": agent["name"],
        "agency": agent["agency"],
        "style": agent["style"],
        "style_label": agent.get("style_label") or _STYLE_LABELS.get(agent["style"], agent["style"]),
        "assigned": True,
    }
    try:
        setattr(player, "agent_profile", profile)
        setattr(player, "agent_id", aid)
    except Exception:
        pass
    if session is not None:
        get_agent_gm_relationship(session, aid)
    return agent


def get_player_agent(player: Any, session: Optional[Any] = None) -> Dict[str, Any]:
    return ensure_player_agent(player, session)


def agent_public_view(player: Any, session: Optional[Any] = None) -> Dict[str, Any]:
    agent = ensure_player_agent(player, session)
    return {
        "id": agent.get("id"),
        "name": agent.get("name"),
        "agency": agent.get("agency"),
        "style": agent.get("style"),
        "style_label": agent.get("style_label") or _STYLE_LABELS.get(str(agent.get("style")), ""),
    }


def assign_league_agents(session: Any) -> None:
    """Ensure every NHL roster player has a persistent agent."""
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return
    for team in list(getattr(league, "teams", None) or []):
        for player in list(getattr(team, "roster", None) or []):
            if getattr(player, "retired", False):
                continue
            ensure_player_agent(player, session)


def agent_patience_modifier(agent: Dict[str, Any], gm_rel: Dict[str, Any]) -> float:
    base = float(agent.get("patience", 0.5) or 0.5)
    trust = float(gm_rel.get("agent_gm_trust", 0.55) or 0.55)
    conflicts = int(gm_rel.get("public_conflicts", 0) or 0)
    return _clamp(base + (trust - 0.5) * 0.35 - conflicts * 0.04, 0.05, 0.98)


def agent_crisis_initial_seconds(
    *,
    character: int,
    agent: Dict[str, Any],
    gm_rel: Dict[str, Any],
    previous_demands: int = 0,
) -> int:
    """Standard 360s; low character / Blake / bad GM history can start at 240 or 120."""
    seconds = 360
    char = int(character or 74)
    if char < 63:
        seconds = min(seconds, 240)
    if char < 58:
        seconds = min(seconds, 120)
    mult = float(agent.get("crisis_timer_mult", 1.0) or 1.0)
    seconds = int(round(seconds * mult))
    patience = agent_patience_modifier(agent, gm_rel)
    if patience < 0.30:
        seconds = min(seconds, 240)
    if patience < 0.20:
        seconds = min(seconds, 120)
    if previous_demands >= 2:
        seconds = min(seconds, 240)
    if previous_demands >= 4:
        seconds = min(seconds, 120)
    return max(120, min(360, seconds))


def agent_leak_probability(agent: Dict[str, Any], *, crisis_stage: int, gm_rel: Dict[str, Any]) -> float:
    base = float(agent.get("leak_tendency", 0.3) or 0.3)
    trust = float(gm_rel.get("agent_gm_trust", 0.55) or 0.55)
    stage_boost = {1: 0.0, 2: 0.12, 3: 0.28, 4: 0.45}.get(int(crisis_stage or 1), 0.0)
    return _clamp(base + stage_boost - (trust - 0.5) * 0.25, 0.02, 0.98)


def agent_destination_shrink_factor(agent: Dict[str, Any], crisis_stage: int) -> float:
    """Higher = tighter market as crisis escalates."""
    base = 1.0 - float(agent.get("ntc_pressure", 0.5) or 0.5) * 0.15
    stage = int(crisis_stage or 1)
    shrink = {1: 1.0, 2: 0.72, 3: 0.42, 4: 0.18}.get(stage, 1.0)
    if agent.get("style") == "disruptor":
        shrink *= 0.85
    if agent.get("style") == "discreet":
        shrink = min(1.0, shrink + 0.12)
    return _clamp(base * shrink, 0.08, 1.0)


def record_agent_negotiation_outcome(
    session: Any,
    agent_id: str,
    *,
    completed: bool = False,
    failed: bool = False,
    promise_broken: bool = False,
    public_conflict: bool = False,
    client_traded: bool = False,
) -> None:
    rel = get_agent_gm_relationship(session, agent_id)
    if completed:
        rel["contracts_completed"] = int(rel.get("contracts_completed") or 0) + 1
        rel["agent_gm_trust"] = _clamp(float(rel.get("agent_gm_trust", 0.55)) + 0.04, 0.0, 1.0)
    if failed:
        rel["negotiations_failed"] = int(rel.get("negotiations_failed") or 0) + 1
        rel["agent_gm_trust"] = _clamp(float(rel.get("agent_gm_trust", 0.55)) - 0.06, 0.0, 1.0)
    if promise_broken:
        rel["promises_broken"] = int(rel.get("promises_broken") or 0) + 1
        rel["agent_gm_trust"] = _clamp(float(rel.get("agent_gm_trust", 0.55)) - 0.10, 0.0, 1.0)
    if public_conflict:
        rel["public_conflicts"] = int(rel.get("public_conflicts") or 0) + 1
        rel["agent_respect"] = _clamp(float(rel.get("agent_respect", 0.5)) - 0.08, 0.0, 1.0)
    if client_traded:
        rel["clients_traded"] = int(rel.get("clients_traded") or 0) + 1


def contract_leverage_pressure(agent: Dict[str, Any], *, gap_pct: float) -> float:
    """Small stability pressure when agent uses trade threat during negotiations."""
    if gap_pct < 0.08:
        return 0.0
    style = str(agent.get("style") or "")
    base = 2.0 + gap_pct * 12.0
    if style in ("leaker", "disruptor", "leverage"):
        base *= 1.35
    if style == "discreet":
        base *= 0.55
    return base
