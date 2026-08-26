"""Player-initiated trade demands driven by Trade Stability Score + real-time crisis timer."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.player_agent_engine import (
    agent_crisis_initial_seconds,
    agent_destination_shrink_factor,
    agent_leak_probability,
    agent_public_view,
    assign_league_agents,
    ensure_player_agent,
    get_agent_gm_relationship,
)
from app.sim_engine.franchise.trade_stability_engine import (
    CRISIS_DEADLINE_MAX,
    apply_trade_hub_exposure,
    clear_demand_temporary_modifiers,
    crisis_distressed_asset_cost,
    crisis_stage_from_remaining,
    crisis_trade_value_multiplier,
    ensure_player_storyline_state,
    ensure_trade_stability_state,
    gather_player_concerns,
    primary_complaint_from_pressures,
    stability_to_escalation_level,
    update_player_stability,
)

CANADA_ABBRS = frozenset({"TOR", "MTL", "OTT", "VAN", "CGY", "EDM", "WPG", "SEA"})
SMALL_MARKET_ABBRS = frozenset(
    {"BUF", "OTT", "CBJ", "CGY", "WPG", "ARI", "UTA", "SEA", "NSH", "MIN", "CAR", "FLA"}
)

REASON_COPY = {
    "role": {
        "headline": "{name} demands a trade over role and deployment",
        "body": "{name} feels misused relative to his ability and has formally requested a trade through his agent.",
    },
    "losing": {
        "headline": "{name} wants out — tired of losing",
        "body": "{name} has asked for a trade after another stretch of losses. The locker room knows.",
    },
    "management": {
        "headline": "{name} has lost faith in management",
        "body": "{name} no longer trusts the front office and has requested a trade through his agent.",
    },
    "trade_exposure": {
        "headline": "{name} wants out after repeated trade talks",
        "body": "{name} is fed up with being shopped and has formally requested a move.",
    },
    "locker_room_disruptor": {
        "headline": "LOCKER-ROOM DISRUPTOR — {name} torpedoes the room",
        "body": (
            "{name} has gone nuclear. Unwilling to be reasoned with, he is trying to force a "
            "move even if it burns bridges and torpedoes his own value."
        ),
    },
    "general": {
        "headline": "{name} has formally requested a trade",
        "body": "{name} has delivered a trade request through his agent. The relationship is at a breaking point.",
    },
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _player_id(player: Any) -> str:
    return str(_get(player, "id", "") or _get(player, "player_id", "") or "")


def _player_name(player: Any) -> str:
    ident = _get(player, "identity", None)
    return str(_get(ident, "name", "") or _get(player, "name", "") or "Player")


def _player_ovr(player: Any) -> float:
    raw = _get(player, "ovr", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = 70.0
    try:
        o = float(raw if raw is not None else _get(player, "overall", 70) or 70)
    except Exception:
        o = 70.0
    if o <= 1.5:
        o *= 99.0
    return o


def _team_abbr(team: Any) -> str:
    return str(_get(team, "abbr", "") or _get(team, "team_id", "") or _get(team, "id", "") or "").upper()


def ensure_trade_demands(session: Any) -> Dict[str, Any]:
    book = getattr(session, "trade_demands", None)
    if not isinstance(book, dict):
        book = {}
        session.trade_demands = book
    return book


def active_demand_for_player(session: Any, player_id: str) -> Optional[Dict[str, Any]]:
    book = ensure_trade_demands(session)
    row = book.get(str(player_id))
    if isinstance(row, dict) and str(row.get("status") or "") == "open":
        return row
    return None


def seed_mntc_destinations(player: Any, league: Any, *, list_size: int = 8, rng: Any = None) -> List[str]:
    contract = _get(player, "contract", None)
    if contract is None:
        return []
    existing = (
        list(_get(contract, "approved_trade_teams", None) or [])
        or list(_get(contract, "approved_trade_team_ids", None) or [])
        or list(_get(contract, "approved_destinations", None) or [])
    )
    if existing:
        return [str(x).upper() for x in existing if x]
    clause = str(
        _get(contract, "clause", "")
        or _get(contract, "clause_type", "")
        or _get(contract, "trade_clause", "")
        or ""
    ).upper()
    if "MNTC" not in clause and "M-NTC" not in clause and "MODIFIED" not in clause:
        size = int(_get(contract, "trade_list_size", 0) or 0)
        if size <= 0:
            return []
    else:
        size = int(_get(contract, "trade_list_size", 0) or list_size or 8)
    size = max(1, min(32, size or list_size))
    teams = list(_get(league, "teams", None) or [])
    abbrs = []
    for t in teams:
        ab = _team_abbr(t)
        if ab:
            abbrs.append(ab)
    abbrs = sorted(set(abbrs))
    if not abbrs:
        return []
    if rng is not None and hasattr(rng, "sample"):
        chosen = list(rng.sample(abbrs, min(size, len(abbrs))))
    else:
        chosen = abbrs[:size]
    try:
        if isinstance(contract, dict):
            contract["approved_trade_teams"] = chosen
            contract["approved_destinations"] = chosen
        else:
            setattr(contract, "approved_trade_teams", chosen)
            setattr(contract, "approved_destinations", chosen)
    except Exception:
        pass
    return chosen


def _preferred_destinations(
    player: Any,
    team: Any,
    league: Any,
    reason: str,
    *,
    rng: Any,
    list_size: int = 12,
) -> List[str]:
    seeded = seed_mntc_destinations(player, league, list_size=list_size, rng=rng)
    if seeded:
        return seeded[:32]
    teams = list(_get(league, "teams", None) or [])
    home = _team_abbr(team)
    candidates: List[str] = []
    for t in teams:
        ab = _team_abbr(t)
        if not ab or ab == home:
            continue
        if reason == "canada" and ab in CANADA_ABBRS:
            continue
        if reason == "small_market" and ab in SMALL_MARKET_ABBRS:
            continue
        candidates.append(ab)
    if not candidates:
        candidates = [ab for ab in (_team_abbr(t) for t in teams) if ab and ab != home]
    n = max(1, min(list_size, len(candidates)))
    if rng is not None and hasattr(rng, "sample") and candidates:
        return list(rng.sample(candidates, min(n, len(candidates))))
    return candidates[:n]


def _snapshot_value(player: Any, team: Any, league: Any, *, crisis_stage: int = 1) -> float:
    try:
        from app.sim_engine.trades.trade_value import evaluate_player_asset_value

        row = evaluate_player_asset_value(player, team, team, league, context={})
        base = float(row.get("trade_value") or 0.0)
        mult = crisis_trade_value_multiplier(crisis_stage)
        distressed = crisis_distressed_asset_cost(base, crisis_stage)
        return max(-20.0, base * mult - distressed)
    except Exception:
        return max(5.0, _player_ovr(player) * 0.7)


def _apply_crisis_value_state(player: Any, *, crisis_stage: int, base_before: float) -> Tuple[float, float]:
    mult = crisis_trade_value_multiplier(crisis_stage)
    distressed = crisis_distressed_asset_cost(base_before, crisis_stage)
    effective_mult = mult
    if crisis_stage >= 4:
        effective_mult = min(effective_mult, 0.12)
    try:
        setattr(player, "_trade_demand_active", True)
        setattr(player, "_crisis_trade_value_mult", mult)
        setattr(player, "_crisis_distressed_asset", distressed)
        setattr(player, "_crisis_trade_stage", crisis_stage)
        setattr(player, "_systemic_trade_value_mult", mult)
        if crisis_stage >= 3:
            pst = ensure_player_storyline_state(player)
            if int(pst.get("character") or 80) < 65 or crisis_stage >= 4:
                setattr(player, "locker_room_disruptor", True)
    except Exception:
        pass
    after = max(-20.0, base_before * mult - distressed)
    return mult, after


def _reason_from_pressures(pressures: Dict[str, float], *, character: int) -> str:
    if not pressures:
        return "general"
    top = max(pressures.items(), key=lambda kv: kv[1])[0]
    if top == "trade_exposure":
        return "trade_exposure"
    if top in ("role", "performance"):
        return "role"
    if top in ("management", "coach"):
        return "management"
    if top == "winning":
        return "losing"
    if character < 62 and pressures.get("role", 0) + pressures.get("management", 0) < 8:
        return "locker_room_disruptor"
    return "general"


def _start_crisis_timer(
    session: Any,
    player: Any,
    *,
    character: int,
    previous_demands: int,
) -> Dict[str, Any]:
    agent = ensure_player_agent(player, session)
    gm_rel = get_agent_gm_relationship(session, str(agent.get("id") or ""))
    initial = agent_crisis_initial_seconds(
        character=character,
        agent=agent,
        gm_rel=gm_rel,
        previous_demands=previous_demands,
    )
    now = time.time()
    return {
        "deadline_seconds": initial,
        "initial_seconds": initial,
        "remaining_seconds": initial,
        "last_sync_unix": now,
        "crisis_stage": 1,
        "started_at_unix": now,
    }


def sync_trade_demand_crises(session: Any, *, elapsed_hint: Optional[float] = None) -> None:
    """Decrement active crisis timers (only while session is synced — not offline punishment)."""
    book = ensure_trade_demands(session)
    now = time.time()
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        crisis = row.get("crisis")
        if not isinstance(crisis, dict):
            continue
        last = float(crisis.get("last_sync_unix") or now)
        elapsed = float(elapsed_hint if elapsed_hint is not None else max(0.0, now - last))
        remaining = max(0.0, float(crisis.get("remaining_seconds") or 0) - elapsed)
        crisis["remaining_seconds"] = round(remaining, 2)
        crisis["last_sync_unix"] = now
        initial = int(crisis.get("initial_seconds") or CRISIS_DEADLINE_MAX)
        stage = crisis_stage_from_remaining(initial, int(round(remaining)))
        prev_stage = int(crisis.get("crisis_stage") or 1)
        crisis["crisis_stage"] = stage
        row["crisis_stage"] = stage

        player = _find_player(session, pid)
        if player is not None:
            team = _find_team_for_player(session, pid)
            league = getattr(getattr(session, "sim", None), "league", None)
            base_before = float(row.get("value_before") or _snapshot_value(player, team, league, crisis_stage=1))
            _, after = _apply_crisis_value_state(player, crisis_stage=stage, base_before=base_before)
            row["value_after"] = round(after, 1)
            row["value_delta"] = round(after - base_before, 1)

            if stage > prev_stage:
                _apply_crisis_stage_effects(session, row, player, team, league, stage)

        if remaining <= 0 and str(row.get("status")) == "open":
            row["status"] = "crisis_expired"
            row["resolved"] = False
            row["crisis_expired"] = True
            _enqueue_crisis_expired(session, row)


def _apply_crisis_stage_effects(
    session: Any,
    row: Dict[str, Any],
    player: Any,
    team: Any,
    league: Any,
    stage: int,
) -> None:
    import random

    agent = ensure_player_agent(player, session)
    gm_rel = get_agent_gm_relationship(session, str(agent.get("id") or ""))
    rng = getattr(getattr(session, "sim", None), "rng", None) or random.Random()

    shrink = agent_destination_shrink_factor(agent, stage)
    full = list(row.get("preferred_destinations_full") or row.get("preferred_destinations") or [])
    if full:
        keep = max(1, int(round(len(full) * shrink)))
        if keep < len(full):
            dests = list(rng.sample(full, keep)) if hasattr(rng, "sample") else full[:keep]
            row["preferred_destinations"] = dests
            row["destination_count"] = len(dests)

    leak_p = agent_leak_probability(agent, crisis_stage=stage, gm_rel=gm_rel)
    if stage >= 2 and float(rng.random()) < leak_p:
        row["leaked"] = True
        row["public_demand"] = stage >= 3

    pst = ensure_player_storyline_state(player)
    pst["gm_trust"] = max(0.05, float(pst.get("gm_trust", 0.72)) - 0.02 * stage)


def _find_player(session: Any, player_id: str) -> Any:
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return None
    pid = str(player_id)
    for team in getattr(league, "teams", None) or []:
        for p in getattr(team, "roster", None) or []:
            if _player_id(p) == pid:
                return p
    return None


def _find_team_for_player(session: Any, player_id: str) -> Any:
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return None
    pid = str(player_id)
    for team in getattr(league, "teams", None) or []:
        for p in getattr(team, "roster", None) or []:
            if _player_id(p) == pid:
                return team
    return None


def open_trade_demand(
    session: Any,
    player: Any,
    team: Any,
    *,
    reason: str,
    calendar_idx: int,
    iso_date: str = "",
    rng: Any = None,
    stability_row: Optional[Dict[str, Any]] = None,
    force_formal: bool = False,
) -> Dict[str, Any]:
    """Create/overwrite an open formal demand with real-time crisis timer."""
    import random

    r = rng if rng is not None else random.Random()
    league = getattr(getattr(session, "sim", None), "league", None)
    pid = _player_id(player)
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    pst = ensure_player_storyline_state(player)

    if stability_row is None:
        stability_row = update_player_stability(session, player, team)

    pressures = dict(stability_row.get("pressures") or {})
    character = int(stability_row.get("character") or pst.get("character") or 74)
    if not reason or reason == "auto":
        reason = _reason_from_pressures(pressures, character=character)

    disruptor = reason == "locker_room_disruptor" or int(stability_row.get("escalation_level") or 0) >= 4
    escalation = int(stability_row.get("escalation_level") or 3)
    if not force_formal and escalation < 3:
        return {"status": "warning", "escalation_level": escalation, "player_id": pid}

    previous_demands = int(pst.get("career_trade_demand_count") or 0)
    crisis = _start_crisis_timer(session, player, character=character, previous_demands=previous_demands)

    agent_view = agent_public_view(player, session)
    before = _snapshot_value(player, team, league, crisis_stage=1)
    _, after = _apply_crisis_value_state(player, crisis_stage=1, base_before=before)
    if disruptor:
        try:
            setattr(player, "locker_room_disruptor", True)
        except Exception:
            pass

    dests = _preferred_destinations(player, team, league, reason, rng=r, list_size=12)
    copy = REASON_COPY.get(reason) or REASON_COPY["general"]
    name = _player_name(player)
    demand_id = f"demand:{pid}:{calendar_idx}:{uuid.uuid4().hex[:6]}"
    complaint = primary_complaint_from_pressures(pressures)

    pst["career_trade_demand_count"] = previous_demands + 1
    pst["season_trade_demand_count"] = int(pst.get("season_trade_demand_count") or 0) + 1
    pst["previous_trade_demand_severity"] = escalation
    pst["previous_trade_demand_team"] = tid
    pst["previous_trade_demand_reason"] = reason

    row = {
        "demand_id": demand_id,
        "player_id": pid,
        "player_name": name,
        "team_id": tid,
        "status": "open",
        "formal": True,
        "reason_code": reason,
        "primary_complaint": complaint,
        "cause_type": "TRADE_DEMAND",
        "cause_event_id": demand_id,
        "opened_day": int(calendar_idx),
        "opened_iso": str(iso_date or ""),
        "trade_stability_score": stability_row.get("trade_stability_score"),
        "escalation_level": escalation,
        "value_before": round(float(before), 1),
        "value_after": round(float(after), 1),
        "value_delta": round(float(after) - float(before), 1),
        "preferred_destinations": dests,
        "preferred_destinations_full": list(dests),
        "destination_count": len(dests),
        "disruptor": disruptor,
        "dossier_label": "Locker-room disruptor" if disruptor else "Trade demand",
        "headline": copy["headline"].format(name=name),
        "body": copy["body"].format(name=name),
        "ovr": round(_player_ovr(player)),
        "agent": agent_view,
        "crisis": crisis,
        "crisis_stage": 1,
        "career_trade_demand_count": pst["career_trade_demand_count"],
        "season_trade_demand_count": pst["season_trade_demand_count"],
    }
    book = ensure_trade_demands(session)
    book[pid] = row
    try:
        _enqueue_demand_surfaces(session, row, team)
    except Exception:
        pass
    return row


def clear_trade_demand(session: Any, player_id: str, *, resolution: str = "cleared") -> None:
    book = ensure_trade_demands(session)
    pid = str(player_id)
    row = book.get(pid)
    if isinstance(row, dict):
        row["status"] = resolution
        row["resolved"] = True
        if resolution == "traded":
            row["crisis"] = None
    player = _find_player(session, pid)
    if player is not None:
        clear_demand_temporary_modifiers(player)


def process_trade_demand_day(session: Any, calendar_idx: int, day_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update stability for all players; open formal demands when warranted — no daily cap."""
    import random

    rng = getattr(getattr(session, "sim", None), "rng", None) or random.Random(int(calendar_idx) + 17)
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"opened": 0}

    assign_league_agents(session)
    sync_trade_demand_crises(session)
    book = ensure_trade_demands(session)

    iso = ""
    try:
        iso = str((day_meta or {}).get("iso") or (day_meta or {}).get("date") or "")
    except Exception:
        iso = ""

    opened: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    teams = list(_get(league, "teams", None) or [])
    for team in teams:
        for player in list(_get(team, "roster", None) or []):
            if _get(player, "retired", False):
                continue
            pid = _player_id(player)
            if not pid:
                continue
            if active_demand_for_player(session, pid):
                continue
            ovr = _player_ovr(player)
            if ovr < 74:
                continue

            stability_row = update_player_stability(session, player, team)
            escalation = int(stability_row.get("escalation_level") or 0)
            score = float(stability_row.get("trade_stability_score") or 100.0)

            if escalation >= 3:
                reason = _reason_from_pressures(
                    dict(stability_row.get("pressures") or {}),
                    character=int(stability_row.get("character") or 74),
                )
                row = open_trade_demand(
                    session,
                    player,
                    team,
                    reason=reason,
                    calendar_idx=int(calendar_idx),
                    iso_date=iso,
                    rng=rng,
                    stability_row=stability_row,
                    force_formal=True,
                )
                if row.get("status") == "open":
                    opened.append(row)
            elif escalation >= 1 and score < 68:
                warnings.append({"player_id": pid, "escalation_level": escalation, "score": score})

    return {"opened": len(opened), "demands": opened, "warnings": len(warnings)}


def build_trade_demand_crisis_payload(session: Any) -> Optional[Dict[str, Any]]:
    """Active user-team crisis for fullscreen overlay."""
    sync_trade_demand_crises(session)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    book = ensure_trade_demands(session)
    for row in book.values():
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        if str(row.get("team_id") or "") != user_tid:
            continue
        crisis = row.get("crisis")
        if not isinstance(crisis, dict):
            continue
        remaining = float(crisis.get("remaining_seconds") or 0)
        initial = int(crisis.get("initial_seconds") or CRISIS_DEADLINE_MAX)
        stage = int(crisis.get("crisis_stage") or 1)
        dests = list(row.get("preferred_destinations") or [])
        return {
            "demand_id": row.get("demand_id"),
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "team_id": row.get("team_id"),
            "headline": row.get("headline"),
            "body": row.get("body"),
            "primary_complaint": row.get("primary_complaint"),
            "escalation_level": row.get("escalation_level"),
            "trade_stability_score": row.get("trade_stability_score"),
            "remaining_seconds": round(remaining, 1),
            "initial_seconds": initial,
            "crisis_stage": stage,
            "deadline_seconds": initial,
            "value_before": row.get("value_before"),
            "value_after": row.get("value_after"),
            "preferred_destinations": dests,
            "destination_count": len(dests),
            "agent": row.get("agent"),
            "leaked": bool(row.get("leaked")),
            "public_demand": bool(row.get("public_demand")),
            "disruptor": bool(row.get("disruptor")),
        }
    return None


def _enqueue_crisis_expired(session: Any, row: Dict[str, Any]) -> None:
    tid = str(row.get("team_id") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if tid != user_tid:
        return
    notes = list(getattr(session, "notifications", None) or [])
    notes.insert(
        0,
        {
            "id": f"{row.get('demand_id')}:expired",
            "type": "trade_demand_crisis_expired",
            "headline": f"Crisis expired — {row.get('player_name')} is now a distressed asset",
            "body": "You may need to attach assets just to move him.",
            "player_id": row.get("player_id"),
            "team_id": tid,
        },
    )
    session.notifications = notes[:120]


def _enqueue_demand_surfaces(session: Any, row: Dict[str, Any], team: Any) -> None:
    tid = str(row.get("team_id") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    is_user = tid == user_tid
    dests = list(row.get("preferred_destinations") or [])
    dest_label = (
        "All 32 clubs"
        if len(dests) >= 30
        else (", ".join(dests[:8]) + ("…" if len(dests) > 8 else ""))
        if dests
        else "Open market"
    )
    crisis = row.get("crisis") or {}
    popup = {
        "id": str(row.get("demand_id")),
        "kind": "storyline",
        "presentation_type": "trade_demand_crisis" if row.get("formal") else (
            "trade_demand_disruptor" if row.get("disruptor") else "trade_demand"
        ),
        "theme": "danger",
        "source_label": "Trade Demand Crisis",
        "headline": row.get("headline"),
        "body": row.get("body"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_id": tid,
        "cause_type": "TRADE_DEMAND",
        "cause_event_id": row.get("cause_event_id"),
        "trade_demand": {
            "reason_code": row.get("reason_code"),
            "primary_complaint": row.get("primary_complaint"),
            "value_before": row.get("value_before"),
            "value_after": row.get("value_after"),
            "value_delta": row.get("value_delta"),
            "preferred_destinations": dests,
            "destination_label": dest_label,
            "disruptor": bool(row.get("disruptor")),
            "dossier_label": row.get("dossier_label"),
            "agent": row.get("agent"),
            "crisis_stage": row.get("crisis_stage"),
            "remaining_seconds": crisis.get("remaining_seconds"),
            "initial_seconds": crisis.get("initial_seconds"),
            "escalation_level": row.get("escalation_level"),
            "formal_crisis": True,
        },
        "priority": "CRITICAL" if is_user else "HIGH",
    }
    pending = list(getattr(session, "pending_ui_popups", None) or [])
    pending.append(popup)
    session.pending_ui_popups = pending[-80:]

    try:
        from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433

        _record_storyline(
            session,
            {
                "id": row.get("demand_id"),
                "storyline_id": str(row.get("demand_id") or ""),
                "type": "trade_demand",
                "category": "trade",
                "headline": row.get("headline"),
                "summary": row.get("body"),
                "body": row.get("body"),
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name"),
                "team_id": tid,
                "tone": "negative",
                "cause_type": "TRADE_DEMAND",
                "priority": "CRITICAL" if is_user else "HIGH",
                "trade_demand": popup["trade_demand"],
            },
        )
    except Exception:
        events = list(getattr(session, "storyline_events", None) or [])
        events.append(
            {
                "id": row.get("demand_id"),
                "type": "trade_demand",
                "headline": row.get("headline"),
                "body": row.get("body"),
                "player_id": row.get("player_id"),
                "team_id": tid,
                "tone": "negative",
                "cause_type": "TRADE_DEMAND",
                "trade_demand": popup["trade_demand"],
            }
        )
        session.storyline_events = events[-200:]

    notes = list(getattr(session, "notifications", None) or [])
    remaining = crisis.get("remaining_seconds")
    timer_label = f"{int(float(remaining or 0) // 60)}:{int(float(remaining or 0) % 60):02d}" if remaining else "6:00"
    notes.insert(
        0,
        {
            "id": row.get("demand_id"),
            "type": "trade_demand",
            "headline": row.get("headline"),
            "body": f"CRISIS {timer_label} · TV {row.get('value_before')} → {row.get('value_after')} · {dest_label}",
            "team_id": tid,
            "player_id": row.get("player_id"),
        },
    )
    session.notifications = notes[:120]


def clear_demands_on_trade(session: Any, player_ids: List[str]) -> None:
    for pid in player_ids:
        if not pid:
            continue
        clear_trade_demand(session, pid, resolution="traded")
        player = _find_player(session, pid)
        if player is not None:
            clear_demand_temporary_modifiers(player)


# Re-export for storyline trade hub hook
__all__ = [
    "apply_trade_hub_exposure",
    "build_trade_demand_crisis_payload",
    "sync_trade_demand_crises",
    "process_trade_demand_day",
    "open_trade_demand",
    "clear_demands_on_trade",
]
