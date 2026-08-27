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
    apply_daily_stability_update,
    apply_trade_hub_exposure,
    clear_demand_temporary_modifiers,
    crisis_distressed_asset_cost,
    crisis_stage_from_remaining,
    crisis_trade_value_multiplier,
    ensure_player_storyline_state,
    ensure_trade_stability_state,
    formal_demand_eligible,
    gather_player_concerns,
    primary_complaint_from_pressures,
    stability_to_escalation_level,
    update_player_stability,
)

CANADA_ABBRS = frozenset({"TOR", "MTL", "OTT", "VAN", "CGY", "EDM", "WPG", "SEA"})
SMALL_MARKET_ABBRS = frozenset(
    {"BUF", "OTT", "CBJ", "CGY", "WPG", "ARI", "UTA", "SEA", "NSH", "MIN", "CAR", "FLA"}
)


def _calendar_day_meta(session: Any) -> Dict[str, Any]:
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    cal = getattr(session, "nhl_calendar", None) or []
    if 0 <= cur < len(cal) and isinstance(cal[cur], dict):
        return dict(cal[cur])
    return {}


def get_trade_deadline_context(session: Any) -> Dict[str, Any]:
    """NHL trade deadline window (Feb 25 – Mar 10) and post-deadline rules."""
    phase = str(getattr(session, "phase", "") or getattr(session, "season_phase", "") or "").lower()
    day = _calendar_day_meta(session)
    iso = str(day.get("iso") or day.get("date") or "")
    tags = tuple(day.get("tags") or ())
    in_window = "trade_deadline" in tags
    past = False
    deadline_iso = ""

    if phase in ("playoffs", "offseason", "post_cup", "preseason"):
        past = True
    elif iso and len(iso) >= 10:
        try:
            y, m, d = int(iso[0:4]), int(iso[5:7]), int(iso[8:10])
            deadline_iso = f"{y}-03-10"
            if (m == 3 and d > 10) or m >= 4:
                past = True
            elif m == 2 and d >= 25:
                in_window = True
            elif m == 3 and d <= 10:
                in_window = True
        except (TypeError, ValueError):
            pass

    days_to_deadline = None
    if iso and deadline_iso and not past:
        try:
            from datetime import date

            cur_d = date(int(iso[0:4]), int(iso[5:7]), int(iso[8:10]))
            dl = date(int(deadline_iso[0:4]), int(deadline_iso[5:7]), int(deadline_iso[8:10]))
            days_to_deadline = max(0, (dl - cur_d).days)
        except (TypeError, ValueError):
            days_to_deadline = None

    return {
        "in_window": bool(in_window),
        "past_deadline": bool(past),
        "deadline_iso": deadline_iso or "03-10",
        "days_to_deadline": days_to_deadline,
        "crisis_timer_ticks": not past,
        "new_demands_allowed": not past and phase in ("", "regular", "regular_season", "in_season"),
        "demand_urgency_mult": 1.18 if in_window and not past else 1.0,
    }


def _close_crises_for_trade_deadline(session: Any) -> int:
    """After Mar 10 — freeze/close open demands; crisis timer cannot expire post-deadline."""
    ctx = get_trade_deadline_context(session)
    if not ctx.get("past_deadline"):
        return 0
    book = ensure_trade_demands(session)
    closed = 0
    now = time.time()
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        crisis = row.get("crisis")
        if isinstance(crisis, dict):
            crisis["remaining_seconds"] = float(crisis.get("remaining_seconds") or 0)
            crisis["timer_frozen"] = True
            crisis["frozen_reason"] = "trade_deadline_passed"
            crisis["last_sync_unix"] = now
            row["crisis"] = crisis
        row["status"] = "deadline_closed"
        row["resolved"] = False
        row["deadline_closed"] = True
        row["resolution"] = "trade_deadline"
        player = _find_player(session, pid)
        if player is not None:
            clear_demand_temporary_modifiers(player)
        closed += 1
        user_tid = str(getattr(session, "user_team_id", "") or "")
        if str(row.get("team_id") or "") == user_tid:
            notes = list(getattr(session, "notifications", None) or [])
            notes.insert(
                0,
                {
                    "id": f"{row.get('demand_id')}:deadline",
                    "type": "trade_demand_deadline_closed",
                    "headline": f"Trade deadline passed — {row.get('player_name')} demand frozen",
                    "body": (
                        "The NHL trade deadline has passed. The crisis timer is paused until the "
                        "offseason. You may revisit this player in the summer trade market."
                    ),
                    "team_id": user_tid,
                    "player_id": pid,
                },
            )
            session.notifications = notes[:120]
    return closed

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


def _apply_crisis_value_state(
    player: Any,
    *,
    crisis_stage: int,
    base_before: float,
    timer_expired: bool = False,
) -> Tuple[float, float]:
    mult = crisis_trade_value_multiplier(crisis_stage)
    distressed = crisis_distressed_asset_cost(base_before, timer_expired=timer_expired)
    effective_mult = mult
    if timer_expired:
        effective_mult = min(effective_mult, 0.12)
    try:
        setattr(player, "_trade_demand_active", True)
        setattr(player, "_crisis_trade_value_mult", mult)
        setattr(player, "_crisis_distressed_asset", distressed)
        setattr(player, "_crisis_trade_stage", 4 if timer_expired else crisis_stage)
        setattr(player, "_systemic_trade_value_mult", mult)
        if crisis_stage >= 3 and timer_expired:
            pst = ensure_player_storyline_state(player)
            if int(pst.get("character") or 80) < 65:
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


def sync_trade_demand_crises(
    session: Any,
    *,
    elapsed_hint: Optional[float] = None,
    tick_timers: bool = True,
) -> None:
    """Decrement active crisis timers only while the client session is actively ticking."""
    _close_crises_for_trade_deadline(session)
    deadline_ctx = get_trade_deadline_context(session)
    if not deadline_ctx.get("crisis_timer_ticks"):
        tick_timers = False

    book = ensure_trade_demands(session)
    now = time.time()
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        crisis = row.get("crisis")
        if not isinstance(crisis, dict):
            continue
        last = float(crisis.get("last_sync_unix") or now)
        if tick_timers:
            elapsed = float(elapsed_hint if elapsed_hint is not None else max(0.0, now - last))
            remaining = max(0.0, float(crisis.get("remaining_seconds") or 0) - elapsed)
        else:
            elapsed = 0.0
            remaining = max(0.0, float(crisis.get("remaining_seconds") or 0))
        crisis["remaining_seconds"] = round(remaining, 2)
        crisis["last_sync_unix"] = now
        crisis["timer_active"] = bool(tick_timers)
        initial = int(crisis.get("initial_seconds") or CRISIS_DEADLINE_MAX)
        expired = remaining <= 0
        stage = 4 if expired else crisis_stage_from_remaining(initial, int(round(remaining)))
        prev_stage = int(crisis.get("crisis_stage") or 1)
        crisis["crisis_stage"] = stage
        row["crisis_stage"] = stage

        player = _find_player(session, pid)
        if player is not None:
            team = _find_team_for_player(session, pid)
            league = getattr(getattr(session, "sim", None), "league", None)
            base_before = float(row.get("value_before") or _snapshot_value(player, team, league, crisis_stage=1))
            _, after = _apply_crisis_value_state(
                player,
                crisis_stage=stage,
                base_before=base_before,
                timer_expired=expired,
            )
            row["value_after"] = round(after, 1)
            row["value_delta"] = round(after - base_before, 1)

            if stage > prev_stage:
                _apply_crisis_stage_effects(session, row, player, team, league, stage)
                row["ntc_waiver_snapshot"] = _build_ntc_waiver_snapshot(session, row, player, team, league)

        if remaining <= 0 and str(row.get("status")) == "open" and tick_timers and deadline_ctx.get("crisis_timer_ticks"):
            row["status"] = "crisis_expired"
            row["resolved"] = False
            row["crisis_expired"] = True
            _enqueue_crisis_expired(session, row)


def _clause_label(player: Any) -> str:
    contract = _get(player, "contract", None)
    if contract is None:
        return "None"
    clause = str(
        _get(contract, "clause", "")
        or _get(contract, "clause_type", "")
        or _get(contract, "trade_clause", "")
        or ""
    ).upper()
    if "NMC" in clause or "NO MOVE" in clause:
        return "NMC"
    if "MNTC" in clause or "M-NTC" in clause or "MODIFIED" in clause:
        return "M-NTC"
    if "NTC" in clause:
        return "NTC"
    return "None"


def evaluate_trade_demand_ntc_waiver(
    session: Any,
    player: Any,
    source_team: Any,
    destination_team: Any,
    *,
    demand_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """NTC/M-NTC willingness during an active trade-demand crisis."""
    from app.sim_engine.franchise.trade_stability_engine import _player_character_0_100, _player_mental_0_100
    from app.sim_engine.trades.trade_rules import evaluate_ntc_waiver_request

    ctx = {
        "calendar_cursor": int(getattr(session, "calendar_cursor", 0) or 0),
        "season_year": int(getattr(session, "season_calendar_year", 2025) or 2025),
        "trade_demand_crisis": True,
    }
    base = evaluate_ntc_waiver_request(
        player,
        source_team=source_team,
        destination_team=destination_team,
        context=ctx,
    )
    if not base.get("can_request"):
        return dict(base)

    character = float(_player_character_0_100(player))
    mental = float(_player_mental_0_100(player))
    row = demand_row or {}
    crisis_stage = int(row.get("crisis_stage") or 1)
    agent = ensure_player_agent(player, session)

    chance = float(base.get("accept_chance") or 0.38)
    # Wants out — more willing to waive for a fresh start
    if row.get("status") == "open" or row.get("formal"):
        chance += 0.10 + crisis_stage * 0.04
    if character >= 85:
        chance += 0.12
    elif character >= 77:
        chance += 0.06
    elif character < 65:
        chance -= 0.14
    elif character < 58:
        chance -= 0.22
    if mental >= 88:
        chance += 0.04
    elif mental < 62:
        chance -= 0.06
    if str(agent.get("style") or "") == "disruptor" and crisis_stage >= 2 and character < 70:
        chance -= 0.18
    if str(agent.get("style") or "") == "discreet" and character >= 80:
        chance += 0.08

    chance = max(0.04, min(0.92, chance))
    roll = float(base.get("roll") or 0.5)
    accepted = roll < chance
    out = dict(base)
    out["accept_chance"] = round(chance, 3)
    out["accepted"] = bool(accepted)
    out["crisis_stage"] = crisis_stage
    out["character"] = int(character)
    out["mental"] = int(mental)
    out["agent_style"] = agent.get("style")
    return out


def _build_ntc_waiver_snapshot(
    session: Any,
    demand_row: Dict[str, Any],
    player: Any,
    source_team: Any,
    league: Any,
) -> Dict[str, Any]:
    """Sample waiver outcomes for approved/blocked destinations at current crisis stage."""
    if league is None or source_team is None:
        return {}
    clause = _clause_label(player)
    if clause == "None":
        return {"clause": clause, "waivers_required": False}

    approved = list(demand_row.get("preferred_destinations") or [])
    teams = list(_get(league, "teams", None) or [])
    team_by_abbr = {}
    for tm in teams:
        ab = _team_abbr(tm)
        if ab:
            team_by_abbr[ab] = tm

    blocked_abbrs = [ab for ab in team_by_abbr if ab not in approved and ab != _team_abbr(source_team)][:4]
    samples = []
    for label, abbrs in (("approved", approved[:3]), ("blocked", blocked_abbrs[:3])):
        for ab in abbrs:
            dest = team_by_abbr.get(ab)
            if dest is None:
                continue
            result = evaluate_trade_demand_ntc_waiver(
                session,
                player,
                source_team,
                dest,
                demand_row=demand_row,
            )
            samples.append(
                {
                    "bucket": label,
                    "destination": ab,
                    "accepted": bool(result.get("accepted")),
                    "accept_chance": result.get("accept_chance"),
                    "reason_code": result.get("reason_code"),
                    "reason": result.get("reason"),
                    "clause_label": result.get("clause_label") or clause,
                }
            )
    return {
        "clause": clause,
        "waivers_required": clause in ("NTC", "M-NTC"),
        "crisis_stage": int(demand_row.get("crisis_stage") or 1),
        "samples": samples,
    }


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

    deadline_ctx = get_trade_deadline_context(session)
    if not deadline_ctx.get("new_demands_allowed"):
        return {
            "status": "blocked",
            "escalation_level": escalation,
            "player_id": pid,
            "reason": "trade_deadline_passed",
        }
    if not force_formal and not formal_demand_eligible(stability_row):
        return {"status": "blocked", "escalation_level": escalation, "player_id": pid, "reason": "insufficient_signals"}

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
    row["ntc_waiver_snapshot"] = _build_ntc_waiver_snapshot(session, row, player, team, league)
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
    _close_crises_for_trade_deadline(session)
    sync_trade_demand_crises(session, tick_timers=False)
    deadline_ctx = get_trade_deadline_context(session)
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

            stability_row = apply_daily_stability_update(session, player, team, int(calendar_idx))
            escalation = int(stability_row.get("escalation_level") or 0)
            score = float(stability_row.get("trade_stability_score") or 100.0)

            if escalation in (1, 2):
                _maybe_enqueue_stability_warning(session, player, team, stability_row, calendar_idx, iso, rng)
            elif escalation == 0:
                _maybe_enqueue_stability_concern_hint(
                    session, player, team, stability_row, calendar_idx, iso, rng
                )

            if (
                escalation >= 3
                and formal_demand_eligible(stability_row)
                and deadline_ctx.get("new_demands_allowed")
            ):
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
                    force_formal=False,
                )
                if row.get("status") == "open":
                    opened.append(row)
            elif escalation >= 1 and score < 68:
                warnings.append({"player_id": pid, "escalation_level": escalation, "score": score})

    return {"opened": len(opened), "demands": opened, "warnings": len(warnings)}


def build_trade_demand_crisis_payload(session: Any, *, tick_timers: bool = False) -> Optional[Dict[str, Any]]:
    """Active user-team crisis for fullscreen overlay."""
    sync_trade_demand_crises(session, tick_timers=tick_timers)
    deadline_ctx = get_trade_deadline_context(session)
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
            "timer_active": bool(crisis.get("timer_active")),
            "timer_frozen": bool(crisis.get("timer_frozen")) or not deadline_ctx.get("crisis_timer_ticks"),
            "trade_deadline": deadline_ctx,
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


def _maybe_enqueue_stability_warning(
    session: Any,
    player: Any,
    team: Any,
    stability_row: Dict[str, Any],
    calendar_idx: int,
    iso_date: str,
    rng: Any,
) -> None:
    """Surface L1–L2 angst/apathy to user + league notifications."""
    pid = _player_id(player)
    pst = ensure_player_storyline_state(player)
    warn_key = f"stability_warn:{pid}:{int(stability_row.get('escalation_level') or 0)}"
    if warn_key in set(pst.get("stability_warnings_sent") or []):
        return
    sent = list(pst.get("stability_warnings_sent") or [])
    sent.append(warn_key)
    pst["stability_warnings_sent"] = sent[-12:]

    level = int(stability_row.get("escalation_level") or 0)
    labels = {1: "Growing frustration", 2: "Disconnecting from organization"}
    name = _player_name(player)
    agent = agent_public_view(player, session)
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    is_user = tid == user_tid

    labels_dossier = list(getattr(player, "dossier_labels", None) or [])
    tag = labels.get(level, "Trade concern")
    if tag not in labels_dossier:
        labels_dossier.append(tag)
        try:
            setattr(player, "dossier_labels", labels_dossier)
        except Exception:
            pass

    popup = {
        "id": warn_key,
        "kind": "storyline",
        "presentation_type": "trade_stability_warning",
        "theme": "warn" if level == 1 else "danger",
        "source_label": f"Agent — {agent.get('name', 'Representative')}",
        "headline": f"{name} — {tag}",
        "body": (
            f"{agent.get('name', 'The player\'s agent')} has reached out regarding {name}'s "
            f"satisfaction (stability {stability_row.get('trade_stability_score')}). "
            f"No formal trade demand yet."
        ),
        "player_id": pid,
        "player_name": name,
        "team_id": tid,
        "trade_stability": {
            "score": stability_row.get("trade_stability_score"),
            "escalation_level": level,
            "agent": agent,
            "primary_complaint": primary_complaint_from_pressures(dict(stability_row.get("pressures") or {})),
        },
        "priority": "HIGH" if is_user else "MID",
    }
    pending = list(getattr(session, "pending_ui_popups", None) or [])
    pending.append(popup)
    session.pending_ui_popups = pending[-80:]

    notes = list(getattr(session, "notifications", None) or [])
    notes.insert(
        0,
        {
            "id": warn_key,
            "type": "trade_stability_warning",
            "headline": popup["headline"],
            "body": popup["body"],
            "team_id": tid,
            "player_id": pid,
        },
    )
    session.notifications = notes[:120]

    if is_user:
        roster_flags = dict(getattr(session, "trade_stability_roster_flags", None) or {})
        roster_flags[pid] = {
            "player_id": pid,
            "player_name": name,
            "escalation_level": level,
            "score": float(stability_row.get("trade_stability_score") or 0),
            "label": tag,
            "top_pressure": max(
                (stability_row.get("pressures") or {}).items(),
                key=lambda kv: kv[1],
                default=("role", 0),
            )[0],
        }
        session.trade_stability_roster_flags = roster_flags


def _maybe_enqueue_stability_concern_hint(
    session: Any,
    player: Any,
    team: Any,
    stability_row: Dict[str, Any],
    calendar_idx: int,
    iso_date: str,
    rng: Any,
) -> None:
    """Early 'something feels off' signals for non-volatile players (L0, score slipping)."""
    from app.sim_engine.franchise.trade_stability_engine import count_significant_pressures

    pid = _player_id(player)
    character = int(stability_row.get("character") or 74)
    if character < 58:
        return

    score = float(stability_row.get("trade_stability_score") or 100.0)
    if score >= 72 or score <= 45:
        return

    pressures = dict(stability_row.get("pressures") or {})
    if count_significant_pressures(pressures) < 1:
        return

    pst = ensure_player_storyline_state(player)
    warn_key = f"stability_hint:{pid}:{int(calendar_idx // 7)}"
    if warn_key in set(pst.get("stability_warnings_sent") or []):
        return
    sent = list(pst.get("stability_warnings_sent") or [])
    sent.append(warn_key)
    pst["stability_warnings_sent"] = sent[-16:]

    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if tid != user_tid:
        return

    name = _player_name(player)
    agent = agent_public_view(player, session)
    top = max(pressures.items(), key=lambda kv: kv[1])[0] if pressures else "role"
    complaint = primary_complaint_from_pressures(pressures)

    labels_dossier = list(getattr(player, "dossier_labels", None) or [])
    tag = "Monitor — early concern"
    if tag not in labels_dossier:
        labels_dossier.append(tag)
        try:
            setattr(player, "dossier_labels", labels_dossier[-8:])
        except Exception:
            pass

    popup = {
        "id": warn_key,
        "kind": "storyline",
        "presentation_type": "trade_stability_hint",
        "theme": "info",
        "source_label": f"Agent — {agent.get('name', 'Representative')}",
        "headline": f"{name} — agent checking in",
        "body": (
            f"{agent.get('name', 'The agent')} called about {name}'s {complaint}. "
            f"No formal demand — but the situation bears watching (stability {score:.0f})."
        ),
        "player_id": pid,
        "player_name": name,
        "team_id": tid,
        "trade_stability": {
            "score": score,
            "escalation_level": 0,
            "top_pressure": top,
            "agent": agent,
        },
        "priority": "MID",
    }
    pending = list(getattr(session, "pending_ui_popups", None) or [])
    pending.append(popup)
    session.pending_ui_popups = pending[-80:]

    roster_flags = dict(getattr(session, "trade_stability_roster_flags", None) or {})
    roster_flags[pid] = {
        "player_id": pid,
        "player_name": name,
        "escalation_level": 0,
        "score": score,
        "label": tag,
        "top_pressure": top,
    }
    session.trade_stability_roster_flags = roster_flags


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
