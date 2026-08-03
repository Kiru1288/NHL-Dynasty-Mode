"""Player-initiated trade demands with before/after trade value snapshots.

Demands surface as caused storyline popups (not fake uncaused rumor spam).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

CANADA_ABBRS = frozenset({"TOR", "MTL", "OTT", "VAN", "CGY", "EDM", "WPG", "SEA"})
SMALL_MARKET_ABBRS = frozenset(
    {"BUF", "OTT", "CBJ", "CGY", "WPG", "ARI", "UTA", "SEA", "NSH", "MIN", "CAR", "FLA"}
)

REASON_COPY = {
    "losing": {
        "headline": "{name} wants out — tired of losing",
        "body": "{name} has asked for a trade after another stretch of losses. The locker room knows.",
    },
    "ice_time": {
        "headline": "{name} demands a trade over ice time",
        "body": "{name} feels buried on the depth chart and wants a fresh start elsewhere.",
    },
    "small_market": {
        "headline": "{name} wants a bigger stage",
        "body": "{name} is unhappy in a small market and has formally requested a trade.",
    },
    "canada": {
        "headline": "{name} wants out of Canada",
        "body": "{name} has asked to be moved to a U.S. club. Destination preferences are attached.",
    },
    "locker_room_disruptor": {
        "headline": "LOCKER-ROOM DISRUPTOR — {name} torpedoes the room",
        "body": (
            "{name} has gone nuclear. Unwilling to be reasoned with, he is trying to force a "
            "move even if it burns bridges and torpedoes his own value."
        ),
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
    """Populate approved_trade_teams when an M-NTC exists without a seeded list."""
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
) -> List[str]:
    seeded = seed_mntc_destinations(player, league, rng=rng)
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
    n = 1
    roll = float(rng.random()) if rng is not None else 0.5
    if roll > 0.82:
        n = min(32, len(candidates))
    elif roll > 0.55:
        n = min(12, len(candidates))
    elif roll > 0.25:
        n = min(5, len(candidates))
    else:
        n = min(2, max(1, len(candidates)))
    if rng is not None and hasattr(rng, "sample") and candidates:
        return list(rng.sample(candidates, min(n, len(candidates))))
    return candidates[:n]


def _snapshot_value(player: Any, team: Any, league: Any) -> float:
    try:
        from app.sim_engine.trades.trade_value import evaluate_player_asset_value

        row = evaluate_player_asset_value(player, team, team, league, context={})
        return float(row.get("trade_value") or 0.0)
    except Exception:
        return max(5.0, _player_ovr(player) * 0.7)


def _apply_demand_value_haircut(player: Any, *, disruptor: bool, severity: float) -> float:
    """Lower systemic mult so Trade Hub / CPU valuation see the haircut."""
    base = float(getattr(player, "_systemic_trade_value_mult", 1.0) or 1.0)
    cut = 0.88 - 0.10 * severity
    if disruptor:
        cut -= 0.18
    mult = max(0.45, min(1.0, base * cut))
    try:
        setattr(player, "_systemic_trade_value_mult", mult)
        setattr(player, "_trade_demand_active", True)
        if disruptor:
            setattr(player, "locker_room_disruptor", True)
            labels = list(getattr(player, "dossier_labels", None) or [])
            if "Locker-room disruptor" not in labels:
                labels.append("Locker-room disruptor")
            setattr(player, "dossier_labels", labels)
    except Exception:
        pass
    return mult


def _pick_reason(player: Any, team: Any, *, rng: Any, ice_time_ok: bool, losing: bool) -> str:
    abbr = _team_abbr(team)
    weights: List[Tuple[str, float]] = []
    if losing:
        weights.append(("losing", 1.4))
    if not ice_time_ok:
        weights.append(("ice_time", 1.5))
    if abbr in SMALL_MARKET_ABBRS:
        weights.append(("small_market", 0.9))
    if abbr in CANADA_ABBRS:
        weights.append(("canada", 0.85))
    # Rare nuclear case — more likely for high ego / low character profiles.
    ego = float(_get(player, "ego", 0.4) or _get(_get(player, "personality", None), "ego", 0.4) or 0.4)
    character = float(
        _get(player, "character", 0.5)
        or _get(_get(player, "personality", None), "character", 0.5)
        or 0.5
    )
    disruptor_w = 0.12 + max(0.0, ego - 0.55) * 0.35 + max(0.0, 0.45 - character) * 0.4
    weights.append(("locker_room_disruptor", disruptor_w))
    if not weights:
        weights = [("losing", 1.0), ("ice_time", 1.0)]
    total = sum(w for _, w in weights) or 1.0
    roll = float(rng.random()) * total
    acc = 0.0
    for code, w in weights:
        acc += w
        if roll <= acc:
            return code
    return weights[-1][0]


def open_trade_demand(
    session: Any,
    player: Any,
    team: Any,
    *,
    reason: str,
    calendar_idx: int,
    iso_date: str = "",
    rng: Any = None,
) -> Dict[str, Any]:
    """Create/overwrite an open demand for a player and return the ledger row."""
    import random

    r = rng if rng is not None else random.Random()
    league = getattr(getattr(session, "sim", None), "league", None)
    pid = _player_id(player)
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    disruptor = reason == "locker_room_disruptor"
    severity = 1.0 if disruptor else (0.75 if reason in ("losing", "ice_time") else 0.55)
    before = _snapshot_value(player, team, league)
    _apply_demand_value_haircut(player, disruptor=disruptor, severity=severity)
    after = _snapshot_value(player, team, league)
    if after >= before:
        after = round(before * (0.72 if disruptor else 0.86), 1)
    dests = _preferred_destinations(player, team, league, reason, rng=r)
    copy = REASON_COPY.get(reason) or REASON_COPY["losing"]
    name = _player_name(player)
    demand_id = f"demand:{pid}:{calendar_idx}:{uuid.uuid4().hex[:6]}"
    row = {
        "demand_id": demand_id,
        "player_id": pid,
        "player_name": name,
        "team_id": tid,
        "status": "open",
        "reason_code": reason,
        "cause_type": "TRADE_DEMAND",
        "cause_event_id": demand_id,
        "opened_day": int(calendar_idx),
        "opened_iso": str(iso_date or ""),
        "deadline_day": int(calendar_idx) + (18 if disruptor else 28),
        "value_before": round(float(before), 1),
        "value_after": round(float(after), 1),
        "value_delta": round(float(after) - float(before), 1),
        "preferred_destinations": dests,
        "destination_count": len(dests),
        "disruptor": disruptor,
        "dossier_label": "Locker-room disruptor" if disruptor else "Trade demand",
        "headline": copy["headline"].format(name=name),
        "body": copy["body"].format(name=name),
        "ovr": round(_player_ovr(player)),
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
    # Keep a short history but drop active flag lookup by overwriting status.


def _team_is_losing(session: Any, team: Any) -> bool:
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    standings = getattr(session, "standings", None)
    try:
        rec = getattr(standings, "records", None) or {}
        row = rec.get(tid)
        if row is None:
            return False
        wins = int(_get(row, "wins", 0) or 0)
        losses = int(_get(row, "losses", 0) or 0)
        otl = int(_get(row, "ot_losses", 0) or _get(row, "otl", 0) or 0)
        gp = wins + losses + otl
        if gp < 10:
            return False
        pts = wins * 2 + otl
        return (pts / max(1, gp * 2)) < 0.42
    except Exception:
        return False


def _ice_time_unhappy(player: Any) -> bool:
    ovr = _player_ovr(player)
    toi = float(_get(player, "avg_toi", 0) or _get(player, "toi_per_game", 0) or 0)
    line = str(_get(player, "line", "") or _get(player, "role", "") or "").lower()
    scratched = bool(_get(player, "scratched", False) or _get(player, "healthy_scratch", False))
    if scratched and ovr >= 78:
        return True
    if ovr >= 84 and toi and toi < 15.5:
        return True
    if ovr >= 80 and ("4" in line or "scratch" in line):
        return True
    return False


def process_trade_demand_day(session: Any, calendar_idx: int, day_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Scan rosters, open a few demands, enqueue user popups."""
    import random

    rng = getattr(getattr(session, "sim", None), "rng", None) or random.Random(int(calendar_idx) + 17)
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"opened": 0}
    book = ensure_trade_demands(session)
    # Expire stale
    for pid, row in list(book.items()):
        if not isinstance(row, dict):
            continue
        if str(row.get("status")) != "open":
            continue
        if int(row.get("deadline_day") or 0) and int(calendar_idx) > int(row.get("deadline_day") or 0):
            row["status"] = "lapsed"
            row["resolved"] = True

    teams = list(_get(league, "teams", None) or [])
    user_tid = str(getattr(session, "user_team_id", "") or "")
    opened: List[Dict[str, Any]] = []
    iso = ""
    try:
        iso = str((day_meta or {}).get("iso") or (day_meta or {}).get("date") or "")
    except Exception:
        iso = ""

    # Soft daily caps — enough to feel alive, not spam.
    max_open_today = 2 if rng.random() < 0.55 else 1
    if rng.random() < 0.12:
        max_open_today = 3

    open_count = sum(1 for r in book.values() if isinstance(r, dict) and r.get("status") == "open")
    if open_count >= 18:
        max_open_today = min(max_open_today, 1)

    candidates: List[Tuple[float, Any, Any]] = []
    for team in teams:
        roster = list(_get(team, "roster", None) or [])
        losing = _team_is_losing(session, team)
        for p in roster:
            if _get(p, "retired", False):
                continue
            pid = _player_id(p)
            if not pid:
                continue
            if active_demand_for_player(session, pid):
                continue
            ovr = _player_ovr(p)
            if ovr < 74:
                continue
            ice_bad = _ice_time_unhappy(p)
            intent = 0.08
            if losing:
                intent += 0.16
            if ice_bad:
                intent += 0.18
            if ovr >= 82:
                intent += 0.05
            # Personality hook when available
            try:
                from app.sim_engine.ai.personality import PersonalityBehavior, BehaviorContext

                pers = getattr(p, "personality", None) or getattr(p, "behavior", None)
                if pers is not None and hasattr(PersonalityBehavior, "sample_trade_request_intent"):
                    # Best-effort; many players may only have floats.
                    pass
            except Exception:
                pass
            if rng.random() > min(0.55, intent):
                continue
            score = intent + (ovr - 74) * 0.01 + (0.08 if losing else 0.0)
            candidates.append((score, p, team))

    candidates.sort(key=lambda x: -x[0])
    for score, player, team in candidates[: max_open_today * 4]:
        if len(opened) >= max_open_today:
            break
        ice_bad = _ice_time_unhappy(player)
        losing = _team_is_losing(session, team)
        reason = _pick_reason(player, team, rng=rng, ice_time_ok=not ice_bad, losing=losing)
        row = open_trade_demand(
            session,
            player,
            team,
            reason=reason,
            calendar_idx=int(calendar_idx),
            iso_date=iso,
            rng=rng,
        )
        opened.append(row)
        # Surfaces already enqueued inside open_trade_demand.

    return {"opened": len(opened), "demands": opened}


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
    popup = {
        "id": str(row.get("demand_id")),
        "kind": "storyline",
        "presentation_type": "trade_demand_disruptor" if row.get("disruptor") else "trade_demand",
        "theme": "danger" if row.get("disruptor") else "warn",
        "source_label": "Player Trade Demand",
        "headline": row.get("headline"),
        "body": row.get("body"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_id": tid,
        "cause_type": "TRADE_DEMAND",
        "cause_event_id": row.get("cause_event_id"),
        "trade_demand": {
            "reason_code": row.get("reason_code"),
            "value_before": row.get("value_before"),
            "value_after": row.get("value_after"),
            "value_delta": row.get("value_delta"),
            "preferred_destinations": dests,
            "destination_label": dest_label,
            "disruptor": bool(row.get("disruptor")),
            "dossier_label": row.get("dossier_label"),
        },
        "priority": "HIGH" if (is_user or row.get("disruptor")) else "MID",
    }
    pending = list(getattr(session, "pending_ui_popups", None) or [])
    # Cap: keep room for other storylines; user/disruptor always enqueue.
    story_like = [
        p
        for p in pending
        if str(p.get("kind") or "") in ("storyline", "legal_trouble", "trade_demand")
    ]
    if is_user or row.get("disruptor") or len(story_like) < 3:
        pending.append(popup)
        session.pending_ui_popups = pending[-80:]

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
    notes.insert(
        0,
        {
            "id": row.get("demand_id"),
            "type": "trade_demand",
            "headline": row.get("headline"),
            "body": f"TV {row.get('value_before')} → {row.get('value_after')} · Destinations: {dest_label}",
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
