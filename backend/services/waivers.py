"""In-season NHL waivers: placement, 24-hour claim window, claims by priority, clearance.

Flow (mirrors the real NHL):
- Sending a non-exempt player to the AHL puts him on waivers for one day. He sits on
  the wire (off both rosters) until the window closes.
- When the window closes, clubs get first crack in reverse standings order (worst
  record first). The user can put in a claim from the waiver popup / wire.
- Claimed → he joins the claiming club's NHL roster (overflow is assigned to the AHL).
  Cleared → he reports to his original club's AHL affiliate.
- CPU clubs also run players through waivers to make roster moves, so the wire is live.
"""

from __future__ import annotations

import random
from typing import Any, Dict
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

WAIVER_WINDOW_DAYS = 1
POPUP_MIN_OVR = 73  # league-wide waiver notices for players worth a look
# ~0.13 placement attempts per day league-wide before the roster-gap filters.
# Real NHL wires are a handful of names a week, not a daily dump.
CPU_WAIVE_DAILY_CHANCE = 0.004


def _tid(team: Any) -> str:
    return str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")


def _league(session: Any) -> Any:
    return getattr(getattr(session, "sim", None), "league", None)


def _cursor(session: Any) -> int:
    return int(getattr(session, "calendar_cursor", 0) or 0)


def _in_season(session: Any) -> bool:
    phase = str(getattr(session, "phase", "") or "").lower()
    return phase not in ("offseason", "post_cup")


def _team_name(team: Any) -> str:
    return str(getattr(team, "name", None) or getattr(team, "abbreviation", None) or _tid(team))


def _popup(session: Any, key: str, payload: Dict[str, Any]) -> None:
    try:
        from services.franchise_sim import _append_showcase_popup

        _append_showcase_popup(session, key, payload)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _standings_points(session: Any, team: Any) -> float:
    """Points percentage. Lower is worse, and worse clubs claim first."""
    key = _waiver_priority_key(session, team)
    return float(key[0])


def _waiver_priority_key(session: Any, team: Any):
    """NHL waiver order: worst points percentage, then fewer points, then fewer wins.

    A team that has not played sorts with the other winless clubs. Team id breaks
    the remaining ties so the order does not depend on dict iteration.
    """
    points = wins = gp = 0
    st = getattr(session, "standings", None)
    recs = getattr(st, "records", None) or {}
    rec = recs.get(_tid(team)) if isinstance(recs, dict) else None
    if rec is not None:
        try:
            wins = int(getattr(rec, "wins", 0) or 0)
            losses = int(getattr(rec, "losses", 0) or 0)
            otl = int(getattr(rec, "otl", 0) or 0)
            gp = wins + losses + otl
            points = int(getattr(rec, "points", 0) or 0)
        except Exception:
            points = wins = gp = 0
    pct = (float(points) / float(gp)) if gp > 0 else 0.0
    return (pct, points, wins, _tid(team))


def _portrait_fields(player: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Headshot, overall, and potential for waiver alerts and the story wire."""
    overall = entry.get("overall")
    fields: Dict[str, Any] = {
        "player_name": entry.get("name"),
        "player_id": entry.get("player_id"),
        "player_position": entry.get("position"),
        "position": entry.get("position"),
        "player_overall": overall,
        "overall": overall,
        "team_name": entry.get("original_team_name"),
        "team_id": entry.get("original_team_id"),
    }
    pot = entry.get("potential")
    if player is not None:
        try:
            from app.sim_engine.progression.potential import read_player_potential99

            pot = int(round(float(read_player_potential99(player) or 0)))
        except Exception:
            pot = pot or 0
        try:
            from app.sim_engine.generation.player_headshots import merge_headshot_into_row

            fields = merge_headshot_into_row(fields, player)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        abbr = str(getattr(player, "team_abbrev", None) or getattr(player, "team_abbreviation", None) or "")
        if abbr:
            fields["team_abbrev"] = abbr
    if pot:
        fields["player_potential"] = int(pot)
        fields["potential"] = int(pot)
    return fields


def place_on_waivers(session: Any, team: Any, player: Any, *, reason: str = "manual", manual: bool = True) -> Dict[str, Any]:
    """Put a player on the wire for the 24-hour claim window."""
    from services.contract_economy import (
        _ensure_waiver_wire,
        _player_id,
        _player_name,
        _player_ovr,
        _position_bucket,
        _contract_years_remaining,
        _append_waiver_history,
        has_nmc,
        player_cap_hit_millions,
        sync_team_cap_fields,
    )

    league = _league(session)
    if league is None:
        return {"ok": False, "reason": "League not ready"}
    if has_nmc(player):
        return {"ok": False, "reason": "No-movement clause — he can't be placed on waivers without consent"}
    pid = _player_id(player)
    wire = _ensure_waiver_wire(league)
    for e in wire:
        if str(e.get("player_id")) == pid and not e.get("cleared") and not e.get("claimed_by"):
            return {"ok": False, "reason": "Already on waivers"}
    now = _cursor(session)
    entry = {
        "player_id": pid,
        "name": _player_name(player),
        "position": _position_bucket(player),
        "overall": round(_player_ovr(player)),
        "cap_hit_m": round(player_cap_hit_millions(player), 3),
        "years_remaining": _contract_years_remaining(player),
        "original_team_id": _tid(team),
        "original_team_name": _team_name(team),
        "reason": reason,
        "waiver_status": "on_waivers",
        "placed_cursor": now,
        "expires_cursor": now + WAIVER_WINDOW_DAYS,
        "claimed_by": None,
        "cleared": False,
        "user_claim": False,
        "player_ref": player,
    }
    try:
        from app.sim_engine.progression.potential import read_player_potential99

        entry["potential"] = int(round(float(read_player_potential99(player) or 0)))
    except Exception:
        entry["potential"] = int(entry["overall"] or 0)
    abbr = str(getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or "")
    if abbr:
        entry["team_abbrev"] = abbr
    wire.append(entry)
    _append_waiver_history(league, {k: v for k, v in entry.items() if k != "player_ref"})
    team.roster = [p for p in list(getattr(team, "roster", None) or []) if p is not player]
    team.ahl_roster = [p for p in list(getattr(team, "ahl_roster", None) or []) if p is not player]
    try:
        player.waiver_status = "on_waivers"
        player.on_waiver_wire = True
        player.active_roster = False
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    sync_team_cap_fields(team, league)

    user_tid = str(getattr(session, "user_team_id", "") or "")
    portrait = _portrait_fields(player, entry)
    # A user send-down resolves on the spot: claimed or cleared. CPU placements
    # stay on the 24-hour wire so the user can still put in a claim.
    if _tid(team) == user_tid and manual:
        _resolve_entry(session, entry)
    elif entry["overall"] >= POPUP_MIN_OVR and _tid(team) != user_tid:
        _popup(session, f"waiver_avail:{pid}:{now}", {
            "kind": "waiver",
            "source_label": "Waiver Wire",
            "headline": f"{_team_name(team)} placed {entry['name']} on waivers",
            "summary": (
                f"{entry['position']} · {entry['overall']} OVR"
                + (f" · {entry['potential']} POT" if entry.get("potential") else "")
                + f" · ${entry['cap_hit_m']:.2f}M × {entry['years_remaining']} yr. "
                "Claims close in 24 hours — the club lowest in the standings has first dibs."
            ),
            "theme": "info",
            "actions": [{"id": "waiver_claim", "label": "Put in a claim", "primary": True, "player_id": pid}],
            **portrait,
        })
    return {"ok": True, "waiver_entry": {k: v for k, v in entry.items() if k != "player_ref"}}


def set_user_claim(session: Any, player_id: str, claim: bool = True) -> Dict[str, Any]:
    from services.contract_economy import _ensure_waiver_wire

    league = _league(session)
    if league is None:
        return {"ok": False, "reason": "League not ready"}
    user_tid = str(getattr(session, "user_team_id", "") or "")
    for e in _ensure_waiver_wire(league):
        if str(e.get("player_id")) == str(player_id) and not e.get("cleared") and not e.get("claimed_by"):
            if str(e.get("original_team_id")) == user_tid:
                return {"ok": False, "reason": "You can't claim your own player"}
            e["user_claim"] = bool(claim)
            return {"ok": True, "claimed": bool(claim), "entry": {k: v for k, v in e.items() if k != "player_ref"}}
    return {"ok": False, "reason": "That player is no longer on waivers"}


def waiver_wire_payload(session: Any) -> Dict[str, Any]:
    from services.contract_economy import _ensure_waiver_wire

    league = _league(session)
    rows = []
    if league is not None:
        for e in _ensure_waiver_wire(league):
            if e.get("cleared") or e.get("claimed_by"):
                continue
            if e.get("expires_cursor") is None:
                continue
            rows.append({k: v for k, v in e.items() if k != "player_ref"})
    rows.sort(key=lambda r: -int(r.get("overall") or 0))
    return {"ok": True, "wire": rows, "today": _cursor(session)}


def _claim_room(team: Any, league: Any, entry: Dict[str, Any]) -> bool:
    from services.contract_economy import get_team_cap_snapshot_full, validate_contract_slots

    try:
        snap = get_team_cap_snapshot_full(team, league)
        if float(snap.get("usable_cap_space_m") or 0.0) < float(entry.get("cap_hit_m") or 0.0) - 0.001:
            return False
    except Exception:
        return False
    try:
        if not validate_contract_slots(team, league, additional=1).get("ok"):
            return False
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return True


def _resolve_entry(session: Any, entry: Dict[str, Any]) -> None:
    from services.contract_economy import (
        _transfer_waiver_player,
        evaluate_team_position_needs,
        score_waiver_claim_fit,
        sync_team_cap_fields,
    )

    league = _league(session)
    player = entry.get("player_ref")
    teams = {_tid(t): t for t in list(getattr(league, "teams", None) or [])}
    orig = teams.get(str(entry.get("original_team_id")))
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if player is None:
        entry["cleared"] = True
        return

    order = sorted(
        [t for tid, t in teams.items() if tid != str(entry.get("original_team_id"))],
        key=lambda t: _waiver_priority_key(session, t),
    )
    winner = None
    for team in order:
        tid = _tid(team)
        if tid == user_tid:
            if entry.get("user_claim") and _claim_room(team, league, entry):
                winner = team
                break
            continue
        try:
            ctx = evaluate_team_position_needs(team, league, getattr(session, "sim", None))
            if int(ctx.get("slots_remaining") or 0) <= 0 or not _claim_room(team, league, entry):
                continue
            if score_waiver_claim_fit(team, player, ctx, league) >= 0.42:
                winner = team
                break
        except Exception:
            continue

    name = entry.get("name") or "Player"
    portrait = _portrait_fields(player, entry)
    if winner is not None:
        _transfer_waiver_player(player, None, winner, league)
        entry["claimed_by"] = _tid(winner)
        entry["waiver_status"] = "claimed"
        try:
            from app.sim_engine.trades.roster_balance import auto_send_down_overflow

            auto_send_down_overflow(winner, protect_ids=[str(entry.get("player_id"))])
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        sync_team_cap_fields(winner, league)
        if _tid(winner) == user_tid:
            _popup(session, f"waiver_won:{entry['player_id']}", {
                "kind": "breaking_news", "source_label": "Waiver Wire", "theme": "positive",
                "headline": f"Claim awarded: {name} joins your club",
                "summary": f"You won the waiver claim on {name} from {entry.get('original_team_name')}. He's on your NHL roster.",
                "actions": [{"id": "roster", "label": "Open roster", "primary": True}],
                **portrait,
            })
        elif str(entry.get("original_team_id")) == user_tid:
            _popup(session, f"waiver_lost:{entry['player_id']}", {
                "kind": "breaking_news", "source_label": "Waiver Wire", "theme": "warning",
                "headline": f"{_team_name(winner)} claimed {name}",
                "summary": f"{name} was claimed off waivers by {_team_name(winner)}. His contract leaves your books.",
                **portrait,
            })
        elif entry.get("user_claim"):
            _popup(session, f"waiver_beat:{entry['player_id']}", {
                "kind": "breaking_news", "source_label": "Waiver Wire", "theme": "info",
                "headline": f"{_team_name(winner)} had priority on {name}",
                "summary": "A club lower in the standings put in a claim and had first dibs.",
                **portrait,
            })
        return

    # Cleared: report to the original club's AHL affiliate.
    entry["cleared"] = True
    entry["waiver_status"] = "cleared"
    if orig is not None:
        ahl = [p for p in list(getattr(orig, "ahl_roster", None) or []) if p is not player]
        ahl.append(player)
        orig.ahl_roster = ahl
        try:
            player.on_waiver_wire = False
            player.waiver_status = "cleared"
            player.in_minors = True
            player.is_buried = True
            player.buried = True
            player.roster_location = "ahl"
            from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _team_label

            _set_assignment(player, org_nhl_team_id=_tid(orig), level="ahl", club=_team_label(orig))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        sync_team_cap_fields(orig, league)
    if str(entry.get("original_team_id")) == user_tid:
        _popup(session, f"waiver_clear:{entry['player_id']}", {
            "kind": "breaking_news", "source_label": "Waiver Wire", "theme": "info",
            "headline": f"{name} cleared waivers",
            "summary": f"No club claimed {name}. He has reported to your AHL affiliate.",
            **portrait,
        })
    elif entry.get("user_claim"):
        _popup(session, f"waiver_noroom:{entry['player_id']}", {
            "kind": "breaking_news", "source_label": "Waiver Wire", "theme": "warning",
            "headline": f"Your claim on {name} didn't go through",
            "summary": "You need cap space and an open contract slot when the window closes.",
            **portrait,
        })


def _cpu_waiver_moves(session: Any, rng: random.Random) -> int:
    """CPU clubs occasionally waive a depth vet to promote a waiver-exempt kid."""
    from services.contract_economy import (
        _player_ovr,
        _position_bucket,
        is_core_player_protected,
        is_waiver_exempt,
    )

    league = _league(session)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    moves = 0
    for team in list(getattr(league, "teams", None) or []):
        if _tid(team) == user_tid or rng.random() >= CPU_WAIVE_DAILY_CHANCE:
            continue
        roster = list(getattr(team, "roster", None) or [])
        if len(roster) < 22:
            continue
        cands = [
            p for p in roster
            if _position_bucket(p) != "G"
            and not is_waiver_exempt(p, team, league)
            and not is_core_player_protected(p, team, league)
            and _player_ovr(p) < 78
        ]
        if not cands:
            continue
        floor = min(_player_ovr(p) for p in cands)
        tier = [p for p in cands if _player_ovr(p) <= floor + 1.5]
        victim = max(tier, key=lambda p: (int(getattr(p, "age", 0) or 0), -_player_ovr(p)))
        bucket = _position_bucket(victim)
        ahl = list(getattr(team, "ahl_roster", None) or [])
        # Call-ups never need waivers, so any AHL skater at the position qualifies.
        callups = [p for p in ahl if _position_bucket(p) == bucket]
        if not callups:
            continue
        up = max(callups, key=_player_ovr)
        gap = _player_ovr(victim) - _player_ovr(up)
        # Waiving a better player for a worse call-up needs a real reason (it used to
        # happen for anything within 3 OVR).
        if gap > 0:
            # A high-ceiling kid can still bump a fading depth vet.
            try:
                from app.sim_engine.progression.potential import read_player_potential99

                pot = float(read_player_potential99(up) or 0)
            except Exception:
                pot = 0.0
            young_bump = gap <= 14 and pot >= _player_ovr(victim) + 4
            try:
                vet_age = int(getattr(victim, "age", 0) or 0)
            except Exception:
                vet_age = 0
            # Or the club simply moves on from an ageing depth vet.
            vet_cut = gap <= 15 and vet_age >= 31 and rng.random() < 0.35
            if not (young_bump or vet_cut):
                continue
        res = place_on_waivers(session, team, victim, reason="cpu_roster_move", manual=False)
        if not res.get("ok"):
            continue
        team.ahl_roster = [p for p in ahl if p is not up]
        team.roster = list(getattr(team, "roster", None) or []) + [up]
        try:
            up.in_minors = False
            up.is_buried = False
            up.buried = False
            up.roster_location = "nhl"
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        moves += 1
    return moves


def process_waiver_wire(session: Any) -> Dict[str, Any]:
    """Daily: close expired claim windows; let CPU clubs make waiver moves."""
    from services.contract_economy import _ensure_waiver_wire

    league = _league(session)
    if league is None or not _in_season(session):
        return {"resolved": 0}
    now = _cursor(session)
    resolved = 0
    for e in list(_ensure_waiver_wire(league)):
        if e.get("cleared") or e.get("claimed_by"):
            continue
        exp = e.get("expires_cursor")
        if exp is None or int(exp) <= now:
            try:
                _resolve_entry(session, e)
                resolved += 1
            except Exception:
                e["cleared"] = True
    rng = random.Random(f"waivers|{now}|{getattr(session, 'session_id', '')}")
    try:
        moves = _cpu_waiver_moves(session, rng)
    except Exception:
        moves = 0
    # Keep the wire list small.
    wire = _ensure_waiver_wire(league)
    if len(wire) > 400:
        league.waiver_wire = [e for e in wire if not (e.get("cleared") or e.get("claimed_by"))] + [
            e for e in wire if (e.get("cleared") or e.get("claimed_by"))
        ][-150:]
    return {"resolved": resolved, "cpu_moves": moves}
