"""Explicit NHL / AHL / junior call-up and send-down moves for the Rosters UI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)


def _pid(player: Any) -> str:
    return str(getattr(player, "id", "") or "")


def _age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    try:
        return int(getattr(ident, "age", None) if ident is not None else getattr(player, "age", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _nhl_gp(player: Any, session: Any = None) -> int:
    from services.contract_economy import _player_nhl_games

    gp = int(_player_nhl_games(player) or 0)
    if session is not None and gp <= 0:
        pid = _pid(player)
        row = dict((getattr(session, "player_season_stats", None) or {}).get(pid) or {})
        try:
            gp = max(gp, int(row.get("gp") or 0))
        except (TypeError, ValueError):
            pass
    return gp


def _locate_on_user_org(team: Any, league: Any, player_id: str) -> Tuple[Optional[Any], str]:
    """Return (player, location) where location is nhl|ahl|echl|junior|prospect_pool|none."""
    pid = str(player_id or "")
    if not pid or team is None:
        return None, "none"

    for attr, loc in (
        ("roster", "nhl"),
        ("ahl_roster", "ahl"),
        ("echl_roster", "echl"),
        ("prospect_pool", "prospect_pool"),
    ):
        for p in getattr(team, attr, None) or []:
            if _pid(p) == pid:
                return p, loc

    uid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "")
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if _pid(p) != pid:
                    continue
                rights = str(
                    getattr(p, "nhl_rights_team_id", None)
                    or getattr(p, "rights_team_id", None)
                    or getattr(p, "drafted_by", None)
                    or ""
                )
                if rights and rights != uid:
                    continue
                return p, f"junior:{code}"
    return None, "none"


def _remove_from_list(team: Any, attr: str, player: Any) -> bool:
    pool = list(getattr(team, attr, None) or [])
    pid = _pid(player)
    kept = [p for p in pool if _pid(p) != pid]
    if len(kept) == len(pool):
        return False
    setattr(team, attr, kept)
    return True


def _append_unique(team: Any, attr: str, player: Any) -> None:
    pool = list(getattr(team, attr, None) or [])
    pid = _pid(player)
    if any(_pid(p) == pid for p in pool):
        return
    pool.append(player)
    setattr(team, attr, pool)


def _reattach_to_juniors(league: Any, player: Any) -> bool:
    """Put a junior-eligible player back on a development club if detached."""
    from services.draft_rights_engine import _detach_from_development_leagues

    path = str(
        getattr(player, "development_path", None)
        or getattr(player, "post_draft_league", None)
        or getattr(player, "current_league_id", None)
        or "JUNIOR"
    ).upper()
    # Already on a junior list?
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            players = tm.get("players")
            if isinstance(players, list) and player in players:
                return True

    _detach_from_development_leagues(league, player)
    targets: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "").upper()
        if path and path not in code and not any(
            tok in code for tok in ("OHL", "WHL", "QMJHL", "CHL", "USHL", "NCAA", "JUNIOR")
        ):
            if "EUROPE" in path and not code.startswith("EU_"):
                continue
            if "NCAA" in path and "NCAA" not in code:
                continue
        for tm in block.get("teams") or []:
            if isinstance(tm, dict):
                targets.append((block, tm))
    if not targets:
        for block in getattr(league, "development_leagues", None) or []:
            for tm in block.get("teams") or []:
                if isinstance(tm, dict):
                    targets.append((block, tm))
    if not targets:
        return False
    # Prefer the thinnest roster so we don't pile onto one club.
    targets.sort(key=lambda pair: len(pair[1].get("players") or []))
    block, tm = targets[0]
    players = list(tm.get("players") or [])
    players.append(player)
    tm["players"] = players
    try:
        setattr(player, "roster_location", "junior")
        setattr(player, "in_minors", False)
        setattr(player, "current_league_id", str(block.get("league_code") or path))
        setattr(player, "development_path", str(block.get("league_code") or path))
        setattr(player, "organizational_status", "signed_junior")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return True


def _waiver_block_reason(player: Any, team: Any) -> Optional[str]:
    """Why this player cannot be exposed to waivers (None when he can)."""
    from services.contract_economy import can_waive_or_bury, is_compliance_protected

    try:
        ok, why = can_waive_or_bury(player)
        if not ok:
            return str(why)
        if is_compliance_protected(player, team):
            return "Core player — the front office won't expose him to waivers"
    except Exception:
        return None
    return None


def available_roster_moves(session: Any, player_id: str) -> Dict[str, Any]:
    from services.contract_economy import (
        has_active_contract,
        is_waiver_exempt,
        is_waiver_required_for_assignment,
        uses_nhl_contract_slot,
    )
    from services.elc_offer_engine import ELC_SLIDE_GAMES_THRESHOLD

    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None) if sim else None
    team = (getattr(session, "team_by_id", None) or {}).get(str(getattr(session, "user_team_id", "") or ""))
    player, loc = _locate_on_user_org(team, league, player_id)
    if player is None:
        return {"ok": False, "reason": "Player not found on your organization", "actions": []}

    nhl_gp = _nhl_gp(player, session)
    age = _age(player)
    slide_threshold = int(
        getattr(player, "slide_games_threshold", None)
        or ELC_SLIDE_GAMES_THRESHOLD
    )
    actions: List[Dict[str, Any]] = []
    nhl_full = len([p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]) >= 23
    full_reason = "NHL roster is full (23) — send a player down first" if nhl_full else None

    if loc == "nhl":
        needs_waivers = bool(is_waiver_required_for_assignment(player, "nhl", "ahl", league))
        waiver_block = _waiver_block_reason(player, team) if needs_waivers else None
        actions.append({
            "id": "send_down_ahl",
            "label": "Send to AHL",
            "direction": "down",
            "destination": "AHL",
            "enabled": waiver_block is None,
            "reason": waiver_block,
            "requires_waivers": needs_waivers,
            "waiver_exempt": bool(is_waiver_exempt(player, team, league)),
        })
        actions.append({
            "id": "send_down_echl",
            "label": "Send to ECHL",
            "direction": "down",
            "destination": "ECHL",
            "enabled": waiver_block is None,
            "reason": waiver_block,
            "requires_waivers": needs_waivers,
            "waiver_exempt": bool(is_waiver_exempt(player, team, league)),
            "note": "Waivers apply the same as an AHL assignment when required",
        })
        if age <= 20 and (nhl_gp < slide_threshold or not has_active_contract(player)):
            actions.append({
                "id": "return_junior",
                "label": f"Return to juniors (slide if <{slide_threshold} NHL GP)",
                "direction": "down",
                "destination": "Juniors",
                "enabled": True,
                "nhl_gp": nhl_gp,
                "slide_games_threshold": slide_threshold,
                "slide_safe": nhl_gp < slide_threshold,
            })
    elif loc == "ahl":
        actions.append({
            "id": "call_up_ahl",
            "label": "Call up to NHL",
            "direction": "up",
            "destination": "NHL",
            "enabled": not nhl_full,
            "reason": full_reason,
        })
        actions.append({
            "id": "send_down_echl",
            "label": "Assign to ECHL",
            "direction": "down",
            "destination": "ECHL",
            "enabled": True,
            "requires_waivers": False,
        })
    elif loc == "echl":
        actions.append({
            "id": "call_up_echl_ahl",
            "label": "Call up to AHL",
            "direction": "up",
            "destination": "AHL",
            "enabled": True,
        })
        actions.append({
            "id": "call_up_ahl",
            "label": "Call up to NHL",
            "direction": "up",
            "destination": "NHL",
            "enabled": not nhl_full,
            "reason": full_reason,
            "note": "Direct NHL recall from ECHL",
        })
    elif loc.startswith("junior") or loc == "prospect_pool":
        signed = has_active_contract(player) or uses_nhl_contract_slot(player)
        actions.append({
            "id": "call_up_junior",
            "label": "Call up from juniors (NHL)",
            "direction": "up",
            "destination": "NHL",
            "enabled": bool(signed) and not nhl_full,
            "reason": None if signed and not nhl_full else ("Sign ELC before an NHL recall" if not signed else full_reason),
            "nhl_gp": nhl_gp,
            "slide_games_threshold": slide_threshold,
            "slide_note": f"ELC year slides if sent back before {slide_threshold} NHL games",
        })

    lineup_slot = None
    lineup_mode = "auto"
    try:
        from services.lineup_integrity import saved_even_strength, user_lineup_slot_for

        if saved_even_strength(session) is not None:
            lineup_mode = "saved"
            lineup_slot = user_lineup_slot_for(session, player) if loc == "nhl" else None
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    return {
        "ok": True,
        "player_id": _pid(player),
        "player_name": _player_name(player),
        "location": loc,
        "lineup_slot": lineup_slot,
        "lineup_mode": lineup_mode,
        "age": age,
        "nhl_gp": nhl_gp,
        "slide_games_threshold": slide_threshold,
        "actions": actions,
    }


def _live_waivers(session: Any) -> bool:
    try:
        from services.waivers import _in_season

        return _in_season(session)
    except Exception:
        return False


def _waive_live(session: Any, team: Any, player: Any, reason: str) -> Dict[str, Any]:
    from services.waivers import place_on_waivers

    res = place_on_waivers(session, team, player, reason=reason, manual=True)
    if not res.get("ok"):
        return res
    return {
        "ok": True,
        "player_id": _pid(player),
        "moved": "on_waivers",
        "waivers": True,
        "message": "Placed on waivers — 24-hour claim window. If he clears he reports to your AHL club.",
        "waiver_entry": res.get("waiver_entry"),
    }


def _call_up_cap_block(session: Any, team: Any, league: Any, player: Any) -> Optional[str]:
    """A recall has to fit under the cap: his full hit replaces the buried residual."""
    try:
        from app.sim_engine.economy.cap_engine import (
            buried_cap_hit_millions,
            player_cap_hit_millions,
        )
        from services.contract_economy import cap_books_for_session, get_team_cap_snapshot_full

        books = cap_books_for_session(session)
        if not books.get("in_season"):
            return None  # offseason rosters are not held to the cap day to day
        snap = get_team_cap_snapshot_full(
            team,
            league,
            season_year=books["season_year"],
            calendar_cursor=books["calendar_cursor"],
            regular_season_last_index=books["regular_season_last_index"],
            count_expiring=True,
        )
        full = float(player_cap_hit_millions(player) or 0.0)
        residual = float(buried_cap_hit_millions(player, season_start_year=books["season_year"]) or 0.0)
        added = max(0.0, full - residual)
        space = float(snap.get("usable_cap_space_m") or 0.0)
        if added > space + 1e-6:
            return (
                f"Calling him up adds ${added:.2f}M against the cap and you have "
                f"${max(0.0, space):.2f}M of space. Clear room first."
            )
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return None


def execute_roster_move(session: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    from services.contract_economy import (
        bury_player_contract,
        expose_player_to_waivers,
        has_active_contract,
        is_waiver_required_for_assignment,
        sync_team_cap_fields,
        unbury_player_contract,
        uses_nhl_contract_slot,
    )

    action = str(body.get("action") or body.get("move") or "").strip()
    player_id = str(body.get("player_id") or body.get("playerId") or "")
    force_waive = bool(body.get("confirm_waivers") or body.get("force_waivers"))

    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None) if sim else None
    team = (getattr(session, "team_by_id", None) or {}).get(str(getattr(session, "user_team_id", "") or ""))
    if team is None or league is None:
        return {"ok": False, "reason": "Franchise session incomplete"}

    player, loc = _locate_on_user_org(team, league, player_id)
    if player is None:
        return {"ok": False, "reason": "Player not found on your organization"}

    result: Dict[str, Any]
    if action in ("call_up_ahl", "call_up_junior"):
        blocked = _call_up_cap_block(session, team, league, player)
        if blocked:
            return {"ok": False, "reason": blocked}
    if action == "call_up_ahl":
        if loc not in ("ahl", "echl") and not (
            loc == "nhl"
            and (getattr(player, "is_buried", False) or getattr(player, "in_minors", False))
        ):
            return {"ok": False, "reason": "Player is not on the AHL/ECHL list"}
        if loc == "nhl":
            result = unbury_player_contract(team, player, league)
            if result.get("ok") and not result.get("moved"):
                result["moved"] = "ahl_to_nhl"
        else:
            roster = list(getattr(team, "roster", None) or [])
            if len(roster) >= 23:
                return {"ok": False, "reason": "NHL roster is full (23)"}
            left = _remove_from_list(team, "ahl_roster", player) or _remove_from_list(team, "echl_roster", player)
            if not left:
                return {"ok": False, "reason": "Could not leave minors list"}
            try:
                player.in_minors = False
                player.is_buried = False
                player.buried = False
                player.waiver_status = None
                player.roster_location = "nhl"
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            _append_unique(team, "roster", player)
            sync_team_cap_fields(team, league)
            result = {"ok": True, "player_id": _pid(player), "moved": f"{loc}_to_nhl"}
    elif action == "call_up_echl_ahl":
        if loc != "echl":
            return {"ok": False, "reason": "Player is not on the ECHL list"}
        if not _remove_from_list(team, "echl_roster", player):
            return {"ok": False, "reason": "Could not leave ECHL list"}
        try:
            player.in_minors = True
            player.is_buried = False
            player.roster_location = "ahl"
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        _append_unique(team, "ahl_roster", player)
        sync_team_cap_fields(team, league)
        result = {"ok": True, "player_id": _pid(player), "moved": "echl_to_ahl"}
    elif action == "send_down_echl":
        if loc not in ("nhl", "ahl"):
            return {"ok": False, "reason": "Player must be on NHL or AHL to assign to ECHL"}
        if loc == "nhl":
            if is_waiver_required_for_assignment(player, "nhl", "ahl", league) and not force_waive:
                return {
                    "ok": False,
                    "reason": "waiver_required",
                    "requires_waivers": True,
                    "hint": "Confirm waivers to assign to the ECHL",
                }
            echl_waived = bool(force_waive and is_waiver_required_for_assignment(player, "nhl", "ahl", league))
            if echl_waived and _live_waivers(session):
                return _waive_live(session, team, player, "send_down_echl")
            if echl_waived:
                w = expose_player_to_waivers(team, player, league)
                if not w.get("ok"):
                    return w
            buried = bury_player_contract(team, player, league, skip_waiver_check=True)
            if not buried.get("ok"):
                return buried
            _remove_from_list(team, "roster", player)
        else:
            echl_waived = False
            _remove_from_list(team, "ahl_roster", player)
        try:
            player.roster_location = "echl"
            player.in_minors = True
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        _append_unique(team, "echl_roster", player)
        sync_team_cap_fields(team, league)
        result = {
            "ok": True,
            "player_id": _pid(player),
            "moved": f"{loc}_to_echl",
            "waivers": echl_waived,
        }
    elif action == "send_down_ahl":
        if loc != "nhl":
            return {"ok": False, "reason": "Player is not on the NHL roster"}
        if is_waiver_required_for_assignment(player, "nhl", "ahl", league) and not force_waive:
            return {
                "ok": False,
                "reason": "waiver_required",
                "requires_waivers": True,
                "hint": "Confirm waivers to assign to the AHL",
            }
        waived = bool(force_waive and is_waiver_required_for_assignment(player, "nhl", "ahl", league))
        if waived and _live_waivers(session):
            return _waive_live(session, team, player, "send_down")
        if waived:
            w = expose_player_to_waivers(team, player, league)
            if not w.get("ok"):
                return w
        buried = bury_player_contract(team, player, league, skip_waiver_check=True)
        if not buried.get("ok"):
            return buried
        _remove_from_list(team, "roster", player)
        _append_unique(team, "ahl_roster", player)
        try:
            player.roster_location = "ahl"
            player.in_minors = True
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        sync_team_cap_fields(team, league)
        result = {**buried, "moved": "nhl_to_ahl", "waivers": waived}
    elif action == "call_up_junior":
        if not (loc.startswith("junior") or loc == "prospect_pool"):
            return {"ok": False, "reason": "Player is not in juniors / prospect pool"}
        if not (has_active_contract(player) or uses_nhl_contract_slot(player)):
            return {"ok": False, "reason": "Sign an ELC before calling up from juniors"}
        roster = list(getattr(team, "roster", None) or [])
        if len(roster) >= 23:
            return {"ok": False, "reason": "NHL roster is full (23)"}
        from services.draft_rights_engine import _detach_from_development_leagues

        _detach_from_development_leagues(league, player)
        _remove_from_list(team, "prospect_pool", player)
        _remove_from_list(team, "ahl_roster", player)
        try:
            player.in_minors = False
            player.is_buried = False
            player.buried = False
            player.roster_location = "nhl"
            player.organizational_status = "signed_nhl"
            player.status = "nhl"
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        _append_unique(team, "roster", player)
        sync_team_cap_fields(team, league)
        nhl_gp = _nhl_gp(player, session)
        result = {
            "ok": True,
            "player_id": _pid(player),
            "moved": "junior_to_nhl",
            "nhl_gp": nhl_gp,
            "slide_note": "Return to juniors before the slide threshold to preserve the ELC year",
        }
    elif action == "return_junior":
        if loc != "nhl":
            return {"ok": False, "reason": "Player is not on the NHL roster"}
        if _age(player) > 20:
            return {"ok": False, "reason": "Player is no longer junior-eligible"}
        nhl_gp = _nhl_gp(player, session)
        from services.elc_offer_engine import ELC_SLIDE_GAMES_THRESHOLD

        threshold = int(getattr(player, "slide_games_threshold", None) or ELC_SLIDE_GAMES_THRESHOLD)
        _remove_from_list(team, "roster", player)
        if not _reattach_to_juniors(league, player):
            _append_unique(team, "prospect_pool", player)
        try:
            player.roster_location = "junior"
            player.in_minors = False
            player.is_buried = False
            if nhl_gp < threshold:
                player.elc_slide_eligible = True
                c = getattr(player, "contract", None)
                if isinstance(c, dict):
                    c["slide_triggered"] = True
                    c["can_slide"] = True
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        sync_team_cap_fields(team, league)
        result = {
            "ok": True,
            "player_id": _pid(player),
            "moved": "nhl_to_junior",
            "nhl_gp": nhl_gp,
            "slide_preserved": nhl_gp < threshold,
            "slide_games_threshold": threshold,
        }
    else:
        return {"ok": False, "reason": f"Unknown roster move: {action}"}

    if result.get("ok"):
        result.update(_move_summary(result.get("moved"), player, bool(result.get("waivers"))))
        result["player_name"] = _player_name(player)
        result["lineup"] = _sync_lineup_after_move(session, player, action, str(result.get("moved") or ""))

    try:
        from services.franchise_sim import invalidate_session_payload_caches

        invalidate_session_payload_caches(session, reason="roster_move")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    avail = available_roster_moves(session, player_id)
    result["available_moves"] = avail.get("actions") or []
    result["location"] = avail.get("location")
    return result


_DESTINATION_LABELS = {
    "nhl": "NHL",
    "ahl": "AHL",
    "echl": "ECHL",
    "junior": "Juniors",
}


def _player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(ident, "name", None) or getattr(player, "name", None) or "Player")


def _move_summary(moved: Any, player: Any, waived: bool) -> Dict[str, Any]:
    """One headline shape for every call-up / send-down so the UI can render them alike."""
    code = str(moved or "")
    src, _, dst = code.partition("_to_")
    src_label = _DESTINATION_LABELS.get(src.split(":")[0], src.upper() or "—")
    dst_label = _DESTINATION_LABELS.get(dst, dst.upper() or "—")
    direction = "up" if dst == "nhl" or (src == "echl" and dst == "ahl") else "down"
    name = _player_name(player)
    if direction == "up":
        headline = f"{name} called up to the {dst_label}"
    elif dst == "junior":
        headline = f"{name} returned to juniors"
    else:
        headline = (
            f"{name} placed on waivers and assigned to the {dst_label}"
            if waived
            else f"{name} assigned to the {dst_label}"
        )
    return {
        "direction": direction,
        "from_level": src_label,
        "to_level": dst_label,
        "headline": headline,
    }


def _sync_lineup_after_move(session: Any, player: Any, action: str, moved: str) -> Dict[str, Any]:
    """Keep the saved Edit Lines sheet in step with the NHL roster.

    A call-up drops into an open slot of his position when one exists (otherwise
    he is an extra in the Edit Lines pool). A send-down vacates his slot, which
    is left open and flagged so the GM picks the replacement.
    """
    try:
        from services.lineup_integrity import player_key, reconcile_user_lineup
        from services.franchise_sim import _log_lineup_integrity

        pid = player_key(player)
        to_nhl = moved.endswith("_to_nhl") or (action == "call_up_ahl" and moved == "")
        report = reconcile_user_lineup(
            session,
            auto_fill=False,
            prefer_ids=[pid] if to_nhl else [],
            reason=f"roster_move:{action}",
        )
        _log_lineup_integrity(session, report, reason="roster move")
    except Exception as exc:  # pragma: no cover - lineup sync must never undo a legal move
        return {"ok": False, "error": str(exc)}

    placed = next((row for row in report.get("placed") or [] if row.get("player_id") == pid), None)
    vacated = [row for row in report.get("vacated") or [] if str(row.get("player_id") or "") in (pid, str(getattr(player, "id", "") or ""))]
    if not report.get("has_lineup"):
        status = "auto_lines"
    elif placed:
        status = "placed"
    elif to_nhl:
        status = "extra"
    elif vacated:
        status = "vacated"
    else:
        status = "unchanged"
    return {
        "ok": True,
        "status": status,
        "slot": placed.get("label") if placed else None,
        "vacated": [row.get("label") for row in vacated],
        "gaps": list(report.get("gap_labels") or []),
        "fillable": bool(report.get("fillable", True)),
        "unfillable": list(report.get("unfillable") or []),
    }
