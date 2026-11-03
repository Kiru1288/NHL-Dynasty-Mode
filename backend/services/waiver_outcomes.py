"""Waiver claim outcomes for the user's club (D5) and roster placement for claimed
players (D6).

* Every claim the user submitted resolves to an explicit result — Claim Successful
  (where he was assigned, your priority slot) or Claim Unsuccessful (who got him and
  their priority, or why your claim was invalid). Results go out as a ``waiver`` popup
  (an important kind, so it interrupts the screen) and are kept on
  ``session.waiver_claim_results`` for the waiver wire payload.
* A player the user claims follows the roster rules used by ``roster_moves.py``:
  he joins the NHL roster only when an active slot is open (23-man max). When the
  roster is full, a waiver-exempt depth player is assigned to the AHL to make room
  (no waivers needed). If nobody can go down without waivers, the claimed player still
  reports to the NHL club (a claimed player must, under NHL rules) and the user is told
  to clear a spot before the next game.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging as _logging_swallow

_swallowed_log = _logging_swallow.getLogger(__name__)

ACTIVE_MAX = 23
MAX_RESULTS = 20


def _tid(team: Any) -> str:
    return str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")


def _team_name(team: Any) -> str:
    return str(getattr(team, "name", None) or getattr(team, "abbreviation", None) or _tid(team))


def _pname(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", "") or "Player")


def _ordinal(n: Optional[int]) -> str:
    if not n:
        return "—"
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def priority_of(order: List[Any], team: Any) -> Optional[int]:
    tid = _tid(team)
    for i, t in enumerate(order):
        if _tid(t) == tid:
            return i + 1
    return None


def _active_count(team: Any) -> int:
    try:
        from services.roster_compliance import iter_active_nhl_roster

        return len(list(iter_active_nhl_roster(team)))
    except Exception:
        return len([p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)])


def place_user_claim(session: Any, team: Any, player: Any, league: Any) -> Dict[str, Any]:
    """Called right after the claimed player was added to the user's NHL roster."""
    count = _active_count(team)
    if count <= ACTIVE_MAX:
        return {"level": "NHL", "label": "NHL roster", "detail": f"Open roster spot ({count}/{ACTIVE_MAX})."}
    try:
        from app.sim_engine.trades.roster_balance import _assign_ahl, _waiver_exempt, send_down_candidates

        cands = send_down_candidates(team, protect_ids=[str(getattr(player, "id", "") or "")], count=6)
        exempt = [p for p in cands if _waiver_exempt(p)]
        if exempt:
            down = exempt[0]
            _assign_ahl(team, down)
            try:
                from services.contract_economy import sync_team_cap_fields

                sync_team_cap_fields(team, league)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            return {
                "level": "NHL",
                "label": "NHL roster",
                "detail": f"Roster was full — {_pname(down)} (waiver-exempt) was assigned to the AHL to open a spot.",
                "sent_down": {"player_id": str(getattr(down, "id", "") or ""), "name": _pname(down)},
            }
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        session._roster_over_limit_flag = True
    except Exception:
        pass
    return {
        "level": "NHL",
        "label": "NHL roster",
        "detail": (
            f"Your NHL roster is now {count}/{ACTIVE_MAX}. Nobody can go down without waivers — "
            "send down or waive a player before the next game."
        ),
        "needs_action": True,
    }


def record_result(session: Any, row: Dict[str, Any]) -> None:
    rows = list(getattr(session, "waiver_claim_results", None) or [])
    rows = [r for r in rows if str(r.get("player_id")) != str(row.get("player_id"))]
    rows.insert(0, row)
    try:
        session.waiver_claim_results = rows[:MAX_RESULTS]
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def build_result(
    session: Any,
    entry: Dict[str, Any],
    *,
    success: bool,
    user_priority: Optional[int],
    winner: Any = None,
    winner_priority: Optional[int] = None,
    assignment: Optional[Dict[str, Any]] = None,
    invalid_reason: str = "",
) -> Dict[str, Any]:
    name = str(entry.get("name") or "Player")
    from_team = str(entry.get("original_team_name") or "his club")
    facts: List[Dict[str, Any]] = []
    if success:
        status = "Claim Successful"
        headline = f"Claim Successful: {name} is yours"
        summary = f"You won the waiver claim on {name} from {from_team}."
        if assignment:
            summary += f" Assigned to: {assignment.get('label')}. {assignment.get('detail') or ''}".rstrip()
            facts.append({"label": "Assigned to", "value": assignment.get("label")})
        facts.append({"label": "Your priority", "value": f"#{user_priority}" if user_priority else "—", "sub": f"{_ordinal(user_priority)} in the waiver order"})
    else:
        status = "Claim Unsuccessful"
        if winner is not None:
            wname = _team_name(winner)
            headline = f"Claim Unsuccessful: {name} awarded to {wname}"
            if invalid_reason:
                summary = f"Your claim on {name} was invalid ({invalid_reason}), so he went to {wname}."
            else:
                summary = (
                    f"{wname} held higher waiver priority (#{winner_priority}) than you (#{user_priority}) "
                    f"and claimed {name}."
                )
            facts.append({"label": "Awarded to", "value": wname})
            facts.append({"label": "Their priority", "value": f"#{winner_priority}" if winner_priority else "—"})
        else:
            headline = f"Claim Unsuccessful: {name}"
            summary = f"Your claim on {name} didn't go through ({invalid_reason or 'no valid claim'}). He cleared waivers and reported to {from_team}'s AHL club."
            facts.append({"label": "Result", "value": "Cleared to AHL"})
        facts.append({"label": "Your priority", "value": f"#{user_priority}" if user_priority else "—"})
        if invalid_reason:
            facts.append({"label": "Why", "value": invalid_reason})
    return {
        "player_id": str(entry.get("player_id") or ""),
        "player_name": name,
        "from_team": from_team,
        "success": bool(success),
        "status_label": status,
        "headline": headline,
        "summary": summary,
        "facts": facts,
        "user_priority": user_priority,
        "winner_team": _team_name(winner) if winner is not None else None,
        "winner_priority": winner_priority,
        "assignment": assignment or None,
        "day": int(getattr(session, "calendar_cursor", 0) or 0),
    }


def popup_payload(result: Dict[str, Any], portrait: Dict[str, Any]) -> Dict[str, Any]:
    actions = [{"id": "roster", "label": "Open roster", "primary": True}] if result.get("success") else []
    return {
        "kind": "waiver",
        "source_label": "Waiver Wire",
        "theme": "positive" if result.get("success") else "warning",
        "headline": result["headline"],
        "summary": result["summary"],
        "body": result["summary"],
        "waiver_result": result,
        "actions": actions,
        **portrait,
    }


def claim_invalid_reason(team: Any, league: Any, entry: Dict[str, Any]) -> str:
    try:
        from services.contract_economy import get_team_cap_snapshot_full, validate_contract_slots

        snap = get_team_cap_snapshot_full(team, league)
        space = float(snap.get("usable_cap_space_m") or 0.0)
        if space < float(entry.get("cap_hit_m") or 0.0) - 0.001:
            return f"not enough cap space: ${space:.2f}M free vs ${float(entry.get('cap_hit_m') or 0):.2f}M hit"
        if not bool(validate_contract_slots(team, league, additional=1).get("ok")):
            return "no open contract slot (50-contract limit)"
    except Exception:
        return "cap or contract-slot check failed"
    return ""
