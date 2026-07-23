"""
Live draft-pick ownership resolution.

Never trust a cached draft_order team_id for the on-clock club. Always resolve through
league.draft_pick_registry before executing a selection or after a mid-draft trade.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def resolve_current_pick_owner(league: Any, pick_id: str) -> Optional[str]:
    pick_id = str(pick_id or "")
    if not pick_id or league is None:
        return None
    try:
        from app.sim_engine.trades.trade_pick_registry import get_pick_by_id

        row = get_pick_by_id(league, pick_id)
    except Exception:
        reg = getattr(league, "draft_pick_registry", None) or {}
        row = reg.get(pick_id) if isinstance(reg, dict) else None
    if not isinstance(row, dict):
        return None
    return str(row.get("current_owner_team_id") or row.get("original_team_id") or "") or None


def resolve_slot_owner(session: Any, slot: Dict[str, Any]) -> str:
    league = getattr(getattr(session, "sim", None), "league", None)
    pick_id = slot.get("pick_id")
    if league is not None and pick_id:
        owner = resolve_current_pick_owner(league, str(pick_id))
        if owner:
            return owner
    return str(slot.get("team_id") or slot.get("original_owner_team_id") or "")


def _display_team_name(session: Any, team_id: str) -> str:
    try:
        from services.franchise_sim import _display_team

        tm = session.team_by_id.get(str(team_id))
        return _display_team(tm) if tm else str(team_id)
    except Exception:
        return str(team_id)


def apply_registry_owner_to_slot(session: Any, slot: Dict[str, Any], draft_year: int) -> Dict[str, Any]:
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return dict(slot)
    orig = str(slot.get("original_owner_team_id") or slot.get("team_id") or "")
    rnd = int(slot.get("round") or 1)
    out = dict(slot)
    out["original_owner_team_id"] = orig
    try:
        from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry, get_pick_by_id
        from app.sim_engine.trades.trade_asset import canonical_pick_id

        ensure_draft_pick_registry(league, start_year=draft_year)
        pick_id = str(slot.get("pick_id") or canonical_pick_id(draft_year, rnd, orig))
        row = get_pick_by_id(league, pick_id)
        if isinstance(row, dict) and not row.get("resolved"):
            current = str(row.get("current_owner_team_id") or orig)
            reg_orig = str(row.get("original_team_id") or orig)
            out["pick_id"] = pick_id
            out["team_id"] = current
            out["original_owner_team_id"] = reg_orig
            out["team_name"] = _display_team_name(session, current)
            out["original_owner_team_name"] = _display_team_name(session, reg_orig)
            if current != reg_orig:
                out["is_traded"] = True
                out["via_team_id"] = reg_orig
                out["via_team_name"] = out["original_owner_team_name"]
            else:
                out["is_traded"] = False
                out.pop("via_team_id", None)
                out.pop("via_team_name", None)
        elif isinstance(row, dict) and row.get("resolved"):
            out["pick_id"] = pick_id
            out["selected_prospect_id"] = row.get("selected_prospect_id")
            out["resolved"] = True
    except Exception:
        out["original_owner_team_name"] = _display_team_name(session, orig)
    if not out.get("original_owner_team_name"):
        out["original_owner_team_name"] = _display_team_name(session, orig)
    return out


def refresh_draft_order_ownership(session: Any) -> List[Dict[str, Any]]:
    """Refresh all unresolved slots from the pick registry and update clock markers."""
    state = getattr(session, "draft_state", None)
    if not isinstance(state, dict) or not state.get("draft_started"):
        return []
    draft_year = int(state.get("draft_year") or int(getattr(session, "season_calendar_year", 2025)) + 1)
    order = list(state.get("draft_order") or [])
    completed_ids = {
        str(p.get("prospect_id") or "")
        for p in (state.get("completed_picks") or [])
        if p.get("prospect_id")
    }
    refreshed: List[Dict[str, Any]] = []
    overall = int(state.get("overall_pick") or 1)
    for idx, slot in enumerate(order):
        pick_overall = int(slot.get("overall_pick") or (idx + 1))
        # Completed picks keep selecting-team history frozen.
        if pick_overall < overall or slot.get("resolved") or slot.get("selected_prospect_id"):
            frozen = dict(slot)
            refreshed.append(frozen)
            continue
        refreshed.append(apply_registry_owner_to_slot(session, dict(slot), draft_year))

    current = refreshed[overall - 1] if 0 < overall <= len(refreshed) else None
    user_id = str(getattr(session, "user_team_id", "") or "")
    state["draft_order"] = refreshed
    state["pick_ownership"] = refreshed
    if current and not state.get("draft_completed"):
        state["current_team_id"] = str(current.get("team_id") or "")
        state["current_round"] = int(current.get("round") or state.get("current_round") or 1)
        state["current_pick"] = int(current.get("pick_in_round") or state.get("current_pick") or 1)
        state["is_user_pick"] = str(current.get("team_id") or "") == user_id
        to_user = None
        for i, s in enumerate(refreshed):
            if i + 1 < overall:
                continue
            if str(s.get("team_id") or "") == user_id:
                to_user = (i + 1) - overall
                break
        state["picks_until_user"] = to_user
    session.draft_state = state
    try:
        from services.franchise_sim import invalidate_session_payload_caches

        invalidate_session_payload_caches(session, "draft_pick_trade")
    except Exception:
        pass
    try:
        if getattr(session, "draft_payload", None) is not None:
            session.draft_payload = None
    except Exception:
        pass
    return refreshed


def sync_draft_clock_after_trade(session: Any) -> Dict[str, Any]:
    """Call after any draft-day (or live) pick trade."""
    order = refresh_draft_order_ownership(session)
    state = getattr(session, "draft_state", None) or {}
    overall = int(state.get("overall_pick") or 1)
    return {
        "ok": True,
        "current_team_id": state.get("current_team_id"),
        "is_user_pick": state.get("is_user_pick"),
        "picks_until_user": state.get("picks_until_user"),
        "slots_refreshed": len(order),
        "current_slot": order[overall - 1]
        if order and not state.get("draft_completed") and overall <= len(order)
        else None,
    }
