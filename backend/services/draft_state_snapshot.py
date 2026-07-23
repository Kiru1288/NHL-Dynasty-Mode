"""
Draft-state snapshot / restore for mid-draft save resilience.

Authorities preserved:
- draft_state (clock / presentation)
- pick registry ownership + resolution
- drafted rights fields on player entities (via players_by_id / development search)
- reserve / prospect pool membership ids
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


def snapshot_draft_moment(session: Any) -> Dict[str, Any]:
    state = getattr(session, "draft_state", None) or {}
    league = getattr(getattr(session, "sim", None), "league", None)
    registry = {}
    if league is not None:
        reg = getattr(league, "draft_pick_registry", None)
        if isinstance(reg, dict):
            registry = {k: dict(v) for k, v in reg.items() if isinstance(v, dict)}

    drafted_rights: List[Dict[str, Any]] = []
    for pid in state.get("drafted_prospect_ids") or []:
        try:
            from services.draft_player_registry import get_player, find_development_home

            player = get_player(league, str(pid)) if league is not None else None
            if player is None and league is not None:
                player, _, _ = find_development_home(league, str(pid))
            if player is None:
                continue
            drafted_rights.append({
                "player_id": str(pid),
                "current_team_id": getattr(player, "current_team_id", None),
                "current_league_id": getattr(player, "current_league_id", None),
                "nhl_rights_team_id": getattr(player, "nhl_rights_team_id", None),
                "rights_status": getattr(player, "rights_status", None),
                "rights_type": getattr(player, "rights_type", None),
                "rights_expiry_year": getattr(player, "rights_expiry_year", None),
                "organizational_status": getattr(player, "organizational_status", None),
                "signed_status": getattr(player, "signed_status", None),
                "drafted": bool(getattr(player, "drafted", False)),
                "draft_overall_pick": getattr(player, "draft_overall_pick", None),
                "development_path": getattr(player, "development_path", None),
                "elc_slide_eligible": getattr(player, "elc_slide_eligible", None),
                "elc_slide_years_remaining": getattr(player, "elc_slide_years_remaining", None),
            })
        except Exception:
            continue

    org_pools: Dict[str, List[str]] = {}
    reserve_rows: Dict[str, List[Dict[str, Any]]] = {}
    for tid, team in (getattr(session, "team_by_id", None) or {}).items():
        org_pools[str(tid)] = [
            str(getattr(p, "id", "") or "")
            for p in (getattr(team, "prospect_pool", None) or [])
            if getattr(p, "id", None)
        ]
        reserve_rows[str(tid)] = [
            {k: v for k, v in e.items() if k != "player_ref"}
            for e in (getattr(team, "reserve_list", None) or [])
            if isinstance(e, dict)
        ]

    return {
        "moment": _classify_moment(state),
        "draft_state": deepcopy({k: v for k, v in state.items() if k != "_cache"}),
        "draft_pick_registry": registry,
        "drafted_rights": drafted_rights,
        "org_prospect_ids": org_pools,
        "reserve_lists": reserve_rows,
        "draft_completed": bool(getattr(session, "draft_completed", False)),
        "pending_trade_offers": list(state.get("trade_offers") or state.get("draft_day_trade_offers") or []),
    }


def _classify_moment(state: Dict[str, Any]) -> str:
    if state.get("draft_completed"):
        return "final_pick_complete"
    overall = int(state.get("overall_pick") or 1)
    if overall == 1:
        return "first_round_start"
    if state.get("is_user_pick"):
        return "user_pick"
    if state.get("trade_offers") or state.get("draft_day_trade_offers"):
        return "pending_trade_offer"
    if overall % 32 == 1 and overall > 1:
        return "end_of_round"
    return "mid_draft"


def restore_draft_moment(session: Any, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(snapshot, dict) or not snapshot.get("draft_state"):
        raise ValueError("Invalid draft snapshot")

    league = getattr(getattr(session, "sim", None), "league", None)
    state = deepcopy(snapshot["draft_state"])
    # Drop ephemeral cache; rebuild on next payload
    state.pop("_cache", None)
    session.draft_state = state
    session.draft_completed = bool(snapshot.get("draft_completed") or state.get("draft_completed"))
    session.draft_payload = None

    if league is not None and isinstance(snapshot.get("draft_pick_registry"), dict):
        try:
            league.draft_pick_registry = {
                k: dict(v) for k, v in snapshot["draft_pick_registry"].items() if isinstance(v, dict)
            }
        except Exception:
            pass

    from services.draft_player_registry import get_player, find_development_home, register_player

    for row in snapshot.get("drafted_rights") or []:
        pid = str(row.get("player_id") or "")
        if not pid or league is None:
            continue
        player = get_player(league, pid)
        if player is None:
            player, _, _ = find_development_home(league, pid)
        if player is None:
            continue
        for key, val in row.items():
            if key == "player_id":
                continue
            try:
                setattr(player, key, val)
            except Exception:
                pass
        register_player(league, player)

    for tid, ids in (snapshot.get("org_prospect_ids") or {}).items():
        team = (getattr(session, "team_by_id", None) or {}).get(str(tid))
        if team is None or league is None:
            continue
        pool = []
        for pid in ids:
            p = get_player(league, str(pid))
            if p is not None:
                pool.append(p)
        try:
            team.prospect_pool = pool
        except Exception:
            pass

    for tid, rows in (snapshot.get("reserve_lists") or {}).items():
        team = (getattr(session, "team_by_id", None) or {}).get(str(tid))
        if team is None:
            continue
        cleaned = [{k: v for k, v in e.items() if k != "player_ref"} for e in rows if isinstance(e, dict)]
        try:
            team.reserve_list = cleaned
        except Exception:
            pass

    # Refresh live ownership markers after restore
    try:
        from services.draft_pick_ownership import refresh_draft_order_ownership

        refresh_draft_order_ownership(session)
    except Exception:
        pass

    restored = getattr(session, "draft_state", None) or {}
    return {
        "ok": True,
        "moment": snapshot.get("moment"),
        "overall_pick": restored.get("overall_pick"),
        "current_team_id": restored.get("current_team_id"),
        "drafted_count": len(restored.get("drafted_prospect_ids") or []),
        "is_user_pick": restored.get("is_user_pick"),
        "draft_completed": bool(getattr(session, "draft_completed", False)),
    }


def assert_draft_restored_identically(before: Dict[str, Any], after_session: Any) -> None:
    """Raise AssertionError if restore diverged on critical fields."""
    after = snapshot_draft_moment(after_session)
    keys = (
        "overall_pick",
        "current_team_id",
        "drafted_prospect_ids",
        "draft_completed",
        "is_user_pick",
    )
    b_state = before.get("draft_state") or {}
    a_state = after.get("draft_state") or {}
    for k in keys:
        if b_state.get(k) != a_state.get(k):
            raise AssertionError(f"draft_state.{k} mismatch: {b_state.get(k)!r} vs {a_state.get(k)!r}")
    if len(before.get("drafted_rights") or []) != len(after.get("drafted_rights") or []):
        raise AssertionError("drafted_rights count mismatch")
