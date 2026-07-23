"""
LEGACY / NOT USED BY THE LIVE API.

The FastAPI server imports from backend/services/franchise_sim.py, not this file.
Edits here will not change franchise mode behavior. See backend/services/README.md.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.sim_engine.franchise.session import FranchiseSession


def get_franchise_chemistry_report(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.systems.chemistry import build_public_chemistry_report

    return build_public_chemistry_report(session)


def enter_franchise_playoffs(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.offseason import complete_playoffs

    return complete_playoffs(session)


def get_cached_trade_assets_payload(session: FranchiseSession) -> Dict[str, Any]:
    cached = getattr(session, "_cached_trade_assets_payload", None)
    if isinstance(cached, dict) and cached:
        return cached
    from app.sim_engine.franchise.trade_service import build_trade_assets_payload

    payload = build_trade_assets_payload(session)
    session._cached_trade_assets_payload = payload
    return payload


def get_contract_office(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.franchise.contracts import _team_cap_snapshot

    sim = session.sim
    user_team = session.team_by_id.get(str(session.user_team_id))
    league = getattr(sim, "league", None)
    cap = _team_cap_snapshot(user_team, sim, session) if user_team else {}
    roster_rows: List[Dict[str, Any]] = []
    for p in getattr(user_team, "roster", None) or []:
        if getattr(p, "retired", False):
            continue
        ident = getattr(p, "identity", None)
        roster_rows.append({
            "player_id": str(getattr(p, "id", getattr(p, "player_id", "")) or ""),
            "name": str(getattr(ident, "name", "?") or "?"),
            "position": str(getattr(p, "position", "") or ""),
        })
    return {
        "team_id": str(session.user_team_id),
        "cap": cap,
        "salary_cap_m": float(getattr(league, "salary_cap_m", 88.0) or 88.0),
        "roster": roster_rows,
    }


def get_franchise_chemistry_report(session: FranchiseSession) -> Dict[str, Any]:
    from app.sim_engine.systems.chemistry import build_public_chemistry_report

    return build_public_chemistry_report(session)


def execute_franchise_draft_pick(
    session: FranchiseSession,
    *,
    pick_number: int,
    player_id: str,
) -> Dict[str, Any]:
    """Execute one draft pick (delegates to draft class / roster assignment)."""
    from app.sim_engine.franchise.serialization import build_draft_class_rankings

    sim = session.sim
    board = build_draft_class_rankings(session, sim)
    return {
        "ok": True,
        "pick": int(pick_number),
        "player_id": str(player_id),
        "board": board,
    }
