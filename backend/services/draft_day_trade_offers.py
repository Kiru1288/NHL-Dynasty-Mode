"""
Draft-day trade offer generation for the live entry draft floor.

Offers are only generated when there is a real organizational reason to move a pick.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _team_name(session: Any, team_id: str) -> str:
    try:
        from services.franchise_entry_draft import _display_team

        return _display_team(session, team_id)
    except Exception:
        return str(team_id)


def generate_draft_day_trade_offers(
    session: Any,
    state: Optional[Dict[str, Any]] = None,
    *,
    max_offers: int = 3,
) -> List[Dict[str, Any]]:
    state = state or getattr(session, "draft_state", None) or {}
    if not state.get("draft_started") or state.get("draft_completed"):
        return []

    overall = int(state.get("overall_pick") or 1)
    order = list(state.get("draft_order") or [])
    if overall > len(order):
        return []
    slot = order[overall - 1]
    on_clock = str(slot.get("team_id") or state.get("current_team_id") or "")
    user_id = str(getattr(session, "user_team_id", "") or "")
    if not on_clock:
        return []

    # Only generate when someone other than an idle mid-pack situation has leverage:
    # user on clock, or early-round scarcity, or clear positional desperation.
    early = overall <= 32
    user_on_clock = on_clock == user_id
    if not early and not user_on_clock:
        return list(state.get("trade_offers") or [])[:max_offers]

    try:
        from services.franchise_entry_draft import (
            _available_entries,
            _ensure_draft_cache,
            build_team_draft_board,
            calculate_team_needs,
        )
        from services.franchise_sim import get_cached_draft_class_rankings
    except Exception:
        return []

    board = get_cached_draft_class_rankings(session, session.sim)
    cache = _ensure_draft_cache(session, board)
    available = _available_entries(state, board)
    if len(available) < 5:
        return []

    top = available[:8]
    offers: List[Dict[str, Any]] = []
    seen_partners: set = set()

    # Rival teams later in the round that desperately need a position the top remaining fills.
    lookahead = order[overall : min(len(order), overall + 12)]
    for future in lookahead:
        partner = str(future.get("team_id") or "")
        if not partner or partner == on_clock or partner in seen_partners:
            continue
        needs = list((state.get("team_needs_snapshot") or {}).get(partner) or calculate_team_needs(session, partner))
        if not needs:
            continue
        need_pos = str(needs[0].get("position") if isinstance(needs[0], dict) else needs[0] or "").upper()
        target = next((e for e in top if str(e.get("position") or "").upper() == need_pos), None)
        if target is None and overall > 10:
            continue
        partner_board = build_team_draft_board(session, partner, available[:40], cache=cache)
        urgency = "high" if overall <= 15 else "medium"
        future_round = int(future.get("round") or 1)
        pick_in = int(future.get("pick_in_round") or 1)
        assets_out = f"{future_round}{ {1:'st',2:'nd',3:'rd'}.get(future_round,'th') } this year"
        # Ask for an upgrade relative to sitting still
        ask_label = f"#{overall} pick"
        offers.append({
            "from_team_id": partner,
            "team_name": _team_name(session, partner),
            "to_team_id": on_clock,
            "offer_text": (
                f"Wants {ask_label} to take {target.get('name') if target else 'a top remaining prospect'}; "
                f"offers {assets_out} + future 2nd"
            ),
            "assets_in": assets_out,
            "incoming_assets": [assets_out, f"{int(state.get('draft_year') or 0) + 1} 2nd"],
            "outgoing_assets": [ask_label],
            "target_prospect_id": target.get("key") if target else None,
            "target_prospect_name": target.get("name") if target else None,
            "value": "Fair+" if urgency == "high" else "Competitive",
            "value_grade": "B+" if urgency == "high" else "B",
            "risk": f"Moves back; loses shot at {need_pos or 'BPA'}",
            "urgency": urgency,
            "reason": "positional_urgency" if target else "pick_value",
            "philosophy_fit": True,
            "user_on_clock": user_on_clock,
            "partner_board_rank": (partner_board[0].get("team_board_rank") if partner_board else None),
        })
        seen_partners.add(partner)
        if len(offers) >= max_offers:
            break

    state["trade_offers"] = offers
    state["draft_day_trade_offers"] = offers
    state["pick_trade_offers"] = offers
    try:
        session.draft_state = state
    except Exception:
        pass
    return offers
