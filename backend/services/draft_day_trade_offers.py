"""
Draft-day trade offer generation for the live entry draft floor.

Partners only climb when they have a real board priority still available who
might not fall to their slot. Packages follow the shared pick-value curve
(slot_curve_value + owned-pick sweeteners), not free future firsts for a
one-slot bump. Target identity is fogged: three names are shown, one is real.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional


def _team_name(session: Any, team_id: str) -> str:
    try:
        from services.franchise_entry_draft import _display_team

        return _display_team(session, team_id)
    except Exception:
        return str(team_id)


def _ordinal(n: int) -> str:
    return {1: "st", 2: "nd", 3: "rd"}.get(int(n), "th")


def _entry_key(entry: Optional[Dict[str, Any]]) -> str:
    if not entry:
        return ""
    return str(entry.get("key") or entry.get("prospect_id") or entry.get("id") or "")


def _pick_display(row: Dict[str, Any], overall_fallback: int = 0) -> str:
    label = str(row.get("display") or "").strip()
    if label:
        return label
    try:
        yr = int(row.get("year") or 0)
        rnd = int(row.get("round") or 0)
    except (TypeError, ValueError):
        yr, rnd = 0, 0
    if yr and rnd:
        return f"{yr} {rnd}{_ordinal(rnd)}"
    if overall_fallback:
        return f"#{overall_fallback}"
    return str(row.get("pick_id") or "pick")


def _partner_true_target(
    partner_board: List[Dict[str, Any]],
    available: List[Dict[str, Any]],
    *,
    partner_overall: int,
) -> Optional[Dict[str, Any]]:
    """Return the climber's must-have only when waiting is risky."""
    if not partner_board or not available:
        return None
    top_keys = {_entry_key(e) for e in available[:10] if _entry_key(e)}
    true_target = None
    for entry in partner_board[:5]:
        key = _entry_key(entry)
        if key and key in top_keys:
            true_target = entry
            break
    if true_target is None:
        return None
    try:
        pub_rank = int(
            true_target.get("rank")
            or true_target.get("public_rank")
            or true_target.get("final_rank")
            or 999
        )
    except (TypeError, ValueError):
        pub_rank = 999
    # If public consensus says he falls past their slot, they stay put.
    if pub_rank > partner_overall + 2:
        return None
    return true_target


def _public_rank_band(rank: int) -> str:
    if rank <= 15:
        return "lottery tier"
    if rank <= 32:
        return "first-round board"
    if rank <= 64:
        return "early day two"
    return "mid-board"


def _candidate_intel_row(entry: Dict[str, Any], *, need_pos: str, rng: random.Random, emphasis: bool) -> Dict[str, Any]:
    try:
        pub = int(entry.get("rank") or entry.get("public_rank") or entry.get("final_rank") or 99)
    except (TypeError, ValueError):
        pub = 99
    pos = str(entry.get("position") or "").upper()
    need_fit = bool(need_pos and pos and (pos == need_pos or (need_pos == "F" and pos in ("C", "LW", "RW", "W"))))
    buzz_pool = [
        "Private workout buzz",
        "Combine interviews went well",
        "Scouts like the toolkit",
        "Medical cleared — clubs circling",
        "Late stock riser on the floor",
        "Strong WJC tape in the room",
    ]
    if emphasis:
        buzz_pool = [
            "Multiple teams checked him twice today",
            "War room has him pinned on the short list",
            "Would be stunned if he lasts past the next run",
            "GM told insiders this is the guy",
        ] + buzz_pool
    return {
        "prospect_id": _entry_key(entry),
        "name": str(entry.get("name") or "Prospect"),
        "position": pos or None,
        "public_rank": pub if pub < 900 else None,
        "public_rank_band": _public_rank_band(pub),
        "need_fit": need_fit,
        "scout_buzz": rng.choice(buzz_pool),
    }


def _decoy_targets(
    true_target: Dict[str, Any],
    available: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    true_key = _entry_key(true_target)
    pool = [e for e in available[:14] if _entry_key(e) and _entry_key(e) != true_key]
    decoys = pool[:]
    rng.shuffle(decoys)
    decoys = decoys[:2]
    while len(decoys) < 2 and pool:
        # Extremely thin boards: pad from remaining available.
        extra = [
            e
            for e in available
            if _entry_key(e) not in {_entry_key(true_target), *(_entry_key(d) for d in decoys)}
        ]
        if not extra:
            break
        decoys.append(extra[0])
    trio = [true_target, *decoys[:2]]
    rng.shuffle(trio)
    return trio[:3]


def _offer_id(overall: int, partner: str, partner_overall: int, pick_a: str, pick_b: str) -> str:
    raw = f"{overall}:{partner}:{partner_overall}:{pick_a}:{pick_b}"
    return "dd-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _build_intel_lines(
    *,
    partner_name: str,
    partner_overall: int,
    overall: int,
    slots_moved: int,
    need_pos: str,
    urgency: str,
    gap: float,
    rng: random.Random,
) -> List[str]:
    lines = [
        f"{partner_name} is trying to jump {slots_moved} slot{'s' if slots_moved != 1 else ''} into #{overall}.",
    ]
    if need_pos:
        lines.append(f"Room intel ties the climb to {need_pos} depth — one name is real, the rest is smoke.")
    if urgency == "high":
        lines.append("Clock pressure: their staff is on the phone with the league table.")
    elif urgency == "medium":
        lines.append("They will walk if another club beats them to this window.")
    if gap <= 6:
        lines.append("Package is tight — mostly a straight swap with light sweetener capital.")
    elif gap >= 14:
        lines.append("They are paying up with future capital to secure the prospect tier.")
    if rng.random() < 0.45:
        lines.append("Accepting moves you to a better value pocket on the chart.")
    return lines[:4]


def _preview_trade_value_meter(
    *,
    user_id: str,
    partner_id: str,
    pick_a: str,
    pick_b: str,
    sweetener_ids: List[str],
    league: Any,
    team_by_id: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """Chart-priced totals for the trade-down UI bar (user on clock)."""
    try:
        from app.sim_engine.trades.trade_evaluator import evaluate_trade_package
    except Exception:
        return {}
    package = {
        user_id: [{"type": "pick", "id": pick_b, "team": partner_id}],
        partner_id: [{"type": "pick", "id": pick_a, "team": user_id}],
    }
    for sid in sweetener_ids:
        pid = str(sid or "").strip()
        if pid and pid not in (pick_a, pick_b):
            package[user_id].append({"type": "pick", "id": pid, "team": partner_id})
    try:
        evaluation = evaluate_trade_package(
            package,
            league=league,
            team_by_id=team_by_id,
            context={**(ctx or {}), "draft_day_trade": True},
            user_team_id=user_id,
        )
    except Exception:
        return {}
    vb = dict(evaluation.get("value_breakdown") or {})
    user_row = vb.get(str(user_id)) or {}
    partner_row = vb.get(str(partner_id)) or {}
    send_val = float(user_row.get("outgoing_total") or 0.0)
    recv_val = float(user_row.get("incoming_total") or 0.0)
    gap = float(evaluation.get("fairness_gap") or 0.0)
    return {
        "user_send_value": round(send_val, 2),
        "user_receive_value": round(recv_val, 2),
        "fairness_gap": round(gap, 2),
        "partner_net": round(float(partner_row.get("net") or 0.0), 2),
        "can_execute": bool(evaluation.get("can_execute")),
        "value_grade": evaluation.get("value_grade") or evaluation.get("grade"),
    }


def generate_draft_day_trade_offers(
    session: Any,
    state: Optional[Dict[str, Any]] = None,
    *,
    max_offers: int = 2,
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

    early = overall <= 32
    user_on_clock = on_clock == user_id
    if not early and not user_on_clock:
        return list(state.get("trade_offers") or [])[:max_offers]

    try:
        from services.franchise_entry_draft import (
            _available_entries,
            _draft_swap_sweeteners,
            _ensure_draft_cache,
            _partner_willing_to_climb,
            build_known_pick_slots,
            build_team_draft_board,
            calculate_team_needs,
        )
        from services.franchise_sim import get_cached_draft_class_rankings
        from app.sim_engine.trades.trade_value import slot_curve_value
        from app.sim_engine.trades.cpu_trade_proposer import build_league_trade_context
    except Exception:
        return []

    board = get_cached_draft_class_rankings(session, session.sim)
    cache = _ensure_draft_cache(session, board)
    available = _available_entries(state, board)
    if len(available) < 5:
        return []

    league = getattr(getattr(session, "sim", None), "league", None)
    team_by_id = dict(getattr(session, "team_by_id", None) or {})
    rng = getattr(getattr(session, "sim", None), "rng", None)
    if rng is None:
        rng = random.Random(int(overall) * 997 + hash(on_clock) % 10007)

    ctx: Dict[str, Any] = {}
    try:
        if league is not None:
            ctx = build_league_trade_context(
                league,
                calendar_cursor=int(getattr(session, "calendar_cursor", 0) or 0),
                regular_season_last_index=int(getattr(session, "nhl_regular_season_last_index", 192) or 192),
                season_year=int(getattr(session, "season_calendar_year", 2025) or 2025),
            )
            ctx["draft_day_trade"] = True
            ctx["known_pick_slots"] = build_known_pick_slots(session)
    except Exception:
        ctx = {"draft_day_trade": True, "known_pick_slots": {}}

    offers: List[Dict[str, Any]] = []
    seen_partners: set = set()
    on_clock_value = float(slot_curve_value(overall))
    on_clock_pick_id = str(slot.get("pick_id") or "")

    # Later slots that might pay to climb — never invent interest without a target.
    # Keep the look-ahead modest so only nearby climbers show up.
    lookahead = order[overall : min(len(order), overall + 14)]
    forced_one = False
    for future in lookahead:
        partner = str(future.get("team_id") or "")
        if not partner or partner == on_clock or partner in seen_partners:
            continue
        partner_overall = int(future.get("overall_pick") or 0)
        if partner_overall <= overall:
            continue

        slots_moved = partner_overall - overall
        gap = abs(on_clock_value - float(slot_curve_value(partner_overall)))
        # Tiny slides (#53→#54 / +1–2 with tiny chart gap) are not trade packages.
        if slots_moved <= 2 and gap < 5.0:
            continue
        if gap < 3.5 and slots_moved < 4:
            continue

        partner_board = build_team_draft_board(session, partner, available[:40], cache=cache)
        true_target = _partner_true_target(
            partner_board,
            available,
            partner_overall=partner_overall,
        )
        if true_target is None:
            continue

        if not user_on_clock and not _partner_willing_to_climb(
            session,
            partner,
            overall=overall,
            partner_overall=partner_overall,
            true_target=true_target,
            rng=rng,
        ):
            continue
        # User on the clock still needs selective partners — but always allow at
        # least one climber so the Trade Down desk is never empty at pick time.
        willing = _partner_willing_to_climb(
            session,
            partner,
            overall=overall,
            partner_overall=partner_overall,
            true_target=true_target,
            rng=rng,
        )
        if user_on_clock and offers and not willing and not forced_one:
            forced_one = True
            willing = True
        elif not user_on_clock and not willing:
            continue
        elif user_on_clock and offers and not willing:
            continue

        partner_pick_id = str(future.get("pick_id") or "")
        exclude = {pid for pid in (on_clock_pick_id, partner_pick_id) if pid}
        sweeteners: List[Dict[str, Any]] = []
        if league is not None:
            # Use the real chart gap — never inflate into a free future 2nd.
            sweet = _draft_swap_sweeteners(
                league,
                team_by_id.get(partner),
                ctx=ctx,
                gap=float(gap),
                exclude=exclude,
            )
            if sweet is None:
                # Cannot cover fairly with late-round capital — no offer.
                continue
            sweeteners = list(sweet or [])
            # Near-equal slots may swap with no add-on; still require a real climb motive.
            if not sweeteners and gap >= 6.0:
                continue
        elif gap > 18.0:
            # Unit tests / no registry: refuse absurd gaps without capital.
            continue

        future_round = int(future.get("round") or 1)
        assets_out = f"#{partner_overall} ({future_round}{_ordinal(future_round)} this year)"
        sweetener_ids = [str(r.get("pick_id") or "") for r in sweeteners if r.get("pick_id")]
        sweetener_labels = [_pick_display(r) for r in sweeteners]
        incoming = [assets_out, *sweetener_labels]

        candidates = _decoy_targets(true_target, available, rng)
        candidate_rows = [
            {
                "prospect_id": _entry_key(c),
                "name": str(c.get("name") or "Prospect"),
                "position": str(c.get("position") or ""),
            }
            for c in candidates
            if _entry_key(c)
        ]
        ask_label = f"#{overall} pick"
        needs = list(
            (state.get("team_needs_snapshot") or {}).get(partner)
            or calculate_team_needs(session, partner)
        )
        need_pos = ""
        if needs:
            need_pos = str(needs[0].get("position") if isinstance(needs[0], dict) else needs[0] or "").upper()

        grade = "Fair" if gap <= 8 else ("Lean +" if gap <= 14 else "Plus")
        urgency = "high" if overall <= 15 or user_on_clock else "medium"
        true_key = _entry_key(true_target)
        candidate_intel = [
            _candidate_intel_row(
                c,
                need_pos=need_pos,
                rng=rng,
                emphasis=_entry_key(c) == true_key,
            )
            for c in candidates
        ]
        intel_lines = _build_intel_lines(
            partner_name=_team_name(session, partner),
            partner_overall=partner_overall,
            overall=overall,
            slots_moved=slots_moved,
            need_pos=need_pos,
            urgency=urgency,
            gap=float(gap),
            rng=rng,
        )
        value_meter: Dict[str, Any] = {}
        if user_on_clock and league is not None and on_clock == user_id:
            value_meter = _preview_trade_value_meter(
                user_id=user_id,
                partner_id=partner,
                pick_a=on_clock_pick_id,
                pick_b=partner_pick_id,
                sweetener_ids=sweetener_ids,
                league=league,
                team_by_id=team_by_id,
                ctx=ctx,
            )
        offer_style = "desperate" if urgency == "high" and slots_moved <= 4 else "standard"
        if float(gap) >= 12 and sweeteners:
            offer_style = "pay_up"
        offers.append(
            {
                "offer_id": _offer_id(overall, partner, partner_overall, on_clock_pick_id, partner_pick_id),
                "from_team_id": partner,
                "team_name": _team_name(session, partner),
                "to_team_id": on_clock,
                "partner_overall_pick": partner_overall,
                "slots_moved": slots_moved,
                "partner_pick_id": partner_pick_id,
                "on_clock_pick_id": on_clock_pick_id,
                "on_clock_overall_pick": overall,
                "offer_text": f"Trade down to #{partner_overall}",
                "assets_in": " · ".join(incoming),
                "incoming_assets": incoming,
                "outgoing_assets": [ask_label],
                "sweetener_pick_ids": sweetener_ids,
                "sweetener_labels": sweetener_labels,
                "slot_value_gap": round(gap, 2),
                # Fogged: FE shows candidates only — do not surface the true name.
                "target_candidates": candidate_rows,
                "candidate_intel": candidate_intel,
                "intel_lines": intel_lines,
                "pitch_headline": f"Move back to #{partner_overall} and add capital",
                "true_target_prospect_id": true_key,
                "target_prospect_id": true_key,
                "target_prospect_name": None,
                "value": grade,
                "value_grade": "B" if gap <= 8 else "B+",
                "risk": "Only one rumored name is their real target",
                "urgency": urgency,
                "urgency_label": "On the phone" if urgency == "high" else "Monitoring",
                "reason": "board_priority" if true_target else "trade_down",
                "reason_headline": "Board priority still on the table",
                "philosophy_fit": True,
                "user_on_clock": user_on_clock,
                "trade_down": True,
                "offer_style": offer_style,
                "value_meter": value_meter,
                "partner_board_rank": (
                    partner_board[0].get("team_board_rank") if partner_board else None
                ),
                "need_position": need_pos or None,
            }
        )
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
