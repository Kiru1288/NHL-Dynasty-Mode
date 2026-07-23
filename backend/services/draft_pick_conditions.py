"""
Resolve pick protections / conditions before draft-order creation.

Protection and conditions must not remain valuation-only metadata. This service rolls
deferred and converted picks forward into the live draft_pick_registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _reg(league: Any) -> Dict[str, Any]:
    reg = getattr(league, "draft_pick_registry", None)
    return reg if isinstance(reg, dict) else {}


def resolve_pick_protections(league: Any, *, draft_year: int, lottery_order: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Apply lottery / top-10 style protections; convert protected picks when triggered."""
    events: List[Dict[str, Any]] = []
    lottery_set = set(str(x) for x in (lottery_order or [])[:10])
    reg = _reg(league)
    for pick_id, row in list(reg.items()):
        if not isinstance(row, dict) or row.get("resolved"):
            continue
        if int(row.get("year") or 0) != int(draft_year):
            continue
        protection = row.get("protection")
        if protection in (None, "", "none", "unprotected"):
            continue
        original = str(row.get("original_team_id") or "")
        current = str(row.get("current_owner_team_id") or original)
        if current == original:
            continue
        prot = str(protection).lower()
        triggered = False
        if "lottery" in prot and original in lottery_set:
            triggered = True
        if "top_10" in prot or "top-10" in prot or "top10" in prot:
            # If original still owns a top-10 slot in lottery outcome, protection fires.
            if original in set(str(x) for x in (lottery_order or [])[:10]):
                triggered = True
        if not triggered:
            row["protection_status"] = "cleared"
            continue
        # Protect: revert ownership to original; convert outbound compensatory when configured.
        row["current_owner_team_id"] = original
        row["protection_status"] = "triggered"
        conversion = row.get("protection_converts_to") or {}
        events.append({
            "pick_id": pick_id,
            "event": "protection_triggered",
            "original_team_id": original,
            "former_owner_team_id": current,
            "converts_to": conversion or None,
        })
        if isinstance(conversion, dict) and conversion.get("year") and conversion.get("round"):
            _spawn_compensatory_pick(
                league,
                from_team_id=original,
                to_team_id=current,
                year=int(conversion["year"]),
                round_num=int(conversion["round"]),
                source_pick_id=pick_id,
            )
    return events


def resolve_pick_conditions(league: Any, *, draft_year: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    reg = _reg(league)
    for pick_id, row in list(reg.items()):
        if not isinstance(row, dict) or row.get("resolved"):
            continue
        conditions = row.get("conditions")
        if not conditions:
            continue
        # Support list or single dict; unmet → mark; met upgrade adjusts round.
        cond_list = conditions if isinstance(conditions, list) else [conditions]
        for cond in cond_list:
            if not isinstance(cond, dict):
                continue
            met = bool(cond.get("met"))
            if cond.get("type") == "round_upgrade" and met:
                new_round = int(cond.get("upgrade_to_round") or row.get("round") or 1)
                row["round"] = new_round
                row["condition_status"] = "upgraded"
                events.append({"pick_id": pick_id, "event": "condition_upgraded", "round": new_round})
            elif cond.get("type") == "defer" and met:
                row["deferred"] = True
                row["year"] = int(cond.get("defer_to_year") or (int(row.get("year") or draft_year) + 1))
                row["condition_status"] = "deferred"
                events.append({"pick_id": pick_id, "event": "condition_deferred", "year": row["year"]})
            elif not met and int(row.get("year") or 0) == int(draft_year):
                row["condition_status"] = "unmet"
                events.append({"pick_id": pick_id, "event": "condition_unmet"})
    return events


def roll_deferred_picks_forward(league: Any, *, draft_year: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    reg = _reg(league)
    for pick_id, row in list(reg.items()):
        if not isinstance(row, dict):
            continue
        if row.get("deferred") and int(row.get("year") or 0) == int(draft_year):
            row["deferred"] = False
            row["condition_status"] = "active"
            events.append({"pick_id": pick_id, "event": "deferred_activated", "year": draft_year})
    return events


def _spawn_compensatory_pick(
    league: Any,
    *,
    from_team_id: str,
    to_team_id: str,
    year: int,
    round_num: int,
    source_pick_id: str,
) -> None:
    try:
        from app.sim_engine.trades.trade_asset import canonical_pick_id
        from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry

        ensure_draft_pick_registry(league, start_year=int(year))
        pick_id = canonical_pick_id(int(year), int(round_num), str(from_team_id))
        reg = _reg(league)
        if pick_id not in reg:
            reg[pick_id] = {
                "pick_id": pick_id,
                "year": int(year),
                "round": int(round_num),
                "original_team_id": str(from_team_id),
                "current_owner_team_id": str(to_team_id),
                "resolved": False,
                "source": "protection_conversion",
                "source_pick_id": source_pick_id,
            }
        else:
            reg[pick_id]["current_owner_team_id"] = str(to_team_id)
    except Exception:
        pass


def finalize_draft_pick_registry(league: Any, *, draft_year: int, lottery_order: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry

        ensure_draft_pick_registry(league, start_year=int(draft_year))
    except Exception:
        pass
    events: List[Dict[str, Any]] = []
    events.extend(roll_deferred_picks_forward(league, draft_year=draft_year))
    events.extend(resolve_pick_conditions(league, draft_year=draft_year))
    events.extend(resolve_pick_protections(league, draft_year=draft_year, lottery_order=lottery_order))
    return {"draft_year": draft_year, "events": events, "registry_size": len(_reg(league))}
