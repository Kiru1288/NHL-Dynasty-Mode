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
        # Protect: revert ownership to original via registry transfer when possible.
        try:
            from app.sim_engine.trades.trade_pick_registry import transfer_pick

            transfer_pick(league, str(pick_id), original)
        except Exception:
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
    try:
        from app.sim_engine.trades.trade_pick_registry import reconcile_pick_registry_consistency

        reconcile_pick_registry_consistency(league)
    except Exception:
        pass
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
                old_round = int(row.get("round") or 1)
                if new_round != old_round:
                    _rekey_pick_row(
                        league,
                        old_pick_id=str(pick_id),
                        row=row,
                        new_year=int(row.get("year") or draft_year),
                        new_round=new_round,
                    )
                else:
                    row["round"] = new_round
                row["condition_status"] = "upgraded"
                events.append({"pick_id": pick_id, "event": "condition_upgraded", "round": new_round})
            elif cond.get("type") == "defer" and met:
                old_year = int(row.get("year") or draft_year)
                new_year = int(cond.get("defer_to_year") or (old_year + 1))
                row["deferred"] = True
                if new_year != old_year:
                    _rekey_pick_row(
                        league,
                        old_pick_id=str(pick_id),
                        row=row,
                        new_year=new_year,
                        new_round=int(row.get("round") or 1),
                    )
                else:
                    row["year"] = new_year
                row["condition_status"] = "deferred"
                events.append({"pick_id": pick_id, "event": "condition_deferred", "year": new_year})
            elif not met and int(row.get("year") or 0) == int(draft_year):
                row["condition_status"] = "unmet"
                events.append({"pick_id": pick_id, "event": "condition_unmet"})
    try:
        from app.sim_engine.trades.trade_pick_registry import reconcile_pick_registry_consistency

        reconcile_pick_registry_consistency(league)
    except Exception:
        pass
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


def _rekey_pick_row(
    league: Any,
    *,
    old_pick_id: str,
    row: Dict[str, Any],
    new_year: int,
    new_round: int,
) -> str:
    """Move a registry row to a canonical pick_id when year/round changes."""
    from app.sim_engine.trades.trade_asset import canonical_pick_id

    orig = str(row.get("original_team_id") or "")
    owner = str(row.get("current_owner_team_id") or orig)
    new_id = canonical_pick_id(int(new_year), int(new_round), orig)
    reg = _reg(league)
    if new_id == old_pick_id:
        row["year"] = int(new_year)
        row["round"] = int(new_round)
        return new_id

    payload = dict(row)
    payload["pick_id"] = new_id
    payload["year"] = int(new_year)
    payload["round"] = int(new_round)
    payload["current_owner_team_id"] = owner
    payload["rekeyed_from"] = old_pick_id

    existing = reg.get(new_id)
    if isinstance(existing, dict) and not existing.get("resolved"):
        # Prefer keeping explicit ownership from the deferred/upgraded pick.
        existing["current_owner_team_id"] = owner
        existing["deferred"] = payload.get("deferred", existing.get("deferred"))
        existing["conditions"] = payload.get("conditions", existing.get("conditions"))
        existing["protection"] = payload.get("protection", existing.get("protection"))
        existing["rekeyed_from"] = old_pick_id
    else:
        reg[new_id] = payload

    # Retire the old key so draft order / ownership cannot double-consume it.
    old = reg.get(old_pick_id)
    if isinstance(old, dict):
        old["resolved"] = True
        old["resolved_reason"] = "rekeyed"
        old["migrated_to"] = new_id
    return new_id


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
        from app.sim_engine.trades.trade_pick_registry import reconcile_pick_registry_consistency

        reconcile_pick_registry_consistency(league)
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
