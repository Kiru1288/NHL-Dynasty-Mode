"""
Atomic trade execution after validation.
"""

from __future__ import annotations

import uuid
import copy
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.trades.trade_asset import (
    DraftPickTradeAsset,
    PlayerTradeAsset,
    RetainedSalaryRecord,
    TradePackage,
    find_player_on_team_roster,
    player_display_name,
    team_id_of,
)
from app.sim_engine.trades.trade_evaluator import evaluate_trade_package
from app.sim_engine.trades.trade_history import append_trade_record
from app.sim_engine.trades.trade_pick_registry import (
    audit_pick_registry_integrity,
    draft_year_from_context,
    ensure_draft_pick_registry,
    sync_owned_pick_ids_from_registry,
    transfer_pick,
)
from app.sim_engine.trades.trade_rules import _contract_years_for_retention
from app.sim_engine.economy.cap_engine import player_cap_hit_millions


def _append_retained_record(team: Any, record: RetainedSalaryRecord, season_label: Optional[str]) -> None:
    rows = getattr(team, "retained_salary_records", None)
    if not isinstance(rows, list):
        rows = []
    rows.append(
        {
            "player_id": record.player_id,
            "player_name": record.player_name,
            "benefiting_team_id": record.benefiting_team_id,
            "retained_pct": record.retained_pct,
            "amount_m": record.retained_cap_hit_m,
            "cap_hit_m": record.retained_cap_hit_m,
            "seasons_remaining": record.seasons_remaining,
            "season": season_label,
        }
    )
    setattr(team, "retained_salary_records", rows)


def _org_list_attrs() -> Tuple[str, ...]:
    return ("roster", "ahl_roster", "echl_roster", "prospect_pool")


def _snapshot_team_org_lists(team: Any) -> Dict[str, List[Any]]:
    return {attr: list(getattr(team, attr, None) or []) for attr in _org_list_attrs()}


def _restore_team_org_lists(team: Any, snap: Dict[str, List[Any]]) -> None:
    for attr, rows in snap.items():
        setattr(team, attr, list(rows))


def _purge_player_id_from_team_lists(team: Any, player_id: str) -> int:
    """Remove every roster/affiliate/scratch reference to player_id on one club."""
    pid = str(player_id or "")
    if not pid or team is None:
        return 0
    removed = 0
    for attr in _org_list_attrs():
        rows = list(getattr(team, attr, None) or [])
        keep = [p for p in rows if str(getattr(p, "id", "")) != pid]
        if len(keep) != len(rows):
            removed += len(rows) - len(keep)
            setattr(team, attr, keep)
    scratches = list(getattr(team, "scratches", None) or [])
    if scratches:
        keep_sc = []
        for entry in scratches:
            eid = str(getattr(entry, "id", "") or entry or "")
            if eid == pid:
                removed += 1
                continue
            keep_sc.append(entry)
        try:
            setattr(team, "scratches", keep_sc)
        except Exception:
            pass
    return removed


def _purge_player_from_other_organizations(
    team_by_id: Dict[str, Any],
    player_id: str,
    *,
    keep_team_id: str,
) -> None:
    """
    After a trade move, ensure the player exists on only the acquiring club.

    Incomplete removals (duplicate org copies, stale scratches) previously let the
    same identity dress for two NHL clubs and accumulate ~164 GP in an 82-game season.
    """
    pid = str(player_id or "")
    keep = str(keep_team_id or "")
    if not pid:
        return
    for tid, tm in (team_by_id or {}).items():
        if str(tid) == keep or str(team_id_of(tm) if tm is not None else "") == keep:
            continue
        _purge_player_id_from_team_lists(tm, pid)


def _resolve_trade_destination_attr(loc: str, player: Any) -> str:
    """Deterministic post-trade assignment (validate before commit).

    Rules:
    - NHL roster source → NHL roster
    - AHL / ECHL / prospect with an NHL SPC → receiving AHL (preserve affiliate level)
    - Unsigned prospect (draft rights) → receiving prospect pool
    - Otherwise → NHL roster
    """
    from app.sim_engine.trades.trade_asset import player_holds_nhl_spc

    if loc == "nhl":
        return "roster"
    if loc in ("ahl", "echl", "prospect") and player_holds_nhl_spc(player):
        return "ahl_roster"
    if loc == "prospect":
        return "prospect_pool"
    return "roster"


def _sync_assignment_flags(player: Any, dest_attr: str) -> None:
    if dest_attr == "roster":
        player.in_minors = False
        player.is_buried = False
        player.roster_location = "nhl"
    elif dest_attr == "ahl_roster":
        player.in_minors = True
        player.roster_location = "ahl"
    elif dest_attr == "echl_roster":
        player.in_minors = True
        player.roster_location = "echl"
    elif dest_attr == "prospect_pool":
        player.in_minors = True
        player.roster_location = "prospect"


def _move_reserve_list_entry(source: Any, acq: Any, player_id: str, acquiring_team_id: Any) -> None:
    """Carry the unsigned-rights reserve row across with the player."""
    src_rows = list(getattr(source, "reserve_list", None) or [])
    moved = [r for r in src_rows if isinstance(r, dict) and str(r.get("player_id") or "") == player_id]
    if not moved:
        return
    setattr(source, "reserve_list", [r for r in src_rows if r not in moved])
    dest_rows = list(getattr(acq, "reserve_list", None) or [])
    for row in moved:
        row["team_id"] = str(acquiring_team_id)
        row["rights_team_id"] = str(acquiring_team_id)
        dest_rows.append(row)
    setattr(acq, "reserve_list", dest_rows)


def _apply_player_move(
    asset: PlayerTradeAsset,
    team_by_id: Dict[str, Any],
    *,
    season_label: Optional[str],
    moved_players: List[Dict[str, Any]],
    retained_records: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    from app.sim_engine.trades.trade_asset import find_player_in_organization

    source = team_by_id.get(str(asset.source_team_id)) or team_by_id.get(asset.source_team_id)
    acq = team_by_id.get(str(asset.acquiring_team_id)) or team_by_id.get(asset.acquiring_team_id)
    if source is None or acq is None:
        raise ValueError(f"Teams missing for player move {asset.player_id}")

    player, loc, idx = find_player_in_organization(source, asset.player_id)
    if player is None or idx < 0:
        raise ValueError(f"Player {asset.player_id} not in source organization during execution")

    list_attr = {
        "nhl": "roster",
        "ahl": "ahl_roster",
        "echl": "echl_roster",
        "prospect": "prospect_pool",
    }.get(loc, "roster")
    src_list = list(getattr(source, list_attr, None) or [])
    if idx >= len(src_list) or str(getattr(src_list[idx], "id", "")) != str(asset.player_id):
        player, loc, idx = find_player_in_organization(source, asset.player_id)
        list_attr = {
            "nhl": "roster",
            "ahl": "ahl_roster",
            "echl": "echl_roster",
            "prospect": "prospect_pool",
        }.get(loc, "roster")
        src_list = list(getattr(source, list_attr, None) or [])
        if player is None or idx < 0 or idx >= len(src_list):
            raise ValueError(f"Player {asset.player_id} list index invalid during execution")

    dest_attr = _resolve_trade_destination_attr(loc, src_list[idx])
    # Validate destination before mutating source.
    if not hasattr(acq, dest_attr) or getattr(acq, dest_attr, None) is None:
        setattr(acq, dest_attr, [])

    player = src_list.pop(idx)
    setattr(source, list_attr, src_list)

    dest_list = list(getattr(acq, dest_attr) or [])
    dest_list.append(player)
    setattr(acq, dest_attr, dest_list)
    try:
        _sync_assignment_flags(player, dest_attr)
    except Exception:
        pass

    # Belt-and-suspenders: drop any leftover copies on every other club.
    _purge_player_from_other_organizations(
        team_by_id,
        str(asset.player_id),
        keep_team_id=str(asset.acquiring_team_id),
    )
    # Also scrub non-destination lists on the acquiring club (e.g. duplicate
    # prospect_pool entry after an NHL move).
    for attr in _org_list_attrs():
        if attr == dest_attr:
            continue
        rows = list(getattr(acq, attr, None) or [])
        keep = [p for p in rows if str(getattr(p, "id", "")) != str(asset.player_id)]
        if len(keep) != len(rows):
            setattr(acq, attr, keep)

    for field in ("team_id", "current_team_id", "last_team_id"):
        try:
            if field == "last_team_id":
                setattr(player, field, asset.source_team_id)
            else:
                setattr(player, field, asset.acquiring_team_id)
        except Exception:
            pass

    # Draft rights follow the player, otherwise the prospect still reads as
    # belonging to the club that drafted him on every rights surface.
    if getattr(player, "nhl_rights_team_id", None) is not None:
        for field in ("nhl_rights_team_id", "rights_team_id"):
            try:
                setattr(player, field, asset.acquiring_team_id)
            except Exception:
                pass
        _move_reserve_list_entry(source, acq, str(asset.player_id), asset.acquiring_team_id)

    ctx = context or {}
    cursor = int(ctx.get("calendar_cursor", 0) or 0)
    for field, val in (
        ("last_acquired_day", cursor),
        ("last_acquired_date", ctx.get("calendar_iso")),
        ("acquired_from_team_id", asset.source_team_id),
        ("acquired_via_trade", True),
        ("acquired_via_trade_season", ctx.get("season_year")),
    ):
        try:
            setattr(player, field, val)
        except Exception:
            pass

    stats = (context or {}).get("player_season_stats") if isinstance(context, dict) else None
    if isinstance(stats, dict):
        row = stats.get(str(asset.player_id))
        if isinstance(row, dict):
            row["team_id"] = str(asset.acquiring_team_id)

    pname = player_display_name(player)
    moved_players.append(
        {
            "asset_type": "player",
            "asset_id": asset.player_id,
            "player_name": pname,
            "source_team_id": asset.source_team_id,
            "acquiring_team_id": asset.acquiring_team_id,
            "applied": True,
            "retained_pct": asset.retained_pct,
            "from_level": loc,
            "to_level": {"roster": "nhl", "ahl_roster": "ahl", "prospect_pool": "prospect"}.get(dest_attr, "nhl"),
        }
    )

    if asset.retained_pct > 0:
        # Cap charge stays with source; SPC / 50-slot follows the player to acquiring.
        cap_hit = player_cap_hit_millions(player)
        retained_m = cap_hit * (asset.retained_pct / 100.0)
        rec = RetainedSalaryRecord(
            player_id=asset.player_id,
            player_name=pname,
            retaining_team_id=asset.source_team_id,
            benefiting_team_id=asset.acquiring_team_id,
            original_cap_hit_m=cap_hit,
            retained_pct=asset.retained_pct,
            retained_cap_hit_m=round(retained_m, 3),
            seasons_remaining=max(1, _contract_years_for_retention(player)),
        )
        _append_retained_record(source, rec, season_label)
        retained_records.append(
            {
                "player_id": rec.player_id,
                "player_name": rec.player_name,
                "retaining_team_id": rec.retaining_team_id,
                "benefiting_team_id": rec.benefiting_team_id,
                "retained_pct": rec.retained_pct,
                "retained_cap_hit_m": rec.retained_cap_hit_m,
            }
        )


def _apply_pick_move(
    asset: DraftPickTradeAsset,
    league: Any,
    moved_picks: List[Dict[str, Any]],
) -> None:
    row = transfer_pick(league, asset.pick_id, asset.acquiring_team_id)
    moved_picks.append(
        {
            "asset_type": "pick",
            "asset_id": asset.pick_id,
            "source_team_id": asset.source_team_id,
            "acquiring_team_id": asset.acquiring_team_id,
            "applied": True,
            "year": row.get("year"),
            "round": row.get("round"),
            "original_team_id": row.get("original_team_id") or getattr(asset, "original_team_id", None) or "",
            "display_name": (
                f"{row.get('year')} Round {row.get('round')}"
                if row.get("year") and row.get("round")
                else f"Pick {asset.pick_id}"
            ),
        }
    )


def execute_validated_trade(
    evaluation: Dict[str, Any],
    *,
    league: Any,
    team_by_id: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    user_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a trade that has already been evaluated. Re-validates before mutating."""
    assets_by_team = {}
    package: TradePackage = evaluation.get("_package")
    if package is None:
        raise ValueError("Evaluation missing normalized package")

    for tid in package.participating_team_ids:
        assets_by_team[tid] = package.assets_by_team.get(tid, [])

    fresh = evaluate_trade_package(
        assets_by_team,
        league=league,
        team_by_id=team_by_id,
        context=context,
        user_team_id=user_team_id,
    )
    if not fresh.get("can_execute"):
        reasons = fresh.get("rejection_reasons") or ["Trade failed validation"]
        raise ValueError("; ".join(str(r) for r in reasons))
    if not fresh.get("accepted"):
        reasons = fresh.get("rejection_reasons") or ["Trade rejected by team evaluation"]
        raise ValueError("; ".join(str(r) for r in reasons))

    package = fresh["_package"]
    ctx = context or {}
    season_label = None
    if ctx.get("season_year"):
        y = int(ctx["season_year"])
        season_label = f"{y}-{(y + 1) % 100:02d}"

    ensure_draft_pick_registry(league, start_year=draft_year_from_context(ctx, league=league))

    # Snapshot mutable org state for full rollback (NHL + affiliates + retention).
    snapshots: Dict[str, Dict[str, List[Any]]] = {}
    retained_snapshots: Dict[str, List[Any]] = {}
    registry_snapshot = copy.deepcopy(dict(getattr(league, "draft_pick_registry", {}) or {}))
    owned_pick_ids_snapshot: Dict[str, List[str]] = {}
    for tid, tm in team_by_id.items():
        snapshots[tid] = _snapshot_team_org_lists(tm)
        retained_snapshots[tid] = list(getattr(tm, "retained_salary_records", None) or [])
        owned_pick_ids_snapshot[tid] = list(getattr(tm, "owned_pick_ids", None) or [])

    moved_players: List[Dict[str, Any]] = []
    moved_picks: List[Dict[str, Any]] = []
    retained_records: List[Dict[str, Any]] = []

    def _rollback_all() -> None:
        for tid, tm in team_by_id.items():
            if tid in snapshots:
                _restore_team_org_lists(tm, snapshots[tid])
            if tid in retained_snapshots:
                setattr(tm, "retained_salary_records", list(retained_snapshots[tid]))
            if tid in owned_pick_ids_snapshot:
                setattr(tm, "owned_pick_ids", list(owned_pick_ids_snapshot[tid]))
        setattr(league, "draft_pick_registry", registry_snapshot)
        try:
            sync_owned_pick_ids_from_registry(league)
        except Exception:
            pass

    try:
        for asset in package.normalized_assets:
            if isinstance(asset, PlayerTradeAsset):
                _apply_player_move(
                    asset,
                    team_by_id,
                    season_label=season_label,
                    moved_players=moved_players,
                    retained_records=retained_records,
                    context=ctx,
                )
            elif isinstance(asset, DraftPickTradeAsset):
                _apply_pick_move(asset, league, moved_picks)
    except Exception as exc:
        _rollback_all()
        raise ValueError(f"Trade execution failed and was rolled back: {exc}") from exc

    try:
        sync_owned_pick_ids_from_registry(league)
        start_y = int(draft_year_from_context(ctx, league=league))
        audit = audit_pick_registry_integrity(
            league,
            start_year=start_y,
            years_ahead=4,
            rounds=7,
        )
        if not audit.get("ok"):
            raise ValueError(
                f"Post-trade pick registry integrity check failed: {(audit.get('errors') or ['unknown'])[0]}"
            )
    except Exception as exc:
        _rollback_all()
        raise ValueError(f"Trade execution failed and was rolled back: {exc}") from exc

    headline_bits = []
    for m in moved_players[:4]:
        headline_bits.append(f"{m.get('player_name')}: {m['source_team_id']} -> {m['acquiring_team_id']}")
    for m in moved_picks[:2]:
        headline_bits.append(f"Pick {m.get('asset_id')}: {m['source_team_id']} -> {m['acquiring_team_id']}")
    headline = "TRADE EXECUTED: " + ("; ".join(headline_bits) if headline_bits else "Assets moved")

    trade_id = f"trade_{uuid.uuid4().hex[:12]}"
    user_involved = bool(user_team_id and str(user_team_id) in package.participating_team_ids)

    history_record = append_trade_record(
        league,
        {
            "trade_id": trade_id,
            "calendar_day": ctx.get("calendar_cursor"),
            "calendar_iso": ctx.get("calendar_iso"),
            "season_year": ctx.get("season_year"),
            "participating_teams": package.participating_team_ids,
            "assets_by_team": package.assets_by_team,
            "moved_players": moved_players,
            "moved_picks": moved_picks,
            "retained_salary": retained_records,
            "cap_impact": fresh.get("cap_impact") or {},
            "value_scores": fresh.get("score_for_teams") or {},
            "fairness_gap": fresh.get("fairness_gap"),
            "accepted": True,
            "rejection_reasons": [],
            "headline": headline,
            "user_involved": user_involved,
        },
    )

    return {
        "trade_id": trade_id,
        "accepted": True,
        "moved_assets": moved_players + moved_picks,
        "moved_players": len(moved_players),
        "moved_picks": len(moved_picks),
        "retained_salary": retained_records,
        "cap_impact": fresh.get("cap_impact") or {},
        "value_breakdown": fresh.get("value_breakdown") or {},
        "fairness_gap": fresh.get("fairness_gap"),
        "headline": headline,
        "history_record": history_record,
        "evaluation": {k: v for k, v in fresh.items() if not str(k).startswith("_")},
    }
