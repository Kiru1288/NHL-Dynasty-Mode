"""
Atomic trade execution after validation.
"""

from __future__ import annotations

import uuid
import copy
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.trades.trade_asset import DraftPickTradeAsset, PlayerTradeAsset, RetainedSalaryRecord, TradePackage, player_display_name, team_id_of
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
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)


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


def _pick_name(row: Dict[str, Any], team_by_id: Dict[str, Any], pick_id: Any) -> str:
    """'2027 EDM 4th-round pick' instead of the registry id (C10)."""
    year, rnd = row.get("year"), row.get("round")
    if not (year and rnd):
        return "a draft pick"
    t = team_by_id.get(str(row.get("original_team_id") or ""))
    abbr = str(getattr(t, "abbreviation", "") or getattr(t, "abbr", "") or "").upper() if t is not None else ""
    n = int(rnd)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n, "th")
    return f"{int(year)} {abbr + ' ' if abbr else ''}{n}{suffix}-round pick"


def _snapshot_player_fields(player: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in dict(getattr(player, "__dict__", {}) or {}).items():
        out[k] = copy.copy(v) if isinstance(v, (dict, list, set)) else v
    return out


def _restore_player_fields(player: Any, snap: Dict[str, Any]) -> None:
    d = getattr(player, "__dict__", None)
    if not isinstance(d, dict):
        return
    for k in [k for k in d if k not in snap]:
        d.pop(k, None)
    d.update(snap)


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
            _swallowed_log.debug("suppressed exception", exc_info=True)
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
        _swallowed_log.debug("suppressed exception", exc_info=True)

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
            _swallowed_log.debug("suppressed exception", exc_info=True)

    # Draft rights follow the player, otherwise the prospect still reads as
    # belonging to the club that drafted him on every rights surface.
    if getattr(player, "nhl_rights_team_id", None) is not None:
        for field in ("nhl_rights_team_id", "rights_team_id"):
            try:
                setattr(player, field, asset.acquiring_team_id)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
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
            _swallowed_log.debug("suppressed exception", exc_info=True)

    stats = (context or {}).get("player_season_stats") if isinstance(context, dict) else None
    if isinstance(stats, dict):
        row = stats.get(str(asset.player_id))
        if isinstance(row, dict):
            # U12: keep a per-team split. The row stays the season total; each split holds
            # what he did for a previous club, so "with new club" = total − splits.
            try:
                old_tid = str(row.get("team_id") or asset.source_team_id)
                splits = list(row.get("team_splits") or [])
                prior: Dict[str, float] = {}
                for sp in splits:
                    for k, v in sp.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            prior[k] = prior.get(k, 0) + v
                piece: Dict[str, Any] = {"team_id": old_tid, "until_day": int((context or {}).get("calendar_cursor", 0) or 0)}
                for k, v in row.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("until_day",):
                        piece[k] = round(v - prior.get(k, 0), 4) if isinstance(v, float) else v - int(prior.get(k, 0))
                if int(piece.get("gp", 0) or 0) > 0:
                    splits.append(piece)
                    row["team_splits"] = splits
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
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
        from app.sim_engine.economy.cap_engine import player_full_cap_hit_millions

        cap_hit = player_full_cap_hit_millions(player)
        try:
            prior = float(getattr(player, "retained_share_pct", 0.0) or 0.0)
            c = getattr(player, "contract", None)
            expiry = c.get("expiry_year") if isinstance(c, dict) else getattr(c, "expiry_year", None)
            if getattr(player, "retained_share_expiry", None) not in (None, expiry):
                prior = 0.0
            try:
                from app.sim_engine.economy.cap_engine import max_retention_pct

                cap_pct = float(max_retention_pct((context or {}).get("league")))
            except Exception:
                cap_pct = 50.0
            prior_count = int(getattr(player, "retention_count", 0) or 0)
            if getattr(player, "retained_share_expiry", None) not in (None, expiry):
                prior_count = 0
            setattr(player, "retained_share_pct", min(cap_pct, prior + float(asset.retained_pct)))
            setattr(player, "retained_share_expiry", expiry)
            # NHL: a contract can be retained on at most twice (U14).
            setattr(player, "retention_count", prior_count + 1)
            setattr(player, "retention_last_day", int((context or {}).get("calendar_cursor", 0) or 0))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
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
        try:
            source.retained_salary_records[-1]["retained_day"] = int((context or {}).get("calendar_cursor", 0) or 0)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
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
    team_by_id: Optional[Dict[str, Any]] = None,
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
            "display_name": _pick_name(row, team_by_id or {}, asset.pick_id),
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

    # U11: player fields (team ids, rights, retention, flags) and stat rows roll back too.
    player_snaps: List[Tuple[Any, Dict[str, Any]]] = []
    stat_snaps: Dict[str, Dict[str, Any]] = {}
    _stats_ref = ctx.get("player_season_stats") if isinstance(ctx.get("player_season_stats"), dict) else None
    for _a in package.normalized_assets:
        if not isinstance(_a, PlayerTradeAsset):
            continue
        _src = team_by_id.get(str(_a.source_team_id))
        for _attr in _org_list_attrs():
            for _p in list(getattr(_src, _attr, None) or []) if _src is not None else []:
                if str(getattr(_p, "id", "")) == str(_a.player_id):
                    player_snaps.append((_p, _snapshot_player_fields(_p)))
        if _stats_ref is not None and isinstance(_stats_ref.get(str(_a.player_id)), dict):
            stat_snaps[str(_a.player_id)] = copy.deepcopy(_stats_ref[str(_a.player_id)])

    def _rollback_all() -> None:
        for _p, _snap in player_snaps:
            try:
                _restore_player_fields(_p, _snap)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
        if _stats_ref is not None:
            for _pid, _row in stat_snaps.items():
                _stats_ref[_pid] = _row
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
            _swallowed_log.debug("suppressed exception", exc_info=True)

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
                _apply_pick_move(asset, league, moved_picks, team_by_id)
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

    # Corresponding moves: clubs pushed past 23 active send their lowest depth to the AHL.
    roster_moves: List[Dict[str, Any]] = []
    try:
        from app.sim_engine.trades.roster_balance import auto_send_down_overflow

        arrived = {str(m.get("player_id") or m.get("asset_id") or "") for m in moved_players}
        for tid in package.participating_team_ids:
            # The user's club is never trimmed automatically: they choose who goes down
            # (through the waiver flow); the roster-compliance check blocks advancing until then.
            if user_team_id and str(tid) == str(user_team_id):
                continue
            tm = team_by_id.get(str(tid))
            if tm is not None:
                roster_moves.extend(auto_send_down_overflow(tm, protect_ids=arrived))
        pending = [m for m in roster_moves if m.get("needs_waivers")]
        if pending:
            queue = list(getattr(league, "_pending_trade_waivers", None) or [])
            queue.extend({"team_id": m["team_id"], "player_id": m["player_id"]} for m in pending)
            setattr(league, "_pending_trade_waivers", queue)
    except Exception:
        roster_moves = []

    def _tname(tid: Any) -> str:
        t = team_by_id.get(str(tid))
        if t is None:
            return str(tid)
        abbr = str(getattr(t, "abbreviation", "") or getattr(t, "abbr", "") or "").upper()
        return abbr or str(getattr(t, "name", "") or tid)

    headline_bits = []
    for m in moved_players[:4]:
        headline_bits.append(f"{m.get('player_name')}: {_tname(m['source_team_id'])} -> {_tname(m['acquiring_team_id'])}")
    for m in moved_picks[:2]:
        headline_bits.append(f"{m.get('display_name') or 'Draft pick'}: {_tname(m['source_team_id'])} -> {_tname(m['acquiring_team_id'])}")
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
            "roster_moves": roster_moves,
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
        "roster_moves": roster_moves,
        "history_record": history_record,
        "evaluation": {k: v for k, v in fresh.items() if not str(k).startswith("_")},
    }
