"""
Atomic trade execution after validation.
"""

from __future__ import annotations

import uuid
import copy
from typing import Any, Dict, List, Optional

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


def _apply_player_move(
    asset: PlayerTradeAsset,
    team_by_id: Dict[str, Any],
    *,
    season_label: Optional[str],
    moved_players: List[Dict[str, Any]],
    retained_records: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    source = team_by_id.get(str(asset.source_team_id)) or team_by_id.get(asset.source_team_id)
    acq = team_by_id.get(str(asset.acquiring_team_id)) or team_by_id.get(asset.acquiring_team_id)
    if source is None or acq is None:
        raise ValueError(f"Teams missing for player move {asset.player_id}")

    src_roster = list(getattr(source, "roster", None) or [])
    idx = next((i for i, p in enumerate(src_roster) if str(getattr(p, "id", "")) == asset.player_id), -1)
    if idx < 0:
        raise ValueError(f"Player {asset.player_id} not on source roster during execution")

    player = src_roster.pop(idx)
    acq_roster = list(getattr(acq, "roster", None) or [])
    acq_roster.append(player)
    setattr(source, "roster", src_roster)
    setattr(acq, "roster", acq_roster)

    for field in ("team_id", "current_team_id", "last_team_id"):
        try:
            if field == "last_team_id":
                setattr(player, field, asset.source_team_id)
            else:
                setattr(player, field, asset.acquiring_team_id)
        except Exception:
            pass

    ctx = context or {}
    cursor = int(ctx.get("calendar_cursor", 0) or 0)
    for field, val in (
        ("last_acquired_day", cursor),
        ("last_acquired_date", ctx.get("calendar_iso")),
        ("acquired_from_team_id", asset.source_team_id),
        ("acquired_via_trade", True),
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
        }
    )

    if asset.retained_pct > 0:
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

    ensure_draft_pick_registry(league, start_year=ctx.get("season_year"))

    # Snapshot mutable state for rollback
    snapshots: Dict[str, List[Any]] = {}
    registry_snapshot = copy.deepcopy(dict(getattr(league, "draft_pick_registry", {}) or {}))
    owned_pick_ids_snapshot: Dict[str, List[str]] = {}
    for tid, tm in team_by_id.items():
        snapshots[tid] = list(getattr(tm, "roster", None) or [])
        owned_pick_ids_snapshot[tid] = list(getattr(tm, "owned_pick_ids", None) or [])

    moved_players: List[Dict[str, Any]] = []
    moved_picks: List[Dict[str, Any]] = []
    retained_records: List[Dict[str, Any]] = []

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
        for tid, tm in team_by_id.items():
            if tid in snapshots:
                setattr(tm, "roster", snapshots[tid])
            if tid in owned_pick_ids_snapshot:
                setattr(tm, "owned_pick_ids", list(owned_pick_ids_snapshot[tid]))
        setattr(league, "draft_pick_registry", registry_snapshot)
        try:
            sync_owned_pick_ids_from_registry(league)
        except Exception:
            pass
        raise ValueError(f"Trade execution failed and was rolled back: {exc}") from exc

    try:
        sync_owned_pick_ids_from_registry(league)
        start_y = int(ctx.get("season_year") or 2025)
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
        for tid, tm in team_by_id.items():
            if tid in snapshots:
                setattr(tm, "roster", snapshots[tid])
            if tid in owned_pick_ids_snapshot:
                setattr(tm, "owned_pick_ids", list(owned_pick_ids_snapshot[tid]))
        setattr(league, "draft_pick_registry", registry_snapshot)
        try:
            sync_owned_pick_ids_from_registry(league)
        except Exception:
            pass
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
