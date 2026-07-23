"""
Trade legality validation — ownership, cap, clauses, roster, retained salary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from app.sim_engine.economy.cap_engine import (
    calculate_team_cap_snapshot,
    can_trade_cap_fit,
    can_trade_contract_slots_fit,
    player_cap_hit_millions,
    _retained_slots_used,
)
from app.sim_engine.trades.trade_asset import (
    DraftPickTradeAsset,
    PlayerTradeAsset,
    TradePackage,
    find_player_on_ahl_roster,
    find_player_on_team_roster,
    player_display_name,
    resolve_pick_id,
)
from app.sim_engine.trades.trade_pick_registry import get_pick_by_id, validate_pick_ownership


ROSTER_MIN = 20
ROSTER_MAX = 23
TRADE_ACQUISITION_COOLDOWN_DAYS = 7


def _player_is_goalie(player: Any) -> bool:
    pos = getattr(player, "position", None)
    return str(getattr(pos, "value", pos) or "").upper() == "G"

_APPROVED_DEST_FIELDS = (
    "approved_trade_teams",
    "approved_trade_team_ids",
    "approved_destinations",
    "no_trade_list",
)


def _approved_trade_destinations(player: Any) -> List[str]:
    """Normalized destination team IDs explicitly approved for M-NTC trades."""
    out: List[str] = []
    seen: Set[str] = set()
    for obj in (player, getattr(player, "contract", None)):
        if obj is None:
            continue
        for field in _APPROVED_DEST_FIELDS:
            raw = getattr(obj, field, None)
            if raw is None and isinstance(obj, dict):
                raw = obj.get(field)
            if not raw:
                continue
            items = raw if isinstance(raw, (list, tuple, set)) else [raw]
            for item in items:
                if isinstance(item, dict):
                    tid = str(
                        item.get("team_id")
                        or item.get("id")
                        or item.get("abbr")
                        or ""
                    ).strip()
                else:
                    tid = str(item).strip()
                if tid and tid not in seen:
                    seen.add(tid)
                    out.append(tid)
    return out


def _player_recently_acquired(player: Any, context: Optional[Dict[str, Any]]) -> bool:
    if not bool(getattr(player, "acquired_via_trade", False)):
        return False
    ctx = context or {}
    cursor = int(ctx.get("calendar_cursor", 0) or 0)
    last_day = getattr(player, "last_acquired_day", None)
    if last_day is not None:
        try:
            return (cursor - int(last_day)) < TRADE_ACQUISITION_COOLDOWN_DAYS
        except (TypeError, ValueError):
            pass
    last_date = str(getattr(player, "last_acquired_date", "") or "").strip()
    cur_date = str(ctx.get("calendar_iso", "") or "").strip()
    if last_date and cur_date and last_date == cur_date:
        return True
    return False


def _clause_summary(player: Any) -> Dict[str, Any]:
    c = getattr(player, "contract", None)
    clauses = getattr(c, "clauses", None) if c else None
    nmc = bool(
        getattr(clauses, "noMoveClause", False)
        if clauses
        else getattr(c, "no_move_clause", False) if c else getattr(player, "no_move_clause", False)
    )
    ntc = bool(
        getattr(clauses, "noTradeClause", False)
        if clauses
        else getattr(c, "no_trade_clause", False) if c else getattr(player, "no_trade_clause", False)
    )
    mntc = 0
    if clauses is not None:
        mntc = int(getattr(clauses, "modifiedNoTradeTeams", 0) or 0)
        clause_type = str(getattr(clauses, "clause_type", "") or "").lower()
        if not nmc and clause_type in ("nmc",):
            nmc = True
        if not ntc and clause_type in ("ntc",):
            ntc = True
        if mntc <= 0 and clause_type in ("m-ntc", "mntc"):
            mntc = max(mntc, int(getattr(clauses, "trade_list_size", 10) or 10))
    elif c is not None:
        mntc = int(getattr(c, "modified_no_trade_teams", 0) or 0)
    label = "None"
    if nmc:
        label = "NMC"
    elif ntc:
        label = "NTC"
    elif mntc > 0:
        label = "M-NTC"
    approved = _approved_trade_destinations(player) if mntc > 0 else []
    return {
        "label": label,
        "nmc": nmc,
        "ntc": ntc,
        "mntc": mntc,
        "approved_destinations": approved,
    }


def _market_size(team: Any) -> str:
    market = getattr(team, "market", None)
    size = str(getattr(market, "market_size", "") or getattr(team, "market_size", "") or "").lower()
    if size in ("small", "medium", "large"):
        return size
    return "medium"


def _team_strength_proxy(team: Any, context: Optional[Dict[str, Any]] = None) -> float:
    """Rough 0-1 team quality: roster OVR average + window/standings hint."""
    roster = list(getattr(team, "roster", None) or [])
    ovrs: List[float] = []
    for p in roster[:23]:
        try:
            fn = getattr(p, "ovr", None)
            v = float(fn() if callable(fn) else fn or 0.0)
            ovrs.append(v * 99.0 if v <= 1.5 else v)
        except Exception:
            continue
    if ovrs:
        avg = sum(ovrs) / len(ovrs)
        quality = max(0.0, min(1.0, (avg - 68.0) / 20.0))
    else:
        # No roster snapshot — lean on window instead of assuming a bad club.
        quality = 0.48
    window = str(getattr(team, "gm_window", None) or getattr(team, "window", "") or "").lower()
    if "contend" in window:
        quality += 0.12
    elif "rebuild" in window or "tank" in window:
        quality -= 0.14
    pts_pct = None
    try:
        st = (context or {}).get("standings")
        tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
        if st is not None and tid:
            rec = None
            if hasattr(st, "find_record"):
                rec = st.find_record(tid)
            if rec is None:
                rec = (getattr(st, "records", None) or {}).get(tid)
            if rec is not None:
                gp = max(1, int(getattr(rec, "gp", 0) or 0))
                pts = float(getattr(rec, "pts", 0) or 0)
                pts_pct = pts / (gp * 2.0)
    except Exception:
        pts_pct = None
    if pts_pct is not None:
        quality = 0.55 * quality + 0.45 * max(0.0, min(1.0, pts_pct))
    return max(0.0, min(1.0, quality))


def _stable_unit_roll(seed_key: str) -> float:
    import hashlib

    digest = hashlib.sha1(seed_key.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def evaluate_ntc_waiver_request(
    player: Any,
    *,
    source_team: Any,
    destination_team: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ask a player with a full NTC whether they will waive for a destination.
    Deterministic per (player, destination, season cursor) so re-asks do not re-roll.
    """
    ctx = context or {}
    clause = _clause_summary(player)
    pname = player_display_name(player)
    if clause.get("nmc"):
        return {
            "ok": False,
            "accepted": False,
            "can_request": False,
            "player_id": str(getattr(player, "id", "") or ""),
            "player_name": pname,
            "clause_label": "NMC",
            "reason": "No-movement clause cannot be waived for a trade.",
            "reason_code": "nmc_hard_block",
            "accept_chance": 0.0,
            "value_penalty_pct": 0.0,
        }
    if not clause.get("ntc") and not (clause.get("mntc", 0) > 0 and not clause.get("ntc")):
        # Full NTC only for this flow; M-NTC uses destination list unless destination blocked.
        if clause.get("mntc", 0) <= 0:
            return {
                "ok": True,
                "accepted": True,
                "can_request": False,
                "player_id": str(getattr(player, "id", "") or ""),
                "player_name": pname,
                "clause_label": clause.get("label") or "None",
                "reason": "Player has no NTC — no waiver required.",
                "reason_code": "no_ntc",
                "accept_chance": 1.0,
                "value_penalty_pct": 0.0,
            }

    dest_id = str(getattr(destination_team, "team_id", None) or getattr(destination_team, "id", "") or "")
    src_id = str(getattr(source_team, "team_id", None) or getattr(source_team, "id", "") or "")
    if clause.get("mntc", 0) > 0 and not clause.get("ntc"):
        approved = clause.get("approved_destinations") or []
        if dest_id and dest_id in approved:
            return {
                "ok": True,
                "accepted": True,
                "can_request": False,
                "player_id": str(getattr(player, "id", "") or ""),
                "player_name": pname,
                "clause_label": "M-NTC",
                "reason": "Destination is already on the player's approved trade list.",
                "reason_code": "mntc_approved",
                "accept_chance": 1.0,
                "value_penalty_pct": 0.0,
            }

    dest_quality = _team_strength_proxy(destination_team, ctx)
    dest_market = _market_size(destination_team)
    src_quality = _team_strength_proxy(source_team, ctx)
    dest_window = str(getattr(destination_team, "gm_window", None) or getattr(destination_team, "window", "") or "").lower()

    age = 28
    try:
        ident = getattr(player, "identity", None)
        age = int(getattr(ident, "age", None) or getattr(player, "age", 28) or 28)
    except Exception:
        age = 28

    chance = 0.38
    chance += (dest_quality - 0.45) * 0.55
    if dest_market == "large":
        chance += 0.10
    elif dest_market == "small":
        chance -= 0.14
    if "contend" in dest_window:
        chance += 0.12
    if "rebuild" in dest_window or "tank" in dest_window:
        chance -= 0.16
    if dest_quality + 0.08 < src_quality:
        chance -= 0.10
    if age >= 33:
        chance -= 0.08
    elif age <= 26:
        chance += 0.04
    chance = max(0.08, min(0.82, chance))

    cursor = int(ctx.get("calendar_cursor", 0) or 0)
    season = int(ctx.get("season_year", 2025) or 2025)
    pid = str(getattr(player, "id", "") or "")
    roll = _stable_unit_roll(f"ntc-waive|{season}|{cursor // 7}|{pid}|{dest_id}|{src_id}")
    accepted = roll < chance

    decline_reasons = []
    if dest_market == "small":
        decline_reasons.append(("small_market", "Does not want to move to a small market"))
    if "rebuild" in dest_window or "tank" in dest_window or dest_quality < 0.42:
        decline_reasons.append(("team_bad", "Destination looks like a weaker / less competitive roster"))
    if dest_quality + 0.05 < src_quality:
        decline_reasons.append(("desire_to_stay", "Prefers to stay with current club"))
    decline_reasons.append(("family", "Family situation — not prepared to relocate"))
    decline_reasons.append(("direction", "Unconvinced about the destination team's direction"))

    accept_reasons = [
        ("fresh_start", "Willing to waive for a fresh start"),
        ("contend", "Sees a better chance to compete with the destination"),
        ("big_market", "Attracted to the destination market / platform"),
        ("term_left", "Open to moving with years left on the deal"),
    ]

    if accepted:
        reason_code, reason = accept_reasons[int(roll * 1000) % len(accept_reasons)]
        if dest_market == "large" and "big_market" not in reason_code:
            reason_code, reason = "big_market", "Attracted to the destination market / platform"
        elif "contend" in dest_window:
            reason_code, reason = "contend", "Sees a better chance to compete with the destination"
    else:
        # Prefer situational decline reasons when available
        pool = decline_reasons[: max(1, len(decline_reasons) - 1)] or decline_reasons
        reason_code, reason = pool[int(roll * 1000) % len(pool)]

    return {
        "ok": True,
        "accepted": bool(accepted),
        "can_request": True,
        "player_id": pid,
        "player_name": pname,
        "clause_label": "NTC" if clause.get("ntc") else "M-NTC",
        "reason": reason,
        "reason_code": reason_code,
        "accept_chance": round(chance, 3),
        "roll": round(roll, 4),
        "destination_team_id": dest_id,
        "source_team_id": src_id,
        # Applied to trade value when the waiver is used in a package.
        "value_penalty_pct": 0.08 if accepted else 0.0,
        "value_note": (
            "NTC waived — trade value slightly reduced"
            if accepted
            else "NTC remains in force — player cannot be traded without a waiver"
        ),
    }


def _asset_has_ntc_waiver(asset: PlayerTradeAsset, context: Optional[Dict[str, Any]] = None) -> bool:
    if bool(getattr(asset, "ntc_waived", False)):
        return True
    raw = getattr(asset, "raw", None) or {}
    if bool(raw.get("ntc_waived") or raw.get("ntcWaived") or raw.get("clause_waived")):
        return True
    waivers = (context or {}).get("ntc_waivers") or {}
    if not isinstance(waivers, dict):
        return False
    key = str(asset.player_id)
    entry = waivers.get(key) or waivers.get(f"{key}->{asset.acquiring_team_id}")
    if isinstance(entry, dict):
        if not bool(entry.get("accepted")):
            return False
        dest = str(entry.get("destination_team_id") or "")
        return (not dest) or dest == str(asset.acquiring_team_id)
    return bool(entry)


def _season_label(context: Optional[Dict[str, Any]]) -> Optional[str]:
    if not context:
        return None
    y = context.get("season_year")
    if y:
        return f"{int(y)}-{(int(y) + 1) % 100:02d}"
    return None


def _contract_years_for_retention(player: Any) -> int:
    c = getattr(player, "contract", None)
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("years_remaining", "term_remaining", "remaining_years", "term"):
            try:
                v = int(getattr(obj, key, 0) or 0)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                continue
    return 0


def _players_for_team_side(
    package: TradePackage,
    team_id: str,
    team_by_id: Dict[str, Any],
) -> Tuple[List[Any], List[Any], List[PlayerTradeAsset], Dict[str, float]]:
    """Return (outgoing_players, incoming_players, player_assets_out, incoming_retained_pct) for cap check."""
    outgoing_objs: List[Any] = []
    incoming_objs: List[Any] = []
    out_assets: List[PlayerTradeAsset] = []
    incoming_retained: Dict[str, float] = {}

    team = team_by_id.get(team_id)
    if team is None:
        return outgoing_objs, incoming_objs, out_assets, incoming_retained

    for asset in package.outgoing_by_team.get(team_id, []):
        if not isinstance(asset, PlayerTradeAsset):
            continue
        p, _ = find_player_on_team_roster(team, asset.player_id)
        if p is not None:
            outgoing_objs.append(p)
            out_assets.append(asset)

    for asset in package.incoming_by_team.get(team_id, []):
        if not isinstance(asset, PlayerTradeAsset):
            continue
        src = team_by_id.get(asset.source_team_id)
        if src is None:
            continue
        p, _ = find_player_on_team_roster(src, asset.player_id)
        if p is not None:
            incoming_objs.append(p)
            if asset.retained_pct > 0:
                incoming_retained[str(asset.player_id)] = float(asset.retained_pct)

    return outgoing_objs, incoming_objs, out_assets, incoming_retained


def validate_trade_rules(
    package: TradePackage,
    league: Any,
    team_by_id: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
    user_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    blocking: List[str] = []
    warnings: List[str] = []
    cap_impact: Dict[str, Dict[str, float]] = {}
    roster_impact: Dict[str, Dict[str, int]] = {}
    contract_slot_impact: Dict[str, Dict[str, int]] = {}
    clause_impact: Dict[str, List[str]] = {}

    ctx = context or {}
    season_year = int(ctx.get("season_year", 2025) or 2025)
    season_label = _season_label(ctx)
    sim = ctx.get("sim")
    seen_players: Set[str] = set()
    seen_picks: Set[str] = set()

    for tid in package.participating_team_ids:
        if tid not in team_by_id:
            blocking.append(f"Unknown team in trade package: {tid}")

    for asset in package.normalized_assets:
        if isinstance(asset, PlayerTradeAsset):
            if asset.player_id in seen_players:
                blocking.append(f"Duplicate player in trade package: {asset.player_id}")
            seen_players.add(asset.player_id)

            if asset.retained_pct < 0 or asset.retained_pct > 50:
                blocking.append(
                    f"Retained salary for {asset.player_id} must be between 0% and 50% (got {asset.retained_pct}%)"
                )

            src = team_by_id.get(asset.source_team_id)
            if src is None:
                blocking.append(f"Source team not found for player {asset.player_id}")
                continue
            player, _ = find_player_on_team_roster(src, asset.player_id)
            if player is None:
                ahl_player, _ = find_player_on_ahl_roster(src, asset.player_id)
                if ahl_player is not None:
                    pname = player_display_name(ahl_player)
                    blocking.append(
                        f"{pname} is assigned to AHL and cannot be traded — player must be on the NHL roster"
                    )
                else:
                    blocking.append(f"Player {asset.player_id} not found on source roster {asset.source_team_id}")
                continue

            pname = player_display_name(player)
            clause = _clause_summary(player)
            if clause["nmc"]:
                blocking.append(f"{pname} has a no-movement clause (NMC) and cannot be traded")
                clause_impact.setdefault(asset.source_team_id, []).append(f"{pname}: NMC blocks trade")
            elif clause["ntc"]:
                if _asset_has_ntc_waiver(asset, ctx):
                    warnings.append(
                        f"{pname} waived NTC for this destination — trade value slightly reduced"
                    )
                    clause_impact.setdefault(asset.source_team_id, []).append(
                        f"{pname}: NTC waived for {asset.acquiring_team_id}"
                    )
                else:
                    blocking.append(
                        f"{pname} has a no-trade clause (NTC) — ask the player to waive before trading"
                    )
                    clause_impact.setdefault(asset.source_team_id, []).append(
                        f"{pname}: NTC blocks trade (waiver required)"
                    )
            elif clause["mntc"] > 0:
                approved = clause.get("approved_destinations") or _approved_trade_destinations(player)
                dest = str(asset.acquiring_team_id)
                if approved and dest in approved:
                    pass
                elif _asset_has_ntc_waiver(asset, ctx):
                    warnings.append(
                        f"{pname} waived M-NTC destination restriction — trade value slightly reduced"
                    )
                    clause_impact.setdefault(asset.source_team_id, []).append(
                        f"{pname}: M-NTC waived for {dest}"
                    )
                else:
                    blocking.append("Modified no-trade clause requires approved destination.")
                    clause_impact.setdefault(asset.source_team_id, []).append(f"{pname}: M-NTC blocks trade")

            if _player_recently_acquired(player, ctx):
                blocking.append(f"{pname}: Recently acquired players cannot be traded yet.")

            if asset.retained_pct > 0:
                retaining = team_by_id.get(asset.source_team_id)
                slots_used = _retained_slots_used(retaining, season_label) if retaining else 0
                if slots_used >= 3:
                    blocking.append(
                        f"{asset.source_team_id} already uses the maximum of 3 retained-salary slots"
                    )
                p_years = _contract_years_for_retention(player)
                if p_years <= 0:
                    blocking.append(
                        f"{pname} has no contract years remaining — cannot retain salary on this trade"
                    )
                elif asset.retained_pct > 0 and p_years < 1:
                    blocking.append(f"{pname}: retention cannot exceed contract term")

        elif isinstance(asset, DraftPickTradeAsset):
            pid = resolve_pick_id(asset.pick_id, asset.source_team_id)
            if pid in seen_picks:
                blocking.append(f"Duplicate pick in trade package: {pid}")
            seen_picks.add(pid)

            row = get_pick_by_id(league, pid)
            if not row:
                blocking.append(f"Pick not found in league registry: {pid}")
                continue
            if bool(row.get("resolved")):
                blocking.append(f"Pick already resolved and unavailable: {pid}")
                continue
            try:
                pick_year = int(row.get("year", 0))
                pick_round = int(row.get("round", 0))
            except Exception:
                blocking.append(f"Pick has invalid year/round metadata: {pid}")
                continue
            if pick_round < 1 or pick_round > 7:
                blocking.append(f"Pick round out of range for {pid}: {pick_round}")
            if pick_year < season_year or pick_year > season_year + 7:
                blocking.append(
                    f"Pick year out of allowed range for {pid}: {pick_year} (season {season_year})"
                )
            if not validate_pick_ownership(league, pid, asset.source_team_id):
                blocking.append(
                    f"Team {asset.source_team_id} does not own pick {pid} (owner: {row.get('current_owner_team_id')})"
                )
            raw_owner = (
                asset.raw.get("current_owner_team_id")
                or asset.raw.get("owner")
                or asset.raw.get("team_id")
            )
            if raw_owner is not None and str(raw_owner) != str(row.get("current_owner_team_id")):
                blocking.append(
                    f"Frontend ownership mismatch for {pid}: payload owner {raw_owner} != registry owner {row.get('current_owner_team_id')}"
                )

    for tid in package.participating_team_ids:
        team = team_by_id.get(tid)
        if team is None:
            continue

        outgoing, incoming, out_assets, incoming_retained = _players_for_team_side(package, tid, team_by_id)
        retained_added = 0.0
        for a in out_assets:
            if a.retained_pct > 0:
                p, _ = find_player_on_team_roster(team, a.player_id)
                if p is not None:
                    retained_added += player_cap_hit_millions(p) * (a.retained_pct / 100.0)

        snap_before = calculate_team_cap_snapshot(
            team,
            league=league,
            sim=sim,
            season_label=season_label,
            calendar_cursor=int(ctx.get("calendar_cursor", 0) or 0),
            regular_season_last_index=int(ctx.get("regular_season_last_index", 192) or 192),
        )
        cap_check = can_trade_cap_fit(
            team,
            outgoing,
            incoming,
            retained_added_m=retained_added,
            league=league,
            incoming_retained_pct=incoming_retained,
            calendar_cursor=int(ctx.get("calendar_cursor", 0) or 0),
            regular_season_last_index=int(ctx.get("regular_season_last_index", 192) or 192),
            deadline_phase=float(ctx.get("deadline_phase", 0.0) or 0.0),
            season_label=season_label,
        )

        before_usable = float(snap_before.get("usableCapSpace", 0.0))
        after_usable = float(cap_check.get("projectedCapSpace", before_usable))
        after_deadline = float(cap_check.get("projectedDeadlineSpace", after_usable))
        delta = float(cap_check.get("capDelta", 0.0))

        cap_impact[tid] = {
            "before_usable": round(before_usable, 3),
            "after_usable": round(after_usable, 3),
            "after_deadline_space": round(after_deadline, 3),
            "delta": round(delta, 3),
            "delta_full": round(float(cap_check.get("capDeltaFull", delta)), 3),
            "proration_factor": round(float(cap_check.get("prorationFactor", 1.0)), 4),
            "ltir_relief_used": bool(cap_check.get("ltirReliefUsed")),
        }

        if cap_check.get("reason") == "ok_with_ltir":
            warnings.append(f"{tid}: trade fits under LTIR effective cap limit")
        elif cap_check.get("reason") == "ok_with_accrual":
            warnings.append(f"{tid}: trade fits using in-season cap accrual projection")

        if not cap_check.get("ok"):
            cap_casualty = bool(ctx.get("cap_casualty_trade"))
            partial_relief = cap_casualty and delta < -0.001 and after_usable > before_usable + 0.001
            if not partial_relief:
                blocking.append(f"{tid}: {cap_check.get('reason', 'Cap validation failed')}")

        proj_count = int(cap_check.get("projectedRosterCount", snap_before.get("activeRosterCount", 0)))
        roster_impact[tid] = {
            "before": int(snap_before.get("activeRosterCount", 0)),
            "after": proj_count,
            "outgoing_players": len(outgoing),
            "incoming_players": len(incoming),
        }

        if proj_count > ROSTER_MAX:
            blocking.append(f"{tid} would exceed maximum roster size ({proj_count} > {ROSTER_MAX})")
        if proj_count < ROSTER_MIN:
            warnings.append(f"{tid} would drop below recommended roster minimum ({proj_count} < {ROSTER_MIN})")

        # Never leave an NHL club with zero goalies after trading its last one away.
        roster_now = list(getattr(team, "roster", None) or [])
        g_now = sum(
            1
            for p in roster_now
            if not getattr(p, "retired", False) and _player_is_goalie(p)
        )
        g_out = sum(1 for p in outgoing if _player_is_goalie(p))
        g_in = sum(1 for p in incoming if _player_is_goalie(p))
        if g_now >= 1 and (g_now - g_out + g_in) < 1:
            blocking.append(f"{tid} would have no NHL goalies after this trade")

        slot_check = can_trade_contract_slots_fit(team, outgoing, incoming)
        contract_slot_impact[tid] = {
            "before": int(slot_check.get("contract_slots_used", 0)),
            "incoming_contracts": len(incoming),
            "outgoing_contracts": len(outgoing),
            "after": int(slot_check.get("projected_contract_slots", 0)),
            "limit": int(slot_check.get("contract_slots_limit", 50)),
            "ok": bool(slot_check.get("ok")),
        }
        if not slot_check.get("ok"):
            blocking.append(f"{tid}: {slot_check.get('reason', 'Contract slot validation failed')}")

    ok = len(blocking) == 0
    return {
        "ok": ok,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "cap_impact": cap_impact,
        "roster_impact": roster_impact,
        "contract_slot_impact": contract_slot_impact,
        "clause_impact": clause_impact,
    }
