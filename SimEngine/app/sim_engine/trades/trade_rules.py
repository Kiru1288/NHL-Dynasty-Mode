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
    max_retained_slots,
    max_retention_pct,
)
from app.sim_engine.trades.trade_asset import (
    DraftPickTradeAsset,
    PlayerTradeAsset,
    TradePackage,
    find_player_on_ahl_roster,
    find_player_in_organization,
    find_player_on_team_roster,
    player_display_name,
    resolve_pick_id,
)
from app.sim_engine.trades.trade_pick_registry import get_pick_by_id, validate_pick_ownership
from app.sim_engine.trades.trade_deadline import POST_DEADLINE_BLOCK_REASON, post_deadline_freeze_active


ROSTER_MIN = 20
ROSTER_MAX = 23
TRADE_ACQUISITION_COOLDOWN_DAYS = 7
# Hard ban: cannot return a player to acquired_from_team_id during the same season.
TRADE_REVERSE_RETURN_SEASON_BLOCK = True


def _player_is_goalie(player: Any) -> bool:
    pos = getattr(player, "position", None)
    return str(getattr(pos, "value", pos) or "").upper() == "G"

_APPROVED_DEST_FIELDS = (
    "approved_trade_teams",
    "approved_trade_team_ids",
    "approved_destinations",
    "no_trade_list",
    "ntc_teams",
)


class _DictView:
    """Attribute access over a contract dict (missing keys → AttributeError → getattr default)."""

    __slots__ = ("_d",)

    def __init__(self, d: Dict[str, Any]) -> None:
        self._d = d

    def __getattr__(self, key: str) -> Any:
        try:
            return self._d[key]
        except KeyError:
            raise AttributeError(key) from None

    def __bool__(self) -> bool:
        return True


def _dest_on_list(dest: str, approved: List[str]) -> bool:
    d = str(dest or "").strip()
    if not d or not approved:
        return False
    for item in approved:
        s = str(item or "").strip()
        if s and (s == d or s.upper() == d.upper()):
            return True
    return False


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
    cur_season = ctx.get("season_year")
    acq_season = getattr(player, "acquired_via_trade_season", None)
    # Stamps from a prior season are stale once the calendar rolls — do not block.
    if cur_season is not None and acq_season is not None:
        try:
            if int(acq_season) != int(cur_season):
                return False
        except (TypeError, ValueError):
            pass
    cursor = int(ctx.get("calendar_cursor", 0) or 0)
    last_day = getattr(player, "last_acquired_day", None)
    if last_day is not None:
        try:
            elapsed = cursor - int(last_day)
            # Negative elapsed means the stamp is from a prior calendar (cursor reset).
            if elapsed < 0:
                return False
            return elapsed < TRADE_ACQUISITION_COOLDOWN_DAYS
        except (TypeError, ValueError):
            pass
    last_date = str(getattr(player, "last_acquired_date", "") or "").strip()
    cur_date = str(ctx.get("calendar_iso", "") or "").strip()
    if last_date and cur_date and last_date == cur_date:
        return True
    return False


def _player_returning_to_prior_club(player: Any, acquiring_team_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """True when this trade would bounce a player back to acquired_from_team_id this season."""
    if not TRADE_REVERSE_RETURN_SEASON_BLOCK:
        return False
    prior = str(getattr(player, "acquired_from_team_id", "") or "").strip()
    dest = str(acquiring_team_id or "").strip()
    if not prior or not dest or prior != dest:
        return False
    ctx = context or {}
    cur_season = ctx.get("season_year")
    acq_season = getattr(player, "acquired_via_trade_season", None)
    if acq_season is None or cur_season is None:
        # Conservative: block when prior club is known and season stamp missing.
        return True
    try:
        return int(acq_season) == int(cur_season)
    except (TypeError, ValueError):
        return True


def _dest_retained_on_player(team: Any, player_id: Any) -> bool:
    pid = str(player_id or "")
    for rec in list(getattr(team, "retained_salary_records", None) or []) if team is not None else []:
        rid = rec.get("player_id") if isinstance(rec, dict) else getattr(rec, "player_id", None)
        if str(rid or "") == pid:
            return True
    return False


def _clause_summary(player: Any) -> Dict[str, Any]:
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        # Contracts are stored as dicts; getattr() on a dict always returned the default,
        # so NMC / NTC / M-NTC were invisible to the rules engine.
        c = _DictView(c)
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
    clause_type = ""
    mode = ""
    if c is not None:
        mode = str(getattr(c, "ntc_mode", "") or "").upper()
        clause_type = str(getattr(c, "clause_type", "") or "").lower()
    if clauses is not None:
        mntc = int(getattr(clauses, "modifiedNoTradeTeams", 0) or 0)
        nested = str(getattr(clauses, "clause_type", "") or "").lower()
        if nested:
            clause_type = nested
        if not nmc and clause_type in ("nmc",):
            nmc = True
        if not ntc and clause_type in ("ntc",):
            ntc = True
        if mntc <= 0 and clause_type in ("m-ntc", "mntc"):
            mntc = max(mntc, int(getattr(clauses, "trade_list_size", 10) or 10))
    elif c is not None:
        mntc = int(getattr(c, "modified_no_trade_teams", 0) or 0)
    # A modified list stores no_trade_clause=True plus a team count. That is not a full NTC.
    modified = (
        mode in ("MODIFIED", "MNTC", "M-NTC")
        or clause_type in ("m-ntc", "mntc")
        or mntc > 0
    )
    if modified and not nmc:
        ntc = False
        if mntc <= 0:
            mntc = 10
    label = "None"
    if nmc:
        label = "NMC"
    elif modified:
        label = "M-NTC"
    elif ntc:
        label = "NTC"
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
    Preview whether a protected player (NMC / NTC / M-NTC) would waive for a destination.

    Pure: records nothing. The roll is seeded by player + season + waiver window only
    (``context["clause_window_key"]``), so the answer never changes by re-asking. Session
    flows (meetings, Trade Hub) go through ``clause_consent.request_clause_waiver``, which
    also enforces the no-re-ask rule and stores consent.
    """
    ctx = context or {}
    clause = _clause_summary(player)
    pname = player_display_name(player)
    if clause.get("nmc"):
        from app.sim_engine.trades.clause_consent import destination_chance

        pid = str(getattr(player, "id", "") or "")
        season = int(ctx.get("season_year", 2025) or 2025)
        window_key = str(ctx.get("clause_window_key") or f"{season}-in")
        chance = destination_chance(
            player,
            source_team=source_team,
            destination_team=destination_team,
            context=ctx,
            rel_adj=float(ctx.get("waiver_rel_adj") or 0.0),
        )
        dest_key = str(getattr(destination_team, "team_id", None) or getattr(destination_team, "id", "") or "")
        roll = _stable_unit_roll(f"clause-waive|{pid}|{season}|{window_key}|{dest_key}")
        accepted = roll < chance
        return {
            "ok": True,
            "accepted": bool(accepted),
            "can_request": True,
            "player_id": pid,
            "player_name": pname,
            "clause_label": "NMC",
            "reason": (
                "Willing to waive his no-movement clause for this move"
                if accepted
                else "Not willing to waive his no-movement clause"
            ),
            "reason_code": "nmc_waive" if accepted else "nmc_decline",
            "accept_chance": round(chance, 3),
            "roll": round(roll, 4),
            "destination_team_id": str(getattr(destination_team, "team_id", None) or getattr(destination_team, "id", "") or ""),
            "source_team_id": str(getattr(source_team, "team_id", None) or getattr(source_team, "id", "") or ""),
            "value_penalty_pct": 0.08 if accepted else 0.0,
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
        if dest_id and _dest_on_list(dest_id, approved):
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

    from app.sim_engine.trades.clause_consent import destination_chance

    chance = destination_chance(
        player,
        source_team=source_team,
        destination_team=destination_team,
        context=ctx,
        rel_adj=float(ctx.get("waiver_rel_adj") or 0.0),
    )

    # Stable per player + season + window + destination. Re-asking the same club
    # does not re-roll; a different club gets its own roll.
    season = int(ctx.get("season_year", 2025) or 2025)
    window_key = str(ctx.get("clause_window_key") or f"{season}-in")
    pid = str(getattr(player, "id", "") or "")
    roll = _stable_unit_roll(f"clause-waive|{pid}|{season}|{window_key}|{dest_id}")
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
    """True when the player has consented to this move (NMC / NTC / M-NTC waiver).

    Franchise trades are authoritative: only the session's consent records count — a
    client-sent ``ntc_waived`` flag is ignored, consent must match the destination,
    the current waiver window and the club that holds the player.
    """
    from app.sim_engine.trades.clause_consent import consent_allows, lookup_consent

    ctx = context or {}
    has_book = isinstance(ctx.get("ntc_waivers"), dict)
    if not ctx.get("clause_consent_authoritative") and not has_book:
        if bool(getattr(asset, "ntc_waived", False)):
            return True
        raw = getattr(asset, "raw", None) or {}
        if bool(raw.get("ntc_waived") or raw.get("ntcWaived") or raw.get("clause_waived")):
            return True
    entry = lookup_consent(ctx.get("ntc_waivers") or {}, str(asset.player_id), str(asset.acquiring_team_id))
    return consent_allows(
        entry,
        str(asset.acquiring_team_id),
        window_key=ctx.get("clause_window_key"),
        source_team_id=str(asset.source_team_id),
    )


def _season_label(context: Optional[Dict[str, Any]]) -> Optional[str]:
    if not context:
        return None
    y = context.get("season_year")
    if y:
        return f"{int(y)}-{(int(y) + 1) % 100:02d}"
    return None


def _contract_years_for_retention(player: Any) -> int:
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        c = _DictView(c)  # was always 0 for dict contracts → retention blocked on every trade
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
        p, _loc, _i = find_player_in_organization(team, asset.player_id)
        if p is not None:
            outgoing_objs.append(p)
            out_assets.append(asset)

    for asset in package.incoming_by_team.get(team_id, []):
        if not isinstance(asset, PlayerTradeAsset):
            continue
        src = team_by_id.get(asset.source_team_id)
        if src is None:
            continue
        p, _loc, _i = find_player_in_organization(src, asset.player_id)
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
    try:
        from app.sim_engine.trades.trade_pick_registry import draft_year_from_context

        draft_year = int(draft_year_from_context(ctx, league=league))
    except Exception:
        draft_year = int(ctx.get("draft_year") or season_year)
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

            if asset.retained_pct < 0 or asset.retained_pct > max_retention_pct(league):
                blocking.append(
                    f"Retained salary for {asset.player_id} must be between 0% and {max_retention_pct(league):.0f}% (got {asset.retained_pct}%)"
                )

            src = team_by_id.get(asset.source_team_id)
            if src is None:
                blocking.append(f"Source team not found for player {asset.player_id}")
                continue
            from app.sim_engine.trades.trade_asset import (
                player_holds_nhl_spc,
                player_is_tradeable_draft_rights,
            )

            player, loc, _i = find_player_in_organization(src, asset.player_id)
            if player is None:
                blocking.append(f"Player {asset.player_id} not found on source roster {asset.source_team_id}")
                continue
            if post_deadline_freeze_active(ctx) and loc != "ahl":
                blocking.append(f"{player_display_name(player)} ({loc.upper() or 'NHL'}): {POST_DEADLINE_BLOCK_REASON}")
                continue
            if bool(getattr(player, "_conduct_trade_restricted", False)):
                warnings.append(
                    f"{player_display_name(player)} is under a restricted trade market after a conduct matter."
                )
                clause_impact.setdefault(asset.source_team_id, []).append(
                    f"{player_display_name(player)}: conduct-restricted market"
                )
            if (
                loc in ("ahl", "echl", "prospect")
                and not player_holds_nhl_spc(player)
                and not (loc == "prospect" and player_is_tradeable_draft_rights(player))
            ):
                pname = player_display_name(player)
                blocking.append(
                    f"{pname} is on the {loc.upper()} list without an NHL SPC and cannot be traded"
                )
                continue

            pname = player_display_name(player)
            clause = _clause_summary(player)
            approved_dests = clause.get("approved_destinations") or []
            if clause["nmc"]:
                # An NMC can be waived by the player (meeting or Trade Hub ask).
                if _asset_has_ntc_waiver(asset, ctx):
                    warnings.append(
                        f"{pname} waived his NMC for this move — trade value slightly reduced"
                    )
                    clause_impact.setdefault(asset.source_team_id, []).append(
                        f"{pname}: NMC waived for {asset.acquiring_team_id}"
                    )
                else:
                    blocking.append(
                        f"{pname} has a no-movement clause (NMC) — ask the player to waive before trading"
                    )
                    clause_impact.setdefault(asset.source_team_id, []).append(f"{pname}: NMC blocks trade (waiver required)")
            elif _dest_on_list(str(asset.acquiring_team_id), approved_dests):
                pass
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
                if _dest_on_list(dest, approved):
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

            # NHL rule: no waiting period after a trade. The only re-acquisition limit is that a
            # club that retained salary on a player can't get him back within a year.
            if _dest_retained_on_player(team_by_id.get(str(asset.acquiring_team_id)), asset.player_id):
                blocking.append(
                    f"{pname}: {asset.acquiring_team_id} retained salary on him — can't reacquire him within a year."
                )

            if asset.retained_pct > 0:
                retaining = team_by_id.get(asset.source_team_id)
                slots_used = _retained_slots_used(retaining, season_label) if retaining else 0
                if slots_used >= max_retained_slots(league):
                    blocking.append(
                        f"{asset.source_team_id} already uses the maximum of {max_retained_slots(league)} retained-salary slots"
                    )
                p_years = _contract_years_for_retention(player)
                # An expiring contract is still running before the deadline — retaining on a
                # rental is the most common deadline structure.
                in_season_expiring = (
                    p_years <= 0
                    and ctx.get("days_to_deadline") is not None
                    and int(ctx.get("days_to_deadline") or 0) >= 0
                    and player_cap_hit_millions(player) > 0
                )
                if in_season_expiring:
                    p_years = 1
                if p_years <= 0:
                    blocking.append(
                        f"{pname} has no contract years remaining — cannot retain salary on this trade"
                    )
                elif asset.retained_pct > 0:
                    prior = float(getattr(player, "retained_share_pct", 0) or 0)
                    if prior + float(asset.retained_pct) > float(max_retention_pct(league)) + 0.01:
                        blocking.append(
                            f"{pname}: retention would stack past the {max_retention_pct(league):.0f}% cap"
                        )

        elif isinstance(asset, DraftPickTradeAsset):
            pid = resolve_pick_id(asset.pick_id, asset.source_team_id)
            if pid in seen_picks:
                blocking.append(f"Duplicate pick in trade package: {pid}")
            seen_picks.add(pid)
            if post_deadline_freeze_active(ctx):
                blocking.append(f"Draft pick {pid}: {POST_DEADLINE_BLOCK_REASON}")
                continue

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
            # NHL: only picks in the next three drafts may be traded.
            if pick_year < draft_year or pick_year > draft_year + 2:
                blocking.append(
                    f"Only picks in the next three drafts ({draft_year}–{draft_year + 2}) can be traded — "
                    f"the {pick_year} pick isn't tradeable yet"
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
                p, _loc, _i = find_player_in_organization(team, a.player_id)
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
        # Only players moving on/off the active NHL roster use a spot; AHL pieces go to the
        # affiliate. Overflow is handled by same-day send-downs instead of killing the deal.
        roster_out_n, roster_in_n, arriving_nhl = 0, 0, []
        try:
            from app.sim_engine.economy.cap_engine import _is_active_roster_player
            from app.sim_engine.trades.roster_balance import demotion_capacity

            for a in package.outgoing_by_team.get(tid, []):
                if isinstance(a, PlayerTradeAsset):
                    p, loc, _i = find_player_in_organization(team, a.player_id)
                    if p is not None and loc == "nhl" and _is_active_roster_player(p):
                        roster_out_n += 1
            for a in package.incoming_by_team.get(tid, []):
                if isinstance(a, PlayerTradeAsset):
                    src = team_by_id.get(a.source_team_id)
                    p, loc, _i = find_player_in_organization(src, a.player_id) if src is not None else (None, "", -1)
                    if p is not None and loc == "nhl":
                        roster_in_n += 1
                        arriving_nhl.append(p)
            roster_flex = demotion_capacity(
                team,
                leaving_ids=[str(a.player_id) for a in out_assets],
                arriving=arriving_nhl,
            )
        except Exception:
            roster_out_n, roster_in_n, roster_flex = None, None, 0
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
            roster_out_n=roster_out_n,
            roster_in_n=roster_in_n,
            roster_flex=roster_flex,
        )
        if int(cap_check.get("rosterSendDowns") or 0) > 0:
            warnings.append(
                f"{tid}: will assign {int(cap_check['rosterSendDowns'])} player(s) to the AHL to make roster room"
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

        proj_raw = int(cap_check.get("projectedRosterCount", snap_before.get("activeRosterCount", 0)))
        send_downs = int(cap_check.get("rosterSendDowns") or 0)
        # Overflow players are assigned to the AHL as part of the deal, so the NHL roster
        # after the trade is the post-assignment count (the hard org limit is 50 SPCs).
        proj_count = proj_raw - send_downs
        roster_impact[tid] = {
            "before": int(snap_before.get("activeRosterCount", 0)),
            "after": proj_count,
            "after_before_send_downs": proj_raw,
            "send_downs": send_downs,
            "outgoing_players": len(outgoing),
            "incoming_players": len(incoming),
        }

        before_count = int(snap_before.get("activeRosterCount", 0))
        # Only block trades that push a club over (or further over) the max — a club
        # already carrying 24+ must still be able to make a trade that shrinks its roster.
        if proj_count > ROSTER_MAX and proj_count > before_count:
            blocking.append(f"{tid} would exceed maximum roster size ({proj_count} > {ROSTER_MAX})")
        elif proj_count > ROSTER_MAX:
            warnings.append(f"{tid} remains over the roster maximum ({proj_count} > {ROSTER_MAX}) after this trade")
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
