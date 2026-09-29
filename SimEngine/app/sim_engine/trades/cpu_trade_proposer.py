"""
CPU-CPU trade proposer — routes ambient trades through the full trade engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.trades.trade_asset import team_id_of
from app.sim_engine.trades.trade_pick_registry import (
    ensure_franchise_pick_registry,
    get_team_owned_picks,
    upcoming_draft_year,
)
from app.sim_engine.trades.trade_value import (
    evaluate_player_asset_value,
    evaluate_pick_asset_value,
    reduced_trade_value_fallback,
)
from app.sim_engine.economy.team_needs import TeamNeeds

logger = logging.getLogger(__name__)

CPU_AMBIENT_FAIRNESS_GAP_MAX = 14.0
CPU_AMBIENT_MIN_INTEREST = 0.40
CPU_PAIR_COOLDOWN_DAYS = 18
CPU_REACQUIRE_SOFT_DAYS = 35
CPU_SEASON_PAIR_SOFT_CAP = 2
CPU_ONE_FOR_ONE_OVR_GAP_MAX = 7.0  # allow more talent asymmetry without futures
CPU_SELLER_CORE_OVR = 86.0
CPU_YOUNG_CORE_MAX_AGE = 23
CPU_PROSPECT_MAX_AGE = 22
CPU_DESPERATION_GAP_MAX = 28.0
CPU_PANIC_BUYS_PER_SEASON = 2  # a sliding contender goes all-in once or twice, not every day

# Motives produced by the needs matcher (+ demand path).
PACKAGE_MOTIVES = (
    "demand_resolution",
    "panic_buy",
    "tank_selloff",
    "late_selloff",
    "rental_purchase",
    "seller_futures",
    "hockey_swap",
    "depth_add",
    "surplus_sale",
    "locker_room",
    "cap_dump",
)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _player_ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            v = float(fn())
        except Exception:
            return 0.0
    else:
        v = _safe_float(getattr(player, "ovr", None), 0.0)
    return v * 99.0 if v <= 1.5 else v


def _player_id(player: Any) -> str:
    return str(getattr(player, "id", "") or "")


def _player_pos_bucket(player: Any) -> str:
    ident = getattr(player, "identity", None)
    pos = str(getattr(getattr(ident, "position", None), "value", getattr(ident, "position", "")) or "").upper()
    if pos in ("G", "GOALIE", "GOALTENDER"):
        return "goalie"
    if pos in ("D", "LD", "RD", "DEFENSE"):
        return "defense"
    return "forward"


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    return _safe_int(getattr(ident, "age", getattr(player, "age", 25)), 25)


def _is_prospect(player: Any) -> bool:
    """Young development pieces — expanded from age<=21 to age<=23."""
    return _player_age(player) <= CPU_PROSPECT_MAX_AGE


def _player_potential_ovr(player: Any) -> float:
    for key in ("potential_ovr", "potential", "ceiling", "pot"):
        raw = getattr(player, key, None)
        if raw is None:
            ident = getattr(player, "identity", None)
            raw = getattr(ident, key, None) if ident is not None else None
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        return v * 99.0 if v <= 1.5 else v
    return _player_ovr(player)


def _is_elc_player(player: Any) -> bool:
    c = getattr(player, "contract", None)
    for obj in (c, player):
        if obj is None:
            continue
        for key in ("is_entry_level", "is_elc", "elc"):
            if bool(getattr(obj, key, False)):
                return True
        ctype = str(getattr(obj, "contract_type", None) or getattr(obj, "type", "") or "").upper()
        if ctype == "ELC":
            return True
    return False


def _is_young_core(player: Any) -> bool:
    """NHL-ready young pieces rebuilders must not casually dump."""
    age = _player_age(player)
    ovr = _player_ovr(player)
    pot = _player_potential_ovr(player)
    if age > CPU_YOUNG_CORE_MAX_AGE:
        return False
    if _is_elc_player(player) and ovr >= 72:
        return True
    if age <= CPU_PROSPECT_MAX_AGE and (ovr >= 74 or pot >= 82):
        return True
    if age <= CPU_YOUNG_CORE_MAX_AGE and ovr >= 78:
        return True
    if age <= CPU_YOUNG_CORE_MAX_AGE and pot >= 86 and ovr >= 70:
        return True
    return False


def _is_rental(player: Any) -> bool:
    c = getattr(player, "contract", None)
    years = 0
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("years_remaining", "term_remaining", "remaining_years", "term"):
            years = max(years, _safe_int(getattr(obj, key, 0), 0))
    if years > 1:
        return False
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("expiry_status", "ufa_rfa_status", "rights_status", "rights"):
            val = str(getattr(obj, key, "") or "").strip().upper()
            if val == "UFA":
                return True
    age = _player_age(player)
    return years <= 1 and age >= 28


def _seller_must_protect(player: Any, *, window: str, deadline: float) -> bool:
    """Hard seller protections — young core / ELC / mid-core stay unless true late rental sell."""
    ovr = _player_ovr(player)
    rental = _is_rental(player)
    late_sell = window in ("rebuild", "declining") and deadline > 0.72 and rental
    if ovr >= 88 and not late_sell:
        return True
    if _is_young_core(player) and not late_sell:
        return True
    if window in ("rebuild", "declining") and ovr >= CPU_SELLER_CORE_OVR and not rental:
        # Rebuilders do not move 82+ non-rentals via ambient without futures motive.
        return True
    if _is_prospect(player) and ovr >= 76 and not late_sell:
        return True
    return False


def _is_reverse_to_prior(player: Any, acquiring_team_id: str, ctx: Optional[Dict[str, Any]] = None) -> bool:
    from app.sim_engine.trades.trade_rules import _player_returning_to_prior_club

    return _player_returning_to_prior_club(player, acquiring_team_id, ctx)


def _team_window(team: Any) -> str:
    for key in ("gm_window", "window"):
        w = str(getattr(team, key, "") or "").lower()
        if w in ("rebuild", "contender", "declining", "emerging"):
            return w
    st = str(getattr(team, "status", "") or "").lower()
    if "rebuild" in st or "tank" in st:
        return "rebuild"
    if "contend" in st:
        return "contender"
    return "emerging"


def _normalize_competitive_window(raw: Any) -> str:
    """Map profile direction/window tokens onto the four ambient trade windows."""
    cw = str(raw or "").lower().strip()
    if cw in ("rebuild", "tank", "declining", "rebuilding", "deep_rebuild", "seller", "cap_correction"):
        return "rebuild"
    if cw in ("contender", "all_in_contender", "playoff_buyer", "contender_push"):
        return "contender"
    if cw in ("emerging", "competitive_retool", "holding", "balanced", "retool"):
        return "emerging"
    return "emerging"


def _playoff_odds(team: Any) -> float:
    for key in ("playoff_odds", "playoffOdds", "playoff_probability"):
        v = getattr(team, key, None)
        if v is not None:
            f = _safe_float(v, -1.0)
            return f / 100.0 if f > 1.0 else f
    return 0.5


def _needs_fit_score(team: Any, player: Any, *, selling: bool) -> float:
    needs = getattr(team, "needs", None) or {}
    pos = _player_pos_bucket(player)
    score = 0.0
    if pos == "goalie":
        score += _safe_float(needs.get("goalie"), 0.0) * 12.0
    elif pos == "defense":
        score += _safe_float(needs.get("top_4_defense"), 0.0) * 10.0
        if selling:
            score -= max(0.0, 0.45 - _safe_float(needs.get("top_4_defense"), 0.0)) * 8.0
    else:
        score += _safe_float(needs.get("top_line_forward"), 0.0) * 9.0
        score += _safe_float(needs.get("depth_forward"), 0.0) * 5.0
        if selling:
            score -= max(0.0, 0.5 - _safe_float(needs.get("depth_forward"), 0.0)) * 6.0
    return score


def _tradeable_player(player: Any, acquiring_team_id: str, *, ctx: Optional[Dict[str, Any]] = None) -> bool:
    from app.sim_engine.trades.trade_rules import _clause_summary, _player_recently_acquired

    clause = _clause_summary(player)
    if clause.get("nmc") or clause.get("ntc"):
        return False
    if clause.get("mntc", 0) > 0:
        approved = clause.get("approved_destinations") or []
        if not (bool(approved) and str(acquiring_team_id) in approved):
            return False
    # Hard reverse-to-prior-club ban (remainder of acquisition season).
    if _is_reverse_to_prior(player, acquiring_team_id, ctx):
        return False
    if ctx is not None and _player_recently_acquired(player, ctx):
        return False
    if ctx is not None and bool(getattr(player, "acquired_via_trade", False)):
        cursor = int(ctx.get("calendar_cursor", 0) or 0)
        last_day = getattr(player, "last_acquired_day", None)
        try:
            if last_day is not None and (cursor - int(last_day)) < CPU_REACQUIRE_SOFT_DAYS:
                return False
        except (TypeError, ValueError):
            pass
    return True


def build_team_by_id(league: Any) -> Dict[str, Any]:
    teams = list(getattr(league, "teams", None) or [])
    out: Dict[str, Any] = {}
    for t in teams:
        tid = team_id_of(t)
        if tid:
            out[tid] = t
    return out


def build_league_trade_context(
    league: Any,
    *,
    calendar_cursor: int = 0,
    regular_season_last_index: int = 192,
    season_year: Optional[int] = None,
    draft_year: Optional[int] = None,
) -> Dict[str, Any]:
    if season_year is None:
        season_year = int(getattr(league, "current_season", 0) or getattr(league, "season_year", 2025) or 2025)
    else:
        season_year = int(season_year)
    if draft_year is None:
        draft_year = int(getattr(league, "draft_year", 0) or 0) or upcoming_draft_year(season_year)
    else:
        draft_year = int(draft_year)
    from app.sim_engine.trades.trade_deadline import (
        days_to_deadline,
        deadline_phase as _deadline_phase_for,
        freeze_applies_to_phase,
    )

    max_d = max(40, int(regular_season_last_index or 192))
    deadline_phase = _deadline_phase_for(league, int(calendar_cursor or 0), max_d)
    days_left = days_to_deadline(league, int(calendar_cursor or 0), max_d)
    phase = str(getattr(league, "_franchise_phase", "") or "regular")
    team_by_id = build_team_by_id(league)
    return {
        "league": league,
        "team_by_id": team_by_id,
        "season_year": season_year,
        "draft_year": draft_year,
        "season_is_calendar": True,
        "use_upcoming_draft_year": True,
        "calendar_cursor": int(calendar_cursor or 0),
        "regular_season_last_index": max_d,
        "deadline_phase": deadline_phase,
        "days_to_deadline": int(days_left),
        "trade_deadline_passed": bool(days_left < 0 and freeze_applies_to_phase(phase)),
        "player_season_stats": getattr(league, "player_season_stats", None),
    }


def _player_trade_value(
    player: Any,
    team: Any,
    league: Any,
    ctx: Dict[str, Any],
    *,
    acquiring_team: Any = None,
) -> float:
    try:
        acq = acquiring_team if acquiring_team is not None else team
        result = evaluate_player_asset_value(player, team, acq, league, context=ctx)
        return float(result.get("total", 0.0))
    except Exception as exc:
        logger.exception(
            "_player_trade_value failed player_id=%s: %s",
            str(getattr(player, "id", None) or ""),
            exc,
        )
        return reduced_trade_value_fallback(player, reason=f"cpu:{type(exc).__name__}")


def _pick_trade_candidates(
    roster: List[Any],
    team: Any,
    *,
    seller: bool,
    league: Any,
    ctx: Dict[str, Any],
    acquiring_team_id: str,
    motive: str = "depth_swap",
) -> List[Any]:
    """Rank movable players by motive — sellers protect young/core; rentals preferred for sales."""
    if not roster:
        return []
    deadline = _safe_float(ctx.get("deadline_phase"), 0.0)
    window = _team_window(team)
    scored: List[Tuple[float, float, Any]] = []
    for p in roster:
        if not _tradeable_player(p, acquiring_team_id, ctx=ctx):
            continue
        ovr = _player_ovr(p)
        rental = _is_rental(p)
        if seller and _seller_must_protect(p, window=window, deadline=deadline):
            # Futures packages may move non-young mid-core (82–87) only when a pick will be required.
            if (
                motive == "futures_package"
                and not _is_young_core(p)
                and ovr < 88
                and window in ("rebuild", "declining")
            ):
                pass
            elif motive in ("rental_sale", "futures_package", "desperation") and rental:
                pass
            elif motive == "desperation" and window in ("rebuild", "declining") and deadline > 0.8 and ovr < 86:
                pass
            elif motive == "depth_swap" and ovr < 81:
                pass
            else:
                continue
        if ovr >= 90 and motive not in ("star_acquisition", "desperation"):
            if not (rental and deadline > 0.70 and window in ("rebuild", "declining")):
                continue
        if ovr >= 88 and motive not in ("star_acquisition", "desperation", "rental_sale", "futures_package"):
            if not (rental and deadline > 0.75 and window in ("rebuild", "declining")):
                continue
        val = _player_trade_value(p, team, league, ctx, acquiring_team=ctx.get("_acquiring_team"))
        fit = _needs_fit_score(team, p, selling=seller)
        priority = fit
        demand = bool(getattr(p, "_trade_demand_active", False) or getattr(p, "trade_demand_active", False))
        if demand:
            priority += 18.0
        if seller:
            if motive in ("rental_sale", "futures_package", "desperation", "star_acquisition"):
                if rental and window in ("rebuild", "declining"):
                    priority += 16.0 + deadline * 10.0
                elif 78.0 <= ovr <= 87.0 and _player_age(p) >= 25:
                    priority += 11.0
                elif ovr >= CPU_SELLER_CORE_OVR and not rental and not demand:
                    priority -= 14.0
            else:
                if rental and window in ("rebuild", "declining") and _playoff_odds(team) < 0.35:
                    priority += 12.0 + deadline * 8.0
                elif 76.0 <= ovr <= 86.0 and _player_age(p) >= 24:
                    priority += 9.0
                elif 68.0 <= ovr < 76.0 and _player_age(p) >= 26:
                    priority += 3.0
                elif ovr > 87 and not demand:
                    priority -= 6.0
                if window in ("rebuild", "declining") and ovr >= 84 and not rental and not demand:
                    priority -= 6.0
        else:
            if motive == "star_acquisition":
                if ovr >= 82:
                    priority += 10.0
                if rental and window == "contender" and deadline > 0.4:
                    priority += 6.0
            elif motive == "rental_sale":
                if rental and window == "contender" and deadline > 0.35:
                    priority += 12.0
                elif 74.0 <= ovr <= 86.0:
                    priority += 5.0
            else:
                if rental and window == "contender" and deadline > 0.45:
                    priority += 10.0
                elif 74.0 <= ovr <= 86.0:
                    priority += 6.0
                if ovr > 88:
                    priority -= 3.0
        scored.append((priority, val, p))
    if seller:
        scored.sort(key=lambda x: (-x[0], abs(x[1] - 55.0)))
    else:
        scored.sort(key=lambda x: (-x[0], abs(x[1] - 52.0)))
    return [p for _, _, p in scored]


def _match_return_player(
    *,
    seller_asset: Any,
    seller: Any,
    buyer: Any,
    buyer_candidates: List[Any],
    league: Any,
    ctx: Dict[str, Any],
    used_players: set,
    value_band: float = 9.0,
    motive: str = "depth_swap",
) -> Optional[Any]:
    target = _player_trade_value(seller_asset, seller, league, ctx, acquiring_team=buyer)
    sold_ovr = _player_ovr(seller_asset)
    ranked: List[Tuple[float, Any]] = []
    for p in buyer_candidates:
        pid = _player_id(p)
        if not pid or pid in used_players or pid == _player_id(seller_asset):
            continue
        if _is_reverse_to_prior(p, team_id_of(seller), ctx):
            continue
        fit_min = -1.5 if motive == "depth_swap" else 0.5
        if _needs_fit_score(buyer, p, selling=False) < fit_min and not _is_rental(seller_asset):
            continue
        # Depth / peer swaps: keep OVR close even before pick compensation.
        if motive == "depth_swap" and abs(_player_ovr(p) - sold_ovr) > CPU_ONE_FOR_ONE_OVR_GAP_MAX + 1.5:
            continue
        val = _player_trade_value(p, buyer, league, ctx, acquiring_team=seller)
        gap = abs(val - target)
        if gap <= value_band:
            ranked.append((gap - 0.15 * _needs_fit_score(buyer, p, selling=False), p))
    if not ranked and motive != "depth_swap":
        for p in buyer_candidates:
            pid = _player_id(p)
            if not pid or pid in used_players or pid == _player_id(seller_asset):
                continue
            if _is_reverse_to_prior(p, team_id_of(seller), ctx):
                continue
            val = _player_trade_value(p, buyer, league, ctx, acquiring_team=seller)
            gap = abs(val - target)
            if gap <= value_band * 1.65:
                ranked.append((gap, p))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]


def _roster_filler(
    seller: Any,
    buyer: Any,
    incoming: Any,
    outgoing_return: Any,
    *,
    ctx: Dict[str, Any],
    exclude: set,
) -> Optional[Any]:
    """Roster player going back when a full club adds a body (keeps both rosters legal).

    Real GMs send a depth player the other way (or down) — without this every
    pick-for-player deal failed the 23-man check, so only 1-for-1 swaps ever cleared.
    """
    from app.sim_engine.trades.trade_rules import ROSTER_MAX

    roster = list(getattr(buyer, "roster", None) or [])
    seller_nhl = {_player_id(p) for p in list(getattr(seller, "roster", None) or [])}
    adds = 1 if _player_id(incoming) in seller_nhl else 0
    sends = 1 if outgoing_return is not None and any(p is outgoing_return for p in roster) else 0
    # Same active count trade validation uses (IR/LTIR excluded) — len(roster) was off.
    try:
        from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot

        active = int(calculate_team_cap_snapshot(buyer, league=ctx.get("league")).get("activeRosterCount", len(roster)))
    except Exception:
        active = len(roster)
    if active + adds - sends <= ROSTER_MAX:
        return None
    # (callers add one body; a second over-max case is handled by salary ballast / validation)
    sid = team_id_of(seller)
    cands = [
        p
        for p in roster
        if p is not outgoing_return
        and _player_pos_bucket(p) != "goalie"
        and _player_id(p) not in exclude
        and not bool(getattr(p, "_trade_demand_active", False))
        and _tradeable_player(p, sid, ctx=ctx)
    ]
    if not cands:
        return None
    return min(cands, key=lambda p: (_player_ovr(p), -_player_age(p)))


def _retention_to_fit_cap(
    buyer: Any,
    incoming: Any,
    outgoing: List[Any],
    *,
    league: Any,
    ctx: Dict[str, Any],
) -> Optional[float]:
    """Smallest seller retention (0–50%) that fits ``incoming`` under the buyer's cap; None if nothing fits."""
    try:
        from app.sim_engine.economy.cap_engine import can_trade_cap_fit
    except Exception:
        return 0.0
    pid = _player_id(incoming)
    for pct in (0.0, 15.0, 25.0, 35.0, 50.0):
        try:
            chk = can_trade_cap_fit(
                buyer,
                outgoing,
                [incoming],
                league=league,
                incoming_retained_pct={pid: pct} if pct else None,
                calendar_cursor=int(ctx.get("calendar_cursor", 0) or 0),
                regular_season_last_index=int(ctx.get("regular_season_last_index", 192) or 192),
                deadline_phase=float(ctx.get("deadline_phase", 0.0) or 0.0),
            )
        except Exception:
            return 0.0
        if chk.get("ok") or chk.get("reason") in ("ok_with_ltir", "ok_with_accrual"):
            return pct
    return None


def _salary_ballast(buyer: Any, exclude: set, *, ctx: Dict[str, Any], seller_id: str) -> List[Any]:
    """Buyer depth players (never top guys / tandem G) ordered by cap hit — salary going back."""
    roster = list(getattr(buyer, "roster", None) or [])
    by_pos: Dict[str, List[Any]] = {}
    for p in roster:
        by_pos.setdefault(_player_pos_bucket(p), []).append(p)
    core: set = set()
    for pos, plist in by_pos.items():
        plist.sort(key=_player_ovr, reverse=True)
        keep = {"forward": 6, "defense": 4, "goalie": 2}.get(pos, 4)
        core.update(_player_id(p) for p in plist[:keep])
    cands = [
        p for p in roster
        if _player_id(p) not in core
        and _player_id(p) not in exclude
        and not bool(getattr(p, "_trade_demand_active", False))
        and _tradeable_player(p, seller_id, ctx=ctx)
    ]
    cands.sort(key=lambda p: -_cap_hit(p))
    return cands[:4]


def _cap_hit(player: Any) -> float:
    try:
        from app.sim_engine.economy.cap_engine import player_cap_hit_millions

        return float(player_cap_hit_millions(player))
    except Exception:
        return 0.0


def _active_count(team: Any, league: Any) -> int:
    try:
        from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot

        return int(calculate_team_cap_snapshot(team, league=league).get("activeRosterCount", 0))
    except Exception:
        return len(list(getattr(team, "roster", None) or []))


def _paper_send_down(team: Any, player: Any) -> None:
    """Assign a player to the club's AHL roster (restored if the trade falls through)."""
    roster = [p for p in list(getattr(team, "roster", None) or []) if p is not player]
    team.roster = roster
    ahl = list(getattr(team, "ahl_roster", None) or [])
    ahl.append(player)
    team.ahl_roster = ahl


def _paper_recall(team: Any, player: Any) -> None:
    team.ahl_roster = [p for p in list(getattr(team, "ahl_roster", None) or []) if p is not player]
    roster = list(getattr(team, "roster", None) or [])
    if not any(p is player for p in roster):
        roster.append(player)
    team.roster = roster


def _build_package(
    seller: Any,
    buyer: Any,
    seller_asset: Any,
    buyer_assets: List[Any],
    *,
    seller_pick: Optional[Dict[str, Any]] = None,
    buyer_pick: Optional[Dict[str, Any]] = None,
    buyer_pick_2: Optional[Dict[str, Any]] = None,
    retained_pct: float = 0.0,
    return_retained: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    sid = team_id_of(seller)
    bid = team_id_of(buyer)
    spid = _player_id(seller_asset)
    if not spid:
        return {}
    asset: Dict[str, Any] = {"type": "player", "id": spid, "team": sid}
    if retained_pct > 0:
        asset["retained"] = float(retained_pct)
    buyer_payload: List[Dict[str, Any]] = [asset]
    seller_payload: List[Dict[str, Any]] = []
    for asset in buyer_assets:
        if isinstance(asset, dict):
            seller_payload.append(asset)
        else:
            bpid = _player_id(asset)
            if bpid:
                row: Dict[str, Any] = {"type": "player", "id": bpid, "team": bid}
                if (return_retained or {}).get(bpid):
                    row["retained"] = float(return_retained[bpid])
                seller_payload.append(row)
    if seller_pick:
        buyer_payload.append(
            {
                "type": "pick",
                "id": str(seller_pick.get("pick_id") or ""),
                "team": sid,
            }
        )
    if buyer_pick:
        seller_payload.append(
            {
                "type": "pick",
                "id": str(buyer_pick.get("pick_id") or ""),
                "team": bid,
            }
        )
    if buyer_pick_2:
        seller_payload.append(
            {
                "type": "pick",
                "id": str(buyer_pick_2.get("pick_id") or ""),
                "team": bid,
            }
        )
    if not seller_payload:
        return {}
    return {bid: buyer_payload, sid: seller_payload}


_REASON_COPY = {
    "TOP_SIX_SCORING_NEED": "Added top-six scoring for the playoff push.",
    "MIDDLE_SIX_SCORING_NEED": "Added middle-six scoring depth.",
    "CENTRE_DEPTH_NEED": "Addressed a need at centre.",
    "BOTTOM_SIX_DEPTH": "Bolstered bottom-six depth.",
    "TOP_PAIR_DEFENCE_NEED": "Upgraded the top defence pair.",
    "SECOND_PAIR_DEFENCE_NEED": "Added second-pair defence.",
    "DEFENSIVE_DEPTH_NEED": "Added defensive depth.",
    "PUCK_MOVING_DEFENCE_NEED": "Added puck-moving defence.",
    "STARTING_GOALIE_NEED": "Addressed starting goaltending.",
    "BACKUP_GOALIE_NEED": "Added goaltending depth.",
    "GOALTENDING_INSURANCE": "Added goaltending insurance.",
    "INJURY_REPLACEMENT": "Filled a hole created by injury.",
    "PLAYOFF_DEPTH": "Added playoff depth.",
    "DEADLINE_RENTAL": "Acquired a low-cost rental before the deadline.",
    "LONG_TERM_CORE_TARGET": "Targeted a longer-term core piece.",
    "CAP_EFFICIENT_UPGRADE": "Found a cap-efficient upgrade.",
    "ROSTER_BALANCE": "Rebalanced the roster.",
    "STAR_ACQUISITION": "Acquired a high-impact roster piece.",
    "PROSPECT_TIMELINE_FIT": "Moved a prospect who fit the timeline better elsewhere.",
    "PENDING_UFA_SALE": "Moved an expiring veteran for future assets.",
    "REBUILDING_FUTURES": "Moved a veteran for future assets.",
    "DEEP_REBUILD_ASSET_SALE": "Moved a veteran during a deep rebuild.",
    "PLAYOFF_ODDS_COLLAPSE": "Sold after playoff odds collapsed.",
    "AGING_VETERAN": "Moved a veteran who no longer fit the timeline.",
    "TIMELINE_MISMATCH": "Exchanged assets that better fit each timeline.",
    "CAP_RELIEF": "Cleared cap space.",
    "CAP_COMPLIANCE": "Cleared cap space for compliance.",
    "ROSTER_SURPLUS": "Moved surplus roster depth.",
    "GOALTENDER_SURPLUS": "Moved an extra goaltender.",
    "PROSPECT_BLOCKED": "Opened a path for a younger player.",
    "DRAFT_CAPITAL_RECOVERY": "Recovered draft capital in the deal.",
    "YOUNG_PLAYER_TARGET": "Acquired a younger NHL-ready piece.",
    "WAIVER_AVOIDANCE": "Moved a player before a waiver risk.",
    "RETOOLING_SWAP": "Completed a retooling roster swap.",
    "POSITIONAL_SWAP": "Exchanged positional surplus for a better roster fit.",
    "AGE_TIMELINE_SWAP": "Swapped assets across age timelines.",
    "SIMILAR_VALUE_DIFFERENT_NEED": "Swapped similar-value assets for different needs.",
    "PICK_VALUE_REALLOCATION": "Reallocated draft capital.",
    "DESPERATION_OVERPAY": "Paid a premium in a desperate push.",
    "DESPERATION_FIRE_SALE": "Accepted a thin return under heavy pressure.",
    "TRADE_DEMAND_RESOLVED": "Granted a player's trade request.",
    "LOCKER_ROOM_DISRUPTOR_MOVED": "Moved a disruptive presence out of the room.",
}


def _classify_trade_reasons(
    *,
    seller: Any,
    buyer: Any,
    sold_player: Any,
    return_player: Any,
    deadline_phase: float,
    seller_pick: Optional[Dict[str, Any]],
    buyer_pick: Optional[Dict[str, Any]],
) -> Tuple[str, List[str], str, str]:
    seller_window = _team_window(seller)
    buyer_window = _team_window(buyer)
    buyer_needs = getattr(buyer, "needs", None) or {}
    seller_direction = str(getattr(seller, "_cpu_direction_state", "") or "").upper()
    buyer_direction = str(getattr(buyer, "_cpu_direction_state", "") or "").upper()
    reasons: List[str] = []
    category = "hockey_trade"
    pos_bucket = _player_pos_bucket(sold_player)
    sold_age = _safe_int(getattr(getattr(sold_player, "identity", None), "age", getattr(sold_player, "age", 25)), 25)
    sold_ovr = _player_ovr(sold_player)

    if pos_bucket == "goalie":
        if _safe_float(buyer_needs.get("goalie"), 0.0) >= 0.62:
            category = "goalie_trade"
            reasons.append("STARTING_GOALIE_NEED")
        elif _safe_float(buyer_needs.get("goalie"), 0.0) >= 0.4:
            category = "goalie_trade"
            reasons.append("BACKUP_GOALIE_NEED")
        elif seller_window in ("rebuild", "declining") or seller_direction in ("REBUILDING", "DEEP_REBUILD", "SELLER"):
            category = "goalie_trade"
            reasons.append("GOALTENDER_SURPLUS")
    if _is_rental(sold_player) and deadline_phase >= 0.35:
        category = "deadline_rental" if deadline_phase >= 0.62 else category
        reasons.append("DEADLINE_RENTAL")
        if sold_age >= 30:
            reasons.append("PENDING_UFA_SALE")
    if seller_window in ("rebuild", "declining") or seller_direction in ("REBUILDING", "DEEP_REBUILD", "SELLER"):
        reasons.append("DEEP_REBUILD_ASSET_SALE" if seller_direction == "DEEP_REBUILD" else "REBUILDING_FUTURES")
        if buyer_pick is not None:
            category = "futures_trade"
            reasons.append("DRAFT_CAPITAL_RECOVERY")
        if sold_age >= 30:
            reasons.append("AGING_VETERAN")
        if deadline_phase >= 0.45:
            reasons.append("PLAYOFF_ODDS_COLLAPSE")
    if buyer_window == "contender" or buyer_direction in ("CONTENDER", "PLAYOFF_BUYER", "ALL_IN_CONTENDER"):
        reasons.append("PLAYOFF_DEPTH")
        if _safe_float(buyer_needs.get("top_line_forward"), 0.0) >= 0.55:
            reasons.append("TOP_SIX_SCORING_NEED")
        elif _safe_float(buyer_needs.get("middle_six"), 0.0) >= 0.5:
            reasons.append("MIDDLE_SIX_SCORING_NEED")
        elif _safe_float(buyer_needs.get("center"), 0.0) >= 0.5:
            reasons.append("CENTRE_DEPTH_NEED")
        elif _safe_float(buyer_needs.get("top_4_defense"), 0.0) >= 0.55:
            reasons.append("TOP_PAIR_DEFENCE_NEED")
        elif _safe_float(buyer_needs.get("defense"), 0.0) >= 0.5:
            reasons.append("DEFENSIVE_DEPTH_NEED")
        elif pos_bucket == "forward" and sold_ovr < 78:
            reasons.append("BOTTOM_SIX_DEPTH")
        if sold_ovr >= 88:
            reasons.append("STAR_ACQUISITION")
        if deadline_phase >= 0.55 and buyer_direction == "ALL_IN_CONTENDER":
            reasons.append("LONG_TERM_CORE_TARGET" if not _is_rental(sold_player) else "DEADLINE_RENTAL")
    if return_player is not None and (_is_prospect(sold_player) or _is_prospect(return_player)):
        reasons.append("PROSPECT_TIMELINE_FIT")
        if _is_prospect(sold_player) and buyer_window == "contender":
            reasons.append("PROSPECT_BLOCKED")
        category = "prospect_trade" if "prospect" not in category else category
    elif _is_prospect(sold_player):
        reasons.append("PROSPECT_TIMELINE_FIT")
        if buyer_window == "contender":
            reasons.append("PROSPECT_BLOCKED")
        category = "prospect_trade" if "prospect" not in category else category
    if seller_pick is not None or buyer_pick is not None:
        reasons.append("PICK_VALUE_REALLOCATION")
    if _safe_float(getattr(seller, "cap_pressure", 0.0), 0.0) >= 0.8 or seller_direction == "CAP_CORRECTION":
        category = "cap_trade"
        reasons.append("CAP_RELIEF" if _safe_float(getattr(seller, "cap_pressure", 0.0), 0.0) < 0.92 else "CAP_COMPLIANCE")
    if seller_window == buyer_window and not reasons:
        reasons = ["SIMILAR_VALUE_DIFFERENT_NEED", "POSITIONAL_SWAP"]
    if return_player is not None and abs(
        sold_age - _safe_int(getattr(getattr(return_player, "identity", None), "age", getattr(return_player, "age", 25)), 25)
    ) >= 6:
        reasons.append("AGE_TIMELINE_SWAP")
    if not reasons:
        reasons = ["POSITIONAL_SWAP", "ROSTER_BALANCE"] if return_player is not None else ["REBUILDING_FUTURES", "DRAFT_CAPITAL_RECOVERY"]
    # Dedupe while preserving order
    deduped: List[str] = []
    for code in reasons:
        if code not in deduped:
            deduped.append(code)
    reasons = deduped[:4]
    reason_text = next((_REASON_COPY[c] for c in reasons if c in _REASON_COPY), "") or " · ".join(
        [r.replace("_", " ").title() for r in reasons[:2]]
    )
    # Importance hint for popup consumers (major stays rare).
    importance = "standard"
    if sold_ovr >= 88 or "STAR_ACQUISITION" in reasons:
        importance = "major"
        category = "major_trade" if category == "hockey_trade" else category
    elif category in ("deadline_rental", "cap_trade", "goalie_trade", "futures_trade", "prospect_trade"):
        importance = "standard"
    elif sold_ovr < 76 and not buyer_pick and not seller_pick:
        importance = "minor"
        category = "depth_trade" if category == "hockey_trade" else category
    return category, reasons, reason_text, importance


CPU_DEMAND_BASE_CHANCE = 0.22  # per market call, day the demand opens
CPU_DEMAND_DAILY_RAMP = 0.022  # GM patience erodes the longer it drags
CPU_DEMAND_BUYER_TRIES = 6
CPU_DEMAND_FAIRNESS_GAP_BASE = 20.0  # + 3 per crisis stage — sellers eat a discount
CPU_DEMAND_MIN_INTEREST = 0.30


def _team_abbr_upper(team: Any) -> str:
    return str(
        getattr(team, "abbr", None) or getattr(team, "abbreviation", None) or team_id_of(team) or ""
    ).strip().upper()


def collect_cpu_trade_demands(teams: List[Any], *, calendar_cursor: int) -> List[Tuple[Any, Any, int]]:
    """(team, player, days_open) for every open demand on these clubs, oldest first."""
    rows: List[Tuple[Any, Any, int]] = []
    for tm in teams:
        tid = team_id_of(tm)
        for p in list(getattr(tm, "roster", None) or []):
            if not bool(getattr(p, "_trade_demand_active", False)):
                continue
            opened = getattr(p, "_trade_demand_opened_day", None)
            if opened is None:
                continue
            owner = str(getattr(p, "_trade_demand_team_id", "") or "")
            if owner and owner != tid:
                continue  # stale flag from before a move; cleared on the next demand tick
            rows.append((tm, p, max(0, int(calendar_cursor) - _safe_int(opened, int(calendar_cursor)))))
    rows.sort(key=lambda r: -r[2])
    return rows


def demand_trade_chance(days_open: int, deadline: float, *, disruptor: bool) -> float:
    return min(
        0.92,
        CPU_DEMAND_BASE_CHANCE
        + CPU_DEMAND_DAILY_RAMP * max(0, int(days_open))
        + 0.45 * max(0.0, float(deadline))
        + (0.12 if disruptor else 0.0),
    )


def _demand_buyer_weight(buyer: Any, player: Any, dests: set) -> float:
    w = 1.0
    if _team_abbr_upper(buyer) in dests:
        w += 3.0
    window = _team_window(buyer)
    ovr = _player_ovr(player)
    if window == "contender":
        w += 1.2 if ovr >= 80 else 0.5
    elif window in ("rebuild", "declining") and _player_age(player) <= 26:
        w += 0.8
    elif window == "emerging":
        w += 0.4
    return w


def _pick_for_value(
    league: Any,
    team: Any,
    *,
    ctx: Dict[str, Any],
    target: float,
    exclude_pick_ids: Optional[set] = None,
    protect_first: bool = False,
    no_first: bool = False,
    value_cache: Optional[Dict[Tuple[str, str], float]] = None,
    acquirer: Any = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Owned pick whose value best matches ``target`` (slightly under preferred).

    Sizes futures to the player instead of reaching for "a mid 1st/2nd" every time.
    """
    if target <= 0:
        return None, 0.0
    tid = team_id_of(team)
    excluded = {str(x) for x in (exclude_pick_ids or set()) if x}
    best: Optional[Dict[str, Any]] = None
    best_val = 0.0
    best_score = float("inf")
    for row in get_team_owned_picks(league, tid):
        if bool(row.get("resolved")):
            continue
        if str(row.get("pick_id") or "") in excluded:
            continue
        if protect_first and _safe_int(row.get("round"), 7) == 1 and str(row.get("original_team_id") or "") == tid:
            continue
        if no_first and _safe_int(row.get("round"), 7) == 1:
            continue
        # Price it the way the RECEIVING club will (rebuilders value picks higher).
        acq = acquirer if acquirer is not None else team
        ck = (str(row.get("pick_id") or ""), team_id_of(acq))
        if value_cache is not None and ck in value_cache:
            val = value_cache[ck]
        else:
            try:
                val = float(evaluate_pick_asset_value(row, acq, team, league, context=ctx).get("total", 0.0))
            except Exception:
                val = max(1.0, 20.0 - _safe_int(row.get("round"), 7) * 3.0)
            if value_cache is not None:
                value_cache[ck] = val
        over = val - target
        score = abs(over) + (0.35 * over if over > 0 else 0.0)
        if score < best_score:
            best, best_val, best_score = row, val, score
    return best, best_val


def propose_and_execute_cpu_trades(
    league: Any,
    *,
    max_executions: int = 1,
    calendar_cursor: int = 0,
    regular_season_last_index: int = 192,
    fairness_gap_max: float = CPU_AMBIENT_FAIRNESS_GAP_MAX,
    season_year: Optional[int] = None,
    draft_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate and execute CPU-CPU trades using evaluate_trade_package + execute_validated_trade.
    Ambient trades require legality, partner interest, and fair value — no AI bypass.
    """
    teams = list(getattr(league, "teams", None) or [])
    if len(teams) < 2:
        return []

    user_tid = str(
        getattr(league, "_franchise_user_team_id", None)
        or getattr(league, "user_team_id", None)
        or ""
    )
    if user_tid:
        teams = [t for t in teams if team_id_of(t) != user_tid]
    if len(teams) < 2:
        return []

    ctx = build_league_trade_context(
        league,
        calendar_cursor=calendar_cursor,
        regular_season_last_index=regular_season_last_index,
        season_year=season_year,
        draft_year=draft_year,
    )
    ctx["cpu_ambient_trade"] = True
    if ctx.get("trade_deadline_passed"):
        return []  # hard deadline — only propose_ahl_depth_trades runs after it
    try:
        setattr(league, "season_year", int(ctx["season_year"]))
        setattr(league, "current_season", int(ctx["season_year"]))
        setattr(league, "draft_year", int(ctx["draft_year"]))
        setattr(league, "season_is_calendar", True)
    except Exception:
        pass
    ensure_franchise_pick_registry(league, season_calendar_year=int(ctx["season_year"]), years_ahead=4)
    team_by_id = ctx["team_by_id"]
    profiles = dict(getattr(league, "cpu_franchise_profiles", None) or {})
    for tm in teams:
        tid = team_id_of(tm)
        prof = profiles.get(tid) or {}
        cw = _normalize_competitive_window(
            prof.get("competitive_window") or prof.get("team_direction") or getattr(tm, "gm_window", "")
        )
        try:
            setattr(tm, "gm_window", cw)
        except Exception:
            pass
    needs_model = TeamNeeds()
    deadline = _safe_float(ctx.get("deadline_phase"), 0.0)

    def _direction_of(tm: Any) -> str:
        tid = team_id_of(tm)
        return str((profiles.get(tid) or {}).get("team_direction") or getattr(tm, "_cpu_direction_state", "") or "").upper()

    executed: List[Dict[str, Any]] = []
    used_pairs: set = set()
    used_players: set = set()
    partner_memory = getattr(league, "cpu_market_runtime", None)
    if not isinstance(partner_memory, dict):
        partner_memory = {}
    recent_pairs = partner_memory.get("recent_pair_days")
    if not isinstance(recent_pairs, dict):
        recent_pairs = {}
        partner_memory["recent_pair_days"] = recent_pairs
    season_pair_counts = partner_memory.get("season_pair_counts")
    if not isinstance(season_pair_counts, dict):
        season_pair_counts = {}
        partner_memory["season_pair_counts"] = season_pair_counts
    telemetry = partner_memory.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {
            "trades": 0,
            "with_pick": 0,
            "rebuild_sales": 0,
            "rebuild_sales_with_futures": 0,
            "desperation": 0,
            "by_motive": {},
            "ovr_gap_sum": 0.0,
            "ovr_gap_n": 0,
            "reverse_blocked": 0,
        }
        partner_memory["telemetry"] = telemetry
    setattr(league, "cpu_market_runtime", partner_memory)

    # Per-motive funnel: where each plan type dies (attempt → executed).
    funnel = telemetry.get("funnel")
    if not isinstance(funnel, dict):
        funnel = {}
        telemetry["funnel"] = funnel
    funnel_reasons = telemetry.get("funnel_reasons")
    if not isinstance(funnel_reasons, dict):
        funnel_reasons = {}
        telemetry["funnel_reasons"] = funnel_reasons

    def _funnel(motive_key: str, stage: str) -> None:
        row = funnel.setdefault(str(motive_key), {})
        row[stage] = int(row.get(stage, 0) or 0) + 1

    def _funnel_reason(motive_key: str, reasons: Any) -> None:
        # Collapse team names/numbers so the same rejection buckets together.
        import re as _re

        row = funnel_reasons.setdefault(str(motive_key), {})
        for r in list(reasons or [])[:2]:
            key = _re.sub(r"[-\d.]+", "#", str(r).split(": ", 1)[-1])[:90]
            row[key] = int(row.get(key, 0) or 0) + 1

    import random as _random

    day_seed = int(calendar_cursor) * 1009 + int(max_executions) * 17 + len(teams)
    base_rng = getattr(league, "rng", None)
    try:
        pair_rng = _random.Random(int(base_rng.randint(1, 2_000_000_000)) ^ day_seed) if hasattr(base_rng, "randint") else _random.Random(day_seed)
    except Exception:
        pair_rng = _random.Random(day_seed)
    team_trade_counts: Dict[str, int] = {}

    def _finalize_trade(
        *,
        seller: Any,
        buyer: Any,
        s_offer: Any,
        b_return: Any,
        pick_only: bool,
        seller_pick: Optional[Dict[str, Any]],
        buyer_pick: Optional[Dict[str, Any]],
        buyer_pick_2: Optional[Dict[str, Any]],
        motive: str,
        attempt_gap_max: float,
        min_interest: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evaluate → execute → classify → record. Returns True when the trade executed."""
        sid = team_id_of(seller)
        bid = team_id_of(buyer)
        pair_key = tuple(sorted((sid, bid)))
        pair_mem_key = f"{pair_key[0]}|{pair_key[1]}"
        season_count = int(season_pair_counts.get(pair_mem_key, 0) or 0)
        seller_window = _team_window(seller)
        buyer_window = _team_window(buyer)
        sold_ovr = _player_ovr(s_offer)
        buyer_assets: List[Any] = [] if pick_only else [b_return]
        filler = _roster_filler(
            seller, buyer, s_offer, None if pick_only else b_return, ctx=ctx, exclude=used_players,
        )
        if filler is not None:
            buyer_assets.append(filler)
        # Deadline deals routinely need the seller to retain salary to fit the buyer's cap,
        # and when that's not enough, salary goes back the other way.
        buyer_nhl_ids = {_player_id(p) for p in list(getattr(buyer, "roster", None) or [])}
        outgoing_nhl = [a for a in buyer_assets if a is not None and _player_id(a) in buyer_nhl_ids]
        retain = _retention_to_fit_cap(buyer, s_offer, outgoing_nhl, league=league, ctx=ctx)
        if retain is None:
            taken = {_player_id(a) for a in buyer_assets if a is not None} | set(used_players)
            for ballast in _salary_ballast(buyer, taken, ctx=ctx, seller_id=team_id_of(seller)):
                retain = _retention_to_fit_cap(buyer, s_offer, outgoing_nhl + [ballast], league=league, ctx=ctx)
                if retain is not None:
                    buyer_assets.append(ballast)
                    if filler is None:
                        filler = ballast  # counted/labelled with the deal
                    break
        if retain is None:
            _funnel(motive, "cap_no_fit")
            return False
        if retain > 0 and _player_id(s_offer) not in {_player_id(p) for p in list(getattr(seller, "roster", None) or [])}:
            retain = 0.0  # only NHL contracts carry retention
        # Seller already at the roster max → the buyer sends its extra body to the AHL
        # instead of shipping it to the seller.
        from app.sim_engine.trades.trade_rules import ROSTER_MAX as _RMAX

        send_down = None
        seller_nhl_ids = {_player_id(p) for p in list(getattr(seller, "roster", None) or [])}
        to_seller = sum(1 for a in buyer_assets if a is not None and _player_id(a) in buyer_nhl_ids)
        from_seller = 1 if _player_id(s_offer) in seller_nhl_ids else 0
        if filler is not None and _active_count(seller, league) + to_seller - from_seller > _RMAX:
            buyer_assets = [a for a in buyer_assets if a is not filler]
            send_down, filler = filler, None
        # Player coming back in a swap may not fit the SELLER's cap — the buyer retains.
        return_retained: Dict[str, float] = {}
        if b_return is not None and _player_id(b_return) in buyer_nhl_ids:
            r2 = _retention_to_fit_cap(
                seller, b_return, [s_offer] if from_seller else [], league=league, ctx=ctx,
            )
            if r2 is None:
                _funnel(motive, "seller_cap_no_fit")
                return False
            if r2 > 0:
                return_retained[_player_id(b_return)] = float(r2)
        package = _build_package(
            seller,
            buyer,
            s_offer,
            buyer_assets,
            return_retained=return_retained,
            retained_pct=float(retain),
            seller_pick=seller_pick,
            buyer_pick=buyer_pick,
            buyer_pick_2=buyer_pick_2,
        )
        if not package:
            _funnel(motive, "empty_package")
            return False

        def _eval_exec() -> Optional[Tuple[Dict[str, Any], float]]:
            try:
                from app.sim_engine.trades.trade_evaluator import evaluate_trade_package

                evaluation = evaluate_trade_package(
                    package,
                    league=league,
                    team_by_id=team_by_id,
                    context=ctx,
                    user_team_id=None,
                )
            except Exception:
                _funnel(motive, "evaluator_error")
                return None

            if not evaluation.get("can_execute"):
                _funnel(motive, "rules_blocked")
                tagged = []
                for r in list(evaluation.get("rejection_reasons") or [])[:2]:
                    side = "buyer" if str(r).startswith(team_id_of(buyer)) else ("seller" if str(r).startswith(team_id_of(seller)) else "?")
                    tagged.append(f"[{side}] {r}")
                _funnel_reason(motive, tagged)
                return None
            gap = _safe_float(evaluation.get("fairness_gap"), 99.0)
            if gap > attempt_gap_max:
                _funnel(motive, "fairness_gap")
                vb = evaluation.get("value_breakdown") or {}
                nets = {t: round(float((vb.get(t) or {}).get("net", 0.0))) for t in package.keys()}
                side = "buyer_overpays" if nets.get(team_id_of(buyer), 0) < 0 else "seller_overpays"
                gkey = f"gap_{int(gap // 10) * 10}s_{side}"
                row = funnel_reasons.setdefault(str(motive), {})
                row[gkey] = int(row.get(gkey, 0) or 0) + 1
                return None
            if not evaluation.get("accepted"):
                _funnel(motive, "not_accepted")
                _funnel_reason(motive, evaluation.get("rejection_reasons"))
                return None
            interest = evaluation.get("interest_level") or {}
            if any(_safe_float(interest.get(t), 0.0) < min_interest for t in package.keys()):
                _funnel(motive, "low_interest")
                return None

            try:
                from app.sim_engine.trades.trade_executor import execute_validated_trade

                result = execute_validated_trade(
                    evaluation,
                    league=league,
                    team_by_id=team_by_id,
                    context=ctx,
                    user_team_id=None,
                )
            except Exception:
                _funnel(motive, "execute_error")
                return None
            return result, gap

        if send_down is not None:
            _paper_send_down(buyer, send_down)
        outcome = _eval_exec()
        if outcome is None:
            if send_down is not None:
                _paper_recall(buyer, send_down)
            return False
        result, gap = outcome

        _funnel(motive, "executed")
        category, reason_codes, reason_text, importance = _classify_trade_reasons(
            seller=seller,
            buyer=buyer,
            sold_player=s_offer,
            return_player=b_return,
            deadline_phase=deadline,
            seller_pick=seller_pick,
            buyer_pick=buyer_pick,
        )
        if motive == "desperation":
            category = "desperation_trade"
            reason_codes = ["DESPERATION_OVERPAY" if buyer_window == "contender" else "DESPERATION_FIRE_SALE"] + list(reason_codes)
            reason_codes = reason_codes[:4]
            importance = "major" if sold_ovr >= 84 else importance
        elif motive == "futures_package":
            category = "futures_trade"
        elif motive == "rental_sale":
            category = "deadline_rental" if deadline >= 0.4 else category
        elif motive == "demand_resolution":
            disruptor = bool(getattr(s_offer, "locker_room_disruptor", False))
            lead = "LOCKER_ROOM_DISRUPTOR_MOVED" if disruptor else "TRADE_DEMAND_RESOLVED"
            category = "trade_demand"
            reason_codes = ([lead] + [c for c in reason_codes if c != lead])[:4]
            pname = str(getattr(s_offer, "name", None) or "The player")
            days = _safe_int((extra or {}).get("demand_days_open"), 0)
            s_abbr = _team_abbr_upper(seller)
            if disruptor:
                reason_text = f"{s_abbr} ships out {pname} after he fractured the room."
            elif days >= 1:
                reason_text = f"{pname} gets his wish — moved {days} days after demanding a trade."
            else:
                reason_text = f"{pname} gets his wish — moved right after demanding a trade."
            importance = "major" if (sold_ovr >= 82 or disruptor) else "standard"
        if extra:
            if extra.get("trade_category"):
                category = str(extra["trade_category"])
            if extra.get("reason_codes"):
                reason_codes = (list(extra["reason_codes"]) + [c for c in reason_codes if c not in extra["reason_codes"]])[:5]
            if extra.get("reason_text"):
                reason_text = str(extra["reason_text"])
            if motive in ("panic_buy", "tank_selloff") or float(extra.get("premium") or 0.0) >= 0.2:
                importance = "major"
        used_players.add(_player_id(s_offer))
        if b_return is not None:
            used_players.add(_player_id(b_return))
        if filler is not None:
            used_players.add(_player_id(filler))
        try:
            hist = list(getattr(league, "trade_history", None) or [])
            for row in reversed(hist):
                if isinstance(row, dict) and str(row.get("trade_id") or "") == str(result.get("trade_id") or ""):
                    row.setdefault("trade_category", category)
                    row.setdefault("importance", importance)
                    row.setdefault("reason_codes", list(reason_codes))
                    row.setdefault("reason_text", reason_text)
                    row.setdefault("package_motive", motive)
                    for k, v in (extra or {}).items():
                        row.setdefault(k, v)
                    break
            setattr(league, "trade_history", hist)
        except Exception:
            pass
        recent_pairs[pair_mem_key] = int(calendar_cursor)
        season_pair_counts[pair_mem_key] = season_count + 1
        team_trade_counts[sid] = int(team_trade_counts.get(sid, 0)) + 1
        team_trade_counts[bid] = int(team_trade_counts.get(bid, 0)) + 1

        # Telemetry for season tuning.
        telemetry["trades"] = int(telemetry.get("trades", 0) or 0) + 1
        if pick_only:
            telemetry["pick_only"] = int(telemetry.get("pick_only", 0) or 0) + 1
        if buyer_pick is not None or seller_pick is not None:
            telemetry["with_pick"] = int(telemetry.get("with_pick", 0) or 0) + 1
        if seller_window in ("rebuild", "declining"):
            telemetry["rebuild_sales"] = int(telemetry.get("rebuild_sales", 0) or 0) + 1
            if buyer_pick is not None:
                telemetry["rebuild_sales_with_futures"] = int(telemetry.get("rebuild_sales_with_futures", 0) or 0) + 1
        if motive == "desperation":
            telemetry["desperation"] = int(telemetry.get("desperation", 0) or 0) + 1
        by_m = telemetry.get("by_motive")
        if not isinstance(by_m, dict):
            by_m = {}
            telemetry["by_motive"] = by_m
        by_m[motive] = int(by_m.get(motive, 0) or 0) + 1
        if b_return is not None:
            telemetry["ovr_gap_sum"] = float(telemetry.get("ovr_gap_sum", 0) or 0) + abs(sold_ovr - _player_ovr(b_return))
            telemetry["ovr_gap_n"] = int(telemetry.get("ovr_gap_n", 0) or 0) + 1

        outgoing_labels = [str(getattr(s_offer, "name", None) or "Player")]
        incoming_labels: List[str] = []
        if b_return is not None:
            incoming_labels.append(str(getattr(b_return, "name", None) or "Asset"))
        if filler is not None:
            incoming_labels.append(str(getattr(filler, "name", None) or "Player"))
        if seller_pick:
            yr = seller_pick.get("year")
            rnd = seller_pick.get("round")
            outgoing_labels.append(f"{yr} Round {rnd}" if yr and rnd else f"Pick {seller_pick.get('pick_id') or '?'}")
        if buyer_pick:
            yr = buyer_pick.get("year")
            rnd = buyer_pick.get("round")
            incoming_labels.append(f"{yr} Round {rnd}" if yr and rnd else f"Pick {buyer_pick.get('pick_id') or '?'}")
        if buyer_pick_2:
            yr2 = buyer_pick_2.get("year")
            rnd2 = buyer_pick_2.get("round")
            incoming_labels.append(f"{yr2} Round {rnd2}" if yr2 and rnd2 else f"Pick {buyer_pick_2.get('pick_id') or '?'}")
        if not incoming_labels:
            incoming_labels = ["draft capital"]
        to_bits = ", ".join(outgoing_labels)
        from_bits = ", ".join(incoming_labels)
        buyer_abbr = (
            str(getattr(buyer, "abbr", None) or getattr(buyer, "abbreviation", None) or "").strip().upper()
            or bid
        )
        seller_abbr = (
            str(getattr(seller, "abbr", None) or getattr(seller, "abbreviation", None) or "").strip().upper()
            or sid
        )
        headline = f"{buyer_abbr} acquires {to_bits} from {seller_abbr} for {from_bits}"
        row_out = {
            "from_team_id": sid,
            "to_team_id": bid,
            "outgoing": outgoing_labels,
            "incoming": incoming_labels,
            "headline": headline,
            "trade_id": result.get("trade_id"),
            "execution": result,
            "trade_category": category,
            "importance": importance,
            "reason_codes": reason_codes,
            "reason_text": reason_text,
            "fairness_gap": gap,
            "package_motive": motive,
        }
        for k, v in (extra or {}).items():
            row_out.setdefault(k, v)
        executed.append(row_out)
        return True

    def _attempt_demand_trade(seller: Any, player: Any, days_open: int) -> bool:
        """Move a player who has formally demanded out. Seller is motivated and eats a discount."""
        from app.sim_engine.trades.trade_asset import player_holds_nhl_spc
        from app.sim_engine.trades.trade_rules import _clause_summary, _player_recently_acquired

        sid = team_id_of(seller)
        pid = _player_id(player)
        clause = _clause_summary(player)
        if clause.get("nmc") or _player_recently_acquired(player, ctx):
            return False
        dests = {str(x).upper() for x in (getattr(player, "_trade_demand_destinations", None) or []) if x}
        clause_limited = bool(clause.get("ntc") or clause.get("mntc", 0) > 0)
        pool = [t for t in teams if team_id_of(t) != sid]
        if clause_limited:
            # He waives only for clubs on his own list.
            pool = [t for t in pool if _team_abbr_upper(t) in dests]
        if not pool:
            return False
        stage = max(1, min(3, _safe_int(getattr(player, "_crisis_trade_stage", 1), 1)))
        disruptor = bool(getattr(player, "locker_room_disruptor", False))

        weighted = [(t, _demand_buyer_weight(t, player, dests)) for t in pool]
        tries: List[Any] = []
        while weighted and len(tries) < CPU_DEMAND_BUYER_TRIES:
            idx = pair_rng.choices(range(len(weighted)), weights=[w for _, w in weighted], k=1)[0]
            tries.append(weighted.pop(idx)[0])

        seller_rebuilding = _team_window(seller) in ("rebuild", "declining") or _direction_of(seller) in (
            "SELLER",
            "REBUILDING",
            "DEEP_REBUILD",
            "CAP_CORRECTION",
        )
        gap_max = CPU_DEMAND_FAIRNESS_GAP_BASE + 3.0 * stage
        ctx.pop("cpu_desperation_trade", None)
        ctx.pop("cpu_futures_trade", None)
        ctx["cpu_package_motive"] = "demand_resolution"
        ctx["cpu_demand_trade"] = {"seller_team_id": sid, "player_id": pid, "stage": stage}
        seller.needs = needs_model.evaluate(seller, context=ctx)
        try:
            for buyer in tries:
                bid = team_id_of(buyer)
                pair_key = tuple(sorted((sid, bid)))
                if int(season_pair_counts.get(f"{pair_key[0]}|{pair_key[1]}", 0) or 0) >= CPU_SEASON_PAIR_SOFT_CAP:
                    continue
                if _is_reverse_to_prior(player, bid, ctx):
                    continue
                ctx["ntc_waivers"] = (
                    {pid: {"accepted": True, "destination_team_id": bid}} if clause_limited else {}
                )
                buyer.needs = needs_model.evaluate(buyer, context=ctx)
                offer_val = _player_trade_value(player, seller, league, ctx, acquiring_team=buyer)

                b_return = None
                if not seller_rebuilding or pair_rng.random() < 0.35:
                    b_roster = list(getattr(buyer, "roster", None) or [])
                    for attr in ("ahl_roster", "echl_roster"):
                        b_roster.extend(p for p in list(getattr(buyer, attr, None) or []) if player_holds_nhl_spc(p))
                    ctx["_acquiring_team"] = seller
                    b_cands = _pick_trade_candidates(
                        b_roster, buyer, seller=False, league=league, ctx=ctx,
                        acquiring_team_id=sid, motive="demand_resolution",
                    )
                    ctx.pop("_acquiring_team", None)
                    b_return = _match_return_player(
                        seller_asset=player, seller=seller, buyer=buyer, buyer_candidates=b_cands,
                        league=league, ctx=ctx, used_players=used_players,
                        value_band=gap_max, motive="demand_resolution",
                    )
                return_val = (
                    _player_trade_value(b_return, buyer, league, ctx, acquiring_team=seller) if b_return is not None else 0.0
                )
                shortfall = offer_val - return_val
                buyer_pick = None
                buyer_pick_2 = None
                if b_return is None or shortfall > 6.0:
                    target = shortfall * (0.85 if b_return is None else 0.9)
                    buyer_pick, v1 = _pick_for_value(league, buyer, ctx=ctx, target=target)
                    if buyer_pick is not None and b_return is None and target - v1 > 12.0:
                        buyer_pick_2, _ = _pick_for_value(
                            league, buyer, ctx=ctx, target=target - v1,
                            exclude_pick_ids={str(buyer_pick.get("pick_id") or "")},
                        )
                _funnel("demand_resolution", "attempt")
                if b_return is None and buyer_pick is None:
                    _funnel("demand_resolution", "no_return_or_pick")
                    continue
                if _finalize_trade(
                    seller=seller,
                    buyer=buyer,
                    s_offer=player,
                    b_return=b_return,
                    pick_only=b_return is None,
                    seller_pick=None,
                    buyer_pick=buyer_pick,
                    buyer_pick_2=buyer_pick_2,
                    motive="demand_resolution",
                    attempt_gap_max=gap_max,
                    min_interest=CPU_DEMAND_MIN_INTEREST,
                    extra={
                        "demand_player_id": pid,
                        "demand_days_open": int(days_open),
                        "demand_stage": stage,
                        "demand_disruptor": disruptor,
                    },
                ):
                    used_pairs.add((sid, bid))
                    return True
        finally:
            ctx.pop("cpu_demand_trade", None)
            ctx.pop("ntc_waivers", None)
            ctx.pop("cpu_package_motive", None)
        return False

    filled_needs: set = set()

    def _run_needs_market() -> None:
        from collections import Counter

        from app.sim_engine.trades.needs_matcher import MatcherTools, build_package_for_plan, generate_plans
        from app.sim_engine.trades.team_assessment import assess_league

        assessments = assess_league(
            league,
            calendar_cursor=int(calendar_cursor),
            deadline_phase=deadline,
            days_to_deadline=int(ctx.get("days_to_deadline", 99) if ctx.get("days_to_deadline") is not None else 99),
        )
        telemetry["last_status_counts"] = dict(Counter(a.status for a in assessments.values()))
        telemetry["last_flags"] = {
            "panic_buyers": sum(1 for a in assessments.values() if a.panic_buyer),
            "chasers": sum(1 for a in assessments.values() if a.chaser),
            "late_sellers": sum(1 for a in assessments.values() if a.late_seller),
        }
        # Values are stable within one market pass — cache them (was ~25 s/pass uncached).
        pv_cache: Dict[Tuple[str, str, str], float] = {}
        pick_cache: Dict[Tuple[str, str], float] = {}

        def _pv(p: Any, frm: Any, to: Any) -> float:
            k = (_player_id(p), team_id_of(frm), team_id_of(to))
            if k not in pv_cache:
                pv_cache[k] = _player_trade_value(p, frm, league, ctx, acquiring_team=to)
            return pv_cache[k]

        def _pfv(lg: Any, team: Any, **kw: Any) -> Tuple[Optional[Dict[str, Any]], float]:
            return _pick_for_value(lg, team, value_cache=pick_cache, **kw)

        def _filler_value(seller: Any, buyer: Any, target: Any, ret: Any) -> float:
            # The roster player going back is part of what the seller receives.
            f = _roster_filler(seller, buyer, target, ret, ctx=ctx, exclude=used_players)
            return _pv(f, buyer, seller) if f is not None else 0.0

        tools = MatcherTools(
            player_value=_pv,
            pick_for_value=_pfv,
            tradeable=lambda p, acq: _tradeable_player(p, acq, ctx=ctx),
            team_abbr=_team_abbr_upper,
            filler_value=_filler_value,
        )
        plans = generate_plans(
            league, teams=teams, assessments=assessments, ctx=ctx, tools=tools, rng=pair_rng, used_players=used_players,
        )
        telemetry["last_plan_count"] = len(plans)
        attempt_budget = 8 + 4 * max(1, int(max_executions))
        buyer_deals: Dict[str, int] = {}
        days_left_ctx = int(ctx.get("days_to_deadline", 99) if ctx.get("days_to_deadline") is not None else 99)
        per_buyer_cap = 4 if days_left_ctx == 0 else (2 if 0 <= days_left_ctx <= 7 else 1)
        # Sellers wait ~20 games to judge the season, then mostly hold vets for the deadline.
        from statistics import median as _median

        from app.sim_engine.trades.needs_matcher import SELLOFF_MOTIVES

        league_gp = _median([a.games_played for a in assessments.values()] or [0])
        if league_gp < 20:
            selloff_cap = 0
        elif deadline < 0.3:
            selloff_cap = 1
        elif days_left_ctx == 0:
            selloff_cap = 99  # deadline day: everything left is for sale
        elif days_left_ctx <= 2:
            selloff_cap = 2
        else:
            selloff_cap = 1  # sellers hold inventory for deadline day
        selloffs_done = 0
        panic_counts = partner_memory.setdefault("season_panic_buys", {})
        # GMs who just dealt with each other talk again quickly in the deadline crunch.
        pair_cooldown = 0 if days_left_ctx == 0 else (3 if 0 <= days_left_ctx <= 7 else CPU_PAIR_COOLDOWN_DAYS)
        for plan in plans:
            if len(executed) >= max(0, int(max_executions)) or attempt_budget <= 0:
                break
            sid, bid = team_id_of(plan.seller), team_id_of(plan.buyer)
            if _player_id(plan.target) in used_players or (bid, plan.need_slot) in filled_needs:
                continue
            if (sid, bid) in used_pairs or (bid, sid) in used_pairs:
                continue
            if buyer_deals.get(bid, 0) >= per_buyer_cap:
                _funnel(plan.motive, "buyer_already_dealt")
                continue
            if plan.motive in SELLOFF_MOTIVES and selloffs_done >= selloff_cap:
                _funnel(plan.motive, "seller_waiting_for_deadline")
                continue
            if plan.motive == "panic_buy" and int(panic_counts.get(bid, 0) or 0) >= CPU_PANIC_BUYS_PER_SEASON:
                _funnel(plan.motive, "panic_budget_spent")
                continue
            pk = tuple(sorted((sid, bid)))
            pair_mem_key = f"{pk[0]}|{pk[1]}"
            if int(season_pair_counts.get(pair_mem_key, 0) or 0) >= CPU_SEASON_PAIR_SOFT_CAP + (1 if days_left_ctx == 0 else 0):
                _funnel(plan.motive, "pair_season_cap")
                continue
            if (
                int(calendar_cursor) - int(recent_pairs.get(pair_mem_key, -999) or -999) < pair_cooldown
                and plan.motive != "panic_buy"
            ):
                _funnel(plan.motive, "pair_cooldown")
                continue
            if not build_package_for_plan(
                plan, league, assessments=assessments, ctx=ctx, tools=tools, used_players=used_players,
            ):
                _funnel(plan.motive, plan.fail_reason or "no_package")
                continue
            # Only packages that reach the other GM's desk count against the pass budget.
            _funnel(plan.motive, "attempt")
            attempt_budget -= 1
            if plan.return_player is not None and _player_id(plan.return_player) in used_players:
                continue
            premium_value = round(plan.premium * max(0.0, plan.target_value), 2)
            ctx["cpu_package_motive"] = plan.motive
            ctx["cpu_intent"] = {
                "buyer_team_id": bid,
                "seller_team_id": sid,
                "premium_value": premium_value,
                "motivated_seller": bool(plan.motivated_seller),
                "spend_first_ok": plan.motive == "panic_buy" or plan.premium >= 0.2,
            }
            picks = list(plan.buyer_picks)
            try:
                ok = _finalize_trade(
                    seller=plan.seller,
                    buyer=plan.buyer,
                    s_offer=plan.target,
                    b_return=plan.return_player,
                    pick_only=plan.return_player is None,
                    seller_pick=plan.seller_pick,
                    buyer_pick=picks[0] if picks else None,
                    buyer_pick_2=picks[1] if len(picks) > 1 else None,
                    motive=plan.motive,
                    attempt_gap_max=CPU_AMBIENT_FAIRNESS_GAP_MAX + 2.0 * premium_value + (8.0 if plan.motivated_seller else 0.0),
                    min_interest=0.30 if (plan.motivated_seller or plan.premium > 0.0) else CPU_AMBIENT_MIN_INTEREST,
                    extra={
                        "reason_codes": plan.reason_codes,
                        "reason_text": plan.reason_text,
                        "trade_category": plan.trade_category,
                        "need_slot": plan.need_slot,
                        "premium": plan.premium,
                        "plan_score": plan.score,
                    },
                )
            finally:
                ctx.pop("cpu_intent", None)
                ctx.pop("cpu_package_motive", None)
            if ok:
                buyer_deals[bid] = buyer_deals.get(bid, 0) + 1
                if plan.motive in SELLOFF_MOTIVES:
                    selloffs_done += 1
                if plan.motive == "panic_buy":
                    panic_counts[bid] = int(panic_counts.get(bid, 0) or 0) + 1
                used_pairs.add((sid, bid))
                if plan.need_slot:
                    filled_needs.add((bid, plan.need_slot))

    # Formal trade demands first — these are the storylines, not ambient filler.
    for d_seller, d_player, d_days in collect_cpu_trade_demands(teams, calendar_cursor=int(calendar_cursor)):
        if len(executed) >= max(0, int(max_executions)):
            break
        if _player_id(d_player) in used_players:
            continue
        disruptor = bool(getattr(d_player, "locker_room_disruptor", False))
        if pair_rng.random() >= demand_trade_chance(d_days, deadline, disruptor=disruptor):
            continue
        telemetry["demand_attempts"] = int(telemetry.get("demand_attempts", 0) or 0) + 1
        if _attempt_demand_trade(d_seller, d_player, d_days):
            telemetry["demand_trades"] = int(telemetry.get("demand_trades", 0) or 0) + 1

    # Needs-and-surplus market: only deals a GM has a reason to make.
    if len(executed) < max(0, int(max_executions)):
        _run_needs_market()

    ctx.pop("cpu_desperation_trade", None)
    ctx.pop("cpu_package_motive", None)
    return executed


def propose_ahl_depth_trades(
    league: Any,
    *,
    calendar_cursor: int = 0,
    regular_season_last_index: int = 192,
) -> List[Dict[str, Any]]:
    """Post-deadline minor-league swaps: AHL player for AHL player, each club fixing a shortage."""
    from app.sim_engine.trades.needs_matcher import plan_ahl_depth_swap
    from app.sim_engine.trades.trade_evaluator import evaluate_trade_package
    from app.sim_engine.trades.trade_executor import execute_validated_trade

    teams = list(getattr(league, "teams", None) or [])
    user_tid = str(getattr(league, "_franchise_user_team_id", None) or getattr(league, "user_team_id", None) or "")
    teams = [t for t in teams if team_id_of(t) != user_tid]
    if len(teams) < 2:
        return []
    ctx = build_league_trade_context(
        league, calendar_cursor=calendar_cursor, regular_season_last_index=regular_season_last_index,
    )
    ctx["cpu_ambient_trade"] = True
    ctx["cpu_package_motive"] = "ahl_depth_swap"
    import random as _random

    base_rng = getattr(league, "rng", None)
    rng = _random.Random(int(base_rng.randint(1, 2_000_000_000)) if hasattr(base_rng, "randint") else calendar_cursor)
    plan = plan_ahl_depth_swap(teams, rng=rng, used_players=set())
    if plan is None:
        return []
    team_a, pa, team_b, pb = plan
    aid, bid = team_id_of(team_a), team_id_of(team_b)
    package = {
        bid: [{"type": "player", "id": _player_id(pa), "team": aid}],
        aid: [{"type": "player", "id": _player_id(pb), "team": bid}],
    }
    try:
        evaluation = evaluate_trade_package(package, league=league, team_by_id=ctx["team_by_id"], context=ctx, user_team_id=None)
        if not evaluation.get("can_execute") or not evaluation.get("accepted"):
            return []
        result = execute_validated_trade(evaluation, league=league, team_by_id=ctx["team_by_id"], context=ctx, user_team_id=None)
    except Exception:
        return []
    a_abbr, b_abbr = _team_abbr_upper(team_a), _team_abbr_upper(team_b)
    name_a = str(getattr(pa, "name", None) or "Player")
    name_b = str(getattr(pb, "name", None) or "Player")
    reason_text = f"Minor-league swap: {a_abbr} and {b_abbr} balance their AHL depth after the deadline."
    try:
        for row in reversed(list(getattr(league, "trade_history", None) or [])):
            if isinstance(row, dict) and str(row.get("trade_id") or "") == str(result.get("trade_id") or ""):
                row.setdefault("trade_category", "minor_league_trade")
                row.setdefault("importance", "minor")
                row.setdefault("reason_codes", ["AHL_DEPTH_SWAP"])
                row.setdefault("reason_text", reason_text)
                row.setdefault("package_motive", "ahl_depth_swap")
                break
    except Exception:
        pass
    return [
        {
            "from_team_id": aid,
            "to_team_id": bid,
            "outgoing": [name_a],
            "incoming": [name_b],
            "headline": f"{b_abbr} acquires {name_a} (AHL) from {a_abbr} for {name_b} (AHL)",
            "trade_id": result.get("trade_id"),
            "execution": result,
            "trade_category": "minor_league_trade",
            "importance": "minor",
            "reason_codes": ["AHL_DEPTH_SWAP"],
            "reason_text": reason_text,
            "package_motive": "ahl_depth_swap",
        }
    ]
