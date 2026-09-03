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
CPU_REVERSE_TRADE_PENALTY = 0.22  # retained as soft demote only; hard ban is season reverse block
CPU_SEASON_PAIR_SOFT_CAP = 2
CPU_PEER_ATTEMPT_MODULO = 6  # peer depth swaps — ambient volume when futures paths fail
CPU_ONE_FOR_ONE_OVR_GAP_MAX = 7.0  # allow more talent asymmetry without futures
CPU_SELLER_CORE_OVR = 86.0
CPU_YOUNG_CORE_MAX_AGE = 23
CPU_PROSPECT_MAX_AGE = 22
CPU_DESPERATION_GAP_MAX = 28.0
CPU_DESPERATION_CHANCE = 0.26  # controlled unfairness when pressure is high
CPU_DUMB_GM_CHANCE = 0.13  # occasional overpay / underpay regardless of window
CPU_DIVERSITY_TARGETS = {
    "max_pair_repetitions_per_season": 2,
    "min_median_unique_partners_per_trading_team": 3.0,
    "max_pct_trades_reuse_prior_pair": 0.35,
    "max_reverse_trade_rate": 0.0,  # hard season reverse block target
    "max_fairness_gap_mean": CPU_AMBIENT_FAIRNESS_GAP_MAX,
    "min_pct_trades_with_pick": 0.34,
    "min_pct_rebuild_sales_with_futures": 0.55,
}

# Motive → construction rules for ambient packages (not one closest-TV factory).
PACKAGE_MOTIVES = (
    "depth_swap",
    "rental_sale",
    "futures_package",
    "star_acquisition",
    "desperation",
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


def _desperation_score(team: Any, *, deadline: float, direction: str = "") -> float:
    """0–1 score for rare asymmetric ambient deals."""
    window = _team_window(team)
    odds = _playoff_odds(team)
    score = 0.0
    cap_p = _safe_float(getattr(team, "cap_pressure", 0.0), 0.0)
    if isinstance(getattr(team, "cap_pressure", None), str):
        tier = str(getattr(team, "cap_pressure")).lower()
        cap_p = 0.95 if tier in ("cap_hell", "critical") else 0.7 if tier == "tight" else cap_p
    if window in ("rebuild", "declining") and odds < 0.28:
        score += 0.35
    if deadline >= 0.55 and window == "contender" and odds < 0.42:
        score += 0.40  # collapsing contender overpay
    if deadline >= 0.55 and window in ("rebuild", "declining") and odds < 0.22:
        score += 0.30  # fire-sale seller
    if cap_p >= 0.85:
        score += 0.25
    if direction in ("DEEP_REBUILD", "CAP_CORRECTION", "ALL_IN_CONTENDER"):
        score += 0.15
    return max(0.0, min(1.0, score))


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
    max_d = max(40, int(regular_season_last_index or 192))
    md = max(40, int(max(120, max_d) * 0.56))
    deadline_phase = max(0.0, min(1.0, (float(calendar_cursor) - float(md)) / max(20.0, float(max_d) * 0.2)))
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


def _talent_gap_ok(
    sold: Any,
    ret: Any,
    *,
    buyer_pick: Optional[Dict[str, Any]] = None,
    seller_pick: Optional[Dict[str, Any]] = None,
    motive: str = "depth_swap",
) -> bool:
    """One-for-one talent-gap veto unless futures compensate or motive allows asymmetry."""
    if ret is None:
        # Straight pick-for-player packages (rebuild sells NHL talent for draft capital).
        return bool(buyer_pick) and motive in (
            "futures_package",
            "rental_sale",
            "desperation",
            "star_acquisition",
        )
    if motive in ("desperation", "star_acquisition"):
        return True
    gap = abs(_player_ovr(sold) - _player_ovr(ret))
    if gap <= CPU_ONE_FOR_ONE_OVR_GAP_MAX:
        return True
    # Futures / second asset compensates a larger talent gap.
    if buyer_pick is not None or seller_pick is not None:
        return gap <= CPU_ONE_FOR_ONE_OVR_GAP_MAX + 6.0
    return False


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


def _select_tradeable_pick(
    league: Any,
    team: Any,
    *,
    ctx: Dict[str, Any],
    max_round: int = 3,
    protect_own_first: bool = False,
    prefer_quality: str = "cheapest",
    pair_rng: Any = None,
    exclude_pick_ids: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Pick selection — futures packages prefer mid/high quality, not always cheapest."""
    tid = team_id_of(team)
    picks = get_team_owned_picks(league, tid)
    excluded = {str(x) for x in (exclude_pick_ids or set()) if x}
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for row in picks:
        if bool(row.get("resolved")):
            continue
        pick_id = str(row.get("pick_id") or "")
        if pick_id and pick_id in excluded:
            continue
        rnd = _safe_int(row.get("round"), 7)
        if rnd > max_round:
            continue
        orig = str(row.get("original_team_id") or "")
        if protect_own_first and rnd == 1 and orig == tid:
            proj = evaluate_pick_asset_value(row, team, team, league, context=ctx)
            if _safe_float(proj.get("total"), 0.0) >= 55.0:
                continue
        try:
            val = float(evaluate_pick_asset_value(row, team, team, league, context=ctx).get("total", 0.0))
        except Exception:
            val = 20.0 - rnd * 3.0
        candidates.append((val, row))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    if prefer_quality == "cheapest":
        return candidates[0][1]
    if prefer_quality == "best":
        return candidates[-1][1]
    # mid / sample: bias toward middle-upper value for real futures returns
    if pair_rng is not None and len(candidates) >= 2:
        start = max(0, len(candidates) // 3)
        pool = candidates[start:]
        return pair_rng.choice(pool)[1]
    return candidates[min(len(candidates) - 1, max(0, len(candidates) // 2))][1]


def _choose_package_motive(
    *,
    seller: Any,
    buyer: Any,
    deadline: float,
    peer_path: bool,
    pair_rng: Any,
    direction_seller: str,
    direction_buyer: str,
) -> str:
    """Pick a construction motive so ambient is not only closest-TV matching."""
    sw = _team_window(seller)
    bw = _team_window(buyer)
    rebuild_dirs = frozenset({"SELLER", "REBUILDING", "DEEP_REBUILD", "CAP_CORRECTION"})
    buyer_dirs = frozenset({"CONTENDER", "PLAYOFF_BUYER", "ALL_IN_CONTENDER"})
    s_desp = _desperation_score(seller, deadline=deadline, direction=direction_seller)
    b_desp = _desperation_score(buyer, deadline=deadline, direction=direction_buyer)
    if max(s_desp, b_desp) >= 0.55 and pair_rng.random() < CPU_DESPERATION_CHANCE + 0.18 * max(s_desp, b_desp):
        return "desperation"
    if peer_path:
        return "depth_swap"
    if direction_seller in rebuild_dirs and direction_buyer in buyer_dirs:
        if deadline >= 0.20 and pair_rng.random() < 0.62:
            return "rental_sale"
        return "futures_package"
    if sw in ("rebuild", "declining") and bw in ("contender", "emerging"):
        if deadline >= 0.25 and pair_rng.random() < 0.62:
            return "rental_sale"
        return "futures_package"
    if direction_buyer in buyer_dirs and pair_rng.random() < 0.42:
        return "star_acquisition"
    if bw == "contender" and deadline >= 0.35 and pair_rng.random() < 0.45:
        return "star_acquisition"
    if sw == bw:
        roll = pair_rng.random()
        if roll < 0.58:
            return "depth_swap"
        if roll < 0.72:
            return "futures_package"
        if roll < 0.84:
            return "rental_sale"
        return "star_acquisition"
    if sw == "emerging" and bw == "emerging":
        return "depth_swap" if pair_rng.random() < 0.72 else "futures_package"
    roll = pair_rng.random()
    if roll < 0.30:
        return "depth_swap"
    if roll < 0.50:
        return "futures_package"
    if roll < 0.72:
        return "rental_sale"
    if roll < 0.86:
        return "star_acquisition"
    return "depth_swap"


def _build_package(
    seller: Any,
    buyer: Any,
    seller_asset: Any,
    buyer_assets: List[Any],
    *,
    seller_pick: Optional[Dict[str, Any]] = None,
    buyer_pick: Optional[Dict[str, Any]] = None,
    buyer_pick_2: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    sid = team_id_of(seller)
    bid = team_id_of(buyer)
    spid = _player_id(seller_asset)
    if not spid:
        return {}
    buyer_payload: List[Dict[str, Any]] = [{"type": "player", "id": spid, "team": sid}]
    seller_payload: List[Dict[str, Any]] = []
    for asset in buyer_assets:
        if isinstance(asset, dict):
            seller_payload.append(asset)
        else:
            bpid = _player_id(asset)
            if bpid:
                seller_payload.append({"type": "player", "id": bpid, "team": bid})
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

    def _ideo(tm: Any, key: str, default: float = 0.5) -> float:
        tid = team_id_of(tm)
        ideo = (profiles.get(tid) or {}).get("ideology") or {}
        try:
            return float(ideo.get(key, default))
        except Exception:
            return default

    sellers = [
        t
        for t in teams
        if _team_window(t) in ("rebuild", "declining")
        or _direction_of(t) in ("SELLER", "REBUILDING", "DEEP_REBUILD", "CAP_CORRECTION")
    ]
    buyers = [
        t
        for t in teams
        if _team_window(t) in ("contender", "emerging")
        or _direction_of(t) in ("CONTENDER", "PLAYOFF_BUYER", "ALL_IN_CONTENDER")
    ]
    # Include same-window partners for depth / hockey swaps (volume without unfair packages).
    peers = [t for t in teams if _team_window(t) in ("emerging", "contender", "rebuild")]
    # Ideology: future-asset preference boosts seller activity; aggression boosts buyer activity.
    sellers = sorted(sellers, key=lambda t: -_ideo(t, "future_asset_preference", 0.5)) or sellers
    buyers = sorted(buyers, key=lambda t: -_ideo(t, "aggression", 0.5)) or buyers
    if deadline > 0.5:
        buyers = sorted(
            buyers,
            key=lambda t: (_team_window(t) != "contender", -_playoff_odds(t)),
        )
    if not sellers:
        sellers = teams[:]
    if not buyers:
        buyers = teams[:]

    def _reclassify_pools() -> None:
        nonlocal sellers, buyers, peers
        sellers = [
            t
            for t in teams
            if _team_window(t) in ("rebuild", "declining")
            or _direction_of(t) in ("SELLER", "REBUILDING", "DEEP_REBUILD", "CAP_CORRECTION")
        ]
        buyers = [
            t
            for t in teams
            if _team_window(t) in ("contender", "emerging")
            or _direction_of(t) in ("CONTENDER", "PLAYOFF_BUYER", "ALL_IN_CONTENDER")
        ]
        peers = [t for t in teams if _team_window(t) in ("emerging", "contender", "rebuild")]
        sellers = sorted(sellers, key=lambda t: -_ideo(t, "future_asset_preference", 0.5)) or list(teams)
        buyers = sorted(buyers, key=lambda t: -_ideo(t, "aggression", 0.5)) or list(teams)
        if deadline > 0.5:
            buyers = sorted(
                buyers,
                key=lambda t: (_team_window(t) != "contender", -_playoff_odds(t)),
            )

    executed: List[Dict[str, Any]] = []
    used_pairs: set = set()
    used_players: set = set()
    attempts = max(12, int(max_executions) * (22 if deadline > 0.45 else 16))
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

    import random as _random

    day_seed = int(calendar_cursor) * 1009 + int(max_executions) * 17 + len(teams)
    base_rng = getattr(league, "rng", None)
    try:
        pair_rng = _random.Random(int(base_rng.randint(1, 2_000_000_000)) ^ day_seed) if hasattr(base_rng, "randint") else _random.Random(day_seed)
    except Exception:
        pair_rng = _random.Random(day_seed)
    team_trade_counts: Dict[str, int] = {}

    for i in range(attempts):
        if len(executed) >= max(0, int(max_executions)):
            break
        peer_path = bool(i % CPU_PEER_ATTEMPT_MODULO == (CPU_PEER_ATTEMPT_MODULO - 1) and peers)
        if peer_path:
            pool_a, pool_b = peers, peers
        else:
            pool_a, pool_b = sellers, buyers
        if not pool_a or not pool_b:
            continue

        def _weight(tm: Any, *, buyer_side: bool) -> float:
            tid = team_id_of(tm)
            w = 1.0 + _ideo(tm, "aggression" if buyer_side else "future_asset_preference", 0.5)
            w /= 1.0 + 0.85 * float(team_trade_counts.get(tid, 0))
            return max(0.05, w)

        seller = pair_rng.choices(pool_a, weights=[_weight(t, buyer_side=False) for t in pool_a], k=1)[0]
        buyer_pool = [t for t in pool_b if team_id_of(t) != team_id_of(seller)]
        if not buyer_pool:
            continue
        buyer = pair_rng.choices(buyer_pool, weights=[_weight(t, buyer_side=True) for t in buyer_pool], k=1)[0]
        try:
            s_div = str(getattr(seller, "division", None) or getattr(seller, "div", "") or "")
            b_div = str(getattr(buyer, "division", None) or getattr(buyer, "div", "") or "")
            if s_div and s_div == b_div and pair_rng.random() < 0.35:
                continue
        except Exception:
            pass
        sid = team_id_of(seller)
        bid = team_id_of(buyer)
        if user_tid and (sid == user_tid or bid == user_tid):
            continue
        pair_key = tuple(sorted((sid, bid)))
        pair_mem_key = f"{pair_key[0]}|{pair_key[1]}"
        season_count = int(season_pair_counts.get(pair_mem_key, 0) or 0)
        if season_count >= CPU_SEASON_PAIR_SOFT_CAP:
            continue
        last_pair_day = int(recent_pairs.get(pair_mem_key, -999) or -999)
        if (int(calendar_cursor) - last_pair_day) < CPU_PAIR_COOLDOWN_DAYS:
            continue
        if (sid, bid) in used_pairs:
            continue
        used_pairs.add((sid, bid))
        if _ideo(seller, "prospect_protection", 0.5) >= 0.72 and _ideo(buyer, "aggression", 0.5) < 0.45:
            if deadline < 0.35 and pair_rng.random() < 0.55:
                continue

        direction_seller = _direction_of(seller)
        direction_buyer = _direction_of(buyer)
        motive = _choose_package_motive(
            seller=seller,
            buyer=buyer,
            deadline=deadline,
            peer_path=peer_path,
            pair_rng=pair_rng,
            direction_seller=direction_seller,
            direction_buyer=direction_buyer,
        )
        attempt_gap_max = fairness_gap_max
        if motive == "desperation":
            attempt_gap_max = max(fairness_gap_max, CPU_DESPERATION_GAP_MAX)
            ctx["cpu_desperation_trade"] = True
        else:
            ctx.pop("cpu_desperation_trade", None)
        ctx["cpu_package_motive"] = motive

        from app.sim_engine.trades.trade_asset import player_holds_nhl_spc

        s_roster = list(getattr(seller, "roster", None) or [])
        b_roster = list(getattr(buyer, "roster", None) or [])
        for attr in ("ahl_roster", "echl_roster"):
            for p in list(getattr(seller, attr, None) or []):
                if player_holds_nhl_spc(p):
                    s_roster.append(p)
            for p in list(getattr(buyer, attr, None) or []):
                if player_holds_nhl_spc(p):
                    b_roster.append(p)
        if not s_roster or not b_roster:
            continue

        seller.needs = needs_model.evaluate(seller, context=ctx)
        buyer.needs = needs_model.evaluate(buyer, context=ctx)
        ctx["_acquiring_team"] = buyer
        s_candidates = _pick_trade_candidates(
            s_roster, seller, seller=True, league=league, ctx=ctx, acquiring_team_id=bid, motive=motive,
        )
        ctx["_acquiring_team"] = seller
        b_candidates = _pick_trade_candidates(
            b_roster, buyer, seller=False, league=league, ctx=ctx, acquiring_team_id=sid, motive=motive,
        )
        ctx.pop("_acquiring_team", None)
        if not s_candidates or not b_candidates:
            continue

        top_n = 5 if motive in ("rental_sale", "futures_package") else 8
        s_offer = s_candidates[pair_rng.randrange(0, min(len(s_candidates), top_n))]
        if _player_id(s_offer) in used_players:
            continue
        if _needs_fit_score(seller, s_offer, selling=True) < -2.0 and not _is_rental(s_offer):
            continue
        if _is_reverse_to_prior(s_offer, bid, ctx):
            telemetry["reverse_blocked"] = int(telemetry.get("reverse_blocked", 0) or 0) + 1
            continue

        value_band = max(10.0, attempt_gap_max + 3.0)
        if motive == "desperation":
            value_band = max(value_band, 24.0)
        elif motive == "star_acquisition":
            value_band = max(value_band, 16.0)
        elif motive in ("rental_sale", "futures_package"):
            value_band = max(value_band, 18.0)
        if pair_rng.random() < CPU_DUMB_GM_CHANCE:
            value_band = max(value_band, 30.0)
            telemetry["dumb_gm_rolls"] = int(telemetry.get("dumb_gm_rolls", 0) or 0) + 1

        seller_window = _team_window(seller)
        buyer_window = _team_window(buyer)
        pick_only = False
        prefer_pick_only = (
            motive in ("futures_package", "rental_sale")
            and (
                seller_window in ("rebuild", "declining")
                or direction_seller in ("SELLER", "REBUILDING", "DEEP_REBUILD", "CAP_CORRECTION")
            )
            and (
                buyer_window in ("contender", "emerging")
                or direction_buyer in ("CONTENDER", "PLAYOFF_BUYER", "ALL_IN_CONTENDER")
            )
            and pair_rng.random() < (0.62 if motive == "futures_package" else 0.42)
        )
        b_return = None
        if not prefer_pick_only:
            b_return = _match_return_player(
                seller_asset=s_offer,
                seller=seller,
                buyer=buyer,
                buyer_candidates=b_candidates,
                league=league,
                ctx=ctx,
                used_players=used_players,
                value_band=value_band,
                motive=motive,
            )
        if b_return is None:
            allow_pick_only = (
                motive in ("futures_package", "rental_sale", "desperation", "star_acquisition")
                and seller_window in ("rebuild", "declining")
                and buyer_window in ("contender", "emerging")
            )
            if not allow_pick_only and motive == "depth_swap":
                b_return = _match_return_player(
                    seller_asset=s_offer,
                    seller=seller,
                    buyer=buyer,
                    buyer_candidates=b_candidates,
                    league=league,
                    ctx=ctx,
                    used_players=used_players,
                    value_band=value_band * 1.75,
                    motive=motive,
                )
            if b_return is None and not allow_pick_only:
                continue
            pick_only = b_return is None
        elif _is_reverse_to_prior(b_return, sid, ctx):
            telemetry["reverse_blocked"] = int(telemetry.get("reverse_blocked", 0) or 0) + 1
            continue

        buyer_pick = None
        seller_pick = None
        protect_first = _ideo(buyer, "draft_pick_protection", 0.5) >= 0.62
        offer_val = _player_trade_value(s_offer, seller, league, ctx, acquiring_team=buyer)
        sold_ovr = _player_ovr(s_offer)
        valuable_sale = sold_ovr >= 78.0 or (_is_rental(s_offer) and sold_ovr >= 76.0)

        if pick_only:
            max_r = 2 if sold_ovr >= 82 else (3 if sold_ovr >= 78 else 4)
            quality = "best" if sold_ovr >= 84 else ("mid" if sold_ovr >= 78 else "cheapest")
            buyer_pick = _select_tradeable_pick(
                league,
                buyer,
                ctx=ctx,
                max_round=max_r,
                protect_own_first=False,
                prefer_quality=quality,
                pair_rng=pair_rng,
            )
            if buyer_pick is None:
                continue
            return_val = 0.0
            value_delta = offer_val
        else:
            return_val = _player_trade_value(b_return, buyer, league, ctx, acquiring_team=seller)
            value_delta = offer_val - return_val

            # Motive-driven futures / sweeteners (not "cheapest only when uneven").
            if motive == "futures_package" or (
                seller_window in ("rebuild", "declining") and valuable_sale and motive != "depth_swap"
            ):
                buyer_pick = _select_tradeable_pick(
                    league,
                    buyer,
                    ctx=ctx,
                    max_round=3 if protect_first else 2,
                    protect_own_first=protect_first and motive != "desperation",
                    prefer_quality="mid",
                    pair_rng=pair_rng,
                )
                if buyer_pick is None and motive == "futures_package":
                    buyer_pick = _select_tradeable_pick(
                        league, buyer, ctx=ctx, max_round=4, protect_own_first=False,
                        prefer_quality="mid", pair_rng=pair_rng,
                    )
                # Require futures on rebuild valuable sales — skip if no pick available.
                if buyer_pick is None and seller_window in ("rebuild", "declining") and valuable_sale:
                    continue
            elif motive == "rental_sale":
                if buyer_window == "contender" and deadline > 0.3:
                    buyer_pick = _select_tradeable_pick(
                        league, buyer, ctx=ctx, max_round=4 if deadline > 0.65 else 3,
                        protect_own_first=protect_first,
                        prefer_quality="mid" if deadline > 0.55 else "cheapest",
                        pair_rng=pair_rng,
                    )
                if buyer_pick is None and value_delta > 2.0:
                    buyer_pick = _select_tradeable_pick(
                        league, buyer, ctx=ctx, max_round=4, protect_own_first=False,
                        prefer_quality="cheapest", pair_rng=pair_rng,
                    )
            elif motive == "star_acquisition":
                if value_delta > 1.5 or sold_ovr >= 84:
                    buyer_pick = _select_tradeable_pick(
                        league, buyer, ctx=ctx, max_round=2 if sold_ovr >= 86 else 3,
                        protect_own_first=False,
                        prefer_quality="best" if sold_ovr >= 86 else "mid",
                        pair_rng=pair_rng,
                    )
            elif motive == "desperation":
                # Contender overpay OR seller accepts thin return + futures.
                if buyer_window == "contender":
                    buyer_pick = _select_tradeable_pick(
                        league, buyer, ctx=ctx, max_round=2, protect_own_first=False,
                        prefer_quality="best", pair_rng=pair_rng,
                    )
                elif seller_window in ("rebuild", "declining"):
                    buyer_pick = _select_tradeable_pick(
                        league, buyer, ctx=ctx, max_round=3, protect_own_first=False,
                        prefer_quality="cheapest", pair_rng=pair_rng,
                    )
            else:
                # Depth swap: only tiny balancers when clearly uneven.
                if value_delta > 5.0:
                    buyer_pick = _select_tradeable_pick(
                        league, buyer, ctx=ctx, max_round=4, protect_own_first=True,
                        prefer_quality="cheapest", pair_rng=pair_rng,
                    )
                elif value_delta < -5.0:
                    seller_pick = _select_tradeable_pick(
                        league, seller, ctx=ctx, max_round=4, protect_own_first=True,
                        prefer_quality="cheapest", pair_rng=pair_rng,
                    )

        buyer_pick_2 = None
        if buyer_pick is not None and sold_ovr >= 78 and motive in (
            "futures_package",
            "rental_sale",
            "desperation",
            "star_acquisition",
        ):
            mult_chance = 0.58 if motive == "futures_package" else (0.44 if motive == "desperation" else 0.32)
            if pair_rng.random() < mult_chance:
                buyer_pick_2 = _select_tradeable_pick(
                    league,
                    buyer,
                    ctx=ctx,
                    max_round=4 if sold_ovr >= 84 else 3,
                    protect_own_first=False,
                    prefer_quality="mid" if sold_ovr >= 82 else "cheapest",
                    pair_rng=pair_rng,
                    exclude_pick_ids={str(buyer_pick.get("pick_id") or "")},
                )

        if not _talent_gap_ok(
            s_offer, b_return, buyer_pick=buyer_pick, seller_pick=seller_pick, motive=motive,
        ):
            # Try one compensatory pick before veto.
            if buyer_pick is None and motive != "depth_swap":
                buyer_pick = _select_tradeable_pick(
                    league, buyer, ctx=ctx, max_round=3, protect_own_first=False,
                    prefer_quality="mid", pair_rng=pair_rng,
                )
            if not _talent_gap_ok(
                s_offer, b_return, buyer_pick=buyer_pick, seller_pick=seller_pick, motive=motive,
            ):
                continue

        package = _build_package(
            seller,
            buyer,
            s_offer,
            [] if pick_only else [b_return],
            seller_pick=seller_pick,
            buyer_pick=buyer_pick,
            buyer_pick_2=buyer_pick_2,
        )
        if not package:
            continue

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
            continue

        if not evaluation.get("can_execute"):
            continue
        gap = _safe_float(evaluation.get("fairness_gap"), 99.0)
        if gap > attempt_gap_max:
            continue
        if not evaluation.get("accepted"):
            continue
        interest = evaluation.get("interest_level") or {}
        min_interest = 0.28 if motive == "desperation" else CPU_AMBIENT_MIN_INTEREST
        if any(_safe_float(interest.get(t), 0.0) < min_interest for t in package.keys()):
            continue

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
            continue

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
        used_players.add(_player_id(s_offer))
        if b_return is not None:
            used_players.add(_player_id(b_return))
        try:
            hist = list(getattr(league, "trade_history", None) or [])
            for row in reversed(hist):
                if isinstance(row, dict) and str(row.get("trade_id") or "") == str(result.get("trade_id") or ""):
                    row.setdefault("trade_category", category)
                    row.setdefault("importance", importance)
                    row.setdefault("reason_codes", list(reason_codes))
                    row.setdefault("reason_text", reason_text)
                    row.setdefault("package_motive", motive)
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

        _reclassify_pools()
        outgoing_labels = [str(getattr(s_offer, "name", None) or "Player")]
        incoming_labels: List[str] = []
        if b_return is not None:
            incoming_labels.append(str(getattr(b_return, "name", None) or "Asset"))
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
        executed.append(
            {
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
        )

    ctx.pop("cpu_desperation_trade", None)
    ctx.pop("cpu_package_motive", None)
    return executed
