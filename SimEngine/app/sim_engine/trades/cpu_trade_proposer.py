"""
CPU-CPU trade proposer — routes ambient trades through the full trade engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.trades.trade_asset import team_id_of
from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry, get_team_owned_picks
from app.sim_engine.trades.trade_value import evaluate_player_asset_value, evaluate_pick_asset_value
from app.sim_engine.economy.team_needs import TeamNeeds

CPU_AMBIENT_FAIRNESS_GAP_MAX = 7.0
CPU_AMBIENT_MIN_INTEREST = 0.50


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


def _is_prospect(player: Any) -> bool:
    ident = getattr(player, "identity", None)
    age = _safe_int(getattr(ident, "age", getattr(player, "age", 25)), 25)
    return age <= 21


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
    ident = getattr(player, "identity", None)
    age = _safe_int(getattr(ident, "age", getattr(player, "age", 25)), 25)
    return years <= 1 and age >= 28


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
    # Soft ambient filter mirrors legality cooldown and extends shopper spam protection.
    if ctx is not None and _player_recently_acquired(player, ctx):
        return False
    if ctx is not None and bool(getattr(player, "acquired_via_trade", False)):
        cursor = int(ctx.get("calendar_cursor", 0) or 0)
        last_day = getattr(player, "last_acquired_day", None)
        try:
            if last_day is not None and (cursor - int(last_day)) < 21:
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
) -> Dict[str, Any]:
    season_year = int(getattr(league, "current_season", 0) or getattr(league, "season_year", 2025) or 2025)
    max_d = max(40, int(regular_season_last_index or 192))
    md = max(40, int(max(120, max_d) * 0.56))
    deadline_phase = max(0.0, min(1.0, (float(calendar_cursor) - float(md)) / max(20.0, float(max_d) * 0.2)))
    team_by_id = build_team_by_id(league)
    return {
        "league": league,
        "team_by_id": team_by_id,
        "season_year": season_year,
        "calendar_cursor": int(calendar_cursor or 0),
        "regular_season_last_index": max_d,
        "deadline_phase": deadline_phase,
        # Franchise socio ticks attach the live season ledger so executed CPU
        # trades retarget player_season_stats.team_id (Stats Central / cards).
        "player_season_stats": getattr(league, "player_season_stats", None),
    }


def _player_trade_value(player: Any, team: Any, league: Any, ctx: Dict[str, Any]) -> float:
    try:
        return float(evaluate_player_asset_value(player, team, team, league, context=ctx).get("total", 0.0))
    except Exception:
        return _player_ovr(player)


def _pick_trade_candidates(
    roster: List[Any],
    team: Any,
    *,
    seller: bool,
    league: Any,
    ctx: Dict[str, Any],
    acquiring_team_id: str,
) -> List[Any]:
    """Rank movable players. Sellers prefer surplus/rental/depth; buyers prefer affordable fits."""
    if not roster:
        return []
    deadline = _safe_float(ctx.get("deadline_phase"), 0.0)
    window = _team_window(team)
    scored: List[Tuple[float, float, Any]] = []
    for p in roster:
        if not _tradeable_player(p, acquiring_team_id, ctx=ctx):
            continue
        if _is_prospect(p) and seller and _player_ovr(p) >= 78:
            # Protect higher-end young pieces unless deep-rebuild late deadline.
            if not (window in ("rebuild", "declining") and deadline > 0.7):
                continue
        val = _player_trade_value(p, team, league, ctx)
        ovr = _player_ovr(p)
        fit = _needs_fit_score(team, p, selling=seller)
        # Soft-exclude franchise cores from ambient market.
        if ovr >= 88 and not (_is_rental(p) and deadline > 0.75 and window in ("rebuild", "declining")):
            continue
        priority = fit
        if seller:
            if _is_rental(p) and window in ("rebuild", "declining") and _playoff_odds(team) < 0.35:
                priority += 12.0 + deadline * 8.0
            elif 68.0 <= ovr <= 80.0:
                priority += 6.0  # depth / minor-market band
            elif ovr > 84:
                priority -= 8.0
            if window in ("rebuild", "declining") and ovr >= 82 and not _is_rental(p):
                priority -= 4.0
        else:
            if _is_rental(p) and window == "contender" and deadline > 0.45:
                priority += 10.0
            elif 68.0 <= ovr <= 82.0:
                priority += 5.0
            if ovr > 86:
                priority -= 6.0
        scored.append((priority, val, p))
    # Sellers: priority desc, then mid value first for fair matching.
    if seller:
        scored.sort(key=lambda x: (-x[0], abs(x[1] - 42.0)))
    else:
        scored.sort(key=lambda x: (-x[0], abs(x[1] - 40.0)))
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
) -> Optional[Any]:
    target = _player_trade_value(seller_asset, seller, league, ctx)
    ranked: List[Tuple[float, Any]] = []
    for p in buyer_candidates:
        pid = _player_id(p)
        if not pid or pid in used_players or pid == _player_id(seller_asset):
            continue
        if _needs_fit_score(buyer, p, selling=False) < 0.5 and not _is_rental(seller_asset):
            continue
        val = _player_trade_value(p, buyer, league, ctx)
        gap = abs(val - target)
        if gap <= value_band:
            ranked.append((gap - 0.15 * _needs_fit_score(buyer, p, selling=False), p))
    if not ranked:
        # Fallback: closest value within a wider but still bounded band.
        for p in buyer_candidates:
            pid = _player_id(p)
            if not pid or pid in used_players or pid == _player_id(seller_asset):
                continue
            val = _player_trade_value(p, buyer, league, ctx)
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
) -> Optional[Dict[str, Any]]:
    tid = team_id_of(team)
    picks = get_team_owned_picks(league, tid)
    season_year = int(ctx.get("season_year", 2025) or 2025)
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for row in picks:
        if bool(row.get("resolved")):
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
    return candidates[0][1]


def _build_package(
    seller: Any,
    buyer: Any,
    seller_asset: Any,
    buyer_assets: List[Any],
    *,
    seller_pick: Optional[Dict[str, Any]] = None,
    buyer_pick: Optional[Dict[str, Any]] = None,
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
    if _is_prospect(sold_player) or _is_prospect(return_player):
        reasons.append("PROSPECT_TIMELINE_FIT")
        if _is_prospect(sold_player) and buyer_window == "contender":
            reasons.append("PROSPECT_BLOCKED")
        category = "prospect_trade" if "prospect" not in category else category
    if seller_pick is not None or buyer_pick is not None:
        reasons.append("PICK_VALUE_REALLOCATION")
    if _safe_float(getattr(seller, "cap_pressure", 0.0), 0.0) >= 0.8 or seller_direction == "CAP_CORRECTION":
        category = "cap_trade"
        reasons.append("CAP_RELIEF" if _safe_float(getattr(seller, "cap_pressure", 0.0), 0.0) < 0.92 else "CAP_COMPLIANCE")
    if seller_window == buyer_window and not reasons:
        reasons = ["SIMILAR_VALUE_DIFFERENT_NEED", "POSITIONAL_SWAP"]
    if abs(sold_age - _safe_int(getattr(getattr(return_player, "identity", None), "age", getattr(return_player, "age", 25)), 25)) >= 6:
        reasons.append("AGE_TIMELINE_SWAP")
    if not reasons:
        reasons = ["POSITIONAL_SWAP", "ROSTER_BALANCE"]
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
    )
    ctx["cpu_ambient_trade"] = True
    ensure_draft_pick_registry(league, start_year=ctx.get("season_year"), years_ahead=4)
    team_by_id = ctx["team_by_id"]
    needs_model = TeamNeeds()
    deadline = _safe_float(ctx.get("deadline_phase"), 0.0)
    profiles = dict(getattr(league, "cpu_franchise_profiles", None) or {})

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

    for i in range(attempts):
        if len(executed) >= max(0, int(max_executions)):
            break
        # Alternate buyer/seller market with peer hockey-swap market.
        if i % 3 == 2 and peers:
            seller = peers[i % len(peers)]
            buyer = peers[(i * 5 + 2) % len(peers)]
        else:
            seller = sellers[i % len(sellers)]
            buyer = buyers[(i * 7) % len(buyers)]
        if seller is buyer:
            continue
        sid = team_id_of(seller)
        bid = team_id_of(buyer)
        if user_tid and (sid == user_tid or bid == user_tid):
            continue
        pair_key = tuple(sorted((sid, bid)))
        last_pair_day = int(recent_pairs.get(f"{pair_key[0]}|{pair_key[1]}", -999) or -999)
        if (int(calendar_cursor) - last_pair_day) < 10:
            continue
        if (sid, bid) in used_pairs:
            continue
        used_pairs.add((sid, bid))
        # Prospect / pick protection from ideology (soft skip, not legality bypass).
        if _ideo(seller, "prospect_protection", 0.5) >= 0.72 and _ideo(buyer, "aggression", 0.5) < 0.45:
            # Patient sellers are less likely to dump youth early.
            if deadline < 0.35 and i % 3 != 0:
                continue

        s_roster = list(getattr(seller, "roster", None) or [])
        b_roster = list(getattr(buyer, "roster", None) or [])
        if not s_roster or not b_roster:
            continue

        seller.needs = needs_model.evaluate(seller, context=ctx)
        buyer.needs = needs_model.evaluate(buyer, context=ctx)

        s_candidates = _pick_trade_candidates(
            s_roster, seller, seller=True, league=league, ctx=ctx, acquiring_team_id=bid,
        )
        b_candidates = _pick_trade_candidates(
            b_roster, buyer, seller=False, league=league, ctx=ctx, acquiring_team_id=sid,
        )
        if not s_candidates or not b_candidates:
            continue

        s_offer = s_candidates[i % len(s_candidates)]
        if _player_id(s_offer) in used_players:
            continue
        if _needs_fit_score(seller, s_offer, selling=True) < -2.0 and not _is_rental(s_offer):
            continue

        b_return = _match_return_player(
            seller_asset=s_offer,
            seller=seller,
            buyer=buyer,
            buyer_candidates=b_candidates,
            league=league,
            ctx=ctx,
            used_players=used_players,
            value_band=max(6.5, fairness_gap_max + 1.5),
        )
        if b_return is None:
            continue

        buyer_pick = None
        seller_pick = None
        seller_window = _team_window(seller)
        buyer_window = _team_window(buyer)
        protect_first = _ideo(buyer, "draft_pick_protection", 0.5) >= 0.62
        offer_val = _player_trade_value(s_offer, seller, league, ctx)
        return_val = _player_trade_value(b_return, buyer, league, ctx)
        value_delta = offer_val - return_val

        # Add a pick only when needed to balance a rental / futures gap — not by default.
        if _is_rental(s_offer) and buyer_window == "contender" and deadline > 0.4 and value_delta > 3.0:
            buyer_pick = _select_tradeable_pick(
                league, buyer, ctx=ctx, max_round=4 if deadline > 0.7 else 3,
                protect_own_first=protect_first,
            )
        elif seller_window in ("rebuild", "declining") and value_delta > 5.0:
            buyer_pick = _select_tradeable_pick(
                league, buyer, ctx=ctx, max_round=3, protect_own_first=protect_first,
            )
        elif value_delta < -5.0 and buyer_window == "contender":
            seller_pick = _select_tradeable_pick(
                league, seller, ctx=ctx, max_round=3, protect_own_first=True,
            )

        package = _build_package(
            seller,
            buyer,
            s_offer,
            [b_return],
            seller_pick=seller_pick,
            buyer_pick=buyer_pick,
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
        if gap > fairness_gap_max:
            continue
        if not evaluation.get("accepted"):
            continue
        interest = evaluation.get("interest_level") or {}
        if any(_safe_float(interest.get(t), 0.0) < CPU_AMBIENT_MIN_INTEREST for t in package.keys()):
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
        used_players.add(_player_id(s_offer))
        used_players.add(_player_id(b_return))
        # Annotate durable trade history with category/reason (additive, save-safe).
        try:
            hist = list(getattr(league, "trade_history", None) or [])
            for row in reversed(hist):
                if isinstance(row, dict) and str(row.get("trade_id") or "") == str(result.get("trade_id") or ""):
                    row.setdefault("trade_category", category)
                    row.setdefault("importance", importance)
                    row.setdefault("reason_codes", list(reason_codes))
                    row.setdefault("reason_text", reason_text)
                    break
            setattr(league, "trade_history", hist)
        except Exception:
            pass
        recent_pairs[f"{pair_key[0]}|{pair_key[1]}"] = int(calendar_cursor)
        outgoing_labels = [str(getattr(s_offer, "name", None) or "Player")]
        incoming_labels = [str(getattr(b_return, "name", None) or "Asset")]
        if seller_pick:
            yr = seller_pick.get("year")
            rnd = seller_pick.get("round")
            outgoing_labels.append(f"{yr} Round {rnd}" if yr and rnd else f"Pick {seller_pick.get('pick_id') or '?'}")
        if buyer_pick:
            yr = buyer_pick.get("year")
            rnd = buyer_pick.get("round")
            incoming_labels.append(f"{yr} Round {rnd}" if yr and rnd else f"Pick {buyer_pick.get('pick_id') or '?'}")
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
            }
        )

    return executed
