"""
Staggered free-agency decision market.

Players do not all sign on one wave. Each FA tracks offers, patience, urgency,
peer signings, and a decision state that evolves day-by-day as the user sims
calendar time or advances the FA market.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from services.contract_economy import (
    LEAGUE_MINIMUM_AAV_M,
    CPU_SIGN_MIN_FIT_SCORE,
    _cpu_negotiate_offer,
    _player_age,
    _player_id,
    _player_name,
    _player_ovr,
    _position_bucket,
    compute_fair_aav,
    cpu_signing_blocked,
    evaluate_team_position_needs,
    generate_contract_terms,
    score_free_agent_fit,
    sign_player_to_team,
    sync_all_team_cap_fields,
)


# Decision states shown in UI / storylines
STATE_AWAITING = "awaiting_offers"
STATE_EVALUATING = "evaluating_offers"
STATE_GAUGING = "gauging_market"
STATE_WAITING_PEERS = "waiting_on_peers"
STATE_HOLDING = "holding_out"
STATE_LEANING = "leaning_to_sign"
STATE_SIGNED = "signed"


def _rng_unit(seed: str) -> float:
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tier_for_ovr(ovr: float) -> str:
    if ovr >= 88:
        return "star"
    if ovr >= 82:
        return "high"
    if ovr >= 76:
        return "mid"
    if ovr >= 70:
        return "depth"
    return "fringe"


def _base_patience(ovr: float, age: int) -> float:
    """0–1: higher = waits longer before accepting."""
    tier = _tier_for_ovr(ovr)
    base = {
        "star": 0.72,
        "high": 0.66,
        "mid": 0.50,
        "depth": 0.32,
        "fringe": 0.18,
    }[tier]
    if age >= 35:
        base -= 0.22
    elif age >= 32:
        base -= 0.12
    elif age <= 26:
        base += 0.06
    return max(0.08, min(0.95, base))


def _ask_for_player(player: Any, league: Any, *, days_on_market: int = 0, offer_count: int = 0) -> float:
    """Central open-market ask — fair value plus mild opener premium, then cold-market decay.

    Must stay coherent with ``compute_player_demand`` / ``compute_fair_aav`` so the UI
    ask and backend acceptance floor do not diverge.
    """
    from services.contract_economy import compute_player_demand

    fair = float(compute_fair_aav(player, None, league) or LEAGUE_MINIMUM_AAV_M)
    ovr = float(_player_ovr(player))
    demand = compute_player_demand(player, None, league, context="ufa")
    want = float(demand.get("want_aav_m") or fair)
    stored = getattr(player, "asking_aav_m", None)
    if stored is None:
        stored = getattr(player, "ask_aav_m", None)

    # Depth / cheap board: ignore inflated stored asks from the old valuation curve.
    if ovr < 70:
        ask = max(LEAGUE_MINIMUM_AAV_M, min(want, fair * 1.02))
    elif ovr < 76:
        ask = max(LEAGUE_MINIMUM_AAV_M, min(want, fair * 1.04))
    elif stored is None:
        ask = want
    else:
        ask = min(float(stored), want * 1.05, fair * 1.12)
        ask = max(fair * 0.98, ask)

    # Time-on-market decay (bounded). Elite players soften modestly; fringe collapse
    # toward the league minimum when nobody is bidding.
    days = max(0, int(days_on_market or 0))
    offers = max(0, int(offer_count or 0))
    if days >= 5 and ovr >= 88 and offers <= 1:
        floor = max(LEAGUE_MINIMUM_AAV_M, fair * 0.84)
        ask = max(floor, ask * (0.988 if offers == 0 else 0.994))
    if days >= 3 and offers == 0:
        if ovr < 70:
            ask = max(LEAGUE_MINIMUM_AAV_M, ask * 0.96)
        elif ovr < 76:
            ask = max(LEAGUE_MINIMUM_AAV_M, ask * 0.975)
    if days >= 8:
        if ovr >= 88:
            floor = max(LEAGUE_MINIMUM_AAV_M, fair * 0.82)
            ask = max(floor, ask * (0.985 if offers == 0 else 0.992))
        elif ovr >= 82:
            floor = max(LEAGUE_MINIMUM_AAV_M, fair * 0.72)
            ask = max(floor, ask * (0.97 if offers == 0 else 0.985))
        elif ovr >= 76:
            floor = max(LEAGUE_MINIMUM_AAV_M, fair * 0.62)
            ask = max(floor, ask * (0.95 if offers == 0 else 0.975))
        else:
            floor = LEAGUE_MINIMUM_AAV_M
            ask = max(floor, ask * (0.92 if offers == 0 else 0.96))
    if days >= 20:
        if ovr >= 88:
            floor = max(LEAGUE_MINIMUM_AAV_M, fair * 0.78)
            ask = max(floor, min(ask, fair * 0.92 if offers == 0 else ask))
        elif ovr < 76:
            ask = max(LEAGUE_MINIMUM_AAV_M, min(ask, fair * 0.88 if offers <= 1 else ask))
    if days >= 28 and ovr < 82 and offers <= 1:
        # Late August / September: replacement and mid-tier accept prove-it money.
        ask = max(LEAGUE_MINIMUM_AAV_M, min(ask, fair * (0.85 if ovr >= 76 else 0.78)))

    ask = round(max(LEAGUE_MINIMUM_AAV_M, float(ask)), 3)
    try:
        setattr(player, "asking_aav_m", ask)
    except Exception:
        pass
    return ask


def _iter_domestic_fa_pool(league: Any) -> List[Any]:
    if league is None:
        return []
    out: List[Any] = []
    seen: set = set()
    for p in list(_get(league, "free_agents", None) or []):
        if getattr(p, "retired", False):
            continue
        pid = _player_id(p)
        key = pid or id(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _iter_overseas_fa_pool(league: Any) -> List[Any]:
    if league is None:
        return []
    out: List[Any] = []
    seen: set = set()
    for p in list(_get(league, "overseas_free_agents", None) or []):
        if getattr(p, "retired", False):
            continue
        pid = _player_id(p)
        key = pid or id(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _iter_fa_market_pool(league: Any, session: Any = None, *, include_overseas: Optional[bool] = None) -> List[Any]:
    """Domestic UFAs on the wire immediately; overseas imports join after day 10."""
    domestic = _iter_domestic_fa_pool(league)
    if include_overseas is None:
        day = 999
        if session is not None:
            day = int(getattr(session, "fa_market_day", 0) or 0)
            book = getattr(session, "fa_market_book", None) or {}
            day = max(day, int(book.get("day") or 0))
        include_overseas = day >= 10
    if not include_overseas:
        return domestic
    return domestic + _iter_overseas_fa_pool(league)


def ensure_fa_market_book(session: Any) -> Dict[str, Any]:
    """Create / refresh the per-player FA decision book without signing anyone."""
    book = getattr(session, "fa_market_book", None)
    if not isinstance(book, dict):
        book = {"version": 1, "entries": {}, "peer_signings": [], "day": 0, "log": []}
        session.fa_market_book = book

    league = getattr(getattr(session, "sim", None), "league", None)
    fa_pool = _iter_fa_market_pool(league, session, include_overseas=True)
    entries: Dict[str, Any] = dict(book.get("entries") or {})
    day = int(book.get("day") or 0)
    season = int(getattr(session, "season_calendar_year", 2025) or 2025)

    live_ids = set()
    for player in fa_pool:
        pid = _player_id(player)
        if not pid:
            continue
        live_ids.add(pid)
        if pid in entries and entries[pid].get("state") != STATE_SIGNED:
            days_live = int(entries[pid].get("days_on_market") or 0)
            offers_live = int(entries[pid].get("offer_count") or 0)
            entries[pid]["ask_aav_m"] = _ask_for_player(
                player, league, days_on_market=days_live, offer_count=offers_live
            )
            entries[pid]["fair_aav_m"] = round(compute_fair_aav(player, None, league), 3)
            entries[pid]["player_ref"] = player
            continue
        ovr = float(_player_ovr(player))
        age = int(_player_age(player))
        patience = _base_patience(ovr, age)
        patience += (_rng_unit(f"pat|{pid}|{season}") - 0.5) * 0.12
        patience = max(0.08, min(0.96, patience))
        entries[pid] = {
            "player_id": pid,
            "name": _player_name(player),
            "position": _position_bucket(player),
            "overall": round(ovr),
            "age": age,
            "tier": _tier_for_ovr(ovr),
            "ask_aav_m": _ask_for_player(player, league, days_on_market=0, offer_count=0),
            "fair_aav_m": round(compute_fair_aav(player, None, league), 3),
            "days_on_market": 0,
            "patience": round(patience, 3),
            "urgency": 0.05,
            "state": STATE_AWAITING,
            "reason": "Waiting for opening offers",
            "offers": [],
            "best_offer_m": 0.0,
            "offer_count": 0,
            "evaluate_until_day": None,
            "peer_watch_threshold_m": None,
            "player_ref": player,
        }

    # Drop signed / gone
    for pid in list(entries.keys()):
        if pid not in live_ids and entries[pid].get("state") != STATE_SIGNED:
            entries.pop(pid, None)

    book["entries"] = entries
    book["day"] = day
    session.fa_market_book = book
    return book


def _is_exclusive_home_ufa(player: Any, session: Any = None) -> bool:
    if bool(getattr(player, "ufa_exclusive", False)):
        return True
    meta = getattr(player, "_franchise_assignment", None) or {}
    if meta.get("overseas"):
        return False
    from_tid = str(
        getattr(player, "ufa_from_team_id", None)
        or getattr(player, "previous_nhl_team_id", None)
        or ""
    )
    if not from_tid:
        return False
    if session is not None and bool(getattr(session, "free_agency_open", False)):
        return False
    return True


def _serious_cpu_offer_aav(
    *,
    fair: float,
    ovr: float,
    cap_space_m: float,
    discount: float,
    rng: Any,
    days_on_market: int = 0,
) -> Optional[float]:
    """Build a market-credible offer. Cap-strapped clubs skip stars instead of insulting them."""
    space = float(cap_space_m or 0.0)
    target = float(fair) * float(discount) * float(rng.uniform(0.92, 1.06))
    floor = LEAGUE_MINIMUM_AAV_M
    days = max(0, int(days_on_market or 0))
    if ovr >= 88:
        # Late market: allow shorter-money serious bids rather than total silence.
        floor_mult = 0.58 if days >= 18 else (0.64 if days >= 10 else 0.68)
        floor = max(floor, fair * floor_mult)
    elif ovr >= 82:
        floor_mult = 0.55 if days >= 18 else 0.62
        floor = max(floor, fair * floor_mult)
    elif ovr >= 76:
        floor = max(floor, fair * 0.52)

    if space < floor:
        # Can't make a serious bid — sit out rather than post a mirage $1M offer.
        return None

    # Spend up to ~98% of space for stars; depth stays near 95%.
    spend_frac = 0.98 if ovr >= 86 else 0.95
    offer = min(target, space * spend_frac)
    offer = max(offer, min(floor, space * spend_frac))
    if offer < floor * 0.98 and ovr >= 80:
        return None
    return round(max(LEAGUE_MINIMUM_AAV_M, offer), 3)


def _min_fit_for_ovr(ovr: float) -> float:
    """Fringe players need a real fit; stars only need a non-broken roster fit."""
    if ovr < 70:
        return 0.42
    if ovr < 76:
        return 0.34
    if ovr < 82:
        return 0.26
    if ovr < 88:
        return 0.18
    return 0.12


def _collect_cpu_offers(
    session: Any,
    *,
    max_new_offers: int,
) -> List[Dict[str, Any]]:
    """CPU clubs pursue free agents by need and quality — not every name on the Wire.

    Critical ordering: evaluate higher-OVR players first so team daily offer budgets
    are not exhausted on 55-OVR fringe players before stars see the market.
    """
    sim = session.sim
    league = getattr(sim, "league", None)
    if league is None:
        return []
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    rng = sim.rng
    try:
        sync_all_team_cap_fields(league, sim, season_year=season_year)
    except Exception:
        pass
    book = ensure_fa_market_book(session)
    day = int(book.get("day") or 0)
    entries = book["entries"]
    fa_pool = [
        p for p in _iter_fa_market_pool(league, session)
        if not _is_exclusive_home_ufa(p, session)
    ]
    if not fa_pool:
        return []

    # Stars / NHL-calibre first — prevents fringe UFAs from consuming every club's
    # daily offer quota (root cause of unsigned elites + 55-OVR bidding wars).
    fa_pool.sort(key=lambda p: (-float(_player_ovr(p)), _player_id(p)))

    user_tid = str(getattr(session, "user_team_id", "") or "")
    cpu_teams = [
        t for t in list(_get(league, "teams", None) or [])
        if str(_get(t, "team_id", "") or _get(t, "id", "")) != user_tid
    ]
    # Bound offer volume to the daily market cap — do not invent unlimited bids.
    offer_budget = max(1, int(max_new_offers or 0))

    # Precompute team contexts once.
    team_ctx: Dict[str, Dict[str, Any]] = {}
    for team in cpu_teams:
        tid = str(_get(team, "team_id", "") or _get(team, "id", ""))
        if not tid:
            continue
        try:
            team_ctx[tid] = {
                "team": team,
                "ctx": evaluate_team_position_needs(team, league, sim, season_year=season_year),
            }
        except Exception:
            continue

    new_offers: List[Dict[str, Any]] = []
    team_offer_counts: Dict[str, int] = {}
    team_fringe_counts: Dict[str, int] = {}

    for player in fa_pool:
        if len(new_offers) >= offer_budget:
            break
        pid = _player_id(player)
        entry = entries.get(pid)
        if not entry or entry.get("state") == STATE_SIGNED:
            continue
        ovr = float(_player_ovr(player))
        pos = _position_bucket(player)
        fair_base = compute_fair_aav(player, None, league)
        existing_offers = list(entry.get("offers") or [])
        # Cap concurrent bidders on replaceable players — bidding wars require
        # genuine multi-team need, which fringe talent almost never creates.
        if ovr < 70 and len(existing_offers) >= 2:
            continue
        if ovr < 76 and len(existing_offers) >= 4:
            continue
        if ovr < 82 and len(existing_offers) >= 8:
            continue

        # Shuffle club order per player so the same teams aren't always first.
        order = list(team_ctx.items())
        rng.shuffle(order)

        for tid, pack in order:
            if len(new_offers) >= offer_budget:
                break
            # Re-check bidder caps inside the team loop — otherwise one day can
            # still attach every club before the outer gate sees the new count.
            existing_offers = list(entry.get("offers") or [])
            if ovr < 70 and len(existing_offers) >= 2:
                break
            if ovr < 76 and len(existing_offers) >= 4:
                break
            if ovr < 82 and len(existing_offers) >= 8:
                break
            # Soft per-team daily load. Fringe/depth share a tighter sub-cap so
            # clubs keep powder dry for real NHL targets.
            per_team_cap = 12 if day <= 2 else (10 if day <= 7 else 8)
            if team_offer_counts.get(tid, 0) >= per_team_cap:
                continue
            if ovr < 70 and team_fringe_counts.get(tid, 0) >= 2:
                continue
            if ovr < 76 and team_fringe_counts.get(tid, 0) >= 3:
                continue
            if any(o.get("team_id") == tid and int(o.get("day", -99)) >= day - 1 for o in existing_offers):
                continue

            team = pack["team"]
            ctx = pack["ctx"]
            if ctx["slots_remaining"] <= 0:
                continue
            spendable = float(ctx.get("spendable_cap_space_m", ctx.get("cap_space_m", 0)) or 0)
            if spendable < LEAGUE_MINIMUM_AAV_M * 1.02:
                continue

            need = float(ctx["need_score"].get(pos, 0))
            counts = ctx.get("counts") or {}
            # Pursuit gate: weak players only when the club has a real hole.
            if ovr < 70:
                if need < 0.48:
                    continue
            elif ovr < 76:
                if need < 0.32:
                    continue
            elif ovr < 82:
                if need < 0.18 and float(ctx.get("best_ovr", {}).get(pos, 0) or 0) >= ovr + 2:
                    continue

            window = ctx["window"]
            discount = 0.94 if window == "rebuilder" else (1.04 if need >= 0.45 else 0.98)
            if window == "cap_strapped":
                discount = 0.90
            if ovr >= 88:
                discount = max(discount, 0.96)
            # Late market: clubs chase unsigned stars more aggressively.
            if day >= 15 and ovr >= 86 and int(entry.get("offer_count") or 0) <= 1:
                discount = max(discount, 1.0)

            fair = compute_fair_aav(player, team, league) or fair_base
            offer_aav = _serious_cpu_offer_aav(
                fair=float(fair),
                ovr=ovr,
                cap_space_m=spendable,
                discount=discount,
                rng=rng,
                days_on_market=int(entry.get("days_on_market") or day),
            )
            if offer_aav is None:
                continue
            if cpu_signing_blocked(team, player, ctx, offer_aav):
                continue

            _, years, _ = generate_contract_terms(player, team, league, rng, context="ufa")
            if window == "rebuilder":
                years = min(years, 4)
            elif window == "cap_strapped":
                years = min(years, 3)
            if ovr >= 88:
                years = max(years, 3)
            # Fringe / late depth: prefer short prove-it deals.
            if ovr < 76:
                years = min(years, 2 if ovr < 70 else 3)
            if day >= 20 and ovr < 82 and int(entry.get("offer_count") or 0) <= 1:
                years = min(years, 2)

            fit, _reasons = score_free_agent_fit(team, player, ctx, offer_aav, years, league)
            min_fit = _min_fit_for_ovr(ovr)
            if ovr >= 86 and day >= 12:
                min_fit = max(0.08, min_fit - 0.06)
            if fit < min_fit:
                continue

            team_name = str(_get(team, "name", "") or _get(team, "team_name", "") or tid)
            team_abbrev = str(
                _get(team, "abbreviation", None)
                or _get(team, "abbrev", None)
                or _get(team, "abbr", None)
                or tid[:3]
            ).upper()
            offer = {
                "team_id": tid,
                "team_name": team_name,
                "team_abbrev": team_abbrev,
                "aav_m": offer_aav,
                "years": years,
                "day": day,
                "ntc": False,
                "nmc": bool(ovr >= 90 and years >= 5),
                "fit": round(fit, 3),
                "need": round(need, 3),
            }
            offers = [o for o in existing_offers if o.get("team_id") != tid] + [offer]
            entry["offers"] = offers[-32:]
            existing_offers = entry["offers"]
            entry["offer_count"] = len(entry["offers"])
            entry["best_offer_m"] = round(
                max(float(entry.get("best_offer_m") or 0), offer_aav), 3
            )
            # Refresh ask with live market pressure.
            entry["ask_aav_m"] = _ask_for_player(
                player,
                league,
                days_on_market=int(entry.get("days_on_market") or 0),
                offer_count=int(entry.get("offer_count") or 0),
            )
            new_offers.append({"player_id": pid, **offer})
            team_offer_counts[tid] = team_offer_counts.get(tid, 0) + 1
            if ovr < 76:
                team_fringe_counts[tid] = team_fringe_counts.get(tid, 0) + 1

    return new_offers


def _update_player_decision(entry: Dict[str, Any], book: Dict[str, Any], *, day: int) -> None:
    """Advance one FA's decision state from offers / patience / peers."""
    if entry.get("state") == STATE_SIGNED:
        return

    patience = float(entry.get("patience") or 0.5)
    days = int(entry.get("days_on_market") or 0)
    offers = list(entry.get("offers") or [])
    offer_count = len(offers)
    best = float(entry.get("best_offer_m") or 0)
    ask = float(entry.get("ask_aav_m") or LEAGUE_MINIMUM_AAV_M)
    fair = float(entry.get("fair_aav_m") or ask)
    tier = str(entry.get("tier") or "mid")
    age = int(entry.get("age") or 28)

    # Urgency rises with days without a close offer, and with age
    no_close = best < ask * 0.92
    star_tier = tier in ("star", "high")
    urgency = 0.08 + days * (0.055 if tier in ("depth", "fringe") else (0.048 if star_tier else 0.035))
    if no_close:
        urgency += 0.05 * max(0, days - 2)
    if star_tier and offer_count == 0 and days >= 8:
        urgency += 0.08 + min(0.12, (days - 7) * 0.015)
    if age >= 33:
        urgency += 0.10
    if offer_count == 0 and days >= 3:
        urgency += 0.12
    urgency = min(0.98, urgency)
    entry["urgency"] = round(urgency, 3)

    # Soften ask when market is cold — centralized decay keeps ask coherent with demand.
    player_ref = entry.get("player_ref")
    if player_ref is not None:
        try:
            entry["ask_aav_m"] = _ask_for_player(
                player_ref,
                None,
                days_on_market=days,
                offer_count=offer_count,
            )
            ask = float(entry["ask_aav_m"])
        except Exception:
            if days >= 5 and offer_count == 0:
                entry["ask_aav_m"] = round(max(fair * 0.92, ask * 0.97, LEAGUE_MINIMUM_AAV_M), 3)
                ask = float(entry["ask_aav_m"])
    elif days >= 5 and offer_count == 0:
        entry["ask_aav_m"] = round(max(fair * 0.92, ask * 0.97, LEAGUE_MINIMUM_AAV_M), 3)
        ask = float(entry["ask_aav_m"])
    elif days >= 8 and best < ask * 0.9:
        entry["ask_aav_m"] = round(
            max(best * 1.02 if best else fair * 0.9, ask * 0.96, LEAGUE_MINIMUM_AAV_M),
            3,
        )
        ask = float(entry["ask_aav_m"])

    # Late-market term flexibility for non-stars with weak boards.
    if days >= 18 and tier in ("depth", "fringe", "mid") and offer_count <= 2:
        entry["prefer_short_term"] = True
    if days >= 25 and tier in ("star", "high") and (offer_count == 0 or best < ask * 0.85):
        entry["prefer_short_term"] = True
        entry["late_market_flex"] = True

    # Peer pressure: similar-tier recent signings
    peer_hit = None
    pos = entry.get("position")
    for peer in list(book.get("peer_signings") or [])[-20:]:
        if peer.get("position") != pos:
            continue
        if abs(int(peer.get("overall") or 0) - int(entry.get("overall") or 0)) > 4:
            continue
        if int(peer.get("day") or 0) >= day - 5:
            peer_hit = peer
            break

    # State machine
    if offer_count == 0:
        if days < 2:
            entry["state"] = STATE_AWAITING
            entry["reason"] = "Awaiting opening offers"
        elif days < 6:
            entry["state"] = STATE_GAUGING
            entry["reason"] = "Gauging the market — no firm offers yet"
        else:
            entry["state"] = STATE_HOLDING
            entry["reason"] = "Holding out; limited interest so far"
        return

    # Has offers
    ratio = best / max(ask, 0.01)
    if offer_count == 1 and days - int(offers[0].get("day") or 0) < 2 and ratio < 1.02:
        entry["state"] = STATE_EVALUATING
        entry["reason"] = "Evaluating first offer"
        entry["evaluate_until_day"] = day + (2 if patience > 0.55 else 1)
        return

    if offer_count >= 2 and ratio < 1.05 and patience > 0.45 and urgency < 0.55:
        entry["state"] = STATE_EVALUATING
        entry["reason"] = f"Comparing {offer_count} offers"
        entry["evaluate_until_day"] = day + 1
        return

    if peer_hit and ratio < 1.0 and patience > 0.5 and urgency < 0.6:
        entry["state"] = STATE_WAITING_PEERS
        entry["reason"] = (
            f"Waiting on peer market after {peer_hit.get('name', 'a peer')} signed "
            f"at ${float(peer_hit.get('aav_m') or 0):.2f}M"
        )
        entry["peer_watch_threshold_m"] = float(peer_hit.get("aav_m") or 0)
        return

    if ratio < 0.92 and urgency < patience and not entry.get("late_market_flex"):
        entry["state"] = STATE_HOLDING
        entry["reason"] = "Holding out for a better offer"
        return

    lean_ratio = 0.90 if entry.get("late_market_flex") or entry.get("prefer_short_term") else 0.94
    if (
        ratio >= lean_ratio
        or urgency >= patience * 0.92
        or (ratio >= 0.88 and days >= int(3 + patience * 6))
        or (entry.get("late_market_flex") and ratio >= 0.82 and days >= 25)
    ):
        entry["state"] = STATE_LEANING
        if ratio >= 1.0:
            entry["reason"] = "Ready to accept top offer"
        elif urgency >= patience * 0.92:
            entry["reason"] = "Urgency rising — leaning toward best available deal"
        elif entry.get("late_market_flex"):
            entry["reason"] = "Late market — accepting realistic terms"
        else:
            entry["reason"] = "Market timing favors signing now"
        return

    entry["state"] = STATE_GAUGING
    entry["reason"] = "Still gauging remaining suitors"


def _try_sign_leaning_players(
    session: Any,
    *,
    max_signings: int,
) -> List[Dict[str, Any]]:
    """Only players in LEANING state can sign today — and only if negotiation accepts."""
    sim = session.sim
    league = getattr(sim, "league", None)
    if league is None:
        return []
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    book = ensure_fa_market_book(session)
    day = int(book.get("day") or 0)
    entries = book["entries"]
    fa_pool = _iter_fa_market_pool(league, session)
    team_by_id = {
        str(_get(t, "team_id", "") or _get(t, "id", "")): t
        for t in list(_get(league, "teams", None) or [])
    }
    user_tid = str(getattr(session, "user_team_id", "") or "")

    leaning = [
        e for e in entries.values()
        if e.get("state") == STATE_LEANING and e.get("offers")
    ]
    # Stars first on day 1–2 feel wrong — sort by urgency * inverse patience so depth clears,
    # but allow stars when urgency catches up.
    leaning.sort(
        key=lambda e: (
            -float(e.get("urgency") or 0) * (1.15 - float(e.get("patience") or 0.5)),
            -float(e.get("best_offer_m") or 0),
        )
    )

    signings: List[Dict[str, Any]] = []
    for entry in leaning:
        if len(signings) >= max_signings:
            break
        player = entry.get("player_ref")
        if player is None or player not in fa_pool:
            # Resolve from pool
            pid = entry.get("player_id")
            player = next((p for p in fa_pool if _player_id(p) == pid), None)
            if player is None:
                continue
            entry["player_ref"] = player

        # Prefer best offer from a CPU team (user must sign via UI)
        offers = sorted(
            [o for o in entry.get("offers") or [] if str(o.get("team_id")) != user_tid],
            key=lambda o: (
                -float(o.get("aav_m") or 0),
                -int(o.get("years") or 0),
            ),
        )
        if not offers:
            continue

        signed = False
        # Late-market stars: try a wider board — first suitor often can't clear clauses/cap.
        attempt_n = 6 if float(entry.get("overall") or 0) >= 86 else 3
        for offer in offers[:attempt_n]:
            tid = str(offer.get("team_id"))
            team = team_by_id.get(tid)
            if team is None:
                continue
            ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
            offer_aav = float(offer.get("aav_m") or 0)
            # Stars may spend full usable room; reserve is for *other* extensions.
            space = float(ctx.get("cap_space_m", 0) or 0)
            spendable = float(ctx.get("spendable_cap_space_m", space) or 0)
            if float(entry.get("overall") or 0) >= 86:
                spendable = max(spendable, space)
            if spendable < offer_aav * 0.5 and space < offer_aav * 0.5:
                continue
            years = int(offer.get("years") or 1)
            if entry.get("prefer_short_term"):
                years = min(years, 2 if float(entry.get("overall") or 0) < 82 else 3)
            try:
                setattr(player, "days_on_market", int(entry.get("days_on_market") or 0))
                setattr(player, "fa_offer_count", int(entry.get("offer_count") or 0))
            except Exception:
                pass
            ctx_sign = dict(ctx)
            ctx_sign["spendable_cap_space_m"] = spendable
            ctx_sign["cap_space_m"] = max(space, spendable)
            ctx_sign["days_on_market"] = int(entry.get("days_on_market") or 0)
            ctx_sign["offer_count"] = int(entry.get("offer_count") or 0)
            agreed, final_aav, final_years = _cpu_negotiate_offer(
                team,
                player,
                league,
                offer_aav,
                years,
                ctx_sign,
                context="ufa",
                max_rounds=6 if float(entry.get("overall") or 0) >= 88 else 3,
            )
            if not agreed:
                entry["state"] = STATE_EVALUATING
                entry["reason"] = "Rejected latest terms — still evaluating"
                continue
            result = sign_player_to_team(
                player,
                team,
                league,
                season_year,
                {
                    "aav_m": final_aav,
                    "years": final_years,
                    "context": "ufa",
                    "force": True,
                    "days_on_market": int(entry.get("days_on_market") or 0),
                    "offer_count": int(entry.get("offer_count") or 0),
                    "ntc": True if float(entry.get("overall") or 0) >= 86 else False,
                    "nmc": False,
                },
            )
            if not result.get("ok"):
                continue
            if player in fa_pool:
                fa_pool.remove(player)
            team_name = str(
                _get(team, "name", "") or _get(team, "team_name", "") or _get(team, "city", "") or tid
            )
            team_abbrev = str(
                _get(team, "abbreviation", None) or _get(team, "abbrev", None) or ""
            ).upper()
            entry["state"] = STATE_SIGNED
            entry["reason"] = f"Signed with {team_name}"
            entry["signed_aav_m"] = final_aav
            entry["signed_years"] = final_years
            entry["signed_team_id"] = tid
            entry["signed_team_name"] = team_name
            row = {
                "team_id": tid,
                "team_name": team_name,
                "team_abbrev": team_abbrev,
                "player_id": _player_id(player),
                "name": _player_name(player),
                "aav_m": final_aav,
                "years": final_years,
                "position": _position_bucket(player),
                "overall": round(_player_ovr(player)),
                "day": day,
                "decision_path": entry.get("reason"),
                "text": (
                    f"{team_name} signs {_player_name(player)} · "
                    f"{final_aav}M × {final_years}y"
                ),
            }
            signings.append(row)
            peers = list(book.get("peer_signings") or [])
            peers.append(row)
            book["peer_signings"] = peers[-40:]
            log = list(book.get("log") or [])
            log.append({"kind": "signing", "text": row["text"]})
            book["log"] = log[-48:]
            signed = True
            break

        if not signed and entry.get("state") == STATE_LEANING:
            # Could not close — drop back to holding/evaluating
            entry["state"] = STATE_HOLDING
            entry["reason"] = "Best suit couldn't close — holding for new offers"

    # Late-market elite closer: when money is on the table but clause/personality
    # deadlock keeps a star unsigned into September, execute the best affordable
    # serious offer. Does not invent terms or force the former team.
    if day >= 22:
        for entry in list(entries.values()):
            if len(signings) >= max_signings:
                break
            if entry.get("state") == STATE_SIGNED:
                continue
            ovr = float(entry.get("overall") or 0)
            if ovr < 88:
                continue
            days_m = int(entry.get("days_on_market") or 0)
            if days_m < 22:
                continue
            ask = float(entry.get("ask_aav_m") or 0)
            best = float(entry.get("best_offer_m") or 0)
            player = entry.get("player_ref")
            pid = entry.get("player_id")
            if player is None:
                player = next((p for p in fa_pool if _player_id(p) == pid), None)
            if player is None or player not in fa_pool:
                continue
            # No serious board after 4 weeks: decay ask and force a fair signing
            # onto a cap-solvent CPU need fit (soak ELITE_UNSIGNED / Sept stars).
            if best < max(ask * 0.88, LEAGUE_MINIMUM_AAV_M * 4):
                if not (days_m >= 28 and ovr >= 90):
                    continue
                fair = float(compute_fair_aav(player, None, league) or LEAGUE_MINIMUM_AAV_M)
                ask = round(max(LEAGUE_MINIMUM_AAV_M, min(ask, fair * 0.9)), 3)
                entry["ask_aav_m"] = ask
                candidates = []
                for tid, team in team_by_id.items():
                    if tid == user_tid:
                        continue
                    try:
                        ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
                    except Exception:
                        continue
                    space = float(ctx.get("cap_space_m", 0) or 0)
                    if space < ask * 0.95:
                        continue
                    need = float((ctx.get("need_score_by_pos") or {}).get(_position_bucket(player), 0) or 0)
                    candidates.append((space + need * 2.0, tid, team, space))
                candidates.sort(key=lambda row: -row[0])
                closed = False
                for _, tid, team, space in candidates[:6]:
                    aav = round(min(space * 0.95, max(ask, fair * 0.88)), 3)
                    years = 3 if ovr >= 92 else 2
                    result = sign_player_to_team(
                        player,
                        team,
                        league,
                        season_year,
                        {
                            "aav_m": aav,
                            "years": years,
                            "context": "ufa",
                            "force": True,
                            "ntc": True,
                            "nmc": False,
                            "days_on_market": days_m,
                            "offer_count": int(entry.get("offer_count") or 0),
                        },
                    )
                    if not result.get("ok"):
                        continue
                    if player in fa_pool:
                        fa_pool.remove(player)
                    team_name = str(_get(team, "name", "") or _get(team, "team_name", "") or tid)
                    entry["state"] = STATE_SIGNED
                    entry["reason"] = f"Late-market emergency signing with {team_name}"
                    entry["best_offer_m"] = aav
                    signings.append({
                        "team_id": tid,
                        "team_name": team_name,
                        "team_abbrev": str(
                            _get(team, "abbreviation", None) or _get(team, "abbrev", None) or ""
                        ).upper(),
                        "player_id": _player_id(player),
                        "name": _player_name(player),
                        "aav_m": aav,
                        "years": years,
                        "position": _position_bucket(player),
                        "overall": round(ovr),
                        "day": day,
                        "decision_path": "late_market_elite_emergency",
                        "text": f"{team_name} signs {_player_name(player)} · {aav}M × {years}y (late market)",
                    })
                    closed = True
                    break
                if closed:
                    continue
                continue
            offers = sorted(
                [o for o in entry.get("offers") or [] if str(o.get("team_id")) != user_tid],
                key=lambda o: -float(o.get("aav_m") or 0),
            )
            for offer in offers[:8]:
                tid = str(offer.get("team_id"))
                team = team_by_id.get(tid)
                if team is None:
                    continue
                ctx = evaluate_team_position_needs(team, league, sim, season_year=season_year)
                space = float(ctx.get("cap_space_m", 0) or 0)
                aav = float(offer.get("aav_m") or 0)
                if aav > space * 0.98 or aav < ask * 0.85:
                    continue
                years = min(int(offer.get("years") or 3), 4)
                result = sign_player_to_team(
                    player,
                    team,
                    league,
                    season_year,
                    {
                        "aav_m": aav,
                        "years": max(1, years),
                        "context": "ufa",
                        "force": True,
                        "ntc": True,
                        "nmc": False,
                        "days_on_market": days_m,
                        "offer_count": int(entry.get("offer_count") or 0),
                    },
                )
                if not result.get("ok"):
                    continue
                if player in fa_pool:
                    fa_pool.remove(player)
                team_name = str(
                    _get(team, "name", "") or _get(team, "team_name", "") or tid
                )
                entry["state"] = STATE_SIGNED
                entry["reason"] = f"Late-market signing with {team_name}"
                row = {
                    "team_id": tid,
                    "team_name": team_name,
                    "team_abbrev": str(
                        _get(team, "abbreviation", None) or _get(team, "abbrev", None) or ""
                    ).upper(),
                    "player_id": _player_id(player),
                    "name": _player_name(player),
                    "aav_m": aav,
                    "years": years,
                    "position": _position_bucket(player),
                    "overall": round(ovr),
                    "day": day,
                    "decision_path": "late_market_elite_close",
                    "text": (
                        f"{team_name} signs {_player_name(player)} · "
                        f"{aav}M × {years}y (late market)"
                    ),
                }
                signings.append(row)
                peers = list(book.get("peer_signings") or [])
                peers.append(row)
                book["peer_signings"] = peers[-40:]
                break

    # sign_player_to_team already pulls the player out of free_agents / overseas pools.
    return signings


def tick_free_agency_market(
    session: Any,
    *,
    days: int = 1,
    max_signings_per_day: Optional[int] = None,
    max_offers_per_day: Optional[int] = None,
    opening_day: bool = False,
) -> Dict[str, Any]:
    """
    Advance the FA market by N days. Safe to call from calendar sim or FA UI.

    Returns aggregated signings, offer activity, and decision snapshots.
    """
    days = max(1, int(days))
    book = ensure_fa_market_book(session)
    all_signings: List[Dict[str, Any]] = []
    all_offers: List[Dict[str, Any]] = []
    day_summaries: List[Dict[str, Any]] = []

    for _ in range(days):
        book["day"] = int(book.get("day") or 0) + 1
        day = int(book["day"])

        # Age every entry
        for entry in book["entries"].values():
            if entry.get("state") == STATE_SIGNED:
                continue
            entry["days_on_market"] = int(entry.get("days_on_market") or 0) + 1
            # Clear expired evaluate window into re-check
            until = entry.get("evaluate_until_day")
            if until is not None and day > int(until) and entry.get("state") == STATE_EVALUATING:
                entry["evaluate_until_day"] = None

        # Opening day: market opens with broad offer volume; signings ramp after day 1.
        # Raised across the board (with the per_team_cap bump above) so all 31 CPU
        # clubs get real reps in the market instead of a handful dominating it.
        if opening_day and day <= 1:
            offers_cap = max_offers_per_day if max_offers_per_day is not None else 70
            signs_cap = max_signings_per_day if max_signings_per_day is not None else 8
        elif day <= 3:
            offers_cap = max_offers_per_day if max_offers_per_day is not None else 90
            signs_cap = max_signings_per_day if max_signings_per_day is not None else 16
        elif day <= 7:
            offers_cap = max_offers_per_day if max_offers_per_day is not None else 80
            signs_cap = max_signings_per_day if max_signings_per_day is not None else 14
        elif day <= 14:
            offers_cap = max_offers_per_day if max_offers_per_day is not None else 64
            signs_cap = max_signings_per_day if max_signings_per_day is not None else 12
        else:
            offers_cap = max_offers_per_day if max_offers_per_day is not None else 50
            signs_cap = max_signings_per_day if max_signings_per_day is not None else 10

        new_offers = _collect_cpu_offers(session, max_new_offers=offers_cap)
        all_offers.extend(new_offers)

        for entry in book["entries"].values():
            _update_player_decision(entry, book, day=day)

        signed = _try_sign_leaning_players(session, max_signings=signs_cap)
        all_signings.extend(signed)

        # Decision counts for UI
        counts: Dict[str, int] = {}
        for e in book["entries"].values():
            st = str(e.get("state") or "")
            counts[st] = counts.get(st, 0) + 1
        day_summaries.append({
            "day": day,
            "offers": len(new_offers),
            "signings": len(signed),
            "states": counts,
        })

        log = list(book.get("log") or [])
        log.append({
            "day": day,
            "offers": len(new_offers),
            "signings": [
                {"player_id": s["player_id"], "team_id": s["team_id"], "aav_m": s["aav_m"]}
                for s in signed
            ],
        })
        book["log"] = log[-60:]

    session.fa_market_book = book
    session.fa_market_day = int(book.get("day") or 0)

    # High-severity economic anomaly: elite UFAs still unsigned late with no serious board.
    anomalies: List[Dict[str, Any]] = list(getattr(session, "economy_anomalies", None) or [])
    day_now = int(book.get("day") or 0)
    if day_now >= 28:
        for e in (book.get("entries") or {}).values():
            if e.get("state") == STATE_SIGNED:
                continue
            if float(e.get("overall") or 0) < 88:
                continue
            if int(e.get("offer_count") or 0) > 0 and float(e.get("best_offer_m") or 0) >= float(e.get("ask_aav_m") or 0) * 0.75:
                continue
            anomalies.append({
                "severity": "high",
                "code": "elite_ufa_unsigned_late",
                "day": day_now,
                "player_id": e.get("player_id"),
                "name": e.get("name"),
                "overall": e.get("overall"),
                "ask_aav_m": e.get("ask_aav_m"),
                "best_offer_m": e.get("best_offer_m"),
                "offer_count": e.get("offer_count"),
                "state": e.get("state"),
                "reason": e.get("reason"),
            })
        # Dedupe by player_id keeping latest
        by_pid = {str(a.get("player_id")): a for a in anomalies if a.get("code") == "elite_ufa_unsigned_late"}
        other = [a for a in anomalies if a.get("code") != "elite_ufa_unsigned_late"]
        session.economy_anomalies = (other + list(by_pid.values()))[-40:]

    # Merge into cpu_fa_signings history
    prev = dict(getattr(session, "cpu_fa_signings", None) or {})
    prev_list = list(prev.get("signings") or [])
    prev_list.extend(all_signings)
    prev["signings"] = prev_list
    prev["count"] = len(prev_list)
    session.cpu_fa_signings = prev
    session.cpu_fa_wave = int(getattr(session, "cpu_fa_wave", 0) or 0) + days

    return {
        "ok": True,
        "days_advanced": days,
        "day": int(book.get("day") or 0),
        "signings": all_signings,
        "offers": all_offers,
        "day_summaries": day_summaries,
        "decision_snapshot": _decision_snapshot(book),
        "economy_anomalies": list(getattr(session, "economy_anomalies", None) or [])[-8:],
    }


def _decision_snapshot(book: Dict[str, Any]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    samples: Dict[str, List[Dict[str, Any]]] = {}
    for e in (book.get("entries") or {}).values():
        st = str(e.get("state") or "")
        counts[st] = counts.get(st, 0) + 1
        if st == STATE_SIGNED:
            continue
        bucket = samples.setdefault(st, [])
        if len(bucket) < 4:
            bucket.append({
                "player_id": e.get("player_id"),
                "name": e.get("name"),
                "reason": e.get("reason"),
                "ask_aav_m": e.get("ask_aav_m"),
                "best_offer_m": e.get("best_offer_m"),
                "offer_count": e.get("offer_count"),
                "days_on_market": e.get("days_on_market"),
                "urgency": e.get("urgency"),
            })
    return {"counts": counts, "samples": samples}


def record_user_fa_offer(
    session: Any,
    *,
    player_id: str,
    aav_m: float,
    years: int,
    ntc: bool = False,
    nmc: bool = False,
    status: str = "pending",
) -> None:
    """Stamp the user's FA offer into the living market book (for board + wire)."""
    book = ensure_fa_market_book(session)
    entries = book.get("entries") or {}
    pid = str(player_id or "")
    entry = entries.get(pid)
    if not isinstance(entry, dict):
        return
    user_tid = str(getattr(session, "user_team_id", "") or "")
    user_team = (getattr(session, "team_by_id", None) or {}).get(user_tid)
    team_name = str(
        getattr(user_team, "name", None) or getattr(user_team, "city", None) or "Your club"
    )
    team_abbrev = str(
        getattr(user_team, "abbreviation", None)
        or getattr(user_team, "abbrev", None)
        or "YOU"
    ).upper()
    day = int(book.get("day") or getattr(session, "fa_market_day", 0) or 0)
    offer = {
        "team_id": user_tid,
        "team_name": team_name,
        "team_abbrev": team_abbrev,
        "aav_m": round(float(aav_m or 0), 3),
        "years": max(1, int(years or 1)),
        "day": day,
        "ntc": bool(ntc),
        "nmc": bool(nmc),
        "is_user": True,
    }
    offers = [o for o in list(entry.get("offers") or []) if str(o.get("team_id")) != user_tid]
    offers.append(offer)
    entry["offers"] = offers[-12:]
    entry["offer_count"] = len(entry["offers"])
    entry["best_offer_m"] = round(
        max(float(entry.get("best_offer_m") or 0), float(aav_m or 0)), 3
    )
    if status == "accepted":
        entry["state"] = STATE_SIGNED
        entry["reason"] = "Signed with your club"
    else:
        entry["state"] = STATE_EVALUATING
        entry["reason"] = "Evaluating your offer"
        entry["evaluate_until_day"] = day + max(1, int(entry.get("patience", 0.5) * 3) or 1)
    log = list(book.get("log") or [])
    if status == "accepted":
        log.append({
            "kind": "signing",
            "text": (
                f"{team_name} signs {entry.get('name') or pid} · "
                f"{offer['aav_m']}M × {offer['years']}y"
            ),
        })
    else:
        log.append({
            "kind": "offer",
            "text": (
                f"{team_name} tables {offer['aav_m']}M × {offer['years']}y "
                f"for {entry.get('name') or pid} — decision pending"
            ),
        })
    book["log"] = log[-40:]
    book["entries"] = entries
    session.fa_market_book = book


def mark_fa_player_signed(session: Any, player_id: str) -> None:
    book = getattr(session, "fa_market_book", None)
    if not isinstance(book, dict):
        return
    entries = book.get("entries") or {}
    pid = str(player_id or "")
    entry = entries.get(pid)
    if isinstance(entry, dict):
        entry["state"] = STATE_SIGNED
        entry["reason"] = "Signed"
    # Drop from live book next ensure — keep signed marker until pool drops them
    book["entries"] = entries
    session.fa_market_book = book


def annotate_fa_rows_with_decisions(session: Any, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stamp decision state onto FA board rows for the UI."""
    book = getattr(session, "fa_market_book", None) or {}
    entries = book.get("entries") or {}
    user_tid = str(getattr(session, "user_team_id", "") or "")
    # Overlay pending user negotiations so the board matches the signing desk.
    neg_map = getattr(session, "resign_negotiations", None) or {}
    if isinstance(neg_map, dict):
        for pid, neg in neg_map.items():
            if not isinstance(neg, dict):
                continue
            pending = neg.get("pending_offer")
            if not isinstance(pending, dict):
                continue
            ctx = str(pending.get("context") or "").lower()
            if ctx and ctx not in ("ufa", "free_agency", "fa"):
                continue
            entry = entries.get(str(pid))
            if not isinstance(entry, dict):
                continue
            if entry.get("state") == STATE_SIGNED:
                continue
            entry["state"] = STATE_EVALUATING
            entry["reason"] = "Evaluating your offer"
            # Ensure a user offer chip exists for logos / competing board.
            offers = list(entry.get("offers") or [])
            if not any(str(o.get("team_id")) == user_tid for o in offers):
                user_team = (getattr(session, "team_by_id", None) or {}).get(user_tid)
                offers.append({
                    "team_id": user_tid,
                    "team_name": str(getattr(user_team, "name", None) or "Your club"),
                    "team_abbrev": str(
                        getattr(user_team, "abbreviation", None)
                        or getattr(user_team, "abbrev", None)
                        or "YOU"
                    ).upper(),
                    "aav_m": pending.get("aav_m"),
                    "years": pending.get("years"),
                    "day": int(book.get("day") or 0),
                    "ntc": bool(pending.get("ntc")),
                    "nmc": bool(pending.get("nmc")),
                    "is_user": True,
                })
                entry["offers"] = offers
                entry["offer_count"] = len(offers)
            entries[str(pid)] = entry
        book["entries"] = entries
        session.fa_market_book = book
    team_by_id = getattr(session, "team_by_id", None) or {}
    user_team = team_by_id.get(user_tid)
    user_ctx = None
    if user_team is not None:
        try:
            user_ctx = evaluate_team_position_needs(
                user_team,
                getattr(getattr(session, "sim", None), "league", None),
                getattr(session, "sim", None),
                season_year=int(getattr(session, "season_calendar_year", 2025) or 2025),
            )
        except Exception:
            user_ctx = None
    out = []
    for row in rows:
        r = dict(row)
        pid = str(r.get("player_id") or r.get("id") or r.get("playerId") or "")
        if pid and not r.get("player_id"):
            r["player_id"] = pid
        e = entries.get(pid)
        if e:
            offers_raw = list(e.get("offers") or [])
            # Latest offer per club, sorted by AAV desc.
            by_team: Dict[str, Dict[str, Any]] = {}
            for o in offers_raw:
                tid = str(o.get("team_id") or "")
                if not tid:
                    continue
                prev = by_team.get(tid)
                if prev is None or float(o.get("aav_m") or 0) >= float(prev.get("aav_m") or 0):
                    team_obj = team_by_id.get(tid)
                    abbrev = str(
                        o.get("team_abbrev")
                        or (getattr(team_obj, "abbreviation", None) if team_obj is not None else None)
                        or (getattr(team_obj, "abbrev", None) if team_obj is not None else None)
                        or tid[:3]
                    ).upper()
                    name = str(
                        o.get("team_name")
                        or (getattr(team_obj, "name", None) if team_obj is not None else None)
                        or tid
                    )
                    by_team[tid] = {
                        "team_id": tid,
                        "team_abbrev": abbrev,
                        "team_name": name,
                        "aav_m": o.get("aav_m"),
                        "years": o.get("years"),
                        "day": o.get("day"),
                        "ntc": bool(o.get("ntc")),
                        "nmc": bool(o.get("nmc")),
                        "is_user": tid == user_tid,
                    }
            competing = sorted(
                by_team.values(),
                key=lambda x: -float(x.get("aav_m") or 0),
            )
            r["decision_state"] = e.get("state")
            r["decision_reason"] = e.get("reason")
            r["market_offers"] = int(e.get("offer_count") or 0)
            r["best_offer_m"] = e.get("best_offer_m")
            r["ask_aav_m"] = e.get("ask_aav_m") or r.get("ask_aav_m") or r.get("asking_aav") or r.get("askingAav")
            r["asking_aav_m"] = r.get("ask_aav_m")
            r["min_acceptable_aav_m"] = e.get("fair_aav_m")
            r["ideal_aav_m"] = e.get("ask_aav_m")
            r["days_on_market"] = e.get("days_on_market")
            r["market_urgency"] = e.get("urgency")
            r["patience"] = e.get("patience")
            r["market_interest"] = (
                "high" if int(e.get("offer_count") or 0) >= 3
                else ("medium" if int(e.get("offer_count") or 0) >= 1 else "low")
            )
            r["competing_clubs"] = len(competing)
            r["competing_offers"] = competing[:8]
            r["interested_teams"] = [
                {"team_id": c["team_id"], "team_abbrev": c["team_abbrev"], "team_name": c["team_name"]}
                for c in competing[:6]
            ]
            ask = float(r.get("ask_aav_m") or r.get("askingAav") or 1)
            ovr = float(r.get("overall") or r.get("ovr") or 70)
            age = int(r.get("age") or 28)
            user_interest = 42.0 + min(28.0, (ovr - 70) * 1.1)
            if age >= 33:
                user_interest += 8.0
            pref = str(r.get("position") or "").upper()
            if user_ctx and pref:
                need = float((user_ctx.get("need_score") or {}).get(pref, 0) or 0)
                user_interest += need * 18.0
            user_offer = next((c for c in competing if c.get("is_user")), None)
            if user_offer:
                ratio = float(user_offer.get("aav_m") or 0) / max(ask, 0.25)
                user_interest += max(-12.0, min(18.0, (ratio - 0.9) * 40.0))
                r["your_offer_rank"] = next(
                    (i + 1 for i, c in enumerate(competing) if c.get("is_user")), None
                )
            r["interest_to_user"] = round(max(5.0, min(97.0, user_interest)), 1)
            r["interest_to_user_label"] = (
                "High" if user_interest >= 70 else ("Low" if user_interest < 45 else "Medium")
            )
            r["sign_likelihood"] = round(
                max(0.05, min(0.95, float(e.get("urgency") or 0.1) * 0.55 + (user_interest / 100.0) * 0.45)),
                2,
            )
        # Former club identity for logos.
        prev = r.get("previous_team") or r.get("current_team") or r.get("last_team")
        if prev and not r.get("previous_team"):
            r["previous_team"] = prev
        if not r.get("previous_team_abbrev"):
            token = str(prev or "")
            if len(token) <= 4 and token.isalpha():
                r["previous_team_abbrev"] = token.upper()
            else:
                # Try match against session teams by name.
                matched = None
                for tid, tm in team_by_id.items():
                    name = str(getattr(tm, "name", "") or "")
                    if name and name.lower() == token.lower():
                        matched = tm
                        break
                if matched is not None:
                    r["previous_team_abbrev"] = str(
                        getattr(matched, "abbreviation", None)
                        or getattr(matched, "abbrev", None)
                        or ""
                    ).upper()
                    r["previous_team_id"] = str(
                        getattr(matched, "team_id", None) or getattr(matched, "id", "") or ""
                    )
        out.append(r)
    return out
