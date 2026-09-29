"""
Trade finder — builds offers around one user-chosen asset and keeps only the
ones the partner GM would actually accept.

Two directions:
  * ``sell``: the user shops their own player/pick; each CPU club returns the
    best package it can afford to give back.
  * ``buy``:  the user targets a CPU player/pick; we assemble the cheapest
    packages from the user's organization that clear the partner's price.

Candidates are sized with the same trade-value function the evaluator uses, then
every surviving candidate is run through ``evaluate_trade_package`` so a result
here means "this exact offer is accepted right now", not a heuristic guess.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from services.franchise_paths import ensure_simengine_path

ensure_simengine_path()

from app.sim_engine.trades.trade_evaluator import evaluate_trade_package  # noqa: E402
from app.sim_engine.trades.trade_pick_registry import serialize_team_picks  # noqa: E402
from app.sim_engine.trades.trade_value import (  # noqa: E402
    evaluate_pick_asset_value,
    evaluate_player_asset_value,
)

MAX_POOL = 22  # per side, by value — keeps combo enumeration cheap
MAX_COMBO_SIZE = 4
MAX_EVALS = 48  # full evaluator runs per request
SELL_MARGIN = 0.94  # return <= 94% of what the partner receives (they need a win)
SELL_VALUE_FLOOR = 0.62  # and >= 62% — anything lighter is a lowball the user wouldn't take
BUY_PREMIUM = 1.04  # offer >= 104% of the target's price
NHL_ROSTER_MAX = 23


def _nhl_room(team: Any, league: Any = None) -> int:
    """Open active-roster spots, counted the way trade validation counts them (IR/LTIR
    excluded). len(roster) disagreed with the rules, so packages that 'fit' here were
    rejected with 'Trade would exceed active roster maximum'."""
    try:
        from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot

        active = int(calculate_team_cap_snapshot(team, league=league).get("activeRosterCount"))
        return NHL_ROSTER_MAX - active
    except Exception:
        return NHL_ROSTER_MAX - len(list(getattr(team, "roster", None) or []))


def _nhl_count(assets) -> int:
    return sum(1 for a in assets if a.get("type") == "player" and a.get("level") == "NHL")


def _roster_fits(user_room: int, partner_room: int, user_gives, partner_gives) -> bool:
    """Both clubs stay at or under the NHL roster maximum after the swap."""
    out_n, in_n = _nhl_count(user_gives), _nhl_count(partner_gives)
    return user_room + out_n - in_n >= 0 and partner_room + in_n - out_n >= 0


def _clause_blocked(player: Any) -> bool:
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        return bool(c.get("no_move_clause") or c.get("nmc") or c.get("no_trade_clause") or c.get("ntc"))
    if c is None:
        return False
    clauses = getattr(c, "clauses", None)
    if clauses is not None:
        return bool(getattr(clauses, "noMoveClause", False) or getattr(clauses, "noTradeClause", False))
    return bool(getattr(c, "no_move_clause", False) or getattr(c, "no_trade_clause", False))


def _player_label(p: Any) -> Dict[str, Any]:
    ident = getattr(p, "identity", None)
    name = str(getattr(ident, "name", None) or getattr(p, "name", "") or "?")
    pos = getattr(ident, "position", None) if ident else getattr(p, "position", "")
    age = getattr(ident, "age", None) if ident else getattr(p, "age", None)
    return {"name": name, "pos": str(getattr(pos, "value", pos) or ""), "age": age}


def _org_players(team: Any) -> List[Tuple[Any, str]]:
    out: List[Tuple[Any, str]] = []
    for attr, level in (("roster", "NHL"), ("ahl_roster", "AHL")):
        for p in list(getattr(team, attr, None) or []):
            out.append((p, level))
    return out


def _find_player(team: Any, player_id: str) -> Optional[Tuple[Any, str]]:
    for p, level in _org_players(team):
        if str(getattr(p, "id", "") or "") == str(player_id):
            return p, level
    return None


def _team_picks(league: Any, team_id: str, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    min_year = int(ctx.get("tradeable_draft_year") or ctx.get("draft_year") or ctx.get("season_year") or 0)
    try:
        return serialize_team_picks(league, str(team_id), min_year=min_year or None)
    except Exception:
        return []


def _value_player(p: Any, source: Any, acquiring: Any, league: Any, ctx: Dict[str, Any]) -> float:
    try:
        return float(evaluate_player_asset_value(p, source, acquiring, league, context=ctx).get("total") or 0.0)
    except Exception:
        return 0.0


def _value_pick(row: Dict[str, Any], source: Any, acquiring: Any, league: Any, ctx: Dict[str, Any]) -> float:
    try:
        return float(evaluate_pick_asset_value(row, acquiring, source, league, context=ctx).get("total") or 0.0)
    except Exception:
        return 0.0


def _pool(
    team: Any,
    team_id: str,
    acquiring: Any,
    league: Any,
    ctx: Dict[str, Any],
    *,
    exclude: set,
    include_ahl: bool = True,
    max_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Tradeable assets of ``team`` valued as ``acquiring`` would see them.

    ``max_value`` drops assets that could never fit the deal, so the top-N cut
    keeps usable pieces instead of stars far above the budget.
    """
    items: List[Dict[str, Any]] = []
    for p, level in _org_players(team):
        pid = str(getattr(p, "id", "") or "")
        if not pid or pid in exclude or _clause_blocked(p):
            continue
        if level == "AHL" and not include_ahl:
            continue
        v = _value_player(p, team, acquiring, league, ctx)
        if v <= 1.0:
            continue
        label = _player_label(p)
        items.append({
            "type": "player",
            "id": pid,
            "team": team_id,
            "value": round(v, 2),
            "level": level,
            **label,
        })
    for row in _team_picks(league, team_id, ctx):
        pick_id = str(row.get("pick_id") or "")
        if not pick_id or pick_id in exclude:
            continue
        v = _value_pick(row, team, acquiring, league, ctx)
        if v <= 1.0:
            continue
        items.append({
            "type": "pick",
            "id": pick_id,
            "team": team_id,
            "value": round(v, 2),
            "name": str(row.get("display") or pick_id),
            "year": row.get("year"),
            "round": row.get("round"),
            "original_team_id": row.get("original_team_id"),
        })
    if max_value is not None:
        items = [a for a in items if a["value"] <= max_value]
    items.sort(key=lambda a: a["value"], reverse=True)
    return items[:MAX_POOL]


def _combos(pool: List[Dict[str, Any]], max_size: int):
    cap = min(max_size, MAX_COMBO_SIZE)
    for size in range(1, cap + 1):
        for combo in combinations(pool, size):
            yield combo


def _combo_shape_score(combo: List[Dict[str, Any]], *, anchor_type: str) -> int:
    """Prefer mixed packages (players + picks) over repetitive pick-only swaps."""
    n_players = sum(1 for a in combo if a.get("type") == "player")
    n_picks = sum(1 for a in combo if a.get("type") == "pick")
    score = n_players * 12 + n_picks * 5 + len(combo) * 2
    if n_players >= 1 and n_picks >= 1:
        score += 28
    if n_players >= 2:
        score += 16
    if n_picks >= 2:
        score += 8
    if anchor_type == "pick":
        if n_players >= 1:
            score += 40
        else:
            score -= 35
    if anchor_type == "player" and n_players == 0 and n_picks >= 2:
        score -= 12
    return score


def _payload_asset(a: Dict[str, Any]) -> Dict[str, Any]:
    out = {"type": a["type"], "id": a["id"], "team": a["team"]}
    if a["type"] == "player":
        out["retained"] = 0
    return out


def _public_asset(a: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("type", "id", "team", "value", "name", "pos", "age", "level", "year", "round", "original_team_id")
    return {k: a[k] for k in keep if k in a}


def _evaluate(
    user_tid: str,
    partner_tid: str,
    user_gives: List[Dict[str, Any]],
    partner_gives: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    assets_by_team = {
        partner_tid: [_payload_asset(a) for a in user_gives],
        user_tid: [_payload_asset(a) for a in partner_gives],
    }
    try:
        result = evaluate_trade_package(
            assets_by_team,
            league=ctx["league"],
            team_by_id=ctx["team_by_id"],
            context=ctx,
            user_team_id=user_tid,
        )
    except Exception as exc:  # malformed / stale asset — treat as rejected
        return {"accepted": False, "rejection_reasons": [str(exc)]}
    return result


def _offer_row(
    partner: Any,
    partner_tid: str,
    user_gives: List[Dict[str, Any]],
    partner_gives: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    interest = (result.get("interest_level") or {}).get(partner_tid)
    return {
        "partner_team_id": partner_tid,
        "partner_name": str(getattr(partner, "name", "") or partner_tid),
        "user_gives": [_public_asset(a) for a in user_gives],
        "user_gets": [_public_asset(a) for a in partner_gives],
        "user_gives_value": round(sum(a["value"] for a in user_gives), 1),
        "user_gets_value": round(sum(a["value"] for a in partner_gives), 1),
        "interest": round(float(interest), 3) if interest is not None else None,
        "verdict": result.get("verdict"),
        "fairness_gap": result.get("fairness_gap"),
    }


def _sell_candidates(
    offered_value: float,
    pool: List[Dict[str, Any]],
    fits,
    *,
    anchor_type: str = "player",
) -> List[List[Dict[str, Any]]]:
    """Diverse return packages under the partner budget — not only pick-for-pick.

    Packages must carry real value (>= SELL_VALUE_FLOOR of the asset). Ranking used to be
    shape-first, so a star drew four-piece piles of depth worth a fraction of him; the
    partner's evaluator rejected every one and only cheap assets ever got offers.
    """
    budget = offered_value * SELL_MARGIN
    floor = offered_value * SELL_VALUE_FLOOR
    ranked: List[Tuple[float, float, List[Dict[str, Any]]]] = []
    for combo in _combos(pool, MAX_COMBO_SIZE):
        total = sum(a["value"] for a in combo)
        if total > budget or total < floor or not fits(combo):
            continue
        shape = _combo_shape_score(combo, anchor_type=anchor_type)
        # Value closeness dominates; shape only breaks ties between similar totals.
        key = 100.0 * (total / max(1.0, budget)) + shape * 0.25
        ranked.append((key, total, list(combo)))
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out: List[List[Dict[str, Any]]] = []
    seen: set = set()
    for shape, _total, combo in ranked:
        if anchor_type == "pick" and not any(a.get("type") == "player" for a in combo):
            continue
        key = tuple(sorted((a.get("type"), a.get("id")) for a in combo))
        if key in seen:
            continue
        seen.add(key)
        out.append(combo)
        if len(out) >= 5:
            break
    if not out and anchor_type != "pick":
        for _shape, _total, combo in ranked[:3]:
            key = tuple(sorted((a.get("type"), a.get("id")) for a in combo))
            if key not in seen:
                out.append(combo)
                seen.add(key)
    return out


def _buy_candidates(price: float, pool: List[Dict[str, Any]], fits) -> List[List[Dict[str, Any]]]:
    """Cheapest user packages that clear the price: a player-led, a pick-led and the overall cheapest."""
    need = price * BUY_PREMIUM
    cheapest: Optional[Tuple[float, List[Dict[str, Any]]]] = None
    picks_only: Optional[Tuple[float, List[Dict[str, Any]]]] = None
    player_led: Optional[Tuple[float, List[Dict[str, Any]]]] = None
    for combo in _combos(pool, MAX_COMBO_SIZE):
        total = sum(a["value"] for a in combo)
        if total < need or not fits(combo):
            continue
        entry = (total, list(combo))
        if cheapest is None or total < cheapest[0]:
            cheapest = entry
        if all(a["type"] == "pick" for a in combo) and (picks_only is None or total < picks_only[0]):
            picks_only = entry
        if any(a["type"] == "player" for a in combo) and (player_led is None or total < player_led[0]):
            player_led = entry
    out: List[List[Dict[str, Any]]] = []
    seen = set()
    for entry in (cheapest, picks_only, player_led):
        if entry is None:
            continue
        key = tuple(sorted(a["id"] for a in entry[1]))
        if key in seen:
            continue
        seen.add(key)
        out.append(entry[1])
    return out


def find_trade_offers(
    session: Any,
    *,
    asset_type: str,
    asset_id: str,
    mode: str = "sell",
    target_team_id: Optional[str] = None,
    exclude_ids: Optional[List[str]] = None,
    limit: int = 8,
) -> Dict[str, Any]:
    """Return accepted offers built around one asset. See module docstring."""
    from services.trade_service import _ensure_trade_infrastructure, _trade_context

    _ensure_trade_infrastructure(session)
    ctx = _trade_context(session)
    league = ctx["league"]
    team_by_id = ctx["team_by_id"] or {}
    user_tid = str(ctx["user_team_id"])
    user_team = team_by_id.get(user_tid)
    if league is None or user_team is None:
        raise ValueError("Franchise league is not ready")
    if ctx.get("trade_deadline_passed"):
        return {"mode": mode, "offers": [], "note": "The trade deadline has passed."}

    asset_type = str(asset_type or "").lower()
    asset_id = str(asset_id or "")
    exclude = {str(x) for x in (exclude_ids or [])} | {asset_id}
    mode = "buy" if str(mode).lower() == "buy" else "sell"

    if mode == "sell":
        source_team, source_tid = user_team, user_tid
        partners = [
            (tid, t) for tid, t in team_by_id.items()
            if str(tid) != user_tid and (not target_team_id or str(tid) == str(target_team_id))
        ]
    else:
        source_tid = str(target_team_id or "")
        source_team = team_by_id.get(source_tid)
        if source_team is None:
            raise ValueError("Target team not found")
        partners = [(source_tid, source_team)]

    # Resolve the anchor asset.
    anchor_player = None
    anchor_pick = None
    if asset_type == "player":
        found = _find_player(source_team, asset_id)
        if not found:
            raise ValueError("Player not found in that organization")
        anchor_player = found[0]
        anchor_label = {**_player_label(anchor_player), "level": found[1]}
    elif asset_type == "pick":
        anchor_pick = next((r for r in _team_picks(league, source_tid, ctx) if str(r.get("pick_id")) == asset_id), None)
        if anchor_pick is None:
            raise ValueError("Pick not owned by that team")
        anchor_label = {
            "name": str(anchor_pick.get("display") or asset_id),
            "year": anchor_pick.get("year"),
            "round": anchor_pick.get("round"),
            "original_team_id": anchor_pick.get("original_team_id"),
        }
    else:
        raise ValueError("asset_type must be 'player' or 'pick'")

    def anchor_value(acquiring: Any) -> float:
        if anchor_player is not None:
            return _value_player(anchor_player, source_team, acquiring, league, ctx)
        return _value_pick(anchor_pick, source_team, acquiring, league, ctx)

    anchor_base = {"type": asset_type, "id": asset_id, "team": source_tid, **anchor_label}
    offers: List[Dict[str, Any]] = []
    near: List[Dict[str, Any]] = []
    evals = 0

    if mode == "sell":
        scored: List[Tuple[float, str, Any, float, List[List[Dict[str, Any]]]]] = []
        for tid, partner in partners:
            tid = str(tid)
            offered = anchor_value(partner)
            if offered <= 2.0:
                continue
            pool = _pool(partner, tid, user_team, league, ctx, exclude=exclude, max_value=offered * SELL_MARGIN)
            user_room, partner_room = _nhl_room(user_team, league), _nhl_room(partner, league)
            cands = _sell_candidates(
                offered,
                pool,
                lambda combo, ur=user_room, pr=partner_room: _roster_fits(ur, pr, [anchor_base], combo),
                anchor_type=asset_type,
            )
            if not cands:
                continue
            best_total = max(sum(a["value"] for a in c) for c in cands)
            scored.append((best_total / max(1.0, offered), tid, partner, offered, cands))
        scored.sort(key=lambda s: s[0], reverse=True)
        for _ratio, tid, partner, offered, cands in scored:
            if evals >= MAX_EVALS or len(offers) >= limit:
                break
            give = [{**anchor_base, "value": round(offered, 2)}]
            accepted_for_partner = 0
            for cand in cands:
                if evals >= MAX_EVALS or len(offers) >= limit or accepted_for_partner >= 2:
                    break
                evals += 1
                result = _evaluate(user_tid, tid, give, cand, ctx)
                row = _offer_row(partner, tid, give, cand, result)
                if result.get("accepted"):
                    offers.append(row)
                    accepted_for_partner += 1
                else:
                    near.append({**row, "reasons": [str(r) for r in (result.get("rejection_reasons") or [])][:2]})
    else:
        partner_tid, partner = partners[0]
        price = anchor_value(user_team)
        user_room, partner_room = _nhl_room(user_team, league), _nhl_room(partner, league)

        def fits(combo):
            return _roster_fits(user_room, partner_room, combo, [anchor_base])

        pool = _pool(user_team, user_tid, partner, league, ctx, exclude=exclude, max_value=price * 1.6)
        cands = _buy_candidates(price, pool, fits)
        if not cands:
            # Only a bigger single piece clears the price.
            cands = _buy_candidates(price, _pool(user_team, user_tid, partner, league, ctx, exclude=exclude), fits)
        get = [{**anchor_base, "value": round(price, 2)}]
        for cand in cands:
            if evals >= MAX_EVALS:
                break
            evals += 1
            result = _evaluate(user_tid, partner_tid, cand, get, ctx)
            row = _offer_row(partner, partner_tid, cand, get, result)
            if result.get("accepted"):
                offers.append(row)
            else:
                near.append({**row, "reasons": [str(r) for r in (result.get("rejection_reasons") or [])][:2]})

    offers.sort(key=lambda o: o["user_gets_value"] - o["user_gives_value"], reverse=True)
    near.sort(key=lambda o: float(o.get("interest") or 0.0), reverse=True)
    return {
        "mode": mode,
        "anchor": anchor_base,
        "offers": offers[:limit],
        "near_misses": near[:4] if not offers else [],
        "evaluated": evals,
    }
