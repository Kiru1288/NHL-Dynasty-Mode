"""
Trade asset market intel — who is interested in one asset and how hot its market is.

Powers the Trade Hub "Asset Intel" panel (Trade Value headline, interested teams,
demand level). Every club's number is the same trade-value function the evaluator
uses (``evaluate_player_asset_value`` / ``evaluate_pick_asset_value`` as that club
would see the asset), plus the buyer-fit premium and a cap/roster feasibility read,
so "interested" here lines up with what a real proposal would be judged on.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, List, Optional

from services.franchise_paths import ensure_simengine_path

ensure_simengine_path()

import logging as _logging

_log = _logging.getLogger(__name__)

_WINDOW_LABEL = {
    "contender": "Contender",
    "rebuild": "Rebuilding",
    "declining": "Retooling",
    "emerging": "On the rise",
}


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _demand_label(n_interested: int, n_hot: int) -> str:
    if n_hot >= 4 or n_interested >= 10:
        return "Red Hot"
    if n_hot >= 2 or n_interested >= 6:
        return "High"
    if n_interested >= 3:
        return "Moderate"
    if n_interested >= 1:
        return "Low"
    return "Cold"


def asset_market_intel(
    session: Any,
    *,
    asset_type: str,
    asset_id: str,
    team_id: str,
    limit: int = 8,
) -> Dict[str, Any]:
    from services.trade_service import _ensure_trade_infrastructure
    from services import trade_finder as tf

    ctx = _ensure_trade_infrastructure(session)
    league = ctx["league"]
    team_by_id = ctx["team_by_id"] or {}
    user_tid = str(ctx["user_team_id"])
    source_tid = str(team_id or "")
    source = team_by_id.get(source_tid)
    if league is None or source is None:
        raise ValueError("Team not found")

    asset_type = str(asset_type or "").lower()
    player = None
    pick = None
    cap_m = 0.0
    if asset_type == "player":
        found = tf._find_player(source, str(asset_id))
        if not found:
            raise ValueError("Player not found in that organization")
        player = found[0]
        cap_m = float(tf._trade_cap_hit(player, ctx) or 0.0)
    elif asset_type == "pick":
        pick = next((r for r in tf._team_picks(league, source_tid, ctx) if str(r.get("pick_id")) == str(asset_id)), None)
        if pick is None:
            raise ValueError("Pick not owned by that team")
    else:
        raise ValueError("asset_type must be 'player' or 'pick'")

    try:
        from app.sim_engine.trades.trade_evaluator import _team_window
    except Exception:  # pragma: no cover
        def _team_window(_t):  # type: ignore
            return "unknown"
    try:
        from app.sim_engine.trades.trade_value import _buyer_fit_premium
    except Exception:  # pragma: no cover
        def _buyer_fit_premium(_t, _p):  # type: ignore
            return 0.0

    rows: List[Dict[str, Any]] = []
    for tid, team in team_by_id.items():
        tid = str(tid)
        if tid == source_tid:
            continue
        try:
            if player is not None:
                v = tf._value_player(player, source, team, league, ctx)
                fit = float(_buyer_fit_premium(team, player) or 0.0)
            else:
                v = tf._value_pick(pick, source, team, league, ctx)
                fit = 0.0
        except Exception:
            _log.debug("asset intel valuation failed", exc_info=True)
            continue
        window = str(_team_window(team) or "unknown")
        cap_ok = True
        cap_room = None
        if player is not None and cap_m > 0:
            try:
                budget = tf._cap_budget(team, ctx)
                cap_room = None if budget == float("inf") else round(float(budget), 2)
                cap_ok = budget == float("inf") or cap_m <= budget + 0.01
            except Exception:
                cap_ok = True
        rows.append({
            "team_id": tid,
            "team_name": str(getattr(team, "name", "") or tid),
            "abbr": str(getattr(team, "abbr", None) or getattr(team, "abbreviation", None) or ""),
            "value": round(float(v), 2),
            "fit": round(fit, 3),
            "window": window,
            "window_label": _WINDOW_LABEL.get(window, "Holding"),
            "cap_ok": bool(cap_ok),
            "cap_room_m": cap_room,
            "is_user": tid == user_tid,
        })

    if not rows:
        return {"asset": {"type": asset_type, "id": str(asset_id), "team": source_tid}, "bidders": [], "interested_count": 0, "demand": "Cold"}

    vals = [r["value"] for r in rows]
    market = float(median(vals)) if vals else 0.0
    base = max(1.0, abs(market))
    for r in rows:
        # Interest: how far above the league's median price this club values him,
        # plus a real roster hole, plus direction (contenders chase NHL players,
        # rebuilders chase picks and kids).
        lean = 0.0
        if player is not None:
            age = None
            try:
                ident = getattr(player, "identity", None)
                age = int(getattr(ident, "age", None) or getattr(player, "age", None) or 0) or None
            except Exception:
                age = None
            if r["window"] == "contender":
                lean += 0.35
            elif r["window"] == "rebuild":
                lean += 0.3 if (age is not None and age <= 23) else -0.45
        else:
            lean += 0.35 if r["window"] == "rebuild" else (-0.2 if r["window"] == "contender" else 0.0)
        x = ((r["value"] - market) / base) * 6.0 + r["fit"] * 8.0 + lean - 0.5
        score = _sigmoid(x) * 100.0
        if not r["cap_ok"]:
            score *= 0.3
        if market <= 0 and r["value"] <= 0:
            score = min(score, 12.0)
        r["interest"] = int(round(max(0.0, min(99.0, score))))
        reasons: List[str] = []
        if r["fit"] >= 0.06:
            reasons.append("Fills a hole in the lineup")
        if r["value"] > market * 1.06 and market > 0:
            reasons.append("Values him above market" if player is not None else "Values the pick above market")
        if r["window"] == "contender":
            reasons.append("Buying for a playoff push")
        elif r["window"] == "rebuild":
            reasons.append("Collecting futures")
        if not r["cap_ok"]:
            reasons.append("No cap room without salary going back")
        r["reasons"] = reasons[:2]

    rows.sort(key=lambda r: (r["interest"], r["value"]), reverse=True)
    others = [r for r in rows if not r["is_user"]]
    interested = [r for r in others if r["interest"] >= 55 and r["cap_ok"]]
    hot = [r for r in interested if r["interest"] >= 75]
    top = max((r["value"] for r in rows), default=0.0)
    return {
        "asset": {"type": asset_type, "id": str(asset_id), "team": source_tid, "cap_m": round(cap_m, 3)},
        "market_value": round(market, 2),
        "top_bid_value": round(top, 2),
        "interested_count": len(interested),
        "hot_count": len(hot),
        "teams_checked": len(rows),
        "demand": _demand_label(len(interested), len(hot)),
        "bidders": others[: max(1, int(limit))],
    }


def find_counter_offers(
    session: Any,
    *,
    assets_by_team: Dict[str, List[Dict[str, Any]]],
    max_evals: int = 10,
    limit: int = 3,
) -> Dict[str, Any]:
    """Greedy "what would make it work" search for a rejected Trade Hub package.

    Tries adding one (then two) of the user's assets, cheapest that could close the
    partner's value gap first, and dropping one piece from the user's ask — every
    candidate is run through the real evaluator, so a suggestion here is a package
    the partner GM accepts right now. Does not count against GM patience.
    """
    from services.trade_service import _ensure_trade_infrastructure
    from services import trade_finder as tf

    ctx = _ensure_trade_infrastructure(session)
    league = ctx["league"]
    tb = ctx["team_by_id"] or {}
    user_tid = str(ctx["user_team_id"])
    keys = [str(k) for k in (assets_by_team or {}).keys()]
    partner_tid = next((k for k in keys if k != user_tid), None)
    if not partner_tid or partner_tid not in tb or user_tid not in tb:
        raise ValueError("Package needs a partner club")
    partner = tb[partner_tid]
    user_team = tb[user_tid]

    def norm(rows):
        out = []
        for a in rows or []:
            t = str(a.get("type") or "").lower()
            if t not in ("player", "pick"):
                continue
            out.append({"type": t, "id": str(a.get("id")), "team": str(a.get("team")), "retained": int(a.get("retained") or 0)})
        return out

    user_gives = norm((assets_by_team or {}).get(partner_tid))
    partner_gives = norm((assets_by_team or {}).get(user_tid))
    tf._SLOT_CACHE.clear()
    evals = 0
    base = tf._evaluate(user_tid, partner_tid, user_gives, partner_gives, ctx)
    evals += 1
    interest0 = (base.get("interest_level") or {}).get(partner_tid)
    if base.get("accepted"):
        return {"already_works": True, "suggestions": [], "base_interest": interest0}
    reasons = tf._clean_reasons(base, tb)
    if base.get("_illegal"):
        return {"already_works": False, "illegal": True, "reasons": reasons, "suggestions": [], "base_interest": interest0}

    p_net = float(((base.get("value_breakdown") or {}).get(partner_tid) or {}).get("net", 0.0) or 0.0)
    gap = max(1.0, -p_net)
    exclude = {a["id"] for a in user_gives} | {a["id"] for a in partner_gives}
    pool = tf._pool(user_team, user_tid, partner, league, ctx, exclude=exclude)
    suggestions: List[Dict[str, Any]] = []
    seen: set = set()

    def try_pkg(kind: str, gives, gets, added=None, dropped=None) -> None:
        nonlocal evals
        if evals >= max_evals or len(suggestions) >= limit:
            return
        key = (tuple(sorted(a["id"] for a in gives)), tuple(sorted(a["id"] for a in gets)))
        if key in seen:
            return
        seen.add(key)
        evals += 1
        res = tf._evaluate(user_tid, partner_tid, gives, gets, ctx)
        if not res.get("accepted"):
            return
        interest = (res.get("interest_level") or {}).get(partner_tid)
        suggestions.append({
            "kind": kind,
            "add": [tf._public_asset(a) for a in (added or [])],
            "drop": [tf._public_asset(a) for a in (dropped or [])],
            "interest": round(float(interest), 3) if interest is not None else None,
            "value_added": round(sum(float(a.get("value") or 0.0) for a in (added or [])), 1),
        })

    # 1) Ask for less: drop one piece of the user's ask (cheapest first) when the ask has 2+.
    if len(partner_gives) >= 2:
        valued = []
        for a in partner_gives:
            src = tb.get(a["team"])
            if a["type"] == "player":
                found = tf._find_player(src, a["id"]) if src is not None else None
                v = tf._value_player(found[0], src, user_team, league, ctx) if found else 0.0
                lbl = tf._player_label(found[0]) if found else {"name": a["id"]}
            else:
                row = next((r for r in tf._team_picks(league, a["team"], ctx) if str(r.get("pick_id")) == a["id"]), None)
                v = tf._value_pick(row, src, user_team, league, ctx) if row else 0.0
                lbl = {"name": str((row or {}).get("display") or a["id"]), "year": (row or {}).get("year"), "round": (row or {}).get("round")}
            valued.append(({**a, **lbl, "value": round(v, 2)}, v))
        valued.sort(key=lambda t: t[1])
        for a, _v in valued[:2]:
            gets = [x for x in partner_gives if x["id"] != a["id"]]
            try_pkg("drop", user_gives, gets, dropped=[a])

    # 2) Add one of the user's assets — cheapest that can plausibly close the gap.
    singles = sorted([a for a in pool if a["value"] >= gap * 0.7], key=lambda a: a["value"])
    for a in singles[:5]:
        try_pkg("add", user_gives + [a], partner_gives, added=[a])

    # 3) Two smaller pieces together.
    if len(suggestions) < limit and evals < max_evals:
        small = sorted([a for a in pool if a["value"] < gap * 0.9], key=lambda a: a["value"], reverse=True)[:6]
        pairs = []
        for i in range(len(small)):
            for j in range(i + 1, len(small)):
                tot = tf._eff([small[i], small[j]])
                if tot >= gap * 0.9:
                    pairs.append((tot, [small[i], small[j]]))
        pairs.sort(key=lambda t: t[0])
        for _tot, combo in pairs[:3]:
            try_pkg("add", user_gives + combo, partner_gives, added=combo)

    return {
        "already_works": False,
        "illegal": False,
        "reasons": reasons,
        "value_gap": round(gap, 1),
        "base_interest": interest0,
        "suggestions": suggestions,
        "evaluated": evals,
    }
