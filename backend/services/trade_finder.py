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
    effective_package_total,
    evaluate_pick_asset_value,
    evaluate_player_asset_value,
)


def _eff(assets) -> float:
    """Package value the evaluator uses (diminishing returns on extra pieces)."""
    return effective_package_total([a["value"] for a in assets])

MAX_POOL = 22  # per side, by value — keeps combo enumeration cheap
MAX_COMBO_SIZE = 4
MAX_EVALS = 48  # full evaluator runs per request
SELL_MARGIN = 0.94  # return <= 94% of what the partner receives (they need a win)
SELL_VALUE_FLOOR = 0.80  # and >= 80% — anything lighter is a lowball the user wouldn't take
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


def _roster_fits(user_room: int, partner_room: int, user_gives, partner_gives, user_flex: int = 0, partner_flex: int = 0) -> bool:
    """Both clubs can get back to the NHL roster maximum after the swap — counting the
    depth players each can send to the AHL as same-day corresponding moves."""
    out_n, in_n = _nhl_count(user_gives), _nhl_count(partner_gives)
    return user_room + user_flex + out_n - in_n >= 0 and partner_room + partner_flex + in_n - out_n >= 0


def _flex(team: Any) -> int:
    try:
        from app.sim_engine.trades.roster_balance import demotion_capacity

        return int(demotion_capacity(team))
    except Exception:
        return 0


def _send_down_names(team: Any, room: int, gives, gets) -> List[str]:
    """Players the club would assign to the AHL to make room for this package."""
    need = _nhl_count(gets) - _nhl_count(gives) - room
    if need <= 0:
        return []
    try:
        from app.sim_engine.trades.roster_balance import send_down_candidates

        leaving = [a["id"] for a in gives if a.get("type") == "player"]
        return [
            str(getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", "") or "?")
            for p in send_down_candidates(team, leaving_ids=leaving, count=need)
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Cap feasibility — mirrors cap_engine.can_trade_cap_fit so packages that the
# rules engine would reject for cap are never sent to the evaluator. Before
# this, every candidate was sized on value + roster spots only, and a capped-out
# partner with a good value ratio burned the whole 48-eval budget on
# "Trade would exceed usable cap space".
# ---------------------------------------------------------------------------

MAX_RETAIN_PCT = 50
_RETENTION_LEAGUE: Dict[str, Any] = {}  # league for the current request (Board retention rules)


def _season_label_from_ctx(ctx: Dict[str, Any]) -> Optional[str]:
    y = ctx.get("season_year")
    try:
        return f"{int(y)}-{(int(y) + 1) % 100:02d}" if y else None
    except Exception:
        return None


def _cap_hit(p: Any) -> float:
    try:
        from app.sim_engine.economy.cap_engine import player_cap_hit_millions

        return float(player_cap_hit_millions(p) or 0.0)
    except Exception:
        return 0.0


def _trade_cap_hit(p: Any, ctx: Dict[str, Any]) -> float:
    """Cap charge the trade engine uses: AHL-assigned players only carry their bury residual."""
    if getattr(p, "in_minors", False) or getattr(p, "is_buried", False) or getattr(p, "buried", False):
        try:
            from app.sim_engine.economy.cap_engine import buried_cap_hit_millions

            return float(buried_cap_hit_millions(p, season_start_year=int(ctx.get("season_year") or 0) or None))
        except Exception:
            return 0.0
    return _cap_hit(p)


def _cap_budget(team: Any, ctx: Dict[str, Any]) -> float:
    """Largest *full-season* net cap increase (incoming − outgoing, $M) this team can
    absorb in a trade right now. Same three paths as can_trade_cap_fit: usable
    space (prorated in-season), in-season accrual before the late deadline, and
    LTIR effective limit."""
    try:
        from app.sim_engine.economy.cap_engine import can_trade_cap_fit

        chk = can_trade_cap_fit(
            team, [], [], league=ctx.get("league"),
            calendar_cursor=int(ctx.get("calendar_cursor", 0) or 0),
            regular_season_last_index=int(ctx.get("regular_season_last_index", 192) or 192),
            deadline_phase=float(ctx.get("deadline_phase", 0.0) or 0.0),
            season_label=_season_label_from_ctx(ctx),
        )
    except Exception:
        return float("inf")  # unknown → let the evaluator decide
    snap = chk.get("snapshot") or {}
    pf = max(0.05, float(chk.get("prorationFactor", 1.0) or 1.0))
    usable = float(snap.get("usableCapSpace", 0.0) or 0.0)
    best = usable / pf
    # Full-season cap hits must also stay under the upper limit (same as the rules engine).
    upper = float(snap.get("upperLimit", 0.0) or 0.0)
    if upper > 0:
        best = min(best, upper - float(snap.get("totalCapHit", 0.0) or 0.0))
    if float(snap.get("ltirPool", 0.0) or 0.0) > 0.001:
        eff = float(snap.get("effectiveCapLimit", snap.get("upperLimit", 0.0)) or 0.0)
        best = max(best, eff - float(snap.get("totalCapHit", 0.0) or 0.0) + 0.02)
    # An over-cap club can still make deals that shed salary (net ≤ 0).
    return max(0.0, best)


def _cap_m(assets) -> float:
    return sum(float(a.get("cap_m") or 0.0) for a in assets if a.get("type") == "player")


def _plan_cap(
    user_budget: float,
    partner_budget: float,
    user_gives: List[Dict[str, Any]],
    partner_gives: List[Dict[str, Any]],
    *,
    retain_on: Optional[Dict[str, Any]] = None,
    retain_allowed: bool = False,
) -> Optional[int]:
    """Return the retention % (0 when none needed) on ``retain_on`` (a user-sent
    player) that makes both sides cap-legal, or None if no legal structure exists."""
    out_u, in_u = _cap_m(user_gives), _cap_m(partner_gives)
    # partner: receives user_gives, sends partner_gives
    partner_net = out_u - in_u
    user_net = in_u - out_u
    if partner_net <= partner_budget + 0.005 and user_net <= user_budget + 0.005:
        return 0
    if not (retain_allowed and retain_on is not None):
        return None
    hit = float(retain_on.get("cap_m") or 0.0)
    if hit <= 0:
        return None
    need = partner_net - partner_budget
    if need <= 0:
        # Partner fits. If the buyer does not, retention has to sit on a player
        # the buyer is acquiring — this helper only prices the seller's outgoing piece.
        return None
    cap_pct = MAX_RETAIN_PCT
    try:
        from app.sim_engine.economy.cap_engine import max_retention_pct

        cap_pct = int(max_retention_pct(_RETENTION_LEAGUE.get("league")))
    except Exception:
        pass
    pct = int(min(cap_pct, ((need / hit) * 100.0 // 5 + 1) * 5))
    if hit * pct / 100.0 + 0.005 < need:
        return None
    if user_net + hit * pct / 100.0 > user_budget + 0.005:
        return None
    return pct


def _buyer_retention_pct(
    user_budget: float,
    partner_budget: float,
    user_gives: List[Dict[str, Any]],
    partner_gives: List[Dict[str, Any]],
) -> Optional[Tuple[str, int]]:
    """Retention the selling club keeps on one outgoing player so the buyer fits."""
    user_net = _cap_m(partner_gives) - _cap_m(user_gives)
    if user_net <= user_budget + 0.005:
        return None
    players = [a for a in partner_gives if a.get("type") == "player" and float(a.get("cap_m") or 0) > 0]
    if not players:
        return None
    target = max(players, key=lambda a: float(a.get("cap_m") or 0))
    hit = float(target.get("cap_m") or 0)
    need_cut = user_net - user_budget
    if need_cut <= 0 or hit <= 0:
        return None
    cap_pct = MAX_RETAIN_PCT
    try:
        from app.sim_engine.economy.cap_engine import max_retention_pct

        cap_pct = int(max_retention_pct(_RETENTION_LEAGUE.get("league")))
    except Exception:
        pass
    pct = int(min(cap_pct, ((need_cut / hit) * 100.0 // 5 + 1) * 5))
    if hit * pct / 100.0 + 0.005 < need_cut:
        return None
    partner_net = _cap_m(user_gives) - _cap_m(partner_gives) + hit * pct / 100.0
    if partner_net > partner_budget + 0.005:
        return None
    return str(target.get("id") or ""), pct


def _clause_blocked(player: Any) -> bool:
    """Full NMC/NTC only. A modified list can still move to the teams on it."""
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        mode = str(c.get("ntc_mode") or "").upper()
        kind = str(c.get("clause_type") or "").upper()
        if mode in ("MODIFIED", "MNTC", "M-NTC") or kind in ("M-NTC", "MNTC") or int(c.get("modified_no_trade_teams") or 0) > 0:
            return bool(c.get("no_move_clause") or c.get("nmc"))
        return bool(c.get("no_move_clause") or c.get("nmc") or c.get("no_trade_clause") or c.get("ntc"))
    if c is None:
        return False
    clauses = getattr(c, "clauses", None)
    if clauses is not None:
        mntc_n = int(getattr(clauses, "modifiedNoTradeTeams", 0) or 0)
        nested = str(getattr(clauses, "clause_type", "") or "").upper()
        mode = str(getattr(c, "ntc_mode", "") or "").upper()
        if mntc_n > 0 or nested in ("M-NTC", "MNTC") or mode in ("MODIFIED", "MNTC", "M-NTC"):
            return bool(getattr(clauses, "noMoveClause", False))
        return bool(getattr(clauses, "noMoveClause", False) or getattr(clauses, "noTradeClause", False))
    mntc_n = int(getattr(c, "modified_no_trade_teams", 0) or 0)
    if mntc_n > 0 or str(getattr(c, "ntc_mode", "") or "").upper() in ("MODIFIED", "MNTC", "M-NTC"):
        return bool(getattr(c, "no_move_clause", False) or getattr(c, "nmc", False))
    return bool(getattr(c, "no_move_clause", False) or getattr(c, "no_trade_clause", False))


def _mntc_blocks_destination(player: Any, acquiring: Any) -> bool:
    """True when a modified list does not include the club that would receive him."""
    c = getattr(player, "contract", None)
    if c is None:
        return False
    if isinstance(c, dict):
        mode = str(c.get("ntc_mode") or "").upper()
        kind = str(c.get("clause_type") or "").upper()
        modified = mode in ("MODIFIED", "MNTC", "M-NTC") or kind in ("M-NTC", "MNTC") or int(c.get("modified_no_trade_teams") or 0) > 0
        if c.get("nmc") or c.get("no_move_clause"):
            return False
        approved = list(c.get("approved_trade_teams") or c.get("ntc_teams") or c.get("approved_trade_team_ids") or [])
    else:
        mode = str(getattr(c, "ntc_mode", "") or "").upper()
        modified = mode in ("MODIFIED", "MNTC", "M-NTC") or int(getattr(c, "modified_no_trade_teams", 0) or 0) > 0
        approved = list(getattr(c, "approved_trade_teams", None) or getattr(c, "ntc_teams", None) or [])
    if not modified:
        return False
    if not approved:
        return True
    keys = {str(a).upper() for a in approved if a}
    dests = []
    for attr in ("team_id", "id", "abbreviation", "abbr", "code"):
        raw = getattr(acquiring, attr, None)
        if raw:
            dests.append(str(raw).upper())
    return not any(d in keys for d in dests)


def _player_label(p: Any) -> Dict[str, Any]:
    ident = getattr(p, "identity", None)
    name = str(getattr(ident, "name", None) or getattr(p, "name", "") or "?")
    pos = getattr(ident, "position", None) if ident else getattr(p, "position", "")
    age = getattr(ident, "age", None) if ident else getattr(p, "age", None)
    out = {"name": name, "pos": str(getattr(pos, "value", pos) or ""), "age": age}
    try:
        from app.sim_engine.economy.player_value import player_ovr_display, player_potential_display

        ovr = player_ovr_display(p)
        out["ovr"] = int(round(ovr))
        out["pot"] = int(round(max(ovr, player_potential_display(p, ovr_display=ovr))))
    except Exception:
        pass
    try:
        from app.sim_engine.generation.player_headshots import ensure_player_headshot, headshot_fields_from_player

        ensure_player_headshot(p)
        out["headshot"] = headshot_fields_from_player(p)
        nhl_id = getattr(p, "nhl_id", None) or getattr(p, "nhl_player_id", None)
        if nhl_id:
            out["headshot"]["nhl_id"] = nhl_id
    except Exception:
        pass
    return out


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
        rows = serialize_team_picks(league, str(team_id), min_year=min_year or None)
        if min_year:
            # NHL rule: only the next three drafts are tradeable.
            rows = [r for r in rows if int(r.get("year") or 0) <= int(min_year) + 2]
        return rows
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
        if not pid or pid in exclude or _clause_blocked(p) or _mntc_blocks_destination(p, acquiring):
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
            "cap_m": round(_trade_cap_hit(p, ctx), 3),
            "cap_full_m": round(_cap_hit(p), 3),
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
    # Quotas per asset kind so prospects and picks survive the cut next to roster players —
    # a pure top-N by value was all NHL depth + 1st/2nd-rounders, which is why every package
    # looked the same.
    quotas = {"NHL": 10, "AHL": 6, "pick": 7}
    taken: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for a in items:
        kind = a["type"] if a["type"] == "pick" else a.get("level", "NHL")
        if taken.get(kind, 0) >= quotas.get(kind, 6):
            continue
        taken[kind] = taken.get(kind, 0) + 1
        out.append(a)
    return out[:MAX_POOL]


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
        out["retained"] = int(a.get("retained") or 0)
    return out


def _public_asset(a: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("type", "id", "team", "value", "name", "pos", "age", "ovr", "pot", "level", "year", "round", "original_team_id", "cap_m", "cap_full_m", "retained")
    out = {k: a[k] for k in keep if k in a}
    if isinstance(a.get("headshot"), dict):
        out.update(a["headshot"])
    return out


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
        "user_gives_value": round(_eff(user_gives), 1),
        "user_gets_value": round(_eff(partner_gives), 1),
        "interest": round(float(interest), 3) if interest is not None else None,
        "verdict": result.get("verdict"),
        "fairness_gap": result.get("fairness_gap"),
    }


PER_PARTNER_TRIES = 3


def _user_can_retain(team: Any, player: Any, ctx: Dict[str, Any]) -> bool:
    """Cheap pre-check of the retention rules (max 3 slots, contract term left)."""
    try:
        from app.sim_engine.trades.trade_rules import _contract_years_for_retention, _retained_slots_used

        from app.sim_engine.economy.cap_engine import max_retained_slots

        if _retained_slots_used(team, _season_label_from_ctx(ctx)) >= max_retained_slots(ctx.get("league")):
            return False
        return _contract_years_for_retention(player) >= 1
    except Exception:
        return True  # rules engine will make the final call


def _team_name(team_by_id: Dict[str, Any], tid: str) -> str:
    t = team_by_id.get(str(tid))
    if t is None:
        return str(tid)
    city = str(getattr(t, "city", "") or "")
    name = str(getattr(t, "name", "") or "")
    return (f"{city} {name}".strip() if city and city not in name else name) or str(tid)


def _clean_reasons(result: Dict[str, Any], team_by_id: Dict[str, Any]) -> List[str]:
    """Rejection reasons with team ids swapped for names, de-duplicated."""
    import re as _re

    out: List[str] = []
    for r in (result.get("rejection_reasons") or []):
        txt = str(r)
        m = _re.match(r"^(\w+): (.*)$", txt)
        if m and m.group(1) in team_by_id:
            txt = f"{_team_name(team_by_id, m.group(1))}: {m.group(2)}"
        if txt not in out:
            out.append(txt)
    return out[:2]


ARCHETYPE_LABELS = {
    "hockey": "Hockey trade",
    "star_for_star": "Roster player swap",
    "youth": "Youth movement",
    "futures": "Futures package",
    "blockbuster": "Mixed package",
    "single": "One-for-one",
    "picks": "Draft capital",
    "depth_dump": "Depth + picks",
}


def _archetype(combo) -> str:
    """Name the shape of a package so finder results aren't all one template."""
    total = _eff(combo) or 1.0
    nhl = [a for a in combo if a.get("type") == "player" and a.get("level") == "NHL"]
    ahl = [a for a in combo if a.get("type") == "player" and a.get("level") != "NHL"]
    picks = [a for a in combo if a.get("type") == "pick"]
    young = [a for a in combo if a.get("type") == "player" and (a.get("age") or 99) <= 23]
    top_nhl = max((a["value"] for a in nhl), default=0.0)
    if len(combo) == 1:
        return "picks" if picks else "single"
    if not nhl and not ahl:
        return "picks"
    if not nhl:
        return "futures"
    if len(nhl) >= 2 and top_nhl / total < 0.7:
        return "star_for_star"
    if top_nhl / total >= 0.55:
        return "hockey"
    if len(nhl) == 1 and not ahl and len(picks) >= 2:
        return "depth_dump"  # low-end NHLer + a pile of picks — the old default
    if ahl or young:
        return "youth" if len(picks) <= 1 else "blockbuster"
    return "blockbuster"


def _pick_by_archetype(
    ranked: List[Tuple[float, List[Dict[str, Any]]]],
    *,
    order: List[str],
    limit: int,
    anchor_type: str,
) -> List[List[Dict[str, Any]]]:
    """Best package per archetype first, then fill; ``depth_dump`` only as a last resort."""
    best: Dict[str, List[List[Dict[str, Any]]]] = {}
    for _key, combo in ranked:
        arch = _archetype(combo)
        if anchor_type == "pick" and not any(a.get("type") == "player" for a in combo):
            continue
        bucket = best.setdefault(arch, [])
        if len(bucket) < 2:
            bucket.append(combo)
    out: List[List[Dict[str, Any]]] = []
    seen: set = set()
    used_ids: Dict[str, int] = {}

    def _add(combo) -> bool:
        key = tuple(sorted((a.get("type"), a.get("id")) for a in combo))
        if key in seen:
            return False
        # Don't let one partner asset headline every offer.
        if any(used_ids.get(a["id"], 0) >= 2 for a in combo):
            return False
        seen.add(key)
        for a in combo:
            used_ids[a["id"]] = used_ids.get(a["id"], 0) + 1
        out.append(combo)
        return True

    for rnd in range(2):
        for arch in order:
            if len(out) >= limit:
                return out
            bucket = best.get(arch) or []
            if len(bucket) > rnd:
                _add(bucket[rnd])
    for combo in best.get("depth_dump") or []:
        if len(out) >= limit:
            break
        _add(combo)
    return out


def _sell_candidates(
    offered_value: float,
    pool: List[Dict[str, Any]],
    fits,
    *,
    anchor_type: str = "player",
    order: Optional[List[str]] = None,
    rng: Any = None,
) -> List[List[Dict[str, Any]]]:
    """Return packages of different archetypes (hockey trade, youth, futures, swap...).

    Packages must carry real value (>= SELL_VALUE_FLOOR of the asset). Within an archetype
    the package closest to the partner's budget wins, with a little jitter so repeat searches
    don't always surface the same deal.
    """
    budget = offered_value * SELL_MARGIN
    floor = offered_value * SELL_VALUE_FLOOR
    ranked: List[Tuple[float, List[Dict[str, Any]]]] = []
    for combo in _combos(pool, MAX_COMBO_SIZE):
        total = _eff(combo)
        if total > budget or total < floor or not fits(combo):
            continue
        key = total / max(1.0, budget) + (rng.uniform(0.0, 0.06) if rng is not None else 0.0)
        # Long piles of small pieces read as filler — mild penalty per extra asset.
        key -= 0.015 * max(0, len(combo) - 2)
        ranked.append((key, list(combo)))
    ranked.sort(key=lambda t: t[0], reverse=True)
    order = order or ["hockey", "youth", "star_for_star", "single", "blockbuster", "futures", "picks"]
    return _pick_by_archetype(ranked, order=order, limit=5, anchor_type=anchor_type)


def _buy_candidates(
    price: float, pool: List[Dict[str, Any]], fits, rng: Any = None, premium: float = BUY_PREMIUM
) -> List[List[Dict[str, Any]]]:
    """Cheapest user package of each archetype that clears the partner's price."""
    need = price * premium
    ranked: List[Tuple[float, List[Dict[str, Any]]]] = []
    for combo in _combos(pool, MAX_COMBO_SIZE):
        total = _eff(combo)
        if total < need or total > need * 1.45 or not fits(combo):
            continue
        key = -(total / max(1.0, need)) + (rng.uniform(0.0, 0.05) if rng is not None else 0.0)
        key -= 0.015 * max(0, len(combo) - 2)
        ranked.append((key, list(combo)))
    if not ranked:  # only an overpay clears it
        for combo in _combos(pool, MAX_COMBO_SIZE):
            total = _eff(combo)
            if total >= need and fits(combo):
                ranked.append((-(total / max(1.0, need)), list(combo)))
    ranked.sort(key=lambda t: t[0], reverse=True)
    order = ["hockey", "youth", "futures", "star_for_star", "single", "blockbuster", "picks"]
    return _pick_by_archetype(ranked, order=order, limit=5, anchor_type="player")


DESPERATE_CHANCE = 0.08  # per asset, per 3-day window
DESPERATE_TTL_DAYS = 3


def _desperate_offer(
    session: Any,
    ctx: Dict[str, Any],
    *,
    mode: str,
    user_tid: str,
    user_team: Any,
    partners: List[Tuple[str, Any]],
    anchor_base: Dict[str, Any],
    anchor_value: Any,
    exclude: set,
) -> Optional[Dict[str, Any]]:
    """Rarely, a desperate club makes an offer you can't refuse.

    Sell: a contender/needy club overpays (~1.5-1.9x value) for your player or pick.
    Buy: a cash-strapped seller lets the target go for ~55-75% of his value.
    Rolled once per asset per 3-day window, so re-searching doesn't re-roll; the deal
    is stored on the session so the evaluator honours it when you propose it.
    """
    import random as _random
    from app.sim_engine.trades.trade_evaluator import package_signature

    league = ctx["league"]
    cursor = int(ctx.get("calendar_cursor") or 0)
    window = cursor // DESPERATE_TTL_DAYS
    seed = f"desperate|{mode}|{anchor_base.get('id')}|{window}|{getattr(session, 'session_id', '')}"
    rng = _random.Random(seed)
    if rng.random() >= DESPERATE_CHANCE:
        return None

    user_room, user_flex, user_budget = _nhl_room(user_team, league), _flex(user_team), _cap_budget(user_team, ctx)
    order = list(partners)
    rng.shuffle(order)
    for tid, partner in order[:8]:
        tid = str(tid)
        partner_room, partner_flex, partner_budget = _nhl_room(partner, league), _flex(partner), _cap_budget(partner, ctx)
        if mode == "sell":
            base_val = float(anchor_value(partner) or 0.0)
            if base_val <= 4.0:
                continue
            target = base_val * rng.uniform(1.45, 1.9)
            pool = _pool(partner, tid, user_team, league, ctx, exclude=exclude)
            give_side, get_side_pool = [dict(anchor_base, value=round(base_val, 2), retained=0)], pool
        else:
            base_val = float(anchor_value(user_team) or 0.0)
            if base_val <= 4.0:
                continue
            target = base_val * rng.uniform(0.55, 0.75)
            pool = _pool(user_team, user_tid, partner, league, ctx, exclude=exclude)
            give_side, get_side_pool = None, pool
        # Greedy package: biggest pieces first until the target is met (max 4 assets).
        combo: List[Dict[str, Any]] = []
        for a in sorted(get_side_pool, key=lambda r: -float(r.get("value") or 0.0)):
            if len(combo) >= 4 or _eff(combo) >= target:
                break
            if mode == "buy" and _eff(combo) + float(a.get("value") or 0.0) > base_val * 0.8:
                continue  # stay a genuine discount
            trial = combo + [a]
            if mode == "sell" and _eff(trial) > base_val * 2.1:
                continue  # an overpay, not a franchise heist
            if mode == "sell":
                if not _roster_fits(user_room, partner_room, give_side, trial, user_flex, partner_flex):
                    continue
                if _plan_cap(user_budget, partner_budget, give_side, trial) is None:
                    continue
            else:
                get_side = [dict(anchor_base, value=round(base_val, 2), retained=0)]
                if not _roster_fits(user_room, partner_room, trial, get_side, user_flex, partner_flex):
                    continue
                if _plan_cap(user_budget, partner_budget, trial, get_side) is None:
                    continue
            combo = trial
        if not combo:
            continue
        if mode == "sell" and _eff(combo) < base_val * 1.3:
            continue
        if mode == "sell":
            user_gives, user_gets = give_side, combo
        else:
            user_gives, user_gets = combo, [dict(anchor_base, value=round(base_val, 2), retained=0)]
        abt = {
            tid: [_payload_asset(a) for a in user_gives],
            user_tid: [_payload_asset(a) for a in user_gets],
        }
        sig = package_signature(abt)
        store = dict(getattr(session, "_desperate_offers", None) or {})
        store = {k: v for k, v in store.items() if int(v or 0) >= cursor}
        store[sig] = cursor + DESPERATE_TTL_DAYS
        session._desperate_offers = store
        ctx2 = dict(ctx)
        ctx2["desperate_offer_keys"] = set(ctx.get("desperate_offer_keys") or ()) | {sig}
        result = _evaluate(user_tid, tid, user_gives, user_gets, ctx2)
        if not result.get("accepted"):
            store.pop(sig, None)
            session._desperate_offers = store
            continue
        row = _offer_row(partner, tid, user_gives, user_gets, result)
        row["send_downs"] = _send_down_names(user_team, user_room, user_gives, user_gets)
        row["desperate"] = True
        team_name = str(getattr(partner, "name", "") or tid)
        row["archetype"] = "Desperate offer"
        row["desperate_note"] = (
            f"{team_name} is desperate — this offer expires in {DESPERATE_TTL_DAYS} days."
            if mode == "sell"
            else f"{team_name} needs to move him now — take him at a discount before it's gone ({DESPERATE_TTL_DAYS} days)."
        )
        return row
    return None


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
    _RETENTION_LEAGUE["league"] = league
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
        if mode == "sell" and _clause_blocked(anchor_player):
            return {
                "mode": mode,
                "anchor": {"type": "player", "id": asset_id, "team": source_tid, **_player_label(anchor_player)},
                "offers": [],
                "near_misses": [],
                "evaluated": 0,
                "note": "He has a no-movement or no-trade clause. Ask him to waive it before shopping him.",
            }
        anchor_label = {**_player_label(anchor_player), "level": found[1], "cap_m": round(_trade_cap_hit(anchor_player, ctx), 3)}
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
    sell_scored = True
    sell_can_retain = False

    import random as _random

    rng = _random.Random()
    if mode == "sell":
        user_budget = _cap_budget(user_team, ctx)
        user_room = _nhl_room(user_team, league)
        user_flex = _flex(user_team)
        base_order = ["hockey", "youth", "star_for_star", "single", "blockbuster", "futures", "picks"]
        p_index = 0
        can_retain = (
            anchor_player is not None
            and anchor_label.get("level") == "NHL"
            and _user_can_retain(user_team, anchor_player, ctx)
        )
        scored: List[Tuple[float, str, Any, float, List[List[Dict[str, Any]]], float]] = []
        for tid, partner in partners:
            tid = str(tid)
            offered = anchor_value(partner)
            if offered <= 0 and anchor_player is not None:
                # Dumping a negative asset: the club that takes him also receives a sweetener.
                sweet = max(6.0, abs(offered))
                user_pool = _pool(user_team, user_tid, partner, league, ctx, exclude=exclude, max_value=sweet * SELL_MARGIN)
                partner_room = _nhl_room(partner, league)
                user_flex_now = _flex(user_team)
                partner_flex = _flex(partner)

                def dump_fits(combo, pr=partner_room, pf=partner_flex, uf=user_flex_now):
                    give = [anchor_base] + list(combo)
                    return _roster_fits(user_room, pr, give, [], uf, pf)

                cands = _sell_candidates(sweet, user_pool, dump_fits, anchor_type="player", rng=rng)
                for cand in cands[:2]:
                    if evals >= MAX_EVALS or len(offers) >= limit:
                        break
                    give = [{**anchor_base, "value": round(offered, 2)}] + list(cand)
                    evals += 1
                    result = _evaluate(user_tid, tid, give, [], ctx)
                    row = _offer_row(partner, tid, give, [], result)
                    row["archetype"] = "Cap dump"
                    if result.get("accepted"):
                        offers.append(row)
                    else:
                        near.append({**row, "reasons": _clean_reasons(result, team_by_id)})
                continue
            if offered <= 2.0:
                continue
            partner_budget = _cap_budget(partner, ctx)
            pool = _pool(partner, tid, user_team, league, ctx, exclude=exclude, max_value=offered * SELL_MARGIN)
            partner_room = _nhl_room(partner, league)
            partner_flex = _flex(partner)

            def fits(combo, ur=user_room, pr=partner_room, ub=user_budget, pb=partner_budget, pf=partner_flex):
                if not _roster_fits(ur, pr, [anchor_base], combo, user_flex, pf):
                    return False
                if _plan_cap(ub, pb, [anchor_base], list(combo),
                             retain_on=anchor_base, retain_allowed=can_retain) is not None:
                    return True
                return _buyer_retention_pct(ub, pb, [anchor_base], list(combo)) is not None

            # Rotate which archetype each club leads with so the board isn't one template.
            rot = p_index % len(base_order)
            p_index += 1
            order = base_order[rot:] + base_order[:rot]
            cands = _sell_candidates(offered, pool, fits, anchor_type=asset_type, order=order, rng=rng)
            if not cands:
                continue
            best_total = max(_eff(c) for c in cands)
            scored.append((best_total / max(1.0, offered), tid, partner, offered, cands, partner_budget))
        scored.sort(key=lambda s: s[0], reverse=True)
        sell_scored = bool(scored)
        sell_can_retain = can_retain
        # Round-robin: each partner gets at most PER_PARTNER_TRIES evaluator runs before
        # we move on, so one club can't consume the whole budget.
        for _ratio, tid, partner, offered, cands, partner_budget in scored:
            if evals >= MAX_EVALS or len(offers) >= limit:
                break
            accepted_for_partner = 0
            tries = 0
            for cand in cands:
                if evals >= MAX_EVALS or len(offers) >= limit or accepted_for_partner >= 2 or tries >= PER_PARTNER_TRIES:
                    break
                pct = _plan_cap(user_budget, partner_budget, [anchor_base], cand,
                                retain_on=anchor_base, retain_allowed=can_retain) or 0
                buyer_ret = None if pct else _buyer_retention_pct(user_budget, partner_budget, [anchor_base], cand)
                give = [{**anchor_base, "value": round(offered, 2), "retained": pct}]
                if buyer_ret:
                    bid, bpct = buyer_ret
                    cand = [{**a, "retained": bpct} if str(a.get("id")) == bid else a for a in cand]
                evals += 1
                tries += 1
                result = _evaluate(user_tid, tid, give, cand, ctx)
                row = _offer_row(partner, tid, give, cand, result)
                row["send_downs"] = _send_down_names(user_team, user_room, give, cand)
                row["archetype"] = ARCHETYPE_LABELS.get(_archetype(cand), "Package")
                if result.get("accepted"):
                    offers.append(row)
                    accepted_for_partner += 1
                else:
                    near.append({**row, "reasons": _clean_reasons(result, team_by_id)})
    else:
        partner_tid, partner = partners[0]
        price = anchor_value(user_team)
        user_room, partner_room = _nhl_room(user_team, league), _nhl_room(partner, league)
        user_flex, partner_flex = _flex(user_team), _flex(partner)

        user_budget = _cap_budget(user_team, ctx)
        partner_budget = _cap_budget(partner, ctx)

        partner_can_retain = (
            anchor_player is not None
            and anchor_label.get("level") == "NHL"
            and _user_can_retain(partner, anchor_player, ctx)
        )

        def buy_retention(combo) -> Optional[int]:
            """Retention % the seller keeps on the target so the buyer fits under the cap
            (0 when none is needed); None when no legal structure exists."""
            pct = _plan_cap(user_budget, partner_budget, list(combo), [anchor_base])
            if pct is not None:
                return pct
            if not partner_can_retain:
                return None
            hit = float(anchor_base.get("cap_m") or 0.0)
            if hit <= 0:
                return None
            user_net = hit - _cap_m(combo)
            need_cut = user_net - user_budget
            if need_cut <= 0:
                return None
            try:
                from app.sim_engine.economy.cap_engine import max_retention_pct

                max_pct = int(max_retention_pct(league))
            except Exception:
                max_pct = MAX_RETAIN_PCT
            pct = int(((need_cut / hit) * 100.0 // 5 + 1) * 5)
            if pct > max_pct:
                return None
            partner_net = _cap_m(combo) - hit + hit * pct / 100.0
            if partner_net > partner_budget + 0.005:
                return None
            return pct

        def fits(combo):
            if not _roster_fits(user_room, partner_room, combo, [anchor_base], user_flex, partner_flex):
                return False
            return buy_retention(combo) is not None

        if price <= 0 and anchor_player is not None:
            # Buying a negative contract: the seller pays you to take him.
            sweet = max(6.0, abs(price))
            seller_pool = _pool(partner, partner_tid, user_team, league, ctx, exclude=exclude, max_value=sweet * 1.2)

            def seller_fits(combo):
                get = [anchor_base] + list(combo)
                return _roster_fits(user_room, partner_room, [], get, user_flex, partner_flex)

            for cand in _sell_candidates(sweet, seller_pool, seller_fits, anchor_type="player", rng=rng)[:4]:
                if evals >= MAX_EVALS or len(offers) >= 4:
                    break
                get = [{**anchor_base, "value": round(price, 2)}] + list(cand)
                evals += 1
                result = _evaluate(user_tid, partner_tid, [], get, ctx)
                row = _offer_row(partner, partner_tid, [], get, result)
                row["send_downs"] = _send_down_names(user_team, user_room, [], get)
                row["archetype"] = "Takes the contract"
                if result.get("accepted"):
                    offers.append(row)
                else:
                    near.append({**row, "reasons": _clean_reasons(result, team_by_id)})

        full_pool = _pool(user_team, user_tid, partner, league, ctx, exclude=exclude)
        pool = _pool(user_team, user_tid, partner, league, ctx, exclude=exclude, max_value=max(price, 0.0) * 1.6)
        tried: set = set()
        # Escalate the offer until the GM bites: fair price first, then real overpays.
        for premium in (BUY_PREMIUM, 1.16, 1.32):
            if price <= 0 or len(offers) >= 3 or evals >= MAX_EVALS:
                break
            cands = _buy_candidates(price, pool, fits, rng=rng, premium=premium)
            if not cands:
                cands = _buy_candidates(price, full_pool, fits, rng=rng, premium=premium)
            for cand in cands:
                if evals >= MAX_EVALS or len(offers) >= 5:
                    break
                key = tuple(sorted(a["id"] for a in cand))
                if key in tried:
                    continue
                tried.add(key)
                pct = buy_retention(cand) or 0
                get = [{**anchor_base, "value": round(price, 2), "retained": pct}]
                evals += 1
                result = _evaluate(user_tid, partner_tid, cand, get, ctx)
                row = _offer_row(partner, partner_tid, cand, get, result)
                row["send_downs"] = _send_down_names(user_team, user_room, cand, get)
                row["archetype"] = ARCHETYPE_LABELS.get(_archetype(cand), "Package")
                if pct:
                    row["archetype"] += f" · {pct}% retained"
                if result.get("accepted"):
                    offers.append(row)
                else:
                    near.append({**row, "reasons": _clean_reasons(result, team_by_id)})

    offers.sort(key=lambda o: o["user_gets_value"] - o["user_gives_value"], reverse=True)
    try:
        dsp = _desperate_offer(
            session, ctx, mode=mode, user_tid=user_tid, user_team=user_team, partners=partners,
            anchor_base=anchor_base, anchor_value=anchor_value, exclude=exclude,
        )
    except Exception:
        dsp = None
    if dsp:
        offers.insert(0, dsp)
    near.sort(key=lambda o: float(o.get("interest") or 0.0), reverse=True)
    near_one_per_team: List[Dict[str, Any]] = []
    seen_partner: set = set()
    for row in near:
        if row["partner_team_id"] in seen_partner:
            continue
        seen_partner.add(row["partner_team_id"])
        near_one_per_team.append(row)
    near = near_one_per_team
    note = None
    if not offers and mode == "sell" and not sell_scored:
        note = (
            "No club has the cap room and roster spot to take this contract right now"
            + (" (even with 50% retained)." if sell_can_retain else ".")
        )
    return {
        "mode": mode,
        "anchor": anchor_base,
        "offers": offers[:limit],
        "near_misses": near[:4] if not offers else [],
        "evaluated": evals,
        "note": note,
    }
