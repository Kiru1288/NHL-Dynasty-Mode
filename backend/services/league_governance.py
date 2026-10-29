"""
League governance — the annual Board of Governors meeting and everything it moves.

Each offseason (after Retirements, before the Cap Report) the Board meets. Five
proposals are drawn from the 200-rule catalog (league_rule_catalog), plus live
relocation / expansion bids when the league's condition produces them. The user's
club casts one of 32 (or more) votes; the 31 CPU governors vote from their own
situation — market size, profit, contender or rebuilder, cap room, the rule's
lean — with a stable per-save personality. Passed rules go into the rulebook and
their effects are read by the live systems listed in EFFECT_SPECS.

Also here, because they all hang off league money:
  * franchise values that move every offseason (small markets can boom and move up
    a market tier, which raises their revenue base);
  * star revenue spikes for the clubs of the league's top performers;
  * revenue-funded scouting budgets;
  * signing-bonus demands from free agents (only clubs that can pay get them);
  * relocation (with players demanding out) and expansion (with an expansion draft).

State lives on ``session.league_governance``; the aggregated modifiers are mirrored
onto ``league.governance_modifiers`` so code that only holds the league can read them.
"""

from __future__ import annotations

import math
import random
import zlib
from typing import Any, Dict, List, Optional, Tuple

from services.league_rule_catalog import (
    ABSOLUTE_EFFECTS,
    CANDIDATE_MARKETS,
    CATEGORIES,
    EXPANSION_NICKNAMES,
    RULE_BY_ID,
    RULES,
)
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

PROPOSALS_PER_MEETING = 5
LOBBY_TOKENS_PER_MEETING = 2
LOBBY_SWING = 0.18
LOBBY_TARGETS = 4
#: Franchise value growth per point of fan confidence above 55 (x/100 per season).
FAN_VALUE_COEF = 0.15
#: Proposals reach the Board after committee vetting, so a neutral governor sits a
#: little above whatever bar the proposal needs — the same head start for every
#: category. Lean, the rule's popularity and personality decide the close ones.
VETTING_MARGIN = 0.04
#: Owners lean slightly toward keeping a rule they already operate under.
REPEAL_STATUS_QUO = -0.03
#: Chance a meeting includes a repeal vote once the rulebook has this many rules.
REPEAL_MIN_RULES = 5
REPEAL_CHANCE = 0.45

#: Hard bounds on the combined effect of every rule in force (so 30 seasons of
#: votes can't stack into absurd numbers).
EFFECT_LIMITS: Dict[str, Tuple[float, float]] = {
    "rev_all": (-12.0, 15.0),
    "rev_small": (-9.0, 12.0),
    "rev_large": (-9.0, 9.0),
    "revenue_share": (-4.0, 6.0),
    "opex": (-4.5, 6.0),
    "star_rev": (-20.0, 40.0),
    "fan": (-25.0, 25.0),
    "playoff_rev": (-40.0, 50.0),
    "value_growth": (-2.0, 2.0),
    "cap_growth": (-1.5, 1.5),
    "floor_ratio": (-0.06, 0.05),
    "max_salary_pct": (-0.05, 0.05),
    "bonus_pct": (-0.15, 0.15),
    "bonus_floor_m": (-40.0, 40.0),
    "min_salary_m": (-0.1, 0.3),
    "fa_bonus_demand": (-25.0, 25.0),
    "retention_slots": (-2.0, 2.0),
    "retention_max_pct": (-20.0, 20.0),
    "trade_volume": (-50.0, 70.0),
    "trade_demand_rate": (-50.0, 50.0),
    "scouting_budget": (-35.0, 45.0),
    "injury_rate": (-35.0, 25.0),
    "relocation_ease": (-0.15, 0.15),
    "expansion_pressure": (-40.0, 50.0),
    # On-ice rules (read by app.sim_engine.league.game_rules).
    "scoring_pace": (-18.0, 25.0),
    "penalty_rate": (-35.0, 45.0),
    "pp_efficiency": (-25.0, 35.0),
    "home_ice": (-60.0, 80.0),
}

#: Catalog values were tuned so small that most rules changed nothing a player could
#: see (+0.6 fan sentiment on a 0-100 scale ≈ +0.3% revenue). Effects are scaled up
#: where they are applied and shown, so ballots and live systems agree.
EFFECT_SCALE: Dict[str, float] = {
    "fan": 5.0,
    "injury_rate": 2.5,
    "trade_volume": 2.0,
    "trade_demand_rate": 1.5,
    "rev_all": 1.5,
    "rev_small": 1.5,
    "rev_large": 1.5,
    "opex": 1.5,
    "scouting_budget": 1.5,
}


def _scaled(key: str, value: float) -> float:
    return float(value) * EFFECT_SCALE.get(key, 1.0)
RECENT_PROPOSAL_YEARS = 3
MAX_LEAGUE_TEAMS = 36

THRESHOLDS = {
    "majority": (0.5, "Majority"),
    "two_thirds": (2.0 / 3.0, "Two-thirds"),
    "three_quarters": (0.75, "Three-quarters"),
}

#: effect key → (label, formatter kind, consumer)
EFFECT_SPECS: Dict[str, Tuple[str, str, str]] = {
    "rev_all": ("League revenue", "pct", "league_operations.calculate_team_revenue"),
    "rev_small": ("Small-market revenue", "pct", "league_operations.calculate_team_revenue"),
    "rev_large": ("Large-market revenue", "pct", "league_operations.calculate_team_revenue"),
    "revenue_share": ("Revenue-sharing pool", "pts", "league_operations payload (large → small transfer)"),
    "opex": ("Operating costs", "pct", "league_operations.calculate_team_revenue"),
    "star_rev": ("Star revenue", "pct", "league_operations superstar boost + star spikes"),
    "fan": ("Fan sentiment", "pts_raw", "league_operations revenue / attendance"),
    "playoff_rev": ("Playoff revenue", "pct", "league_operations._playoff_revenue_bonus"),
    "value_growth": ("Franchise value growth", "pts_yr", "league_governance.advance_franchise_values"),
    "cap_growth": ("Cap growth (model seasons)", "pts_yr", "league_operations cap projection"),
    "cap_adjust_m": ("Next season's cap", "money", "league_operations projection → offseason rollover"),
    "floor_ratio": ("Payroll floor", "pct_of_cap", "cap_engine.nhl_lower_limit_millions"),
    "max_salary_pct": ("Max salary", "pct_of_cap", "contract_economy.compute_market_value"),
    "max_term_own": ("Max term (re-sign)", "years", "contract_economy term limits"),
    "max_term_ufa": ("Max term (free agents)", "years", "contract_economy term limits"),
    "bonus_pct": ("Signing-bonus room", "pts", "franchise_offseason.signing_bonus_max_pct_for_revenue"),
    "bonus_floor_m": ("Bonus revenue floor", "money", "franchise_offseason.signing_bonus_revenue_floor"),
    "min_salary_m": ("League minimum", "money_k", "contract_economy.compute_market_value"),
    "fa_bonus_demand": ("Free agents demanding bonuses", "pts", "fa_market_engine / contract_economy bonus gate"),
    "retention_slots": ("Retained-salary slots", "count", "trade_rules retention check"),
    "retention_max_pct": ("Max salary retention", "pts", "trade_rules retention check"),
    "trade_volume": ("CPU trade volume", "pct", "cpu_trade_proposer.propose_and_execute_cpu_trades"),
    "trade_demand_rate": ("Trade-request rate", "pct", "trade_demand_engine + relocation demands"),
    "scouting_budget": ("Scouting budgets", "pct", "league_governance.revenue_scouting_budget"),
    "injury_rate": ("Injury rate", "pct", "world.injuries.maybe_injure_roster_subset"),
    "relocation_ease": ("Relocation pressure", "risk", "league_operations.calculate_relocation_risk"),
    "expansion_pressure": ("Expansion momentum", "pts_raw", "league_governance expansion proposals"),
    # On-ice (app.sim_engine.league.game_rules → engine + standings)
    "scoring_pace": ("Goal scoring", "pct", "engine goal rate (event + fast sim)"),
    "penalty_rate": ("Penalties called", "pct", "engine penalty generation"),
    "pp_efficiency": ("Power-play scoring", "pct", "engine power-play goals"),
    "home_ice": ("Home-ice edge", "pct", "engine home goal bonus"),
    "ot_minutes": ("Overtime length", "minutes", "game_rules.shootout_share + event OT"),
    "ties_allowed": ("Shootout", "flag_ties", "standings ties"),
    "pts_reg_win": ("Regulation win", "points", "standings points"),
    "pts_ot_win": ("OT / shootout win", "points", "standings points"),
    "pts_otl": ("OT / shootout loss", "points", "standings points"),
}

#: Defaults the absolute keys fall back to when no rule sets them.
ABSOLUTE_DEFAULTS: Dict[str, Any] = {
    "ot_minutes": 5,
    "ties_allowed": 0,
    "pts_reg_win": 2,
    "pts_ot_win": 2,
    "pts_otl": 1,
}

CANADA_ABBRS = frozenset({"TOR", "MTL", "OTT", "VAN", "CGY", "EDM", "WPG", "QUE", "HAM", "SAS", "HFX"})

RELOCATION_DEMAND_COPY = {
    "headline": "{name} wants out after the move",
    "body": "{name} doesn't want to uproot his family for the relocation and has asked for a trade.",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(d)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _league(session: Any) -> Any:
    return getattr(getattr(session, "sim", None), "league", None)


def _season(session: Any) -> int:
    return int(getattr(session, "season_calendar_year", 2025) or 2025)


def _salt(session: Any) -> int:
    try:
        from app.sim_engine.trades.needs_matcher import save_entropy_salt

        return int(save_entropy_salt(_league(session)))
    except Exception:
        return 1


def _unit(*parts: Any) -> float:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return (zlib.crc32(key) & 0xFFFFFFFF) / 0xFFFFFFFF


def _team_abbr(team: Any) -> str:
    try:
        from services.franchise_sim import _franchise_team_abbrev

        return str(_franchise_team_abbrev(team) or "").upper()
    except Exception:
        return str(getattr(team, "abbr", "") or "").upper()


def _team_name(team: Any) -> str:
    try:
        from services.franchise_sim import _display_team

        return str(_display_team(team))
    except Exception:
        return f"{getattr(team, 'city', '')} {getattr(team, 'name', '')}".strip()


def _news(session: Any, headline: str, details: str, *, team_id: str = "", kind: str = "league") -> None:
    """Post to the league news / storylines feed (best effort)."""
    try:
        from services.franchise_sim import _record_storyline

        sy = _season(session)
        _record_storyline(session, {
            "id": f"story:gov:{sy}:{zlib.crc32(headline.encode()) & 0xFFFFFF:x}",
            "type": "league",
            "kind": kind,
            "headline": headline,
            "details": details,
            "cause": "Board of Governors",
            "team": team_id,
            "team_id": team_id,
            "priority": 3,
            "date": str(sy),
        })
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def ensure_governance(session: Any) -> Dict[str, Any]:
    gov = getattr(session, "league_governance", None)
    if not isinstance(gov, dict):
        gov = {}
        try:
            session.league_governance = gov
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    gov.setdefault("rulebook", {})  # rule_id → {season, title, cat, effects}
    gov.setdefault("meetings", {})  # str(season) → meeting
    gov.setdefault("history", [])  # flattened decided proposals, newest last
    gov.setdefault("proposed_log", {})  # rule_id → last season proposed
    gov.setdefault("pending_cap_adjust_m", 0.0)
    gov.setdefault("applied_cap_adjust", [])
    gov.setdefault("franchise_values", {})  # tid → {"value_b", "history": [...]}
    gov.setdefault("values_advanced_for", [])
    gov.setdefault("star_spikes", {})  # str(season) → {tid: [{player, m, reason}]}
    gov.setdefault("fee_shares", {})  # str(season) → $M per club
    gov.setdefault("honeymoon", {})  # tid → {str(season): $M}
    gov.setdefault("relocations", [])
    gov.setdefault("expansion", {"pending": [], "joined": []})
    gov.setdefault("used_markets", [])
    return gov


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------


def compute_modifiers(gov: Dict[str, Any]) -> Dict[str, float]:
    mods: Dict[str, float] = {}
    absolute: Dict[str, Tuple[int, float]] = {}
    for rid, row in (gov.get("rulebook") or {}).items():
        season = int(row.get("season") or 0)
        for k, v in (row.get("effects") or {}).items():
            if k == "cap_adjust_m":
                continue  # one-time, handled through pending_cap_adjust_m
            if k in ABSOLUTE_EFFECTS:
                prev = absolute.get(k)
                if prev is None or season >= prev[0]:
                    absolute[k] = (season, float(v))
            else:
                mods[k] = mods.get(k, 0.0) + _scaled(k, float(v))
    for k, v in list(mods.items()):
        lo, hi = EFFECT_LIMITS.get(k, (-1e9, 1e9))
        mods[k] = _clamp(v, lo, hi)
    for k, (_s, v) in absolute.items():
        mods[k] = v
    return {k: round(v, 4) for k, v in mods.items()}


def sync_governance_to_league(session: Any) -> Dict[str, float]:
    """Mirror modifiers onto the league so league-only code paths can read them."""
    gov = ensure_governance(session)
    mods = compute_modifiers(gov)
    gov["modifiers"] = mods
    league = _league(session)
    if league is not None:
        try:
            league.governance_modifiers = dict(mods)
            league.governance_floor_delta = float(mods.get("floor_ratio", 0.0))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from app.sim_engine.economy.cap_engine import set_board_min_salary_delta

        set_board_min_salary_delta(float(mods.get("min_salary_m", 0.0)))
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return mods


def gov_mod(holder: Any, key: str, default: float = 0.0) -> float:
    """Read one modifier from a league or a session (league mirror preferred)."""
    league = holder
    if hasattr(holder, "sim"):
        league = _league(holder)
    mods = getattr(league, "governance_modifiers", None) if league is not None else None
    if isinstance(mods, dict) and key in mods:
        return _sf(mods.get(key), default)
    if hasattr(holder, "league_governance"):
        gov = getattr(holder, "league_governance", None)
        if isinstance(gov, dict):
            return _sf((gov.get("modifiers") or {}).get(key, default), default)
    return float(default)


def injury_rate_multiplier(session: Any) -> float:
    return _clamp(1.0 + gov_mod(session, "injury_rate") / 100.0, 0.5, 1.6)


def trade_volume_multiplier(league: Any) -> float:
    return _clamp(1.0 + gov_mod(league, "trade_volume") / 100.0, 0.4, 1.8)


def trade_demand_multiplier(holder: Any) -> float:
    return _clamp(1.0 + gov_mod(holder, "trade_demand_rate") / 100.0, 0.3, 2.0)


# ---------------------------------------------------------------------------
# Team context (traits) for voting
# ---------------------------------------------------------------------------


def _team_rows(session: Any) -> Dict[str, Dict[str, Any]]:
    """Annualised revenue / profit rows for every club (phase-independent)."""
    from services.league_operations import _apply_revenue_sharing, calculate_team_revenue

    out: Dict[str, Dict[str, Any]] = {}
    uid = str(getattr(session, "user_team_id", "") or "")
    for tid, team in (getattr(session, "team_by_id", None) or {}).items():
        if team is None:
            continue
        try:
            out[str(tid)] = calculate_team_revenue(session, team, str(tid), is_user=str(tid) == uid, annual=True)
        except Exception:
            out[str(tid)] = {"market_tier_key": "medium", "profit": 0.0, "revenue": 180.0}
    if out:
        _apply_revenue_sharing(session, list(out.values()))
        _refresh_relocation_risk(session, out)
    return out


def _refresh_relocation_risk(session: Any, rows: Dict[str, Dict[str, Any]]) -> None:
    """Relocation risk from the post-sharing profit (it was scored before sharing, so a
    club in the black after sharing could still look like a relocation case)."""
    try:
        from services.league_operations import calculate_relocation_risk
    except Exception:
        return
    ease = gov_mod(session, "relocation_ease")
    teams = getattr(session, "team_by_id", None) or {}
    for tid, row in rows.items():
        if "revenue_sharing_m" not in row:
            continue
        try:
            risk = calculate_relocation_risk(
                teams.get(tid), _sf(row.get("revenue")), _sf(row.get("profit")),
                _sf(row.get("fan_sentiment"), 55.0), _sf(row.get("win_pct"), 0.5),
                str(row.get("market_tier_key") or "medium"),
            )
            row["relocation_risk"] = round(_clamp(risk + ease, 0.05, 0.95), 3)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)


def _standing_ranks(session: Any) -> Dict[str, float]:
    """Points % by club (0.5 when no games)."""
    st = getattr(session, "standings", None)
    recs = getattr(st, "records", None) if st is not None else None
    out: Dict[str, float] = {}
    if isinstance(recs, dict):
        for tid, rec in recs.items():
            gp = int(getattr(rec, "gp", 0) or 0)
            pts = _sf(getattr(rec, "points", None), -1)
            if pts < 0:
                w = int(getattr(rec, "wins", 0) or getattr(rec, "w", 0) or 0)
                otl = int(getattr(rec, "otl", 0) or 0)
                pts = 2 * w + otl
            out[str(tid)] = (pts / (2.0 * gp)) if gp > 0 else 0.5
    return out


def _cap_space(session: Any, team: Any) -> float:
    try:
        from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot

        sy = _season(session)
        # The Board meets in the offseason: clubs vote on next season's books.
        phase = str(getattr(session, "phase", "") or getattr(session, "season_phase", "") or "").lower()
        if phase in ("offseason", "post_cup", "complete"):
            sy += 1
        snap = calculate_team_cap_snapshot(team, league=_league(session), season_label=f"{sy}-{(sy + 1) % 100:02d}")
        return _sf(snap.get("usableCapSpace"), 5.0)
    except Exception:
        return 5.0


def team_traits(session: Any, rows: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    rows = rows if rows is not None else _team_rows(session)
    pct = _standing_ranks(session)
    ordered = sorted(pct.items(), key=lambda kv: -kv[1])
    n = max(1, len(ordered))
    rank_of = {tid: i for i, (tid, _v) in enumerate(ordered)}
    out: Dict[str, Dict[str, Any]] = {}
    for tid, team in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid)
        row = rows.get(tid) or {}
        tier = str(row.get("market_tier_key") or "medium")
        profit = _sf(row.get("profit"), 0.0)
        r = rank_of.get(tid, n // 2)
        traits = {"all", tier}
        if profit >= 15.0:
            traits.add("rich")
        if profit < 0.0:
            traits.add("poor")
        if r < max(4, n // 3):
            traits.add("contender")
        elif r >= n - max(4, n // 3):
            traits.add("rebuilder")
        space = _cap_space(session, team)
        if space < 3.0:
            traits.add("cap_tight")
        elif space > 12.0:
            traits.add("cap_room")
        abbr = _team_abbr(team)
        out[tid] = {
            "traits": traits,
            "tier": tier,
            "profit": round(profit, 1),
            "revenue": round(_sf(row.get("revenue"), 0.0), 1),
            "abbr": abbr,
            "name": _team_name(team),
            "city": str(getattr(team, "city", "") or ""),
            "conference": str(getattr(team, "conference", "") or ""),
            "cap_space": round(space, 2),
            "relocation_risk": _sf(row.get("relocation_risk"), 0.2),
            "canadian": abbr in CANADA_ABBRS,
        }
    return out


_TRAIT_COPY = {
    "small": "Small market",
    "medium": "Mid market",
    "large": "Large market",
    "rich": "Profitable club",
    "poor": "Losing money",
    "contender": "Contender",
    "rebuilder": "Rebuilding",
    "cap_tight": "Cap-tight",
    "cap_room": "Cap room to burn",
    "all": "League-wide stance",
}


# ---------------------------------------------------------------------------
# Proposal drawing
# ---------------------------------------------------------------------------


def _threshold_for(cat: str) -> str:
    return str((CATEGORIES.get(cat) or {}).get("threshold") or "two_thirds")


def _votes_needed(kind: str, n_teams: int) -> int:
    frac = THRESHOLDS.get(kind, THRESHOLDS["two_thirds"])[0]
    if kind == "majority":
        return n_teams // 2 + 1
    return int(math.ceil(frac * n_teams - 1e-9))


#: League-average franchise value the candidate-market values in the catalog are quoted at.
CATALOG_VALUE_ANCHOR_B = 2.55


def market_value_b(session: Any, market: Dict[str, Any]) -> float:
    """A candidate city's franchise value in today's money. The catalog values are fixed,
    while league values grow every year, so unscaled they made every relocated or
    expansion club a "small market" (and a relocation target) within a few seasons."""
    try:
        avg = league_average_value(session)
    except Exception:
        avg = CATALOG_VALUE_ANCHOR_B
    return round(_sf(market.get("value_b"), 1.8) * max(0.5, avg / CATALOG_VALUE_ANCHOR_B), 3)


def _record_profit_log(session: Any, traits: Dict[str, Dict[str, Any]]) -> None:
    """Keep each club's offseason profit so relocation needs sustained losses."""
    gov = ensure_governance(session)
    log = gov.setdefault("profit_log", {})
    sy = str(_season(session))
    for tid, t in traits.items():
        row = log.setdefault(str(tid), {})
        row[sy] = round(_sf(t.get("profit")), 1)
        if len(row) > 12:
            for k in sorted(row, key=lambda x: int(x))[:-12]:
                row.pop(k, None)


def _prior_profit(session: Any, tid: str) -> Optional[float]:
    row = (ensure_governance(session).get("profit_log") or {}).get(str(tid)) or {}
    prev = str(_season(session) - 1)
    return _sf(row.get(prev)) if prev in row else None


RELOCATION_MIN_RISK = 0.55


def _relocation_candidate(
    session: Any,
    traits: Dict[str, Dict[str, Any]],
    *,
    exclude_markets: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """A club applies to move only when it is actually losing money (after revenue
    sharing) for a second straight year — or badly this year — its risk is high, and
    its owner decides to apply. The user's own club is never put up for a move."""
    gov = ensure_governance(session)
    uid = str(getattr(session, "user_team_id", "") or "")
    recent = {str(r.get("team_id")) for r in gov.get("relocations") or [] if _season(session) - int(r.get("season") or 0) < 8}
    joined_recent = {
        str(j.get("team_id")) for j in (gov.get("expansion") or {}).get("joined") or []
        if _season(session) - int(j.get("season") or 0) < 6
    }
    best = None
    for tid, t in traits.items():
        if tid == uid or tid in recent or tid in joined_recent:
            continue
        profit = _sf(t.get("profit"))
        if profit >= 0.0:
            continue  # profitable clubs do not move
        prior = _prior_profit(session, tid)
        if not ((prior is not None and prior < 0.0) or profit <= -8.0):
            continue  # one soft year is not a relocation case
        risk = _sf(t.get("relocation_risk"))  # already includes Board relocation rules
        if risk < RELOCATION_MIN_RISK:
            continue
        intent = _clamp((risk - 0.45) * 2.2 + (0.15 if profit <= -15.0 else 0.0), 0.10, 0.90)
        if _unit(_salt(session), "reloc-intent", _season(session), tid) > intent:
            continue  # ownership chooses to stay and fight this year
        if best is None or risk > best[1]:
            best = (tid, risk)
    if best is None:
        return None
    used = set(gov.get("used_markets") or []) | set(exclude_markets or ())
    existing = {t["abbr"] for t in traits.values()}
    cur_tier = traits[best[0]]["tier"]
    tier_rank = {"small": 0, "medium": 1, "large": 2}
    options = [
        m for m in CANDIDATE_MARKETS
        if m["abbr"] not in used and m["abbr"] not in existing and tier_rank[m["tier"]] >= tier_rank.get(cur_tier, 0)
    ]
    # Prefer a city in the club's own conference so the league stays balanced.
    conf = str(traits[best[0]].get("conference") or "")
    same = [m for m in options if conf and str(m.get("conference") or "") == conf]
    options = same or options
    if not options:
        return None
    pick = options[int(_unit(_salt(session), "reloc", _season(session), best[0]) * len(options)) % len(options)]
    return {"team_id": best[0], "risk": round(best[1], 3), "market": pick}


def _expansion_candidate(
    session: Any,
    traits: Dict[str, Dict[str, Any]],
    *,
    exclude_markets: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    gov = ensure_governance(session)
    exp = gov.get("expansion") or {}
    if exp.get("pending"):
        return None
    n = len(traits)
    if n >= MAX_LEAGUE_TEAMS:
        return None
    last = max([int(j.get("season") or 0) for j in exp.get("joined") or []] + [0])
    if last and _season(session) - last < 4:
        return None
    pressure = gov_mod(session, "expansion_pressure")
    profits = [t["profit"] for t in traits.values()]
    losing = sum(1 for p in profits if p < 0)
    chance = _clamp(0.18 + pressure / 100.0 + (0.08 if losing <= 6 else -0.08), 0.02, 0.85)
    # League option: put an expansion bid on the agenda every N seasons.
    every = int(exp.get("schedule_every") or 0)
    if every > 0:
        anchor = last or int(exp.get("schedule_anchor") or _season(session))
        exp.setdefault("schedule_anchor", anchor)
        if _season(session) - anchor >= every:
            chance = 1.0
    if _unit(_salt(session), "expansion", _season(session)) > chance:
        return None
    used = set(gov.get("used_markets") or []) | set(exclude_markets or ())
    existing = {t["abbr"] for t in traits.values()}
    options = [m for m in CANDIDATE_MARKETS if m["abbr"] not in used and m["abbr"] not in existing]
    options.sort(key=lambda m: -m["value_b"])
    if not options:
        return None
    count = 2 if (n + 2 <= MAX_LEAGUE_TEAMS and n % 2 == 0 and len(options) >= 2) else 1
    top = options[: max(4, count)]
    rng = random.Random(_salt(session) * 31 + _season(session))
    cities = [dict(c, value_b=market_value_b(session, c)) for c in rng.sample(top, count)]
    avg_val = league_average_value(session)
    fee_b = round(avg_val * 0.8 * count, 2)
    return {"cities": cities, "fee_b": fee_b, "start_season": _season(session) + 2}


def _repeal_candidate(session: Any) -> Optional[str]:
    gov = ensure_governance(session)
    sy = _season(session)
    book = gov.get("rulebook") or {}
    eligible = [rid for rid, row in book.items() if sy - int(row.get("season") or sy) >= 2 and rid in RULE_BY_ID]
    if len(book) < REPEAL_MIN_RULES or not eligible:
        return None
    if _unit(_salt(session), "repeal", sy) > REPEAL_CHANCE:
        return None
    eligible.sort()
    return eligible[int(_unit(_salt(session), "repeal-pick", sy) * len(eligible)) % len(eligible)]


def _draw_rules(session: Any, k: int) -> List[Dict[str, Any]]:
    gov = ensure_governance(session)
    sy = _season(session)
    active = set((gov.get("rulebook") or {}).keys()) | set(gov.get("spent_rules") or [])
    proposed = gov.get("proposed_log") or {}
    # A rule is never put to the Board twice until every rule in the catalog has had
    # its vote (700 rules ≈ 140+ seasons of meetings). Only then do the oldest
    # proposals come back, longest-ago first.
    pool = [r for r in RULES if r["id"] not in active and r["id"] not in proposed]
    if len(pool) < k:
        stale = sorted(
            (r for r in RULES if r["id"] not in active and r["id"] in proposed
             and sy - int(proposed.get(r["id"]) or 0) >= RECENT_PROPOSAL_YEARS),
            key=lambda r: int(proposed.get(r["id"]) or 0),
        )
        pool = pool + stale[: max(k * 4, 20)]
    if len(pool) < k:
        pool = [r for r in RULES if r["id"] not in active]
    rng = random.Random(_salt(session) * 7919 + sy)
    rng.shuffle(pool)
    picked: List[Dict[str, Any]] = []
    per_cat: Dict[str, int] = {}
    for r in pool:
        if per_cat.get(r["cat"], 0) >= 1 and len(per_cat) < 4:
            continue  # spread the first picks across categories
        if per_cat.get(r["cat"], 0) >= 2:
            continue
        picked.append(r)
        per_cat[r["cat"]] = per_cat.get(r["cat"], 0) + 1
        if len(picked) >= k:
            break
    return picked


# ---------------------------------------------------------------------------
# Ballots
# ---------------------------------------------------------------------------


def _taste(session: Any, tid: str, pid: str) -> float:
    return (_unit(_salt(session), "gov-taste", tid, pid) * 2.0 - 1.0) * 0.12


def _rule_probability(rule: Dict[str, Any], t: Dict[str, Any], session: Any, tid: str, pid: str, *, repeal: bool = False) -> Tuple[float, Dict[str, str]]:
    """P(yes) and the reason each way. A repeal flips the rule's lean."""
    # A repeal reverses the rule's lean, but owners are less dug in on undoing a rule
    # they already live with (60% weight) and lean slightly toward the status quo.
    sign = -0.6 if repeal else 1.0
    th = _threshold_for(rule["cat"])
    base = float(rule.get("base") or 0.0)
    if repeal:
        p = 0.5 + REPEAL_STATUS_QUO + sign * base
    else:
        # Same head start relative to the bar for every category (it used to be
        # +2% for majority votes but +15% for two-thirds and three-quarters ones).
        p = THRESHOLDS.get(th, THRESHOLDS["two_thirds"])[0] + VETTING_MARGIN + base
    pos, neg = ("", 0.0), ("", 0.0)
    for trait, w in (rule.get("lean") or {}).items():
        if trait in t["traits"]:
            w = sign * float(w)
            p += w
            if trait == "all":
                continue  # league-wide lean moves the vote but is not a club's reason
            if w > pos[1]:
                pos = (_TRAIT_COPY.get(trait, trait), w)
            if -w > neg[1]:
                neg = (_TRAIT_COPY.get(trait, trait), -w)
    taste = _taste(session, tid, pid)
    p += taste
    pop = sign * base
    if pos[0]:
        yes_r = f"{pos[0]} — helps them"
    elif pop >= 0.08:
        yes_r = "Popular around the league"
    elif taste > 0.03:
        yes_r = "Owner likes it"
    else:
        yes_r = "Backs the committee"
    if neg[0]:
        no_r = f"{neg[0]} — hurts them"
    elif pop <= -0.05:
        no_r = "Unpopular with owners"
    elif taste < -0.03:
        no_r = "Owner unconvinced"
    else:
        no_r = "Sees no upside"
    return _clamp(p, 0.04, 0.96), {"yes": yes_r, "no": no_r}


def _recent_passed(session: Any, kind: str, within: int) -> List[int]:
    """Seasons in the last ``within`` years when the board approved this kind of move."""
    gov = ensure_governance(session)
    now = _season(session)
    return [int(h.get("season") or 0) for h in gov.get("history") or []
            if h.get("kind") == kind and h.get("status") == "passed" and now - int(h.get("season") or 0) < within]


def _relocation_probability(prop: Dict[str, Any], t: Dict[str, Any], session: Any, tid: str) -> Tuple[float, Dict[str, str]]:
    """Governors guard their markets: a relocation needs a real reason.

    Real boards approve moves rarely (Arizona to Utah was the first in decades) and
    only when a club is plainly failing where it is. A healthy club moving for a
    slightly bigger city, or a second move soon after the last one, should struggle.
    """
    if tid == str(prop["team_id"]):
        return 0.97, {"yes": "Their own club is moving", "no": "Their own club is moving"}
    p = 0.50
    yes_r, no_r = "Better market for the league", "Wants the club to stay put"
    mover = prop.get("_mover_traits") or {}
    if "poor" in (mover.get("traits") or ()):
        p += 0.20
        yes_r = "Club is losing money where it is"
    else:
        p -= 0.20
        no_r = "The club is healthy where it is"
    tier = str((prop.get("market") or {}).get("tier") or "").lower()
    if tier == "large":
        p += 0.06
        yes_r = yes_r if "losing" in yes_r else "Big new market lifts league revenue"
    elif tier == "small":
        p -= 0.10
        no_r = "Moving to another small market fixes nothing"
    recent = _recent_passed(session, "relocation", 6)
    if recent:
        p -= 0.15 * len(recent)
        no_r = "Too soon after the last relocation"
    if t["canadian"] and prop["market"]["abbr"] not in CANADA_ABBRS and mover.get("canadian"):
        p -= 0.25
        no_r = "Won't lose a Canadian club"
    if "small" in t["traits"]:
        p -= 0.08
        no_r = "Small markets fear they're next"
    p += gov_mod(session, "relocation_ease")
    p += _taste(session, tid, prop["id"])
    return _clamp(p, 0.05, 0.92), {"yes": yes_r, "no": no_r}


def _expansion_probability(prop: Dict[str, Any], t: Dict[str, Any], session: Any, tid: str) -> Tuple[float, Dict[str, str]]:
    """Expansion fees are tempting, but every new club dilutes TV money and talent.

    Owners weigh their fee share against the dilution: the more clubs the league
    already has, the more recently it expanded, and the more cities in one go, the
    harder the sell. Momentum helps, but no longer decides the vote by itself.
    """
    n_teams = max(1, int(prop.get("_n_teams") or 32))
    share_m = prop["fee_b"] * 1000.0 / n_teams
    n_new = len(prop.get("cities") or []) or 1
    p = 0.50
    yes_r, no_r = f"Takes the ${share_m:.0f}M fee share", "Doesn't want to share revenue"
    p += _clamp((share_m - 40.0) / 200.0, -0.10, 0.15)
    p += _clamp(gov_mod(session, "expansion_pressure") / 250.0, -0.16, 0.20)
    if n_teams > 32:
        p -= 0.06 * (n_teams - 32)
        no_r = f"Already {n_teams} clubs; the talent is thin enough"
    if n_new > 1:
        p -= 0.06 * (n_new - 1)
    recent = _recent_passed(session, "expansion", 10)
    if recent:
        gap = _season(session) - max(recent)
        p -= 0.28 if gap < 6 else 0.12
        no_r = "The last expansion clubs are still finding their feet"
    if "poor" in t["traits"]:
        p += 0.15
        yes_r = f"Needs the ${share_m:.0f}M fee share"
    if "large" in t["traits"]:
        p -= 0.08
        no_r = "Dilutes national TV money"
    if "contender" in t["traits"]:
        p -= 0.08
        no_r = "Won't lose a player to the expansion draft"
    p += _taste(session, tid, prop["id"])
    return _clamp(p, 0.05, 0.92), {"yes": yes_r, "no": no_r}


def _vote_climate(session: Any, prop: Dict[str, Any], mean: float, spread: float) -> Tuple[str, float]:
    """Room mood for one proposal, driven by how the room actually leans.

    Votes are correlated, not 31 coin flips, but the mood now follows the merits:
    a rule most clubs already like can snowball into a consensus, one most clubs
    dislike can die on arrival, and a rule that divides clubs along market lines
    splits harder. (Moods used to be rolled at random, so a rule that hurt most
    owners could still pass unanimously.)"""
    m = _unit(_salt(session), "climate", prop["id"])
    drift = (_unit(_salt(session), "climate-drift", prop["id"]) * 2.0 - 1.0) * 0.06
    need = int(prop.get("votes_needed") or 0)
    n = max(1, int(prop.get("_n_voters") or 1))
    bar = need / float(n + 1)
    if prop.get("kind") in ("relocation", "expansion"):
        if spread >= 0.10 and m < 0.5:
            return "split", drift
        return "normal", drift
    if mean >= bar + 0.10 and m < 0.45:
        return "consensus", 0.0
    if mean <= bar - 0.12 and m < 0.45:
        return "doa", 0.0
    if spread >= 0.10 and m < 0.55:
        return "split", drift * 0.5
    return "normal", drift


def _apply_climate(p: float, climate: str, drift: float, mean: float = 0.5) -> float:
    if climate == "consensus":
        return p + (1.0 - p) * 0.5
    if climate == "doa":
        return p * 0.5
    if climate == "split":
        return mean + (p - mean) * 1.8 + drift
    return p + drift


def _preroll_ballots(session: Any, prop: Dict[str, Any], traits: Dict[str, Dict[str, Any]]) -> None:
    uid = str(getattr(session, "user_team_id", "") or "")
    raw: List[Tuple[str, Dict[str, Any], float, Dict[str, str]]] = []
    for tid, t in traits.items():
        if tid == uid:
            continue
        if prop["kind"] == "relocation":
            p, reason = _relocation_probability(prop, t, session, tid)
        elif prop["kind"] == "expansion":
            p, reason = _expansion_probability(prop, t, session, tid)
        else:
            p, reason = _rule_probability(RULE_BY_ID[prop["rule_id"]], t, session, tid, prop["id"], repeal=prop["kind"] == "repeal")
        raw.append((tid, t, p, reason))
    ps = [r[2] for r in raw] or [0.5]
    mean = sum(ps) / len(ps)
    spread = (sum((x - mean) ** 2 for x in ps) / len(ps)) ** 0.5
    prop["_n_voters"] = len(raw)
    climate, drift = _vote_climate(session, prop, mean, spread)
    prop["_climate"] = climate
    ballots = []
    expected = 0.0
    for tid, t, p, reason in raw:
        if not (prop["kind"] == "relocation" and tid == str(prop.get("team_id"))):
            p = _clamp(_apply_climate(p, climate, drift, mean), 0.02, 0.98)
        u = _unit(_salt(session), "ballot", prop["id"], tid)
        ballots.append({"team_id": tid, "abbr": t["abbr"], "name": t["name"], "p": round(p, 4), "u": round(u, 4), "reason": reason, "lobbied": 0.0, "traits": sorted(t["traits"])})
        expected += p
    prop["_ballots"] = ballots
    prop["expected_yes_cpu"] = round(expected, 2)
    prop["room_mood"] = {
        "consensus": "The room is lining up behind it",
        "doa": "Most governors are against it",
        "split": "Clubs are split along market lines",
        "normal": "Opinions vary club by club",
    }.get(climate, "")


def _forecast(prop: Dict[str, Any]) -> Dict[str, Any]:
    exp = sum(b["p"] + b.get("lobbied", 0.0) for b in prop.get("_ballots") or [])
    need = int(prop["votes_needed"])
    # The user's own vote can add one.
    if exp >= need + 1.5:
        label = "Likely to pass"
    elif exp >= need - 1.5:
        label = "Toss-up"
    else:
        label = "Unlikely to pass"
    return {"expected_yes": round(exp, 1), "votes_needed": need, "label": label}


# ---------------------------------------------------------------------------
# Meeting lifecycle
# ---------------------------------------------------------------------------


def _effect_rows(effects: Dict[str, float], *, scaled: bool = False) -> List[Dict[str, Any]]:
    """Ballot / rulebook rows. ``scaled=True`` when ``effects`` already went through
    EFFECT_SCALE (the live modifier totals); catalog effects are scaled here so the
    ballot shows what the live systems will apply."""
    out = []
    for k, v in effects.items():
        label, kind, _consumer = EFFECT_SPECS.get(k, (k, "pts", ""))
        v = float(v) if (scaled or k in ABSOLUTE_EFFECTS) else _scaled(k, float(v))
        if kind == "minutes":
            txt = f"{int(v)}-minute OT"
        elif kind == "flag_ties":
            txt = "Abolished — ties stand" if v >= 0.5 else "Restored"
        elif kind == "points":
            txt = f"{int(v)} pt{'s' if int(v) != 1 else ''}"
        elif kind == "pct":
            txt = f"{v:+.1f}%" if abs(v) < 10 else f"{v:+.0f}%"
        elif kind == "pts":
            txt = f"{v:+.1f} pts"
        elif kind == "pts_yr":
            txt = f"{v:+.1f} pts / season"
        elif kind == "pts_raw":
            txt = f"{v:+.1f}"
        elif kind == "money":
            txt = f"{'+' if v >= 0 else '−'}${abs(v):.2f}M".replace(".00M", "M")
        elif kind == "money_k":
            txt = f"{'+' if v >= 0 else '−'}${abs(v) * 1000:.0f}K"
        elif kind == "pct_of_cap":
            if k == "max_salary_pct":
                txt = f"{(0.20 + v) * 100:.0f}% of cap"
            else:
                txt = f"{v * 100:+.1f}% of cap"
        elif kind == "years":
            txt = f"{int(v)} years"
        elif kind == "count":
            txt = f"{3 + int(v)} slots" if k == "retention_slots" else f"{int(v):+d}"
        elif kind == "risk":
            txt = f"{v:+.2f}"
        else:
            txt = f"{v:+g}"
        tone, note = _effect_tone(k, v)
        out.append({"key": k, "label": label, "value": v, "text": txt, "tone": tone, "note": note})
    return out


# Which direction is good for clubs, with the knock-on effect players will feel.
_EFFECT_TONE: Dict[str, Tuple[int, str, str]] = {
    # key: (sign that is good, why it helps, why it hurts)
    "rev_all": (1, "More money for every club", "Every club takes in less"),
    "rev_small": (1, "Small markets earn more", "Small markets earn less"),
    "rev_large": (1, "Big markets earn more", "Big markets earn less"),
    "revenue_share": (1, "Bigger safety net for small markets", "Less help for small markets"),
    "opex": (-1, "Cheaper to run a club", "Higher operating costs eat profit"),
    "star_rev": (1, "Stars sell more tickets and jerseys", "Star players draw less money"),
    "fan": (1, "Fans like it: attendance and gate revenue rise", "Fan backlash: attendance and gate revenue can drop"),
    "playoff_rev": (1, "Bigger playoff paydays", "Smaller playoff paydays"),
    "value_growth": (1, "Franchise values grow faster", "Franchise values grow slower"),
    "cap_growth": (0, "Cap rises faster: more room to spend", "Cap rises slower: tighter budgets"),
    "cap_adjust_m": (0, "Higher cap next season: more room", "Lower cap next season: cap crunches"),
    "floor_ratio": (0, "Lower floor: cheap rosters allowed", "Higher floor: budget clubs must spend"),
    "max_salary_pct": (0, "Stars can earn more", "Max contracts get capped lower"),
    "max_term_own": (0, "Longer re-sign terms allowed", "Shorter re-sign terms"),
    "max_term_ufa": (0, "Longer free-agent terms allowed", "Shorter free-agent terms"),
    "bonus_pct": (0, "More signing-bonus room", "Less signing-bonus room"),
    "bonus_floor_m": (0, "Bonuses open to more clubs", "Fewer clubs can pay bonuses"),
    "min_salary_m": (0, "Depth players paid more", "Cheaper depth players"),
    "fa_bonus_demand": (-1, "Fewer free agents demand bonuses", "More free agents demand bonuses: poor clubs lose out"),
    "retention_slots": (0, "More retained-salary trades possible", "Fewer retention slots"),
    "retention_max_pct": (0, "Teams can retain more salary in trades", "Less salary retention in trades"),
    "trade_volume": (0, "Busier trade market", "Quieter trade market"),
    "trade_demand_rate": (-1, "Fewer trade requests", "More players demand trades"),
    "scouting_budget": (1, "Bigger scouting budgets", "Smaller scouting budgets"),
    "injury_rate": (-1, "Fewer injuries", "More injuries"),
    "relocation_ease": (-1, "Struggling clubs stay put", "Easier for clubs to relocate"),
    "expansion_pressure": (0, "Expansion momentum builds", "Expansion momentum fades"),
    "scoring_pace": (0, "More goals league-wide: skill teams gain", "Fewer goals: tight, defensive hockey pays"),
    "penalty_rate": (0, "More penalties: power plays decide more games", "Fewer penalties: more 5-on-5 hockey"),
    "pp_efficiency": (0, "Power plays score more: special teams matter more", "Power plays score less"),
    "home_ice": (0, "Bigger home-ice edge", "Smaller home-ice edge"),
    "ot_minutes": (0, "Overtime length changes how many games reach a shootout", "Overtime length changes"),
    "ties_allowed": (0, "Shootout gone: games tied after OT end as ties", "Shootout restored"),
    "pts_reg_win": (0, "Regulation wins worth more: chasing points late changes", "Regulation win points change"),
    "pts_ot_win": (0, "OT / shootout wins pay differently", "OT / shootout wins pay differently"),
    "pts_otl": (0, "Loser point changes the playoff race", "Loser point changes the playoff race"),
}


def _effect_tone(key: str, value: float) -> Tuple[str, str]:
    good, helps, hurts = _EFFECT_TONE.get(key, (0, "Changes the landscape", "Changes the landscape"))
    if good == 0:
        # Neutral keys: the first text describes a rise, the second a drop.
        return "mixed", (helps if value >= 0 else hurts)
    # The note follows the tone. Picking it by sign alone told owners a cost increase
    # made clubs "cheaper to run" and fewer injuries meant "more injuries".
    pro = (value >= 0) == (good > 0)
    return ("pro" if pro else "con"), (helps if pro else hurts)


def _pros_cons(effects: Dict[str, float]) -> Dict[str, List[str]]:
    rows = [(k, float(v)) + _effect_tone(k, float(v)) for k, v in (effects or {}).items()]
    pros = [n for _k, _v, t, n in rows if t == "pro"]
    cons = [n for _k, _v, t, n in rows if t == "con"]
    mixed = [n for _k, _v, t, n in rows if t == "mixed"]
    keys = {k: v for k, v, _t, _n in rows}
    # Second-order trade-offs owners argue about.
    if keys.get("fan", 0) < 0 and any(keys.get(k, 0) > 0 for k in ("rev_all", "rev_small", "rev_large", "star_rev")):
        cons.append("Fan backlash can eat into the new revenue over time")
    if keys.get("fan", 0) > 0 and keys.get("opex", 0) > 0:
        pros.append("Fans love it even though it costs money to run")
    if keys.get("floor_ratio", 0) > 0:
        cons.append("Low-revenue clubs may be forced into bad contracts to reach the floor")
    if keys.get("revenue_share", 0) > 0:
        cons.append("Big-market owners pay for it")
    if keys.get("trade_volume", 0) > 0:
        mixed.append("More trades: more chances to buy, and more rivals buying")
    return {"pros": pros, "cons": cons, "mixed": mixed}


def _repeal_effect_rows(effects: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What striking a rule really does. One-time cap bumps were already paid out and
    are not reversed; rules that SET a value (term limits, OT format, points) fall
    back to the default."""
    rows = _effect_rows({k: -float(v) for k, v in effects.items() if k not in ABSOLUTE_EFFECTS and k != "cap_adjust_m"})
    for k in [k for k in effects if k in ABSOLUTE_EFFECTS]:
        label = EFFECT_SPECS.get(k, (k, "", ""))[0]
        default = ABSOLUTE_DEFAULTS.get(k)
        if k in ("max_term_own", "max_term_ufa"):
            txt = "Back to the CBA limit"
        elif default is not None:
            txt = f"Back to the NHL default ({_effect_rows({k: default})[0]['text']})"
        else:
            txt = "Back to the default"
        rows.append({"key": k, "label": label, "value": 0, "text": txt, "tone": "mixed", "note": "Reverts to the league default"})
    return rows


def _new_rule_proposal(session: Any, rule: Dict[str, Any], idx: int, n_teams: int) -> Dict[str, Any]:
    th = _threshold_for(rule["cat"])
    return {
        "id": f"{_season(session)}-{idx}-{rule['id']}",
        "kind": "rule",
        "rule_id": rule["id"],
        "category": rule["cat"],
        "category_label": CATEGORIES[rule["cat"]]["label"],
        "title": rule["title"],
        "summary": rule["summary"],
        "effects": _effect_rows(rule["effects"]),
        "pros_cons": _pros_cons(rule["effects"]),
        "threshold": th,
        "threshold_label": THRESHOLDS[th][1],
        "votes_needed": _votes_needed(th, n_teams),
        "status": "pending",
        "user_vote": None,
        "lobbied": None,
    }


def open_board_meeting(session: Any) -> Dict[str, Any]:
    """Create (once per offseason) the meeting: values/spikes/budget pass + 5 proposals."""
    gov = ensure_governance(session)
    sy = _season(session)
    key = str(sy)
    meeting = (gov.get("meetings") or {}).get(key)
    if isinstance(meeting, dict) and meeting.get("proposals"):
        return build_governance_payload(session)

    rows = _team_rows(session)
    ensure_franchise_values(session, rows)
    # Star spikes first: franchise values read them (they used to be computed after,
    # so the "Star spike" value driver never fired).
    compute_star_spikes(session, rows)
    advance_franchise_values(session, rows)
    apply_revenue_scouting_budget(session, rows)
    traits = team_traits(session, rows)
    n_teams = len(traits)
    _record_profit_log(session, traits)

    proposals: List[Dict[str, Any]] = []
    reloc = _relocation_candidate(session, traits)
    # The two bids must not chase the same city (two Houston clubs).
    expn = _expansion_candidate(session, traits, exclude_markets={reloc["market"]["abbr"]} if reloc else None)
    repeal = _repeal_candidate(session)
    n_rules = PROPOSALS_PER_MEETING - (1 if reloc else 0) - (1 if expn else 0) - (1 if repeal else 0)
    for i, rule in enumerate(_draw_rules(session, n_rules)):
        proposals.append(_new_rule_proposal(session, rule, i, n_teams))
        gov["proposed_log"][rule["id"]] = sy
    if repeal:
        rule = RULE_BY_ID[repeal]
        row = gov["rulebook"].get(repeal) or {}
        th = _threshold_for(rule["cat"])
        proposals.append({
            "id": f"{sy}-repeal-{repeal}",
            "kind": "repeal",
            "rule_id": repeal,
            "category": rule["cat"],
            "category_label": f"Repeal · {CATEGORIES[rule['cat']]['label']}",
            "title": f"Repeal: {rule['title']}",
            "summary": f"Strike the {row.get('season', '')} rule from the books. {rule['summary']}",
            "effects": _repeal_effect_rows(row.get("effects") or rule["effects"]),
            "pros_cons": _pros_cons({k: -float(v) for k, v in (row.get("effects") or rule["effects"]).items() if k not in ABSOLUTE_EFFECTS and k != "cap_adjust_m"}),
            "threshold": th,
            "threshold_label": THRESHOLDS[th][1],
            "votes_needed": _votes_needed(th, n_teams),
            "status": "pending",
            "user_vote": None,
            "lobbied": None,
        })
    if reloc:
        mover = traits[reloc["team_id"]]
        m = reloc["market"]
        team = session.team_by_id.get(reloc["team_id"])
        nick = str(getattr(team, "name", "") or "")
        proposals.append({
            "id": f"{sy}-reloc-{reloc['team_id']}",
            "kind": "relocation",
            "team_id": reloc["team_id"],
            "market": m,
            "category": "structure",
            "category_label": "Relocation",
            "title": f"Relocate the {mover['name']} to {m['city']}",
            "summary": (
                f"{mover['name']} ({mover['abbr']}) lost ${abs(mover['profit']):.1f}M this year"
                + (
                    f" after losing ${abs(_prior_profit(session, reloc['team_id']) or 0):.1f}M the year before"
                    if (_prior_profit(session, reloc["team_id"]) or 0) < 0 else ""
                )
            ) + f"; ownership applies to move to {m['city']} as the {m['city']} {nick}"
            + (f" (and switch to the {m.get('conference')} Conference)" if m.get("conference") and m.get("conference") != mover.get("conference") else "")
            + ".",
            "effects": [
                {"key": "relocation", "label": "Moves to", "value": 0, "text": f"{m['city']} ({m['tier']} market)"},
                {"key": "demands", "label": "Player reaction", "value": 0, "text": "Some veterans may demand trades"},
                {"key": "fans", "label": "New market", "value": 0, "text": "Fan honeymoon + revenue bump for 2 seasons"},
            ],
            "pros_cons": {
                "pros": [f"Move from a struggling market to {m['city']} ({m['tier']})", "Relocated club's value and revenue jump", "New-market honeymoon for two seasons"],
                "cons": [f"{mover.get('city') or 'The old city'} loses its team: fan backlash league-wide", "Veterans on the moving club may demand trades", "Sets a precedent: other small markets feel the heat"],
                "mixed": [],
            },
            "threshold": "three_quarters",
            "threshold_label": THRESHOLDS["three_quarters"][1],
            "votes_needed": _votes_needed("three_quarters", n_teams),
            "status": "pending",
            "user_vote": None,
            "lobbied": None,
            "_mover_traits": {"traits": sorted(mover["traits"]), "canadian": mover["canadian"]},
        })
    if expn:
        names = " & ".join(c["city"] for c in expn["cities"])
        share_m = expn["fee_b"] * 1000.0 / max(1, n_teams)
        n_new = len(expn["cities"])
        loses = "one unprotected player" if n_new == 1 else f"{n_new} unprotected players (one to each new club)"
        proposals.append({
            "id": f"{sy}-expand-{'-'.join(c['abbr'] for c in expn['cities'])}",
            "kind": "expansion",
            "cities": expn["cities"],
            "fee_b": expn["fee_b"],
            "start_season": expn["start_season"],
            "category": "structure",
            "category_label": "Expansion",
            "title": f"Expand to {n_teams + len(expn['cities'])} clubs: {names}",
            "summary": (
                f"{names} join for the {expn['start_season']}-{(expn['start_season'] + 1) % 100:02d} season. "
                f"Expansion fee ${expn['fee_b']:.2f}B (≈${share_m:.0f}M to every current club). "
                f"The expansion draft is held at the {expn['start_season'] - 1} Board of Governors meeting, before that year's Entry Draft: "
                f"every club protects 7 forwards, 3 defensemen and 1 goalie and loses {loses}."
            ),
            "effects": [
                {"key": "fee", "label": "Fee share per club", "value": share_m, "text": f"+${share_m:.0f}M once"},
                {"key": "teams", "label": "League size", "value": len(expn["cities"]), "text": f"{n_teams} → {n_teams + len(expn['cities'])} clubs"},
                {"key": "draft", "label": "Expansion draft", "value": 0, "text": f"{expn['start_season'] - 1} offseason · protect 7F / 3D / 1G; lose {loses}"},
            ],
            "pros_cons": {
                "pros": [f"Every club gets ≈${share_m:.0f}M in fee money", "Bigger league: more markets and TV money long term"],
                "cons": [f"Every club loses {loses} in the expansion draft", "National TV money split more ways", "New clubs join the race for free agents"],
                "mixed": [],
            },
            "threshold": "three_quarters",
            "threshold_label": THRESHOLDS["three_quarters"][1],
            "votes_needed": _votes_needed("three_quarters", n_teams),
            "status": "pending",
            "user_vote": None,
            "lobbied": None,
            "_n_teams": n_teams,
        })

    for prop in proposals:
        _preroll_ballots(session, prop, traits)
        prop["forecast"] = _forecast(prop)

    gov["meetings"][key] = {
        "season": sy,
        "status": "open",
        "proposals": proposals,
        "lobby_tokens": LOBBY_TOKENS_PER_MEETING,
        "n_teams": n_teams,
    }
    sync_governance_to_league(session)
    return build_governance_payload(session)


def _current_meeting(session: Any) -> Optional[Dict[str, Any]]:
    gov = ensure_governance(session)
    return (gov.get("meetings") or {}).get(str(_season(session)))


def user_fan_confidence(session: Any) -> float:
    try:
        from services.franchise_sim import _ensure_team_fan_profile

        prof = _ensure_team_fan_profile(session, str(getattr(session, "user_team_id", "") or ""))
        return _sf(prof.get("fan_confidence"), 55.0)
    except Exception:
        return 55.0


def fan_lobby_mult(fan: float) -> float:
    """x0.80 with hostile fans, up to x1.20 when the fan base is electric."""
    return 0.80 + 0.40 * _clamp((float(fan) - 40.0) / 50.0, 0.0, 1.0)


def lobby_proposal(session: Any, proposal_id: str, side: str) -> Dict[str, Any]:
    """Spend a lobby token: sway the most persuadable like-minded governors."""
    meeting = _current_meeting(session)
    if not meeting:
        raise ValueError("No Board of Governors meeting is open")
    if int(meeting.get("lobby_tokens") or 0) <= 0:
        raise ValueError("No lobbying calls left this meeting")
    prop = next((p for p in meeting["proposals"] if p["id"] == proposal_id), None)
    if prop is None:
        raise ValueError("Unknown proposal")
    if prop["status"] != "pending":
        raise ValueError("That proposal has already been decided")
    if prop.get("lobbied"):
        raise ValueError("You already lobbied on this proposal")
    want_yes = str(side).lower() in ("yes", "agree", "for")
    uid = str(getattr(session, "user_team_id", "") or "")
    try:
        my = set(_user_traits_cache(session))
    except Exception:
        my = set()
    my.discard("all")  # every club shares it; it said nothing about who to call
    scored = []
    for b in prop["_ballots"]:
        p = float(b["p"])
        # Skip governors already firmly on your side: a call there changes nothing.
        if (want_yes and p >= 0.85) or ((not want_yes) and p <= 0.15):
            continue
        overlap = len(my & (set(b.get("traits") or ()) - {"all"}))
        persuadable = 1.0 - abs(p - 0.5) * 2.0
        scored.append((persuadable + overlap * 0.15, b))
    scored.sort(key=lambda s: -s[0])
    # A club whose fans are behind it carries weight in the room: governors take the
    # call more seriously (bigger swing) and an electric market reaches one more.
    fan = user_fan_confidence(session)
    swing = LOBBY_SWING * fan_lobby_mult(fan)
    targets = LOBBY_TARGETS + (1 if fan >= 80 else 0)
    swayed = []
    for _score, b in scored[:targets]:
        b["lobbied"] = round(swing if want_yes else -swing, 3)
        swayed.append(b["abbr"])
    prop["lobbied"] = {"side": "yes" if want_yes else "no", "teams": swayed, "fan_confidence": round(fan, 1)}
    meeting["lobby_tokens"] = int(meeting.get("lobby_tokens") or 0) - 1
    prop["forecast"] = _forecast(prop)
    return build_governance_payload(session)


def _user_traits_cache(session: Any) -> List[str]:
    meeting = _current_meeting(session) or {}
    cached = meeting.get("_user_traits")
    if isinstance(cached, list):
        return cached
    uid = str(getattr(session, "user_team_id", "") or "")
    out = sorted((team_traits(session).get(uid) or {}).get("traits") or ())
    meeting["_user_traits"] = out
    return out


def cast_vote(session: Any, proposal_id: str, vote: str) -> Dict[str, Any]:
    """Record the user's vote, reveal the Board's ballots, apply the outcome."""
    meeting = _current_meeting(session)
    if not meeting:
        raise ValueError("No Board of Governors meeting is open")
    prop = next((p for p in meeting["proposals"] if p["id"] == proposal_id), None)
    if prop is None:
        raise ValueError("Unknown proposal")
    if prop["status"] != "pending":
        return {"proposal": public_proposal(prop, reveal=True), "governance": build_governance_payload(session)}
    v = str(vote or "").lower()
    v = "yes" if v in ("yes", "agree", "for", "y") else ("no" if v in ("no", "disagree", "against", "n") else "abstain")
    _decide(session, meeting, prop, v)
    return {"proposal": public_proposal(prop, reveal=True), "governance": build_governance_payload(session)}


def _decide(session: Any, meeting: Dict[str, Any], prop: Dict[str, Any], user_vote: str) -> None:
    uid = str(getattr(session, "user_team_id", "") or "")
    user_team = (getattr(session, "team_by_id", None) or {}).get(uid)
    yes = no = 0
    ballots_out = []
    for b in prop["_ballots"]:
        p = _clamp(b["p"] + b.get("lobbied", 0.0), 0.01, 0.99)
        cast = "yes" if b["u"] < p else "no"
        if cast == "yes":
            yes += 1
        else:
            no += 1
        why = b["reason"].get(cast) if isinstance(b.get("reason"), dict) else b.get("reason")
        climate = str(prop.get("_climate") or "")
        if climate == "consensus" and cast == "yes" and why in ("Backs the committee", "Owner likes it"):
            why = "Went with the room"
        elif climate == "doa" and cast == "no" and why in ("Sees no upside", "Owner unconvinced"):
            why = "Went with the room"
        if b.get("lobbied") and ((b["lobbied"] > 0) == (cast == "yes")):
            # Only credit the call when it actually flipped the vote.
            if (b["u"] < _clamp(b["p"], 0.01, 0.99)) != (cast == "yes"):
                why = "Persuaded by your call"
        ballots_out.append({"team_id": b["team_id"], "abbr": b["abbr"], "name": b["name"], "vote": cast, "reason": why, "lobbied": bool(b.get("lobbied"))})
    abstain = 0
    if user_vote == "yes":
        yes += 1
    elif user_vote == "no":
        no += 1
    else:
        abstain += 1
    ballots_out.insert(0, {"team_id": uid, "abbr": _team_abbr(user_team), "name": _team_name(user_team), "vote": user_vote, "reason": "Your vote", "is_user": True})
    passed = yes >= int(prop["votes_needed"])
    prop["status"] = "passed" if passed else "failed"
    prop["user_vote"] = user_vote
    prop["tally"] = {"yes": yes, "no": no, "abstain": abstain}
    prop["ballots"] = ballots_out
    gov = ensure_governance(session)
    gov["history"].append({
        "season": _season(session),
        "proposal_id": prop["id"],
        "kind": prop["kind"],
        "rule_id": prop.get("rule_id"),
        "title": prop["title"],
        "category_label": prop["category_label"],
        "status": prop["status"],
        "yes": yes,
        "no": no,
        "votes_needed": prop["votes_needed"],
        "user_vote": user_vote,
    })
    gov["history"] = gov["history"][-120:]
    if passed:
        _apply_passed(session, prop)
        if prop["kind"] in ("rule", "repeal"):
            _news(session, f"Board passes: {prop['title']}", f"{prop['summary']} Vote: {yes}-{no} ({prop['votes_needed']} needed).", kind="rule_change")
        elif prop["kind"] == "expansion":
            _news(session, f"NHL expands: {', '.join(c['city'] for c in prop['cities'])} approved", prop["summary"], kind="expansion")
    if all(p["status"] != "pending" for p in meeting["proposals"]):
        meeting["status"] = "complete"
    try:
        from services.league_operations import invalidate_league_ops_cache

        invalidate_league_ops_cache(session)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _apply_passed(session: Any, prop: Dict[str, Any]) -> None:
    gov = ensure_governance(session)
    sy = _season(session)
    if prop["kind"] == "rule":
        rule = RULE_BY_ID[prop["rule_id"]]
        effects = dict(rule["effects"])
        # A newer absolute rule supersedes older ones setting the same key.
        for k in [k for k in effects if k in ABSOLUTE_EFFECTS]:
            for rid, row in list(gov["rulebook"].items()):
                if k in (row.get("effects") or {}):
                    row["effects"] = {kk: vv for kk, vv in row["effects"].items() if kk != k}
                    if not row["effects"]:
                        gov["rulebook"].pop(rid, None)
        gov["rulebook"][rule["id"]] = {
            "season": sy,
            "title": rule["title"],
            "category": rule["cat"],
            "category_label": CATEGORIES[rule["cat"]]["label"],
            "summary": rule["summary"],
            "effects": effects,
        }
        if effects.get("cap_adjust_m"):
            gov["pending_cap_adjust_m"] = round(_sf(gov.get("pending_cap_adjust_m")) + float(effects["cap_adjust_m"]), 3)
        sync_governance_to_league(session)
        if "scouting_budget" in effects:
            apply_revenue_scouting_budget(session)  # takes effect now, not next summer
        return
    if prop["kind"] == "repeal":
        row = gov["rulebook"].pop(str(prop["rule_id"]), None) or {}
        sync_governance_to_league(session)
        if "scouting_budget" in (row.get("effects") or {}):
            apply_revenue_scouting_budget(session)
        return
    if prop["kind"] == "relocation":
        relocate_team(session, str(prop["team_id"]), prop["market"])
        return
    if prop["kind"] == "expansion":
        exp = gov["expansion"]
        n = int(prop.get("_n_teams") or 32)
        exp["pending"] = [{
            "cities": prop["cities"],
            "fee_b": prop["fee_b"],
            "start_season": prop["start_season"],
            "approved_season": sy,
        }]
        for c in prop["cities"]:
            gov["used_markets"].append(c["abbr"])
        # Fee is paid to existing owners next season.
        gov["fee_shares"][str(sy + 1)] = round(prop["fee_b"] * 1000.0 / max(1, n), 1)


def finalize_board_meeting(session: Any) -> Dict[str, Any]:
    """Close the meeting; any undecided proposals are voted on with the user abstaining."""
    meeting = _current_meeting(session)
    if meeting is None:
        open_board_meeting(session)
        meeting = _current_meeting(session)
    for prop in list(meeting.get("proposals") or []):
        if prop["status"] == "pending":
            _decide(session, meeting, prop, "abstain")
    meeting["status"] = "complete"
    sync_governance_to_league(session)
    # Clubs approved at an earlier meeting join now: the expansion draft runs at the
    # end of this meeting, before the lottery, Entry Draft and free agency.
    try:
        joined = run_pending_expansion(session, _season(session) + 1)
        if joined:
            announce_expansion(session, joined, _season(session) + 1)
    except Exception:
        _swallowed_log.exception("expansion draft failed")
    return build_governance_payload(session)


def announce_expansion(session: Any, joined: List[Dict[str, Any]], season_year: int) -> None:
    """Breaking-news popup + timeline line for clubs that just joined."""
    try:
        session.expansion_joined_payload = joined
        from services.franchise_sim import _append_showcase_popup

        names = ", ".join(
            f"{row.get('city') or ''} {row.get('name') or ''}".strip() or str(row.get("abbr") or "Expansion club")
            for row in joined if isinstance(row, dict)
        ) or "New clubs"
        taken = sum(int(row.get("players") or 0) for row in joined if isinstance(row, dict))
        uid = str(getattr(session, "user_team_id", "") or "")
        drafts = (ensure_governance(session).get("expansion") or {}).get("drafts") or []
        lost = [pk for pk in (drafts[-1].get("picks") if drafts else []) or [] if str(pk.get("from_team_id") or "") == uid]
        lost_txt = (" You lost " + ", ".join(f"{pk.get('player')} ({pk.get('team')})" for pk in lost) + ".") if lost else ""
        _append_showcase_popup(session, f"expansion:{int(season_year)}", {
            "kind": "breaking_news",
            "source_label": "Board of Governors",
            "theme": "info",
            "headline": "Expansion draft is complete",
            "summary": (
                f"{names} join the league for {season_year}-{(season_year + 1) % 100:02d}. "
                f"{taken or 'Several'} players changed teams in the expansion draft.{lost_txt} "
                "The new clubs take part in free agency this summer and are on next season's schedule."
            ),
        })
        session.timeline.append(f"Expansion: {names} joined the league.")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def consume_pending_cap_adjust(session: Any) -> float:
    """Called by the cap rollover: returns and clears one-time cap adjustments."""
    gov = ensure_governance(session)
    adj = round(_sf(gov.get("pending_cap_adjust_m")), 3)
    if adj:
        gov["applied_cap_adjust"].append({"season": _season(session) + 1, "adjust_m": adj})
    gov["pending_cap_adjust_m"] = 0.0
    # One-time cap bumps are spent once applied: drop that effect from the rulebook so
    # it no longer reads as "in force" (or as reversible by a repeal). Rules with
    # nothing else left are retired; they still never come back to the ballot
    # before the rest of the catalog has had its vote.
    spent = gov.setdefault("spent_rules", [])
    for rid, row in list((gov.get("rulebook") or {}).items()):
        eff = row.get("effects") or {}
        if "cap_adjust_m" not in eff:
            continue
        rest = {k: v for k, v in eff.items() if k != "cap_adjust_m"}
        if rest:
            row["effects"] = rest
            row["applied_cap_adjust_m"] = eff["cap_adjust_m"]
        else:
            gov["rulebook"].pop(rid, None)
            if rid not in spent:
                spent.append(rid)
    sync_governance_to_league(session)
    return adj


def pending_cap_adjust(session: Any) -> float:
    return _sf(ensure_governance(session).get("pending_cap_adjust_m"))


# ---------------------------------------------------------------------------
# Franchise values, star spikes, scouting budget
# ---------------------------------------------------------------------------


def ensure_franchise_values(session: Any, rows: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    gov = ensure_governance(session)
    vals = gov["franchise_values"]
    try:
        from services.league_operations import _TEAM_VALUE_B
    except Exception:
        _TEAM_VALUE_B = {}
    missing = [tid for tid in (getattr(session, "team_by_id", None) or {}) if str(tid) not in vals]
    if not missing:
        return vals
    rows = rows if rows is not None else _team_rows(session)
    for tid in missing:
        team = session.team_by_id.get(tid)
        abbr = _team_abbr(team)
        v = _TEAM_VALUE_B.get(abbr)
        if v is None:
            v = round(_sf((rows.get(str(tid)) or {}).get("revenue"), 185.0) * 0.0115, 2)
        vals[str(tid)] = {"value_b": float(v), "history": [{"season": _season(session), "value_b": float(v), "change_pct": 0.0, "drivers": ["Opening valuation"]}]}
    return vals


def league_average_value(session: Any) -> float:
    vals = [_sf(v.get("value_b")) for v in (ensure_governance(session).get("franchise_values") or {}).values() if isinstance(v, dict)]
    return (sum(vals) / len(vals)) if vals else 2.55


def team_value_b(session: Any, team_id: str) -> Optional[float]:
    row = (ensure_governance(session).get("franchise_values") or {}).get(str(team_id))
    return _sf(row.get("value_b")) if isinstance(row, dict) else None


def advance_franchise_values(session: Any, rows: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Once per offseason: values move with profit, winning, stars, fans and market booms."""
    gov = ensure_governance(session)
    sy = _season(session)
    if sy in gov["values_advanced_for"]:
        return
    rows = rows if rows is not None else _team_rows(session)
    ensure_franchise_values(session, rows)
    pct = _standing_ranks(session)
    champ = str(getattr(session, "champion_id", "") or getattr(session, "stanley_cup_winner", "") or "")
    vg = gov_mod(session, "value_growth") / 100.0
    rng = random.Random(_salt(session) * 104729 + sy)
    spikes = (gov.get("star_spikes") or {}).get(str(sy + 1)) or {}
    for tid, row in (gov["franchise_values"] or {}).items():
        r = rows.get(str(tid)) or {}
        rev = max(1.0, _sf(r.get("revenue"), 180.0))
        profit = _sf(r.get("profit"))
        win = pct.get(str(tid), 0.5)
        fan = _sf(r.get("fan_sentiment"), 55.0)
        tier = str(r.get("market_tier_key") or "medium")
        g = 0.04 + vg
        drivers: List[str] = []
        margin = profit / rev
        g += 0.20 * margin
        if margin >= 0.15:
            drivers.append("Strong profit")
        elif margin < 0:
            drivers.append("Operating losses")
        g += 0.20 * (win - 0.5)
        if win >= 0.6:
            drivers.append("Winning")
        if champ and champ == str(tid):
            g += 0.05
            drivers.append("Stanley Cup")
        g += (fan - 55.0) / 100.0 * FAN_VALUE_COEF
        if fan >= 80:
            drivers.append("Passionate fan base")
        elif fan < 45:
            drivers.append("Fan apathy")
        stars = _sf(r.get("star_power"))
        spike = sum(_sf(s.get("m")) for s in spikes.get(str(tid), []))
        if spike:
            g += min(0.06, spike / rev * 0.5)
            drivers.append("Star spike")
        if tier == "small" and (win >= 0.58 or (champ and champ == str(tid))) and (stars >= 1.0 or spike >= 6.0):
            boom = rng.uniform(0.06, 0.18)
            g += boom
            drivers.append("Market boom")
        g += rng.uniform(-0.02, 0.02)
        g = _clamp(g, -0.12, 0.30)
        new_v = round(_sf(row.get("value_b")) * (1.0 + g), 3)
        row["value_b"] = new_v
        row.setdefault("history", []).append({"season": sy, "value_b": new_v, "change_pct": round(g * 100.0, 1), "drivers": drivers[:3]})
        row["history"] = row["history"][-20:]
    gov["values_advanced_for"].append(sy)


def compute_star_spikes(session: Any, rows: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Top performers of the season just played lift their club's revenue next season.
    Small markets feel it most — that is how they get the chance to boom."""
    gov = ensure_governance(session)
    nxt = str(_season(session) + 1)
    if nxt in gov["star_spikes"]:
        return
    stats = getattr(session, "player_season_stats", None) or {}
    if not isinstance(stats, dict) or not stats:
        gov["star_spikes"][nxt] = {}
        return
    rows = rows if rows is not None else _team_rows(session)
    star_mult = 1.0 + gov_mod(session, "star_rev") / 100.0
    tier_mult = {"small": 1.4, "medium": 1.15, "large": 0.9}
    skaters = [r for r in stats.values() if isinstance(r, dict) and str(r.get("position", "")).upper() not in ("G",) and int(r.get("gp") or 0) >= 20]
    goalies = [r for r in stats.values() if isinstance(r, dict) and str(r.get("position", "")).upper() == "G" and int(r.get("gp") or 0) >= 20]
    skaters.sort(key=lambda r: (-int(r.get("pts") or 0), -int(r.get("g") or 0)))
    goalies.sort(key=lambda r: -int(r.get("w") or 0))
    out: Dict[str, List[Dict[str, Any]]] = {}

    def _add(r: Dict[str, Any], base_m: float, reason: str) -> None:
        tid = str(r.get("team_id") or "")
        if not tid or tid not in (getattr(session, "team_by_id", None) or {}):
            return
        tier = str((rows.get(tid) or {}).get("market_tier_key") or "medium")
        m = round(base_m * tier_mult.get(tier, 1.0) * star_mult, 1)
        out.setdefault(tid, []).append({"player": r.get("name"), "player_id": r.get("player_id"), "m": m, "reason": reason})

    for i, r in enumerate(skaters[:10]):
        base = 16.0 if i == 0 else 11.0 if i < 3 else 7.0 if i < 6 else 4.0
        _add(r, base, f"#{i + 1} in league scoring ({int(r.get('pts') or 0)} pts)")
    for i, r in enumerate(goalies[:3]):
        _add(r, (6.0, 4.0, 3.0)[i], f"#{i + 1} in goalie wins ({int(r.get('w') or 0)} W)")
    gov["star_spikes"][nxt] = out


def star_spike_m(session: Any, team_id: str, season: Optional[int] = None) -> Tuple[float, List[Dict[str, Any]]]:
    gov = ensure_governance(session)
    sy = str(int(season if season is not None else _season(session)))
    rows = ((gov.get("star_spikes") or {}).get(sy) or {}).get(str(team_id)) or []
    return round(sum(_sf(r.get("m")) for r in rows), 1), rows


def honeymoon_m(session: Any, team_id: str, season: Optional[int] = None) -> float:
    gov = ensure_governance(session)
    sy = str(int(season if season is not None else _season(session)))
    return _sf(((gov.get("honeymoon") or {}).get(str(team_id)) or {}).get(sy))


def fee_share_m(session: Any, season: Optional[int] = None) -> float:
    gov = ensure_governance(session)
    sy = str(int(season if season is not None else _season(session)))
    return _sf((gov.get("fee_shares") or {}).get(sy))


DEFAULT_SCOUTING_BUDGET_DOLLARS = 2_500_000


def revenue_scouting_budget(session: Any, team_id: str, rows: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    """Scouting budget funded by revenue: $2.5M at a ~$185M club, more for rich clubs."""
    if rows is None:
        try:
            from services.league_operations import calculate_team_revenue

            team = session.team_by_id.get(str(team_id))
            rev = _sf(calculate_team_revenue(session, team, str(team_id), annual=True).get("revenue"), 185.0)
        except Exception:
            rev = 185.0
    else:
        rev = _sf((rows.get(str(team_id)) or {}).get("revenue"), 185.0)
    try:
        from services.league_operations import _cap_index

        idx = _cap_index(session)
    except Exception:
        idx = 1.0
    budget = DEFAULT_SCOUTING_BUDGET_DOLLARS * (max(60.0, rev) / (185.0 * idx)) ** 0.8
    budget *= 1.0 + gov_mod(session, "scouting_budget") / 100.0
    return int(round(_clamp(budget, 1_200_000, 6_000_000) / 10_000.0) * 10_000)


def apply_revenue_scouting_budget(session: Any, rows: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    uid = str(getattr(session, "user_team_id", "") or "")
    budget = revenue_scouting_budget(session, uid, rows)
    try:
        from services.franchise_scouting import _ensure_scouting_state

        state = _ensure_scouting_state(session)
        state["budget"] = float(budget)
        state["budget_source"] = "revenue"
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return budget


# ---------------------------------------------------------------------------
# Signing-bonus demands (free agency)
# ---------------------------------------------------------------------------


def fa_bonus_demand_pct(player: Any, league: Any = None) -> float:
    """Share of total contract value this free agent insists on as a signing bonus
    (0 = no demand). Stars and veterans want cash up front; only clubs above the
    bonus revenue floor can pay it."""
    try:
        from services.contract_economy import _player_age, _player_id, _player_ovr
    except Exception:
        return 0.0
    ovr = _sf(_player_ovr(player))
    age = int(_player_age(player) or 27)
    share = 0.45 + gov_mod(league, "fa_bonus_demand") / 100.0
    u = _unit("fa-bonus", _player_id(player))
    if ovr >= 85 and u < share:
        return round(0.08 + (ovr - 85) * 0.006 + u * 0.05, 3)
    if ovr >= 80 and age >= 30 and u < share * 0.5:
        return round(0.06 + u * 0.04, 3)
    return 0.0


# ---------------------------------------------------------------------------
# Relocation
# ---------------------------------------------------------------------------


def relocate_team(session: Any, team_id: str, market: Dict[str, Any]) -> Dict[str, Any]:
    gov = ensure_governance(session)
    team = (getattr(session, "team_by_id", None) or {}).get(str(team_id))
    if team is None:
        return {}
    old_city, old_abbr = str(getattr(team, "city", "") or ""), _team_abbr(team)
    team.city = market["city"]
    team.abbr = market["abbr"]
    try:
        team.abbreviation = market["abbr"]
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        mp = getattr(team, "market", None)
        if mp is not None:
            mp.market_size = market["tier"]
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    # A club that moves across the continent changes conference (and division).
    new_conf = str(market.get("conference") or "")
    league = _league(session)
    if new_conf and league is not None and str(getattr(team, "conference", "") or "") != new_conf:
        divs: Dict[str, int] = {}
        for t in list(getattr(league, "teams", None) or []):
            if t is not team and str(getattr(t, "conference", "") or "") == new_conf:
                d = str(getattr(t, "division", "") or "")
                divs[d] = divs.get(d, 0) + 1
        try:
            team.conference = new_conf
            if divs:
                team.division = min(divs, key=lambda d: divs[d])
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    sy = _season(session)
    vals = ensure_franchise_values(session)
    row = vals.get(str(team_id))
    if isinstance(row, dict):
        new_v = round(max(_sf(row.get("value_b")) * 1.15, market_value_b(session, market)), 3)
        row["value_b"] = new_v
        row.setdefault("history", []).append({"season": sy, "value_b": new_v, "change_pct": 0.0, "drivers": [f"Relocated to {market['city']}"]})
    gov["honeymoon"][str(team_id)] = {str(sy + 1): 14.0, str(sy + 2): 7.0}
    gov["used_markets"].append(market["abbr"])
    # Fans start fresh in the new city.
    try:
        from services.franchise_sim import _ensure_team_fan_profile

        prof = _ensure_team_fan_profile(session, str(team_id))
        prof["fan_confidence"] = 68.0
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    demands = _relocation_trade_demands(session, team)
    rec = {"season": sy, "team_id": str(team_id), "from_city": old_city, "from_abbr": old_abbr, "to_city": market["city"], "to_abbr": market["abbr"], "demands": demands}
    gov["relocations"].append(rec)
    names = ", ".join(d["name"] for d in demands if d.get("name"))
    _news(
        session,
        f"Board approves move: {old_city} club relocates to {market['city']}",
        f"The franchise becomes {market['city']} {getattr(team, 'name', '')} ({market['abbr']}) next season."
        + (f" {names} asked for trades rather than move." if names else ""),
        team_id=str(team_id),
        kind="relocation",
    )
    return rec


def _relocation_trade_demands(session: Any, team: Any) -> List[Dict[str, Any]]:
    """Veterans with roots in the old city may refuse to move."""
    try:
        from services.contract_economy import _player_age, _player_id, _player_negotiation_profile, _player_ovr
        from services.trade_demand_engine import open_trade_demand
    except Exception:
        return []
    mult = trade_demand_multiplier(session)
    cands = []
    for p in list(getattr(team, "roster", None) or []):
        age = int(_player_age(p) or 27)
        if _sf(_player_ovr(p)) < 72:
            continue
        prof = _player_negotiation_profile(p)
        base = 0.32 if age >= 31 else 0.18 if age >= 27 else 0.06
        chance = base * (1.4 - prof.get("loyalty", 0.5)) * mult
        u = _unit(_salt(session), "reloc-demand", _player_id(p), _season(session))
        if u < chance:
            cands.append((chance - u, p))
    cands.sort(key=lambda c: -c[0])
    out = []
    for _gap, p in cands[:3]:
        try:
            open_trade_demand(
                session, p, team, reason="relocation",
                calendar_idx=int(getattr(session, "calendar_cursor", 0) or 0), force_formal=True,
            )
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        out.append({"player_id": _player_id(p), "name": str(getattr(getattr(p, "identity", None), "name", "") or getattr(p, "name", ""))})
    return out


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _pos_group(p: Any) -> str:
    try:
        from app.sim_engine.trades.team_assessment import position_group

        g = position_group(p)
        return "G" if g == "G" else ("D" if g in ("LD", "RD", "D") else "F")
    except Exception:
        return "F"


def _ovr(p: Any) -> float:
    try:
        from app.sim_engine.trades.team_assessment import player_ovr

        return float(player_ovr(p))
    except Exception:
        return 60.0


def _age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    try:
        return int(getattr(ident, "age", getattr(p, "age", 27)) or 27)
    except Exception:
        return 27


def _has_contract(p: Any) -> bool:
    try:
        from app.sim_engine.trades.trade_asset import player_holds_nhl_spc

        return bool(player_holds_nhl_spc(p))
    except Exception:
        return getattr(p, "contract", None) is not None


EXPANSION_PROTECT = {"F": 7, "D": 3, "G": 1}


# NHL expansion rules: either 7F / 3D / 1G, or any 8 skaters + 1G.
EXPANSION_SCHEMES: Dict[str, Dict[str, int]] = {"7-3-1": {"F": 7, "D": 3, "G": 1}, "8-1": {"S": 8, "G": 1}}
# Exposure: 2F + 1D under contract who meet the games-played bar, plus one signed goalie.
EXPANSION_EXPOSE = {"F": 2, "D": 1, "G": 1}
EXPOSE_GP_SEASON = 40
EXPOSE_GP_TWO_YEAR = 70


def _has_nmc(p: Any) -> bool:
    try:
        from services import contract_economy as _ce

        fn = getattr(_ce, "is_nmc", None) or getattr(_ce, "has_nmc", None)
        if fn is not None and fn(p):
            return True
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from app.sim_engine.trades.trade_rules import _clause_summary

        return bool((_clause_summary(p) or {}).get("nmc"))
    except Exception:
        return False


def _slot(group: str, scheme: str) -> str:
    return group if group == "G" or scheme == "7-3-1" else "S"


def _scheme_limits(scheme: str) -> Dict[str, int]:
    return dict(EXPANSION_SCHEMES.get(scheme) or EXPANSION_SCHEMES["7-3-1"])


def _season_gp(p: Any, session: Any) -> int:
    pid = str(getattr(p, "id", "") or "")
    row = (getattr(session, "player_season_stats", None) or {}).get(pid) if session is not None else None
    try:
        return int((row or {}).get("gp") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def _career_gp(p: Any) -> int:
    try:
        from services.contract_economy import _player_nhl_games

        return int(_player_nhl_games(p) or 0)
    except Exception:
        return 0


def _exposure_eligible(p: Any, session: Any) -> bool:
    """Signed player who can count toward the exposure minimum (GP bar for skaters)."""
    if not _has_contract(p) or _exempt_young_pro(p):
        return False
    if _pos_group(p) == "G":
        return True
    return _season_gp(p, session) >= EXPOSE_GP_SEASON or _career_gp(p) >= EXPOSE_GP_TWO_YEAR


def _exempt_young_pro(p: Any) -> bool:
    """First- and second-year pros are exempt (NHL rule); age only when pro years are unknown."""
    for attr in ("pro_seasons", "nhl_seasons", "seasons_pro"):
        v = getattr(p, attr, None)
        if v is None:
            v = getattr(getattr(p, "career", None), attr, None)
        if v is not None:
            try:
                return int(v) <= 2
            except (TypeError, ValueError):
                break
    return _age(p) <= 20


def _org_players(team: Any) -> List[Any]:
    return [p for p in list(getattr(team, "roster", None) or []) + list(getattr(team, "ahl_roster", None) or []) if not getattr(p, "retired", False)]


def _protect_value(p: Any) -> float:
    """CPU protection priority: rating, with a lean toward players in their prime."""
    age = _age(p)
    return _ovr(p) + (2.0 if age <= 26 else 0.0) - max(0, age - 31) * 1.2


def _protected_ids(
    team: Any,
    chosen: Optional[List[str]] = None,
    *,
    scheme: str = "7-3-1",
    session: Any = None,
    autofill: bool = True,
) -> set:
    """7F / 3D / 1G or 8 skaters + 1G. No-move-clause players must be protected and use a
    slot; first- and second-year pros are exempt. ``chosen`` = the user's own list
    (validated here). Exposure minimums are enforced afterwards by unprotecting the
    weakest protected (non-NMC) player of a short position group."""
    org = _org_players(team)
    limits = _scheme_limits(scheme)
    keep: set = set()
    used = {k: 0 for k in limits}
    by_id = {str(getattr(p, "id", "")): p for p in org}
    nmc_ids: set = set()
    for p in org:
        if _exempt_young_pro(p):
            keep.add(str(getattr(p, "id", "")))
    for p in sorted([p for p in org if _has_nmc(p)], key=_ovr, reverse=True):
        s = _slot(_pos_group(p), scheme)
        pid = str(getattr(p, "id", ""))
        if pid not in keep and used[s] < limits[s]:
            keep.add(pid)
            nmc_ids.add(pid)
            used[s] += 1
    for pid in [str(x) for x in (chosen or [])]:
        p = by_id.get(pid)
        if p is None or pid in keep:
            continue
        s = _slot(_pos_group(p), scheme)
        if used[s] < limits[s]:
            keep.add(pid)
            used[s] += 1
    if autofill:
        for p in sorted(org, key=_protect_value, reverse=True):
            pid = str(getattr(p, "id", ""))
            s = _slot(_pos_group(p), scheme)
            if pid in keep or used[s] >= limits[s]:
                continue
            keep.add(pid)
            used[s] += 1
    # Exposure requirements.
    for grp, need in EXPANSION_EXPOSE.items():
        while True:
            exposed = [p for p in org if _pos_group(p) == grp and str(getattr(p, "id", "")) not in keep and _exposure_eligible(p, session)]
            if len(exposed) >= need:
                break
            cands = [
                p for p in org
                if _pos_group(p) == grp and str(getattr(p, "id", "")) in keep
                and str(getattr(p, "id", "")) not in nmc_ids and _exposure_eligible(p, session)
            ]
            if not cands:
                break
            keep.discard(str(getattr(min(cands, key=_ovr), "id", "")))
    return keep


def exposure_report(team: Any, keep: set, session: Any = None) -> Dict[str, Any]:
    org = _org_players(team)
    have = {g: 0 for g in EXPANSION_EXPOSE}
    for p in org:
        if str(getattr(p, "id", "")) not in keep and _exposure_eligible(p, session):
            have[_pos_group(p)] += 1
    return {"need": dict(EXPANSION_EXPOSE), "have": have, "ok": all(have[g] >= n for g, n in EXPANSION_EXPOSE.items())}


def cpu_protection(team: Any, session: Any = None) -> Tuple[str, set]:
    """CPU clubs pick whichever scheme leaves the weakest best-available player exposed
    (you only lose one), breaking ties on total protected value."""
    best: Optional[Tuple[Tuple[float, float], str, set]] = None
    for scheme in EXPANSION_SCHEMES:
        keep = _protected_ids(team, None, scheme=scheme, session=session)
        org = _org_players(team)
        exposed = [_protect_value(p) for p in org if str(getattr(p, "id", "")) not in keep and _has_contract(p)]
        kept = sum(_protect_value(p) for p in org if str(getattr(p, "id", "")) in keep and not _exempt_young_pro(p))
        val = (-(max(exposed) if exposed else 0.0), kept)
        if best is None or val > best[0]:
            best = (val, scheme, keep)
    return best[1], best[2]


def expansion_protection_payload(session: Any) -> Dict[str, Any]:
    """The user's protection board for an expansion draft due this offseason."""
    gov = ensure_governance(session)
    exp = gov.get("expansion") or {}
    nxt = _season(session) + 1
    due = [e for e in exp.get("pending") or [] if int(e.get("start_season") or 0) == nxt]
    upcoming = [e for e in exp.get("pending") or [] if int(e.get("start_season") or 0) > nxt]
    out: Dict[str, Any] = {
        "due": bool(due),
        "limits": dict(EXPANSION_PROTECT),
        "schedule_every": int(exp.get("schedule_every") or 0),
        "upcoming": [
            {"cities": [c.get("city") for c in e.get("cities") or []], "draft_offseason": int(e.get("start_season") or 0) - 1,
             "start_season": int(e.get("start_season") or 0)}
            for e in upcoming
        ],
    }
    if not due:
        return out
    uid = str(getattr(session, "user_team_id", "") or "")
    team = (getattr(session, "team_by_id", None) or {}).get(uid)
    chosen = list(exp.get("user_protected") or [])
    scheme = str(exp.get("user_scheme") or "7-3-1")
    if scheme not in EXPANSION_SCHEMES:
        scheme = "7-3-1"
    protected = _protected_ids(team, chosen, scheme=scheme, session=session) if team is not None else set()
    out["limits"] = _scheme_limits(scheme)
    out["scheme"] = scheme
    out["schemes"] = list(EXPANSION_SCHEMES.keys())
    out["exposure"] = exposure_report(team, protected, session) if team is not None else {}
    rows = []
    for p in _org_players(team) if team is not None else []:
        pid = str(getattr(p, "id", ""))
        rows.append({
            "player_id": pid,
            "name": str(getattr(getattr(p, "identity", None), "name", "") or getattr(p, "name", "")),
            "pos": _pos_group(p),
            "ovr": round(_ovr(p), 1),
            "age": _age(p),
            "exempt": _exempt_young_pro(p),
            "nmc": _has_nmc(p),
            "protected": pid in protected,
            "contract": _has_contract(p),
            "exposable": _exposure_eligible(p, session),
        })
    rows.sort(key=lambda r: ({"F": 0, "D": 1, "G": 2}[r["pos"]], -r["ovr"]))
    out.update({
        "cities": [c.get("city") for c in due[0].get("cities") or []],
        "n_new": len(due[0].get("cities") or []),
        "when": "At the end of this Board of Governors meeting (before the Entry Draft)",
        "players": rows,
        "user_set": bool(chosen),
    })
    return out


def set_expansion_protection(session: Any, player_ids: List[str], scheme: Optional[str] = None) -> Dict[str, Any]:
    gov = ensure_governance(session)
    exp = gov.setdefault("expansion", {"pending": [], "joined": []})
    uid = str(getattr(session, "user_team_id", "") or "")
    team = (getattr(session, "team_by_id", None) or {}).get(uid)
    if team is None:
        raise ValueError("No club")
    scheme = str(scheme or exp.get("user_scheme") or "7-3-1")
    if scheme not in EXPANSION_SCHEMES:
        raise ValueError("Protection scheme must be 7-3-1 or 8-1")
    limits = _scheme_limits(scheme)
    by_id = {str(getattr(p, "id", "")): p for p in _org_players(team)}
    counts = {k: 0 for k in limits}
    clean: List[str] = []
    names = {"F": "forwards", "D": "defensemen", "G": "goalie", "S": "skaters"}
    for pid in [str(x) for x in player_ids or []]:
        p = by_id.get(pid)
        if p is None or pid in clean or _exempt_young_pro(p):
            continue
        s = _slot(_pos_group(p), scheme)
        counts[s] += 1
        if counts[s] > limits[s]:
            raise ValueError(f"You can protect at most {limits[s]} {names[s]}")
        clean.append(pid)
    # No-move clauses must be protected.
    for p in by_id.values():
        pid = str(getattr(p, "id", ""))
        if _has_nmc(p) and not _exempt_young_pro(p) and pid not in clean:
            s = _slot(_pos_group(p), scheme)
            if counts[s] >= limits[s]:
                nm = str(getattr(getattr(p, "identity", None), "name", "") or pid)
                raise ValueError(f"{nm} has a no-move clause and must be protected")
            counts[s] += 1
            clean.append(pid)
    exp["user_protected"] = clean
    exp["user_scheme"] = scheme
    return expansion_protection_payload(session)


def set_expansion_schedule(session: Any, every_n_seasons: int) -> Dict[str, Any]:
    """League option: 0 = expansion only when the market produces a bid; N = a bid every N seasons."""
    gov = ensure_governance(session)
    exp = gov.setdefault("expansion", {"pending": [], "joined": []})
    n = max(0, min(20, int(every_n_seasons or 0)))
    exp["schedule_every"] = n
    exp["schedule_anchor"] = _season(session)
    return expansion_protection_payload(session)


def _cap_hit_m(p: Any) -> float:
    try:
        from services.contract_economy import _contract_cap_hit_m

        return float(_contract_cap_hit_m(p) or 0.0)
    except Exception:
        return 0.0


def _fair_cap_m(p: Any) -> float:
    """Rough market AAV for a rating, so the new club avoids dead-weight contracts."""
    return max(0.8, (_ovr(p) - 62.0) * 0.45)


def _side_deal_pick(league: Any, team_id: str, rnd: int, season_year: int, exclude: Optional[set] = None) -> Optional[str]:
    """The club's own pick of the given round in the nearest draft it still owns."""
    try:
        from app.sim_engine.trades.trade_pick_registry import get_team_owned_picks

        for row in get_team_owned_picks(league, str(team_id), min_year=int(season_year)):
            if int(row.get("round") or 0) == int(rnd) and str(row.get("pick_id") or "") not in (exclude or set()):
                return str(row.get("pick_id") or "") or None
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return None


def run_pending_expansion(session: Any, next_season_year: int) -> List[Dict[str, Any]]:
    """Create approved expansion clubs (with an expansion draft) before the schedule
    for ``next_season_year`` is generated. Returns the joined clubs."""
    gov = ensure_governance(session)
    exp = gov["expansion"]
    due = [e for e in exp.get("pending") or [] if int(e.get("start_season") or 0) == int(next_season_year)]
    if not due:
        return []
    league = _league(session)
    if league is None:
        return []
    from app.sim_engine.entities.team import Team
    from app.sim_engine.trades.trade_executor import _purge_player_id_from_team_lists, _sync_assignment_flags

    joined: List[Dict[str, Any]] = []
    rng = random.Random(_salt(session) * 15485863 + int(next_season_year))
    existing = list(league.teams)
    ids = [int(getattr(t, "team_id", 0) or 0) for t in league.teams if str(getattr(t, "team_id", "")).isdigit()]
    next_id = (max(ids) + 1) if ids else len(league.teams)
    for entry in due:
        new_teams = []
        for c in entry["cities"]:
            new_id = next_id
            next_id += 1
            conf = c.get("conference") or ""
            if not conf:
                confs: Dict[str, int] = {}
                for t in list(league.teams) + new_teams:
                    k = str(getattr(t, "conference", "") or "")
                    if k:
                        confs[k] = confs.get(k, 0) + 1
                conf = min(confs, key=lambda k: confs[k]) if confs else "Western"
            divs: Dict[str, int] = {}
            for t in list(league.teams) + new_teams:
                if str(getattr(t, "conference", "")) == conf:
                    d = str(getattr(t, "division", "") or "")
                    divs[d] = divs.get(d, 0) + 1
            division = min(divs, key=lambda d: divs[d]) if divs else conf
            taken_names = {str(getattr(t, "name", "")) for t in list(league.teams) + new_teams}
            nick_pool = [n for n in EXPANSION_NICKNAMES if n not in taken_names]
            nickname = nick_pool[int(rng.random() * len(nick_pool)) % len(nick_pool)] if nick_pool else "Expansion"
            team = Team(team_id=new_id, city=c["city"], name=nickname, division=division, conference=conf, archetype="balanced", rng=random.Random(rng.random()))
            team.abbr = c["abbr"]
            try:
                team.abbreviation = c["abbr"]
                team.market.market_size = c.get("tier", "medium")
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            new_teams.append(team)
        # Expansion draft: each new club takes one unprotected contracted player per club.
        picks_log: List[Dict[str, Any]] = []
        needs = {t: {"F": 14, "D": 9, "G": 3} for t in new_teams}
        uid = str(getattr(session, "user_team_id", "") or "")
        user_chosen = list(exp.get("user_protected") or [])
        user_scheme = str(exp.get("user_scheme") or "7-3-1")
        pools: Dict[str, List[Any]] = {}
        protection_log: Dict[str, Dict[str, Any]] = {}
        for src in existing:
            sid = str(getattr(src, "team_id", ""))
            if sid == uid:
                scheme = user_scheme if user_scheme in EXPANSION_SCHEMES else "7-3-1"
                protected = _protected_ids(src, user_chosen, scheme=scheme, session=session)
            else:
                scheme, protected = cpu_protection(src, session)
            pools[sid] = [
                p for p in _org_players(src)
                if str(getattr(p, "id", "")) not in protected and _has_contract(p)
            ]
            protection_log[sid] = {"scheme": scheme, "protected": sorted(protected), "exposure": exposure_report(src, protected, session)}
        side_deals: List[Dict[str, Any]] = []
        claimed: set = set()  # (source club, new club) pairs already used

        def _take(src: Any, nt: Any, p: Any) -> None:
            sid = str(getattr(src, "team_id", ""))
            pools[sid].remove(p)
            claimed.add((sid, str(nt.team_id)))
            pid = str(getattr(p, "id", ""))
            _purge_player_id_from_team_lists(src, pid)
            nt.roster.append(p)
            for field in ("team_id", "current_team_id"):
                try:
                    setattr(p, field, nt.team_id)
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
            needs[nt][_pos_group(p)] = needs[nt].get(_pos_group(p), 0) - 1
            picks_log.append({
                "team": c_abbr(nt), "player": str(getattr(getattr(p, "identity", None), "name", "") or ""),
                "player_id": pid, "from": _team_abbr(src), "from_team_id": sid, "ovr": round(_ovr(p), 1),
                "pos": _pos_group(p),
            })

        # Goalies first: every new club leaves with two (unprotected goalies are usually
        # backups, so a best-available pass could leave a club without a starter).
        for nt in new_teams:
            for _ in range(2):
                best = None
                for src in existing:
                    sid = str(getattr(src, "team_id", ""))
                    if (sid, str(nt.team_id)) in claimed:
                        continue
                    for p in pools.get(sid) or []:
                        if _pos_group(p) == "G" and (best is None or _ovr(p) > _ovr(best[1])):
                            best = (src, p)
                if best is not None:
                    _take(best[0], nt, best[1])

        for src in existing:
            sid = str(getattr(src, "team_id", ""))
            for nt in new_teams:
                pool = pools.get(sid) or []
                if (sid, str(nt.team_id)) in claimed or not pool:
                    continue
                need = needs[nt]

                def _score(p: Any) -> float:
                    g = _pos_group(p)
                    return (
                        _ovr(p) + (6.0 if need.get(g, 0) > 0 else -4.0) - max(0, _age(p) - 31) * 1.5
                        + (2.0 if _age(p) <= 26 else 0.0) - max(0.0, _cap_hit_m(p) - _fair_cap_m(p)) * 1.5
                    )

                pool.sort(key=_score, reverse=True)
                choice = pool[0]
                # Side deal: a CPU club pays a pick to steer the new club off a prized player.
                if sid != uid and len(pool) > 1:
                    gap = _score(pool[0]) - _score(pool[1])
                    if gap >= 4.0 and _ovr(pool[0]) >= 74.0:
                        rnd = 2 if gap >= 8.0 else 3
                        pick_id = _side_deal_pick(league, sid, rnd, int(next_season_year) + 1, {d["pick_id"] for d in side_deals})
                        if pick_id:
                            choice = pool[1]
                            side_deals.append({
                                "from_team_id": sid, "to_team_id": str(nt.team_id), "pick_id": pick_id, "round": rnd,
                                "kept": str(getattr(getattr(pool[0], "identity", None), "name", "") or ""),
                                "from": _team_abbr(src), "team": c_abbr(nt),
                            })
                _take(src, nt, choice)
        for nt in new_teams:
            org = sorted(nt.roster, key=_ovr, reverse=True)
            nhl: List[Any] = []
            counts = {"F": 0, "D": 0, "G": 0}
            limits = {"F": 13, "D": 8, "G": 2}
            for p in org:
                g = _pos_group(p)
                if counts[g] < limits[g] and len(nhl) < 23:
                    nhl.append(p)
                    counts[g] += 1
            ahl = [p for p in org if p not in nhl]
            nt.roster = nhl
            nt.ahl_roster = ahl
            for p in nhl:
                _sync_assignment_flags(p, "roster")
            for p in ahl:
                _sync_assignment_flags(p, "ahl_roster")
            league.teams.append(nt)
            tid = str(nt.team_id)
            session.team_by_id[tid] = nt
            ensure_franchise_values(session)
            _city = next((c for c in entry["cities"] if c["abbr"] == c_abbr(nt)), {"value_b": 1.8})
            _val = market_value_b(session, _city)
            gov["franchise_values"][tid] = {
                "value_b": float(_val),
                "history": [{"season": int(next_season_year), "value_b": float(_val), "change_pct": 0.0, "drivers": ["Expansion entry"]}],
            }
            gov["honeymoon"][tid] = {str(next_season_year): 18.0, str(next_season_year + 1): 9.0}
            joined.append({"team_id": tid, "city": nt.city, "name": nt.name, "abbr": c_abbr(nt), "season": int(next_season_year), "players": len(nhl) + len(ahl)})
            _news(session, f"The {nt.city} {nt.name} are here", f"{nt.city} joins the league for {next_season_year}-{(next_season_year + 1) % 100:02d} with {len(nhl) + len(ahl)} players from the expansion draft.", team_id=tid, kind="expansion")
        done_deals: List[Dict[str, Any]] = []
        if side_deals:
            from app.sim_engine.trades.trade_pick_registry import transfer_pick

            for d in side_deals:
                try:
                    transfer_pick(league, d["pick_id"], d["to_team_id"])
                    done_deals.append(d)
                    _news(
                        session,
                        f"{d['from']} send a round-{d['round']} pick to {d['team']} in expansion side deal",
                        f"{d['team']} agreed to pass on {d['kept'] or 'a prized player'} in the expansion draft in exchange for the pick.",
                        team_id=d["from_team_id"], kind="expansion",
                    )
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
        exp.setdefault("drafts", []).append({
            "season": int(next_season_year), "picks": picks_log, "side_deals": done_deals,
            "protection": {sid: {"scheme": row["scheme"], "exposure_ok": row["exposure"]["ok"]} for sid, row in protection_log.items()},
        })
    exp["pending"] = [e for e in exp.get("pending") or [] if e not in due]
    exp["joined"] = list(exp.get("joined") or []) + joined
    exp["user_protected"] = []
    try:
        from app.sim_engine.trades.trade_pick_registry import ensure_franchise_pick_registry

        ensure_franchise_pick_registry(league, season_calendar_year=int(next_season_year), years_ahead=4)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return joined


def c_abbr(team: Any) -> str:
    return str(getattr(team, "abbr", "") or "").upper()


# ---------------------------------------------------------------------------
# Europe / KHL: departures, returns and undrafted European breakouts
# ---------------------------------------------------------------------------

EUROPE_MAX_DEPARTURES = 8
EUROPE_BREAKOUTS = (2, 4)
_EURO_LEAGUE_TEAMS = {"KHL": "KHL_TEAMS", "SHL": "SHL_TEAMS", "Liiga": "LIIGA_TEAMS", "NL": "SWISS_TEAMS", "DEL": "DEL_TEAMS", "Czech Extraliga": "CZECH_TEAMS"}


def _euro_club(rng: random.Random, league_code: str) -> str:
    try:
        from app.sim_engine.generation import prospect_league_teams as plt

        rows = getattr(plt, _EURO_LEAGUE_TEAMS.get(league_code, "SHL_TEAMS"), None) or []
        if rows:
            return str(rng.choice(rows).get("name") or "")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return ""


def _player_name(p: Any) -> str:
    return str(getattr(getattr(p, "identity", None), "name", "") or getattr(p, "name", "") or "")


def _morale(session: Any, pid: str) -> float:
    try:
        ent = (getattr(session, "universe_players", None) or {}).get(pid) or {}
        return float((ent.get("state") or {}).get("morale", 60) or 60)
    except Exception:
        return 60.0


def _set_overseas(p: Any, league_code: str, club: str, **extra: Any) -> None:
    meta = dict(getattr(p, "_franchise_assignment", None) or {})
    meta.update({"level": "ufa", "overseas": True, "overseas_league": league_code, "club": club or None}, **extra)
    try:
        p._franchise_assignment = meta
        p.context.current_team_id = "OVERSEAS"
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def run_europe_moves(session: Any) -> Dict[str, Any]:
    """Once per summer (FA opening): depth vets / unqualified RFAs / squeezed-out players
    leave for Europe or the KHL; earlier departures can come back as overseas free agents;
    a few undrafted 20-25 year-old European breakouts hit the overseas wire with real
    stat lines (the FA board projects their season from ``overseas_league``/``club``)."""
    league = _league(session)
    if league is None:
        return {}
    gov = ensure_governance(session)
    eu = gov.setdefault("europe", {"done": [], "log": []})
    sy = _season(session)
    if sy in eu["done"]:
        return {}
    eu["done"].append(sy)
    rng = random.Random(_salt(session) * 7919 + sy * 31 + 5)
    abroad: List[Any] = list(getattr(session, "europe_abroad_players", None) or [])
    report: Dict[str, Any] = {"season": sy, "departed": [], "returned": [], "breakouts": []}

    # 1) Departures from the domestic UFA pool.
    fa = list(getattr(league, "free_agents", None) or [])
    cands: List[Tuple[float, Any, str]] = []
    for p in fa:
        meta = getattr(p, "_franchise_assignment", None) or {}
        if meta.get("overseas") or getattr(p, "retired", False):
            continue
        ovr, age, pid = _ovr(p), _age(p), str(getattr(p, "id", "") or "")
        gp = _season_gp(p, session)
        reason = ""
        chance = 0.0
        if age <= 25 and 60 <= ovr <= 72:
            reason, chance = "unsigned RFA", 0.10
        elif 27 <= age <= 33 and 60 <= ovr <= 74:
            reason, chance = "depth veteran", 0.08
            if _morale(session, pid) < 45:
                reason, chance = "unhappy depth veteran", 0.20
            elif gp and gp < 25:
                reason, chance = "squeezed out of a role", 0.16
        if chance and rng.random() < chance:
            cands.append((ovr, p, reason))
    cands.sort(key=lambda x: -x[0])
    for ovr, p, reason in cands[:EUROPE_MAX_DEPARTURES]:
        code = "KHL" if (ovr >= 68 or rng.random() < 0.45) else rng.choice(["SHL", "Liiga", "NL", "DEL", "Czech Extraliga"])
        club = _euro_club(rng, code)
        _set_overseas(p, code, club, europe_departed=sy)
        try:
            fa.remove(p)
        except ValueError:
            pass
        abroad.append(p)
        row = {"player_id": str(getattr(p, "id", "")), "name": _player_name(p), "league": code, "club": club, "reason": reason, "ovr": round(ovr, 1), "age": _age(p)}
        report["departed"].append(row)
        _news(session, f"{row['name']} signs with {club or code}", f"The {row['age']}-year-old ({reason}) heads to the {code} after finding no NHL fit.", kind="europe")
    league.free_agents = fa

    # 2) Returns: players who spent at least one season abroad.
    overseas = list(getattr(league, "overseas_free_agents", None) or [])
    keep_abroad: List[Any] = []
    for p in abroad:
        meta = getattr(p, "_franchise_assignment", None) or {}
        left = int(meta.get("europe_departed") or sy)
        age = _age(p)
        if age >= 37:
            continue  # finishes his career overseas
        if left < sy and rng.random() < (0.30 + 0.10 * min(3, sy - left - 1)) and _ovr(p) >= 62:
            meta = dict(meta)
            meta.update({"returning_from_europe": True})
            p._franchise_assignment = meta
            overseas.append(p)
            row = {"player_id": str(getattr(p, "id", "")), "name": _player_name(p), "league": meta.get("overseas_league"), "club": meta.get("club"), "ovr": round(_ovr(p), 1), "age": age}
            report["returned"].append(row)
            _news(session, f"{row['name']} eyes an NHL return", f"After time with {row['club'] or row['league']}, the {age}-year-old is an overseas free agent again.", kind="europe")
        else:
            keep_abroad.append(p)
    session.europe_abroad_players = keep_abroad

    # 3) Undrafted European breakouts, aged 20-25.
    try:
        from app.sim_engine.entities.player import Position
        from app.sim_engine.league_hierarchy_bootstrap import _spawn_player

        used = {_player_name(p) for p in list(getattr(league, "players", None) or []) + overseas}
        for _ in range(rng.randint(*EUROPE_BREAKOUTS)):
            code = rng.choice(["SHL", "SHL", "Liiga", "Liiga", "KHL"])
            pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.D, Position.G])
            p = _spawn_player(
                rng, pos=pos, ovr_lo=0.64, ovr_hi=0.76, age_lo=20, age_hi=25, used_names=used,
                league_players=list(getattr(league, "players", None) or []), pool_context="overseas",
            )
            club = _euro_club(rng, code)
            _set_overseas(p, code, club, europe_breakout=sy, undrafted=True)
            overseas.append(p)
            row = {"player_id": str(getattr(p, "id", "")), "name": _player_name(p), "league": code, "club": club, "ovr": round(_ovr(p), 1), "age": _age(p)}
            report["breakouts"].append(row)
            _news(session, f"Undrafted {code} breakout {row['name']} draws NHL interest", f"The {row['age']}-year-old from {club or code} is an overseas free agent after a breakout season.", kind="europe")
    except Exception:
        _swallowed_log.debug("europe breakouts failed", exc_info=True)
    league.overseas_free_agents = overseas
    eu["log"] = (list(eu.get("log") or []) + [report])[-6:]
    return report


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def public_proposal(prop: Dict[str, Any], *, reveal: bool) -> Dict[str, Any]:
    out = {k: v for k, v in prop.items() if not k.startswith("_")}
    if not reveal or prop.get("status") == "pending":
        out.pop("ballots", None)
    return out


def _user_impact(session: Any) -> Dict[str, Any]:
    uid = str(getattr(session, "user_team_id", "") or "")
    out: Dict[str, Any] = {}
    try:
        fin = _team_rows(session).get(uid) or {}
        out["annual_revenue_m"] = fin.get("revenue")
        out["annual_profit_m"] = fin.get("profit")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from services.franchise_offseason import team_signing_bonus_eligibility

        elig = team_signing_bonus_eligibility(session, uid)
        out["bonus_eligible"] = bool(elig.get("eligible"))
        out["bonus_max_pct"] = elig.get("max_bonus_pct")
        out["bonus_floor_m"] = elig.get("floor_m")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from services.franchise_scouting import _ensure_scouting_state

        out["scouting_budget"] = float(_ensure_scouting_state(session).get("budget") or 0)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    row = (ensure_governance(session).get("franchise_values") or {}).get(uid)
    if isinstance(row, dict):
        hist = row.get("history") or []
        out["franchise_value_b"] = row.get("value_b")
        out["value_change_pct"] = hist[-1].get("change_pct") if hist else 0.0
        out["value_drivers"] = hist[-1].get("drivers") if hist else []
    m, spikes = star_spike_m(session, uid, _season(session) + 1)
    out["star_spike_next_m"] = m
    out["star_spikes"] = spikes
    return out


FAN_TIERS: Tuple[Tuple[float, str], ...] = (
    (85.0, "Electric"),
    (70.0, "Loyal"),
    (55.0, "Steady"),
    (40.0, "Restless"),
    (-1.0, "Hostile"),
)


def fan_tier(value: float) -> str:
    v = float(value)
    for floor, label in FAN_TIERS:
        if v >= floor:
            return label
    return "Hostile"


def _fan_profile(session: Any, tid: str) -> Dict[str, Any]:
    try:
        from services.franchise_sim import _ensure_team_fan_profile

        return _ensure_team_fan_profile(session, str(tid)) or {}
    except Exception:
        return {}


def fan_ticker(session: Any) -> List[Dict[str, Any]]:
    """Every club's fan confidence, tier and weekly move, best first (Board screen ticker)."""
    uid = str(getattr(session, "user_team_id", "") or "")
    rows: List[Dict[str, Any]] = []
    for tid, team in (getattr(session, "team_by_id", None) or {}).items():
        if team is None:
            continue
        prof = _fan_profile(session, str(tid))
        v = _sf(prof.get("fan_confidence"), 55.0)
        last = _sf(prof.get("fan_confidence_last_week"), v)
        rows.append({
            "team_id": str(tid),
            "abbr": _team_abbr(team),
            "value": round(v, 1),
            "delta": round(v - last, 1),
            "tier": fan_tier(v),
            "is_user": str(tid) == uid,
        })
    rows.sort(key=lambda r: -r["value"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def user_fan_report(session: Any) -> Dict[str, Any]:
    """Why the user's fans feel the way they do, and what that is worth right now."""
    uid = str(getattr(session, "user_team_id", "") or "")
    team = (getattr(session, "team_by_id", None) or {}).get(uid)
    prof = _fan_profile(session, uid)
    v = _sf(prof.get("fan_confidence"), 55.0)
    last = _sf(prof.get("fan_confidence_last_week"), v)
    baseline = prof.get("fan_baseline")
    if baseline is None:
        try:
            from services.franchise_sim import compute_team_playoff_outlook

            odds = int((compute_team_playoff_outlook(session, team) or {}).get("playoff_odds", 50) or 50)
            baseline = _clamp(40 + odds * 0.35, 0.0, 100.0)
        except Exception:
            baseline = 55.0
    baseline = float(baseline)

    reasons: List[Dict[str, str]] = [{
        "tone": "pro" if baseline >= 60 else ("con" if baseline < 50 else "mixed"),
        "text": f"Your record and playoff odds support about {baseline:.0f}. Results set the baseline.",
    }]
    gap = v - baseline
    if gap >= 5:
        reasons.append({"tone": "pro", "text": f"Buzz: recent storylines have fans {gap:.0f} points above what the record alone supports."})
    elif gap <= -5:
        reasons.append({"tone": "con", "text": f"Frustration: fans are {abs(gap):.0f} points below what the record supports."})
    heat = _sf(prof.get("recent_trade_heat"))
    if heat >= 25:
        reasons.append({"tone": "con", "text": f"Unpopular trades are still stinging (trade heat {heat:.0f}/100)."})
    backlash = len(prof.get("season_backlash_events") or [])
    if backlash:
        reasons.append({"tone": "con", "text": f"{backlash} fan backlash moment{'s' if backlash != 1 else ''} this season."})
    trust = _sf(prof.get("fan_trust_in_gm"), 58.0)
    if trust < 50:
        reasons.append({"tone": "con", "text": f"Fans doubt the front office (trust {trust:.0f}/100)."})
    elif trust >= 70:
        reasons.append({"tone": "pro", "text": f"Fans trust the front office (trust {trust:.0f}/100)."})

    effects: List[Dict[str, str]] = []
    try:
        from services.league_operations import calculate_team_revenue

        row = calculate_team_revenue(session, team, uid, is_user=True, annual=True) if team is not None else {}
        rev = _sf(row.get("revenue"))
        fan_mult = 0.70 + 0.45 * _clamp(v, 0.0, 100.0) / 100.0
        neutral = 0.70 + 0.45 * 0.55
        gate = rev * (fan_mult - neutral) / max(0.1, fan_mult)
        effects.append({
            "label": "Gate, merch and sponsorship",
            "text": f"{'+' if gate >= 0 else '-'}${abs(gate):.1f}M a year vs. average fans",
            "tone": "pro" if gate >= 0 else "con",
        })
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from app.sim_engine.league.game_rules import fan_home_ice_share

        share = fan_home_ice_share(v) * 100.0
        effects.append({
            "label": "Home-ice edge",
            "text": f"{share:+.0f}% of the normal home advantage",
            "tone": "pro" if share >= 0 else "con",
        })
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from services.contract_economy import fan_fa_interest_points

        fa_pts = fan_fa_interest_points(v)
    except Exception:
        fa_pts = 0.0
    effects.append({
        "label": "Free-agent appeal",
        "text": f"{fa_pts:+.0f} interest with every free agent ({fa_pts / 2:+.0f} with your own players)",
        "tone": "pro" if fa_pts >= 0 else "con",
    })
    try:
        from app.sim_engine.franchise.storyline_engine import fan_mood_weekly_deltas

        dm, db = fan_mood_weekly_deltas(v)
        if dm or db:
            effects.append({
                "label": "Locker room",
                "text": f"Morale {dm:+.1f} and belonging {db:+.1f} a week for your players",
                "tone": "pro" if dm > 0 else "con",
            })
        else:
            effects.append({"label": "Locker room", "text": "No crowd effect on the room (fans are steady)", "tone": "mixed"})
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    mult = fan_lobby_mult(v)
    effects.append({
        "label": "Board lobbying",
        "text": f"x{mult:.2f} sway per call" + (", reaches one more governor" if v >= 80 else ""),
        "tone": "pro" if mult >= 1.0 else "con",
    })
    vg = FAN_VALUE_COEF * (v - 55.0)
    effects.append({
        "label": "Franchise value growth",
        "text": f"{vg:+.1f} pts a season",
        "tone": "pro" if vg >= 0 else "con",
    })
    return {
        "value": round(v, 1),
        "tier": fan_tier(v),
        "delta": round(v - last, 1),
        "baseline": round(baseline, 1),
        "reasons": reasons,
        "effects": effects,
        "how_to_raise": [
            "Win: the standings set what fans settle at.",
            "Keep popular stars; trading a fan favourite costs trust.",
            "Back players publicly at press conferences.",
            "Hot streaks and breakout seasons add short-term buzz.",
        ],
    }


def build_governance_payload(session: Any) -> Dict[str, Any]:
    gov = ensure_governance(session)
    sy = _season(session)
    meeting = (gov.get("meetings") or {}).get(str(sy)) or {}
    mods = gov.get("modifiers") or compute_modifiers(gov)
    values = []
    for tid, row in (gov.get("franchise_values") or {}).items():
        team = (getattr(session, "team_by_id", None) or {}).get(str(tid))
        if team is None or not isinstance(row, dict):
            continue
        hist = row.get("history") or []
        values.append({
            "team_id": str(tid),
            "abbr": _team_abbr(team),
            "name": _team_name(team),
            "value_b": round(_sf(row.get("value_b")), 2),
            "change_pct": hist[-1].get("change_pct") if hist else 0.0,
            "drivers": hist[-1].get("drivers") if hist else [],
        })
    values.sort(key=lambda r: -r["value_b"])
    rulebook = [
        {"rule_id": rid, **{k: v for k, v in row.items() if k != "effects"}, "effects": _effect_rows(row.get("effects") or {})}
        for rid, row in (gov.get("rulebook") or {}).items()
    ]
    rulebook.sort(key=lambda r: (-int(r.get("season") or 0), r["rule_id"]))
    return {
        "season": sy,
        "season_label": f"{sy}-{(sy + 1) % 100:02d}",
        "meeting": {
            "status": meeting.get("status") or "none",
            "n_teams": meeting.get("n_teams"),
            "lobby_tokens": meeting.get("lobby_tokens", 0),
            "proposals": [public_proposal(p, reveal=True) for p in meeting.get("proposals") or []],
        },
        "modifiers": _effect_rows({k: v for k, v in mods.items() if k in EFFECT_SPECS}, scaled=True),
        "rulebook": rulebook,
        "history": list(reversed(gov.get("history") or []))[:40],
        "franchise_values": values,
        "relocations": list(gov.get("relocations") or [])[-6:],
        "expansion": {
            "pending": [
                {**e, "draft_offseason": int(e.get("start_season") or 0) - 1}
                for e in (gov.get("expansion", {}).get("pending") or [])
            ],
            "joined": gov.get("expansion", {}).get("joined") or [],
            "drafts": (gov.get("expansion", {}).get("drafts") or [])[-2:],
        },
        "pending_cap_adjust_m": round(_sf(gov.get("pending_cap_adjust_m")), 2),
        "user_impact": _user_impact(session),
        "catalog_size": len(RULES),
        "rules_never_proposed": sum(1 for r in RULES if r["id"] not in (gov.get("proposed_log") or {})),
        "expansion_draft": _safe_protection_payload(session),
        "on_ice": _on_ice_summary(session),
        "fan_ticker": _safe(lambda: fan_ticker(session), []),
        "fan_report": _safe(lambda: user_fan_report(session), {}),
    }


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
        return default


def _safe_protection_payload(session: Any) -> Dict[str, Any]:
    try:
        return expansion_protection_payload(session)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
        return {"due": False}


def _on_ice_summary(session: Any) -> Dict[str, Any]:
    """Current on-ice rulebook (points system, OT, shootout) for the UI."""
    try:
        from app.sim_engine.league.game_rules import points_label, rules_for

        r = rules_for(_league(session))
        return {
            "points": points_label(r),
            "ot_minutes": int(r["ot_minutes"]),
            "shootout": not r["ties_allowed"],
            "scoring_pct": round((r["scoring_mult"] - 1.0) * 100.0, 1),
            "penalty_pct": round((r["penalty_mult"] - 1.0) * 100.0, 1),
            "pp_pct": round((r["pp_mult"] - 1.0) * 100.0, 1),
        }
    except Exception:
        return {}
