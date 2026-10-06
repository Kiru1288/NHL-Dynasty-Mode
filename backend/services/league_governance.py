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

PROPOSALS_PER_MEETING = 5
LOBBY_TOKENS_PER_MEETING = 2
LOBBY_SWING = 0.18
LOBBY_TARGETS = 4
#: Proposals reach the Board after committee vetting, so governors start out leaning
#: toward approval; lean and personality decide the close ones. Lower bars get less.
BOARD_VETTING_BIAS = {"majority": 0.02, "two_thirds": 0.15, "three_quarters": 0.15}
#: Chance a meeting includes a repeal vote once the rulebook has this many rules.
REPEAL_MIN_RULES = 5
REPEAL_CHANCE = 0.45

#: Hard bounds on the combined effect of every rule in force (so 30 seasons of
#: votes can't stack into absurd numbers).
EFFECT_LIMITS: Dict[str, Tuple[float, float]] = {
    "rev_all": (-8.0, 10.0),
    "rev_small": (-6.0, 8.0),
    "rev_large": (-6.0, 6.0),
    "revenue_share": (-4.0, 6.0),
    "opex": (-3.0, 4.0),
    "star_rev": (-20.0, 40.0),
    "fan": (-6.0, 6.0),
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
    "trade_volume": (-30.0, 40.0),
    "trade_demand_rate": (-40.0, 40.0),
    "scouting_budget": (-30.0, 40.0),
    "injury_rate": (-15.0, 10.0),
    "relocation_ease": (-0.15, 0.15),
    "expansion_pressure": (-40.0, 50.0),
}
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
        pass


def ensure_governance(session: Any) -> Dict[str, Any]:
    gov = getattr(session, "league_governance", None)
    if not isinstance(gov, dict):
        gov = {}
        try:
            session.league_governance = gov
        except Exception:
            pass
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
                mods[k] = mods.get(k, 0.0) + float(v)
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
            pass
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
    return out


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


def _relocation_candidate(session: Any, traits: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    gov = ensure_governance(session)
    ease = gov_mod(session, "relocation_ease")
    recent = {str(r.get("team_id")) for r in gov.get("relocations") or [] if _season(session) - int(r.get("season") or 0) < 8}
    best = None
    for tid, t in traits.items():
        if tid in recent:
            continue
        risk = t["relocation_risk"] + ease
        if t["tier"] != "small" and t["profit"] >= 0:
            continue
        if risk < 0.5 or t["profit"] > 2.0:
            continue
        if best is None or risk > best[1]:
            best = (tid, risk)
    if best is None:
        return None
    used = set(gov.get("used_markets") or [])
    existing = {t["abbr"] for t in traits.values()}
    cur_tier = traits[best[0]]["tier"]
    tier_rank = {"small": 0, "medium": 1, "large": 2}
    options = [
        m for m in CANDIDATE_MARKETS
        if m["abbr"] not in used and m["abbr"] not in existing and tier_rank[m["tier"]] >= tier_rank.get(cur_tier, 0)
    ]
    if not options:
        return None
    pick = options[int(_unit(_salt(session), "reloc", _season(session), best[0]) * len(options)) % len(options)]
    return {"team_id": best[0], "risk": round(best[1], 3), "market": pick}


def _expansion_candidate(session: Any, traits: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    if _unit(_salt(session), "expansion", _season(session)) > chance:
        return None
    used = set(gov.get("used_markets") or [])
    existing = {t["abbr"] for t in traits.values()}
    options = [m for m in CANDIDATE_MARKETS if m["abbr"] not in used and m["abbr"] not in existing]
    options.sort(key=lambda m: -m["value_b"])
    if not options:
        return None
    count = 2 if (n + 2 <= MAX_LEAGUE_TEAMS and n % 2 == 0 and len(options) >= 2) else 1
    top = options[: max(4, count)]
    rng = random.Random(_salt(session) * 31 + _season(session))
    cities = rng.sample(top, count)
    vals = [v.get("value_b") for v in (gov.get("franchise_values") or {}).values() if isinstance(v, dict)]
    avg_val = (sum(_sf(v) for v in vals) / len(vals)) if vals else 2.55
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
    active = set((gov.get("rulebook") or {}).keys())
    recent = {rid for rid, y in (gov.get("proposed_log") or {}).items() if sy - int(y) < RECENT_PROPOSAL_YEARS}
    pool = [r for r in RULES if r["id"] not in active and r["id"] not in recent]
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
    # they already live with (60% weight), and a few always want it gone.
    sign = -0.6 if repeal else 1.0
    th = _threshold_for(rule["cat"])
    p = 0.5 + BOARD_VETTING_BIAS.get(th, 0.1) + sign * float(rule.get("base") or 0.0)
    pos, neg = ("", 0.0), ("", 0.0)
    for trait, w in (rule.get("lean") or {}).items():
        if trait in t["traits"]:
            w = sign * float(w)
            p += w
            if w > pos[1]:
                pos = (_TRAIT_COPY.get(trait, trait), w)
            if -w > neg[1]:
                neg = (_TRAIT_COPY.get(trait, trait), -w)
    taste = _taste(session, tid, pid)
    p += taste
    reasons = {
        "yes": (f"{pos[0]} — helps them" if pos[0] else ("Owner likes it" if taste > 0.03 else "Backs the committee")),
        "no": (f"{neg[0]} — hurts them" if neg[0] else ("Owner unconvinced" if taste < -0.03 else "Sees no upside")),
    }
    return _clamp(p, 0.04, 0.96), reasons


def _relocation_probability(prop: Dict[str, Any], t: Dict[str, Any], session: Any, tid: str) -> Tuple[float, Dict[str, str]]:
    if tid == str(prop["team_id"]):
        return 0.97, {"yes": "Their own club is moving", "no": "Their own club is moving"}
    p = 0.68
    yes_r, no_r = "Better market for the league", "Wants the club to stay put"
    mover = prop.get("_mover_traits") or {}
    if "poor" in (mover.get("traits") or ()):
        p += 0.1
        yes_r = "Club is losing money where it is"
    if t["canadian"] and prop["market"]["abbr"] not in CANADA_ABBRS and mover.get("canadian"):
        p -= 0.25
        no_r = "Won't lose a Canadian club"
    if "small" in t["traits"]:
        p -= 0.05
        no_r = "Small markets fear they're next"
    p += gov_mod(session, "relocation_ease")
    p += _taste(session, tid, prop["id"])
    return _clamp(p, 0.05, 0.95), {"yes": yes_r, "no": no_r}


def _expansion_probability(prop: Dict[str, Any], t: Dict[str, Any], session: Any, tid: str) -> Tuple[float, Dict[str, str]]:
    share_m = prop["fee_b"] * 1000.0 / max(1, prop.get("_n_teams") or 32)
    p = 0.72 + gov_mod(session, "expansion_pressure") / 100.0
    yes_r, no_r = f"Takes the ${share_m:.0f}M fee share", "Doesn't want to share revenue"
    if "poor" in t["traits"]:
        p += 0.15
        yes_r = f"Needs the ${share_m:.0f}M fee share"
    if "large" in t["traits"]:
        p -= 0.06
        no_r = "Dilutes national TV money"
    if "contender" in t["traits"]:
        p -= 0.04
        no_r = "Won't lose a player to the expansion draft"
    p += _taste(session, tid, prop["id"])
    return _clamp(p, 0.05, 0.95), {"yes": yes_r, "no": no_r}


def _vote_climate(session: Any, prop: Dict[str, Any]) -> Tuple[str, float]:
    """Room mood for one proposal. Votes are correlated, not 31 coin flips: some items
    are housekeeping that sails through unanimously, some die on arrival, some split
    the room along market lines, the rest drift with a shared lean."""
    m = _unit(_salt(session), "climate", prop["id"])
    drift = (_unit(_salt(session), "climate-drift", prop["id"]) * 2.0 - 1.0) * 0.2
    if prop.get("kind") in ("relocation", "expansion"):
        # Big structural votes are rarely unanimous.
        if m < 0.35:
            return "split", drift
        return "normal", drift
    if m < 0.16:
        return "consensus", 0.0
    if m < 0.28:
        return "doa", 0.0
    if m < 0.55:
        return "split", drift * 0.5
    return "normal", drift


def _apply_climate(p: float, climate: str, drift: float) -> float:
    if climate == "consensus":
        return 0.5 + (p - 0.5) * 0.35 + 0.42
    if climate == "doa":
        return 0.5 + (p - 0.5) * 0.35 - 0.36
    if climate == "split":
        return 0.5 + (p - 0.5) * 2.3 + drift
    return p + drift


def _preroll_ballots(session: Any, prop: Dict[str, Any], traits: Dict[str, Dict[str, Any]]) -> None:
    uid = str(getattr(session, "user_team_id", "") or "")
    ballots = []
    expected = 0.0
    climate, drift = _vote_climate(session, prop)
    prop["_climate"] = climate
    for tid, t in traits.items():
        if tid == uid:
            continue
        if prop["kind"] == "relocation":
            p, reason = _relocation_probability(prop, t, session, tid)
        elif prop["kind"] == "expansion":
            p, reason = _expansion_probability(prop, t, session, tid)
        else:
            p, reason = _rule_probability(RULE_BY_ID[prop["rule_id"]], t, session, tid, prop["id"], repeal=prop["kind"] == "repeal")
        if not (prop["kind"] == "relocation" and tid == str(prop.get("team_id"))):
            p = _clamp(_apply_climate(p, climate, drift), 0.02, 0.98)
        u = _unit(_salt(session), "ballot", prop["id"], tid)
        ballots.append({"team_id": tid, "abbr": t["abbr"], "name": t["name"], "p": round(p, 4), "u": round(u, 4), "reason": reason, "lobbied": 0.0})
        expected += p
    prop["_ballots"] = ballots
    prop["expected_yes_cpu"] = round(expected, 2)


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


def _effect_rows(effects: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    for k, v in effects.items():
        label, kind, _consumer = EFFECT_SPECS.get(k, (k, "pts", ""))
        v = float(v)
        if kind == "pct":
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
}


def _effect_tone(key: str, value: float) -> Tuple[str, str]:
    good, up, down = _EFFECT_TONE.get(key, (0, "Changes the landscape", "Changes the landscape"))
    note = up if value >= 0 else down
    if good == 0:
        return "mixed", note
    return ("pro" if (value >= 0) == (good > 0) else "con"), note


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
    advance_franchise_values(session, rows)
    compute_star_spikes(session, rows)
    apply_revenue_scouting_budget(session, rows)
    traits = team_traits(session, rows)
    n_teams = len(traits)

    proposals: List[Dict[str, Any]] = []
    reloc = _relocation_candidate(session, traits)
    expn = _expansion_candidate(session, traits)
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
            "effects": _effect_rows({k: -float(v) for k, v in rule["effects"].items() if k not in ABSOLUTE_EFFECTS}),
            "pros_cons": _pros_cons({k: -float(v) for k, v in rule["effects"].items() if k not in ABSOLUTE_EFFECTS}),
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
                f"{mover['name']} ({mover['abbr']}) lost ${abs(mover['profit']):.1f}M"
                if mover["profit"] < 0 else f"{mover['name']} ({mover['abbr']}) is a small market under pressure"
            ) + f"; ownership applies to move to {m['city']} as the {m['city']} {nick}.",
            "effects": [
                {"key": "relocation", "label": "Moves to", "value": 0, "text": f"{m['city']} ({m['tier']} market)"},
                {"key": "demands", "label": "Player reaction", "value": 0, "text": "Some veterans may demand trades"},
                {"key": "fans", "label": "New market", "value": 0, "text": "Fan honeymoon + revenue bump for 2 seasons"},
            ],
            "pros_cons": {
                "pros": [f"Move from a struggling market to {m['city']} ({m['tier']})", "Relocated club's value and revenue jump", "New-market honeymoon for two seasons"],
                "cons": [f"{mover['city'] if 'city' in mover else 'The old city'} loses its team: fan backlash league-wide", "Veterans on the moving club may demand trades", "Sets a precedent: other small markets feel the heat"],
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
                f"Expansion fee ${expn['fee_b']:.2f}B (≈${share_m:.0f}M to every current club); an expansion draft takes one player from each club."
            ),
            "effects": [
                {"key": "fee", "label": "Fee share per club", "value": share_m, "text": f"+${share_m:.0f}M once"},
                {"key": "teams", "label": "League size", "value": len(expn["cities"]), "text": f"{n_teams} → {n_teams + len(expn['cities'])} clubs"},
                {"key": "draft", "label": "Expansion draft", "value": 0, "text": "Protect 7F / 3D / 1G; lose one player"},
            ],
            "pros_cons": {
                "pros": [f"Every club gets ≈${share_m:.0f}M in fee money", "Bigger league: more markets and TV money long term"],
                "cons": ["Every club loses one unprotected player in the expansion draft", "National TV money split more ways", "Weaker draft positions as new clubs pick high"],
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
    traits = team_traits(session)
    my = traits.get(uid, {}).get("traits", set())
    scored = []
    for b in prop["_ballots"]:
        t = traits.get(b["team_id"], {})
        overlap = len(my & t.get("traits", set()))
        persuadable = 1.0 - abs(b["p"] - 0.5) * 2.0
        scored.append((overlap * 0.4 + persuadable, b))
    scored.sort(key=lambda s: -s[0])
    swayed = []
    for _score, b in scored[:LOBBY_TARGETS]:
        b["lobbied"] = LOBBY_SWING if want_yes else -LOBBY_SWING
        swayed.append(b["abbr"])
    prop["lobbied"] = {"side": "yes" if want_yes else "no", "teams": swayed}
    meeting["lobby_tokens"] = int(meeting.get("lobby_tokens") or 0) - 1
    prop["forecast"] = _forecast(prop)
    return build_governance_payload(session)


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
        if b.get("lobbied") and ((b["lobbied"] > 0) == (cast == "yes")):
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
        pass


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
        return
    if prop["kind"] == "repeal":
        gov["rulebook"].pop(str(prop["rule_id"]), None)
        sync_governance_to_league(session)
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
    return build_governance_payload(session)


def consume_pending_cap_adjust(session: Any) -> float:
    """Called by the cap rollover: returns and clears one-time cap adjustments."""
    gov = ensure_governance(session)
    adj = round(_sf(gov.get("pending_cap_adjust_m")), 3)
    if adj:
        gov["applied_cap_adjust"].append({"season": _season(session) + 1, "adjust_m": adj})
    gov["pending_cap_adjust_m"] = 0.0
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
        g += (fan - 55.0) / 100.0 * 0.10
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
        pass
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
        pass
    try:
        mp = getattr(team, "market", None)
        if mp is not None:
            mp.market_size = market["tier"]
    except Exception:
        pass
    sy = _season(session)
    vals = ensure_franchise_values(session)
    row = vals.get(str(team_id))
    if isinstance(row, dict):
        new_v = round(max(_sf(row.get("value_b")) * 1.15, _sf(market.get("value_b"), 1.6)), 3)
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
        pass
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
            pass
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


def _protected_ids(team: Any) -> set:
    org = [p for p in list(getattr(team, "roster", None) or []) + list(getattr(team, "ahl_roster", None) or []) if not getattr(p, "retired", False)]
    keep: set = set()
    for grp, n in (("F", 7), ("D", 3), ("G", 1)):
        ps = sorted([p for p in org if _pos_group(p) == grp], key=_ovr, reverse=True)
        keep.update(str(getattr(p, "id", "")) for p in ps[:n])
    for p in org:  # first/second-year pros are exempt
        if _age(p) <= 21:
            keep.add(str(getattr(p, "id", "")))
    return keep


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
            conf = c.get("conference") or "Western"
            divs: Dict[str, int] = {}
            for t in list(league.teams) + new_teams:
                if str(getattr(t, "conference", "")) == conf:
                    d = str(getattr(t, "division", "") or "")
                    divs[d] = divs.get(d, 0) + 1
            division = min(divs, key=lambda d: divs[d]) if divs else conf
            nick_pool = [n for n in EXPANSION_NICKNAMES if n not in {str(getattr(t, 'name', '')) for t in league.teams}]
            nickname = nick_pool[int(rng.random() * len(nick_pool)) % len(nick_pool)] if nick_pool else "Expansion"
            team = Team(team_id=new_id, city=c["city"], name=nickname, division=division, conference=conf, archetype="balanced", rng=random.Random(rng.random()))
            team.abbr = c["abbr"]
            try:
                team.abbreviation = c["abbr"]
                team.market.market_size = c.get("tier", "medium")
            except Exception:
                pass
            new_teams.append(team)
        # Expansion draft: each new club takes one unprotected contracted player per club.
        picks_log: List[Dict[str, Any]] = []
        needs = {t: {"F": 14, "D": 9, "G": 3} for t in new_teams}
        for src in existing:
            protected = _protected_ids(src)
            pool = [
                p for p in list(getattr(src, "roster", None) or []) + list(getattr(src, "ahl_roster", None) or [])
                if str(getattr(p, "id", "")) not in protected and _has_contract(p) and not getattr(p, "retired", False)
            ]
            for nt in new_teams:
                if not pool:
                    break
                need = needs[nt]

                def _score(p: Any) -> float:
                    g = _pos_group(p)
                    return _ovr(p) + (6.0 if need.get(g, 0) > 0 else -4.0) - max(0, _age(p) - 31) * 1.5

                pool.sort(key=_score, reverse=True)
                p = pool.pop(0)
                pid = str(getattr(p, "id", ""))
                _purge_player_id_from_team_lists(src, pid)
                nt.roster.append(p)
                for field in ("team_id", "current_team_id"):
                    try:
                        setattr(p, field, nt.team_id)
                    except Exception:
                        pass
                need[_pos_group(p)] = need.get(_pos_group(p), 0) - 1
                picks_log.append({"team": c_abbr(nt), "player": str(getattr(getattr(p, "identity", None), "name", "") or ""), "from": _team_abbr(src), "ovr": round(_ovr(p), 1)})
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
            gov["franchise_values"][tid] = {
                "value_b": float(next((c["value_b"] for c in entry["cities"] if c["abbr"] == c_abbr(nt)), 1.8)),
                "history": [{"season": int(next_season_year), "value_b": float(next((c["value_b"] for c in entry["cities"] if c["abbr"] == c_abbr(nt)), 1.8)), "change_pct": 0.0, "drivers": ["Expansion entry"]}],
            }
            gov["honeymoon"][tid] = {str(next_season_year): 18.0, str(next_season_year + 1): 9.0}
            joined.append({"team_id": tid, "city": nt.city, "name": nt.name, "abbr": c_abbr(nt), "season": int(next_season_year), "players": len(nhl) + len(ahl)})
            _news(session, f"The {nt.city} {nt.name} are here", f"{nt.city} joins the league for {next_season_year}-{(next_season_year + 1) % 100:02d} with {len(nhl) + len(ahl)} players from the expansion draft.", team_id=tid, kind="expansion")
        exp.setdefault("drafts", []).append({"season": int(next_season_year), "picks": picks_log})
    exp["pending"] = [e for e in exp.get("pending") or [] if e not in due]
    exp["joined"] = list(exp.get("joined") or []) + joined
    try:
        from app.sim_engine.trades.trade_pick_registry import ensure_franchise_pick_registry

        ensure_franchise_pick_registry(league, season_calendar_year=int(next_season_year), years_ahead=4)
    except Exception:
        pass
    return joined


def c_abbr(team: Any) -> str:
    return str(getattr(team, "abbr", "") or "").upper()


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
        pass
    try:
        from services.franchise_offseason import team_signing_bonus_eligibility

        elig = team_signing_bonus_eligibility(session, uid)
        out["bonus_eligible"] = bool(elig.get("eligible"))
        out["bonus_max_pct"] = elig.get("max_bonus_pct")
        out["bonus_floor_m"] = elig.get("floor_m")
    except Exception:
        pass
    try:
        from services.franchise_scouting import _ensure_scouting_state

        out["scouting_budget"] = float(_ensure_scouting_state(session).get("budget") or 0)
    except Exception:
        pass
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
        "modifiers": _effect_rows({k: v for k, v in mods.items() if k in EFFECT_SPECS}),
        "rulebook": rulebook,
        "history": list(reversed(gov.get("history") or []))[:40],
        "franchise_values": values,
        "relocations": list(gov.get("relocations") or [])[-6:],
        "expansion": {
            "pending": gov.get("expansion", {}).get("pending") or [],
            "joined": gov.get("expansion", {}).get("joined") or [],
            "drafts": (gov.get("expansion", {}).get("drafts") or [])[-2:],
        },
        "pending_cap_adjust_m": round(_sf(gov.get("pending_cap_adjust_m")), 2),
        "user_impact": _user_impact(session),
        "catalog_size": len(RULES),
    }
