"""Derived measures: numbers the save does not store directly but can be built from it.

Player measures read the season ledger (even-strength vs power-play scoring,
ice time, on-ice goals and expected goals), ratings, contracts and market value.
Team measures read rosters, standings and the same ledger. Every function is
pure over a LiveView and cached on it for the life of the view.

Nothing here is invented: when a input is missing the measure is None and the
player or team is left out of the ranking.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from .metrics import metric_value

Num = Optional[float]


def _st(view: Any, pid: str) -> Dict[str, Any]:
    return view.stats.get(pid) or {}


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _cache(view: Any) -> Dict[str, Any]:
    c = getattr(view, "_derived", None)
    if c is None:
        c = {}
        view._derived = c
    return c


def _skater(row: Dict[str, Any]) -> bool:
    return bool(row) and row.get("pos") != "G"


# --------------------------------------------------------------------------- aging and value
def age_factor(age: int) -> float:
    """Expected change in a player's value per season at this age."""
    if age <= 21:
        return 1.08
    if age <= 24:
        return 1.04
    if age <= 27:
        return 1.0
    if age <= 30:
        return 0.96
    if age <= 33:
        return 0.90
    return 0.84


def market(view: Any, pid: str) -> Num:
    c = _cache(view).setdefault("market", {})
    if pid not in c:
        try:
            from services.contract_economy import compute_market_value

            c[pid] = float(compute_market_value(view.players[pid], view.league))
        except Exception:
            c[pid] = None
    return c[pid]


def surplus_years(view: Any, pid: str, years: int = 3) -> Num:
    """Market value minus cap hit, summed over the next ``years`` seasons with aging."""
    r = view.row(pid)
    mv = market(view, pid)
    if not r or mv is None:
        return None
    cap = r["contract"]["cap_hit_m"] or 0.95  # unsigned prospects: entry-level money
    left = max(1, min(years, r["contract"]["years_left"] or years))
    total, v, age = 0.0, mv, r["age"]
    for y in range(left):
        total += v - cap
        v *= age_factor(age + y)
    return round(total, 2)


def contract_risk_7(view: Any, pid: str) -> Num:
    """Overpay on a seven-year deal signed at today's market value, as the player ages."""
    r = view.row(pid)
    mv = market(view, pid)
    if not r or mv is None or mv < 2.0:
        return None
    v, age, over = mv, r["age"], 0.0
    for y in range(7):
        over += max(0.0, mv - v)
        v *= age_factor(age + y)
    inj = metric_value(view, pid, "hd:injury_proneness_inv") if False else None
    return round(over, 2)


def unmovable(view: Any, pid: str) -> Num:
    r = view.row(pid)
    mv = market(view, pid)
    if not r or mv is None or not r["contract"]["cap_hit_m"]:
        return None
    yl = r["contract"]["years_left"] or 0
    gap = r["contract"]["cap_hit_m"] - mv
    if yl < 2 or gap <= 0.5:
        return 0.0
    clause = 1.25 if r["contract"]["nmc"] or r["contract"]["ntc"] else 1.0
    future = sum(r["contract"]["cap_hit_m"] - mv * (age_factor(r["age"]) ** y) for y in range(yl))
    return round(future * clause, 2)


def aging_risk(view: Any, pid: str) -> Num:
    r = view.row(pid)
    if not r:
        return None
    age = r["age"]
    speed = metric_value(view, pid, "ch:transition") or 75
    phys = metric_value(view, pid, "ch:physical") or 75
    dur = metric_value(view, pid, "hd:durability") or metric_value(view, pid, "hd:injury_resistance")
    score = max(0, age - 27) * 6 + (speed - 75) * 0.25 + (80 - (dur or 80)) * 0.3 - (phys - 75) * 0.05
    if r["pos"] == "G":
        score -= 6  # goalies age later
    return round(score, 1)


# --------------------------------------------------------------------------- production and ice time
def toi_pg(view: Any, pid: str) -> Num:
    st = _st(view, pid)
    gp = _f(st.get("gp"))
    return _f(st.get("toi_sec")) / gp / 60.0 if gp else None


def p60(view: Any, pid: str) -> Num:
    st = _st(view, pid)
    toi = _f(st.get("toi_sec"))
    return _f(st.get("pts")) / toi * 3600 if toi > 600 else None


def ev_p60(view: Any, pid: str) -> Num:
    st = _st(view, pid)
    ev = _f(st.get("ev_toi_sec"))
    if ev < 600:
        return None
    ev_pts = _f(st.get("pts")) - _f(st.get("ppg")) - _f(st.get("ppa")) - _f(st.get("shg")) - _f(st.get("sha"))
    return max(0.0, ev_pts) / ev * 3600


def pp_p60(view: Any, pid: str) -> Num:
    st = _st(view, pid)
    pp_toi = _f(st.get("toi_sec")) - _f(st.get("ev_toi_sec"))
    if pp_toi < 120:
        return None
    return (_f(st.get("ppg")) + _f(st.get("ppa"))) / pp_toi * 3600


def pp_share(view: Any, pid: str) -> Num:
    st = _st(view, pid)
    pts = _f(st.get("pts"))
    pp = _f(st.get("ppg")) + _f(st.get("ppa"))
    # Share weighted toward volume so a 4-of-5 start doesn't top the list.
    return round(pp / pts * min(1.0, pts / 15.0), 3) if pts >= 8 else None


def _pctile(view: Any, key: str, fn: Callable[[Any, str], Num]) -> Dict[str, float]:
    c = _cache(view).setdefault("pct", {})
    if key in c:
        return c[key]
    vals = []
    for pid, (t, lv) in view.where.items():
        if lv != "NHL":
            continue
        r = view.row(pid)
        if not _skater(r) or (r.get("gp") or 0) < 5:
            continue
        v = fn(view, pid)
        if v is not None:
            vals.append((v, pid))
    vals.sort()
    n = max(1, len(vals) - 1)
    c[key] = {pid: i / n for i, (_v, pid) in enumerate(vals)}
    return c[key]


def prod_vs_toi(view: Any, pid: str) -> Num:
    """Scoring-rate percentile minus ice-time percentile: positive = producing beyond his minutes."""
    a = _pctile(view, "p60", p60).get(pid)
    b = _pctile(view, "toi", toi_pg).get(pid)
    return round((a - b) * 100, 1) if a is not None and b is not None else None


def line_tier(view: Any, pid: str) -> Optional[str]:
    """first/second/third/fourth line (or pair) by ice-time rank on his team."""
    r = view.row(pid)
    if not _skater(r):
        return None
    mates = [x for x in view.team_rows(r["team_id"]) if _skater(x) and ((x["pos"] in ("D", "LD", "RD")) == (r["pos"] in ("D", "LD", "RD")))]
    mates.sort(key=lambda x: -(toi_pg(view, x["id"]) or 0))
    idx = next((i for i, x in enumerate(mates) if x["id"] == pid), None)
    if idx is None:
        return None
    if r["pos"] in ("D", "LD", "RD"):
        return ["first pair", "second pair", "third pair"][min(2, idx // 2)]
    return ["first line", "second line", "third line", "fourth line"][min(3, idx // 3)]


# --------------------------------------------------------------------------- on-ice impact and luck
def team_gf_pct(view: Any, tid: str) -> Num:
    st = next((r for r in view.standings() if str(r.get("team_id")) == str(tid)), None)
    if not st:
        return None
    gf, ga = _f(st.get("gf")), _f(st.get("ga"))
    return gf / (gf + ga) if gf + ga else None


def onice_rel(view: Any, pid: str) -> Num:
    """On-ice goal share minus his team's goal share, in points of percentage."""
    st = _st(view, pid)
    r = view.row(pid)
    gf, ga = _f(st.get("gf_on")), _f(st.get("ga_on"))
    t = team_gf_pct(view, r.get("team_id")) if r else None
    if gf + ga < 6 or t is None:
        return None
    return round((gf / (gf + ga) - t) * 100, 1)


def xg_share(view: Any, pid: str) -> Num:
    st = _st(view, pid)
    xf, xa = _f(st.get("xgf")), _f(st.get("xga"))
    return round(xf / (xf + xa) * 100, 1) if xf + xa > 3 else None


def ipp(view: Any, pid: str) -> Num:
    """Share of on-ice goals for that he scored or assisted."""
    st = _st(view, pid)
    gf = _f(st.get("gf_on"))
    return round(_f(st.get("pts")) / gf * 100, 1) if gf >= 6 else None


def carried(view: Any, pid: str) -> Num:
    """Good on-ice results he takes little part in: high goal share, low involvement."""
    rel, inv = onice_rel(view, pid), ipp(view, pid)
    if rel is None or inv is None:
        return None
    return round(rel - (inv - 60) * 0.4, 1)


def elevates(view: Any, pid: str) -> Num:
    """Team does much better with him on the ice and he drives it."""
    rel, inv, xs = onice_rel(view, pid), ipp(view, pid), xg_share(view, pid)
    if rel is None:
        return None
    return round(rel + ((xs or 50) - 50) * 0.5 + ((inv or 60) - 60) * 0.1, 1)


def hurts_linemates(view: Any, pid: str) -> Num:
    """Decent points but the team is outscored with him on the ice."""
    rel = onice_rel(view, pid)
    pct = _pctile(view, "pts", lambda v, p: _f(_st(v, p).get("pts"))).get(pid)
    if rel is None or pct is None or pct < 0.5:
        return None
    return round(-rel + pct * 5, 1)


def luck(view: Any, pid: str) -> Num:
    """Goals above expected, plus on-ice goal differential above expected. Positive = lucky."""
    st = _st(view, pid)
    if _f(st.get("gp")) < 5:
        return None
    r = view.row(pid)
    if r and r["pos"] == "G":
        sa, ga, xga = _f(st.get("shots_against") or st.get("goalie_shots_against")), _f(st.get("goalie_ga") or st.get("ga")), _f(st.get("goalie_xga") or st.get("xga"))
        return round(xga - ga, 2) if sa > 50 else None
    own = _f(st.get("g")) - _f(st.get("ixg"))
    on = (_f(st.get("gf_on")) - _f(st.get("xgf"))) - (_f(st.get("ga_on")) - _f(st.get("xga")))
    return round(own + 0.5 * on, 2)


def sh_pct(view: Any, pid: str) -> Num:
    st = _st(view, pid)
    sog = _f(st.get("sog"))
    return round(_f(st.get("g")) / sog * 100, 1) if sog >= 15 else None


# --------------------------------------------------------------------------- development
def form(view: Any, pid: str) -> Num:
    """Points above the line his overall should be scoring (league-centred)."""
    try:
        from app.sim_engine.progression.development import _skater_form_perf_z

        p = view.players[pid]
        st = _st(view, pid)
        setattr(p, "_dev_season_points", int(_f(st.get("pts"))))
        if not _f(st.get("gp")):
            return None
        return round(float(_skater_form_perf_z(p) or 0.0), 2)
    except Exception:
        return None


def season_change(view: Any, pid: str) -> Num:
    r = view.row(pid)
    s = r.get("season_start_ovr") if r else None
    return round(r["ovr"] - float(s), 1) if s else None


def stagnation(view: Any, pid: str) -> Num:
    r = view.row(pid)
    if not r or r["age"] > 26 or (r["pot"] - r["ovr"]) < 3:
        return None
    score = 0.0
    ch = season_change(view, pid)
    score += 3 if (ch is not None and ch <= 0) else 0
    t = toi_pg(view, pid)
    if r["level"] == "NHL":
        score += 2 if (t or 0) < 12 else 0
    elif not r.get("gp"):
        score += 3  # not playing anywhere
    fm = form(view, pid)
    score += 2 if (fm is not None and fm < -0.3) else 0
    mor = metric_value(view, pid, "morale")
    score += 1 if (mor is not None and mor < 45) else 0
    score += max(0, r["age"] - 22) * 0.5
    return round(score, 1)


def gap_to_potential(view: Any, pid: str) -> Num:
    r = view.row(pid)
    return round(r["pot"] - r["ovr"], 1) if r else None


def lacks_tools(view: Any, pid: str) -> Num:
    """Elite potential but the attributes that get there are thin."""
    r = view.row(pid)
    if not r or r["pot"] < 88 or r["pos"] == "G":
        return None
    chs = [metric_value(view, pid, f"ch:{c}") for c in ("offence", "defence", "transition", "mental", "physical")]
    chs = [c for c in chs if c is not None]
    if not chs:
        return None
    return round(r["pot"] - statistics.mean(chs), 1)


def top_line_chance(view: Any, pid: str) -> Num:
    r = view.row(pid)
    if not r:
        return None
    fm = form(view, pid) or 0.0
    return round(r["pot"] * 0.7 + r["ovr"] * 0.3 + fm * 2, 1)


def surprise_elite(view: Any, pid: str) -> Num:
    r = view.row(pid)
    if not r or r["age"] > 24 or r["ovr"] > 86 or r["pot"] < 88:
        return None
    fm = form(view, pid) or 0.0
    ch = season_change(view, pid) or 0.0
    return round((r["pot"] - r["ovr"]) * 0.6 + fm * 4 + ch * 2, 1)


def skating_weighted(view: Any, pid: str) -> Num:
    r = view.row(pid)
    t = metric_value(view, pid, "ch:transition")
    return round((r["ovr"] + t) / 2, 1) if r and t is not None else None


def trait_rarity(view: Any, pid: str) -> Num:
    """How unusual his two best chapters are together, against all prospects."""
    c = _cache(view).setdefault("trait_stats", None)
    names = ("offence", "defence", "transition", "mental", "physical", "character")
    if c is None:
        pool = [p for p, (t, lv) in view.where.items() if lv in ("Prospect", "AHL", "ECHL")]
        stats = {}
        for n in names:
            vals = [metric_value(view, p, f"ch:{n}") for p in pool]
            vals = [v for v in vals if v is not None]
            stats[n] = (statistics.mean(vals), statistics.pstdev(vals) or 1.0) if len(vals) > 5 else (75.0, 8.0)
        _cache(view)["trait_stats"] = c = stats
    zs = []
    for n in names:
        v = metric_value(view, pid, f"ch:{n}")
        if v is not None:
            mu, sd = c[n]
            zs.append((v - mu) / sd)
    zs.sort(reverse=True)
    return round(zs[0] + zs[1], 2) if len(zs) >= 2 else None


def chemistry_total(view: Any, pid: str) -> Num:
    p = view.players.get(pid)
    rel = getattr(p, "chemistry_relationships", None)
    r = view.row(pid)
    if not isinstance(rel, dict) or not r:
        return None
    mates = {x["id"] for x in view.team_rows(r["team_id"])}
    vals = [float(v) for k, v in rel.items() if k in mates]
    lead = metric_value(view, pid, "hd:locker_room_presence") or 50
    return round(sum(max(0.0, v - 50) for v in vals) + (lead - 50) * 0.5, 1) if vals else None


def best_partner(view: Any, pid: str) -> Optional[Tuple[str, float]]:
    p = view.players.get(pid)
    rel = getattr(p, "chemistry_relationships", None)
    r = view.row(pid)
    if not isinstance(rel, dict) or not r:
        return None
    mates = {x["id"] for x in view.team_rows(r["team_id"])}
    best = max(((k, float(v)) for k, v in rel.items() if k in mates), key=lambda kv: kv[1], default=None)
    return best


def partner_lift(view: Any, pid: str) -> Num:
    """A defenseman whose results run ahead of his own rating next to his best-chemistry partner."""
    r = view.row(pid)
    if not r or r["pos"] not in ("D", "LD", "RD"):
        return None
    bp = best_partner(view, pid)
    rel = onice_rel(view, pid)
    if not bp or rel is None or bp[0] not in view.players:
        return None
    return round((view.row(bp[0])["ovr"] - r["ovr"]) * 0.6 + rel * 0.5, 1)


def draft_expectation(overall_pick: int) -> float:
    """What a pick slot usually becomes: #1 ~ 92, #32 ~ 84, #100 ~ 78, #200 ~ 74."""
    return 92 - 8 * (min(overall_pick, 224) ** 0.5 - 1) / 13.97 * 2.2


def draft_vs_slot(view: Any, pid: str) -> Num:
    try:
        from services.award_history import _draft_payload_from_player

        d = _draft_payload_from_player(view.players[pid])
    except Exception:
        return None
    ov = d.get("draft_overall_pick")
    r = view.row(pid)
    if not ov or not r:
        return None
    return round(r["ovr"] - draft_expectation(int(ov)), 1)


def blocked(view: Any, pid: str) -> Num:
    """A prospect good enough for a lineup spot that is locked by signed veterans."""
    r = view.row(pid)
    if not r or r["level"] not in ("AHL", "Prospect", "ECHL") or r["team_id"] != view.user_tid:
        return None
    grp = "D" if r["pos"] in ("D", "LD", "RD") else ("G" if r["pos"] == "G" else "F")
    slots = {"D": 6, "F": 12, "G": 2}[grp]
    nhl = sorted((x for x in view.team_rows(view.user_tid) if (x["pos"] in ("D", "LD", "RD")) == (grp == "D") and ((x["pos"] == "G") == (grp == "G"))),
                 key=lambda x: -x["ovr"])[:slots]
    if not nhl:
        return None
    projected = r["ovr"] * 0.5 + r["pot"] * 0.5
    locked = [x for x in nhl if x["ovr"] >= projected and (x["contract"]["years_left"] or 0) >= 2]
    weakest = nhl[-1]["ovr"]
    if r["ovr"] < weakest - 3:
        return round(len(locked) - slots + (r["pot"] - weakest) * 0.3, 1)
    return round((r["ovr"] - weakest) + len(locked) * 0.5, 1)


def waiver_risk(view: Any, pid: str) -> Num:
    """Close to losing waiver exemption while still outside the NHL lineup."""
    r = view.row(pid)
    if not r or r["team_id"] != view.user_tid or r["level"] == "NHL":
        return None
    try:
        from services.contract_economy import is_waiver_exempt

        exempt = is_waiver_exempt(view.players[pid])
    except Exception:
        exempt = True
    gap = 0.0
    nhl = sorted((x["ovr"] for x in view.team_rows(view.user_tid) if (x["pos"] in ("D", "LD", "RD")) == (r["pos"] in ("D", "LD", "RD"))), reverse=True)
    if nhl:
        gap = max(0.0, nhl[min(len(nhl) - 1, 11 if r["pos"] not in ("D", "LD", "RD") else 5)] - r["ovr"])
    if not exempt:
        return round(10 + gap, 1)
    return round(max(0, r["age"] - 20) * 2.5 + gap * 0.5, 1)


def overperformance(view: Any, pid: str) -> Num:
    """Career-year signal: scoring well above his rating's line and above his shot quality."""
    r = view.row(pid)
    fm = form(view, pid)
    lk = luck(view, pid)
    if not r or fm is None:
        return None
    return round(fm * 2 + (lk or 0) * 0.5 + (2 if r["age"] >= 27 else 0), 2)


def overrated(view: Any, pid: str) -> Num:
    r = view.row(pid)
    if not r or r["pos"] == "G":
        return None
    rel = onice_rel(view, pid)
    mv = market(view, pid)
    cap = r["contract"]["cap_hit_m"] or 0
    if rel is None:
        return None
    return round((r["ovr"] - 80) * 0.3 - rel * 0.6 + (cap - (mv or cap)) * 0.8 - (form(view, pid) or 0) * 2, 1)


def underrated(view: Any, pid: str) -> Num:
    r = view.row(pid)
    if not r or r["pos"] == "G" or r["ovr"] > 88:
        return None
    rel = onice_rel(view, pid)
    mv = market(view, pid)
    cap = r["contract"]["cap_hit_m"] or 0
    if rel is None or not cap:
        return None
    return round(rel * 0.6 + ((mv or cap) - cap) * 0.6 + (form(view, pid) or 0) * 2 + (88 - r["ovr"]) * 0.1, 1)


def face_of_franchise(view: Any, pid: str) -> Num:
    r = view.row(pid)
    if not r:
        return None
    lead = metric_value(view, pid, "hd:leadership") or 60
    return round(r["ovr"] * 0.6 + max(0, 28 - r["age"]) * 0.8 + (r["contract"]["years_left"] or 0) * 1.2 + (lead - 60) * 0.15 + _f(r.get("pts")) * 0.2, 1)


def is_rookie(view: Any, pid: str) -> bool:
    """Calder rule: under 25 prior NHL games in no single season, and 26 or younger."""
    r = view.row(pid)
    if not r or r["age"] > 26:
        return False
    cs = getattr(view.players.get(pid), "career_stats", None)
    seasons = (cs.get("seasons") or []) if isinstance(cs, dict) else []
    prior = 0
    for s in seasons:
        if not isinstance(s, dict):
            continue
        lg = str(s.get("league") or s.get("league_abbrev") or "NHL").upper()
        yr = str(s.get("season") or s.get("year") or "")[:4]
        if "NHL" in lg and yr.isdigit() and int(yr) < int(view.season):
            prior += int(_f(s.get("gp") or s.get("games_played")))
    return prior < 25


def award_chance(view: Any, pid: str) -> Num:
    """Calder case for an NHL rookie: rating plus production."""
    r = view.row(pid)
    if not r or r["level"] != "NHL" or not is_rookie(view, pid):
        return None
    return round(r["ovr"] * 0.5 + _f(r.get("pts")) * 1.5, 1)


# Registry: name -> (label, function, higher_is_better, unit hint)
PLAYER_MEASURES: Dict[str, Tuple[str, Callable[[Any, str], Num], bool, str]] = {
    "surplus3": ("Surplus value, next 3 seasons", surplus_years, True, "M"),
    "contract_risk_7": ("Overpay risk on a 7-year deal", contract_risk_7, True, "M"),
    "unmovable": ("Unmovable-contract risk", unmovable, True, "M"),
    "aging_risk": ("Aging risk", aging_risk, True, ""),
    "p60": ("Points per 60", p60, True, ""),
    "ev_p60": ("Even-strength points per 60", ev_p60, True, ""),
    "pp_p60": ("Power-play points per 60", pp_p60, True, ""),
    "pp_share": ("Share of points on the power play", pp_share, True, "%"),
    "prod_vs_toi": ("Production beyond his minutes", prod_vs_toi, True, ""),
    "onice_rel": ("On-ice goal share vs team", onice_rel, True, "pp"),
    "xg_share": ("On-ice expected-goal share", xg_share, True, "%"),
    "ipp": ("Points on on-ice goals", ipp, True, "%"),
    "carried": ("Carried by linemates", carried, True, ""),
    "elevates": ("Elevates linemates", elevates, True, ""),
    "hurts_linemates": ("Hurts linemates despite points", hurts_linemates, True, ""),
    "luck": ("Luck (goals above expected)", luck, True, ""),
    "sh_pct": ("Shooting %", sh_pct, True, "%"),
    "form": ("Production vs his rating", form, True, ""),
    "stagnation": ("Stagnation risk", stagnation, True, ""),
    "gap_to_potential": ("Gap to potential", gap_to_potential, True, ""),
    "lacks_tools": ("Potential above his tools", lacks_tools, True, ""),
    "top_line_chance": ("Top-line projection", top_line_chance, True, ""),
    "surprise_elite": ("Breakout-to-elite signal", surprise_elite, True, ""),
    "skating_weighted": ("Overall with skating double-weighted", skating_weighted, True, ""),
    "trait_rarity": ("Trait rarity", trait_rarity, True, ""),
    "chemistry_total": ("Chemistry with teammates", chemistry_total, True, ""),
    "partner_lift": ("Lift from his partner", partner_lift, True, ""),
    "draft_vs_slot": ("Rating vs draft slot", draft_vs_slot, True, ""),
    "blocked": ("Blocked by the NHL roster", blocked, True, ""),
    "waiver_risk": ("Waiver risk", waiver_risk, True, ""),
    "overperformance": ("Career-year signal", overperformance, True, ""),
    "overrated": ("Overrated score", overrated, True, ""),
    "underrated": ("Underrated score", underrated, True, ""),
    "face_of_franchise": ("Face-of-franchise score", face_of_franchise, True, ""),
    "award_chance": ("Award case (rookie)", award_chance, True, ""),
    "market": ("Market value", market, True, "M"),
    "toi_pg": ("Ice time per game", toi_pg, True, ""),
}


def value_of(view: Any, pid: str, name: str) -> Num:
    c = _cache(view).setdefault("vals", {})
    key = (pid, name)
    if key not in c:
        try:
            c[key] = PLAYER_MEASURES[name][1](view, pid)
        except Exception:
            c[key] = None
    return c[key]


# =========================================================================== teams
def _group(view: Any, tid: str, grp: str, levels: Tuple[str, ...] = ("NHL",)) -> List[Dict[str, Any]]:
    rows = view.team_rows(tid, levels)
    if grp == "G":
        out = [r for r in rows if r["pos"] == "G"]
    elif grp == "D":
        out = [r for r in rows if r["pos"] in ("D", "LD", "RD")]
    elif grp == "LD":
        out = [r for r in rows if r["pos"] == "LD" or (r["pos"] == "D" and r["shoots"] == "L")]
    elif grp == "RD":
        out = [r for r in rows if r["pos"] == "RD" or (r["pos"] == "D" and r["shoots"] == "R")]
    elif grp == "C":
        out = [r for r in rows if r["pos"] == "C"]
    elif grp == "W":
        out = [r for r in rows if r["pos"] in ("LW", "RW", "W")]
    else:
        out = [r for r in rows if r["pos"] not in ("G", "D", "LD", "RD")]
    return sorted(out, key=lambda r: -r["ovr"])


def team_strength(view: Any, tid: str) -> Num:
    f = [r["ovr"] for r in _group(view, tid, "F")[:12]]
    d = [r["ovr"] for r in _group(view, tid, "D")[:6]]
    g = [r["ovr"] for r in _group(view, tid, "G")[:1]]
    if not f or not d or not g:
        return None
    return round(statistics.mean(f) * 0.5 + statistics.mean(d) * 0.3 + g[0] * 0.2, 2)


def team_xg_share(view: Any, tid: str) -> Num:
    """Ice-time-weighted on-ice expected-goal share of the team's skaters."""
    num = den = 0.0
    for r in view.team_rows(tid):
        if r["pos"] == "G":
            continue
        st = _st(view, r["id"])
        xf, xa, toi = _f(st.get("xgf")), _f(st.get("xga")), _f(st.get("toi_sec"))
        if xf + xa > 0 and toi > 0:
            num += xf / (xf + xa) * toi
            den += toi
    return round(num / den * 100, 1) if den > 3600 else None


def team_status(view: Any, tid: str) -> str:
    st = sorted(view.standings(), key=lambda r: int(r.get("league_rank") or r.get("rank") or 99))
    rank = next((i + 1 for i, r in enumerate(st) if str(r.get("team_id")) == str(tid)), 16)
    strengths = sorted(((team_strength(view, t) or 0, t) for t in view.teams), reverse=True)
    srank = next((i + 1 for i, (_s, t) in enumerate(strengths) if t == tid), 16)
    if rank <= 10 or srank <= 8:
        return "contender"
    if rank >= 23 and srank >= 20:
        return "rebuilding"
    return "middle"


def pipeline(view: Any, tid: str) -> Num:
    pros = sorted((r["pot"] for r in view.team_rows(tid, ("Prospect", "AHL", "ECHL")) if r["age"] <= 23), reverse=True)[:5]
    return round(statistics.mean(pros), 1) if pros else None


def committed_next(view: Any, tid: str) -> float:
    return round(sum(r["contract"]["cap_hit_m"] or 0 for r in view.team_rows(tid, ("NHL", "AHL")) if (r["contract"]["years_left"] or 0) >= 2), 2)


def avg_age(view: Any, tid: str) -> Num:
    rows = sorted(view.team_rows(tid), key=lambda r: -r["ovr"])[:20]
    return round(statistics.mean(r["age"] for r in rows), 1) if rows else None


def team_measure(view: Any, tid: str, name: str) -> Num:
    if name == "center_depth":
        c = [r["ovr"] for r in _group(view, tid, "C")[:3]]
        return round(statistics.mean(c), 1) if c else None
    if name == "goalie_injury_risk":
        g = _group(view, tid, "G", ("NHL", "AHL"))
        if len(g) < 2:
            return 30.0
        return round((g[0]["ovr"] - g[1]["ovr"]) + max(0, 80 - g[1]["ovr"]), 1)
    if name == "ld_surplus":
        ld = [r for r in _group(view, tid, "LD", ("NHL", "AHL")) if r["ovr"] >= 76 and r["age"] <= 28]
        return float(max(0, len(ld) - 3))
    if name == "wing_need":
        w = [r["ovr"] for r in _group(view, tid, "W")[:6]]
        return round(90 - statistics.mean(w), 1) if w else None
    if name == "strength":
        return team_strength(view, tid)
    if name == "xg_share":
        return team_xg_share(view, tid)
    if name == "pipeline":
        return pipeline(view, tid)
    if name == "avg_age":
        return avg_age(view, tid)
    if name == "future":
        young = [r["ovr"] for r in view.team_rows(tid) if r["age"] <= 25]
        p = pipeline(view, tid) or 70
        space = 95 - committed_next(view, tid)
        return round((statistics.mean(sorted(young, reverse=True)[:8]) if young else 70) * 0.5 + p * 0.4 + space * 0.1, 1)
    if name == "luck_gap":
        st = next((r for r in view.standings() if str(r.get("team_id")) == str(tid)), None)
        xs = team_xg_share(view, tid)
        if not st or xs is None or not st.get("gp"):
            return None
        pts_pct = _f(st.get("pts")) / (2 * _f(st.get("gp"))) * 100
        return round(pts_pct - xs, 1)
    if name == "cap_space":
        t = view.teams.get(tid)
        for a in ("cap_space_m", "cap_space"):
            v = getattr(t, a, None)
            try:
                v = float(v)
                return v / 1e6 if v > 1000 else v
            except (TypeError, ValueError):
                continue
        return None
    return None


TEAM_MEASURES: Dict[str, Tuple[str, bool]] = {
    "center_depth": ("Top-three centre average", True), "goalie_injury_risk": ("Drop-off to the backup goalie", True),
    "ld_surplus": ("Spare NHL-ready left D", True), "wing_need": ("Wing need", True), "strength": ("Roster strength", True),
    "xg_share": ("Expected-goal share", True), "pipeline": ("Prospect pipeline", True), "avg_age": ("Average age of top 20", True),
    "future": ("Ten-year outlook", True), "luck_gap": ("Points % above expected-goal share", True), "cap_space": ("Cap space", True),
}
