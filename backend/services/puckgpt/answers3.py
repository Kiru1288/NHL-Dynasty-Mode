"""PuckGPT, third layer: analytical questions.

One engine ranks players by any derived measure with every condition in the
question applied together (handedness, age, position, organisation level,
line tier, contract status). A second ranks teams. Reasoning answers combine
several measures and show the numbers behind the call. Questions the save
cannot answer get an exact reason: never recorded, or needs finished seasons.

``match`` is checked before the other layers, on the raw question, because
these questions are long and specific and their wording carries the meaning.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from .answers import HANDLERS as BASE, POS_NAMES, SINGULAR, _answer, _card, _cap_space, _money, _player_table, _pos_ok, _resolve_players, _standing, _team_label
from .data import LiveView, archive
from .derived import (
    PLAYER_MEASURES, TEAM_MEASURES, _group, committed_next, form, line_tier, market, pipeline, season_change,
    team_measure, team_status, team_strength, toi_pg, value_of,
)
from .metrics import metric_value

Answer = Dict[str, Any]


# =========================================================================== conditions
def parse_filters(raw: str, slots: Dict[str, Any], view: LiveView) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if re.search(r"\bleft[\s-]?(shot|shooting|handed)|\blefty\b|\bshoots left\b", raw):
        f["shoots"] = "L"
    elif re.search(r"\bright[\s-]?(shot|shooting|handed)|\brighty\b|\bshoots right\b", raw):
        f["shoots"] = "R"
    if re.search(r"\b(organization|organisation|org|system|pipeline)\b", raw):
        f["scope"] = "org"
    elif re.search(r"\bmy (roster|team|lineup|club)\b|\bon my\b|\bour (roster|team)\b", raw):
        f["scope"] = "team"
    if re.search(r"\bahl\b", raw):
        f["level"] = "AHL"
    if re.search(r"\bprospects?\b", raw) and "level" not in f:
        f["level"] = "prospect"
    pos = slots.get("position")
    if re.search(r"\bdefense ?m[ae]n\b|\bdefence ?m[ae]n\b|\bd[ -]?m[ae]n\b|\bdefenders?\b", raw):
        pos = pos or "D"
    if re.search(r"\bwingers?\b", raw):
        pos = "W"
    if re.search(r"\bcent(er|re)s?\b", raw):
        pos = "C"
    if re.search(r"\bforwards?\b", raw):
        pos = pos or "F"
    if re.search(r"\bgoalies?\b|\bgoaltenders?\b", raw):
        pos = "G"
    if pos:
        f["position"] = pos
    m = re.search(r"\bunder (?:the )?(?:age (?:of )?)?(\d{2})\b", raw)
    if m:
        f["age_max"] = int(m.group(1)) - 1
    m = re.search(r"\b(\d{2})[\s-]?year[\s-]?olds?\b", raw)
    if m:
        f["age_min"] = f["age_max"] = int(m.group(1))
    m = re.search(r"\bover (?:the )?(?:age (?:of )?)?(\d{2})\b(?!\s*overall)", raw)
    if m and int(m.group(1)) < 45:
        f["age_min"] = int(m.group(1)) + 1
    if re.search(r"\brfas?\b", raw):
        f["status"] = "RFA"
    elif re.search(r"\bufas?\b", raw):
        f["status"] = "UFA"
    if re.search(r"\bexpiring\b|\bcontract year\b|\bbefore free agency\b", raw):
        f["expiring"] = True
    m = re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[\s-](line|pair)", raw)
    if m:
        f["tier_word"] = m.group(1)
    return f


def _in_scope(view: LiveView, pid: str, f: Dict[str, Any]) -> bool:
    tid, lv = view.where.get(pid, ("", ""))
    scope, level = f.get("scope"), f.get("level")
    if level == "AHL" and lv != "AHL":
        return False
    if level == "prospect" and (lv not in ("Prospect", "AHL", "ECHL") or view.row(pid).get("age", 99) > 23):
        return False
    if scope == "org" and tid != view.user_tid:
        return False
    if scope == "team" and (tid != view.user_tid or lv != "NHL"):
        return False
    if not scope and not level and lv not in ("NHL",) and not f.get("any_level"):
        return False
    if level == "AHL" and scope is None and "my" in f.get("_raw", ""):
        return tid == view.user_tid
    r = view.row(pid)
    if f.get("position") and not _pos_ok(r, f["position"]):
        return False
    if f.get("shoots") and r.get("shoots") != f["shoots"]:
        return False
    if f.get("age_max") and not (0 < r["age"] <= f["age_max"]):
        return False
    if f.get("age_min") and r["age"] < f["age_min"]:
        return False
    if f.get("status"):
        c = r["contract"]
        if c["status"] != f["status"] or (c["years_left"] or 0) > 1:
            return False
    if f.get("expiring") and (r["contract"]["years_left"] or 0) > 1:
        return False
    return True


def _describe(f: Dict[str, Any]) -> str:
    bits = []
    if f.get("shoots"):
        bits.append("left-shot" if f["shoots"] == "L" else "right-shot")
    if f.get("position"):
        bits.append(SINGULAR.get(POS_NAMES.get(f["position"], ""), f["position"]))
    if f.get("age_max") and f.get("age_min") == f.get("age_max"):
        bits.append(f"aged {f['age_max']}")
    elif f.get("age_max"):
        bits.append(f"under {f['age_max'] + 1}")
    if f.get("status"):
        bits.append(f"pending {f['status']}")
    if f.get("expiring"):
        bits.append("on an expiring deal")
    where = {"org": "in your organization", "team": "on your roster"}.get(f.get("scope") or "", "")
    if f.get("level") == "AHL":
        where = "in the AHL" if not where else where + " (AHL)"
    elif f.get("level") == "prospect":
        where = "among prospects" if not where else where + " (prospects)"
    if not where:
        where = "in the league"
    return (" ".join(bits) + " " + where).strip()


def player_rank(view: LiveView, measure: str, f: Dict[str, Any], *, low: bool = False, limit: int = 10,
                note: str = "", extra_cols: Tuple[str, ...] = ()) -> Answer:
    label, fn, higher, unit = PLAYER_MEASURES[measure]
    rows = []
    for pid in view.where:
        if not _in_scope(view, pid, f):
            continue
        v = value_of(view, pid, measure)
        if v is None:
            continue
        rows.append((v, view.row(pid)))
    if not rows:
        return _answer(f"No player {_describe(f)} has enough data for {label.lower()} yet.",
                       analysis=["Season measures need about five games played; ratings and contracts are always available."])
    rows.sort(key=lambda vr: vr[0], reverse=(higher != low))
    shown = rows[:limit]
    v0, top = shown[0]
    if measure == "blocked" and v0 <= 0:
        note = (note + " " if note else "") + "Nobody is truly blocked right now: every prospect is still below the lineup bar. Listed: the closest."
    t = _player_table([r for _v, r in shown], ("gp", "pts", "cap") + extra_cols)
    t["cols"].append(label)
    for line, (v, _r) in zip(t["rows"], shown):
        line.append(_fmt(v, unit))
    head = f"{top['name']} ({top['pos']}, {top['team'] or top['level']}, {top['age']}): {label.lower()} {_fmt(v0, unit)}."
    if measure == "blocked" and v0 <= 0:
        head = f"Nobody is truly blocked yet. Closest: {top['name']} ({top['pos']}, {top['age']}, {top['ovr']:.0f}/{top['pot']:.0f})."
    analysis = [f"Ranked {len(rows)} players {_describe(f)}."] + ([note] if note else [])
    if len(rows) >= 5:
        analysis.append(f"League median among them: {_fmt(statistics.median(v for v, _ in rows), unit)}.")
    return _answer(head, analysis=analysis, table=t, cards=[_card(r) for _v, r in shown[:6]],
                   list_ids=[r["id"] for _v, r in shown], focus_ids=[top["id"]])


def _fmt(v: Any, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "M":
        return f"{v:+.2f}M" if v < 0 or unit == "M" and v < 50 else f"{v:.2f}M"
    if unit == "%":
        return f"{v * 100:.0f}%" if abs(v) <= 1.5 else f"{v:.1f}%"
    if unit == "pp":
        return f"{v:+.1f} pts"
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def team_rank(view: LiveView, measure: str, *, status: Optional[str] = None, low: bool = False, note: str = "",
              limit: int = 8) -> Answer:
    label, higher = TEAM_MEASURES[measure]
    rows = []
    for tid in view.teams:
        if status and team_status(view, tid) != status:
            continue
        v = team_measure(view, tid, measure)
        if v is not None:
            rows.append((v, tid))
    if not rows:
        return _answer(f"No {status or ''} team has data for {label.lower()} yet.".replace("  ", " "))
    rows.sort(key=lambda vt: vt[0], reverse=(higher != low))
    v0, t0 = rows[0]
    table = {"cols": ["Team", label, "Status", "Strength", "Cap space"],
             "rows": [[_team_label(view, t), round(v, 1), team_status(view, t), team_strength(view, t),
                       _money(team_measure(view, t, "cap_space") or 0)] for v, t in rows[:limit]]}
    return _answer(f"The {_team_label(view, t0)}: {label.lower()} {round(v0, 1)}.", table=table,
                   analysis=[f"{len(rows)} {status + ' ' if status else ''}teams ranked."] + ([note] if note else []))


# =========================================================================== draft board
def _board(view: LiveView) -> List[Dict[str, Any]]:
    c = getattr(view, "_board", None)
    if c is None:
        try:
            from services.franchise_sim import get_cached_draft_class_rankings

            c = list((get_cached_draft_class_rankings(view.session, view.session.sim) or {}).get("entries") or [])
        except Exception:
            c = []
        view._board = c
    return c


def draft_board(view: LiveView, kind: str) -> Answer:
    e = [x for x in _board(view) if x.get("rank")]
    if not e:
        return _answer("The draft board isn't available right now.")
    def num(x: Dict[str, Any], k: str) -> float:
        try:
            return float(x.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0
    rank = lambda x: int(x.get("rank") or 999)
    pick, why, cols = None, "", ("potential_score", "floor_score", "ceiling_score")
    if kind == "undervalued":
        use = "analytics_score" if sum(1 for x in e if num(x, "analytics_score")) > len(e) // 2 else "potential_score"
        by_metric = sorted(e, key=lambda x: -(num(x, use) + num(x, "production_score") * 0.3))
        arank = {x["key"]: i + 1 for i, x in enumerate(by_metric)}
        pool = sorted((x for x in e if 10 < rank(x) <= 200), key=lambda x: arank[x["key"]] - rank(x))
        label = "analytics" if use == "analytics_score" else "talent (potential plus production)"
        pick, why = pool[:10], f"Ranked by how far his {label} rank sits above the scouting rank; top ten excluded."
    elif kind == "second_round_talent":
        top_pot = sorted(e, key=lambda x: -num(x, "potential_score"))[:32]
        pick = sorted((x for x in top_pot if 33 <= rank(x) <= 64), key=lambda x: -num(x, "potential_score"))
        pick = pick or sorted((x for x in e if 33 <= rank(x) <= 64), key=lambda x: -num(x, "potential_score"))[:10]
        why = "Projected 33rd to 64th, with potential that ranks among the top 32 in the class."
    elif kind == "steal":
        pick = sorted((x for x in e if rank(x) > 32), key=lambda x: -(num(x, "ceiling_score") - rank(x) * 0.08 + (6 if x.get("is_gem") else 0)))[:10]
        why = "Ranked outside the first round, sorted by ceiling with a bonus for scout-flagged gems."
    elif kind == "carried":
        pick = sorted(e[:150], key=lambda x: -(num(x, "quality_of_teammates") * 0.6 + num(x, "production_score") * 0.4 - num(x, "overall") * 0.5))[:10]
        why = "Strong production next to high-quality teammates, relative to his own ratings."
    elif kind == "bust_top10":
        pick = sorted((x for x in e if rank(x) <= 10), key=lambda x: (not x.get("is_bust_risk"), num(x, "floor_score") - num(x, "outcome_volatility")))
        why = "The top ten by scouting rank, sorted by bust flag, then lowest floor and widest outcome range."
    elif kind == "skating2x":
        pick = sorted(e[:300], key=lambda x: -(num(x, "potential_score") + num(x, "skating")))[:10]
        why = "Board potential plus skating counted a second time."
    elif kind == "low_floor_r1":
        pick = sorted((x for x in e if rank(x) <= 32), key=lambda x: num(x, "floor_score"))[:10]
        why = "First-round projections (top 32) with the lowest floor scores."
    elif kind == "ceiling_outside_lottery":
        pick = sorted((x for x in e if rank(x) > 16), key=lambda x: -num(x, "ceiling_score"))[:10]
        why = "Outside the top 16 (the lottery picks), highest ceiling score."
    elif kind == "rare_traits":
        names = ("skating", "shooting", "passing", "defense", "physical", "hockey_iq")
        stats = {n: (statistics.mean(num(x, n) for x in e), statistics.pstdev([num(x, n) for x in e]) or 1) for n in names}
        def rarity(x: Dict[str, Any]) -> float:
            zs = sorted(((num(x, n) - stats[n][0]) / stats[n][1] for n in names), reverse=True)
            return zs[0] + zs[1]
        skaters = [x for x in e if str(x.get("position") or "") != "G" and rank(x) <= 150]
        pick = sorted(skaters, key=lambda x: -rarity(x))[:10]
        why = "His two best skills, measured against the whole class, are the least common pairing."
    elif kind == "ready":
        pick = sorted(e[:200], key=lambda x: -(num(x, "overall") or num(x, "current_ovr_estimate")))[:10]
        why = "Draft-eligible players sorted by current overall estimate."
    else:
        pick = e[:10]
    if not pick:
        return _answer("No prospect on the board matches that.")
    p = pick[0]
    table = {"cols": ["Board rank", "Prospect", "Pos", "Age", "League", "POT score", "Floor", "Ceiling", "Bust risk"],
             "rows": [[x.get("rank"), x.get("name"), x.get("position"), x.get("age"), x.get("league_display") or x.get("league"),
                       round(num(x, "potential_score"), 1), round(num(x, "floor_score"), 1), round(num(x, "ceiling_score"), 1),
                       "yes" if x.get("is_bust_risk") else ""] for x in pick[:10]]}
    return _answer(f"{p.get('name')} ({p.get('position')}, {p.get('league_display') or p.get('league')}, board #{p.get('rank')}).",
                   analysis=[why, "This is the upcoming draft class; scouting confidence rises as the season goes on."], table=table)


# =========================================================================== reasoning
def _rank_of(view: LiveView, tid: str, measure: str = "strength") -> Tuple[int, int]:
    vals = sorted(((team_measure(view, t, measure) or 0, t) for t in view.teams), reverse=True)
    return next((i + 1 for i, (_v, t) in enumerate(vals) if t == tid), len(vals)), len(vals)


def _strength_with(view: LiveView, tid: str, add: Optional[Dict[str, Any]] = None, remove_ids: Tuple[str, ...] = ()) -> float:
    rows = [r for r in view.team_rows(tid) if r["id"] not in remove_ids]
    if add:
        rows.append(add)
    f = sorted((r["ovr"] for r in rows if r["pos"] not in ("G", "D", "LD", "RD")), reverse=True)[:12]
    d = sorted((r["ovr"] for r in rows if r["pos"] in ("D", "LD", "RD")), reverse=True)[:6]
    g = sorted((r["ovr"] for r in rows if r["pos"] == "G"), reverse=True)[:1]
    if not f or not d or not g:
        return 0.0
    return statistics.mean(f) * 0.5 + statistics.mean(d) * 0.3 + g[0] * 0.2


def _cup_odds(view: LiveView, strength: float) -> float:
    """Rough title odds from where this strength would rank (top team ~18%, #8 ~5%, outside 16 ~0)."""
    vals = sorted((team_strength(view, t) or 0 for t in view.teams), reverse=True)
    rank = sum(1 for v in vals if v > strength) + 1
    return round(max(0.0, 18.0 * (0.80 ** (rank - 1))) if rank <= 16 else 0.3, 1)


def _sellers(view: LiveView) -> List[str]:
    return [t for t in view.teams if team_status(view, t) in ("rebuilding", "middle") and t != view.user_tid]


def _best_target(view: LiveView, grp: str, cap_room: Optional[float] = None) -> Optional[Dict[str, Any]]:
    sellers = set(_sellers(view))
    cands = [view.row(p) for p, (t, lv) in view.where.items() if (lv == "FA" or (lv == "NHL" and t in sellers))]
    cands = [r for r in cands if _pos_ok(r, grp) and r["age"] <= 32]
    if cap_room is not None:
        cands = [r for r in cands if (r["contract"]["cap_hit_m"] or 0) <= cap_room + 4.0]
    return max(cands, key=lambda r: r["ovr"], default=None)


def _weakest_group(view: LiveView, tid: str) -> Tuple[str, float]:
    gaps = {}
    for grp, n in (("C", 3), ("W", 6), ("D", 4), ("G", 1)):
        mine = [r["ovr"] for r in _group(view, tid, grp)[:n]]
        league = sorted((statistics.mean([r["ovr"] for r in _group(view, t, grp)[:n]] or [0]) for t in view.teams), reverse=True)
        if mine and league:
            gaps[grp] = statistics.mean(league[:8]) - statistics.mean(mine)
    g = max(gaps, key=gaps.get) if gaps else "D"
    return g, gaps.get(g, 0.0)


def reason(view: LiveView, kind: str, raw: str, slots: Dict[str, Any], conv: Dict[str, Any]) -> Answer:
    tid = view.user_tid
    s0 = team_strength(view, tid) or 0
    srank, n = _rank_of(view, tid)
    st = _standing(view, tid) or {}
    space = _cap_space(view, tid) or 0.0
    status = team_status(view, tid)
    lines: List[str] = []

    if kind == "prospect_for_d":
        pros = sorted(view.team_rows(tid, ("Prospect", "AHL", "ECHL")), key=lambda r: -r["pot"])
        target = _best_target(view, "D", space)
        if not pros or not target:
            return _answer("I can't find both a top prospect and a top-pair defenseman to swap.")
        p = pros[0]
        weakest_d = sorted(_group(view, tid, "D")[:4], key=lambda r: r["ovr"])[0]
        s1 = _strength_with(view, tid, add=target, remove_ids=(weakest_d["id"],))
        o0, o1 = _cup_odds(view, s0), _cup_odds(view, s1)
        future_p = round(p["ovr"] * 0.4 + p["pot"] * 0.6)
        lines = [f"Now: roster strength {s0:.1f} ({srank} of {n}); with {target['name']} ({target['ovr']:.0f}) replacing {weakest_d['name']} it's {s1:.1f}.",
                 f"Rough title odds this year: {o0}% → {o1}%.",
                 f"{p['name']} ({p['age']}, {p['ovr']:.0f}/{p['pot']:.0f}) projects to about {future_p} in three years; {target['name']} will be {target['age'] + 3} then.",
                 f"Cap: {target['name']} costs {_money(target['contract']['cap_hit_m'])} for {target['contract']['years_left']} more seasons; you have {_money(space)} now."]
        verdict = ("Worth it for the next two or three seasons if you're contending; it costs you later."
                   if o1 - o0 >= 1.5 and status == "contender" else
                   "It doesn't move your odds enough to give up the prospect.")
        return _answer(verdict, analysis=lines + ["Five-year odds aren't something I can simulate; these are this-season odds from roster strength."],
                       rows=[target, p], extra=("cap", "status"))
    if kind == "now_vs_flex":
        nxt = committed_next(view, tid)
        lines = [f"Standing: {st.get('league_rank') or '?'} of {n} ({status}).", f"Cap space now {_money(space)}; {_money(nxt)} already committed next season.",
                 f"Roster strength {srank} of {n}."]
        verdict = "Buy now: you're in contention and help this year matters more." if status == "contender" else \
            "Keep the flexibility: you're not a contender this year, and next summer's space buys more." if status == "rebuilding" else \
            "Make only moves that also help next year: young players on term, no rentals."
        return _answer(verdict, analysis=lines)
    if kind == "roster_vs_luck":
        from .derived import team_xg_share
        xs = team_xg_share(view, tid)
        pts_pct = (st.get("pts", 0) / max(1, 2 * st.get("gp", 1))) * 100 if st.get("gp") else None
        lines = [f"Roster strength {srank} of {n}; standings {st.get('league_rank') or '?'} of {n}."]
        if xs is not None and pts_pct is not None:
            lines.append(f"Points % {pts_pct:.0f} vs expected-goal share {xs:.0f}%.")
        luck_gap = (pts_pct or 50) - (xs or 50)
        roster = srank > n // 2
        verdict = ("Both: the roster is below average and results are under the underlying numbers." if roster and luck_gap < -5 else
                   "Roster construction: the talent ranks low and the record matches it." if roster else
                   "Mostly bad luck: the roster ranks well and the expected goals are better than the results." if luck_gap < -5 else
                   "Neither, really: talent and results line up; it's a middle-of-the-pack team.")
        return _answer(verdict, analysis=lines)
    if kind == "single_move":
        grp, gap = _weakest_group(view, tid)
        target = _best_target(view, grp, space)
        if not target:
            return _answer("No available player clearly upgrades your weakest spot.")
        weakest = sorted(_group(view, tid, grp)[: {"C": 3, "W": 6, "D": 4, "G": 1}[grp]], key=lambda r: r["ovr"])[0]
        s1 = _strength_with(view, tid, add=target, remove_ids=(weakest["id"],))
        lines = [f"Your weakest group vs the league's top eight: {POS_NAMES.get(grp, grp)}, {gap:.1f} overall behind.",
                 f"{target['name']} ({target['team'] or 'free agent'}, {target['ovr']:.0f}) for {weakest['name']} ({weakest['ovr']:.0f}).",
                 f"Roster strength {s0:.1f} → {s1:.1f}; rough title odds {_cup_odds(view, s0)}% → {_cup_odds(view, s1)}%.",
                 f"He costs {_money(target['contract']['cap_hit_m'])}; you have {_money(space)} in space."]
        return _answer(f"Get {target['name']}.", analysis=lines, rows=[target, weakest], extra=("cap", "status"))
    if kind == "rebuild_or_compete":
        core = sorted(view.team_rows(tid), key=lambda r: -r["ovr"])[:6]
        core_age = statistics.mean(r["age"] for r in core) if core else 0
        pipe, prank = pipeline(view, tid), _rank_of(view, tid, "pipeline")[0]
        lines = [f"Standing {st.get('league_rank') or '?'} of {n}, roster strength {srank} of {n}.",
                 f"Your top six players average {core_age:.1f} years old.", f"Prospect pipeline ranks {prank} of {n} (top-five potential {pipe})."]
        verdict = ("Stay competitive: you're a top-half roster and the core is in its prime." if srank <= n // 2 and core_age <= 30 else
                   "Rebuild now: the core is aging and the roster isn't a contender." if core_age > 29 and srank > n // 2 else
                   "Retool, don't rebuild: sell aging depth, keep the young core, and use the pipeline.")
        return _answer(verdict, analysis=lines)
    if kind == "superstar_away":
        top5 = sorted((team_strength(view, t) or 0 for t in view.teams), reverse=True)[4]
        star = {"id": "_star", "pos": "C", "ovr": 96.0}
        weakest_f = sorted(_group(view, tid, "F")[:6], key=lambda r: r["ovr"])[0]
        s1 = _strength_with(view, tid, add=star, remove_ids=(weakest_f["id"],))
        lines = [f"Your strength {s0:.1f}; the #5 team is {top5:.1f}.", f"Adding a 96-overall star in place of {weakest_f['name']} makes it {s1:.1f}."]
        verdict = "One superstar away: a single star gets you into the top five." if s1 >= top5 else \
            f"Several pieces away: even a superstar leaves you {top5 - s1:.1f} short of the top five."
        return _answer(verdict, analysis=lines)
    if kind == "pipeline":
        pipe, prank = pipeline(view, tid), _rank_of(view, tid, "pipeline")[0]
        expiring = [r for r in view.team_rows(tid) if (r["contract"]["years_left"] or 0) <= 1 and r["ovr"] >= 85]
        pros = sorted(view.team_rows(tid, ("Prospect", "AHL", "ECHL")), key=lambda r: -r["pot"])[:5]
        lines = [f"Pipeline ranks {prank} of {n} (top-five potential average {pipe}).",
                 f"Core players on expiring deals: {', '.join(r['name'] for r in expiring) or 'none'}."]
        verdict = "Yes, strong enough to retool without a full rebuild." if prank <= 10 else \
            "Borderline: it can fill depth, not replace the core." if prank <= 20 else "No: it's among the league's thinnest."
        return _answer(verdict, analysis=lines, rows=pros, extra=("gp", "pts"))
    if kind == "vet_blocking":
        young = [r for r in view.team_rows(tid, ("NHL", "AHL", "Prospect")) if r["age"] <= 23 and r["pot"] >= 82]
        pairs = []
        for y in young:
            grp = "D" if y["pos"] in ("D", "LD", "RD") else ("G" if y["pos"] == "G" else "F")
            for v in _group(view, tid, grp):
                if v["age"] >= 30 and v["ovr"] - y["ovr"] <= 4 and (toi_pg(view, v["id"]) or 0) >= (toi_pg(view, y["id"]) or 0):
                    pairs.append((v["ovr"] - y["ovr"], v, y))
        if not pairs:
            return _answer("No veteran is clearly in a young player's way right now.")
        pairs.sort(key=lambda x: x[0])
        _g, v, y = pairs[0]
        return _answer(f"{v['name']} ({v['age']}, {v['ovr']:.0f}) is taking minutes {y['name']} ({y['age']}, {y['ovr']:.0f}/{y['pot']:.0f}) could have.",
                       analysis=["Within four overall points and playing as much or more, at the same position.",
                                 "In-season growth follows ice time and production, so those minutes matter for the kid."],
                       rows=[v, y], extra=("gp", "pts", "cap"))
    if kind == "promote":
        rows = _resolve_players(view, slots)
        if not rows:
            return _answer("Promote who? Name the prospect.")
        p = rows[0]
        grp = "D" if p["pos"] in ("D", "LD", "RD") else ("G" if p["pos"] == "G" else "F")
        nhl = _group(view, tid, grp)
        slot = sum(1 for x in nhl if x["ovr"] > p["ovr"]) + 1
        top = {"D": 4, "F": 9, "G": 1}[grp]
        lines = [f"{p['name']} is {p['ovr']:.0f}/{p['pot']:.0f} at {p['age']}; he'd be #{slot} among your {POS_NAMES.get(grp, grp)}."]
        if slot <= top:
            verdict = "Help: he'd play real minutes, and in-season growth comes from ice time and production."
        else:
            verdict = "Hurt: he'd sit low in the lineup on fourth-line or press-box minutes; big minutes in the AHL develop him faster."
        return _answer(verdict, analysis=lines, rows=[p], extra=("gp", "pts"))
    if kind == "smartest":
        grp, gap = _weakest_group(view, tid)
        target = _best_target(view, grp, space)
        bad = sorted(((value_of(view, r["id"], "unmovable") or 0, r) for r in view.team_rows(tid)), key=lambda x: -x[0])
        expiring = sorted((r for r in view.team_rows(tid) if (r["contract"]["years_left"] or 0) <= 1 and r["age"] <= 28),
                          key=lambda r: -r["ovr"])
        lines = [f"You're {st.get('league_rank') or '?'} of {n}, roster strength {srank} of {n}, {_money(space)} in cap space ({status})."]
        if target:
            lines.append(f"Biggest gap: {POS_NAMES.get(grp, grp)} ({gap:.1f} behind the top eight). Best fit available: {target['name']} ({target['ovr']:.0f}, {_money(target['contract']['cap_hit_m'])}).")
        if expiring:
            lines.append(f"Expiring young core to lock up: {expiring[0]['name']} ({expiring[0]['age']}, {expiring[0]['ovr']:.0f}).")
        if bad and bad[0][0] > 3:
            lines.append(f"Contract to move before it hardens: {bad[0][1]['name']} ({_money(bad[0][1]['contract']['cap_hit_m'])}).")
        if status == "rebuilding":
            head = "Sell: move your best veteran on an expiring deal for picks and prospects."
        elif expiring and expiring[0]["ovr"] >= 86:
            head = f"Extend {expiring[0]['name']} before he reaches the market."
        elif target:
            head = f"Trade for {target['name']} to fix your {POS_NAMES.get(grp, grp)}."
        else:
            head = "Stand pat; nothing available beats what you have."
        return _answer(head, analysis=lines)
    if kind == "feel_worse":
        from .derived import team_xg_share
        games = view.team_games(tid)
        last10 = games[-10:]
        rec10 = sum(1 for g in last10 if g[1])
        xs = team_xg_share(view, tid)
        gd = int(st.get("gf") or 0) - int(st.get("ga") or 0)
        lines = [f"Last ten: {rec10}-{len(last10) - rec10}; season goal difference {gd:+d}; expected-goal share {xs if xs is not None else '?'}%.",
                 f"One-goal losses: {sum(1 for _d, w, gf, ga in games if not w and ga - gf == 1)}."]
        return _answer("Recent results and close losses make it feel worse than the underlying numbers say.", analysis=lines)
    return BASE["diagnose"](view, slots, {"raw": raw, "conv": conv})


# =========================================================================== player and team specials
def replacement(view: LiveView, raw: str, slots: Dict[str, Any]) -> Answer:
    f = parse_filters(raw, slots, view)
    pos = "RW" if re.search(r"\brw\b|right wing", raw) else "LW" if re.search(r"\blw\b|left wing", raw) else \
        "C" if re.search(r"\bcent(er|re)\b", raw) else ("D" if re.search(r"\bd\b|defense", raw) else "F")
    nhl = sorted((x for x in view.team_rows(view.user_tid) if _pos_ok(x, pos)), key=lambda x: -(toi_pg(view, x["id"]) or x["ovr"]))
    tier = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3}.get(f.get("tier_word") or "third", 2)
    holder = nhl[tier] if len(nhl) > tier else (nhl[-1] if nhl else None)
    ahl = sorted((x for x in view.team_rows(view.user_tid, ("AHL",)) if _pos_ok(x, pos) or (pos in ("LW", "RW") and x["pos"] in ("LW", "RW", "W"))),
                 key=lambda x: -x["ovr"])
    if not ahl:
        return _answer(f"No AHL {pos} in your system.")
    best = ahl[0]
    lines = [f"{best['name']} ({best['age']}, {best['ovr']:.0f}/{best['pot']:.0f}) is your best AHL option at {pos}."]
    if holder:
        lines.insert(0, f"Your current {['first', 'second', 'third', 'fourth'][tier]}-line {pos}: {holder['name']} ({holder['ovr']:.0f}).")
        lines.append(f"Drop-off: {holder['ovr'] - best['ovr']:+.0f} overall.")
    return _answer(f"{best['name']} is next up.", analysis=lines, rows=ahl[:6], extra=("gp", "pts"))


def surplus_ld(view: LiveView) -> Answer:
    rows = []
    for t in view.teams:
        if team_status(view, t) != "rebuilding" or t == view.user_tid:
            continue
        ld = [r for r in _group(view, t, "LD", ("NHL", "AHL")) if r["ovr"] >= 76 and r["age"] <= 28]
        if len(ld) > 3:
            rows.append((len(ld) - 3, t, ld))
    if not rows:
        out = team_rank(view, "ld_surplus", note="No rebuilding team has more than three NHL-ready left D; all teams shown.")
        return out
    rows.sort(key=lambda x: -x[0])
    n, t, ld = rows[0]
    return _answer(f"The {_team_label(view, t)} have {len(ld)} NHL-ready left D ({n} more than they need).",
                   rows=ld[:8], extra=("cap", "status"), analysis=["Rebuilding teams only; NHL-ready = 76+ overall, 28 or younger."])


def rental_winger(view: LiveView) -> Answer:
    rows = []
    for t in view.teams:
        if team_status(view, t) != "contender" or t == view.user_tid:
            continue
        need = team_measure(view, t, "wing_need") or 0
        space = team_measure(view, t, "cap_space") or 0
        rows.append((need + space * 0.8, t, need, space))
    rows.sort(key=lambda x: -x[0])
    if not rows:
        return _answer("No contenders identified yet.")
    table = {"cols": ["Team", "Wing need", "Cap space"], "rows": [[_team_label(view, t), round(nd, 1), _money(sp)] for _s, t, nd, sp in rows[:8]]}
    return _answer(f"The {_team_label(view, rows[0][1])}: a contender with the weakest wings and room to add salary.", table=table,
                   analysis=["Contenders only, ranked by how far their top-six wingers sit below a 90 average, plus cap space."])


def max_return(view: LiveView, slots: Dict[str, Any], raw: str) -> Answer:
    rows = _resolve_players(view, slots)
    if not rows:
        return _answer("For which player? Name him.")
    r = rows[0]
    mv = market(view, r["id"]) or 0
    cap = r["contract"]["cap_hit_m"] or 0
    yl = r["contract"]["years_left"] or 0
    surplus = (mv - cap) * max(1, yl)
    tier = "a top-pair defenseman or first-line forward, plus a first-round pick" if r["ovr"] >= 92 else \
        "a top-six forward or top-four defenseman, plus a first" if r["ovr"] >= 88 else \
        "a first-round pick plus a B prospect" if r["ovr"] >= 84 else "a second-round pick or a mid prospect"
    lines = [f"{r['name']}: {r['ovr']:.0f} overall, {r['age']}, {_money(cap)} x {yl} years; market value about {_money(mv)}.",
             f"Contract surplus over the term: {surplus:+.1f}M, which is what pushes the price up or down.",
             f"Ceiling for a realistic offer: {tier}."]
    if yl <= 1:
        lines.append("Expiring: buyers treat him as a rental, which caps the price near the deadline.")
    if r["contract"]["nmc"]:
        lines.append("His no-move clause limits the buyers, which also lowers the ceiling.")
    return _answer(f"Top realistic return for {r['name']}: {tier}.", analysis=lines, rows=[r], extra=("cap", "status"))


def better_fit_elsewhere(view: LiveView, slots: Dict[str, Any], conv: Dict[str, Any]) -> Answer:
    rows = _resolve_players(view, slots) or [view.row(p) for p in (conv.get("players") or []) if p in view.players]
    if not rows:
        rows = sorted(view.team_rows(view.user_tid), key=lambda r: (toi_pg(view, r["id"]) or 0) - r["ovr"] * 0.3)[:1]
    r = rows[0]
    grp = "D" if r["pos"] in ("D", "LD", "RD") else ("G" if r["pos"] == "G" else "F")
    mine_slot = sum(1 for x in _group(view, view.user_tid, grp) if x["ovr"] > r["ovr"]) + 1
    out = []
    for t in view.teams:
        if t == view.user_tid:
            continue
        slot = sum(1 for x in _group(view, t, grp) if x["ovr"] > r["ovr"]) + 1
        space = team_measure(view, t, "cap_space") or 0
        if slot < mine_slot and space + 1 >= (r["contract"]["cap_hit_m"] or 0):
            out.append((mine_slot - slot, t, slot))
    out.sort(key=lambda x: -x[0])
    if not out:
        return _answer(f"No team would use {r['name']} higher in the lineup than you do and could fit him.")
    table = {"cols": ["Team", "His slot there", "Cap space"], "rows": [[_team_label(view, t), f"#{s}", _money(team_measure(view, t, 'cap_space') or 0)] for _d, t, s in out[:8]]}
    return _answer(f"The {_team_label(view, out[0][1])} would play {r['name']} #{out[0][2]} at his position; he's #{mine_slot} for you.",
                   table=table, analysis=["Teams where he'd rank higher on the depth chart and his cap hit fits."])


def team_veterans(view: LiveView) -> Answer:
    rows = []
    for t in view.teams:
        if t == view.user_tid:
            continue
        a = team_measure(view, t, "avg_age") or 0
        bonus = 3 if team_status(view, t) == "contender" else 0
        rows.append((a + bonus, t, a))
    rows.sort(key=lambda x: -x[0])
    table = {"cols": ["Team", "Avg age (top 20)", "Status"], "rows": [[_team_label(view, t), a, team_status(view, t)] for _s, t, a in rows[:8]]}
    return _answer(f"The {_team_label(view, rows[0][1])}: an older, win-now roster that already leans on veterans.", table=table,
                   analysis=["Ranked by the average age of their top 20, with contenders pushed up."])


def cap_partner(view: LiveView) -> Answer:
    bad = sorted(((value_of(view, r["id"], "unmovable") or 0, r) for r in view.team_rows(view.user_tid)), key=lambda x: -x[0])
    worst = bad[0][1] if bad and bad[0][0] > 0 else None
    rows = []
    for t in view.teams:
        if t == view.user_tid:
            continue
        space = team_measure(view, t, "cap_space") or 0
        nxt = committed_next(view, t)
        rows.append((space * 1.0 + (95 - nxt) * 0.3, t, space, nxt))
    rows.sort(key=lambda x: -x[0])
    lines = [f"Your committed cap next season: {_money(committed_next(view, view.user_tid))}."]
    if worst:
        lines.append(f"The contract to move: {worst['name']} ({_money(worst['contract']['cap_hit_m'])} x {worst['contract']['years_left']}).")
    table = {"cols": ["Team", "Space now", "Committed next year"], "rows": [[_team_label(view, t), _money(s), _money(n)] for _x, t, s, n in rows[:8]]}
    return _answer(f"The {_team_label(view, rows[0][1])}: the most room to absorb salary now and next season.", table=table, analysis=lines)


def negotiation(view: LiveView, kind: str, raw: str) -> Answer:
    mine = view.team_rows(view.user_tid, ("NHL", "AHL"))
    exp = [r for r in mine if (r["contract"]["years_left"] or 0) <= 1 and r["contract"]["cap_hit_m"]]
    if kind == "priority" or kind == "urgent":
        scored = sorted(exp, key=lambda r: -(r["ovr"] + (r["pot"] - r["ovr"]) * 0.5 - max(0, r["age"] - 28) * 1.5 + (3 if r["contract"]["status"] == "UFA" else 0)))
        if not scored:
            return _answer("No contracts on your roster expire this year.")
        r = scored[0]
        lines = [f"{r['name']}: {r['contract']['status'] or 'status n/a'}, {r['age']}, {r['ovr']:.0f}/{r['pot']:.0f}, market value about {_money(market(view, r['id']) or 0)}."]
        if r["contract"]["status"] == "UFA":
            lines.append("As a UFA he can walk for nothing; an RFA you can still qualify.")
        return _answer(f"{'Do this one first' if kind == 'urgent' else 'Prioritize'}: {r['name']}.", analysis=lines, rows=scored[:8], extra=("cap", "status"))
    if kind == "bridge":
        cands = [r for r in mine if r["contract"]["status"] == "RFA" and (r["contract"]["years_left"] or 0) <= 1 and r["age"] <= 24]
        cands.sort(key=lambda r: -((r["pot"] - r["ovr"]) * 2 - (form(view, r["id"]) or 0) * 3))
        if not cands:
            return _answer("No pending RFAs on your roster.")
        r = cands[0]
        return _answer(f"{r['name']} is the likeliest to take a bridge deal.", rows=cands[:8], extra=("cap", "status"),
                       analysis=["Young RFAs with room to grow and modest production take short deals to bet on themselves."])
    if kind == "breakout_rfa":
        cands = [view.row(p) for p, (t, lv) in view.where.items() if lv == "NHL"]
        cands = [r for r in cands if r["contract"]["status"] == "RFA" and (r["contract"]["years_left"] or 0) <= 1]
        cands.sort(key=lambda r: -((r["pot"] - r["ovr"]) * 1.5 + (form(view, r["id"]) or 0) * 4 + (season_change(view, r["id"]) or 0) * 2))
        if not cands:
            return _answer("No pending RFAs in the league right now.")
        return _answer(f"{cands[0]['name']} ({cands[0]['team']}) is the RFA most likely to break out on a long deal.", rows=cands[:8],
                       extra=("cap", "status"), analysis=["Room to potential, production above his rating, and growth this season."])
    if kind == "below_market":
        cands = [view.row(p) for p, (t, lv) in view.where.items() if lv in ("NHL", "FA") and (view.row(p)["contract"]["years_left"] or 0) <= 1]
        cands = [r for r in cands if market(view, r["id"]) and r["ovr"] >= 78]
        def discount(r: Dict[str, Any]) -> float:
            tr = getattr(view.players[r["id"]], "traits", None)
            loyal = float(getattr(tr, "loyalty", 0.5) or 0.5)
            money = float(getattr(tr, "money_focus", 0.5) or 0.5)
            own = 0.15 if r["team_id"] == view.user_tid else 0.0
            return loyal - money + own + max(0, r["age"] - 31) * 0.03
        cands.sort(key=discount, reverse=True)
        if not cands:
            return _answer("No expiring players rated 78+ right now.")
        r = cands[0]
        tr = getattr(view.players[r["id"]], "traits", None)
        return _answer(f"{r['name']} is the likeliest to sign below his market value (about {_money(market(view, r['id']) or 0)}).",
                       rows=cands[:8], extra=("cap", "status"),
                       analysis=[f"Loyalty {float(getattr(tr, 'loyalty', 0.5)) * 100:.0f}/100, money focus {float(getattr(tr, 'money_focus', 0.5)) * 100:.0f}/100.",
                                 "Expiring players rated 78+, ranked by loyalty over money focus; your own players get a hometown edge."])
    if kind == "asking_more":
        cands = []
        neg = getattr(view.session, "resign_negotiations", None) or {}
        for pid, d in (neg.items() if isinstance(neg, dict) else []):
            ask = None
            if isinstance(d, dict):
                for k in ("ask_aav_m", "asking_aav_m", "player_ask_m", "ask_m", "aav_m"):
                    if d.get(k):
                        ask = float(d[k])
                        break
            if ask and pid in view.players:
                cands.append((ask - (market(view, pid) or ask), view.row(pid), ask))
        if not cands:
            return _answer("No contract asks are on the table right now; asks appear during re-sign season.")
        cands.sort(key=lambda x: -x[0])
        g, r, ask = cands[0]
        return _answer(f"{r['name']} is asking {_money(ask)}, about {g:.1f}M above comparable players.", rows=[x[1] for x in cands[:6]], extra=("cap",))
    return _answer("I couldn't read that contract question.")


# =========================================================================== exact limits
NOT_TRACKED = {
    "important_goal": "Goals aren't stored with their stakes or moment, so there's no way to rank the most important one.",
    "gm_decision": "GM decisions aren't logged as decisions; only their results are (trades, signings, drafts). Ask about your best or worst trade or draft class instead.",
    "proposals": "Trade offers you received and declined aren't saved, so they can't be reviewed in hindsight.",
    "zone_starts": "Zone starts aren't recorded in the game simulation, so I can't rank players by them.",
    "competition": "Quality of competition isn't recorded per shift, so I can't rank lines against elite opponents.",
    "line_combos": "Stats aren't split by line combination or by linemate type (playmakers vs snipers), only by player.",
    "injury_star": "There's no record linking a player's rise to another player's injury.",
    "coaching": "Coaching changes aren't recorded, so there's no before/after to compare.",
    "suppressing": "Player-vs-player suppression would need per-shift linemate data, which the save doesn't keep.",
    "non_natural": "Overall isn't stored per position, so the save can't say how much a full-time position change would cost. Handedness and chapter ratings are the closest guide; ask “what if I move X to RD”.",
}
NEEDS_HISTORY = {
    "draft_value_history": "draft results",
    "draft_class_games": "draft classes with NHL games",
    "playoff_upset": "playoff series",
    "improved_after_leaving": "former players' seasons elsewhere",
    "best_fa_signing": "free-agent signings with results",
    "hyped_failed": "drafted players who had time to develop",
    "season_expectations": "finished seasons",
    "weakest_champion": "champions",
    "draft_pick_regret": "draft classes",
    "second_round": "playoff history",
    "redo_decision": "seasons of decisions and outcomes",
    "three_seasons": "three seasons of results",
    "trade_direction": "trades with results after them",
}


def needs_history(view: LiveView, what: str) -> Answer:
    doc = archive(view.session)
    done = len(doc.get("seasons") or {})
    return _answer(f"That needs {what}, and this save has {done} finished season{'s' if done != 1 else ''}.",
                   analysis=["It becomes answerable once seasons are archived at each rollover; nothing here is invented.",
                             "Until then, ask the same question about this season."])


# =========================================================================== router
# (regex on the raw lowercase question, action, argument)
RULES3: List[Tuple[str, str, Any]] = [
    # exact limits first
    (r"most important goal|biggest goal", "not_tracked", "important_goal"),
    (r"\bgm decision|my decisions?\b.*\bimpact", "not_tracked", "gm_decision"),
    (r"trade (proposals?|offers?) (i |we )?received|proposal received", "not_tracked", "proposals"),
    (r"zone starts?", "not_tracked", "zone_starts"),
    (r"elite competition|quality of competition|against (the )?(best|top) (lines|players)", "not_tracked", "competition"),
    (r"with playmakers|beside snipers|line combination|line combo", "not_tracked", "line_combos"),
    (r"because of an injury to|injury to another", "not_tracked", "injury_star"),
    (r"coaching change|new coach", "not_tracked", "coaching"),
    (r"suppressing", "not_tracked", "suppressing"),
    (r"non[- ]natural position|off position full|moved to (his )?(non|other) ", "not_tracked", "non_natural"),
    (r"relative to (his )?draft position|value relative to draft", "history", "draft_value_history"),
    (r"draft class (produced|with) the most (nhl )?games", "history", "draft_class_games"),
    (r"improved the most after leaving", "history", "improved_after_leaving"),
    (r"free[- ]agent signing in franchise", "history", "best_fa_signing"),
    (r"most hyped but", "history", "hyped_failed"),
    (r"season exceeded expectations", "history", "season_expectations"),
    (r"championship team was .*weakest|weakest champion", "history", "weakest_champion"),
    (r"hurt the franchise most because of the players selected", "history", "draft_pick_regret"),
    (r"always lose in the second round", "history", "second_round"),
    (r"redo one decision", "history", "redo_decision"),
    (r"(luckiest|unluckiest).*(three|3|last few) seasons", "history", "three_seasons"),
    (r"trade changed the direction", "history", "trade_direction"),
    (r"playoff series .*upset|largest upset", "history", "playoff_upset"),
    # reasoning
    (r"trade my top prospect for", "reason", "prospect_for_d"),
    (r"immediate help or (preserve|keep) cap|cap flexibility", "reason", "now_vs_flex"),
    (r"roster construction or (bad )?luck|failing because of", "reason", "roster_vs_luck"),
    (r"single move|one move would|championship probability", "reason", "single_move"),
    (r"rebuild now or|remain competitive", "reason", "rebuild_or_compete"),
    (r"superstar away|several pieces away", "reason", "superstar_away"),
    (r"pipeline strong enough|avoid a rebuild", "reason", "pipeline"),
    (r"holding back (younger|young)", "reason", "vet_blocking"),
    (r"would promoting|should i (promote|call up)|promote .* (help|hurt)", "reason", "promote"),
    (r"single smartest move|smartest move i can make|best move right now", "reason", "smartest"),
    (r"feel worse even though|statistically better", "reason", "feel_worse"),
    # draft board
    (r"ranked higher than scouting|underrated by (the )?scouts", "board", "undervalued"),
    (r"second[- ]round pick has first[- ]round", "board", "second_round_talent"),
    (r"prospect .*likely to become a draft steal", "board", "steal"),
    (r"production is being carried|carried by stronger teammates", "board", "carried"),
    (r"bust risk among the top ten|bust risk .*top 10", "board", "bust_top10"),
    (r"skating (was )?weighted", "board", "skating2x"),
    (r"lowest floor in the first round", "board", "low_floor_r1"),
    (r"highest ceiling outside the lottery", "board", "ceiling_outside_lottery"),
    (r"rarest combination|rarest .*traits", "board", "rare_traits"),
    # teams
    (r"surplus of nhl[- ]ready left", "special", "surplus_ld"),
    (r"overpay for a rental", "special", "rental_winger"),
    (r"one injury away .*goal(ie|tender)", "team", ("goalie_injury_risk", None, False, "The drop from starter to backup, plus how far the backup sits below 80.")),
    (r"values? prospects more than (draft )?picks", "team", ("pipeline", "rebuilding", False, "Rebuilding teams that already invest in their pipeline value prospects over picks.")),
    (r"overvalue veteran|value veterans", "special", "team_veterans"),
    (r"playoff push despite poor underlying|despite poor (underlying|analytics)", "team", ("luck_gap", None, False, "Points percentage minus expected-goal share: the biggest gap is winning beyond their play.")),
    (r"see my player as a better fit|better fit than i do", "special", "better_fit"),
    (r"weakest cent(er|re) depth", "team", ("center_depth", None, True, "Average of each team's top three centres.")),
    (r"brightest future|next decade", "team", ("future", None, False, "Young core, prospect pipeline and future cap room.")),
    (r"future cap structure", "special", "cap_partner"),
    (r"maximum return|most i could get for|realistic(ally)? expect for", "special", "max_return"),
    # contracts
    (r"\brfa\b.*break ?out|break ?out .*long term", "neg", "breakout_rfa"),
    (r"risk on a (7|seven)[- ]year|7[- ]year contract", "player", ("contract_risk_7", {}, False, "Overpay over seven years if signed at today's market value, with normal aging.")),
    (r"asking for (significantly )?more money", "neg", "asking_more"),
    (r"become unmovable|unmovable", "player", ("unmovable", {"scope": "team"}, False, "Cap hit above market value, compounded over the remaining term and any no-move clause.")),
    (r"expiring contract should be prioritized|prioritized before free agency", "neg", "priority"),
    (r"inflated by a career year|career year", "player", ("overperformance", {}, False, "Scoring well above his rating's line, finishing above expected goals.")),
    (r"signed below (expected )?market", "neg", "below_market"),
    (r"accept a bridge", "neg", "bridge"),
    (r"negotiation is most urgent|most urgent negotiation", "neg", "urgent"),
    (r"least likely to age well|age (badly|poorly)", "player", ("aging_risk", {}, False, "Age past 27, reliance on skating, and durability.")),
    # players
    (r"surplus value", "player", ("surplus3", {}, False, "Market value minus cap hit, summed over the next three seasons with normal aging.")),
    (r"closest to replacing|replace my .*(line|pair)", "special", "replacement"),
    (r"first[- ]line results despite|producing .* despite .*(minutes|ice time)", "player", ("prod_vs_toi", {}, False, "Scoring-rate percentile minus ice-time percentile among NHL skaters.")),
    (r"benefiting most from his partner|benefit.* from (his )?partner", "player", ("partner_lift", {}, False, "Partner taken as his best-chemistry defense teammate; results above his own rating.")),
    (r"blocked by", "player", ("blocked", {"scope": "org"}, False, "Prospects good enough for a lineup spot that's held by signed players.")),
    (r"waiver[- ]eligible|waiver eligib|need(s)? waivers", "player", ("waiver_risk", {"scope": "org"}, False, "Not exempt, or close to losing exemption, while still outside the NHL lineup.")),
    (r"power[- ]play points (that are )?inflat|inflated by (the )?power play", "player", ("pp_share", {}, False, "Share of his points that came on the power play (five points minimum).")),
    (r"benefit most from (increased|more) ice time|more ice time", "player", ("prod_vs_toi", {}, False, "Producing beyond his minutes: the best candidates for more.")),
    (r"hurting his linemates|hurts? (his )?linemates", "player", ("hurts_linemates", {}, False, "Top-half points, but the team is outscored with him on the ice.")),
    (r"grown the least relative|grown the least", "player", ("draft_vs_slot", {}, True, "Current rating against what his draft slot usually becomes; lowest first.")),
    (r"hidden development value", "player", ("gap_to_potential", {"level": "prospect"}, False, "Room left between current rating and potential.")),
    (r"largest gap between (current )?overall and potential|gap between .*potential", "player", ("gap_to_potential", {}, False, "Potential minus current overall.")),
    (r"chance of becoming a top[- ]line|top[- ]line player", "player", ("top_line_chance", {}, False, "Potential first, then current rating and production.")),
    (r"slower than (historical )?comparables|developing slower", "player", ("stagnation", {}, False, "No historical database yet; compared against expected in-season growth instead.")),
    (r"stagnat", "player", ("stagnation", {}, False, "No growth this season, low minutes, production under his line, low morale, and age.")),
    (r"outperformed his development|ahead of (his )?development", "player", ("season_change", {}, False, "")),
    (r"suggests an upcoming (overall )?increase|production suggests", "player", ("form", {}, False, "Scoring above his rating's line is what pays at the next growth checkpoint.")),
    (r"growth is most dependent on nhl ice time|dependent on .*ice time", "player", ("stagnation", {"scope": "org"}, False, "Young players whose growth is stalling for lack of minutes.")),
    (r"elite potential but lacks", "player", ("lacks_tools", {}, False, "Potential minus the average of his chapter ratings.")),
    (r"elevates his linemates|makes (his )?linemates better", "player", ("elevates", {}, False, "On-ice goal share vs team, expected-goal share, and his involvement.")),
    (r"carried by his linemates|appears carried", "player", ("carried", {}, False, "Team does well with him on the ice but he's in on few of the goals.")),
    (r"consistently controls possession|controls possession", "player", ("xg_share", {"position": "D"}, False, "Defensemen by on-ice expected-goal share; pairs aren't stored, so it's ranked per player.")),
    (r"never be separated|should stay together", "special", "never_separate"),
    (r"departure would hurt team chemistry|hurt team chemistry", "player", ("chemistry_total", {"scope": "team"}, False, "Chemistry with current teammates plus locker-room presence.")),
    (r"untouchable but analytics", "player", ("overrated", {"scope": "team"}, False, "High rating and pay, poor on-ice results and value.")),
    (r"chance of winning a major (nhl )?award|win (the )?calder", "player", ("award_chance", {"level": None}, False, "Rookie-age NHL players by rating and production (Calder case).")),
    (r"face of the franchise", "player", ("face_of_franchise", {"scope": "team"}, False, "Rating, youth, term, leadership and production.")),
    (r"surprise everyone and become elite|become elite", "player", ("surprise_elite", {}, False, "Young, under 87 overall, high potential, producing and growing.")),
    (r"looks good on paper but hurts winning", "player", ("hurts_linemates", {}, False, "Strong rating and points, outscored on the ice.")),
    (r"most overrated", "player", ("overrated", {}, False, "Rating and pay against on-ice results and market value.")),
    (r"most underrated", "player", ("underrated", {}, False, "On-ice results and value above what his rating suggests, stars excluded.")),
    (r"trust despite terrible|despite (terrible|bad|poor) (recent )?production", "player", ("gap_to_potential", {"level": "prospect", "scope": "org"}, False, "Your prospects ranked by potential still to come; production slumps don't move potential.")),
    (r"\bluckiest\b", "player", ("luck", {}, False, "Goals above expected plus on-ice goal differential above expected, this season.")),
    (r"\bunluckiest\b", "player", ("luck", {}, True, "Goals below expected plus on-ice goal differential below expected, this season.")),
    (r"most nhl[- ]ready", "board_or_org", "ready"),
    # plain team questions, after the specific analytics wording above
    (r"\b(home|road|away)\s+record\b|\brecord (at home|on the road|away)\b|\b(at home|on the road)\b.*\brecord\b", "teamq", "split"),
    (r"(?<!expected )\bgoal (diff|differential)\b|\bgoals? for and against\b|\bgoals against\b.*\bteam\b", "teamq", "goal_diff"),
    (r"^(?!.*\b(free|clear|create|make)\b.*\bcap\b).*(?:\bcap (space|room)\b|\bunder the cap\b|\bover the cap\b)|\bhow much (money|cap)\b.*\b(left|spend|have)\b", "teamq", "cap"),
    (r"\blast (game|night|match)\b|\byesterday\b|\bmost recent game\b|\bprevious game\b", "teamq", "last_game"),
    (r"\blast (5|five|10|ten) games\b|\brecent form\b|\bhow are we (playing|doing) (lately|recently)\b|\bhow are we playing\b|\bhot or cold\b|\blately\b", "teamq", "form"),
    (r"\b(make|making|made|miss|missing) the playoffs\b|\bin a playoff spot\b|\bplayoff (chances?|odds|picture|spot|position|race|line)\b|\bwild ?card\b", "teamq", "playoffs"),
    (r"^(?!.*\b(franchise|all[- ]time|career|single[- ]season|league record|nhl record)\b)"
     r".*(?:\b(what'?s|what is|whats|how'?s|show) (our|my|the [a-z ]+?) record\b|^(our|my) record\b|\brecord this season\b"
     r"|\bhow many (wins|games) (do|have) (we|i)\b)", "teamq", "record"),
    (r"\bshould (i|we) sign\b|\bwho (should|can|could) (i|we) sign\b|\bbest free agents?\b|\bfree agents? (available|out there|left)\b", "teamq", "sign"),
]


def match(raw_question: str) -> Optional[Tuple[str, Any]]:
    q = " ".join(str(raw_question or "").lower().replace("’", "'").split())
    for pattern, action, arg in RULES3:
        if re.search(pattern, q):
            return action, arg
    return None


def run(action: str, arg: Any, view: LiveView, raw_question: str, slots: Dict[str, Any], conv: Dict[str, Any]) -> Answer:
    raw = " ".join(str(raw_question or "").lower().replace("’", "'").split())
    if action == "teamq":
        return team_question(view, arg, raw, slots, conv)
    if action == "not_tracked":
        return _answer(NOT_TRACKED[arg], analysis=["Ask about ratings, stats, contracts, standings, the draft board or this save's history instead."])
    if action == "history":
        return needs_history(view, NEEDS_HISTORY[arg])
    if action == "reason":
        return reason(view, arg, raw, slots, conv)
    if action == "board":
        return draft_board(view, arg)
    if action == "board_or_org":
        return BASE["top_prospects"](view, slots, {"raw": raw + " closest ready", "conv": conv})
    if action == "team":
        measure, status, low, note = arg
        return team_rank(view, measure, status=status, low=low, note=note)
    if action == "neg":
        return negotiation(view, arg, raw)
    if action == "player":
        measure, base, low, note = arg
        f = {**parse_filters(raw, slots, view), **base}
        if "level" in base and base["level"] is None:
            f.pop("level", None)
            f["any_level"] = False
        f["_raw"] = raw
        if measure == "season_change":
            return player_rank(view, "gap_to_potential", f, note="Growth since the season started vs room to grow.") if False else \
                _season_change_rank(view, f)
        return player_rank(view, measure, f, low=low, note=note)
    if action == "special":
        fn = {
            "surplus_ld": lambda: surplus_ld(view), "rental_winger": lambda: rental_winger(view),
            "team_veterans": lambda: team_veterans(view), "better_fit": lambda: better_fit_elsewhere(view, slots, conv),
            "cap_partner": lambda: cap_partner(view), "max_return": lambda: max_return(view, slots, raw),
            "replacement": lambda: replacement(view, raw, slots), "never_separate": lambda: never_separate(view),
        }[arg]
        return fn()
    return _answer("I couldn't work that one out.")


# =========================================================================== team questions
def _games_for(view: LiveView, tid: str) -> List[Dict[str, Any]]:
    """This season's regular-season finals for one club, oldest first, from its side."""
    out = []
    for box in list(getattr(view.session, "game_results", None) or []):
        if not isinstance(box, dict) or box.get("is_playoff"):
            continue
        h, a = str(box.get("home_id") or ""), str(box.get("away_id") or "")
        if tid not in (h, a):
            continue
        hg = int(box.get("home_goals", box.get("home_score")) or 0)
        ag = int(box.get("away_goals", box.get("away_score")) or 0)
        home = tid == h
        gf, ga = (hg, ag) if home else (ag, hg)
        extra = bool(box.get("overtime") or box.get("ot") or box.get("shootout"))
        out.append({"date": str(box.get("calendar_iso") or box.get("iso") or "")[:10], "home": home,
                    "opp": a if home else h, "gf": gf, "ga": ga, "won": gf > ga, "extra": extra,
                    "so": bool(box.get("shootout")), "box": box})
    out.sort(key=lambda g: (g["date"], int(g["box"].get("day") or 0)))
    return out


def _wlo(games: List[Dict[str, Any]]) -> str:
    w = sum(1 for g in games if g["won"])
    otl = sum(1 for g in games if not g["won"] and g["extra"])
    return f"{w}-{len(games) - w - otl}-{otl}"


def _team_for(view: LiveView, raw: str, slots: Dict[str, Any], conv: Dict[str, Any]) -> str:
    if re.search(r"\b(our|my|we|us)\b", raw) or slots.get("self"):
        return view.user_tid
    teams = slots.get("teams") or []
    if teams:
        return str(teams[0])
    if re.search(r"\b(their|they)\b", raw) and conv.get("team"):
        return str(conv["team"])
    return view.user_tid


def _scorers(view: LiveView, box: Dict[str, Any], tid: str) -> List[str]:
    names: Dict[str, int] = {}
    for ev in box.get("scoring_events") or []:
        if not isinstance(ev, dict) or str(ev.get("for_team_id") or ev.get("team_id") or "") != tid:
            continue
        pid = str(ev.get("scorer_id") or ev.get("player_id") or "")
        name = ev.get("scorer_name") or ev.get("scorer") or (view.row(pid).get("name") if pid in view.players else "")
        if name and isinstance(name, str):
            names[name] = names.get(name, 0) + 1
    return [f"{n} ({c})" if c > 1 else n for n, c in names.items()]


def team_question(view: LiveView, kind: str, raw: str, slots: Dict[str, Any], conv: Dict[str, Any]) -> Answer:
    tid = _team_for(view, raw, slots, conv)
    name = _team_label(view, tid)
    you = tid == view.user_tid
    st = _standing(view, tid) or {}
    games = _games_for(view, tid)
    conv["team"] = tid

    if kind == "sign":
        return BASE["find_players"](view, {**slots, "filters": list(set((slots.get("filters") or []) + ["free"]))},
                                    {"raw": raw, "conv": conv})
    if kind == "cap":
        space = _cap_space(view, tid)
        if space is None:
            return _answer(f"I can't read the {name}' cap right now.")
        rows = sorted(view.team_rows(tid), key=lambda r: -(r["contract"]["cap_hit_m"] or 0))
        return _answer(f"{'You have' if you else f'The {name} have'} {_money(space)} in cap space.",
                       analysis=[f"Biggest cap hits: " + ", ".join(f"{r['name']} {_money(r['contract']['cap_hit_m'])}" for r in rows[:3]) + "."] if rows else [],
                       rows=rows[:8], extra=("cap",), followups=["Best contracts", "Who are our free agents next summer?"])
    if not games and kind != "playoffs":
        return _answer(f"The {name} haven't played a regular-season game yet.")

    if kind == "record":
        rec = f"{st.get('w', 0)}-{st.get('l', 0)}-{st.get('otl', 0)}" if st else _wlo(games)
        gf, ga = sum(g["gf"] for g in games), sum(g["ga"] for g in games)
        last10 = games[-10:]
        return _answer(f"The {name} are {rec}" + (f", {st.get('pts')} points in {st.get('gp', len(games))} games." if st else "."),
                       analysis=[f"Last {len(last10)}: {_wlo(last10)}.", f"Home {_wlo([g for g in games if g['home']])}, road {_wlo([g for g in games if not g['home']])}.",
                                 f"Goals {gf} for, {ga} against ({gf - ga:+d})."],
                       followups=["Are we making the playoffs?", "How did we do last game?", "Why are we losing?"])
    if kind == "split":
        home = [g for g in games if g["home"]]
        road = [g for g in games if not g["home"]]
        def gd(gs): return sum(g["gf"] - g["ga"] for g in gs)
        want_road = bool(re.search(r"\b(road|away)\b", raw))
        first, second = (("Road", road), ("Home", home)) if want_road else (("Home", home), ("Road", road))
        return _answer(f"The {name} are {_wlo(first[1])} {'on the road' if want_road else 'at home'} ({gd(first[1]):+d} goal differential).",
                       analysis=[f"{second[0]}: {_wlo(second[1])} ({gd(second[1]):+d}).", f"Overall: {_wlo(games)}."])
    if kind == "goal_diff":
        gf, ga = sum(g["gf"] for g in games), sum(g["ga"] for g in games)
        n = len(games)
        diffs = sorted(((r.get("gf") or 0) - (r.get("ga") or 0), r.get("team_id")) for r in view.standings())
        rank = next((i + 1 for i, (_d, t) in enumerate(sorted(diffs, reverse=True)) if str(t) == str(tid)), None)
        return _answer(f"The {name} are {gf - ga:+d}: {gf} goals for, {ga} against in {n} games.",
                       analysis=[f"{gf / n:.2f} for and {ga / n:.2f} against per game."] + ([f"{rank} of {len(diffs)} in the league by goal differential."] if rank else []))
    if kind == "last_game":
        g = games[-1]
        opp = _team_label(view, g["opp"])
        res = "beat" if g["won"] else "lost to"
        tag = " in a shootout" if g["so"] else (" in overtime" if g["extra"] else "")
        scorers = _scorers(view, g["box"], tid)
        analysis = [f"{'Home' if g['home'] else 'Road'} game on {g['date']}."]
        analysis.append(f"Goals: {', '.join(scorers)}." if scorers else "No goal-by-goal detail was stored for this game.")
        return _answer(f"The {name} {res} the {opp} {g['gf']}-{g['ga']}{tag}.", analysis=analysis,
                       table={"cols": ["Date", "Opponent", "Where", "Score", "Result"],
                              "rows": [[x["date"], _team_label(view, x["opp"]), "Home" if x["home"] else "Road", f"{x['gf']}-{x['ga']}",
                                        "W" if x["won"] else ("OTL" if x["extra"] else "L")] for x in games[-5:][::-1]]})
    if kind == "form":
        last = games[-10:]
        streak = 0
        for g in reversed(games):
            if g["won"] != games[-1]["won"]:
                break
            streak += 1
        return _answer(f"The {name} are {_wlo(last)} in their last {len(last)}.",
                       analysis=[f"Current streak: {streak} {'win' if games[-1]['won'] else 'loss'}{'s' if streak != 1 else ''}.",
                                 f"Goals {sum(g['gf'] for g in last)}-{sum(g['ga'] for g in last)} over that stretch."])
    if kind == "playoffs":
        rows = sorted(view.standings(), key=lambda r: int(r.get("league_rank") or r.get("rank") or 99))
        if not st or not rows:
            return _answer("Standings aren't available yet.")
        conf = [r for r in rows if r.get("conference") == st.get("conference")]
        pos = next((i + 1 for i, r in enumerate(conf) if str(r.get("team_id")) == str(tid)), None)
        from .answers import _playoff_line
        line = _playoff_line(st, conf, pos or 0).replace("You're", f"The {name} are" if not you else "You're")
        gp = int(st.get("gp") or 0)
        verdict = ("Too early to call." if gp < 20 else
                   "On track for the playoffs." if (pos or 99) <= 8 else "On the outside right now.")
        return _answer(f"{verdict} {line}",
                       table={"cols": ["#", "Team", "GP", "PTS"], "rows": [[i + 1, r.get("name"), r.get("gp"), r.get("pts")] for i, r in enumerate(conf[:12])]})
    return _answer("I couldn't work that one out.")


def _season_change_rank(view: LiveView, f: Dict[str, Any]) -> Answer:
    rows = []
    for pid in view.where:
        if not _in_scope(view, pid, {**f, "scope": f.get("scope")}):
            continue
        ch = season_change(view, pid)
        r = view.row(pid)
        if ch is None or r["age"] > 25:
            continue
        rows.append((ch, r))
    rows.sort(key=lambda x: -x[0])
    if not rows or rows[0][0] <= 0:
        return _answer("Nobody has gained yet this season: growth pays at about 20, 41 and 61 games, and no checkpoint has been reached.")
    t = _player_table([r for _c, r in rows[:10]], ("gp", "pts"))
    t["cols"].append("Change")
    for line, (c, _r) in zip(t["rows"], rows[:10]):
        line.append(f"{c:+.1f}")
    return _answer(f"{rows[0][1]['name']} has outpaced his curve most this season ({rows[0][0]:+.1f}).", table=t,
                   analysis=["Players 25 and under, overall gained since the season started."])


def never_separate(view: LiveView) -> Answer:
    from .derived import best_partner

    best = None
    for r in view.team_rows(view.user_tid):
        bp = best_partner(view, r["id"])
        if bp and bp[0] in view.players and (best is None or bp[1] > best[0]):
            best = (bp[1], r, view.row(bp[0]))
    if not best:
        return _answer("No chemistry readings yet between your players.")
    c, a, b = best
    return _answer(f"{a['name']} and {b['name']}: your strongest chemistry pair ({c:.0f}/100).", rows=[a, b], extra=("gp", "pts", "plus_minus"),
                   analysis=["Chemistry is the engine's pair rating between teammates; splitting the top pair costs the most."])
