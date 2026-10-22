"""League rankings for the Storylines desk: who's best, worst, tallest, most important.

Every board is computed from data the sim actually holds (ratings, OVR, the season
ledger, contracts, body size) and every entry carries the numbers that put it there,
so the gossip page can say *why* someone tops a list. Sample-size floors keep a
three-game call-up off the "most useful" board.

Boards are cached per (season, stats revision, roster signature); a read is a dict
lookup once built. In the offseason the season ledger is empty, so stat boards fall
back to last season's archived lines (services/legacy_hall) and say so.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

TOP_N = 10

QUALITATIVE = [
    ("best_player", "Best player", "Highest overall rating in the league."),
    ("worst_player", "Worst regular", "Lowest overall among skaters with 20+ games."),
    ("most_improved", "Most improved", "Biggest OVR gain since the season started."),
    ("biggest_decline", "Biggest decline", "Biggest OVR drop since the season started."),
    ("best_value", "Best bargain", "Points per $1M of cap hit (20+ GP)."),
    ("worst_contract", "Worst contract", "Most cap dollars per point (20+ GP, $3M+)."),
    ("best_finisher", "Best finisher", "Goals above expected (G minus ixG)."),
    ("unluckiest", "Snakebitten", "Expected goals that never went in (ixG minus G)."),
    ("best_two_way", "Best two-way", "xGF% on ice (40+ GP)."),
    ("best_leader", "Room leader", "Leadership, professionalism and room impact ratings."),
    ("biggest_hothead", "Biggest hothead", "Low composure plus penalty minutes."),
    ("most_underrated", "Most underrated", "Point-ranking far above OVR-ranking."),
    ("best_rookie", "Best young skater", "Points by skaters 21 and under."),
    ("best_prospect", "Highest ceiling", "Development potential, players 21 and under."),
]

QUANTITATIVE = [
    ("tallest", "Tallest", "Height in cm."),
    ("heaviest", "Heaviest", "Weight in kg."),
    ("shortest", "Shortest", "Height in cm."),
    ("lightest", "Lightest", "Weight in kg."),
    ("most_useful", "Most useful", "On-ice goal share (GF%) with 30+ GP, weighted by ice time."),
    ("least_useful", "Least useful", "Lowest points per 60 among forwards with 30+ GP."),
    ("most_important", "Most important", "Share of his team's points times share of its ice time."),
    ("fastest", "Fastest", "Top-speed rating."),
    ("hardest_shot", "Hardest shot", "Slap-shot power rating."),
    ("most_ice_time", "Ice-time hog", "Average time on ice per game."),
    ("most_hits", "Biggest hitter", "Hits this season."),
    ("most_blocks", "Shot blocker", "Blocked shots this season."),
    ("most_pim", "Penalty box regular", "Penalty minutes this season."),
    ("highest_paid", "Highest paid", "Cap hit."),
    ("oldest", "Oldest", "Age."),
    ("sniper_pct", "Deadliest shot", "Shooting % with 60+ shots."),
]


def _i(v: Any) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return 0


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _ovr(p: Any) -> float:
    from services.franchise_sim import _player_ovr99
    return float(_player_ovr99(p))


def _cap(p: Any) -> float:
    from services.franchise_sim import _player_cap_hit_millions
    try:
        return float(_player_cap_hit_millions(p))
    except Exception:
        return 0.0


def _age(p: Any) -> int:
    from services.franchise_sim import _player_age_int
    return int(_player_age_int(p) or 0)


def _rating(p: Any, key: str) -> Optional[float]:
    r = getattr(p, "ratings", None)
    if isinstance(r, dict) and r.get(key) is not None:
        try:
            return float(r[key])
        except (TypeError, ValueError):
            return None
    return None


def _body(p: Any) -> Tuple[float, float]:
    out = []
    for k in ("height_cm", "weight_kg"):
        v = getattr(p, k, None) or getattr(getattr(p, "identity", None), k, None)
        out.append(_f(v))
    return out[0], out[1]


def _pos(p: Any) -> str:
    pos = getattr(p, "position", None) or getattr(getattr(p, "identity", None), "position", None)
    return str(getattr(pos, "value", pos) or "").upper()


def _name(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", None) or getattr(p, "id", ""))


def _team_abbr(session: Any, tid: str) -> str:
    t = (getattr(session, "team_by_id", None) or {}).get(str(tid))
    return str(getattr(t, "abbreviation", None) or tid)


def _nhl_players(session: Any) -> List[Tuple[Any, str]]:
    out = []
    for t in list(getattr(getattr(session.sim, "league", None), "teams", None) or []):
        raw = getattr(t, "team_id", None)
        tid = str(raw if raw is not None else getattr(t, "id", ""))  # team_id 0 (Boston) is valid
        for p in list(getattr(t, "roster", None) or []):
            if not getattr(p, "retired", False):
                out.append((p, tid))
    return out


def _season_lines(session: Any) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Current season ledger, or last finalized season from the legacy archive."""
    pss = getattr(session, "player_season_stats", None) or {}
    if any(_i(r.get("gp")) > 0 for r in pss.values() if isinstance(r, dict)):
        return pss, "this season"
    lg = getattr(session, "legacy", None)
    if isinstance(lg, dict) and lg.get("seasons"):
        last = max(int(k) for k, s in lg["seasons"].items() if s.get("complete"))
        lines: Dict[str, Dict[str, Any]] = {}
        for pid, pr in (lg.get("players") or {}).items():
            for s in pr.get("seasons") or []:
                if s.get("season") == last and s.get("source") == "simulated":
                    lines[pid] = dict(s, position=pr.get("position"))
        if lines:
            return lines, f"{last}-{str(last + 1)[-2:]} season"
    return {}, "no games yet"


def _entry(session: Any, p: Any, tid: str, value: float, display: str, why: str) -> Dict[str, Any]:
    return {"player_id": str(getattr(p, "id", "")), "name": _name(p), "position": _pos(p), "team_id": tid,
            "team": _team_abbr(session, tid), "age": _age(p), "ovr": round(_ovr(p)), "value": round(value, 3),
            "display": display, "why": why}


def _board(session: Any, players: Iterable[Tuple[Any, str]], score: Callable[[Any, str], Optional[Tuple[float, str, str]]],
           *, reverse: bool = True, n: int = TOP_N) -> List[Dict[str, Any]]:
    rows = []
    for p, tid in players:
        try:
            got = score(p, tid)
        except Exception:
            got = None
        if got is None:
            continue
        v, disp, why = got
        rows.append(_entry(session, p, tid, v, disp, why))
    rows.sort(key=lambda r: (-r["value"] if reverse else r["value"], r["name"]))
    for i, r in enumerate(rows[:n], 1):
        r["rank"] = i
    return rows[:n]


def _fmt_toi(sec: float) -> str:
    m = int(sec // 60)
    return f"{m}:{int(sec - m * 60):02d}"


def build_rankings(session: Any, *, team_id: Optional[str] = None) -> Dict[str, Any]:
    key = (int(getattr(session, "season_calendar_year", 0) or 0), int(getattr(session, "_stats_revision", 0) or 0),
           int(getattr(session, "calendar_cursor", 0) or 0), str(team_id or ""))
    cache = getattr(session, "_rankings_cache", None)
    if isinstance(cache, tuple) and cache[0] == key:
        return cache[1]
    out = _build(session, team_id)
    try:
        session._rankings_cache = (key, out)
    except Exception:
        pass
    return out


def _build(session: Any, team_id: Optional[str]) -> Dict[str, Any]:
    allp = _nhl_players(session)
    players = [(p, t) for p, t in allp if not team_id or t == str(team_id)]
    skaters = [(p, t) for p, t in players if _pos(p) != "G"]
    lines, source = _season_lines(session)
    L = lambda p: lines.get(str(getattr(p, "id", ""))) or {}  # noqa: E731

    # league-wide context: team totals for "importance", ranks for "underrated"
    team_pts: Dict[str, int] = {}
    team_toi: Dict[str, float] = {}
    for p, t in allp:
        r = L(p)
        if _pos(p) != "G":
            team_pts[t] = team_pts.get(t, 0) + _i(r.get("pts"))
            team_toi[t] = team_toi.get(t, 0.0) + _f(r.get("toi_sec"))
    eligible = [(p, t) for p, t in allp if _pos(p) != "G" and _i(L(p).get("gp")) >= 30]
    pts_rank = {id(p): i for i, (p, _) in enumerate(sorted(eligible, key=lambda x: -_i(L(x[0]).get("pts"))), 1)}
    ovr_rank = {id(p): i for i, (p, _) in enumerate(sorted(eligible, key=lambda x: -_ovr(x[0])), 1)}

    def gp(p: Any) -> int:
        return _i(L(p).get("gp"))

    q: Dict[str, Callable[[Any, str], Optional[Tuple[float, str, str]]]] = {}
    def season_line(p: Any) -> str:
        if not gp(p):
            return "no games yet"
        r = L(p)
        if _pos(p) == "G":
            sa = _i(r.get("shots_against") or r.get("goalie_shots_against"))
            sv = f", {1 - _i(r.get('ga') or r.get('goalie_ga')) / sa:.3f} sv%" if sa else ""
            return f"{_i(r.get('w'))} wins in {gp(p)} GP{sv}"
        return f"{_i(r.get('pts'))} pts in {gp(p)} GP"

    q["best_player"] = lambda p, t: (_ovr(p), f"{_ovr(p):.0f} OVR", season_line(p))
    q["worst_player"] = lambda p, t: None if _pos(p) == "G" or gp(p) < 20 else (-_ovr(p), f"{_ovr(p):.0f} OVR", f"still dressed for {gp(p)} games")

    def improved(p: Any, t: str, sign: int) -> Optional[Tuple[float, str, str]]:
        start = getattr(p, "_season_start_ovr", None)
        if start is None:
            return None
        d = _ovr(p) - float(start)
        if sign * d <= 0.5:
            return None
        return (sign * d, f"{d:+.1f} OVR", f"{float(start):.0f} → {_ovr(p):.0f} since opening night")
    q["most_improved"] = lambda p, t: improved(p, t, 1)
    q["biggest_decline"] = lambda p, t: improved(p, t, -1)

    def value(p: Any, t: str, worst: bool) -> Optional[Tuple[float, str, str]]:
        cap, pts = _cap(p), _i(L(p).get("pts"))
        if _pos(p) == "G" or gp(p) < 20 or cap <= 0:
            return None
        if worst:
            if cap < 3.0:
                return None
            per = cap / max(1, pts)
            return (per, f"${per:.2f}M / pt", f"${cap:.2f}M for {pts} pts")
        per = pts / cap
        return (per, f"{per:.1f} pts / $1M", f"{pts} pts on ${cap:.2f}M")
    q["best_value"] = lambda p, t: value(p, t, False)
    q["worst_contract"] = lambda p, t: value(p, t, True)

    def finishing(p: Any, t: str, sign: int) -> Optional[Tuple[float, str, str]]:
        r = L(p)
        if _pos(p) == "G" or gp(p) < 20 or "ixg" not in r:
            return None
        d = (_i(r.get("g")) - _f(r.get("ixg"))) * sign
        return (d, f"{sign * d:+.1f} G vs xG", f"{_i(r.get('g'))} goals on {_f(r.get('ixg')):.1f} expected")
    q["best_finisher"] = lambda p, t: finishing(p, t, 1)
    q["unluckiest"] = lambda p, t: finishing(p, t, -1)

    def two_way(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        r = L(p)
        xf, xa = _f(r.get("xgf")), _f(r.get("xga"))
        if _pos(p) == "G" or gp(p) < 40 or xf + xa <= 0:
            return None
        pct = xf / (xf + xa) * 100
        return (pct, f"{pct:.1f} xGF%", f"{xf:.0f} xGF / {xa:.0f} xGA on ice")
    q["best_two_way"] = two_way

    def leader(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        vals = [(_rating(p, k), w) for k, w in (("per_leadership", 0.45), ("per_professionalism", 0.25), ("st_locker_room_impact", 0.3))]
        have = [(v, w) for v, w in vals if v is not None]
        if not have:
            return None
        s = sum(v * w for v, w in have) / sum(w for _, w in have)
        cap = " · captain" if getattr(p, "is_captain", False) else ""
        return (s, f"{s:.0f}", f"leadership {_rating(p, 'per_leadership') or 0:.0f}, professionalism {_rating(p, 'per_professionalism') or 0:.0f}{cap}")
    q["best_leader"] = leader

    def hothead(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        comp = _rating(p, "per_emotional_stability")
        if comp is None:
            return None
        pim = _i(L(p).get("pim"))
        s = (100 - comp) + pim * 0.6
        return (s, f"{pim} PIM", f"composure {comp:.0f}, {pim} penalty minutes")
    q["biggest_hothead"] = hothead

    def underrated(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        if id(p) not in pts_rank:
            return None
        d = ovr_rank[id(p)] - pts_rank[id(p)]
        if d < 25:
            return None
        return (float(d), f"#{pts_rank[id(p)]} in pts", f"#{ovr_rank[id(p)]} by OVR ({_ovr(p):.0f}) but {_i(L(p).get('pts'))} pts")
    q["most_underrated"] = underrated
    q["best_rookie"] = lambda p, t: None if _pos(p) == "G" or _age(p) > 21 or gp(p) < 10 else (
        float(_i(L(p).get("pts"))), f"{_i(L(p).get('pts'))} pts", f"age {_age(p)}, {gp(p)} GP")

    def ceiling(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        pot = _rating(p, "dev_potential")
        if pot is None or _age(p) > 21:
            return None
        return (pot, f"{pot:.0f} POT", f"age {_age(p)}, {_ovr(p):.0f} OVR today")
    q["best_prospect"] = ceiling

    n: Dict[str, Callable[[Any, str], Optional[Tuple[float, str, str]]]] = {}

    def body(idx: int, unit: str, sign: int) -> Callable[[Any, str], Optional[Tuple[float, str, str]]]:
        def f(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
            v = _body(p)[idx]
            if v <= 0:
                return None
            other = _body(p)[1 - idx]
            imp = f"{int(v // 2.54 // 12)}'{int(round(v / 2.54)) % 12}\"" if idx == 0 else f"{round(v * 2.2046)} lb"
            return (sign * v, f"{v:.0f} {unit}", f"{imp} · {other:.0f} {'kg' if idx == 0 else 'cm'}")
        return f
    n["tallest"], n["shortest"] = body(0, "cm", 1), body(0, "cm", -1)
    n["heaviest"], n["lightest"] = body(1, "kg", 1), body(1, "kg", -1)

    def useful(p: Any, t: str, sign: int) -> Optional[Tuple[float, str, str]]:
        r = L(p)
        if gp(p) < 30:
            return None
        if sign > 0:
            gf, ga = _f(r.get("gf_on")), _f(r.get("ga_on"))
            if gf + ga <= 0 or _pos(p) == "G":
                return None
            toi = _f(r.get("toi_sec")) / max(1, gp(p))
            share = gf / (gf + ga)
            return (share * (toi / 60.0) ** 0.5, f"{share * 100:.1f} GF%", f"{gf:.0f}-{ga:.0f} on ice in {_fmt_toi(toi)} a night")
        if _pos(p) in ("G", "D"):
            return None
        toi = _f(r.get("toi_sec"))
        if toi <= 0:
            return None
        p60 = _i(r.get("pts")) / (toi / 3600.0)
        return (-p60, f"{p60:.2f} pts/60", f"{_i(r.get('pts'))} pts in {gp(p)} GP")
    n["most_useful"] = lambda p, t: useful(p, t, 1)
    n["least_useful"] = lambda p, t: useful(p, t, -1)

    def important(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        r = L(p)
        if _pos(p) == "G" or gp(p) < 20 or not team_pts.get(t) or not team_toi.get(t):
            return None
        ps = _i(r.get("pts")) / team_pts[t]
        ts = _f(r.get("toi_sec")) / team_toi[t]
        return (ps * ts * 1000, f"{ps * 100:.0f}% of pts", f"{ps * 100:.0f}% of team points, {ts * 100:.1f}% of its ice time")
    n["most_important"] = important

    def rating_board(key: str, label: str) -> Callable[[Any, str], Optional[Tuple[float, str, str]]]:
        def f(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
            v = _rating(p, key)
            return None if v is None or _pos(p) == "G" else (v, f"{v:.0f}", label)
        return f
    n["fastest"] = rating_board("skg_speed", "top-speed rating")
    n["hardest_shot"] = rating_board("off_slap_shot_power", "slap-shot power rating")

    def stat(key: str, label: str, min_gp: int = 10, skater: bool = True) -> Callable[[Any, str], Optional[Tuple[float, str, str]]]:
        def f(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
            if skater and _pos(p) == "G":
                return None
            if gp(p) < min_gp:
                return None
            v = _i(L(p).get(key))
            return None if v <= 0 else (float(v), f"{v} {label}", f"in {gp(p)} games")
        return f
    n["most_hits"] = stat("hit", "hits")
    n["most_blocks"] = stat("blk", "blocks")
    n["most_pim"] = stat("pim", "PIM")

    def toi(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        if _pos(p) == "G" or gp(p) < 20:
            return None
        v = _f(L(p).get("toi_sec")) / gp(p)
        return (v, _fmt_toi(v), f"{gp(p)} GP, {_i(L(p).get('pts'))} pts")
    n["most_ice_time"] = toi
    n["highest_paid"] = lambda p, t: None if _cap(p) <= 0 else (_cap(p), f"${_cap(p):.2f}M", f"{_ovr(p):.0f} OVR, age {_age(p)}")
    n["oldest"] = lambda p, t: None if not _age(p) else (float(_age(p)), f"{_age(p)}", f"{_ovr(p):.0f} OVR")

    def sniper(p: Any, t: str) -> Optional[Tuple[float, str, str]]:
        r = L(p)
        sog = _i(r.get("sog"))
        if _pos(p) == "G" or sog < 60:
            return None
        pct = _i(r.get("g")) / sog * 100
        return (pct, f"{pct:.1f}%", f"{_i(r.get('g'))} goals on {sog} shots")
    n["sniper_pct"] = sniper

    stat_boards = {"worst_player", "best_value", "worst_contract", "best_finisher", "unluckiest", "best_two_way", "biggest_hothead",
                   "most_underrated", "best_rookie", "most_useful", "least_useful", "most_important", "most_ice_time", "most_hits",
                   "most_blocks", "most_pim", "sniper_pct"}

    def section(defs: List[Tuple[str, str, str]], fns: Dict[str, Callable[..., Any]]) -> List[Dict[str, Any]]:
        res = []
        for key, label, how in defs:
            pool = skaters if key in ("best_rookie", "worst_player") else players
            rows = _board(session, pool, fns[key])
            res.append({"key": key, "label": label, "method": how, "rows": rows,
                        "source": source if key in stat_boards else "current ratings",
                        "empty_reason": None if rows else ("Not enough games played yet." if key in stat_boards else "No player qualifies.")})
        return res

    return {"scope": "team" if team_id else "league", "team_id": team_id, "stats_source": source,
            "qualitative": section(QUALITATIVE, q), "quantitative": section(QUANTITATIVE, n)}
