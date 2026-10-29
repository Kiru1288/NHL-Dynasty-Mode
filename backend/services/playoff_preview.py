"""Series Hype Panel: the story of one playoff matchup, built from the save.

Everything here is read from data the sim already keeps (finals, player stat
ledgers, the legacy archive), so it costs one pass over this season's games.
The result is cached until another game is played.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

# Named rivalries, by abbreviation pair.
RIVALRIES = {
    frozenset(("OTT", "TOR")): "Battle of Ontario",
    frozenset(("EDM", "CGY")): "Battle of Alberta",
    frozenset(("BOS", "MTL")): "Original Six Grudge Match",
    frozenset(("TOR", "MTL")): "Original Six Classic",
    frozenset(("CHI", "DET")): "Original Six Classic",
    frozenset(("BOS", "TOR")): "Original Six Classic",
    frozenset(("NYR", "NYI")): "Battle of New York",
    frozenset(("NYR", "NJD")): "Hudson River Rivalry",
    frozenset(("PHI", "PIT")): "Battle of Pennsylvania",
    frozenset(("WSH", "PIT")): "Capitals vs Penguins",
    frozenset(("ANA", "LAK")): "Freeway Face-Off",
    frozenset(("FLA", "TBL")): "Battle of Florida",
    frozenset(("COL", "DET")): "Avalanche vs Red Wings",
    frozenset(("VAN", "CGY")): "Pacific Grudge",
    frozenset(("MTL", "QUE")): "Battle of Quebec",
}

# Last Stanley Cup before the save began (real NHL clubs); None = never.
REAL_LAST_CUP = {
    "ANA": 2007, "BOS": 2011, "BUF": None, "CGY": 1989, "CAR": 2006, "CHI": 2015, "COL": 2022, "CBJ": None,
    "DAL": 1999, "DET": 2008, "EDM": 1990, "FLA": 2025, "LAK": 2014, "MIN": None, "MTL": 1993, "NSH": None,
    "NJD": 2003, "NYI": 1983, "NYR": 1994, "OTT": None, "PHI": 1975, "PIT": 2017, "SJS": None, "SEA": None,
    "STL": 2019, "TBL": 2021, "TOR": 1967, "UTA": None, "VAN": None, "VGK": 2023, "WSH": 2018, "WPG": None,
}


def _i(v: Any) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return 0


def _series_prob(p: float) -> float:
    """Chance to win four games first with per-game chance p."""
    q = 1.0 - p
    return sum(math.comb(3 + k, k) * p ** 4 * q ** k for k in range(4))


def _likely_length(p: float) -> Tuple[bool, int]:
    q = 1.0 - p
    best, best_p = (True, 4), -1.0
    for fav, pw, pl in ((True, p, q), (False, q, p)):
        for k in range(4):
            prob = math.comb(3 + k, k) * pw ** 4 * pl ** k
            if prob > best_p:
                best, best_p = (fav, 4 + k), prob
    return best


def _team_season(results: List[Dict[str, Any]], tid: str) -> Dict[str, Any]:
    games = []
    for b in results:
        if not isinstance(b, dict) or b.get("is_playoff") or str(b.get("stat_scope") or "regular_season") != "regular_season":
            continue
        h, a = str(b.get("home_id") or ""), str(b.get("away_id") or "")
        if tid not in (h, a):
            continue
        hg, ag = _i(b.get("home_goals", b.get("home_score"))), _i(b.get("away_goals", b.get("away_score")))
        gf, ga = (hg, ag) if tid == h else (ag, hg)
        won = gf > ga or (gf == ga and str(b.get("shootout_winner_id") or b.get("winner_id") or "") == tid)
        extra = bool(b.get("overtime") or b.get("ot") or b.get("shootout"))
        games.append({"key": (str(b.get("calendar_iso") or b.get("iso") or ""), _i(b.get("day"))), "opp": a if tid == h else h,
                      "gf": gf, "ga": ga, "won": won, "extra": extra})
    games.sort(key=lambda g: g["key"])
    gp = len(games)
    w = sum(1 for g in games if g["won"])
    otl = sum(1 for g in games if not g["won"] and g["extra"])
    gf = sum(g["gf"] for g in games)
    ga = sum(g["ga"] for g in games)
    last = games[-10:]
    lw = sum(1 for g in last if g["won"])
    lotl = sum(1 for g in last if not g["won"] and g["extra"])
    streak, kind = 0, None
    for g in reversed(games):
        k = "W" if g["won"] else "L"
        if kind is None:
            kind = k
        if k != kind:
            break
        streak += 1
    return {"games": games, "gp": gp, "w": w, "otl": otl, "l": gp - w - otl, "pts": 2 * w + otl, "gf": gf, "ga": ga,
            "last10": f"{lw}-{len(last) - lw - lotl}-{lotl}", "last10_w": lw, "streak": f"{kind}{streak}" if kind else ""}


def _players(session: Any, tid: str) -> List[Dict[str, Any]]:
    try:
        from services.puckgpt.data import live_view

        view = live_view(session)
        return [r for r in view.team_rows(tid) if r]
    except Exception:
        log.debug("preview roster read failed", exc_info=True)
        return []


def _abbr(session: Any, tid: str) -> str:
    t = (getattr(session, "team_by_id", None) or {}).get(tid)
    return str(getattr(t, "abbreviation", None) or getattr(t, "abbr", None) or tid)[:3].upper()


def _name(session: Any, tid: str) -> str:
    t = (getattr(session, "team_by_id", None) or {}).get(tid)
    city = str(getattr(t, "city", "") or "")
    return city or str(getattr(t, "full_name", "") or _abbr(session, tid))


def build_preview(session: Any, high_id: str, low_id: str) -> Dict[str, Any]:
    high_id, low_id = str(high_id or ""), str(low_id or "")
    results = list(getattr(session, "game_results", None) or [])
    key = f"{id(session)}|{len(results)}|{high_id}|{low_id}"
    with _LOCK:
        hit = _CACHE.get(key)
    if hit:
        return hit

    hs, ls = _team_season(results, high_id), _team_season(results, low_id)
    ha, la = _abbr(session, high_id), _abbr(session, low_id)
    hn, ln = _name(session, high_id), _name(session, low_id)

    # Per-game strength: points share blended with a Pythagorean goal share.
    def strength(s):
        pct = s["pts"] / max(1, 2 * s["gp"]) if s["gp"] else 0.5
        pyth = (s["gf"] ** 2) / max(1, s["gf"] ** 2 + s["ga"] ** 2) if s["gf"] + s["ga"] else 0.5
        return max(0.05, min(0.95, 0.55 * pct + 0.45 * pyth))

    sh, sl = strength(hs), strength(ls)
    p_game = (sh * (1 - sl)) / max(1e-6, sh * (1 - sl) + sl * (1 - sh))
    p_game = min(0.8, max(0.2, 0.5 + (p_game - 0.5) * 0.85 + 0.015))  # small home-ice edge
    p_series = _series_prob(p_game)
    fav_high, length = _likely_length(p_game)
    pick = f"{ha if fav_high else la} in {length}"

    # Head to head this season, from the higher seed's side.
    h2h = [g for g in hs["games"] if g["opp"] == low_id]
    h2h_w = sum(1 for g in h2h if g["won"])
    h2h_line = f"{ha} {h2h_w}-{len(h2h) - h2h_w} {la}" if h2h else "Didn't meet this season"

    # Save history: past series and Cups.
    lg: Dict[str, Any] = {}
    try:
        from services.legacy_hall import ensure_legacy

        lg = ensure_legacy(session)
    except Exception:
        lg = getattr(session, "legacy", None) or {}
    pair = "|".join(sorted((high_id, low_id)))
    rec = (lg.get("h2h") or {}).get(pair) or {}
    past_series = int(rec.get("series") or 0)
    sw = rec.get("series_wins") or {}
    history = None
    if past_series:
        history = f"Met {past_series}× in the playoffs: {ha} {int(sw.get(high_id) or 0)}, {la} {int(sw.get(low_id) or 0)}"
    cups: Dict[str, List[int]] = {}
    for yr, s in (lg.get("seasons") or {}).items():
        cid = str((s or {}).get("champion_id") or "")
        if cid:
            cups.setdefault(cid, []).append(_i(yr))
    year = _i(getattr(session, "season_calendar_year", 0)) or 2026

    def drought(tid: str, ab: str) -> Dict[str, Any]:
        mine = sorted(cups.get(tid) or [])
        if mine:
            last = mine[-1]
            return {"last_cup": last, "label": f"Last Cup: {last}", "years": max(0, year - last)}
        last = REAL_LAST_CUP.get(ab, "unknown") if ab in REAL_LAST_CUP else "unknown"
        if last == "unknown":
            return {"last_cup": None, "label": "No Cup in this save", "years": None}
        if last is None:
            return {"last_cup": None, "label": "Still chasing a first Cup", "years": None, "never": True}
        return {"last_cup": last, "label": f"Last Cup: {last}", "years": max(0, year - last)}

    dh, dl = drought(high_id, ha), drought(low_id, la)

    hp, lp = _players(session, high_id), _players(session, low_id)

    def top_skater(rows):
        sk = [r for r in rows if r.get("pos") != "G"]
        return max(sk, key=lambda r: (r.get("pts", 0), r.get("ovr", 0)), default=None)

    def starter(rows):
        gs = [r for r in rows if r.get("pos") == "G"]
        return max(gs, key=lambda r: (r.get("gp", 0), r.get("ovr", 0)), default=None)

    def xfactor(rows, top):
        sk = [r for r in rows if r.get("pos") != "G" and r.get("gp", 0) >= 20 and (not top or r["id"] != top["id"])]
        return max(sk, key=lambda r: (r.get("gwg", 0), r.get("pts", 0) / max(1, r.get("gp", 1))), default=None)

    def mini(r, kind="skater"):
        if not r:
            return None
        if kind == "goalie":
            sv = r.get("sv_pct")
            line = f".{int(round(sv * 1000)):03d} SV%" if sv else f"{r.get('w', 0)} W"
            return {"id": r["id"], "name": r["name"], "pos": "G", "line": f"{r.get('w', 0)} W · {line}"}
        return {"id": r["id"], "name": r["name"], "pos": r.get("pos"), "line": f"{r.get('g', 0)}G {r.get('a', 0)}A · {r.get('pts', 0)} pts"}

    th, tl = top_skater(hp), top_skater(lp)
    injuries = [{"team": ab, "name": r["name"], "ovr": r.get("ovr")}
                for ab, rows in ((ha, hp), (la, lp)) for r in rows if r.get("injured") and (r.get("ovr") or 0) >= 78]
    injuries.sort(key=lambda x: -(x.get("ovr") or 0))

    # Pressure (0-100): favourites, long droughts and old cores feel it most.
    def pressure(s, d, rows, is_fav):
        score = 38 + (22 if is_fav else 0)
        score += 12 * max(0.0, s["pts"] / max(1, 2 * s["gp"]) - 0.55) / 0.15
        yrs = d.get("years")
        score += 14 if d.get("never") else (min(14, (yrs or 0) / 3) if yrs is not None else 4)
        core = sorted((r for r in rows if r.get("pos") != "G"), key=lambda r: -(r.get("ovr") or 0))[:5]
        if core and sum(r.get("age", 0) for r in core) / len(core) >= 29:
            score += 8
        return int(max(5, min(99, round(score))))

    pr_h = pressure(hs, dh, hp, fav_high)
    pr_l = pressure(ls, dl, lp, not fav_high)

    gap = hs["pts"] - ls["pts"]
    rivalry = RIVALRIES.get(frozenset((ha, la)))
    tags: List[str] = []
    if rivalry:
        tags.append("Rivalry")
    if past_series:
        tags.append("Playoff rematch")
    if abs(gap) >= 12:
        tags.append("Upset watch")
    if dh.get("never") or dl.get("never"):
        tags.append("First-Cup hunt")
    hot = max((hs, ha), (ls, la), key=lambda x: x[0]["last10_w"])
    if hot[0]["last10_w"] >= 7:
        tags.append(f"{hot[1]} red hot")

    if rivalry:
        headline = f"{rivalry.upper()} HITS THE PLAYOFFS"
    elif past_series:
        headline = f"{ha} AND {la} RENEW THEIR PLAYOFF FEUD"
    elif abs(gap) >= 12:
        dog = la if gap > 0 else ha
        headline = f"CAN {dog} PULL OFF THE UPSET?"
    elif h2h and h2h_w in (0, len(h2h)) and len(h2h) >= 3:
        boss = ha if h2h_w else la
        headline = f"{boss} OWNED THIS MATCHUP ALL SEASON"
    else:
        headline = f"{hn.upper()} DRAWS {ln.upper()}"

    story = []
    if abs(gap) >= 1:
        lead_ab, trail_ab = (ha, la) if gap > 0 else (la, ha)
        story.append(f"{lead_ab} finished {abs(gap)} point{'s' if abs(gap) != 1 else ''} ahead of {trail_ab}.")
    if h2h:
        story.append(f"Season series: {h2h_line}.")
    for ab, d in ((ha, dh), (la, dl)):
        if d.get("never"):
            story.append(f"{ab} is still chasing its first Stanley Cup.")
        elif d.get("years") and d["years"] >= 25:
            story.append(f"{ab} hasn't lifted the Cup in {d['years']} years.")
    story.append(f"Form: {ha} {hs['last10']} in the last 10, {la} {ls['last10']}.")

    out = {
        "headline": headline,
        "rivalry": rivalry,
        "tags": tags[:4],
        "story": story[:4],
        "odds": {"high": round(p_series * 100), "low": 100 - round(p_series * 100)},
        "pick": pick,
        "season_series": h2h_line,
        "history": history,
        "teams": {
            "high": {"abbrev": ha, "record": f"{hs['w']}-{hs['l']}-{hs['otl']}", "pts": hs["pts"], "gf": hs["gf"], "ga": hs["ga"],
                     "last10": hs["last10"], "streak": hs["streak"], "drought": dh, "pressure": pr_h,
                     "star": mini(th), "goalie": mini(starter(hp), "goalie"), "xfactor": mini(xfactor(hp, th))},
            "low": {"abbrev": la, "record": f"{ls['w']}-{ls['l']}-{ls['otl']}", "pts": ls["pts"], "gf": ls["gf"], "ga": ls["ga"],
                    "last10": ls["last10"], "streak": ls["streak"], "drought": dl, "pressure": pr_l,
                    "star": mini(tl), "goalie": mini(starter(lp), "goalie"), "xfactor": mini(xfactor(lp, tl))},
        },
        "injuries": injuries[:4],
    }
    with _LOCK:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = out
    return out
