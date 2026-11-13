"""Bench Buzz feature desks: Three Stars, Mailbag, Hot Seat and Rumour Ledger.

Everything here is read from the save (box-score book, standings-from-results,
fan profiles, coach/owner state, the story feed). Built once per sim day and
cached on the session; the mailbag and rumour ledger persist small dicts on the
session so answers stick and resolved rumours survive story pruning.
"""
from __future__ import annotations

import logging
import random
import zlib
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

FEATURES_VERSION = 1
WINDOW_DAYS = 7
RUMOUR_MISS_DAYS = 30
MAILBAG_KEEP = 80

# Compact box line layout (team_id first) — mirrors serialization.BOX_LINE_KEYS.
_KEYS = (
    "gp", "g", "a", "plus_minus", "pim", "sog", "hit", "blk", "toi_sec",
    "ppg", "ppa", "shg", "sha", "pp_toi_sec", "pk_toi_sec", "cf", "ca", "ixg",
    "saves", "shots_against", "ga", "w", "l", "otl", "so",
)
_I = {k: i + 1 for i, k in enumerate(_KEYS)}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _fs():
    from services import franchise_sim as fs  # noqa: WPS433

    return fs


def _num(row: Any, key: str) -> float:
    try:
        v = row[_I[key]]
        return float(v or 0)
    except (IndexError, KeyError, TypeError, ValueError):
        return 0.0


def _iso_to_date(iso: str) -> Optional[date]:
    try:
        return date.fromisoformat(str(iso or "")[:10])
    except ValueError:
        return None


def _days_between(a: str, b: str) -> Optional[int]:
    da, db = _iso_to_date(a), _iso_to_date(b)
    if da is None or db is None:
        return None
    return (db - da).days


def _pretty_date(iso: str) -> str:
    d = _iso_to_date(iso)
    if d is None:
        return ""
    return f"{d.strftime('%b')} {d.day}"


def _sv_str(sv: Optional[float]) -> str:
    if sv is None:
        return "—"
    return f"{sv:.3f}".lstrip("0") if sv < 1 else "1.000"


def _plural(n: int, word: str, many: Optional[str] = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


_NUM_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _nw(n: int) -> str:
    return _NUM_WORDS[n] if 0 <= n < len(_NUM_WORDS) else str(n)


def _team(session: Any, tid: str) -> Any:
    return (getattr(session, "team_by_id", None) or {}).get(str(tid))


def _abbr(session: Any, tid: str) -> str:
    tm = _team(session, tid)
    if tm is None:
        return str(tid or "")
    try:
        return str(_fs()._franchise_team_abbrev(tm) or "")
    except Exception:
        return str(getattr(tm, "abbreviation", "") or getattr(tm, "abbr", "") or tid)


def _team_name(session: Any, tid: str) -> str:
    tm = _team(session, tid)
    if tm is None:
        return str(tid or "")
    try:
        return str(_fs()._display_team(tm))
    except Exception:
        return str(getattr(tm, "full_name", "") or getattr(tm, "name", "") or tid)


def _nickname(session: Any, tid: str) -> str:
    tm = _team(session, tid)
    nm = str(getattr(tm, "name", "") or "") if tm is not None else ""
    return nm or _team_name(session, tid).split(" ")[-1]


def _today_iso(session: Any) -> str:
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    try:
        iso = _fs()._calendar_iso_for_day(session, cur)
        if iso:
            return str(iso)
    except Exception:
        _log.debug("suppressed", exc_info=True)
    return ""


def _season_bucket(session: Any) -> Dict[str, Any]:
    book = getattr(session, "box_score_book", None)
    if not isinstance(book, dict):
        return {}
    season = str(int(getattr(session, "season_calendar_year", 0) or 0))
    return ((book.get("seasons") or {}).get(season) or {}) if isinstance(book.get("seasons"), dict) else {}


def _regular_games(session: Any) -> List[Dict[str, Any]]:
    games = _season_bucket(session).get("games") or {}
    out = [
        g for g in games.values()
        if isinstance(g, dict) and str(g.get("scope") or "regular_season") == "regular_season"
    ]
    out.sort(key=lambda g: (int(g.get("day") or 0), str(g.get("id") or "")))
    return out


def _people(session: Any) -> Dict[str, Any]:
    return _season_bucket(session).get("people") or {}


def _player_name(session: Any, pid: str) -> str:
    row = _people(session).get(str(pid))
    if isinstance(row, (list, tuple)) and row:
        return str(row[0] or "Player")
    st = (getattr(session, "player_season_stats", None) or {}).get(str(pid)) or {}
    return str(st.get("name") or "Player")


def _player_pos(session: Any, pid: str) -> str:
    row = _people(session).get(str(pid))
    if isinstance(row, (list, tuple)) and len(row) > 1:
        return str(row[1] or "")
    return ""


def _winner_loser(g: Dict[str, Any]) -> Tuple[str, str, bool]:
    """(winner_id, loser_id, loser_gets_point)."""
    hid, aid = str(g.get("home") or ""), str(g.get("away") or "")
    hg, ag = int(g.get("hg") or 0), int(g.get("ag") or 0)
    extra = bool(g.get("ot") or g.get("so"))
    if hg > ag:
        return hid, aid, extra
    if ag > hg:
        return aid, hid, extra
    sow = str(g.get("sow") or "")
    if sow:
        return sow, (aid if sow == hid else hid), True
    return "", "", False


def _team_records(session: Any, games: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    recs: Dict[str, Dict[str, Any]] = {}

    def _r(tid: str) -> Dict[str, Any]:
        return recs.setdefault(tid, {"gp": 0, "w": 0, "l": 0, "otl": 0, "pts": 0, "gf": 0, "ga": 0, "results": []})

    for g in games:
        hid, aid = str(g.get("home") or ""), str(g.get("away") or "")
        if not hid or not aid:
            continue
        win, lose, otl = _winner_loser(g)
        hg, ag = int(g.get("hg") or 0), int(g.get("ag") or 0)
        for tid, gf, ga in ((hid, hg, ag), (aid, ag, hg)):
            r = _r(tid)
            r["gp"] += 1
            r["gf"] += gf
            r["ga"] += ga
            if tid == win:
                r["w"] += 1
                r["pts"] += 2
                r["results"].append("W")
            elif tid == lose and otl:
                r["otl"] += 1
                r["pts"] += 1
                r["results"].append("O")
            else:
                r["l"] += 1
                r["results"].append("L")
    for r in recs.values():
        res = r["results"]
        streak_kind, n = "", 0
        for x in reversed(res):
            k = "W" if x == "W" else "L"
            if not streak_kind:
                streak_kind, n = k, 1
            elif k == streak_kind:
                n += 1
            else:
                break
        r["streak_kind"] = streak_kind
        r["streak"] = n
        last10 = res[-10:]
        r["last10"] = f"{last10.count('W')}-{last10.count('L')}-{last10.count('O')}"
        r["pts_pct"] = round(r["pts"] / (2 * r["gp"]), 3) if r["gp"] else None
        r["record"] = f"{r['w']}-{r['l']}-{r['otl']}"
        r.pop("results", None)
    return recs


def _attach_faces(session: Any, rows: List[Dict[str, Any]]) -> None:
    try:
        _fs()._attach_storyline_headshots(session, rows)
    except Exception:
        _log.debug("suppressed", exc_info=True)


def _roster_index(session: Any) -> Dict[str, Tuple[str, Any]]:
    """player_id -> (team_id, player) across NHL + AHL rosters."""
    idx: Dict[str, Tuple[str, Any]] = {}
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        for attr in ("roster", "ahl_roster", "injured_reserve", "scratches"):
            for pl in list(getattr(tm, attr, None) or []):
                pid = str(getattr(pl, "id", "") or "")
                if pid and pid not in idx:
                    idx[pid] = (str(tid), pl)
    return idx


# ---------------------------------------------------------------------------
# 1. Three Stars
# ---------------------------------------------------------------------------

def _aggregate(games: List[Dict[str, Any]], team_filter: str = "") -> Dict[str, Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = {}
    for g in games:
        lines = g.get("lines") or {}
        gwg = str(g.get("gwg") or "")
        hid, aid = str(g.get("home") or ""), str(g.get("away") or "")
        for pid, row in lines.items():
            if not isinstance(row, (list, tuple)) or not row:
                continue
            tid = str(row[0] or "")
            if team_filter and tid != team_filter:
                continue
            a = agg.setdefault(pid, {
                "player_id": pid, "team_id": tid, "gp": 0, "g": 0, "a": 0, "pm": 0, "sog": 0,
                "gwg": 0, "ppg": 0, "saves": 0, "sa": 0, "ga": 0, "w": 0, "l": 0, "otl": 0, "so": 0,
                "goalie_gp": 0, "best": None,
            })
            a["team_id"] = tid
            sa = _num(row, "shots_against")
            gp = _num(row, "gp")
            toi = _num(row, "toi_sec")
            if gp <= 0 and toi <= 0 and sa <= 0:
                continue
            a["gp"] += 1
            g_, a_ = int(_num(row, "g")), int(_num(row, "a"))
            a["g"] += g_
            a["a"] += a_
            a["pm"] += int(_num(row, "plus_minus"))
            a["sog"] += int(_num(row, "sog"))
            a["ppg"] += int(_num(row, "ppg"))
            if pid == gwg:
                a["gwg"] += 1
            opp = aid if tid == hid else hid
            if sa > 0:
                a["goalie_gp"] += 1
                a["saves"] += int(_num(row, "saves"))
                a["sa"] += int(sa)
                ga = int(_num(row, "ga"))
                a["ga"] += ga
                a["w"] += int(_num(row, "w"))
                a["l"] += int(_num(row, "l"))
                a["otl"] += int(_num(row, "otl"))
                a["so"] += int(_num(row, "so"))
                gscore = _num(row, "saves") * 0.16 + (2.2 if ga == 0 else 0.0) - ga * 0.7
                cand = {"score": gscore, "opp": opp, "iso": g.get("iso"), "saves": int(_num(row, "saves")), "ga": ga, "shutout": ga == 0}
            else:
                gscore = g_ * 3.0 + a_ * 1.9 + _num(row, "plus_minus") * 0.35 + _num(row, "sog") * 0.12
                cand = {"score": gscore, "opp": opp, "iso": g.get("iso"), "g": g_, "a": a_, "gwg": pid == gwg}
            if a["best"] is None or cand["score"] > a["best"]["score"]:
                a["best"] = cand
    return agg


def _star_score(a: Dict[str, Any]) -> float:
    if a["goalie_gp"] > 0 and a["sa"] > 0:
        sv = a["saves"] / a["sa"]
        return a["w"] * 2.4 + a["so"] * 3.0 + a["saves"] * 0.05 - a["ga"] * 0.6 + (sv - 0.905) * 60.0 * min(1.0, a["goalie_gp"] / 2.0)
    return a["g"] * 3.0 + a["a"] * 2.0 + a["gwg"] * 1.3 + a["pm"] * 0.3 + a["sog"] * 0.06


def _star_row(session: Any, a: Dict[str, Any], rank: int) -> Dict[str, Any]:
    pid = a["player_id"]
    name = _player_name(session, pid)
    pos = _player_pos(session, pid)
    tid = a["team_id"]
    goalie = a["goalie_gp"] > 0 and a["sa"] > 0
    best = a.get("best") or {}
    opp = _abbr(session, best.get("opp") or "") if best.get("opp") else ""
    when = _pretty_date(str(best.get("iso") or ""))
    last = name.split(" ")[-1]
    if goalie:
        sv = a["saves"] / a["sa"] if a["sa"] else None
        gaa_games = max(1, a["goalie_gp"])
        line = f"{a['w']}-{a['l']}-{a['otl']} · {_sv_str(sv)} SV%" + (f" · {_plural(a['so'], 'SO', 'SO')}" if a["so"] else "")
        bits = [f"{last} went {a['w']}-{a['l']}-{a['otl']} in {_plural(gaa_games, 'start')}, stopping {a['saves']} of {a['sa']} shots ({_sv_str(sv)})."]
        if a["so"] == 1 and best.get("shutout") and opp:
            bits.append(f"The shutout came against {opp} on {when}.")
        elif a["so"] > 1:
            bits.append(f"{_nw(a['so']).capitalize()} shutouts headline the week.")
        elif best.get("saves") and opp:
            bits.append(f"Best night: {best['saves']} saves against {opp} on {when}.")
        stats = [
            {"k": "W", "v": a["w"]}, {"k": "SV%", "v": _sv_str(sv)}, {"k": "SV", "v": a["saves"]},
            {"k": "GA", "v": a["ga"]}, {"k": "SO", "v": a["so"]},
        ]
    else:
        pts = a["g"] + a["a"]
        line = f"{a['g']}G {a['a']}A · {pts} PTS in {a['gp']} GP" + (f" · {a['pm']:+d}" if a["pm"] else "")
        lead = f"{last} put up {_plural(pts, 'point')} in {_plural(a['gp'], 'game')}"
        if a["g"] and a["a"]:
            lead += f" ({_plural(a['g'], 'goal')}, {_plural(a['a'], 'assist')})"
        elif a["g"]:
            lead += f", all of them goals"
        bits = [lead + "."]
        if a["gwg"]:
            bits.append("He scored the winner." if a["gwg"] == 1 else f"{_nw(a['gwg']).capitalize()} of those goals were game-winners.")
        if best.get("g", 0) >= 3 and opp:
            bits.append(f"The hat trick against {opp} on {when} was the highlight.")
        elif (best.get("g", 0) + best.get("a", 0)) >= 3 and opp:
            bits.append(f"His {best.get('g', 0)}G-{best.get('a', 0)}A night against {opp} on {when} stood out.")
        stats = [
            {"k": "G", "v": a["g"]}, {"k": "A", "v": a["a"]}, {"k": "PTS", "v": pts},
            {"k": "+/-", "v": f"{a['pm']:+d}"}, {"k": "GWG", "v": a["gwg"]},
        ]
    return {
        "rank": rank,
        "player_id": pid,
        "name": name,
        "position": "G" if goalie else pos,
        "team_id": tid,
        "team_abbr": _abbr(session, tid),
        "team_name": _team_name(session, tid),
        "gp": a["goalie_gp"] if goalie else a["gp"],
        "is_goalie": goalie,
        "line": line,
        "stats": stats,
        "blurb": " ".join(bits),
    }


def _three_stars(session: Any, games: List[Dict[str, Any]]) -> Dict[str, Any]:
    utid = str(getattr(session, "user_team_id", "") or "")
    if not games:
        return {"available": False, "note": "No regular-season games are in the book yet. The first three stars land after opening night."}
    last_day = int(games[-1].get("day") or 0)
    start_day = last_day - (WINDOW_DAYS - 1)
    week = [g for g in games if int(g.get("day") or 0) >= start_day]
    first_iso = next((str(g.get("iso") or "") for g in week if g.get("iso")), "")
    last_iso = str(week[-1].get("iso") or "") if week else ""

    def _top(agg: Dict[str, Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
        pool = [a for a in agg.values() if a["gp"] > 0]
        pool.sort(key=lambda a: -_star_score(a))
        return [_star_row(session, a, i + 1) for i, a in enumerate(pool[:n])]

    league = _top(_aggregate(week))
    club = _top(_aggregate(week, utid)) if utid else []
    # Last night: every game on the latest game day, scored one game at a time.
    night_games = [g for g in games if int(g.get("day") or 0) == last_day]
    night_rows: List[Dict[str, Any]] = []
    for g in night_games:
        agg = _aggregate([g])
        hid, aid = str(g.get("home") or ""), str(g.get("away") or "")
        score = f"{_abbr(session, aid)} {int(g.get('ag') or 0)} @ {_abbr(session, hid)} {int(g.get('hg') or 0)}" + (" (SO)" if g.get("so") else (" (OT)" if g.get("ot") else ""))
        for a in agg.values():
            if a["gp"] <= 0:
                continue
            s = _star_score(a) - (a["w"] * 1.2 if a["goalie_gp"] else 0)
            night_rows.append({"s": s, "a": a, "score": score})
    night_rows.sort(key=lambda r: -r["s"])
    night = []
    for i, r in enumerate(night_rows[:3]):
        row = _star_row(session, r["a"], i + 1)
        row["game"] = r["score"]
        night.append(row)
    rows = league + club + night
    _attach_faces(session, rows)
    return {
        "available": True,
        "window_days": WINDOW_DAYS,
        "window_label": f"{_pretty_date(first_iso)} – {_pretty_date(last_iso)}" if first_iso and last_iso and first_iso != last_iso else _pretty_date(last_iso),
        "games_in_window": len(week),
        "league": league,
        "club": club,
        "club_name": _team_name(session, utid),
        "last_night": {"iso": last_iso, "label": _pretty_date(last_iso), "games": len(night_games), "stars": night},
    }


# ---------------------------------------------------------------------------
# 2. Mailbag
# ---------------------------------------------------------------------------

_FIRST = (
    "Dave", "Marc", "Karen", "Priya", "Tom", "Lucie", "Jordan", "Steph", "Mike", "Aisha", "Brent", "Colleen",
    "Raj", "Nathalie", "Gord", "Erin", "Tyler", "Sandra", "Kevin", "Mei", "Darnell", "Rosie", "Pat", "Luc",
)
_LAST_INITIAL = "BCDFGHKLMNPRSTW"
_TOWNS_CA = (
    "Kanata, ON", "Moncton, NB", "Red Deer, AB", "Kamloops, BC", "Sudbury, ON", "Trois-Rivières, QC",
    "Brandon, MB", "Regina, SK", "Kingston, ON", "Saint John, NB", "Lethbridge, AB", "Nanaimo, BC",
)
_TOWNS_US = (
    "Duluth, MN", "Erie, PA", "Worcester, MA", "Grand Rapids, MI", "Fargo, ND", "Spokane, WA",
    "Toledo, OH", "Manchester, NH", "Rochester, NY", "Madison, WI", "Boise, ID", "Hershey, PA",
)


def _fan_voice(rng: random.Random, canadian: bool, used: set) -> Dict[str, str]:
    for _ in range(12):
        first = rng.choice(_FIRST)
        if first not in used:
            break
    used.add(first)
    pool = [t for t in (_TOWNS_CA if canadian else _TOWNS_US) if t not in used]
    town = rng.choice(pool or list(_TOWNS_CA if canadian else _TOWNS_US))
    used.add(town)
    return {"author": f"{first} {rng.choice(_LAST_INITIAL)}.", "town": town}


def _effects_preview(eff: Dict[str, float]) -> str:
    labels = {"fan_confidence": "Fan confidence", "media_pressure": "Media pressure"}
    parts = []
    for k in ("fan_confidence", "media_pressure"):
        v = eff.get(k)
        if v:
            parts.append(f"{labels[k]} {'+' if v > 0 else ''}{int(v) if float(v).is_integer() else v}")
    return " · ".join(parts) or "No change"


def _reply(rid: str, label: str, text: str, tone: str, fan: float, media: float) -> Dict[str, Any]:
    eff = {"fan_confidence": fan, "media_pressure": media}
    return {"id": rid, "label": label, "text": text, "tone": tone, "effects": eff, "effect_preview": _effects_preview(eff)}


def _user_goalie(session: Any, utid: str) -> Optional[Dict[str, Any]]:
    best = None
    for pid, st in (getattr(session, "player_season_stats", None) or {}).items():
        if str(st.get("team_id") or "") != utid:
            continue
        sa = float(st.get("shots_against") or st.get("goalie_shots_against") or 0)
        if sa <= 0:
            continue
        gp = int(st.get("gp") or 0)
        if best is None or gp > best["gp"]:
            sv = float(st.get("saves") or (sa - float(st.get("goalie_ga") or st.get("ga") or 0)))
            best = {"player_id": str(pid), "name": str(st.get("name") or "the starter"), "gp": gp, "sv": sv / sa if sa else None, "sa": sa}
    return best


def _league_sv(session: Any) -> Optional[float]:
    sv = sa = 0.0
    for st in (getattr(session, "player_season_stats", None) or {}).values():
        a = float(st.get("shots_against") or 0)
        if a > 0:
            sa += a
            sv += float(st.get("saves") or 0)
    return sv / sa if sa > 0 else None


def _slumping_star(session: Any, utid: str, games: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Top-paid/top-rated skater whose last five games trail his season pace badly."""
    tm = _team(session, utid)
    if tm is None:
        return None
    bucket = _season_bucket(session)
    pidx = bucket.get("players") or {}
    gmap = bucket.get("games") or {}
    fs = _fs()
    cands = []
    for pl in list(getattr(tm, "roster", None) or []):
        try:
            if fs._pos_str(pl) == "G":
                continue
            ovr = float(fs._trade_popup_ovr(pl) or 0)
        except Exception:
            continue
        aav = float((getattr(pl, "contract", None) or {}).get("aav_m") or getattr(pl, "aav_m", 0) or 0) if isinstance(getattr(pl, "contract", None), dict) else float(getattr(pl, "aav_m", 0) or 0)
        cands.append((ovr + aav * 1.5, pl, aav, ovr))
    cands.sort(key=lambda t: -t[0])
    out = None
    for _score, pl, aav, ovr in cands[:6]:
        pid = str(getattr(pl, "id", "") or "")
        gids = [g for g in (pidx.get(pid) or []) if str((gmap.get(g) or {}).get("scope") or "regular_season") == "regular_season"]
        if len(gids) < 6:
            continue
        def _pts(ids):
            p = g_ = 0
            for gid in ids:
                row = ((gmap.get(gid) or {}).get("lines") or {}).get(pid)
                if row:
                    g_ += int(_num(row, "g"))
                    p += int(_num(row, "g") + _num(row, "a"))
            return p, g_
        last = gids[-5:]
        lp, lg = _pts(last)
        sp, sg = _pts(gids)
        season_ppg = sp / len(gids)
        if season_ppg >= 0.55 and lp <= 1 or (lp == 0 and season_ppg >= 0.35):
            # goalless streak length
            drought = 0
            for gid in reversed(gids):
                row = ((gmap.get(gid) or {}).get("lines") or {}).get(pid)
                if row and int(_num(row, "g")) > 0:
                    break
                drought += 1
            out = {"player_id": pid, "name": fs._name_str(pl), "last_n": len(last), "last_pts": lp, "season_pts": sp,
                   "season_gp": len(gids), "drought": drought, "aav": round(aav, 2), "season_g": sg}
            break
    return out


def _hot_prospect(session: Any, utid: str) -> Optional[Dict[str, Any]]:
    ahl = getattr(session, "ahl_league", None)
    players = (ahl or {}).get("players") if isinstance(ahl, dict) else None
    if not isinstance(players, dict):
        return None
    utm = _team(session, utid)
    uabbr = _abbr(session, utid).upper()
    best = None
    for pid, r in players.items():
        if not isinstance(r, dict):
            continue
        if str(r.get("parent_abbr") or "").upper() != uabbr and str(r.get("team_id") or "") != utid:
            continue
        if int(r.get("gp") or 0) < 3 or int(r.get("age") or 99) > 24:
            continue
        if str(r.get("pos") or "") == "G":
            continue
        pts = int(r.get("points") or 0)
        ppg = pts / max(1, int(r.get("gp") or 1))
        if best is None or ppg > best["ppg"]:
            best = {"player_id": str(pid), "name": str(r.get("name") or "a prospect"), "gp": int(r.get("gp") or 0),
                    "g": int(r.get("goals") or 0), "a": int(r.get("assists") or 0), "pts": pts, "ppg": ppg,
                    "age": int(r.get("age") or 0), "affiliate": str(r.get("team") or "the farm club")}
    if best and best["pts"] >= 3 and best["ppg"] >= 0.6:
        return best
    _ = utm
    return None


def _user_trade_rumour(session: Any, utid: str) -> Optional[Dict[str, Any]]:
    for ev in reversed(list(getattr(session, "storyline_events", None) or [])):
        if not isinstance(ev, dict):
            continue
        if not _is_trade_rumour(ev):
            continue
        teams = {str(ev.get("team_id") or ""), str(ev.get("from_team_id") or ""), str(ev.get("to_team_id") or "")}
        if utid in teams and ev.get("player_name"):
            return ev
    return None


def _mailbag_store(session: Any) -> Dict[str, Any]:
    st = getattr(session, "bench_buzz_mailbag", None)
    if not isinstance(st, dict):
        st = {"answered": {}}
        try:
            session.bench_buzz_mailbag = st
        except Exception:
            pass
    st.setdefault("answered", {})
    return st


def _mailbag(session: Any, games: List[Dict[str, Any]], recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    fs = _fs()
    utid = str(getattr(session, "user_team_id", "") or "")
    tm = _team(session, utid)
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    week = cur // 7
    week_key = f"{season}w{week}"
    rng = random.Random(zlib.crc32(f"mailbag|{week_key}|{utid}".encode()))
    try:
        canadian = bool(fs._is_canadian_franchise_team(tm))
    except Exception:
        canadian = False
    nick = _nickname(session, utid)
    used: set = set()
    letters: List[Dict[str, Any]] = []

    def _add(topic: str, subject: str, body: str, facts: List[str], replies: List[Dict[str, Any]], player_id: str = "", player_name: str = "") -> None:
        v = _fan_voice(rng, canadian, used)
        letters.append({
            "id": f"{week_key}:{topic}",
            "topic": topic,
            "subject": subject,
            "body": body,
            "facts": facts,
            "replies": replies,
            "player_id": player_id,
            "player_name": player_name,
            **v,
        })

    rec = recs.get(utid) or {}
    gp = int(rec.get("gp") or 0)
    if gp > 0:
        streak = int(rec.get("streak") or 0)
        kind = rec.get("streak_kind")
        record = rec.get("record")
        pct = rec.get("pts_pct") or 0
        exp_pts = float(getattr(tm, "expected_points", 0) or 0)
        sched = int(getattr(session, "games_per_team_schedule", 82) or 82)
        exp_pct = exp_pts / (2 * sched) if exp_pts > 0 else None
        facts = [f"Record {record} ({rec.get('pts')} pts in {gp} GP)", f"Last 10: {rec.get('last10')}"]
        if streak >= 2:
            facts.append(f"{'Won' if kind == 'W' else 'Winless in'} {streak} straight")
        if kind == "L" and streak >= 3:
            body = (f"{_nw(streak).capitalize()} straight without a win and we're sitting at {record}. I've had season tickets through worse, "
                    f"but I need to hear that somebody upstairs has a plan before this turns into another lost November. What changes this week?")
            subj = f"{_nw(streak).capitalize()} in a row. Now what?"
            replies = [
                _reply("own", "Own it", "That's on me and the staff. We've looked hard at the details, and you'll see changes in how we play before you see changes in who plays.", "accountable", 3, -1),
                _reply("patience", "Ask for patience", "Losing streaks feel endless from the stands. The underlying game is closer than the record, and I'm not going to panic-trade our way out of a week.", "measured", 1, 0),
                _reply("shake", "Promise a shake-up", "Nobody's job is safe right now, including the comfortable ones. If it doesn't turn this week, moves are coming.", "fiery", 2, 2),
            ]
        elif kind == "W" and streak >= 3:
            body = (f"{_nw(streak).capitalize()} straight wins! I'm {record} happy. My buddy says it's a hot goalie and a soft schedule. "
                    f"Tell me he's wrong, and tell me we're not going to sit on our hands at the deadline with a team this good.")
            subj = f"{_nw(streak).capitalize()} straight. Is this real?"
            replies = [
                _reply("real", "It's real", "It's real. The way we're winning — structure, not luck — is what makes me believe it'll hold.", "confident", 3, 1),
                _reply("grounded", "Stay grounded", "Enjoy it, but it's a long year. We judge ourselves over 82, not over one good week.", "measured", 1, -1),
                _reply("add", "Hint at adding", "If the group keeps earning it, we'll look at adding. I'd never rule that out for a team that's competing.", "bold", 2, 2),
            ]
        else:
            ahead = exp_pct is not None and pct >= exp_pct + 0.05
            behind = exp_pct is not None and pct <= exp_pct - 0.05
            if behind:
                subj = f"{record} isn't what we were sold"
                pace = int(round(float(rec.get("pts") or 0) / gp * sched))
                body = (f"Preseason everyone had us around {int(round(exp_pts))} points. At {record} we're on pace for about {pace}. "
                        f"I'm not asking for miracles, just tell me whether this group is who we thought it was.")
            elif ahead:
                subj = f"Ahead of schedule at {record}?"
                body = (f"Nobody picked us to start {record}. The pundits had us around {int(round(exp_pts))} points and we're well ahead of that pace. "
                        f"Do we push our chips in, or is this the year we stay patient?")
            else:
                subj = f"{record} and I can't tell who we are"
                body = (f"We're {record}, last ten {rec.get('last10')}. Some nights we look like a playoff team, some nights we don't. "
                        f"What's the honest read from the man making the calls?")
            replies = [
                _reply("honest", "Give the honest read", "Honestly? We're in the mix and we're not finished. I like the effort; the consistency has to come.", "candid", 2, 0),
                _reply("trust", "Ask for trust", "I see the full picture every day and I like where it's heading. Give the plan a few more weeks.", "measured", 1, -1),
                _reply("demand", "Demand more", "I'm not satisfied either. Middle of the pack isn't the standard here and the players know it.", "fiery", 1, 2),
            ]
        _add("record", subj, body, facts, replies)

    slump = _slumping_star(session, utid, games)
    if slump:
        last = slump["name"].split(" ")[-1]
        facts = [f"{slump['last_pts']} pts in his last {slump['last_n']} games", f"{slump['season_pts']} pts in {slump['season_gp']} GP this season"]
        if slump["aav"]:
            facts.append(f"${slump['aav']:.2f}M cap hit")
        drought = slump["drought"]
        dtxt = f"no goals in {drought} straight" if drought >= 4 else f"{slump['last_pts']} points in his last {slump['last_n']}"
        money = f" For ${slump['aav']:.1f} million a year" if slump["aav"] >= 3 else ""
        _add(
            "slump",
            f"What's wrong with {last}?",
            f"{slump['name']} has {dtxt}.{money} I expect more than that. Is he hurt, is he unhappy, or is he just in his own head? Be straight with us.",
            facts,
            [
                _reply("back", "Back him", f"{last} is a big part of who we are. Slumps end, and his will. He has my full support.", "supportive", 1, -1),
                _reply("details", "Point to the details", f"The chances are there for {last}, the finish isn't yet. We're tweaking his linemates and his power-play spot.", "analytical", 2, 0),
                _reply("challenge", "Challenge him", f"{last} knows the standard. He's paid to produce and he'll be held to that like everyone else.", "tough", 2, 2),
            ],
            slump["player_id"],
            slump["name"],
        )

    goalie = _user_goalie(session, utid)
    lg_sv = _league_sv(session)
    if goalie and goalie["gp"] >= 3 and goalie["sv"] is not None and lg_sv is not None:
        gl = goalie["name"].split(" ")[-1]
        diff = goalie["sv"] - lg_sv
        facts = [f"{goalie['name']}: {_sv_str(goalie['sv'])} SV% in {goalie['gp']} GP", f"League average {_sv_str(lg_sv)}"]
        if diff <= -0.008:
            _add("goalie", f"Is {gl} the answer in net?",
                 f"{goalie['name']} is stopping {_sv_str(goalie['sv'])} of the shots he faces. The league average is {_sv_str(lg_sv)}. "
                 f"We can't keep chasing games from behind. Is it time to ride the backup?",
                 facts,
                 [
                     _reply("starter", "He's our starter", f"{gl} is our guy. Goalies run hot and cold, and the team in front of him owes him more.", "loyal", 0, -1),
                     _reply("compete", "Open the crease", "Starts are earned. If the backup gives us a better chance on a given night, he'll play.", "competitive", 2, 1),
                     _reply("defence", "Blame the structure", "Look at the chances we're giving up. That's a team problem before it's a goalie problem.", "analytical", 1, 0),
                 ], goalie["player_id"], goalie["name"])
        elif diff >= 0.008:
            _add("goalie", f"Pay {gl} whatever he wants",
                 f"{goalie['name']} at {_sv_str(goalie['sv'])} while the league sits at {_sv_str(lg_sv)}. He's stealing us games. "
                 f"Please tell me he isn't going to be overworked into the ground by February.",
                 facts,
                 [
                     _reply("praise", "Praise him", f"{gl} has been outstanding. We'll manage his workload so he's fresh when it matters.", "warm", 2, -1),
                     _reply("team", "Credit the team", "He's been great, and the guys in front of him are making his life easier too.", "measured", 1, 0),
                     _reply("contract", "Talk contract", f"Players like {gl} get taken care of here. That's all I'll say publicly.", "bold", 2, 1),
                 ], goalie["player_id"], goalie["name"])

    try:
        cap_space = float(getattr(tm, "cap_space_m", None) if getattr(tm, "cap_space_m", None) is not None else 0)
    except (TypeError, ValueError):
        cap_space = 0.0
    if tm is not None and getattr(tm, "cap_space_m", None) is not None:
        if cap_space >= 4.0:
            _add("cap", f"${cap_space:.1f}M sitting there?",
                 f"The cap sheet says we've got about ${cap_space:.1f} million in space. That's a real player. "
                 f"Why are we saving it — for a rainy day, or for a big swing?",
                 [f"Cap space ${cap_space:.2f}M"],
                 [
                     _reply("flex", "Keep the flexibility", "Space is a weapon at the deadline. When the right player is available, we'll be ready to use it.", "measured", 1, 0),
                     _reply("swing", "Promise to spend", "That money isn't there to sit. We're actively looking to turn it into help.", "bold", 2, 2),
                     _reply("future", "Point to the future", "Some of that is earmarked for our own guys' next contracts. Keeping the core together matters.", "prudent", 0, -1),
                 ])
        elif cap_space <= 1.0:
            _add("cap", "We're capped out — now what?",
                 f"We've got about ${max(0.0, cap_space):.1f} million in room. If someone gets hurt, can we even call anybody up? "
                 f"How did we paint ourselves into this corner?",
                 [f"Cap space ${cap_space:.2f}M"],
                 [
                     _reply("plan", "There's a plan", "Every dollar on that sheet was a choice. We've planned for injuries and we have options.", "confident", 1, 0),
                     _reply("fair", "Admit it's tight", "It's tight, no argument. It's the price of keeping a competitive group together.", "candid", 2, -1),
                     _reply("move", "Hint at a move", "Tight cap situations get solved one way or another. Stay tuned.", "bold", 1, 2),
                 ])

    prospect = _hot_prospect(session, utid)
    if prospect:
        pl = prospect["name"].split(" ")[-1]
        _add("prospect", f"Call up {pl}!",
             f"{prospect['name']} has {prospect['pts']} points ({prospect['g']}G, {prospect['a']}A) in {prospect['gp']} games with {prospect['affiliate']}, "
             f"and he's only {prospect['age']}. What's he still doing down there? Give the kid a look.",
             [f"{prospect['affiliate']}: {prospect['g']}G {prospect['a']}A in {prospect['gp']} GP", f"Age {prospect['age']}"],
             [
                 _reply("soon", "Soon", f"{pl}'s knocking on the door, and we hear him. When the opening's right, he'll get it.", "encouraging", 2, 1),
                 _reply("develop", "Protect his development", f"We won't rush {pl}. Big minutes in the AHL now beat eight minutes a night up here.", "patient", 1, -1),
                 _reply("earn", "Make him earn it", "Nobody gets promoted off a hot two weeks. Keep it up for a month and we'll talk.", "tough", 0, 0),
             ], prospect["player_id"], prospect["name"])

    rumour = _user_trade_rumour(session, utid)
    if rumour:
        pname = str(rumour.get("player_name") or "")
        age = _days_between(str(rumour.get("calendar_iso") or ""), _today_iso(session))
        selling = str(rumour.get("team_id") or rumour.get("from_team_id") or "") == utid
        partner = str(rumour.get("to_team_id") or "") if selling else str(rumour.get("team_id") or rumour.get("from_team_id") or "")
        pabbr = _abbr(session, partner) if partner else ""
        when = "this week" if age is not None and age <= 7 else "recently"
        if selling:
            body = (f"I read {when} that {pabbr or 'other teams'} {'have' if pabbr else 'are'} been asking about {pname}. "
                    f"He's one of the few guys who shows up every night. Please tell me we're not shopping him.")
            subj = f"Don't you dare trade {pname.split(' ')[-1]}"
        else:
            body = (f"The insiders have us linked to {pname}{' of ' + pabbr if pabbr else ''}. Is that real? "
                    f"Because I'd drive him to the rink myself.")
            subj = f"Are we really after {pname.split(' ')[-1]}?"
        _add("rumour", subj, body,
             [str(rumour.get("headline") or ""), f"Filed {_pretty_date(str(rumour.get('calendar_iso') or ''))}"],
             [
                 _reply("deny", "Shut it down", "I don't comment on rumours, but I'll say this: nothing is close and nobody should lose sleep.", "firm", 1, -1),
                 _reply("open", "Keep options open", "My job is to listen. Listening isn't the same as doing, and I'll only move if it makes us better.", "measured", 1, 1),
                 _reply("tease", "Tease it", "Part of the job is being aggressive. If something makes sense, we'll act.", "bold", 0, 2),
             ], str(rumour.get("player_id") or ""), pname)

    roster = list(getattr(tm, "roster", None) or []) if tm is not None else []
    stats = getattr(session, "player_season_stats", None) or {}

    def _aav(pl: Any) -> float:
        c = getattr(pl, "contract", None)
        try:
            return float((c or {}).get("aav_m") or getattr(pl, "aav_m", 0) or 0) if isinstance(c, dict) else float(getattr(pl, "aav_m", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    topics = {lt["topic"] for lt in letters}
    # Hot hand: club's leading scorer, when nobody is slumping.
    if gp > 0 and "slump" not in topics and len(letters) < 5:
        best = None
        for pid, st in stats.items():
            if str(st.get("team_id") or "") != utid or float(st.get("shots_against") or 0) > 0:
                continue
            pts = int(st.get("pts") or (int(st.get("g") or 0) + int(st.get("a") or 0)))
            if best is None or pts > best[1]:
                best = (str(pid), pts, st)
        if best and best[1] >= 3:
            pid, pts, st = best
            nm = str(st.get("name") or "")
            ln = nm.split(" ")[-1]
            _add("hot_hand", f"{ln} for the Hart?",
                 f"{nm} has {pts} points ({int(st.get('g') or 0)}G, {int(st.get('a') or 0)}A) in {int(st.get('gp') or 0)} games and leads the team. "
                 f"Do the coaches know they're allowed to give him more ice time? Because I'd play him 30 minutes a night.",
                 [f"{int(st.get('g') or 0)}G {int(st.get('a') or 0)}A in {int(st.get('gp') or 0)} GP", "Team scoring leader"],
                 [
                     _reply("praise", "Praise him", f"{ln} has been terrific. He sets the tone and the room follows.", "warm", 2, 0),
                     _reply("manage", "Manage his minutes", f"We want {ln} at his best in April, not just October. His minutes are where they should be.", "measured", 1, -1),
                     _reply("more", "Promise more", f"You're not wrong. {ln} will see more of the ice when games are on the line.", "bold", 2, 1),
                 ], pid, nm)
    # Big ticket: highest cap hit on the roster.
    if roster and len(letters) < 4:
        pl = max(roster, key=_aav)
        aav = _aav(pl)
        if aav >= 3.0:
            fs_ = _fs()
            nm = fs_._name_str(pl)
            ln = nm.split(" ")[-1]
            c = getattr(pl, "contract", None) or {}
            exp_year = c.get("expiry_year") if isinstance(c, dict) else None
            st = stats.get(str(getattr(pl, "id", "") or "")) or {}
            line = ""
            if int(st.get("gp") or 0) > 0:
                line = f" He has {int(st.get('pts') or 0)} points in {int(st.get('gp') or 0)} games so far."
            _add("big_ticket", f"Is {ln} worth ${aav:.1f}M?",
                 f"{nm} carries the biggest cap hit on the team at ${aav:.2f} million" + (f", signed through {exp_year}" if exp_year else "") + f".{line} "
                 f"I want to know what we're getting for that kind of money.",
                 [f"${aav:.2f}M cap hit"] + ([f"Through {exp_year}"] if exp_year else []),
                 [
                     _reply("worth", "Every penny", f"{ln} drives our team on and off the ice. You pay for that.", "confident", 1, 0),
                     _reply("more", "Expect more", f"We expect a lot from {ln}, and so does he. The standard comes with the contract.", "tough", 1, 1),
                     _reply("whole", "Look at the whole picture", "Cap hits don't win games, players do. Judge the team, not one line on a spreadsheet.", "measured", 0, -1),
                 ], str(getattr(pl, "id", "") or ""), nm)
    # Kid on the roster: youngest NHL regular.
    if roster and len(letters) < 4:
        def _age(p: Any) -> int:
            try:
                return int(getattr(getattr(p, "identity", None), "age", 99) or 99)
            except (TypeError, ValueError):
                return 99
        kid = min(roster, key=_age)
        if _age(kid) <= 22:
            nm = _fs()._name_str(kid)
            ln = nm.split(" ")[-1]
            st = stats.get(str(getattr(kid, "id", "") or "")) or {}
            line = f" He's got {int(st.get('pts') or 0)} points in {int(st.get('gp') or 0)} games." if int(st.get("gp") or 0) > 0 else ""
            _add("kid", f"Let {ln} play!",
                 f"{nm} is {_age(kid)} and the youngest guy on our roster.{line} I love watching him play. "
                 f"Please don't bury him on the fourth line to protect some veteran.",
                 [f"Age {_age(kid)}", "Youngest on the roster"] + ([f"{int(st.get('pts') or 0)} pts in {int(st.get('gp') or 0)} GP"] if int(st.get("gp") or 0) > 0 else []),
                 [
                     _reply("earned", "He'll earn it", f"{ln}'s ice time grows as fast as his game does. That's how it works here.", "measured", 1, 0),
                     _reply("believe", "Show belief", f"We think {ln} can be a big part of this. He'll get every chance.", "warm", 2, 1),
                     _reply("patience", "Ask for patience", f"Young players have ups and downs. Our job is to protect {ln} from too much, too soon.", "patient", 0, -1),
                 ], str(getattr(kid, "id", "") or ""), nm)

    if gp == 0:
        exp_pts = float(getattr(tm, "expected_points", 0) or 0) if tm is not None else 0
        _add("preseason", f"What's the target for the {nick}?",
             (f"The previews have us around {int(round(exp_pts))} points this year. " if exp_pts else "")
             + "Before the puck drops, I want to hear it from you: what counts as a good season?",
             ([f"Preseason projection {int(round(exp_pts))} pts"] if exp_pts else []),
             [
                 _reply("playoffs", "Playoffs or bust", "Playoffs. Anything less and we've fallen short.", "bold", 2, 2),
                 _reply("growth", "Growth", "Steps forward from our young core, and meaningful games in March.", "measured", 1, -1),
                 _reply("cup", "The Cup", "We're here to win the whole thing. That's always the target.", "fiery", 3, 3),
             ])

    letters = letters[:6]
    store = _mailbag_store(session)
    answered = store.get("answered") or {}
    for lt in letters:
        a = answered.get(lt["id"])
        lt["answered"] = dict(a) if isinstance(a, dict) else None
    faces = [lt for lt in letters if lt.get("player_id")]
    _attach_faces(session, faces)
    try:
        prof = fs._get_team_fan_profile(session, utid)
    except Exception:
        prof = {}
    history = [dict(v) for v in answered.values() if isinstance(v, dict) and v.get("week_key") != week_key]
    history.sort(key=lambda r: -int(r.get("day") or 0))
    return {
        "week_key": week_key,
        "week_label": f"Week of {_pretty_date(_today_iso(session))}" if _today_iso(session) else "This week",
        "team_name": _team_name(session, utid),
        "letters": letters,
        "fan_confidence": round(float(prof.get("fan_confidence", 0) or 0), 1) if prof else None,
        "fan_confidence_last_week": round(float(prof.get("fan_confidence_last_week", prof.get("fan_confidence", 0)) or 0), 1) if prof else None,
        "fan_trust_in_gm": round(float(prof.get("fan_trust_in_gm", 0) or 0), 1) if prof and prof.get("fan_trust_in_gm") is not None else None,
        "answered_count": sum(1 for lt in letters if lt.get("answered")),
        "history": history[:6],
        "effect_note": "Replies move fan confidence and media pressure through the same channel as Press Room answers, at story strength.",
    }


def answer_mailbag_letter(session: Any, letter_id: str, reply_id: str) -> Dict[str, Any]:
    """Apply one mailbag reply. Each letter can be answered once."""
    lid, rid = str(letter_id or ""), str(reply_id or "")
    store = _mailbag_store(session)
    if lid in (store.get("answered") or {}):
        raise ValueError("You already answered that letter.")
    games = _regular_games(session)
    recs = _team_records(session, games)
    bag = _mailbag(session, games, recs)
    letter = next((lt for lt in bag["letters"] if lt["id"] == lid), None)
    if letter is None:
        raise ValueError("That letter is no longer in this week's mailbag.")
    reply = next((r for r in letter["replies"] if r["id"] == rid), None)
    if reply is None:
        raise ValueError("Unknown reply.")
    utid = str(getattr(session, "user_team_id", "") or "")
    fs = _fs()
    before = None
    try:
        before = float(fs._get_team_fan_profile(session, utid).get("fan_confidence", 0) or 0)
    except Exception:
        before = None
    effects = {k: float(v) for k, v in (reply.get("effects") or {}).items() if v}
    if effects and utid:
        from app.sim_engine.franchise.storyline_engine import _apply_storyline_effects  # noqa: WPS433

        _apply_storyline_effects(session, utid, "", effects)
    after = None
    try:
        after = float(fs._get_team_fan_profile(session, utid).get("fan_confidence", 0) or 0)
    except Exception:
        after = None
    row = {
        "letter_id": lid,
        "week_key": bag["week_key"],
        "reply_id": rid,
        "label": reply["label"],
        "text": reply["text"],
        "tone": reply["tone"],
        "subject": letter["subject"],
        "author": letter["author"],
        "town": letter["town"],
        "day": int(getattr(session, "calendar_cursor", 0) or 0),
        "iso": _today_iso(session),
        "effect_preview": reply["effect_preview"],
        "fan_confidence_before": round(before, 1) if before is not None else None,
        "fan_confidence_after": round(after, 1) if after is not None else None,
    }
    answered = store["answered"]
    answered[lid] = row
    if len(answered) > MAILBAG_KEEP:
        for k in sorted(answered, key=lambda k: int((answered[k] or {}).get("day") or 0))[: len(answered) - MAILBAG_KEEP]:
            answered.pop(k, None)
    session.bench_buzz_mailbag = store
    try:
        session._bench_buzz_cache = None
    except Exception:
        pass
    return row


# ---------------------------------------------------------------------------
# 3. Hot Seat
# ---------------------------------------------------------------------------

def _seat_label(heat: float) -> str:
    if heat >= 75:
        return "Scorching"
    if heat >= 60:
        return "Hot"
    if heat >= 45:
        return "Warm"
    if heat >= 25:
        return "Comfortable"
    return "Ice cold"


def _hot_seat(session: Any, recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    utid = str(getattr(session, "user_team_id", "") or "")
    sched = int(getattr(session, "games_per_team_schedule", 82) or 82)
    org_all = getattr(session, "_conduct_org_pressure", None) or {}
    fan_profiles = getattr(session, "fan_profiles", None) or {}
    coaches: List[Dict[str, Any]] = []
    gms: List[Dict[str, Any]] = []
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid)
        rec = recs.get(tid) or {}
        gp = int(rec.get("gp") or 0)
        pct = rec.get("pts_pct")
        exp_pts = float(getattr(tm, "expected_points", 0) or 0)
        exp_pct = exp_pts / (2 * sched) if exp_pts > 0 else None
        sample = min(1.0, gp / 20.0)
        coach = getattr(tm, "coach", None)
        st = getattr(tm, "state", None)
        own = getattr(tm, "ownership", None)
        mkt = getattr(tm, "market", None)
        fan = fan_profiles.get(tid) if isinstance(fan_profiles.get(tid), dict) else {}
        org = org_all.get(tid) if isinstance(org_all.get(tid), dict) else {}
        reasons: List[Dict[str, Any]] = []

        def _r(text: str, delta: float) -> None:
            if abs(delta) >= 0.5:
                reasons.append({"text": text, "delta": round(delta, 1)})

        # --- coach ---
        js = getattr(coach, "job_security", None)
        heat = 0.0
        if js is not None:
            base = (1.0 - float(js)) * 40.0
            heat += base
            _r(f"Front-office security rating {int(round(float(js) * 100))}/100", base - 20.0)
        else:
            heat += 20.0
        perf = 0.0
        if pct is not None and exp_pct is not None and gp > 0:
            gap = float(pct) - exp_pct
            perf = max(-18.0, min(30.0, -gap * 140.0)) * sample
            heat += perf
            _r(f"{rec.get('record')} ({int(round(float(pct) * 100))}% of points) vs a {int(round(exp_pts))}-point projection", perf)
        streak, sk = int(rec.get("streak") or 0), rec.get("streak_kind")
        if sk == "L" and streak >= 3:
            d = min(16.0, 3.0 * streak)
            heat += d
            _r(f"{_nw(streak).capitalize()} straight without a win", d)
        elif sk == "W" and streak >= 3:
            d = -min(10.0, 2.0 * streak)
            heat += d
            _r(f"{_nw(streak).capitalize()}-game winning streak", d)
        if st is not None and getattr(st, "organizational_pressure", None) is not None:
            d = (float(st.organizational_pressure) - 0.5) * 30.0
            heat += d
            _r(f"Organizational pressure {int(round(float(st.organizational_pressure) * 100))}/100", d)
        if own is not None and getattr(own, "patience", None) is not None:
            d = (0.5 - float(own.patience)) * 12.0
            heat += d
            _r("Impatient ownership" if d > 0 else "Patient ownership", d)
        if fan:
            fc = float(fan.get("fan_confidence", 55) or 55)
            d = (55.0 - fc) * 0.25
            heat += d
            _r(f"Fan confidence {int(round(fc))}/100", d)
        if org.get("owner_confidence") is not None:
            oc = float(org["owner_confidence"])
            d = (0.6 - oc) * 25.0
            heat += d
            _r(f"Owner confidence {int(round(oc * 100))}/100", d)
        if coach is not None and bool(getattr(coach, "lost_room", False)):
            heat += 10.0
            _r("Has lost the room", 10.0)
        if coach is not None and bool(getattr(coach, "hot_seat", False)):
            heat += 8.0
            _r("Front office has him on notice", 8.0)
        yrs = getattr(coach, "contract_years_remaining", None) if coach is not None else None
        if yrs is not None and int(yrs) <= 1:
            heat += 4.0
            _r("Final year of his deal" if int(yrs) == 1 else "Working without a contract beyond this season", 4.0)
        heat = max(0.0, min(100.0, heat))
        reasons.sort(key=lambda r: -abs(r["delta"]))
        coaches.append({
            "team_id": tid,
            "team_abbr": _abbr(session, tid),
            "team_name": _team_name(session, tid),
            "name": str(getattr(coach, "name", "") or "Head coach") if coach is not None else "Head coach",
            "role": "Head coach",
            "is_user_team": tid == utid,
            "heat": round(heat, 1),
            "label": _seat_label(heat),
            "record": rec.get("record") or "0-0-0",
            "gp": gp,
            "pts_pct": pct,
            "expected_points": int(round(exp_pts)) if exp_pts else None,
            "streak": f"{sk}{streak}" if sk and streak else "",
            "last10": rec.get("last10") or "",
            "contract_years": int(yrs) if yrs is not None else None,
            "reasons": reasons[:5],
        })

        # --- GM ---
        g_reasons: List[Dict[str, Any]] = []

        def _g(text: str, delta: float) -> None:
            if abs(delta) >= 0.5:
                g_reasons.append({"text": text, "delta": round(delta, 1)})

        gh = 25.0
        if pct is not None and exp_pct is not None and gp > 0:
            gap = float(pct) - exp_pct
            d = max(-15.0, min(25.0, -gap * 110.0)) * sample
            gh += d
            _g(f"{rec.get('record')} against a {int(round(exp_pts))}-point projection", d)
        if own is not None:
            if getattr(own, "patience", None) is not None:
                d = (0.5 - float(own.patience)) * 16.0
                gh += d
                _g("Impatient ownership" if d > 0 else "Patient ownership", d)
            if getattr(own, "ambition", None) is not None and pct is not None and exp_pct is not None and float(pct) < exp_pct:
                d = (float(own.ambition) - 0.5) * 12.0 * sample
                gh += d
                _g("Ambitious owner expects a winner", d)
        if fan:
            tr = fan.get("fan_trust_in_gm")
            if tr is not None:
                d = (55.0 - float(tr)) * 0.35
                gh += d
                _g(f"Fan trust in the GM {int(round(float(tr)))}/100", d)
            heat_t = float(fan.get("recent_trade_heat", 0) or 0)
            if heat_t > 0:
                d = min(8.0, heat_t * 0.1)
                gh += d
                _g("Fans still angry about a recent trade", d)
        if org.get("owner_confidence") is not None:
            oc = float(org["owner_confidence"])
            d = (0.6 - oc) * 35.0
            gh += d
            _g(f"Owner confidence {int(round(oc * 100))}/100", d)
        if mkt is not None and getattr(mkt, "media_pressure", None) is not None:
            d = (float(mkt.media_pressure) - 0.5) * 8.0
            gh += d
            _g("Big, loud market" if d > 0 else "Quiet market", d)
        gh = max(0.0, min(100.0, gh))
        g_reasons.sort(key=lambda r: -abs(r["delta"]))
        gms.append({
            "team_id": tid,
            "team_abbr": _abbr(session, tid),
            "team_name": _team_name(session, tid),
            "name": "You" if tid == utid else str(getattr(tm, "gm_name", "") or "General manager"),
            "role": "General manager",
            "is_user_team": tid == utid,
            "heat": round(gh, 1),
            "label": _seat_label(gh),
            "record": rec.get("record") or "0-0-0",
            "gp": gp,
            "pts_pct": pct,
            "expected_points": int(round(exp_pts)) if exp_pts else None,
            "reasons": g_reasons[:5],
        })
    coaches.sort(key=lambda r: -r["heat"])
    gms.sort(key=lambda r: -r["heat"])
    for i, r in enumerate(coaches):
        r["rank"] = i + 1
    for i, r in enumerate(gms):
        r["rank"] = i + 1
    user_coach = next((r for r in coaches if r["is_user_team"]), None)
    user_gm = next((r for r in gms if r["is_user_team"]), None)
    min_gp = min((int(r.get("gp") or 0) for r in coaches), default=0)
    return {
        "coaches": coaches,
        "gms": gms,
        "user_coach": user_coach,
        "user_gm": user_gm,
        "early": min_gp < 10,
        "note": (
            "Heat blends each club's points pace against its preseason projection (weighted up as games pile up), "
            "the current streak, organizational and ownership pressure, fan confidence and each coach's standing with his front office."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Rumour Ledger
# ---------------------------------------------------------------------------

_TRADE_WORDS = ("trade talk", "trade chatter", "trade rumour", "trade rumor", "trade market", "trade bait", "trade request", "inquired", "listening on", "kicking tires", "drawing pre-deadline", "open to moving", "shopping", "on the block", "interested clubs")
_CONTRACT_WORDS = ("next contract", "extension", "contract talks", "camp is restless", "holdout")


def _is_trade_rumour(ev: Dict[str, Any]) -> bool:
    cat = str(ev.get("category") or "").lower()
    cause = str(ev.get("cause_type") or "").upper()
    if cat == "trade_rumor" or cause == "TRADE_RUMOR":
        return True
    level = str(ev.get("public_knowledge_level") or ev.get("knowledge_type") or "").lower()
    if level in ("rumor", "rumour", "claim", "speculation", "chatter"):
        hl = str(ev.get("headline") or "").lower()
        return any(w in hl for w in _TRADE_WORDS)
    return False


def _rumour_kind(ev: Dict[str, Any]) -> str:
    if _is_trade_rumour(ev):
        return "trade"
    level = str(ev.get("public_knowledge_level") or ev.get("knowledge_type") or "").lower()
    hl = str(ev.get("headline") or "").lower()
    if level in ("rumor", "rumour", "claim", "speculation", "chatter") and any(w in hl for w in _CONTRACT_WORDS):
        return "contract"
    return ""


def _ledger_store(session: Any) -> Dict[str, Any]:
    st = getattr(session, "bench_buzz_rumour_ledger", None)
    if not isinstance(st, dict):
        st = {"items": {}}
        try:
            session.bench_buzz_rumour_ledger = st
        except Exception:
            pass
    st.setdefault("items", {})
    return st


def _contract_snap(pl: Any) -> Dict[str, Any]:
    c = getattr(pl, "contract", None) if pl is not None else None
    if not isinstance(c, dict):
        return {}
    return {"expiry_year": c.get("expiry_year"), "years_remaining": c.get("years_remaining"), "aav_m": c.get("aav_m")}


def _trade_dates(session: Any) -> Dict[str, Tuple[str, str]]:
    """player_id -> (iso, destination team) from completed trade beats on the feed."""
    out: Dict[str, Tuple[str, str]] = {}
    for ev in list(getattr(session, "storyline_events", None) or []):
        if not isinstance(ev, dict) or str(ev.get("type") or "") != "trade" or _is_trade_rumour(ev):
            continue
        iso = str(ev.get("calendar_iso") or "")
        for side in list(ev.get("teams") or []):
            if not isinstance(side, dict):
                continue
            for asset in list(side.get("acquired_assets") or []):
                pid = str((asset or {}).get("player_id") or "")
                if pid and pid not in out:
                    out[pid] = (iso, str(side.get("team_id") or ""))
    return out


def _rumour_ledger(session: Any) -> Dict[str, Any]:
    store = _ledger_store(session)
    items: Dict[str, Dict[str, Any]] = store["items"]
    today = _today_iso(session)
    roster = _roster_index(session)
    # 1) capture new rumours from the live feed + insider knowledge graph
    sources = list(getattr(session, "storyline_events", None) or []) + list(getattr(session, "knowledge_graph", None) or [])
    for ev in sources:
        if not isinstance(ev, dict):
            continue
        kind = _rumour_kind(ev)
        if not kind:
            continue
        pid = str(ev.get("player_id") or "")
        tid = str(ev.get("from_team_id") or ev.get("team_id") or "")
        if not pid or not tid:
            continue
        key = str(ev.get("storyline_id") or ev.get("id") or ev.get("world_event_id") or "") or f"{pid}|{ev.get('calendar_iso')}"
        if key in items:
            continue
        # one open rumour per player+kind is enough; repeats just add a mention
        dup = next((it for it in items.values() if it.get("player_id") == pid and it.get("kind") == kind and it.get("verdict") == "pending"), None)
        if dup is not None:
            dup["mentions"] = int(dup.get("mentions") or 1) + 1
            items[key] = {"alias_of": dup["key"]}
            continue
        pl = roster.get(pid, (None, None))[1]
        reporter = str(ev.get("reporter_name") or "").strip() or str(ev.get("source_label") or "").strip() or "League insider wire"
        items[key] = {
            "key": key,
            "kind": kind,
            "headline": str(ev.get("headline") or ""),
            "player_id": pid,
            "player_name": str(ev.get("player_name") or ""),
            "team_id": tid,
            "partner_team_id": str(ev.get("to_team_id") or ""),
            "reporter": reporter,
            "outlet": str(ev.get("outlet_name") or ""),
            "iso": str(ev.get("calendar_iso") or today),
            "contract": _contract_snap(pl),
            "verdict": "pending",
            "mentions": 1,
        }
    # 2) resolve pending rumours against the save
    trade_dates = _trade_dates(session)
    deadline_idx = None
    try:
        cache = getattr(session, "_trade_deadline_idx_cache", None)
        if isinstance(cache, tuple) and cache:
            deadline_idx = int(cache[0])
    except Exception:
        deadline_idx = None
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    past_deadline = deadline_idx is not None and cur > deadline_idx and str(getattr(session, "phase", "")) == "regular"
    season_now = int(getattr(session, "season_calendar_year", 0) or 0)
    for it in items.values():
        if it.get("alias_of") or it.get("verdict") != "pending":
            continue
        pid = it["player_id"]
        now_tid, pl = roster.get(pid, ("", None))
        age = _days_between(it.get("iso") or "", today) or 0
        moved = bool(now_tid) and now_tid != it["team_id"]
        if it["kind"] == "trade":
            if moved:
                t_iso, dest = trade_dates.get(pid, ("", now_tid))
                dest = dest or now_tid
                it.update(verdict="hit", resolved_iso=t_iso or today, resolved_team_id=dest)
                partner = it.get("partner_team_id") or ""
                it["outcome"] = f"Traded to {_abbr(session, dest)}" + (" — the club named in the report" if partner and partner == dest else "")
            elif now_tid and (age >= RUMOUR_MISS_DAYS or past_deadline):
                it.update(verdict="miss", resolved_iso=today, outcome=(f"Still with {_abbr(session, now_tid)} after the deadline" if past_deadline else f"Still with {_abbr(session, now_tid)} {age} days later"))
        elif it["kind"] == "contract":
            snap = it.get("contract") or {}
            nowc = _contract_snap(pl)
            extended = bool(snap) and bool(nowc) and (nowc.get("expiry_year") or 0) > (snap.get("expiry_year") or 0)
            if moved:
                it.update(verdict="hit", resolved_iso=trade_dates.get(pid, (today, ""))[0] or today, resolved_team_id=now_tid,
                          outcome=f"The unrest ended with a move to {_abbr(session, now_tid)}")
            elif extended:
                it.update(verdict="settled", resolved_iso=today,
                          outcome=f"Signed through {nowc.get('expiry_year')} — the camp got its deal")
            elif now_tid and age >= 45:
                it.update(verdict="miss", resolved_iso=today, outcome=f"Nothing came of it {age} days later")
        if it.get("verdict") == "pending" and not now_tid and age >= 3:
            it["note"] = "Player not on an NHL or AHL roster right now"
    _ = season_now
    # trim very old resolved rows
    real = [it for it in items.values() if not it.get("alias_of")]
    if len(real) > 400:
        real.sort(key=lambda it: str(it.get("iso") or ""))
        for it in real[: len(real) - 400]:
            items.pop(it["key"], None)
    store["items"] = items
    real = [dict(it) for it in items.values() if not it.get("alias_of")]
    for it in real:
        it["team_abbr"] = _abbr(session, it.get("team_id") or "")
        if it.get("partner_team_id"):
            it["partner_abbr"] = _abbr(session, it["partner_team_id"])
        if it.get("resolved_team_id"):
            it["resolved_abbr"] = _abbr(session, it["resolved_team_id"])
        it["age_days"] = _days_between(it.get("iso") or "", today)
        it["filed_label"] = _pretty_date(it.get("iso") or "")
        it["resolved_label"] = _pretty_date(it.get("resolved_iso") or "")
        it.pop("contract", None)
    resolved = [it for it in real if it["verdict"] in ("hit", "miss", "settled")]
    resolved.sort(key=lambda it: str(it.get("resolved_iso") or ""), reverse=True)
    pending = [it for it in real if it["verdict"] == "pending"]
    pending.sort(key=lambda it: str(it.get("iso") or ""), reverse=True)
    reps: Dict[str, Dict[str, Any]] = {}
    for it in real:
        r = reps.setdefault(it["reporter"], {"reporter": it["reporter"], "outlet": it.get("outlet") or "", "filed": 0, "hits": 0, "misses": 0, "settled": 0, "pending": 0})
        if it.get("outlet") and not r["outlet"]:
            r["outlet"] = it["outlet"]
        r["filed"] += 1
        v = it["verdict"]
        r[{"hit": "hits", "miss": "misses", "settled": "settled", "pending": "pending"}[v]] += 1
    board = list(reps.values())
    for r in board:
        dec = r["hits"] + r["misses"]
        r["hit_rate"] = round(r["hits"] / dec, 3) if dec else None
        r["decided"] = dec
    board.sort(key=lambda r: (-(r["decided"] > 0), -(r["hit_rate"] or 0), -r["decided"], -r["filed"]))
    decided_total = sum(1 for it in resolved if it["verdict"] in ("hit", "miss"))
    resolved_faces = resolved[:30]
    pending_faces = pending[:24]
    _attach_faces(session, resolved_faces + pending_faces)
    if decided_total < 3:
        note = (f"Only {decided_total} rumour{'s have' if decided_total != 1 else ' has'} played out so far, too few to rate anyone. "
                f"Trade talk counts as a miss after {RUMOUR_MISS_DAYS} days with no deal (or once the deadline passes). Contract-unrest reports wait 45 days.")
    else:
        note = (f"{decided_total} rumours have played out. Trade talk counts as a miss after {RUMOUR_MISS_DAYS} days with no deal or once the deadline passes. "
                "An extension after contract-unrest talk is marked Settled and doesn't count either way. Locker-room whispers can't be checked against the record, so they aren't scored.")
    return {
        "resolved": resolved_faces,
        "pending": pending_faces,
        "reporters": board,
        "tracked": len(real),
        "decided": decided_total,
        "hits": sum(1 for it in resolved if it["verdict"] == "hit"),
        "misses": sum(1 for it in resolved if it["verdict"] == "miss"),
        "enough": decided_total >= 3,
        "note": note,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_bench_buzz_features(session: Any) -> Dict[str, Any]:
    """Cheap per-day build, cached on the session."""
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    store = getattr(session, "bench_buzz_mailbag", None)
    answered_n = len((store or {}).get("answered") or {}) if isinstance(store, dict) else 0
    key = (
        FEATURES_VERSION,
        int(getattr(session, "season_calendar_year", 0) or 0),
        str(getattr(session, "phase", "") or ""),
        cur,
        len(getattr(session, "game_results", None) or []),
        len(getattr(session, "storyline_events", None) or []),
        answered_n,
    )
    cache = getattr(session, "_bench_buzz_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key and isinstance(cache.get("payload"), dict):
        return cache["payload"]
    out: Dict[str, Any] = {"version": FEATURES_VERSION, "as_of_iso": _today_iso(session), "calendar_cursor": cur}
    games = _regular_games(session)
    recs = _team_records(session, games)
    for name, fn in (
        ("three_stars", lambda: _three_stars(session, games)),
        ("mailbag", lambda: _mailbag(session, games, recs)),
        ("hot_seat", lambda: _hot_seat(session, recs)),
        ("rumour_ledger", lambda: _rumour_ledger(session)),
    ):
        try:
            out[name] = fn()
        except Exception as e:  # one broken desk must not blank the others
            _log.warning("bench buzz %s failed: %s", name, e, exc_info=True)
            out[name] = {"error": True}
    try:
        session._bench_buzz_cache = {"key": key, "payload": out}
    except Exception:
        pass
    return out
