"""Generic team/game planner: turns a question into a plan and runs it on the game log.

Instead of one pattern per question, a team question is read as five parts:

  subject   which club(s): yours, a named club, or "which team" (rank all 32)
  metric    record, wins, losses, points, goals for/against, goal differential,
            per-game rates, shutouts, one-goal games, overtime, power play, kill
  window    whole season, last N games, a month, since a month, after the deadline
  split     home / road, against one opponent, playoffs
  op        describe one club, or rank every club

Everything is computed from this save's game results (and the legacy archive for
multi-season head-to-head), so "how did we do against Boston in our last 10 road
games" and "which team allows the fewest goals since January" need no new code.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .answers import _answer, _team_label
from .data import LiveView

Answer = Dict[str, Any]

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], start=1)}
for _m, _i in list(_MONTHS.items()):
    _MONTHS[_m[:3]] = _i

# metric -> (label, higher_is_better, phrases)
METRICS: List[Tuple[str, str, bool, str]] = [
    ("gd", "Goal differential", True, r"goal (diff|differential)|\+/- as a team|outscor"),
    ("ga_pg", "Goals against per game", False, r"goals? against (per|a) game|gaa as a team|allow(s|ed)? per game"),
    ("gf_pg", "Goals for per game", True, r"goals? (for )?(per|a) game|scor(e|es|ing) per game"),
    ("ga", "Goals against", False, r"goals? against|allow(s|ed|ing)?\b|give(s)? up|conced|keep(s)? the puck out|best defen[cs]e\b|worst defen[cs]e\b"),
    ("gf", "Goals for", True, r"goals? for|goals? scored|score(s|d)? the most|scor(e|es|ing) the most|best offen[cs]e\b|worst offen[cs]e\b|most goals|fewest goals"),
    ("so", "Shutouts", True, r"shutouts?|blank(ed)?"),
    ("one_goal", "One-goal games", True, r"one[- ]goal games?|close games?"),
    ("ot", "Overtime / shootout games", True, r"overtime|\bot\b|shootouts?|extra time"),
    ("pp", "Power play %", True, r"power ?play|\bpp\b"),
    ("pk", "Penalty kill %", True, r"penalty ?kill|\bpk\b"),
    ("l", "Losses", False, r"\bloss(es)?\b|\blost\b|\blose\b"),
    ("w", "Wins", True, r"\bwins?\b|\bwon\b"),
    ("pts", "Points", True, r"\bpoints?\b"),
    ("record", "Record", True, r"\brecord\b|how (did|do|have|has|are|were|is) .*\b(do|done|doing|play|played|playing|fare|fared|faring|look)\b|\bperform|\bbeat(en|s)?\b|\bhow (good|bad|well|poorly|strong|weak) (are|were|have|is) (we|they|the [a-z]+)\b|\bhow (are|were) (we|they) (at home|on the road)\b"),
]

_TEAM_WORDS = r"\b(we|us|our|ours|team|teams|club|clubs|franchise|they|them|their)\b"
_RANK_WORDS = r"\b(which|what) (nhl )?(team|club)s?\b|\bwho (has|have|had|allows|allowed|gives|scores|scored|is|was)\b.*\b(team|club)s?\b|\b(most|least|fewest|best|worst|highest|lowest|top|bottom)\b.*\b(team|club)s?\b|\b(team|club)s? with\b|\brank\b.*\bteams?\b"

_CACHE: Dict[str, Any] = {}


# --------------------------------------------------------------------------- parsing
def _window(q: str, session: Any) -> Dict[str, Any]:
    w: Dict[str, Any] = {}
    m = re.search(r"\b(?:last|past|previous|recent) (\d+|" + "|".join(_NUM) + r") (?:(?:home|road|away|regular[- ]season|playoff) )?(?:games?|outings|contests|matches)\b", q) \
        or re.search(r"\b(?:last|past) (\d+|" + "|".join(_NUM) + r")\b(?! (?:years?|seasons?|days?|weeks?|months?|picks?|drafts?))", q)
    if m:
        w["last_n"] = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
    if re.search(r"\blately\b|\brecently\b|\bthese days\b", q) and "last_n" not in w:
        w["last_n"] = 10
    m = re.search(r"\b(since|after|from) (" + "|".join(_MONTHS) + r")\b", q)
    if m:
        w["since_month"] = _MONTHS[m.group(2)]
    else:
        m = re.search(r"\b(in|during|through|for) (" + "|".join(_MONTHS) + r")\b", q)
        if m:
            w["month"] = _MONTHS[m.group(2)]
    if re.search(r"\b(since|after) the (trade )?deadline\b|post[- ]deadline", q):
        w["after_deadline"] = True
    elif re.search(r"\bbefore the (trade )?deadline\b|pre[- ]deadline", q):
        w["before_deadline"] = True
    if re.search(r"\b(at home|home games?|home ice|in our barn)\b", q):
        w["venue"] = "home"
    elif re.search(r"\b(on the road|road games?|away games?|away from home|\baway\b)", q):
        w["venue"] = "road"
    if re.search(r"\bplayoffs?\b|\bpostseason\b", q) and not re.search(r"\b(make|making|made|miss|missing|clinch)\b", q):
        w["playoffs"] = True
    if re.search(r"\b(when|if) (we|they|the [a-z]+) score(s|d)? first\b|\bscoring first\b|\bscored first\b", q):
        w["first"] = "for"
    elif re.search(r"\b(when|if) (we|they) (allow|give up|concede)(s|ed)? (the )?first\b|\b(trail|trailing|down) (first|early)\b|\bopponent scores first\b", q):
        w["first"] = "against"
    m = re.search(r"\b(?:last|past) (\d+|" + "|".join(_NUM) + r") (?:years|seasons)\b|\ball[- ]time\b|\bever\b|\bhistory\b", q)
    if m:
        w["seasons"] = (int(m.group(1)) if m.group(1) and m.group(1).isdigit() else _NUM.get(m.group(1) or "", 99)) if m.lastindex else 99
    return w


def _metric(q: str) -> Optional[Tuple[str, str, bool]]:
    for key, label, higher, pat in METRICS:
        if re.search(pat, q):
            return key, label, higher
    return None


def plan(q: str, slots: Dict[str, Any], view: LiveView) -> Optional[Dict[str, Any]]:
    """Return a plan for a team/game question, or None when it isn't one."""
    q = " ".join(str(q or "").lower().replace("’", "'").split())
    club_q = bool(re.search(r"\b(what|which) (nhl )?(team|club)s?\b|\b(team|club)s? (gives?|allows?|scores?|has|have)\b", q))
    if slots.get("players") and not club_q:
        return None  # a named player makes it a player question (unless it's plainly "which team ...")
    from .answers5 import match as game_record_match

    if game_record_match(q):
        return None  # blown leads, comebacks, blowouts: the game-records layer owns these
    if re.search(r"\b(player|players|defenseman|defensemen|defenceman|forwards?|wingers?|cent(er|re)s?|goalies?|goaltenders?|skaters?|rookies?|scorers?|guy|guys|prospects?)\b", q) \
            and not re.search(r"\b(team|teams|club|clubs)\b", q):
        return None  # a question about players, not clubs
    met = _metric(q)
    if not met:
        return None
    teams = [str(t) for t in (slots.get("teams") or [])]
    self_ref = bool(re.search(r"\b(we|us|our|ours)\b", q)) or bool(slots.get("self"))
    team_ctx = bool(re.search(_TEAM_WORDS, q)) or bool(teams)
    if not team_ctx:
        return None
    rank = bool(re.search(_RANK_WORDS, q)) and not (self_ref and not re.search(r"\b(which|what) (team|club)", q))
    focus: Optional[str] = None
    opp: Optional[str] = None
    if self_ref:
        focus = view.user_tid
        opp = next((t for t in teams if t != view.user_tid), None)
    elif teams:
        focus = teams[0]
        opp = teams[1] if len(teams) > 1 else None
        if rank and len(teams) == 1 and re.search(r"\b(against|vs\.?|versus|facing)\b", q):
            focus, opp = None, teams[0]
    if rank and not self_ref and len(teams) <= 1 and not re.search(r"\b(against|vs\.?|versus)\b", q):
        focus = None
    if not rank and focus is None:
        focus = view.user_tid
    low = bool(re.search(r"\b(least|fewest|worst|lowest|bottom|weakest)\b", q))
    # "worst defence" ranks by goals against ascending-bad; flip direction for lower-is-better metrics.
    key, label, higher = met
    if key == "record" and rank:
        key, label, higher = "pts_pct", "Points %", True
    return {"metric": key, "label": label, "higher": higher, "low": low, "rank": rank and focus is None,
            "focus": focus, "opp": opp, "window": _window(q, view.session), "q": q}


# --------------------------------------------------------------------------- data
def _deadline_iso(session: Any) -> str:
    for attr in ("trade_deadline_iso", "trade_deadline_date", "_trade_deadline_iso"):
        v = str(getattr(session, attr, "") or "")
        if len(v) >= 10:
            return v[:10]
    year = int(getattr(session, "season_calendar_year", 0) or 0)
    return f"{year + 1 if year else 2027}-03-06"


def _games(view: LiveView) -> List[Dict[str, Any]]:
    results = list(getattr(view.session, "game_results", None) or [])
    key = f"{id(view.session)}:{len(results)}"
    if _CACHE.get("key") == key:
        return _CACHE["games"]
    out = []
    for b in results:
        if not isinstance(b, dict):
            continue
        h, a = str(b.get("home_id") or ""), str(b.get("away_id") or "")
        if not h or not a:
            continue
        hg = int(b.get("home_goals", b.get("home_score")) or 0)
        ag = int(b.get("away_goals", b.get("away_score")) or 0)
        so_win = str(b.get("shootout_winner_id") or b.get("winner_id") or "")
        first = str(b.get("first_goal_team_id") or "")
        if not first:
            # Earliest period with goals tells who opened the scoring when only one side scored in it.
            for i in (1, 2, 3):
                hp, apd = b.get(f"home_p{i}"), b.get(f"away_p{i}")
                if hp is None or apd is None:
                    break
                if int(hp or 0) and not int(apd or 0):
                    first = h
                    break
                if int(apd or 0) and not int(hp or 0):
                    first = a
                    break
                if int(hp or 0) and int(apd or 0):
                    break
        home_won = hg > ag or (hg == ag and so_win == h)
        out.append({"date": str(b.get("calendar_iso") or b.get("iso") or "")[:10], "day": int(b.get("day") or 0),
                    "h": h, "a": a, "hg": hg, "ag": ag, "home_won": home_won,
                    "extra": bool(b.get("overtime") or b.get("ot") or b.get("shootout")),
                    "first": first,
                    "playoff": bool(b.get("is_playoff")) or str(b.get("stat_scope") or "") == "playoffs"})
    out.sort(key=lambda g: (g["date"], g["day"]))
    _CACHE.update(key=key, games=out)
    return out


def _team_view(g: Dict[str, Any], tid: str) -> Dict[str, Any]:
    home = g["h"] == tid
    gf, ga = (g["hg"], g["ag"]) if home else (g["ag"], g["hg"])
    won = g["home_won"] if home else not g["home_won"]
    return {**g, "home": home, "opp": g["a"] if home else g["h"], "gf": gf, "ga": ga, "won": won}


def _select(view: LiveView, tid: str, w: Dict[str, Any], opp: Optional[str]) -> List[Dict[str, Any]]:
    rows = [_team_view(g, tid) for g in _games(view) if tid in (g["h"], g["a"])]
    rows = [g for g in rows if g["playoff"] == bool(w.get("playoffs"))]
    if opp:
        rows = [g for g in rows if g["opp"] == opp]
    if w.get("venue") == "home":
        rows = [g for g in rows if g["home"]]
    elif w.get("venue") == "road":
        rows = [g for g in rows if not g["home"]]
    if w.get("month"):
        rows = [g for g in rows if g["date"][5:7].isdigit() and int(g["date"][5:7]) == w["month"]]
    if w.get("since_month"):
        m0 = w["since_month"]
        # Season runs Oct -> Jun: months before October belong to the second calendar year.
        order = lambda m: m + 12 if m < 8 else m  # noqa: E731
        rows = [g for g in rows if g["date"][5:7].isdigit() and order(int(g["date"][5:7])) >= order(m0)]
    if w.get("after_deadline") or w.get("before_deadline"):
        dl = _deadline_iso(view.session)
        rows = [g for g in rows if (g["date"] > dl) == bool(w.get("after_deadline"))]
    if w.get("first"):
        mine = w["first"] == "for"
        rows = [g for g in rows if g.get("first") and ((g["first"] == tid) == mine)]
    if w.get("last_n"):
        rows = rows[-int(w["last_n"]):]
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gp = len(rows)
    w = sum(1 for g in rows if g["won"])
    otl = sum(1 for g in rows if not g["won"] and g["extra"])
    l = gp - w - otl
    gf = sum(g["gf"] for g in rows)
    ga = sum(g["ga"] for g in rows)
    one = [g for g in rows if abs(g["gf"] - g["ga"]) == 1 or (g["extra"] and g["gf"] == g["ga"])]
    ot = [g for g in rows if g["extra"]]
    return {"gp": gp, "w": w, "l": l, "otl": otl, "pts": 2 * w + otl, "pts_pct": (2 * w + otl) / (2 * gp) if gp else 0.0,
            "gf": gf, "ga": ga, "gd": gf - ga, "gf_pg": gf / gp if gp else 0.0, "ga_pg": ga / gp if gp else 0.0,
            "so": sum(1 for g in rows if g["ga"] == 0 and g["won"]),
            "one_goal": f"{sum(1 for g in one if g['won'])}-{len(one) - sum(1 for g in one if g['won'])}",
            "one_goal_w": sum(1 for g in one if g["won"]), "one_goal_n": len(one),
            "ot": f"{sum(1 for g in ot if g['won'])}-{len(ot) - sum(1 for g in ot if g['won'])}",
            "ot_w": sum(1 for g in ot if g["won"]), "ot_n": len(ot)}


def _rec(s: Dict[str, Any]) -> str:
    return f"{s['w']}-{s['l']}-{s['otl']}"


def _window_text(p: Dict[str, Any], view: LiveView) -> str:
    w = p["window"]
    bits = []
    if w.get("last_n"):
        bits.append(f"over the last {w['last_n']} games")
    if w.get("month"):
        bits.append(f"in {[k for k, v in _MONTHS.items() if v == w['month'] and len(k) > 3][0].title()}")
    if w.get("since_month"):
        bits.append(f"since {[k for k, v in _MONTHS.items() if v == w['since_month'] and len(k) > 3][0].title()}")
    if w.get("after_deadline"):
        bits.append("since the deadline")
    if w.get("before_deadline"):
        bits.append("before the deadline")
    if w.get("venue"):
        bits.append("at home" if w["venue"] == "home" else "on the road")
    if p.get("opp"):
        bits.append(f"against the {_team_label(view, p['opp'])}")
    if w.get("playoffs"):
        bits.append("in the playoffs")
    if w.get("first"):
        bits.append("when scoring first" if w["first"] == "for" else "when the other team scores first")
    return ", ".join(bits) if bits else ("in the playoffs" if w.get("playoffs") else "this season")


def _special(view: LiveView, tid: str, key: str) -> Optional[float]:
    for r in view.standings():
        if str(r.get("team_id") or r.get("id") or "") == str(tid):
            v = r.get("pp_pct" if key == "pp" else "pk_pct")
            return float(v) if v is not None else None
    return None


def _value(view: LiveView, tid: str, s: Dict[str, Any], key: str) -> Optional[float]:
    if key in ("pp", "pk"):
        return _special(view, tid, key)
    if key == "one_goal":
        return s["one_goal_w"] / s["one_goal_n"] if s["one_goal_n"] else None
    if key == "ot":
        return s["ot_w"] / s["ot_n"] if s["ot_n"] else None
    if key == "record":
        return s["pts_pct"]
    v = s.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _fmt(key: str, v: Optional[float], s: Optional[Dict[str, Any]] = None) -> str:
    if v is None:
        return "—"
    if key in ("pp", "pk"):
        return f"{v * 100:.1f}%"
    if key in ("gf_pg", "ga_pg"):
        return f"{v:.2f}"
    if key in ("pts_pct", "record"):
        return f".{int(round(v * 1000)):03d}"
    if key == "one_goal" and s:
        return s["one_goal"]
    if key == "ot" and s:
        return s["ot"]
    if key == "gd":
        return f"{int(v):+d}"
    return str(int(round(v)))


# --------------------------------------------------------------------------- answers
def run(p: Dict[str, Any], view: LiveView) -> Answer:
    key, label = p["metric"], p["label"]
    where = _window_text(p, view)
    notes: List[str] = []
    if key in ("pp", "pk") and any(p["window"].get(k) for k in ("last_n", "month", "since_month", "after_deadline", "before_deadline", "venue")) or (key in ("pp", "pk") and p.get("opp")):
        notes.append("Special-teams totals are kept for the season, not per game, so this is the full-season figure.")
    if p["window"].get("after_deadline") or p["window"].get("before_deadline"):
        notes.append(f"Deadline taken as {_deadline_iso(view.session)}.")

    if p["rank"]:
        rows = []
        for tid in view.team_meta:
            s = _summary(_select(view, tid, p["window"], p.get("opp")))
            if not s["gp"]:
                continue
            v = _value(view, tid, s, key)
            if v is None:
                continue
            rows.append((tid, v, s))
        if not rows:
            return _answer(f"No games {where} to rank yet.")
        # "most goals against" is still a count; "best defence" flips to fewest.
        want_high = p["higher"]
        if key in ("ga", "ga_pg", "l"):
            want_high = not re.search(r"\b(best|fewest|least|lowest)\b", p["q"]) if re.search(r"\b(most|worst|highest)\b", p["q"]) else False
        elif p["low"]:
            want_high = not want_high
        rows.sort(key=lambda r: r[1], reverse=want_high)
        top_tid, top_v, top_s = rows[0]
        mine = next((i + 1 for i, r in enumerate(rows) if r[0] == view.user_tid), None)
        head = f"{_team_label(view, top_tid)}: {_fmt(key, top_v, top_s)} {label.lower().replace(' %', '')} {where} ({_rec(top_s)} in {top_s['gp']} GP)."
        if mine and rows[0][0] != view.user_tid:
            ms = rows[mine - 1]
            notes.insert(0, f"You're {mine} of {len(rows)} at {_fmt(key, ms[1], ms[2])}.")
        table = {"cols": ["#", "Team", label, "Record", "GF", "GA"],
                 "rows": [[i + 1, _team_label(view, t), _fmt(key, v, s), _rec(s), s["gf"], s["ga"]] for i, (t, v, s) in enumerate(rows[:12])]}
        return _answer(head, analysis=notes, table=table,
                       followups=["How are we doing lately?", "Which team allows the fewest goals?", "Our record against the top teams"])

    tid = p["focus"] or view.user_tid
    rows = _select(view, tid, p["window"], p.get("opp"))
    team = _team_label(view, tid)
    s = _summary(rows)
    if not s["gp"]:
        extra = ""
        if p.get("opp"):
            extra = " They haven't met yet this season."
        hist = _h2h_history(view, tid, p.get("opp")) if p.get("opp") else None
        return _answer(f"No games for the {team} {where}.{extra}", analysis=[hist] if hist else [])
    v = _value(view, tid, s, key)
    if key == "record":
        head = f"The {team} are {_rec(s)} {where} — {s['pts']} points in {s['gp']} games, {s['gf']} goals for and {s['ga']} against ({s['gd']:+d})."
    else:
        head = f"The {team} {where}: {_fmt(key, v, s)} {label.lower().replace(' %', '')} ({_rec(s)} in {s['gp']} GP)."
    # Context against the full season when the window is a slice of it.
    if any(p["window"].get(k) for k in ("last_n", "month", "since_month", "after_deadline", "before_deadline", "venue", "first")) or p.get("opp"):
        full = _summary(_select(view, tid, {"playoffs": p["window"].get("playoffs")}, None))
        if full["gp"] and full["gp"] != s["gp"]:
            notes.append(f"Full season: {_rec(full)}, {full['pts_pct']:.3f} points %, {full['gf_pg']:.2f} GF and {full['ga_pg']:.2f} GA per game.")
            notes.append(f"In this stretch: {s['pts_pct']:.3f} points %, {s['gf_pg']:.2f} GF and {s['ga_pg']:.2f} GA per game.")
    if p.get("opp"):
        hist = _h2h_history(view, tid, p["opp"])
        if hist:
            notes.append(hist)
    if key not in ("record",):
        notes.append(f"Record {where}: {_rec(s)}; one-goal games {s['one_goal']}; overtime {s['ot']}.")
    table = {"cols": ["Date", "Opponent", "Score", "Result"],
             "rows": [[g["date"], ("vs " if g["home"] else "@ ") + _team_label(view, g["opp"]), f"{g['gf']}-{g['ga']}",
                       ("W" if g["won"] else ("OTL" if g["extra"] else "L")) + (" (OT)" if g["extra"] and g["won"] else "")]
                      for g in rows[-12:][::-1]]}
    return _answer(head, analysis=notes, table=table,
                   followups=["How are we doing lately?", "Our record on the road", "Which team allows the fewest goals?"])


def _h2h_history(view: LiveView, tid: str, opp: Optional[str]) -> Optional[str]:
    if not opp:
        return None
    try:
        from services.legacy_hall import ensure_legacy

        lg = ensure_legacy(view.session)
    except Exception:
        lg = getattr(view.session, "legacy", None) or {}
    rec = (lg.get("h2h") or {}).get("|".join(sorted((str(tid), str(opp))))) or {}
    gp = int(rec.get("gp") or 0)
    if not gp:
        return None
    wins = rec.get("wins") or {}
    seasons = len(rec.get("seasons") or [])
    line = f"Archived seasons ({seasons}): {int(wins.get(str(tid)) or 0)}-{int(wins.get(str(opp)) or 0)} in {gp} meetings"
    if int(rec.get("series") or 0):
        sw = rec.get("series_wins") or {}
        line += f"; playoff series {int(sw.get(str(tid)) or 0)}-{int(sw.get(str(opp)) or 0)}"
    return line + "."
