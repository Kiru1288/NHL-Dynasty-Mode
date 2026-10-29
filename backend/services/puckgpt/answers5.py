"""Game-record questions: blown leads, comebacks, blowouts, wild games, long overtimes.

Each final is replayed goal by goal from its scoring events when the box has
them; light boxes fall back to their period lines, so a lead is then measured
at each intermission. Results are cached on the game count, so asking again
costs nothing until another game is played.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .answers import _answer, _team_label
from .data import LiveView

Answer = Dict[str, Any]

RULES5: List[Tuple[str, str]] = [
    (r"\b(blow|blew|blown|blowing|below|bloe)\w*\s+(\w+[- ]?\s*){0,3}?leads?\b"
     r"|\blead (was )?(blown|lost|squandered|surrendered)\b|\b(lost|surrendered|squandered|coughed up)\s+(an?\s+|the\s+)?(\d+[- ]goal\s+)?leads?\b", "blown_lead"),
    (r"\bcome ?backs?\b|\bcame back from\b|\brall(y|ied) from\b|\bovercame\b.*\bdeficit\b", "comeback"),
    (r"\b(biggest|largest|worst|most lopsided)\s+(win|blowout|rout|beating|loss|margin|defeat)\b|\bblowouts?\b", "blowout"),
    (r"\b(highest|most)[- ]scoring games?\b|\bmost goals in (a|one) game\b|\bwildest games?\b", "high_scoring"),
    (r"\blongest (overtime|ot)\b|\bmost overtimes\b|\b(multi|double|triple)[- ]?overtime\b", "long_ot"),
]

_CACHE: Dict[str, Any] = {}


def match(raw: str) -> Optional[str]:
    q = " ".join(str(raw or "").lower().replace("’", "'").split())
    for pattern, kind in RULES5:
        if re.search(pattern, q):
            return kind
    return None


def _clock(ev: Dict[str, Any]) -> int:
    try:
        m, s = str(ev.get("clock") or "0:0").split(":")
        return int(m) * 60 + int(s)
    except (ValueError, TypeError):
        return 0


def _sequence(box: Dict[str, Any], hid: str, aid: str) -> Tuple[List[Tuple[int, int]], bool]:
    """Running (home, away) score at each intermission.

    Goal clocks in the scoring log are random within a period, so ordering goals by
    clock invents leads that never existed (a "9-goal comeback"). Only the period
    totals are real, so leads are measured at the end of each period.
    """
    hg = int(box.get("home_goals", box.get("home_score")) or 0)
    ag = int(box.get("away_goals", box.get("away_score")) or 0)
    hp = [box.get(f"home_p{i}") for i in (1, 2, 3)]
    ap = [box.get(f"away_p{i}") for i in (1, 2, 3)]
    if None in hp or None in ap:
        events = [e for e in box.get("scoring_events") or [] if isinstance(e, dict)]
        if events:
            hp, ap = [0, 0, 0], [0, 0, 0]
            for ev in events:
                try:
                    per = int(ev.get("period") or 0)
                except (TypeError, ValueError):
                    continue
                if 1 <= per <= 3:
                    tid = str(ev.get("for_team_id") or ev.get("team_id") or "")
                    if tid == hid:
                        hp[per - 1] += 1
                    elif tid == aid:
                        ap[per - 1] += 1
    if None not in hp and None not in ap:
        h = a = 0
        out = [(0, 0)]
        for i in range(3):
            h += int(hp[i] or 0)
            a += int(ap[i] or 0)
            out.append((h, a))
        out.append((hg, ag))
        return out, False
    return [(0, 0), (hg, ag)], False


def _games(view: LiveView) -> List[Dict[str, Any]]:
    results = list(getattr(view.session, "game_results", None) or [])
    key = f"{id(view.session)}:{len(results)}"
    if _CACHE.get("key") == key:
        return _CACHE["games"]
    games = []
    for box in results:
        if not isinstance(box, dict):
            continue
        hid, aid = str(box.get("home_id") or ""), str(box.get("away_id") or "")
        if not hid or not aid:
            continue
        hg = int(box.get("home_goals", box.get("home_score")) or 0)
        ag = int(box.get("away_goals", box.get("away_score")) or 0)
        if hg == ag and not box.get("shootout_winner_id"):
            continue
        seq, exact = _sequence(box, hid, aid)
        home_won = hg > ag or str(box.get("shootout_winner_id") or box.get("winner_id") or "") == hid and hg == ag
        # Biggest lead the eventual loser held, and biggest deficit the winner erased.
        loser_lead = max((a - h) if home_won else (h - a) for h, a in seq)
        games.append({
            "box": box, "hid": hid, "aid": aid, "hg": hg, "ag": ag, "home_won": home_won, "exact": exact,
            "blown": max(0, loser_lead),
            "margin": abs(hg - ag), "total": hg + ag,
            "ot_periods": int(box.get("ot_periods") or (1 if box.get("overtime") or box.get("ot") else 0)),
            "playoff": bool(box.get("is_playoff")),
            "date": str(box.get("calendar_iso") or box.get("iso") or "")[:10],
        })
    _CACHE.update(key=key, games=games)
    return games


def _abbr(view: LiveView, tid: str) -> str:
    return _team_label(view, tid)


def _line(view: LiveView, g: Dict[str, Any]) -> str:
    h, a = _abbr(view, g["hid"]), _abbr(view, g["aid"])
    tag = " (SO)" if g["box"].get("shootout") else (" (OT)" if g["ot_periods"] else "")
    return f"{a} {g['ag']} at {h} {g['hg']}{tag}"


def run(kind: str, view: LiveView, raw: str, slots: Dict[str, Any]) -> Answer:
    q = raw.lower()
    games = _games(view)
    mine = bool(re.search(r"\b(we|us|our|my)\b", q))
    teams = [str(t) for t in (slots.get("teams") or [])]
    focus = view.user_tid if mine else (teams[0] if teams and not re.search(r"\bany team\b|\bleague\b", q) else None)
    playoffs = bool(re.search(r"\bplayoffs?\b|\bpostseason\b", q))
    pool = [g for g in games if g["playoff"] == playoffs] or games
    scope = "playoffs" if playoffs else "this season"

    def involves(g: Dict[str, Any]) -> bool:
        return focus is None or focus in (g["hid"], g["aid"])

    if kind in ("blown_lead", "comeback"):
        # Blown lead: rank by the loser's lead; "we blew" means our side lost it.
        def lost_by(g):
            return g["aid"] if g["home_won"] else g["hid"]

        def won_by(g):
            return g["hid"] if g["home_won"] else g["aid"]

        side = lost_by if kind == "blown_lead" else won_by
        if kind == "blown_lead" and re.search(r"\b(vs|against|on|to) (us|me)\b", q):
            # "who blew a lead against us": the club that lost a lead to the user.
            side = won_by
            focus = view.user_tid
        rows = [g for g in pool if g["blown"] >= 2 and (focus is None or side(g) == focus)]
        rows.sort(key=lambda g: (-g["blown"], g["date"]))
        if not rows:
            who = _abbr(view, focus) if focus else "No team"
            what = "blown a multi-goal lead" if kind == "blown_lead" else "come back from a multi-goal deficit"
            return _answer(f"{who} hasn't {what} {scope}." if focus else f"No team has {what} {scope}.")
        top = rows[0]
        loser, winner = _abbr(view, lost_by(top)), _abbr(view, won_by(top))
        n = top["blown"]
        head = (f"Biggest blown lead {scope}: {loser} led by {n} and lost to {winner} — {_line(view, top)}, {top['date']}."
                if kind == "blown_lead" else
                f"Biggest comeback {scope}: {winner} erased a {n}-goal deficit against {loser} — {_line(view, top)}, {top['date']}.")
        count = len(rows)
        notes = [f"{count} game{'s' if count != 1 else ''} {scope} saw a lead of 2+ goals end in a loss."]
        notes.append("Leads are measured at each intermission (goal times within a period aren't recorded precisely).")
        table = {"cols": ["Date", "Game", "Lead blown", "By"],
                 "rows": [[g["date"], _line(view, g), g["blown"], _abbr(view, lost_by(g))] for g in rows[:10]]}
        return _answer(head, analysis=notes, table=table,
                       followups=["Biggest comeback this season", "Biggest blowout this season", "Longest overtime game"])

    if kind == "blowout":
        rows = sorted((g for g in pool if involves(g)), key=lambda g: (-g["margin"], -g["total"], g["date"]))
        if focus and re.search(r"\b(loss|lost|beating|defeat|worst)\b", q):
            rows = [g for g in rows if (g["aid"] if g["home_won"] else g["hid"]) == focus]
        if not rows:
            return _answer(f"No finished games {scope} yet.")
        top = rows[0]
        head = f"Biggest margin {scope}: {_line(view, top)} — a {top['margin']}-goal game, {top['date']}."
        table = {"cols": ["Date", "Game", "Margin"], "rows": [[g["date"], _line(view, g), g["margin"]] for g in rows[:10]]}
        return _answer(head, table=table, followups=["Biggest blown lead this season", "Highest-scoring game"])

    if kind == "high_scoring":
        rows = sorted((g for g in pool if involves(g)), key=lambda g: (-g["total"], g["date"]))
        if not rows:
            return _answer(f"No finished games {scope} yet.")
        top = rows[0]
        head = f"Highest-scoring game {scope}: {_line(view, top)} — {top['total']} goals, {top['date']}."
        table = {"cols": ["Date", "Game", "Goals"], "rows": [[g["date"], _line(view, g), g["total"]] for g in rows[:10]]}
        return _answer(head, table=table, followups=["Biggest blowout this season", "Biggest blown lead this season"])

    rows = sorted((g for g in pool if involves(g) and g["ot_periods"]), key=lambda g: (-g["ot_periods"], g["date"]))
    if not rows:
        return _answer(f"No overtime games {scope} yet.")
    top = rows[0]
    n = top["ot_periods"]
    head = f"Longest overtime {scope}: {_line(view, top)} — {n} overtime period{'s' if n != 1 else ''}, {top['date']}."
    table = {"cols": ["Date", "Game", "OT periods"], "rows": [[g["date"], _line(view, g), g["ot_periods"]] for g in rows[:10]]}
    return _answer(head, table=table, followups=["Biggest comeback this season"])
