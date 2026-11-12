"""PuckGPT front-office facts: specific questions with a single right answer in the live save.

These used to fall into a near-miss handler (the oldest player came back as the youngest,
"who's my captain" wasn't understood, "when do we play next" got a diagnosis). Each one here
reads the save directly and answers only what was asked.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .converse import _DEF, _FWD, _cards, _clean, _focus_team, _ord, _pos_in, _who, roster_read

Answer = Dict[str, Any]
_POS_SETS = {"goalie": ("G",), "center": ("C",), "left wing": ("LW",), "right wing": ("RW",), "winger": ("LW", "RW", "W"),
             "defenseman": _DEF, "forward": _FWD}
_POS_PLURAL = {"goalie": "goalies", "center": "centers", "left wing": "left wingers", "right wing": "right wingers",
               "winger": "wingers", "defenseman": "defensemen", "forward": "forwards", "": "players"}
_NUM_WORDS = {"three": 3, "five": 5, "ten": 10, "four": 4, "six": 6, "seven": 7, "eight": 8}


def _money(m: float) -> str:
    return f"${m:.2f}M"


def _table(rows: List[Dict[str, Any]], extra: Tuple[str, ...] = ()) -> Dict[str, Any]:
    cols = ["Player", "Pos", "Age", "OVR", "POT", "Cap hit", "Years", "Status"]
    body = []
    for r in rows:
        c = r.get("contract") or {}
        body.append([r.get("name"), r.get("pos"), r.get("age"), round(float(r.get("ovr") or 0)), round(float(r.get("pot") or 0)),
                     _money(float(c.get("cap_hit_m") or 0)), c.get("years_left") or 0, c.get("status") or ""])
    return {"cols": cols, "rows": body, "ids": [r.get("id") for r in rows]}


def _listing(head: str, rows: List[Dict[str, Any]], lines: List[str], followups: List[str]) -> Answer:
    return {"answer": head, "analysis": lines, "table": _table(rows) if rows else None, "cards": _cards(rows[:4]),
            "focus_ids": [r["id"] for r in rows[:1]], "list_ids": [r["id"] for r in rows[:15]], "followups": followups}


def _roster(view: Any, tid: str) -> List[Dict[str, Any]]:
    return [r for r in view.team_rows(tid) if r]


def _team_name(view: Any, tid: str) -> str:
    return view.team_meta.get(str(tid), {}).get("name") or "that club"


def _cap(view: Any, tid: str) -> Optional[float]:
    t = view.teams.get(str(tid))
    for attr in ("cap_space_m", "cap_space"):
        try:
            v = float(getattr(t, attr, None))
            return v / 1e6 if v > 1000 else v
        except (TypeError, ValueError):
            continue
    return None


# --------------------------------------------------------------------------- roster facts

def age_extreme(view: Any, tid: str, oldest: bool, league: bool) -> Answer:
    rows = [view.row(p) for p, (_t, lv) in view.where.items() if lv == "NHL"] if league else _roster(view, tid)
    rows = sorted((r for r in rows if r.get("age")), key=lambda r: (-r["age"] if oldest else r["age"], -float(r.get("ovr") or 0)))
    if not rows:
        return {"answer": "I couldn't read any ages on that roster.", "analysis": []}
    top = rows[0]
    where = "in the league" if league else ("on your team" if str(tid) == str(view.user_tid) else f"on the {_team_name(view, tid)}")
    head = f"The {'oldest' if oldest else 'youngest'} player {where} is {top['name']} at {top['age']} ({top.get('pos')}, {top['ovr']:.0f} overall)."
    nxt = ", ".join(f"{r['name']} ({r['age']})" for r in rows[1:4])
    return _listing(head, rows[:8], [f"Next: {nxt}."] if nxt else [],
                    [f"Tell me about {top['name']}", "Who's my youngest player?" if oldest else "Who's my oldest player?"])


def pay_extreme(view: Any, tid: str, highest: bool) -> Answer:
    rows = [r for r in _roster(view, tid) if float((r.get("contract") or {}).get("cap_hit_m") or 0) > 0]
    rows.sort(key=lambda r: float(r["contract"]["cap_hit_m"]), reverse=highest)
    if not rows:
        return {"answer": "I couldn't read any contracts on that roster.", "analysis": []}
    top = rows[0]
    c = top["contract"]
    subj, _v, poss = _who(view, tid)
    head = (f"{poss.capitalize()} {'highest' if highest else 'lowest'}-paid player is {top['name']} at {_money(c['cap_hit_m'])}"
            + (f" for {c['years_left']} more year{'s' if c['years_left'] != 1 else ''}." if c.get("years_left") else "."))
    total = sum(float(r["contract"]["cap_hit_m"]) for r in rows)
    lines = [f"Next: {', '.join(r['name'] + ' ' + _money(r['contract']['cap_hit_m']) for r in rows[1:4])}.",
             f"NHL roster cap hits add up to {_money(total)}."]
    space = _cap(view, tid)
    if space is not None:
        lines.append(f"Cap space: {_money(space)}.")
    return _listing(head, rows[:8], lines, [f"Is {top['name']} overpaid?", "Worst contracts on my team", "How much cap space do I have?"])


def captain(view: Any, tid: str) -> Answer:
    letters = []
    for pid, (t, lv) in view.where.items():
        if str(t) != str(tid) or lv != "NHL":
            continue
        p = view.players[pid]
        mark = str(getattr(p, "captaincy", "") or "").upper()
        if getattr(p, "is_captain", False) or mark in ("C", "CAPTAIN"):
            letters.insert(0, ("C", view.row(pid)))
        elif getattr(p, "is_alternate", False) or mark in ("A", "ALTERNATE"):
            letters.append(("A", view.row(pid)))
    subj, _v, poss = _who(view, tid)
    caps = [r for l, r in letters if l == "C"]
    if not caps:
        rows = sorted(_roster(view, tid), key=lambda r: (-(r.get("age") or 0) * 0.4 - float(r.get("ovr") or 0)))[:3]
        return {"answer": f"Nobody is wearing the C for {poss} club right now.",
                "analysis": ["Veteran, high-rated candidates: " + ", ".join(f"{r['name']} ({r['age']}, {r['ovr']:.0f} overall)" for r in rows) + "."] if rows else [],
                "cards": _cards(rows), "followups": ["Who's my best player?", "Who's my oldest player?"]}
    c = caps[0]
    lines = [f"{c['name']} is {c['age']}, plays {c.get('pos')} and is {c['ovr']:.0f} overall."]
    alts = [r["name"] for l, r in letters if l == "A"]
    if alts:
        lines.append("Alternates: " + ", ".join(alts) + ".")
    return {"answer": f"{poss.capitalize()} captain is {c['name']}.", "analysis": lines, "cards": _cards([c]), "focus_ids": [c["id"]],
            "followups": [f"Tell me about {c['name']}", f"{c['name']} contract"]}


def injured(view: Any, tid: str) -> Answer:
    rows = sorted((r for r in _roster(view, tid) if r.get("injured")), key=lambda r: -float(r.get("ovr") or 0))
    subj, _v, poss = _who(view, tid)
    if not rows:
        return {"answer": f"Nobody on {poss} NHL roster is injured right now.", "analysis": [], "followups": ["What do we need?", "How is my team doing?"]}
    head = f"{len(rows)} player{'s are' if len(rows) != 1 else ' is'} out for {poss} club: " + ", ".join(f"{r['name']} ({r.get('pos')})" for r in rows[:5]) + "."
    return _listing(head, rows, [f"The biggest loss is {rows[0]['name']} ({rows[0]['ovr']:.0f} overall)."], [f"Tell me about {rows[0]['name']}", "What do we need?"])


def contract_list(view: Any, tid: str, kind: str) -> Answer:
    rows = _roster(view, tid) + ([r for r in view.team_rows(tid, ("AHL",)) if r] if kind in ("rfa", "ufa", "expiring") else [])
    subj, _v, poss = _who(view, tid)

    def c(r: Dict[str, Any]) -> Dict[str, Any]:
        return r.get("contract") or {}

    if kind == "clauses":
        rows = [r for r in rows if c(r).get("nmc") or c(r).get("ntc")]
        if not rows:
            return {"answer": f"No. Nobody on {poss} roster has a no-move or no-trade clause, so every contract can be moved.", "analysis": [],
                    "followups": ["Who should I trade?", "Worst contracts on my team"]}
        rows.sort(key=lambda r: -float(c(r).get("cap_hit_m") or 0))
        tags = ", ".join(f"{r['name']} ({'NMC' if c(r).get('nmc') else 'NTC'})" for r in rows[:6])
        return _listing(f"Yes, {len(rows)} of {poss} players {'have' if len(rows) != 1 else 'has'} trade protection: {tags}.", rows,
                        ["A no-move clause blocks trades and waivers. A no-trade clause blocks trades only."], ["Who should I trade?", "Worst contracts on my team"])
    expiring = [r for r in rows if int(c(r).get("years_left") or 0) <= 1 and float(c(r).get("cap_hit_m") or 0) > 0]
    if kind == "rfa":
        rows, label = [r for r in expiring if c(r).get("status") == "RFA"], "pending RFAs"
    elif kind == "ufa":
        rows, label = [r for r in expiring if c(r).get("status") == "UFA"], "pending UFAs"
    else:
        rows, label = expiring, "expiring contracts"
    if not rows:
        return {"answer": f"{subj} have no {label} this year.", "analysis": [], "followups": ["Worst contracts on my team", "How much cap space do I have?"]}
    rows.sort(key=lambda r: -float(r.get("ovr") or 0))
    head = f"{subj} have {len(rows)} {label}: " + ", ".join(f"{r['name']} ({r['ovr']:.0f})" for r in rows[:5]) + ("…" if len(rows) > 5 else ".")
    freed = sum(float(c(r).get("cap_hit_m") or 0) for r in rows)
    lines = [f"Together they carry {_money(freed)} in cap hits.",
             f"The one to sort out first is {rows[0]['name']} ({rows[0].get('pos')}, {rows[0]['age']}, {_money(c(rows[0])['cap_hit_m'])})."]
    if kind != "ufa":
        lines.append("RFAs stay under your control: you can match any offer sheet. UFAs walk if they aren't extended.")
    return _listing(head, rows, lines, [f"Should I extend {rows[0]['name']}?", "How much cap space do I have?"])


def cap_ranking(view: Any, most: bool) -> Answer:
    spaces = [(float(_cap(view, t) or 0), t) for t in view.teams if _cap(view, t) is not None]
    if not spaces:
        return {"answer": "I couldn't read cap space for the league.", "analysis": []}
    spaces.sort(reverse=most)
    top = spaces[0]
    mine = next((i + 1 for i, (_s, t) in enumerate(sorted(spaces, reverse=True)) if str(t) == str(view.user_tid)), 0)
    head = f"The {_team_name(view, top[1])} have the {'most' if most else 'least'} cap space in the league: {_money(top[0])}."
    lines = ["Next: " + ", ".join(f"{view.team_meta[str(t)]['abbr']} {_money(s)}" for s, t in spaces[1:5]) + "."]
    if mine:
        lines.append(f"You're {_ord(mine)} of {len(spaces)} with {_money(float(_cap(view, view.user_tid) or 0))}.")
    table = {"cols": ["Team", "Cap space"], "rows": [[_team_name(view, t), _money(s)] for s, t in spaces[:10]]}
    return {"answer": head, "analysis": lines, "table": table, "followups": ["How much cap space do I have?", "Best trade targets"]}


def cheap_goalie(view: Any, q: str) -> Answer:
    m = re.search(r"under \$?(\d+(?:\.\d+)?)\s*(?:m|mil|million)?", q)
    limit = float(m.group(1)) if m else 2.0
    pool = []
    for pid, (t, lv) in view.where.items():
        if str(t) == str(view.user_tid) or lv not in ("FA", "NHL"):
            continue
        r = view.row(pid)
        if r.get("pos") != "G":
            continue
        cap = float((r.get("contract") or {}).get("cap_hit_m") or 0)
        if lv == "FA" or cap <= limit:
            pool.append((0 if lv == "FA" else 1, r))
    fas = sorted((r for k, r in pool if k == 0), key=lambda r: -float(r.get("ovr") or 0))[:5]
    trade = sorted((r for k, r in pool if k == 1), key=lambda r: -float(r.get("ovr") or 0))[:5]
    best = (fas or trade or [None])[0]
    if not best:
        return {"answer": f"I can't find a goalie at or under {_money(limit)} right now.", "analysis": []}
    mine = sorted((r for r in _roster(view, view.user_tid) if r.get("pos") == "G"), key=lambda r: -float(r.get("ovr") or 0))
    head = (f"The best cheap option in net is {best['name']} ({best['ovr']:.0f} overall, {best['age']}), "
            + ("a free agent, so he only costs a contract." if fas else f"on the {best.get('team')} at {_money(best['contract']['cap_hit_m'])}."))
    lines = []
    if len(mine) > 1:
        lines.append(f"Your current backup is {mine[1]['name']} ({mine[1]['ovr']:.0f} overall), so this is {'an upgrade' if best['ovr'] > mine[1]['ovr'] else 'not an upgrade'}.")
    if fas and trade:
        lines.append(f"By trade at or under {_money(limit)}: " + ", ".join(f"{r['name']} ({r.get('team')}, {r['ovr']:.0f})" for r in trade[:3]) + ".")
    return _listing(head, fas + trade, lines, [f"Tell me about {best['name']}", "Which goalie should start?"])


def top_n(view: Any, tid: str, pos: str, n: int) -> Answer:
    want = _POS_SETS.get(pos)
    rows = sorted((r for r in _roster(view, tid) if not want or r.get("pos") in want), key=lambda r: -float(r.get("ovr") or 0))[:n]
    if not rows:
        return {"answer": "I couldn't find anyone at that position on the roster.", "analysis": []}
    subj, _v, poss = _who(view, tid)
    head = f"{poss.capitalize()} top {len(rows)} {_POS_PLURAL.get(pos, 'players')}: " + ", ".join(f"{i + 1}. {r['name']} ({r['ovr']:.0f})" for i, r in enumerate(rows)) + "."
    gap = rows[0]["ovr"] - rows[-1]["ovr"]
    lines = [f"There's a {gap:.0f}-point drop from first to last on that list."] if len(rows) > 1 else []
    return _listing(head, rows, lines, [f"Tell me about {rows[0]['name']}", "What do we need?"])


def extend_advice(view: Any, pid: str) -> Answer:
    r = view.row(pid)
    if not r:
        return {"answer": "Which player? Name him and I'll weigh an extension.", "analysis": []}
    c = r.get("contract") or {}
    age, ovr, pot, years = int(r.get("age") or 0), float(r.get("ovr") or 0), float(r.get("pot") or 0), int(c.get("years_left") or 0)
    cap = float(c.get("cap_hit_m") or 0)
    mine = str(r.get("team_id")) == str(view.user_tid)
    if not mine:
        return {"answer": f"{r['name']} isn't yours to extend. He's with the {r.get('team') or 'free-agent pool'}.", "analysis": [],
                "followups": [f"Is {r['name']} available?", f"Tell me about {r['name']}"], "focus_ids": [pid]}
    if years > 2:
        verdict = f"No rush. {r['name']} is signed for {years} more years at {_money(cap)}, so there's nothing to extend yet."
    elif age <= 26 and pot >= ovr:
        verdict = f"Yes, lock him up. {r['name']} is {age} with room to grow ({ovr:.0f} now, {pot:.0f} potential), so a long deal buys his best years."
    elif age <= 30 and ovr >= 84:
        verdict = f"Yes, but keep the term sensible. {r['name']} is {age} and {ovr:.0f} overall, so three to five years covers his prime without paying for the decline."
    elif age >= 33:
        verdict = f"Only short and cheap. {r['name']} is {age}, so anything past one or two years is paying for decline."
    elif ovr < 78:
        verdict = f"Probably not. {r['name']} is {ovr:.0f} overall, and that kind of player is replaceable in free agency or from the AHL."
    else:
        verdict = f"It's a fair bet at the right price. {r['name']} is {age} and {ovr:.0f} overall: useful, but not someone to overpay."
    lines = [f"Current deal: {_money(cap)} for {years} more year{'s' if years != 1 else ''}, {c.get('status') or 'status unknown'} at expiry."]
    if c.get("nmc") or c.get("ntc"):
        lines.append(f"He already has a {'no-move' if c.get('nmc') else 'no-trade'} clause.")
    space = _cap(view, view.user_tid)
    if space is not None:
        lines.append(f"You have {_money(space)} in cap space today.")
    return {"answer": verdict, "analysis": lines, "cards": _cards([r]), "focus_ids": [pid],
            "followups": [f"{r['name']} contract", f"Should I trade {r['name']}?"]}


_ORDINALS = {"first": 1, "1st": 1, "top": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}


def depth_chart(view: Any, tid: str, unit: str, which: int) -> Answer:
    """Lines and pairs by rating. The save's own line card can differ if the coach set it by hand."""
    rows = [r for r in _roster(view, tid) if not r.get("injured")]
    fw = sorted((r for r in rows if r.get("pos") in _FWD), key=lambda r: -float(r.get("ovr") or 0))
    de = sorted((r for r in rows if r.get("pos") in _DEF), key=lambda r: -float(r.get("ovr") or 0))
    lines = [fw[i:i + 3] for i in range(0, 12, 3) if fw[i:i + 3]]
    pairs = [de[i:i + 2] for i in range(0, 6, 2) if de[i:i + 2]]
    subj, _v, poss = _who(view, tid)

    def names(group: List[Dict[str, Any]]) -> str:
        return ", ".join(f"{r['name']} ({r['ovr']:.0f})" for r in group)

    note = "This is the depth chart by rating with injured players left out. Your saved line card can differ if you set it by hand."
    if unit in ("line", "pair") and which:
        groups = lines if unit == "line" else pairs
        if which > len(groups):
            return {"answer": f"{subj} don't have enough healthy {'forwards' if unit == 'line' else 'defensemen'} for that {unit}.", "analysis": [note]}
        g = groups[which - 1]
        label = {1: "first", 2: "second", 3: "third", 4: "fourth"}[which]
        avg = sum(r["ovr"] for r in g) / len(g)
        return _listing(f"{poss.capitalize()} {label} {unit}: {names(g)}.", g, [f"That group averages {avg:.1f} overall.", note],
                        [f"Who fits my {label} {unit}?", "What are my lines?"])
    out_lines = [f"Line {i + 1}: {names(g)}." for i, g in enumerate(lines)] + [f"Pair {i + 1}: {names(g)}." for i, g in enumerate(pairs)]
    gl = sorted((r for r in rows if r.get("pos") == "G"), key=lambda r: -float(r.get("ovr") or 0))
    if gl:
        out_lines.append("In net: " + names(gl[:2]) + ".")
    out_lines.append(note)
    head = f"{poss.capitalize()} top line is {names(lines[0])}." if lines else "I couldn't build a line from that roster."
    return _listing(head, (lines[0] if lines else []) + (pairs[0] if pairs else []), out_lines[1:] if lines else out_lines, ["What do we need?", "Who fits my second pair?"])


def best_by_age(view: Any, tid: str, pos: str, limit: int, under: bool) -> Answer:
    want = _POS_SETS.get(pos)
    rows = [r for r in _roster(view, tid) + [x for x in view.team_rows(tid, ("AHL", "Prospect")) if x]
            if (not want or r.get("pos") in want) and r.get("age") and (r["age"] < limit if under else r["age"] > limit)]
    rows.sort(key=lambda r: -float(r.get("ovr") or 0))
    subj, _v, poss = _who(view, tid)
    span = f"{'under' if under else 'over'} {limit}"
    if not rows:
        return {"answer": f"{subj} have no {_POS_PLURAL.get(pos, 'players')} {span}.", "analysis": []}
    top = rows[0]
    head = f"{poss.capitalize()} best {pos or 'player'} {span} is {top['name']} ({top['age']}, {top['ovr']:.0f} overall, {top['pot']:.0f} potential, {top.get('level')})."
    nxt = ", ".join(f"{r['name']} ({r['age']}, {r['ovr']:.0f})" for r in rows[1:4])
    return _listing(head, rows[:8], [f"Next: {nxt}."] if nxt else [], [f"Tell me about {top['name']}", "My top prospects"])


def _team_in(q: str, view: Any) -> str:
    """A club named in the question by city, nickname or abbreviation (so "boston's captain" works)."""
    text = " " + re.sub(r"'s\b", "", q) + " "
    for tid, meta in view.team_meta.items():
        if str(tid) == str(view.user_tid):
            continue
        for key in ("nickname", "city", "name"):
            word = str(meta.get(key) or "").lower()
            if len(word) >= 4 and f" {word} " in text:
                return str(tid)
    return ""


# --------------------------------------------------------------------------- calendar

def _slots_from(view: Any) -> Tuple[int, int, Dict[int, list], list]:
    s = view.session
    cur = int(getattr(s, "calendar_cursor", 0) or 0)
    last = int(getattr(s, "nhl_regular_season_last_index", 0) or 0)
    return cur, last, (getattr(s, "by_day", None) or {}), list(getattr(s, "nhl_calendar", None) or [])


def _day_label(cal: list, idx: int) -> str:
    try:
        row = cal[idx]
        return str(row.get("iso") or row.get("date") or row.get("label") or "")[:10]
    except Exception:
        return ""


def next_game(view: Any) -> Answer:
    cur, last, by_day, cal = _slots_from(view)
    uid = str(view.user_tid)
    for idx in sorted(k for k in by_day if int(k) >= cur):
        for slot in by_day[idx] or []:
            h, a = str(getattr(slot, "home_id", "")), str(getattr(slot, "away_id", ""))
            if uid in (h, a):
                opp = a if h == uid else h
                when = "today" if int(idx) == cur else (f"in {int(idx) - cur} day{'s' if int(idx) - cur != 1 else ''}")
                date = _day_label(cal, int(idx))
                head = f"Your next game is {when}{f' ({date})' if date else ''}: {'home against' if h == uid else 'away at'} the {_team_name(view, opp)}."
                lines = []
                read_me, read_them = roster_read(view, uid), roster_read(view, opp)
                if read_me and read_them:
                    edge = "you have the stronger roster on paper" if read_me["score"] < read_them["score"] else "they have the stronger roster on paper"
                    lines.append(f"{edge[0].upper() + edge[1:]}. Their best group is {read_them['best'][1]} ({_ord(read_them['ranks'][read_them['best'][0]])} in the league).")
                if getattr(slot, "is_playoff", False):
                    lines.append("It's a playoff game.")
                return {"answer": head, "analysis": lines,
                        "followups": [f"What is our record against the {_team_name(view, opp)}?", f"Best player on the {_team_name(view, opp)}", "Which goalie should start?"]}
    return {"answer": "There are no more games on your schedule right now. Advance the calendar to reach the next phase.", "analysis": [],
            "followups": ["How is my team doing?", "What do we need?"]}


def today(view: Any) -> Answer:
    cur, last, by_day, cal = _slots_from(view)
    date = _day_label(cal, max(0, min(cur, len(cal) - 1))) if cal else ""
    phase = str(getattr(view.session, "phase", "") or "").replace("_", " ")
    head = f"It's {date}, " if date else "It's "
    head += f"the {view.season} season" + (f" ({phase})." if phase else ".")
    left = sum(1 for k, slots in by_day.items() if cur <= int(k) <= max(last, cur) for sl in slots or []
               if str(view.user_tid) in (str(getattr(sl, "home_id", "")), str(getattr(sl, "away_id", ""))))
    lines = [f"You have {left} regular-season game{'s' if left != 1 else ''} left."] if left else []
    return {"answer": head, "analysis": lines, "followups": ["When do we play next?", "What's my record?"]}


def games_left(view: Any) -> Answer:
    cur, last, by_day, _cal = _slots_from(view)
    uid = str(view.user_tid)
    left = sum(1 for k, slots in by_day.items() if cur <= int(k) <= max(last, cur) for sl in slots or []
               if uid in (str(getattr(sl, "home_id", "")), str(getattr(sl, "away_id", ""))))
    played = len(view.team_games(uid))
    if not left:
        return {"answer": "Your regular season is done: there are no games left on the schedule.", "analysis": [f"Games played: {played}."], "followups": ["What's my record?"]}
    return {"answer": f"You have {left} regular-season game{'s' if left != 1 else ''} left, with {played} played.", "analysis": [],
            "followups": ["When do we play next?", "Are we going to make the playoffs?"]}


# --------------------------------------------------------------------------- league outlook and next steps

def cup_favourites(view: Any) -> Answer:
    reads = sorted(((roster_read(view, t).get("score", 9.0), str(t)) for t in view.teams))
    if not reads:
        return {"answer": "I couldn't read the league's rosters.", "analysis": []}
    top = reads[:5]
    mine = next((i + 1 for i, (_s, t) in enumerate(reads) if t == str(view.user_tid)), 0)
    head = f"Nobody can know, but on paper the favourite is the {_team_name(view, top[0][1])}, with the strongest roster in the league."
    lines = ["Next in line: " + ", ".join(_team_name(view, t) for _s, t in top[1:5]) + "."]
    if mine:
        lines.append(f"Your roster ranks {_ord(mine)} of {len(reads)} by the same measure.")
    try:
        rows = [r for r in view.standings() if int(r.get("gp") or 0) >= 10]
        if rows:
            lead = min(rows, key=lambda r: int(r.get("league_rank") or r.get("rank") or 99))
            lines.append(f"On the ice so far, the {lead.get('name')} lead the standings with {lead.get('pts')} points.")
    except Exception:
        pass
    lines.append("This ranks top-six forwards, top-four defense, goaltending and depth by rating. Injuries, form and matchups will move it.")
    return {"answer": head, "analysis": lines, "followups": ["Are we good enough to win the Cup?", "Standings", "What do we need?"]}


def checklist(view: Any) -> Answer:
    """"What should I do next?": the open items a GM would actually look at, most pressing first."""
    tid = str(view.user_tid)
    rows = _roster(view, tid)
    items: List[str] = []
    read = roster_read(view, tid)
    hurt = [r for r in rows if r.get("injured")]
    if hurt:
        items.append(f"Cover for injuries: {', '.join(r['name'] for r in hurt[:3])} {'are' if len(hurt) != 1 else 'is'} out.")
    if read and read["ranks"][read["worst"][0]] > read["n"] // 3:
        items.append(f"Upgrade {read['worst'][1]}: you're {_ord(read['ranks'][read['worst'][0]])} of {read['n']} there, your weakest group.")
    exp = sorted((r for r in rows if int((r.get("contract") or {}).get("years_left") or 0) <= 1 and float((r.get("contract") or {}).get("cap_hit_m") or 0) > 0),
                 key=lambda r: -float(r.get("ovr") or 0))
    if exp:
        items.append(f"Decide on expiring deals: {', '.join(r['name'] for r in exp[:3])}{' and others' if len(exp) > 3 else ''} need an extension or a trade.")
    space = _cap(view, tid)
    if space is not None:
        items.append(f"Use your cap room: you have {_money(space)} free." if space >= 3 else f"Watch the cap: only {_money(space)} of room, so any add needs salary going out.")
    cur, _last, by_day, _cal = _slots_from(view)
    if any(tid in (str(getattr(sl, "home_id", "")), str(getattr(sl, "away_id", ""))) for sl in by_day.get(cur, []) or []):
        items.insert(0, "You play today. Check your lines and your starting goalie before you sim.")
    if not items:
        return {"answer": "You're in good shape: no injuries, no expiring deals and no weak group. Sim ahead and check back.", "analysis": [],
                "followups": ["When do we play next?", "How is my team doing?"]}
    head = "Here's what I'd look at next, most pressing first. " + items[0]
    return {"answer": head, "analysis": items[1:], "followups": ["What do we need?", "Which contracts expire this year?", "Best trade targets", "When do we play next?"]}


_JOKES = (
    "Why did the goalie bring a ladder to practice? Coach told him to work on his high glove.",
    "My fourth line asked for more ice time, so I told them the Zamboni driver gets 15 minutes a night and never complains.",
    "A puck walks into a bar. The bar, the post and the other post. Still no goal.",
    "Why don't rebuilding teams play cards? They keep trading away all their aces for picks.",
    "The backup goalie has the best job in hockey: a great seat, a clean jersey and a ball cap.",
)


def banter(q: str, view: Any) -> Optional[Tuple[str, Answer]]:
    if re.search(r"\b(?:joke|make me laugh|say something funny|funny)\b", q):
        pick = _JOKES[sum(ord(ch) for ch in q) % len(_JOKES)]
        return "chat:joke", {"answer": pick, "analysis": [], "followups": ["Tell me another joke", "What should I do next?"]}
    if re.search(r"^(?:i'?m|im|i am) (?:so )?bored\b|^bored$|^entertain me\b|^surprise me\b", q):
        out = checklist(view)
        out["answer"] = "Let's find you something to do. " + str(out["answer"]).replace("Here's what I'd look at next, most pressing first. ", "")
        return "chat:bored", out
    return None


# --------------------------------------------------------------------------- matching

_MY = r"(?:my|our|the|we|i|us)"
_RX = {
    "oldest": re.compile(r"\b(?:oldest|eldest|most senior|greybeard|graybeard|old(?:est)? guy|longest in the tooth)\b"),
    "youngest": re.compile(r"\b(?:youngest|baby of the team|young(?:est)? guy|youngest kid)\b"),
    "paid_hi": re.compile(r"\b(?:highest[- ]paid|makes? the most(?: money)?|earns? the most|biggest (?:contract|cap hit|salary)|most expensive player|top earner|paid the most|largest (?:contract|cap hit|salary)|highest (?:cap hit|salary))\b"),
    "paid_lo": re.compile(r"\b(?:lowest[- ]paid|makes? the least|earns? the least|cheapest (?:player|contract|guy)|smallest (?:contract|cap hit|salary)|paid the least|lowest (?:cap hit|salary))\b"),
    "captain": re.compile(r"\b(?:who(?:'?s| is| are)?|do (?:we|i) have an?|name) .*\b(?:captain|wears? the c|the c|leadership group|alternates?|alternate captains?)\b|^(?:my |our )?captain$"),
    "injured": re.compile(r"\b(?:who(?:'?s| is| are)?|anyone|anybody|any (?:players?|guys?)|which players?(?: are)?|how many(?: players| guys)?(?: are)?) .*\b(?:injured|hurt|banged up|on (?:the )?(?:ir|ltir|injured reserve|shelf)|out (?:right now|injured|hurt)|unavailable|sidelined)\b|^(?:injury report|injuries|injury list|who'?s out)$"),
    "rfa": re.compile(r"\b(?:rfas?|restricted free agents?)\b"),
    "ufa": re.compile(r"\b(?:ufas?|unrestricted free agents?)\b"),
    "expiring": re.compile(r"\b(?:expir\w+|contracts? (?:are )?(?:up|ending|running out)|(?:who(?:'?s| is| are)?|whose) .*(?:contract (?:is |are )?up|in a contract year|needs? a new (?:deal|contract))|pending free agents?|walk(?:ing)? (?:this|in the) summer|free agents? (?:this|next) summer|needs? (?:to be )?(?:re-?sign\w*|extend\w*))\b"),
    "clauses": re.compile(r"\b(?:no[- ]?move|no[- ]?trade|nmcs?|ntcs?|trade protection|clauses?)\b"),
    "cap_league": re.compile(r"\b(?:who|which team|what team)\b.*\b(most|least|biggest|smallest|lowest|highest)\b.*\bcap (?:space|room)\b|\bcap (?:space|room)\b.*\b(?:in the league|league[- ]wide|by team|rank\w*)\b"),
    "cheap_goalie": re.compile(r"\b(?:cheap|budget|bargain|affordable|inexpensive|low[- ]cost|league minimum)\b.*\b(?:goalie|goaltender|netminder|backup)\b|\bbackup (?:goalie|goaltender|netminder)\b.*\b(?:find|need|get|sign|available|cheap|target|options?)\b|\b(?:find|need|get|sign)\b.*\bbackup (?:goalie|goaltender|netminder)?\b"),
    "top_n": re.compile(r"\b(?:top|best) (\d{1,2}|three|four|five|six|seven|eight|ten)\b(?: of)?(?: " + _MY + r")? ([\w -]+?)$|^(?:rank|list|order|sort|show(?: me)?)(?: all)?(?: of)? (?:my|our) ([\w -]+?)(?: by (?:overall|rating|ovr))?$"),
    "extend": re.compile(r"\b(?:should|do|can|would) (?:i|we|you) (?:extend|re-?sign|resign|keep|give .* (?:a new|an) (?:deal|contract|extension))\b|\bworth (?:an extension|extending|re-?signing|keeping)\b|\bextension for\b"),
    "next_game": re.compile(r"\b(?:when|who|what time|where) (?:do|are|is|does) (?:we|i|my team|our team|the team) (?:play(?:ing)?|face|facing|up against)(?: next| tonight| today| tomorrow)?\b|\bnext (?:game|opponent|matchup)\b|\bwho(?:'?s| is) (?:next|up next)\b|\b(?:who|what)(?:'?s| is) (?:on )?(?:the )?schedule\b|\bdo we play (?:today|tonight|tomorrow)\b|\bupcoming (?:game|schedule|opponent)\b"),
    "today": re.compile(r"\bwhat(?:'?s| is)? (?:the |today'?s )?(?:date|day)(?: is it| today)?\b|\bwhat (?:season|year|month) (?:is it|are we in)\b|\bwhich (?:season|year) (?:is it|are we in)\b|\bwhere are we in the (?:season|calendar|schedule)\b|\bwhat part of the season\b"),
    "games_left": re.compile(r"\bhow many (?:games|more games) (?:are |do we have |have we got )?(?:left|remaining|to go|to play)\b|\bgames (?:left|remaining)\b|\bhow (?:far|deep) (?:into|in) the season\b|\bhow much of the season is left\b"),
    "cup_pick": re.compile(r"\b(?:who(?:'?s| is|se)?|which team|what team)\b.*\b(?:going to|gonna|will|favou?rites?|favou?red|likely to|most likely|pick|chances?)\b.*\b(?:cup|championship|title|it all)\b|\bcup favou?rites?\b|\bwho wins the cup\b|\bwho(?:'?s| is) the (?:favou?rite|team to beat)\b"),
    "checklist": re.compile(r"\bwhat (?:should|do|can|would) (?:i|we|you) (?:do|focus on|work on|tackle|prioriti[sz]e|look at)(?: (?:next|now|first|today|here))?$|\bwhat(?:'?s| is) (?:next|my next move|the plan|the priority|my to[- ]?do(?: list)?)\b|\bwhere (?:do|should) i start\b|\bhelp me (?:fix|improve|run|manage|build|with) (?:my|our|this|the) (?:team|roster|club)\b|\bwhat would you do\b|\bany advice\b|\bto[- ]?do list\b|\bgive me (?:a |some )?(?:advice|tips|plan|game ?plan)\b|\bwhat needs (?:my )?attention\b"),
}


def _first_player(view: Any, slots: Dict[str, Any], ctx: Dict[str, Any], q: str) -> str:
    for cands in slots.get("players") or []:
        if cands and cands[0] in view.players:
            return cands[0]
    if re.search(r"\b(?:him|he|his|that guy|this guy|the guy)\b", q):
        for pid in ctx.get("players") or []:
            if pid in view.players:
                return pid
    return ""


def match(q: str, view: Any, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[Tuple[str, Answer]]:
    """A direct answer for a specific front-office question, or None to let the intent router take it."""
    ql = _clean(q)
    if not ql:
        return None
    chat = banter(ql, view)
    if chat:
        return chat
    other = _team_in(ql, view) if not re.search(r"\b(?:against|vs|versus|trade with|from the|play)\b", ql) else ""
    named = bool(slots.get("players")) and not other
    tid = other or _focus_team(view, slots, ql)
    league = bool(re.search(r"\b(?:in the (?:league|nhl|game)|league[- ]wide|in hockey|of all|anywhere)\b", ql))

    if _RX["extend"].search(ql):
        pid = _first_player(view, slots, ctx, ql)
        if pid:
            return "fact:extend", extend_advice(view, pid)
        if re.search(r"\bextend|re-?sign|resign\b", ql):
            return "fact:expiring", contract_list(view, tid, "expiring")
    if named:
        return None  # everything below is about a club, not one named player
    if _RX["next_game"].search(ql):
        return "fact:next_game", next_game(view)
    if _RX["games_left"].search(ql):
        return "fact:games_left", games_left(view)
    if _RX["today"].search(ql):
        return "fact:today", today(view)
    if _RX["cup_pick"].search(ql) and not re.search(r"\b(?:last|won|ago|previous|past|history)\b", ql):
        return "fact:cup_pick", cup_favourites(view)
    if _RX["checklist"].search(ql):
        return "fact:checklist", checklist(view)
    if _RX["cap_league"].search(ql):
        return "fact:cap_league", cap_ranking(view, most=not re.search(r"\b(?:least|smallest|lowest)\b", ql))
    if _RX["captain"].search(ql) and not re.search(r"\b(?:first|ever|history|all[- ]time|used to|was)\b", ql):
        return "fact:captain", captain(view, tid)
    if _RX["injured"].search(ql) and not re.search(r"\b(?:history|ever|last season|most games|career)\b", ql):
        return "fact:injured", injured(view, tid)
    if _RX["clauses"].search(ql):
        return "fact:clauses", contract_list(view, tid, "clauses")
    if _RX["cheap_goalie"].search(ql):
        return "fact:cheap_goalie", cheap_goalie(view, ql)
    own = bool(re.search(r"\b(?:my|our|we|i|do i have|on (?:my|our) (?:team|roster))\b", ql)) or str(tid) != str(view.user_tid)
    if own and _RX["rfa"].search(ql) and not re.search(r"\b(?:best|available|sign|target|market|league)\b", ql):
        return "fact:rfa", contract_list(view, tid, "rfa")
    if own and _RX["ufa"].search(ql) and not re.search(r"\b(?:best|available|sign|target|market|league)\b", ql):
        return "fact:ufa", contract_list(view, tid, "ufa")
    if (own or not league) and _RX["expiring"].search(ql) and not re.search(r"\b(?:best|available|market|sign)\b", ql):
        return "fact:expiring", contract_list(view, tid, "expiring")
    m = re.search(r"\b(?:my|our) (?:best|top) ([\w -]*?)\s?(under|younger than|below|over|older than|above) (?:the age of |age )?(\d{2})\b", ql)
    if m:
        return "fact:best_by_age", best_by_age(view, tid, _pos_in(m.group(1)), int(m.group(3)), m.group(2) in ("under", "younger than", "below"))
    m = re.search(r"\b(?:who(?:'?s| is| are)?|show|what(?:'?s| is| are)?|list)\b.*\b(first|1st|top|second|2nd|third|3rd|fourth|4th)[- ](line|pair|pairing|d pair|d-pair|defen[cs]e pair)\b", ql)
    if m and not re.search(r"\b(?:fit|fits|upgrade|need|should|power ?play|pp|pk|penalty)\b", ql):
        return "fact:lines", depth_chart(view, tid, "line" if m.group(2) == "line" else "pair", _ORDINALS[m.group(1)])
    if re.search(r"\b(?:what are|show(?: me)?|list|give me|who(?:'?s| is) in) (?:my|our|the) (?:lines|line ?up|lineup|depth chart|line combos|line combinations|pairs|pairings|d pairs)\b|^(?:my )?(?:lines|lineup|depth chart)$", ql):
        return "fact:lines", depth_chart(view, tid, "", 0)
    if _RX["oldest"].search(ql):
        return "fact:oldest", age_extreme(view, tid, True, league)
    if _RX["youngest"].search(ql) and not re.search(r"\b(?:under|below|younger than)\b", ql):
        return "fact:youngest", age_extreme(view, tid, False, league)
    if not league and _RX["paid_hi"].search(ql):
        return "fact:highest_paid", pay_extreme(view, tid, True)
    if not league and _RX["paid_lo"].search(ql):
        return "fact:lowest_paid", pay_extreme(view, tid, False)
    m = _RX["top_n"].search(ql)
    if m and not league and not re.search(r"\b(?:available|free agents?|trade|targets?|prospects?|draft|contracts?|by (?!overall|rating|ovr))\b", ql):
        count, what = (m.group(1), m.group(2)) if m.group(2) else ("8", m.group(3))
        n = _NUM_WORDS.get(count) or (int(count) if str(count).isdigit() else 5)
        pos = _pos_in(what or "")
        if pos or re.search(r"\b(?:players?|guys|skaters|roster|team)\b", what or ""):
            return "fact:top_n", top_n(view, tid, pos, max(2, min(n, 12)))
    return None
