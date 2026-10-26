"""Answer handlers. Each takes (view, slots, ctx) and returns one answer dict.

Every number in an answer is read from the save. Analysis lines are templates
filled with those numbers, so nothing is invented.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from .data import LiveView, archive

Answer = Dict[str, Any]

SINGULAR = {"defensemen": "defenseman", "players": "player", "centers": "center", "left wings": "left wing",
            "right wings": "right wing", "wingers": "winger", "forwards": "forward", "goalies": "goalie",
            "left D": "left D", "right D": "right D"}

POS_NAMES = {"LD": "left D", "RD": "right D", "D": "defensemen", "C": "centers", "LW": "left wings",
             "RW": "right wings", "W": "wingers", "F": "forwards", "G": "goalies"}
STAT_NAMES = {"g": "goals", "a": "assists", "pts": "points", "plus_minus": "plus/minus", "pim": "PIM",
              "hit": "hits", "blk": "blocks", "sog": "shots", "w": "wins", "so": "shutouts", "sv_pct": "save %",
              "gp": "games", "toi": "ice time", "ppg": "power-play goals", "gaa": "GAA", "fo_pct": "faceoff %"}
PAIR_RANK = {"pair1": (0, 2), "pair2": (2, 4), "pair3": (4, 6), "top4": (0, 4), "top6": (0, 6),
             "line1": (0, 3), "line2": (3, 6), "line3": (6, 9), "line4": (9, 12), "bot6": (6, 12)}


# --------------------------------------------------------------------------- helpers
def _pos_ok(row: Dict[str, Any], want: Optional[str]) -> bool:
    if not want:
        return True
    pos = row.get("pos") or ""
    if want == "G":
        return pos == "G"
    if want in ("D", "LD", "RD"):
        if pos not in ("D", "LD", "RD"):
            return False
        if want == "LD":
            return pos == "LD" or (pos == "D" and row.get("shoots") == "L")
        if want == "RD":
            return pos == "RD" or (pos == "D" and row.get("shoots") == "R")
        return True
    if want == "F":
        return pos in ("C", "LW", "RW", "W", "F")
    if want == "W":
        return pos in ("LW", "RW", "W")
    return pos == want


def _money(m: Any) -> str:
    try:
        return f"${float(m):.2f}M"
    except (TypeError, ValueError):
        return "—"


def _card(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "player", "id": row.get("id"), "label": row.get("name"),
            "sub": f"{row.get('pos')} · {row.get('team') or row.get('level')} · {row.get('ovr', 0):.0f} OVR"}


def _player_table(rows: List[Dict[str, Any]], extra: Tuple[str, ...] = ("pts",)) -> Dict[str, Any]:
    cols = ["Player", "Pos", "Team", "Age", "OVR", "POT"] + [STAT_NAMES.get(e, e).title() if e not in ("cap", "status") else
                                                             {"cap": "Cap hit", "status": "Status"}[e] for e in extra]
    out = []
    for r in rows:
        line = [r.get("name"), r.get("pos"), r.get("team") or r.get("level"), r.get("age"), round(r.get("ovr") or 0), round(r.get("pot") or 0)]
        for e in extra:
            if e == "cap":
                line.append(_money((r.get("contract") or {}).get("cap_hit_m")))
            elif e == "status":
                c = r.get("contract") or {}
                line.append(f"{c.get('status') or '—'} {c.get('years_left') or 0}y")
            else:
                v = r.get(e)
                line.append(v if v is not None else "—")
        out.append(line)
    return {"cols": cols, "rows": out, "ids": [r.get("id") for r in rows]}


def _answer(text: str, *, analysis: Optional[List[str]] = None, rows: Optional[List[Dict[str, Any]]] = None,
            extra: Tuple[str, ...] = ("pts",), followups: Optional[List[str]] = None, table: Optional[Dict[str, Any]] = None,
            cards: Optional[List[Dict[str, Any]]] = None, focus_ids: Optional[List[str]] = None,
            list_ids: Optional[List[str]] = None) -> Answer:
    out: Answer = {"answer": text, "analysis": [a for a in (analysis or []) if a], "followups": followups or []}
    if rows:
        out["table"] = _player_table(rows, extra)
        out["cards"] = [_card(r) for r in rows[:6]]
        out["list_ids"] = [r.get("id") for r in rows]
    if table:
        out["table"] = table
    if cards:
        out["cards"] = cards
    if focus_ids:
        out["focus_ids"] = focus_ids
    if list_ids:
        out["list_ids"] = list_ids
    return out


def _team_id(view: LiveView, slots: Dict[str, Any]) -> str:
    teams = slots.get("teams") or []
    if teams and not slots.get("self"):
        return str(teams[0])
    return view.user_tid


def _team_label(view: LiveView, tid: str) -> str:
    return view.team_meta.get(tid, {}).get("name") or tid


def _resolve_players(view: LiveView, slots: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per mention: prefer the candidate on the user's team, then the highest rated."""
    out = []
    for cands in slots.get("players") or []:
        rows = [view.row(pid) for pid in cands if pid in view.players]
        if not rows:
            continue
        rows.sort(key=lambda r: (r.get("team_id") != view.user_tid, -(r.get("ovr") or 0)))
        out.append(rows[0])
    return out


def _cap_space(view: LiveView, tid: str) -> Optional[float]:
    t = view.teams.get(tid)
    for attr in ("cap_space_m", "cap_space"):
        v = getattr(t, attr, None)
        try:
            v = float(v)
            return v / 1e6 if v > 1000 else v
        except (TypeError, ValueError):
            continue
    return None


def _market_value(view: LiveView, pid: str) -> Optional[float]:
    try:
        from services.contract_economy import compute_market_value

        return float(compute_market_value(view.players[pid], view.league))
    except Exception:
        return None


def _standing(view: LiveView, tid: str) -> Optional[Dict[str, Any]]:
    for r in view.standings():
        if str(r.get("team_id")) == str(tid):
            return r
    return None


# --------------------------------------------------------------------------- handlers
def best_on_team(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    tid = _team_id(view, slots)
    pos = slots.get("position")
    rows = [r for r in view.team_rows(tid) if _pos_ok(r, pos)]
    if "young" in (slots.get("filters") or []) or slots.get("age_max"):
        rows = [r for r in rows if r["age"] <= int(slots.get("age_max") or 24)]
    low = slots.get("direction") == "low"
    rows.sort(key=lambda r: r["ovr"] if low else -r["ovr"])
    label = POS_NAMES.get(pos, "players")
    if not rows:
        return _answer(f"No {label} on the {_team_label(view, tid)} roster right now.")
    top = rows[0]
    owner = "Your" if tid == view.user_tid else f"The {_team_label(view, tid)}'"
    extra = ("w", "sv_pct") if pos == "G" else ("gp", "pts", "plus_minus")
    analysis = [f"{top['name']} is {top['ovr']:.0f} overall with {top['pot']:.0f} potential at {top['age']}."]
    if len(rows) > 1:
        analysis.append(f"Gap to #{2}: {top['ovr'] - rows[1]['ovr']:.1f} overall ({rows[1]['name']}).")
    best_prod = max(rows, key=lambda r: r.get("pts") or 0)
    if pos != "G" and best_prod is not top and (best_prod.get("pts") or 0) > (top.get("pts") or 0):
        analysis.append(f"By production this season it is {best_prod['name']} ({best_prod['pts']} pts in {best_prod['gp']}).")
    return _answer(f"{owner} {'lowest-rated' if low else 'best'} {SINGULAR.get(label, label)}: {top['name']} ({top['ovr']:.0f} OVR).",
                   analysis=analysis, rows=rows[:int(slots.get('limit') or 10)], extra=extra,
                   followups=[f"What about {top['name']}'s contract?", f"Best {label} in the league", "Who fits my second pair?"])


def find_players(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    pos = slots.get("position")
    filters = set(slots.get("filters") or [])
    levels = {"NHL", "FA"}
    if slots.get("direction") == "low" or slots.get("both"):
        levels = {"NHL"}  # the worst player "in the league" is a rostered NHL player, not an unsigned depth body
    if "prospect" in filters:
        levels = {"Prospect", "AHL", "ECHL"}
    rows = [view.row(pid) for pid, (tid, lv) in view.where.items() if lv in levels]
    rows = [r for r in rows if _pos_ok(r, pos)]
    if slots.get("from_list"):
        keep = set(slots["from_list"])
        rows = [r for r in rows if r["id"] in keep] or [view.row(pid) for pid in keep if pid in view.players]
    if "free" in filters or "available" in filters:
        rows = [r for r in rows if r["level"] == "FA"]
    if "ufa" in filters:
        rows = [r for r in rows if r["level"] == "FA" or ((r["contract"]["status"] == "UFA") and r["contract"]["years_left"] <= 1)]
    if "rfa" in filters:
        rows = [r for r in rows if r["contract"]["status"] == "RFA" and r["contract"]["years_left"] <= 1]
    if "rookie" in filters:
        rows = [r for r in rows if r["age"] <= 22]
    if "young" in filters and not slots.get("age_max"):
        rows = [r for r in rows if r["age"] <= 24]
    if "veteran" in filters:
        rows = [r for r in rows if r["age"] >= 30]
    if slots.get("age_max"):
        rows = [r for r in rows if 0 < r["age"] <= slots["age_max"]]
    if slots.get("age_min"):
        rows = [r for r in rows if r["age"] >= slots["age_min"]]
    if slots.get("ovr_min"):
        rows = [r for r in rows if r["ovr"] >= slots["ovr_min"]]
    if slots.get("cap_max"):
        rows = [r for r in rows if (r["contract"]["cap_hit_m"] or 0) <= slots["cap_max"]]
    if slots.get("role"):
        role = slots["role"].lower()
        tagged = [r for r in rows if role in (r.get("role") or "").lower() or slots["role"] in (r.get("role") or "")]
        rows = tagged or rows
    young = "prospect" in filters or "young" in filters or (slots.get("age_max") or 99) <= 23
    if slots.get("sort") in ("cap_asc", "cap_desc"):
        rows = [r for r in rows if (r["contract"]["cap_hit_m"] or 0) > 0]
        rows.sort(key=lambda r: (r["contract"]["cap_hit_m"] or 0) * (1 if slots["sort"] == "cap_asc" else -1))
        rows = rows[: int(slots.get("limit") or 12)]
        if not rows:
            return _answer("No contracts match that.")
        word = "Cheapest" if slots["sort"] == "cap_asc" else "Most expensive"
        label = POS_NAMES.get(pos, "players")
        return _answer(f"{word} {SINGULAR.get(label, label)}{' over ' + str(slots['ovr_min']) + ' overall' if slots.get('ovr_min') else ''}: "
                       f"{rows[0]['name']} ({_money(rows[0]['contract']['cap_hit_m'])}, {rows[0]['ovr']:.0f} OVR).",
                       rows=rows, extra=("cap", "status", "pts"))
    if slots.get("direction") == "mid" and rows:
        rows = [r for r in rows if r["level"] == "NHL"]
        rows.sort(key=lambda r: r["ovr"])
        mid = len(rows) // 2
        rows = rows[max(0, mid - 4): mid + 5]
        m = rows[len(rows) // 2]
        return _answer(f"The most middle-of-the-road NHL player: {m['name']} ({m['ovr']:.0f} OVR, the league median).",
                       rows=rows, extra=("gp", "pts", "cap"), analysis=["The nine players either side of the median overall."])
    sign = 1 if slots.get("direction") == "low" and not slots.get("both") else -1
    rows.sort(key=lambda r: sign * ((r["pot"] * 0.6 + r["ovr"] * 0.4) if young else r["ovr"]))
    if slots.get("both") and len(rows) >= 2:
        best, worst = rows[0], rows[-1]
        label = POS_NAMES.get(pos, "players")
        both_rows = rows[:5] + rows[-5:][::-1]
        return _answer(f"Best {SINGULAR.get(label, label)} in the league: {best['name']} ({best['ovr']:.0f} OVR). "
                       f"Worst: {worst['name']} ({worst['ovr']:.0f} OVR).",
                       analysis=[f"Top five then bottom five {label} on NHL rosters, by overall.",
                                 f"A {best['ovr'] - worst['ovr']:.0f}-point gap between them."],
                       rows=both_rows, extra=("gp", "pts", "cap"))
    rows = rows[: int(slots.get("limit") or 12)]
    label = POS_NAMES.get(pos, "players")
    if not rows:
        return _answer(f"No {label} match those filters right now.", analysis=["Try widening the age or cap limit."])
    bits = []
    if slots.get("age_max") and slots.get("age_min") == slots.get("age_max"):
        bits.append(f"aged {slots['age_max']}")
    elif slots.get("age_max"):
        bits.append(f"aged {slots['age_max']} or under")
    if slots.get("age_min") and slots.get("age_min") != slots.get("age_max"):
        bits.append(f"aged {slots['age_min']} or older")
    if slots.get("ovr_min"):
        bits.append(f"rated {slots['ovr_min']}+")
    if slots.get("cap_max"):
        bits.append(f"at or under {_money(slots['cap_max'])}")
    if "free" in filters or "available" in filters:
        bits.append("unsigned")
    where = " " + ", ".join(bits) if bits else (" in the league" if not filters else "")
    one = SINGULAR.get(label, label)
    if sign > 0:
        head = f"Lowest-rated {one}{where}: {rows[0]['name']} ({rows[0]['ovr']:.0f} OVR, {rows[0]['team'] or 'unsigned'})."
    else:
        head = f"Best {one}{where}: {rows[0]['name']} ({rows[0]['ovr']:.0f} OVR / {rows[0]['pot']:.0f} POT, {rows[0]['team'] or 'unsigned'})."
    analysis = [f"Sorted by {'potential first' if young else 'overall'}.",
                f"{sum(1 for r in rows if r['team_id'] == view.user_tid)} of these are already yours."]
    return _answer(head, analysis=analysis, rows=rows, extra=("cap", "status", "pts"),
                   followups=[f"Compare {rows[0]['name']} and {rows[1]['name']}" if len(rows) > 1 else "Best trade targets",
                              "Which one fits my lineup?"])


def fit_partner(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    pos = slots.get("position") or ("D" if (slots.get("depth") or "").startswith("pair") or slots.get("role") in ("DFD", "OFD", "TWD") else None)
    if slots.get("role") == "PWF" and not slots.get("position"):
        pos = "F"
    depth = slots.get("depth") or ("pair2" if pos in ("D", "LD", "RD") else "line2")
    anchor = _resolve_players(view, slots)
    if anchor and not pos:
        pos = anchor[0]["pos"]
        if pos == "D":
            pos = "RD" if anchor[0]["shoots"] == "L" else "LD"
    group = "D" if pos in ("D", "LD", "RD") else ("G" if pos == "G" else "F")
    mine = [r for r in view.team_rows(view.user_tid) if _pos_ok(r, group)]
    mine.sort(key=lambda r: -r["ovr"])
    lo, hi = PAIR_RANK.get(depth, (2, 4))
    slot_rows = mine[lo:hi]
    bar = min((r["ovr"] for r in slot_rows), default=0.0)
    space = _cap_space(view, view.user_tid)
    cands = []
    limit_to = set(slots.get("from_list") or [])
    for pid, (tid, lv) in view.where.items():
        if limit_to and pid not in limit_to:
            continue
        if not limit_to and (tid == view.user_tid or lv not in ("NHL", "FA")):
            continue
        r = view.row(pid)
        if not _pos_ok(r, pos or group) or r["age"] > 33 or r["ovr"] <= bar:
            continue
        cap = r["contract"]["cap_hit_m"] or 0.0
        fits_cap = space is None or cap <= space + 0.05 or lv == "FA"
        if not fits_cap:
            continue
        score = (r["ovr"] - bar) - max(0.0, cap - 4.0) * 0.4 + (1.5 if lv == "FA" else 0.0)
        if group == "D" and slots.get("role") == "DFD":
            score += (r.get("blk") or 0) * 0.02
        cands.append((score, r))
    cands.sort(key=lambda kv: -kv[0])
    rows = [r for _s, r in cands[:10]]
    slot_name = {"pair1": "top pair", "pair2": "second pair", "pair3": "third pair", "top4": "top four",
                 "line1": "first line", "line2": "second line", "line3": "third line", "top6": "top six"}.get(depth, depth)
    current = ", ".join(f"{r['name']} {r['ovr']:.0f}" for r in slot_rows) or "nobody"
    analysis = [f"Your {slot_name} now: {current}. A real upgrade has to beat {bar:.0f} overall."]
    if space is not None:
        analysis.append(f"Cap space: {_money(space)}. Only players who fit under it (or unsigned) are listed.")
    if anchor:
        analysis.insert(0, f"Partner for {anchor[0]['name']} ({anchor[0]['pos']}, shoots {anchor[0]['shoots'] or '?'}).")
    if not rows:
        return _answer(f"No one available clearly upgrades your {slot_name} at your cap space.", analysis=analysis)
    cost = _money(rows[0]['contract']['cap_hit_m']) if rows[0]['contract']['cap_hit_m'] else "unsigned"
    return _answer(f"Best fit for your {slot_name}: {rows[0]['name']} ({rows[0]['team'] or 'free agent'}, {rows[0]['ovr']:.0f} OVR, {cost}).",
                   analysis=analysis, rows=rows, extra=("cap", "status", "pts"),
                   followups=[f"Should I trade for {rows[0]['name']}?", f"What about {rows[0]['name']}'s contract?"])


def trade_target(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    standings = view.standings()
    sellers = {str(r.get("team_id")) for r in sorted(standings, key=lambda r: r.get("pts", 0))[:12]}
    pos = slots.get("position")
    if not pos:
        # Weakest group on the user's roster relative to the league.
        mine = view.team_rows(view.user_tid)
        gaps = {}
        for grp in ("C", "W", "D", "G"):
            mine_top = sorted((r["ovr"] for r in mine if _pos_ok(r, grp)), reverse=True)[:2 if grp in ("D", "G") else 3]
            league = sorted((view.row(pid)["ovr"] for pid, (t, lv) in view.where.items() if lv == "NHL" and _pos_ok(view.row(pid), grp)), reverse=True)
            if mine_top and league:
                gaps[grp] = statistics.mean(league[: max(1, len(league) // 3)]) - statistics.mean(mine_top)
        pos = max(gaps, key=gaps.get) if gaps else None
    space = _cap_space(view, view.user_tid)
    rows = []
    for pid, (tid, lv) in view.where.items():
        if lv != "NHL" or tid == view.user_tid or tid not in sellers:
            continue
        r = view.row(pid)
        if not _pos_ok(r, pos) or (slots.get("age_max") and r["age"] > slots["age_max"]):
            continue
        rows.append(r)
    rows.sort(key=lambda r: -(r["ovr"] - (0.5 if (space is not None and (r["contract"]["cap_hit_m"] or 0) > space) else 0) * 5))
    rows = rows[:10]
    label = POS_NAMES.get(pos, "players")
    if not rows:
        return _answer("No clear trade targets on selling teams right now.")
    analysis = [f"Targets come from the 12 lowest teams in the standings, the likeliest sellers.",
                f"Position focus: {label}{' (your weakest group vs the league)' if not slots.get('position') else ''}."]
    if space is not None:
        over = [r["name"] for r in rows[:5] if (r["contract"]["cap_hit_m"] or 0) > space]
        if over:
            analysis.append(f"Needs salary going back (over your {_money(space)} space): {', '.join(over)}.")
    return _answer(f"Top trade target: {rows[0]['name']} ({rows[0]['team']}, {rows[0]['ovr']:.0f} OVR).", analysis=analysis,
                   rows=rows, extra=("cap", "status", "pts"), followups=[f"Should I trade for {rows[0]['name']}?", "Who fits my second pair?"])


def trade_advice(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _resolve_players(view, slots)
    if not rows:
        return _answer("Which player? Name him and I'll weigh keeping vs moving him.")
    r = rows[0]
    c = r["contract"]
    mv = _market_value(view, r["id"])
    st = _standing(view, view.user_tid)
    keep, sell, notes = 0, 0, []
    if r["age"] <= 24 and r["pot"] - r["ovr"] >= 4:
        keep += 2; notes.append(f"Still growing: {r['pot'] - r['ovr']:.0f} points of potential left at {r['age']}.")
    if r["age"] >= 31:
        sell += 1; notes.append(f"At {r['age']} his value only drops from here.")
    if mv is not None and c["cap_hit_m"]:
        diff = mv - c["cap_hit_m"]
        if diff > 0.75:
            keep += 1; notes.append(f"Underpaid: market value about {_money(mv)} vs {_money(c['cap_hit_m'])} cap hit.")
        elif diff < -0.75:
            sell += 1; notes.append(f"Overpaid: market value about {_money(mv)} vs {_money(c['cap_hit_m'])} cap hit.")
    if c["years_left"] <= 1 and c["status"] == "UFA":
        sell += 1; notes.append("Contract ends this year as a UFA: you could lose him for nothing.")
    if st:
        rank = int(st.get("league_rank") or st.get("rank") or 0)
        if rank and rank >= 22:
            sell += 1; notes.append(f"Team is {rank}th in the league, a seller's position.")
        elif rank and rank <= 8:
            keep += 1; notes.append(f"Team is {rank}th in the league, a contender keeps its core.")
    verdict = "Keep him" if keep > sell else ("Shop him" if sell > keep else "Only move him for a clear win")
    return _answer(f"{verdict}: {r['name']} ({r['age']}, {r['ovr']:.0f}/{r['pot']:.0f}, {_money(c['cap_hit_m'])} x {c['years_left']}y).",
                   analysis=notes or ["Nothing strongly pushes either way."], rows=[r], extra=("cap", "status", "pts"),
                   focus_ids=[r["id"]], followups=[f"What about {r['name']}'s contract?", "Best trade targets"])


def player_card(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _resolve_players(view, slots)
    if not rows:
        return _answer("I couldn't find that player in this save. Try the last name or sweater number (#97).")
    r = rows[0]
    c = r["contract"]
    line = (f"{r['w']} W, {r['sv_pct']:.3f} SV%" if r["pos"] == "G" and r.get("sv_pct") is not None
            else f"{r['g']} G, {r['a']} A, {r['pts']} PTS, {r['plus_minus']:+d} in {r['gp']} GP")
    analysis = [f"{r['pos']} · {r['team'] or r['level']} · age {r['age']} · shoots {r['shoots'] or '?'}",
                f"This season: {line}" + (f", {r['toi_pg']} min a night" if r.get("toi_pg") else ""),
                (f"Unsigned prospect; the {_team_label(view, r['team_id'])} hold his rights." if not c["cap_hit_m"] and r["team_id"]
                 else "Unsigned." if not c["cap_hit_m"]
                 else f"Contract: {_money(c['cap_hit_m'])} x {c['years_left']}y, {c['status'] or 'status n/a'}" + (" (NMC)" if c["nmc"] else " (NTC)" if c["ntc"] else ""))]
    if r.get("season_start_ovr"):
        d = r["ovr"] - float(r["season_start_ovr"])
        if abs(d) >= 0.5:
            analysis.append(f"Overall {'up' if d > 0 else 'down'} {abs(d):.1f} since the season started.")
    if r.get("injured"):
        analysis.append("Currently injured.")
    return _answer(f"{r['name']}: {r['ovr']:.0f} overall, {r['pot']:.0f} potential.", analysis=analysis, rows=[r],
                   extra=("gp", "pts", "cap"), focus_ids=[r["id"]],
                   followups=[f"Should I trade {r['name']}?", f"Compare {r['name']} with", f"What about his contract?"])


def player_contract(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _resolve_players(view, slots)
    if not rows:
        return _answer("Whose contract? Name the player.")
    r = rows[0]
    c = r["contract"]
    if not c["cap_hit_m"]:
        holder = f"The {_team_label(view, r['team_id'])} hold his rights." if r["team_id"] else "He is a free agent."
        return _answer(f"{r['name']} isn't signed to a contract.", analysis=[holder], rows=[r], extra=("status",), focus_ids=[r["id"]])
    mv = _market_value(view, r["id"])
    analysis = [f"Cap hit {_money(c['cap_hit_m'])}, {c['years_left']} year{'s' if c['years_left'] != 1 else ''} left"
                + (f" (through {c['expiry_year']})" if c.get("expiry_year") else "") + ".",
                f"Expiry status: {c['status'] or 'unknown'}." + (" No-move clause." if c["nmc"] else " No-trade clause." if c["ntc"] else "")]
    if mv is not None:
        verdict = "fair" if abs(mv - c["cap_hit_m"]) <= 0.75 else ("a bargain" if mv > c["cap_hit_m"] else "an overpay")
        analysis.append(f"Market value about {_money(mv)}, so the deal is {verdict}.")
    if c["years_left"] <= 1:
        analysis.append("Up for a new deal this offseason." + (" RFA: you keep his rights." if c["status"] == "RFA" else ""))
    return _answer(f"{r['name']}: {_money(c['cap_hit_m'])} x {c['years_left']}y, {c['status'] or 'status n/a'}.",
                   analysis=analysis, rows=[r], extra=("cap", "status"), focus_ids=[r["id"]],
                   followups=[f"Should I trade {r['name']}?", "Worst contracts on my team"])


def compare(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = _resolve_players(view, slots)
    if len(rows) < 2 and len(slots.get("teams") or []) >= 2:
        tids = [str(x) for x in slots["teams"][:4]]
        st = {str(r.get("team_id")): r for r in view.standings()}
        table = {"cols": ["Team", "GP", "W", "L", "OTL", "PTS", "GF", "GA", "Best player"], "rows": []}
        for tid in tids:
            r = st.get(tid, {})
            best = max(view.team_rows(tid), key=lambda x: x["ovr"], default=None)
            table["rows"].append([_team_label(view, tid), r.get("gp"), r.get("w"), r.get("l"), r.get("otl"), r.get("pts"),
                                  r.get("gf"), r.get("ga"), f"{best['name']} ({best['ovr']:.0f})" if best else "—"])
        lead = max(tids, key=lambda x: (int(st.get(x, {}).get("pts") or 0),
                                        int(st.get(x, {}).get("gf") or 0) - int(st.get(x, {}).get("ga") or 0)))
        return _answer(f"{_team_label(view, lead)} are ahead in the standings.", table=table)
    if len(rows) < 2 and slots.get("from_list"):
        rows = [view.row(pid) for pid in slots["from_list"][:4] if pid in view.players]
    if len(rows) < 2:
        return _answer("Name two players to compare, e.g. “compare McDavid and Matthews”.")
    rows = rows[:4]
    angle = slots.get("angle")
    def score(r: Dict[str, Any]) -> float:
        if angle == "defense":
            return (r.get("plus_minus") or 0) * 0.5 + (r.get("blk") or 0) * 0.05 + (r.get("hit") or 0) * 0.02 + r["ovr"] * 0.3
        if angle == "offense":
            gp = max(1, r.get("gp") or 1)
            return (r.get("pts") or 0) / gp * 40 + r["ovr"] * 0.3
        return r["ovr"]
    if angle == "upside":
        top = max(rows, key=lambda r: r["pot"] - r["ovr"])
        return _answer(f"{top['name']} has more upside ({top['pot'] - top['ovr']:+.0f} to potential {top['pot']:.0f}).",
                       analysis=[", ".join(f"{r['name']} {r['ovr']:.0f}/{r['pot']:.0f}" for r in rows)], rows=rows,
                       extra=("gp", "pts"), focus_ids=[r["id"] for r in rows])
    if angle == "age":
        young_ = min(rows, key=lambda r: r["age"])
        return _answer(f"{young_['name']} is younger ({young_['age']}).",
                       analysis=[", ".join(f"{r['name']} {r['age']}" for r in rows)], rows=rows, extra=("gp", "pts", "cap"),
                       focus_ids=[r["id"] for r in rows])
    best = max(rows, key=score)
    what = {"defense": "defensively (plus/minus, blocks, hits, overall)", "offense": "offensively (points per game, overall)"}.get(angle, "on overall")
    analysis = []
    a, b = rows[0], rows[1]
    analysis.append(f"Overall {a['ovr']:.0f} vs {b['ovr']:.0f}; potential {a['pot']:.0f} vs {b['pot']:.0f}; age {a['age']} vs {b['age']}.")
    analysis.append(f"This season {a['pts']} vs {b['pts']} points in {a['gp']} vs {b['gp']} games; {a['plus_minus']:+d} vs {b['plus_minus']:+d}.")
    analysis.append(f"Cap hit {_money(a['contract']['cap_hit_m'])} vs {_money(b['contract']['cap_hit_m'])}.")
    return _answer(f"{best['name']} comes out ahead {what}.", analysis=analysis, rows=rows,
                   extra=("gp", "pts", "plus_minus", "blk", "cap"), focus_ids=[r["id"] for r in rows],
                   followups=["Who's better defensively?", "Who's better offensively?", "Who has more potential?"])


def leaders_season(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    if "hat trick" in str(ctx.get("raw", "")):
        return _answer("Hat tricks aren't counted per player yet, so I can't rank them.",
                       analysis=["Goal leaders are the closest thing I can show."], followups=["Most goals this season"])
    stat = slots.get("stat") or "pts"
    if slots.get("scope") == "last_season":
        return _archive_leaders(view, slots, last=True)
    pos = slots.get("position") or ("G" if stat in ("w", "so", "sv_pct") else None)
    rows = [view.row(pid) for pid, (tid, lv) in view.where.items() if lv == "NHL"]
    if slots.get("self") or slots.get("teams"):
        tid = _team_id(view, slots)
        rows = [r for r in rows if r["team_id"] == tid]
    rows = [r for r in rows if _pos_ok(r, pos) and (r.get("gp") or 0) > 0]
    key = stat if stat in ("g", "a", "pts", "plus_minus", "pim", "hit", "blk", "sog", "w", "so", "sv_pct", "gp", "fo_pct") else "pts"
    if key == "fo_pct":
        rows = [r for r in rows if (r.get("fo_total") or 0) >= 40]
    if key == "sv_pct":
        rows = [r for r in rows if r.get("sv_pct") is not None and (r.get("gp") or 0) >= 5]
    low = slots.get("direction") == "low"
    rows.sort(key=lambda r: (r.get(key) or 0), reverse=not low)
    rows = rows[: int(slots.get("limit") or 10)]
    name = STAT_NAMES.get(key, key)
    if not rows:
        return _answer(f"No {name} recorded yet this season.")
    top = rows[0]
    scope = f" on the {_team_label(view, _team_id(view, slots))}" if (slots.get("self") or slots.get("teams")) else " in the league"
    return _answer(f"{'Fewest' if low else 'Most'} {name}{scope} this season: {top['name']} ({top.get(key)}).",
                   analysis=[f"{top['name']} has {top.get(key)} {name} in {top['gp']} games."], rows=rows,
                   extra=(key, "gp") if key != "gp" else ("gp",), followups=[f"Most {name} ever", "Who carried us?"])


def _archive_leaders(view: LiveView, slots: Dict[str, Any], last: bool = False) -> Answer:
    doc = archive(view.session)
    stat = slots.get("stat") or "pts"
    seasons = sorted((doc.get("seasons") or {}).keys(), key=int)
    if not seasons:
        return _answer("No finished seasons in this save yet, so there's no history to rank.",
                       analysis=["History starts once the first season is finalised."])
    year = seasons[-1]
    lines = [ln for ln in doc["seasons"][year].get("player_lines") or [] if (ln.get("scope") or "regular") == "regular"]
    lines.sort(key=lambda ln: -(ln.get(stat) or 0))
    table = {"cols": ["Player", "Pos", "Team", STAT_NAMES.get(stat, stat).title(), "GP"],
             "rows": [[ln.get("name"), ln.get("pos"), view.team_meta.get(str(ln.get("team_id")), {}).get("abbr", ln.get("team_id")), ln.get(stat), ln.get("gp")] for ln in lines[:10]],
             "ids": [ln.get("id") for ln in lines[:10]]}
    if not lines:
        return _answer(f"No player lines stored for {year}.")
    return _answer(f"{year} leader in {STAT_NAMES.get(stat, stat)}: {lines[0].get('name')} ({lines[0].get(stat)}).", table=table)


def leaders_career(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    stat = slots.get("stat") or "pts"
    stat = stat if stat in ("gp", "g", "a", "pts", "plus_minus", "pim", "sog", "w", "so", "hit", "blk") else "pts"
    archive(view.session)  # keeps the archive file current; legacy is the record book it mirrors
    try:
        from services.legacy_hall import leaders

        team = _team_id(view, slots) if (slots.get("self") or slots.get("teams") or "franchise" in str(ctx.get("raw", ""))) else None
        pos = slots.get("position")
        res = leaders(view.session, stat=stat, scope="career", team_id=team, position=pos if pos in ("C", "D", "G", "LW", "RW") else None, limit=10)
    except Exception:
        res = {"rows": []}
    rows = res.get("rows") or []
    name = STAT_NAMES.get(stat, stat)
    if not rows:
        return _answer(f"No career {name} on record yet.", analysis=[res.get("coverage_note") or ""])
    table = {"cols": ["Player", "Pos", name.title(), "GP", "Seasons"],
             "rows": [[r.get("name"), r.get("position"), r.get("value"), r.get("gp"), r.get("seasons")] for r in rows],
             "ids": [r.get("player_id") for r in rows]}
    where = f" for the {_team_label(view, team)}" if team else ""
    return _answer(f"All-time {name} leader{where}: {rows[0].get('name')} ({rows[0].get('value')}).",
                   analysis=[res.get("coverage_note") or ""], table=table, followups=[f"Most {name} this season"])


def streak(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    tid = _team_id(view, slots)
    loss = slots.get("streak_kind") == "loss"
    games = view.team_games(tid)
    best, run, start, best_span, cur = 0, 0, "", ("", ""), 0
    for date, won, _gf, _ga in games:
        hit = (not won) if loss else won
        if hit:
            run += 1
            start = start or date
            if run > best:
                best, best_span = run, (start, date)
        else:
            run, start = 0, ""
    for _d, won, _gf, _ga in reversed(games):
        if won == games[-1][1]:
            cur += 1
        else:
            break
    kind = "losing" if loss else "winning"
    team = _team_label(view, tid)
    if not games:
        return _answer(f"No games played yet this season for the {team}.")
    cur_txt = f"Current run: {cur} straight {'wins' if games[-1][1] else 'losses'}."
    analysis = [cur_txt, f"{len(games)} games this season, {sum(1 for g in games if g[1])} wins."]
    analysis.append("Per-game results are kept for the current season; earlier seasons store totals only.")
    return _answer(f"Longest {kind} streak this season for the {team}: {best} games" + (f" ({best_span[0]} to {best_span[1]})." if best else "."),
                   analysis=analysis, followups=["Why are we losing?" if loss else "Who carried us?"])


def diagnose(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    tid = _team_id(view, slots)
    standings = view.standings()
    n = max(1, len(standings))
    me = _standing(view, tid)
    if not me or not me.get("gp"):
        return _answer("No games played yet, so there's nothing to diagnose.")
    def rank(key: str, high_good: bool = True) -> int:
        vals = sorted(standings, key=lambda r: float(r.get(key) or 0), reverse=high_good)
        return next((i + 1 for i, r in enumerate(vals) if str(r.get("team_id")) == str(tid)), n)
    gp = float(me.get("gp") or 1)
    gf_r, ga_r = rank("gf", True), rank("ga", False)
    pp_r, pk_r = rank("pp_pct", True), rank("pk_pct", True)
    reasons = []
    def mark(r: int) -> str:
        return f"{r}{'st' if r == 1 else 'nd' if r == 2 else 'rd' if r == 3 else 'th'}"
    reasons.append((gf_r, f"Scoring: {float(me.get('gf') or 0) / gp:.2f} goals a game ({mark(gf_r)} of {n})."))
    reasons.append((ga_r, f"Defense: {float(me.get('ga') or 0) / gp:.2f} against a game ({mark(ga_r)} of {n})."))
    reasons.append((pp_r, f"Power play: {float(me.get('pp_pct') or 0) * 100:.1f}% ({mark(pp_r)})."))
    reasons.append((pk_r, f"Penalty kill: {float(me.get('pk_pct') or 0) * 100:.1f}% ({mark(pk_r)})."))
    goalies = [r for r in view.team_rows(tid) if r["pos"] == "G" and r.get("sv_pct") is not None and r.get("gp")]
    if goalies:
        g = max(goalies, key=lambda r: r["gp"])
        lg_sv = [view.row(pid).get("sv_pct") for pid, (t, lv) in view.where.items() if lv == "NHL" and view.row(pid)["pos"] == "G" and view.row(pid).get("sv_pct") and view.row(pid).get("gp", 0) >= 5]
        avg = statistics.mean(lg_sv) if lg_sv else 0.905
        reasons.append((n if g["sv_pct"] < avg - 0.008 else n // 2, f"Goaltending: {g['name']} at {g['sv_pct']:.3f} vs a league average of {avg:.3f}."))
    injured = [r["name"] for r in view.team_rows(tid) if r.get("injured")]
    if injured:
        weight = n - 2 if len(injured) >= 3 else n // 3
        reasons.append((weight, f"Injuries: {len(injured)} out ({', '.join(injured[:4])})."))
    focus = slots.get("focus")
    prefixes = {"pk": ("Penalty",), "pp": ("Power",), "goalie": ("Goaltending",)}
    asked_ok = None
    if focus in prefixes:
        asked = next((kv for kv in reasons if kv[1].startswith(prefixes[focus])), None)
        if asked and asked[0] <= n // 3:
            asked_ok = asked[1]  # what they asked about is actually fine
            reasons.sort(key=lambda kv: -kv[0])
        else:
            reasons.sort(key=lambda kv: (0 if kv[1].startswith(prefixes[focus]) else 1, -kv[0]))
    elif focus == "offense":
        reasons.sort(key=lambda kv: (0 if kv[1].startswith(("Scoring", "Power")) else 1, -kv[0]))
    elif focus == "defense":
        reasons.sort(key=lambda kv: (0 if kv[1].startswith(("Defense", "Penalty", "Goaltending")) else 1, -kv[0]))
    else:
        reasons.sort(key=lambda kv: -kv[0])
    worst = reasons[0][1]
    record = f"{me.get('w')}-{me.get('l')}-{me.get('otl')}, {me.get('pts')} pts ({mark(int(me.get('league_rank') or me.get('rank') or 0) or n)} overall)"
    fix = {"Scoring": "a scoring winger or centre, or more power-play time for your best shooters",
           "Defense": "a top-four defenseman or tighter defensive pairings", "Power": "a power-play quarterback or new PP1 unit",
           "Penalty": "defensive forwards on the PK units", "Goaltending": "a goalie upgrade or more starts for the backup",
           "Injuries": "call-ups from the AHL to cover the holes"}
    key = worst.split(":")[0].split()[0]
    if asked_ok:
        return _answer(f"That isn't actually the problem. {asked_ok} The bigger issue: {worst}",
                       analysis=[r for _k, r in reasons] + [f"Most direct fix: {fix.get(key, 'targeted depth')}."],
                       followups=["Best trade targets", "Who fits my second pair?"])
    return _answer(f"{_team_label(view, tid)} ({record}). Biggest problem: {worst}",
                   analysis=[r for _k, r in reasons] + [f"Most direct fix: {fix.get(key, 'targeted depth')}."],
                   followups=["Best trade targets", "Who fits my second pair?", "Which goalie should start?"])


def standings(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = view.standings()
    if not rows:
        return _answer("Standings aren't available yet.")
    rows = sorted(rows, key=lambda r: int(r.get("league_rank") or r.get("rank") or 99))
    me = _standing(view, view.user_tid)
    raw = str(ctx.get("raw", ""))
    if slots.get("direction") == "low" and not slots.get("both"):
        last = rows[-1]
        table_rows = [[r.get("league_rank") or r.get("rank"), r.get("name"), r.get("gp"), r.get("w"), r.get("l"), r.get("otl"),
                       r.get("pts"), r.get("gf"), r.get("ga")] for r in rows[::-1][:10]]
        return _answer(f"Worst team: the {last.get('name')} ({last.get('pts')} points in {last.get('gp')} games).",
                       table={"cols": ["#", "Team", "GP", "W", "L", "OTL", "PTS", "GF", "GA"], "rows": table_rows})
    if slots.get("direction") == "mid":
        mid = len(rows) // 2
        m = rows[mid]
        return _answer(f"The most average team: the {m.get('name')} ({m.get('pts')} points, {mid + 1} of {len(rows)}).",
                       table={"cols": ["#", "Team", "GP", "W", "L", "OTL", "PTS", "GF", "GA"],
                              "rows": [[r.get("league_rank") or r.get("rank"), r.get("name"), r.get("gp"), r.get("w"), r.get("l"),
                                        r.get("otl"), r.get("pts"), r.get("gf"), r.get("ga")] for r in rows[max(0, mid - 4): mid + 5]]})
    if "tank" in raw and me:
        rank = int(me.get("league_rank") or me.get("rank") or 0)
        verdict = ("You're near the bottom already; a lottery spot is realistic." if rank >= len(rows) - 6
                   else "You're in the playoff mix; tanking would throw away a real chance." if rank <= len(rows) // 2
                   else "You're between the race and the lottery; selling veterans gets you picks either way.")
        return _answer(f"You're {rank} of {len(rows)} with {me.get('pts')} points. {verdict}",
                       analysis=["The lottery is for the 16 non-playoff teams; the worst records get the best odds."])
    if slots.get("both"):
        return _answer(f"Best team: the {rows[0].get('name')} ({rows[0].get('pts')} pts). Worst: the {rows[-1].get('name')} ({rows[-1].get('pts')} pts).",
                       table={"cols": ["#", "Team", "GP", "W", "L", "OTL", "PTS", "GF", "GA"],
                              "rows": [[r.get("league_rank") or r.get("rank"), r.get("name"), r.get("gp"), r.get("w"), r.get("l"),
                                        r.get("otl"), r.get("pts"), r.get("gf"), r.get("ga")] for r in rows[:5] + rows[-5:]]})
    table = {"cols": ["#", "Team", "GP", "W", "L", "OTL", "PTS", "GF", "GA"],
             "rows": [[r.get("league_rank") or r.get("rank"), r.get("name"), r.get("gp"), r.get("w"), r.get("l"), r.get("otl"), r.get("pts"), r.get("gf"), r.get("ga")] for r in rows]}
    analysis = []
    if me:
        conf = [r for r in rows if r.get("conference") == me.get("conference")]
        pos = next((i + 1 for i, r in enumerate(conf) if r.get("team_id") == me.get("team_id")), None)
        if pos:
            line = conf[7] if len(conf) >= 8 else None
            gap = (int(me.get("pts") or 0) - int(line.get("pts") or 0)) if line else 0
            analysis.append(f"You're {pos} in the {me.get('conference')} Conference; the 8th spot is {'+' if gap >= 0 else ''}{gap} points away." if line else f"You're {pos} in your conference.")
    head = (f"Best team: the {rows[0].get('name')} ({rows[0].get('pts')} points)." if "best" in raw
            else f"{rows[0].get('name')} lead the league with {rows[0].get('pts')} points.")
    return _answer(head, analysis=analysis, table=table,
                   followups=["Why are we losing?", "Longest win streak"])


def champions(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    doc = archive(view.session)
    seasons = sorted(((int(k), v) for k, v in (doc.get("seasons") or {}).items() if v.get("champion_id")), reverse=True)
    if not seasons:
        return _answer("No Stanley Cup has been awarded in this save yet.")
    if slots.get("years_ago"):
        target = view.season - int(slots["years_ago"])
        hit = [s for s in seasons if s[0] == target] or [s for s in seasons if s[0] <= target][:1]
        if hit:
            y, s = hit[0]
            return _answer(f"{y}: the {_team_label(view, str(s['champion_id']))} won the Stanley Cup.",
                           analysis=[f"They beat the {_team_label(view, str(s.get('finalist_id')))} in the final." if s.get("finalist_id") else ""])
        return _answer(f"The save has no Cup winner on record for {target}.", analysis=[f"Champions on record start in {seasons[-1][0]}."])
    table = {"cols": ["Season", "Champion", "Finalist"],
             "rows": [[y, _team_label(view, str(s["champion_id"])), _team_label(view, str(s.get("finalist_id"))) if s.get("finalist_id") else "—"] for y, s in seasons]}
    mine = sum(1 for _y, s in seasons if str(s["champion_id"]) == view.user_tid)
    return _answer(f"Most recent champion: {_team_label(view, str(seasons[0][1]['champion_id']))} ({seasons[0][0]}).",
                   analysis=[f"Your Cups in this save: {mine}."], table=table)


def awards(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    published = int(getattr(view.session, "awards_published_through", 0) or 0)
    doc = archive(view.session)
    rows = []
    for y, s in sorted((doc.get("seasons") or {}).items(), key=lambda kv: -int(kv[0])):
        if int(y) > published:
            continue  # results stay hidden until the awards are announced
        for a in s.get("awards") or []:
            rows.append([y, a.get("award") or a.get("award_name"), a.get("name") or a.get("player_name") or a.get("winner"), a.get("team_id") or a.get("team")])
    players = _resolve_players(view, slots)
    if players:
        name = players[0]["name"]
        rows = [r for r in rows if str(r[2] or "") == name]
        if not rows:
            return _answer(f"{name} hasn't won an award in this save.")
    raw = str(ctx.get("raw", "")).lower()
    for trophy in ("hart", "norris", "vezina", "calder", "selke", "art ross", "rocket", "ted lindsay", "conn smythe", "jack adams"):
        if trophy in raw:
            rows = [r for r in rows if trophy in str(r[1] or "").lower()]
            break
    if not rows:
        return _answer("No awards have been announced in this save yet.", analysis=["Winners appear here after the awards ceremony, never before."])
    return _answer(f"{rows[0][0]} {rows[0][1]}: {rows[0][2]}.", table={"cols": ["Season", "Award", "Winner", "Team"], "rows": rows[:40]})


def draft_history(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    archive(view.session)
    try:
        from services.legacy_hall import draft_history as dh

        tid = None if slots.get("teams") is None and not slots.get("self") and "my" not in str(ctx.get("raw", "")) else _team_id(view, slots)
        res = dh(view.session, team_id=tid)
    except Exception:
        res = {"classes": []}
    classes = res.get("classes") or []
    if not classes:
        return _answer("No drafts on record in this save yet.", analysis=["Classes appear after the first entry draft."])
    raw = str(ctx.get("raw", "")).lower()
    picks = [pk for c in classes for pk in c.get("picks") or []]
    if "first round" in raw or "1st round" in raw:
        picks = [pk for pk in picks if int(pk.get("round") or 0) == 1]
    table = {"cols": ["Year", "Rd", "Pick", "Player", "Team", "NHL GP", "NHL PTS"],
             "rows": [[pk.get("draft_year"), pk.get("round"), pk.get("pick") or pk.get("overall"), pk.get("name") or pk.get("player_name"),
                       view.team_meta.get(str(pk.get("team_id")), {}).get("abbr", pk.get("team_id")), pk.get("nhl_gp"), pk.get("nhl_pts")] for pk in picks[:60]],
             "ids": [pk.get("player_id") for pk in picks[:60]]}
    best = max(classes, key=lambda c: (c.get("class_pts") or 0, c.get("class_gp") or 0))
    return _answer(f"Best draft class: {best.get('label')} ({best.get('class_pts')} NHL points from {best.get('size')} picks).",
                   analysis=[res.get("rule") or ""], table=table)


def top_prospects(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    pool = slots.get("from_list")
    league_wide = "league" in str(ctx.get("raw", "")).lower()
    if pool:
        rows = [view.row(pid) for pid in pool if pid in view.players]
    else:
        rows = [view.row(pid) for pid, (tid, lv) in view.where.items()
                if (lv in ("Prospect", "AHL", "ECHL") or (lv == "NHL" and view.row(pid)["age"] <= 21))
                and (league_wide or tid == view.user_tid)]
    pos = slots.get("position")
    rows = [r for r in rows if _pos_ok(r, pos)]
    angle = slots.get("angle") or slots.get("focus")
    develops = any(w in str(ctx.get("raw", "")).lower() for w in ("develop", "grow", "potential", "project", "ceiling"))
    ready = any(w in str(ctx.get("raw", "")).lower() for w in ("closest", "ready", "nhl ready", "call up", "soonest"))
    if ready:
        rows.sort(key=lambda r: -r["ovr"])
        top = rows[0] if rows else None
        if top:
            return _answer(f"Closest to the NHL: {top['name']} ({top['ovr']:.0f} overall at {top['age']}).",
                           rows=rows[:10], extra=("gp", "pts"), analysis=["Ranked by current overall."])
    def key(r: Dict[str, Any]) -> float:
        base = r["pot"] * (0.75 if develops else 0.55) + r["ovr"] * (0.25 if develops else 0.45) - max(0, r["age"] - 20) * 0.6
        if angle == "defense":
            base += 2.0 if r["pos"] in ("D", "LD", "RD") else 0.0
        return base
    rows.sort(key=lambda r: -key(r))
    rows = rows[: int(slots.get("limit") or 12)]
    if not rows:
        return _answer("No prospects in the system match that.")
    top = rows[0]
    head = (f"Projects best: {top['name']} ({top['pot']:.0f} POT at {top['age']})." if develops or pool
            else f"Top prospect{' in the league' if league_wide else ''}: {top['name']} ({top['ovr']:.0f}/{top['pot']:.0f}, {top['age']}).")
    return _answer(head, analysis=["Ranked by potential, then current overall, with younger players favoured."],
                   rows=rows, extra=("gp", "pts"), followups=["Which one develops best?", "Who is ready for the NHL?"])


def most_improved(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    rows = [view.row(pid) for pid, (tid, lv) in view.where.items() if lv in ("NHL", "AHL", "Prospect")]
    if slots.get("self") or slots.get("teams"):
        tid = _team_id(view, slots)
        rows = [r for r in rows if r["team_id"] == tid]
    rows = [r for r in rows if r.get("season_start_ovr")]
    for r in rows:
        r["_delta"] = round(r["ovr"] - float(r["season_start_ovr"]), 1)
    low = slots.get("direction") == "low"
    rows.sort(key=lambda r: r["_delta"] if low else -r["_delta"])
    rows = [r for r in rows if (r["_delta"] < 0 if low else r["_delta"] > 0)][:12]
    if not rows:
        return _answer("No overall changes yet this season.", analysis=["Players move at their in-season checkpoints (about 20, 41 and 61 games)."])
    table = _player_table(rows, ("gp", "pts"))
    table["cols"].append("Change")
    for line, r in zip(table["rows"], rows):
        line.append(f"{r['_delta']:+.1f}")
    return _answer(f"{'Biggest drop' if low else 'Most improved'} this season: {rows[0]['name']} ({rows[0]['_delta']:+.1f} overall).",
                   table=table, cards=[_card(r) for r in rows[:6]], analysis=["Change is current overall minus the overall he started the season at."])


def contracts_value(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    raw = str(ctx.get("raw", "")).lower()
    tid = _team_id(view, slots)
    team_only = slots.get("self") or slots.get("teams") or "my" in raw
    if any(w in raw for w in ("expir", "ufa this", "resign", "upcoming", "free agents")):
        rows = [r for r in view.team_rows(tid) if r["contract"]["years_left"] <= 1]
        rows.sort(key=lambda r: -r["ovr"])
        return _answer(f"{len(rows)} contracts on the {_team_label(view, tid)} expire this year.", rows=rows, extra=("cap", "status", "pts"))
    rows = [view.row(pid) for pid, (t, lv) in view.where.items() if lv == "NHL" and (not team_only or t == tid)]
    scored = []
    for r in rows:
        cap = r["contract"]["cap_hit_m"] or 0
        if cap <= 0:
            continue
        mv = _market_value(view, r["id"])
        if mv is None:
            continue
        r["_surplus"] = round(mv - cap, 2)
        scored.append(r)
    worst = slots.get("direction") == "low" or "overpaid" in raw or "buy" in raw
    if slots.get("both"):
        scored.sort(key=lambda r: -r["_surplus"])
        if len(scored) >= 2:
            b, w = scored[0], scored[-1]
            table = _player_table(scored[:5] + scored[-5:], ("cap", "status"))
            return _answer(f"Best contract{' on your team' if team_only else ''}: {b['name']} ({_money(b['contract']['cap_hit_m'])}). "
                           f"Worst: {w['name']} ({_money(w['contract']['cap_hit_m'])}, worth about {_money(w['contract']['cap_hit_m'] + w['_surplus'])}).",
                           table=table)
    if "biggest cap" in raw or "highest paid" in raw:
        scored.sort(key=lambda r: -(r["contract"]["cap_hit_m"] or 0))
    else:
        scored.sort(key=lambda r: r["_surplus"] if worst else -r["_surplus"])
    scored = scored[:10]
    if not scored:
        return _answer("No contracts to rate.")
    table = _player_table(scored, ("cap", "status"))
    table["cols"].append("Value vs cap")
    for line, r in zip(table["rows"], scored):
        line.append(f"{r['_surplus']:+.2f}M")
    top = scored[0]
    if "biggest cap" in raw or "highest paid" in raw:
        return _answer(f"Biggest cap hit{' on your team' if team_only else ''}: {top['name']} ({_money(top['contract']['cap_hit_m'])}).",
                       table=table, cards=[_card(r) for r in scored[:6]])
    return _answer(f"{'Worst' if worst else 'Best'} contract{' on your team' if team_only else ''}: {top['name']} ({_money(top['contract']['cap_hit_m'])}, worth about {_money(top['contract']['cap_hit_m'] + top['_surplus'])}).",
                   table=table, cards=[_card(r) for r in scored[:6]], analysis=["Value is the league's market price for his rating, age and production minus his cap hit."])


def goalie_choice(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    tid = _team_id(view, slots)
    rows = [r for r in view.team_rows(tid, ("NHL",)) if r["pos"] == "G"] or         [r for r in view.team_rows(tid, ("NHL", "AHL")) if r["pos"] == "G"]
    if not rows:
        return _answer("No goalies on the roster.")
    def score(r: Dict[str, Any]) -> float:
        # Rating early, save percentage once the sample is real (.905 ~ an 85).
        rel = min(1.0, (r.get("gp") or 0) / 20.0) * 0.6
        sv_ovr = 85.0 + ((r.get("sv_pct") or 0.905) - 0.905) * 500.0
        return r["ovr"] * (1 - rel) + sv_ovr * rel
    rows.sort(key=lambda r: -score(r))
    top = rows[0]
    analysis = [f"{r['name']}: {r['ovr']:.0f} OVR, {r.get('w', 0)} W, {('%.3f' % r['sv_pct']) if r.get('sv_pct') else 'no'} SV% in {r.get('gp', 0)} GP ({r['level']})." for r in rows[:4]]
    return _answer(f"Start {top['name']}.", analysis=analysis + ["Early in the season the rating counts more; save percentage takes over as games pile up."],
                   rows=rows, extra=("gp", "w", "sv_pct"), focus_ids=[top["id"]])


def carried_us(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    tid = _team_id(view, slots)
    raw = str(ctx.get("raw", "")).lower()
    cold = slots.get("direction") == "low" or any(w in raw for w in ("cold", "slump", "struggl", "bad", "losing"))
    rows = [r for r in view.team_rows(tid) if (r.get("gp") or 0) > 0 and r["pos"] != "G"]
    if not rows:
        return _answer("No games played yet.")
    exp = {r["id"]: max(0.1, (r["ovr"] - 55) / 45.0) for r in rows}
    for r in rows:
        r["_ppg"] = (r.get("pts") or 0) / max(1, r["gp"])
        r["_vs"] = r["_ppg"] - exp[r["id"]]
    rows.sort(key=lambda r: r["_vs"] if cold else -(r.get("pts") or 0) - r["_vs"] * 5)
    top = rows[0]
    if cold:
        return _answer(f"Coldest relative to his rating: {top['name']} ({top['pts']} pts in {top['gp']}).", rows=rows[:8], extra=("gp", "pts", "plus_minus"),
                       analysis=["Compared with what a player of his overall usually scores."])
    return _answer(f"{top['name']} has carried the {_team_label(view, tid)}: {top['pts']} points in {top['gp']} games.",
                   rows=rows[:8], extra=("gp", "g", "a", "pts", "plus_minus"), analysis=[f"{top['_ppg']:.2f} points a game."],
                   followups=["Who's cold?", "Why are we losing?"])


def injuries(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    players = _resolve_players(view, slots)
    if players:
        r = players[0]
        return _answer(f"{r['name']} is {'injured' if r.get('injured') else 'healthy'}.", rows=[r], extra=("gp", "pts"))
    try:
        from services.franchise_sim import _build_injuries_payload

        inj = _build_injuries_payload(view.session, limit=60)
    except Exception:
        inj = []
    league_wide = "league" in str(ctx.get("raw", "")) or bool(slots.get("teams"))
    note = []
    if not league_wide:
        mine = [i for i in inj if str(i.get("team_id")) == view.user_tid]
        if mine:
            inj = mine
        else:
            note = ["Nobody on your team is hurt. League-wide list below."]
            league_wide = True
    elif slots.get("teams"):
        inj = [i for i in inj if str(i.get("team_id")) in {str(x) for x in slots["teams"]}]
    if not inj:
        return _answer("No injuries right now.")
    team_only = not league_wide
    table = {"cols": ["Player", "Team", "Injury", "Games left"],
             "rows": [[i.get("player_name"), view.team_meta.get(str(i.get("team_id")), {}).get("abbr", i.get("team_id")), i.get("injury") or i.get("type") or i.get("label"), i.get("games_remaining")] for i in inj[:30]],
             "ids": [i.get("player_id") for i in inj[:30]]}
    return _answer(f"{len(inj)} injured{' on your team' if team_only else ' around the league'}.", table=table, analysis=note)


def transactions(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    tid = _team_id(view, slots) if (slots.get("self") or slots.get("teams") or "my" in str(ctx.get("raw", ""))) else None
    try:
        from services.legacy_hall import trade_history

        res = trade_history(view.session, team_id=tid, limit=25)
    except Exception:
        res = {"trades": []}
    trades = res.get("trades") or []
    if not trades:
        return _answer("No trades on record" + (f" for the {_team_label(view, tid)}." if tid else "."))
    rows = []
    for t in trades:
        teams = [view.team_meta.get(str(x), {}).get("abbr", x) for x in t.get("teams") or []]
        moved = ", ".join(f"{p.get('name')} → {view.team_meta.get(str(p.get('to_team_id')), {}).get('abbr', '?')}" for p in (t.get("players") or [])[:4])
        rows.append([t.get("season"), str(t.get("iso") or "")[:10], " / ".join(teams), moved or "picks only"])
    return _answer(f"{res.get('total', len(trades))} trades on record" + (f" involving the {_team_label(view, tid)}." if tid else "."),
                   table={"cols": ["Season", "Date", "Teams", "Players moved"], "rows": rows}, analysis=[res.get("note") or ""])


def team_card(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    tid = _team_id(view, slots)
    st = _standing(view, tid) or {}
    rows = sorted(view.team_rows(tid), key=lambda r: -r["ovr"])
    space = _cap_space(view, tid)
    analysis = [f"Record {st.get('w', 0)}-{st.get('l', 0)}-{st.get('otl', 0)}, {st.get('pts', 0)} pts, {st.get('league_rank') or st.get('rank') or '?'} in the league.",
                f"Goals {st.get('gf', 0)} for, {st.get('ga', 0)} against."]
    if space is not None:
        analysis.append(f"Cap space {_money(space)}.")
    return _answer(f"{_team_label(view, tid)}: best player {rows[0]['name']} ({rows[0]['ovr']:.0f})." if rows else _team_label(view, tid),
                   analysis=analysis, rows=rows[:12], extra=("gp", "pts", "cap"))


def social(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    from services.sim_history import _social_rank

    res = _social_rank(view.session, str(ctx.get("raw", "")).lower())
    return _answer(res.get("answer") or "Nothing on social yet.",
                   table={"cols": ["Date", "Post"], "rows": [[h.get("date"), h.get("summary")] for h in res.get("hits") or []]})


def help_answer(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    return _answer("Ask me anything about this save, however you type it.", analysis=[
        "Your team: “whos my best dman”, “why are we losing”, “who carried us”, “which goalie”",
        "Players: “mcdavid”, “#34 contract”, “compare matthews and mcdavid”, “should i trade him”",
        "Market: “best ufa d under 30”, “who fits my 2nd pair”, “best trade targets”, “worst contract”",
        "History: “most goals ever”, “who won the cup 5 yrs ago”, “best draft class”, “longest win streak”",
        "Other languages work too: “qui est mon meilleur défenseur”, “wer ist mein bester verteidiger”.",
    ], followups=["Why are we losing?", "Who's my best D?", "Best trade targets", "Top prospects"])


def hall_of_fame(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    archive(view.session)
    try:
        from services.legacy_hall import league_overview

        hof = league_overview(view.session).get("hall_of_fame") or []
    except Exception:
        hof = []
    if not hof:
        return _answer("Nobody has been inducted into the Hall of Fame in this save yet.",
                       analysis=["Retired players are inducted after their careers end, once the save has history."])
    table = {"cols": ["Player", "Inducted", "Position"], "rows": [[h.get("name"), h.get("inducted_season"), h.get("position")] for h in hof]}
    return _answer(f"{len(hof)} Hall of Famers in this save.", table=table)


def unsupported(view: LiveView, slots: Dict[str, Any], ctx: Dict[str, Any]) -> Answer:
    if slots.get("prediction"):
        rows = sorted(view.standings(), key=lambda r: int(r.get("league_rank") or r.get("rank") or 99))
        if rows:
            return _answer("I can't see the future, but the standings point at a favourite: "
                           f"the {rows[0].get('name')} ({rows[0].get('pts')} points).",
                           analysis=[f"Next: {', '.join(str(r.get('name')) for r in rows[1:4])}."],
                           followups=["Standings", "Why are we losing?"])
    return _answer("That isn't something this save records, so I can't answer it.",
                   analysis=["I answer from your save: players, teams, stats, contracts, standings, history."],
                   followups=["Why are we losing?", "Who's my best player?", "Best trade targets"])


HANDLERS: Dict[str, Callable[[LiveView, Dict[str, Any], Dict[str, Any]], Answer]] = {
    "best_on_team": best_on_team, "find_players": find_players, "fit_partner": fit_partner,
    "trade_target": trade_target, "trade_advice": trade_advice, "player_card": player_card,
    "player_contract": player_contract, "compare": compare, "leaders_season": leaders_season,
    "leaders_career": leaders_career, "streak": streak, "diagnose": diagnose, "standings": standings,
    "champions": champions, "awards": awards, "draft_history": draft_history, "top_prospects": top_prospects,
    "most_improved": most_improved, "contracts_value": contracts_value, "goalie_choice": goalie_choice,
    "carried_us": carried_us, "injuries": injuries, "transactions": transactions, "team_card": team_card,
    "social": social, "help": help_answer, "unsupported": unsupported, "hall_of_fame": hall_of_fame,
}

INTENT_LABELS = {
    "best_on_team": "Best on your team", "find_players": "Search players league-wide", "fit_partner": "Lineup fit",
    "trade_target": "Trade targets", "trade_advice": "Trade advice", "player_card": "Player profile",
    "player_contract": "Contract details", "compare": "Compare players", "leaders_season": "Season leaders",
    "leaders_career": "All-time records", "streak": "Streaks", "diagnose": "Team diagnosis", "standings": "Standings",
    "champions": "Cup winners", "awards": "Awards", "draft_history": "Draft history", "top_prospects": "Top prospects",
    "most_improved": "Development", "contracts_value": "Contract value", "goalie_choice": "Goalie choice",
    "carried_us": "Who's carrying the team", "injuries": "Injuries", "transactions": "Trades and moves",
    "team_card": "Team profile", "social": "Social media", "help": "Help", "unsupported": "Not in the save", "hall_of_fame": "Hall of Fame",
}
