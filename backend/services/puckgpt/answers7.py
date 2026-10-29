"""League events and opinion questions.

Two jobs the fixed handlers couldn't do:

* EVENTS — "what happened around the league", "who got hurt this week", "latest
  trades", "any news on Edmonton". One searchable corpus is built from what the save
  records (trades, injuries, storyline/news items, legacy milestones), then scored by
  the question's words, named teams/players, event type and time words.

* VERDICTS — subjective questions ("biggest draft bust", "who's to blame for missing
  the playoffs", "most disappointing player", "who won that trade"). Each verdict
  states the yardstick it used, ranks every candidate on it, and shows the runners-up,
  so the opinion is reasoned from the save rather than invented.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .answers import _answer, _team_label
from .data import LiveView

Answer = Dict[str, Any]

VERDICTS: List[Tuple[str, str]] = [
    (r"\b(draft )?bust(s|ed)?\b|\bbiggest (draft )?(miss|whiff|mistake)\b|\bwasted (a |the )?(high |top |first[- ]round )?pick\b", "bust"),
    (r"\bdraft steals?\b|\bbiggest steal\b.*\bdraft\b|\bbest late[- ]round pick\b|\bdrafted (way )?too late\b", "steal"),
    (r"\b(reason|why) (we|they|the [a-z ]+?) (missed|miss|didn't make|did not make|aren't making|are not making|won't make)\b|\bto blame\b|\bwho('s| is) (at )?fault\b|\bwhose fault\b|\bblame\b|\bcost (us|them) the playoffs\b|\bscapegoat\b", "blame"),
    (r"\bmost disappointing\b|\bbiggest disappointment\b|\bunderachiev|\blet (us|them) down\b|\bbiggest letdown\b", "disappoint"),
    (r"\bbiggest surprise\b|\bsurprise (player|of the season)\b|\boverachiev|\bbreakout (player|star|season)\b|\bcame out of nowhere\b", "surprise"),
    (r"\b(best|worst|most lopsided|biggest|most one[- ]sided|who won|who lost|winner of)\b.*\btrades?\b(?!\s*(targets?|options?|partners?|for))|\btrade (winner|loser)s?\b|\bwho got fleeced\b"
     r"|\btrading (our|my|the|him|[a-z]+) [a-z ]{0,30}(a )?(mistake|error|good idea|bad idea|right call|wrong call)\b|\bshould(n'?t| not) have traded\b", "trade_verdict"),
]

EVENT_TRIGGERS = (r"\bwhat'?s (new|happening|going on)\b|\bwhat (happened|is happening|'s happening|went down|'s new|is new)\b|\bnews\b|\bheadlines?\b|\blatest\b|\brecent(ly)?\b.*\b(trades?|injur|signings?|moves?|news)"
                  r"|\b(who|anyone|anybody|any players?) (get|got|gets|was|were|been) (hurt|injured|traded|signed|waived|suspended)\b|\bwho('s| is) (hurt|injured)\b|\binjur(y|ies|ed)\b.*\b(around|league|this week|today|recent|lately)\b"
                  r"|\b(around|across) the league\b|\bany (news|updates?|moves?|rumou?rs?)\b|\brumou?rs?\b|\bstorylines?\b|\bmilestones?\b"
                  r"|\b(trades?|deals?|moves?) (this|last) (week|month|season|year)\b|\bwhat did [a-z ]+ do\b")

_STOP = set("the a an of to in on at for and or is was were be been are what who whom which when where why how did do does "
            "has have had any around league this that these those last latest recent recently news happen happened happening "
            "me tell show give about with from by it its their there week month season year today lately".split())

_TYPE_WORDS = {
    "trade": r"\btrades?|traded|deal|deals|swap|acquir",
    "injury": r"\binjur|hurt|ir\b|injured reserve|out for|day[- ]to[- ]day",
    "contract": r"\bsign(ed|ing|ings)?\b|contract|extension|extend|re-?sign",
    "milestone": r"\bmilestones?|record|career|first goal|hat trick",
    "champion": r"\bcup|champion|title",
    "retirement": r"\bretir",
    "captain": r"\bcaptain",
}


def verdict_kind(q: str) -> Optional[str]:
    q = " ".join(str(q or "").lower().replace("’", "'").split())
    if re.search(r"\balmost\b|\bnearly\b", q):
        return None  # "the trade we almost made" has its own answer
    for pat, kind in VERDICTS:
        if re.search(pat, q):
            return kind
    return None


def is_event_question(q: str) -> bool:
    q = " ".join(str(q or "").lower().replace("’", "'").split())
    if re.search(r"\b(last|previous|that) (year|season)\b|\b(two|2|three|3) (years|seasons) ago\b", q) and not re.search(r"\b(trades?|injur|news|signings?)\b", q):
        return False  # season recaps belong to the history answers
    return bool(re.search(EVENT_TRIGGERS, q))


# =========================================================================== events
def _corpus(view: LiveView) -> List[Dict[str, Any]]:
    s = view.session
    items: List[Dict[str, Any]] = []
    league = getattr(getattr(s, "sim", None), "league", None)
    for t in list(getattr(league, "trade_history", None) or [])[-300:]:
        if not isinstance(t, dict) or not t.get("accepted", True):
            continue
        teams = [str(x) for x in (t.get("participating_teams") or [])]
        names = [str(m.get("player_name") or m.get("name") or "") for m in (t.get("moved_players") or []) if isinstance(m, dict)]
        head = str(t.get("headline") or "")
        if not head:
            head = " / ".join(_team_label(view, x) for x in teams) + " trade: " + ", ".join(n for n in names if n)
        items.append({"type": "trade", "date": str(t.get("calendar_iso") or "")[:10], "text": head,
                      "teams": teams, "players": [str(m.get("asset_id") or m.get("player_id") or "") for m in (t.get("moved_players") or []) if isinstance(m, dict)],
                      "names": names, "weight": 1.2})
    for r in list(getattr(s, "injury_log_all", None) or [])[-400:]:
        if not isinstance(r, dict):
            continue
        g = int(r.get("games") or 0)
        text = f"{r.get('player_name')} ({r.get('team_abbr') or r.get('team_abbrev')}) injured — {r.get('injury_type') or r.get('tier')}, about {g} games"
        items.append({"type": "injury", "date": str(r.get("calendar_iso") or r.get("date") or "")[:10], "text": text,
                      "teams": [str(r.get("team_id") or "")], "players": [str(r.get("player_id") or "")],
                      "names": [str(r.get("player_name") or "")], "weight": 1.0 + min(1.0, g / 30.0)})
    for e in list(getattr(s, "storyline_events", None) or [])[-200:]:
        if not isinstance(e, dict):
            continue
        text = str(e.get("headline") or e.get("text") or e.get("title") or "")
        if not text:
            continue
        kind = str(e.get("kind") or e.get("type") or "story")
        items.append({"type": "contract" if "contract" in kind else kind, "date": str(e.get("calendar_iso") or e.get("date") or "")[:10],
                      "text": text, "teams": [str(x) for x in (e.get("team_ids") or [e.get("team_id")]) if x],
                      "players": [str(x) for x in (e.get("player_ids") or [e.get("player_id")]) if x], "names": [], "weight": 0.9})
    try:
        from services.legacy_hall import ensure_legacy

        lg = ensure_legacy(s)
        for ev in list((lg.get("events") or {}).values())[-300:]:
            pay = ev.get("payload") or {}
            text = str(pay.get("headline") or pay.get("label") or ev.get("type", "").replace("_", " ").title())
            if pay.get("name"):
                text = f"{pay['name']}: {text}"
            items.append({"type": "champion" if "champion" in ev.get("type", "") else ev.get("type", "event").split("_")[0],
                          "date": str(ev.get("iso") or f"{ev.get('season') or ''}")[:10], "text": text,
                          "teams": list(ev.get("team_ids") or []), "players": list(ev.get("player_ids") or []), "names": [], "weight": 1.1})
    except Exception:
        pass
    return items


def _today(view: LiveView) -> str:
    s = view.session
    for attr in ("current_date_iso", "calendar_iso", "current_iso"):
        v = str(getattr(s, attr, "") or "")
        if len(v) >= 10:
            return v[:10]
    return ""


def _days_between(a: str, b: str) -> Optional[int]:
    try:
        from datetime import date

        da, db = date.fromisoformat(a[:10]), date.fromisoformat(b[:10])
        return (db - da).days
    except Exception:
        return None


def events(view: LiveView, raw: str, slots: Dict[str, Any]) -> Answer:
    q = " ".join(str(raw or "").lower().replace("’", "'").split())
    words = [w for w in re.findall(r"[a-z0-9']+", q) if w not in _STOP and len(w) > 2]
    teams = {str(t) for t in (slots.get("teams") or [])}
    if re.search(r"\b(we|us|our)\b", q):
        teams.add(view.user_tid)
    players = {c[0] for c in (slots.get("players") or []) if c}
    want = [k for k, pat in _TYPE_WORDS.items() if re.search(pat, q)]
    today = _today(view)
    horizon = 7 if re.search(r"\b(this week|past week|today|yesterday|last few days)\b", q) else (
        31 if re.search(r"\b(this month|past month|last month)\b", q) else (21 if re.search(r"\b(latest|recent|lately|just)\b", q) else None))
    league_wide = bool(re.search(r"\b(around|across|in) the league\b|\bother teams\b|\bleague[- ]wide\b", q))
    scored = []
    for it in _corpus(view):
        if want and it["type"] not in want:
            continue
        if teams and not (teams & set(it["teams"])):
            continue
        if players and not (players & set(it["players"])):
            continue
        if league_wide and view.user_tid in it["teams"] and len(it["teams"]) == 1:
            continue
        age = _days_between(it["date"], today) if (today and it["date"]) else None
        if horizon is not None and age is not None and age > horizon:
            continue
        text = it["text"].lower()
        overlap = sum(1 for w in words if w in text or any(w in n.lower() for n in it["names"]))
        recency = 1.0 / (1.0 + (age or 0) / 10.0) if age is not None else 0.3
        score = it["weight"] * (1.0 + overlap) + recency
        scored.append((score, it))
    if not scored:
        what = " / ".join(want) if want else "events"
        return _answer(f"Nothing in the save matches that ({what}{', this week' if horizon == 7 else ''}).",
                       analysis=["The save records trades, injuries, contract news, storylines and milestones. Try a team, a player, or a type like trades or injuries."])
    scored.sort(key=lambda x: (-x[0], x[1]["date"]), reverse=False)
    top = [it for _, it in scored[:10]]
    top.sort(key=lambda it: it["date"], reverse=True)
    lead = top[0]
    head = f"Latest: {lead['text']}" + (f" ({lead['date']})." if lead["date"] else ".")
    by_type: Dict[str, int] = {}
    for _, it in scored:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
    summary = ", ".join(f"{n} {k}{'s' if n != 1 and not k.endswith('y') else ''}" for k, n in sorted(by_type.items(), key=lambda x: -x[1])[:4])
    table = {"cols": ["Date", "Type", "What happened"], "rows": [[it["date"] or "—", it["type"].title(), it["text"]] for it in top]}
    return _answer(head, analysis=[f"Matched {len(scored)} records: {summary}."], table=table,
                   followups=["Latest trades around the league", "Who got hurt this week?", "Any news on my team?"])


# =========================================================================== verdicts
def _draft(p: Any) -> Dict[str, Any]:
    raw = getattr(p, "draft_info", None) or getattr(p, "draft", None)
    d: Dict[str, Any] = {}
    if isinstance(raw, dict):
        d = dict(raw)
    elif raw is not None:
        d = {k: getattr(raw, k) for k in dir(raw) if not k.startswith("_") and not callable(getattr(raw, k, None))}
    ident = getattr(p, "identity", None)
    for k in ("draft_year", "draft_round", "draft_pick", "draft_overall"):
        if k not in d and ident is not None and getattr(ident, k, None) is not None:
            d[k] = getattr(ident, k)
    def first(*keys):
        for k in keys:
            v = d.get(k)
            try:
                if v is not None and int(v) > 0:
                    return int(v)
            except (TypeError, ValueError):
                continue
        return None
    return {"overall": first("overall_pick", "draft_overall", "overall", "pick_overall", "draft_pick", "pick"),
            "round": first("round", "draft_round"), "year": first("year", "draft_year", "season")}


def _expected_ovr_for_pick(pick: int, years: int) -> float:
    """What that slot usually becomes, given years since the draft (rough NHL curve)."""
    base = 86.0 if pick <= 3 else (83.0 if pick <= 10 else (80.0 if pick <= 20 else (78.0 if pick <= 32 else (74.0 if pick <= 64 else 71.0))))
    ramp = min(1.0, 0.55 + 0.11 * max(0, years))
    return 62.0 + (base - 62.0) * ramp


def _bust_or_steal(view: LiveView, kind: str, raw: str) -> Answer:
    year_now = int(view.season or 0) or 2026
    mine = bool(re.search(r"\b(my|our|we)\b", raw))
    rows = []
    for pid, p in view.players.items():
        d = _draft(p)
        pick = d["overall"] or ((d["round"] - 1) * 32 + 16 if d["round"] else None)
        if not pick or not d["year"]:
            continue
        years = year_now - d["year"]
        if years < 3:
            continue  # too soon to judge
        r = view.row(pid)
        if not r or r.get("level") == "FA" and kind == "steal":
            continue
        if mine and r.get("team_id") != view.user_tid:
            continue
        exp = _expected_ovr_for_pick(pick, years)
        gap = float(r.get("ovr") or 0) - exp
        if kind == "bust" and pick > 32:
            continue
        if kind == "steal" and pick <= 20:
            continue
        rows.append((gap, pick, years, r))
    if not rows:
        return _answer("No drafted players with at least three years since their draft to judge yet.")
    rows.sort(key=lambda x: x[0], reverse=(kind == "steal"))
    gap, pick, years, r = rows[0]
    if kind == "bust":
        head = f"Biggest draft bust: {r['name']} — taken #{pick}, now {r['ovr']:.0f} overall {years} years later, about {abs(gap):.0f} below what that slot usually becomes."
        how = "First-round picks only, 3+ years after their draft. Yardstick: current overall against the typical overall for that draft slot at that many years out."
    else:
        head = f"Biggest draft steal: {r['name']} — taken #{pick}, now {r['ovr']:.0f} overall, about {gap:.0f} above a typical #{pick} pick {years} years on."
        how = "Picks after #20, 3+ years after their draft. Yardstick: current overall above the typical overall for that slot."
    table = {"cols": ["Player", "Team", "Pick", "Years", "OVR", "vs slot"],
             "rows": [[x[3]["name"], x[3].get("team") or "—", f"#{x[1]}", x[2], f"{x[3]['ovr']:.0f}", f"{x[0]:+.0f}"] for x in rows[:10]]}
    return _answer(head, analysis=[how, "Injuries and late development aren't held against anyone beyond what the rating shows."], table=table,
                   followups=["Biggest draft steal", "My best draft pick", "Highest ceiling in the league"])


def _expected_pts(ovr: float, gp: int, pos: str) -> float:
    # Points-per-game line by overall (forwards; defense at ~55%).
    ppg = max(0.05, 0.035 * (ovr - 66.0))
    if pos in ("D", "LD", "RD"):
        ppg *= 0.55
    return ppg * gp


def _league_sv(view: LiveView) -> float:
    ga = sa = 0.0
    for st in view.stats.values():
        if not isinstance(st, dict):
            continue
        s_a = float(st.get("shots_against") or st.get("goalie_shots_against") or 0)
        if s_a > 0:
            sa += s_a
            ga += float(st.get("goalie_ga", st.get("ga")) or 0)
    return 1.0 - ga / sa if sa else 0.905


def _impact_rows(view: LiveView, tid: str) -> List[Dict[str, Any]]:
    """Goals each player gave or cost relative to what his rating and pay said he should."""
    sv_lg = _league_sv(view)
    out = []
    for r in view.team_rows(tid):
        gp = int(r.get("gp") or 0)
        if gp < 15:
            continue
        cap = float((r.get("contract") or {}).get("cap_hit_m") or 0)
        if r.get("pos") == "G":
            st = view.stats.get(r["id"]) or {}
            sa = float(st.get("shots_against") or st.get("goalie_shots_against") or 0)
            if sa <= 0:
                continue
            ga = float(st.get("goalie_ga", st.get("ga")) or 0)
            xga = float(st.get("goalie_xga") or 0)
            # Goals saved above expected when the sim graded the shots; league save % otherwise.
            goals = (xga - ga) if xga > 0 else ((1.0 - sv_lg) * sa - ga)
            why = (f"{xga - ga:+.1f} GSAx ({int(ga)} allowed on {xga:.1f} xGA)" if xga > 0 else f"{r.get('sv_pct') or 0:.3f} SV% vs league {sv_lg:.3f} on {int(sa)} shots")
        else:
            exp = _expected_pts(float(r.get("ovr") or 0), gp, r.get("pos") or "")
            pts = float(r.get("pts") or 0)
            st = view.stats.get(r["id"]) or {}
            xf, xa = float(st.get("xgf") or 0), float(st.get("xga") or 0)
            if xf + xa > 3:
                # Expected-goal differential on ice is a steadier read of two-way impact than plus/minus.
                goals = 0.5 * (pts - exp) + 0.35 * (xf - xa)
                why = f"{int(pts)} pts vs ~{exp:.0f} expected for a {r.get('ovr', 0):.0f}, on-ice xG {xf:.1f}-{xa:.1f}"
            else:
                goals = 0.5 * (pts - exp) + 0.25 * float(r.get("plus_minus") or 0)
                why = f"{int(pts)} pts vs ~{exp:.0f} expected for a {r.get('ovr', 0):.0f}, {int(r.get('plus_minus') or 0):+d}"
        out.append({**r, "impact": goals, "why": why, "cap": cap, "weight": goals - (0.4 * cap if goals < 0 else 0.0)})
    return out


def _blame(view: LiveView, raw: str, slots: Dict[str, Any]) -> Answer:
    teams = [str(t) for t in (slots.get("teams") or [])]
    tid = view.user_tid if (re.search(r"\b(we|us|our)\b", raw) or not teams) else teams[0]
    team = _team_label(view, tid)
    rows = _impact_rows(view, tid)
    if not rows:
        return _answer(f"Not enough games yet to judge anyone on the {team}.")
    rows.sort(key=lambda x: x["weight"])
    worst = rows[0]
    injured = [r for r in view.team_rows(tid) if r.get("injured") and (r.get("ovr") or 0) >= 80]
    head = (f"The biggest single drag on the {team}: {worst['name']} — about {abs(worst['impact']):.1f} goals "
            f"{'below' if worst['impact'] < 0 else 'above'} what his rating and role called for ({worst['why']}).")
    notes = ["Yardstick: points against what each player's overall and position should produce, plus his on-ice expected-goal differential "
             "(goalies: goals saved above expected); big contracts weigh heavier when they underdeliver.",
             "One player rarely sinks a season alone — the table shows everyone who fell short."]
    if injured:
        notes.append("Missing through injury now: " + ", ".join(f"{r['name']} ({r['ovr']:.0f})" for r in injured[:4]) + ".")
    table = {"cols": ["Player", "Pos", "GP", "Impact (goals)", "Why"],
             "rows": [[r["name"], r.get("pos"), r.get("gp"), f"{r['impact']:+.1f}", r["why"]] for r in rows[:8]]}
    return _answer(head, analysis=notes, table=table, followups=["Why are we losing?", "Who carried us?", "Which goalie should start?"])


def _disappoint_or_surprise(view: LiveView, kind: str, raw: str, slots: Dict[str, Any]) -> Answer:
    mine = bool(re.search(r"\b(my|our|we|us)\b", raw))
    teams = [str(t) for t in (slots.get("teams") or [])]
    pool = [view.user_tid] if mine else (teams[:1] or list(view.team_meta))
    rows = []
    for tid in pool:
        rows.extend(_impact_rows(view, tid))
    if not rows:
        return _answer("Not enough games yet to call anyone a surprise or a disappointment.")
    if kind == "disappoint":
        # Disappointment needs expectations: stars and big contracts only.
        cand = [r for r in rows if (r.get("ovr") or 0) >= 80 or r["cap"] >= 5.0]
        cand.sort(key=lambda r: r["weight"])
        if not cand:
            return _answer("Nobody with big expectations has fallen short yet.")
        r = cand[0]
        head = f"Most disappointing: {r['name']} ({r.get('team')}) — {r['why']}, about {abs(r['impact']):.1f} goals short of his billing."
        how = "Among 80+ overall players or $5M+ contracts: the biggest gap between results and what the rating and pay promised."
    else:
        cand = [r for r in rows if (r.get("ovr") or 0) < 82 and r["cap"] < 5.0]
        cand.sort(key=lambda r: r["impact"], reverse=True)
        if not cand:
            return _answer("No surprise packages yet.")
        r = cand[0]
        head = f"Biggest surprise: {r['name']} ({r.get('team')}) — {r['why']}, about {r['impact']:.1f} goals above what a {r.get('ovr', 0):.0f} usually gives."
        how = "Players under 82 overall and $5M: the most production above what their rating predicts."
    table = {"cols": ["Player", "Team", "OVR", "Cap", "Impact", "Why"],
             "rows": [[x["name"], x.get("team"), f"{x.get('ovr', 0):.0f}", f"${x['cap']:.1f}M", f"{x['impact']:+.1f}", x["why"]] for x in cand[:8]]}
    return _answer(head, analysis=[how], table=table, followups=["Biggest surprise", "Most disappointing player", "Most underrated"])


def _trade_verdict(view: LiveView, raw: str, slots: Optional[Dict[str, Any]] = None) -> Answer:
    league = getattr(getattr(view.session, "sim", None), "league", None)
    hist = [t for t in list(getattr(league, "trade_history", None) or []) if isinstance(t, dict) and t.get("accepted", True)]
    mine = bool(re.search(r"\b(my|our|we|i)\b", raw))
    named = {c[0] for c in ((slots or {}).get("players") or []) if c}
    named_teams = {str(t) for t in ((slots or {}).get("teams") or [])}
    results = []
    for t in hist[-200:]:
        teams = [str(x) for x in (t.get("participating_teams") or [])]
        if len(teams) < 2 or (mine and view.user_tid not in teams):
            continue
        moved_ids = {str(m.get("asset_id") or m.get("player_id") or "") for m in (t.get("moved_players") or []) if isinstance(m, dict)}
        if named and not (named & moved_ids):
            continue  # "who won the Hanifin trade" judges only the deal he was in
        if named_teams and not (named_teams & set(teams)):
            continue
        got: Dict[str, float] = {x: 0.0 for x in teams}
        detail: Dict[str, List[str]] = {x: [] for x in teams}
        for m in t.get("moved_players") or []:
            if not isinstance(m, dict):
                continue
            pid = str(m.get("asset_id") or m.get("player_id") or "")
            to = str(m.get("acquiring_team_id") or m.get("to_team_id") or "")
            r = view.row(pid) if pid in view.players else {}
            ovr = float(r.get("ovr") or 0) if r else 0.0
            pts = float(r.get("pts") or 0) if r else 0.0
            # Present-day worth: rating above replacement, plus this season's production.
            val = max(0.0, ovr - 70.0) * 1.5 + pts * 0.25
            if to in got:
                got[to] += val
                detail[to].append(f"{r.get('name') or m.get('player_name') or 'player'} ({ovr:.0f})")
        for pk in t.get("moved_picks") or []:
            if not isinstance(pk, dict):
                continue
            to = str(pk.get("acquiring_team_id") or pk.get("to_team_id") or "")
            rnd = int(pk.get("round") or 3)
            if to in got:
                got[to] += {1: 12.0, 2: 5.0}.get(rnd, 2.0)
                detail[to].append(f"R{rnd} pick")
        ordered = sorted(got.items(), key=lambda x: -x[1])
        gap = ordered[0][1] - ordered[-1][1]
        results.append((gap, ordered, detail, t))
    if not results:
        return _answer("No completed trades to judge yet" + (" involving your club." if mine else "."))
    best = re.search(r"\bbest\b", raw) and mine
    results.sort(key=lambda x: x[0], reverse=True)
    if best:
        results = [x for x in results if x[1][0][0] == view.user_tid] or results
    gap, ordered, detail, t = results[0]
    win, lose = ordered[0][0], ordered[-1][0]
    head = (f"Most lopsided trade: the {_team_label(view, win)} won it — got {', '.join(detail[win]) or 'value'} "
            f"for {', '.join(detail[lose]) or 'little'} ({str(t.get('calendar_iso') or '')[:10]}).")
    how = ("Yardstick: what each side holds today — rating above a replacement player (70), this season's points, "
           "and picks at a fixed value (R1 12, R2 5, later 2). Picks not yet used can still change the verdict.")
    table = {"cols": ["Date", "Winner", "Got", "Loser", "Got", "Margin"],
             "rows": [[str(x[3].get("calendar_iso") or "")[:10], _team_label(view, x[1][0][0]), ", ".join(x[2][x[1][0][0]])[:60],
                       _team_label(view, x[1][-1][0]), ", ".join(x[2][x[1][-1][0]])[:60], f"{x[0]:.0f}"] for x in results[:8]]}
    return _answer(head, analysis=[how], table=table, followups=["My worst trade", "Latest trades around the league", "Biggest draft bust"])


def verdict(view: LiveView, kind: str, raw: str, slots: Dict[str, Any]) -> Answer:
    q = " ".join(str(raw or "").lower().replace("’", "'").split())
    if kind in ("bust", "steal"):
        return _bust_or_steal(view, kind, q)
    if kind == "blame":
        return _blame(view, q, slots)
    if kind in ("disappoint", "surprise"):
        return _disappoint_or_surprise(view, kind, q, slots)
    return _trade_verdict(view, q, slots)
