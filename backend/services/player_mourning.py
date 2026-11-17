"""When a player dies: the obituary, the mourning arc, and what it does to a club.

A death used to be one paragraph and the sim kept rolling. Now it stops every sim (day, week,
month, season) and opens a real arc. The 30 linked features:

 1. Full obituary: family, hometown, how it happened (told respectfully), career line, honours.
 2. Every sim stops, bulk included (``session._sim_hard_stop``).
 3. League-wide moment of silence before the next slate.
 4. Memorial patch on his club's sweaters for the rest of the season.
 5. Your call on his number: retire it, banner + patch, decide at season's end, or nothing.
 6. Your call on support: grief counsellors, two days off, or business as usual.
 7. Club response arc: the room either rallies ("play for him") or grieves (real, timed OVR
    changes through readiness modifiers), shaped by leadership, room morale and your choices.
 8. His closest friend on the team takes it hardest and may step away for a few games.
 9. Linemates / partner lose chemistry for a while.
10. Captain's eulogy with a quote.
11. Tribute night at the next home game, family on the ice, crowd lift for one night.
12. A memorial foundation in his name.
13. A youth hockey programme in his hometown.
14. A rival club wears his number in warmups.
15. His former clubs pay tribute.
16. Opposing crowds give a standing ovation in his old barn.
17. A league award named after him (stars and long careers).
18. Hall of Fame fast-track flag for an elite career.
19. Insurance settles his contract with the family; the cap hit is gone the same day.
20. Funeral: the team attends, the room's morale dips.
21. Cause-specific league reforms (screening, travel, off-ice activity clauses).
22. Season dedication; it carries into the playoffs story.
23. Social tributes from fans and players.
24. His number raised to the rafters in a ceremony (if retired).
25. Months later a young teammate asks to wear his number (if it wasn't retired).
26. Fans react if a beloved player's number isn't honoured.
27. The family's meeting with the GM: handle it well and the room's trust in you rises.
28. Room check-in three weeks later: did the team pull through?
29. Anniversary remembrance a year on.
30. A suspicious death opens a police inquiry (ties into the cap-dump plot).

State: ``session.mourning`` = {"records": [...], "league_memorials": [...]}.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple
import logging as _logging_swallow

_log = _logging_swallow.getLogger(__name__)

CAUSE_STORY = {
    "a car accident": ("was killed in a car accident on his way home", "travel"),
    "a private plane crash": ("was among those lost when a private plane went down", "travel"),
    "an off-ice training accident": ("died after an accident during an off-ice training session", "training"),
    "an undiagnosed medical condition": ("collapsed at home from a heart condition no one knew he had", "medical"),
    "a house fire": ("died in a fire at his home", "home"),
    "a skiing accident": ("died after a skiing accident during a family trip", "activity"),
    "a motorcycle accident": ("died in a motorcycle accident", "activity"),
    "a drowning accident": ("drowned in a boating accident at the family cottage", "activity"),
    "a sudden illness": ("died after a sudden, short illness", "medical"),
    "a hiking accident": ("died in a fall while hiking", "activity"),
    "carbon monoxide exposure": ("died from carbon monoxide exposure at a rental property", "home"),
    "a lightning strike": ("was struck by lightning on a golf course", "activity"),
    "complications after a routine medical procedure": ("died from complications after a routine procedure", "medical"),
}
REFORM = {
    "travel": "The league is reviewing travel rules: no private flights in bad weather and an independent safety check on every charter.",
    "training": "Clubs must now have a certified trainer at every off-ice session.",
    "medical": "The league is mandating full cardiac screening for every player at training camp.",
    "home": "The players' association is funding home safety inspections and detectors for every member.",
    "activity": "Clubs are tightening the risky-activity clauses in standard player contracts.",
    "suspicious": "Police have opened an inquiry and the league is cooperating fully.",
}


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16))


def _state(session: Any) -> Dict[str, Any]:
    st = getattr(session, "mourning", None)
    if not isinstance(st, dict):
        st = {"records": [], "league_memorials": []}
        session.mourning = st
    st.setdefault("records", [])
    st.setdefault("league_memorials", [])
    return st


def _ce():
    from services import contract_economy as ce

    return ce


def _pid(p: Any) -> str:
    try:
        return str(_ce()._player_id(p))
    except Exception:
        return str(getattr(p, "id", "") or getattr(p, "player_id", "") or "")


def _name(p: Any) -> str:
    try:
        return str(_ce()._player_name(p))
    except Exception:
        return str(getattr(p, "name", "") or "A player")


def _first(n: str) -> str:
    return (str(n).split() or ["He"])[0]


def _last(n: str) -> str:
    return (str(n).split() or ["him"])[-1]


def _age(p: Any) -> int:
    try:
        return int(_ce()._player_age(p) or 27)
    except Exception:
        return 27


def _ovr(p: Any) -> float:
    try:
        return float(_ce()._player_ovr(p))
    except Exception:
        return 70.0


def _country(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "birth_country", None) or getattr(ident, "nationality", None) or getattr(p, "nationality", None) or "").strip()


def _team_name(team: Any) -> str:
    return str(getattr(team, "name", None) or getattr(team, "abbreviation", None) or getattr(team, "team_id", "") or "his club")


def _tid(team: Any) -> str:
    return str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")


def _cursor(session: Any) -> int:
    return int(getattr(session, "calendar_cursor", 0) or 0)


def _names_pool(country: str):
    try:
        from app.sim_engine.generation.name_generator import NAME_POOLS

        key = {"CAN": "Canada", "USA": "USA", "SWE": "Sweden", "FIN": "Finland", "CZE": "Czechia", "RUS": "Russia", "SVK": "Slovakia",
               "GER": "Germany", "SUI": "Switzerland", "LAT": "Latvia", "DEN": "Denmark", "NOR": "Norway"}.get(country.upper(), country)
        return NAME_POOLS.get(key) or NAME_POOLS.get("Canada") or {}
    except Exception:
        return {}


FEMALE_FIRST = ["Emily", "Sarah", "Jessica", "Hannah", "Olivia", "Sofia", "Emma", "Anna", "Julia", "Maria", "Elin", "Laura", "Kate", "Natalie"]


def _family(session: Any, player: Any) -> Dict[str, Any]:
    """Family from the player's life file when the universe has one, otherwise a seeded, stable family."""
    ent = dict((getattr(session, "universe_players", None) or {}).get(_pid(player)) or {})
    life = dict(ent.get("life") or {})
    r = _rng("family", _pid(player))
    age = _age(player)
    last = _last(_name(player))
    pool = _names_pool(_country(player))
    firsts = list(pool.get("first") or ["Jack", "Liam", "Noah", "Leo", "Max"])
    partner = life.get("spouse_name") or life.get("partner_name")
    married = bool(partner) or (age >= 26 and r.random() < min(0.85, 0.25 + (age - 22) * 0.07))
    if married and not partner:
        partner = f"{r.choice(FEMALE_FIRST)}"
    n_kids = int(life.get("children") or 0) if isinstance(life.get("children"), (int, float)) else (
        (r.choice([0, 1, 1, 2, 2, 3]) if married and age >= 27 else 0))
    kids = []
    for _ in range(n_kids):
        boy = r.random() < 0.5
        kids.append({"name": r.choice(firsts) if boy else r.choice(FEMALE_FIRST), "age": r.randint(1, max(1, min(12, age - 24)))})
    hometown = str(getattr(getattr(player, "identity", None), "birth_city", None) or getattr(player, "birth_city", None) or life.get("hometown") or "").strip()
    return {
        "partner": partner if married else "",
        "partner_label": "wife" if married else "",
        "children": kids,
        "parents": f"{r.choice(firsts)} and {r.choice(FEMALE_FIRST)} {last}",
        "siblings": r.choice([0, 1, 1, 2, 3]),
        "hometown": hometown,
        "country": _country(player),
    }


def _career(session: Any, player: Any, team: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"gp": None, "g": None, "a": None, "pts": None, "awards": [], "seasons": None, "teams": []}
    try:
        from services.franchise_sim import _serialize_player_row

        row = _serialize_player_row(player, include_ratings=False, session=session, _team=team)
        tot = dict(row.get("career_totals") or {})
        out.update({
            "gp": tot.get("gp") or row.get("nhl_career_gp"), "g": tot.get("g") or tot.get("goals"), "a": tot.get("a") or tot.get("assists"),
            "pts": tot.get("pts") or tot.get("points"),
            "awards": [a.get("award") if isinstance(a, dict) else str(a) for a in (row.get("career_awards") or [])][:6],
        })
        seasons = [s for s in (row.get("career_seasons") or []) if isinstance(s, dict)]
        out["seasons"] = len({str(s.get("season")) for s in seasons}) or None
        out["teams"] = list(dict.fromkeys(str(s.get("team_name") or s.get("team") or "") for s in seasons if s.get("team_name") or s.get("team")))[-5:]
    except Exception:
        _log.debug("career summary failed", exc_info=True)
    return out


def _roster(team: Any) -> List[Any]:
    return list(getattr(team, "roster", None) or [])


def _captain(team: Any, exclude: str) -> Optional[Any]:
    cid = str(getattr(team, "captain_id", getattr(team, "captain", "")) or "")
    rows = [p for p in _roster(team) if _pid(p) != exclude]
    for p in rows:
        if cid and _pid(p) == cid:
            return p
    vets = sorted(rows, key=lambda p: -(_ovr(p) + _age(p) * 0.5))
    return vets[0] if vets else None


def _best_friend(team: Any, player: Any) -> Optional[Any]:
    rows = [p for p in _roster(team) if _pid(p) != _pid(player)]
    if not rows:
        return None
    c, a = _country(player), _age(player)
    return min(rows, key=lambda p: (0 if _country(p) == c else 1, abs(_age(p) - a), -_ovr(p)))


def _linemates(team: Any, player: Any) -> List[Any]:
    try:
        pos = str(_ce()._position_bucket(player))
    except Exception:
        pos = ""
    rows = [p for p in _roster(team) if _pid(p) != _pid(player)]
    fwd = {"C", "LW", "RW", "F"}

    def same(p):
        try:
            b = str(_ce()._position_bucket(p))
        except Exception:
            return False
        return (b in fwd) == (pos in fwd) and (b == "G") == (pos == "G")

    pool = sorted([p for p in rows if same(p)], key=lambda p: abs(_ovr(p) - _ovr(player)))
    return pool[:2]


def _storyline(session: Any, headline: str, summary: str, team_id: str = "", heat: int = 70, typ: str = "mourning") -> None:
    try:
        from app.sim_engine.franchise.state import _record_storyline

        _record_storyline(session, {"headline": headline, "summary": summary, "team_id": team_id, "category": "major_news",
                                    "type": typ, "priority": "HIGH" if heat >= 75 else "MEDIUM", "heat": heat})
    except Exception:
        _log.debug("storyline failed", exc_info=True)
    try:
        session.timeline.append(headline)
    except Exception:
        pass


def _ready(session: Any, pid: str, delta: float, days: int, reason: str, src: str) -> None:
    try:
        from app.sim_engine.franchise.storyline_engine import _u_apply_readiness_modifier

        _u_apply_readiness_modifier(session, pid, source_id=src, ovr_delta=float(delta), days=int(days), reason=reason)
    except Exception:
        _log.debug("readiness failed", exc_info=True)


def _morale(session: Any, pid: str, delta: float, key: str = "morale") -> None:
    ent = (getattr(session, "universe_players", None) or {}).get(str(pid))
    if isinstance(ent, dict):
        st = ent.setdefault("state", {})
        st[key] = max(0.0, min(100.0, float(st.get(key, 55) or 55) + float(delta)))


def _team_morale(team: Any, delta: float) -> None:
    st = getattr(team, "state", None)
    if st is not None:
        try:
            st.team_morale = max(0.0, min(1.0, float(getattr(st, "team_morale", 0.5) or 0.5) + float(delta)))
        except Exception:
            pass


def _fans(session: Any, tid: str, delta: float) -> None:
    try:
        from services.franchise_sim import _ensure_team_fan_profile

        prof = _ensure_team_fan_profile(session, tid)
        if isinstance(prof, dict):
            prof["fan_confidence"] = max(0.0, min(100.0, float(prof.get("fan_confidence", 55) or 55) + float(delta)))
    except Exception:
        _log.debug("fan bump failed", exc_info=True)


def _team_by_id(session: Any, tid: str) -> Any:
    tm = (getattr(session, "team_by_id", None) or {}).get(str(tid))
    if tm is not None:
        return tm
    league = getattr(getattr(session, "sim", None), "league", None)
    for t in list(getattr(league, "teams", None) or []):
        if _tid(t) == str(tid):
            return t
    return None


# --------------------------------------------------------------------------- obituary + arc


def build_obituary(session: Any, player: Any, team: Any, cause: str, *, suspicious: bool = False) -> Dict[str, Any]:
    name = _name(player)
    fam = _family(session, player)
    car = _career(session, player, team)
    story, reform_key = CAUSE_STORY.get(cause, (f"died in {cause}", "medical"))
    if suspicious:
        reform_key = "suspicious"
    first = _first(name)
    pieces = []
    if fam["partner"]:
        pieces.append(f"his {fam['partner_label']} {fam['partner']}")
    if fam["children"]:
        kids = ", ".join(f"{k['name']} ({k['age']})" for k in fam["children"])
        pieces.append(f"{'their children' if fam['partner'] else 'his children'} {kids}")
    pieces.append(f"his parents {fam['parents']}")
    if fam["siblings"]:
        pieces.append(f"{fam['siblings']} {'sibling' if fam['siblings'] == 1 else 'siblings'}")
    survived = "He is survived by " + (", ".join(pieces[:-1]) + " and " + pieces[-1] if len(pieces) > 1 else pieces[0]) + "."
    cap = _captain(team, _pid(player))
    friend = _best_friend(team, player)
    cap_name = _name(cap) if cap is not None else "his captain"
    quotes = [
        {"who": cap_name, "role": "Captain", "text": f"{first} was the first guy at the rink and the last to leave. We'll carry him with us every night."},
        {"who": _name(friend) if friend is not None else "A teammate", "role": "Teammate and friend",
         "text": f"He was my best friend in this game. I don't know how to walk into that room without him."},
        {"who": "Head coach", "role": "Coach", "text": f"Some players you coach. {first} you were lucky to know."},
    ]
    legacy = []
    if (car.get("gp") or 0) and (car.get("pts") is not None):
        legacy.append(f"{car['gp']} games, {car['g'] or 0} goals and {car['pts']} points")
    if car.get("awards"):
        legacy.append("honours: " + ", ".join(str(a) for a in car["awards"][:4]))
    return {
        "name": name, "age": _age(player), "team_name": _team_name(team), "team_id": _tid(team),
        "position": str(getattr(player, "position", "") or ""), "ovr": round(_ovr(player)),
        "cause": cause, "cause_story": f"{name} {story}.", "suspicious": suspicious, "reform_key": reform_key,
        "family": fam, "survived_by": survived, "career": car, "legacy_line": "; ".join(legacy),
        "tributes": quotes, "hometown": fam.get("hometown") or fam.get("country") or "",
        "jersey_number": getattr(player, "jersey_number", None) or getattr(getattr(player, "identity", None), "jersey_number", None),
    }


def on_player_death(session: Any, player: Any, team: Any, cause: str, *, suspicious: bool = False, source: str = "natural",
                    quiet: bool = False) -> Dict[str, Any]:
    """Call BEFORE the player is purged from the roster. Returns obituary fields for the popup."""
    st = _state(session)
    pid = _pid(player)
    if any(r.get("player_id") == pid for r in st["records"]):
        return next(r["obituary"] for r in st["records"] if r.get("player_id") == pid)
    obit = build_obituary(session, player, team, cause, suspicious=suspicious)
    tid = _tid(team)
    user = str(getattr(session, "user_team_id", "") or "") == tid
    friend = _best_friend(team, player)
    mates = _linemates(team, player)
    r = _rng("mourn", pid, _cursor(session))
    beloved = obit["ovr"] >= 80 or (obit["career"].get("gp") or 0) >= 500 or r.random() < 0.3
    rec = {
        "player_id": pid, "name": obit["name"], "team_id": tid, "start": _cursor(session), "season": int(getattr(session, "season_calendar_year", 0) or 0),
        "obituary": obit, "user_team": user, "beloved": beloved, "friend_id": _pid(friend) if friend is not None else "",
        "friend_name": _name(friend) if friend is not None else "", "linemate_ids": [_pid(m) for m in mates],
        "honor": None, "support": None, "response": None, "done_beats": [], "source": source,
    }
    if quiet:
        # Another victim of the same tragedy: remembered with the first, no separate arc or decisions.
        rec["done_beats"] = [b for _o, b in _beats(rec) if b != "anniversary"]
        rec["response"] = "group"
        st["records"].append(rec)
        st["records"] = st["records"][-20:]
        for p in _roster(team):
            _morale(session, _pid(p), -4.0)
        return obit
    st["records"].append(rec)
    st["records"] = st["records"][-20:]

    # 2. stop every sim
    session._sim_hard_stop = {"reason": "player_death", "player": obit["name"]}
    # 3/4. moment of silence, memorial patch
    league = getattr(getattr(session, "sim", None), "league", None)
    for t in list(getattr(league, "teams", None) or []):
        _team_morale(t, -0.01)
    _team_morale(team, -0.12)
    for p in _roster(team):
        _morale(session, _pid(p), -8.0)
    rec["memorial_patch"] = True
    _storyline(session, f"League holds a moment of silence for {obit['name']}",
               f"Every arena will pause before puck drop. The {obit['team_name']} will wear a memorial patch for the rest of the season.",
               heat=80)
    # 19. insurance
    rec["insurance_note"] = "The club's insurance policy settles the rest of his contract with his family; the cap hit is gone today."
    # 22. dedication
    _storyline(session, f"The {obit['team_name']} dedicate the season to {obit['name']}", "\"Everything we do this year is for him.\"", team_id=tid, heat=72)
    if user:
        _queue_user_decisions(session, rec)
    else:
        _cpu_honor(session, rec, team)
    return obit


def _queue_user_decisions(session: Any, rec: Dict[str, Any]) -> None:
    name = rec["name"]
    num = rec["obituary"].get("jersey_number")
    numtxt = f"No. {num}" if num else "his number"
    pend = getattr(session, "pending_decisions", None)
    if not isinstance(pend, list):
        session.pending_decisions = pend = []
    pend.append({
        "id": f"mourn_honor_{rec['player_id']}", "kind": "mourning_honor", "requires_action": True,
        "title": f"Honouring {name}", "player_id": rec["player_id"], "player_name": name, "team_id": rec["team_id"],
        "description": f"The family, the room and the fans are waiting to hear what the club will do with {numtxt}.",
        "options": [
            {"id": "retire", "label": f"Retire {numtxt}", "effect_summary": "Fans and the room will remember you did this. Banner ceremony later this season."},
            {"id": "banner", "label": "Memorial banner + patch, number stays", "effect_summary": "A real tribute; some fans want more."},
            {"id": "later", "label": "Decide at season's end", "effect_summary": "Buys time. People will notice if it never comes."},
            {"id": "none", "label": "No ceremony", "effect_summary": "Business first. Expect a backlash."},
        ],
    })
    pend.append({
        "id": f"mourn_support_{rec['player_id']}", "kind": "mourning_support", "requires_action": True,
        "title": f"Taking care of the room after {name}", "player_id": rec["player_id"], "team_id": rec["team_id"],
        "description": "How does the club support the players right now?",
        "options": [
            {"id": "counsel", "label": "Bring in grief counsellors", "effect_summary": "The best chance the room pulls through together."},
            {"id": "days_off", "label": "Two days off for the team", "effect_summary": "Room appreciates it; a little rust in the next game."},
            {"id": "work", "label": "Business as usual", "effect_summary": "Some players need the routine. Others won't forgive it."},
        ],
    })


def _cpu_honor(session: Any, rec: Dict[str, Any], team: Any) -> None:
    r = _rng("cpu_honor", rec["player_id"])
    retire = rec["beloved"] and r.random() < 0.8 or r.random() < 0.25
    rec["honor"] = "retire" if retire else "banner"
    rec["support"] = r.choice(["counsel", "days_off", "work"])
    if retire:
        _retire_number(team, rec)
        _storyline(session, f"The {rec['obituary']['team_name']} will retire {rec['name']}'s number", "No one will wear it again.", team_id=rec["team_id"], heat=65)
    elif rec["beloved"]:
        _fans(session, rec["team_id"], -6.0)
        _storyline(session, f"Fans want {rec['name']}'s number retired", f"The {rec['obituary']['team_name']} went with a banner instead. Supporters aren't happy.",
                   team_id=rec["team_id"], heat=60)
    _decide_response(session, rec, team)


def _retire_number(team: Any, rec: Dict[str, Any]) -> None:
    num = rec["obituary"].get("jersey_number")
    try:
        lst = getattr(team, "retired_numbers", None)
        if not isinstance(lst, list):
            lst = []
            setattr(team, "retired_numbers", lst)
        lst.append({"number": num, "player": rec["name"], "player_id": rec["player_id"], "season": rec["season"], "memorial": True})
    except Exception:
        pass
    rec["number_retired"] = True


def resolve_mourning_decision(session: Any, d: Dict[str, Any], cid: str) -> Dict[str, Any]:
    st = _state(session)
    pid = str(d.get("player_id") or "")
    rec = next((r for r in st["records"] if r.get("player_id") == pid), None)
    if rec is None:
        return {}
    team = _team_by_id(session, rec["team_id"])
    kind = str(d.get("kind") or "")
    out: Dict[str, Any] = {}
    if kind == "mourning_honor":
        rec["honor"] = cid
        fan = {"retire": 7.0, "banner": 2.0, "later": 0.0, "none": -9.0}[cid] * (1.0 if rec["beloved"] else 0.6)
        room = {"retire": 0.05, "banner": 0.02, "later": 0.0, "none": -0.07}[cid]
        _fans(session, rec["team_id"], fan)
        if team is not None:
            _team_morale(team, room)
            if cid == "retire":
                _retire_number(team, rec)
        if cid == "none":
            _storyline(session, f"Backlash: no tribute for {rec['name']}", "Fans and players are furious the club won't honour him.", team_id=rec["team_id"], heat=82)
        out = {"fan_confidence": fan, "team_morale": room}
    elif kind == "mourning_support":
        rec["support"] = cid
        if team is not None:
            for p in _roster(team):
                _morale(session, _pid(p), {"counsel": 4.0, "days_off": 3.0, "work": -2.0}[cid])
                if cid == "days_off":
                    _ready(session, _pid(p), -1.0, 3, "Two days off after the loss", f"mourn_off_{pid}")
        out = {"room": cid}
    if rec.get("honor") and rec.get("support") and team is not None and not rec.get("response"):
        _decide_response(session, rec, team)
    return out


def _decide_response(session: Any, rec: Dict[str, Any], team: Any) -> None:
    """7/8/9: rally or grief, with timed, real OVR effects."""
    r = _rng("response", rec["player_id"])
    cap = _captain(team, rec["player_id"])
    lead = (_ovr(cap) - 70) / 30 if cap is not None else 0.3
    morale = float(getattr(getattr(team, "state", None), "team_morale", 0.5) or 0.5)
    p_rally = 0.35 + lead * 0.25 + (morale - 0.45) * 0.6
    p_rally += {"counsel": 0.15, "days_off": 0.07, "work": -0.12}.get(rec.get("support") or "", 0.0)
    p_rally += {"retire": 0.06, "banner": 0.02, "none": -0.10}.get(rec.get("honor") or "", 0.0)
    rally = r.random() < max(0.1, min(0.85, p_rally))
    rec["response"] = "rally" if rally else "grief"
    src = f"mourn_{rec['player_id']}"
    for p in _roster(team):
        pid = _pid(p)
        if pid == rec["friend_id"]:
            continue
        if rally:
            _ready(session, pid, r.uniform(1.0, 2.5), r.randint(15, 25), f"Playing for {rec['name']}", src)
        else:
            _ready(session, pid, -r.uniform(1.5, 3.5), r.randint(12, 25), f"Grieving {rec['name']}", src)
    for pid in rec["linemate_ids"]:
        _ready(session, pid, -2.0, 20, f"Lost his linemate {rec['name']}", src + "_line")
    if rec["friend_id"]:
        _ready(session, rec["friend_id"], -6.0 if not rally else -3.0, 30, f"Lost his best friend {rec['name']}", src + "_friend")
        _morale(session, rec["friend_id"], -15.0)
    head = (f"The {rec['obituary']['team_name']} rally around {rec['name']}'s memory" if rally
            else f"Grief hangs over the {rec['obituary']['team_name']}")
    body = ("Stick tape with his number, a shared tattoo, and a room that's playing for something bigger." if rally
            else "Players are struggling. The coaching staff says it'll take time.")
    _storyline(session, head, body, team_id=rec["team_id"], heat=70)


# --------------------------------------------------------------------------- daily beats

def _beats(rec: Dict[str, Any]) -> List[Tuple[int, str]]:
    b = [(1, "eulogy"), (2, "tributes"), (4, "funeral"), (7, "tribute_night"), (10, "foundation"), (14, "reform"),
         (21, "checkin"), (30, "legacy"), (45, "family_meeting"), (365, "anniversary")]
    if rec.get("obituary", {}).get("suspicious"):
        b.append((3, "inquiry"))
    b.append((95, "number_request"))
    return b


def mourning_daily_tick(session: Any) -> None:
    st = getattr(session, "mourning", None)
    if not isinstance(st, dict):
        return
    cur = _cursor(session)
    for rec in st.get("records") or []:
        for off, beat in _beats(rec):
            if beat in rec["done_beats"] or cur - int(rec["start"]) < off:
                continue
            rec["done_beats"].append(beat)
            try:
                _run_beat(session, rec, beat)
            except Exception:
                _log.debug("mourning beat failed", exc_info=True)


def _run_beat(session: Any, rec: Dict[str, Any], beat: str) -> None:
    obit, tid, name = rec["obituary"], rec["team_id"], rec["name"]
    team = _team_by_id(session, tid)
    first = _first(name)
    r = _rng("beat", rec["player_id"], beat)
    if beat == "eulogy":
        q = obit["tributes"][0]
        _storyline(session, f"{q['who']}: \"We'll carry him with us\"", f"The captain spoke for the room about {name}. {q['text']}", team_id=tid, heat=66)
    elif beat == "tributes":
        league = getattr(getattr(session, "sim", None), "league", None)
        others = [t for t in list(getattr(league, "teams", None) or []) if _tid(t) != tid]
        rival = r.choice(others) if others else None
        if rival is not None:
            _storyline(session, f"The {_team_name(rival)} wear {first}'s number in warmups", "Rivalries stop at moments like this.", heat=60)
        if obit["career"].get("teams"):
            _storyline(session, f"{', '.join(obit['career']['teams'][-2:])} remember {name}", "His former clubs held tributes before their games; the crowds stood for the whole minute.", heat=55)
    elif beat == "funeral":
        _storyline(session, f"The {obit['team_name']} attend {name}'s funeral", f"The whole team travelled to {obit['hometown'] or 'his hometown'}. {obit['survived_by']}", team_id=tid, heat=70)
        if team is not None:
            _team_morale(team, -0.04)
        if rec["friend_id"] and rec.get("response") == "grief" and r.random() < 0.6:
            try:
                from app.sim_engine.franchise.storyline_engine import _u_set_player_availability

                g = r.randint(2, 5)
                _u_set_player_availability(session, rec["friend_id"], status="personal_leave", reason=f"Bereavement leave after {name}'s death", games=g)
                _storyline(session, f"{rec['friend_name']} steps away from the team", f"He'll miss about {g} games to be with {first}'s family.", team_id=tid, heat=58)
            except Exception:
                _log.debug("leave failed", exc_info=True)
    elif beat == "tribute_night":
        _storyline(session, f"Tribute night for {name}", f"His family dropped the ceremonial puck. The building was loud for him all night.", team_id=tid, heat=72)
        if team is not None:
            _team_morale(team, 0.06)
            for p in _roster(team):
                _ready(session, _pid(p), 1.5, 2, f"Tribute night for {name}", f"mourn_night_{rec['player_id']}")
        _fans(session, tid, 3.0)
    elif beat == "foundation":
        _storyline(session, f"The {name} Foundation is launched", f"It will fund youth hockey in {obit['hometown'] or 'his hometown'} and mental-health programmes for young players.", team_id=tid, heat=55)
        _fans(session, tid, 2.0)
    elif beat == "reform":
        _storyline(session, "League responds after the tragedy", REFORM.get(obit["reform_key"], REFORM["medical"]), heat=58)
    elif beat == "inquiry":
        _storyline(session, f"Police open an inquiry into {name}'s death", "Investigators say the circumstances need a closer look. The club is cooperating.", team_id=tid, heat=85, typ="investigation")
    elif beat == "checkin":
        ok = rec.get("response") == "rally"
        _storyline(session, f"Three weeks on, the {obit['team_name']} {'are playing for him' if ok else 'are still hurting'}",
                   "The room has found its voice again." if ok else "The staff are giving players space; results have dipped.", team_id=tid, heat=55)
        if team is not None:
            _team_morale(team, 0.04 if ok else 0.02)
    elif beat == "legacy":
        car = obit["career"]
        elite = obit["ovr"] >= 85 or (car.get("pts") or 0) >= 600 or (car.get("gp") or 0) >= 800
        if elite:
            st = _state(session)
            award = f"{name} Memorial Trophy"
            st["league_memorials"].append({"award": award, "player": name, "season": rec["season"]})
            _storyline(session, f"League creates the {award}", "It will go each year to the player who best shows his character on and off the ice.", heat=62)
            pl = _find_player(session, rec["player_id"])
            if pl is not None:
                try:
                    pl.hof_fast_track = True
                except Exception:
                    pass
        if rec.get("number_retired"):
            _storyline(session, f"{first}'s number goes to the rafters", f"The {obit['team_name']} raised his banner with his family at centre ice.", team_id=tid, heat=68)
        elif rec.get("honor") == "later" and rec.get("beloved"):
            _fans(session, tid, -2.0)
    elif beat == "family_meeting":
        if rec.get("user_team"):
            good = rec.get("honor") in ("retire", "banner") and rec.get("support") in ("counsel", "days_off")
            if team is not None:
                for p in _roster(team):
                    ent = (getattr(session, "universe_players", None) or {}).get(_pid(p))
                    if isinstance(ent, dict):
                        s = ent.setdefault("state", {})
                        s["gm_trust"] = max(0.0, min(100.0, float(s.get("gm_trust", 55) or 55) + (4.0 if good else -3.0)))
            _storyline(session, f"{name}'s family meets with the GM",
                       "They thanked the club for how it has looked after them. The players noticed." if good
                       else "The family left the meeting disappointed. Word got back to the room.", team_id=tid, heat=55)
    elif beat == "number_request":
        if not rec.get("number_retired") and team is not None:
            young = [p for p in _roster(team) if _age(p) <= 23]
            if young:
                kid = max(young, key=_ovr)
                _storyline(session, f"{_name(kid)} asks to wear {first}'s number", f"With the family's blessing, he'll wear it this season.", team_id=tid, heat=55)
                _morale(session, _pid(kid), 5.0)
    elif beat == "anniversary":
        _storyline(session, f"One year since {name} died", "His club and the league paused again tonight to remember him.", team_id=tid, heat=55)


def _find_player(session: Any, pid: str) -> Any:
    league = getattr(getattr(session, "sim", None), "league", None)
    for t in list(getattr(league, "teams", None) or []):
        for p in _roster(t) + list(getattr(t, "ahl_roster", None) or []):
            if _pid(p) == str(pid):
                return p
    return None


def popup_fields(obit: Dict[str, Any], rec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extra fields for the obituary popup."""
    fam = obit.get("family") or {}
    return {
        "cause_story": obit.get("cause_story"), "survived_by": obit.get("survived_by"), "family": fam,
        "career": obit.get("career"), "legacy_line": obit.get("legacy_line"), "tributes": obit.get("tributes"),
        "hometown": obit.get("hometown"), "jersey_number": obit.get("jersey_number"),
        "mourning_plan": [
            "Every sim has been stopped.",
            "League moment of silence before the next games.",
            f"Memorial patch on {obit.get('team_name')} sweaters for the rest of the season.",
            "Funeral, tribute night and a foundation in his name are coming.",
        ],
        "suspicious": obit.get("suspicious", False),
    }
