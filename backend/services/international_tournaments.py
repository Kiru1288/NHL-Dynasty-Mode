"""Best-on-best international hockey: 4 Nations Face-Off, Winter Olympics, IIHF World Championship.

Real calendar
-------------
* **4 Nations Face-Off** — NHL/NHLPA event, Feb 12–20 2025 (Montreal & Boston): Canada, USA,
  Sweden, Finland; round robin (3 games each) then a final; replaced the All-Star Game.
  The NHL's mid-Olympic-cycle best-on-best returns in Feb 2028 and every four years after;
  the franchise runs it in the same 4 Nations format.
* **Winter Olympics** — every four years (Milano Cortina Feb 11–22 2026, French Alps 2030,
  Salt Lake City 2034). 12 nations in three groups of four; every team reaches the playoffs:
  the three group winners and the best runner-up get a bye to the quarterfinals, seeds 5–12
  play a qualification round. NHL schedule pauses around the tournament.
* **IIHF World Championship** — every May (2025 Stockholm/Herning May 9–25, 2026 Zurich/Fribourg
  May 15–31, 2027 Germany May 14–30). 16 nations, two groups of eight, top four of each to the
  quarterfinals. Runs during the Stanley Cup playoffs: only players whose clubs are out can go,
  and some stars decline.
* Russia has been excluded from IIHF events and the Olympics since 2022. In this franchise its
  players compete as neutral **Olympic Athletes from Russia (OAR)** at the Olympics and the Worlds
  (Olympic flag, no anthem, as in 2018). Belarus stays out.

These games mean something: medals move morale, go on the player's career awards, make news,
and a heated tournament (20% chance) can spill into the NHL: players ask to be traded across the
border, or rival countrymen stop talking in the room.

State lives on ``session.international`` = {"tournaments": {tid: bundle}, "history": [...], ...}.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging as _logging_swallow

_log = _logging_swallow.getLogger(__name__)

POLITICAL_TENSION_CHANCE = 0.20

# code -> (name, flag iso2)
NATIONS: Dict[str, Tuple[str, str]] = {
    "CAN": ("Canada", "ca"), "USA": ("United States", "us"), "SWE": ("Sweden", "se"), "FIN": ("Finland", "fi"),
    "CZE": ("Czechia", "cz"), "SUI": ("Switzerland", "ch"), "SVK": ("Slovakia", "sk"), "GER": ("Germany", "de"),
    "LAT": ("Latvia", "lv"), "DEN": ("Denmark", "dk"), "NOR": ("Norway", "no"), "AUT": ("Austria", "at"),
    "FRA": ("France", "fr"), "KAZ": ("Kazakhstan", "kz"), "SLO": ("Slovenia", "si"), "ITA": ("Italy", "it"),
    "HUN": ("Hungary", "hu"), "GBR": ("Great Britain", "gb"), "POL": ("Poland", "pl"), "UKR": ("Ukraine", "ua"),
    # Russian players compete as neutral "Olympic Athletes from Russia" (Olympic flag, no anthem).
    "OAR": ("Olympic Athletes from Russia", "oar"), "BLR": ("Belarus", "by"), "JPN": ("Japan", "jp"),
}
# Baseline OVR for depth players a federation brings from its domestic league.
NATION_DEPTH_OVR = {
    "CAN": 80, "USA": 79, "SWE": 77, "FIN": 76, "CZE": 75, "SUI": 74, "OAR": 77, "SVK": 72, "GER": 72,
    "LAT": 69, "DEN": 68, "NOR": 67, "AUT": 66, "FRA": 65, "KAZ": 66, "SLO": 64, "ITA": 63, "HUN": 62,
    "GBR": 61, "POL": 61, "UKR": 61, "BLR": 67, "JPN": 59,
}
NAME_POOL_FOR = {
    "CAN": "Canada", "USA": "USA", "SWE": "Sweden", "FIN": "Finland", "CZE": "Czechia", "SUI": "Switzerland",
    "SVK": "Slovakia", "GER": "Germany", "LAT": "Latvia", "DEN": "Denmark", "NOR": "Norway", "AUT": "Germany",
    "FRA": "France", "KAZ": "Kazakhstan", "SLO": "Czechia", "ITA": "Switzerland", "HUN": "Czechia", "GBR": "UK",
    "POL": "Poland", "UKR": "Ukraine", "OAR": "Russia", "BLR": "Belarus", "JPN": "Japan",
}
COUNTRY_ALIASES = {
    "CANADA": "CAN", "CA": "CAN", "UNITED STATES": "USA", "UNITED STATES OF AMERICA": "USA", "US": "USA", "AMERICA": "USA",
    "SWEDEN": "SWE", "SE": "SWE", "FINLAND": "FIN", "FI": "FIN", "CZECHIA": "CZE", "CZECH REPUBLIC": "CZE", "CZ": "CZE",
    "SWITZERLAND": "SUI", "CH": "SUI", "SLOVAKIA": "SVK", "SK": "SVK", "GERMANY": "GER", "DE": "GER", "DEU": "GER",
    "LATVIA": "LAT", "LV": "LAT", "DENMARK": "DEN", "DK": "DEN", "NORWAY": "NOR", "NO": "NOR", "AUSTRIA": "AUT", "AT": "AUT",
    "FRANCE": "FRA", "FR": "FRA", "KAZAKHSTAN": "KAZ", "KZ": "KAZ", "SLOVENIA": "SLO", "SI": "SLO", "SVN": "SLO",
    "ITALY": "ITA", "IT": "ITA", "HUNGARY": "HUN", "HU": "HUN", "GREAT BRITAIN": "GBR", "UNITED KINGDOM": "GBR", "UK": "GBR",
    "GB": "GBR", "ENGLAND": "GBR", "SCOTLAND": "GBR", "POLAND": "POL", "PL": "POL", "UKRAINE": "UKR", "UA": "UKR",
    "RUSSIA": "OAR", "RU": "OAR", "RUS": "OAR", "RUSSIAN FEDERATION": "OAR", "BELARUS": "BLR", "BY": "BLR", "JAPAN": "JPN", "JP": "JPN", "CHE": "SUI", "DNK": "DEN",
}
RIVALS = {frozenset(p) for p in (("CAN", "USA"), ("SWE", "FIN"), ("CZE", "SVK"), ("SUI", "GER"), ("LAT", "KAZ"), ("NOR", "SWE"), ("DEN", "NOR"), ("OAR", "FIN"), ("OAR", "CAN"), ("OAR", "USA"), ("UKR", "OAR"))}

OLYMPIC_HOSTS = {2026: ("ITA", "Milano Cortina"), 2030: ("FRA", "French Alps"), 2034: ("USA", "Salt Lake City")}
OLYMPIC_FIELD_2026 = {"A": ["CAN", "SUI", "CZE", "FRA"], "B": ["FIN", "SWE", "SVK", "ITA"], "C": ["USA", "GER", "LAT", "DEN"]}
WORLDS_HOSTS = {2025: ("SWE", "Stockholm & Herning"), 2026: ("SUI", "Zurich & Fribourg"), 2027: ("GER", "Düsseldorf & Mannheim")}
WORLDS_DATES = {2025: (date(2025, 5, 9), date(2025, 5, 25)), 2026: (date(2026, 5, 15), date(2026, 5, 31)), 2027: (date(2027, 5, 14), date(2027, 5, 30))}
WORLDS_ROTATION = ["CZE", "FIN", "SWE", "SUI", "GER", "SVK", "LAT", "DEN", "NOR", "AUT"]
WORLDS_CORE = ["CAN", "USA", "SWE", "FIN", "CZE", "SUI", "GER", "SVK", "LAT", "DEN", "NOR", "AUT", "FRA", "KAZ"]
WORLDS_PROMOTED = ["SLO", "HUN", "GBR", "ITA", "POL", "UKR", "JPN"]

ROSTER_SIZE = {"four_nations": (13, 7, 3), "olympics": (14, 8, 3), "worlds": (14, 8, 3)}
KIND_LABEL = {"four_nations": "4 Nations Face-Off", "olympics": "Winter Olympics", "worlds": "IIHF World Championship"}


# --------------------------------------------------------------------------- calendar

def _second_weekday(year: int, month: int, weekday: int, nth: int = 2) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (nth - 1))


def four_nations_year(end_year: int) -> bool:
    end_year = int(end_year)
    return end_year == 2025 or (end_year >= 2028 and end_year % 4 == 0)


def olympic_year(end_year: int) -> bool:
    end_year = int(end_year)
    return end_year >= 2026 and end_year % 4 == 2


def four_nations_dates(end_year: int) -> Tuple[date, date]:
    if int(end_year) == 2025:
        return date(2025, 2, 12), date(2025, 2, 20)
    start = _second_weekday(int(end_year), 2, 2)  # second Wednesday
    return start, start + timedelta(days=8)


def olympic_dates(end_year: int) -> Tuple[date, date]:
    if int(end_year) == 2026:
        return date(2026, 2, 11), date(2026, 2, 22)
    start = _second_weekday(int(end_year), 2, 2)
    return start, start + timedelta(days=11)


def worlds_dates(end_year: int) -> Tuple[date, date]:
    if int(end_year) in WORLDS_DATES:
        return WORLDS_DATES[int(end_year)]
    start = _second_weekday(int(end_year), 5, 4)  # second Friday of May
    return start, start + timedelta(days=16)


def nhl_break_window(kind: str, end_year: int) -> Tuple[Optional[date], Optional[date]]:
    """NHL schedule pause around a February tournament (none for the Worlds)."""
    if kind == "olympics" and olympic_year(end_year):
        s, e = olympic_dates(end_year)
        return s - timedelta(days=5), e + timedelta(days=2)
    if kind == "four_nations" and four_nations_year(end_year):
        s, e = four_nations_dates(end_year)
        return s - timedelta(days=2), e
    return None, None


def tournaments_for_season(season_start_year: int) -> List[Dict[str, Any]]:
    sy = int(season_start_year)
    ey = sy + 1
    out: List[Dict[str, Any]] = []
    if four_nations_year(ey):
        s, e = four_nations_dates(ey)
        host = ("CAN", "Montreal & Boston") if ey == 2025 else (("USA", "Boston & Toronto") if ey % 8 == 0 else ("CAN", "Toronto & Boston"))
        out.append({"id": f"four_nations_{ey}", "kind": "four_nations", "year": ey, "start": s.isoformat(), "end": e.isoformat(),
                    "host": host[0], "host_label": host[1]})
    if olympic_year(ey):
        s, e = olympic_dates(ey)
        host = OLYMPIC_HOSTS.get(ey, (WORLDS_ROTATION[ey % len(WORLDS_ROTATION)], "TBD"))
        out.append({"id": f"olympics_{ey}", "kind": "olympics", "year": ey, "start": s.isoformat(), "end": e.isoformat(),
                    "host": host[0], "host_label": host[1]})
    s, e = worlds_dates(ey)
    host = WORLDS_HOSTS.get(ey, (WORLDS_ROTATION[ey % len(WORLDS_ROTATION)], ""))
    out.append({"id": f"worlds_{ey}", "kind": "worlds", "year": ey, "start": s.isoformat(), "end": e.isoformat(),
                "host": host[0], "host_label": host[1] or NATIONS.get(host[0], (host[0],))[0]})
    for t in out:
        t["name"] = f"{ey} {KIND_LABEL[t['kind']]}"
    return out


def calendar_markers(season_start_year: int) -> List[Dict[str, Any]]:
    """(eid, iso, type, title, priority, description) rows for season_anchor_event_markers."""
    rows = []
    for t in tournaments_for_season(season_start_year):
        nm = KIND_LABEL[t["kind"]]
        where = t.get("host_label") or ""
        final_word = "Gold Medal Game" if t["kind"] != "four_nations" else "Championship Game"
        rows.append((f"{t['id']}_start", t["start"], f"intl_{t['kind']}_start", f"{nm} Begins", "CRITICAL",
                     f"{nm}{f' · {where}' if where else ''}. Open the International hub for rosters and results."))
        rows.append((f"{t['id']}_final", t["end"], f"intl_{t['kind']}_final", f"{nm} {final_word}", "CRITICAL",
                     f"{nm} {final_word.lower()}."))
    return rows


# --------------------------------------------------------------------------- state & helpers

def _state(session: Any) -> Dict[str, Any]:
    st = getattr(session, "international", None)
    if not isinstance(st, dict):
        st = {"tournaments": {}, "history": []}
        session.international = st
    st.setdefault("tournaments", {})
    st.setdefault("history", [])
    return st


def _rng(*parts: Any) -> random.Random:
    seed = int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def _salt(session: Any) -> str:
    return str(getattr(session, "session_id", "") or getattr(session, "franchise_seed", "") or getattr(session, "user_team_id", "") or "x")


def _ce():
    from services import contract_economy as ce

    return ce


def country_code(p: Any) -> str:
    ce = _ce()
    ident = ce._get(p, "identity", None)
    for raw in (ce._get(ident, "nationality", ""), ce._get(p, "nationality", ""), ce._get(ident, "birth_country", ""), ce._get(p, "birth_country", "")):
        s = str(raw or "").strip().upper()
        if not s:
            continue
        if s in NATIONS:
            return s
        if s in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[s]
    return ""


def _pos_group(p: Any) -> str:
    try:
        b = str(_ce()._position_bucket(p))
    except Exception:
        b = ""
    if b == "G":
        return "G"
    if b in ("LD", "RD", "D"):
        return "D"
    return "F"


def _team_abbr(team: Any) -> str:
    for k in ("abbrev", "abbr", "abbreviation", "team_abbr"):
        v = getattr(team, k, None)
        if v:
            return str(v)
    return str(getattr(team, "team_id", "") or "")


def _player_row(p: Any, team: Any, level: str, user_tid: str) -> Dict[str, Any]:
    ce = _ce()
    tid = str(getattr(team, "team_id", "") or "") if team is not None else ""
    return {
        "id": str(ce._player_id(p)), "name": str(ce._player_name(p)), "pos": _pos_group(p),
        "position": str(ce._position_bucket(p) or ""), "ovr": round(float(ce._player_ovr(p)), 1),
        "age": int(ce._player_age(p) or 0), "club": _team_abbr(team) if team is not None else ("Free agent" if level == "FA" else level),
        "team_id": tid, "level": level, "is_user": bool(user_tid and tid == user_tid), "npc": False,
    }


def _pool_by_nation(session: Any, alive_team_ids: Optional[set] = None, include_minors: bool = True) -> Dict[str, List[Tuple[Any, Dict[str, Any]]]]:
    league = getattr(getattr(session, "sim", None), "league", None)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    pool: Dict[str, List[Tuple[Any, Dict[str, Any]]]] = {}
    seen = set()

    def add(p: Any, team: Any, level: str) -> None:
        try:
            pid = str(_ce()._player_id(p))
        except Exception:
            return
        if not pid or pid in seen:
            return
        code = country_code(p)
        if not code:
            return
        seen.add(pid)
        pool.setdefault(code, []).append((p, _player_row(p, team, level, user_tid)))

    for t in list(getattr(league, "teams", None) or []):
        tid = str(getattr(t, "team_id", "") or "")
        if alive_team_ids and tid in alive_team_ids:
            continue
        for p in list(getattr(t, "roster", None) or []) + list(getattr(t, "injured_reserve", None) or []):
            add(p, t, "NHL")
        if include_minors:
            for p in list(getattr(t, "ahl_roster", None) or []):
                add(p, t, "AHL")
    if include_minors:
        for attr, lvl in (("free_agents", "FA"), ("overseas_free_agents", "Europe")):
            for p in list(getattr(league, attr, None) or []):
                add(p, None, lvl)
    return pool


def _npc_name(rng: random.Random, code: str) -> str:
    try:
        from app.sim_engine.generation.name_generator import NAME_POOLS

        pool = NAME_POOLS.get(NAME_POOL_FOR.get(code, "Canada")) or NAME_POOLS.get("Canada") or {}
        return f"{rng.choice(pool.get('first') or ['Alex'])} {rng.choice(pool.get('last') or ['Smith'])}"
    except Exception:
        return f"{rng.choice(['Alex', 'Erik', 'Janis', 'Mika', 'Luca'])} {rng.choice(['Berg', 'Novak', 'Keller', 'Rossi'])}"


DOMESTIC_LEAGUE = {"SWE": "SHL", "FIN": "Liiga", "CZE": "Extraliga", "SUI": "National League", "SVK": "Extraliga", "GER": "DEL",
                   "LAT": "Latvia", "DEN": "Metal Ligaen", "NOR": "EHL", "AUT": "ICEHL", "FRA": "Ligue Magnus", "KAZ": "KHL",
                   "SLO": "ICEHL", "ITA": "ICEHL", "HUN": "Erste Liga", "GBR": "EIHL", "POL": "PHL", "UKR": "UHL", "OAR": "KHL",
                   "BLR": "Extraleague", "JPN": "Asia League", "CAN": "Europe", "USA": "Europe"}


def _pick_roster(session: Any, code: str, cands: List[Tuple[Any, Dict[str, Any]]], kind: str, tid: str,
                 nhl_only: bool, decline_stars: bool) -> List[Dict[str, Any]]:
    nf, nd, ng = ROSTER_SIZE[kind]
    rng = _rng(_salt(session), tid, code, "roster")
    rows = [r for _, r in cands if (not nhl_only or r["level"] == "NHL")]
    if decline_stars:
        kept = []
        for r in rows:
            if r["ovr"] >= 86 and _rng(_salt(session), tid, r["id"], "decline").random() < 0.5:
                continue
            if r["level"] == "NHL" and r["ovr"] >= 80 and _rng(_salt(session), tid, r["id"], "decline2").random() < 0.25:
                continue
            kept.append(r)
        rows = kept
    out: List[Dict[str, Any]] = []
    for grp, n in (("F", nf), ("D", nd), ("G", ng)):
        g = sorted((r for r in rows if r["pos"] == grp), key=lambda r: -r["ovr"])[:n]
        out.extend(dict(r) for r in g)
        base = NATION_DEPTH_OVR.get(code, 60) - (3 if grp == "G" else 0)
        lg = DOMESTIC_LEAGUE.get(code, "Europe")
        while sum(1 for r in out if r["pos"] == grp) < n:
            i = sum(1 for r in out if r["npc"])
            out.append({
                "id": f"npc_{tid}_{code}_{grp}_{i}", "name": _npc_name(rng, code), "pos": grp,
                "position": {"F": rng.choice(["C", "LW", "RW"]), "D": rng.choice(["LD", "RD"]), "G": "G"}[grp],
                "ovr": round(base - rng.uniform(0, 6), 1), "age": rng.randint(21, 33), "club": lg, "team_id": "",
                "level": lg, "is_user": False, "npc": True,
            })
    return out


def _strength(roster: List[Dict[str, Any]]) -> float:
    f = sorted((r["ovr"] for r in roster if r["pos"] == "F"), reverse=True)[:12]
    d = sorted((r["ovr"] for r in roster if r["pos"] == "D"), reverse=True)[:6]
    g = sorted((r["ovr"] for r in roster if r["pos"] == "G"), reverse=True)[:1]
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    return round(mean(f) * 0.5 + mean(d) * 0.3 + mean(g) * 0.2, 2)


def russia_reinstated(session: Any, end_year: int) -> bool:
    if int(end_year) < 2028:
        return False
    st = _state(session)
    rev = st.setdefault("russia_review", {})
    key = str(int(end_year))
    if key not in rev:
        prev = rev.get(str(int(end_year) - 1))
        rev[key] = bool(prev) or (_rng(_salt(session), "russia", end_year).random() < 0.30)
    return bool(rev[key])


# --------------------------------------------------------------------------- field

def _field(session: Any, meta: Dict[str, Any], strengths: Dict[str, float]) -> Dict[str, List[str]]:
    kind, ey, host = meta["kind"], int(meta["year"]), meta.get("host")
    allowed = [c for c in NATIONS if c != "BLR"]
    if kind == "four_nations":
        return {"RR": ["CAN", "USA", "SWE", "FIN"]}
    if kind == "olympics":
        if ey == 2026:
            # Real Milano Cortina groups, with OAR taking the weakest non-host slot.
            groups = {k: list(v) for k, v in OLYMPIC_FIELD_2026.items()}
            pool = [c for g in groups.values() for c in g if c != host]
            weakest = min(pool, key=lambda c: strengths.get(c, 0))
            for g in groups.values():
                if weakest in g:
                    g[g.index(weakest)] = "OAR"
            return groups
        ranked = sorted((c for c in allowed if c != host), key=lambda c: -strengths.get(c, 0))[:11]
        teams = ([host] if host else []) + ranked
        teams = teams[:12]
        seeded = sorted(teams, key=lambda c: -strengths.get(c, 0))
        groups = {"A": [], "B": [], "C": []}
        order = ["A", "B", "C", "C", "B", "A"] * 2
        for i, c in enumerate(seeded):
            groups[order[i]].append(c)
        return groups
    core = [c for c in WORLDS_CORE if c in allowed] + ["OAR"]
    promoted = [c for c in WORLDS_PROMOTED if c in allowed]
    r = _rng(_salt(session), "worlds_field", ey)
    r.shuffle(promoted)
    teams = list(dict.fromkeys(([host] if host in allowed else []) + core + promoted))[:16]
    seeded = sorted(teams, key=lambda c: -strengths.get(c, 0))
    groups = {"A": [], "B": []}
    order = ["A", "B", "B", "A"] * 4
    for i, c in enumerate(seeded):
        groups[order[i]].append(c)
    return groups


# --------------------------------------------------------------------------- game sim

def _poisson(rng: random.Random, lam: float) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _pick_weighted(rng: random.Random, rows: List[Dict[str, Any]], wfn) -> Dict[str, Any]:
    ws = [max(0.05, wfn(r)) for r in rows]
    t = sum(ws)
    x = rng.random() * t
    for r, w in zip(rows, ws):
        x -= w
        if x <= 0:
            return r
    return rows[-1]


def _sim_game(rng: random.Random, a: str, b: str, rosters: Dict[str, List[Dict[str, Any]]], strengths: Dict[str, float],
              knockout: bool, host: str = "") -> Dict[str, Any]:
    sa, sb = strengths[a] + (0.8 if a == host else 0), strengths[b] + (0.8 if b == host else 0)
    # Best-on-best mismatches still end around 5-1, not 12-0.
    lam_a = max(0.9, min(4.8, 2.8 * math.exp((sa - sb) / 15.0)))
    lam_b = max(0.9, min(4.8, 2.8 * math.exp((sb - sa) / 15.0)))
    ga, gb = _poisson(rng, lam_a), _poisson(rng, lam_b)
    decided = "REG"
    if ga == gb:
        p_a = 1 / (1 + math.exp(-(sa - sb) / 4.0))
        if knockout or rng.random() < 0.62:
            decided = "OT"
        else:
            decided = "SO"
        if rng.random() < p_a:
            ga += 1
        else:
            gb += 1
    goals: List[Dict[str, Any]] = []
    for side, n in ((a, ga), (b, gb)):
        n_play = n - (1 if decided == "SO" and ((side == a and ga > gb) or (side == b and gb > ga)) else 0)
        skaters = [r for r in rosters[side] if r["pos"] != "G"]
        for _ in range(max(0, n_play)):
            sc = _pick_weighted(rng, skaters, lambda r: ((r["ovr"] - 55) ** 2) * (1.0 if r["pos"] == "F" else 0.4))
            others = [r for r in skaters if r["id"] != sc["id"]]
            ast = []
            for _k in range(rng.choice([0, 1, 2, 2, 2])):
                if not others:
                    break
                x = _pick_weighted(rng, others, lambda r: ((r["ovr"] - 55) ** 2) * (1.0 if r["pos"] == "F" else 0.7))
                ast.append(x["id"])
                others = [r for r in others if r["id"] != x["id"]]
            goals.append({"team": side, "scorer": sc["id"], "assists": ast})
    goalies = {}
    for side in (a, b):
        gs = sorted((r for r in rosters[side] if r["pos"] == "G"), key=lambda r: -r["ovr"])
        goalies[side] = (gs[0] if (len(gs) < 2 or rng.random() < 0.8) else gs[1])["id"] if gs else ""
    shots = {a: max(ga + 12, int(round(30 * math.exp((sa - sb) / 25.0) + rng.uniform(-5, 5)))),
             b: max(gb + 12, int(round(30 * math.exp((sb - sa) / 25.0) + rng.uniform(-5, 5))))}
    return {"home": a, "away": b, "home_score": ga, "away_score": gb, "decided": decided, "goals": goals,
            "goalies": goalies, "shots": shots, "winner": a if ga > gb else b, "loser": b if ga > gb else a}


def _table(codes: List[str], games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = {c: {"code": c, "gp": 0, "w": 0, "otw": 0, "otl": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for c in codes}
    for g in games:
        h, a = g["home"], g["away"]
        if h not in rows or a not in rows:
            continue
        for side, gf, ga_ in ((h, g["home_score"], g["away_score"]), (a, g["away_score"], g["home_score"])):
            r = rows[side]
            r["gp"] += 1
            r["gf"] += gf
            r["ga"] += ga_
            won = gf > ga_
            if g["decided"] == "REG":
                r["w" if won else "l"] += 1
                r["pts"] += 3 if won else 0
            else:
                r["otw" if won else "otl"] += 1
                r["pts"] += 2 if won else 1
    out = sorted(rows.values(), key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"]))
    for i, r in enumerate(out):
        r["rank"] = i + 1
        r["gd"] = r["gf"] - r["ga"]
    return out


def _rr_pairs(codes: List[str]) -> List[List[Tuple[str, str]]]:
    teams = list(codes) + (["BYE"] if len(codes) % 2 else [])
    n = len(teams)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            x, y = teams[i], teams[n - 1 - i]
            if "BYE" not in (x, y):
                pairs.append((x, y) if r % 2 == 0 else (y, x))
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def _simulate(session: Any, meta: Dict[str, Any], groups: Dict[str, List[str]], rosters: Dict[str, List[Dict[str, Any]]],
              strengths: Dict[str, float]) -> Dict[str, Any]:
    rng = _rng(_salt(session), meta["id"], "games")
    start = date.fromisoformat(meta["start"])
    end = date.fromisoformat(meta["end"])
    host = meta.get("host", "")
    games: List[Dict[str, Any]] = []

    def play(a: str, b: str, day: date, stage: str, label: str, knockout: bool) -> Dict[str, Any]:
        g = _sim_game(rng, a, b, rosters, strengths, knockout, host)
        g.update({"date": day.isoformat(), "stage": stage, "label": label, "id": f"{meta['id']}_{len(games)}"})
        games.append(g)
        return g

    kind = meta["kind"]
    placements: List[str] = []
    if kind == "four_nations":
        codes = groups["RR"]
        # 2025 order: CAN-SWE, USA-FIN, SWE-FIN, CAN-USA, CAN-FIN, USA-SWE
        sched = [(("CAN", "SWE"), 0), (("USA", "FIN"), 1), (("SWE", "FIN"), 3), (("CAN", "USA"), 3), (("CAN", "FIN"), 5), (("USA", "SWE"), 5)]
        for (a, b), off in sched:
            play(a, b, start + timedelta(days=off), "round_robin", "Round robin", False)
        table = _table(codes, games)
        fin = play(table[0]["code"], table[1]["code"], end, "final", "Championship game", True)
        placements = [fin["winner"], fin["loser"], table[2]["code"], table[3]["code"]]
        standings = {"RR": table}
        third = None
    elif kind == "olympics":
        standings = {}
        for gname, codes in groups.items():
            for ri, pairs in enumerate(_rr_pairs(codes)):
                for a, b in pairs:
                    play(a, b, start + timedelta(days=min(4, ri * 2 + (0 if gname == "B" else 1))), f"group_{gname}", f"Group {gname}", False)
            standings[gname] = _table(codes, [g for g in games if g["stage"] == f"group_{gname}"])
        winners = sorted((standings[g][0] for g in standings), key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
        seconds = sorted((standings[g][1] for g in standings), key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
        thirds = sorted((standings[g][2] for g in standings), key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
        fourths = sorted((standings[g][3] for g in standings), key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
        seeds = [r["code"] for r in winners + seconds + thirds + fourths]
        q_day, qf_day, sf_day, final_day = start + timedelta(days=6), start + timedelta(days=7), start + timedelta(days=9), end
        qual = [play(seeds[i], seeds[15 - i], q_day, "qualification", "Qualification playoff", True) for i in range(4, 8)]
        # seeds 5v12, 6v11, 7v10, 8v9 -> winners face 4,3,2,1
        qw = [g["winner"] for g in qual]
        qfs = [play(seeds[0], qw[3], qf_day, "quarterfinal", "Quarterfinal", True), play(seeds[1], qw[2], qf_day, "quarterfinal", "Quarterfinal", True),
               play(seeds[2], qw[1], qf_day, "quarterfinal", "Quarterfinal", True), play(seeds[3], qw[0], qf_day, "quarterfinal", "Quarterfinal", True)]
        sfs = [play(qfs[0]["winner"], qfs[3]["winner"], sf_day, "semifinal", "Semifinal", True),
               play(qfs[1]["winner"], qfs[2]["winner"], sf_day, "semifinal", "Semifinal", True)]
        third = play(sfs[0]["loser"], sfs[1]["loser"], end - timedelta(days=1), "bronze", "Bronze medal game", True)
        fin = play(sfs[0]["winner"], sfs[1]["winner"], final_day, "final", "Gold medal game", True)
        placements = [fin["winner"], fin["loser"], third["winner"], third["loser"]]
        rest = [c for c in seeds if c not in placements]
        placements += rest
    else:
        standings = {}
        for gname, codes in groups.items():
            for ri, pairs in enumerate(_rr_pairs(codes)):
                for a, b in pairs:
                    play(a, b, start + timedelta(days=min(11, round(ri * 1.6) + (1 if gname == "B" else 0))), f"group_{gname}", f"Group {gname}", False)
            standings[gname] = _table(codes, [g for g in games if g["stage"] == f"group_{gname}"])
        A, B = [r["code"] for r in standings["A"]], [r["code"] for r in standings["B"]]
        qf_day, sf_day = end - timedelta(days=3), end - timedelta(days=1)
        qfs = [play(A[0], B[3], qf_day, "quarterfinal", "Quarterfinal", True), play(A[1], B[2], qf_day, "quarterfinal", "Quarterfinal", True),
               play(B[0], A[3], qf_day, "quarterfinal", "Quarterfinal", True), play(B[1], A[2], qf_day, "quarterfinal", "Quarterfinal", True)]
        sfs = [play(qfs[0]["winner"], qfs[3]["winner"], sf_day, "semifinal", "Semifinal", True),
               play(qfs[1]["winner"], qfs[2]["winner"], sf_day, "semifinal", "Semifinal", True)]
        third = play(sfs[0]["loser"], sfs[1]["loser"], end, "bronze", "Bronze medal game", True)
        fin = play(sfs[0]["winner"], sfs[1]["winner"], end, "final", "Gold medal game", True)
        placements = [fin["winner"], fin["loser"], third["winner"], third["loser"]]
        placements += [c for c in A[:4] + B[:4] if c not in placements] + [c for c in A[4:] + B[4:]]
    return {"games": games, "standings": standings, "placements": placements}


def _player_stats(games: List[Dict[str, Any]], rosters: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for code, roster in rosters.items():
        for r in roster:
            stats[r["id"]] = {"id": r["id"], "code": code, "gp": 0, "g": 0, "a": 0, "pts": 0, "w": 0, "ga": 0, "sa": 0}
    for g in games:
        for side in (g["home"], g["away"]):
            for r in rosters.get(side, []):
                if r["pos"] != "G":
                    stats[r["id"]]["gp"] += 1
            gid = g["goalies"].get(side)
            if gid in stats:
                st = stats[gid]
                st["gp"] += 1
                opp = g["away"] if side == g["home"] else g["home"]
                st["ga"] += g["away_score"] if side == g["home"] else g["home_score"]
                st["sa"] += int(g["shots"].get(opp, 30))
                if g["winner"] == side:
                    st["w"] += 1
        for goal in g["goals"]:
            if goal["scorer"] in stats:
                stats[goal["scorer"]]["g"] += 1
                stats[goal["scorer"]]["pts"] += 1
            for a in goal["assists"]:
                if a in stats:
                    stats[a]["a"] += 1
                    stats[a]["pts"] += 1
    for st in stats.values():
        if st["sa"]:
            st["sv_pct"] = round(1 - st["ga"] / st["sa"], 3)
    return stats


# --------------------------------------------------------------------------- build / complete

def _season_alive_playoff_teams(session: Any) -> set:
    """Clubs still playing for the Cup when the Worlds roster is picked."""
    live = getattr(session, "playoff_live", None)
    alive = set()
    if isinstance(live, dict) and not live.get("completed"):
        cur = int(live.get("current_round") or 1)
        for s in live.get("series") or []:
            if int(s.get("round_index") or 0) >= cur and s.get("status") in ("active", "pending"):
                for k in ("team_high_id", "team_low_id"):
                    if s.get(k):
                        alive.add(str(s.get(k)))
    return alive


def build_rosters(session: Any, meta: Dict[str, Any], alive: Optional[set] = None) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
    kind = meta["kind"]
    pool = _pool_by_nation(session, alive_team_ids=alive if kind == "worlds" else None, include_minors=kind != "four_nations")
    nations = list(NATIONS)
    rosters: Dict[str, List[Dict[str, Any]]] = {}
    for code in nations:
        rosters[code] = _pick_roster(session, code, pool.get(code, []), kind, meta["id"], nhl_only=(kind == "four_nations"),
                                     decline_stars=(kind == "worlds"))
    strengths = {c: _strength(r) for c, r in rosters.items()}
    groups = _field(session, meta, strengths)
    entered = {c for g in groups.values() for c in g}
    rosters = {c: r for c, r in rosters.items() if c in entered}
    strengths = {c: s for c, s in strengths.items() if c in entered}
    return groups, rosters, strengths


def ensure_tournament(session: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    st = _state(session)
    b = st["tournaments"].get(meta["id"])
    if isinstance(b, dict) and b.get("games"):
        return b
    alive = _season_alive_playoff_teams(session) if meta["kind"] == "worlds" else None
    groups, rosters, strengths = build_rosters(session, meta, alive)
    res = _simulate(session, meta, groups, rosters, strengths)
    b = {**meta, "groups": groups, "rosters": rosters, "strengths": strengths, **res,
         "stats": _player_stats(res["games"], rosters), "completed": False, "applied": False, "story": []}
    st["tournaments"][meta["id"]] = b
    return b


def _bump_morale(session: Any, pid: str, delta: float) -> None:
    ents = getattr(session, "universe_players", None) or {}
    ent = ents.get(str(pid))
    if isinstance(ent, dict):
        s = ent.setdefault("state", {})
        s["morale"] = max(0.0, min(100.0, float(s.get("morale", 55) or 55) + float(delta)))


def _find_player_obj(session: Any, pid: str) -> Tuple[Any, Any]:
    league = getattr(getattr(session, "sim", None), "league", None)
    for t in list(getattr(league, "teams", None) or []):
        for p in list(getattr(t, "roster", None) or []) + list(getattr(t, "ahl_roster", None) or []) + list(getattr(t, "injured_reserve", None) or []):
            try:
                if str(_ce()._player_id(p)) == str(pid):
                    return p, t
            except Exception:
                continue
    return None, None


def _add_award(session: Any, pid: str, label: str, year: int) -> None:
    p, _t = _find_player_obj(session, pid)
    if p is None:
        return
    try:
        awards = getattr(p, "career_awards", None)
        if not isinstance(awards, list):
            awards = []
            setattr(p, "career_awards", awards)
        awards.append({"award": label, "name": label, "season": int(year), "year": int(year), "kind": "international"})
    except Exception:
        _log.debug("award attach failed", exc_info=True)


def _storyline(session: Any, headline: str, summary: str, team_id: str = "", heat: int = 60, typ: str = "international") -> None:
    try:
        from app.sim_engine.franchise.state import _record_storyline

        _record_storyline(session, {"headline": headline, "summary": summary, "team_id": team_id, "category": "international",
                                    "type": typ, "priority": "HIGH" if heat >= 70 else "MEDIUM", "heat": heat})
    except Exception:
        _log.debug("storyline failed", exc_info=True)


def _mvp(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = {r["id"]: (c, r) for c, rs in b["rosters"].items() for r in rs}
    gold = b["placements"][0] if b.get("placements") else ""
    best = None
    for pid, s in b["stats"].items():
        code, r = rows.get(pid, ("", {}))
        if r.get("pos") == "G":
            score = s.get("w", 0) * 1.6 + (s.get("sv_pct", 0.88) - 0.9) * 60
        else:
            score = s["pts"] + s["g"] * 0.3
        score += 1.5 if code == gold else 0
        if best is None or score > best[0]:
            best = (score, pid, code, r)
    if not best:
        return None
    _s, pid, code, r = best
    return {"id": pid, "name": r.get("name"), "code": code, "npc": r.get("npc", False), "stats": b["stats"].get(pid)}


def _apply_tension(session: Any, b: Dict[str, Any]) -> None:
    rng = _rng(_salt(session), b["id"], "tension")
    try:
        from app.sim_engine.tuning.command_knobs import knob as _cmd_knob

        _tension_p = min(0.95, POLITICAL_TENSION_CHANCE * max(0.0, float(_cmd_knob("political_tension", 1.0))))
    except Exception:
        _tension_p = POLITICAL_TENSION_CHANCE
    if rng.random() >= _tension_p:
        b["tension"] = {"heated": False}
        return
    met = {frozenset((g["home"], g["away"])) for g in b["games"]}
    pair = frozenset(("CAN", "USA")) if frozenset(("CAN", "USA")) in met else None
    if pair is None:
        fin = next((g for g in b["games"] if g["stage"] == "final"), None)
        if fin and frozenset((fin["home"], fin["away"])) in RIVALS:
            pair = frozenset((fin["home"], fin["away"]))
        elif fin:
            pair = frozenset((fin["home"], fin["away"]))
    if not pair:
        b["tension"] = {"heated": False}
        return
    a, c = sorted(pair)
    name = KIND_LABEL[b["kind"]]
    tension = {"heated": True, "pair": [a, c], "demands": [], "frost": []}
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if pair == frozenset(("CAN", "USA")):
        _storyline(session, f"Politics boil over at the {name}",
                   "Booed anthems, line brawls and a war of words between governments. Canada–USA has never felt this heated, and NHL dressing rooms are feeling it.",
                   heat=85, typ="international_tension")
        try:
            from services.trade_demand_engine import CANADA_ABBRS, open_trade_demand
        except Exception:
            CANADA_ABBRS, open_trade_demand = set(), None
        league = getattr(getattr(session, "sim", None), "league", None)
        cands = []
        for t in list(getattr(league, "teams", None) or []):
            in_canada = _team_abbr(t) in CANADA_ABBRS
            for p in list(getattr(t, "roster", None) or []):
                code = country_code(p)
                if (code == "CAN" and not in_canada) or (code == "USA" and in_canada):
                    try:
                        ovr = float(_ce()._player_ovr(p))
                        age = int(_ce()._player_age(p) or 27)
                        prof = _ce()._player_negotiation_profile(p)
                    except Exception:
                        continue
                    if ovr < 74 or age < 23:
                        continue
                    pid = str(_ce()._player_id(p))
                    played = any(r["id"] == pid for r in b["rosters"].get(code, []))
                    chance = 0.04 + (0.08 if played else 0.0) + (1.0 - float(prof.get("loyalty", 0.5))) * 0.06
                    u = _rng(_salt(session), b["id"], pid, "demand").random()
                    is_user = str(getattr(t, "team_id", "")) == user_tid
                    if u < chance * (1.6 if is_user else 1.0):
                        cands.append((chance - u + (0.5 if is_user else 0), p, t, code))
        cands.sort(key=lambda x: -x[0])
        for _g, p, t, code in cands[:3]:
            reason = "political_canada" if code == "CAN" else "political_usa"
            if open_trade_demand is not None:
                try:
                    open_trade_demand(session, p, t, reason=reason, calendar_idx=int(getattr(session, "calendar_cursor", 0) or 0), force_formal=True)
                except Exception:
                    _log.debug("tension demand failed", exc_info=True)
            tension["demands"].append({"id": str(_ce()._player_id(p)), "name": str(_ce()._player_name(p)), "team": _team_abbr(t),
                                       "wants": "Canada" if code == "CAN" else "the United States", "is_user": str(getattr(t, "team_id", "")) == user_tid})
    else:
        _storyline(session, f"{NATIONS[a][0]}–{NATIONS[c][0]} feud spills into the NHL",
                   f"Bad blood from the {name} followed players home. Countrymen of the two nations aren't speaking in a few dressing rooms.", heat=70,
                   typ="international_tension")
        league = getattr(getattr(session, "sim", None), "league", None)
        for t in list(getattr(league, "teams", None) or []):
            roster = list(getattr(t, "roster", None) or [])
            codes = {country_code(p) for p in roster}
            if a in codes and c in codes:
                for p in roster:
                    if country_code(p) in (a, c):
                        _bump_morale(session, str(_ce()._player_id(p)), -4.0)
                tension["frost"].append(_team_abbr(t))
    b["tension"] = tension


def complete_tournament(session: Any, b: Dict[str, Any]) -> None:
    if b.get("applied"):
        b["completed"] = True
        return
    b["completed"] = True
    b["applied"] = True
    year = int(b["year"])
    kind = b["kind"]
    name = KIND_LABEL[kind]
    pl = b.get("placements") or []
    medals = {0: "Gold", 1: "Silver", 2: "Bronze"}
    big = kind in ("olympics", "four_nations")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    user_lines = []
    for idx, code in enumerate(pl):
        for r in b["rosters"].get(code, []):
            if r.get("npc"):
                continue
            if idx in medals:
                _bump_morale(session, r["id"], {0: 6.0, 1: 3.0, 2: 3.0}[idx] * (1.0 if big else 0.6))
                if kind == "four_nations" and idx == 0:
                    _add_award(session, r["id"], "4 Nations Face-Off Champion", year)
                elif kind != "four_nations":
                    _add_award(session, r["id"], f"{'Olympic' if kind == 'olympics' else 'World Championship'} {medals[idx]}", year)
            elif big and code in ("CAN", "USA", "SWE", "FIN"):
                _bump_morale(session, r["id"], -2.0)
            if r.get("is_user"):
                place = medals.get(idx, f"{idx + 1}th")
                user_lines.append(f"{r['name']} ({code}, {place.lower() if idx in medals else place})")
    mvp = _mvp(b)
    if mvp and not mvp.get("npc"):
        _bump_morale(session, mvp["id"], 4.0)
        _add_award(session, mvp["id"], f"{'Olympic' if kind == 'olympics' else '4 Nations' if kind == 'four_nations' else 'World Championship'} MVP", year)
    b["mvp"] = mvp
    if pl:
        gold = NATIONS.get(pl[0], (pl[0],))[0]
        silver = NATIONS.get(pl[1], (pl[1],))[0] if len(pl) > 1 else ""
        fin = next((g for g in b["games"] if g["stage"] == "final"), None)
        score = f" {max(fin['home_score'], fin['away_score'])}-{min(fin['home_score'], fin['away_score'])}{' in overtime' if fin and fin['decided'] == 'OT' else ''}" if fin else ""
        word = "gold" if kind != "four_nations" else "the 4 Nations title"
        _storyline(session, f"{gold} wins {word} at the {year} {name}",
                   f"{gold} beat {silver}{score}." + (f" Tournament MVP: {mvp['name']}." if mvp else ""), heat=80 if big else 60)
    if user_lines:
        _storyline(session, f"Your players at the {name}", "; ".join(user_lines[:8]) + ".", team_id=user_tid, heat=55)
    _apply_tension(session, b)
    st = _state(session)
    st["history"] = (st["history"] + [{"id": b["id"], "kind": kind, "year": year, "name": b.get("name"), "podium": pl[:3],
                                       "mvp": (mvp or {}).get("name"), "tension": bool((b.get("tension") or {}).get("heated"))}])[-40:]


# --------------------------------------------------------------------------- hooks

def daily_tick(session: Any, iso: str) -> None:
    """Called once per simulated calendar day. Builds a tournament when its window opens and
    applies its consequences once the final has been played."""
    try:
        today = date.fromisoformat(str(iso)[:10])
    except Exception:
        return
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    for meta in tournaments_for_season(sy):
        if meta["kind"] == "worlds":
            continue  # the Worlds run off the playoff bracket (see playoff_tick)
        s, e = date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])
        if today >= s - timedelta(days=1):
            b = ensure_tournament(session, meta)
            if today >= e and not b.get("applied"):
                complete_tournament(session, b)


def playoff_tick(session: Any, current_round: int, completed: bool = False) -> None:
    """The Worlds start once the second round is under way: rosters come from eliminated clubs."""
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    if int(current_round or 1) < 2 and not completed:
        return
    for meta in tournaments_for_season(sy):
        if meta["kind"] != "worlds":
            continue
        b = ensure_tournament(session, meta)
        if (completed or int(current_round or 1) >= 3) and not b.get("applied"):
            complete_tournament(session, b)


def catch_up(session: Any) -> None:
    """Offseason safety net: anything this season that should be finished is finished."""
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    phase = str(getattr(session, "phase", "") or "").lower()
    if phase not in ("offseason", "post_cup", "complete"):
        return
    for meta in tournaments_for_season(sy):
        b = ensure_tournament(session, meta)
        if not b.get("applied"):
            complete_tournament(session, b)


# --------------------------------------------------------------------------- client payload

def _today(session: Any) -> str:
    try:
        from services.franchise_sim import _today_iso

        return _today_iso(session) or ""
    except Exception:
        return ""


def _nation(code: str) -> Dict[str, Any]:
    nm, iso2 = NATIONS.get(code, (code, ""))
    return {"code": code, "name": nm, "flag": iso2}


def hub_payload(session: Any) -> Dict[str, Any]:
    catch_up(session)
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    st = _state(session)
    today = _today(session)
    rows = []
    for meta in tournaments_for_season(sy):
        b = st["tournaments"].get(meta["id"]) or {}
        status = "complete" if b.get("completed") else ("live" if b.get("games") and today and meta["start"] <= today <= meta["end"] else
                                                        ("ready" if b.get("games") else "upcoming"))
        if meta["kind"] == "worlds" and b.get("games") and not b.get("completed"):
            status = "live"
        rows.append({**meta, "label": KIND_LABEL[meta["kind"]], "status": status, "host_nation": _nation(meta.get("host", "")),
                     "podium": [_nation(c) for c in (b.get("placements") or [])[:3]] if b.get("completed") else []})
    nxt = []
    for k in range(1, 6):
        for meta in tournaments_for_season(sy + k):
            if meta["kind"] != "worlds":
                nxt.append({"id": meta["id"], "name": meta["name"], "start": meta["start"], "kind": meta["kind"]})
        if len(nxt) >= 2:
            break
    return {"ok": True, "season": sy, "today": today, "tournaments": rows, "next_best_on_best": nxt[:2],
            "history": list(reversed(st["history"]))[:12], "tension_chance_pct": int(POLITICAL_TENSION_CHANCE * 100)}


def _projected(session: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    alive = _season_alive_playoff_teams(session) if meta["kind"] == "worlds" else None
    groups, rosters, strengths = build_rosters(session, meta, alive)
    return {"groups": groups, "rosters": rosters, "strengths": strengths}


def tournament_payload(session: Any, tid: str) -> Dict[str, Any]:
    catch_up(session)
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    metas = {m["id"]: m for k in (0, 1) for m in tournaments_for_season(sy + k)}
    st = _state(session)
    b = st["tournaments"].get(tid)
    if b is None and tid not in metas:
        hist = next((h for h in st["history"] if h.get("id") == tid), None)
        if not hist:
            return {"ok": False, "reason": "Tournament not found"}
    meta = metas.get(tid) or b
    today = _today(session)
    if not b:
        proj = _projected(session, meta)
        return {"ok": True, "id": tid, "kind": meta["kind"], "name": meta["name"], "label": KIND_LABEL[meta["kind"]], "start": meta["start"],
                "end": meta["end"], "host_nation": _nation(meta.get("host", "")), "host_label": meta.get("host_label"), "status": "upcoming",
                "projected": True, "groups": proj["groups"],
                "teams": [{**_nation(c), "strength": proj["strengths"][c], "roster": proj["rosters"][c]} for c in proj["strengths"]],
                "games": [], "standings": {}, "leaders": [], "placements": []}
    complete = bool(b.get("completed"))
    reveal_all = complete or b["kind"] == "worlds"
    visible = [g for g in b["games"] if reveal_all or (today and g["date"] <= today)]
    upcoming = [] if reveal_all else [{k: g[k] for k in ("date", "stage", "label", "home", "away", "id")} for g in b["games"] if not (today and g["date"] <= today)]
    names = {r["id"]: r for rs in b["rosters"].values() for r in rs}
    stats = _player_stats(visible, b["rosters"])
    leaders = sorted((s for s in stats.values() if names.get(s["id"], {}).get("pos") != "G" and s["pts"] > 0),
                     key=lambda s: (-s["pts"], -s["g"]))[:15]
    standings = {}
    for gname, codes in b["groups"].items():
        stage_key = "round_robin" if gname == "RR" else f"group_{gname}"
        standings[gname] = _table(codes, [g for g in visible if g["stage"] == stage_key])

    def game_view(g: Dict[str, Any]) -> Dict[str, Any]:
        return {**{k: g[k] for k in ("id", "date", "stage", "label", "home", "away", "home_score", "away_score", "decided", "winner", "shots")},
                "goals": [{"team": x["team"], "scorer": names.get(x["scorer"], {}).get("name", "?"),
                           "assists": [names.get(a, {}).get("name", "?") for a in x["assists"]]} for x in g["goals"]],
                "goalies": {k: names.get(v, {}).get("name", "") for k, v in g["goalies"].items()}}

    return {
        "ok": True, "id": b["id"], "kind": b["kind"], "name": b.get("name"), "label": KIND_LABEL[b["kind"]], "start": b["start"], "end": b["end"],
        "host_nation": _nation(b.get("host", "")), "host_label": b.get("host_label"),
        "status": "complete" if complete else "live", "projected": False, "today": today, "groups": b["groups"],
        "teams": [{**_nation(c), "strength": b["strengths"].get(c), "roster": [{**r, **{k: v for k, v in stats.get(r["id"], {}).items() if k in ("gp", "g", "a", "pts", "w", "sv_pct")}} for r in b["rosters"][c]]}
                  for c in b["rosters"]],
        "games": [game_view(g) for g in visible], "upcoming": upcoming, "standings": standings,
        "leaders": [{**s, "name": names.get(s["id"], {}).get("name"), "club": names.get(s["id"], {}).get("club"),
                     "is_user": names.get(s["id"], {}).get("is_user", False)} for s in leaders],
        "placements": [_nation(c) for c in b.get("placements") or []] if complete else [],
        "mvp": b.get("mvp") if complete else None,
        "tension": b.get("tension") if complete else None,
        "user_players": [{**r, "nation": c} for c, rs in b["rosters"].items() for r in rs if r.get("is_user")],
    }
