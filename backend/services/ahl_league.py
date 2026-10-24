"""AHL league sim: affiliate lineups, game-by-game results, standings and player stat ledger.

Every NHL organization's AHL roster plays a real schedule (Oct 10 – Apr 19, ~72 GP).
Ice time follows each club's AHL lines (the user's saved AHL lines, CPU auto lines),
so a top-line prospect gets top-line minutes and production, and a scratch gets nothing.
Season lines are installed on each prospect through prospect_league_scoring's
``apply_external_season_line`` so boards, dossiers and development read the same numbers.
"""
from __future__ import annotations

import datetime as _dt
import math
import random
import zlib
from typing import Any, Dict, List, Optional, Tuple

from services.lineup_integrity import empty_lines, player_key, player_name, player_ovr, position_bucket
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

SEASON_START = (10, 10)
SEASON_END = (4, 19)
TARGET_GP = 72
F_TOI = [18.5, 15.5, 12.5, 8.5]  # minutes per forward by line
D_TOI = [23.0, 19.5, 15.5]
LEDGER_VERSION = 1
AHL_STRENGTH_SLOPE = 30.0  # OVR points of roster edge per e-fold of scoring rate


# ----------------------------------------------------------------- helpers
def _sy(session: Any) -> int:
    return int(getattr(session, "season_calendar_year", 2026) or 2026)


def _league(session: Any) -> Any:
    return getattr(getattr(session, "sim", None), "league", None)


def _teams(session: Any) -> List[Any]:
    return list(getattr(_league(session), "teams", None) or [])


def _tid(team: Any) -> str:
    return str(getattr(team, "team_id", getattr(team, "id", "")) or "")


def _affiliate_name(team: Any) -> str:
    try:
        from services.franchise_sim import _ahl_affiliate_display_name

        name = _ahl_affiliate_display_name(team)
        if name:
            return str(name)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return f"{getattr(team, 'city', '')} AHL".strip()


def _ahl_players(team: Any) -> List[Any]:
    return [p for p in (getattr(team, "ahl_roster", None) or []) if not getattr(p, "retired", False)]


def _in_season(d: _dt.date) -> bool:
    m, day = d.month, d.day
    if m >= 10:
        return (m, day) >= SEASON_START
    if m <= 4:
        return (m, day) <= SEASON_END
    return False


def _season_days(season_year: int) -> int:
    a = _dt.date(season_year, *SEASON_START)
    b = _dt.date(season_year + 1, *SEASON_END)
    return (b - a).days + 1


def _rng(*parts: Any) -> random.Random:
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF)


def _skill_rating(player: Any, *names: str, default: Optional[float] = None) -> float:
    for name in names:
        raw = getattr(player, name, None)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    if default is not None:
        return float(default)
    return float(player_ovr(player))


def _scoring_weight(player: Any, toi_min: float) -> float:
    fin = _skill_rating(player, "shooting", "wrist_shot", "finishing", default=player_ovr(player))
    mult = 0.45 if position_bucket(player) == "D" else 1.0
    return float(toi_min) * (fin / 70.0) ** 3 * mult


def _assist_weight(player: Any, toi_min: float) -> float:
    pm = _skill_rating(player, "playmaking", "passing", "vision", "hands", default=player_ovr(player) * 0.95)
    mult = 0.72 if position_bucket(player) == "D" else 1.0
    return float(toi_min) * (pm / 70.0) ** 3 * mult


def _finish_factor(player: Any) -> float:
    fin = _skill_rating(player, "shooting", "wrist_shot", "finishing", default=player_ovr(player))
    return max(0.85, min(1.15, 0.88 + (fin - 72.0) / 80.0))


def _discipline_weight(player: Any, toi_min: float) -> float:
    disc = _skill_rating(player, "discipline", "composure", default=68.0)
    phys = _skill_rating(player, "physicality", "checking", "aggression", default=player_ovr(player))
    return float(toi_min) * max(0.35, (phys / 75.0) ** 2) * max(0.5, 1.15 - disc / 95.0)


def _poisson(lam: float, rng: random.Random) -> int:
    lam = max(0.05, lam)
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


# ----------------------------------------------------------------- state
def ensure_ahl_state(session: Any) -> Dict[str, Any]:
    st = getattr(session, "ahl_league", None)
    sy = _sy(session)
    if not isinstance(st, dict) or st.get("version") != LEDGER_VERSION:
        st = {"version": LEDGER_VERSION, "season": sy, "teams": {}, "players": {}, "last_iso": "", "history": {}}
        session.ahl_league = st
    if int(st.get("season") or 0) != sy:
        hist = st.setdefault("history", {})
        if st.get("teams") or st.get("players"):
            hist[str(st.get("season"))] = {"teams": st.get("teams") or {}, "players": st.get("players") or {}}
            for k in sorted(hist.keys())[:-5]:
                hist.pop(k, None)
        st.update({"season": sy, "teams": {}, "players": {}, "last_iso": ""})
    if not isinstance(getattr(session, "ahl_lines", None), dict):
        session.ahl_lines = {}
    return st


def _team_row(st: Dict[str, Any], team: Any) -> Dict[str, Any]:
    tid = _tid(team)
    row = st["teams"].get(tid)
    if not isinstance(row, dict):
        row = {"team_id": tid, "name": _affiliate_name(team), "parent_abbr": str(getattr(team, "abbreviation", getattr(team, "abbr", "")) or ""),
               "gp": 0, "w": 0, "l": 0, "otl": 0, "pts": 0, "gf": 0, "ga": 0, "shots": 0, "shots_against": 0, "streak": "", "last10": []}
        st["teams"][tid] = row
    return row


def _player_row(st: Dict[str, Any], p: Any, team: Any) -> Dict[str, Any]:
    pid = player_key(p)
    row = st["players"].get(pid)
    if not isinstance(row, dict):
        row = {"player_id": pid, "gp": 0, "goals": 0, "assists": 0, "points": 0, "plus_minus": 0, "pim": 0,
               "shots": 0, "toi_sec": 0, "wins": 0, "losses": 0, "ot_losses": 0, "shots_against": 0,
               "goals_against": 0, "shutouts": 0, "recent": []}
        st["players"][pid] = row
    row["name"] = player_name(p)
    row["pos"] = position_bucket(p)
    row["team_id"] = _tid(team)
    row["team"] = _affiliate_name(team)
    row["parent_abbr"] = str(getattr(team, "abbreviation", getattr(team, "abbr", "")) or "")
    try:
        row["age"] = int(getattr(getattr(p, "identity", None), "age", 0) or 0)
    except Exception:
        row["age"] = 0
    row["ovr"] = round(player_ovr(p))
    try:
        pot = getattr(p, "potential", None)
        pot = pot() if callable(pot) else pot
        pot = float(pot or 0)
        row["pot"] = round(pot * 99 if pot <= 1.5 else pot)
    except Exception:
        row["pot"] = None
    return row


# ----------------------------------------------------------------- lines
def auto_ahl_lines(team: Any) -> Dict[str, Any]:
    lines = empty_lines()
    pool = sorted(_ahl_players(team), key=player_ovr, reverse=True)
    fw = [p for p in pool if position_bucket(p) == "F"]
    dm = [p for p in pool if position_bucket(p) == "D"]
    gk = [p for p in pool if position_bucket(p) == "G"]
    for i, unit in enumerate(lines["forwards"]):
        for j, slot in enumerate(unit["slots"].keys()):
            k = i * 3 + j
            unit["slots"][slot] = player_key(fw[k]) if k < len(fw) else ""
    for i, unit in enumerate(lines["defense"]):
        for j, slot in enumerate(unit["slots"].keys()):
            k = i * 2 + j
            unit["slots"][slot] = player_key(dm[k]) if k < len(dm) else ""
    gslots = list(lines["goalies"][0]["slots"].keys())
    for j, slot in enumerate(gslots):
        lines["goalies"][0]["slots"][slot] = player_key(gk[j]) if j < len(gk) else ""
    return lines


def _clean_lines(team: Any, lines: Any) -> Dict[str, Any]:
    """Drop players no longer on the AHL roster; fill holes from healthy extras."""
    roster = {player_key(p): p for p in _ahl_players(team)}
    base = empty_lines()
    if not isinstance(lines, dict):
        return auto_ahl_lines(team)
    used = set()
    for group in ("forwards", "defense", "goalies"):
        src = {u.get("id"): u for u in (lines.get(group) or []) if isinstance(u, dict)}
        for unit in base[group]:
            got = (src.get(unit["id"]) or {}).get("slots") or {}
            for slot in unit["slots"]:
                pid = str(got.get(slot) or "")
                if pid in roster and pid not in used:
                    unit["slots"][slot] = pid
                    used.add(pid)
    want = {"forwards": "F", "defense": "D", "goalies": "G"}
    for group, bucket in want.items():
        extras = sorted((p for k, p in roster.items() if k not in used and position_bucket(p) == bucket), key=player_ovr, reverse=True)
        for unit in base[group]:
            for slot, pid in unit["slots"].items():
                if not pid and extras:
                    p = extras.pop(0)
                    unit["slots"][slot] = player_key(p)
                    used.add(player_key(p))
    return base


def team_ahl_lines(session: Any, team: Any) -> Dict[str, Any]:
    ensure_ahl_state(session)
    tid = _tid(team)
    if tid == str(getattr(session, "user_team_id", "")):
        saved = session.ahl_lines.get(tid)
        if saved:
            cleaned = _clean_lines(team, saved)
            session.ahl_lines[tid] = cleaned
            return cleaned
    return auto_ahl_lines(team)


def save_user_ahl_lines(session: Any, lines: Dict[str, Any]) -> Dict[str, Any]:
    ensure_ahl_state(session)
    team = session.team_by_id.get(str(session.user_team_id))
    if team is None:
        raise ValueError("No user team")
    roster = {player_key(p) for p in _ahl_players(team)}
    seen = set()
    for group in ("forwards", "defense", "goalies"):
        for unit in (lines or {}).get(group) or []:
            for slot, pid in ((unit or {}).get("slots") or {}).items():
                pid = str(pid or "")
                if not pid:
                    continue
                if pid not in roster:
                    raise ValueError("A player in these lines is not on your AHL roster.")
                if pid in seen:
                    raise ValueError("A player appears in two AHL slots.")
                seen.add(pid)
    session.ahl_lines[str(session.user_team_id)] = _clean_lines(team, lines)
    return build_ahl_lines_payload(session)


def _deployment(team: Any, lines: Dict[str, Any]) -> Tuple[List[Tuple[Any, float, int]], List[Any]]:
    roster = {player_key(p): p for p in _ahl_players(team)}
    skaters: List[Tuple[Any, float, int]] = []
    for i, unit in enumerate(lines.get("forwards") or []):
        for pid in (unit.get("slots") or {}).values():
            if pid in roster:
                skaters.append((roster[pid], F_TOI[min(i, 3)], i))
    for i, unit in enumerate(lines.get("defense") or []):
        for pid in (unit.get("slots") or {}).values():
            if pid in roster:
                skaters.append((roster[pid], D_TOI[min(i, 2)], i))
    goalies = [roster[pid] for pid in ((lines.get("goalies") or [{}])[0].get("slots") or {}).values() if pid in roster]
    return skaters, goalies


# ----------------------------------------------------------------- game sim
def _strength(skaters: List[Tuple[Any, float, int]]) -> float:
    tot = sum(t for _, t, _ in skaters) or 1.0
    base = sum(player_ovr(p) * t for p, t, _ in skaters) / tot if skaters else 55.0
    # A short bench (call-ups) tires the dressed players: small penalty per missing skater.
    return base - max(0, 18 - len(skaters)) * 0.35


def _pick(rng: random.Random, items: List[Tuple[Any, float]]) -> Any:
    total = sum(w for _, w in items)
    if total <= 0:
        return None
    r = rng.random() * total
    for it, w in items:
        r -= w
        if r <= 0:
            return it
    return items[-1][0]


def _on_ice_keys(rng: random.Random, units: Dict[str, Tuple[str, int]], involved: List[str],
                 f_unit: int, d_unit: int) -> List[str]:
    """Three forwards and two D on the ice for a goal: the players in on it, then
    their line / pair (each stays on about 4 times in 5), then anyone else."""
    out = [k for k in involved if k in units]
    for bucket, want, base in (("F", 3, f_unit), ("D", 2, d_unit)):
        have = [k for k in out if units[k][0] == bucket]
        if have:
            base = units[have[0]][1]
        need = want - len(have)
        mates = [k for k, (b, u) in units.items() if b == bucket and u == base and k not in out]
        rng.shuffle(mates)
        for k in mates:
            if need > 0 and rng.random() < 0.8:
                out.append(k)
                need -= 1
        others = [k for k, (b, _u) in units.items() if b == bucket and k not in out]
        rng.shuffle(others)
        out.extend(others[:max(0, need)])
    return out


def _play_game(session: Any, st: Dict[str, Any], home: Any, away: Any, iso: str) -> None:
    save_id = str(getattr(session, "session_id", "") or "")
    rng = _rng("ahlgame", save_id, st["season"], iso, _tid(home), _tid(away))
    sides = []
    for team in (home, away):
        lines = team_ahl_lines(session, team)
        skaters, goalies = _deployment(team, lines)
        if not skaters:
            return
        starter = goalies[0] if goalies else None
        if len(goalies) > 1:
            ranked = sorted(goalies, key=player_ovr, reverse=True)
            starter = ranked[0] if rng.random() < 0.85 else ranked[1]
        sides.append({"team": team, "skaters": skaters, "goalie": starter, "strength": _strength(skaters)})
    h, a = sides
    for me, opp, home_adv in ((h, a, 1.04), (a, h, 1.0)):
        g_ovr = player_ovr(opp["goalie"]) if opp["goalie"] is not None else 55.0
        # AHL parity is high (best clubs ~.650, worst ~.400). The old /11 slope turned a
        # 4-OVR roster edge into near-undefeated seasons; nightly form noise matters too.
        lam = 3.0 * math.exp((me["strength"] - opp["strength"]) / AHL_STRENGTH_SLOPE) * math.exp(-(g_ovr - 74.0) / 90.0) * home_adv
        lam *= math.exp(rng.gauss(0.0, 0.16))
        me["xg"] = min(5.5, max(1.5, lam))
        me["goals"] = _poisson(me["xg"], rng)
        me["penalties"] = max(0, _poisson(3.0, rng))
        n_attempts = max(me["goals"] + 14, int(rng.gauss(57 + (me["strength"] - opp["strength"]) * 0.5, 5)))
        me["shots"] = 0
        me["_attempts"] = n_attempts
    ot = h["goals"] == a["goals"]
    ot_shootout = False
    if ot:
        if rng.random() < 0.32:
            ot_shootout = True
            if rng.random() < 0.5 + (h["strength"] - a["strength"]) / 60.0:
                h["goals"] += 1
            else:
                a["goals"] += 1
        elif rng.random() < 0.5 + (h["strength"] - a["strength"]) / 60.0:
            h["goals"] += 1
        else:
            a["goals"] += 1
    winner, loser = (h, a) if h["goals"] > a["goals"] else (a, h)
    for side in (h, a):
        if ot_shootout and side is winner:
            side["_stat_goals"] = max(0, int(side["goals"]) - 1)
        else:
            side["_stat_goals"] = int(side["goals"])

    def _prepare_shot_chart(side: Dict[str, Any]) -> None:
        shoot_w = [(_scoring_weight(p, toi), p, toi) for p, toi, _ in side["skaters"]]
        sog_list: List[Any] = []
        for _ in range(int(side.get("_attempts") or 0)):
            shooter = _pick(rng, [(p, w) for w, p, _ in shoot_w])
            if shooter is None:
                continue
            if rng.random() < 0.54:
                sog_list.append(shooter)
        goals_target = int(side.get("_stat_goals", side["goals"]))
        while len(sog_list) < goals_target:
            extra = _pick(rng, [(p, w) for w, p, _ in shoot_w])
            if extra is not None:
                sog_list.append(extra)
        goal_idx: set = set()
        if sog_list and goals_target > 0:
            pool = list(range(len(sog_list)))
            for _ in range(min(goals_target, len(pool))):
                weights = []
                for i in pool:
                    sh = sog_list[i]
                    weights.append(max(0.02, _finish_factor(sh)))
                pick = rng.choices(pool, weights=weights, k=1)[0]
                goal_idx.add(pick)
                pool.remove(pick)
        side["shots"] = len(sog_list)
        side["_sog_list"] = sog_list
        side["_goal_idx"] = goal_idx
        side["_shoot_w"] = shoot_w

    _prepare_shot_chart(h)
    _prepare_shot_chart(a)

    for side, opp in ((h, a), (a, h)):
        trow = _team_row(st, side["team"])
        trow["gp"] += 1
        trow["gf"] += side["goals"]
        trow["ga"] += opp["goals"]
        trow["shots"] += side["shots"]
        trow["shots_against"] += opp["shots"]
        res = "W" if side is winner else ("OTL" if ot else "L")
        trow["w" if res == "W" else ("otl" if res == "OTL" else "l")] += 1
        trow["pts"] = trow["w"] * 2 + trow["otl"]
        trow["last10"] = (trow.get("last10") or [])[-9:] + [res]
        prev = trow.get("streak") or ""
        code = "W" if res == "W" else "L"
        trow["streak"] = f"{code}{int(prev[1:] or 0) + 1}" if prev[:1] == code else f"{code}1"

        rows = {player_key(p): _player_row(st, p, side["team"]) for p, _, _ in side["skaters"]}
        game_pts = {k: 0 for k in rows}
        raw_toi = [float(toi) for _, toi, _ in side["skaters"]]
        toi_sum = sum(raw_toi) or 1.0
        team_toi_min = 300.0
        for p, toi, _ in side["skaters"]:
            r = rows[player_key(p)]
            r["gp"] += 1
            r["toi_sec"] += int(team_toi_min * 60.0 * (float(toi) / toi_sum))
        assist_w = [(_assist_weight(p, toi), p) for p, toi, _ in side["skaters"]]
        sog_list = list(side.get("_sog_list") or [])
        goal_idx = set(side.get("_goal_idx") or set())
        opp_pp_left = max(0, int(opp.get("penalties", 0) or 0))
        on_ice_unit = {player_key(p): (position_bucket(p), i) for p, _, i in side["skaters"]}
        for i, scorer in enumerate(sog_list):
            sk = player_key(scorer)
            rows[sk]["shots"] += 1
            if i not in goal_idx:
                continue
            is_pp = opp_pp_left > 0 and rng.random() < 0.20
            if is_pp:
                opp_pp_left -= 1
            rows[sk]["goals"] += 1
            game_pts[sk] += 1
            if is_pp:
                rows[sk]["pp_goals"] = int(rows[sk].get("pp_goals", 0) or 0) + 1
            involved = [sk]
            if rng.random() >= 0.05:
                mates = [(p, w) for w, p in assist_w if player_key(p) != sk]
                n_ast = 2 if rng.random() < 0.73 else 1
                for _ in range(n_ast):
                    ast = _pick(rng, mates)
                    if ast is None:
                        break
                    ak = player_key(ast)
                    rows[ak]["assists"] += 1
                    game_pts[ak] += 1
                    involved.append(ak)
                    mates = [(p, w) for p, w in mates if player_key(p) != ak]
            if not is_pp:
                bucket, unit = on_ice_unit.get(sk, ("F", 0))
                for k in _on_ice_keys(rng, on_ice_unit, involved, unit, min(unit, 2)):
                    rows[k]["plus_minus"] += 1
                opp_unit = {player_key(p): (position_bucket(p), i) for p, _, i in opp["skaters"]}
                opp_rows = {player_key(p): (p, opp["team"]) for p, _, _ in opp["skaters"]}
                pk_unit = rng.randint(0, 2)
                for k in _on_ice_keys(rng, opp_unit, [], pk_unit, min(pk_unit, 2)):
                    op, op_team = opp_rows[k]
                    _player_row(st, op, op_team)["plus_minus"] -= 1
        pim_w = [(p, _discipline_weight(p, toi)) for p, toi, _ in side["skaters"]]
        for _ in range(int(side.get("penalties", 0) or 0)):
            who = _pick(rng, pim_w)
            if who is not None:
                rows[player_key(who)]["pim"] += 2
        for k, r in rows.items():
            r["points"] = r["goals"] + r["assists"]
            r["recent"] = (r.get("recent") or [])[-7:] + [game_pts[k]]
        g = side["goalie"]
        if g is not None:
            gr = _player_row(st, g, side["team"])
            gr["gp"] += 1
            gr["shots_against"] += opp["shots"]
            gr["goals_against"] += int(opp.get("_stat_goals", opp["goals"]))
            gr["toi_sec"] += 3600 + (300 if ot else 0)
            if side is winner:
                gr["wins"] += 1
                if opp["goals"] == 0:
                    gr["shutouts"] += 1
            elif ot:
                gr["ot_losses"] += 1
            else:
                gr["losses"] += 1
            gr["recent"] = (gr.get("recent") or [])[-7:] + [0]
            setattr(g, "_ahl_ledger_owned", st["season"])
        for p, _, _ in side["skaters"]:
            setattr(p, "_ahl_ledger_owned", st["season"])


def _install_lines(session: Any, st: Dict[str, Any], iso: str, touched: set) -> None:
    try:
        from app.sim_engine.generation.prospect_league_scoring import apply_external_season_line
    except Exception:
        return
    for team in _teams(session):
        for p in _ahl_players(team):
            pid = player_key(p)
            if pid not in touched:
                continue
            row = st["players"].get(pid)
            if not row:
                continue
            line = {k: row[k] for k in ("gp", "goals", "assists", "points", "plus_minus", "pim", "shots", "toi_sec",
                                         "wins", "losses", "ot_losses", "shots_against", "goals_against", "shutouts")}
            try:
                apply_external_season_line(p, "AHL", line, iso, season_year=st["season"], recent_points=row.get("recent"))
            except Exception:
                continue


def sync_ahl_to_date(session: Any, iso: Optional[str] = None, max_days: int = 400) -> int:
    """Play every AHL game day from the last synced date through ``iso``. Returns games played."""
    st = ensure_ahl_state(session)
    if iso is None:
        try:
            from services.franchise_sim import _scouting_calendar_iso

            iso = _scouting_calendar_iso(session)
        except Exception:
            iso = ""
    try:
        target = _dt.date.fromisoformat(str(iso)[:10])
    except Exception:
        return 0
    sy = st["season"]
    season_start = _dt.date(sy, *SEASON_START)
    season_end = _dt.date(sy + 1, *SEASON_END)
    try:
        cur = _dt.date.fromisoformat(st["last_iso"]) + _dt.timedelta(days=1) if st.get("last_iso") else season_start
    except Exception:
        cur = season_start
    cur = max(cur, season_start)
    end = min(target, season_end)
    if cur > end:
        return 0
    teams = [t for t in _teams(session) if _ahl_players(t)]
    if len(teams) < 2:
        return 0
    p_play = TARGET_GP / float(_season_days(sy))
    games = 0
    touched: set = set()
    days = 0
    while cur <= end and days < max_days:
        d_iso = cur.isoformat()
        rng = _rng("ahlday", sy, d_iso)
        order = sorted(teams, key=_tid)
        rng.shuffle(order)
        playing = [t for t in order if rng.random() < p_play]
        for i in range(0, len(playing) - 1, 2):
            home, away = playing[i], playing[i + 1]
            _play_game(session, st, home, away, d_iso)
            for t in (home, away):
                for p in _ahl_players(t):
                    touched.add(player_key(p))
            games += 1
        st["last_iso"] = d_iso
        cur += _dt.timedelta(days=1)
        days += 1
    if touched:
        _install_lines(session, st, end.isoformat(), touched)
    return games


# ----------------------------------------------------------------- accessors / payloads
def get_player_ahl_season_line(session: Any, player_id: str, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Season AHL line for a player (gp, goals, assists, points, plus_minus, pim, shots, toi_sec,
    toi_per_gp_min, ppg; goalies add wins/losses/ot_losses/save_pct/gaa/shutouts), or None.
    Used by development to reward AHL production and usage."""
    st = ensure_ahl_state(session)
    if season is None or int(season) == int(st["season"]):
        row = st["players"].get(str(player_id))
    else:
        row = ((st.get("history") or {}).get(str(season)) or {}).get("players", {}).get(str(player_id))
    if not row:
        return None
    out = dict(row)
    gp = max(1, int(out.get("gp") or 0))
    out["toi_per_gp_min"] = round((out.get("toi_sec") or 0) / 60.0 / gp, 1)
    out["ppg"] = round((out.get("points") or 0) / gp, 2)
    sa = int(out.get("shots_against") or 0)
    out["save_pct"] = round(1 - (out.get("goals_against") or 0) / sa, 3) if sa else None
    out["gaa"] = round((out.get("goals_against") or 0) * 3600 / max(1, out.get("toi_sec") or 1), 2) if out.get("pos") == "G" else None
    return out


def build_ahl_ledger_payload(session: Any, season: Optional[int] = None) -> Dict[str, Any]:
    st = ensure_ahl_state(session)
    sy = int(season or st["season"])
    if sy == int(st["season"]):
        teams, players = st["teams"], st["players"]
    else:
        block = (st.get("history") or {}).get(str(sy)) or {}
        teams, players = block.get("teams") or {}, block.get("players") or {}
    uid = str(getattr(session, "user_team_id", ""))
    trows = []
    for r in teams.values():
        gp = max(1, r["gp"])
        trows.append({**r, "pts_pct": round(r["pts"] / (2.0 * gp), 3), "diff": r["gf"] - r["ga"],
                      "sh_pct": round(r["gf"] / max(1, r["shots"]) * 100, 1), "sv_pct": round(1 - r["ga"] / max(1, r["shots_against"]), 3),
                      "user": r["team_id"] == uid, "last10": "-".join(str(sum(1 for x in r.get("last10") or [] if x == k)) for k in ("W", "L", "OTL"))})
    trows.sort(key=lambda r: (r["pts"], r["w"], r["diff"]), reverse=True)
    skaters, goalies = [], []
    for r in players.values():
        gp = max(1, r["gp"])
        base = {k: r.get(k) for k in ("player_id", "name", "pos", "team", "team_id", "parent_abbr", "age", "ovr", "pot", "gp")}
        base["user_org"] = r.get("team_id") == uid
        if r.get("pos") == "G":
            sa = r.get("shots_against") or 0
            goalies.append({**base, "w": r["wins"], "l": r["losses"], "otl": r["ot_losses"], "so": r["shutouts"],
                            "sv_pct": round(1 - r["goals_against"] / sa, 3) if sa else None,
                            "gaa": round(r["goals_against"] * 3600 / max(1, r["toi_sec"]), 2)})
        else:
            skaters.append({**base, "g": r["goals"], "a": r["assists"], "p": r["points"], "pm": r["plus_minus"], "pim": r["pim"],
                            "sog": r["shots"], "toi": round(r["toi_sec"] / 60.0 / gp, 1), "ppg": round(r["points"] / gp, 2)})
    skaters.sort(key=lambda r: (r["p"], r["g"]), reverse=True)
    goalies.sort(key=lambda r: (r["w"], r["sv_pct"] or 0), reverse=True)
    return {"season": sy, "seasons": sorted([int(k) for k in (st.get("history") or {}).keys()] + [int(st["season"])], reverse=True),
            "last_game_date": st.get("last_iso") or None, "teams": trows, "skaters": skaters, "goalies": goalies}


def build_ahl_lines_payload(session: Any) -> Dict[str, Any]:
    team = session.team_by_id.get(str(session.user_team_id))
    if team is None:
        return {"ok": False, "reason": "No user team"}
    lines = team_ahl_lines(session, team)
    st = ensure_ahl_state(session)
    roster = []
    for p in _ahl_players(team):
        r = st["players"].get(player_key(p)) or {}
        ident = getattr(p, "identity", None)
        pot = getattr(p, "potential", None)
        try:
            pot = pot() if callable(pot) else pot
            pot = round(float(pot) * 99 if float(pot) <= 1.5 else float(pot))
        except Exception:
            pot = None
        gp = max(1, int(r.get("gp") or 0))
        roster.append({
            "id": player_key(p), "name": player_name(p), "pos": position_bucket(p),
            "position": str(getattr(getattr(ident, "position", None), "value", getattr(ident, "position", "")) or "").split(".")[-1],
            "age": int(getattr(ident, "age", 0) or 0) if ident else None, "ovr": round(player_ovr(p)), "pot": pot,
            "nhl_id": getattr(p, "nhl_id", None) or getattr(ident, "nhl_id", None),
            "gp": r.get("gp", 0), "g": r.get("goals", 0), "a": r.get("assists", 0), "p": r.get("points", 0),
            "toi": round((r.get("toi_sec") or 0) / 60.0 / gp, 1) if r.get("gp") else 0,
        })
        try:
            from app.sim_engine.generation.player_headshots import merge_headshot_into_row

            roster[-1] = merge_headshot_into_row(roster[-1], p) or roster[-1]
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    roster.sort(key=lambda x: x["ovr"], reverse=True)
    return {"ok": True, "team_name": _affiliate_name(team), "lines": lines, "roster": roster,
            "custom": bool(session.ahl_lines.get(str(session.user_team_id))), "toi_by_line": {"forwards": F_TOI, "defense": D_TOI}}


def reset_user_ahl_lines(session: Any) -> Dict[str, Any]:
    ensure_ahl_state(session)
    session.ahl_lines.pop(str(session.user_team_id), None)
    return build_ahl_lines_payload(session)
