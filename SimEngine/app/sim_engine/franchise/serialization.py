"""API serialization for rosters, stats, and standings."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

def _stable_hash(value):
    """Process-stable replacement for built-in hash() (salted per process via PYTHONHASHSEED)."""
    import hashlib as _hl

    return int.from_bytes(_hl.sha256(str(value).encode("utf-8")).digest()[:8], "big", signed=True)


def _display_team(t: Any) -> str:
    city = str(getattr(t, "city", "") or "").strip()
    name = str(getattr(t, "name", "") or "").strip()
    if city and name:
        return f"{city} {name}"
    return rs._team_name(t)
def _franchise_team_abbrev(tm: Any) -> str:
    if tm is None:
        return "?"
    for attr in ("abbr", "code", "abbreviation", "short_name"):
        v = getattr(tm, attr, None)
        if v is not None and str(v).strip():
            s = "".join(c for c in str(v).upper() if c.isalnum())
            if 2 <= len(s) <= 5:
                return s[:3]
    disp_raw = _display_team(tm).strip()
    disp = disp_raw.lower()
    if disp in _NHL_DISPLAY_LOWER_TO_ABBR:
        return _NHL_DISPLAY_LOWER_TO_ABBR[disp]
    for full_name, abbr in _NHL_DISPLAY_LOWER_TO_ABBR.items():
        if full_name in disp or disp in full_name:
            return abbr
    city = str(getattr(tm, "city", "") or "").strip()
    name = str(getattr(tm, "name", "") or "").strip()
    if city and name:
        combo = f"{city} {name}".strip().lower()
        if combo in _NHL_DISPLAY_LOWER_TO_ABBR:
            return _NHL_DISPLAY_LOWER_TO_ABBR[combo]
    tid_v = getattr(tm, "team_id", None)
    if tid_v is None:
        tid_v = getattr(tm, "id", None)
    if tid_v is None:
        tid_v = rs._team_id(tm)
    tid = str(tid_v) if tid_v is not None else ""
    u = "".join(c for c in tid.upper() if c.isalnum())
    if len(u) >= 3:
        return u[:3]
    nm = name.upper() if name else ""
    u2 = "".join(c for c in nm if c.isalnum())
    if len(u2) >= 3:
        return u2[:3]
    return (tid[:3] if tid else "?").upper()
def _name_str(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or "?")
def _pos_str(p: Any) -> str:
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident else None
    return str(getattr(pos, "value", pos) or "?")
def _rating_label(key: str) -> str:
    for pfx in ("off_", "pm_", "def_", "phy_", "skg_", "iqm_", "pc_", "dev_", "per_", "st_", "g_"):
        if key.startswith(pfx):
            return key[len(pfx) :].replace("_", " ").title()
    return key.replace("_", " ").title()
def _rating_groups_for_player(p: Any) -> List[Dict[str, Any]]:
    pos = (_pos_str(p) or "").strip().upper()
    is_g = pos == "G"
    r = getattr(p, "ratings", None) or {}

    def rows(keys: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for k in keys:
            out.append({"id": k, "label": _rating_label(k), "v": int(float(r.get(k, 68)))})
        return out

    groups: List[Dict[str, Any]] = [
        {"title": "Offense", "rows": rows(OFFENSE_ATTRS)},
        {"title": "Playmaking", "rows": rows(PLAYMAKING_ATTRS)},
        {"title": "Defense", "rows": rows(DEFENSE_ATTRS)},
        {"title": "Physical", "rows": rows(PHYSICAL_ATTRS)},
        {"title": "Skating", "rows": rows(SKATING_ATTRS)},
        {"title": "IQ / Mental", "rows": rows(IQ_ATTRS)},
        {"title": "Skill / Puck Control", "rows": rows(SKILL_ATTRS)},
        {"title": "Development", "rows": rows(DEV_ATTRS)},
        {"title": "Personality", "rows": rows(PERSONALITY_ATTRS)},
        {"title": "Special Traits", "rows": rows(SPECIAL_ATTRS)},
    ]
    if is_g:
        groups.insert(0, {"title": "Goalie", "rows": rows(GOALIE_ATTRS)})
    return groups
def _active_roster(team: Any) -> List[Any]:
    return [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
def _skaters(team: Any) -> List[Any]:
    return [p for p in _active_roster(team) if _pos_str(p).upper() != "G"]
def _goalies(team: Any) -> List[Any]:
    return [p for p in _active_roster(team) if _pos_str(p).upper() == "G"]
def _available_goalies(team: Any) -> List[Any]:
    return [g for g in _goalies(team) if not _is_player_live_injured(g)]
def _goalie_availability_status(team: Any) -> Dict[str, Any]:
    all_goalies = _goalies(team)
    healthy = _available_goalies(team)
    return {
        "total": int(len(all_goalies)),
        "healthy": int(len(healthy)),
        "forced_injured_start": bool(all_goalies and not healthy),
    }
def _ovr_weight(p: Any) -> float:
    try:
        fn = getattr(p, "ovr", None)
        o = float(fn() if callable(fn) else fn)
        if o <= 1.5:
            o *= 99.0
        return max(14.0, o) ** 1.2
    except Exception:
        return 52.0
def _player_role_usage_mult(p: Any) -> float:
    """Approximate TOI/opportunity weighting by lineup role and quality."""
    role_raw = str(
        getattr(p, "line_role", None)
        or getattr(p, "role", None)
        or getattr(p, "depth_role", None)
        or ""
    ).lower()
    if "top" in role_raw or "line1" in role_raw or "first" in role_raw:
        return 2.35
    if "second" in role_raw or "line2" in role_raw:
        return 1.65
    if "third" in role_raw or "line3" in role_raw or "middle" in role_raw:
        return 0.92
    if "fourth" in role_raw or "line4" in role_raw or "depth" in role_raw:
        return 0.58
    return 0.85
def _rating_avg(p: Any, keys: List[str], default: float = 68.0) -> float:
    r = getattr(p, "ratings", None) or {}
    vals = [float(r.get(k, default) or default) for k in keys]
    if not vals:
        return default
    return sum(vals) / len(vals)
def _offense_opportunity_weight(p: Any) -> float:
    """
    Heavily bias offensive event participation to top-end talent and usage.
    """
    ovr = _ovr_weight(p)
    off = _rating_avg(p, OFFENSE_ATTRS)
    pm = _rating_avg(p, PLAYMAKING_ATTRS)
    iq = _rating_avg(p, IQ_ATTRS)
    pos = _pos_str(p).upper()
    pos_mult = 0.98 if pos == "D" else 1.0
    usage_mult = _player_role_usage_mult(p)
    base = (0.40 * off + 0.34 * pm + 0.26 * iq) / 99.0
    return max(0.04, (ovr ** 1.22) * (base ** 1.08) * usage_mult * pos_mult)
def _team_shooting_profile(team: Any) -> float:
    """Return team shooting baseline ~8%..12% with roster skill scaling."""
    sk = _skaters(team)
    if not sk:
        return 0.095
    top = sorted(sk, key=_offense_opportunity_weight, reverse=True)[:10]
    off = sum(_rating_avg(p, OFFENSE_ATTRS) for p in top) / max(1, len(top))
    iq = sum(_rating_avg(p, IQ_ATTRS) for p in top) / max(1, len(top))
    profile = 0.092 + 0.00028 * (off - 68.0) + 0.00012 * (iq - 68.0)
    return max(0.08, min(0.12, profile))


def _empty_league_team_row(tid: str) -> Dict[str, Any]:
    return {
        "team_id": tid,
        "id": tid,
        "gp": 0,
        "w": 0,
        "l": 0,
        "otl": 0,
        "pts": 0,
        "gf": 0,
        "ga": 0,
        "sf": 0,
        "sa": 0,
        "cf": 0,
        "ca": 0,
        "xgf": 0.0,
        "xga": 0.0,
        "ppg": 0,
        "ppo": 0,
        "ppga": 0,
        "opp_ppo": 0,
        "xgf_pct_sum": 0.0,
        "xgf_pct_gp": 0,
    }


def _build_league_teams_payload(
    session: FranchiseSession,
    *,
    team_totals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Full league team analytics rows + lightweight directory for Stats Central."""
    rows_by_tid: Dict[str, Dict[str, Any]] = {}

    for tid_raw, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid_raw)
        row = _empty_league_team_row(tid)
        row["name"] = _display_team(tm)
        abbr = _franchise_team_abbrev(tm)
        row["abbrev"] = abbr
        row["abbr"] = abbr
        rows_by_tid[tid] = row

    st = getattr(session, "standings", None)
    if st is not None:
        for tid_raw, rec in (getattr(st, "records", None) or {}).items():
            tid = str(tid_raw)
            row = rows_by_tid.setdefault(tid, _empty_league_team_row(tid))
            if not row.get("name"):
                row["name"] = str(getattr(rec, "name", tid) or tid)
            row["gp"] = int(getattr(rec, "gp", 0) or 0)
            row["w"] = int(getattr(rec, "wins", 0) or 0)
            row["l"] = int(getattr(rec, "losses", 0) or 0)
            row["otl"] = int(getattr(rec, "otl", 0) or 0)
            row["pts"] = int(getattr(rec, "points", 0) or 0)
            row["wins"] = row["w"]
            row["losses"] = row["l"]
            row["points"] = row["pts"]

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue
        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")
        if not hid or not aid:
            continue
        try:
            hg = int(g.get("home_goals", g.get("home_score", 0)) or 0)
            ag = int(g.get("away_goals", g.get("away_score", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if hg < 0 or ag < 0 or hg == ag:
            continue

        home_shots = int(g.get("home_shots", 0) or 0)
        away_shots = int(g.get("away_shots", 0) or 0)
        home_cf = int(g.get("home_shot_attempts", g.get("home_cf", 0)) or 0)
        away_cf = int(g.get("away_shot_attempts", g.get("away_cf", 0)) or 0)
        home_xgf = float(g.get("home_xgf", g.get("home_xg", 0.0)) or 0.0)
        away_xgf = float(g.get("away_xgf", g.get("away_xg", 0.0)) or 0.0)

        for tid, gf, ga, sf, sa, cf, ca, xgf, xga, ppg, ppo, ppga, opp_ppo, xgf_pct in (
            (
                hid,
                hg,
                ag,
                home_shots,
                away_shots,
                home_cf,
                away_cf,
                home_xgf,
                away_xgf,
                int(g.get("home_pp_goals", 0) or 0),
                int(g.get("home_ppo", 0) or 0),
                int(g.get("home_ppga", g.get("away_pp_goals", 0)) or 0),
                int(g.get("home_opp_ppo", g.get("away_ppo", 0)) or 0),
                float(g.get("home_xgf_pct", 0) or 0),
            ),
            (
                aid,
                ag,
                hg,
                away_shots,
                home_shots,
                away_cf,
                home_cf,
                away_xgf,
                home_xgf,
                int(g.get("away_pp_goals", 0) or 0),
                int(g.get("away_ppo", 0) or 0),
                int(g.get("away_ppga", g.get("home_pp_goals", 0)) or 0),
                int(g.get("away_opp_ppo", g.get("home_ppo", 0)) or 0),
                float(g.get("away_xgf_pct", 0) or 0),
            ),
        ):
            row = rows_by_tid.setdefault(tid, _empty_league_team_row(tid))
            row["gf"] += gf
            row["ga"] += ga
            row["sf"] += sf
            row["sa"] += sa
            row["cf"] += cf
            row["ca"] += ca
            row["xgf"] += xgf
            row["xga"] += xga
            row["ppg"] += ppg
            row["ppo"] += ppo
            row["ppga"] += ppga
            row["opp_ppo"] += opp_ppo
            if xgf_pct > 0:
                row["xgf_pct_sum"] += xgf_pct
                row["xgf_pct_gp"] += 1

    if team_totals:
        for tid, ttot in team_totals.items():
            row = rows_by_tid.setdefault(str(tid), _empty_league_team_row(str(tid)))
            if not row.get("gf"):
                row["gf"] = int(ttot.get("goals", 0) or 0)

    league_teams: List[Dict[str, Any]] = []
    for row in rows_by_tid.values():
        gf = int(row.get("gf", 0) or 0)
        ga = int(row.get("ga", 0) or 0)
        sf = int(row.get("sf", 0) or 0)
        sa = int(row.get("sa", 0) or 0)
        ppg = int(row.get("ppg", 0) or 0)
        ppo = int(row.get("ppo", 0) or 0)
        ppga = int(row.get("ppga", 0) or 0)
        opp_ppo = int(row.get("opp_ppo", 0) or 0)
        xgf = float(row.get("xgf", row.get("expected_goals_for", 0.0)) or 0.0)
        xga = float(row.get("xga", row.get("expected_goals_against", 0.0)) or 0.0)
        xgf_gp = int(row.get("xgf_pct_gp", 0) or 0)
        xgf_sum = float(row.get("xgf_pct_sum", 0.0) or 0.0)

        row["goals_for"] = gf
        row["goals_against"] = ga
        row["goal_diff"] = gf - ga
        row["shots_for"] = sf
        row["shots_against"] = sa

        if ppo > 0:
            row["pp_pct"] = ppg / float(ppo)
            row["power_play_pct"] = row["pp_pct"]
        if opp_ppo > 0:
            row["pk_pct"] = 1.0 - (ppga / float(opp_ppo))
            row["penalty_kill_pct"] = row["pk_pct"]
        cf = int(row.get("cf", 0) or 0)
        ca = int(row.get("ca", 0) or 0)
        if cf + ca > 0:
            row["cf_pct"] = cf / float(cf + ca)
            row["corsi_pct"] = row["cf_pct"]
        if xgf + xga > 0:
            row["xgf_pct"] = xgf / float(xgf + xga)
        elif xgf_gp > 0:
            row["xgf_pct"] = xgf_sum / float(xgf_gp)
        else:
            row["xgf_pct"] = None

        league_teams.append(row)

    league_teams.sort(
        key=lambda r: (
            -int(r.get("pts", 0) or 0),
            -int(r.get("goal_diff", 0) or 0),
            -int(r.get("gf", 0) or 0),
        )
    )
    for idx, row in enumerate(league_teams, start=1):
        row["league_rank"] = idx

    teams_directory = [
        {
            "team_id": str(r.get("team_id") or ""),
            "id": str(r.get("team_id") or ""),
            "name": str(r.get("name") or ""),
            "abbrev": str(r.get("abbrev") or r.get("abbr") or ""),
        }
        for r in league_teams
    ]
    return league_teams, teams_directory


def _build_stats_central_payload(session: FranchiseSession) -> Dict[str, Any]:
    """
    StatsCentral payload.

    Important:
    - uses only the game-derived stat ledger
    - normalizes skaters and goalies separately
    - exposes league leaders, user leaders, goalie leaders, and diagnostics
    - never invents scoring
    """
    all_results = [
        g for g in list(getattr(session, "game_results", None) or [])
        if isinstance(g, dict)
    ]

    games = list(reversed(all_results[-100:]))

    by_day: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for g in all_results:
        try:
            di = int(g.get("day", g.get("calendar_day", -1)))
        except (TypeError, ValueError):
            continue

        if di < 0:
            continue

        by_day[di].append(g)

    day_keys = sorted(by_day.keys(), reverse=True)
    calendar = [
        {
            "day": int(d),
            "games": list(reversed(by_day[d])),
        }
        for d in day_keys
    ]

    raw_rows = list((getattr(session, "player_season_stats", None) or {}).values())
    rows = [_normalize_player_stat_row(dict(r or {})) for r in raw_rows]

    skaters = [r for r in rows if not r.get("is_goalie")]
    goalies = [r for r in rows if r.get("is_goalie")]

    skaters.sort(
        key=lambda r: (
            -int(r.get("pts", 0) or 0),
            -int(r.get("g", 0) or 0),
            -int(r.get("a", 0) or 0),
            -int(r.get("sog", 0) or 0),
            str(r.get("name") or ""),
        )
    )

    leaders = skaters[:45]

    uid = str(getattr(session, "user_team_id", "") or "")

    user_only = [r for r in skaters if str(r.get("team_id") or "") == uid]
    user_only.sort(
        key=lambda r: (
            -int(r.get("pts", 0) or 0),
            -int(r.get("g", 0) or 0),
            -int(r.get("a", 0) or 0),
            str(r.get("name") or ""),
        )
    )

    goalie_leaders = [
        r for r in goalies
        if int(r.get("shots_against", 0) or 0) >= 50 or int(r.get("gp", 0) or 0) >= 3
    ]

    goalie_leaders.sort(
        key=lambda r: (
            -float(r.get("save_pct", 0.0) or 0.0),
            float(r.get("gaa", 99.0) or 99.0),
            -int(r.get("gp", 0) or 0),
            str(r.get("name") or ""),
        )
    )

    team_totals: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "team_id": "",
            "gp_player_rows": 0,
            "goals": 0,
            "assists": 0,
            "points": 0,
            "shots": 0,
            "hits": 0,
            "blocks": 0,
            "pim": 0,
        }
    )

    for r in skaters:
        tid = str(r.get("team_id") or "")
        if not tid:
            continue

        trow = team_totals[tid]
        trow["team_id"] = tid
        trow["gp_player_rows"] += int(r.get("gp", 0) or 0)
        trow["goals"] += int(r.get("g", 0) or 0)
        trow["assists"] += int(r.get("a", 0) or 0)
        trow["points"] += int(r.get("pts", 0) or 0)
        trow["shots"] += int(r.get("sog", 0) or 0)
        trow["hits"] += int(r.get("hit", 0) or 0)
        trow["blocks"] += int(r.get("blk", 0) or 0)
        trow["pim"] += int(r.get("pim", 0) or 0)

    integrity = _stats_integrity_payload(rows, all_results)
    league_teams, teams_directory = _build_league_teams_payload(session, team_totals=team_totals)

    return {
        "games": games,
        "calendar": calendar,
        "players": rows,
        "skaters": skaters,
        "goalies": goalies,
        "leaders": leaders,
        "league_leaders": leaders,
        "user_leaders": user_only[:20],
        "goalie_leaders": goalie_leaders[:20],
        "team_totals": list(team_totals.values()),
        "teams": league_teams,
        "league_teams": league_teams,
        "league_team_stats": league_teams,
        "teams_directory": teams_directory,
        "integrity": integrity,
        "meta": {
            "source": "game_derived_player_season_stats",
            "stat_source": "game_ledger",
            "player_rows": len(rows),
            "skater_rows": len(skaters),
            "goalie_rows": len(goalies),
            "completed_games": len(all_results),
            "leader_limit": 45,
            "game_derived_only": True,
        },
    }
def _stat_int(row: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in row:
            try:
                return int(round(float(row.get(key) or 0)))
            except (TypeError, ValueError):
                continue
    return int(default)
def _stat_float(row: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            try:
                return float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
    return float(default)
def _normalize_stat_position(value: Any) -> str:
    pos = str(value or "").strip().upper()
    if pos in {"G", "GOALIE", "GK"}:
        return "G"
    if pos in {"D", "LD", "RD"}:
        return "D"
    if pos in {"C", "LW", "RW", "F"}:
        return pos
    return pos or "F"
def _is_goalie_stat_row(row: Dict[str, Any]) -> bool:
    pos = _normalize_stat_position(row.get("position") or row.get("pos"))
    if pos == "G":
        return True

    goalie_markers = (
        "shots_against",
        "saves",
        "save_pct",
        "gaa",
        "ga",
        "w",
        "l",
        "otl",
    )

    return any(k in row for k in goalie_markers) and not (_stat_int(row, "g") or _stat_int(row, "a"))
def _normalize_player_stat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean one player season stat row before sending it to frontend.

    Important:
    - does not create fake production
    - only derives totals from existing game-derived stats
    - splits skater and goalie interpretation
    """
    if not isinstance(row, dict):
        row = {}

    player_id = str(row.get("player_id") or row.get("id") or "")
    team_id = str(row.get("team_id") or row.get("team") or "")
    name = str(row.get("name") or row.get("player") or row.get("player_name") or "Player")
    position = _normalize_stat_position(row.get("position") or row.get("pos") or "F")

    gp = max(0, _stat_int(row, "gp", "games_played"))
    goals = max(0, _stat_int(row, "g", "goals"))
    assists = max(0, _stat_int(row, "a", "assists"))
    points = goals + assists

    sog = max(0, _stat_int(row, "sog", "shots", "shots_on_goal"))
    pim = max(0, _stat_int(row, "pim", "penalty_minutes"))
    hits = max(0, _stat_int(row, "hit", "hits"))
    blocks = max(0, _stat_int(row, "blk", "blocks"))
    toi_sec = max(0, _stat_int(row, "toi_sec", "time_on_ice_sec", "toi_total_sec"))

    is_goalie = _is_goalie_stat_row(row)

    normalized = {
        **row,
        "player_id": player_id,
        "id": player_id,
        "team_id": team_id,
        "name": name,
        "player_name": name,
        "position": position,
        "gp": gp,
        "g": goals,
        "goals": goals,
        "a": assists,
        "assists": assists,
        "pts": points,
        "points": points,
        "sog": sog,
        "shots": sog,
        "pim": pim,
        "hit": hits,
        "hits": hits,
        "blk": blocks,
        "blocks": blocks,
        "toi_sec": toi_sec,
        "toi_total_sec": toi_sec,
        "toi": round((toi_sec / max(1, gp)) / 60.0, 1) if gp > 0 else 0.0,
        "points_per_game": round(points / max(1, gp), 3) if gp > 0 else 0.0,
        "goals_per_game": round(goals / max(1, gp), 3) if gp > 0 else 0.0,
        "assists_per_game": round(assists / max(1, gp), 3) if gp > 0 else 0.0,
        "is_goalie": bool(is_goalie),
        "stat_type": "goalie" if is_goalie else "skater",
    }

    if is_goalie:
        shots_against = max(0, _stat_int(row, "shots_against", "sa"))
        saves = max(0, _stat_int(row, "saves", "sv"))
        goals_against = max(0, _stat_int(row, "ga", "goals_against"))

        if shots_against <= 0 and saves > 0:
            shots_against = saves + goals_against

        if saves <= 0 and shots_against > 0:
            saves = max(0, shots_against - goals_against)

        save_pct = saves / max(1, shots_against)
        goalie_toi_sec = max(0, _stat_int(row, "toi_sec", "time_on_ice_sec", "toi_total_sec"))
        if goalie_toi_sec <= 0 and gp > 0:
            goalie_toi_sec = gp * 3600
        gaa = (goals_against * 3600.0) / goalie_toi_sec if goalie_toi_sec > 0 else 0.0

        normalized.update(
            {
                "shots_against": shots_against,
                "saves": saves,
                "ga": goals_against,
                "goals_against": goals_against,
                "w": max(0, _stat_int(row, "w", "wins")),
                "l": max(0, _stat_int(row, "l", "losses")),
                "otl": max(0, _stat_int(row, "otl", "ot_losses")),
                "toi_sec": goalie_toi_sec,
                "toi_total_sec": goalie_toi_sec,
                "toi": round((goalie_toi_sec / max(1, gp)) / 60.0, 1) if gp > 0 else 0.0,
                "save_pct": round(save_pct, 4),
                "gaa": round(gaa, 3),
            }
        )

    return normalized
def _stats_integrity_payload(rows: List[Dict[str, Any]], game_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Dev-facing truth meter for stat sanity.
    """
    warnings: List[str] = []

    skaters = [r for r in rows if not r.get("is_goalie")]
    goalies = [r for r in rows if r.get("is_goalie")]

    # Goalies count too: the (very rare) goofy empty-net goalie goal is a real goal.
    total_player_goals = sum(int(r.get("g", 0) or 0) for r in skaters) + sum(
        int(r.get("g", 0) or 0) for r in goalies
    )

    total_box_goals = 0
    valid_games = 0

    for g in game_results or []:
        if not isinstance(g, dict):
            continue

        try:
            hg = int(round(float(g.get("home_goals", g.get("home_score", 0)) or 0)))
            ag = int(round(float(g.get("away_goals", g.get("away_score", 0)) or 0)))
        except (TypeError, ValueError):
            continue

        if hg < 0 or ag < 0:
            continue

        # Do not count obvious placeholders.
        status = str(g.get("status") or g.get("game_status") or "").lower()
        if hg == 0 and ag == 0 and status not in {"final", "completed", "complete", "simmed", "played"}:
            continue

        total_box_goals += hg + ag
        valid_games += 1

    if valid_games and abs(total_player_goals - total_box_goals) > 0:
        warnings.append(
            f"PLAYER_GOALS_MISMATCH: skater goals {total_player_goals} != game box goals {total_box_goals}."
        )

    top_pts = max([int(r.get("pts", 0) or 0) for r in skaters], default=0)
    top_gp = max([int(r.get("gp", 0) or 0) for r in skaters], default=0)

    if valid_games >= 300 and top_pts < 45:
        warnings.append(
            f"LOW_LEAGUE_SCORING: top scorer has only {top_pts} points after {valid_games} completed games."
        )

    return {
        "skater_rows": len(skaters),
        "goalie_rows": len(goalies),
        "valid_games_counted": int(valid_games),
        "total_player_goals": int(total_player_goals),
        "total_box_goals": int(total_box_goals),
        "goals_match_boxscores": bool(total_player_goals == total_box_goals),
        "top_scorer_points": int(top_pts),
        "top_player_gp": int(top_gp),
        "warnings": warnings,
    }
def _build_schedule_upcoming(session: FranchiseSession, *, limit: int = 14) -> List[Dict[str, Any]]:
    """Next NHL calendar days from the current cursor (real dates) — hub / calendar UI."""
    if str(getattr(session, "phase", "")) != "regular":
        return []
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)
    hi = min(last + 1, cur + int(limit))
    by_day = getattr(session, "by_day", None) or {}
    out: List[Dict[str, Any]] = []
    uid = str(getattr(session, "user_team_id", "") or "")
    for idx in range(cur, hi):
        if idx > last:
            break
        row = cal[idx] if idx < len(cal) else {}
        slots = by_day.get(idx, []) or []
        games: List[Dict[str, Any]] = []
        for sl in slots:
            hid = _safe_slot_team_id(sl, "home_id")
            aid = _safe_slot_team_id(sl, "away_id")
            ht = session.team_by_id.get(hid)
            at = session.team_by_id.get(aid)
            games.append(
                {
                    "home_id": hid,
                    "away_id": aid,
                    "home_name": _display_team(ht) if ht else hid,
                    "away_name": _display_team(at) if at else aid,
                }
            )
        user_plays = bool(uid) and any(uid in (g["home_id"], g["away_id"]) for g in games)
        out.append(
            {
                "day": int(idx),
                "calendar_index": int(idx),
                "iso": str(row.get("iso") or ""),
                "weekday": str(row.get("weekday") or ""),
                "segment": str(row.get("segment") or ""),
                "ui_phase": str(row.get("ui_phase") or ""),
                "tags": list(row.get("tags") or []),
                "games": games,
                "user_plays": user_plays,
            }
        )
    return out
def _nhl_today_payload(session: FranchiseSession) -> Dict[str, Any]:
    """Current calendar row + gameday headline for the hub command deck."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal or str(getattr(session, "phase", "")) != "regular":
        return {}
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)
    if cur > last:
        return {"headline": "Regular season complete — advance for playoffs", "iso": "", "segment": "regular", "calendar_index": cur}
    cur = max(0, min(cur, len(cal) - 1))
    row = dict(cal[cur])
    row["calendar_index"] = int(cur)
    slots = list((getattr(session, "by_day", None) or {}).get(int(session.calendar_cursor), []) or [])
    uid = str(session.user_team_id)
    user_plays = any(uid in (str(s.home_id), str(s.away_id)) for s in slots)
    row["has_league_games"] = bool(slots)
    row["user_game_today"] = bool(user_plays)
    opp = None
    for s in slots:
        hid, aid = str(s.home_id), str(s.away_id)
        if hid == uid:
            opp = session.team_by_id.get(aid)
        elif aid == uid:
            opp = session.team_by_id.get(hid)
        if opp:
            break
    if user_plays and opp:
        row["headline"] = f"Game day vs {_display_team(opp)}"
    elif slots:
        row["headline"] = "League gameday (your club is off)"
    else:
        row["headline"] = str(row.get("ui_note") or "Off day")
    return row
def _results_by_calendar_index(session: FranchiseSession) -> Dict[int, List[Dict[str, Any]]]:
    """
    Simmed box scores keyed by franchise calendar index.

    Important:
    Only valid final games get scores.
    Scheduled placeholders must not leak 0-0 scores to the frontend.
    """
    by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue

        try:
            di = int(g.get("day", -1))
        except (TypeError, ValueError):
            continue

        if di < 0:
            continue

        if not _saved_game_is_final(g):
            continue

        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")

        try:
            hg, ag = _validate_final_game_result_payload(
                home_id=hid,
                away_id=aid,
                home_goals=g.get("home_goals", g.get("home_score")),
                away_goals=g.get("away_goals", g.get("away_score")),
                calendar_day=di,
            )
        except ValueError:
            continue

        by_idx[di].append(
            {
                "game_id": _stable_franchise_game_id(g),
                "home_id": hid,
                "away_id": aid,
                "home_goals": int(hg),
                "away_goals": int(ag),
                "overtime": bool(g.get("overtime") or g.get("went_ot") or g.get("ot")),
                "iso": str(g.get("iso") or ""),
                "status": "final",
                "completed": True,
                "is_final": True,
                "simmed": True,
            }
        )

    return dict(by_idx)
def _coerce_final_score(value: Any) -> int:
    """
    Backend score sanitizer.
    Game scores must be non-negative integers.
    """
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid game score value: {value!r}")

    if n < 0:
        raise ValueError(f"Negative game score value: {value!r}")

    return n
def _validate_final_game_result_payload(
    *,
    home_id: str,
    away_id: str,
    home_goals: Any,
    away_goals: Any,
    calendar_day: int,
) -> Tuple[int, int]:
    """
    Hard validator for a completed game.

    A completed hockey game must have:
    - two different teams
    - valid non-negative integer scores
    - no tie after final
    """
    hid = str(home_id or "").strip()
    aid = str(away_id or "").strip()

    if not hid or not aid:
        raise ValueError(f"Game result missing team id(s) on calendar day {calendar_day}.")

    if hid == aid:
        raise ValueError(f"Self-play game result detected on calendar day {calendar_day}: {hid}.")

    hg = _coerce_final_score(home_goals)
    ag = _coerce_final_score(away_goals)

    if hg == ag:
        raise ValueError(
            f"Final game result cannot be tied on calendar day {calendar_day}: {hid} {hg}, {aid} {ag}."
        )

    return hg, ag
def _saved_game_is_final(g: Dict[str, Any]) -> bool:
    """
    Saved backend result should only be considered final if explicitly final
    OR it came from the sim result store with valid final scores.

    This prevents schedule placeholders from becoming fake 0-0 finals.
    """
    if not isinstance(g, dict):
        return False

    status = str(
        g.get("status")
        or g.get("game_status")
        or g.get("state")
        or ""
    ).strip().lower()

    explicit_final = status in {
        "final",
        "completed",
        "complete",
        "played",
        "done",
        "simmed",
        "finished",
    }

    has_home = g.get("home_goals", None) is not None or g.get("home_score", None) is not None
    has_away = g.get("away_goals", None) is not None or g.get("away_score", None) is not None

    if not has_home or not has_away:
        return False

    try:
        hg = _coerce_final_score(g.get("home_goals", g.get("home_score")))
        ag = _coerce_final_score(g.get("away_goals", g.get("away_score")))
    except ValueError:
        return False

    # Placeholder scheduled games are often 0-0.
    # A true final can never be tied anyway.
    if hg == ag:
        return False

    return explicit_final or bool(g.get("simmed") or g.get("completed") or g.get("is_final"))
def _game_result_calendar_index(g: Dict[str, Any]) -> Optional[int]:
    """Calendar index from a saved game box (day 0 is valid — never use `value or default` on day)."""
    if not isinstance(g, dict):
        return None
    v = g.get("day")
    if v is None:
        v = g.get("calendar_day")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
def _remaining_regular_games_count(session: FranchiseSession) -> int:
    """
    Count unplayed regular-season scheduled games remaining.
    Used to prevent playoffs from starting too early.
    """
    cal = getattr(session, "nhl_calendar", None) or []
    by_day = getattr(session, "by_day", None) or {}

    remaining = 0

    for day_idx, slots in (by_day or {}).items():
        try:
            di = int(day_idx)
        except (TypeError, ValueError):
            continue

        if di < 0 or di >= len(cal):
            continue

        row = cal[di] or {}
        seg = str(row.get("segment") or row.get("season_segment") or "")

        if seg != "regular":
            continue

        remaining += len(slots or [])

    return int(remaining)
def _completed_regular_games_count(session: FranchiseSession) -> int:
    """
    Count valid completed regular-season games from session.game_results.
    """
    cal = getattr(session, "nhl_calendar", None) or []
    count = 0

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue

        try:
            di = int(g.get("day", -1))
        except (TypeError, ValueError):
            continue

        if di < 0 or di >= len(cal):
            continue

        row = cal[di] or {}
        seg = str(row.get("segment") or row.get("season_segment") or "")

        if seg != "regular":
            continue

        if _saved_game_is_final(g):
            count += 1

    return int(count)
def _regular_season_is_truly_complete(session: FranchiseSession) -> bool:
    """
    True only when:
    - calendar cursor is past the regular season boundary
    - no regular-season slots remain in by_day
    """
    _sync_nhl_calendar_bounds(session)

    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "nhl_regular_season_last_index", 0) or 0)

    if cursor <= last:
        return False

    return _remaining_regular_games_count(session) == 0
def _game_results_by_calendar_day(session: FranchiseSession) -> Dict[int, List[Dict[str, Any]]]:
    """
    Full completed game payloads grouped by calendar index.

    Slots are cleared after sim, so this preserves past games.
    But it must only return real completed games.
    """
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for g in getattr(session, "game_results", None) or []:
        if not isinstance(g, dict):
            continue

        try:
            di = int(g.get("day", -1))
        except (TypeError, ValueError):
            continue

        if di < 0:
            continue

        if not _saved_game_is_final(g):
            continue

        out[di].append(g)

    return dict(out)
def _stable_franchise_game_id(g: Dict[str, Any]) -> str:
    """Stable recap key: prefer stored game_id, else day+matchup (older saves / edge cases)."""
    existing = str(g.get("game_id") or "").strip()
    if existing:
        return existing
    try:
        di = int(g.get("day", -1))
    except (TypeError, ValueError):
        di = -1
    return f"d{di}_{g.get('home_id')}_{g.get('away_id')}"
def _stable_franchise_game_id_from_row(hid: str, aid: str, calendar_day: int) -> str:
    return f"d{int(calendar_day)}_{hid}_{aid}"
def _ginfo_from_saved(session: FranchiseSession, saved: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a saved completed result into frontend game info.

    Important:
    - Only attaches scores if the game is truly final.
    - Never turns missing scores into 0-0.
    """
    hid = str(saved.get("home_id") or "")
    aid = str(saved.get("away_id") or "")
    ht = session.team_by_id.get(hid)
    at = session.team_by_id.get(aid)

    try:
        di = int(saved.get("day", -1))
    except (TypeError, ValueError):
        di = -1

    row = {
        "game_id": _stable_franchise_game_id(saved),
        "id": _stable_franchise_game_id(saved),
        "home_id": hid,
        "away_id": aid,
        "home_name": str(saved.get("home_name") or (_display_team(ht) if ht else hid)),
        "away_name": str(saved.get("away_name") or (_display_team(at) if at else aid)),
        "home_abbr": _team_abbr(ht, hid),
        "away_abbr": _team_abbr(at, aid),
        "day": di,
        "calendar_day": di,
        "iso": str(saved.get("iso") or ""),
        "date": str(saved.get("iso") or ""),
        "status": "scheduled",
        "completed": False,
        "is_final": False,
        "simmed": False,
    }

    if _saved_game_is_final(saved):
        try:
            hg, ag = _validate_final_game_result_payload(
                home_id=hid,
                away_id=aid,
                home_goals=saved.get("home_goals", saved.get("home_score")),
                away_goals=saved.get("away_goals", saved.get("away_score")),
                calendar_day=di,
            )
        except ValueError:
            return row

        row.update(
            {
                "home_goals": int(hg),
                "away_goals": int(ag),
                "home_score": int(hg),
                "away_score": int(ag),
                "overtime": bool(saved.get("overtime") or saved.get("went_ot") or saved.get("ot")),
                "status": "final",
                "completed": True,
                "is_final": True,
                "simmed": True,
            }
        )

    return row
def _nhl_calendar_full_with_slates(session: FranchiseSession) -> List[Dict[str, Any]]:
    """Full season calendar rows plus NHL slate + user focus + final scores for the franchise UI."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return []
    by_day = getattr(session, "by_day", None) or {}
    uid = str(session.user_team_id)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    results_by_idx = _results_by_calendar_index(session)
    saved_by_day = _game_results_by_calendar_day(session)

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(cal):
        d = dict(row)
        d["calendar_index"] = int(i)
        d["is_today_cursor"] = bool(i == cur)
        d["is_past"] = bool(i < cur)
        slots = list(by_day.get(i, []) or [])
        res_list = list(results_by_idx.get(i, []) or [])
        saved_list = list(saved_by_day.get(i, []) or [])
        games: List[Dict[str, Any]] = []

        for sl in slots:
            hid = _safe_slot_team_id(sl, "home_id")
            aid = _safe_slot_team_id(sl, "away_id")
            ht = session.team_by_id.get(hid)
            at = session.team_by_id.get(aid)
            ginfo: Dict[str, Any] = {
                "home_id": hid,
                "away_id": aid,
                "home_name": _display_team(ht) if ht else hid,
                "away_name": _display_team(at) if at else aid,
                "home_abbr": _team_abbr(ht, hid),
                "away_abbr": _team_abbr(at, aid),
            }
            for rg in res_list:
                if str(rg.get("home_id")) == hid and str(rg.get("away_id")) == aid:
                    ginfo["home_goals"] = int(rg.get("home_goals", 0) or 0)
                    ginfo["away_goals"] = int(rg.get("away_goals", 0) or 0)
                    ginfo["overtime"] = bool(rg.get("overtime"))
                    ginfo["game_id"] = str(rg.get("game_id") or "") or _stable_franchise_game_id_from_row(hid, aid, i)
                    break
            if "game_id" not in ginfo:
                ginfo["game_id"] = _stable_franchise_game_id_from_row(hid, aid, i)
            games.append(ginfo)

        if not games and saved_list:
            for sg in saved_list:
                games.append(_ginfo_from_saved(session, sg))

        d["has_slate"] = bool(slots or saved_list)
        d["games"] = games
        user_game: Optional[Dict[str, Any]] = None
        for g in games:
            hid = str(g.get("home_id") or "")
            aid = str(g.get("away_id") or "")
            if uid not in (hid, aid):
                continue
            is_home = hid == uid
            opp_id = aid if is_home else hid
            opp_name = str(g.get("away_name") if is_home else g.get("home_name") or opp_id)
            opp_abbr = str(g.get("away_abbr") if is_home else g.get("home_abbr") or opp_id)
            res_bits: Dict[str, Any] = {
                "home_away": "home" if is_home else "away",
                "opponent_id": opp_id,
                "opponent_name": opp_name,
                "opponent_abbr": opp_abbr,
            }
            res_bits["game_id"] = str(g.get("game_id") or "").strip() or _stable_franchise_game_id_from_row(hid, aid, i)
            if "home_goals" in g:
                hg = int(g["home_goals"])
                ag = int(g["away_goals"])
                ot = bool(g.get("overtime"))
                won = (hg > ag) if is_home else (ag > hg)
                user_sc = int(hg) if is_home else int(ag)
                opp_sc = int(ag) if is_home else int(hg)
                if won:
                    letter = "W"
                elif ot and not won:
                    letter = "OTL"
                else:
                    letter = "L"
                res_bits["record_line"] = f"{user_sc}-{opp_sc}{' OT' if ot else ''} ({letter})"
                res_bits["result_letter"] = letter
                res_bits["user_score"] = user_sc
                res_bits["opp_score"] = opp_sc
                res_bits["overtime"] = ot
            user_game = res_bits
            break
        d["user_game"] = user_game
        d["special_event"] = _special_event_overlay(session, d)
        out.append(d)
    return out
def _team_abbr(team: Any, tid: str) -> str:
    standard = {
        "anaheim ducks": "ANA",
        "boston bruins": "BOS",
        "buffalo sabres": "BUF",
        "calgary flames": "CGY",
        "carolina hurricanes": "CAR",
        "chicago blackhawks": "CHI",
        "colorado avalanche": "COL",
        "columbus blue jackets": "CBJ",
        "dallas stars": "DAL",
        "detroit red wings": "DET",
        "edmonton oilers": "EDM",
        "florida panthers": "FLA",
        "los angeles kings": "LAK",
        "minnesota wild": "MIN",
        "montreal canadiens": "MTL",
        "montr├⌐al canadiens": "MTL",
        "nashville predators": "NSH",
        "new jersey devils": "NJD",
        "new york islanders": "NYI",
        "new york rangers": "NYR",
        "ottawa senators": "OTT",
        "philadelphia flyers": "PHI",
        "pittsburgh penguins": "PIT",
        "seattle kraken": "SEA",
        "san jose sharks": "SJS",
        "st. louis blues": "STL",
        "st louis blues": "STL",
        "tampa bay lightning": "TBL",
        "toronto maple leafs": "TOR",
        "utah hockey club": "UTA",
        "vancouver canucks": "VAN",
        "vegas golden knights": "VGK",
        "washington capitals": "WSH",
        "winnipeg jets": "WPG",
    }

    raw_tid = str(tid or "").strip().upper()
    if raw_tid in standard.values():
        return raw_tid

    if team is not None:
        city = str(getattr(team, "city", "") or "").strip().lower()
        name = str(getattr(team, "name", "") or "").strip().lower()
        full = f"{city} {name}".strip()
        if full in standard:
            return standard[full]
        if name:
            # fallback by nickname token when city text is unusual
            by_name = {
                "ducks": "ANA",
                "bruins": "BOS",
                "sabres": "BUF",
                "flames": "CGY",
                "hurricanes": "CAR",
                "blackhawks": "CHI",
                "avalanche": "COL",
                "blue jackets": "CBJ",
                "stars": "DAL",
                "red wings": "DET",
                "oilers": "EDM",
                "panthers": "FLA",
                "kings": "LAK",
                "wild": "MIN",
                "canadiens": "MTL",
                "predators": "NSH",
                "devils": "NJD",
                "islanders": "NYI",
                "rangers": "NYR",
                "senators": "OTT",
                "flyers": "PHI",
                "penguins": "PIT",
                "kraken": "SEA",
                "sharks": "SJS",
                "blues": "STL",
                "lightning": "TBL",
                "maple leafs": "TOR",
                "hockey club": "UTA",
                "canucks": "VAN",
                "golden knights": "VGK",
                "capitals": "WSH",
                "jets": "WPG",
            }
            if name in by_name:
                return by_name[name]

    return raw_tid[:3] if raw_tid else "NHL"
def _city_lower(team: Any) -> str:
    return str(getattr(team, "city", "") or "").strip().lower()
def _is_canadian_franchise_team(team: Any) -> bool:
    if team is None:
        return False
    city = _city_lower(team)
    hints = ("toronto", "montr├⌐al", "montreal", "ottawa", "winnipeg", "edmonton", "calgary", "vancouver")
    if any(h in city for h in hints):
        return True
    nm = str(getattr(team, "name", "") or "").lower()
    for hint in ("maple leaf", "canadien", "senator", "jet", "oiler", "flame", "canuck"):
        if hint in nm:
            return True
    return False
def _standing_row_snapshot(session: FranchiseSession, tid: str, r: Any) -> Dict[str, Any]:
    tm = session.team_by_id.get(tid)
    return {
        "id": str(tid),
        "abbr": _team_abbr(tm, str(tid)),
        "name": str(getattr(r, "name", tid)),
    }
def _top_two_teams_overall(session: FranchiseSession) -> List[Dict[str, Any]]:
    if not getattr(session, "standings", None):
        return []
    rows: List[Tuple[int, str, Any]] = []
    for tid, r in session.standings.records.items():
        rows.append((int(getattr(r, "points", 0) or 0), str(tid), r))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [_standing_row_snapshot(session, tid, r) for _, tid, r in rows[:2]]
def _top_two_canadian_teams(session: FranchiseSession) -> List[Dict[str, Any]]:
    if not getattr(session, "standings", None):
        return []
    rows: List[Tuple[int, str, Any]] = []
    for tid, r in session.standings.records.items():
        tm = session.team_by_id.get(tid)
        if not _is_canadian_franchise_team(tm):
            continue
        rows.append((int(getattr(r, "points", 0) or 0), str(tid), r))
    rows.sort(key=lambda x: (-x[0], x[1]))
    if len(rows) >= 2:
        return [_standing_row_snapshot(session, tid, r) for _, tid, r in rows[:2]]
    return _top_two_teams_overall(session)
def _special_event_overlay(session: FranchiseSession, day: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Showcase matchups derived from current standings (same pass as calendar cells)."""
    tags = {str(x).lower() for x in (day.get("tags") or [])}
    tbd = {"id": "", "abbr": "?", "name": "TBD"}
    if "winter_classic" in tags:
        pair = _top_two_teams_overall(session)
        if len(pair) >= 2:
            return {"kind": "winter_classic", "title": "Winter Classic", "home": pair[0], "away": pair[1]}
        return {"kind": "winter_classic", "title": "Winter Classic", "home": tbd, "away": tbd}
    if "heritage_classic" in tags:
        pair = _top_two_canadian_teams(session)
        if len(pair) >= 2:
            return {"kind": "heritage_classic", "title": "Heritage Classic", "home": pair[0], "away": pair[1]}
        return {"kind": "heritage_classic", "title": "Heritage Classic", "home": tbd, "away": tbd}
    if "four_nations" in tags:
        return {
            "kind": "four_nations",
            "title": "4 Nations Face-Off",
            "nations": [
                {"code": "CAN", "label": "Canada"},
                {"code": "USA", "label": "United States"},
                {"code": "SWE", "label": "Sweden"},
                {"code": "FIN", "label": "Finland"},
            ],
        }
    return None
def _nhl_calendar_strip(session: FranchiseSession, *, before: int = 10, after: int = 28) -> List[Dict[str, Any]]:
    """Window of NHL dates around the cursor for the calendar screen."""
    cal = getattr(session, "nhl_calendar", None) or []
    if not cal:
        return []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    lo = max(0, cur - int(before))
    hi = min(len(cal), cur + int(after))
    by_day = getattr(session, "by_day", None) or {}
    out: List[Dict[str, Any]] = []
    for i in range(lo, hi):
        d = dict(cal[i]) if i < len(cal) else {}
        d["calendar_index"] = i
        d["has_game"] = bool(by_day.get(i))
        d["is_today"] = i == cur
        out.append(d)
    return out
def _serialize_player_row(
    p: Any,
    *,
    include_ratings: bool = False,
    session: Optional[FranchiseSession] = None,
    _team: Optional[Any] = None,
) -> Dict[str, Any]:
    ident = getattr(p, "identity", None)
    ovr_f = getattr(p, "ovr", None)
    try:
        ov = float(ovr_f() if callable(ovr_f) else ovr_f)
    except Exception:
        ov = 0.0
    pid = str(getattr(p, "id", "") or "")
    hcm = int(getattr(ident, "height_cm", 0) or 0) if ident else 0
    row: Dict[str, Any] = {
        "player_id": pid,
        "name": str(getattr(ident, "name", None) or "?"),
        "position": str(getattr(getattr(ident, "position", None), "value", ident) if ident else "?"),
        "ovr": round(ov * 99, 1) if ov <= 1.5 else round(ov, 1),
        "morale": round(float(getattr(getattr(p, "psych", None), "morale", 0.5) or 0.5), 3),
        "age": int(getattr(ident, "age", 0) or 0),
        "nationality": str(getattr(ident, "birth_country", "") or ""),
        "height_cm": hcm,
        "height_display": height_cm_to_imperial(hcm) if hcm else "—",
        "archetype": str(getattr(p, "archetype", "") or ""),
        "contract": {
            "salary": round(_player_cap_hit_millions(p), 3),
            "cap_hit": round(_player_cap_hit_millions(p), 3),
        },
    }
    gr = _get_live_injury_games_remaining(p)
    hstat = _get_player_health_status(p)
    if gr > 0 and hstat == "HEALTHY":
        hstat = "INJURED"
    injured = _is_player_live_injured(p)
    tier_guess = _get_live_injury_tier(p)
    if injured and tier_guess is None:
        tier_guess = "minor"
    tier = str(tier_guess or "").lower() if tier_guess else ""
    inj_label = _tier_human_label(tier) if injured else ""
    if gr > 0:
        avail = "Out"
    elif hstat == "DAY_TO_DAY":
        avail = "Day-to-day"
    else:
        avail = "Available"
    ret_est, ret_iso = ("", "")
    if session is not None and gr > 0:
        ret_est, ret_iso = _estimate_return_from_games_remaining(session, gr)
    elif gr > 0:
        ret_est = f"In {gr} games"
    disp_status = "HEALTHY"
    if injured:
        disp_status = "INJURED" if gr > 0 else hstat
    row.update(
        {
            "injury_status": disp_status,
            "health_status": disp_status,
            "injury": inj_label if injured else "",
            "injury_tier": tier if injured else "",
            "injury_type": tier if injured else "",
            "injury_games_remaining": int(gr),
            "games_remaining": int(gr),
            "is_injured": bool(injured),
            "availability_status": avail,
            "return_estimate": ret_est,
            "return_date": ret_iso,
        }
    )
    if include_ratings:
        row["rating_groups"] = _rating_groups_for_player(p)
    try:
        from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
            get_base_ovr_display,
            get_effective_ovr_display,
            serialize_ovr_modifiers_for_ui,
        )

        base_ovr = get_base_ovr_display(p)
        eff_ovr = get_effective_ovr_display(p)
        mods = serialize_ovr_modifiers_for_ui(p)
        row["base_ovr"] = base_ovr
        row["effective_ovr"] = eff_ovr
        row["ovr_modifiers"] = mods
        if mods and eff_ovr != base_ovr:
            row["ovr"] = eff_ovr
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from app.sim_engine.franchise.conduct_incidents import (  # noqa: WPS433
            get_active_incident_for_player,
            player_eligible_to_dress,
            serialize_incident_for_ui,
        )

        eligible = bool(player_eligible_to_dress(p, session))
        row["conduct_eligible_to_play"] = eligible
        row["conduct_incident_id"] = str(getattr(p, "_conduct_incident_id", "") or "")
        row["conduct_dress_backlash_risk"] = float(getattr(p, "_conduct_dress_backlash_risk", 0) or 0)
        row["conduct_trade_restricted"] = bool(getattr(p, "_conduct_trade_restricted", False))
        cgr = int(getattr(p, "_world_conduct_games_remaining", 0) or 0)
        row["conduct_games_remaining"] = cgr
        if not eligible:
            row["availability_status"] = (
                "Suspended"
                if str(getattr(p, "_world_conduct_status", "") or "") == "league_suspended"
                else "Leave"
            )
            if cgr > 0:
                row["return_estimate"] = f"In {cgr} games"
        if session is not None and row.get("conduct_incident_id"):
            inc = get_active_incident_for_player(
                session, str(getattr(p, "id", "") or getattr(p, "player_id", "") or "")
            )
            if isinstance(inc, dict):
                row["conduct_incident"] = serialize_incident_for_ui(inc)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return row
def _prospect_league_for_player(p: Any) -> Optional[str]:
    asg = getattr(p, "_franchise_assignment", None) or {}
    if isinstance(asg, dict) and str(asg.get("level") or "").lower() == "junior":
        return str(asg.get("league_code") or asg.get("league_name") or "") or None
    pool = str(getattr(p, "pool_context", "") or "").lower()
    if pool in ("junior", "college", "european_junior"):
        if isinstance(asg, dict):
            code = str(asg.get("league_code") or asg.get("league_name") or "")
            if code:
                return code
    return None


_LEAGUE_DISPLAY_BY_CODE: Dict[str, str] = {
    "CHL_OHL": "OHL",
    "CHL_WHL": "WHL",
    "CHL_QMJHL": "QMJHL",
    "USHL": "USHL",
    "NCAA": "NCAA",
    "EU_J_SHL": "SHL",
    "EU_J_LIIGA": "Liiga",
    "EU_J_DEL": "DEL",
    "EU_J_SWISS": "NLA",
    "EU_J_CZ": "Czechia",
    "EU_J_SK": "Slovakia",
    "EU_J_KHL_JR": "MHL",
    "EU_J_NOR": "Norway",
    "EU_J_DEN": "Denmark",
    "EU_J_AUT": "Austria",
    "AHL": "AHL",
    "OHL": "OHL",
    "WHL": "WHL",
    "QMJHL": "QMJHL",
    "SHL": "SHL",
    "LIIGA": "Liiga",
}


def _league_display_for_code(code: str, name: str = "") -> str:
    """Short fan-facing league label from internal code/name."""
    c = str(code or "").strip().upper()
    if c in _LEAGUE_DISPLAY_BY_CODE:
        return _LEAGUE_DISPLAY_BY_CODE[c]
    n = str(name or "").strip()
    n_low = n.lower()
    if n and "cluster" not in n_low and "ladder" not in n_low and "canadian hockey league" not in n_low:
        if len(n) <= 14:
            return n
    if "QMJ" in c:
        return "QMJHL"
    if "OHL" in c:
        return "OHL"
    if "WHL" in c:
        return "WHL"
    if "NCAA" in c:
        return "NCAA"
    if "USHL" in c:
        return "USHL"
    if "SHL" in c:
        return "SHL"
    if "LIIGA" in c:
        return "Liiga"
    if "AHL" in c:
        return "AHL"
    return c.replace("_", " ") if c else "Junior"


def _team_name_from_assignment(p: Any) -> str:
    asg = getattr(p, "_franchise_assignment", None) or {}
    if isinstance(asg, dict):
        return str(asg.get("club") or asg.get("team_name") or "")
    return ""


def _attach_prospect_context_to_row(
    row: Dict[str, Any],
    p: Any,
    *,
    league_code: Optional[str] = None,
    league_name: Optional[str] = None,
    team_id: Optional[str] = None,
    team_name: Optional[str] = None,
) -> bool:
    """Attach junior/prospect league + team context to a serialized player row."""
    asg = getattr(p, "_franchise_assignment", None) or {}
    lc = str(league_code or row.get("league_code") or _prospect_league_for_player(p) or "")
    ln = str(
        league_name
        or row.get("league_name")
        or (asg.get("league_name") if isinstance(asg, dict) else "")
        or ""
    )
    if not lc and isinstance(asg, dict):
        lc = str(asg.get("league_code") or "")

    pool = str(getattr(p, "pool_context", "") or "").lower()
    is_junior = (
        (isinstance(asg, dict) and str(asg.get("level") or "").lower() == "junior")
        or pool in ("junior", "college", "european_junior")
        or bool(lc)
        or bool(ln)
    )
    if not is_junior:
        return False

    tid = str(team_id or row.get("team_id") or "")
    if not tid and isinstance(asg, dict):
        tid = str(asg.get("team_id") or "")
    if not tid:
        ctx = getattr(p, "context", None)
        tid = str(getattr(ctx, "current_team_id", "") or "")

    tn = str(team_name or row.get("team_name") or _team_name_from_assignment(p) or "")
    display = _league_display_for_code(lc, ln)

    if lc:
        row["league_code"] = lc
    if ln:
        row["league_name"] = ln
    row["league_display"] = display
    row["league"] = display
    if tid:
        row["team_id"] = tid
    if tn:
        row["team_name"] = tn
    return True


def _attach_prospect_stats_to_row(
    row: Dict[str, Any],
    p: Any,
    *,
    league_code: Optional[str] = None,
    session: Optional[FranchiseSession] = None,
) -> None:
    """Junior/prospect season lines only — never NHL ledger stats."""
    lc = league_code or _prospect_league_for_player(p)
    if not lc:
        return
    try:
        from app.sim_engine.generation.prospect_league_scoring import prospect_stats_for_api

        sim = getattr(session, "sim", None) if session is not None else None
        rng = getattr(sim, "rng", None) if sim is not None else None
        calendar_iso = ""
        season_year = None
        if session is not None:
            try:
                from app.sim_engine.franchise.schedule import _calendar_iso_for_day

                calendar_iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
            except Exception:
                cal = list(getattr(session, "nhl_calendar", None) or [])
                cur = int(getattr(session, "calendar_cursor", 0) or 0)
                if cal and 0 <= cur < len(cal):
                    calendar_iso = str(cal[cur].get("iso") or "")
            season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
        stats = prospect_stats_for_api(
            p,
            lc,
            rng=rng,
            calendar_iso=calendar_iso or None,
            season_year=season_year,
        )
        for k, v in stats.items():
            if v is not None:
                row[k] = v
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _rows_from_players_list(
    players: Any,
    *,
    include_ratings: bool = False,
    session: Optional[FranchiseSession] = None,
    team: Optional[Any] = None,
    league_code: Optional[str] = None,
    league_name: Optional[str] = None,
    team_id: Optional[str] = None,
    team_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in players or []:
        if getattr(p, "retired", False):
            continue
        row = _serialize_player_row(p, include_ratings=include_ratings, session=session, _team=team)
        _attach_prospect_context_to_row(
            row,
            p,
            league_code=league_code,
            league_name=league_name,
            team_id=team_id,
            team_name=team_name,
        )
        _attach_prospect_stats_to_row(row, p, league_code=league_code, session=session)
        rows.append(row)
    rows.sort(key=lambda x: -float(x.get("ovr") or 0))
    return rows
def _serialize_development_leagues(
    blocks: Any,
    *,
    session: Optional[FranchiseSession] = None,
) -> List[Dict[str, Any]]:
    """API-safe copy: league stores Player objects under each junior team."""
    out: List[Dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        teams_out: List[Dict[str, Any]] = []
        for tm in block.get("teams") or []:
            if not isinstance(tm, dict):
                continue
            teams_out.append(
                {
                    "team_id": tm.get("team_id"),
                    "name": tm.get("name"),
                    "players": _rows_from_players_list(
                        tm.get("players"),
                        session=session,
                        league_code=str(block.get("league_code") or ""),
                        league_name=str(block.get("league_name") or ""),
                        team_id=str(tm.get("team_id") or ""),
                        team_name=str(tm.get("name") or ""),
                    ),
                }
            )
        out.append(
            {
                "league_code": block.get("league_code"),
                "league_name": block.get("league_name"),
                "teams": teams_out,
            }
        )
    return out
def _build_roster_browser(
    sim: Any,
    user_team_id: Optional[str] = None,
    franchise_session: Optional[FranchiseSession] = None,
) -> Dict[str, Any]:
    league = getattr(sim, "league", None)
    if league is None:
        return {
            "organizations": [],
            "free_agents": [],
            "overseas_free_agents": [],
            "development_leagues": [],
            "counts": {},
        }
    uid = str(user_team_id) if user_team_id is not None else ""
    orgs: List[Dict[str, Any]] = []
    for t in getattr(league, "teams", None) or []:
        raw_tid = getattr(t, "team_id", None)
        if raw_tid is None:
            raw_tid = getattr(t, "id", None)
        tid = str(raw_tid) if raw_tid is not None else ""
        is_user = bool(uid) and tid == uid
        orgs.append(
            {
                "team_id": tid,
                "name": _display_team(t),
                "nhl": _rows_from_players_list(
                    getattr(t, "roster", None),
                    include_ratings=is_user,
                    session=franchise_session,
                    team=t,
                ),
                "ahl": _rows_from_players_list(
                    getattr(t, "ahl_roster", None),
                    include_ratings=is_user,
                    session=franchise_session,
                    team=t,
                ),
                "echl": _rows_from_players_list(
                    getattr(t, "echl_roster", None),
                    include_ratings=is_user,
                    session=franchise_session,
                    team=t,
                ),
            }
        )
    try:
        from app.sim_engine.league_hierarchy_bootstrap import count_pool_players

        counts: Dict[str, Any] = dict(count_pool_players(league))
    except Exception:
        counts = {}
    return {
        "organizations": orgs,
        "free_agents": _rows_from_players_list(getattr(league, "free_agents", None), session=franchise_session)[
            :420
        ],
        "overseas_free_agents": _rows_from_players_list(
            getattr(league, "overseas_free_agents", None), session=franchise_session
        )[:260],
        "development_leagues": _serialize_development_leagues(
            getattr(league, "development_leagues", None),
            session=franchise_session,
        ),
        "counts": counts,
    }
def build_draft_class_rankings(session: FranchiseSession, sim: Any) -> Dict[str, Any]:
    """Combined junior/prospect board for UI; ranks draft-age skaters in development leagues."""
    league = getattr(sim, "league", None)
    if league is None:
        return {"entries": [], "subtitle": "", "total": 0}
    from app.sim_engine.generation.prospect_league_scoring import (
        prospect_stats_for_api,
        normalize_league_leader_board,
    )

    rng = getattr(sim, "rng", None)
    calendar_iso = ""
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    try:
        from app.sim_engine.franchise.schedule import _calendar_iso_for_day

        calendar_iso = _calendar_iso_for_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    stat_keys = (
        "gp",
        "games_played",
        "goals",
        "assists",
        "points",
        "ppg",
        "points_per_game",
        "wins",
        "losses",
        "ot_losses",
        "save_pct",
        "gaa",
        "shutouts",
        "pim",
        "production_context",
        "translation_risk",
        "scoring_environment",
        "league_difficulty",
        "production_adjusted_score",
        "league_scoring_profile",
        "actual_stats",
        "projected_stats",
        "recent_form",
        "projected_gp",
        "projected_points",
        "projected_ppg",
        "stock_delta",
        "stock_label",
        "stock_trend",
    )
    prospects: List[Dict[str, Any]] = []
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "")
        title = str(block.get("league_name") or "")
        for tm in block.get("teams") or []:
            team_name = str(tm.get("name") or "")
            team_id = str(tm.get("team_id") or "")
            for p in tm.get("players") or []:
                if getattr(p, "retired", False):
                    continue
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 99) or 99) if ident else 99
                if age > 20:
                    continue
                ovr_f = getattr(p, "ovr", None)
                try:
                    ov = float(ovr_f() if callable(ovr_f) else ovr_f)
                    ovr99 = round(ov * 99, 1) if ov <= 1.5 else round(ov, 1)
                except Exception:
                    ovr99 = 0.0
                pk = str(getattr(p, "id", "") or "")
                if not pk:
                    continue
                h = abs(_stable_hash(pk)) % 997
                scout = max(18.0, min(99.0, ovr99 + (h % 23) - 11))
                tier = ("A", "B", "C")[h % 3]
                stats = prospect_stats_for_api(
                    p,
                    code,
                    rng=rng,
                    calendar_iso=calendar_iso or None,
                    season_year=season_year,
                )
                row = {
                    "key": pk,
                    "name": _name_str(p),
                    "position": _pos_str(p),
                    "age": age,
                    "true_ovr": ovr99,
                    "scout_grade": round(scout, 1),
                    "scout_tier": tier,
                    "league_code": code,
                    "league_name": title,
                    "league_display": _league_display_for_code(code, title),
                    "league": _league_display_for_code(code, title),
                    "team_id": team_id,
                    "team_name": team_name,
                    "team": team_name,
                    "_sort": ovr99,
                }
                for sk in stat_keys:
                    if stats.get(sk) is not None:
                        row[sk] = stats[sk]
                prospects.append(row)

    by_league: Dict[str, List[Dict[str, Any]]] = {}
    for row in prospects:
        by_league.setdefault(str(row.get("league_code") or ""), []).append(row)
    for code, group in by_league.items():
        normalize_league_leader_board(group, code, rng=rng)

    prospects.sort(key=lambda x: -float(x["_sort"]))
    prev = dict(getattr(session, "draft_rank_prev", None) or {})
    entries: List[Dict[str, Any]] = []
    for i, row in enumerate(prospects[:320]):
        rank = i + 1
        key = str(row["key"])
        pr = int(prev.get(key, rank))
        delta = pr - rank
        if key not in prev:
            trend = "NEW"
        elif delta >= 2:
            trend = "UP"
        elif delta <= -2:
            trend = "DOWN"
        else:
            trend = "SAME"
        entries.append(
            {
                **{k: v for k, v in row.items() if k != "_sort"},
                "rank": rank,
                "rank_prev": pr,
                "rank_delta": abs(delta),
                "trend": trend,
            }
        )
    return {
        "entries": entries,
        "subtitle": f"Draft-age (≤20) in dev leagues ┬╖ showing {len(entries)}",
        "total": len(prospects),
    }
def snapshot_draft_rank_prev(session: FranchiseSession, sim: Any) -> None:
    """Store current draft ranks so the next payload can show trend vs last advance."""
    board = build_draft_class_rankings(session, sim)
    entries = board.get("entries") or []
    session.draft_rank_prev = {str(e.get("key")): int(e.get("rank", i + 1)) for i, e in enumerate(entries) if e.get("key")}


def _stable_storyline_id(raw: Dict[str, Any]) -> str:
    explicit = str(raw.get("id") or "").strip()
    if explicit:
        return explicit
    parts = [
        str(raw.get("type") or raw.get("tone") or "storyline"),
        str(raw.get("calendar_iso") or raw.get("date") or raw.get("day") or ""),
        str(raw.get("team_id") or raw.get("team") or ""),
        str(raw.get("player_id") or raw.get("player_name") or ""),
        str(raw.get("headline") or raw.get("title") or raw.get("text") or raw.get("summary") or ""),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8", "ignore")).hexdigest()[:12]
    return f"story_{digest}"


def _normalize_storyline_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Unify franchise + SimEngine-style rows for the Storylines UI."""
    out: Dict[str, Any] = {}
    out["type"] = str(raw.get("type") or raw.get("tone") or "news")
    d = raw.get("date")
    if d is None or d == "":
        d = raw.get("calendar_iso") or raw.get("day") or ""
    out["date"] = d
    out["headline"] = str(raw.get("headline") or raw.get("title") or "").strip()
    out["team"] = str(raw.get("team") or raw.get("team_id") or raw.get("to_team_id") or "").strip()
    plist = raw.get("players")
    if isinstance(plist, list):
        out["players"] = [str(x) for x in plist if str(x).strip()]
    else:
        out["players"] = []
    out["priority"] = str(raw.get("priority") or "MEDIUM").upper()
    if raw.get("from_team_id") is not None:
        out["from_team_id"] = str(raw.get("from_team_id") or "")
    if raw.get("calendar_iso") is not None:
        out["calendar_iso"] = str(raw.get("calendar_iso") or "")
    if raw.get("tone") is not None:
        out["tone"] = raw.get("tone")
    if raw.get("details") is not None:
        out["details"] = str(raw.get("details") or "")
    if raw.get("team_id") is not None:
        out["team_id"] = str(raw.get("team_id") or "").strip()
    if raw.get("team_abbrev") is not None:
        out["team_abbrev"] = str(raw.get("team_abbrev") or "").strip()
    if raw.get("player_name") is not None:
        out["player_name"] = str(raw.get("player_name") or "").strip()
    if raw.get("summary") is not None:
        out["summary"] = str(raw.get("summary") or "").strip()
    if raw.get("text") is not None:
        out["text"] = str(raw.get("text") or "").strip()
    out["id"] = _stable_storyline_id(raw)
    action_opts = raw.get("action_options")
    if isinstance(action_opts, list):
        norm_opts: List[Dict[str, Any]] = []
        for idx, opt in enumerate(action_opts):
            if not isinstance(opt, dict):
                continue
            oid = str(opt.get("id") or f"opt_{idx}")
            norm_opts.append(
                {
                    "id": oid,
                    "label": str(opt.get("label") or oid.replace("_", " ").title()),
                    "effects": dict(opt.get("effects") or {}),
                    "effect_summary": str(opt.get("effect_summary") or "").strip(),
                }
            )
        if norm_opts:
            out["action_options"] = norm_opts
    if raw.get("cause") is not None:
        out["cause"] = str(raw.get("cause") or "").strip()
    if raw.get("effects") is not None:
        out["effects"] = dict(raw.get("effects") or {})
    if raw.get("effect_summary") is not None:
        out["effect_summary"] = str(raw.get("effect_summary") or "").strip()
    for key in (
        "storyline_id",
        "stable_key",
        "category",
        "severity",
        "short_summary",
        "description",
        "player_id",
        "player_position",
        "player_overall",
        "team_name",
        "evidence",
        "requires_action",
        "status",
        "source",
        "heat",
        "credibility",
        "repeat_count",
        "escalated_from",
        "related_team_ids",
        "related_player_ids",
        "games_remaining",
        "games_initial",
        "return_date",
        "return_estimate",
        "overall_before",
        "overall_after",
        "overall_delta",
        "base_overall",
        "effective_overall",
        "impact_reason",
        "follow_up",
        "arc_status",
        "legal_severity",
        "event_type",
        "cause_type",
        "cause_event_id",
        "culprit_player_id",
        "affected_player_ids",
        "source_label",
        "user_visible_explanation",
        "recovery_conditions",
        "resolution_condition",
        "resolution_reason",
        "culprit_player_name",
        "incident_id",
        "eligible_to_play",
        "team_can_override",
        "allegation_note",
        "information_status",
        "legal_status",
        "league_status",
        "team_status",
        "conduct_model",
        "dress_backlash_risk",
        "incident_family",
        "evidence_confidence",
        "resolution",
        "kind",
        "arc_tier",
        "calendar_day",
        "team_abbr",
        "from_team_name",
        "to_team_name",
        "from_team_abbrev",
        "to_team_abbrev",
        "related_teams",
        "teams",
        "trade_category",
        "trade_id",
        "reason_codes",
        "reason_text",
        "execution",
        # Rink Oddities section + trade-demand cards (dropped here before).
        "goofy",
        "goofy_kind",
        "trade_demand",
        "tags",
    ):
        if raw.get(key) is not None:
            out[key] = raw.get(key)
    if raw.get("requires_action") is not None:
        out["requires_action"] = bool(raw.get("requires_action"))
    if raw.get("eligible_to_play") is not None:
        out["eligible_to_play"] = bool(raw.get("eligible_to_play"))
    if raw.get("team_can_override") is not None:
        out["team_can_override"] = bool(raw.get("team_can_override"))
    return out
def _normalize_notification_payload(raw: Any, index: int) -> Dict[str, Any]:
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("headline") or raw.get("message") or raw.get("title") or "").strip()
        return {
            "id": str(raw.get("id") or f"notif_{index}_{uuid.uuid4().hex[:8]}"),
            "type": str(raw.get("type") or "news").lower(),
            "priority": str(raw.get("priority") or "LOW").upper(),
            "title": str(raw.get("title") or raw.get("headline") or "").strip(),
            "text": text,
            "date": raw.get("date") or raw.get("calendar_iso") or "",
            "team_id": str(raw.get("team_id") or raw.get("team") or "").strip(),
            "source": str(raw.get("source") or "franchise").strip(),
        }
    text = str(raw or "").strip()
    return {
        "id": f"notif_{index}_{uuid.uuid4().hex[:8]}",
        "type": "news",
        "priority": "LOW",
        "title": "",
        "text": text,
        "date": "",
        "team_id": "",
        "source": "franchise",
    }
def _build_injuries_payload(session: FranchiseSession, *, limit: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    utid = str(getattr(session, "user_team_id", "") or "")
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid_s = str(tid)
        for p in getattr(tm, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            if not _is_player_live_injured(p):
                continue
            rows.append(_build_active_injury_row(session, p, tm, tid_s))

    def _sort_key(r: Dict[str, Any]) -> Tuple[int, int, str]:
        user_first = 0 if utid and str(r.get("team_id") or "") == utid else 1
        gr = -int(r.get("games_remaining") or 0)
        name = str(r.get("player_name") or "")
        return (user_first, gr, name)

    rows.sort(key=_sort_key)
    return rows[: int(limit)]
def _build_injury_history_payload(session: FranchiseSession, *, limit: int = 80) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for inj in list(getattr(session, "injury_log_all", None) or [])[-int(limit) :]:
        if not isinstance(inj, dict):
            continue
        row = dict(inj)
        row["historical"] = True
        row["source"] = "injury_log"
        tid = str(inj.get("team_id") or "")
        pid = str(inj.get("player_id") or "")
        d = inj.get("date", inj.get("calendar_day", "unk"))
        row.setdefault("id", str(inj.get("id") or f"injlog:{tid}:{pid}:{d}"))
        ta = str(inj.get("team_abbr") or inj.get("team_abbrev") or "")
        row["team_abbr"] = ta
        row["team_abbrev"] = ta
        out.append(row)
    return out
def list_teams_summary() -> List[Dict[str, str]]:
    """Lightweight listing for setup UI (bootstraps engine, then throws away)."""
    ensure_simengine_path()
    from app.sim_engine.engine import SimEngine

    sim = SimEngine(seed=1, debug=False)
    teams = list(getattr(sim.league, "teams", None) or [])
    out: List[Dict[str, str]] = []
    for t in teams:
        raw = getattr(t, "team_id", None)
        tid = str(raw) if raw is not None else str(rs._team_id(t))
        out.append({"team_id": tid, "name": _display_team(t)})
    out.sort(key=lambda x: x["name"])
    return out
def _user_team_record_from_game_results(session: FranchiseSession) -> Dict[str, Any]:
    """
    User W-L-OTL and points derived only from game_results (same source as the calendar).
    Matches NHL-style counting: OTL is not a regulation loss; OT loss earns 1 point.
    """
    uid = str(session.user_team_id or "")
    out: Dict[str, Any] = {"gp": 0, "w": 0, "l": 0, "otl": 0, "t": 0, "pts": 0}
    if not uid:
        return out
    for g in getattr(session, "game_results", None) or []:
        hid = str(g.get("home_id") or "")
        aid = str(g.get("away_id") or "")
        if uid not in (hid, aid):
            continue
        try:
            hg = int(g.get("home_goals") or 0)
            ag = int(g.get("away_goals") or 0)
        except (TypeError, ValueError):
            continue
        if hg == ag:
            continue
        ot = bool(g.get("overtime"))
        is_home = hid == uid
        us = hg if is_home else ag
        them = ag if is_home else hg
        if us > them:
            out["w"] += 1
            out["pts"] += 2
        elif ot and us < them:
            out["otl"] += 1
            out["pts"] += 1
        else:
            out["l"] += 1
        out["gp"] += 1
    return out
def get_franchise_game_detail(session: FranchiseSession, game_id: str) -> Optional[Dict[str, Any]]:
    """Return a single saved box score (goals, skater lines, team shot totals) for the recap UI."""
    gid = str(game_id or "").strip()
    if not gid:
        return None
    for g in reversed(getattr(session, "game_results", None) or []):
        if str(g.get("game_id") or "").strip() == gid or _stable_franchise_game_id(g) == gid:
            out = dict(g)
            out["user_team_id"] = str(session.user_team_id)
            return out
    return None



# ---------------------------------------------------------------------------
# Box scores + player game logs
#
# Each finished NHL game keeps one compact box copied from data the sim already
# produced (final, period goals, scoring events, per-player ledger deltas).
# Stored on ``session.box_score_book`` (pickled with the session) and bounded
# to the current + previous season.
# ---------------------------------------------------------------------------

# Wire order of each ``lines`` row after team_id — mirrors engine._BOX_LINE_KEYS.
BOX_LINE_KEYS = (
    "gp", "g", "a", "plus_minus", "pim", "sog", "hit", "blk", "toi_sec",
    "ppg", "ppa", "shg", "sha", "pp_toi_sec", "pk_toi_sec", "cf", "ca", "ixg",
    "saves", "shots_against", "ga", "w", "l", "otl", "t", "so",
    # Stamped at store time (not tracked by either game sim): faceoff wins/losses, takeaways, giveaways.
    "fow", "fol", "tk", "gv",
)
_BOX_IDX = {k: i + 1 for i, k in enumerate(BOX_LINE_KEYS)}
_BOX_ENGINE_LEN = 1 + BOX_LINE_KEYS.index("so") + 1  # team_id + engine-produced keys
_BOX_SEASONS_KEPT = 2


def _box_book(session: Any) -> Dict[str, Any]:
    book = getattr(session, "box_score_book", None)
    if not isinstance(book, dict):
        book = {"seasons": {}, "order": []}
        try:
            session.box_score_book = book
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    book.setdefault("seasons", {})
    book.setdefault("order", [])
    return book


def _box_season_bucket(book: Dict[str, Any], season: int) -> Dict[str, Any]:
    key = str(int(season or 0))
    bucket = book["seasons"].get(key)
    if bucket is None:
        bucket = {"games": {}, "players": {}, "people": {}}
        book["seasons"][key] = bucket
        book["order"].append(key)
        while len(book["order"]) > _BOX_SEASONS_KEPT:
            old = book["order"].pop(0)
            book["seasons"].pop(old, None)
    return bucket


def _box_shot_split(gid: str, total: int, periods: int, goals: Optional[List[int]] = None) -> List[int]:
    """Spread a team's SOG over the periods played (stable per game; no sim).

    Each period gets at least as many shots as goals scored in it (when the total allows).
    """
    total = max(0, int(total or 0))
    if periods <= 0:
        return []
    base = [0.32, 0.34, 0.34] + ([0.07] if periods > 3 else [])
    rng = random.Random(_stable_hash(f"shots|{gid}"))
    weights = [w * (0.7 + 0.6 * rng.random()) for w in base[:periods]]
    tot_w = sum(weights) or 1.0
    out = [int(total * w / tot_w) for w in weights]
    i = rng.randrange(len(out))
    while sum(out) < total:
        out[i % len(out)] += 1
        i += 1
    need = [max(0, int(x or 0)) for x in (goals or [])][:periods]
    need += [0] * (periods - len(need))
    if sum(need) <= total:
        for p in range(periods):
            while out[p] < need[p]:
                donor = max(range(periods), key=lambda q: out[q] - need[q])
                if out[donor] - need[donor] <= 0:
                    break
                out[donor] -= 1
                out[p] += 1
    return out


def _box_fmt_clock(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def _box_light_timeline(
    gid: str,
    hid: str,
    aid: str,
    hp: List[int],
    ap: List[int],
    lines: Dict[str, List[Any]],
    first_tid: str,
    playoff: bool,
    shootout: bool,
) -> List[List[Any]]:
    """Goal timeline for a bulk-sim game from what it recorded (no sim).

    Period goal counts come from the period line; scorers, PP/SH strength and
    assisters come from each player's ledger deltas, so the goal list and the
    player lines agree. Clocks and within-period order are seeded on the game id.
    """
    rng = random.Random(_stable_hash(f"timeline|{gid}"))
    gi, ai = _BOX_IDX["g"], _BOX_IDX["a"]
    ppgi, shgi, ppai, shai = _BOX_IDX["ppg"], _BOX_IDX["shg"], _BOX_IDX["ppa"], _BOX_IDX["sha"]
    scorers: Dict[str, List[Tuple[str, str]]] = {hid: [], aid: []}
    pools: Dict[str, Dict[str, Dict[str, int]]] = {hid: {"PP": {}, "SH": {}, "EV": {}}, aid: {"PP": {}, "SH": {}, "EV": {}}}
    for pid in sorted(lines):
        row = lines[pid]
        tid = str(row[0]) if row else ""
        if tid not in scorers:
            continue
        try:
            g, a = int(row[gi] or 0), int(row[ai] or 0)
            ppg, shg = min(g, int(row[ppgi] or 0)), int(row[shgi] or 0)
            ppa, sha = min(a, int(row[ppai] or 0)), int(row[shai] or 0)
        except (IndexError, TypeError, ValueError):
            continue
        shg = min(shg, g - ppg)
        sha = min(sha, a - ppa)
        scorers[tid].extend([(pid, "PP")] * ppg + [(pid, "SH")] * shg + [(pid, "EV")] * (g - ppg - shg))
        for st, n in (("PP", ppa), ("SH", sha), ("EV", a - ppa - sha)):
            if n > 0:
                pools[tid][st][pid] = n
    for tid in scorers:
        rng.shuffle(scorers[tid])
    goals: List[List[Any]] = []
    nper = len(hp)
    for p in range(nper):
        toks = [hid] * int(hp[p] if p < len(hp) else 0) + [aid] * int(ap[p] if p < len(ap) else 0)
        if not toks:
            continue
        rng.shuffle(toks)
        if p == 0 and first_tid in toks and toks[0] != first_tid:
            j = toks.index(first_tid)
            toks[0], toks[j] = toks[j], toks[0]
        ot = p >= 3
        span = (1200 if playoff else 300) if ot else 1200
        if ot:
            toks = toks[:1]
        secs = sorted(rng.sample(range(8, span - 2), len(toks)))
        for tid, sec in zip(toks, secs):
            if not scorers[tid]:
                continue  # shootout winner / unrecorded goal: no scorer, not a goal event
            pid, st = scorers[tid].pop()
            goals.append([p + 1, _box_fmt_clock(sec), tid, pid, [], st, 0])
    if shootout:
        goals = [g for g in goals if g[0] <= 3]
    # Assists: two passes (primary, then secondary) over each club's goals, drawing
    # from the matching strength pool of players whose assist deltas went up.
    for _pass in range(2):
        for gl in goals:
            if len(gl[4]) != _pass:
                continue
            pool = pools[gl[2]]["EV" if gl[5] == "EV" else gl[5]]
            cands = [q for q, n in pool.items() if n > 0 and q != gl[3] and q not in gl[4]]
            if not cands:
                continue
            best = max(cands, key=lambda q: (pool[q], rng.random()))
            pool[best] -= 1
            gl[4].append(best)
    # Leftover assists (strength pools that ran dry) go to any open slot on that club.
    for gl in goals:
        while len(gl[4]) < 2:
            left = [(q, st) for st, pool in pools[gl[2]].items() for q, n in pool.items() if n > 0 and q != gl[3] and q not in gl[4]]
            if not left:
                break
            q, st = left[0]
            pools[gl[2]][st][q] -= 1
            gl[4].append(q)
    # A scorer's own leftover assist: swap him onto a teammate's goal and move that
    # goal's assister here, so every assist delta shows up in the goal list.
    for gl in goals:
        if len(gl[4]) >= 2:
            continue
        own = [(st, pool) for st, pool in pools[gl[2]].items() if pool.get(gl[3], 0) > 0]
        if not own:
            continue
        st, pool = own[0]
        for g2 in goals:
            if g2 is gl or g2[2] != gl[2] or g2[3] == gl[3] or gl[3] in g2[4]:
                continue
            xs = [x for x in g2[4] if x != gl[3] and x not in gl[4]]
            if xs:
                g2[4][g2[4].index(xs[0])] = gl[3]
                gl[4].append(xs[0])
                pool[gl[3]] -= 1
                break
    # Empty net: a late, last goal by the winner that doubles a one-goal lead.
    if goals and not shootout:
        last = goals[-1]
        h = sum(1 for g in goals if g[2] == hid)
        a = sum(1 for g in goals if g[2] == aid)
        lead_after = (h - a) if last[2] == hid else (a - h)
        if last[0] == 3 and last[5] == "EV" and lead_after == 2:
            sec = int(last[1].split(":")[0]) * 60 + int(last[1].split(":")[1])
            if sec >= 1050 and rng.random() < 0.6:
                last[6] = 1
    return goals


def _box_stamp_extras(session: Any, gid: str, hid: str, aid: str, lines: Dict[str, List[Any]], people: Dict[str, List[str]]) -> None:
    """Append faceoff wins/losses, takeaways and giveaways to each skater line (seeded; no sim).

    Draws go to centres by TOI and are won by faceoff rating; the two clubs' totals balance.
    """
    rng = random.Random(_stable_hash(f"extras|{gid}"))
    toi_i = _BOX_IDX["toi_sec"]
    sa_i = _BOX_IDX["shots_against"]
    tk_i, gv_i = _BOX_IDX["tk"], _BOX_IDX["gv"]
    fo_rating: Dict[str, float] = {}
    team_by_id = getattr(session, "team_by_id", None) or {}
    for tid in (hid, aid):
        for p in list(getattr(team_by_id.get(tid), "roster", None) or ()):
            r = getattr(p, "ratings", None)
            if isinstance(r, dict) and r.get("def_faceoffs") is not None:
                fo_rating[str(getattr(p, "id", "") or "")] = float(r.get("def_faceoffs") or 74.0)
    side_sk: Dict[str, List[Tuple[str, List[Any]]]] = {hid: [], aid: []}
    for pid in sorted(lines):
        row = lines[pid]
        if len(row) < _BOX_ENGINE_LEN:
            row.extend([0] * (_BOX_ENGINE_LEN - len(row)))
        del row[_BOX_ENGINE_LEN:]
        row.extend([0, 0, 0, 0])
        tid = str(row[0])
        pos = str((people.get(pid) or ["", ""])[1] or "").upper()
        if pos == "G" or int(row[sa_i] or 0) > 0 or tid not in side_sk:
            continue
        side_sk[tid].append((pid, row))
        toi_min = float(row[toi_i] or 0) / 60.0
        u = rng.random()
        x = toi_min * 0.028 * (0.35 + 1.15 * u)
        y = toi_min * 0.034 * (1.35 - 0.9 * u)
        row[tk_i] = int(x + (u * 7.31) % 1.0)
        row[gv_i] = int(y + (u * 3.17) % 1.0)
    takers: Dict[str, List[Tuple[str, List[Any], float]]] = {}
    for tid, sk in side_sk.items():
        cs = [(pid, row) for pid, row in sk if str((people.get(pid) or ["", ""])[1] or "").upper().startswith("C")]
        if len(cs) < 2:
            fw = [(pid, row) for pid, row in sk if str((people.get(pid) or ["", ""])[1] or "").upper() not in ("D", "LD", "RD")]
            fw.sort(key=lambda t: -float(t[1][toi_i] or 0))
            cs = (cs + [t for t in fw if t not in cs])[:4]
        takers[tid] = [(pid, row, fo_rating.get(pid, 74.0)) for pid, row in cs if float(row[toi_i] or 0) > 0]
    if not takers.get(hid) or not takers.get(aid):
        return
    n = 48 + rng.randrange(19)

    def _split(total: int, weights: List[float], caps: Optional[List[int]] = None) -> List[int]:
        tw = sum(weights) or 1.0
        raw = [total * w / tw for w in weights]
        out = [int(x) for x in raw]
        if caps:
            out = [min(o, c) for o, c in zip(out, caps)]
        order = sorted(range(len(raw)), key=lambda i: -(raw[i] - int(raw[i])))
        k = 0
        while sum(out) < total and k < 4 * len(out) + total:
            i = order[k % len(order)]
            if not caps or out[i] < caps[i]:
                out[i] += 1
            k += 1
        return out

    def _skill(tid: str) -> float:
        t = takers[tid]
        tw = sum(float(r[toi_i] or 0) for _p, r, _f in t) or 1.0
        return sum(f * float(r[toi_i] or 0) for _p, r, f in t) / tw

    p_home = 0.5 + (_skill(hid) - _skill(aid)) * 0.004 + rng.gauss(0.0, 0.05)
    p_home = min(0.72, max(0.28, p_home))
    h_wins = int(round(n * p_home))
    for tid, wins in ((hid, h_wins), (aid, n - h_wins)):
        t = takers[tid]
        draws = _split(n, [float(r[toi_i] or 0) for _p, r, _f in t])
        won = _split(wins, [d * max(5.0, f) ** 2 for d, (_p, _r, f) in zip(draws, t)], caps=draws)
        for (pid, row, _f), d, w in zip(t, draws, won):
            row[_BOX_IDX["fow"]] = w
            row[_BOX_IDX["fol"]] = d - w


def _unit_ice_minutes(raw: Any) -> Optional[List[float]]:
    """Even-strength unit clock, minutes: L1 L2 L3 L4 P1 P2 P3."""
    if not isinstance(raw, dict) or not raw:
        return None
    keys = ("F0", "F1", "F2", "F3", "D0", "D1", "D2")
    vals = [round(float(raw.get(k) or 0.0), 2) for k in keys]
    return vals if any(v > 0 for v in vals) else None


def _box_three_stars(lines: Dict[str, List[Any]], gwg: str) -> List[str]:
    gi, ai, pmi, sogi = _BOX_IDX["g"], _BOX_IDX["a"], _BOX_IDX["plus_minus"], _BOX_IDX["sog"]
    svi, gai, sai, hi, bi = _BOX_IDX["saves"], _BOX_IDX["ga"], _BOX_IDX["shots_against"], _BOX_IDX["hit"], _BOX_IDX["blk"]
    scored: List[Tuple[float, str]] = []
    for pid, row in lines.items():
        try:
            sa = float(row[sai] or 0)
            if sa > 0:
                ga = float(row[gai] or 0)
                score = float(row[svi] or 0) * 0.16 + (2.2 if ga == 0 else 0.0) - ga * 0.7
            else:
                score = (
                    float(row[gi] or 0) * 3.0 + float(row[ai] or 0) * 1.9 + float(row[pmi] or 0) * 0.35
                    + float(row[sogi] or 0) * 0.12 + (float(row[hi] or 0) + float(row[bi] or 0)) * 0.05
                )
            if pid == gwg:
                score += 0.8
        except (IndexError, TypeError, ValueError):
            continue
        scored.append((score, pid))
    scored.sort(key=lambda t: -t[0])
    return [pid for _s, pid in scored[:3]]


def store_compact_box_score(session: Any, box: Dict[str, Any], lines: Optional[Dict[str, List[Any]]]) -> Optional[Dict[str, Any]]:
    """Copy one finished game into the season box-score book. Returns the compact box."""
    if not isinstance(box, dict):
        return None
    gid = str(box.get("game_id") or box.get("id") or "").strip()
    if not gid:
        return None
    season = int(getattr(session, "season_calendar_year", 0) or box.get("season_calendar_year") or 0)
    bucket = _box_season_bucket(_box_book(session), season)
    hid = str(box.get("home_id") or "")
    aid = str(box.get("away_id") or "")
    hg = int(box.get("home_goals", box.get("home_score", 0)) or 0)
    ag = int(box.get("away_goals", box.get("away_score", 0)) or 0)
    ot = bool(box.get("overtime") or box.get("ot"))
    so = bool(box.get("shootout"))
    hp = [int(box.get(f"home_p{i}", 0) or 0) for i in (1, 2, 3)]
    ap = [int(box.get(f"away_p{i}", 0) or 0) for i in (1, 2, 3)]
    if ot:
        hp.append(int(box.get("home_ot_goals", 0) or 0))
        ap.append(int(box.get("away_ot_goals", 0) or 0))
    nper = len(hp)
    hs = int(box.get("home_sog", box.get("home_shots", 0)) or 0)
    as_ = int(box.get("away_sog", box.get("away_shots", 0)) or 0)
    goals: List[List[Any]] = []
    for ev in list(box.get("scoring_events") or []):
        if not isinstance(ev, dict):
            continue
        goals.append([
            int(ev.get("period") or 0),
            str(ev.get("clock") or ""),
            str(ev.get("for_team_id") or ""),
            str(ev.get("scorer_id") or ""),
            [str(x) for x in list(ev.get("assist_ids") or []) if x],
            str(ev.get("strength") or "EV"),
            1 if ev.get("empty_net") else 0,
        ])
    lines = dict(lines or {})
    people = bucket["people"]
    ledger = getattr(session, "player_season_stats", None) or {}
    for pid in lines:
        if pid not in people:
            row = ledger.get(pid) or {}
            people[pid] = [str(row.get("name") or "Player"), str(row.get("position") or "")]
    scope = str(box.get("stat_scope") or "regular_season")
    if not goals and lines and (hg or ag):
        goals = _box_light_timeline(
            gid, hid, aid, hp, ap, lines, str(box.get("first_goal_team_id") or ""),
            "playoff" in scope, so,
        )
    _box_stamp_extras(session, gid, hid, aid, lines, people)
    # Game-winning goal: the winner's (loser_goals + 1)th goal in recorded order.
    gwg = ""
    if goals and hg != ag and not so:
        win_tid = hid if hg > ag else aid
        need = min(hg, ag) + 1
        n = 0
        for gl in goals:
            if gl[2] == win_tid:
                n += 1
                if n == need:
                    gwg = gl[3]
                    break
    compact = {
        "id": gid,
        "season": season,
        "day": int(box.get("calendar_day", box.get("day", 0)) or 0),
        "iso": str(box.get("calendar_iso") or box.get("iso") or ""),
        "scope": scope,
        "home": hid,
        "away": aid,
        "hn": str(box.get("home_name") or ""),
        "an": str(box.get("away_name") or ""),
        "hg": hg,
        "ag": ag,
        "ot": ot,
        "so": so,
        "light": bool(box.get("light_box")),
        "periods": [hp, ap],
        "shots": [_box_shot_split(gid + "h", hs, nper, hp), _box_shot_split(gid + "a", as_, nper, ap)],
        "pp": [
            [int(box.get("home_pp_goals", 0) or 0), int(box.get("home_ppo", 0) or 0)],
            [int(box.get("away_pp_goals", 0) or 0), int(box.get("away_ppo", 0) or 0)],
        ],
        "pim": [int(box.get("home_pim", 0) or 0), int(box.get("away_pim", 0) or 0)],
        "cf": [
            int(box.get("home_shot_attempts", box.get("home_cf", 0)) or 0),
            int(box.get("away_shot_attempts", box.get("away_cf", 0)) or 0),
        ],
        "xg": [round(float(box.get("home_xgf", 0) or 0), 2), round(float(box.get("away_xgf", 0) or 0), 2)],
        "goals": goals,
        "lines": lines,
        "gwg": gwg,
        "stars": _box_three_stars(lines, gwg),
        "ice": [
            _unit_ice_minutes(box.get("away_unit_toi")),
            _unit_ice_minutes(box.get("home_unit_toi")),
        ],
    }
    bucket["games"][gid] = compact
    pidx = bucket["players"]
    for pid in lines:
        pidx.setdefault(pid, []).append(gid)
    return compact


def _find_compact_box(session: Any, game_id: str) -> Optional[Dict[str, Any]]:
    book = getattr(session, "box_score_book", None)
    if not isinstance(book, dict):
        return None
    for key in reversed(list(book.get("order") or [])):
        g = (((book.get("seasons") or {}).get(key) or {}).get("games") or {}).get(str(game_id))
        if g:
            return g
    return None


def _box_people(session: Any, season: int) -> Dict[str, List[str]]:
    book = getattr(session, "box_score_book", None) or {}
    return ((book.get("seasons") or {}).get(str(int(season or 0))) or {}).get("people") or {}


def _box_line_dict(row: List[Any]) -> Dict[str, Any]:
    out = {k: (row[i] if i < len(row) else 0) for k, i in _BOX_IDX.items()}
    out["team_id"] = str(row[0]) if row else ""
    return out


def expand_box_score(session: Any, compact: Dict[str, Any]) -> Dict[str, Any]:
    """Full box-score payload for the UI from a compact stored box."""
    people = _box_people(session, int(compact.get("season") or 0))
    team_by_id = getattr(session, "team_by_id", None) or {}

    def _nm(pid: str) -> str:
        return str((people.get(pid) or ["Player"])[0])

    def _pos(pid: str) -> str:
        p = people.get(pid) or ["", ""]
        return str(p[1] if len(p) > 1 else "")

    hid, aid = str(compact.get("home")), str(compact.get("away"))
    hg, ag = int(compact.get("hg") or 0), int(compact.get("ag") or 0)
    skaters: Dict[str, List[Dict[str, Any]]] = {"home": [], "away": []}
    goalies: Dict[str, List[Dict[str, Any]]] = {"home": [], "away": []}
    team_tot = {side: {"hits": 0, "blocks": 0, "fow": 0, "fol": 0, "tk": 0, "gv": 0} for side in ("home", "away")}
    for pid, row in (compact.get("lines") or {}).items():
        d = _box_line_dict(row)
        side = "home" if d["team_id"] == hid else "away"
        pos = _pos(pid)
        if pos.upper() == "G" or int(d.get("shots_against") or 0) > 0:
            sa = int(d.get("shots_against") or 0)
            sv = int(d.get("saves") or 0)
            dec = "W" if d.get("w") else ("OTL" if d.get("otl") else ("L" if d.get("l") else ""))
            goalies[side].append({
                "player_id": pid, "name": _nm(pid), "sa": sa, "sv": sv, "ga": int(d.get("ga") or 0),
                "sv_pct": round(sv / sa, 3) if sa else None, "toi_sec": int(d.get("toi_sec") or 0),
                "decision": dec, "shutout": bool(d.get("so")),
            })
            continue
        team_tot[side]["hits"] += int(d.get("hit") or 0)
        team_tot[side]["blocks"] += int(d.get("blk") or 0)
        fow, fol = int(d.get("fow") or 0), int(d.get("fol") or 0)
        for k in ("fow", "fol", "tk", "gv"):
            team_tot[side][k] += int(d.get(k) or 0)
        cf, ca = float(d.get("cf") or 0), float(d.get("ca") or 0)
        g, a = int(d["g"] or 0), int(d["a"] or 0)
        skaters[side].append({
            "player_id": pid, "name": _nm(pid), "pos": pos,
            "g": g, "a": a, "pts": g + a,
            "pm": int(d["plus_minus"] or 0), "pim": int(d["pim"] or 0), "sog": int(d["sog"] or 0),
            "toi_sec": int(d["toi_sec"] or 0), "pp_toi_sec": int(d["pp_toi_sec"] or 0), "pk_toi_sec": int(d["pk_toi_sec"] or 0),
            "hit": int(d["hit"] or 0), "blk": int(d["blk"] or 0),
            "ppg": int(d["ppg"] or 0), "shg": int(d["shg"] or 0),
            "cf_pct": round(cf / (cf + ca), 3) if (cf + ca) > 0 else None,
            "ixg": round(float(d.get("ixg") or 0), 2),
            "fow": fow, "fol": fol, "fo_pct": round(fow / (fow + fol), 3) if (fow + fol) else None,
            "tk": int(d.get("tk") or 0), "gv": int(d.get("gv") or 0),
        })
    for side in ("home", "away"):
        skaters[side].sort(key=lambda r: (-r["pts"], -r["g"], -r["toi_sec"]))
        goalies[side].sort(key=lambda r: -r["toi_sec"])
    goals_out: List[Dict[str, Any]] = []
    run_h = run_a = 0
    gwg_tid = (hid if hg > ag else aid) if (hg != ag and not compact.get("so")) else ""
    gwg_n = min(hg, ag) + 1
    goal_tags = compact.get("goal_tags") or {}
    for gi, (per, clock, tid, scorer, assists, strength, en) in enumerate(compact.get("goals") or []):
        if tid == hid:
            run_h += 1
        else:
            run_a += 1
        is_gwg = bool(gwg_tid and tid == gwg_tid and (run_h if tid == hid else run_a) == gwg_n)
        goals_out.append({
            "period": per, "clock": clock, "team_id": tid,
            "scorer": {"id": scorer, "name": _nm(scorer)},
            "assists": [{"id": a, "name": _nm(a)} for a in assists],
            "strength": strength, "empty_net": bool(en), "overtime": int(per or 0) >= 4,
            "gwg": is_gwg, "score": f"{run_a}-{run_h}",
            # Goofy tag: fluke_ref_skate / fluke_broken_stick / fluke_boards / goalie_goal / penalty_shot
            "tag": str(goal_tags.get(str(gi)) or ""),
        })
    if not goals_out:
        # Light (bulk) games keep who scored but not when: list scorers per club.
        for side, tid in (("away", aid), ("home", hid)):
            for r in skaters[side]:
                for i in range(int(r["g"])):
                    goals_out.append({
                        "period": None, "clock": "", "team_id": tid,
                        "scorer": {"id": r["player_id"], "name": r["name"]}, "assists": [],
                        "strength": "PP" if i < r["ppg"] else ("SH" if i < r["ppg"] + r["shg"] else "EV"),
                        "empty_net": False, "overtime": False, "gwg": False, "score": "",
                    })

    def _team(side: str, tid: str, idx: int) -> Dict[str, Any]:
        tm = team_by_id.get(tid)
        shots = list((compact.get("shots") or [[], []])[idx] or [])
        pp = (compact.get("pp") or [[0, 0], [0, 0]])[idx]
        return {
            "id": tid,
            "name": str(compact.get("hn" if side == "home" else "an") or (_display_team(tm) if tm is not None else tid)),
            "abbr": _franchise_team_abbrev(tm) if tm is not None else "",
            "goals": hg if side == "home" else ag,
            "periods": list((compact.get("periods") or [[], []])[idx] or []),
            "shots": shots,
            "sog": sum(shots),
            "pp_goals": int(pp[0]),
            "ppo": int(pp[1]),
            "pim": int((compact.get("pim") or [0, 0])[idx]),
            "shot_attempts": int((compact.get("cf") or [0, 0])[idx]),
            "xg": float((compact.get("xg") or [0, 0])[idx]),
            "hits": team_tot[side]["hits"],
            "blocks": team_tot[side]["blocks"],
            "faceoffs": (
                {"won": team_tot[side]["fow"], "lost": team_tot[side]["fol"],
                 "pct": round(team_tot[side]["fow"] / (team_tot[side]["fow"] + team_tot[side]["fol"]), 3)}
                if (team_tot[side]["fow"] + team_tot[side]["fol"]) else None
            ),
            "takeaways": team_tot[side]["tk"],
            "giveaways": team_tot[side]["gv"],
            "unit_toi": (compact.get("ice") or [None, None])[idx],
        }

    nper = len(((compact.get("periods") or [[]])[0]) or [])
    star_rows = []
    all_rows = {r["player_id"]: (r, side) for side in ("home", "away") for r in skaters[side] + goalies[side]}
    for rank, pid in enumerate(compact.get("stars") or [], start=1):
        r, side = all_rows.get(pid, ({}, ""))
        if "sa" in r:
            line = f"{r.get('sv')} saves on {r.get('sa')} shots"
        else:
            line = f"{r.get('g', 0)}G {r.get('a', 0)}A, {r.get('sog', 0)} SOG"
        star_rows.append({
            "rank": rank, "player_id": pid, "name": _nm(pid),
            "team_id": hid if side == "home" else aid, "line": line,
        })
    return {
        "game_id": compact.get("id"),
        "season": compact.get("season"),
        "date": compact.get("iso"),
        "day": compact.get("day"),
        "scope": compact.get("scope"),
        "overtime": bool(compact.get("ot")),
        "shootout": bool(compact.get("so")),
        "event_detail": bool(compact.get("goals")) or not bool(compact.get("light")) or not (hg or ag),
        "period_labels": ["1st", "2nd", "3rd", "OT"][:nper] if nper else ["1st", "2nd", "3rd"],
        "home": _team("home", hid, 0),
        "away": _team("away", aid, 1),
        "goals": goals_out,
        "skaters": skaters,
        "goalies": goalies,
        "stars": star_rows,
        # Goofy on-ice moments (hat-trick rain, mascot race, brawl, EBUG, ...).
        "fun_events": list(compact.get("fun") or []),
        "user_team_id": str(getattr(session, "user_team_id", "") or ""),
    }


def get_box_score_payload(session: Any, game_id: str) -> Optional[Dict[str, Any]]:
    compact = _find_compact_box(session, str(game_id or "").strip())
    return expand_box_score(session, compact) if compact else None


def _player_game_row(compact: Dict[str, Any], pid: str) -> Optional[Dict[str, Any]]:
    row = (compact.get("lines") or {}).get(pid)
    if not row:
        return None
    d = _box_line_dict(row)
    home = d["team_id"] == str(compact.get("home"))
    gf = int((compact.get("hg") if home else compact.get("ag")) or 0)
    ga = int((compact.get("ag") if home else compact.get("hg")) or 0)
    res = "W" if gf > ga else ("OTL" if (compact.get("ot") or compact.get("so")) else "L")
    suffix = " (SO)" if compact.get("so") else (" (OT)" if compact.get("ot") else "")
    cf, ca = float(d.get("cf") or 0), float(d.get("ca") or 0)
    g, a = int(d["g"] or 0), int(d["a"] or 0)
    stars = list(compact.get("stars") or [])
    out = {
        "game_id": compact.get("id"),
        "date": compact.get("iso"),
        "scope": compact.get("scope"),
        "home": home,
        "opp_id": str(compact.get("away") if home else compact.get("home")),
        "opp_name": str(compact.get("an") if home else compact.get("hn")),
        "result": f"{res} {gf}-{ga}{suffix}",
        "team_result": res,
        "g": g, "a": a, "pts": g + a,
        "pm": int(d["plus_minus"] or 0), "pim": int(d["pim"] or 0), "sog": int(d["sog"] or 0),
        "hit": int(d["hit"] or 0), "blk": int(d["blk"] or 0), "toi_sec": int(d["toi_sec"] or 0),
        "pp_toi_sec": int(d["pp_toi_sec"] or 0), "pk_toi_sec": int(d["pk_toi_sec"] or 0),
        "ppg": int(d["ppg"] or 0), "ppa": int(d["ppa"] or 0), "shg": int(d["shg"] or 0), "sha": int(d["sha"] or 0),
        "cf": round(cf, 1), "ca": round(ca, 1), "cf_pct": round(cf / (cf + ca), 3) if (cf + ca) > 0 else None,
        "ixg": round(float(d.get("ixg") or 0), 2),
        "gwg": bool(pid and pid == compact.get("gwg")),
        "star": (stars.index(pid) + 1) if pid in stars else 0,
        "fow": int(d.get("fow") or 0), "fol": int(d.get("fol") or 0),
        "tk": int(d.get("tk") or 0), "gv": int(d.get("gv") or 0),
    }
    sa = int(d.get("shots_against") or 0)
    if sa > 0 or int(d.get("saves") or 0) + int(d.get("ga") or 0) > 0:
        sv = int(d.get("saves") or 0)
        out.update({
            "goalie": True, "sa": sa, "sv": sv, "ga": int(d.get("ga") or 0),
            "sv_pct": round(sv / sa, 3) if sa else None,
            "decision": "W" if d.get("w") else ("OTL" if d.get("otl") else ("L" if d.get("l") else "ND")),
            "shutout": bool(d.get("so")),
        })
    return out


def _game_log_totals(rows: List[Dict[str, Any]], goalie: bool) -> Dict[str, Any]:
    gp = len(rows)
    if goalie:
        sa = sum(int(r.get("sa", 0) or 0) for r in rows)
        sv = sum(int(r.get("sv", 0) or 0) for r in rows)
        ga = sum(int(r.get("ga", 0) or 0) for r in rows)
        toi = sum(int(r.get("toi_sec", 0) or 0) for r in rows)
        starts = [r for r in rows if int(r.get("toi_sec", 0) or 0) >= 1800]
        # Quality start (Hockey Prospectus): SV% >= .913, or >= .885 on 20 shots or fewer.
        qs = sum(
            1 for r in starts
            if r.get("sa") and ((r.get("sv_pct") or 0) >= 0.913 or ((r.get("sv_pct") or 0) >= 0.885 and r.get("sa", 0) <= 20))
        )
        return {
            "gp": gp, "gs": len(starts),
            "w": sum(1 for r in rows if r.get("decision") == "W"),
            "l": sum(1 for r in rows if r.get("decision") == "L"),
            "otl": sum(1 for r in rows if r.get("decision") == "OTL"),
            "sa": sa, "sv": sv, "ga": ga, "sv_pct": round(sv / sa, 3) if sa else None,
            "gaa": round(ga * 3600.0 / toi, 2) if toi else None,
            "so": sum(1 for r in rows if r.get("shutout")),
            "qs": qs, "qs_pct": round(qs / len(starts), 3) if starts else None, "toi_sec": toi,
            "stars": sum(1 for r in rows if r.get("star")),
        }
    keys = ("g", "a", "pts", "pm", "pim", "sog", "hit", "blk", "toi_sec", "pp_toi_sec", "pk_toi_sec", "ppg", "ppa", "shg", "sha", "fow", "fol", "tk", "gv")
    t: Dict[str, Any] = {k: sum(int(r.get(k, 0) or 0) for r in rows) for k in keys}
    cf = sum(float(r.get("cf") or 0) for r in rows)
    ca = sum(float(r.get("ca") or 0) for r in rows)
    longest = cur = 0
    for r in rows:
        cur = cur + 1 if r.get("pts") else 0
        longest = max(longest, cur)
    t.update({
        "gp": gp,
        "ppp": t["ppg"] + t["ppa"], "shp": t["shg"] + t["sha"],
        "evg": t["g"] - t["ppg"] - t["shg"],
        "evp": t["pts"] - t["ppg"] - t["ppa"] - t["shg"] - t["sha"],
        "gwg": sum(1 for r in rows if r.get("gwg")),
        "sh_pct": round(t["g"] / t["sog"], 3) if t["sog"] else None,
        "toi_per_gp": round(t["toi_sec"] / gp) if gp else 0,
        "pp_toi_per_gp": round(t["pp_toi_sec"] / gp) if gp else 0,
        "pk_toi_per_gp": round(t["pk_toi_sec"] / gp) if gp else 0,
        "ev_toi_per_gp": round((t["toi_sec"] - t["pp_toi_sec"] - t["pk_toi_sec"]) / gp) if gp else 0,
        "cf": round(cf, 1), "ca": round(ca, 1), "cf_pct": round(cf / (cf + ca), 3) if (cf + ca) > 0 else None,
        "ixg": round(sum(float(r.get("ixg") or 0) for r in rows), 2),
        "multi_point_games": sum(1 for r in rows if int(r.get("pts", 0) or 0) >= 2),
        "longest_point_streak": longest,
        "p60": round(t["pts"] * 3600.0 / t["toi_sec"], 2) if t["toi_sec"] else None,
        "stars": sum(1 for r in rows if r.get("star")),
        "fo_pct": round(t["fow"] / (t["fow"] + t["fol"]), 3) if (t["fow"] + t["fol"]) else None,
    })
    return t


def get_player_game_log_payload(session: Any, player_id: str, season: Optional[int] = None) -> Dict[str, Any]:
    """One player's per-game lines for a stored season, totals by scope, ledger analytics and WJC history."""
    pid = str(player_id or "").strip()
    book = getattr(session, "box_score_book", None) or {}
    seasons = [int(s) for s in (book.get("order") or [])]
    want = int(season) if season else (seasons[-1] if seasons else int(getattr(session, "season_calendar_year", 0) or 0))
    bucket = (book.get("seasons") or {}).get(str(want)) or {}
    games = bucket.get("games") or {}
    rows: List[Dict[str, Any]] = []
    for gid in (bucket.get("players") or {}).get(pid) or []:
        g = games.get(gid)
        r = _player_game_row(g, pid) if g else None
        if r:
            rows.append(r)
    rows.sort(key=lambda r: (str(r.get("date") or ""), str(r.get("game_id"))))
    pos = str(((bucket.get("people") or {}).get(pid) or ["", ""])[1]).upper()
    goalie = pos == "G" or any(r.get("goalie") for r in rows)
    totals: Dict[str, Dict[str, Any]] = {}
    for scope in sorted({str(r.get("scope") or "regular_season") for r in rows}):
        totals[scope] = _game_log_totals([r for r in rows if str(r.get("scope") or "regular_season") == scope], goalie)
    ledger_row = dict((getattr(session, "player_season_stats", None) or {}).get(pid) or {})
    ledger_keep = (
        "gp", "g", "a", "pts", "sog", "pim", "hit", "blk", "toi_sec", "ev_toi_sec", "pp_toi_sec", "pk_toi_sec",
        "ppg", "ppa", "shg", "sha", "plus_minus", "cf", "ca", "ff", "fa", "xgf", "xga", "ixg", "xa",
        "gf_on", "ga_on", "primary_assists", "secondary_assists", "missed_shots",
        "saves", "shots_against", "ga", "w", "l", "otl", "t", "so", "goalie_xga",
    )
    ledger_out = {k: ledger_row.get(k) for k in ledger_keep if k in ledger_row}
    wjc = [dict(w) for w in list(((getattr(session, "wjc_player_history", None) or {}).get(pid)) or []) if isinstance(w, dict)]
    cur = (getattr(session, "wjc_prospect_tournament_results", None) or {}).get(pid)
    if isinstance(cur, dict) and isinstance(cur.get("wjc_stats"), dict):
        year = int(cur["wjc_stats"].get("year") or 0)
        if not any(int(w.get("year") or 0) == year for w in wjc):
            wjc.append(dict(cur["wjc_stats"]))
    wjc.sort(key=lambda w: int(w.get("year") or 0))
    season_label = f"{want}-{str(int(want) + 1)[-2:]}" if int(want or 0) > 1900 else ""
    return {
        "player_id": pid,
        "season": want,
        "season_label": season_label,
        "seasons": seasons,
        "is_goalie": goalie,
        "games": rows,
        "totals": totals,
        "season_ledger": ledger_out,
        "wjc": wjc,
    }
