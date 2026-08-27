"""Interconnected Trade Stability Score (0–100) and escalation levels.

Player concerns feed component pressures with personality × context interactions.
No single variable should independently trigger a formal trade demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.player_agent_engine import (
    ensure_player_agent,
    get_agent_gm_relationship,
)

STABILITY_STABLE_MIN = 70
STABILITY_ANGST_MIN = 55
STABILITY_APATHY_MIN = 40
STABILITY_ANGER_MIN = 20

CRISIS_DEADLINE_MAX = 360


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_0_100(raw: Any, default: float = 50.0) -> float:
    try:
        v = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        v = default
    if v <= 1.5:
        v *= 100.0
    return _clamp(v, 0.0, 100.0)


def _player_id(player: Any) -> str:
    return str(_get(player, "id", "") or _get(player, "player_id", "") or "")


def _player_ovr(player: Any) -> float:
    raw = _get(player, "ovr", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = 70.0
    try:
        o = float(raw if raw is not None else _get(player, "overall", 70) or 70)
    except Exception:
        o = 70.0
    if o <= 1.5:
        o *= 99.0
    return o


def _player_character_0_100(player: Any) -> int:
    try:
        chapters = player.get_chapter_ratings() if hasattr(player, "get_chapter_ratings") else {}
        if isinstance(chapters, dict) and chapters.get("Character"):
            return int(_clamp(float(chapters["Character"]), 55.0, 99.0))
    except Exception:
        pass
    c = getattr(player, "character", None)
    if c is not None:
        try:
            ci = int(c)
            if 20 <= ci <= 99:
                return ci
        except (TypeError, ValueError):
            pass
    tr = getattr(player, "traits", None)
    if tr is None:
        return 74
    blend = (
        0.22 * float(getattr(tr, "coachability", 0.5))
        + 0.20 * float(getattr(tr, "mental_toughness", 0.5))
        + 0.18 * float(getattr(tr, "work_ethic", 0.5))
        + 0.16 * float(getattr(tr, "leadership", 0.5))
        + 0.14 * float(getattr(tr, "competitiveness", 0.5))
        + 0.10 * (1.0 - float(getattr(tr, "volatility", 0.5)))
    )
    return int(round(_clamp(blend, 0.55, 0.95) * 100.0))


def _player_mental_0_100(player: Any) -> int:
    try:
        chapters = player.get_chapter_ratings() if hasattr(player, "get_chapter_ratings") else {}
        if isinstance(chapters, dict) and chapters.get("Mental"):
            return int(_clamp(float(chapters["Mental"]), 50.0, 99.0))
    except Exception:
        pass
    chem = getattr(player, "chemistry_profile", None) or {}
    if isinstance(chem, dict):
        for key in ("mental", "resilience", "adaptability"):
            if chem.get(key):
                return int(_clamp(float(chem[key]), 50.0, 99.0))
    tr = getattr(player, "traits", None)
    if tr is None:
        return 72
    blend = (
        0.45 * float(getattr(tr, "mental_toughness", 0.5))
        + 0.30 * (1.0 - float(getattr(tr, "volatility", 0.5)))
        + 0.25 * float(getattr(tr, "patience", 0.5))
    )
    return int(round(_clamp(blend, 0.45, 0.98) * 100.0))


def ensure_player_storyline_state(player: Any) -> Dict[str, Any]:
    st = getattr(player, "_franchise_storyline_state", None)
    if not isinstance(st, dict):
        st = {}
        setattr(player, "_franchise_storyline_state", st)
    for k, default in (
        ("trade_attempt_count", 0),
        ("was_recently_shopped", False),
        ("trade_rumor_heat", 0),
        ("gm_trust", 0.72),
        ("career_trade_demand_count", 0),
        ("season_trade_demand_count", 0),
        ("previous_trade_demand_severity", 0),
        ("previous_trade_demand_team", ""),
        ("previous_trade_demand_reason", ""),
        ("broken_promises", 0),
        ("promises_kept", 0),
        ("captaincy_promised", False),
        ("captaincy_stripped", False),
        ("family_satisfaction", 62),
        ("home_stability", 65),
        ("relocation_strain", 0),
        ("media_stress", 0),
        ("stability_concern_days", 0),
        ("stability_warnings_sent", []),
    ):
        st.setdefault(k, default)
    return st


def ensure_trade_stability_state(session: Any) -> Dict[str, Any]:
    book = getattr(session, "trade_stability_state", None)
    if not isinstance(book, dict):
        book = {}
        session.trade_stability_state = book
    return book


@dataclass
class PlayerConcernSnapshot:
    role_satisfaction: float = 50.0
    gm_trust: float = 50.0
    coach_trust: float = 50.0
    winning_satisfaction: float = 50.0
    competitiveness: float = 50.0
    character: float = 74.0
    mental: float = 72.0
    loyalty: float = 50.0
    ego: float = 50.0
    ambition: float = 50.0
    professionalism: float = 74.0
    resilience: float = 50.0
    contract_satisfaction: float = 55.0
    contract_security: float = 55.0
    trade_exposure: float = 0.0
    broken_promises: float = 0.0
    development_satisfaction: float = 55.0
    team_belonging: float = 55.0
    locker_room_relationships: float = 55.0
    leadership_treatment: float = 55.0
    performance_vs_deployment: float = 55.0
    career_stage_pressure: float = 0.0
    nhl_experience: float = 50.0
    has_ntc: bool = False
    family_satisfaction: float = 60.0
    relocation_strain: float = 0.0
    media_stress: float = 0.0
    organizational_direction: float = 55.0
    recent_team_performance: float = 50.0
    previous_trade_demands: int = 0
    agent_patience: float = 0.5
    agent_pressure: float = 0.0
    human_life_pressure: float = 0.0
    pressures: Dict[str, float] = field(default_factory=dict)


def _team_win_pct(session: Any, team: Any) -> float:
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    standings = getattr(session, "standings", None)
    try:
        rec = getattr(standings, "records", None) or {}
        row = rec.get(tid)
        if row is None:
            return 0.50
        wins = int(_get(row, "wins", 0) or 0)
        losses = int(_get(row, "losses", 0) or 0)
        otl = int(_get(row, "ot_losses", 0) or _get(row, "otl", 0) or 0)
        gp = wins + losses + otl
        if gp < 8:
            return 0.50
        pts = wins * 2 + otl
        return pts / max(1, gp * 2)
    except Exception:
        return 0.50


@dataclass
class PlayerDeploymentSnapshot:
    """Real deployment inputs for role / ice-time satisfaction."""

    player_id: str = ""
    gp: int = 0
    pts: float = 0.0
    avg_toi_min: Optional[float] = None
    pp_toi_min_pg: float = 0.0
    pk_toi_min_pg: float = 0.0
    ev_toi_min_pg: float = 0.0
    ev_line_rank: int = 0
    pp_unit: int = 0
    pk_unit: int = 0
    scratched: bool = False
    line_role: str = ""
    stat_source: str = ""
    line_source: str = ""


_EV_LINE_RANK = {"f1": 1, "f2": 2, "f3": 3, "f4": 4, "line1": 1, "line2": 2, "line3": 3, "line4": 4}
_DEF_PAIR_RANK = {"d1": 1, "d2": 2, "d3": 3, "pair1": 1, "pair2": 2, "pair3": 3}
_PP_UNIT_RANK = {"pp1": 1, "pp2": 2}
_PK_UNIT_RANK = {"pk1": 1, "pk2": 2, "pk3": 3}


def _team_id(team: Any) -> str:
    return str(_get(team, "team_id", "") or _get(team, "id", "") or "")


def _lines_unit_payload(session: Any, team: Any, unit_type: str) -> Optional[Any]:
    """Saved Edit Lines payload for this team (user team only)."""
    if session is None or team is None:
        return None
    if _team_id(team) != str(getattr(session, "user_team_id", "") or ""):
        return None
    lines_root = getattr(session, "lines", None)
    if not isinstance(lines_root, dict):
        return None
    block = lines_root.get(unit_type)
    if not isinstance(block, dict):
        return None
    inner = block.get("lines")
    return inner if inner is not None else block


def _line_rank_from_unit(group: str, line_id: str, slot: str) -> int:
    lid = str(line_id or "").lower()
    if group == "forwards":
        return int(_EV_LINE_RANK.get(lid, 4))
    if group == "defense":
        return int(_DEF_PAIR_RANK.get(lid, 3))
    if group == "goalies":
        slot_u = str(slot or "").lower()
        if slot_u in ("starter", "start"):
            return 1
        if slot_u in ("backup", "back"):
            return 2
        return 3
    return 5


def _parse_line_ranks(lines: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(lines, dict):
        return out
    for group in ("forwards", "defense", "goalies"):
        for line in lines.get(group) or []:
            if not isinstance(line, dict):
                continue
            line_id = str(line.get("id") or "")
            for slot, pid in (line.get("slots") or {}).items():
                spid = str(pid or "")
                if spid:
                    out[spid] = _line_rank_from_unit(group, line_id, str(slot))
    return out


def _parse_special_team_units(lines: Any, prefix: str, rank_map: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if isinstance(lines, dict) and (lines.get("forwards") or lines.get("defense") or lines.get("units")):
        units = lines.get("units") or lines.get("forwards") or []
        if isinstance(lines.get("forwards"), list):
            units = lines.get("forwards")
        for unit in units if isinstance(units, list) else []:
            if not isinstance(unit, dict):
                continue
            uid = str(unit.get("id") or "").lower()
            rank = rank_map.get(uid, 0)
            if rank <= 0 and uid.startswith(prefix):
                try:
                    rank = int(uid.replace(prefix, "") or "0")
                except ValueError:
                    rank = 0
            for pid in (unit.get("slots") or {}).values():
                spid = str(pid or "")
                if spid and rank > 0:
                    out[spid] = rank
        return out
    if isinstance(lines, list):
        for unit in lines:
            if not isinstance(unit, dict):
                continue
            uid = str(unit.get("id") or "").lower()
            rank = rank_map.get(uid, 0)
            for pid in (unit.get("slots") or {}).values():
                spid = str(pid or "")
                if spid and rank > 0:
                    out[spid] = rank
    return out


def _season_stat_row(player: Any, session: Any = None) -> Dict[str, Any]:
    """Current-season box — prefers session.player_season_stats authority."""
    pid = _player_id(player)
    if session is not None and pid:
        book = getattr(session, "player_season_stats", None) or {}
        row = book.get(pid)
        if isinstance(row, dict) and row:
            return dict(row)

    raw = getattr(player, "season_stats", None)
    if not isinstance(raw, dict):
        return {}
    if raw.get("gp") is not None or raw.get("toi_sec") is not None:
        return raw
    for value in raw.values():
        if isinstance(value, dict) and (value.get("gp") is not None or value.get("toi_sec") is not None):
            return value
    return raw


def _avg_toi_minutes_from_row(row: Dict[str, Any]) -> Optional[float]:
    gp = int(row.get("gp") or row.get("games") or row.get("games_played") or 0)
    toi_sec = int(row.get("toi_sec") or row.get("toi_total_sec") or row.get("time_on_ice_sec") or 0)
    if toi_sec <= 0 and row.get("toi") is not None:
        try:
            toi_val = float(row.get("toi"))
            if toi_val > 0:
                if toi_val <= 45.0 and gp > 0:
                    return toi_val
                if gp > 0:
                    return toi_val / gp if toi_val > 45.0 else toi_val
        except (TypeError, ValueError):
            pass
    if gp >= 3 and toi_sec > 0:
        return toi_sec / gp / 60.0
    return None


def _player_avg_toi_minutes(player: Any, session: Any = None) -> Optional[float]:
    row = _season_stat_row(player, session)
    return _avg_toi_minutes_from_row(row)


def _project_ev_line_rank(player: Any, team: Any) -> int:
    """Fallback EV line rank from roster OVR sort when no saved lines."""
    roster = list(getattr(team, "roster", None) or [])
    if not roster:
        return 2
    pid = _player_id(player)
    pos = str(
        getattr(player, "position", "")
        or getattr(getattr(player, "identity", None), "position", "")
        or "C"
    ).upper()
    if pos == "G":
        return 1
    if pos in {"D", "LD", "RD", "DEF", "DEFENSE"}:
        defs = sorted(
            [p for p in roster if str(getattr(p, "position", "") or "").upper() in {"D", "LD", "RD", "DEF", "DEFENSE"}],
            key=lambda p: -_player_ovr(p),
        )
        for idx, p in enumerate(defs[:6], start=1):
            if _player_id(p) == pid:
                return min(3, (idx - 1) // 2 + 1)
        return 3
    fw = sorted(
        [
            p
            for p in roster
            if str(getattr(p, "position", "") or "").upper() not in {"D", "LD", "RD", "DEF", "DEFENSE", "G", "GOALIE"}
        ],
        key=lambda p: -_player_ovr(p),
    )
    for idx, p in enumerate(fw[:12], start=1):
        if _player_id(p) == pid:
            return min(4, (idx - 1) // 3 + 1)
    return 4


def resolve_player_deployment(session: Any, player: Any, team: Any) -> PlayerDeploymentSnapshot:
    """Merge franchise stats + saved lines into one deployment snapshot."""
    pid = _player_id(player)
    row = _season_stat_row(player, session)
    gp = int(row.get("gp") or row.get("games") or 0)
    pts = float(row.get("pts") or row.get("points") or (int(row.get("g") or 0) + int(row.get("a") or 0)))
    avg_toi = _avg_toi_minutes_from_row(row)

    pp_toi_sec = int(row.get("pp_toi_sec") or row.get("power_play_toi_sec") or 0)
    pk_toi_sec = int(row.get("pk_toi_sec") or row.get("penalty_kill_toi_sec") or 0)
    ev_toi_sec = int(row.get("ev_toi_sec") or row.get("even_strength_toi_sec") or 0)
    if ev_toi_sec <= 0 and int(row.get("toi_sec") or 0) > 0:
        ev_toi_sec = max(0, int(row.get("toi_sec") or 0) - pp_toi_sec - pk_toi_sec)

    gp_div = max(1, gp)
    pp_min = pp_toi_sec / gp_div / 60.0
    pk_min = pk_toi_sec / gp_div / 60.0
    ev_min = ev_toi_sec / gp_div / 60.0 if ev_toi_sec > 0 else (avg_toi or 0.0) - pp_min - pk_min

    ev_payload = _lines_unit_payload(session, team, "even_strength")
    pp_payload = _lines_unit_payload(session, team, "power_play")
    pk_payload = _lines_unit_payload(session, team, "penalty_kill")

    ev_ranks = _parse_line_ranks(ev_payload) if ev_payload else {}
    pp_units = _parse_special_team_units(pp_payload, "pp", _PP_UNIT_RANK) if pp_payload else {}
    pk_units = _parse_special_team_units(pk_payload, "pk", _PK_UNIT_RANK) if pk_payload else {}

    ev_rank = int(ev_ranks.get(pid) or 0)
    pp_unit = int(pp_units.get(pid) or 0)
    pk_unit = int(pk_units.get(pid) or 0)

    line_source = "none"
    if ev_rank > 0:
        line_source = "session.lines.even_strength"
    elif ev_payload is not None:
        line_source = "session.lines.missing_from_ev"
    else:
        ev_rank = _project_ev_line_rank(player, team)
        line_source = "roster_projection"

    pos = str(getattr(player, "position", "") or "").upper()
    is_goalie = pos in {"G", "GOALIE", "GOALTENDER"}
    roster_ids = {_player_id(p) for p in (getattr(team, "roster", None) or []) if _player_id(p)}
    scratched = (
        not is_goalie
        and pid in roster_ids
        and ev_payload is not None
        and pid not in ev_ranks
        and pid not in pp_units
        and pid not in pk_units
    )

    if scratched:
        line_role = "scratch"
    elif is_goalie:
        line_role = "G1" if ev_rank <= 1 else "G2" if ev_rank == 2 else "G3"
    elif pos in {"D", "LD", "RD", "DEF", "DEFENSE"}:
        line_role = f"D{ev_rank}"
    else:
        line_role = f"L{ev_rank}"

    stat_source = "session.player_season_stats" if session is not None and pid in (getattr(session, "player_season_stats", None) or {}) else ""
    if not stat_source and row:
        stat_source = str(row.get("stat_authority") or "player.season_stats")

    return PlayerDeploymentSnapshot(
        player_id=pid,
        gp=gp,
        pts=pts,
        avg_toi_min=avg_toi,
        pp_toi_min_pg=round(pp_min, 2),
        pk_toi_min_pg=round(pk_min, 2),
        ev_toi_min_pg=round(max(0.0, ev_min), 2),
        ev_line_rank=ev_rank,
        pp_unit=pp_unit,
        pk_unit=pk_unit,
        scratched=scratched,
        line_role=line_role,
        stat_source=stat_source,
        line_source=line_source,
    )


def sync_player_role_from_real_data(session: Any, player: Any, team: Any) -> PlayerDeploymentSnapshot:
    """Attach line role + psych satisfaction from stats/lines onto the live player object."""
    deploy = resolve_player_deployment(session, player, team)
    try:
        setattr(player, "line_role", deploy.line_role)
        setattr(player, "pp_unit", deploy.pp_unit)
        setattr(player, "pk_unit", deploy.pk_unit)
        setattr(player, "_deployment_snapshot", deploy)
        if deploy.scratched:
            setattr(player, "_recently_scratched", True)
    except Exception:
        pass

    satisfaction = infer_role_satisfaction_from_deployment(player, team, session, deploy=deploy)
    if satisfaction is not None:
        sat_norm = round(_clamp(satisfaction / 100.0, 0.0, 1.0), 4)
        psych = getattr(player, "psych", None)
        if psych is not None:
            try:
                setattr(psych, "role_satisfaction", sat_norm)
                setattr(psych, "ice_time_satisfaction", sat_norm)
            except Exception:
                pass
    return deploy


def _expected_toi_from_deployment(
    deploy: PlayerDeploymentSnapshot,
    *,
    ovr: float,
    is_defense: bool,
) -> float:
    if deploy.scratched:
        return 7.5 if deploy.gp > 0 else 0.0

    rank = max(1, int(deploy.ev_line_rank or 4))
    if is_defense:
        base = {1: 23.0, 2: 20.0, 3: 16.5}.get(rank, 14.0)
    else:
        base = {1: 20.5, 2: 17.0, 3: 14.0, 4: 11.0}.get(rank, 9.5)

    if ovr >= 90 and rank >= 3:
        base += 1.5
    elif ovr >= 86 and rank >= 4:
        base += 1.0

    if deploy.pp_unit == 1:
        base += 1.8
    elif deploy.pp_unit == 2:
        base += 0.9
    elif deploy.pp_toi_min_pg >= 1.2:
        base += min(1.5, deploy.pp_toi_min_pg * 0.45)

    if deploy.pk_unit == 1:
        base += 0.6
    elif deploy.pk_toi_min_pg >= 0.8:
        base += 0.4

    return base


def infer_role_satisfaction_from_deployment(
    player: Any,
    team: Any,
    session: Any,
    *,
    deploy: Optional[PlayerDeploymentSnapshot] = None,
) -> Optional[float]:
    """Derive role satisfaction from franchise stats + saved lines (0–100)."""
    deploy = deploy or resolve_player_deployment(session, player, team)
    avg_toi = deploy.avg_toi_min
    if avg_toi is None:
        avg_toi = _player_avg_toi_minutes(player, session)

    ovr = _player_ovr(player)
    pos = str(
        getattr(player, "position", "")
        or getattr(getattr(player, "identity", None), "position", "")
        or "C"
    ).upper()
    is_defense = pos in {"D", "LD", "RD", "DEF", "DEFENSE"}

    if deploy.scratched:
        if ovr >= 80:
            return 22.0
        return 30.0

    expected = _expected_toi_from_deployment(deploy, ovr=ovr, is_defense=is_defense)

    if avg_toi is None:
        if expected > 0:
            rank = max(1, deploy.ev_line_rank or _project_ev_line_rank(player, team))
            if rank >= 4 and ovr >= 82:
                return 36.0
            if rank <= 2 and ovr >= 86:
                return 74.0
            return 52.0
        psych = getattr(player, "psych", None)
        ice = getattr(psych, "ice_time_satisfaction", None) if psych is not None else None
        if ice is not None:
            return _to_0_100(float(ice) * 100.0 if float(ice) <= 1.5 else float(ice))
        return None

    ratio = avg_toi / max(7.5, expected)
    if ratio >= 1.05:
        satisfaction = 88.0 + min(12.0, (ratio - 1.0) * 45.0)
    elif ratio >= 0.92:
        satisfaction = 72.0 + (ratio - 0.92) / 0.13 * 16.0
    elif ratio >= 0.78:
        satisfaction = 52.0 + (ratio - 0.78) / 0.14 * 20.0
    elif ratio >= 0.62:
        satisfaction = 32.0 + (ratio - 0.62) / 0.16 * 20.0
    else:
        satisfaction = max(8.0, ratio / 0.62 * 32.0)

    if deploy.gp < 5 and int(getattr(session, "calendar_cursor", 40) or 40) > 20:
        satisfaction = min(satisfaction, 35.0)

    if deploy.ev_line_rank >= 4 and ovr >= 80:
        satisfaction = min(satisfaction, 38.0)
    elif deploy.ev_line_rank <= 2 and ovr >= 84 and ratio >= 0.9:
        satisfaction = max(satisfaction, 70.0)

    if (
        deploy.pp_unit == 0
        and ovr >= 86
        and not is_defense
        and deploy.ev_line_rank <= 2
        and deploy.line_source.startswith("session.lines")
        and deploy.pp_toi_min_pg < 0.25
    ):
        satisfaction -= 10.0

    return round(_clamp(satisfaction, 0.0, 100.0), 2)


def infer_performance_vs_deployment(
    player: Any,
    team: Any,
    role_satisfaction: float,
    *,
    session: Any = None,
    deploy: Optional[PlayerDeploymentSnapshot] = None,
) -> float:
    """Production (pts/60) relative to minutes — benched producers get extra frustration."""
    deploy = deploy or resolve_player_deployment(session, player, team)
    avg_toi = deploy.avg_toi_min
    if avg_toi is None:
        avg_toi = _player_avg_toi_minutes(player, session)
    ovr = _player_ovr(player)
    if avg_toi is None or ovr < 70:
        return role_satisfaction

    row = _season_stat_row(player, session)
    pts = float(deploy.pts or row.get("pts") or row.get("points") or 0)
    toi_sec = int(row.get("toi_sec") or row.get("toi_total_sec") or 0)
    if toi_sec <= 0 and avg_toi and deploy.gp > 0:
        toi_sec = int(avg_toi * deploy.gp * 60)
    if toi_sec <= 0:
        return role_satisfaction

    pts60 = pts / (toi_sec / 3600.0)
    expected_pts60 = 1.85 if ovr >= 88 else 1.45 if ovr >= 82 else 1.05 if ovr >= 76 else 0.70
    prod_ratio = pts60 / max(0.25, expected_pts60)

    pos = str(getattr(player, "position", "") or "").upper()
    is_defense = pos in {"D", "LD", "RD", "DEF", "DEFENSE"}
    expected = _expected_toi_from_deployment(deploy, ovr=ovr, is_defense=is_defense)

    if prod_ratio >= 1.08 and avg_toi + 1.5 < expected:
        return min(role_satisfaction, 28.0)
    if prod_ratio >= 1.05 and role_satisfaction < 45:
        return min(role_satisfaction, 35.0)
    if ovr >= 84 and role_satisfaction < 45:
        return min(role_satisfaction, 35.0)
    return role_satisfaction


def _infer_captaincy_treatment(player: Any, pst: Dict[str, Any]) -> float:
    cap = str(
        getattr(player, "captaincy", "")
        or getattr(player, "captain_role", "")
        or pst.get("captaincy", "")
        or ""
    ).upper()
    is_c = cap in ("C", "CAPTAIN") or bool(getattr(player, "is_captain", False))
    is_a = cap in ("A", "ALT", "ALTERNATE")
    score = 62.0
    if is_c:
        score = 80.0
    elif is_a:
        score = 72.0
    if bool(pst.get("captaincy_promised")) and not is_c and not is_a:
        score = 38.0
    if bool(pst.get("captaincy_stripped")):
        score = 24.0
    return score


def _infer_contract_satisfaction(player: Any, contract: Any, ovr: float) -> float:
    if contract is None:
        return 55.0
    try:
        aav = float(
            _get(contract, "aav_m", 0)
            or _get(contract, "cap_hit_m", 0)
            or _get(contract, "salary_m", 0)
            or 0
        )
    except (TypeError, ValueError):
        aav = 0.0
    if aav <= 0:
        return 55.0
    expected = max(1.2, float(ovr) * 0.105)
    ratio = aav / expected
    if ratio >= 1.10:
        return 84.0
    if ratio >= 0.95:
        return 70.0
    if ratio >= 0.80:
        return 54.0
    return 36.0


def _infer_family_satisfaction(pst: Dict[str, Any], psych: Dict[str, Any]) -> float:
    base = float(pst.get("family_satisfaction") or pst.get("home_stability") or 62)
    if base <= 1.5:
        base *= 100.0
    stress = float(pst.get("personal_stress") or 0)
    if stress > 0:
        base -= min(18.0, stress * 0.22)
    return _clamp(base, 0.0, 100.0)


def _infer_relocation_strain(pst: Dict[str, Any]) -> float:
    raw = float(pst.get("relocation_strain") or pst.get("relocation_pressure") or 0)
    if raw <= 1.5:
        return _clamp(raw * 100.0, 0.0, 100.0)
    return _clamp(raw, 0.0, 100.0)


def _infer_media_stress(pst: Dict[str, Any], psych: Dict[str, Any], trade_exposure: float) -> float:
    raw = psych.get("media_stress", pst.get("media_stress", 0.28))
    if raw is None:
        raw = 28.0
    ms = float(raw)
    if ms <= 1.5:
        ms *= 100.0
    return _clamp(ms * 0.55 + float(trade_exposure) * 0.45, 0.0, 100.0)


def _infer_organizational_direction(
    winning_sat: float,
    gm_trust: float,
    coach_trust: float,
    team: Any,
) -> float:
    window = str(getattr(team, "gm_window", "") or getattr(team, "window", "") or "").lower()
    score = winning_sat * 0.42 + gm_trust * 0.33 + coach_trust * 0.25
    if "rebuild" in window or "tank" in window or "declin" in window:
        score -= 10.0
    elif "contend" in window or "win" in window:
        score += 5.0
    return _clamp(score, 0.0, 100.0)


def gather_player_concerns(session: Any, player: Any, team: Any) -> PlayerConcernSnapshot:
    from app.sim_engine.systems.chemistry import ensure_player_chemistry_profile, safe_get_psych

    deploy = sync_player_role_from_real_data(session, player, team)
    psych = safe_get_psych(player)
    chem = ensure_player_chemistry_profile(player)
    pst = ensure_player_storyline_state(player)
    agent = ensure_player_agent(player, session)
    gm_rel = get_agent_gm_relationship(session, str(agent.get("id") or ""))

    role = _to_0_100(psych.get("role_satisfaction", 0.5) * 100.0 if psych.get("role_satisfaction", 0.5) <= 1.5 else psych.get("role_satisfaction", 50))
    deployment_role = infer_role_satisfaction_from_deployment(player, team, session, deploy=deploy)
    if deployment_role is not None:
        role = deployment_role
    morale = _to_0_100(psych.get("morale", 0.5) * 100.0 if psych.get("morale", 0.5) <= 1.5 else psych.get("morale", 50))
    conf = _to_0_100(psych.get("confidence", 0.5) * 100.0 if psych.get("confidence", 0.5) <= 1.5 else psych.get("confidence", 50))

    coach_trust_raw = getattr(getattr(player, "psych", None), "coach_trust", None)
    if coach_trust_raw is None:
        coach_trust = (conf + role) / 2.0
    else:
        coach_trust = _to_0_100(coach_trust_raw)

    win_pct = _team_win_pct(session, team)
    winning_sat = _clamp(35.0 + win_pct * 65.0, 0.0, 100.0)

    character = float(_player_character_0_100(player))
    mental = float(_player_mental_0_100(player))
    competitiveness = _to_0_100(chem.get("competitiveness", chem.get("compete", 50)))
    loyalty = _to_0_100(chem.get("loyalty", 50))
    ego = _to_0_100(getattr(getattr(player, "traits", None), "ego", 0.5) * 100.0)
    ambition = _to_0_100(chem.get("ambition", chem.get("drive", competitiveness)))
    resilience = _to_0_100(chem.get("resilience", chem.get("adaptability", mental)))
    belonging = _to_0_100(chem.get("belonging", chem.get("team_player", morale)))

    trade_exposure = min(
        100.0,
        int(pst.get("trade_rumor_heat") or 0) * 0.85 + int(pst.get("trade_attempt_count") or 0) * 8.0,
    )
    if pst.get("was_recently_shopped"):
        trade_exposure = min(100.0, trade_exposure + 12.0)

    gm_trust = _to_0_100(float(pst.get("gm_trust", 0.72)) * 100.0 if float(pst.get("gm_trust", 0.72)) <= 1.5 else pst.get("gm_trust", 72))

    age = int(getattr(getattr(player, "identity", None), "age", None) or getattr(player, "age", 27) or 27)
    ovr = _player_ovr(player)

    contract = getattr(player, "contract", None)
    has_ntc = False
    contract_sat = _infer_contract_satisfaction(player, contract, ovr)
    contract_sec = 55.0
    if contract is not None:
        clause = str(
            _get(contract, "clause", "")
            or _get(contract, "clause_type", "")
            or _get(contract, "trade_clause", "")
            or ""
        ).upper()
        has_ntc = "NTC" in clause or "NMC" in clause or "NO MOVE" in clause or "NO TRADE" in clause
        yrs = int(_get(contract, "years_remaining", 0) or _get(contract, "term", 0) or 0)
        contract_sec = _clamp(40.0 + yrs * 8.0, 0.0, 100.0)

    career_pressure = 0.0
    if age >= 32 and ovr >= 80:
        career_pressure = min(35.0, (age - 31) * 4.0)
    dev_sat = 70.0 if age >= 26 else _clamp(role + (winning_sat - 50) * 0.25, 0.0, 100.0)

    perf_vs_deploy = infer_performance_vs_deployment(player, team, role, session=session, deploy=deploy)

    prev_demands = int(pst.get("career_trade_demand_count") or 0)
    agent_patience = float(agent.get("patience", 0.5) or 0.5) + (float(gm_rel.get("agent_gm_trust", 0.55)) - 0.5) * 0.2

    captaincy_treatment = _infer_captaincy_treatment(player, pst)
    family_sat = _infer_family_satisfaction(pst, psych)
    relocation = _infer_relocation_strain(pst)
    media_stress = _infer_media_stress(pst, psych, trade_exposure)
    org_direction = _infer_organizational_direction(winning_sat, gm_trust, coach_trust, team)
    broken_promises = float(int(pst.get("broken_promises") or 0) * 18.0)
    if bool(pst.get("captaincy_stripped")):
        broken_promises = min(100.0, broken_promises + 12.0)

    human_pressure_score = 0.0
    human_pressure_tier = 0
    human_life_pressure = 0.0
    try:
        from app.sim_engine.franchise.storyline_engine import _u_sync_player_entities  # noqa: WPS433

        pid = str(getattr(player, "id", "") or getattr(player, "player_id", "") or "")
        if pid:
            entities = _u_sync_player_entities(session)
            entity = entities.get(pid) or {}
            life = entity.get("life") or {}
            est = entity.get("state") or {}
            hp = entity.get("human_pressure") or {}
            if life:
                family_sat = _clamp(
                    float(life.get("home_stability") or family_sat) * 0.45
                    + (100.0 - float(life.get("relocation_strain") or relocation)) * 0.25
                    + float(life.get("city_attachment") or 50) * 0.30,
                    0.0,
                    100.0,
                )
                partner = life.get("partner") if isinstance(life.get("partner"), dict) else {}
                if partner:
                    family_sat = _clamp(family_sat * 0.7 + float(partner.get("city_satisfaction") or 55) * 0.3, 0.0, 100.0)
                relocation = _clamp(float(life.get("relocation_strain") or relocation), 0.0, 100.0)
            if est.get("belonging") is not None:
                belonging = _to_0_100(float(est.get("belonging")))
            if est.get("gm_trust") is not None:
                gm_trust = _to_0_100(float(est.get("gm_trust")))
            human_pressure_score = float(hp.get("score") or 0)
            human_pressure_tier = int(hp.get("tier") or 0)
            if human_pressure_tier >= 2:
                human_life_pressure = human_pressure_score * 0.35
            elif human_pressure_tier >= 1:
                human_life_pressure = human_pressure_score * 0.18
    except Exception:
        pass

    snap = PlayerConcernSnapshot(
        role_satisfaction=role,
        gm_trust=gm_trust,
        coach_trust=coach_trust,
        winning_satisfaction=winning_sat,
        competitiveness=competitiveness,
        character=character,
        mental=mental,
        loyalty=loyalty,
        ego=ego,
        ambition=ambition,
        professionalism=character,
        resilience=resilience,
        contract_satisfaction=contract_sat,
        contract_security=contract_sec,
        trade_exposure=trade_exposure,
        broken_promises=broken_promises,
        development_satisfaction=dev_sat,
        team_belonging=belonging,
        locker_room_relationships=belonging,
        leadership_treatment=captaincy_treatment,
        performance_vs_deployment=perf_vs_deploy,
        career_stage_pressure=career_pressure,
        nhl_experience=min(100.0, max(0.0, (age - 18) * 4.5)),
        has_ntc=has_ntc,
        family_satisfaction=family_sat,
        relocation_strain=relocation,
        media_stress=media_stress,
        organizational_direction=org_direction,
        recent_team_performance=winning_sat,
        previous_trade_demands=prev_demands,
        agent_patience=_clamp(agent_patience, 0.05, 0.98),
        agent_pressure=max(0.0, (1.0 - agent_patience) * 20.0),
        human_life_pressure=human_life_pressure,
    )
    return snap


def _dissatisfaction(satisfaction: float, *, neutral_floor: float = 62.0) -> float:
    """Only count dissatisfaction below neutral_floor — stubbed inputs at ~55 won't stack."""
    if satisfaction >= neutral_floor:
        return 0.0
    scale = (neutral_floor - satisfaction) / neutral_floor
    return _clamp(scale * neutral_floor, 0.0, 100.0)


def character_tolerance_bonus(character: float) -> float:
    c = float(character)
    if c >= 92:
        return 14.0
    if c >= 85:
        return 9.0
    if c >= 77:
        return 4.0
    if c >= 70:
        return 0.0
    if c >= 63:
        return -5.0
    if c >= 55:
        return -11.0
    return -16.0


def mental_resilience_bonus(mental: float) -> float:
    m = float(mental)
    if m >= 90:
        return 10.0
    if m >= 80:
        return 6.0
    if m >= 70:
        return 2.0
    if m >= 60:
        return -2.0
    return -8.0


def _role_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.role_satisfaction)
    sens = 0.85 + (snap.ego / 100.0) * 0.45
    if snap.competitiveness >= 75:
        if snap.winning_satisfaction >= 65:
            dissat *= 0.42
        elif snap.winning_satisfaction < 42:
            dissat *= 1.38
    if snap.character < 68:
        dissat *= 1.12
    elif snap.character >= 85:
        dissat *= 0.82
    return dissat * sens * 0.22


def _management_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = (_dissatisfaction(snap.gm_trust) * 0.55 + _dissatisfaction(snap.coach_trust) * 0.45)
    if snap.loyalty >= 72 and snap.gm_trust >= 55:
        dissat *= 0.68
    return dissat * 0.20


def _winning_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.winning_satisfaction)
    weight = 0.12 + (snap.competitiveness / 100.0) * 0.18
    if snap.competitiveness >= 78 and snap.winning_satisfaction < 45:
        dissat *= 1.55
    return dissat * weight


def _contract_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = (
        _dissatisfaction(snap.contract_satisfaction, neutral_floor=60.0) * 0.6
        + _dissatisfaction(snap.contract_security, neutral_floor=52.0) * 0.4
    )
    return dissat * 0.10


def _development_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.development_satisfaction, neutral_floor=58.0)
    if snap.ambition >= 72:
        dissat *= 1.20
    return dissat * 0.08


def _coach_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.coach_trust, neutral_floor=58.0)
    if snap.leadership_treatment < 50:
        dissat = max(dissat, _dissatisfaction(snap.leadership_treatment, neutral_floor=55.0) * 0.45)
    return dissat * 0.10


def _belonging_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = (
        _dissatisfaction(snap.team_belonging, neutral_floor=58.0) * 0.55
        + _dissatisfaction(snap.locker_room_relationships, neutral_floor=58.0) * 0.45
    )
    return dissat * 0.10


def _trade_exposure_pressure(snap: PlayerConcernSnapshot) -> float:
    base = snap.trade_exposure
    if snap.mental >= 88 and snap.character >= 82:
        base *= 0.35
    elif snap.mental < 62:
        base *= 1.45
    elif snap.mental < 70:
        base *= 1.15
    if snap.character < 65:
        base *= 1.22
    return base * 0.18


def _personal_pressure(snap: PlayerConcernSnapshot) -> float:
    family_dissat = _dissatisfaction(snap.family_satisfaction, neutral_floor=62.0)
    dissat = (
        snap.relocation_strain * 0.30
        + snap.media_stress * 0.22
        + family_dissat * 0.28
        + snap.broken_promises * 0.35
    )
    if dissat < 4.0:
        return 0.0
    return dissat * 0.07


def _organizational_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.organizational_direction, neutral_floor=55.0)
    return dissat * 0.08 + snap.career_stage_pressure * 0.05


def _performance_pressure(snap: PlayerConcernSnapshot) -> float:
    dissat = _dissatisfaction(snap.performance_vs_deployment)
    if snap.ego >= 75:
        dissat *= 1.18
    return dissat * 0.12


def compute_component_pressures(snap: PlayerConcernSnapshot) -> Dict[str, float]:
    pressures = {
        "role": _role_pressure(snap),
        "management": _management_pressure(snap),
        "winning": _winning_pressure(snap),
        "contract": _contract_pressure(snap),
        "development": _development_pressure(snap),
        "belonging": _belonging_pressure(snap),
        "trade_exposure": _trade_exposure_pressure(snap),
        "personal": _personal_pressure(snap),
        "coach": _coach_pressure(snap),
        "organizational": _organizational_pressure(snap),
        "performance": _performance_pressure(snap),
    }
    if snap.broken_promises > 0:
        pressures["broken_promise"] = min(28.0, snap.broken_promises * 1.1)
    if snap.previous_trade_demands > 0:
        pressures["demand_history"] = min(18.0, snap.previous_trade_demands * 4.5)
    if float(snap.human_life_pressure or 0) > 0:
        pressures["human_life"] = float(snap.human_life_pressure)
    snap.pressures = pressures
    return pressures


def compute_trade_stability(snap: PlayerConcernSnapshot) -> Tuple[float, Dict[str, float]]:
    pressures = compute_component_pressures(snap)
    cumulative = sum(pressures.values()) + snap.agent_pressure * 0.35

    loyalty_buffer = (snap.loyalty / 100.0) * 6.0
    if snap.loyalty >= 75 and snap.gm_trust >= 58:
        loyalty_buffer += 4.0

    char_buf = character_tolerance_bonus(snap.character)
    mental_buf = mental_resilience_bonus(snap.mental)

    winning_mit = 0.0
    if snap.winning_satisfaction >= 62:
        winning_mit = min(8.0, (snap.winning_satisfaction - 60) * 0.15)

    relationship_mit = 0.0
    if snap.gm_trust >= 65 and snap.coach_trust >= 60:
        relationship_mit = min(7.0, (snap.gm_trust + snap.coach_trust - 120) * 0.08)

    agent_mod = (snap.agent_patience - 0.5) * 6.0

    score = 100.0 - cumulative + loyalty_buffer + char_buf + mental_buf + winning_mit + relationship_mit + agent_mod
    score = _clamp(score, 0.0, 100.0)
    return round(score, 2), pressures


def stability_to_escalation_level(stability: float) -> int:
    s = float(stability)
    if s >= STABILITY_STABLE_MIN:
        return 0
    if s >= STABILITY_ANGST_MIN:
        return 1
    if s >= STABILITY_APATHY_MIN:
        return 2
    if s >= STABILITY_ANGER_MIN:
        return 3
    return 4


def character_daily_drift_multiplier(character: float) -> float:
    """Low character erodes stability faster over time — not instant level skips."""
    c = float(character)
    if c >= 85:
        return 0.75
    if c >= 77:
        return 0.90
    if c >= 70:
        return 1.0
    if c >= 63:
        return 1.15
    if c >= 55:
        return 1.30
    return 1.45


def count_significant_pressures(pressures: Dict[str, float], threshold: float = 7.5) -> int:
    return sum(1 for v in (pressures or {}).values() if float(v or 0) >= threshold)


def formal_demand_eligible(stability_row: Dict[str, Any]) -> bool:
    """Require meaningful multi-signal breakdown before a formal L3+ demand."""
    score = float(stability_row.get("trade_stability_score") or 100.0)
    level = stability_to_escalation_level(score)
    if level < 3:
        return False
    concern_days = int(stability_row.get("stability_concern_days") or 0)
    if score > 22 and concern_days < 21:
        return False
    pressures = dict(stability_row.get("pressures") or {})
    sig = count_significant_pressures(pressures)
    if score <= 22:
        return True
    if sig >= 2:
        return True
    top = max(pressures.values()) if pressures else 0.0
    return top >= 14.0 and score <= 32


def readiness_penalties(stability: float, character: float, mental: float, escalation: int) -> Dict[str, float]:
    """Mental stress + character disengagement (separate channels)."""
    if escalation < 2:
        return {"mental_stress": 0.0, "character_disengagement": 0.0, "ovr_readiness": 0.0}

    stress_base = max(0.0, (55.0 - float(stability)) * 0.06)
    mental_stress = stress_base * _clamp(1.35 - float(mental) / 100.0, 0.15, 1.25)

    disengage_base = max(0.0, (50.0 - float(character)) * 0.05) * (escalation - 1) * 0.35
    if escalation >= 4:
        disengage_base += 2.5
    character_disengagement = disengage_base * _clamp(1.2 - float(mental) / 120.0, 0.2, 1.0)

    ovr = -min(6.0, mental_stress + character_disengagement)
    return {
        "mental_stress": round(mental_stress, 2),
        "character_disengagement": round(character_disengagement, 2),
        "ovr_readiness": round(ovr, 2),
    }


def apply_readiness_to_player(player: Any, penalties: Dict[str, float]) -> None:
    ovr_pen = float(penalties.get("ovr_readiness") or 0.0)
    if ovr_pen >= 0:
        return
    st = ensure_player_storyline_state(player)
    st["trade_demand_readiness_penalty"] = ovr_pen
    try:
        setattr(player, "_trade_demand_readiness_penalty", ovr_pen)
    except Exception:
        pass


def clear_demand_temporary_modifiers(player: Any) -> None:
    st = ensure_player_storyline_state(player)
    for key in ("trade_demand_readiness_penalty",):
        st.pop(key, None)
    for attr in (
        "_trade_demand_readiness_penalty",
        "_trade_demand_active",
        "_systemic_trade_value_mult",
        "_crisis_trade_value_mult",
        "_crisis_distressed_asset",
        "_crisis_trade_stage",
        "locker_room_disruptor",
    ):
        try:
            if hasattr(player, attr):
                delattr(player, attr)
        except Exception:
            try:
                setattr(player, attr, False if attr == "locker_room_disruptor" else None)
            except Exception:
                pass


def compute_instant_stability(session: Any, player: Any, team: Any) -> Dict[str, Any]:
    """Snapshot stability from current concerns — no day drift applied."""
    snap = gather_player_concerns(session, player, team)
    score, pressures = compute_trade_stability(snap)
    escalation = stability_to_escalation_level(score)
    penalties = readiness_penalties(score, snap.character, snap.mental, escalation)
    return {
        "player_id": _player_id(player),
        "trade_stability_score": score,
        "escalation_level": escalation,
        "pressures": {k: round(v, 2) for k, v in pressures.items()},
        "character": int(snap.character),
        "mental": int(snap.mental),
        "readiness_penalties": penalties,
        "snap": snap,
    }


def apply_daily_stability_update(session: Any, player: Any, team: Any, calendar_idx: int) -> Dict[str, Any]:
    """Drift stored stability toward target — weeks/months pacing, not days."""
    instant = compute_instant_stability(session, player, team)
    target_score = float(instant["trade_stability_score"])
    character = float(instant["character"])
    mental = float(instant["mental"])
    drift_mult = character_daily_drift_multiplier(character)
    pst = ensure_player_storyline_state(player)

    pid = _player_id(player)
    book = ensure_trade_stability_state(session)
    prev = book.get(pid) if isinstance(book.get(pid), dict) else {}
    prev_score = float(prev.get("trade_stability_score") if prev.get("trade_stability_score") is not None else 88.0)
    prev_level = int(prev.get("escalation_level") or 0)
    prev_day = int(prev.get("last_calendar_day") or -1)
    last_level_change_day = int(prev.get("last_level_change_day") or prev_day if prev_day >= 0 else calendar_idx)

    if prev_day == int(calendar_idx):
        return prev if prev else instant

    delta = (target_score - prev_score) * 0.18
    max_drop = 1.15 * drift_mult
    max_rise = 1.35
    if target_score < prev_score:
        drift = _clamp(delta, -max_drop, 0.0)
    else:
        drift = _clamp(delta, 0.0, max_rise)
    score = round(_clamp(prev_score + drift, 0.0, 100.0), 2)

    target_level = stability_to_escalation_level(score)
    days_since_level_change = int(calendar_idx) - last_level_change_day
    level_change_cooldown = 10 if target_level > prev_level else 14
    if target_level > prev_level + 1:
        if days_since_level_change >= level_change_cooldown:
            level = prev_level + 1
            last_level_change_day = int(calendar_idx)
        else:
            level = prev_level
    elif target_level < prev_level - 1:
        if days_since_level_change >= level_change_cooldown:
            level = prev_level - 1
            last_level_change_day = int(calendar_idx)
        else:
            level = prev_level
    else:
        level = target_level

    sig_count = count_significant_pressures(instant["pressures"])
    if score < 70.0 or sig_count >= 1:
        pst["stability_concern_days"] = int(pst.get("stability_concern_days") or 0) + 1
    else:
        pst["stability_concern_days"] = max(0, int(pst.get("stability_concern_days") or 0) - 1)

    penalties = readiness_penalties(score, character, mental, level)
    apply_readiness_to_player(player, penalties)

    row = {
        "player_id": pid,
        "trade_stability_score": score,
        "target_stability_score": target_score,
        "escalation_level": level,
        "pressures": instant["pressures"],
        "character": int(character),
        "mental": int(mental),
        "readiness_penalties": penalties,
        "prev_escalation_level": prev_level,
        "last_calendar_day": int(calendar_idx),
        "last_level_change_day": last_level_change_day,
        "stability_concern_days": int(pst.get("stability_concern_days") or 0),
        "significant_pressure_count": sig_count,
    }
    book[pid] = row
    return row


def update_player_stability(session: Any, player: Any, team: Any) -> Dict[str, Any]:
    """Immediate stability refresh (trade exposure, crisis hooks)."""
    instant = compute_instant_stability(session, player, team)
    apply_readiness_to_player(player, instant["readiness_penalties"])

    pid = _player_id(player)
    book = ensure_trade_stability_state(session)
    prev = book.get(pid) if isinstance(book.get(pid), dict) else {}
    prev_level = int(prev.get("escalation_level") or 0)
    row = {
        "player_id": pid,
        "trade_stability_score": instant["trade_stability_score"],
        "escalation_level": instant["escalation_level"],
        "pressures": instant["pressures"],
        "character": instant["character"],
        "mental": instant["mental"],
        "readiness_penalties": instant["readiness_penalties"],
        "prev_escalation_level": prev_level,
        "significant_pressure_count": count_significant_pressures(instant["pressures"]),
    }
    book[pid] = row
    return row


def apply_trade_hub_exposure(
    session: Any,
    player: Any,
    *,
    attempt_n: int = 1,
    rejection_kind: str = "rejected",
) -> Dict[str, Any]:
    """Feed trade hub shopping into cumulative stability — rarely instant formal demand."""
    pst = ensure_player_storyline_state(player)
    if rejection_kind == "technical_no_fallout":
        return {"stability_delta": 0.0}

    mental = _player_mental_0_100(player)
    character = _player_character_0_100(player)

    if rejection_kind == "soft_blocked":
        heat_delta = 1
        stability_delta = -0.5
    else:
        heat_delta = 10 + min(8, attempt_n * 2)
        stability_delta = -1.5 - min(4.0, attempt_n * 0.75)
        if mental >= 88 and character >= 82:
            stability_delta = max(stability_delta, -2.0)
        elif mental < 62:
            stability_delta -= 2.5
        if character < 65:
            stability_delta -= 1.5

    pst["trade_rumor_heat"] = min(100, int(pst.get("trade_rumor_heat") or 0) + heat_delta)
    pst["was_recently_shopped"] = True
    gm_drop = 0.01 if rejection_kind == "soft_blocked" else 0.03 + min(0.06, attempt_n * 0.015)
    if mental >= 88:
        gm_drop *= 0.45
    pst["gm_trust"] = _clamp(float(pst.get("gm_trust", 0.72)) - gm_drop, 0.05, 1.0)

    team = None
    league = getattr(getattr(session, "sim", None), "league", None)
    pid = _player_id(player)
    if league is not None:
        for tm in getattr(league, "teams", None) or []:
            for p in getattr(tm, "roster", None) or []:
                if _player_id(p) == pid:
                    team = tm
                    break
            if team:
                break
    if team is not None:
        row = update_player_stability(session, player, team)
        book = ensure_trade_stability_state(session)
        cur = float(row.get("trade_stability_score") or 70.0)
        book[pid]["trade_stability_score"] = _clamp(cur + stability_delta, 0.0, 100.0)
        book[pid]["escalation_level"] = stability_to_escalation_level(book[pid]["trade_stability_score"])

    return {"stability_delta": stability_delta, "heat_delta": heat_delta}


def crisis_stage_from_remaining(initial_seconds: int, remaining_seconds: int) -> int:
    if remaining_seconds <= 0:
        return 4
    if initial_seconds <= 0:
        initial_seconds = CRISIS_DEADLINE_MAX
    ratio = remaining_seconds / float(initial_seconds)
    if ratio > 2.0 / 3.0:
        return 1
    if ratio > 1.0 / 3.0:
        return 2
    return 3


def crisis_trade_value_multiplier(crisis_stage: int) -> float:
    return {
        1: 0.93,
        2: 0.80,
        3: 0.52,
        4: 0.15,
    }.get(int(crisis_stage or 1), 0.93)


def crisis_distressed_asset_cost(base_value: float, *, timer_expired: bool = False) -> float:
    """Negative-value distressed pricing only when the crisis timer fully expires."""
    if not timer_expired:
        return 0.0
    return max(8.0, base_value * 0.35)


def primary_complaint_from_pressures(pressures: Dict[str, float]) -> str:
    if not pressures:
        return "general dissatisfaction"
    top = max(pressures.items(), key=lambda kv: kv[1])
    labels = {
        "role": "deployment and ice time (TOI vs expected usage)",
        "management": "management trust",
        "winning": "team competitiveness",
        "contract": "contract situation",
        "development": "development path",
        "belonging": "place in the locker room",
        "trade_exposure": "being shopped in trade talks",
        "broken_promise": "broken organizational promises",
        "performance": "role relative to production",
        "organizational": "organizational direction",
    }
    return labels.get(top[0], top[0].replace("_", " "))
