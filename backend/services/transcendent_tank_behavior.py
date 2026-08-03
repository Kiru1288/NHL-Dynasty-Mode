"""Transcendent draft tank pressure — compute, persist, and drive AI behavior."""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from services.draft_ranking_logic import compute_tank_pressure_for_team

logger = logging.getLogger(__name__)

TANK_STORYLINE_TYPES = {
    "TEAM_ENTERS_TANK_MODE": "Front office pivots to lottery positioning",
    "TEAM_SHOPS_VETERANS": "Club listening on veteran rentals",
    "TEAM_SELLS_RENTAL": "Expiring veteran on the market",
    "TEAM_PLAYS_YOUTH": "Coaching staff leaning on young lineup",
    "TEAM_SCRATCHES_VETERAN": "Veteran scratched for youth look",
    "TEAM_ABORTS_PLAYOFF_PUSH": "Club abandons short-term push",
}

DEV_LEAGUE_SPAWN_VERSION = 2


def _team_id(team: Any) -> str:
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", "")
    return str(tid or "")


def _team_status_for_tank(team: Any, standings: Any = None) -> str:
    window = str(getattr(team, "gm_window", "") or getattr(team, "window", "") or "").lower()
    if "rebuild" in window or "retool" in window:
        return "rebuilding"
    if "tank" in window:
        return "tanking"
    if "contend" in window or "playoff" in window or "win_now" in window:
        return "playoff_contender"
    st = str(getattr(team, "status", "") or "").lower()
    if "rebuild" in st or "tank" in st:
        return "tanking"
    if "contend" in st or "playoff" in st:
        return "playoff_contender"
    if standings is not None:
        tid = _team_id(team)
        rec = getattr(standings, "records", {}).get(tid)
        if rec is not None:
            gp = int(getattr(rec, "gp", 0) or 0)
            if gp >= 20:
                pts = int(getattr(rec, "points", 0) or 0)
                ppg = pts / max(1, gp)
                if ppg < 0.92:
                    return "tanking"
                if ppg < 1.02:
                    return "middling"
    return "middling"


def refresh_transcendent_tank_pressure(
    session: Any,
    sim: Any,
    *,
    transcendent_present: bool,
    draft_year: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Recompute per-team tank pressure using real pick ownership."""
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry, team_owns_own_first

    league = getattr(sim, "league", None)
    if league is None:
        return {}
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    dy = int(draft_year or season_year + 1)
    ensure_draft_pick_registry(league, start_year=dy, years_ahead=4)

    standings = getattr(session, "standings", None)
    out: Dict[str, Dict[str, Any]] = {}
    for tm in getattr(league, "teams", None) or []:
        tid = _team_id(tm)
        if not tid:
            continue
        own = team_owns_own_first(league, tid, draft_year=dy)
        setattr(tm, "team_status", _team_status_for_tank(tm, standings))
        payload = compute_tank_pressure_for_team(
            tm,
            transcendent_present=transcendent_present,
            owns_own_first=bool(own.get("owns_own_first")),
            pick_ownership_reason=str(own.get("pick_ownership_reason") or "unknown"),
            owns_protected_first=bool(own.get("owns_protected_first")),
        )
        out[tid] = payload
        setattr(tm, "_franchise_tank_mode", payload.get("tank_mode"))
        setattr(tm, "_franchise_tank_pressure", int(payload.get("tank_pressure") or 0))
    session.transcendent_tank_pressure = out
    setattr(league, "transcendent_tank_pressure", out)
    setattr(league, "transcendent_active", bool(transcendent_present))
    return out


def get_tank_payload(session: Any, team_id: str) -> Dict[str, Any]:
    m = getattr(session, "transcendent_tank_pressure", None) or {}
    if isinstance(m, dict):
        row = m.get(str(team_id))
        if isinstance(row, dict):
            return row
    return {"tank_pressure": 0, "tank_mode": "none"}


def record_tank_storyline(session: Any, team_id: str, event_type: str, headline: str) -> None:
    from services.franchise_sim import _record_storyline

    _record_storyline(
        session,
        {
            "type": event_type,
            "team_id": str(team_id),
            "team": str(team_id),
            "headline": headline,
            "priority": "HIGH",
            "effects": {"tank_behavior": event_type},
        },
    )


def _player_age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    if ident is not None:
        return int(getattr(ident, "age", 0) or 0)
    return int(getattr(p, "age", 0) or 0)


def apply_tank_daily_behavior(
    session: Any,
    sim: Any,
    teams: List[Any],
    rng: random.Random,
    news_out: List[Dict[str, Any]],
    counters: Dict[str, int],
) -> None:
    """Daily CPU reactions when a transcendent prospect exists."""
    if not getattr(session, "transcendent_draft_prospect_id", None):
        return
    tank_map = getattr(session, "transcendent_tank_pressure", None) or {}
    if not isinstance(tank_map, dict) or not tank_map:
        refresh_transcendent_tank_pressure(session, sim, transcendent_present=True)

    for tm in teams:
        tid = _team_id(tm)
        row = get_tank_payload(session, tid)
        pressure = int(row.get("tank_pressure") or 0)
        mode = str(row.get("tank_mode") or "none")
        if pressure < 30 or mode == "none":
            continue
        if str(_team_status_for_tank(tm, getattr(session, "standings", None))) == "playoff_contender":
            continue

        setattr(tm, "_franchise_tank_mode", mode)
        setattr(tm, "_franchise_tank_pressure", pressure)

        if mode == "hard_tank" and not row.get("owns_own_first", True):
            continue

        roster = list(getattr(tm, "roster", None) or [])
        veterans = [p for p in roster if _player_age(p) >= 30 and not getattr(p, "retired", False)]
        youth = [p for p in roster if _player_age(p) <= 22 and not getattr(p, "retired", False)]

        if pressure >= 50 and veterans and rng.random() < 0.08:
            record_tank_storyline(session, tid, "TEAM_SHOPS_VETERANS", f"{getattr(tm, 'name', tid)} listening on veteran pieces")
            news_out.append({"type": "tank", "headline": f"{getattr(tm, 'name', tid)} shopping veterans", "team": tid, "priority": "MEDIUM"})

        if pressure >= 70 and veterans and rng.random() < 0.06:
            record_tank_storyline(session, tid, "TEAM_SELLS_RENTAL", f"{getattr(tm, 'name', tid)} open to moving a rental")
            counters["tank_sell_signals"] = int(counters.get("tank_sell_signals", 0)) + 1

        if pressure >= 50 and youth and rng.random() < 0.10:
            record_tank_storyline(session, tid, "TEAM_PLAYS_YOUTH", f"{getattr(tm, 'name', tid)} giving youth a longer look")
            counters["tank_youth_plays"] = int(counters.get("tank_youth_plays", 0)) + 1

        if mode == "hard_tank" and veterans and rng.random() < 0.05:
            vet = max(veterans, key=lambda p: _player_age(p))
            scratched = getattr(tm, "_tank_scratched_ids", None) or set()
            scratched = set(scratched)
            scratched.add(str(getattr(vet, "id", "")))
            setattr(tm, "_tank_scratched_ids", scratched)
            record_tank_storyline(session, tid, "TEAM_SCRATCHES_VETERAN", f"{getattr(tm, 'name', tid)} scratches veteran for youth push")
            news_out.append({"type": "scratch", "headline": "Veteran scratched in youth push", "team": tid, "priority": "LOW"})

        if pressure >= 90 and mode == "hard_tank" and rng.random() < 0.04:
            record_tank_storyline(session, tid, "TEAM_ENTERS_TANK_MODE", f"{getattr(tm, 'name', tid)} enters full lottery mode")
            counters["hard_tank_teams"] = int(counters.get("hard_tank_teams", 0)) + 1

        if pressure >= 60 and _team_status_for_tank(tm) == "middling" and rng.random() < 0.05:
            record_tank_storyline(session, tid, "TEAM_ABORTS_PLAYOFF_PUSH", f"{getattr(tm, 'name', tid)} abandons playoff push")


def tank_trade_interest_adjustment(team_id: str, context: Optional[Dict[str, Any]]) -> Tuple[float, List[str]]:
    """Boost seller interest when shopping veterans under transcendent tank pressure."""
    ctx = context or {}
    m = ctx.get("tank_pressure_by_team") or {}
    row = m.get(str(team_id)) or {}
    pressure = int(row.get("tank_pressure") or 0)
    mode = str(row.get("tank_mode") or "none")
    bonus = 0.0
    notes: List[str] = []
    if pressure >= 30:
        bonus += 0.04
    if pressure >= 50:
        bonus += 0.06
        notes.append("Transcendent lottery race — open to moving veterans")
    if pressure >= 70:
        bonus += 0.08
    if mode == "hard_tank" and row.get("owns_own_first"):
        bonus += 0.05
        notes.append("Hard tank mode — prioritizing future assets")
    if mode == "hard_tank" and not row.get("owns_own_first"):
        bonus = min(bonus, 0.02)
        notes.append("Pick traded away — limited incentive to crater")
    return bonus, notes


def check_dev_league_generation_version(league: Any) -> Dict[str, Any]:
    """Warn when dev-league players predate body/team normalization."""
    stale = 0
    total = 0
    bad_names = 0
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            tname = str(tm.get("name") or "")
            if "EU_J" in tname.upper() or " CHL " in tname.upper():
                bad_names += 1
            for p in tm.get("players") or []:
                total += 1
                if int(getattr(p, "_spawn_version", 0) or 0) < DEV_LEAGUE_SPAWN_VERSION:
                    stale += 1
    needs_rebootstrap = stale > 0 or bad_names > 0
    return {
        "needs_rebootstrap": needs_rebootstrap,
        "stale_player_count": stale,
        "dev_player_total": total,
        "bad_team_name_count": bad_names,
        "expected_spawn_version": DEV_LEAGUE_SPAWN_VERSION,
        "warning": (
            "Development league was generated with an older prospect template. "
            "Start a new franchise or run dev rebootstrap for normalized bodies/team names."
            if needs_rebootstrap
            else None
        ),
    }


def rebootstrap_development_leagues(session: Any) -> Dict[str, Any]:
    """Dev-only: rebuild junior/college dev leagues in-place."""
    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None)
    if league is None or sim is None:
        return {"ok": False, "error": "no league"}
    rng = getattr(sim, "rng", None) or random.Random()
    from app.sim_engine.league_hierarchy_bootstrap import bootstrap_full_league_hierarchy

    old_dev = list(getattr(league, "development_leagues", None) or [])
    old_ids = set()
    for block in old_dev:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                old_ids.add(id(p))
    league.players = [p for p in (getattr(league, "players", None) or []) if id(p) not in old_ids]
    bootstrap_full_league_hierarchy(league, rng)
    check = check_dev_league_generation_version(league)
    return {"ok": True, "rebootstrap": check}
