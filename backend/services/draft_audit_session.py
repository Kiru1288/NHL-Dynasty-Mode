"""Fast draft-class audit sessions — canonical path uses live ranking code only."""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

from services.franchise_paths import ensure_simengine_path

ensure_simengine_path()

import run_sim as rs  # noqa: E402
from app.sim_engine.engine import SimEngine  # noqa: E402
from app.sim_engine.league_hierarchy_bootstrap import bootstrap_full_league_hierarchy  # noqa: E402
from app.sim_engine.league.standings import StandingsTable  # noqa: E402
from services.franchise_session import FranchiseSession  # noqa: E402


def create_fast_audit_session(seed: int, *, team_query: str = "Toronto") -> FranchiseSession:
    """Bootstrap dev leagues + draft pool without full NHL schedule generation."""
    sim = SimEngine(seed=int(seed), debug=False)
    league = sim.league
    teams = list(getattr(league, "teams", None) or [])
    if not teams:
        raise RuntimeError("League has no teams after initialization")

    user_team = teams[0]
    for t in teams:
        name = str(getattr(t, "name", "") or "").lower()
        if team_query.lower() in name:
            user_team = t
            break

    uid = rs._team_id(user_team)
    sim.team = user_team
    bootstrap_full_league_hierarchy(league, sim.rng)

    team_by_id: Dict[str, Any] = {rs._team_id(t): t for t in teams}
    standings = StandingsTable(teams)

    return FranchiseSession(
        session_id=FranchiseSession.new_id(),
        sim=sim,
        user_team_id=uid,
        head_coach_name="Audit",
        coach_archetype="balanced",
        team_by_id=team_by_id,
        standings=standings,
        nhl_calendar=[],
        calendar_cursor=0,
        season_calendar_year=2025,
        phase="preseason",
        season_phase="preseason",
    )


def create_audit_session(seed: int, *, fast: bool = False, team_query: str = "Toronto") -> FranchiseSession:
    if fast:
        return create_fast_audit_session(seed, team_query=team_query)
    from services.franchise_sim import start_franchise

    return start_franchise(
        team_query=team_query,
        head_coach_name="Audit",
        coach_archetype="balanced",
        seed=seed,
    )
