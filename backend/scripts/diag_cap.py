"""Diagnostic: inspect cap snapshots for all 32 teams right after real-NHL bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
SIM = ROOT / "SimEngine" / "app"
for p in (str(BACKEND), str(SIM), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services import franchise_sim  # noqa: E402
from services.contract_economy import get_team_cap_snapshot_full  # noqa: E402


def main() -> None:
    session = franchise_sim.start_franchise(
        team_query="Ottawa Senators",
        head_coach_name="Diag Coach",
        coach_archetype="balanced",
        seed=777,
        player_universe="real_nhl",
        games_per_team=82,
    )
    league = session.sim.league
    teams = list(league.teams)
    print(f"teams={len(teams)}")
    rows = []
    for t in teams:
        snap = get_team_cap_snapshot_full(t, league, session.sim, season_year=session.season_calendar_year)
        rows.append((
            getattr(t, "abbreviation", None) or getattr(t, "abbrev", "?"),
            snap["upper_limit_m"],
            snap["total_cap_hit_m"],
            snap["usable_cap_space_m"],
            snap["real_cap_space_m"],
            snap["bonus_reserve_m"],
            snap["retained_salary_m"],
            snap["buyout_cap_hit_m"],
            snap["other_dead_cap_m"],
            snap["ltir_pool_m"],
            snap["active_roster_count"],
        ))
    rows.sort(key=lambda r: r[3])
    print(f"{'team':6}{'upper':>8}{'total':>8}{'usable':>8}{'real':>8}{'bonusRes':>10}{'retained':>9}{'buyout':>8}{'otherDead':>10}{'ltir':>7}{'roster':>7}")
    for r in rows:
        print(f"{r[0]:6}{r[1]:8.2f}{r[2]:8.2f}{r[3]:8.2f}{r[4]:8.2f}{r[5]:10.2f}{r[6]:9.2f}{r[7]:8.2f}{r[8]:10.2f}{r[9]:7.2f}{r[10]:7d}")

    print("\n--- user team (Ottawa) raw snapshot ---")
    user_team = session.team_by_id.get(str(session.user_team_id))
    snap = get_team_cap_snapshot_full(user_team, league, session.sim, season_year=session.season_calendar_year)
    import json
    print(json.dumps({k: v for k, v in snap.items() if k != "_raw"}, indent=2, default=str))
    print(json.dumps(snap["_raw"], indent=2, default=str))


if __name__ == "__main__":
    main()
