from __future__ import annotations

"""
League seasonal framework package.

This package exposes helpers for:
    - Regular-season schedule generation
    - Standings tracking and views
    - Playoff bracket + series simulation
    - Yearly league awards
"""

from .schedule_generator import (
    GameSlot,
    TeamScheduleMeta,
    normalize_teams,
    generate_regular_season_schedule,
    games_for_team,
    team_schedule,
)

from .standings import (
    StandingsTable,
    TeamStandingRecord,
)

from .playoffs import (
    PlayoffSeries,
    PlayoffResult,
    simulate_playoffs,
)

from .awards import (
    Award,
    compute_awards,
)

__all__ = [
    # schedule
    "GameSlot",
    "TeamScheduleMeta",
    "normalize_teams",
    "generate_regular_season_schedule",
    "games_for_team",
    "team_schedule",
    # standings
    "StandingsTable",
    "TeamStandingRecord",
    # playoffs
    "PlayoffSeries",
    "PlayoffResult",
    "simulate_playoffs",
    # awards
    "Award",
    "compute_awards",
]

