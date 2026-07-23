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
    build_playoff_first_round,
    simulate_playoffs,
)

from .awards import (
    Award,
    AWARD_REGISTRY,
    apply_career_award_history,
    build_awards_payload,
    calder_eligibility,
    compute_awards,
    compute_official_watch_lists,
    normalize_percentage,
    serialize_award,
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
    "build_playoff_first_round",
    "simulate_playoffs",
    # awards
    "Award",
    "AWARD_REGISTRY",
    "compute_awards",
    "build_awards_payload",
    "serialize_award",
    "calder_eligibility",
    "normalize_percentage",
    "compute_official_watch_lists",
    "apply_career_award_history",
]

