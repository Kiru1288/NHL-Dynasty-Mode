"""Mutable franchise session held in memory (wraps live SimEngine)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class FranchiseSession:
    session_id: str
    sim: Any
    user_team_id: str
    head_coach_name: str
    coach_archetype: str
    season_calendar_year: int = 2025
    games_per_team_schedule: int = 82
    calendar_days_finished: int = 0

    schedule: List[Any] = field(default_factory=list)
    by_day: Dict[int, List[Any]] = field(default_factory=dict)
    days_sorted: List[int] = field(default_factory=list)  # legacy: game-only indices; may be empty

    # NHL calendar (Sept → June); cursor advances one real day per franchise day.
    nhl_calendar: List[Dict[str, Any]] = field(default_factory=list)
    calendar_cursor: int = 0
    nhl_regular_season_last_index: int = 0

    standings: Any = None
    team_by_id: Dict[str, Any] = field(default_factory=dict)
    team_ids: List[str] = field(default_factory=list)
    strength_map: Dict[str, float] = field(default_factory=dict)

    prev_calendar_day: Optional[int] = None
    last_game_day: Dict[str, Optional[int]] = field(default_factory=dict)
    play_days: Dict[str, Any] = field(default_factory=dict)
    injury_log_major: List[Dict[str, Any]] = field(default_factory=list)
    injury_log_all: List[Dict[str, Any]] = field(default_factory=list)

    chaos_index: float = 0.5
    use_world: bool = False
    preseason_applied: bool = False

    phase: str = "regular"  # regular | complete
    playoffs_simulated: bool = False
    champion_id: Optional[str] = None

    pending_decisions: List[Dict[str, Any]] = field(default_factory=list)
    notifications: List[Any] = field(default_factory=list)
    timeline: List[str] = field(default_factory=list)
    # Structured narrative + sim hooks (trades, injuries, arcs); API feeds Calendar "Storylines" tab.
    storyline_events: List[Dict[str, Any]] = field(default_factory=list)

    # Central-style draft list: player_key -> rank (1 = best), updated after each advance day
    draft_rank_prev: Dict[str, int] = field(default_factory=dict)

    # Sim output: every league game + running skater/goalie counting numbers (franchise session only)
    game_results: List[Dict[str, Any]] = field(default_factory=list)
    player_season_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # One-off UI recaps (WJC, outdoor games, All-Star) — shown until dismissed
    pending_ui_popups: List[Dict[str, Any]] = field(default_factory=list)
    shown_event_keys: Set[str] = field(default_factory=set)
    showcase_archive: List[Dict[str, Any]] = field(default_factory=list)

    # World Juniors (national teams only — NHL U20 only if user loans them)
    wjc_tournament_bundle: Optional[Dict[str, Any]] = None
    wjc_loan_prompts_enqueued: bool = False
    wjc_nhl_u20_loan: Dict[str, bool] = field(default_factory=dict)  # player_id -> True if loaned to WJC

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())
