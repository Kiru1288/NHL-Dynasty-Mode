"""Mutable franchise session held in memory (wraps live SimEngine)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FranchiseSession:
    session_id: str
    sim: Any
    user_team_id: str
    head_coach_name: str
    coach_archetype: str
    season_calendar_year: int = 2025

    schedule: List[Any] = field(default_factory=list)
    by_day: Dict[int, List[Any]] = field(default_factory=dict)
    days_sorted: List[int] = field(default_factory=list)
    day_index: int = 0

    standings: Any = None
    team_by_id: Dict[str, Any] = field(default_factory=dict)
    team_ids: List[str] = field(default_factory=list)
    strength_map: Dict[str, float] = field(default_factory=dict)

    prev_calendar_day: Optional[int] = None
    last_game_day: Dict[str, Optional[int]] = field(default_factory=dict)
    play_days: Dict[str, Any] = field(default_factory=dict)
    injury_log_major: List[Dict[str, Any]] = field(default_factory=list)

    chaos_index: float = 0.5
    use_world: bool = False
    preseason_applied: bool = False

    phase: str = "regular"  # regular | complete
    playoffs_simulated: bool = False
    champion_id: Optional[str] = None

    pending_decisions: List[Dict[str, Any]] = field(default_factory=list)
    notifications: List[str] = field(default_factory=list)
    timeline: List[str] = field(default_factory=list)

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())
