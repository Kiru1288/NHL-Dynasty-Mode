"""Mutable franchise session held in memory (wraps live SimEngine)."""

from __future__ import annotations

import copy
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
    injuries_enabled: bool = True
    player_universe: str = "generated"  # generated | real_nhl
    preseason_applied: bool = False

    phase: str = "preseason"  # regular | playoff_ready | playoffs | post_cup | offseason | preseason
    season_phase: str = "preseason"  # mirrors phase; legacy compat
    offseason_stage: Optional[str] = None  # awards | retirements | salary_cap | ...
    # League Operations persistence (YoY revenue + escrow ledger).
    market_revenue_history: Dict[str, Dict[str, float]] = field(default_factory=dict)
    escrow_ledger: Dict[str, Any] = field(default_factory=dict)
    regular_season_complete: bool = False
    playoffs_generated: bool = False
    playoff_payload: Dict[str, Any] = field(default_factory=dict)
    playoffs_simulated: bool = False
    playoffs_done: bool = False
    champion_id: Optional[str] = None
    stanley_cup_winner: Optional[str] = None

    # Offseason progression flags
    awards_generated: bool = False
    awards_payload: Dict[str, Any] = field(default_factory=dict)
    retirements_processed: bool = False
    retirements_payload: Dict[str, Any] = field(default_factory=dict)
    retired_players_archive: List[Dict[str, Any]] = field(default_factory=list)
    contracts_ticked: bool = False
    salary_cap_payload: Dict[str, Any] = field(default_factory=dict)
    development_report_payload: Dict[str, Any] = field(default_factory=dict)
    development_report_done: bool = False
    development_report_completed_season: int = 0
    development_report_generated_at: str = ""
    draft_lottery_done: bool = False
    draft_lottery_payload: Dict[str, Any] = field(default_factory=dict)
    draft_combine_done: bool = False
    draft_combine_payload: Dict[str, Any] = field(default_factory=dict)
    draft_completed: bool = False
    draft_payload: Dict[str, Any] = field(default_factory=dict)
    draft_review_payload: Dict[str, Any] = field(default_factory=dict)
    prospect_rights_payload: Dict[str, Any] = field(default_factory=dict)
    draft_rights_review_payload: Dict[str, Any] = field(default_factory=dict)
    resign_payload: Dict[str, Any] = field(default_factory=dict)
    resign_negotiations: Dict[str, Any] = field(default_factory=dict)
    # Phase-scoped re-sign board: Accepted / Rejected / etc. survive until Free Agency.
    resign_phase_outcomes: Dict[str, Any] = field(default_factory=dict)
    # Player trade demands keyed by player_id (open / resolved ledger).
    trade_demands: Dict[str, Any] = field(default_factory=dict)
    # Exclusive own-FA window before open market (days elapsed of OWN_FA_MORATORIUM_DAYS).
    own_fa_window_day: int = 0
    own_fa_window_active: bool = False
    own_fa_window_signings: List[Dict[str, Any]] = field(default_factory=list)
    free_agency_open: bool = False
    free_agents_payload: List[Dict[str, Any]] = field(default_factory=list)
    free_agency_market_payload: Dict[str, Any] = field(default_factory=dict)
    fa_market_book: Dict[str, Any] = field(default_factory=dict)
    fa_market_day: int = 0
    cpu_fa_signings: Dict[str, Any] = field(default_factory=dict)
    cpu_rfa_decisions: Dict[str, Any] = field(default_factory=dict)
    cpu_fa_wave: int = 0
    roster_cleanup_payload: Dict[str, Any] = field(default_factory=dict)
    next_season_generated: bool = False
    next_season_payload: Dict[str, Any] = field(default_factory=dict)
    next_important_event: str = ""
    # Ordered list of completed offseason stage ids (for resume / timeline UI).
    offseason_completed_stages: List[str] = field(default_factory=list)
    offseason_stage_entered_at: Dict[str, str] = field(default_factory=dict)
    offseason_stage_completed_at: Dict[str, str] = field(default_factory=dict)
    season_history: List[Dict[str, Any]] = field(default_factory=list)

    pending_decisions: List[Dict[str, Any]] = field(default_factory=list)
    notifications: List[Any] = field(default_factory=list)
    timeline: List[str] = field(default_factory=list)
    # Structured narrative + sim hooks (trades, injuries, arcs); API feeds Calendar "Storylines" tab.
    storyline_events: List[Dict[str, Any]] = field(default_factory=list)

    # Central-style draft list: player_key -> rank (1 = best), updated after each advance day
    draft_rank_prev: Dict[str, int] = field(default_factory=dict)
    draft_preseason_rank: Dict[str, int] = field(default_factory=dict)
    draft_midseason_rank: Dict[str, int] = field(default_factory=dict)
    draft_rank_snapshot_week: str = ""
    draft_stock_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    draft_state: Dict[str, Any] = field(default_factory=dict)
    draft_results_archive: List[Dict[str, Any]] = field(default_factory=list)

    # GM scouting assignments, coverage overlays, and budget (see franchise_scouting.py)
    scouting_state: Dict[str, Any] = field(default_factory=dict)

    # General Manager's World progression (skill tree, goals, modifiers)
    # Sim output: every league game + running skater/goalie counting numbers (franchise session only)
    game_results: List[Dict[str, Any]] = field(default_factory=list)
    player_season_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    processed_game_ids: Set[str] = field(default_factory=set)

    # One-off UI recaps (WJC, outdoor games, All-Star) — shown until dismissed
    pending_ui_popups: List[Dict[str, Any]] = field(default_factory=list)
    shown_event_keys: Set[str] = field(default_factory=set)
    showcase_archive: List[Dict[str, Any]] = field(default_factory=list)
    # CPU franchise cognition/state (optional; safe for older saves)
    cpu_franchise_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cpu_scheduler_state: Dict[str, Any] = field(default_factory=dict)
    cpu_trade_event_seen_ids: Set[str] = field(default_factory=set)

    # World Juniors (national teams only — NHL U20 only if user loans them)
    wjc_tournament_bundle: Optional[Dict[str, Any]] = None
    wjc_loan_prompts_enqueued: bool = False
    wjc_nhl_u20_loan: Dict[str, bool] = field(default_factory=dict)  # player_id -> True if loaned to WJC
    wjc_draft_score_boosts: Dict[str, float] = field(default_factory=dict)
    wjc_stock_evaluated_seasons: Set[int] = field(default_factory=set)

    # Contract/cap bootstrap health: ready | repaired | partial | failed
    financials_status: str = "partial"

    # Saved lineups (even-strength / PP / PK) keyed by unit type; source of truth for Edit Lines
    lines: Dict[str, Any] = field(default_factory=dict)

    # Code fingerprint at session create time — mismatched vs live → session is expired
    code_revision: str = ""

    # Preseason archive (set when regular season starts)
    preseason_player_stats_snapshot: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    preseason_standings_snapshot: Any = None
    preseason_game_results_snapshot: List[Dict[str, Any]] = field(default_factory=list)

    # Cached read-model payloads (invalidated on advance/trade/scouting)
    _cached_draft_class_rankings: Optional[Dict[str, Any]] = None
    _draft_rankings_cache_state: str = "missing"  # missing | dirty | valid
    _cached_scouting_prospects_payload: Optional[Dict[str, Any]] = None
    _cached_scouting_world_payload: Optional[Dict[str, Any]] = None
    _cached_roster_browser_payload: Optional[Dict[str, Any]] = None
    _cached_trade_assets_payload: Optional[Dict[str, Any]] = None

    # Cause-and-effect storyline system (decision log + active arcs)
    decision_event_log: List[Dict[str, Any]] = field(default_factory=list)
    active_cause_storylines: List[Dict[str, Any]] = field(default_factory=list)
    # Narrative universe: persistent arcs, social feed, player reputation memory
    story_arcs: List[Dict[str, Any]] = field(default_factory=list)
    social_posts: List[Dict[str, Any]] = field(default_factory=list)
    player_narrative_memory: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph: List[Dict[str, Any]] = field(default_factory=list)
    press_conference_queue: List[Dict[str, Any]] = field(default_factory=list)
    narrative_archive: List[Dict[str, Any]] = field(default_factory=list)
    narrative_eras: List[Dict[str, Any]] = field(default_factory=list)
    prospect_social_profiles: Dict[str, Any] = field(default_factory=dict)
    agent_relationships: Dict[str, Any] = field(default_factory=dict)
    _narrative_sealed_seasons: List[int] = field(default_factory=list)
    _storyline_blocked_log: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def __getstate__(self) -> Dict[str, Any]:
        """Drop non-serializable runtime locks/caches so pickle/save clones work."""
        state = dict(self.__dict__)
        for key in list(state.keys()):
            if key.endswith("_lock") or key.endswith("_Lock"):
                state.pop(key, None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "FranchiseSession":
        """deepcopy cannot clone threading.Lock — reuse pickle-safe state."""
        cls = type(self)
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for key, value in self.__getstate__().items():
            setattr(clone, key, copy.deepcopy(value, memo))
        return clone
