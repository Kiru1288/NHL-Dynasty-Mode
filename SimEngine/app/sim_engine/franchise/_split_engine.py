"""Safe AST split of engine.py into franchise submodules."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

FRANCHISE = Path(__file__).resolve().parent
SRC = FRANCHISE / "engine.py"

RULES: list[tuple[str, callable]] = [
    ("schedule", lambda n: n.startswith((
        "_sync_nhl_calendar", "_calendar_", "_team_plays", "_can_place", "_slot_",
        "_regular_game", "_build_team_game", "_team_schedule", "_team_has_impossible",
        "_validate_league_cadence", "_get_regular_row", "_league_schedule", "_league_penalty",
        "_pair_penalty", "_clone_slot", "_find_slot", "_move_slot", "_schedule_", "_repair_",
        "_dedupe_", "_candidate_eligible", "_smooth_", "_merge_abstract", "_finalize_schedule",
        "_resolve_calendar_day", "_would_create_bad_cadence", "_calendar_day_fits",
    )) or n in ("_team_ids_for_slot",)),
    ("contracts", lambda n: n.startswith((
        "_player_cap_hit", "_team_cap_snapshot", "_ensure_league_cap", "_contract_",
        "_generate_contract", "_player_needs_contract", "_apply_fallback_minimum",
        "_ensure_player_contract", "_validate_team_financials", "_ensure_team_roster_contracts",
        "_ensure_league_roster_contracts", "_supported_contract_rules", "_player_pool_",
        "_normalize_contract", "_build_team_cap_summary", "_estimate_buyout", "_estimate_extension",
        "_is_true_free_agent", "_estimate_fa_demands", "_contract_status_chips",
        "_build_contract_row", "_build_free_agent_row", "_next_year_cap_projection",
        "build_contract_office_payload", "_player_season_production", "_team_competitiveness",
        "_player_tenure_years",
    )) or n in (
        "ensure_player_financials", "get_contract_office",
        "_apply_player_cap_hit", "_scale_player_cap_hit",
    )),
    ("decisions", lambda n: n.startswith((
        "_auto_resolve_pending", "_apply_injury_decision", "_apply_ice_time_decision",
        "_apply_generic_storyline", "_pending_decision_snapshot", "_advance_blocked",
        "auto_resolve_franchise_decisions", "_append_decision_feedback", "apply_storyline_choice",
        "apply_decision", "_nudge_player_psych", "_nudge_team_room", "_find_player_on_team",
        "_player_display_name", "_maybe_enqueue_post_day_decisions",
    ))),
    ("serialization", lambda n: n.startswith((
        "_serialize_", "_build_standings", "_build_stats", "_build_schedule_upcoming",
        "_nhl_today", "_results_by_calendar", "_coerce_final", "_validate_final_game",
        "_normalize_player_stat", "_normalize_stat_position", "_is_goalie_stat",
        "_lookup_franchise_team", "_teams_directory", "_attach_team_meta", "_stats_integrity",
        "_stat_int", "_stat_float", "_league_team_stats", "_user_team_stats", "_user_team_record",
        "_player_scouting", "_rating_", "_active_roster", "_skaters", "_goalies",
        "_available_goalies", "_goalie_availability", "_ovr_weight", "_player_role_usage",
        "_rating_avg", "_offense_opportunity", "_team_shooting", "build_draft_class_rankings",
        "_name_str", "_pos_str", "_playoff_team_row", "_build_playoff_payload",
        "_find_development_league", "_remove_player_from_development", "_rows_from_players",
        "_serialize_development", "_build_roster_browser", "_normalize_storyline",
        "_normalize_notification", "_build_injuries", "_build_injury_history",
        "_standing_row", "_top_two_", "_special_event", "_nhl_calendar_strip",
        "_nhl_calendar_full", "_team_abbr", "_city_lower", "_is_canadian_franchise",
        "_ginfo_from_saved", "_stable_franchise_game", "_game_results_by_calendar",
        "_remaining_regular", "_completed_regular", "_regular_season_is_truly",
        "_saved_game_is_final", "_game_result_calendar",
    )) or n in (
        "list_teams_summary", "get_franchise_game_detail", "get_franchise_chemistry_report",
        "_display_team", "_franchise_team_abbrev", "snapshot_draft_rank_prev",
    )),
    ("advance", lambda n: n.startswith((
        "advance_franchise", "_finalize_regular_calendar", "_accumulate_franchise_game",
        "_franchise_sync_ledger", "_merge_skater_possession", "_stat_ensure", "_stat_add",
        "_pick_assist", "_scoring_chunk", "_goalie_game", "_skater_box_rows", "_goals_play_by_play",
        "_simulate_franchise", "_simulate_slots", "_franchise_daily", "_franchise_fanout",
        "_franchise_enqueue", "_maybe_roll_storyline", "_split_preseason", "_enter_postseason",
        "_purge_retired", "_depth_pool_progression",
    ))),
    ("state", lambda n: n.startswith((
        "build_state_payload", "invalidate_session_payload_caches", "get_cached_",
        "_storyline_choices_payload", "_build_trade_assets", "_trade_execution",
        "execute_trade_package", "execute_franchise_draft_pick", "_bootstrap_depth_pool",
        "dismiss_franchise_popups", "_append_showcase_popup", "_record_storyline",
        "_merge_simengine_league_news", "_storyline_dedupe",
    ))),
    ("events", lambda n: n.startswith((
        "_wjc_", "build_world_juniors", "_maybe_enqueue_wjc", "_simulate_showcase",
        "_allstar_game", "_maybe_enqueue_showcase", "_strip_wjc", "_push_wjc", "_rr_standings",
        "_simulate_wjc", "_rng_for_event",
    ))),
    ("progression", lambda n: n.startswith((
        "_strip_retired_from_nhl", "_franchise_nhl_age", "_run_franchise_season_end",
        "_FranchiseLifecycleLogger",
    ))),
    ("playoffs", lambda n: n.startswith((
        "_playoff_", "enter_franchise_playoffs", "_simulate_postseason",
        "_transition_to_playoff", "_enqueue_playoff",
    )) or n == "advance_season_phase"),
    ("common", lambda n: n.startswith((
        "_franchise_startup", "_fr_dbg", "_franchise_log_injury", "resolve_user_team",
        "_use_world", "_franchise_injuries", "apply_coach_archetype", "_chaos_index",
        "_ensure_session_event", "_append_unique_dict_event", "_normalized_notification",
        "_normalized_timeline", "_clamp", "_franchise_clamp",
    )) or n == "start_franchise"),
]


def assign(name: str) -> str:
    for mod, pred in RULES:
        if pred(name):
            return mod
    return "engine_core"


DOCS = {
    "common": "Shared helpers and franchise startup.",
    "schedule": "Schedule generation, validation, and repair.",
    "contracts": "Contracts, cap hits, and contract office.",
    "serialization": "API serialization for rosters, stats, and standings.",
    "advance": "Day advancement and game simulation.",
    "state": "State payload, caches, trades, and popups.",
    "decisions": "Pending user decisions and storyline choices.",
    "events": "WJC, All-Star, and showcase events.",
    "progression": "Season-end aging and progression.",
    "playoffs": "Playoff entry and payloads.",
    "engine_core": "Remaining orchestration.",
}


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    module = ast.parse(source)

    defs = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    first_def_line = min(n.lineno for n in defs)

    # Shared header: imports/constants only (skip leading module docstrings)
    header_lines = lines[: first_def_line - 1]
    start = 0
    for i, ln in enumerate(header_lines):
        s = ln.strip()
        if s.startswith("from __future__"):
            start = i
            break
    shared_header = "".join(header_lines[start:])
    shared_path = FRANCHISE / "_shared.py"
    shared_path.write_text(
        '"""Shared imports and constants for franchise submodules."""\n' + shared_header,
        encoding="utf-8",
    )
    print("wrote _shared.py")

    chunks: dict[str, list[str]] = defaultdict(list)
    core: list[str] = []

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            chunk = "".join(lines[node.lineno - 1 : node.end_lineno])
            mod = assign(node.name)
            if mod == "engine_core":
                core.append(chunk)
            else:
                chunks[mod].append(chunk)

    import_line = "from app.sim_engine.franchise._shared import *  # noqa: F401,F403\n\n"

    for mod, bodies in chunks.items():
        doc = DOCS.get(mod, mod)
        content = f'"""{doc}"""\n\nfrom __future__ import annotations\n\n{import_line}{"".join(bodies)}'
        (FRANCHISE / f"{mod}.py").write_text(content, encoding="utf-8")
        print(f"wrote {mod}.py ({len(bodies)} defs)")

    core_content = (
        '"""Remaining franchise orchestration."""\n\nfrom __future__ import annotations\n\n'
        + import_line
        + "".join(core)
    )
    (FRANCHISE / "engine_core.py").write_text(core_content, encoding="utf-8")
    print(f"wrote engine_core.py ({len(core)} defs)")

    bridge = '''\
"""
Franchise mode public API facade.

Submodules: common, schedule, contracts, serialization, advance, state, decisions,
events, progression, playoffs, engine_core.
"""

from __future__ import annotations

from app.sim_engine.franchise.engine_core import *  # noqa: F401,F403
from app.sim_engine.franchise.common import *  # noqa: F401,F403
from app.sim_engine.franchise.schedule import *  # noqa: F401,F403
from app.sim_engine.franchise.contracts import *  # noqa: F401,F403
from app.sim_engine.franchise.serialization import *  # noqa: F401,F403
from app.sim_engine.franchise.advance import *  # noqa: F401,F403
from app.sim_engine.franchise.state import *  # noqa: F401,F403
from app.sim_engine.franchise.decisions import *  # noqa: F401,F403
from app.sim_engine.franchise.events import *  # noqa: F401,F403
from app.sim_engine.franchise.progression import *  # noqa: F401,F403
from app.sim_engine.franchise.playoffs import *  # noqa: F401,F403

# Offseason API (separate module from pre-split monolith)
from app.sim_engine.franchise.offseason import (  # noqa: E402
    advance_season_phase,
    build_offseason_state_extras,
    continue_offseason,
    generate_next_season,
    complete_playoffs,
)


def continue_franchise_offseason(session):
    return continue_offseason(session)


def generate_franchise_next_season(session):
    return generate_next_season(session)


def enter_franchise_playoffs(session):
    from app.sim_engine.franchise.playoffs import enter_franchise_playoffs as _enter

    return _enter(session)


def get_cached_trade_assets_payload(session):
    from app.sim_engine.franchise.state import get_cached_trade_assets_payload as _fn

    return _fn(session)


def execute_franchise_draft_pick(session, *args, **kwargs):
    from app.sim_engine.franchise.state import execute_franchise_draft_pick as _fn

    return _fn(session, *args, **kwargs)


def get_contract_office(session):
    from app.sim_engine.franchise.contracts import get_contract_office as _fn

    return _fn(session)
'''
    # Backup monolith before replacing
    backup = FRANCHISE / "engine_monolith.py"
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    SRC.write_text(bridge, encoding="utf-8")
    print("rewrote engine.py facade + bridge")


if __name__ == "__main__":
    main()
