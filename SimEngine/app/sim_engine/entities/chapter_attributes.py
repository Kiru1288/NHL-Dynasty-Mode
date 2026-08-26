"""
Chapter-Based Player Rating System — authoritative schema + generation foundation.

Visible chapters summarize player identity for UI/scouting.
Hidden sub-attributes drive simulation behaviour via legacy rating adapters.

This module is intentionally centralized and schema-driven so chapters,
hidden components, weights, and mappings can evolve without rewiring the game.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.sim_engine.entities.player import (  # noqa: WPS433
    ALIASES,
    ATTRIBUTE_KEYS,
    DEFAULT_NHL_RATING,
    clamp_rating,
    display_rating,
    normalize_ratings_dict,
)

SCHEMA_VERSION = 1
RATING_MIN = 20
RATING_MAX = 99

SKATER_CHAPTER_IDS: Tuple[str, ...] = (
    "overall",
    "character",
    "offence",
    "defence",
    "transition",
    "mental",
    "physical",
    "potential",
)

GOALIE_CHAPTER_IDS: Tuple[str, ...] = (
    "overall",
    "glove",
    "blocker",
    "stick",
    "potential",
)


@dataclass(frozen=True)
class HiddenComponent:
    id: str
    label: str
    aggregate_weight: float = 1.0


@dataclass(frozen=True)
class ChapterDefinition:
    id: str
    label: str
    hidden: Tuple[HiddenComponent, ...]
    positions: Tuple[str, ...] = ("C", "LW", "RW", "F", "D", "LD", "RD", "G")
    goalie_only: bool = False
    skater_only: bool = False


def _hc(cid: str, label: str, weight: float = 1.0) -> HiddenComponent:
    return HiddenComponent(id=cid, label=label, aggregate_weight=weight)


# ---------------------------------------------------------------------------
# Authoritative chapter schema (extensible — edit here, not scattered refs)
# ---------------------------------------------------------------------------

SKATER_CHAPTERS: Tuple[ChapterDefinition, ...] = (
    ChapterDefinition(
        id="character",
        label="Character",
        hidden=(
            _hc("leadership", "Leadership", 1.1),
            _hc("professionalism", "Professionalism", 1.0),
            _hc("work_ethic", "Work Ethic", 0.9),
            _hc("coachability", "Coachability", 1.0),
            _hc("loyalty", "Loyalty", 0.8),
            _hc("competitiveness", "Competitiveness", 1.0),
            _hc("accountability", "Accountability", 0.9),
            _hc("selflessness", "Selflessness", 0.8),
            _hc("temperament", "Temperament", 0.9),
            _hc("locker_room_presence", "Locker Room Presence", 1.0),
            _hc("resilience", "Resilience", 1.0),
            _hc("ambition", "Ambition", 0.85),
            _hc("ego", "Ego", 0.7),
            _hc("emotional_stability", "Emotional Stability", 1.0),
            _hc("mentoring", "Mentoring", 0.6),
            _hc("adversity_response", "Adversity Response", 0.95),
        ),
    ),
    ChapterDefinition(
        id="offence",
        label="Offence",
        hidden=(
            _hc("shooting_accuracy", "Shooting Accuracy", 1.15),
            _hc("shooting_power", "Shooting Power", 1.05),
            _hc("finishing", "Finishing", 1.25),
            _hc("shot_selection", "Shot Selection", 1.1),
            _hc("passing", "Passing", 1.0),
            _hc("offensive_vision", "Offensive Vision", 1.05),
            _hc("playmaking", "Playmaking", 1.1),
            _hc("puck_skills", "Puck Skills", 0.95),
            _hc("offensive_awareness", "Offensive Awareness", 1.0),
            _hc("net_front", "Net-Front Ability", 0.9),
            _hc("scoring_instincts", "Scoring Instincts", 1.15),
            _hc("one_timer", "One-Timer Ability", 0.85),
            _hc("rebound_finishing", "Rebound Finishing", 0.9),
            _hc("creativity", "Offensive Creativity", 0.95),
            _hc("offensive_positioning", "Offensive Positioning", 1.0),
            _hc("rush_offence", "Rush Offence", 0.9),
            _hc("cycle_offence", "Cycle Offence", 0.85),
            _hc("pp_offence", "Power-Play Offence", 0.75),
        ),
    ),
    ChapterDefinition(
        id="defence",
        label="Defence",
        hidden=(
            _hc("defensive_positioning", "Defensive Positioning", 1.15),
            _hc("stick_checking", "Stick Checking", 1.05),
            _hc("gap_control", "Gap Control", 1.0),
            _hc("lane_denial", "Lane Denial", 0.95),
            _hc("shot_suppression", "Shot Suppression", 1.1),
            _hc("passing_lane_recognition", "Passing-Lane Recognition", 0.95),
            _hc("defensive_awareness", "Defensive Awareness", 1.1),
            _hc("board_defence", "Board Defence", 0.85),
            _hc("net_front_defence", "Net-Front Defence", 0.9),
            _hc("rush_defence", "Rush Defence", 0.95),
            _hc("cycle_defence", "Cycle Defence", 0.9),
            _hc("takeaway_ability", "Takeaway Ability", 0.85),
            _hc("pk_ability", "Penalty-Kill Ability", 0.8),
            _hc("defensive_recovery", "Defensive Recovery", 0.9),
            _hc("containment", "Containment", 1.0),
            _hc("opponent_tracking", "Opponent Tracking", 0.95),
        ),
    ),
    ChapterDefinition(
        id="transition",
        label="Transition",
        hidden=(
            _hc("acceleration", "Acceleration", 1.05),
            _hc("top_speed", "Top Speed", 0.95),
            _hc("agility", "Agility", 0.95),
            _hc("puck_carrying", "Puck Carrying", 1.1),
            _hc("zone_exits", "Zone Exits", 1.15),
            _hc("zone_entries", "Zone Entries", 1.15),
            _hc("first_pass", "First-Pass Ability", 1.0),
            _hc("breakout_passing", "Breakout Passing", 1.1),
            _hc("controlled_entry", "Controlled Entry", 1.05),
            _hc("controlled_exit", "Controlled Exit", 1.05),
            _hc("rush_creation", "Rush Creation", 0.95),
            _hc("retrieval_to_breakout", "Retrieval-to-Breakout", 0.9),
            _hc("pressure_escape", "Pressure Escape", 0.95),
            _hc("neutral_zone", "Neutral-Zone Effectiveness", 0.9),
            _hc("skating_with_possession", "Skating w/ Possession", 1.0),
            _hc("skating_without_possession", "Skating w/o Possession", 0.85),
        ),
    ),
    ChapterDefinition(
        id="mental",
        label="Mental",
        hidden=(
            _hc("hockey_iq", "Hockey IQ", 1.15),
            _hc("decision_making", "Decision Making", 1.15),
            _hc("anticipation", "Anticipation", 1.05),
            _hc("composure", "Composure", 1.1),
            _hc("consistency", "Consistency", 1.0),
            _hc("adaptability", "Adaptability", 0.9),
            _hc("pressure_performance", "Pressure Performance", 1.0),
            _hc("discipline", "Discipline", 0.95),
            _hc("situational_awareness", "Situational Awareness", 1.0),
            _hc("reaction_speed", "Reaction Speed", 0.85),
            _hc("tactical_understanding", "Tactical Understanding", 0.95),
            _hc("risk_management", "Risk Management", 0.9),
            _hc("mistake_frequency", "Mistake Avoidance", 1.0),
            _hc("late_game_decisions", "Late-Game Decisions", 0.85),
        ),
    ),
    ChapterDefinition(
        id="physical",
        label="Physical",
        hidden=(
            _hc("strength", "Strength", 1.1),
            _hc("body_checking", "Body Checking", 0.95),
            _hc("balance", "Balance", 0.95),
            _hc("board_battles", "Board Battles", 0.9),
            _hc("puck_protection", "Puck Protection", 0.95),
            _hc("net_front_strength", "Net-Front Strength", 0.85),
            _hc("physical_intimidation", "Physical Intimidation", 0.75),
            _hc("durability", "Durability", 0.8),
            _hc("physical_engagement", "Physical Engagement", 0.9),
            _hc("contact_resistance", "Contact Resistance", 0.85),
            _hc("forechecking_pressure", "Forechecking Pressure", 0.85),
            _hc("physical_defence", "Physical Defence", 0.9),
        ),
    ),
    ChapterDefinition(
        id="potential",
        label="Potential",
        hidden=(
            _hc("ceiling", "Ceiling", 1.2),
            _hc("development_speed", "Development Speed", 1.0),
            _hc("development_consistency", "Development Consistency", 0.9),
            _hc("bust_probability", "Bust Avoidance", 0.85),
            _hc("breakout_probability", "Breakout Probability", 0.85),
            _hc("late_bloomer", "Late-Bloomer Tendency", 0.7),
            _hc("physical_maturation", "Physical Maturation", 0.75),
            _hc("skill_growth", "Skill Growth", 1.0),
            _hc("mental_growth", "Mental Growth", 0.9),
            _hc("adaptability_growth", "Adaptability", 0.85),
            _hc("environment_sensitivity", "Environment Sensitivity", 0.7),
        ),
    ),
)

GOALIE_CHAPTERS: Tuple[ChapterDefinition, ...] = (
    ChapterDefinition(
        id="glove",
        label="Glove",
        goalie_only=True,
        hidden=(
            _hc("glove_reaction", "Glove Reaction", 1.2),
            _hc("glove_positioning", "Glove Positioning", 1.1),
            _hc("glove_tracking", "Glove Tracking", 1.15),
            _hc("high_glove", "High Glove Saves", 0.9),
        ),
    ),
    ChapterDefinition(
        id="blocker",
        label="Blocker",
        goalie_only=True,
        hidden=(
            _hc("blocker_reaction", "Blocker Reaction", 1.15),
            _hc("blocker_positioning", "Blocker Positioning", 1.1),
            _hc("rebound_direction", "Rebound Direction", 1.0),
        ),
    ),
    ChapterDefinition(
        id="stick",
        label="Stick",
        goalie_only=True,
        hidden=(
            _hc("puck_handling_g", "Puck Handling", 1.0),
            _hc("poke_check", "Poke Check", 0.85),
            _hc("stick_saves", "Stick Saves", 1.1),
            _hc("passing_g", "Passing", 0.75),
            _hc("rebound_control_g", "Rebound Control", 1.05),
        ),
    ),
    ChapterDefinition(
        id="potential",
        label="Potential",
        goalie_only=True,
        hidden=(
            _hc("ceiling", "Ceiling", 1.2),
            _hc("development_speed", "Development Speed", 1.0),
            _hc("development_consistency", "Development Consistency", 0.9),
            _hc("bust_probability", "Bust Avoidance", 0.85),
            _hc("breakout_probability", "Breakout Probability", 0.85),
            _hc("late_bloomer", "Late-Bloomer Tendency", 0.7),
            _hc("skill_growth", "Skill Growth", 1.0),
            _hc("mental_growth", "Mental Growth", 0.9),
        ),
    ),
)

# Tendency presets bias hidden distribution within a chapter (not rigid archetypes).
TENDENCY_PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "scorer": {
        "offence": {
            "finishing": 1.28,
            "shot_selection": 1.22,
            "scoring_instincts": 1.20,
            "shooting_accuracy": 1.15,
            "shooting_power": 1.10,
            "offensive_positioning": 1.08,
            "passing": 0.88,
            "playmaking": 0.85,
            "creativity": 0.92,
        },
    },
    "playmaker": {
        "offence": {
            "passing": 1.28,
            "playmaking": 1.26,
            "offensive_vision": 1.22,
            "creativity": 1.15,
            "puck_skills": 1.08,
            "finishing": 0.86,
            "shot_selection": 0.88,
            "shooting_power": 0.90,
        },
    },
    "shutdown_defender": {
        "defence": {
            "defensive_positioning": 1.22,
            "gap_control": 1.15,
            "containment": 1.12,
            "shot_suppression": 1.10,
            "defensive_awareness": 1.08,
        },
        "transition": {
            "breakout_passing": 0.92,
            "zone_entries": 0.88,
            "rush_creation": 0.85,
        },
    },
    "puck_moving_defender": {
        "defence": {
            "defensive_awareness": 1.05,
            "takeaway_ability": 1.08,
        },
        "transition": {
            "breakout_passing": 1.25,
            "first_pass": 1.20,
            "zone_exits": 1.18,
            "puck_carrying": 1.12,
        },
    },
    "power_forward": {
        "physical": {
            "strength": 1.25,
            "net_front_strength": 1.20,
            "board_battles": 1.15,
            "body_checking": 1.10,
        },
        "offence": {
            "net_front": 1.18,
            "finishing": 1.08,
        },
    },
}

# Hidden component → legacy rating keys (weighted). Centralized adapter surface.
HIDDEN_TO_LEGACY_MAP: Dict[str, List[Tuple[str, float]]] = {
    # Offence
    "shooting_accuracy": [("off_wrist_shot_accuracy", 0.7), ("off_slap_shot_accuracy", 0.3)],
    "shooting_power": [("off_wrist_shot_power", 0.65), ("off_slap_shot_power", 0.35)],
    "finishing": [("off_finishing", 1.0)],
    "shot_selection": [("off_shot_iq", 0.8), ("off_offensive_awareness", 0.2)],
    "passing": [("pm_passing_accuracy", 0.45), ("pm_passing_vision", 0.35), ("pm_puck_distribution", 0.2)],
    "offensive_vision": [("pm_passing_vision", 0.6), ("pm_offensive_read", 0.4)],
    "playmaking": [("pm_playmaking_creativity", 0.5), ("pm_assist_instinct", 0.5)],
    "puck_skills": [("pc_stickhandling", 0.5), ("pc_puck_control", 0.5)],
    "offensive_awareness": [("off_offensive_awareness", 1.0)],
    "net_front": [("off_net_front_presence", 1.0)],
    "scoring_instincts": [("off_finishing", 0.5), ("off_offensive_awareness", 0.5)],
    "one_timer": [("off_one_timer", 1.0)],
    "rebound_finishing": [("off_rebound_control_off", 1.0)],
    "creativity": [("off_creativity", 0.6), ("pm_playmaking_creativity", 0.4)],
    "offensive_positioning": [("off_offensive_awareness", 0.6), ("off_puck_placement", 0.4)],
    "rush_offence": [("off_offensive_awareness", 0.5), ("skg_transition_speed", 0.5)],
    "cycle_offence": [("pm_tempo_control", 0.5), ("pc_puck_protection", 0.5)],
    "pp_offence": [("off_one_timer", 0.4), ("pm_passing_vision", 0.3), ("off_finishing", 0.3)],
    # Defence
    "defensive_positioning": [("def_body_positioning", 0.6), ("def_defensive_awareness", 0.4)],
    "stick_checking": [("def_stick_checking", 1.0)],
    "gap_control": [("def_gap_control", 1.0)],
    "lane_denial": [("def_interception_skill", 0.6), ("def_containment_ability", 0.4)],
    "shot_suppression": [("def_pressure_defense", 0.5), ("def_shot_blocking", 0.5)],
    "passing_lane_recognition": [("def_interception_skill", 0.7), ("def_defensive_reads", 0.3)],
    "defensive_awareness": [("def_defensive_awareness", 1.0)],
    "board_defence": [("def_board_battles", 1.0)],
    "net_front_defence": [("def_net_coverage", 1.0)],
    "rush_defence": [("def_backchecking_effort", 0.5), ("def_gap_control", 0.5)],
    "cycle_defence": [("def_containment_ability", 0.6), ("def_pressure_defense", 0.4)],
    "takeaway_ability": [("def_stick_checking", 0.5), ("def_interception_skill", 0.5)],
    "pk_ability": [("def_pk_awareness", 1.0)],
    "defensive_recovery": [("def_backchecking_effort", 0.6), ("def_defensive_consistency", 0.4)],
    "containment": [("def_containment_ability", 1.0)],
    "opponent_tracking": [("def_defensive_reads", 0.6), ("def_defensive_iq", 0.4)],
    # Transition
    "acceleration": [("skg_acceleration", 1.0)],
    "top_speed": [("skg_speed", 0.7), ("skg_top_speed_control", 0.3)],
    "agility": [("skg_agility", 0.7), ("skg_edge_work", 0.3)],
    "puck_carrying": [("pc_puck_control", 0.6), ("pc_stickhandling", 0.4)],
    "zone_exits": [("skg_transition_speed", 0.4), ("pm_puck_distribution", 0.3), ("pc_puck_control", 0.3)],
    "zone_entries": [("skg_speed", 0.35), ("pc_deking", 0.35), ("pc_puck_control", 0.3)],
    "first_pass": [("pm_passing_accuracy", 0.6), ("pm_decision_making", 0.4)],
    "breakout_passing": [("pm_passing_vision", 0.5), ("pm_puck_distribution", 0.5)],
    "controlled_entry": [("pc_tight_space_control", 0.5), ("skg_agility", 0.5)],
    "controlled_exit": [("pm_decision_making", 0.5), ("pc_control_under_pressure", 0.5)],
    "rush_creation": [("skg_explosiveness", 0.5), ("off_offensive_awareness", 0.5)],
    "retrieval_to_breakout": [("pm_reaction_time", 0.5), ("pm_passing_accuracy", 0.5)],
    "pressure_escape": [("pc_control_under_pressure", 0.6), ("skg_agility", 0.4)],
    "neutral_zone": [("iqm_game_sense", 0.5), ("pm_offensive_anticipation", 0.5)],
    "skating_with_possession": [("pc_puck_control", 0.5), ("skg_balance_skating", 0.5)],
    "skating_without_possession": [("skg_speed", 0.5), ("skg_stride_efficiency", 0.5)],
    # Physical
    "strength": [("phy_strength", 1.0)],
    "body_checking": [("phy_checking", 0.7), ("phy_physicality", 0.3)],
    "balance": [("phy_balance", 1.0)],
    "board_battles": [("def_board_battles", 0.5), ("phy_strength", 0.5)],
    "puck_protection": [("pc_puck_protection", 1.0)],
    "net_front_strength": [("phy_strength", 0.6), ("off_net_front_presence", 0.4)],
    "physical_intimidation": [("phy_aggression", 0.6), ("phy_physicality", 0.4)],
    "durability": [("phy_durability", 0.5), ("phy_injury_resistance", 0.5)],
    "physical_engagement": [("phy_aggression", 0.5), ("phy_checking", 0.5)],
    "contact_resistance": [("phy_balance", 0.5), ("phy_strength", 0.5)],
    "forechecking_pressure": [("phy_aggression", 0.4), ("def_pressure_defense", 0.6)],
    "physical_defence": [("phy_checking", 0.5), ("def_board_battles", 0.5)],
    # Mental
    "hockey_iq": [("iqm_hockey_iq", 1.0)],
    "decision_making": [("pm_decision_making", 0.5), ("iqm_game_sense", 0.5)],
    "anticipation": [("pm_offensive_anticipation", 0.5), ("def_defensive_reads", 0.5)],
    "composure": [("iqm_composure", 1.0)],
    "consistency": [("iqm_consistency", 0.7), ("def_defensive_consistency", 0.3)],
    "adaptability": [("iqm_adaptability", 1.0)],
    "pressure_performance": [("st_pressure_handling", 0.6), ("iqm_clutch_factor", 0.4)],
    "discipline": [("iqm_discipline", 1.0)],
    "situational_awareness": [("iqm_awareness", 0.6), ("iqm_game_sense", 0.4)],
    "reaction_speed": [("pm_reaction_time", 1.0)],
    "tactical_understanding": [("iqm_hockey_iq", 0.6), ("def_defensive_iq", 0.4)],
    "risk_management": [("iqm_discipline", 0.5), ("pm_decision_making", 0.5)],
    "mistake_frequency": [("iqm_consistency", 0.6), ("iqm_focus", 0.4)],
    "late_game_decisions": [("iqm_clutch_factor", 0.6), ("iqm_composure", 0.4)],
    # Character
    "leadership": [("per_leadership", 1.0)],
    "professionalism": [("per_professionalism", 1.0)],
    "work_ethic": [("dev_work_ethic", 1.0)],
    "coachability": [("dev_coachability", 1.0)],
    "loyalty": [("per_team_chemistry", 0.6), ("per_emotional_stability", 0.4)],
    "competitiveness": [("per_leadership", 0.4), ("iqm_confidence", 0.6)],
    "accountability": [("per_professionalism", 0.6), ("per_leadership", 0.4)],
    "selflessness": [("per_team_chemistry", 1.0)],
    "temperament": [("per_emotional_stability", 1.0)],
    "locker_room_presence": [("st_locker_room_impact", 0.6), ("per_leadership", 0.4)],
    "resilience": [("per_emotional_stability", 0.6), ("iqm_adaptability", 0.4)],
    "ambition": [("dev_growth_rate", 0.5), ("iqm_confidence", 0.5)],
    "ego": [("per_media_handling", 0.5), ("per_emotional_stability", 0.5)],
    "emotional_stability": [("per_emotional_stability", 1.0)],
    "mentoring": [("per_leadership", 0.7), ("dev_coachability", 0.3)],
    "adversity_response": [("per_emotional_stability", 0.5), ("st_pressure_handling", 0.5)],
    # Potential
    "ceiling": [("dev_potential", 1.0)],
    "development_speed": [("dev_growth_rate", 1.0)],
    "development_consistency": [("dev_learning_ability", 0.6), ("iqm_consistency", 0.4)],
    "bust_probability": [("dev_potential", 0.5), ("dev_work_ethic", 0.5)],
    "breakout_probability": [("dev_potential", 0.6), ("dev_growth_rate", 0.4)],
    "late_bloomer": [("dev_growth_rate", 0.5), ("dev_learning_ability", 0.5)],
    "physical_maturation": [("phy_strength", 0.4), ("dev_growth_rate", 0.6)],
    "skill_growth": [("dev_growth_rate", 0.7), ("dev_learning_ability", 0.3)],
    "mental_growth": [("dev_learning_ability", 0.6), ("iqm_adaptability", 0.4)],
    "adaptability_growth": [("iqm_adaptability", 0.6), ("dev_coachability", 0.4)],
    "environment_sensitivity": [("dev_coachability", 0.5), ("per_team_chemistry", 0.5)],
    # Goalie
    "glove_reaction": [("g_reflexes", 0.7), ("g_athleticism", 0.3)],
    "glove_positioning": [("g_positioning", 1.0)],
    "glove_tracking": [("g_reflexes", 0.6), ("g_positioning", 0.4)],
    "high_glove": [("g_athleticism", 0.6), ("g_reflexes", 0.4)],
    "blocker_reaction": [("g_reflexes", 1.0)],
    "blocker_positioning": [("g_positioning", 1.0)],
    "rebound_direction": [("g_rebound_control_g", 1.0)],
    "puck_handling_g": [("g_athleticism", 0.5), ("pc_puck_control", 0.5)],
    "poke_check": [("g_athleticism", 0.6), ("def_stick_checking", 0.4)],
    "stick_saves": [("g_reflexes", 0.6), ("g_positioning", 0.4)],
    "passing_g": [("pm_passing_accuracy", 0.7), ("pm_decision_making", 0.3)],
    "rebound_control_g": [("g_rebound_control_g", 1.0)],
}


def player_type_for_position(position: Any) -> str:
    pos = str(getattr(position, "value", position) or "F").strip().upper()
    return "goalie" if pos == "G" else "skater"


def get_chapter_definitions(player_type: str) -> Tuple[ChapterDefinition, ...]:
    return GOALIE_CHAPTERS if str(player_type).lower() == "goalie" else SKATER_CHAPTERS


def get_visible_chapter_ids(player_type: str) -> Tuple[str, ...]:
    return GOALIE_CHAPTER_IDS if str(player_type).lower() == "goalie" else SKATER_CHAPTER_IDS


def _clamp_rating(value: float) -> int:
    return int(clamp_rating(value))


def _chapter_by_id(player_type: str) -> Dict[str, ChapterDefinition]:
    return {ch.id: ch for ch in get_chapter_definitions(player_type)}


def aggregate_chapter_score(
    chapter_id: str,
    hidden: Mapping[str, float],
    *,
    player_type: str = "skater",
) -> Optional[float]:
    """Weighted aggregate of hidden components → chapter score (not a flat mean)."""
    chapter = _chapter_by_id(player_type).get(str(chapter_id))
    if chapter is None:
        return None
    total_w = 0.0
    total_v = 0.0
    for comp in chapter.hidden:
        val = hidden.get(comp.id)
        if val is None:
            continue
        w = float(comp.aggregate_weight)
        total_w += w
        total_v += float(val) * w
    if total_w <= 0:
        return None
    return total_v / total_w


def _resolve_tendency_multipliers(
    chapter_id: str,
    tendencies: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """Blend tendency presets into per-hidden multipliers."""
    if not tendencies:
        return {}
    out: Dict[str, float] = {}
    for tendency, strength in tendencies.items():
        preset = TENDENCY_PRESETS.get(str(tendency), {})
        chapter_bias = preset.get(str(chapter_id), {})
        s = float(strength)
        for hid, mult in chapter_bias.items():
            # Blend toward preset: 1.0 = neutral, mult = full preset at strength 1.
            delta = (float(mult) - 1.0) * s
            out[hid] = out.get(hid, 0.0) + delta
    return out


def _correlated_noise(rng: random.Random, primary: float, correlation: float) -> float:
    """Return noise correlated with a primary value (controlled, not identical)."""
    independent = rng.uniform(-1.0, 1.0)
    return primary * correlation + independent * (1.0 - abs(correlation))


def generate_hidden_profile(
    chapters: Mapping[str, float],
    *,
    position: str = "F",
    player_type: Optional[str] = None,
    tendencies: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    preserve_weaknesses: bool = True,
) -> Dict[str, float]:
    """
    Generate a coherent hidden distribution from chapter targets.

    Chapter scores establish talent level; tendencies + noise establish identity.
    Does NOT flatten players or force uniform hidden values.
    """
    ptype = player_type or player_type_for_position(position)
    rng = random.Random(int(seed if seed is not None else random.randint(1, 2_000_000_000)))
    hidden: Dict[str, float] = {}
    chapter_defs = _chapter_by_id(ptype)

    for chapter_id, target in chapters.items():
        if str(chapter_id).lower() == "overall":
            continue
        chapter = chapter_defs.get(str(chapter_id))
        if chapter is None:
            continue
        target_f = float(target)
        bias = _resolve_tendency_multipliers(str(chapter_id), tendencies)
        comp_values: List[Tuple[str, float, float]] = []
        for comp in chapter.hidden:
            mult = 1.0 + float(bias.get(comp.id, 0.0))
            # Spread around target; stronger components vary a bit more for identity.
            spread = rng.uniform(4.0, 11.0) * (0.85 + 0.15 * float(comp.aggregate_weight))
            direction = _correlated_noise(rng, rng.choice([-1.0, 1.0]), 0.35)
            raw = target_f * mult + direction * spread
            if preserve_weaknesses and mult < 0.95:
                raw = min(raw, target_f + spread * 0.35)
            if preserve_weaknesses and mult > 1.05:
                raw = max(raw, target_f - spread * 0.20)
            comp_values.append((comp.id, float(comp.aggregate_weight), raw))

        # Light validation: nudge one strong component so aggregate ≈ target (±2).
        if comp_values:
            trial = {cid: val for cid, _, val in comp_values}
            agg = aggregate_chapter_score(str(chapter_id), trial, player_type=ptype)
            if agg is not None and abs(agg - target_f) > 2.5:
                adjust_id = max(comp_values, key=lambda row: row[1])[0]
                delta = target_f - float(agg)
                for i, (cid, wt, val) in enumerate(comp_values):
                    if cid == adjust_id:
                        comp_values[i] = (cid, wt, val + delta)
                        break

        for cid, _, val in comp_values:
            hidden[cid] = float(_clamp_rating(val))

    return hidden


def build_attribute_profile(
    chapters: Mapping[str, float],
    *,
    position: str = "F",
    player_type: Optional[str] = None,
    tendencies: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    overall_source: str = "assigned",
    preserve_weaknesses: bool = True,
) -> Dict[str, Any]:
    """Build a full attribute profile dict suitable for Player.attribute_profile."""
    ptype = player_type or player_type_for_position(position)
    hidden = generate_hidden_profile(
        chapters,
        position=position,
        player_type=ptype,
        tendencies=tendencies,
        seed=seed,
        preserve_weaknesses=preserve_weaknesses,
    )
    derived: Dict[str, int] = {}
    for chapter_id in get_visible_chapter_ids(ptype):
        if chapter_id == "overall":
            continue
        if chapter_id in chapters:
            derived[chapter_id] = int(round(float(chapters[chapter_id])))
            continue
        agg = aggregate_chapter_score(chapter_id, hidden, player_type=ptype)
        if agg is not None:
            derived[chapter_id] = _clamp_rating(agg)

    overall_val = chapters.get("overall")
    if overall_val is None:
        overall_val = chapters.get("Overall")
    profile = {
        "schema_version": SCHEMA_VERSION,
        "player_type": ptype,
        "position": str(position).upper(),
        "chapters": {str(k).lower(): int(round(float(v))) for k, v in chapters.items()},
        "derived_chapters": derived,
        "hidden": hidden,
        "tendencies": {str(k): float(v) for k, v in (tendencies or {}).items()},
        "generation_seed": int(seed) if seed is not None else None,
        "overall_source": str(overall_source or "assigned"),
    }
    if overall_val is not None:
        profile["chapters"]["overall"] = _clamp_rating(float(overall_val))
    return profile


def detect_emergent_tendencies(hidden: Mapping[str, float]) -> Dict[str, float]:
    """
    Soft classification from hidden attributes — not rigid archetype assignment.
    Returns tendency strengths (0–1+) for scouting/sim hints.
    """
    h = dict(hidden or {})

    def _avg(*keys: str) -> float:
        vals = [float(h.get(k, 0) or 0) for k in keys if k in h]
        return sum(vals) / max(1, len(vals))

    offence = _avg("finishing", "shot_selection", "scoring_instincts", "shooting_accuracy")
    passing = _avg("passing", "playmaking", "offensive_vision", "creativity")
    defence = _avg("defensive_positioning", "gap_control", "containment", "shot_suppression")
    transition = _avg("zone_exits", "zone_entries", "breakout_passing", "puck_carrying")
    physical = _avg("strength", "body_checking", "board_battles", "net_front_strength")
    mental = _avg("hockey_iq", "decision_making", "composure", "consistency")

    tendencies: Dict[str, float] = {}
    if offence > 0 and passing > 0:
        tendencies["scorer"] = max(0.0, (offence - passing) / 20.0)
        tendencies["playmaker"] = max(0.0, (passing - offence) / 20.0)
    if defence > 0 and transition > 0:
        tendencies["shutdown_defender"] = max(0.0, (defence - transition) / 20.0)
        tendencies["puck_moving_defender"] = max(0.0, (transition - defence) / 20.0)
    if physical > 0 and offence > 0:
        tendencies["power_forward"] = max(0.0, (physical + offence * 0.5) / 99.0)
    if mental > 0:
        tendencies["high_iq"] = mental / 99.0
    return tendencies


def legacy_ratings_from_hidden(hidden: Mapping[str, float]) -> Dict[str, float]:
    """Project hidden profile onto legacy prefixed rating keys."""
    accum: Dict[str, float] = {}
    counts: Dict[str, float] = {}
    for hid, value in (hidden or {}).items():
        for legacy_key, weight in HIDDEN_TO_LEGACY_MAP.get(str(hid), []):
            accum[legacy_key] = accum.get(legacy_key, 0.0) + float(value) * float(weight)
            counts[legacy_key] = counts.get(legacy_key, 0.0) + float(weight)
    out: Dict[str, float] = {}
    for key, total in accum.items():
        denom = counts.get(key, 1.0)
        out[key] = float(_clamp_rating(total / max(0.01, denom)))
    return out


def sync_legacy_ratings_from_profile(player: Any, *, overwrite: bool = True) -> Dict[str, float]:
    """
    Write legacy player.ratings from attribute_profile.hidden.

    Only affects players with an attribute_profile. Does not touch others.
    """
    profile = getattr(player, "attribute_profile", None)
    if not isinstance(profile, dict):
        return {}
    hidden = profile.get("hidden") or {}
    if not hidden:
        return {}
    projected = legacy_ratings_from_hidden(hidden)
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict):
        return projected
    for key, val in projected.items():
        if key not in ATTRIBUTE_KEYS:
            continue
        if overwrite or key not in ratings or ratings.get(key) in (None, 0):
            ratings[key] = float(val)
    return projected


def ensure_player_attribute_profile(
    player: Any,
    *,
    chapters: Optional[Mapping[str, float]] = None,
    tendencies: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    regenerate: bool = False,
) -> Dict[str, Any]:
    """Attach or return attribute_profile on a player object."""
    existing = getattr(player, "attribute_profile", None)
    if isinstance(existing, dict) and not regenerate:
        return existing
    pos = getattr(getattr(player, "identity", None), "position", None)
    pos_str = str(getattr(pos, "value", pos) or getattr(player, "position", "F") or "F")
    ptype = player_type_for_position(pos_str)
    chapter_input = dict(chapters or {})
    if not chapter_input:
        chapter_input = dict((existing or {}).get("chapters") or {})
    if not chapter_input:
        raise ValueError("chapters required to build attribute_profile")
    overall = getattr(player, "overall", None)
    if overall is not None and "overall" not in chapter_input:
        try:
            chapter_input["overall"] = int(overall)
        except (TypeError, ValueError):
            pass
    if seed is None and isinstance(existing, dict):
        seed = existing.get("generation_seed")
    profile = build_attribute_profile(
        chapter_input,
        position=pos_str,
        player_type=ptype,
        tendencies=tendencies or (existing or {}).get("tendencies"),
        seed=seed,
        overall_source=str((existing or {}).get("overall_source") or "assigned"),
    )
    setattr(player, "attribute_profile", profile)
    return profile


def get_player_chapters(player: Any) -> Dict[str, int]:
    """Visible chapter ratings for UI/API."""
    profile = getattr(player, "attribute_profile", None)
    if isinstance(profile, dict):
        chapters = dict(profile.get("chapters") or {})
        derived = profile.get("derived_chapters") or {}
        for key, val in derived.items():
            chapters.setdefault(str(key), int(val))
        return {str(k): int(v) for k, v in chapters.items()}
    return {}


def serialize_chapter_profile_for_api(player: Any) -> Optional[Dict[str, Any]]:
    """Compact chapter payload for API rows — None when profile absent."""
    profile = getattr(player, "attribute_profile", None)
    if not isinstance(profile, dict):
        return None
    chapters = get_player_chapters(player)
    hidden = profile.get("hidden") or {}
    return {
        "schema_version": int(profile.get("schema_version") or SCHEMA_VERSION),
        "player_type": str(profile.get("player_type") or "skater"),
        "chapters": chapters,
        "overall_source": str(profile.get("overall_source") or "assigned"),
        "tendencies": dict(profile.get("tendencies") or {}),
        "emergent_tendencies": detect_emergent_tendencies(hidden),
        "hidden_count": len(hidden),
    }


def chapter_schema_export() -> Dict[str, Any]:
    """Export full schema for tooling / future frontend scouting panels."""

    def _pack(chapters: Sequence[ChapterDefinition]) -> List[Dict[str, Any]]:
        return [
            {
                "id": ch.id,
                "label": ch.label,
                "hidden": [{"id": c.id, "label": c.label, "weight": c.aggregate_weight} for c in ch.hidden],
            }
            for ch in chapters
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "skater_chapters": _pack(SKATER_CHAPTERS),
        "goalie_chapters": _pack(GOALIE_CHAPTERS),
        "tendency_presets": sorted(TENDENCY_PRESETS.keys()),
    }


def profile_deepcopy(profile: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(profile))
