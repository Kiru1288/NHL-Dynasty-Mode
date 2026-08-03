# app/sim_engine/entities/player.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import random


# ============================================================
# ENUMS
# ============================================================

class Position(str, Enum):
    C = "C"
    LW = "LW"
    RW = "RW"
    D = "D"
    G = "G"


class Shoots(str, Enum):
    L = "L"
    R = "R"


class BackstoryType(str, Enum):
    PRODIGY = "prodigy"
    LATE_BLOOMER = "late_bloomer"
    GRINDER = "grinder"
    PROJECT = "project_player"
    BUST_SURVIVOR = "bust_survivor"
    COMEBACK = "comeback_story"


class UpbringingType(str, Enum):
    PRIVILEGED = "privileged"
    STABLE_MIDDLE_CLASS = "stable_middle_class"
    WORKING_CLASS = "working_class"
    ROUGH = "rough"
    EXTREME_ADVERSITY = "extreme_adversity"


class SupportLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PressureLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    INTENSE = "intense"


class DevResources(str, Enum):
    ELITE = "elite_academies"
    LOCAL = "local_clubs"
    UNDERFUNDED = "underfunded_programs"


class InjuryStatus(str, Enum):
    HEALTHY = "healthy"
    DAY_TO_DAY = "day_to_day"
    INJURED = "injured"


class CareerArcType(str, Enum):
    STEADY = "steady"
    PEAK_EARLY = "peak_early"
    LATE_BLOOM = "late_bloom"
    VOLATILE = "volatile"


# ============================================================
# UTILITIES / LEAGUE RATING TARGETS
# ============================================================

RATING_MIN = 20
RATING_MAX = 99

# NHL active-roster balance targets.
# Ratings are 20–99. Player.ovr() returns 0.0–1.0.
NHL_MIN_ACTIVE_OVR = 74.0
NHL_SUPERSTAR_OVR = 90.0
NHL_TARGET_SUPERSTARS_90_PLUS = 15

# Context-sensitive OVR floors (0–100 scale). Non-NHL pools must not inherit the NHL floor.
POOL_OVR_FLOORS: Dict[str, float] = {
    "nhl": 72.0,
    "ahl": 60.0,
    "echl": 52.0,
    "ufa": 55.0,
    "overseas": 55.0,
    "junior": 38.0,
    "college": 42.0,
    "european_junior": 45.0,
    "none": 0.0,
}

# Missing generated ratings should produce usable NHL depth, not 68 OVR sludge.
DEFAULT_NHL_RATING = 74


def get_ovr_floor_for_pool(pool_context: Optional[str]) -> float:
    """Return the minimum OVR (0–100) for a player pool; 0 disables the floor."""
    key = str(pool_context or "nhl").strip().lower()
    if key in POOL_OVR_FLOORS:
        return float(POOL_OVR_FLOORS[key])
    if key.startswith("eu_") or key.startswith("chl"):
        return float(POOL_OVR_FLOORS.get("junior", 38.0))
    if key in ("ncaa", "ushl"):
        return float(POOL_OVR_FLOORS.get("college", 42.0))
    return float(POOL_OVR_FLOORS.get("nhl", NHL_MIN_ACTIVE_OVR))


def clamp_rating(x: float) -> int:
    return int(max(RATING_MIN, min(RATING_MAX, round(float(x)))))


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return float(x)


def normalize_rating(value: Any) -> float:
    """Return a clamped internal 0.0–1.0 rating.

    Accepts legacy 0–1 fractions and 0–99 / 20–99 display-scale values.
    Values in (1.5, 20) are treated as malformed display fragments and scaled to 0–1.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not (v == v):  # NaN
        return 0.0
    if v <= 1.5:
        return clamp01(v)
    # Display / attribute-adjacent scales → canonical 0–1.
    if v <= 99.0:
        return clamp01(v / 99.0)
    return clamp01(v / 100.0)


def display_rating(value: Any) -> int:
    """Convert canonical internal rating to a displayed 0–99 integer."""
    return int(max(0, min(99, round(normalize_rating(value) * 99.0))))


def normalize_rating_gap(current: Any, ceiling: Any) -> float:
    """Return a safe non-negative gap using the canonical 0–1 scale."""
    return max(0.0, normalize_rating(ceiling) - normalize_rating(current))


def player_current_ovr_01(player: Any) -> float:
    """Authoritative current ability on 0–1: prefer compute_ovr from attributes."""
    ratings = getattr(player, "ratings", None)
    if isinstance(ratings, dict) and ratings:
        ovr_fn = getattr(player, "ovr", None)
        if callable(ovr_fn):
            try:
                return normalize_rating(ovr_fn())
            except Exception:
                pass
        pos = getattr(player, "position", None)
        arch = getattr(player, "archetype", None)
        try:
            return normalize_rating(compute_ovr(ratings, pos, arch))
        except Exception:
            pass
    for key in ("overall", "ovr", "true_ovr", "current_ovr"):
        raw = getattr(player, key, None)
        if raw is None:
            continue
        if callable(raw):
            try:
                raw = raw()
            except Exception:
                continue
        return normalize_rating(raw)
    return 0.55


def persist_recomputed_ovr(player: Any) -> float:
    """Recompute OVR from attributes and persist both 0–1 and display mirrors.

    Direct OVR mutation outside this helper (or controlled migration) is forbidden.
    """
    ovr01 = player_current_ovr_01(player)
    # Sync display mirrors from attributes — never invent overall independently.
    try:
        setattr(player, "overall", float(display_rating(ovr01)))
    except Exception:
        pass
    try:
        # Some legacy fixtures store ovr as a float attribute, not a method. Check the
        # instance attribute (not the type) — plain objects/SimpleNamespaces never have
        # class-level attributes, so `type(player).ovr` is always None for them even when
        # the instance itself already holds a callable (e.g. `ovr=lambda: 0.6`), which
        # used to overwrite that callable with a float and break every later `player.ovr()`.
        existing = getattr(player, "ovr", None)
        if existing is not None and not callable(existing):
            setattr(player, "ovr", float(ovr01))
    except Exception:
        pass
    try:
        inval = getattr(player, "_invalidate_ovr_memo", None)
        if callable(inval):
            inval()
    except Exception:
        pass
    return ovr01


def validate_stored_ovr_matches_compute(
    player: Any,
    *,
    allowed_rounding_tolerance: float = 0.015,
) -> Dict[str, Any]:
    """Confirm stored OVR matches attribute-derived compute_ovr within tolerance (0–1)."""
    computed = player_current_ovr_01(player)
    stored = None
    ovr_fn = getattr(player, "ovr", None)
    if callable(ovr_fn):
        try:
            stored = normalize_rating(ovr_fn())
        except Exception:
            stored = None
    if stored is None:
        stored = normalize_rating(getattr(player, "overall", None) or getattr(player, "true_ovr", None) or computed)
    delta = abs(float(stored) - float(computed))
    return {
        "ok": delta <= float(allowed_rounding_tolerance),
        "stored_ovr": stored,
        "computed_ovr": computed,
        "delta": delta,
        "tolerance": float(allowed_rounding_tolerance),
    }


def normalize_ratings_dict(r, keys, default=DEFAULT_NHL_RATING):
    """
    Normalize ratings into the game's real 20–99 scale.

    Important:
    - Ratings are 20–99 here.
    - Player.ovr() converts weighted rating into 0–1.
    - Missing attributes default to 74 instead of 68 so generated NHL players
      do not become unusable depth by accident.
    - Meta keys (e.g. _generated_profile) are skipped — not numeric attributes.
    """
    out = {}
    src = r or {}
    for k in keys:
        if str(k).startswith("_"):
            continue
        raw = src.get(k, default)
        try:
            out[k] = clamp_rating(raw)
        except (TypeError, ValueError):
            out[k] = clamp_rating(default)
    return out


def height_cm_to_imperial(height_cm: int) -> str:
    """Feet/inches string; inches always 0–11."""
    if height_cm is None or int(height_cm) <= 0:
        return "—"
    total_in = int(round(float(height_cm) / 2.54))
    total_in = max(48, min(84, total_in))
    ft = total_in // 12
    inch = total_in % 12
    return f"{ft}'{inch}\""


def minimum_height_cm_for_position(position: Any) -> int:
    """Plausible NHL floor by position (cm)."""
    p = str(getattr(position, "value", position) or "").upper()
    if p == "G":
        return 183
    if p in ("D", "LD", "RD", "LHD", "RHD"):
        return 178
    return 170


def maximum_height_cm_for_position(position: Any) -> int:
    """Plausible NHL ceiling by position (cm)."""
    p = str(getattr(position, "value", position) or "").upper()
    if p == "G":
        return 204
    if p in ("D", "LD", "RD", "LHD", "RHD"):
        return 201
    return 198


def clamp_height_cm_for_position(height_cm: Any, position: Any) -> int:
    """Clamp stored height into a position-realistic band (no reroll)."""
    lo = minimum_height_cm_for_position(position)
    hi = maximum_height_cm_for_position(position)
    try:
        v = int(round(float(height_cm)))
    except (TypeError, ValueError):
        v = 0
    if v <= 0:
        return lo
    return max(lo, min(hi, v))


def random_height_cm(rng: random.Random, position: Any = None) -> int:
    """Legacy helper — prefer generate_position_height_cm for new players."""
    if position is not None:
        from app.sim_engine.generation.prospect_body import generate_position_height_cm

        return generate_position_height_cm(rng, position)
    ft = rng.randint(5, 6)
    inch = rng.randint(7, 11) if ft == 5 else rng.randint(0, 6)
    return int(round((ft * 12 + inch) * 2.54))


def sanitize_height_cm(raw: Any, rng: random.Random, position: Any = None) -> int:
    """Clamp to plausible NHL cm; replace garbage with a valid position-aware height."""
    try:
        v = int(round(float(raw)))
    except (TypeError, ValueError):
        v = 0

    if position is not None:
        lo = minimum_height_cm_for_position(position)
        hi = maximum_height_cm_for_position(position)
        if v < lo or v > hi or v <= 0:
            return random_height_cm(rng, position)
        return v

    if v < 170 or v > 213:
        return random_height_cm(rng)
    return v


# ============================================================
# ATTRIBUTE KEYS — skater facets + goalie tools
# ============================================================

OFFENSE_ATTRS: List[str] = [
    "off_wrist_shot_accuracy",
    "off_wrist_shot_power",
    "off_slap_shot_accuracy",
    "off_slap_shot_power",
    "off_one_timer",
    "off_shot_iq",
    "off_shooting_under_pressure",
    "off_net_front_presence",
    "off_tip_deflection",
    "off_rebound_control_off",
    "off_offensive_awareness",
    "off_creativity",
    "off_deception",
    "off_puck_placement",
    "off_finishing",
]

PLAYMAKING_ATTRS: List[str] = [
    "pm_passing_accuracy",
    "pm_passing_vision",
    "pm_passing_speed",
    "pm_puck_distribution",
    "pm_offensive_read",
    "pm_decision_making",
    "pm_reaction_time",
    "pm_assist_instinct",
    "pm_give_and_go",
    "pm_playmaking_creativity",
    "pm_tempo_control",
    "pm_offensive_anticipation",
]

DEFENSE_ATTRS: List[str] = [
    "def_defensive_awareness",
    "def_stick_checking",
    "def_body_positioning",
    "def_shot_blocking",
    "def_gap_control",
    "def_defensive_iq",
    "def_interception_skill",
    "def_board_battles",
    "def_faceoffs",
    "def_net_coverage",
    "def_backchecking_effort",
    "def_defensive_reads",
    "def_pk_awareness",
    "def_pressure_defense",
    "def_containment_ability",
    "def_defensive_consistency",
]

PHYSICAL_ATTRS: List[str] = [
    "phy_strength",
    "phy_balance",
    "phy_checking",
    "phy_aggression",
    "phy_durability",
    "phy_stamina",
    "phy_endurance",
    "phy_injury_resistance",
    "phy_physicality",
    "phy_recovery_rate",
]

SKATING_ATTRS: List[str] = [
    "skg_speed",
    "skg_acceleration",
    "skg_agility",
    "skg_edge_work",
    "skg_balance_skating",
    "skg_transition_speed",
    "skg_explosiveness",
    "skg_pivot_speed",
    "skg_stride_efficiency",
    "skg_top_speed_control",
]

IQ_ATTRS: List[str] = [
    "iqm_hockey_iq",
    "iqm_awareness",
    "iqm_composure",
    "iqm_confidence",
    "iqm_consistency",
    "iqm_clutch_factor",
    "iqm_discipline",
    "iqm_focus",
    "iqm_adaptability",
    "iqm_game_sense",
]

SKILL_ATTRS: List[str] = [
    "pc_stickhandling",
    "pc_puck_control",
    "pc_deking",
    "pc_puck_protection",
    "pc_hand_eye_coordination",
    "pc_creativity_puck",
    "pc_tight_space_control",
    "pc_control_under_pressure",
]

DEV_ATTRS: List[str] = [
    "dev_potential",
    "dev_growth_rate",
    "dev_work_ethic",
    "dev_coachability",
    "dev_learning_ability",
]

PERSONALITY_ATTRS: List[str] = [
    "per_leadership",
    "per_professionalism",
    "per_team_chemistry",
    "per_media_handling",
    "per_emotional_stability",
]

SPECIAL_ATTRS: List[str] = [
    "st_big_game_performance",
    "st_rivalry_performance",
    "st_momentum_impact",
    "st_fan_influence",
    "st_locker_room_impact",
    "st_injury_proneness_inv",
    "st_suspension_risk",
    "st_consistency_variance",
    "st_pressure_handling",
    "st_dev_ceiling_modifier",
]

GOALIE_ATTRS: List[str] = [
    "g_reflexes",
    "g_positioning",
    "g_rebound_control_g",
    "g_athleticism",
]

ATTRIBUTE_KEYS: List[str] = (
    OFFENSE_ATTRS
    + PLAYMAKING_ATTRS
    + DEFENSE_ATTRS
    + PHYSICAL_ATTRS
    + SKATING_ATTRS
    + IQ_ATTRS
    + SKILL_ATTRS
    + DEV_ATTRS
    + PERSONALITY_ATTRS
    + SPECIAL_ATTRS
    + GOALIE_ATTRS
)

ALIASES = {
    # Centers: dedicated faceoff attribute (not board battles).
    "faceoff": "def_faceoffs",
    "faceoffs": "def_faceoffs",
    # Legacy unprefixed gameplay lookups → live attribute keys.
    "puck_control": "pc_puck_control",
    "puck_handling": "pc_stickhandling",
    "deking": "pc_deking",
    "speed": "skg_speed",
    "acceleration": "skg_acceleration",
    "agility": "skg_agility",
    "wrist_shot_accuracy": "off_wrist_shot_accuracy",
    "wrist_accuracy": "off_wrist_shot_accuracy",
    "slap_shot_accuracy": "off_slap_shot_accuracy",
    "slap_accuracy": "off_slap_shot_accuracy",
    "shot_power": "off_wrist_shot_power",
    "wrist_shot_power": "off_wrist_shot_power",
    "shot_blocking": "def_shot_blocking",
    "stick_checking": "def_stick_checking",
    "strength": "phy_strength",
    "aggression": "phy_aggression",
    "discipline": "iqm_discipline",
    "composure": "iqm_composure",
    "reflexes": "g_reflexes",
    "positioning": "g_positioning",
}

# Enum / legacy style labels → ARCHETYPE_CATEGORY_MULT keys (OVR leverage).
ARCHETYPE_ALIASES: Dict[str, str] = {
    "STAY_AT_HOME_DEFENSEMAN": "DEFENSIVE_D",
    "STAY_AT_HOME": "DEFENSIVE_D",
    "SHUTDOWN_DEFENSEMAN": "DEFENSIVE_D",
    "SHUTDOWN": "DEFENSIVE_D",
    "DEFENSIVE_DEFENSEMAN": "DEFENSIVE_D",
    "TWO_WAY_FORWARD": "TWO_WAY_F",
    "TWO_WAY_F": "TWO_WAY_F",
    "TWO_WAY_DEFENSEMAN": "TWO_WAY",
    "GRINDER": "GRINDER",
    "ENFORCER": "GRINDER",
    "ENFORCER_D": "GRINDER",
    "BUTTERFLY_G": "BUTTERFLY_G",
    "HYBRID_G": "HYBRID_G",
    "BALANCED_G": "BALANCED_G",
}

# Generation role profiles → OVR archetype strings (keeps rating shape + OVR leverage aligned).
GENERATION_PROFILE_TO_ARCHETYPE: Dict[str, str] = {
    "sniper": "SNIPER",
    "playmaker": "PLAYMAKER",
    "power_forward": "POWER_FORWARD",
    "grinder": "GRINDER",
    "two_way": "TWO_WAY_F",
    "two_way_d": "TWO_WAY",
    "defensive_d": "DEFENSIVE_D",
    "offensive_d": "OFFENSIVE_D",
    "enforcer_d": "GRINDER",
    "hybrid_g": "HYBRID_G",
    "butterfly_g": "BUTTERFLY_G",
    "balanced_g": "BALANCED_G",
}

# Legacy group names used by engine / chemistry / decay.
OFFENSE_KEYS = OFFENSE_ATTRS
PASSING_KEYS = PLAYMAKING_ATTRS
SKATING_KEYS = SKATING_ATTRS
DEFENSE_KEYS = DEFENSE_ATTRS
IQ_KEYS = IQ_ATTRS
PHYS_KEYS = PHYSICAL_ATTRS
SKILL_KEYS = SKILL_ATTRS
HIDDEN_KEYS = DEV_ATTRS
PERSONALITY_KEYS = PERSONALITY_ATTRS
SPECIAL_KEYS = SPECIAL_ATTRS
GOALIE_KEYS = GOALIE_ATTRS


# Current OVR should be current ability.
# Potential/growth exists, but it should not hard-cap current OVR.
OVR_CATEGORY_WEIGHTS_SKATER: Dict[str, float] = {
    "offense": 0.19,
    "playmaking": 0.15,
    "defense": 0.17,
    "skating": 0.15,
    "physical": 0.10,
    "iq": 0.13,
    "skill": 0.09,
    "hidden": 0.00,
    "personality": 0.01,
    "special": 0.01,
}


ARCHETYPE_CATEGORY_MULT: Dict[str, Dict[str, float]] = {
    "SNIPER": {
        "offense": 1.12,
        "playmaking": 0.96,
        "defense": 0.80,
        "skating": 1.02,
        "physical": 0.95,
        "iq": 1.00,
        "skill": 1.06,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "PLAYMAKER": {
        "offense": 1.02,
        "playmaking": 1.12,
        "defense": 0.85,
        "skating": 1.02,
        "physical": 0.80,
        "iq": 1.05,
        "skill": 1.08,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "DEFENSIVE_D": {
        "offense": 0.70,
        "playmaking": 0.85,
        "defense": 1.30,
        "skating": 1.00,
        "physical": 1.08,
        "iq": 1.05,
        "skill": 0.92,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "OFFENSIVE_D": {
        "offense": 1.15,
        "playmaking": 0.95,
        "defense": 0.90,
        "skating": 1.08,
        "physical": 0.92,
        "iq": 1.02,
        "skill": 1.05,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "POWER_FORWARD": {
        "offense": 1.10,
        "playmaking": 0.92,
        "defense": 0.88,
        "skating": 0.95,
        "physical": 1.30,
        "iq": 0.98,
        "skill": 1.02,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "TWO_WAY": {
        "offense": 1.00,
        "playmaking": 1.00,
        "defense": 1.15,
        "skating": 1.05,
        "physical": 1.02,
        "iq": 1.10,
        "skill": 1.00,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "TWO_WAY_F": {
        "offense": 1.00,
        "playmaking": 1.00,
        "defense": 1.15,
        "skating": 1.05,
        "physical": 1.02,
        "iq": 1.10,
        "skill": 1.00,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "GRINDER": {
        "offense": 0.82,
        "playmaking": 0.88,
        "defense": 1.18,
        "skating": 0.96,
        "physical": 1.28,
        "iq": 1.02,
        "skill": 0.90,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "ELITE_FRANCHISE": {},
    "BALANCED": {},
    "BALANCED_G": {},
    "BUTTERFLY_G": {
        "offense": 1.00,
        "playmaking": 1.00,
        "defense": 1.00,
        "skating": 0.96,
        "physical": 1.04,
        "iq": 1.06,
        "skill": 1.00,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
    "HYBRID_G": {
        "offense": 1.00,
        "playmaking": 1.00,
        "defense": 1.00,
        "skating": 1.08,
        "physical": 0.98,
        "iq": 1.04,
        "skill": 1.00,
        "hidden": 1.00,
        "personality": 1.00,
        "special": 1.00,
    },
}


def resolve_ovr_archetype_key(archetype: Optional[str]) -> str:
    """Map enum / legacy / generation labels onto ARCHETYPE_CATEGORY_MULT keys."""
    raw = str(archetype or "BALANCED").strip().upper()
    if not raw:
        return "BALANCED"
    if raw in ARCHETYPE_CATEGORY_MULT:
        return raw
    aliased = ARCHETYPE_ALIASES.get(raw)
    if aliased and (aliased in ARCHETYPE_CATEGORY_MULT or aliased == "ELITE_FRANCHISE"):
        return aliased
    return raw


def archetype_from_generation_profile(profile: Optional[str], position: Position) -> Optional[str]:
    """Convert build_role_shaped_ratings profile → OVR archetype string."""
    if not profile:
        return None
    key = str(profile).strip().lower()
    mapped = GENERATION_PROFILE_TO_ARCHETYPE.get(key)
    if mapped:
        return mapped
    pos = str(getattr(position, "value", position) or "").upper()
    if pos == "G":
        return "BALANCED_G"
    return None


def assign_skater_archetype(position: Position, rng: random.Random) -> str:
    if position == Position.G:
        return rng.choices(
            ["BALANCED_G", "BUTTERFLY_G", "HYBRID_G", "ELITE_FRANCHISE"],
            weights=[0.42, 0.28, 0.22, 0.08],
            k=1,
        )[0]

    if position == Position.D:
        return rng.choices(
            ["DEFENSIVE_D", "OFFENSIVE_D", "TWO_WAY", "BALANCED", "GRINDER"],
            weights=[0.32, 0.20, 0.26, 0.14, 0.08],
            k=1,
        )[0]

    return rng.choices(
        ["SNIPER", "PLAYMAKER", "POWER_FORWARD", "TWO_WAY_F", "GRINDER", "BALANCED", "ELITE_FRANCHISE"],
        weights=[0.16, 0.18, 0.14, 0.24, 0.12, 0.14, 0.02],
        k=1,
    )[0]


def _avg(ratings: Dict[str, Any], keys: List[str]) -> float:
    """Category average feeding compute_ovr.

    Kept as an unrounded float: rounding here (categories can span 10-20+ raw
    attributes) used to swallow any single-season progression/decline event, since a
    1-2 point nudge on one or two attributes moves a 15-key category average by well
    under 0.5 and used to vanish under int(round(...)) before it ever reached the
    weighted OVR sum. Individual attribute values in `ratings` are still stored/clamped
    as ints via clamp_rating(); only this aggregate stays continuous.
    """
    if not keys:
        return float(DEFAULT_NHL_RATING)

    total = 0.0
    count = 0
    for k in keys:
        if k in ratings:
            v = ratings[k]
            try:
                total += float(v)
            except (TypeError, ValueError):
                continue
            count += 1

    if count == 0:
        return float(DEFAULT_NHL_RATING)

    avg = total / count
    if avg < RATING_MIN:
        return float(RATING_MIN)
    if avg > RATING_MAX:
        return float(RATING_MAX)
    return avg


def _skater_category_raw_avgs(ratings: Dict[str, Any]) -> Dict[str, float]:
    return {
        "offense": _avg(ratings, OFFENSE_ATTRS),
        "playmaking": _avg(ratings, PLAYMAKING_ATTRS),
        "defense": _avg(ratings, DEFENSE_ATTRS),
        "skating": _avg(ratings, SKATING_ATTRS),
        "physical": _avg(ratings, PHYSICAL_ATTRS),
        "iq": _avg(ratings, IQ_ATTRS),
        "skill": _avg(ratings, SKILL_ATTRS),
        "hidden": _avg(ratings, DEV_ATTRS),
        "personality": _avg(ratings, PERSONALITY_ATTRS),
        "special": _avg(ratings, SPECIAL_ATTRS),
    }


def _group_avgs(ratings: Dict[str, Any], position: Position) -> Dict[str, float]:
    """Backward-compatible group averages for chemistry / debug."""
    if position == Position.G:
        return {
            "goalie": float(_avg(ratings, GOALIE_ATTRS)),
            "iq": float(_avg(ratings, IQ_ATTRS)),
            "physical": float(_avg(ratings, PHYSICAL_ATTRS)),
            "skating": float(_avg(ratings, SKATING_ATTRS)),
        }

    s = _skater_category_raw_avgs(ratings)
    return {
        "skating": float(s["skating"]),
        "offense": float(s["offense"]),
        "passing": float(s["playmaking"]),
        "defense": float(s["defense"]),
        "iq": float(s["iq"]),
        "physical": float(s["physical"]),
    }


def compute_ovr(ratings: Dict[str, Any], position: Position, archetype: Optional[str] = None) -> float:
    """
    Returns current OVR on a 0.0–1.0 scale.

    Design rule:
    - Current ability is calculated from current ratings.
    - Potential/development ratings influence progression, not current OVR hard-capping.
    - Archetype multipliers redistribute category leverage without free OVR inflation
      (weights are renormalized after applying style mults).
    """

    def norm_pts(x: float) -> float:
        return clamp01(float(x) / RATING_MAX)

    if position == Position.G:
        g_skill = _avg(ratings, GOALIE_ATTRS)
        sk = _avg(ratings, SKATING_ATTRS)
        iq = _avg(ratings, IQ_ATTRS)
        phy = _avg(ratings, PHYSICAL_ATTRS)

        pts = 0.64 * g_skill + 0.13 * sk + 0.14 * iq + 0.09 * phy

        arch = resolve_ovr_archetype_key(archetype or "BALANCED_G")
        # Style nudge for goalies without gross OVR inflation.
        if arch == "ELITE_FRANCHISE":
            pts *= 1.04
        elif arch == "BUTTERFLY_G":
            pts = 0.68 * g_skill + 0.10 * sk + 0.15 * iq + 0.07 * phy
        elif arch == "HYBRID_G":
            pts = 0.60 * g_skill + 0.18 * sk + 0.14 * iq + 0.08 * phy

        return norm_pts(min(99.0, pts))

    cats = _skater_category_raw_avgs(ratings)
    arch_u = resolve_ovr_archetype_key(archetype or "BALANCED")

    if arch_u == "ELITE_FRANCHISE":
        mults = {
            "offense": 1.16,
            "playmaking": 1.14,
            "defense": 1.06,
            "skating": 1.10,
            "physical": 1.04,
            "iq": 1.14,
            "skill": 1.12,
            "hidden": 1.00,
            "personality": 1.00,
            "special": 1.06,
        }
    else:
        table = ARCHETYPE_CATEGORY_MULT.get(arch_u)
        mults = dict(table) if table else {k: 1.0 for k in cats}

    weighted_pts = 0.0
    weight_mass = 0.0
    base_weight_sum = 0.0
    for cat, base in cats.items():
        w = OVR_CATEGORY_WEIGHTS_SKATER.get(cat, 0.0)
        if w <= 0:
            continue
        m = float(mults.get(cat, 1.0))
        weighted_pts += float(base) * m * w
        weight_mass += m * w
        base_weight_sum += w

    # Renormalize so archetypes reweight categories without printing free OVR.
    if weight_mass > 1e-9 and base_weight_sum > 1e-9:
        weighted_pts = weighted_pts / weight_mass * base_weight_sum

    # Soft situational bonuses removed — those attrs already live in iq/special categories.

    return norm_pts(min(99.0, weighted_pts))


# ============================================================
# NHL OVR DISTRIBUTION ENFORCEMENT
# ============================================================

def _player_ovr_0_100(player: Any) -> float:
    try:
        fn = getattr(player, "ovr", None)
        val = float(fn()) if callable(fn) else float(getattr(player, "ovr", 0.0))
    except Exception:
        val = 0.0

    if val <= 1.25:
        return val * 99.0
    return val


def _player_position_for_distribution(player: Any) -> str:
    pos = getattr(player, "position", None)
    if hasattr(pos, "value"):
        return str(pos.value).upper()

    ident = getattr(player, "identity", None)
    if ident is not None:
        ipos = getattr(ident, "position", None)
        if hasattr(ipos, "value"):
            return str(ipos.value).upper()
        if ipos:
            return str(ipos).upper()

    return str(pos or "").upper()


def _rating_keys_for_distribution_boost(player: Any) -> List[str]:
    pos = _player_position_for_distribution(player)

    if pos == "G":
        return GOALIE_KEYS + IQ_KEYS + SKATING_KEYS + PHYS_KEYS

    if pos == "D":
        return (
            DEFENSE_KEYS
            + SKATING_KEYS
            + IQ_KEYS
            + PHYS_KEYS
            + PASSING_KEYS
            + SKILL_KEYS
            + OFFENSE_KEYS
            + PERSONALITY_KEYS
            + SPECIAL_KEYS
        )

    return (
        OFFENSE_KEYS
        + PASSING_KEYS
        + SKATING_KEYS
        + IQ_KEYS
        + SKILL_KEYS
        + PHYS_KEYS
        + DEFENSE_KEYS
        + PERSONALITY_KEYS
        + SPECIAL_KEYS
    )


def _boost_player_toward_ovr(
    player: Any,
    target_ovr_100: float,
    *,
    max_passes: int = 16,
    per_key_step: float = 1.9,
) -> float:
    """
    Raises a player toward a target OVR without completely destroying archetype shape.
    Returns final OVR on 0–100 scale.
    """
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict) or not ratings:
        return _player_ovr_0_100(player)

    target = float(max(1.0, min(99.0, target_ovr_100)))
    keys = [k for k in _rating_keys_for_distribution_boost(player) if k in ratings]

    if not keys:
        keys = list(ratings.keys())

    for _ in range(max_passes):
        cur = _player_ovr_0_100(player)
        if cur >= target:
            break

        gap = max(0.0, target - cur)
        step = min(per_key_step, max(0.55, gap * 0.42))

        for k in keys:
            ratings[k] = clamp_rating(float(ratings.get(k, DEFAULT_NHL_RATING)) + step)

        if "dev_potential" in ratings:
            ratings["dev_potential"] = clamp_rating(max(float(ratings["dev_potential"]), target))

        # OVR is memoized on Player — invalidate after each attribute write.
        inval = getattr(player, "_invalidate_ovr_memo", None)
        if callable(inval):
            inval()

    return _player_ovr_0_100(player)


def _lower_player_toward_ovr(
    player: Any,
    target_ovr_100: float,
    *,
    max_passes: int = 16,
    per_key_step: float = 0.75,
) -> float:
    """
    Optional softener for accidental mega-inflation.
    This does not touch players under the target.
    """
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict) or not ratings:
        return _player_ovr_0_100(player)

    target = float(max(1.0, min(99.0, target_ovr_100)))
    keys = [k for k in _rating_keys_for_distribution_boost(player) if k in ratings]

    if not keys:
        keys = [k for k in ratings.keys() if not str(k).startswith("_")]

    for _ in range(max_passes):
        cur = _player_ovr_0_100(player)
        if cur <= target:
            break

        gap = max(0.0, cur - target)
        step = min(per_key_step, max(0.20, gap * 0.15))

        for k in keys:
            ratings[k] = clamp_rating(float(ratings.get(k, DEFAULT_NHL_RATING)) - step)

        inval = getattr(player, "_invalidate_ovr_memo", None)
        if callable(inval):
            inval()

    return _player_ovr_0_100(player)


def enforce_minimum_player_ovr(player: Any, min_ovr_100: float = NHL_MIN_ACTIVE_OVR) -> float:
    """
    Guarantees an individual NHL player is not below the active-roster floor.
    Use this on player creation and after major yearly regression.
    """
    cur = _player_ovr_0_100(player)
    if cur < float(min_ovr_100):
        return _boost_player_toward_ovr(
            player,
            float(min_ovr_100),
            max_passes=12,
            per_key_step=1.65,
        )
    return cur


def enforce_nhl_roster_ovr_distribution(
    players: List[Any],
    rng: Optional[random.Random] = None,
    *,
    min_ovr_100: float = NHL_MIN_ACTIVE_OVR,
    target_90_plus: int = NHL_TARGET_SUPERSTARS_90_PLUS,
    superstar_floor_100: float = NHL_SUPERSTAR_OVR,
    include_goalies: bool = True,
) -> Dict[str, Any]:
    """
    League-wide OVR correction pass.

    Guarantees:
    - No active NHL player below 74 OVR.
    - At least 15 players at 90+ OVR.
    - Superstars are created from the strongest existing players, not random depth guys.
    - Archetype shape is mostly preserved because boosts use position-relevant keys.

    Call this after all NHL rosters are generated/finalized.
    """
    rng = rng or random.Random()

    eligible: List[Any] = []
    for p in players or []:
        if p is None or getattr(p, "retired", False):
            continue

        pos = _player_position_for_distribution(p)
        if pos == "G" and not include_goalies:
            continue

        ratings = getattr(p, "ratings", None)
        if not isinstance(ratings, dict) or not ratings:
            continue

        eligible.append(p)

    if not eligible:
        return {
            "players_seen": 0,
            "floor_boosted": 0,
            "superstars_before": 0,
            "superstars_after": 0,
            "superstars_created": 0,
            "min_ovr": None,
            "max_ovr": None,
        }

    floor_boosted = 0

    for p in eligible:
        before = _player_ovr_0_100(p)
        if before < min_ovr_100:
            enforce_minimum_player_ovr(p, min_ovr_100)
            floor_boosted += 1

    def score_for_star_boost(p: Any) -> float:
        o = _player_ovr_0_100(p)
        ratings = getattr(p, "ratings", {}) or {}

        iq = float(ratings.get("iqm_hockey_iq", DEFAULT_NHL_RATING))
        cons = float(ratings.get("iqm_consistency", DEFAULT_NHL_RATING))
        clutch = float(ratings.get("iqm_clutch_factor", DEFAULT_NHL_RATING))
        pot = float(ratings.get("dev_potential", DEFAULT_NHL_RATING))
        big = float(ratings.get("st_big_game_performance", DEFAULT_NHL_RATING))

        return (
            o * 1.00
            + iq * 0.055
            + cons * 0.045
            + clutch * 0.040
            + pot * 0.050
            + big * 0.030
            + rng.random() * 0.35
        )

    superstars_before = sum(1 for p in eligible if _player_ovr_0_100(p) >= superstar_floor_100)

    if superstars_before < target_90_plus:
        needed = int(target_90_plus - superstars_before)

        candidates = sorted(
            eligible,
            key=score_for_star_boost,
            reverse=True,
        )

        created = 0
        for p in candidates:
            if created >= needed:
                break

            cur = _player_ovr_0_100(p)
            if cur >= superstar_floor_100:
                continue

            # The higher the candidate already is, the easier it is to become a real star.
            target = superstar_floor_100 + rng.uniform(0.0, 2.2)
            if cur >= 87.0:
                target += rng.uniform(0.6, 1.6)
            elif cur < 82.0:
                target = superstar_floor_100

            final = _boost_player_toward_ovr(
                p,
                target,
                max_passes=36,
                per_key_step=1.35,
            )

            if final >= superstar_floor_100:
                created += 1
                setattr(p, "_distribution_promoted_to_superstar", True)

                ratings = getattr(p, "ratings", None)
                if isinstance(ratings, dict):
                    ratings["dev_potential"] = clamp_rating(max(float(ratings.get("dev_potential", 90)), final))
                    ratings["iqm_consistency"] = clamp_rating(max(float(ratings.get("iqm_consistency", 80)), 84))
                    ratings["iqm_clutch_factor"] = clamp_rating(max(float(ratings.get("iqm_clutch_factor", 80)), 84))
                    ratings["st_pressure_handling"] = clamp_rating(max(float(ratings.get("st_pressure_handling", 80)), 84))
                    ratings["st_big_game_performance"] = clamp_rating(max(float(ratings.get("st_big_game_performance", 80)), 83))

    superstars_after = sum(1 for p in eligible if _player_ovr_0_100(p) >= superstar_floor_100)

    # Final safety pass. No one under floor.
    for p in eligible:
        enforce_minimum_player_ovr(p, min_ovr_100)

    ovrs = [_player_ovr_0_100(p) for p in eligible]

    return {
        "players_seen": len(eligible),
        "floor_boosted": floor_boosted,
        "superstars_before": superstars_before,
        "superstars_after": superstars_after,
        "superstars_created": max(0, superstars_after - superstars_before),
        "min_ovr": round(min(ovrs), 2) if ovrs else None,
        "max_ovr": round(max(ovrs), 2) if ovrs else None,
    }


def enforce_league_ovr_distribution_from_league(
    league: Any,
    rng: Optional[random.Random] = None,
    *,
    min_ovr_100: float = NHL_MIN_ACTIVE_OVR,
    target_90_plus: int = NHL_TARGET_SUPERSTARS_90_PLUS,
) -> Dict[str, Any]:
    """
    Convenience wrapper for your League object.

    Call this after rosters are created:
        enforce_league_ovr_distribution_from_league(league, rng)
    """
    players: List[Any] = []

    for team in getattr(league, "teams", None) or []:
        for p in getattr(team, "roster", None) or []:
            if p is not None and not getattr(p, "retired", False):
                players.append(p)

    return enforce_nhl_roster_ovr_distribution(
        players,
        rng=rng,
        min_ovr_100=min_ovr_100,
        target_90_plus=target_90_plus,
    )


# ============================================================
# LIFE PRESSURE
# ============================================================

@dataclass
class LifePressureState:
    career_identity: float = 0.0
    health: float = 0.0
    family: float = 0.0
    psychological: float = 0.0
    security: float = 0.0
    environment: float = 0.0

    def clamp_all(self) -> None:
        for k, v in self.__dict__.items():
            self.__dict__[k] = clamp01(float(v))

    def decay(self, rate: float = 0.92) -> None:
        for k in self.__dict__:
            self.__dict__[k] = clamp01(float(self.__dict__[k]) * rate)

    def overall(self) -> float:
        vals = list(self.__dict__.values())
        if not vals:
            return 0.0
        return clamp01(sum(float(v) for v in vals) / len(vals))


# ============================================================
# DATA CONTAINERS
# ============================================================

@dataclass
class IdentityBio:
    name: str
    age: int
    birth_year: int
    birth_country: str
    birth_city: str
    height_cm: int
    weight_kg: int
    position: Position
    shoots: Shoots
    draft_year: int
    draft_round: int
    draft_pick: int


@dataclass
class BackstoryUpbringing:
    backstory: BackstoryType
    upbringing: UpbringingType
    family_support: SupportLevel
    early_pressure: PressureLevel
    dev_resources: DevResources


@dataclass
class PersonalityTraits:
    loyalty: float = 0.5
    ambition: float = 0.5
    money_focus: float = 0.5
    family_priority: float = 0.5
    legacy_drive: float = 0.5
    risk_tolerance: float = 0.5
    adaptability: float = 0.5
    patience: float = 0.5
    stability_need: float = 0.5
    ego: float = 0.5
    confidence: float = 0.5
    volatility: float = 0.5
    competitiveness: float = 0.5
    leadership: float = 0.5
    coachability: float = 0.5
    media_comfort: float = 0.5
    introversion: float = 0.5
    work_ethic: float = 0.5
    mental_toughness: float = 0.5
    clutch_tendency: float = 0.5

    def clamp_all(self) -> None:
        for k, v in self.__dict__.items():
            self.__dict__[k] = clamp01(float(v))


@dataclass
class CareerArcSeeds:
    career_arc: CareerArcType = CareerArcType.STEADY
    expected_peak_age: int = 27
    decline_rate: float = 0.5
    breakout_probability: float = 0.15
    bust_probability: float = 0.10
    prime_duration: float = 0.5
    season_consistency: float = 0.5
    dev_curve_seed: int = 0
    regression_resistance: float = 0.5
    ceiling_floor_gap: float = 0.5

    def clamp_all(self) -> None:
        self.decline_rate = clamp01(self.decline_rate)
        self.breakout_probability = clamp01(self.breakout_probability)
        self.bust_probability = clamp01(self.bust_probability)
        self.prime_duration = clamp01(self.prime_duration)
        self.season_consistency = clamp01(self.season_consistency)
        self.regression_resistance = clamp01(self.regression_resistance)
        self.ceiling_floor_gap = clamp01(self.ceiling_floor_gap)


@dataclass
class HealthState:
    fatigue: float = 0.0
    max_stamina: float = 1.0
    injury_risk_baseline: float = 0.25
    wear_and_tear: float = 0.0
    chronic_flags: List[str] = field(default_factory=list)
    pain_tolerance: float = 0.5
    recovery_speed: float = 0.5
    injury_status: InjuryStatus = InjuryStatus.HEALTHY
    days_injured_career: int = 0
    injury_history: List[Dict[str, Any]] = field(default_factory=list)

    def clamp_all(self) -> None:
        self.fatigue = clamp01(self.fatigue)
        self.max_stamina = clamp01(self.max_stamina)
        self.injury_risk_baseline = clamp01(self.injury_risk_baseline)
        self.wear_and_tear = clamp01(self.wear_and_tear)
        self.pain_tolerance = clamp01(self.pain_tolerance)
        self.recovery_speed = clamp01(self.recovery_speed)


@dataclass
class PsychologyState:
    morale: float = 0.5
    morale_sensitivity: float = 0.5
    team_success_dependency: float = 0.5
    role_satisfaction: float = 0.5
    ice_time_satisfaction: float = 0.5
    coach_relationship: float = 0.5
    locker_room_fit: float = 0.5
    pressure_response: float = 0.5

    confidence_level: float = 0.5
    confidence_volatility: float = 0.5
    self_doubt_bias: float = 0.5
    resilience_after_mistakes: float = 0.5
    response_to_benching: float = 0.5
    response_to_praise: float = 0.5
    response_to_criticism: float = 0.5
    tilt_susceptibility: float = 0.5
    bounce_back_tendency: float = 0.5
    anxiety_level: float = 0.5
    pressure_fatigue: float = 0.0
    mental_fatigue: float = 0.0
    playoff_nerves: float = 0.5
    media_stress: float = 0.5
    internal_motivation: float = 0.5

    locker_influence: float = 0.5
    peer_pressure: float = 0.5
    clique_affinity: float = 0.5
    isolation_tendency: float = 0.5
    veteran_respect_bias: float = 0.5
    rookie_mentor_tendency: float = 0.5
    conflict_escalation: float = 0.5
    conflict_resolution: float = 0.5
    leadership_emergence: float = 0.5
    confront_willingness: float = 0.5
    trust_in_teammates: float = 0.5
    cultural_fit: float = 0.5
    chemistry_contribution: float = 0.5

    coach_trust: float = 0.5
    coach_patience_tolerance: float = 0.5
    system_buy_in: float = 0.5
    tactical_flexibility: float = 0.5
    responsiveness_to_adjustments: float = 0.5
    system_preference_structure: float = 0.5
    role_acceptance_threshold: float = 0.5
    scratch_reaction: float = 0.5
    ice_time_justification_sensitivity: float = 0.5
    coaching_stability_dependency: float = 0.5

    decision_fatigue_spillover: float = 0.5
    momentum_carryover: float = 0.5
    performance_memory_length: float = 0.5
    streak_amplification: float = 0.5
    slump_duration_tendency: float = 0.5
    comeback_boost: float = 0.5
    front_runner_effect: float = 0.5
    chaser_effect: float = 0.5
    rivalry_intensity: float = 0.5
    home_ice_boost: float = 0.5
    road_fatigue_sensitivity: float = 0.5
    back_to_back_penalty: float = 0.5
    overtime_composure: float = 0.5
    shootout_composure: float = 0.5
    line_stability_preference: float = 0.5
    game_importance_sensitivity: float = 0.5
    playoff_grind_tolerance: float = 0.5

    contract_pressure: float = 0.5
    contract_year_bias: float = 0.5
    trade_rumor_sensitivity: float = 0.5
    ntc_security_effect: float = 0.5
    relocation_stress: float = 0.5
    market_size_sensitivity: float = 0.5
    fan_pressure: float = 0.5
    narrative_awareness: float = 0.5
    legacy_anxiety: float = 0.5
    career_satisfaction: float = 0.5
    play_hurt_willingness: float = 0.5
    risk_under_uncertainty: float = 0.5
    trust_in_management: float = 0.5
    org_stability_perception: float = 0.5
    long_term_commitment_comfort: float = 0.5

    randomness_amplification: float = 0.5
    consistency_dampener: float = 0.5
    upset_boost: float = 0.5
    implosion_threshold: float = 0.5
    hero_game_chance: float = 0.5
    liability_game_chance: float = 0.5
    narrative_spike: float = 0.5
    personality_variance_override: float = 0.5
    hidden_intangibles_bias: float = 0.5

    def clamp_all(self) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                self.__dict__[k] = clamp01(v)


@dataclass
class ContextState:
    current_team_id: Optional[str] = None
    current_contract_id: Optional[str] = None

    line_assignment: Optional[str] = None
    special_teams_role: Optional[str] = None
    on_ice: bool = False

    recent_performance_trend: float = 0.5
    hot_cold_state: float = 0.5
    momentum_susceptibility: float = 0.5
    penalty_tendency_mod: float = 0.5

    chaos_seed: int = 0

    def clamp_all(self) -> None:
        self.recent_performance_trend = clamp01(self.recent_performance_trend)
        self.hot_cold_state = clamp01(self.hot_cold_state)
        self.momentum_susceptibility = clamp01(self.momentum_susceptibility)
        self.penalty_tendency_mod = clamp01(self.penalty_tendency_mod)


# ============================================================
# ATTRIBUTE DECAY + INJURY SCARRING HELPERS
# ============================================================

def _decay_targeted(
    ratings: Dict[str, float],
    keys: List[str],
    amount: float,
    rng: random.Random,
    noise: float = 0.15,
) -> None:
    """
    Reduce selected ratings by rating points with mild randomness.
    """
    if not keys:
        return

    for k in keys:
        if k not in ratings:
            continue
        d = amount * (1.0 + rng.uniform(-noise, noise))
        ratings[k] = clamp_rating(float(ratings[k]) - d)


def _apply_global_decay(
    ratings: Dict[str, int],
    amount: float,
    rng: random.Random,
    noise: float = 0.10,
) -> None:
    for k in list(ratings.keys()):
        d = amount * (1.0 + rng.uniform(-noise, noise))
        ratings[k] = clamp_rating(float(ratings[k]) - d)


def _apply_injury_scarring(
    ratings: Dict[str, float],
    *,
    injury_severity: float,
    position: Position,
    rng: random.Random,
) -> Dict[str, float]:
    """
    Apply a permanent scar to ratings depending on severity.
    Returns a small scar report dict so the engine can log narrative.
    """
    severity = clamp01(injury_severity)

    if position == Position.G:
        target_groups = [
            (GOALIE_KEYS, 0.55),
            (IQ_KEYS, 0.20),
            (PHYS_KEYS, 0.25),
        ]
    else:
        target_groups = [
            (SKATING_KEYS, 0.40),
            (PHYS_KEYS, 0.35),
            (IQ_KEYS, 0.25),
        ]

    base = 0.35 + 2.50 * (severity ** 1.35)

    for keys, weight in target_groups:
        _decay_targeted(ratings, keys, amount=base * weight, rng=rng, noise=0.25)

    return {
        "severity": severity,
        "base_scar": base,
    }


# ============================================================
# PLAYER
# ============================================================

class Player:
    """
    Player entity:
    - identity + upbringing/backstory
    - ratings dict, 20–99
    - AI traits / career arc seeds
    - psychology + context + health
    - life pressure
    - yearly evolution helpers
    """

    def __init__(
        self,
        identity: IdentityBio,
        backstory: BackstoryUpbringing,
        ratings: Optional[Dict[str, float]] = None,
        traits: Optional[PersonalityTraits] = None,
        career: Optional[CareerArcSeeds] = None,
        psychology: Optional[PsychologyState] = None,
        health: Optional[HealthState] = None,
        context: Optional[ContextState] = None,
        retired: bool = False,
        rng_seed: Optional[int] = None,
        archetype: Optional[str] = None,
        pool_context: Optional[str] = "nhl",
        enforce_floor_on_init: bool = True,
    ):
        self.identity = identity
        self.backstory = backstory

        self.ratings: Dict[str, float] = normalize_ratings_dict(
            ratings or {},
            keys=ATTRIBUTE_KEYS,
            default=DEFAULT_NHL_RATING,
        )

        self.traits = traits or PersonalityTraits()
        if traits is not None:
            self.traits.clamp_all()

        self.career = career or CareerArcSeeds()
        if career is not None:
            self.career.clamp_all()

        self.psych = psychology or PsychologyState()
        if psychology is not None:
            self.psych.clamp_all()

        self.health = health or HealthState()
        if health is not None:
            self.health.clamp_all()

        self.life_pressure = LifePressureState()

        if rng_seed is None:
            rng_seed = random.randint(1, 2_000_000_000)

        self.context = context or ContextState(chaos_seed=rng_seed)
        if not getattr(self.context, "chaos_seed", 0):
            self.context.chaos_seed = rng_seed
        if context is not None:
            self.context.clamp_all()

        self.id = f"PLAYER_{rng_seed}"
        self.retired = retired

        self._rng = random.Random(self.context.chaos_seed)

        arch_raw = str(archetype or "").strip()
        # Prefer generation profile → OVR archetype so leverage matches the shaped attributes.
        gen_prof = ""
        if isinstance(ratings, dict) and ratings.get("_generated_profile"):
            gen_prof = str(ratings.get("_generated_profile") or "")
        if not arch_raw and gen_prof:
            mapped = archetype_from_generation_profile(gen_prof, self.identity.position)
            if mapped:
                arch_raw = mapped
        resolved = (
            resolve_ovr_archetype_key(arch_raw)
            if arch_raw
            else assign_skater_archetype(self.identity.position, self._rng)
        )
        self.archetype: str = str(resolved)

        # Preserve generation shaping label when present (popped from ratings dict).
        self._generated_profile: str = str(getattr(self, "_generated_profile", "") or "") or gen_prof
        if isinstance(ratings, dict) and ratings.get("_generated_profile"):
            self._generated_profile = str(ratings.get("_generated_profile") or "")
            # Keep meta out of live ratings if caller forgot to pop.
            try:
                self.ratings.pop("_generated_profile", None)
            except Exception:
                pass
            ratings.pop("_generated_profile", None)

        self._narrative_prog_growth_mult: float = 1.0
        self._narrative_regression_rate_mult: float = 1.0
        self._narrative_breakout_p_mult: float = 1.0
        self._narrative_decline_p_mult: float = 1.0
        self._narrative_consistency_shift: float = 0.0
        self._narrative_performance_variance: float = 0.0
        self._narrative_mechanics_year: int = 0

        self._dev_archetype: str = ""
        self._pipeline_dev_curve: str = "normal"
        self._dev_curve_hint: str = "normal"
        self._nhl_adjustment_years_remaining: int = 0
        self._bust_pressure: float = 0.08
        self._steal_momentum: float = 0.06
        self._dev_env_growth_mult: float = 1.0
        self._dev_env_variance_mult: float = 1.0

        # Chemistry state (backward-compatible lazy profile fill).
        self.chemistry_profile: Dict[str, Any] = {}
        self.chemistry_relationships: Dict[str, float] = {}
        self.chemistry_history: List[Dict[str, Any]] = []

        self._apply_creation_biases()

        self._pool_context = str(pool_context or "nhl").strip().lower()
        if bool(enforce_floor_on_init):
            floor = get_ovr_floor_for_pool(self._pool_context)
            if floor > 0:
                enforce_minimum_player_ovr(self, floor)

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def age(self) -> int:
        return self.identity.age

    @property
    def position(self) -> Position:
        return self.identity.position

    @property
    def shoots(self) -> Shoots:
        return self.identity.shoots

    def get(self, key: str, default: int = DEFAULT_NHL_RATING) -> int:
        if key in ALIASES:
            key = ALIASES[key]
        return int(self.ratings.get(key, default))

    def set(self, key: str, value: float) -> None:
        if key in ALIASES:
            key = ALIASES[key]
        if key not in self.ratings:
            raise KeyError(f"Unknown rating key: {key}")
        self.ratings[key] = clamp_rating(value)
        self._invalidate_ovr_memo()

    def group_averages(self) -> Dict[str, float]:
        return _group_avgs(self.ratings, self.position)

    def ovr(self) -> float:
        memo = getattr(self, "_ovr_memo", None)
        if memo is not None:
            return float(memo)
        val = compute_ovr(self.ratings, self.position, getattr(self, "archetype", None))
        self._ovr_memo = val
        return val

    def _invalidate_ovr_memo(self) -> None:
        self._ovr_memo = None

    def reset_game(self) -> None:
        self.context.on_ice = False

        self.context.hot_cold_state = clamp01(self.context.hot_cold_state * 0.90 + 0.05)
        self.context.recent_performance_trend = clamp01(self.context.recent_performance_trend * 0.90 + 0.05)

        self.psych.pressure_fatigue = clamp01(self.psych.pressure_fatigue * 0.85)
        self.psych.mental_fatigue = clamp01(self.psych.mental_fatigue * 0.90)

        self.health.fatigue = clamp01(self.health.fatigue * 0.45)

    def reset_season(self) -> None:
        self.context.line_assignment = None
        self.context.special_teams_role = None

        self.psych.morale = clamp01(self.psych.morale * 0.70 + 0.15)

        self.context.hot_cold_state = 0.5
        self.context.recent_performance_trend = 0.5

        self.health.fatigue = 0.0
        self.life_pressure.environment = clamp01(self.life_pressure.environment * 0.85)

    def advance_year(
        self,
        *,
        season_morale: Optional[float] = None,
        season_injury_risk: Optional[float] = None,
        major_injury_severity: Optional[float] = None,
        role_change: float = 0.0,
        team_instability: float = 0.0,
        development_modifier: float = 0.0,
        apply_peak_decline: bool = True,
    ) -> Dict[str, Any]:
        """
        One call per simulated year.

        This mutates:
        - age
        - life pressure
        - personality drift
        - young-player growth
        - age decline (when apply_peak_decline=True)
        - optional injury scarring

        NOTE:
        If engine.py already runs a separate progression controller, do not call
        this twice in the same season. Pick one seasonal aging authority.

        When the career lifecycle owns AGING DECLINE (franchise / universe seasonal
        pass), pass apply_peak_decline=False so post-peak attribute decay is not
        stacked with lifecycle cliffs and injury/morale regression.
        """
        report: Dict[str, Any] = {"age_before": self.age}

        self.identity.age += 1
        report["age_after"] = self.age

        morale = self.psych.morale if season_morale is None else clamp01(season_morale)
        injury_risk = self.health.injury_risk_baseline if season_injury_risk is None else clamp01(season_injury_risk)

        age_factor = clamp01((self.age - 26) / 15.0)

        self.life_pressure.health = clamp01(self.life_pressure.health + injury_risk * (0.10 + 0.10 * age_factor))
        self.life_pressure.career_identity = clamp01(self.life_pressure.career_identity + (0.05 + 0.06 * age_factor))
        self.life_pressure.family = clamp01(self.life_pressure.family + self.traits.family_priority * (0.02 + 0.04 * age_factor))
        self.life_pressure.security = clamp01(self.life_pressure.security + (1.0 - morale) * (0.06 + 0.06 * age_factor))
        self.life_pressure.psychological = clamp01(self.life_pressure.psychological + (1.0 - morale) * (0.07 + 0.08 * age_factor))
        self.life_pressure.environment = clamp01(self.life_pressure.environment + clamp01(team_instability) * 0.10)

        if role_change < 0.0:
            self.life_pressure.psychological = clamp01(self.life_pressure.psychological + abs(role_change) * 0.10)
            self.life_pressure.security = clamp01(self.life_pressure.security + abs(role_change) * 0.08)
        elif role_change > 0.0:
            self.life_pressure.career_identity = clamp01(self.life_pressure.career_identity + role_change * 0.04)

        self.life_pressure.decay(rate=0.92)
        self.life_pressure.clamp_all()
        report["life_pressure"] = dict(self.life_pressure.__dict__)

        self.traits.family_priority = clamp01(self.traits.family_priority + 0.004 + 0.008 * age_factor)

        ego_drop = 0.005 + 0.010 * age_factor
        if self.life_pressure.overall() > 0.60:
            ego_drop *= 0.6
        self.traits.ego = clamp01(self.traits.ego - ego_drop)

        amb_drop = 0.003 + 0.008 * age_factor
        amb_drop *= 1.0 - 0.35 * self.traits.legacy_drive
        self.traits.ambition = clamp01(self.traits.ambition - amb_drop)

        self.traits.confidence = clamp01(
            self.traits.confidence
            + (morale - 0.5) * 0.05
            - (self.life_pressure.psychological - 0.5) * 0.02
        )

        self.traits.volatility = clamp01(
            self.traits.volatility + (self.life_pressure.overall() - 0.30) * 0.02
        )

        self.traits.clamp_all()
        report["traits"] = dict(self.traits.__dict__)

        years_past_peak = max(0, self.age - int(self.career.expected_peak_age))

        # Young-player attribute growth is owned by progression.development
        # (one seasonal development result). advance_year only ages (+ optional decline).
        report["yearly_growth_deferred"] = True

        if years_past_peak > 0 and apply_peak_decline:
            base = 1.5 + 2.0 * self.career.decline_rate
            base *= 1.0 - 0.55 * clamp01(self.career.regression_resistance)

            accel = 1.0 + 0.06 * min(20, years_past_peak)
            pressure_mult = 1.0 + 0.35 * self.life_pressure.overall()

            yearly_decay = base * accel * pressure_mult

            if self.position == Position.G:
                _decay_targeted(self.ratings, GOALIE_KEYS, amount=yearly_decay * 0.55, rng=self._rng, noise=0.20)
                _decay_targeted(self.ratings, PHYS_KEYS, amount=yearly_decay * 0.30, rng=self._rng, noise=0.20)
                _decay_targeted(self.ratings, IQ_KEYS, amount=yearly_decay * 0.15, rng=self._rng, noise=0.20)
            else:
                _decay_targeted(self.ratings, SKATING_KEYS, amount=yearly_decay * 0.40, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, PHYS_KEYS, amount=yearly_decay * 0.35, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, OFFENSE_KEYS, amount=yearly_decay * 0.12, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, PASSING_KEYS, amount=yearly_decay * 0.05, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, SKILL_KEYS, amount=yearly_decay * 0.05, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, DEFENSE_KEYS, amount=yearly_decay * 0.04, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, IQ_KEYS, amount=yearly_decay * 0.02, rng=self._rng, noise=0.10)

            report["yearly_decay"] = yearly_decay
        else:
            report["yearly_decay"] = 0.0
            if years_past_peak > 0 and not apply_peak_decline:
                report["yearly_decay_deferred_to_lifecycle"] = True

        if major_injury_severity is not None and clamp01(major_injury_severity) > 0.0:
            scar = _apply_injury_scarring(
                self.ratings,
                injury_severity=major_injury_severity,
                position=self.position,
                rng=self._rng,
            )
            self.health.wear_and_tear = clamp01(
                self.health.wear_and_tear + 0.04 * clamp01(major_injury_severity)
            )
            report["injury_scar"] = scar

        for k in list(self.ratings.keys()):
            self.ratings[k] = clamp_rating(self.ratings[k])

        floor = get_ovr_floor_for_pool(getattr(self, "_pool_context", "nhl"))
        if floor > 0:
            enforce_minimum_player_ovr(self, floor)

        self.health.wear_and_tear = clamp01(self.health.wear_and_tear + 0.005 + 0.010 * injury_risk)
        report["wear_and_tear"] = self.health.wear_and_tear

        report["ovr_after"] = self.ovr()
        report["groups_after"] = self.group_averages()

        return report

    def _apply_creation_biases(self) -> None:
        """
        Upbringing/backstory biases:
        - AI traits
        - psychology baselines
        - career arc seeds
        """

        up = self.backstory.upbringing

        if up == UpbringingType.ROUGH:
            self.traits.work_ethic += 0.12
            self.traits.mental_toughness += 0.12
            self.traits.patience -= 0.08
            self.psych.trust_in_management -= 0.08
            self.psych.resilience_after_mistakes += 0.08

        elif up == UpbringingType.EXTREME_ADVERSITY:
            self.traits.work_ethic += 0.16
            self.traits.mental_toughness += 0.16
            self.traits.volatility += 0.08
            self.psych.anxiety_level += 0.08
            self.psych.bounce_back_tendency += 0.08

        elif up == UpbringingType.PRIVILEGED:
            self.traits.media_comfort += 0.10
            self.traits.confidence += 0.08
            self.psych.market_size_sensitivity += 0.10
            self.psych.contract_pressure += 0.06
            self.traits.loyalty -= 0.05

        elif up == UpbringingType.WORKING_CLASS:
            self.traits.work_ethic += 0.08
            self.traits.coachability += 0.06
            self.psych.system_buy_in += 0.06

        else:
            self.traits.volatility -= 0.03
            self.psych.tilt_susceptibility -= 0.03

        # Use support/pressure/resources fields so they are not decorative.
        if self.backstory.family_support == SupportLevel.HIGH:
            self.psych.resilience_after_mistakes += 0.05
            self.psych.anxiety_level -= 0.04
            self.traits.confidence += 0.04
        elif self.backstory.family_support == SupportLevel.LOW:
            self.psych.anxiety_level += 0.05
            self.psych.relocation_stress += 0.04
            self.traits.volatility += 0.03

        if self.backstory.early_pressure == PressureLevel.INTENSE:
            self.psych.contract_pressure += 0.05
            self.psych.media_stress += 0.05
            self.traits.clutch_tendency += 0.03
            self.career.ceiling_floor_gap += 0.03
        elif self.backstory.early_pressure == PressureLevel.LOW:
            self.psych.media_stress -= 0.03
            self.traits.patience += 0.03

        if self.backstory.dev_resources == DevResources.ELITE:
            self.career.breakout_probability += 0.03
            self.career.regression_resistance += 0.03
            self.traits.coachability += 0.04
        elif self.backstory.dev_resources == DevResources.UNDERFUNDED:
            self.career.ceiling_floor_gap += 0.05
            self.traits.work_ethic += 0.04
            self.psych.adaptability = clamp01(getattr(self.psych, "adaptability", 0.5) if hasattr(self.psych, "adaptability") else 0.5)

        bs = self.backstory.backstory

        if bs == BackstoryType.PRODIGY:
            self.career.breakout_probability += 0.08
            self.psych.legacy_anxiety += 0.08

        elif bs == BackstoryType.LATE_BLOOMER:
            self.career.breakout_probability += 0.05
            self.career.expected_peak_age = max(self.career.expected_peak_age, 29)
            self.career.regression_resistance += 0.06

        elif bs == BackstoryType.GRINDER:
            self.traits.work_ethic += 0.08
            self.traits.coachability += 0.06
            self.career.ceiling_floor_gap -= 0.05

        elif bs == BackstoryType.PROJECT:
            self.career.ceiling_floor_gap += 0.10
            self.career.season_consistency -= 0.06

        elif bs == BackstoryType.BUST_SURVIVOR:
            self.career.bust_probability -= 0.05
            self.psych.bounce_back_tendency += 0.08

        elif bs == BackstoryType.COMEBACK:
            self.psych.internal_motivation += 0.10
            self.psych.contract_year_bias += 0.06

        self.traits.clamp_all()
        self.career.clamp_all()
        self.psych.clamp_all()

    def __repr__(self) -> str:
        return f"<Player {self.name} {self.position.value} age={self.age} shoots={self.shoots.value}>"