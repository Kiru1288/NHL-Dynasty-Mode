# app/sim_engine/entities/player.py

from __future__ import annotations

from dataclasses import dataclass, field
from email.policy import default
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
# UTILITIES
# ============================================================

RATING_MIN = 20
RATING_MAX = 99

def clamp_rating(x: float) -> int:
    return int(max(RATING_MIN, min(RATING_MAX, round(x))))

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


def normalize_ratings_dict(r, keys, default=50):
    out = {}
    for k in keys:
        out[k] = clamp_rating(r.get(k, default))
    return out



# ============================================================
# ATTRIBUTE KEYS (100 TOTAL)
# Stored in ratings dict (0.0–1.0 internal scale)
# ============================================================

ATTRIBUTE_KEYS: List[str] = [
    # --- Skating & Movement (18)
    "sk_top_speed",
    "sk_first_step_accel",
    "sk_backward_speed",
    "sk_edge_control",
    "sk_turning_radius",
    "sk_lateral_quickness",
    "sk_balance",
    "sk_body_control",
    "sk_pivots_transitions",
    "sk_crossovers",
    "sk_stride_efficiency",
    "sk_glide_efficiency",
    "sk_agility_under_pressure",
    "sk_recovery_skating",
    "sk_change_of_direction",
    "sk_sprint_stamina",
    "sk_stop_start_explosiveness",
    "sk_net_drive_speed",

    # --- Puck Skills & Offense (20)
    "of_wrist_power",
    "of_wrist_accuracy",
    "of_snap_power",
    "of_snap_accuracy",
    "of_slap_power",
    "of_slap_accuracy",
    "of_one_timer_timing",
    "of_release_speed",
    "of_deception",
    "of_hand_eye",
    "of_puck_control_speed",
    "of_puck_protection",
    "of_puck_reception",
    "of_tight_space_control",
    "of_creativity",
    "of_awareness",
    "of_net_front_finishing",
    "of_rebound_conversion",
    "of_scoring_instinct",
    "of_rush_finishing",

    # --- Passing & Playmaking (16)
    "ps_short_accuracy",
    "ps_long_accuracy",
    "ps_passing_power",
    "ps_reception_awareness",
    "ps_no_look",
    "ps_saucer",
    "ps_breakout_passing",
    "ps_cross_ice",
    "ps_touch_passing",
    "ps_play_anticipation",
    "ps_vision",
    "ps_creativity_pressure",
    "ps_decision_speed",
    "ps_turnover_avoidance",
    "ps_give_and_go",
    "ps_pp_qb",

    # --- Defensive Ability (18)
    "df_positioning",
    "df_gap_control",
    "df_stick_check",
    "df_poke_timing",
    "df_hit_power",
    "df_hit_timing",
    "df_angling",
    "df_block_willingness",
    "df_block_technique",
    "df_awareness",
    "df_anticipation",
    "df_interceptions",
    "df_net_front",
    "df_board_defense",
    "df_pk_iq",
    "df_recovery_speed",
    "df_discipline",
    "df_backcheck_intensity",

    # --- Game Intelligence & Intangibles (16)
    "iq_hockey_macro",
    "iq_spatial_awareness",
    "iq_pattern_recognition",
    "iq_reaction_time",
    "iq_composure",
    "iq_consistency",
    "iq_adaptability",
    "iq_learning_rate",
    "iq_system_fit",
    "iq_situational_awareness",
    "iq_clutch",
    "iq_mistake_recovery",
    "iq_anticipation_fatigue",
    "iq_risk_assessment",
    "iq_decision_randomness",
    "iq_momentum_sensitivity",

    # --- Physicality & Endurance (8)
    "ph_strength",
    "ph_core_strength",
    "ph_upper_strength",
    "ph_lower_strength",
    "ph_endurance",
    "ph_contact_resilience",
    "ph_durability",
    "ph_injury_resistance",

    # --- Goalie-specific (4)
    "g_reflex_speed",
    "g_positioning_discipline",
    "g_rebound_control",
    "g_lateral_recovery",
]

# These are "classic" special stats many gameplay engines want quickly
ALIASES = {
    "faceoff": "df_positioning",  # placeholder if you want; better: add a true "sp_faceoffs" key later
}


# ============================================================
# OPTION A: ATTRIBUTE GROUPS + WEIGHT MAPS
# (used for OVR + decline targeting)
# ============================================================

SKATING_KEYS = [k for k in ATTRIBUTE_KEYS if k.startswith("sk_")]
OFFENSE_KEYS = [k for k in ATTRIBUTE_KEYS if k.startswith("of_")]
PASSING_KEYS = [k for k in ATTRIBUTE_KEYS if k.startswith("ps_")]
DEFENSE_KEYS = [k for k in ATTRIBUTE_KEYS if k.startswith("df_")]
IQ_KEYS = [k for k in ATTRIBUTE_KEYS if k.startswith("iq_")]
PHYS_KEYS = [k for k in ATTRIBUTE_KEYS if k.startswith("ph_")]
GOALIE_KEYS = [k for k in ATTRIBUTE_KEYS if k.startswith("g_")]

# Overall weighting for skaters vs goalies (feel free to tune later)
OVR_WEIGHTS_SKATER: Dict[str, float] = {
    "skating": 0.22,
    "offense": 0.22,
    "passing": 0.16,
    "defense": 0.18,
    "iq": 0.14,
    "physical": 0.08,
}

OVR_WEIGHTS_GOALIE: Dict[str, float] = {
    "goalie": 0.70,
    "iq": 0.20,
    "physical": 0.10,
}


def _avg(ratings: Dict[str, int], keys: List[str]) -> int:
    if not keys:
        return 50

    total = 0
    count = 0
    for k in keys:
        if k in ratings:
            total += int(ratings[k])
            count += 1

    if count == 0:
        return 50

    return clamp_rating(total / count)




def _group_avgs(ratings: Dict[str, float], position: Position) -> Dict[str, float]:
    if position == Position.G:
        return {
            "goalie": _avg(ratings, GOALIE_KEYS),
            "iq": _avg(ratings, IQ_KEYS),
            "physical": _avg(ratings, PHYS_KEYS),
        }
    return {
        "skating": _avg(ratings, SKATING_KEYS),
        "offense": _avg(ratings, OFFENSE_KEYS),
        "passing": _avg(ratings, PASSING_KEYS),
        "defense": _avg(ratings, DEFENSE_KEYS),
        "iq": _avg(ratings, IQ_KEYS),
        "physical": _avg(ratings, PHYS_KEYS),
    }


def compute_ovr(ratings: Dict[str, int], position: Position) -> float:
    g = _group_avgs(ratings, position)

    def norm(x: int) -> float:
        return x / RATING_MAX  # 99 → 1.0

    if position == Position.G:
        w = OVR_WEIGHTS_GOALIE
        return clamp01(
            norm(g["goalie"]) * w["goalie"]
            + norm(g["iq"]) * w["iq"]
            + norm(g["physical"]) * w["physical"]
        )

    w = OVR_WEIGHTS_SKATER
    return clamp01(
        norm(g["skating"]) * w["skating"]
        + norm(g["offense"]) * w["offense"]
        + norm(g["passing"]) * w["passing"]
        + norm(g["defense"]) * w["defense"]
        + norm(g["iq"]) * w["iq"]
        + norm(g["physical"]) * w["physical"]
    )

# ============================================================
# LIFE PRESSURE (CRITICAL FIX)
# Stored here so Player has "off-ice life" state.
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
    # AI trait profile (0.0–1.0)
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
            self.__dict__[k] = clamp01(v)


@dataclass
class CareerArcSeeds:
    career_arc: CareerArcType = CareerArcType.STEADY
    expected_peak_age: int = 27
    decline_rate: float = 0.5                 # 0–1 (higher = declines faster)
    breakout_probability: float = 0.15         # 0–1
    bust_probability: float = 0.10             # 0–1
    prime_duration: float = 0.5               # 0–1 (longer prime)
    season_consistency: float = 0.5           # 0–1
    dev_curve_seed: int = 0                   # rng seed for progression engine
    regression_resistance: float = 0.5        # 0–1
    ceiling_floor_gap: float = 0.5            # 0–1 (bigger gap = more swing)

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
    fatigue: float = 0.0                      # 0–1
    max_stamina: float = 1.0                  # 0–1
    injury_risk_baseline: float = 0.25        # 0–1
    wear_and_tear: float = 0.0                # 0–1
    chronic_flags: List[str] = field(default_factory=list)
    pain_tolerance: float = 0.5               # 0–1
    recovery_speed: float = 0.5               # 0–1
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
    # Core morale block (0–1)
    morale: float = 0.5
    morale_sensitivity: float = 0.5
    team_success_dependency: float = 0.5
    role_satisfaction: float = 0.5
    ice_time_satisfaction: float = 0.5
    coach_relationship: float = 0.5
    locker_room_fit: float = 0.5
    pressure_response: float = 0.5

    # Expanded psychology (0–1)
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

    # Social/locker room (0–1)
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

    # Coach/system (0–1)
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

    # Game-to-game (0–1)
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

    # Career context (0–1)
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

    # Chaos/variance (0–1)
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

    # Gameplay usage
    line_assignment: Optional[str] = None         # e.g., "L1", "D2", "G"
    special_teams_role: Optional[str] = None      # e.g., "PP1", "PK2"
    on_ice: bool = False

    # Trends / streaks
    recent_performance_trend: float = 0.5         # 0–1
    hot_cold_state: float = 0.5                   # 0–1 (low=cold, high=hot)
    momentum_susceptibility: float = 0.5          # 0–1
    penalty_tendency_mod: float = 0.5             # 0–1

    # Seeds / chaos
    chaos_seed: int = 0

    def clamp_all(self) -> None:
        self.recent_performance_trend = clamp01(self.recent_performance_trend)
        self.hot_cold_state = clamp01(self.hot_cold_state)
        self.momentum_susceptibility = clamp01(self.momentum_susceptibility)
        self.penalty_tendency_mod = clamp01(self.penalty_tendency_mod)


# ============================================================
# OPTION A: ATTRIBUTE DECAY + INJURY SCARRING HELPERS
# These mutate self.ratings directly (since your system is unified).
# ============================================================

def _decay_targeted(
    ratings: Dict[str, float],
    keys: List[str],
    amount: float,
    rng: random.Random,
    noise: float = 0.15,
) -> None:
    """
    Reduce selected ratings by a small amount with mild randomness.
    amount is absolute (0.0–1.0 space), typically very small (e.g., 0.002).
    """
    if not keys:
        return
    for k in keys:
        if k not in ratings:
            continue
        # add some variability so decline isn't perfectly smooth
        d = amount * (1.0 + rng.uniform(-noise, noise))
        ratings[k] = clamp_rating(ratings[k] - d)



def _apply_global_decay(
    ratings: Dict[str, int],
    amount: float,
    rng: random.Random,
    noise: float = 0.10,
) -> None:
    for k in list(ratings.keys()):
        d = amount * (1.0 + rng.uniform(-noise, noise))
        ratings[k] = clamp_rating(ratings[k] - d)



def _apply_injury_scarring(
    ratings: Dict[str, float],
    *,
    injury_severity: float,
    position: Position,
    rng: random.Random,
) -> Dict[str, float]:
    """
    Apply a permanent scar to ratings depending on severity.
    Returns a small "scar report" dict so your engine can log narrative.
    """
    severity = clamp01(injury_severity)

    # pick which groups are most affected
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

    # baseline scar magnitude
    # severity 0.2 -> ~0.01-0.02 hits, severity 0.9 -> ~0.05-0.08 hits (per key group weighting)
    base = 0.01 + 0.07 * (severity ** 1.35)

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
    Ultimate Player entity:
    - identity + upbringing/backstory
    - ratings dict (100 keys, 0–1)
    - AI traits / career arc seeds
    - psychology + context + health
    - life pressure (off-ice realism)
    - option A attribute dynamics (decline + injury scarring)
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
    ):
        self.identity = identity
        self.backstory = backstory

        # Unified ratings dict (0–1)
        self.ratings: Dict[str, float] = normalize_ratings_dict(
            ratings or {},
            keys=ATTRIBUTE_KEYS,
            default=50
        )

        self.traits = traits or PersonalityTraits()
        self.traits.clamp_all()

        self.career = career or CareerArcSeeds()
        self.career.clamp_all()

        self.psych = psychology or PsychologyState()
        self.psych.clamp_all()

        self.health = health or HealthState()
        self.health.clamp_all()

        # NEW: life pressure state (off-ice)
        self.life_pressure = LifePressureState()
        self.life_pressure.clamp_all()

        # Context includes player-specific chaos seed
        if rng_seed is None:
            rng_seed = random.randint(1, 2_000_000_000)
        self.context = context or ContextState(chaos_seed=rng_seed)
        self.context.clamp_all()
                # Stable player ID (required by engine, contracts, logging)
        self.id = f"PLAYER_{rng_seed}"


        self.retired = retired

        # Dedicated RNG for this player
        self._rng = random.Random(self.context.chaos_seed)

        # Apply upbringing/backstory biases ONCE at creation
        self._apply_creation_biases()

    # -----------------------------
    # Convenience properties
    # -----------------------------
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

       # -----------------------------
    # Ratings access
    # -----------------------------
    def get(self, key: str, default: int = 50) -> int:
        if key in ALIASES:
            key = ALIASES[key]
        return int(self.ratings.get(key, default))

    def set(self, key: str, value: float) -> None:
        if key in ALIASES:
            key = ALIASES[key]
        if key not in self.ratings:
            raise KeyError(f"Unknown rating key: {key}")
        self.ratings[key] = clamp_rating(value)



    def group_averages(self) -> Dict[str, float]:
        return _group_avgs(self.ratings, self.position)

    def ovr(self) -> float:
        return compute_ovr(self.ratings, self.position)

    # -----------------------------
    # Lifecycle resets
    # -----------------------------
    def reset_game(self) -> None:
        self.context.on_ice = False

        # hot/cold and trend drift toward 0.5
        self.context.hot_cold_state = clamp01(self.context.hot_cold_state * 0.90 + 0.05)
        self.context.recent_performance_trend = clamp01(self.context.recent_performance_trend * 0.90 + 0.05)

        # mental fatigue fades slightly between games
        self.psych.pressure_fatigue = clamp01(self.psych.pressure_fatigue * 0.85)
        self.psych.mental_fatigue = clamp01(self.psych.mental_fatigue * 0.90)

        # fatigue resets partially between games
        self.health.fatigue = clamp01(self.health.fatigue * 0.45)

    def reset_season(self) -> None:
        self.context.line_assignment = None
        self.context.special_teams_role = None

        # season morale slight normalization
        self.psych.morale = clamp01(self.psych.morale * 0.70 + 0.15)

        # streak resets
        self.context.hot_cold_state = 0.5
        self.context.recent_performance_trend = 0.5

        # fatigue cleared
        self.health.fatigue = 0.0

        # environment pressure cools off a bit in offseason
        self.life_pressure.environment = clamp01(self.life_pressure.environment * 0.85)

    # -----------------------------
    # NEW: Yearly evolution hook
    # (call this once per season/year in your SimEngine)
    # -----------------------------
    def advance_year(
    self,
    *,
    season_morale: Optional[float] = None,
    season_injury_risk: Optional[float] = None,
    major_injury_severity: Optional[float] = None,
    role_change: float = 0.0,
    team_instability: float = 0.0,
    development_modifier: float = 0.0,  # 🔥 NEW (coach impact)
) -> Dict[str, Any]:

        """
        One call per simulated year.
        This is where:
        - Life pressure accumulates/decays
        - Personality drifts
        - Ratings decline with age (Option A)
        - Optional injury scarring applies

        Returns a small report dict for logging/narrative.
        """
        report: Dict[str, Any] = {"age_before": self.age}

        # -------------------------
        # age ticks
        # -------------------------
        self.identity.age += 1
        report["age_after"] = self.age

        morale = self.psych.morale if season_morale is None else clamp01(season_morale)
        injury_risk = self.health.injury_risk_baseline if season_injury_risk is None else clamp01(season_injury_risk)

        # -------------------------
        # LIFE PRESSURE BUILDUP (fixes the "all zeros forever" problem)
        # -------------------------
        # age factor starts rising mid/late 20s and ramps into 30s
        age_factor = clamp01((self.age - 26) / 15.0)

        # accumulate
        self.life_pressure.health = clamp01(self.life_pressure.health + injury_risk * (0.10 + 0.10 * age_factor))
        self.life_pressure.career_identity = clamp01(self.life_pressure.career_identity + (0.05 + 0.06 * age_factor))
        self.life_pressure.family = clamp01(self.life_pressure.family + self.traits.family_priority * (0.02 + 0.04 * age_factor))
        self.life_pressure.security = clamp01(self.life_pressure.security + (1.0 - morale) * (0.06 + 0.06 * age_factor))
        self.life_pressure.psychological = clamp01(self.life_pressure.psychological + (1.0 - morale) * (0.07 + 0.08 * age_factor))
        self.life_pressure.environment = clamp01(self.life_pressure.environment + clamp01(team_instability) * 0.10)

        # demotions/promotion ripple into pressure
        if role_change < 0.0:
            self.life_pressure.psychological = clamp01(self.life_pressure.psychological + abs(role_change) * 0.10)
            self.life_pressure.security = clamp01(self.life_pressure.security + abs(role_change) * 0.08)
        elif role_change > 0.0:
            self.life_pressure.career_identity = clamp01(self.life_pressure.career_identity + role_change * 0.04)

        # decay (still persists; doesn’t wipe clean)
        self.life_pressure.decay(rate=0.92)
        self.life_pressure.clamp_all()
        report["life_pressure"] = dict(self.life_pressure.__dict__)

        # -------------------------
        # PERSONALITY DRIFT (slow epigenetics)
        # -------------------------
        # family tends to rise with age
        self.traits.family_priority = clamp01(self.traits.family_priority + 0.004 + 0.008 * age_factor)

        # ego tends to soften late, unless pressure is high
        ego_drop = 0.005 + 0.010 * age_factor
        if self.life_pressure.overall() > 0.60:
            ego_drop *= 0.6  # stressed people cling to ego a bit more
        self.traits.ego = clamp01(self.traits.ego - ego_drop)

        # ambition often declines unless legacy drive is huge
        amb_drop = 0.003 + 0.008 * age_factor
        amb_drop *= (1.0 - 0.35 * self.traits.legacy_drive)
        self.traits.ambition = clamp01(self.traits.ambition - amb_drop)

        # confidence reacts to morale and pressure
        self.traits.confidence = clamp01(
            self.traits.confidence
            + (morale - 0.5) * 0.05
            - (self.life_pressure.psychological - 0.5) * 0.02
        )

        # volatility rises when pressure rises
        self.traits.volatility = clamp01(
            self.traits.volatility + (self.life_pressure.overall() - 0.30) * 0.02
        )

        self.traits.clamp_all()
        report["traits"] = dict(self.traits.__dict__)

        # -------------------------
        # OPTION A: ATTRIBUTE DECAY WITH AGE (ratings finally change)
        # -------------------------
        # decline starts after peak age; faster if decline_rate high and regression_resistance low
        years_past_peak = max(0, self.age - int(self.career.expected_peak_age))
        if years_past_peak > 0:
            # base yearly decay in absolute rating units
            # tuned to be subtle yearly but meaningful over a decade
            base = 1.5 + 2.0 * self.career.decline_rate   # points per year

            base *= (1.0 - 0.55 * clamp01(self.career.regression_resistance))  # resistance dampens

            # age accelerates late
            accel = 1.0 + 0.06 * min(20, years_past_peak)

            # pressure makes decline harsher (burnout/inconsistency)
            pressure_mult = 1.0 + 0.35 * self.life_pressure.overall()

            yearly_decay = base * accel * pressure_mult

            # targeted decay: skating/physical first, IQ last
            if self.position == Position.G:
                _decay_targeted(self.ratings, GOALIE_KEYS, amount=yearly_decay * 0.55, rng=self._rng, noise=0.20)
                _decay_targeted(self.ratings, PHYS_KEYS, amount=yearly_decay * 0.30, rng=self._rng, noise=0.20)
                _decay_targeted(self.ratings, IQ_KEYS, amount=yearly_decay * 0.15, rng=self._rng, noise=0.20)
            else:
                _decay_targeted(self.ratings, SKATING_KEYS, amount=yearly_decay * 0.40, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, PHYS_KEYS, amount=yearly_decay * 0.35, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, OFFENSE_KEYS, amount=yearly_decay * 0.15, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, PASSING_KEYS, amount=yearly_decay * 0.06, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, DEFENSE_KEYS, amount=yearly_decay * 0.04, rng=self._rng, noise=0.22)
                _decay_targeted(self.ratings, IQ_KEYS, amount=yearly_decay * 0.02, rng=self._rng, noise=0.10)

            report["yearly_decay"] = yearly_decay
        else:
            report["yearly_decay"] = 0.0

        # -------------------------
        # OPTIONAL: Injury scarring event (when your injury engine fires)
        # -------------------------
        if major_injury_severity is not None and clamp01(major_injury_severity) > 0.0:
            scar = _apply_injury_scarring(
                self.ratings,
                injury_severity=major_injury_severity,
                position=self.position,
                rng=self._rng,
            )
            self.health.wear_and_tear = clamp01(self.health.wear_and_tear + 0.04 * clamp01(major_injury_severity))
            report["injury_scar"] = scar

        # clamp ratings
        for k in list(self.ratings.keys()):
            self.ratings[k] = clamp_rating(self.ratings[k])


        # update wear slightly over time even without major injury
        self.health.wear_and_tear = clamp01(self.health.wear_and_tear + 0.005 + 0.010 * injury_risk)
        report["wear_and_tear"] = self.health.wear_and_tear

        report["ovr_after"] = self.ovr()
        report["groups_after"] = self.group_averages()

        return report

    # -----------------------------
    # Creation biases (one-time)
    # -----------------------------
    def _apply_creation_biases(self) -> None:
        """
        Upbringing/backstory should bias:
        - AI traits
        - psychology baselines
        - career arc seeds (variance)
        It should NOT directly give +0.20 shooting or anything like that.
        """

        # --- Upbringing effects (light biases)
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
            # stable_middle_class default: slight smoothing
            self.traits.volatility -= 0.03
            self.psych.tilt_susceptibility -= 0.03

        # --- Backstory effects (variance + arc flavor)
        bs = self.backstory.backstory
        if bs == BackstoryType.PRODIGY:
            self.career.breakout_probability += 0.08
            # (kept safe: your PsychologyState doesn't have early_pressure; avoid adding ghost fields)
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

        # clamp after biases
        self.traits.clamp_all()
        self.career.clamp_all()
        self.psych.clamp_all()

    # -----------------------------
    # Debug / display
    # -----------------------------
    def __repr__(self) -> str:
        return f"<Player {self.name} {self.position.value} age={self.age} shoots={self.shoots.value}>"
