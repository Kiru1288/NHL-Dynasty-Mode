# backend/app/sim_engine/entities/prospect.py
"""
Prospect Entity System (Pre-Draft Player Asset)

Core idea:
- A Prospect is NOT a downgraded Player.
- A Prospect is a probabilistic future asset with uncertainty, hype, and risk.
- Prospects have *signals* (ceiling-ish potentials), not NHL ratings.
- Scouting reports are *perception* of ceiling/floor/certainty, not truth.
- Conversion to a Player happens only at (or after) the draft via a projection roll.

Hard rules:
❌ Prospects do NOT have NHL ratings
❌ No deterministic outcomes
✅ Everything is probabilistic, high-variance, context-driven
✅ Development > raw stats
✅ Scouting ≠ truth

Networking:
- This module uses ONLY Python standard library. No HTTP, no external calls, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import random


# ---------------------------------------------------------------------------
# Helpers (no external deps)
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def sigmoid(x: float) -> float:
    # stable-ish sigmoid for moderate ranges
    return 1.0 / (1.0 + math.exp(-x))


def normalish(rng: random.Random, mean: float = 0.0, stdev: float = 1.0) -> float:
    # standard library normal
    return rng.gauss(mean, stdev)


def weighted_choice(rng: random.Random, items: List[Tuple[Any, float]]) -> Any:
    total = sum(w for _, w in items)
    if total <= 0:
        return items[-1][0]
    r = rng.random() * total
    upto = 0.0
    for it, w in items:
        upto += w
        if upto >= r:
            return it
    return items[-1][0]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DevelopmentSystem(str, Enum):
    CHL = "CHL"                # OHL/WHL/QMJHL path
    NCAA = "NCAA"              # USHL->NCAA path
    EURO_JR = "EURO_JR"        # European junior systems
    PREP = "PREP"              # Prep / private school
    ACADEMY = "ACADEMY"        # Specialty academy


class LeagueType(str, Enum):
    MINOR_LOCAL = "MINOR_LOCAL"
    BANTAM = "BANTAM"
    AAA = "AAA"
    CHL_OHL = "CHL_OHL"
    CHL_WHL = "CHL_WHL"
    CHL_QMJHL = "CHL_QMJHL"
    USHL = "USHL"
    NCAA = "NCAA"
    EURO_JR_TOP = "EURO_JR_TOP"
    EURO_JR_SECONDARY = "EURO_JR_SECONDARY"
    MHL = "MHL"
    ALLSVENSKAN_JR = "ALLSVENSKAN_JR"


class TournamentType(str, Enum):
    U17_CHALLENGE = "U17_CHALLENGE"
    HLINKA_GRETZKY = "HLINKA_GRETZKY"
    U18_WORLDS = "U18_WORLDS"
    WJC_U20 = "WJC_U20"


class Position(str, Enum):
    C = "C"
    LW = "LW"
    RW = "RW"
    D = "D"
    G = "G"


class Shoots(str, Enum):
    L = "L"
    R = "R"


class ProspectPhase(str, Enum):
    EARLY_DEV = "EARLY_DEV"          # 12–14
    STRUCTURED_JUNIOR = "STRUCTURED_JUNIOR"  # 15–17
    DRAFT_YEAR = "DRAFT_YEAR"        # 18


class CurveType(str, Enum):
    PRODIGY = "PRODIGY"
    LATE_BLOOMER = "LATE_BLOOMER"
    LINEAR = "LINEAR"
    BOOM_BUST = "BOOM_BUST"
    PHYSICAL_FIRST = "PHYSICAL_FIRST"
    IQ_FIRST = "IQ_FIRST"


# ---------------------------------------------------------------------------
# Identity / Context
# ---------------------------------------------------------------------------

@dataclass
class ProspectIdentity:
    name: str
    birth_year: int
    birth_country: str
    birth_city: str
    position: Position
    shoots: Shoots

    # measureables (dynamic over time)
    height_cm: int
    weight_kg: int

    # useful metadata
    handedness: Optional[str] = None  # e.g. "left-dominant", "ambidextrous", etc.


@dataclass
class DevelopmentContext:
    """
    Where and how the prospect develops.

    Values are 0..1.
    """
    country: str
    region: str
    development_system: DevelopmentSystem

    coaching_quality: float         # development instruction, structure
    ice_time_quality: float         # usage quality (top-line vs sheltered)
    competition_level: float        # strength of league/age group
    resources: float                # facilities, nutrition, travel

    def normalized(self) -> "DevelopmentContext":
        self.coaching_quality = clamp(self.coaching_quality)
        self.ice_time_quality = clamp(self.ice_time_quality)
        self.competition_level = clamp(self.competition_level)
        self.resources = clamp(self.resources)
        return self


# ---------------------------------------------------------------------------
# Non-NHL Prospect "signals" (ceiling-ish potentials)
# ---------------------------------------------------------------------------

DEFAULT_SIGNAL_KEYS = (
    "skating_potential",
    "puck_skill_potential",
    "hockey_iq_potential",
    "physical_potential",
    "scoring_instinct",
    "defensive_instinct",
)

@dataclass
class SkillSignals:
    """
    Latent ceiling-ish signals (0..1), NOT current ability, NOT NHL ratings.
    """
    values: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k in DEFAULT_SIGNAL_KEYS:
            if k not in self.values:
                self.values[k] = 0.5
        # clamp
        for k in list(self.values.keys()):
            self.values[k] = clamp(float(self.values[k]))

    def get(self, key: str) -> float:
        return float(self.values.get(key, 0.5))

    def set(self, key: str, val: float) -> None:
        self.values[key] = clamp(val)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)

    def mean(self) -> float:
        if not self.values:
            return 0.5
        return sum(self.values.values()) / len(self.values)


# ---------------------------------------------------------------------------
# Psychology, Risk, Narrative
# ---------------------------------------------------------------------------

@dataclass
class ProspectPsychology:
    confidence: float          # 0..1
    anxiety: float             # 0..1
    coachability: float        # 0..1
    work_ethic: float          # 0..1
    competitiveness: float     # 0..1
    pressure_response: float   # 0..1
    maturity: float            # 0..1

    def clamp_all(self) -> "ProspectPsychology":
        self.confidence = clamp(self.confidence)
        self.anxiety = clamp(self.anxiety)
        self.coachability = clamp(self.coachability)
        self.work_ethic = clamp(self.work_ethic)
        self.competitiveness = clamp(self.competitiveness)
        self.pressure_response = clamp(self.pressure_response)
        self.maturity = clamp(self.maturity)
        return self


@dataclass
class RiskProfile:
    """
    High risk ≠ bad prospect. High risk = polarized outcomes.
    """
    injury_risk: float                 # 0..1
    development_risk: float            # 0..1
    psychological_risk: float          # 0..1
    competition_translation_risk: float # 0..1
    boom_bust_risk: float              # 0..1

    def clamp_all(self) -> "RiskProfile":
        self.injury_risk = clamp(self.injury_risk)
        self.development_risk = clamp(self.development_risk)
        self.psychological_risk = clamp(self.psychological_risk)
        self.competition_translation_risk = clamp(self.competition_translation_risk)
        self.boom_bust_risk = clamp(self.boom_bust_risk)
        return self

    def total_risk(self) -> float:
        vals = [
            self.injury_risk,
            self.development_risk,
            self.psychological_risk,
            self.competition_translation_risk,
            self.boom_bust_risk,
        ]
        return sum(vals) / len(vals)


@dataclass
class NarrativeState:
    """
    Prospects live in story space.
    """
    hype_level: float          # 0..1
    media_attention: float     # 0..1
    draft_momentum: float      # 0..1
    reputation: float          # 0..1

    # history flags for flavor + scouting biases
    narrative_flags: List[str] = field(default_factory=list)

    def clamp_all(self) -> "NarrativeState":
        self.hype_level = clamp(self.hype_level)
        self.media_attention = clamp(self.media_attention)
        self.draft_momentum = clamp(self.draft_momentum)
        self.reputation = clamp(self.reputation)
        return self


# ---------------------------------------------------------------------------
# Development Curve (heart of it)
# ---------------------------------------------------------------------------

@dataclass
class DevelopmentCurve:
    curve_type: CurveType

    # rates roughly represent "capacity to improve" in each stage
    early_growth_rate: float   # 0..1
    mid_growth_rate: float     # 0..1
    late_growth_rate: float    # 0..1

    volatility: float          # 0..1  (how noisy year-to-year)
    plateau_risk: float        # 0..1  (chance to stall / stagnate)

    def clamp_all(self) -> "DevelopmentCurve":
        self.early_growth_rate = clamp(self.early_growth_rate)
        self.mid_growth_rate = clamp(self.mid_growth_rate)
        self.late_growth_rate = clamp(self.late_growth_rate)
        self.volatility = clamp(self.volatility)
        self.plateau_risk = clamp(self.plateau_risk)
        return self

    def stage_rate(self, age: int) -> float:
        if age <= 14:
            return self.early_growth_rate
        if 15 <= age <= 17:
            return self.mid_growth_rate
        return self.late_growth_rate


# ---------------------------------------------------------------------------
# League & Tournament Definitions (exposure, bias, pressure)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeagueProfile:
    league_type: LeagueType
    game_density: float         # 0..1 (fatigue, injury pressure, reps)
    physicality: float          # 0..1
    skill_bias: float           # 0..1 (how much it rewards skill vs chaos)
    exposure_multiplier: float  # 0..2+ (scout visibility)
    competition_multiplier: float # 0..2+ (how strong opponents are)


LEAGUE_PROFILES: Dict[LeagueType, LeagueProfile] = {
    LeagueType.MINOR_LOCAL: LeagueProfile(LeagueType.MINOR_LOCAL, 0.25, 0.20, 0.55, 0.10, 0.30),
    LeagueType.BANTAM: LeagueProfile(LeagueType.BANTAM, 0.35, 0.35, 0.55, 0.25, 0.45),
    LeagueType.AAA: LeagueProfile(LeagueType.AAA, 0.45, 0.45, 0.60, 0.40, 0.60),
    LeagueType.CHL_OHL: LeagueProfile(LeagueType.CHL_OHL, 0.85, 0.75, 0.55, 1.10, 1.05),
    LeagueType.CHL_WHL: LeagueProfile(LeagueType.CHL_WHL, 0.85, 0.80, 0.52, 1.10, 1.05),
    LeagueType.CHL_QMJHL: LeagueProfile(LeagueType.CHL_QMJHL, 0.85, 0.70, 0.60, 1.10, 1.00),
    LeagueType.USHL: LeagueProfile(LeagueType.USHL, 0.70, 0.55, 0.60, 0.95, 0.90),
    LeagueType.NCAA: LeagueProfile(LeagueType.NCAA, 0.55, 0.55, 0.62, 0.90, 0.95),
    LeagueType.EURO_JR_TOP: LeagueProfile(LeagueType.EURO_JR_TOP, 0.65, 0.55, 0.66, 0.90, 1.00),
    LeagueType.EURO_JR_SECONDARY: LeagueProfile(LeagueType.EURO_JR_SECONDARY, 0.55, 0.50, 0.62, 0.65, 0.80),
    LeagueType.MHL: LeagueProfile(LeagueType.MHL, 0.70, 0.70, 0.55, 0.85, 1.00),
    LeagueType.ALLSVENSKAN_JR: LeagueProfile(LeagueType.ALLSVENSKAN_JR, 0.60, 0.55, 0.64, 0.75, 0.90),
}


@dataclass(frozen=True)
class TournamentProfile:
    tournament_type: TournamentType
    exposure_multiplier: float     # how many eyes / narrative swing
    pressure_level: float          # 0..1
    role_variance: float           # 0..1 (how random roles/usage are)
    narrative_weight: float        # 0..2 (how much story impacts momentum)


TOURNAMENT_PROFILES: Dict[TournamentType, TournamentProfile] = {
    TournamentType.U17_CHALLENGE: TournamentProfile(TournamentType.U17_CHALLENGE, 1.0, 0.55, 0.35, 0.9),
    TournamentType.HLINKA_GRETZKY: TournamentProfile(TournamentType.HLINKA_GRETZKY, 1.2, 0.65, 0.35, 1.1),
    TournamentType.U18_WORLDS: TournamentProfile(TournamentType.U18_WORLDS, 1.35, 0.70, 0.40, 1.2),
    TournamentType.WJC_U20: TournamentProfile(TournamentType.WJC_U20, 1.8, 0.90, 0.55, 1.8),
}


# ---------------------------------------------------------------------------
# Performance objects (prospect-level, not NHL stats)
# ---------------------------------------------------------------------------

@dataclass
class TournamentPerformance:
    tournament_type: TournamentType
    games_played: int
    role: str  # e.g. "top_line", "middle_six", "depth", "top_pair", "third_pair", "starter", "backup"
    production: float  # abstract 0..1
    pressure_response: float  # 0..1 (how well they handled it)
    narrative_flag: Optional[str] = None


@dataclass
class SeasonOutcome:
    league_type: LeagueType
    age: int
    usage_quality: float     # 0..1
    competition: float       # 0..1
    season_signal: float     # 0..1: what the season "looked like" to evaluators
    fatigue: float           # 0..1
    injury_event: bool
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scouting model: Perception vs Reality
# ---------------------------------------------------------------------------

@dataclass
class ScoutingView:
    """
    What a specific scout/team believes.
    observed_ceiling/floor are 0..1, but represent *estimated* outcomes.
    certainty is how tight the scout believes that range is.
    """
    scout_id: str
    observed_ceiling: float
    observed_floor: float
    certainty: float  # 0..1
    bias_flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def clamp_all(self) -> "ScoutingView":
        self.observed_ceiling = clamp(self.observed_ceiling)
        self.observed_floor = clamp(self.observed_floor)
        self.certainty = clamp(self.certainty)
        # ensure floor <= ceiling
        if self.observed_floor > self.observed_ceiling:
            self.observed_floor, self.observed_ceiling = self.observed_ceiling, self.observed_floor
        return self


@dataclass
class ScoutProfile:
    """
    Scout tendencies (how they get fooled / what they value).
    """
    scout_id: str
    prefers_size: float      # -1..+1 (positive = overvalues size)
    prefers_iq: float        # -1..+1
    prefers_skill: float     # -1..+1
    risk_tolerance: float    # 0..1 (higher = less punishing of risk)
    hype_susceptibility: float  # 0..1 (how much media affects them)
    accuracy: float          # 0..1 baseline; affects noise
    region_bias: Optional[str] = None  # if set, favors that region/country a bit

    def __post_init__(self) -> None:
        self.prefers_size = float(max(-1.0, min(1.0, self.prefers_size)))
        self.prefers_iq = float(max(-1.0, min(1.0, self.prefers_iq)))
        self.prefers_skill = float(max(-1.0, min(1.0, self.prefers_skill)))
        self.risk_tolerance = clamp(self.risk_tolerance)
        self.hype_susceptibility = clamp(self.hype_susceptibility)
        self.accuracy = clamp(self.accuracy)


# ---------------------------------------------------------------------------
# Prospect core entity
# ---------------------------------------------------------------------------

@dataclass
class Prospect:
    """
    A Prospect is a pre-draft stochastic asset.

    - No NHL ratings.
    - Carries potentials, psychology, risk, hype, and uncertainty.
    - Simulates youth-to-draft journey (12-18) with leagues + tournaments.
    - Generates scouting views and draft-year ranking ranges.
    - Converts to a Player *only* when drafted (via projection roll).
    """

    # identity & environment
    identity: ProspectIdentity
    context: DevelopmentContext

    # state
    age: int
    phase: ProspectPhase

    # latent traits (ceiling-ish)
    skill_signals: SkillSignals

    # curve + mind + risk + story
    curve: DevelopmentCurve
    psychology: ProspectPsychology
    risk: RiskProfile
    narrative: NarrativeState

    # tracking
    current_league: LeagueType = LeagueType.MINOR_LOCAL
    season_history: List[SeasonOutcome] = field(default_factory=list)
    tournament_history: List[TournamentPerformance] = field(default_factory=list)
    scouting_history: Dict[str, List[ScoutingView]] = field(default_factory=dict)

    # draft model outputs (ranges, not a single rank)
    draft_eligibility_year: Optional[int] = None
    draft_rank_range: Tuple[int, int] = (999, 999)  # (best, worst)
    draft_value_range: Tuple[float, float] = (0.0, 0.0)  # (floor-ish, ceiling-ish) abstract 0..1

    # internal RNG
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.context.normalized()
        self.curve.clamp_all()
        self.psychology.clamp_all()
        self.risk.clamp_all()
        self.narrative.clamp_all()
        self.age = int(self.age)
        self._rng = random.Random(int(self.seed))

        # Ensure phase aligns with age
        self.phase = self._infer_phase_from_age(self.age)

        # Ensure reasonable league for age
        self.current_league = self._default_league_for_age(self.age, self.context.development_system)

        # If not set, derive draft eligibility
        if self.draft_eligibility_year is None:
            # Draft year is when they turn 18; birth_year + 18
            self.draft_eligibility_year = self.identity.birth_year + 18

    # ---------------------------
    # Construction helpers
    # ---------------------------

    @staticmethod
    def _infer_phase_from_age(age: int) -> ProspectPhase:
        if age <= 14:
            return ProspectPhase.EARLY_DEV
        if 15 <= age <= 17:
            return ProspectPhase.STRUCTURED_JUNIOR
        return ProspectPhase.DRAFT_YEAR

    @staticmethod
    def _default_league_for_age(age: int, system: DevelopmentSystem) -> LeagueType:
        if age <= 12:
            return LeagueType.MINOR_LOCAL
        if age <= 14:
            # Bantam / AAA-ish
            return LeagueType.BANTAM if system in (DevelopmentSystem.CHL, DevelopmentSystem.ACADEMY, DevelopmentSystem.PREP) else LeagueType.AAA
        # 15-17: structured path
        if system == DevelopmentSystem.CHL:
            return weighted_choice(random.Random(age * 999), [
                (LeagueType.CHL_OHL, 1.0),
                (LeagueType.CHL_WHL, 1.0),
                (LeagueType.CHL_QMJHL, 1.0),
            ])
        if system == DevelopmentSystem.NCAA:
            return LeagueType.USHL
        if system == DevelopmentSystem.EURO_JR:
            return LeagueType.EURO_JR_TOP
        if system == DevelopmentSystem.ACADEMY:
            return LeagueType.AAA
        return LeagueType.PREP

    @classmethod
    def create_random(
        cls,
        *,
        name: str,
        birth_year: int,
        birth_country: str,
        birth_city: str,
        position: Position,
        shoots: Shoots,
        height_cm: int,
        weight_kg: int,
        system: DevelopmentSystem,
        country: str,
        region: str,
        age: int,
        seed: int,
    ) -> "Prospect":
        """
        Convenience constructor for testing / draft class generation.
        Produces a plausible distribution of signals/curve/psych/risk/narrative.
        """
        rng = random.Random(seed)

        identity = ProspectIdentity(
            name=name,
            birth_year=birth_year,
            birth_country=birth_country,
            birth_city=birth_city,
            position=position,
            shoots=shoots,
            height_cm=height_cm,
            weight_kg=weight_kg,
            handedness=None,
        )

        # context
        context = DevelopmentContext(
            country=country,
            region=region,
            development_system=system,
            coaching_quality=clamp(0.35 + rng.random() * 0.60),
            ice_time_quality=clamp(0.25 + rng.random() * 0.70),
            competition_level=clamp(0.30 + rng.random() * 0.65),
            resources=clamp(0.25 + rng.random() * 0.70),
        )

        # signals: push distribution wide so draft classes have personality
        base = clamp(0.35 + rng.random() * 0.45)
        signals = {}
        for k in DEFAULT_SIGNAL_KEYS:
            signals[k] = clamp(base + normalish(rng, 0.0, 0.12))

        # positional flavor
        if position == Position.D:
            signals["defensive_instinct"] = clamp(signals["defensive_instinct"] + 0.08)
            signals["physical_potential"] = clamp(signals["physical_potential"] + 0.05)
        elif position in (Position.C,):
            signals["hockey_iq_potential"] = clamp(signals["hockey_iq_potential"] + 0.08)
        elif position in (Position.LW, Position.RW):
            signals["scoring_instinct"] = clamp(signals["scoring_instinct"] + 0.08)

        skill_signals = SkillSignals(signals)

        # curve type weighted
        curve_type = weighted_choice(rng, [
            (CurveType.LINEAR, 3.0),
            (CurveType.LATE_BLOOMER, 2.0),
            (CurveType.PHYSICAL_FIRST, 1.5),
            (CurveType.IQ_FIRST, 1.5),
            (CurveType.PRODIGY, 1.0),
            (CurveType.BOOM_BUST, 1.2),
        ])

        # curve params vary by type
        if curve_type == CurveType.PRODIGY:
            curve = DevelopmentCurve(curve_type, 0.78, 0.70, 0.55, 0.30, 0.18)
        elif curve_type == CurveType.LATE_BLOOMER:
            curve = DevelopmentCurve(curve_type, 0.40, 0.55, 0.80, 0.45, 0.25)
        elif curve_type == CurveType.BOOM_BUST:
            curve = DevelopmentCurve(curve_type, 0.62, 0.62, 0.60, 0.85, 0.40)
        elif curve_type == CurveType.PHYSICAL_FIRST:
            curve = DevelopmentCurve(curve_type, 0.68, 0.60, 0.52, 0.45, 0.28)
        elif curve_type == CurveType.IQ_FIRST:
            curve = DevelopmentCurve(curve_type, 0.55, 0.70, 0.55, 0.35, 0.22)
        else:
            curve = DevelopmentCurve(curve_type, 0.55, 0.62, 0.58, 0.40, 0.25)

        # psychology
        psychology = ProspectPsychology(
            confidence=clamp(0.35 + rng.random() * 0.55),
            anxiety=clamp(0.15 + rng.random() * 0.60),
            coachability=clamp(0.30 + rng.random() * 0.65),
            work_ethic=clamp(0.25 + rng.random() * 0.70),
            competitiveness=clamp(0.30 + rng.random() * 0.65),
            pressure_response=clamp(0.25 + rng.random() * 0.70),
            maturity=clamp(0.20 + rng.random() * 0.65),
        ).clamp_all()

        # risk
        # boom/bust curve increases boom_bust risk; high game density/physicality also nudges injury risk
        base_risk = clamp(0.20 + rng.random() * 0.55)
        risk = RiskProfile(
            injury_risk=clamp(base_risk + (0.15 if curve_type in (CurveType.BOOM_BUST,) else 0.0) + rng.random() * 0.10),
            development_risk=clamp(base_risk + (0.10 if curve_type in (CurveType.LATE_BLOOMER,) else 0.0)),
            psychological_risk=clamp(base_risk + (psychology.anxiety * 0.25) - (psychology.maturity * 0.20)),
            competition_translation_risk=clamp(base_risk + (0.10 if system in (DevelopmentSystem.PREP, DevelopmentSystem.ACADEMY) else 0.0)),
            boom_bust_risk=clamp(base_risk + (0.25 if curve_type == CurveType.BOOM_BUST else 0.0) + rng.random() * 0.10),
        ).clamp_all()

        # narrative (start modest)
        narrative = NarrativeState(
            hype_level=clamp(0.10 + rng.random() * 0.40),
            media_attention=clamp(0.05 + rng.random() * 0.35),
            draft_momentum=clamp(0.10 + rng.random() * 0.35),
            reputation=clamp(0.15 + rng.random() * 0.35),
            narrative_flags=[],
        ).clamp_all()

        return cls(
            identity=identity,
            context=context,
            age=age,
            phase=cls._infer_phase_from_age(age),
            skill_signals=skill_signals,
            curve=curve.clamp_all(),
            psychology=psychology,
            risk=risk,
            narrative=narrative,
            current_league=cls._default_league_for_age(age, system),
            seed=seed,
        )

    # ---------------------------
    # Public API: yearly sim
    # ---------------------------

    def step_year(self) -> SeasonOutcome:
        """
        Simulate one development year:
        - update measureables (height/weight) with growth spurts
        - choose/confirm league
        - simulate season exposure/pressure
        - optional tournament selection/performance
        - update signals (trajectory, volatility, plateau)
        - update narrative/hype/momentum
        - update draft-year outputs if applicable
        """
        self.phase = self._infer_phase_from_age(self.age)

        # 1) Body development (frame projection matters)
        self._apply_growth_spurt()

        # 2) Pick league based on age + system + narrative pull
        self.current_league = self._choose_league_for_year()

        # 3) Sim season outcome (abstract)
        season = self._simulate_season(self.current_league)
        self.season_history.append(season)

        # 4) Tournaments (probabilistic selection)
        tourneys = self._simulate_tournaments_for_year()
        self.tournament_history.extend(tourneys)

        # 5) Update story
        self._update_narrative_from_year(season, tourneys)

        # 6) Update skill signals (true underlying growth)
        self._apply_development_from_year(season, tourneys)

        # 7) Draft-year finalize ranges
        if self.phase == ProspectPhase.DRAFT_YEAR:
            self._finalize_draft_ranges()

        # 8) Age up
        self.age += 1
        return season

    # ---------------------------
    # Scouting API
    # ---------------------------

    def generate_scouting_view(self, scout: ScoutProfile) -> ScoutingView:
        """
        Generate a single scout's perception of this prospect right now.
        This is intentionally biased + noisy.
        """
        rng = self._rng

        # "Truth" bands: a floor-ish and ceiling-ish outcome derived from signals and risk
        true_ceiling = self._true_ceiling_value()
        true_floor = self._true_floor_value()

        # baseline certainty influenced by exposure + scout accuracy
        exposure = self._total_exposure_signal()
        certainty_base = clamp(0.20 + 0.55 * exposure + 0.35 * scout.accuracy)

        # biases
        bias_flags: List[str] = []
        bias = 0.0

        # size bias: larger frames get extra shine (or discounted)
        size_score = self._size_projection_score()
        if scout.prefers_size != 0:
            bias += scout.prefers_size * (size_score - 0.5) * 0.20
            if scout.prefers_size > 0.35:
                bias_flags.append("size_bias_plus")
            elif scout.prefers_size < -0.35:
                bias_flags.append("size_bias_minus")

        # IQ bias
        iq = self.skill_signals.get("hockey_iq_potential")
        bias += scout.prefers_iq * (iq - 0.5) * 0.18
        if scout.prefers_iq > 0.35:
            bias_flags.append("iq_bias_plus")
        elif scout.prefers_iq < -0.35:
            bias_flags.append("iq_bias_minus")

        # skill bias
        skill = 0.5 * (self.skill_signals.get("skating_potential") + self.skill_signals.get("puck_skill_potential"))
        bias += scout.prefers_skill * (skill - 0.5) * 0.18
        if scout.prefers_skill > 0.35:
            bias_flags.append("skill_bias_plus")
        elif scout.prefers_skill < -0.35:
            bias_flags.append("skill_bias_minus")

        # hype susceptibility
        hype = self.narrative.hype_level
        bias += (hype - 0.5) * 0.25 * scout.hype_susceptibility
        if scout.hype_susceptibility > 0.60:
            bias_flags.append("hype_susceptible")

        # region bias
        if scout.region_bias and scout.region_bias.lower() in (self.context.region.lower(), self.context.country.lower(), self.identity.birth_country.lower()):
            bias += 0.04
            bias_flags.append("region_familiarity")

        # risk tolerance affects perceived floor more than ceiling
        total_risk = self.risk.total_risk()
        risk_penalty = (total_risk - 0.5) * (1.0 - scout.risk_tolerance) * 0.35

        # noise (inversely proportional to scout accuracy and exposure)
        noise_scale = (1.0 - scout.accuracy) * 0.20 + (1.0 - exposure) * 0.18
        noise_c = normalish(rng, 0.0, noise_scale)
        noise_f = normalish(rng, 0.0, noise_scale)

        observed_ceiling = clamp(true_ceiling + bias + noise_c)
        observed_floor = clamp(true_floor + bias - risk_penalty + noise_f)

        # certainty modified by volatility/hype chaos: boom-bust kids are harder to pin down
        certainty = clamp(certainty_base - 0.25 * self.curve.volatility - 0.18 * self.risk.boom_bust_risk)

        notes: List[str] = []
        if "WJC" in " ".join(self.narrative.narrative_flags):
            notes.append("Recent high-visibility international event may be inflating consensus.")
        if self.curve.curve_type == CurveType.LATE_BLOOMER:
            notes.append("Late-bloomer profile: scouts may disagree on timing and ceiling.")
        if self.risk.total_risk() > 0.65:
            notes.append("High-variance asset: wide distribution of outcomes.")

        view = ScoutingView(
            scout_id=scout.scout_id,
            observed_ceiling=observed_ceiling,
            observed_floor=observed_floor,
            certainty=certainty,
            bias_flags=bias_flags,
            notes=notes,
        ).clamp_all()

        self.scouting_history.setdefault(scout.scout_id, []).append(view)
        return view

    # ---------------------------
    # Draft-year logic
    # ---------------------------

    def lock_draft_year_outputs(self, estimated_class_size: int = 224) -> None:
        """
        Call in draft year to ensure draft ranges are generated even if you
        didn't run a full year sim.
        """
        if self.phase != ProspectPhase.DRAFT_YEAR:
            self.phase = ProspectPhase.DRAFT_YEAR
        self._finalize_draft_ranges(estimated_class_size=estimated_class_size)

    def convert_to_player_payload(
        self,
        *,
        drafted_by_team_id: Optional[str],
        org_dev_quality: float,
        coach_fit: float,
        market_pressure: float,
    ) -> Dict[str, Any]:
        """
        Convert prospect into a Player-like payload. No imports, no Player dependency.
        This is the "projection roll" where busts/steals happen.

        Returns a dict payload that your draft / roster system can pass into
        your actual Player factory.

        Inputs:
        - org_dev_quality: 0..1 development staff/structure
        - coach_fit: 0..1 fit with coach philosophy (usage & growth)
        - market_pressure: 0..1 (big market; volatile prospects suffer more)

        Output fields are still normalized 0..1 "NHL-ready-ish" skill bands.
        You can map them later into your Player attribute system however you want.
        """
        rng = self._rng

        org_dev_quality = clamp(org_dev_quality)
        coach_fit = clamp(coach_fit)
        market_pressure = clamp(market_pressure)

        # base distribution: floor->ceiling roll
        true_floor = self._true_floor_value()
        true_ceiling = self._true_ceiling_value()

        # risk shapes distribution: higher risk increases variance and adds negative tail
        total_risk = self.risk.total_risk()
        variance = 0.10 + 0.22 * total_risk + 0.18 * self.curve.volatility

        # org & fit help push outcomes upward and reduce bust odds
        support = 0.35 * org_dev_quality + 0.25 * coach_fit + 0.10 * self.psychology.coachability + 0.10 * self.psychology.work_ethic

        # market pressure punishes volatility kids
        pressure_penalty = market_pressure * (0.25 * self.curve.volatility + 0.20 * self.risk.psychological_risk) * (1.0 - self.psychology.pressure_response)

        # draw a roll in [0..1] then skew it
        u = rng.random()

        # boom-bust increases polarization: more mass near 0 and 1
        if self.risk.boom_bust_risk > 0.60 or self.curve.curve_type == CurveType.BOOM_BUST:
            # Beta-ish shape via power transforms
            if rng.random() < 0.50:
                u = u ** (1.6 + 1.8 * self.risk.boom_bust_risk)
            else:
                u = 1.0 - (1.0 - u) ** (1.6 + 1.8 * self.risk.boom_bust_risk)

        projected = lerp(true_floor, true_ceiling, u)

        # apply support + pressure + random noise
        projected += 0.10 * (support - 0.5) - pressure_penalty
        projected += normalish(rng, 0.0, variance)

        projected = clamp(projected)

        # build a player-ish skill band payload (normalized)
        # mapping: blend prospect signals into an "initial readiness" profile
        # (you can replace this later with your Player 100-attr mapping)
        skating = clamp(0.55 * self.skill_signals.get("skating_potential") + 0.45 * projected)
        puck = clamp(0.55 * self.skill_signals.get("puck_skill_potential") + 0.45 * projected)
        iq = clamp(0.60 * self.skill_signals.get("hockey_iq_potential") + 0.40 * projected)
        physical = clamp(0.50 * self.skill_signals.get("physical_potential") + 0.50 * projected)
        scoring = clamp(0.55 * self.skill_signals.get("scoring_instinct") + 0.45 * projected)
        defense = clamp(0.55 * self.skill_signals.get("defensive_instinct") + 0.45 * projected)

        payload = {
            "source": "prospect_projection",
            "drafted_by_team_id": drafted_by_team_id,
            "identity": asdict(self.identity),
            "development_context": asdict(self.context),
            "prospect_seed": self.seed,
            "projection": {
                "projected_value": projected,
                "true_floor": true_floor,
                "true_ceiling": true_ceiling,
                "total_risk": total_risk,
                "support": support,
                "pressure_penalty": pressure_penalty,
            },
            "bands": {
                "skating": skating,
                "puck_skill": puck,
                "hockey_iq": iq,
                "physical": physical,
                "scoring": scoring,
                "defense": defense,
            },
            "psychology": asdict(self.psychology),
            "risk": asdict(self.risk),
            "narrative": asdict(self.narrative),
            "draft_rank_range": self.draft_rank_range,
            "draft_value_range": self.draft_value_range,
        }
        return payload

    # -----------------------------------------------------------------------
    # Internal: season simulation (abstract, no NHL stats)
    # -----------------------------------------------------------------------

    def _choose_league_for_year(self) -> LeagueType:
        """
        League choice logic:
        - primarily system + age
        - hype can bump exposure leagues (CHL top league, EU top Jr)
        - risk/injury can cause safer leagues
        """
        rng = self._rng
        age = self.age
        system = self.context.development_system

        # baseline
        base = self._default_league_for_age(age, system)

        # in early dev, keep simple
        if age <= 14:
            # AAA if resources/coaching strong
            if (self.context.coaching_quality + self.context.resources) / 2 > 0.65 and rng.random() < 0.55:
                return LeagueType.AAA
            return base

        # structured
        if system == DevelopmentSystem.NCAA:
            # USHL -> NCAA jump (some do at 17/18)
            if age >= 17 and rng.random() < (0.25 + 0.30 * self.skill_signals.get("hockey_iq_potential")):
                return LeagueType.NCAA
            return LeagueType.USHL

        if system == DevelopmentSystem.EURO_JR:
            # top jr vs secondary based on talent + resources
            strength = 0.5 * (self.skill_signals.mean() + self.context.resources)
            if rng.random() < clamp(0.35 + 0.55 * (strength - 0.4)):
                return LeagueType.EURO_JR_TOP
            return LeagueType.EURO_JR_SECONDARY

        if system == DevelopmentSystem.CHL:
            # stick to assigned CHL league
            return base

        # academy/prep: sometimes into USHL/CHL-ish exposure if hype is high
        hype = self.narrative.hype_level
        if age >= 16 and hype > 0.70 and rng.random() < 0.25:
            return LeagueType.USHL

        return base

    def _simulate_season(self, league_type: LeagueType) -> SeasonOutcome:
        rng = self._rng
        lp = LEAGUE_PROFILES.get(league_type, LEAGUE_PROFILES[LeagueType.AAA])

        # usage quality depends on:
        # - ice_time_quality context
        # - confidence/coachability/work ethic
        # - season "luck" (role assignment isn't fully controlled by talent)
        role_noise = normalish(rng, 0.0, 0.10 + 0.10 * lp.game_density)
        usage_quality = clamp(
            0.50 * self.context.ice_time_quality
            + 0.15 * self.psychology.confidence
            + 0.10 * self.psychology.coachability
            + 0.10 * self.psychology.work_ethic
            + role_noise
        )

        # competition is a blend of league strength + context competition level
        competition = clamp(0.55 * self.context.competition_level + 0.45 * clamp(lp.competition_multiplier / 1.2))

        # fatigue increases with game density and physicality
        fatigue = clamp(0.15 + 0.55 * lp.game_density + 0.20 * lp.physicality + normalish(rng, 0.0, 0.06))

        # injury event probability
        # higher fatigue + physicality + injury risk
        injury_p = clamp(
            0.03
            + 0.12 * fatigue
            + 0.10 * lp.physicality
            + 0.22 * self.risk.injury_risk
        )
        injury_event = rng.random() < injury_p

        # season signal: what evaluators "see"
        # influenced by: true underlying signals, usage, competition, fatigue, injury
        truth = self._true_current_signal_strength()
        season_signal = truth
        season_signal += 0.18 * (usage_quality - 0.5)
        season_signal += 0.08 * (lp.exposure_multiplier - 1.0)
        season_signal -= 0.10 * (fatigue - 0.5)
        if injury_event:
            season_signal -= 0.12 + 0.12 * rng.random()

        # high pressure kids can underperform in high exposure leagues (narrative pressure)
        if lp.exposure_multiplier >= 1.0:
            under = (1.0 - self.psychology.pressure_response) * (self.psychology.anxiety) * 0.10
            season_signal -= under

        season_signal += normalish(rng, 0.0, 0.08 + 0.10 * self.curve.volatility)
        season_signal = clamp(season_signal)

        notes: List[str] = []
        if injury_event:
            notes.append("injury_disruption")
        if usage_quality > 0.75:
            notes.append("high_usage_role")
        if usage_quality < 0.35:
            notes.append("low_usage_role")
        if lp.exposure_multiplier > 1.0:
            notes.append("high_exposure_league")

        return SeasonOutcome(
            league_type=league_type,
            age=self.age,
            usage_quality=usage_quality,
            competition=competition,
            season_signal=season_signal,
            fatigue=fatigue,
            injury_event=injury_event,
            notes=notes,
        )

    # -----------------------------------------------------------------------
    # Internal: tournaments (selection + performance)
    # -----------------------------------------------------------------------

    def _simulate_tournaments_for_year(self) -> List[TournamentPerformance]:
        rng = self._rng
        age = self.age
        out: List[TournamentPerformance] = []

        # Not everyone gets selected. Selection depends on:
        # - perceived talent (signals + recent season)
        # - country pipeline strength (approx via resources/competition)
        # - narrative/hype (coaches pick "names" sometimes)
        talent = self._true_current_signal_strength()
        exposure = self._total_exposure_signal()
        hype = self.narrative.hype_level

        selection_strength = clamp(0.55 * talent + 0.20 * exposure + 0.15 * hype + 0.10 * self.psychology.competitiveness)

        # Candidate tournaments by age band
        candidates: List[TournamentType] = []
        if age <= 14:
            candidates = []  # keep it local; you can add regional showcases later
        elif age == 15:
            candidates = [TournamentType.U17_CHALLENGE]
        elif age == 16:
            candidates = [TournamentType.U17_CHALLENGE, TournamentType.HLINKA_GRETZKY]
        elif age == 17:
            candidates = [TournamentType.HLINKA_GRETZKY, TournamentType.U18_WORLDS]
        else:
            # 18 draft year can still have WJC (U20) depending on birthday, but we simplify
            candidates = [TournamentType.WJC_U20, TournamentType.U18_WORLDS]

        for t in candidates:
            tp = TOURNAMENT_PROFILES[t]

            # selection probability
            # harder selection for WJC
            hardness = 0.55 if t in (TournamentType.U17_CHALLENGE,) else 0.65
            if t == TournamentType.WJC_U20:
                hardness = 0.78

            p_select = clamp(0.10 + 0.85 * (selection_strength - hardness))

            if rng.random() >= p_select:
                continue

            # role assignment: depends on talent but has variance
            role = self._assign_tournament_role(talent, tp.role_variance)

            games_played = int(max(2, min(7, round(4 + 2 * rng.random()))))

            # pressure response matters most here
            pressure = tp.pressure_level
            pr = clamp(
                0.55 * self.psychology.pressure_response
                + 0.25 * self.psychology.maturity
                + 0.10 * self.psychology.confidence
                - 0.20 * self.psychology.anxiety * pressure
                + normalish(rng, 0.0, 0.08)
            )

            # production: abstract "impact" in high pressure
            # role influences opportunity; competition influences difficulty
            role_mult = {
                "top_line": 1.10,
                "middle_six": 1.00,
                "depth": 0.85,
                "top_pair": 1.05,
                "third_pair": 0.90,
                "starter": 1.08,
                "backup": 0.85,
            }.get(role, 1.0)

            production = clamp(
                0.55 * talent
                + 0.15 * (self.skill_signals.get("scoring_instinct") - 0.5)
                + 0.10 * (self.skill_signals.get("hockey_iq_potential") - 0.5)
                + 0.15 * (pr - 0.5)
                + 0.10 * (role_mult - 1.0)
                + normalish(rng, 0.0, 0.10 + 0.12 * tp.role_variance)
            )

            # narrative flag triggers: big stage = overreactions
            narrative_flag = None
            if t == TournamentType.WJC_U20:
                if production > 0.78 and pr > 0.70:
                    narrative_flag = "WJC_star_turn"
                elif production < 0.35 and pr < 0.45:
                    narrative_flag = "WJC_exposed"
            else:
                if production > 0.82 and rng.random() < 0.45:
                    narrative_flag = f"{t.value}_breakout"
                elif production < 0.32 and rng.random() < 0.35:
                    narrative_flag = f"{t.value}_struggled"

            out.append(TournamentPerformance(
                tournament_type=t,
                games_played=games_played,
                role=role,
                production=production,
                pressure_response=pr,
                narrative_flag=narrative_flag,
            ))

        return out

    def _assign_tournament_role(self, talent: float, role_variance: float) -> str:
        rng = self._rng
        # Add randomness to role assignment; high variance tournaments create weird usage
        t = clamp(talent + normalish(rng, 0.0, 0.10 + 0.20 * role_variance))

        if self.identity.position == Position.D:
            if t > 0.78:
                return "top_pair"
            if t > 0.55:
                return "third_pair" if rng.random() < 0.35 else "top_pair"
            return "third_pair"
        if self.identity.position == Position.G:
            if t > 0.78:
                return "starter"
            return "backup" if rng.random() < 0.60 else "starter"

        # forwards
        if t > 0.80:
            return "top_line"
        if t > 0.55:
            return "middle_six"
        return "depth"

    # -----------------------------------------------------------------------
    # Internal: narrative updates
    # -----------------------------------------------------------------------

    def _update_narrative_from_year(self, season: SeasonOutcome, tourneys: List[TournamentPerformance]) -> None:
        rng = self._rng

        lp = LEAGUE_PROFILES.get(season.league_type, LEAGUE_PROFILES[LeagueType.AAA])

        # Season impacts hype and reputation:
        # - High exposure leagues move needles more
        exposure_weight = clamp(0.30 + 0.50 * clamp(lp.exposure_multiplier / 1.2))
        delta_season = (season.season_signal - 0.5) * exposure_weight

        # "Momentum": hot/cold streak effect
        self.narrative.draft_momentum = clamp(self.narrative.draft_momentum + 0.35 * delta_season)

        # Reputation lags performance (stickier)
        self.narrative.reputation = clamp(self.narrative.reputation + 0.18 * delta_season)

        # Hype is more volatile and media-driven
        hype_noise = normalish(rng, 0.0, 0.04 + 0.08 * lp.exposure_multiplier)
        self.narrative.hype_level = clamp(self.narrative.hype_level + 0.25 * delta_season + hype_noise)

        # Media attention follows hype + exposure leagues
        self.narrative.media_attention = clamp(
            0.55 * self.narrative.media_attention
            + 0.30 * self.narrative.hype_level
            + 0.15 * clamp(lp.exposure_multiplier / 1.2)
        )

        # Tournament overreactions can override entire seasons
        for tp in tourneys:
            prof = TOURNAMENT_PROFILES[tp.tournament_type]
            swing = (tp.production - 0.5) * prof.narrative_weight * prof.exposure_multiplier
            # pressure response adds to "clutch narrative"
            swing += (tp.pressure_response - 0.5) * 0.60 * prof.narrative_weight

            # one WJC can spike or tank a kid
            self.narrative.draft_momentum = clamp(self.narrative.draft_momentum + 0.45 * swing)
            self.narrative.hype_level = clamp(self.narrative.hype_level + 0.40 * swing)
            self.narrative.media_attention = clamp(self.narrative.media_attention + 0.35 * swing)
            self.narrative.reputation = clamp(self.narrative.reputation + 0.20 * swing)

            if tp.narrative_flag:
                self.narrative.narrative_flags.append(tp.narrative_flag)

        # Keep flags from exploding forever
        if len(self.narrative.narrative_flags) > 24:
            self.narrative.narrative_flags = self.narrative.narrative_flags[-24:]

        self.narrative.clamp_all()

    # -----------------------------------------------------------------------
    # Internal: true underlying development updates
    # -----------------------------------------------------------------------

    def _apply_development_from_year(self, season: SeasonOutcome, tourneys: List[TournamentPerformance]) -> None:
        rng = self._rng

        # Base development rate depends on curve stage
        stage_rate = self.curve.stage_rate(self.age)

        # Development quality multiplier depends on context + psychology
        dev_env = (
            0.35 * self.context.coaching_quality
            + 0.25 * self.context.resources
            + 0.20 * self.context.ice_time_quality
            + 0.20 * self.context.competition_level
        )

        mindset = (
            0.25 * self.psychology.work_ethic
            + 0.20 * self.psychology.coachability
            + 0.20 * self.psychology.competitiveness
            + 0.15 * self.psychology.maturity
            + 0.10 * self.psychology.confidence
            - 0.10 * self.psychology.anxiety
        )

        # Season performance can accelerate confidence-driven growth (or slow it)
        feedback = (season.season_signal - 0.5)

        # Injury event can stall development this year (or redirect it: skill->iq, etc.)
        injury_stall = 0.0
        if season.injury_event:
            injury_stall = 0.15 + 0.25 * self.risk.injury_risk

        # Plateau risk: some kids hit a wall
        plateau = (rng.random() < (self.curve.plateau_risk * (0.70 + 0.60 * self.risk.development_risk)))
        plateau_mult = 0.55 if plateau else 1.0

        # Tournament reps can give small true-signal boosts (esp IQ/pressure)
        t_boost = 0.0
        if tourneys:
            # if they handled pressure well, it can become real growth
            avg_pr = sum(tp.pressure_response for tp in tourneys) / len(tourneys)
            t_boost = 0.08 * (avg_pr - 0.5)

        # Total effective growth budget
        growth_budget = stage_rate
        growth_budget *= (0.55 + 0.70 * dev_env)
        growth_budget *= (0.55 + 0.70 * mindset)
        growth_budget *= (0.85 + 0.30 * feedback)
        growth_budget *= plateau_mult
        growth_budget -= injury_stall
        growth_budget += t_boost

        # volatility = random swings (some years are weird)
        growth_budget += normalish(rng, 0.0, 0.05 + 0.12 * self.curve.volatility)
        growth_budget = max(-0.15, min(0.22, growth_budget))  # keep sane

        # Apply growth to signals, with curve-type shaping
        self._apply_growth_to_signals(growth_budget, season, plateau)

        # Psychology can shift subtly year to year (confidence/maturity)
        self._apply_psychology_drift(season, tourneys)

        # Risk can crystalize with time (draft year)
        self._apply_risk_drift(season, tourneys)

    def _apply_growth_to_signals(self, growth_budget: float, season: SeasonOutcome, plateau: bool) -> None:
        rng = self._rng
        ct = self.curve.curve_type

        # choose emphasis weights by curve type
        weights = {k: 1.0 for k in DEFAULT_SIGNAL_KEYS}

        if ct == CurveType.PHYSICAL_FIRST:
            weights["physical_potential"] = 1.45
            weights["skating_potential"] = 1.10
            weights["hockey_iq_potential"] = 0.85
        elif ct == CurveType.IQ_FIRST:
            weights["hockey_iq_potential"] = 1.45
            weights["defensive_instinct"] = 1.15
            weights["physical_potential"] = 0.85
        elif ct == CurveType.PRODIGY:
            weights["skating_potential"] = 1.20
            weights["puck_skill_potential"] = 1.20
            weights["scoring_instinct"] = 1.15
        elif ct == CurveType.LATE_BLOOMER:
            # late bloomers: early years add less, later years add more (handled via stage_rate)
            weights["hockey_iq_potential"] = 1.10
            weights["physical_potential"] = 1.10
        elif ct == CurveType.BOOM_BUST:
            # chaotic: some signals jump, others lag
            pass

        # normalize weights
        s = sum(weights.values())
        for k in weights:
            weights[k] /= s

        # Season environment influences what "improves"
        # high physical leagues push physical/defense; high skill bias helps skill
        lp = LEAGUE_PROFILES.get(season.league_type, LEAGUE_PROFILES[LeagueType.AAA])

        env_push = {
            "physical_potential": 0.12 * (lp.physicality - 0.5),
            "defensive_instinct": 0.08 * (lp.physicality - 0.5),
            "puck_skill_potential": 0.10 * (lp.skill_bias - 0.5),
            "skating_potential": 0.08 * (lp.skill_bias - 0.5),
            "hockey_iq_potential": 0.06 * (self.context.coaching_quality - 0.5),
            "scoring_instinct": 0.07 * (season.usage_quality - 0.5),
        }

        # boom-bust: apply randomness to which signals "pop"
        pop_key = None
        if self.curve.curve_type == CurveType.BOOM_BUST and rng.random() < 0.55:
            pop_key = weighted_choice(rng, [(k, 1.0) for k in DEFAULT_SIGNAL_KEYS])

        for k in DEFAULT_SIGNAL_KEYS:
            val = self.skill_signals.get(k)

            # base delta scaled by weight and growth_budget
            delta = growth_budget * (0.75 + 0.70 * weights[k])

            # environmental push
            delta += env_push.get(k, 0.0)

            # plateau reduces progress; sometimes even regression in one area
            if plateau and rng.random() < 0.18:
                delta -= 0.02 + 0.05 * rng.random()

            # boom-bust pop
            if pop_key == k:
                delta += 0.05 + 0.08 * rng.random()
            elif pop_key is not None and rng.random() < 0.12:
                delta -= 0.03 * rng.random()

            # ceiling compression: improvements slow near 1.0
            ceiling_drag = (val - 0.75) * 0.35 if val > 0.75 else 0.0
            delta -= ceiling_drag

            # clamp change
            delta = max(-0.08, min(0.10, delta))

            self.skill_signals.set(k, val + delta)

    def _apply_psychology_drift(self, season: SeasonOutcome, tourneys: List[TournamentPerformance]) -> None:
        rng = self._rng

        # confidence responds to season signal and tournaments
        self.psychology.confidence = clamp(self.psychology.confidence + 0.18 * (season.season_signal - 0.5))
        self.psychology.anxiety = clamp(self.psychology.anxiety - 0.10 * (season.season_signal - 0.5))

        # maturity increases with age (slight), but can be slowed by hype chaos
        self.psychology.maturity = clamp(self.psychology.maturity + 0.03 + 0.02 * (self.age / 18.0) - 0.03 * (self.narrative.media_attention - 0.5))

        # pressure response can improve after big events
        if tourneys:
            avg_pr = sum(tp.pressure_response for tp in tourneys) / len(tourneys)
            self.psychology.pressure_response = clamp(
                self.psychology.pressure_response + 0.10 * (avg_pr - 0.5) + normalish(rng, 0.0, 0.02)
            )

        # if injury, anxiety can spike a bit
        if season.injury_event:
            self.psychology.anxiety = clamp(self.psychology.anxiety + 0.06 + 0.06 * rng.random())
            self.psychology.confidence = clamp(self.psychology.confidence - 0.04 * rng.random())

        self.psychology.clamp_all()

    def _apply_risk_drift(self, season: SeasonOutcome, tourneys: List[TournamentPerformance]) -> None:
        rng = self._rng

        # injury risk slightly increases if repeated injuries
        if season.injury_event:
            self.risk.injury_risk = clamp(self.risk.injury_risk + 0.03 + 0.05 * rng.random())

        # psychological risk: tied to anxiety and pressure response
        self.risk.psychological_risk = clamp(
            0.60 * self.risk.psychological_risk
            + 0.25 * self.psychology.anxiety
            + 0.15 * (1.0 - self.psychology.pressure_response)
        )

        # competition translation risk: if dominating low competition / sheltered usage
        low_comp = clamp(1.0 - season.competition)
        sheltered = clamp(0.5 - (season.usage_quality - 0.5))
        self.risk.competition_translation_risk = clamp(
            self.risk.competition_translation_risk + 0.06 * (low_comp - 0.5) + 0.04 * (sheltered - 0.5)
                )
        # boom/bust risk can creep up if volatility + inconsistent seasons/tourneys
        inconsistency = 0.0
        if tourneys:
            prod_var = 0.0
            if len(tourneys) >= 2:
                m = sum(tp.production for tp in tourneys) / len(tourneys)
                prod_var = sum((tp.production - m) ** 2 for tp in tourneys) / len(tourneys)
            inconsistency += clamp(prod_var * 2.0) * 0.06

        # if season was noisy relative to truth, increase boom-bust perception risk
        truth = self._true_current_signal_strength()
        performance_gap = abs(season.season_signal - truth)
        inconsistency += clamp(performance_gap * 1.8) * 0.05

        self.risk.boom_bust_risk = clamp(
            0.75 * self.risk.boom_bust_risk
            + 0.15 * self.curve.volatility
            + 0.10 * inconsistency
        )

        # development risk tightens as draft year approaches (projection crystallizes)
        if self.phase == ProspectPhase.DRAFT_YEAR:
            self.risk.development_risk = clamp(
                0.70 * self.risk.development_risk
                + 0.20 * (1.0 - self.psychology.work_ethic)
                + 0.10 * (1.0 - self.context.coaching_quality)
            )

        self.risk.clamp_all()

    # -----------------------------------------------------------------------
    # Internal: body growth / frame projection
    # -----------------------------------------------------------------------

    def _apply_growth_spurt(self) -> None:
        """
        Height/weight changes matter because scouts project frames.

        - 12–15: highest chance of large height jumps
        - 16–17: some late growth (important for "late frame" kids)
        - 18: minimal

        Weight follows height + physical potential + resources, with randomness.
        """
        rng = self._rng
        age = self.age

        # base probabilities of a notable height jump
        if age <= 12:
            p_spurt = 0.10
            max_cm = 5
        elif age <= 14:
            p_spurt = 0.22
            max_cm = 7
        elif age == 15:
            p_spurt = 0.18
            max_cm = 6
        elif age == 16:
            p_spurt = 0.12
            max_cm = 5
        elif age == 17:
            p_spurt = 0.08
            max_cm = 4
        else:
            p_spurt = 0.04
            max_cm = 3

        # some curve types skew growth timing
        if self.curve.curve_type == CurveType.LATE_BLOOMER:
            # late bloomers: a bit more late growth chance
            if age >= 15:
                p_spurt += 0.05
        if self.curve.curve_type == CurveType.PHYSICAL_FIRST:
            p_spurt += 0.03

        # resources / nutrition can amplify the probability slightly
        p_spurt = clamp(p_spurt + 0.06 * (self.context.resources - 0.5))

        grew_cm = 0
        if rng.random() < p_spurt:
            grew_cm = int(max(1, round(1 + rng.random() * (max_cm - 1))))
            # rare BIG spurt event
            if rng.random() < 0.07 and age <= 15:
                grew_cm += int(1 + 2 * rng.random())

        # smaller incremental growth even without a spurt
        if grew_cm == 0 and age <= 16 and rng.random() < 0.55:
            grew_cm = 1 if rng.random() < 0.75 else 2

        if grew_cm > 0:
            self.identity.height_cm = int(min(210, self.identity.height_cm + grew_cm))

        # Weight gain: driven by physical potential + resources + age + (height change)
        phys = self.skill_signals.get("physical_potential")
        base_gain = 0.8 + 2.2 * rng.random()  # kg
        base_gain += 1.2 * (phys - 0.5)
        base_gain += 0.8 * (self.context.resources - 0.5)
        base_gain += 0.4 * (self.context.competition_level - 0.5)  # strength programs in higher comp
        base_gain += 0.6 * grew_cm  # taller → usually heavier over time

        # injuries can stall weight gain
        if self.season_history and self.season_history[-1].injury_event:
            base_gain -= 0.6 + 0.8 * rng.random()

        # keep sane
        base_gain = max(-1.0, min(5.5, base_gain))
        if abs(base_gain) > 0.05:
            self.identity.weight_kg = int(max(45, min(125, round(self.identity.weight_kg + base_gain))))

    def _size_projection_score(self) -> float:
        """
        0..1 score for "NHL frame projection" (NOT a rating).
        Scouts (and teams) over/under-weight it depending on philosophy.

        Uses:
        - current height/weight
        - age (younger big kids look different than older big kids)
        - physical_potential (future frame / strength capacity)
        """
        h = float(self.identity.height_cm)
        w = float(self.identity.weight_kg)
        age = float(self.age)
        phys = self.skill_signals.get("physical_potential")

        # typical NHL-ish frame centering (rough, for signal purposes only)
        # 178–193 cm range heavily matters for skaters; goalies skew bigger but keep general
        h_score = clamp((h - 168.0) / 28.0)  # 168->0, 196->1
        w_score = clamp((w - 62.0) / 30.0)   # 62->0, 92->1

        # younger kids get more "projection uncertainty"
        # big at 13 isn't the same as big at 17; compress extremes at young ages
        age_t = clamp((age - 12.0) / 6.0)  # 12->0, 18->1
        compress = 0.65 + 0.35 * age_t

        base = 0.55 * h_score + 0.45 * w_score
        base = 0.5 + (base - 0.5) * compress

        # physical potential nudges the projection score
        base += 0.10 * (phys - 0.5)

        # positional framing: goalies/defense get a slight systematic boost
        if self.identity.position == Position.G:
            base += 0.05
        elif self.identity.position == Position.D:
            base += 0.03

        return clamp(base)

    # -----------------------------------------------------------------------
    # Internal: "truth" functions (latent reality behind scouting)
    # -----------------------------------------------------------------------

    def _true_current_signal_strength(self) -> float:
        """
        A latent "how good this prospect looks right now" truth signal, 0..1.
        This is NOT a rating and NOT deterministic performance.

        Built from:
        - skill signal blend (position-weighted)
        - psychology exposure to pressure (small)
        - context (competition/coaching) modestly amplifies real growth
        """
        s = self.skill_signals

        # position-weighted blend
        if self.identity.position == Position.G:
            # for now, goalies still use same bands; treat IQ + physical + pressure a bit higher
            base = (
                0.18 * s.get("skating_potential")
                + 0.16 * s.get("puck_skill_potential")
                + 0.22 * s.get("hockey_iq_potential")
                + 0.22 * s.get("physical_potential")
                + 0.10 * s.get("defensive_instinct")
                + 0.12 * self.psychology.pressure_response
            )
        elif self.identity.position == Position.D:
            base = (
                0.18 * s.get("skating_potential")
                + 0.14 * s.get("puck_skill_potential")
                + 0.18 * s.get("hockey_iq_potential")
                + 0.18 * s.get("physical_potential")
                + 0.10 * s.get("scoring_instinct")
                + 0.22 * s.get("defensive_instinct")
            )
        elif self.identity.position == Position.C:
            base = (
                0.18 * s.get("skating_potential")
                + 0.16 * s.get("puck_skill_potential")
                + 0.24 * s.get("hockey_iq_potential")
                + 0.12 * s.get("physical_potential")
                + 0.18 * s.get("scoring_instinct")
                + 0.12 * s.get("defensive_instinct")
            )
        else:
            # wingers: scoring/skill heavier
            base = (
                0.20 * s.get("skating_potential")
                + 0.20 * s.get("puck_skill_potential")
                + 0.16 * s.get("hockey_iq_potential")
                + 0.12 * s.get("physical_potential")
                + 0.22 * s.get("scoring_instinct")
                + 0.10 * s.get("defensive_instinct")
            )

        # context amplifies real-world translation modestly (good coaching/resources help actual growth show up)
        env = 0.40 * self.context.coaching_quality + 0.30 * self.context.resources + 0.30 * self.context.competition_level
        base += 0.06 * (env - 0.5)

        # psychology: confidence helps show it; anxiety can suppress
        base += 0.04 * (self.psychology.confidence - 0.5)
        base -= 0.03 * (self.psychology.anxiety - 0.5)

        # small dampener if heavy plateau risk and already in structured years
        if self.age >= 15:
            base -= 0.03 * (self.curve.plateau_risk - 0.5)

        return clamp(base)

    def _true_ceiling_value(self) -> float:
        """
        Latent ceiling-ish outcome (0..1) derived from:
        - top-end signals (skill mean + best two areas)
        - curve late growth and volatility (volatility increases tail)
        - narrative DOES NOT directly change truth
        """
        s = self.skill_signals
        vals = [s.get(k) for k in DEFAULT_SIGNAL_KEYS]
        vals_sorted = sorted(vals, reverse=True)
        top2 = (vals_sorted[0] + vals_sorted[1]) / 2.0
        mean = sum(vals) / len(vals)

        # late growth matters for ceiling
        late = self.curve.late_growth_rate
        mid = self.curve.mid_growth_rate

        # volatility creates upside tail (but also downside; handled in floor)
        vol = self.curve.volatility

        # base ceiling
        ceiling = 0.50 * mean + 0.35 * top2 + 0.10 * late + 0.05 * mid
        ceiling += 0.06 * vol

        # physical frame can elevate ceiling for certain archetypes (D, G, power forwards)
        ceiling += 0.04 * (self._size_projection_score() - 0.5)

        # cap
        return clamp(ceiling)

    def _true_floor_value(self) -> float:
        """
        Latent floor-ish outcome (0..1) derived from:
        - weakest signals (bottom two)
        - IQ/defense as stabilizers
        - risk penalties (development, injury, psychological)
        """
        s = self.skill_signals
        vals = [s.get(k) for k in DEFAULT_SIGNAL_KEYS]
        vals_sorted = sorted(vals)
        bottom2 = (vals_sorted[0] + vals_sorted[1]) / 2.0
        mean = sum(vals) / len(vals)

        stabilizers = 0.55 * s.get("hockey_iq_potential") + 0.45 * s.get("defensive_instinct")

        total_risk = self.risk.total_risk()
        risk_penalty = 0.18 * (total_risk - 0.5) + 0.10 * (self.risk.psychological_risk - 0.5)

        floor = 0.45 * mean + 0.35 * bottom2 + 0.20 * stabilizers
        floor -= risk_penalty

        # boom-bust kids have softer floors
        floor -= 0.10 * (self.risk.boom_bust_risk - 0.5)

        # pressure fragility reduces floor in big moments
        floor -= 0.05 * ((1.0 - self.psychology.pressure_response) - 0.5)

        return clamp(floor)

    def _total_exposure_signal(self) -> float:
        """
        Exposure is what drives:
        - scouting certainty
        - hype accelerants
        - draft momentum volatility

        Built from:
        - current league exposure multiplier
        - tournament exposure multipliers (recent year weighted)
        - media attention (feedback loop, but NOT truth)
        """
        lp = LEAGUE_PROFILES.get(self.current_league, LEAGUE_PROFILES[LeagueType.AAA])
        league_exp = clamp(lp.exposure_multiplier / 1.2)  # normalize-ish around CHL exposure

        # tournament exposure: last 2 years count more
        tour_exp = 0.0
        if self.tournament_history:
            # only use recent-ish tournaments (age >= current-2)
            recent = [tp for tp in self.tournament_history if tp.games_played > 0]
            if recent:
                # weight by tournament profile exposure and narrative weight
                wsum = 0.0
                vsum = 0.0
                for tp in recent[-6:]:
                    prof = TOURNAMENT_PROFILES[tp.tournament_type]
                    w = 0.6 + 0.4 * prof.exposure_multiplier
                    wsum += w
                    vsum += w * clamp(prof.exposure_multiplier / 1.8)
                tour_exp = (vsum / wsum) if wsum > 0 else 0.0

        media = self.narrative.media_attention
        hype = self.narrative.hype_level

        # exposure is mostly league+tournaments; media/hype add a smaller reinforcement
        exp = 0.55 * league_exp + 0.30 * tour_exp + 0.10 * media + 0.05 * hype
        return clamp(exp)

    # -----------------------------------------------------------------------
    # Internal: draft range generation (rank range + value range)
    # -----------------------------------------------------------------------

    def _finalize_draft_ranges(self, estimated_class_size: int = 224) -> None:
        """
        Produces:
        - draft_value_range: (floor-ish, ceiling-ish) in 0..1
        - draft_rank_range: (best_pick, worst_pick)

        Important: This is NOT a single rank.
        It's a range to represent uncertainty + disagreement.
        """
        rng = self._rng

        # true bands
        floor = self._true_floor_value()
        ceil = self._true_ceiling_value()

        # perception amplifiers: exposure & narrative create volatility in ranking
        exposure = self._total_exposure_signal()
        hype = self.narrative.hype_level
        momentum = self.narrative.draft_momentum

        # risk widens the range (boom/bust kids have massive rank spreads)
        total_risk = self.risk.total_risk()
        widen = (
            0.18
            + 0.28 * total_risk
            + 0.20 * self.curve.volatility
            + 0.10 * self.risk.boom_bust_risk
        )

        # more exposure tightens uncertainty (teams have more looks)
        widen *= (1.05 - 0.45 * exposure)

        # hype/momentum can distort consensus (wider volatility in where teams slot them)
        widen *= (1.00 + 0.25 * abs(hype - 0.5) + 0.20 * abs(momentum - 0.5))

        # compute value range around truth, with asymmetry:
        # high risk increases downside tail more than upside (since upside already in ceiling)
        value_floor = clamp(floor - (0.10 + 0.20 * total_risk + 0.08 * self.curve.volatility))
        value_ceil = clamp(ceil + (0.04 + 0.10 * self.curve.volatility) - 0.04 * (total_risk - 0.5))

        # add small randomness so draft classes aren't static
        value_floor = clamp(value_floor + normalish(rng, 0.0, 0.02))
        value_ceil = clamp(value_ceil + normalish(rng, 0.0, 0.02))

        if value_floor > value_ceil:
            value_floor, value_ceil = value_ceil, value_floor

        self.draft_value_range = (value_floor, value_ceil)

        # convert "consensus value" to pick range.
        # consensus uses ceiling more for rebuild-chasers, but floor matters for safety.
        consensus = clamp(0.55 * value_ceil + 0.45 * value_floor)

        # map value -> expected pick (nonlinear: top picks are harder to reach)
        # value 0.90+ should cluster near top 5; value 0.70 around mid 1st-ish; etc.
        # We'll use a curve that pushes high values toward low pick numbers.
        # pick_score = 1 means pick 1, 0 means last pick.
        pick_score = consensus ** 2.2
        expected_pick = int(round(1 + (estimated_class_size - 1) * (1.0 - pick_score)))

        # widen translates to pick spread
        spread = int(round(8 + widen * 60))

        # exposure tightens spread a bit more
        spread = int(round(spread * (1.10 - 0.35 * exposure)))

        # keep within class size
        best = max(1, expected_pick - spread)
        worst = min(estimated_class_size, expected_pick + spread)

        # narrative can cause "late riser" / "early lock" effects:
        # strong momentum can tighten the best side (teams agree he's high)
        if momentum > 0.70 and exposure > 0.55:
            best = max(1, best - int(2 + 6 * (momentum - 0.70)))
            worst = min(estimated_class_size, worst - int(1 + 4 * (momentum - 0.70)))
        elif momentum < 0.35 and total_risk > 0.60:
            # cold streak + risk: teams drop him further
            worst = min(estimated_class_size, worst + int(2 + 8 * (0.35 - momentum)))

        # ensure order
        if best > worst:
            best, worst = worst, best

        self.draft_rank_range = (best, worst)

    # -----------------------------------------------------------------------
    # END
    # -----------------------------------------------------------------------

