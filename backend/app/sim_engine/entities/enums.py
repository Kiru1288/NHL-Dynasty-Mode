# backend/app/sim_engine/entities/enums.py
"""
NHL Franchise Mode — Enums Ontology

This module defines the canonical discrete classifications used across the sim engine.
These enums are designed to be:
- stable (values must not change once introduced)
- serializable (stored as strings in DB/JSON)
- human-readable (logs/debug)
- round-trippable (Enum(value) works)

Usage guideline:
- In-memory: store enum instances (e.g., TeamStatus.REBUILDING)
- Persisted: store enum.value (string)
- Load: TeamStatus(db_value)

NOTE: Do not put numeric logic in enums. Enums define meaning, not math.
"""

from __future__ import annotations

from enum import Enum

# Python 3.11 compatibility: StrEnum exists in 3.11+
try:
    from enum import StrEnum  # type: ignore
except Exception:  # pragma: no cover
    class StrEnum(str, Enum):  # fallback, should not hit on 3.11+
        pass


# ============================================================
# CORE SHARED ENUMS
# ============================================================

class Position(StrEnum):
    C = "C"
    LW = "LW"
    RW = "RW"
    LD = "LD"
    RD = "RD"
    G = "G"


class ShotHandedness(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class PlayerType(StrEnum):
    PROSPECT = "PROSPECT"
    DRAFT_ELIGIBLE = "DRAFT_ELIGIBLE"
    ROOKIE = "ROOKIE"
    NHL_PLAYER = "NHL_PLAYER"
    AHL_PLAYER = "AHL_PLAYER"
    DEPTH_PLAYER = "DEPTH_PLAYER"
    FRINGE_PLAYER = "FRINGE_PLAYER"
    VETERAN = "VETERAN"
    DECLINING_VETERAN = "DECLINING_VETERAN"
    FRANCHISE_CORNERSTONE = "FRANCHISE_CORNERSTONE"
    JOURNEYMAN = "JOURNEYMAN"
    RETIRED = "RETIRED"


# ============================================================
# PLAYER & PROSPECT ENUMS
# ============================================================

class PlayerArchetype(StrEnum):
    # Forwards
    SNIPER = "SNIPER"
    PLAYMAKER = "PLAYMAKER"
    POWER_FORWARD = "POWER_FORWARD"
    TWO_WAY_FORWARD = "TWO_WAY_FORWARD"
    GRINDER = "GRINDER"
    ENFORCER = "ENFORCER"
    CHECKING_FORWARD = "CHECKING_FORWARD"

    # Defense
    DEFENSIVE_DEFENSEMAN = "DEFENSIVE_DEFENSEMAN"
    OFFENSIVE_DEFENSEMAN = "OFFENSIVE_DEFENSEMAN"
    PUCK_MOVING_DEFENSEMAN = "PUCK_MOVING_DEFENSEMAN"
    STAY_AT_HOME_DEFENSEMAN = "STAY_AT_HOME_DEFENSEMAN"
    HYBRID_DEFENSEMAN = "HYBRID_DEFENSEMAN"

    # Goalies
    BUTTERFLY_GOALIE = "BUTTERFLY_GOALIE"
    POSITIONAL_GOALIE = "POSITIONAL_GOALIE"
    ATHLETIC_GOALIE = "ATHLETIC_GOALIE"
    HYBRID_GOALIE = "HYBRID_GOALIE"


class DevelopmentCurve(StrEnum):
    EARLY_BOOMER = "EARLY_BOOMER"
    LATE_BLOOMER = "LATE_BLOOMER"
    LINEAR = "LINEAR"
    VOLATILE = "VOLATILE"
    PLATEAU_EARLY = "PLATEAU_EARLY"
    SLOW_AND_STEADY = "SLOW_AND_STEADY"
    ELITE_TRAJECTORY = "ELITE_TRAJECTORY"
    PEAK_AND_CRASH = "PEAK_AND_CRASH"
    BUST_RISK = "BUST_RISK"


class MentalTrait(StrEnum):
    LEADER = "LEADER"
    COMPETITIVE = "COMPETITIVE"
    COACHABLE = "COACHABLE"
    STUBBORN = "STUBBORN"
    MERCURIAL = "MERCURIAL"
    WORK_ETHIC_HIGH = "WORK_ETHIC_HIGH"
    WORK_ETHIC_LOW = "WORK_ETHIC_LOW"
    CONFIDENCE_DEPENDENT = "CONFIDENCE_DEPENDENT"
    PRESSURE_PROOF = "PRESSURE_PROOF"
    PLAYOFF_RISER = "PLAYOFF_RISER"
    PLAYOFF_SHRINKER = "PLAYOFF_SHRINKER"
    EMOTIONALLY_VOLATILE = "EMOTIONALLY_VOLATILE"


class InjuryProneness(StrEnum):
    IRONMAN = "IRONMAN"
    LOW = "LOW"
    AVERAGE = "AVERAGE"
    HIGH = "HIGH"
    FRAGILE = "FRAGILE"


class ConsistencyType(StrEnum):
    CONSISTENT = "CONSISTENT"
    STREAKY = "STREAKY"
    NIGHT_TO_NIGHT = "NIGHT_TO_NIGHT"
    HIGH_VARIANCE = "HIGH_VARIANCE"


# ============================================================
# PROSPECT-SPECIFIC ENUMS
# ============================================================

class ProspectTier(StrEnum):
    GENERATIONAL = "GENERATIONAL"
    ELITE = "ELITE"
    TOP_LINE = "TOP_LINE"
    TOP_PAIR = "TOP_PAIR"
    SOLID_PRO = "SOLID_PRO"
    PROJECT = "PROJECT"
    LONGSHOT = "LONGSHOT"
    CAMP_BODY = "CAMP_BODY"


class ScoutingCertainty(StrEnum):
    CAN_T_MISS = "CAN_T_MISS"
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"
    WILD_CARD = "WILD_CARD"


class ProspectNarrative(StrEnum):
    NEXT_SUPERSTAR = "NEXT_SUPERSTAR"
    SAFE_PICK = "SAFE_PICK"
    RAW_BUT_INTRIGUING = "RAW_BUT_INTRIGUING"
    BOOM_OR_BUST = "BOOM_OR_BUST"
    OVERHYPED = "OVERHYPED"
    UNDER_THE_RADAR = "UNDER_THE_RADAR"
    LATE_RISER = "LATE_RISER"
    PHYSICAL_SPECIMEN = "PHYSICAL_SPECIMEN"


class DevelopmentEnvironment(StrEnum):
    AAA_MINOR = "AAA_MINOR"
    PREP_SCHOOL = "PREP_SCHOOL"
    EURO_JUNIOR = "EURO_JUNIOR"
    NCAA = "NCAA"
    CHL = "CHL"
    USHL = "USHL"
    PRO_OVERSEAS = "PRO_OVERSEAS"
    DEVELOPMENT_HELL = "DEVELOPMENT_HELL"


# ============================================================
# TEAM ENUMS
# ============================================================

class TeamStatus(StrEnum):
    REBUILDING = "REBUILDING"
    TRANSITIONING = "TRANSITIONING"
    BUBBLE = "BUBBLE"
    CONTENDER = "CONTENDER"
    WIN_NOW = "WIN_NOW"
    DYNASTY = "DYNASTY"
    COLLAPSING = "COLLAPSING"


class MarketType(StrEnum):
    BIG_MARKET = "BIG_MARKET"
    SMALL_MARKET = "SMALL_MARKET"
    TRADITIONAL_HOCKEY_MARKET = "TRADITIONAL_HOCKEY_MARKET"
    NON_TRADITIONAL = "NON_TRADITIONAL"
    HIGH_PRESSURE = "HIGH_PRESSURE"
    LOW_PRESSURE = "LOW_PRESSURE"
    MEDIA_CIRCUS = "MEDIA_CIRCUS"


class OwnershipStability(StrEnum):
    ROCK_SOLID = "ROCK_SOLID"
    STABLE = "STABLE"
    QUESTIONABLE = "QUESTIONABLE"
    CHAOTIC = "CHAOTIC"
    FOR_SALE = "FOR_SALE"
    RELOCATING = "RELOCATING"


class TeamCulture(StrEnum):
    PLAYER_FRIENDLY = "PLAYER_FRIENDLY"
    HARD_NOSED = "HARD_NOSED"
    DEVELOPMENT_FOCUSED = "DEVELOPMENT_FOCUSED"
    ANALYTICS_DRIVEN = "ANALYTICS_DRIVEN"
    OLD_SCHOOL = "OLD_SCHOOL"
    WIN_AT_ALL_COSTS = "WIN_AT_ALL_COSTS"
    CHEAP = "CHEAP"


# ============================================================
# COACH ENUMS
# ============================================================

class CoachRole(StrEnum):
    HEAD_COACH = "HEAD_COACH"
    ASSISTANT_COACH = "ASSISTANT_COACH"
    GOALIE_COACH = "GOALIE_COACH"
    DEVELOPMENT_COACH = "DEVELOPMENT_COACH"
    SPECIAL_TEAMS_COACH = "SPECIAL_TEAMS_COACH"


class CoachingStyle(StrEnum):
    DEFENSIVE_SYSTEM = "DEFENSIVE_SYSTEM"
    AGGRESSIVE_FORECHECK = "AGGRESSIVE_FORECHECK"
    TRANSITION_HEAVY = "TRANSITION_HEAVY"
    POSSESSION_FOCUSED = "POSSESSION_FOCUSED"
    DUMP_AND_CHASE = "DUMP_AND_CHASE"
    ADAPTIVE = "ADAPTIVE"
    RIGID = "RIGID"


class CoachPersonality(StrEnum):
    PLAYER_FRIENDLY = "PLAYER_FRIENDLY"
    AUTHORITARIAN = "AUTHORITARIAN"
    MOTIVATOR = "MOTIVATOR"
    TACTICIAN = "TACTICIAN"
    DEVELOPMENT_FIRST = "DEVELOPMENT_FIRST"
    MEDIA_SAVVY = "MEDIA_SAVVY"
    OLD_SCHOOL = "OLD_SCHOOL"


# ============================================================
# CONTRACT & ECONOMICS ENUMS
# ============================================================

class ContractType(StrEnum):
    ELC = "ELC"
    BRIDGE = "BRIDGE"
    STANDARD = "STANDARD"
    EXTENSION = "EXTENSION"
    VETERAN_MIN = "VETERAN_MIN"
    ONE_YEAR_PROVE_IT = "ONE_YEAR_PROVE_IT"


class ContractClause(StrEnum):
    NO_TRADE = "NO_TRADE"
    MODIFIED_NO_TRADE = "MODIFIED_NO_TRADE"
    NO_MOVE = "NO_MOVE"
    SIGNING_BONUS_HEAVY = "SIGNING_BONUS_HEAVY"
    FRONT_LOADED = "FRONT_LOADED"
    BACK_LOADED = "BACK_LOADED"
    PERFORMANCE_BONUS = "PERFORMANCE_BONUS"


class SalaryExpectation(StrEnum):
    TEAM_FRIENDLY = "TEAM_FRIENDLY"
    MARKET_VALUE = "MARKET_VALUE"
    PREMIUM = "PREMIUM"
    OVERPAY_REQUIRED = "OVERPAY_REQUIRED"
    HOMETOWN_DISCOUNT = "HOMETOWN_DISCOUNT"


# ============================================================
# LEAGUE & SEASON ENUMS
# ============================================================

class LeaguePhase(StrEnum):
    PRESEASON = "PRESEASON"
    REGULAR_SEASON = "REGULAR_SEASON"
    TRADE_DEADLINE = "TRADE_DEADLINE"
    PLAYOFFS = "PLAYOFFS"
    DRAFT = "DRAFT"
    FREE_AGENCY = "FREE_AGENCY"
    OFFSEASON = "OFFSEASON"


class SeasonOutcome(StrEnum):
    MISSED_PLAYOFFS = "MISSED_PLAYOFFS"
    FIRST_ROUND_EXIT = "FIRST_ROUND_EXIT"
    SECOND_ROUND_EXIT = "SECOND_ROUND_EXIT"
    CONFERENCE_FINAL = "CONFERENCE_FINAL"
    CUP_FINAL_LOSS = "CUP_FINAL_LOSS"
    STANLEY_CUP_CHAMPION = "STANLEY_CUP_CHAMPION"


# ============================================================
# GAMEPLAY & SIMULATION ENUMS
# ============================================================

class PenaltyType(StrEnum):
    MINOR = "MINOR"
    DOUBLE_MINOR = "DOUBLE_MINOR"
    MAJOR = "MAJOR"
    MISCONDUCT = "MISCONDUCT"
    GAME_MISCONDUCT = "GAME_MISCONDUCT"


class GoalType(StrEnum):
    EVEN_STRENGTH = "EVEN_STRENGTH"
    POWER_PLAY = "POWER_PLAY"
    SHORT_HANDED = "SHORT_HANDED"
    EMPTY_NET = "EMPTY_NET"
    OVERTIME = "OVERTIME"
    SHOOTOUT = "SHOOTOUT"


class GameIntensity(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    PLAYOFF = "PLAYOFF"


# ============================================================
# NARRATIVE & MEDIA ENUMS
# ============================================================

class MediaSentiment(StrEnum):
    PRAISE = "PRAISE"
    CRITICISM = "CRITICISM"
    HYPE = "HYPE"
    DOUBT = "DOUBT"
    CONTROVERSY = "CONTROVERSY"
    APATHY = "APATHY"


class NarrativeEventType(StrEnum):
    BREAKOUT_SEASON = "BREAKOUT_SEASON"
    REGRESSION = "REGRESSION"
    TRADE_REQUEST = "TRADE_REQUEST"
    LOCKER_ROOM_DRAMA = "LOCKER_ROOM_DRAMA"
    COACH_ON_HOT_SEAT = "COACH_ON_HOT_SEAT"
    FAN_BACKLASH = "FAN_BACKLASH"
    LEGACY_MOMENT = "LEGACY_MOMENT"


# ============================================================
# SYSTEM & META ENUMS
# ============================================================

class SimulationEventSeverity(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AIConfidenceLevel(StrEnum):
    CERTAIN = "CERTAIN"
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    DESPERATE = "DESPERATE"
