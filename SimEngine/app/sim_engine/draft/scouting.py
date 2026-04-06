# backend/app/sim_engine/scouting/scouting.py
"""
Scouting System (Perception Engine) — NHL Franchise Mode Sim

Core ideas:
- Scouting ≠ truth.
- Scouts observe partial, noisy, biased signals that become more reliable with more viewings.
- Reports decay (staleness) unless refreshed.
- Tournaments create "event shocks" (more visibility + more hype + more overreaction).
- Teams aggregate scout reports into a team belief model: tiers > ranks, confidence included.

Hard rules:
❌ Never expose "true NHL ratings"
✅ Work on prospect *signals* / context, with uncertainty + bias.
✅ Deterministic under seeded RNG.

Standard library only. No I/O. No HTTP. No DB calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math
import random
import uuid


# =============================================================================
# Helpers
# =============================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Best-effort attribute/dict accessor for integration with your Prospect class."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def safe_get_nested(obj: Any, path: List[str], default: Any = None) -> Any:
    cur = obj
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            cur = getattr(cur, p, None)
    return default if cur is None else cur


def stable_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def normal(rng: random.Random, mean: float, std: float) -> float:
    # Python's gauss is deterministic given a seeded rng.
    return rng.gauss(mean, std)


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# Enums
# =============================================================================

class ScoutRole(str, Enum):
    AREA = "area_scout"
    CROSSCHECK = "crosschecker"
    HEAD = "head_scout"
    DIRECTOR = "amateur_director"


class Region(str, Enum):
    OHL = "OHL"
    QMJHL = "QMJHL"
    WHL = "WHL"
    USHL = "USHL"
    NCAA = "NCAA"
    EUROPE = "EUROPE"
    SWEDEN = "SWEDEN"
    FINLAND = "FINLAND"
    CZECH = "CZECH"
    SLOVAKIA = "SLOVAKIA"
    RUSSIA = "RUSSIA"
    OTHER = "OTHER"


class Position(str, Enum):
    F = "F"
    C = "C"
    LW = "LW"
    RW = "RW"
    D = "D"
    G = "G"


class Recommendation(str, Enum):
    DRAFT_NOW = "draft_now"
    MONITOR = "monitor"
    AVOID = "avoid"
    TRADE_DOWN_TARGET = "trade_down_target"
    HIDDEN_GEM = "hidden_gem"
    OVERRATED = "overrated"


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ScoutBiasProfile:
    """
    Bias values around 0.5 (neutral). Above 0.5 pushes *up* weighting/grades for that dimension.
    """
    size: float = 0.5
    skill: float = 0.5
    safety: float = 0.5
    role_fit: float = 0.5
    physical: float = 0.5
    iq: float = 0.5
    skating: float = 0.5
    character: float = 0.5


@dataclass
class ScoutSkillProfile:
    """
    Skill values 0..1
    """
    eval_tools: float = 0.55          # skating, shot, hands, physical tools
    projection: float = 0.55          # ceiling/floor projection
    character: float = 0.50           # coachability/leadership/read
    goalies: float = 0.45             # goalie evaluation specialization
    defense_read: float = 0.50        # defensive IQ recognition
    skating_eye: float = 0.55         # skating evaluation accuracy


@dataclass
class Scout:
    id: str
    name: str
    team_id: str
    role: ScoutRole = ScoutRole.AREA
    region: Region = Region.OTHER

    skills: ScoutSkillProfile = field(default_factory=ScoutSkillProfile)
    bias: ScoutBiasProfile = field(default_factory=ScoutBiasProfile)

    work_ethic: float = 0.55          # impacts viewings volume
    patience: float = 0.55            # resists overreacting
    confidence: float = 0.55          # influences how strongly reports are pushed (aggregation weight)
    networking: float = 0.40          # affects rumor/character/injury intel
    risk_tolerance: float = 0.50      # loves boom/bust or prefers safe
    reputation: float = 0.50          # influences org trust

    volatility: float = 0.45          # overreaction tendency
    notes_style_harshness: float = 0.45  # narrative tone: blunt vs rosy

    def weight_multiplier(self) -> float:
        role_mult = {
            ScoutRole.AREA: 1.0,
            ScoutRole.CROSSCHECK: 1.15,
            ScoutRole.HEAD: 1.25,
            ScoutRole.DIRECTOR: 1.30,
        }.get(self.role, 1.0)
        return role_mult * lerp(0.85, 1.20, self.reputation)


@dataclass
class LeagueContextSnapshot:
    """
    Minimal context for scouting; you can pass your full league context in and extract what you want.
    """
    season: int = 0
    week: int = 0
    active_era: str = "speed_and_skill"
    league_health: float = 0.5


@dataclass
class ViewingEvent:
    """
    A discrete "scout saw player at X event" record.
    """
    scout_id: str
    prospect_id: str
    week: int
    event_name: str = "regular_season"
    importance: float = 0.5    # 0..1
    games_seen: int = 1


@dataclass
class TraitObservation:
    """
    Observed trait estimate with uncertainty. Value in 0..1, uncertainty in 0..1.
    """
    value: float
    uncertainty: float
    visibility: float


@dataclass
class ScoutingReport:
    """
    Snapshot of a scout's belief at a moment in time.
    """
    prospect_id: str
    scout_id: str
    week: int
    viewings: int
    event_history: List[ViewingEvent] = field(default_factory=list)

    observed: Dict[str, TraitObservation] = field(default_factory=dict)

    confidence: float = 0.0
    grade: float = 0.0  # 0..1 internal numeric grade
    tier: int = 5       # 1..5
    floor_est: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # low, mid, high
    ceiling_est: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    nhl_probs: Dict[str, float] = field(default_factory=dict)  # role probabilities

    risk_flags: List[str] = field(default_factory=list)
    recommendation: Recommendation = Recommendation.MONITOR
    narrative_notes: List[str] = field(default_factory=list)

    stale_factor: float = 1.0  # 1 fresh -> 0 stale

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # enums already serialize as strings because they inherit str Enum, but keep safe:
        d["recommendation"] = str(self.recommendation.value)
        return d


@dataclass
class OrgBiasProfile:
    """
    Team-level bias / philosophy.
    Values around 0.5 (neutral).
    """
    size: float = 0.5
    skill: float = 0.5
    safety: float = 0.5
    skating: float = 0.5
    iq: float = 0.5
    physical: float = 0.5
    character: float = 0.5
    goalies: float = 0.5

    # Philosophy toggles
    best_player_available: float = 0.65  # 0..1
    needs_weight: float = 0.35           # 0..1
    upside_weight: float = 0.55          # 0..1
    floor_weight: float = 0.45           # 0..1


@dataclass
class TeamProspectView:
    """
    Aggregated team belief about a prospect at a given week.
    """
    team_id: str
    prospect_id: str
    week: int

    grade: float
    tier: int
    confidence: float
    floor_est: Tuple[float, float, float]
    ceiling_est: Tuple[float, float, float]
    nhl_probs: Dict[str, float]
    risk_flags: List[str]
    recommendation: Recommendation
    narrative_notes: List[str]

    # Diagnostics
    contributing_reports: List[str] = field(default_factory=list)  # scout_ids
    disagreement: float = 0.0  # 0..1 measure

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recommendation"] = str(self.recommendation.value)
        return d


@dataclass
class DraftBoardView:
    team_id: str
    week: int
    tiers: Dict[int, List[str]]  # tier -> prospect_ids
    ranked: List[str]            # overall ranking (by grade then confidence)
    notes: List[str] = field(default_factory=list)


@dataclass
class TeamScoutingDepartment:
    team_id: str
    scouts: List[Scout] = field(default_factory=list)

    budget_level: float = 0.5        # affects coverage and viewings
    coverage_quality: float = 0.5    # affects which prospects you can observe reliably
    org_bias: OrgBiasProfile = field(default_factory=OrgBiasProfile)

    # report cache: prospect_id -> scout_id -> report
    reports: Dict[str, Dict[str, ScoutingReport]] = field(default_factory=dict)

    # aggregated cache: prospect_id -> TeamProspectView
    team_views: Dict[str, TeamProspectView] = field(default_factory=dict)

    # coverage map: region -> emphasis (0..1)
    region_focus: Dict[str, float] = field(default_factory=dict)

    # optional: team needs vector (positional needs etc.)
    needs: Dict[str, float] = field(default_factory=dict)  # e.g. {"D":0.7,"G":0.3,"C":0.6}

    rng_seed: int = 0  # for deterministic internal randomness


# =============================================================================
# Trait model configuration
# =============================================================================

# Trait visibility: how easy it is to observe (0..1)
TRAIT_VISIBILITY: Dict[str, float] = {
    "skating": 0.90,
    "hands": 0.70,
    "shot": 0.70,
    "passing": 0.65,
    "offense_iq": 0.55,
    "defense_iq": 0.45,
    "physical": 0.75,
    "size": 0.80,
    "compete": 0.50,
    "consistency": 0.40,
    "leadership": 0.25,
    "coachability": 0.25,
    "risk_injury": 0.30,     # rumor + history
    "volatility": 0.35,      # pattern recognition
    "goalie_tracking": 0.55,
    "goalie_rebound": 0.50,
    "goalie_composure": 0.40,
}

# Which scout skill influences each trait's accuracy
TRAIT_SKILL_MAP: Dict[str, str] = {
    "skating": "skating_eye",
    "hands": "eval_tools",
    "shot": "eval_tools",
    "passing": "eval_tools",
    "offense_iq": "projection",
    "defense_iq": "defense_read",
    "physical": "eval_tools",
    "size": "eval_tools",
    "compete": "character",
    "consistency": "projection",
    "leadership": "character",
    "coachability": "character",
    "risk_injury": "character",
    "volatility": "character",
    "goalie_tracking": "goalies",
    "goalie_rebound": "goalies",
    "goalie_composure": "goalies",
}

# Trait categories to bias against
TRAIT_BIAS_CATEGORY: Dict[str, str] = {
    "skating": "skating",
    "hands": "skill",
    "shot": "skill",
    "passing": "skill",
    "offense_iq": "iq",
    "defense_iq": "iq",
    "physical": "physical",
    "size": "size",
    "compete": "character",
    "consistency": "safety",
    "leadership": "character",
    "coachability": "character",
    "risk_injury": "safety",
    "volatility": "safety",
    "goalie_tracking": "goalies",
    "goalie_rebound": "goalies",
    "goalie_composure": "goalies",
}

# Default weight of traits in grade calculation (rough, can be tuned later)
TRAIT_GRADE_WEIGHTS: Dict[str, float] = {
    "skating": 0.12,
    "hands": 0.10,
    "shot": 0.08,
    "passing": 0.08,
    "offense_iq": 0.10,
    "defense_iq": 0.10,
    "physical": 0.08,
    "size": 0.05,
    "compete": 0.10,
    "consistency": 0.08,
    "leadership": 0.05,
    "coachability": 0.06,
    "risk_injury": -0.05,     # penalty
    "volatility": -0.05,      # penalty
}

GOALIE_EXTRA_WEIGHTS: Dict[str, float] = {
    "goalie_tracking": 0.14,
    "goalie_rebound": 0.10,
    "goalie_composure": 0.10,
}


# =============================================================================
# Factory / setup
# =============================================================================

def create_scout(
    team_id: str,
    region: Region,
    role: ScoutRole = ScoutRole.AREA,
    archetype: Optional[str] = None,
    rng: Optional[random.Random] = None,
    name: Optional[str] = None,
) -> Scout:
    rng = rng or random.Random()

    # Base randomness (bounded)
    def r(mu: float, sigma: float = 0.08) -> float:
        return clamp(normal(rng, mu, sigma), 0.05, 0.95)

    skills = ScoutSkillProfile(
        eval_tools=r(0.55),
        projection=r(0.55),
        character=r(0.50),
        goalies=r(0.45),
        defense_read=r(0.50),
        skating_eye=r(0.55),
    )

    bias = ScoutBiasProfile(
        size=r(0.50),
        skill=r(0.50),
        safety=r(0.50),
        role_fit=r(0.50),
        physical=r(0.50),
        iq=r(0.50),
        skating=r(0.50),
        character=r(0.50),
    )

    work_ethic = r(0.55)
    patience = r(0.55)
    confidence = r(0.55)
    networking = r(0.40)
    risk_tolerance = r(0.50)
    reputation = r(0.50)
    volatility = r(0.45)
    harshness = r(0.45)

    # Archetype shaping
    if archetype:
        a = archetype.lower().strip()
        if a in ("tools_guy", "tool_guy"):
            skills.eval_tools = clamp(skills.eval_tools + 0.15, 0, 1)
            bias.size = clamp(bias.size + 0.12, 0, 1)
            bias.physical = clamp(bias.physical + 0.10, 0, 1)
            bias.skill = clamp(bias.skill - 0.05, 0, 1)
        elif a in ("brain_guy", "iq_guy"):
            skills.projection = clamp(skills.projection + 0.15, 0, 1)
            skills.defense_read = clamp(skills.defense_read + 0.12, 0, 1)
            bias.iq = clamp(bias.iq + 0.12, 0, 1)
            bias.safety = clamp(bias.safety + 0.05, 0, 1)
        elif a in ("character_hawk", "character"):
            skills.character = clamp(skills.character + 0.20, 0, 1)
            bias.character = clamp(bias.character + 0.15, 0, 1)
            networking = clamp(networking + 0.15, 0, 1)
            harshness = clamp(harshness + 0.10, 0, 1)
        elif a in ("swing_for_fences", "upside"):
            risk_tolerance = clamp(risk_tolerance + 0.20, 0, 1)
            bias.skill = clamp(bias.skill + 0.10, 0, 1)
            bias.safety = clamp(bias.safety - 0.10, 0, 1)
            volatility = clamp(volatility + 0.15, 0, 1)
        elif a in ("safe_picker", "floor"):
            bias.safety = clamp(bias.safety + 0.20, 0, 1)
            risk_tolerance = clamp(risk_tolerance - 0.15, 0, 1)
            patience = clamp(patience + 0.10, 0, 1)
            volatility = clamp(volatility - 0.10, 0, 1)
        elif a in ("goalie_specialist", "goalies"):
            skills.goalies = clamp(skills.goalies + 0.25, 0, 1)
            skills.projection = clamp(skills.projection + 0.05, 0, 1)

    # Role shaping
    if role in (ScoutRole.HEAD, ScoutRole.DIRECTOR):
        reputation = clamp(reputation + 0.10, 0, 1)
        confidence = clamp(confidence + 0.10, 0, 1)
    if role == ScoutRole.CROSSCHECK:
        skills.projection = clamp(skills.projection + 0.05, 0, 1)
        patience = clamp(patience + 0.05, 0, 1)

    return Scout(
        id=stable_id("scout"),
        name=name or f"Scout_{rng.randint(100, 999)}",
        team_id=team_id,
        role=role,
        region=region,
        skills=skills,
        bias=bias,
        work_ethic=work_ethic,
        patience=patience,
        confidence=confidence,
        networking=networking,
        risk_tolerance=risk_tolerance,
        reputation=reputation,
        volatility=volatility,
        notes_style_harshness=harshness,
    )


def create_scouting_department(
    team_id: str,
    budget_level: float = 0.55,
    coverage_quality: float = 0.55,
    org_bias: Optional[OrgBiasProfile] = None,
    rng_seed: int = 0,
    scouts: Optional[List[Scout]] = None,
    region_focus: Optional[Dict[str, float]] = None,
    needs: Optional[Dict[str, float]] = None,
) -> TeamScoutingDepartment:
    dept = TeamScoutingDepartment(
        team_id=team_id,
        scouts=scouts or [],
        budget_level=clamp(budget_level),
        coverage_quality=clamp(coverage_quality),
        org_bias=org_bias or OrgBiasProfile(),
        region_focus=region_focus or {},
        needs=needs or {},
        rng_seed=rng_seed,
    )
    return dept


# =============================================================================
# Core observation model
# =============================================================================

def trait_visibility(trait: str, prospect: Any, event_importance: float) -> float:
    """
    Visibility depends on base trait visibility + event importance.
    You can extend with league strength or prospect role.
    """
    base = TRAIT_VISIBILITY.get(trait, 0.45)
    # big events reveal more, but not perfectly:
    return clamp(base + 0.25 * (event_importance - 0.5), 0.05, 0.98)


def scout_trait_skill(scout: Scout, trait: str) -> float:
    skill_name = TRAIT_SKILL_MAP.get(trait, "eval_tools")
    return clamp(getattr(scout.skills, skill_name, 0.5))


def apply_bias_shift(value: float, bias_strength: float, amount: float) -> float:
    """
    Push value up or down around 0.5 by 'amount', scaled by bias strength.
    bias_strength ~ 0.5 neutral; >0.5 pushes upward preference when trait aligns.
    """
    # Convert bias_strength (0..1) into signed push factor around neutral:
    # 0.5 => 0; 1.0 => +1; 0.0 => -1
    push = (bias_strength - 0.5) * 2.0
    return clamp(value + push * amount, 0.0, 1.0)


def scout_bias_for_trait(scout: Scout, dept: TeamScoutingDepartment, trait: str) -> float:
    """
    Blend scout bias with org bias (org is weaker than scout at the report level,
    but present).
    """
    cat = TRAIT_BIAS_CATEGORY.get(trait, "skill")

    scout_val = getattr(scout.bias, cat, 0.5)
    org_val = getattr(dept.org_bias, cat, 0.5)

    # scout dominates in their report; org adds gentle anchor
    return clamp(lerp(scout_val, org_val, 0.25))


def observation_uncertainty(
    viewings: int,
    vis: float,
    scout_skill: float,
    event_noise: float,
) -> float:
    """
    Uncertainty decreases with viewings and scout skill, increases with low visibility and event noise.
    """
    v = max(1, viewings)
    # baseline uncertainty depends on visibility and scout skill
    base = lerp(0.55, 0.20, vis)           # higher vis => lower base
    base = lerp(base, base * 0.70, scout_skill)  # better scout => less uncertainty
    # viewings reduce uncertainty with diminishing returns
    reduced = base / math.sqrt(v)
    # events (tournaments) add volatility / small sample bias
    return clamp(reduced + event_noise * 0.10, 0.02, 0.90)


def observe_trait(
    rng: random.Random,
    scout: Scout,
    dept: TeamScoutingDepartment,
    prospect: Any,
    trait: str,
    viewings: int,
    event_importance: float,
    event_name: str,
) -> TraitObservation:
    """
    Produce a noisy, biased estimate of a prospect trait (0..1).
    """
    # prospect truth signals (hidden)
    signals = safe_get(prospect, "signals", None)
    true_val = None
    if isinstance(signals, dict) and trait in signals:
        true_val = float(signals[trait])
    else:
        # fallback for loose integration:
        true_val = safe_get(prospect, trait, 0.5)
        true_val = float(true_val) if true_val is not None else 0.5

    true_val = clamp(true_val)

    vis = trait_visibility(trait, prospect, event_importance)
    skill = scout_trait_skill(scout, trait)

    # Tournament-like events: more visibility but also more overreaction noise
    is_tourney = ("wjc" in event_name.lower()) or ("u18" in event_name.lower()) or ("hlinka" in event_name.lower())
    event_noise = 0.0
    if is_tourney:
        # volatile scouts overreact more; patient scouts resist
        overreact = clamp(0.25 + 0.45 * scout.volatility - 0.35 * scout.patience)
        event_noise = overreact * (0.6 + 0.8 * (event_importance - 0.5))

    uncert = observation_uncertainty(viewings=viewings, vis=vis, scout_skill=skill, event_noise=event_noise)

    # noise around truth; scale by uncertainty
    noise = normal(rng, 0.0, uncert * 0.65)

    # bias shift: scouts/org push interpretation
    bias_strength = scout_bias_for_trait(scout, dept, trait)
    # bias amount: how much bias can move this trait estimate (trait-dependent)
    bias_amount = lerp(0.01, 0.06, 1.0 - vis)  # low visibility -> more bias sway
    biased = apply_bias_shift(true_val, bias_strength, bias_amount)

    # event overreaction: small-sample "saw him pop" effect (or dud)
    if is_tourney and event_noise > 0:
        swing = normal(rng, 0.0, event_noise * 0.20)
        biased = clamp(biased + swing)

    observed = clamp(biased + noise)

    return TraitObservation(value=observed, uncertainty=uncert, visibility=vis)


# =============================================================================
# Report generation
# =============================================================================

def _infer_position(prospect: Any) -> Position:
    pos = safe_get(prospect, "position", None) or safe_get(prospect, "pos", None) or "F"
    try:
        return Position(str(pos))
    except Exception:
        # normalize common variants
        s = str(pos).upper()
        if s in ("D", "DEF", "DEFENSE"):
            return Position.D
        if s in ("G", "GK", "GOALIE"):
            return Position.G
        if s == "C":
            return Position.C
        if s == "LW":
            return Position.LW
        if s == "RW":
            return Position.RW
        return Position.F


def _prospect_region(prospect: Any) -> Region:
    r = safe_get(prospect, "region", None) or safe_get(prospect, "league_region", None) or "OTHER"
    try:
        return Region(str(r))
    except Exception:
        return Region.OTHER


def _base_viewings_for_scout(
    scout: Scout,
    dept: TeamScoutingDepartment,
    prospect: Any,
    week: int,
) -> int:
    """
    Determine how many viewings this scout gets this update cycle.
    Higher budget + work_ethic => more.
    Better coverage => less blind spots.
    Region focus => more viewings in covered region.
    """
    base = 1.0 + 2.0 * dept.budget_level + 1.0 * dept.coverage_quality
    base *= lerp(0.70, 1.35, scout.work_ethic)

    # region focus multiplier
    p_region = _prospect_region(prospect).value
    focus = dept.region_focus.get(p_region, 0.5)
    base *= lerp(0.75, 1.25, focus)

    # if scout is in same region, boost; otherwise reduce (unless crosschecker/head)
    same_region = (_prospect_region(prospect) == scout.region)
    if scout.role == ScoutRole.AREA:
        base *= 1.20 if same_region else 0.70
    elif scout.role == ScoutRole.CROSSCHECK:
        base *= 1.10
    else:
        base *= 1.05

    # clamp to int viewings per "tick"
    viewings = int(max(1, round(base)))
    return min(viewings, 10)


def simulate_viewing_event(
    rng: random.Random,
    scout: Scout,
    dept: TeamScoutingDepartment,
    prospect: Any,
    week: int,
) -> ViewingEvent:
    """
    Decide if a viewing is regular season vs special event. You can pass explicit events instead.
    """
    # Prospect context can include flags like "in_wjc", "in_u18", etc.
    flags = safe_get(prospect, "event_flags", None) or {}
    in_wjc = bool(flags.get("WJC", False)) if isinstance(flags, dict) else False
    in_u18 = bool(flags.get("U18", False)) if isinstance(flags, dict) else False

    # small chance of tournament view even if not flagged (public hype / showcases)
    tourney_roll = rng.random()
    if in_wjc or (tourney_roll < 0.03 and dept.budget_level > 0.5):
        return ViewingEvent(
            scout_id=scout.id,
            prospect_id=safe_get(prospect, "id", "unknown"),
            week=week,
            event_name="WJC",
            importance=0.90,
            games_seen=1 + (1 if rng.random() < 0.35 else 0),
        )
    if in_u18 or (tourney_roll < 0.025 and dept.budget_level > 0.55):
        return ViewingEvent(
            scout_id=scout.id,
            prospect_id=safe_get(prospect, "id", "unknown"),
            week=week,
            event_name="U18",
            importance=0.85,
            games_seen=1,
        )

    # regular season default
    return ViewingEvent(
        scout_id=scout.id,
        prospect_id=safe_get(prospect, "id", "unknown"),
        week=week,
        event_name="regular_season",
        importance=0.55,
        games_seen=1,
    )


def compute_report_confidence(observed: Dict[str, TraitObservation], viewings: int) -> float:
    if not observed:
        return 0.0
    # Lower uncertainty and more viewings -> higher confidence.
    avg_uncert = sum(o.uncertainty for o in observed.values()) / max(1, len(observed))
    base = lerp(0.15, 0.85, 1.0 - avg_uncert)
    view_bonus = lerp(0.00, 0.12, clamp(math.log(viewings + 1, 6)))  # small diminishing returns
    return clamp(base + view_bonus, 0.05, 0.95)


def compute_grade(
    scout: Scout,
    dept: TeamScoutingDepartment,
    prospect: Any,
    observed: Dict[str, TraitObservation],
) -> float:
    """
    Internal numeric grade (0..1) derived from observed traits.
    Applies scout + org philosophy slightly.
    """
    pos = _infer_position(prospect)

    weights = dict(TRAIT_GRADE_WEIGHTS)
    if pos == Position.G:
        weights.update(GOALIE_EXTRA_WEIGHTS)

    total_w = 0.0
    score = 0.0

    for trait, w in weights.items():
        obs = observed.get(trait)
        if obs is None:
            continue
        total_w += abs(w)
        # penalize uncertain reads: we trust low-uncertainty more
        trust = clamp(1.0 - obs.uncertainty, 0.15, 1.0)
        value = obs.value

        # Org and scout biases subtly affect final grade weighting
        cat = TRAIT_BIAS_CATEGORY.get(trait, "skill")
        bias_scout = getattr(scout.bias, cat, 0.5)
        bias_org = getattr(dept.org_bias, cat, 0.5)
        bias = lerp(bias_scout, bias_org, 0.35)

        # bias affects how much we "care" about this trait
        w_eff = w * lerp(0.85, 1.15, bias)

        score += w_eff * value * trust

    if total_w <= 0:
        return 0.5

    # Normalize score roughly to 0..1
    raw = score / total_w

    # Upside vs safety preference: incorporate observed volatility / risk
    risk_injury = observed.get("risk_injury", TraitObservation(0.45, 0.8, 0.3)).value
    vol = observed.get("volatility", TraitObservation(0.45, 0.8, 0.3)).value
    consistency = observed.get("consistency", TraitObservation(0.50, 0.8, 0.4)).value

    upside = clamp((observed.get("hands", TraitObservation(0.5, 0.8, 0.7)).value +
                    observed.get("skating", TraitObservation(0.5, 0.8, 0.9)).value +
                    observed.get("offense_iq", TraitObservation(0.5, 0.8, 0.5)).value) / 3.0)

    floor = clamp((consistency +
                   observed.get("defense_iq", TraitObservation(0.5, 0.8, 0.45)).value +
                   observed.get("coachability", TraitObservation(0.5, 0.8, 0.25)).value) / 3.0)

    # scout preference blend
    pref_up = lerp(0.45, 0.65, scout.risk_tolerance)
    pref_safe = lerp(0.65, 0.45, scout.risk_tolerance)

    # org preference blend
    org_up = dept.org_bias.upside_weight
    org_floor = dept.org_bias.floor_weight

    up_w = lerp(pref_up, org_up, 0.45)
    floor_w = lerp(pref_safe, org_floor, 0.45)

    # penalize risk (injury/volatility)
    risk_pen = 0.12 * risk_injury + 0.10 * vol

    adj = raw
    adj = clamp(adj + (upside - 0.5) * (up_w - 0.5) * 0.30)
    adj = clamp(adj + (floor - 0.5) * (floor_w - 0.5) * 0.30)
    adj = clamp(adj - risk_pen)

    return clamp(adj, 0.02, 0.98)


def grade_to_tier(grade: float) -> int:
    # Tunable cutoffs; tiers are more important than exact rank.
    if grade >= 0.84:
        return 1
    if grade >= 0.72:
        return 2
    if grade >= 0.60:
        return 3
    if grade >= 0.48:
        return 4
    return 5


def estimate_floor_ceiling(
    rng: random.Random,
    scout: Scout,
    confidence: float,
    grade: float,
    observed: Dict[str, TraitObservation],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Produce [low, mid, high] for floor and ceiling as perceived by the scout.
    Wider ranges when confidence is low.
    """
    # uncertainty width: low confidence => wide ranges
    width = lerp(0.22, 0.06, confidence)

    # ceiling is driven by upside traits
    upside = clamp((observed.get("hands", TraitObservation(0.5, 0.8, 0.7)).value +
                    observed.get("skating", TraitObservation(0.5, 0.8, 0.9)).value +
                    observed.get("shot", TraitObservation(0.5, 0.8, 0.7)).value +
                    observed.get("offense_iq", TraitObservation(0.5, 0.8, 0.5)).value) / 4.0)

    # floor is driven by stability traits
    floor_core = clamp((observed.get("consistency", TraitObservation(0.5, 0.8, 0.4)).value +
                        observed.get("defense_iq", TraitObservation(0.5, 0.8, 0.45)).value +
                        observed.get("coachability", TraitObservation(0.5, 0.8, 0.25)).value +
                        observed.get("compete", TraitObservation(0.5, 0.8, 0.5)).value) / 4.0)

    # scout projection skill reduces error
    proj_err = lerp(0.08, 0.03, scout.skills.projection)

    # midpoints
    floor_mid = clamp(lerp(grade, floor_core, 0.55) + normal(rng, 0.0, proj_err))
    ceil_mid = clamp(lerp(grade, upside, 0.60) + normal(rng, 0.0, proj_err))

    floor_low = clamp(floor_mid - width * 1.10)
    floor_high = clamp(floor_mid + width * 0.90)

    ceil_low = clamp(ceil_mid - width * 0.70)
    ceil_high = clamp(ceil_mid + width * 1.20)

    return (floor_low, floor_mid, floor_high), (ceil_low, ceil_mid, ceil_high)


def estimate_nhl_probabilities(
    grade: float,
    confidence: float,
    pos: Position,
    observed: Dict[str, TraitObservation],
) -> Dict[str, float]:
    """
    Coarse role probabilities. Your sim can later map these to actual NHL outcomes.
    """
    risk_injury = observed.get("risk_injury", TraitObservation(0.45, 0.8, 0.3)).value
    vol = observed.get("volatility", TraitObservation(0.45, 0.8, 0.3)).value
    consistency = observed.get("consistency", TraitObservation(0.50, 0.8, 0.4)).value

    # Base probability of "NHL regular" shaped by grade and confidence
    base = logistic((grade - 0.55) * 6.0)  # 0.55 is ~ bubble
    base = clamp(base * lerp(0.85, 1.10, confidence))
    base = clamp(base - 0.10 * risk_injury - 0.08 * vol + 0.06 * (consistency - 0.5))

    # Role splits
    p_star = clamp(logistic((grade - 0.78) * 9.0) * lerp(0.70, 1.20, confidence), 0, 0.95)
    p_top = clamp(logistic((grade - 0.70) * 8.0) * lerp(0.75, 1.15, confidence), 0, 0.95)
    p_regular = clamp(base, 0.0, 0.98)

    # Bust probability increases with low grade and low confidence
    p_bust = clamp(logistic((0.50 - grade) * 7.0) * lerp(0.80, 1.30, 1.0 - confidence), 0.02, 0.98)

    # Normalize-ish and make position-specific labels
    probs: Dict[str, float] = {}
    if pos == Position.D:
        probs["p_top_pair"] = clamp(p_star * 0.85)
        probs["p_top4"] = clamp(p_top)
        probs["p_nhl_regular"] = clamp(p_regular)
        probs["p_depth"] = clamp(p_regular * (1.0 - probs["p_top4"]) * 0.75)
    elif pos == Position.G:
        # goalies: higher variance, lower confidence conversion
        probs["p_starter"] = clamp(p_star * 0.70)
        probs["p_1b"] = clamp(p_top * 0.80)
        probs["p_nhl_regular"] = clamp(p_regular * 0.80)
        probs["p_backup"] = clamp(p_regular * 0.70)
    else:
        probs["p_top_line"] = clamp(p_star)
        probs["p_top6"] = clamp(p_top)
        probs["p_nhl_regular"] = clamp(p_regular)
        probs["p_bottom6"] = clamp(p_regular * (1.0 - probs["p_top6"]) * 0.75)

    probs["p_bust"] = p_bust
    return probs


def build_risk_flags(
    scout: Scout,
    dept: TeamScoutingDepartment,
    prospect: Any,
    observed: Dict[str, TraitObservation],
    confidence: float,
) -> List[str]:
    flags: List[str] = []

    risk_injury = observed.get("risk_injury")
    if risk_injury and risk_injury.value > 0.62 and (1.0 - risk_injury.uncertainty) > 0.20:
        flags.append("injury_concern")

    vol = observed.get("volatility")
    if vol and vol.value > 0.62 and (1.0 - vol.uncertainty) > 0.20:
        flags.append("boom_bust")

    coachability = observed.get("coachability")
    if coachability and coachability.value < 0.38 and (1.0 - coachability.uncertainty) > 0.15:
        flags.append("coachability_concern")

    consistency = observed.get("consistency")
    if consistency and consistency.value < 0.38 and (1.0 - consistency.uncertainty) > 0.15:
        flags.append("inconsistent")

    compete = observed.get("compete")
    if compete and compete.value < 0.35 and (1.0 - compete.uncertainty) > 0.15:
        flags.append("low_compete")

    # Size flag (org-specific)
    size = observed.get("size")
    if size and size.value < 0.35:
        # only matters if org or scout is size-biased
        if (dept.org_bias.size > 0.58) or (scout.bias.size > 0.58):
            flags.append("undersized")

    # Optional: "overager" context
    age = safe_get(prospect, "age", None)
    draft_age = safe_get(prospect, "draft_age", 18)
    if age is not None:
        try:
            if float(age) - float(draft_age) >= 1.2:
                flags.append("overager_discount")
        except Exception:
            pass

    # Low confidence meta-flag
    if confidence < 0.40:
        flags.append("limited_viewings")

    return flags


def recommend(
    rng: random.Random,
    scout: Scout,
    dept: TeamScoutingDepartment,
    prospect: Any,
    grade: float,
    tier: int,
    confidence: float,
    risk_flags: List[str],
    pos: Position,
) -> Recommendation:
    """
    Produce a recommendation, reflecting both quality and org/scout biases.
    """
    # baseline
    rec = Recommendation.MONITOR

    # avoid if major red flags + middling grade
    red = {"injury_concern", "coachability_concern", "low_compete"}
    red_hits = len(red.intersection(set(risk_flags)))

    if tier <= 2 and confidence >= 0.55 and red_hits == 0:
        rec = Recommendation.DRAFT_NOW
    elif grade >= 0.66 and confidence >= 0.50 and red_hits == 0:
        rec = Recommendation.DRAFT_NOW
    elif grade < 0.45 and red_hits >= 1 and confidence >= 0.45:
        rec = Recommendation.AVOID
    elif grade < 0.40 and confidence >= 0.45:
        rec = Recommendation.AVOID

    # hidden gem logic: good grade but low confidence (or low public rank if you track it)
    public_rank = safe_get(prospect, "public_rank", None)
    if public_rank is not None:
        try:
            pr = int(public_rank)
            if grade >= 0.62 and pr >= 45 and confidence < 0.55:
                rec = Recommendation.HIDDEN_GEM
        except Exception:
            pass
    else:
        if grade >= 0.62 and confidence < 0.50 and "limited_viewings" in risk_flags:
            rec = Recommendation.HIDDEN_GEM

    # overrated: public rank high but our grade low
    if public_rank is not None:
        try:
            pr = int(public_rank)
            if pr <= 12 and grade < 0.58 and confidence >= 0.50:
                rec = Recommendation.OVERRATED
        except Exception:
            pass

    # trade-down target: mid grade, high confidence, low risk, safe
    if grade >= 0.54 and grade <= 0.64 and confidence >= 0.60 and red_hits == 0:
        safe_pref = lerp(scout.bias.safety, dept.org_bias.safety, 0.45)
        if safe_pref >= 0.55:
            if rng.random() < 0.35:
                rec = Recommendation.TRADE_DOWN_TARGET

    # needs nudge (optional)
    need_key = pos.value
    need = dept.needs.get(need_key, 0.5)
    if need >= 0.70 and grade >= 0.55 and rec == Recommendation.MONITOR and confidence >= 0.55:
        if rng.random() < 0.35:
            rec = Recommendation.DRAFT_NOW

    return rec


# =============================================================================
# Narrative generation
# =============================================================================

def _note_tone(scout: Scout) -> str:
    # style switches
    return "harsh" if scout.notes_style_harshness > 0.58 else "neutral"


def generate_narrative_notes(
    rng: random.Random,
    scout: Scout,
    prospect: Any,
    pos: Position,
    observed: Dict[str, TraitObservation],
    grade: float,
    confidence: float,
    risk_flags: List[str],
) -> List[str]:
    notes: List[str] = []
    tone = _note_tone(scout)

    def hi(tr: str) -> bool:
        o = observed.get(tr)
        return bool(o and o.value >= 0.68)

    def lo(tr: str) -> bool:
        o = observed.get(tr)
        return bool(o and o.value <= 0.36)

    # Identity line
    name = safe_get(prospect, "name", None) or safe_get(prospect, "display_name", None) or safe_get(prospect, "id", "Prospect")
    notes.append(f"{name}: {pos.value} — perceived grade {grade:.2f} (conf {confidence:.2f}).")

    # Skating / pace
    if hi("skating"):
        notes.append("Explosive feet; wins retrievals and escapes pressure with pace.")
    elif lo("skating"):
        notes.append("Skating may be a limiting factor at NHL pace; needs cleaner mechanics and pace management.")

    # Skill / offense
    if hi("hands") and hi("offense_iq"):
        notes.append("High-end skill package with processing — creates offense, manipulates defenders.")
    elif hi("hands") and lo("offense_iq"):
        notes.append("Hands pop, but read/decision speed lags — relies on talent more than anticipation.")
    elif lo("hands") and hi("offense_iq"):
        notes.append("Not flashy, but smart — finds simple plays and keeps possessions alive.")

    # Defense / IQ
    if pos == Position.D:
        if hi("defense_iq"):
            notes.append("Defensive reads are advanced; closes lanes early and keeps rushes outside.")
        elif lo("defense_iq"):
            notes.append("Defensive processing is a concern — can get dragged out of structure and lose inside positioning.")

    # Physicality / size
    if hi("physical"):
        notes.append("Engages physically; competes on puck battles and net-front work.")
    elif lo("physical"):
        notes.append("Doesn't consistently win contact situations; may need strength and edge work.")

    # Character-ish
    if hi("coachability") and hi("compete"):
        notes.append("Strong worker profile: competes and appears receptive to coaching.")
    elif lo("coachability"):
        notes.append("Coachability concerns — may resist structure or struggle with role acceptance.")
    elif lo("compete"):
        notes.append("Compete level comes and goes; needs more consistent urgency away from the puck.")

    # Consistency / volatility
    if "inconsistent" in risk_flags:
        notes.append("Game-to-game variance shows up; evaluation requires continued tracking.")
    if "boom_bust" in risk_flags:
        notes.append("Boom/bust profile — if it hits, impact is real; if not, the floor is shaky.")

    # Injury / availability
    if "injury_concern" in risk_flags:
        notes.append("Availability risk noted; medical/background should be prioritized.")

    # Limited intel
    if "limited_viewings" in risk_flags:
        if tone == "harsh":
            notes.append("Limited viewings: projection risk remains high — do not overcommit without more exposure.")
        else:
            notes.append("Limited viewings: more exposure needed to firm up projection and risk profile.")

    # Closing / comp
    # lightweight comps based on style clusters
    comp = None
    if pos == Position.D:
        if hi("skating") and hi("defense_iq"):
            comp = "a modern puck-moving defender who survives on feet and reads"
        elif hi("physical") and hi("defense_iq"):
            comp = "a steady matchup defender with pro-level detail"
        elif hi("hands") and hi("offense_iq"):
            comp = "an offensive-leaning defender who needs the right usage"
    elif pos == Position.G:
        if hi("goalie_tracking") and hi("goalie_composure"):
            comp = "a calm, tracking-based goalie profile"
        elif hi("goalie_rebound") and lo("goalie_composure"):
            comp = "technically capable, but composure under chaos is a question"
    else:
        if hi("skating") and hi("hands"):
            comp = "a pace+skill winger profile that can create separation"
        elif hi("offense_iq") and hi("passing"):
            comp = "a connector/play-driver who raises linemates"
        elif hi("physical") and hi("compete"):
            comp = "a forechecking/battle-forward profile with middle-six utility"

    if comp:
        notes.append(f"Stylistic comp: {comp}.")

    return notes[:10]  # keep it tight


def generate_report(
    scout: Scout,
    dept: TeamScoutingDepartment,
    prospect: Any,
    league_ctx: LeagueContextSnapshot,
    viewings: int,
    events: List[ViewingEvent],
    rng: random.Random,
) -> ScoutingReport:
    pid = safe_get(prospect, "id", "unknown")
    pos = _infer_position(prospect)

    # Determine traits to focus based on position
    traits = list(TRAIT_GRADE_WEIGHTS.keys())
    if pos == Position.G:
        # emphasize goalie traits
        traits = traits + list(GOALIE_EXTRA_WEIGHTS.keys())

    # Observe
    observed: Dict[str, TraitObservation] = {}
    # Use event importance average for this report snapshot
    if events:
        avg_imp = sum(e.importance for e in events) / len(events)
        event_name = events[-1].event_name
    else:
        avg_imp = 0.55
        event_name = "regular_season"

    for tr in traits:
        observed[tr] = observe_trait(
            rng=rng,
            scout=scout,
            dept=dept,
            prospect=prospect,
            trait=tr,
            viewings=viewings,
            event_importance=avg_imp,
            event_name=event_name,
        )

    conf = compute_report_confidence(observed, viewings)
    grade = compute_grade(scout, dept, prospect, observed)
    tier = grade_to_tier(grade)
    floor_est, ceil_est = estimate_floor_ceiling(rng, scout, conf, grade, observed)
    probs = estimate_nhl_probabilities(grade, conf, pos, observed)
    flags = build_risk_flags(scout, dept, prospect, observed, conf)
    rec = recommend(rng, scout, dept, prospect, grade, tier, conf, flags, pos)
    notes = generate_narrative_notes(rng, scout, prospect, pos, observed, grade, conf, flags)

    return ScoutingReport(
        prospect_id=str(pid),
        scout_id=scout.id,
        week=league_ctx.week,
        viewings=viewings,
        event_history=events[:],
        observed=observed,
        confidence=conf,
        grade=grade,
        tier=tier,
        floor_est=floor_est,
        ceiling_est=ceil_est,
        nhl_probs=probs,
        risk_flags=flags,
        recommendation=rec,
        narrative_notes=notes,
        stale_factor=1.0,
    )


# =============================================================================
# Staleness / decay
# =============================================================================

def stale_factor(current_week: int, last_week: int, half_life_weeks: float = 6.0) -> float:
    """
    Exponential decay: after half_life_weeks, weight becomes ~0.5.
    """
    dt = max(0, current_week - last_week)
    if half_life_weeks <= 0:
        return 1.0
    return clamp(math.exp(-math.log(2.0) * (dt / half_life_weeks)), 0.05, 1.0)


def decay_reports(dept: TeamScoutingDepartment, current_week: int, half_life_weeks: float = 6.0) -> None:
    for pid, scout_reports in dept.reports.items():
        for sid, rep in scout_reports.items():
            rep.stale_factor = stale_factor(current_week, rep.week, half_life_weeks=half_life_weeks)


# =============================================================================
# Department update / aggregation
# =============================================================================

def update_scouting(
    dept: TeamScoutingDepartment,
    prospects: List[Any],
    league_ctx: LeagueContextSnapshot,
    week: Optional[int] = None,
) -> None:
    """
    Simulate the passage of a week/tick: scouts watch prospects and update reports.
    """
    w = league_ctx.week if week is None else week
    league_ctx = LeagueContextSnapshot(
        season=league_ctx.season,
        week=w,
        active_era=league_ctx.active_era,
        league_health=league_ctx.league_health,
    )

    rng = random.Random(dept.rng_seed + w * 9973)

    # Apply staleness decay first
    decay_reports(dept, current_week=w)

    # For each scout, decide which prospects they see and update reports
    for scout in dept.scouts:
        scout_rng = random.Random((dept.rng_seed + w * 10007) ^ hash(scout.id))

        # coverage / selection:
        # - area scouts see more from their region
        # - crosscheck/head see top public ranks + random
        # We'll do a simple heuristic selection without external lists.
        candidate_pool: List[Any] = prospects[:]

        # Soft filter by region for area scouts
        if scout.role == ScoutRole.AREA:
            same_region = [p for p in prospects if _prospect_region(p) == scout.region]
            other_region = [p for p in prospects if _prospect_region(p) != scout.region]
            # area scouts: 75% same region, 25% others (showcases)
            pick_count = int(lerp(6, 16, dept.budget_level) * lerp(0.75, 1.20, scout.work_ethic))
            pick_count = max(4, min(24, pick_count))
            selected = []

            same_n = int(pick_count * 0.75)
            other_n = pick_count - same_n

            scout_rng.shuffle(same_region)
            scout_rng.shuffle(other_region)
            selected.extend(same_region[:same_n])
            selected.extend(other_region[:other_n])
            candidate_pool = selected
        else:
            # crosscheck/head: cover more broadly, but emphasize top public ranks if present
            ranked = sorted(prospects, key=lambda p: safe_get(p, "public_rank", 9999) or 9999)
            top_slice = ranked[:max(20, int(lerp(20, 60, dept.budget_level)))]
            scout_rng.shuffle(top_slice)
            extra = prospects[:]
            scout_rng.shuffle(extra)
            candidate_pool = (top_slice[:25] + extra[:15])

        # For each candidate, generate viewings and update report
        for prospect in candidate_pool:
            # chance the scout actually sees them this week
            see_chance = lerp(0.25, 0.65, dept.coverage_quality) * lerp(0.70, 1.15, scout.work_ethic)
            # if region mismatch for area scout, reduce
            if scout.role == ScoutRole.AREA and _prospect_region(prospect) != scout.region:
                see_chance *= 0.55

            if scout_rng.random() > see_chance:
                continue

            v = _base_viewings_for_scout(scout, dept, prospect, week=w)

            events: List[ViewingEvent] = []
            for _ in range(v):
                events.append(simulate_viewing_event(scout_rng, scout, dept, prospect, week=w))

            rep = generate_report(
                scout=scout,
                dept=dept,
                prospect=prospect,
                league_ctx=league_ctx,
                viewings=v,
                events=events,
                rng=scout_rng,
            )

            pid = rep.prospect_id
            if pid not in dept.reports:
                dept.reports[pid] = {}
            dept.reports[pid][scout.id] = rep

    # After updates, refresh aggregated team views
    for p in prospects:
        pid = str(safe_get(p, "id", "unknown"))
        dept.team_views[pid] = aggregate_team_view(dept, pid, league_ctx.week)


def _report_weight(rep: ScoutingReport, scout: Scout, current_week: int) -> float:
    # Weight by confidence, scout role/reputation, and staleness
    stale = rep.stale_factor
    base = lerp(0.15, 1.00, rep.confidence)
    return base * scout.weight_multiplier() * stale * lerp(0.85, 1.15, scout.confidence)


def _disagreement_metric(grades: List[float], weights: List[float]) -> float:
    if not grades:
        return 0.0
    # weighted variance normalized
    wsum = sum(weights) if weights else 0.0
    if wsum <= 1e-9:
        mean = sum(grades) / len(grades)
        var = sum((g - mean) ** 2 for g in grades) / max(1, len(grades))
    else:
        mean = sum(g * w for g, w in zip(grades, weights)) / wsum
        var = sum(w * (g - mean) ** 2 for g, w in zip(grades, weights)) / wsum
    # normalize by max possible variance ~0.25 (for 0..1 range worst-case)
    return clamp(var / 0.12, 0.0, 1.0)


def _merge_flags(flag_lists: List[List[str]]) -> List[str]:
    # keep flags that appear at least twice, unless it's "limited_viewings"
    counts: Dict[str, int] = {}
    for fl in flag_lists:
        for f in fl:
            counts[f] = counts.get(f, 0) + 1
    out: List[str] = []
    for f, c in counts.items():
        if f == "limited_viewings":
            if c >= 1:
                out.append(f)
        elif c >= 2:
            out.append(f)
    return out


def aggregate_team_view(
    dept: TeamScoutingDepartment,
    prospect_id: str,
    current_week: int,
) -> TeamProspectView:
    scout_reports = dept.reports.get(prospect_id, {})
    if not scout_reports:
        return TeamProspectView(
            team_id=dept.team_id,
            prospect_id=prospect_id,
            week=current_week,
            grade=0.50,
            tier=4,
            confidence=0.15,
            floor_est=(0.35, 0.45, 0.55),
            ceiling_est=(0.40, 0.55, 0.70),
            nhl_probs={"p_nhl_regular": 0.25, "p_bust": 0.55},
            risk_flags=["limited_viewings"],
            recommendation=Recommendation.MONITOR,
            narrative_notes=["No reliable scouting intel yet."],
            contributing_reports=[],
            disagreement=0.0,
        )

    # map scout_id -> Scout
    scout_map = {s.id: s for s in dept.scouts}

    grades: List[float] = []
    weights: List[float] = []
    confs: List[float] = []
    floors: List[Tuple[float, float, float]] = []
    ceilings: List[Tuple[float, float, float]] = []
    probs_list: List[Dict[str, float]] = []
    flags_list: List[List[str]] = []
    notes: List[str] = []

    contributing: List[str] = []

    for sid, rep in scout_reports.items():
        scout = scout_map.get(sid)
        if scout is None:
            continue
        w = _report_weight(rep, scout, current_week=current_week)
        grades.append(rep.grade)
        weights.append(w)
        confs.append(rep.confidence * rep.stale_factor)
        floors.append(rep.floor_est)
        ceilings.append(rep.ceiling_est)
        probs_list.append(rep.nhl_probs)
        flags_list.append(rep.risk_flags)
        contributing.append(sid)

        # Keep a small number of narrative notes, preferring head/crosscheck
        if scout.role in (ScoutRole.HEAD, ScoutRole.DIRECTOR, ScoutRole.CROSSCHECK):
            notes.extend(rep.narrative_notes[:2])

    if not grades:
        # fallback
        return TeamProspectView(
            team_id=dept.team_id,
            prospect_id=prospect_id,
            week=current_week,
            grade=0.50,
            tier=4,
            confidence=0.15,
            floor_est=(0.35, 0.45, 0.55),
            ceiling_est=(0.40, 0.55, 0.70),
            nhl_probs={"p_nhl_regular": 0.25, "p_bust": 0.55},
            risk_flags=["limited_viewings"],
            recommendation=Recommendation.MONITOR,
            narrative_notes=["Scouting data missing scout references."],
            contributing_reports=[],
            disagreement=0.0,
        )

    wsum = sum(weights) if sum(weights) > 1e-9 else 1.0
    team_grade = sum(g * w for g, w in zip(grades, weights)) / wsum

    # Confidence: weighted average of confidence, reduced by disagreement
    team_conf = sum(c * w for c, w in zip(confs, weights)) / wsum
    disagreement = _disagreement_metric(grades, weights)
    team_conf = clamp(team_conf * lerp(1.00, 0.80, disagreement))

    # Floor/Ceiling: weighted average of components
    def wavg_trip(trips: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        a = sum(t[0] * w for t, w in zip(trips, weights)) / wsum
        b = sum(t[1] * w for t, w in zip(trips, weights)) / wsum
        c = sum(t[2] * w for t, w in zip(trips, weights)) / wsum
        return (clamp(a), clamp(b), clamp(c))

    team_floor = wavg_trip(floors)
    team_ceil = wavg_trip(ceilings)

    # Probabilities: average keys
    prob_keys: set[str] = set()
    for p in probs_list:
        prob_keys.update(p.keys())
    team_probs: Dict[str, float] = {}
    for k in prob_keys:
        team_probs[k] = clamp(sum(p.get(k, 0.0) * w for p, w in zip(probs_list, weights)) / wsum)

    # Merge flags
    team_flags = _merge_flags(flags_list)

    # Recommendation logic at team level:
    tier = grade_to_tier(team_grade)

    # If too much disagreement, lean monitor
    if disagreement > 0.55 and team_conf < 0.55:
        team_rec = Recommendation.MONITOR
    else:
        # pick the most common recommendation among scouts (weighted)
        rec_scores: Dict[str, float] = {}
        for sid, rep in scout_reports.items():
            scout = scout_map.get(sid)
            if not scout:
                continue
            w = _report_weight(rep, scout, current_week=current_week)
            k = rep.recommendation.value
            rec_scores[k] = rec_scores.get(k, 0.0) + w
        best = max(rec_scores.items(), key=lambda kv: kv[1])[0] if rec_scores else Recommendation.MONITOR.value
        team_rec = Recommendation(best) if best in Recommendation._value2member_map_ else Recommendation.MONITOR

        # sanity: if team grade is bad, do not draft now
        if team_grade < 0.52 and team_rec == Recommendation.DRAFT_NOW:
            team_rec = Recommendation.MONITOR
        # if serious flags present, avoid
        if ("injury_concern" in team_flags or "low_compete" in team_flags) and team_grade < 0.55:
            team_rec = Recommendation.AVOID

    # Notes: dedupe
    deduped_notes: List[str] = []
    seen = set()
    for n in notes:
        if n not in seen:
            deduped_notes.append(n)
            seen.add(n)
    if not deduped_notes:
        deduped_notes = ["Aggregated view formed from scout reports."]

    return TeamProspectView(
        team_id=dept.team_id,
        prospect_id=prospect_id,
        week=current_week,
        grade=clamp(team_grade),
        tier=tier,
        confidence=team_conf,
        floor_est=team_floor,
        ceiling_est=team_ceil,
        nhl_probs=team_probs,
        risk_flags=team_flags,
        recommendation=team_rec,
        narrative_notes=deduped_notes[:8],
        contributing_reports=contributing,
        disagreement=disagreement,
    )


# =============================================================================
# Draft board construction
# =============================================================================

def build_team_draft_board(
    dept: TeamScoutingDepartment,
    prospect_pool: List[Any],
    current_week: int,
    limit: Optional[int] = None,
) -> DraftBoardView:
    """
    Build a tiered draft board from team views.
    Ranking is by tier then grade then confidence, with optional needs nudges.
    """
    # Ensure team_views exist
    for p in prospect_pool:
        pid = str(safe_get(p, "id", "unknown"))
        if pid not in dept.team_views or dept.team_views[pid].week != current_week:
            dept.team_views[pid] = aggregate_team_view(dept, pid, current_week)

    # Build scoring with mild needs effect
    def needs_nudge(prospect: Any) -> float:
        if not dept.needs:
            return 0.0
        pos = _infer_position(prospect).value
        n = dept.needs.get(pos, 0.5)
        # needs_weight is org-level
        return (n - 0.5) * dept.org_bias.needs_weight * 0.06

    items: List[Tuple[str, float, float, int]] = []
    for p in prospect_pool:
        pid = str(safe_get(p, "id", "unknown"))
        tv = dept.team_views.get(pid)
        if not tv:
            continue
        # base = grade, plus a small confidence bonus, plus needs nudge
        score = tv.grade + 0.05 * (tv.confidence - 0.5) + needs_nudge(p)
        items.append((pid, score, tv.confidence, tv.tier))

    # Sort: tier asc, score desc, confidence desc
    items.sort(key=lambda x: (x[3], -x[1], -x[2], x[0]))

    if limit is not None:
        items = items[: max(0, int(limit))]

    ranked = [pid for pid, _, __, ___ in items]

    tiers: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for pid in ranked:
        t = dept.team_views[pid].tier
        tiers.setdefault(t, []).append(pid)

    notes: List[str] = []
    # Summary notes
    if ranked:
        top = ranked[:5]
        notes.append(f"Top targets (week {current_week}): {', '.join(top)}")
    # highlight high disagreement prospects
    disagreements = sorted(
        ((pid, dept.team_views[pid].disagreement) for pid in ranked),
        key=lambda x: -x[1]
    )
    hot = [pid for pid, d in disagreements[:5] if d >= 0.55]
    if hot:
        notes.append(f"High-disagreement prospects (needs debate): {', '.join(hot)}")

    return DraftBoardView(
        team_id=dept.team_id,
        week=current_week,
        tiers=tiers,
        ranked=ranked,
        notes=notes,
    )


# =============================================================================
# Debug utilities (optional)
# =============================================================================

def debug_top_prospects(dept: TeamScoutingDepartment, n: int = 10) -> List[Tuple[str, float, float, int, float]]:
    """
    Returns list of (prospect_id, grade, confidence, tier, disagreement)
    """
    views = list(dept.team_views.values())
    views.sort(key=lambda v: (v.tier, -v.grade, -v.confidence))
    out = []
    for v in views[:n]:
        out.append((v.prospect_id, v.grade, v.confidence, v.tier, v.disagreement))
    return out


def debug_scout_differences(dept: TeamScoutingDepartment, prospect_id: str) -> List[Tuple[str, float, float, int, float]]:
    """
    Returns per-scout: (scout_id, grade, confidence, tier, stale_factor)
    """
    res = []
    for sid, rep in (dept.reports.get(prospect_id, {}) or {}).items():
        res.append((sid, rep.grade, rep.confidence, rep.tier, rep.stale_factor))
    res.sort(key=lambda x: -x[1])
    return res
