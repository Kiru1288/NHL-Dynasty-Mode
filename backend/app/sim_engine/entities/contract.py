

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math
import random
import time


# ============================================================
# Utilities
# ============================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def sigmoid(x: float) -> float:
    # numerically safe-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)

def softplus(x: float) -> float:
    # log(1+exp(x)) safe-ish
    if x > 30:
        return x
    return math.log1p(math.exp(x))

def weighted_sum(parts: List[Tuple[float, float]]) -> float:
    # (value, weight) pairs -> sum(value*weight) / sum(weights)
    w = sum(abs(p[1]) for p in parts) or 1.0
    return sum(p[0] * p[1] for p in parts) / w

def now_ts() -> float:
    return time.time()

def chance(rng: random.Random, p: float) -> bool:
    return rng.random() < clamp(p, 0.0, 1.0)


# ============================================================
# Enums / Types
# ============================================================

class ClauseType(str, Enum):
    NONE = "none"
    NTC = "ntc"          # full no-trade
    MNTC = "m-ntc"       # modified no-trade list
    NMC = "nmc"          # no-movement (trade + waivers + demotion)

class NegotiationOutcome(str, Enum):
    ACCEPT = "accept"
    WALK = "walk"
    STALL = "stall"
    FORCE_TRADE = "force_trade"
    ONE_YEAR_PROVE_IT = "one_year_prove_it"

class NegotiationPhase(str, Enum):
    OPENING = "opening"
    COUNTERS = "counters"
    PRESSURE = "pressure"
    RESOLUTION = "resolution"

class ContractContextKind(str, Enum):
    UFA = "ufa"
    RFA = "rfa"
    RESIGN = "resign"
    EXTENSION = "extension"


# ============================================================
# Data Models
# ============================================================

@dataclass
class MarketProfile:
    market_size: str = "medium"         # "small" | "medium" | "large"
    media_pressure: float = 0.5
    fan_expectations: float = 0.5
    tax_advantage: float = 0.5
    weather_quality: float = 0.5
    travel_burden: float = 0.5

@dataclass
class OwnershipProfile:
    patience: float = 0.5
    ambition: float = 0.5
    budget_willingness: float = 0.5
    meddling: float = 0.5

@dataclass
class ReputationProfile:
    league_reputation: float = 0.5
    player_reputation: float = 0.5
    management_reputation: float = 0.5
    development_reputation: float = 0.5

@dataclass
class OrgPhilosophy:
    development_quality: float = 0.5
    prospect_patience: float = 0.5
    risk_tolerance: float = 0.5

@dataclass
class TeamDynamicState:
    competitive_score: float = 0.5
    team_morale: float = 0.5
    org_pressure: float = 0.5
    stability: float = 0.5
    ownership_stability: float = 0.7
    arena_security: float = 0.8
    financial_health: float = 0.7

@dataclass
class TeamRosterProxy:
    star_count: int = 0
    core_count: int = 5
    depth_quality: float = 0.5

@dataclass
class TeamProfile:
    team_id: str
    name: str
    archetype: str = "normal"           # "chaotic" | "stable" | "spender" | "cheap" etc.
    status: str = "bubble"              # "rebuild" | "bubble" | "contender" | "dysfunctional" etc.

    market: MarketProfile = field(default_factory=MarketProfile)
    ownership: OwnershipProfile = field(default_factory=OwnershipProfile)
    reputation: ReputationProfile = field(default_factory=ReputationProfile)
    philosophy: OrgPhilosophy = field(default_factory=OrgPhilosophy)
    state: TeamDynamicState = field(default_factory=TeamDynamicState)
    roster: TeamRosterProxy = field(default_factory=TeamRosterProxy)

    # cap context (you can override per season)
    cap_total: float = 88_000_000.0
    cap_space: float = 10_000_000.0
    cap_projection_growth: float = 0.05   # 5% projected growth

    # memory / reputation hooks
    # maps agent_id -> trust score [0,1]
    agent_trust: Dict[str, float] = field(default_factory=dict)
    # maps player_id -> relationship score [-1,1]
    player_relationship: Dict[str, float] = field(default_factory=dict)
    # "bad actor" score [0,1]
    front_office_chaos: float = 0.5

@dataclass
class PlayerPersonality:
    loyalty: float = 0.5
    ambition: float = 0.5
    money_focus: float = 0.5
    family_priority: float = 0.5
    legacy_drive: float = 0.5
    ego: float = 0.5
    patience: float = 0.5
    risk_tolerance: float = 0.5
    stability_need: float = 0.5
    market_comfort: float = 0.5
    media_comfort: float = 0.5

    # extra useful levers (optional)
    work_ethic: float = 0.5
    mental_toughness: float = 0.5
    volatility: float = 0.5

@dataclass
class PlayerCareerState:
    age: int = 19
    ovr: float = 0.5
    position: str = "C"
    shoots: str = "L"

    wear_and_tear: float = 0.0
    chronic_injuries: int = 0

    # team + role satisfaction
    ice_time_satisfaction: float = 0.5
    role_mismatch: float = 0.0

    # career arc pressures
    legacy_pressure: float = 0.0
    identity_instability: float = 0.0
    emotional_fatigue: float = 0.0
    security_anxiety: float = 0.0

    # money / leverage
    ufa_pressure: float = 0.0  # 0-> not UFA, 1 -> UFA now
    offer_respect: float = 0.5 # how respected they feel in current talks
    last_contract_aav: float = 0.0
    last_contract_term: int = 0

@dataclass
class PlayerMemory:
    drafted_by_team_id: Optional[str] = None
    developed_by_team_id: Optional[str] = None

    # team_id -> "relationship memory" [-1,1]
    relationship: Dict[str, float] = field(default_factory=dict)

    # team_id -> count of lowball offers / betrayals
    lowball_count: Dict[str, int] = field(default_factory=dict)
    betrayal_count: Dict[str, int] = field(default_factory=dict)

    # promises made/kept (team_id -> kept ratio [0,1])
    promise_kept_ratio: Dict[str, float] = field(default_factory=dict)

    # negotiation history log (append-only)
    negotiation_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PlayerProfile:
    player_id: str
    name: str
    current_team_id: str
    personality: PlayerPersonality = field(default_factory=PlayerPersonality)
    career: PlayerCareerState = field(default_factory=PlayerCareerState)
    memory: PlayerMemory = field(default_factory=PlayerMemory)


@dataclass
class AgentProfile:
    agent_id: str
    name: str

    aggression: float = 0.5          # pushes harder on AAV/term/clauses
    loyalty_to_player: float = 0.7   # prioritizes player desires vs agent ego
    league_influence: float = 0.5    # ability to create outside offers/pressure
    media_leak_tendency: float = 0.3 # increases pressure events
    risk_tolerance: float = 0.5

    # memory of teams (team_id -> trust [-1,1])
    team_memory: Dict[str, float] = field(default_factory=dict)

    def remember_team(self, team_id: str, delta: float):
        self.team_memory[team_id] = clamp(self.team_memory.get(team_id, 0.0) + delta, -1.0, 1.0)


@dataclass
class ClausePackage:
    clause_type: ClauseType = ClauseType.NONE
    trade_list_size: int = 0               # for M-NTC
    waiver_protection: bool = False
    retention_allowed: bool = True
    buyout_penalty_profile: float = 0.5    # abstract [0,1] severity; higher = worse for team

    def security_power(self) -> float:
        if self.clause_type == ClauseType.NMC:
            return 1.0
        if self.clause_type == ClauseType.NTC:
            return 0.8
        if self.clause_type == ClauseType.MNTC:
            # smaller list = more protection
            if self.trade_list_size <= 0:
                return 0.6
            return clamp(1.0 - (self.trade_list_size / 32.0), 0.2, 0.7)
        return 0.0

    def flexibility_cost(self) -> float:
        # cost to team flexibility
        base = self.security_power()
        if self.waiver_protection:
            base += 0.1
        base += 0.2 * self.buyout_penalty_profile
        return clamp(base, 0.0, 1.2)

@dataclass
class Contract:
    # Core Fields
    salary_aav: float
    term_years: int
    total_value: float

    signing_bonus: float = 0.0
    performance_bonuses: float = 0.0

    front_loaded_ratio: float = 0.5
    back_loaded_ratio: float = 0.5
    two_way: bool = False
    expiry_age: int = 0

    # Clauses
    clauses: ClausePackage = field(default_factory=ClausePackage)

    # Contextual metadata / memory hooks
    signed_team_id: str = ""
    signed_market_type: str = "medium"
    signing_year: int = 0
    cap_percent_at_signing: float = 0.0
    league_cap_context: Dict[str, Any] = field(default_factory=dict)
    player_leverage_score: float = 0.0

    # Negotiation history hooks
    negotiated_by_agent_id: Optional[str] = None
    negotiation_length_days: int = 0
    promises: Dict[str, Any] = field(default_factory=dict)
    declined_offers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["clauses"]["clause_type"] = self.clauses.clause_type.value
        return d


# ============================================================
# Contract Offer (during negotiation)
# ============================================================

@dataclass
class ContractOffer:
    salary_aav: float
    term_years: int
    signing_bonus: float = 0.0
    performance_bonuses: float = 0.0
    front_loaded_ratio: float = 0.5
    back_loaded_ratio: float = 0.5
    two_way: bool = False
    clauses: ClausePackage = field(default_factory=ClausePackage)

    # offer metadata
    offered_by_team_id: str = ""
    week: int = 0
    note: str = ""  # can store "lowball", "final", etc.

    def total_value(self) -> float:
        return self.salary_aav * self.term_years + self.signing_bonus + self.performance_bonuses


# ============================================================
# PCDS — Player Contract Desire Score
# ============================================================

@dataclass
class PCDSBreakdown:
    financial: float
    team_fit: float
    market: float
    memory: float
    timing: float
    total: float
    notes: List[str] = field(default_factory=list)

def _age_stage(age: int) -> str:
    if age <= 21:
        return "prospect"
    if age <= 26:
        return "young"
    if age <= 31:
        return "prime"
    if age <= 36:
        return "late_prime"
    if age <= 41:
        return "veteran"
    return "retirement_risk"

def _team_competitive_trajectory(team: TeamProfile) -> float:
    # very rough: competitive_score adjusted by stability and pressure
    score = team.state.competitive_score
    score += 0.15 * (team.philosophy.development_quality - 0.5)
    score += 0.10 * (team.roster.depth_quality - 0.5)
    score -= 0.10 * (team.ownership.meddling - 0.5)
    score -= 0.10 * (0.7 - team.state.stability)  # low stability hurts
    return clamp(score, 0.0, 1.0)

def _market_compatibility(player: PlayerProfile, team: TeamProfile) -> float:
    p = player.personality
    m = team.market
    # players with low media comfort dislike media pressure; same for market comfort vs fan expectations
    media_fit = 1.0 - abs(p.media_comfort - (1.0 - m.media_pressure))
    fan_fit = 1.0 - abs(p.market_comfort - (1.0 - m.fan_expectations))
    travel_fit = 1.0 - abs(0.5 - (1.0 - m.travel_burden))  # most players dislike travel
    weather_fit = 1.0 - abs(0.5 - m.weather_quality)       # weak preference
    tax_fit = m.tax_advantage                               # everyone likes lower tax
    return clamp(weighted_sum([
        (media_fit, 0.30),
        (fan_fit, 0.25),
        (travel_fit, 0.20),
        (tax_fit, 0.15),
        (weather_fit, 0.10),
    ]), 0.0, 1.0)

def _memory_fit(player: PlayerProfile, team: TeamProfile) -> float:
    mem = player.memory
    rel = mem.relationship.get(team.team_id, 0.0)  # [-1,1]
    betrayals = mem.betrayal_count.get(team.team_id, 0)
    lowballs = mem.lowball_count.get(team.team_id, 0)
    kept = mem.promise_kept_ratio.get(team.team_id, 0.5)

    score = 0.5 + 0.35 * rel + 0.25 * (kept - 0.5)
    score -= 0.10 * lowballs
    score -= 0.18 * betrayals
    return clamp(score, 0.0, 1.0)

def _financial_satisfaction(player: PlayerProfile, team: TeamProfile, offer: ContractOffer, league: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    league can include:
      - "cap": float
      - "salary_percentile": callable or dict
      - "expected_aav": float (for player)
    """
    notes: List[str] = []
    p = player.personality
    age = player.career.age
    stage = _age_stage(age)

    cap = float(league.get("cap", team.cap_total))
    expected_aav = float(league.get("expected_aav", max(800_000.0, player.career.ovr * 10_000_000.0)))

    # Raw money satisfaction around expected AAV
    money_ratio = offer.salary_aav / max(1.0, expected_aav)
    money_sat = sigmoid((money_ratio - 1.0) * 3.0)  # 0.5 at 1.0 ratio

    # Term security vs age/stage
    desired_term = 7 if stage in ("young", "prime") else 3 if stage in ("late_prime", "veteran") else 2
    term_fit = 1.0 - abs((offer.term_years - desired_term) / max(1.0, desired_term))
    term_fit = clamp(term_fit, 0.0, 1.0)

    # Bonuses: risk-averse like signing bonus; ambitious might accept performance bonuses
    bonus_total = offer.signing_bonus + offer.performance_bonuses
    bonus_sat = sigmoid((bonus_total / max(1.0, offer.salary_aav * offer.term_years)) * 8.0 - 0.5)
    # weight signing bonus more if risk_tolerance is low
    bonus_weight = lerp(0.6, 0.2, p.risk_tolerance)
    perf_weight = 1.0 - bonus_weight

    # Loading: front-load preference increases with age and injury concern
    front_pref = clamp(0.35 + 0.02 * (age - 25) + 0.4 * (player.career.wear_and_tear), 0.0, 1.0)
    loading_fit = 1.0 - abs(offer.front_loaded_ratio - front_pref)

    # Cap percent awareness: money_focus players care about relative share
    cap_percent = offer.salary_aav / max(1.0, cap)
    cap_share_sat = sigmoid((cap_percent - 0.09) * 12.0)  # ~9% cap is “star money”
    cap_share_weight = 0.15 + 0.35 * p.money_focus

    financial = weighted_sum([
        (money_sat, 0.45 + 0.25 * p.money_focus),
        (term_fit, 0.20 + 0.25 * p.stability_need),
        (bonus_sat, 0.15 + 0.20 * (1.0 - p.risk_tolerance)),
        (loading_fit, 0.10),
        (cap_share_sat, cap_share_weight),
    ])

    if money_ratio < 0.85:
        notes.append("offer feels like a lowball vs expectation")
    if offer.clauses.clause_type != ClauseType.NONE:
        notes.append(f"clauses add security ({offer.clauses.clause_type.value})")
    if offer.signing_bonus > 0:
        notes.append("signing bonus reduces perceived risk")

    return clamp(financial, 0.0, 1.0), notes

def _team_fit_score(player: PlayerProfile, team: TeamProfile) -> Tuple[float, List[str]]:
    notes: List[str] = []
    p = player.personality
    c = player.career

    trajectory = _team_competitive_trajectory(team)
    stability = clamp(team.state.stability, 0.0, 1.0)

    # management/ownership quality (player perspective)
    mgmt = weighted_sum([
        (team.reputation.management_reputation, 0.45),
        (team.reputation.development_reputation, 0.25),
        (1.0 - team.ownership.meddling, 0.30),
    ])
    coach_stability_proxy = clamp(stability * (1.0 - team.state.org_pressure), 0.0, 1.0)

    # roster: stars attract ambitious/legacy-driven players
    star_pull = clamp(team.roster.star_count / 4.0, 0.0, 1.0)
    core_pull = clamp(team.roster.core_count / 10.0, 0.0, 1.0)
    roster_fit = weighted_sum([
        (star_pull, 0.55),
        (core_pull, 0.25),
        (team.roster.depth_quality, 0.20),
    ])

    # role promise proxy: if current ice-time satisfaction low, player demands improvement
    role_fit = clamp(0.5 + 0.5 * c.ice_time_satisfaction - 0.6 * c.role_mismatch, 0.0, 1.0)

    # weights depend on personality
    team_fit = weighted_sum([
        (trajectory, 0.35 + 0.25 * p.ambition + 0.25 * p.legacy_drive),
        (mgmt, 0.20 + 0.20 * (1.0 - p.volatility)),
        (coach_stability_proxy, 0.15 + 0.25 * p.stability_need),
        (roster_fit, 0.15 + 0.20 * p.ambition),
        (role_fit, 0.15),
    ])

    if team.ownership.meddling > 0.7:
        notes.append("ownership meddling is high (trust risk)")
    if trajectory > 0.65:
        notes.append("team feels like a real window")
    if stability < 0.35:
        notes.append("org stability is shaky")

    return clamp(team_fit, 0.0, 1.0), notes

def compute_pcds(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    offer: ContractOffer,
    league: Dict[str, Any],
    context_kind: ContractContextKind = ContractContextKind.UFA,
    week: int = 0,
) -> PCDSBreakdown:
    """
    Computes Player Contract Desire Score (0..1).
    Fluctuates slightly with week (negotiation fatigue / narrative volatility).
    """
    notes: List[str] = []
    p = player.personality
    c = player.career

    financial, fin_notes = _financial_satisfaction(player, team, offer, league)
    notes += fin_notes

    team_fit, team_notes = _team_fit_score(player, team)
    notes += team_notes

    market = _market_compatibility(player, team)
    memory = _memory_fit(player, team)

    # timing: prime window, injury/wear, legacy urgency, family events
    stage = _age_stage(c.age)
    prime_factor = 1.0 if stage == "prime" else 0.8 if stage in ("young", "late_prime") else 0.6
    injury_factor = 1.0 - clamp(c.wear_and_tear + 0.15 * c.chronic_injuries, 0.0, 1.0)
    legacy_urgency = clamp(p.legacy_drive * (0.2 + c.legacy_pressure), 0.0, 1.0)
    family_pull = clamp(p.family_priority * (0.2 + 0.5 * c.security_anxiety), 0.0, 1.0)

    # UFA / resign timing differences
    if context_kind == ContractContextKind.RESIGN:
        # resign weighs memory & stability more
        timing = weighted_sum([
            (prime_factor, 0.25),
            (injury_factor, 0.25),
            (1.0 - family_pull, 0.15),
            (legacy_urgency, 0.15),
            (memory, 0.20),
        ])
    else:
        timing = weighted_sum([
            (prime_factor, 0.25),
            (injury_factor, 0.25),
            (1.0 - family_pull, 0.10),
            (legacy_urgency, 0.20),
            (clamp(team_fit, 0.0, 1.0), 0.20),
        ])

    # Clause security boosts desire for stability-need / risk-averse players
    clause_sec = offer.clauses.security_power()
    clause_bonus = clause_sec * (0.15 + 0.25 * p.stability_need + 0.20 * (1.0 - p.risk_tolerance))
    if clause_sec > 0.0:
        notes.append("security clause increases comfort")

    # Negotiation fatigue / volatility: weekly wiggle
    wiggle = (rng.random() - 0.5) * 0.04
    fatigue = clamp((week / 12.0), 0.0, 1.0)  # after ~12 weeks, gets tired
    fatigue_penalty = fatigue * (0.03 + 0.07 * (1.0 - p.patience))

    # total blend
    base_total = weighted_sum([
        (financial, 0.34 + 0.22 * p.money_focus),
        (team_fit, 0.24 + 0.20 * p.ambition + 0.15 * p.legacy_drive),
        (market, 0.14 + 0.12 * p.family_priority),
        (memory, 0.14 + 0.18 * p.loyalty),
        (timing, 0.14),
    ])

    total = clamp(base_total + clause_bonus + wiggle - fatigue_penalty, 0.0, 1.0)

    # annotate some key states
    if memory < 0.35:
        notes.append("relationship memory is poor")
    if market < 0.35:
        notes.append("market/location fit is poor")
    if financial < 0.40:
        notes.append("financial structure isn't landing")
    if fatigue > 0.75:
        notes.append("negotiations feel dragged out")

    return PCDSBreakdown(
        financial=financial,
        team_fit=team_fit,
        market=market,
        memory=memory,
        timing=clamp(timing, 0.0, 1.0),
        total=total,
        notes=notes
    )


# ============================================================
# Hometown Deal Logic
# ============================================================

@dataclass
class HometownDealResult:
    eligible: bool
    triggered: bool
    discount_pct: float = 0.0
    notes: List[str] = field(default_factory=list)

def evaluate_hometown_deal(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    base_offer: ContractOffer,
) -> HometownDealResult:
    """
    Hometown deals must be rare, contextual, and earned.
    Preconditions:
      - drafted or developed by team
      - high loyalty
      - low ego
      - good emotional memory
      - team competitiveness and respectful negotiation history
    Outcomes:
      - reduced AAV (5–20%)
      - may accept longer term
      - bonus-heavy preference
      - NMC more likely
    """
    notes: List[str] = []
    p = player.personality
    mem = player.memory

    drafted = (mem.drafted_by_team_id == team.team_id)
    developed = (mem.developed_by_team_id == team.team_id)
    if not (drafted or developed):
        return HometownDealResult(False, False, notes=["not drafted/developed here"])

    loyalty_gate = p.loyalty
    ego_gate = 1.0 - p.ego
    memory_fit = _memory_fit(player, team)
    window = _team_competitive_trajectory(team)
    respect = clamp(player.career.offer_respect, 0.0, 1.0)

    eligible_score = weighted_sum([
        (1.0 if drafted else 0.7 if developed else 0.0, 0.20),
        (loyalty_gate, 0.25),
        (ego_gate, 0.15),
        (memory_fit, 0.20),
        (window, 0.10),
        (respect, 0.10),
    ])

    eligible = eligible_score >= 0.62
    if not eligible:
        return HometownDealResult(False, False, notes=["preconditions not met strongly enough"])

    # Rareness: even eligible doesn't always trigger
    # More likely if team is strong and memory is high and ego low
    trigger_p = clamp(0.04 + 0.18 * (memory_fit - 0.6) + 0.10 * (window - 0.6) + 0.08 * (ego_gate - 0.6), 0.02, 0.22)
    triggered = chance(rng, trigger_p)

    if not triggered:
        notes.append("eligible, but hometown discount did not trigger this time")
        return HometownDealResult(True, False, notes=notes)

    # discount 5–20%
    discount = clamp(0.05 + rng.random() * 0.15, 0.05, 0.20)
    notes.append(f"hometown deal triggered: discount {discount*100:.1f}%")

    return HometownDealResult(True, True, discount_pct=discount, notes=notes)


# ============================================================
# Offer Generation / Team-side constraints
# ============================================================

@dataclass
class TeamNegotiationPlan:
    # budget ceiling on AAV and total commitment
    max_aav: float
    max_term: int
    willing_to_give_clause: float  # 0..1
    willingness_to_bonus: float    # 0..1
    risk_aversion: float           # 0..1 (prefers shorter term / lower bonus)
    notes: List[str] = field(default_factory=list)

def build_team_plan(rng: random.Random, team: TeamProfile, player: PlayerProfile, league: Dict[str, Any]) -> TeamNegotiationPlan:
    cap = float(league.get("cap", team.cap_total))
    expected_aav = float(league.get("expected_aav", max(800_000.0, player.career.ovr * 10_000_000.0)))

    # base ceiling from cap space and willingness
    budget = team.ownership.budget_willingness
    pressure = team.state.org_pressure
    meddling = team.ownership.meddling
    window = _team_competitive_trajectory(team)

    # if window is high and pressure high, they stretch
    stretch = 1.0 + 0.35 * (window - 0.5) + 0.25 * (pressure - 0.5) + 0.15 * (meddling - 0.5)
    stretch = clamp(stretch, 0.75, 1.6)

    # cap-space constraint: don't exceed space * a factor, but contenders may go close
    soft_max = team.cap_space * (0.55 + 0.55 * budget + 0.25 * (window - 0.5))
    soft_max = max(1_000_000.0, soft_max)

    max_aav = min(expected_aav * stretch, soft_max)
    max_aav = max(max_aav, expected_aav * 0.75)  # teams can always lowball below expected, but plan includes a floor

    # term appetite depends on risk tolerance and player age
    age = player.career.age
    if age <= 25:
        base_term = 6
    elif age <= 29:
        base_term = 5
    elif age <= 33:
        base_term = 4
    elif age <= 36:
        base_term = 3
    else:
        base_term = 2

    risk_aversion = clamp(0.35 + 0.35 * (1.0 - team.philosophy.risk_tolerance) + 0.20 * (1.0 - team.state.financial_health), 0.0, 1.0)
    max_term = int(clamp(base_term + (1 if window > 0.65 else 0) - (1 if risk_aversion > 0.7 else 0), 1, 8))

    willing_to_give_clause = clamp(0.10 + 0.25 * (pressure - 0.5) + 0.25 * (budget - 0.5) - 0.20 * risk_aversion, 0.0, 0.85)
    willingness_to_bonus = clamp(0.10 + 0.35 * budget + 0.15 * (window - 0.5) - 0.15 * risk_aversion, 0.0, 0.95)

    notes = []
    if window > 0.65:
        notes.append("team feels competitive; willing to stretch")
    if meddling > 0.7:
        notes.append("ownership meddling may force aggressive offers")
    if team.cap_space < expected_aav * 0.75:
        notes.append("cap space is tight; expect hard ceiling")

    return TeamNegotiationPlan(
        max_aav=float(max_aav),
        max_term=max_term,
        willing_to_give_clause=float(willing_to_give_clause),
        willingness_to_bonus=float(willingness_to_bonus),
        risk_aversion=float(risk_aversion),
        notes=notes
    )

def propose_opening_offer(
    rng: random.Random,
    team: TeamProfile,
    player: PlayerProfile,
    agent: AgentProfile,
    plan: TeamNegotiationPlan,
    league: Dict[str, Any]
) -> ContractOffer:
    expected_aav = float(league.get("expected_aav", max(800_000.0, player.career.ovr * 10_000_000.0)))

    # opening stance: lowball chance increases with risk_aversion and if team is "cheap"
    cheap_bias = 0.10 if team.archetype in ("cheap", "rebuild") else 0.0
    lowball_p = clamp(0.15 + 0.25 * plan.risk_aversion + cheap_bias - 0.10 * (agent.league_influence - 0.5), 0.05, 0.55)
    lowball = chance(rng, lowball_p)

    if lowball:
        aav = expected_aav * (0.72 + rng.random() * 0.12)  # 72–84%
        note = "lowball"
    else:
        aav = expected_aav * (0.88 + rng.random() * 0.16)  # 88–104%
        note = "opening"

    aav = min(aav, plan.max_aav)
    aav = max(800_000.0, aav)

    # term offer
    term = max(1, min(plan.max_term, int(round(lerp(2.0, float(plan.max_term), 0.55 + 0.25 * rng.random())))))

    # bonuses
    bonus_scale = plan.willingness_to_bonus * (0.3 + 0.7 * rng.random())
    signing_bonus = aav * term * 0.05 * bonus_scale
    perf_bonus = aav * term * 0.02 * (bonus_scale * 0.6)

    # clauses: usually none in opening unless player is high leverage
    clauses = ClausePackage(clause_type=ClauseType.NONE)
    leverage = estimate_leverage_score(rng, player, team, agent, league)
    if leverage > 0.75 and chance(rng, plan.willing_to_give_clause * 0.55):
        clauses = _generate_clause_package(rng, player, team, agent, leverage)

    return ContractOffer(
        salary_aav=float(aav),
        term_years=int(term),
        signing_bonus=float(signing_bonus),
        performance_bonuses=float(perf_bonus),
        front_loaded_ratio=float(clamp(0.45 + 0.25 * rng.random(), 0.25, 0.85)),
        back_loaded_ratio=float(clamp(0.55 - 0.25 * rng.random(), 0.15, 0.75)),
        two_way=False,
        clauses=clauses,
        offered_by_team_id=team.team_id,
        week=0,
        note=note
    )

def estimate_leverage_score(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    league: Dict[str, Any]
) -> float:
    """
    Abstract leverage: ability to demand term/money/clauses.
    Includes:
      - player OVR
      - UFA pressure
      - agent influence
      - team desperation/window/pressure
      - market heat proxy
    """
    c = player.career
    window = _team_competitive_trajectory(team)
    desperation = clamp(0.35 + 0.35 * (team.state.org_pressure) + 0.25 * (window - 0.5), 0.0, 1.0)
    ufa = clamp(c.ufa_pressure, 0.0, 1.0)
    skill = clamp(c.ovr, 0.0, 1.0)
    influence = clamp(agent.league_influence, 0.0, 1.0)

    # slight random market heat
    market_heat = clamp(0.45 + (rng.random() - 0.5) * 0.15 + 0.15 * ufa + 0.10 * influence, 0.0, 1.0)

    leverage = weighted_sum([
        (skill, 0.35),
        (ufa, 0.20),
        (influence, 0.15),
        (desperation, 0.20),
        (market_heat, 0.10),
    ])
    return clamp(leverage, 0.0, 1.0)

def _generate_clause_package(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    leverage: float
) -> ClausePackage:
    """
    Clauses traded for lower AAV / shorter term / bonuses.
    NMC rare and dangerous, more likely with high leverage, high stability need, low risk tolerance.
    """
    p = player.personality
    clause_hunger = clamp(
        0.15 + 0.35 * p.stability_need + 0.25 * (1.0 - p.risk_tolerance) + 0.20 * leverage + 0.10 * agent.aggression,
        0.0, 1.0
    )

    # decide clause type
    nmc_p = clamp(0.02 + 0.08 * (leverage - 0.7) + 0.10 * (p.stability_need - 0.7), 0.0, 0.18)
    ntc_p = clamp(0.08 + 0.22 * clause_hunger, 0.0, 0.55)
    mntc_p = clamp(0.18 + 0.25 * clause_hunger, 0.0, 0.65)

    r = rng.random()
    if r < nmc_p:
        clause_type = ClauseType.NMC
        trade_list_size = 0
    elif r < nmc_p + ntc_p:
        clause_type = ClauseType.NTC
        trade_list_size = 0
    else:
        clause_type = ClauseType.MNTC
        # smaller list if ego high or market comfort low or agent aggressive
        base = int(round(lerp(14, 6, 0.45 * p.ego + 0.35 * (1.0 - p.market_comfort) + 0.20 * agent.aggression)))
        trade_list_size = int(clamp(base + int((rng.random() - 0.5) * 4), 4, 18))

    waiver_protection = chance(rng, clamp(0.10 + 0.25 * clause_hunger, 0.0, 0.6))
    buyout_penalty_profile = clamp(0.35 + 0.35 * leverage + 0.20 * (1.0 - team.state.financial_health), 0.0, 1.0)

    return ClausePackage(
        clause_type=clause_type,
        trade_list_size=trade_list_size,
        waiver_protection=waiver_protection,
        retention_allowed=True,
        buyout_penalty_profile=buyout_penalty_profile
    )


# ============================================================
# Negotiation Engine (Multi-Phase)
# ============================================================

@dataclass
class NegotiationState:
    phase: NegotiationPhase = NegotiationPhase.OPENING
    week: int = 0
    max_weeks: int = 16

    player_internal_ask_aav: float = 0.0
    player_internal_ask_term: int = 0
    team_ceiling_aav: float = 0.0
    team_ceiling_term: int = 0

    player_patience: float = 0.5
    player_trust: float = 0.5
    player_ego_heat: float = 0.5
    public_heat: float = 0.3

    competing_offers: int = 0
    rumor_active: bool = False

    last_offer: Optional[ContractOffer] = None
    best_offer: Optional[ContractOffer] = None

    log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class NegotiationResult:
    outcome: NegotiationOutcome
    accepted_offer: Optional[ContractOffer]
    contract: Optional[Contract]
    pcds_at_accept: Optional[PCDSBreakdown]
    negotiation_log: List[Dict[str, Any]]
    notes: List[str] = field(default_factory=list)

def _compute_player_internal_ask(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    league: Dict[str, Any],
    context_kind: ContractContextKind
) -> Tuple[float, int]:
    """
    Player sets internal ask (not revealed). Agent adjusts posture.
    """
    expected_aav = float(league.get("expected_aav", max(800_000.0, player.career.ovr * 10_000_000.0)))
    p = player.personality
    stage = _age_stage(player.career.age)

    # ambition/money/ego/agent aggression push ask up
    push = 1.0 + 0.10 * (p.ambition - 0.5) + 0.18 * (p.money_focus - 0.5) + 0.12 * (p.ego - 0.5) + 0.16 * (agent.aggression - 0.5)
    # loyalty/memory/team fit can reduce ask slightly on resign
    mem_fit = _memory_fit(player, team)
    if context_kind in (ContractContextKind.RESIGN, ContractContextKind.EXTENSION):
        push -= 0.08 * (p.loyalty - 0.5) - 0.10 * (mem_fit - 0.5)

    # cap context: if cap rising, demand shorter for upside (money-focus/ambition)
    cap_growth = float(league.get("cap_growth", team.cap_projection_growth))
    wants_shorter = clamp(0.30 + 0.50 * cap_growth + 0.35 * p.ambition, 0.0, 1.0)
    base_term = 7 if stage in ("young", "prime") else 4 if stage in ("late_prime", "veteran") else 2
    term = base_term - (1 if wants_shorter > 0.65 and stage in ("prime", "young") else 0)
    term += (1 if p.stability_need > 0.7 else 0)
    term = int(clamp(term + int((rng.random() - 0.5) * 2), 1, 8))

    ask_aav = expected_aav * clamp(push, 0.75, 1.45)
    ask_aav *= (0.98 + 0.06 * rng.random())
    ask_aav = max(850_000.0, ask_aav)

    return float(ask_aav), int(term)

def _apply_lowball_consequences(
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    state: NegotiationState
) -> None:
    """
    Lowball offers increase hostility, reduce future willingness, may trigger leaks.
    """
    p = player.personality
    # trust drops, ego heat rises
    state.player_trust = clamp(state.player_trust - (0.04 + 0.08 * (1.0 - p.patience)), 0.0, 1.0)
    state.player_ego_heat = clamp(state.player_ego_heat + (0.05 + 0.06 * p.ego), 0.0, 1.0)
    player.career.offer_respect = clamp(player.career.offer_respect - 0.10, 0.0, 1.0)
    # memory update
    player.memory.lowball_count[team.team_id] = player.memory.lowball_count.get(team.team_id, 0) + 1
    player.memory.relationship[team.team_id] = clamp(player.memory.relationship.get(team.team_id, 0.0) - 0.08, -1.0, 1.0)
    agent.remember_team(team.team_id, -0.04)

def _pressure_events(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    state: NegotiationState
) -> List[str]:
    """
    Phase 3 — Pressure Events
      - media rumors
      - competing offers
      - team desperation
      - ultimatums
    """
    notes: List[str] = []
    p = player.personality

    # media leak chance
    leak_p = clamp(0.05 + 0.20 * agent.media_leak_tendency + 0.12 * (1.0 - state.player_trust) + 0.10 * state.player_ego_heat, 0.0, 0.45)
    if chance(rng, leak_p):
        state.rumor_active = True
        state.public_heat = clamp(state.public_heat + 0.15, 0.0, 1.0)
        notes.append("agent leak triggers media rumor cycle")
        # rumors strain trust
        state.player_trust = clamp(state.player_trust - 0.03, 0.0, 1.0)

    # competing offers appear (agent influence)
    comp_p = clamp(0.10 + 0.35 * agent.league_influence + 0.15 * player.career.ufa_pressure, 0.0, 0.70)
    if chance(rng, comp_p):
        new_offers = 1 + (1 if chance(rng, 0.35 + 0.25 * agent.league_influence) else 0)
        state.competing_offers += new_offers
        notes.append(f"competing offers appear ({new_offers})")
        # competing offers boost internal ask slightly
        state.player_internal_ask_aav *= (1.02 + 0.02 * rng.random())

    # player ultimatum (if low patience + high ego heat + long talks)
    ultimatum_p = clamp(0.03 + 0.12 * (1.0 - p.patience) + 0.10 * state.player_ego_heat + 0.08 * (state.week / max(1, state.max_weeks)), 0.0, 0.35)
    if chance(rng, ultimatum_p):
        notes.append("player issues an ultimatum (time pressure)")
        state.public_heat = clamp(state.public_heat + 0.10, 0.0, 1.0)

    # team desperation: org pressure rises, may increase ceiling slightly or force owner meddle
    desperation = clamp(team.state.org_pressure + 0.25 * (0.7 - team.state.stability), 0.0, 1.0)
    if chance(rng, 0.06 + 0.18 * desperation):
        team.state.org_pressure = clamp(team.state.org_pressure + 0.05, 0.0, 1.0)
        notes.append("team desperation spikes (injuries/window pressure)")

    return notes

def _team_counter_offer(
    rng: random.Random,
    team: TeamProfile,
    player: PlayerProfile,
    agent: AgentProfile,
    plan: TeamNegotiationPlan,
    state: NegotiationState
) -> ContractOffer:
    """
    Team responds with a counter; may trade clauses for money/term.
    """
    assert state.last_offer is not None
    last = state.last_offer
    p = player.personality

    # move AAV towards player's ask but cap at plan.max_aav
    desired = state.player_internal_ask_aav
    step = 0.12 + 0.18 * (team.state.org_pressure) + 0.08 * agent.league_influence
    step -= 0.12 * plan.risk_aversion
    step = clamp(step, 0.06, 0.30)

    new_aav = lerp(last.salary_aav, min(desired, plan.max_aav), step)
    # if rumors/competing offers, step up
    if state.competing_offers > 0 or state.rumor_active:
        new_aav *= (1.01 + 0.02 * rng.random())

    new_aav = min(new_aav, plan.max_aav)
    new_aav = max(800_000.0, new_aav)

    # term adjustment: stability-need pushes for longer; team risk reduces
    desired_term = state.player_internal_ask_term
    term_step = 1 if chance(rng, 0.45 + 0.15 * p.stability_need - 0.20 * plan.risk_aversion) else 0
    new_term = last.term_years
    if new_term < desired_term and new_term < plan.max_term and term_step:
        new_term += 1
    elif new_term > desired_term and chance(rng, 0.25 + 0.25 * plan.risk_aversion):
        new_term -= 1
    new_term = int(clamp(new_term, 1, plan.max_term))

    # bonuses: increase to soothe risk-averse players, or to trade for lower AAV
    bonus_scale = plan.willingness_to_bonus * (0.35 + 0.65 * rng.random())
    signing_bonus = new_aav * new_term * 0.05 * bonus_scale
    perf_bonus = new_aav * new_term * 0.02 * (bonus_scale * 0.5)

    # clauses: if player stability need high and agent aggressive, may add MNTC
    clauses = last.clauses
    leverage = estimate_leverage_score(rng, player, team, agent, {"cap": team.cap_total, "expected_aav": state.player_internal_ask_aav})
    clause_add_p = clamp(0.04 + 0.18 * plan.willing_to_give_clause + 0.10 * p.stability_need + 0.10 * agent.aggression + 0.12 * (leverage - 0.6), 0.0, 0.55)
    if clauses.clause_type == ClauseType.NONE and chance(rng, clause_add_p):
        clauses = _generate_clause_package(rng, player, team, agent, leverage)

        # clauses traded for a slight AAV haircut if money focus is not extreme
        haircut = 1.0 - clamp(0.01 + 0.05 * clauses.security_power(), 0.0, 0.07)
        new_aav *= haircut

    return ContractOffer(
        salary_aav=float(new_aav),
        term_years=int(new_term),
        signing_bonus=float(signing_bonus),
        performance_bonuses=float(perf_bonus),
        front_loaded_ratio=float(clamp(last.front_loaded_ratio + (rng.random() - 0.5) * 0.05, 0.25, 0.85)),
        back_loaded_ratio=float(clamp(1.0 - last.front_loaded_ratio, 0.15, 0.75)),
        clauses=clauses,
        offered_by_team_id=team.team_id,
        week=state.week,
        note="counter"
    )

def _player_acceptance_threshold(player: PlayerProfile, context_kind: ContractContextKind) -> float:
    """
    Higher threshold => player needs higher PCDS to accept.
    Varies by patience, ego, UFA pressure, resign loyalty.
    """
    p = player.personality
    c = player.career

    base = 0.64
    base += 0.08 * (p.ego - 0.5)
    base += 0.06 * (p.ambition - 0.5)
    base += 0.06 * (p.money_focus - 0.5)
    base -= 0.08 * (p.patience - 0.5)

    # UFA players can walk -> higher bar
    base += 0.08 * c.ufa_pressure

    # resigning (loyal players accept easier if memory good)
    if context_kind == ContractContextKind.RESIGN:
        base -= 0.05 * (p.loyalty - 0.5)

    return clamp(base, 0.50, 0.85)

def negotiate_contract(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    league: Dict[str, Any],
    signing_year: int,
    context_kind: ContractContextKind = ContractContextKind.UFA,
    max_weeks: int = 16
) -> NegotiationResult:
    """
    Runs multi-phase negotiation and returns outcome + accepted Contract (if any).
    """
    notes: List[str] = []
    state = NegotiationState(max_weeks=max_weeks)

    # Phase 1 — Opening Stance
    state.phase = NegotiationPhase.OPENING
    plan = build_team_plan(rng, team, player, league)
    ask_aav, ask_term = _compute_player_internal_ask(rng, player, team, agent, league, context_kind)
    state.player_internal_ask_aav = ask_aav
    state.player_internal_ask_term = ask_term
    state.team_ceiling_aav = plan.max_aav
    state.team_ceiling_term = plan.max_term

    # initial trust baseline from memory/agent trust
    mem_fit = _memory_fit(player, team)
    state.player_trust = clamp(0.45 + 0.35 * mem_fit + 0.10 * (team.reputation.management_reputation - 0.5), 0.0, 1.0)
    state.player_patience = clamp(player.personality.patience, 0.0, 1.0)
    state.player_ego_heat = clamp(0.35 + 0.25 * player.personality.ego, 0.0, 1.0)

    opening_offer = propose_opening_offer(rng, team, player, agent, plan, league)
    state.last_offer = opening_offer
    state.best_offer = opening_offer

    # record
    state.log.append({
        "phase": state.phase.value,
        "week": state.week,
        "action": "opening_offer",
        "offer": asdict(opening_offer),
        "player_internal_ask": {"aav": ask_aav, "term": ask_term},
        "team_plan": asdict(plan)
    })

    # hometown deal check can modify player's willingness (not auto-accept)
    hometown = evaluate_hometown_deal(rng, player, team, opening_offer)
    if hometown.eligible:
        notes += hometown.notes
    if hometown.triggered:
        # apply discount "willingness": player will accept lower AAV if other fits good
        state.player_internal_ask_aav *= (1.0 - hometown.discount_pct * (0.6 + 0.4 * player.personality.loyalty))
        # also more likely to demand security clause
        if opening_offer.clauses.clause_type == ClauseType.NONE and chance(rng, 0.30):
            opening_offer.clauses = ClausePackage(clause_type=ClauseType.MNTC, trade_list_size=10)
            state.last_offer = opening_offer
            state.best_offer = opening_offer

    # negotiation loop
    accept_threshold = _player_acceptance_threshold(player, context_kind)

    # an offer can be accepted at any week if PCDS >= threshold
    for week in range(0, max_weeks):
        state.week = week

        # Evaluate last offer
        offer = state.last_offer
        assert offer is not None
        pcds = compute_pcds(rng, player, team, offer, league, context_kind=context_kind, week=week)

        # Update best offer
        if state.best_offer is None or offer.total_value() > state.best_offer.total_value():
            state.best_offer = offer

        # Acceptance check
        if pcds.total >= accept_threshold:
            # Accept
            state.phase = NegotiationPhase.RESOLUTION
            contract = finalize_contract(
                rng=rng,
                player=player,
                team=team,
                agent=agent,
                offer=offer,
                league=league,
                signing_year=signing_year,
                leverage=estimate_leverage_score(rng, player, team, agent, league),
            )
            _apply_reputation_on_sign(team, player, agent, pcds, accepted=True, lowball=False, rumor=state.rumor_active)
            _write_player_memory_on_resolution(player, team, agent, state, contract, accepted=True)

            state.log.append({
                "phase": state.phase.value,
                "week": week,
                "action": "accept",
                "pcds": asdict(pcds),
                "accept_threshold": accept_threshold,
                "final_contract": contract.to_dict(),
            })

            return NegotiationResult(
                outcome=NegotiationOutcome.ACCEPT,
                accepted_offer=offer,
                contract=contract,
                pcds_at_accept=pcds,
                negotiation_log=state.log,
                notes=notes
            )

        # If offer is lowball, apply consequences early
        if offer.note == "lowball":
            _apply_lowball_consequences(player, team, agent, state)

        # Phase transitions:
        if week == 0:
            state.phase = NegotiationPhase.COUNTERS
        if week >= max(2, max_weeks // 2):
            state.phase = NegotiationPhase.PRESSURE

        # Pressure events
        if state.phase == NegotiationPhase.PRESSURE:
            pe_notes = _pressure_events(rng, player, team, agent, state)
            if pe_notes:
                state.log.append({
                    "phase": state.phase.value,
                    "week": week,
                    "action": "pressure_event",
                    "notes": pe_notes
                })

        # Decide to walk / stall / force trade near end
        end_frac = week / max(1, max_weeks - 1)
        walk_p = clamp(0.02 + 0.18 * end_frac + 0.10 * (1.0 - state.player_trust) + 0.08 * (1.0 - player.personality.patience) + 0.12 * player.career.ufa_pressure, 0.0, 0.55)
        # if best offer is still far from ask, increases walk
        gap = (state.player_internal_ask_aav - offer.salary_aav) / max(1.0, state.player_internal_ask_aav)
        walk_p += clamp(gap, 0.0, 0.4) * 0.25

        # Force trade logic: if resign/extension and trust collapses
        force_trade_p = 0.0
        if context_kind in (ContractContextKind.RESIGN, ContractContextKind.EXTENSION):
            force_trade_p = clamp(0.01 + 0.10 * end_frac + 0.15 * (1.0 - state.player_trust) + 0.10 * state.player_ego_heat, 0.0, 0.35)

        if week >= max_weeks - 2:
            if chance(rng, force_trade_p):
                state.phase = NegotiationPhase.RESOLUTION
                _apply_reputation_on_sign(team, player, agent, pcds, accepted=False, lowball=False, rumor=state.rumor_active)
                _write_player_memory_on_resolution(player, team, agent, state, None, accepted=False)
                state.log.append({
                    "phase": state.phase.value,
                    "week": week,
                    "action": "force_trade",
                    "pcds": asdict(pcds),
                })
                return NegotiationResult(
                    outcome=NegotiationOutcome.FORCE_TRADE,
                    accepted_offer=None,
                    contract=None,
                    pcds_at_accept=None,
                    negotiation_log=state.log,
                    notes=notes + ["player forces trade after negotiation collapse"]
                )

            if chance(rng, walk_p):
                state.phase = NegotiationPhase.RESOLUTION
                _apply_reputation_on_sign(team, player, agent, pcds, accepted=False, lowball=False, rumor=state.rumor_active)
                _write_player_memory_on_resolution(player, team, agent, state, None, accepted=False)
                state.log.append({
                    "phase": state.phase.value,
                    "week": week,
                    "action": "walk",
                    "pcds": asdict(pcds),
                    "walk_probability": walk_p
                })
                return NegotiationResult(
                    outcome=NegotiationOutcome.WALK,
                    accepted_offer=None,
                    contract=None,
                    pcds_at_accept=None,
                    negotiation_log=state.log,
                    notes=notes + ["player walks to market"]
                )

        # Otherwise counteroffer
        state.phase = NegotiationPhase.COUNTERS
        counter = _team_counter_offer(rng, team, player, agent, plan, state)

        # Team cannot exceed cap space in a simple way (you can make this more realistic)
        if counter.salary_aav > team.cap_space * 1.05:
            # degrade offer and/or shorten term
            counter.salary_aav = min(counter.salary_aav, team.cap_space * 1.05)
            counter.term_years = max(1, min(counter.term_years, 3))
            counter.note = "cap_limited"

        state.last_offer = counter
        state.log.append({
            "phase": state.phase.value,
            "week": week + 1,
            "action": "team_counter",
            "offer": asdict(counter),
        })

    # If still not resolved, stall or one-year prove-it
    final_offer = state.best_offer
    assert final_offer is not None
    final_pcds = compute_pcds(rng, player, team, final_offer, league, context_kind=context_kind, week=max_weeks)

    # One-year prove-it if money focus low and team fit good but financial not
    prove_it_p = clamp(0.05 + 0.20 * (final_pcds.team_fit - 0.6) + 0.15 * (player.personality.ambition - 0.5) - 0.20 * (player.personality.money_focus - 0.5), 0.0, 0.35)
    if chance(rng, prove_it_p):
        state.phase = NegotiationPhase.RESOLUTION
        prove_offer = ContractOffer(
            salary_aav=max(1_000_000.0, final_offer.salary_aav * 0.92),
            term_years=1,
            signing_bonus=final_offer.signing_bonus * 0.40,
            performance_bonuses=final_offer.performance_bonuses * 0.50,
            front_loaded_ratio=0.65,
            back_loaded_ratio=0.35,
            two_way=False,
            clauses=ClausePackage(clause_type=ClauseType.NONE),
            offered_by_team_id=team.team_id,
            week=max_weeks,
            note="prove_it"
        )
        pcds_pi = compute_pcds(rng, player, team, prove_offer, league, context_kind=context_kind, week=max_weeks)
        contract = finalize_contract(rng, player, team, agent, prove_offer, league, signing_year, leverage=estimate_leverage_score(rng, player, team, agent, league))

        _apply_reputation_on_sign(team, player, agent, pcds_pi, accepted=True, lowball=False, rumor=state.rumor_active)
        _write_player_memory_on_resolution(player, team, agent, state, contract, accepted=True)

        state.log.append({
            "phase": state.phase.value,
            "week": max_weeks,
            "action": "one_year_prove_it",
            "offer": asdict(prove_offer),
            "pcds": asdict(pcds_pi),
            "final_contract": contract.to_dict(),
        })

        return NegotiationResult(
            outcome=NegotiationOutcome.ONE_YEAR_PROVE_IT,
            accepted_offer=prove_offer,
            contract=contract,
            pcds_at_accept=pcds_pi,
            negotiation_log=state.log,
            notes=notes + ["agreement reached on one-year prove-it deal"]
        )

    # Otherwise stall
    state.phase = NegotiationPhase.RESOLUTION
    _apply_reputation_on_sign(team, player, agent, final_pcds, accepted=False, lowball=False, rumor=state.rumor_active)
    _write_player_memory_on_resolution(player, team, agent, state, None, accepted=False)

    state.log.append({
        "phase": state.phase.value,
        "week": max_weeks,
        "action": "stall",
        "best_offer": asdict(final_offer),
        "pcds_best": asdict(final_pcds)
    })

    return NegotiationResult(
        outcome=NegotiationOutcome.STALL,
        accepted_offer=None,
        contract=None,
        pcds_at_accept=None,
        negotiation_log=state.log,
        notes=notes + ["negotiations stall; revisit later"]
    )


# ============================================================
# Finalization + Reputation + Memory
# ============================================================

def finalize_contract(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    offer: ContractOffer,
    league: Dict[str, Any],
    signing_year: int,
    leverage: float
) -> Contract:
    cap = float(league.get("cap", team.cap_total))
    expiry_age = player.career.age + offer.term_years

    # contract remembers environment
    ctx = {
        "cap": cap,
        "cap_growth": float(league.get("cap_growth", team.cap_projection_growth)),
        "team_status": team.status,
        "team_stability": team.state.stability,
        "signed_under_rumors": False,  # can be injected from negotiation state
        "timestamp": now_ts(),
    }

    c = Contract(
        salary_aav=float(offer.salary_aav),
        term_years=int(offer.term_years),
        total_value=float(offer.total_value()),
        signing_bonus=float(offer.signing_bonus),
        performance_bonuses=float(offer.performance_bonuses),
        front_loaded_ratio=float(offer.front_loaded_ratio),
        back_loaded_ratio=float(offer.back_loaded_ratio),
        two_way=bool(offer.two_way),
        expiry_age=int(expiry_age),
        clauses=offer.clauses,
        signed_team_id=team.team_id,
        signed_market_type=team.market.market_size,
        signing_year=int(signing_year),
        cap_percent_at_signing=float(offer.salary_aav / max(1.0, cap)),
        league_cap_context=ctx,
        player_leverage_score=float(leverage),
        negotiated_by_agent_id=agent.agent_id,
        negotiation_length_days=offer.week * 7,
        promises={},              # you can populate externally
        declined_offers=[],
    )

    # update player career last contract info
    player.career.last_contract_aav = c.salary_aav
    player.career.last_contract_term = c.term_years
    player.career.offer_respect = clamp(player.career.offer_respect + 0.10, 0.0, 1.0)

    return c

def _apply_reputation_on_sign(
    team: TeamProfile,
    player: PlayerProfile,
    agent: AgentProfile,
    pcds: PCDSBreakdown,
    accepted: bool,
    lowball: bool,
    rumor: bool
) -> None:
    """
    Reputation feedback loops:
      - bad actors struggle to sign UFAs, overpay more, clauses harder, volatility higher
      - good actors get discounts, easier negotiations
    """
    # Team reputation adjusts with fairness and outcomes
    fairness = pcds.financial
    rel = player.memory.relationship.get(team.team_id, 0.0)

    if accepted:
        # successful signing improves management reputation slightly
        team.reputation.management_reputation = clamp(team.reputation.management_reputation + 0.01 + 0.02 * (fairness - 0.5), 0.0, 1.0)
        team.reputation.player_reputation = clamp(team.reputation.player_reputation + 0.01 + 0.01 * (pcds.team_fit - 0.5), 0.0, 1.0)
        # agent trust improves
        team.agent_trust[agent.agent_id] = clamp(team.agent_trust.get(agent.agent_id, 0.5) + 0.03, 0.0, 1.0)
        agent.remember_team(team.team_id, +0.04)
        # relationship memory becomes more positive
        player.memory.relationship[team.team_id] = clamp(rel + 0.06 + 0.08 * (pcds.total - 0.65), -1.0, 1.0)
    else:
        # failed negotiations hurt trust
        team.reputation.management_reputation = clamp(team.reputation.management_reputation - 0.01 - (0.01 if rumor else 0.0), 0.0, 1.0)
        team.agent_trust[agent.agent_id] = clamp(team.agent_trust.get(agent.agent_id, 0.5) - 0.03, 0.0, 1.0)
        agent.remember_team(team.team_id, -0.03)
        player.memory.relationship[team.team_id] = clamp(rel - 0.04, -1.0, 1.0)

    if lowball:
        team.reputation.management_reputation = clamp(team.reputation.management_reputation - 0.015, 0.0, 1.0)
        team.front_office_chaos = clamp(team.front_office_chaos + 0.02, 0.0, 1.0)

def _write_player_memory_on_resolution(
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
    state: NegotiationState,
    contract: Optional[Contract],
    accepted: bool
) -> None:
    """
    Persist negotiation memory hooks:
      - who negotiated
      - what was promised
      - what was declined
      - how long it lasted
    """
    record = {
        "timestamp": now_ts(),
        "team_id": team.team_id,
        "agent_id": agent.agent_id,
        "accepted": accepted,
        "weeks": state.week,
        "rumor": state.rumor_active,
        "competing_offers": state.competing_offers,
        "final_offer": state.last_offer.__dict__ if state.last_offer else None,
        "best_offer": state.best_offer.__dict__ if state.best_offer else None,
        "internal_ask": {"aav": state.player_internal_ask_aav, "term": state.player_internal_ask_term},
        "contract": contract.to_dict() if contract else None,
    }
    player.memory.negotiation_log.append(record)

    # Adjust promise-kept ratio baseline on acceptance (initial goodwill)
    if accepted:
        prev = player.memory.promise_kept_ratio.get(team.team_id, 0.5)
        player.memory.promise_kept_ratio[team.team_id] = clamp(prev + 0.03, 0.0, 1.0)
    else:
        prev = player.memory.promise_kept_ratio.get(team.team_id, 0.5)
        player.memory.promise_kept_ratio[team.team_id] = clamp(prev - 0.02, 0.0, 1.0)


# ============================================================
# Trade Response Logic
# ============================================================

@dataclass
class TradeAttempt:
    from_team_id: str
    to_team_id: str
    rumored: bool = False
    respected_clause: bool = True
    role_change: float = 0.0   # [-1,1] negative = reduced role, positive = bigger role
    destination_team: Optional[TeamProfile] = None

@dataclass
class TradeReaction:
    pre_trade_morale_delta: float
    post_trade_morale_delta: float
    toxicity_risk: float
    media_outburst_risk: float
    early_retirement_seed_delta: float
    notes: List[str] = field(default_factory=list)

def evaluate_trade_reaction(
    rng: random.Random,
    player: PlayerProfile,
    current_team: TeamProfile,
    contract: Contract,
    attempt: TradeAttempt
) -> TradeReaction:
    """
    Handles both pre-trade rumor and post-trade reaction.
    """
    notes: List[str] = []
    p = player.personality
    c = player.career

    clause = contract.clauses
    has_block = clause.clause_type in (ClauseType.NTC, ClauseType.NMC)
    has_list = clause.clause_type == ClauseType.MNTC

    pre = 0.0
    post = 0.0

    # Pre-trade: rumors
    if attempt.rumored:
        rumble = 0.02 + 0.06 * (1.0 - p.stability_need) + 0.05 * p.ego
        pre -= rumble
        notes.append("trade rumors create anxiety / ego heat")
        # if player values stability, rumors hit more
        pre -= 0.04 * p.stability_need

    # Clause respect
    if has_block and not attempt.respected_clause:
        post -= 0.35
        notes.append("clause violated: severe betrayal")
        player.memory.betrayal_count[current_team.team_id] = player.memory.betrayal_count.get(current_team.team_id, 0) + 1
        player.memory.relationship[current_team.team_id] = clamp(player.memory.relationship.get(current_team.team_id, 0.0) - 0.35, -1.0, 1.0)
    elif has_list and not attempt.respected_clause:
        post -= 0.18
        notes.append("modified clause ignored: betrayal")
        player.memory.betrayal_count[current_team.team_id] = player.memory.betrayal_count.get(current_team.team_id, 0) + 1
        player.memory.relationship[current_team.team_id] = clamp(player.memory.relationship.get(current_team.team_id, 0.0) - 0.20, -1.0, 1.0)

    # Destination fit
    dest = attempt.destination_team
    if dest is not None:
        market_fit = _market_compatibility(player, dest)
        team_fit, _ = _team_fit_score(player, dest)
        dest_fit = weighted_sum([(market_fit, 0.45), (team_fit, 0.55)])
        post += (dest_fit - 0.5) * 0.25
        if dest_fit < 0.40:
            notes.append("destination fit is poor (market/team)")

    # Role change
    post += 0.18 * clamp(attempt.role_change, -1.0, 1.0)
    if attempt.role_change < -0.4:
        notes.append("role reduced; resentment risk")
    if attempt.role_change > 0.4:
        notes.append("bigger role; motivation spike")

    # Emotional memory impact
    rel = player.memory.relationship.get(current_team.team_id, 0.0)
    post += 0.10 * rel

    # Toxicity + media risks
    toxicity = clamp(0.12 + 0.25 * (1.0 - p.coachability if hasattr(p, "coachability") else 0.5) + 0.25 * p.volatility + 0.20 * (0.5 - rel), 0.0, 1.0)
    media = clamp(0.10 + 0.25 * p.media_comfort + 0.25 * p.ego + 0.10 * (1.0 - p.patience), 0.0, 1.0)

    # Early retirement seed increase (rare but meaningful)
    early_retire_delta = 0.0
    if not attempt.respected_clause:
        early_retire_delta += 0.08
    early_retire_delta += 0.02 * c.wear_and_tear
    early_retire_delta += 0.04 * p.family_priority * (0.4 + c.security_anxiety)

    return TradeReaction(
        pre_trade_morale_delta=float(pre),
        post_trade_morale_delta=float(post),
        toxicity_risk=float(clamp(toxicity, 0.0, 1.0)),
        media_outburst_risk=float(clamp(media, 0.0, 1.0)),
        early_retirement_seed_delta=float(clamp(early_retire_delta, 0.0, 0.25)),
        notes=notes
    )


# ============================================================
# Re-signing Logic (distinct from UFA)
# ============================================================

@dataclass
class ResignDecision:
    wants_to_resign: bool
    desire_score: float
    reasons: List[str] = field(default_factory=list)

def decide_resign(
    rng: random.Random,
    player: PlayerProfile,
    team: TeamProfile,
    agent: AgentProfile,
) -> ResignDecision:
    """
    Re-signing is not UFA logic:
      - was the player respected?
      - promises kept?
      - ice time vs expectations
      - trajectory honesty
      - trade rumors endured
      - previous discounts taken
    """
    reasons: List[str] = []
    p = player.personality
    c = player.career
    mem = player.memory

    rel = mem.relationship.get(team.team_id, 0.0)
    kept = mem.promise_kept_ratio.get(team.team_id, 0.5)
    lowballs = mem.lowball_count.get(team.team_id, 0)
    betrayals = mem.betrayal_count.get(team.team_id, 0)

    trajectory = _team_competitive_trajectory(team)
    stability = team.state.stability

    respect = clamp(c.offer_respect, 0.0, 1.0)
    ice = clamp(c.ice_time_satisfaction, 0.0, 1.0)

    score = weighted_sum([
        (0.5 + 0.5 * rel, 0.22 + 0.20 * p.loyalty),
        (kept, 0.18 + 0.20 * p.stability_need),
        (respect, 0.18 + 0.10 * (1.0 - p.ego)),
        (ice, 0.12),
        (trajectory, 0.20 + 0.20 * p.ambition + 0.20 * p.legacy_drive),
        (stability, 0.10 + 0.15 * p.stability_need),
    ])

    score -= 0.05 * lowballs
    score -= 0.12 * betrayals

    score += (rng.random() - 0.5) * 0.05  # small human noise
    score = clamp(score, 0.0, 1.0)

    # Threshold: loyal players need less; ego/ambition need more
    threshold = clamp(0.60 - 0.08 * (p.loyalty - 0.5) + 0.06 * (p.ego - 0.5) + 0.06 * (p.ambition - 0.5), 0.45, 0.78)

    if lowballs >= 2:
        reasons.append("multiple lowball negotiations left a bad taste")
    if betrayals >= 1:
        reasons.append("betrayal history makes re-signing difficult")
    if kept < 0.45:
        reasons.append("team hasn't kept promises")
    if trajectory > 0.65:
        reasons.append("team feels close to a real run")
    if stability < 0.35:
        reasons.append("org instability is a red flag")

    wants = score >= threshold
    if wants and p.loyalty > 0.7 and rel > 0.2:
        reasons.append("loyalty + relationship tilt toward staying")
    if not wants and p.ambition > 0.65 and trajectory < 0.5:
        reasons.append("ambition wants a better situation")

    return ResignDecision(wants_to_resign=wants, desire_score=score, reasons=reasons)


# ============================================================
# Convenience: build profiles from dicts
# ============================================================

def team_from_dict(d: Dict[str, Any]) -> TeamProfile:
    # shallow+safe conversion; expects nested dicts similar to your printout
    market = d.get("market", d.get("MARKET", {}))
    ownership = d.get("ownership", d.get("OWNERSHIP", {}))
    rep = d.get("reputation", d.get("REPUTATION", {}))
    phil = d.get("philosophy", d.get("ORG_PHILOSOPHY", {}))
    state = d.get("state", d.get("DYNAMIC_STATE", {}))
    roster = d.get("roster", d.get("ROSTER_QUALITY_PROXY", {}))

    return TeamProfile(
        team_id=str(d.get("team_id", d.get("id", d.get("name", "team")))),
        name=str(d.get("name", "Team")),
        archetype=str(d.get("archetype", "normal")),
        status=str(d.get("status", "bubble")),
        market=MarketProfile(**_pick_keys(market, MarketProfile)),
        ownership=OwnershipProfile(**_pick_keys(ownership, OwnershipProfile)),
        reputation=ReputationProfile(**_pick_keys(rep, ReputationProfile)),
        philosophy=OrgPhilosophy(**_pick_keys(phil, OrgPhilosophy)),
        state=TeamDynamicState(**_pick_keys(state, TeamDynamicState)),
        roster=TeamRosterProxy(**_pick_keys(roster, TeamRosterProxy)),
        cap_total=float(d.get("cap_total", 88_000_000.0)),
        cap_space=float(d.get("cap_space", 10_000_000.0)),
        cap_projection_growth=float(d.get("cap_projection_growth", 0.05)),
        front_office_chaos=float(d.get("front_office_chaos", 0.5)),
    )

def player_from_dict(d: Dict[str, Any]) -> PlayerProfile:
    pers = d.get("personality", d.get("PERSONALITY_TRAITS", {}))
    career = d.get("career", {})
    # allow your state dump keys
    career = {
        **career,
        "age": d.get("age", d.get("Age", d.get("AGE", career.get("age", 19)))),
        "ovr": d.get("ovr", d.get("OVR", career.get("ovr", 0.5))),
        "position": d.get("position", d.get("Position", career.get("position", "C"))),
        "shoots": d.get("shoots", d.get("Shoots", career.get("shoots", "L"))),
        "wear_and_tear": d.get("wear_and_tear", d.get("Wear & Tear", career.get("wear_and_tear", 0.0))),
        "chronic_injuries": d.get("chronic_injuries", d.get("Chronic Injuries", career.get("chronic_injuries", 0))),
        "ice_time_satisfaction": d.get("ice_time_satisfaction", career.get("ice_time_satisfaction", 0.5)),
        "role_mismatch": d.get("role_mismatch", career.get("role_mismatch", 0.0)),
        "legacy_pressure": d.get("legacy_pressure", career.get("legacy_pressure", 0.0)),
        "identity_instability": d.get("identity_instability", career.get("identity_instability", 0.0)),
        "emotional_fatigue": d.get("emotional_fatigue", career.get("emotional_fatigue", 0.0)),
        "security_anxiety": d.get("security_anxiety", career.get("security_anxiety", 0.0)),
        "ufa_pressure": d.get("ufa_pressure", career.get("ufa_pressure", 0.0)),
        "offer_respect": d.get("offer_respect", career.get("offer_respect", 0.5)),
    }

    mem = d.get("memory", {})
    return PlayerProfile(
        player_id=str(d.get("player_id", d.get("id", d.get("name", "player")))),
        name=str(d.get("name", "Player")),
        current_team_id=str(d.get("current_team_id", d.get("team_id", d.get("Team", "")))),
        personality=PlayerPersonality(**_pick_keys(pers, PlayerPersonality)),
        career=PlayerCareerState(**_pick_keys(career, PlayerCareerState)),
        memory=PlayerMemory(**_pick_keys(mem, PlayerMemory)),
    )

def _pick_keys(src: Dict[str, Any], cls: Any) -> Dict[str, Any]:
    # picks fields present on dataclass
    out: Dict[str, Any] = {}
    for f in cls.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        if f in src:
            out[f] = src[f]
    return out


# ============================================================
# Minimal demo (safe to delete)
# ============================================================

if __name__ == "__main__":
    rng = random.Random(42)

    team = TeamProfile(
        team_id="OTT",
        name="Ottawa Red Wings",
        archetype="chaotic",
        status="bubble",
        market=MarketProfile(market_size="small", media_pressure=0.50, fan_expectations=0.54, tax_advantage=0.45, weather_quality=0.51, travel_burden=0.62),
        ownership=OwnershipProfile(patience=0.41, ambition=0.65, budget_willingness=0.57, meddling=0.77),
        reputation=ReputationProfile(league_reputation=0.47, player_reputation=0.46, management_reputation=0.41, development_reputation=0.48),
        philosophy=OrgPhilosophy(development_quality=0.35, prospect_patience=0.22, risk_tolerance=0.90),
        state=TeamDynamicState(competitive_score=0.50, team_morale=0.50, org_pressure=0.50, stability=0.50),
        roster=TeamRosterProxy(star_count=0, core_count=5, depth_quality=0.47),
        cap_total=88_000_000.0,
        cap_space=9_500_000.0,
        cap_projection_growth=0.05,
    )

    player = PlayerProfile(
        player_id="P6138",
        name="SimPlayer_6138",
        current_team_id="OTT",
        personality=PlayerPersonality(
            loyalty=0.168, ambition=0.354, money_focus=0.059, family_priority=0.029,
            legacy_drive=0.080, ego=0.495, patience=0.50, risk_tolerance=0.50,
            stability_need=0.50, market_comfort=0.50, media_comfort=0.60,
            work_ethic=0.63, mental_toughness=0.62, volatility=0.65
        ),
        career=PlayerCareerState(age=27, ovr=0.62, ice_time_satisfaction=0.55, ufa_pressure=1.0, offer_respect=0.50),
        memory=PlayerMemory(drafted_by_team_id="OTT", developed_by_team_id="OTT")
    )

    agent = AgentProfile(agent_id="A1", name="Shark Agent", aggression=0.65, loyalty_to_player=0.75, league_influence=0.55, media_leak_tendency=0.35, risk_tolerance=0.55)

    league = {
        "cap": 88_000_000.0,
        "cap_growth": 0.05,
        "expected_aav": 5_500_000.0,  # your external valuation model can feed this
    }

    result = negotiate_contract(
        rng=rng,
        player=player,
        team=team,
        agent=agent,
        league=league,
        signing_year=2026,
        context_kind=ContractContextKind.UFA,
        max_weeks=14
    )

    print("Outcome:", result.outcome.value)
    if result.contract:
        print("Signed AAV:", result.contract.salary_aav, "Term:", result.contract.term_years, "Clause:", result.contract.clauses.clause_type.value)
        print("Cap%:", f"{result.contract.cap_percent_at_signing*100:.2f}%")
    else:
        print("No contract.")
    print("Notes:", result.notes[:5])
