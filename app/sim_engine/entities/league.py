# backend/app/sim_engine/entities/league.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math
import random


# ============================================================
# Helpers
# ============================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t, 0.0, 1.0)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def weighted_choice(rng: random.Random, items: List[Tuple[Any, float]]) -> Any:
    total = sum(max(0.0, w) for _, w in items)
    if total <= 0:
        return items[-1][0]
    r = rng.random() * total
    acc = 0.0
    for v, w in items:
        acc += max(0.0, w)
        if r <= acc:
            return v
    return items[-1][0]


# ============================================================
# Enums / Types
# ============================================================

class EraType(str, Enum):
    GRIND_LOW_SCORING = "grind_low_scoring"
    SPEED_AND_SKILL = "speed_and_skill"
    GOALIE_DOMINANCE = "goalie_dominance"
    CAP_CRUNCH = "cap_crunch"
    EXPANSION_DILUTION = "expansion_dilution"
    OFFENSE_BOOM = "offense_boom"


class NarrativeType(str, Enum):
    SUPERTEAMS_BACKLASH = "superteams_backlash"
    YOUTH_MOVEMENT = "youth_movement"
    GOALTENDING_CRISIS = "goaltending_crisis"
    CANADIAN_MARKET_PRESSURE = "canadian_market_pressure"
    REF_CRACKDOWN = "ref_crackdown"
    PARITY_PRIDE = "parity_pride"
    DYNASTY_ALERT = "dynasty_alert"
    LEAGUE_STABILITY = "league_stability"
    LEAGUE_TURMOIL = "league_turmoil"
    SMALL_MARKETS_SQUEEZED = "small_markets_squeezed"


class ShockType(str, Enum):
    LOCKOUT = "lockout"
    PANDEMIC_LITE = "pandemic_lite"
    SCANDAL = "scandal"
    CAP_FREEZE = "cap_freeze"
    LOTTERY_CONTROVERSY = "lottery_controversy"
    REF_INVESTIGATION = "ref_investigation"
    TV_DEAL_WINDFALL = "tv_deal_windfall"
    RECESSION = "recession"


# ============================================================
# Core Data Models
# ============================================================

@dataclass
class LeagueIdentity:
    league_name: str = "National Hockey League"
    abbreviation: str = "NHL"
    founding_year: int = 1917

    current_season: int = 2026  # start year of season
    max_teams: int = 36
    min_teams: int = 28

    conferences: Tuple[str, str] = ("Eastern", "Western")
    divisions: Tuple[str, ...] = ("Atlantic", "Metropolitan", "Central", "Pacific")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleEnvironment:
    """
    Soft levers. Do NOT enforce outcomes. These modify probabilities elsewhere.
    """
    officiating_strictness: float = 0.50   # higher = more penalties / less clutch-and-grab
    scoring_emphasis: float = 0.50         # higher = more offense-friendly environment
    goalie_protection: float = 0.50        # higher = fewer crease battles, less contact
    physicality_tolerance: float = 0.50    # higher = heavier hits allowed, more board battles

    def drift(self, rng: random.Random, intensity: float = 0.02) -> None:
        # slow drift, tiny random moves
        self.officiating_strictness = clamp(self.officiating_strictness + rng.uniform(-intensity, intensity))
        self.scoring_emphasis = clamp(self.scoring_emphasis + rng.uniform(-intensity, intensity))
        self.goalie_protection = clamp(self.goalie_protection + rng.uniform(-intensity, intensity))
        self.physicality_tolerance = clamp(self.physicality_tolerance + rng.uniform(-intensity, intensity))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EconomicEnvironment:
    """
    League cap philosophy + revenue sharing climate.
    """
    salary_cap: float = 88.0
    cap_floor: float = 65.0

    cap_growth_rate: float = 0.045     # baseline, then modulated by health/shocks
    cap_volatility: float = 0.20       # 0=stable era, 1=wild swings
    inflation_rate: float = 0.025      # simulation inflation; can drift

    revenue_sharing_strength: float = 0.55   # higher keeps small markets alive
    luxury_tax_pressure: float = 0.15        # higher punishes spenders (softly)
    owner_collusion_risk: float = 0.02       # rare but spicy (nudges negotiation climate)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompetitiveBalance:
    """
    Parity is an index describing how concentrated success is.
    0.0 -> superteam/dynasty league
    1.0 -> chaos/parity league
    """
    parity_index: float = 0.55

    # Rolling history inputs (league-level aggregates)
    cup_concentration: float = 0.50          # higher means same teams winning often
    playoff_concentration: float = 0.50      # higher means same teams making playoffs often
    superstar_clustering: float = 0.50       # higher means stars concentrated in few markets
    lottery_controversy_heat: float = 0.10   # rising pressure for reform

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeagueHealth:
    """
    Composite stability metric 0.0–1.0
    """
    health_score: float = 0.65

    avg_profitability: float = 0.60
    attendance_trend: float = 0.55
    media_sentiment: float = 0.60
    competitive_balance: float = 0.55
    star_power: float = 0.60
    scandal_frequency: float = 0.15  # higher = worse
    small_market_share: float = 0.40 # higher = more small markets

    def recompute(self) -> None:
        # Scandals & too many small markets reduce stability.
        positives = (
            0.20 * self.avg_profitability +
            0.15 * self.attendance_trend +
            0.15 * self.media_sentiment +
            0.20 * self.competitive_balance +
            0.20 * self.star_power
        )
        negatives = (
            0.06 * self.scandal_frequency +
            0.04 * self.small_market_share
        )
        self.health_score = clamp(positives - negatives)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EraDefinition:
    era_type: EraType
    name: str
    description: str

    # Multipliers / nudges (not outcomes)
    scoring_rate_mod: float = 1.0
    contract_inflation_mod: float = 1.0
    draft_variance_mod: float = 1.0
    injury_rate_mod: float = 1.0
    career_length_mod: float = 1.0

    # How the league "values" certain archetypes (soft biases)
    value_speed_skill: float = 0.50
    value_size_grit: float = 0.50
    value_goalie_elite: float = 0.50
    value_two_way: float = 0.50

    # How stable the era is (higher = slower to change)
    stability: float = 0.70

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["era_type"] = self.era_type.value
        return d


@dataclass
class EraState:
    """
    Eras overlap and fade in/out.
    """
    active_era: EraType = EraType.SPEED_AND_SKILL
    intensity: float = 0.55  # how strongly the era flavor applies (0..1)
    years_in_era: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_era": self.active_era.value,
            "intensity": self.intensity,
            "years_in_era": self.years_in_era,
        }


@dataclass
class ShockEvent:
    shock_type: ShockType
    severity: float                 # 0..1
    duration_years: int
    years_remaining: int
    started_season: int

    # Long-term scars: persistent small debuffs/buffs that fade slowly
    scar_econ: float = 0.0          # affects cap growth / volatility
    scar_sentiment: float = 0.0     # affects media sentiment
    scar_trust: float = 0.0         # affects parity reform pressure, narrative

    def is_active(self) -> bool:
        return self.years_remaining > 0

    def tick(self) -> None:
        if self.years_remaining > 0:
            self.years_remaining -= 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shock_type": self.shock_type.value,
            "severity": self.severity,
            "duration_years": self.duration_years,
            "years_remaining": self.years_remaining,
            "started_season": self.started_season,
            "scar_econ": self.scar_econ,
            "scar_sentiment": self.scar_sentiment,
            "scar_trust": self.scar_trust,
        }


@dataclass
class LeagueForecast:
    season: int

    # Contender structure (tiers, not standings)
    expected_contender_count: int
    expected_tank_count: int

    # Chaos / volatility
    chaos_index: float
    bubble_chaos_probability: float
    cup_favorite_volatility: float

    # League-wide behaviors
    expected_trade_deadline_activity: float
    expected_coaching_turnover: float
    expected_scandals: float

    # Player-level climate (nudges to other engines)
    star_emergence_probability: float
    star_decline_probability: float

    # Meta
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LongTermForecast:
    start_season: int
    horizon_years: int

    dynasty_likelihood: float
    talent_pipeline_strength: float
    aging_league_concern: float
    expansion_window_score: float
    market_shift_pressure: float

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeagueMemory:
    """
    Historical memory & scars that persist for decades.
    This is NOT a full stats archive; it’s narrative/historical context.
    """
    legendary_seasons: List[Dict[str, Any]] = field(default_factory=list)
    dynasties: List[Dict[str, Any]] = field(default_factory=list)
    cursed_franchises: Dict[str, int] = field(default_factory=dict)  # team_id -> drought years
    long_droughts: Dict[str, int] = field(default_factory=dict)      # team_id -> years since cup

    shock_history: List[Dict[str, Any]] = field(default_factory=list)
    rule_change_history: List[Dict[str, Any]] = field(default_factory=list)
    expansion_history: List[Dict[str, Any]] = field(default_factory=list)
    relocation_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# League
# ============================================================

class League:
    """
    league.py = the ecosystem.

    - Holds persistent league state
    - Generates forecasts
    - Nudges probabilities (never micromanages team/player outcomes)
    - Controls league-wide randomness & shock events
    - Tracks eras & long-term trends
    - Feeds context into other systems (contracts, morale, trades, retirements, etc.)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        identity: Optional[LeagueIdentity] = None,
        economics: Optional[EconomicEnvironment] = None,
        rules: Optional[RuleEnvironment] = None,
        balance: Optional[CompetitiveBalance] = None,
        health: Optional[LeagueHealth] = None,
    ) -> None:
        self.rng = random.Random(seed)

        self.identity: LeagueIdentity = identity or LeagueIdentity()
        self.economics: EconomicEnvironment = economics or EconomicEnvironment()
        self.rules: RuleEnvironment = rules or RuleEnvironment()
        self.balance: CompetitiveBalance = balance or CompetitiveBalance()
        self.health: LeagueHealth = health or LeagueHealth()

        self.era_defs: Dict[EraType, EraDefinition] = self._default_era_definitions()
        self.era_state: EraState = EraState()

        self.active_shocks: List[ShockEvent] = []
        self.narratives: Dict[NarrativeType, float] = {n: 0.0 for n in NarrativeType}

        self.memory: LeagueMemory = LeagueMemory()

        # Expansion/relocation strictness is political and can drift
        self.relocation_approval_strictness: float = 0.65  # higher = harder to relocate
        self.expansion_gatekeeping: float = 0.60           # higher = league resists expansion

        # Derived, cached per season
        self.last_season_forecast: Optional[LeagueForecast] = None
        self.last_long_term_forecast: Optional[LongTermForecast] = None

    # ------------------------------------------------------------
    # Era Definitions
    # ------------------------------------------------------------

    def _default_era_definitions(self) -> Dict[EraType, EraDefinition]:
        return {
            EraType.GRIND_LOW_SCORING: EraDefinition(
                era_type=EraType.GRIND_LOW_SCORING,
                name="Grind & Clutch Era",
                description="Lower scoring, heavier play, defense-first identities dominate.",
                scoring_rate_mod=0.88,
                contract_inflation_mod=0.95,
                draft_variance_mod=1.10,
                injury_rate_mod=1.08,
                career_length_mod=0.97,
                value_speed_skill=0.40,
                value_size_grit=0.70,
                value_goalie_elite=0.55,
                value_two_way=0.68,
                stability=0.75,
            ),
            EraType.SPEED_AND_SKILL: EraDefinition(
                era_type=EraType.SPEED_AND_SKILL,
                name="Speed & Skill Boom",
                description="Skill wins. Pace is high. Teams chase speed, puck movement, transition.",
                scoring_rate_mod=1.05,
                contract_inflation_mod=1.05,
                draft_variance_mod=0.98,
                injury_rate_mod=0.96,
                career_length_mod=1.02,
                value_speed_skill=0.80,
                value_size_grit=0.40,
                value_goalie_elite=0.50,
                value_two_way=0.55,
                stability=0.70,
            ),
            EraType.GOALIE_DOMINANCE: EraDefinition(
                era_type=EraType.GOALIE_DOMINANCE,
                name="Goalie Dominance Cycle",
                description="Elite goaltending swings seasons. Scoring dries up; teams panic-buy goalies.",
                scoring_rate_mod=0.92,
                contract_inflation_mod=1.00,
                draft_variance_mod=1.05,
                injury_rate_mod=0.98,
                career_length_mod=1.00,
                value_speed_skill=0.50,
                value_size_grit=0.50,
                value_goalie_elite=0.85,
                value_two_way=0.62,
                stability=0.68,
            ),
            EraType.CAP_CRUNCH: EraDefinition(
                era_type=EraType.CAP_CRUNCH,
                name="Cap Crunch Era",
                description="Flat cap vibes. Term, flexibility, and internal budgets matter more than talent.",
                scoring_rate_mod=1.00,
                contract_inflation_mod=0.92,
                draft_variance_mod=1.12,
                injury_rate_mod=1.00,
                career_length_mod=0.99,
                value_speed_skill=0.55,
                value_size_grit=0.52,
                value_goalie_elite=0.55,
                value_two_way=0.72,
                stability=0.80,
            ),
            EraType.EXPANSION_DILUTION: EraDefinition(
                era_type=EraType.EXPANSION_DILUTION,
                name="Expansion Dilution Phase",
                description="More teams, thinner depth. Stars shine brighter; bottom rosters get uglier.",
                scoring_rate_mod=1.03,
                contract_inflation_mod=1.08,
                draft_variance_mod=1.18,
                injury_rate_mod=1.02,
                career_length_mod=0.98,
                value_speed_skill=0.62,
                value_size_grit=0.45,
                value_goalie_elite=0.55,
                value_two_way=0.60,
                stability=0.60,
            ),
            EraType.OFFENSE_BOOM: EraDefinition(
                era_type=EraType.OFFENSE_BOOM,
                name="Offense Boom",
                description="Rule emphasis + skill concentration pushes scoring. Defensemen become gold.",
                scoring_rate_mod=1.12,
                contract_inflation_mod=1.12,
                draft_variance_mod=0.95,
                injury_rate_mod=0.95,
                career_length_mod=1.03,
                value_speed_skill=0.78,
                value_size_grit=0.35,
                value_goalie_elite=0.45,
                value_two_way=0.55,
                stability=0.62,
            ),
        }

    def get_active_era_definition(self) -> EraDefinition:
        return self.era_defs[self.era_state.active_era]

    # ------------------------------------------------------------
    # League Context API (what other systems should query)
    # ------------------------------------------------------------

    def get_league_context(self) -> Dict[str, Any]:
        """
        This is what other engines (contracts, morale, trades, retirements, draft)
        should read from. They should not override league state.
        """
        era_def = self.get_active_era_definition()
        shocks = [s.to_dict() for s in self.active_shocks if s.is_active()]

        # Aggregate shock impacts as soft nudges.
        shock_econ = sum(s.scar_econ * (0.3 + 0.7 * s.severity) for s in self.active_shocks if s.is_active())
        shock_sent = sum(s.scar_sentiment * (0.3 + 0.7 * s.severity) for s in self.active_shocks if s.is_active())
        shock_trust = sum(s.scar_trust * (0.3 + 0.7 * s.severity) for s in self.active_shocks if s.is_active())

        ctx = {
            "identity": self.identity.to_dict(),
            "economics": self.economics.to_dict(),
            "rules": self.rules.to_dict(),
            "balance": self.balance.to_dict(),
            "health": self.health.to_dict(),
            "era": {"state": self.era_state.to_dict(), "definition": era_def.to_dict()},
            "narratives": {k.value: v for k, v in self.narratives.items()},
            "active_shocks": shocks,

            # Derived global nudges:
            "nudges": {
                "scoring_rate_mod": lerp(1.0, era_def.scoring_rate_mod, self.era_state.intensity),
                "contract_inflation_mod": lerp(1.0, era_def.contract_inflation_mod, self.era_state.intensity),
                "draft_variance_mod": lerp(1.0, era_def.draft_variance_mod, self.era_state.intensity),
                "injury_rate_mod": lerp(1.0, era_def.injury_rate_mod, self.era_state.intensity),
                "career_length_mod": lerp(1.0, era_def.career_length_mod, self.era_state.intensity),

                "shock_econ": shock_econ,
                "shock_sentiment": shock_sent,
                "shock_trust": shock_trust,
            },

            # Forecast caches (optional)
            "last_season_forecast": self.last_season_forecast.to_dict() if self.last_season_forecast else None,
            "last_long_term_forecast": self.last_long_term_forecast.to_dict() if self.last_long_term_forecast else None,

            "memory": self.memory.to_dict(),
        }
        return ctx

    # ------------------------------------------------------------
    # Inputs (from engine) to update league macro signals
    # ------------------------------------------------------------

    def update_from_team_snapshots(self, team_snapshots: List[Dict[str, Any]]) -> None:
        """
        Optional: feed macro aggregates from team-level results.
        This is NOT micromanagement; it updates league health/balance inputs.

        Expected fields per team snapshot (best-effort, missing ok):
        - profitability (0..1)
        - attendance_trend (0..1)
        - market_size: "small"|"mid"|"large" (or numeric)
        - media_pressure (0..1)
        - contender_status: "contender"|"bubble"|"tank"
        - superstar_count (int)
        - made_playoffs (bool)
        - won_cup (bool)
        """
        if not team_snapshots:
            return

        profits = []
        attend = []
        media = []
        stars = []
        small_markets = 0

        playoff_flags = []
        cup_flags = []
        contender_flags = []
        tank_flags = []

        for t in team_snapshots:
            if "profitability" in t:
                profits.append(float(t["profitability"]))
            if "attendance_trend" in t:
                attend.append(float(t["attendance_trend"]))
            if "media_pressure" in t:
                media.append(float(t["media_pressure"]))
            if "superstar_count" in t:
                stars.append(float(t["superstar_count"]))

            ms = t.get("market_size")
            if isinstance(ms, str) and ms.lower() == "small":
                small_markets += 1

            if "made_playoffs" in t:
                playoff_flags.append(bool(t["made_playoffs"]))
            if "won_cup" in t:
                cup_flags.append(bool(t["won_cup"]))

            cs = (t.get("contender_status") or "").lower()
            contender_flags.append(1 if cs == "contender" else 0)
            tank_flags.append(1 if cs == "tank" else 0)

        # Update health components
        if profits:
            self.health.avg_profitability = clamp(sum(profits) / len(profits))
        if attend:
            self.health.attendance_trend = clamp(sum(attend) / len(attend))

        # media sentiment inversely related to pressure (but not perfectly)
        if media:
            avg_pressure = clamp(sum(media) / len(media))
            self.health.media_sentiment = clamp(1.0 - 0.65 * avg_pressure)

        # star power scaled from superstar counts; saturating
        if stars:
            avg_stars = sum(stars) / len(stars)
            self.health.star_power = clamp(1.0 - math.exp(-avg_stars / 1.75))

        # small market share
        self.health.small_market_share = clamp(small_markets / max(1, len(team_snapshots)))

        # Competitive balance proxy updates
        if playoff_flags:
            playoff_rate = sum(1 for x in playoff_flags if x) / len(playoff_flags)
            # if too stable (same teams always playoffs), playoff_concentration rises
            # here we can’t know "same teams" without history; this is a placeholder drift.
            self.balance.playoff_concentration = clamp(lerp(self.balance.playoff_concentration, 0.5 + (0.5 - playoff_rate) * 0.2, 0.15))

        if cup_flags and any(cup_flags):
            # winning concentration rises slightly if repeat champs are common; without history, mild drift
            self.balance.cup_concentration = clamp(self.balance.cup_concentration + 0.01)

        # Derive parity index from concentrations (higher concentration -> lower parity)
        conc = (
            0.45 * self.balance.cup_concentration +
            0.35 * self.balance.playoff_concentration +
            0.20 * self.balance.superstar_clustering
        )
        self.balance.parity_index = clamp(1.0 - conc)

        # feed competitive balance into health
        self.health.competitive_balance = clamp(self.balance.parity_index)

        # recompute overall health
        self.health.recompute()

    # ------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------

    def generate_forecast(self) -> LeagueForecast:
        """
        Generates a season-level forecast that biases simulation.
        It should sometimes be wrong.
        """
        season = self.identity.current_season
        era_def = self.get_active_era_definition()

        # Chaos is higher when parity is high, rules are in flux, and health is middling (instability).
        parity = self.balance.parity_index
        health = self.health.health_score

        # narrative heat can also raise chaos
        turmoil = self.narratives[NarrativeType.LEAGUE_TURMOIL]
        crackdown = self.narratives[NarrativeType.REF_CRACKDOWN]

        base_chaos = 0.35 + 0.35 * parity + 0.10 * turmoil + 0.05 * crackdown
        health_instability = 0.20 * (1.0 - abs(health - 0.60) / 0.60)  # peak around 0.6
        chaos_index = clamp(base_chaos + health_instability + self.rng.uniform(-0.08, 0.08))

        # contender count: fewer in superteam eras, more in parity eras
        expected_contenders = int(round(4 + 10 * parity + self.rng.uniform(-1.5, 1.5)))
        expected_contenders = max(3, min(14, expected_contenders))

        # tank count increases when cap crunch, poor health, or expansion dilution (depth gaps)
        tank_pressure = 0.25 * (1.0 - health) + 0.20 * (1.0 - self.economics.revenue_sharing_strength)
        if self.era_state.active_era in (EraType.CAP_CRUNCH, EraType.EXPANSION_DILUTION):
            tank_pressure += 0.15
        expected_tanks = int(round(3 + 8 * clamp(tank_pressure) + self.rng.uniform(-1.0, 1.0)))
        expected_tanks = max(2, min(12, expected_tanks))

        bubble_chaos_probability = clamp(0.35 + 0.45 * chaos_index + 0.10 * parity)
        cup_favorite_volatility = clamp(0.25 + 0.55 * chaos_index + 0.10 * (1.0 - health))

        # Trade deadline activity increases with volatility, cap uncertainty, and owner ambition climate
        cap_uncertainty = clamp(self.economics.cap_volatility + 0.5 * abs(self.economics.cap_growth_rate - 0.045))
        deadline_activity = clamp(0.30 + 0.45 * chaos_index + 0.25 * cap_uncertainty)

        # Coaching turnover increases with media pressure + turmoil + low health
        coaching_turnover = clamp(0.25 + 0.25 * (1.0 - health) + 0.25 * turmoil + self.rng.uniform(-0.05, 0.10))

        # Scandals increase with low health, weak sentiment, and certain shock scars
        scandal = clamp(0.08 + 0.20 * (1.0 - health) + 0.12 * (1.0 - self.health.media_sentiment) + self.rng.uniform(-0.03, 0.07))

        # Star emergence tends to rise in offense eras and parity eras (more opportunity),
        # but also when league wants stars (low star_power).
        offense_push = clamp(self.rules.scoring_emphasis * 0.6 + (era_def.scoring_rate_mod - 1.0) * 0.9)
        star_need = clamp(1.0 - self.health.star_power)
        star_emerge = clamp(0.10 + 0.20 * offense_push + 0.12 * parity + 0.18 * star_need + self.rng.uniform(-0.03, 0.05))

        # Star decline increases with high physicality + low goalie protection + injury era
        brutality = clamp(0.40 * self.rules.physicality_tolerance + 0.25 * (1.0 - self.rules.goalie_protection))
        star_decline = clamp(0.10 + 0.18 * brutality + 0.10 * (era_def.injury_rate_mod - 1.0) + self.rng.uniform(-0.02, 0.05))

        notes: List[str] = []
        if chaos_index > 0.70:
            notes.append("High chaos climate: expect weird playoff brackets and surprise sellers/buyers.")
        if self.era_state.active_era == EraType.CAP_CRUNCH:
            notes.append("Cap-crunch behavior: rentals valued, term feared, flexibility worshiped.")
        if self.era_state.active_era in (EraType.OFFENSE_BOOM, EraType.SPEED_AND_SKILL):
            notes.append("Offense climate: scorers get paid and depth defense gets exposed.")
        if health < 0.45:
            notes.append("League health is shaky: owners tighten budgets, stability politics intensify.")

        forecast = LeagueForecast(
            season=season,
            expected_contender_count=expected_contenders,
            expected_tank_count=expected_tanks,
            chaos_index=chaos_index,
            bubble_chaos_probability=bubble_chaos_probability,
            cup_favorite_volatility=cup_favorite_volatility,
            expected_trade_deadline_activity=deadline_activity,
            expected_coaching_turnover=coaching_turnover,
            expected_scandals=scandal,
            star_emergence_probability=star_emerge,
            star_decline_probability=star_decline,
            notes=notes,
        )

        self.last_season_forecast = forecast
        return forecast

    def generate_long_term_forecast(self, horizon_years: int = 4) -> LongTermForecast:
        """
        3–5 year macro trend forecast. Used by owners/GMs/player long-term planning.
        """
        start = self.identity.current_season
        health = self.health.health_score
        parity = self.balance.parity_index
        era_def = self.get_active_era_definition()

        # dynasty likelihood rises when parity is low and superstar clustering is high
        dynasty = clamp(0.12 + 0.55 * (1.0 - parity) + 0.25 * self.balance.superstar_clustering + self.rng.uniform(-0.05, 0.05))

        # talent pipeline strength: inverse of draft variance + expansion dilution pressure
        dilution = 0.25 if self.era_state.active_era == EraType.EXPANSION_DILUTION else 0.0
        pipeline = clamp(0.55 - 0.20 * (era_def.draft_variance_mod - 1.0) - dilution + 0.10 * parity + self.rng.uniform(-0.05, 0.05))

        # aging league concerns: more in physical eras and high injury climates; also if stars are older (not tracked here)
        brutality = clamp(0.50 * self.rules.physicality_tolerance + 0.30 * (era_def.injury_rate_mod - 1.0))
        aging = clamp(0.20 + 0.35 * brutality + 0.20 * (1.0 - self.rules.goalie_protection) + self.rng.uniform(-0.05, 0.05))

        # expansion window score rises with health, TV windfalls, and small-market stability
        tv_windfall = 0.0
        for s in self.active_shocks:
            if s.is_active() and s.shock_type == ShockType.TV_DEAL_WINDFALL:
                tv_windfall += 0.25 * s.severity
        window = clamp(0.10 + 0.55 * health + 0.15 * self.economics.revenue_sharing_strength + tv_windfall - 0.20 * self.expansion_gatekeeping)

        # market shift pressure rises when revenue sharing is weak and small markets are common
        market_pressure = clamp(0.10 + 0.40 * (1.0 - self.economics.revenue_sharing_strength) + 0.25 * self.health.small_market_share + self.rng.uniform(-0.05, 0.05))

        notes: List[str] = []
        if window > 0.65:
            notes.append("Expansion window is opening: ownership groups + league confidence are aligning.")
        if market_pressure > 0.60:
            notes.append("Market pressure rising: weak markets face arena/ownership stress.")
        if dynasty > 0.55:
            notes.append("Dynasty risk: the league may drift toward backlash narratives and reform heat.")
        if health < 0.50:
            notes.append("Long-term stability concerns: owners may prioritize self-preservation over fairness.")

        lt = LongTermForecast(
            start_season=start,
            horizon_years=horizon_years,
            dynasty_likelihood=dynasty,
            talent_pipeline_strength=pipeline,
            aging_league_concern=aging,
            expansion_window_score=window,
            market_shift_pressure=market_pressure,
            notes=notes,
        )
        self.last_long_term_forecast = lt
        return lt

    # ------------------------------------------------------------
    # Economics update (cap philosophy)
    # ------------------------------------------------------------

    def apply_economic_update(self) -> None:
        """
        Updates cap, floor, growth expectations based on health, inflation, shocks, and volatility.
        """
        self._apply_shock_effects_economy()

        # Inflation can drift slowly
        self.economics.inflation_rate = clamp(self.economics.inflation_rate + self.rng.uniform(-0.003, 0.003), 0.0, 0.08)

        health = self.health.health_score
        era_def = self.get_active_era_definition()

        # Growth baseline: inflation + health-driven lift + era-driven modifier
        base = self.economics.inflation_rate + 0.015 + 0.035 * health
        base *= lerp(1.0, era_def.contract_inflation_mod, self.era_state.intensity)

        # Volatility introduces unpredictability; also increases when health is unstable
        instability = 1.0 - abs(health - 0.70) / 0.70
        vol = clamp(self.economics.cap_volatility + 0.15 * instability)

        # Random shock to growth (mean 0, scales with vol)
        growth_shock = self.rng.gauss(0.0, 0.012 + 0.020 * vol)
        new_growth = clamp(base + growth_shock, -0.05, 0.15)

        # Apply growth to cap unless frozen by shock (handled in shock effects)
        self.economics.cap_growth_rate = new_growth

        # Cap/floor update
        # Floor usually trails cap but can tighten in strong health climates
        cap_before = self.economics.salary_cap
        cap_after = max(30.0, cap_before * (1.0 + self.economics.cap_growth_rate))

        self.economics.salary_cap = cap_after

        floor_ratio = lerp(0.72, 0.78, health)
        self.economics.cap_floor = max(20.0, cap_after * floor_ratio)

        # revenue sharing is political: stronger when small markets are many or health is low
        rs_target = clamp(0.45 + 0.25 * self.health.small_market_share + 0.15 * (1.0 - health))
        self.economics.revenue_sharing_strength = lerp(self.economics.revenue_sharing_strength, rs_target, 0.15)

        # luxury tax pressure rises if parity is low (league tries to discourage superteams)
        lt_target = clamp(0.08 + 0.35 * (1.0 - self.balance.parity_index))
        self.economics.luxury_tax_pressure = lerp(self.economics.luxury_tax_pressure, lt_target, 0.12)

        # owner collusion risk is rare; edges up in low health & high pressure climates
        collusion_target = clamp(0.01 + 0.06 * (1.0 - health) + 0.02 * self.narratives[NarrativeType.LEAGUE_TURMOIL])
        self.economics.owner_collusion_risk = lerp(self.economics.owner_collusion_risk, collusion_target, 0.10)

    def _apply_shock_effects_economy(self) -> None:
        """
        Active shocks can freeze cap, alter volatility, or boost growth.
        """
        for s in self.active_shocks:
            if not s.is_active():
                continue

            if s.shock_type == ShockType.CAP_FREEZE:
                # Force near-zero cap growth while active
                self.economics.cap_growth_rate = min(self.economics.cap_growth_rate, 0.005 * (1.0 - 0.7 * s.severity))
                self.economics.cap_volatility = clamp(self.economics.cap_volatility + 0.10 * s.severity)

            elif s.shock_type == ShockType.RECESSION:
                self.economics.cap_growth_rate = min(self.economics.cap_growth_rate, -0.01 * s.severity)
                self.economics.cap_volatility = clamp(self.economics.cap_volatility + 0.15 * s.severity)
                self.economics.revenue_sharing_strength = clamp(self.economics.revenue_sharing_strength + 0.05 * s.severity)

            elif s.shock_type == ShockType.TV_DEAL_WINDFALL:
                self.economics.cap_growth_rate = max(self.economics.cap_growth_rate, 0.03 + 0.08 * s.severity)
                self.economics.cap_volatility = clamp(self.economics.cap_volatility + 0.05 * s.severity)

            elif s.shock_type == ShockType.LOCKOUT:
                # lockouts create uncertainty and a short-term confidence hit
                self.economics.cap_volatility = clamp(self.economics.cap_volatility + 0.10 * s.severity)

            elif s.shock_type == ShockType.PANDEMIC_LITE:
                self.economics.cap_volatility = clamp(self.economics.cap_volatility + 0.12 * s.severity)
                self.economics.cap_growth_rate = min(self.economics.cap_growth_rate, 0.01 * (1.0 - s.severity))

    # ------------------------------------------------------------
    # Rule changes (rare, controversial, political)
    # ------------------------------------------------------------

    def maybe_apply_rule_change(self) -> Optional[Dict[str, Any]]:
        """
        League may change rules when parity issues + health politics align.
        Rare. Controversial. Sometimes a dumb decision.
        """
        health = self.health.health_score
        parity = self.balance.parity_index
        scandal_heat = self.health.scandal_frequency
        turmoil = self.narratives[NarrativeType.LEAGUE_TURMOIL]
        crackdown = self.narratives[NarrativeType.REF_CRACKDOWN]

        # Rule change pressure rises if parity is low (dynasty league), or scandals spike,
        # or the league wants to "look like it’s doing something".
        pressure = (
            0.06 +
            0.20 * (1.0 - parity) +
            0.10 * scandal_heat +
            0.10 * turmoil +
            0.05 * crackdown
        )

        # Healthy leagues resist upheaval unless parity is ugly.
        if health > 0.75:
            pressure -= 0.06

        # absolute rarity gate
        chance = clamp(pressure * 0.35)
        if self.rng.random() > chance:
            return None

        # Pick a rule "direction"
        change_type = weighted_choice(self.rng, [
            ("officiating_crackdown", 0.35 + 0.40 * turmoil),
            ("scoring_push",          0.25 + 0.35 * (1.0 - self.rules.scoring_emphasis)),
            ("goalie_protection",     0.20 + 0.30 * (1.0 - self.rules.goalie_protection)),
            ("let_them_play",         0.20 + 0.30 * (1.0 - self.rules.physicality_tolerance)),
        ])

        delta = 0.03 + 0.10 * self.rng.random()
        if change_type == "officiating_crackdown":
            self.rules.officiating_strictness = clamp(self.rules.officiating_strictness + delta)
            self.rules.physicality_tolerance = clamp(self.rules.physicality_tolerance - 0.5 * delta)
            self.narratives[NarrativeType.REF_CRACKDOWN] = clamp(self.narratives[NarrativeType.REF_CRACKDOWN] + 0.20)

        elif change_type == "scoring_push":
            self.rules.scoring_emphasis = clamp(self.rules.scoring_emphasis + delta)
            self.narratives[NarrativeType.PARITY_PRIDE] = clamp(self.narratives[NarrativeType.PARITY_PRIDE] + 0.05)

        elif change_type == "goalie_protection":
            self.rules.goalie_protection = clamp(self.rules.goalie_protection + delta)
            self.rules.physicality_tolerance = clamp(self.rules.physicality_tolerance - 0.3 * delta)

        elif change_type == "let_them_play":
            self.rules.physicality_tolerance = clamp(self.rules.physicality_tolerance + delta)
            self.rules.officiating_strictness = clamp(self.rules.officiating_strictness - 0.4 * delta)

        record = {
            "season": self.identity.current_season,
            "type": change_type,
            "magnitude": delta,
            "rules": self.rules.to_dict(),
            "reason": {
                "health": health,
                "parity": parity,
                "scandal_heat": scandal_heat,
                "turmoil": turmoil,
            }
        }
        self.memory.rule_change_history.append(record)
        return record

    # ------------------------------------------------------------
    # Era shifts (slow, overlapping, sometimes snapped by shocks)
    # ------------------------------------------------------------

    def apply_era_shift(self) -> Optional[Dict[str, Any]]:
        """
        Slowly drifts era intensity; occasionally transitions.
        The league should feel different by decade.
        """
        current = self.get_active_era_definition()
        self.era_state.years_in_era += 1

        # intensity drifts slowly (eras fade, not flip)
        self.era_state.intensity = clamp(self.era_state.intensity + self.rng.uniform(-0.03, 0.03))

        # transition pressure builds when parity/health/rules mismatch current era "fit"
        parity = self.balance.parity_index
        health = self.health.health_score

        # Example: if scoring emphasis is high, offense eras become more likely.
        offense_fit = clamp(0.40 * self.rules.scoring_emphasis + 0.20 * (1.0 - self.rules.physicality_tolerance))
        defense_fit = clamp(0.35 * self.rules.physicality_tolerance + 0.25 * (1.0 - self.rules.scoring_emphasis))

        # if cap growth is low or volatility high -> cap crunch fit
        cap_fit = clamp(0.40 * (1.0 - clamp(self.economics.cap_growth_rate / 0.08)) + 0.25 * self.economics.cap_volatility + 0.15 * (1.0 - health))

        # expansion dilution fit if the league is expanding or planning to
        expansion_fit = 0.0
        if self.identity.max_teams >= 34:
            expansion_fit = 0.15
        if self.last_long_term_forecast and self.last_long_term_forecast.expansion_window_score > 0.65:
            expansion_fit = max(expansion_fit, 0.30)

        goalie_fit = clamp(0.35 * (1.0 - self.rules.scoring_emphasis) + 0.20 * self.rules.goalie_protection)

        # Transition chance: rare, increases if current era is old and intensity is weak.
        age_factor = clamp((self.era_state.years_in_era - 6) / 10.0)  # ramps after ~6 years
        weaken_factor = 1.0 - self.era_state.intensity
        base_transition = 0.02 + 0.05 * age_factor + 0.03 * weaken_factor

        # Shocks can snap eras
        snap = 0.0
        for s in self.active_shocks:
            if s.is_active() and s.shock_type in (ShockType.LOCKOUT, ShockType.TV_DEAL_WINDFALL, ShockType.RECESSION):
                snap += 0.03 + 0.05 * s.severity

        transition_chance = clamp(base_transition + snap, 0.0, 0.20)
        if self.rng.random() > transition_chance:
            return None

        # choose next era based on fits and politics (parity & league appetite)
        candidates: List[Tuple[EraType, float]] = [
            (EraType.OFFENSE_BOOM, 0.20 + 0.60 * offense_fit),
            (EraType.GRIND_LOW_SCORING, 0.18 + 0.55 * defense_fit),
            (EraType.CAP_CRUNCH, 0.20 + 0.70 * cap_fit),
            (EraType.EXPANSION_DILUTION, 0.10 + 0.60 * expansion_fit),
            (EraType.GOALIE_DOMINANCE, 0.18 + 0.55 * goalie_fit),
            (EraType.SPEED_AND_SKILL, 0.22 + 0.40 * parity),
        ]

        # Avoid staying the same unless it's extremely likely
        candidates = [(e, w * (0.25 if e == self.era_state.active_era else 1.0)) for e, w in candidates]

        new_era = weighted_choice(self.rng, candidates)
        prev = self.era_state.active_era

        self.era_state.active_era = new_era
        self.era_state.years_in_era = 0
        self.era_state.intensity = clamp(0.45 + 0.25 * self.rng.random())  # re-seed intensity

        record = {
            "season": self.identity.current_season,
            "from": prev.value,
            "to": new_era.value,
            "reason": {
                "offense_fit": offense_fit,
                "defense_fit": defense_fit,
                "cap_fit": cap_fit,
                "goalie_fit": goalie_fit,
                "expansion_fit": expansion_fit,
                "health": health,
                "parity": parity,
                "transition_chance": transition_chance,
            }
        }
        return record

    # ------------------------------------------------------------
    # Shocks (rare but impactful; league remembers)
    # ------------------------------------------------------------

    def check_shocks(self) -> Optional[ShockEvent]:
        """
        Roll once per season for a shock event.
        Shocks create scars that persist for decades (memory).
        """
        # baseline shock chance depends on league health and existing turmoil/scandal
        health = self.health.health_score
        turmoil = self.narratives[NarrativeType.LEAGUE_TURMOIL]
        scandal_heat = self.health.scandal_frequency

        base = 0.03 + 0.06 * (1.0 - health) + 0.05 * turmoil + 0.04 * scandal_heat
        # Cap it; shocks must stay rare.
        chance = clamp(base, 0.01, 0.14)

        # If a shock is already active, reduce chance of another huge shock.
        if any(s.is_active() for s in self.active_shocks):
            chance *= 0.55

        if self.rng.random() > chance:
            return None

        # Choose shock type. Political + economic climate influences weights.
        cap_stress = clamp(self.economics.cap_volatility + 0.5 * (0.045 - self.economics.cap_growth_rate))
        recession_weight = 0.10 + 0.35 * (1.0 - health) + 0.25 * cap_stress
        windfall_weight = 0.08 + 0.35 * health
        lockout_weight = 0.06 + 0.25 * turmoil + 0.15 * cap_stress
        scandal_weight = 0.10 + 0.35 * scandal_heat + 0.10 * (1.0 - self.health.media_sentiment)

        shock_type = weighted_choice(self.rng, [
            (ShockType.RECESSION, recession_weight),
            (ShockType.TV_DEAL_WINDFALL, windfall_weight),
            (ShockType.LOCKOUT, lockout_weight),
            (ShockType.SCANDAL, scandal_weight),
            (ShockType.CAP_FREEZE, 0.10 + 0.30 * cap_stress),
            (ShockType.LOTTERY_CONTROVERSY, 0.08 + 0.30 * self.balance.lottery_controversy_heat),
            (ShockType.REF_INVESTIGATION, 0.06 + 0.20 * turmoil),
            (ShockType.PANDEMIC_LITE, 0.03 + 0.10 * (1.0 - health)),
        ])

        severity = clamp(0.25 + 0.60 * self.rng.random())  # meaningful when it hits
        duration = 1 if severity < 0.55 else (2 if severity < 0.80 else 3)

        # Scars (long-term memory)
        scar_econ = 0.0
        scar_sent = 0.0
        scar_trust = 0.0

        if shock_type in (ShockType.RECESSION, ShockType.CAP_FREEZE, ShockType.LOCKOUT):
            scar_econ = -0.04 * severity
            scar_sent = -0.05 * severity
            scar_trust = -0.03 * severity
        elif shock_type == ShockType.TV_DEAL_WINDFALL:
            scar_econ = +0.05 * severity
            scar_sent = +0.03 * severity
        elif shock_type == ShockType.SCANDAL:
            scar_sent = -0.08 * severity
            scar_trust = -0.06 * severity
        elif shock_type == ShockType.LOTTERY_CONTROVERSY:
            scar_trust = -0.07 * severity
        elif shock_type == ShockType.REF_INVESTIGATION:
            scar_trust = -0.05 * severity
            scar_sent = -0.03 * severity
        elif shock_type == ShockType.PANDEMIC_LITE:
            scar_econ = -0.03 * severity
            scar_sent = -0.04 * severity

        shock = ShockEvent(
            shock_type=shock_type,
            severity=severity,
            duration_years=duration,
            years_remaining=duration,
            started_season=self.identity.current_season,
            scar_econ=scar_econ,
            scar_sentiment=scar_sent,
            scar_trust=scar_trust,
        )

        self.active_shocks.append(shock)
        self.memory.shock_history.append(shock.to_dict())

        # immediate narrative effects
        if shock_type in (ShockType.RECESSION, ShockType.LOCKOUT, ShockType.CAP_FREEZE, ShockType.PANDEMIC_LITE):
            self.narratives[NarrativeType.LEAGUE_TURMOIL] = clamp(self.narratives[NarrativeType.LEAGUE_TURMOIL] + 0.25 * severity)
        if shock_type == ShockType.TV_DEAL_WINDFALL:
            self.narratives[NarrativeType.LEAGUE_STABILITY] = clamp(self.narratives[NarrativeType.LEAGUE_STABILITY] + 0.20 * severity)
        if shock_type == ShockType.SCANDAL:
            self.health.scandal_frequency = clamp(self.health.scandal_frequency + 0.15 * severity)

        return shock

    def tick_shocks(self) -> None:
        for s in self.active_shocks:
            s.tick()
        # keep inactive ones in history but not in active list clutter forever
        self.active_shocks = [s for s in self.active_shocks if s.is_active()]

    # ------------------------------------------------------------
    # Expansion / Relocation logic (pressure, not guaranteed outcomes)
    # ------------------------------------------------------------

    def check_expansion(self, current_team_count: int) -> Dict[str, Any]:
        """
        Returns a decision object:
        {
          "consider_expansion": bool,
          "probability": float,
          "recommended_teams": int (0/1/2),
          "dilution_penalty": float,
          "reason": {...}
        }
        """
        health = self.health.health_score
        rs = self.economics.revenue_sharing_strength
        pipeline = 0.55
        if self.last_long_term_forecast:
            pipeline = self.last_long_term_forecast.talent_pipeline_strength

        # expansion pressure rises with health + windfall + pipeline strength
        windfall = 0.0
        for s in self.active_shocks:
            if s.is_active() and s.shock_type == ShockType.TV_DEAL_WINDFALL:
                windfall += 0.30 * s.severity

        # league gatekeeping reduces it
        pressure = clamp(0.10 + 0.55 * health + 0.20 * pipeline + windfall - 0.25 * self.expansion_gatekeeping)

        # revenue sharing needs to be decent to protect small markets
        pressure *= lerp(0.70, 1.05, rs)

        # if already near max teams, reduce
        if current_team_count >= self.identity.max_teams:
            pressure *= 0.10

        prob = clamp(pressure * 0.35, 0.0, 0.35)
        consider = self.rng.random() < prob

        # expansion size
        teams = 0
        if consider:
            # 2-team expansion more likely if league is very healthy
            teams = 2 if (health > 0.78 and pipeline > 0.55 and self.rng.random() < 0.45) else 1

        # dilution penalty increases if pipeline is weak
        dilution_penalty = clamp(0.10 + 0.50 * (1.0 - pipeline) + (0.15 if teams == 2 else 0.0))

        return {
            "consider_expansion": consider,
            "probability": prob,
            "recommended_teams": teams,
            "dilution_penalty": dilution_penalty,
            "reason": {
                "health": health,
                "pipeline": pipeline,
                "revenue_sharing_strength": rs,
                "windfall_factor": windfall,
                "gatekeeping": self.expansion_gatekeeping,
                "current_team_count": current_team_count,
                "max_teams": self.identity.max_teams,
            }
        }

    def check_relocation_pressure(self, team_stress_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Given stress signals from teams (arena issues, attendance collapse, ownership instability),
        returns a relocation climate object.

        team_stress_signals expected fields (best effort):
        - team_id (str)
        - attendance_trend (0..1)  # low = bad
        - profitability (0..1)     # low = bad
        - arena_stability (0..1)   # low = drama
        - ownership_stability (0..1)
        - market_size ("small"|"mid"|"large")
        """
        if not team_stress_signals:
            return {"pressure": 0.0, "at_risk": [], "approval_strictness": self.relocation_approval_strictness}

        at_risk: List[Dict[str, Any]] = []

        for t in team_stress_signals:
            att = float(t.get("attendance_trend", 0.5))
            prof = float(t.get("profitability", 0.5))
            arena = float(t.get("arena_stability", 0.6))
            own = float(t.get("ownership_stability", 0.6))

            ms = t.get("market_size", "mid")
            small = 1.0 if (isinstance(ms, str) and ms.lower() == "small") else 0.0

            stress = (
                0.30 * (1.0 - att) +
                0.30 * (1.0 - prof) +
                0.20 * (1.0 - arena) +
                0.20 * (1.0 - own) +
                0.10 * small
            )
            stress = clamp(stress)

            # Only flag if meaningfully stressed
            if stress > 0.55:
                at_risk.append({
                    "team_id": t.get("team_id", "unknown"),
                    "stress": stress,
                    "inputs": {"attendance": att, "profitability": prof, "arena": arena, "ownership": own, "small_market": small},
                })

        # League climate: pressure increases if revenue sharing is weak and health is low
        base = 0.10 + 0.35 * (1.0 - self.economics.revenue_sharing_strength) + 0.20 * (1.0 - self.health.health_score)
        # Also increases with number of at-risk teams
        base += 0.08 * min(6, len(at_risk))
        pressure = clamp(base)

        # Approval strictness: league tightens when health is low (protects brand),
        # loosens slightly if too many teams are dying.
        strict_target = clamp(0.70 - 0.20 * min(1.0, len(at_risk) / 6.0) + 0.15 * (1.0 - self.health.health_score))
        self.relocation_approval_strictness = lerp(self.relocation_approval_strictness, strict_target, 0.10)

        if pressure > 0.55:
            self.narratives[NarrativeType.SMALL_MARKETS_SQUEEZED] = clamp(self.narratives[NarrativeType.SMALL_MARKETS_SQUEEZED] + 0.10)

        return {
            "pressure": pressure,
            "at_risk": sorted(at_risk, key=lambda x: x["stress"], reverse=True),
            "approval_strictness": self.relocation_approval_strictness,
        }

    # ------------------------------------------------------------
    # Narratives (soft, political, sometimes irrational)
    # ------------------------------------------------------------

    def update_narratives(self) -> None:
        """
        Narratives drift each year based on parity, shocks, and league health.
        They influence media pressure, confidence, and league politics.
        """
        health = self.health.health_score
        parity = self.balance.parity_index

        # Natural decay (narratives fade)
        for k in list(self.narratives.keys()):
            self.narratives[k] = clamp(self.narratives[k] * (0.85 + 0.08 * self.rng.random()))

        # Parity pride vs dynasty alert
        if parity > 0.62:
            self.narratives[NarrativeType.PARITY_PRIDE] = clamp(self.narratives[NarrativeType.PARITY_PRIDE] + 0.10)
        if parity < 0.42:
            self.narratives[NarrativeType.DYNASTY_ALERT] = clamp(self.narratives[NarrativeType.DYNASTY_ALERT] + 0.15)
            self.narratives[NarrativeType.SUPERTEAMS_BACKLASH] = clamp(self.narratives[NarrativeType.SUPERTEAMS_BACKLASH] + 0.10)

        # Health stability vs turmoil
        if health > 0.75:
            self.narratives[NarrativeType.LEAGUE_STABILITY] = clamp(self.narratives[NarrativeType.LEAGUE_STABILITY] + 0.12)
        if health < 0.50:
            self.narratives[NarrativeType.LEAGUE_TURMOIL] = clamp(self.narratives[NarrativeType.LEAGUE_TURMOIL] + 0.18)

        # Youth movement rises if star_power is low (league wants new faces)
        if self.health.star_power < 0.55:
            self.narratives[NarrativeType.YOUTH_MOVEMENT] = clamp(self.narratives[NarrativeType.YOUTH_MOVEMENT] + 0.10)

        # Goaltending crisis rises in low-scoring / goalie-dominant climates
        if self.era_state.active_era == EraType.GOALIE_DOMINANCE or self.rules.scoring_emphasis < 0.42:
            self.narratives[NarrativeType.GOALTENDING_CRISIS] = clamp(self.narratives[NarrativeType.GOALTENDING_CRISIS] + 0.08)

        # Canadian market pressure rises when turmoil is high or if parity is low (fans hate imbalance)
        if self.narratives[NarrativeType.LEAGUE_TURMOIL] > 0.45 or self.narratives[NarrativeType.DYNASTY_ALERT] > 0.45:
            self.narratives[NarrativeType.CANADIAN_MARKET_PRESSURE] = clamp(self.narratives[NarrativeType.CANADIAN_MARKET_PRESSURE] + 0.05)

    # ------------------------------------------------------------
    # League Health drift (macro feedback)
    # ------------------------------------------------------------

    def apply_health_drift(self) -> None:
        """
        Health changes slowly unless shocked.
        """
        # Slow drift around current values
        self.health.avg_profitability = clamp(self.health.avg_profitability + self.rng.uniform(-0.03, 0.03))
        self.health.attendance_trend = clamp(self.health.attendance_trend + self.rng.uniform(-0.03, 0.03))

        # media sentiment follows health and narratives
        stability = self.narratives[NarrativeType.LEAGUE_STABILITY]
        turmoil = self.narratives[NarrativeType.LEAGUE_TURMOIL]
        sentiment_target = clamp(0.40 + 0.45 * self.health.health_score + 0.15 * stability - 0.20 * turmoil)
        self.health.media_sentiment = lerp(self.health.media_sentiment, sentiment_target, 0.20)

        # scandal frequency decays slowly unless boosted
        self.health.scandal_frequency = clamp(self.health.scandal_frequency * (0.92 + 0.05 * self.rng.random()))

        # star power drifts; in offense eras it rises, in cap crunch it can stagnate
        era = self.get_active_era_definition()
        star_target = clamp(self.health.star_power + 0.02 * (era.scoring_rate_mod - 1.0) + self.rng.uniform(-0.02, 0.02))
        self.health.star_power = clamp(lerp(self.health.star_power, star_target, 0.15))

        # competitive balance component ties to parity
        self.health.competitive_balance = clamp(self.balance.parity_index)

        # Recompute composite
        self.health.recompute()

    # ------------------------------------------------------------
    # Awards Environment (soft biases) — minimal here, but exposed
    # ------------------------------------------------------------

    def get_award_environment(self) -> Dict[str, float]:
        """
        Biases for award voting. Used by award system (not implemented here).
        """
        # narrative bias: points vs story vs defense respect
        points_bias = clamp(0.50 + 0.25 * self.rules.scoring_emphasis + 0.10 * (self.era_state.active_era == EraType.OFFENSE_BOOM))
        narrative_bias = clamp(0.35 + 0.25 * self.narratives[NarrativeType.YOUTH_MOVEMENT] + 0.20 * self.narratives[NarrativeType.DYNASTY_ALERT])
        defense_respect = clamp(0.40 + 0.25 * (1.0 - self.rules.scoring_emphasis) + 0.15 * (self.era_state.active_era == EraType.GRIND_LOW_SCORING))
        goalie_fatigue = clamp(0.30 + 0.30 * self.narratives[NarrativeType.GOALTENDING_CRISIS])

        return {
            "mvp_points_bias": points_bias,
            "mvp_narrative_bias": narrative_bias,
            "defenseman_respect": defense_respect,
            "goalie_fatigue": goalie_fatigue,
            "rookie_hype_inflation": clamp(0.35 + 0.35 * self.narratives[NarrativeType.YOUTH_MOVEMENT]),
        }

    # ------------------------------------------------------------
    # Main seasonal advancement
    # ------------------------------------------------------------

    def advance_season(
        self,
        team_snapshots: Optional[List[Dict[str, Any]]] = None,
        team_count: Optional[int] = None,
        relocation_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates macro season-to-season evolution.

        You can pass team_snapshots & relocation_signals if your engine has them.
        league.py still functions without them (it will drift based on internal state).
        """
        # 1) Optionally ingest results
        if team_snapshots:
            self.update_from_team_snapshots(team_snapshots)

        # 2) Narrative update (political memory)
        self.update_narratives()

        # 3) Health drift (slow unless shocked)
        self.apply_health_drift()

        # 4) Rule drift (tiny)
        self.rules.drift(self.rng, intensity=0.02)

        # 5) Shocks: roll for a new one, then tick existing ones
        new_shock = self.check_shocks()
        self.tick_shocks()

        # 6) Economics update (cap philosophy)
        self.apply_economic_update()

        # 7) Era shift (rare)
        era_shift_record = self.apply_era_shift()

        # 8) Possible rule change (rare)
        rule_change_record = self.maybe_apply_rule_change()

        # 9) Forecasts for upcoming season
        season_forecast = self.generate_forecast()
        long_term = self.generate_long_term_forecast(horizon_years=4)

        # 10) Expansion / relocation climates
        expansion_decision = None
        if team_count is not None:
            expansion_decision = self.check_expansion(current_team_count=team_count)

        relocation_climate = None
        if relocation_signals is not None:
            relocation_climate = self.check_relocation_pressure(relocation_signals)

        # 11) Advance the season counter last (so records belong to the season that just ran)
        result = {
            "season_completed": self.identity.current_season,
            "new_shock": new_shock.to_dict() if new_shock else None,
            "era_shift": era_shift_record,
            "rule_change": rule_change_record,
            "season_forecast": season_forecast.to_dict(),
            "long_term_forecast": long_term.to_dict(),
            "expansion_decision": expansion_decision,
            "relocation_climate": relocation_climate,
            "league_context": self.get_league_context(),
        }

        self.identity.current_season += 1
        return result

    # ------------------------------------------------------------
    # Convenience: manual hooks requested in your prompt
    # ------------------------------------------------------------

    def apply_era_shift_manual(self, era: EraType, intensity: float = 0.55) -> None:
        self.era_state.active_era = era
        self.era_state.intensity = clamp(intensity)
        self.era_state.years_in_era = 0

    def apply_economic_update_manual(self, salary_cap: Optional[float] = None, cap_floor: Optional[float] = None) -> None:
        if salary_cap is not None:
            self.economics.salary_cap = float(salary_cap)
        if cap_floor is not None:
            self.economics.cap_floor = float(cap_floor)

    # ------------------------------------------------------------
    # Serialization (save/load persistent league state)
    # ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "economics": self.economics.to_dict(),
            "rules": self.rules.to_dict(),
            "balance": self.balance.to_dict(),
            "health": self.health.to_dict(),
            "era_state": self.era_state.to_dict(),
            "narratives": {k.value: v for k, v in self.narratives.items()},
            "active_shocks": [s.to_dict() for s in self.active_shocks],
            "memory": self.memory.to_dict(),
            "relocation_approval_strictness": self.relocation_approval_strictness,
            "expansion_gatekeeping": self.expansion_gatekeeping,
            "last_season_forecast": self.last_season_forecast.to_dict() if self.last_season_forecast else None,
            "last_long_term_forecast": self.last_long_term_forecast.to_dict() if self.last_long_term_forecast else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], seed: Optional[int] = None) -> "League":
        league = cls(seed=seed)

        # identity
        ident = data.get("identity", {})
        league.identity = LeagueIdentity(**{k: ident[k] for k in LeagueIdentity().__dict__.keys() if k in ident})

        # economics / rules / balance / health
        econ = data.get("economics", {})
        league.economics = EconomicEnvironment(**{k: econ[k] for k in EconomicEnvironment().__dict__.keys() if k in econ})

        rules = data.get("rules", {})
        league.rules = RuleEnvironment(**{k: rules[k] for k in RuleEnvironment().__dict__.keys() if k in rules})

        bal = data.get("balance", {})
        league.balance = CompetitiveBalance(**{k: bal[k] for k in CompetitiveBalance().__dict__.keys() if k in bal})

        hl = data.get("health", {})
        league.health = LeagueHealth(**{k: hl[k] for k in LeagueHealth().__dict__.keys() if k in hl})
        league.health.recompute()

        # era state
        es = data.get("era_state", {})
        if es:
            league.era_state = EraState(
                active_era=EraType(es.get("active_era", league.era_state.active_era.value)),
                intensity=float(es.get("intensity", league.era_state.intensity)),
                years_in_era=int(es.get("years_in_era", league.era_state.years_in_era)),
            )

        # narratives
        n = data.get("narratives", {})
        for k, v in n.items():
            try:
                league.narratives[NarrativeType(k)] = float(v)
            except Exception:
                continue

        # shocks
        league.active_shocks = []
        for sd in data.get("active_shocks", []):
            try:
                league.active_shocks.append(ShockEvent(
                    shock_type=ShockType(sd["shock_type"]),
                    severity=float(sd["severity"]),
                    duration_years=int(sd["duration_years"]),
                    years_remaining=int(sd["years_remaining"]),
                    started_season=int(sd["started_season"]),
                    scar_econ=float(sd.get("scar_econ", 0.0)),
                    scar_sentiment=float(sd.get("scar_sentiment", 0.0)),
                    scar_trust=float(sd.get("scar_trust", 0.0)),
                ))
            except Exception:
                continue

        # memory
        mem = data.get("memory", {})
        league.memory = LeagueMemory(**{k: mem.get(k, getattr(LeagueMemory(), k)) for k in LeagueMemory().__dict__.keys()})

        # misc
        league.relocation_approval_strictness = float(data.get("relocation_approval_strictness", league.relocation_approval_strictness))
        league.expansion_gatekeeping = float(data.get("expansion_gatekeeping", league.expansion_gatekeeping))

        # forecasts (optional caches)
        lf = data.get("last_season_forecast")
        if lf:
            league.last_season_forecast = LeagueForecast(**lf)
        ltf = data.get("last_long_term_forecast")
        if ltf:
            league.last_long_term_forecast = LongTermForecast(**ltf)

        return league
