# coach.py
# NHL Dynasty Mode Simulation Engine
# Core realism pillar: Coaching as the "intent + culture + deployment" layer.
#
# Copy-paste ready, designed to be usable immediately while still expandable.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import random
import math


# ============================================================
# Enums / constants
# ============================================================

class CoachRole(str, Enum):
    HEAD_COACH = "head_coach"
    ASSISTANT = "assistant"
    ASSOCIATE = "associate"
    GOALIE_COACH = "goalie_coach"


class CoachStage(str, Enum):
    ROOKIE = "rookie"
    PRIME = "prime"
    VETERAN = "veteran"
    WASHED = "washed"
    LEGACY = "legacy"


class PacePreference(str, Enum):
    SLOW_CONTROLLED = "slow_controlled"
    BALANCED = "balanced"
    HIGH_TEMPO = "high_tempo"


class NeutralZoneScheme(str, Enum):
    TRAP = "trap"
    HYBRID = "hybrid"
    PRESSURE = "pressure"


class DefensiveStructure(str, Enum):
    COLLAPSE = "collapse"
    BOX_1 = "box+1"
    MAN = "man"


class CoachEventType(str, Enum):
    PRAISE_PUBLIC = "praise_public"
    CRITICIZE_PUBLIC = "criticize_public"
    BENCH = "bench"
    PROMOTE_ROLE = "promote_role"
    DEMOTE_ROLE = "demote_role"
    ICE_TIME_INCREASE = "ice_time_increase"
    ICE_TIME_DECREASE = "ice_time_decrease"
    DEVELOPMENT_SUCCESS = "development_success"
    DEVELOPMENT_FAILURE = "development_failure"
    LOCKER_ROOM_LOSS = "locker_room_loss"
    WIN_STREAK = "win_streak"
    LOSING_STREAK = "losing_streak"
    SCAPEGOAT_MEDIA = "scapegoat_media"


# ============================================================
# Utilities
# ============================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def sigmoid(x: float) -> float:
    # stable-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def rand_name(rng: random.Random) -> str:
    first = ["Mike", "Paul", "Dave", "Ryan", "Corey", "Brad", "Dan", "Todd", "Jeff", "Rob",
             "Marc", "Jon", "Chris", "Pat", "Glen", "Sean", "Travis", "Jamie", "Steve", "Ken"]
    last = ["Hawkins", "Laviolette", "Boucher", "Fletcher", "Savard", "Keane", "Hartman", "Clarke",
            "Byrne", "Morrison", "Sutter", "Cassidy", "Brindley", "Tortsen", "Vigneault", "Quinn",
            "Gallant", "Carbery", "Cooper", "Maurice"]
    return f"{rng.choice(first)} {rng.choice(last)}"


def market_pressure_multiplier(market_tag: str) -> float:
    # Used to amplify media pressure / job security volatility.
    tag = (market_tag or "").lower().strip()
    if tag in {"toronto", "montreal", "new york", "ny", "nyc", "boston", "philadelphia", "vancouver"}:
        return 1.25
    if tag in {"chicago", "detroit", "los angeles", "la"}:
        return 1.10
    if tag in {"arizona", "columbus", "anaheim", "san jose", "sj"}:
        return 0.85
    return 1.0


# ============================================================
# Relationship model (Coach ↔ Player)
# ============================================================

@dataclass
class CoachPlayerRelationship:
    trust: float = 0.50
    respect: float = 0.50
    conflict: float = 0.10

    def score(self) -> float:
        # overall relationship score: positive minus conflict
        return clamp((self.trust * 0.45 + self.respect * 0.45) - (self.conflict * 0.60), 0.0, 1.0)

    def apply_event(self, event_type: CoachEventType, severity: float, rng: random.Random) -> None:
        """
        severity should be 0..1. Event updates are intentionally messy:
        - players overreact sometimes
        - coach communication style / accountability are applied at Coach-level elsewhere
        """
        s = clamp(severity, 0.0, 1.0)

        if event_type == CoachEventType.PRAISE_PUBLIC:
            self.trust = clamp(self.trust + 0.04 * s)
            self.respect = clamp(self.respect + 0.06 * s)
            self.conflict = clamp(self.conflict - 0.03 * s)

        elif event_type == CoachEventType.CRITICIZE_PUBLIC:
            self.trust = clamp(self.trust - 0.06 * s)
            self.respect = clamp(self.respect - 0.04 * s)
            self.conflict = clamp(self.conflict + 0.08 * s)

        elif event_type == CoachEventType.BENCH:
            self.trust = clamp(self.trust - 0.05 * s)
            self.respect = clamp(self.respect - 0.03 * s)
            self.conflict = clamp(self.conflict + 0.07 * s)

        elif event_type == CoachEventType.PROMOTE_ROLE:
            self.trust = clamp(self.trust + 0.05 * s)
            self.respect = clamp(self.respect + 0.04 * s)
            self.conflict = clamp(self.conflict - 0.02 * s)

        elif event_type == CoachEventType.DEMOTE_ROLE:
            self.trust = clamp(self.trust - 0.05 * s)
            self.respect = clamp(self.respect - 0.04 * s)
            self.conflict = clamp(self.conflict + 0.06 * s)

        elif event_type == CoachEventType.ICE_TIME_INCREASE:
            self.trust = clamp(self.trust + 0.03 * s)
            self.respect = clamp(self.respect + 0.03 * s)

        elif event_type == CoachEventType.ICE_TIME_DECREASE:
            self.trust = clamp(self.trust - 0.04 * s)
            self.conflict = clamp(self.conflict + 0.04 * s)

        elif event_type == CoachEventType.DEVELOPMENT_SUCCESS:
            self.trust = clamp(self.trust + 0.05 * s)
            self.respect = clamp(self.respect + 0.06 * s)
            self.conflict = clamp(self.conflict - 0.03 * s)

        elif event_type == CoachEventType.DEVELOPMENT_FAILURE:
            # Sometimes the player blames themselves, sometimes the coach.
            blame = rng.random()
            self.respect = clamp(self.respect - (0.02 * s if blame < 0.45 else 0.06 * s))
            self.trust = clamp(self.trust - (0.02 * s if blame < 0.45 else 0.05 * s))
            self.conflict = clamp(self.conflict + 0.04 * s)


# ============================================================
# Coach Philosophy
# ============================================================

@dataclass
class TacticalPhilosophy:
    forecheck_style: float = 0.50               # 0 passive -> 1 aggressive
    neutral_zone_scheme: NeutralZoneScheme = NeutralZoneScheme.HYBRID
    defensive_structure: DefensiveStructure = DefensiveStructure.COLLAPSE
    offensive_activation: float = 0.50          # 0 low -> 1 high
    risk_tolerance: float = 0.50                # 0 safe -> 1 risky
    pace_preference: PacePreference = PacePreference.BALANCED

    def volatility_factor(self) -> float:
        # Higher risk -> higher variance
        base = 0.15 + (self.risk_tolerance * 0.45)
        if self.pace_preference == PacePreference.HIGH_TEMPO:
            base += 0.08
        return clamp(base, 0.10, 0.85)


@dataclass
class UsagePhilosophy:
    trust_veterans: float = 0.55
    trust_youth: float = 0.45
    meritocracy: float = 0.50

    line_blending_frequency: float = 0.50       # 0 stable lines -> 1 blender
    scratch_rotation: float = 0.45

    powerplay_creativity: float = 0.50
    penalty_kill_conservatism: float = 0.55

    def deployment_bias(self) -> float:
        # positive -> vets, negative -> youth
        return clamp(self.trust_veterans - self.trust_youth, -1.0, 1.0)


@dataclass
class DevelopmentModifiers:
    skill_growth_multiplier: float = 1.00
    iq_growth_multiplier: float = 1.00
    defensive_growth_multiplier: float = 1.00
    confidence_growth_modifier: float = 0.00   # additive to confidence delta

    def as_dict(self) -> Dict[str, float]:
        return {
            "skill_growth_multiplier": self.skill_growth_multiplier,
            "iq_growth_multiplier": self.iq_growth_multiplier,
            "defensive_growth_multiplier": self.defensive_growth_multiplier,
            "confidence_growth_modifier": self.confidence_growth_modifier,
        }


# ============================================================
# Coach Core Entity
# ============================================================

@dataclass
class Coach:
    # Identity
    coach_id: str
    name: str
    age: int
    nationality: str
    experience_years: int
    career_stage: CoachStage

    # Role
    role: CoachRole
    specialization: str  # offense/defense/special_teams/development

    # Reputation (0..1)
    league_reputation: float = 0.50
    media_reputation: float = 0.50
    player_reputation: float = 0.50
    gm_trust: float = 0.50

    # Security
    contract_years_remaining: int = 2
    job_security: float = 0.60
    hot_seat: bool = False

    # Philosophy
    tactics: TacticalPhilosophy = field(default_factory=TacticalPhilosophy)
    usage: UsagePhilosophy = field(default_factory=UsagePhilosophy)
    development: DevelopmentModifiers = field(default_factory=DevelopmentModifiers)

    # Locker room control
    authority_level: float = 0.55            # can enforce decisions
    communication_style: float = 0.55        # 0 transparent -> 1 authoritarian
    accountability: float = 0.55             # 0 soft -> 1 ruthless

    # Adaptation
    adaptability: float = 0.55
    ego: float = 0.50
    learning_rate: float = 0.45

    # Political ecosystem
    gm_alignment_vision_match: float = 0.55
    gm_alignment_patience_sync: float = 0.55
    roster_control_conflict: float = 0.35

    # State / memory
    lost_room: bool = False
    room_temperature: float = 0.55           # 0 frozen mutiny -> 1 all-in
    fatigue_burnout: float = 0.15            # increases over years, affects decisions

    relationships: Dict[str, CoachPlayerRelationship] = field(default_factory=dict)

    # "Narrative" tags for debugging / flavor
    tags: List[str] = field(default_factory=list)

    # Last season memory
    last_season: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------
    # Relationship helpers
    # ------------------------------------------------------------

    def get_relationship(self, player_id: str) -> CoachPlayerRelationship:
        if player_id not in self.relationships:
            self.relationships[player_id] = CoachPlayerRelationship()
        return self.relationships[player_id]

    def apply_player_event(
        self,
        player_id: str,
        event_type: CoachEventType,
        severity: float,
        rng: random.Random,
        player_personality: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Player personality can amplify reaction. Expected keys (0..1):
        ego, patience, coachability, volatility, competitiveness, loyalty
        """
        rel = self.get_relationship(player_id)

        # Personality amplifier (optional, safe default)
        amp = 1.0
        if player_personality:
            volatility = clamp(player_personality.get("volatility", 0.50))
            ego = clamp(player_personality.get("ego", 0.50))
            coachability = clamp(player_personality.get("coachability", 0.50))
            patience = clamp(player_personality.get("patience", 0.50))

            # authoritarian comm style makes egos angrier, transparency reduces conflict
            comm_pressure = lerp(0.85, 1.15, self.communication_style)
            coachability_buffer = lerp(1.15, 0.85, coachability)

            # low patience + high volatility + high ego => bigger swings
            amp *= lerp(0.85, 1.25, volatility)
            amp *= lerp(0.90, 1.20, ego)
            amp *= lerp(1.15, 0.85, patience)
            amp *= comm_pressure
            amp *= coachability_buffer

        rel.apply_event(event_type, clamp(severity * amp), rng)

        # Relationship impacts room temperature a tiny bit (aggregate)
        # More conflict -> colder room.
        self.room_temperature = clamp(self.room_temperature - 0.010 * rel.conflict + 0.007 * rel.score())

    # ------------------------------------------------------------
    # Drift / evolution over time
    # ------------------------------------------------------------

    def year_tick(
        self,
        rng: random.Random,
        market_tag: str = "",
        team_status: str = "bubble",
        owner_expectations: float = 0.55,
        results: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run once per season.
        results can include:
          - wins, points_pct
          - made_playoffs (0/1)
          - xgf, xga, gf_pct, etc (optional)
          - morale_avg (optional)
        """
        mp = market_pressure_multiplier(market_tag)
        results = results or {}

        # Aging
        self.age += 1
        self.experience_years += 1

        # Contract tick
        if self.contract_years_remaining > 0:
            self.contract_years_remaining -= 1

        # Burnout slowly increases with pressure + authoritarian style + accountability + market
        pressure = self._compute_pressure(team_status=team_status, owner_expectations=owner_expectations, market_tag=market_tag)
        burnout_gain = 0.03 * mp * (0.6 + pressure) * lerp(0.85, 1.15, self.accountability) * lerp(0.90, 1.15, self.communication_style)
        self.fatigue_burnout = clamp(self.fatigue_burnout + burnout_gain)

        # Reputation drift (not guaranteed; narratives can lie)
        points_pct = results.get("points_pct", None)
        made_playoffs = results.get("made_playoffs", None)

        if points_pct is not None:
            # league reputation: slow-moving, tied to outcomes but noisy
            perf = clamp(points_pct)
            noise = rng.uniform(-0.03, 0.03)
            self.league_reputation = clamp(self.league_reputation + (perf - 0.50) * 0.08 + noise)

            # media reputation amplified by market pressure and expectations
            media_noise = rng.uniform(-0.05, 0.05) * mp
            self.media_reputation = clamp(self.media_reputation + (perf - 0.50) * 0.10 * mp + media_noise)

            # player reputation tied to room temperature + development results vibe (noisy)
            player_noise = rng.uniform(-0.04, 0.04)
            self.player_reputation = clamp(self.player_reputation + (self.room_temperature - 0.55) * 0.10 + player_noise)

        if made_playoffs is not None:
            if made_playoffs >= 0.5:
                self.gm_trust = clamp(self.gm_trust + 0.06 + rng.uniform(-0.02, 0.02))
            else:
                self.gm_trust = clamp(self.gm_trust - 0.08 * mp + rng.uniform(-0.02, 0.02))

        # Adjust job security from outcomes + pressure + politics
        self._update_job_security(
            rng=rng,
            market_tag=market_tag,
            owner_expectations=owner_expectations,
            team_status=team_status,
            results=results,
        )

        # Adaptation (change or die)
        self._adapt_after_season(rng=rng, results=results)

        # Career stage drift (rough)
        self._update_career_stage()

        # Room recovery / decay
        self._room_tick(rng=rng, pressure=pressure)

        # Save last season memory
        self.last_season = {
            "points_pct": results.get("points_pct"),
            "made_playoffs": results.get("made_playoffs"),
            "pressure": pressure,
            "market": market_tag,
            "job_security": self.job_security,
            "lost_room": self.lost_room,
            "room_temperature": self.room_temperature,
            "fatigue_burnout": self.fatigue_burnout,
            "tactics": {
                "forecheck_style": self.tactics.forecheck_style,
                "offensive_activation": self.tactics.offensive_activation,
                "risk_tolerance": self.tactics.risk_tolerance,
                "pace_preference": self.tactics.pace_preference.value,
            },
        }

        return self.last_season

    def _update_career_stage(self) -> None:
        # This is deliberately fuzzy.
        if self.experience_years < 4:
            self.career_stage = CoachStage.ROOKIE
        elif self.experience_years < 11:
            self.career_stage = CoachStage.PRIME
        elif self.experience_years < 20:
            self.career_stage = CoachStage.VETERAN
        else:
            # If reputation is high, legacy. Otherwise washed.
            if self.league_reputation > 0.72:
                self.career_stage = CoachStage.LEGACY
            else:
                self.career_stage = CoachStage.WASHED

    def _room_tick(self, rng: random.Random, pressure: float) -> None:
        # Pressure makes room harder to manage.
        # Strong authority can stabilize unless lost_room is already set.
        stability = (self.authority_level * 0.35 + self.player_reputation * 0.25 + self.gm_trust * 0.15)
        volatility = (pressure * 0.25 + self.fatigue_burnout * 0.25 + (1.0 - stability) * 0.20)

        drift = rng.uniform(-0.05, 0.05) * (0.7 + volatility)
        self.room_temperature = clamp(self.room_temperature + drift + (stability - 0.50) * 0.03)

        # Losing the room check
        if not self.lost_room:
            # If room temp is low AND coach is authoritarian AND accountability is high -> faster mutiny
            mutiny_risk = (0.55 - self.room_temperature) * 1.25
            mutiny_risk *= lerp(0.85, 1.20, self.communication_style)
            mutiny_risk *= lerp(0.90, 1.15, self.accountability)
            mutiny_risk = clamp(mutiny_risk, 0.0, 1.0)

            if rng.random() < mutiny_risk * 0.18:
                self.lost_room = True
                self.tags.append("lost_room")
                # when you lose the room, it gets colder fast
                self.room_temperature = clamp(self.room_temperature - rng.uniform(0.08, 0.18))

        else:
            # Hard to recover, but possible if authority is high and communication is more transparent
            recover_chance = (self.authority_level * 0.08) * lerp(1.20, 0.85, self.communication_style)
            recover_chance *= (0.7 - pressure) if pressure < 0.7 else 0.5
            recover_chance = clamp(recover_chance, 0.0, 0.12)
            if rng.random() < recover_chance:
                self.lost_room = False
                self.tags.append("room_recovered")
                self.room_temperature = clamp(self.room_temperature + rng.uniform(0.06, 0.12))

    # ------------------------------------------------------------
    # Pressure + Security
    # ------------------------------------------------------------

    def _compute_pressure(self, team_status: str, owner_expectations: float, market_tag: str) -> float:
        """
        Pressure is the engine that turns small problems into spiral collapses.
        """
        mp = market_pressure_multiplier(market_tag)
        status = (team_status or "bubble").lower().strip()

        # Baseline pressure by team context
        base = 0.45
        if status in {"win_now", "contender"}:
            base = 0.65
        elif status in {"rebuild", "tank"}:
            base = 0.35

        # Owner expectations amplify
        base += (clamp(owner_expectations) - 0.50) * 0.55

        # Market amplification
        base *= mp

        # Hot seat adds pressure feedback
        if self.hot_seat:
            base += 0.10

        return clamp(base, 0.0, 1.0)

    def _update_job_security(
        self,
        rng: random.Random,
        market_tag: str,
        owner_expectations: float,
        team_status: str,
        results: Dict[str, float],
    ) -> None:
        mp = market_pressure_multiplier(market_tag)
        pressure = self._compute_pressure(team_status=team_status, owner_expectations=owner_expectations, market_tag=market_tag)

        points_pct = results.get("points_pct", None)
        made_playoffs = results.get("made_playoffs", None)
        losing_streaks = results.get("losing_streaks", 0.0)
        player_conflicts = results.get("player_conflicts", 0.0)  # 0..1 estimated
        media_pressure = results.get("media_pressure", pressure) # 0..1 fallback

        # Performance delta (if missing, assume "meh")
        perf = 0.50 if points_pct is None else clamp(points_pct)

        # Owner expectation gap
        expectation_gap = clamp(owner_expectations) - perf  # positive gap means underperformed

        # Big drivers
        sec_delta = 0.0
        sec_delta -= expectation_gap * (0.35 + 0.25 * mp)
        sec_delta -= clamp(losing_streaks) * 0.08 * mp
        sec_delta -= clamp(player_conflicts) * 0.12
        sec_delta -= clamp(media_pressure) * 0.06 * mp

        if made_playoffs is not None:
            if made_playoffs >= 0.5:
                sec_delta += 0.08
            else:
                sec_delta -= 0.10 * mp

        # Politics can sink a coach even if the record is decent
        politics = (1.0 - self.gm_alignment_vision_match) * 0.10 + self.roster_control_conflict * 0.06
        sec_delta -= politics

        # Random unfairness / scapegoating (mandatory realism)
        # Sometimes coach survives despite bad season; sometimes fired unfairly.
        chaos = rng.uniform(-0.06, 0.06) * (0.6 + pressure)
        sec_delta += chaos

        # Apply and clamp
        self.job_security = clamp(self.job_security + sec_delta)

        # Hot seat threshold
        self.hot_seat = self.job_security < 0.35 or (self.contract_years_remaining <= 0 and self.job_security < 0.45)

    def should_be_fired(self, rng: random.Random, market_tag: str = "") -> Tuple[bool, str]:
        """
        Returns (fired, reason).
        Not deterministic: bad coaches can survive, good coaches can be scapegoated.
        """
        mp = market_pressure_multiplier(market_tag)

        # Base chance from insecurity
        base = clamp((0.40 - self.job_security) * 1.8, 0.0, 1.0)

        # Losing the room accelerates
        if self.lost_room:
            base += 0.18

        # Contract expiry makes it easier to "mutually part ways"
        if self.contract_years_remaining <= 0:
            base += 0.08

        # Media reputation matters more in big markets
        base += (0.45 - self.media_reputation) * 0.20 * mp

        # Randomness to keep it human
        base += rng.uniform(-0.05, 0.05)

        base = clamp(base)

        roll = rng.random()
        if roll < base:
            # Choose a reason
            if self.lost_room and roll < base * 0.55:
                return True, "lost_the_room"
            if self.job_security < 0.20:
                return True, "results_unacceptable"
            if self.media_reputation < 0.35 and mp > 1.05:
                return True, "media_pressure"
            return True, "ownership_direction_change"

        return False, "survived"

    # ------------------------------------------------------------
    # Adaptation engine
    # ------------------------------------------------------------

    def _adapt_after_season(self, rng: random.Random, results: Dict[str, float]) -> None:
        """
        Coach changes systems after failure... or doubles down because ego.
        This is where identity gets forged or destroyed.
        """
        perf = results.get("points_pct", 0.50)
        perf = clamp(perf)

        # If good season, coach gets confident (and can become stubborn)
        if perf >= 0.56:
            self.ego = clamp(self.ego + 0.03 + rng.uniform(-0.01, 0.02))
            # mild improvements
            self.learning_rate = clamp(self.learning_rate + 0.01 + rng.uniform(-0.01, 0.01))
            return

        # If bad season, chance to adapt depends on adaptability and ego
        failure_weight = clamp(0.56 - perf, 0.0, 0.20) / 0.20  # 0..1
        adapt_score = (self.adaptability * 0.55 + self.learning_rate * 0.35) - (self.ego * 0.45)
        adapt_prob = clamp(0.20 + adapt_score * 0.45 + failure_weight * 0.35)

        # Burnout reduces smart change; exhausted coaches do dumb things
        adapt_prob *= lerp(1.0, 0.75, self.fatigue_burnout)

        if rng.random() < adapt_prob:
            # Adapt in a direction that makes sense
            # e.g., if too risky, get safer; if too passive, get more aggressive.
            shift = rng.uniform(0.04, 0.12) * (0.8 + self.adaptability)

            # Risk adjustment
            if self.tactics.risk_tolerance > 0.55 and rng.random() < 0.60:
                self.tactics.risk_tolerance = clamp(self.tactics.risk_tolerance - shift)
                self.tags.append("adapted_safer")
            elif self.tactics.risk_tolerance < 0.45 and rng.random() < 0.50:
                self.tactics.risk_tolerance = clamp(self.tactics.risk_tolerance + shift)
                self.tags.append("adapted_riskier")

            # Pace adjustment
            if rng.random() < 0.45:
                self.tactics.pace_preference = rng.choice(list(PacePreference))
                self.tags.append("pace_shift")

            # Usage adjustment: stop blendering or start blendering
            if rng.random() < 0.40:
                self.usage.line_blending_frequency = clamp(self.usage.line_blending_frequency + rng.uniform(-0.12, 0.12))
                self.tags.append("deployment_tweaks")

            # Small morale repair if adapts (players respect self-awareness)
            self.room_temperature = clamp(self.room_temperature + rng.uniform(0.02, 0.06))
            self.player_reputation = clamp(self.player_reputation + rng.uniform(0.01, 0.03))

        else:
            # Double down: ego-driven stubbornness
            self.ego = clamp(self.ego + 0.04 + rng.uniform(-0.01, 0.03))
            self.tags.append("doubled_down")

            # Stubborn coaches may increase accountability/authoritarianism under pressure
            if rng.random() < 0.55:
                self.accountability = clamp(self.accountability + rng.uniform(0.02, 0.07))
                self.communication_style = clamp(self.communication_style + rng.uniform(0.01, 0.05))
                self.tags.append("tightened_leash")

            # Room can get colder
            self.room_temperature = clamp(self.room_temperature - rng.uniform(0.02, 0.07))

    # ------------------------------------------------------------
    # Development impacts
    # ------------------------------------------------------------

    def development_effects_for_player(
        self,
        player_id: str,
        player_age: int,
        player_archetype: str = "",
        player_personality: Optional[Dict[str, float]] = None,
        in_role_fit: float = 0.55,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, float]:
        """
        Returns growth modifiers to feed into your player growth engine.

        This is where "mismanagement is mandatory" lives:
        - Poor fit + low trust + buried youth -> suppressed growth
        - Development-first coaches boost young players but may cost wins elsewhere
        """
        rng = rng or random.Random()

        rel = self.get_relationship(player_id)
        rel_score = rel.score()

        # Age factor (young players are more affected by coaching)
        age = player_age
        youth_factor = 1.0
        if age <= 20:
            youth_factor = 1.20
        elif age <= 23:
            youth_factor = 1.10
        elif age <= 26:
            youth_factor = 1.02
        else:
            youth_factor = 0.95

        # Coach usage bias can bury youth
        bury_risk = clamp(self.usage.trust_veterans - self.usage.trust_youth, -1.0, 1.0)
        bury_penalty = 0.0
        if age <= 22 and bury_risk > 0.15:
            bury_penalty = 0.06 * bury_risk

        # Role fit is huge: playing a skill winger like a grinder hurts growth
        fit_penalty = clamp(0.55 - in_role_fit, 0.0, 0.55) * 0.18

        # Relationship impacts confidence/growth
        confidence_delta = (rel_score - 0.50) * 0.10 - (rel.conflict * 0.06)
        confidence_delta += self.development.confidence_growth_modifier

        # Base multipliers from coach development profile
        skill_mult = self.development.skill_growth_multiplier
        iq_mult = self.development.iq_growth_multiplier
        def_mult = self.development.defensive_growth_multiplier

        # Mismanagement: if relationship is bad OR fit is poor, growth can stall/regress
        mismanage = 0.0
        if rel_score < 0.38:
            mismanage += 0.08
        if in_role_fit < 0.45:
            mismanage += 0.10
        mismanage += bury_penalty

        # Personality interplay: low coachability + high ego amplifies mismanagement pain
        if player_personality:
            coachability = clamp(player_personality.get("coachability", 0.50))
            ego = clamp(player_personality.get("ego", 0.50))
            mismanage *= lerp(0.85, 1.20, ego)
            mismanage *= lerp(1.15, 0.85, coachability)

        # Apply youth factor (young players are more coach-sensitive)
        skill_mult *= youth_factor
        iq_mult *= youth_factor
        def_mult *= youth_factor

        # Apply mismanagement penalties (can drop below 1.0)
        skill_mult *= clamp(1.0 - mismanage, 0.75, 1.25)
        iq_mult *= clamp(1.0 - (mismanage * 0.85), 0.78, 1.25)
        def_mult *= clamp(1.0 - (mismanage * 0.80), 0.78, 1.25)

        # Noise (development isn't deterministic)
        noise = rng.uniform(-0.03, 0.03)
        skill_mult = clamp(skill_mult + noise, 0.70, 1.30)
        iq_mult = clamp(iq_mult + noise * 0.8, 0.70, 1.30)
        def_mult = clamp(def_mult + noise * 0.8, 0.70, 1.30)

        return {
            "skill_growth_multiplier": skill_mult,
            "iq_growth_multiplier": iq_mult,
            "defensive_growth_multiplier": def_mult,
            "confidence_growth_delta": clamp(confidence_delta, -0.20, 0.20),
            "mismanagement_index": clamp(mismanage, 0.0, 0.35),
            "relationship_score": rel_score,
        }

    # ------------------------------------------------------------
    # Game-to-game decision making (abstracted)
    # ------------------------------------------------------------

    def game_plan_modifiers(
        self,
        rng: random.Random,
        opponent_style_hint: str = "",
        team_morale: float = 0.55,
        injuries_key_players: float = 0.0,  # 0..1
        is_back_to_back: bool = False,
    ) -> Dict[str, float]:
        """
        Returns small modifiers that your game sim can apply.
        This is how the coach actually causes different outcomes without hardcoding.
        """
        morale = clamp(team_morale)
        fatigue = clamp(self.fatigue_burnout + (0.10 if is_back_to_back else 0.0))
        vol = self.tactics.volatility_factor()

        # System impacts (abstract):
        # - aggressive forecheck increases shots & chances, but also giveaways/xGA variance
        shot_push = (self.tactics.forecheck_style * 0.08) + (self.tactics.offensive_activation * 0.10)
        defense_stability = (1.0 - self.tactics.risk_tolerance) * 0.06

        # Pace:
        pace_mod = 0.0
        if self.tactics.pace_preference == PacePreference.HIGH_TEMPO:
            pace_mod = 0.06
        elif self.tactics.pace_preference == PacePreference.SLOW_CONTROLLED:
            pace_mod = -0.05

        # Lost room => effort variance increases, execution drops
        execution = (self.room_temperature - 0.50) * 0.08
        if self.lost_room:
            execution -= 0.06
            vol += 0.12

        # Fatigue reduces execution
        execution -= fatigue * 0.06

        # Injuries force line blending, which can hurt structure
        blender = self.usage.line_blending_frequency
        structure_hit = (blender * 0.05) + (injuries_key_players * 0.08)

        # morale slightly offsets
        morale_boost = (morale - 0.50) * 0.06

        # Noise based on volatility: higher volatility makes results swing
        noise = rng.uniform(-0.04, 0.04) * (0.6 + vol)

        return {
            "shot_rate_mod": shot_push + pace_mod + morale_boost + noise,     # affects shot volume / xGF
            "defense_rate_mod": defense_stability - structure_hit + noise,    # affects xGA prevention
            "execution_mod": execution + morale_boost + noise,                # affects finishing / mistakes
            "variance_mod": clamp(vol, 0.10, 0.95),                           # affects game outcome randomness
        }

    def late_game_decisions(
        self,
        rng: random.Random,
        goal_diff: int,
        time_remaining_minutes: int,
        star_usage_health_risk: float = 0.0,  # 0..1
    ) -> Dict[str, Any]:
        """
        Abstracted late-game behavior:
        - timeout usage
        - pull goalie aggressiveness
        - star player overuse
        """
        risk = clamp(self.tactics.risk_tolerance)
        ruthless = clamp(self.accountability)
        burnout = clamp(self.fatigue_burnout)

        # Pull goalie decision threshold
        # risky coaches pull earlier; conservative later
        base_pull_min = 2  # around 2 mins remaining as a baseline
        pull_adjust = int(round(lerp(+1.0, -1.0, risk)))  # risk=1 => -1 minute earlier
        pull_time = max(1, base_pull_min + pull_adjust)

        pull_goalie = (goal_diff == -1 and time_remaining_minutes <= pull_time)

        # Timeout usage: tired/chaotic coaches may misuse or save too long
        timeout_prob = 0.25 + (risk * 0.20) + ((1.0 - burnout) * 0.10)
        if self.lost_room:
            timeout_prob *= 0.85

        use_timeout = (time_remaining_minutes <= 2 and rng.random() < clamp(timeout_prob, 0.05, 0.65))

        # Star overuse: win-now + ruthless + risk => overload stars even with health risk
        star_overuse = False
        overuse_score = (risk * 0.35 + ruthless * 0.25) - (star_usage_health_risk * 0.45)
        if rng.random() < clamp(overuse_score, 0.0, 0.55):
            star_overuse = True

        return {
            "use_timeout": use_timeout,
            "pull_goalie": pull_goalie,
            "pull_goalie_time_threshold_min": pull_time,
            "overuse_stars": star_overuse,
        }

    # ------------------------------------------------------------
    # Retirement / career arcs
    # ------------------------------------------------------------

    def retirement_check(self, rng: random.Random) -> Tuple[bool, str]:
        """
        Coaches can retire for:
        - burnout
        - legacy preservation
        - repeated firings
        - age/health (abstract)
        """
        age = self.age
        burnout = self.fatigue_burnout
        rep = self.league_reputation

        # Base chance low until older, but burnout can spike it earlier.
        base = 0.001

        if age >= 62:
            base += 0.05
        elif age >= 58:
            base += 0.025
        elif age >= 54:
            base += 0.012

        # Burnout can cause early retirement
        base += burnout * 0.03

        # Legacy coaches sometimes quit to preserve rep
        if rep > 0.75 and burnout > 0.55:
            base += 0.015

        # Randomness
        base += rng.uniform(-0.002, 0.004)
        base = clamp(base, 0.0, 0.12)

        if rng.random() < base:
            if burnout > 0.65:
                return True, "burnout"
            if age >= 60:
                return True, "age"
            if rep > 0.75:
                return True, "legacy_exit"
            return True, "personal"

        return False, "continues"


# ============================================================
# Coach Factory + Hiring Market
# ============================================================

@dataclass
class OwnerProfile:
    # 0..1
    budget_sensitivity: float = 0.50
    market_pressure: float = 0.50
    legacy_expectations: float = 0.55
    patience: float = 0.55


@dataclass
class TeamContext:
    team_id: str
    team_name: str
    market_tag: str = ""
    status: str = "bubble"   # rebuild/bubble/win_now/contender/tank
    owner: OwnerProfile = field(default_factory=OwnerProfile)

    # optionally, provide how the GM wants to build
    gm_preference_risk: float = 0.50
    gm_preference_youth: float = 0.50
    gm_preference_structure: float = 0.50


def generate_coach(rng: random.Random, coach_id: str, role: CoachRole = CoachRole.HEAD_COACH) -> Coach:
    age = rng.randint(36, 66)
    exp = rng.randint(0, max(1, age - 30))

    # Stage initial guess
    if exp < 4:
        stage = CoachStage.ROOKIE
    elif exp < 11:
        stage = CoachStage.PRIME
    elif exp < 20:
        stage = CoachStage.VETERAN
    else:
        stage = CoachStage.WASHED

    nationality = rng.choice(["CAN", "USA", "SWE", "FIN", "CZE", "RUS", "SVK", "GER"])
    specialization = rng.choice(["offense", "defense", "special_teams", "development"])

    # Philosophy generation with coherent biases
    risk = clamp(rng.random() ** 0.9)  # slightly more mid/high risk
    forecheck = clamp(rng.random() * 0.9 + (0.2 if risk > 0.65 else 0.0))
    activation = clamp(rng.random() * 0.9 + (0.15 if risk > 0.65 else 0.0))

    pace = rng.choice(list(PacePreference))
    nz = rng.choice(list(NeutralZoneScheme))
    dstruct = rng.choice(list(DefensiveStructure))

    # Usage
    trust_vets = clamp(rng.uniform(0.35, 0.75))
    trust_youth = clamp(rng.uniform(0.25, 0.75))
    meritocracy = clamp(rng.uniform(0.30, 0.80))

    # Some coaches have extreme tendencies
    if rng.random() < 0.18:
        trust_vets = clamp(trust_vets + rng.uniform(0.10, 0.20))
        trust_youth = clamp(trust_youth - rng.uniform(0.10, 0.20))
    if rng.random() < 0.18:
        trust_youth = clamp(trust_youth + rng.uniform(0.10, 0.20))
        trust_vets = clamp(trust_vets - rng.uniform(0.10, 0.20))

    tactics = TacticalPhilosophy(
        forecheck_style=forecheck,
        neutral_zone_scheme=nz,
        defensive_structure=dstruct,
        offensive_activation=activation,
        risk_tolerance=risk,
        pace_preference=pace,
    )

    usage = UsagePhilosophy(
        trust_veterans=trust_vets,
        trust_youth=trust_youth,
        meritocracy=meritocracy,
        line_blending_frequency=clamp(rng.uniform(0.20, 0.85)),
        scratch_rotation=clamp(rng.uniform(0.20, 0.80)),
        powerplay_creativity=clamp(rng.uniform(0.20, 0.90)),
        penalty_kill_conservatism=clamp(rng.uniform(0.20, 0.90)),
    )

    # Development profile
    # Development guru boosts growth but can sacrifice short-term structure.
    base_skill = rng.uniform(0.92, 1.10)
    base_iq = rng.uniform(0.92, 1.12)
    base_def = rng.uniform(0.92, 1.12)
    conf_mod = rng.uniform(-0.03, 0.05)

    if specialization == "development":
        base_skill += rng.uniform(0.03, 0.09)
        base_iq += rng.uniform(0.03, 0.10)
        conf_mod += rng.uniform(0.01, 0.04)

    if specialization == "defense":
        base_def += rng.uniform(0.03, 0.10)

    development = DevelopmentModifiers(
        skill_growth_multiplier=clamp(base_skill, 0.85, 1.25),
        iq_growth_multiplier=clamp(base_iq, 0.85, 1.25),
        defensive_growth_multiplier=clamp(base_def, 0.85, 1.25),
        confidence_growth_modifier=clamp(conf_mod, -0.10, 0.10),
    )

    # Personality / control traits
    authority = clamp(rng.uniform(0.35, 0.85))
    comm = clamp(rng.uniform(0.20, 0.85))   # higher = more authoritarian
    acc = clamp(rng.uniform(0.25, 0.90))

    adaptability = clamp(rng.uniform(0.20, 0.85))
    ego = clamp(rng.uniform(0.15, 0.90))
    learning = clamp(rng.uniform(0.20, 0.80))

    # Reputation baseline influenced by exp
    rep_base = clamp(0.35 + (exp / 25.0) * 0.25 + rng.uniform(-0.05, 0.05))
    league_rep = clamp(rep_base + rng.uniform(-0.08, 0.08))
    media_rep = clamp(rep_base + rng.uniform(-0.10, 0.10))
    player_rep = clamp(rep_base + rng.uniform(-0.10, 0.10))
    gm_trust = clamp(0.45 + rng.uniform(-0.10, 0.10))

    # Contract/security
    contract_years = rng.randint(1, 4) if role == CoachRole.HEAD_COACH else rng.randint(1, 3)
    job_sec = clamp(rng.uniform(0.45, 0.75))
    hot_seat = job_sec < 0.40

    coach = Coach(
        coach_id=coach_id,
        name=rand_name(rng),
        age=age,
        nationality=nationality,
        experience_years=exp,
        career_stage=stage,
        role=role,
        specialization=specialization,

        league_reputation=league_rep,
        media_reputation=media_rep,
        player_reputation=player_rep,
        gm_trust=gm_trust,

        contract_years_remaining=contract_years,
        job_security=job_sec,
        hot_seat=hot_seat,

        tactics=tactics,
        usage=usage,
        development=development,

        authority_level=authority,
        communication_style=comm,
        accountability=acc,

        adaptability=adaptability,
        ego=ego,
        learning_rate=learning,

        gm_alignment_vision_match=clamp(rng.uniform(0.30, 0.80)),
        gm_alignment_patience_sync=clamp(rng.uniform(0.30, 0.80)),
        roster_control_conflict=clamp(rng.uniform(0.10, 0.70)),
    )

    # Tag some archetypes for flavor/debug
    if coach.tactics.risk_tolerance > 0.70:
        coach.tags.append("high_risk")
    if coach.usage.line_blending_frequency > 0.70:
        coach.tags.append("line_blender")
    if coach.specialization == "development":
        coach.tags.append("development_guru")
    if coach.communication_style > 0.70 and coach.accountability > 0.70:
        coach.tags.append("hard_ass")

    return coach


def coach_fit_score(coach: Coach, team: TeamContext) -> float:
    """
    How well does this coach fit this team?
    Not a perfect science; should produce weird hires sometimes.
    """
    mp = market_pressure_multiplier(team.market_tag)

    # Match risk preference
    risk_fit = 1.0 - abs(coach.tactics.risk_tolerance - clamp(team.gm_preference_risk))
    # Youth preference: coach trust_youth vs gm preference
    youth_fit = 1.0 - abs(coach.usage.trust_youth - clamp(team.gm_preference_youth))

    # Team status synergy
    status = (team.status or "bubble").lower().strip()
    status_fit = 0.55
    if status in {"rebuild", "tank"}:
        status_fit = 0.45 + (coach.usage.trust_youth * 0.35) + ((coach.development.skill_growth_multiplier - 1.0) * 0.25)
    elif status in {"win_now", "contender"}:
        status_fit = 0.55 + (coach.usage.trust_veterans * 0.25) + (coach.league_reputation * 0.25)

    # Market: harsh markets punish low media rep
    media_fit = coach.media_reputation if mp >= 1.05 else 0.60 + (coach.media_reputation * 0.40)

    # Ownership patience: if owner is impatient, want higher "authority" and "league reputation"
    owner = team.owner
    impatient = 1.0 - clamp(owner.patience)
    authority_need = coach.authority_level * (0.50 + impatient * 0.40)
    rep_need = coach.league_reputation * (0.55 + impatient * 0.35)

    # Politics: roster control conflict hurts
    politics_penalty = coach.roster_control_conflict * 0.18

    score = (
        risk_fit * 0.20 +
        youth_fit * 0.18 +
        clamp(status_fit) * 0.22 +
        clamp(media_fit) * 0.12 +
        clamp(authority_need) * 0.14 +
        clamp(rep_need) * 0.14
    ) - politics_penalty

    # Chaos: sometimes teams hire bad fits (MANDATORY realism)
    score += random.uniform(-0.08, 0.06)

    return clamp(score, 0.0, 1.0)


def generate_coach_market(rng: random.Random, n: int = 18) -> List[Coach]:
    coaches: List[Coach] = []
    for i in range(n):
        coach_id = f"coach_{rng.randint(100000, 999999)}"
        coaches.append(generate_coach(rng=rng, coach_id=coach_id, role=CoachRole.HEAD_COACH))
    return coaches


def hire_best_fit_coach(
    rng: random.Random,
    team: TeamContext,
    market: Optional[List[Coach]] = None,
) -> Tuple[Coach, List[Tuple[str, float]]]:
    """
    Returns (chosen_coach, shortlist_debug)
    shortlist_debug: list of (coach_name, fit_score)
    """
    market = market or generate_coach_market(rng, n=18)
    scored = [(c, coach_fit_score(c, team)) for c in market]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Teams do not always pick the best fit.
    # Choose from top 3 with weighted randomness.
    top = scored[:3] if len(scored) >= 3 else scored
    weights = [0.55, 0.30, 0.15][:len(top)]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    chosen = top[0][0]
    for (c, _s), w in zip(top, weights):
        acc += w
        if pick <= acc:
            chosen = c
            break

    shortlist = [(c.name, s) for c, s in scored[:8]]
    return chosen, shortlist


# ============================================================
# Integration-friendly helpers (optional, safe defaults)
# ============================================================

def coach_influences_team_identity(coach: Coach) -> Dict[str, Any]:
    """
    Returns a compact identity object that Team/League modules can store.
    """
    return {
        "coach_id": coach.coach_id,
        "coach_name": coach.name,
        "style": {
            "forecheck": coach.tactics.forecheck_style,
            "activation": coach.tactics.offensive_activation,
            "risk": coach.tactics.risk_tolerance,
            "pace": coach.tactics.pace_preference.value,
            "nz_scheme": coach.tactics.neutral_zone_scheme.value,
            "def_structure": coach.tactics.defensive_structure.value,
        },
        "usage": {
            "trust_vets": coach.usage.trust_veterans,
            "trust_youth": coach.usage.trust_youth,
            "meritocracy": coach.usage.meritocracy,
            "blending": coach.usage.line_blending_frequency,
        },
        "culture": {
            "authority": coach.authority_level,
            "comm_style": coach.communication_style,
            "accountability": coach.accountability,
            "lost_room": coach.lost_room,
            "room_temperature": coach.room_temperature,
        },
        "reputation": {
            "league": coach.league_reputation,
            "media": coach.media_reputation,
            "player": coach.player_reputation,
            "gm_trust": coach.gm_trust,
        }
    }
