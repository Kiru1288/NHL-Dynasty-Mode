# app/ai/personality.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterable, Dict
import random
import math


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def sigmoid(x: float) -> float:
    # Stable-ish sigmoid for small sims
    return 1.0 / (1.0 + math.exp(-x))


def gaussian01(rng: random.Random, mean: float, std: float) -> float:
    # clamp gaussian to [0,1]
    return clamp01(rng.gauss(mean, std))


def chance(rng: random.Random, p: float) -> bool:
    return rng.random() < clamp01(p)


# --------------------------------------------------
# Archetypes (INFLUENCES, NOT CLASSES)
# --------------------------------------------------

class PersonalityArchetype(str, Enum):
    LOYALIST = "loyalist"
    FAMILY_FIRST = "family_first"
    MONEY_HUNGRY = "money_hungry"
    COMPETITOR = "competitor"
    PROFESSIONAL = "professional"
    JOURNEYMAN = "journeyman"
    STAR = "star"
    CHIP_ON_SHOULDER = "chip_on_shoulder"
    LEADER = "leader"
    INTROVERT = "introvert"
    EXTROVERT = "extrovert"
    RISK_TAKER = "risk_taker"
    STABILITY_SEEKER = "stability_seeker"


# --------------------------------------------------
# Context passed into behavior sampling
# --------------------------------------------------

@dataclass(slots=True, frozen=True)
class BehaviorContext:
    """
    All fields are normalized [0,1] where possible.

    You can keep adding fields later without breaking everything.
    """

    # Team / season context
    team_success: float = 0.5          # 1.0 = elite contender, 0.0 = dumpster fire
    losing_streak: float = 0.0         # 0..1 scaled (e.g. 0.0 none, 1.0 huge)
    rebuild_mode: float = 0.0          # 0..1 how openly rebuilding the org is

    # Role / usage context
    role_mismatch: float = 0.0         # 0..1 player thinks role < deserved
    ice_time_satisfaction: float = 0.5 # 1.0 happy, 0.0 furious
    scratched_recently: float = 0.0    # 0..1 frequency / recency

    # Contract context
    offer_respect: float = 0.5         # 1.0 fair/strong offer, 0.0 insulting
    ufa_pressure: float = 0.0          # 0..1 (how close to UFA / leverage)
    market_heat: float = 0.5           # 0..1 (lots of bidders)

    # Health / life context
    injury_burden: float = 0.0         # 0..1 (chronic pain, rehab, etc.)
    family_event: float = 0.0          # 0..1 major family pull (new kid, sick parent, etc.)

    # Career context (optional to use)
    age_factor: float = 0.3            # 0..1 (older -> higher)
    cup_satisfaction: float = 0.0      # 0..1 (already won, less chasing)


# --------------------------------------------------
# Core Personality Model
# --------------------------------------------------

@dataclass(slots=True, frozen=True)
class PersonalityProfile:
    """
    Traits are stable long-term.
    These are NOT direct probabilities.
    """

    # ---- Motivation & Values ----
    loyalty: float
    ambition: float
    money_focus: float
    family_priority: float
    legacy_drive: float

    # ---- Risk & Adaptability ----
    risk_tolerance: float
    adaptability: float
    patience: float
    stability_need: float

    # ---- Ego & Psychology ----
    ego: float
    confidence: float
    volatility: float
    competitiveness: float

    # ---- Social / Locker Room ----
    leadership: float
    coachability: float
    media_comfort: float
    introversion: float

    def __post_init__(self) -> None:
        # Ensure everything is clamped
        for field in (
            "loyalty","ambition","money_focus","family_priority","legacy_drive",
            "risk_tolerance","adaptability","patience","stability_need",
            "ego","confidence","volatility","competitiveness",
            "leadership","coachability","media_comfort","introversion"
        ):
            object.__setattr__(self, field, clamp01(getattr(self, field)))


# --------------------------------------------------
# Dynamic Behavior (sampled, not hardcoded)
# --------------------------------------------------

class PersonalityBehavior:
    """
    The entire point of this class:
    - Same player can react differently at different times.
    - Traits set ranges, NOT outcomes.
    - Context + noise + rare "snap" events produce emergent stories.
    """

    def __init__(self, profile: PersonalityProfile, rng: Optional[random.Random] = None):
        self.p = profile
        self.rng = rng or random.Random()

    # --------------------------
    # Internal: "snap" mechanics
    # --------------------------

    def _snap_probability(self, ctx: BehaviorContext) -> float:
        """
        Probability of an emotional spike (out-of-character moment).
        Coachable/professional players still can snap, just less often.

        Drivers: losing streak, role mismatch, disrespectful offer, injury burden.
        Reducers: confidence, leadership, patience.
        Volatility amplifies everything.
        """
        stress = (
            0.35 * ctx.losing_streak +
            0.25 * ctx.role_mismatch +
            0.20 * (1.0 - ctx.offer_respect) +
            0.20 * ctx.injury_burden +
            0.10 * ctx.rebuild_mode +
            0.10 * ctx.scratched_recently +
            0.10 * (1.0 - ctx.ice_time_satisfaction)
        )

        resilience = (
            0.25 * self.p.confidence +
            0.20 * self.p.patience +
            0.15 * self.p.leadership +
            0.15 * self.p.coachability
        )

        # baseline 0.5% plus stress/resilience, volatility amplifies
        base = 0.005
        amp = lerp(0.6, 1.6, self.p.volatility)

        p = base + amp * (stress - 0.7 * resilience)
        return clamp01(p)

    def _snap_multiplier(self, ctx: BehaviorContext) -> float:
        """
        If snap triggers, how intense is it?
        Volatility + ego increase intensity, confidence reduces.
        """
        intensity = (
            0.45 * self.p.volatility +
            0.25 * self.p.ego +
            0.15 * self.p.competitiveness -
            0.25 * self.p.confidence
        )
        # Map to ~[1.15, 2.0]
        return lerp(1.15, 2.00, clamp01(0.5 + intensity))

    # --------------------------
    # Sampling helpers
    # --------------------------

    def _sample_range(self, lo: float, hi: float, ctx: BehaviorContext) -> float:
        """
        Sample from a distribution within [lo,hi].
        - More volatility => wider variance
        - Confidence narrows variance (more consistent)
        """
        lo = clamp01(lo); hi = clamp01(hi)
        if hi < lo:
            lo, hi = hi, lo

        center = (lo + hi) / 2.0
        width = (hi - lo)

        # Std grows with volatility, shrinks with confidence
        std = 0.10 + 0.25 * self.p.volatility - 0.15 * self.p.confidence
        std = max(0.04, std)

        # Sample around center, then clamp to range
        x = gaussian01(self.rng, mean=center, std=std)
        return clamp01(lerp(lo, hi, clamp01((x - lo) / max(1e-6, (hi - lo)))))

    # --------------------------
    # Public behavior samplers
    # --------------------------

    def sample_contract_aggression(self, ctx: BehaviorContext) -> float:
        """
        Returns a momentary aggression level [0,1] for negotiation.

        Money_focus biases the range upward, BUT:
        - insulting offers
        - losing streak
        - role mismatch
        can push even "coachable" guys to spike.
        """
        # Baseline envelope (range)
        lo = 0.05 + 0.45 * self.p.money_focus - 0.15 * self.p.coachability
        hi = 0.35 + 0.45 * self.p.money_focus + 0.25 * self.p.ego + 0.10 * self.p.confidence

        # Situational shift
        shift = (
            0.35 * (1.0 - ctx.offer_respect) +
            0.20 * ctx.losing_streak +
            0.20 * ctx.role_mismatch +
            0.10 * ctx.market_heat +
            0.10 * ctx.ufa_pressure
            - 0.10 * ctx.team_success
        )

        value = self._sample_range(lo, hi, ctx) + (shift - 0.25)

        # Rare snap: sudden "agent goes nuclear" moment
        if chance(self.rng, self._snap_probability(ctx)):
            value = clamp01(value * self._snap_multiplier(ctx) + 0.10)

        return clamp01(value)

    def sample_trade_request_intent(self, ctx: BehaviorContext) -> float:
        """
        [0,1] intent to request a trade.

        Loyal players usually low, but:
        - long losing streak
        - repeated scratches
        - major role mismatch
        can still produce "I want out" stories.
        """
        lo = 0.02 + 0.35 * (1.0 - self.p.loyalty) + 0.15 * self.p.risk_tolerance
        hi = 0.20 + 0.45 * (1.0 - self.p.loyalty) + 0.30 * self.p.ego + 0.15 * self.p.ambition

        shift = (
            0.35 * ctx.losing_streak +
            0.30 * ctx.role_mismatch +
            0.25 * ctx.scratched_recently +
            0.15 * (1.0 - ctx.ice_time_satisfaction) +
            0.10 * (1.0 - ctx.team_success)
            - 0.20 * self.p.patience
            - 0.10 * self.p.leadership
        )

        value = self._sample_range(lo, hi, ctx) + (shift - 0.20)

        # Snap: "even the loyal guy is done"
        if chance(self.rng, self._snap_probability(ctx) * 0.75):
            value = clamp01(value * self._snap_multiplier(ctx) + 0.15)

        return clamp01(value)

    def sample_early_retirement_thought(self, ctx: BehaviorContext) -> float:
        """
        [0,1] momentary retirement *thought* (not the final retirement decision).
        This should be used by retirement_engine.py along with hard constraints.

        Family-first can drift higher, but:
        - injury burden
        - mental fatigue from losing
        can make *anyone* consider it (rarely).
        """
        lo = 0.00 + 0.35 * self.p.family_priority + 0.15 * self.p.stability_need - 0.20 * self.p.ambition
        hi = 0.08 + 0.50 * self.p.family_priority + 0.25 * ctx.age_factor + 0.20 * ctx.injury_burden

        shift = (
            0.35 * ctx.injury_burden +
            0.20 * ctx.family_event +
            0.15 * ctx.losing_streak +
            0.10 * ctx.rebuild_mode
            - 0.15 * self.p.legacy_drive
            - 0.10 * self.p.competitiveness
            - 0.10 * ctx.cup_satisfaction
        )

        value = self._sample_range(lo, hi, ctx) + (shift - 0.15)

        # Snap here should be VERY rare; scale down heavily
        if chance(self.rng, self._snap_probability(ctx) * 0.25):
            value = clamp01(value + 0.20 + 0.20 * self.p.volatility)

        return clamp01(value)

    def sample_morale_reaction(self, ctx: BehaviorContext) -> float:
        """
        [0,1] how strongly morale reacts to recent events this tick.
        Higher = bigger swing (good or bad depending on outcome).
        """
        lo = 0.15 + 0.45 * self.p.volatility + 0.20 * self.p.ego
        hi = 0.40 + 0.50 * self.p.volatility + 0.25 * self.p.competitiveness

        # Poor team success + losing streak = stronger negative reaction
        shift = (
            0.25 * ctx.losing_streak +
            0.20 * (1.0 - ctx.team_success) +
            0.15 * (1.0 - ctx.ice_time_satisfaction)
            - 0.20 * self.p.confidence
            - 0.10 * self.p.leadership
        )

        value = self._sample_range(lo, hi, ctx) + (shift - 0.10)

        if chance(self.rng, self._snap_probability(ctx) * 0.5):
            value = clamp01(value * self._snap_multiplier(ctx))

        return clamp01(value)

    def sample_locker_room_impact(self, ctx: BehaviorContext) -> float:
        """
        [0,1] positive impact on locker room this tick.
        Can dip if volatility/ego spike under stress.
        """
        lo = 0.15 + 0.55 * self.p.leadership + 0.25 * self.p.coachability
        hi = 0.35 + 0.60 * self.p.leadership + 0.25 * self.p.confidence

        stress_drag = (
            0.25 * ctx.losing_streak +
            0.20 * ctx.role_mismatch +
            0.15 * ctx.scratched_recently
        )

        value = self._sample_range(lo, hi, ctx) - 0.20 * self.p.volatility * stress_drag - 0.15 * self.p.ego * stress_drag

        if chance(self.rng, self._snap_probability(ctx) * 0.35):
            value = clamp01(value - 0.25 * self._snap_multiplier(ctx))

        return clamp01(value)


# --------------------------------------------------
# Factory (Multi-Archetype Mixing)
# --------------------------------------------------

class PersonalityFactory:
    """
    Archetypes here only shape BASE TRAITS.
    Behavior is NEVER hardcoded by archetype.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate(
        self,
        archetypes: Optional[Iterable[PersonalityArchetype]] = None,
        *,
        seed: Optional[int] = None
    ) -> PersonalityProfile:
        rng = random.Random(seed) if seed is not None else self.rng

        # Wider creation range for generated players (more variety)
        traits: Dict[str, float] = dict(
            loyalty=rng.uniform(0.05, 0.95),
            ambition=rng.uniform(0.05, 0.95),
            money_focus=rng.uniform(0.05, 0.95),
            family_priority=rng.uniform(0.05, 0.95),
            legacy_drive=rng.uniform(0.05, 0.95),

            risk_tolerance=rng.uniform(0.05, 0.95),
            adaptability=rng.uniform(0.05, 0.95),
            patience=rng.uniform(0.05, 0.95),
            stability_need=rng.uniform(0.05, 0.95),

            ego=rng.uniform(0.05, 0.95),
            confidence=rng.uniform(0.05, 0.95),
            volatility=rng.uniform(0.05, 0.95),
            competitiveness=rng.uniform(0.05, 0.95),

            leadership=rng.uniform(0.05, 0.95),
            coachability=rng.uniform(0.05, 0.95),
            media_comfort=rng.uniform(0.05, 0.95),
            introversion=rng.uniform(0.0, 1.0),
        )

        # Apply multiple archetypes as SOFT BIASES
        if archetypes:
            for arch in archetypes:
                self._apply_archetype(traits, arch)

        # Small imperfection wobble
        for k in traits:
            traits[k] = clamp01(traits[k] + rng.uniform(-0.05, 0.05))

        return PersonalityProfile(**traits)

    def _apply_archetype(self, t: Dict[str, float], a: PersonalityArchetype) -> None:
        def bump(key: str, delta: float) -> None:
            t[key] = clamp01(t[key] + delta)

        # These only bias baseline traits. They DO NOT define behavior.
        if a == PersonalityArchetype.LOYALIST:
            bump("loyalty", +0.25)
            bump("risk_tolerance", -0.10)

        elif a == PersonalityArchetype.FAMILY_FIRST:
            bump("family_priority", +0.30)
            bump("stability_need", +0.20)

        elif a == PersonalityArchetype.MONEY_HUNGRY:
            bump("money_focus", +0.30)
            bump("loyalty", -0.10)

        elif a == PersonalityArchetype.COMPETITOR:
            bump("competitiveness", +0.30)
            bump("ambition", +0.20)

        elif a == PersonalityArchetype.PROFESSIONAL:
            bump("coachability", +0.25)
            bump("volatility", -0.20)

        elif a == PersonalityArchetype.JOURNEYMAN:
            bump("adaptability", +0.30)
            bump("ego", -0.20)

        elif a == PersonalityArchetype.STAR:
            bump("ego", +0.25)
            bump("confidence", +0.20)
            bump("media_comfort", +0.20)

        elif a == PersonalityArchetype.CHIP_ON_SHOULDER:
            bump("ego", +0.30)
            bump("volatility", +0.25)
            bump("confidence", -0.10)

        elif a == PersonalityArchetype.LEADER:
            bump("leadership", +0.30)
            bump("coachability", +0.15)

        elif a == PersonalityArchetype.INTROVERT:
            bump("introversion", +0.35)
            bump("media_comfort", -0.15)

        elif a == PersonalityArchetype.EXTROVERT:
            bump("introversion", -0.35)
            bump("media_comfort", +0.15)

        elif a == PersonalityArchetype.RISK_TAKER:
            bump("risk_tolerance", +0.30)
            bump("stability_need", -0.20)

        elif a == PersonalityArchetype.STABILITY_SEEKER:
            bump("stability_need", +0.30)
            bump("risk_tolerance", -0.20)
