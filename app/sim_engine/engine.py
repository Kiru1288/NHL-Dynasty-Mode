from __future__ import annotations

from dataclasses import fields, asdict
from typing import Dict
import random
import time

# -------------------------------
# AI systems
# -------------------------------
from app.sim_engine.ai.personality import (
    PersonalityFactory,
    PersonalityArchetype,
    PersonalityBehavior,
    BehaviorContext,
)

from app.sim_engine.ai.ai_manager import AIManager
from app.sim_engine.ai.morale_engine import MoraleEngine, MoraleState
from app.sim_engine.ai.career_arc import CareerArcEngine
from app.sim_engine.ai.injury_risk import InjuryRiskEngine
from app.sim_engine.ai.retirement_engine import RetirementEngine
from app.sim_engine.ai.randomness import RandomnessEngine

# -------------------------------
# Player entity
# -------------------------------
from app.sim_engine.entities.player import (
    Player,
    IdentityBio,
    BackstoryUpbringing,
    PersonalityTraits,
    CareerArcSeeds,
    Position,
    Shoots,
    BackstoryType,
    UpbringingType,
    SupportLevel,
    PressureLevel,
    DevResources,
)


class SimEngine:
    """
    Simulation driver for a single NHL player career.
    """

    def __init__(self, seed: int | None = None):
        self.year = 0
        self.rng = random.Random(seed)
        self.retired = False

        # --------------------------------------------------
        # Core systems
        # --------------------------------------------------
        self.ai_manager = AIManager(self.rng)
        self.morale_engine = MoraleEngine()
        self.career_arc_engine = CareerArcEngine()
        self.injury_risk_engine = InjuryRiskEngine()
        self.retirement_engine = RetirementEngine(seed=seed)
        self.randomness = RandomnessEngine(self.rng)

        # --------------------------------------------------
        # Player
        # --------------------------------------------------
        self.player = self._create_test_player(seed)

        # --------------------------------------------------
        # Personality (AI-side)
        # --------------------------------------------------
        factory = PersonalityFactory(self.rng)
        self.personality = factory.generate(
            archetypes=[
                PersonalityArchetype.LOYALIST,
                PersonalityArchetype.MONEY_HUNGRY,
            ]
        )
        self.behavior = PersonalityBehavior(self.personality, self.rng)

        # --------------------------------------------------
        # State
        # --------------------------------------------------
        self.morale: MoraleState = self.morale_engine.create_state()
        self.career_arc = self.career_arc_engine.create_state()
        self.injury_risk = self.injury_risk_engine.create_state()

        print("\n=== TEST PLAYER CREATED ===")
        print(self.player)

        print("\n=== TEST PLAYER PERSONALITY PROFILE ===")
        for f in fields(self.personality):
            print(f"{f.name:22s}: {getattr(self.personality, f.name):.2f}")

    # --------------------------------------------------
    # Player factory
    # --------------------------------------------------

    def _create_test_player(self, seed: int | None) -> Player:
        rng = random.Random(seed)

        identity = IdentityBio(
            name="Test Player",
            age=18,
            birth_year=2007,
            birth_country="Canada",
            birth_city="Toronto",
            height_cm=183,
            weight_kg=86,
            position=Position.C,
            shoots=Shoots.L,
            draft_year=2025,
            draft_round=1,
            draft_pick=12,
        )

        backstory = BackstoryUpbringing(
            backstory=BackstoryType.GRINDER,
            upbringing=UpbringingType.ROUGH,
            family_support=SupportLevel.MEDIUM,
            early_pressure=PressureLevel.MODERATE,
            dev_resources=DevResources.LOCAL,
        )

        traits = PersonalityTraits(
            loyalty=0.75,
            ambition=0.30,
            money_focus=0.55,
            family_priority=0.35,
            legacy_drive=0.70,
            risk_tolerance=0.50,
            adaptability=0.70,
            patience=0.25,
            stability_need=0.45,
            ego=0.10,
            confidence=0.35,
            volatility=0.45,
            competitiveness=0.20,
            leadership=0.30,
            coachability=0.65,
            media_comfort=0.50,
            introversion=0.30,
            work_ethic=0.80,
            mental_toughness=0.75,
            clutch_tendency=0.40,
        )

        career = CareerArcSeeds(
            expected_peak_age=28,
            decline_rate=0.45,
            breakout_probability=0.18,
            bust_probability=0.10,
            prime_duration=0.55,
            season_consistency=0.50,
            dev_curve_seed=rng.randint(1, 999999),
            regression_resistance=0.60,
            ceiling_floor_gap=0.40,
        )

        return Player(
            identity=identity,
            backstory=backstory,
            ratings={},
            traits=traits,
            career=career,
            rng_seed=seed,
        )

    # --------------------------------------------------
    # Retirement adapter
    # --------------------------------------------------

    def _build_retirement_player(self):
        class PlayerProxy:
            pass

        p = PlayerProxy()
        p.age = self.player.age
        p.personality = self.personality
        p.morale = self.morale.overall()

        p.injury_wear = self.player.health.wear_and_tear
        p.career_injury_score = self.player.health.wear_and_tear
        p.chronic_injuries = len(self.player.health.chronic_flags)
        p.durability = 1.0 - self.player.health.wear_and_tear

        p.life_pressure = asdict(self.player.life_pressure)
        p.ovr = self.player.ovr()

        return p

    # --------------------------------------------------
    # Career stage
    # --------------------------------------------------

    def _derive_career_stage(self) -> str:
        age = self.player.age
        if age < 22:
            return "Prospect / Development"
        if age < 26:
            return "Young NHL Regular"
        if age < 30:
            return "Prime Years"
        if age < 34:
            return "Late Prime / Early Decline"
        if age < 38:
            return "Veteran"
        if age < 42:
            return "Aging Veteran"
        return "Fringe / Retirement Risk"

    # --------------------------------------------------
    # DEBUG
    # --------------------------------------------------

    def _debug_dump_year(self, *, ctx, signals, decision):
        print("\n================ PLAYER STATE DUMP ================")

        print("\n[IDENTITY]")
        print(f"Name        : {self.player.name}")
        print(f"Age         : {self.player.age}")
        print(f"Position    : {self.player.position.value}")
        print(f"Shoots      : {self.player.shoots.value}")

        print("\n[OVR / ATTRIBUTES]")
        print(f"OVR         : {self.player.ovr():.3f}")
        for g, v in self.player.group_averages().items():
            print(f"{g:22s}: {v:.3f}")

        print("\n[TRAITS]")
        for k, v in vars(self.player.traits).items():
            print(f"{k:22s}: {v:.3f}")

        print("\n[LIFE PRESSURE]")
        for k, v in vars(self.player.life_pressure).items():
            print(f"{k:22s}: {v:.3f}")

        print("\n[HEALTH]")
        print(f"Wear & tear : {self.player.health.wear_and_tear:.3f}")

        print("\n[MORALE]")
        print(f"Overall morale: {self.morale.overall():.3f}")

        print("\n[RETIREMENT]")
        print(f"Retire chance : {decision.retire_chance:.6f}")
        print(f"Net score     : {decision.net_score:.3f}")
        print(f"Reason        : {decision.primary_reason}")

        print("\n[CAREER STAGE]")
        print(self._derive_career_stage())
        print("===================================================")

    # --------------------------------------------------
    # Single season
    # --------------------------------------------------

    def sim_year(self):
        if self.retired:
            return

        self.year += 1

        print("\n==============================")
        print(f"      SIM YEAR {self.year}")
        print("==============================")

        team_success = self.rng.random()

        ctx = BehaviorContext(
            team_success=team_success,
            losing_streak=self.rng.random() * (1.0 - team_success),
            rebuild_mode=0.4 if team_success < 0.45 else 0.0,
            role_mismatch=self.rng.random() * 0.5,
            ice_time_satisfaction=0.4 + self.rng.random() * 0.6,
            scratched_recently=self.rng.random() * 0.3,
            offer_respect=0.4 + self.rng.random() * 0.6,
            ufa_pressure=min(1.0, self.year / 7.0),
            market_heat=0.3 + self.rng.random() * 0.4,
            injury_burden=self.rng.random() * 0.3,
            family_event=0.15 if self.year in (4, 7, 12, 18, 25) else 0.0,
            age_factor=min(1.0, self.year / 18.0),
            cup_satisfaction=0.0,
        )

        ctx = BehaviorContext(**self.randomness.apply_context_noise(
            asdict(ctx), self.personality
        ))

        signals = self.ai_manager.evaluate_player(
            behavior=self.behavior,
            ctx=ctx,
        )

        self.morale_engine.update(
            self.morale,
            personality=self.personality,
            behavior=self.behavior,
            ctx=ctx,
        )

        self.career_arc_engine.update(
            self.career_arc,
            personality=self.personality,
            morale_axes=self.morale.axes,
        )

        self.injury_risk_engine.update(
            self.injury_risk,
            personality=self.personality,
            morale_axes=self.morale.axes,
            career=self.career_arc,
        )

        # 🔑 THIS IS THE BIG CHANGE
        self.player.advance_year(
            season_morale=self.morale.overall(),
            season_injury_risk=self.injury_risk.total_risk,
            team_instability=ctx.rebuild_mode,
        )

        proxy = self._build_retirement_player()
        decision = self.retirement_engine.evaluate_player(proxy, {})

        self._debug_dump_year(
            ctx=ctx,
            signals=signals,
            decision=decision,
        )

        if decision.retired:
            self.retired = True
            print("\n PLAYER HAS RETIRED ")

    # --------------------------------------------------
    # Multi-year
    # --------------------------------------------------

    def sim_years(self, years: int = 40):
        for _ in range(years):
            if self.retired:
                break
            self.sim_year()
            time.sleep(0.05)
