from __future__ import annotations

from dataclasses import fields, asdict

from typing import Dict
import random
import time

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

        # Randomness (controlled chaos)
        self.randomness = RandomnessEngine(self.rng)

        # --------------------------------------------------
        # Personality
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

        # --------------------------------------------------
        # Life pressure (off-ice human factors)
        # --------------------------------------------------
        self.pressure: Dict[str, float] = {
            "career_identity": 0.0,
            "health": 0.0,
            "family": 0.0,
            "psychological": 0.0,
            "security": 0.0,
            "environment": 0.0,
        }
        self.base_pressure_decay = 0.12

        print("\n=== TEST PLAYER PERSONALITY PROFILE ===")
        for f in fields(self.personality):
            print(f"{f.name:20s}: {getattr(self.personality, f.name):.2f}")

    # --------------------------------------------------
    # Retirement adapter
    # --------------------------------------------------

    def _build_retirement_player(self):
        class PlayerProxy:
            pass

        p = PlayerProxy()
        p.age = 18 + self.year
        p.personality = self.personality
        p.morale = max(0.0, min(1.0, self.morale.overall()))

        total_risk = float(getattr(self.injury_risk, "total_risk", 0.0) or 0.0)
        physical = float(getattr(self.injury_risk, "physical_risk", 0.0) or 0.0)
        fatigue = float(getattr(self.injury_risk, "fatigue_risk", 0.0) or 0.0)

        p.injury_wear = min(1.0, total_risk)
        p.career_injury_score = min(1.0, total_risk)
        p.chronic_injuries = int((physical + fatigue) * 2)
        p.durability = max(0.0, 1.0 - total_risk)

        # Snapshot of off-ice life pressure
        p.life_pressure = self.pressure.copy()
        return p

    # --------------------------------------------------
    # Life pressure update
    # --------------------------------------------------

    def _update_life_pressure(self, ctx: BehaviorContext):
        # Stress sensitivity (personality-weighted)
        sensitivity = self.randomness.stress_sensitivity(self.personality)

        self.pressure["family"] += ctx.family_event * 0.8 * sensitivity
        self.pressure["psychological"] += (
            (1.0 - ctx.ice_time_satisfaction) * 0.2
            + ctx.losing_streak * 0.3
        ) * sensitivity
        self.pressure["career_identity"] += (
            ctx.role_mismatch * 0.4
            + (0.5 - ctx.team_success) * 0.2
        ) * sensitivity
        self.pressure["environment"] += ctx.market_heat * 0.15 * sensitivity
        self.pressure["health"] += ctx.injury_burden * 0.6 * sensitivity

        # --------------------------------------------------
        # Rare life events (from randomness engine)
        # --------------------------------------------------
        events = self.randomness.roll_life_events(
            year=self.year,
            age=18 + self.year,
            personality=self.personality,
            context={
                "morale": self.morale.overall(),
            },
        )

        for domain, delta in events.items():
            if domain in self.pressure:
                self.pressure[domain] += delta

        # --------------------------------------------------
        # Clamp + decay (recovery variance)
        # --------------------------------------------------
        recovery = self.randomness.recovery_modifier(self.personality)

        for k in self.pressure:
            self.pressure[k] = max(0.0, min(1.0, self.pressure[k]))
            self.pressure[k] *= (1.0 - self.base_pressure_decay * recovery)

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

        # --------------------------------------------------
        # Apply controlled randomness to context
        # --------------------------------------------------
        ctx_dict = self.randomness.apply_context_noise(asdict(ctx), self.personality)

        ctx = BehaviorContext(**ctx_dict)


        # AI decision signals
        self.ai_manager.evaluate_player(
            behavior=self.behavior,
            ctx=ctx,
        )

        # --------------------------------------------------
        # Morale
        # --------------------------------------------------
        circumstances = {
            "team_results": {
                "intensity": team_success - 0.5,
                "axes": {"confidence": 0.6, "motivation": 0.4},
                "half_life": 25,
            },
            "role_fit": {
                "intensity": -ctx.role_mismatch,
                "axes": {"confidence": 0.4, "trust": 0.3},
                "half_life": 30,
            },
            "ice_time": {
                "intensity": ctx.ice_time_satisfaction - 0.5,
                "axes": {"confidence": 0.4, "belonging": 0.3},
                "half_life": 15,
            },
            "life_stress": {
                "intensity": -(ctx.injury_burden + ctx.family_event),
                "axes": {"stability": 0.7, "motivation": 0.3},
                "half_life": 40,
            },
        }

        self.morale_engine.add_circumstances(self.morale, circumstances)
        self.morale_engine.update(
            self.morale,
            personality=self.personality,
            behavior=self.behavior,
            ctx=ctx,
        )

        # --------------------------------------------------
        # Career arc & injuries
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Life pressure
        # --------------------------------------------------
        self._update_life_pressure(ctx)

        # --------------------------------------------------
        # Retirement
        # --------------------------------------------------
        player = self._build_retirement_player()

        retirement_ctx = {
            "career_ending_injury": self.injury_risk.total_risk > 0.85,
            "recent_major_injury": (
                self.injury_risk.physical_risk > 0.6
                or self.injury_risk.fatigue_risk > 0.6
            ),
            "usage_heavy": ctx.age_factor,
            "healthy_scratches": int(ctx.scratched_recently * 82),
            "family_event": self.pressure["family"] > 0.55,
            "questioning_identity": (
                self.career_arc.questioning_identity
                or self.pressure["career_identity"] > 0.6
            ),
            "mental_fatigue": self.pressure["psychological"],
            "no_offers": ctx.offer_respect < 0.35 and self.year > 26,
            "career_earnings": self.year * 3_500_000,
        }

        decision = self.retirement_engine.evaluate_player(player, retirement_ctx)

        print("\n--- RETIREMENT CHECK ---")
        print(f"Age           : {player.age}")
        print(f"Retire Chance : {decision.retire_chance:.6f}")
        print(f"Net Score     : {decision.net_score:.2f}")
        print(f"Reason        : {decision.primary_reason}")
        print(f"Tags          : {decision.tags}")

        if decision.considering:
            print("⚠ Considering retirement")

        if decision.retired:
            self.retired = True
            print("\n🛑 PLAYER HAS RETIRED 🛑")
            print(f"Final Age: {player.age}")
            print(f"Reason  : {decision.primary_reason}")

    # --------------------------------------------------
    # Multi-year
    # --------------------------------------------------

    def sim_years(self, years: int = 5):
        for _ in range(years):
            if self.retired:
                break
            self.sim_year()
            time.sleep(0.05)
