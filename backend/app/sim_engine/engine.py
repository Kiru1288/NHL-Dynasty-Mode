from __future__ import annotations

from dataclasses import fields
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
<<<<<<< HEAD
=======
from app.sim_engine.ai.career_arc import CareerArcEngine
from app.sim_engine.ai.injury_risk import InjuryRiskEngine
>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)


class SimEngine:
    """
    TEMP TEST ENGINE

    Purpose:
    - Validate personality behavior
    - Validate AIManager intent signals
    - Validate MoraleEngine dynamics
<<<<<<< HEAD
=======
    - Validate CareerArcEngine long-horizon pressures
    - Validate InjuryRiskEngine latent injury-proneness

    NOTE:
    - No narratives
    - No forced decisions
    - All systems run side-by-side
>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)
    """

    def __init__(self, seed: int | None = None):
        self.year = 0
        self.rng = random.Random(seed)

<<<<<<< HEAD
        self.ai_manager = AIManager(self.rng)
        self.morale_engine = MoraleEngine()

=======
        # -------------------------------
        # Core Systems
        # -------------------------------
        self.ai_manager = AIManager(self.rng)
        self.morale_engine = MoraleEngine()
        self.career_arc_engine = CareerArcEngine()
        self.injury_risk_engine = InjuryRiskEngine()

        # -------------------------------
        # Personality
        # -------------------------------
>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)
        factory = PersonalityFactory(self.rng)

        self.personality = factory.generate(
            archetypes=[
                PersonalityArchetype.LOYALIST,
                PersonalityArchetype.MONEY_HUNGRY,
            ]
        )

        self.behavior = PersonalityBehavior(self.personality, self.rng)
<<<<<<< HEAD
        self.morale: MoraleState = self.morale_engine.create_state()

        print("\n=== TEST PLAYER PERSONALITY PROFILE ===")
        for f in fields(self.personality):
            print(f"{f.name:20s}: {getattr(self.personality, f.name):.2f}")

    def sim_year(self):
        self.year += 1
        print(f"\n==============================")
        print(f"      SIM YEAR {self.year}")
        print(f"==============================")

        team_success = self.rng.random()

=======

        # -------------------------------
        # State
        # -------------------------------
        self.morale: MoraleState = self.morale_engine.create_state()
        self.career_arc = self.career_arc_engine.create_state()
        self.injury_risk = self.injury_risk_engine.create_state()

        # -------------------------------
        # Debug: Personality Print
        # -------------------------------
        print("\n=== TEST PLAYER PERSONALITY PROFILE ===")
        for f in fields(self.personality):
            print(f"{f.name:20s}: {getattr(self.personality, f.name):.2f}")

    # --------------------------------------------------
    # Single Season Simulation
    # --------------------------------------------------

    def sim_year(self):
        self.year += 1

        print(f"\n==============================")
        print(f"      SIM YEAR {self.year}")
        print(f"==============================")

        # -------------------------------
        # TEMP CONTEXT (placeholder only)
        # -------------------------------
        team_success = self.rng.random()

>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)
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
            family_event=0.15 if self.year in (4, 7) else 0.0,
            age_factor=min(1.0, self.year / 18.0),
            cup_satisfaction=0.0,
        )

<<<<<<< HEAD
=======
        # -------------------------------
        # AI Signals (intent only)
        # -------------------------------
>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)
        signals = self.ai_manager.evaluate_player(
            behavior=self.behavior,
            ctx=ctx,
        )

<<<<<<< HEAD
=======
        # -------------------------------
        # TEMP Morale Circumstances
        # (placeholder — narratives later)
        # -------------------------------
>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)
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

<<<<<<< HEAD
=======
        # -------------------------------
        # Career Arc Update
        # -------------------------------
        self.career_arc_engine.update(
            self.career_arc,
            personality=self.personality,
            morale_axes=self.morale.axes,
        )

        # -------------------------------
        # Injury Risk Update (latent)
        # -------------------------------
        self.injury_risk_engine.update(
            self.injury_risk,
            personality=self.personality,
            morale_axes=self.morale.axes,
            career=self.career_arc,
        )

        # -------------------------------
        # DEBUG OUTPUT
        # -------------------------------
>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)
        print("\n--- AI SIGNALS ---")
        print(f"Contract Aggression : {signals.contract_aggression:.2f}")
        print(f"Trade Request Intent: {signals.trade_request_intent:.2f}")
        print(f"Retirement Thought  : {signals.retirement_thought:.2f}")
        print(f"Morale Reactivity   : {signals.morale_reaction:.2f}")
        print(f"Locker Room Impact  : {signals.locker_room_impact:.2f}")

        print("\n--- MORALE AXES ---")
        for k, v in self.morale.axes.items():
            print(f"{k:12s}: {v:.2f}")

        print(f"\nOverall Morale: {self.morale.overall():.2f}")
<<<<<<< HEAD
        print(f"Flags: {self.morale_engine.narrative_flags(self.morale)}")

=======
        print(f"Morale Flags: {self.morale_engine.narrative_flags(self.morale)}")

        print("\n--- CAREER ARC ---")
        print(self.career_arc_engine.summary(self.career_arc))

        print("\n--- INJURY RISK ---")
        print(self.injury_risk_engine.summary(self.injury_risk))

    # --------------------------------------------------
    # Multi-Year Simulation
    # --------------------------------------------------

>>>>>>> 3307d89 (Add career arc and injury risk engines with ambient wear-and-tear)
    def sim_years(self, years: int = 5):
        for _ in range(years):
            self.sim_year()
            time.sleep(0.1)
