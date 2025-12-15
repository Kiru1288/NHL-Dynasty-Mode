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


class SimEngine:
    """
    TEMP TEST ENGINE

    Purpose:
    - Validate personality behavior
    - Validate AIManager intent signals
    - Validate MoraleEngine dynamics
    """

    def __init__(self, seed: int | None = None):
        self.year = 0
        self.rng = random.Random(seed)

        self.ai_manager = AIManager(self.rng)
        self.morale_engine = MoraleEngine()

        factory = PersonalityFactory(self.rng)

        self.personality = factory.generate(
            archetypes=[
                PersonalityArchetype.LOYALIST,
                PersonalityArchetype.MONEY_HUNGRY,
            ]
        )

        self.behavior = PersonalityBehavior(self.personality, self.rng)
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

        signals = self.ai_manager.evaluate_player(
            behavior=self.behavior,
            ctx=ctx,
        )

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
        print(f"Flags: {self.morale_engine.narrative_flags(self.morale)}")

    def sim_years(self, years: int = 5):
        for _ in range(years):
            self.sim_year()
            time.sleep(0.1)
