from dataclasses import fields

from app.sim_engine.random_manager import RandomManager
from app.sim_engine.generation.draft_class_generator import generate_draft_class
from app.sim_engine.progression.aging_curves import age_player
from app.sim_engine.progression.retirement import check_retirement

from app.sim_engine.ai.personality import (
    PersonalityFactory,
    PersonalityArchetype,
    PersonalityBehavior,
)

from app.sim_engine.ai.ai_manager import AIManager


class SimEngine:
    def __init__(self, seed=None):
        self.year = 0
        self.rng = RandomManager(seed)
        self.players = []

        # --------------------------------------------------
        # AI Manager (shared coordinator)
        # --------------------------------------------------
        self.ai_manager = AIManager(self.rng.random)

        # --------------------------------------------------
        # TEMP: Personality test subject (NOT a real player)
        # --------------------------------------------------
        factory = PersonalityFactory(self.rng.random)

        self.test_personality = factory.generate(
            archetypes=[
                PersonalityArchetype.LOYALIST,
                PersonalityArchetype.MONEY_HUNGRY,
            ]
        )

        self.test_behavior = PersonalityBehavior(
            self.test_personality,
            self.rng.random
        )

        print("\n=== TEST PLAYER PERSONALITY ===")
        for f in fields(self.test_personality):
            value = getattr(self.test_personality, f.name)
            print(f"{f.name:20s}: {value:.2f}")

    def sim_year(self):
        self.year += 1
        print(f"\n=== Simulating Year {self.year} ===")

        print("\n--- AI MANAGER EVALUATION ---")

        # --------------------------------------------------
        # Build realistic context THROUGH AIManager
        # --------------------------------------------------

        team_success = self.rng.random.random()

        ctx = self.ai_manager.build_behavior_context(
            team_success=team_success,
            rebuild_mode=0.3 if team_success < 0.4 else 0.0,

            role_mismatch=self.rng.random.random() * 0.4,
            ice_time_satisfaction=0.6 + self.rng.random.random() * 0.4,

            offer_respect=0.6 + self.rng.random.random() * 0.4,
            ufa_pressure=min(1.0, self.year / 7.0),
            market_heat=0.4 + self.rng.random.random() * 0.3,

            injury_burden=self.rng.random.random() * 0.25,
            family_event=0.1 if self.year in (4, 7) else 0.0,

            age_factor=min(1.0, self.year / 20.0),
            cup_satisfaction=0.0,
        )

        # --------------------------------------------------
        # Evaluate player via AIManager
        # --------------------------------------------------

        signals = self.ai_manager.evaluate_player(
            behavior=self.test_behavior,
            ctx=ctx,
        )

        # --------------------------------------------------
        # Display AI signals (INTENT, not decisions)
        # --------------------------------------------------

        print(f"Contract Aggression : {signals.contract_aggression:.2f}")
        print(f"Trade Request Intent: {signals.trade_request_intent:.2f}")
        print(f"Retirement Thought  : {signals.retirement_thought:.2f}")
        print(f"Morale Reaction     : {signals.morale_reaction:.2f}")
        print(f"Locker Room Impact  : {signals.locker_room_impact:.2f}")

        # --------------------------------------------------
        # Normal simulation flow
        # --------------------------------------------------

        for p in self.players:
            age_player(p)
            check_retirement(p, self.rng)

        retired = [p for p in self.players if p.retired]
        self.players = [p for p in self.players if not p.retired]

        draft_class = generate_draft_class(self.rng)
        self.players.extend(draft_class)

        print(f"\nDrafted {len(draft_class)} players")
        print(f"Retired {len(retired)} players")
        print(f"Active players: {len(self.players)}")

    def sim_years(self, years=1):
        for _ in range(years):
            self.sim_year()
