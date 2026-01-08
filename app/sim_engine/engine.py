from __future__ import annotations

from dataclasses import asdict
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
# Entities
# -------------------------------
from app.sim_engine.entities.player import Player
from app.sim_engine.entities.team import Team


class SimEngine:
    """
    Simulation driver for a single NHL player career.

    This engine now:
    - Feeds real season outcomes into team.py
    - Allows teams to evolve, panic, stabilize, or collapse
    - Prevents immortal careers caused by static org context
    """

    def __init__(self, seed: int | None = None):
        self.year = 0
        self.rng = random.Random(seed)
        self.retired = False

        # Core systems
        self.ai_manager = AIManager(self.rng)
        self.morale_engine = MoraleEngine()
        self.career_arc_engine = CareerArcEngine()
        self.injury_risk_engine = InjuryRiskEngine()
        self.retirement_engine = RetirementEngine(seed=seed)
        self.randomness = RandomnessEngine(self.rng)

        # Injected entities
        self.player: Player | None = None
        self.team: Team | None = None

        # Personality (AI-side)
        factory = PersonalityFactory(self.rng)
        self.personality = factory.generate(
            archetypes=[
                PersonalityArchetype.LOYALIST,
                PersonalityArchetype.MONEY_HUNGRY,
            ]
        )
        self.behavior = PersonalityBehavior(self.personality, self.rng)

        # State
        self.morale: MoraleState = self.morale_engine.create_state()
        self.career_arc = self.career_arc_engine.create_state()
        self.injury_risk = self.injury_risk_engine.create_state()

    # --------------------------------------------------
    # Injection
    # --------------------------------------------------

    def set_player(self, player: Player):
        self.player = player

    def set_team(self, team: Team):
        self.team = team
        team.add_player(self.player)

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
    # Career stage (purely descriptive)
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
    # FULL DEBUG DUMP
    # --------------------------------------------------

    def _debug_dump_year(self, *, ctx, decision, team_ctx, win_pct):
        print("\n================ PLAYER STATE DUMP ================")

        print("\n[IDENTITY]")
        print(f"Name        : {self.player.name}")
        print(f"Age         : {self.player.age}")
        print(f"Position    : {self.player.position.value}")
        print(f"Shoots      : {self.player.shoots.value}")
        print(f"Team        : {self.team.city} {self.team.name}")
        print(f"Team Style  : {self.team.archetype}")

        print("\n[OVR / ATTRIBUTES]")
        print(f"OVR         : {self.player.ovr():.3f}")
        for g, v in self.player.group_averages().items():
            print(f"{g:22s}: {v:.3f}")

        print("\n[PERSONALITY TRAITS]")
        for k, v in vars(self.player.traits).items():
            print(f"{k:22s}: {v:.3f}")

        print("\n[LIFE PRESSURE]")
        for k, v in vars(self.player.life_pressure).items():
            print(f"{k:22s}: {v:.3f}")

        print("\n[HEALTH]")
        print(f"Wear & Tear         : {self.player.health.wear_and_tear:.3f}")
        print(f"Chronic Injuries    : {len(self.player.health.chronic_flags)}")

        print("\n[MORALE]")
        print(f"Overall Morale      : {self.morale.overall():.3f}")
        for axis, val in self.morale.axes.items():
            print(f"{axis:22s}: {val:.3f}")

        print("\n[CAREER ARC]")
        for k, v in vars(self.career_arc).items():
            if isinstance(v, float):
                print(f"{k:22s}: {v:.3f}")

        print("\n[INJURY RISK]")
        print(f"Total Risk          : {self.injury_risk.total_risk:.3f}")

        print("\n[TEAM CONTEXT]")
        print(f"Season Win %        : {win_pct:.3f}")
        for k, v in team_ctx.items():
            if isinstance(v, float):
                print(f"{k:22s}: {v:.3f}")
            else:
                print(f"{k:22s}: {v}")

        print("\n[BEHAVIOR CONTEXT]")
        for k, v in asdict(ctx).items():
            print(f"{k:22s}: {v:.3f}")

        print("\n[RETIREMENT]")
        print(f"Retire Chance       : {decision.retire_chance:.6f}")
        print(f"Net Score           : {decision.net_score:.3f}")
        print(f"Primary Reason      : {decision.primary_reason}")

        print("\n[CAREER STAGE]")
        print(self._derive_career_stage())

        print("===================================================")

    # --------------------------------------------------
    # Single season
    # --------------------------------------------------

    def sim_year(self):
        if self.retired or self.player is None or self.team is None:
            return

        self.year += 1

        print("\n==============================")
        print(f"      SIM YEAR {self.year}")
        print("==============================")

        # --------------------------------------------------
        # 1. Simulate TEAM SEASON RESULT (abstract)
        # --------------------------------------------------
        # Expected win % comes from team internals
        expected = self.team._expected_win_pct()

        # Noise + chaos (bad orgs swing more)
        chaos = (1.0 - self.team.state.stability) * self.rng.uniform(-0.10, 0.10)
        luck = self.rng.uniform(-0.06, 0.06)

        win_pct = max(0.25, min(0.75, expected + chaos + luck))

        # --------------------------------------------------
        # 2. Update TEAM STATE (CRITICAL)
        # --------------------------------------------------
        self.team.update_team_state(win_pct=win_pct)

        team_ctx = self.team.team_context_for_player(self.player)

        rebuild_mode = team_ctx["rebuild_mode"]

        # --------------------------------------------------
        # 3. Build BEHAVIOR CONTEXT
        # --------------------------------------------------
        ctx = BehaviorContext(
            team_success=win_pct,
            losing_streak=max(0.0, 0.5 - win_pct),
            rebuild_mode=rebuild_mode,
            role_mismatch=team_ctx["role_mismatch"],
            ice_time_satisfaction=0.35 + self.rng.random() * 0.55,
            scratched_recently=0.0,
            offer_respect=team_ctx["stability"],
            ufa_pressure=min(1.0, self.year / 7.0),
            market_heat=team_ctx.get("market_pressure", 0.5),
            injury_burden=self.injury_risk.total_risk,
            family_event=0.15 if self.year in (4, 7, 12, 18, 25) else 0.0,
            age_factor=min(1.0, self.player.age / 35.0),
            cup_satisfaction=0.0,
        )

        ctx = BehaviorContext(
            **self.randomness.apply_context_noise(asdict(ctx), self.personality)
        )

        # --------------------------------------------------
        # 4. AI + PSYCHOLOGY
        # --------------------------------------------------
        self.ai_manager.evaluate_player(
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

        # --------------------------------------------------
        # 5. PLAYER AGING / DEVELOPMENT
        # --------------------------------------------------
        self.player.advance_year(
            season_morale=self.morale.overall(),
            season_injury_risk=self.injury_risk.total_risk,
            team_instability=1.0 - team_ctx["stability"],
        )

        # --------------------------------------------------
        # 6. RETIREMENT CHECK
        # --------------------------------------------------
        decision = self.retirement_engine.evaluate_player(
            self._build_retirement_player(), {}
        )

        # --------------------------------------------------
        # 7. DEBUG OUTPUT
        # --------------------------------------------------
        self._debug_dump_year(
            ctx=ctx,
            decision=decision,
            team_ctx=team_ctx,
            win_pct=win_pct,
        )

        if decision.retired:
            self.retired = True
            self.player.retired = True
            self.player.retirement_reason = decision.primary_reason
            print("\n PLAYER HAS RETIRED ")

    # --------------------------------------------------
    # Multi-year
    # --------------------------------------------------

    def sim_years(self, years: int = 40):
        for _ in range(years):
            if self.retired:
                break
            self.sim_year()
            time.sleep(0.03)
