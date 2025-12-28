from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from app.sim_engine.ai.personality import (
    PersonalityBehavior,
    BehaviorContext,
    clamp01,
)


# --------------------------------------------------
# AI Output Snapshot (signals, not decisions)
# --------------------------------------------------

@dataclass(slots=True)
class AISignals:
    """
    These are INTENT / PRESSURE signals.
    Other systems (GM AI, contract engine, etc.)
    decide what to do with them.
    """

    contract_aggression: float
    trade_request_intent: float
    retirement_thought: float
    morale_reaction: float
    locker_room_impact: float


# --------------------------------------------------
# AI Manager
# --------------------------------------------------

class AIManager:
    """
    Central coordinator for player AI.

    Responsibilities:
    - Build behavior context
    - Sample personality-driven reactions
    - Return signals (NOT outcomes)

    This keeps AI modular and realistic.
    """

    def __init__(self, rng):
        self.rng = rng

    # --------------------------------------------------
    # Context builder (shared across AI systems)
    # --------------------------------------------------

    def build_behavior_context(
        self,
        *,
        team_success: float,
        rebuild_mode: float,
        role_mismatch: float,
        ice_time_satisfaction: float,
        offer_respect: float,
        ufa_pressure: float,
        market_heat: float,
        injury_burden: float,
        family_event: float,
        age_factor: float,
        cup_satisfaction: float,
    ) -> BehaviorContext:
        """
        Build a correlated, normalized context object.

        The CALLER (engine / season sim) decides inputs.
        AIManager just packages them consistently.
        """

        losing_streak = clamp01((1.0 - team_success) * self.rng.random())
        scratched_recently = self.rng.random() * 0.15

        return BehaviorContext(
            team_success=team_success,
            losing_streak=losing_streak,
            rebuild_mode=rebuild_mode,

            role_mismatch=role_mismatch,
            ice_time_satisfaction=ice_time_satisfaction,
            scratched_recently=scratched_recently,

            offer_respect=offer_respect,
            ufa_pressure=ufa_pressure,
            market_heat=market_heat,

            injury_burden=injury_burden,
            family_event=family_event,

            age_factor=age_factor,
            cup_satisfaction=cup_satisfaction,
        )

    # --------------------------------------------------
    # Main evaluation entry point
    # --------------------------------------------------

    def evaluate_player(
        self,
        *,
        behavior: PersonalityBehavior,
        ctx: BehaviorContext,
    ) -> AISignals:
        """
        Run all personality-driven AI samplers
        and return a snapshot of player intent.
        """

        contract_aggression = behavior.sample_contract_aggression(ctx)
        trade_intent = behavior.sample_trade_request_intent(ctx)
        retirement_thought = behavior.sample_early_retirement_thought(ctx)
        morale_reaction = behavior.sample_morale_reaction(ctx)
        locker_room_impact = behavior.sample_locker_room_impact(ctx)

        return AISignals(
            contract_aggression=contract_aggression,
            trade_request_intent=trade_intent,
            retirement_thought=retirement_thought,
            morale_reaction=morale_reaction,
            locker_room_impact=locker_room_impact,
        )
