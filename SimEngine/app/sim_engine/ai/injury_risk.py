from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from app.sim_engine.ai.personality import PersonalityProfile, clamp01
from app.sim_engine.ai.career_arc import CareerArcState, CareerPhase


# --------------------------------------------------
# Injury Risk State (latent, slow-moving)
# --------------------------------------------------

@dataclass
class InjuryRiskState:
    year: int = 0

    # Core latent risk dimensions (0..1)
    physical_risk: float = 0.0        # playstyle, usage, contact
    fatigue_risk: float = 0.0         # mental + physical exhaustion
    behavioral_risk: float = 0.0      # choices, recklessness
    recovery_risk: float = 0.0        # healing speed & discipline
    life_stress_risk: float = 0.0     # off-ice load

    # Derived composite (cached per tick)
    total_risk: float = 0.0


# --------------------------------------------------
# Injury Risk Engine
# --------------------------------------------------

class InjuryRiskEngine:
    """
    Latent injury-proneness engine.

    DESIGN PRINCIPLES:
    - No injury events
    - No randomness
    - Slow accumulation + slow recovery
    - Personality-filtered
    - Career-arc aware
    - Ambient wear ensures no immortal bodies
    """

    # -------------------------------
    # Creation
    # -------------------------------

    def create_state(self) -> InjuryRiskState:
        return InjuryRiskState()

    # -------------------------------
    # Update
    # -------------------------------

    def update(
        self,
        state: InjuryRiskState,
        *,
        personality: PersonalityProfile,
        morale_axes: Dict[str, float],
        career: CareerArcState,
    ) -> None:
        """
        Call once per season.
        """

        state.year += 1

        confidence = morale_axes.get("confidence", 0.5)
        motivation = morale_axes.get("motivation", 0.5)
        stability = morale_axes.get("stability", 0.5)
        trust = morale_axes.get("trust", 0.5)

        # --------------------------------------------------
        # 0. Ambient Wear & Tear (baseline hockey tax)
        # --------------------------------------------------
        if career.phase == CareerPhase.ENTRY:
            wear = 0.002
        elif career.phase == CareerPhase.PRIME:
            wear = 0.004
        elif career.phase == CareerPhase.MIDCAREER:
            wear = 0.006
        elif career.phase == CareerPhase.LATE:
            wear = 0.010
        else:  # TWILIGHT
            wear = 0.015

        state.physical_risk = clamp01(state.physical_risk + wear)
        state.recovery_risk = clamp01(state.recovery_risk + wear * 1.2)

        # --------------------------------------------------
        # 1. Physical Risk (on-ice style & usage)
        # --------------------------------------------------
        physical_input = (
            personality.competitiveness * 0.30 +
            personality.risk_tolerance * 0.25 +
            personality.ego * 0.15 +
            (1.0 - personality.patience) * 0.15 +
            (1.0 - confidence) * 0.15
        )

        physical_rate = 0.03 + 0.04 * personality.competitiveness
        state.physical_risk = clamp01(
            state.physical_risk + physical_rate * (physical_input - 0.45)
        )

        # --------------------------------------------------
        # 2. Fatigue Risk (mental + physical exhaustion)
        # --------------------------------------------------
        fatigue_input = (
            (1.0 - motivation) * 0.35 +
            career.emotional_fatigue * 0.35 +
            career.expectation_gap * 0.20 +
            personality.volatility * 0.10
        )

        fatigue_rate = 0.03 + 0.05 * personality.volatility
        state.fatigue_risk = clamp01(
            state.fatigue_risk + fatigue_rate * (fatigue_input - 0.40)
        )

        # --------------------------------------------------
        # 3. Behavioral Risk (reckless decisions)
        # --------------------------------------------------
        behavioral_input = (
            personality.ego * 0.30 +
            personality.competitiveness * 0.25 +
            personality.volatility * 0.20 +
            (1.0 - personality.coachability) * 0.15 +
            (1.0 - trust) * 0.10
        )

        behavioral_rate = 0.03 + 0.04 * personality.ego
        state.behavioral_risk = clamp01(
            state.behavioral_risk + behavioral_rate * (behavioral_input - 0.45)
        )

        # --------------------------------------------------
        # 4. Recovery Risk (healing discipline)
        # --------------------------------------------------
        recovery_input = (
            (1.0 - personality.patience) * 0.30 +
            (1.0 - personality.coachability) * 0.25 +
            state.fatigue_risk * 0.25 +
            (1.0 - stability) * 0.20
        )

        recovery_rate = 0.03 + 0.04 * (1.0 - personality.patience)
        state.recovery_risk = clamp01(
            state.recovery_risk + recovery_rate * (recovery_input - 0.40)
        )

        # --------------------------------------------------
        # 5. Life Stress Risk (off-ice load)
        # --------------------------------------------------
        life_input = (
            personality.family_priority * 0.30 +
            personality.stability_need * 0.30 +
            career.security_anxiety * 0.25 +
            (1.0 - stability) * 0.15
        )

        life_rate = 0.02 + 0.04 * personality.stability_need
        state.life_stress_risk = clamp01(
            state.life_stress_risk + life_rate * (life_input - 0.40)
        )

        # --------------------------------------------------
        # Composite Total Risk
        # --------------------------------------------------
        state.total_risk = clamp01(
            0.25 * state.physical_risk +
            0.20 * state.fatigue_risk +
            0.20 * state.behavioral_risk +
            0.20 * state.recovery_risk +
            0.15 * state.life_stress_risk
        )

    # -------------------------------
    # Debug / Introspection
    # -------------------------------

    def summary(self, state: InjuryRiskState) -> Dict[str, float]:
        return {
            "year": state.year,
            "physical_risk": round(state.physical_risk, 2),
            "fatigue_risk": round(state.fatigue_risk, 2),
            "behavioral_risk": round(state.behavioral_risk, 2),
            "recovery_risk": round(state.recovery_risk, 2),
            "life_stress_risk": round(state.life_stress_risk, 2),
            "total_risk": round(state.total_risk, 2),
        }
