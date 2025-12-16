from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from app.sim_engine.ai.personality import PersonalityProfile, clamp01


# --------------------------------------------------
# Career Phase (time-based, not performance-based)
# --------------------------------------------------

class CareerPhase(str, Enum):
    ENTRY = "entry"
    PRIME = "prime"
    MIDCAREER = "midcareer"
    LATE = "late"
    TWILIGHT = "twilight"


# --------------------------------------------------
# Career Arc State (latent, expressive)
# --------------------------------------------------

@dataclass
class CareerArcState:
    year: int = 0
    phase: CareerPhase = CareerPhase.ENTRY

    # Core pressure dimensions (0..1)
    expectation_gap: float = 0.0
    identity_instability: float = 0.0
    emotional_fatigue: float = 0.0
    legacy_pressure: float = 0.0
    security_anxiety: float = 0.0

    # Soft flags (read-only, no decisions)
    questioning_identity: bool = False
    questioning_future: bool = False


# --------------------------------------------------
# Career Arc Engine
# --------------------------------------------------

class CareerArcEngine:
    """
    Long-horizon career psychology engine.

    DESIGN PHILOSOPHY:
    - Accumulates pressure slowly
    - Expresses meaning via interpretation, not outcomes
    - Personality-filtered
    - Morale-informed
    - Narrative-agnostic
    """

    # -------------------------------
    # Creation
    # -------------------------------

    def create_state(self) -> CareerArcState:
        return CareerArcState()

    # -------------------------------
    # Phase Progression (time only)
    # -------------------------------

    def _advance_phase(self, state: CareerArcState) -> None:
        if state.year < 3:
            state.phase = CareerPhase.ENTRY
        elif state.year < 7:
            state.phase = CareerPhase.PRIME
        elif state.year < 12:
            state.phase = CareerPhase.MIDCAREER
        elif state.year < 16:
            state.phase = CareerPhase.LATE
        else:
            state.phase = CareerPhase.TWILIGHT

    # -------------------------------
    # Update (ACCUMULATION ONLY)
    # -------------------------------

    def update(
        self,
        state: CareerArcState,
        *,
        personality: PersonalityProfile,
        morale_axes: Dict[str, float],
    ) -> None:
        """
        Call once per season.

        This function:
        - Mutates state
        - Accumulates pressure
        - NEVER interprets
        """

        state.year += 1
        self._advance_phase(state)

        confidence = morale_axes.get("confidence", 0.5)
        trust = morale_axes.get("trust", 0.5)
        motivation = morale_axes.get("motivation", 0.5)
        stability = morale_axes.get("stability", 0.5)

        # --------------------------------------------------
        # 1. Expectation Gap (self-image vs reality)
        # --------------------------------------------------

        expected_self = (
            0.4 * personality.ambition +
            0.3 * personality.ego +
            0.3 * personality.legacy_drive
        )

        reality = confidence * 0.6 + motivation * 0.4
        gap = expected_self - reality

        gap_rate = 0.04 + 0.05 * (1.0 - personality.patience)

        state.expectation_gap = clamp01(
            state.expectation_gap + gap_rate * gap
        )

        # --------------------------------------------------
        # 2. Identity Instability
        # --------------------------------------------------

        identity_input = (
            (1.0 - trust) * 0.4 +
            (1.0 - confidence) * 0.4 +
            personality.volatility * 0.2
        )

        id_rate = 0.03 + 0.05 * (1.0 - personality.adaptability)

        state.identity_instability = clamp01(
            state.identity_instability + id_rate * (identity_input - 0.35)
        )

        # --------------------------------------------------
        # 3. Emotional Fatigue
        # --------------------------------------------------

        fatigue_input = (
            (1.0 - motivation) * 0.4 +
            state.expectation_gap * 0.3 +
            state.identity_instability * 0.3
        )

        fatigue_rate = 0.03 + 0.05 * personality.volatility

        state.emotional_fatigue = clamp01(
            state.emotional_fatigue + fatigue_rate * (fatigue_input - 0.30)
        )

        # --------------------------------------------------
        # 4. Legacy Pressure (phase-gated)
        # --------------------------------------------------

        if state.phase in (
            CareerPhase.MIDCAREER,
            CareerPhase.LATE,
            CareerPhase.TWILIGHT,
        ):
            legacy_input = (
                (1.0 - confidence) * 0.5 +
                personality.legacy_drive * 0.5
            )

            legacy_rate = 0.02 + 0.05 * personality.legacy_drive

            state.legacy_pressure = clamp01(
                state.legacy_pressure + legacy_rate * (legacy_input - 0.35)
            )

        # --------------------------------------------------
        # 5. Security Anxiety
        # --------------------------------------------------

        security_input = (
            (1.0 - stability) * 0.6 +
            personality.stability_need * 0.4
        )

        sec_rate = 0.03 + 0.04 * personality.stability_need

        state.security_anxiety = clamp01(
            state.security_anxiety + sec_rate * (security_input - 0.40)
        )

        # --------------------------------------------------
        # Soft Flags (NO decisions)
        # --------------------------------------------------

        state.questioning_identity = (
            state.identity_instability > 0.55 or
            state.expectation_gap > 0.60
        )

        state.questioning_future = (
            state.emotional_fatigue > 0.60 or
            state.legacy_pressure > 0.65 or
            state.security_anxiety > 0.65
        )

    # -------------------------------
    # Evaluate (INTERPRETATION ONLY)
    # -------------------------------

    def evaluate(
        self,
        state: CareerArcState,
        *,
        personality: PersonalityProfile,
        morale_axes: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Interprets latent pressure into expressive tensions.

        - Read-only
        - No state mutation
        - No decisions
        - No narratives
        """

        confidence = morale_axes.get("confidence", 0.5)

        return {
            # Who am I vs what I am doing
            "identity_strain": clamp01(
                state.expectation_gap * 0.4 +
                state.identity_instability * 0.4 +
                (1.0 - confidence) * 0.2
            ),

            # Silent emotional shutdown
            "emotional_suppression": clamp01(
                (1.0 - personality.volatility) * state.emotional_fatigue
            ),

            # Loyalty vs personal fulfillment
            "loyalty_conflict": clamp01(
                personality.loyalty * state.expectation_gap
            ),

            # Time running out without confidence
            "legacy_dissonance": clamp01(
                state.legacy_pressure * (1.0 - confidence)
            ),

            # Fear-driven security behavior
            "security_spiral": clamp01(
                state.security_anxiety * (1.0 - personality.risk_tolerance)
            ),
        }

    # -------------------------------
    # Debug / Introspection
    # -------------------------------

    def summary(self, state: CareerArcState) -> Dict[str, float | str | bool]:
        return {
            "year": state.year,
            "phase": state.phase.value,
            "expectation_gap": round(state.expectation_gap, 2),
            "identity_instability": round(state.identity_instability, 2),
            "emotional_fatigue": round(state.emotional_fatigue, 2),
            "legacy_pressure": round(state.legacy_pressure, 2),
            "security_anxiety": round(state.security_anxiety, 2),
            "questioning_identity": state.questioning_identity,
            "questioning_future": state.questioning_future,
        }
