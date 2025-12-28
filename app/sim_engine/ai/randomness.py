"""
randomness.py

Controlled human unpredictability layer.

This module:
- Injects SMALL, personality-weighted noise
- Triggers RARE, contextual life events
- Modifies pressure accumulation & recovery
- NEVER decides outcomes
- NEVER overrides engines

Philosophy:
Randomness alters HOW things are felt — not WHAT happens.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import random


# =============================================================================
# Helpers
# =============================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# =============================================================================
# Randomness Engine
# =============================================================================

class RandomnessEngine:
    """
    Injects controlled noise and rare human events.

    All randomness is:
    - Personality weighted
    - Context gated
    - Pressure-based
    - Deterministic with seed
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

        # Cooldowns prevent event spam
        self._cooldowns: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Seasonal emotional noise
    # ------------------------------------------------------------------

    def emotional_noise(self, personality) -> float:
        """
        Returns a small seasonal modifier (-0.15 .. +0.15)
        representing emotional volatility for this year.
        """

        volatility = clamp(float(getattr(personality, "volatility", 0.5)))
        base = self.rng.gauss(0.0, 0.06)
        return clamp(base * (0.4 + volatility), -0.15, 0.15)

    # ------------------------------------------------------------------
    # Stress sensitivity
    # ------------------------------------------------------------------

    def stress_sensitivity(self, personality) -> float:
        """
        Multiplier for how strongly pressure is felt this season.
        """

        patience = clamp(float(getattr(personality, "patience", 0.5)))
        adaptability = clamp(float(getattr(personality, "adaptability", 0.5)))

        # More patience/adaptability = less sensitivity
        sensitivity = 1.0 + (0.6 - patience) * 0.25
        sensitivity -= adaptability * 0.15

        return clamp(sensitivity, 0.75, 1.35)

    # ------------------------------------------------------------------
    # Recovery variance
    # ------------------------------------------------------------------

    def recovery_modifier(self, personality) -> float:
        """
        Modifies how quickly pressure/morale recover.
        """

        stability = clamp(float(getattr(personality, "stability_need", 0.5)))
        confidence = clamp(float(getattr(personality, "confidence", 0.5)))

        base = 1.0
        base += confidence * 0.15
        base -= stability * 0.10

        return clamp(base, 0.85, 1.20)

    # ------------------------------------------------------------------
    # Rare human life events
    # ------------------------------------------------------------------

    def roll_life_events(
        self,
        year: int,
        age: int,
        personality,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Possibly triggers rare off-ice events.

        Returns:
            Dict[str, float] -> domain pressure deltas
        """

        events: Dict[str, float] = {}

        # Respect cooldowns
        for k in list(self._cooldowns):
            self._cooldowns[k] -= 1
            if self._cooldowns[k] <= 0:
                del self._cooldowns[k]

        # --------------------------------------------------
        # Mental health crisis (EXTREMELY rare)
        # --------------------------------------------------
        if "mental_health_crisis" not in self._cooldowns:
            volatility = clamp(float(getattr(personality, "volatility", 0.5)))
            morale = float(context.get("morale", 0.6))

            chance = 0.002 + volatility * 0.004
            if morale < 0.45:
                chance += 0.004

            if self.rng.random() < chance:
                events["psychological"] = 0.7
                events["health"] = 0.3
                self._cooldowns["mental_health_crisis"] = 6  # years

        # --------------------------------------------------
        # Family emergency
        # --------------------------------------------------
        if "family_emergency" not in self._cooldowns:
            family_priority = clamp(float(getattr(personality, "family_priority", 0.3)))
            chance = 0.003 + family_priority * 0.006

            if self.rng.random() < chance:
                events["family"] = 0.8
                events["psychological"] = 0.4
                self._cooldowns["family_emergency"] = 5

        # --------------------------------------------------
        # Legal / media trouble (VERY rare, personality gated)
        # --------------------------------------------------
        if "legal_trouble" not in self._cooldowns:
            risk = clamp(float(getattr(personality, "risk_tolerance", 0.3)))
            media = clamp(float(getattr(personality, "media_comfort", 0.4)))

            chance = 0.001 + risk * 0.003
            if media < 0.3:
                chance *= 0.7

            if self.rng.random() < chance:
                events["environment"] = 0.7
                events["psychological"] = 0.5
                self._cooldowns["legal_trouble"] = 8

        # --------------------------------------------------
        # Loss of love for the game (mid/late career)
        # --------------------------------------------------
        if age >= 28 and "loss_of_love" not in self._cooldowns:
            ambition = clamp(float(getattr(personality, "ambition", 0.5)))
            competitiveness = clamp(float(getattr(personality, "competitiveness", 0.5)))

            chance = 0.002
            chance += (0.4 - ambition) * 0.004
            chance += (0.4 - competitiveness) * 0.003

            if self.rng.random() < max(0.0, chance):
                events["psychological"] = 0.9
                events["career_identity"] = 0.7
                self._cooldowns["loss_of_love"] = 10

        return events

    # ------------------------------------------------------------------
    # Context noise injection
    # ------------------------------------------------------------------

    def apply_context_noise(self, ctx: Dict[str, Any], personality) -> Dict[str, Any]:
        """
        Applies small noise to season context values.
        """

        noisy = dict(ctx)

        noise = self.emotional_noise(personality)

        if "ice_time_satisfaction" in noisy:
            noisy["ice_time_satisfaction"] = clamp(
                noisy["ice_time_satisfaction"] + noise
            )

        if "offer_respect" in noisy:
            noisy["offer_respect"] = clamp(
                noisy["offer_respect"] + noise * 0.5
            )

        if "losing_streak" in noisy:
            noisy["losing_streak"] = clamp(
                noisy["losing_streak"] + abs(noise) * 0.4
            )

        return noisy
