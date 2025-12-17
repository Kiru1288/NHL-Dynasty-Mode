from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import random


# =============================================================================
# Data models
# =============================================================================

@dataclass
class RetirementFactors:
    age_pressure: float = 0.0
    injury_pressure: float = 0.0
    morale_pressure: float = 0.0
    contract_pressure: float = 0.0
    career_context_pressure: float = 0.0
    fatigue_pressure: float = 0.0
    voluntary_exit_pressure: float = 0.0

    legacy_resistance: float = 0.0
    money_resistance: float = 0.0
    competitiveness_resistance: float = 0.0
    morale_resistance: float = 0.0
    health_resistance: float = 0.0

    early_outlier_boost: float = 0.0
    late_outlier_boost: float = 0.0

    def total_pressure(self) -> float:
        return (
            self.age_pressure
            + self.injury_pressure
            + self.morale_pressure
            + self.contract_pressure
            + self.career_context_pressure
            + self.fatigue_pressure
            + self.voluntary_exit_pressure
            + self.early_outlier_boost
        )

    def total_resistance(self) -> float:
        return (
            self.legacy_resistance
            + self.money_resistance
            + self.competitiveness_resistance
            + self.morale_resistance
            + self.health_resistance
            + self.late_outlier_boost
        )


@dataclass
class RetirementDecision:
    retired: bool
    retire_chance: float
    net_score: float
    threshold: float
    primary_reason: str
    secondary_reasons: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    factors: RetirementFactors = field(default_factory=RetirementFactors)
    considering: bool = False


# =============================================================================
# Helpers
# =============================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float, k: float = 6.0) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-k * x))


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default)


def is_truthy(x: Any) -> bool:
    return bool(x) and x != 0


# =============================================================================
# Retirement Engine
# =============================================================================

class RetirementEngine:
    """
    Conservative retirement model with a realistic early-late voluntary exit window.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

        self.base_threshold = 0.45
        self.considering_threshold = 0.25
        self.noise_std = 0.03

        self.early_outlier_base = 0.00005
        self.late_outlier_base = 0.0020

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_player(self, player: Any, context: Optional[Dict[str, Any]] = None) -> RetirementDecision:
        context = context or {}
        factors = RetirementFactors()

        age = float(safe_get(player, "age", 27))
        morale = float(safe_get(player, "morale", 0.6))
        personality = safe_get(player, "personality", None)

        # --------------------------------------------------
        # HARD BLOCK (<25)
        # --------------------------------------------------
        if age < 25 and not is_truthy(context.get("career_ending_injury")):
            return RetirementDecision(
                retired=False,
                retire_chance=0.000001,
                net_score=-1.0,
                threshold=1.0,
                primary_reason="too young",
                tags=["hard_age_block"],
            )

        # --------------------------------------------------
        # Core pressures
        # --------------------------------------------------
        factors.age_pressure = self._age_pressure(age)
        factors.injury_pressure = self._injury_pressure(player, context)
        factors.morale_pressure, factors.morale_resistance = self._morale_influence(morale)
        factors.contract_pressure = self._contract_pressure(age, context)
        factors.career_context_pressure = self._career_context_pressure(context)
        factors.fatigue_pressure = self._fatigue_pressure(context)
        factors.voluntary_exit_pressure = self._voluntary_exit_pressure(player, context)

        (
            factors.legacy_resistance,
            factors.money_resistance,
            factors.competitiveness_resistance,
        ) = self._personality_resistance(personality)

        factors.health_resistance = self._health_resistance(player)

        # --------------------------------------------------
        # Net score
        # --------------------------------------------------
        net = factors.total_pressure() - factors.total_resistance()
        net += self.rng.gauss(0.0, self.noise_std)

        threshold = self._threshold_adjustment(age)
        raw_chance = sigmoid(net - threshold)

        # --------------------------------------------------
        # AGE CAPS (35–36 INTENTIONALLY HIGHER)
        # --------------------------------------------------
        age_caps = [
            (25, 0.00001),
            (28, 0.0001),
            (31, 0.001),
            (34, 0.004),
            (36, 0.10),   # ⭐ voluntary exit window
            (38, 0.18),
            (40, 0.35),
        ]

        cap = 1.0
        for max_age, max_prob in age_caps:
            if age <= max_age:
                cap = max_prob
                break

        retire_chance = min(raw_chance, cap)
        retired = self.rng.random() < retire_chance

        primary, secondary, tags = self._classify_reasons(age, factors)
        considering = (not retired) and ((net - threshold) > self.considering_threshold)

        return RetirementDecision(
            retired=retired,
            retire_chance=retire_chance,
            net_score=net,
            threshold=threshold,
            primary_reason=primary,
            secondary_reasons=secondary,
            tags=tags,
            factors=factors,
            considering=considering,
        )

    # ------------------------------------------------------------------
    # Pressure components
    # ------------------------------------------------------------------

    def _age_pressure(self, age: float) -> float:
        if age < 30: return 0.0
        if age < 34: return 0.05
        if age < 37: return 0.12
        if age < 40: return 0.30
        if age < 43: return 0.55
        return 0.85

    def _injury_pressure(self, player, context) -> float:
        wear = float(safe_get(player, "injury_wear", 0.0))
        return clamp(wear * 0.7, 0.0, 1.5)

    def _morale_influence(self, morale: float) -> Tuple[float, float]:
        pressure = max(0.0, 0.45 - morale) * 0.6
        resistance = max(0.0, morale - 0.7) * 0.5
        return pressure, resistance

    def _contract_pressure(self, age: float, context) -> float:
        if age < 32:
            return 0.0
        return 0.15 if is_truthy(context.get("no_offers")) else 0.0

    def _career_context_pressure(self, context) -> float:
        return 0.10 if is_truthy(context.get("bought_out")) else 0.0

    def _fatigue_pressure(self, context) -> float:
        return float(context.get("usage_heavy", 0.0)) * 0.06

    def _voluntary_exit_pressure(self, player, context) -> float:
        age = float(safe_get(player, "age", 27))
        if age < 35 or age > 36:
            return 0.0

        morale = float(safe_get(player, "morale", 0.6))
        family = clamp(float(safe_get(player.personality, "family_priority", 0.0)))
        competitiveness = clamp(float(safe_get(player.personality, "competitiveness", 0.5)))
        identity = float(context.get("identity_instability", 0.0))
        fatigue = float(context.get("emotional_fatigue", 0.0))

        base = (
            (0.55 - morale) * 0.4
            + family * 0.35
            + identity * 0.45
            + fatigue * 0.45
            - competitiveness * 0.6
        )

        return clamp(base, 0.0, 0.6)

    def _personality_resistance(self, p) -> Tuple[float, float, float]:
        return (
            clamp(float(safe_get(p, "legacy_drive", 0.5))) * 0.5,
            clamp(float(safe_get(p, "money_focus", 0.4))) * 0.2,
            clamp(float(safe_get(p, "competitiveness", 0.5))) * 0.45,
        )

    def _health_resistance(self, player) -> float:
        return clamp(float(safe_get(player, "durability", 0.6)) * 0.35, 0.0, 0.4)

    # ------------------------------------------------------------------
    # Threshold + classification
    # ------------------------------------------------------------------

    def _threshold_adjustment(self, age: float) -> float:
        if age < 30: return 1.0
        if age < 35: return 0.6
        if age < 40: return 0.42
        return 0.35

    def _classify_reasons(self, age, factors):
        if age < 30:
            return "extreme outlier", [], ["extreme_early"]
        if factors.voluntary_exit_pressure > 0.25:
            return "voluntary_exit", [], ["early_late"]
        if factors.injury_pressure > 0.4:
            return "injuries", [], []
        if factors.morale_pressure > 0.25:
            return "burnout", [], []
        return "age", [], []
