from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import random

from app.sim_engine.ai.decision_weights import get_decision_weights


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

    # NEW: life/off-ice pressure (from decision_weights)
    life_pressure: float = 0.0

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
            + self.life_pressure
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

    UPDATED:
    - Pulls player.life_pressure and context signals
    - Uses decision_weights retirement mapping
    - Adds life pressure into net score so it can beat "age"
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

        # Existing voluntary_exit window (35–36)
        factors.voluntary_exit_pressure = self._voluntary_exit_pressure(player, context)

        # NEW: life/off-ice pressure weighted by decision_weights
        factors.life_pressure = self._life_pressure_weighted(player, context)

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
        # AGE CAPS (still conservative, but life pressure can win within caps)
        # --------------------------------------------------
        age_caps = [
            (25, 0.00001),
            (28, 0.0001),
            (31, 0.001),
            (34, 0.004),
            (36, 0.10),   # voluntary exit window
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

        primary, secondary, tags = self._classify_reasons(age, factors, player, context)
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
        base = clamp(wear * 0.7, 0.0, 1.5)
        if is_truthy(context.get("career_ending_injury")):
            base += 0.55
        if is_truthy(context.get("recent_major_injury")):
            base += 0.20
        return clamp(base, 0.0, 1.5)

    def _morale_influence(self, morale: float) -> Tuple[float, float]:
        pressure = max(0.0, 0.45 - morale) * 0.6
        resistance = max(0.0, morale - 0.7) * 0.5
        return pressure, resistance

    def _contract_pressure(self, age: float, context) -> float:
        if age < 32:
            return 0.0
        return 0.15 if is_truthy(context.get("no_offers")) else 0.0

    def _career_context_pressure(self, context) -> float:
        pressure = 0.0
        pressure += 0.10 if is_truthy(context.get("bought_out")) else 0.0
        pressure += 0.05 if is_truthy(context.get("healthy_scratches", 0) >= 25) else 0.0
        return pressure

    def _fatigue_pressure(self, context) -> float:
        # keep light; main fatigue is in life pressure weighting
        return float(context.get("usage_heavy", 0.0)) * 0.06

    def _voluntary_exit_pressure(self, player, context) -> float:
        age = float(safe_get(player, "age", 27))
        if age < 35 or age > 36:
            return 0.0

        morale = float(safe_get(player, "morale", 0.6))
        family = clamp(float(safe_get(player.personality, "family_priority", 0.0)))
        competitiveness = clamp(float(safe_get(player.personality, "competitiveness", 0.5)))

        # Use the signals you already pass from engine.py
        identity = 1.0 if is_truthy(context.get("questioning_identity")) else float(context.get("identity_instability", 0.0))
        fatigue = float(context.get("mental_fatigue", 0.0))  # engine provides this

        base = (
            (0.55 - morale) * 0.4
            + family * 0.35
            + identity * 0.45
            + fatigue * 0.55
            - competitiveness * 0.6
        )

        return clamp(base, 0.0, 0.6)

    # ------------------------------------------------------------------
    # NEW: decision_weights retirement model integration
    # ------------------------------------------------------------------

    def _life_pressure_weighted(self, player, context) -> float:
        """
        Convert player.life_pressure + context signals into a weighted retirement pressure
        using decision_weights.py ("retirement" -> domains -> traits).
        """

        # Domain pressures from engine (0..1). This is your "human life state"
        domain_pressure = safe_get(player, "life_pressure", {}) or {}

        # Pull retirement weights from decision_weights
        weights = get_decision_weights("retirement")  # domain -> trait weights

        # Map engine state/context into trait values (0..1)
        # We keep this simple and grounded in what you already have.
        morale = float(safe_get(player, "morale", 0.6))
        wear = float(safe_get(player, "injury_wear", 0.0))
        p = safe_get(player, "personality", None)

        trait_values: Dict[str, float] = {
            # health domain traits
            "injury_risk": clamp(wear),
            "chronic_pain": clamp(wear * 0.85),
            "mental_fatigue": clamp(float(context.get("mental_fatigue", domain_pressure.get("psychological", 0.0)))),

            # family domain traits
            "family_priority": clamp(float(safe_get(p, "family_priority", 0.3)) if p else 0.3),
            "relationship_strain": clamp(domain_pressure.get("family", 0.0) * 0.7),
            "desire_for_normal_life": clamp(domain_pressure.get("family", 0.0) * 0.6),

            # psychological domain traits
            "burnout": clamp(max(domain_pressure.get("psychological", 0.0), float(context.get("mental_fatigue", 0.0)))),
            "anxiety": clamp(domain_pressure.get("psychological", 0.0) * 0.6),
            "loss_of_joy": clamp(max(0.0, 0.55 - morale) + domain_pressure.get("psychological", 0.0) * 0.4),

            # career_identity domain traits (note: these are resistances via negative weights)
            "legacy_drive": clamp(float(safe_get(p, "legacy_drive", 0.5)) if p else 0.5),
            "ambition": clamp(float(safe_get(p, "ambition", 0.5)) if p else 0.5),
            "confidence": clamp(float(safe_get(p, "confidence", 0.5)) if p else 0.5),

            # security domain traits
            "financial_security": clamp(float(context.get("career_earnings", 0.0)) / 200_000_000.0),
            "money_focus": clamp(float(safe_get(p, "money_focus", 0.4)) if p else 0.4),
        }

        # Weighted sum:
        # - We compute traits according to decision_weights
        # - We also lightly multiply by domain pressure to reflect "current life strain"
        score = 0.0
        for domain, trait_map in weights.items():
            dom_mult = 0.65 + 0.55 * clamp(domain_pressure.get(domain, 0.0))  # 0.65..1.20
            for trait, w in trait_map.items():
                v = clamp(float(trait_values.get(trait, 0.0)))
                score += (v * w) * dom_mult

        # Clamp and scale into the same "pressure units" as other factors
        # Typical useful range: 0.0..~0.6 (so it competes with age_pressure)
        return clamp(score, 0.0, 0.75)

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

    def _classify_reasons(self, age: float, factors: RetirementFactors, player: Any, context: Dict[str, Any]):
        tags: List[str] = []
        secondary: List[str] = []

        # Keep the "extreme early" label
        if age < 30:
            return "extreme outlier", [], ["extreme_early"]

        # Strong non-age reasons should beat "age"
        if factors.injury_pressure > 0.70:
            return "injuries", [], []

        # Burnout / mental
        if factors.life_pressure > 0.35 and float(context.get("mental_fatigue", 0.0)) > 0.45:
            return "burnout", [], ["life_pressure"]

        # Family pull
        if is_truthy(context.get("family_event")) or (safe_get(player, "life_pressure", {}).get("family", 0.0) > 0.55):
            if factors.life_pressure > 0.30:
                return "family", [], ["life_pressure"]

        # Identity collapse
        if is_truthy(context.get("questioning_identity")) and factors.life_pressure > 0.30:
            return "identity", [], ["life_pressure"]

        # Your 35–36 voluntary exit window stays
        if factors.voluntary_exit_pressure > 0.25:
            return "voluntary_exit", [], ["early_late"]

        # Traditional morale burnout
        if factors.morale_pressure > 0.25:
            return "burnout", [], []

        return "age", secondary, tags
