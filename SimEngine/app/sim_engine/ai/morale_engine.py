from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import time

from app.sim_engine.ai.personality import (
    PersonalityProfile,
    PersonalityBehavior,
    BehaviorContext,
    clamp01,
    sigmoid,
)

# --------------------------------------------------
# Morale Axes
# --------------------------------------------------

AXES = (
    "confidence",
    "belonging",
    "trust",
    "motivation",
    "stability",
)

# Floors (hard safety only)
MOTIVATION_FLOOR = 0.08
TRUST_FLOOR = 0.05
STABILITY_FLOOR = 0.06

MAX_STABILITY_DROP_PER_TICK = 0.20
MAX_TRUST_GAIN_PER_TICK = 0.08

BELONGING_SOFTCAP_START = 0.85


# --------------------------------------------------
# Morale Signal
# --------------------------------------------------

@dataclass
class MoraleSignal:
    key: str
    intensity: float
    axis_weights: Dict[str, float]
    half_life_days: float
    created_ts: float = field(default_factory=lambda: time.time())

    def value(self, now_ts: float) -> float:
        if self.half_life_days <= 0:
            return 0.0
        age_days = max(0.0, (now_ts - self.created_ts) / 86400.0)
        decay = 2 ** (-age_days / self.half_life_days)
        return self.intensity * decay


# --------------------------------------------------
# Morale State
# --------------------------------------------------

@dataclass
class MoraleState:
    axes: Dict[str, float]
    signals: List[MoraleSignal]

    def overall(self) -> float:
        avg = sum(self.axes.values()) / len(self.axes)
        return clamp01(sigmoid((avg - 0.5) * 3.0))


# --------------------------------------------------
# Morale Engine
# --------------------------------------------------

class MoraleEngine:
    """
    Personality-anchored morale system.

    Key properties:
    - Each player has their own emotional center per axis
    - Most players live near the middle
    - Extremes decay unless reinforced
    """

    # -------------------------------
    # Creation
    # -------------------------------

    def create_state(self) -> MoraleState:
        return MoraleState(
            axes={axis: 0.5 for axis in AXES},
            signals=[]
        )

    # -------------------------------
    # Personality Anchors
    # -------------------------------

    def _axis_center(self, axis: str, p: PersonalityProfile) -> float:
        if axis == "confidence":
            return clamp01(0.40 + 0.40 * p.confidence)
        if axis == "belonging":
            return clamp01(0.35 + 0.45 * p.loyalty)
        if axis == "trust":
            return clamp01(0.30 + 0.45 * p.coachability)
        if axis == "motivation":
            return clamp01(0.35 + 0.45 * p.competitiveness)
        if axis == "stability":
            return clamp01(0.40 + 0.50 * p.stability_need)
        return 0.5

    def _swing_damping(self, p: PersonalityProfile) -> float:
        """
        Lower = more stable emotional range.
        """
        return clamp01(
            0.65
            + 0.60 * p.volatility
            - 0.40 * p.patience
            - 0.25 * p.confidence
        )

    # -------------------------------
    # Add Circumstances
    # -------------------------------

    def add_circumstances(
        self,
        state: MoraleState,
        circumstances: Dict[str, Dict],
    ) -> None:
        for key, c in circumstances.items():
            state.signals.append(
                MoraleSignal(
                    key=key,
                    intensity=clamp01(abs(c["intensity"])) * (1 if c["intensity"] >= 0 else -1),
                    axis_weights=c["axes"],
                    half_life_days=float(c["half_life"]),
                )
            )

    # -------------------------------
    # Update Step
    # -------------------------------

    def update(
        self,
        state: MoraleState,
        *,
        personality: PersonalityProfile,
        behavior: PersonalityBehavior,
        ctx: BehaviorContext,
        now_ts: float | None = None,
    ) -> None:
        now_ts = now_ts or time.time()
        prev_axes = dict(state.axes)

        damping = self._swing_damping(personality)

        # --------------------------------------------------
        # Reversion toward PERSONALITY CENTER
        # --------------------------------------------------
        for axis in state.axes:
            center = self._axis_center(axis, personality)

            reversion_rate = 0.015 + 0.05 * personality.patience
            if axis == "trust":
                reversion_rate *= 0.6 + 0.4 * personality.coachability
            if axis == "stability":
                reversion_rate *= 0.6 + 0.6 * (1.0 - ctx.injury_burden)

            state.axes[axis] = clamp01(
                state.axes[axis] + reversion_rate * (center - state.axes[axis])
            )

        # --------------------------------------------------
        # Apply Signals
        # --------------------------------------------------
        for signal in list(state.signals):
            v = signal.value(now_ts)
            if abs(v) < 0.01:
                state.signals.remove(signal)
                continue

            reaction = behavior.sample_morale_reaction(ctx)

            for axis, weight in signal.axis_weights.items():
                if axis not in state.axes:
                    continue

                delta = v * weight * reaction * damping

                # Extremes resist further movement
                dist = abs(state.axes[axis] - self._axis_center(axis, personality))
                resistance = clamp01(1.0 - dist * 1.25)
                delta *= resistance

                if axis == "belonging" and state.axes["trust"] < 0.15 and delta > 0:
                    delta *= 0.5

                state.axes[axis] = clamp01(state.axes[axis] + delta)

        # --------------------------------------------------
        # Interactions
        # --------------------------------------------------
        self._apply_interactions(state, personality)

        # --------------------------------------------------
        # Stability / Trust Limits
        # --------------------------------------------------
        if state.axes["stability"] < prev_axes["stability"]:
            drop = prev_axes["stability"] - state.axes["stability"]
            if drop > MAX_STABILITY_DROP_PER_TICK:
                state.axes["stability"] = prev_axes["stability"] - MAX_STABILITY_DROP_PER_TICK

        state.axes["trust"] = min(
            state.axes["trust"],
            prev_axes["trust"] + MAX_TRUST_GAIN_PER_TICK
        )

        # --------------------------------------------------
        # Floors
        # --------------------------------------------------
        state.axes["motivation"] = max(state.axes["motivation"], MOTIVATION_FLOOR)
        state.axes["trust"] = max(state.axes["trust"], TRUST_FLOOR)
        state.axes["stability"] = max(state.axes["stability"], STABILITY_FLOOR)

    # -------------------------------
    # Interactions
    # -------------------------------

    def _apply_interactions(self, state: MoraleState, p: PersonalityProfile) -> None:
        if state.axes["confidence"] < 0.35 and state.axes["trust"] < 0.35:
            state.axes["motivation"] = clamp01(
                state.axes["motivation"] - (0.02 + 0.04 * p.volatility)
            )

        if (
            state.axes["belonging"] > 0.85
            and state.axes["trust"] < 0.20
            and state.axes["stability"] < 0.20
        ):
            state.axes["motivation"] = clamp01(
                state.axes["motivation"] - (0.02 + 0.04 * p.legacy_drive)
            )

    # -------------------------------
    # Narrative Flags
    # -------------------------------

    def narrative_flags(self, state: MoraleState) -> Dict[str, bool]:
        return {
            "disengaged": state.axes["motivation"] < 0.25,
            "burnout_risk": state.axes["motivation"] > 0.7 and state.axes["stability"] < 0.3,
            "loyal_anchor": state.axes["belonging"] > 0.75 and state.axes["trust"] > 0.65,
            "volatile": state.overall() < 0.4,
        }
