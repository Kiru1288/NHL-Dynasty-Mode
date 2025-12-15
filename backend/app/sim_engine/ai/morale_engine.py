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
    "confidence",   # belief in self & performance
    "belonging",    # locker room / identity / fit
    "trust",        # coach & management
    "motivation",   # will to push through adversity
    "stability",    # life / contract / health security
)

# Floors prevent cartoon outcomes
MOTIVATION_FLOOR = 0.08
TRUST_FLOOR = 0.05
STABILITY_FLOOR = 0.06

# Limits: prevent one tick from nuking a whole axis
MAX_STABILITY_DROP_PER_TICK = 0.20
MAX_TRUST_GAIN_PER_TICK = 0.08

# Belonging soft-cap range
BELONGING_SOFTCAP_START = 0.85


# --------------------------------------------------
# Morale Signal
# --------------------------------------------------

@dataclass
class MoraleSignal:
    key: str
    intensity: float                 # [-1, +1]
    axis_weights: Dict[str, float]
    half_life_days: float
    created_ts: float = field(default_factory=lambda: time.time())

    def value(self, now_ts: float) -> float:
        # Safe handling
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
    Emergent morale system driven by:
    - accumulated circumstances
    - personality filtering
    - decay and nonlinear interactions

    This version fixes:
    - trust flatlining forever
    - stability collapsing unrealistically fast
    - belonging hitting 1.00 too easily
    """

    def create_state(self) -> MoraleState:
        return MoraleState(
            axes={axis: 0.5 for axis in AXES},
            signals=[]
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

        # Snapshot for per-tick limiting
        prev_axes = dict(state.axes)

        # --------------------------------------------------
        # Natural reversion toward neutral
        # --------------------------------------------------
        for axis in state.axes:
            base_reversion = 0.02 + 0.08 * personality.patience

            # trust: slower baseline, but can recover when things stabilize
            if axis == "trust":
                base_reversion *= 0.55 + 0.45 * personality.coachability

            # stability: recovery depends heavily on life stress
            if axis == "stability":
                low_stress = 1.0 - clamp01(ctx.injury_burden + ctx.family_event)
                base_reversion *= 0.65 + 0.85 * low_stress

            state.axes[axis] = clamp01(
                state.axes[axis] + base_reversion * (0.5 - state.axes[axis])
            )

        # --------------------------------------------------
        # Apply circumstance signals
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

                sensitivity = self._axis_sensitivity(axis, personality)
                delta = v * weight * sensitivity * reaction

                # Belonging softcap: diminishing returns near the top
                if axis == "belonging":
                    delta *= self._belonging_softcap_multiplier(state.axes["belonging"])

                    # If trust is rock-bottom, belonging shouldn't inflate endlessly
                    if state.axes["trust"] < 0.15 and delta > 0:
                        delta *= 0.55

                state.axes[axis] = clamp01(state.axes[axis] + delta)

        # --------------------------------------------------
        # Interactions
        # --------------------------------------------------
        self._apply_interactions(state, personality)

        # --------------------------------------------------
        # Trust natural repair (fixes “trust flatlines forever”)
        # No explicit “event” yet — just slow repair when conditions improve
        # --------------------------------------------------
        trust_repair = self._trust_repair(ctx, personality)
        if trust_repair > 0:
            state.axes["trust"] = clamp01(state.axes["trust"] + trust_repair)

        # Cap trust gain per tick (prevents magical rebounds)
        state.axes["trust"] = min(state.axes["trust"], prev_axes["trust"] + MAX_TRUST_GAIN_PER_TICK)

        # --------------------------------------------------
        # Stability damage limiter (fixes “stability hits 0 too easily”)
        # --------------------------------------------------
        if state.axes["stability"] < prev_axes["stability"]:
            drop = prev_axes["stability"] - state.axes["stability"]
            if drop > MAX_STABILITY_DROP_PER_TICK:
                state.axes["stability"] = prev_axes["stability"] - MAX_STABILITY_DROP_PER_TICK

        # --------------------------------------------------
        # Floors
        # --------------------------------------------------
        state.axes["motivation"] = max(state.axes["motivation"], MOTIVATION_FLOOR)
        state.axes["trust"] = max(state.axes["trust"], TRUST_FLOOR)
        state.axes["stability"] = max(state.axes["stability"], STABILITY_FLOOR)

    # -------------------------------
    # Axis Sensitivity
    # -------------------------------

    def _axis_sensitivity(self, axis: str, p: PersonalityProfile) -> float:
        if axis == "confidence":
            return 0.6 + 0.8 * (1.0 - p.confidence) + 0.4 * p.volatility
        if axis == "belonging":
            return 0.6 + 0.8 * (1.0 - p.introversion)
        if axis == "trust":
            return 0.6 + 0.8 * (1.0 - p.coachability)
        if axis == "motivation":
            return 0.6 + 0.8 * p.competitiveness
        if axis == "stability":
            return 0.6 + 0.8 * p.stability_need
        return 1.0

    # -------------------------------
    # Axis Interactions
    # -------------------------------

    def _apply_interactions(self, state: MoraleState, p: PersonalityProfile) -> None:
        # Confidence + Trust collapse → disengagement
        if state.axes["confidence"] < 0.35 and state.axes["trust"] < 0.35:
            state.axes["motivation"] = clamp01(
                state.axes["motivation"] - (0.03 + 0.05 * p.volatility)
            )

        # Burnout bleed
        if state.axes["motivation"] > 0.7 and state.axes["stability"] < 0.3:
            state.axes["confidence"] = clamp01(
                state.axes["confidence"] - (0.02 + 0.04 * (1.0 - p.patience))
            )

        # Locker room buffer
        if state.axes["belonging"] > 0.7 and state.axes["confidence"] < 0.4:
            state.axes["confidence"] = clamp01(
                state.axes["confidence"] + (0.02 + 0.03 * p.leadership)
            )

        # Stability + Trust inertia
        if state.axes["stability"] > 0.7 and state.axes["trust"] > 0.7:
            for axis in state.axes:
                state.axes[axis] = clamp01(state.axes[axis] * 0.97 + 0.5 * 0.03)

    # -------------------------------
    # Helpers (fixes)
    # -------------------------------

    def _belonging_softcap_multiplier(self, belonging: float) -> float:
        """
        Diminishing returns for belonging > softcap start.
        """
        if belonging <= BELONGING_SOFTCAP_START:
            return 1.0
        # as belonging approaches 1.0, gains shrink toward ~0.25
        t = (belonging - BELONGING_SOFTCAP_START) / max(1e-6, (1.0 - BELONGING_SOFTCAP_START))
        return 1.0 - 0.75 * clamp01(t)

    def _trust_repair(self, ctx: BehaviorContext, p: PersonalityProfile) -> float:
        """
        Natural trust repair without explicit events:
        - role mismatch low
        - ice time satisfaction decent
        - not in a hardcore losing spiral
        - coachability increases the repair rate
        """
        role_ok = clamp01(1.0 - ctx.role_mismatch)
        ice_ok = clamp01((ctx.ice_time_satisfaction - 0.45) / 0.55)
        stress = clamp01(ctx.losing_streak + ctx.rebuild_mode)

        # If conditions are good, trust can slowly rebuild
        condition = role_ok * ice_ok * (1.0 - 0.7 * stress)

        if condition <= 0.15:
            return 0.0

        # Slow repair rate; coachable players repair faster
        rate = 0.01 + 0.05 * p.coachability
        return clamp01(condition) * rate

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
