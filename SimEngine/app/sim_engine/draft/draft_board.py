# backend/app/sim_engine/draft/draft_board.py
"""
Draft Board System (Realistic NHL Draft Chaos)

Goals:
- Each team has its OWN board (belief system, not truth)
- Prospects slide / reach due to mentality/personality, risk, fit, and org psychology
- Board updates dynamically after each pick (iceberg effect, rumor contagion)
- Supports tiering, positional runs, panic trades, and "my guy" fixation

Hard constraints:
- Standard library only (no external deps)
- Works with "duck-typed" Prospect/Team objects (doesn't require specific classes)

How to integrate:
- Provide a list of prospects (objects or dict-like) with at least:
    id, name, position, age (optional), league (optional)
    signals/tools: e.g. "upside", "floor", "certainty", "production" (optional)
    mentality/personality: "coachability", "work_ethic", "resilience", "volatility", etc. (optional)
    risk flags: injury_risk, boom_bust (optional)
- Provide a TeamProfile for each team (style, risk tolerance, needs, coach influence, job security)

This file focuses on board creation + pick recommendation + dynamic reactions.
Trading logic is included as "signals" and "suggestions" (you can wire into your trade engine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Iterable
import math
import random


# ---------------------------
# Enums / constants
# ---------------------------

class DraftMood(str, Enum):
    CONFIDENT = "confident"
    CONSERVATIVE = "conservative"
    DESPERATE = "desperate"
    RECKLESS = "reckless"
    APATHETIC = "apathetic"


class PickIntent(str, Enum):
    TAKE_BPA = "take_bpa"
    TAKE_FIT = "take_fit"
    REACH_SAFE = "reach_safe"
    SWING_CEILING = "swing_ceiling"


DEFAULT_POSITIONS = ("C", "LW", "RW", "D", "G")


# ---------------------------
# Utility helpers (duck typing)
# ---------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Duck-typed field getter: supports dicts and objects with attributes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    # Stable-ish sigmoid for moderate ranges
    return 1.0 / (1.0 + math.exp(-x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------
# Team + board models
# ---------------------------

@dataclass
class TeamProfile:
    team_id: str
    name: str

    # Philosophy / psychology knobs
    archetype: str = "balanced"               # e.g. "draft_and_develop", "win_now", "perpetual_mediocrity"
    risk_tolerance: float = 0.5               # 0 conservative, 1 aggressive
    gm_job_security: float = 0.6              # 0 fired soon, 1 safe
    coach_patience: float = 0.5               # 0 hates projects, 1 loves development
    coach_influence: float = 0.4              # 0 minimal, 1 veto power
    scouting_quality: float = 0.5             # affects noise and misinformation resistance
    analytics_bias: float = 0.5               # 0 "old school", 1 "numbers"
    personality_strictness: float = 0.5       # 0 doesn't care, 1 strongly filters mentality/personality
    medical_strictness: float = 0.5           # 0 ignores, 1 avoids risk
    rumor_susceptibility: float = 0.5         # 0 independent, 1 herd behavior

    # Needs: weights by position and by timeline
    needs_by_position: Dict[str, float] = field(default_factory=lambda: {p: 0.5 for p in DEFAULT_POSITIONS})
    timeline_pressure: float = 0.5            # 0 rebuild, 1 contender pressure (NHL-ready bias)
    owner_pressure: float = 0.3               # 0 none, 1 heavy
    fan_pressure: float = 0.3                 # 0 none, 1 heavy

    # "My guy" fixation probability & strength
    my_guy_chance: float = 0.20
    my_guy_strength: float = 0.15             # additive bonus to board score for that player

    # Board variance controls
    board_randomness: float = 0.12            # higher = more unique board / more mistakes
    tier_sensitivity: float = 0.65            # higher = more tier-based decisions
    reach_likelihood: float = 0.20            # baseline reach probability

    # Optional: pre-set mood or let the board compute it
    mood: Optional[DraftMood] = None


@dataclass
class BoardItem:
    prospect_id: str
    name: str
    position: str

    # "Belief" fields
    base_value: float                 # shared-ish signal aggregate
    fit_value: float                  # team-specific fit (position/system/timeline)
    psyche_value: float               # mentality/personality confidence
    risk_penalty: float               # injury/boom-bust/volatility penalty (team-specific)
    rumor_penalty: float              # herd effect penalty (slide contagion)

    # Final internal score (what the team believes)
    score: float

    # Metadata for storytelling / debugging
    tier: int = 0
    flags: List[str] = field(default_factory=list)


@dataclass
class DraftEvent:
    pick_number: int
    team_id: str
    prospect_id: str
    prospect_name: str
    note: str = ""


@dataclass
class DraftContext:
    seed: int = 0
    year: int = 0
    # Positional run tracking (how many taken recently)
    recent_picks_window: int = 8
    # Whether to amplify rumor penalties as sliding happens
    iceberg_effect_strength: float = 0.6
    # How quickly runs form
    run_strength: float = 0.55


# ---------------------------
# DraftBoard
# ---------------------------

class DraftBoard:
    """
    One team's internal draft board.
    """

    def __init__(self, team: TeamProfile, ctx: DraftContext, rng: Optional[random.Random] = None):
        self.team = team
        self.ctx = ctx
        self.rng = rng or random.Random(ctx.seed ^ hash(team.team_id))

        self.items: List[BoardItem] = []
        self.by_id: Dict[str, BoardItem] = {}

        # Dynamic state during draft
        self.my_guy_id: Optional[str] = None
        self.mood: DraftMood = team.mood or self._derive_mood(team)

        # Prospect sliding “iceberg effect” memory:
        # how many times a prospect has been unexpectedly passed (league-wide)
        self.passed_counter: Dict[str, int] = {}

        # Positional run memory: list of last N picked positions
        self.recent_positions: List[str] = []

        # Draft events (optional)
        self.events: List[DraftEvent] = []

    # ---------------------------
    # Public API
    # ---------------------------

    def build(self, prospects: Iterable[Any]) -> None:
        """
        Build initial board rankings for this team.
        """
        prospects_list = list(prospects)

        # "My guy" selection BEFORE scoring (so it can bias)
        self.my_guy_id = self._pick_my_guy(prospects_list)

        items: List[BoardItem] = []
        for p in prospects_list:
            item = self._score_prospect(p)
            items.append(item)

        # Sort by internal score
        items.sort(key=lambda x: x.score, reverse=True)

        # Create tiers (soft tiering that can drive trade-up behavior)
        self._assign_tiers(items)

        self.items = items
        self.by_id = {i.prospect_id: i for i in items}

    def available(self, drafted_ids: set[str]) -> List[BoardItem]:
        """
        Return available board items in current rank order.
        """
        return [i for i in self.items if i.prospect_id not in drafted_ids]

    def top(self, drafted_ids: set[str], n: int = 10) -> List[BoardItem]:
        return self.available(drafted_ids)[:n]

    def recommend_pick(
        self,
        pick_number: int,
        drafted_ids: set[str],
        league_events: List[DraftEvent],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Recommend a prospect_id for the current pick, plus decision metadata.
        - Updates internal dynamic state based on league_events (recent picks, sliding, etc.)
        - Uses mood + team psychology to decide BPA vs Fit vs Reach vs Swing
        """
        self._ingest_league_events(league_events)

        avail = self.available(drafted_ids)
        if not avail:
            return None, {"reason": "no_prospects_left"}

        # Update rumor penalties from sliding
        self._apply_iceberg_effect(avail)

        # Adjust for positional runs
        self._apply_positional_run_pressure(avail)

        # Re-sort after dynamic adjustments
        avail.sort(key=lambda x: x.score, reverse=True)

        intent = self._choose_intent(pick_number, avail)

        # Candidate set depends on intent and tiers
        chosen = self._choose_candidate(intent, avail)

        meta = {
            "team": self.team.name,
            "pick_number": pick_number,
            "mood": self.mood.value,
            "intent": intent.value,
            "my_guy_id": self.my_guy_id,
            "top_5": [(x.prospect_id, x.name, round(x.score, 4), x.tier, x.flags) for x in avail[:5]],
            "chosen": (chosen.prospect_id, chosen.name, chosen.position, round(chosen.score, 4), chosen.tier, chosen.flags),
            "trade_signal": self._trade_signal(pick_number, avail, chosen),
        }

        return chosen.prospect_id, meta

    def on_pick_made(self, event: DraftEvent) -> None:
        """
        Feed a finalized pick result into the board, so it can update passing counts, runs, etc.
        """
        self.events.append(event)
        # Update positional run memory
        picked_item = self.by_id.get(event.prospect_id)
        pos = picked_item.position if picked_item else "?"
        self._push_recent_position(pos)

    # ---------------------------
    # Mood + intent
    # ---------------------------

    def _derive_mood(self, team: TeamProfile) -> DraftMood:
        """
        Derive a mood from job security + pressure + archetype.
        """
        pressure = clamp(0.35 * team.owner_pressure + 0.35 * team.fan_pressure + 0.30 * team.timeline_pressure)
        insecurity = 1.0 - clamp(team.gm_job_security)
        volatility = clamp(0.6 * insecurity + 0.4 * pressure)

        # Risk tolerance modulates whether that volatility becomes reckless or conservative
        if volatility > 0.75 and team.risk_tolerance > 0.55:
            return DraftMood.RECKLESS
        if volatility > 0.75 and team.risk_tolerance <= 0.55:
            return DraftMood.DESPERATE
        if team.risk_tolerance < 0.35:
            return DraftMood.CONSERVATIVE
        if team.risk_tolerance > 0.70:
            return DraftMood.CONFIDENT
        return DraftMood.CONFIDENT if pressure < 0.35 else DraftMood.CONSERVATIVE

    def _choose_intent(self, pick_number: int, avail: List[BoardItem]) -> PickIntent:
        """
        Pick decision intent: BPA vs Fit vs Reach vs Ceiling swing.
        Driven by mood, coach influence, tier cliff, and risk tolerance.
        """
        if not avail:
            return PickIntent.TAKE_BPA

        top = avail[0]
        # "Tier cliff": if top tier is about to evaporate, teams panic / trade up / take BPA
        top_tier = top.tier
        tier_left = sum(1 for x in avail[:15] if x.tier == top_tier)

        tier_cliff = clamp(1.0 - (tier_left / 6.0))  # fewer left => bigger cliff

        coach_push_safe = clamp(self.team.coach_influence * (1.0 - self.team.coach_patience))
        gm_insecure = 1.0 - self.team.gm_job_security

        base_reach = clamp(self.team.reach_likelihood + 0.20 * coach_push_safe + 0.20 * gm_insecure - 0.10 * self.team.risk_tolerance)

        # Mood overrides
        if self.mood == DraftMood.CONSERVATIVE:
            # tends toward fit / safe
            if self.rng.random() < 0.55:
                return PickIntent.TAKE_FIT
            return PickIntent.REACH_SAFE if self.rng.random() < base_reach else PickIntent.TAKE_BPA

        if self.mood == DraftMood.DESPERATE:
            # reaches more, but also panics if tier cliff
            if tier_cliff > 0.65 and self.rng.random() < 0.7:
                return PickIntent.TAKE_BPA
            return PickIntent.REACH_SAFE if self.rng.random() < clamp(base_reach + 0.15) else PickIntent.TAKE_FIT

        if self.mood == DraftMood.RECKLESS:
            # loves ceiling swings
            if self.rng.random() < clamp(0.35 + 0.40 * self.team.risk_tolerance):
                return PickIntent.SWING_CEILING
            return PickIntent.TAKE_BPA

        if self.mood == DraftMood.APATHETIC:
            # randomish
            return self.rng.choice([PickIntent.TAKE_BPA, PickIntent.TAKE_FIT])

        # CONFIDENT
        if tier_cliff > 0.70 and self.rng.random() < 0.55:
            return PickIntent.TAKE_BPA
        if self.rng.random() < 0.25:
            return PickIntent.TAKE_FIT
        return PickIntent.TAKE_BPA

    # ---------------------------
    # Candidate choice logic
    # ---------------------------

    def _choose_candidate(self, intent: PickIntent, avail: List[BoardItem]) -> BoardItem:
        """
        Choose a prospect given an intent.
        """
        if intent == PickIntent.TAKE_BPA:
            return self._softmax_pick(avail[:7], temperature=0.12)

        if intent == PickIntent.TAKE_FIT:
            # Promote high-fit within a reasonable band of BPA
            window = avail[:18]
            # Re-score window emphasizing fit + psyche over raw
            rescored = []
            for x in window:
                s = (0.45 * x.score) + (0.35 * x.fit_value) + (0.20 * x.psyche_value)
                rescored.append((s, x))
            rescored.sort(key=lambda t: t[0], reverse=True)
            candidates = [x for _, x in rescored[:6]]
            return self._softmax_pick(candidates, temperature=0.18)

        if intent == PickIntent.REACH_SAFE:
            # Reach for "low risk, high psyche" profiles even if slightly lower raw score
            window = avail[:25]
            rescored = []
            for x in window:
                safe = (0.55 * x.psyche_value) + (0.30 * (1.0 - x.risk_penalty)) + (0.15 * x.fit_value)
                # small anchor to overall score to avoid insanity
                s = 0.25 * x.score + 0.75 * safe
                rescored.append((s, x))
            rescored.sort(key=lambda t: t[0], reverse=True)
            candidates = [x for _, x in rescored[:5]]
            return self._softmax_pick(candidates, temperature=0.20)

        # SWING_CEILING
        window = avail[:30]
        rescored = []
        for x in window:
            # Encourage upside picks: reduce penalty for risk if team is aggressive
            risk_relief = self.team.risk_tolerance * 0.25
            s = (0.55 * x.base_value) + (0.20 * x.fit_value) + (0.25 * (1.0 - max(0.0, x.risk_penalty - risk_relief)))
            rescored.append((s, x))
        rescored.sort(key=lambda t: t[0], reverse=True)
        candidates = [x for _, x in rescored[:6]]
        return self._softmax_pick(candidates, temperature=0.16)

    def _softmax_pick(self, candidates: List[BoardItem], temperature: float = 0.15) -> BoardItem:
        """
        Softmax-like stochastic selection: realistic "humans aren't perfect"
        Lower temperature => more deterministic.
        """
        if not candidates:
            raise ValueError("No candidates provided")

        # Convert scores to weights (avoid overflow by shifting)
        scores = [c.score for c in candidates]
        mx = max(scores)
        weights = []
        for c in candidates:
            z = (c.score - mx) / max(1e-6, temperature)
            weights.append(math.exp(z))

        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for c, w in zip(candidates, weights):
            acc += w
            if acc >= r:
                return c
        return candidates[0]

    # ---------------------------
    # Board scoring
    # ---------------------------

    def _pick_my_guy(self, prospects: List[Any]) -> Optional[str]:
        if not prospects or self.rng.random() > self.team.my_guy_chance:
            return None

        # Prefer players who align with coach preferences + timeline + position needs
        scored: List[Tuple[float, Any]] = []
        for p in prospects:
            pos = _get(p, "position", "C")
            need = clamp(self.team.needs_by_position.get(pos, 0.5))
            psyche = self._psyche_score(p)
            readiness = clamp(_get(p, "nhl_readiness", _get(p, "readiness", 0.5)))
            upside = clamp(_get(p, "upside", _get(p, "ceiling", 0.5)))
            base = 0.35 * need + 0.25 * psyche + 0.20 * readiness + 0.20 * upside
            # Noise so it isn't always obvious
            base += self.rng.uniform(-0.05, 0.05)
            scored.append((base, p))

        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][1]
        return str(_get(best, "id", _get(best, "prospect_id", None)))

    def _score_prospect(self, p: Any) -> BoardItem:
        pid = str(_get(p, "id", _get(p, "prospect_id", "")))
        name = str(_get(p, "name", f"Prospect_{pid}"))
        position = str(_get(p, "position", "C"))

        # Shared-ish signals
        base_value = self._base_signal_score(p)

        # Team fit: position need + system + timeline/readiness
        fit_value = self._fit_score(p)

        # Personality/mentality confidence
        psyche_value = self._psyche_score(p)

        # Risk penalty (team-specific strictness)
        risk_penalty = self._risk_penalty(p)

        # Rumor penalty starts at 0; will increase dynamically if player slides
        rumor_penalty = 0.0

        # "My guy" bonus
        my_guy_bonus = self.team.my_guy_strength if (self.my_guy_id and pid == self.my_guy_id) else 0.0

        # Noise: scouting variance + uniqueness (worse scouting => more noise)
        # Better scouting reduces noise amplitude.
        noise_amp = self.team.board_randomness * lerp(1.35, 0.55, clamp(self.team.scouting_quality))
        noise = self.rng.uniform(-noise_amp, noise_amp)

        # Coach influence: penalize projects if coach impatient
        project_penalty = 0.0
        coach_hates_projects = clamp(self.team.coach_influence * (1.0 - self.team.coach_patience))
        projectiness = clamp(_get(p, "projectiness", 1.0 - _get(p, "certainty", 0.5)))
        project_penalty = coach_hates_projects * 0.18 * projectiness

        # Compose score
        # Base is king early, fit & psyche pull the board apart mid/late.
        score = (
            0.52 * base_value +
            0.26 * fit_value +
            0.22 * psyche_value
        ) + my_guy_bonus + noise

        # Apply penalties
        score -= (0.30 * risk_penalty)
        score -= project_penalty
        score -= rumor_penalty

        score = clamp(score)

        flags: List[str] = []
        if my_guy_bonus > 0:
            flags.append("MY_GUY")
        if risk_penalty > 0.55:
            flags.append("RISKY")
        if psyche_value < 0.35 and self.team.personality_strictness > 0.55:
            flags.append("BAD_INT")
        if fit_value > 0.70:
            flags.append("TEAM_FIT")
        if base_value > 0.75:
            flags.append("ELITE_SIGNAL")

        return BoardItem(
            prospect_id=pid,
            name=name,
            position=position,
            base_value=base_value,
            fit_value=fit_value,
            psyche_value=psyche_value,
            risk_penalty=risk_penalty,
            rumor_penalty=rumor_penalty,
            score=score,
            tier=0,
            flags=flags,
        )

    def _base_signal_score(self, p: Any) -> float:
        """
        Aggregate common signals. If fields aren't present, defaults keep it stable.
        """
        upside = clamp(_get(p, "upside", _get(p, "ceiling", 0.5)))
        floor = clamp(_get(p, "floor", 0.5))
        certainty = clamp(_get(p, "certainty", 0.5))
        production = clamp(_get(p, "production", _get(p, "points_signal", 0.5)))
        skating = clamp(_get(p, "skating", 0.5))
        iq = clamp(_get(p, "hockey_iq", _get(p, "iq", 0.5)))

        # Analytics bias changes weighting (numbers vs tools)
        ana = clamp(self.team.analytics_bias)
        tools = 0.35 * skating + 0.35 * iq + 0.30 * upside
        stats = 0.55 * production + 0.25 * certainty + 0.20 * floor

        base = lerp(tools, stats, ana)

        # Add slight premium to upside for aggressive teams
        base += 0.07 * (self.team.risk_tolerance - 0.5) * (upside - 0.5)

        return clamp(base)

    def _fit_score(self, p: Any) -> float:
        pos = str(_get(p, "position", "C"))
        need = clamp(self.team.needs_by_position.get(pos, 0.5))

        readiness = clamp(_get(p, "nhl_readiness", _get(p, "readiness", 0.5)))
        # contenders care more about readiness, rebuilders care less
        timeline = clamp(self.team.timeline_pressure)
        readiness_weight = lerp(0.20, 0.55, timeline)
        upside_weight = 1.0 - readiness_weight

        upside = clamp(_get(p, "upside", _get(p, "ceiling", 0.5)))

        # Coach system preference (optional hook):
        # if prospect has "system_fit" keyed by archetype or coach style
        system_fit = _get(p, "system_fit", None)
        sys = 0.5
        if isinstance(system_fit, dict):
            sys = clamp(system_fit.get(self.team.archetype, system_fit.get("default", 0.5)))
        else:
            sys = clamp(_get(p, "system_fit", 0.5))

        fit = (0.45 * need) + (0.25 * sys) + (readiness_weight * readiness) + (upside_weight * upside)
        return clamp(fit)

    def _psyche_score(self, p: Any) -> float:
        """
        Convert mentality/personality into a confidence score.
        Strict teams punish bad traits more.
        """
        # Common trait fields (optional)
        coachability = clamp(_get(p, "coachability", 0.5))
        work_ethic = clamp(_get(p, "work_ethic", _get(p, "effort", 0.5)))
        resilience = clamp(_get(p, "resilience", 0.5))
        leadership = clamp(_get(p, "leadership", 0.5))

        # Negatives
        volatility = clamp(_get(p, "volatility", _get(p, "emotional_volatility", 0.5)))
        entitlement = clamp(_get(p, "entitlement", 0.5))
        consistency = clamp(_get(p, "consistency", 0.5))

        # Base psyche profile
        positives = 0.30 * coachability + 0.25 * work_ethic + 0.25 * resilience + 0.20 * leadership
        negatives = 0.45 * volatility + 0.35 * entitlement + 0.20 * (1.0 - consistency)

        strict = clamp(self.team.personality_strictness)
        psyche = positives - lerp(0.35, 0.70, strict) * negatives

        # If team is desperate / conservative, they value psyche more
        if self.mood in (DraftMood.CONSERVATIVE, DraftMood.DESPERATE):
            psyche += 0.05 * strict

        return clamp(psyche)

    def _risk_penalty(self, p: Any) -> float:
        """
        Injury + boom/bust + general risk penalty, filtered by medical strictness.
        """
        injury = clamp(_get(p, "injury_risk", 0.3))
        boom_bust = clamp(_get(p, "boom_bust", _get(p, "variance", 0.4)))
        off_ice = clamp(_get(p, "off_ice_risk", 0.2))

        strict = clamp(self.team.medical_strictness)
        # Conservative teams amplify risk; aggressive teams reduce it
        conservative_bias = clamp(0.55 - self.team.risk_tolerance)  # >0 means conservative-ish

        penalty = (0.50 * injury + 0.35 * boom_bust + 0.15 * off_ice)
        penalty *= lerp(0.75, 1.35, strict + 0.25 * conservative_bias)

        return clamp(penalty)

    # ---------------------------
    # Tier assignment
    # ---------------------------

    def _assign_tiers(self, items: List[BoardItem]) -> None:
        """
        Create tiers by score clustering. Tiers are team-specific.
        """
        if not items:
            return

        # Determine thresholds using score drops
        # Higher tier_sensitivity => more tiers (more "cliffs")
        sens = clamp(self.team.tier_sensitivity)
        drop_threshold = lerp(0.09, 0.04, sens)  # sensitive => smaller threshold => more tiers

        tier = 0
        last = items[0].score
        items[0].tier = tier

        for i in range(1, len(items)):
            s = items[i].score
            drop = last - s
            if drop > drop_threshold:
                tier += 1
                last = s
            items[i].tier = tier

    # ---------------------------
    # Dynamic draft reactions
    # ---------------------------

    def _ingest_league_events(self, league_events: List[DraftEvent]) -> None:
        """
        Read the latest league events to update recent positions and pass counters.
        Assumes league_events are ordered by pick_number ascending.
        """
        if not league_events:
            return

        # Update recent positions from last N picks
        # Use only last recent_picks_window events
        recent = league_events[-self.ctx.recent_picks_window:]
        for ev in recent:
            item = self.by_id.get(ev.prospect_id)
            pos = item.position if item else "?"
            self._push_recent_position(pos)

        # Update pass counters: if a prospect hasn't been taken yet but was expected earlier,
        # we model "iceberg effect" by incrementing when they pass certain pick gates.
        # We approximate by counting how far past their tier rank they are.
        # (Your central draft engine can also directly call increment_passed(prospect_id).)

        # This board can only estimate using its own ranking:
        drafted_ids = {e.prospect_id for e in league_events}
        # Identify top candidates that are still undrafted and deep into the draft
        # We'll do a light bump to passed_counter for "surprising slips"
        for idx, item in enumerate(self.items[:60]):  # only consider top 60 for slip logic
            if item.prospect_id in drafted_ids:
                continue
            # If many picks have happened past where we'd "expect" this tier to go, increment passes
            # Expectation line: earlier tiers should go earlier
            expected_pick = 1 + (item.tier * 10) + idx * 0.15
            actual_pick = league_events[-1].pick_number
            if actual_pick > expected_pick + 8:
                self.passed_counter[item.prospect_id] = self.passed_counter.get(item.prospect_id, 0) + 1

    def _apply_iceberg_effect(self, avail: List[BoardItem]) -> None:
        """
        Increase rumor penalties for prospects with high pass counts,
        especially if team is rumor-susceptible and strict about personality/medical.
        """
        if not avail:
            return

        iceberg = clamp(self.ctx.iceberg_effect_strength)
        herd = clamp(self.team.rumor_susceptibility)

        for item in avail[:80]:
            passes = self.passed_counter.get(item.prospect_id, 0)
            if passes <= 0:
                # ease off any old rumor penalties
                item.rumor_penalty = max(0.0, item.rumor_penalty * 0.85)
                item.score = clamp(item.score + 0.02 * (1.0 - item.rumor_penalty))
                continue

            # Rumor penalty grows nonlinearly with passes
            pass_factor = clamp(sigmoid((passes - 1.0) * 1.1))  # 0..1
            strict = clamp(0.50 * self.team.personality_strictness + 0.50 * self.team.medical_strictness)

            penalty = iceberg * herd * (0.06 + 0.18 * pass_factor) * (0.55 + 0.45 * strict)

            # High-psyche players resist rumor slides more
            resistance = 0.35 + 0.65 * item.psyche_value
            penalty *= (1.0 - 0.35 * resistance)

            item.rumor_penalty = clamp(item.rumor_penalty + penalty, 0.0, 0.45)
            item.score = clamp(item.score - penalty)

            if "SLIDING" not in item.flags and item.rumor_penalty > 0.10:
                item.flags.append("SLIDING")

    def _apply_positional_run_pressure(self, avail: List[BoardItem]) -> None:
        """
        If a run on a position is happening, bump fit for that position (scarcity anxiety),
        especially for conservative/desperate teams.
        """
        if not self.recent_positions:
            return

        window = self.ctx.recent_picks_window
        recent = self.recent_positions[-window:]
        counts: Dict[str, int] = {}
        for pos in recent:
            counts[pos] = counts.get(pos, 0) + 1

        # Determine strongest run
        run_pos, run_count = None, 0
        for pos, c in counts.items():
            if pos != "?" and c > run_count:
                run_pos, run_count = pos, c

        if not run_pos or run_count < max(3, window // 3):
            return

        run_strength = clamp(self.ctx.run_strength)
        anxiety = 0.0
        if self.mood in (DraftMood.CONSERVATIVE, DraftMood.DESPERATE):
            anxiety = 0.10
        anxiety += 0.10 * (1.0 - self.team.risk_tolerance)

        bump = run_strength * anxiety * (run_count / window)

        for item in avail[:35]:
            if item.position == run_pos:
                item.score = clamp(item.score + bump * 0.12)
                item.fit_value = clamp(item.fit_value + bump * 0.10)
                if "RUN_BUMP" not in item.flags:
                    item.flags.append("RUN_BUMP")

    def _push_recent_position(self, pos: str) -> None:
        self.recent_positions.append(pos)
        # Keep a reasonable bound
        if len(self.recent_positions) > 64:
            self.recent_positions = self.recent_positions[-48:]

    # ---------------------------
    # Trade signaling (hooks)
    # ---------------------------

    def _trade_signal(self, pick_number: int, avail: List[BoardItem], chosen: BoardItem) -> Dict[str, Any]:
        """
        Returns a lightweight 'trade signal' dict that your trade engine can use.
        - Suggest trade up if tier cliff + "my guy" available just above expected range
        - Suggest trade down if board depth in tier is strong
        """
        if not avail:
            return {"action": "none"}

        top = avail[0]
        top_tier = top.tier

        tier_left = sum(1 for x in avail[:20] if x.tier == top_tier)
        tier_depth = clamp(tier_left / 8.0)

        tier_cliff = 1.0 - tier_depth  # higher => cliff

        # My guy urgency: if my guy is available but rumor penalties rising / tier cliff
        my_guy_avail = self.my_guy_id is not None and any(x.prospect_id == self.my_guy_id for x in avail[:25])
        my_guy = self.by_id.get(self.my_guy_id) if self.my_guy_id else None

        # Panic factor: insecure GM + coach influence + cliff
        panic = clamp(
            0.45 * (1.0 - self.team.gm_job_security) +
            0.25 * self.team.coach_influence +
            0.30 * tier_cliff
        )

        # Trade up signal
        if my_guy_avail and my_guy and (panic > 0.55 or my_guy.tier == top_tier and tier_left <= 3):
            return {
                "action": "consider_trade_up",
                "panic": round(panic, 3),
                "reason": "my_guy_or_tier_cliff",
                "target_prospect_id": my_guy.prospect_id,
                "target_name": my_guy.name,
                "target_tier": my_guy.tier,
            }

        # Trade down signal: if many players left in current tier and team is confident
        confident = self.mood == DraftMood.CONFIDENT and self.team.risk_tolerance >= 0.45
        if confident and tier_left >= 6 and self.rng.random() < 0.35:
            return {
                "action": "consider_trade_down",
                "reason": "tier_depth",
                "tier_left": tier_left,
                "top_tier": top_tier,
            }

        return {"action": "none"}

    # ---------------------------
    # Convenience: external hooks
    # ---------------------------

    def increment_passed(self, prospect_id: str, n: int = 1) -> None:
        """
        If your central draft engine detects a "surprising slide" event, call this.
        """
        self.passed_counter[prospect_id] = self.passed_counter.get(prospect_id, 0) + max(1, n)

    def debug_snapshot(self, drafted_ids: set[str], top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts for quick logging or UI.
        """
        out = []
        for x in self.top(drafted_ids, n=top_n):
            out.append({
                "id": x.prospect_id,
                "name": x.name,
                "pos": x.position,
                "score": round(x.score, 4),
                "tier": x.tier,
                "base": round(x.base_value, 3),
                "fit": round(x.fit_value, 3),
                "psyche": round(x.psyche_value, 3),
                "risk": round(x.risk_penalty, 3),
                "rumor": round(x.rumor_penalty, 3),
                "flags": list(x.flags),
                "passed": self.passed_counter.get(x.prospect_id, 0),
            })
        return out


# ---------------------------
# Minimal example usage (optional)
# ---------------------------

if __name__ == "__main__":
    # Example prospects (dict-like)
    prospects = [
        {
            "id": "p1", "name": "High Ceiling Chaos", "position": "C",
            "upside": 0.92, "floor": 0.45, "certainty": 0.40, "production": 0.65,
            "coachability": 0.35, "work_ethic": 0.55, "resilience": 0.40,
            "volatility": 0.80, "injury_risk": 0.25, "boom_bust": 0.85,
            "nhl_readiness": 0.35,
        },
        {
            "id": "p2", "name": "Safe Two-Way Stud", "position": "C",
            "upside": 0.72, "floor": 0.70, "certainty": 0.78, "production": 0.70,
            "coachability": 0.85, "work_ethic": 0.80, "resilience": 0.75,
            "volatility": 0.20, "injury_risk": 0.20, "boom_bust": 0.30,
            "nhl_readiness": 0.62,
        },
        {
            "id": "p3", "name": "Puck-Moving D Project", "position": "D",
            "upside": 0.80, "floor": 0.42, "certainty": 0.45, "production": 0.52,
            "coachability": 0.70, "work_ethic": 0.65, "resilience": 0.60,
            "volatility": 0.35, "injury_risk": 0.15, "boom_bust": 0.65,
            "nhl_readiness": 0.40,
        },
    ]

    team = TeamProfile(
        team_id="BOS",
        name="Boston Bruins",
        archetype="perpetual_mediocrity",
        risk_tolerance=0.38,
        gm_job_security=0.45,
        coach_patience=0.35,
        coach_influence=0.55,
        scouting_quality=0.60,
        analytics_bias=0.45,
        personality_strictness=0.70,
        medical_strictness=0.60,
        rumor_susceptibility=0.55,
        needs_by_position={"C": 0.75, "LW": 0.45, "RW": 0.45, "D": 0.60, "G": 0.40},
        timeline_pressure=0.55,
        owner_pressure=0.40,
        fan_pressure=0.55,
    )

    ctx = DraftContext(seed=123, year=2026)
    board = DraftBoard(team, ctx)
    board.build(prospects)

    drafted = set()
    league_events: List[DraftEvent] = []

    pid, meta = board.recommend_pick(pick_number=10, drafted_ids=drafted, league_events=league_events)
    print("RECOMMEND:", pid, meta["chosen"])
    print("TOP:", board.debug_snapshot(drafted, top_n=5))
