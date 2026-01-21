from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random

from app.sim_engine.entities.player import Player


# ==================================================
# TEAM ARCHETYPES
# ==================================================

class TeamArchetype:
    PATIENT_BUILDER = "patient_builder"
    WIN_NOW = "win_now"
    CHAOTIC = "chaotic"
    DRAFT_AND_DEVELOP = "draft_and_develop"
    MEDIOCRE = "perpetual_mediocrity"


# ==================================================
# TEAM STATUS (CRITICAL: lifecycle states)
# ==================================================

class TeamStatus:
    REBUILDING = "rebuilding"
    RETOOLING = "retooling"
    BUBBLE = "bubble"
    CONTENDING = "contending"
    POWERHOUSE = "powerhouse"
    DECLINING = "declining"
    DYSFUNCTIONAL = "dysfunctional"
    DESPERATE = "desperate"
    COMPLACENT = "complacent"


# ==================================================
# MARKET / OWNERSHIP / REPUTATION
# ==================================================

@dataclass
class MarketProfile:
    """
    Mostly-stable environmental context.
    These do NOT swing wildly year-to-year.
    """
    market_size: str = "medium"          # small / medium / large
    media_pressure: float = 0.5          # 0..1
    fan_expectations: float = 0.5        # 0..1
    tax_advantage: float = 0.5           # 0..1 (higher = more attractive)
    weather_quality: float = 0.5         # 0..1 (higher = more attractive)
    travel_burden: float = 0.5           # 0..1 (higher = worse)

    def clamp(self):
        self.media_pressure = max(0.0, min(1.0, self.media_pressure))
        self.fan_expectations = max(0.0, min(1.0, self.fan_expectations))
        self.tax_advantage = max(0.0, min(1.0, self.tax_advantage))
        self.weather_quality = max(0.0, min(1.0, self.weather_quality))
        self.travel_burden = max(0.0, min(1.0, self.travel_burden))


@dataclass
class OwnershipProfile:
    """
    Ownership is an active force. This drives panic, patience, spending, meddling.
    """
    patience: float = 0.5               # 0..1 (higher = more patient)
    ambition: float = 0.5               # 0..1 (higher = wants big outcomes)
    budget_willingness: float = 0.5     # 0..1 (higher = spend)
    meddling: float = 0.5               # 0..1 (higher = chaos / overreaction)

    def clamp(self):
        self.patience = max(0.0, min(1.0, self.patience))
        self.ambition = max(0.0, min(1.0, self.ambition))
        self.budget_willingness = max(0.0, min(1.0, self.budget_willingness))
        self.meddling = max(0.0, min(1.0, self.meddling))


@dataclass
class Reputation:
    """
    Reputation changes slowly, but can be damaged quickly by scandals / dysfunction.
    Affects: player willingness to sign, pressure, and internal stability.
    """
    league_reputation: float = 0.5
    player_reputation: float = 0.5
    management_reputation: float = 0.5
    development_reputation: float = 0.5

    def clamp(self):
        self.league_reputation = max(0.0, min(1.0, self.league_reputation))
        self.player_reputation = max(0.0, min(1.0, self.player_reputation))
        self.management_reputation = max(0.0, min(1.0, self.management_reputation))
        self.development_reputation = max(0.0, min(1.0, self.development_reputation))


# ==================================================
# PERFORMANCE MEMORY
# ==================================================

@dataclass
class PerformanceMemory:
    """
    Teams remember. Pressure is a 3–5 year story, not last night’s scoreboard.
    """
    last_win_pcts: List[float] = field(default_factory=list)  # most recent at end
    playoff_appearances: int = 0
    series_wins: int = 0
    years_since_playoffs: int = 0
    years_since_round2: int = 0
    years_since_cup: int = 0

    def record_season(self, win_pct: float, made_playoffs: bool, series_wins: int, won_cup: bool):
        self.last_win_pcts.append(win_pct)
        if len(self.last_win_pcts) > 5:
            self.last_win_pcts.pop(0)

        if made_playoffs:
            self.playoff_appearances += 1
            self.years_since_playoffs = 0
        else:
            self.years_since_playoffs += 1

        if series_wins >= 1:
            self.years_since_round2 = 0
        else:
            self.years_since_round2 += 1

        self.series_wins += max(0, series_wins)

        if won_cup:
            self.years_since_cup = 0
        else:
            self.years_since_cup += 1

    def rolling_win_pct(self) -> float:
        if not self.last_win_pcts:
            return 0.5
        return sum(self.last_win_pcts) / len(self.last_win_pcts)


# ==================================================
# ROSTER QUALITY PROXY (until full league/roster sim)
# ==================================================

@dataclass
class RosterQuality:
    """
    Temporary system that trends gradually.
    star_count: 0–5
    core_count: 3–10
    depth_quality: 0–1
    """
    star_count: int = 1
    core_count: int = 5
    depth_quality: float = 0.5

    def clamp(self):
        self.star_count = max(0, min(5, int(self.star_count)))
        self.core_count = max(0, min(10, int(self.core_count)))
        self.depth_quality = max(0.0, min(1.0, self.depth_quality))

    def strength_score(self) -> float:
        # Soft mapping into 0..1 (not “true overall”, just expectation proxy)
        score = 0.10 * self.star_count + 0.04 * self.core_count + 0.50 * self.depth_quality
        return max(0.0, min(1.0, score))


# ==================================================
# TEAM STATE SNAPSHOT (dynamic)
# ==================================================

@dataclass
class TeamState:
    """
    Dynamic season-to-season state.
    """
    competitive_score: float = 0.5            # contender vs rebuild (0..1)
    team_morale: float = 0.5                  # locker room health (0..1)
    organizational_pressure: float = 0.5      # heat from fans/media/ownership (0..1)
    stability: float = 0.5                    # coach + GM stability (0..1)

    status: str = TeamStatus.BUBBLE

    # Organizational stability subfactors (simplified, but gives you levers)
    ownership_stability: float = 0.7          # 0..1
    arena_security: float = 0.8               # 0..1
    financial_health: float = 0.7             # 0..1

    # Event flags for the season
    triggered_events: List[str] = field(default_factory=list)

    def clamp(self):
        self.competitive_score = max(0.0, min(1.0, self.competitive_score))
        self.team_morale = max(0.0, min(1.0, self.team_morale))
        self.organizational_pressure = max(0.0, min(1.0, self.organizational_pressure))
        self.stability = max(0.0, min(1.0, self.stability))
        self.ownership_stability = max(0.0, min(1.0, self.ownership_stability))
        self.arena_security = max(0.0, min(1.0, self.arena_security))
        self.financial_health = max(0.0, min(1.0, self.financial_health))


# ==================================================
# TEAM ENTITY
# ==================================================

class Team:
    """
    Represents an NHL organization.

    A Team is NOT just a roster — it is a decision-making entity
    that applies pressure, culture, patience, and opportunity
    onto player careers.

    This file ENFORCES team cycles so you never get 25 years of
    permanent rebuilding or permanent contending by accident.
    """

    def __init__(
        self,
        *,
        team_id: int,
        city: str,
        name: str,
        division: str,
        conference: str,
        archetype: str,
        rng: random.Random,
    ):
        self.team_id = team_id
        self.city = city
        self.name = name
        self.division = division
        self.conference = conference
        self.archetype = archetype
        self.rng = rng

        # -------------------------------
        # Rosters
        # -------------------------------
        self.roster: List[Player] = []
        self.scratches: List[Player] = []
        self.prospects: List[Player] = []
        self.injured_reserve: List[Player] = []

        # -------------------------------
        # Persistent identity blocks
        # -------------------------------
        self.market = self._init_market_profile()
        self.ownership = self._init_ownership_profile()
        self.reputation = self._init_reputation()

        # -------------------------------
        # Organizational philosophy
        # -------------------------------
        self.development_quality = self._init_development_quality()
        self.prospect_patience = self._init_prospect_patience()
        self.risk_tolerance = self._init_risk_tolerance()

        # -------------------------------
        # Dynamic state + memory
        # -------------------------------
        self.state = TeamState()
        self.memory = PerformanceMemory()
        self.roster_quality = self._init_roster_quality()

        # -------------------------------
        # Franchise memory (narrative log)
        # -------------------------------
        self.franchise_memory: List[str] = []

    # --------------------------------------------------
    # INITIALIZATION HELPERS
    # --------------------------------------------------

    def _init_market_profile(self) -> MarketProfile:
        """
        Lightweight market defaults. You can later load these from a league data file.
        For now: infer from city name (very rough), else archetype-based bias.
        """
        city = (self.city or "").lower()

        # very small heuristic list (safe defaults if unknown)
        big_markets = {"toronto", "montreal", "new york", "los angeles", "chicago", "boston", "philadelphia"}
        small_markets = {"winnipeg", "ottawa", "calgary", "edmonton", "columbus", "buffalo", "nashville"}

        if city in big_markets:
            mp = MarketProfile(market_size="large", media_pressure=0.80, fan_expectations=0.80, tax_advantage=0.40, weather_quality=0.45, travel_burden=0.45)
        elif city in small_markets:
            mp = MarketProfile(market_size="small", media_pressure=0.45, fan_expectations=0.55, tax_advantage=0.45, weather_quality=0.50, travel_burden=0.60)
        else:
            mp = MarketProfile(market_size="medium", media_pressure=0.55, fan_expectations=0.55, tax_advantage=0.50, weather_quality=0.55, travel_burden=0.50)

        # Archetype seasoning
        if self.archetype == TeamArchetype.WIN_NOW:
            mp.fan_expectations += 0.05
            mp.media_pressure += 0.05
        elif self.archetype == TeamArchetype.PATIENT_BUILDER:
            mp.fan_expectations -= 0.03
        elif self.archetype == TeamArchetype.CHAOTIC:
            mp.media_pressure += 0.05

        # Tiny drift randomness (market doesn’t “flip”, it drifts)
        mp.media_pressure += self.rng.uniform(-0.02, 0.02)
        mp.fan_expectations += self.rng.uniform(-0.02, 0.02)
        mp.tax_advantage += self.rng.uniform(-0.02, 0.02)
        mp.weather_quality += self.rng.uniform(-0.02, 0.02)
        mp.travel_burden += self.rng.uniform(-0.02, 0.02)

        mp.clamp()
        return mp

    def _init_ownership_profile(self) -> OwnershipProfile:
        base = {
            TeamArchetype.WIN_NOW: OwnershipProfile(patience=0.35, ambition=0.80, budget_willingness=0.70, meddling=0.55),
            TeamArchetype.PATIENT_BUILDER: OwnershipProfile(patience=0.70, ambition=0.55, budget_willingness=0.55, meddling=0.35),
            TeamArchetype.DRAFT_AND_DEVELOP: OwnershipProfile(patience=0.65, ambition=0.50, budget_willingness=0.45, meddling=0.30),
            TeamArchetype.MEDIOCRE: OwnershipProfile(patience=0.55, ambition=0.45, budget_willingness=0.50, meddling=0.40),
            TeamArchetype.CHAOTIC: OwnershipProfile(patience=0.40, ambition=0.60, budget_willingness=0.55, meddling=0.80),
        }.get(self.archetype, OwnershipProfile())

        # market can influence patience/ambition slightly
        if self.market.market_size == "large":
            base.patience -= 0.05
            base.ambition += 0.05
        if self.market.market_size == "small":
            base.patience += 0.03

        # random seasoning
        base.patience += self.rng.uniform(-0.05, 0.05)
        base.ambition += self.rng.uniform(-0.05, 0.05)
        base.budget_willingness += self.rng.uniform(-0.05, 0.05)
        base.meddling += self.rng.uniform(-0.05, 0.05)
        base.clamp()
        return base

    def _init_reputation(self) -> Reputation:
        # Start roughly neutral; archetype influences perception slightly
        rep = Reputation()

        if self.archetype == TeamArchetype.DRAFT_AND_DEVELOP:
            rep.development_reputation += 0.10
        elif self.archetype == TeamArchetype.CHAOTIC:
            rep.management_reputation -= 0.10
            rep.player_reputation -= 0.05
        elif self.archetype == TeamArchetype.WIN_NOW:
            rep.league_reputation += 0.05

        # tiny noise
        rep.league_reputation += self.rng.uniform(-0.03, 0.03)
        rep.player_reputation += self.rng.uniform(-0.03, 0.03)
        rep.management_reputation += self.rng.uniform(-0.03, 0.03)
        rep.development_reputation += self.rng.uniform(-0.03, 0.03)
        rep.clamp()
        return rep

    def _init_roster_quality(self) -> RosterQuality:
        # Start around mid, with archetype bias
        base = RosterQuality(star_count=1, core_count=5, depth_quality=0.50)

        if self.archetype == TeamArchetype.WIN_NOW:
            base.star_count += 1
            base.depth_quality += 0.05
        elif self.archetype == TeamArchetype.DRAFT_AND_DEVELOP:
            base.core_count += 1
        elif self.archetype == TeamArchetype.CHAOTIC:
            base.depth_quality -= 0.05

        # slight randomness (but not huge swings)
        base.star_count += int(round(self.rng.uniform(-1, 1)))
        base.core_count += int(round(self.rng.uniform(-1, 1)))
        base.depth_quality += self.rng.uniform(-0.05, 0.05)
        base.clamp()
        return base

    def _init_development_quality(self) -> float:
        base = {
            TeamArchetype.DRAFT_AND_DEVELOP: 0.75,
            TeamArchetype.PATIENT_BUILDER: 0.65,
            TeamArchetype.WIN_NOW: 0.50,
            TeamArchetype.MEDIOCRE: 0.45,
            TeamArchetype.CHAOTIC: 0.35,
        }.get(self.archetype, 0.5)

        # dev reputation subtly matters
        base += (self.reputation.development_reputation - 0.5) * 0.10

        return max(0.3, min(0.85, base + self.rng.uniform(-0.05, 0.05)))

    def _init_prospect_patience(self) -> float:
        base = {
            TeamArchetype.DRAFT_AND_DEVELOP: 0.80,
            TeamArchetype.PATIENT_BUILDER: 0.70,
            TeamArchetype.MEDIOCRE: 0.50,
            TeamArchetype.WIN_NOW: 0.35,
            TeamArchetype.CHAOTIC: 0.20,
        }.get(self.archetype, 0.5)

        # ownership patience nudges prospect patience
        base += (self.ownership.patience - 0.5) * 0.10

        return max(0.1, min(0.9, base + self.rng.uniform(-0.05, 0.05)))

    def _init_risk_tolerance(self) -> float:
        base = {
            TeamArchetype.CHAOTIC: 0.85,
            TeamArchetype.WIN_NOW: 0.70,
            TeamArchetype.MEDIOCRE: 0.50,
            TeamArchetype.PATIENT_BUILDER: 0.40,
            TeamArchetype.DRAFT_AND_DEVELOP: 0.30,
        }.get(self.archetype, 0.5)

        base += (self.ownership.meddling - 0.5) * 0.10

        return max(0.2, min(0.9, base + self.rng.uniform(-0.05, 0.05)))

    # --------------------------------------------------
    # ROSTER MANAGEMENT
    # --------------------------------------------------

    def add_player(self, player: Player):
        if player not in self.roster:
            self.roster.append(player)

    def remove_player(self, player: Player):
        if player in self.roster:
            self.roster.remove(player)

    def assign_prospect(self, player: Player):
        if player not in self.prospects:
            self.prospects.append(player)

    # --------------------------------------------------
    # ROLE & CONTEXT
    # --------------------------------------------------

    def role_mismatch_factor(self, player: Player) -> float:
        """
        Returns a 0–1 value representing how poorly the player fits
        their current role on this team.

        NOTE: Until full roster roles exist, we infer role from roster depth index.
        """
        if player in self.prospects:
            return 0.0

        depth_index = self.roster.index(player) if player in self.roster else 99

        if depth_index < 3:
            return 0.0
        if depth_index < 6:
            return 0.2
        if depth_index < 12:
            return 0.4
        return 0.6

    # --------------------------------------------------
    # DEVELOPMENT MODIFIERS
    # --------------------------------------------------

    def development_modifier(self, player: Player) -> float:
        """
        Final multiplier applied to player growth.

        This should feel like:
        - stable org + good development → growth
        - chaotic org + pressure + low morale → stagnation
        """
        modifier = self.development_quality

        # prospects benefit from patience + stability
        if player.age < 23:
            modifier *= 1.0 + (self.prospect_patience - 0.5)

        # pressure harms development
        modifier *= 1.0 - self.state.organizational_pressure * 0.25

        # dysfunction & low morale hurts skill growth
        morale_penalty = (0.5 - self.state.team_morale) * 0.20
        modifier *= 1.0 - max(0.0, morale_penalty)

        # organizational stability helps
        modifier *= 0.90 + self.state.stability * 0.20

        # clamp
        return max(0.5, min(1.5, modifier))

    # --------------------------------------------------
    # INTERNAL: TEAM EXPECTATIONS / PRESSURE
    # --------------------------------------------------

    def _expected_win_pct(self) -> float:
        """
        Expectation proxy: roster quality + market expectations + ownership ambition + reputation.
        This is NOT an on-ice simulator; it’s the expectation baseline that creates pressure.
        """
        base = 0.45 + 0.25 * self.roster_quality.strength_score()
        base += (self.market.fan_expectations - 0.5) * 0.10
        base += (self.ownership.ambition - 0.5) * 0.08
        base += (self.reputation.league_reputation - 0.5) * 0.05
        return max(0.30, min(0.70, base))

    def _pressure_from_context(self) -> float:
        """
        Fan/media/ownership pressure baseline.
        """
        p = 0.25
        p += self.market.media_pressure * 0.35
        p += self.market.fan_expectations * 0.25
        p += (1.0 - self.ownership.patience) * 0.20
        p += self.ownership.meddling * 0.10
        return max(0.0, min(1.0, p))

    # --------------------------------------------------
    # BLACK SWAN EVENTS (rare, decisive)
    # --------------------------------------------------

    def _roll_black_swan(self) -> Optional[str]:
        """
        <1% chance per season baseline. Archetype/instability can raise it slightly.
        These are org-level shocks that override normal logic.
        """
        base = 0.003  # 0.3%
        base += (1.0 - self.state.stability) * 0.005
        base += (1.0 - self.state.financial_health) * 0.004
        if self.archetype == TeamArchetype.CHAOTIC:
            base += 0.004  # chaotic teams attract chaos

        # cap under 1.5% overall (still rare, but will happen in long sims)
        chance = min(0.015, base)

        if self.rng.random() > chance:
            return None

        events = [
            "ownership_scandal",
            "gm_meltdown",
            "coach_room_lost",
            "arena_drama",
            "public_trade_demand",
            "league_punishment",
            "ownership_sale",
        ]
        return self.rng.choice(events)

    def _apply_black_swan(self, event: str):
        """
        Apply high-impact state changes.
        """
        self.state.triggered_events.append(event)
        self.franchise_memory.append(f"BLACK_SWAN:{event}")

        if event == "ownership_scandal":
            self.state.ownership_stability -= 0.25
            self.state.team_morale -= 0.12
            self.reputation.management_reputation -= 0.15
            self.reputation.league_reputation -= 0.10

        elif event == "gm_meltdown":
            self.state.stability -= 0.18
            self.state.organizational_pressure += 0.18
            self.reputation.management_reputation -= 0.18

        elif event == "coach_room_lost":
            self.state.team_morale -= 0.20
            self.state.stability -= 0.12

        elif event == "arena_drama":
            self.state.arena_security -= 0.25
            self.state.financial_health -= 0.10
            self.state.organizational_pressure += 0.15

        elif event == "public_trade_demand":
            self.state.team_morale -= 0.10
            self.state.organizational_pressure += 0.12
            self.reputation.player_reputation -= 0.10

        elif event == "league_punishment":
            self.state.organizational_pressure += 0.20
            self.reputation.league_reputation -= 0.15
            self.state.stability -= 0.08

        elif event == "ownership_sale":
            # Can be good or bad
            swing = self.rng.uniform(-0.20, 0.20)
            self.ownership.patience += swing * -0.5
            self.ownership.ambition += swing * 0.7
            self.ownership.meddling += self.rng.uniform(-0.10, 0.10)
            self.state.ownership_stability += 0.10
            self.state.organizational_pressure += self.rng.uniform(-0.05, 0.10)

        self.ownership.clamp()
        self.reputation.clamp()
        self.state.clamp()

    # --------------------------------------------------
    # TEAM STATUS ENGINE (enforces cycles)
    # --------------------------------------------------

    def _infer_status(self, win_pct: float) -> str:
        """
        Status is not record alone. It’s:
        - performance window
        - roster quality proxy
        - market pressure
        - ownership mood
        - stability/dysfunction
        """
        roll = self.memory.rolling_win_pct()
        strength = self.roster_quality.strength_score()
        pressure = self.state.organizational_pressure
        stable = self.state.stability
        morale = self.state.team_morale

        # Dysfunction overrides
        if stable < 0.25 and pressure > 0.75:
            return TeamStatus.DYSFUNCTIONAL

        # Powerhouse: strong roster + sustained performance
        if roll > 0.62 and strength > 0.70 and pressure > 0.45:
            return TeamStatus.POWERHOUSE

        # Contending
        if roll > 0.56 and strength > 0.55:
            return TeamStatus.CONTENDING

        # Desperate: high pressure but only bubble performance
        if pressure > 0.75 and roll < 0.55 and self.market.market_size in ("large", "medium"):
            return TeamStatus.DESPERATE

        # Bubble
        if 0.48 <= roll <= 0.56:
            return TeamStatus.BUBBLE

        # Declining: used to be good, now slipping
        if self.memory.years_since_round2 >= 3 and roll < 0.52 and strength < 0.55 and pressure > 0.55:
            return TeamStatus.DECLINING

        # Rebuild/retool depends on strength + patience
        if roll < 0.48:
            if self.ownership.patience > 0.55 or self.archetype in (TeamArchetype.DRAFT_AND_DEVELOP, TeamArchetype.PATIENT_BUILDER):
                return TeamStatus.REBUILDING
            return TeamStatus.RETOOLING

        # Complacent: decent record, low pressure, low ambition
        if roll > 0.52 and pressure < 0.45 and self.ownership.ambition < 0.45:
            return TeamStatus.COMPLACENT

        return TeamStatus.RETOOLING

    def _enforce_trajectory(self):
        """
        Soft correction forces to prevent 25-year static realities.
        This does NOT hard-script success. It changes probabilities and drift.
        """
        # The longer you miss playoffs, the more likely ownership forces a direction change.
        miss = self.memory.years_since_playoffs

        # Rebuilds should not be eternal: lottery luck / prospect pipeline should drift upward slightly.
        if self.state.status == TeamStatus.REBUILDING and miss >= 4:
            self.roster_quality.depth_quality += 0.03 + self.rng.uniform(0.00, 0.03)
            if self.rng.random() < 0.25:
                self.roster_quality.core_count += 1

        # Powerhouses face pressure and volatility: keep it hard to stay on top forever
        if self.state.status in (TeamStatus.POWERHOUSE, TeamStatus.CONTENDING) and self.memory.years_since_cup >= 4:
            # "wasted prime" pressure in big markets
            spike = 0.02 + self.market.media_pressure * 0.03
            self.state.organizational_pressure += spike
            # slow erosion / cap squeeze proxy
            self.roster_quality.depth_quality -= 0.02 + self.rng.uniform(0.00, 0.02)

        # Desperate/Dysfunctional teams get unstable
        if self.state.status in (TeamStatus.DESPERATE, TeamStatus.DYSFUNCTIONAL):
            self.state.stability -= 0.03 + self.ownership.meddling * 0.04

        # Clamp quality drift
        self.roster_quality.clamp()
        self.state.clamp()

    def _update_roster_quality_proxy(self, win_pct: float):
        """
        Changes slowly, with memory. No 0 stars -> 5 stars overnight.
        """
        expected = self._expected_win_pct()
        delta = (win_pct - expected)

        # If you out-perform expectation, you "feel" better and might be stronger next year.
        # If you under-perform, you may lose depth / cohesion.
        self.roster_quality.depth_quality += delta * 0.10 + self.rng.uniform(-0.015, 0.015)

        # Stars change rarely. Use long-run trend.
        roll = self.memory.rolling_win_pct()
        if roll > 0.58 and self.rng.random() < 0.08:
            self.roster_quality.star_count += 1
        elif roll < 0.46 and self.rng.random() < 0.10:
            self.roster_quality.star_count -= 1

        # Core count drifts
        self.roster_quality.core_count += int(round(delta * 2.0 + self.rng.uniform(-0.3, 0.3)))

        self.roster_quality.clamp()

    # --------------------------------------------------
    # TEAM MORALE & PRESSURE (main seasonal update)
    # --------------------------------------------------

    def update_team_state(self, *, win_pct: float):
        """
        Update morale, pressure, stability, reputation, memory, and lifecycle status after a season.

        IMPORTANT:
        - This is where we prevent 25-year static rebuilds.
        - This is where market context matters.
        - This is where rare, catastrophic org events can happen.
        """
        self.state.triggered_events = []

        expected = self._expected_win_pct()
        context_pressure = self._pressure_from_context()

        # Performance effect (relative to expectation matters more than raw win%)
        perf_delta = win_pct - expected

        # Morale: winning helps, but so does exceeding expectation
        self.state.team_morale += (win_pct - 0.5) * 0.25 + perf_delta * 0.20

        # Competitive score: crude, but should move
        self.state.competitive_score += (win_pct - 0.5) * 0.35 + perf_delta * 0.10

        # Pressure: context baseline + underperformance + market
        self.state.organizational_pressure = (
            0.65 * self.state.organizational_pressure
            + 0.35 * context_pressure
        )
        # underperforming spikes pressure, overperforming relieves it slightly (but less in big markets)
        market_spike = 0.12 + 0.10 * self.market.media_pressure
        self.state.organizational_pressure += (-perf_delta) * market_spike
        self.state.organizational_pressure -= max(0.0, perf_delta) * (0.04 + 0.02 * (1.0 - self.market.media_pressure))

        # Stability: pressure + meddling + chaos reduces stability, patient builders hold steadier
        stability_hit = self.state.organizational_pressure * (0.04 + 0.06 * self.ownership.meddling)
        stability_help = (self.ownership.patience - 0.5) * 0.03
        archetype_noise = 0.0
        if self.archetype == TeamArchetype.CHAOTIC:
            archetype_noise -= 0.03
        elif self.archetype == TeamArchetype.PATIENT_BUILDER:
            archetype_noise += 0.01

        self.state.stability += stability_help + archetype_noise - stability_hit + self.rng.uniform(-0.02, 0.02)

        # Financial health: rough proxy (winning and market help attendance)
        self.state.financial_health += (win_pct - 0.5) * (0.06 + 0.05 * (1.0 if self.market.market_size == "large" else 0.6))
        self.state.financial_health -= (1.0 - win_pct) * (0.03 + 0.02 * self.market.travel_burden)

        # Arena security: mostly stable, but poor finances can cause drift
        self.state.arena_security += (self.state.financial_health - 0.6) * 0.02 + self.rng.uniform(-0.01, 0.01)

        # Ownership stability: pressure hurts, success helps (slightly)
        self.state.ownership_stability += (win_pct - 0.5) * 0.03 - self.state.organizational_pressure * 0.02 + self.rng.uniform(-0.01, 0.01)

        # Reputation: slow drift; scandals hit hard (handled elsewhere)
        self.reputation.league_reputation += perf_delta * 0.03
        self.reputation.player_reputation += (self.state.team_morale - 0.5) * 0.02 - self.state.organizational_pressure * 0.01
        self.reputation.management_reputation += perf_delta * 0.02 - (0.5 - self.state.stability) * 0.02
        self.reputation.development_reputation += (self.development_quality - 0.5) * 0.01

        # Clamp before events
        self.state.clamp()
        self.reputation.clamp()

        # Black swan roll + apply
        event = self._roll_black_swan()
        if event:
            self._apply_black_swan(event)

        # Record memory (for now: playoffs/series/cup are not simulated here, so keep them false/0)
        # Later: pass these in from your season sim results.
        self.memory.record_season(win_pct=win_pct, made_playoffs=False, series_wins=0, won_cup=False)

        # Update roster-quality proxy AFTER memory update
        self._update_roster_quality_proxy(win_pct)

        # Determine status
        prev_status = self.state.status
        self.state.status = self._infer_status(win_pct)
        if self.state.status != prev_status:
            self.franchise_memory.append(f"STATUS_CHANGE:{prev_status}->{self.state.status}")

        # Trajectory enforcement (soft corrections)
        self._enforce_trajectory()

        # Final clamp
        self.state.clamp()

    # --------------------------------------------------
    # SIGNALS FOR ENGINE / AI (STRICT CONTRACT)
    # --------------------------------------------------

    def team_context_for_player(self, player: Player) -> dict:
        """
        Returns ALL keys required by SimEngine.
        Never remove keys from this dict.
        """
        rebuild_mode = max(0.0, 1.0 - self.state.competitive_score)

        return {
            # 🔑 REQUIRED BY ENGINE
            "rebuild_mode": rebuild_mode,
            "role_mismatch": self.role_mismatch_factor(player),
            "stability": self.state.stability,
            "dev_modifier": self.development_modifier(player),

            # 🔬 OPTIONAL / FUTURE USE (safe to add more; do not remove existing ones)
            "team_morale": self.state.team_morale,
            "team_pressure": self.state.organizational_pressure,

            # extra context hooks (optional, but useful)
            "team_status": self.state.status,
            "market_pressure": self.market.media_pressure,
            "fan_expectations": self.market.fan_expectations,
            "ownership_meddling": self.ownership.meddling,
            "ownership_patience": self.ownership.patience,
            "rep_player": self.reputation.player_reputation,
            "rep_mgmt": self.reputation.management_reputation,
        }

    # --------------------------------------------------
    # DEBUG
    # --------------------------------------------------

    def debug_dump(self):
        print(f"\n=== TEAM DEBUG: {self.city} {self.name} ===")
        print(f"Archetype              : {self.archetype}")
        print(f"Status                 : {self.state.status}")
        print("--- MARKET ---")
        print(f"Market Size            : {self.market.market_size}")
        print(f"Media Pressure         : {self.market.media_pressure:.3f}")
        print(f"Fan Expectations       : {self.market.fan_expectations:.3f}")
        print(f"Tax Advantage          : {self.market.tax_advantage:.3f}")
        print(f"Weather Quality        : {self.market.weather_quality:.3f}")
        print(f"Travel Burden          : {self.market.travel_burden:.3f}")
        print("--- OWNERSHIP ---")
        print(f"Patience               : {self.ownership.patience:.3f}")
        print(f"Ambition               : {self.ownership.ambition:.3f}")
        print(f"Budget Willingness     : {self.ownership.budget_willingness:.3f}")
        print(f"Meddling               : {self.ownership.meddling:.3f}")
        print("--- REPUTATION ---")
        print(f"League                 : {self.reputation.league_reputation:.3f}")
        print(f"Player                 : {self.reputation.player_reputation:.3f}")
        print(f"Management             : {self.reputation.management_reputation:.3f}")
        print(f"Development            : {self.reputation.development_reputation:.3f}")
        print("--- ORG PHILOSOPHY ---")
        print(f"Development Quality    : {self.development_quality:.3f}")
        print(f"Prospect Patience      : {self.prospect_patience:.3f}")
        print(f"Risk Tolerance         : {self.risk_tolerance:.3f}")
        print("--- DYNAMIC STATE ---")
        print(f"Competitive Score      : {self.state.competitive_score:.3f}")
        print(f"Team Morale            : {self.state.team_morale:.3f}")
        print(f"Org Pressure           : {self.state.organizational_pressure:.3f}")
        print(f"Stability              : {self.state.stability:.3f}")
        print(f"Ownership Stability    : {self.state.ownership_stability:.3f}")
        print(f"Arena Security         : {self.state.arena_security:.3f}")
        print(f"Financial Health       : {self.state.financial_health:.3f}")
        print("--- ROSTER QUALITY PROXY ---")
        print(f"Stars                  : {self.roster_quality.star_count}")
        print(f"Core Count             : {self.roster_quality.core_count}")
        print(f"Depth Quality          : {self.roster_quality.depth_quality:.3f}")
        print("--- MEMORY ---")
        print(f"Rolling Win% (5y)      : {self.memory.rolling_win_pct():.3f}")
        print(f"Years Since Playoffs   : {self.memory.years_since_playoffs}")
        print(f"Years Since Round2     : {self.memory.years_since_round2}")
        print(f"Years Since Cup        : {self.memory.years_since_cup}")
        if self.state.triggered_events:
            print("--- EVENTS ---")
            for e in self.state.triggered_events:
                print(f"Triggered              : {e}")
        print("========================================\n")
