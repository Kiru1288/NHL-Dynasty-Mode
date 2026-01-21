from __future__ import annotations

"""
NHL DYNASTY MODE — SIM ENGINE (CORE ORCHESTRATOR)
=================================================

This file is intentionally "big" and stable.

What belongs here:
- The SimEngine class (year loop orchestration)
- Adapters between your entities + AI systems + league macro + contracts
- Debug printing helpers
- OPTIONAL: lightweight stat sampling helpers (so run_sim.py can call sim.sample_stat / sim.stat_percentile)
  without importing Player_Stats directly.

What MUST NOT belong here:
- run_sim.py runner / __main__ entrypoint
- random player factory
- team factory
- file output / redirect_stdout logic

If you see create_random_player / dump_team_snapshot / redirect_stdout in this file,
it means engine.py got overwritten accidentally. Keep those in run_sim.py ONLY.
"""

from dataclasses import asdict
from typing import Any, Dict, Optional, List, Tuple, Callable
import math
import random
import time


# -------------------------------
# AI systems
# -------------------------------
from app.sim_engine.ai.personality import (
    PersonalityFactory,
    PersonalityArchetype,
    PersonalityBehavior,
    BehaviorContext,
)

from app.sim_engine.ai.ai_manager import AIManager
from app.sim_engine.ai.morale_engine import MoraleEngine, MoraleState
from app.sim_engine.ai.career_arc import CareerArcEngine
from app.sim_engine.ai.injury_risk import InjuryRiskEngine
from app.sim_engine.ai.retirement_engine import RetirementEngine
from app.sim_engine.ai.randomness import RandomnessEngine
from app.sim_engine.context.League_Stats import LeagueStats
from app.sim_engine.context.Player_Stats import PlayerStatsEngine
from app.sim_engine.ai.coach import (
    Coach,
    CoachStyle,
    CoachImpact,
)



# -------------------------------
# Entities
# -------------------------------
from app.sim_engine.entities.player import Player
from app.sim_engine.entities.team import Team
from app.sim_engine.entities.league import League


# -------------------------------
# Contract system
# -------------------------------
from app.sim_engine.entities.contract import (
    TeamProfile,
    MarketProfile,
    OwnershipProfile,
    ReputationProfile,
    OrgPhilosophy,
    TeamDynamicState,
    TeamRosterProxy,
    PlayerProfile,
    PlayerPersonality,
    PlayerCareerState,
    PlayerMemory,
    AgentProfile,
    negotiate_contract,
    ContractContextKind,
)




# =====================================================================
# SMALL HELPERS
# =====================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def sigmoid(x: float) -> float:
    # stable-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def soft_clip(x: float, max_abs: float) -> float:
    # compress extremes smoothly
    if max_abs <= 0:
        return 0.0
    return max_abs * math.tanh(x / max_abs)


# =====================================================================
# FALLBACK STAT MODEL
# =====================================================================
# This is NOT your real Player_Stats.py. It’s a stability layer so sim runs today.
# If Player_Stats.py exists and has the functions below, we use it instead.
#
# Domain: "skater" | "goalie"
# Metric examples:
#   skater: goals, assists, points, shots, ixG, xGF%, CF%, WAR
#   goalie: save_pct, gaa, gsaa, quality_start_pct
#
# Tier examples:
#   "elite", "top_line", "middle_six", "bottom_six"
#   "starter", "tandem", "backup"
# =====================================================================




# =====================================================================
# SIM ENGINE
# =====================================================================

class SimEngine:
    """
    CORE SIMULATION ENGINE (STABLE)

    Responsibilities:
    - Orchestrates yearly flow
    - Advances league MACRO state (league.py)
    - Updates AI, morale, injuries, careers (ai/*)
    - Runs contracts & retirement (contract.py, retirement_engine.py)
    - Provides stable helper APIs for run_sim.py:
        - sample_stat(...)
        - stat_percentile(...)
        - get_league_context_snapshot()
        - get_last_contract_snapshot()

    This file should not be rewritten frequently.
    """

    # --------------------------------------------------
    # Construction
    # --------------------------------------------------

    def __init__(self, seed: int | None = None):
        self.year: int = 0
        self.seed: int = seed if seed is not None else random.randrange(1, 10**18)
        self.rng: random.Random = random.Random(self.seed)
        self.retired: bool = False

        # ------------------------------------
        # League ecosystem (MACRO ONLY)
        # ------------------------------------
        self.league: League = League(seed=self.seed)

        # Cached league outputs for the latest season
        self.last_league_context: dict | None = None
        self.last_league_forecast: dict | None = None
        self.last_league_shocks: list[dict] = []

        # ------------------------------------
# Stats engines (season-level)
# ------------------------------------
        self.league_stats: LeagueStats | None = None
        self.player_stats_engine: PlayerStatsEngine | None = None


        # ------------------------------------
        # Core systems
        # ------------------------------------
        self.ai_manager = AIManager(self.rng)
        self.morale_engine = MoraleEngine()
        self.career_arc_engine = CareerArcEngine()
        self.injury_risk_engine = InjuryRiskEngine()
        self.retirement_engine = RetirementEngine(seed=self.seed)
        self.randomness = RandomnessEngine(self.rng)

        # ------------------------------------
        # Injected entities
        # ------------------------------------
        self.player: Player | None = None
        self.team: Team | None = None

        # ------------------------------------
        # Personality (AI-side)
        # ------------------------------------
        factory = PersonalityFactory(self.rng)
        self.personality = factory.generate(
            archetypes=[
                PersonalityArchetype.LOYALIST,
                PersonalityArchetype.MONEY_HUNGRY,
            ]
        )
        self.behavior = PersonalityBehavior(self.personality, self.rng)

        # ------------------------------------
        # State
        # ------------------------------------
        self.morale: MoraleState = self.morale_engine.create_state()
        self.career_arc = self.career_arc_engine.create_state()
        self.injury_risk = self.injury_risk_engine.create_state()

        # ------------------------------------
        # Contract state
        # ------------------------------------
        self.contract_years_left: int = 0
        self.contract_aav: float = 0.0
        self.contract_clause: str = "none"
        self.last_contract_result: dict | None = None

        self.agent = AgentProfile(
            agent_id=f"AGENT_{self.seed}",
            name="Default Agent",
            aggression=0.55,
            loyalty_to_player=0.75,
            league_influence=0.50,
            media_leak_tendency=0.25,
            risk_tolerance=0.55,
        )

        
    # --------------------------------------------------
    # Injection
    # --------------------------------------------------

    def set_player(self, player: Player) -> None:
        self.player = player

    def set_team(self, team: Team) -> None:
        self.team = team
        # Guard: team.add_player may expect not-None
        if self.player is not None:
            team.add_player(self.player)

    # --------------------------------------------------
    # League helpers
    # --------------------------------------------------

    def _advance_league_and_cache(self) -> dict:

       
        """
        Advances the league by one season and caches outputs.
        Always safe: if team.snapshot() doesn't exist, we fall back to minimal snapshot.
        """
        team_snapshots: list[dict] = []

        if self.team is not None:
            try:
                team_snapshots.append(self.team.snapshot())  # recommended API
            except Exception:
                team_snapshots.append(
                    {
                        "team_id": str(getattr(self.team, "id", getattr(self.team, "team_id", "TEAM"))),
                        "name": f"{getattr(self.team, 'city', '')} {getattr(self.team, 'name', '')}".strip(),
                        "market_size": str(getattr(self.team, "market_size", "medium")),
                        "financial_health": float(getattr(self.team, "financial_health", 0.7)),
                        "stability": float(getattr(getattr(self.team, "state", None), "stability", 0.6)),
                        "competitive_score": float(getattr(getattr(self.team, "state", None), "competitive_score", 0.5)),
                    }
                )

        result = self.league.advance_season(
            team_snapshots=team_snapshots,
            team_count=int(getattr(self.league, "max_teams", 32) or 32),
        )

        self.last_league_context = result.get("league_context") or {}
        self.last_league_forecast = result.get("forecast") or {}
        self.last_league_shocks = result.get("shocks") or []

        # ------------------------------------
# Build season-level stat engines from league context
# ------------------------------------
        econ = (self.last_league_context.get("economics") or {})
        era = ((self.last_league_context.get("era") or {}).get("state") or {}).get(
            "active_era", "modern_offense"
        )

        self.league_stats = LeagueStats(
            seed=self.seed + self.year,
            season=2025 + self.year,
            era=str(era),
            teams=int(getattr(self.league, "max_teams", 32) or 32),
            salary_cap=int(econ.get("salary_cap", 92_000_000)),
        )
        self.league_stats.generate()

        self.player_stats_engine = PlayerStatsEngine(
            league=self.league_stats,
            seed=self.seed + (self.year * 17),
        )


        return result
    
    def _league_nudges(self) -> dict:
        """
        Safe modifiers from league context. Defaults are neutral.
        """
        ctx = self.last_league_context or {}
        nudges = ctx.get("nudges") or {}
        return {
            "chaos_mod": float(nudges.get("chaos_mod", 1.0)),
            "injury_rate_mod": float(nudges.get("injury_rate_mod", 1.0)),
            "morale_volatility_mod": float(nudges.get("morale_volatility_mod", 1.0)),
            "cap_growth_mod": float(nudges.get("cap_growth_mod", 1.0)),
        }

    def get_league_context_snapshot(self) -> dict:
        """
        Public helper for run_sim.py or UI debugging.
        """
        return {
            "league_context": self.last_league_context or {},
            "forecast": self.last_league_forecast or {},
            "shocks": self.last_league_shocks or [],
        }

    # --------------------------------------------------
    # Retirement adapter
    # --------------------------------------------------

    def _build_retirement_player(self):
        class PlayerProxy:
            pass

        if self.player is None:
            raise RuntimeError("SimEngine._build_retirement_player() called without player set.")

        p = PlayerProxy()
        p.age = self.player.age
        p.personality = self.personality
        p.morale = self.morale.overall()
        p.injury_wear = self.player.health.wear_and_tear
        p.career_injury_score = self.player.health.wear_and_tear
        p.chronic_injuries = len(self.player.health.chronic_flags)
        p.durability = 1.0 - self.player.health.wear_and_tear
        p.life_pressure = asdict(self.player.life_pressure)
        p.ovr = self.player.ovr()
        return p

    # --------------------------------------------------
    # Career stage (purely descriptive)
    # --------------------------------------------------

    def _derive_career_stage(self) -> str:
        if self.player is None:
            return "Unknown"

        age = int(self.player.age)
        if age < 22:
            return "Prospect / Development"
        if age < 26:
            return "Young NHL Regular"
        if age < 30:
            return "Prime Years"
        if age < 34:
            return "Late Prime / Early Decline"
        if age < 38:
            return "Veteran"
        if age < 42:
            return "Aging Veteran"
        return "Fringe / Retirement Risk"

    # --------------------------------------------------
    # Contract adapters
    # --------------------------------------------------

    def _build_contract_player_profile(self, ctx: BehaviorContext) -> PlayerProfile:
        if self.player is None or self.team is None:
            raise RuntimeError("Contract profile build requires player + team.")

        traits = self.player.traits
        life = self.player.life_pressure

        pers = PlayerPersonality(
            loyalty=float(getattr(traits, "loyalty", 0.5)),
            ambition=float(getattr(traits, "ambition", 0.5)),
            money_focus=float(getattr(traits, "money_focus", 0.5)),
            family_priority=float(getattr(traits, "family_priority", 0.5)),
            legacy_drive=float(getattr(traits, "legacy_drive", 0.5)),
            ego=float(getattr(traits, "ego", 0.5)),
            patience=float(getattr(traits, "patience", 0.5)),
            risk_tolerance=float(getattr(traits, "risk_tolerance", 0.5)),
            stability_need=float(getattr(traits, "stability_need", 0.5)),
            market_comfort=float(getattr(traits, "market_comfort", 0.5)),
            media_comfort=float(getattr(traits, "media_comfort", 0.5)),
            # optional
            work_ethic=float(getattr(traits, "work_ethic", 0.5)),
            mental_toughness=float(getattr(traits, "mental_toughness", 0.5)),
            volatility=float(getattr(traits, "volatility", 0.5)),
        )

        career = PlayerCareerState(
            age=int(self.player.age),
            ovr=float(self.player.ovr()),
            position=str(self.player.position.value),
            shoots=str(self.player.shoots.value),

            wear_and_tear=float(self.player.health.wear_and_tear),
            chronic_injuries=int(len(self.player.health.chronic_flags)),

            ice_time_satisfaction=float(getattr(ctx, "ice_time_satisfaction", 0.5)),
            role_mismatch=float(getattr(ctx, "role_mismatch", 0.0)),

            legacy_pressure=float(getattr(life, "legacy_pressure", 0.0)),
            identity_instability=float(getattr(life, "identity_instability", 0.0)),
            emotional_fatigue=float(getattr(life, "emotional_fatigue", 0.0)),
            security_anxiety=float(getattr(life, "security_anxiety", 0.0)),

            ufa_pressure=float(getattr(ctx, "ufa_pressure", 0.0)),
            offer_respect=float(getattr(ctx, "offer_respect", 0.5)),

            last_contract_aav=float(self.contract_aav),
            last_contract_term=int(self.contract_years_left),
        )

        mem = PlayerMemory(
            drafted_by_team_id=str(getattr(self.player, "drafted_by_team_id", None) or getattr(self.team, "id", None) or ""),
            developed_by_team_id=str(getattr(self.player, "developed_by_team_id", None) or getattr(self.team, "id", None) or ""),
        )

        return PlayerProfile(
            player_id=str(getattr(self.player, "id", self.player.name)),
            name=str(self.player.name),
            current_team_id=str(getattr(self.team, "id", getattr(self.team, "team_id", "TEAM"))),
            personality=pers,
            career=career,
            memory=mem,
        )

    def _build_contract_team_profile(self, win_pct: float) -> TeamProfile:
        if self.team is None:
            raise RuntimeError("Contract team profile build requires team.")

        team_id = str(getattr(self.team, "id", getattr(self.team, "team_id", f"{self.team.city}_{self.team.name}")))
        team_name = f"{self.team.city} {self.team.name}".strip()

        market_pressure = float(getattr(getattr(self.team, "market", None), "pressure", 0.5))
        if hasattr(self.team, "market_pressure"):
            market_pressure = float(getattr(self.team, "market_pressure", 0.5))

        stability = float(getattr(getattr(self.team, "state", None), "stability", 0.5))
        competitive = float(getattr(getattr(self.team, "state", None), "competitive_score", win_pct))
        org_pressure = float(getattr(getattr(self.team, "state", None), "org_pressure", 0.5))

        ownership_meddling = float(getattr(getattr(self.team, "ownership", None), "meddling", 0.5))
        ownership_budget = float(getattr(getattr(self.team, "ownership", None), "budget_willingness", 0.55))

        # League cap as truth (if available)
        league_ctx = self.last_league_context or self.league.get_league_context()
        econ = (league_ctx.get("economics") or {})
        league_cap = float(econ.get("salary_cap", 88_000_000.0))
        league_growth = float(econ.get("cap_growth_rate", 0.05))

        cap_total = float(getattr(self.team, "cap_total", league_cap))
        cap_space = float(getattr(self.team, "cap_space", 10_000_000.0))
        cap_growth = float(getattr(self.team, "cap_projection_growth", league_growth))

        star_count = int(getattr(self.team, "star_count", 0))
        core_count = int(getattr(self.team, "core_count", 5))
        depth_quality = float(getattr(self.team, "depth_quality", 0.5))

        return TeamProfile(
            team_id=team_id,
            name=team_name,
            archetype=str(getattr(self.team, "archetype", "normal")),
            status=str(getattr(self.team, "status", "bubble")),

            market=MarketProfile(
                market_size=str(getattr(self.team, "market_size", "medium")),
                media_pressure=float(clamp(market_pressure, 0.0, 1.0)),
                fan_expectations=float(clamp(market_pressure, 0.0, 1.0)),
                tax_advantage=float(getattr(self.team, "tax_advantage", 0.5)),
                weather_quality=float(getattr(self.team, "weather_quality", 0.5)),
                travel_burden=float(getattr(self.team, "travel_burden", 0.5)),
            ),
            ownership=OwnershipProfile(
                patience=float(getattr(self.team, "ownership_patience", 0.5)),
                ambition=float(getattr(self.team, "ownership_ambition", 0.5)),
                budget_willingness=float(clamp(ownership_budget, 0.0, 1.0)),
                meddling=float(clamp(ownership_meddling, 0.0, 1.0)),
            ),
            reputation=ReputationProfile(
                league_reputation=float(getattr(self.team, "league_reputation", 0.5)),
                player_reputation=float(getattr(self.team, "player_reputation", 0.5)),
                management_reputation=float(getattr(self.team, "management_reputation", 0.5)),
                development_reputation=float(getattr(self.team, "development_reputation", 0.5)),
            ),
            philosophy=OrgPhilosophy(
                development_quality=float(getattr(self.team, "development_quality", 0.5)),
                prospect_patience=float(getattr(self.team, "prospect_patience", 0.5)),
                risk_tolerance=float(getattr(self.team, "risk_tolerance", 0.5)),
            ),
            state=TeamDynamicState(
                competitive_score=float(clamp(competitive, 0.0, 1.0)),
                team_morale=float(getattr(getattr(self.team, "state", None), "team_morale", 0.5)),
                org_pressure=float(clamp(org_pressure, 0.0, 1.0)),
                stability=float(clamp(stability, 0.0, 1.0)),
                ownership_stability=float(getattr(self.team, "ownership_stability", 0.7)),
                arena_security=float(getattr(self.team, "arena_security", 0.8)),
                financial_health=float(getattr(self.team, "financial_health", 0.7)),
            ),
            roster=TeamRosterProxy(
                star_count=star_count,
                core_count=core_count,
                depth_quality=float(clamp(depth_quality, 0.0, 1.0)),
            ),
            cap_total=cap_total,
            cap_space=cap_space,
            cap_projection_growth=cap_growth,
        )

    def _estimate_expected_aav(self) -> float:
        if self.player is None:
            return 800_000.0

        if (
            self.league_stats is not None
            and hasattr(self.player, "season_stats")
            and self.player.season_stats
        ):
            latest = max(
                self.player.season_stats.values(),
                key=lambda x: x.get("season", 0),
            )

            if latest.get("type") == "goalie":
                gsax = float(latest.get("gsax", 0.0))
                pct = self.league_stats.value_to_percentile("goalie", "gsax", gsax)
            else:
                war = float(latest.get("war", 0.0))
                pct = self.league_stats.value_to_percentile("skater", "war", war)

            aav = 900_000.0 + (pct / 100.0) ** 1.35 * 12_500_000.0
            return float(clamp(aav, 800_000.0, 14_500_000.0))

        # fallback if no stats yet
        ovr = float(self.player.ovr())
        return 800_000.0 + (ovr ** 1.7) * 11_200_000.0



           


    def _maybe_run_offseason_contracts(self, ctx: BehaviorContext, win_pct: float) -> None:
        """
        Runs if:
          - contract expired OR
          - player entering UFA window (ufa_pressure high)
        """
        if self.retired or self.player is None or self.team is None:
            return

        if self.contract_years_left > 0:
            self.contract_years_left -= 1

        if self.contract_years_left > 0 and float(getattr(ctx, "ufa_pressure", 0.0)) < 0.85:
            return

        team_profile = self._build_contract_team_profile(win_pct=win_pct)
        player_profile = self._build_contract_player_profile(ctx=ctx)

        context_kind = (
            ContractContextKind.RESIGN
            if player_profile.current_team_id == team_profile.team_id
            else ContractContextKind.UFA
        )

        league_ctx = self.last_league_context or self.league.get_league_context()
        econ = league_ctx.get("economics") or {}
        health = league_ctx.get("health") or {}
        era = league_ctx.get("era") or {}
        nudges = self._league_nudges()

        cap = float(econ.get("salary_cap", team_profile.cap_total))
        cap_growth = float(econ.get("cap_growth_rate", team_profile.cap_projection_growth)) * nudges["cap_growth_mod"]

        league = {
            "cap": cap,
            "cap_growth": cap_growth,
            "expected_aav": self._estimate_expected_aav(),
            "league_health": float(health.get("health_score", 0.6)),
            "era": (era.get("state") or {}).get("active_era", "unknown"),
            "coach_security": (
            self.coach_impact.job_security if self.coach_impact else 0.5
        ),

        }

        result = negotiate_contract(
            rng=self.rng,
            player=player_profile,
            team=team_profile,
            agent=self.agent,
            league=league,
            signing_year=2025 + self.year,
            context_kind=context_kind,
            max_weeks=14,
        )

        self.last_contract_result = {
            "outcome": result.outcome.value,
            "notes": result.notes,
            "contract": result.contract.to_dict() if result.contract else None,
        }

        # Update engine-visible contract state (used by future decisions)
        if result.contract:
            self.contract_years_left = int(result.contract.term_years)
            self.contract_aav = float(result.contract.salary_aav)
            self.contract_clause = str(result.contract.clauses.clause_type.value)
        else:
            # unsigned: player might be "in limbo"
            # keep years_left at 0 to force re-check next offseason
            self.contract_years_left = 0
            self.contract_aav = 0.0
            self.contract_clause = "none"

    def get_last_contract_snapshot(self) -> dict:
        return self.last_contract_result or {}

    # --------------------------------------------------
    # Debug dump (stable, verbose)
    # --------------------------------------------------

    def _debug_dump_year(self, *, ctx: BehaviorContext, decision: Any, team_ctx: dict, win_pct: float) -> None:
        """
        Heavy debug dump. Safe even if league context missing.
        """
        if self.player is None or self.team is None:
            return

        print("\n================ PLAYER STATE DUMP ================")

        print("\n[IDENTITY]")
        print(f"Name        : {self.player.name}")
        print(f"Age         : {self.player.age}")
        print(f"Position    : {self.player.position.value}")
        print(f"Shoots      : {self.player.shoots.value}")
        print(f"Team        : {self.team.city} {self.team.name}")
        print(f"Team Style  : {getattr(self.team, 'archetype', 'unknown')}")
        print("\n[COACH]")
        print(f"Name                : {self.coach.name}")
        print(f"Style               : {self.coach.style.value}")
        print(f"Job Security        : {self.coach.job_security:.3f}")


        print("\n[OVR / ATTRIBUTES]")
        print(f"OVR         : {self.player.ovr():.3f}")

        # --------------------------------------------------
        # SEASON STATS (LATEST)
        # --------------------------------------------------
        if self.coach_impact:
            print("\n[COACH IMPACT]")
            for k, v in vars(self.coach_impact).items():
                if isinstance(v, float):
                    print(f"{k:22s}: {v:.3f}")

        if hasattr(self.player, "season_stats") and self.player.season_stats:
            latest = self.player.season_stats.get(2025 + self.year)
            if latest:
                print("\n[SEASON STATS]")
                for k, v in latest.items():
                    if isinstance(v, float):
                        print(f"{k:22s}: {v:.3f}")
                    else:
                        print(f"{k:22s}: {v}")

        try:
            for g, v in self.player.group_averages().items():
                print(f"{g:22s}: {v:.3f}")
        except Exception:
            pass

        print("\n[PERSONALITY TRAITS]")
        for k, v in vars(self.player.traits).items():
            if isinstance(v, (int, float)):
                print(f"{k:22s}: {float(v):.3f}")
            else:
                print(f"{k:22s}: {v}")

        print("\n[LIFE PRESSURE]")
        for k, v in vars(self.player.life_pressure).items():
            if isinstance(v, (int, float)):
                print(f"{k:22s}: {float(v):.3f}")
            else:
                print(f"{k:22s}: {v}")

        print("\n[HEALTH]")
        print(f"Wear & Tear         : {self.player.health.wear_and_tear:.3f}")
        print(f"Chronic Injuries    : {len(self.player.health.chronic_flags)}")

        print("\n[MORALE]")
        print(f"Overall Morale      : {self.morale.overall():.3f}")
        for axis, val in self.morale.axes.items():
            print(f"{axis:22s}: {val:.3f}")

        print("\n[CAREER ARC]")
        for k, v in vars(self.career_arc).items():
            if isinstance(v, float):
                print(f"{k:22s}: {v:.3f}")

        print("\n[INJURY RISK]")
        print(f"Total Risk          : {self.injury_risk.total_risk:.3f}")

        print("\n[TEAM CONTEXT]")
        print(f"Season Win %        : {win_pct:.3f}")
        for k, v in team_ctx.items():
            if isinstance(v, float):
                print(f"{k:22s}: {v:.3f}")
            else:
                print(f"{k:22s}: {v}")

        print("\n[BEHAVIOR CONTEXT]")
        for k, v in asdict(ctx).items():
            print(f"{k:22s}: {v:.3f}")

        lc = self.last_league_context or {}
        econ = lc.get("economics") or {}
        health = lc.get("health") or {}
        era = lc.get("era") or {}
        parity = lc.get("parity") or {}
        narrative = lc.get("narratives") or {}
        nudges = lc.get("nudges") or {}

        print("\n[LEAGUE SNAPSHOT]")
        print(f"Season              : {lc.get('season', 'unknown')}")
        print(f"League Health       : {float(health.get('health_score', 0.0)):.3f}")
        print(f"Salary Cap          : {float(econ.get('salary_cap', 0.0)):,.0f}")
        print(f"Cap Growth Rate     : {float(econ.get('cap_growth_rate', 0.0)):.3f}")
        print(f"Parity Index        : {float(parity.get('parity_index', 0.0)):.3f}")
        print(f"Chaos Index         : {float((self.last_league_forecast or {}).get('chaos_index', 0.0)):.3f}")
        print(f"Active Era          : {(era.get('state') or {}).get('active_era', 'unknown')}")

        top_narr = (narrative.get("top_narratives") or [])
        if top_narr:
            print("Top Narratives      :")
            for n in top_narr[:4]:
                print(f"  - {n}")

        if self.last_league_shocks:
            print("Active Shocks       :")
            for s in self.last_league_shocks[:3]:
                name = s.get("name", "shock")
                sev = float(s.get("severity", 0.0))
                dur = s.get("duration_years", "?")
                print(f"  - {name} | severity={sev:.2f} | duration={dur}")

        if nudges:
            print("League Nudges       :")
            for k in ["chaos_mod", "injury_rate_mod", "morale_volatility_mod", "cap_growth_mod"]:
                if k in nudges:
                    print(f"  {k:20s}: {float(nudges[k]):.3f}")

        print("\n[CONTRACT STATUS]")
        print(f"Years Left          : {self.contract_years_left}")
        print(f"AAV                 : {self.contract_aav:,.0f}" if self.contract_aav > 0 else "AAV                 : (none)")
        print(f"Clause              : {self.contract_clause}")

        if self.last_contract_result:
            print("\n[LAST CONTRACT RESULT]")
            print(f"Outcome             : {self.last_contract_result.get('outcome')}")
            notes = self.last_contract_result.get("notes") or []
            if notes:
                print("Notes               :")
                for n in notes[:8]:
                    print(f"  - {n}")

        print("\n[RETIREMENT]")
        print(f"Retire Chance       : {float(getattr(decision, 'retire_chance', 0.0)):.6f}")
        print(f"Net Score           : {float(getattr(decision, 'net_score', 0.0)):.3f}")
        print(f"Primary Reason      : {getattr(decision, 'primary_reason', 'unknown')}")

        print("\n[CAREER STAGE]")
        print(self._derive_career_stage())
        print("===================================================")


         

   

    # --------------------------------------------------
    # Single season
    # ------------------------------------------------__
    def sim_year(self, *, debug_dump: bool = True) -> None:
        """
        Simulates one year of the career.
        """
        if self.retired or self.player is None or self.team is None:
            return

        self.year += 1

        print("\n==============================")
        print(f"      SIM YEAR {self.year}")
        print("==============================")

        # --------------------------------------------------
        # 0. League macro
        # --------------------------------------------------
        self._advance_league_and_cache()
        league_nudges = self._league_nudges()

        # --------------------------------------------------
        # 1. Team season result (abstract)
        # --------------------------------------------------
        expected = float(self.team._expected_win_pct())

        # Chaos influenced by stability + league macro chaos
        chaos = (1.0 - float(self.team.state.stability)) * self.rng.uniform(-0.10, 0.10) * float(league_nudges["chaos_mod"])
        luck = self.rng.uniform(-0.06, 0.06)

        win_pct = max(0.25, min(0.75, expected + chaos + luck))

        # --------------------------------------------------
        # 2. Update team state
        # --------------------------------------------------
        self.team.update_team_state(win_pct=win_pct)
        team_ctx = self.team.team_context_for_player(self.player)
        rebuild_mode = team_ctx.get("rebuild_mode", 0.0)

        # --------------------------------------------------
# Coach evaluation (seasonal)
# --------------------------------------------------
        self.coach_impact = self.coach.evaluate_season(
            team=self.team,
            win_pct=win_pct,
            roster_context=team_ctx,
            league_context=self.last_league_context or {},
        )


        # --------------------------------------------------
# 2.5 Player season stats (real distribution + feedback)
# --------------------------------------------------
        season_stats = {}
        if self.player_stats_engine is not None:
            season_stats = self.player_stats_engine.simulate_player_season(
                player=self.player,
                season=2025 + self.year,
            )
        # Defensive persistence (even if engine changes later)
        if season_stats and hasattr(self.player, "season_stats"):
            self.player.season_stats[season_stats["season"]] = season_stats


        # --------------------------------------------------
        # 3. Build behavior context
        # --------------------------------------------------
        injury_burden = float(self.injury_risk.total_risk) * float(league_nudges["injury_rate_mod"])
        league_morale_pressure = max(
            0.0,
            float(league_nudges["morale_volatility_mod"]) - 1.0
            + (self.coach_impact.morale_volatility if self.coach_impact else 0.0)
        )


        ctx = BehaviorContext(
            team_success=win_pct,
            losing_streak=max(0.0, 0.5 - win_pct),
            rebuild_mode=float(rebuild_mode),
            role_mismatch=float(team_ctx.get("role_mismatch", 0.0)),
            ice_time_satisfaction=clamp(
                (
                    0.35
                    + float(season_stats.get("performance_score", 0.50)) * 0.65
                    + (self.coach_impact.ice_time_bias if self.coach_impact else 0.0)
                ),
                0.15,
                0.98,
            ),


            scratched_recently=0.0,
            offer_respect=float(team_ctx.get("stability", 0.5)),
            ufa_pressure=min(1.0, self.year / 7.0),
            market_heat=float(team_ctx.get("market_pressure", 0.5)),
            injury_burden=injury_burden,
            family_event=(0.15 if self.year in (4, 7, 12, 18, 25) else 0.0) + (0.05 * league_morale_pressure),
            age_factor=min(1.0, float(self.player.age) / 35.0),
            cup_satisfaction=0.0,
        )

        # Apply personality-based context noise
        ctx = BehaviorContext(
            **self.randomness.apply_context_noise(asdict(ctx), self.personality)
        )

        # --------------------------------------------------
        # 4. AI & psychology
        # --------------------------------------------------
        self.ai_manager.evaluate_player(
            behavior=self.behavior,
            ctx=ctx,
        )

        self.morale_engine.update(
            self.morale,
            personality=self.personality,
            behavior=self.behavior,
            ctx=ctx,
        )

        self.career_arc_engine.update(
            self.career_arc,
            personality=self.personality,
            morale_axes=self.morale.axes,
        )

        self.injury_risk_engine.update(
            self.injury_risk,
            personality=self.personality,
            morale_axes=self.morale.axes,
            career=self.career_arc,
        )

        # --------------------------------------------------
        # 5. Aging / development
        # --------------------------------------------------
        self.player.advance_year(
    season_morale=self.morale.overall(),
    season_injury_risk=self.injury_risk.total_risk,
    team_instability=1.0 - float(team_ctx.get("stability", 0.5)),
    development_modifier=(
        self.coach_impact.development_boost if self.coach_impact else 0.0
    ),
)


        # --------------------------------------------------
        # 6. Offseason contracts
        # --------------------------------------------------
        self._maybe_run_offseason_contracts(ctx=ctx, win_pct=win_pct)

        # --------------------------------------------------
        # 7. Retirement
        # --------------------------------------------------
        decision = self.retirement_engine.evaluate_player(
            self._build_retirement_player(), {}
        )

        # --------------------------------------------------
        # 8. Debug output
        # --------------------------------------------------
        if debug_dump:
            self._debug_dump_year(
                ctx=ctx,
                decision=decision,
                team_ctx=team_ctx,
                win_pct=win_pct,
            )

        if bool(getattr(decision, "retired", False)):
            self.retired = True
            self.player.retired = True
            self.player.retirement_reason = getattr(decision, "primary_reason", "unknown")
            print("\nPLAYER HAS RETIRED")

    # --------------------------------------------------
    # Multi-year
    # --------------------------------------------------

    def sim_years(self, years: int = 40, *, debug_dump: bool = True, sleep_s: float = 0.02) -> None:
        for _ in range(int(years)):
            if self.retired:
                break
            self.sim_year(debug_dump=debug_dump)
            if sleep_s and sleep_s > 0:
                time.sleep(float(sleep_s))