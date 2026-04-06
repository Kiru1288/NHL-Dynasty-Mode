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

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, List, Tuple, Callable, Set, Sequence
import math
import random
import re
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

from app.sim_engine.draft.draft_lottery import (
    LotteryTeam,
    LotteryResult,
    run_draft_lottery,
)

from app.sim_engine.draft.draft_board import (
    DraftBoard,
    DraftContext,
    DraftEvent,
    TeamProfile as DraftTeamProfile,
)

from app.sim_engine.progression import run_player_progression
from app.sim_engine.progression.development import (
    PHASE_DECLINING,
    PHASE_EMERGING,
    PHASE_PRIME,
    PHASE_PROSPECT,
    PHASE_VETERAN,
    assign_career_phase_from_age,
    career_phase_for_age,
    set_player_trend,
    tick_career_trend,
)

try:
    from app.sim_engine.tuning import probability_tables as _tuning_probability_tables
except Exception:  # pragma: no cover
    _tuning_probability_tables = None



# -------------------------------
# Entities
# -------------------------------
from app.sim_engine.entities.player import (
    Player,
    Position,
    Shoots,
    IdentityBio,
    BackstoryUpbringing,
    BackstoryType,
    UpbringingType,
    SupportLevel,
    PressureLevel,
    DevResources,
    ATTRIBUTE_KEYS,
    OFFENSE_KEYS,
    PASSING_KEYS,
    DEFENSE_KEYS,
    IQ_KEYS,
    PHYS_KEYS,
    SKATING_KEYS,
    clamp_rating,
    compute_ovr,
)
from app.sim_engine.entities.team import Team, TeamArchetype
from app.sim_engine.entities.league import League
from app.sim_engine.entities.coach import Coach, CoachRole, generate_coach
from app.sim_engine.generation.name_generator import generate_human_identity
from app.sim_engine.entities.prospect import (
    Prospect,
    ProspectPhase,
    ScoutProfile,
    Position as ProspectPosition,
    Shoots as ProspectShoots,
    DevelopmentSystem,
)
from app.sim_engine.league import (
    GameSlot,
    StandingsTable,
    PlayoffResult,
    generate_regular_season_schedule,
    simulate_playoffs,
    compute_awards,
)

try:
    from app.sim_engine.world import momentum as world_momentum
    from app.sim_engine.world import fatigue as world_fatigue
    from app.sim_engine.world import morale as world_morale
    from app.sim_engine.world import chemistry as world_chemistry
    from app.sim_engine.world import injuries as world_injuries
    from app.sim_engine.world import durability as world_durability
    from app.sim_engine.world import calendar as world_calendar
except Exception:  # pragma: no cover
    world_momentum = None  # type: ignore
    world_fatigue = None  # type: ignore
    world_morale = None  # type: ignore
    world_chemistry = None  # type: ignore
    world_injuries = None  # type: ignore
    world_durability = None  # type: ignore
    world_calendar = None  # type: ignore
# -------------------------------
# Economy Systems
# -------------------------------
from app.sim_engine.economy.waiver_ai import (
    WaiverEngine,
    WaiverConfig,
    update_priority_after_claim,
)


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

# -------------------------------
# Scouting System
# -------------------------------
from app.sim_engine.draft.scouting import (
    create_scout,
    create_scouting_department,
    update_scouting,
    build_team_draft_board,
    LeagueContextSnapshot,
    Region,
    ScoutRole,
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
# LEAGUE SEASON RESULTS (STRUCTURAL SPINE)
# =====================================================================


@dataclass
class LeagueSeasonResult:
    year: int
    schedule: List[GameSlot]
    standings: StandingsTable
    playoff_result: Optional[PlayoffResult]
    awards: Dict[str, Any]


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
# PLAYER STORYLINE ENGINE (character + lifecycle + performance-driven)
# =====================================================================

_PLAYER_STORYLINE_CATALOG_CACHE: Optional[List[Dict[str, Any]]] = None


def _get_player_storyline_catalog() -> List[Dict[str, Any]]:
    global _PLAYER_STORYLINE_CATALOG_CACHE
    if _PLAYER_STORYLINE_CATALOG_CACHE is None:
        _PLAYER_STORYLINE_CATALOG_CACHE = _build_player_storyline_catalog()
    elif (
        _PLAYER_STORYLINE_CATALOG_CACHE
        and isinstance(_PLAYER_STORYLINE_CATALOG_CACHE[0], dict)
        and "tier" not in _PLAYER_STORYLINE_CATALOG_CACHE[0]
    ):
        _PLAYER_STORYLINE_CATALOG_CACHE = _build_player_storyline_catalog()
    return _PLAYER_STORYLINE_CATALOG_CACHE


# --- Explicit character (20–90) + storyline polarity (engine-only; no new files) ---

STORYLINE_POLARITY_POSITIVE_KEYWORDS = (
    "breakout",
    "surge",
    "leader",
    "mentor",
    "rallies",
    "resilience",
    "charity",
    "extension celebration",
    "endorsement",
    "clutch reputation",
    "rebound narrative",
)
STORYLINE_POLARITY_NEGATIVE_KEYWORDS = (
    "arrest",
    "scandal",
    "fight",
    "holdout",
    "suspension",
    "collapse",
    "benching",
    "demand",
    "bust",
    "gambling",
    "dui",
    "violence",
    "leak",
)


def generate_player_character_score(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.10:
        return rng.randint(75, 90)
    if roll < 0.35:
        return rng.randint(60, 75)
    if roll < 0.70:
        return rng.randint(45, 60)
    if roll < 0.90:
        return rng.randint(30, 45)
    return rng.randint(20, 30)


def assign_player_personality_tag_from_character(player: Any) -> str:
    ch = int(getattr(player, "character", 50) or 50)
    ch = max(20, min(90, ch))
    if ch >= 75:
        tag = "leader"
    elif ch >= 60:
        tag = "professional"
    elif ch >= 45:
        tag = "neutral"
    elif ch >= 30:
        tag = "volatile"
    else:
        tag = "toxic"
    setattr(player, "personality", tag)
    return tag


def ensure_player_character_initialized(player: Any, rng: random.Random) -> int:
    raw = getattr(player, "character", None)
    need = raw is None
    if not need and isinstance(raw, (int, float)):
        ri = int(raw)
        need = ri < 20 or ri > 90
    if need:
        setattr(player, "character", generate_player_character_score(rng))
    assign_player_personality_tag_from_character(player)
    return int(getattr(player, "character", 50))


def initialize_league_player_characters(league: Any, rng: random.Random) -> int:
    n = 0
    for team in getattr(league, "teams", None) or []:
        for p in getattr(team, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            ensure_player_character_initialized(p, rng)
            assign_development_profile(p, rng)
            n += 1
    return n


# --- Cap / contract economy (shared with run_sim universe; dollars vs millions normalized) ---


def _economy_player_cap_hit_millions(player: Any) -> float:
    for k in ("cap_hit_m", "contract_aav_m", "aav_m"):
        v = getattr(player, k, None)
        if v is not None:
            try:
                x = float(v)
                if x > 0:
                    return x
            except (TypeError, ValueError):
                pass
    c = getattr(player, "contract", None)
    if c is not None:
        for k in ("cap_hit_m", "aav_m", "aav"):
            v = getattr(c, k, None)
            if v is not None:
                try:
                    x = float(v)
                    if x > 0:
                        return x if x < 500.0 else x / 1_000_000.0
                except (TypeError, ValueError):
                    pass
        sa = getattr(c, "salary_aav", None)
        if sa is not None:
            try:
                sd = float(sa)
                if sd > 0:
                    return sd / 1_000_000.0 if sd > 500_000.0 else sd
            except (TypeError, ValueError):
                pass
    try:
        ovr = float(player.ovr()) if callable(getattr(player, "ovr", None)) else float(getattr(player, "ovr", 0.5))
    except Exception:
        ovr = 0.5
    return max(0.75, 1.0 + 9.0 * max(0.0, ovr - 0.50))


def _team_roster_players(team: Any) -> List[Any]:
    r = getattr(team, "roster", None) or getattr(team, "players", None) or []
    return list(r)


def _team_payroll_millions(team: Any) -> float:
    total = 0.0
    for p in _team_roster_players(team):
        if getattr(p, "retired", False):
            continue
        total += _economy_player_cap_hit_millions(p)
    return float(total)


def _resolve_salary_cap_millions(team: Any, salary_cap_m: Optional[float] = None) -> float:
    if salary_cap_m is not None and float(salary_cap_m) > 0:
        return float(salary_cap_m)
    v = float(getattr(team, "salary_cap_m", 0) or 0)
    if v > 0:
        return v
    raw = float(getattr(team, "salary_cap", 0) or getattr(team, "cap_total", 0) or 0)
    if raw <= 0:
        return 0.0
    if raw > 250.0:
        return raw / 1_000_000.0
    return raw


def _resolve_total_salary_millions(team: Any, total_salary_m: Optional[float] = None) -> float:
    if total_salary_m is not None:
        return float(total_salary_m)
    ts = getattr(team, "total_salary", None)
    if ts is not None:
        try:
            tsv = float(ts)
            if tsv > 0:
                return tsv / 1_000_000.0 if tsv > 250.0 else tsv
        except (TypeError, ValueError):
            pass
    return _team_payroll_millions(team)


def cap_tier_from_usage_ratio(ratio: float) -> str:
    """Tier from cap usage ratio (0..1+). Includes cap_hell above 100%."""
    u = float(ratio)
    if u > 1.0:
        return "cap_hell"
    if u >= 0.95:
        return "critical"
    if u >= 0.85:
        return "high"
    if u >= 0.70:
        return "moderate"
    return "low"


def _cap_pressure_scalar_from_ratio(ratio: float) -> float:
    r = float(ratio)
    if r < 0.75:
        return 0.2 + r * 0.3
    if r < 0.90:
        return 0.4 + (r - 0.75) * 1.2
    if r < 1.00:
        return 0.7 + (r - 0.90) * 2.5
    return 1.0 + (r - 1.00) * 3.0


def resolve_cap_target_ratio_for_identity(archetype: str, power_state: str, rng: random.Random) -> float:
    """
    Target payroll / cap for the season based on org identity + prior power state.
    Teams are nudged toward this each universe year (runner cap pass).
    """
    a = str(archetype or "balanced").lower()
    ps = str(power_state or "").lower()
    contender_like = ps in (
        "dynasty",
        "powerhouse",
        "repeat_contender",
        "rising_contender",
        "contender",
        "fragile_contender",
        "playoff_team",
    )
    if a == "win_now":
        t = float(rng.uniform(0.88, 1.00))
    elif a == "contender":
        t = float(rng.uniform(0.82, 0.96))
    elif a == "chaos_agent":
        t = float(rng.uniform(0.70, 1.05))
    elif a == "rebuild":
        t = float(rng.uniform(0.55, 0.75))
    elif a == "draft_and_develop":
        t = float(rng.uniform(0.65, 0.80))
    elif a == "balanced":
        t = float(rng.uniform(0.70, 0.85))
    else:
        t = float(rng.uniform(0.70, 0.85))
    if contender_like and a not in ("rebuild", "draft_and_develop"):
        c = float(rng.uniform(0.85, 0.95))
        t = max(t, c * 0.55 + t * 0.45)
    return float(max(0.48, min(1.08, t)))


def _apply_cap_hit_m_to_player(player: Any, millions: float) -> None:
    m = max(0.35, float(millions))
    setattr(player, "cap_hit_m", m)
    c = getattr(player, "contract", None)
    if c is None:
        return
    try:
        setattr(c, "cap_hit_m", m)
        if hasattr(c, "aav_m"):
            setattr(c, "aav_m", m)
        if hasattr(c, "aav"):
            setattr(c, "aav", m)
        if hasattr(c, "salary_aav"):
            setattr(c, "salary_aav", m * 1_000_000.0)
    except Exception:
        pass


def nudge_team_payroll_toward_cap_target(
    team: Any,
    salary_cap_m: float,
    target_ratio: float,
    rng: random.Random,
    *,
    archetype: str = "balanced",
) -> Dict[str, Any]:
    """
    Move roster cap hits toward target usage for the season (partial convergence).
    """
    cap_m = float(salary_cap_m)
    if cap_m <= 0:
        return {"applied": False, "factor": 1.0, "target_ratio": float(target_ratio)}
    roster = [p for p in _team_roster_players(team) if not getattr(p, "retired", False)]
    if not roster:
        return {"applied": False, "factor": 1.0, "target_ratio": float(target_ratio)}
    payroll = sum(_economy_player_cap_hit_millions(p) for p in roster)
    if payroll <= 1e-6:
        return {"applied": False, "factor": 1.0, "target_ratio": float(target_ratio)}
    target_pay = cap_m * float(target_ratio)
    raw_factor = target_pay / payroll
    blend = 0.42 + rng.random() * 0.28
    factor = 1.0 + (raw_factor - 1.0) * blend
    hi = 1.30
    a = str(archetype or "balanced").lower()
    if a == "chaos_agent":
        hi = 1.46
    elif a == "win_now":
        hi = 1.38
    elif a == "contender":
        hi = 1.32
    elif a in ("rebuild", "draft_and_develop"):
        hi = 1.18
    factor = max(0.55, min(hi, factor))
    for p in roster:
        cur = _economy_player_cap_hit_millions(p)
        _apply_cap_hit_m_to_player(p, cur * factor)
    return {"applied": True, "factor": float(factor), "target_ratio": float(target_ratio)}


def runner_team_roster_identity_signals(team: Any) -> Dict[str, Any]:
    """
    Roster age curve + simple prospect pipeline score for franchise identity evolution.
    pipeline_score ~0.15..0.95 (higher = more young talent / upside on roster).
    """
    roster = [p for p in _team_roster_players(team) if not getattr(p, "retired", False)]
    if not roster:
        return {
            "avg_age": 27.0,
            "frac_30p": 0.0,
            "frac_u24": 0.0,
            "pipeline_score": 0.45,
            "n": 0,
        }
    ages: List[int] = []
    n30 = 0
    u24_ovr: List[float] = []
    for p in roster:
        age = int(career_player_age(p))
        ages.append(age)
        if age >= 30:
            n30 += 1
        if age < 24:
            try:
                fn = getattr(p, "ovr", None)
                ov = float(fn()) if callable(fn) else float(getattr(p, "ovr", 0.5))
                if ov > 1.2:
                    ov /= 99.0
                u24_ovr.append(ov)
            except Exception:
                u24_ovr.append(0.52)
    n = len(ages)
    avg_age = float(sum(ages)) / float(n)
    frac_30p = float(n30) / float(n)
    frac_u24 = float(sum(1 for a in ages if a < 24)) / float(n)
    base_pipe = 0.40
    if u24_ovr:
        base_pipe = 0.32 + 0.62 * (sum(u24_ovr) / float(len(u24_ovr)))
    pipeline_score = clamp(float(base_pipe * (1.0 + 0.42 * frac_u24)), 0.12, 0.95)
    return {
        "avg_age": round(avg_age, 2),
        "frac_30p": round(frac_30p, 3),
        "frac_u24": round(frac_u24, 3),
        "pipeline_score": round(pipeline_score, 3),
        "n": n,
    }


def calculate_cap_pressure(
    team: Any,
    *,
    salary_cap_m: Optional[float] = None,
    total_salary_m: Optional[float] = None,
) -> float:
    cap = _resolve_salary_cap_millions(team, salary_cap_m)
    if cap <= 0:
        return 0.0
    tot_base = _resolve_total_salary_millions(team, total_salary_m)
    stress = bad_contract_payroll_stress_millions(team)
    ratio = (tot_base + stress) / cap
    return _cap_pressure_scalar_from_ratio(ratio)


def update_team_strategy(
    team: Any,
    *,
    pressure: Optional[float] = None,
    salary_cap_m: Optional[float] = None,
    total_salary_m: Optional[float] = None,
    forced_pressure_tier: Optional[str] = None,
) -> str:
    cap = _resolve_salary_cap_millions(team, salary_cap_m)
    tot_base = _resolve_total_salary_millions(team, total_salary_m)
    stress = bad_contract_payroll_stress_millions(team)
    raw_ratio = (tot_base / cap) if cap > 0 else 0.0
    eff_ratio = ((tot_base + stress) / cap) if cap > 0 else 0.0
    tier = cap_tier_from_usage_ratio(eff_ratio)
    if forced_pressure_tier:
        tier = str(forced_pressure_tier).strip().lower()
    p = float(pressure) if pressure is not None else _cap_pressure_scalar_from_ratio(eff_ratio)
    if tier == "cap_hell":
        s = "cap_emergency"
    elif tier == "critical":
        s = "panic_dump"
    elif tier == "high":
        s = "cap_squeeze"
    elif tier == "low":
        s = "spender"
    else:
        s = "balanced"
    trade_m = {
        "low": 0.90,
        "moderate": 1.0,
        "high": 1.24,
        "critical": 1.58,
        "cap_hell": 2.15,
    }.get(tier, 1.0)
    fa_m = {
        "low": 1.16,
        "moderate": 1.0,
        "high": 0.74,
        "critical": 0.36,
        "cap_hell": 0.10,
    }.get(tier, 1.0)
    setattr(team, "strategy", s)
    setattr(team, "cap_pressure", p)
    setattr(team, "cap_pressure_tier", tier)
    setattr(team, "cap_usage_ratio", float(raw_ratio))
    setattr(team, "cap_effective_usage_ratio", float(eff_ratio))
    setattr(team, "_runner_trade_pressure_mult", float(trade_m))
    setattr(team, "_runner_fa_budget_mult", float(fa_m))
    try:
        setattr(team, "_tuning_trade_aggression", float(trade_m))
    except Exception:
        pass
    return s


def _league_cap_growth_rate(league: Any) -> float:
    if league is None:
        return 0.05
    econ = getattr(league, "economics", None)
    if econ is not None:
        try:
            return float(getattr(econ, "cap_growth_rate", 0.05) or 0.05)
        except (TypeError, ValueError):
            pass
    for attr in ("cap_growth_rate", "cap_growth"):
        if hasattr(league, attr):
            try:
                return float(getattr(league, attr) or 0.05)
            except (TypeError, ValueError):
                pass
    if isinstance(league, dict):
        return float(league.get("cap_growth_rate", league.get("cap_growth", 0.05)) or 0.05)
    return 0.05


def _league_chaos_index_value(league: Any) -> float:
    if league is None:
        return 0.5
    for attr in ("chaos_index",):
        if hasattr(league, attr):
            try:
                return float(getattr(league, attr) or 0.5)
            except (TypeError, ValueError):
                pass
    v = getattr(league, "_chaos_index", None)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    fc = getattr(league, "last_season_forecast", None)
    if fc is not None:
        try:
            return float(getattr(fc, "chaos_index", 0.5) or 0.5)
        except (TypeError, ValueError):
            pass
    ctx = getattr(league, "_tuning_context", None) or {}
    try:
        return float(ctx.get("chaos_index", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5


def calculate_contract_inflation(league: Any) -> float:
    cap_growth = _league_cap_growth_rate(league)
    chaos_index = _league_chaos_index_value(league)
    base = 1.0 + cap_growth * 1.5
    chaos_factor = 1.0 + chaos_index * 0.5
    return float(base * chaos_factor)


def _player_rating_0_100(player: Any) -> float:
    try:
        ovr = float(player.ovr()) if callable(getattr(player, "ovr", None)) else float(getattr(player, "ovr", 0.5))
    except Exception:
        ovr = 0.5
    return max(1.0, min(99.0, ovr * 100.0))


def _player_performance_rating_base_0_100(player: Any) -> float:
    pr = getattr(player, "performance_rating", None)
    if pr is not None:
        try:
            pf = float(pr)
            if pf > 0:
                return max(1.0, min(120.0, pf))
        except (TypeError, ValueError):
            pass
    return _player_rating_0_100(player)


def is_bad_contract(player: Any) -> bool:
    contract_value = _economy_player_cap_hit_millions(player)
    rating = _player_rating_0_100(player)
    value_ratio = contract_value / max(rating, 1.0)
    return bool(value_ratio > 0.35)


def sync_bad_contract_flag(player: Any) -> bool:
    bad = is_bad_contract(player)
    setattr(player, "bad_contract", bool(bad))
    return bool(bad)


def mark_team_roster_bad_contracts(team: Any) -> int:
    n = 0
    for p in _team_roster_players(team):
        if getattr(p, "retired", False):
            continue
        if sync_bad_contract_flag(p):
            n += 1
    return n


def bad_contract_payroll_stress_millions(team: Any) -> float:
    """Synthetic cap burden from bad deals (raises pressure / urgency without changing real payroll)."""
    s = 0.0
    for p in _team_roster_players(team):
        if getattr(p, "retired", False):
            continue
        sync_bad_contract_flag(p)
        if getattr(p, "bad_contract", False):
            s += _economy_player_cap_hit_millions(p) * 0.065
    return float(s)


def cap_casualty_check(team: Any, *, salary_cap_m: Optional[float] = None) -> Optional[Dict[str, Any]]:
    pressure = calculate_cap_pressure(team, salary_cap_m=salary_cap_m)
    tier = str(getattr(team, "cap_pressure_tier", "") or "").lower()
    urgent = tier in ("critical", "cap_hell")
    if pressure <= 1.0 and not urgent:
        return None
    roster = [p for p in _team_roster_players(team) if not getattr(p, "retired", False)]
    bad_contracts = [p for p in roster if is_bad_contract(p)]
    if not bad_contracts:
        return None
    player = max(bad_contracts, key=lambda p: _economy_player_cap_hit_millions(p))
    return {"type": "cap_dump", "player": player}


def can_afford(team: Any, contract: float, *, salary_cap_m: Optional[float] = None) -> bool:
    cap_m = _resolve_salary_cap_millions(team, salary_cap_m)
    if cap_m <= 0:
        return True
    payroll = _team_payroll_millions(team)
    projected = payroll + float(contract)
    tier = str(getattr(team, "cap_pressure_tier", "") or "").lower()
    room = 1.05
    if tier == "high":
        room = 1.02
    elif tier == "critical":
        room = 1.0
    elif tier == "cap_hell":
        room = 0.97
    return projected <= cap_m * room


def adjust_player_demands(
    player: Any,
    league: Any,
    *,
    base_contract: float,
    team: Any = None,
    rng: Optional[random.Random] = None,
) -> float:
    demand = float(base_contract)
    pers = str(getattr(player, "personality", "") or "").lower()
    if pers == "toxic":
        demand *= 1.25
    if _league_chaos_index_value(league) > 0.6:
        demand *= 1.15
    perf = _player_performance_rating_base_0_100(player)
    if perf > 85:
        demand *= 1.20
    cg = _league_cap_growth_rate(league)
    if cg >= 0.055 and perf > 82:
        demand *= 1.0 + min(0.22, (cg - 0.04) * 3.4)
    r = rng or random.Random((id(player) ^ int(demand * 1e6)) % 2**32)
    arch = ""
    if team is not None:
        arch = str(
            getattr(team, "_runner_team_archetype", None)
            or getattr(team, "runner_archetype", None)
            or getattr(team, "team_archetype", None)
            or ""
        ).lower()
    if arch == "win_now":
        demand *= float(r.uniform(1.05, 1.25))
    elif arch == "chaos_agent":
        demand *= 1.0 + float(r.uniform(0.0, 0.40))
    return float(demand)


def apply_cap_pressure_effects(team: Any, *, salary_cap_m: Optional[float] = None) -> None:
    tier = str(getattr(team, "cap_pressure_tier", "") or "").lower()
    pressure = calculate_cap_pressure(team, salary_cap_m=salary_cap_m)
    for player in _team_roster_players(team):
        if getattr(player, "retired", False):
            continue
        base = _player_rating_0_100(player)
        if tier == "cap_hell":
            base *= 0.935
        elif tier == "critical":
            base *= 0.965
        elif tier == "high":
            base *= 0.985
        elif tier == "low":
            base *= 1.018
        elif pressure > 0.9:
            base *= 0.97
        elif pressure < 0.4:
            base *= 1.02
        setattr(player, "performance_rating", max(1.0, min(120.0, base)))
    if tier == "cap_hell":
        st = getattr(team, "state", None)
        if st is not None and hasattr(st, "team_morale"):
            try:
                cur = float(getattr(st, "team_morale", 0.5) or 0.5)
                setattr(st, "team_morale", max(0.12, cur - 0.045))
            except Exception:
                pass
        for player in _team_roster_players(team):
            if getattr(player, "retired", False):
                continue
            psych = getattr(player, "psych", None)
            if psych is None:
                continue
            try:
                m = float(getattr(psych, "morale", 0.5) or 0.5)
                setattr(psych, "morale", max(0.15, m - 0.022))
            except Exception:
                pass


# --- Career lifecycle (major progression: resolve_authoritative_major_progression_event) ---
# AUDIT: Logged BREAKOUT/LATE BLOOM/BUST/AGING DECLINE apply only there. run_sim runs
# run_player_progression once, then lifecycle with skip_base_progress=True (no duplicate progress_player).
# apply_aging_calibration is a no-op; coach/system fit nudges psych only (no permanent OVR).


def assign_development_profile(player: Any, rng: random.Random) -> str:
    existing = getattr(player, "dev_type", None)
    if existing:
        return str(existing)
    roll = rng.random()
    if roll < 0.20:
        dt = "elite"
    elif roll < 0.50:
        dt = "standard"
    elif roll < 0.75:
        dt = "slow"
    elif roll < 0.90:
        dt = "late_bloomer"
    else:
        dt = "bust"
    setattr(player, "dev_type", dt)
    return dt


def get_age_curve(age: int) -> float:
    if age <= 20:
        return 1.3
    if age <= 23:
        return 1.15
    if age <= 26:
        return 1.05
    if age <= 29:
        return 1.0
    if age <= 32:
        return 0.95
    if age <= 35:
        return 0.90
    return 0.80


def development_multiplier(player: Any) -> float:
    dt = str(getattr(player, "dev_type", "standard") or "standard")
    age = career_player_age(player)
    if dt == "elite":
        return 1.3
    if dt == "standard":
        return 1.0
    if dt == "slow":
        return 0.8
    if dt == "late_bloomer":
        return 0.6 if age < 24 else 1.4
    if dt == "bust":
        return 0.5
    return 1.0


def career_player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    if ident is not None and hasattr(ident, "age"):
        try:
            return int(getattr(ident, "age"))
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(player, "age", 25))
    except (TypeError, ValueError):
        return 25


def career_player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    if ident is not None and getattr(ident, "name", None):
        return str(ident.name)
    n = getattr(player, "name", None)
    if n:
        return str(n)
    return "?"


def career_ovr_0_100(player: Any) -> float:
    return float(_player_rating_0_100(player))


def _career_apply_rating_delta_0_100(player: Any, delta_0_100: float) -> None:
    if abs(delta_0_100) < 1e-9:
        return
    ratings = getattr(player, "ratings", None)
    if not ratings:
        return
    keys = list(ratings.keys())
    if not keys:
        return
    per = float(delta_0_100) / float(len(keys))
    set_fn = getattr(player, "set", None)
    get_fn = getattr(player, "get", None)
    if callable(set_fn) and callable(get_fn):
        for k in keys:
            try:
                set_fn(k, float(get_fn(k, 50)) + per)
            except Exception:
                pass
        return
    for k in keys:
        try:
            ratings[k] = clamp_rating(float(ratings[k]) + per)
        except Exception:
            pass


def _career_clamp_ovr_window(player: Any, lo: float = 40.0, hi: float = 99.0) -> None:
    cur = career_ovr_0_100(player)
    if lo <= cur <= hi:
        return
    target = max(lo, min(hi, cur))
    _career_apply_rating_delta_0_100(player, target - cur)


def progress_player(player: Any, rng: random.Random) -> None:
    assign_career_phase_from_age(player)
    age = career_player_age(player)
    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(age))
    age_factor = get_age_curve(age)
    dev_factor = development_multiplier(player)
    change = rng.uniform(0.5, 2.5)
    delta = change * age_factor * dev_factor
    style = str(getattr(player, "playstyle", "") or "").lower()
    if style in ("sniper", "playmaker", "offensive_d"):
        delta *= 1.045
    elif style in ("defensive_d", "enforcer_d", "grinder"):
        delta *= 1.028
    elif style in ("two_way", "two_way_d", "hybrid", "butterfly", "aggressive"):
        delta *= 1.018
    if age > 30:
        # Light background drift; structured aging is regression_check V3 (AGING DECLINE log).
        if age <= 32:
            delta = rng.uniform(-0.35, 0.48) * dev_factor
        elif age <= 35:
            delta = rng.uniform(-0.28, 0.22) * dev_factor
        else:
            delta = -rng.uniform(0.12, 0.55) * dev_factor
    trend = str(getattr(player, "trend", "stable") or "stable").lower()
    if trend == "hot":
        if delta > 0:
            delta *= 1.16
        elif delta < 0:
            delta *= 0.74
    elif trend == "declining":
        if delta < 0:
            delta *= 1.12
        elif delta > 0:
            delta *= 0.72
    if phase == PHASE_PRIME:
        delta *= 0.5
        if delta > 3.0:
            delta = 3.0
        if delta < -3.0:
            delta = -3.0
    if delta > 0:
        delta = min(float(delta), 1.22)
    _career_apply_rating_delta_0_100(player, delta)
    setattr(player, "rating", round(career_ovr_0_100(player), 3))
    _career_clamp_ovr_window(player, 40.0, 99.0)


def _lifecycle_macro_from_league(league: Any) -> Dict[str, float]:
    if league is None:
        return {
            "breakout_p_mult": 1.0,
            "decline_mag_mult": 1.0,
            "bust_p_mult": 1.0,
            "late_bloom_p_mult": 1.0,
        }
    ctx = dict(getattr(league, "_tuning_context", None) or {})
    try:
        from app.sim_engine.tuning import normalization as _norm

        return _norm.macro_progression_scales(ctx)
    except Exception:
        return {
            "breakout_p_mult": 1.0,
            "decline_mag_mult": 1.0,
            "bust_p_mult": 1.0,
            "late_bloom_p_mult": 1.0,
        }


def _lifecycle_bump_breakout(league: Any) -> None:
    if league is None:
        return
    try:
        setattr(league, "_lifecycle_used_breakouts", int(getattr(league, "_lifecycle_used_breakouts", 0)) + 1)
    except Exception:
        pass


# Console-only trace for diagnosing duplicate/legacy breakout paths (keep False in production).
_BREAKOUT_RESOLUTION_TRACE: bool = False
_LATE_BLOOM_RESOLUTION_TRACE: bool = False
_PROGRESSION_CONTROLLER_TRACE: bool = False
# Temporary: full authoritative progression audit lines (player/season/event/pre/delta/post).
_AUTHORITATIVE_PROGRESSION_DEBUG: bool = False
# One-line proof of clamp/budget/cooldown per approved special event (set False to silence).
_LOG_SPECIAL_PROGRESSION_ENFORCEMENT: bool = True

# --- Central special-progression hard limits (0–100 OVR scale); impossible to exceed via engine apply+log ---
SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP: float = 5.3
SPECIAL_PROGRESSION_LATE_BLOOM_HARD_CAP: float = 4.0
_SPECIAL_TOP_BREAKOUT_DELTA: float = 4.22
_SPECIAL_TOP_LATE_BLOOM_DELTA: float = 3.12
_SPECIAL_ENFORCE_FLOOR: float = 0.17


def _emit_breakout_resolution_line(msg: str) -> None:
    if _BREAKOUT_RESOLUTION_TRACE:
        print(msg)


def _emit_late_bloom_resolution_line(msg: str) -> None:
    if _LATE_BLOOM_RESOLUTION_TRACE:
        print(msg)


def _emit_progression_controller_line(msg: str) -> None:
    if _PROGRESSION_CONTROLLER_TRACE:
        print(msg)


def _breakout_potential_category(player: Any) -> str:
    """Map float/string potential + position to elite | top6 | top4 | top6d | top9 | bottom6."""
    raw = getattr(player, "potential", None)
    if isinstance(raw, str):
        s = raw.lower().strip()
        if s in ("elite", "franchise", "generational", "superstar"):
            return "elite"
        if s in ("top6d", "top_6d", "top6_d"):
            return "top6d"
        if s in ("top6", "top_6", "topline", "top_line"):
            return "top6"
        if s in ("top4", "top_4", "toppair", "top_pair"):
            return "top4"
        if s in ("top9", "top_9", "middle6", "middle_6", "middle_six", "third_line", "middle"):
            return "top9"
        if s in ("bottom6", "bottom_6", "depth", "replaceable", "ahl"):
            return "bottom6"
    try:
        p = float(raw) if raw is not None else 0.62
    except (TypeError, ValueError):
        p = 0.62
    pos = getattr(player, "position", None)
    pv = getattr(pos, "value", pos)
    pos_s = str(pv or "").upper()
    is_d = pos_s in ("D", "LD", "RD") or pos_s == "D"
    if p >= 0.84:
        return "elite"
    if p >= 0.70:
        return "top6d" if is_d else "top6"
    if p >= 0.55:
        return "top9"
    return "bottom6"


def reset_career_breakout_season_flags(teams: Any) -> None:
    """Clear per-player seasonal progression guards at lifecycle start (run_sim / sim_year)."""
    if not teams:
        return
    for tm in teams:
        for pl in getattr(tm, "roster", None) or []:
            try:
                setattr(pl, "_career_breakout_logged_this_season", False)
                setattr(pl, "_career_late_bloom_logged_this_season", False)
                setattr(pl, "progression_event_this_season", None)
                setattr(pl, "major_progression_event_this_season", None)
                setattr(pl, "_lifecycle_ovr_before_special", None)
            except Exception:
                pass


def _late_bloom_trajectory_allows(player: Any, ovr100: float) -> bool:
    tr = str(getattr(player, "trend", "stable") or "stable").lower()
    if tr == "hot":
        return True
    raw_p = getattr(player, "potential", None)
    try:
        pf = float(raw_p) if raw_p is not None else 0.66
    except (TypeError, ValueError):
        pf = 0.66
    if pf > 1.5:
        pf = pf / 99.0
    ceiling = min(93.0, pf * 100.0 + 3.0)
    return bool(ovr100 < ceiling - 3.5)


def _prog_breakout_global_slot_available(league: Any) -> bool:
    if league is None or not getattr(league, "_progression_controller_primed", False):
        return True
    u = int(getattr(league, "_prog_global_breakouts_used", 0) or 0)
    mx = int(getattr(league, "_prog_max_breakouts", 8) or 8)
    return u < mx


def _prog_late_bloom_global_slot_available(league: Any) -> bool:
    if league is None or not getattr(league, "_progression_controller_primed", False):
        return True
    u = int(getattr(league, "_prog_global_late_blooms_used", 0) or 0)
    mx = int(getattr(league, "_prog_max_late_blooms", 4) or 4)
    return u < mx


def _prog_bust_global_slot_available(league: Any) -> bool:
    if league is None or not getattr(league, "_progression_controller_primed", False):
        return True
    u = int(getattr(league, "_prog_global_busts_used", 0) or 0)
    mx = int(getattr(league, "_prog_max_busts", 5) or 5)
    return u < mx


def _player_has_late_bloomed_career(player: Any) -> bool:
    return bool(getattr(player, "has_late_bloomed", False)) or bool(
        getattr(player, "_career_late_bloom_done", False)
    )


def _major_progression_slot_clear(player: Any) -> bool:
    return (
        getattr(player, "major_progression_event_this_season", None) is None
        and getattr(player, "progression_event_this_season", None) is None
    )


def _emit_authoritative_progression_debug(msg: str) -> None:
    if _AUTHORITATIVE_PROGRESSION_DEBUG:
        print(msg)


def _emit_special_enforcement_line(
    *,
    pname: str,
    kind: str,
    raw_draw: float,
    tapered_pre: float,
    final_applied: float,
    hard_cap: float,
    budget_line: str,
    notes: str,
) -> None:
    if not _LOG_SPECIAL_PROGRESSION_ENFORCEMENT:
        return
    print(
        f"PROGRESSION ENFORCE: {kind} player={pname} raw={raw_draw:+.2f} "
        f"tapered_pre={tapered_pre:+.2f} final={final_applied:+.2f} cap={hard_cap:.2f} "
        f"budget={budget_line} notes={notes}"
    )


def _special_progression_budget_line(league: Any, kind: str) -> str:
    if league is None or not getattr(league, "_progression_controller_primed", False):
        return "n/a"
    if kind == "breakout":
        u = int(getattr(league, "_prog_global_breakouts_used", 0) or 0)
        m = int(getattr(league, "_prog_max_breakouts", 0) or 0)
        tu = int(getattr(league, "_prog_top_breakouts_used", 0) or 0)
        tm = int(getattr(league, "_prog_max_top_breakouts", 0) or 0)
        return f"bo={u}/{m} top_bo={tu}/{tm}"
    if kind == "late_bloom":
        u = int(getattr(league, "_prog_global_late_blooms_used", 0) or 0)
        m = int(getattr(league, "_prog_max_late_blooms", 0) or 0)
        tu = int(getattr(league, "_prog_top_late_blooms_used", 0) or 0)
        tm = int(getattr(league, "_prog_max_top_late_blooms", 0) or 0)
        return f"lb={u}/{m} top_lb={tu}/{tm}"
    return "n/a"


def _enforce_positive_special_delta(
    league: Any,
    player: Any,
    kind: str,
    tapered_from_roll: float,
    raw_draw: float,
    ovr_snapshot: float,
) -> Tuple[float, str]:
    """
    Final approval for positive breakout/late_bloom: hard cap, league top-tier budgets,
    main-budget stress, per-player cooldown, anti-stack vs pre-special OVR. Cannot exceed cap.
    """
    cap = (
        float(SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP)
        if kind == "breakout"
        else float(SPECIAL_PROGRESSION_LATE_BLOOM_HARD_CAP)
    )
    note_parts: List[str] = []
    v = min(float(tapered_from_roll), cap)
    if v > cap + 1e-9:
        v = cap
        note_parts.append("hard_cap_clip")

    top_thr = _SPECIAL_TOP_BREAKOUT_DELTA if kind == "breakout" else _SPECIAL_TOP_LATE_BLOOM_DELTA

    if league is not None:
        if kind == "breakout":
            top_u = int(getattr(league, "_prog_top_breakouts_used", 0) or 0)
            top_m = max(0, int(getattr(league, "_prog_max_top_breakouts", 0) or 0))
            if v > top_thr and top_m > 0 and top_u >= top_m:
                v = min(v, top_thr)
                note_parts.append("top_tier_quota_downgrade")
            used = int(getattr(league, "_prog_global_breakouts_used", 0) or 0)
            mx = max(1, int(getattr(league, "_prog_max_breakouts", 8) or 8))
            if used >= max(1, int(mx * 0.72)):
                v = min(v, 3.42)
                note_parts.append("main_budget_stress")
        else:
            top_u = int(getattr(league, "_prog_top_late_blooms_used", 0) or 0)
            top_m = max(0, int(getattr(league, "_prog_max_top_late_blooms", 0) or 0))
            if v > top_thr and top_m > 0 and top_u >= top_m:
                v = min(v, top_thr)
                note_parts.append("top_tier_lb_downgrade")
            used = int(getattr(league, "_prog_global_late_blooms_used", 0) or 0)
            mx = max(1, int(getattr(league, "_prog_max_late_blooms", 4) or 4))
            if used >= max(1, int(mx * 0.65)):
                v = min(v, 2.62)
                note_parts.append("lb_budget_stress")

    sy = int(getattr(league, "_progression_season_year", -1) or -1) if league is not None else -1
    last_sy = getattr(player, "_special_progression_last_season_year", None)
    last_mag = float(getattr(player, "_special_progression_last_positive_mag", 0) or 0)
    if last_sy is not None and sy >= 0:
        gap = sy - int(last_sy)
        if gap <= 1 and last_mag >= 3.05:
            v = min(v, 2.72)
            note_parts.append("player_cooldown_tight")
        elif gap <= 2 and last_mag >= 4.05:
            v = min(v, 3.02)
            note_parts.append("player_cooldown_mid")

    s = float(ovr_snapshot)
    if s >= 86.0:
        v = min(v, 2.62)
        note_parts.append("anti_stack_ovr86+")
    elif s >= 84.0:
        v = min(v, 3.28)
        note_parts.append("anti_stack_ovr84+")
    elif s >= 82.0:
        v = min(v, min(cap, 3.95))
        note_parts.append("anti_stack_ovr82+")

    v = min(v, cap)
    v = max(0.0, v)
    if v < _SPECIAL_ENFORCE_FLOOR:
        note_parts.append("below_enforce_floor")
    return v, "+".join(note_parts) if note_parts else "ok"


def _clamp_breakout_magnitude(
    raw: float,
    *,
    age: int,
    ovr_pre: float,
    pot_cat: str,
) -> float:
    """Taper then hard-cap breakout delta on 0–100 OVR scale."""
    amt = float(raw)
    if amt <= 0:
        return 0.0
    if pot_cat in ("top9", "bottom6"):
        amt = min(amt, 3.8)
    elif pot_cat not in ("elite", "top6", "top4", "top6d"):
        amt = min(amt, 3.8)
    if age >= 24:
        amt = min(amt, 4.2)
    if ovr_pre >= 84.0:
        amt *= 0.88
    if ovr_pre >= 86.5:
        amt *= 0.92
    amt = min(amt, float(SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP))
    return max(0.0, amt)


def _roll_breakout_delta_controller(
    player: Any,
    rng: random.Random,
    league: Any,
    macro: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[float], float]:
    """Roll-only: (tapered_candidate, raw_draw) for enforcement layer; raw_draw is pre-taper uniform."""
    if getattr(player, "retired", False):
        return None, 0.0
    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return None, 0.0
    if bool(getattr(player, "has_had_breakout", False)):
        return None, 0.0
    if _player_has_late_bloomed_career(player):
        return None, 0.0
    if not _prog_breakout_global_slot_available(league):
        return None, 0.0

    assign_career_phase_from_age(player)
    age = career_player_age(player)
    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(age))
    if phase == PHASE_DECLINING:
        return None, 0.0
    if age < 18 or age > 24:
        return None, 0.0

    ovr_pre = career_ovr_0_100(player)
    if ovr_pre >= 88.0:
        return None, 0.0

    pot_cat = _breakout_potential_category(player)
    if pot_cat == "bottom6":
        return None, 0.0
    tr = str(getattr(player, "trend", "stable") or "stable").lower()
    high_pot = pot_cat in ("elite", "top6", "top4", "top6d", "top9")
    if not high_pot and tr != "hot":
        return None, 0.0

    brm = float((macro or {}).get("breakout_p_mult", 1.0))
    nar_bo = float(getattr(player, "_narrative_breakout_p_mult", 1.0) or 1.0)
    p = 0.032 * brm * nar_bo
    if pot_cat == "elite":
        p *= 1.15
    if tr == "hot":
        p *= 1.08
    if tr == "hot" and ovr_pre >= 83.0:
        p *= 0.65
    used_bo = int(getattr(league, "_prog_global_breakouts_used", 0) or 0) if league is not None else 0
    mx_bo = max(1, int(getattr(league, "_prog_max_breakouts", 8) or 8)) if league is not None else 8
    if league is not None and used_bo >= max(1, int(mx_bo * 0.55)):
        p *= 0.68
    p = max(0.0, min(0.048, p))
    if rng.random() >= p:
        return None, 0.0

    r = rng.random()
    if r < 0.48:
        raw = float(rng.uniform(2.0, 2.75))
    elif r < 0.78:
        raw = float(rng.uniform(2.75, 3.35))
    elif r < 0.92:
        raw = float(rng.uniform(3.35, 3.95))
    elif r < 0.985:
        raw = float(rng.uniform(3.95, 4.45))
    else:
        cap_u = float(SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP) - 0.02
        raw = float(rng.uniform(4.45, max(4.46, cap_u)))

    amt = _clamp_breakout_magnitude(raw, age=age, ovr_pre=ovr_pre, pot_cat=pot_cat)
    if amt < 0.2:
        return None, raw
    amt = min(float(amt), float(SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP))
    return float(amt), float(raw)


def _clamp_late_bloom_magnitude(raw: float, *, age: int, ovr_pre: float, pot_cat: str) -> float:
    amt = float(raw)
    if amt <= 0:
        return 0.0
    if pot_cat in ("top9", "bottom6"):
        amt = min(amt, 2.65)
    if age >= 28:
        amt = min(amt, 3.15)
    if ovr_pre >= 85.0:
        amt *= 0.80
    if ovr_pre >= 87.5:
        amt *= 0.86
    amt = min(amt, float(SPECIAL_PROGRESSION_LATE_BLOOM_HARD_CAP))
    return max(0.0, amt)


def _roll_late_bloom_delta_controller(
    player: Any,
    rng: random.Random,
    league: Any,
    macro: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[float], float]:
    if getattr(player, "retired", False):
        return None, 0.0
    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return None, 0.0
    if bool(getattr(player, "has_had_breakout", False)):
        return None, 0.0
    if str(getattr(player, "dev_type", "")) != "late_bloomer":
        return None, 0.0
    if _player_has_late_bloomed_career(player):
        return None, 0.0
    if not _prog_late_bloom_global_slot_available(league):
        return None, 0.0

    assign_career_phase_from_age(player)
    age = career_player_age(player)
    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(age))
    if phase == PHASE_DECLINING:
        return None, 0.0
    if age < 23 or age > 29:
        return None, 0.0

    ovr_pre = career_ovr_0_100(player)
    if ovr_pre >= 90.0:
        return None, 0.0
    pot_cat = _breakout_potential_category(player)
    if pot_cat == "bottom6":
        return None, 0.0
    if not _late_bloom_trajectory_allows(player, ovr_pre):
        return None, 0.0

    if 24 <= age <= 27:
        base_p = 0.016
    elif age == 23:
        base_p = 0.012
    else:
        base_p = 0.009
    mult = {
        PHASE_EMERGING: 1.0,
        PHASE_PRIME: 0.74,
        PHASE_VETERAN: 0.46,
        PHASE_PROSPECT: 0.42,
    }.get(phase, 0.45)
    try:
        raw_pf = getattr(player, "potential", None)
        pf_lb = float(raw_pf) if raw_pf is not None else 0.66
    except (TypeError, ValueError):
        pf_lb = 0.66
    if pf_lb > 1.5:
        pf_lb = pf_lb / 99.0
    ceiling_lb = min(93.0, pf_lb * 100.0 + 3.0)
    tr_lb = str(getattr(player, "trend", "stable") or "stable").lower()
    if ovr_pre < ceiling_lb - 5.0 and tr_lb in ("hot", "improving", "rising", "up"):
        base_p *= 1.08
    lb = float((macro or {}).get("late_bloom_p_mult", 1.0))
    used_lb = int(getattr(league, "_prog_global_late_blooms_used", 0) or 0) if league is not None else 0
    mx_lb = max(1, int(getattr(league, "_prog_max_late_blooms", 4) or 4)) if league is not None else 4
    if league is not None and used_lb >= max(1, int(mx_lb * 0.5)):
        base_p *= 0.58
    nar_lb = float(getattr(player, "_narrative_breakout_p_mult", 1.0) or 1.0)
    p = min(0.032, max(0.0, base_p * mult * lb * nar_lb))
    if rng.random() >= p:
        return None, 0.0

    r = rng.random()
    cap_lb = float(SPECIAL_PROGRESSION_LATE_BLOOM_HARD_CAP)
    if r < 0.58:
        raw = float(rng.uniform(1.5, 2.35))
    elif r < 0.86:
        raw = float(rng.uniform(2.35, 2.95))
    elif r < 0.965:
        raw = float(rng.uniform(2.95, 3.45))
    elif r < 0.995:
        raw = float(rng.uniform(3.45, 3.78))
    else:
        raw = float(rng.uniform(3.78, max(3.79, cap_lb - 0.04)))

    amt = _clamp_late_bloom_magnitude(raw, age=age, ovr_pre=ovr_pre, pot_cat=pot_cat)
    if amt < 0.22:
        return None, raw
    amt = min(float(amt), cap_lb)
    return float(amt), float(raw)


def _roll_bust_delta_controller(
    player: Any,
    rng: random.Random,
    league: Any,
    macro: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    if str(getattr(player, "dev_type", "")) != "bust":
        return None
    assign_career_phase_from_age(player)
    age = career_player_age(player)
    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(age))
    if age >= 25:
        return None
    if not _prog_bust_global_slot_available(league):
        return None
    bust_m = float((macro or {}).get("bust_p_mult", 1.0))
    nar_decl = float(getattr(player, "_narrative_decline_p_mult", 1.0) or 1.0)
    thr = min(0.14, 0.085 * bust_m) * max(0.72, min(1.35, nar_decl))
    thr = min(0.17, max(0.04, thr))
    if rng.random() >= thr:
        return None
    r = rng.random()
    if r < 0.72:
        drop = float(rng.uniform(1.5, 3.0))
    elif r < 0.96:
        drop = float(rng.uniform(3.0, 4.2))
    else:
        drop = float(rng.uniform(4.2, 4.8))
    if phase == PHASE_PRIME:
        drop = min(drop, 3.2)
    return max(1.2, min(drop, 4.8))


def resolve_authoritative_major_progression_event(
    player: Any,
    rng: random.Random,
    *,
    macro: Optional[Dict[str, float]] = None,
    league: Any = None,
    season_year: Optional[int] = None,
) -> Optional[str]:
    """
    Single seasonal authority for major progression: at most one of
    breakout / late_bloom / bust_trend / AGING DECLINE per player per season.
    Magnitudes are clamped before apply; logged value equals applied delta.
    """
    m = macro or {}
    if getattr(player, "retired", False):
        return None
    if not _major_progression_slot_clear(player):
        return None
    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return None

    sy = int(season_year) if season_year is not None else -1
    pname = career_player_name(player)
    pot_cat = _breakout_potential_category(player)

    bo_tapered, bo_raw = _roll_breakout_delta_controller(player, rng, league, m)
    if bo_tapered is not None and bo_tapered > 0:
        o_snap = float(getattr(player, "_lifecycle_ovr_before_special", None) or career_ovr_0_100(player))
        final_bo, bo_notes = _enforce_positive_special_delta(
            league, player, "breakout", float(bo_tapered), float(bo_raw), o_snap
        )
        bo_cap = float(SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP)
        bo_budget = _special_progression_budget_line(league, "breakout")
        if final_bo < _SPECIAL_ENFORCE_FLOOR:
            _emit_special_enforcement_line(
                pname=pname,
                kind="breakout",
                raw_draw=float(bo_raw),
                tapered_pre=float(bo_tapered),
                final_applied=0.0,
                hard_cap=bo_cap,
                budget_line=bo_budget,
                notes=f"rejected:{bo_notes}",
            )
        else:
            applied_bo = round(min(float(final_bo), bo_cap), 1)
            applied_bo = min(applied_bo, bo_cap)
            o_pre = career_ovr_0_100(player)
            _career_apply_rating_delta_0_100(player, float(applied_bo))
            setattr(player, "rating", round(career_ovr_0_100(player), 3))
            _career_clamp_ovr_window(player, 40.0, 99.0)
            set_player_trend(player, "hot", 0, rng)
            try:
                setattr(player, "has_had_breakout", True)
                setattr(player, "_career_breakout_logged_this_season", True)
                setattr(player, "progression_event_this_season", "breakout")
                setattr(player, "major_progression_event_this_season", "breakout")
                if sy >= 0:
                    setattr(player, "_special_progression_last_season_year", sy)
                    setattr(player, "_special_progression_last_positive_mag", float(applied_bo))
            except Exception:
                pass
            if league is not None:
                try:
                    setattr(
                        league,
                        "_prog_global_breakouts_used",
                        int(getattr(league, "_prog_global_breakouts_used", 0) or 0) + 1,
                    )
                    setattr(
                        league,
                        "_season_breakout_events",
                        int(getattr(league, "_season_breakout_events", 0) or 0) + 1,
                    )
                    if float(applied_bo) > _SPECIAL_TOP_BREAKOUT_DELTA:
                        setattr(
                            league,
                            "_prog_top_breakouts_used",
                            int(getattr(league, "_prog_top_breakouts_used", 0) or 0) + 1,
                        )
                except Exception:
                    pass
            o_post = career_ovr_0_100(player)
            _emit_special_enforcement_line(
                pname=pname,
                kind="breakout",
                raw_draw=float(bo_raw),
                tapered_pre=float(bo_tapered),
                final_applied=float(applied_bo),
                hard_cap=bo_cap,
                budget_line=bo_budget,
                notes=bo_notes,
            )
            _emit_authoritative_progression_debug(
                f"DEBUG PROGRESSION:\n  player={pname} season={sy} event=breakout\n"
                f"  age={career_player_age(player)} pot={pot_cat} pre={o_pre:.2f} delta=+{applied_bo:.2f} post={o_post:.2f}\n"
                f"  source=authoritative_progression_controller blocked_other_events=True"
            )
            if _PROGRESSION_CONTROLLER_TRACE:
                _emit_progression_controller_line(
                    f"DEBUG: [{getattr(player, 'id', None) or pname}] event=breakout age={career_player_age(player)} "
                    f"pre={o_pre:.1f} delta=+{applied_bo:.1f} post={o_post:.1f}"
                )
            return f"BREAKOUT: +{applied_bo:.1f} OVR"

    lb_tapered, lb_raw = _roll_late_bloom_delta_controller(player, rng, league, m)
    if lb_tapered is not None and lb_tapered > 0:
        o_snap_lb = float(getattr(player, "_lifecycle_ovr_before_special", None) or career_ovr_0_100(player))
        final_lb, lb_notes = _enforce_positive_special_delta(
            league, player, "late_bloom", float(lb_tapered), float(lb_raw), o_snap_lb
        )
        lb_cap = float(SPECIAL_PROGRESSION_LATE_BLOOM_HARD_CAP)
        lb_budget = _special_progression_budget_line(league, "late_bloom")
        if final_lb < _SPECIAL_ENFORCE_FLOOR:
            _emit_special_enforcement_line(
                pname=pname,
                kind="late_bloom",
                raw_draw=float(lb_raw),
                tapered_pre=float(lb_tapered),
                final_applied=0.0,
                hard_cap=lb_cap,
                budget_line=lb_budget,
                notes=f"rejected:{lb_notes}",
            )
        else:
            applied_lb = round(min(float(final_lb), lb_cap), 1)
            applied_lb = min(applied_lb, lb_cap)
            o_pre = career_ovr_0_100(player)
            _career_apply_rating_delta_0_100(player, float(applied_lb))
            setattr(player, "rating", round(career_ovr_0_100(player), 3))
            _career_clamp_ovr_window(player, 40.0, 99.0)
            set_player_trend(player, "hot", 0, rng)
            try:
                setattr(player, "has_late_bloomed", True)
                setattr(player, "_career_late_bloom_done", True)
                setattr(player, "_career_late_bloom_logged_this_season", True)
                setattr(player, "progression_event_this_season", "late_bloom")
                setattr(player, "major_progression_event_this_season", "late_bloom")
                if sy >= 0:
                    setattr(player, "_special_progression_last_season_year", sy)
                    setattr(player, "_special_progression_last_positive_mag", float(applied_lb))
            except Exception:
                pass
            if league is not None:
                try:
                    setattr(
                        league,
                        "_prog_global_late_blooms_used",
                        int(getattr(league, "_prog_global_late_blooms_used", 0) or 0) + 1,
                    )
                    if float(applied_lb) > _SPECIAL_TOP_LATE_BLOOM_DELTA:
                        setattr(
                            league,
                            "_prog_top_late_blooms_used",
                            int(getattr(league, "_prog_top_late_blooms_used", 0) or 0) + 1,
                        )
                except Exception:
                    pass
            o_post = career_ovr_0_100(player)
            _emit_special_enforcement_line(
                pname=pname,
                kind="late_bloom",
                raw_draw=float(lb_raw),
                tapered_pre=float(lb_tapered),
                final_applied=float(applied_lb),
                hard_cap=lb_cap,
                budget_line=lb_budget,
                notes=lb_notes,
            )
            _emit_authoritative_progression_debug(
                f"DEBUG PROGRESSION:\n  player={pname} season={sy} event=late_bloom\n"
                f"  age={career_player_age(player)} pot={pot_cat} pre={o_pre:.2f} delta=+{applied_lb:.2f} post={o_post:.2f}\n"
                f"  source=authoritative_progression_controller blocked_other_events=True"
            )
            return f"LATE BLOOM: +{applied_lb:.1f} OVR"

    bust_drop = _roll_bust_delta_controller(player, rng, league, m)
    if bust_drop is not None and bust_drop > 0:
        o_pre = career_ovr_0_100(player)
        _career_apply_rating_delta_0_100(player, -float(bust_drop))
        setattr(player, "rating", round(career_ovr_0_100(player), 3))
        _career_clamp_ovr_window(player, 40.0, 99.0)
        try:
            setattr(player, "has_had_major_bust", True)
            setattr(player, "progression_event_this_season", "bust_trend")
            setattr(player, "major_progression_event_this_season", "bust_trend")
        except Exception:
            pass
        if league is not None:
            try:
                setattr(
                    league,
                    "_prog_global_busts_used",
                    int(getattr(league, "_prog_global_busts_used", 0) or 0) + 1,
                )
            except Exception:
                pass
        o_post = career_ovr_0_100(player)
        _emit_authoritative_progression_debug(
            f"DEBUG PROGRESSION:\n  player={pname} season={sy} event=bust_trend\n"
            f"  age={career_player_age(player)} pot={pot_cat} pre={o_pre:.2f} delta=-{bust_drop:.2f} post={o_post:.2f}\n"
            f"  source=authoritative_progression_controller blocked_other_events=True"
        )
        return f"BUST TREND: -{round(bust_drop, 1)} OVR"

    decline = _career_aging_decline_try_v3(player, rng, league, m)
    if decline is not None and decline > 0:
        o_pre = career_ovr_0_100(player)
        _career_apply_rating_delta_0_100(player, -float(decline))
        setattr(player, "rating", round(career_ovr_0_100(player), 3))
        _career_clamp_ovr_window(player, 40.0, 99.0)
        set_player_trend(player, "declining", 0, rng)
        _lifecycle_bump_decline(league)
        try:
            setattr(player, "progression_event_this_season", "major_decline")
            setattr(player, "major_progression_event_this_season", "major_decline")
        except Exception:
            pass
        o_post = career_ovr_0_100(player)
        _emit_authoritative_progression_debug(
            f"DEBUG PROGRESSION:\n  player={pname} season={sy} event=major_decline\n"
            f"  age={career_player_age(player)} pot={pot_cat} pre={o_pre:.2f} delta=-{decline:.2f} post={o_post:.2f}\n"
            f"  source=authoritative_progression_controller blocked_other_events=True"
        )
        return f"AGING DECLINE: -{round(decline, 1)} OVR"

    return None


def run_exclusive_progression_event(
    player: Any,
    rng: random.Random,
    *,
    macro: Optional[Dict[str, float]] = None,
    league: Any = None,
    season_year: Optional[int] = None,
) -> Optional[str]:
    """Backward-compatible alias for resolve_authoritative_major_progression_event."""
    return resolve_authoritative_major_progression_event(
        player, rng, macro=macro, league=league, season_year=season_year
    )


def _resolve_authoritative_breakout_amount(
    player: Any,
    rng: random.Random,
    league: Any,
    macro: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Deprecated: use run_exclusive_progression_event from lifecycle."""
    return None


def _career_breakout_try_v3(
    player: Any,
    rng: random.Random,
    league: Any,
    macro: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    return None


def _lifecycle_bump_decline(league: Any) -> None:
    if league is None:
        return
    try:
        setattr(league, "_lifecycle_used_declines", int(getattr(league, "_lifecycle_used_declines", 0)) + 1)
    except Exception:
        pass


def breakout_check(
    player: Any,
    rng: random.Random,
    *,
    macro: Optional[Dict[str, float]] = None,
    league: Any = None,
) -> Optional[str]:
    """Deprecated path: logged breakouts run only via run_exclusive_progression_event."""
    return None


def bust_check(
    player: Any,
    rng: random.Random,
    *,
    macro: Optional[Dict[str, float]] = None,
    league: Any = None,
) -> Optional[str]:
    """Deprecated: bust is resolved only inside resolve_authoritative_major_progression_event."""
    return None


def _resolve_authoritative_late_bloom_amount(
    player: Any,
    rng: random.Random,
    league: Any,
    macro: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Deprecated: use run_exclusive_progression_event from lifecycle."""
    return None


def late_bloomer_check(
    player: Any,
    rng: random.Random,
    *,
    macro: Optional[Dict[str, float]] = None,
    league: Any = None,
) -> Optional[str]:
    """Deprecated path: logged late blooms run only via run_exclusive_progression_event."""
    return None


def _aging_v3_base_decline_chance(age: int) -> float:
    if age <= 24:
        return 0.0
    if age <= 29:
        return 0.12
    if age <= 33:
        return 0.28
    if age <= 36:
        return 0.42
    return 0.58


def prime_league_season_aging_v3(league: Any, total_players: int) -> None:
    """Reset global aging budget for one season (max ~18% of roster can log a decline)."""
    if league is None:
        return
    tp = max(0, int(total_players))
    mx = int(tp * 0.18)
    if tp > 0 and mx < 1:
        mx = 1
    try:
        setattr(league, "_season_aging_events", 0)
        setattr(league, "_season_player_count", tp)
        setattr(league, "_max_aging_events", mx)
        setattr(league, "_season_aging_v3_primed", True)
    except Exception:
        pass


def prime_league_season_breakout_v3(
    league: Any, total_players: int, season_year: Optional[int] = None
) -> None:
    """Prime authoritative major-progression caps (breakouts, late blooms, bust trends)."""
    if league is None:
        return
    tp = max(0, int(total_players))
    if tp <= 0:
        mx_bo, mx_lb, mx_bust = 6, 2, 4
    else:
        mx_bo = max(6, min(10, int(round(6 + (tp / 900.0) * 4))))
        mx_lb = max(2, min(5, int(round(2 + (tp / 1000.0) * 3))))
        mx_bust = max(3, min(8, int(round(3 + (tp / 600.0) * 5))))
    mx_top_bo = max(1, min(3, max(1, mx_bo // 3)))
    try:
        setattr(league, "_season_breakout_events", 0)
        setattr(league, "_season_breakout_player_total", tp)
        setattr(league, "_max_season_breakouts", mx_bo)
        setattr(league, "_season_breakout_v3_primed", True)
        setattr(league, "_progression_controller_primed", True)
        setattr(league, "_prog_max_breakouts", mx_bo)
        setattr(league, "_prog_max_late_blooms", mx_lb)
        setattr(league, "_prog_max_busts", mx_bust)
        setattr(league, "_prog_global_breakouts_used", 0)
        setattr(league, "_prog_global_late_blooms_used", 0)
        setattr(league, "_prog_global_busts_used", 0)
        setattr(league, "_prog_top_breakouts_used", 0)
        setattr(league, "_prog_top_late_blooms_used", 0)
        setattr(league, "_prog_max_top_breakouts", mx_top_bo)
        setattr(league, "_prog_max_top_late_blooms", 1)
        if season_year is not None:
            setattr(league, "_progression_season_year", int(season_year))
    except Exception:
        pass
    if _LOG_SPECIAL_PROGRESSION_ENFORCEMENT:
        print(
            f"PROGRESSION ENFORCE: seasonal_prime year={season_year} "
            f"breakout_max={mx_bo} late_bloom_max={mx_lb} bust_max={mx_bust} "
            f"top_breakout_max={mx_top_bo} top_late_bloom_max=1 "
            f"hard_caps breakout={SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP} "
            f"late_bloom={SPECIAL_PROGRESSION_LATE_BLOOM_HARD_CAP}"
        )


def apply_league_ovr_soft_regression_if_needed(
    teams: Any,
    rng: random.Random,
    *,
    avg_trigger: float = 76.0,
) -> None:
    """
    Anti-inflation guard: if mean roster OVR is above target band, nudge high-end players down slightly.
    """
    if not teams:
        return
    ovs: List[float] = []
    hi_candidates: List[Any] = []
    for tm in teams:
        for pl in getattr(tm, "roster", None) or []:
            if getattr(pl, "retired", False):
                continue
            ratings = getattr(pl, "ratings", None)
            if not ratings or not isinstance(ratings, dict) or not ratings:
                continue
            o = float(career_ovr_0_100(pl))
            ovs.append(o)
            if o > 85.0:
                hi_candidates.append(pl)
    if not ovs:
        return
    league_avg = sum(ovs) / float(len(ovs))
    if league_avg <= avg_trigger:
        return
    excess = min(5.0, league_avg - avg_trigger)
    scale = excess / 5.0
    for pl in hi_candidates:
        lo_mag = 0.5
        hi_mag = min(1.5, 0.5 + scale * 1.0)
        if hi_mag < lo_mag:
            hi_mag = lo_mag
        delta = -float(rng.uniform(lo_mag, hi_mag))
        _career_apply_rating_delta_0_100(pl, delta)
        setattr(pl, "rating", round(career_ovr_0_100(pl), 3))
        _career_clamp_ovr_window(pl, 40.0, 99.0)


def _breakout_v3_global_cap_reached(league: Any) -> bool:
    if league is None or not getattr(league, "_season_breakout_v3_primed", False):
        return False
    mx = int(getattr(league, "_max_season_breakouts", 10**9) or 0)
    if mx <= 0:
        return False
    ev = int(getattr(league, "_season_breakout_events", 0) or 0)
    return ev >= mx


def _aging_v3_global_cap_reached(league: Any) -> bool:
    if league is None or not getattr(league, "_season_aging_v3_primed", False):
        return False
    mx = int(getattr(league, "_max_aging_events", 10**9) or 0)
    if mx <= 0:
        return False
    ev = int(getattr(league, "_season_aging_events", 0) or 0)
    return ev >= mx


def _career_aging_decline_try_v3(
    player: Any,
    rng: random.Random,
    league: Any,
    macro: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """
    Hard-enforcement aging: weighted loss tiers, clamps before return, global season cap on league.
    Returns positive OVR loss magnitude for logging as AGING DECLINE: -X.X OVR.
    """
    if getattr(player, "retired", False):
        return None
    ratings = getattr(player, "ratings", None)
    if not ratings or not isinstance(ratings, dict) or not ratings:
        return None

    assign_career_phase_from_age(player)
    age = career_player_age(player)
    ovr = career_ovr_0_100(player)

    if _aging_v3_global_cap_reached(league):
        decline_chance = 0.0
    else:
        base = _aging_v3_base_decline_chance(age)
        decline_chance = base * 0.55
        if age <= 29:
            decline_chance *= 0.4
        if ovr >= 90.0:
            decline_chance *= 0.5
        decline_chance = max(0.0, min(0.92, decline_chance))

    if decline_chance <= 0.0:
        return None
    roll = rng.random()
    if roll >= decline_chance:
        return None

    r = rng.random()
    if r < 0.70:
        decline_amount = -float(rng.uniform(0.4, 1.2))
    elif r < 0.92:
        decline_amount = -float(rng.uniform(1.2, 2.0))
    elif r < 0.985:
        decline_amount = -float(rng.uniform(2.0, 2.6))
    else:
        if age >= 35:
            decline_amount = -float(rng.uniform(2.6, 3.2))
        else:
            decline_amount = -float(rng.uniform(1.5, 2.2))

    if -2.6 < decline_amount <= -2.0:
        if rng.random() < 0.5:
            decline_amount = -float(rng.uniform(1.4, 1.9))

    if ovr >= 90.0:
        decline_amount *= 0.7

    if age < 34:
        decline_amount = max(decline_amount, -1.8)
    else:
        decline_amount = max(decline_amount, -3.2)

    dmm = float((macro or {}).get("decline_mag_mult", 1.0))
    decline_amount *= dmm
    if age < 34:
        decline_amount = max(decline_amount, -1.8)
    else:
        decline_amount = max(decline_amount, -3.2)

    loss = -float(decline_amount)
    if loss <= 1e-9:
        return None

    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(age))
    if phase == PHASE_PRIME:
        loss = min(loss, 3.0)

    if league is not None:
        try:
            setattr(
                league,
                "_season_aging_events",
                int(getattr(league, "_season_aging_events", 0) or 0) + 1,
            )
        except (TypeError, ValueError):
            pass

    return float(loss)


def regression_check(
    player: Any,
    rng: random.Random,
    *,
    macro: Optional[Dict[str, float]] = None,
    league: Any = None,
) -> Optional[str]:
    if league is not None:
        if int(getattr(league, "_lifecycle_used_declines", 0)) >= int(
            getattr(league, "_lifecycle_cap_declines", 10**9)
        ):
            return None
    decline = _career_aging_decline_try_v3(player, rng, league, macro)
    if decline is None or decline <= 0:
        return None
    _career_apply_rating_delta_0_100(player, -float(decline))
    setattr(player, "rating", round(career_ovr_0_100(player), 3))
    _career_clamp_ovr_window(player, 40.0, 99.0)
    set_player_trend(player, "declining", 0, rng)
    _lifecycle_bump_decline(league)
    return f"AGING DECLINE: -{round(decline, 1)} OVR"


def _career_last_season_points(player: Any) -> int:
    v = getattr(player, "last_season_points", None)
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    ss = getattr(player, "season_stats", None) or {}
    if not ss:
        return -1
    try:
        latest = max(ss.values(), key=lambda x: int(x.get("season", 0) or 0))
        return int(latest.get("points", 0) or 0)
    except Exception:
        return -1


def performance_growth_modifier(player: Any) -> None:
    if getattr(player, "major_progression_event_this_season", None) is not None:
        return
    pts = _career_last_season_points(player)
    if pts < 0:
        return
    assign_career_phase_from_age(player)
    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(career_player_age(player)))
    if pts > 70:
        d = 0.65
    elif pts < 20:
        d = -0.85
    else:
        return
    if phase == PHASE_PRIME:
        d *= 0.55
        d = max(-1.2, min(0.85, d))
    _career_apply_rating_delta_0_100(player, d)
    setattr(player, "rating", round(career_ovr_0_100(player), 3))
    _career_clamp_ovr_window(player, 40.0, 99.0)


def run_career_lifecycle_for_player(
    player: Any,
    rng: random.Random,
    *,
    do_print: bool = True,
    log_emit: Optional[Callable[[str], None]] = None,
    verbose_main_line: bool = True,
    league: Any = None,
    skip_base_progress: bool = False,
    season_year: Optional[int] = None,
) -> List[str]:
    """
    One offseason-style lifecycle tick: assign dev profile if missing, optional base progress,
    then resolve_authoritative_major_progression_event (single major slot: breakout / late bloom /
    bust / aging decline), then bounded performance nudge.
    Universe runner sets skip_base_progress=True when run_player_progression already ran this season.
    """
    out: List[str] = []
    if getattr(player, "retired", False):
        return out
    if not callable(getattr(player, "ovr", None)) and getattr(player, "ratings", None) is None:
        return out

    assign_development_profile(player, rng)
    assign_career_phase_from_age(player)
    tick_career_trend(player)
    if not skip_base_progress:
        progress_player(player, rng)

    macro = _lifecycle_macro_from_league(league)
    sy = season_year
    if sy is None and league is not None:
        sy = getattr(league, "season_year", None) or getattr(league, "current_season", None)

    try:
        setattr(player, "_lifecycle_ovr_before_special", float(career_ovr_0_100(player)))
    except Exception:
        pass

    major = resolve_authoritative_major_progression_event(
        player, rng, macro=macro, league=league, season_year=sy
    )
    if major:
        out.append(major)
        if do_print:
            print(major)
        if log_emit:
            log_emit(major)

    performance_growth_modifier(player)

    pname = career_player_name(player)
    age = career_player_age(player)
    dev = str(getattr(player, "dev_type", "standard"))
    rating = round(career_ovr_0_100(player), 1)
    main = f"{pname} | Age {age} | Dev: {dev} | New OVR: {rating}"
    setattr(player, "rating", float(rating))
    if verbose_main_line:
        if do_print:
            print(main)
        if log_emit:
            log_emit(main)
    out.append(main)
    return out


# --- Team system identity, coaching, era fit, roster coherence ---

TEAM_SYSTEMS: List[str] = [
    "run_and_gun",
    "defensive_lock",
    "balanced",
    "physical",
    "young_fast",
    "veteran_structured",
]


def _identity_team_label(team: Any) -> str:
    tid = str(getattr(team, "team_id", getattr(team, "id", "?")) or "?")
    city = str(getattr(team, "city", "") or "").strip()
    name = str(getattr(team, "name", "") or "").strip()
    if not name and city:
        return city
    if not city:
        return name if name else tid
    if city.lower() == name.lower():
        return city
    if name.lower().startswith(city.lower()) and len(name) > len(city):
        return name
    return f"{city} {name}".strip() or tid


def assign_team_system(team: Any, rng: random.Random) -> str:
    if getattr(team, "system", None):
        return str(team.system)
    sys = str(rng.choice(TEAM_SYSTEMS))
    setattr(team, "system", sys)
    return sys


def _normalize_era_key(era: Any) -> str:
    if era is None:
        return ""
    if hasattr(era, "value"):
        try:
            return str(era.value).lower().replace("-", "_")
        except Exception:
            pass
    return str(era).lower().replace("-", "_").replace(" ", "_")


def era_system_fit_multiplier(era: Any, system: str) -> float:
    """
    >1.0 when team system fits league era (e.g. defensive_lock in dead_puck).
    """
    e = _normalize_era_key(era)
    s = (system or "balanced").lower()
    m = 1.0
    if "dead_puck" in e:
        if s == "defensive_lock":
            m = 1.12
        elif s == "physical":
            m = 1.06
        elif s == "run_and_gun":
            m = 0.88
        elif s == "young_fast":
            m = 0.94
    elif "speed" in e or "skill" in e or "offense" in e or "run_and_gun" in e:
        if s == "run_and_gun":
            m = 1.10
        elif s == "young_fast":
            m = 1.08
        elif s == "defensive_lock":
            m = 0.94
    elif "goalie" in e:
        if s == "defensive_lock":
            m = 1.08
        elif s == "veteran_structured":
            m = 1.04
        elif s == "run_and_gun":
            m = 0.90
    elif "power_play" in e:
        if s == "run_and_gun":
            m = 1.07
        elif s == "balanced":
            m = 1.02
    elif "two_way" in e or "chess" in e:
        if s in ("balanced", "veteran_structured"):
            m = 1.06
        elif s == "run_and_gun":
            m = 0.94
    elif "expansion" in e or "dilution" in e:
        if s == "young_fast":
            m = 1.05
    else:
        if s == "balanced":
            m = 1.02
    return float(max(0.82, min(1.14, m)))


def assign_team_coach_profile(team: Any, rng: random.Random) -> Tuple[int, str]:
    rating = int(rng.randint(60, 95))
    if rating > 85:
        ct = "elite"
    elif rating > 75:
        ct = "strong"
    elif rating > 65:
        ct = "average"
    else:
        ct = "poor"
    setattr(team, "coach_rating", rating)
    setattr(team, "coach_type", ct)
    coach = getattr(team, "coach", None)
    if coach is not None:
        try:
            coach.job_security = float(clamp((rating - 55) / 90.0, 0.35, 0.92))
        except Exception:
            pass
        try:
            if ct == "elite":
                coach.development.skill_growth_multiplier = min(
                    1.18, float(coach.development.skill_growth_multiplier) + 0.08
                )
                coach.development.defensive_growth_multiplier = min(
                    1.15, float(coach.development.defensive_growth_multiplier) + 0.06
                )
            elif ct == "poor":
                coach.development.skill_growth_multiplier = max(
                    0.82, float(coach.development.skill_growth_multiplier) - 0.06
                )
        except Exception:
            pass
    return rating, ct


def coach_type_strength_multiplier(team: Any) -> float:
    cr = float(getattr(team, "coach_rating", 72) or 72)
    return float(clamp(0.94 + (cr - 60.0) / 90.0 * 0.12, 0.92, 1.08))


def team_identity_strength_multiplier(team: Any, era: Any) -> float:
    sys = str(getattr(team, "system", "balanced") or "balanced")
    era_m = era_system_fit_multiplier(era, sys)
    skew = {
        "run_and_gun": 1.03,
        "defensive_lock": 1.025,
        "balanced": 1.01,
        "physical": 1.02,
        "young_fast": 1.02,
        "veteran_structured": 1.022,
    }.get(sys, 1.0)
    cm = coach_type_strength_multiplier(team)
    return float(max(0.86, min(1.14, era_m * skew * cm)))


def team_identity_win_pct_nudge(team: Any, era: Any) -> float:
    """Small additive nudge to expected win rate (± ~0.03)."""
    mult = team_identity_strength_multiplier(team, era) - 1.0
    return float(clamp(mult * 0.55, -0.028, 0.028))


def team_scoring_pace_bias(team: Any) -> float:
    """Goals-per-game bias in abstract sim (not roster talent)."""
    sys = str(getattr(team, "system", "balanced") or "balanced")
    if sys == "run_and_gun":
        return 0.16
    if sys == "defensive_lock":
        return -0.14
    if sys == "physical":
        return -0.04
    if sys == "young_fast":
        return 0.08
    return 0.0


def team_system_development_modifier(team: Any) -> float:
    sys = str(getattr(team, "system", "") or "")
    if sys == "young_fast":
        return 0.07
    if sys == "veteran_structured":
        return -0.02
    if sys == "run_and_gun":
        return 0.02
    return 0.0


def _scale_player_keys(player: Any, keys: Sequence[str], factor: float) -> None:
    if abs(factor - 1.0) < 1e-6:
        return
    set_fn = getattr(player, "set", None)
    get_fn = getattr(player, "get", None)
    if not callable(set_fn) or not callable(get_fn):
        ratings = getattr(player, "ratings", None)
        if not ratings:
            return
        for k in keys:
            if k in ratings:
                try:
                    ratings[k] = clamp_rating(float(ratings[k]) * factor)
                except Exception:
                    pass
        return
    for k in keys:
        try:
            set_fn(k, float(get_fn(k, 50)) * factor)
        except Exception:
            pass


def apply_team_system_effects(team: Any, year_tag: Optional[int] = None) -> None:
    if year_tag is not None and getattr(team, "_identity_system_fx_year", None) == year_tag:
        return
    if year_tag is not None:
        setattr(team, "_identity_system_fx_year", int(year_tag))
    roster = list(getattr(team, "roster", None) or getattr(team, "players", None) or [])
    sys = str(getattr(team, "system", "balanced") or "balanced")
    for player in roster:
        if getattr(player, "retired", False):
            continue
        if sys == "run_and_gun":
            _scale_player_keys(player, OFFENSE_KEYS, 1.018)
            _scale_player_keys(player, DEFENSE_KEYS, 0.993)
        elif sys == "defensive_lock":
            _scale_player_keys(player, DEFENSE_KEYS, 1.02)
            _scale_player_keys(player, OFFENSE_KEYS, 0.992)
        elif sys == "balanced":
            _scale_player_keys(player, IQ_KEYS, 1.006)
        elif sys == "physical":
            _scale_player_keys(player, DEFENSE_KEYS, 1.01)
            _scale_player_keys(player, PHYS_KEYS, 1.012)
            h = getattr(player, "health", None)
            if h is not None:
                try:
                    h.injury_risk_baseline = float(clamp(float(getattr(h, "injury_risk_baseline", 0.2)) * 1.04, 0.05, 0.95))
                except Exception:
                    pass
        elif sys == "young_fast":
            if career_player_age(player) < 25:
                _scale_player_keys(player, SKATING_KEYS, 1.012)
                _scale_player_keys(player, OFFENSE_KEYS, 1.008)
        elif sys == "veteran_structured":
            if career_player_age(player) > 28:
                _scale_player_keys(player, IQ_KEYS, 1.01)
                _scale_player_keys(player, DEFENSE_KEYS, 1.008)


def apply_coach_effects(team: Any, year_tag: Optional[int] = None) -> None:
    if year_tag is not None and getattr(team, "_identity_coach_fx_year", None) == year_tag:
        return
    if year_tag is not None:
        setattr(team, "_identity_coach_fx_year", int(year_tag))
    roster = list(getattr(team, "roster", None) or getattr(team, "players", None) or [])
    ct = str(getattr(team, "coach_type", "average") or "average")
    delta_map = {"elite": 0.42, "strong": 0.22, "average": 0.0, "poor": -0.32}
    d = float(delta_map.get(ct, 0.0))
    if abs(d) < 1e-6:
        return
    for player in roster:
        if getattr(player, "retired", False):
            continue
        psych = getattr(player, "psych", None)
        if psych is None:
            continue
        if d > 0:
            psych.internal_motivation = clamp(
                float(getattr(psych, "internal_motivation", 0.5)) + 0.012 * min(1.0, d / 0.42)
            )
            psych.confidence_level = clamp(
                float(getattr(psych, "confidence_level", 0.5)) + 0.008 * min(1.0, d / 0.42)
            )
        else:
            psych.morale = clamp(float(getattr(psych, "morale", 0.5)) + 0.014 * max(-1.0, d / 0.32))
        psych.clamp_all()


def player_offense_defense_proxy(player: Any) -> Tuple[float, float]:
    ga_fn = getattr(player, "group_averages", None)
    if callable(ga_fn):
        try:
            ga = ga_fn()
            off = float(ga.get("offense", 50)) + 0.35 * float(ga.get("passing", ga.get("skating", 50)))
            df = float(ga.get("defense", 50)) + 0.25 * float(ga.get("physical", 50))
            return off, df
        except Exception:
            pass
    try:
        o = float(player.ovr()) * 100.0
    except Exception:
        o = 50.0
    return o, o


def system_fit(player: Any, team: Any) -> float:
    sys = str(getattr(team, "system", "balanced") or "balanced")
    off, deff = player_offense_defense_proxy(player)
    age = career_player_age(player)
    fit = 1.0
    if sys == "run_and_gun" and off > deff:
        fit += 0.05
    if sys == "defensive_lock" and deff > off:
        fit += 0.05
    if sys == "young_fast" and age < 25:
        fit += 0.05
    if sys == "veteran_structured" and age > 28:
        fit += 0.05
    if sys == "physical" and deff >= off * 0.98:
        fit += 0.02
    return float(min(1.12, max(0.88, fit)))


def apply_system_fit_nudges(
    team: Any,
    rng: random.Random,
    *,
    log_emit: Optional[Callable[[str], None]] = None,
    do_print: bool = True,
) -> List[str]:
    lines: List[str] = []
    roster = list(getattr(team, "roster", None) or getattr(team, "players", None) or [])
    for player in roster:
        if getattr(player, "retired", False):
            continue
        fit = system_fit(player, team)
        pname = career_player_name(player)
        psych = getattr(player, "psych", None)
        if fit >= 1.045 and rng.random() < 0.55:
            if psych is not None:
                psych.role_satisfaction = clamp(float(getattr(psych, "role_satisfaction", 0.5)) + 0.05)
                psych.internal_motivation = clamp(float(getattr(psych, "internal_motivation", 0.5)) + 0.03)
                psych.clamp_all()
            msg = f"{pname} thrives in system (+fit boost)"
            lines.append(msg)
            if do_print:
                print(msg)
            if log_emit:
                log_emit(msg)
        elif fit <= 0.93 and rng.random() < 0.40:
            if psych is not None:
                psych.morale = clamp(float(getattr(psych, "morale", 0.5)) - 0.04)
                psych.role_satisfaction = clamp(float(getattr(psych, "role_satisfaction", 0.5)) - 0.035)
                psych.clamp_all()
            msg = f"{pname} struggling in system (-fit)"
            lines.append(msg)
            if do_print:
                print(msg)
            if log_emit:
                log_emit(msg)
    return lines


def prefers_player(team: Any, player: Any) -> bool:
    sys = str(getattr(team, "system", "balanced") or "balanced")
    off, deff = player_offense_defense_proxy(player)
    age = career_player_age(player)
    if sys == "run_and_gun":
        return off > 70
    if sys == "defensive_lock":
        return deff > 70
    if sys == "young_fast":
        return age < 26
    if sys == "veteran_structured":
        return age > 27
    if sys == "physical":
        return deff >= 62
    return True


def prefers_free_agent_match(team: Any, fa_rating_0_1: float, fa_age: int = 24) -> bool:
    """Identity filter for macro FA objects (rating is 0..1)."""
    sys = str(getattr(team, "system", "balanced") or "balanced")
    off = float(fa_rating_0_1) * 100.0
    deff = off * 0.96
    if sys == "run_and_gun":
        return off > 70
    if sys == "defensive_lock":
        return deff > 70
    if sys == "young_fast":
        return fa_age < 26
    if sys == "veteran_structured":
        return fa_age > 27
    if sys == "physical":
        return deff >= 60
    return True


def evolve_team_identity(team: Any, rng: random.Random) -> Optional[str]:
    missed = int(getattr(team, "missed_playoffs_years", 0) or 0)
    old = str(getattr(team, "system", "balanced") or "balanced")
    if missed >= 3:
        if old == "young_fast":
            return None
        setattr(team, "system", "young_fast")
        return f"IDENTITY SHIFT: {_identity_team_label(team)} -> young_fast (was {old})"
    if bool(getattr(team, "is_contender", False)) and rng.random() < 0.30:
        if old == "veteran_structured":
            return None
        setattr(team, "system", "veteran_structured")
        return f"IDENTITY SHIFT: {_identity_team_label(team)} -> veteran_structured (was {old})"
    return None


def league_chaos_delta_from_team_systems(teams: Sequence[Any]) -> float:
    if not teams:
        return 0.0
    phys = sum(1 for t in teams if str(getattr(t, "system", "")) == "physical")
    vet = sum(1 for t in teams if str(getattr(t, "system", "")) == "veteran_structured")
    n = float(len(teams))
    return float(0.02 * (phys - vet) / max(1.0, n))


def runner_identity_bootstrap(teams: Sequence[Any], rng: random.Random) -> None:
    for team in teams:
        assign_team_system(team, rng)
        if getattr(team, "coach_rating", None) is None:
            assign_team_coach_profile(team, rng)


def runner_identity_annual_application(
    teams: Sequence[Any],
    rng: random.Random,
    era: Any,
    *,
    year: int = 0,
    log_emit: Optional[Callable[[str], None]] = None,
    do_print: bool = False,
) -> List[str]:
    out: List[str] = []
    for team in teams:
        assign_team_system(team, rng)
        if getattr(team, "coach_rating", None) is None:
            assign_team_coach_profile(team, rng)
        line = evolve_team_identity(team, rng)
        if line:
            out.append(line)
            if do_print:
                print(line)
            if log_emit:
                log_emit(line)
        apply_team_system_effects(team, year_tag=year if year else None)
        apply_coach_effects(team, year_tag=year if year else None)
        out.extend(
            apply_system_fit_nudges(team, rng, log_emit=log_emit, do_print=do_print)
        )
        summ = (
            f"System: {getattr(team, 'system', '?')} | Coach: {getattr(team, 'coach_type', '?')} "
            f"({getattr(team, 'coach_rating', '?')}) {_identity_team_label(team)}"
        )
        out.append(summ)
        if do_print:
            print(summ)
        if log_emit:
            log_emit(summ)
    return out


def _player_character_rating_0_100(player: Any) -> int:
    c = getattr(player, "character", None)
    if c is not None:
        try:
            ci = int(c)
            if 20 <= ci <= 90:
                return ci
        except (TypeError, ValueError):
            pass
    tr = getattr(player, "traits", None)
    psych = getattr(player, "psych", None)
    if tr is None and psych is None:
        return 50
    blend = 0.5
    if tr is not None:
        blend = (
            0.20 * float(getattr(tr, "coachability", 0.5))
            + 0.18 * float(getattr(tr, "mental_toughness", 0.5))
            + 0.16 * float(getattr(tr, "work_ethic", 0.5))
            + 0.14 * float(getattr(tr, "leadership", 0.5))
            + 0.12 * float(getattr(tr, "competitiveness", 0.5))
            + 0.10 * (1.0 - float(getattr(tr, "confront_willingness", 0.5)))
            + 0.10 * (1.0 - float(getattr(tr, "volatility", 0.5)))
        )
    if psych is not None:
        blend = 0.72 * blend + 0.28 * (
            0.5 * (1.0 - float(getattr(psych, "tilt_susceptibility", 0.5)))
            + 0.5 * float(getattr(psych, "conflict_resolution", 0.5))
        )
    return int(round(clamp(float(blend), 0.0, 1.0) * 100.0))


def get_storyline_polarity_weights(character: int) -> Dict[str, float]:
    base = {"positive": 1.0, "neutral": 1.0, "negative": 1.0}
    ch = int(character)
    if ch >= 75:
        base["positive"] *= 2.0
        base["negative"] *= 0.4
    elif ch >= 60:
        base["positive"] *= 1.5
        base["negative"] *= 0.7
    elif ch < 30:
        base["negative"] *= 3.0
        base["positive"] *= 0.3
    elif ch < 45:
        base["negative"] *= 1.8
        base["positive"] *= 0.7
    return base


def classify_storyline_polarity(d: Dict[str, Any]) -> str:
    if d.get("legal"):
        return "negative"
    pool = str(d.get("pool", "") or "")
    txt = (d.get("text") or "").lower()
    fx = d.get("fx") or {}
    net = sum(float(v) for v in fx.values()) if isinstance(fx, dict) else 0.0
    if pool == "legal_crime":
        return "negative"
    for kw in STORYLINE_POLARITY_POSITIVE_KEYWORDS:
        if kw in txt:
            return "positive"
    for kw in STORYLINE_POLARITY_NEGATIVE_KEYWORDS:
        if kw in txt:
            return "negative"
    if pool in ("money_career", "team_dynamics") and net > 0.045:
        return "positive"
    if pool == "mental_psychological" and net > 0.04:
        return "positive"
    if net > 0.035:
        return "positive"
    if net < -0.032:
        return "negative"
    return "neutral"


def character_storyline_effect_multiplier(character: int) -> float:
    ch = int(character)
    if ch < 30:
        return 1.5
    if ch > 75:
        return 0.7
    return 1.0


def synthetic_extreme_low_character_storyline(rng: random.Random) -> Dict[str, Any]:
    headlines = [
        "Arrested for off-ice incident (conduct investigation)",
        "Major locker room divide linked to player behavior",
        "Public refusal of coach system — discipline meeting",
        "Team suspension for conduct violation",
        "Media scandal leaks private details",
    ]
    return {
        "id": f"extreme_low_char_{rng.randint(1, 9_999_999)}",
        "pool": "legal_crime",
        "text": rng.choice(headlines),
        "fx": {
            "confidence": -0.24,
            "morale": -0.30,
            "clutch": -0.18,
            "media_stress": 0.20,
            "chemistry": -0.12,
        },
        "dur": "medium",
        "legal": True,
        "char_max": 100,
        "volatile": True,
        "polarity": "negative",
        "tier": "major",
    }


def _player_ovr01(player: Any) -> float:
    try:
        fn = getattr(player, "ovr", None)
        return float(fn()) if callable(fn) else float(fn or 0.5)
    except Exception:
        return 0.5


def _player_lifecycle_tag(player: Any, ovr: float, age: int) -> str:
    if age <= 23 and ovr < 0.74:
        return "rookie"
    if ovr >= 0.78 or (age <= 26 and ovr >= 0.73):
        return "star"
    if age >= 31:
        return "veteran"
    return "regular"


def _dur_seasons_band(dur: str, rng: random.Random) -> float:
    if dur == "short":
        return rng.uniform(0.20, 0.40)
    if dur == "long":
        return 1.0
    return rng.uniform(0.45, 0.88)


def _legal_pool_weight_mult(char: int) -> float:
    if char >= 78:
        return 0.015
    if char >= 60:
        return 0.06
    if char >= 50:
        return 0.10
    if char >= 30:
        return 0.55
    return 1.0


def _pool_weights_for_character(char: int) -> Dict[str, float]:
    if char < 40:
        return {
            "legal_crime": 0.26,
            "chaotic_weird": 0.14,
            "mental_psychological": 0.18,
            "personal_life": 0.12,
            "media_pressure": 0.12,
            "team_dynamics": 0.10,
            "money_career": 0.08,
        }
    if char < 70:
        return {
            "media_pressure": 0.22,
            "team_dynamics": 0.20,
            "money_career": 0.18,
            "personal_life": 0.14,
            "mental_psychological": 0.14,
            "chaotic_weird": 0.07,
            "legal_crime": 0.05,
        }
    return {
        "team_dynamics": 0.32,
        "money_career": 0.16,
        "mental_psychological": 0.08,
        "personal_life": 0.08,
        "media_pressure": 0.12,
        "chaotic_weird": 0.03,
        "legal_crime": 0.01,
    }


_STORYLINE_POOL_TIER: Dict[str, str] = {
    "legal_crime": "major",
    "mental_psychological": "mid",
    "personal_life": "mid",
    "media_pressure": "mid",
    "team_dynamics": "mid",
    "money_career": "mid",
    "chaotic_weird": "minor",
}

_STORYLINE_OVERUSED_SUBSTRINGS: Tuple[str, ...] = (
    "documentary crew",
    "reality tv",
    "bridge deal stalemate",
    "exploded onto",
    "netflix",
    "mascot feud",
)


def _storyline_tier_for_def(d: Dict[str, Any]) -> str:
    t = d.get("tier")
    if t:
        return str(t).lower()
    return str(_STORYLINE_POOL_TIER.get(str(d.get("pool", "") or ""), "mid"))


def _storyline_template_stem(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"\s*\(arc thread \d+\)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\(beat \d+\)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\(cycle \d+\)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\(thread \d+\)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\(variant \d+\)\s*$", "", s, flags=re.I)
    return s[:120] if s else "?"


def _storyline_overused_template_penalty(text: str) -> float:
    t = str(text or "").lower()
    p = 1.0
    for s in _STORYLINE_OVERUSED_SUBSTRINGS:
        if s in t:
            p *= 0.26
    return p


def _storyline_context_fit_weight(d: Dict[str, Any], tag: str, char: int, age: int, perf_delta: float, ovr: float) -> float:
    tx = str(d.get("text") or "").lower()
    w = 1.0
    if any(k in tx for k in ("leader", "captaincy", "mentor", "leadership group")):
        if tag != "veteran" and age < 27:
            w *= 0.22
        if char < 56:
            w *= 0.30
    if any(k in tx for k in ("collapse", "panic", "crisis", "benching spiral", "burnout")):
        if perf_delta > 0.015:
            w *= 0.18
        if char > 82:
            w *= 0.20
    if any(k in tx for k in ("surge", "endorsement rush", "contract year surge")):
        if perf_delta < -0.025:
            w *= 0.22
        if tag == "rookie" and ovr < 0.66:
            w *= 0.35
    if d.get("legal") or "arrest" in tx or "scandal" in tx:
        if char >= 55:
            w *= 0.12
    return w


def _storyline_fx_apply(player: Any, fx: Dict[str, float], scale: float = 1.0) -> None:
    if not fx or scale == 0.0:
        return
    traits = getattr(player, "traits", None)
    psych = getattr(player, "psych", None)
    career = getattr(player, "career", None)
    for k, v in fx.items():
        dv = float(v) * scale
        if abs(dv) < 1e-9:
            continue
        if k == "confidence":
            if traits is not None:
                traits.confidence = clamp(float(getattr(traits, "confidence", 0.5)) + dv * 0.62)
            if psych is not None:
                psych.confidence_level = clamp(float(getattr(psych, "confidence_level", 0.5)) + dv * 0.55)
        elif k == "morale" and psych is not None:
            psych.morale = clamp(float(getattr(psych, "morale", 0.5)) + dv)
        elif k == "clutch" and traits is not None:
            traits.clutch_tendency = clamp(float(getattr(traits, "clutch_tendency", 0.5)) + dv)
        elif k == "leadership" and traits is not None:
            traits.leadership = clamp(float(getattr(traits, "leadership", 0.5)) + dv)
        elif k == "mental_toughness" and traits is not None:
            traits.mental_toughness = clamp(float(getattr(traits, "mental_toughness", 0.5)) + dv)
        elif k == "media_comfort" and traits is not None:
            traits.media_comfort = clamp(float(getattr(traits, "media_comfort", 0.5)) + dv)
        elif k == "media_stress" and psych is not None:
            psych.media_stress = clamp(float(getattr(psych, "media_stress", 0.5)) + dv)
        elif k == "internal_motivation" and psych is not None:
            psych.internal_motivation = clamp(float(getattr(psych, "internal_motivation", 0.5)) + dv)
        elif k == "chemistry" and psych is not None:
            psych.chemistry_contribution = clamp(float(getattr(psych, "chemistry_contribution", 0.5)) + dv)
        elif k == "consistency" and career is not None:
            career.season_consistency = clamp(float(getattr(career, "season_consistency", 0.5)) + dv)
            if psych is not None:
                psych.consistency_dampener = clamp(float(getattr(psych, "consistency_dampener", 0.5)) - dv * 0.35)
        elif k == "decision" and psych is not None:
            psych.decision_fatigue_spillover = clamp(float(getattr(psych, "decision_fatigue_spillover", 0.5)) + dv)
        elif k == "anxiety" and psych is not None:
            psych.anxiety_level = clamp(float(getattr(psych, "anxiety_level", 0.5)) + dv)
        elif k == "contract_pressure" and psych is not None:
            psych.contract_pressure = clamp(float(getattr(psych, "contract_pressure", 0.5)) + dv)
        elif k == "performance" and career is not None and psych is not None:
            career.season_consistency = clamp(float(getattr(career, "season_consistency", 0.5)) + dv * 0.55)
            psych.internal_motivation = clamp(float(getattr(psych, "internal_motivation", 0.5)) + dv * 0.45)
    if traits is not None:
        traits.clamp_all()
    if psych is not None:
        psych.clamp_all()
    if career is not None:
        career.clamp_all()


def _classify_systemic_event_from_storyline(picked: Dict[str, Any], fx: Dict[str, float]) -> Tuple[str, float]:
    pool = str(picked.get("pool", "") or "")
    legal = bool(picked.get("legal"))
    net = sum(float(v) for v in (fx or {}).values())
    txt = (picked.get("text") or "").lower()
    if legal or pool == "legal_crime":
        return "legal_trouble", 1.12
    if pool == "media_pressure" and net < -0.02:
        return "scandal", 1.0
    if pool == "team_dynamics":
        if net > 0.06 or "leader" in txt or "mentor" in txt or "rallies" in txt:
            return "leader_emergence", 1.0
        if net < -0.04:
            return "locker_room_issue", 1.05
        return "team_conflict", 0.9
    if pool == "mental_psychological" and net < -0.06:
        return "mental_collapse", 1.08
    if pool == "mental_psychological" and net > 0.05:
        return "confidence_surge", 0.95
    if pool == "money_career" and net > 0.07:
        return "breakout", 1.0
    if pool == "money_career" and net < -0.04:
        return "team_conflict", 0.95
    if "clutch" in txt:
        return "clutch_run", 0.95
    if net > 0.05:
        return "emergence", 0.9
    if net < -0.05:
        return "scandal", 0.85
    return "generic", 0.75


def _systemic_default_player_effects(event_type: str, severity: float) -> Dict[str, float]:
    s = float(severity)
    if event_type == "legal_trouble":
        return {"media_stress": 0.032 * s, "morale": -0.022 * s}
    if event_type == "scandal":
        return {"media_stress": 0.028 * s, "morale": -0.015 * s}
    if event_type == "locker_room_issue":
        return {"chemistry": -0.022 * s, "morale": -0.018 * s}
    if event_type == "team_conflict":
        return {"morale": -0.018 * s, "internal_motivation": -0.012 * s}
    if event_type == "breakout":
        return {"internal_motivation": 0.032 * s, "performance": 0.022 * s}
    if event_type == "clutch_run":
        return {"clutch": 0.028 * s, "confidence": 0.018 * s}
    if event_type == "leader_emergence":
        return {"leadership": 0.024 * s, "chemistry": 0.015 * s}
    if event_type == "mental_collapse":
        return {"anxiety": 0.038 * s, "decision": 0.028 * s}
    if event_type == "confidence_surge":
        return {"confidence": 0.032 * s, "morale": 0.018 * s}
    if event_type == "emergence":
        return {"performance": 0.022 * s, "confidence": 0.015 * s}
    return {}


def apply_systemic_consequences(
    player: Any,
    team: Any,
    league_state: Optional[Dict[str, float]],
    event: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ripple effects from a storyline/event across player (extra nudges), team locker room,
    trade-value signal, role satisfaction, and league chaos/parity deltas.
    """
    out: Dict[str, Any] = {
        "event_type": "generic",
        "player_line": "",
        "team_line": "",
        "league_line": "",
        "trade_line": "",
        "role_line": "",
    }
    if not event:
        return out

    event_type = str(event.get("type", "generic") or "generic")
    severity = float(event.get("severity", 1.0) or 1.0)
    out["event_type"] = event_type

    extra = event.get("effects") or {}
    merged: Dict[str, float] = dict(_systemic_default_player_effects(event_type, severity))
    if isinstance(extra, dict):
        for k, v in extra.items():
            merged[str(k)] = merged.get(str(k), 0.0) + float(v) * severity

    if merged:
        _storyline_fx_apply(player, merged, scale=1.0)
        out["player_line"] = _storyline_effect_summary_pct(merged)

    psych = getattr(player, "psych", None)
    if psych is not None:
        if event_type in ("mental_collapse", "legal_trouble"):
            psych.role_satisfaction = clamp(float(getattr(psych, "role_satisfaction", 0.5)) - 0.10 * severity)
            out["role_line"] = f"role satisfaction {-10 * severity:.0f}%"
        elif event_type in ("leader_emergence", "confidence_surge", "mentor_boost"):
            psych.role_satisfaction = clamp(float(getattr(psych, "role_satisfaction", 0.5)) + 0.08 * severity)
            out["role_line"] = f"role satisfaction {+8 * severity:.0f}%"
        psych.clamp_all()

    m = float(getattr(player, "_systemic_trade_value_mult", 1.0) or 1.0)
    if event_type in ("legal_trouble", "team_conflict"):
        m *= 0.85 ** min(1.2, max(0.5, severity))
        out["trade_line"] = "trade signal x0.85"
    elif event_type in ("breakout", "clutch_run", "emergence"):
        m *= 1.10 ** min(1.15, max(0.5, severity))
        out["trade_line"] = "trade signal x1.10"
    setattr(player, "_systemic_trade_value_mult", clamp(m, 0.45, 1.55))

    st = getattr(team, "state", None)
    team_bits: List[str] = []
    if st is not None:
        if event_type in ("legal_trouble", "scandal", "locker_room_issue"):
            st.team_morale = clamp(float(getattr(st, "team_morale", 0.5)) - 0.05 * severity)
            st.organizational_pressure = clamp(float(getattr(st, "organizational_pressure", 0.5)) + 0.035 * severity)
            team_bits.append(f"morale {-0.05 * severity:.3f}")
            team_bits.append(f"org pressure {+0.035 * severity:.3f}")
        if event_type in ("breakout", "clutch_run", "emergence", "leader_emergence"):
            st.team_morale = clamp(float(getattr(st, "team_morale", 0.5)) + 0.04 * severity)
            team_bits.append(f"morale {+0.04 * severity:.3f}")
        if hasattr(st, "clamp"):
            st.clamp()

    mom = float(getattr(team, "momentum_score", 0.5) or 0.5)
    if event_type in ("legal_trouble", "scandal", "locker_room_issue", "team_conflict"):
        mom = clamp(mom - 0.04 * severity)
        team_bits.append(f"momentum {-0.04 * severity:.3f}")
    if event_type in ("breakout", "clutch_run", "emergence", "leader_emergence"):
        mom = clamp(mom + 0.03 * severity)
        team_bits.append(f"momentum {+0.03 * severity:.3f}")
    setattr(team, "momentum_score", mom)

    if league_state is not None:
        if event_type in ("scandal", "legal_trouble"):
            league_state["chaos_index"] = float(league_state.get("chaos_index", 0.0)) + 0.01 * severity
            out["league_line"] = f"chaos {+0.01 * severity:.3f}"
        if event_type in ("breakout", "emergence"):
            league_state["parity_index"] = float(league_state.get("parity_index", 0.0)) + 0.005 * severity
            if out["league_line"]:
                out["league_line"] += f"  parity {+0.005 * severity:.3f}"
            else:
                out["league_line"] = f"parity {+0.005 * severity:.3f}"

    if team_bits:
        out["team_line"] = "; ".join(team_bits)

    traits = getattr(player, "traits", None)
    if traits is not None:
        traits.clamp_all()
    if psych is not None:
        psych.clamp_all()

    return out


def apply_team_ripple(team: Any, source_player: Any, event: Optional[Dict[str, Any]]) -> int:
    if not event or team is None:
        return 0
    event_type = str(event.get("type", "generic") or "generic")
    severity = float(event.get("severity", 1.0) or 1.0)
    n = 0
    for teammate in getattr(team, "roster", None) or []:
        if teammate is source_player or getattr(teammate, "retired", False):
            continue
        psych = getattr(teammate, "psych", None)
        if psych is None:
            continue
        if event_type in ("scandal", "locker_room_issue", "legal_trouble", "team_conflict"):
            psych.morale = clamp(float(getattr(psych, "morale", 0.5)) - 0.02 * severity)
            n += 1
        elif event_type in ("leader_emergence",):
            psych.morale = clamp(float(getattr(psych, "morale", 0.5)) + 0.015 * severity)
            n += 1
        psych.clamp_all()
    return n


def _normalize_systemic_after_consequences(player: Any, team: Any) -> None:
    traits = getattr(player, "traits", None)
    psych = getattr(player, "psych", None)
    if traits is not None:
        traits.clamp_all()
    if psych is not None:
        psych.clamp_all()
    tm = float(getattr(player, "_systemic_trade_value_mult", 1.0) or 1.0)
    setattr(player, "_systemic_trade_value_mult", clamp(tm, 0.45, 1.55))
    st = getattr(team, "state", None)
    if st is not None and hasattr(st, "clamp"):
        st.clamp()
    ms = float(getattr(team, "momentum_score", 0.5) or 0.5)
    setattr(team, "momentum_score", clamp(ms, 0.0, 1.0))


# =====================================================================
# LINE CHEMISTRY (forward lines + D pairs; yearly snapshot / restore)
# =====================================================================

LINE_CHEM_OFFENSE_KEYS: List[str] = [
    "of_wrist_accuracy",
    "of_snap_accuracy",
    "of_scoring_instinct",
    "of_net_front_finishing",
]
LINE_CHEM_PASS_KEYS: List[str] = [
    "ps_vision",
    "ps_short_accuracy",
    "ps_play_anticipation",
    "ps_cross_ice",
]
LINE_CHEM_IQ_KEYS: List[str] = [
    "iq_situational_awareness",
    "iq_pattern_recognition",
    "iq_risk_assessment",
]
LINE_CHEM_ALL_KEYS: List[str] = LINE_CHEM_OFFENSE_KEYS + LINE_CHEM_PASS_KEYS + LINE_CHEM_IQ_KEYS
LINE_CHEM_ELITE_PASS_KEYS: List[str] = ["ps_vision", "ps_cross_ice", "ps_play_anticipation", "ps_long_accuracy"]
LINE_CHEM_ELITE_SHOT_KEYS: List[str] = [
    "of_wrist_accuracy",
    "of_snap_accuracy",
    "of_scoring_instinct",
    "of_release_speed",
]


def _player_position_label(p: Any) -> str:
    try:
        pos = getattr(p, "position", None)
        if hasattr(pos, "value"):
            return str(pos.value).upper()
    except Exception:
        pass
    ident = getattr(p, "identity", None)
    if ident is not None:
        pv = getattr(ident, "position", None)
        if hasattr(pv, "value"):
            return str(pv.value).upper()
    return str(getattr(p, "position", "") or "").upper()


def _avg_keys_01(player: Any, keys: List[str]) -> float:
    r = getattr(player, "ratings", None) or {}
    if not keys:
        return 0.5
    s = 0.0
    n = 0
    for k in keys:
        s += float(r.get(k, 50)) / 99.0
        n += 1
    return s / n if n else 0.5


def _player_playstyle_seed_u01(player: Any) -> float:
    ident = getattr(player, "identity", None)
    nm = str(getattr(ident, "name", None) or getattr(player, "name", None) or id(player))
    h = abs(hash(nm)) % 10_007
    return (h % 1000) / 1000.0


def _weighted_pick_forward_style(rng: random.Random, oa: float, pa: float, da: float, ph: float) -> str:
    w = {
        "sniper": 0.18,
        "playmaker": 0.23,
        "power_forward": 0.17,
        "grinder": 0.13,
        "two_way": 0.29,
    }
    w["sniper"] *= 1.0 + max(0.0, oa - 0.52) * 2.1
    w["playmaker"] *= 1.0 + max(0.0, pa - 0.52) * 2.0
    w["power_forward"] *= 1.0 + max(0.0, ph - 0.52) * 1.35 + max(0.0, oa - 0.48) * 0.6
    w["grinder"] *= 1.0 + max(0.0, ph - 0.54) * 1.5
    w["two_way"] *= 1.0 + max(0.0, da - 0.52) * 1.1
    w["two_way"] = max(0.06, w["two_way"] * 0.72)
    tot = sum(w.values())
    roll = rng.random() * tot
    acc = 0.0
    for k, v in w.items():
        acc += v
        if roll <= acc:
            return k
    return "two_way"


def _weighted_pick_defense_style(rng: random.Random, oa: float, da: float, ph: float) -> str:
    if ph > 0.58 and oa < 0.53:
        return "enforcer_d" if rng.random() < 0.62 else "defensive_d"
    if oa > da + 0.055:
        return "offensive_d"
    if da > oa + 0.055:
        return "defensive_d"
    w = {"offensive_d": 0.24, "defensive_d": 0.24, "two_way_d": 0.42, "enforcer_d": 0.10}
    roll = rng.random()
    acc = 0.0
    for k, v in w.items():
        acc += v
        if roll <= acc:
            return k
    return "two_way_d"


def ensure_player_playstyle(player: Any) -> str:
    existing = getattr(player, "playstyle", None)
    if existing is not None and str(existing).strip().lower() in ("goalie", "g"):
        setattr(player, "playstyle", None)
        existing = None
    if str(existing or "").strip().lower() == "two_way" and _player_playstyle_seed_u01(player) < 0.48:
        setattr(player, "playstyle", None)
        existing = None
    if existing:
        exs = str(existing).strip().lower()
        if exs and exs not in ("", "none", "generic"):
            return str(existing)
    pos_l = _player_position_label(player)
    seed = _player_playstyle_seed_u01(player)
    rng = random.Random(int(seed * 1_000_000) ^ id(player) % 2**30)
    if pos_l == "G":
        ps = rng.choices(["hybrid", "butterfly", "aggressive"], weights=[0.42, 0.36, 0.22], k=1)[0]
    elif pos_l == "D":
        oa = _avg_keys_01(player, OFFENSE_KEYS)
        da = _avg_keys_01(player, DEFENSE_KEYS)
        ph = _avg_keys_01(player, PHYS_KEYS)
        ps = _weighted_pick_defense_style(rng, oa, da, ph)
    else:
        oa = _avg_keys_01(player, OFFENSE_KEYS)
        pa = _avg_keys_01(player, PASSING_KEYS)
        da = _avg_keys_01(player, DEFENSE_KEYS)
        ph = _avg_keys_01(player, PHYS_KEYS)
        ps = _weighted_pick_forward_style(rng, oa, pa, da, ph)
    setattr(player, "playstyle", ps)
    return ps


def _system_archetype_line_bonus(team: Any, styles: List[str]) -> float:
    if not team or not styles:
        return 0.0
    sys = str(getattr(team, "system", "balanced") or "balanced").lower()
    b = 0.0
    if sys == "run_and_gun":
        b += 0.042 * sum(1 for s in styles if s == "sniper")
        b += 0.038 * sum(1 for s in styles if s == "playmaker")
    elif sys == "defensive_lock":
        b += 0.04 * sum(1 for s in styles if s in ("two_way", "two_way_d", "defensive_d", "grinder"))
    elif sys == "physical":
        b += 0.045 * sum(1 for s in styles if s in ("power_forward", "grinder", "enforcer_d"))
    elif sys == "young_fast":
        b += 0.035 * sum(1 for s in styles if s in ("playmaker", "sniper"))
        b += 0.022 * sum(1 for s in styles if s in ("two_way",))
    return min(0.09, b)


def calculate_line_chemistry(line: List[Any], team: Any = None) -> float:
    """
    Synergy-based chemistry for forward trios or D pairs (0–1).
    """
    if not line:
        return 0.5
    for p in line:
        ensure_player_playstyle(p)
    styles = [str(getattr(p, "playstyle", "two_way") or "two_way").lower() for p in line]
    pos = [_player_position_label(p) for p in line]
    base = 0.52

    if len(line) >= 2 and all(x == "D" for x in pos):
        if styles.count("offensive_d") >= 2:
            base = 0.38
        elif styles.count("defensive_d") >= 2:
            base = 0.46
        elif "offensive_d" in styles and "defensive_d" in styles:
            base = 0.76
        elif "two_way_d" in styles:
            base = max(base, 0.68)
        elif "enforcer_d" in styles and "offensive_d" in styles:
            base = 0.58
        else:
            base = 0.55
        base += _system_archetype_line_bonus(team, styles)
        v = 0.035 * math.sin(float(sum(ord(c) for s in styles for c in s[:6])) * 0.08)
        return clamp(base + v, 0.28, 0.92)

    if len(line) < 2:
        return clamp(0.5 + _system_archetype_line_bonus(team, styles), 0.35, 0.78)

    st_set = set(styles)
    sn = styles.count("sniper")
    pm = styles.count("playmaker")
    pf = styles.count("power_forward")
    gr = styles.count("grinder")
    tw = styles.count("two_way")

    if pm >= 1 and sn >= 1 and pf >= 1:
        base = 0.74 + 0.12 * min(1.0, pm + sn + pf - 2)
    elif pm >= 1 and sn >= 2:
        base = 0.72
    elif pf >= 1 and sn >= 1 and gr >= 1:
        base = 0.71
    elif pm >= 1 and sn >= 1:
        base = 0.64
    elif sn >= 3 or gr >= 3 or pm >= 3:
        base = 0.44
    elif sn >= 2 and pm == 0:
        base = 0.42
    elif pm >= 2 and sn == 0:
        base = 0.45
    elif tw >= 2 and sn == 0 and pm == 0:
        base = 0.56
    else:
        base = 0.58
        if len(st_set) >= 3:
            base += 0.06
        if sn >= 1 and pm >= 1:
            base += 0.05

    puck_carriers = pm + sn
    if puck_carriers >= 3 and sn < 2:
        base -= 0.12
    if sn >= 2 and pm == 0 and pf == 0:
        base -= 0.08

    for i, p1 in enumerate(line):
        for p2 in line[i + 1 :]:
            pa = _avg_keys_01(p1, LINE_CHEM_ELITE_PASS_KEYS)
            sa = _avg_keys_01(p2, LINE_CHEM_ELITE_SHOT_KEYS)
            sa2 = _avg_keys_01(p2, LINE_CHEM_ELITE_PASS_KEYS)
            pa2 = _avg_keys_01(p1, LINE_CHEM_ELITE_SHOT_KEYS)
            if (pa > 0.78 and sa > 0.76) or (pa2 > 0.78 and sa2 > 0.76):
                base += 0.07
                break
        else:
            continue
        break

    base += _system_archetype_line_bonus(team, styles)
    v = 0.04 * math.sin(float(sum(ord(c) for s in styles for c in s[:8])) * 0.07)
    return clamp(base + v, 0.26, 0.91)


def apply_line_chemistry_effects(line: List[Any], chemistry: float) -> None:
    off_m = 1.0 + (chemistry - 0.5) * 0.62
    pas_m = 1.0 + (chemistry - 0.5) * 0.48
    iqm = 1.0 + (chemistry - 0.5) * 0.36
    to_m = 1.0 - (chemistry - 0.5) * 0.38
    if chemistry < 0.42:
        off_m *= 0.94
        pas_m *= 0.93
        to_m *= 0.88
    if chemistry < 0.34:
        off_m *= 0.92
        psych_hit = 0.012 * (0.36 - chemistry)
        for player in line:
            psych = getattr(player, "psych", None)
            if psych is not None and hasattr(psych, "morale"):
                try:
                    cur = float(getattr(psych, "morale", 0.5) or 0.5)
                    setattr(psych, "morale", max(0.12, cur - psych_hit))
                except Exception:
                    pass
    for player in line:
        r = getattr(player, "ratings", None)
        if not isinstance(r, dict):
            continue
        snap = getattr(player, "_line_chem_snapshot", None) or {}
        for k in LINE_CHEM_OFFENSE_KEYS:
            if k not in r:
                continue
            b = float(snap.get(k, r.get(k, 50)))
            r[k] = clamp_rating(b * off_m)
        for k in LINE_CHEM_PASS_KEYS:
            if k not in r:
                continue
            b = float(snap.get(k, r.get(k, 50)))
            r[k] = clamp_rating(b * pas_m * to_m)
        for k in LINE_CHEM_IQ_KEYS:
            if k not in r:
                continue
            b = float(snap.get(k, r.get(k, 50)))
            r[k] = clamp_rating(b * iqm)


def _line_chemistry_effect_label(chemistry: float) -> str:
    if chemistry >= 0.78:
        return "elite offensive synergy"
    if chemistry >= 0.70:
        return "high synergy (scoring / assists / momentum-friendly)"
    if chemistry >= 0.58:
        return "solid complementary mix"
    if chemistry >= 0.48:
        return "average chemistry"
    if chemistry >= 0.40:
        return "stale / turnover-prone mix"
    return "poor fit — role conflict"


def _runner_arch_for_team(league: Any, team: Any) -> str:
    d = getattr(league, "_runner_team_archetypes", None) or {}
    tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
    return str(d.get(tid, "balanced")).lower()


def _optimize_defense_pair_sequence(ds: List[Any]) -> List[Any]:
    if len(ds) < 2:
        return ds
    for p in ds:
        ensure_player_playstyle(p)
    offs = [p for p in ds if str(getattr(p, "playstyle", "")).lower() == "offensive_d"]
    deff = [p for p in ds if str(getattr(p, "playstyle", "")).lower() == "defensive_d"]
    tw = [p for p in ds if str(getattr(p, "playstyle", "")).lower() == "two_way_d"]
    enf = [p for p in ds if str(getattr(p, "playstyle", "")).lower() == "enforcer_d"]
    out: List[Any] = []
    while offs and deff:
        out.append(offs.pop(0))
        out.append(deff.pop(0))
    rest = offs + deff + tw + enf
    rest.sort(key=_player_ovr01, reverse=True)
    out.extend(rest)
    seen = {id(x) for x in out}
    for p in ds:
        if id(p) not in seen:
            out.append(p)
            seen.add(id(p))
    return out


def _best_forward_triplet(team: Any, pool: List[Any], league: Any) -> List[Any]:
    if len(pool) <= 3:
        return list(pool)
    top = sorted(pool, key=_player_ovr01, reverse=True)[: min(11, len(pool))]
    n = len(top)
    best: Optional[List[Any]] = None
    best_score = -1e9
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri = [top[i], top[j], top[k]]
                chem = calculate_line_chemistry(tri, team)
                ovr = _player_ovr01(tri[0]) + _player_ovr01(tri[1]) + _player_ovr01(tri[2])
                score = chem * 1.22 + ovr * 0.11
                if score > best_score:
                    best_score = score
                    best = tri
    return list(best) if best else top[:3]


def _optimize_forward_line_assignments(team: Any, league: Any, rng: random.Random) -> None:
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    fw = [p for p in roster if _player_position_label(p) in ("C", "LW", "RW")]
    if not fw:
        return
    for p in fw:
        ensure_player_playstyle(p)
        ctx = getattr(p, "context", None)
        if ctx is not None:
            try:
                ctx.line_assignment = None
            except Exception:
                pass
    arch = _runner_arch_for_team(league, team)
    pool = list(fw)
    pool.sort(key=_player_ovr01, reverse=True)
    for label in ("L1", "L2", "L3", "L4"):
        if len(pool) < 2:
            break
        messy_p = 0.0
        if arch == "rebuild":
            messy_p = 0.58
        elif arch == "win_now":
            messy_p = 0.07
        elif arch == "contender":
            messy_p = 0.11
        elif arch in ("balanced", "draft_and_develop"):
            messy_p = 0.26
        elif arch == "chaos_agent":
            messy_p = 0.34
        if messy_p > 0 and rng.random() < messy_p:
            line = pool[: min(3, len(pool))]
            pool = pool[len(line) :]
        else:
            if len(pool) < 3:
                line = list(pool)
                pool = []
            else:
                line = _best_forward_triplet(team, pool, league)
                for pl in line:
                    pool.remove(pl)
            if arch == "chaos_agent" and rng.random() < 0.20:
                rng.shuffle(line)
        for pl in line:
            ctx = getattr(pl, "context", None)
            if ctx is not None:
                try:
                    ctx.line_assignment = label
                except Exception:
                    pass
    extra = 5
    while len(pool) >= 2 and extra <= 8:
        label = f"L{extra}"
        line = pool[: min(3, len(pool))]
        pool = pool[len(line) :]
        for pl in line:
            ctx = getattr(pl, "context", None)
            if ctx is not None:
                try:
                    ctx.line_assignment = label
                except Exception:
                    pass
        extra += 1


def _iter_team_forward_lines(team: Any) -> List[Tuple[str, List[Any]]]:
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    fw: List[Any] = []
    for p in roster:
        pl = _player_position_label(p)
        if pl in ("C", "LW", "RW"):
            fw.append(p)
    if not fw:
        return []
    by_slot: Dict[str, List[Any]] = {"L1": [], "L2": [], "L3": [], "L4": []}
    loose: List[Any] = []
    for p in fw:
        ctx = getattr(p, "context", None)
        la = getattr(ctx, "line_assignment", None) if ctx is not None else None
        if la in by_slot:
            by_slot[str(la)].append(p)
        else:
            loose.append(p)
    loose.sort(key=lambda x: _player_ovr01(x), reverse=True)
    out: List[Tuple[str, List[Any]]] = []
    for label in ("L1", "L2", "L3", "L4"):
        grp = list(by_slot[label])
        if len(grp) < 3 and loose:
            need = 3 - len(grp)
            grp.extend(loose[:need])
            loose = loose[need:]
        if len(grp) >= 2:
            grp.sort(key=lambda x: _player_ovr01(x), reverse=True)
            out.append((label, grp[:3]))
    while len(loose) >= 2:
        chunk = loose[:3]
        loose = loose[3:]
        idx = len(out) + 1
        out.append((f"L{idx}", chunk))
    return out


def _iter_team_defense_pairs(team: Any) -> List[Tuple[str, List[Any]]]:
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    ds = [p for p in roster if _player_position_label(p) == "D"]
    if len(ds) < 2:
        return []
    ds = _optimize_defense_pair_sequence(ds)
    out: List[Tuple[str, List[Any]]] = []
    for i in range(0, len(ds) - 1, 2):
        pair = ds[i : i + 2]
        if len(pair) == 2:
            out.append((f"D{i // 2 + 1}", pair))
    return out


def _capture_line_chemistry_snapshots(players: List[Any]) -> None:
    for p in players:
        r = getattr(p, "ratings", None)
        if not isinstance(r, dict):
            continue
        snap: Dict[str, int] = dict(getattr(p, "_line_chem_snapshot", None) or {})
        for k in LINE_CHEM_ALL_KEYS:
            if k in r:
                snap[k] = int(round(float(r.get(k, 50))))
        setattr(p, "_line_chem_snapshot", snap)


def restore_league_line_chemistry_ratings(league: Any) -> None:
    """Call before yearly progression: revert rating keys to pre-line-chem snapshot."""
    teams = getattr(league, "teams", None) or []
    for team in teams:
        for p in (getattr(team, "roster", None) or []) + (getattr(team, "scratches", None) or []):
            if getattr(p, "retired", False):
                continue
            snap = getattr(p, "_line_chem_snapshot", None)
            if not snap:
                continue
            r = getattr(p, "ratings", None)
            if not isinstance(r, dict):
                setattr(p, "_line_chem_snapshot", {})
                continue
            for k, v in snap.items():
                if k in r:
                    r[k] = clamp_rating(float(v))
            setattr(p, "_line_chem_snapshot", {})


def run_line_chemistry_pass(league: Any) -> List[Dict[str, Any]]:
    """
    Apply forward-line and D-pair chemistry to selected rating keys.
    Snapshots must be taken inside this pass before multipliers (see _capture).
    """
    report: List[Dict[str, Any]] = []
    arch_logs: List[str] = []
    teams = getattr(league, "teams", None) or []
    base_rng = random.Random((id(league) % 2**31) ^ 14041997)
    for team in teams:
        tname = str(getattr(team, "name", None) or getattr(team, "city", "") or getattr(team, "team_id", "Team"))
        trng = random.Random(base_rng.randint(1, 2**30) ^ (id(team) % 2**20))
        _optimize_forward_line_assignments(team, league, trng)
        roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
        for p in roster:
            prev = getattr(p, "_logged_hockey_archetype", None)
            ps = ensure_player_playstyle(p)
            if prev != ps:
                setattr(p, "_logged_hockey_archetype", ps)
                if len(arch_logs) < 52:
                    nm = career_player_name(p)
                    arch_logs.append(f"PLAYER ARCHETYPE: {nm} assigned archetype: {ps}")
        for label, line in _iter_team_forward_lines(team):
            for p in line:
                ensure_player_playstyle(p)
            _capture_line_chemistry_snapshots(line)
            chem = calculate_line_chemistry(line, team)
            apply_line_chemistry_effects(line, chem)
            styles = " / ".join(str(getattr(p, "playstyle", "?") or "?") for p in line)
            st_list = [str(getattr(p, "playstyle", "") or "").lower() for p in line]
            note = ""
            if len(line) == 3 and len(set(st_list)) == 1:
                note = f"BAD FIT: Line mismatch: 3 {st_list[0]} — poor distribution"
            elif chem < 0.40:
                note = "BAD FIT: Line mismatch — conflicting styles / redundant puck carriers"
            report.append(
                {
                    "team": tname,
                    "unit": "forwards",
                    "line": label,
                    "styles": styles,
                    "chemistry": round(chem, 3),
                    "effect": _line_chemistry_effect_label(chem),
                    "note": note,
                }
            )
        for label, pair in _iter_team_defense_pairs(team):
            for p in pair:
                ensure_player_playstyle(p)
            _capture_line_chemistry_snapshots(pair)
            chem = calculate_line_chemistry(pair, team)
            apply_line_chemistry_effects(pair, chem)
            styles = " / ".join(str(getattr(p, "playstyle", "?") or "?") for p in pair)
            st_list = [str(getattr(p, "playstyle", "") or "").lower() for p in pair]
            note = ""
            if st_list.count("offensive_d") >= 2:
                note = "BAD FIT: offensive_d pair — defensive liability risk"
            elif st_list.count("defensive_d") >= 2:
                note = "BAD FIT: defensive_d pair — stagnant breakout risk"
            report.append(
                {
                    "team": tname,
                    "unit": "defense",
                    "line": label,
                    "styles": styles,
                    "chemistry": round(chem, 3),
                    "effect": _line_chemistry_effect_label(chem),
                    "note": note,
                }
            )
    try:
        setattr(league, "_player_archetype_assignment_logs", list(arch_logs))
    except Exception:
        pass
    return report


# =====================================================================
# PLAYER RATING DISTRIBUTION (variance, tiers, roles, post-normalize rescue)
# =====================================================================

def collect_league_roster_players(league: Any) -> List[Any]:
    out: List[Any] = []
    for team in getattr(league, "teams", None) or []:
        for p in getattr(team, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            out.append(p)
    return out


def _ovr_nhl_scale(player: Any) -> float:
    return _player_ovr01(player) * 99.0


def _scale_player_ratings(player: Any, factor: float) -> None:
    r = getattr(player, "ratings", None)
    if not isinstance(r, dict) or factor <= 0:
        return
    for k in list(r.keys()):
        try:
            r[k] = clamp_rating(float(r[k]) * factor)
        except (TypeError, ValueError):
            pass


def _nudge_player_ovr_toward(player: Any, target01: float, strength: float) -> None:
    cur = _player_ovr01(player)
    if abs(cur - target01) < 0.004:
        return
    f = 1.0 + (target01 - cur) * strength
    f = max(0.965, min(1.035, f))
    _scale_player_ratings(player, f)


def _random_rating_keys_bump(player: Any, rng: random.Random, delta: int) -> None:
    r = getattr(player, "ratings", None)
    if not isinstance(r, dict) or not r:
        return
    keys = list(r.keys())
    n = max(1, min(12, len(keys) // 6 + 4))
    rng.shuffle(keys)
    for k in keys[:n]:
        try:
            r[k] = clamp_rating(float(r[k]) + float(delta))
        except (TypeError, ValueError):
            pass


def apply_distribution_variance(players: Sequence[Any], rng: random.Random) -> int:
    """Per-player OVR variance via scattered rating bumps (runs each season)."""
    touched = 0
    for p in players:
        if getattr(p, "retired", False):
            continue
        roll = rng.random()
        if roll < 0.08:
            _random_rating_keys_bump(p, rng, rng.randint(3, 8))
            touched += 1
        elif roll < 0.18:
            _random_rating_keys_bump(p, rng, rng.randint(1, 3))
            touched += 1
        elif roll > 0.92:
            _random_rating_keys_bump(p, rng, -rng.randint(3, 7))
            touched += 1
        elif roll > 0.82:
            _random_rating_keys_bump(p, rng, -rng.randint(1, 3))
            touched += 1
    return touched


def ensure_elite_players_ovr(players: Sequence[Any], rng: random.Random) -> int:
    """Guarantee a modest elite NHL band (85–92+); backs off hard when many are already elite."""
    plist = [p for p in players if not getattr(p, "retired", False)]
    if len(plist) < 10:
        return 0
    plist.sort(key=_player_ovr01, reverse=True)
    target_lo = 85.0 / 99.0
    already_elite = sum(1 for p in plist if _player_ovr01(p) >= target_lo - 0.004)
    if already_elite >= 18:
        return 0
    if already_elite >= 14:
        n = max(1, int(len(plist) * 0.028))
    elif already_elite >= 10:
        n = max(1, int(len(plist) * 0.045))
    elif already_elite >= 6:
        n = max(1, int(len(plist) * 0.065))
    else:
        n = max(1, int(len(plist) * 0.085))
    adj = 0
    target_hi = min(0.94, 92.0 / 99.0)
    for p in plist[:n]:
        if _player_ovr01(p) < target_lo:
            tgt = rng.uniform(target_lo, target_hi)
            _nudge_player_ovr_toward(p, tgt, strength=0.62)
            adj += 1
    return adj


def enforce_bottom_tier_ovr(players: Sequence[Any], rng: random.Random) -> int:
    """Bottom ~30%: pull inflated depth players down toward realistic depth band."""
    plist = [p for p in players if not getattr(p, "retired", False)]
    if len(plist) < 8:
        return 0
    plist.sort(key=_player_ovr01)
    n = max(1, int(len(plist) * 0.30))
    ceiling = 72.0 / 99.0
    adj = 0
    for p in plist[:n]:
        if _player_ovr01(p) > ceiling:
            _random_rating_keys_bump(p, rng, -rng.randint(2, 6))
            adj += 1
    return adj


def enforce_player_distribution(players: Sequence[Any], rng: random.Random) -> int:
    """
    Percentile tiers (by league rank): elite / top / middle / depth target bands on OVR scale.
    Caps how many sub-85 players can be nudged into the elite band per pass (rookie waves used to inflate 85+).
    """
    plist = [p for p in players if not getattr(p, "retired", False)]
    if len(plist) < 16:
        return 0
    plist.sort(key=_player_ovr01, reverse=True)
    n = len(plist)
    t85 = 85.0 / 99.0
    elite_now = sum(1 for p in plist if _player_ovr01(p) >= t85 - 0.003)
    elite_soft_target = min(26, max(12, int(n * 0.028)))
    upward_elite_budget = max(0, elite_soft_target - elite_now)
    upward_elite_budget = min(upward_elite_budget, 6)
    if elite_now >= 17:
        upward_elite_budget = min(upward_elite_budget, max(0, 22 - elite_now))
    e_end = max(1, int(n * 0.065))
    bands = (
        (0, e_end, t85, 0.99),
        (e_end, int(n * 0.30), 78.0 / 99.0, t85),
        (int(n * 0.30), int(n * 0.70), 70.0 / 99.0, 78.0 / 99.0),
        (int(n * 0.70), n, 60.0 / 99.0, 70.0 / 99.0),
    )
    adj = 0
    for band_idx, (i_lo, i_hi, t_lo, t_hi) in enumerate(bands):
        mid = 0.5 * (t_lo + t_hi)
        is_elite_band = t_lo >= t85 - 0.001
        for i in range(i_lo, min(i_hi, n)):
            p = plist[i]
            cur = _player_ovr01(p)
            if cur < t_lo - 0.01:
                if is_elite_band and upward_elite_budget <= 0:
                    continue
                st = 0.19 if is_elite_band else 0.32
                if band_idx == 1 and elite_now >= 16:
                    st = min(st, 0.20)
                if band_idx == 1 and elite_now >= 22:
                    st = min(st, 0.14)
                if is_elite_band:
                    upward_elite_budget -= 1
                _nudge_player_ovr_toward(p, min(mid + 0.02, t_hi - 0.01), strength=st)
                adj += 1
            elif cur > t_hi + 0.01:
                _nudge_player_ovr_toward(p, max(mid - 0.02, t_lo + 0.01), strength=0.28)
                adj += 1
    return adj


def assign_player_roles_percentile(players: Sequence[Any]) -> int:
    """
    Roles from league rank within position group (overwrites prior role strings for skaters).
    """
    buckets: Dict[str, List[Any]] = {"F": [], "D": [], "G": []}
    for p in players:
        if getattr(p, "retired", False):
            continue
        lab = _player_position_label(p)
        if lab == "G":
            buckets["G"].append(p)
        elif lab == "D":
            buckets["D"].append(p)
        elif lab in ("C", "LW", "RW"):
            buckets["F"].append(p)
    moved = 0
    for grp, plist in buckets.items():
        if len(plist) < 2:
            continue
        plist.sort(key=_player_ovr01, reverse=True)
        n = len(plist)
        for i, p in enumerate(plist):
            pct = i / n
            if grp == "F":
                if pct < 0.10:
                    role, narr = "superstar", "superstar"
                elif pct < 0.30:
                    role, narr = "top_line", "top_line"
                elif pct < 0.70:
                    role, narr = "middle_6", "middle_6"
                else:
                    role, narr = "bottom_6", "bottom_6"
            elif grp == "D":
                if pct < 0.10:
                    role, narr = "elite", "superstar_lane"
                elif pct < 0.30:
                    role, narr = "top_4", "top_line"
                elif pct < 0.70:
                    role, narr = "middle_6", "middle_6"
                else:
                    role, narr = "bottom_6", "bottom_6"
            else:
                if pct < 0.10:
                    role, narr = "elite", "superstar"
                elif pct < 0.30:
                    role, narr = "starter", "top_line"
                elif pct < 0.70:
                    role, narr = "backup", "middle_6"
                else:
                    role, narr = "depth", "bottom_6"
            try:
                if getattr(p, "role", None) != role:
                    moved += 1
                setattr(p, "role", role)
                setattr(p, "role_narrative", narr)
            except Exception:
                pass
    return moved


def summarize_roster_distribution(league: Any) -> Dict[str, Any]:
    players = collect_league_roster_players(league)
    ov = [_player_ovr01(p) for p in players]
    nh = [_ovr_nhl_scale(p) for p in players]
    if not ov:
        return {"n_players": 0}
    mean = sum(ov) / len(ov)
    var = sum((x - mean) ** 2 for x in ov) / len(ov)
    std = var**0.5
    count_elite = sum(1 for v in nh if v >= 85.0)
    count_top = sum(1 for v in nh if 78.0 <= v < 85.0)
    count_mid = sum(1 for v in nh if 70.0 <= v < 78.0)
    count_bot = sum(1 for v in nh if v < 70.0)
    return {
        "n_players": len(players),
        "mean_ovr01": round(mean, 4),
        "std_ovr01": round(std, 4),
        "count_elite_85p": count_elite,
        "count_top_line_78_85": count_top,
        "count_middle_70_78": count_mid,
        "count_bottom_under_70": count_bot,
        "min_nhl_ovr": round(min(nh), 1),
        "max_nhl_ovr": round(max(nh), 1),
    }


def post_normalize_distribution_rescue(league: Any, rng: random.Random) -> Dict[str, Any]:
    """
    If league OVR std is still too tight after tuning normalize, widen ends (counter-compress).
    """
    rep: Dict[str, Any] = {"std_before": 0.0, "std_after": 0.0, "widened": False}
    players = collect_league_roster_players(league)
    if len(players) < 12:
        return rep
    ov = [_player_ovr01(p) for p in players]
    mean = sum(ov) / len(ov)
    var = sum((x - mean) ** 2 for x in ov) / len(ov)
    std = var**0.5
    rep["std_before"] = round(std, 5)
    if std >= 0.047:
        rep["std_after"] = round(std, 5)
        return rep
    plist = sorted(players, key=_player_ovr01)
    n = max(2, len(plist) // 7)
    for p in plist[:n]:
        _scale_player_ratings(p, 0.991)
    for p in plist[-n:]:
        _scale_player_ratings(p, 1.009)
    ov2 = [_player_ovr01(p) for p in players]
    mean2 = sum(ov2) / len(ov2)
    var2 = sum((x - mean2) ** 2 for x in ov2) / len(ov2)
    std2 = var2**0.5
    rep["std_after"] = round(std2, 5)
    rep["widened"] = True
    return rep


def run_player_distribution_pipeline(league: Any, rng: random.Random) -> Dict[str, Any]:
    """Order: variance → elite → bottom → tier enforcement (roles after tuning in runner)."""
    char_n = initialize_league_player_characters(league, rng)
    players = collect_league_roster_players(league)
    out: Dict[str, Any] = {
        "character_players_initialized": char_n,
        "variance_touches": apply_distribution_variance(players, rng),
        "elite_adjusted": ensure_elite_players_ovr(players, rng),
        "bottom_adjusted": enforce_bottom_tier_ovr(players, rng),
        "tier_adjusted": enforce_player_distribution(players, rng),
    }
    out["summary_before_tuning"] = summarize_roster_distribution(league)
    return out


def _maybe_volatile_storyline_fx(fx: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    if rng.random() > 0.28 or not fx:
        return fx
    out = dict(fx)
    negs = [k for k, v in out.items() if float(v) < -0.025]
    if negs and rng.random() < 0.42:
        for k in negs:
            out[k] = -float(out[k]) * rng.uniform(0.30, 0.70)
    return out


def _storyline_effect_summary_pct(fx: Dict[str, float]) -> str:
    if not fx:
        return "minor ripple effects"
    label = {
        "confidence": "confidence",
        "morale": "morale",
        "clutch": "clutch",
        "leadership": "leadership",
        "mental_toughness": "mental toughness",
        "media_stress": "media stress",
        "internal_motivation": "drive",
        "chemistry": "team chemistry",
        "consistency": "consistency",
        "decision": "decision load",
        "anxiety": "anxiety",
        "contract_pressure": "contract pressure",
        "performance": "performance",
        "media_comfort": "media comfort",
    }
    parts: List[str] = []
    for k, v in sorted(fx.items(), key=lambda kv: kv[0]):
        parts.append(f"{label.get(k, k)} {float(v) * 100.0:+.0f}%")
    return "  ".join(parts)


def _duration_phrase(seasons_left: float, dur_key: str) -> str:
    if dur_key == "short" or seasons_left <= 0.35:
        w = int(round(seasons_left * 52.0))
        w = max(2, min(8, w))
        return f"{w} weeks (sim year fraction)"
    if dur_key == "long" or seasons_left >= 0.95:
        return "full season"
    m = int(round(seasons_left * 9.0))
    m = max(1, min(10, m))
    return f"~{m} months (arc)"


def _eligible_storyline_def(
    d: Dict[str, Any],
    char: int,
    tag: str,
) -> bool:
    if char > int(d.get("char_max", 100)):
        return False
    if char < int(d.get("char_min", 0)):
        return False
    if d.get("legal") and char >= 50:
        return False
    if d.get("star_only") and tag != "star":
        return False
    if d.get("vet_only") and tag != "veteran":
        return False
    if d.get("rookie_only") and tag != "rookie":
        return False
    return True


def _pick_weight_storyline(
    d: Dict[str, Any],
    perf_delta: float,
    tag: str,
) -> float:
    w = 1.0
    t = (d.get("text") or "").lower()
    if perf_delta <= -0.035 and any(x in t for x in ("slump", "bench", "collapse", "crisis", "panic")):
        w *= 1.65
    if perf_delta >= 0.035 and any(x in t for x in ("surge", "breakout", "leader", "clutch", "mentor")):
        w *= 1.55
    if tag == "rookie" and any(x in t for x in ("rookie", "first", "draft", "learning")):
        w *= 1.12
    if tag == "veteran" and any(x in t for x in ("veteran", "legacy", "mentor", "last", "final")):
        w *= 1.2
    return w


def _build_player_storyline_catalog() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def push(
        pool: str,
        text: str,
        fx: Dict[str, float],
        dur: str = "medium",
        *,
        legal: bool = False,
        char_max: int = 100,
        char_min: int = 0,
        volatile: bool = False,
        star_only: bool = False,
        vet_only: bool = False,
        rookie_only: bool = False,
        tier: Optional[str] = None,
    ) -> None:
        tier_eff = tier if tier is not None else _STORYLINE_POOL_TIER.get(pool, "mid")
        out.append(
            {
                "id": f"{pool}_{len(out)}",
                "pool": pool,
                "text": text,
                "fx": dict(fx),
                "dur": dur,
                "legal": legal,
                "char_max": char_max,
                "char_min": char_min,
                "volatile": volatile,
                "star_only": star_only,
                "vet_only": vet_only,
                "rookie_only": rookie_only,
                "tier": tier_eff,
            }
        )

    legal_lines = [
        "Arrested for DUI with teammate in car",
        "DUI checkpoint fail after team win",
        "Arrested for street racing in team city",
        "Illegal drag racing ring involvement",
        "Bar fight with opposing fan",
        "Bar fight with teammate",
        "Assault after trash talk escalates",
        "Punching security guard",
        "Arrested outside casino",
        "Gambling debt leads to threats",
        "Betting on own team games",
        "Betting against own team (huge scandal)",
        "Involved in underground betting ring",
        "Suspicious betting patterns linked to player",
        "Friend betting using player info",
        "Crypto scam involvement",
        "NFT scam endorsement backlash",
        "Fraud through fake business partner",
        "Ponzi scheme victim (or participant)",
        "Money laundering through club ownership",
        "Domestic dispute (no charges, media chaos)",
        "Domestic violence accusation",
        "Restraining order issued",
        "Public argument caught on video",
        "Stalking allegation",
        "Destroying hotel room",
        "Kicked off airplane",
        "Airport altercation arrest",
        "Refuses security check, detained",
        "Public meltdown in restaurant",
        "Illegal firearm possession",
        "Bringing weapon to public event",
        "Reckless discharge (accidental)",
        "Threatening someone with weapon",
        "Social media weapon video backlash",
        "Drug possession arrest",
        "Performance-enhancing substance scandal",
        "Party drug overdose scare",
        "Suspended for substance abuse",
        "Rehab entry mid-season",
        "Arrested due to mistaken identity",
        "Impersonation scam tied to player",
        "Fake charity fraud scandal",
        "Identity theft victim, mental spiral",
        "Blackmail attempt exposed",
        "Secret recorded video leak",
        "Police investigation for unknown reason (mystery arc)",
        "Arrested abroad (visa/legal chaos)",
        "Political protest arrest",
        "Viral police bodycam footage incident",
        "DUI after team party",
        "Suspended pending league investigation",
        "Tax evasion probe becomes public",
        "Witness in organized crime case (distraction)",
    ]
    for i, lt in enumerate(legal_lines):
        base = -0.18 - (i % 7) * 0.015
        push(
            "legal_crime",
            lt,
            {
                "confidence": round(base, 3),
                "morale": round(base * 1.1, 3),
                "clutch": round(base * 0.75, 3),
                "media_stress": 0.12 + (i % 4) * 0.02,
            },
            dur="short" if i % 3 == 0 else "medium",
            legal=True,
            char_max=49,
            volatile=True,
        )

    mental_tpl = [
        ("Playoff exit triggers buried panic attacks", {"confidence": -0.14, "morale": -0.11, "anxiety": 0.13}, "medium", True),
        ("Sports psychologist enlisted after benching spiral", {"confidence": -0.10, "morale": -0.08, "internal_motivation": 0.06}, "medium", False),
        ("Imposter syndrome on promoted scoring line", {"confidence": -0.16, "consistency": -0.10, "decision": 0.08}, "short", True),
        ("Sleep disorder wrecks morning skate habits", {"performance": -0.08, "morale": -0.09, "anxiety": 0.10}, "medium", False),
        ("Hyperfixation on metrics erodes on-ice instinct", {"decision": 0.10, "confidence": -0.08}, "long", True),
        ("Private depression diagnosis leaks to press", {"morale": -0.18, "media_stress": 0.16, "confidence": -0.12}, "medium", True),
        ("Confidence crater after repeated shootout failures", {"confidence": -0.20, "clutch": -0.18}, "short", True),
        ("Burnout: skips optional skates, tension with staff", {"internal_motivation": -0.14, "morale": -0.10}, "long", False),
        ("Therapist clearance saga becomes distraction", {"media_stress": 0.11, "anxiety": 0.09}, "medium", True),
        ("OCD rituals disrupt road routine", {"consistency": -0.08, "anxiety": 0.11}, "long", False),
    ]
    for extra in range(48):
        stem, fx0, dur0, vol = mental_tpl[extra % len(mental_tpl)]
        push(
            "mental_psychological",
            f"{stem} (arc thread {extra + 1})",
            {k: round(float(v) + (extra % 5) * 0.006 * (1 if v > 0 else -1), 3) for k, v in fx0.items()},
            dur=dur0,
            volatile=vol,
        )

    personal_tpl = [
        ("New parent sleep debt craters recovery", {"performance": -0.09, "morale": -0.07}, "medium"),
        ("Messy breakup splashed across gossip blogs", {"confidence": -0.12, "media_stress": 0.14, "morale": -0.10}, "short", True),
        ("Engagement announced, hometown hero spotlight", {"confidence": 0.08, "media_comfort": 0.06}, "short"),
        ("Family illness forces repeated travel absences", {"morale": -0.13, "internal_motivation": -0.06}, "long"),
        ("Wedding planning chaos during road trip grind", {"decision": 0.07, "morale": -0.06}, "medium"),
        ("Custody hearing dates clash with playoff push", {"anxiety": 0.15, "morale": -0.14}, "medium", True),
        ("Long-distance relationship strain on West coast swing", {"morale": -0.09, "consistency": -0.07}, "medium"),
        ("New house burglary rattles sense of safety", {"anxiety": 0.12, "confidence": -0.09}, "short", True),
        ("Sibling rivalry in local media comparisons", {"media_stress": 0.10, "confidence": -0.07}, "short"),
        ("Charity work overload, stretched thin", {"internal_motivation": 0.05, "performance": -0.05}, "medium"),
    ]
    for extra in range(46):
        row = personal_tpl[extra % len(personal_tpl)]
        stem, fx0 = row[0], row[1]
        dur0 = row[2] if len(row) > 2 else "medium"
        vol = row[3] if len(row) > 3 else False
        push(
            "personal_life",
            f"{stem} (beat {extra + 1})",
            {k: round(float(v) + (extra % 4) * 0.005 * (1 if v > 0 else -1), 3) for k, v in fx0.items()},
            dur=dur0,
            volatile=vol,
        )

    media_tpl = [
        ("Radio host questions heart nightly", {"media_stress": 0.14, "confidence": -0.10}, "medium"),
        ("Viral clip misread as dirty play", {"media_stress": 0.18, "morale": -0.11}, "short", True),
        ("Softball interview turns into trap question meltdown", {"media_comfort": -0.12, "confidence": -0.08}, "short", True),
        ("Podcast tour exposes awkward quotes", {"media_stress": 0.11, "chemistry": -0.05}, "medium"),
        ("Trade rumor mill pins star target on him", {"anxiety": 0.13, "contract_pressure": 0.10}, "medium"),
        ("Local paper runs anonymous source hit piece", {"morale": -0.12, "media_stress": 0.15}, "medium", True),
        ("National panel debates his ceiling endlessly", {"confidence": -0.07, "media_stress": 0.10}, "long"),
        ("Social media pile-on after bad turnover", {"confidence": -0.14, "anxiety": 0.12}, "short", True),
        ("PR team stages rebound narrative", {"media_comfort": 0.09, "confidence": 0.06}, "short"),
        ("Documentary crew embeds, locker room tightens", {"chemistry": -0.06, "internal_motivation": 0.05}, "long"),
    ]
    for extra in range(44):
        row = media_tpl[extra % len(media_tpl)]
        stem, fx0 = row[0], row[1]
        dur0 = row[2] if len(row) > 2 else "medium"
        vol = row[3] if len(row) > 3 else False
        push(
            "media_pressure",
            f"{stem} (cycle {extra + 1})",
            {k: round(float(v) + (extra % 4) * 0.004 * (1 if v > 0 else -1), 3) for k, v in fx0.items()},
            dur=dur0,
            volatile=vol,
        )

    team_tpl = [
        ("Quiet feud with assistant coach spills into ice time", {"morale": -0.12, "chemistry": -0.08}, "medium", True),
        ("Chemistry experiment line demotion sparks sulk", {"confidence": -0.10, "internal_motivation": -0.08}, "short", True),
        ("Leadership group vote snub stings publicly", {"leadership": -0.10, "morale": -0.14}, "medium", True),
        ("Roomie clash on long road trip", {"chemistry": -0.10, "morale": -0.07}, "short"),
        ("Veteran calls him out in film session", {"mental_toughness": 0.06, "confidence": -0.08}, "short", True),
        ("Young core rallies around his energy", {"chemistry": 0.12, "leadership": 0.10}, "long"),
        ("Captaincy chatter divides fan base", {"media_stress": 0.12, "confidence": 0.07}, "medium"),
        ("Healthy scratch streak, agent whispers trade demand", {"morale": -0.16, "contract_pressure": 0.12}, "medium", True),
        ("PP1 promotion tightens internal jealousy", {"chemistry": -0.07, "performance": 0.06}, "medium", True),
        ("Locker room leader emerges in losing skid", {"leadership": 0.14, "chemistry": 0.11, "morale": 0.08}, "long"),
    ]
    for extra in range(44):
        row = team_tpl[extra % len(team_tpl)]
        stem, fx0 = row[0], row[1]
        dur0 = row[2] if len(row) > 2 else "medium"
        vol = row[3] if len(row) > 3 else False
        char_min = 72 if ("leader emerges" in stem or "Young core rallies" in stem) else 0
        push(
            "team_dynamics",
            f"{stem} (thread {extra + 1})",
            {k: round(float(v) + (extra % 4) * 0.005 * (1 if v > 0 else -1), 3) for k, v in fx0.items()},
            dur=dur0,
            volatile=vol,
            char_min=char_min,
        )

    money_tpl = [
        ("Contract year surge: betting on himself nightly", {"performance": 0.14, "contract_pressure": 0.12, "confidence": 0.08}, "long"),
        ("Bridge deal stalemate wears on focus", {"morale": -0.10, "contract_pressure": 0.14}, "medium", True),
        ("Holdout threat leaked by camp", {"media_stress": 0.16, "chemistry": -0.09}, "short", True),
        ("Arbitration hearing prep becomes obsession", {"decision": 0.09, "anxiety": 0.11}, "medium"),
        ("NTC chatter dominates interviews", {"media_stress": 0.10, "morale": 0.05}, "medium"),
        ("Underpaid vs peers chip on shoulder", {"internal_motivation": 0.10, "confidence": 0.06}, "long"),
        ("Agent change mid-season whispers", {"anxiety": 0.10, "confidence": -0.06}, "short", True),
        ("Bonus structure incentives risky play", {"performance": 0.06, "decision": 0.08}, "medium", True),
        ("Buyout buzz after rough October", {"confidence": -0.14, "morale": -0.12}, "short", True),
        ("Extension celebration, hometown discount narrative", {"morale": 0.12, "chemistry": 0.07}, "medium"),
    ]
    for extra in range(40):
        row = money_tpl[extra % len(money_tpl)]
        stem, fx0 = row[0], row[1]
        dur0 = row[2] if len(row) > 2 else "medium"
        vol = row[3] if len(row) > 3 else False
        push(
            "money_career",
            f"{stem} (beat {extra + 1})",
            {k: round(float(v) + (extra % 4) * 0.004 * (1 if v > 0 else -1), 3) for k, v in fx0.items()},
            dur=dur0,
            volatile=vol,
        )

    chaos_tpl = [
        ("Bizarre injury from freak pregame ritual", {"morale": -0.08, "anxiety": 0.10}, "short", True),
        ("Lost passport strand in foreign city", {"morale": -0.09, "media_stress": 0.08}, "short"),
        ("Wrong bag swapped, plays in borrowed skates", {"confidence": -0.06, "consistency": -0.08}, "short", True),
        ("Cryptid meme account claims he is witness", {"media_stress": 0.09, "media_comfort": -0.05}, "medium"),
        ("Accidental live mic confession during intermission", {"media_stress": 0.14, "chemistry": -0.07}, "short", True),
        ("Charity auction item is cursed joke that goes viral", {"media_comfort": 0.07, "media_stress": 0.08}, "medium", True),
        ("Team bus detour into county fair chaos", {"morale": 0.06, "chemistry": 0.07}, "short"),
        ("Mascot feud becomes running bit", {"morale": 0.08, "confidence": 0.05}, "short"),
        ("Weather delay traps team at airport for 30 hours", {"morale": -0.06, "anxiety": 0.07}, "short"),
        ("Reality TV cameo request divides management", {"media_stress": 0.10, "morale": -0.05}, "medium"),
    ]
    for extra in range(46):
        row = chaos_tpl[extra % len(chaos_tpl)]
        stem, fx0 = row[0], row[1]
        dur0 = row[2] if len(row) > 2 else "medium"
        vol = row[3] if len(row) > 3 else False
        push(
            "chaotic_weird",
            f"{stem} (variant {extra + 1})",
            {k: round(float(v) + (extra % 5) * 0.004 * (1 if v > 0 else -1), 3) for k, v in fx0.items()},
            dur=dur0,
            volatile=vol,
        )

    push(
        "team_dynamics",
        "Clutch reputation cements after playoff OT winner",
        {"confidence": 0.10, "clutch": 0.16, "morale": 0.10},
        dur="medium",
        star_only=True,
        volatile=True,
        tier="major",
    )
    push(
        "team_dynamics",
        "Mentorship arc: shelters rookie roommate through slump",
        {"leadership": 0.14, "chemistry": 0.10, "internal_motivation": 0.08},
        dur="long",
        char_min=68,
        vet_only=True,
    )
    push(
        "money_career",
        "Rookie deal overperformance sparks endorsement rush",
        {"confidence": 0.09, "media_stress": 0.08, "performance": 0.08},
        dur="medium",
        rookie_only=True,
    )
    push(
        "mental_psychological",
        "Resilience arc: sports psych + staff alignment clicks",
        {"confidence": 0.12, "mental_toughness": 0.10, "anxiety": -0.10},
        dur="long",
        char_min=55,
    )
    push(
        "media_pressure",
        "Handled pressure tour: turns boos into fuel",
        {"mental_toughness": 0.12, "confidence": 0.10, "media_stress": -0.08},
        dur="medium",
        char_min=62,
        volatile=True,
    )

    return out


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

    def __init__(self, seed: int | None = None, debug: bool = False):
        self.year: int = 0
        self.season_aging_events: int = 0
        self.season_player_count: int = 0
        self.max_aging_events: int = 0
        self.season_breakouts: int = 0
        self.max_breakouts: int = 0
        self.seed: int = seed if seed is not None else random.randrange(1, 10**18)
        self.rng: random.Random = random.Random(self.seed)
        self.retired: bool = False
        self.debug: bool = debug
        self.last_draft_lottery = None

        # ------------------------------------
        # League ecosystem (MACRO ONLY)
        # ------------------------------------
        self.league: League = League(seed=self.seed)

        # Cached league outputs for the latest season
        self.last_league_context: dict | None = None
        self.last_league_forecast: dict | None = None
        self.last_league_shocks: list[dict] = []
        self.league_history: list[LeagueSeasonResult] = []

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
        # Waiver System
        # ------------------------------------
        self.waiver_engine = WaiverEngine(
            config=WaiverConfig(early_season_cutoff_day=30)
        )
        self.waiver_priority: list[str] = []


        # ------------------------------------
        # Injected entities
        # ------------------------------------
        self.player: Player | None = None
        self.team: Team | None = None
        # ------------------------------------
        # Prospect pipeline (pre-draft)
        # ------------------------------------
        self.prospects: list[Prospect] = []
        self.draft_class: list[Prospect] = []
        self.scout_pool: list[ScoutProfile] = []
        self.last_draft_results: list[dict] = []
        self._pipeline_log_buffer: List[str] = []
        self._last_promotion_actual: int = 0
        self._last_draft_class_tier_counts: Dict[str, int] = {}

                # ------------------------------------
        # Scouting Departments (NEW)
        # ------------------------------------
        self.team_scouting_departments: dict[str, Any] = {}



        # ------------------------------------
        # Coach (ACTIVE system)
        # ------------------------------------
        self.coach: Coach | None = None
        self.coach_last_season: dict | None = None


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

        # Ensure league has a teams list and populate it so universe layer can iterate
        if not hasattr(self.league, "teams") or self.league.teams is None:
            self.league.teams = []
        if len(self.league.teams) == 0:
            self.initialize_universe()

    # --------------------------------------------------
    # League initialization (exposes league.teams to universe layer)
    # --------------------------------------------------
    def initialize_universe(self, team_count: int = 32) -> None:
        """
        If league already has teams, return. Otherwise build deterministic
        teams + coaches and append to self.league.teams. Safe to call twice.
        """
        existing = getattr(self.league, "teams", None)
        if existing is not None and len(existing) > 0:
            return
        if not hasattr(self.league, "teams") or self.league.teams is None:
            self.league.teams = []
        rng = random.Random(self.seed)
        cities = [
            "New York", "Boston", "Toronto", "Montreal", "Chicago", "Detroit",
            "Los Angeles", "San Jose", "Dallas", "Calgary", "Edmonton", "Vancouver",
            "Ottawa", "Buffalo", "Pittsburgh", "Philadelphia", "Washington", "Tampa Bay",
            "Florida", "Carolina", "Columbus", "Nashville", "Minnesota", "St. Louis",
            "Winnipeg", "Arizona", "Seattle", "Vegas", "Colorado", "New Jersey",
            "Anaheim", "Islanders",
        ]
        names = [
            "Rangers", "Bruins", "Maple Leafs", "Canadiens", "Blackhawks", "Red Wings",
            "Kings", "Sharks", "Stars", "Flames", "Oilers", "Canucks", "Senators",
            "Sabres", "Penguins", "Flyers", "Capitals", "Lightning", "Panthers",
            "Hurricanes", "Blue Jackets", "Predators", "Wild", "Blues", "Jets",
            "Coyotes", "Kraken", "Golden Knights", "Avalanche", "Devils", "Ducks",
            "Islanders",
        ]
        divisions = ("Atlantic", "Metropolitan", "Central", "Pacific")
        for i in range(team_count):
            city = cities[i % len(cities)]
            name = names[i % len(names)]
            division = divisions[i % 4]
            conference = "Eastern" if (i % 4) < 2 else "Western"
            team = Team(
                team_id=i,
                city=city,
                name=name,
                division=division,
                conference=conference,
                archetype=TeamArchetype.PATIENT_BUILDER,
                rng=rng,
            )
            coach = generate_coach(rng, f"COACH_{i:03d}", CoachRole.HEAD_COACH)
            coach.name = f"Coach_{i:03d}"
            coach.job_security = 0.6
            team.coach = coach
            assign_team_system(team, rng)
            assign_team_coach_profile(team, rng)
            self.league.teams.append(team)
        self.league.identity.max_teams = team_count
        self._populate_initial_rosters()

    def _populate_initial_rosters(self) -> None:
        """
        Deterministic roster generation: 12 F (4C, 4LW, 4RW), 6 D, 3 G per team.
        Uses self.rng only. Populates league.players and each team.roster.
        """
        if not getattr(self.league, "teams", None):
            return
        rng = self.rng
        if not hasattr(self.league, "players"):
            self.league.players = []
        if not hasattr(self.league, "retired_players"):
            self.league.retired_players = []
        age_ranges = (
            [(31, 36)] * 2 + [(24, 29)] * 6 + [(20, 23)] * 6 + [(26, 32)] * 4 + [(22, 30)] * 3
        )
        fwd_positions = [Position.C] * 4 + [Position.LW] * 4 + [Position.RW] * 4
        def_positions = [Position.D] * 6
        g_positions = [Position.G] * 3
        ovr_tiers = (
            [(0.52, 0.68)] * 6 + [(0.62, 0.76)] * 5 + [(0.70, 0.82)] * 5 + [(0.76, 0.86)] * 3 + [(0.82, 0.90)] * 2
        )
        assert len(ovr_tiers) >= 21
        GEN_P, ELITE_P, STAR_P = 0.003, 0.018, 0.045
        used_names = set()
        for team_idx, team in enumerate(self.league.teams):
            if not hasattr(team, "roster") or team.roster is None:
                team.roster = []
            team.roster.clear()
            tier_cycle = [ovr_tiers[(team_idx * 7 + i) % len(ovr_tiers)] for i in range(21)]
            rng.shuffle(tier_cycle)
            age_order = list(age_ranges)
            rng.shuffle(age_order)
            slot_idx = 0
            for pos_list in (fwd_positions, def_positions, g_positions):
                for pos in pos_list:
                    lo, hi = tier_cycle[slot_idx]
                    target_ovr = lo + rng.uniform(0, hi - lo)
                    roll = rng.random()
                    if roll < GEN_P:
                        target_ovr = max(target_ovr, rng.uniform(0.93, 0.96))
                    elif roll < GEN_P + ELITE_P:
                        target_ovr = max(target_ovr, rng.uniform(0.90, 0.92))
                    elif roll < GEN_P + ELITE_P + STAR_P:
                        target_ovr = max(target_ovr, rng.uniform(0.86, 0.89))
                    else:
                        target_ovr = min(target_ovr, 0.78)
                    if roll >= GEN_P + ELITE_P:
                        target_ovr = min(target_ovr, 0.89)
                    age_lo, age_hi = age_order[slot_idx]
                    age = rng.randint(age_lo, age_hi)
                    birth_year = (2025 - age)
                    base_rating = int(target_ovr * 99)
                    ratings = {k: clamp_rating(base_rating + rng.randint(-2, 2)) for k in ATTRIBUTE_KEYS}
                    seed = rng.randint(1, 2_000_000_000)
                    ident = generate_human_identity(rng)
                    for _ in range(5):
                        nm = str(getattr(ident, "full_name", "Unknown"))
                        if nm not in used_names:
                            used_names.add(nm)
                            break
                        ident = generate_human_identity(rng)
                    hometown = str(ident.hometown or "Unknown")
                    birth_city = hometown.split(",")[0].strip() if hometown else "Unknown"
                    identity = IdentityBio(
                        name=str(ident.full_name),
                        age=age,
                        birth_year=birth_year,
                        birth_country=str(ident.nationality),
                        birth_city=birth_city or "Unknown",
                        height_cm=180 + rng.randint(-8, 8),
                        weight_kg=85 + rng.randint(-10, 10),
                        position=pos,
                        shoots=Shoots.L if rng.random() < 0.6 else Shoots.R,
                        draft_year=max(2018, birth_year + 18),
                        draft_round=1 + (slot_idx % 3),
                        draft_pick=1 + (slot_idx % 30),
                    )
                    backstory = BackstoryUpbringing(
                        backstory=BackstoryType.GRINDER,
                        upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
                        family_support=SupportLevel.MEDIUM,
                        early_pressure=PressureLevel.MODERATE,
                        dev_resources=DevResources.LOCAL,
                    )
                    player = Player(
                        identity=identity,
                        backstory=backstory,
                        ratings=ratings,
                        rng_seed=seed,
                    )
                    player.context.current_team_id = str(getattr(team, "team_id", team_idx))
                    team.roster.append(player)
                    self.league.players.append(player)
                    slot_idx += 1
            if hasattr(team, "state") and hasattr(team.state, "competitive_score"):
                roster = getattr(team, "roster", []) or []
                if roster:
                    ovrs = sorted((p.ovr() for p in roster if callable(getattr(p, "ovr", None))), reverse=True)[:12]
                    team.state.competitive_score = sum(ovrs) / len(ovrs) if ovrs else 0.5

    # --------------------------------------------------
    # Injection
    # --------------------------------------------------
    def add_prospect(self, prospect: Prospect) -> None:
        self.prospects.append(prospect)

    def set_draft_class(self, prospects: list[Prospect]) -> None:
        self.draft_class = prospects

    # --- Global persistent player ecosystem (900–1200; no yearly synthetic draft class) ---
    GLOBAL_POOL_MIN: int = 900
    GLOBAL_POOL_HARD_MAX: int = 1200
    GLOBAL_YEARLY_INTAKE_MIN: int = 80
    GLOBAL_YEARLY_INTAKE_MAX: int = 120
    DRAFT_CLASS_SIZE_MIN: int = 200
    DRAFT_CLASS_SIZE_MAX: int = 300
    NHL_PROSPECT_PIPELINE_CAP: int = 16
    NHL_PROMOTIONS_PER_TEAM_PER_YEAR: int = 5

    def _global_pool_bootstrap_prospect(self, rng: random.Random, year: int, *, age: int, bucket: str) -> Prospect:
        birth_year = int(year) - int(age)
        country = str(rng.choice(["Canada", "Canada", "USA", "Sweden", "Finland", "Russia", "Germany", "Czechia"]))
        city = "Unknown"
        pos_str = str(rng.choice(["C", "C", "LW", "RW", "D", "D", "G"]))
        position = (
            ProspectPosition.C
            if pos_str == "C"
            else ProspectPosition.D
            if pos_str == "D"
            else ProspectPosition.G
            if pos_str == "G"
            else (ProspectPosition.LW if pos_str == "LW" else ProspectPosition.RW)
        )
        system = rng.choice(
            [
                DevelopmentSystem.CHL,
                DevelopmentSystem.NCAA,
                DevelopmentSystem.EURO_JR,
                DevelopmentSystem.PREP,
            ]
        )
        seed_val = abs(hash(f"GP|{year}|{age}|{bucket}|{rng.random()}")) % (2**31 - 1) or 1
        pr = Prospect.create_random(
            name=f"Global {bucket[:3]} {year}-{seed_val % 100000}",
            birth_year=birth_year,
            birth_country=country,
            birth_city=city,
            position=position,
            shoots=ProspectShoots.L if rng.random() < 0.58 else ProspectShoots.R,
            height_cm=176 + rng.randint(-10, 14),
            weight_kg=80 + rng.randint(-12, 16),
            system=system,
            country=country,
            region="",
            age=int(age),
            seed=seed_val,
        )
        pr.team_id = None
        pr.status = "global"
        pr.phase = ProspectPhase.STRUCTURED_JUNIOR if age < 18 else ProspectPhase.DRAFT_YEAR
        setattr(pr, "_global_league_bucket", str(bucket).upper())
        setattr(pr, "_years_to_draft_eligibility", max(0, 18 - int(age)))
        roll = rng.random()
        if roll < 0.008:
            ovr_lo, ovr_hi = 0.58 + rng.random() * 0.12, 0.78 + rng.random() * 0.14
            tier_name, dev_idx, franchise_flag = "elite", 0, rng.random() < 0.10
        elif roll < 0.045:
            ovr_lo, ovr_hi = 0.52 + rng.random() * 0.10, 0.72 + rng.random() * 0.12
            tier_name, dev_idx, franchise_flag = "high", 1, False
        elif roll < 0.22:
            ovr_lo, ovr_hi = 0.46 + rng.random() * 0.10, 0.66 + rng.random() * 0.12
            tier_name, dev_idx, franchise_flag = "mid", 2, False
        elif roll < 0.58:
            ovr_lo, ovr_hi = 0.40 + rng.random() * 0.10, 0.58 + rng.random() * 0.12
            tier_name, dev_idx, franchise_flag = "depth", 3, False
        else:
            ovr_lo, ovr_hi = 0.36 + rng.random() * 0.10, 0.52 + rng.random() * 0.12
            tier_name, dev_idx, franchise_flag = "longshot", 4, False
        ovr_lo = max(0.35, min(0.96, float(ovr_lo)))
        ovr_hi = max(ovr_lo + 0.02, min(0.99, float(ovr_hi)))
        pr.draft_value_range = (ovr_lo, ovr_hi)
        pr.id = f"GP_{year}_{bucket[:2]}_{seed_val % 10_000_000}"
        setattr(pr, "_pipeline_potential_tier", tier_name)
        setattr(pr, "_pipeline_franchise_flag", bool(franchise_flag))
        w_fast = [0.26, 0.46, 0.20, 0.08] if tier_name == "elite" else [0.20, 0.50, 0.22, 0.08]
        cv = rng.choices(["fast", "normal", "slow", "boom_bust"], weights=w_fast, k=1)[0]
        if tier_name == "longshot" and rng.random() < 0.24:
            cv = "boom_bust"
        setattr(pr, "_pipeline_dev_curve", cv)
        setattr(pr, "_pipeline_ceiling", float(ovr_hi))
        setattr(pr, "_pipeline_floor", float(ovr_lo))
        pr.development_years_remaining = int({0: 2, 1: 2, 2: 3, 3: 3, 4: 4}[dev_idx] + rng.randint(0, 1))
        self._assign_prospect_dev_archetype(pr, rng, tier_name, 0.5 * (ovr_lo + ovr_hi), bool(franchise_flag), cv)
        arch0 = str(getattr(pr, "_dev_archetype", "") or "")
        if tier_name in ("longshot", "depth") and rng.random() < 0.24:
            setattr(pr, "_scouting_visibility_factor", float(rng.uniform(0.72, 0.90)))
            setattr(pr, "_hidden_gem_candidate", True)
        elif arch0 == "LATE_BLOOMER" and rng.random() < 0.35:
            setattr(pr, "_scouting_visibility_factor", float(rng.uniform(0.78, 0.92)))
            setattr(pr, "_hidden_gem_candidate", True)
        elif rng.random() < 0.10:
            setattr(pr, "_scouting_visibility_factor", float(rng.uniform(0.84, 0.94)))
        else:
            setattr(pr, "_scouting_visibility_factor", float(rng.uniform(0.96, 1.07)))
        return pr

    def _bootstrap_initial_global_player_pool(self, rng: random.Random, year: int) -> None:
        """One-time ~1000 players: 35% junior / 35% minor / 25% Europe / 5% unsigned."""
        if getattr(self.league, "_global_player_pool_bootstrapped", False):
            return
        gp = getattr(self.league, "global_player_pool", None)
        if not isinstance(gp, list):
            self.league.global_player_pool = []
            gp = self.league.global_player_pool
        if len(gp) >= int(self.GLOBAL_POOL_MIN):
            return
        gp.clear()
        specs: List[Tuple[str, int]] = []
        for _ in range(350):
            specs.append(("JUNIOR", int(rng.randint(15, 19))))
        for _ in range(350):
            specs.append(("MINOR_LEAGUE", int(rng.randint(19, 28))))
        for _ in range(250):
            specs.append(("EUROPE", int(rng.randint(18, 30))))
        for _ in range(50):
            specs.append(("UNSIGNED", int(rng.randint(21, 28))))
        rng.shuffle(specs)
        for bucket, ag in specs:
            gp.append(self._global_pool_bootstrap_prospect(rng, year, age=ag, bucket=bucket))

    def _histogram_global_player_pool(self) -> Dict[str, int]:
        hist: Dict[str, int] = {"JUNIOR": 0, "MINOR_LEAGUE": 0, "EUROPE": 0, "UNSIGNED": 0}
        gp = getattr(self.league, "global_player_pool", None) or []
        for p in gp:
            if getattr(p, "team_id", None):
                continue
            b = str(getattr(p, "_global_league_bucket", "JUNIOR") or "JUNIOR").upper()
            if b in hist:
                hist[b] = hist.get(b, 0) + 1
        return hist

    def _advance_global_prospect_season(self, year: int, rng: random.Random) -> Dict[str, int]:
        """Age, develop, cull, small junior intake; stable 900–1200. No refill-to-target spikes."""
        gp = getattr(self.league, "global_player_pool", None)
        if not isinstance(gp, list):
            self.league.global_player_pool = []
            gp = self.league.global_player_pool

        if not getattr(self.league, "_global_player_pool_bootstrapped", False):
            self._bootstrap_initial_global_player_pool(rng, year)
            setattr(self.league, "_global_player_pool_bootstrapped", True)
            hist0 = self._histogram_global_player_pool()
            try:
                tot = sum(hist0.values())
                self._pipeline_log_buffer.append(
                    "GLOBAL PLAYER REPORT (bootstrap): "
                    f"total_world={tot} junior={hist0.get('JUNIOR', 0)} minor={hist0.get('MINOR_LEAGUE', 0)} "
                    f"europe={hist0.get('EUROPE', 0)} unsigned={hist0.get('UNSIGNED', 0)}"
                )
            except Exception:
                pass
            try:
                self.ensure_prospect_pipeline_depth(year, rng)
            except Exception:
                pass
            return hist0

        arch_map: Dict[str, str] = {}
        try:
            setattr(self.league, "_global_pool_spike_cap", max(22, min(78, 20 + len(gp) // 48)))
            setattr(self.league, "_global_pool_spike_count", 0)
        except Exception:
            pass

        nxt: List[Any] = []
        hist: Dict[str, int] = {"JUNIOR": 0, "MINOR_LEAGUE": 0, "EUROPE": 0, "UNSIGNED": 0}
        j_pre_mids: List[float] = []
        j_post_mids: List[float] = []
        eu_pre_mids: List[float] = []
        eu_post_mids: List[float] = []

        for p in list(gp):
            if getattr(p, "team_id", None):
                continue
            if str(getattr(p, "status", "") or "") != "global":
                setattr(p, "status", "global")
            try:
                dr0 = getattr(p, "draft_value_range", (0.5, 0.55))
                pre_mid = (float(dr0[0]) + float(dr0[1])) / 2.0 if dr0 and len(dr0) >= 2 else 0.5
            except (TypeError, ValueError, IndexError):
                pre_mid = 0.5
            try:
                p.age = int(getattr(p, "age", 17) or 17) + 1
            except Exception:
                continue
            ag = int(getattr(p, "age", 17) or 17)
            setattr(p, "_years_to_draft_eligibility", max(0, 18 - ag))
            if ag >= 34:
                continue
            tier_cull = str(getattr(p, "_pipeline_potential_tier", "mid") or "mid").lower()
            try:
                dr = getattr(p, "draft_value_range", (0.5, 0.5))
                mid = (float(dr[0]) + float(dr[1])) / 2.0 if dr and len(dr) >= 2 else 0.5
            except (TypeError, ValueError, IndexError):
                mid = 0.5
            if tier_cull == "longshot" and ag >= 21 and mid < 0.44 and rng.random() < 0.18:
                continue
            if float(getattr(p, "_bust_pressure", 0) or 0) >= 0.91 and rng.random() < 0.26:
                continue

            b = str(getattr(p, "_global_league_bucket", "JUNIOR") or "JUNIOR").upper()
            if b == "JUNIOR" and ag >= 18:
                b = str(rng.choice(["MINOR_LEAGUE", "MINOR_LEAGUE", "EUROPE"]))
            elif b == "MINOR_LEAGUE" and ag >= 24 and rng.random() < 0.38:
                b = "UNSIGNED"
            elif b == "EUROPE" and ag >= 30 and rng.random() < 0.25:
                b = "UNSIGNED"
            setattr(p, "_global_league_bucket", b)
            if b == "EUROPE" and rng.random() < 0.06:
                setattr(p, "_pipeline_dev_curve", "slow")
            if b == "JUNIOR":
                j_pre_mids.append(pre_mid)
            if b == "EUROPE":
                eu_pre_mids.append(pre_mid)
            self._develop_prospect_one_year(p, None, rng, arch_map)
            try:
                dr1 = getattr(p, "draft_value_range", dr0)
                post_mid = (float(dr1[0]) + float(dr1[1])) / 2.0 if dr1 and len(dr1) >= 2 else pre_mid
            except (TypeError, ValueError, IndexError):
                post_mid = pre_mid
            if b == "JUNIOR":
                j_post_mids.append(post_mid)
            if b == "EUROPE":
                eu_post_mids.append(post_mid)
            nxt.append(p)
            hist[b] = hist.get(b, 0) + 1

        self.league.global_player_pool = nxt

        n_in = int(rng.randint(int(self.GLOBAL_YEARLY_INTAKE_MIN), int(self.GLOBAL_YEARLY_INTAKE_MAX)))
        for _ in range(n_in):
            self.league.global_player_pool.append(
                self._global_pool_bootstrap_prospect(rng, year, age=int(rng.randint(15, 17)), bucket="JUNIOR")
            )
            hist["JUNIOR"] = hist.get("JUNIOR", 0) + 1

        shortage = int(self.GLOBAL_POOL_MIN) - len(self.league.global_player_pool)
        if shortage > 0:
            cap_fill = min(shortage, 200)
            for _ in range(cap_fill):
                self.league.global_player_pool.append(
                    self._global_pool_bootstrap_prospect(rng, year, age=int(rng.randint(15, 17)), bucket="JUNIOR")
                )
                hist["JUNIOR"] = hist.get("JUNIOR", 0) + 1

        while len(self.league.global_player_pool) > int(self.GLOBAL_POOL_HARD_MAX):
            try:
                self.league.global_player_pool.sort(
                    key=lambda x: sum(getattr(x, "draft_value_range", (0.5, 0.5))[:2]) / 2.0
                    if getattr(x, "draft_value_range", None)
                    else 0.5
                )
                self.league.global_player_pool.pop(0)
            except Exception:
                self.league.global_player_pool.pop()

        j_gain = (
            (sum(j_post_mids) - sum(j_pre_mids)) / max(1, len(j_pre_mids)) if j_pre_mids else 0.0
        )
        eu_gain = (
            (sum(eu_post_mids) - sum(eu_pre_mids)) / max(1, len(eu_pre_mids)) if eu_pre_mids else 0.0
        )

        try:
            self._pipeline_log_buffer.append(
                "DEVELOPMENT REPORT (global): "
                f"junior_avg_skill_delta~{j_gain:+.4f} (n={len(j_pre_mids)}) "
                f"europe_avg_skill_delta~{eu_gain:+.4f} (n={len(eu_pre_mids)}) "
                f"yearly_intake_juniors={n_in}"
            )
        except Exception:
            pass
        try:
            setattr(self.league, "_last_global_junior_skill_delta", float(j_gain))
            setattr(self.league, "_last_global_europe_skill_delta", float(eu_gain))
            setattr(self.league, "_last_global_yearly_intake", int(n_in))
        except Exception:
            pass
        return hist

    def _select_global_draft_class(self, rng: random.Random, year: int, target_n: int) -> List[Prospect]:
        _ = year
        gp = getattr(self.league, "global_player_pool", None) or []
        want = max(int(self.DRAFT_CLASS_SIZE_MIN), min(int(self.DRAFT_CLASS_SIZE_MAX), int(target_n)))
        elig: List[Prospect] = []
        for p in list(gp):
            if getattr(p, "team_id", None):
                continue
            if str(getattr(p, "status", "") or "") != "global":
                continue
            ag = int(getattr(p, "age", 0) or 0)
            if 18 <= ag <= 21:
                elig.append(p)
        if len(elig) < 28:
            return []
        take = min(len(elig), want)
        elig.sort(
            key=lambda x: sum(getattr(x, "draft_value_range", (0.55, 0.6))[:2]) / 2.0,
            reverse=True,
        )
        head = max(1, int(take * 0.12))
        mid = int(take * 0.38)
        tail = max(0, take - head - mid)
        bucketed: List[Prospect] = []
        bucketed.extend(elig[:head])
        if mid > 0 and len(elig) > head + mid:
            lo = max(head, len(elig) // 2 - mid // 2)
            bucketed.extend(elig[lo : lo + mid])
        if tail > 0:
            pool_tail = elig[-max(tail * 3, tail + 8) :]
            rng.shuffle(pool_tail)
            bucketed.extend(pool_tail[:tail])
        seen: Set[str] = set()
        out: List[Prospect] = []
        for p in bucketed:
            pid = str(getattr(p, "id", ""))
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
        while len(out) < take and len(out) < len(elig):
            c = rng.choice(elig)
            pid = str(getattr(c, "id", ""))
            if pid not in seen:
                seen.add(pid)
                out.append(c)
        return out

    def _global_unsigned_signing_wave(self, rng: random.Random, year: int) -> int:
        teams = getattr(self.league, "teams", None) or []
        gp = getattr(self.league, "global_player_pool", None) or []
        cap = int(self.NHL_PROSPECT_PIPELINE_CAP)
        if not teams or not isinstance(gp, list):
            return 0
        signed = 0
        eu_unsigned = [
            p
            for p in gp
            if str(getattr(p, "_global_league_bucket", "")).upper() in ("EUROPE", "UNSIGNED")
            and not getattr(p, "team_id", None)
            and str(getattr(p, "status", "")) == "global"
            and float(getattr(p, "_steal_momentum", 0) or 0) >= 0.46
            and int(getattr(p, "age", 99) or 99) <= 28
        ]
        late_bloom = [
            p
            for p in gp
            if str(getattr(p, "_global_league_bucket", "")).upper() in ("MINOR_LEAGUE", "EUROPE")
            and not getattr(p, "team_id", None)
            and str(getattr(p, "status", "")) == "global"
            and 22 <= int(getattr(p, "age", 0) or 0) <= 26
            and str(getattr(p, "_pipeline_potential_tier", "mid") or "mid").lower()
            in ("mid", "depth", "longshot", "high")
            and float(getattr(p, "_steal_momentum", 0) or 0) >= 0.40
        ]
        merged: Dict[int, Any] = {}
        for p in eu_unsigned + late_bloom:
            merged[id(p)] = p
        late_ids = {id(p) for p in late_bloom}
        candidates = list(merged.values())
        rng.shuffle(candidates)
        n_late = 0
        for p in candidates[: min(26, len(candidates))]:
            if rng.random() > 0.40:
                continue
            team = rng.choice(teams)
            tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
            if not tid:
                continue
            pool = getattr(team, "prospect_pool", None)
            if pool is None:
                team.prospect_pool = []
                pool = team.prospect_pool
            if len(pool) >= cap:
                continue
            if p not in gp:
                continue
            gp.remove(p)
            p.team_id = tid
            p.status = "prospect"
            setattr(p, "_global_league_bucket", "MINOR_LEAGUE")
            pool.append(p)
            if id(p) in late_ids:
                n_late += 1
            if p not in (self.prospects or []):
                self.prospects.append(p)
            nm = getattr(getattr(p, "identity", None), "name", getattr(p, "name", "?"))
            self._pipeline_log_buffer.append(
                f"SIGNING EVENT: Global pool — {nm} (EU/unsigned/late-bloom path) signed to org {tid} (year {year})"
            )
            signed += 1
        try:
            setattr(self.league, "_last_signing_late_bloom_count", int(n_late))
        except Exception:
            pass
        return signed

    def _count_global_draft_eligibles(self) -> int:
        gp = getattr(self.league, "global_player_pool", None) or []
        n = 0
        for p in gp:
            if getattr(p, "team_id", None):
                continue
            if str(getattr(p, "status", "") or "") != "global":
                continue
            ag = int(getattr(p, "age", 0) or 0)
            if 18 <= ag <= 21:
                n += 1
        return n

    def _emergency_inject_draft_eligibles(self, rng: random.Random, year: int, n: int) -> int:
        """Sustainable repair: add real draft-age world players to global_player_pool (not synthetic draft_class_generator)."""
        gp = getattr(self.league, "global_player_pool", None)
        if not isinstance(gp, list):
            self.league.global_player_pool = []
            gp = self.league.global_player_pool
        added = 0
        cap_extra = 280
        for i in range(max(0, min(int(n), cap_extra))):
            age = int(rng.choice([18, 18, 19, 19, 20]))
            pr = self._global_pool_bootstrap_prospect(rng, year, age=age, bucket="JUNIOR")
            pr.phase = ProspectPhase.DRAFT_YEAR
            pr.status = "global"
            pr.team_id = None
            setattr(pr, "_global_league_bucket", "JUNIOR")
            setattr(pr, "_repair_cohort_marker", True)
            gp.append(pr)
            added += 1
        if added:
            try:
                self._pipeline_log_buffer.append(
                    f"ECOSYSTEM REPAIR: emergency_draft_eligible_injection n={added} year={year}"
                )
            except Exception:
                pass
        return added

    def reclassify_stale_prospect_pipelines(self, rng: random.Random) -> int:
        """Age out eternal 'prospects': fringe org depth vs return to world pool."""
        teams = getattr(self.league, "teams", None) or []
        gpp = getattr(self.league, "global_player_pool", None)
        if not isinstance(gpp, list):
            self.league.global_player_pool = []
            gpp = self.league.global_player_pool
        moves = 0
        for team in teams:
            pool = getattr(team, "prospect_pool", None) or []
            for pr in list(pool):
                try:
                    age = int(getattr(pr, "age", None) or getattr(getattr(pr, "identity", None), "age", 18) or 18)
                    dr = getattr(pr, "draft_value_range", (0.5, 0.5))
                    mid = (float(dr[0]) + float(dr[1])) / 2.0 if dr and len(dr) >= 2 else 0.5
                    tier = str(getattr(pr, "_pipeline_potential_tier", "mid") or "mid").lower()
                except (TypeError, ValueError, IndexError):
                    continue
                if age >= 25 and tier in ("depth", "longshot") and mid < 0.56:
                    setattr(pr, "_prospect_lifecycle_stage", "org_fringe")
                    setattr(pr, "status", "prospect")
                    moves += 1
                if age >= 27 and mid < 0.53:
                    try:
                        pool.remove(pr)
                    except ValueError:
                        continue
                    pr.team_id = None
                    pr.status = "global"
                    setattr(pr, "_global_league_bucket", "UNSIGNED")
                    setattr(pr, "_prospect_lifecycle_stage", "released_org_depth")
                    setattr(pr, "phase", ProspectPhase.STRUCTURED_JUNIOR)
                    if pr not in gpp:
                        gpp.append(pr)
                    if pr in (self.prospects or []):
                        try:
                            self.prospects.remove(pr)
                        except ValueError:
                            pass
                    moves += 1
        if moves:
            try:
                self._pipeline_log_buffer.append(
                    f"PROSPECT LIFECYCLE: stale_pipeline_reclassified_or_returned n={moves}"
                )
            except Exception:
                pass
        return moves

    def ecosystem_operational_repairs(self, teams: Sequence[Any], rng: random.Random, year: int) -> List[str]:
        """
        Operational failsafe: expand draft eligibles / thin pipelines; snapshot population.
        Called from validators and optionally the runner after draft.
        """
        logs: List[str] = []
        tlist = list(teams) if teams is not None else (getattr(self.league, "teams", None) or [])
        gp = getattr(self.league, "global_player_pool", None)
        if not isinstance(gp, list):
            self.league.global_player_pool = []
            gp = self.league.global_player_pool

        elig = self._count_global_draft_eligibles()
        if elig < 52:
            need = min(160, 72 + max(0, 52 - elig) * 2)
            ad = self._emergency_inject_draft_eligibles(rng, year, need)
            logs.append(f"inject_draft_eligibles={ad} (pre_elig~{elig})")
            elig = self._count_global_draft_eligibles()

        pipe = sum(len(getattr(t, "prospect_pool", None) or []) for t in tlist)
        n_act = sum(
            1
            for t in tlist
            for p in getattr(t, "roster", None) or []
            if not getattr(p, "retired", False)
        )
        u20 = u23 = o30 = 0
        for t in tlist:
            for p in getattr(t, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                try:
                    ident = getattr(p, "identity", None)
                    ag = int(getattr(ident, "age", 0) or 0) if ident is not None else int(getattr(p, "age", 0) or 0)
                except (TypeError, ValueError):
                    ag = 26
                if ag < 20:
                    u20 += 1
                if ag < 24:
                    u23 += 1
                if ag >= 30:
                    o30 += 1
        snap = {
            "year": int(year),
            "global_pool": len(gp),
            "draft_eligibles_18_21": elig,
            "pipeline_total": pipe,
            "active_nhl": n_act,
            "roster_u20": u20,
            "roster_u23": u23,
            "roster_30plus": o30,
        }
        try:
            setattr(self.league, "_last_ecosystem_snapshot", snap)
        except Exception:
            pass
        logs.append(
            "PIPELINE_HEALTH active_nhl={active_nhl} global_pool={global_pool} eligibles={draft_eligibles_18_21} "
            "pipeline={pipeline_total} u23_on_rosters={roster_u23}".format(**snap)
        )

        thin = max(8, int(len(tlist) * 1.25))
        pool_floor = int(self.GLOBAL_POOL_MIN) + 24
        if pipe < thin and len(gp) > pool_floor:
            try:
                self.ensure_prospect_pipeline_depth(year, rng)
                logs.append("ensure_prospect_pipeline_depth_repair_run")
            except Exception as ex:
                logs.append(f"pipeline_depth_repair_failed:{type(ex).__name__}")

        if n_act < max(120, 14 * max(1, len(tlist))) and gp:
            logs.append(
                f"WARN:active_players_low n={n_act} (expect~{14 * max(1, len(tlist))}+) — check roster init/retirement balance"
            )

        return logs

    # --------------------------------------------------
    # Prospect generation (yearly draft class pipeline)
    # --------------------------------------------------

    def generate_prospect_class(self, year: int, rng: Optional[random.Random] = None) -> int:
        """
        Entry draft class is SELECTED from the persistent global_player_pool only (200–300).
        No synthetic draft_class_generator path — strength emerges from the living pool.
        """
        r = rng if rng is not None else self.rng
        self.prospects = [p for p in self.prospects if p.phase != ProspectPhase.DRAFT_YEAR]
        self.draft_class = []

        teams = getattr(self.league, "teams", None) or []
        hist = self._advance_global_prospect_season(year, r)
        signing_n = 0
        try:
            signing_n = int(self._global_unsigned_signing_wave(r, year))
        except Exception:
            signing_n = 0

        n_teams = len(teams) if teams else 32
        slots = max(1, n_teams * 7)
        target_n = max(int(self.DRAFT_CLASS_SIZE_MIN), min(int(self.DRAFT_CLASS_SIZE_MAX), slots + 12))
        elig_pre = self._count_global_draft_eligibles()
        if elig_pre < 58:
            self._emergency_inject_draft_eligibles(
                r, year, min(140, 56 + max(0, 58 - elig_pre) * 2)
            )

        global_class = self._select_global_draft_class(r, year, target_n)
        if not global_class:
            self._emergency_inject_draft_eligibles(r, year, min(200, int(self.DRAFT_CLASS_SIZE_MIN) + 24))
            global_class = self._select_global_draft_class(r, year, target_n)

        tot_pop = len(getattr(self.league, "global_player_pool", None) or [])
        try:
            self._pipeline_log_buffer.append(
                f"GLOBAL PLAYER REPORT: total_population={tot_pop} "
                f"junior={hist.get('JUNIOR', 0)} minor={hist.get('MINOR_LEAGUE', 0)} "
                f"europe={hist.get('EUROPE', 0)} unsigned={hist.get('UNSIGNED', 0)}"
            )
            if signing_n:
                lb = int(getattr(self.league, "_last_signing_late_bloom_count", 0) or 0)
                self._pipeline_log_buffer.append(
                    f"SIGNING REPORT: global_pool_signings={signing_n} late_bloom_subpath~{lb}"
                )
        except Exception:
            pass

        if not global_class:
            self._pipeline_log_buffer.append(
                "ECOSYSTEM CRITICAL: draft class still empty after repair attempts — final emergency cohort"
            )
            self._emergency_inject_draft_eligibles(r, year, min(320, slots + 48))
            global_class = self._select_global_draft_class(r, year, target_n)
        if not global_class:
            self._pipeline_log_buffer.append(
                "DRAFT REPORT: fatal insufficient_eligibles even after emergency — check global_player_pool wiring"
            )
            try:
                setattr(self.league, "_last_draft_source", "none")
                setattr(self.league, "_last_draft_class_quality", "empty_pool")
                setattr(self.league, "_last_draft_class_strength_mean10", 0.0)
            except Exception:
                pass
            self._last_draft_class_size = 0
            self._last_draft_class_top_ovr = 0.0
            self._last_draft_class_tier_counts = {k: 0 for k in ("elite", "high", "mid", "depth", "longshot")}
            return 0

        out_ids = {str(getattr(x, "id", "")) for x in global_class}
        self.league.global_player_pool = [
            p
            for p in (getattr(self.league, "global_player_pool", None) or [])
            if str(getattr(p, "id", "")) not in out_ids
        ]
        for p in global_class:
            p.phase = ProspectPhase.DRAFT_YEAR
            try:
                p.lock_draft_year_outputs(estimated_class_size=max(len(global_class), int(self.DRAFT_CLASS_SIZE_MIN)))
            except Exception:
                pass
        self.draft_class = global_class
        for p in global_class:
            if p not in self.prospects:
                self.prospects.append(p)

        top_ovr = 0.0
        hi_sorted: List[float] = []
        mid_sorted: List[float] = []
        tier_counts: Dict[str, int] = {k: 0 for k in ("elite", "high", "mid", "depth", "longshot")}
        for p in global_class:
            lo, hi = getattr(p, "draft_value_range", (0.5, 0.55))
            try:
                flo, fh = float(lo), float(hi)
                mid = 0.5 * (flo + fh)
                mid_sorted.append(mid)
                top_ovr = max(top_ovr, mid)
                hi_sorted.append(fh)
            except Exception:
                mid_sorted.append(0.55)
                hi_sorted.append(0.55)
            tn = str(getattr(p, "_pipeline_potential_tier", "mid") or "mid").lower()
            if tn in tier_counts:
                tier_counts[tn] += 1
        hi_sorted.sort(reverse=True)
        mid_sorted.sort(reverse=True)
        strength_mean_hi = sum(hi_sorted[:10]) / max(1, min(10, len(hi_sorted)))
        strength_mean_mid = sum(mid_sorted[:10]) / max(1, min(10, len(mid_sorted)))
        if top_ovr <= 0.0 and mid_sorted:
            top_ovr = float(mid_sorted[0])
        if strength_mean_hi <= 0.0 and hi_sorted:
            strength_mean_hi = sum(hi_sorted) / max(1, len(hi_sorted))
        if strength_mean_mid <= 0.0 and mid_sorted:
            strength_mean_mid = sum(mid_sorted) / max(1, len(mid_sorted))

        self._last_draft_class_size = len(global_class)
        self._last_draft_class_top_ovr = top_ovr if top_ovr > 0 else max(0.52, strength_mean_mid)
        self._last_draft_class_tier_counts = dict(tier_counts)
        _franchise_n = sum(1 for p in global_class if bool(getattr(p, "_pipeline_franchise_flag", False)))
        try:
            setattr(self.league, "_last_draft_franchise_count", int(_franchise_n))
            setattr(self.league, "_last_draft_class_quality", "global_player_pool")
            setattr(self.league, "_last_draft_source", "global_player_pool")
            setattr(self.league, "_last_draft_class_strength_top", float(top_ovr))
            setattr(self.league, "_last_draft_class_strength_mean10", float(strength_mean_mid))
        except Exception:
            pass
        if not hasattr(self.league, "draft_pool"):
            self.league.draft_pool = []
        self.league.draft_pool = list(global_class)
        self._pipeline_log_buffer.append(
            f"DRAFT REPORT: pool_size={len(global_class)} strength_top_mid~{top_ovr:.3f} "
            f"strength_mean_top10_mid~{strength_mean_mid:.3f} ceiling_mean_top10~{strength_mean_hi:.3f} "
            f"tier_mix elite={tier_counts.get('elite', 0)} high={tier_counts.get('high', 0)} "
            f"mid={tier_counts.get('mid', 0)} depth={tier_counts.get('depth', 0)} "
            f"longshot={tier_counts.get('longshot', 0)}"
        )
        return len(global_class)

    def _pipeline_team_arche(self, team: Any, arch_map: Dict[str, str]) -> str:
        tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
        return str(arch_map.get(tid) or getattr(team, "_runner_team_archetype", "") or "balanced").lower()

    def _assign_prospect_dev_archetype(
        self,
        prospect: Any,
        rng: random.Random,
        tier_name: str,
        mid: float,
        franchise: bool,
        curve: str,
    ) -> None:
        tier = str(tier_name or "mid").lower()
        curve = str(curve or "normal").lower()
        psych = getattr(prospect, "psychology", None)
        risk = getattr(prospect, "risk", None)
        bb = float(getattr(risk, "boom_bust_risk", 0.35) or 0.35)
        dr = float(getattr(risk, "development_risk", 0.35) or 0.35)
        anx = float(getattr(psych, "anxiety", 0.35) or 0.35)
        conf = float(getattr(psych, "confidence", 0.5) or 0.5)
        coach = float(getattr(psych, "coachability", 0.5) or 0.5)
        if tier == "elite":
            opts: List[Tuple[str, float]] = [
                ("ELITE_CEILING_VOLATILE", 0.22 + 0.16 * bb),
                ("HIGH_VARIANCE", 0.16 + 0.06 * anx),
                ("FAST_RISER", 0.20 if curve == "fast" else 0.12),
                ("LATE_BLOOMER", 0.08),
                ("SAFE_LOW_CEILING", 0.04),
                ("STALLED_DEVELOPER", 0.12 + 0.08 * dr),
            ]
        elif tier == "high":
            opts = [
                ("ELITE_CEILING_VOLATILE", 0.12 + 0.10 * bb),
                ("HIGH_VARIANCE", 0.16),
                ("FAST_RISER", 0.16 if curve == "fast" else 0.13),
                ("LATE_BLOOMER", 0.12),
                ("SAFE_LOW_CEILING", 0.08),
                ("STALLED_DEVELOPER", 0.14 + 0.05 * dr),
            ]
        elif tier == "mid":
            opts = [
                ("LATE_BLOOMER", 0.20),
                ("HIGH_VARIANCE", 0.18),
                ("SAFE_LOW_CEILING", 0.09),
                ("FAST_RISER", 0.16),
                ("STALLED_DEVELOPER", 0.15 + 0.05 * dr),
                ("ELITE_CEILING_VOLATILE", 0.10 + 0.04 * bb),
            ]
        elif tier == "depth":
            opts = [
                ("SAFE_LOW_CEILING", 0.16),
                ("STALLED_DEVELOPER", 0.18),
                ("LATE_BLOOMER", 0.16),
                ("HIGH_VARIANCE", 0.14),
                ("FAST_RISER", 0.10),
                ("ELITE_CEILING_VOLATILE", 0.08),
            ]
        else:
            opts = [
                ("LATE_BLOOMER", 0.26),
                ("HIGH_VARIANCE", 0.22),
                ("SAFE_LOW_CEILING", 0.14),
                ("STALLED_DEVELOPER", 0.14),
                ("FAST_RISER", 0.10),
                ("ELITE_CEILING_VOLATILE", 0.10),
            ]
        if curve == "slow":
            opts = [(n, w * (1.32 if n in ("LATE_BLOOMER", "STALLED_DEVELOPER") else 0.9)) for n, w in opts]
        if curve == "boom_bust":
            opts = [(n, w * (1.42 if n in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE") else 0.88)) for n, w in opts]
        if franchise:
            opts = [(n, w * (1.12 if n in ("FAST_RISER", "ELITE_CEILING_VOLATILE") else 1.0)) for n, w in opts]
        ws = sum(w for _, w in opts)
        if ws <= 0:
            ws = 1.0
        names = [n for n, _ in opts]
        weights = [w / ws for _, w in opts]
        arch = str(rng.choices(names, weights=weights, k=1)[0])
        var_p = "MEDIUM"
        cons_p = "MEDIUM"
        if arch in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE"):
            var_p, cons_p = "HIGH", "LOW"
        elif arch == "SAFE_LOW_CEILING":
            var_p, cons_p = "LOW", "HIGH"
        elif arch == "STALLED_DEVELOPER":
            var_p, cons_p = "MEDIUM", "LOW"
        elif arch == "LATE_BLOOMER":
            var_p = "HIGH" if rng.random() < 0.38 else "MEDIUM"
        setattr(prospect, "_dev_archetype", arch)
        setattr(prospect, "_dev_variance_profile", var_p)
        setattr(prospect, "_dev_consistency_profile", cons_p)
        bp = (
            0.07
            + 0.16 * anx
            + (0.11 if tier == "elite" else 0.04)
            - 0.07 * coach
            + 0.05 * (1.0 - conf)
        )
        sm = 0.05 + 0.11 * conf + 0.09 * coach + (0.06 if tier in ("longshot", "depth") else 0.0)
        setattr(prospect, "_bust_pressure", max(0.0, min(0.94, bp)))
        setattr(prospect, "_steal_momentum", max(0.0, min(0.94, sm)))

    def _develop_prospect_one_year(self, prospect: Any, team: Any, rng: random.Random, arch_map: Dict[str, str]) -> None:
        from app.sim_engine.progression import development as _pdev

        mult = 1.0
        try:
            mult *= float(getattr(self.league, "_pipeline_dev_boost_one_year", 1.0) or 1.0)
        except Exception:
            pass
        t_arch = "balanced"
        pscore = 0.52
        if team is None:
            gbuck = str(getattr(prospect, "_global_league_bucket", "MINOR_LEAGUE") or "MINOR_LEAGUE").upper()
            if gbuck == "JUNIOR":
                mult *= 1.15 + rng.uniform(-0.07, 0.11)
            elif gbuck == "EUROPE":
                mult *= 0.83 + rng.uniform(-0.05, 0.05)
            elif gbuck == "UNSIGNED":
                mult *= 0.89 + rng.uniform(-0.05, 0.07)
            elif gbuck == "MINOR_LEAGUE":
                mult *= 1.05 + rng.uniform(-0.06, 0.09)
            else:
                mult *= 0.97 + rng.uniform(-0.05, 0.08)
        else:
            t_arch = self._pipeline_team_arche(team, arch_map)
            if t_arch in ("draft_and_develop", "development"):
                mult *= 1.20
            elif t_arch in ("rebuild", "tank"):
                mult *= 1.10
            elif t_arch in ("win_now", "contend"):
                mult *= 0.88
            elif t_arch in ("chaos_agent", "chaos"):
                mult *= 1.0 + rng.uniform(-0.12, 0.14)
        tier = str(getattr(prospect, "_pipeline_potential_tier", "") or "mid").lower()
        dr = getattr(prospect, "draft_value_range", (0.5, 0.55))
        try:
            lo, hi = float(dr[0]), float(dr[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = 0.5, 0.55
        mid = 0.5 * (lo + hi)
        if tier not in ("elite", "high", "mid", "depth", "longshot"):
            if mid >= 0.82:
                tier = "elite"
            elif mid >= 0.72:
                tier = "high"
            elif mid >= 0.58:
                tier = "mid"
            elif mid >= 0.48:
                tier = "depth"
            else:
                tier = "longshot"
        curve = str(getattr(prospect, "_pipeline_dev_curve", "normal") or "normal").lower()
        cmul = {"fast": 1.35, "normal": 1.0, "slow": 0.62, "boom_bust": 1.05}.get(curve, 1.0)
        if bool(getattr(prospect, "_pipeline_franchise_flag", False)):
            mult *= 1.10
        elif tier == "elite" and mid >= 0.80:
            mult *= 1.07
        if team is not None and t_arch in ("chaos_agent", "chaos"):
            cmul *= 1.0 + rng.uniform(-0.2, 0.22)
        ctx = getattr(prospect, "context", None)
        if ctx is not None:
            try:
                mult *= 0.84 + 0.32 * float(getattr(ctx, "coaching_quality", 0.5) or 0.5)
                mult *= 0.86 + 0.28 * float(getattr(ctx, "ice_time_quality", 0.5) or 0.5)
                mult *= 0.90 + 0.20 * float(getattr(ctx, "competition_level", 0.5) or 0.5)
            except (TypeError, ValueError):
                pass
        if team is not None:
            pscore = float(getattr(team, "prospect_pipeline_score", 0.5) or 0.5)
            mult *= 0.82 + 0.36 * max(0.0, min(1.0, pscore))
        if str(getattr(prospect, "_global_league_bucket", "") or "").upper() == "EUROPE":
            cmul *= 0.93 + rng.uniform(-0.03, 0.03)
        if mult > 1.34:
            mult = 1.34 + (mult - 1.34) ** 0.76
        if tier == "elite":
            base_lo, base_hi = 0.018, 0.048
        elif tier == "high":
            base_lo, base_hi = 0.014, 0.034
        elif tier == "mid":
            base_lo, base_hi = 0.004, 0.019
        elif tier == "depth":
            base_lo, base_hi = 0.002, 0.013
        else:
            base_lo, base_hi = 0.0, 0.010
        growth = rng.uniform(base_lo, base_hi) * mult * cmul
        nm = getattr(prospect, "name", None) or getattr(getattr(prospect, "identity", None), "name", "Prospect")
        label = str(nm)
        d_arch = str(getattr(prospect, "_dev_archetype", "") or "SAFE_LOW_CEILING")
        if not str(getattr(prospect, "_dev_archetype", "") or "").strip():
            self._assign_prospect_dev_archetype(
                prospect, rng, tier, mid, bool(getattr(prospect, "_pipeline_franchise_flag", False)), curve
            )
            d_arch = str(getattr(prospect, "_dev_archetype", "") or "SAFE_LOW_CEILING")
        page = int(getattr(prospect, "age", 18) or 18)
        dev_phase = _pdev._dev_archetype_phase_roll(d_arch, page, curve, rng)
        if team is not None and 20 <= page <= 22 and dev_phase == "STALL" and rng.random() < 0.34:
            dev_phase = "NORMAL"
        if team is None and 17 <= page <= 21 and dev_phase == "STALL" and rng.random() < 0.30:
            dev_phase = "NORMAL"
        if dev_phase == "STALL" and rng.random() < 0.26:
            dev_phase = str(rng.choices(["NORMAL", "SPIKE"], weights=[0.74, 0.26], k=1)[0])
        vp = str(getattr(prospect, "_dev_variance_profile", "MEDIUM") or "MEDIUM").upper()
        if vp == "HIGH" and dev_phase == "NORMAL" and rng.random() < 0.13:
            dev_phase = str(rng.choice(["SPIKE", "STALL", "REGRESSION"]))
        if vp == "LOW" and dev_phase in ("SPIKE", "REGRESSION") and rng.random() < 0.17:
            dev_phase = "NORMAL"
        if team is None:
            spike_cap = int(getattr(self.league, "_global_pool_spike_cap", 48) or 48)
            spike_used = int(getattr(self.league, "_global_pool_spike_count", 0) or 0)
        else:
            spike_cap = int(getattr(self.league, "_pipeline_spike_cap", 40) or 40)
            spike_used = int(getattr(self.league, "_pipeline_spike_count", 0) or 0)
        if dev_phase == "STALL":
            growth *= rng.uniform(0.02, 0.14)
        elif dev_phase == "SPIKE":
            if spike_used < spike_cap:
                growth *= rng.uniform(2.05, 3.85)
                try:
                    if team is None:
                        setattr(self.league, "_global_pool_spike_count", spike_used + 1)
                    else:
                        setattr(self.league, "_pipeline_spike_count", spike_used + 1)
                except Exception:
                    pass
            else:
                growth *= rng.uniform(1.12, 1.62)
        elif dev_phase == "REGRESSION":
            growth = -rng.uniform(0.007, 0.028)
        bp = float(getattr(prospect, "_bust_pressure", 0.08) or 0.08)
        sm = float(getattr(prospect, "_steal_momentum", 0.06) or 0.06)
        if dev_phase == "REGRESSION":
            bp += rng.uniform(0.028, 0.088)
        elif dev_phase == "SPIKE":
            sm += rng.uniform(0.048, 0.118)
        elif dev_phase == "STALL":
            bp += rng.uniform(0.014, 0.048)
        setattr(prospect, "_bust_pressure", max(0.0, min(0.96, bp)))
        setattr(prospect, "_steal_momentum", max(0.0, min(0.96, sm)))
        if dev_phase != "NORMAL" or rng.random() < 0.085:
            self._pipeline_log_buffer.append(
                f"PROSPECT DEVELOPMENT REPORT: {label} archetype={d_arch} growth_phase={dev_phase} "
                f"tier={tier} team_org_arch={t_arch} env_pscore={pscore:.2f} delta_growth={growth:+.4f}"
            )
        if float(getattr(prospect, "_bust_pressure", 0) or 0) >= 0.52 and rng.random() < 0.18:
            self._pipeline_log_buffer.append(
                f"BUST/STEAL TRACKING: {label} bust_pressure={float(getattr(prospect, '_bust_pressure', 0) or 0):.2f} "
                f"steal_momentum={float(getattr(prospect, '_steal_momentum', 0) or 0):.2f}"
            )
        elif float(getattr(prospect, "_steal_momentum", 0) or 0) >= 0.56 and rng.random() < 0.16:
            self._pipeline_log_buffer.append(
                f"BUST/STEAL TRACKING: {label} emerging_steal_signal momentum="
                f"{float(getattr(prospect, '_steal_momentum', 0) or 0):.2f}"
            )
        if curve == "boom_bust" and rng.random() < 0.11:
            growth += rng.uniform(0.008, 0.026)
        ceil = float(getattr(prospect, "_pipeline_ceiling", hi))
        fl = float(getattr(prospect, "_pipeline_floor", lo))
        lo = max(0.35, min(0.97, lo + growth * 0.55))
        hi = max(0.36, min(0.99, hi + growth))
        if hi < lo:
            lo, hi = hi - 0.02, hi
        hi = min(hi, ceil + 0.04)
        lo = max(lo, fl - 0.02)
        if hi < lo:
            lo = hi - 0.03
        prospect.draft_value_range = (lo, hi)

    def ensure_prospect_pipeline_depth(self, year: int, rng: random.Random) -> int:
        """Shallow pools: small pulls from global_player_pool only (no synthetic factory spam)."""
        teams = getattr(self.league, "teams", None) or []
        added = 0
        MIN_DEPTH = 9
        MAX_PULL_PER_TEAM = 9
        cap = int(self.NHL_PROSPECT_PIPELINE_CAP)
        gp = getattr(self.league, "global_player_pool", None) or []
        use_global = bool(getattr(self.league, "_global_player_pool_bootstrapped", False)) and isinstance(gp, list)
        pool_floor_hi = int(self.GLOBAL_POOL_MIN) + 20
        pool_floor_med = 520
        pool_floor_lo = 340
        backfill_teams: List[str] = []
        team_list = list(teams)
        rng.shuffle(team_list)
        team_list.sort(key=lambda t: len(getattr(t, "prospect_pool", None) or []))

        def _pick_from_pool(pool_gp: List[Any]) -> Optional[Any]:
            """Prefer junior (15–17) and over-age org depth (22–24); avoid stripping 18–21 draft eligibles."""
            eligible = [
                p
                for p in pool_gp
                if not getattr(p, "team_id", None)
                and str(getattr(p, "status", "") or "") == "global"
            ]
            if not eligible:
                return None
            young = [p for p in eligible if 15 <= int(getattr(p, "age", 0) or 0) <= 17]
            over = [p for p in eligible if 22 <= int(getattr(p, "age", 0) or 0) <= 24]
            draft_age = [p for p in eligible if 18 <= int(getattr(p, "age", 0) or 0) <= 21]
            if young:
                return rng.choice(young)
            if over:
                return rng.choice(over)
            if draft_age:
                return rng.choice(draft_age)
            return rng.choice(eligible)

        for team in team_list:
            pool = getattr(team, "prospect_pool", None)
            if pool is None:
                team.prospect_pool = []
                pool = team.prospect_pool
            need = max(0, MIN_DEPTH - len(pool))
            if need <= 0:
                continue
            tid = str(getattr(team, "team_id", getattr(team, "id", "T")) or "T")
            backfill_teams.append(tid)
            pulls = 0
            while need > 0 and pulls < MAX_PULL_PER_TEAM and len(pool) < cap:
                if use_global and len(gp) > pool_floor_hi:
                    pr = _pick_from_pool(gp)
                    if pr is None:
                        break
                    try:
                        gp.remove(pr)
                    except ValueError:
                        break
                    try:
                        pr.team_id = tid
                        pr.status = "prospect"
                        pr.phase = ProspectPhase.STRUCTURED_JUNIOR
                        setattr(pr, "_global_league_bucket", "MINOR_LEAGUE")
                        pool.append(pr)
                        if pr not in (self.prospects or []):
                            self.prospects.append(pr)
                        added += 1
                        pulls += 1
                        need -= 1
                    except Exception:
                        break
                else:
                    break
        MIN_FALLBACK = 7
        for team in team_list:
            pool = getattr(team, "prospect_pool", None)
            if pool is None:
                team.prospect_pool = []
                pool = team.prospect_pool
            need = max(0, MIN_FALLBACK - len(pool))
            if need <= 0:
                continue
            tid = str(getattr(team, "team_id", getattr(team, "id", "T")) or "T")
            pulls = 0
            while need > 0 and pulls < MAX_PULL_PER_TEAM and len(pool) < cap:
                if use_global and len(gp) > pool_floor_med:
                    pr = _pick_from_pool(gp)
                    if pr is None:
                        break
                    try:
                        gp.remove(pr)
                    except ValueError:
                        break
                    try:
                        pr.team_id = tid
                        pr.status = "prospect"
                        pr.phase = ProspectPhase.STRUCTURED_JUNIOR
                        setattr(pr, "_global_league_bucket", "MINOR_LEAGUE")
                        pool.append(pr)
                        if pr not in (self.prospects or []):
                            self.prospects.append(pr)
                        added += 1
                        pulls += 1
                        need -= 1
                    except Exception:
                        break
                else:
                    break
        MIN_SOFT = 6
        for team in team_list:
            pool = getattr(team, "prospect_pool", None)
            if pool is None:
                team.prospect_pool = []
                pool = team.prospect_pool
            need = max(0, MIN_SOFT - len(pool))
            if need <= 0:
                continue
            tid = str(getattr(team, "team_id", getattr(team, "id", "T")) or "T")
            pulls = 0
            while need > 0 and pulls < MAX_PULL_PER_TEAM and len(pool) < cap:
                if use_global and len(gp) > pool_floor_lo:
                    pr = _pick_from_pool(gp)
                    if pr is None:
                        break
                    try:
                        gp.remove(pr)
                    except ValueError:
                        break
                    try:
                        pr.team_id = tid
                        pr.status = "prospect"
                        pr.phase = ProspectPhase.STRUCTURED_JUNIOR
                        setattr(pr, "_global_league_bucket", "MINOR_LEAGUE")
                        pool.append(pr)
                        if pr not in (self.prospects or []):
                            self.prospects.append(pr)
                        added += 1
                        pulls += 1
                        need -= 1
                    except Exception:
                        break
                else:
                    break
        if backfill_teams and added:
            self._pipeline_log_buffer.append(
                f"PIPELINE STATUS: org depth assist from global_player_pool teams_touched={len(backfill_teams)} "
                f"players_moved={added} (cap_per_team_pull={MAX_PULL_PER_TEAM})"
            )
        return added

    def apply_pipeline_elite_and_retirement_pass(self, year: int, rng: random.Random) -> None:
        league = self.league
        teams = getattr(league, "teams", None) or []
        elite_n = 0
        elite_85 = 0
        for team in teams:
            for p in getattr(team, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                try:
                    fn = getattr(p, "ovr", None)
                    ov = float(fn()) if callable(fn) else float(fn)
                except Exception:
                    ov = 0.5
                ovn = ov / 99.0 if ov > 1.5 else ov
                if ovn >= 0.88:
                    elite_n += 1
                if ovn >= 0.85:
                    elite_85 += 1
        if elite_85 < 20:
            need = max(1, 20 - elite_85)
            for team in teams:
                pool = list(getattr(team, "prospect_pool", None) or [])
                pool.sort(
                    key=lambda x: sum(getattr(x, "draft_value_range", (0.5, 0.5))[:2]) / 2.0
                    if getattr(x, "draft_value_range", None)
                    else 0.5,
                    reverse=True,
                )
                for pr in pool[:6]:
                    dr = getattr(pr, "draft_value_range", (0.5, 0.55))
                    try:
                        lo, hi = float(dr[0]), float(dr[1])
                    except Exception:
                        continue
                    bump = rng.uniform(0.014, 0.034) * (1.0 + 0.04 * need)
                    lo = max(0.4, min(0.92, lo + bump * 0.5))
                    hi = max(lo + 0.02, min(0.97, hi + bump))
                    pr.draft_value_range = (lo, hi)
                    setattr(pr, "_pipeline_ceiling", max(float(getattr(pr, "_pipeline_ceiling", hi)), hi))
            self._pipeline_log_buffer.append(
                f"PIPELINE HEALTH: elite talent correction applied (league_elite_88plus={elite_n})"
            )
        ret = int(getattr(league, "_runner_retirements_this_year", 0) or 0)
        last_pro = int(getattr(self, "_last_promotion_actual", 0) or 0)
        if ret > last_pro + 4 and teams:
            self._pipeline_log_buffer.append(
                f"PIPELINE HEALTH: retirement replacement boost (retirements={ret} vs last_promotions={last_pro})"
            )
            for team in teams:
                pool = getattr(team, "prospect_pool", None) or []
                ranked = sorted(
                    pool,
                    key=lambda x: sum(getattr(x, "draft_value_range", (0.5, 0.5))[:2]) / 2.0
                    if getattr(x, "draft_value_range", None)
                    else 0.0,
                    reverse=True,
                )
                for pr in ranked[:5]:
                    try:
                        y = int(getattr(pr, "development_years_remaining", 2) or 2)
                        pr.development_years_remaining = max(0, y - 1)
                    except Exception:
                        pass
                    dr = getattr(pr, "draft_value_range", None)
                    if dr and len(dr) >= 2:
                        try:
                            lo, hi = float(dr[0]), float(dr[1])
                            pr.draft_value_range = (
                                max(0.4, lo + 0.006),
                                min(0.97, hi + rng.uniform(0.004, 0.018)),
                            )
                        except Exception:
                            pass
        health = "stable"
        tot = sum(len(getattr(t, "prospect_pool", None) or []) for t in teams)
        cap = max(1, len(teams) * 14)
        if tot < len(teams) * 10:
            health = "thin"
        elif tot > cap * 1.35:
            health = "overloaded"
        setattr(league, "_pipeline_health_label", health)
        self._pipeline_log_buffer.append(
            f"PIPELINE HEALTH: Pipeline balance: {health} (total_prospects={tot})"
        )
        for tm in teams:
            pool = getattr(tm, "prospect_pool", None) or []
            if not pool:
                try:
                    setattr(tm, "prospect_pipeline_score", 0.38)
                except Exception:
                    pass
                continue
            mids: List[float] = []
            for pr in pool:
                dr = getattr(pr, "draft_value_range", None)
                if dr and len(dr) >= 2:
                    try:
                        mids.append((float(dr[0]) + float(dr[1])) / 2.0)
                    except (TypeError, ValueError):
                        pass
            try:
                setattr(tm, "prospect_pipeline_score", sum(mids) / len(mids) if mids else 0.45)
            except Exception:
                pass

    def _trim_team_prospect_pipeline_for_cap(self, team: Any, gpp: List[Any], rng: random.Random) -> None:
        cap = int(self.NHL_PROSPECT_PIPELINE_CAP)
        pool = getattr(team, "prospect_pool", None)
        if pool is None:
            return
        while len(pool) >= cap:

            def _mid(pr: Any) -> float:
                dr = getattr(pr, "draft_value_range", (0.5, 0.5))
                try:
                    return (float(dr[0]) + float(dr[1])) / 2.0
                except (TypeError, ValueError, IndexError):
                    return 0.5

            worst = min(pool, key=_mid)
            pool.remove(worst)
            worst.team_id = None
            worst.status = "global"
            setattr(
                worst,
                "_global_league_bucket",
                str(rng.choice(["MINOR_LEAGUE", "MINOR_LEAGUE", "EUROPE"])),
            )
            if isinstance(gpp, list) and worst not in gpp:
                gpp.append(worst)
            if worst in (self.prospects or []):
                self.prospects.remove(worst)
            nm = str(getattr(getattr(worst, "identity", None), "name", getattr(worst, "name", "?")))
            try:
                self._pipeline_log_buffer.append(
                    f"PIPELINE CAP: {nm} returned to global_player_pool (team_cap={cap})"
                )
            except Exception:
                pass

    def run_universe_draft(
        self,
        non_playoff_teams: list[tuple[str, int]],
        year: int,
        rng: Optional[random.Random] = None,
        standings: Optional[Any] = None,
        full_team_order: Optional[List[str]] = None,
        draft_seed: Optional[int] = None,
    ) -> Tuple[int, List[int], List[float]]:
        """
        Run draft lottery + 7-round draft (if draft class exists), assign prospects to team prospect_pool,
        then always promote ready prospects to rosters.
        Returns (n_promoted, promoted_ages, promoted_potentials).
        """
        r = rng if rng is not None else self.rng
        if self.draft_class or self.prospects:
            if draft_seed is not None:
                seed = int(draft_seed)
            else:
                seed = int(year + int(getattr(self, "seed", 0))) if hasattr(self, "seed") else int(year)
            playoff_team_ids: List[str] = []
            if standings is not None and len(standings) >= 16:
                playoff_team_ids = [getattr(s, "team_id", s) for s in reversed(standings[:16])]
            all_team_ids = {t[0] for t in non_playoff_teams} | set(playoff_team_ids)
            org_dev = {tid: 0.5 for tid in all_team_ids}
            coach_fit = {tid: 0.5 for tid in all_team_ids}
            market_pressure = {tid: 0.5 for tid in all_team_ids}
            self.run_offseason_draft(
                non_playoff_teams=non_playoff_teams,
                org_dev_quality=org_dev,
                coach_fit=coach_fit,
                market_pressure=market_pressure,
                seed=seed,
                playoff_team_ids=playoff_team_ids if playoff_team_ids else None,
                full_team_order=full_team_order,
            )
        self.ensure_prospect_pipeline_depth(year, r)
        self.apply_pipeline_elite_and_retirement_pass(year, r)
        n_promoted, promoted_ages, promoted_potentials = self._run_prospect_promotion(r, year)
        self._prune_rosters(r, max_roster_size=23)
        return (n_promoted, promoted_ages, promoted_potentials)

    def progress_prospects(self) -> None:
        """
        Offseason: age prospects, annual skill growth (tier/curve/org), trim only very deep pools.
        """
        MAX_PROSPECTS = 24
        teams = getattr(self.league, "teams", None) or []
        try:
            self.reclassify_stale_prospect_pipelines(self.rng)
        except Exception:
            pass
        total_pool = sum(len(getattr(t, "prospect_pool", None) or []) for t in teams)
        cap_spikes = max(7, min(58, 7 + total_pool // 38))
        try:
            setattr(self.league, "_pipeline_spike_cap", int(cap_spikes))
            setattr(self.league, "_pipeline_spike_count", 0)
        except Exception:
            pass
        arch_map: Dict[str, str] = {}
        raw_arch = getattr(self.league, "_promotion_team_archetypes", None) or {}
        if isinstance(raw_arch, dict):
            for k, v in raw_arch.items():
                arch_map[str(k)] = str(v).lower()
        r = self.rng
        for team in teams:
            pool = getattr(team, "prospect_pool", None) or []
            for prospect in list(pool):
                try:
                    prospect.age = int(getattr(prospect, "age", 18)) + 1
                    years = int(getattr(prospect, "development_years_remaining", 2) or 0)
                    prospect.development_years_remaining = max(0, years - 1)
                    self._develop_prospect_one_year(prospect, team, r, arch_map)
                except Exception:
                    continue
            pool = getattr(team, "prospect_pool", None) or []
            while len(pool) > MAX_PROSPECTS:
                try:
                    def _potential(p: Any) -> float:
                        dr = getattr(p, "draft_value_range", (0.5, 0.5))
                        if dr and len(dr) >= 2:
                            return (float(dr[0]) + float(dr[1])) / 2.0
                        return 0.5

                    worst = min(pool, key=_potential)
                    pool.remove(worst)
                except Exception:
                    break
        try:
            setattr(self.league, "_pipeline_dev_boost_one_year", 1.0)
        except Exception:
            pass

    def _assign_drafted_rookies_to_rosters(self, rng: random.Random, year: int) -> int:
        """Convert last_draft_results player_payloads into Player entities and add to team rosters."""
        raw = getattr(self, "last_draft_results", None)
        if isinstance(raw, dict):
            results = raw.get("results") or []
        else:
            results = raw or []
        teams = getattr(self.league, "teams", None) or []
        if not teams:
            return 0
        # Build team lookup by ALL common id attributes so draft team_id (from standings) matches
        team_by_id: Dict[str, Any] = {}
        for t in teams:
            for attr in ("team_id", "id", "abbr", "code", "name"):
                v = getattr(t, attr, None)
                if v is not None and str(v).strip():
                    team_by_id[str(v).strip()] = t
        count = 0
        for rec in results:
            payload = rec.get("player_payload")
            team_id = rec.get("team_id")
            if not payload or not team_id:
                continue
            tid = str(team_id).strip()
            team = team_by_id.get(tid) or team_by_id.get(str(team_id))
            if not team:
                continue
            try:
                identity_dict = payload.get("identity") or {}
                proj = payload.get("projection") or {}
                dvr = payload.get("draft_value_range")
                if dvr and isinstance(dvr, (list, tuple)) and len(dvr) >= 2:
                    try:
                        lo, hi = float(dvr[0]), float(dvr[1])
                        if hi > lo:
                            ovr = rng.uniform(lo, hi)
                        else:
                            ovr = float(proj.get("projected_value", 0.55))
                    except (TypeError, ValueError):
                        ovr = float(proj.get("projected_value", 0.55))
                else:
                    ovr = float(proj.get("projected_value", 0.55))
                ovr = max(0.50, min(0.99, ovr))
                base_rating = int(ovr * 99)
                ratings = {k: clamp_rating(base_rating + rng.randint(-2, 2)) for k in ATTRIBUTE_KEYS}
                name = identity_dict.get("name", "Rookie")
                birth_year = int(identity_dict.get("birth_year", year - 18))
                age = year - birth_year
                birth_country = str(identity_dict.get("birth_country", "Canada"))
                birth_city = str(identity_dict.get("birth_city", "Unknown"))
                height_cm = int(identity_dict.get("height_cm", 180))
                weight_kg = int(identity_dict.get("weight_kg", 85))
                pos_val = identity_dict.get("position", "C")
                if hasattr(pos_val, "value"):
                    pos_val = pos_val.value
                pos_val = str(pos_val) if pos_val else "C"
                position = Position(pos_val) if pos_val in ("C", "LW", "RW", "D", "G") else Position.C
                shoots_val = identity_dict.get("shoots", "R")
                if hasattr(shoots_val, "value"):
                    shoots_val = shoots_val.value
                shoots = Shoots.L if str(shoots_val).upper() == "L" else Shoots.R
                identity = IdentityBio(
                    name=name,
                    age=age,
                    birth_year=birth_year,
                    birth_country=birth_country,
                    birth_city=birth_city,
                    height_cm=height_cm,
                    weight_kg=weight_kg,
                    position=position,
                    shoots=shoots,
                    draft_year=year,
                    draft_round=1,
                    draft_pick=1,
                )
                backstory = BackstoryUpbringing(
                    backstory=BackstoryType.GRINDER,
                    upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
                    family_support=SupportLevel.MEDIUM,
                    early_pressure=PressureLevel.MODERATE,
                    dev_resources=DevResources.LOCAL,
                )
                player = Player(
                    identity=identity,
                    backstory=backstory,
                    ratings=ratings,
                    rng_seed=rng.randint(1, 2_000_000_000),
                )
                player.context.current_team_id = str(team_id)
                _arches = [
                    "FAST_RISER",
                    "LATE_BLOOMER",
                    "HIGH_VARIANCE",
                    "SAFE_LOW_CEILING",
                    "STALLED_DEVELOPER",
                    "ELITE_CEILING_VOLATILE",
                ]
                _aw = [0.17, 0.17, 0.14, 0.20, 0.14, 0.18]
                setattr(player, "_dev_archetype", str(rng.choices(_arches, weights=_aw, k=1)[0]))
                setattr(player, "_nhl_adjustment_years_remaining", 2 if age <= 21 else (1 if age <= 23 else 0))
                roster = getattr(team, "roster", None)
                if roster is None:
                    team.roster = []
                    roster = team.roster
                roster.append(player)
                if hasattr(self.league, "players") and self.league.players is not None:
                    self.league.players.append(player)
                count += 1
            except Exception:
                continue
        return count

    def _player_roster_age(self, player: Any) -> int:
        ident = getattr(player, "identity", None)
        if ident is not None:
            try:
                if isinstance(ident, dict):
                    return int(ident.get("age", 0) or 0)
                return int(getattr(ident, "age", 0) or 0)
            except (TypeError, ValueError):
                pass
        try:
            return int(getattr(player, "age", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def compute_league_age_distribution(self, league: Optional[Any] = None) -> Dict[str, Any]:
        """
        Roster-only demographics (non-retired). Percentages are 0–100.
        """
        lg = league if league is not None else self.league
        teams = getattr(lg, "teams", None) or []
        u24 = prime = v30p = 0
        for team in teams:
            for p in getattr(team, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                age = self._player_roster_age(p)
                if age < 24:
                    u24 += 1
                elif age <= 29:
                    prime += 1
                else:
                    v30p += 1
        total = u24 + prime + v30p
        if total <= 0:
            return {
                "total_players": 0,
                "count_under_24": 0,
                "count_prime": 0,
                "count_30_plus": 0,
                "pct_under_24": 0.0,
                "pct_prime": 0.0,
                "pct_30_plus": 0.0,
            }
        return {
            "total_players": total,
            "count_under_24": u24,
            "count_prime": prime,
            "count_30_plus": v30p,
            "pct_under_24": 100.0 * u24 / total,
            "pct_prime": 100.0 * prime / total,
            "pct_30_plus": 100.0 * v30p / total,
        }

    def apply_age_balance_youth_development(self, rng: random.Random) -> None:
        """
        Soft acceleration for youth when league under-24 share is low (runner-triggered).
        Does not spawn or delete players.
        """
        league = self.league
        teams = getattr(league, "teams", None) or []
        if not teams:
            return
        strength = float(getattr(league, "_age_balance_dev_strength", 0.0) or 0.0)
        if strength <= 0.0:
            return
        for team in teams:
            for p in getattr(team, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                age = self._player_roster_age(p)
                if 20 <= age <= 23:
                    if rng.random() >= min(0.45, 0.12 + 0.22 * strength):
                        continue
                    ratings = getattr(p, "ratings", None)
                    if isinstance(ratings, dict) and ratings:
                        keys = list(ratings.keys())
                        for _ in range(min(2, len(keys))):
                            k = rng.choice(keys)
                            ratings[k] = clamp_rating(int(ratings[k]) + rng.randint(0, 1))
            for prospect in getattr(team, "prospect_pool", None) or []:
                if rng.random() >= min(0.40, 0.08 + 0.18 * strength):
                    continue
                try:
                    y = int(getattr(prospect, "development_years_remaining", 0) or 0)
                    if y > 0:
                        prospect.development_years_remaining = max(0, y - 1)
                except Exception:
                    pass

    def _compute_promotion_target(self, league: Any) -> Tuple[int, str]:
        teams = getattr(league, "teams", None) or []
        ret_raw = getattr(league, "_runner_retirements_this_year", None)
        if ret_raw is None:
            ret = 36
        else:
            try:
                ret = int(ret_raw)
            except (TypeError, ValueError):
                ret = 36
        gaps = 0
        for team in teams:
            roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
            n = len(roster)
            if n < 23:
                gaps += 23 - n
        promo = getattr(league, "_age_balance_promotion", None) or {}
        pu = float(promo.get("pct_u24", 25.0))
        age_push = int(round((24.0 - pu) * 1.0))
        age_push = max(-8, min(10, age_push))
        raw = ret + gaps + age_push
        target = int(max(44, min(92, raw)))
        return target, "retirements/roster_need"

    def _promote_prospect_to_roster(
        self,
        team: Any,
        prospect: Any,
        year: int,
        rng: random.Random,
        *,
        promoted_ages: List[int],
        promoted_potentials: List[float],
        age: int,
        potential: float,
    ) -> bool:
        MAX_ROSTER = 23
        try:
            payload = prospect.convert_to_player_payload(
                drafted_by_team_id=getattr(prospect, "team_id", ""),
                org_dev_quality=0.5,
                coach_fit=0.5,
                market_pressure=0.5,
            )
            identity_dict = payload.get("identity") or {}
            proj = payload.get("projection") or {}
            dvr = payload.get("draft_value_range")
            if dvr and isinstance(dvr, (list, tuple)) and len(dvr) >= 2:
                try:
                    lo, hi = float(dvr[0]), float(dvr[1])
                    ovr = rng.uniform(lo, hi) if hi > lo else float(proj.get("projected_value", 0.55))
                except (TypeError, ValueError):
                    ovr = float(proj.get("projected_value", 0.55))
            else:
                ovr = float(proj.get("projected_value", 0.55))
            ovr = max(0.50, min(0.99, ovr))
            entry_age = int(year - int(identity_dict.get("birth_year", year - 18)))
            if entry_age <= 20:
                ovr = min(ovr, 0.838 + rng.uniform(0, 0.028))
            elif entry_age <= 22:
                ovr = min(ovr, 0.868 + rng.uniform(0, 0.022))
            elif entry_age <= 24:
                ovr = min(ovr, 0.898 + rng.uniform(0, 0.015))
            base_rating = int(ovr * 99)
            ratings = {k: clamp_rating(base_rating + rng.randint(-2, 2)) for k in ATTRIBUTE_KEYS}
            tid = str(getattr(prospect, "team_id", getattr(team, "team_id", "")))
            name = identity_dict.get("name", "Rookie")
            birth_year = int(identity_dict.get("birth_year", year - 18))
            birth_country = str(identity_dict.get("birth_country", "Canada"))
            birth_city = str(identity_dict.get("birth_city", "Unknown"))
            height_cm = int(identity_dict.get("height_cm", 180))
            weight_kg = int(identity_dict.get("weight_kg", 85))
            pos_val = identity_dict.get("position", "C")
            if hasattr(pos_val, "value"):
                pos_val = pos_val.value
            pos_val = str(pos_val) if pos_val else "C"
            position = Position(pos_val) if pos_val in ("C", "LW", "RW", "D", "G") else Position.C
            shoots_val = identity_dict.get("shoots", "R")
            if hasattr(shoots_val, "value"):
                shoots_val = shoots_val.value
            shoots = Shoots.L if str(shoots_val).upper() == "L" else Shoots.R
            identity = IdentityBio(
                name=name,
                age=year - birth_year,
                birth_year=birth_year,
                birth_country=birth_country,
                birth_city=birth_city,
                height_cm=height_cm,
                weight_kg=weight_kg,
                position=position,
                shoots=shoots,
                draft_year=year,
                draft_round=1,
                draft_pick=1,
            )
            backstory = BackstoryUpbringing(
                backstory=BackstoryType.GRINDER,
                upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
                family_support=SupportLevel.MEDIUM,
                early_pressure=PressureLevel.MODERATE,
                dev_resources=DevResources.LOCAL,
            )
            player = Player(
                identity=identity,
                backstory=backstory,
                ratings=ratings,
                rng_seed=rng.randint(1, 2_000_000_000),
            )
            player.context.current_team_id = tid
            pcv = str(getattr(prospect, "_pipeline_dev_curve", "normal") or "normal")
            if not str(getattr(prospect, "_dev_archetype", "") or "").strip():
                self._assign_prospect_dev_archetype(
                    prospect,
                    rng,
                    str(getattr(prospect, "_pipeline_potential_tier", "mid") or "mid"),
                    float(ovr),
                    bool(getattr(prospect, "_pipeline_franchise_flag", False)),
                    pcv,
                )
            da = str(getattr(prospect, "_dev_archetype", "") or "").strip() or "SAFE_LOW_CEILING"
            setattr(player, "_dev_archetype", da)
            setattr(player, "_pipeline_dev_curve", pcv)
            setattr(player, "_dev_curve_hint", pcv)
            for _attr in ("_dev_variance_profile", "_dev_consistency_profile"):
                if getattr(prospect, _attr, None) is not None:
                    try:
                        setattr(player, _attr, getattr(prospect, _attr))
                    except Exception:
                        pass
            setattr(player, "_bust_pressure", float(getattr(prospect, "_bust_pressure", 0.08) or 0.08))
            setattr(player, "_steal_momentum", float(getattr(prospect, "_steal_momentum", 0.06) or 0.06))
            a_age = int(year - birth_year)
            adj_y = 2 if a_age <= 21 else (1 if a_age <= 23 else 0)
            early = a_age <= 20
            if early:
                setattr(
                    player,
                    "_bust_pressure",
                    min(0.96, float(getattr(player, "_bust_pressure", 0) or 0) + 0.075),
                )
            setattr(player, "_nhl_adjustment_years_remaining", int(adj_y))
            try:
                self._pipeline_log_buffer.append(
                    f"TRANSITION STATUS: {name} NHL_adjustment_years={adj_y} archetype={da or 'UNSET'} "
                    f"early_promotion_risk={'high' if early else 'normal'}"
                )
            except Exception:
                pass
            try:
                ceil = float(getattr(prospect, "_pipeline_ceiling", 0.0) or 0.0)
                if ceil >= 0.62:
                    po = min(0.99, max(float(ovr), ceil * float(rng.uniform(0.98, 1.05))))
                    setattr(player, "potential", po)
                else:
                    setattr(player, "potential", min(0.99, max(float(ovr) * 1.06, float(ovr) + 0.02)))
            except Exception:
                pass
            roster = getattr(team, "roster", None)
            if roster is None:
                team.roster = []
                roster = team.roster
            if len(roster) >= MAX_ROSTER:
                try:
                    prune_ctx = getattr(self.league, "_age_balance_prune", None) or {}
                    pu = float(prune_ctx.get("pct_u24", 25.0))

                    def _p_ovr(p: Any) -> float:
                        f = getattr(p, "ovr", None)
                        o = float(f()) if callable(f) else 0.5
                        a = self._player_roster_age(p)
                        if pu < 22.0:
                            o -= 0.014 * max(0, a - 28)
                        elif pu > 30.0:
                            o += 0.018 * max(0, 23 - a)
                        return o

                    worst = min(roster, key=_p_ovr)
                    roster.remove(worst)
                    if hasattr(self.league, "players") and self.league.players is not None and worst in (self.league.players or []):
                        self.league.players.remove(worst)
                except Exception:
                    pass
            roster.append(player)
            if hasattr(self.league, "players") and self.league.players is not None:
                self.league.players.append(player)
            pool = getattr(team, "prospect_pool", None) or []
            if prospect in pool:
                pool.remove(prospect)
            promoted_ages.append(age)
            promoted_potentials.append(potential)
            try:
                self._pipeline_log_buffer.append(
                    f"PROMOTION EVENT: Prospect promoted to NHL: {name} (OVR {ovr:.2f})"
                )
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _promotion_age_score(self, age: int) -> float:
        if 22 <= age <= 24:
            return 1.0
        if age == 21 or age == 25:
            return 0.84
        if age == 20 or age == 26:
            return 0.62
        if age == 19:
            return 0.42
        if age >= 27:
            return 0.38
        return 0.35

    def _run_prospect_promotion(
        self,
        rng: random.Random,
        year: int,
        *,
        max_promotions: Optional[int] = None,
    ) -> Tuple[int, List[int], List[float]]:
        """
        League-wide promotion budget from retirements + roster gaps + age balance (30–70).
        Eligible prospects are scored (age, potential, development, projected quality) and
        promoted with team-context weights and a soft stop near the budget to avoid waves.
        """
        _ = max_promotions
        league = self.league
        teams = getattr(league, "teams", None) or []
        if not teams:
            diag = {"target": 0, "target_eff": 0, "actual": 0, "reason": "retirements/roster_need", "cap_applied": False}
            setattr(league, "_last_promotion_control", diag)
            setattr(self, "_last_promotion_control", diag)
            return 0, [], []

        pipe_total = 0
        for t in teams:
            pipe_total += len(getattr(t, "prospect_pool", None) or [])

        promo_ctx = getattr(league, "_age_balance_promotion", None) or {}
        pct_u24 = float(promo_ctx.get("pct_u24", 25.0))
        target, reason = self._compute_promotion_target(league)
        if pct_u24 < 16.0:
            target = int(max(int(target), 28 + int(max(0.0, 16.0 - pct_u24) * 2.8)))
        if pct_u24 < 12.0:
            target = int(max(int(target), 40))
        target_eff = min(target, pipe_total)
        promo_cycles = int(getattr(league, "_promotion_cycles_completed", 0) or 0)
        first_wave = promo_cycles < 1
        arch_map: Dict[str, str] = {}
        raw_arch = getattr(league, "_promotion_team_archetypes", None) or {}
        if isinstance(raw_arch, dict):
            for k, v in raw_arch.items():
                arch_map[str(k)] = str(v).lower()

        candidates: List[Dict[str, Any]] = []
        for team in teams:
            tid = str(getattr(team, "team_id", "") or "")
            arch = arch_map.get(tid, "balanced")
            roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
            roster_n = len(roster)
            pool = getattr(team, "prospect_pool", None) or []
            for prospect in list(pool):
                try:
                    years_left = int(getattr(prospect, "development_years_remaining", 2) or 0)
                    age = getattr(prospect, "age", None) or getattr(getattr(prospect, "identity", None), "age", 18) or 18
                    age = int(age)
                    dr = getattr(prospect, "draft_value_range", (0.5, 0.5))
                    potential = (float(dr[0]) + float(dr[1])) / 2.0 if dr and len(dr) >= 2 else 0.55
                    mid_dvr = potential
                    ready = (
                        years_left <= 0
                        or age >= 22
                        or (potential >= 0.80 and age >= 19)
                        or potential >= 0.72
                    )
                    if 22 <= age <= 24 and 0.50 <= potential < 0.68 and rng.random() < 0.40:
                        ready = True
                    if not ready and pct_u24 < 20.0:
                        if (years_left <= 1 and age >= 20 and potential >= 0.72) or (
                            potential >= 0.77 and age >= 19
                        ):
                            ready = True
                    if not ready and pct_u24 < 14.0:
                        if age >= 21 and potential >= 0.54:
                            ready = True
                        if age >= 23 and potential >= 0.49:
                            ready = True
                    if ready and pct_u24 > 30.0:
                        border = years_left > 0 and age < 22 and 0.79 <= potential < 0.84
                        if border and rng.random() > 0.78:
                            continue
                    if not ready:
                        continue
                    age_s = self._promotion_age_score(age)
                    dev_s = 1.0 - min(1.0, float(years_left) / 3.0)
                    base_score = 0.24 * age_s + 0.30 * potential + 0.20 * dev_s + 0.26 * mid_dvr
                    base_score += rng.uniform(0.0, 0.015)
                    cap_tier = str(getattr(team, "cap_pressure_tier", "") or "").lower()
                    if cap_tier in ("high", "critical", "cap_hell") and age <= 22 and 0.52 <= potential < 0.74:
                        base_score *= 1.19
                    if arch in ("rebuild", "tank"):
                        base_score *= 1.14
                    elif arch in ("draft_and_develop", "development"):
                        base_score *= 1.07
                    elif arch in ("chaos",):
                        base_score *= 1.04
                    elif arch in ("win_now", "contend"):
                        if potential < 0.62:
                            base_score *= 0.38
                    if roster_n < 20:
                        base_score *= 1.20
                    elif roster_n < 23:
                        base_score *= 1.07
                    candidates.append(
                        {
                            "team": team,
                            "prospect": prospect,
                            "arch": arch,
                            "score": base_score,
                            "potential": potential,
                            "years_left": years_left,
                            "age": age,
                            "roster_n": roster_n,
                        }
                    )
                except Exception:
                    continue

        seen_ids = {id(c["prospect"]) for c in candidates}
        need_relax = first_wave or (
            pipe_total >= 150
            and target_eff >= 40
            and len(candidates) < max(40, int(0.44 * max(1, target_eff)))
        )
        if need_relax:
            for team in teams:
                tid = str(getattr(team, "team_id", "") or "")
                arch = arch_map.get(tid, "balanced")
                roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
                roster_n = len(roster)
                pool = getattr(team, "prospect_pool", None) or []
                for prospect in list(pool):
                    if id(prospect) in seen_ids:
                        continue
                    try:
                        years_left = int(getattr(prospect, "development_years_remaining", 2) or 0)
                        age = getattr(prospect, "age", None) or getattr(getattr(prospect, "identity", None), "age", 18) or 18
                        age = int(age)
                        dr = getattr(prospect, "draft_value_range", (0.5, 0.5))
                        potential = (float(dr[0]) + float(dr[1])) / 2.0 if dr and len(dr) >= 2 else 0.55
                        mid_dvr = potential
                        if first_wave:
                            relaxed = (
                                (years_left <= 2 and age >= 19 and potential >= 0.44)
                                or (years_left <= 1 and age >= 18 and potential >= 0.47)
                                or (age >= 20 and potential >= 0.49)
                            )
                        else:
                            relaxed = (years_left <= 1 and age >= 20 and potential >= 0.46) or (
                                age >= 21 and potential >= 0.43
                            )
                        if not relaxed:
                            continue
                        if pct_u24 > 30.0:
                            border = years_left > 0 and age < 22 and 0.79 <= potential < 0.84
                            if border and rng.random() > 0.78:
                                continue
                        age_s = self._promotion_age_score(age)
                        dev_s = 1.0 - min(1.0, float(years_left) / 3.0)
                        base_score = 0.20 * age_s + 0.28 * potential + 0.18 * dev_s + 0.24 * mid_dvr
                        base_score += rng.uniform(0.0, 0.012)
                        cap_tier = str(getattr(team, "cap_pressure_tier", "") or "").lower()
                        if cap_tier in ("high", "critical", "cap_hell") and age <= 22 and 0.52 <= potential < 0.74:
                            base_score *= 1.14
                        if arch in ("rebuild", "tank"):
                            base_score *= 1.10
                        elif arch in ("draft_and_develop", "development"):
                            base_score *= 1.05
                        elif arch in ("chaos",):
                            base_score *= 1.03
                        elif arch in ("win_now", "contend"):
                            if potential < 0.62:
                                base_score *= 0.42
                        if roster_n < 20:
                            base_score *= 1.16
                        elif roster_n < 23:
                            base_score *= 1.05
                        candidates.append(
                            {
                                "team": team,
                                "prospect": prospect,
                                "arch": arch,
                                "score": base_score,
                                "potential": potential,
                                "years_left": years_left,
                                "age": age,
                                "roster_n": roster_n,
                            }
                        )
                        seen_ids.add(id(prospect))
                    except Exception:
                        continue

        candidates.sort(key=lambda x: -float(x["score"]))
        max_score = max((float(c["score"]) for c in candidates), default=1.0)

        promoted_ages: List[int] = []
        promoted_potentials: List[float] = []
        count = 0
        stop_due_target = False
        promo_by_team: Dict[str, int] = {}
        per_team_cap = int(self.NHL_PROMOTIONS_PER_TEAM_PER_YEAR)
        if pct_u24 < 16.0:
            per_team_cap = max(per_team_cap, 7)
        if pct_u24 < 12.0:
            per_team_cap = max(per_team_cap, 9)

        def _try_one(item: Dict[str, Any], *, force: bool) -> bool:
            nonlocal count
            tid_promo = str(getattr(item["team"], "team_id", getattr(item["team"], "id", "")) or "")
            if tid_promo and promo_by_team.get(tid_promo, 0) >= per_team_cap:
                return False
            pool = getattr(item["team"], "prospect_pool", None) or []
            if item["prospect"] not in pool:
                return False
            sn = float(item["score"]) / max_score if max_score > 1e-9 else 0.5
            p_accept = 0.38 + 0.50 * (sn**0.92)
            if first_wave:
                p_accept = min(0.94, p_accept + 0.12)
            if pct_u24 < 15.0:
                p_accept = min(0.93, p_accept + 0.12 + max(0.0, (15.0 - pct_u24) * 0.014))
            if item["arch"] in ("win_now", "contend") and float(item["potential"]) < 0.62:
                p_accept *= 0.32 if pct_u24 > 13.0 else 0.52
            tail = max(1, int(0.93 * target_eff))
            if count >= tail:
                p_accept *= 0.58
            if not force and rng.random() >= p_accept:
                return False
            ok = self._promote_prospect_to_roster(
                item["team"],
                item["prospect"],
                year,
                rng,
                promoted_ages=promoted_ages,
                promoted_potentials=promoted_potentials,
                age=int(item["age"]),
                potential=float(item["potential"]),
            )
            if ok:
                count += 1
                if tid_promo:
                    promo_by_team[tid_promo] = promo_by_team.get(tid_promo, 0) + 1
            return bool(ok)

        for item in candidates:
            if target_eff <= 0:
                break
            if count >= target_eff:
                stop_due_target = True
                break
            _try_one(item, force=False)
            if count >= target_eff:
                stop_due_target = True
                break

        fill_floor = int(max(1, round(0.88 * target_eff)))
        if count < fill_floor and target_eff > 0:
            for item in candidates:
                if count >= target_eff:
                    break
                if count >= fill_floor:
                    break
                _try_one(item, force=True)

        pipeline_capped = pipe_total > 0 and target > pipe_total
        cap_applied = bool(
            pipeline_capped or (stop_due_target and len(candidates) > count)
        )
        diag = {
            "target": target,
            "target_eff": target_eff,
            "actual": count,
            "reason": reason,
            "cap_applied": bool(cap_applied),
            "pipe_total": pipe_total,
        }
        setattr(league, "_last_promotion_control", diag)
        setattr(self, "_last_promotion_control", diag)
        try:
            self._last_promotion_actual = int(count)
            setattr(league, "_last_pipeline_promotions", int(count))
            setattr(league, "_last_rookie_entries_via_promotion", int(count))
            setattr(league, "_promotion_cycles_completed", promo_cycles + 1)
        except Exception:
            pass
        return count, promoted_ages, promoted_potentials

    def apply_progression_rebalance(self, rng: random.Random) -> None:
        """
        Runner-visible soft cap on elite counts and damp early-career rating inflation (20–23),
        without touching progression modules.
        """
        league = self.league
        teams = getattr(league, "teams", None) or []
        if not teams:
            return
        elite_cap = 25
        elites: List[Tuple[float, Any]] = []
        elite85: List[Tuple[float, Any]] = []
        band85 = 85.0 / 99.0
        for team in teams:
            for p in getattr(team, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                try:
                    ovr_fn = getattr(p, "ovr", None)
                    ov = float(ovr_fn()) if callable(ovr_fn) else float(ovr_fn)
                except Exception:
                    ov = 0.5
                ovn = ov / 99.0 if ov > 1.5 else float(ov)
                if ovn >= band85:
                    elite85.append((ovn, p))
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 25)) if ident is not None else 25
                if 20 <= age <= 23 and ov >= 0.78:
                    ratings = getattr(p, "ratings", None)
                    if isinstance(ratings, dict):
                        damp = 0.988 + rng.uniform(0.0, 0.006)
                        for k in list(ratings.keys()):
                            ratings[k] = clamp_rating(int(float(ratings[k]) * damp))
                if ov >= 0.88:
                    elites.append((ov, p))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > elite_cap:
            for _, p in elites[elite_cap:]:
                ratings = getattr(p, "ratings", None)
                if isinstance(ratings, dict):
                    for k in list(ratings.keys()):
                        ratings[k] = clamp_rating(int(float(ratings[k]) * 0.982))
        elite85.sort(key=lambda x: -x[0])
        cap85 = 30
        if len(elite85) > cap85:
            damp85 = 0.983 + rng.uniform(0.0, 0.005)
            for _, p in elite85[cap85:]:
                ratings = getattr(p, "ratings", None)
                if isinstance(ratings, dict):
                    for k in list(ratings.keys()):
                        ratings[k] = clamp_rating(int(float(ratings[k]) * damp85))

    def _prune_rosters(self, rng: Optional[random.Random] = None, max_roster_size: int = 23) -> int:
        """
        Enforce roster cap (default 23). Sort by lowest OVR then highest age; remove excess.
        Released players are removed from roster (free agents if league tracks them).
        Returns total number of players removed across all teams.
        """
        r = rng if rng is not None else self.rng
        teams = getattr(self.league, "teams", None) or []
        prune_ctx = getattr(self.league, "_age_balance_prune", None) or {}
        pu = float(prune_ctx.get("pct_u24", 25.0))
        total_released = 0
        for team in teams:
            roster = getattr(team, "roster", None)
            if roster is None:
                continue
            roster = list(roster)
            if len(roster) <= max_roster_size:
                continue
            def _ovr_age(p: Any) -> tuple:
                ovr_fn = getattr(p, "ovr", None)
                ovr_val = float(ovr_fn()) if callable(ovr_fn) else 0.5
                age = self._player_roster_age(p) or 30
                if pu < 22.0:
                    ovr_val -= 0.012 * max(0, age - 29)
                elif pu > 30.0:
                    ovr_val += 0.010 * max(0, 23 - age)
                return (ovr_val, -age)
            roster.sort(key=_ovr_age)
            excess = roster[max_roster_size:]
            for p in excess:
                try:
                    getattr(team, "roster", []).remove(p)
                    total_released += 1
                    if hasattr(self.league, "players") and self.league.players is not None and p in self.league.players:
                        self.league.players.remove(p)
                except (ValueError, AttributeError):
                    pass
        return total_released

    def set_scout_pool(self, scouts: list[ScoutProfile]) -> None:
        self.scout_pool = scouts

    def set_player(self, player: Player) -> None:
        self.player = player

    def set_team(self, team: Team) -> None:
        if team.coach is None:
            raise RuntimeError(
                "SimEngine requires team.coach to be set before sim_year()."
            )

        self.team = team
        self.coach = team.coach

                # ------------------------------------
        # Initialize scouting department for team
        # ------------------------------------
        if self.team.team_id not in self.team_scouting_departments:

            rng = random.Random(self.seed)

            scouts = [
                create_scout(team_id=self.team.team_id, region=Region.OHL, role=ScoutRole.AREA, rng=rng),
                create_scout(team_id=self.team.team_id, region=Region.WHL, role=ScoutRole.AREA, rng=rng),
                create_scout(team_id=self.team.team_id, region=Region.QMJHL, role=ScoutRole.AREA, rng=rng),
                create_scout(team_id=self.team.team_id, region=Region.USHL, role=ScoutRole.AREA, rng=rng),
                create_scout(team_id=self.team.team_id, region=Region.EUROPE, role=ScoutRole.CROSSCHECK, rng=rng),
                create_scout(team_id=self.team.team_id, region=Region.OTHER, role=ScoutRole.HEAD, rng=rng),
            ]

            dept = create_scouting_department(
                team_id=self.team.team_id,
                budget_level=0.6,
                coverage_quality=0.6,
                scouts=scouts,
                rng_seed=self.seed,
            )

            self.team_scouting_departments[self.team.team_id] = dept





    # Attach team coach if present
        if hasattr(team, "coach"):
            self.coach = team.coach


    # --------------------------------------------------
    # League helpers
    # --------------------------------------------------
    def _prospect_to_board_payload(self, p: Prospect) -> dict:
        """
        Adapter: Prospect entity -> dict payload for DraftBoard scoring.
        Safe: uses getattr fallbacks so it won't crash if fields aren't present.
        """
        # Identity
        pid = str(getattr(p, "id", getattr(getattr(p, "identity", None), "id", "")) or "")
        name = str(getattr(getattr(p, "identity", None), "name", getattr(p, "name", f"Prospect_{pid}")))
        pos = str(getattr(getattr(p, "position", None), "value", getattr(p, "position", "C")))

        # Draft "truth-ish" signals (still imperfect – but this is what teams *think*)
        # If your Prospect has draft_value_range like (floor, ceiling) or similar, map it
        dvr = getattr(p, "draft_value_range", None)
        if isinstance(dvr, (list, tuple)) and len(dvr) >= 2:
            floor_sig = float(dvr[0])
            ceiling_sig = float(dvr[1])
        else:
            # fallback: if your Prospect stores other signals
            floor_sig = float(getattr(p, "floor", 0.5))
            ceiling_sig = float(getattr(p, "ceiling", getattr(p, "upside", 0.5)))
        vis = float(getattr(p, "_scouting_visibility_factor", 1.0) or 1.0)
        floor_sig = max(0.30, min(0.98, floor_sig * max(0.86, min(1.08, vis))))
        ceiling_sig = max(0.32, min(0.99, ceiling_sig * max(0.86, min(1.08, vis))))

        # Certainty / variance (if you have it)
        certainty = float(getattr(p, "certainty", getattr(p, "certainty_signal", 0.5)))
        variance = float(getattr(p, "variance", getattr(p, "boom_bust", 0.4)))

        # Optional: readiness / production / tools
        production = float(getattr(p, "production", getattr(p, "points_signal", 0.5)))
        skating = float(getattr(p, "skating", 0.5))
        hockey_iq = float(getattr(p, "hockey_iq", getattr(p, "iq", 0.5)))
        nhl_readiness = float(getattr(p, "nhl_readiness", getattr(p, "readiness", 0.5)))

        # Mentality/personality (IF your Prospect already has these; otherwise safe defaults)
        # If your Prospect stores a "mentality" dict or dataclass, we try to read it.
        ment = getattr(p, "mentality", None)
        coachability = float(getattr(ment, "coachability", 0.5)) if ment else float(getattr(p, "coachability", 0.5))
        work_ethic = float(getattr(ment, "work_ethic", 0.5)) if ment else float(getattr(p, "work_ethic", 0.5))
        resilience = float(getattr(ment, "resilience", 0.5)) if ment else float(getattr(p, "resilience", 0.5))
        leadership = float(getattr(ment, "leadership", 0.5)) if ment else float(getattr(p, "leadership", 0.5))
        volatility = float(getattr(ment, "volatility", 0.5)) if ment else float(getattr(p, "volatility", 0.5))
        entitlement = float(getattr(ment, "entitlement", 0.5)) if ment else float(getattr(p, "entitlement", 0.5))
        consistency = float(getattr(ment, "consistency", 0.5)) if ment else float(getattr(p, "consistency", 0.5))

        # Risk flags (injury etc)
        injury_risk = float(getattr(p, "injury_risk", getattr(getattr(p, "risk", None), "injury_risk", 0.3)))
        off_ice_risk = float(getattr(p, "off_ice_risk", 0.2))

        return {
            "id": pid,
            "name": name,
            "position": pos,
            # "signals"
            "upside": max(0.0, min(1.0, ceiling_sig)),
            "floor": max(0.0, min(1.0, floor_sig)),
            "certainty": max(0.0, min(1.0, certainty)),
            "variance": max(0.0, min(1.0, variance)),
            "production": max(0.0, min(1.0, production)),
            "skating": max(0.0, min(1.0, skating)),
            "hockey_iq": max(0.0, min(1.0, hockey_iq)),
            "nhl_readiness": max(0.0, min(1.0, nhl_readiness)),
            # mentality/personality
            "coachability": max(0.0, min(1.0, coachability)),
            "work_ethic": max(0.0, min(1.0, work_ethic)),
            "resilience": max(0.0, min(1.0, resilience)),
            "leadership": max(0.0, min(1.0, leadership)),
            "volatility": max(0.0, min(1.0, volatility)),
            "entitlement": max(0.0, min(1.0, entitlement)),
            "consistency": max(0.0, min(1.0, consistency)),
            # risk
            "injury_risk": max(0.0, min(1.0, injury_risk)),
            "boom_bust": max(0.0, min(1.0, variance)),
            "off_ice_risk": max(0.0, min(1.0, off_ice_risk)),
        }

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

        base_expect = float(self._estimate_expected_aav())
        exp_aav = adjust_player_demands(
            self.player, self.league, base_contract=base_expect, team=self.team, rng=self.rng
        )

        league = {
            "cap": cap,
            "cap_growth": cap_growth,
            "expected_aav": exp_aav,
            "league_health": float(health.get("health_score", 0.6)),
            "era": (era.get("state") or {}).get("active_era", "unknown"),
            "coach_security": (
        self.coach.job_security if self.coach else 0.5
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
            arch_tm = str(
                getattr(self.team, "_runner_team_archetype", None)
                or getattr(self.team, "team_archetype", None)
                or "balanced"
            ).lower()
            ty = int(getattr(result.contract, "term_years", 1) or 1)
            if arch_tm in ("rebuild", "draft_and_develop") and ty > 4 and self.rng.random() < 0.52:
                ty = max(2, min(4, ty - self.rng.randint(1, 2)))
                try:
                    result.contract.term_years = ty
                except Exception:
                    pass
            inf = calculate_contract_inflation(self.league)
            self.contract_years_left = int(getattr(result.contract, "term_years", ty))
            self.contract_aav = float(result.contract.salary_aav) * float(inf)
            try:
                result.contract.salary_aav = float(self.contract_aav)
            except Exception:
                pass
            cap_for_afford_m = float(cap)
            if cap_for_afford_m > 200.0:
                cap_for_afford_m = cap_for_afford_m / 1_000_000.0
            pay_m = _team_payroll_millions(self.team)
            old_m = _economy_player_cap_hit_millions(self.player)
            new_m = self.contract_aav / 1_000_000.0
            projected_m = pay_m - old_m + new_m
            if cap_for_afford_m > 0 and projected_m > cap_for_afford_m * 1.05:
                room = max(0.0, cap_for_afford_m * 1.05 - (pay_m - old_m))
                trimmed = max(800_000.0, room * 1_000_000.0)
                self.contract_aav = min(self.contract_aav, trimmed)
                try:
                    result.contract.salary_aav = float(self.contract_aav)
                except Exception:
                    pass
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
        
        print(f"Job Security        : {self.coach.job_security:.3f}")


        print("\n[OVR / ATTRIBUTES]")
        print(f"OVR         : {self.player.ovr():.3f}")

        # --------------------------------------------------
        # SEASON STATS (LATEST)
        # --------------------------------------------------
        

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
        print("\n[COACH]")
        print(f"Name                : {self.coach.name}")
        print(f"Job Security        : {self.coach.job_security:.3f}")
        print(f"Risk Tolerance      : {self.coach.tactics.risk_tolerance:.3f}")
        print(f"Pace Preference     : {self.coach.tactics.pace_preference.value}")
        print(f"Lost Room           : {self.coach.lost_room}")
        print(f"Room Temperature    : {self.coach.room_temperature:.3f}")


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
    # Draft orchestration (lottery + selection)
    # --------------------------------------------------

    def _build_draft_team_profile(
        self,
        team_id: str,
        *,
        org_dev_quality: dict[str, float],
        coach_fit: dict[str, float],
        market_pressure: dict[str, float],
    ) -> DraftTeamProfile:
        """
        Analyze a team's roster and return a draft profile (needs, window) for draft board.
        """
        teams = getattr(self.league, "teams", None) or []
        team = None
        for t in teams:
            tid = str(getattr(t, "team_id", getattr(t, "id", "")))
            if tid == str(team_id):
                team = t
                break
        if team is None:
            for t in teams:
                for attr in ("abbr", "code", "name"):
                    if str(getattr(t, attr, "")).strip() == str(team_id).strip():
                        team = t
                        break
                if team is not None:
                    break
        if team is None:
            return DraftTeamProfile(
                team_id=str(team_id),
                name=str(team_id),
                needs_by_position={"C": 0.5, "LW": 0.5, "RW": 0.5, "D": 0.5, "G": 0.5},
                timeline_pressure=0.5,
            )
        name = str(getattr(team, "name", getattr(team, "city", team_id)))
        roster = list(getattr(team, "roster", None) or [])
        forwards = 0
        defense = 0
        goalies = 0
        age_sum = 0
        age_count = 0
        for p in roster:
            pos = None
            ident = getattr(p, "identity", None)
            if ident is not None:
                pos = getattr(ident, "position", None) or (ident.get("position") if isinstance(ident, dict) else None)
            if pos is None:
                pos = getattr(p, "position", None)
            pos_str = str(pos).upper() if pos else ""
            if pos_str in ("C", "LW", "RW"):
                forwards += 1
            elif pos_str == "D":
                defense += 1
            elif pos_str == "G":
                goalies += 1
            age_val = None
            if ident is not None:
                age_val = getattr(ident, "age", None) or (ident.get("age") if isinstance(ident, dict) else None)
            if age_val is None and hasattr(p, "identity"):
                age_val = getattr(getattr(p, "identity", None), "age", None)
            if age_val is not None:
                try:
                    age_sum += int(age_val)
                    age_count += 1
                except (TypeError, ValueError):
                    pass
        need_forward = 0.5
        if forwards < 12:
            need_forward += 0.4
        need_defense = 0.5
        if defense < 6:
            need_defense += 0.4
        need_goalie = 0.5
        if goalies < 2:
            need_goalie += 0.5
        avg_age = (age_sum / age_count) if age_count else 27.0
        age_pressure = (avg_age - 27.0) / 10.0
        bucket = str(getattr(team, "bucket", getattr(team, "status", "bubble"))).lower()
        if "contend" in bucket or "contender" in bucket:
            window = 1.0
        elif "rebuild" in bucket:
            window = 0.2
        else:
            window = 0.5
        needs_by_position = {"C": 0.5, "LW": 0.5, "RW": 0.5, "D": 0.5, "G": 0.5}
        needs_by_position["C"] = min(1.0, 0.5 + need_forward * 0.3)
        needs_by_position["LW"] = min(1.0, 0.5 + need_forward * 0.3)
        needs_by_position["RW"] = min(1.0, 0.5 + need_forward * 0.3)
        needs_by_position["D"] = min(1.0, 0.5 + need_defense * 0.3)
        needs_by_position["G"] = min(1.0, 0.5 + need_goalie * 0.3)
        return DraftTeamProfile(
            team_id=str(team_id),
            name=name,
            needs_by_position=needs_by_position,
            timeline_pressure=window,
        )

    def run_offseason_draft(
        self,
        *,
        non_playoff_teams: list[tuple[str, int]],
        org_dev_quality: dict[str, float],
        coach_fit: dict[str, float],
        market_pressure: dict[str, float],
        seed: int | None = None,
        playoff_team_ids: Optional[List[str]] = None,
        full_team_order: Optional[List[str]] = None,
    ) -> list[dict]:
        """
        Full draft pipeline:
        1) Run NHL-style draft lottery (top 2 picks) OR use runner-provided full_team_order
        2) Build full 32-team draft order: lottery order (16) + playoff teams worst→best (16)
        3) Execute 7-round draft via run_draft()

        non_playoff_teams: [(team_id, season_points), ...] sorted WORST → BEST
        playoff_team_ids: [team_id, ...] worst playoff team first (picks 17–32)
        full_team_order: when set (e.g. from universe lottery), must align with same lottery seed.
        """
        lottery_seed = int(seed if seed is not None else (getattr(self, "seed", 0) + getattr(self, "year", 0)))

        if full_team_order is not None and len(full_team_order) >= 16:
            seen: Set[str] = set()
            team_order: List[str] = []
            for tid in full_team_order:
                s = str(tid).strip()
                if s and s not in seen:
                    seen.add(s)
                    team_order.append(s)
            if playoff_team_ids:
                for p in playoff_team_ids:
                    ps = str(p).strip()
                    if ps and ps not in seen:
                        seen.add(ps)
                        team_order.append(ps)
            while len(team_order) < 32:
                for t in non_playoff_teams:
                    tid = str(t[0]).strip()
                    if tid and tid not in seen:
                        seen.add(tid)
                        team_order.append(tid)
                    if len(team_order) >= 32:
                        break
                if len(team_order) >= 32:
                    break
                if playoff_team_ids:
                    for p in playoff_team_ids:
                        ps = str(p).strip()
                        if ps and ps not in seen:
                            seen.add(ps)
                            team_order.append(ps)
                        if len(team_order) >= 32:
                            break
                if len(team_order) < 32:
                    break
            team_order = team_order[:32]
            lot16 = team_order[:16]
            lottery_result = LotteryResult(
                pick_order=list(lot16),
                lottery_winners=list(lot16[:2]) if len(lot16) >= 2 else list(lot16),
            )
            self.last_draft_lottery = lottery_result
        else:
            lottery_teams = [
                LotteryTeam(team_id=t[0], points=t[1])
                for t in non_playoff_teams
            ]
            lottery_result = run_draft_lottery(
                teams=lottery_teams,
                seed=lottery_seed,
            )
            team_order = list(lottery_result.pick_order)
            if playoff_team_ids:
                team_order = team_order + list(playoff_team_ids)
            self.last_draft_lottery = lottery_result

        draft_results = self.run_draft(
            team_order=team_order,
            org_dev_quality=org_dev_quality,
            coach_fit=coach_fit,
            market_pressure=market_pressure,
        )
        self.last_draft_results = {
            "order": team_order,
            "lottery_winners": lottery_result.lottery_winners,
            "results": draft_results,
        }
        return draft_results

    def run_draft(
        self,
        *,
        team_order: list[str],
        org_dev_quality: dict[str, float],
        coach_fit: dict[str, float],
        market_pressure: dict[str, float],
    ) -> list[dict]:
        """
        Executes a realistic draft:
        - Each team builds its OWN internal DraftBoard
        - Picks are made via board.recommend_pick() using sliding/rumors/runs
        - Prospect conversion stays the same
        - Draft results include pick meta (mood/intent/top5/trade_signal) for storytelling/debug

        NOTE:
        - This replaces the old "sort by consensus and pop(0)" logic.
        """

        # ----------------------------
        # 1) Draft-eligible prospects (use draft_class if prospects list has none)
        # ----------------------------
        eligible: list[Prospect] = [p for p in self.prospects if getattr(p, "phase", None) == ProspectPhase.DRAFT_YEAR]
        if not eligible and getattr(self, "draft_class", None):
            eligible = list(self.draft_class)
        if not eligible:
            self.last_draft_results = []
            return []

        trim_rng = getattr(self, "rng", None) or random.Random()

        # Convert to board payloads once
        dept = self.team_scouting_departments.get(team_order[0])  # example: your team

        payloads = []
        payload_by_id = {}

        for p in eligible:
            pid = str(p.id)

            if dept and pid in dept.team_views:
                tv = dept.team_views[pid]

                payload = {
                    "id": pid,
                    "name": getattr(p, "name", f"Prospect_{pid}"),
                    "position": getattr(p.position, "value", "C"),
                    "upside": tv.ceiling_est[1],
                    "floor": tv.floor_est[1],
                    "certainty": tv.confidence,
                    "variance": 1.0 - tv.confidence,
                    "production": tv.grade,
                    "skating": 0.5,
                    "hockey_iq": 0.5,
                    "nhl_readiness": tv.grade,
                    "coachability": 0.5,
                    "work_ethic": 0.5,
                    "resilience": 0.5,
                    "leadership": 0.5,
                    "volatility": 0.5,
                    "entitlement": 0.5,
                    "consistency": 0.5,
                    "injury_risk": 0.5,
                    "boom_bust": 0.5,
                    "off_ice_risk": 0.2,
                }

            else:
                payload = self._prospect_to_board_payload(p)

            payloads.append(payload)
            payload_by_id[pid] = payload

        prospect_by_id = {}
        for p in eligible:
            pid = str(getattr(p, "id", getattr(getattr(p, "identity", None), "id", "")) or "")
            prospect_by_id[pid] = p

        team_by_id: Dict[str, Any] = {}
        for t in getattr(self.league, "teams", None) or []:
            for attr in ("team_id", "id", "abbr", "code", "name"):
                v = getattr(t, attr, None)
                if v is not None and str(v).strip():
                    team_by_id[str(v).strip()] = t

        # ----------------------------
        # 2) Build DraftBoards (one per team)
        # ----------------------------
        ctx = DraftContext(
            seed=self.seed + self.year,
            year=2025 + self.year,
            recent_picks_window=8,
            iceberg_effect_strength=0.65,
            run_strength=0.55,
        )

        boards: dict[str, DraftBoard] = {}
        for team_id in team_order:
            profile = self._build_draft_team_profile(
                team_id,
                org_dev_quality=org_dev_quality,
                coach_fit=coach_fit,
                market_pressure=market_pressure,
            )
            b = DraftBoard(profile, ctx)
            b.build(payloads)
            boards[team_id] = b

        # ----------------------------
        # 3) Run the draft (7 rounds, snake order)
        # ----------------------------
        drafted_ids: set[str] = set()
        league_events: list[DraftEvent] = []
        results: list[dict] = []
        full_pick_order: list[str] = []
        for round_num in range(7):
            order = team_order if round_num % 2 == 0 else list(reversed(team_order))
            full_pick_order.extend(order)

        for pick_number, team_id in enumerate(full_pick_order, start=1):
            if len(drafted_ids) >= len(payloads):
                break

            board = boards[team_id]

            user_cb = getattr(self, "user_draft_pick_callback", None)
            user_tid = getattr(self, "user_draft_team_id", None)
            use_interactive = (
                user_cb is not None
                and user_tid is not None
                and str(team_id).strip() == str(user_tid).strip()
            )

            def _ai_pick() -> Tuple[Optional[str], dict]:
                return board.recommend_pick(
                    pick_number=pick_number,
                    drafted_ids=drafted_ids,
                    league_events=league_events,
                )

            chosen_id: Optional[str] = None
            meta: dict = {}
            if use_interactive:
                try:
                    user_pid = user_cb(
                        self,
                        str(team_id),
                        pick_number,
                        board,
                        drafted_ids,
                        league_events,
                        prospect_by_id,
                        _ai_pick,
                    )
                except Exception:
                    user_pid = None
                if user_pid is None:
                    chosen_id, meta = _ai_pick()
                else:
                    up = str(user_pid).strip()
                    if up and up not in drafted_ids and up in prospect_by_id:
                        chosen_id = up
                        meta = {"mood": "USER", "intent": "USER_PICK"}
                    else:
                        chosen_id, meta = _ai_pick()
            else:
                chosen_id, meta = _ai_pick()

            # Safety fallback: if something goes weird, take best available by that board
            if not chosen_id or chosen_id in drafted_ids or chosen_id not in payload_by_id:
                avail = board.available(drafted_ids)
                if not avail:
                    break
                chosen_id = avail[0].prospect_id

            drafted_ids.add(chosen_id)

            chosen_payload = payload_by_id[chosen_id]
            prospect = prospect_by_id.get(chosen_id)

            # Another safety fallback (shouldn't happen)
            if prospect is None:
                # find by matching name/id
                # If missing, just skip conversion but still record the pick
                prospect_name = chosen_payload.get("name", "Unknown Prospect")
                event = DraftEvent(
                    pick_number=pick_number,
                    team_id=team_id,
                    prospect_id=chosen_id,
                    prospect_name=prospect_name,
                    note="(missing prospect entity)",
                )
                league_events.append(event)
                # update boards with the pick
                for b in boards.values():
                    b.on_pick_made(event)

                results.append({
                    "pick": pick_number,
                    "team_id": team_id,
                    "prospect_id": chosen_id,
                    "prospect_name": prospect_name,
                    "player_payload": None,
                    "draft_meta": meta,
                })
                continue

            # Convert prospect -> player payload (your existing system)
            payload = prospect.convert_to_player_payload(
                drafted_by_team_id=team_id,
                org_dev_quality=org_dev_quality.get(team_id, 0.5),
                coach_fit=coach_fit.get(team_id, 0.5),
                market_pressure=market_pressure.get(team_id, 0.5),
            )

            # Draft event (feeds iceberg/runs)
            event = DraftEvent(
                pick_number=pick_number,
                team_id=team_id,
                prospect_id=chosen_id,
                prospect_name=str(getattr(getattr(prospect, "identity", None), "name", chosen_payload.get("name", "Unknown"))),
                note=f"{meta.get('mood','')}/{meta.get('intent','')}",
            )
            league_events.append(event)

            # Push pick into every board so their run trackers stay synced
            for b in boards.values():
                b.on_pick_made(event)

            # Remove from active prospects list (engine truth)
            if prospect in self.prospects:
                self.prospects.remove(prospect)
            _gpp = getattr(self.league, "global_player_pool", None)
            if isinstance(_gpp, list) and prospect in _gpp:
                try:
                    _gpp.remove(prospect)
                except ValueError:
                    pass

            # Assign prospect to team pipeline (prospect_pool, development path)
            tid = str(team_id).strip()
            team = team_by_id.get(tid) or team_by_id.get(team_id)
            if team:
                pool = getattr(team, "prospect_pool", None)
                if pool is None:
                    team.prospect_pool = []
                    pool = team.prospect_pool
                prospect.team_id = tid
                prospect.status = "prospect"
                lo, hi = getattr(prospect, "draft_value_range", (0.5, 0.6))
                mid = (float(lo) + float(hi)) / 2.0
                age = getattr(getattr(prospect, "identity", None), "age", 18) or 18
                prospect.development_league = "junior" if age <= 19 else ("AHL" if mid > 0.70 else "development")
                prospect.development_years_remaining = 0 if (age >= 20 and mid > 0.65) else (1 if age >= 19 else 2)
                gpp_live = getattr(self.league, "global_player_pool", None) or []
                if isinstance(gpp_live, list):
                    self._trim_team_prospect_pipeline_for_cap(team, gpp_live, trim_rng)
                pool.append(prospect)

            results.append({
                "pick": pick_number,
                "team_id": team_id,
                "prospect_id": chosen_id,
                "prospect_name": event.prospect_name,
                "player_payload": payload,
                "draft_range": getattr(prospect, "draft_rank_range", None),
                "draft_meta": meta,
            })

        gpp = getattr(self.league, "global_player_pool", None)
        r_pool = getattr(self, "rng", None) or random.Random()
        for p in list(getattr(self, "draft_class", None) or []):
            pid = str(getattr(p, "id", ""))
            if not pid or pid in drafted_ids:
                continue
            try:
                p.phase = ProspectPhase.STRUCTURED_JUNIOR
                p.status = "global"
                p.team_id = None
                setattr(
                    p,
                    "_global_league_bucket",
                    str(r_pool.choice(["MINOR_LEAGUE", "MINOR_LEAGUE", "EUROPE"])),
                )
                if isinstance(gpp, list) and p not in gpp:
                    gpp.append(p)
                if p in self.prospects:
                    self.prospects.remove(p)
                nm = str(getattr(getattr(p, "identity", None), "name", getattr(p, "name", "?")))
                self._pipeline_log_buffer.append(f"GLOBAL POOL: undrafted returned to world pool: {nm}")
            except Exception:
                continue

        self.last_draft_results = results
        return results

    # --------------------------------------------------
    # Waiver Processing
    # --------------------------------------------------
    def process_waiver_player(self, player_payload: dict) -> None:
        """
        Simulates waiving a player and processes league claims.
        """

        if not self.waiver_priority:
            return

        # Build simplified team dictionaries for waiver engine
        team_dicts = []

        for team in self.league.teams:
            team_dicts.append({
                "team_id": str(team.id),
                "points": getattr(team, "points", 0),
                "point_pct": getattr(team, "point_pct", 0.5),
                "goal_diff": getattr(team, "goal_diff", 0),
                "cap_space": getattr(team, "cap_space", 5_000_000),
                "competitive_window": getattr(team, "status", "bubble"),
                "roster_needs": getattr(team, "roster_needs", []),
            })

        winner = self.waiver_engine.process_player(
            player=player_payload,
            teams=team_dicts,
            priority_order=self.waiver_priority,
        )

        if winner:
            self.waiver_priority = update_priority_after_claim(
                self.waiver_priority,
                winner
            )

            if self.debug:
                print(f"\nWAIVER CLAIM: {winner} claimed player.")

        else:
            if self.debug:
                print("\nWAIVER CLEAR: Player cleared waivers.")


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
        tp = 1
        self.season_aging_events = 0
        self.season_player_count = tp
        self.max_aging_events = max(0, int(tp * 0.18))
        if self.season_player_count > 0 and self.max_aging_events < 1:
            self.max_aging_events = 1
        prime_league_season_aging_v3(self.league, tp)
        prime_league_season_breakout_v3(self.league, tp, season_year=int(getattr(self, "year", 0) or 0))
        if self.player is not None:
            try:
                setattr(self.player, "_career_breakout_logged_this_season", False)
                setattr(self.player, "_career_late_bloom_logged_this_season", False)
                setattr(self.player, "progression_event_this_season", None)
                setattr(self.player, "major_progression_event_this_season", None)
            except Exception:
                pass
        try:
            self.season_aging_events = int(getattr(self.league, "_season_aging_events", 0) or 0)
            self.season_player_count = int(getattr(self.league, "_season_player_count", tp) or tp)
            self.max_aging_events = int(getattr(self.league, "_max_aging_events", self.max_aging_events) or 0)
            self.season_breakouts = int(getattr(self.league, "_season_breakout_events", 0) or 0)
            self.max_breakouts = int(getattr(self.league, "_max_season_breakouts", 0) or 0)
        except Exception:
            pass
        # Draft-year locking for eligible prospects
        for p in self.prospects:
            if p.phase == ProspectPhase.DRAFT_YEAR:
                p.lock_draft_year_outputs()

        print("\n==============================")
        print(f"      SIM YEAR {self.year}")
        print("==============================")

        # --------------------------------------------------
        # 0. League macro
        # --------------------------------------------------
        self._advance_league_and_cache()
        league_nudges = self._league_nudges()

                # --------------------------------------------------
        # Initialize waiver priority for season
        # --------------------------------------------------
        league_ctx = self.last_league_context or {}
        season_ctx = {
            "day": 1,
            "standings_current": getattr(self.league, "teams", []),
        }

        self.waiver_priority = self.waiver_engine.build_priority(
            league_context={
                "standings_prev": getattr(self.league, "teams", [])
            },
            season_context=season_ctx,
        )

        # --------------------------------------------------
        # 0A. Prospect development year (PRE-NHL)
        # --------------------------------------------------
        for p in self.prospects:
            # Only simulate prospects not yet drafted
            if p.phase != ProspectPhase.DRAFT_YEAR:
                p.step_year()
                # --------------------------------------------------
        # 0B. Weekly scouting simulation (NEW)
        # --------------------------------------------------
        dept = self.team_scouting_departments.get(self.team.team_id)


        if dept:
            # simulate 26 scouting ticks per season
            for week in range(1, 27):
                league_snapshot = LeagueContextSnapshot(
                    season=2025 + self.year,
                    week=week,
                    active_era="modern_offense",
                    league_health=0.6,
                )

                update_scouting(
                    dept=dept,
                    prospects=self.prospects,
                    league_ctx=league_snapshot,
                    week=week,
                )



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
        # Coach season tick (ACTIVE)
        # --------------------------------------------------
        coach_results = {
            "points_pct": win_pct,
            "made_playoffs": 1.0 if win_pct >= 0.55 else 0.0,
            "player_conflicts": float(team_ctx.get("player_conflicts", 0.0)),
            "media_pressure": float(team_ctx.get("market_pressure", 0.5)),
        }

        self.coach_last_season = self.coach.year_tick(
            rng=self.rng,
            market_tag=str(getattr(self.team, "market_tag", "")),
            team_status=str(getattr(self.team, "status", "bubble")),
            owner_expectations=float(getattr(self.team, "owner_expectations", 0.55)),
            results=coach_results,
        )


        # --------------------------------------------------
# Coach evaluation (seasonal)
# --------------------------------------------------
        


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
        league_morale_pressure = (
            float(league_nudges["morale_volatility_mod"]) - 1.0
            + (self.coach.tactics.volatility_factor() * 0.35)
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
                    + ((self.coach.room_temperature - 0.5) * 0.12)

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
# 5. Aging / development (coach-aware)
# --------------------------------------------------

        coach_dev = self.coach.development_effects_for_player(
            player_id=str(self.player.id),
            player_age=int(self.player.age),
            player_personality=vars(self.player.traits),
            in_role_fit=float(team_ctx.get("role_fit", 0.55)),
            rng=self.rng,
        )

        self.player.advance_year(
            season_morale=self.morale.overall(),
            season_injury_risk=self.injury_risk.total_risk,
            team_instability=1.0 - float(team_ctx.get("stability", 0.5)),
            development_modifier=coach_dev["skill_growth_multiplier"] - 1.0,
        )

        try:
            run_career_lifecycle_for_player(
                self.player,
                self.rng,
                do_print=True,
                log_emit=None,
                verbose_main_line=True,
                league=getattr(self, "league", None),
                skip_base_progress=False,
                season_year=int(getattr(self, "year", 0) or 0),
            )
        except Exception:
            pass
        try:
            _teams = list(getattr(getattr(self, "league", None), "teams", None) or [])
            if _teams:
                apply_league_ovr_soft_regression_if_needed(_teams, self.rng)
        except Exception:
            pass

                            # Example: random waiver test for depth player
        if self.player.ovr() < 0.42 and self.rng.random() < 0.15:
            waiver_payload = {
                "position": self.player.position.value,
                "age": self.player.age,
                "cap_hit": 1_200_000,
                "contract_years_left": 1,
                "overall_projection": self.player.ovr(),
            }

            self.process_waiver_player(waiver_payload)



                # --------------------------------------------------
        # OFFSEASON: Draft (lottery + selection)
        # --------------------------------------------------
        if self.draft_class:
            non_playoff = []

            for team in self.league.teams:
                if not getattr(team, "made_playoffs", False):
                    non_playoff.append(
                        (team.id, int(getattr(team, "points", 0)))
                    )

            # Sort WORST → BEST
            non_playoff.sort(key=lambda x: x[1])

            self.run_offseason_draft(
                non_playoff_teams=non_playoff,
                org_dev_quality={t[0]: 0.5 for t in non_playoff},
                coach_fit={t[0]: 0.5 for t in non_playoff},
                market_pressure={t[0]: 0.5 for t in non_playoff},
            )
            


        # --------------------------------------------------
        # 6. Offseason contracts
        # --------------------------------------------------
        self._maybe_run_offseason_contracts(ctx=ctx, win_pct=win_pct)
        # --------------------------------------------------
# DRAFT LOTTERY OUTPUT (DEBUG)
# --------------------------------------------------
        if self.debug and self.last_draft_lottery:
            print("\n================ DRAFT LOTTERY =================")
            for i, team_id in enumerate(
                self.last_draft_lottery.pick_order[:16], start=1
            ):
                marker = (
                    " (LOTTERY WINNER)"
                    if team_id in self.last_draft_lottery.lottery_winners
                    else ""
                )
                print(f"Pick #{i}: {team_id}{marker}")
            print("===============================================\n")

    # Clear so it prints only once per season
        self.last_draft_lottery = None
                        # ------------------------------------
        # SCOUTING SNAPSHOT
        # ------------------------------------
        dept = self.team_scouting_departments.get(self.team.team_id)

        if dept:
            print("\n[SCOUTING SNAPSHOT]")
            top = sorted(
                dept.team_views.values(),
                key=lambda v: (v.tier, -v.grade),
            )[:5]

            for v in top:
                print(
                    f"{v.prospect_id} | grade={v.grade:.3f} "
                    f"conf={v.confidence:.3f} "
                    f"tier={v.tier} "
                    f"disagree={v.disagreement:.2f}"
                )


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

    # --------------------------------------------------
    # LEAGUE SEASON: full-structure season using league/*
    # --------------------------------------------------

    def _identity_runner_strength_noise_factors(self, team: Any) -> Tuple[float, float]:
        """Runner team archetypes → (strength_mult, game_noise_mult). Subtle, ~1–3% strength."""
        league = self.league
        d = getattr(league, "_runner_team_archetypes", None) or {}
        tid = str(getattr(team, "team_id", getattr(team, "id", "")))
        a = str(d.get(tid, "balanced")).lower()
        r = self.rng
        if a == "win_now":
            return 1.021, 0.91
        if a == "contender":
            return 1.012, 0.96
        if a == "rebuild":
            return 0.977, 1.15
        if a == "draft_and_develop":
            return 0.991, 1.05
        if a == "chaos_agent":
            sm = float(1.0 + r.uniform(-0.013, 0.015))
            return max(0.968, min(1.036, sm)), 1.38
        return 1.0, 1.0

    def _runner_cap_strength_multiplier(self, team: Any) -> float:
        """Runner universe cap tier → subtle on-ice strength trim (set by run_sim cap pass)."""
        league = self.league
        d = getattr(league, "_runner_cap_team_pressure", None) or {}
        tid = str(getattr(team, "team_id", getattr(team, "id", "")))
        p = str(d.get(tid, "moderate")).lower()
        if p == "cap_hell":
            return 0.948
        if p == "critical":
            return 0.968
        if p == "high":
            return 0.982
        if p == "low":
            return 1.008
        return 1.0

    def _runner_line_composite_for_team(self, team: Any) -> float:
        try:
            v = float(getattr(team, "_runner_line_composite_strength", 0.6) or 0.6)
        except Exception:
            v = 0.6
        return max(0.35, min(0.92, v))

    def _line_composite_strength_multiplier(self, team: Any) -> float:
        lc = self._runner_line_composite_for_team(team)
        m = 0.91 + 0.22 * lc
        return max(0.935, min(1.065, m))

    def _preseason_line_synergy_refresh(self, teams: List[Any], rng: random.Random) -> None:
        league = self.league
        for team in teams:
            trng = random.Random(rng.randint(1, 2**30) ^ (id(team) % 2**20))
            _optimize_forward_line_assignments(team, league, trng)
            roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
            for p in roster:
                ensure_player_playstyle(p)
            fchems: List[float] = []
            for _, line in _iter_team_forward_lines(team):
                if len(line) == 3:
                    fchems.append(calculate_line_chemistry(line, team))
            dchems: List[float] = []
            for _, pair in _iter_team_defense_pairs(team):
                if len(pair) == 2:
                    dchems.append(calculate_line_chemistry(pair, team))
            avg_f = sum(fchems) / len(fchems) if fchems else 0.58
            avg_d = sum(dchems) / len(dchems) if dchems else 0.58
            composite = 0.64 * avg_f + 0.36 * avg_d
            jit = rng.uniform(-0.034, 0.034)
            setattr(team, "_runner_line_composite_strength", max(0.38, min(0.91, composite + jit)))

    def _team_strength(self, team: Any) -> float:
        """
        Derive a 0..1 strength estimate from team state / roster.
        This is intentionally lightweight and deterministic.
        """
        # Prefer an explicit competitive score if the team exposes one.
        state = getattr(team, "state", None)
        comp = getattr(state, "competitive_score", None)
        base: float
        if comp is not None:
            try:
                base = float(comp)
            except Exception:
                base = 0.5
            else:
                sm, _ = self._identity_runner_strength_noise_factors(team)
                cm = self._runner_cap_strength_multiplier(team)
                tid_m = team_identity_strength_multiplier(team, self._active_era_str())
                lm = self._line_composite_strength_multiplier(team)
                return max(0.2, min(1.0, base * sm * cm * tid_m * lm))
        else:
            base = 0.5

        # Fallback: average OVR across roster if available.
        roster = list(getattr(team, "roster", None) or [])
        ovrs: List[float] = []
        for p in roster:
            fn = getattr(p, "ovr", None)
            if callable(fn):
                try:
                    ovrs.append(float(fn()))
                except Exception:
                    continue
        if ovrs:
            avg = sum(ovrs) / len(ovrs)
            # OVR is roughly 0..1
            base = avg
        sm, _ = self._identity_runner_strength_noise_factors(team)
        cm = self._runner_cap_strength_multiplier(team)
        tid_m = team_identity_strength_multiplier(team, self._active_era_str())
        lm = self._line_composite_strength_multiplier(team)
        return max(0.2, min(1.0, base * sm * cm * tid_m * lm))

    def _active_era_str(self) -> str:
        lg = getattr(self, "league", None)
        if lg is None:
            return ""
        try:
            es = getattr(lg, "era_state", None)
            if es is not None:
                ae = getattr(es, "active_era", None)
                if ae is not None and hasattr(ae, "value"):
                    return str(ae.value)
        except Exception:
            pass
        return ""

    def _build_strength_map(self, teams: List[Any]) -> Dict[str, float]:
        m: Dict[str, float] = {}
        for idx, t in enumerate(teams):
            tid = getattr(t, "team_id", None) or getattr(t, "id", None) or f"T{idx:02d}"
            m[str(tid)] = self._team_strength(t)
        return m

    def _narrative_team_goal_sigma_multiplier(self, team: Any) -> float:
        """Widen/tighten single-game goal variance from roster narrative modifiers (season-sticky)."""
        roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
        if not roster:
            return 1.0
        vs: List[float] = []
        cs: List[float] = []
        for p in roster:
            vs.append(float(getattr(p, "_narrative_performance_variance", 0.0) or 0.0))
            cs.append(float(getattr(p, "_narrative_consistency_shift", 0.0) or 0.0))
        av = sum(vs) / len(vs)
        ac = sum(cs) / len(cs)
        mult = 1.0 + 0.48 * av - 0.32 * ac
        return max(0.82, min(1.30, mult))

    def _simulate_game(
        self,
        rng: random.Random,
        home: Any,
        away: Any,
        strength_map: Dict[str, float],
        home_strength_scale: float = 1.0,
        away_strength_scale: float = 1.0,
        noise_scale: float = 1.0,
    ) -> Tuple[int, int, bool]:
        """
        Abstract single-game simulation returning (home_goals, away_goals, overtime_flag).

        Uses team strength and a light home-ice advantage to bias results,
        but leaves significant room for variance and upsets.
        Optional scales apply world-layer momentum/chemistry/fatigue/morale (subtle).
        """
        hid = str(getattr(home, "team_id", getattr(home, "id", "H")))
        aid = str(getattr(away, "team_id", getattr(away, "id", "A")))
        s_home = max(0.15, min(0.92, strength_map.get(hid, 0.5) * float(home_strength_scale)))
        s_away = max(0.15, min(0.92, strength_map.get(aid, 0.5) * float(away_strength_scale)))

        # Base scoring environment: ~2.8 goals per team
        base = 2.8
        diff = s_home - s_away
        home_mu = base + 1.0 * diff + 0.25  # home-ice bump
        away_mu = base - 1.0 * diff
        home_mu += team_scoring_pace_bias(home)
        away_mu += team_scoring_pace_bias(away)

        # Clamp to reasonable ranges
        home_mu = max(1.3, min(5.5, home_mu))
        away_mu = max(1.0, min(5.0, away_mu))

        sg = max(0.75, min(1.25, float(noise_scale)))
        nh = self._narrative_team_goal_sigma_multiplier(home)
        na = self._narrative_team_goal_sigma_multiplier(away)
        home_goals = max(0, int(round(rng.gauss(home_mu, 1.2 * sg * nh))))
        away_goals = max(0, int(round(rng.gauss(away_mu, 1.2 * sg * na))))

        overtime = False
        if home_goals == away_goals:
            # Resolve ties via an overtime coin-flip with a tiny home bias.
            overtime = True
            if rng.random() < 0.52:
                home_goals += 1
            else:
                away_goals += 1

        return home_goals, away_goals, overtime

    def simulate_league_season(self, year: int, rng: Optional[random.Random] = None) -> Optional[LeagueSeasonResult]:
        """
        Run a full league season using the league package:
            - schedule generation
            - regular-season simulation
            - standings tracking
            - playoff bracket + champion
            - awards

        Returns a LeagueSeasonResult or None if league/teams are missing.
        When world.* modules load, integrates momentum, fatigue, injuries, morale,
        chemistry, and schedule stress into the regular-season loop (deterministic rng).
        Macro trade / UFA event volume is generated in run_sim.simulate_universe_year, not in this path.
        """
        if not getattr(self.league, "teams", None):
            return None

        r = rng if rng is not None else self.rng
        teams = list(self.league.teams)
        if not teams:
            return None

        try:
            from app.sim_engine.narrative.player_journeys import touch_league_narrative_profiles

            touch_league_narrative_profiles(self.league, r)
        except Exception:
            pass

        try:
            _econ = (self.league.get_league_context().get("economics") or {})
            _cap_raw = float(_econ.get("salary_cap", 88.0) or 88.0)
            _cap_m = _cap_raw / 1_000_000.0 if _cap_raw > 200.0 else _cap_raw
            for _tm in teams:
                update_team_strategy(_tm, salary_cap_m=_cap_m)
                apply_cap_pressure_effects(_tm, salary_cap_m=_cap_m)
            _career_tm = getattr(self, "team", None)
            _cid = getattr(_career_tm, "team_id", None) if _career_tm is not None else None
            if _cid is not None:
                for _tm in teams:
                    if getattr(_tm, "team_id", None) == _cid:
                        _p = float(getattr(_tm, "cap_pressure", 0.0) or 0.0)
                        _s = str(getattr(_tm, "strategy", "balanced") or "balanced")
                        print(f"Cap Pressure: {round(_p, 3)} Strategy: {_s}")
                        break
        except Exception:
            pass

        schedule = generate_regular_season_schedule(r, teams, games_per_team=82)
        standings = StandingsTable(teams)
        self._preseason_line_synergy_refresh(teams, r)
        strength_map = self._build_strength_map(teams)

        # id -> team mapping for quick lookup
        team_by_id: Dict[str, Any] = {}
        team_ids: List[str] = []
        for idx, t in enumerate(teams):
            tid = getattr(t, "team_id", None) or getattr(t, "id", None) or f"T{idx:02d}"
            tid = str(tid)
            team_ids.append(tid)
            team_by_id[tid] = t

        ctx = getattr(self.league, "_tuning_context", None) or {}
        chaos_index = float(ctx.get("chaos_index", getattr(self.league, "_chaos_index", 0.5)) or 0.5)

        use_world = all(
            m is not None
            for m in (
                world_momentum,
                world_fatigue,
                world_morale,
                world_chemistry,
                world_injuries,
                world_durability,
                world_calendar,
            )
        )
        play_days: Dict[str, Any] = {}
        if use_world:
            play_days = world_calendar.build_team_play_days(schedule)

        last_game_day: Dict[str, Optional[int]] = {tid: None for tid in team_ids}
        prev_calendar_day: Optional[int] = None
        injury_log_major: List[Dict[str, Any]] = []

        for slot in schedule:
            home = team_by_id.get(slot.home_id)
            away = team_by_id.get(slot.away_id)
            if home is None or away is None:
                continue
            d = int(slot.day)
            hid, aid = str(slot.home_id), str(slot.away_id)

            if use_world:
                if prev_calendar_day is not None and d > prev_calendar_day:
                    span = float(d - prev_calendar_day)
                    world_momentum.decay_all_teams(teams, span * 0.06)
                prev_calendar_day = d

                for tid, tm in ((hid, home), (aid, away)):
                    lg = last_game_day.get(tid)
                    if lg is not None:
                        gap = d - lg - 1
                        if gap > 0:
                            world_fatigue.rest_roster(tm, gap, r)
                    last_game_day[tid] = d

                hb2b = bool(play_days and world_calendar.is_back_to_back(play_days.get(hid, set()), d))
                ab2b = bool(play_days and world_calendar.is_back_to_back(play_days.get(aid, set()), d))

                hm = world_momentum.team_strength_modifier(home)
                am = world_momentum.team_strength_modifier(away)
                hc = world_chemistry.team_strength_modifier(home)
                ac = world_chemistry.team_strength_modifier(away)
                hf = world_fatigue.team_fatigue_strength_factor(home)
                af = world_fatigue.team_fatigue_strength_factor(away)
                hmr = world_morale.team_morale_strength_factor(home)
                amr = world_morale.team_morale_strength_factor(away)

                h_scale = hm * hc * hf * hmr
                a_scale = am * ac * af * amr
                h_scale = max(0.93, min(1.07, h_scale))
                a_scale = max(0.93, min(1.07, a_scale))

                base_noise = 1.0 + 0.22 * (chaos_index - 0.5)
                nh = world_chemistry.chemistry_chaos_dampen(home, base_noise)
                na = world_chemistry.chemistry_chaos_dampen(away, base_noise)
                _, ih = self._identity_runner_strength_noise_factors(home)
                _, ia = self._identity_runner_strength_noise_factors(away)
                noise_scale = 0.5 * (nh + na) * (0.5 * (ih + ia))

                world_fatigue.tick_roster_fatigue_for_game(home, r, hb2b, schedule, d, hid)
                world_fatigue.tick_roster_fatigue_for_game(away, r, ab2b, schedule, d, aid)

                hg, ag, ot = self._simulate_game(
                    r, home, away, strength_map,
                    home_strength_scale=h_scale,
                    away_strength_scale=a_scale,
                    noise_scale=noise_scale,
                )

                world_momentum.update_momentum_after_game(home, hg, ag, r)
                world_momentum.update_momentum_after_game(away, ag, hg, r)
                blow = abs(hg - ag) >= 3
                world_chemistry.update_after_game(home, hg > ag, blow, r)
                world_chemistry.update_after_game(away, ag > hg, blow, r)

                for p in getattr(home, "roster", None) or []:
                    if getattr(p, "retired", False):
                        continue
                    world_morale.update_after_team_result(
                        p, hg > ag, hg - ag, r,
                        role_satisfaction_proxy=float(
                            getattr(getattr(p, "psych", None), "role_satisfaction", 0.5) or 0.5
                        ),
                    )
                for p in getattr(away, "roster", None) or []:
                    if getattr(p, "retired", False):
                        continue
                    world_morale.update_after_team_result(
                        p, ag > hg, ag - hg, r,
                        role_satisfaction_proxy=float(
                            getattr(getattr(p, "psych", None), "role_satisfaction", 0.5) or 0.5
                        ),
                    )

                for tm in (home, away):
                    for pl in getattr(tm, "roster", None) or []:
                        if int(getattr(pl, "_world_injury_games_remaining", 0) or 0) > 0:
                            world_injuries.tick_games_missed(pl)

                for tm in (home, away):
                    ev = world_injuries.maybe_injure_roster_subset(tm, r, chaos_index, max_checks=7)
                    for label, tier, games in ev:
                        if tier == "major":
                            injury_log_major.append(
                                {
                                    "player": label,
                                    "tier": tier,
                                    "games": games,
                                    "team_id": str(getattr(tm, "team_id", None) or getattr(tm, "id", None) or ""),
                                }
                            )
            else:
                _, nh = self._identity_runner_strength_noise_factors(home)
                _, na = self._identity_runner_strength_noise_factors(away)
                id_noise = 0.5 * (nh + na)
                hg, ag, ot = self._simulate_game(r, home, away, strength_map, noise_scale=id_noise)

            standings.record_game(slot.home_id, slot.away_id, hg, ag, overtime=ot)

        if use_world and world_fatigue is not None and world_injuries is not None:
            for tm in teams:
                for pl in getattr(tm, "roster", None) or []:
                    if getattr(pl, "retired", False):
                        continue
                    g = int(getattr(pl, "_world_injury_games_remaining", 0) or 0)
                    if g > 38:
                        setattr(pl, "_world_injury_games_remaining", r.randint(14, 38))
                    try:
                        f = float(world_fatigue.get_fatigue(pl))
                    except Exception:
                        f = 0.0
                    f = max(0.0, min(100.0, f))
                    try:
                        world_fatigue.set_fatigue(pl, f)
                    except Exception:
                        pass

        if use_world and world_durability is not None:
            for tm in teams:
                for pl in getattr(tm, "roster", None) or []:
                    if getattr(pl, "retired", False):
                        continue
                    world_durability.apply_season_aging_durability(pl)

        playoff_result = simulate_playoffs(r, standings, teams, strength_map)
        awards = compute_awards(standings, playoff_result, teams)

        try:
            from app.sim_engine.entities.team import update_team_gm_strategic_profile

            arch_map = dict(getattr(self.league, "_runner_team_archetypes", None) or {})
            for tid, rec in standings.records.items():
                tm = team_by_id.get(tid)
                if tm is None:
                    continue
                gp = max(1, int(getattr(rec, "gp", 0) or 0))
                pts = float(getattr(rec, "points", 0) or 0)
                pt_pct = clamp(pts / float(gp * 2), 0.28, 0.72)
                if pt_pct > 0.58:
                    bkt = "contender"
                elif pt_pct >= 0.52:
                    bkt = "playoff"
                elif pt_pct >= 0.47:
                    bkt = "bubble"
                else:
                    bkt = "rebuild"
                arch = self._pipeline_team_arche(tm, arch_map)
                pres = str(getattr(tm, "cap_pressure_tier", "moderate") or "moderate").lower()
                pipe = float(getattr(tm, "prospect_pipeline_score", 0.5) or 0.5)
                update_team_gm_strategic_profile(
                    tm,
                    runner_archetype=arch,
                    point_pct=pt_pct,
                    standings_bucket=bkt,
                    pipeline_score=pipe,
                    cap_pressure=pres,
                    rng=r,
                )
        except Exception:
            pass

        result = LeagueSeasonResult(
            year=year,
            schedule=schedule,
            standings=standings,
            playoff_result=playoff_result,
            awards=awards,
        )
        self.league_history.append(result)

        if use_world and world_calendar is not None:
            b2b_count, _rest_avg = world_calendar.summarize_schedule_stress(schedule, team_ids)
            snap_teams = []
            for t in teams:
                tid = str(getattr(t, "team_id", getattr(t, "id", "")))
                snap_teams.append(
                    {
                        "team_id": tid,
                        "momentum": world_momentum.get_team_momentum(t),
                        "chemistry": world_chemistry.get_chemistry(t),
                        "avg_morale": world_morale.team_avg_morale(t),
                        "avg_fatigue": world_fatigue.team_avg_fatigue(t),
                        "back_to_backs": int(b2b_count.get(tid, 0)),
                    }
                )
            snap_teams.sort(key=lambda x: x["momentum"], reverse=True)
            prone: List[Tuple[int, str, str]] = []
            for tm in teams:
                tid = str(getattr(tm, "team_id", getattr(tm, "id", "")))
                for pl in getattr(tm, "roster", None) or []:
                    c = int(getattr(pl, "_world_injury_event_count", 0) or 0)
                    if c <= 0:
                        continue
                    nm = str(
                        getattr(pl, "name", None)
                        or getattr(getattr(pl, "identity", None), "name", None)
                        or "?"
                    )
                    prone.append((c, tid, nm))
            prone.sort(reverse=True)
            setattr(
                self.league,
                "_world_season_snapshot",
                {
                    "year": year,
                    "chaos_index": chaos_index,
                    "teams": snap_teams,
                    "back_to_backs_ranked": sorted(b2b_count.items(), key=lambda z: z[1], reverse=True)[:10],
                    "fatigue_ranked": sorted(
                        ((x["team_id"], x["avg_fatigue"]) for x in snap_teams),
                        key=lambda z: z[1],
                        reverse=True,
                    )[:10],
                    "major_injuries": injury_log_major[-24:],
                    "injury_prone": [{"count": a[0], "team_id": a[1], "name": a[2]} for a in prone[:12]],
                },
            )
        return result

    # --------------------------------------------------
    # League-wide retirement (for universe / roster health)
    # --------------------------------------------------
    def run_league_retirements(self, league: Any, year: int) -> int:
        """
        Run progression + retirement_engine, then apply at most 60 retirements (soft) / 70 (hard)
        with priority oldest–lowest OVR; elite under 35 excluded unless engine already forced.
        """
        RET_SOFT = 60
        RET_HARD = 70
        if not getattr(league, "teams", None):
            return 0
        if not hasattr(league, "retired_players") or league.retired_players is None:
            league.retired_players = []
        rng = getattr(self, "rng", random.Random())

        ovs: List[float] = []
        for team in league.teams:
            for player in getattr(team, "roster", None) or []:
                if getattr(player, "retired", False):
                    continue
                try:
                    ovr_fn = getattr(player, "ovr", None)
                    ovs.append(float(ovr_fn()) if callable(ovr_fn) else 0.5)
                except Exception:
                    ovs.append(0.5)
        ovs.sort()
        elite_cut = 0.88
        if len(ovs) >= 8:
            elite_cut = ovs[max(0, int(len(ovs) * 0.90) - 1)]

        pitched: Set[int] = set()
        batch: List[Tuple[Any, Any, str, int, float]] = []

        for team in league.teams:
            roster = getattr(team, "roster", None) or []
            for player in roster:
                if getattr(player, "retired", False):
                    continue
                try:
                    _, retired = run_player_progression(player, rng)
                    if not retired:
                        continue
                    pid = id(player)
                    if pid in pitched:
                        continue
                    age = self._player_roster_age(player)
                    try:
                        ovr_fn = getattr(player, "ovr", None)
                        ovr = float(ovr_fn()) if callable(ovr_fn) else 0.5
                    except Exception:
                        ovr = 0.5
                    if ovr >= elite_cut and age < 35:
                        continue
                    pitched.add(pid)
                    batch.append((player, team, "progression", age, ovr))
                except Exception:
                    pass

        for team in league.teams:
            roster = getattr(team, "roster", None) or []
            for player in roster:
                if getattr(player, "retired", False):
                    continue
                pid = id(player)
                if pid in pitched:
                    continue
                self.player = player
                try:
                    retire_ctx: Dict[str, Any] = {}
                    try:
                        ovr_fn = getattr(player, "ovr", None)
                        retire_ctx["ovr"] = float(ovr_fn()) if callable(ovr_fn) else 0.5
                    except Exception:
                        retire_ctx["ovr"] = 0.5
                    decision = self.retirement_engine.evaluate_player(
                        self._build_retirement_player(), retire_ctx
                    )
                    if not bool(getattr(decision, "retired", False)):
                        continue
                    age = self._player_roster_age(player)
                    ovr = float(retire_ctx.get("ovr", 0.5))
                    if ovr >= elite_cut and age < 35:
                        continue
                    reason = str(getattr(decision, "primary_reason", "unknown"))[:120]
                    pitched.add(pid)
                    batch.append((player, team, reason, age, ovr))
                except Exception:
                    pass
                finally:
                    self.player = None

        batch.sort(key=lambda x: (-int(x[3]), float(x[4])))
        if len(batch) > RET_HARD:
            batch = batch[:RET_HARD]
        if len(batch) > RET_SOFT:
            batch = batch[:RET_SOFT]

        retired_count = 0
        for player, team, reason, _, _ in batch:
            player.retired = True
            try:
                player.retirement_reason = reason
            except Exception:
                pass
            league.retired_players.append(player)
            retired_count += 1
            roster = getattr(team, "roster", None) or []
            try:
                if roster is not None and player in roster:
                    roster.remove(player)
            except ValueError:
                pass
            if hasattr(league, "players") and league.players is not None and player in league.players:
                try:
                    league.players.remove(player)
                except ValueError:
                    pass

        for team in league.teams:
            roster = list(getattr(team, "roster", None) or [])
            to_remove = [p for p in roster if getattr(p, "retired", False)]
            for p in to_remove:
                try:
                    if p not in league.retired_players:
                        league.retired_players.append(p)
                except Exception:
                    pass
                try:
                    roster.remove(p)
                except ValueError:
                    pass
                if hasattr(league, "players") and league.players is not None and p in league.players:
                    try:
                        league.players.remove(p)
                    except ValueError:
                        pass
            team.roster = roster

        return retired_count

    # --------------------------------------------------
    # Superstar emergence / bust / late bloomer (lifecycle texture)
    # --------------------------------------------------
    def _apply_emergence_or_bust(self, player: Any, rng: random.Random, league: Any = None) -> None:
        """
        Rare silent negative for some young players only. Skips if major progression slot already used.
        Logged jumps: resolve_authoritative_major_progression_event only.
        """
        if getattr(player, "major_progression_event_this_season", None) is not None:
            return
        try:
            age = int(getattr(getattr(player, "identity", None), "age", 26))
            ovr_fn = getattr(player, "ovr", None)
            ovr = float(ovr_fn()) if callable(ovr_fn) else 0.5
            ratings = getattr(player, "ratings", None)
            if not ratings or not isinstance(ratings, dict):
                return
        except Exception:
            return

        roll = rng.random()
        delta = 0.0
        # REMOVED: tuning-driven silent breakout (+normalized bump), franchise emergence (+1.5..4),
        # and duplicate emergence late-bloom (+1..3). Those inflated OVR without BREAKOUT: logs.
        if age < 24 and 0.65 <= ovr < 0.80 and roll < 0.018 and roll >= 0.012:
            delta = float(rng.uniform(-3.5, -1.2))
        if delta == 0.0:
            return

        keys = list(ratings.keys())
        if not keys:
            return
        n_affect = max(1, min(8, int(abs(delta)) + rng.randint(0, 4)))
        chosen = rng.sample(keys, min(n_affect, len(keys)))
        for k in chosen:
            ratings[k] = clamp_rating(ratings[k] + delta)

    def run_player_storyline_pass(
        self,
        rng: Optional[random.Random] = None,
        year: int = 0,
    ) -> Dict[str, Any]:
        """
        Character-driven player storylines plus systemic consequences (team/league ripples).
        Returns {"player_storylines": [...], "narrative_consequences": [...], "league_delta": {...}}.
        """
        r = rng if rng is not None else self.rng
        league = self.league
        teams = getattr(league, "teams", None) or []
        initialize_league_player_characters(league, r)
        catalog = _get_player_storyline_catalog()
        report: List[Dict[str, Any]] = []
        consequence_log: List[Dict[str, Any]] = []
        league_delta: Dict[str, float] = {"chaos_index": 0.0, "parity_index": 0.0}
        bal: Dict[str, int] = {
            "major_arcs": 0,
            "mid_arcs": 0,
            "minor_events": 0,
            "rookie_headlines": 0,
            "suppressed_events": 0,
            "repeated_templates_trimmed": 0,
            "major_arc_cooldowns_applied": 0,
            "rookie_spam_trimmed": 0,
        }
        major_cap = r.randint(8, 15)
        mid_cap = r.randint(20, 35)
        minor_cap = r.randint(25, 50)
        total_cap = major_cap + mid_cap + minor_cap
        total_assignments = 0
        team_major: Dict[str, int] = {}
        stem_season: Dict[str, int] = {}
        family_season: Dict[str, int] = {}
        carry_stem = getattr(league, "_narrative_storyline_stem_carry", None) or {}
        if isinstance(carry_stem, dict):
            for k, v in carry_stem.items():
                try:
                    stem_season[str(k)] = stem_season.get(str(k), 0) + int(v)
                except Exception:
                    pass

        def _tier_scale_fx(tier: str) -> float:
            if tier == "major":
                return 1.0
            if tier == "minor":
                return 0.30
            return 0.64

        def _tier_scale_systemic(tier: str) -> float:
            if tier == "major":
                return 1.0
            if tier == "minor":
                return 0.36
            return 0.74

        def _pick_tier_for_player(tid0: str, lmaj: Optional[int], pl: Any) -> Optional[str]:
            rem_m = major_cap - bal["major_arcs"]
            rem_i = mid_cap - bal["mid_arcs"]
            rem_n = minor_cap - bal["minor_events"]
            if rem_m <= 0 and rem_i <= 0 and rem_n <= 0:
                return None
            rw_m = float(max(0, rem_m))
            blocked_cd = lmaj is not None and year - int(lmaj) < 2
            if blocked_cd:
                if rem_m > 0 and int(getattr(pl, "_narr_major_cd_log_year", -1) or -1) != int(year):
                    setattr(pl, "_narr_major_cd_log_year", int(year))
                    bal["major_arc_cooldowns_applied"] += 1
                rw_m = 0.0
            if team_major.get(tid0, 0) >= 2:
                rw_m *= 0.20
            rw_i = float(max(0, rem_i)) * 0.92
            rw_n = float(max(0, rem_n)) * 1.05
            s = rw_m + rw_i + rw_n
            if s <= 0:
                return None
            t0 = r.random() * s
            if t0 < rw_m:
                return "major"
            if t0 < rw_m + rw_i:
                return "mid"
            return "minor"

        for team in teams:
            roster = list(getattr(team, "roster", None) or [])
            tid = str(getattr(team, "team_id", "") or getattr(team, "id", "") or "")
            tname = str(getattr(team, "name", "") or getattr(team, "team_name", "") or tid)
            seed_mix = r.randint(1, 2**30) ^ sum((ord(c) & 0xFF) for c in tid[:24])
            tr_local = random.Random(seed_mix & 0x7FFFFFFF)
            tr_local.shuffle(roster)
            for player in roster:
                if getattr(player, "retired", False):
                    continue
                ensure_player_character_initialized(player, r)
                ident = getattr(player, "identity", None)
                age = int(getattr(ident, "age", 26) or 26) if ident is not None else 26
                pname = str(
                    getattr(player, "name", None)
                    or (getattr(ident, "full_name", None) if ident is not None else None)
                    or (getattr(ident, "name", None) if ident is not None else None)
                    or "Unknown"
                )
                ovr = _player_ovr01(player)
                char = _player_character_rating_0_100(player)
                tag = _player_lifecycle_tag(player, ovr, age)

                prev_b = getattr(player, "_storyline_ovr_baseline", None)
                perf_delta = float(ovr - float(prev_b)) if prev_b is not None else 0.0
                setattr(player, "_storyline_ovr_baseline", ovr)

                active: List[Dict[str, Any]] = list(getattr(player, "_storyline_active", None) or [])
                new_active: List[Dict[str, Any]] = []
                for sl in active:
                    slc = dict(sl)
                    left = float(slc.get("seasons_left", 0.0)) - 1.0
                    if left <= 0.01:
                        t_end = str(slc.get("tier") or _storyline_tier_for_def(slc))
                        sc_end = _tier_scale_fx(t_end)
                        _storyline_fx_apply(player, slc.get("fx") or {}, scale=-0.72 * max(0.35, sc_end))
                        continue
                    slc["seasons_left"] = left
                    new_active.append(slc)
                player._storyline_active = new_active

                cooldowns: Dict[str, int] = dict(getattr(player, "_storyline_cooldowns", None) or {})

                lmaj_y = getattr(player, "_storyline_last_major_year", None)
                try:
                    lmaj_i = int(lmaj_y) if lmaj_y is not None else None
                except Exception:
                    lmaj_i = None

                may_assign = (
                    len(new_active) < 2
                    and total_assignments < total_cap
                    and not (ovr < 0.43 and char > 62)
                )
                won_roll = False
                if may_assign:
                    p_try = 0.036 if char < 40 else 0.023 if char < 70 else 0.014
                    if abs(perf_delta) >= 0.035:
                        p_try *= 1.12
                    won_roll = r.random() <= p_try

                if may_assign and won_roll:
                    chosen_tier = _pick_tier_for_player(tid, lmaj_i, player)
                    picked: Optional[Dict[str, Any]] = None
                    if chosen_tier is None:
                        bal["suppressed_events"] += 1
                    else:
                        if chosen_tier == "major" and char < 33 and r.random() < 0.095:
                            ex = synthetic_extreme_low_character_storyline(r)
                            if _eligible_storyline_def(ex, char, tag):
                                picked = ex

                        if picked is None:
                            pw = _pool_weights_for_character(char)
                            pools_tier = sorted(
                                {
                                    str(d.get("pool"))
                                    for d in catalog
                                    if _storyline_tier_for_def(d) == chosen_tier and str(d.get("pool", ""))
                                }
                            )
                            if not pools_tier:
                                pass
                            else:
                                weights = [max(0.001, float(pw.get(p, 0.04))) for p in pools_tier]
                                li = pools_tier.index("legal_crime") if "legal_crime" in pools_tier else -1
                                if li >= 0:
                                    weights[li] *= _legal_pool_weight_mult(char)
                                sw = sum(weights)
                                pool = pools_tier[-1]
                                if sw > 0:
                                    x = r.random() * sw
                                    acc = 0.0
                                    for pi, w in zip(pools_tier, weights):
                                        acc += w
                                        if x <= acc:
                                            pool = pi
                                            break

                                candidates = [
                                    d
                                    for d in catalog
                                    if str(d.get("pool")) == pool
                                    and _storyline_tier_for_def(d) == chosen_tier
                                    and _eligible_storyline_def(d, char, tag)
                                ]
                                if not candidates:
                                    pass
                                else:
                                    pol_w = get_storyline_polarity_weights(char)
                                    sid_block = {str(sl.get("id", "")) for sl in new_active}
                                    c_weights: List[float] = []
                                    for d in candidates:
                                        stem = _storyline_template_stem(str(d.get("text", "")))
                                        rep = float(stem_season.get(stem, 0))
                                        fam = str(d.get("pool") or "")
                                        fam_c = float(family_season.get(fam, 0))
                                        w0 = _pick_weight_storyline(d, perf_delta, tag) * float(
                                            pol_w.get(classify_storyline_polarity(d), 1.0)
                                        )
                                        w0 *= _storyline_context_fit_weight(d, tag, char, age, perf_delta, ovr)
                                        w0 *= _storyline_overused_template_penalty(str(d.get("text", "")))
                                        w0 /= 1.0 + 0.55 * rep + 0.20 * fam_c
                                        c_weights.append(w0)

                                    for _ in range(28):
                                        tw = sum(c_weights)
                                        if tw <= 0:
                                            bal["repeated_templates_trimmed"] += 1
                                            break
                                        x = r.random() * tw
                                        acc = 0.0
                                        idx = 0
                                        for i, w in enumerate(c_weights):
                                            acc += w
                                            if x <= acc:
                                                idx = i
                                                break
                                        cand = candidates[idx]
                                        cid = str(cand.get("id", ""))
                                        if cid in sid_block:
                                            c_weights[idx] = 0.0
                                            continue
                                        ly = cooldowns.get(cid)
                                        if ly is not None and year - int(ly) < 2:
                                            c_weights[idx] = 0.0
                                            continue
                                        picked = cand
                                        break

                    if chosen_tier is not None and picked is None:
                        bal["suppressed_events"] += 1

                    if picked is not None:
                        eff_tier = str(picked.get("tier") or chosen_tier or "mid").lower()
                        if eff_tier != chosen_tier:
                            eff_tier = chosen_tier or eff_tier
                        fx0 = dict(picked.get("fx") or {})
                        if picked.get("volatile"):
                            fx0 = _maybe_volatile_storyline_fx(fx0, r)
                        dur_key = str(picked.get("dur", "medium"))
                        seasons_left = _dur_seasons_band(dur_key, r)
                        pol = str(picked.get("polarity") or classify_storyline_polarity(picked))
                        ch_mult = character_storyline_effect_multiplier(char)
                        tfx = _tier_scale_fx(eff_tier)
                        _storyline_fx_apply(player, fx0, scale=ch_mult * tfx)
                        entry = {
                            "id": picked.get("id", ""),
                            "text": picked.get("text", ""),
                            "fx": fx0,
                            "dur": dur_key,
                            "seasons_left": seasons_left,
                            "polarity": pol,
                            "tier": eff_tier,
                            "pool": str(picked.get("pool") or ""),
                        }
                        new_active.append(entry)
                        player._storyline_active = new_active
                        cooldowns[str(picked.get("id", ""))] = int(year)
                        player._storyline_cooldowns = cooldowns
                        if eff_tier == "major":
                            setattr(player, "_storyline_last_major_year", int(year))
                            team_major[tid] = team_major.get(tid, 0) + 1
                        st = _storyline_template_stem(str(picked.get("text", "")))
                        stem_season[st] = stem_season.get(st, 0) + 1
                        fam_k = str(picked.get("pool") or "")
                        family_season[fam_k] = family_season.get(fam_k, 0) + 1
                        total_assignments += 1
                        if eff_tier == "major":
                            bal["major_arcs"] += 1
                        elif eff_tier == "minor":
                            bal["minor_events"] += 1
                        else:
                            bal["mid_arcs"] += 1

                        etype, sev = _classify_systemic_event_from_storyline(picked, fx0)
                        sev *= _tier_scale_systemic(eff_tier)
                        event_obj: Dict[str, Any] = {
                            "type": etype,
                            "severity": sev,
                            "effects": {},
                            "storyline": str(picked.get("text") or ""),
                        }
                        log_rec = apply_systemic_consequences(player, team, league_delta, event_obj)
                        ripple_n = apply_team_ripple(team, player, event_obj)
                        _normalize_systemic_after_consequences(player, team)
                        log_rec["player_name"] = pname
                        log_rec["team_name"] = tname
                        log_rec["storyline_text"] = event_obj.get("storyline", "")
                        log_rec["teammates_rippled"] = ripple_n
                        log_rec["storyline_polarity"] = pol
                        log_rec["arc_tier"] = eff_tier
                        consequence_log.append(log_rec)

                final_active = list(getattr(player, "_storyline_active", None) or [])
                if final_active:
                    fx_combined: Dict[str, float] = {}
                    for sl in final_active:
                        for k, v in (sl.get("fx") or {}).items():
                            fx_combined[k] = fx_combined.get(k, 0.0) + float(v)
                    pols: List[str] = []
                    tiers_g: List[str] = []
                    for sl in final_active:
                        tiers_g.append(str(sl.get("tier") or _storyline_tier_for_def(sl)))
                        ptag = sl.get("polarity")
                        if ptag:
                            pols.append(str(ptag))
                        else:
                            pols.append(
                                classify_storyline_polarity(
                                    {
                                        "text": sl.get("text", ""),
                                        "fx": sl.get("fx") or {},
                                        "pool": str(sl.get("pool") or ""),
                                        "legal": False,
                                    }
                                )
                            )
                    report.append(
                        {
                            "player": pname,
                            "team_id": tid,
                            "team": tname,
                            "character": char,
                            "personality": str(getattr(player, "personality", "") or ""),
                            "status": tag,
                            "storylines": [sl.get("text", "") for sl in final_active],
                            "storyline_polarities": pols,
                            "arc_tiers": tiers_g,
                            "effect": _storyline_effect_summary_pct(fx_combined),
                            "duration": " / ".join(
                                [
                                    _duration_phrase(float(sl.get("seasons_left", 0.5)), str(sl.get("dur", "medium")))
                                    for sl in final_active[:2]
                                ]
                            ),
                            "_narr_sort": (
                                0 if any(x == "major" for x in tiers_g) else 1 if any(x == "mid" for x in tiers_g) else 2,
                                -ovr,
                            ),
                        }
                    )

        if len(report) > 64:
            report.sort(key=lambda rec: (rec.get("_narr_sort") or (99, 0)))
            bal["suppressed_events"] += len(report) - 64
            report = report[:64]
        for rec in report:
            rec.pop("_narr_sort", None)

        if len(consequence_log) > 82:
            consequence_log.sort(
                key=lambda x: (
                    0 if str(x.get("arc_tier")) == "major" else 1 if str(x.get("arc_tier")) == "mid" else 2,
                    str(x.get("team_name") or ""),
                )
            )
            bal["suppressed_events"] += len(consequence_log) - 82
            consequence_log = consequence_log[:82]

        try:
            dec_stem = {k: max(0, int(v) - 1) for k, v in stem_season.items() if int(v) > 0}
            setattr(league, "_narrative_storyline_stem_carry", dec_stem)
            setattr(league, "_narrative_storyline_family_counts", dict(family_season))
        except Exception:
            pass

        league_delta["chaos_index"] = min(0.09, max(0.0, float(league_delta.get("chaos_index", 0.0))))
        league_delta["parity_index"] = min(0.06, max(0.0, float(league_delta.get("parity_index", 0.0))))
        return {
            "player_storylines": report,
            "narrative_consequences": consequence_log,
            "league_delta": league_delta,
            "narrative_balance": bal,
        }

    def restore_line_chemistry_ratings(self) -> None:
        """Undo last season's line-chemistry rating multipliers before progression runs."""
        restore_league_line_chemistry_ratings(self.league)

    def apply_forward_line_chemistry_pass(self) -> List[Dict[str, Any]]:
        """Build lines, snapshot keys, apply chemistry multipliers; returns log rows."""
        return run_line_chemistry_pass(self.league)

    def run_player_distribution_pass(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Widen OVR spread, force elite/depth bands, assign percentile roles (before tuning normalize)."""
        r = rng if rng is not None else self.rng
        return run_player_distribution_pipeline(self.league, r)

    def post_normalize_distribution_rescue(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """After tuning normalization: re-widen if variance collapsed."""
        r = rng if rng is not None else self.rng
        return post_normalize_distribution_rescue(self.league, r)

    def apply_percentile_player_roles(self) -> int:
        """Re-apply league percentile roles after tuning (overwrites narrative roles)."""
        return assign_player_roles_percentile(collect_league_roster_players(self.league))

    def summarize_roster_distribution(self) -> Dict[str, Any]:
        return summarize_roster_distribution(self.league)

    def apply_emergence_and_bust_pass(self, league: Any, rng: Optional[random.Random] = None) -> None:
        """
        One pass over all league roster players: apply rare emergence/bust/late-bloomer
        rating changes for elite turnover and prospect variance. Deterministic if rng provided.
        """
        r = rng if rng is not None else self.rng
        if not getattr(league, "teams", None):
            return
        for team in league.teams:
            roster = getattr(team, "roster", None) or []
            for player in roster:
                if getattr(player, "retired", False):
                    continue
                self._apply_emergence_or_bust(player, r, league)

    # --------------------------------------------------
    # League talent stabilization & aging calibration
    # --------------------------------------------------
    def _league_ovr_stats(self, league: Any) -> Tuple[List[Tuple[Any, float]], float, float, float, float, float]:
        """Collect roster OVRs; return (players_with_ovr), top_ovr, top_10_avg, top_50_avg, mean_ovr, median_ovr."""
        all_ovrs: List[Tuple[Any, float]] = []
        for team in getattr(league, "teams", None) or []:
            for p in getattr(team, "roster", None) or []:
                if getattr(p, "retired", False):
                    continue
                try:
                    ovr_fn = getattr(p, "ovr", None)
                    ovr = float(ovr_fn()) if callable(ovr_fn) else 0.5
                    all_ovrs.append((p, ovr))
                except Exception:
                    pass
        if not all_ovrs:
            return [], 0.0, 0.0, 0.0, 0.0, 0.0
        ovrs_only = [o for _, o in all_ovrs]
        top = max(ovrs_only)
        mean = sum(ovrs_only) / len(ovrs_only)
        sorted_ovrs = sorted(ovrs_only, reverse=True)
        n = len(sorted_ovrs)
        top_10_avg = sum(sorted_ovrs[: min(10, n)]) / min(10, n) if n else 0.0
        top_50_avg = sum(sorted_ovrs[: min(50, n)]) / min(50, n) if n else 0.0
        mid = n // 2
        median = (sorted_ovrs[mid] + sorted_ovrs[mid - 1]) / 2.0 if mid > 0 else sorted_ovrs[0]
        return all_ovrs, top, top_10_avg, top_50_avg, mean, median

    def apply_aging_calibration(self, league: Any, rng: Optional[random.Random] = None) -> None:
        """
        No-op: youth growth and logged aging are handled by progression.development + career lifecycle
        resolve_authoritative_major_progression_event (AGING DECLINE). This pass previously duplicated
        rating bumps and inflated league OVR when combined with lifecycle.
        """
        return

    def league_balance_check(self, league: Any, rng: Optional[random.Random] = None, year: Optional[int] = None) -> Dict[str, float]:
        """
        League equilibrium: targets top 0.94-0.98, top_50 0.86-0.92, mean 0.68-0.72, median 0.67-0.71.
        If median < 0.66: stronger floor boost. If median > 0.74: no boost (natural decay). Soft only.
        """
        r = rng if rng is not None else self.rng
        all_ovrs, top_ovr, top_10_avg, top_50_avg, mean_ovr, median_ovr = self._league_ovr_stats(league)
        if not all_ovrs:
            return {"top_ovr": 0.0, "top_10_avg": 0.0, "top_50_avg": 0.0, "mean_ovr": 0.0, "median_ovr": 0.0}
        # When league mean is already healthy, do not re-inflate stars (stacked with progression).
        _suppress_star_pullup = bool(mean_ovr > 0.72 or top_10_avg > 0.88)
        # Top player below 0.94: soft boost to top 5%
        if not _suppress_star_pullup and top_ovr < 0.94:
            sorted_by_ovr = sorted(all_ovrs, key=lambda x: x[1], reverse=True)
            n_top = max(1, len(sorted_by_ovr) // 20)
            boost_mult = 1.0 + 0.012 * (0.94 - top_ovr)
            for player, _ in sorted_by_ovr[:n_top]:
                ratings = getattr(player, "ratings", None)
                if ratings and isinstance(ratings, dict):
                    for k in list(ratings.keys()):
                        ratings[k] = clamp_rating(ratings[k] * boost_mult)
        # Top 50 below 0.86: soft boost to top 50
        if not _suppress_star_pullup and top_50_avg < 0.86 and len(all_ovrs) >= 10:
            sorted_by_ovr = sorted(all_ovrs, key=lambda x: x[1], reverse=True)
            n_50 = min(50, len(sorted_by_ovr))
            boost_mult = 1.0 + 0.008 * (0.86 - top_50_avg)
            for player, _ in sorted_by_ovr[:n_50]:
                ratings = getattr(player, "ratings", None)
                if ratings and isinstance(ratings, dict):
                    for k in list(ratings.keys()):
                        ratings[k] = clamp_rating(ratings[k] * boost_mult)
        # Median below 0.66: stronger floor; 0.66-0.67: gentle floor
        if median_ovr < 0.67:
            sorted_by_ovr = sorted(all_ovrs, key=lambda x: x[1])
            n_floor = max(1, int(len(sorted_by_ovr) * 0.30))
            strength = 0.6 if median_ovr < 0.66 else 0.35
            add_amt = (0.67 - median_ovr) * strength * 99.0 / 100.0
            for player, _ in sorted_by_ovr[:n_floor]:
                ratings = getattr(player, "ratings", None)
                if ratings and isinstance(ratings, dict):
                    keys = list(ratings.keys())
                    if keys:
                        per_k = add_amt / len(keys)
                        for k in keys:
                            ratings[k] = clamp_rating(ratings[k] + per_k)
        # Cap young OVR so top boost cannot create 0.99 at 20-22
        for player, _ in all_ovrs:
            age = int(getattr(getattr(player, "identity", None), "age", 26))
            cap = 0.88 if age <= 20 else (0.92 if age <= 22 else (0.95 if age <= 24 else 0.99))
            ovr_fn = getattr(player, "ovr", None)
            current_ovr = float(ovr_fn()) if callable(ovr_fn) else 0.5
            if current_ovr <= cap:
                continue
            ratings = getattr(player, "ratings", None)
            if ratings and isinstance(ratings, dict) and current_ovr > 0:
                scale = cap / current_ovr
                for k in list(ratings.keys()):
                    ratings[k] = clamp_rating(ratings[k] * scale)
        return {"top_ovr": top_ovr, "top_10_avg": top_10_avg, "top_50_avg": top_50_avg, "mean_ovr": mean_ovr, "median_ovr": median_ovr}

    def get_league_talent_metrics(self, league: Any) -> Dict[str, float]:
        """Return current league OVR stats without applying any corrections. For diagnostics."""
        _, top_ovr, top_10_avg, top_50_avg, mean_ovr, median_ovr = self._league_ovr_stats(league)
        return {"top_ovr": top_ovr, "top_10_avg": top_10_avg, "top_50_avg": top_50_avg, "mean_ovr": mean_ovr, "median_ovr": median_ovr}

    def apply_era_tuning_from_context(
        self,
        league: Any,
        tuning_context: Dict[str, Any],
        teams: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runner/universe hook: apply era modifiers to all rosters and cache context on the league for emergence/aging.
        """
        try:
            from app.sim_engine.tuning.era_modifiers import apply_era_to_league
        except Exception:
            return {}
        setattr(league, "_tuning_context", tuning_context)
        return apply_era_to_league(tuning_context, league, teams)

