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

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Callable, Set, Sequence
import hashlib
import math
import os
import random
import re
import time


# Set True, or export NHL_DEBUG_STATS_PIPELINE=1, for verbose stat ledger tracing.
DEBUG_STATS_PIPELINE: bool = False


def _stats_pipeline_debug() -> bool:
    return bool(DEBUG_STATS_PIPELINE) or os.environ.get("NHL_DEBUG_STATS_PIPELINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


_GM_FLOAT_LEDGER_KEYS = frozenset(
    {"cf", "ca", "ff", "fa", "xgf", "xga", "ixg", "xa", "gf_on", "ga_on", "xgf_pct_sum", "on_ice_shots_for", "on_ice_shots_against", "goalie_xga"}
)

_SKATER_LEDGER_ANALYTICS_KEYS = (
    "cf",
    "ca",
    "ff",
    "fa",
    "xgf",
    "xga",
    "ixg",
    "xa",
    "gf_on",
    "ga_on",
    "missed_shots",
    "blocked_attempts_for",
    "ppg",
    "ppa",
    "shg",
    "sha",
    "primary_assists",
    "secondary_assists",
    "xgf_pct_sum",
    "xgf_pct_gp",
)


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
    PersonalityTraits,
    ATTRIBUTE_KEYS,
    OFFENSE_KEYS,
    PASSING_KEYS,
    DEFENSE_KEYS,
    IQ_KEYS,
    PHYS_KEYS,
    SKATING_KEYS,
    GOALIE_KEYS,
    clamp_rating,
    compute_ovr,
    assign_skater_archetype,
    archetype_from_generation_profile,
    ALIASES,
    DEFAULT_NHL_RATING,
    random_height_cm,
    sanitize_height_cm,
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
from app.sim_engine.economy.trade_ai import evaluate_trade_market
from app.sim_engine.economy.waiver_ai import process_waivers
from app.sim_engine.economy.roster_manager import RosterManager

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
from app.sim_engine.economy.cap_engine import advance_league_salary_cap


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
    # Game-derived only (no distribution injection). Empty list if aggregation skipped.
    player_season_stats: List[Dict[str, Any]] = field(default_factory=list)
    # Game-derived goalie season rows (wins/saves/SV%/GAA/shutouts). Empty if none.
    goalie_season_stats: List[Dict[str, Any]] = field(default_factory=list)
    simulation_meta: Dict[str, Any] = field(default_factory=dict)
    news_events: List[Dict[str, Any]] = field(default_factory=list)


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
    elif _PLAYER_STORYLINE_CATALOG_CACHE and not any(
        d.get("legal_severity")
        for d in _PLAYER_STORYLINE_CATALOG_CACHE
        if str(d.get("pool") or "") == "legal_crime"
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


_TRAIT_AXIS_NAMES = (
    "loyalty",
    "ambition",
    "money_focus",
    "family_priority",
    "legacy_drive",
    "risk_tolerance",
    "adaptability",
    "patience",
    "stability_need",
    "ego",
    "confidence",
    "volatility",
    "competitiveness",
    "leadership",
    "coachability",
    "media_comfort",
    "introversion",
    "work_ethic",
    "mental_toughness",
    "clutch_tendency",
)


def _personality_traits_are_generic(traits: Any) -> bool:
    if traits is None:
        return True
    vals: List[float] = []
    for name in _TRAIT_AXIS_NAMES:
        raw = getattr(traits, name, None)
        if raw is None:
            continue
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            continue
    if len(vals) < 8:
        return True
    spread = max(vals) - min(vals)
    mean = sum(vals) / len(vals)
    return spread < 0.12 and abs(mean - 0.5) < 0.06


def _archetypes_for_character_tag(tag: str, rng: random.Random) -> List[Any]:
    pool = {
        "leader": [PersonalityArchetype.LEADER, PersonalityArchetype.PROFESSIONAL, PersonalityArchetype.COMPETITOR],
        "professional": [PersonalityArchetype.PROFESSIONAL, PersonalityArchetype.LOYALIST, PersonalityArchetype.STABILITY_SEEKER],
        "neutral": [PersonalityArchetype.JOURNEYMAN, PersonalityArchetype.FAMILY_FIRST, PersonalityArchetype.INTROVERT, PersonalityArchetype.EXTROVERT],
        "volatile": [PersonalityArchetype.CHIP_ON_SHOULDER, PersonalityArchetype.RISK_TAKER, PersonalityArchetype.STAR],
        "toxic": [PersonalityArchetype.CHIP_ON_SHOULDER, PersonalityArchetype.MONEY_HUNGRY, PersonalityArchetype.STAR],
    }.get(str(tag or "neutral"), [PersonalityArchetype.JOURNEYMAN])
    extra = [
        PersonalityArchetype.FAMILY_FIRST,
        PersonalityArchetype.MONEY_HUNGRY,
        PersonalityArchetype.COMPETITOR,
        PersonalityArchetype.INTROVERT,
        PersonalityArchetype.EXTROVERT,
        PersonalityArchetype.RISK_TAKER,
        PersonalityArchetype.STABILITY_SEEKER,
    ]
    chosen = list(rng.sample(pool, k=min(2, len(pool))))
    extra_pick = rng.choice(extra)
    if extra_pick not in chosen:
        chosen.append(extra_pick)
    return chosen


def diversify_player_personality_and_psych(player: Any, rng: random.Random) -> None:
    """Give NHL roster players distinct PersonalityTraits / psych instead of the 0.5 defaults."""
    from app.sim_engine.ai.personality import clamp01

    traits = getattr(player, "traits", None)
    if not _personality_traits_are_generic(traits):
        vol = float(getattr(traits, "volatility", 0.5) or 0.5)
        setattr(player, "_narrative_performance_variance", max(0.0, min(0.9, (vol - 0.32) * 1.15)))
        patience = float(getattr(traits, "patience", 0.5) or 0.5)
        setattr(player, "_narrative_consistency_shift", max(-0.25, min(0.55, patience * 0.45 + (1.0 - vol) * 0.35 - 0.38)))
        return

    pid = str(getattr(player, "id", "") or "")
    seed_src = pid or str(getattr(player, "name", "") or "player")
    seed = int(hashlib.sha256(f"{seed_src}:story_personality_v1".encode("utf-8")).hexdigest()[:16], 16)
    local = random.Random(seed)
    tag = str(getattr(player, "personality", "") or "neutral")
    profile = PersonalityFactory(local).generate(archetypes=_archetypes_for_character_tag(tag, local), seed=local.randint(1, 10**9))
    new_traits = PersonalityTraits(
        loyalty=profile.loyalty,
        ambition=profile.ambition,
        money_focus=profile.money_focus,
        family_priority=profile.family_priority,
        legacy_drive=profile.legacy_drive,
        risk_tolerance=profile.risk_tolerance,
        adaptability=profile.adaptability,
        patience=profile.patience,
        stability_need=profile.stability_need,
        ego=profile.ego,
        confidence=profile.confidence,
        volatility=profile.volatility,
        competitiveness=profile.competitiveness,
        leadership=profile.leadership,
        coachability=profile.coachability,
        media_comfort=profile.media_comfort,
        introversion=profile.introversion,
        work_ethic=clamp01(0.12 + 0.76 * profile.patience + local.uniform(-0.18, 0.22)),
        mental_toughness=clamp01(0.10 + 0.55 * (1.0 - profile.volatility) + 0.30 * profile.confidence + local.uniform(-0.12, 0.12)),
        clutch_tendency=clamp01(0.15 + 0.55 * profile.competitiveness + 0.20 * profile.confidence + local.uniform(-0.18, 0.18)),
    )
    ch = float(getattr(player, "character", 50) or 50)
    # Character score still biases a few public axes so "leader" vs "toxic" is visible.
    bias = (ch - 50.0) / 80.0
    new_traits.loyalty = clamp01(new_traits.loyalty + 0.16 * bias)
    new_traits.leadership = clamp01(new_traits.leadership + 0.18 * bias)
    new_traits.ego = clamp01(new_traits.ego - 0.12 * bias)
    new_traits.volatility = clamp01(new_traits.volatility - 0.14 * bias)
    new_traits.clamp_all()
    player.traits = new_traits

    psych = getattr(player, "psych", None)
    if psych is not None:
        def _wobble(spread: float) -> float:
            return local.gauss(0.0, spread)

        psych.morale = clamp01(0.38 + 0.28 * new_traits.confidence + _wobble(0.14))
        psych.morale_sensitivity = clamp01(0.22 + 0.62 * new_traits.volatility + _wobble(0.08))
        psych.confidence_level = clamp01(new_traits.confidence + _wobble(0.08))
        psych.confidence_volatility = clamp01(new_traits.volatility + _wobble(0.06))
        psych.coach_trust = clamp01(0.30 + 0.50 * new_traits.coachability + _wobble(0.12))
        psych.coach_relationship = clamp01(getattr(psych, "coach_relationship", 0.5) * 0.2 + 0.35 + 0.45 * new_traits.coachability + _wobble(0.10))
        psych.locker_room_fit = clamp01(0.28 + 0.35 * (1.0 - new_traits.introversion) + 0.22 * new_traits.leadership + _wobble(0.10))
        psych.trust_in_teammates = clamp01(0.32 + 0.40 * new_traits.loyalty + _wobble(0.12))
        psych.media_stress = clamp01(0.12 + 0.55 * (1.0 - new_traits.media_comfort) + 0.20 * new_traits.volatility + _wobble(0.08))
        psych.tilt_susceptibility = clamp01(0.15 + 0.70 * new_traits.volatility + _wobble(0.08))
        psych.bounce_back_tendency = clamp01(new_traits.mental_toughness + _wobble(0.08))
        psych.role_satisfaction = clamp01(0.40 + _wobble(0.16))
        psych.ice_time_satisfaction = clamp01(0.42 + _wobble(0.14))
        if hasattr(psych, "clamp_all"):
            psych.clamp_all()

    vol = float(new_traits.volatility)
    setattr(player, "_narrative_performance_variance", max(0.0, min(0.9, (vol - 0.28) * 1.25)))
    setattr(player, "_narrative_consistency_shift", max(-0.25, min(0.55, new_traits.patience * 0.45 + (1.0 - vol) * 0.35 - 0.38)))
    setattr(player, "_story_personality_gen", 1)


def ensure_player_character_initialized(player: Any, rng: random.Random) -> int:
    raw = getattr(player, "character", None)
    need = raw is None
    if not need and isinstance(raw, (int, float)):
        ri = int(raw)
        need = ri < 20 or ri > 90
    if need:
        setattr(player, "character", generate_player_character_score(rng))
    assign_player_personality_tag_from_character(player)
    diversify_player_personality_and_psych(player, rng)
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
# =====================================================================
# CANONICAL PLAYER CREATION FINALIZER
# =====================================================================
# Purpose:
# Every new Player created inside engine.py must be made compatible with:
# - real game stat ledger
# - progression
# - storylines
# - roster screens
# - career summaries
#
# Do NOT use Player_Stats.py / League_Stats.py for this.
# The real game ledger remains the source of truth.


def _safe_player_name_for_id(player: Any) -> str:
    ident = getattr(player, "identity", None)
    name = ""
    if ident is not None:
        name = str(getattr(ident, "name", "") or getattr(ident, "full_name", "") or "")
    if not name:
        name = str(getattr(player, "name", "") or "player")
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return name or "player"


def _safe_team_id_for_player_creation(team: Any, fallback: Any = "") -> str:
    for attr in ("team_id", "id", "abbr", "code"):
        v = getattr(team, attr, None)
        if v is not None and str(v).strip():
            return str(v)
    return str(fallback or "TEAM")


def _existing_player_ids_from_league(league: Any) -> Set[str]:
    ids: Set[str] = set()
    for p in getattr(league, "players", None) or []:
        pid = getattr(p, "id", None)
        if pid is not None and str(pid).strip():
            ids.add(str(pid))
    for t in getattr(league, "teams", None) or []:
        for p in getattr(t, "roster", None) or []:
            pid = getattr(p, "id", None)
            if pid is not None and str(pid).strip():
                ids.add(str(pid))
    return ids


def _assign_stable_player_id(
    player: Any,
    league: Any,
    rng: random.Random,
    *,
    team_id: str,
    source: str,
    season_year: int,
) -> str:
    """
    The game ledger keys player stats by player.id.
    If player.id is missing/blank/duplicated, that player can disappear from stats.
    """
    existing = _existing_player_ids_from_league(league)
    cur = getattr(player, "id", None)

    if cur is not None and str(cur).strip() and str(cur) not in existing:
        return str(cur)

    name_slug = _safe_player_name_for_id(player)
    base = f"{source}_{season_year}_{team_id}_{name_slug}"
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", base).strip("_")

    candidate = base
    n = 1
    while candidate in existing:
        n += 1
        candidate = f"{base}_{n}"

    setattr(player, "id", candidate)
    setattr(player, "player_id", candidate)
    return candidate


def _infer_player_potential_01(player: Any, rng: random.Random) -> float:
    """
    Potential is a ceiling signal, not current OVR.
    Young players can have bigger gap. Older players should be closer to current.
    """
    raw = getattr(player, "potential", None)
    if raw is not None:
        try:
            pf = float(raw)
            if pf > 1.5:
                pf /= 99.0
            return max(0.35, min(0.99, pf))
        except Exception:
            pass

    try:
        ovr = float(player.ovr()) if callable(getattr(player, "ovr", None)) else 0.62
    except Exception:
        ovr = 0.62

    age = career_player_age(player)

    if age <= 20:
        gap = rng.uniform(0.08, 0.18)
    elif age <= 23:
        gap = rng.uniform(0.05, 0.14)
    elif age <= 26:
        gap = rng.uniform(0.02, 0.09)
    elif age <= 29:
        gap = rng.uniform(0.00, 0.05)
    else:
        gap = rng.uniform(-0.04, 0.02)

    return max(0.40, min(0.99, ovr + gap))


def _infer_dev_type_from_player(player: Any, rng: random.Random) -> str:
    existing = getattr(player, "dev_type", None)
    if existing:
        return str(existing)

    potential = _infer_player_potential_01(player, rng)
    age = career_player_age(player)

    if potential >= 0.88 and age <= 23:
        dt = "elite"
    elif age >= 24 and potential >= 0.78 and rng.random() < 0.28:
        dt = "late_bloomer"
    elif potential < 0.58 and age <= 24:
        dt = "bust" if rng.random() < 0.28 else "slow"
    elif potential >= 0.72:
        dt = "standard"
    else:
        dt = rng.choices(
            ["standard", "slow", "late_bloomer", "bust"],
            weights=[0.50, 0.25, 0.15, 0.10],
            k=1,
        )[0]

    setattr(player, "dev_type", dt)
    return dt


def _ensure_player_stat_containers(player: Any) -> None:
    """
    These are not the abstract Player_Stats engine.
    These containers are just safe landing zones for the actual game ledger sync.
    """
    if getattr(player, "season_stats", None) is None:
        setattr(player, "season_stats", {})
    if getattr(player, "career_stats", None) is None:
        setattr(player, "career_stats", {})
    if getattr(player, "game_log", None) is None:
        setattr(player, "game_log", [])


def finalize_created_player_for_game_ledger(
    player: Any,
    *,
    league: Any,
    team: Any,
    rng: random.Random,
    source: str,
    season_year: int,
) -> Any:
    """
    Call this once immediately after every Player(...) creation in engine.py.
    """
    team_id = _safe_team_id_for_player_creation(team)

    pid = _assign_stable_player_id(
        player,
        league,
        rng,
        team_id=team_id,
        source=source,
        season_year=int(season_year),
    )

    ctx = getattr(player, "context", None)
    if ctx is not None:
        try:
            ctx.current_team_id = str(team_id)
        except Exception:
            pass

    setattr(player, "current_team_id", str(team_id))
    setattr(player, "team_id", str(team_id))

    potential = _infer_player_potential_01(player, rng)
    setattr(player, "potential", potential)

    _infer_dev_type_from_player(player, rng)
    ensure_player_character_initialized(player, rng)
    assign_player_personality_tag_from_character(player)

    try:
        assign_career_phase_from_age(player)
    except Exception:
        pass

    ensure_player_playstyle(player)
    _ensure_player_stat_containers(player)

    try:
        from app.sim_engine.generation.player_headshots import ensure_player_headshot

        ensure_player_headshot(player)
    except Exception:
        pass

    setattr(player, "_game_ledger_ready", True)
    setattr(player, "_ledger_player_id", pid)

    return player


def _invoke_nhl_promotion_contract_hook(league: Any, player: Any, team: Any, season_year: int) -> None:
    hook = getattr(league, "_on_nhl_roster_promotion", None)
    if not callable(hook):
        return
    try:
        hook(player, team, league, int(season_year))
    except Exception:
        pass


def _invoke_roster_make_room_hook(league: Any, team: Any, incoming_player: Any, season_year: int) -> bool:
    hook = getattr(league, "_on_roster_make_room", None)
    if not callable(hook):
        return False
    try:
        result = hook(team, incoming_player, league, int(season_year))
        return bool(isinstance(result, dict) and result.get("ok"))
    except Exception:
        return False


def _active_roster_count_on_team(team: Any) -> int:
    roster = getattr(team, "roster", None) or []
    return sum(
        1 for p in roster
        if not getattr(p, "retired", False)
        and not getattr(p, "is_buried", False)
        and not getattr(p, "buried", False)
        and not getattr(p, "in_minors", False)
    )


def _make_room_before_promotion(
    league: Any,
    team: Any,
    incoming_player: Any,
    year: int,
    roster: List[Any],
    *,
    prune_ctx_pct_u24: float,
    player_roster_age_fn: Any,
) -> None:
    if _active_roster_count_on_team(team) < 23:
        return
    if _invoke_roster_make_room_hook(league, team, incoming_player, year):
        return
    try:
        pu = float(prune_ctx_pct_u24)

        def _p_ovr(p: Any) -> float:
            f = getattr(p, "ovr", None)
            o = float(f()) if callable(f) else 0.5
            a = player_roster_age_fn(p)
            if pu < 22.0:
                o -= 0.014 * max(0, a - 28)
            elif pu > 30.0:
                o += 0.018 * max(0, 23 - a)
            return o

        active = [
            p for p in roster
            if not getattr(p, "retired", False)
            and not getattr(p, "is_buried", False)
            and not getattr(p, "buried", False)
            and not getattr(p, "in_minors", False)
        ]
        if not active:
            return
        worst = min(active, key=_p_ovr)
        try:
            worst.is_buried = True
            worst.buried = True
            worst.in_minors = True
        except Exception:
            pass
    except Exception:
        pass


def build_role_shaped_ratings(
    *,
    position: Position,
    target_ovr: float,
    rng: random.Random,
    profile: Optional[str] = None,
) -> Dict[str, int]:
    """
    Creates real player shape.
    This avoids every generated player becoming the same 74 OVR blob.

    Optional ``profile`` forces an archetype shape (used by real-NHL R2 usage model).
    """
    base = int(max(0.40, min(0.99, float(target_ovr))) * 99)
    ratings: Dict[str, int] = {}

    pos_s = str(getattr(position, "value", position)).upper()

    if pos_s == "G":
        if profile not in ("balanced_g", "butterfly_g", "hybrid_g"):
            profile = rng.choices(
                ["balanced_g", "butterfly_g", "hybrid_g"],
                weights=[0.40, 0.32, 0.28],
                k=1,
            )[0]
        for k in ATTRIBUTE_KEYS:
            lk = k.lower()
            bonus = 0
            if "goalie" in lk or k.startswith("g_"):
                bonus += rng.randint(2, 7)
                if profile == "butterfly_g" and ("position" in lk or "rebound" in lk):
                    bonus += rng.randint(1, 3)
                if profile == "hybrid_g" and ("reflex" in lk or "athlet" in lk):
                    bonus += rng.randint(1, 3)
            elif "off_" in lk or "pm_" in lk:
                bonus -= rng.randint(5, 13)
            ratings[k] = clamp_rating(base + bonus + rng.randint(-3, 3))
        ratings["_generated_profile"] = profile
        return ratings

    if pos_s == "D":
        if profile not in ("two_way_d", "defensive_d", "offensive_d", "enforcer_d"):
            profile = rng.choices(
                ["two_way_d", "defensive_d", "offensive_d", "enforcer_d"],
                weights=[0.42, 0.28, 0.22, 0.08],
                k=1,
            )[0]
    else:
        if profile not in ("two_way", "sniper", "playmaker", "power_forward", "grinder"):
            profile = rng.choices(
                ["two_way", "sniper", "playmaker", "power_forward", "grinder"],
                weights=[0.34, 0.20, 0.22, 0.14, 0.10],
                k=1,
            )[0]

    for k in ATTRIBUTE_KEYS:
        lk = k.lower()
        bonus = 0

        if profile == "sniper":
            if "shot" in lk or "finishing" in lk or "off_" in lk:
                bonus += rng.randint(2, 8)
            if "def_" in lk or "block" in lk:
                bonus -= rng.randint(1, 6)

        elif profile == "playmaker":
            if "pass" in lk or "pm_" in lk or "vision" in lk:
                bonus += rng.randint(3, 8)
            if "shot" in lk or "hit" in lk:
                bonus -= rng.randint(1, 5)

        elif profile == "power_forward":
            if "phy_" in lk or "hit" in lk or "net_front" in lk:
                bonus += rng.randint(3, 8)
            if "skating" in lk or "speed" in lk:
                bonus -= rng.randint(0, 4)

        elif profile == "grinder":
            if "def_" in lk or "phy_" in lk or "hit" in lk or "block" in lk:
                bonus += rng.randint(2, 7)
            if "off_" in lk or "finish" in lk:
                bonus -= rng.randint(2, 7)

        elif profile == "offensive_d":
            if "off_" in lk or "pm_" in lk or "shot" in lk:
                bonus += rng.randint(2, 7)
            if "def_" in lk or "block" in lk:
                bonus -= rng.randint(0, 4)

        elif profile == "defensive_d":
            if "def_" in lk or "block" in lk or "stick" in lk or "iq" in lk:
                bonus += rng.randint(3, 8)
            if "off_" in lk:
                bonus -= rng.randint(1, 5)

        elif profile == "enforcer_d":
            if "phy_" in lk or "hit" in lk or "strength" in lk:
                bonus += rng.randint(4, 9)
            if "off_" in lk or "pm_" in lk:
                bonus -= rng.randint(2, 7)

        else:
            if "iq" in lk or "def_" in lk or "pm_" in lk:
                bonus += rng.randint(0, 4)

        ratings[k] = clamp_rating(base + bonus + rng.randint(-4, 4))

    # Profile is returned as a sidecar meta key; callers must pop + sync archetype.
    # Kept outside ATTRIBUTE_KEYS / normalize so it never becomes a fake 74 attr.
    ratings["_generated_profile"] = profile
    return ratings


def pop_generation_profile(ratings: Dict[str, Any]) -> Optional[str]:
    """Extract and remove `_generated_profile` from a ratings dict."""
    if not isinstance(ratings, dict):
        return None
    prof = ratings.pop("_generated_profile", None)
    return str(prof) if prof else None


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


def _id_str(obj: Any, *attrs: str, default: str = "") -> str:
    """ID-safe attribute extraction: 0 is a valid id; only None/'' count as missing.

    Replaces unsafe `getattr(x, 'team_id', None) or ...` chains that silently
    remapped team_id=0 (Boston) / player_id=0 to fallbacks.
    """
    for a in attrs:
        v = getattr(obj, a, None)
        if v is None:
            continue
        s = str(v)
        if s != "":
            return s
    return default


def _hydrated_identity_name(name: Any, country: Any, rng: random.Random) -> str:
    """Return a real human name for NHL-visible players.

    Legacy world-pool players could carry procedural placeholder identities such
    as "Global JUN 2026-65119". Those must be hydrated into proper generated
    names before entering NHL rosters/stats/awards. Real names pass through
    unchanged.
    """
    nm = str(name or "").strip()
    looks_placeholder = (
        not nm
        or nm.lower() in ("rookie", "prospect", "player", "?")
        or (nm.startswith("Global ") and any(ch.isdigit() for ch in nm))
        or (nm.startswith("Prospect ") and any(ch.isdigit() for ch in nm))
        or nm.startswith("GP_")
    )
    if not looks_placeholder:
        return nm
    try:
        ident = generate_human_identity(rng, nationality=str(country) if country else None)
        return str(ident.full_name)
    except Exception:
        return nm or "Rookie"


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


def _player_contract_years_remaining(player: Any) -> int:
    c = getattr(player, "contract", None)
    for obj in (player, c):
        if obj is None:
            continue
        for key in ("years_remaining", "term_remaining", "remaining_years", "term"):
            v = getattr(obj, key, None)
            if v is not None:
                try:
                    return max(0, int(v))
                except (TypeError, ValueError):
                    pass
    return 1


def _player_recent_production_score(player: Any) -> float:
    stats = getattr(player, "season_stats", None) or getattr(player, "stats", None) or {}
    if isinstance(stats, dict):
        pts = float(stats.get("pts", stats.get("points", 0)) or 0)
        gp = float(stats.get("gp", stats.get("games_played", 0)) or 0)
        if gp > 0:
            return max(0.0, min(1.4, pts / gp))
    return 0.45


def is_bad_contract(player: Any) -> bool:
    cap_hit = _economy_player_cap_hit_millions(player)
    rating = _player_rating_0_100(player)
    age = career_player_age(player)
    years = _player_contract_years_remaining(player)
    production = _player_recent_production_score(player)

    # Expected fair cap hit in millions.
    # Stars can justify money. Depth players cannot.
    # Two-slope curve: linear value through the middle class, plus an elite premium
    # above ~82 so genuine stars ($10M+) are not auto-flagged as bad contracts.
    fair = 1.0 + max(0.0, rating - 58.0) * 0.16 + max(0.0, rating - 82.0) * 0.45

    if production >= 1.0:
        fair *= 1.18
    elif production >= 0.75:
        fair *= 1.08
    elif production < 0.35:
        fair *= 0.86

    if age >= 35:
        fair *= 0.72
    elif age >= 32:
        fair *= 0.84
    elif age <= 24 and rating >= 78:
        fair *= 1.10

    # Term risk: long expensive older contracts are more dangerous.
    term_risk = 1.0
    if years >= 5 and age >= 30:
        term_risk = 1.22
    elif years >= 4 and age >= 32:
        term_risk = 1.32
    elif years <= 1:
        term_risk = 0.92

    badness = (cap_hit / max(0.75, fair)) * term_risk

    # Require both a meaningful relative overpay AND an absolute overpay floor,
    # so marginal deals and cheap depth contracts do not spam bad-contract flags.
    overpay_m = cap_hit - fair
    return bool(badness >= 1.35 and overpay_m >= 1.25 and cap_hit >= 2.0)


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

    affected_players = 0
    avg_multiplier = 1.0

    for player in _team_roster_players(team):
        if getattr(player, "retired", False):
            continue

        base = _player_rating_0_100(player)
        mult = 1.0

        if tier == "cap_hell":
            mult = 0.955
        elif tier == "critical":
            mult = 0.975
        elif tier == "high":
            mult = 0.99
        elif tier == "low":
            mult = 1.012
        elif pressure > 0.9:
            mult = 0.982
        elif pressure < 0.4:
            mult = 1.012

        # High character players handle cap/media drama better.
        try:
            character = float(getattr(player, "character", 50) or 50)
            if character >= 75 and mult < 1.0:
                mult = 1.0 - ((1.0 - mult) * 0.55)
            elif character <= 35 and mult < 1.0:
                mult = 1.0 - ((1.0 - mult) * 1.20)
        except Exception:
            pass

        affected_players += 1
        avg_multiplier += (mult - 1.0)
        setattr(player, "performance_rating", max(1.0, min(120.0, base * mult)))

    if affected_players:
        avg_multiplier = 1.0 + ((avg_multiplier - 1.0) / float(affected_players))

    setattr(
        team,
        "_cap_pressure_effect_summary",
        {
            "tier": tier or "unknown",
            "pressure": float(pressure),
            "affected_players": int(affected_players),
            "avg_multiplier": round(float(avg_multiplier), 4),
            "visible_to_frontend": tier in ("high", "critical", "cap_hell") or pressure > 0.9,
        },
    )

    if tier == "cap_hell":
        st = getattr(team, "state", None)
        if st is not None and hasattr(st, "team_morale"):
            try:
                cur = float(getattr(st, "team_morale", 0.5) or 0.5)
                setattr(st, "team_morale", max(0.12, cur - 0.035))
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
                setattr(psych, "morale", max(0.15, m - 0.017))
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
    """Scales base progression_player draw; gentler peak, clearer post-30 decay."""
    if age <= 20:
        return 1.22
    if age <= 23:
        return 1.08
    if age <= 26:
        return 1.02
    if age <= 29:
        return 0.96
    if age <= 32:
        return 0.86
    if age <= 35:
        return 0.76
    return 0.66


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


def _career_attribute_weights_for_player(player: Any) -> Dict[str, float]:
    """
    Prevent every OVR bump from touching every rating equally.
    A sniper breakout should not magically become a faceoff/shot-blocking god.
    """
    pos = getattr(player, "position", None)
    pos_s = str(getattr(pos, "value", pos) or "").upper()

    style = str(
        getattr(player, "playstyle", None)
        or getattr(player, "archetype", None)
        or getattr(player, "player_type", None)
        or ""
    ).lower()

    if pos_s in ("G", "GOALIE"):
        return {
            "goalie": 1.0,
            "glove": 1.0,
            "blocker": 1.0,
            "rebound": 0.9,
            "angles": 0.9,
            "vision": 0.8,
            "iq": 0.5,
        }

    if "sniper" in style:
        return {
            "shot": 1.25,
            "shoot": 1.25,
            "off": 1.0,
            "puck": 0.85,
            "skating": 0.65,
            "iq": 0.55,
            "def": 0.15,
            "faceoff": 0.05,
            "hit": 0.1,
            "block": 0.1,
        }

    if "playmaker" in style:
        return {
            "pass": 1.25,
            "play": 1.15,
            "puck": 1.0,
            "off": 0.85,
            "iq": 0.85,
            "skating": 0.55,
            "def": 0.2,
            "shot": 0.25,
        }

    if "defensive" in style or "shutdown" in style or pos_s in ("D", "LD", "RD"):
        return {
            "def": 1.2,
            "block": 1.0,
            "stick": 0.9,
            "iq": 0.85,
            "physical": 0.75,
            "hit": 0.7,
            "skating": 0.55,
            "off": 0.25,
            "shot": 0.15,
            "faceoff": 0.05,
        }

    if "grinder" in style or "power" in style:
        return {
            "physical": 1.15,
            "hit": 1.1,
            "def": 0.8,
            "skating": 0.65,
            "iq": 0.45,
            "off": 0.35,
            "shot": 0.3,
        }

    return {
        "off": 0.8,
        "def": 0.65,
        "iq": 0.65,
        "skating": 0.65,
        "shot": 0.55,
        "pass": 0.55,
        "physical": 0.45,
        "faceoff": 0.2,
    }


def _rating_key_weight(key: str, weights: Dict[str, float]) -> float:
    k = str(key or "").lower()

    best = 0.35
    for token, weight in weights.items():
        if token in k:
            best = max(best, float(weight))

    return max(0.02, best)


def _career_apply_rating_delta_0_100(player: Any, delta_0_100: float) -> None:
    """Shift a player's *computed* OVR by ~delta_0_100 (0-99 scale).

    compute_ovr averages 10-20 raw attributes per category, weights the categories,
    then renormalizes by archetype — several layers of dilution between "add X to a
    rating" and "OVR actually moves by X". Naively spreading delta_0_100 across every
    attribute (old behaviour) produced sub-0.01 real OVR movement per seasonal event:
    BREAKOUT / BUST TREND / LATE BLOOM / AGING DECLINE would log but current ability
    (and therefore displayed OVR, especially veteran decline) never visibly changed.

    Fix: probe this player's actual attribute -> OVR sensitivity for their highest-
    leverage rating family, then apply a raw per-attribute change scaled to land on
    the intended delta, spread across a whole family (~10-20 keys) so no single
    attribute has to swing wildly and saturate the 20/99 clamp.
    """
    if abs(delta_0_100) < 1e-9:
        return

    ratings = getattr(player, "ratings", None)
    if not ratings:
        return

    keys = list(ratings.keys())
    if not keys:
        return

    pos = getattr(player, "position", None)
    arch = getattr(player, "archetype", None)

    def _ovr99() -> float:
        try:
            return float(compute_ovr(ratings, pos, arch)) * 99.0
        except Exception:
            return 0.0

    weights = _career_attribute_weights_for_player(player)

    # Group by rating-family prefix (off_, def_, pm_, skg_, ...) so the change lands
    # on a whole category rather than 1-3 isolated attributes.
    families: Dict[str, List[str]] = {}
    for k in keys:
        prefix = str(k).split("_", 1)[0]
        families.setdefault(prefix, []).append(k)
    if not families:
        return
    best_prefix = max(
        families,
        key=lambda fam: sum(_rating_key_weight(k, weights) for k in families[fam]) / max(1, len(families[fam])),
    )
    family_keys = families[best_prefix]

    baseline = _ovr99()
    probe_amt = 1.0 if delta_0_100 > 0 else -1.0
    saved = {k: ratings.get(k) for k in family_keys}
    for k in family_keys:
        try:
            ratings[k] = clamp_rating(float(ratings[k]) + probe_amt)
        except Exception:
            pass
    probed = _ovr99()
    for k, v in saved.items():
        if v is not None:
            ratings[k] = v
    inval = getattr(player, "_invalidate_ovr_memo", None)
    if callable(inval):
        inval()

    sensitivity = (probed - baseline) / probe_amt
    if abs(sensitivity) > 1e-6:
        per_key_raw = float(delta_0_100) / sensitivity
    else:
        per_key_raw = float(delta_0_100)
    # Sanity guard: never let a single event swing one attribute more than ~10 pts,
    # even if the probe measured near-zero sensitivity (e.g. category already clamped).
    per_key_raw = max(-10.0, min(10.0, per_key_raw))

    carry = getattr(player, "_rating_round_carry", None)
    if not isinstance(carry, dict):
        carry = {}

    set_fn = getattr(player, "set", None)
    get_fn = getattr(player, "get", None)
    use_accessors = callable(set_fn) and callable(get_fn)

    for k in family_keys:
        try:
            raw_delta = per_key_raw + float(carry.get(k, 0.0))
            old = float(get_fn(k, 50)) if use_accessors else float(ratings[k])
            new_val = clamp_rating(old + raw_delta)
            carry[k] = (old + raw_delta) - float(new_val)
            if use_accessors:
                set_fn(k, new_val)
            else:
                ratings[k] = new_val
        except Exception:
            pass

    try:
        setattr(player, "_rating_round_carry", carry)
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
_LOG_SPECIAL_PROGRESSION_ENFORCEMENT: bool = False

# --- Central special-progression hard limits (0–100 OVR scale); impossible to exceed via engine apply+log ---
SPECIAL_PROGRESSION_BREAKOUT_HARD_CAP: float = 4.45
SPECIAL_PROGRESSION_LATE_BLOOM_HARD_CAP: float = 3.65
_SPECIAL_TOP_BREAKOUT_DELTA: float = 3.55
_SPECIAL_TOP_LATE_BLOOM_DELTA: float = 2.85
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
    p = 0.021 * brm * nar_bo
    if pot_cat == "elite":
        p *= 1.15
    if tr == "hot":
        p *= 1.08
    if tr == "hot" and ovr_pre >= 83.0:
        p *= 0.65
    used_bo = int(getattr(league, "_prog_global_breakouts_used", 0) or 0) if league is not None else 0
    mx_bo = max(1, int(getattr(league, "_prog_max_breakouts", 8) or 8)) if league is not None else 8
    if league is not None and used_bo >= max(1, int(mx_bo * 0.55)):
        p *= 0.62
    p = max(0.0, min(0.034, p))
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
        mx_bo, mx_lb, mx_bust = 4, 2, 3
    else:
        # Tighter league-wide special events vs inflated OVR seasons (~700 NHL skaters).
        mx_bo = max(3, min(7, int(round(3 + (tp / 780.0) * 3.2))))
        mx_lb = max(2, min(4, int(round(2 + (tp / 950.0) * 2.0))))
        mx_bust = max(2, min(6, int(round(2 + (tp / 650.0) * 3.5))))
    mx_top_bo = max(1, min(2, max(1, mx_bo // 4)))
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
        # Do not claw back players who just posted positive seasonal development.
        ledger = getattr(pl, "development_ledger", None) or {}
        if isinstance(ledger, dict) and ledger.get("development_applied"):
            try:
                ob = float(ledger.get("ovr_before"))
                oa = float(ledger.get("ovr_after"))
                if oa > ob + 0.004:
                    continue
            except Exception:
                pass
        # Protect young breakout / momentum paths from broad inflation clawbacks.
        try:
            age = int(getattr(getattr(pl, "identity", None), "age", None) or getattr(pl, "age", 99) or 99)
            mom = float(getattr(pl, "_dev_breakout_momentum", 0.0) or 0.0)
            if age <= 24 and mom >= 0.35:
                continue
            if age <= 23:
                continue
        except Exception:
            pass
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
    """Role-relative season performance nudge — no raw point cliffs."""
    if getattr(player, "major_progression_event_this_season", None) is not None:
        return
    assign_career_phase_from_age(player)
    phase = str(getattr(player, "career_phase", "") or career_phase_for_age(career_player_age(player)))
    pos = str(getattr(getattr(player, "identity", None), "position", "") or getattr(player, "position", "F") or "F").upper()
    is_goalie = pos == "G" or "GOALIE" in pos
    gp = int(getattr(player, "gp", 0) or getattr(player, "games_played", 0) or 0)
    if gp < 12:
        return

    def _f(name: str, default: float = 0.0) -> float:
        ss = getattr(player, "season_stats", None) or {}
        if not ss:
            return default
        try:
            row = max(ss.values(), key=lambda x: int(x.get("season", 0) or 0))
            return float(row.get(name, default) or default)
        except Exception:
            return default

    pts = _f("points", _f("pts", 0))
    g = _f("goals", _f("g", 0))
    a = _f("assists", _f("a", 0))
    toi = _f("toi_sec", 0)
    ixg = _f("ixg", 0)
    xgf_pct = _f("xgf_pct", 0.5)
    xga60 = _f("xga_per_60", 0)
    blk = _f("blk", _f("blocks", 0))
    gsax = _f("gsax", 0)
    sv_pct = _f("save_pct", _f("sv_pct", 0.9))

    p60 = pts / max(1.0, gp) * (60.0 / max(1.0, toi / 60.0)) if toi > 0 else pts / max(1.0, gp)
    finishing = g - ixg if ixg > 0 else 0.0
    pt = str(getattr(player, "player_type", "") or getattr(player, "archetype", "") or "").lower()

    score = 0.0
    if is_goalie:
        score = (sv_pct - 0.905) * 8.0 + gsax * 0.12
    elif "defensive" in pt or (pos in ("D", "LD", "RD") and "offensive" not in pt):
        score = (0.50 - xga60) * 1.8 + (xgf_pct - 0.50) * 2.5 + blk / max(1.0, gp) * 0.35
    elif "playmaker" in pt:
        score = a / max(1.0, gp) * 0.55 + (xgf_pct - 0.50) * 3.0 + finishing * 0.08
    elif "sniper" in pt or "shooter" in pt:
        score = g / max(1.0, gp) * 0.65 + finishing * 0.22 + p60 * 0.12
    elif "grinder" in pt or "two" in pt:
        score = (xgf_pct - 0.50) * 3.2 + (0.50 - xga60) * 0.8 + pts / max(1.0, gp) * 0.18
    else:
        score = p60 * 0.28 + (xgf_pct - 0.50) * 2.0 + finishing * 0.10

    d = max(-0.85, min(0.65, score * 0.22))
    if phase == PHASE_PRIME:
        d *= 0.55
    d = max(-1.2, min(0.85, d))
    if abs(d) < 0.08:
        return
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
    # Keep system flavor small — modern NHL combined GPG sits ~6.0–6.2.
    if sys == "run_and_gun":
        return 0.14
    if sys == "defensive_lock":
        return -0.10
    if sys == "physical":
        return -0.04
    if sys == "young_fast":
        return 0.07
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


_LEGAL_CRIME_BASE_CHANCE = 0.000015  # ultra-rare league-wide (~1–2 / season target)
_MAJOR_STORYLINE_SEASON_CAP = 2
_LEGAL_MAJOR_SEASON_CAP = 2
_FRANCHISE_MAJOR_DAY_GATE = 0.0025  # daily chance a major slot opens when under cap


def _narr_season_major_count(league: Any, year: int) -> int:
    return int(getattr(league, f"_narr_major_season_{int(year)}", 0) or 0)


def _narr_season_legal_count(league: Any, year: int) -> int:
    return int(getattr(league, f"_narr_legal_major_season_{int(year)}", 0) or 0)


def _bump_narr_season_major(league: Any, year: int, *, legal: bool = False) -> None:
    sk = int(year)
    setattr(league, f"_narr_major_season_{sk}", _narr_season_major_count(league, sk) + 1)
    if legal:
        setattr(league, f"_narr_legal_major_season_{sk}", _narr_season_legal_count(league, sk) + 1)


def _legal_crime_roll_passes(rng: random.Random, char: int, *, franchise_tick: bool = False) -> bool:
    """Gate legal_crime pool picks — ultra rare; blocked when season legal cap reached."""
    p = _LEGAL_CRIME_BASE_CHANCE
    if char < 30:
        p = 0.00008
    elif char < 45:
        p = 0.00003
    if franchise_tick:
        p *= 0.85
    return rng.random() <= p


def _legal_fx_for_severity(severity: str) -> Dict[str, float]:
    if severity == "minor":
        return {"morale": -0.04, "media_stress": 0.06, "chemistry": -0.03, "confidence": -0.03}
    if severity == "moderate":
        return {"morale": -0.10, "media_stress": 0.12, "chemistry": -0.08, "confidence": -0.08}
    return {"morale": -0.18, "media_stress": 0.20, "chemistry": -0.12, "confidence": -0.14}


def _legal_tier_for_severity(severity: str) -> str:
    if severity == "minor":
        return "minor"
    if severity == "moderate":
        return "mid"
    return "major"


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
    ls = str(d.get("legal_severity") or "").lower()
    if ls:
        return _legal_tier_for_severity(ls)
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
        sev = 1.12
        ls = str(picked.get("legal_severity") or "").lower()
        if ls == "minor":
            sev = 0.62
        elif ls == "moderate":
            sev = 0.88
        return "legal_trouble", sev
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
    "off_wrist_shot_accuracy",
    "off_slap_shot_accuracy",
    "off_shot_iq",
    "off_net_front_presence",
    "off_finishing",
]
LINE_CHEM_PASS_KEYS: List[str] = [
    "pm_passing_vision",
    "pm_passing_accuracy",
    "pm_offensive_anticipation",
    "pm_puck_distribution",
]
LINE_CHEM_IQ_KEYS: List[str] = [
    "iqm_awareness",
    "iqm_game_sense",
    "iqm_hockey_iq",
]
LINE_CHEM_ALL_KEYS: List[str] = LINE_CHEM_OFFENSE_KEYS + LINE_CHEM_PASS_KEYS + LINE_CHEM_IQ_KEYS
LINE_CHEM_ELITE_PASS_KEYS: List[str] = [
    "pm_passing_vision",
    "pm_puck_distribution",
    "pm_passing_accuracy",
    "pm_offensive_read",
]
LINE_CHEM_ELITE_SHOT_KEYS: List[str] = [
    "off_wrist_shot_accuracy",
    "off_one_timer",
    "off_shot_iq",
    "off_finishing",
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
        s += float(r.get(k, 68)) / 99.0
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
    Prefer canonical systems.chemistry scores when available.
    """
    if not line:
        return 0.5
    try:
        from app.sim_engine.systems.chemistry import (  # noqa: WPS433
            calculate_defense_pair_chemistry,
            calculate_forward_line_chemistry,
        )

        pos = [_player_position_label(p) for p in line]
        if len(line) >= 2 and all(x == "D" for x in pos):
            score100 = float(calculate_defense_pair_chemistry(line, context={"team": team}).get("chemistry", 50))
        else:
            score100 = float(calculate_forward_line_chemistry(line, context={"team": team}).get("chemistry", 50))
        return clamp(score100 / 100.0, 0.26, 0.91)
    except Exception:
        pass
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
    # Allow larger steps when far from target so real-NHL alignment can converge.
    gap = abs(target01 - cur)
    if gap >= 0.05:
        lo, hi = 0.90, 1.10
    elif gap >= 0.03:
        lo, hi = 0.94, 1.07
    else:
        lo, hi = 0.965, 1.035
    f = max(lo, min(hi, f))
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

    def push(pool: str, text: str, fx: Dict[str, float], **kwargs) -> None:
        out.append({
            "pool": pool,
            "text": text,
            "fx": fx,
            **kwargs
        })

    # ==================================================
    # LEGAL / CRIME / DISCIPLINE STORYLINES — tiered severity
    # ==================================================

    _legal_tiered: List[Tuple[str, str]] = [
        # minor — small morale hit, coach trust drop, media noise
        ("minor", "Excessive speeding ticket after late-night road trip"),
        ("minor", "Noise complaint escalates to police visit at player residence"),
        ("minor", "Public intoxication arrest outside downtown bar"),
        ("minor", "Breaking curfew during team travel"),
        ("minor", "Traffic stop for speeding stunt-driving on highway"),
        ("minor", "Disorderly conduct citation outside hotel after team dinner"),
        # moderate — suspension chance, sponsor loss, chemistry damage
        ("moderate", "Bar fight involvement under police review"),
        ("moderate", "Reckless driving charge after practice"),
        ("moderate", "Vandalism of property complaint filed against player"),
        ("moderate", "Harassment complaint filed by arena staff member"),
        ("moderate", "Hotel room damage incident triggers team review"),
        ("moderate", "Parking lot confrontation captured on phone video"),
        ("moderate", "Fan altercation after game results in investigation"),
        ("moderate", "Driving with suspended license charge surfaces"),
        ("moderate", "Trespassing incident at closed training facility"),
        ("moderate", "Breach of peace charge after heated argument with police"),
        # major — league investigation, indefinite leave, trade value collapse
        ("major", "DUI / impaired driving arrest after team event"),
        ("major", "Domestic violence allegation under preliminary review"),
        ("major", "Betting on games investigation names player"),
        ("major", "Insider betting information scandal linked to player"),
        ("major", "Sexual misconduct allegation triggers league review"),
        ("major", "Fraud investigation involving player business dealings"),
        ("major", "Drug possession charge creates league discipline concern"),
        ("major", "Drug trafficking allegation becomes national story"),
        ("major", "Weapons possession charge after off-ice incident"),
        ("major", "Tax evasion investigation involving player becomes public"),
        ("major", "Non-consensual image-sharing allegation leaks publicly"),
        ("major", "Obstruction of justice allegation during league investigation"),
        ("major", "Assault charge after nightclub incident"),
        ("major", "Stalking allegation leads to restraining order filing"),
        ("major", "Illegal poker room raid connection names player"),
        ("major", "Prescription drug misuse investigation opened"),
        ("major", "Hazing investigation names veteran player"),
        ("major", "Identity fraud allegation tied to off-ice business"),
        ("major", "Civil lawsuit after off-ice altercation goes public"),
        ("major", "Animal cruelty allegation triggers media firestorm"),
        ("major", "Crypto/NFT investment scam connection drags player into headlines"),
    ]

    for i, (sev, lt) in enumerate(_legal_tiered):
        tier = _legal_tier_for_severity(sev)
        push(
            "legal_crime",
            lt,
            _legal_fx_for_severity(sev),
            legal=True,
            legal_severity=sev,
            tier=tier,
            rarity="rare" if sev == "major" else "uncommon",
            tone="negative",
            polarity="negative",
        )

    # ==================================================
    # MENTAL / PSYCHOLOGICAL STORYLINES — 60
    # ==================================================

    mental_lines = [
        "Confidence collapses after repeated defensive mistakes",
        "Player admits pressure has been affecting his game",
        "Sports psychologist begins working with player privately",
        "Player shows signs of burnout during heavy schedule",
        "Long scoring drought creates visible frustration",
        "Player becomes withdrawn after costly overtime turnover",
        "Bench demotion shakes player’s confidence",
        "Goalie loses composure after string of weak goals",
        "Young player overwhelmed by sudden top-line role",
        "Veteran struggles emotionally after losing leadership role",
        "Player returns from personal reset with sharper focus",
        "Mental fatigue becomes concern during road-heavy month",
        "Player speaks openly about needing a confidence rebuild",
        "Coaching staff notices player overthinking simple plays",
        "Player starts gripping the stick too tightly during slump",
        "Media pressure causes player to avoid interviews",
        "Player regains confidence after strong practice week",
        "Leadership group helps struggling player reset mentally",
        "Player’s anxiety rises before rivalry matchup",
        "Player looks rattled after hostile road crowd reaction",
        "Goaltender requests extra work with mental performance coach",
        "Player develops pregame routine to manage nerves",
        "Team worries player is spiraling after social media criticism",
        "Player blocks out media and goes quiet publicly",
        "Pressure of contract year weighs heavily on player",
        "Rookie admits NHL pace is mentally exhausting",
        "Veteran becomes mentor during teammate’s confidence crisis",
        "Player responds well after private meeting with coach",
        "Player’s body language alarms coaching staff",
        "Player avoids puck in key moments during confidence dip",
        "Player rebuilds confidence after shootout winner",
        "Mental reset day away from rink helps player recover",
        "Player’s frustration boils over during practice drill",
        "Team considers reducing player’s minutes to protect confidence",
        "Player uses criticism as motivation after bad headlines",
        "Player becomes obsessed with stat tracking during slump",
        "Player’s fear of mistakes hurts offensive creativity",
        "Coach publicly protects player from mounting criticism",
        "Player admits he has not felt like himself lately",
        "Player slowly regains swagger after physical game",
        "Teammates rally around player after emotional interview",
        "Player’s confidence spikes after promotion to power play",
        "Player’s slump turns into full mental block",
        "Goalie battles nerves after being pulled twice in a week",
        "Player appears mentally refreshed after family visit",
        "Player struggles with focus after trade rumors",
        "Player handles pressure better after veteran guidance",
        "Player’s confidence grows after strong defensive assignment",
        "Player enters quiet leadership phase after adversity",
        "Mental toughness praised after bounce-back performance",
        "Player’s frustration with himself becomes obvious on bench",
        "Player requests film session to rebuild trust in his game",
        "Coach gives player simplified role to reduce pressure",
        "Player’s confidence dips after being scratched",
        "Player regains edge after emotional players-only meeting",
        "Rookie’s confidence shaken after viral mistake clip",
        "Player’s mental resilience becomes locker room storyline",
        "Player starts journaling as part of performance routine",
        "Heavy expectations begin affecting player’s decision-making",
        "Player finally breaks through after weeks of visible tension",
    ]

    for i, mt in enumerate(mental_lines):
        positive = any(word in mt.lower() for word in ["regains", "reset", "refreshed", "praised", "resilience", "breaks through", "motivation"])
        push(
            "mental_psychological",
            mt,
            {
                "morale": 0.08 if positive else -0.10,
                "confidence": 0.14 if positive else -0.16,
                "chemistry": 0.04 if positive else -0.04,
            },
            rarity="common",
            tone="positive" if positive else "negative",
        )

    # ==================================================
    # PERSONAL LIFE STORYLINES — 60
    # ==================================================

    personal_lines = [
        "Birth of child gives player emotional boost",
        "Family emergency causes player to miss practice",
        "Player dedicates strong game to sick family member",
        "Wedding planning becomes distraction during playoff race",
        "Public breakup creates unwanted media attention",
        "Relationship rumors follow player during road trip",
        "Player’s family relocation helps him settle in market",
        "Homesickness affects young European player",
        "Player buys home in city, signaling long-term commitment",
        "New parenthood affects player’s sleep and routine",
        "Player’s charity work earns praise from local community",
        "Family tragedy weighs heavily on player’s performance",
        "Player returns from bereavement leave with emotional goal",
        "Player’s sibling signs pro contract, creating proud moment",
        "Player’s parent gives emotional interview after milestone game",
        "Private family issue kept player away from optional skate",
        "Player hosts youth hockey camp during off day",
        "Player becomes active in local hospital visits",
        "Player’s spouse criticizes organization on social media",
        "Family travel issues delay player before road game",
        "Player’s child appears at practice and lifts locker room mood",
        "Player struggles balancing fatherhood and road schedule",
        "Player’s offseason home purchase becomes local headline",
        "Player’s family dislikes trade destination rumors",
        "Rookie’s parents attend first NHL game, boosting morale",
        "Player quietly supports teammate through family crisis",
        "Personal milestone creates positive locker room energy",
        "Player’s engagement announcement goes viral among fans",
        "Divorce rumors create distraction during contract talks",
        "Player changes routine to spend more time with family",
        "Player credits family support for improved play",
        "Player’s charity foundation launches in home market",
        "Family illness leads to emotional postgame interview",
        "Player misses morning skate for birth of child",
        "Player returns from family leave with renewed focus",
        "Player’s personal life becomes tabloid story",
        "Teammates support player after difficult family news",
        "Player’s partner relocates, improving off-ice stability",
        "Player considers trade request for family reasons",
        "Player’s family publicly praises city and fanbase",
        "Personal stress affects player’s practice intensity",
        "Player’s mentor from childhood passes away",
        "Player honors late friend with strong performance",
        "Player reconnects with old junior coach for support",
        "Player’s family celebrates citizenship milestone",
        "Off-ice routine stabilizes after difficult month",
        "Player’s private life leaks into media cycle",
        "Player deletes social media after family harassment",
        "Player hosts teammates for family dinner, improving chemistry",
        "Player’s newborn illness creates emotional strain",
        "Player becomes guardian figure for younger sibling",
        "Family vacation incident becomes minor team distraction",
        "Player’s community involvement improves public image",
        "Player’s personal growth impresses coaching staff",
        "Player struggles after being away from family too long",
        "Player’s family attends ceremony for career milestone",
        "Player credits spouse for helping him through slump",
        "Player takes leave to handle urgent personal matter",
        "Player’s off-ice happiness translates into better play",
        "Player’s personal adversity becomes rallying point",
        "Gets married during the offseason",
        "Has a child and becomes more motivated",
        "Goes through a breakup/divorce and struggles mentally",
        "Buys a new home in the city",
        "Has trouble adjusting to a new city after a trade",
        "Misses family back home",
        "Becomes homesick as a young prospect",
        "Starts dating a celebrity/local influencer",
        "Gets caught partying too much",
        "Becomes more mature after settling down",
    ]

    for i, pt in enumerate(personal_lines):
        positive = any(word in pt.lower() for word in ["boost", "praise", "settle", "charity", "support", "improved", "renewed", "stabilizes", "happiness", "rallying"])
        push(
            "personal_life",
            pt,
            {
                "morale": 0.10 if positive else -0.08,
                "confidence": 0.06 if positive else -0.05,
                "chemistry": 0.05 if positive else -0.03,
            },
            rarity="common",
            tone="positive" if positive else "mixed",
        )

    # ==================================================
    # MEDIA PRESSURE STORYLINES — 60
    # ==================================================

    media_lines = [
        "Local radio calls player the biggest disappointment of the season",
        "National broadcast questions player’s commitment level",
        "Player’s postgame quote sparks controversy",
        "Beat reporter hints at tension between player and coach",
        "Player trends online after brutal defensive mistake",
        "Fanbase turns on player after another quiet night",
        "Player praised by analysts for elite two-way effort",
        "Media labels player a future captain after strong month",
        "Viral highlight boosts player’s popularity overnight",
        "Player refuses interview after being benched",
        "Reporter asks uncomfortable question about contract value",
        "Player’s sarcastic answer becomes headline",
        "Coach defends player during tense media scrum",
        "Player becomes target of trade deadline speculation",
        "Podcast rumor creates unnecessary locker room drama",
        "Player’s analytics profile becomes hot debate online",
        "Media credits player for changing team culture",
        "Fan poll ranks player as most frustrating on roster",
        "Player’s playoff struggles dominate morning shows",
        "Player’s hot streak becomes national story",
        "Media compares player unfavorably to former teammate",
        "Player calls out unfair coverage after loss",
        "Team PR limits player availability after controversy",
        "Player wins back fans with honest interview",
        "Player’s quiet leadership praised by insiders",
        "Columnist says player has lost a step",
        "Player’s goal celebration becomes viral meme",
        "Misquoted interview creates needless controversy",
        "Player clarifies comments after social media backlash",
        "Analyst claims player is being misused by coaching staff",
        "Media questions whether player can handle market pressure",
        "Player laughs off trade rumors in confident interview",
        "Reporter reveals player has been playing through injury",
        "Player’s leadership questioned after locker room leak",
        "Fan chants support player after difficult week",
        "Player’s slow start becomes daily media obsession",
        "Rookie receives overwhelming media attention after debut",
        "Player publicly thanks fans after hostile stretch",
        "Media praises player’s response to adversity",
        "Player criticized for skipping optional media availability",
        "Panel debate argues player deserves more ice time",
        "Former player rips his effort on national TV",
        "Player responds with dominant performance after criticism",
        "Media narrative shifts after one massive game",
        "Player’s agent leaks frustration through reporter",
        "Team insider reports player is unhappy with role",
        "Player becomes face of team’s collapse storyline",
        "Player’s comeback story gains national attention",
        "Fans defend player after unfair media pile-on",
        "Press conference joke lands badly with fanbase",
        "Player’s old quote resurfaces at worst possible time",
        "Media pressure increases after captain avoids questions",
        "Player becomes symbol of failed offseason plan",
        "Analysts praise player’s maturity under pressure",
        "Media speculates player may waive no-trade clause",
        "Player’s emotional interview changes public perception",
        "Reporter feud with player becomes unwanted subplot",
        "Player’s slump is dissected frame-by-frame online",
        "Media declares player has turned season around",
        "Player’s reputation improves after accountability interview",
        "Gives a controversial interview",
        "Becomes a fan favourite through charity work",
        "Gets criticized by local media",
        "Deletes social media after backlash",
        "Goes viral for a funny locker room clip",
        "Starts a podcast and creates distractions",
        "Makes a bad joke publicly and has to apologize",
        "Becomes the face of the franchise marketing campaign",
        "Gets booed by home fans after poor play",
        "Gets praised nationally after a leadership moment",
    ]

    for i, ml in enumerate(media_lines):
        positive = any(word in ml.lower() for word in ["praised", "praises", "credits", "support", "wins back", "dominant", "comeback", "maturity", "turned", "improves"])
        push(
            "media_pressure",
            ml,
            {
                "morale": 0.06 if positive else -0.08,
                "confidence": 0.10 if positive else -0.12,
                "media": 0.18,
            },
            rarity="common",
            tone="positive" if positive else "negative",
        )

    # ==================================================
    # TEAM DYNAMICS STORYLINES — 60
    # ==================================================

    team_lines = [
        "Locker room tension grows after player criticizes effort",
        "Player and coach clash over reduced ice time",
        "Veteran calls out young teammates during practice",
        "Players-only meeting sparks improved team focus",
        "Player emerges as unexpected locker room leader",
        "Line chemistry explodes after new combination is tested",
        "Player refuses to blame goalie after ugly loss",
        "Teammates frustrated by player’s risky turnovers",
        "Captain privately challenges player to raise intensity",
        "Player earns respect after blocking shots while hurt",
        "Practice fight breaks out between two competitive teammates",
        "Player apologizes after heated bench exchange",
        "Assistant coach praises player’s attitude improvement",
        "Veteran mentors struggling rookie through difficult stretch",
        "Player feels isolated after trade rumors swirl",
        "Locker room rallies behind player after media attack",
        "Player’s selfish penalties frustrate coaching staff",
        "Leadership group demands more accountability from player",
        "Player buys dinner for team after milestone night",
        "Rookie gains trust after strong defensive effort",
        "Player loses trust after ignoring system details",
        "Coach rewards player with late-game defensive assignment",
        "Player’s positive attitude lifts room during losing streak",
        "Teammates notice player staying late after practice",
        "Player becomes bridge between veterans and young core",
        "Bench argument creates awkward postgame atmosphere",
        "Player’s effort level questioned internally",
        "Coaching staff credits player for changing practice tempo",
        "Player takes blame publicly to protect teammate",
        "Line mate frustration grows over missed passes",
        "Player’s leadership improves after captain’s injury",
        "Player organizes team bonding event on road trip",
        "Locker room divides over player’s role on power play",
        "Player earns alternate captain consideration",
        "Coach benches player to send message to roster",
        "Player responds to benching with mature attitude",
        "Veterans unhappy with player’s defensive shortcuts",
        "Player becomes trusted voice during rough stretch",
        "Chemistry with new linemates transforms his season",
        "Player privately unhappy with deployment",
        "Goalie thanks player for defensive support after win",
        "Team morale improves after player’s emotional speech",
        "Player’s ego becomes concern among teammates",
        "Younger players gravitate toward veteran’s guidance",
        "Player resents being scratched for accountability reasons",
        "Locker room celebrates player’s first goal after slump",
        "Player’s work ethic sets tone for entire practice",
        "Coach questions whether player is buying into system",
        "Player publicly backs coach despite fan criticism",
        "Teammate praises player for handling criticism well",
        "Player’s defensive laziness creates staff frustration",
        "Player accepts reduced role for team success",
        "Team bonds around player returning from personal leave",
        "Player’s complaining begins irritating teammates",
        "Player becomes emotional heartbeat of fourth line",
        "Player loses power play spot, creating quiet tension",
        "Coach trusts player in final minute for first time",
        "Player’s unselfish play improves line chemistry",
        "Veteran leadership calms team after chaotic week",
        "Player’s attitude turnaround changes internal perception",
        "Has tension with the head coach",
        "Becomes close friends with a teammate",
        "Mentors a rookie",
        "Clashes with a veteran leader",
        "Loses confidence after being scratched",
        "Requests a bigger role privately",
        "Quietly asks management about a trade",
        "Is voted alternate captain",
        "Loses the room's trust after selfish comments",
        "Organizes a players-only meeting",
    ]

    for i, tl in enumerate(team_lines):
        positive = any(word in tl.lower() for word in ["improved", "leader", "chemistry", "respect", "praises", "mentors", "rallies", "trust", "positive", "bonding", "mature", "trusted", "unselfish", "calms", "turnaround"])
        push(
            "team_dynamics",
            tl,
            {
                "morale": 0.08 if positive else -0.08,
                "chemistry": 0.14 if positive else -0.14,
                "discipline": 0.04 if positive else -0.06,
            },
            rarity="common",
            tone="positive" if positive else "negative",
        )

    # ==================================================
    # MONEY / CAREER / CONTRACT STORYLINES — 60
    # ==================================================

    career_lines = [
        "Contract talks stall after player rejects first offer",
        "Player’s agent leaks frustration over negotiation pace",
        "Team worries player may price himself out of market",
        "Player takes hometown discount to stay with club",
        "Extension talks create positive buzz around locker room",
        "Trade rumors intensify after player changes agents",
        "Player quietly asks management for clarity on future",
        "Agent pushes for larger role before extension talks",
        "Player’s production dips during contract year pressure",
        "Career-best season strengthens player’s arbitration case",
        "Team explores trade market due to cap concerns",
        "Player unhappy with bridge deal proposal",
        "Negotiations pause until after playoff race",
        "Player says he wants to retire with organization",
        "Long-term extension rumor energizes fanbase",
        "Player refuses to discuss contract during slump",
        "Salary dispute becomes distraction during road trip",
        "Player’s no-trade clause becomes major storyline",
        "Management uncertain about committing long term",
        "Player bets on himself by delaying extension",
        "Agent publicly denies trade request rumor",
        "Player’s camp unhappy with special teams usage",
        "Bonus clause controversy creates awkward situation",
        "Player’s arbitration filing shocks front office",
        "Team fears player could walk in free agency",
        "Player’s role expands after trade deadline passes",
        "Career decline concerns affect player’s value",
        "Player’s resurgence changes front office plans",
        "Retirement speculation grows around veteran",
        "Player considers Europe if NHL role disappears",
        "Prospect worries about being buried in depth chart",
        "Player’s call-up chance becomes career crossroads",
        "Waiver risk creates tension around roster decisions",
        "Player accepts two-way deal to stay in system",
        "Veteran wants one more playoff run before retirement",
        "Player’s camp pushes back against healthy scratch pattern",
        "Team receives calls on player but hesitates",
        "Player’s leadership value complicates trade decision",
        "Extension deadline pressure affects player’s focus",
        "Player changes offseason trainer to revive career",
        "Management sees player as future core piece",
        "Player’s inconsistency hurts negotiation leverage",
        "Trade protection blocks potential deadline move",
        "Player open to relocation for bigger opportunity",
        "Agent meeting with GM sparks speculation",
        "Player’s playoff performance may decide his future",
        "Rookie ELC bonuses become quiet cap concern",
        "Veteran’s cap hit becomes fanbase obsession",
        "Player embraces prove-it contract mentality",
        "Team wants shorter term than player expects",
        "Player’s loyalty praised after rejecting rival offer",
        "Rumored extension falls apart over signing bonus",
        "Player’s market value rises after injury replacement success",
        "Front office debates buying out veteran’s contract",
        "Player’s trade value peaks during hot streak",
        "Player wants stability after multiple deadline rumors",
        "Team’s cap crunch places player in awkward position",
        "Player’s camp seeks modified no-trade protection",
        "Management praises player while avoiding contract questions",
        "Player’s future becomes defining offseason storyline",
        "Signs a major endorsement deal",
        "Starts a clothing brand",
        "Invests in a local restaurant",
        "Gets into financial trouble from bad investments",
        "Fires his agent",
        "Changes agents before contract talks",
        "Public contract dispute hurts morale",
        "Donates money to a community cause",
        "Becomes involved in a charity tournament",
        "Gets named to a league/player association committee",
    ]

    for i, cl in enumerate(career_lines):
        positive = any(word in cl.lower() for word in ["discount", "positive", "retire", "energizes", "resurgence", "future core", "loyalty", "rises", "stability", "praises"])
        push(
            "money_career",
            cl,
            {
                "morale": 0.07 if positive else -0.07,
                "confidence": 0.06 if positive else -0.05,
                "media": 0.10,
            },
            rarity="common",
            tone="positive" if positive else "mixed",
        )

    # ==================================================
    # PERFORMANCE / HOCKEY-SPECIFIC STORYLINES — 60
    # ==================================================

    performance_lines = [
        "Player changes stick curve after prolonged scoring drought",
        "Skating coach helps player regain first-step quickness",
        "Player adds physical edge after being criticized as soft",
        "Faceoff specialist begins mentoring young center",
        "Defenseman simplifies game after brutal turnover stretch",
        "Goalie adjusts glove positioning after repeated high shots",
        "Player’s shot volume spikes after coaching adjustment",
        "Power play role unlocks player’s offensive confidence",
        "Penalty kill promotion improves player’s overall trust",
        "Player studies film obsessively after defensive breakdowns",
        "Coach praises player for finally attacking middle ice",
        "Player’s zone entries become key tactical weapon",
        "Player struggles with new defensive system",
        "Player thrives after switching wings",
        "Rookie learns to manage NHL pace after rough start",
        "Veteran loses step but improves positional awareness",
        "Player’s conditioning becomes issue late in games",
        "Player’s offseason training finally shows results",
        "Goalie’s rebound control becomes staff concern",
        "Player’s finishing luck finally normalizes",
        "Player’s shooting percentage crash alarms coaches",
        "Player’s underlying numbers suggest breakout is coming",
        "Player becomes trusted matchup weapon against stars",
        "Player’s defensive reads improve dramatically",
        "Coaching staff reduces player’s minutes to simplify role",
        "Player earns more ice time through strong forecheck",
        "Player’s backchecking effort wins over coaching staff",
        "Defense pair struggles badly with communication",
        "Player discovers chemistry with unexpected linemate",
        "Goalie works with new goalie coach to fix stance",
        "Player’s passing creativity returns after confidence boost",
        "Player’s turnovers force temporary demotion",
        "Young defender gains confidence after first NHL goal",
        "Player’s heavy shot becomes power play focal point",
        "Player’s neutral zone play frustrates opponents",
        "Player’s discipline improves after penalty-heavy month",
        "Player gets rewarded for net-front presence",
        "Player’s defensive stick becomes quiet strength",
        "Player’s transition game drives team offense",
        "Player’s poor gap control becomes video-room focus",
        "Coach gives player tougher assignments to test growth",
        "Player embraces shutdown role after offensive slump",
        "Player adds deception to shot release",
        "Rookie’s defensive detail earns veteran praise",
        "Player’s lack of pace creates lineup questions",
        "Player’s board battles improve after strength work",
        "Player earns trust as late-game defensive option",
        "Goalie’s workload becomes concern after heavy stretch",
        "Player’s cycle game wears down opponents",
        "Player’s puck protection becomes major development win",
        "Player’s rush defense remains major weakness",
        "Player starts driving play despite low point totals",
        "Player’s net-front screens create hidden value",
        "Player’s one-timer finally becomes dangerous weapon",
        "Player’s defensive-zone exits stabilize his pair",
        "Player’s penalty trouble costs team momentum",
        "Coach rewards player for consistent details",
        "Player’s game matures after midseason reset",
        "Player’s special teams versatility boosts value",
        "Player’s all-around game takes noticeable leap",
        "Changes offseason trainer and improves skating",
        "Adds muscle over the summer",
        "Loses speed due to poor conditioning",
        "Works with a shooting coach",
        "Improves faceoffs after extra practice",
        "Hires a sports psychologist",
        "Changes diet and improves stamina",
        "Comes into camp out of shape",
        "Studies film obsessively and improves hockey IQ",
        "Switches stick/skate brand and has an adjustment period",
    ]

    for i, pl in enumerate(performance_lines):
        negative = any(word in pl.lower() for word in ["struggles", "concern", "issue", "crash", "turnovers", "poor", "weakness", "trouble"])
        push(
            "performance_hockey",
            pl,
            {
                "morale": -0.05 if negative else 0.07,
                "confidence": -0.08 if negative else 0.10,
                "chemistry": -0.03 if negative else 0.04,
            },
            rarity="common",
            tone="negative" if negative else "positive",
        )

    # ==================================================
    # CHAOTIC / WEIRD / NHL-STYLE ABSURDITY — 60
    # ==================================================

    weird_lines = [
        "Player blames scoring slump on cursed hotel room",
        "Player switches pregame meal and immediately scores twice",
        "Teammates ban player’s lucky shirt after losing streak",
        "Player’s bizarre superstition becomes locker room joke",
        "Fan conspiracy theory claims player only scores on Tuesdays",
        "Player’s dog becomes unofficial team mascot",
        "Player loses equipment bag before rivalry game",
        "Player accidentally wears wrong gloves during warmup",
        "Goalie refuses to change mask after shutout streak",
        "Player’s lucky stick breaks and panic spreads online",
        "Player becomes meme after strange bench reaction",
        "Teammates recreate player’s odd celebration in practice",
        "Player insists new tape job changed his season",
        "Arena DJ plays wrong goal song and player gets blamed",
        "Player’s coffee addiction becomes running locker room gag",
        "Rookie forced into absurd team karaoke tradition",
        "Player claims road city has cursed ice",
        "Equipment manager becomes hero after skate emergency",
        "Player accidentally likes trade rumor post",
        "Player’s fantasy football punishment goes viral",
        "Teammates hide player’s helmet before practice prank",
        "Player starts growing playoff beard two months early",
        "Player’s strange warmup dance becomes fan favorite",
        "Player refuses to step on team logo after bad luck claim",
        "Veteran bans rookies from touching stereo after losses",
        "Player’s unusual smelling salts routine alarms broadcast crew",
        "Player misses bus after falling asleep in hotel lobby",
        "Player’s pregame playlist causes locker room debate",
        "Goalie talks to posts during hot streak",
        "Player’s lucky socks become local merchandise idea",
        "Teammate accuses player of jinxing shutout bid",
        "Player’s broken stick gets displayed like trophy",
        "Player starts using same parking spot during winning streak",
        "Player’s odd handshake ritual spreads through team",
        "Fan brings sign about player’s superstition and he scores",
        "Player’s helmet sticker placement becomes online obsession",
        "Player’s accidental quote becomes team slogan",
        "Coach jokes player is powered by gas station snacks",
        "Player gets stuck in elevator before morning skate",
        "Player’s luggage sent to wrong city before back-to-back",
        "Player blames poor game on bad nap timing",
        "Team refuses to change hotel after comeback win",
        "Player’s lucky hoodie goes missing before big game",
        "Player’s unusual stretching routine confuses rookies",
        "Mascot prank involving player goes viral",
        "Player gets roasted for terrible aux cord choices",
        "Teammates credit player’s ugly shoes for winning streak",
        "Player becomes obsessed with same postgame restaurant",
        "Player’s accidental fall during warmup becomes meme",
        "Goalie changes water bottle order after bad outing",
        "Player claims new mouthguard improves vision",
        "Broadcast catches player arguing with broken stick",
        "Player’s glove smell becomes locker room controversy",
        "Player forgets rookie lap and gets chirped for weeks",
        "Team starts fake award for player’s weird habits",
        "Player’s superstition delays entire warmup routine",
        "Player’s lucky charm confiscated by equipment staff",
        "Fans chant about player’s strange celebration",
        "Player’s bizarre confidence quote becomes viral audio",
        "Locker room adopts ridiculous motto after player joke",
    ]

    for i, wl in enumerate(weird_lines):
        push(
            "chaotic_weird",
            wl,
            {
                "morale": 0.05,
                "confidence": 0.03,
                "chemistry": 0.06,
                "media": 0.08,
            },
            rarity="uncommon",
            tone="chaotic",
        )

    return out


# Modern 32-team NHL-style alignment (8 teams × 4 divisions). Procedural sandbox teams.
_FRANCHISE_NHL_TEAM_SPECS: List[Tuple[str, str, str, str, str]] = [
    ("Boston", "Bruins", "BOS", "Atlantic", "Eastern"),
    ("Buffalo", "Sabres", "BUF", "Atlantic", "Eastern"),
    ("Detroit", "Red Wings", "DET", "Atlantic", "Eastern"),
    ("Florida", "Panthers", "FLA", "Atlantic", "Eastern"),
    ("Montreal", "Canadiens", "MTL", "Atlantic", "Eastern"),
    ("Ottawa", "Senators", "OTT", "Atlantic", "Eastern"),
    ("Tampa Bay", "Lightning", "TBL", "Atlantic", "Eastern"),
    ("Toronto", "Maple Leafs", "TOR", "Atlantic", "Eastern"),
    ("Carolina", "Hurricanes", "CAR", "Metropolitan", "Eastern"),
    ("Columbus", "Blue Jackets", "CBJ", "Metropolitan", "Eastern"),
    ("New Jersey", "Devils", "NJD", "Metropolitan", "Eastern"),
    ("New York", "Islanders", "NYI", "Metropolitan", "Eastern"),
    ("New York", "Rangers", "NYR", "Metropolitan", "Eastern"),
    ("Philadelphia", "Flyers", "PHI", "Metropolitan", "Eastern"),
    ("Pittsburgh", "Penguins", "PIT", "Metropolitan", "Eastern"),
    ("Washington", "Capitals", "WSH", "Metropolitan", "Eastern"),
    ("Chicago", "Blackhawks", "CHI", "Central", "Western"),
    ("Colorado", "Avalanche", "COL", "Central", "Western"),
    ("Dallas", "Stars", "DAL", "Central", "Western"),
    ("Minnesota", "Wild", "MIN", "Central", "Western"),
    ("Nashville", "Predators", "NSH", "Central", "Western"),
    ("St. Louis", "Blues", "STL", "Central", "Western"),
    ("Utah", "Mammoth", "UTA", "Central", "Western"),
    ("Winnipeg", "Jets", "WPG", "Central", "Western"),
    ("Anaheim", "Ducks", "ANA", "Pacific", "Western"),
    ("Calgary", "Flames", "CGY", "Pacific", "Western"),
    ("Edmonton", "Oilers", "EDM", "Pacific", "Western"),
    ("Los Angeles", "Kings", "LAK", "Pacific", "Western"),
    ("San Jose", "Sharks", "SJS", "Pacific", "Western"),
    ("Seattle", "Kraken", "SEA", "Pacific", "Western"),
    ("Vancouver", "Canucks", "VAN", "Pacific", "Western"),
    ("Vegas", "Golden Knights", "VGK", "Pacific", "Western"),
]

_GM_STRATEGY_BY_ARCHETYPE: Dict[str, Tuple[str, str, str]] = {
    TeamArchetype.WIN_NOW: ("contender", "aggressive_buyer", "win_now"),
    TeamArchetype.PATIENT_BUILDER: ("retool", "conservative_builder", "balanced"),
    TeamArchetype.DRAFT_AND_DEVELOP: ("rebuild", "prospect_pipeline", "draft_focus"),
    TeamArchetype.MEDIOCRE: ("bubble", "patient_builder", "balanced"),
    TeamArchetype.CHAOTIC: ("bubble", "aggressive_buyer", "chaotic"),
}


def _assign_franchise_team_personality(team: Any, rng: random.Random) -> None:
    """Assign GM window, strategy, and named GM for trade AI / franchise screens."""
    arch = str(getattr(team, "archetype", TeamArchetype.MEDIOCRE) or TeamArchetype.MEDIOCRE)
    window, strategy, risk_band = _GM_STRATEGY_BY_ARCHETYPE.get(
        arch,
        ("bubble", "conservative_builder", "balanced"),
    )
    setattr(team, "gm_window", window)
    setattr(team, "window", window)
    setattr(team, "gm_strategy", strategy)
    setattr(team, "gm_risk_band", risk_band)
    try:
        ident = generate_human_identity(rng)
        setattr(team, "gm_name", str(getattr(ident, "full_name", "GM")))
    except Exception:
        setattr(team, "gm_name", f"GM {getattr(team, 'abbreviation', getattr(team, 'city', ''))}")
    try:
        from app.sim_engine.entities.team import TeamStatus as _TS

        if window == "contender":
            team.state.status = _TS.CONTENDING
        elif window == "rebuild":
            team.state.status = _TS.REBUILDING
        elif window == "retool":
            team.state.status = _TS.RETOOLING
        else:
            team.state.status = _TS.BUBBLE
    except Exception:
        pass


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

    def __init__(
        self,
        seed: int | None = None,
        debug: bool = False,
        *,
        populate_initial_rosters: bool = True,
    ):
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
        self._populate_initial_rosters_enabled: bool = bool(populate_initial_rosters)

        # ------------------------------------
        # League ecosystem (MACRO ONLY)
        # ------------------------------------
        self.league: League = League(seed=self.seed)

        # Cached league outputs for the latest season
        self.last_league_context: dict | None = None
        self.last_league_forecast: dict | None = None
        self.last_league_shocks: list[dict] = []
        self.league_history: list[LeagueSeasonResult] = []

        # Last full-league sim: game-derived stat ledger keyed by player id (str)
        self._last_league_season_stat_ledger: Dict[str, Dict[str, Any]] = {}
        self._last_league_sim_calendar_year: int = 0
        self._last_league_season_validation: Dict[str, Any] = {}

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
    # Live franchise mode uses this path — NOT engine_universe.py macro sim.
    # Franchise mode uses game ledger stats only. Do not call abstract season
    # stat simulation here.
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
        specs = _FRANCHISE_NHL_TEAM_SPECS[: max(1, min(int(team_count), 32))]
        archetype_pool = [
            TeamArchetype.WIN_NOW,
            TeamArchetype.PATIENT_BUILDER,
            TeamArchetype.DRAFT_AND_DEVELOP,
            TeamArchetype.MEDIOCRE,
            TeamArchetype.CHAOTIC,
        ]
        for i, (city, name, abbr, division, conference) in enumerate(specs):
            archetype = archetype_pool[i % len(archetype_pool)]
            team = Team(
                team_id=i,
                city=city,
                name=name,
                division=division,
                conference=conference,
                archetype=archetype,
                rng=rng,
            )
            setattr(team, "abbreviation", abbr)
            setattr(team, "abbr", abbr)
            setattr(team, "full_name", f"{city} {name}".strip())
            coach = generate_coach(rng, f"COACH_{i:03d}", CoachRole.HEAD_COACH)
            team.coach = coach
            assign_team_system(team, rng)
            assign_team_coach_profile(team, rng)
            _assign_franchise_team_personality(team, rng)
            self.league.teams.append(team)
        self.league.identity.max_teams = len(specs)
        if getattr(self, "_populate_initial_rosters_enabled", True):
            self._populate_initial_rosters()
        else:
            if not hasattr(self.league, "players") or self.league.players is None:
                self.league.players = []
            if not hasattr(self.league, "retired_players") or self.league.retired_players is None:
                self.league.retired_players = []
            for team in self.league.teams:
                if not hasattr(team, "roster") or team.roster is None:
                    team.roster = []
                if not hasattr(team, "scratches") or team.scratches is None:
                    team.scratches = []
                if not hasattr(team, "injured_reserve") or team.injured_reserve is None:
                    team.injured_reserve = []
                team.roster.clear()
                team.scratches.clear()

    def _populate_initial_rosters(self) -> None:
  
        if not getattr(self.league, "teams", None) or len(self.league.teams) == 0:
            return

        rng = self.rng

        if not hasattr(self.league, "players") or self.league.players is None:
            self.league.players = []

        if not hasattr(self.league, "retired_players") or self.league.retired_players is None:
            self.league.retired_players = []

        season_year = int(
            getattr(self.league, "season_year", None)
            or getattr(self.league, "current_season", None)
            or 2025
        )

        age_ranges = (
            [(31, 36)] * 2
            + [(24, 29)] * 7
            + [(20, 23)] * 7
            + [(26, 32)] * 4
            + [(22, 30)] * 3
        )

        # 23-man NHL roster target: 13F / 7D / 3G (active lineup UI still uses 12/6/2)
        fwd_positions = [Position.C] * 5 + [Position.LW] * 4 + [Position.RW] * 4
        def_positions = [Position.D] * 7
        g_positions = [Position.G] * 3
        roster_size = len(fwd_positions) + len(def_positions) + len(g_positions)

        ovr_tiers = (
            [(0.58, 0.74)] * 7
            + [(0.66, 0.80)] * 6
            + [(0.72, 0.86)] * 5
            + [(0.78, 0.90)] * 3
            + [(0.84, 0.93)] * 2
        )

        GEN_P, ELITE_P, STAR_P = 0.003, 0.018, 0.045
        used_names: Set[str] = set()

        for team_idx, team in enumerate(self.league.teams):
            if not hasattr(team, "roster") or team.roster is None:
                team.roster = []
            if not hasattr(team, "scratches") or team.scratches is None:
                team.scratches = []
            if not hasattr(team, "injured_reserve") or team.injured_reserve is None:
                team.injured_reserve = []

            team.roster.clear()
            team.scratches.clear()

            team_id = _safe_team_id_for_player_creation(team, team_idx)

            tier_cycle = [ovr_tiers[(team_idx * 7 + i) % len(ovr_tiers)] for i in range(roster_size)]
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
                        target_ovr = min(target_ovr, 0.84)

                    if roll >= GEN_P + ELITE_P:
                        target_ovr = min(target_ovr, 0.89)

                    age_lo, age_hi = age_order[slot_idx]
                    age = rng.randint(age_lo, age_hi)
                    birth_year = season_year - age

                    ratings = build_role_shaped_ratings(
                        position=pos,
                        target_ovr=target_ovr,
                        rng=rng,
                    )
                    gen_profile = pop_generation_profile(ratings)
                    synced_arch = archetype_from_generation_profile(gen_profile, pos)
                    if not synced_arch:
                        synced_arch = assign_skater_archetype(pos, rng)

                    seed = rng.randint(1, 2_000_000_000)

                    ident = generate_human_identity(rng)
                    for _ in range(8):
                        nm = str(getattr(ident, "full_name", "Unknown"))
                        if nm not in used_names:
                            used_names.add(nm)
                            break
                        ident = generate_human_identity(rng)

                    hometown = str(getattr(ident, "hometown", "") or "Unknown")
                    birth_city = hometown.split(",")[0].strip() if hometown else "Unknown"

                    from app.sim_engine.generation.prospect_body import (
                        generate_position_height_cm,
                        generate_realistic_weight_kg,
                    )

                    arch_str = str(getattr(synced_arch, "value", synced_arch) or "")
                    h_cm = generate_position_height_cm(rng, pos, archetype=arch_str)
                    w_kg = generate_realistic_weight_kg(h_cm, pos, archetype=arch_str, age=age)

                    identity = IdentityBio(
                        name=str(getattr(ident, "full_name", "Unknown")),
                        age=age,
                        birth_year=birth_year,
                        birth_country=str(getattr(ident, "nationality", "Canada")),
                        birth_city=birth_city or "Unknown",
                        height_cm=h_cm,
                        weight_kg=w_kg,
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
                        archetype=synced_arch,
                        pool_context="nhl",
                    )
                    if gen_profile:
                        try:
                            setattr(player, "_generated_profile", gen_profile)
                        except Exception:
                            pass

                    finalize_created_player_for_game_ledger(
                        player,
                        league=self.league,
                        team=team,
                        rng=rng,
                        source="init_roster",
                        season_year=season_year,
                    )

                    team.roster.append(player)
                    self.league.players.append(player)

                    slot_idx += 1

            if hasattr(team, "state") and hasattr(team.state, "competitive_score"):
                roster = getattr(team, "roster", []) or []
                if roster:
                    ovrs = sorted(
                        (
                            p.ovr()
                            for p in roster
                            if callable(getattr(p, "ovr", None))
                        ),
                        reverse=True,
                    )[:12]
                    team.state.competitive_score = sum(ovrs) / len(ovrs) if ovrs else 0.5

        try:
            from app.sim_engine.entities.player import enforce_league_ovr_distribution_from_league

            enforce_league_ovr_distribution_from_league(self.league, rng=rng)
        except Exception:
            pass

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
        # Real human identity for every world-pool player: these players can reach
        # the NHL, so they must never carry "Global JUN/EUR ..." placeholder text.
        try:
            human_ident = generate_human_identity(rng, nationality=country)
            human_name = str(human_ident.full_name)
            city = str((human_ident.hometown or "").split(",")[0].strip() or "Unknown")
        except Exception:
            human_ident = None
            human_name = ""
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
        # Height/weight must be correlated (a 6'5" prospect can't be 150 lb). Reuse the
        # shared position/age-aware body generator instead of two independent rolls.
        from app.sim_engine.generation.prospect_body import (
            generate_position_height_cm,
            generate_realistic_weight_kg,
        )

        gen_height_cm = generate_position_height_cm(rng, position)
        gen_weight_kg = generate_realistic_weight_kg(gen_height_cm, position, age=int(age))
        pr = Prospect.create_random(
            name=human_name or f"Prospect {year}-{seed_val % 100000}",
            birth_year=birth_year,
            birth_country=country,
            birth_city=city,
            position=position,
            shoots=ProspectShoots.L if rng.random() < 0.58 else ProspectShoots.R,
            height_cm=gen_height_cm,
            weight_kg=gen_weight_kg,
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
            if getattr(p, "team_id", None) is not None:
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
            if getattr(p, "team_id", None) is not None:
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
            if getattr(p, "team_id", None) is not None:
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
            tid = _id_str(team, "team_id", "id")
            if tid == "":
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
            if getattr(p, "team_id", None) is not None:
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
                name = _hydrated_identity_name(
                    identity_dict.get("name", "Rookie"), identity_dict.get("birth_country"), rng
                )
                birth_year = int(identity_dict.get("birth_year", year - 18))
                age = year - birth_year
                birth_country = str(identity_dict.get("birth_country", "Canada"))
                birth_city = str(identity_dict.get("birth_city", "Unknown"))
                height_cm = sanitize_height_cm(identity_dict.get("height_cm", 180), rng, position)
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
                try:
                    from app.sim_engine.generation.player_headshots import ensure_player_headshot

                    ensure_player_headshot(player)
                except Exception:
                    pass
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

    def _promote_existing_player_from_pool(
        self,
        team: Any,
        player: Any,
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
            roster = getattr(team, "roster", None)
            if roster is None:
                team.roster = []
                roster = team.roster
            prune_ctx = getattr(self.league, "_age_balance_prune", None) or {}
            _make_room_before_promotion(
                self.league,
                team,
                player,
                year,
                roster,
                prune_ctx_pct_u24=float(prune_ctx.get("pct_u24", 25.0)),
                player_roster_age_fn=self._player_roster_age,
            )
            roster.append(player)
            if hasattr(self.league, "players") and self.league.players is not None and player not in (self.league.players or []):
                self.league.players.append(player)
            pool = getattr(team, "prospect_pool", None) or []
            if player in pool:
                pool.remove(player)
            _invoke_nhl_promotion_contract_hook(self.league, player, team, year)
            promoted_ages.append(age)
            promoted_potentials.append(potential)
            name = str(getattr(getattr(player, "identity", None), "name", getattr(player, "name", "Prospect")) or "Prospect")
            try:
                self._pipeline_log_buffer.append(f"PROMOTION EVENT: Pool player promoted to NHL: {name}")
            except Exception:
                pass
            return True
        except Exception:
            return False

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
        if not callable(getattr(prospect, "convert_to_player_payload", None)):
            return self._promote_existing_player_from_pool(
                team,
                prospect,
                year,
                rng,
                promoted_ages=promoted_ages,
                promoted_potentials=promoted_potentials,
                age=age,
                potential=potential,
            )
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
            name = _hydrated_identity_name(
                identity_dict.get("name", "Rookie"), identity_dict.get("birth_country"), rng
            )
            birth_year = int(identity_dict.get("birth_year", year - 18))
            birth_country = str(identity_dict.get("birth_country", "Canada"))
            birth_city = str(identity_dict.get("birth_city", "Unknown"))
            pos_val = identity_dict.get("position", "C")
            if hasattr(pos_val, "value"):
                pos_val = pos_val.value
            pos_val = str(pos_val) if pos_val else "C"
            position = Position(pos_val) if pos_val in ("C", "LW", "RW", "D", "G") else Position.C
            height_cm = sanitize_height_cm(identity_dict.get("height_cm", 180), rng, position)
            weight_kg = int(identity_dict.get("weight_kg", 85))
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
            finalize_created_player_for_game_ledger(
    player,
    league=self.league,
    team=team,
    rng=rng,
    source="prospect_promotion",
    season_year=int(year),
)
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
            prune_ctx = getattr(self.league, "_age_balance_prune", None) or {}
            _make_room_before_promotion(
                self.league,
                team,
                player,
                year,
                roster,
                prune_ctx_pct_u24=float(prune_ctx.get("pct_u24", 25.0)),
                player_roster_age_fn=self._player_roster_age,
            )
            roster.append(player)
            if hasattr(self.league, "players") and self.league.players is not None:
                self.league.players.append(player)
            pool = getattr(team, "prospect_pool", None) or []
            if prospect in pool:
                pool.remove(prospect)
            _invoke_nhl_promotion_contract_hook(self.league, player, team, year)
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
            tid = _id_str(team, "team_id", "id")
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
                tid = _id_str(team, "team_id", "id")
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
        pid = _id_str(p, "id") or _id_str(getattr(p, "identity", None), "id")
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

        season_year = 2025 + int(getattr(self, "year", 0) or 0)
        cap_row = advance_league_salary_cap(self.league, self.rng, season_year=season_year)

        result = self.league.advance_season(
            team_snapshots=team_snapshots,
            team_count=int(getattr(self.league, "max_teams", 32) or 32),
        )

        self.last_league_context = result.get("league_context") or {}
        self.last_league_forecast = result.get("forecast") or {}
        self.last_league_shocks = result.get("shocks") or []
        try:
            econ_ctx = self.last_league_context.setdefault("economics", {})
            econ_ctx["salary_cap"] = float(getattr(self.league, "salary_cap_m", cap_row.get("upperLimit", 92.0)))
            econ_ctx["cap_floor"] = float(getattr(self.league, "cap_floor_m", cap_row.get("lowerLimit", 68.0)))
            econ_ctx["cap_growth_rate"] = float(getattr(self.league, "cap_growth_rate", 0.05))
        except Exception:
            pass

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
            drafted_by_team_id=_id_str(self.player, "drafted_by_team_id", default=_id_str(self.team, "id")),
            developed_by_team_id=_id_str(self.player, "developed_by_team_id", default=_id_str(self.team, "id")),
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
            pid = _id_str(p, "id") or _id_str(getattr(p, "identity", None), "id")
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
        # 2.5 Player season stats — game-derived ledger only (no distribution injection)
        # --------------------------------------------------
        if not hasattr(self.player, "season_stats") or self.player.season_stats is None:
            self.player.season_stats = {}
        season_stats: Dict[str, Any] = {}
        season_tag = int(
            getattr(self, "_last_league_sim_calendar_year", 0) or (2025 + int(getattr(self, "year", 0) or 0))
        )
        ledger = getattr(self, "_last_league_season_stat_ledger", None) or {}
        pid = _id_str(self.player, "id")
        row = ledger.get(pid) if pid else None
        if row:
            season_stats = self._season_stat_line_from_ledger_row(dict(row), season_tag)
            self.player.season_stats[int(season_tag)] = season_stats
            if _stats_pipeline_debug():
                rname = str(
                    row.get("name")
                    or getattr(getattr(self.player, "identity", None), "name", None)
                    or getattr(self.player, "name", "?")
                )
                if str(row.get("position", "")).upper() == "G":
                    print(
                        f"[PLAYER SEASON SYNC] {rname} {int(season_tag)} ga={season_stats.get('ga')} gp={season_stats.get('gp')}"
                    )
                else:
                    print(
                        f"[PLAYER SEASON SYNC] {rname} {int(season_tag)} g={season_stats.get('g')} a={season_stats.get('a')} pts={season_stats.get('pts')} gp={season_stats.get('gp')}"
                    )
        else:
            perf_fb = clamp(30.0 + 58.0 * float(win_pct), 22.0, 86.0)
            season_stats = {
                "season": season_tag,
                "role": "skater",
                "goals": 0,
                "assists": 0,
                "points": 0,
                "g": 0,
                "a": 0,
                "gp": 0,
                "sog": 0,
                "toi_sec": 0,
                "pim": 0,
                "war": 0.0,
                "xgf_pct": 0.5,
                "performance_score": round(perf_fb, 2),
                "expected_score": 60.0,
                "delta": round(perf_fb - 60.0, 2),
            }
            self.player.season_stats[int(season_tag)] = season_stats


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


            scratched_recently=1.0 if bool(getattr(self.player, "_recently_scratched", False)) else 0.0,
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
            # Explicit None checks: team_id=0 is a valid id (e.g. Boston) and must not
            # fall through to the synthetic T## fallback.
            tid = getattr(t, "team_id", None)
            if tid is None:
                tid = getattr(t, "id", None)
            if tid is None:
                tid = f"T{idx:02d}"
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
        vol_bonus = 0.0
        roster_n = 0
        for p in roster:
            traits = getattr(p, "traits", None)
            if traits is None:
                continue
            try:
                vol_bonus += float(getattr(traits, "volatility", 0.5) or 0.5)
                roster_n += 1
            except (TypeError, ValueError):
                continue
        avg_vol = (vol_bonus / roster_n) if roster_n else 0.5
        mult = 1.0 + 0.72 * av - 0.28 * ac + 0.38 * (avg_vol - 0.5)
        return max(0.78, min(1.55, mult))

    def _era_combined_gpg_cap(self) -> float:
        """Clamp stacked era + baseline scoring so normal seasons stay franchise-fun, not arcade."""
        era = self._active_era_str().lower()
        era_m = float(getattr(self.league, "_era_scoring_multiplier", 1.0) or 1.0)
        if "run_and_gun" in era or "chaos" in era:
            return 6.9
        if "power_play" in era or era_m >= 1.10:
            return 6.6
        # Modern NHL combined GPG ~6.0–6.2 (≈3.0–3.1 per team).
        return 6.35

    def _era_scoring_mu_factor(self) -> float:
        """Dampened era lift so new franchise baseline + era modifiers do not stack into 8+ GPG."""
        era_m = float(getattr(self.league, "_era_scoring_multiplier", 1.0) or 1.0)
        return float(1.0 + (era_m - 1.0) * 0.55)

    def _gm_player_scoring_tier(self, p: Any) -> str:
        """Elite concentration tiers for goal/assist distribution (not every star is GOAT-tier)."""
        explicit = getattr(p, "_scoring_tier", None)
        if explicit:
            return str(explicit).lower()
        try:
            ovr = float(self._gm_ovr_0_100(p))
        except Exception:
            ovr = 70.0
        try:
            ovr = float(self._gm_ovr_0_100(p))
        except Exception:
            ovr = 70.0
        raw_pot = getattr(p, "potential", None)
        if isinstance(raw_pot, str):
            ps = raw_pot.lower().strip()
            if ps in ("goat", "generational") and ovr >= 88.0:
                return "goat" if ovr >= 91.0 else "generational"
            if ps in ("franchise", "elite", "superstar") and ovr >= 86.0:
                return "franchise"
        if bool(getattr(p, "_pipeline_franchise_flag", False)):
            if ovr >= 93.0:
                return "goat"
            if ovr >= 88.0:
                return "generational"
            if ovr >= 85.0:
                return "franchise"
        if ovr >= 94.0:
            return "goat"
        if ovr >= 90.0:
            return "generational"
        if ovr >= 87.0:
            return "franchise"
        if ovr >= 84.0:
            return "elite"
        return "normal"

    def _gm_scoring_involvement_mult(self, p: Any) -> float:
        """Star concentration: elite forwards pop; defensive D / depth stay grounded."""
        tier = self._gm_player_scoring_tier(p)
        tier_mult = {
            "elite": 1.10,
            "franchise": 1.18,
            "generational": 1.28,
            "goat": 1.14,
        }.get(tier, 1.0)
        hist = str(getattr(p, "_historic_scoring_season", "") or "").lower()
        if hist == "historic":
            tier_mult *= 1.24
        elif hist == "goat":
            tier_mult *= 1.44
        elif hist == "absurd":
            tier_mult *= 1.55
        li = getattr(p, "_gm_game_line_idx", None)
        rank = int(getattr(p, "_gm_game_line_rank", 0) or 0)
        fwd_rank = getattr(p, "_gm_game_fwd_rank", None)
        elite_ids = set(getattr(self.league, "_scoring_elite_player_ids", None) or ())
        pid = _id_str(p, "id")
        in_elite_pool = bool(pid and pid in elite_ids)
        if not hist:
            if not in_elite_pool:
                tier_mult = 1.0 + (tier_mult - 1.0) * 0.16
            if li == 0 and rank == 0:
                pass
            elif int(li or 99) == 0:
                tier_mult = 1.0 + (tier_mult - 1.0) * 0.52
            elif int(li or 99) == 1:
                tier_mult = 1.0 + (tier_mult - 1.0) * 0.36
            else:
                tier_mult = 1.0 + (tier_mult - 1.0) * 0.20
            if fwd_rank is not None:
                fr = int(fwd_rank)
                if fr == 0:
                    pass
                elif fr == 1:
                    tier_mult = 1.0 + (tier_mult - 1.0) * 0.42
                elif fr == 2:
                    tier_mult = 1.0 + (tier_mult - 1.0) * 0.22
                else:
                    tier_mult = 1.0 + (tier_mult - 1.0) * 0.10
            if in_elite_pool and fwd_rank is not None and int(fwd_rank) > 0:
                tier_mult = 1.0 + (tier_mult - 1.0) * 0.48
        pos = self._gm_pos_str(p).upper()
        if pos == "D":
            off, df = player_offense_defense_proxy(p)
            if df > off * 1.06:
                tier_mult = 1.0 + (tier_mult - 1.0) * 0.32
        usage = self._gm_role_usage_mult(p)
        if usage < 0.95:
            tier_mult = 1.0 + (tier_mult - 1.0) * 0.42
        elif usage < 1.45:
            tier_mult = 1.0 + (tier_mult - 1.0) * 0.72
        return float(max(0.85, min(1.58, tier_mult * self._gm_franchise_alloc_mult(
            p, "overall_equivalent", "effort", "shot_involvement", "assist_involvement"
        ))))

    def _gm_repeat_goal_damp_for(self, p: Any) -> float:
        """Per-game repeat damping; generational/GOAT historic nights can still explode."""
        damp = float(self._GM_REPEAT_GOAL_DAMP)
        tier = self._gm_player_scoring_tier(p)
        hist = str(getattr(p, "_historic_scoring_season", "") or "").lower()
        if hist in ("goat", "absurd") and tier in ("goat", "generational"):
            damp *= 0.48
        elif hist == "historic" and tier in ("goat", "generational", "franchise"):
            damp *= 0.62
        elif tier in ("goat", "generational"):
            damp *= 0.78
        return float(max(0.18, min(0.55, damp)))

    def _team_goalie_suppression(self, team: Any) -> float:
        """Elite goaltending shaves opponent expected goals (SV% stays honest via shot volume)."""
        goalies = self._gm_goalies(team)
        if not goalies:
            return 0.0
        g = max(goalies, key=lambda x: self._gm_ovr_bonus(x))
        g_skill = self._gm_rating_avg(g, GOALIE_KEYS) / 99.0
        era_g = float(getattr(g, "_tuning_goalie_value", 1.0) or 1.0)
        g_skill = max(0.25, min(0.98, g_skill * (0.94 + 0.06 * era_g)))
        return float(max(0.0, min(0.32, (g_skill - 0.48) * 0.62)))

    def _team_defense_suppression(self, team: Any) -> float:
        """Strong team defense slightly suppresses opponent finishing quality."""
        sk = self._gm_skaters(team)
        defs = [p for p in sk if str(self._gm_pos_str(p)).upper() == "D"]
        if not defs:
            return 0.0
        rated = sorted((self._gm_ovr_bonus(p) for p in defs), reverse=True)[:4]
        avg = sum(rated) / max(1, len(rated))
        return float(max(0.0, min(0.14, (avg - 52.0) / 220.0)))

    def _team_pp_danger(self, team: Any) -> float:
        """0..1 PP threat proxy for special-teams goal share in the stat ledger."""
        off = self._team_offense_skill(team)
        sk = self._gm_skaters(team)
        fw = [p for p in sk if str(self._gm_pos_str(p)).upper() != "D"]
        if not fw:
            return 0.5
        top = sorted(fw, key=lambda x: self._gm_ovr_bonus(x), reverse=True)[:6]
        pp_skill = sum(self._gm_offense_weight(p) for p in top) / max(1, len(top))
        return float(max(0.22, min(0.92, 0.42 * off + 0.38 * min(1.0, pp_skill / 2.4) + 0.20 * off)))

    def _refresh_league_scoring_elite_set(self, teams: List[Any]) -> None:
        """League-wide top offensive talents — only this pool gets full star scoring gravity."""
        rated: List[Tuple[float, str]] = []
        for tm in teams:
            for p in self._gm_skaters(tm):
                if str(self._gm_pos_str(p)).upper() == "D":
                    continue
                pid = _id_str(p, "id")
                if pid:
                    rated.append((self._gm_ovr_bonus(p), pid))
        rated.sort(key=lambda z: z[0], reverse=True)
        elite_n = max(12, min(18, int(round(len(teams) * 0.44))))
        elite_ids = {pid for _, pid in rated[:elite_n]}
        setattr(self.league, "_scoring_elite_player_ids", elite_ids)

    def _roll_historic_scoring_seasons(self, teams: List[Any], rng: random.Random, year: int) -> None:
        """Rare GOAT/historic seasons — requires talent, usage, health, and team context."""
        for tm in teams:
            off_ctx = self._team_offense_skill(tm)
            for p in self._gm_skaters(tm):
                setattr(p, "_historic_scoring_season", None)
                if self._injury_sidelined(p):
                    continue
                tier = self._gm_player_scoring_tier(p)
                if tier == "normal":
                    continue
                try:
                    ovr = float(career_ovr_0_100(p))
                except Exception:
                    ovr = 70.0
                usage = self._gm_role_usage_mult(p)
                if tier == "elite" and (ovr < 84.0 or usage < 1.45):
                    continue
                psych = getattr(p, "psych", None)
                conf = float(getattr(psych, "confidence_level", 0.5) or 0.5) if psych else 0.5
                base_p = {
                    "elite": 0.004,
                    "franchise": 0.018,
                    "generational": 0.048,
                    "goat": 0.13,
                }.get(tier, 0.0)
                base_p *= (0.72 + 0.36 * off_ctx) * (0.80 + 0.32 * conf)
                if usage >= 1.95:
                    base_p *= 1.18
                if rng.random() >= base_p:
                    continue
                if tier == "goat" and rng.random() < 0.07:
                    setattr(p, "_historic_scoring_season", "absurd")
                elif tier in ("goat", "generational") and rng.random() < 0.34:
                    setattr(p, "_historic_scoring_season", "goat")
                else:
                    setattr(p, "_historic_scoring_season", "historic")
                setattr(p, "_historic_scoring_season_year", int(year))

    def _team_offense_skill(self, team: Any, skaters_subset: Optional[List[Any]] = None) -> float:
        """
        Approx 0..1 offense+awareness signal from active skaters.
        Uses weighted top-end roster talent so top-line clubs create more offense.
        Optional skaters_subset limits the pool (e.g. injury-eligible players only).
        """
        if skaters_subset is not None:
            roster = [p for p in skaters_subset if not getattr(p, "retired", False)]
        else:
            roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
        skaters = []
        for p in roster:
            pos = str(getattr(getattr(p, "identity", None), "position", "") or "").upper()
            if pos == "G":
                continue
            skaters.append(p)
        if not skaters:
            return 0.5

        rated: List[Tuple[float, Any]] = []
        for p in skaters:
            fn = getattr(p, "ovr", None)
            ovr = float(fn() if callable(fn) else fn or 0.5)
            if ovr <= 1.5:
                ovr *= 99.0
            rated.append((ovr, p))
        rated.sort(key=lambda z: z[0], reverse=True)
        top = [z[1] for z in rated[:12]]

        vals: List[float] = []
        for p in top:
            shoot = self._gm_rating_avg(p, OFFENSE_KEYS)
            aware = self._gm_rating_avg(p, IQ_KEYS)
            passq = self._gm_rating_avg(p, PASSING_KEYS)
            vals.append(0.44 * shoot + 0.32 * aware + 0.24 * passq)

        avg = sum(vals) / max(1, len(vals))
        return max(0.25, min(0.95, avg / 99.0))

    def _team_superstar_offense_impact(self, team: Any) -> float:
        """
        0..~1 nonlinear star-force proxy.
        84 OVR is baseline; 95+ bends team scoring environment.
        """
        sk = self._gm_skaters(team)
        fw = [p for p in sk if str(self._gm_pos_str(p)).upper() != "D"]
        if not fw:
            return 0.0
        top = sorted(fw, key=lambda x: self._gm_ovr_bonus(x), reverse=True)[:4]
        impact = 0.0
        for p in top:
            try:
                ovr = float(career_ovr_0_100(p))
            except Exception:
                ovr = 74.0
            over = max(0.0, ovr - 84.0)
            if over <= 0.0:
                continue
            role_u = self._gm_role_usage_mult(p)
            usage = max(0.72, min(2.35, role_u))
            # Nonlinear curve: elite and generational players separate sharply.
            player_impact = ((over / 10.0) ** 1.78) * (0.72 + 0.28 * min(1.0, usage / 2.0))
            hist = str(getattr(p, "_historic_scoring_season", "") or "").lower()
            if hist in ("historic", "goat", "absurd"):
                player_impact *= 1.08
            impact += player_impact
        return float(max(0.0, min(1.05, impact)))

    # ------------------------------------------------------------------
    # Unified game stat ledger (single source of truth with franchise UI)
    # ------------------------------------------------------------------

    def _gm_active_roster(self, team: Any) -> List[Any]:
        """NHL roster players eligible to dress — one entry per player id."""
        seen: Set[str] = set()
        out: List[Any] = []
        for p in list(getattr(team, "roster", None) or []):
            if getattr(p, "retired", False):
                continue
            pid = _id_str(p, "id")
            if pid and pid in seen:
                continue
            if pid:
                seen.add(pid)
            out.append(p)
        return out

    def _gm_player_on_credited_team(self, p: Any, team_id: str) -> bool:
        """True when the player currently sits on the NHL roster for team_id.

        Blocks dual-roster ghosts: a traded identity left on a former club must
        not keep accruing GP/points for that club after the move.
        """
        pid = _id_str(p, "id")
        tid = str(team_id or "")
        if not pid or not tid:
            return True
        teams = list(getattr(self.league, "teams", None) or [])
        if not teams:
            return True
        found_any = False
        for tm in teams:
            tm_id = str(getattr(tm, "team_id", None) or getattr(tm, "id", "") or "")
            if tm_id != tid:
                continue
            found_any = True
            for cand in list(getattr(tm, "roster", None) or []):
                if _id_str(cand, "id") == pid:
                    return True
            return False
        return True if not found_any else False

    def _gm_pos_str(self, p: Any) -> str:
        ident = getattr(p, "identity", None)
        pos = getattr(ident, "position", None) if ident else None
        if pos is None or str(getattr(pos, "value", pos) or "").strip() in ("", "?"):
            pos = getattr(p, "position", None)
        return str(getattr(pos, "value", pos) or "?")

    def _gm_skaters(self, team: Any) -> List[Any]:
        return [p for p in self._gm_active_roster(team) if str(self._gm_pos_str(p)).upper() != "G"]

    def _gm_goalies(self, team: Any) -> List[Any]:
        all_goalies = [p for p in self._gm_active_roster(team) if str(self._gm_pos_str(p)).upper() == "G"]
        healthy = [p for p in all_goalies if not self._injury_sidelined(p)]
        # Hard runtime truth: prefer available goalies; only fall back if none are healthy.
        return healthy or all_goalies

    def set_franchise_game_stat_modifiers(
        self,
        *,
        home_player_modifiers: Optional[Dict[str, Dict[str, Any]]] = None,
        away_player_modifiers: Optional[Dict[str, Dict[str, Any]]] = None,
        home_win_probability_delta: float = 0.0,
    ) -> None:
        """Attach storyline / universe per-player stat fingerprints for one game."""
        self._franchise_home_player_mods = {
            str(pid): {str(k): float(v) for k, v in (row or {}).items()}
            for pid, row in (home_player_modifiers or {}).items()
            if pid
        }
        self._franchise_away_player_mods = {
            str(pid): {str(k): float(v) for k, v in (row or {}).items()}
            for pid, row in (away_player_modifiers or {}).items()
            if pid
        }
        self._franchise_home_win_prob_delta = float(home_win_probability_delta or 0.0)

    def clear_franchise_game_stat_modifiers(self) -> None:
        self._franchise_home_player_mods = {}
        self._franchise_away_player_mods = {}
        self._franchise_home_win_prob_delta = 0.0

    def _gm_franchise_player_modifiers(self, p: Any) -> Dict[str, float]:
        pid = _id_str(p, "id")
        if not pid:
            return {}
        home_mods = getattr(self, "_franchise_home_player_mods", None) or {}
        away_mods = getattr(self, "_franchise_away_player_mods", None) or {}
        row = home_mods.get(pid) or away_mods.get(pid) or {}
        return {str(k): float(v) for k, v in row.items()}

    def _gm_franchise_mod_sum(self, p: Any, *keys: str) -> float:
        mods = self._gm_franchise_player_modifiers(p)
        return sum(float(mods.get(k, 0) or 0) for k in keys)

    def _gm_franchise_alloc_mult(self, p: Any, *keys: str, lo: float = 0.55, hi: float = 1.48) -> float:
        delta = self._gm_franchise_mod_sum(p, *keys)
        return float(max(lo, min(hi, 1.0 + delta)))

    def _gm_ovr_0_100(self, p: Any) -> float:
        """Canonical 0-100 overall for usage / scoring allocation (matches UI OVR)."""
        cache = getattr(p, "_gm_runtime_cache", None)
        if isinstance(cache, dict) and "ovr_0_100" in cache:
            return float(cache["ovr_0_100"])
        o = None
        try:
            from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
                get_effective_ovr_display,
            )

            o = float(get_effective_ovr_display(p))
        except Exception:
            o = None
        if o is None or o <= 0:
            try:
                o = float(career_ovr_0_100(p))
            except Exception:
                try:
                    fn = getattr(p, "ovr", None)
                    o = float(fn() if callable(fn) else fn)
                    if o <= 1.5:
                        o *= 99.0
                except Exception:
                    o = 68.0
        o = max(30.0, min(99.0, float(o)))
        if isinstance(cache, dict):
            cache["ovr_0_100"] = o
        return o

    def _gm_ovr_norm(self, p: Any) -> float:
        return self._gm_ovr_0_100(p) / 99.0

    def _gm_ovr_bonus(self, p: Any) -> float:
        cache = getattr(p, "_gm_runtime_cache", None)
        if isinstance(cache, dict) and "ovr_bonus" in cache:
            return float(cache["ovr_bonus"])
        # Steeper star curve so 86+ pull away from mid/low OVR depth.
        val = max(14.0, self._gm_ovr_0_100(p)) ** 1.55
        if isinstance(cache, dict):
            cache["ovr_bonus"] = val
        return val

    def _gm_player_runtime_cache(self, p: Any) -> Dict[str, float]:
        cache = getattr(p, "_gm_runtime_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(p, "_gm_runtime_cache", cache)
        return cache

    def _gm_prime_team_game_caches(self, team: Any) -> None:
        """Precompute per-game team weights so hot loops never re-sort full rosters."""
        # Always refresh — line/usage context changes every game.
        setattr(team, "_gm_team_game_cache_primed", False)
        sk = self._gm_skaters(team)
        hub_mult: Dict[int, float] = {}
        impact = float(self._team_superstar_offense_impact(team))
        if impact <= 0.04 or not sk:
            for p in sk:
                hub_mult[id(p)] = 1.0
        else:
            ranked = sorted(sk, key=self._gm_offensive_skill_composite, reverse=True)
            for p in sk:
                hub_mult[id(p)] = 1.0
            if ranked:
                hub_mult[id(ranked[0])] = 1.0 + impact * 0.92
                for p in ranked[1:3]:
                    hub_mult[id(p)] = 1.0 + impact * 0.36
        setattr(team, "_gm_hub_mult_by_player", hub_mult)
        setattr(team, "_gm_cached_star_impact", impact)
        setattr(team, "_gm_team_game_cache_primed", True)

    def _gm_prime_game_player_caches(
        self,
        home: Any,
        away: Any,
        home_dressed: Sequence[Any],
        away_dressed: Sequence[Any],
    ) -> None:
        """Warm per-player event weights once per game (ratings/OVR are stable in-game)."""
        for p in list(home_dressed) + list(away_dressed):
            cache = self._gm_player_runtime_cache(p)
            # Drop game-sensitive entries so line/TOI changes recompute correctly.
            for key in (
                "offensive_skill",
                "shot_quality",
                "finishing_adj",
                "block_weight",
                "ovr_bonus",
                "ovr_0_100",
            ):
                cache.pop(key, None)
            setattr(p, "_gm_game_cache_primed", False)

        setattr(home, "_gm_team_game_cache_primed", False)
        setattr(away, "_gm_team_game_cache_primed", False)
        self._gm_prime_team_game_caches(home)
        self._gm_prime_team_game_caches(away)
        for p in list(home_dressed) + list(away_dressed):
            cache = self._gm_player_runtime_cache(p)
            cache["ovr_0_100"] = self._gm_ovr_0_100(p)
            cache["ovr_bonus"] = self._gm_ovr_bonus(p)
            cache["offensive_skill"] = self._gm_offensive_skill_composite(p)
            cache["block_weight"] = self._gm_block_weight(p)
            cache["shot_quality"] = self._gm_shot_quality_weight(p)
            cache["finishing_adj"] = self._gm_finishing_adjustment(p)
            setattr(p, "_gm_game_cache_primed", True)

    def _gm_role_usage_mult(self, p: Any) -> float:
        pos = self._gm_pos_str(p).upper()
        pair_idx = getattr(p, "_gm_game_pair_idx", None)
        # Defensemen dress by pair — use that for usage so PP1 / top-pair D
        # actually drive offense instead of falling through to flat OVR defaults.
        if pos == "D" and pair_idx is not None:
            pair_usage_map = (2.02, 1.28, 0.78)
            line_usage = pair_usage_map[min(2, max(0, int(pair_idx)))]
            rank = int(getattr(p, "_gm_game_pair_rank", 0) or 0)
            line_usage *= (1.0, 0.92)[min(1, max(0, rank))]
        else:
            line_usage_map = (2.06, 1.38, 0.96, 0.72)
            li = getattr(p, "_gm_game_line_idx", None)
            if li is not None:
                line_usage = line_usage_map[min(3, max(0, int(li)))]
                rank = int(getattr(p, "_gm_game_line_rank", 0) or 0)
                if int(li) <= 1:
                    line_usage *= (1.0, 0.72, 0.58, 0.48)[min(3, max(0, rank))]
            else:
                # No deployed line — approximate from overall so stars are not flattened
                # to depth usage when saved lines are missing.
                ovr_n = self._gm_ovr_norm(p)
                if ovr_n >= 0.90:
                    line_usage = 1.95
                elif ovr_n >= 0.86:
                    line_usage = 1.55
                elif ovr_n >= 0.80:
                    line_usage = 1.20
                elif ovr_n >= 0.74:
                    line_usage = 0.95
                else:
                    line_usage = 0.72
        role_raw = str(
            getattr(p, "line_role", None)
            or getattr(p, "role", None)
            or getattr(p, "depth_role", None)
            or ""
        ).lower()
        role_usage = line_usage
        if "top" in role_raw or "line1" in role_raw or "first" in role_raw or "pp1" in role_raw:
            role_usage = 2.38 if pos != "D" else 2.15
        elif "second" in role_raw or "line2" in role_raw or "pair2" in role_raw:
            role_usage = 1.56 if pos != "D" else 1.32
        elif "third" in role_raw or "line3" in role_raw or "middle" in role_raw:
            role_usage = 1.00
        elif "fourth" in role_raw or "line4" in role_raw or "depth" in role_raw:
            role_usage = 0.70
        return float(max(line_usage, role_usage))

    def _gm_rating_avg(self, p: Any, keys: List[str], default: float = None) -> float:
        if default is None:
            default = float(DEFAULT_NHL_RATING)
        vals = [self._gm_rating_lookup(p, k, default=default) for k in keys]
        return sum(vals) / max(1, len(vals))

    def _gm_player_type_str(self, p: Any) -> str:
        pt = getattr(p, "player_type", None) or getattr(p, "archetype", None) or ""
        return str(getattr(pt, "value", pt) or "").lower()

    def _gm_rating_lookup(self, p: Any, *keys: str, default: float = None) -> float:
        """Read a rating with legacy→prefixed alias support. Default matches entity floor."""
        if default is None:
            default = float(DEFAULT_NHL_RATING)
        r = getattr(p, "ratings", None) or {}
        for key in keys:
            if key is None:
                continue
            if key in r:
                try:
                    return float(r.get(key) or default)
                except (TypeError, ValueError):
                    continue
            alias = ALIASES.get(str(key))
            if alias and alias in r:
                try:
                    return float(r.get(alias) or default)
                except (TypeError, ValueError):
                    continue
        return float(default)

    def _gm_readiness_usage_mult(self, p: Any) -> float:
        """Age/readiness gates opportunity — not event ability."""
        try:
            age = int(career_player_age(p) or 0)
        except Exception:
            age = 0
        pos = self._gm_pos_str(p).upper()
        if pos == "G":
            ability = self._gm_rating_avg(p, GOALIE_KEYS) / 99.0
        elif pos == "D":
            ability = self._gm_rating_avg(p, DEFENSE_KEYS) / 99.0
        else:
            ability = self._gm_rating_avg(p, OFFENSE_KEYS) / 99.0
        if age <= 0:
            return 1.0
        if age <= 19:
            return 0.82 if ability < 0.82 else 0.95
        if age == 20:
            return 0.90 if ability < 0.80 else 0.96
        if age == 21:
            return 0.96
        return 1.0

    def _gm_production_balance_score(self, p: Any) -> float:
        """Harmonic mean of shooting and passing — rewards balanced two-way producers."""
        shoot = self._gm_rating_avg(p, OFFENSE_KEYS) / 99.0
        passing = self._gm_rating_avg(p, PASSING_KEYS) / 99.0
        denom = max(0.08, shoot + passing)
        return max(0.04, (2.0 * shoot * passing) / denom)

    def _gm_is_certified_sniper(self, p: Any) -> bool:
        pt = self._gm_player_type_str(p)
        if "sniper" not in pt and "finisher" not in pt:
            return False
        shoot = self._gm_rating_avg(p, OFFENSE_KEYS) / 99.0
        passing = self._gm_rating_avg(p, PASSING_KEYS) / 99.0
        return shoot - passing >= 0.14

    def _gm_is_certified_playmaker(self, p: Any) -> bool:
        pt = self._gm_player_type_str(p)
        if "playmaker" not in pt:
            return False
        shoot = self._gm_rating_avg(p, OFFENSE_KEYS) / 99.0
        passing = self._gm_rating_avg(p, PASSING_KEYS) / 99.0
        return passing - shoot >= 0.12

    def _gm_goal_assist_balance_mult(
        self,
        p: Any,
        ledger: Dict[str, Dict[str, Any]],
        team_id: str,
        *,
        role: str,
    ) -> float:
        """
        Season/game running G-A feedback — pushes typical producers toward balanced splits.
        Only certified sniper/playmaker archetypes may stay extreme.
        """
        pid = str(getattr(p, "id", "") or "")
        row = ledger.get(pid)
        if not row:
            return 1.0
        g = int(row.get("g", 0) or 0)
        a = int(row.get("a", 0) or 0)
        if g + a < 1:
            return 1.0
        diff = g - a
        certified_sniper = self._gm_is_certified_sniper(p)
        certified_playmaker = self._gm_is_certified_playmaker(p)

        if role == "score":
            if certified_sniper:
                if diff >= 26:
                    return 0.92
                if diff >= 22:
                    return 0.96
                return 1.08
            if certified_playmaker:
                return 1.08 if diff <= 0 else 0.94
            # Soft per-game / season nudge only — never crush star finishers down
            # to depth rates (old curve went to 0.30× at +20 G-A).
            if diff <= -4:
                return 1.10
            if diff <= 0:
                return 1.05
            if diff <= 6:
                return 0.97
            if diff <= 12:
                return 0.92
            if diff <= 18:
                return 0.88
            return 0.85

        if role == "assist":
            if certified_playmaker:
                return 1.06 if diff <= 8 else 1.0
            if certified_sniper and diff >= 10:
                return 0.90
            if diff >= 14:
                return 1.12
            if diff >= 8:
                return 1.06
            if diff <= -6:
                return 0.92
            return 1.0

        return 1.0

    def _gm_scoring_hub_bonus(self, p: Any, team: Any) -> float:
        """Reduced hub involvement for who takes the shot."""
        base = self._gm_offensive_hub_bonus(p, team)
        if base <= 1.0:
            return 1.0
        return 1.0 + (base - 1.0) * 0.55

    def _gm_playmaking_hub_bonus(self, p: Any, team: Any) -> float:
        """Full hub involvement for who creates the chance."""
        return self._gm_offensive_hub_bonus(p, team)

    def _gm_shot_volume_weight(self, p: Any) -> float:
        shoot = self._gm_rating_avg(p, OFFENSE_KEYS)
        aware = self._gm_rating_avg(p, IQ_KEYS)
        ovr_n = self._gm_ovr_norm(p)
        usage = self._gm_role_usage_mult(p) * self._gm_readiness_usage_mult(p)
        pt = self._gm_player_type_str(p)
        type_mult = 1.08 if "sniper" in pt or "shooter" in pt else (0.94 if "playmaker" in pt else 1.0)
        pos = self._gm_pos_str(p).upper()
        if pos == "D":
            # Offensive / high-OVR D take more point shots; stay-at-home less so.
            if "offensive" in pt or ovr_n >= 0.88:
                pos_mult = 1.12
            elif ovr_n >= 0.82:
                pos_mult = 1.00
            else:
                pos_mult = 0.88
        else:
            pos_mult = 1.0
        talent = (0.68 * ovr_n + 0.22 * (shoot / 99.0) + 0.10 * (aware / 99.0)) ** 1.55
        if ovr_n < 0.72:
            talent *= 0.62
        elif ovr_n < 0.78:
            talent *= 0.82
        return max(0.03, talent * usage * type_mult * pos_mult * self._gm_franchise_alloc_mult(
            p, "shot_involvement", "effort", "shooting", "overall_equivalent"
        ))

    def _gm_shot_quality_weight(self, p: Any) -> float:
        cache = getattr(p, "_gm_runtime_cache", None)
        if isinstance(cache, dict) and "shot_quality" in cache:
            return float(cache["shot_quality"])
        aware = sum(self._gm_rating_lookup(p, k) for k in IQ_KEYS) / max(1, len(IQ_KEYS))
        puck = self._gm_rating_lookup(p, "puck_control", "pc_puck_control")
        deking = self._gm_rating_lookup(p, "deking", "pc_deking", "agility", "skg_agility")
        speed = self._gm_rating_lookup(p, "speed", "skg_speed", "acceleration", "skg_acceleration")
        strength = self._gm_rating_lookup(p, "strength", "phy_strength")
        pt = self._gm_player_type_str(p)
        net_front = 1.22 if "power" in pt or "net" in pt else 1.0
        val = max(0.04, (0.34 * aware + 0.26 * puck + 0.18 * deking + 0.14 * speed + 0.08 * strength) / 99.0 * net_front)
        val *= self._gm_franchise_alloc_mult(p, "shooting", "shot_accuracy", "offensive_awareness")
        if isinstance(cache, dict):
            cache["shot_quality"] = val
        return val

    def _gm_finishing_adjustment(self, p: Any) -> float:
        cache = getattr(p, "_gm_runtime_cache", None)
        if isinstance(cache, dict) and "finishing_adj" in cache:
            return float(cache["finishing_adj"])
        wrist = self._gm_rating_lookup(p, "wrist_shot_accuracy", "off_wrist_shot_accuracy")
        slap = self._gm_rating_lookup(p, "slap_shot_accuracy", "off_slap_shot_accuracy")
        power = self._gm_rating_lookup(p, "shot_power", "wrist_shot_power", "off_wrist_shot_power")
        aware = sum(self._gm_rating_lookup(p, k) for k in IQ_KEYS) / max(1, len(IQ_KEYS))
        comp = self._gm_rating_lookup(p, "composure", "iqm_composure", default=aware)
        ovr_n = self._gm_ovr_norm(p)
        pt = self._gm_player_type_str(p)
        pos = self._gm_pos_str(p).upper()
        acc = wrist if pos != "D" else 0.55 * wrist + 0.45 * slap
        fin = (0.34 * acc + 0.16 * power + 0.12 * aware + 0.08 * comp) / 99.0 + 0.30 * ovr_n
        if "sniper" in pt or "finisher" in pt:
            fin *= 1.06
        elif "playmaker" in pt:
            fin *= 0.97
        val = max(0.72, min(1.25, 0.80 + 0.38 * fin + max(0.0, ovr_n - 0.84) * 0.42))
        if isinstance(cache, dict):
            cache["finishing_adj"] = val
        return val

    def _gm_offensive_skill_composite(self, p: Any) -> float:
        """Ratings + overall + usage — stars must dominate involvement."""
        cache = getattr(p, "_gm_runtime_cache", None)
        if isinstance(cache, dict) and "offensive_skill" in cache:
            return float(cache["offensive_skill"])
        shoot = self._gm_rating_avg(p, OFFENSE_KEYS) / 99.0
        passing = self._gm_rating_avg(p, PASSING_KEYS) / 99.0
        puck = self._gm_rating_lookup(p, "puck_control", "pc_puck_control") / 99.0
        aware = self._gm_rating_avg(p, IQ_KEYS) / 99.0
        ovr_n = self._gm_ovr_norm(p)
        # Overall is the primary talent signal; ratings refine style within that band.
        base = (
            0.50 * ovr_n
            + 0.16 * shoot
            + 0.14 * passing
            + 0.10 * puck
            + 0.10 * aware
        )
        usage = self._gm_role_usage_mult(p) * self._gm_readiness_usage_mult(p)
        # Superstar curve: 86+ OVR pull away hard from mid-80s / depth.
        star_curve = max(0.0, max(0.0, ovr_n - 0.82) ** 1.15) * 2.85
        depth_penalty = 1.0
        if ovr_n < 0.72:
            depth_penalty = 0.52 + 0.30 * (ovr_n / 0.72)
        elif ovr_n < 0.78:
            depth_penalty = 0.78 + 0.22 * ((ovr_n - 0.72) / 0.06)
        pt = self._gm_player_type_str(p)
        nudge = 1.02 if "playmaker" in pt else (1.01 if "sniper" in pt or "finisher" in pt else 1.0)
        val = max(0.04, (base + star_curve * 0.42) * usage * nudge * depth_penalty)
        val *= self._gm_franchise_alloc_mult(
            p, "overall_equivalent", "effort", "composure", "offensive_awareness", "readiness_ovr_delta"
        )
        if isinstance(cache, dict):
            cache["offensive_skill"] = val
        return val

    def _gm_offensive_hub_bonus(self, p: Any, team: Any) -> float:
        """Stars who bend team offense must also bend their own involvement."""
        hub = getattr(team, "_gm_hub_mult_by_player", None)
        if isinstance(hub, dict):
            return float(hub.get(id(p), 1.0))
        impact = float(self._team_superstar_offense_impact(team))
        if impact <= 0.04:
            return 1.0
        ranked = sorted(self._gm_skaters(team), key=self._gm_offensive_skill_composite, reverse=True)
        if not ranked:
            return 1.0
        if p is ranked[0]:
            return 1.0 + impact * 0.62
        if p in ranked[1:3]:
            return 1.0 + impact * 0.24
        return 1.0

    def _gm_sequence_driver_weight(
        self,
        p: Any,
        team: Any,
        *,
        last_driver: Any = None,
        strength: str = "EV",
    ) -> float:
        w = self._gm_offensive_skill_composite(p) * self._gm_playmaking_hub_bonus(p, team)
        if last_driver is p:
            w *= 1.18
        elif last_driver is not None and last_driver in (getattr(team, "_gm_cached_hubs", None) or []):
            w *= 1.08
        if str(strength or "").upper() == "PP":
            w *= 1.18
        line_idx = int(getattr(p, "_gm_game_line_idx", 3) or 3)
        if line_idx == 0:
            w *= 1.14
        elif line_idx >= 2:
            w *= 0.88
        return max(0.04, w)

    def _gm_primary_assist_weight(self, p: Any) -> float:
        balance = self._gm_production_balance_score(p)
        passing = self._gm_rating_avg(p, PASSING_KEYS) / 99.0
        ovr_n = self._gm_ovr_norm(p)
        base = self._gm_offensive_skill_composite(p)
        combined = 0.44 * (ovr_n ** 1.55) + 0.22 * base + 0.20 * passing + 0.14 * balance
        pt = self._gm_player_type_str(p)
        shoot = self._gm_rating_avg(p, OFFENSE_KEYS) / 99.0
        if "playmaker" in pt and passing - shoot >= 0.10:
            combined *= 1.06
        elif "sniper" in pt and shoot - passing >= 0.10:
            combined *= 0.94
        if self._gm_pos_str(p).upper() == "D":
            # Top-pair / elite D QB primary assists on entries and PP.
            if ovr_n >= 0.90:
                combined *= 1.28
            elif ovr_n >= 0.86:
                combined *= 1.12
            else:
                combined *= 0.92
        if ovr_n < 0.72:
            combined *= 0.62
        elif ovr_n < 0.78:
            combined *= 0.82
        return max(0.05, combined ** 1.12 * self._gm_franchise_alloc_mult(
            p, "assist_involvement", "passing", "offensive_awareness", "puck_control"
        ))

    def _gm_secondary_assist_weight(self, p: Any) -> float:
        balance = self._gm_production_balance_score(p)
        base = self._gm_offensive_skill_composite(p)
        process = self._gm_possession_weight(p)
        ovr_n = self._gm_ovr_norm(p)
        combined = 0.34 * (ovr_n ** 1.25) + 0.22 * base + 0.24 * process + 0.20 * balance
        return max(0.06, combined ** 1.08 * self._gm_franchise_alloc_mult(
            p, "assist_involvement", "passing", "puck_control"
        ))

    def _gm_event_involvement_weight(self, p: Any) -> float:
        return self._gm_offensive_skill_composite(p)

    def _gm_pick_sequence_driver(
        self,
        rng: random.Random,
        unit: Sequence[Any],
        team: Any,
        *,
        last_driver: Any = None,
        strength: str = "EV",
    ) -> Any:
        if not unit:
            raise ValueError("empty unit")
        return self._gm_pick_weighted(
            rng,
            unit,
            lambda p: self._gm_sequence_driver_weight(p, team, last_driver=last_driver, strength=strength),
        )

    def _gm_pick_shooter_from_unit(
        self,
        rng: random.Random,
        unit: Sequence[Any],
        chance_type: str,
        strength: str,
        team: Any,
        driver: Any,
        *,
        ledger: Optional[Dict[str, Dict[str, Any]]] = None,
        team_id: str = "",
    ) -> Any:
        if not unit:
            raise ValueError("empty unit")

        def _shooter_weight(p: Any) -> float:
            ovr_n = self._gm_ovr_norm(p)
            shoot = self._gm_rating_avg(p, OFFENSE_KEYS) / 99.0
            balance = self._gm_production_balance_score(p)
            vol = min(1.55, self._gm_shot_volume_weight(p) ** 0.72)
            # Overall leads finishing share — depth talent should not outscore stars.
            w = (
                0.48 * (ovr_n ** 1.85)
                + 0.16 * shoot
                + 0.10 * vol
                + 0.10 * balance
                + 0.16 * self._gm_offensive_skill_composite(p)
            )
            w *= self._gm_scoring_hub_bonus(p, team)
            if ledger is not None:
                w *= self._gm_goal_assist_balance_mult(p, ledger, team_id, role="score")
            if self._gm_is_certified_sniper(p):
                w *= 1.08
            elif self._gm_is_certified_playmaker(p):
                w *= 0.90
            if p is driver:
                w *= 1.08
            ct = str(chance_type or "")
            if ct in ("RUSH_MEDIUM", "SH_RUSH", "HIGH_DANGER_SLOT", "ONE_TIMER", "PP_ONE_TIMER"):
                w *= 1.05 if p is driver else 1.0
            if ovr_n >= 0.90:
                w *= 1.38
            elif ovr_n >= 0.86:
                w *= 1.24
            elif ovr_n >= 0.82:
                w *= 1.12
            elif ovr_n >= 0.78:
                w *= 1.04
            elif ovr_n < 0.72:
                w *= 0.52
            elif ovr_n < 0.78:
                w *= 0.72
            # D finish less than F on EV, but elite PP QBs still threaten.
            if self._gm_pos_str(p).upper() == "D":
                if ovr_n >= 0.90:
                    w *= 0.78
                elif ovr_n >= 0.86:
                    w *= 0.62
                else:
                    w *= 0.48
            return max(0.04, w)

        # temperature > 1 sharpens toward higher weights (stars finish more)
        return self._gm_pick_weighted(rng, unit, _shooter_weight, temperature=1.68, weight_floor=0.04)

    def _gm_possession_weight(self, p: Any) -> float:
        puck = self._gm_rating_lookup(p, "puck_control", "pc_puck_control")
        pm = self._gm_rating_avg(p, PASSING_KEYS)
        aware = self._gm_rating_avg(p, IQ_KEYS)
        df = self._gm_rating_avg(p, DEFENSE_KEYS)
        speed = self._gm_rating_lookup(p, "speed", "skg_speed")
        pt = self._gm_player_type_str(p)
        two_way = 1.10 if "two" in pt else 1.0
        return max(0.04, (0.28 * puck + 0.26 * pm + 0.22 * aware + 0.14 * df + 0.10 * speed) / 99.0 * two_way)

    def _gm_defensive_suppression_weight(self, p: Any) -> float:
        df = self._gm_rating_avg(p, DEFENSE_KEYS)
        stick = self._gm_rating_lookup(p, "stick_checking", "def_stick_checking", default=df)
        block = self._gm_rating_lookup(p, "shot_blocking", "def_shot_blocking", default=df)
        speed = self._gm_rating_lookup(p, "speed", "skg_speed")
        pt = self._gm_player_type_str(p)
        mult = 1.14 if "defensive" in pt or "shutdown" in pt or "stay_at_home" in pt else (
            0.92 if "offensive" in pt else 1.0
        )
        # Extreme defensive awareness / stick work shows as a real sim strength/flaw.
        aware = self._gm_rating_lookup(p, "def_defensive_awareness", default=df)
        extremes = 1.0 + max(-0.12, min(0.14, (aware - float(DEFAULT_NHL_RATING)) / 180.0))
        return max(0.04, (0.42 * df + 0.28 * stick + 0.18 * block + 0.12 * speed) / 99.0 * mult * extremes * self._gm_franchise_alloc_mult(
            p, "defensive_effort", "defensive_awareness", "discipline"
        ))

    def _gm_block_weight(self, p: Any) -> float:
        cache = getattr(p, "_gm_runtime_cache", None)
        if isinstance(cache, dict) and "block_weight" in cache:
            return float(cache["block_weight"])
        block = self._gm_rating_lookup(p, "shot_blocking", "def_shot_blocking")
        df = self._gm_rating_avg(p, DEFENSE_KEYS)
        pos = self._gm_pos_str(p).upper()
        val = max(0.04, (0.55 * block + 0.45 * df) / 99.0 * (1.12 if pos == "D" else 0.82))
        val *= self._gm_franchise_alloc_mult(p, "defensive_effort", "defensive_awareness")
        if isinstance(cache, dict):
            cache["block_weight"] = val
        return val

    def _gm_physical_weight(self, p: Any) -> float:
        phys = self._gm_rating_avg(p, PHYS_KEYS)
        strength = self._gm_rating_lookup(p, "strength", "phy_strength", default=phys)
        agg = self._gm_rating_lookup(p, "aggression", "phy_aggression", default=50.0)
        pt = self._gm_player_type_str(p)
        mult = 1.18 if "grinder" in pt or "power" in pt or "enforcer" in pt else 1.0
        return max(0.04, (0.55 * phys + 0.30 * strength + 0.15 * agg) / 99.0 * mult)

    def _gm_penalty_risk_weight(self, p: Any) -> float:
        phys = self._gm_rating_avg(p, PHYS_KEYS)
        agg = self._gm_rating_lookup(p, "aggression", "phy_aggression", default=50.0)
        discipline = self._gm_rating_lookup(p, "discipline", "iqm_discipline", default=72.0)
        return max(0.02, (0.35 * agg + 0.35 * phys - 0.30 * discipline + 18.0) / 99.0 * self._gm_franchise_alloc_mult(
            p, "penalty_risk", lo=0.35, hi=1.65
        ))

    def _gm_goalie_save_adjustment(self, g: Any, chance_type: str) -> float:
        g_skill = self._gm_rating_avg(g, GOALIE_KEYS) / 99.0
        reflex = self._gm_rating_lookup(g, "reflexes", "g_reflexes", default=g_skill * 99.0) / 99.0
        positioning = self._gm_rating_lookup(g, "positioning", "g_positioning", default=g_skill * 99.0) / 99.0
        # Milder curve: elite ~.912, average ~.900, backup ~.888 — avoids brick-wall user starters.
        base = 0.78 + 0.40 * g_skill
        pt = self._gm_player_type_str(g)
        if "butterfly" in pt:
            reflex *= 1.03
            positioning *= 0.98
        elif "hybrid" in pt:
            reflex *= 0.98
            positioning *= 1.02
        if chance_type in ("HIGH_DANGER_SLOT", "NET_FRONT", "REBOUND", "PP_ONE_TIMER"):
            base *= 0.93 + 0.09 * reflex
        else:
            base *= 0.95 + 0.07 * positioning
        base *= self._gm_franchise_alloc_mult(
            g, "goalie_positioning", "positioning", "rebound_control", "overall_equivalent"
        )
        return max(0.70, min(1.26, base))

    def _gm_offense_weight(self, p: Any) -> float:
        """Legacy composite — prefer specific event-skill helpers for new code."""
        vol = self._gm_shot_volume_weight(p)
        qual = self._gm_shot_quality_weight(p)
        fin = self._gm_finishing_adjustment(p)
        return max(0.04, vol * 0.42 + qual * 0.33 + (fin - 0.9) * 0.25)

    def _gm_forward_lines(self, skaters: List[Any], team: Any = None) -> Tuple[List[List[Any]], List[Any]]:
        deployed = getattr(team, "_franchise_deployed_lineup", None) if team is not None else None
        if isinstance(deployed, dict) and deployed.get("ok"):
            lines = [list(ln) for ln in (deployed.get("forward_lines") or [])]
            while len(lines) < 4:
                lines.append([])
            defs = []
            for pair in deployed.get("defense_pairs") or []:
                defs.extend(list(pair or []))
            if not defs:
                defs = [p for p in skaters if str(self._gm_pos_str(p)).upper() == "D"]
            return lines[:4], defs

        active = list(skaters)
        scratched = set(getattr(team, "_tank_scratched_ids", None) or []) if team is not None else set()
        user_scratch = set(getattr(team, "_user_scratched_ids", None) or []) if team is not None else set()
        scratched |= user_scratch
        if scratched:
            active = [p for p in active if str(getattr(p, "id", "")) not in scratched]
        tank_pressure = int(getattr(team, "_franchise_tank_pressure", 0) or 0) if team is not None else 0

        fw: List[Any] = []
        defs: List[Any] = []
        for p in active:
            if str(self._gm_pos_str(p)).upper() == "D":
                defs.append(p)
            else:
                fw.append(p)

        def _line_sort_key(p: Any) -> float:
            bonus = float(self._gm_ovr_bonus(p))
            if tank_pressure >= 50:
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", getattr(p, "age", 0)) or 0)
                if age <= 22:
                    bonus += 2.5 + (tank_pressure / 35.0)
                elif age >= 32 and tank_pressure >= 70:
                    bonus -= 3.5
            return bonus

        fw.sort(key=_line_sort_key, reverse=True)
        lines: List[List[Any]] = [[], [], [], []]
        n = len(fw)
        if n > 0:
            q = max(1, (n + 3) // 4)
            idx = 0
            for li in range(4):
                chunk = fw[idx : idx + q]
                idx += len(chunk)
                lines[li] = chunk
        return lines, defs

    def _gm_saved_lines_payload(self, team: Any) -> Optional[Dict[str, Any]]:
        """Raw Edit Lines payload attached by franchise before sim (even_strength.lines)."""
        raw = getattr(team, "_franchise_saved_lines", None)
        if isinstance(raw, dict) and (raw.get("forwards") or raw.get("defense") or raw.get("goalies")):
            return raw
        bundled = getattr(team, "_franchise_saved_lines_bundle", None)
        if isinstance(bundled, dict):
            inner = bundled.get("lines")
            if isinstance(inner, dict) and (inner.get("forwards") or inner.get("defense") or inner.get("goalies")):
                return inner
        return None

    def _gm_roster_by_id(self, team: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in self._gm_active_roster(team):
            pid = _id_str(p, "id")
            if pid:
                out[pid] = p
        return out

    def _gm_try_resolve_saved_lineup(self, team: Any) -> Optional[Dict[str, Any]]:
        """
        Convert session.lines even_strength payload into a deployable lineup.
        Returns None when missing/invalid (caller falls back to auto OVR dress).
        """
        payload = self._gm_saved_lines_payload(team)
        if not payload:
            return None

        by_id = self._gm_roster_by_id(team)
        tank = set(str(x) for x in (getattr(team, "_tank_scratched_ids", None) or []))
        tank_pressure = int(getattr(team, "_franchise_tank_pressure", 0) or 0)

        def _healthy(pid: Any) -> Optional[Any]:
            spid = str(pid or "")
            if not spid or spid in tank:
                return None
            p = by_id.get(spid)
            if p is None or self._injury_sidelined(p):
                return None
            return p

        intended: Set[str] = set()
        for group in ("forwards", "defense"):
            for unit in payload.get(group) or []:
                if not isinstance(unit, dict):
                    continue
                for pid in (unit.get("slots") or {}).values():
                    spid = str(pid or "")
                    if spid and spid in by_id:
                        intended.add(spid)
        if len(intended) < 6:
            return None

        assigned_ids: Set[str] = set()

        def _take_from_scratch_pool(is_d: bool) -> Optional[Any]:
            """Injury cover only — prefer skaters who were never in the saved lineup."""
            pool = [
                p
                for p in self._gm_skaters(team)
                if not self._injury_sidelined(p)
                and _id_str(p, "id") not in assigned_ids
                and _id_str(p, "id") not in tank
                and _id_str(p, "id") not in intended
                and ((str(self._gm_pos_str(p)).upper() == "D") if is_d else (str(self._gm_pos_str(p)).upper() != "D"))
            ]
            if not pool:
                # No pure scratches left — last-resort fill from remaining healthy.
                pool = [
                    p
                    for p in self._gm_skaters(team)
                    if not self._injury_sidelined(p)
                    and _id_str(p, "id") not in assigned_ids
                    and _id_str(p, "id") not in tank
                    and ((str(self._gm_pos_str(p)).upper() == "D") if is_d else (str(self._gm_pos_str(p)).upper() != "D"))
                ]
            pool.sort(key=lambda p: self._gm_lineup_sort_key(p, tank_pressure), reverse=True)
            for p in pool:
                pid = _id_str(p, "id")
                if not pid:
                    continue
                assigned_ids.add(pid)
                return p
            return None

        # Resolve slots in saved order so later lines are not stolen as injury covers.
        forward_lines: List[List[Any]] = [[], [], [], []]
        for i, line in enumerate(list(payload.get("forwards") or [])[:4]):
            if not isinstance(line, dict):
                continue
            slots = line.get("slots") or {}
            trio: List[Any] = []
            for slot in ("LW", "C", "RW"):
                raw_pid = str(slots.get(slot) or "")
                if not raw_pid:
                    continue
                p = _healthy(raw_pid)
                if p is not None:
                    pid = _id_str(p, "id")
                    if not pid or pid in assigned_ids:
                        continue
                    assigned_ids.add(pid)
                    trio.append(p)
                    continue
                if raw_pid in by_id:
                    cover = _take_from_scratch_pool(is_d=False)
                    if cover is not None:
                        trio.append(cover)
            forward_lines[i] = trio

        defense_pairs: List[List[Any]] = [[], [], []]
        for i, pair in enumerate(list(payload.get("defense") or [])[:3]):
            if not isinstance(pair, dict):
                continue
            slots = pair.get("slots") or {}
            duo: List[Any] = []
            for slot in ("LD", "RD"):
                raw_pid = str(slots.get(slot) or "")
                if not raw_pid:
                    continue
                p = _healthy(raw_pid)
                if p is not None:
                    pid = _id_str(p, "id")
                    if not pid or pid in assigned_ids:
                        continue
                    assigned_ids.add(pid)
                    duo.append(p)
                    continue
                if raw_pid in by_id:
                    cover = _take_from_scratch_pool(is_d=True)
                    if cover is not None:
                        duo.append(cover)
            defense_pairs[i] = duo

        starter_id = backup_id = third_id = ""
        for gline in list(payload.get("goalies") or [])[:1]:
            if not isinstance(gline, dict):
                continue
            slots = gline.get("slots") or {}
            for key, dest in (("Starter", "starter"), ("Backup", "backup"), ("Third", "third")):
                pid = str(slots.get(key) or "")
                if not pid or pid not in by_id:
                    continue
                if dest == "starter":
                    starter_id = pid
                elif dest == "backup":
                    backup_id = pid
                else:
                    third_id = pid

        dressed_fw = [p for ln in forward_lines for p in ln]
        dressed_d = [p for pr in defense_pairs for p in pr]
        dressed_ids = {_id_str(p, "id") for p in dressed_fw + dressed_d if _id_str(p, "id")}

        if len(dressed_ids) < 6 or len(dressed_fw) < 6 or len(dressed_d) < 2:
            return None

        scratch_ids: Set[str] = set()
        for p in self._gm_skaters(team):
            if self._injury_sidelined(p):
                continue
            pid = _id_str(p, "id")
            if pid and pid not in dressed_ids and pid not in tank:
                scratch_ids.add(pid)

        return {
            "ok": True,
            "source": "user",
            "forward_lines": forward_lines,
            "defense_pairs": defense_pairs,
            "dressed_fw": dressed_fw,
            "dressed_d": dressed_d,
            "dressed_ids": dressed_ids,
            "scratch_ids": scratch_ids,
            "starter_id": starter_id,
            "backup_id": backup_id,
            "third_id": third_id,
        }

    def _gm_line_index_for_player(self, lines: List[List[Any]], p: Any) -> int:
        for i, ln in enumerate(lines):
            if p in ln:
                return i
        return 3

    def _gm_toi_seconds_for_line(self, rng: random.Random, line_idx: int, is_d: bool) -> int:
        if is_d:
            lo, hi = 17, 24
        elif line_idx == 0:
            lo, hi = 18, 22
        elif line_idx == 1:
            lo, hi = 15, 18
        elif line_idx == 2:
            lo, hi = 12, 15
        else:
            lo, hi = 8, 12
        mins = int(rng.randint(lo, hi))
        secs = int(rng.randint(0, 55))
        return mins * 60 + secs

    def _gm_distribute_integer_shares(
        self, rng: random.Random, weights: List[float], total: int
    ) -> List[int]:
        n = len(weights)
        if n == 0 or total <= 0:
            return [0] * n
        ws = [max(1e-9, float(w)) for w in weights]
        s = sum(ws)
        raw = [total * (w / s) for w in ws]
        out = [int(math.floor(x)) for x in raw]
        rem = total - sum(out)
        frac = sorted([(raw[i] - out[i], rng.random(), i) for i in range(n)], reverse=True)
        t = 0
        while rem > 0 and t < max(n * 3, 1):
            _, _, idx = frac[t % len(frac)]
            out[idx] += 1
            rem -= 1
            t += 1
        guard = 0
        while sum(out) > total and guard < n * 8:
            j = min(range(n), key=lambda k: (out[k], -raw[k]))
            if out[j] > 0:
                out[j] -= 1
            guard += 1
        return out

    def _gm_ledger_ensure(self, ledger: Dict[str, Dict[str, Any]], p: Any, team_id: str) -> Dict[str, Any]:
        pid = _id_str(p, "id")
        if not pid:
            return {}
        if pid not in ledger:
            nm = str(getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", None) or "?")
            ledger[pid] = {
                "player_id": pid,
                "name": nm,
                "team_id": str(team_id),
                "position": self._gm_pos_str(p),
                "gp": 0,
                "g": 0,
                "a": 0,
                "pts": 0,
                "sog": 0,
                "pim": 0,
                "hit": 0,
                "blk": 0,
                "toi_sec": 0,
                "ppg": 0,
                "ppa": 0,
                "shg": 0,
                "sha": 0,
                "ga": 0,
                "w": 0,
                "l": 0,
                "otl": 0,
                "saves": 0,
                "shots_against": 0,
                "goalie_shots_against": 0,
                "goalie_ga": 0,
                "so": 0,
                "missed_shots": 0,
                "blocked_attempts_for": 0,
                "cf": 0.0,
                "ca": 0.0,
                "ff": 0.0,
                "fa": 0.0,
                "xgf": 0.0,
                "xga": 0.0,
                "ixg": 0.0,
                "xa": 0.0,
                "gf_on": 0.0,
                "ga_on": 0.0,
                "plus_minus": 0,
                "xgf_pct_sum": 0.0,
                "xgf_pct_gp": 0,
                "on_ice_shots_for": 0.0,
                "on_ice_shots_against": 0.0,
                "goalie_xga": 0.0,
                "primary_assists": 0,
                "secondary_assists": 0,
                "analytics_gp": 0,
            }
        row = ledger[pid]
        row["name"] = str(
            getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", None) or row.get("name")
        )
        row["position"] = self._gm_pos_str(p)
        row["team_id"] = str(team_id)
        return row

    def _gm_ledger_add(self, ledger: Dict[str, Dict[str, Any]], p: Any, team_id: str, **kwargs: Any) -> None:
        if not self._gm_player_on_credited_team(p, team_id):
            return
        pid = _id_str(p, "id")
        # NHL regular season: once a player has 82 GP, refuse further counting
        # credits. Mid-season trades plus uneven schedule pacing previously
        # pushed a few players to 83–85 by stacking both clubs' remaining games.
        if pid:
            existing = ledger.get(pid) or ledger.get(str(pid))
            if isinstance(existing, dict) and int(existing.get("gp", 0) or 0) >= 82:
                return
        row = self._gm_ledger_ensure(ledger, p, team_id)
        if not row:
            return
        pos_u = str(row.get("position") or "").upper()
        for k, v in kwargs.items():
            if not v:
                continue
            if k in _GM_FLOAT_LEDGER_KEYS:
                row[k] = round(float(row.get(k, 0) or 0) + float(v), 4)
            else:
                row[k] = int(row.get(k, 0) or 0) + int(v)
        if pos_u != "G":
            row["pts"] = int(row.get("g", 0)) + int(row.get("a", 0))
        elif "pts" in row:
            row["pts"] = 0
        if int(row.get("gp", 0) or 0) > 82:
            row["gp"] = 82
        if _stats_pipeline_debug() and ("g" in kwargs or "a" in kwargs):
            nm = row.get("name") or getattr(getattr(p, "identity", None), "name", None) or getattr(p, "name", "?")
            print(f"[STAT LEDGER ADD] {nm} {team_id} {kwargs} {dict(row)}")


    def _gm_lineup_sort_key(self, p: Any, tank_pressure: int = 0) -> float:
        ovr = self._gm_ovr_0_100(p)
        off = self._gm_rating_avg(p, OFFENSE_KEYS)
        df = self._gm_rating_avg(p, DEFENSE_KEYS)
        iq = self._gm_rating_avg(p, IQ_KEYS)
        pos = self._gm_pos_str(p).upper()
        role_u = self._gm_role_usage_mult(p)
        # Overall leads dressing order so stars actually play with stars.
        score = 0.58 * ovr + 0.22 * off + 0.12 * df + 0.08 * iq
        score *= 0.90 + 0.10 * min(2.2, role_u)
        if tank_pressure >= 50:
            try:
                age = int(career_player_age(p) or 0)
            except Exception:
                age = 0
            if age <= 22:
                score += 2.0 + tank_pressure / 40.0
            elif age >= 32 and tank_pressure >= 70:
                score -= 3.0
        if pos == "D":
            score = 0.52 * ovr + 0.18 * off + 0.24 * df + 0.06 * iq
        return float(score)

    def _gm_build_dressed_lineup(
        self, team: Any, rng: random.Random
    ) -> Tuple[List[Any], List[Any], List[Any], Set[str]]:
        """Return (dressed_skaters, goalies, scratches_not_dressed, tank_scratches)."""
        tank_scratches = set(str(x) for x in (getattr(team, "_tank_scratched_ids", None) or []))
        setattr(team, "_franchise_deployed_lineup", None)
        setattr(team, "_user_scratched_ids", set())

        deployed = self._gm_try_resolve_saved_lineup(team)
        if deployed is not None:
            setattr(team, "_franchise_deployed_lineup", deployed)
            scratch_ids = set(deployed.get("scratch_ids") or [])
            setattr(team, "_user_scratched_ids", set(scratch_ids))
            dressed_fw = list(deployed.get("dressed_fw") or [])
            dressed_d = list(deployed.get("dressed_d") or [])
            dressed_ids = set(deployed.get("dressed_ids") or [])
            available = [
                p
                for p in self._gm_skaters(team)
                if not self._injury_sidelined(p) and _id_str(p, "id") not in tank_scratches
            ]
            healthy_scratches = [p for p in available if _id_str(p, "id") not in dressed_ids]
            for p in available:
                pid = _id_str(p, "id")
                if pid and pid in scratch_ids:
                    setattr(p, "_recently_scratched", True)
                elif pid and pid in dressed_ids:
                    setattr(p, "_recently_scratched", False)
            # Also exclude user scratches from any accidental re-dress later in the game.
            tank_scratches = set(tank_scratches) | set(scratch_ids)
            gl = [g for g in self._gm_goalies(team) if not self._injury_sidelined(g)] or self._gm_goalies(team)
            return dressed_fw + dressed_d, gl, healthy_scratches, tank_scratches

        all_sk = [p for p in self._gm_skaters(team) if not self._injury_sidelined(p)]
        available = [p for p in all_sk if str(getattr(p, "id", "")) not in tank_scratches]
        tank_pressure = int(getattr(team, "_franchise_tank_pressure", 0) or 0)
        fw = [p for p in available if str(self._gm_pos_str(p)).upper() != "D"]
        defs = [p for p in available if str(self._gm_pos_str(p)).upper() == "D"]
        fw.sort(key=lambda p: self._gm_lineup_sort_key(p, tank_pressure), reverse=True)
        defs.sort(key=lambda p: self._gm_lineup_sort_key(p, tank_pressure), reverse=True)
        target_f, target_d = 12, 6
        dressed_fw = fw[:target_f]
        dressed_d = defs[:target_d]
        if len(dressed_fw) < 10 and len(fw) > len(dressed_fw):
            dressed_fw = fw[: max(10, min(len(fw), target_f))]
        if len(dressed_d) < 4 and len(defs) > len(dressed_d):
            dressed_d = defs[: max(4, min(len(defs), target_d))]
        if len(dressed_fw) + len(dressed_d) < 16:
            pool = dressed_fw + dressed_d
            for p in available:
                if p not in pool:
                    pool.append(p)
                if len(pool) >= 16:
                    break
            dressed_fw = [p for p in pool if str(self._gm_pos_str(p)).upper() != "D"]
            dressed_d = [p for p in pool if str(self._gm_pos_str(p)).upper() == "D"]
        dressed_ids = {str(getattr(p, "id", "")) for p in dressed_fw + dressed_d}
        healthy_scratches = [p for p in available if str(getattr(p, "id", "")) not in dressed_ids]
        gl = [g for g in self._gm_goalies(team) if not self._injury_sidelined(g)] or self._gm_goalies(team)
        return dressed_fw + dressed_d, gl, healthy_scratches, tank_scratches

    def _gm_build_game_units(
        self, dressed_sk: List[Any], team: Any = None
    ) -> Dict[str, Any]:
        deployed = getattr(team, "_franchise_deployed_lineup", None) if team is not None else None
        use_saved = isinstance(deployed, dict) and deployed.get("ok")

        if use_saved:
            lines = [list(ln) for ln in (deployed.get("forward_lines") or [])]
            while len(lines) < 4:
                lines.append([])
            lines = lines[:4]
            pairs = [list(pr) for pr in (deployed.get("defense_pairs") or [])]
            while len(pairs) < 3:
                pairs.append([])
            pairs = pairs[:3]
            fw = [p for ln in lines for p in ln]
            defs = [p for pr in pairs for p in pr]
            # Include any dressed players missing from saved units (injury fills).
            dressed_ids = {_id_str(p, "id") for p in fw + defs if _id_str(p, "id")}
            for p in dressed_sk:
                pid = _id_str(p, "id")
                if not pid or pid in dressed_ids:
                    continue
                if str(self._gm_pos_str(p)).upper() == "D":
                    for pr in pairs:
                        if len(pr) < 2:
                            pr.append(p)
                            defs.append(p)
                            dressed_ids.add(pid)
                            break
                else:
                    for ln in lines:
                        if len(ln) < 3:
                            ln.append(p)
                            fw.append(p)
                            dressed_ids.add(pid)
                            break
        else:
            fw = [p for p in dressed_sk if str(self._gm_pos_str(p)).upper() != "D"]
            defs = [p for p in dressed_sk if str(self._gm_pos_str(p)).upper() == "D"]
            # Auto path: stack by overall so L1/PP1 are the real stars.
            fw.sort(key=self._gm_ovr_bonus, reverse=True)
            defs.sort(key=self._gm_ovr_bonus, reverse=True)
            lines = [[], [], [], []]
            n = len(fw)
            if n:
                q = max(1, (n + 3) // 4)
                idx = 0
                for li in range(4):
                    chunk = fw[idx : idx + q]
                    idx += len(chunk)
                    lines[li] = chunk
            pairs = [[], [], []]
            for i in range(3):
                pairs[i] = defs[i * 2 : i * 2 + 2] if len(defs) > i * 2 else (defs[i * 2 :] if len(defs) > i else [])

        def _resolve_special_unit(payload: Any, unit_id: str, slot_order: List[str]) -> List[Any]:
            if not isinstance(payload, (list, dict)):
                return []
            units = payload if isinstance(payload, list) else payload.get("units") or payload.get("lines") or []
            if isinstance(payload, dict) and not units:
                # raw list stored under even_strength-like wrapper
                units = payload.get("lines") if isinstance(payload.get("lines"), list) else []
            by_id = {_id_str(p, "id"): p for p in (fw + defs) if _id_str(p, "id")}
            # also allow any healthy dressed skater
            for p in dressed_sk:
                pid = _id_str(p, "id")
                if pid:
                    by_id.setdefault(pid, p)
            for unit in units or []:
                if not isinstance(unit, dict):
                    continue
                if str(unit.get("id") or "").lower() != unit_id.lower():
                    continue
                slots = unit.get("slots") or {}
                out: List[Any] = []
                seen: Set[str] = set()
                for slot in slot_order:
                    pid = str(slots.get(slot) or "")
                    p = by_id.get(pid)
                    if p is None or pid in seen:
                        continue
                    seen.add(pid)
                    out.append(p)
                return out
            return []

        pp_payload = getattr(team, "_franchise_saved_pp", None) if team is not None else None
        pk_payload = getattr(team, "_franchise_saved_pk", None) if team is not None else None
        pp1 = _resolve_special_unit(pp_payload, "pp1", ["LW", "C", "RW", "LD", "RD"])
        pp2 = _resolve_special_unit(pp_payload, "pp2", ["LW", "C", "RW", "LD", "RD"])
        pk1 = _resolve_special_unit(pk_payload, "pk1", ["F1", "F2", "D1", "D2"])
        pk2 = _resolve_special_unit(pk_payload, "pk2", ["F1", "F2", "D1", "D2"])

        if len(pp1) < 4 or len(pp2) < 3:
            pp_pool = sorted(
                fw + defs[:2],
                key=lambda p: (
                    (self._gm_ovr_norm(p) * 0.48)
                    + (self._gm_production_balance_score(p) * 0.22)
                    + (self._gm_shot_quality_weight(p) * 0.15)
                    + (self._gm_primary_assist_weight(p) * 0.15)
                ),
                reverse=True,
            )
            if len(pp1) < 4:
                pp1 = pp_pool[:5]
            if len(pp2) < 3:
                pp2 = pp_pool[5:10] if len(pp_pool) > 5 else pp_pool[2:7]
        if len(pk1) < 3 or len(pk2) < 3:
            pk_pool = sorted(
                defs + [p for p in fw if self._gm_defensive_suppression_weight(p) > 0.45],
                key=lambda p: (
                    self._gm_defensive_suppression_weight(p) * 0.7
                    + self._gm_ovr_norm(p) * 0.3
                ),
                reverse=True,
            )
            if len(pk1) < 3:
                pk1 = pk_pool[:4]
            if len(pk2) < 3:
                pk2 = pk_pool[4:8] if len(pk_pool) > 4 else pk_pool[2:6]
        for fi, fp in enumerate(fw):
            setattr(fp, "_gm_game_fwd_rank", fi)
            setattr(fp, "_deployed_line_rank", getattr(fp, "_gm_game_line_idx", fi // 3))
        for li, ln in enumerate(lines):
            # Saved lines preserve slot order for TOI; auto path still ranks within line by OVR.
            ordered = list(ln) if use_saved else sorted(ln, key=self._gm_ovr_bonus, reverse=True)
            for ri, p in enumerate(ordered):
                setattr(p, "_gm_game_line_idx", li)
                setattr(p, "_gm_game_line_rank", ri)
                setattr(p, "_deployed_line_rank", li)
        for pi, pair in enumerate(pairs):
            ordered = list(pair) if use_saved else sorted(pair, key=self._gm_ovr_bonus, reverse=True)
            for ri, p in enumerate(ordered):
                setattr(p, "_gm_game_pair_idx", pi)
                setattr(p, "_gm_game_pair_rank", ri)
                setattr(p, "_deployed_line_rank", pi)
        return {
            "lines": lines,
            "pairs": pairs,
            "pp1": pp1,
            "pp2": pp2,
            "pk1": pk1,
            "pk2": pk2,
            "fw": fw,
            "defs": defs,
        }

    def _gm_allocate_conserved_toi(
        self, rng: random.Random, dressed_sk: List[Any]
    ) -> Dict[str, int]:
        fw = [p for p in dressed_sk if str(self._gm_pos_str(p)).upper() != "D"]
        defs = [p for p in dressed_sk if str(self._gm_pos_str(p)).upper() == "D"]
        fwd_budget = 10800
        def_budget = 7200
        fw_w = [self._gm_toi_usage_weight(p, is_d=False) for p in fw]
        d_w = [self._gm_toi_usage_weight(p, is_d=True) for p in defs]
        fw_sec = self._gm_distribute_integer_shares(rng, fw_w, fwd_budget) if fw else []
        d_sec = self._gm_distribute_integer_shares(rng, d_w, def_budget) if defs else []
        out: Dict[str, int] = {}
        for p, sec in zip(fw, fw_sec):
            pid = _id_str(p, "id")
            if pid:
                # Cap realistic NHL max ATOI even if dressed pool is thin (real-NHL edge).
                out[pid] = int(min(int(sec), 24 * 60))
        for p, sec in zip(defs, d_sec):
            pid = _id_str(p, "id")
            if pid:
                out[pid] = int(min(int(sec), 28 * 60))
        return out

    def _gm_toi_usage_weight(self, p: Any, *, is_d: bool) -> float:
        """
        TOI distribution weight only.

        This is intentionally separate from offensive involvement weights. The
        previous allocator reused scoring/role multipliers, which let elite
        forwards consume defenseman-level minutes while still conserving team TOI.
        """
        if is_d:
            pair_idx = min(2, max(0, int(getattr(p, "_gm_game_pair_idx", 1) or 1)))
            pair_weights = (1.38, 1.05, 0.62)
            base = pair_weights[pair_idx]
            rank = min(1, max(0, int(getattr(p, "_gm_game_pair_rank", 0) or 0)))
            return float(max(0.35, base * (1.04 if rank == 0 else 0.96) * self._gm_readiness_usage_mult(p) * self._gm_franchise_alloc_mult(
                p, "toi_readiness", "effort", "stamina"
            )))

        line_idx = min(3, max(0, int(getattr(p, "_gm_game_line_idx", 2) or 2)))
        line_weights = (1.52, 1.18, 0.86, 0.48)
        base = line_weights[line_idx]
        rank = min(2, max(0, int(getattr(p, "_gm_game_line_rank", 1) or 1)))
        rank_mult = (1.06, 1.00, 0.94)[rank]
        return float(max(0.25, base * rank_mult * self._gm_readiness_usage_mult(p) * self._gm_franchise_alloc_mult(
            p, "toi_readiness", "effort", "stamina"
        )))

    def _gm_estimate_special_teams_toi_splits(
        self, total_toi: int, p: Any, units: Optional[Dict[str, Any]]
    ) -> Tuple[int, int, int]:
        """Estimate EV/PP/PK TOI from line + special-teams deployment."""
        total = max(0, int(total_toi or 0))
        if total <= 0:
            return 0, 0, 0
        if not isinstance(units, dict):
            return total, 0, 0
        pid = _id_str(p, "id")
        pp_ids = {
            _id_str(x, "id")
            for x in list(units.get("pp1") or []) + list(units.get("pp2") or [])
            if _id_str(x, "id")
        }
        pk_ids = {
            _id_str(x, "id")
            for x in list(units.get("pk1") or []) + list(units.get("pk2") or [])
            if _id_str(x, "id")
        }
        line_idx = min(3, max(0, int(getattr(p, "_gm_game_line_idx", 2) or 2)))
        pp_share = 0.18 if pid in pp_ids else (0.11 if line_idx <= 1 else 0.05)
        pk_share = 0.14 if pid in pk_ids else (0.08 if line_idx <= 1 else 0.04)
        pp = min(int(round(total * pp_share)), int(total * 0.30))
        pk = min(int(round(total * pk_share)), int(total * 0.24))
        if pp + pk > total:
            scale = float(total) / float(max(1, pp + pk))
            pp = int(pp * scale)
            pk = int(pk * scale)
        ev = max(0, total - pp - pk)
        return ev, pp, pk

    def _gm_team_attempt_environment(
        self,
        rng: random.Random,
        team: Any,
        opponent: Any,
        *,
        strength_scale: float = 1.0,
        is_home: bool = False,
        chem_mod: float = 1.0,
        fatigue_mod: float = 1.0,
        momentum_mod: float = 1.0,
        score_state: float = 0.0,
    ) -> int:
        """Target shot attempts (CF) for one team — independent of goals."""
        off = self._team_offense_skill(team)
        opp_def = self._team_defense_suppression(opponent)
        opp_g = self._team_goalie_suppression(opponent)
        own_def = self._team_defense_suppression(team)
        cached_star = getattr(team, "_gm_cached_star_impact", None)
        star = (float(cached_star) if cached_star is not None else self._team_superstar_offense_impact(team)) * 0.42
        # Own defense mildly supports transition / zone exits — do not starve
        # offense when a club is built around elite D + goaltending.
        base = (
            58.0
            + 24.0 * (off - 0.5)
            - 12.0 * opp_def
            - 5.5 * opp_g
            + 6.0 * star
            + 4.5 * own_def
        )
        base *= float(strength_scale) * float(chem_mod) * float(fatigue_mod) * float(momentum_mod)
        if is_home:
            base += 1.8
        base += 4.0 * score_state
        base += rng.uniform(-7.0, 7.0)
        return int(round(max(42.0, min(86.0, base))))

    def _gm_regulation_attempt_split(
        self,
        rng: random.Random,
        home: Any,
        away: Any,
        *,
        home_strength_scale: float = 1.0,
        away_strength_scale: float = 1.0,
        chem_h: float = 1.0,
        chem_a: float = 1.0,
    ) -> Tuple[int, int]:
        """
        Zero-sum regulation attempt budget from team talent gap.
        Produces wider seasonal CF%/xGF% spread than independent attempt budgets.
        """
        home_off = self._team_offense_skill(home)
        away_off = self._team_offense_skill(away)
        home_def = self._team_defense_suppression(home)
        away_def = self._team_defense_suppression(away)
        home_star = getattr(home, "_gm_cached_star_impact", None)
        away_star = getattr(away, "_gm_cached_star_impact", None)
        if home_star is None:
            home_star = self._team_superstar_offense_impact(home)
        if away_star is None:
            away_star = self._team_superstar_offense_impact(away)

        edge = (
            (home_off - away_off) * 1.45
            # Elite defense used to *reduce* own shot share (trap stereotype), which
            # made best-D/G clubs chronically lowest-scoring. Possession-style D
            # should slightly help CF%, while opp finishing is already suppressed
            # via _team_defense_suppression on shot quality.
            + (home_def - away_def) * 0.28
            + (float(home_star) - float(away_star)) * 0.62
            + 0.050
        )
        edge *= float(home_strength_scale) / max(0.85, float(away_strength_scale))
        edge += (float(chem_h) - float(chem_a)) * 0.12

        home_share = 0.50 + edge * 0.95
        home_share += rng.uniform(-0.045, 0.045)
        home_share = max(0.30, min(0.70, home_share))

        pace = 102.0 + 16.0 * ((home_off + away_off) * 0.5 - 0.5)
        pace += rng.uniform(-9.0, 9.0)
        total_attempts = int(round(max(90.0, min(128.0, pace))))

        h_attempts = max(26, int(round(total_attempts * home_share)))
        a_attempts = max(26, total_attempts - h_attempts)
        return h_attempts, a_attempts

    def _gm_pick_weighted(
        self,
        rng: random.Random,
        pool: Sequence[Any],
        weight_fn: Callable[[Any], float],
        *,
        temperature: float = 1.0,
        weight_floor: float = 0.02,
    ) -> Any:
        if not pool:
            raise ValueError("empty pool")
        temp = max(0.05, float(temperature))
        w = [max(weight_floor, float(weight_fn(p)) ** temp) for p in pool]
        return rng.choices(list(pool), weights=w, k=1)[0]

    def _gm_on_ice_unit(
        self, units: Dict[str, Any], strength: str, rng: random.Random
    ) -> Tuple[List[Any], List[Any]]:
        st = str(strength or "EV").upper()
        if st == "PP":
            pp_share = 0.78 + 0.06 * min(1.0, sum(self._gm_primary_assist_weight(p) for p in units.get("pp1", [])[:3]) / 3.0)
            pp = units["pp1"] if rng.random() < pp_share else units["pp2"]
            return list(pp), []
        if st in ("SH", "PK"):
            pk_share = 0.74
            pk = units["pk1"] if rng.random() < pk_share else units["pk2"]
            return list(pk), []
        li = rng.choices(range(4), weights=[0.42, 0.28, 0.19, 0.11], k=1)[0]
        pi = rng.choices(range(3), weights=[0.48, 0.33, 0.19], k=1)[0]
        line = list(units["lines"][li] if li < len(units["lines"]) else [])
        pair = list(units["pairs"][pi] if pi < len(units["pairs"]) else [])
        return line + pair, []

    def _gm_goalie_role_weight(self, g: Any) -> float:
        role = str(
            getattr(g, "goalie_role", None)
            or getattr(g, "role", None)
            or getattr(getattr(g, "identity", None), "goalie_role", None)
            or ""
        ).lower()
        if role in ("starter", "starting", "start"):
            return 2.35
        if role in ("backup", "secondary"):
            return 0.72
        if role in ("third", "emergency", "ahl"):
            return 0.35
        return 1.0

    def _gm_goalie_ovr_0_100(self, g: Any) -> float:
        try:
            from app.sim_engine.franchise.storyline_conduct import (  # noqa: WPS433
                get_effective_ovr_display,
            )

            ovr = float(get_effective_ovr_display(g))
        except Exception:
            ovr = None
        if ovr is None or ovr <= 0:
            try:
                ovr = float(career_ovr_0_100(g))
            except Exception:
                ovr = 78.0
        if ovr <= 1.5:
            ovr *= 99.0
        return float(ovr)

    def _gm_goalie_workhorse_factor(self, g: Any) -> float:
        ovr = self._gm_goalie_ovr_0_100(g)
        role_w = self._gm_goalie_role_weight(g)
        return 1.0 if role_w >= 1.8 and ovr >= 86.0 else (0.55 if role_w >= 1.8 else 0.0)

    def _gm_goalie_start_state(self, team: Any) -> Dict[str, Dict[str, int]]:
        st = getattr(team, "_gm_goalie_start_state", None)
        if not isinstance(st, dict):
            st = {}
            setattr(team, "_gm_goalie_start_state", st)
        return st

    def _gm_goalie_usage_strategy(self, goalies: List[Any]) -> str:
        """
        Derive a season usage strategy from live goalie quality/role gap.
        Does not assign fixed GP targets — workload emerges from rest decisions.
        """
        if not goalies:
            return "NORMAL_STARTER"
        ranked = sorted(goalies, key=lambda g: (self._gm_goalie_role_weight(g), self._gm_goalie_ovr_0_100(g)), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        top_ovr = self._gm_goalie_ovr_0_100(top)
        gap = (top_ovr - self._gm_goalie_ovr_0_100(second)) if second is not None else 99.0
        top_role = self._gm_goalie_role_weight(top)
        if second is None:
            return "WORKHORSE"
        if gap >= 10.0 and top_ovr >= 90.0:
            return "WORKHORSE"
        if gap >= 7.0 and (top_role >= 1.8 or top_ovr >= 88.0):
            return "STARTER_HEAVY"
        if gap <= 2.0:
            return "TRUE_TANDEM"
        if gap <= 4.5:
            return "SOFT_TANDEM"
        return "NORMAL_STARTER"

    def _gm_determine_preferred_goalie(self, goalies: List[Any], team: Any = None) -> Any:
        if not goalies:
            return None
        deployed = getattr(team, "_franchise_deployed_lineup", None) if team is not None else None
        if isinstance(deployed, dict) and deployed.get("ok"):
            by_id = {_id_str(g, "id"): g for g in goalies if _id_str(g, "id")}
            for key in ("starter_id", "backup_id", "third_id"):
                pid = str(deployed.get(key) or "")
                if pid and pid in by_id:
                    return by_id[pid]
        return max(
            goalies,
            key=lambda g: (
                self._gm_goalie_role_weight(g),
                self._gm_goalie_ovr_0_100(g),
                float(self._gm_ovr_bonus(g)),
            ),
        )

    def _gm_should_rest_preferred_goalie(
        self,
        preferred: Any,
        backup: Any,
        team: Any,
        *,
        strategy: str,
        is_b2b: bool,
        rng: random.Random,
    ) -> bool:
        if preferred is None or backup is None:
            return False
        history = self._gm_goalie_start_state(team)
        pid = _id_str(preferred, "id")
        rec = history.get(pid, {}) if pid else {}
        consec = int(rec.get("consecutive", 0) or 0)
        recent5 = int(rec.get("starts_last_5", 0) or 0)
        gap = self._gm_goalie_ovr_0_100(preferred) - self._gm_goalie_ovr_0_100(backup)

        # Hard rest triggers — hierarchy first, lottery never decides the starter.
        if is_b2b:
            if strategy in ("WORKHORSE", "STARTER_HEAVY"):
                return consec >= 3 or recent5 >= 4
            if strategy == "NORMAL_STARTER":
                return consec >= 2 or recent5 >= 3
            # Soft/true tandem: B2B almost always rests preferred.
            return True

        if strategy == "WORKHORSE":
            if consec >= 12:
                return True
            if consec >= 9 and rng.random() < 0.35:
                return True
            return False
        if strategy == "STARTER_HEAVY":
            if consec >= 8:
                return True
            if consec >= 6 and rng.random() < 0.40:
                return True
            return False
        if strategy == "NORMAL_STARTER":
            if consec >= 5:
                return True
            if consec >= 3 and rng.random() < 0.28:
                return True
            return False
        if strategy == "SOFT_TANDEM":
            # Prefer ~55/45-ish without nightly lottery: rest every 2–3 starts.
            if consec >= 2:
                return True
            return rng.random() < 0.22
        # TRUE_TANDEM
        if consec >= 1:
            return True
        return rng.random() < 0.45 if gap < 1.5 else rng.random() < 0.35

    def _gm_select_starting_goalie(
        self,
        rng: random.Random,
        goalies: List[Any],
        team: Any,
        *,
        calendar_day: int = 0,
        is_b2b: bool = False,
    ) -> Any:
        """
        Hierarchy + rest model (not a nightly weighted lottery).

        1) Derive usage strategy from quality/role gap.
        2) Identify preferred starter.
        3) Rest preferred only for real workload/B2B/tandem reasons.
        4) Otherwise start preferred.
        """
        if not goalies:
            return None
        available = [g for g in goalies if not self._injury_sidelined(g)]
        if not available:
            available = list(goalies)
        if len(available) == 1:
            chosen = available[0]
        else:
            strategy = self._gm_goalie_usage_strategy(available)
            preferred = self._gm_determine_preferred_goalie(available, team)
            others = [g for g in available if g is not preferred]
            backup = self._gm_determine_preferred_goalie(others, team) if others else None
            setattr(team, "_gm_goalie_usage_strategy", strategy)
            if backup is not None and self._gm_should_rest_preferred_goalie(
                preferred,
                backup,
                team,
                strategy=strategy,
                is_b2b=bool(is_b2b),
                rng=rng,
            ):
                # Among non-preferred options, pick the next-best (still hierarchical).
                chosen = backup
                hist = self._gm_goalie_start_state(team)
                team_rec = hist.setdefault("_team", {})
                team_rec["rest_decisions"] = int(team_rec.get("rest_decisions", 0) or 0) + 1
            else:
                chosen = preferred

        history = self._gm_goalie_start_state(team)
        cid = _id_str(chosen, "id")
        for g in goalies:
            gid = _id_str(g, "id")
            if not gid:
                continue
            rec = history.setdefault(gid, {"gp": 0, "consecutive": 0, "starts_last_5": 0, "last_day": -1})
            if gid == cid:
                rec["gp"] = int(rec.get("gp", 0) or 0) + 1
                rec["consecutive"] = int(rec.get("consecutive", 0) or 0) + 1
                prev_max = int(rec.get("max_consecutive", 0) or 0)
                rec["max_consecutive"] = max(prev_max, int(rec["consecutive"]))
                rec["last_day"] = int(calendar_day)
                recent = list(rec.get("recent_starts", []) or [])
                recent.append(1)
                rec["recent_starts"] = recent[-5:]
                rec["starts_last_5"] = sum(rec["recent_starts"])
                if is_b2b:
                    rec["b2b_starts"] = int(rec.get("b2b_starts", 0) or 0) + 1
            else:
                rec["consecutive"] = 0
                recent = list(rec.get("recent_starts", []) or [])
                recent.append(0)
                rec["recent_starts"] = recent[-5:]
                rec["starts_last_5"] = sum(rec["recent_starts"])
        return chosen

    def _gm_update_goalie_recent_performance(self, team: Any, goalie: Any, saves: int, sa: int) -> None:
        if not goalie or sa <= 0:
            return
        gid = _id_str(goalie, "id")
        if not gid:
            return
        history = self._gm_goalie_start_state(team)
        rec = history.setdefault(gid, {})
        sv_pct = saves / float(sa)
        prev = float(rec.get("recent_save_pct", sv_pct) or sv_pct)
        rec["recent_save_pct"] = round(prev * 0.72 + sv_pct * 0.28, 4)

    def _gm_generate_penalties(
        self,
        rng: random.Random,
        dressed_sk: List[Any],
        team_id: str,
        ledger: Dict[str, Dict[str, Any]],
        *,
        intensity: float = 1.0,
    ) -> Tuple[int, int, float, List[Tuple[float, float]]]:
        """Return (total_pim, pp_opportunities_for_opponent, pp_seconds_for_opponent, penalty_intervals).

        penalty_intervals: (start_sec, duration_sec) within regulation for each infraction.
        """
        if not dressed_sk:
            return 0, 0, 0.0, []
        team_risk = sum(self._gm_penalty_risk_weight(p) for p in dressed_sk) / max(1, len(dressed_sk))
        target_events = max(0, int(round(rng.gauss(3.15, 1.05) * (0.92 + 0.16 * team_risk * intensity))))
        n_events = min(6, max(0, target_events))
        pim_total = 0
        ppo = 0
        pp_seconds = 0.0
        intervals: List[Tuple[float, float]] = []
        for _ in range(n_events):
            mins = int(rng.choices([2, 4, 5], weights=[0.78, 0.16, 0.06], k=1)[0])
            offender = self._gm_pick_weighted(rng, dressed_sk, self._gm_penalty_risk_weight)
            self._gm_ledger_add(ledger, offender, team_id, pim=mins)
            pim_total += mins
            ppo += 1
            duration = float(mins) * 60.0
            pp_seconds += duration
            start = float(rng.uniform(45.0, max(90.0, 3540.0 - duration)))
            intervals.append((start, duration))
        return pim_total, ppo, pp_seconds, intervals

    def _gm_pick_assists_from_unit(
        self,
        rng: random.Random,
        unit: Sequence[Any],
        scorer: Any,
        chance_type: str,
        strength: str,
        driver: Any = None,
        *,
        ledger: Optional[Dict[str, Dict[str, Any]]] = None,
        team_id: str = "",
    ) -> List[Any]:
        from app.sim_engine.gameplay.game_analytics_ledger import assist_count_probability

        pool = [p for p in unit if p is not scorer]
        if not pool:
            return []
        p0, p1, p2 = assist_count_probability(chance_type, strength)
        n = int(rng.choices([0, 1, 2], weights=[p0, p1, p2], k=1)[0])
        n = min(n, len(pool))
        if n <= 0:
            return []
        out: List[Any] = []

        def _primary_w(p: Any) -> float:
            w = self._gm_primary_assist_weight(p)
            if ledger is not None:
                w *= self._gm_goal_assist_balance_mult(p, ledger, team_id, role="assist")
            if driver is not None and p is driver:
                w *= 1.14
            return max(0.10, w)

        if n >= 1:
            out.append(self._gm_pick_weighted(rng, pool, _primary_w, temperature=1.42, weight_floor=0.04))
        if n >= 2:
            pool2 = [p for p in pool if p not in out]
            if pool2:
                def _secondary_w(p: Any) -> float:
                    w = self._gm_secondary_assist_weight(p)
                    if ledger is not None:
                        w *= self._gm_goal_assist_balance_mult(p, ledger, team_id, role="assist")
                    return max(0.08, w)
                out.append(self._gm_pick_weighted(rng, pool2, _secondary_w, temperature=1.32, weight_floor=0.04))
        return out[:2]

    def _run_event_driven_game(
        self,
        rng: random.Random,
        home: Any,
        away: Any,
        hid: str,
        aid: str,
        ledger: Dict[str, Dict[str, Any]],
        *,
        strength_map: Optional[Dict[str, float]] = None,
        home_strength_scale: float = 1.0,
        away_strength_scale: float = 1.0,
        noise_scale: float = 1.0,
        light_mode: bool = False,
        is_playoff: bool = False,
        calendar_day: int = 0,
        home_b2b: bool = False,
        away_b2b: bool = False,
    ) -> Dict[str, Any]:
        from app.sim_engine.gameplay.game_analytics_ledger import (
            CHANCE_TYPE_RAW_XG,
            credit_assist_xa,
            credit_shot_attempt_event,
            pick_chance_type,
            raw_xg_for_chance,
            resolve_goal_probability,
            validate_game_integrity,
        )

        strength_map = strength_map or {}
        home_dressed, home_gl, home_scratches, _ = self._gm_build_dressed_lineup(home, rng)
        away_dressed, away_gl, away_scratches, _ = self._gm_build_dressed_lineup(away, rng)
        home_units = self._gm_build_game_units(home_dressed, home)
        away_units = self._gm_build_game_units(away_dressed, away)
        home_toi = self._gm_allocate_conserved_toi(rng, home_dressed)
        away_toi = self._gm_allocate_conserved_toi(rng, away_dressed)
        home_starter = self._gm_select_starting_goalie(rng, home_gl, home, calendar_day=int(calendar_day), is_b2b=bool(home_b2b))
        away_starter = self._gm_select_starting_goalie(rng, away_gl, away, calendar_day=int(calendar_day), is_b2b=bool(away_b2b))
        self._gm_prime_game_player_caches(home, away, home_dressed, away_dressed)
        setattr(home, "_gm_cached_hubs", sorted(self._gm_skaters(home), key=self._gm_offensive_skill_composite, reverse=True)[:3])
        setattr(away, "_gm_cached_hubs", sorted(self._gm_skaters(away), key=self._gm_offensive_skill_composite, reverse=True)[:3])
        last_driver_by_team: Dict[str, Any] = {hid: None, aid: None}

        chem_h = self._chemistry_game_modifier(home, situation="even")
        chem_a = self._chemistry_game_modifier(away, situation="even")

        home_pim, away_ppo_from_home, away_pp_sec_from_home, home_penalty_intervals = self._gm_generate_penalties(rng, home_dressed, hid, ledger, intensity=noise_scale)
        away_pim, home_ppo_from_away, home_pp_sec_from_away, away_penalty_intervals = self._gm_generate_penalties(rng, away_dressed, aid, ledger, intensity=noise_scale)
        _GM_ATTEMPT_GAME_SECONDS = 18.0

        reg_home = reg_away = 0
        home_pp_goals = away_pp_goals = 0
        scoring_events: List[Dict[str, Any]] = []
        home_team_event_stats: Dict[str, float] = {
            "cf": 0.0,
            "ff": 0.0,
            "sog": 0.0,
            "goals": 0.0,
            "xgf": 0.0,
            "missed": 0.0,
            "blocked_for": 0.0,
        }
        away_team_event_stats: Dict[str, float] = dict(home_team_event_stats)
        event_period = 1
        _REGULATION_GAME_SECONDS = 3600.0
        _audit_pp = bool(getattr(self, "_audit_pp_event_ownership", False))
        _chance_types = list(CHANCE_TYPE_RAW_XG.keys()) if _audit_pp else []

        def _new_strength_funnel_block() -> Dict[str, Any]:
            return {
                "opportunity_slots": 0,
                "possessions": 0,
                "shot_attempts": 0,
                "blocked_attempts": 0,
                "missed_attempts": 0,
                "unblocked_attempts": 0,
                "shots_on_goal": 0,
                "xg": 0.0,
                "goals": 0,
                "chance_types": {ct: 0 for ct in _chance_types},
                "chance_type_xg": {ct: 0.0 for ct in _chance_types},
            }

        _pp_own: Dict[str, Any] = {
            "main_events": 0,
            "main_ev": 0,
            "main_pp": 0,
            "main_sh": 0,
            "post_main_events": 0,
            "post_main_pp_sog": 0,
            "post_main_pp_goals": 0,
            "ev_environment_target": 0,
            "ev_opportunity_slots": 0,
            "pp_opportunity_slots": 0,
            "sh_opportunity_slots": 0,
            "ev_events_during_active_pp": 0,
            "ev_sec_during_active_pp": 0.0,
            "pp_sec_consumed_ev": 0.0,
            "pp_sec_consumed_pp": 0.0,
            "pp_sec_consumed_sh": 0.0,
            "pp_sec_reaching_post_main": 0.0,
            "unconsumed_pp_sec": 0.0,
            "timeline_5v5_sec": 0.0,
            "timeline_home_adv_sec": 0.0,
            "timeline_away_adv_sec": 0.0,
            "timeline_four_four_sec": 0.0,
            "game_sec_elapsed": 0.0,
            "strength_funnel": {
                "EV": _new_strength_funnel_block(),
                "PP": _new_strength_funnel_block(),
                "SH": _new_strength_funnel_block(),
            },
            "pp_possession": {
                "pp_cycles": 0,
                "pp_cycles_before_shot_total": 0,
                "pp_possession_seconds": 0.0,
                "pk_cycle_seconds": 0.0,
                "pp_controlled_seconds": 0.0,
                "sh_attack_seconds": 0.0,
                "pp_possessions_ending_in_shot": 0,
                "pp_rebound_shots": 0,
                "pp_rebound_goals": 0,
            },
            "pp_pre_modifier_block_p_samples": [],
            "pp_shot_counterfactual_attempts": [],
            "timeline_segment_audit": {},
        }
        _audit_game_key = f"{int(calendar_day)}|{hid}|{aid}"
        _pp_shot_attempt_seq = 0
        _last_regulation_tick_audit: Optional[Dict[str, Any]] = None
        _pp_cycles_since_shot = 0
        _pp_rebound_next = False

        def _audit_strength_key(st: str) -> str:
            s = str(st or "EV").upper()
            return s if s in ("EV", "PP", "SH") else "EV"

        def _audit_bump_opportunity(st: str) -> None:
            if not _audit_pp:
                return
            sk = _audit_strength_key(st)
            _pp_own["strength_funnel"][sk]["opportunity_slots"] += 1

        def _audit_bump_possession(st: str) -> None:
            if not _audit_pp:
                return
            sk = _audit_strength_key(st)
            _pp_own["strength_funnel"][sk]["possessions"] += 1

        def _audit_record_shot(
            st: str,
            chance: str,
            outcome: str,
            raw_xg: float,
            *,
            is_rebound: bool = False,
        ) -> None:
            if not _audit_pp:
                return
            nonlocal _pp_cycles_since_shot
            sk = _audit_strength_key(st)
            blk = _pp_own["strength_funnel"][sk]
            blk["shot_attempts"] += 1
            ct = str(chance or "")
            if ct in blk["chance_types"]:
                blk["chance_types"][ct] += 1
            if ct in blk.get("chance_type_xg", {}):
                blk["chance_type_xg"][ct] += float(raw_xg)
            blk["xg"] += float(raw_xg)
            oc = str(outcome or "").upper()
            if oc == "BLOCKED":
                blk["blocked_attempts"] += 1
            elif oc == "MISSED":
                blk["missed_attempts"] += 1
            elif oc in ("SAVED", "GOAL"):
                blk["shots_on_goal"] += 1
            else:
                blk["missed_attempts"] += 1
            if oc != "BLOCKED":
                blk["unblocked_attempts"] += 1
            if oc == "GOAL":
                blk["goals"] += 1
            if sk == "PP":
                ppf = _pp_own["pp_possession"]
                ppf["pp_possessions_ending_in_shot"] += 1
                ppf["pp_cycles_before_shot_total"] += int(_pp_cycles_since_shot)
                _pp_cycles_since_shot = 0
                if is_rebound:
                    ppf["pp_rebound_shots"] += 1
                    if outcome == "GOAL":
                        ppf["pp_rebound_goals"] += 1

        def _active_penalty_count(intervals: List[Tuple[float, float]], t: float) -> int:
            return sum(1 for start, dur in intervals if start <= t < start + dur)

        def _remaining_pp_seconds(opponent_intervals: List[Tuple[float, float]], t: float) -> float:
            return sum(max(0.0, (start + dur) - t) for start, dur in opponent_intervals if t < start + dur)

        def _expire_one_active_penalty(intervals: List[Tuple[float, float]], t: float) -> None:
            for idx, (start, dur) in enumerate(intervals):
                if start <= t < start + dur:
                    intervals[idx] = (start, max(0.0, t - start))
                    return

        def _manpower_at(t: float) -> Tuple[int, int]:
            home_box = _active_penalty_count(home_penalty_intervals, t)
            away_box = _active_penalty_count(away_penalty_intervals, t)
            return max(3, 5 - home_box), max(3, 5 - away_box)

        def _resolve_manpower_state(t: float) -> str:
            sk_h, sk_a = _manpower_at(t)
            if sk_h == 5 and sk_a == 5:
                return "ev"
            if sk_h > sk_a:
                return "home_pp"
            if sk_a > sk_h:
                return "away_pp"
            return "four_four"

        def _team_agg(sk: List[Any], tid: str) -> Dict[str, int]:
            cf = ff = sog = goals = 0
            for p in sk:
                pid = _id_str(p, "id")
                if not pid:
                    continue
                row = ledger.get(pid, {})
                cf += int(float(row.get("cf", 0) or 0))
                ff += int(float(row.get("ff", 0) or 0))
                sog += int(row.get("sog", 0) or 0)
                goals += int(row.get("g", 0) or 0)
            return {"cf": cf, "ff": ff, "sog": sog, "goals": goals, "team_id": tid}

        def _record_team_attempt(side_label: str, outcome: str, raw_xg_value: float) -> None:
            stats = home_team_event_stats if side_label == "home" else away_team_event_stats
            stats["cf"] += 1.0
            stats["xgf"] += float(raw_xg_value)
            oc = str(outcome or "").upper()
            if oc == "BLOCKED":
                stats["blocked_for"] += 1.0
                return
            stats["ff"] += 1.0
            if oc == "MISSED":
                stats["missed"] += 1.0
                return
            if oc in ("SAVED", "GOAL"):
                stats["sog"] += 1.0
            if oc == "GOAL":
                stats["goals"] += 1.0

        def _simulate_segment(
            attempts_home: int,
            attempts_away: int,
            *,
            track_goals: bool = True,
            sudden_death: bool = False,
            chronological: bool = False,
            ev_mean_delay: float = 31.0,
        ) -> bool:
            nonlocal reg_home, reg_away, home_pp_goals, away_pp_goals, event_period, _pp_rebound_next, _pp_cycles_since_shot, _pp_shot_attempt_seq, _last_regulation_tick_audit
            total = max(1, attempts_home + attempts_away)
            home_share = attempts_home / float(max(1, attempts_home + attempts_away))
            _event_i = 0
            game_sec_elapsed = 0.0
            if chronological:
                _pp_own["ev_environment_target"] = int(h_attempts + a_attempts)

            while True:
                if chronological:
                    if game_sec_elapsed >= _REGULATION_GAME_SECONDS:
                        break
                elif _event_i >= total:
                    break
                if sudden_death and reg_home != reg_away:
                    return True

                side: str = ""
                strength: str = "EV"
                mp_state = "ev"
                cycle_only = False
                is_pp_rebound = bool(_pp_rebound_next)
                _pp_rebound_next = False

                if chronological:
                    mp_state = _resolve_manpower_state(game_sec_elapsed)
                    if mp_state == "ev":
                        side = "home" if rng.random() < home_share else "away"
                        strength = "EV"
                    elif mp_state == "home_pp":
                        r = rng.random()
                        if r < 0.10:
                            side, strength = "away", "SH"
                        elif r < 0.64:
                            cycle_only = True
                        else:
                            side, strength = "home", "PP"
                    elif mp_state == "away_pp":
                        r = rng.random()
                        if r < 0.10:
                            side, strength = "home", "SH"
                        elif r < 0.64:
                            cycle_only = True
                        else:
                            side, strength = "away", "PP"
                    else:
                        side = "home" if rng.random() < 0.5 else "away"
                        strength = "EV"
                else:
                    sk_h, sk_a = _manpower_at(game_sec_elapsed)
                    if sk_h > 5 and sk_a < 5:
                        side = "home" if rng.random() < 0.86 else "away"
                    elif sk_a > 5 and sk_h < 5:
                        side = "away" if rng.random() < 0.86 else "home"
                    else:
                        side = "home" if rng.random() < home_share else "away"

                if not chronological:
                    sk_h, sk_a = _manpower_at(game_sec_elapsed)
                    if side == "home":
                        if sk_h > sk_a and sk_a < 5:
                            strength = "PP"
                        elif sk_a > sk_h and sk_h < 5:
                            strength = "SH"
                        else:
                            strength = "EV"
                    else:
                        if sk_a > sk_h and sk_h < 5:
                            strength = "PP"
                        elif sk_h > sk_a and sk_a < 5:
                            strength = "SH"
                        else:
                            strength = "EV"

                if chronological and cycle_only:
                    delay = rng.uniform(8.0, 12.0)
                    if _audit_pp and mp_state in ("home_pp", "away_pp"):
                        _pp_own["pp_possession"]["pp_cycles"] += 1
                        _pp_cycles_since_shot += 1
                        _audit_bump_opportunity("PP")
                    game_sec_before = float(game_sec_elapsed)
                    if mp_state == "home_pp":
                        _pp_own["timeline_home_adv_sec"] += delay
                        _pp_own["pp_sec_consumed_pp"] += delay
                        _pp_own["pp_possession"]["pk_cycle_seconds"] += delay
                    elif mp_state == "away_pp":
                        _pp_own["timeline_away_adv_sec"] += delay
                        _pp_own["pp_sec_consumed_pp"] += delay
                        _pp_own["pp_possession"]["pk_cycle_seconds"] += delay
                    game_sec_elapsed += delay
                    if _audit_pp:
                        _last_regulation_tick_audit = {
                            "tick_sec": round(float(delay), 6),
                            "remaining_before_tick_sec": round(
                                max(0.0, _REGULATION_GAME_SECONDS - game_sec_before), 6
                            ),
                            "game_sec_before": round(game_sec_before, 6),
                            "game_sec_after": round(float(game_sec_elapsed), 6),
                            "mp_state": str(mp_state),
                            "strength": "PK_CYCLE",
                            "cycle_only": True,
                        }
                    _event_i += 1
                    continue

                if chronological and strength == "EV" and mp_state in ("home_pp", "away_pp"):
                    _pp_own["ev_events_during_active_pp"] += 1

                if _audit_pp:
                    _audit_bump_opportunity(strength)
                    if strength == "PP":
                        _audit_bump_possession("PP")
                    elif strength == "SH":
                        _audit_bump_possession("SH")
                    elif strength == "EV" and mp_state == "ev":
                        _audit_bump_possession("EV")

                if side == "home":
                    if not chronological:
                        sk_h, sk_a = _manpower_at(game_sec_elapsed)
                        if sk_h > sk_a and sk_a < 5:
                            strength = "PP"
                        elif sk_a > sk_h and sk_h < 5:
                            strength = "SH"
                        else:
                            strength = "EV"
                    atk_team, def_team = home, away
                    atk_sk, def_sk = home_dressed, away_dressed
                    atk_units, def_units = home_units, away_units
                    atk_gl, def_gl = home_gl, away_gl
                    tid, oid = hid, aid
                    score_state = 0.12 if reg_home < reg_away else (-0.08 if reg_home > reg_away else 0.0)
                else:
                    if not chronological:
                        sk_h, sk_a = _manpower_at(game_sec_elapsed)
                        if sk_a > sk_h and sk_h < 5:
                            strength = "PP"
                        elif sk_h > sk_a and sk_a < 5:
                            strength = "SH"
                        else:
                            strength = "EV"
                    atk_team, def_team = away, home
                    atk_sk, def_sk = away_dressed, home_dressed
                    atk_units, def_units = away_units, home_units
                    atk_gl, def_gl = away_gl, home_gl
                    tid, oid = aid, hid
                    score_state = 0.12 if reg_away < reg_home else (-0.08 if reg_away > reg_home else 0.0)

                atk_unit, _ = self._gm_on_ice_unit(atk_units, strength, rng)
                def_unit, _ = self._gm_on_ice_unit(def_units, "PK" if strength == "PP" else ("PP" if strength == "SH" else "EV"), rng)
                if not atk_unit:
                    atk_unit = atk_sk[:5] if atk_sk else atk_sk
                if not def_unit:
                    def_unit = def_sk[:5] if def_sk else def_sk

                chance = pick_chance_type(
                    rng,
                    strength,
                    quality_bias=max((self._gm_shot_quality_weight(p) for p in atk_unit), default=0.5),
                )
                driver = self._gm_pick_sequence_driver(
                    rng,
                    atk_unit,
                    atk_team,
                    last_driver=last_driver_by_team.get(tid),
                    strength=strength,
                )
                shooter = self._gm_pick_shooter_from_unit(
                    rng, atk_unit, chance, strength, atk_team, driver, ledger=ledger, team_id=tid,
                )
                last_driver_by_team[tid] = driver
                qual = self._gm_shot_quality_weight(shooter)
                raw_xg = raw_xg_for_chance(chance, rng)
                atk_off = self._team_offense_skill(atk_team)
                def_sup = self._team_defense_suppression(def_team)
                raw_xg *= max(0.90, min(1.08, 0.98 + (atk_off - def_sup) * 0.28 + (qual - 0.5) * 0.06))

                base_block_p = min(0.38, 0.12 + 0.18 * sum(self._gm_block_weight(p) for p in def_unit) / max(1, len(def_unit)))
                avg_def_block_weight = sum(self._gm_block_weight(p) for p in def_unit) / max(1, len(def_unit))
                base_on_goal_p = 0.62 + 0.08 * qual
                block_p = base_block_p
                if strength == "PP":
                    if _audit_pp:
                        _pp_own["pp_pre_modifier_block_p_samples"].append(round(base_block_p, 6))
                        post_live_block_p = base_block_p * 0.64
                        post_live_on_goal_p = min(0.86, base_on_goal_p + 0.10)
                        _pp_own["pp_shot_counterfactual_attempts"].append({
                            "audit_game_key": _audit_game_key,
                            "calendar_day": int(calendar_day),
                            "home_team_id": str(hid),
                            "away_team_id": str(aid),
                            "pp_attempt_index": int(_pp_shot_attempt_seq),
                            "attacking_side": str(side),
                            "strength": "PP",
                            "chance_type": str(chance or ""),
                            "raw_xg": round(float(raw_xg), 6),
                            "qual": round(float(qual), 6),
                            "pre_pp_block_p": round(float(base_block_p), 6),
                            "post_modifier_live_block_p": round(float(post_live_block_p), 6),
                            "base_on_goal_p": round(float(base_on_goal_p), 6),
                            "post_modifier_live_on_goal_p": round(float(post_live_on_goal_p), 6),
                            "avg_def_block_weight": round(float(avg_def_block_weight), 6),
                            "def_unit_size": int(len(def_unit)),
                            "def_block_weights": [
                                round(float(self._gm_block_weight(p)), 6) for p in def_unit
                            ],
                            "is_rebound": bool(is_pp_rebound),
                        })
                        _pp_shot_attempt_seq += 1
                    block_p *= 0.64
                elif strength == "SH":
                    block_p = min(0.42, block_p * 1.12)
                blocker = None
                outcome = "MISSED"
                if rng.random() < block_p:
                    outcome = "BLOCKED"
                    blocker = self._gm_pick_weighted(rng, def_unit, self._gm_block_weight)
                else:
                    on_goal_p = 0.62 + 0.08 * qual
                    if strength == "PP":
                        on_goal_p = min(0.86, on_goal_p + 0.10)
                    elif strength == "SH":
                        on_goal_p = max(0.45, on_goal_p - 0.06)
                    if rng.random() < on_goal_p:
                        d0 = away_starter if side == "home" else home_starter
                        if d0 is None and def_gl:
                            d0 = max(def_gl, key=lambda x: self._gm_ovr_bonus(x))
                        fin = self._gm_finishing_adjustment(shooter)
                        g_adj = self._gm_goalie_save_adjustment(d0, chance) if d0 else 1.0
                        g_adj = 2.0 - g_adj
                        prob = resolve_goal_probability(raw_xg, fin, g_adj)
                        outcome = "GOAL" if rng.random() < prob else "SAVED"
                    else:
                        outcome = "MISSED"

                def_goalie = away_starter if side == "home" else home_starter
                if def_goalie is None and def_gl:
                    def_goalie = max(def_gl, key=lambda x: self._gm_ovr_bonus(x))

                if not light_mode:
                    credit_shot_attempt_event(
                        ledger,
                        attacking_skaters=atk_unit,
                        defending_skaters=def_unit,
                        shooter=shooter,
                        defending_goalie=def_goalie if def_goalie and outcome in ("SAVED", "GOAL") else None,
                        team_id=tid,
                        opp_team_id=oid,
                        raw_xg=raw_xg,
                        outcome=outcome,
                        blocker=blocker,
                        ledger_add=self._gm_ledger_add,
                        player_id=lambda p: _id_str(p, "id"),
                        strength=strength,
                    )
                else:
                    for p in atk_unit:
                        self._gm_ledger_add(ledger, p, tid, cf=1.0)
                    for p in def_unit:
                        self._gm_ledger_add(ledger, p, oid, ca=1.0)
                    if outcome != "BLOCKED":
                        for p in atk_unit:
                            self._gm_ledger_add(ledger, p, tid, ff=1.0)
                        for p in def_unit:
                            self._gm_ledger_add(ledger, p, oid, fa=1.0)
                    if outcome in ("SAVED", "GOAL"):
                        self._gm_ledger_add(ledger, shooter, tid, sog=1, ixg=raw_xg)
                        if strength == "PP":
                            self._gm_ledger_add(ledger, shooter, tid, pp_sog=1)
                        for p in atk_unit:
                            self._gm_ledger_add(ledger, p, tid, xgf=raw_xg, on_ice_shots_for=1)
                        for p in def_unit:
                            self._gm_ledger_add(ledger, p, oid, xga=raw_xg, on_ice_shots_against=1)
                        if def_goalie is not None:
                            gkw: Dict[str, Any] = {"goalie_xga": raw_xg, "goalie_shots_against": 1}
                            if outcome == "GOAL":
                                gkw["goalie_ga"] = 1
                            self._gm_ledger_add(ledger, def_goalie, oid, **gkw)
                    if outcome == "BLOCKED" and blocker is not None:
                        self._gm_ledger_add(ledger, blocker, oid, blk=1)
                    if outcome == "GOAL":
                        gkw: Dict[str, Any] = {"g": 1, "gf_on": 1.0, "plus_minus": 1}
                        if strength == "PP":
                            gkw["ppg"] = 1
                        elif strength == "SH":
                            gkw["shg"] = 1
                        self._gm_ledger_add(ledger, shooter, tid, **gkw)
                        for p in atk_unit:
                            if p is not shooter:
                                self._gm_ledger_add(ledger, p, tid, gf_on=1.0, plus_minus=1)
                        for p in def_unit:
                            self._gm_ledger_add(ledger, p, oid, ga_on=1.0, plus_minus=-1)

                _record_team_attempt(side, outcome, raw_xg)

                if _audit_pp:
                    _audit_record_shot(
                        strength, chance, outcome, raw_xg, is_rebound=is_pp_rebound,
                    )

                if outcome == "GOAL" and track_goals:
                    if side == "home":
                        reg_home += 1
                        if strength == "PP":
                            home_pp_goals += 1
                            _expire_one_active_penalty(away_penalty_intervals, game_sec_elapsed)
                    else:
                        reg_away += 1
                        if strength == "PP":
                            away_pp_goals += 1
                            _expire_one_active_penalty(home_penalty_intervals, game_sec_elapsed)
                    assists = self._gm_pick_assists_from_unit(
                        rng, atk_unit, shooter, chance, strength, driver, ledger=ledger, team_id=tid,
                    )
                    for i, ap in enumerate(assists):
                        akw: Dict[str, int] = {"a": 1}
                        if i == 0:
                            akw["primary_assists"] = 1
                        else:
                            akw["secondary_assists"] = 1
                        if strength == "PP":
                            akw["ppa"] = 1
                        elif strength == "SH":
                            akw["sha"] = 1
                        self._gm_ledger_add(ledger, ap, tid, **akw)
                        if not light_mode:
                            credit_assist_xa(
                                ledger, ap, tid,
                                ledger_add=self._gm_ledger_add,
                                primary_assist_weight=self._gm_primary_assist_weight,
                            )
                    scoring_events.append({
                        "for_team_id": tid,
                        "period": event_period,
                        "clock": f"{rng.randint(0, 19)}:{rng.choice([0, 15, 30, 45]):02d}",
                        "scorer": str(getattr(getattr(shooter, "identity", None), "name", None) or getattr(shooter, "name", "?")),
                        "scorer_id": _id_str(shooter, "id"),
                        "assists": [str(getattr(getattr(a, "identity", None), "name", None) or getattr(a, "name", "?")) for a in assists],
                        "assist_ids": [_id_str(a, "id") for a in assists],
                        "assist_count": len(assists),
                        "strength": strength,
                        "goalie_in_net": bool(def_goalie is not None),
                        "defending_goalie_id": _id_str(def_goalie, "id") if def_goalie is not None else "",
                        "empty_net": bool(def_goalie is None),
                    })
                    if event_period < 3:
                        event_period += int(rng.random() < 0.12)
                if sudden_death and outcome == "GOAL":
                    return True

                pp_live = strength in ("PP", "SH")
                _pp_rebound_continuation = False
                if chronological:
                    if strength == "PP" and outcome == "SAVED" and rng.random() < 0.25:
                        tick = 3.0
                        _pp_rebound_continuation = True
                    elif strength == "PP":
                        tick = rng.uniform(32.0, 46.0)
                    elif strength == "SH":
                        tick = rng.uniform(10.0, 14.0)
                    elif mp_state == "four_four":
                        tick = max(14.0, min(65.0, rng.expovariate(1.0 / max(12.0, ev_mean_delay * 1.4))))
                    else:
                        tick = max(12.0, min(65.0, rng.expovariate(1.0 / max(12.0, ev_mean_delay))))
                    remaining = _REGULATION_GAME_SECONDS - game_sec_elapsed
                    tick = min(tick, max(0.5, remaining))
                    game_sec_before = float(game_sec_elapsed)
                    game_sec_elapsed += tick
                    if _audit_pp:
                        _last_regulation_tick_audit = {
                            "tick_sec": round(float(tick), 6),
                            "remaining_before_tick_sec": round(float(remaining), 6),
                            "game_sec_before": round(game_sec_before, 6),
                            "game_sec_after": round(float(game_sec_elapsed), 6),
                            "mp_state": str(mp_state),
                            "strength": str(strength),
                            "cycle_only": False,
                        }
                    if mp_state == "ev":
                        _pp_own["timeline_5v5_sec"] += tick
                    elif mp_state == "four_four":
                        _pp_own["timeline_four_four_sec"] += tick
                    elif mp_state == "home_pp":
                        _pp_own["timeline_home_adv_sec"] += tick
                    elif mp_state == "away_pp":
                        _pp_own["timeline_away_adv_sec"] += tick
                    if mp_state in ("home_pp", "away_pp"):
                        _pp_own["pp_possession"]["pp_possession_seconds"] += tick
                        if strength == "EV":
                            _pp_own["pp_sec_consumed_ev"] += tick
                            _pp_own["ev_sec_during_active_pp"] += tick
                        elif strength == "PP":
                            _pp_own["pp_sec_consumed_pp"] += tick
                            _pp_own["pp_possession"]["pp_controlled_seconds"] += tick
                        elif strength == "SH":
                            _pp_own["pp_sec_consumed_sh"] += tick
                            _pp_own["pp_possession"]["sh_attack_seconds"] += tick
                    _pp_own["main_events"] += 1
                    if strength == "EV":
                        _pp_own["main_ev"] += 1
                        if mp_state == "ev":
                            _pp_own["ev_opportunity_slots"] += 1
                    elif strength == "PP":
                        _pp_own["main_pp"] += 1
                        _pp_own["pp_opportunity_slots"] += 1
                    elif strength == "SH":
                        _pp_own["main_sh"] += 1
                        _pp_own["sh_opportunity_slots"] += 1
                    if _pp_rebound_continuation:
                        _pp_rebound_next = True
                else:
                    if pp_live:
                        tick = 7.0
                    else:
                        tick = _GM_ATTEMPT_GAME_SECONDS
                    game_sec_elapsed += tick
                _event_i += 1
            if chronological:
                _pp_own["game_sec_elapsed"] = float(game_sec_elapsed)
                _pp_own["unconsumed_pp_sec"] = float(
                    _remaining_pp_seconds(away_penalty_intervals, game_sec_elapsed)
                    + _remaining_pp_seconds(home_penalty_intervals, game_sec_elapsed)
                )
                if _audit_pp:
                    t5 = float(_pp_own["timeline_5v5_sec"])
                    th = float(_pp_own["timeline_home_adv_sec"])
                    ta = float(_pp_own["timeline_away_adv_sec"])
                    t4 = float(_pp_own["timeline_four_four_sec"])
                    bucket_sum = t5 + th + ta + t4
                    state_delta = bucket_sum - _REGULATION_GAME_SECONDS
                    clock_delta = float(game_sec_elapsed) - _REGULATION_GAME_SECONDS
                    _pp_own["timeline_segment_audit"] = {
                        "regulation_target_sec": _REGULATION_GAME_SECONDS,
                        "final_regulation_clock": round(float(game_sec_elapsed), 6),
                        "timeline_5v5_sec": round(t5, 6),
                        "timeline_home_adv_sec": round(th, 6),
                        "timeline_away_adv_sec": round(ta, 6),
                        "timeline_four_four_sec": round(t4, 6),
                        "timeline_bucket_sum": round(bucket_sum, 6),
                        "timeline_state_sum_delta": round(state_delta, 6),
                        "game_clock_vs_target_delta": round(clock_delta, 6),
                        "bucket_sum_vs_clock_delta": round(bucket_sum - float(game_sec_elapsed), 6),
                        "last_regulation_tick": _last_regulation_tick_audit,
                        "pk_cycle_seconds": round(float(_pp_own["pp_possession"].get("pk_cycle_seconds", 0) or 0), 6),
                        "pp_controlled_seconds": round(float(_pp_own["pp_possession"].get("pp_controlled_seconds", 0) or 0), 6),
                        "sh_attack_seconds": round(float(_pp_own["pp_possession"].get("sh_attack_seconds", 0) or 0), 6),
                    }
            return sudden_death and reg_home != reg_away

        h_attempts, a_attempts = self._gm_regulation_attempt_split(
            rng, home, away,
            home_strength_scale=home_strength_scale,
            away_strength_scale=away_strength_scale,
            chem_h=chem_h,
            chem_a=chem_a,
        )
        ev_active_budget = _REGULATION_GAME_SECONDS * 0.91
        ev_mean_delay = (ev_active_budget / max(1.0, float(h_attempts + a_attempts))) * 1.12
        _simulate_segment(
            h_attempts, a_attempts,
            track_goals=True, sudden_death=False,
            chronological=True, ev_mean_delay=ev_mean_delay,
        )
        if _audit_pp and _pp_own["unconsumed_pp_sec"] > 0.5:
            _pp_own["pp_sec_reaching_post_main"] = float(_pp_own["unconsumed_pp_sec"])

        ot_home = ot_away = 0
        goalie_ot_toi_bonus = 0
        overtime = False
        shootout = False
        shootout_home_win = False
        if reg_home == reg_away:
            overtime = True
            if is_playoff:
                ot_period = 0
                while reg_home == reg_away:
                    ot_period += 1
                    ot_attempts_h = rng.randint(14, 22)
                    ot_attempts_a = rng.randint(14, 22)
                    before_h, before_a = reg_home, reg_away
                    _simulate_segment(ot_attempts_h, ot_attempts_a, track_goals=True, sudden_death=True)
                    ot_home += reg_home - before_h
                    ot_away += reg_away - before_a
                    ot_toi_bonus = rng.randint(180, 300)
                    goalie_ot_toi_bonus += int(ot_toi_bonus)
                    for p in home_dressed:
                        pid = _id_str(p, "id")
                        if pid:
                            home_toi[pid] = int(home_toi.get(pid, 0) or 0) + ot_toi_bonus
                    for p in away_dressed:
                        pid = _id_str(p, "id")
                        if pid:
                            away_toi[pid] = int(away_toi.get(pid, 0) or 0) + ot_toi_bonus
                    if ot_period > 12:
                        break
            else:
                ot_attempts_h = rng.randint(4, 8)
                ot_attempts_a = rng.randint(4, 8)
                before_h, before_a = reg_home, reg_away
                _simulate_segment(ot_attempts_h, ot_attempts_a, track_goals=True, sudden_death=True)
                ot_home = reg_home - before_h
                ot_away = reg_away - before_a
                ot_toi_bonus = rng.randint(90, 180)
                goalie_ot_toi_bonus += int(ot_toi_bonus)
                for p in home_dressed:
                    pid = _id_str(p, "id")
                    if pid:
                        home_toi[pid] = int(home_toi.get(pid, 0) or 0) + ot_toi_bonus
                for p in away_dressed:
                    pid = _id_str(p, "id")
                    if pid:
                        away_toi[pid] = int(away_toi.get(pid, 0) or 0) + ot_toi_bonus
                if reg_home == reg_away and not is_playoff:
                    shootout = True
                    shootout_home_win = rng.random() < 0.52

        for p in home_dressed + away_dressed:
            pid = _id_str(p, "id")
            tid = hid if p in home_dressed else aid
            toi = int(home_toi.get(pid, 0) if p in home_dressed else away_toi.get(pid, 0))
            units = home_units if p in home_dressed else away_units
            ev_toi, pp_toi, pk_toi = self._gm_estimate_special_teams_toi_splits(toi, p, units)
            self._gm_ledger_add(
                ledger,
                p,
                tid,
                gp=1,
                toi_sec=toi,
                ev_toi_sec=ev_toi,
                pp_toi_sec=pp_toi,
                pk_toi_sec=pk_toi,
                analytics_gp=1,
            )

        hit_targets_h = int(18 + 8 * (1.0 - self._team_offense_skill(home)))
        hit_targets_a = int(18 + 8 * (1.0 - self._team_offense_skill(away)))
        if home_dressed:
            hit_shares = self._gm_distribute_integer_shares(rng, [self._gm_physical_weight(p) * (home_toi.get(_id_str(p, "id"), 600) / 600.0) for p in home_dressed], hit_targets_h)
            for p, h in zip(home_dressed, hit_shares):
                if h:
                    self._gm_ledger_add(ledger, p, hid, hit=h)
        if away_dressed:
            hit_shares = self._gm_distribute_integer_shares(rng, [self._gm_physical_weight(p) * (away_toi.get(_id_str(p, "id"), 600) / 600.0) for p in away_dressed], hit_targets_a)
            for p, h in zip(away_dressed, hit_shares):
                if h:
                    self._gm_ledger_add(ledger, p, aid, hit=h)

        def _goalie_rows(goalies: List[Any], tid: str, team_ga: int, opp_sog: int, won: bool, otl: bool, starter: Any, team: Any) -> List[Dict[str, Any]]:
            if not goalies:
                return []
            rows: List[Dict[str, Any]] = []
            full_toi = int(3600 + max(0, goalie_ot_toi_bonus))
            for g0 in goalies:
                pid = _id_str(g0, "id")
                if not pid:
                    continue
                gr = ledger.get(pid, {})
                is_starter = bool(starter is not None and pid == _id_str(starter, "id"))
                faced_sa = int(gr.get("goalie_shots_against", 0) or 0)
                faced_ga = int(gr.get("goalie_ga", 0) or 0)
                if is_starter and faced_sa <= 0 and int(opp_sog) > 0:
                    faced_sa = int(opp_sog)
                    faced_ga = int(team_ga)
                appeared = is_starter or faced_sa > 0 or faced_ga > 0
                if not appeared:
                    continue
                saves = max(0, faced_sa - faced_ga)
                gkw: Dict[str, int] = {
                    "gp": 1,
                    "ga": int(faced_ga),
                    "shots_against": int(faced_sa),
                    "saves": int(saves),
                    "toi_sec": int(gr.get("goalie_toi_sec", 0) or (full_toi if is_starter else 0)),
                }
                if is_starter:
                    if won:
                        gkw["w"] = 1
                        if int(team_ga) == 0:
                            gkw["so"] = 1
                    elif otl:
                        gkw["otl"] = 1
                    else:
                        gkw["l"] = 1
                self._gm_ledger_add(ledger, g0, tid, **gkw)
                gr = self._gm_ledger_ensure(ledger, g0, tid)
                xga_sum = float(gr.get("goalie_xga", 0) or 0)
                if xga_sum > 0:
                    gr["xga"] = round(xga_sum, 4)
                if is_starter:
                    self._gm_update_goalie_recent_performance(team, g0, saves, faced_sa)
                rows.append({
                    "player_id": pid,
                    "name": str(getattr(getattr(g0, "identity", None), "name", None) or getattr(g0, "name", "?")),
                    "ga": int(faced_ga),
                    "saves": int(saves),
                    "shots_against": int(faced_sa),
                    "starter": bool(is_starter),
                })
            return rows

        # reg_* already includes OT goals (OT segments mutate the same counters).
        # Do NOT add ot_* again — that double-counts OT in player-box metadata.
        player_goals_home = int(reg_home)
        player_goals_away = int(reg_away)
        regulation_only_home = int(reg_home) - int(ot_home)
        regulation_only_away = int(reg_away) - int(ot_away)
        home_sog = sum(int(ledger.get(_id_str(p, "id"), {}).get("sog", 0) or 0) for p in home_dressed)
        away_sog = sum(int(ledger.get(_id_str(p, "id"), {}).get("sog", 0) or 0) for p in away_dressed)

        display_home = player_goals_home + (1 if shootout and shootout_home_win else 0)
        display_away = player_goals_away + (1 if shootout and not shootout_home_win else 0)
        if shootout and regulation_only_home == regulation_only_away and ot_home == ot_away:
            if shootout_home_win:
                display_home = player_goals_home + 1
            else:
                display_away = player_goals_away + 1

        home_won = display_home > display_away
        away_won = not home_won
        home_goalies_box = _goalie_rows(home_gl, hid, player_goals_away, away_sog, home_won, bool(overtime and away_won), home_starter, home)
        away_goalies_box = _goalie_rows(away_gl, aid, player_goals_home, home_sog, away_won, bool(overtime and home_won), away_starter, away)
        hgk = next((row for row in home_goalies_box if row.get("starter")), home_goalies_box[0] if home_goalies_box else None)
        agk = next((row for row in away_goalies_box if row.get("starter")), away_goalies_box[0] if away_goalies_box else None)

        home_agg = _team_agg(home_dressed, hid)
        away_agg = _team_agg(away_dressed, aid)
        home_agg["goalie_sa"] = away_sog
        away_agg["goalie_sa"] = home_sog
        for key, value in home_team_event_stats.items():
            home_agg[f"team_{key}"] = int(value) if float(value).is_integer() else round(float(value), 4)
        for key, value in away_team_event_stats.items():
            away_agg[f"team_{key}"] = int(value) if float(value).is_integer() else round(float(value), 4)
        integrity = validate_game_integrity(home_agg, away_agg)

        # Optional audit invariant: skater box goals must equal canonical scoring events.
        if bool(getattr(self, "_audit_goal_reconciliation", False)):
            event_goals = len(scoring_events or [])
            box_skater_goals = int(home_agg.get("goals", 0) or 0) + int(away_agg.get("goals", 0) or 0)
            if box_skater_goals != event_goals or box_skater_goals != player_goals_home + player_goals_away:
                raise RuntimeError(
                    "GOAL RECONCILIATION FAILURE: "
                    f"box_skater_goals={box_skater_goals} event_goals={event_goals} "
                    f"player_meta={player_goals_home + player_goals_away} "
                    f"reg={regulation_only_home}-{regulation_only_away} "
                    f"ot={ot_home}-{ot_away} so={shootout} "
                    f"home={hid} away={aid} day={calendar_day}"
                )

        if _audit_pp:
            for sk in ("EV", "PP", "SH"):
                sf = _pp_own["strength_funnel"][sk]
                ba = int(sf.get("blocked_attempts", 0) or 0)
                ma = int(sf.get("missed_attempts", 0) or 0)
                sog = int(sf.get("shots_on_goal", 0) or 0)
                sa = int(sf.get("shot_attempts", 0) or 0)
                if ba + ma + sog != sa:
                    raise RuntimeError(
                        "OUTCOME ACCOUNTING FAILURE: "
                        f"strength={sk} blocked={ba} missed={ma} sog={sog} attempts={sa} "
                        f"home={hid} away={aid} day={calendar_day}"
                    )

        out: Dict[str, Any] = {
            "home_goals": int(display_home),
            "away_goals": int(display_away),
            "display_home_goals": int(display_home),
            "display_away_goals": int(display_away),
            "regulation_home_goals": int(regulation_only_home),
            "regulation_away_goals": int(regulation_only_away),
            "overtime_home_goals": int(ot_home),
            "overtime_away_goals": int(ot_away),
            "player_home_goals": int(player_goals_home),
            "player_away_goals": int(player_goals_away),
            "hockey_home_goals": int(player_goals_home),
            "hockey_away_goals": int(player_goals_away),
            "overtime": bool(overtime),
            "shootout": bool(shootout),
            "shootout_home_win": bool(shootout_home_win),
            "home_shot_attempts": int(home_team_event_stats["cf"]),
            "away_shot_attempts": int(away_team_event_stats["cf"]),
            "home_ff": int(home_team_event_stats["ff"]),
            "away_ff": int(away_team_event_stats["ff"]),
            "home_blocked_attempts_for": int(home_team_event_stats["blocked_for"]),
            "away_blocked_attempts_for": int(away_team_event_stats["blocked_for"]),
            "home_missed_shots": int(home_team_event_stats["missed"]),
            "away_missed_shots": int(away_team_event_stats["missed"]),
            "home_sog": int(home_team_event_stats["sog"]),
            "away_sog": int(away_team_event_stats["sog"]),
            "home_pp_goals": int(home_pp_goals),
            "away_pp_goals": int(away_pp_goals),
            "home_ppo": int(home_ppo_from_away),
            "away_ppo": int(away_ppo_from_home),
            "home_pp_seconds": float(home_pp_sec_from_away),
            "away_pp_seconds": float(away_pp_sec_from_home),
            "home_pim": int(home_pim),
            "away_pim": int(away_pim),
            "scoring_events": scoring_events,
            "home_goalie": hgk,
            "away_goalie": agk,
            "home_goalies": home_goalies_box,
            "away_goalies": away_goalies_box,
            "home_dressed": home_dressed,
            "away_dressed": away_dressed,
            "integrity_issues": integrity,
            "home_xgf": round(float(home_team_event_stats["xgf"]), 4),
            "away_xgf": round(float(away_team_event_stats["xgf"]), 4),
        }
        if _audit_pp:
            out["pp_event_ownership"] = dict(_pp_own)
            if not hasattr(self, "_pp_ownership_season_audit"):
                self._pp_ownership_season_audit = []
            self._pp_ownership_season_audit.append(dict(_pp_own))
        return out

    def _injury_sidelined(self, player: Any) -> bool:
        """True when player should not dress / produce counting stats this game."""
        if int(getattr(player, "_world_injury_games_remaining", 0) or 0) > 0:
            return True
        h = getattr(player, "health", None)
        if h is not None:
            st = getattr(h, "injury_status", None)
            if st is not None:
                nm = str(getattr(st, "name", st))
                if nm not in ("HEALTHY", "healthy", "Healthy"):
                    return True
        return False

    def _roster_injury_depth_penalty(self, team: Any) -> float:
        """Strength multiplier when injuries thin the lineup (game outcome only)."""
        all_sk = self._gm_skaters(team)
        if not all_sk:
            return 1.0
        active = [p for p in all_sk if not self._injury_sidelined(p)]
        ratio = len(active) / max(9.0, float(len(all_sk)))
        return max(0.87, min(1.0, 0.88 + 0.14 * ratio))

    def _accumulate_light_strength_game_stats(
        self,
        rng: random.Random,
        home: Any,
        away: Any,
        hid: str,
        aid: str,
        hg: int,
        ag: int,
        ot: bool,
        ledger: Dict[str, Dict[str, Any]],
        *,
        home_strength_scale: float = 1.0,
        away_strength_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Fast bulk path: allocates G/A/SOG plus talent-driven CF/CA/xGF/xGA so
        WAR / CF% / xGF% stay populated without the full event loop.

        Possession comes from the same zero-sum attempt split as the event path
        (talent gap), not from finishing luck alone — otherwise every skater on a
        team clones ~50% CF% and elite clubs never clear ~53%.
        """
        hg = max(0, int(hg))
        ag = max(0, int(ag))
        home_dressed, home_gl, _, _ = self._gm_build_dressed_lineup(home, rng)
        away_dressed, away_gl, _, _ = self._gm_build_dressed_lineup(away, rng)
        # Stamp _gm_game_line_idx before TOI / goal weights — otherwise every
        # skater defaults to line 2 and stars lose usage separation from depth.
        home_units = self._gm_build_game_units(home_dressed, home)
        away_units = self._gm_build_game_units(away_dressed, away)
        home_sk = [p for p in home_dressed if self._gm_pos_str(p).upper() != "G"]
        away_sk = [p for p in away_dressed if self._gm_pos_str(p).upper() != "G"]
        home_toi = self._gm_allocate_conserved_toi(rng, home_dressed) if home_dressed else {}
        away_toi = self._gm_allocate_conserved_toi(rng, away_dressed) if away_dressed else {}

        # Talent-driven attempt budget (same model as event regulation split).
        try:
            home_cf_n, away_cf_n = self._gm_regulation_attempt_split(
                rng,
                home,
                away,
                home_strength_scale=float(home_strength_scale),
                away_strength_scale=float(away_strength_scale),
            )
        except Exception:
            home_cf_n = max(40, int(round(rng.gauss(55, 6))))
            away_cf_n = max(40, int(round(rng.gauss(55, 6))))
        # Mild finishing tether so blowouts don't look like pure process disasters.
        gd = float(hg - ag)
        if abs(gd) >= 2.0:
            nudge = int(round(gd * 1.15))
            home_cf_n = max(28, min(95, home_cf_n + nudge))
            away_cf_n = max(28, min(95, away_cf_n - nudge))

        # Prime hub multipliers once so goal weight math doesn't re-sort rosters.
        for tm in (home, away):
            if isinstance(getattr(tm, "_gm_hub_mult_by_player", None), dict):
                continue
            hub_map: Dict[Any, float] = {}
            try:
                impact = float(self._team_superstar_offense_impact(tm))
                ranked = sorted(self._gm_skaters(tm), key=self._gm_offensive_skill_composite, reverse=True)
                for i, p in enumerate(ranked[:3]):
                    if impact <= 0.04:
                        hub_map[id(p)] = 1.0
                    elif i == 0:
                        hub_map[id(p)] = 1.0 + impact * 0.62
                    else:
                        hub_map[id(p)] = 1.0 + impact * 0.24
            except Exception:
                hub_map = {}
            try:
                setattr(tm, "_gm_hub_mult_by_player", hub_map)
            except Exception:
                pass

        home_starter = self._gm_determine_preferred_goalie(home_gl, home) if home_gl else None
        away_starter = self._gm_determine_preferred_goalie(away_gl, away) if away_gl else None

        for p in home_dressed:
            pid = _id_str(p, "id")
            toi = int(home_toi.get(pid, 0) or 0)
            ev_toi, pp_toi, pk_toi = self._gm_estimate_special_teams_toi_splits(toi, p, home_units)
            self._gm_ledger_add(
                ledger, p, hid, gp=1, toi_sec=toi, ev_toi_sec=ev_toi, pp_toi_sec=pp_toi, pk_toi_sec=pk_toi,
            )
        for p in away_dressed:
            pid = _id_str(p, "id")
            toi = int(away_toi.get(pid, 0) or 0)
            ev_toi, pp_toi, pk_toi = self._gm_estimate_special_teams_toi_splits(toi, p, away_units)
            self._gm_ledger_add(
                ledger, p, aid, gp=1, toi_sec=toi, ev_toi_sec=ev_toi, pp_toi_sec=pp_toi, pk_toi_sec=pk_toi,
            )

        def _credit_team_goals(skaters: List[Any], tid: str, goals: int, team: Any) -> Tuple[Dict[str, float], int]:
            """Allocate G/A; return (offensive SOG weights, PP goals tagged this game)."""
            off_w: Dict[str, float] = {}
            ast_w: Dict[str, float] = {}
            ppg_tagged = 0
            if not skaters:
                return off_w, 0
            # Precompute weights once — calling scoring-hub / OVR helpers inside
            # _gm_pick_weighted re-sorted the roster per candidate (~80ms/game).
            for p in skaters:
                pid = _id_str(p, "id") or str(id(p))
                ovr_n = self._gm_ovr_norm(p)
                pos = self._gm_pos_str(p).upper()
                try:
                    shoot = float(self._gm_rating_avg(p, OFFENSE_KEYS)) / 99.0
                except Exception:
                    shoot = ovr_n
                try:
                    iq = float(self._gm_rating_avg(p, IQ_KEYS)) / 99.0
                except Exception:
                    iq = ovr_n
                role = self._gm_role_usage_mult(p)
                hub = self._gm_scoring_hub_bonus(p, team)
                # Star-weighted finishing — concentrate points so leaders land ~120–140.
                w_g = 0.54 * (ovr_n ** 1.88) + 0.26 * shoot + 0.20 * (ovr_n ** 1.65)
                usage_scale = 0.50 + 0.50 * min(2.3, float(role))
                hub_scale = 0.80 + 0.20 * float(hub)
                w_g *= usage_scale * hub_scale
                if pos == "D":
                    # Elite D (Makar/Hughes tier) should land ~15–25 G / 70–100 P;
                    # depth D stay well below forwards.
                    w_g *= 0.36
                    if ovr_n >= 0.90:
                        w_g *= 1.55
                    elif ovr_n >= 0.86:
                        w_g *= 1.32
                    elif ovr_n >= 0.82:
                        w_g *= 1.12
                    else:
                        w_g *= 0.78
                elif pos in ("C", "LW", "RW", "F"):
                    w_g *= 1.07
                if ovr_n >= 0.90:
                    w_g *= 1.42
                elif ovr_n >= 0.86:
                    w_g *= 1.26
                elif ovr_n >= 0.82:
                    w_g *= 1.14
                elif ovr_n < 0.72:
                    w_g *= 0.48
                elif ovr_n < 0.78:
                    w_g *= 0.66
                off_w[pid] = max(0.05, w_g)

                w_a = 0.44 * (ovr_n ** 1.58) + 0.32 * iq + 0.24 * shoot
                w_a *= (0.68 + 0.32 * min(2.3, float(role)))
                if pos == "D":
                    # PP QBs and top-pair D own a large assist share — still below star F.
                    w_a *= 0.78
                    if ovr_n >= 0.90:
                        w_a *= 1.55
                    elif ovr_n >= 0.86:
                        w_a *= 1.32
                    elif ovr_n >= 0.82:
                        w_a *= 1.12
                    else:
                        w_a *= 0.88
                if ovr_n >= 0.90:
                    w_a *= 1.28
                elif ovr_n >= 0.86:
                    w_a *= 1.16
                elif ovr_n < 0.72:
                    w_a *= 0.58
                ast_w[pid] = max(0.05, w_a)

            if goals <= 0:
                return off_w, 0

            def _pick(pool: List[Any], weights: Dict[str, float], temp: float) -> Any:
                ids = [_id_str(p, "id") or str(id(p)) for p in pool]
                raw = [max(0.05, float(weights.get(i, 0.05)) ** temp) for i in ids]
                return rng.choices(pool, weights=raw, k=1)[0]

            for _ in range(goals):
                scorer = _pick(skaters, off_w, 1.28)
                gkw: Dict[str, Any] = {"g": 1}
                # ~21% of goals are PP (modern NHL); mark for team PP counting.
                if rng.random() < 0.21:
                    gkw["ppg"] = 1
                    ppg_tagged += 1
                self._gm_ledger_add(ledger, scorer, tid, **gkw)
                remaining = [p for p in skaters if p is not scorer]
                # Primary + secondary assist most of the time (~1.82 A/G for star totals).
                n_ast = 1 if rng.random() < 0.18 else 2
                n_ast = min(n_ast, len(remaining))
                for i_a in range(n_ast):
                    if not remaining:
                        break
                    assister = _pick(remaining, ast_w, 1.18)
                    akw: Dict[str, Any] = {"a": 1}
                    if gkw.get("ppg"):
                        akw["ppa"] = 1
                    if i_a == 0:
                        akw["primary_assists"] = 1
                    else:
                        akw["secondary_assists"] = 1
                    self._gm_ledger_add(ledger, assister, tid, **akw)
                    remaining = [p for p in remaining if p is not assister]
            return off_w, ppg_tagged

        def _snapshot_g_a_sog(skaters: List[Any]) -> Dict[str, Tuple[int, int, int]]:
            out: Dict[str, Tuple[int, int, int]] = {}
            for p in skaters:
                pid = _id_str(p, "id")
                if not pid:
                    continue
                row = ledger.get(pid) or {}
                out[pid] = (
                    int(row.get("g", 0) or 0),
                    int(row.get("a", 0) or 0),
                    int(row.get("sog", 0) or 0),
                )
            return out

        # Snapshot before G/A so SOG floors and iXG/xA use this-game deltas only.
        # Using season totals here previously triangular-summed iXG (~3000) and WAR.
        pre_home_gas = _snapshot_g_a_sog(home_sk)
        pre_away_gas = _snapshot_g_a_sog(away_sk)

        home_off_w, home_ppg = _credit_team_goals(home_sk, hid, hg, home)
        away_off_w, away_ppg = _credit_team_goals(away_sk, aid, ag, away)

        def _pick_on_ice_unit(skaters: List[Any], toi_map: Dict[str, int], n: int = 5) -> List[Any]:
            if not skaters:
                return []
            n = min(max(1, int(n)), len(skaters))
            weights = []
            for p in skaters:
                pid = _id_str(p, "id")
                toi = max(1, int(toi_map.get(pid, 0) or 0))
                ovr_n = max(0.35, self._gm_ovr_norm(p))
                weights.append(toi * (0.55 + 0.45 * ovr_n))
            chosen: List[Any] = []
            pool = list(skaters)
            wpool = list(weights)
            for _ in range(n):
                if not pool:
                    break
                pick = rng.choices(pool, weights=wpool, k=1)[0]
                idx = pool.index(pick)
                chosen.append(pick)
                pool.pop(idx)
                wpool.pop(idx)
            return chosen

        def _credit_on_ice_plus_minus(
            skaters: List[Any],
            toi_map: Dict[str, int],
            tid: str,
            goals_for: int,
            goals_against: int,
        ) -> None:
            """Light-path +/- / on-ice GF-GA so season +/- is not stuck at 0."""
            for _ in range(max(0, int(goals_for))):
                for p in _pick_on_ice_unit(skaters, toi_map, 5):
                    self._gm_ledger_add(ledger, p, tid, gf_on=1.0, plus_minus=1)
            for _ in range(max(0, int(goals_against))):
                for p in _pick_on_ice_unit(skaters, toi_map, 5):
                    self._gm_ledger_add(ledger, p, tid, ga_on=1.0, plus_minus=-1)

        _credit_on_ice_plus_minus(home_sk, home_toi, hid, hg, ag)
        _credit_on_ice_plus_minus(away_sk, away_toi, aid, ag, hg)

        def _team_sog_target(goals: int, team_cf: int) -> int:
            # SOG tracks talent-driven attempts (~52% of CF on net) with mild goal tether.
            base = float(team_cf) * 0.52 + (float(goals) - 3.05) * 0.90
            n = int(round(rng.gauss(base, 2.4)))
            return max(int(goals) + 14, min(42, n))

        def _allocate_team_sog(
            skaters: List[Any],
            tid: str,
            goals: int,
            off_w: Dict[str, float],
            pre_gas: Dict[str, Tuple[int, int, int]],
            team_cf: int,
        ) -> int:
            if not skaters:
                return 0
            sog_n = _team_sog_target(goals, team_cf)
            sog_w = []
            for p in skaters:
                pid = _id_str(p, "id") or str(id(p))
                ovr_n = self._gm_ovr_norm(p)
                pos = self._gm_pos_str(p).upper()
                # Prefer same offensive weights as goals so SH% stays believable.
                base_w = float(off_w.get(pid, 0.0) or 0.0)
                if base_w <= 0:
                    # D still shoot less than F, but elite D need volume for ~70–100 P seasons.
                    d_sog = 0.58 if ovr_n >= 0.86 else (0.48 if ovr_n >= 0.80 else 0.38)
                    base_w = max(0.08, (ovr_n ** 1.35) * (d_sog if pos == "D" else 1.0))
                else:
                    # Extra shot volume for finishers / wings.
                    base_w *= 1.15 if pos in ("LW", "RW", "C", "F") else 0.85
                sog_w.append(max(0.06, base_w))
            shares = self._gm_distribute_integer_shares(rng, sog_w, sog_n)
            for p, n in zip(skaters, shares):
                if n:
                    self._gm_ledger_add(ledger, p, tid, sog=int(n))
            # Floor: this-game scorers need enough THIS-GAME SOG (~25% SH% cap).
            for p in skaters:
                pid = _id_str(p, "id")
                if not pid:
                    continue
                row = ledger.get(pid) or {}
                pre_g, _pre_a, pre_sog = pre_gas.get(pid, (0, 0, 0))
                game_g = max(0, int(row.get("g", 0) or 0) - int(pre_g))
                if game_g <= 0:
                    continue
                game_sog = max(0, int(row.get("sog", 0) or 0) - int(pre_sog))
                need = game_g * 5 + (1 if game_g >= 2 else 0)
                if game_sog >= need:
                    continue
                deficit = need - game_sog
                donors = sorted(
                    (
                        q
                        for q in skaters
                        if q is not p
                        and max(
                            0,
                            int((ledger.get(_id_str(q, "id")) or {}).get("g", 0) or 0)
                            - int(pre_gas.get(_id_str(q, "id") or "", (0, 0, 0))[0]),
                        )
                        == 0
                    ),
                    key=lambda q: max(
                        0,
                        int((ledger.get(_id_str(q, "id")) or {}).get("sog", 0) or 0)
                        - int(pre_gas.get(_id_str(q, "id") or "", (0, 0, 0))[2]),
                    ),
                    reverse=True,
                )
                for donor in donors:
                    if deficit <= 0:
                        break
                    did = _id_str(donor, "id")
                    drow = ledger.get(did) or {}
                    d_pre_sog = int(pre_gas.get(did or "", (0, 0, 0))[2])
                    d_game_sog = max(0, int(drow.get("sog", 0) or 0) - d_pre_sog)
                    avail = max(0, d_game_sog - 1)
                    take = min(avail, deficit)
                    if take <= 0:
                        continue
                    drow["sog"] = int(drow.get("sog", 0) or 0) - take
                    row["sog"] = int(row.get("sog", 0) or 0) + take
                    deficit -= take
                if deficit > 0:
                    row["sog"] = int(row.get("sog", 0) or 0) + deficit
                    sog_n += deficit
            # Individual xG / xA for THIS GAME only (ledger_add accumulates season).
            for p in skaters:
                pid = _id_str(p, "id")
                if not pid:
                    continue
                row = ledger.get(pid) or {}
                pre_g, pre_a, pre_sog = pre_gas.get(pid, (0, 0, 0))
                game_sog = max(0, int(row.get("sog", 0) or 0) - int(pre_sog))
                game_g = max(0, int(row.get("g", 0) or 0) - int(pre_g))
                game_a = max(0, int(row.get("a", 0) or 0) - int(pre_a))
                if game_sog <= 0 and game_g <= 0 and game_a <= 0:
                    continue
                ovr_n = max(0.40, self._gm_ovr_norm(p))
                sh_exp = 0.095 + 0.035 * (ovr_n - 0.75)
                ixg = max(float(game_g) * 0.72, float(game_sog) * sh_exp)
                xa = max(0.0, float(game_a) * (0.55 + 0.20 * ovr_n))
                if ixg > 0 or xa > 0:
                    self._gm_ledger_add(ledger, p, tid, ixg=round(ixg, 4), xa=round(xa, 4))
            return sog_n

        home_sog_n = _allocate_team_sog(home_sk, hid, hg, home_off_w, pre_home_gas, home_cf_n)
        away_sog_n = _allocate_team_sog(away_sk, aid, ag, away_off_w, pre_away_gas, away_cf_n)

        # Peripherals — without these, bulk seasons show nearly empty hit/block columns.
        home_hits = int(max(8, round(rng.gauss(18.0, 3.5))))
        away_hits = int(max(8, round(rng.gauss(18.0, 3.5))))
        home_blks = int(max(6, round(rng.gauss(14.0, 3.0))))
        away_blks = int(max(6, round(rng.gauss(14.0, 3.0))))
        if home_sk and home_hits > 0:
            hit_w = [
                max(0.05, self._gm_physical_weight(p) * (home_toi.get(_id_str(p, "id"), 600) / 600.0))
                for p in home_sk
            ]
            for p, n in zip(home_sk, self._gm_distribute_integer_shares(rng, hit_w, home_hits)):
                if n:
                    self._gm_ledger_add(ledger, p, hid, hit=int(n))
        if away_sk and away_hits > 0:
            hit_w = [
                max(0.05, self._gm_physical_weight(p) * (away_toi.get(_id_str(p, "id"), 600) / 600.0))
                for p in away_sk
            ]
            for p, n in zip(away_sk, self._gm_distribute_integer_shares(rng, hit_w, away_hits)):
                if n:
                    self._gm_ledger_add(ledger, p, aid, hit=int(n))
        if home_sk and home_blks > 0:
            blk_w = []
            for p in home_sk:
                pos = self._gm_pos_str(p).upper()
                base = 1.35 if pos == "D" else 0.55
                blk_w.append(max(0.05, base * self._gm_ovr_norm(p) * (home_toi.get(_id_str(p, "id"), 600) / 600.0)))
            for p, n in zip(home_sk, self._gm_distribute_integer_shares(rng, blk_w, home_blks)):
                if n:
                    self._gm_ledger_add(ledger, p, hid, blk=int(n))
        if away_sk and away_blks > 0:
            blk_w = []
            for p in away_sk:
                pos = self._gm_pos_str(p).upper()
                base = 1.35 if pos == "D" else 0.55
                blk_w.append(max(0.05, base * self._gm_ovr_norm(p) * (away_toi.get(_id_str(p, "id"), 600) / 600.0)))
            for p, n in zip(away_sk, self._gm_distribute_integer_shares(rng, blk_w, away_blks)):
                if n:
                    self._gm_ledger_add(ledger, p, aid, blk=int(n))

        home_team_xga = max(float(ag) * 0.92, away_sog_n * 0.105)
        away_team_xga = max(float(hg) * 0.92, home_sog_n * 0.105)
        goalie_toi = int(3600 + (150 if ot else 0))
        if home_starter is not None:
            sa = max(ag, away_sog_n)
            saves = max(0, sa - ag)
            kw: Dict[str, Any] = {
                "gp": 1,
                "shots_against": sa,
                "goalie_shots_against": sa,
                "saves": saves,
                "ga": ag,
                "goalie_ga": ag,
                "toi_sec": goalie_toi,
                "goalie_xga": round(home_team_xga, 4),
            }
            if hg > ag:
                kw["w"] = 1
            elif ot:
                kw["otl"] = 1
            else:
                kw["l"] = 1
            if ag == 0:
                kw["so"] = 1
            self._gm_ledger_add(ledger, home_starter, hid, **kw)
        if away_starter is not None:
            sa = max(hg, home_sog_n)
            saves = max(0, sa - hg)
            kw = {
                "gp": 1,
                "shots_against": sa,
                "goalie_shots_against": sa,
                "saves": saves,
                "ga": hg,
                "goalie_ga": hg,
                "toi_sec": goalie_toi,
                "goalie_xga": round(away_team_xga, 4),
            }
            if ag > hg:
                kw["w"] = 1
            elif ot:
                kw["otl"] = 1
            else:
                kw["l"] = 1
            if hg == 0:
                kw["so"] = 1
            self._gm_ledger_add(ledger, away_starter, aid, **kw)

        # Possession / expected goals from talent attempt split, allocated with
        # separate CF vs CA player weights so individuals don't all clone team CF%.
        _ON_ICE_UNIT = 5.0

        home_xgf_n = max(float(hg) * 0.92, home_sog_n * 0.105, home_cf_n * 0.055)
        away_xgf_n = max(float(ag) * 0.92, away_sog_n * 0.105, away_cf_n * 0.055)
        home_xga_n = away_xgf_n
        away_xga_n = home_xgf_n
        home_ca_n = int(away_cf_n)
        away_ca_n = int(home_cf_n)

        def _alloc_possession(
            skaters: List[Any],
            toi_map: Dict[str, int],
            tid: str,
            team_cf: int,
            team_ca: int,
            team_xgf: float,
            team_xga: float,
        ) -> None:
            if not skaters:
                return
            cf_weights: List[float] = []
            ca_weights: List[float] = []
            for p in skaters:
                pid = _id_str(p, "id")
                toi = max(1, int(toi_map.get(pid, 0) or 0))
                ovr_n = max(0.35, self._gm_ovr_norm(p))
                pos = self._gm_pos_str(p).upper()
                # Offensive tilt: stars / forwards drive CF; D absorb more CA.
                off_tilt = (ovr_n - 0.72) * 0.28
                if pos == "D":
                    off_tilt -= 0.05
                elif pos in ("LW", "RW", "C", "F"):
                    off_tilt += 0.04
                base = toi * (0.50 + 0.50 * ovr_n)
                cf_weights.append(max(0.05, base * (1.0 + off_tilt)))
                ca_weights.append(max(0.05, base * (1.0 - off_tilt * 0.55)))
            cf_total = sum(cf_weights) or 1.0
            ca_total = sum(ca_weights) or 1.0
            for p, cf_w, ca_w in zip(skaters, cf_weights, ca_weights):
                cf_share = cf_w / cf_total
                ca_share = ca_w / ca_total
                self._gm_ledger_add(
                    ledger,
                    p,
                    tid,
                    cf=round(team_cf * cf_share * _ON_ICE_UNIT, 3),
                    ca=round(team_ca * ca_share * _ON_ICE_UNIT, 3),
                    ff=round(team_cf * cf_share * _ON_ICE_UNIT * 0.78, 3),
                    fa=round(team_ca * ca_share * _ON_ICE_UNIT * 0.78, 3),
                    xgf=round(team_xgf * cf_share * _ON_ICE_UNIT, 4),
                    xga=round(team_xga * ca_share * _ON_ICE_UNIT, 4),
                )

        _alloc_possession(home_sk, home_toi, hid, home_cf_n, home_ca_n, home_xgf_n, home_xga_n)
        _alloc_possession(away_sk, away_toi, aid, away_cf_n, away_ca_n, away_xgf_n, away_xga_n)

        # Special-teams counting for light boxes (CPU–CPU). Without this, Stats Central
        # PP%/PK% only reflect the handful of full-event games vs the user club.
        def _light_ppo(ppg: int) -> int:
            # ~21% conversion ⇒ PPO ≈ PPG / 0.21, with NHL-like 2–4 chances.
            if ppg <= 0:
                return max(1, int(round(rng.gauss(2.7, 0.8))))
            base = max(ppg + 1, int(round(ppg / 0.21)))
            return max(ppg, min(7, base + rng.randint(0, 1)))

        home_ppo = _light_ppo(home_ppg)
        away_ppo = _light_ppo(away_ppg)

        return {
            "home_goals": hg,
            "away_goals": ag,
            "overtime": bool(ot),
            "shootout": False,
            "scoring_events": [],
            "home_dressed": home_dressed,
            "away_dressed": away_dressed,
            "home_sog": int(home_sog_n),
            "away_sog": int(away_sog_n),
            "home_shots": int(home_sog_n),
            "away_shots": int(away_sog_n),
            "home_shot_attempts": int(home_cf_n),
            "away_shot_attempts": int(away_cf_n),
            "home_cf": int(home_cf_n),
            "away_cf": int(away_cf_n),
            "home_ff": int(round(home_cf_n * 0.78)),
            "away_ff": int(round(away_cf_n * 0.78)),
            "home_xgf": round(float(home_xgf_n), 4),
            "away_xgf": round(float(away_xgf_n), 4),
            "home_xg": round(float(home_xgf_n), 4),
            "away_xg": round(float(away_xgf_n), 4),
            "home_pp_goals": int(home_ppg),
            "away_pp_goals": int(away_ppg),
            "home_ppo": int(home_ppo),
            "away_ppo": int(away_ppo),
            "home_ppga": int(away_ppg),
            "away_ppga": int(home_ppg),
            "home_opp_ppo": int(away_ppo),
            "away_opp_ppo": int(home_ppo),
            "light_box": True,
            "stat_source": "light_strength",
        }

    def accumulate_unified_game_stats(
        self,
        rng: random.Random,
        home: Any,
        away: Any,
        hid: str,
        aid: str,
        hg: int,
        ag: int,
        ot: bool,
        ledger: Dict[str, Dict[str, Any]],
        *,
        build_game_payload: bool = False,
        light_mode: bool = False,
        calendar_day: int = 0,
        calendar_iso: str = "",
        game_id: str = "",
        stat_scope: str = "regular_season",
        strength_map: Optional[Dict[str, float]] = None,
        home_strength_scale: float = 1.0,
        away_strength_scale: float = 1.0,
        noise_scale: float = 1.0,
        is_playoff: bool = False,
        home_b2b: bool = False,
        away_b2b: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Event-driven game stat pipeline. Score and stats share one simulation universe.
        Uses cached result from _simulate_game when scores match; otherwise runs fresh.

        light_mode=True (bulk franchise): never re-roll an event game — that would change
        the score and crash franchise integrity checks. Use strength light allocation.
        """
        from app.sim_engine.gameplay.game_analytics_ledger import estimate_pp_opportunities

        if light_mode:
            setattr(self, "_pending_event_game", None)
            result = self._accumulate_light_strength_game_stats(
                rng,
                home,
                away,
                str(hid),
                str(aid),
                int(hg),
                int(ag),
                bool(ot),
                ledger,
                home_strength_scale=float(home_strength_scale),
                away_strength_scale=float(away_strength_scale),
            )
            if not build_game_payload:
                return {
                    "scoring_events": [],
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "overtime": bool(ot),
                    "shootout": False,
                    "stat_scope": str(stat_scope or "regular_season"),
                    "home_sog": int(result.get("home_sog", 0) or 0),
                    "away_sog": int(result.get("away_sog", 0) or 0),
                    "home_shots": int(result.get("home_shots", 0) or 0),
                    "away_shots": int(result.get("away_shots", 0) or 0),
                    "home_shot_attempts": int(result.get("home_shot_attempts", result.get("home_cf", 0)) or 0),
                    "away_shot_attempts": int(result.get("away_shot_attempts", result.get("away_cf", 0)) or 0),
                    "home_cf": int(result.get("home_cf", 0) or 0),
                    "away_cf": int(result.get("away_cf", 0) or 0),
                    "home_ff": int(result.get("home_ff", 0) or 0),
                    "away_ff": int(result.get("away_ff", 0) or 0),
                    "home_xgf": float(result.get("home_xgf", result.get("home_xg", 0)) or 0),
                    "away_xgf": float(result.get("away_xgf", result.get("away_xg", 0)) or 0),
                    "home_xg": float(result.get("home_xg", result.get("home_xgf", 0)) or 0),
                    "away_xg": float(result.get("away_xg", result.get("away_xgf", 0)) or 0),
                    "home_pp_goals": int(result.get("home_pp_goals", 0) or 0),
                    "away_pp_goals": int(result.get("away_pp_goals", 0) or 0),
                    "home_ppo": int(result.get("home_ppo", 0) or 0),
                    "away_ppo": int(result.get("away_ppo", 0) or 0),
                    "home_ppga": int(result.get("home_ppga", 0) or 0),
                    "away_ppga": int(result.get("away_ppga", 0) or 0),
                    "home_opp_ppo": int(result.get("home_opp_ppo", 0) or 0),
                    "away_opp_ppo": int(result.get("away_opp_ppo", 0) or 0),
                    "light_box": True,
                    "stat_source": "light_strength",
                }
            # Fall through with forced result for rare heavy+light callers.
        else:
            cache = getattr(self, "_pending_event_game", None)
            cache_key = (str(hid), str(aid), int(hg), int(ag), bool(ot), False)
            if isinstance(cache, dict) and cache.get("_key") == cache_key:
                result = cache["result"]
                self._merge_game_ledger(ledger, cache.get("scratch") or {})
            else:
                result = self._run_event_driven_game(
                    rng, home, away, str(hid), str(aid), ledger,
                    strength_map=strength_map,
                    home_strength_scale=home_strength_scale,
                    away_strength_scale=away_strength_scale,
                    noise_scale=noise_scale,
                    light_mode=False,
                    is_playoff=is_playoff,
                    calendar_day=int(calendar_day),
                    home_b2b=bool(home_b2b),
                    away_b2b=bool(away_b2b),
                )

        if _stats_pipeline_debug():
            issues = result.get("integrity_issues") or []
            print(f"[STAT LEDGER GAME DONE] {hid} {aid} {result.get('home_goals')} {result.get('away_goals')} issues={issues}")

        setattr(self, "_pending_event_game", None)

        if not build_game_payload:
            light_out: Dict[str, Any] = {
                "scoring_events": list(result.get("scoring_events") or []),
                "home_goals": int(result.get("home_goals", hg)),
                "away_goals": int(result.get("away_goals", ag)),
                "player_home_goals": int(result.get("player_home_goals", 0)),
                "player_away_goals": int(result.get("player_away_goals", 0)),
                "regulation_home_goals": int(result.get("regulation_home_goals", 0)),
                "regulation_away_goals": int(result.get("regulation_away_goals", 0)),
                "overtime_home_goals": int(result.get("overtime_home_goals", 0)),
                "overtime_away_goals": int(result.get("overtime_away_goals", 0)),
                "overtime": bool(result.get("overtime", ot)),
                "shootout": bool(result.get("shootout", False)),
                "stat_scope": str(stat_scope or "regular_season"),
                "home_pp_goals": int(result.get("home_pp_goals", 0)),
                "away_pp_goals": int(result.get("away_pp_goals", 0)),
                "home_ppo": int(result.get("home_ppo", 0)),
                "away_ppo": int(result.get("away_ppo", 0)),
                "home_pp_seconds": float(result.get("home_pp_seconds", 0) or 0),
                "away_pp_seconds": float(result.get("away_pp_seconds", 0) or 0),
                "home_pim": int(result.get("home_pim", 0)),
                "away_pim": int(result.get("away_pim", 0)),
                "home_shot_attempts": int(result.get("home_shot_attempts", 0)),
                "away_shot_attempts": int(result.get("away_shot_attempts", 0)),
                "home_shots": int(result.get("home_sog", 0)),
                "away_shots": int(result.get("away_sog", 0)),
                "home_sog": int(result.get("home_sog", 0)),
                "away_sog": int(result.get("away_sog", 0)),
                "home_ff": int(result.get("home_ff", 0)),
                "away_ff": int(result.get("away_ff", 0)),
                "home_xgf": round(float(result.get("home_xgf", 0) or 0), 4),
                "away_xgf": round(float(result.get("away_xgf", 0) or 0), 4),
                "light_box": True,
            }
            if result.get("pp_event_ownership"):
                light_out["pp_event_ownership"] = dict(result["pp_event_ownership"])
            return light_out

        home_dressed = result.get("home_dressed") or []
        away_dressed = result.get("away_dressed") or []

        def _skater_box_list(dressed: List[Any], tid: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for p in dressed:
                pid = _id_str(p, "id")
                if not pid:
                    continue
                row = ledger.get(pid, {})
                out.append({
                    "player_id": pid,
                    "name": row.get("name", "?"),
                    "position": row.get("position", "?"),
                    "g": int(row.get("g", 0) or 0),
                    "a": int(row.get("a", 0) or 0),
                    "sog": int(row.get("sog", 0) or 0),
                    "pim": int(row.get("pim", 0) or 0),
                    "hit": int(row.get("hit", 0) or 0),
                    "blk": int(row.get("blk", 0) or 0),
                    "toi_sec": int(row.get("toi_sec", 0) or 0),
                })
            return sorted(out, key=lambda x: (-int(x.get("g", 0)), -int(x.get("a", 0)), str(x.get("name", ""))))

        home_sk_list = _skater_box_list(home_dressed, hid)
        away_sk_list = _skater_box_list(away_dressed, aid)
        home_high = [str(e.get("scorer", "")) for e in result.get("scoring_events", []) if str(e.get("for_team_id")) == str(hid)]
        away_high = [str(e.get("scorer", "")) for e in result.get("scoring_events", []) if str(e.get("for_team_id")) == str(aid)]
        hxgf = float(result.get("home_xgf", 0) or 0)
        axgf = float(result.get("away_xgf", 0) or 0)
        home_xga = axgf
        away_xga = hxgf
        home_game_xgf_pct = hxgf / (hxgf + home_xga) if (hxgf + home_xga) > 0 else 0.5
        away_game_xgf_pct = axgf / (axgf + away_xga) if (axgf + away_xga) > 0 else 0.5
        gid = str(game_id or "").strip() or f"g{calendar_day}{hid}{aid}"[:18]
        return {
            "game_id": gid,
            "day": int(calendar_day),
            "iso": str(calendar_iso or ""),
            "home_id": hid,
            "away_id": aid,
            "home_name": str(getattr(home, "city", "") or "").strip() + " " + str(getattr(home, "name", "") or "").strip(),
            "away_name": str(getattr(away, "city", "") or "").strip() + " " + str(getattr(away, "name", "") or "").strip(),
            "home_goals": int(result.get("home_goals", hg)),
            "away_goals": int(result.get("away_goals", ag)),
            "display_home_goals": int(result.get("home_goals", hg)),
            "display_away_goals": int(result.get("away_goals", ag)),
            "overtime": bool(result.get("overtime", ot)),
            "shootout": bool(result.get("shootout", False)),
            "stat_scope": str(stat_scope or "regular_season"),
            "home_scoring": home_high[:14],
            "away_scoring": away_high[:14],
            "scoring_events": sorted(result.get("scoring_events") or [], key=lambda e: (int(e.get("period") or 0), str(e.get("clock") or ""))),
            "home_skaters": home_sk_list,
            "away_skaters": away_sk_list,
            "home_goalie": result.get("home_goalie"),
            "away_goalie": result.get("away_goalie"),
            "home_goalies": list(result.get("home_goalies") or []),
            "away_goalies": list(result.get("away_goalies") or []),
            "home_shots": int(result.get("home_sog", 0)),
            "away_shots": int(result.get("away_sog", 0)),
            "home_shot_attempts": int(result.get("home_shot_attempts", 0)),
            "away_shot_attempts": int(result.get("away_shot_attempts", 0)),
            "home_ff": int(result.get("home_ff", 0)),
            "away_ff": int(result.get("away_ff", 0)),
            "home_missed_shots": int(result.get("home_missed_shots", 0)),
            "away_missed_shots": int(result.get("away_missed_shots", 0)),
            "home_blocked_attempts_for": int(result.get("home_blocked_attempts_for", 0)),
            "away_blocked_attempts_for": int(result.get("away_blocked_attempts_for", 0)),
            "home_xgf": round(float(result.get("home_xgf", 0) or 0), 4),
            "away_xgf": round(float(result.get("away_xgf", 0) or 0), 4),
            "home_pim": int(result.get("home_pim", 0)),
            "away_pim": int(result.get("away_pim", 0)),
            "home_pp_goals": int(result.get("home_pp_goals", 0)),
            "away_pp_goals": int(result.get("away_pp_goals", 0)),
            "home_ppo": int(estimate_pp_opportunities(rng, int(result.get("home_pp_goals", 0)), int(result.get("away_pim", 0)), explicit_ppo=int(result.get("home_ppo", 0)))),
            "away_ppo": int(estimate_pp_opportunities(rng, int(result.get("away_pp_goals", 0)), int(result.get("home_pim", 0)), explicit_ppo=int(result.get("away_ppo", 0)))),
            "home_ppga": int(result.get("away_pp_goals", 0)),
            "away_ppga": int(result.get("home_pp_goals", 0)),
            "home_opp_ppo": int(result.get("away_ppo", 0)),
            "away_opp_ppo": int(result.get("home_ppo", 0)),
            "home_xgf_pct": round(float(home_game_xgf_pct), 4),
            "away_xgf_pct": round(float(away_game_xgf_pct), 4),
            "regulation_home_goals": int(result.get("regulation_home_goals", 0)),
            "regulation_away_goals": int(result.get("regulation_away_goals", 0)),
            "overtime_home_goals": int(result.get("overtime_home_goals", 0)),
            "overtime_away_goals": int(result.get("overtime_away_goals", 0)),
            "player_home_goals": int(result.get("player_home_goals", 0)),
            "player_away_goals": int(result.get("player_away_goals", 0)),
            "hockey_home_goals": int(result.get("player_home_goals", 0)),
            "hockey_away_goals": int(result.get("player_away_goals", 0)),
            "scoring_events": list(result.get("scoring_events") or []),
            "home_pp_seconds": float(result.get("home_pp_seconds", 0) or 0),
            "away_pp_seconds": float(result.get("away_pp_seconds", 0) or 0),
        }
        if result.get("pp_event_ownership"):
            out["pp_event_ownership"] = dict(result["pp_event_ownership"])
        return out

    def _season_stat_line_from_ledger_row(self, row: Dict[str, Any], season_year: int) -> Dict[str, Any]:
        """Build one player season stat dict from a game ledger row (source of truth)."""
        from app.sim_engine.gameplay.game_analytics_ledger import season_xgf_pct_from_row

        pos_u = str(row.get("position") or "").upper()
        gp = max(0, int(row.get("gp", 0) or 0))
        g = int(row.get("g", 0) or 0)
        a = int(row.get("a", 0) or 0)
        pts = g + a
        gp_div = max(1, gp)
        per_gp = pts / float(gp_div)

        if pos_u == "G":
            sa = int(row.get("shots_against", row.get("goalie_shots_against", 0)) or 0)
            sv = int(row.get("saves", 0) or 0)
            ga_ct = int(row.get("ga", row.get("goalie_ga", 0)) or 0)
            if sa <= 0 and (sv > 0 or ga_ct > 0):
                sa = max(sv + ga_ct, 0)
            sv_pct = (sv / float(sa)) if sa > 0 else 0.0
            toi_sec = int(row.get("toi_sec", 0) or 0)
            if toi_sec <= 0 and gp > 0:
                toi_sec = gp * 3600
            gaa = (float(ga_ct) * 3600.0 / float(toi_sec)) if toi_sec > 0 else 0.0
            perf_g = clamp(40.0 + 520.0 * (sv_pct - 0.88) - 1.15 * (gaa - 2.85), 22.0, 96.0)
            out: Dict[str, Any] = {
                "season": int(season_year),
                "stat_source": "game_ledger",
                "role": "starter",
                "goals": 0,
                "assists": 0,
                "points": 0,
                "ga": ga_ct,
                "goals_against": ga_ct,
                "gp": gp,
                "games_played": gp,
                "w": int(row.get("w", 0) or 0),
                "wins": int(row.get("w", 0) or 0),
                "l": int(row.get("l", 0) or 0),
                "losses": int(row.get("l", 0) or 0),
                "otl": int(row.get("otl", 0) or 0),
                "saves": sv,
                "shots_against": sa,
                "sa": sa,
                "save_pct": round(sv_pct, 4),
                "gaa": round(gaa, 3),
                "shutouts": int(row.get("so", 0) or 0),
                "so": int(row.get("so", 0) or 0),
                "toi_sec": toi_sec,
                "performance_score": round(perf_g, 2),
                "expected_score": 60.0,
                "delta": 0.0,
                "war": 0.0,
                "xgf_pct": 0.5,
            }
            for k in _SKATER_LEDGER_ANALYTICS_KEYS:
                if k in row and row[k] is not None:
                    if k in _GM_FLOAT_LEDGER_KEYS:
                        out[k] = round(float(row[k]), 4)
                    else:
                        out[k] = int(row[k])
            out["xgf_pct"] = round(season_xgf_pct_from_row(row), 3)
            return out

        perf = clamp(28.0 + 62.0 * min(1.35, per_gp / 1.15), 18.0, 98.0)
        toi_sec = int(row.get("toi_sec", 0) or 0)
        toi_pg_min = round((float(toi_sec) / float(gp_div)) / 60.0, 1) if gp > 0 else 0.0
        xgf_pct_v = round(season_xgf_pct_from_row(row), 3)
        out_sk: Dict[str, Any] = {
            "season": int(season_year),
            "stat_source": "game_ledger",
            "role": str(row.get("position") or "skater"),
            "goals": g,
            "assists": a,
            "points": pts,
            "g": g,
            "a": a,
            "pts": pts,
            "gp": gp,
            "games_played": gp,
            "sog": int(row.get("sog", 0) or 0),
            "toi_sec": toi_sec,
            "toi": toi_pg_min,
            "pim": int(row.get("pim", 0) or 0),
            "hit": int(row.get("hit", 0) or 0),
            "blk": int(row.get("blk", 0) or 0),
            "ppg": int(row.get("ppg", 0) or 0),
            "ppa": int(row.get("ppa", 0) or 0),
            "shg": int(row.get("shg", 0) or 0),
            "sha": int(row.get("sha", 0) or 0),
            "war": round(min(6.5, max(-2.0, (pts / float(max(1, gp))) * 0.22)), 2),
            "xgf_pct": xgf_pct_v,
            "performance_score": round(perf, 2),
            "expected_score": 62.0,
            "delta": round(perf - 62.0, 2),
        }
        for k in _SKATER_LEDGER_ANALYTICS_KEYS:
            if k in row and row[k] is not None:
                if k in _GM_FLOAT_LEDGER_KEYS:
                    out_sk[k] = round(float(row[k]), 4)
                else:
                    out_sk[k] = int(row[k])
        return out_sk

    def _sync_ledger_to_player_season_stats(self, ledger: Dict[str, Dict[str, Any]], season_year: int) -> None:
        """Write game-derived totals onto player.season_stats for tooling that reads player objects."""
        teams = list(getattr(self.league, "teams", None) or [])
        by_id: Dict[str, Any] = {}
        for tm in teams:
            for p in getattr(tm, "roster", None) or []:
                pid = _id_str(p, "id")
                if pid:
                    by_id[pid] = p
        for pid, row in ledger.items():
            pl = by_id.get(pid)
            if pl is None:
                continue
            if not hasattr(pl, "season_stats") or pl.season_stats is None:
                pl.season_stats = {}
            line = self._season_stat_line_from_ledger_row(dict(row), int(season_year))
            pl.season_stats[int(season_year)] = line
            if _stats_pipeline_debug():
                rname = str(
                    row.get("name")
                    or getattr(getattr(pl, "identity", None), "name", None)
                    or getattr(pl, "name", "?")
                )
                if str(row.get("position") or "").upper() == "G":
                    print(
                        f"[PLAYER SEASON SYNC] {rname} {int(season_year)} ga={line.get('ga')} gp={line.get('gp')} sv%={line.get('save_pct')}"
                    )
                else:
                    print(
                        f"[PLAYER SEASON SYNC] {rname} {int(season_year)} g={line.get('g')} a={line.get('a')} pts={line.get('pts')} gp={line.get('gp')}"
                    )

    def _validate_league_season_scoring(
        self,
        standings: StandingsTable,
        schedule: List[Any],
        ledger: Dict[str, Dict[str, Any]],
        year: int,
        counters: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"year": int(year)}
        n_games = max(1, len(schedule))
        tg = sum(int(rec.gf) for rec in standings.records.values())
        meta["league_goals_total"] = int(tg)
        meta["league_avg_goals_per_game"] = round(tg / float(n_games), 3)
        sk = [
            (
                int(v.get("g", 0)) + int(v.get("a", 0)),
                int(v.get("g", 0)),
                str(v.get("name", "?")),
            )
            for v in ledger.values()
            if str(v.get("position") or "").upper() != "G"
        ]
        sk.sort(reverse=True)
        meta["top_scorers"] = [{"name": n, "points": p, "goals": g} for p, g, n in sk[:10]]
        top_pts = int(sk[0][0]) if sk else 0
        top_goals = max((g for _, g, _ in sk), default=0)
        n_100pt = sum(1 for p, _, _ in sk if p >= 100)
        n_120pt = sum(1 for p, _, _ in sk if p >= 120)
        n_140pt = sum(1 for p, _, _ in sk if p >= 140)
        n_50g = sum(1 for _, g, _ in sk if g >= 50)
        n_60g = sum(1 for _, g, _ in sk if g >= 60)
        n_70g = sum(1 for _, g, _ in sk if g >= 70)
        meta["top_scorer_points"] = top_pts
        meta["rocket_goals"] = int(top_goals)
        meta["players_100pt"] = int(n_100pt)
        meta["players_120pt"] = int(n_120pt)
        meta["players_140pt"] = int(n_140pt)
        meta["players_50g"] = int(n_50g)
        meta["players_60g"] = int(n_60g)
        meta["players_70g"] = int(n_70g)

        gk_rows = [v for v in ledger.values() if str(v.get("position") or "").upper() == "G"]
        sa_total = sum(int(v.get("shots_against", 0) or 0) for v in gk_rows)
        sv_total = sum(int(v.get("saves", 0) or 0) for v in gk_rows)
        league_sv_pct = (sv_total / float(sa_total)) if sa_total > 0 else 0.0
        meta["league_avg_save_pct"] = round(league_sv_pct, 4)
        team_gf = [int(rec.gf) for rec in standings.records.values() if int(getattr(rec, "gp", 0) or 0) > 0]
        if team_gf:
            meta["league_avg_team_gf"] = round(sum(team_gf) / float(len(team_gf)), 1)

        warnings: List[str] = []
        structured: List[Dict[str, Any]] = []

        def _warn(severity: str, code: str, message: str, **extra: Any) -> None:
            warnings.append(f"{code}: {message}")
            w: Dict[str, Any] = {"severity": severity, "code": code, "message": message}
            w.update(extra)
            structured.append(w)

        gpg = float(meta["league_avg_goals_per_game"])
        if gpg < 5.7:
            _warn("P1", "LOW_SCORING", f"league avg goals/game {gpg:.3f} < 5.7 (modern NHL combined ~6.0-6.2).")
        elif gpg < 5.9 or gpg > 6.45:
            _warn("P2", "GPG_OUT_OF_RANGE", f"league avg goals/game {gpg:.3f} outside ideal ~5.9-6.45.")
        if gpg > 6.8:
            _warn("P1", "HIGH_SCORING", f"league avg goals/game {gpg:.3f} > 6.8 (arcade drift).")
        if top_pts < 95:
            _warn("P2", "LOW_LEADER", f"top skater points {top_pts} < 95 (ideal Art Ross ~110-130).")
        elif top_pts < 110 or top_pts > 130:
            _warn("P2", "LEADER_OUT_OF_IDEAL", f"top skater points {top_pts} outside ideal ~110-130.")
        if top_pts >= 165:
            _warn("P1", "INFLATED_LEADER", f"top skater points {top_pts} >= 165 (absurd outlier cap).")
        if n_100pt < 3:
            _warn("P2", "FEW_100PT", f"{n_100pt} players at 100+ points (ideal 5-10).")
        elif n_100pt > 14:
            _warn("P1", "TOO_MANY_100PT", f"{n_100pt} players at 100+ points (ideal 5-10).")
        if n_120pt > 5:
            _warn("P1", "TOO_MANY_120PT", f"{n_120pt} players at 120+ points (ideal 1-3).")
        if n_140pt > 2:
            _warn("P1", "TOO_MANY_140PT", f"{n_140pt} players at 140+ points (rare 0-1).")
        if n_50g < 2:
            _warn("P2", "FEW_50G", f"{n_50g} players at 50+ goals (ideal 3-8).")
        elif n_50g > 12:
            _warn("P1", "TOO_MANY_50G", f"{n_50g} players at 50+ goals (ideal 3-8).")
        if n_60g > 5:
            _warn("P1", "TOO_MANY_60G", f"{n_60g} players at 60+ goals (ideal 1-3).")
        if n_70g > 2:
            _warn("P1", "TOO_MANY_70G", f"{n_70g} players at 70+ goals (should be rare).")
        if sa_total > 0:
            if league_sv_pct < 0.895:
                _warn("P1", "LOW_LEAGUE_SV", f"league avg save % {league_sv_pct:.3f} < .895.")
            elif league_sv_pct < 0.900 or league_sv_pct > 0.907:
                _warn("P2", "SV_OUT_OF_IDEAL", f"league avg save % {league_sv_pct:.3f} outside ideal .900-.907.")
            if league_sv_pct > 0.914:
                _warn("P1", "HIGH_LEAGUE_SV", f"league avg save % {league_sv_pct:.3f} > .914.")

        # P0: any team that played zero games after a full season schedule.
        for tid, rec in standings.records.items():
            if int(getattr(rec, "gp", 0) or 0) == 0:
                _warn(
                    "P0",
                    "TEAM_ZERO_GP",
                    f"team played 0 games after regular season (falsy-id or schedule mapping bug).",
                    team_id=str(tid),
                    abbr=str(getattr(rec, "abbr", "") or ""),
                )

        # P0: schedule references team ids missing from the standings lookup.
        sched_ids: Set[str] = set()
        for sl in schedule:
            sched_ids.add(str(getattr(sl, "home_id", "")))
            sched_ids.add(str(getattr(sl, "away_id", "")))
        known_ids = {str(k) for k in standings.records.keys()}
        missing_ids = sorted(i for i in sched_ids if i and i not in known_ids)
        if missing_ids:
            _warn(
                "P0",
                "SCHEDULE_TEAM_MISMATCH",
                f"schedule references team ids not present in standings: {missing_ids[:8]}",
            )

        # P1: placeholder identities must never reach NHL-visible leader lists.
        placeholder_leaders = [n for _, _, n in sk[:30] if n.startswith("Global ") or n.startswith("GP_")]
        if placeholder_leaders:
            _warn(
                "P1",
                "PLACEHOLDER_NAME_IN_LEADERS",
                f"placeholder identities in top-30 scorers: {placeholder_leaders[:5]}",
            )

        # P2: goalie rows must exist in the ledger for exports/awards.
        n_goalies = sum(1 for v in ledger.values() if str(v.get("position") or "").upper() == "G")
        meta["goalie_rows"] = int(n_goalies)
        if n_goalies == 0 and ledger:
            _warn("P2", "MISSING_GOALIE_STATS", "no goalie rows present in the season stat ledger.")

        ct = counters or {}
        meta["trade_executions"] = int(ct.get("trade_executions", 0))
        meta["waiver_claims"] = int(ct.get("waiver_claims", 0))
        meta["major_injuries"] = int(ct.get("major_injuries", 0))
        inj_per_team_gp = float(meta["major_injuries"]) / max(1.0, float(n_games) * 0.5)
        meta["injury_event_density"] = round(inj_per_team_gp, 4)
        if meta["trade_executions"] < 1:
            _warn("P3", "LOW_TRADES", "fewer than 1 executed in-season trade (expect deadline lift).")
        if meta["major_injuries"] < 2:
            _warn("P3", "LOW_INJURIES", "very few major injury events logged (check world injury layer).")
        if meta["major_injuries"] > int(len(standings.records) * 4.5):
            _warn("P3", "HIGH_INJURIES", "major injury count unusually high vs team count.")

        meta["warnings"] = warnings
        meta["validation_warnings"] = structured
        meta["stat_source"] = "game_ledger"
        if warnings and self.debug:
            for w in warnings:
                print(f"[SimEngine season {year}] WARNING: {w}")
        return meta

    def _chemistry_game_modifier(self, team: Any, players: Optional[List[Any]] = None, situation: Optional[str] = None) -> float:
        """
        Soft chemistry probability hook (never hard-sets scores).
        Consumes canonical systems.chemistry room cache; may refresh from it.
        """
        room = {}
        try:
            room = dict(getattr(team, "_chemistry_cache", None) or {})
        except Exception:
            room = {}
        if not room:
            try:
                from app.sim_engine.systems.chemistry import calculate_team_room_chemistry  # noqa: WPS433

                room = calculate_team_room_chemistry(team, session=None)
                setattr(team, "_chemistry_cache", room)
                # Derived world chemistry summary (0–1) from canonical room score.
                setattr(team, "_world_chemistry", max(0.08, min(0.96, float(room.get("overall", 50)) / 100.0)))
            except Exception:
                room = {}
        overall = float(room.get("overall", 50.0) or 50.0)
        tension = float(room.get("tension", 25.0) or 25.0)
        buy_in = float(room.get("buy_in", 50.0) or 50.0)
        confidence = float(room.get("confidence", 50.0) or 50.0)
        chaos_res = float(room.get("chaos_resistance", 50.0) or 50.0)
        chaos = float(getattr(getattr(self, "league", None), "chaos_index", 0.35) or 0.35)

        # Optional line-level boost when canonical deployed units are present.
        line_boost = 0.0
        try:
            deployed = getattr(team, "_franchise_deployed_lineup", None)
            if isinstance(deployed, dict) and deployed.get("ok"):
                from app.sim_engine.systems.chemistry import calculate_forward_line_chemistry  # noqa: WPS433

                scores = []
                for ln in (deployed.get("forward_lines") or [])[:2]:
                    if ln and len(ln) >= 2:
                        scores.append(float(calculate_forward_line_chemistry(ln, context={"team": team}).get("chemistry", 50)))
                if scores:
                    line_boost = (sum(scores) / len(scores) - 55.0) * 0.00035
        except Exception:
            line_boost = 0.0

        m = 1.0
        m += (overall - 55.0) * 0.0008
        m += (buy_in - 50.0) * 0.00035
        m += (confidence - 50.0) * 0.00025
        m -= max(0.0, tension - 42.0) * 0.0006
        m -= max(0.0, chaos * 100.0 - chaos_res) * 0.00022
        m += line_boost

        if situation == "penalty_risk":
            m += max(0.0, tension - 48.0) * 0.0009
            m -= max(0.0, buy_in - 60.0) * 0.0004
        return max(0.94, min(1.06, m))

    def _merge_game_ledger(self, target: Dict[str, Dict[str, Any]], source: Dict[str, Dict[str, Any]]) -> None:
        for pid, row in source.items():
            if pid not in target:
                target[pid] = dict(row)
                if int(target[pid].get("gp", 0) or 0) > 82:
                    target[pid]["gp"] = 82
                continue
            dst = target[pid]
            # Already at the NHL regular-season GP ceiling — skip further credits.
            if int(dst.get("gp", 0) or 0) >= 82:
                continue
            for k, v in row.items():
                if k in _GM_FLOAT_LEDGER_KEYS:
                    dst[k] = round(float(dst.get(k, 0) or 0) + float(v or 0), 4)
                elif k in (
                    "gp", "g", "a", "pts", "sog", "pp_sog", "pim", "hit", "blk", "toi_sec",
                    "ev_toi_sec", "pp_toi_sec", "pk_toi_sec",
                    "ppg", "ppa", "shg", "sha", "ga", "w", "l", "otl", "saves", "shots_against",
                    "goalie_shots_against", "goalie_ga", "so", "missed_shots", "blocked_attempts_for",
                    "analytics_gp", "primary_assists", "secondary_assists", "xgf_pct_gp",
                ):
                    dst[k] = int(dst.get(k, 0) or 0) + int(v or 0)
                elif k not in dst or dst[k] in (None, "", 0, 0.0):
                    dst[k] = v
            if int(dst.get("gp", 0) or 0) > 82:
                dst["gp"] = 82
            if str(dst.get("position") or "").upper() != "G":
                dst["pts"] = int(dst.get("g", 0) or 0) + int(dst.get("a", 0) or 0)
    def _simulate_game_strength(
        self,
        rng: random.Random,
        home: Any,
        away: Any,
        strength_map: Dict[str, float],
        home_strength_scale: float = 1.0,
        away_strength_scale: float = 1.0,
        noise_scale: float = 1.0,
    ) -> Tuple[int, int, bool]:
        """Fast franchise/bulk score path — strength + variance (not event loop)."""
        hid = str(getattr(home, "team_id", getattr(home, "id", "H")))
        aid = str(getattr(away, "team_id", getattr(away, "id", "A")))
        s_home = max(0.15, min(0.92, float(strength_map.get(hid, 0.5)) * float(home_strength_scale)))
        s_away = max(0.15, min(0.92, float(strength_map.get(aid, 0.5)) * float(away_strength_scale)))
        win_delta = float(getattr(self, "_franchise_home_win_prob_delta", 0) or 0)
        if win_delta:
            s_home = max(0.12, min(0.94, s_home + win_delta * 1.15))
            s_away = max(0.12, min(0.94, s_away - win_delta * 1.15))

        base = 2.85
        diff = s_home - s_away
        # Target ~3.05–3.10 goals/team (combined ~6.1–6.2) after OT bump.
        home_mu = base + 0.90 * diff + 0.15
        away_mu = base - 0.90 * diff
        try:
            home_mu += float(team_scoring_pace_bias(home))
            away_mu += float(team_scoring_pace_bias(away))
        except Exception:
            pass

        home_mu = max(1.2, min(5.0, home_mu))
        away_mu = max(1.0, min(4.8, away_mu))

        sg = max(0.72, min(1.62, float(noise_scale)))
        nh = self._narrative_team_goal_sigma_multiplier(home)
        na = self._narrative_team_goal_sigma_multiplier(away)
        home_goals = max(0, int(round(rng.gauss(home_mu, 1.48 * sg * nh))))
        away_goals = max(0, int(round(rng.gauss(away_mu, 1.48 * sg * na))))

        overtime = False
        if home_goals == away_goals:
            # Slightly fewer pure regulation ties than raw discrete gauss (OTL bloat).
            if rng.random() < 0.38:
                if rng.random() < 0.52:
                    home_goals += 1
                else:
                    away_goals += 1
            else:
                overtime = True
                if rng.random() < 0.52:
                    home_goals += 1
                else:
                    away_goals += 1

        return home_goals, away_goals, overtime

    def _simulate_game(
        self,
        rng: random.Random,
        home: Any,
        away: Any,
        strength_map: Dict[str, float],
        home_strength_scale: float = 1.0,
        away_strength_scale: float = 1.0,
        noise_scale: float = 1.0,
        *,
        is_playoff: bool = False,
        calendar_day: int = 0,
        home_b2b: bool = False,
        away_b2b: bool = False,
        light_mode: bool = False,
    ) -> Tuple[int, int, bool]:
        """
        Single-game simulation.

        - light_mode=True (bulk season / Sim Regular Season): fast strength+variance path.
          Event-driven GM games are far too expensive for ~1,300 league games.
        - light_mode=False: event-driven score for detailed day/playoff presentation.
        """
        if light_mode:
            setattr(self, "_pending_event_game", None)
            return self._simulate_game_strength(
                rng,
                home,
                away,
                strength_map,
                home_strength_scale=home_strength_scale,
                away_strength_scale=away_strength_scale,
                noise_scale=noise_scale,
            )

        hid = str(getattr(home, "team_id", getattr(home, "id", "H")))
        aid = str(getattr(away, "team_id", getattr(away, "id", "A")))
        scratch: Dict[str, Dict[str, Any]] = {}
        result = self._run_event_driven_game(
            rng, home, away, hid, aid, scratch,
            strength_map=strength_map,
            home_strength_scale=home_strength_scale,
            away_strength_scale=away_strength_scale,
            noise_scale=noise_scale,
            light_mode=False,
            is_playoff=is_playoff,
            calendar_day=int(calendar_day),
            home_b2b=bool(home_b2b),
            away_b2b=bool(away_b2b),
        )
        hg = int(result.get("home_goals", 0))
        ag = int(result.get("away_goals", 0))
        ot = bool(result.get("overtime", False))
        setattr(self, "_pending_event_game", {
            "_key": (hid, aid, hg, ag, ot, False),
            "result": result,
            "scratch": scratch,
        })
        return hg, ag, ot

    def _standings_sync_team_metrics(self, standings: StandingsTable, teams: List[Any]) -> None:
        """Mirror standings rows onto team objects for waiver/trade heuristics."""
        for t in teams:
            tid = str(getattr(t, "team_id", getattr(t, "id", "")))
            rec = standings.records.get(tid)
            if rec is None:
                continue
            try:
                setattr(t, "points", int(rec.points))
                setattr(t, "point_pct", float(rec.point_pct()))
                setattr(t, "goal_diff", int(rec.goal_diff()))
            except Exception:
                pass

    def _season_daily_socio_economics(
        self,
        rng: random.Random,
        day: int,
        max_day: int,
        standings: StandingsTable,
        teams: List[Any],
        news_out: List[Dict[str, Any]],
        counters: Dict[str, int],
    ) -> None:
        """Per simulated calendar day: waivers, optional trades, roster pressure."""
        self._standings_sync_team_metrics(standings, teams)
        wire = list(getattr(self.league, "waiver_wire", None) or [])
        for tm in teams:
            tid = str(getattr(tm, "team_id", getattr(tm, "id", "")))
            rs = list(getattr(tm, "roster", None) or [])
            if len(rs) <= 23:
                continue
            rs_sorted = sorted(rs, key=lambda p: float(self._gm_ovr_bonus(p)))
            while len(rs_sorted) > 23 and rng.random() < 0.65:
                cand = rs_sorted[0]
                if self._injury_sidelined(cand):
                    rs_sorted.pop(0)
                    continue
                wire.append({"player": cand, "from_team": tid})
                try:
                    rs_sorted.remove(cand)
                except ValueError:
                    break
            tm.roster = rs_sorted
        setattr(self.league, "waiver_wire", wire)

        for ln in process_waivers(self.league) or []:
            counters["waiver_claims"] = int(counters.get("waiver_claims", 0)) + 1
            news_out.append(
                {
                    "type": "waiver",
                    "date": int(day),
                    "headline": str(ln),
                    "team": "",
                    "players": [],
                    "priority": "LOW",
                }
            )

        md = max(40, int(max(120, max_day) * 0.56))
        deadline_window = max(20.0, float(max_day) * 0.2)
        deadline_day = int(md + deadline_window)
        deadline_phase = max(0.0, min(1.0, (float(day) - float(md)) / deadline_window))
        # Hard stop: no standard NHL trades after the trade deadline calendar day.
        post_deadline = int(day) > int(deadline_day)
        trade_prob = 0.030 + 0.26 * deadline_phase
        # League-mean trade-frequency preference slightly modulates market cadence.
        try:
            profiles = dict(getattr(self.league, "cpu_franchise_profiles", None) or {})
            freqs = [
                float((p or {}).get("ideology", {}).get("trade_frequency_preference", 0.5) or 0.5)
                for p in profiles.values()
                if isinstance(p, dict)
            ]
            if freqs:
                trade_prob += (sum(freqs) / len(freqs) - 0.5) * 0.04
        except Exception:
            pass
        market_state = getattr(self.league, "cpu_market_runtime", None)
        if not isinstance(market_state, dict):
            market_state = {}
            setattr(self.league, "cpu_market_runtime", market_state)
        last_day = int(market_state.get("last_day", -1) or -1)
        if last_day > int(day):
            market_state = {}
            setattr(self.league, "cpu_market_runtime", market_state)
        market_state["last_day"] = int(day)
        market_state["deadline_day"] = int(deadline_day)
        market_state["post_deadline"] = bool(post_deadline)
        seen_ids = market_state.get("seen_trade_ids")
        if not isinstance(seen_ids, set):
            seen_ids = set(seen_ids or [])
            market_state["seen_trade_ids"] = seen_ids
        season_cpu_trades = int(market_state.get("season_cpu_trades", 0) or 0)
        history = list(getattr(self.league, "trade_history", None) or [])
        scan_from = int(market_state.get("trade_history_scan_idx", 0) or 0)
        if scan_from < 0 or scan_from > len(history):
            scan_from = 0
        for row in history[scan_from:]:
            if not isinstance(row, dict):
                continue
            if bool(row.get("user_involved")):
                continue
            tid = str(row.get("trade_id") or "")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            season_cpu_trades += 1
        market_state["trade_history_scan_idx"] = len(history)
        market_state["season_cpu_trades"] = int(season_cpu_trades)

        day_ratio = max(0.0, min(1.0, float(day) / max(1.0, float(max_day))))
        # Quieter league cadence — prior 48–78 season target felt spammy.
        if "seasonal_target" not in market_state:
            market_state["seasonal_target"] = int(38 + round((rng.random() - 0.5) * 10))
        seasonal_target = int(market_state.get("seasonal_target", 38) or 38)
        seasonal_target = max(30, min(48, seasonal_target))
        if day_ratio < 0.25:
            expected_curve = 0.14
        elif day_ratio < 0.5:
            expected_curve = 0.34
        elif day_ratio < 0.75:
            expected_curve = 0.62
        elif day_ratio < 0.9:
            expected_curve = 0.86
        else:
            expected_curve = 0.98
        expected_by_now = max(0, int(round(float(seasonal_target) * expected_curve)))
        trade_deficit = max(0, expected_by_now - season_cpu_trades)
        if trade_deficit >= 3:
            trade_prob += min(0.10, 0.015 * trade_deficit)
        if deadline_phase > 0.7:
            trade_prob += 0.05
        trade_prob = max(0.015, min(0.72, trade_prob))
        if getattr(self.league, "transcendent_active", False):
            tank_sellers = sum(1 for tm in teams if int(getattr(tm, "_franchise_tank_pressure", 0) or 0) >= 50)
            if tank_sellers:
                trade_prob += min(0.06, 0.010 * tank_sellers)
        max_exec = 1
        if deadline_phase > 0.50 or trade_deficit >= 4:
            max_exec += 1
        if deadline_phase > 0.75 or trade_deficit >= 6:
            max_exec += 1
        if deadline_phase > 0.90 or trade_deficit >= 8:
            max_exec += 1
        max_exec = max(1, min(4, max_exec))
        forced_market_check = (not post_deadline) and trade_deficit >= 4 and (int(day) % 3 == 0)
        if (not post_deadline) and (rng.random() < trade_prob or forced_market_check):
            tr = evaluate_trade_market(
                self.league,
                max_executions=max_exec,
                calendar_cursor=int(day),
                regular_season_last_index=int(max_day),
            )
            counters["trade_executions"] = int(counters.get("trade_executions", 0)) + len(tr)
            for t in tr:
                news_out.append(
                    {
                        "type": "trade",
                        "date": int(day),
                        "headline": str(t.get("headline") or "Trade completed"),
                        "team": str(t.get("to_team_id") or ""),
                        "from_team_id": str(t.get("from_team_id") or ""),
                        "players": list(t.get("outgoing") or []) + list(t.get("incoming") or []),
                        "trade_id": str(t.get("trade_id") or ""),
                        "execution": dict(t.get("execution") or {}),
                        "trade_category": str(t.get("trade_category") or ""),
                        "importance": str(t.get("importance") or "standard"),
                        "reason_codes": list(t.get("reason_codes") or []),
                        "reason_text": str(t.get("reason_text") or ""),
                        "priority": "HIGH",
                    }
                )
        elif post_deadline:
            counters["post_deadline_blocked"] = int(counters.get("post_deadline_blocked", 0)) + 1

        rm = RosterManager()
        tbl = standings.league_table()
        n_t = max(1, len(tbl))
        rank_by_tid = {str(getattr(x, "team_id", "") or ""): i for i, x in enumerate(tbl)}
        for tm in teams:
            tid = str(getattr(tm, "team_id", getattr(tm, "id", "")))
            rec = standings.records.get(tid)
            rank = int(rank_by_tid.get(tid, n_t))
            inj_ct = sum(1 for p in self._gm_skaters(tm) if self._injury_sidelined(p))
            if inj_ct < 2 and rank <= int(0.42 * n_t):
                continue
            if rng.random() > 0.22:
                continue
            try:
                logs = rm.manage(tm, self.league)
            except Exception:
                logs = []
            for ln in logs[:2]:
                u = str(ln).upper()
                if "PROMOTION" in u or "CALL" in u or "RECALL" in u:
                    news_out.append(
                        {
                            "type": "callup",
                            "date": int(day),
                            "headline": str(ln),
                            "team": tid,
                            "players": [],
                            "priority": "LOW",
                        }
                    )

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
        Each calendar day runs waivers, a light trade tick (higher near deadline), and roster
        pressure (call-ups) before that day’s games. Macro UFA volume may still be generated
        in run_sim.simulate_universe_year depending on mode.
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
        self._refresh_league_scoring_elite_set(teams)
        self._roll_historic_scoring_seasons(teams, r, int(year))
        strength_map = self._build_strength_map(teams)

        # id -> team mapping for quick lookup
        # Explicit None checks: team_id=0 is a valid id (Boston) and must not be
        # remapped to T00 while the schedule references "0" (P0 fix: skipped games).
        team_by_id: Dict[str, Any] = {}
        team_ids: List[str] = []
        for idx, t in enumerate(teams):
            tid = getattr(t, "team_id", None)
            if tid is None:
                tid = getattr(t, "id", None)
            if tid is None:
                tid = f"T{idx:02d}"
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
        stat_ledger: Dict[str, Dict[str, Any]] = {}
        news_events: List[Dict[str, Any]] = []
        season_counters: Dict[str, int] = {"trade_executions": 0, "waiver_claims": 0, "major_injuries": 0}
        milestone_seen: Set[Tuple[str, str, int]] = set()
        run_hist: Dict[str, List[str]] = {}
        max_cal = max((int(s.day) for s in schedule), default=0)

        by_day: Dict[int, List[Any]] = {}
        for sl in schedule:
            by_day.setdefault(int(sl.day), []).append(sl)

        for day in sorted(by_day.keys()):
            self._season_daily_socio_economics(r, day, max_cal, standings, teams, news_events, season_counters)
            for slot in by_day[day]:
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
                    h_scale = max(0.88, min(1.12, h_scale))
                    a_scale = max(0.88, min(1.12, a_scale))
                    h_scale *= self._roster_injury_depth_penalty(home)
                    a_scale *= self._roster_injury_depth_penalty(away)

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
                        ev = world_injuries.maybe_injure_roster_subset(tm, r, chaos_index, max_checks=8)
                        for label, tier, games, _pid in ev:
                            if tier == "major":
                                tid_inj = _id_str(tm, "team_id", "id")
                                injury_log_major.append(
                                    {
                                        "player": label,
                                        "tier": tier,
                                        "games": games,
                                        "team_id": tid_inj,
                                    }
                                )
                                season_counters["major_injuries"] = int(season_counters.get("major_injuries", 0)) + 1
                                news_events.append(
                                    {
                                        "type": "injury",
                                        "date": int(day),
                                        "headline": f"{label} ({tier}, {games}g) — {tid_inj}",
                                        "team": tid_inj,
                                        "players": [str(label)],
                                        "priority": "MEDIUM",
                                    }
                                )
                else:
                    _, nh = self._identity_runner_strength_noise_factors(home)
                    _, na = self._identity_runner_strength_noise_factors(away)
                    id_noise = 0.5 * (nh + na)
                    h_inj = self._roster_injury_depth_penalty(home)
                    a_inj = self._roster_injury_depth_penalty(away)
                    hg, ag, ot = self._simulate_game(
                        r, home, away, strength_map,
                        home_strength_scale=h_inj,
                        away_strength_scale=a_inj,
                        noise_scale=id_noise,
                    )

                game_box = self.accumulate_unified_game_stats(
                    r, home, away, hid, aid, int(hg), int(ag), bool(ot), stat_ledger
                )
                standings.record_game(
                    slot.home_id,
                    slot.away_id,
                    hg,
                    ag,
                    overtime=ot,
                    shootout=bool((game_box or {}).get("shootout", False)),
                    stats_home_goals=int((game_box or {}).get("player_home_goals", hg) or 0),
                    stats_away_goals=int((game_box or {}).get("player_away_goals", ag) or 0),
                )

                for tm_m, tid_m in ((home, hid), (away, aid)):
                    for p in self._gm_skaters(tm_m):
                        pid = _id_str(p, "id")
                        if not pid:
                            continue
                        row = stat_ledger.get(pid)
                        if not row:
                            continue
                        pts = int(row.get("g", 0) or 0) + int(row.get("a", 0) or 0)
                        gf = int(row.get("g", 0) or 0)
                        nm = str(row.get("name") or "?")
                        for th, kind, label in (
                            (50, "pts", "50 points"),
                            (80, "pts", "80 points"),
                            (100, "pts", "100 points"),
                            (30, "g", "30 goals"),
                            (50, "g", "50 goals"),
                        ):
                            val = pts if kind == "pts" else gf
                            if val < th:
                                continue
                            key = (pid, kind, th)
                            if key in milestone_seen:
                                continue
                            milestone_seen.add(key)
                            if r.random() > 0.65:
                                continue
                            news_events.append(
                                {
                                    "type": "milestone",
                                    "date": int(day),
                                    "headline": f"{nm} reaches {label}",
                                    "team": str(tid_m),
                                    "players": [nm],
                                    "priority": "MEDIUM",
                                }
                            )
                            break

                for tid, won in ((hid, hg > ag), (aid, ag > hg)):
                    hist = run_hist.setdefault(tid, [])
                    letter = "W" if won else "L"
                    hist.append(letter)
                    hist[:] = hist[-14:]
                    streak = 1
                    for j in range(len(hist) - 2, -1, -1):
                        if hist[j] == letter:
                            streak += 1
                        else:
                            break
                    if streak >= 5 and r.random() < 0.55:
                        news_events.append(
                            {
                                "type": "streak",
                                "date": int(day),
                                "headline": f"{tid} riding a {streak}-game {letter} streak",
                                "team": tid,
                                "players": [],
                                "priority": "LOW",
                            }
                        )

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

        self._last_league_sim_calendar_year = int(year)
        self._last_league_season_stat_ledger = dict(stat_ledger)
        self._sync_ledger_to_player_season_stats(stat_ledger, int(year))
        sim_meta = self._validate_league_season_scoring(
            standings, schedule, stat_ledger, int(year), season_counters
        )
        self._last_league_season_validation = sim_meta
        for w in sim_meta.get("warnings") or []:
            print(f"[SimEngine] WARNING [{year}]: {w}")
        sk_export = [v for v in stat_ledger.values() if str(v.get("position", "")).upper() != "G"]
        sk_export.sort(key=lambda z: -(int(z.get("g", 0)) + int(z.get("a", 0))))
        player_stat_export = sk_export[:520]
        # Goalie export rows: derive SV%/GAA honestly from accumulated totals;
        # fields that cannot be computed are left as None rather than faked.
        g_export: List[Dict[str, Any]] = []
        for v in stat_ledger.values():
            if str(v.get("position", "")).upper() != "G":
                continue
            row = dict(v)
            sa = int(row.get("shots_against", 0) or 0)
            sv = int(row.get("saves", 0) or 0)
            gp_g = int(row.get("gp", 0) or 0)
            row["save_pct"] = round(sv / sa, 4) if sa > 0 else None
            toi_sec = int(row.get("toi_sec", 0) or 0)
            if toi_sec <= 0 and gp_g > 0:
                toi_sec = gp_g * 3600
                row["toi_sec"] = toi_sec
            row["gaa"] = round(float(row.get("ga", 0) or 0) * 3600.0 / toi_sec, 3) if toi_sec > 0 else None
            row["shutouts"] = int(row.get("so", 0) or 0)
            row["stat_source"] = "game_ledger"
            g_export.append(row)
        g_export.sort(key=lambda z: (-int(z.get("w", 0) or 0), -(float(z.get("save_pct") or 0.0))))
        goalie_stat_export = g_export[:120]
        award_stat_rows = list(stat_ledger.values())
        awards = compute_awards(standings, playoff_result, teams, player_season_stats=award_stat_rows)

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

        _pri_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

        def _news_sort_key(e: Dict[str, Any]) -> Tuple[int, int, int]:
            pr = str(e.get("priority") or "LOW").upper()
            pr_i = int(_pri_rank.get(pr, 4))
            ty = str(e.get("type") or "")
            ty_i = {"trade": 0, "injury": 1, "milestone": 2, "waiver": 3, "streak": 4, "callup": 5}.get(ty, 9)
            return (pr_i, ty_i, int(e.get("date") or 0))

        news_sorted = sorted(news_events, key=_news_sort_key)

        result = LeagueSeasonResult(
            year=year,
            schedule=schedule,
            standings=standings,
            playoff_result=playoff_result,
            awards=awards,
            player_season_stats=list(player_stat_export),
            goalie_season_stats=list(goalie_stat_export),
            simulation_meta=dict(sim_meta),
            news_events=list(news_sorted),
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
        *,
        franchise_tick: bool = False,
    ) -> Dict[str, Any]:
        """
        Character-driven player storylines plus systemic consequences (team/league ripples).
        Returns {"player_storylines": [...], "narrative_consequences": [...], "league_delta": {...}}.

        franchise_tick: when True (franchise day advance), use small per-day caps and higher
        per-player try rate so a few league-wide beats surface without a full-season batch.
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
        if franchise_tick:
            sk = int(year)
            maj_used = _narr_season_major_count(league, sk)
            major_cap = 0
            if maj_used < _MAJOR_STORYLINE_SEASON_CAP and r.random() < max(_FRANCHISE_MAJOR_DAY_GATE, 0.22):
                major_cap = 1
            mid_cap = r.randint(1, 2)
            minor_cap = r.randint(1, 3)
        else:
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
            if franchise_tick and _narr_season_major_count(league, int(year)) >= _MAJOR_STORYLINE_SEASON_CAP:
                rw_m = 0.0
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

        uid_fr = str(getattr(league, "_franchise_user_team_id", "") or "")

        for team in teams:
            roster = list(getattr(team, "roster", None) or [])
            tid = _id_str(team, "team_id", "id")
            # Equal rules: user team is eligible for franchise storyline ticks
            # (including legal/conduct). Fake-text spam is filtered later in fanout.
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
                    p_try = 0.012 if char < 40 else 0.008 if char < 70 else 0.005
                    if abs(perf_delta) >= 0.035:
                        p_try *= 1.08
                    if franchise_tick:
                        p_try *= 1.25
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
                                    leg_cap_hit = _narr_season_legal_count(league, int(year)) >= _LEGAL_MAJOR_SEASON_CAP
                                    # Real-player safeguard: imported NHL names get far lower legal_crime weight
                                    # (still allowed so user-team equal rules remain), not zeroed out.
                                    is_generated = bool(
                                        str(getattr(player, "_generated_profile", "") or "").strip()
                                    )
                                    if (
                                        not leg_cap_hit
                                        and _legal_crime_roll_passes(r, char, franchise_tick=franchise_tick)
                                    ):
                                        weights[li] *= _legal_pool_weight_mult(char)
                                        if not is_generated:
                                            weights[li] *= 0.22
                                    else:
                                        weights[li] = 0.0
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
                        # Real-player crime safeguard: sanitize invented crime copy for imported NHL names.
                        if str(picked.get("pool") or "") == "legal_crime" and not bool(
                            str(getattr(player, "_generated_profile", "") or "").strip()
                        ):
                            picked = dict(picked)
                            picked["text"] = (
                                "League opens an off-ice conduct investigation after public reports. "
                                "Details remain unconfirmed; availability decisions are organizational."
                            )
                            if str(picked.get("legal_severity") or "").lower() == "major":
                                picked["legal_severity"] = "moderate"
                                if eff_tier == "major":
                                    eff_tier = "mid"
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
                            _bump_narr_season_major(
                                league,
                                int(year),
                                legal=str(picked.get("pool") or "") == "legal_crime"
                                or str(picked.get("legal_severity") or "").lower() == "major",
                            )
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
                        log_rec["team_id"] = tid
                        log_rec["player_id"] = str(getattr(player, "id", "") or "")
                        log_rec["storyline_text"] = event_obj.get("storyline", "")
                        log_rec["event_type"] = etype
                        log_rec["pool"] = str(picked.get("pool") or "")
                        log_rec["legal_severity"] = str(picked.get("legal_severity") or "")
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

        cons_cap = 3 if franchise_tick else 82
        if franchise_tick and uid_fr:
            consequence_log = [
                row for row in consequence_log
                if str(row.get("team_id") or "") != uid_fr
            ]
        if len(consequence_log) > cons_cap:
            consequence_log.sort(
                key=lambda x: (
                    0 if str(x.get("arc_tier")) == "major" else 1 if str(x.get("arc_tier")) == "mid" else 2,
                    str(x.get("team_name") or ""),
                )
            )
            bal["suppressed_events"] += len(consequence_log) - cons_cap
            consequence_log = consequence_log[:cons_cap]

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


# ---------------------------------------------------------------------------
# Franchise Trade Hub — fan attachment / window tolerance helpers
# ---------------------------------------------------------------------------

def _clamp_fan_score(value: float, low: int = 0, high: int = 100) -> int:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 50.0
    return int(max(low, min(high, round(v))))


def _franchise_player_ovr99(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0)
    except Exception:
        v = 0.0
    return v * 99.0 if v <= 1.5 else v


def _franchise_team_window_fan_tolerance(team: Any) -> Dict[str, float]:
    """How forgiving fans are when selling/buying by competitive window."""
    direction = str(getattr(team, "gm_window", getattr(team, "window", "unknown")) or "unknown").lower()
    if direction in ("rebuild", "rebuilding", "emerging", "seller", "tank", "tanking"):
        return {
            "sell_veteran_relief": 0.55,
            "sell_youth_penalty": 1.35,
            "sell_star_penalty": 1.05,
            "buy_prospect_bonus": 1.25,
            "buy_star_penalty": 0.85,
        }
    if direction in ("contender", "playoff", "buyer"):
        return {
            "sell_veteran_relief": 0.75,
            "sell_youth_penalty": 1.55,
            "sell_star_penalty": 1.45,
            "buy_prospect_bonus": 0.85,
            "buy_star_bonus": 1.35,
        }
    return {
        "sell_veteran_relief": 0.9,
        "sell_youth_penalty": 1.25,
        "sell_star_penalty": 1.2,
        "buy_prospect_bonus": 1.0,
        "buy_star_bonus": 1.1,
    }


def _franchise_player_fan_attachment(player: Any, team: Any = None, *, session_stats: Optional[Dict[str, Any]] = None) -> float:
    """0–100 fan attachment score for a roster player."""
    if player is None:
        return 20.0
    ovr = _franchise_player_ovr99(player)
    ident = getattr(player, "identity", None)
    age = int(getattr(ident, "age", getattr(player, "age", 26)) or 26)
    score = 18.0 + min(38.0, ovr * 0.38)
    if ovr >= 88:
        score += 14.0
    elif ovr >= 84:
        score += 9.0
    elif ovr >= 80:
        score += 5.0
    if age <= 24 and ovr >= 78:
        score += 8.0
    if age >= 30 and ovr >= 82:
        score += 4.0
    ratings = getattr(player, "ratings", None) or {}
    try:
        leadership = float(ratings.get("per_leadership", 0) or 0)
        if leadership >= 85:
            score += 6.0
        elif leadership >= 78:
            score += 3.0
    except (TypeError, ValueError):
        pass
    if bool(getattr(player, "is_captain", False)) or str(getattr(player, "captaincy", "") or "").upper() in ("C", "CAPTAIN"):
        score += 16.0
    elif str(getattr(player, "captaincy", "") or "").upper() in ("A", "ALT", "ALTERNATE"):
        score += 8.0
    drafted = str(getattr(player, "drafted_by_team_id", "") or getattr(player, "origin_team_id", "") or "")
    tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "") if team else ""
    if drafted and tid and drafted == tid:
        score += 5.0
    pst = getattr(player, "_franchise_storyline_state", None) or {}
    if bool(pst.get("was_recently_shopped")):
        score -= 4.0
    if session_stats:
        gp = int(session_stats.get("gp", 0) or 0)
        pts = int(session_stats.get("pts", 0) or 0)
        if gp >= 20 and pts / max(1, gp) >= 0.75:
            score += 4.0
    return float(max(5.0, min(100.0, score)))

