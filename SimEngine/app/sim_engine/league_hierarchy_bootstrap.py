"""
Full organizational depth: NHL org AHL/ECHL, free agents, overseas tracking,
and junior / college / European development leagues.

Bootstrapped when a franchise session starts (see backend franchise_sim).
Out-of-league players receive light per-day development ticks during advance_day.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.sim_engine.entities.player import Player, Position, Shoots, IdentityBio, BackstoryUpbringing, BackstoryType, UpbringingType, SupportLevel, PressureLevel, DevResources, clamp_rating, assign_skater_archetype, enforce_minimum_player_ovr, get_ovr_floor_for_pool
from app.sim_engine.generation.name_generator import generate_human_identity
from app.sim_engine.generation.prospect_body import (
    apply_body_tradeoffs_to_ratings,
    generate_position_height_cm,
    generate_realistic_weight_kg,
)
from app.sim_engine.generation.prospect_identity import spawn_youth_baseline_profile
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

TRANSCENDENT_CLASS_PROB = 0.0001
_BUILD_ROLE_SHAPED_RATINGS = None
_ENSURE_PLAYER_CHEMISTRY_PROFILE = None
_ENSURE_PLAYER_HEADSHOT = None


def _build_role_shaped_ratings_cached():
    global _BUILD_ROLE_SHAPED_RATINGS
    if _BUILD_ROLE_SHAPED_RATINGS is None:
        from app.sim_engine.engine import build_role_shaped_ratings

        _BUILD_ROLE_SHAPED_RATINGS = build_role_shaped_ratings
    return _BUILD_ROLE_SHAPED_RATINGS


def _ensure_player_chemistry_profile_cached():
    global _ENSURE_PLAYER_CHEMISTRY_PROFILE
    if _ENSURE_PLAYER_CHEMISTRY_PROFILE is None:
        try:
            from app.sim_engine.systems.chemistry import ensure_player_chemistry_profile

            _ENSURE_PLAYER_CHEMISTRY_PROFILE = ensure_player_chemistry_profile
        except Exception:
            _ENSURE_PLAYER_CHEMISTRY_PROFILE = False
    return _ENSURE_PLAYER_CHEMISTRY_PROFILE


def _ensure_player_headshot_cached():
    global _ENSURE_PLAYER_HEADSHOT
    if _ENSURE_PLAYER_HEADSHOT is None:
        try:
            from app.sim_engine.generation.player_headshots import ensure_player_headshot

            _ENSURE_PLAYER_HEADSHOT = ensure_player_headshot
        except Exception:
            _ENSURE_PLAYER_HEADSHOT = False
    return _ENSURE_PLAYER_HEADSHOT


def _pool_context_for_level(level: str, league_code: str = "") -> str:
    lv = str(level or "").strip().lower()
    code = str(league_code or "").strip().upper()
    if lv == "ahl":
        return "ahl"
    if lv == "echl":
        return "echl"
    if lv == "ufa":
        return "overseas" if code else "ufa"
    if lv == "junior":
        if code == "NCAA":
            return "college"
        if code.startswith("EU_"):
            return "european_junior"
        return "junior"
    return "junior"


def _team_label(team: Any) -> str:
    city = str(getattr(team, "city", "") or "").strip()
    name = str(getattr(team, "name", "") or "").strip()
    if city and name:
        return f"{city} {name}"
    return str(getattr(team, "team_id", ""))


def _set_assignment(player: Any, **meta: Any) -> None:
    cur = getattr(player, "_franchise_assignment", None)
    d: Dict[str, Any] = dict(cur) if isinstance(cur, dict) else {}
    d.update(meta)
    setattr(player, "_franchise_assignment", d)


_SPAWN_AS_OF_YEAR = 0

# CHL / USHL / Euro junior rosters are mostly 16–18. Uniform 17–20 (and a
# hardcoded 2025 birth year) made the NHL draft board look like an overager
# combine. Weights are NHL-entry-draft shaped: first-year kids dominate.
# Draft-eligible slice (18 / 19 / 20+) is about 80 / 15 / 5. Younger ages stay
# in the pool so next year's class already exists.
_CHL_AGE_TABLE = ((16, 17, 18, 19, 20), (12, 28, 48, 9, 3))
_USHL_AGE_TABLE = ((16, 17, 18, 19), (14, 34, 44, 8))
_NCAA_AGE_TABLE = ((18, 19, 20, 21), (46, 30, 16, 8))
_EU_J_AGE_TABLE = ((16, 17, 18, 19, 20), (10, 30, 48, 9, 3))


def set_spawn_as_of_year(year: Optional[int]) -> None:
    global _SPAWN_AS_OF_YEAR
    try:
        y = int(year or 0)
    except Exception:
        y = 0
    _SPAWN_AS_OF_YEAR = y if y >= 2000 else 0


def spawn_as_of_year(explicit: Optional[int] = None) -> int:
    try:
        y = int(explicit or 0)
    except Exception:
        y = 0
    if y >= 2000:
        return y
    if _SPAWN_AS_OF_YEAR >= 2000:
        return _SPAWN_AS_OF_YEAR
    try:
        from datetime import date

        today = date.today()
        return int(today.year) if int(today.month) >= 7 else int(today.year) - 1
    except Exception:
        return 2026


def _pick_spawn_age(
    rng: random.Random,
    age_lo: int,
    age_hi: int,
    *,
    pool_context: str = "",
    league_code: str = "",
) -> int:
    lo = int(age_lo)
    hi = int(age_hi)
    if hi < lo:
        lo, hi = hi, lo
    code = str(league_code or "").upper()
    ctx = str(pool_context or "").lower()
    table = None
    if code.startswith("CHL_") or code in ("OHL", "WHL", "QMJHL"):
        table = _CHL_AGE_TABLE
    elif code == "USHL":
        table = _USHL_AGE_TABLE
    elif code == "NCAA":
        table = _NCAA_AGE_TABLE
    elif code.startswith("EU_J"):
        table = _EU_J_AGE_TABLE
    elif ctx in ("junior", "european_junior", "college"):
        table = _CHL_AGE_TABLE if ctx != "college" else _NCAA_AGE_TABLE
    if table:
        ages, weights = table
        filtered = [(a, w) for a, w in zip(ages, weights) if lo <= int(a) <= hi]
        if filtered:
            picks, wts = zip(*filtered)
            return int(rng.choices(list(picks), weights=list(wts), k=1)[0])
    return int(rng.randint(lo, hi))


def _ident_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    try:
        return int(getattr(ident, "age", 99) or 99) if ident is not None else 99
    except Exception:
        return 99


def _upcoming_draft_year(league: Any) -> int:
    y = int(
        getattr(league, "season_start_year", 0)
        or getattr(league, "season_year", 0)
        or getattr(league, "current_season", 0)
        or 0
    )
    if y >= 2000:
        return y + 1
    return int(spawn_as_of_year()) + 1


def _player_sept15_age(player: Any, draft_year: int) -> int:
    try:
        from services.season_carryover import _age_on_sept15

        sept = _age_on_sept15(player, int(draft_year))
        if sept is not None:
            return int(sept)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return _ident_age(player)


def _is_first_year_draft_eligible(player: Any, draft_year: int) -> bool:
    return _player_sept15_age(player, int(draft_year)) == 18


def _pipeline_already_shaped(player: Any) -> bool:
    return getattr(player, "_pipeline_shaped_year", None) is not None


def _stamp_pipeline_shaped(player: Any, draft_year: int) -> None:
    try:
        setattr(player, "_pipeline_shaped_year", int(draft_year))
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _mark_overage_juniors_off_open_board(league: Any, rng: random.Random, season_year: int) -> int:
    """At bootstrap, most 19+ juniors already belong to an NHL org (or are passed over)."""
    teams = list(getattr(league, "teams", None) or [])
    if not teams:
        return 0
    marked = 0
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "").upper()
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "real_nhl_import", False):
                    continue
                if bool(getattr(p, "drafted", False)):
                    continue
                if str(
                    getattr(p, "nhl_rights_team_id", None)
                    or getattr(p, "rights_team_id", None)
                    or getattr(p, "drafted_by", None)
                    or ""
                ).strip():
                    continue
                age = _ident_age(p)
                if age < 19:
                    continue
                rights_p = 0.78
                if code == "NCAA":
                    rights_p = 0.62 if age == 19 else 0.86
                elif code.startswith("CHL_") or code in ("OHL", "WHL", "QMJHL"):
                    rights_p = 0.84 if age >= 20 else 0.72
                if rng.random() < rights_p:
                    org = rng.choice(teams)
                    tid = str(getattr(org, "team_id", "") or "")
                    try:
                        p.nhl_rights_team_id = tid
                        p.rights_team_id = tid
                        p.drafted = True
                    except Exception:
                        _swallowed_log.debug("suppressed exception", exc_info=True)
                else:
                    try:
                        setattr(p, "passed_over_draft", True)
                    except Exception:
                        _swallowed_log.debug("suppressed exception", exc_info=True)
                marked += 1
    return marked


def reanchor_generated_junior_dobs(league: Any, as_of_year: int) -> int:
    """Fix juniors spawned with the old `birth_year = 2025 - age` formula.

    Year-one 2026 franchises displayed those kids as 19–20. One-shot per league.
    Skips real NHL imports and anyone already stamped with `_dob_anchor_year`.
    """
    if league is None or bool(getattr(league, "_junior_dobs_reanchored", False)):
        return 0
    try:
        year = int(as_of_year or 0)
    except Exception:
        year = 0
    if year < 2000:
        return 0
    fixed = 0
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "real_nhl_import", False):
                    continue
                if getattr(p, "_dob_anchor_year", None):
                    continue
                ident = getattr(p, "identity", None)
                if ident is None:
                    continue
                try:
                    birth_year = int(getattr(ident, "birth_year", 0) or 0)
                except Exception:
                    continue
                spawn_age = 2025 - birth_year
                if spawn_age < 15 or spawn_age > 24:
                    continue
                ident.birth_year = year - spawn_age
                ident.age = spawn_age
                try:
                    ident.birth_month = 7
                    ident.birth_day = 1
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
                try:
                    p.age = spawn_age
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
                setattr(p, "_dob_anchor_year", year)
                fixed += 1
    try:
        setattr(league, "_junior_dobs_reanchored", True)
        setattr(league, "season_start_year", year)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return fixed


def _spawn_player(
    rng: random.Random,
    *,
    pos: Position,
    ovr_lo: float,
    ovr_hi: float,
    age_lo: int,
    age_hi: int,
    used_names: set,
    league_players: List[Any],
    pool_context: str = "junior",
    nationality: Optional[str] = None,
    league_code: str = "",
    as_of_year: Optional[int] = None,
) -> Player:
    target_ovr = ovr_lo + rng.uniform(0, max(1e-6, ovr_hi - ovr_lo))
    target_ovr = max(0.30, min(0.92, target_ovr))
    build_role_shaped_ratings = _build_role_shaped_ratings_cached()
    junior_pool = str(pool_context or "").lower() in (
        "junior",
        "chl",
        "ncaa",
        "euro_jr",
        "prep",
        "development",
        "juniors",
    )
    if junior_pool:
        from app.sim_engine.engine import pop_generation_profile
        from app.sim_engine.entities.player import archetype_from_generation_profile

        youth_profile = spawn_youth_baseline_profile(pos, rng)
        ratings = build_role_shaped_ratings(
            position=pos, target_ovr=target_ovr, rng=rng, profile=youth_profile
        )
        gen_profile = pop_generation_profile(ratings)
        archetype = archetype_from_generation_profile(gen_profile, pos) or (
            "TWO_WAY" if str(getattr(pos, "value", pos)).upper() == "D" else "TWO_WAY_F"
        )
    else:
        archetype = assign_skater_archetype(pos, rng)
        ratings = build_role_shaped_ratings(position=pos, target_ovr=target_ovr, rng=rng)
    age = _pick_spawn_age(
        rng, age_lo, age_hi, pool_context=pool_context, league_code=league_code
    )
    year = spawn_as_of_year(as_of_year)
    # July 1 birthday: listed age stays stable from camp through the June draft.
    birth_year = int(year) - int(age)
    seed = rng.randint(1, 2_000_000_000)
    ident = generate_human_identity(rng, nationality=nationality) if nationality else generate_human_identity(rng)
    for _ in range(6):
        nm = str(getattr(ident, "full_name", "Unknown"))
        if nm not in used_names:
            used_names.add(nm)
            break
        # Regenerate on a duplicate name WITHOUT discarding the forced nationality,
        # otherwise a duplicate Canadian/Swedish/Russian silently becomes a random
        # world nationality (which also breaks league-fit validation).
        ident = generate_human_identity(rng, nationality=nationality) if nationality else generate_human_identity(rng)
    hometown = str(ident.hometown or "Unknown")
    birth_city = hometown.split(",")[0].strip() if hometown else "Unknown"
    arch_name = str(getattr(archetype, "value", archetype) or "")
    if junior_pool:
        h_cm = generate_position_height_cm(rng, pos)
        w_kg = generate_realistic_weight_kg(h_cm, pos, age=age)
    else:
        h_cm = generate_position_height_cm(rng, pos, archetype=arch_name)
        w_kg = generate_realistic_weight_kg(h_cm, pos, archetype=arch_name, age=age)
    identity = IdentityBio(
        name=str(ident.full_name),
        age=age,
        birth_year=birth_year,
        birth_country=str(ident.nationality),
        birth_city=birth_city or "Unknown",
        height_cm=h_cm,
        weight_kg=w_kg,
        position=pos,
        shoots=Shoots.L if rng.random() < 0.58 else Shoots.R,
        draft_year=max(2015, birth_year + 18),
        draft_round=1 + (rng.randint(1, 7)),
        draft_pick=1 + (rng.randint(0, 30)),
    )
    try:
        identity.birth_month = 7
        identity.birth_day = 1
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
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
        archetype=archetype,
        pool_context=pool_context,
        enforce_floor_on_init=False,
    )
    setattr(player, "_dob_anchor_year", year)
    if junior_pool:
        setattr(player, "_identity_provisional", True)
        setattr(player, "_identity_committed", False)
    try:
        player.age = age
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    apply_body_tradeoffs_to_ratings(player, rng)
    floor = get_ovr_floor_for_pool(pool_context)
    if floor > 0:
        enforce_minimum_player_ovr(player, floor)

    ensure_player_chemistry_profile = _ensure_player_chemistry_profile_cached()
    if ensure_player_chemistry_profile:
        try:
            ensure_player_chemistry_profile(player, rng)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    ensure_player_headshot = _ensure_player_headshot_cached()
    if ensure_player_headshot:
        try:
            ensure_player_headshot(player)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    if str(pool_context or "").lower() in ("junior", "college", "european_junior"):
        # Draft picks (rounds 1-7) are all made expecting the prospect to play NHL
        # games, so true "never sniffs the NHL" busts are kept rare. Risk/reward is
        # preserved via the floor/ceiling bands, not by generating dead-on-arrival kids.
        bust = rng.random() < 0.06
        steal = (not bust) and rng.random() < 0.042
        setattr(player, "pipeline_bust", bool(bust))
        setattr(player, "pipeline_steal", bool(steal))
        if bust:
            setattr(player, "dev_type", "bust")
        elif steal:
            setattr(player, "dev_type", "elite")

    league_players.append(player)
    setattr(player, "_spawn_version", 2)
    return player


def _init_chars(league: Any, rng: random.Random, players: Sequence[Any]) -> None:
    from app.sim_engine.engine import assign_development_profile, ensure_player_character_initialized

    for p in players:
        if getattr(p, "retired", False):
            continue
        ensure_player_character_initialized(p, rng)
        assign_development_profile(p, rng)


def _positions_for_block(
    rng: random.Random,
    *,
    forwards: int,
    defense: int,
    goalies: int,
) -> List[Position]:
    slots: List[Position] = []
    for _ in range(forwards // 3):
        slots.extend([Position.C, Position.LW, Position.RW])
    rem = forwards - len(slots)
    extra = [Position.C, Position.LW, Position.RW, Position.LW, Position.RW]
    for i in range(rem):
        slots.append(extra[i % len(extra)])
    rng.shuffle(slots)
    slots.extend([Position.D] * defense)
    slots.extend([Position.G] * goalies)
    return slots


def development_league_player_count(league: Any) -> int:
    total = 0
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            total += len(tm.get("players") or [])
    return total


def ensure_development_leagues_for_draft_board(
    league: Any,
    rng: random.Random,
    season_year: Optional[int] = None,
) -> bool:
    """Real-NHL fast bootstrap skips junior trees; rebuild them for draft/scouting boards."""
    if development_league_player_count(league) > 0:
        return False
    teams = list(getattr(league, "teams", None) or [])
    if not teams:
        return False

    year = spawn_as_of_year(season_year)
    set_spawn_as_of_year(year)
    if not hasattr(league, "players") or league.players is None:
        league.players = []
    league_players: List[Any] = list(league.players)
    used_names: set = set()
    for p in league_players:
        ident = getattr(p, "identity", None)
        if ident and getattr(ident, "name", None):
            used_names.add(str(ident.name))

    from app.sim_engine.generation.prospect_league_teams import (
        LEAGUE_REGISTRY,
        choose_nationality_for_league,
        teams_for_league,
        validate_prospect_league_fit,
    )

    dev: List[Dict[str, Any]] = []

    def _add_league(code: str, title: str, f: int, d: int, g: int, age_lo: int, age_hi: int, ovr_lo: float, ovr_hi: float) -> None:
        team_specs = teams_for_league(code)
        if not team_specs:
            return
        teams_out: List[Dict[str, Any]] = []
        for ti, spec in enumerate(team_specs):
            roster: List[Player] = []
            slots = _positions_for_block(rng, forwards=f, defense=d, goalies=g)
            city = str(spec.get("city") or "")
            club_name = str(spec.get("name") or city)
            tid_j = f"{code}_{ti + 1}"
            for pos in slots:
                nat = choose_nationality_for_league(rng, code)
                p = None
                for _attempt in range(10):
                    p = _spawn_player(
                        rng,
                        pos=pos,
                        ovr_lo=ovr_lo,
                        ovr_hi=ovr_hi,
                        age_lo=age_lo,
                        age_hi=age_hi,
                        used_names=used_names,
                        league_players=league_players,
                        pool_context=_pool_context_for_level("junior", code),
                        nationality=nat,
                        league_code=code,
                    )
                    birth_country = str(getattr(getattr(p, "identity", None), "birth_country", "") or "")
                    if validate_prospect_league_fit(birth_country, code):
                        break
                    nat = choose_nationality_for_league(rng, code)
                if p is None:
                    continue
                p.context.current_team_id = tid_j
                _set_assignment(p, level="junior", league_code=code, club=club_name)
                try:
                    from app.sim_engine.generation.prospect_league_scoring import initialize_prospect_season

                    initialize_prospect_season(p, code, rng=rng)
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
                roster.append(p)
            teams_out.append({"team_id": tid_j, "name": club_name, "city": city, "players": roster})
        display = str(LEAGUE_REGISTRY.get(code, {}).get("display") or title)
        dev.append({"league_code": code, "league_name": display, "teams": teams_out})

    _add_league("CHL_OHL", "OHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("CHL_WHL", "WHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("CHL_QMJHL", "QMJHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("USHL", "USHL", 12, 6, 3, 16, 19, 0.30, 0.48)
    _add_league("NCAA", "NCAA", 13, 7, 3, 18, 21, 0.34, 0.55)
    for code in (
        "EU_J_SHL",
        "EU_J_LIIGA",
        "EU_J_DEL",
        "EU_J_SWISS",
        "EU_J_CZ",
        "EU_J_SK",
        "EU_J_KHL_JR",
        "EU_J_NOR",
        "EU_J_DEN",
        "EU_J_AUT",
    ):
        label = str(LEAGUE_REGISTRY.get(code, {}).get("display") or code)
        _add_league(code, label, 5, 3, 1, 16, 20, 0.30, 0.50)

    if not dev:
        return False

    league.development_leagues = dev
    league.players = league_players
    _mark_overage_juniors_off_open_board(league, rng, year)
    _shape_draft_class_pipeline(league, rng, draft_year=year + 1)

    all_new: List[Any] = []
    for block in dev:
        for tm in block.get("teams") or []:
            all_new.extend(tm.get("players") or [])
    _init_chars(league, rng, all_new)

    try:
        from services.player_bio_parser import apply_player_bios_to_league

        apply_player_bios_to_league(league, as_of_year=year)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return True


def bootstrap_full_league_hierarchy(
    league: Any,
    rng: random.Random,
    season_year: Optional[int] = None,
    dynasty_registry: Optional[Any] = None,
    *,
    fast_depth: bool = False,
) -> None:
    """
    Mutates league + NHL Team objects:
      - team.ahl_roster, team.echl_roster
      - league.free_agents, league.overseas_free_agents
      - league.development_leagues (structured trees for UI)
    Appends all new Player objects to league.players.
    """
    teams = list(getattr(league, "teams", None) or [])
    if not teams:
        return

    year = spawn_as_of_year(season_year)
    set_spawn_as_of_year(year)
    try:
        setattr(league, "season_start_year", year)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    if not hasattr(league, "players") or league.players is None:
        league.players = []
    league_players: List[Any] = list(league.players)
    used_names: set = set()
    for p in league_players:
        ident = getattr(p, "identity", None)
        if ident and getattr(ident, "name", None):
            used_names.add(str(ident.name))

    if dynasty_registry is None and getattr(league, "real_nhl_import_meta", None):
        try:
            from services.dynasty_ratings_parser import load_dynasty_ratings_registry

            dynasty_registry = load_dynasty_ratings_registry()
        except Exception:
            dynasty_registry = None

    # --- Affiliate rosters per NHL org ---
    if fast_depth:
        AHL_F, AHL_D, AHL_G = 12, 6, 2
        ECHL_F, ECHL_D, ECHL_G = 6, 3, 1
    else:
        AHL_F, AHL_D, AHL_G = 14, 8, 2
        ECHL_F, ECHL_D, ECHL_G = 11, 5, 2

    spawn_align_rounds = 6 if fast_depth else 40
    spawn_apply_bio = not fast_depth
    spawn_bio_registry = None
    if dynasty_registry and getattr(league, "real_nhl_import_meta", None):
        try:
            from services.dynasty_ratings_parser import _spawn_bio_registry

            spawn_bio_registry = _spawn_bio_registry(int(year))
        except Exception:
            spawn_bio_registry = None

    for team in teams:
        # Keep real-NHL overflow already assigned to the affiliate (23-man trim).
        preserved_ahl = [
            p
            for p in (getattr(team, "ahl_roster", None) or [])
            if getattr(p, "real_nhl_import", False)
        ]
        preserved_echl = [
            p
            for p in (getattr(team, "echl_roster", None) or [])
            if getattr(p, "real_nhl_import", False)
        ]

        if not hasattr(team, "ahl_roster") or team.ahl_roster is None:
            team.ahl_roster = []
        else:
            team.ahl_roster.clear()
        if not hasattr(team, "echl_roster") or team.echl_roster is None:
            team.echl_roster = []
        else:
            team.echl_roster.clear()

        tid = str(getattr(team, "team_id", ""))
        abbr = str(
            getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or tid
        ).upper()
        use_dynasty = bool(dynasty_registry) and bool(getattr(league, "real_nhl_import_meta", None))

        ahl_slots = _positions_for_block(rng, forwards=AHL_F, defense=AHL_D, goalies=AHL_G)
        # Leave room for preserved real NHL overflow so affiliates aren't bloated.
        generated_ahl_budget = max(0, len(ahl_slots) - len(preserved_ahl))
        dynasty_ahl: List[Any] = []
        if use_dynasty:
            try:
                from services.dynasty_ratings_parser import spawn_player_from_dynasty_entry

                for entry in dynasty_registry.entries_for_team(abbr, "ahl"):
                    nm = str(entry.raw_name)
                    if nm in used_names:
                        continue
                    if len(dynasty_ahl) >= generated_ahl_budget:
                        break
                    p = spawn_player_from_dynasty_entry(
                        entry,
                        rng=rng,
                        pool_context="ahl",
                        used_names=used_names,
                        league_players=league_players,
                        as_of_year=year,
                        align_rounds=spawn_align_rounds,
                        apply_bio=spawn_apply_bio,
                        bio_registry=spawn_bio_registry,
                    )
                    p.context.current_team_id = f"AHL_{tid}"
                    _set_assignment(p, org_nhl_team_id=tid, level="ahl", club=_team_label(team))
                    dynasty_ahl.append(p)
            except Exception:
                logging.getLogger("uvicorn.error").exception(
                    "dynasty AHL spawn failed for %s", abbr
                )
                dynasty_ahl = []

        slot_idx = 0
        for pos in ahl_slots[:generated_ahl_budget]:
            if slot_idx < len(dynasty_ahl):
                team.ahl_roster.append(dynasty_ahl[slot_idx])
                slot_idx += 1
                continue
            lo, hi = (0.42, 0.62) if pos != Position.G else (0.48, 0.68)
            p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=20, age_hi=28, used_names=used_names, league_players=league_players, pool_context="ahl")
            p.context.current_team_id = f"AHL_{tid}"
            _set_assignment(p, org_nhl_team_id=tid, level="ahl", club=_team_label(team))
            team.ahl_roster.append(p)
        for p in preserved_ahl:
            try:
                p.in_minors = True
                p.is_buried = True
                p.buried = True
                p.roster_location = "ahl"
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            try:
                p.context.current_team_id = f"AHL_{tid}"
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            _set_assignment(p, org_nhl_team_id=tid, level="ahl", club=_team_label(team))
            if p not in team.ahl_roster:
                team.ahl_roster.append(p)

        if use_dynasty:
            try:
                from services.dynasty_ratings_parser import spawn_player_from_dynasty_entry

                if not hasattr(team, "prospect_pool") or team.prospect_pool is None:
                    team.prospect_pool = []
                else:
                    team.prospect_pool.clear()
                for entry in dynasty_registry.entries_for_team(abbr, "prospect"):
                    nm = str(entry.raw_name)
                    if nm in used_names:
                        continue
                    p = spawn_player_from_dynasty_entry(
                        entry,
                        rng=rng,
                        pool_context="prospect",
                        used_names=used_names,
                        league_players=league_players,
                        as_of_year=year,
                        align_rounds=spawn_align_rounds,
                        apply_bio=spawn_apply_bio,
                        bio_registry=spawn_bio_registry,
                    )
                    try:
                        p.nhl_rights_team_id = tid
                        p.rights_team_id = tid
                        p.signed = False
                    except Exception:
                        _swallowed_log.debug("suppressed exception", exc_info=True)
                    _set_assignment(p, org_nhl_team_id=tid, level="prospect", club=_team_label(team))
                    team.prospect_pool.append(p)
            except Exception:
                logging.getLogger("uvicorn.error").exception(
                    "dynasty prospect spawn failed for %s", abbr
                )

        echl_slots = _positions_for_block(rng, forwards=ECHL_F, defense=ECHL_D, goalies=ECHL_G)
        generated_echl_budget = max(0, len(echl_slots) - len(preserved_echl))
        if fast_depth:
            generated_echl_budget = 0
        for pos in echl_slots[:generated_echl_budget]:
            lo, hi = (0.36, 0.55) if pos != Position.G else (0.42, 0.60)
            p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=21, age_hi=30, used_names=used_names, league_players=league_players, pool_context="echl")
            p.context.current_team_id = f"ECHL_{tid}"
            _set_assignment(p, org_nhl_team_id=tid, level="echl", club=_team_label(team))
            team.echl_roster.append(p)
        for p in preserved_echl:
            try:
                p.in_minors = True
                p.roster_location = "echl"
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            if p not in team.echl_roster:
                team.echl_roster.append(p)

    # --- Free agents (NHL-contract eligible pool) ---
    ufa_pool_size = 48 if fast_depth else 520
    league.free_agents = []
    for _ in range(ufa_pool_size):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.D, Position.G])
        lo, hi = (0.38, 0.58) if pos != Position.G else (0.45, 0.62)
        age_lo, age_hi = (22, 34)
        p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=age_lo, age_hi=age_hi, used_names=used_names, league_players=league_players, pool_context="ufa")
        p.context.current_team_id = "UFA"
        _set_assignment(p, level="ufa", overseas=False)
        league.free_agents.append(p)

    # Generated "stud" free agents used to arrive 12 at a time and dominate the market.
    # Real FA classes come from real expiring contracts; a manufactured star is rare.
    elite_ufa_count = 1 if rng.random() < 0.25 else 0
    for _ in range(elite_ufa_count):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.D, Position.G])
        lo, hi = (0.78, 0.88) if pos != Position.G else (0.80, 0.90)
        p = _spawn_player(
            rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=27, age_hi=34,
            used_names=used_names, league_players=league_players, pool_context="ufa_elite",
        )
        p.context.current_team_id = "UFA"
        _set_assignment(p, level="ufa", overseas=False)
        league.free_agents.append(p)

    overseas_pool_size = 24 if fast_depth else 220
    league.overseas_free_agents = []
    for _ in range(overseas_pool_size):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.G])
        lo, hi = (0.40, 0.62) if pos != Position.G else (0.48, 0.70)
        p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=23, age_hi=32, used_names=used_names, league_players=league_players, pool_context="overseas")
        p.context.current_team_id = "OVERSEAS"
        _set_assignment(p, level="ufa", overseas=True, overseas_league=rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"]))
        league.overseas_free_agents.append(p)

    # A Panarin-style import from Europe happens ~1 summer in 4, not six at once.
    overseas_elite_count = 1 if rng.random() < 0.25 else 0
    for _ in range(overseas_elite_count):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.G])
        lo, hi = (0.80, 0.90) if pos != Position.G else (0.82, 0.92)
        p = _spawn_player(
            rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=26, age_hi=33,
            used_names=used_names, league_players=league_players, pool_context="overseas_elite",
        )
        p.context.current_team_id = "OVERSEAS"
        _set_assignment(
            p, level="ufa", overseas=True,
            overseas_league=rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"]),
        )
        league.overseas_free_agents.append(p)

    # --- Junior / NCAA style development leagues (real club names per league) ---
    from app.sim_engine.generation.prospect_league_teams import (
        LEAGUE_REGISTRY,
        choose_nationality_for_league,
        teams_for_league,
        validate_prospect_league_fit,
    )

    dev: List[Dict[str, Any]] = []

    if fast_depth:
        ensure_development_leagues_for_draft_board(league, rng, season_year=year)
        league.players = league_players
        all_new: List[Any] = []
        for team in teams:
            all_new.extend(team.ahl_roster)
            all_new.extend(team.echl_roster)
            all_new.extend(getattr(team, "prospect_pool", None) or [])
        all_new.extend(league.free_agents)
        all_new.extend(league.overseas_free_agents)
        org_players: List[Any] = []
        for team in teams:
            org_players.extend(getattr(team, "roster", None) or [])
            org_players.extend(team.ahl_roster)
            org_players.extend(team.echl_roster)
            org_players.extend(getattr(team, "prospect_pool", None) or [])
        org_players.extend(league.free_agents)
        org_players.extend(league.overseas_free_agents)
        try:
            from services.player_bio_parser import apply_player_bios_to_league

            apply_player_bios_to_league(league, as_of_year=year)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return

    def _add_league(code: str, title: str, f: int, d: int, g: int, age_lo: int, age_hi: int, ovr_lo: float, ovr_hi: float) -> None:
        team_specs = teams_for_league(code)
        if not team_specs:
            return
        teams_out: List[Dict[str, Any]] = []
        for ti, spec in enumerate(team_specs):
            roster: List[Player] = []
            slots = _positions_for_block(rng, forwards=f, defense=d, goalies=g)
            city = str(spec.get("city") or "")
            club_name = str(spec.get("name") or city)
            tid_j = f"{code}_{ti + 1}"
            for pos in slots:
                nat = choose_nationality_for_league(rng, code)
                p = None
                for _attempt in range(10):
                    p = _spawn_player(
                        rng,
                        pos=pos,
                        ovr_lo=ovr_lo,
                        ovr_hi=ovr_hi,
                        age_lo=age_lo,
                        age_hi=age_hi,
                        used_names=used_names,
                        league_players=league_players,
                        pool_context=_pool_context_for_level("junior", code),
                        nationality=nat,
                        league_code=code,
                    )
                    birth_country = str(getattr(getattr(p, "identity", None), "birth_country", "") or "")
                    if validate_prospect_league_fit(birth_country, code):
                        break
                    nat = choose_nationality_for_league(rng, code)
                if p is None:
                    continue
                p.context.current_team_id = tid_j
                _set_assignment(p, level="junior", league_code=code, club=club_name)
                if not fast_depth:
                    try:
                        from app.sim_engine.generation.prospect_league_scoring import initialize_prospect_season

                        initialize_prospect_season(p, code, rng=rng)
                    except Exception:
                        _swallowed_log.debug("suppressed exception", exc_info=True)
                roster.append(p)
            teams_out.append({"team_id": tid_j, "name": club_name, "city": city, "players": roster})
        display = str(LEAGUE_REGISTRY.get(code, {}).get("display") or title)
        dev.append({"league_code": code, "league_name": display, "teams": teams_out})

    _add_league("CHL_OHL", "OHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("CHL_WHL", "WHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("CHL_QMJHL", "QMJHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("USHL", "USHL", 12, 6, 3, 16, 19, 0.30, 0.48)
    _add_league("NCAA", "NCAA", 13, 7, 3, 18, 21, 0.34, 0.55)

    # European junior blocks span far more clubs (112 teams) than the CHL (61).
    # In reality the CHL is the dominant NHL-draft feeder, so each European club
    # contributes a smaller draft-eligible roster. This keeps every league/team
    # intact while preventing Europe from over-populating the eligible pool and
    # diluting Canadian representation (see LEAGUE_NATIONALITY_WEIGHTS audit).
    for code in (
        "EU_J_SHL",
        "EU_J_LIIGA",
        "EU_J_DEL",
        "EU_J_SWISS",
        "EU_J_CZ",
        "EU_J_SK",
        "EU_J_KHL_JR",
        "EU_J_NOR",
        "EU_J_DEN",
        "EU_J_AUT",
    ):
        label = str(LEAGUE_REGISTRY.get(code, {}).get("display") or code)
        _add_league(code, label, 5, 3, 1, 16, 20, 0.30, 0.50)

    league.development_leagues = dev
    league.players = league_players

    if not fast_depth:
        _mark_overage_juniors_off_open_board(league, rng, year)
        _shape_draft_class_pipeline(league, rng, draft_year=year + 1)

    all_new: List[Any] = []
    for team in teams:
        all_new.extend(team.ahl_roster)
        all_new.extend(team.echl_roster)
        all_new.extend(getattr(team, "prospect_pool", None) or [])
    all_new.extend(league.free_agents)
    all_new.extend(league.overseas_free_agents)
    for block in dev:
        for tm in block.get("teams") or []:
            all_new.extend(tm.get("players") or [])

    if fast_depth:
        org_players: List[Any] = []
        for team in teams:
            org_players.extend(getattr(team, "roster", None) or [])
            org_players.extend(team.ahl_roster)
            org_players.extend(team.echl_roster)
            org_players.extend(getattr(team, "prospect_pool", None) or [])
        org_players.extend(league.free_agents)
        org_players.extend(league.overseas_free_agents)
        _init_chars(league, rng, org_players)
    else:
        _init_chars(league, rng, all_new)

    try:
        from services.player_bio_parser import apply_player_bios_to_league

        apply_player_bios_to_league(league, as_of_year=year)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


# Star-power tiers: (label, weight, franchise/elite/top slot counts, top target ovr)
# Current ability is NHL-scale. Top picks should look like near-NHL talents a GM
# would promote within 1–3 years (~78–86 current), not mid-60s juniors.
# Years stay in the same band. A class can be a little deeper than the last one.
# A "weak" roll used to strip the top of the draft after the real first class.
_CLASS_STRENGTH_TIERS = [
    ("average", 0.55, 3, 6, 12, (0.82, 0.87)),
    ("strong", 0.35, 3, 7, 13, (0.83, 0.88)),
    ("elite", 0.10, 4, 8, 14, (0.84, 0.89)),
]

# Minimum current OVR (0–1) for shaped pipeline stars — used to repair older saves
# that were generated under the too-low 0.50–0.58 franchise bands.
_PIPELINE_OVR_REPAIR = {
    "transcendent": (0.82, 0.84, 0.90),
    "franchise": (0.78, 0.79, 0.86),
    "elite": (0.74, 0.74, 0.82),
    "top": (0.70, 0.70, 0.78),
}

# Depth quality is independent of star power — a class can be top-heavy or deep.
_DEPTH_QUALITY_TIERS = [
    ("average", 0.58, {"round1_tail": 40, "nhl_floor": 22, "round2": 36, "round3_4": 56, "round5_7": 72, "upside_pct": 0.24}),
    ("strong", 0.32, {"round1_tail": 48, "nhl_floor": 28, "round2": 44, "round3_4": 68, "round5_7": 88, "upside_pct": 0.30}),
    ("elite", 0.10, {"round1_tail": 55, "nhl_floor": 36, "round2": 52, "round3_4": 80, "round5_7": 100, "upside_pct": 0.36}),
]

_TRANSCENDENT_BACKSTORY_KEYS = (
    "backyard_rink_kid",
    "small_town_superstar",
    "late_bloomer",
    "hockey_family_legacy",
    "outdoor_pond_grinder",
    "multi_sport_athlete",
    "immigrant_family_dream",
    "undersized_skill_wizard",
    "captain_since_childhood",
    "troublemaker_turned_competitor",
)


def _player_ovr_frac(player: Any) -> float:
    try:
        from app.sim_engine.entities.player import player_current_ovr_01

        return max(0.0, min(1.0, float(player_current_ovr_01(player))))
    except Exception:
        try:
            ovr_fn = getattr(player, "ovr", None)
            if callable(ovr_fn):
                from app.sim_engine.entities.player import normalize_rating

                return max(0.0, min(1.0, float(normalize_rating(ovr_fn()))))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return 0.45


def _apply_shaped_player(
    p: Any,
    *,
    tier: str,
    lo: float,
    hi: float,
    pot_lo: int,
    pot_hi: int,
    rng: random.Random,
    code_by_id: dict,
    rng_inst: random.Random,
) -> None:
    from app.sim_engine.engine import build_role_shaped_ratings

    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", Position.C) if ident else Position.C
    target = max(0.36, min(0.92, rng_inst.uniform(lo, hi)))
    try:
        p.ratings = build_role_shaped_ratings(position=pos, target_ovr=target, rng=rng_inst)
        # Stale ovr() memo would keep the pre-shape ~50 overall after ratings change.
        inval = getattr(p, "_invalidate_ovr_memo", None)
        if callable(inval):
            inval()
        try:
            from app.sim_engine.entities.player import persist_recomputed_ovr

            persist_recomputed_ovr(p)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        from app.sim_engine.engine import pop_generation_profile
        from app.sim_engine.entities.player import archetype_from_generation_profile, assign_skater_archetype

        gen_profile = pop_generation_profile(p.ratings)
        synced = archetype_from_generation_profile(gen_profile, pos)
        if synced:
            setattr(p, "archetype", synced)
        elif not getattr(p, "archetype", None):
            setattr(p, "archetype", assign_skater_archetype(pos, rng_inst))
        if gen_profile:
            setattr(p, "_generated_profile", gen_profile)
    except Exception:
        return
    p.ratings["dev_potential"] = rng_inst.randint(int(pot_lo), int(pot_hi))
    try:
        from app.sim_engine.progression.development import assign_prospect_realization

        assign_prospect_realization(p, force=True)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    setattr(p, "pipeline_tier", tier)
    setattr(p, "pipeline_bust", False)
    if tier in ("franchise", "transcendent", "hidden_upside"):
        setattr(p, "pipeline_steal", True)
        setattr(p, "dev_type", "elite")
    if tier == "transcendent":
        setattr(p, "is_transcendent", True)
        setattr(p, "transcendent_talent", True)
    if tier == "nhl_floor":
        _boost_defensive_tools(p, rng_inst)
    try:
        from app.sim_engine.systems.chemistry import ensure_player_chemistry_profile

        ensure_player_chemistry_profile(p, rng_inst)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    if rng_inst.random() < 0.12:
        setattr(p, "character_concerns", True)
    code = code_by_id.get(id(p), "JUNIOR")
    try:
        from app.sim_engine.generation.prospect_league_scoring import initialize_prospect_season

        initialize_prospect_season(p, code, rng=rng_inst, force=True, preserve_actual=True)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _boost_defensive_tools(player: Any, rng: random.Random) -> None:
    """Safe-floor NHL prospects: strong defensive tools, limited offensive ceiling."""
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict):
        return
    ident = getattr(player, "identity", None)
    pos = getattr(ident, "position", Position.C) if ident else Position.C
    if pos == Position.G:
        for k in ratings:
            if any(x in str(k).lower() for x in ("reflex", "position", "rebound", "glove")):
                ratings[k] = clamp_rating(int(float(ratings[k])) + rng.randint(2, 6))
        return
    if pos == Position.D:
        for k in ratings:
            kl = str(k).lower()
            if any(x in kl for x in ("def", "stick", "gap", "block", "poke", "position")):
                ratings[k] = clamp_rating(int(float(ratings[k])) + rng.randint(3, 8))
            elif any(x in kl for x in ("skat", "strength", "physical")):
                ratings[k] = clamp_rating(int(float(ratings[k])) + rng.randint(1, 4))
    else:
        for k in ratings:
            kl = str(k).lower()
            if any(x in kl for x in ("def", "faceoff", "stick", "check")):
                ratings[k] = clamp_rating(int(float(ratings[k])) + rng.randint(2, 7))
            elif "skat" in kl:
                ratings[k] = clamp_rating(int(float(ratings[k])) + rng.randint(1, 3))


_SHAPED_PIPELINE_TIERS = frozenset({
    "transcendent", "franchise", "elite", "top", "round1_tail",
    "nhl_floor", "round2", "round3_4", "round5_7", "hidden_upside",
})


def _assign_residual_dev_potential(
    player: Any,
    rng: random.Random,
    *,
    depth_label: str,
    upside_pct: float,
) -> None:
    """Every draft-eligible player gets a true ceiling — not only pyramid picks."""
    tier = str(getattr(player, "pipeline_tier", "") or "")
    if tier in _SHAPED_PIPELINE_TIERS:
        return
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict):
        return
    ovr99 = _player_ovr_frac(player) * 99.0
    depth_shift = {"weak": -3, "average": 0, "strong": 2, "elite": 4}.get(depth_label, 0)
    roll = rng.random()

    if roll < upside_pct * 0.28:
        pot = rng.randint(82, 93)
        setattr(player, "pipeline_steal", True)
        setattr(player, "dev_type", "elite")
        setattr(player, "pipeline_tier", "pool_upside")
    elif roll < upside_pct * 0.75:
        pot = rng.randint(77, 86)
        setattr(player, "pipeline_tier", "pool_upside")
    elif roll < upside_pct:
        pot = rng.randint(73, 82)
        setattr(player, "pipeline_tier", "pool_upside")
    elif roll < 0.20:
        pot = rng.randint(76, 86)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.42:
        pot = rng.randint(72, 82)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.64:
        pot = rng.randint(68, 78)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.82:
        pot = rng.randint(64, 74)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.93:
        pot = rng.randint(62, 72)
        setattr(player, "pipeline_tier", "pool")
    else:
        pot = rng.randint(60, 70)
        setattr(player, "pipeline_tier", "pool")

    pot = int(max(ovr99 + 2, min(99, pot + depth_shift)))
    ratings["dev_potential"] = pot
    try:
        from app.sim_engine.progression.development import assign_prospect_realization

        assign_prospect_realization(player, force=True)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _pop_shaping_pool(pool: List[Any]) -> Optional[Any]:
    return pool.pop(0) if pool else None


def _shape_draft_class_pipeline(
    league: Any,
    rng: random.Random,
    *,
    draft_year: Optional[int] = None,
) -> None:
    """
    Post-spawn pass: builds a full draft-class talent pool with star power AND depth.

    Star-power tiers control the very top of the class. Depth-quality tiers control
    how many NHL-calibre ceilings exist in rounds 2–7. Residual dev_potential is
    assigned to every remaining draft-age player so the class is never a flat cliff
    after pick ~15.

    Each prospect is shaped exactly once (``_pipeline_shaped_year``) when they enter
    their first draft year (Sept 15 age = 18). Returning players and 17-year-olds are
    never touched here — in-season growth is preserved.
    """
    dy = int(draft_year or _upcoming_draft_year(league))

    draft_age: List[Any] = []
    code_by_id: dict = {}
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "JUNIOR")
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                # Already-drafted prospects keep their current ability — this pass only
                # shapes the fresh, undrafted cohort feeding the next draft class. Reshaping
                # drafted players here would randomly re-roll (and often crater) their OVR.
                if bool(getattr(p, "drafted", False)) or getattr(p, "nhl_rights_team_id", None) or getattr(
                    p, "rights_team_id", None
                ) or getattr(p, "drafted_by", None):
                    continue
                if bool(getattr(p, "passed_over_draft", False)):
                    continue
                if _pipeline_already_shaped(p):
                    continue
                if not _is_first_year_draft_eligible(p, dy):
                    continue
                draft_age.append(p)
                code_by_id[id(p)] = code
    if len(draft_age) < 12:
        try:
            setattr(league, "_pipeline_shaped_for_draft_year", dy)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return

    roll = rng.random()
    acc = 0.0
    tier_label, n_franchise, n_elite, n_top, top_range = "average", 1, 4, 8, (0.69, 0.74)
    for label, weight, nf, ne, nt, rng_top in _CLASS_STRENGTH_TIERS:
        acc += weight
        if roll <= acc:
            tier_label, n_franchise, n_elite, n_top, top_range = label, nf, ne, nt, rng_top
            break
    league.draft_class_strength = tier_label

    d_roll = rng.random()
    d_acc = 0.0
    depth_label = "average"
    depth_slots = _DEPTH_QUALITY_TIERS[1][2]
    for label, weight, slots in _DEPTH_QUALITY_TIERS:
        d_acc += weight
        if d_roll <= d_acc:
            depth_label, depth_slots = label, slots
            break
    league.draft_class_depth = depth_label

    g_roll = rng.random()
    g_acc = 0.0
    g_label, g_boost = "normal", 0.0
    force_g = os.environ.get("NHL_FORCE_GOALIE_CLASS", "").strip().lower()
    _g_boost_map = {
        "weak": -1.5,
        "normal": 0.0,
        "strong": 1.2,
        "elite": 2.5,
        "generational": 4.5,
    }
    if force_g in _g_boost_map:
        g_label = force_g
        g_boost = _g_boost_map[force_g]
    else:
        for label, weight, boost in (
            ("normal", 0.62, 0.0),
            ("strong", 0.28, 1.0),
            ("elite", 0.08, 2.0),
            ("weak", 0.02, -0.6),
        ):
            g_acc += weight
            if g_roll <= g_acc:
                g_label, g_boost = label, boost
                break
    league.goalie_class_strength = g_label
    league.goalie_class_boost = g_boost

    force_transcendent = os.environ.get("NHL_FORCE_TRANSCENDENT", "0") == "1"
    transcendent_roll = force_transcendent or rng.random() < TRANSCENDENT_CLASS_PROB
    league.has_transcendent_talent = bool(transcendent_roll)

    skaters = [p for p in draft_age if getattr(getattr(p, "identity", None), "position", None) != Position.G]
    goalies = [p for p in draft_age if getattr(getattr(p, "identity", None), "position", None) == Position.G]
    rng.shuffle(skaters)
    rng.shuffle(goalies)

    def _next_skater(p: float = 0.92) -> Optional[Any]:
        return _pop_shaping_pool(skaters)

    def _next_goalie(p: float = 0.90) -> Optional[Any]:
        return _pop_shaping_pool(goalies)

    generational_goalie = _next_goalie(0.95) if (goalies and rng.random() < 0.05) else None
    if getattr(league, "goalie_class_strength", "") in ("elite", "generational") and generational_goalie is None:
        generational_goalie = _next_goalie(0.95)

    chosen: List[tuple] = []

    if transcendent_roll:
        tp = _next_skater(0.99)
        if tp is not None:
            chosen.append((tp, "transcendent", top_range[1], min(0.92, top_range[1] + 0.08), 99, 99))
            setattr(tp, "is_transcendent", True)
            setattr(tp, "transcendent_talent", True)
            setattr(tp, "aura_tier", "gold")
            setattr(tp, "draft_hype_tier", "mythic")
            setattr(tp, "tank_target", True)
            setattr(tp, "storyline_priority", "legendary")
            setattr(tp, "pipeline_tier", "transcendent")
            setattr(tp, "backstory_key", rng.choice(_TRANSCENDENT_BACKSTORY_KEYS))

    for _ in range(n_franchise):
        sp = _next_skater(0.94)
        if sp is None:
            break
        chosen.append((sp, "franchise", top_range[0], top_range[1], 93, 98))
    if generational_goalie is not None:
        chosen.append((generational_goalie, "franchise", top_range[0], top_range[1], 90, 98))
        setattr(generational_goalie, "generational_goalie", True)

    _goalie_shape = {
        "weak": (1, 4),
        "normal": (2, 6),
        "strong": (3, 8),
        "elite": (4, 8),
        "generational": (3, 6),
    }
    n_elite_g, n_top_g = _goalie_shape.get(g_label, (0, 2))
    if n_elite_g > 0:
        for _ in range(n_elite_g):
            gp = _next_goalie(0.90)
            if gp is None:
                break
            chosen.append((gp, "elite", top_range[0] - 0.03, top_range[1] - 0.02, 86, 93))
    for _ in range(n_top_g):
        gp = _next_goalie(0.88)
        if gp is None:
            break
        chosen.append((gp, "top", top_range[0] - 0.08, top_range[0] - 0.02, 80, 89))

    for _ in range(n_elite):
        sp = _next_skater(0.92)
        if sp is None:
            break
        chosen.append((sp, "elite", top_range[0] - 0.04, top_range[1] - 0.04, 89, 95))
    for _ in range(n_top):
        sp = _next_skater(0.90)
        if sp is None:
            break
        chosen.append((sp, "top", top_range[0] - 0.07, top_range[1] - 0.07, 86, 92))

    n_round1_tail = int(depth_slots.get("round1_tail", 38))
    for _ in range(n_round1_tail):
        sp = _next_skater(0.86)
        if sp is None:
            break
        chosen.append((sp, "round1_tail", top_range[0] - 0.10, top_range[0] - 0.04, 84, 91))

    n_nhl_floor = int(depth_slots.get("nhl_floor", 16))
    for _ in range(n_nhl_floor):
        sp = _next_skater(0.78)
        if sp is None:
            break
        # Solid NHL-bound depth — below top-of-class current ability.
        chosen.append((sp, "nhl_floor", 0.64, 0.71, 76, 82))

    n_round2 = int(depth_slots.get("round2", 28))
    for _ in range(n_round2):
        sp = _next_skater(0.70)
        if sp is None:
            break
        chosen.append((sp, "round2", 0.66, 0.75, 79, 87))

    n_round3_4 = int(depth_slots.get("round3_4", 40))
    for _ in range(n_round3_4):
        sp = _next_skater(0.58)
        if sp is None:
            break
        chosen.append((sp, "round3_4", 0.62, 0.71, 75, 84))

    n_round5_7 = int(depth_slots.get("round5_7", 48))
    upside_pct = float(depth_slots.get("upside_pct", 0.08))
    for _ in range(n_round5_7):
        sp = _next_skater(0.48)
        if sp is None:
            break
        if rng.random() < upside_pct * 0.70:
            chosen.append((sp, "hidden_upside", 0.60, 0.69, 82, 92))
        elif rng.random() < 0.45:
            chosen.append((sp, "round5_7", 0.58, 0.67, 73, 82))
        else:
            chosen.append((sp, "round5_7", 0.56, 0.65, 70, 79))

    # Spread real goalie prospects through the class. Non-franchise goalies stay
    # out of round 1 via the board scatter; these bands are rounds 2-7.
    for i in range(26):
        gp = _next_goalie(0.72)
        if gp is None:
            break
        if i < 8:
            chosen.append((gp, "round2", 0.64, 0.73, 78, 87))
        elif i < 16:
            chosen.append((gp, "round3_4", 0.60, 0.69, 74, 83))
        else:
            chosen.append((gp, "round5_7", 0.56, 0.65, 70, 79))

    shaped_ids = set()
    for p, tier, lo, hi, pot_lo, pot_hi in chosen:
        shaped_ids.add(id(p))
        _apply_shaped_player(
            p,
            tier=tier,
            lo=lo,
            hi=hi,
            pot_lo=pot_lo,
            pot_hi=pot_hi,
            rng=rng,
            code_by_id=code_by_id,
            rng_inst=rng,
        )
        _stamp_pipeline_shaped(p, dy)

    upside_pct = float(depth_slots.get("upside_pct", 0.08))
    for p in draft_age:
        if id(p) in shaped_ids:
            continue
        _assign_residual_dev_potential(p, rng, depth_label=depth_label, upside_pct=upside_pct)
        _stamp_pipeline_shaped(p, dy)

    try:
        _sprinkle_draft_oddballs(league, draft_age)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    try:
        setattr(league, "_pipeline_shaped_for_draft_year", dy)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


#: Chance a draft class gets one oddball (and a smaller chance of a second).
DRAFT_ODDBALL_CLASS_CHANCE = 0.35
#: (key, backstory, dossier label, {rating prefix: delta on the 0-99 scale}, height inches or None)
_DRAFT_ODDBALLS: Tuple[Tuple[str, str, str, Dict[str, int], Optional[int]], ...] = (
    ("jamaica_rollerblades", "Learned the game on rollerblades in a Kingston parking lot; first saw real ice at 15.",
     "Oddball: rollerblade kid from Jamaica", {"skg_": 6, "def_": -3}, None),
    ("figure_skater", "A former junior figure skating champion who switched sports after 'one too many sequins'.",
     "Oddball: ex-figure skater", {"skg_": 8, "phy_": -4}, None),
    ("giant", "Six-foot-nine and still growing. His skates are custom-made by a boat company.",
     "Oddball: the 6'9\" kid", {"phy_": 7, "skg_": -4}, 81),
    ("farm_kid", "Built his own rink on the family dairy farm; claims the cows taught him positioning.",
     "Oddball: dairy-farm rink rat", {"def_": 4, "iqm_": 3, "off_": -3}, None),
    ("chess_prodigy", "A nationally ranked chess player who reads plays three passes ahead.",
     "Oddball: chess prodigy", {"iqm_": 6, "pm_": 3, "phy_": -5}, None),
)


def _rating_shift(p: Any, deltas: Dict[str, int]) -> None:
    ratings = getattr(p, "ratings", None)
    if not isinstance(ratings, dict):
        return
    for key in list(ratings.keys()):
        for prefix, d in deltas.items():
            if str(key).startswith(prefix):
                try:
                    v = float(ratings[key])
                    if v <= 1.5:
                        ratings[key] = max(0.20, min(0.95, v + d / 99.0))
                    else:
                        ratings[key] = int(max(20, min(95, round(v + d))))
                except (TypeError, ValueError):
                    pass
    inval = getattr(p, "_invalidate_ovr_memo", None)
    if callable(inval):
        inval()
    try:
        from app.sim_engine.entities.player import persist_recomputed_ovr

        persist_recomputed_ovr(p)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _tag_oddball(p: Any, key: str, backstory: str, label: str) -> None:
    try:
        setattr(p, "goofy_backstory_key", key)
        setattr(p, "goofy_backstory", backstory)
        labels = list(getattr(p, "dossier_labels", None) or [])
        if label not in labels:
            labels.append(label)
        setattr(p, "dossier_labels", labels[-8:])
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _sprinkle_draft_oddballs(league: Any, draft_age: List[Any]) -> int:
    """Goofy: rare draft prospects with wild backstories and real, quirky rating shifts.

    Uses its own crc32-seeded RNG so the main generation stream is untouched.
    ``league.goofy_events_enabled`` mirrors the franchise session setting.
    """
    if not getattr(league, "goofy_events_enabled", True) or len(draft_age) < 20:
        return 0
    import zlib

    pool = sorted(
        (p for p in draft_age if getattr(getattr(p, "identity", None), "position", None) != Position.G
         and not getattr(p, "goofy_backstory_key", None) and not getattr(p, "is_transcendent", False)),
        key=lambda p: str(getattr(p, "id", "")),
    )
    if len(pool) < 4:
        return 0
    seed = "|".join(str(getattr(p, "id", "")) for p in pool[:6])
    r = random.Random(zlib.crc32(f"goofy_draft|{seed}".encode()))
    if r.random() >= DRAFT_ODDBALL_CLASS_CHANCE:
        return 0
    count = 2 if r.random() < 0.20 else 1
    made = 0
    for _ in range(count):
        if r.random() < 0.20 and len(pool) >= 2:
            # Twins: same surname, both a touch better at finding each other.
            a, b = r.sample(pool, 2)
            pool = [p for p in pool if p is not a and p is not b]
            ia, ib = getattr(a, "identity", None), getattr(b, "identity", None)
            surname = str(getattr(ia, "name", "") or "").split(" ")[-1]
            first_b = str(getattr(ib, "name", "") or "").split(" ")[0]
            if surname and first_b and ib is not None:
                try:
                    ib.name = f"{first_b} {surname}"
                    ib.age = getattr(ia, "age", getattr(ib, "age", 18))
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
            story = "Twin brothers who insist on being drafted together. They finish each other's passes."
            for p, other in ((a, b), (b, a)):
                _rating_shift(p, {"pm_": 3})
                _tag_oddball(p, "twins", story, "Oddball: one of the twins")
                try:
                    setattr(p, "twin_of", str(getattr(other, "id", "")))
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
            made += 2
            continue
        p = r.choice(pool)
        pool.remove(p)
        key, story, label, deltas, height = r.choice(_DRAFT_ODDBALLS)
        _rating_shift(p, deltas)
        _tag_oddball(p, key, story, label)
        if height:
            ident = getattr(p, "identity", None)
            for attr in ("height_in", "height_inches"):
                if ident is not None and hasattr(ident, attr):
                    try:
                        setattr(ident, attr, int(height))
                    except Exception:
                        _swallowed_log.debug("suppressed exception", exc_info=True)
        made += 1
    return made


_LATER_ROUND_LIFT_SKIP = frozenset({
    "transcendent", "franchise", "elite", "top", "round1_tail",
})


def lift_shallow_draft_depth(
    league: Any,
    rng: Optional[random.Random] = None,
    *,
    draft_year: Optional[int] = None,
) -> int:
    """Raise later-round first-year eligibles left at ~40-55 overall.

    Stars and the top of round 1 are left alone. Runs once per upcoming draft year.
    """
    if league is None:
        return 0
    dy = int(draft_year or _upcoming_draft_year(league))
    lifted_years = getattr(league, "_later_round_lift_years", None)
    if not isinstance(lifted_years, set):
        lifted_years = set()
        try:
            setattr(league, "_later_round_lift_years", lifted_years)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    if dy in lifted_years:
        return 0
    anchor = int(getattr(league, "season_start_year", 0) or getattr(league, "season_year", 0) or dy - 1)
    rng_inst = rng if rng is not None else random.Random(anchor * 7919 + dy)
    lifted = 0
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "JUNIOR")
        max_age = 24 if code.upper() == "NCAA" else 20
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "retired", False) or getattr(p, "drafted", False):
                    continue
                if not _is_first_year_draft_eligible(p, dy):
                    continue
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 99) or 99) if ident else 99
                if age < 17 or age > max_age:
                    continue
                tier = str(getattr(p, "pipeline_tier", "") or "")
                if tier in _LATER_ROUND_LIFT_SKIP:
                    continue
                ovr = _player_ovr_frac(p)
                ratings = getattr(p, "ratings", None)
                pot = 0
                if isinstance(ratings, dict):
                    try:
                        pot = int(float(ratings.get("dev_potential") or 0))
                    except (TypeError, ValueError):
                        pot = 0
                if ovr >= 0.60 and pot >= 70:
                    continue
                roll = rng_inst.random()
                if roll < 0.18:
                    lo, hi, plo, phi, shaped = 0.62, 0.72, 80, 90, "hidden_upside"
                elif roll < 0.50:
                    lo, hi, plo, phi, shaped = 0.60, 0.69, 75, 84, "round3_4"
                else:
                    lo, hi, plo, phi, shaped = 0.56, 0.66, 70, 80, "round5_7"
                _apply_shaped_player(
                    p,
                    tier=shaped,
                    lo=lo,
                    hi=hi,
                    pot_lo=plo,
                    pot_hi=phi,
                    rng=rng_inst,
                    code_by_id={id(p): code},
                    rng_inst=rng_inst,
                )
                lifted += 1
    try:
        lifted_years.add(dy)
        setattr(league, "_later_round_lift_years", lifted_years)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return lifted


def prepare_draft_class_for_season(
    league: Any,
    rng: random.Random,
    *,
    season_year: int,
    pos_fn: Any = None,
) -> Dict[str, Any]:
    """One offseason mutation pass: shape new first-years, lift depth, backfill pool size.

    The live draft board builder must stay read-only — call this before rankings.
    """
    season_year = int(season_year)
    draft_year = season_year + 1
    out: Dict[str, Any] = {"season_year": season_year, "draft_year": draft_year}
    if league is None:
        out["ok"] = False
        return out
    if int(getattr(league, "_draft_class_prepared_year", 0) or 0) == draft_year:
        out["ok"] = True
        out["skipped"] = True
        return out

    set_spawn_as_of_year(season_year)
    try:
        setattr(league, "season_start_year", season_year)
        setattr(league, "season_year", season_year)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    if not getattr(league, "_draft_pipeline_ovr_repaired", False):
        repair_undervalued_draft_pipeline_stars(league, rng)
        try:
            setattr(league, "_draft_pipeline_ovr_repaired", True)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    if not getattr(league, "_junior_dobs_reanchored", False):
        reanchor_generated_junior_dobs(league, season_year)

    _shape_draft_class_pipeline(league, rng, draft_year=draft_year)
    out["lifted"] = lift_shallow_draft_depth(league, rng, draft_year=draft_year)

    from services.draft_ranking_logic import (
        DRAFT_ELIGIBLE_TARGET,
        MIN_BOARD_GOALIES,
        backfill_draft_eligible_goalies,
        backfill_draft_eligible_skaters,
        collect_goalie_pipeline_stats,
        count_draft_eligible_players,
    )

    if pos_fn is None:
        def pos_fn(p: Any) -> str:
            ident = getattr(p, "identity", None)
            raw = getattr(ident, "position", None) if ident else getattr(p, "position", "C")
            return str(getattr(raw, "value", raw) or "C").upper()

    pipeline_stats = collect_goalie_pipeline_stats(league, age_max=20, pos_fn=pos_fn)
    if pipeline_stats.get("draft_eligible_dev_goalies", 0) < MIN_BOARD_GOALIES:
        out["goalies_backfilled"] = backfill_draft_eligible_goalies(
            league,
            rng,
            MIN_BOARD_GOALIES - int(pipeline_stats.get("draft_eligible_dev_goalies", 0)),
        )
    eligible = count_draft_eligible_players(league)
    if eligible < DRAFT_ELIGIBLE_TARGET:
        out["skaters_backfilled"] = backfill_draft_eligible_skaters(
            league, rng, DRAFT_ELIGIBLE_TARGET - eligible
        )

    try:
        setattr(league, "_draft_class_prepared_year", draft_year)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    out["ok"] = True
    return out


def repair_undervalued_draft_pipeline_stars(league: Any, rng: Optional[random.Random] = None) -> int:
    """Reshape franchise/elite/top stars that still carry the old ~50 OVR draft bands.

    Returns the number of players reshaped. Safe to call repeatedly — no-ops once
    current ability is at or above the tier floor.
    """
    if league is None:
        return 0
    rng_inst = rng if rng is not None else random.Random(42)
    code_by_id: dict = {}
    repaired = 0
    for block in getattr(league, "development_leagues", None) or []:
        code = str(block.get("league_code") or "JUNIOR")
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "retired", False):
                    continue
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 99) or 99) if ident else 99
                if age > 20:
                    continue
                tier = str(getattr(p, "pipeline_tier", "") or "").lower()
                band = _PIPELINE_OVR_REPAIR.get(tier)
                if not band:
                    continue
                floor, lo, hi = band
                cur = _player_ovr_frac(p)
                if cur + 1e-6 >= floor:
                    continue
                # Preserve high ceilings that were already rolled for these stars.
                pot = int(getattr(p, "ratings", {}) and p.ratings.get("dev_potential") or 0)
                if tier == "transcendent":
                    pot_lo, pot_hi = 99, 99
                elif tier == "franchise":
                    pot_lo, pot_hi = max(91, pot), max(97, pot, 91)
                elif tier == "elite":
                    pot_lo, pot_hi = max(88, pot), max(94, pot, 88)
                else:
                    pot_lo, pot_hi = max(85, pot), max(91, pot, 85)
                code_by_id[id(p)] = code
                _apply_shaped_player(
                    p,
                    tier=tier,
                    lo=lo,
                    hi=hi,
                    pot_lo=pot_lo,
                    pot_hi=pot_hi,
                    rng=rng_inst,
                    code_by_id=code_by_id,
                    rng_inst=rng_inst,
                )
                repaired += 1
    return repaired


def ensure_board_prospect_ovr_floors(
    board_rows: List[Dict[str, Any]],
    *,
    player_by_key: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
) -> int:
    """Guarantee top-of-board prospects have realistic NHL-scale current OVR.

    Ranking can promote a raw ~50 OVR junior to #1 on production alone. Reshape
    those players in place so draft-day ability matches their board slot.
    """
    if not board_rows:
        return 0
    rng_inst = rng if rng is not None else random.Random(42)
    floors = (
        (3, "franchise", 0.79, 0.80, 0.86, 92, 98),
        (10, "elite", 0.75, 0.76, 0.83, 88, 94),
        (20, "top", 0.71, 0.73, 0.80, 85, 91),
        (32, "round1_tail", 0.68, 0.70, 0.77, 82, 88),
    )
    repaired = 0
    lookup = player_by_key or {}
    for idx, row in enumerate(board_rows):
        rank = idx + 1
        band = None
        for max_rank, tier, floor, lo, hi, pot_lo, pot_hi in floors:
            if rank <= max_rank:
                band = (tier, floor, lo, hi, pot_lo, pot_hi)
                break
        if not band:
            break
        tier, floor, lo, hi, pot_lo, pot_hi = band
        key = str(row.get("key") or row.get("id") or "")
        p = row.get("_player") or lookup.get(key)
        cur99 = 0.0
        try:
            cur99 = float(row.get("true_ovr") or 0)
        except Exception:
            cur99 = 0.0
        if p is not None:
            cur99 = max(cur99, _player_ovr_frac(p) * 99.0)
        # Do not NHL-floor underagers — year-roll inject bugs already over-rated them.
        try:
            age = int((getattr(getattr(p, "identity", None), "age", None) if p is not None else None) or row.get("age") or 18)
        except Exception:
            age = int(row.get("age") or 18)
        if age < 17:
            continue
        if cur99 + 1e-6 >= floor * 99.0:
            continue
        if p is None:
            # No live player handle — at least lift the board/profile numbers.
            target99 = round(((lo + hi) / 2.0) * 99.0, 1)
            row["true_ovr"] = target99
            row["current_ovr_estimate"] = target99
            row["pipeline_tier"] = tier
            repaired += 1
            continue
        code_by_id = {id(p): str(row.get("league_code") or "JUNIOR")}
        existing_pot = 0
        try:
            existing_pot = int((getattr(p, "ratings", {}) or {}).get("dev_potential") or 0)
        except Exception:
            existing_pot = 0
        _apply_shaped_player(
            p,
            tier=tier,
            lo=lo,
            hi=hi,
            pot_lo=max(pot_lo, existing_pot),
            pot_hi=max(pot_hi, existing_pot, pot_lo),
            rng=rng_inst,
            code_by_id=code_by_id,
            rng_inst=rng_inst,
        )
        new99 = round(_player_ovr_frac(p) * 99.0, 1)
        row["true_ovr"] = new99
        row["current_ovr_estimate"] = new99
        row["pipeline_tier"] = tier
        row["_ovr_floor_repaired"] = True
        repaired += 1
    return repaired


def _tick_ratings(rng: random.Random, player: Any, *, overseas: bool, junior: bool) -> None:
    if rng.random() > (0.0022 if overseas else 0.00065 if not junior else 0.0011):
        return
    keys = list(player.ratings.keys())
    if not keys:
        return
    k = rng.choice(keys)
    cap = 93 if junior else 90 if overseas else 88
    bump = rng.choice([1, 1, 1, 2])
    cur = int(float(player.ratings.get(k, 50)))
    player.ratings[k] = clamp_rating(cur + bump)
    if int(float(player.ratings[k])) > cap:
        player.ratings[k] = cap


def _tick_extra_league_development_once(sim: Any, rng: random.Random) -> None:
    league = getattr(sim, "league", None)
    if league is None:
        return
    for tm in getattr(league, "teams", None) or []:
        for p in getattr(tm, "ahl_roster", None) or []:
            _tick_ratings(rng, p, overseas=False, junior=False)
        for p in getattr(tm, "echl_roster", None) or []:
            _tick_ratings(rng, p, overseas=False, junior=False)
    for p in getattr(league, "free_agents", None) or []:
        meta = getattr(p, "_franchise_assignment", None) or {}
        if meta.get("overseas"):
            continue
        _tick_ratings(rng, p, overseas=False, junior=False)
    for p in getattr(league, "overseas_free_agents", None) or []:
        _tick_ratings(rng, p, overseas=True, junior=False)
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                if getattr(p, "pipeline_bust", False):
                    setattr(p, "dev_type", "bust")
                elif getattr(p, "pipeline_steal", False):
                    setattr(p, "dev_type", "elite")
                _tick_ratings(rng, p, overseas=False, junior=True)


def tick_extra_league_development(sim: Any, rng: random.Random, days: int = 1) -> None:
    """Apply ``days`` independent daily rating rolls (Bernoulli per player per day)."""
    n = max(1, int(days or 1))
    for _ in range(n):
        _tick_extra_league_development_once(sim, rng)


def maybe_spawn_overseas_star(league: Any, rng: random.Random, season_year: int) -> Optional[Any]:
    """Once per summer: ~25% chance one established European pro (78-86) wants an NHL deal."""
    if league is None:
        return None
    done = getattr(league, "_overseas_star_rolls", None)
    if not isinstance(done, dict):
        done = {}
    key = str(int(season_year or 0))
    if key in done:
        return None
    done[key] = False
    try:
        league._overseas_star_rolls = done
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    if rng.random() >= 0.25:
        return None
    used_names: set = set()
    for p in list(getattr(league, "players", None) or []) + list(getattr(league, "overseas_free_agents", None) or []):
        nm = str(getattr(getattr(p, "identity", None), "name", "") or "")
        if nm:
            used_names.add(nm)
    pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D])
    p = _spawn_player(
        rng, pos=pos, ovr_lo=0.78, ovr_hi=0.86, age_lo=25, age_hi=30,
        used_names=used_names, league_players=list(getattr(league, "players", None) or []),
        pool_context="overseas_elite",
    )
    p.context.current_team_id = "OVERSEAS"
    _set_assignment(
        p, level="ufa", overseas=True,
        overseas_league=rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"]),
    )
    pool = list(getattr(league, "overseas_free_agents", None) or [])
    pool.append(p)
    league.overseas_free_agents = pool
    done[key] = True
    return p


def _euro_display_ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    try:
        v = float(fn() if callable(fn) else fn or 0)
    except Exception:
        return 0.0
    return v * 99.0 if v <= 1.5 else v


def _decorate_euro_depth_free_agent(player: Any, rng: random.Random) -> None:
    """60–75 overall depth with a mostly modest ceiling and a fake European year."""
    ovr = int(round(_euro_display_ovr(player)))
    roll = rng.random()
    if roll < 0.02:
        pot = rng.randint(90, 96)
    elif roll < 0.08:
        pot = rng.randint(82, 89)
    elif roll < 0.22:
        pot = rng.randint(76, 82)
    else:
        pot = rng.randint(max(55, ovr - 4), max(ovr, ovr + rng.randint(0, 6)))
    pot = max(pot, ovr)
    pot = min(99, pot)
    ratings = getattr(player, "ratings", None)
    if isinstance(ratings, dict):
        ratings["dev_potential"] = pot
    try:
        player.potential = pot
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    # Most sit at the league minimum. Some are priced under it as pure cap savers.
    if rng.random() < 0.22:
        aav = round(rng.uniform(0.70, 0.84), 2)
    else:
        aav = 0.85
    contract = {
        "type": "STANDARD",
        "contract_type": "STANDARD",
        "years": 1,
        "years_remaining": 1,
        "aav_m": aav,
        "cap_hit_m": aav,
        "salary_m": aav,
        "rights_status": "UFA",
        "source": "european_depth",
    }
    try:
        player.contract = contract
        player.aav_m = aav
        player.cap_hit_m = aav
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    league_name = rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"])
    gp = rng.randint(36, 58)
    goals = max(0, int((ovr - 52) * rng.uniform(0.12, 0.42)))
    if str(getattr(getattr(player, "identity", None), "position", "") or "").upper() == "G":
        goals = rng.randint(14, 32)
    assists = int(goals * rng.uniform(0.6, 1.6)) if goals else rng.randint(4, 18)
    try:
        player.games_played = gp
        player.goals = goals
        player.assists = assists
        player.season_stats = {
            "gp": gp,
            "g": goals,
            "a": assists,
            "pts": goals + assists,
            "league": league_name,
        }
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    _set_assignment(
        player,
        level="ufa",
        overseas=True,
        overseas_league=league_name,
        euro_gp=gp,
        euro_g=goals,
        euro_a=assists,
    )


def ensure_overseas_fa_pool(
    league: Any,
    rng: random.Random,
    *,
    min_count: int = 120,
    min_goalies: int = 12,
) -> int:
    """Top up the overseas free-agent runway.

    The wire used to fall off from NHL regulars to the mid-50s. This keeps a
    few hundred Europeans in the 60–75 band, with varied potential and some
    deals priced under the league minimum.
    """
    if league is None:
        return 0
    pool = list(getattr(league, "overseas_free_agents", None) or [])
    in_band = sum(1 for p in pool if 60.0 <= _euro_display_ovr(p) <= 76.0)
    need = max(0, 280 - in_band, int(min_count) - len(pool))
    g_have = sum(
        1
        for p in pool
        if str(getattr(getattr(p, "identity", None), "position", "") or "").upper() in ("G", "GOALIE")
        or str(getattr(getattr(getattr(p, "identity", None), "position", None), "value", "") or "").upper() == "G"
    )
    need_g = max(0, int(min_goalies) - g_have)
    if need <= 0 and need_g <= 0:
        league.overseas_free_agents = pool
        try:
            from services.roster_compliance import fill_league_roster_minimums

            fill_league_roster_minimums(league, rng)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return 0

    used_names: set = set()
    for p in list(getattr(league, "players", None) or []) + pool + list(getattr(league, "free_agents", None) or []):
        ident = getattr(p, "identity", None)
        nm = str(getattr(ident, "name", "") or "")
        if nm:
            used_names.add(nm)
    league_players = list(getattr(league, "players", None) or [])
    added = 0
    total_spawn = need + need_g
    for i in range(total_spawn):
        force_g = i < need_g
        pos = Position.G if force_g else rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.G])
        lo, hi = (0.606, 0.758) if pos != Position.G else (0.596, 0.740)
        p = _spawn_player(
            rng,
            pos=pos,
            ovr_lo=lo,
            ovr_hi=hi,
            age_lo=23,
            age_hi=33,
            used_names=used_names,
            league_players=league_players,
            pool_context="overseas",
        )
        p.context.current_team_id = "OVERSEAS"
        _decorate_euro_depth_free_agent(p, rng)
        pool.append(p)
        added += 1
    league.overseas_free_agents = pool
    try:
        from services.roster_compliance import fill_league_roster_minimums

        fill_league_roster_minimums(league, rng)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return added


def count_pool_players(league: Any) -> Dict[str, int]:
    nhl_sk = sum(len(getattr(t, "roster", None) or []) for t in getattr(league, "teams", None) or [])
    ahl_sk = sum(len(getattr(t, "ahl_roster", None) or []) for t in getattr(league, "teams", None) or [])
    echl_sk = sum(len(getattr(t, "echl_roster", None) or []) for t in getattr(league, "teams", None) or [])
    return {
        "nhl_contracted": nhl_sk,
        "ahl_contracted": ahl_sk,
        "echl_contracted": echl_sk,
        "free_agents": len(getattr(league, "free_agents", None) or []),
        "overseas": len(getattr(league, "overseas_free_agents", None) or []),
        "junior_skaters": sum(
            len(tm.get("players") or [])
            for blk in getattr(league, "development_leagues", None) or []
            for tm in blk.get("teams") or []
        ),
    }
