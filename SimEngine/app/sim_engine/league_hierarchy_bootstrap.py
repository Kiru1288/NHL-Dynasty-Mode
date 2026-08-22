"""
Full organizational depth: NHL org AHL/ECHL, free agents, overseas tracking,
and junior / college / European development leagues.

Bootstrapped when a franchise session starts (see backend franchise_sim).
Out-of-league players receive light per-day development ticks during advance_day.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    clamp_rating,
    assign_skater_archetype,
    random_height_cm,
    enforce_minimum_player_ovr,
    get_ovr_floor_for_pool,
)
from app.sim_engine.generation.name_generator import generate_human_identity
from app.sim_engine.generation.prospect_body import (
    apply_body_tradeoffs_to_ratings,
    generate_position_height_cm,
    generate_realistic_weight_kg,
)

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
_CHL_AGE_TABLE = ((16, 17, 18, 19, 20), (12, 28, 36, 16, 8))
_USHL_AGE_TABLE = ((16, 17, 18, 19), (14, 34, 38, 14))
_NCAA_AGE_TABLE = ((18, 19, 20, 21, 22, 23, 24), (12, 18, 18, 16, 14, 12, 10))
_EU_J_AGE_TABLE = ((16, 17, 18, 19, 20), (10, 30, 38, 14, 8))


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
                    pass
                try:
                    p.age = spawn_age
                except Exception:
                    pass
                setattr(p, "_dob_anchor_year", year)
                fixed += 1
    try:
        setattr(league, "_junior_dobs_reanchored", True)
        setattr(league, "season_start_year", year)
    except Exception:
        pass
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
    archetype = assign_skater_archetype(pos, rng)
    build_role_shaped_ratings = _build_role_shaped_ratings_cached()
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
        pass
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
    try:
        player.age = age
    except Exception:
        pass
    apply_body_tradeoffs_to_ratings(player, rng)
    floor = get_ovr_floor_for_pool(pool_context)
    if floor > 0:
        enforce_minimum_player_ovr(player, floor)

    ensure_player_chemistry_profile = _ensure_player_chemistry_profile_cached()
    if ensure_player_chemistry_profile:
        try:
            ensure_player_chemistry_profile(player, rng)
        except Exception:
            pass

    ensure_player_headshot = _ensure_player_headshot_cached()
    if ensure_player_headshot:
        try:
            ensure_player_headshot(player)
        except Exception:
            pass

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


def bootstrap_full_league_hierarchy(league: Any, rng: random.Random, season_year: Optional[int] = None) -> None:
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
        pass

    if not hasattr(league, "players") or league.players is None:
        league.players = []
    league_players: List[Any] = list(league.players)
    used_names: set = set()
    for p in league_players:
        ident = getattr(p, "identity", None)
        if ident and getattr(ident, "name", None):
            used_names.add(str(ident.name))

    # --- Affiliate rosters per NHL org ---
    AHL_F, AHL_D, AHL_G = 14, 8, 2
    ECHL_F, ECHL_D, ECHL_G = 11, 5, 2

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

        ahl_slots = _positions_for_block(rng, forwards=AHL_F, defense=AHL_D, goalies=AHL_G)
        # Leave room for preserved real NHL overflow so affiliates aren't bloated.
        generated_ahl_budget = max(0, len(ahl_slots) - len(preserved_ahl))
        for pos in ahl_slots[:generated_ahl_budget]:
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
                pass
            try:
                p.context.current_team_id = f"AHL_{tid}"
            except Exception:
                pass
            _set_assignment(p, org_nhl_team_id=tid, level="ahl", club=_team_label(team))
            if p not in team.ahl_roster:
                team.ahl_roster.append(p)

        echl_slots = _positions_for_block(rng, forwards=ECHL_F, defense=ECHL_D, goalies=ECHL_G)
        generated_echl_budget = max(0, len(echl_slots) - len(preserved_echl))
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
                pass
            if p not in team.echl_roster:
                team.echl_roster.append(p)

    # --- Free agents (NHL-contract eligible pool) ---
    league.free_agents = []
    for _ in range(520):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.D, Position.G])
        lo, hi = (0.38, 0.58) if pos != Position.G else (0.45, 0.62)
        age_lo, age_hi = (22, 34)
        p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=age_lo, age_hi=age_hi, used_names=used_names, league_players=league_players, pool_context="ufa")
        p.context.current_team_id = "UFA"
        _set_assignment(p, level="ufa", overseas=False)
        league.free_agents.append(p)

    # --- Overseas / KHL-tier style unsigned runway ---
    league.overseas_free_agents = []
    for _ in range(220):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.G])
        lo, hi = (0.40, 0.62) if pos != Position.G else (0.48, 0.70)
        p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=23, age_hi=32, used_names=used_names, league_players=league_players, pool_context="overseas")
        p.context.current_team_id = "OVERSEAS"
        _set_assignment(p, level="ufa", overseas=True, overseas_league=rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"]))
        league.overseas_free_agents.append(p)

    # --- Junior / NCAA style development leagues (real club names per league) ---
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
                    pass
                roster.append(p)
            teams_out.append({"team_id": tid_j, "name": club_name, "city": city, "players": roster})
        display = str(LEAGUE_REGISTRY.get(code, {}).get("display") or title)
        dev.append({"league_code": code, "league_name": display, "teams": teams_out})

    _add_league("CHL_OHL", "OHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("CHL_WHL", "WHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("CHL_QMJHL", "QMJHL", 12, 7, 3, 16, 20, 0.32, 0.52)
    _add_league("USHL", "USHL", 12, 6, 3, 16, 19, 0.30, 0.48)
    _add_league("NCAA", "NCAA", 13, 7, 3, 18, 24, 0.34, 0.55)

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

    _shape_draft_class_pipeline(league, rng)

    all_new: List[Any] = []
    for team in teams:
        all_new.extend(team.ahl_roster)
        all_new.extend(team.echl_roster)
    all_new.extend(league.free_agents)
    all_new.extend(league.overseas_free_agents)
    for block in dev:
        for tm in block.get("teams") or []:
            all_new.extend(tm.get("players") or [])

    _init_chars(league, rng, all_new)


# Star-power tiers: (label, weight, franchise/elite/top slot counts, top target ovr)
# Current ability is NHL-scale. Top picks should look like near-NHL talents a GM
# would promote within 1–3 years (~78–86 current), not mid-60s juniors.
_CLASS_STRENGTH_TIERS = [
    ("weak", 0.20, 1, 3, 6, (0.76, 0.81)),
    ("average", 0.40, 1, 4, 8, (0.78, 0.84)),
    ("strong", 0.24, 2, 5, 9, (0.80, 0.86)),
    ("elite", 0.12, 2, 6, 10, (0.82, 0.88)),
    ("generational", 0.04, 3, 6, 12, (0.84, 0.90)),
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
    ("weak", 0.18, {"round1_tail": 28, "nhl_floor": 8, "round2": 16, "round3_4": 24, "round5_7": 30, "upside_pct": 0.06}),
    ("average", 0.42, {"round1_tail": 38, "nhl_floor": 16, "round2": 28, "round3_4": 40, "round5_7": 48, "upside_pct": 0.10}),
    ("strong", 0.28, {"round1_tail": 48, "nhl_floor": 24, "round2": 36, "round3_4": 52, "round5_7": 62, "upside_pct": 0.14}),
    ("elite", 0.12, {"round1_tail": 55, "nhl_floor": 32, "round2": 44, "round3_4": 64, "round5_7": 72, "upside_pct": 0.18}),
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
            pass
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
            pass
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
        pass
    if rng_inst.random() < 0.12:
        setattr(p, "character_concerns", True)
    code = code_by_id.get(id(p), "JUNIOR")
    try:
        from app.sim_engine.generation.prospect_league_scoring import initialize_prospect_season

        initialize_prospect_season(p, code, rng=rng_inst, force=True, preserve_actual=True)
    except Exception:
        pass


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
        pot = rng.randint(70, 79)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.38:
        pot = rng.randint(66, 75)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.58:
        pot = rng.randint(62, 71)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.76:
        pot = rng.randint(58, 67)
        setattr(player, "pipeline_tier", "pool")
    elif roll < 0.90:
        pot = rng.randint(54, 63)
        setattr(player, "pipeline_tier", "pool")
    else:
        pot = rng.randint(50, 59)
        setattr(player, "pipeline_tier", "pool")

    pot = int(max(ovr99 + 2, min(99, pot + depth_shift)))
    ratings["dev_potential"] = pot


def _split_first_year_overage(players: List[Any], rng: random.Random) -> Tuple[List[Any], List[Any]]:
    fy = [p for p in players if _ident_age(p) <= 18]
    og = [p for p in players if _ident_age(p) >= 19]
    rng.shuffle(fy)
    rng.shuffle(og)
    return fy, og


def _take_age_biased(
    fy: List[Any],
    og: List[Any],
    rng: random.Random,
    first_year_p: float,
) -> Optional[Any]:
    if fy and (not og or rng.random() < float(first_year_p)):
        return fy.pop(0)
    if og:
        return og.pop(0)
    if fy:
        return fy.pop(0)
    return None


def _shape_draft_class_pipeline(league: Any, rng: random.Random) -> None:
    """
    Post-spawn pass: builds a full draft-class talent pool with star power AND depth.

    Star-power tiers control the very top of the class. Depth-quality tiers control
    how many NHL-calibre ceilings exist in rounds 2–7. Residual dev_potential is
    assigned to every remaining draft-age player so the class is never a flat cliff
    after pick ~15.
    """
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
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 99) or 99) if ident else 99
                if 17 <= age <= 20:
                    draft_age.append(p)
                    code_by_id[id(p)] = code
    if len(draft_age) < 20:
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
            ("weak", 0.25, -1.5),
            ("normal", 0.55, 0.0),
            ("strong", 0.15, 1.2),
            ("elite", 0.045, 2.5),
            ("generational", 0.005, 4.5),
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
    fy_sk, og_sk = _split_first_year_overage(skaters, rng)
    fy_g, og_g = _split_first_year_overage(goalies, rng)

    def _next_skater(p: float = 0.92) -> Optional[Any]:
        return _take_age_biased(fy_sk, og_sk, rng, p)

    def _next_goalie(p: float = 0.90) -> Optional[Any]:
        return _take_age_biased(fy_g, og_g, rng, p)

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
        chosen.append((sp, "franchise", top_range[0], top_range[1], 88, 97))
    if generational_goalie is not None:
        chosen.append((generational_goalie, "franchise", top_range[0], top_range[1], 90, 98))
        setattr(generational_goalie, "generational_goalie", True)

    _goalie_shape = {
        "weak": (0, 1),
        "normal": (0, 3),
        "strong": (1, 3),
        "elite": (1, 4),
        "generational": (1, 2),
    }
    n_elite_g, n_top_g = _goalie_shape.get(g_label, (0, 2))
    if n_elite_g > 0:
        for _ in range(n_elite_g):
            gp = _next_goalie(0.90)
            if gp is None:
                break
            chosen.append((gp, "elite", top_range[0] - 0.03, top_range[1] - 0.02, 84, 92))
    for _ in range(n_top_g):
        gp = _next_goalie(0.88)
        if gp is None:
            break
        chosen.append((gp, "top", top_range[0] - 0.08, top_range[0] - 0.02, 76, 88))

    for _ in range(n_elite):
        sp = _next_skater(0.92)
        if sp is None:
            break
        chosen.append((sp, "elite", top_range[0] - 0.05, top_range[1] - 0.05, 82, 92))
    for _ in range(n_top):
        sp = _next_skater(0.90)
        if sp is None:
            break
        chosen.append((sp, "top", top_range[0] - 0.09, top_range[1] - 0.09, 76, 86))

    n_round1_tail = int(depth_slots.get("round1_tail", 38))
    for _ in range(n_round1_tail):
        sp = _next_skater(0.86)
        if sp is None:
            break
        chosen.append((sp, "round1_tail", top_range[0] - 0.12, top_range[0] - 0.06, 70, 82))

    n_nhl_floor = int(depth_slots.get("nhl_floor", 16))
    for _ in range(n_nhl_floor):
        sp = _next_skater(0.78)
        if sp is None:
            break
        # Solid NHL-bound depth — below top-of-class current ability.
        chosen.append((sp, "nhl_floor", 0.58, 0.65, 72, 80))

    n_round2 = int(depth_slots.get("round2", 28))
    for _ in range(n_round2):
        sp = _next_skater(0.70)
        if sp is None:
            break
        chosen.append((sp, "round2", 0.54, 0.62, 74, 84))

    n_round3_4 = int(depth_slots.get("round3_4", 40))
    for _ in range(n_round3_4):
        sp = _next_skater(0.58)
        if sp is None:
            break
        chosen.append((sp, "round3_4", 0.50, 0.58, 70, 82))

    n_round5_7 = int(depth_slots.get("round5_7", 48))
    upside_pct = float(depth_slots.get("upside_pct", 0.08))
    for _ in range(n_round5_7):
        sp = _next_skater(0.48)
        if sp is None:
            break
        if rng.random() < upside_pct * 0.55:
            chosen.append((sp, "hidden_upside", 0.46, 0.56, 80, 92))
        elif rng.random() < 0.22:
            chosen.append((sp, "round5_7", 0.48, 0.56, 72, 84))
        else:
            chosen.append((sp, "round5_7", 0.42, 0.52, 65, 78))

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

    upside_pct = float(depth_slots.get("upside_pct", 0.08))
    for p in draft_age:
        if id(p) in shaped_ids:
            continue
        _assign_residual_dev_potential(p, rng, depth_label=depth_label, upside_pct=upside_pct)


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
                    pot_lo, pot_hi = max(88, pot), max(97, pot, 88)
                elif tier == "elite":
                    pot_lo, pot_hi = max(82, pot), max(92, pot, 82)
                else:
                    pot_lo, pot_hi = max(76, pot), max(86, pot, 76)
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
        (3, "franchise", 0.78, 0.79, 0.86, 90, 98),
        (10, "elite", 0.74, 0.74, 0.82, 86, 94),
        (20, "top", 0.70, 0.70, 0.78, 80, 90),
        (32, "round1_tail", 0.66, 0.66, 0.74, 76, 86),
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


def tick_extra_league_development(sim: Any, rng: random.Random) -> None:
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


def ensure_overseas_fa_pool(
    league: Any,
    rng: random.Random,
    *,
    min_count: int = 120,
    min_goalies: int = 12,
) -> int:
    """Top up the overseas / Euro free-agent runway so the Wire is never empty.

    Real-NHL franchises and long saves can drain this pool; the FA Wire should
    still list overseas talent alongside unsigned summer UFAs. Always keeps a
    replacement-level goalie lane so clubs can sign a netminder in July / camp.
    """
    if league is None:
        return 0
    pool = list(getattr(league, "overseas_free_agents", None) or [])
    need = max(0, int(min_count) - len(pool))
    g_have = sum(
        1
        for p in pool
        if str(getattr(getattr(p, "identity", None), "position", "") or "").upper() in ("G", "GOALIE")
        or str(getattr(getattr(getattr(p, "identity", None), "position", None), "value", "") or "").upper() == "G"
    )
    need_g = max(0, int(min_goalies) - g_have)
    if need <= 0 and need_g <= 0:
        league.overseas_free_agents = pool
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
        lo, hi = (0.40, 0.62) if pos != Position.G else (0.48, 0.70)
        p = _spawn_player(
            rng,
            pos=pos,
            ovr_lo=lo,
            ovr_hi=hi,
            age_lo=23,
            age_hi=32,
            used_names=used_names,
            league_players=league_players,
            pool_context="overseas",
        )
        p.context.current_team_id = "OVERSEAS"
        _set_assignment(
            p,
            level="ufa",
            overseas=True,
            overseas_league=rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"]),
        )
        pool.append(p)
        added += 1
    league.overseas_free_agents = pool
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
