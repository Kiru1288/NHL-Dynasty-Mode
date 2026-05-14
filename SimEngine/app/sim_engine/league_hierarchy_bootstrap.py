"""
Full organizational depth: NHL org AHL/ECHL, free agents, overseas tracking,
and junior / college / European development leagues.

Bootstrapped when a franchise session starts (see backend franchise_sim).
Out-of-league players receive light per-day development ticks during advance_day.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

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
)
from app.sim_engine.generation.name_generator import generate_human_identity


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
) -> Player:
    target_ovr = ovr_lo + rng.uniform(0, max(1e-6, ovr_hi - ovr_lo))
    target_ovr = max(0.35, min(0.92, target_ovr))
    base_rating = int(target_ovr * 99)
    ratings = {k: clamp_rating(base_rating + rng.randint(-4, 4)) for k in ATTRIBUTE_KEYS}
    age = rng.randint(age_lo, age_hi)
    birth_year = 2025 - age
    seed = rng.randint(1, 2_000_000_000)
    ident = generate_human_identity(rng)
    for _ in range(6):
        nm = str(getattr(ident, "full_name", "Unknown"))
        if nm not in used_names:
            used_names.add(nm)
            break
        ident = generate_human_identity(rng)
    hometown = str(ident.hometown or "Unknown")
    birth_city = hometown.split(",")[0].strip() if hometown else "Unknown"
    h_cm = random_height_cm(rng)
    w_kg = int(78 + (h_cm - 178) * 0.38 + rng.randint(-9, 11))
    w_kg = max(65, min(118, w_kg))
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
        archetype=assign_skater_archetype(pos, rng),
    )
    league_players.append(player)
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


def bootstrap_full_league_hierarchy(league: Any, rng: random.Random) -> None:
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
        for pos in ahl_slots:
            lo, hi = (0.42, 0.62) if pos != Position.G else (0.48, 0.68)
            p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=20, age_hi=28, used_names=used_names, league_players=league_players)
            p.context.current_team_id = f"AHL_{tid}"
            _set_assignment(p, org_nhl_team_id=tid, level="ahl", club=_team_label(team))
            team.ahl_roster.append(p)

        echl_slots = _positions_for_block(rng, forwards=ECHL_F, defense=ECHL_D, goalies=ECHL_G)
        for pos in echl_slots:
            lo, hi = (0.36, 0.55) if pos != Position.G else (0.42, 0.60)
            p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=21, age_hi=30, used_names=used_names, league_players=league_players)
            p.context.current_team_id = f"ECHL_{tid}"
            _set_assignment(p, org_nhl_team_id=tid, level="echl", club=_team_label(team))
            team.echl_roster.append(p)

    # --- Free agents (NHL-contract eligible pool) ---
    league.free_agents = []
    for _ in range(520):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.D, Position.G])
        lo, hi = (0.38, 0.58) if pos != Position.G else (0.45, 0.62)
        age_lo, age_hi = (22, 34)
        p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=age_lo, age_hi=age_hi, used_names=used_names, league_players=league_players)
        p.context.current_team_id = "UFA"
        _set_assignment(p, level="ufa", overseas=False)
        league.free_agents.append(p)

    # --- Overseas / KHL-tier style unsigned runway ---
    league.overseas_free_agents = []
    for _ in range(220):
        pos = rng.choice([Position.C, Position.LW, Position.RW, Position.D, Position.G])
        lo, hi = (0.40, 0.62) if pos != Position.G else (0.48, 0.70)
        p = _spawn_player(rng, pos=pos, ovr_lo=lo, ovr_hi=hi, age_lo=23, age_hi=32, used_names=used_names, league_players=league_players)
        p.context.current_team_id = "OVERSEAS"
        _set_assignment(p, level="ufa", overseas=True, overseas_league=rng.choice(["KHL", "SHL", "Liiga", "NL", "DEL", "Czech Extraliga"]))
        league.overseas_free_agents.append(p)

    # --- Junior / NCAA style development leagues (aggregated clubs) ---
    dev: List[Dict[str, Any]] = []

    city_pool = [
        "Barrie",
        "Saginaw",
        "Madison",
        "Omaha",
        "Jönköping",
        "Tampere",
        "Bern",
        "Davos",
        "Trinec",
        "Bratislava",
        "Kitchener",
        "Portland",
        "Green Bay",
        "Lethbridge",
        "Medicine Hat",
        "Rimouski",
        "Sherbrooke",
        "Notre Dame",
        "Boston",
        "Denver",
        "North Dakota",
        "Zug",
        "Ambri",
    ]

    def _add_league(code: str, title: str, n_teams: int, f: int, d: int, g: int, age_lo: int, age_hi: int, ovr_lo: float, ovr_hi: float) -> None:
        teams_out: List[Dict[str, Any]] = []
        for ti in range(n_teams):
            roster: List[Player] = []
            slots = _positions_for_block(rng, forwards=f, defense=d, goalies=g)
            city = city_pool[(hash(code) + ti) % len(city_pool)]
            tname = f"{code.replace('_', ' ')} {ti + 1}"
            tid_j = f"{code}_{ti + 1}"
            for pos in slots:
                p = _spawn_player(
                    rng,
                    pos=pos,
                    ovr_lo=ovr_lo,
                    ovr_hi=ovr_hi,
                    age_lo=age_lo,
                    age_hi=age_hi,
                    used_names=used_names,
                    league_players=league_players,
                )
                p.context.current_team_id = tid_j
                _set_assignment(p, level="junior", league_code=code, club=f"{city} {tname}")
                roster.append(p)
            teams_out.append({"team_id": tid_j, "name": f"{city} {tname}", "players": roster})
        dev.append({"league_code": code, "league_name": title, "teams": teams_out})

    _add_league("CHL_OHL", "Canadian Hockey League — OHL cluster", 5, 12, 7, 2, 17, 20, 0.32, 0.52)
    _add_league("CHL_WHL", "Canadian Hockey League — WHL cluster", 5, 12, 7, 2, 17, 20, 0.32, 0.52)
    _add_league("CHL_QMJHL", "Canadian Hockey League — QMJHL cluster", 5, 12, 7, 2, 17, 20, 0.32, 0.52)
    _add_league("USHL", "United States Hockey League", 6, 12, 6, 2, 17, 19, 0.30, 0.48)
    _add_league("NCAA", "NCAA Division I cluster", 8, 13, 7, 2, 18, 24, 0.34, 0.55)

    euro_specs = [
        ("EU_J_SHL", "Sweden J20 / junior ladder", 2, 11, 6, 2, 17, 19, 0.30, 0.50),
        ("EU_J_LIIGA", "Finland U20 junior ladder", 2, 11, 6, 2, 17, 19, 0.30, 0.50),
        ("EU_J_DEL", "Germany DNL junior", 2, 11, 6, 2, 17, 19, 0.28, 0.48),
        ("EU_J_SWISS", "Swiss Elite Jr.", 2, 11, 6, 2, 17, 19, 0.30, 0.50),
        ("EU_J_CZ", "Czech U20 extraliga junior", 2, 11, 6, 2, 17, 19, 0.28, 0.48),
        ("EU_J_SK", "Slovakia U20", 2, 11, 6, 2, 17, 19, 0.28, 0.48),
        ("EU_J_KHL_JR", "Russia / MHL style junior", 2, 11, 6, 2, 17, 20, 0.30, 0.52),
        ("EU_J_NOR", "Norway junior elite", 2, 11, 6, 2, 17, 19, 0.28, 0.46),
        ("EU_J_DEN", "Denmark U20", 2, 11, 6, 2, 17, 19, 0.28, 0.46),
        ("EU_J_AUT", "Austria junior league", 2, 11, 6, 2, 17, 19, 0.28, 0.46),
    ]
    for code, title, nt, ff, dd, gg, alo, ahi, olo, ohi in euro_specs:
        _add_league(code, title, nt, ff, dd, gg, alo, ahi, olo, ohi)

    league.development_leagues = dev
    league.players = league_players

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
                _tick_ratings(rng, p, overseas=False, junior=True)


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
