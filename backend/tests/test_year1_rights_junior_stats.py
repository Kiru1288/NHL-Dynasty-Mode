"""Year-1 rights must land in a junior league and receive a real scoring line."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine"), str(ROOT / "backend" / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.entities.player import display_rating, player_current_ovr_01  # noqa: E402
from app.sim_engine.generation.prospect_league_scoring import (  # noqa: E402
    advance_all_development_league_stats,
)
from app.sim_engine.league_hierarchy_bootstrap import (  # noqa: E402
    seat_rights_held_into_junior_leagues,
)
from app.sim_engine.progression.development import (  # noqa: E402
    apply_live_overall_step,
    season_growth_context,
)
from services.franchise_offseason import _dev_stamp_season_production  # noqa: E402
from test_chapter_attributes import _make_player  # noqa: E402


def _ovr(player) -> int:
    return int(round(display_rating(player_current_ovr_01(player))))


def _rights(seed: int, *, name: str, country: str, age: int = 18):
    player = _make_player("C", seed=seed)
    player.identity.age = age
    player.identity.birth_country = country
    player.identity.name = name
    player.signed = False
    player.id = f"rights-{seed}"
    player.rng_seed = seed
    return player


def _league_block(code: str, name: str):
    return {
        "league_code": code,
        "league_name": name,
        "teams": [{"team_id": f"{code}_1", "name": name, "city": name, "players": []}],
    }


def _world(pool):
    blocks = [
        _league_block("CHL_OHL", "OHL"),
        _league_block("CHL_WHL", "WHL"),
        _league_block("CHL_QMJHL", "QMJHL"),
        _league_block("USHL", "USHL"),
        _league_block("NCAA", "NCAA"),
        _league_block("EU_J_SHL", "SHL J20"),
    ]
    ahl_vet = _rights(90, name="Already In The AHL", country="Canada", age=21)
    ahl_vet.roster_location = "ahl"
    league = SimpleNamespace(
        teams=[
            SimpleNamespace(
                id="OTT",
                team_id="OTT",
                roster=[],
                ahl_roster=[ahl_vet],
                echl_roster=[],
                prospect_pool=list(pool) + [ahl_vet],
            )
        ],
        development_leagues=blocks,
        _prospect_stat_sync_state={},
    )
    return league, ahl_vet, {block["league_code"]: block for block in blocks}


def _scoring_board(league):
    """Same source the prospect-scoring screen reads: development-league rosters."""
    rows = []
    for block in league.development_leagues:
        code = block["league_code"]
        for team in block["teams"]:
            for player in team["players"]:
                stats = getattr(player, "_prospect_season_stats", None) or {}
                rows.append(
                    {
                        "id": player.id,
                        "league": code,
                        "team": team["name"],
                        "gp": int(stats.get("gp") or 0),
                        "goals": int(stats.get("goals") or 0),
                        "assists": int(stats.get("assists") or 0),
                        "points": int(stats.get("points") or 0),
                    }
                )
    return rows


def test_year1_rights_get_a_league_and_a_scoring_line():
    canadian = _rights(1, name="Canadian Rights", country="Canada")
    american = _rights(2, name="American Rights", country="USA", age=19)
    european = _rights(3, name="European Rights", country="Sweden")
    already = _rights(4, name="Already In The OHL", country="Canada")
    signed = _rights(5, name="Already Signed", country="Canada")
    signed.signed = True

    league, ahl_vet, blocks = _world([canadian, american, european, already, signed])
    blocks["CHL_OHL"]["teams"][0]["players"].append(already)
    already.current_league_id = "CHL_OHL"

    placed = seat_rights_held_into_junior_leagues(league)
    assert placed == 3
    assert canadian in blocks["CHL_OHL"]["teams"][0]["players"] or any(
        canadian in block["teams"][0]["players"]
        for code, block in blocks.items()
        if code.startswith("CHL_")
    )
    assert american.current_league_id == "NCAA"
    assert str(european.current_league_id).startswith("EU_")
    assert ahl_vet not in blocks["CHL_OHL"]["teams"][0]["players"]
    assert signed not in blocks["CHL_OHL"]["teams"][0]["players"]
    assert blocks["CHL_OHL"]["teams"][0]["players"].count(already) == 1

    # Before the calendar moves, the new line is empty. That is the bug the
    # screen was showing: a player with a league name and zero scoring.
    for kid in (canadian, american, european):
        fresh = getattr(kid, "_prospect_season_stats", None) or {}
        assert int(fresh.get("gp") or 0) == 0

    sim = SimpleNamespace(league=league, rng=None)
    updated = advance_all_development_league_stats(sim, "2026-04-15", season_year=2025)
    assert updated >= 4

    board = _scoring_board(league)
    by_id = {row["id"]: row for row in board}
    for kid in (canadian, american, european, already):
        row = by_id[kid.id]
        floor = 30 if row["league"] == "NCAA" else 40
        assert row["gp"] >= floor, row
        assert row["points"] >= 8, row
        assert row["goals"] >= 1 and row["assists"] >= 1, row
        assert row["points"] == row["goals"] + row["assists"], row
    assert "rights-90" not in by_id
    assert "rights-5" not in by_id

    # The same numbers are what overall growth reads. A full junior year moves.
    # An empty year does not.
    for kid in (canadian, american, european):
        kid.ratings["dev_potential"] = 94
        kid.potential = 94 / 99.0
        stats = kid._prospect_season_stats
        ppg = int(stats["points"]) / int(stats["gp"])
        gain = season_growth_context(kid, bucket="prospect")["gain"]
        if ppg >= 0.80:
            assert gain >= 1.0, (kid.identity.name, ppg, gain)
        elif ppg >= 0.50:
            assert gain > 0.0, (kid.identity.name, ppg, gain)
        start = _ovr(kid)
        kid.season_start_ovr = start
        apply_live_overall_step(kid, season_id=2025, bucket="prospect", day=1)
        assert _ovr(kid) >= start

    empty = _rights(6, name="No Games Yet", country="Canada")
    empty._dev_growth_league = "CHL_OHL"
    empty.current_league_id = "CHL_OHL"
    empty._prospect_season_stats = {"gp": 0, "goals": 0, "assists": 0, "points": 0, "league": "CHL_OHL"}
    assert season_growth_context(empty, bucket="prospect")["gain"] == 0.0


def test_young_ahl_prospects_and_journeymen_both_grow_from_a_real_season():
    kid = _make_player("C", seed=31)
    kid.identity.age = 20
    kid.ratings["dev_potential"] = 92
    kid.potential = 92 / 99.0
    kid._prospect_season_stats = {"gp": 55, "points": 30, "goals": 12, "assists": 18, "league": "AHL"}
    kid_gain = season_growth_context(kid, bucket="ahl")["gain"]
    assert kid_gain >= 2.0, kid_gain

    vet = _make_player("LW", seed=32)
    vet.identity.age = 28
    ceiling = _ovr(vet)
    vet.ratings["dev_potential"] = ceiling
    vet.potential = ceiling / 99.0
    vet._prospect_season_stats = {"gp": 60, "points": 40, "goals": 16, "assists": 24, "league": "AHL"}
    vet_gain = season_growth_context(vet, bucket="ahl")["gain"]
    assert vet_gain >= 1.0, vet_gain

    defender = _make_player("D", seed=33)
    defender.identity.age = 27
    defender.ratings["dev_potential"] = _ovr(defender)
    defender.potential = _ovr(defender) / 99.0
    defender._prospect_season_stats = {"gp": 60, "points": 28, "goals": 6, "assists": 22, "league": "AHL"}
    assert season_growth_context(defender, bucket="ahl")["gain"] >= 1.0

    empty = _make_player("C", seed=34)
    empty.identity.age = 20
    empty.ratings["dev_potential"] = 94
    empty.potential = 94 / 99.0
    assert season_growth_context(empty, bucket="ahl")["gain"] == 0.0

    buried = _make_player("C", seed=35)
    buried.identity.age = 29
    buried.ratings["dev_potential"] = _ovr(buried)
    buried.potential = _ovr(buried) / 99.0
    buried._prospect_season_stats = {"gp": 50, "points": 8, "goals": 3, "assists": 5, "league": "AHL"}
    assert season_growth_context(buried, bucket="ahl")["gain"] == 0.0


def test_a_stamped_ahl_line_grows_and_an_empty_ahl_year_does_not():
    producer = _make_player("LW", seed=12)
    producer.identity.age = 21
    producer.id = "ahl-scorer"
    producer._prospect_season_stats = {
        "gp": 60,
        "goals": 28,
        "assists": 32,
        "points": 60,
        "league": "AHL",
    }
    session = SimpleNamespace(player_season_stats={})
    _dev_stamp_season_production(session, producer)
    assert producer._dev_growth_league == "AHL"
    assert int(producer._prospect_season_stats["points"]) == 60
    assert int(producer.games_played) == 60
    gain = season_growth_context(producer, bucket="ahl")["gain"]
    assert gain >= 2.0, gain
    start = _ovr(producer)
    producer.season_start_ovr = start
    apply_live_overall_step(producer, season_id=2026, bucket="ahl", day=1)
    assert _ovr(producer) > start

    quiet = _make_player("LW", seed=13)
    quiet.identity.age = 21
    quiet._dev_growth_league = "AHL"
    quiet._prospect_season_stats = {"gp": 0, "points": 0, "league": "AHL"}
    assert season_growth_context(quiet, bucket="ahl")["gain"] == 0.0


def _ahl_roster(prefix: str, seed: int):
    players = []
    specs = [("C", 12), ("D", 6), ("G", 2), ("LW", 1)]
    n = 0
    for pos, count in specs:
        for _ in range(count):
            player = _make_player(pos, seed=seed + n)
            player.id = f"{prefix}-{pos}-{n}"
            player.identity.age = 21
            player.identity.name = player.id
            player.ratings["dev_potential"] = 90
            player.potential = 90 / 99.0
            player.season_start_ovr = _ovr(player)
            players.append(player)
            n += 1
    return players


def test_an_ahl_season_grows_overall_after_the_games_are_played():
    """Season sim used to judge AHL growth before any AHL game existed."""
    from services.ahl_league import sync_ahl_to_date
    from services.franchise_sim import _apply_earned_minor_league_growth

    home_roster = _ahl_roster("home", 100)
    away_roster = _ahl_roster("away", 200)
    scratch = home_roster[-1]
    home = SimpleNamespace(team_id="HOME", id="HOME", abbreviation="HOM", city="Home", ahl_roster=home_roster, roster=[])
    away = SimpleNamespace(team_id="AWAY", id="AWAY", abbreviation="AWY", city="Away", ahl_roster=away_roster, roster=[])
    league = SimpleNamespace(teams=[home, away], development_leagues=[])
    session = SimpleNamespace(
        season_calendar_year=2025,
        session_id="ahl-growth",
        calendar_days_finished=200,
        sim=SimpleNamespace(league=league, rng=__import__("random").Random(3)),
        ahl_lines={},
        ahl_league=None,
        player_season_stats={},
    )

    before = {p.id: _ovr(p) for p in home_roster + away_roster}
    # No games yet: the growth pass has nothing to pay.
    assert _apply_earned_minor_league_growth(session) == 0
    assert {p.id: _ovr(p) for p in home_roster + away_roster} == before

    games = sync_ahl_to_date(session, "2026-04-15")
    assert games >= 24
    played = [p for p in home_roster + away_roster if int((getattr(p, "_prospect_season_stats", None) or {}).get("gp") or 0) >= 20]
    assert len(played) >= 20
    scratch_stats = getattr(scratch, "_prospect_season_stats", None) or {}
    assert int(scratch_stats.get("gp") or 0) == 0

    moved = _apply_earned_minor_league_growth(session)
    assert moved >= 1
    growers = []
    for player in played:
        delta = _ovr(player) - before[player.id]
        stats = player._prospect_season_stats
        growers.append((delta, int(stats.get("points") or 0), int(stats.get("gp") or 0), player.id))
    growers.sort(reverse=True)
    assert growers[0][0] >= 1, growers[:8]
    assert growers[0][0] <= 6, growers[:8]
    assert _ovr(scratch) == before[scratch.id]
    # The season is already on the card. Another pass cannot walk anyone up again.
    settled = {p.id: _ovr(p) for p in home_roster + away_roster}
    _apply_earned_minor_league_growth(session)
    for player in home_roster + away_roster:
        assert _ovr(player) <= settled[player.id]


def test_a_stamped_junior_line_is_not_rewritten_as_nhl_points():
    kid = _make_player("RW", seed=14)
    kid.identity.age = 18
    kid.id = "ohl-line"
    kid.current_league_id = "CHL_OHL"
    kid._dev_growth_league = "CHL_OHL"
    kid.ratings["dev_potential"] = 94
    kid.potential = 94 / 99.0
    kid._prospect_season_stats = {
        "gp": 62,
        "goals": 36,
        "assists": 48,
        "points": 84,
        "league": "CHL_OHL",
        "ppg": 1.35,
    }
    session = SimpleNamespace(player_season_stats={})
    _dev_stamp_season_production(session, kid)
    line = kid._prospect_season_stats
    assert int(line["gp"]) == 62
    assert int(line["goals"]) == 36
    assert int(line["assists"]) == 48
    assert int(line["points"]) == 84
    assert "NHL" not in str(kid._dev_growth_league).upper()
    gain = season_growth_context(kid, bucket="prospect")["gain"]
    assert gain >= 2.0, gain
