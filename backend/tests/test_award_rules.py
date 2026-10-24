"""Award rules: who can win, and that subjective trophies follow their stated criteria."""
from __future__ import annotations

from types import SimpleNamespace as NS

from services._simengine_bootstrap import ensure_simengine_path

ensure_simengine_path()

from app.sim_engine.league import awards as A  # noqa: E402
from services.franchise_offseason import _regular_season_award_rows  # noqa: E402
from services.franchise_sim import _franchise_stat_scope  # noqa: E402


def test_playoff_ready_is_still_regular_season():
    assert _franchise_stat_scope(NS(phase="playoff_ready")) == "regular_season"
    assert _franchise_stat_scope(NS(phase="playoffs")) == "playoffs"
    assert _franchise_stat_scope(NS(phase="preseason")) == "preseason"


def test_mis_tagged_ledger_rows_are_regular_season_for_awards():
    s = NS(player_season_stats={"p": {"gp": 82, "pts": 90, "stat_scope": "playoffs"}})
    rows = _regular_season_award_rows(s)
    assert rows[0]["stat_scope"] == "regular_season"
    assert A.filter_regular_season_rows(rows)


def _row(**kw):
    base = {"player_id": "p", "name": "P", "team_id": "1", "position": "C", "gp": 70, "pts": 50, "g": 20, "a": 30}
    base.update(kw)
    return base


def test_calder_drafted_years_ago_with_no_nhl_games_is_a_rookie():
    # The old row stamp marked anyone drafted in an earlier year as an NHL veteran.
    row = _row(prior_nhl_seasons=1, is_rookie=False, draft_year=2025)
    hist = {"max_prior_season_gp": 0, "prior_seasons_6plus_gp": 0, "age_sept15": 20}
    assert A.calder_eligibility(row, history=hist)["eligible"] is True


def test_calder_real_import_without_career_rows_uses_rating_year_gp():
    from types import SimpleNamespace as NS

    from services.season_carryover import calder_history

    veteran = NS(
        id="NHL_97",
        real_nhl_import=True,
        career_stats={"seasons": []},
        real_nhl_import_stats={"gp": 82, "gamesPlayed": 82},
        identity=NS(draft_year=2015, birth_year=1997, birth_month=1, birth_day=13),
    )
    session = NS(sim=NS(league=NS(teams=[NS(roster=[veteran])])), season_calendar_year=2025, season_history=[])
    row = calder_history(session, 2025)["NHL_97"]
    assert row["is_rookie"] is False
    assert row["max_prior_season_gp"] == 82


def test_calder_nhl_games_rule_and_age():
    assert A.calder_eligibility(_row(), history={"max_prior_season_gp": 30, "prior_seasons_6plus_gp": 1, "age_sept15": 21})["eligible"] is False
    assert A.calder_eligibility(_row(), history={"max_prior_season_gp": 12, "prior_seasons_6plus_gp": 2, "age_sept15": 21})["eligible"] is False
    assert A.calder_eligibility(_row(), history={"max_prior_season_gp": 20, "prior_seasons_6plus_gp": 1, "age_sept15": 26})["eligible"] is False
    assert A.calder_eligibility(_row(), history={"max_prior_season_gp": 20, "prior_seasons_6plus_gp": 1, "age_sept15": 25})["eligible"] is True


def test_lady_byng_is_not_the_scoring_race():
    clean = _row(gp=82, pts=80, pim=4)
    star = _row(gp=82, pts=120, pim=60)
    assert A.lady_byng_score(clean) > A.lady_byng_score(star)


def test_vezina_rewards_stopping_pucks_over_workload():
    ctx_rows = [
        {"player_id": "a", "position": "G", "gp": 50, "shots_against": 1500, "ga": 105, "goalie_xga": 135, "w": 30},
        {"player_id": "b", "position": "G", "gp": 68, "shots_against": 2000, "ga": 190, "goalie_xga": 180, "w": 36},
    ]
    A.prepare_award_context(ctx_rows)
    elite, workhorse = ctx_rows
    assert A.vezina_ballot_score(elite) > A.vezina_ballot_score(workhorse)


def test_ted_lindsay_ignores_team_standings():
    ctx_good = {"1": {"points_pct_norm": 1.0, "playoff_qualified": True}}
    ctx_bad = {"1": {"points_pct_norm": 0.0}}
    r = _row(gp=82, pts=110, g=45)
    assert A.ted_lindsay_score(r, ctx_good) == A.ted_lindsay_score(r, ctx_bad)


def test_conn_smythe_one_game_wonder_cannot_win():
    hero = _row(player_id="h", team_id="9", gp=1, pts=4, g=3)
    leader = _row(player_id="l", team_id="9", gp=24, pts=28, g=12)
    assert A.conn_smythe_score(leader, champion_id="9") > A.conn_smythe_score(hero, champion_id="9")
    champ = _row(player_id="c", team_id="9", gp=24, pts=22, g=9)
    loser = _row(player_id="x", team_id="3", gp=24, pts=24, g=10)
    assert A.conn_smythe_score(champ, champion_id="9", finalist_ids=["9", "3"]) > A.conn_smythe_score(loser, champion_id="9", finalist_ids=["9", "3"])


def test_every_award_has_a_plain_language_rule():
    for aid in A.AWARD_REGISTRY:
        assert A.AWARD_CRITERIA.get(aid), aid
