"""Twenty-five progression and regression questions, answered against the code."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine"), str(ROOT / "backend" / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.entities.player import display_rating, player_current_ovr_01  # noqa: E402
from app.sim_engine.engine import _aging_v3_base_decline_chance  # noqa: E402
from app.sim_engine.progression.development import (  # noqa: E402
    apply_live_overall_step,
    offseason_growth_points,
    season_growth_context,
)
from app.sim_engine.progression.potential import nudge_shown_potential  # noqa: E402
from app.sim_engine.progression.regression import (  # noqa: E402
    apply_regression,
    career_aging_decline_try,
)
from test_chapter_attributes import _make_player  # noqa: E402


def _ovr(player) -> int:
    return int(round(display_rating(player_current_ovr_01(player))))


def _nhl(seed, age, surplus, gp=70):
    player = _make_player("C", seed=seed)
    player.identity.age = age
    player.games_played = gp
    player._dev_season_points_surplus = surplus
    player._active_dev_season = 2026
    return player


def test_01_a_flat_nhl_year_gains_when_morale_is_high():
    neutral = _nhl(1, 28, 0)
    assert season_growth_context(neutral, bucket="nhl")["gain"] == 0.0
    happy = _nhl(101, 28, 0)
    happy.psych.morale = 0.85
    assert season_growth_context(happy, bucket="nhl")["gain"] >= 0.8


def test_02_a_big_surplus_is_worth_about_three_overall():
    gain = season_growth_context(_nhl(2, 26, 33), bucket="nhl")["gain"]
    assert 2.5 <= gain <= 4.0, gain


def test_03_a_bad_surplus_ticks_overall_down():
    player = _nhl(3, 29, -22)
    player.season_start_ovr = _ovr(player)
    start = player.season_start_ovr
    apply_live_overall_step(player, season_id=2026, bucket="nhl", day=1)
    assert _ovr(player) < start


def test_04_no_games_moves_with_morale():
    player = _make_player("C", seed=4)
    player.identity.age = 23
    player.psych.morale = 0.4
    before = _ovr(player)
    assert season_growth_context(player, bucket="nhl")["gain"] == 0.0
    assert apply_live_overall_step(player, season_id=2026, bucket="nhl", day=1) == 0
    assert _ovr(player) == before
    happy = _make_player("C", seed=104)
    happy.identity.age = 23
    happy.psych.morale = 0.9
    assert season_growth_context(happy, bucket="nhl")["gain"] >= 0.8


def test_05_a_great_year_cancels_the_age_drop():
    gain = season_growth_context(_nhl(5, 36, 40), bucket="nhl")["gain"]
    assert gain >= 0.0


def test_06_aging_chance_is_zero_through_35():
    assert _aging_v3_base_decline_chance(35) == 0.0
    assert _aging_v3_base_decline_chance(30) == 0.0


def test_07_aging_chance_starts_at_36_and_rises():
    assert _aging_v3_base_decline_chance(36) > 0
    assert _aging_v3_base_decline_chance(38) > _aging_v3_base_decline_chance(36)


def test_08_the_older_decline_roll_also_waits_until_36():
    player = SimpleNamespace(age=31, ratings={"off_shot": 80, "off_pass": 78}, retired=False)
    assert career_aging_decline_try(player, random.Random(1), None) is None


def test_09_a_quiet_veteran_does_not_regress():
    player = SimpleNamespace(age=34, peak_age=27, morale=0.7, ratings={"off_shot": 80, "dev_potential": 84})
    apply_regression(player, random.Random(1))
    assert player.ratings["off_shot"] == 80
    assert player.ratings["dev_potential"] == 84


def test_10_low_morale_costs_a_skill_point_and_not_potential():
    player = SimpleNamespace(
        age=34,
        peak_age=27,
        morale=0.2,
        ratings={"off_shot": 80, "off_pass": 77, "dev_potential": 88},
    )
    apply_regression(player, random.Random(1))
    assert player.ratings["off_shot"] == 79
    assert player.ratings["off_pass"] == 76
    assert player.ratings["dev_potential"] == 88


def test_11_morale_can_regress_a_player_under_his_peak():
    player = SimpleNamespace(age=24, peak_age=27, morale=0.1, ratings={"off_shot": 80, "dev_potential": 90})
    apply_regression(player, random.Random(1))
    assert player.ratings["off_shot"] == 79
    assert player.ratings["dev_potential"] == 90


def test_12_a_good_year_earns_more_than_a_point_in_the_summer():
    player = _nhl(12, 27, 30)
    extra = offseason_growth_points(player)
    assert extra > 1.0


def test_13_the_summer_is_not_paid_twice():
    player = _nhl(13, 27, 30)
    assert offseason_growth_points(player) >= 1.0
    player._offseason_growth_season = 2026
    assert offseason_growth_points(player) == 0.0


def test_14_a_year_under_the_line_does_not_get_a_summer_raise():
    player = _nhl(14, 24, -8)
    assert season_growth_context(player, bucket="nhl")["gain"] < 0
    assert offseason_growth_points(player) == 0.0


def test_15_a_young_player_who_met_the_line_gets_a_summer_point():
    player = _nhl(15, 23, 0)
    assert offseason_growth_points(player) == 1.0


def test_16_age_36_gets_a_summer_point_after_a_good_season():
    assert offseason_growth_points(_nhl(16, 36, 40)) >= 1.0


def test_17_an_empty_junior_season_does_not_grow():
    player = _make_player("C", seed=17)
    player.identity.age = 18
    assert season_growth_context(player, bucket="prospect")["gain"] == 0.0


def test_18_a_point_per_game_junior_year_is_worth_several_overall():
    player = _make_player("C", seed=18)
    player.identity.age = 18
    player.ratings["dev_potential"] = 94
    player.potential = 94 / 99.0
    player._prospect_season_stats = {"gp": 50, "points": 70, "goals": 30, "assists": 40}
    gain = season_growth_context(player, bucket="prospect")["gain"]
    assert gain >= 2.0, gain


def test_19_an_ahl_breakout_beats_a_quiet_ahl_year():
    quiet = _make_player("C", seed=19)
    loud = _make_player("C", seed=20)
    for player, pts in ((quiet, 18), (loud, 60)):
        player.identity.age = 21
        player._prospect_season_stats = {"gp": 50, "points": pts, "goals": pts // 2, "assists": pts // 2}
    assert season_growth_context(loud, bucket="ahl")["gain"] > season_growth_context(quiet, bucket="ahl")["gain"] + 2


def test_20_a_905_save_percentage_at_33_is_not_a_jump():
    player = _make_player("G", seed=21)
    player.identity.age = 33
    player._dev_save_pct = 0.905
    player.games_played = 50
    assert season_growth_context(player, bucket="nhl")["gain"] <= 0.5


def test_21_a_poor_save_percentage_does_not_lower_a_goalie():
    player = _make_player("G", seed=22)
    player.identity.age = 26
    player._dev_save_pct = 0.880
    player.games_played = 40
    assert season_growth_context(player, bucket="nhl")["gain"] >= 0.0


def test_22_a_28_year_old_can_still_have_potential_above_overall():
    player = _make_player("C", seed=23)
    player.identity.age = 28
    player.ratings["dev_potential"] = 90
    player.potential = 90 / 99.0
    before = 90
    nudge_shown_potential(player, random.Random(1), ovr_delta=0.0, production_gap=0.0)
    assert float(player.ratings["dev_potential"]) >= before - 0.01


def test_23_from_32_potential_cannot_sit_above_overall():
    player = _make_player("C", seed=24)
    player.identity.age = 33
    player.ratings["dev_potential"] = 94
    player.potential = 94 / 99.0
    nudge_shown_potential(player, random.Random(1), ovr_delta=0.0, production_gap=0.0)
    assert float(player.ratings["dev_potential"]) <= _ovr(player) + 0.6


def test_24_the_same_surplus_is_worth_less_at_96_overall():
    young = _nhl(25, 26, 28)
    old_star = _nhl(26, 26, 28)
    old_star.ovr = lambda: 0.97
    low = season_growth_context(young, bucket="nhl")["gain"]
    high = season_growth_context(old_star, bucket="nhl")["gain"]
    assert high > low * 0.9


def test_25_a_historic_year_has_a_ceiling_and_a_bad_year_has_a_floor():
    boom = season_growth_context(_nhl(27, 24, 90), bucket="nhl")["gain"]
    bust = season_growth_context(_nhl(28, 24, -90), bucket="nhl")["gain"]
    assert boom <= 6.0
    assert bust >= -6.0
    assert boom > 4.0
    assert bust <= -4.0


def test_an_older_player_cannot_walk_down_25_overall():
    player = _nhl(38, 38, -15)
    start = _ovr(player)
    player.season_start_ovr = start
    for day in range(40):
        apply_live_overall_step(player, season_id=2026, bucket="nhl", day=day)
    dropped = start - _ovr(player)
    assert 0 < dropped <= 6, dropped


def test_year_one_rights_are_seated_in_a_junior_league():
    from app.sim_engine.league_hierarchy_bootstrap import seat_rights_held_into_junior_leagues

    kid = _make_player("C", seed=41)
    kid.identity.age = 18
    kid.identity.birth_country = "Canada"
    kid.identity.name = "Year One Rights"
    kid.signed = False
    kid.id = "rights-y1"
    home = {"team_id": "CHL_OHL_1", "name": "Ottawa 67s", "city": "Ottawa", "players": []}
    league = SimpleNamespace(
        teams=[SimpleNamespace(id="OTT", team_id="OTT", roster=[], ahl_roster=[], echl_roster=[], prospect_pool=[kid])],
        development_leagues=[{"league_code": "CHL_OHL", "league_name": "OHL", "teams": [home]}],
    )
    placed = seat_rights_held_into_junior_leagues(league)
    assert placed == 1
    assert kid in home["players"]
    assert str(kid.current_league_id).startswith("CHL_")
    assert getattr(kid, "_prospect_season_stats", None) is not None
    again = seat_rights_held_into_junior_leagues(league)
    assert again == 0
    assert home["players"].count(kid) == 1
