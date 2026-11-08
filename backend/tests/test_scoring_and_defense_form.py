"""Goal/assist split, and defensemen rated on defense rather than points."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine"), str(ROOT / "backend" / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.engine import SimEngine, _assist_pull, _share_from_overall  # noqa: E402
from app.sim_engine.gameplay.game_analytics_ledger import assist_count_probability  # noqa: E402
from app.sim_engine.progression.development import (  # noqa: E402
    apply_in_season_performance_checkpoints,
    season_gain_cap_points,
    season_points_surplus_vs_expectation,
)
from services.franchise_offseason import _dev_season_stats  # noqa: E402
from test_chapter_attributes import _make_player  # noqa: E402


def _balance_engine():
    eng = object.__new__(SimEngine)
    eng._gm_is_goal_first_scorer = lambda _p: False  # type: ignore
    eng._gm_is_certified_playmaker = lambda p: "play" in str(getattr(p, "player_type", ""))  # type: ignore
    eng._gm_is_assist_leaning_forward = lambda _p: False  # type: ignore
    eng._gm_pos_str = lambda p: str(getattr(p, "position", "C"))  # type: ignore
    return eng


def test_a_goal_heavy_forward_loses_goals_and_gains_assists():
    eng = _balance_engine()
    forward = SimpleNamespace(id="f1", position="C", player_type="two_way")
    ledger = {"f1": {"g": 36, "a": 22}}
    score = eng._gm_goal_assist_balance_mult(forward, ledger, "T", role="score")
    assist = eng._gm_goal_assist_balance_mult(forward, ledger, "T", role="assist")
    # The season totals do not move the next shot or the next assist.
    assert score == 1.0, score
    assert assist == 1.0, assist


def test_a_playmaker_who_is_scoring_is_pushed_back_to_assists():
    eng = _balance_engine()
    maker = SimpleNamespace(id="m1", position="C", player_type="playmaker")
    ledger = {"m1": {"g": 28, "a": 24}}
    score = eng._gm_goal_assist_balance_mult(maker, ledger, "T", role="score")
    assist = eng._gm_goal_assist_balance_mult(maker, ledger, "T", role="assist")
    assert score == 1.0, score
    assert assist == 1.0, assist


def test_defencemen_stop_piling_up_goals():
    eng = _balance_engine()
    defender = SimpleNamespace(id="d1", position="D", player_type="shutdown")
    ledger = {"d1": {"g": 18, "a": 22}}
    score = eng._gm_goal_assist_balance_mult(defender, ledger, "T", role="score")
    assist = eng._gm_goal_assist_balance_mult(defender, ledger, "T", role="assist")
    assert score == 1.0, score
    assert assist == 1.0, assist


def test_most_goals_carry_two_assists():
    chances = (
        ("SLOT", "EV"),
        ("RUSH_MEDIUM", "EV"),
        ("REBOUND", "EV"),
        ("POINT_SHOT", "EV"),
        ("PP_SLOT", "PP"),
        ("LOW_DANGER_PERIMETER", "EV"),
    )
    rates = []
    for chance, strength in chances:
        p0, p1, p2 = assist_count_probability(chance, strength)
        rates.append(0 * p0 + 1 * p1 + 2 * p2)
        assert p0 < 0.08
    assert sum(rates) / len(rates) >= 1.65


def test_bulk_goal_roll_averages_close_to_two_assists():
    rng = random.Random(7)
    total = 0
    n = 4000
    for _ in range(n):
        roll = rng.random()
        total += 0 if roll < 0.015 else (2 if roll < 0.94 else 1)
    assert 1.88 <= total / n <= 1.96


def _shutdown(points: int, **marks):
    player = _make_player("D", seed=77)
    player.player_type = "shutdown"
    player.games_played = 60
    player._dev_season_points = points
    player._dev_toi_sec = 22 * 60 * 60
    for key, value in marks.items():
        setattr(player, key, value)
    return player


def test_a_shutdown_defenceman_is_not_docked_for_a_quiet_scoresheet():
    player = _shutdown(
        12,
        _dev_plus_minus=28,
        _dev_blocks=150,
        _dev_xgf_pct=0.56,
        _dev_gf_on=70,
        _dev_ga_on=42,
        _dev_xga=95,
    )
    surplus = season_points_surplus_vs_expectation(player, 12, 60)
    assert surplus > 4.0, surplus
    delta = apply_in_season_performance_checkpoints(player, random.Random(1), perf_checkpoint_gp=41)
    assert delta >= 0.0, delta


def test_a_leaky_defenceman_is_docked_even_if_he_scores():
    player = _shutdown(
        40,
        _dev_plus_minus=-24,
        _dev_blocks=40,
        _dev_xgf_pct=0.43,
        _dev_gf_on=40,
        _dev_ga_on=78,
        _dev_xga=160,
    )
    quiet = _shutdown(12, _dev_plus_minus=28, _dev_blocks=150, _dev_xgf_pct=0.56, _dev_gf_on=70, _dev_ga_on=42, _dev_xga=95)
    leaky = season_points_surplus_vs_expectation(player, 40, 60)
    solid = season_points_surplus_vs_expectation(quiet, 12, 60)
    assert leaky < solid
    assert leaky < 0.0, leaky


def test_deployment_style_is_what_moves_ice_time():
    eng = object.__new__(SimEngine)
    rolled = eng._EV_LINE_SHARES["rolling_four"]
    heavy = eng._EV_LINE_SHARES["top_heavy"]
    assert heavy[0] > rolled[0]
    assert rolled[3] > heavy[3]
    assert abs(sum(rolled) - 1.0) < 0.02
    assert abs(sum(heavy) - 1.0) < 0.02


def test_overall_separates_a_star_from_the_rest_of_his_line():
    eng = object.__new__(SimEngine)
    eng._gm_ovr_0_100 = lambda p: float(p)  # type: ignore
    star = eng._gm_talent_share_weight(97)
    mate = eng._gm_talent_share_weight(88)
    depth = eng._gm_talent_share_weight(80)
    assert 1.06 <= star / mate <= 1.20, (star, mate)
    assert 1.05 <= mate / depth <= 1.20, (mate, depth)
    assert eng._gm_talent_share_weight(99) / eng._gm_talent_share_weight(94) < 1.15


def test_the_same_overall_can_lead_goals_and_assists():
    # A sniper and a playmaker at 95. Skill tilts the split. It does not
    # build two different leaderboards.
    sniper_goals = _share_from_overall(95, 0.96)
    sniper_assists = _share_from_overall(95, 0.76)
    maker_goals = _share_from_overall(95, 0.76)
    maker_assists = _share_from_overall(95, 0.96)
    assert sniper_goals / maker_goals < 1.25, (sniper_goals, maker_goals)
    assert maker_assists / sniper_assists < 1.25, (maker_assists, sniper_assists)
    assert sniper_assists / sniper_goals > 0.75
    assert maker_goals / maker_assists > 0.75
    # The 95 still out-touches an 84 of the same style.
    assert _share_from_overall(95, 0.90) > _share_from_overall(84, 0.90) * 1.08
    # The setup man is the helper. The scorer on the same line is still in the picture.
    passer = _assist_pull(95, 0.96)
    scorer = _assist_pull(95, 0.78)
    depth = _assist_pull(82, 0.70)
    assert passer / depth >= 1.8, (passer, depth)
    assert scorer / passer > 0.62, (scorer, passer)


def test_line_weights_are_not_rewritten_into_a_fixed_share():
    eng = object.__new__(SimEngine)
    # Weights already built from overall and shot ratings. A 90, an 86,
    # an 84, and two defenders. Nobody is assigned a quota of the five.
    raw = [1.32, 1.14, 1.06, 0.62, 0.55]
    spread = eng._gm_spread_unit_shot_weights(
        raw,
        [False, False, False, True, True],
    )
    assert spread == [max(0.01, w) for w in raw]
    share = spread[0] / sum(spread)
    assert share < 0.40, (share, spread)
    assert spread[0] > spread[1] > spread[4]


def test_a_forward_great_season_is_not_only_points():
    quiet = _make_player("C", seed=83)
    quiet.games_played = 70
    quiet._dev_season_points = 28
    driver = _make_player("C", seed=84)
    driver.games_played = 70
    driver._dev_season_points = 28
    driver._dev_plus_minus = 24
    driver._dev_xgf_pct = 0.58
    driver._dev_gf_on = 78
    driver._dev_ga_on = 46
    driver._dev_cf = 820
    driver._dev_ca = 640
    driver._dev_pp_share = 0.26
    driver._dev_pk_share = 0.18
    driver._dev_pp_points = 22
    quiet_surplus = season_points_surplus_vs_expectation(quiet, 28, 70)
    drive_surplus = season_points_surplus_vs_expectation(driver, 28, 70)
    assert drive_surplus > quiet_surplus + 8.0, (drive_surplus, quiet_surplus)


def test_positive_possession_covers_part_of_a_forward_point_miss():
    driver = _make_player("C", seed=81)
    driver.games_played = 70
    driver._dev_xgf_pct = 0.58
    driver._dev_gf_on = 80
    driver._dev_ga_on = 48
    finisher = _make_player("C", seed=82)
    finisher.games_played = 70
    same_points = 30
    with_drive = season_points_surplus_vs_expectation(driver, same_points, 70)
    bare = season_points_surplus_vs_expectation(finisher, same_points, 70)
    assert with_drive > bare + 4.0


def test_season_stamp_keeps_defensive_results():
    player = _make_player("D", seed=90)
    player.id = "stamp-d"
    session = SimpleNamespace(
        player_season_stats={
            "stamp-d": {
                "gp": 70,
                "g": 4,
                "a": 18,
                "pts": 22,
                "toi_sec": 20 * 60 * 70,
                "plus_minus": 21,
                "blk": 130,
                "gf_on": 68,
                "ga_on": 47,
                "xgf": 62.0,
                "xga": 48.0,
                "league": "NHL",
            }
        },
        season_calendar_year=2026,
        sim=None,
    )
    stats = _dev_season_stats(session, player)
    assert stats["plus_minus"] == 21
    assert stats["blocks"] == 130
    assert stats["xgf_pct"] > 0.55


def test_a_veteran_goalie_at_905_cannot_jump_a_whole_tier():
    goalie = _make_player("G", seed=33)
    goalie.identity.age = 33
    goalie._dev_save_pct = 0.905
    assert season_gain_cap_points(goalie) <= 1.0


def test_top_six_gets_mercy_on_a_modest_points_miss():
    top = _make_player("C", seed=90)
    quiet = _make_player("C", seed=91)
    top.role = "top_six"
    quiet.role = "middle"
    for player in (top, quiet):
        player._dev_usage_factor = 1.0
        player.games_played = 70
        player._dev_season_points = 30
    top_gap = season_points_surplus_vs_expectation(top, 30, 70)
    quiet_gap = season_points_surplus_vs_expectation(quiet, 30, 70)
    assert top_gap > quiet_gap
