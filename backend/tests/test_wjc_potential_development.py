"""World Juniors exclusivity, post-year-1 draft ceilings, and live overall movement."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine"), str(ROOT / "backend" / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.entities.player import Position, display_rating, player_current_ovr_01  # noqa: E402
from app.sim_engine.league_hierarchy_bootstrap import (  # noqa: E402
    _shape_draft_class_pipeline,
    _spawn_player,
    _stored_potential99,
    set_spawn_as_of_year,
)
from app.sim_engine.engine import _career_aging_decline_try_v3  # noqa: E402
from app.sim_engine.progression.development import (  # noqa: E402
    apply_live_overall_step,
    apply_player_development,
    offseason_growth_points,
    calculate_season_growth_budget,
    resolve_development_profile,
    sheltered_positive_season,
)
from services.draft_ranking_logic import (  # noqa: E402
    compute_consensus_potential_evaluation,
    compute_enhanced_draft_score,
)
from services.franchise_sim import _clamp_potential99, _wjc_player_may_play_tournament  # noqa: E402
from test_chapter_attributes import _make_player  # noqa: E402


def _pot(player) -> float:
    return _stored_potential99(player)


def _class_of(year: int, n: int, seed: int):
    set_spawn_as_of_year(year)
    rng = random.Random(seed)
    players = []
    for i in range(n):
        pos = Position.G if i % 11 == 0 else Position.C
        players.append(
            _spawn_player(
                rng,
                pos=pos,
                ovr_lo=0.48,
                ovr_hi=0.62,
                age_lo=18,
                age_hi=18,
                used_names=set(),
                league_players=[],
                pool_context="junior",
                league_code="CHL_OHL",
                as_of_year=year,
            )
        )
        ident = players[-1].identity
        ident.birth_month = 10
        ident.birth_day = 1
        if getattr(players[-1], "birth_date", None):
            players[-1].birth_date = f"{ident.birth_year}-10-01"
    league = SimpleNamespace(
        development_leagues=[{"league_code": "CHL_OHL", "teams": [{"players": players}]}],
        season_start_year=year,
    )
    _shape_draft_class_pipeline(league, rng, draft_year=year + 1)
    return players


def test_nhl_roster_player_cannot_play_wjc_without_a_loan():
    assert _wjc_player_may_play_tournament(False, {}, "junior") is True
    assert _wjc_player_may_play_tournament(True, {}, "nhl-kid") is False
    assert _wjc_player_may_play_tournament(True, {"nhl-kid": "full"}, "nhl-kid") is True
    assert _wjc_player_may_play_tournament(True, {"nhl-kid": "none"}, "nhl-kid") is False


def test_year_two_draft_class_keeps_a_high_end():
    year_one = _class_of(2025, 90, 3)
    for p in year_one:
        p.drafted = True
    year_two = _class_of(2026, 90, 9)
    pots = sorted((_pot(p) for p in year_two if getattr(getattr(p, "identity", None), "position", None) != Position.G), reverse=True)
    assert pots[0] >= 93, pots[:8]
    assert sum(1 for v in pots if v >= 93) >= 2
    assert sum(1 for v in pots if v >= 89) >= 6
    # The first generated class is not a special case. The next one matches it.
    first = sorted((_pot(p) for p in year_one if getattr(getattr(p, "identity", None), "position", None) != Position.G), reverse=True)
    assert first[0] >= 93
    assert pots[0] >= first[0] - 5


def test_every_development_tick_moves_overall():
    """A tick moves when the season earned it, and stops once that number is reached."""
    player = _make_player("C", seed=80)
    player.id = "live-nhl"
    player.identity.age = 24
    player.games_played = 50
    player._dev_season_points_surplus = 30
    player.season_start_ovr = int(round(display_rating(player_current_ovr_01(player))))
    start = player.season_start_ovr
    moved = False
    for day in range(8):
        step = apply_live_overall_step(player, season_id=2026, bucket="nhl", day=day)
        if step:
            moved = True
    assert moved
    end = int(round(display_rating(player_current_ovr_01(player))))
    assert start < end <= start + 4, (start, end)
    held = apply_live_overall_step(player, season_id=2026, bucket="nhl", day=9)
    assert held == 0

    empty = _make_player("C", seed=81)
    empty.identity.age = 22
    before = int(round(display_rating(player_current_ovr_01(empty))))
    assert apply_live_overall_step(empty, season_id=2026, bucket="prospect", day=1) == 0
    assert int(round(display_rating(player_current_ovr_01(empty)))) == before


def test_later_classes_keep_the_high_end_across_seeds():
    for seed in (2, 11, 17, 29, 41):
        players = _class_of(2027, 90, seed)
        pots = [
            _pot(p)
            for p in players
            if getattr(getattr(p, "identity", None), "position", None) != Position.G
        ]
        assert max(pots) >= 93, seed
        assert sum(1 for v in pots if v >= 93) >= 2, seed
        assert sum(1 for v in pots if v >= 89) >= 6, seed


def test_franchise_elite_and_top_floors_survive_the_board_clamp():
    cases = (("franchise", 93.0), ("elite", 89.0), ("top", 86.0))
    for tier, floor in cases:
        player = SimpleNamespace(pipeline_tier=tier, ratings={"dev_potential": floor + 3})
        held = _clamp_potential99(player, 58.0, floor + 3)
        assert held >= floor, (tier, held)


def test_resolving_a_profile_does_not_drop_a_star_ceiling():
    player = _make_player("C", seed=12)
    player.pipeline_tier = "franchise"
    player.ratings["dev_potential"] = 96
    player.potential = 96 / 99.0
    profile = resolve_development_profile(player, {"current_ovr_hint": 0.62})
    ceiling = float(profile["expected_ceiling"])
    shown = ceiling * 99.0 if ceiling <= 1.5 else ceiling
    assert shown >= 93, shown
    assert float(player.ratings["dev_potential"]) >= 93


def test_young_players_under_potential_move_up():
    player = _make_player("C", seed=21)
    player.id = "kid-up"
    player.identity.age = 19
    player.ratings["dev_potential"] = 96
    start = int(round(display_rating(player_current_ovr_01(player))))
    assert start + 2 <= 96
    step = apply_live_overall_step(player, season_id=2026, bucket="prospect", day=1)
    # No junior line yet, so the rating stays where it is.
    assert step == 0, step
    player._prospect_season_stats = {"gp": 48, "points": 70, "goals": 30, "assists": 40}
    step = apply_live_overall_step(player, season_id=2026, bucket="prospect", day=2)
    assert step > 0, step
    for day in range(3, 10):
        apply_live_overall_step(player, season_id=2026, bucket="prospect", day=day)
    now = int(round(display_rating(player_current_ovr_01(player))))
    assert now <= start + 6, (start, now)


def test_age_decline_starts_at_36():
    from app.sim_engine.engine import _aging_v3_base_decline_chance

    for age in (28, 30, 33, 35):
        assert _aging_v3_base_decline_chance(age) == 0.0, age
    assert _aging_v3_base_decline_chance(36) > 0.0
    assert _aging_v3_base_decline_chance(38) > _aging_v3_base_decline_chance(36)

    prime = _make_player("C", seed=22)
    prime.id = "age-30"
    prime.identity.age = 30
    prime.ratings["dev_potential"] = 70
    prime.games_played = 60
    prime._dev_season_points = 8
    prime._dev_season_points_surplus = -20
    prime.season_start_ovr = int(round(display_rating(player_current_ovr_01(prime))))
    prime_start = prime.season_start_ovr
    for day in range(4):
        apply_live_overall_step(prime, season_id=2026, bucket="nhl", day=day)
    assert int(round(display_rating(player_current_ovr_01(prime)))) < prime_start
    flat = _make_player("C", seed=24)
    flat.id = "flat-year"
    flat.identity.age = 28
    flat.games_played = 70
    flat._dev_season_points_surplus = 0
    flat.season_start_ovr = int(round(display_rating(player_current_ovr_01(flat))))
    assert apply_live_overall_step(flat, season_id=2026, bucket="nhl", day=1) == 0

    old = _make_player("C", seed=23)
    old.id = "age-36"
    old.identity.age = 36
    old.games_played = 70
    old._dev_season_points = 90
    old._dev_season_points_surplus = 30
    old._dev_xgf_pct = 0.60
    old._dev_plus_minus = 25
    old.season_start_ovr = int(round(display_rating(player_current_ovr_01(old))))
    old_start = old.season_start_ovr
    for day in range(4):
        apply_live_overall_step(old, season_id=2026, bucket="nhl", day=day)
    assert int(round(display_rating(player_current_ovr_01(old)))) >= old_start


def test_a_player_ticks_toward_the_season_he_earned():
    from app.sim_engine.progression.development import season_growth_context

    quiet = _make_player("C", seed=31)
    quiet.id = "quiet-nhl"
    quiet.identity.age = 33
    quiet.games_played = 70
    quiet._dev_season_points_surplus = -20
    quiet.season_start_ovr = int(round(display_rating(player_current_ovr_01(quiet))))
    earned = float(season_growth_context(quiet, bucket="nhl")["gain"])
    assert earned <= 1.5
    step = apply_live_overall_step(quiet, season_id=2026, bucket="nhl", day=2)
    assert step < 0, step

    star = _make_player("C", seed=32)
    star.id = "ultimate"
    star.identity.age = 23
    star._dev_season_points_surplus = 55
    star.season_start_ovr = int(round(display_rating(player_current_ovr_01(star))))
    boom = float(season_growth_context(star, bucket="nhl")["gain"])
    assert 3.0 <= boom <= 5.5, boom
    up = apply_live_overall_step(star, season_id=2026, bucket="nhl", day=3)
    assert up > 0, up


def _ovr(player) -> float:
    return float(display_rating(player_current_ovr_01(player)))


def _later_high_counts(seed: int):
    players = _class_of(2028, 120, seed)
    skaters = [
        p
        for p in players
        if getattr(getattr(p, "identity", None), "position", None) != Position.G
    ]
    later = [p for p in skaters if _ovr(p) <= 74]
    pots = [_pot(p) for p in later]
    return (
        sum(v >= 90 for v in pots),
        sum(v >= 86 for v in pots),
        sum(1 for p in skaters if _pot(p) >= 93),
    )


def test_later_rounds_carry_a_varying_high_end():
    counts = [_later_high_counts(seed) for seed in (4, 19, 28, 41, 63)]
    n90 = [row[0] for row in counts]
    n86 = [row[1] for row in counts]
    assert min(n90) >= 12, counts
    assert min(n86) >= 22, counts
    assert max(n90) - min(n90) >= 1
    tops = [row[2] for row in counts]
    assert min(tops) >= 2
    assert max(tops) != min(tops)

    players = _class_of(2028, 120, 19)
    skaters = [
        p
        for p in players
        if getattr(getattr(p, "identity", None), "position", None) != Position.G
    ]
    rows = []
    for p in skaters:
        ovr = _ovr(p)
        row = {
            "true_ovr": ovr,
            "age": 18,
            "position": "C",
            "league_code": "CHL_OHL",
            "key": str(getattr(p, "id", id(p))),
            "ppg": 0.35,
            "production_adjusted_score": 0.25,
            "true_potential_score": _pot(p),
        }
        row["consensus_potential_score"] = compute_consensus_potential_evaluation(row)
        row["_score"] = compute_enhanced_draft_score(row)
        row["_pot"] = _pot(p)
        row["_ovr"] = ovr
        rows.append(row)
    rows.sort(key=lambda r: -float(r["_score"]))
    buried = [
        r for i, r in enumerate(rows)
        if i >= 32 and r["_pot"] >= 89 and r["_ovr"] <= 74
    ]
    assert len(buried) >= 2, [(r["_ovr"], r["_pot"]) for r in rows[32:48]]


def test_a_hidden_upside_ceiling_survives_the_clamp():
    player = SimpleNamespace(pipeline_tier="hidden_upside", ratings={"dev_potential": 92})
    assert _clamp_potential99(player, 64.0, 92.0) >= 86.0


def test_a_great_year_cancels_the_age_drop():
    player = _make_player("C", seed=141)
    player.identity.age = 36
    player.games_played = 70
    player._dev_season_points = 80
    player._dev_season_points_surplus = 24
    player._dev_plus_minus = 20
    player._dev_xgf_pct = 0.58
    player._dev_pp_share = 0.24
    losses = [
        _career_aging_decline_try_v3(player, random.Random(seed), None)
        for seed in range(24)
    ]
    assert all(loss is None for loss in losses)

    ordinary = _make_player("C", seed=143)
    ordinary.identity.age = 36
    ordinary.games_played = 70
    ordinary._dev_season_points_surplus = -20
    ordinary_losses = [
        _career_aging_decline_try_v3(ordinary, random.Random(seed), None)
        for seed in range(24)
    ]
    assert any(loss is not None for loss in ordinary_losses)

    younger = _make_player("C", seed=142)
    younger.identity.age = 31
    younger.games_played = 70
    younger._dev_season_points = 12
    younger._dev_season_points_surplus = -15
    for seed in range(12):
        assert _career_aging_decline_try_v3(younger, random.Random(seed), None) is None


def test_a_new_season_resets_the_swing():
    player = _make_player("C", seed=50)
    player.id = "new-year"
    player.identity.age = 27
    player.games_played = 60
    player._dev_season_points_surplus = 26
    player._live_ovr_step = {"season": 2025, "net": 4}
    step = apply_live_overall_step(player, season_id=2026, bucket="nhl", day=0)
    assert step > 0
    assert player._live_ovr_step["season"] == 2026
    assert abs(int(player._live_ovr_step["net"])) == 1


def _depth_kid(position, role, seed):
    player = _make_player(position, seed=seed)
    player.identity.age = 21
    player.role = role
    player.games_played = 55
    player._dev_usage_factor = 0.55
    player._dev_growth_league = "NHL"
    player._dev_season_points = 8
    player.ratings["dev_potential"] = 96
    player.potential = 96 / 99.0
    player._active_dev_season = 2026
    return player


def test_a_positive_third_pair_kid_grows_overall_and_potential():
    player = _depth_kid("D", "third_pair", 160)
    player._dev_plus_minus = 6
    player._dev_xgf_pct = 0.53
    player._dev_gf_on = 40
    player._dev_ga_on = 34
    player._dev_blocks = 70
    assert sheltered_positive_season(player) is True
    profile = resolve_development_profile(player)
    before = float(player.ratings["dev_potential"])
    budget = calculate_season_growth_budget(
        player, profile=profile, rng=random.Random(4), dev_phase="NORMAL"
    )
    assert 0.012 <= budget <= 0.028, budget
    assert float(player.ratings["dev_potential"]) == before

    quiet = _depth_kid("D", "third_pair", 161)
    quiet._dev_plus_minus = -14
    quiet._dev_xgf_pct = 0.43
    quiet._dev_gf_on = 28
    quiet._dev_ga_on = 48
    assert sheltered_positive_season(quiet) is False


def test_a_positive_bottom_six_forward_is_not_judged_on_points():
    player = _depth_kid("C", "bottom_six", 162)
    player._dev_plus_minus = 4
    player._dev_xgf_pct = 0.52
    player._dev_gf_on = 36
    player._dev_ga_on = 32
    player._dev_cf = 500
    player._dev_ca = 460
    assert sheltered_positive_season(player) is True
    budget = calculate_season_growth_budget(
        player, profile=resolve_development_profile(player), rng=random.Random(5), dev_phase="NORMAL"
    )
    assert 0.010 <= budget <= 0.024, budget


def test_a_big_junior_year_does_not_walk_potential_to_99():
    from app.sim_engine.progression.development import season_growth_context

    player = _make_player("C", seed=190)
    player.identity.age = 18
    player.ratings["dev_potential"] = 86
    player.potential = 86 / 99.0
    player._active_dev_season = 2027
    player._prospect_season_stats = {"gp": 55, "points": 90, "goals": 42, "assists": 48}
    before = 86.0
    for _ in range(12):
        season_growth_context(player, bucket="prospect")
    after = float(player.ratings["dev_potential"])
    assert after <= before + 2.01, after
    assert after < 95, after


def test_an_nhl_season_moves_overall_more_than_a_flat_veteran_year():
    from app.sim_engine.progression.development import season_growth_context

    star = _make_player("C", seed=191)
    star.identity.age = 26
    star.role = "top_six"
    star._dev_usage_factor = 1.1
    star._dev_season_points_surplus = 28
    star.ratings["dev_potential"] = 94
    star.potential = 94 / 99.0
    earned = float(season_growth_context(star, bucket="nhl")["gain"])
    assert 2.0 <= earned <= 4.2, earned

    veteran = _make_player("C", seed=192)
    veteran.identity.age = 34
    veteran._dev_season_points_surplus = 0
    quiet = float(season_growth_context(veteran, bucket="nhl")["gain"])
    assert quiet <= 1.0, quiet
    assert earned > quiet + 1.5

    already = _make_player("C", seed=193)
    already.identity.age = 26
    already.role = "top_six"
    already._dev_season_points_surplus = 28
    already.ratings["dev_potential"] = 99
    already.potential = 1.0
    already.ovr = lambda: 0.97
    high = float(season_growth_context(already, bucket="nhl")["gain"])
    assert high > earned * 0.9, (high, earned)


def test_an_ahl_breakout_earns_far_more_than_a_quiet_ahl_year():
    from app.sim_engine.progression.development import season_growth_context

    quiet = _make_player("C", seed=170)
    quiet.identity.age = 21
    quiet._prospect_season_stats = {"gp": 50, "points": 22, "goals": 10, "assists": 12}
    loud = _make_player("C", seed=171)
    loud.identity.age = 21
    loud._prospect_season_stats = {"gp": 50, "points": 62, "goals": 28, "assists": 34}
    quiet_gain = float(season_growth_context(quiet, bucket="ahl")["gain"])
    loud_gain = float(season_growth_context(loud, bucket="ahl")["gain"])
    assert 3.0 <= loud_gain <= 6.0, loud_gain
    assert loud_gain > quiet_gain + 2.0, (loud_gain, quiet_gain)


def test_a_rights_held_junior_grows_from_his_potential_and_his_stats():
    from app.sim_engine.progression.development import season_growth_context

    star = _make_player("C", seed=180)
    star.identity.age = 19
    star.ratings["dev_potential"] = 94
    star.potential = 94 / 99.0
    star._prospect_season_stats = {"gp": 48, "points": 78, "goals": 36, "assists": 42}
    boom = season_growth_context(star, bucket="prospect")
    assert boom["level"] == "rights"
    assert 3.2 <= float(boom["gain"]) <= 6.0, boom

    steady = _make_player("C", seed=181)
    steady.identity.age = 19
    steady.ratings["dev_potential"] = 88
    steady.potential = 88 / 99.0
    steady._prospect_season_stats = {"gp": 48, "points": 32, "goals": 12, "assists": 20}
    plain = float(season_growth_context(steady, bucket="prospect")["gain"])
    assert plain < 2.0, plain
    assert plain < float(boom["gain"])

    blank = _make_player("C", seed=182)
    blank.identity.age = 19
    blank.ratings["dev_potential"] = 99
    blank.potential = 1.0
    # The season column is empty. Potential room is not a +6.
    empty = float(season_growth_context(blank, bucket="prospect")["gain"])
    assert empty <= 1.2, empty


def test_a_young_depth_player_ticks_up_while_under_his_potential():
    player = _depth_kid("D", "third_pair", 163)
    start = int(round(display_rating(player_current_ovr_01(player))))
    assert start + 2 <= 96
    quiet = apply_live_overall_step(player, season_id=2026, bucket="ahl", day=1)
    assert quiet == 0, quiet
    player._prospect_season_stats = {"gp": 50, "points": 48, "goals": 20, "assists": 28}
    step = apply_live_overall_step(player, season_id=2026, bucket="ahl", day=2)
    assert step > 0, step


def test_offseason_growth_stacks_on_the_season():
    player = _make_player("C", seed=210)
    player.id = "summer"
    player.identity.age = 24
    player.games_played = 78
    player._dev_season_points_surplus = 32
    player._active_dev_season = 2026
    player.season_start_ovr = int(round(display_rating(player_current_ovr_01(player))))
    for day in range(8):
        apply_live_overall_step(player, season_id=2026, bucket="nhl", day=day)
    after_season = int(round(display_rating(player_current_ovr_01(player))))
    assert after_season > player.season_start_ovr
    extra = offseason_growth_points(player)
    assert extra >= 1.0, extra
    apply_player_development(player, random.Random(3))
    after_summer = int(round(display_rating(player_current_ovr_01(player))))
    assert after_summer > after_season, (player.season_start_ovr, after_season, after_summer)
    # A second pass does not pay the summer again.
    again = offseason_growth_points(player)
    assert again == 0.0

    quiet = _make_player("C", seed=211)
    quiet.identity.age = 29
    quiet.games_played = 70
    quiet._dev_season_points_surplus = -18
    quiet._active_dev_season = 2026
    assert offseason_growth_points(quiet) == 0.0
