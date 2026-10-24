"""Young-player growth reads the league they actually played."""

from types import SimpleNamespace

from services.franchise_offseason import (
    _dev_age_ppg_factor,
    _dev_nhl_eq_save_pct,
    _dev_season_stats,
    _dev_stamp_season_production,
)
from services.prospect_in_season_growth import PROSPECT_MAX_AGE_AHL, iter_growth_prospects
from app.sim_engine.progression.development import (
    _IN_SEASON_POOL_SHARE,
    _elite_growth_scale,
    apply_in_season_development_pulse,
    assign_prospect_realization,
    calculate_season_growth_budget,
    expected_season_points_for_player,
    nudge_prospect_realization,
    prospect_year_cap,
    prospect_year_payment,
    resolve_development_profile,
    roll_prospect_realization,
    settle_season_displayed_ovr,
)


def _skater(age, ovr, pot, **extra):
    player = SimpleNamespace(
        id="p1",
        identity=SimpleNamespace(age=age, position="F"),
        position="F",
        potential=pot,
        ratings={"offense": ovr, "skating": ovr, "IQ": ovr},
        ovr=lambda: ovr / 99.0,
        games_played=0,
        gp=0,
        role="fourth",
        psych=SimpleNamespace(morale=0.62),
        production_score=0.2,
        toi_quality=0.2,
        _dev_growth_league="",
        _dev_usage_factor=1.0,
    )
    for key, value in extra.items():
        setattr(player, key, value)
    return player


def test_no_games_resets_last_year():
    player = _skater(20, 78, 90, production_score=0.2, toi_quality=0.15, _dev_season_points_surplus=-12)
    session = SimpleNamespace(player_season_stats={}, season_calendar_year=2026, sim=None)
    _dev_stamp_season_production(session, player)
    assert player.production_score == 0.5
    assert player.toi_quality == 0.5
    assert player._dev_season_points_surplus == 0.0
    assert player.games_played == 0


def test_ahl_point_per_game_converts_and_rates_the_ice():
    player = _skater(20, 78, 92)
    player._prospect_season_stats = {
        "league": "AHL",
        "gp": 60,
        "goals": 25,
        "assists": 35,
        "points": 60,
        "toi_sec": 18 * 60 * 60,
    }
    session = SimpleNamespace(player_season_stats={}, season_calendar_year=2026, sim=None)
    stats = _dev_season_stats(session, player)
    assert stats["league"] == "AHL"
    assert abs(stats["nhl_eq_ppg"] - 1.0 * 0.47 * _dev_age_ppg_factor(20)) < 0.02
    assert abs(stats["age_ppg_factor"] - 0.85) < 0.02
    _dev_stamp_season_production(session, player)
    assert player._dev_growth_league == "AHL"
    assert player.toi_quality >= 0.90
    assert player.games_played == 60
    assert player.production_score > 0.5


def test_chl_points_use_the_junior_factor():
    player = _skater(18, 70, 88)
    player._prospect_season_stats = {
        "league": "OHL",
        "gp": 60,
        "goals": 40,
        "assists": 50,
        "points": 90,
        "toi_sec": 20 * 60 * 60,
    }
    session = SimpleNamespace(player_season_stats={}, season_calendar_year=2026, sim=None)
    stats = _dev_season_stats(session, player)
    assert abs(stats["nhl_eq_ppg"] - 1.5 * 0.30 * _dev_age_ppg_factor(18)) < 0.02


def test_fourth_line_expectation_uses_deployment_for_all_ages():
    kid = _skater(21, 78, 92, games_played=70, gp=70, _dev_usage_factor=9 / 15.5, _dev_growth_league="NHL")
    vet = _skater(29, 78, 92, games_played=70, gp=70, _dev_usage_factor=9 / 15.5, _dev_growth_league="NHL")
    kid_ppg = expected_season_points_for_player(kid, 70) / 70
    vet_ppg = expected_season_points_for_player(vet, 70) / 70
    assert kid_ppg < 0.40
    assert abs(vet_ppg - kid_ppg) < 0.02


def test_fourth_liner_budget_is_a_couple_of_points_and_a_veteran_is_not():
    kid = _skater(
        21,
        78,
        92,
        games_played=70,
        gp=70,
        role="fourth",
        toi_quality=0.18,
        production_score=0.32 + 0.35 * 0.55,
        _dev_usage_factor=9 / 15.5,
        _dev_growth_league="NHL",
        _dev_season_points_surplus=2.0,
    )
    vet = _skater(
        29,
        78,
        92,
        games_played=70,
        gp=70,
        role="fourth",
        toi_quality=0.18,
        production_score=0.32 + 0.35 * 0.55,
        _dev_usage_factor=9 / 15.5,
        _dev_growth_league="NHL",
        _dev_season_points_surplus=2.0,
    )
    import random

    kid_budget = calculate_season_growth_budget(
        kid, profile=resolve_development_profile(kid), rng=random.Random(3), dev_phase="NORMAL"
    )
    vet_budget = calculate_season_growth_budget(
        vet, profile=resolve_development_profile(vet), rng=random.Random(3), dev_phase="NORMAL"
    )
    assert 0.018 <= kid_budget <= 0.040, kid_budget
    assert vet_budget < kid_budget
    assert vet_budget < 0.016, vet_budget


def test_prospect_year_scales_with_the_gap():
    player = _skater(20, 78, 92)
    cap = prospect_year_cap(player, "NORMAL")
    assert 4.0 <= cap * 99.0 <= 7.5, cap * 99.0


def test_ahl_prospect_age_reaches_24_and_stops():
    def _p(age, name):
        return SimpleNamespace(identity=SimpleNamespace(age=age), retired=False, id=name)

    ahl_23 = _p(23, "a23")
    ahl_24 = _p(24, "a24")
    ahl_25 = _p(25, "a25")
    pool_23 = _p(23, "p23")
    league = SimpleNamespace(
        development_leagues=[],
        teams=[
            SimpleNamespace(
                ahl_roster=[ahl_23, ahl_24, ahl_25],
                prospect_pool=[pool_23],
                echl_roster=[],
            )
        ],
    )
    found = list(iter_growth_prospects(league))
    assert ahl_23 in found and ahl_24 in found
    assert ahl_25 not in found
    assert pool_23 not in found
    assert PROSPECT_MAX_AGE_AHL == 24


def _counting_line(league, *, points, gp=60, save_pct=None, starts=0):
    row = {
        "league": league,
        "gp": gp,
        "goals": points // 2,
        "assists": points - points // 2,
        "points": points,
        "toi_sec": 18 * 60 * gp,
    }
    if save_pct is not None:
        row["save_pct"] = save_pct
        row["starts"] = starts or gp
    return row


def test_seventeen_year_old_converts_higher_than_a_twenty_year_old():
    session = SimpleNamespace(player_season_stats={}, season_calendar_year=2026, sim=None)
    young = _skater(17, 74, 90)
    older = _skater(20, 74, 90)
    for player in (young, older):
        player._prospect_season_stats = _counting_line("OHL", points=72)
    young_stats = _dev_season_stats(session, young)
    older_stats = _dev_season_stats(session, older)
    assert young_stats["nhl_eq_ppg"] > older_stats["nhl_eq_ppg"] * 1.25
    _dev_stamp_season_production(session, young)
    _dev_stamp_season_production(session, older)
    assert prospect_year_payment(young) > prospect_year_payment(older) * 1.25


def test_fractional_year_floors_and_the_rest_carries():
    import random

    from app.sim_engine.entities.player import display_rating, player_current_ovr_01
    from test_chapter_attributes import _make_player

    player = _make_player("C", seed=11)
    start = float(player_current_ovr_01(player))
    start_disp = int(display_rating(start))
    settle_season_displayed_ovr(
        player, random.Random(1), earned_display=1.4, start_01=start, finalize=True
    )
    shown = int(display_rating(player_current_ovr_01(player))) - start_disp
    assert shown == 1, shown
    assert abs(float(player._dev_ovr_carry) - 0.4) < 0.05
    # Next year adds the 0.4 before it floors. 0.4 + 0.7 = 1.1 → +1, carry 0.1.
    player._dev_carry_locked = False
    start2 = float(player_current_ovr_01(player))
    start2_disp = int(display_rating(start2))
    settle_season_displayed_ovr(
        player, random.Random(2), earned_display=0.7, start_01=start2, finalize=True
    )
    assert int(display_rating(player_current_ovr_01(player))) - start2_disp == 1
    assert abs(float(player._dev_ovr_carry) - 0.1) < 0.05


def test_reroll_pays_the_new_target_minus_what_was_spent(monkeypatch):
    import random

    from app.sim_engine.progression import development as dev
    from test_chapter_attributes import _make_player

    budgets = iter((0.014, 0.016, 0.020))

    def _fixed(*_args, **_kwargs):
        return next(budgets)

    monkeypatch.setattr(dev, "calculate_season_growth_budget", _fixed)

    def _drain(player, gp, times=25, checkpoint_gp=None):
        player.games_played = gp
        player.gp = gp
        rng = random.Random(4)
        for _ in range(times):
            apply_in_season_development_pulse(
                player, rng, checkpoint_gp=checkpoint_gp if checkpoint_gp is not None else None
            )

    stepped = _make_player("C", seed=21)
    stepped.production_score = 0.55
    stepped.toi_quality = 0.7
    stepped._dev_growth_league = "NHL"
    stepped._nhl_season_plan = {
        "phase": "NORMAL",
        "annual": 0.012,
        "annual_at_gp": [],
        "in_season_remaining": 0.012 * _IN_SEASON_POOL_SHARE,
    }
    for checkpoint in (8, 20, 41, 61):
        _drain(stepped, checkpoint, checkpoint_gp=checkpoint if checkpoint in (20, 41, 61) else None)

    steady = _make_player("C", seed=21)
    steady.production_score = 0.55
    steady.toi_quality = 0.7
    steady._dev_growth_league = "NHL"
    steady.games_played = 61
    steady.gp = 61
    final = 0.020
    steady._nhl_season_plan = {
        "phase": "NORMAL",
        "annual": final,
        "annual_at_gp": [20, 41, 61],
        "in_season_remaining": final * _IN_SEASON_POOL_SHARE,
    }
    _drain(steady, 61, times=40)
    stepped_gain = float(stepped._in_season_growth_spent_01) * 99.0
    steady_gain = float(steady._in_season_growth_spent_01) * 99.0
    assert abs(stepped_gain - steady_gain) <= 0.3, (stepped_gain, steady_gain)


def test_ordinary_ahl_and_chl_goalies_are_not_a_breakout():
    session = SimpleNamespace(player_season_stats={}, season_calendar_year=2026, sim=None)

    def _goalie(league):
        player = _skater(20, 78, 92)
        player.identity.position = "G"
        player.position = "G"
        player._prospect_season_stats = _counting_line(league, points=0, gp=40, save_pct=0.915, starts=40)
        return player

    ahl = _goalie("AHL")
    chl = _goalie("OHL")
    ahl_stats = _dev_season_stats(session, ahl)
    chl_stats = _dev_season_stats(session, chl)
    assert abs(ahl_stats["save_pct"] - _dev_nhl_eq_save_pct(0.915, "AHL")) < 0.004
    assert abs(chl_stats["save_pct"] - _dev_nhl_eq_save_pct(0.915, "OHL")) < 0.004
    assert ahl_stats["save_pct"] < 0.910
    assert chl_stats["save_pct"] < 0.910
    assert ahl_stats["goalie_modest"] and chl_stats["goalie_modest"]
    _dev_stamp_season_production(session, ahl)
    _dev_stamp_season_production(session, chl)
    skater = _skater(20, 78, 92, _dev_age_ppg_factor=1.0, _dev_growth_league="AHL")
    skater_year = prospect_year_payment(skater) * 99.0
    for goalie in (ahl, chl):
        paid = prospect_year_payment(goalie) * 99.0
        assert 0.8 <= paid <= 3.2, paid
        assert paid < skater_year * 0.6


def test_growth_does_not_lift_a_78_92_ceiling_with_overall():
    import random

    from app.sim_engine.progression.development import (
        apply_player_development,
        apply_prospect_in_season_pulse,
        apply_prospect_potential_review,
    )
    from app.sim_engine.progression.potential import read_player_potential99

    player = _skater(20, 78, 92, role="top_line", toi_quality=1.0, production_score=0.58)
    player.games_played = 60
    player.gp = 60
    player._dev_growth_league = "AHL"
    player._dev_usage_factor = 1.05
    player._dev_age_ppg_factor = _dev_age_ppg_factor(20)
    player.season_start_ovr = 78
    player._prospect_season_stats = _counting_line("AHL", points=60)
    player._prospect_projected_stats = {"gp": 60, "points": 58, "ppg": 0.97}
    start_pot = read_player_potential99(player)
    rng = random.Random(1)
    for _ in range(22):
        apply_prospect_in_season_pulse(player, rng, 2026)
        plan = getattr(player, "_prospect_season_plan", None) or {}
        if int(plan.get("pulses") or 0) % 4 == 0:
            apply_prospect_potential_review(player, rng, 2026)
    player._active_dev_season = 2026
    apply_player_development(player, rng)
    end_pot = read_player_potential99(player)
    assert end_pot <= start_pot + 1.15, (start_pot, end_pot)


def test_realization_falls_short_of_the_scouted_potential():
    import random

    within = mid = bust = 0
    realizations = []
    gap = 16.0
    for i in range(4000):
        realization = roll_prospect_realization(random.Random(i), gap)
        short = (1.0 - realization) * gap
        realizations.append(realization)
        if short <= 2.0:
            within += 1
        elif short <= 8.0:
            mid += 1
        else:
            bust += 1
    n = 4000
    assert 0.35 <= within / n <= 0.42
    assert mid / n > 0.45
    assert bust / n >= 0.08
    assert 0.68 <= sum(realizations) / n <= 0.82

    player = _skater(18, 70, 88)
    player.ratings["dev_potential"] = 88
    shown = player.ratings["dev_potential"]
    assign_prospect_realization(player, random.Random(4), force=True)
    assert player.ratings["dev_potential"] == shown
    assert player._dev_real_ceiling < shown
    assert player._dev_real_ceiling >= 70
    cap = prospect_year_cap(player, "NORMAL") * 99.0
    assert cap < (shown - 70) * 0.40


def test_good_and_bad_seasons_nudge_realization_not_the_shown_potential():
    import random

    player = _skater(19, 72, 90, games_played=50, gp=50)
    player.ratings["dev_potential"] = 90
    assign_prospect_realization(player, random.Random(2), force=True)
    player._dev_realization = 0.70
    base = float(player._dev_realization_base_ovr)
    pot = float(player._dev_realization_pot)
    player._dev_real_ceiling = base + 0.70 * (pot - base)
    start = 0.70
    player._dev_season_points_surplus = 10
    nudge_prospect_realization(player, 2026)
    assert abs(float(player._dev_realization) - (start + 0.05)) < 0.001
    assert player.ratings["dev_potential"] == 90
    player._dev_season_points_surplus = -10
    nudge_prospect_realization(player, 2027)
    assert abs(float(player._dev_realization) - start) < 0.001


def test_growth_above_84_slows_down():
    import math

    assert abs(_elite_growth_scale(84) - 1.0) < 1e-9
    assert abs(_elite_growth_scale(85) - math.exp(-0.25)) < 1e-9
    assert abs(_elite_growth_scale(87) - math.exp(-0.75)) < 1e-6
    assert abs(_elite_growth_scale(89) - math.exp(-1.25)) < 1e-6
    below = _skater(22, 80, 88)
    above = _skater(22, 87, 95)
    below_cap = prospect_year_cap(below, "NORMAL") * 99.0
    above_cap = prospect_year_cap(above, "NORMAL") * 99.0
    assert below_cap > above_cap
    assert abs(above_cap - (8 * 0.40 * _elite_growth_scale(87))) < 0.2
