"""Regression checks for birth-date aging and home-team UFA desk."""

from __future__ import annotations

from types import SimpleNamespace

from services.franchise_sim import _age_years_as_of, sync_player_age_to_season
from services.contract_economy import build_own_ufa_resign_row
from services.fa_market_engine import _serious_cpu_offer_aav
import random


def test_sanderson_kucherov_sept_2027_ages():
    assert _age_years_as_of((2002, 7, 8), 2027, 9, 15) == 25
    assert _age_years_as_of((1993, 6, 17), 2027, 9, 15) == 34


def test_sync_player_age_from_birth_date():
    ident = SimpleNamespace(age=24, birth_year=2002, birth_month=7, birth_day=8)
    player = SimpleNamespace(identity=ident, birth_date="2002-07-08")
    assert sync_player_age_to_season(player, 2027) == 25
    assert ident.age == 25


def test_offseason_serialize_does_not_undo_year_end_aging():
    """After year-end tick, ages stay on next Sept 15 until the season year bumps."""
    from services.franchise_sim import session_age_as_of, sync_player_age_to_session

    ident = SimpleNamespace(age=24, birth_year=2002, birth_month=7, birth_day=8)
    player = SimpleNamespace(identity=ident, birth_date="2002-07-08")
    session = SimpleNamespace(
        season_calendar_year=2025,
        phase="offseason",
        _year_end_progression_done=True,
        nhl_calendar=[],
        calendar_cursor=0,
    )
    y, m, d = session_age_as_of(session)
    assert (y, m, d) == (2026, 9, 15)
    assert sync_player_age_to_session(player, session) == 24
    assert ident.age == 24


def test_year_end_age_pin_beats_live_june_calendar():
    """Stale June calendar must not roll Sanderson back after the Sept-15 year-end tick."""
    from services.franchise_sim import session_age_as_of, sync_player_age_to_session

    ident = SimpleNamespace(age=25, birth_year=2002, birth_month=7, birth_day=8)
    player = SimpleNamespace(identity=ident, birth_date="2002-07-08")
    # Completed 2025-26 season: calendar still pointing at June, year-end aging done.
    session = SimpleNamespace(
        season_calendar_year=2025,
        phase="playoff_ready",
        _year_end_progression_done=True,
        nhl_calendar=[{"iso": "2026-06-15"}, {"iso": "2026-06-16"}],
        calendar_cursor=0,
    )
    assert session_age_as_of(session) == (2026, 9, 15)
    assert sync_player_age_to_session(player, session) == 24
    assert ident.age == 24


def test_year_end_resync_cannot_undo_ages_before_flag_race():
    """Simulate the old bug: tick ages to SY+1, then resync without pin → age -1."""
    from services.franchise_sim import session_age_as_of, sync_player_age_to_season, sync_player_age_to_session

    ident = SimpleNamespace(age=23, birth_year=2002, birth_month=7, birth_day=8)
    player = SimpleNamespace(identity=ident, birth_date="2002-07-08")
    sync_player_age_to_season(player, 2026)  # year-end tick
    assert ident.age == 24
    # Fixed pin: once flag is set (even in playoffs), age stays 24
    session = SimpleNamespace(
        season_calendar_year=2025,
        phase="playoff_ready",
        _year_end_progression_done=True,
        nhl_calendar=[{"iso": "2026-06-01"}],
        calendar_cursor=0,
    )
    assert session_age_as_of(session) == (2026, 9, 15)
    assert sync_player_age_to_session(player, session) == 24


def test_cap_snapshot_does_not_reinflate_pending_july1_via_total_cap_hit():
    from app.sim_engine.economy.cap_engine import calculate_team_cap_snapshot

    kept = SimpleNamespace(
        retired=False,
        pending_july1_expiry=False,
        contract={"aav_m": 5.0, "cap_hit_m": 5.0},
        cap_hit_m=5.0,
    )
    pending = SimpleNamespace(
        retired=False,
        pending_july1_expiry=True,
        contract={"aav_m": 40.0, "cap_hit_m": 40.0, "pending_july1_expiry": True},
        cap_hit_m=40.0,
    )
    for p in (kept, pending):
        for flag in ("is_buried", "buried", "in_minors", "on_ir", "on_ltir", "is_ir", "is_ltir"):
            setattr(p, flag, False)
    team = SimpleNamespace(
        roster=[kept, pending],
        ahl_roster=[],
        echl_roster=[],
        total_cap_hit=95.0,  # stale inflated mirror — must NOT become active hit
        salary_cap_m=95.5,
        retained_salary_records=[],
        buyouts=[],
        bonus_overages=[],
        other_dead_cap_m=0,
    )
    league = SimpleNamespace(salary_cap_m=95.5, salary_floor_m=70.0)
    snap = calculate_team_cap_snapshot(team, league=league)
    # Pending excluded → ~5M hit → usable space well above zero (not $0).
    assert float(snap.get("activeRosterCapHit") or 0) < 6.0
    assert float(snap.get("usableCapSpace") or snap.get("capSpace") or 0) > 50.0


def test_depth_fa_market_value_near_league_min():
    from services.contract_economy import LEAGUE_MINIMUM_AAV_M, compute_market_value

    depth = SimpleNamespace(
        id="cheap",
        identity=SimpleNamespace(age=28, position=SimpleNamespace(value="LW")),
        ratings={"overall": 71},
        ovr=lambda: 0.71,
        season_stats={"pts": 12, "gp": 60},
        age=28,
        position="LW",
    )
    val = compute_market_value(depth)
    assert val <= LEAGUE_MINIMUM_AAV_M + 1.25, f"depth FA too expensive: {val}"


def test_replacement_demand_decays_without_2m_floor():
    """Low-rated veterans must not cling to an artificial ~$2M ask."""
    from services.contract_economy import LEAGUE_MINIMUM_AAV_M, compute_player_demand

    vet = SimpleNamespace(
        id="repl1",
        identity=SimpleNamespace(age=33, position=SimpleNamespace(value="C"), name="Repl"),
        ratings={"overall": 58, "dev_potential": 60},
        ovr=lambda: 0.58,
        season_stats={"pts": 8, "gp": 40},
        age=33,
        position="C",
        potential=60,
    )
    early = compute_player_demand(vet, None, None, context="ufa", days_on_market=0, offer_count=0)
    late = compute_player_demand(vet, None, None, context="ufa", days_on_market=22, offer_count=0)
    assert early["want_aav_m"] < 2.0, f"replacement opening ask too high: {early['want_aav_m']}"
    assert late["want_aav_m"] <= early["want_aav_m"]
    assert late["min_acceptable_aav_m"] <= LEAGUE_MINIMUM_AAV_M + 0.25
    assert late["want_years"] <= 2


def test_elite_demand_does_not_collapse_to_minimum():
    from services.contract_economy import LEAGUE_MINIMUM_AAV_M, compute_player_demand

    star = SimpleNamespace(
        id="elite1",
        identity=SimpleNamespace(age=27, position=SimpleNamespace(value="D"), name="Star"),
        ratings={"overall": 93, "dev_potential": 94},
        ovr=lambda: 0.93,
        season_stats={"pts": 85, "gp": 80},
        age=27,
        position="D",
        potential=94,
    )
    late = compute_player_demand(star, None, None, context="ufa", days_on_market=28, offer_count=0)
    assert late["want_aav_m"] >= 6.0, f"elite ask collapsed unrealistically: {late['want_aav_m']}"
    assert late["min_acceptable_aav_m"] > LEAGUE_MINIMUM_AAV_M * 4


def test_extension_reserve_protects_core_expirings():
    from services.contract_economy import (
        compute_priority_extension_reserve_m,
        normalize_contract_dict,
    )

    star = SimpleNamespace(
        id="core1",
        identity=SimpleNamespace(age=26, position="C", name="Core"),
        position="C",
        age=26,
        ovr=lambda: 0.91,
        ratings={"dev_potential": 92},
        season_stats={"pts": 90, "gp": 82},
        contract=normalize_contract_dict({"aav_m": 8.0, "cap_hit_m": 8.0, "years_remaining": 1}),
        retired=False,
    )
    depth = SimpleNamespace(
        id="d1",
        identity=SimpleNamespace(age=29, position="LW", name="Depth"),
        position="LW",
        age=29,
        ovr=lambda: 0.74,
        ratings={"dev_potential": 74},
        season_stats={"pts": 20, "gp": 70},
        contract=normalize_contract_dict({"aav_m": 1.2, "cap_hit_m": 1.2, "years_remaining": 1}),
        retired=False,
    )
    for flag in ("is_buried", "buried", "in_minors", "on_ir", "on_ltir"):
        setattr(star, flag, False)
        setattr(depth, flag, False)
    team = SimpleNamespace(
        team_id="AAA",
        id="AAA",
        roster=[star, depth],
        rfa_rights=[],
        buyout_cap_hits=[],
        salary_cap_m=95.0,
    )
    league = SimpleNamespace(teams=[team], salary_cap_m=95.0, cap_floor_m=70.0, free_agents=[])
    reserve = compute_priority_extension_reserve_m(team, league)
    assert reserve > 0.0
    # Depth expirings must not dominate the reserve; core star drives it.
    assert reserve >= 4.0


def test_pending_july1_excluded_from_active_cap_hit():
    from app.sim_engine.economy.cap_engine import team_active_roster_cap_hit_millions

    kept = SimpleNamespace(
        retired=False,
        pending_july1_expiry=False,
        contract={"aav_m": 5.0, "cap_hit_m": 5.0},
        cap_hit_m=5.0,
    )
    pending = SimpleNamespace(
        retired=False,
        pending_july1_expiry=True,
        contract={"aav_m": 8.0, "cap_hit_m": 8.0, "pending_july1_expiry": True},
        cap_hit_m=8.0,
    )
    # Bypass roster_compliance by stubbing active check via attributes used in fallback
    for p in (kept, pending):
        for flag in ("is_buried", "buried", "in_minors", "on_ir", "on_ltir", "is_ir", "is_ltir"):
            setattr(p, flag, False)
    team = SimpleNamespace(roster=[kept, pending])
    hit = team_active_roster_cap_hit_millions(team)
    assert abs(hit - 5.0) < 0.01


def test_star_cap_strapped_cpu_does_not_lowball():
    rng = random.Random(0)
    assert (
        _serious_cpu_offer_aav(fair=11.5, ovr=95, cap_space_m=2.0, discount=0.9, rng=rng)
        is None
    )
    offer = _serious_cpu_offer_aav(fair=11.5, ovr=95, cap_space_m=18.0, discount=1.0, rng=rng)
    assert offer is not None
    assert offer >= 8.0


def test_cpu_negotiate_star_ceiling_uses_most_of_cap_space():
    """Stars must be closable when a club has space — 35% ceiling was a permanent stall."""
    from services.contract_economy import _cpu_negotiate_offer, compute_market_value

    player = SimpleNamespace(
        id="star",
        identity=SimpleNamespace(age=28, name="Star", position=SimpleNamespace(value="C")),
        ratings={"overall": 94, "dev_potential": 94},
        ovr=lambda: 0.94,
        season_stats={"pts": 110, "gp": 82},
        age=28,
        position="C",
        asking_aav_m=11.0,
    )
    team = SimpleNamespace(team_id="CPU", roster=[], prospect_pool=[], salary_cap_m=95.0)
    league = SimpleNamespace(teams=[team], free_agents=[], salary_cap_m=95.0)
    market = compute_market_value(player, league)
    ctx = {"cap_space_m": 20.0, "slots_remaining": 5, "window": "contender", "need_score": {"C": 0.8}}
    # Patch evaluate to accept once offer reaches ~market
    import services.contract_economy as ce

    def _ev(player, team, offer, league, context="ufa"):
        aav = float(offer.get("aav_m") or 0)
        if aav >= market * 0.92:
            return {"accepted": True}
        return {"accepted": False, "counter_offer": {"aav_m": round(market * 1.02, 3), "years": 6}}

    old = ce.evaluate_contract_offer
    ce.evaluate_contract_offer = _ev
    try:
        ok, final, yrs = _cpu_negotiate_offer(team, player, league, start_aav=market * 0.85, years=5, ctx=ctx)
    finally:
        ce.evaluate_contract_offer = old
    assert ok, f"star should close with $20M space; final={final} market={market}"
    assert final <= 20.0


def test_own_ufa_resign_row_negotiable():
    player = SimpleNamespace(
        id="p1",
        identity=SimpleNamespace(age=28, name="Drake Batherson", position=SimpleNamespace(value="RW")),
        ratings={"overall": 91},
        ufa_from_team_id="ott",
        ovr=lambda: 0.91,
    )
    team = SimpleNamespace(team_id="ott", name="Ottawa Senators")
    row = build_own_ufa_resign_row(player, team, 2027, None)
    assert row["own_ufa"] is True
    assert row["can_negotiate"] is True
    assert row["expiry_status"] == "UFA"
