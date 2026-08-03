"""End-to-end audits for aging, scoring distribution, FA negotiate, draft-day trades."""

from __future__ import annotations

import random
from types import SimpleNamespace

from app.sim_engine.engine import SimEngine


def test_aging_pin_survives_playoffs_and_june_calendar():
    from services.franchise_sim import session_age_as_of, sync_player_age_to_season, sync_player_age_to_session

    ident = SimpleNamespace(age=23, birth_year=2002, birth_month=7, birth_day=8)
    player = SimpleNamespace(identity=ident, birth_date="2002-07-08")
    sync_player_age_to_season(player, 2026)
    assert ident.age == 24

    for phase in ("playoff_ready", "playoffs", "offseason", "post_cup"):
        session = SimpleNamespace(
            season_calendar_year=2025,
            phase=phase,
            _year_end_progression_done=True,
            nhl_calendar=[{"iso": "2026-06-12"}],
            calendar_cursor=0,
        )
        assert session_age_as_of(session) == (2026, 9, 15), phase
        assert sync_player_age_to_session(player, session) == 24, phase


def test_ga_balance_mult_no_longer_crushes_stars():
    eng = object.__new__(SimEngine)
    p = SimpleNamespace(id="s1", player_type="power_forward", ratings={"shot": 90, "passing": 70})
    eng._gm_is_certified_sniper = lambda _p: False  # type: ignore
    eng._gm_is_certified_playmaker = lambda _p: False  # type: ignore
    ledger = {"s1": {"g": 40, "a": 18}}
    mult = eng._gm_goal_assist_balance_mult(p, ledger, "T", role="score")
    assert mult >= 0.85, mult


def test_trade_context_sets_draft_day_flag():
    from services.trade_service import _trade_context

    session = SimpleNamespace(
        sim=SimpleNamespace(league=SimpleNamespace()),
        team_by_id={},
        user_team_id="ott",
        season_calendar_year=2025,
        draft_completed=False,
        draft_state={"draft_started": True, "draft_completed": False},
        nhl_calendar=[],
        calendar_cursor=0,
        nhl_regular_season_last_index=192,
        transcendent_tank_pressure={},
        transcendent_draft_prospect_id=None,
        standings={},
        ntc_waivers={},
        player_season_stats={},
    )
    ctx = _trade_context(session)
    assert ctx.get("draft_day_trade") is True


def test_light_path_builds_units_before_toi_in_source():
    import inspect

    src = inspect.getsource(SimEngine._accumulate_light_strength_game_stats)
    units_at = src.find("_gm_build_game_units")
    toi_at = src.find("_gm_allocate_conserved_toi")
    assert units_at > 0 and toi_at > units_at


def test_empty_fa_market_payload_is_not_ready():
    """Version-stamped empty boards must rehydrate (overseas / July 1 runway)."""
    from services.franchise_offseason import STAGE_PAYLOAD_VERSION, _offseason_stage_ready

    session = SimpleNamespace(
        free_agency_market_payload={
            "version": STAGE_PAYLOAD_VERSION["free_agency"],
            "available_count": 0,
            "free_agents": [],
        }
    )
    assert _offseason_stage_ready(session, "free_agency") is False

    session.free_agency_market_payload = {
        "version": STAGE_PAYLOAD_VERSION["free_agency"],
        "available_count": 3,
        "free_agents": [{"player_id": "a"}, {"player_id": "b"}, {"player_id": "c"}],
    }
    assert _offseason_stage_ready(session, "free_agency") is True


def test_elite_defense_does_not_starve_own_shot_share():
    """Best-D / best-G clubs must not become the league's lowest offense via CF share."""
    eng = object.__new__(SimEngine)
    rng = random.Random(7)

    def _off(team, skaters_subset=None):
        return float(getattr(team, "_off", 0.55))

    def _def(team):
        return float(getattr(team, "_def", 0.0))

    eng._team_offense_skill = _off  # type: ignore
    eng._team_defense_suppression = _def  # type: ignore
    eng._team_superstar_offense_impact = lambda _t: 0.2  # type: ignore

    strong_d = SimpleNamespace(_off=0.52, _def=0.13, _gm_cached_star_impact=0.15)
    weak_d = SimpleNamespace(_off=0.52, _def=0.02, _gm_cached_star_impact=0.15)
    h1, a1 = eng._gm_regulation_attempt_split(rng, strong_d, weak_d)
    h2, a2 = eng._gm_regulation_attempt_split(random.Random(7), weak_d, strong_d)
    # Strong defense at home should not receive fewer attempts than the weak-D mirror.
    assert h1 >= a1 - 6, (h1, a1)
    assert h1 >= h2 - 2, (h1, h2, a2)


def test_user_on_clock_gets_trade_down_offers():
    from services.draft_day_trade_offers import generate_draft_day_trade_offers
    import services.franchise_entry_draft as fed
    import services.franchise_sim as fs

    order = []
    for i in range(1, 20):
        tid = "ott" if i == 2 else f"t{i}"
        order.append(
            {
                "team_id": tid,
                "overall_pick": i,
                "round": 1,
                "pick_in_round": i,
                "pick_id": f"p{i}",
            }
        )
    available = [
        {"key": f"pr{i}", "name": f"Prospect {i}", "position": pos, "rank": i}
        for i, pos in enumerate(["C", "LW", "D", "RW", "G", "C", "D", "LW"], start=1)
    ]
    state = {
        "draft_started": True,
        "draft_completed": False,
        "overall_pick": 2,
        "draft_order": order,
        "draft_year": 2026,
        "current_team_id": "ott",
        "team_needs_snapshot": {},
    }
    session = SimpleNamespace(
        user_team_id="ott",
        draft_state=state,
        team_by_id={},
        sim=SimpleNamespace(league=None, rng=None),
    )

    orig_avail = fed._available_entries
    orig_cache = fed._ensure_draft_cache
    orig_board = fed.build_team_draft_board
    orig_needs = fed.calculate_team_needs
    orig_rank = fs.get_cached_draft_class_rankings
    try:
        fs.get_cached_draft_class_rankings = lambda *_a, **_k: {"entries": available}
        fed._available_entries = lambda *_a, **_k: available
        fed._ensure_draft_cache = lambda *_a, **_k: {}
        fed.build_team_draft_board = lambda *_a, **_k: [
            {**available[0], "team_board_rank": 1, "team_board_score": 100},
            {**available[1], "team_board_rank": 2, "team_board_score": 90},
        ]
        fed.calculate_team_needs = lambda *_a, **_k: [{"position": "C"}]
        offers = generate_draft_day_trade_offers(session, state, max_offers=3)
    finally:
        fed._available_entries = orig_avail
        fed._ensure_draft_cache = orig_cache
        fed.build_team_draft_board = orig_board
        fed.calculate_team_needs = orig_needs
        fs.get_cached_draft_class_rankings = orig_rank

    assert len(offers) >= 1, offers
    assert all(o.get("trade_down") or o.get("user_on_clock") for o in offers)
    assert offers[0].get("partner_overall_pick", 0) > 2
    assert len(offers[0].get("target_candidates") or []) == 3
    assert offers[0].get("true_target_prospect_id") == "pr1"
    assert offers[0].get("target_prospect_name") is None
    assert isinstance(offers[0].get("incoming_assets"), list)
    assert len(offers[0].get("incoming_assets") or []) >= 1


def test_solid_player_accepts_near_market_cheap_deal():
    from services.contract_economy import compute_player_demand, evaluate_contract_offer

    solid = SimpleNamespace(
        id="solid1",
        identity=SimpleNamespace(age=28, position=SimpleNamespace(value="C"), name="Solid"),
        ratings={"overall": 78, "dev_potential": 79},
        ovr=lambda: 0.78,
        season_stats={"pts": 42, "gp": 78},
        age=28,
        position="C",
        potential=79,
        morale=68,
        happiness=68,
    )
    team = SimpleNamespace(team_id="ott", roster=[], name="Ottawa")
    demand = compute_player_demand(solid, team, None, context="ufa", days_on_market=6, offer_count=1)
    cheap = round(max(float(demand["min_acceptable_aav_m"]), float(demand["want_aav_m"]) * 0.92), 3)
    result = evaluate_contract_offer(
        solid,
        team,
        {
            "aav_m": cheap,
            "years": max(1, min(3, int(demand["want_years"]))),
            "ntc": False,
            "nmc": False,
            "days_on_market": 6,
            "offer_count": 1,
        },
        None,
        context="ufa",
    )
    assert result.get("accepted") is True, (
        cheap,
        result.get("want_aav_m"),
        result.get("min_acceptable_aav_m"),
        result.get("interest"),
        result.get("reason"),
    )


def test_ahl_career_season_uses_affiliate_team_and_league():
    """AHL-assigned players must show Belleville/AHL, not parent NHL/Ottawa."""
    from services.franchise_sim import (
        _ahl_affiliate_display_name,
        _merge_current_season_into_career_seasons,
        _serialize_player_row,
    )

    ott = SimpleNamespace(
        team_id="ott",
        id="ott",
        city="Ottawa",
        name="Senators",
        abbr="OTT",
        abbreviation="OTT",
    )
    assert _ahl_affiliate_display_name(ott) == "Belleville Senators"

    player = SimpleNamespace(
        id="NHL_cousins",
        identity=SimpleNamespace(
            name="Nick Cousins",
            age=33,
            position=SimpleNamespace(value="C"),
            shoots="L",
            height_cm=180,
            weight_kg=84,
            birth_country="CA",
        ),
        ovr=lambda: 0.70,
        ratings={"dev_potential": 70},
        roster_location="ahl",
        in_minors=True,
        retired=False,
        career_stats={"seasons": []},
        contract=SimpleNamespace(cap_hit_m=1.59, aav_m=1.59, years_remaining=2),
    )
    session = SimpleNamespace(
        season_calendar_year=2025,
        calendar_cursor=120,
        player_season_stats={},
        team_by_id={"ott": ott},
        nhl_calendar=[{"iso": "2026-01-15"}] * 200,
    )
    row = _serialize_player_row(
        player,
        include_ratings=False,
        session=session,
        _team=ott,
        roster_kind="ahl",
    )
    assert row.get("league") == "AHL"
    assert "Belleville" in str(row.get("team_name") or "")
    ss = row.get("season_stats") or {}
    assert ss.get("league") == "AHL"
    assert "Belleville" in str(ss.get("team_name") or ss.get("team") or "")
    current = [s for s in (row.get("career_seasons") or []) if s.get("is_current_season")]
    assert current, row.get("career_seasons")
    assert current[-1].get("league") == "AHL"
    assert "Belleville" in str(current[-1].get("team") or "")

    # Explicit merge helper respects compact league/team.
    host = {"position": "C", "career_seasons": []}
    _merge_current_season_into_career_seasons(
        host,
        {
            "gp": 40,
            "g": 12,
            "a": 18,
            "pts": 30,
            "league": "AHL",
            "team_name": "Belleville Senators",
            "plusMinus": 4,
            "pim": 10,
        },
        session=session,
        team=ott,
    )
    assert host["career_seasons"][-1]["league"] == "AHL"
    assert host["career_seasons"][-1]["team"] == "Belleville Senators"
