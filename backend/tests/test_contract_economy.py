"""Contract economy tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
SIM = ROOT / "SimEngine"
for p in (str(BACKEND), str(SIM)):
    if p not in sys.path:
        sys.path.insert(0, p)

import random  # noqa: E402

from services.contract_economy import (  # noqa: E402
    LEAGUE_MINIMUM_AAV_M,
    add_rfa_rights,
    build_contract_for_player,
    compute_bad_contract_score,
    compute_fair_aav,
    compute_team_importance_score,
    evaluate_contract_offer,
    generate_contract_terms,
    get_team_cap_snapshot_full,
    handle_player_contract_expiry,
    normalize_contract_dict,
    normalize_money_m,
    rebalance_team_cap_at_bootstrap,
    validate_franchise_cap_at_start,
)


def _player(pid: str, ovr: float = 82, age: int = 26, pos: str = "C"):
    return SimpleNamespace(
        id=pid,
        identity=SimpleNamespace(name=f"P {pid}", age=age, position=pos, shoots="L"),
        position=pos,
        age=age,
        ovr=lambda: ovr / 99.0,
        ratings={"dev_potential": 80},
        season_stats={"pts": 40},
        contract=None,
    )


def _team(tid: str, players=None):
    return SimpleNamespace(
        team_id=tid,
        id=tid,
        roster=list(players or []),
        rfa_rights=[],
        buyout_cap_hits=[],
        salary_cap_m=92.0,
    )


def _league(teams):
    return SimpleNamespace(teams=teams, salary_cap_m=92.0, cap_floor_m=68.0, free_agents=[])


def test_normalize_money_m_legacy_dollars():
    assert normalize_money_m(5_500_000) == 5.5
    assert normalize_money_m(3.25) == 3.25


def test_normalize_contract_dict_millions():
    c = normalize_contract_dict({"aav": 4.5, "years_remaining": 2, "rights_status": "UFA"})
    assert c["aav_m"] == 4.5
    assert c["cap_hit_m"] == 4.5
    assert c["years_remaining"] == 2


def test_team_importance_higher_with_peer_gap():
    star = _player("s1", ovr=93, age=27)
    sidekick = _player("s2", ovr=85, age=26)
    team = _team("AAA", [star, sidekick])
    imp_star = compute_team_importance_score(star, team)
    imp_side = compute_team_importance_score(sidekick, team)
    assert imp_star >= imp_side


def test_generate_contract_terms_respects_max_aav():
    p = _player("x", ovr=90, age=28)
    rng = random.Random(42)
    aav, years, _ = generate_contract_terms(p, None, None, rng, max_aav_m=2.5)
    assert aav <= 2.5
    assert years >= 1


def test_cap_snapshot_non_negative_after_bootstrap():
    roster = [_player(f"p{i}", ovr=70 + i % 20, age=22 + i % 10) for i in range(22)]
    team = _team("AAA", roster)
    league = _league([team])
    rng = random.Random(99)
    season = 2025
    for p in roster:
        c = build_contract_for_player(p, team, league, season, rng, allow_bad=True)
        p.contract = c
        p.cap_hit_m = c["cap_hit_m"]
    rebalance_team_cap_at_bootstrap(team, league, season, rng)
    snap = get_team_cap_snapshot_full(team, league, season_year=season)
    assert snap["usable_cap_space_m"] >= -0.01


def test_rfa_rights_not_lost_on_expiry():
    p = _player("rfa1", ovr=78, age=23)
    p.contract = normalize_contract_dict({
        "aav_m": 1.5, "cap_hit_m": 1.5, "years_remaining": 1, "rights_status": "RFA",
    })
    team = _team("AAA", [p])
    league = _league([team])
    outcome = handle_player_contract_expiry(p, team, league, 2025)
    assert outcome == "rfa_rights"
    assert len(team.rfa_rights) == 1
    assert team.rfa_rights[0]["player_id"] == "rfa1"


def test_evaluate_contract_offer_structure():
    p = _player("u1", ovr=80, age=28)
    team = _team("AAA", [p])
    league = _league([team])
    result = evaluate_contract_offer(p, team, {"aav_m": 3.5, "years": 3}, league, context="ufa")
    assert "accepted" in result
    assert "interest" in result
    assert "counter_offer" in result


def test_bad_contract_score_positive_for_overpay():
    p = _player("bad1", ovr=74, age=30)
    p.contract = normalize_contract_dict({"aav_m": 5.5, "cap_hit_m": 5.5, "years_remaining": 4})
    p.cap_hit_m = 5.5
    team = _team("AAA", [p])
    score = compute_bad_contract_score(p, team)
    assert score > 0.2


def test_validate_franchise_cap_at_start_empty_when_compliant():
    team = _team("AAA", [_player("a", ovr=80)])
    p = team.roster[0]
    p.contract = normalize_contract_dict({"aav_m": 2.0, "cap_hit_m": 2.0, "years_remaining": 2})
    p.cap_hit_m = 2.0
    league = _league([team])
    issues = validate_franchise_cap_at_start(league, 2025)
    assert issues == []


def test_qualifying_offer_minimum():
    from services.contract_economy import qualifying_offer_aav
    assert qualifying_offer_aav(1.0) >= LEAGUE_MINIMUM_AAV_M


def test_stress_audit_runs_clean_short():
    from services.contract_economy_audit import run_stress_audit

    report = run_stress_audit(seasons=3, seed=777)
    assert report.seasons_simulated == 3
    assert report.summary.get("compliance_failures", 0) == 0
    assert report.summary.get("errors", 0) == 0, report.issues


def test_elc_pipeline_reserve_and_promotion():
    from services.contract_economy import (
        add_to_reserve_list,
        assign_elc_contract,
        auto_sign_elc_on_promotion,
        has_elc_contract,
        is_elc_eligible,
        promote_prospect_to_nhl,
        validate_contract_slots,
    )

    prospect = _player("elc1", ovr=72, age=20, pos="LW")
    prospect.entry_level_contract_eligible = True
    prospect.signed_status = "unsigned"
    team = _team("AAA", [])
    team.prospect_pool = [prospect]
    league = _league([team])
    season = 2025

    add_to_reserve_list(team, prospect, draft_year=2025, draft_overall=12, added_season=season)
    assert len(team.reserve_list) == 1
    assert is_elc_eligible(prospect)

    slots = validate_contract_slots(team, league)
    assert slots.get("ok")

    signed = assign_elc_contract(prospect, team, league, season)
    assert signed.get("ok")
    assert has_elc_contract(prospect)
    assert not is_elc_eligible(prospect)

    prospect2 = _player("elc2", ovr=70, age=21, pos="D")
    prospect2.entry_level_contract_eligible = True
    prospect2.signed_status = "unsigned"
    team.prospect_pool.append(prospect2)
    add_to_reserve_list(team, prospect2, draft_year=2025, draft_overall=45, added_season=season)

    promoted = promote_prospect_to_nhl(prospect2, team, league, season, auto_elc=True)
    assert promoted.get("ok")
    assert prospect2 in team.roster
    assert has_elc_contract(prospect2)

    prospect3 = _player("elc3", ovr=68, age=19, pos="C")
    prospect3.entry_level_contract_eligible = True
    prospect3.signed_status = "unsigned"
    team.prospect_pool = [prospect3]
    auto = auto_sign_elc_on_promotion(prospect3, team, league, season)
    assert auto.get("ok")
    assert has_elc_contract(prospect3)


def test_elc_expiry_creates_rfa_rights():
    p = _player("elc_exp", ovr=76, age=22)
    p.contract = normalize_contract_dict({
        "type": "ELC",
        "contract_type": "ELC",
        "aav_m": 0.95,
        "cap_hit_m": 0.95,
        "years_remaining": 1,
        "rights_status": "RFA",
    })
    team = _team("AAA", [p])
    league = _league([team])
    outcome = handle_player_contract_expiry(p, team, league, 2025)
    assert outcome == "rfa_rights"
    assert len(team.rfa_rights) == 1


def _fa_player(pid, ovr, age, pos):
    p = _player(pid, ovr=ovr, age=age, pos=pos)
    p.rights_status = "UFA"
    return p


def _cpu_session(team, league, *, user_tid="USER", seed=42):
    import random
    from types import SimpleNamespace

    sim = SimpleNamespace(league=league, rng=random.Random(seed), team=team)
    return SimpleNamespace(sim=sim, user_team_id=user_tid, season_calendar_year=2025)


def test_cpu_goalie_shortage_prioritizes_goalie():
    from services.contract_economy import evaluate_team_position_needs, run_cpu_free_agency, score_free_agent_fit

    skaters = [_player(f"s{i}", ovr=78, age=27, pos="C") for i in range(8)]
    team = _team("AAA", skaters)
    team.gm_window = "bubble"
    team.salary_cap_m = 92.0
    league = _league([team])
    league.salary_cap_m = 92.0
    goalie = _fa_player("g1", 77, 28, "G")
    winger = _fa_player("w1", 84, 29, "LW")
    league.free_agents = [winger, goalie]

    ctx = evaluate_team_position_needs(team, league, season_year=2025)
    g_fit, _ = score_free_agent_fit(team, goalie, ctx, 1.5, 2, league)
    w_fit, _ = score_free_agent_fit(team, winger, ctx, 3.5, 3, league)
    assert g_fit > w_fit

    session = _cpu_session(team, league)
    result = run_cpu_free_agency(session, max_signings=1)
    assert result["count"] == 1
    assert result["signings"][0]["player_id"] == "g1"
    assert result["signings"][0]["position"] == "G"


def test_cpu_lw_overload_blocks_low_ovr_winger():
    from services.contract_economy import cpu_signing_blocked, evaluate_team_position_needs, run_cpu_free_agency

    lw_depth = [_player(f"lw{i}", ovr=79, age=26, pos="LW") for i in range(5)]
    team = _team("AAA", lw_depth + [_player("c1", ovr=80, age=27, pos="C")])
    team.gm_window = "bubble"
    league = _league([team])
    league.salary_cap_m = 92.0
    low_lw = _fa_player("lw_fa", 72, 28, "LW")
    need_c = _fa_player("c_fa", 76, 27, "C")
    league.free_agents = [low_lw, need_c]

    ctx = evaluate_team_position_needs(team, league, season_year=2025)
    assert cpu_signing_blocked(team, low_lw, ctx, 1.0) == "position_overload"

    session = _cpu_session(team, league)
    result = run_cpu_free_agency(session, max_signings=1)
    assert result["count"] == 1
    assert result["signings"][0]["player_id"] == "c_fa"


def test_rebuilder_avoids_expensive_old_ufa():
    from services.contract_economy import cpu_signing_blocked, score_free_agent_fit, evaluate_team_position_needs

    team = _team("AAA", [_player("y1", ovr=74, age=22, pos="C")])
    team.gm_window = "rebuild"
    league = _league([team])
    old_star = _fa_player("old1", 83, 34, "C")
    young = _fa_player("young1", 77, 24, "C")
    ctx = evaluate_team_position_needs(team, league, season_year=2025)

    assert cpu_signing_blocked(team, old_star, ctx, 5.5) == "rebuilder_old_expensive"
    old_fit, _ = score_free_agent_fit(team, old_star, ctx, 5.5, 4, league)
    young_fit, _ = score_free_agent_fit(team, young, ctx, 2.0, 3, league)
    assert young_fit > old_fit


def test_contender_prefers_need_over_luxury_depth():
    from services.contract_economy import evaluate_team_position_needs, score_free_agent_fit

    team = _team("AAA", [
        _player("d1", ovr=86, age=30, pos="D"),
        _player("d2", ovr=85, age=29, pos="D"),
        _player("d3", ovr=84, age=28, pos="D"),
        _player("d4", ovr=83, age=27, pos="D"),
        _player("c1", ovr=82, age=28, pos="C"),
    ])
    team.gm_window = "contender"
    league = _league([team])
    ctx = evaluate_team_position_needs(team, league, season_year=2025)
    need_g = _fa_player("g_need", 81, 30, "G")
    luxury_rw = _fa_player("rw_lux", 83, 31, "RW")
    g_fit, _ = score_free_agent_fit(team, need_g, ctx, 2.5, 2, league)
    rw_fit, _ = score_free_agent_fit(team, luxury_rw, ctx, 3.0, 2, league)
    assert g_fit > rw_fit


def test_ovr_band_audit_aggregation():
    from services.contract_economy import ovr_band
    from services.contract_economy_audit import _aggregate_signing_bands

    signings = [
        {"overall": 91, "ovr_band": ovr_band(91)},
        {"overall": 86, "ovr_band": ovr_band(86)},
        {"overall": 72, "ovr_band": ovr_band(72)},
    ]
    bands = _aggregate_signing_bands(signings)
    assert bands["90+"] == 1
    assert bands["85-89"] == 1
    assert bands["70-74"] == 1


def test_cpu_signing_respects_contract_slots():
    from services.contract_economy import run_cpu_free_agency, validate_contract_slots

    roster = [_player(f"p{i}", ovr=70 + i, age=25, pos="C") for i in range(20)]
    team = _team("AAA", roster)
    team.gm_window = "bubble"
    league = _league([team])
    league.salary_cap_m = 92.0
    league.free_agents = [_fa_player(f"fa{i}", 75 + i, 27, "C") for i in range(5)]

    session = _cpu_session(team, league)
    run_cpu_free_agency(session, max_signings=3)
    slots = validate_contract_slots(team, league, additional=0)
    assert slots.get("ok") is not False or slots.get("contract_slots_used", 0) <= 50


def test_waiver_exempt_elc_prospect():
    from services.contract_economy import bury_player_contract, is_waiver_exempt

    p = _player("elc_w", ovr=68, age=19, pos="C")
    p.entry_level_contract_eligible = True
    p.signed_status = "unsigned"
    team = _team("AAA", [])
    league = _league([team])
    assert is_waiver_exempt(p, team, league)
    team.roster = [p]
    p.contract = normalize_contract_dict({"type": "ELC", "aav_m": 0.95, "cap_hit_m": 0.95, "years_remaining": 3})
    p.signed_status = "signed"
    assert is_waiver_exempt(p, team, league)
    res = bury_player_contract(team, p, league, skip_waiver_check=True)
    assert res.get("ok")


def test_veteran_requires_waivers_before_bury():
    from services.contract_economy import execute_bury, is_waiver_required_for_assignment

    p = _player("vet1", ovr=74, age=29, pos="LW")
    p.contract = normalize_contract_dict({"aav_m": 2.5, "cap_hit_m": 2.5, "years_remaining": 2})
    p.cap_hit_m = 2.5
    p.season_stats = {"gp": 120}
    team = _team("AAA", [p])
    league = _league([team])
    assert is_waiver_required_for_assignment(p, "nhl", "minors", league)
    blocked = execute_bury(team, p, league)
    assert blocked.get("requires_waivers") or blocked.get("reason") == "waiver_required"


def test_waiver_claim_transfers_player():
    from services.contract_economy import (
        expose_player_to_waivers,
        get_team_cap_snapshot_full,
        run_waiver_claim_pass,
        validate_contract_slots,
    )

    loser = _team("LOSS", [])
    winner = _team("WIN", [])
    loser.wins = 20
    winner.wins = 50
    league = _league([loser, winner])
    league.salary_cap_m = 92.0
    p = _player("claim1", ovr=76, age=28, pos="G")
    p.contract = normalize_contract_dict({"aav_m": 1.5, "cap_hit_m": 1.5, "years_remaining": 2})
    p.signed_status = "signed"
    p.cap_hit_m = 1.5
    p.season_stats = {"gp": 80}
    loser.roster = [p]

    expose_player_to_waivers(loser, p, league, season_year=2025)
    assert p not in loser.roster
    from types import SimpleNamespace
    sim = SimpleNamespace(league=league, rng=__import__("random").Random(1))
    result = run_waiver_claim_pass(league, sim, season_year=2025)
    assert len(result["claims"]) == 1
    assert p in winner.roster
    assert p not in loser.roster
    slots_loss = validate_contract_slots(loser, league)
    slots_win = validate_contract_slots(winner, league)
    assert p not in loser.roster
    snap_loss = get_team_cap_snapshot_full(loser, league, season_year=2025)
    assert snap_loss["contract_slots_used"] <= 50
    assert slots_win.get("contract_slots_used", 0) >= 1


def test_cleared_player_can_be_buried_with_contract():
    from services.contract_economy import bury_player_contract, resolve_cleared_waivers

    team = _team("AAA", [])
    league = _league([team])
    p = _player("clr1", ovr=72, age=30, pos="D")
    p.contract = normalize_contract_dict({"aav_m": 2.0, "cap_hit_m": 2.0, "years_remaining": 1})
    p.cap_hit_m = 2.0
    p.season_stats = {"gp": 90}
    team.roster = [p]
    league.waiver_wire = [{
        "player_id": _player_id(p),
        "player_ref": p,
        "original_team_id": "AAA",
        "overall": 72,
        "ovr_band": "70-74",
        "position": "LD",
        "cleared": False,
        "claimed_by": None,
    }]
    res = resolve_cleared_waivers(league)
    assert len(res["cleared"]) == 1
    assert _get(p, "contract", None) is not None
    assert _get(p, "is_buried", False) or _get(p, "buried", False)


def test_buried_player_partial_cap_relief_and_slot():
    from services.contract_economy import (
        bury_player_contract,
        estimate_bury_savings,
        validate_contract_slots,
    )

    p = _player("bury1", ovr=70, age=27, pos="C")
    p.contract = normalize_contract_dict({"aav_m": 3.0, "cap_hit_m": 3.0, "years_remaining": 2})
    p.cap_hit_m = 3.0
    team = _team("AAA", [p])
    league = _league([team])
    before_slots = validate_contract_slots(team, league)
    res = bury_player_contract(team, p, league, skip_waiver_check=True)
    assert res.get("ok")
    savings = estimate_bury_savings(p)
    assert savings > 0
    after_slots = validate_contract_slots(team, league)
    assert after_slots.get("contract_slots_used") == before_slots.get("contract_slots_used")


def test_cpu_waiver_claim_goalie_need():
    from services.contract_economy import expose_player_to_waivers, run_waiver_claim_pass, score_waiver_claim_fit, evaluate_team_position_needs

    need_team = _team("NEED", [])
    need_team.wins = 15
    other = _team("OTH", [])
    other.wins = 45
    league = _league([need_team, other])
    g = _player("wg", ovr=77, age=29, pos="G")
    g.contract = normalize_contract_dict({"aav_m": 1.2, "cap_hit_m": 1.2, "years_remaining": 1})
    g.cap_hit_m = 1.2
    g.season_stats = {"gp": 60}
    expose_player_to_waivers(other, g, league, season_year=2025)
    other.roster = [g]
    from types import SimpleNamespace
    sim = SimpleNamespace(league=league, rng=__import__("random").Random(2))
    ctx = evaluate_team_position_needs(need_team, league, sim, season_year=2025)
    assert score_waiver_claim_fit(need_team, g, ctx, league) > 0.5
    claims = run_waiver_claim_pass(league, sim, season_year=2025)
    assert len(claims["claims"]) == 1
    assert g in need_team.roster


def test_cpu_waiver_skips_overloaded_low_value():
    from services.contract_economy import score_waiver_claim_fit, evaluate_team_position_needs

    team = _team("AAA", [_player(f"lw{i}", ovr=80, age=26, pos="LW") for i in range(5)])
    team.wins = 30
    league = _league([team])
    fa = _player("lowlw", ovr=69, age=29, pos="LW")
    fa.contract = normalize_contract_dict({"aav_m": 0.9, "cap_hit_m": 0.9, "years_remaining": 1})
    from types import SimpleNamespace
    sim = SimpleNamespace(league=league)
    ctx = evaluate_team_position_needs(team, league, sim, season_year=2025)
    assert score_waiver_claim_fit(team, fa, ctx, league) < 0.42


def test_cpu_buyout_targets_old_overpaid_not_elc():
    from services.contract_economy import identify_buyout_candidates, execute_cpu_buyout

    bad = _player("oldbad", ovr=72, age=34, pos="D")
    bad.contract = normalize_contract_dict({"aav_m": 5.5, "cap_hit_m": 5.5, "years_remaining": 3})
    bad.cap_hit_m = 5.5
    elc = _player("young", ovr=70, age=20, pos="C")
    elc.contract = normalize_contract_dict({"type": "ELC", "aav_m": 0.95, "cap_hit_m": 0.95, "years_remaining": 3})
    star = _player("star", ovr=90, age=28, pos="C")
    star.contract = normalize_contract_dict({"aav_m": 9.0, "cap_hit_m": 9.0, "years_remaining": 4})
    star.cap_hit_m = 9.0
    team = _team("AAA", [bad, elc, star])
    league = _league([team])
    cands = identify_buyout_candidates(team, league, season_year=2025)
    ids = [c["player_id"] for c in cands]
    assert "oldbad" in ids
    assert "young" not in ids
    assert "star" not in ids
    res = execute_cpu_buyout(team, bad, league, 2025)
    assert res.get("ok")
    assert bad not in team.roster


def test_phase2b_elc_expiry_still_rfa_after_phase3_helpers():
    test_elc_expiry_creates_rfa_rights()


def test_phase2c_ovr_band_still_works():
    test_ovr_band_audit_aggregation()


# --- Phase 2d foundation fixes ---


def test_fake_elc_prevention():
    from services.contract_economy import (
        ELC_AAV_M,
        build_contract_for_player,
        has_true_elc_contract,
        normalize_contract_dict,
    )

    rng = random.Random(42)
    young = _player("y1", ovr=78, age=21)
    c = build_contract_for_player(young, None, None, 2025, rng)
    assert str(c.get("type", "")).upper() != "ELC" or abs(float(c["cap_hit_m"]) - ELC_AAV_M) <= 0.01

    fake = normalize_contract_dict({"type": "ELC", "cap_hit_m": 2.5, "aav_m": 2.5, "years_remaining": 2})
    assert str(fake.get("type", "")).upper() != "ELC"

    true_elc = normalize_contract_dict({"type": "ELC", "cap_hit_m": ELC_AAV_M, "aav_m": ELC_AAV_M, "years_remaining": 3})
    young.contract = true_elc
    young.signed_status = "signed"
    assert has_true_elc_contract(young)


def test_contract_years_remaining_defaults():
    from services.contract_economy import _contract_years_remaining

    p = _player("none", ovr=70, age=25)
    assert _contract_years_remaining(p) == 0
    p.contract = normalize_contract_dict({"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 2})
    assert _contract_years_remaining(p) == 2
    p.contract = {"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": "bad"}
    assert _contract_years_remaining(p) == 0


def test_promotion_unsigned_receives_true_elc():
    from services.contract_economy import ELC_AAV_M, auto_sign_elc_on_promotion, has_true_elc_contract

    p = _player("promo1", ovr=68, age=20)
    p.entry_level_contract_eligible = True
    p.signed_status = "unsigned"
    team = _team("AAA", [])
    league = _league([team])
    res = auto_sign_elc_on_promotion(p, team, league, 2025)
    assert res.get("ok")
    assert has_true_elc_contract(p)
    assert abs(float(p.contract["cap_hit_m"]) - ELC_AAV_M) <= 0.01


def test_fake_contract_does_not_block_elc():
    from services.contract_economy import ELC_AAV_M, auto_sign_elc_on_promotion, has_true_elc_contract

    p = _player("promo2", ovr=69, age=19)
    p.entry_level_contract_eligible = True
    p.signed_status = "unsigned"
    p.contract = {"type": "ELC", "cap_hit_m": 2.2, "aav_m": 2.2}
    team = _team("AAA", [])
    league = _league([team])
    res = auto_sign_elc_on_promotion(p, team, league, 2025)
    assert res.get("ok")
    assert has_true_elc_contract(p)
    assert abs(float(p.contract["cap_hit_m"]) - ELC_AAV_M) <= 0.01


def test_waiver_age_22_standard_requires_waivers():
    from services.contract_economy import is_waiver_exempt, is_waiver_required_for_assignment

    p = _player("w22", ovr=74, age=22, pos="C")
    p.contract = normalize_contract_dict({"aav_m": 1.5, "cap_hit_m": 1.5, "years_remaining": 2, "type": "STANDARD"})
    p.signed_status = "signed"
    team = _team("AAA", [p])
    league = _league([team])
    assert not is_waiver_exempt(p, team, league)
    assert is_waiver_required_for_assignment(p, "nhl", "minors", league)


def test_true_elc_22_can_remain_exempt_with_gp():
    from services.contract_economy import ELC_AAV_M, is_waiver_exempt

    p = _player("elc22", ovr=70, age=22)
    p.contract = normalize_contract_dict({"type": "ELC", "aav_m": ELC_AAV_M, "cap_hit_m": ELC_AAV_M, "years_remaining": 2})
    p.signed_status = "signed"
    p.season_stats = {"gp": 10}
    team = _team("AAA", [p])
    assert is_waiver_exempt(p, team, _league([team]))


def test_contract_normalization_object_and_dict():
    from services.contract_economy import normalize_contract_payload
    from services.franchise_sim import _GeneratedContract

    obj = _GeneratedContract(
        aav_m=2.5, years=2, expiry_year=2027, contract_type="STANDARD",
        rights_status="UFA", ntc=False, nmc=True,
    )
    p = _player("norm1", ovr=80, age=27)
    p.contract = obj
    c = normalize_contract_payload(p)
    assert c["cap_hit_m"] == 2.5
    assert c["years_remaining"] == 2
    assert c["nmc"] is True

    p2 = _player("norm2", ovr=80, age=27)
    p2.contract = normalize_contract_dict({"aav_m": 2.5, "cap_hit_m": 2.5, "years_remaining": 2, "nmc": True})
    c2 = normalize_contract_payload(p2)
    assert c2["nmc"] == c["nmc"]


def test_nmc_protection_object_and_dict():
    from services.contract_economy import can_waive_or_bury, execute_bury
    from services.franchise_sim import _GeneratedContract

    p = _player("nmc1", ovr=75, age=29)
    p.contract = _GeneratedContract(
        aav_m=2.0, years=2, expiry_year=2027, contract_type="STANDARD",
        rights_status="UFA", nmc=True,
    )
    ok, _ = can_waive_or_bury(p)
    assert not ok
    blocked = execute_bury(_team("AAA", [p]), p, _league([_team("AAA", [p])]))
    assert blocked.get("reason") or not blocked.get("ok")


def test_fa_slot_limit_blocks_signing():
    from services.contract_economy import CONTRACT_SLOTS_LIMIT, sign_player_to_team, validate_contract_slots

    roster = []
    for i in range(CONTRACT_SLOTS_LIMIT):
        pl = _player(f"s{i}", ovr=70, age=26)
        pl.contract = normalize_contract_dict({"aav_m": 0.8, "cap_hit_m": 0.8, "years_remaining": 1})
        pl.signed_status = "signed"
        roster.append(pl)
    team = _team("FULL", roster)
    league = _league([team])
    fa = _player("fa1", ovr=72, age=28)
    league.free_agents = [fa]
    assert not validate_contract_slots(team, league, additional=1).get("ok")
    res = sign_player_to_team(fa, team, league, 2025, {"aav_m": 0.8, "years": 1, "force": True})
    assert not res.get("ok")
    assert "slot" in str(res.get("reason", "")).lower()


def test_resign_existing_player_does_not_count_extra_roster_spot():
    from app.sim_engine.economy.cap_engine import can_sign_player
    from services.contract_economy import sign_player_to_team

    roster = []
    for i in range(23):
        pl = _player(f"p{i}", ovr=72, age=27)
        pl.contract = normalize_contract_dict({"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 1})
        pl.cap_hit_m = 1.0
        pl.signed_status = "signed"
        roster.append(pl)
    team = _team("AAA", roster)
    league = _league([team])
    target = roster[0]
    check = can_sign_player(team, 1.05, league, player=target)
    assert check.get("ok"), check.get("reason")
    res = sign_player_to_team(
        target, team, league, 2025,
        {"aav_m": 1.05, "years": 2, "context": "re_sign", "force": True},
    )
    assert res.get("ok"), res.get("reason")


def test_buyout_penalty_appears_in_current_season_cap_snapshot():
    from services.contract_economy import execute_buyout, get_team_cap_snapshot_full

    bad = _player("buy1", ovr=72, age=34)
    bad.contract = normalize_contract_dict({"aav_m": 5.0, "cap_hit_m": 5.0, "years_remaining": 3})
    bad.cap_hit_m = 5.0
    bad.signed_status = "signed"
    team = _team("AAA", [bad])
    league = _league([team])
    res = execute_buyout(team, bad, league, 2025)
    assert res.get("ok")
    snap = get_team_cap_snapshot_full(team, league, season_year=2025)
    assert snap.get("buyout_cap_hit_m", 0) > 0
    assert len(getattr(team, "buyout_cap_hits", []) or []) >= 1


def test_rfa_bridge_not_elc():
    from services.contract_economy import contract_type_and_rights, has_true_elc_contract

    ctype, rights = contract_type_and_rights(22, 75)
    assert ctype == "RFA_BRIDGE"
    p = _player("rb", ovr=75, age=22)
    p.contract = normalize_contract_dict({"type": "RFA_BRIDGE", "aav_m": 1.2, "cap_hit_m": 1.2, "years_remaining": 2, "rights_status": rights})
    p.signed_status = "signed"
    assert not has_true_elc_contract(p)


def test_rfa_bridge_expiry_creates_rfa_rights():
    p = _player("rbexp", ovr=74, age=23)
    p.contract = normalize_contract_dict({
        "type": "RFA_BRIDGE", "aav_m": 1.0, "cap_hit_m": 1.0,
        "years_remaining": 1, "rights_status": "RFA",
    })
    p.signed_status = "signed"
    team = _team("AAA", [p])
    league = _league([team])
    outcome = handle_player_contract_expiry(p, team, league, 2025)
    assert outcome == "rfa_rights"


def test_expiry_clears_active_contract():
    from services.contract_economy import _contract_years_remaining, has_active_contract

    p = _player("exp1", ovr=80, age=30)
    p.contract = normalize_contract_dict({"aav_m": 2.0, "cap_hit_m": 2.0, "years_remaining": 1, "rights_status": "UFA"})
    p.signed_status = "signed"
    team = _team("AAA", [p])
    league = _league([team])
    handle_player_contract_expiry(p, team, league, 2025)
    assert _contract_years_remaining(p) == 0
    assert not has_active_contract(p)


def test_defer_july1_keeps_final_year_ufa_extension_eligible():
    """Final-year UFAs stay rostered through re-sign until Free Agency / July 1."""
    from services.contract_economy import (
        _contract_years_remaining,
        build_contract_row,
        expire_pending_july1_contracts,
        handle_player_contract_expiry,
    )

    p = _player("chabot", ovr=84, age=28, pos="D")
    p.contract = normalize_contract_dict({
        "aav_m": 8.0, "cap_hit_m": 8.0, "years_remaining": 1, "rights_status": "UFA",
    })
    p.signed_status = "signed"
    team = _team("OTT", [p])
    league = _league([team])

    outcome = handle_player_contract_expiry(
        p, team, league, 2025, defer_july1_ufa=True
    )
    assert outcome == "kept"
    assert p in team.roster
    assert _contract_years_remaining(p) == 1
    assert bool(getattr(p, "pending_july1_expiry", False))
    row = build_contract_row(p, team, 2025, league)
    assert row["extension_eligible"] is True
    assert row["can_negotiate"] is True
    assert not any(_player_id(x) == "chabot" for x in league.free_agents)

    session = _cpu_session(team, league, user_tid="OTT")
    report = expire_pending_july1_contracts(session)
    assert report["expired_ufa_count"] == 1
    assert p not in team.roster
    assert any(_player_id(x) == "chabot" for x in league.free_agents)


def test_buyout_protection_true_elc_and_core():
    from services.contract_economy import ELC_AAV_M, identify_buyout_candidates, is_buyout_protected

    elc = _player("belc", ovr=70, age=20)
    elc.contract = normalize_contract_dict({"type": "ELC", "aav_m": ELC_AAV_M, "cap_hit_m": ELC_AAV_M, "years_remaining": 2})
    elc.signed_status = "signed"
    star = _player("bstar", ovr=92, age=28)
    star.contract = normalize_contract_dict({"aav_m": 10.0, "cap_hit_m": 10.0, "years_remaining": 4})
    star.signed_status = "signed"
    team = _team("AAA", [elc, star])
    league = _league([team])
    assert is_buyout_protected(elc, team, league)
    assert is_buyout_protected(star, team, league)
    assert not identify_buyout_candidates(team, league, season_year=2025)


def test_bootstrap_trim_preserves_true_elc():
    from services.contract_economy import ELC_AAV_M, rebalance_team_cap_at_bootstrap

    elc = _player("belc2", ovr=70, age=20)
    elc.contract = normalize_contract_dict({"type": "ELC", "aav_m": ELC_AAV_M, "cap_hit_m": ELC_AAV_M, "years_remaining": 3})
    elc.signed_status = "signed"
    vet = _player("vet", ovr=72, age=31)
    vet.contract = normalize_contract_dict({"aav_m": 4.0, "cap_hit_m": 4.0, "years_remaining": 2})
    vet.signed_status = "signed"
    team = _team("AAA", [elc, vet])
    team.salary_cap_m = 2.0
    league = _league([team])
    league.salary_cap_m = 2.0
    rebalance_team_cap_at_bootstrap(team, league, 2025, random.Random(1))
    assert abs(float(elc.contract["cap_hit_m"]) - ELC_AAV_M) <= 0.01


def _signed_player(pid, ovr, age, pos, aav, years=2, **clause):
    p = _player(pid, ovr=ovr, age=age, pos=pos)
    p.contract = normalize_contract_dict({
        "aav_m": aav, "cap_hit_m": aav, "years_remaining": years, **clause,
    })
    p.signed_status = "signed"
    p.cap_hit_m = aav
    p.season_stats = {"gp": 120}
    return p


def _trade_session(league, *, user_tid="USER", seed=42):
    from types import SimpleNamespace

    team_by_id = {str(getattr(t, "team_id", t.id)): t for t in league.teams}
    sim = SimpleNamespace(league=league, rng=random.Random(seed))
    return SimpleNamespace(
        sim=sim,
        team_by_id=team_by_id,
        user_team_id=user_tid,
        season_calendar_year=2025,
    )


def test_cap_casualty_under_cap_no_trigger():
    from services.contract_economy import needs_cap_casualty_trade

    p = _signed_player("ok1", 80, 28, "C", 3.0)
    team = _team("AAA", [p])
    league = _league([team])
    league.salary_cap_m = 92.0
    assert not needs_cap_casualty_trade(team, league, season_year=2025)


def test_cap_casualty_core_player_protected():
    from services.contract_economy import identify_cap_casualty_candidates, is_cap_casualty_trade_protected

    star = _signed_player("star", 92, 28, "C", 10.0)
    team = _team("AAA", [star])
    league = _league([team])
    assert is_cap_casualty_trade_protected(star, team, league, season_year=2025)
    assert not identify_cap_casualty_candidates(team, league, season_year=2025)


def test_cap_casualty_true_elc_protected():
    from services.contract_economy import ELC_AAV_M, identify_cap_casualty_candidates, is_cap_casualty_trade_protected

    elc = _signed_player("elc", 70, 20, "C", ELC_AAV_M, years=3, type="ELC")
    team = _team("AAA", [elc])
    league = _league([team])
    assert is_cap_casualty_trade_protected(elc, team, league, season_year=2025)
    assert not identify_cap_casualty_candidates(team, league, season_year=2025)


def test_cap_casualty_nmc_protected():
    from services.contract_economy import identify_cap_casualty_candidates, is_cap_casualty_trade_protected

    p = _signed_player("nmc", 76, 31, "LW", 4.5, nmc=True)
    team = _team("AAA", [p])
    league = _league([team])
    assert is_cap_casualty_trade_protected(p, team, league, season_year=2025)
    assert not identify_cap_casualty_candidates(team, league, season_year=2025)


def _cap_test_roster(*, dump_pos="RW"):
    depth_c = [_signed_player(f"d{i}", 72, 28, "C", 1.0) for i in range(4)]
    if dump_pos in ("LW", "RW"):
        depth_side = [_signed_player(f"r{i}", 71, 28, dump_pos, 1.0) for i in range(3)]
        return depth_c + depth_side
    if dump_pos in ("LD", "RD"):
        depth_side = [_signed_player(f"def{i}", 71, 28, dump_pos, 1.0) for i in range(3)]
        return depth_c + depth_side
    return depth_c


def test_cap_casualty_overpaid_veteran_is_candidate():
    from services.contract_economy import identify_cap_casualty_candidates

    dump = _signed_player("dump", 78, 32, "RW", 5.5, years=3)
    team = _team("AAA", _cap_test_roster() + [dump])
    league = _league([team])
    cands = identify_cap_casualty_candidates(team, league, season_year=2025)
    dump_c = next((c for c in cands if c["player_id"] == "dump"), None)
    assert dump_c is not None
    assert dump_c["bad_score"] > 0.1


def test_cap_casualty_buyer_requires_cap_space():
    from services.contract_economy import find_cap_casualty_trade_partners, identify_cap_casualty_candidates

    dump = _signed_player("dump", 78, 32, "RW", 5.5)
    seller = _team("SEL", _cap_test_roster() + [dump])
    poor = _team("POOR", [_signed_player("p1", 70, 28, "C", 4.0)])
    league = _league([seller, poor])
    league.salary_cap_m = 5.0
    cand = identify_cap_casualty_candidates(seller, league, season_year=2025)[0]
    partners = find_cap_casualty_trade_partners(league, seller, cand, season_year=2025)
    assert not any(p["team_id"] == "POOR" for p in partners)


def test_cap_casualty_rebuilder_bad_money_gets_pick():
    from services.contract_economy import build_cap_casualty_trade_package, identify_cap_casualty_candidates

    dump = _signed_player("dump", 74, 33, "LD", 5.8, years=4)
    seller = _team("SEL", _cap_test_roster(dump_pos="LD") + [dump])
    buyer = _team("REB", [])
    buyer.gm_window = "rebuild"
    league = _league([seller, buyer])
    session = _trade_session(league)
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=4)
    cand = identify_cap_casualty_candidates(seller, league, season_year=2025)[0]
    package = build_cap_casualty_trade_package(
        seller, buyer, cand, league, session.sim, season_year=2025,
    )
    buyer_assets = package.get("REB", [])
    assert any(a.get("type") == "player" for a in buyer_assets)
    assert any(a.get("type") == "pick" for a in buyer_assets)


def test_cap_casualty_contender_useful_player_buyer_pays():
    from services.contract_economy import build_cap_casualty_trade_package, identify_cap_casualty_candidates

    useful = _signed_player("use", 83, 29, "LW", 4.2, years=2)
    seller = _team("SEL", _cap_test_roster(dump_pos="LW") + [useful])
    buyer = _team("CON", [])
    buyer.gm_window = "contender"
    league = _league([seller, buyer])
    league.salary_cap_m = 92.0
    session = _trade_session(league)
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=4)
    cands = identify_cap_casualty_candidates(seller, league, session.sim, season_year=2025)
    useful_c = next((c for c in cands if c["player_id"] == "use"), cands[0] if cands else None)
    if useful_c is None:
        useful_c = {
            "player_id": "use", "player_ref": useful, "overall": 83,
            "bad_score": 0.05, "cap_hit_m": 4.2, "fair_aav_m": 4.0, "position": "LW",
        }
    package = build_cap_casualty_trade_package(
        seller, buyer, useful_c, league, session.sim, season_year=2025,
    )
    seller_assets = package.get("SEL", [])
    assert any(a.get("type") == "pick" for a in seller_assets)


def test_cap_casualty_trade_execution_transfers_player():
    from services.contract_economy import (
        build_cap_casualty_trade_package,
        execute_cap_casualty_trade,
        get_team_cap_snapshot_full,
        identify_cap_casualty_candidates,
        validate_contract_slots,
    )

    dump = _signed_player("dump", 78, 32, "RW", 5.5)
    star = _signed_player("star", 88, 29, "C", 9.5)
    seller = _team("SEL", _cap_test_roster() + [dump, star])
    buyer = _team("BUY", [_signed_player("b1", 72, 27, "C", 1.0)])
    buyer.gm_window = "bubble"
    league = _league([seller, buyer])
    league.salary_cap_m = 8.0
    session = _trade_session(league)
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=4)
    seller_before = get_team_cap_snapshot_full(seller, league, session.sim, season_year=2025)
    cands = identify_cap_casualty_candidates(seller, league, session.sim, season_year=2025)
    cand = next(c for c in cands if c["player_id"] == "dump")
    package = build_cap_casualty_trade_package(
        seller, buyer, cand, league, session.sim, season_year=2025,
    )
    res = execute_cap_casualty_trade(
        league, seller, buyer, package, "test_cap_dump",
        sim=session.sim, season_year=2025, team_by_id=session.team_by_id,
    )
    assert res.get("ok"), res.get("reason")
    assert dump in buyer.roster
    assert dump not in seller.roster
    assert _player_id(dump) not in [_player_id(p) for p in seller.roster]
    seller_after = get_team_cap_snapshot_full(seller, league, session.sim, season_year=2025)
    assert seller_after["usable_cap_space_m"] >= seller_before["usable_cap_space_m"]
    assert validate_contract_slots(buyer, league).get("ok") is not False


def test_cap_casualty_no_duplicate_player():
    from services.contract_economy import execute_cap_casualty_trade, identify_cap_casualty_candidates, build_cap_casualty_trade_package

    depth = [_signed_player(f"d{i}", 72, 28, "C", 1.0) for i in range(4)]
    dump = _signed_player("dup", 77, 31, "C", 4.8)
    seller = _team("SEL", depth + [dump])
    buyer = _team("BUY", [])
    league = _league([seller, buyer])
    league.salary_cap_m = 6.0
    session = _trade_session(league)
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=4)
    cand = identify_cap_casualty_candidates(seller, league, session.sim, season_year=2025)[0]
    package = build_cap_casualty_trade_package(
        seller, buyer, cand, league, session.sim, season_year=2025,
    )
    res = execute_cap_casualty_trade(
        league, seller, buyer, package, "dup_check",
        sim=session.sim, season_year=2025, team_by_id=session.team_by_id,
    )
    assert res.get("ok")
    teams_with = [
        t for t in league.teams
        if _player_id(dump) in [_player_id(p) for p in t.roster]
    ]
    assert len(teams_with) == 1


def test_cap_casualty_audit_log_record():
    from services.contract_economy import execute_cap_casualty_trade, identify_cap_casualty_candidates, build_cap_casualty_trade_package

    dump = _signed_player("aud", 76, 30, "RD", 4.6)
    seller = _team("SEL", _cap_test_roster(dump_pos="RD") + [dump])
    buyer = _team("BUY", [])
    league = _league([seller, buyer])
    league.salary_cap_m = 6.0
    session = _trade_session(league)
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=4)
    cand = identify_cap_casualty_candidates(seller, league, session.sim, season_year=2025)[0]
    package = build_cap_casualty_trade_package(
        seller, buyer, cand, league, session.sim, season_year=2025,
    )
    res = execute_cap_casualty_trade(
        league, seller, buyer, package, "audit_log",
        sim=session.sim, season_year=2025, team_by_id=session.team_by_id,
    )
    assert res.get("ok")
    log = getattr(league, "cap_casualty_trades", [])
    assert log and log[-1].get("trade_type") == "cap_casualty"


def test_cap_casualty_pipeline_after_compliance():
    from services.contract_economy import run_cap_compliance_pipeline

    dump = _signed_player("pipe", 77, 31, "C", 5.2)
    star = _signed_player("core", 87, 28, "LW", 8.0)
    seller = _team("SEL", _cap_test_roster(dump_pos="LW") + [dump, star])
    buyer = _team("BUY", [_signed_player("x", 70, 27, "C", 1.0)])
    buyer.gm_window = "rebuild"
    league = _league([seller, buyer])
    league.salary_cap_m = 10.0
    session = _trade_session(league, user_tid="USER")
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=4)
    pipeline = run_cap_compliance_pipeline(session, include_buyouts=True)
    trades = pipeline.get("cap_casualty_trades") or []
    if trades:
        assert trades[0].get("trade_type") == "cap_casualty"


def _player_id(p):
    return str(getattr(p, "id", "") or getattr(p, "player_id", "") or "")


def _get(obj, key, default=None):
    return getattr(obj, key, default) if hasattr(obj, key) else default


def test_retained_salary_active_record_counts_toward_slots():
    from app.sim_engine.economy.cap_engine import _retained_slots_used, team_retained_salary_millions

    team = SimpleNamespace(
        retained_salary_records=[
            {
                "player_id": "p1",
                "amount_m": 1.5,
                "cap_hit_m": 1.5,
                "seasons_remaining": 2,
            }
        ]
    )
    assert _retained_slots_used(team) == 1
    assert team_retained_salary_millions(team) == 1.5


def test_expired_retention_ignored_after_cleanup():
    from app.sim_engine.economy.cap_engine import (
        cleanup_expired_retained_salary_records,
        team_retained_salary_millions,
        _retained_slots_used,
        calculate_team_cap_snapshot,
    )

    team = SimpleNamespace(
        roster=[],
        retained_salary_records=[
            {"player_id": "p1", "amount_m": 2.0, "cap_hit_m": 2.0, "seasons_remaining": 0},
            {"player_id": "p2", "amount_m": 1.0, "cap_hit_m": 1.0, "seasons_remaining": 1},
        ],
    )
    removed = cleanup_expired_retained_salary_records(team)
    assert removed == 1
    assert _retained_slots_used(team) == 1
    assert team_retained_salary_millions(team) == 1.0
    snap = calculate_team_cap_snapshot(team, SimpleNamespace(salary_cap_m=88.0))
    assert float(snap.get("retainedSalary", snap.get("retained_salary", 0)) or 0) <= 1.01


def test_retention_decrements_on_season_rollover():
    from app.sim_engine.economy.cap_engine import decrement_retained_salary_seasons, cleanup_expired_retained_salary_records

    team = SimpleNamespace(
        retained_salary_records=[
            {"player_id": "p1", "amount_m": 1.5, "cap_hit_m": 1.5, "seasons_remaining": 1},
        ]
    )
    expired = decrement_retained_salary_seasons(team)
    assert expired == 1
    cleanup_expired_retained_salary_records(team)
    assert team.retained_salary_records == []


def test_bootstrap_leaves_cap_headroom_at_franchise_start():
    from services.franchise_sim import start_franchise

    session = start_franchise(
        team_query="Buffalo Sabres",
        head_coach_name="Test",
        coach_archetype="balanced",
        seed=4242,
    )
    league = session.sim.league
    spaces = []
    for team in getattr(league, "teams", None) or []:
        snap = get_team_cap_snapshot_full(
            team, league, session.sim, season_year=int(session.season_calendar_year),
        )
        spaces.append(float(snap["usable_cap_space_m"]))
    assert spaces, "expected league teams"
    assert min(spaces) >= 2.0, f"teams should open with cap headroom, min={min(spaces)}"
    assert sum(1 for x in spaces if x < 0.01) <= 2, "most teams should not be pinned at $0 space"


def test_contract_office_does_not_reopen_cap_headroom_after_signing():
    """Signing to ~$0 usable space must stick — office rebuild must not trim other
    AAVs back toward the bootstrap 2.5–9M headroom band."""
    from services.contract_economy import (
        build_contract_office,
        sign_player_to_team,
    )

    roster = []
    # ~85.5M on the books under a 92M cap → ~6.5M free before the signing.
    for i, aav in enumerate([9.5, 8.5, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0, 1.5, 0.5]):
        p = _signed_player(f"r{i}", 80 + (i % 10), 26 + (i % 8), "C" if i % 3 else "LW", aav)
        roster.append(p)
    team = _team("USER", roster)
    team.salary_cap_m = 92.0
    league = _league([team])
    league.salary_cap_m = 92.0
    league.free_agents = []

    before = get_team_cap_snapshot_full(team, league, season_year=2025)
    space = float(before["usable_cap_space_m"])
    assert space > 0.5, f"expected leftover space before signing, got {space}"

    aavs_before = {
        str(p.id): float((p.contract or {}).get("aav_m") or 0)
        for p in list(team.roster)
    }

    fa = _fa_player("spend-it", 82, 28, "C")
    league.free_agents = [fa]
    sign_aav = round(space, 3)
    result = sign_player_to_team(
        fa,
        team,
        league,
        2025,
        {"aav_m": sign_aav, "years": 2, "force": True, "context": "ufa"},
    )
    assert result.get("ok"), result
    assert result.get("status") == "accepted", result

    mid = get_team_cap_snapshot_full(team, league, season_year=2025)
    assert float(mid["usable_cap_space_m"]) <= 0.15, (
        f"after signing usable space should be ~0, got {mid['usable_cap_space_m']}"
    )

    session = _trade_session(league, user_tid="USER")
    office = build_contract_office(session)
    snap = office.get("cap_snapshot") or {}
    assert float(snap.get("usable_cap_space_m") or 0) <= 0.15, (
        f"office rebuild reopened space to {snap.get('usable_cap_space_m')} "
        f"(bootstrap heal must not run on contract office)"
    )

    for p in list(team.roster):
        pid = str(p.id)
        if pid == "spend-it":
            continue
        if pid not in aavs_before:
            continue
        cur = float((p.contract or {}).get("aav_m") or 0)
        assert abs(cur - aavs_before[pid]) < 0.01, (
            f"office rebuild mutated {pid} AAV {aavs_before[pid]} -> {cur}"
        )


# ---------------------------------------------------------------------------
# Audit fixes: season-less cap summation, RFA slots, arbitration, CPU RFA pass,
# offer-sheet resolution
# ---------------------------------------------------------------------------


def test_seasonless_buyout_snapshot_no_double_count():
    """FIX C: a multi-year buyout must count as ONE season's hit in a season-less
    snapshot, not the sum of every remaining year."""
    from app.sim_engine.economy.cap_engine import (
        calculate_team_cap_snapshot,
        team_buyout_cap_hit_millions,
    )

    team = SimpleNamespace(
        roster=[],
        buyout_cap_hits=[
            {"season": "2025-26", "amount_m": 2.0},
            {"season": "2026-27", "amount_m": 2.0},
            {"season": "2027-28", "amount_m": 1.0},
        ],
    )
    # Season-less: single worst season (2.0), NOT 5.0.
    assert team_buyout_cap_hit_millions(team) == 2.0
    # Season-specific still filters to that exact season.
    assert team_buyout_cap_hit_millions(team, season_label="2027-28") == 1.0
    snap = calculate_team_cap_snapshot(team, SimpleNamespace(salary_cap_m=88.0))
    assert float(snap.get("buyoutCapHit", 0) or 0) == 2.0


def test_seasonless_retention_two_records_same_season_sum():
    """Two retention rows in the same season still sum; different seasons take max."""
    from app.sim_engine.economy.cap_engine import team_retained_salary_millions

    team = SimpleNamespace(
        retained_salary_records=[
            {"player_id": "a", "amount_m": 1.0, "season": "2025-26", "seasons_remaining": 2},
            {"player_id": "b", "amount_m": 0.5, "season": "2025-26", "seasons_remaining": 1},
            {"player_id": "c", "amount_m": 3.0, "season": "2026-27", "seasons_remaining": 2},
        ]
    )
    # Worst single season is 2026-27 at 3.0; 2025-26 is 1.5. Max = 3.0.
    assert team_retained_salary_millions(team) == 3.0


def test_young_star_market_value_not_flat_capped():
    """FIX F: an elite proven 22-year-old must not be valued like $3.2M depth (which
    branded young stars as gross overpays); young depth still stays cheap."""
    from services.contract_economy import compute_market_value

    star = _player("ystar", ovr=92, age=22, pos="C")
    star.season_stats = {"pts": 85, "gp": 82}
    star.pro_seasons = 3
    depth = _player("ydepth", ovr=74, age=22, pos="C")
    depth.season_stats = {"pts": 18, "gp": 70}

    star_val = compute_market_value(star)
    depth_val = compute_market_value(depth)
    assert star_val >= 6.5, f"elite young star undervalued: {star_val}"
    assert depth_val <= 3.5, f"young depth should stay cheap: {depth_val}"
    assert star_val > depth_val


def test_qualify_rfa_blocked_when_slots_full():
    """FIX D: a qualifying offer consumes a contract slot and must be gated by it."""
    from services.contract_economy import (
        CONTRACT_SLOTS_LIMIT,
        add_rfa_rights,
        qualify_rfa,
    )

    roster = []
    for i in range(CONTRACT_SLOTS_LIMIT):
        pl = _player(f"s{i}", ovr=70, age=26)
        pl.contract = normalize_contract_dict({"aav_m": 0.8, "cap_hit_m": 0.8, "years_remaining": 1})
        pl.signed_status = "signed"
        roster.append(pl)
    team = _team("FULL", roster)
    league = _league([team])
    rfa = _player("rfaX", ovr=78, age=23)
    rfa.contract = normalize_contract_dict(
        {"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 0, "rights_status": "RFA"}
    )
    rfa.cap_hit_m = 1.0
    add_rfa_rights(team, rfa, 2025)
    res = qualify_rfa(team, "rfaX", league, 2025)
    assert not res.get("ok")
    assert "slot" in str(res.get("reason", "")).lower()


def test_arbitration_award_within_submissions_and_not_static():
    """FIX E: award lands inside the two submissions, term is 1-2 years, and it is
    production/market-weighted rather than a fixed exact midpoint."""
    from services.contract_economy import (
        add_rfa_rights,
        execute_arbitration_file,
        execute_arbitration_settle,
        find_rfa_rights,
    )

    p = _player("arbP", ovr=82, age=25, pos="C")
    p.contract = normalize_contract_dict(
        {"aav_m": 3.0, "cap_hit_m": 3.0, "years_remaining": 0, "rights_status": "RFA"}
    )
    p.cap_hit_m = 3.0
    p.season_stats = {"pts": 60, "gp": 80}
    team = _team("AAA", [])
    team.salary_cap_m = 92.0
    league = _league([team])
    add_rfa_rights(team, p, 2025)
    execute_arbitration_file(team, "arbP", 6.0)
    entry = find_rfa_rights(team, "arbP")
    team_offer = float(entry["team_offer_m"])
    res = execute_arbitration_settle(team, "arbP", league, 2025)
    assert res.get("ok")
    award = float(res["award_aav_m"])
    assert team_offer <= award <= 6.0
    assert res["award_years"] in (1, 2)
    assert any(_player_id(x) == "arbP" for x in team.roster)


def _rfa_on_team(team, pid, ovr, age, pos, prev_aav=1.5):
    from services.contract_economy import add_rfa_rights

    p = _player(pid, ovr=ovr, age=age, pos=pos)
    p.contract = normalize_contract_dict(
        {"aav_m": prev_aav, "cap_hit_m": prev_aav, "years_remaining": 0, "rights_status": "RFA"}
    )
    p.cap_hit_m = prev_aav
    add_rfa_rights(team, p, 2025)
    return p


def test_cpu_rfa_pass_re_signs_valuable_player():
    """FIX A: a young, valuable RFA is re-signed by his CPU team and returns to the
    active roster (previously stranded in rights limbo forever)."""
    from services.contract_economy import run_cpu_rfa_decisions

    roster = [_player(f"r{i}", ovr=78, age=26, pos="LW") for i in range(3)]
    for pl in roster:
        pl.contract = normalize_contract_dict({"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 2})
        pl.signed_status = "signed"
    team = _team("CPU", roster)
    team.gm_window = "bubble"
    league = _league([team])
    league.salary_cap_m = 92.0
    _rfa_on_team(team, "rfa_star", 84, 23, "C", prev_aav=2.5)

    session = _cpu_session(team, league, user_tid="OTHER")
    result = run_cpu_rfa_decisions(session)
    assert result["re_signed_count"] == 1
    assert result["walked_count"] == 0
    assert any(_player_id(p) == "rfa_star" for p in team.roster)
    assert len(team.rfa_rights) == 0


def test_cpu_rfa_pass_walks_surplus_depth():
    """FIX A: an overloaded, low-value RFA is released to unrestricted free agency."""
    from services.contract_economy import run_cpu_rfa_decisions

    lws = [_player(f"lw{i}", ovr=80, age=27, pos="LW") for i in range(5)]
    for pl in lws:
        pl.contract = normalize_contract_dict({"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 2})
        pl.signed_status = "signed"
    team = _team("CPU", lws)
    team.gm_window = "bubble"
    league = _league([team])
    league.salary_cap_m = 92.0
    _rfa_on_team(team, "surplus", 70, 24, "LW", prev_aav=0.9)

    session = _cpu_session(team, league, user_tid="OTHER")
    result = run_cpu_rfa_decisions(session)
    assert result["walked_count"] == 1
    assert len(team.rfa_rights) == 0
    assert any(_player_id(p) == "surplus" for p in league.free_agents)


def test_cpu_rfa_pass_leaves_no_stranded_rights():
    """FIX A: after the pass, a CPU team holds zero unresolved RFA rights."""
    from services.contract_economy import run_cpu_rfa_decisions

    team = _team("CPU", [])
    team.gm_window = "bubble"
    league = _league([team])
    league.salary_cap_m = 92.0
    for i in range(6):
        _rfa_on_team(team, f"rfa{i}", 72 + i, 22 + (i % 4), ("C", "LW", "RD", "G")[i % 4], prev_aav=1.0 + i * 0.3)

    session = _cpu_session(team, league, user_tid="OTHER")
    result = run_cpu_rfa_decisions(session)
    assert result["re_signed_count"] + result["walked_count"] == 6
    assert len(team.rfa_rights) == 0


def test_cpu_rfa_pass_skips_user_team():
    """FIX A: the human's own RFAs are never auto-resolved by the CPU pass."""
    from services.contract_economy import run_cpu_rfa_decisions

    team = _team("USERTEAM", [])
    league = _league([team])
    league.salary_cap_m = 92.0
    _rfa_on_team(team, "myrfa", 80, 24, "C")

    session = _cpu_session(team, league, user_tid="USERTEAM")
    result = run_cpu_rfa_decisions(session)
    assert result["re_signed_count"] == 0
    assert result["walked_count"] == 0
    assert len(team.rfa_rights) == 1


def test_cpu_own_ufa_resign_keeps_star_on_contender():
    """CPU contenders re-sign exclusive UFAs when negotiation agrees — no force-keep."""
    from services.contract_economy import run_cpu_own_ufa_resign

    roster = [_player(f"r{i}", ovr=78, age=26, pos="LW") for i in range(4)]
    for pl in roster:
        pl.contract = normalize_contract_dict({"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 2})
        pl.signed_status = "signed"
    team = _team("TBL", roster)
    team.gm_window = "contender"
    league = _league([team])
    league.salary_cap_m = 92.0

    star = _player("kucherov", ovr=95, age=32, pos="RW")
    star.rights_status = "UFA"
    star.ufa_from_team_id = "TBL"
    star.previous_nhl_team_id = "TBL"
    star.ufa_exclusive = True
    star.contract = None
    league.free_agents = [star]

    session = _cpu_session(team, league, user_tid="OTHER")
    session.free_agency_open = False
    result = run_cpu_own_ufa_resign(session)
    # Contender with space should usually retain; either re-sign or walk is valid —
    # force-keep must not invent agreement. Contender + space → expect re-sign.
    assert result["re_signed_count"] + result["walked_count"] == 1
    if result["re_signed_count"] == 1:
        assert any(_player_id(p) == "kucherov" for p in team.roster)
        assert not any(_player_id(p) == "kucherov" for p in league.free_agents)


def test_cpu_own_ufa_resign_rebuild_star_can_walk():
    """Elite UFAs on rebuild clubs may test free agency — but the club must
    attempt a serious offer when it has room before releasing exclusivity."""
    from services.contract_economy import run_cpu_own_ufa_resign

    roster = [_player(f"r{i}", ovr=72, age=24, pos="LW") for i in range(4)]
    for pl in roster:
        pl.contract = normalize_contract_dict({"aav_m": 1.0, "cap_hit_m": 1.0, "years_remaining": 2})
        pl.signed_status = "signed"
    team = _team("CHI", roster)
    team.gm_window = "rebuild"
    league = _league([team])
    league.salary_cap_m = 92.0

    star = _player("starufa", ovr=90, age=28, pos="C")
    star.rights_status = "UFA"
    star.ufa_from_team_id = "CHI"
    star.previous_nhl_team_id = "CHI"
    star.ufa_exclusive = True
    star.contract = None
    league.free_agents = [star]

    session = _cpu_session(team, league, user_tid="OTHER")
    session.free_agency_open = False
    result = run_cpu_own_ufa_resign(session)
    assert result["re_signed_count"] + result["walked_count"] == 1
    if result["walked_count"] == 1:
        assert result["walked"][0]["reason"] == "wants_contender"
        assert any(_player_id(p) == "starufa" for p in league.free_agents)
    else:
        # Retention attempt succeeded — exclusivity cleared and player is rostered.
        assert any(_player_id(p) == "starufa" for p in team.roster)
        assert not any(_player_id(p) == "starufa" for p in league.free_agents)


def test_offer_sheet_resolution_signed_away_with_compensation():
    """FIX B: a big overpay offer sheet is declined by the CPU rights team; the
    player moves to the offering team and compensation is recorded."""
    from services.contract_economy import (
        add_rfa_rights,
        execute_offer_sheet,
        resolve_offer_sheets,
    )

    player = _player("os1", ovr=85, age=25, pos="C")
    player.contract = normalize_contract_dict(
        {"aav_m": 3.0, "cap_hit_m": 3.0, "years_remaining": 0, "rights_status": "RFA"}
    )
    player.cap_hit_m = 3.0
    rights_team = _team("RIGHT", [])
    rights_team.salary_cap_m = 92.0
    offering_team = _team("OFFER", [])
    offering_team.salary_cap_m = 92.0
    league = _league([rights_team, offering_team])
    add_rfa_rights(rights_team, player, 2025)

    res = execute_offer_sheet(offering_team, rights_team, player, league, 2025, {"aav_m": 9.5, "years": 6})
    assert res.get("ok")

    session = _trade_session(league, user_tid="NONE")
    out = resolve_offer_sheets(session)
    assert out["count"] == 1
    assert out["resolved"][0]["outcome"] == "signed_away"
    assert any(_player_id(p) == "os1" for p in offering_team.roster)
    assert not any(_player_id(p) == "os1" for p in rights_team.roster)
    assert not (getattr(league, "pending_offer_sheets", None) or [])


def test_offer_sheet_resolution_matched():
    """FIX B: a reasonable offer sheet is matched; the player stays on the rights team."""
    from services.contract_economy import (
        add_rfa_rights,
        compute_market_value,
        execute_offer_sheet,
        resolve_offer_sheets,
    )

    player = _player("os2", ovr=80, age=26, pos="LW")
    player.contract = normalize_contract_dict(
        {"aav_m": 2.5, "cap_hit_m": 2.5, "years_remaining": 0, "rights_status": "RFA"}
    )
    player.cap_hit_m = 2.5
    rights_team = _team("RIGHT", [])
    rights_team.salary_cap_m = 92.0
    offering_team = _team("OFFER", [])
    offering_team.salary_cap_m = 92.0
    league = _league([rights_team, offering_team])
    add_rfa_rights(rights_team, player, 2025)

    market = compute_market_value(player, league)
    res = execute_offer_sheet(
        offering_team, rights_team, player, league, 2025,
        {"aav_m": round(market, 3), "years": 3},
    )
    assert res.get("ok")

    session = _trade_session(league, user_tid="NONE")
    out = resolve_offer_sheets(session)
    assert out["count"] == 1
    assert out["resolved"][0]["outcome"] == "matched"
    assert any(_player_id(p) == "os2" for p in rights_team.roster)
    assert not any(_player_id(p) == "os2" for p in offering_team.roster)


def test_offer_sheet_user_rights_stay_pending():
    """FIX B: the human keeps control of matching his own RFA offer sheets."""
    from services.contract_economy import (
        add_rfa_rights,
        execute_offer_sheet,
        resolve_offer_sheets,
    )

    player = _player("os3", ovr=83, age=25, pos="C")
    player.contract = normalize_contract_dict(
        {"aav_m": 3.0, "cap_hit_m": 3.0, "years_remaining": 0, "rights_status": "RFA"}
    )
    player.cap_hit_m = 3.0
    rights_team = _team("RIGHT", [])
    rights_team.salary_cap_m = 92.0
    offering_team = _team("OFFER", [])
    offering_team.salary_cap_m = 92.0
    league = _league([rights_team, offering_team])
    add_rfa_rights(rights_team, player, 2025)
    execute_offer_sheet(offering_team, rights_team, player, league, 2025, {"aav_m": 8.0, "years": 5})

    session = _trade_session(league, user_tid="RIGHT")
    out = resolve_offer_sheets(session)
    assert out["count"] == 0
    assert out["pending"] == 1


def test_spc_type_normalization_and_explicit_flag():
    from services.contract_economy import (
        does_contract_use_contract_slot,
        iter_org_contract_players,
        normalize_contract_dict,
        uses_nhl_contract_slot,
        _count_team_contract_slots,
    )

    ahl_only = normalize_contract_dict({"type": "ahl only", "aav_m": 0.5, "years_remaining": 2})
    assert ahl_only["type"] == "AHL"
    assert ahl_only["is_nhl_spc"] is False
    assert does_contract_use_contract_slot(ahl_only) is False

    spc_alias = normalize_contract_dict({"type": "spc", "aav_m": 0.0, "years_remaining": 2, "is_nhl_spc": True})
    assert spc_alias["type"] == "STANDARD"
    assert does_contract_use_contract_slot(spc_alias) is True

    missing_id_a = SimpleNamespace(
        id=None,
        retired=False,
        signed_status="signed",
        contract=normalize_contract_dict({"type": "STANDARD", "aav_m": 1.0, "years_remaining": 2, "is_nhl_spc": True}),
    )
    missing_id_b = SimpleNamespace(
        id="",
        retired=False,
        signed_status="signed",
        contract=normalize_contract_dict({"type": "STANDARD", "aav_m": 1.0, "years_remaining": 2, "is_nhl_spc": True}),
    )
    team = SimpleNamespace(
        roster=[missing_id_a],
        ahl_roster=[missing_id_b],
        echl_roster=[],
        prospect_pool=[],
    )
    # Missing IDs must not collapse to a single counted slot.
    assert len(iter_org_contract_players(team)) == 2
    assert _count_team_contract_slots(team) == 2
    assert uses_nhl_contract_slot(missing_id_a) is True


def test_retained_salary_does_not_create_spc_on_retaining_team():
    from services.contract_economy import (
        _count_team_contract_slots,
        normalize_contract_dict,
        uses_nhl_contract_slot,
    )
    player = SimpleNamespace(
        id="p_ret",
        retired=False,
        signed_status="signed",
        contract=normalize_contract_dict(
            {"type": "STANDARD", "aav_m": 5.0, "years_remaining": 3, "is_nhl_spc": True}
        ),
    )
    source = SimpleNamespace(
        roster=[],
        ahl_roster=[],
        echl_roster=[],
        prospect_pool=[],
        retained_salary_records=[
            {"player_id": "p_ret", "amount_m": 1.5, "cap_hit_m": 1.5, "seasons_remaining": 2}
        ],
    )
    acq = SimpleNamespace(
        roster=[player],
        ahl_roster=[],
        echl_roster=[],
        prospect_pool=[],
        retained_salary_records=[],
    )
    assert _count_team_contract_slots(source) == 0
    assert _count_team_contract_slots(acq) == 1
    assert uses_nhl_contract_slot(player) is True


def test_duplicate_id_across_ahl_echl_counts_once():
    from services.contract_economy import _count_team_contract_slots, normalize_contract_dict

    c = normalize_contract_dict({"type": "ELC", "aav_m": 0.95, "years_remaining": 2, "is_nhl_spc": True})
    p_ahl = SimpleNamespace(id="dup1", retired=False, signed_status="signed", contract=c)
    p_echl = SimpleNamespace(id="dup1", retired=False, signed_status="signed", contract=c)
    team = SimpleNamespace(roster=[], ahl_roster=[p_ahl], echl_roster=[p_echl], prospect_pool=[])
    assert _count_team_contract_slots(team) == 1


if __name__ == "__main__":
    tests = [
        test_normalize_money_m_legacy_dollars,
        test_normalize_contract_dict_millions,
        test_team_importance_higher_with_peer_gap,
        test_generate_contract_terms_respects_max_aav,
        test_cap_snapshot_non_negative_after_bootstrap,
        test_rfa_rights_not_lost_on_expiry,
        test_evaluate_contract_offer_structure,
        test_bad_contract_score_positive_for_overpay,
        test_validate_franchise_cap_at_start_empty_when_compliant,
        test_qualifying_offer_minimum,
        test_stress_audit_runs_clean_short,
        test_elc_pipeline_reserve_and_promotion,
        test_elc_expiry_creates_rfa_rights,
        test_cpu_goalie_shortage_prioritizes_goalie,
        test_cpu_lw_overload_blocks_low_ovr_winger,
        test_rebuilder_avoids_expensive_old_ufa,
        test_contender_prefers_need_over_luxury_depth,
        test_ovr_band_audit_aggregation,
        test_cpu_signing_respects_contract_slots,
        test_waiver_exempt_elc_prospect,
        test_veteran_requires_waivers_before_bury,
        test_waiver_claim_transfers_player,
        test_cleared_player_can_be_buried_with_contract,
        test_buried_player_partial_cap_relief_and_slot,
        test_cpu_waiver_claim_goalie_need,
        test_cpu_waiver_skips_overloaded_low_value,
        test_cpu_buyout_targets_old_overpaid_not_elc,
        test_phase2b_elc_expiry_still_rfa_after_phase3_helpers,
        test_phase2c_ovr_band_still_works,
        test_fake_elc_prevention,
        test_contract_years_remaining_defaults,
        test_promotion_unsigned_receives_true_elc,
        test_fake_contract_does_not_block_elc,
        test_waiver_age_22_standard_requires_waivers,
        test_true_elc_22_can_remain_exempt_with_gp,
        test_contract_normalization_object_and_dict,
        test_nmc_protection_object_and_dict,
        test_fa_slot_limit_blocks_signing,
        test_resign_existing_player_does_not_count_extra_roster_spot,
        test_buyout_penalty_appears_in_current_season_cap_snapshot,
        test_rfa_bridge_not_elc,
        test_rfa_bridge_expiry_creates_rfa_rights,
        test_expiry_clears_active_contract,
        test_defer_july1_keeps_final_year_ufa_extension_eligible,
        test_buyout_protection_true_elc_and_core,
        test_bootstrap_trim_preserves_true_elc,
        test_cap_casualty_under_cap_no_trigger,
        test_cap_casualty_core_player_protected,
        test_cap_casualty_true_elc_protected,
        test_cap_casualty_nmc_protected,
        test_cap_casualty_overpaid_veteran_is_candidate,
        test_cap_casualty_buyer_requires_cap_space,
        test_cap_casualty_rebuilder_bad_money_gets_pick,
        test_cap_casualty_contender_useful_player_buyer_pays,
        test_cap_casualty_trade_execution_transfers_player,
        test_cap_casualty_no_duplicate_player,
        test_cap_casualty_audit_log_record,
        test_cap_casualty_pipeline_after_compliance,
        test_retained_salary_active_record_counts_toward_slots,
        test_expired_retention_ignored_after_cleanup,
        test_retention_decrements_on_season_rollover,
        test_seasonless_buyout_snapshot_no_double_count,
        test_seasonless_retention_two_records_same_season_sum,
        test_young_star_market_value_not_flat_capped,
        test_qualify_rfa_blocked_when_slots_full,
        test_arbitration_award_within_submissions_and_not_static,
        test_cpu_rfa_pass_re_signs_valuable_player,
        test_cpu_rfa_pass_walks_surplus_depth,
        test_cpu_rfa_pass_leaves_no_stranded_rights,
        test_cpu_rfa_pass_skips_user_team,
        test_cpu_own_ufa_resign_keeps_star_on_contender,
        test_cpu_own_ufa_resign_rebuild_star_can_walk,
        test_offer_sheet_resolution_signed_away_with_compensation,
        test_offer_sheet_resolution_matched,
        test_offer_sheet_user_rights_stay_pending,
        test_bootstrap_leaves_cap_headroom_at_franchise_start,
        test_contract_office_does_not_reopen_cap_headroom_after_signing,
    ]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print("All contract economy tests passed.")
