"""
Basic trade foundation tests (run: python -m pytest backend/tests/test_trade_foundation.py -q)
"""

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

from app.sim_engine.trades.trade_asset import normalize_trade_package, canonical_pick_id  # noqa: E402
from app.sim_engine.trades.trade_pick_registry import (  # noqa: E402
    ensure_draft_pick_registry,
    get_pick_by_id,
    transfer_pick,
    validate_pick_ownership,
)
from app.sim_engine.trades.trade_evaluator import evaluate_trade_package  # noqa: E402
from app.sim_engine.trades.trade_rules import validate_trade_rules  # noqa: E402


def _player(pid: str, cap_hit: float = 3.0, ntc: bool = False, nmc: bool = False, mntc: int = 0, approved=None):
    contract = SimpleNamespace(
        cap_hit_m=cap_hit,
        aav_m=cap_hit,
        years_remaining=2,
        no_trade_clause=ntc,
        no_move_clause=nmc,
        modified_no_trade_teams=mntc,
        approved_trade_teams=list(approved or []),
        clauses=None,
    )
    ident = SimpleNamespace(name=f"Player {pid}", age=26, position=SimpleNamespace(value="C"))
    return SimpleNamespace(
        id=pid,
        identity=ident,
        contract=contract,
        cap_hit_m=cap_hit,
        ovr=lambda: 0.72,
        season_stats={"gp": 40, "pts": 30, "g": 12, "a": 18},
    )


def _team(tid: str, players=None):
    roster = list(players or [])
    return SimpleNamespace(
        team_id=tid,
        id=tid,
        roster=roster,
        owned_pick_ids=[],
        needs={},
        gm_window="emerging",
        window="emerging",
        cap_pressure="moderate",
        cap_pressure_tier="moderate",
        retained_salary_records=[],
    )


def _league(teams):
    lg = SimpleNamespace(teams=teams, salary_cap_m=88.0, cap_floor_m=65.0, trade_history=[])
    ensure_draft_pick_registry(lg, start_year=2025, years_ahead=4)
    return lg


def test_pick_registry_and_transfer():
    t1 = _team("AAA")
    t2 = _team("BBB")
    league = _league([t1, t2])
    pick_id = canonical_pick_id(2025, 1, "AAA")
    assert validate_pick_ownership(league, pick_id, "AAA")
    transfer_pick(league, pick_id, "BBB")
    assert validate_pick_ownership(league, pick_id, "BBB")
    assert not validate_pick_ownership(league, pick_id, "AAA")


def test_reject_fake_pick():
    t1 = _team("AAA", [_player("p1")])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    package = normalize_trade_package(
        {
            "BBB": [{"type": "pick", "id": "2099-round1-FAKE", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p1", "team": "AAA"}],
        },
        team_by_id=team_by_id,
    )
    rules = validate_trade_rules(package, league, team_by_id, context={"season_year": 2025})
    assert not rules["ok"]
    assert any("not found" in r.lower() or "does not own" in r.lower() for r in rules["blocking_reasons"])


def test_reject_ahl_prospect_not_on_nhl_roster():
    p1 = _player("p1")
    t1 = _team("AAA", [])
    t1.ahl_roster = [p1]
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    package = normalize_trade_package(
        {
            "BBB": [{"type": "player", "id": "p1", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        team_by_id=team_by_id,
    )
    rules = validate_trade_rules(package, league, team_by_id, context={"season_year": 2025})
    assert not rules["ok"]
    assert any("ahl" in r.lower() for r in rules["blocking_reasons"])


def test_reject_duplicate_player():
    t1 = _team("AAA", [_player("p1")])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    package = normalize_trade_package(
        {
            "BBB": [
                {"type": "player", "id": "p1", "team": "AAA"},
                {"type": "player", "id": "p1", "team": "AAA"},
            ],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        team_by_id=team_by_id,
    )
    rules = validate_trade_rules(package, league, team_by_id, context={"season_year": 2025})
    assert not rules["ok"]
    assert any("duplicate" in r.lower() for r in rules["blocking_reasons"])


def test_retained_over_50_rejected():
    t1 = _team("AAA", [_player("p1")])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    try:
        normalize_trade_package(
            {
                "BBB": [{"type": "player", "id": "p1", "team": "AAA", "retained": 55}],
                "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
            },
            team_by_id=team_by_id,
        )
        assert False, "expected retained pct error"
    except ValueError as e:
        assert "50" in str(e)


def test_evaluate_player_for_player():
    t1 = _team("AAA", [_player("p1", cap_hit=4.0)])
    t2 = _team("BBB", [_player("p2", cap_hit=4.0)])
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    ev = evaluate_trade_package(
        {
            "BBB": [{"type": "player", "id": "p1", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        league=league,
        team_by_id=team_by_id,
        context={"season_year": 2025, "calendar_cursor": 100, "regular_season_last_index": 192},
        user_team_id="AAA",
    )
    assert ev["can_execute"] is True
    assert "cap_impact" in ev


def test_reject_quantity_spam_for_premium_first():
    t1 = _team("AAA", [_player("a1")])
    t2 = _team("BBB", [_player("b1")])
    t2.gm_window = "rebuild"
    t2.window = "rebuild"
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    ev = evaluate_trade_package(
        {
            "AAA": [
                {"type": "pick", "id": "2025-round1-BBB", "team": "BBB"},
            ],
            "BBB": [
                {"type": "pick", "id": "2025-round5-AAA", "team": "AAA"},
                {"type": "pick", "id": "2025-round6-AAA", "team": "AAA"},
                {"type": "pick", "id": "2025-round7-AAA", "team": "AAA"},
                {"type": "pick", "id": "2026-round5-AAA", "team": "AAA"},
                {"type": "pick", "id": "2026-round6-AAA", "team": "AAA"},
            ],
        },
        league=league,
        team_by_id=team_by_id,
        context={"season_year": 2025, "calendar_cursor": 100, "regular_season_last_index": 192},
        user_team_id="AAA",
    )
    assert ev["can_execute"] is True
    assert ev["accepted"] is False
    blob = " ".join(ev.get("rejection_reasons") or []).lower()
    assert "premium" in blob or "quantity" in blob or "first" in blob


def test_reject_pick_owner_spoof_from_frontend_payload():
    t1 = _team("AAA", [_player("p1")])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    package = normalize_trade_package(
        {
            "BBB": [
                {
                    "type": "pick",
                    "id": "2025-round1-AAA",
                    "team": "AAA",
                    "current_owner_team_id": "BBB",
                }
            ],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        team_by_id=team_by_id,
    )
    rules = validate_trade_rules(package, league, team_by_id, context={"season_year": 2025})
    assert rules["ok"] is False
    assert any("frontend ownership mismatch" in r.lower() for r in rules["blocking_reasons"])


def test_elite_player_values_above_depth_prospect():
    from app.sim_engine.trades.trade_value import evaluate_player_asset_value

    def _mk(ovr_99: float, age: int, pos: str, pot: float):
        contract = SimpleNamespace(
            cap_hit_m=3.0,
            years_remaining=3,
            no_trade_clause=False,
            no_move_clause=False,
            modified_no_trade_teams=0,
            clauses=None,
        )
        ident = SimpleNamespace(name="X", age=age, position=SimpleNamespace(value=pos))
        return SimpleNamespace(
            id="x",
            identity=ident,
            contract=contract,
            ratings={"dev_potential": pot},
            ovr=lambda o=ovr_99: o / 99.0,
            season_stats={"gp": 40, "pts": 20, "g": 8, "a": 12},
        )

    team = SimpleNamespace(
        gm_window="rebuild",
        needs={"top_line_forward": 0.6, "depth_forward": 0.5, "top_4_defense": 0.7, "goalie": 0.2},
    )
    league = SimpleNamespace()
    depth = _mk(72, 20, "C", 78)
    elite = _mk(85, 22, "D", 92)
    v_depth = float(evaluate_player_asset_value(depth, team, team, league)["total"])
    v_elite = float(evaluate_player_asset_value(elite, team, team, league)["total"])
    assert v_elite - v_depth >= 18.0, f"expected large gap, got elite={v_elite} depth={v_depth}"


def test_contract_terms_never_inverts_for_veteran_elite():
    import random

    from services.franchise_sim import _generate_contract_terms, _ensure_league_roster_contracts
    from app.sim_engine.economy.cap_engine import player_cap_hit_millions

    rng = random.Random(7)
    for _ in range(40):
        aav, years = _generate_contract_terms(92.0, 36, "C", rng)
        assert years >= 1
        assert aav >= 0.775

    ident = SimpleNamespace(name="Veteran Star", age=36, position=SimpleNamespace(value="G"))
    player = SimpleNamespace(
        id="vet-g",
        identity=ident,
        contract=None,
        ovr=lambda: 0.96,
        retired=False,
    )
    team = SimpleNamespace(
        team_id="TOR",
        id="TOR",
        roster=[player],
        _contracts_bootstrapped=False,
    )
    league = SimpleNamespace(teams=[team], salary_cap_m=88.0, cap_floor_m=65.0)
    _ensure_league_roster_contracts(league, 2025)
    assert player_cap_hit_millions(player) > 0
    assert _generate_contract_terms(96.0, 28, "G", random.Random(1))[1] >= 1


def test_cap_accrual_mid_season_more_lenient():
    from app.sim_engine.economy.cap_engine import can_trade_cap_fit, calculate_team_cap_snapshot

    contract = SimpleNamespace(cap_hit_m=4.0, years_remaining=2, clauses=None)
    ident = SimpleNamespace(name="A", age=27, position=SimpleNamespace(value="C"))
    p_out = SimpleNamespace(id="out", identity=ident, contract=contract, ovr=lambda: 0.75)
    p_in = SimpleNamespace(id="in", identity=ident, contract=SimpleNamespace(cap_hit_m=5.0, years_remaining=2), ovr=lambda: 0.78)
    team = SimpleNamespace(
        team_id="T",
        roster=[p_out],
        retained_salary_records=[],
        buried_cap_hits=[],
        buyout_cap_hits=[],
    )
    league = SimpleNamespace(salary_cap_m=88.0, cap_floor_m=65.0)
    early = can_trade_cap_fit(
        team, [p_out], [p_in], league=league,
        calendar_cursor=20, regular_season_last_index=192, deadline_phase=0.2,
    )
    late = can_trade_cap_fit(
        team, [p_out], [p_in], league=league,
        calendar_cursor=180, regular_season_last_index=192, deadline_phase=0.85,
    )
    assert early.get("prorationFactor", 1.0) > late.get("prorationFactor", 1.0)


def test_cpu_trade_proposer_validates_package():
    from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry

    t1 = _team("AAA", [_player("a1", cap_hit=3.5), _player("a2", cap_hit=2.5)])
    t2 = _team("BBB", [_player("b1", cap_hit=3.5), _player("b2", cap_hit=2.0)])
    t1.gm_window = "rebuild"
    t2.gm_window = "contender"
    league = _league([t1, t2])
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=4)
    trades = propose_and_execute_cpu_trades(
        league, max_executions=1, calendar_cursor=100, regular_season_last_index=192,
    )
    assert isinstance(trades, list)


def test_acceptance_matrix_ntc_blocks():
    p_ntc = _player("p_ntc", ntc=True)
    t1 = _team("AAA", [p_ntc])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    ev = evaluate_trade_package(
        {
            "BBB": [{"type": "player", "id": "p_ntc", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        league=league,
        team_by_id={"AAA": t1, "BBB": t2},
        context={"season_year": 2025, "calendar_cursor": 100, "regular_season_last_index": 192},
        user_team_id="BBB",
    )
    assert ev["can_execute"] is False
    assert any("ntc" in r.lower() or "no-trade" in r.lower() for r in ev.get("rejection_reasons") or [])


def test_ntc_waive_allows_trade_with_value_penalty():
    from app.sim_engine.trades.trade_rules import evaluate_ntc_waiver_request, validate_trade_rules
    from app.sim_engine.trades.trade_asset import normalize_trade_package
    from app.sim_engine.trades.trade_value import evaluate_player_asset_value

    p_ntc = _player("p_ntc", ntc=True)
    p2 = _player("p2")
    t1 = _team("AAA", [p_ntc])
    t2 = _team("BBB", [p2])
    t2.gm_window = "contender"
    t2.market = SimpleNamespace(market_size="large")
    decision = evaluate_ntc_waiver_request(
        p_ntc,
        source_team=t1,
        destination_team=t2,
        context={"season_year": 2025, "calendar_cursor": 120},
    )
    assert decision["can_request"] is True
    assert "reason" in decision

    pkg = normalize_trade_package(
        {
            "BBB": [{"type": "player", "id": "p_ntc", "team": "AAA", "ntc_waived": True}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        }
    )
    rules = validate_trade_rules(
        pkg,
        _league([t1, t2]),
        {"AAA": t1, "BBB": t2},
        context={"season_year": 2025, "calendar_cursor": 120, "regular_season_last_index": 192},
        user_team_id="BBB",
    )
    assert not any("ntc" in r.lower() and "ask" in r.lower() for r in rules["blocking_reasons"])
    assert any("waived" in w.lower() for w in rules["warnings"])

    base = evaluate_player_asset_value(p_ntc, t1, t2, _league([t1, t2]), context={})
    waived = evaluate_player_asset_value(
        p_ntc, t1, t2, _league([t1, t2]), context={"ntc_waived": True, "ntc_value_penalty_pct": 0.08}
    )
    assert float(waived["total"]) < float(base["total"])


def test_phase_c_rental_premium_contender_deadline():
    from app.sim_engine.trades.trade_value import evaluate_player_asset_value

    ident = SimpleNamespace(name="Rental Star", age=34, position=SimpleNamespace(value="C"))
    contract = SimpleNamespace(
        cap_hit_m=5.5,
        years_remaining=1,
        expiry_status="UFA",
        rights_status="UFA",
        contract_type="STANDARD",
        clauses=None,
    )
    player = SimpleNamespace(
        id="rental1",
        identity=ident,
        contract=contract,
        ovr=lambda: 0.84,
        season_stats={"gp": 50, "pts": 55, "g": 22, "a": 33},
    )
    contender = SimpleNamespace(
        team_id="CONT",
        roster=[player],
        gm_window="contender",
        window="contender",
        cap_pressure="moderate",
    )
    rebuilder = SimpleNamespace(
        team_id="REB",
        roster=[],
        gm_window="rebuild",
        window="rebuild",
        cap_pressure="moderate",
    )
    league = SimpleNamespace()
    ctx = {"deadline_phase": 0.85, "season_year": 2025}
    v_cont = float(evaluate_player_asset_value(player, rebuilder, contender, league, context=ctx)["total"])
    v_reb = float(evaluate_player_asset_value(player, contender, rebuilder, league, context=ctx)["total"])
    assert v_cont > v_reb + 5.0


def test_phase_c_injured_player_discount():
    from app.sim_engine.trades.trade_value import evaluate_player_asset_value

    ident = SimpleNamespace(name="Injured F", age=28, position=SimpleNamespace(value="C"))
    contract = SimpleNamespace(
        cap_hit_m=4.0,
        years_remaining=2,
        expiry_status="UFA",
        rights_status="UFA",
        contract_type="STANDARD",
        clauses=None,
    )
    healthy = SimpleNamespace(
        id="h1",
        identity=ident,
        contract=contract,
        ovr=lambda: 0.80,
        season_stats={"gp": 40, "pts": 35},
    )
    injured = SimpleNamespace(
        id="i1",
        identity=ident,
        contract=contract,
        ovr=lambda: 0.80,
        season_stats={"gp": 40, "pts": 35},
        injury_games_remaining=18,
        injury_status="INJURED",
    )
    team = SimpleNamespace(
        team_id="T",
        roster=[healthy],
        gm_window="contender",
        window="contender",
        cap_pressure="moderate",
    )
    league = SimpleNamespace()
    ctx = {"deadline_phase": 0.2, "season_year": 2025}
    v_h = float(evaluate_player_asset_value(healthy, team, team, league, context=ctx)["total"])
    v_i = float(evaluate_player_asset_value(injured, team, team, league, context=ctx)["total"])
    assert v_i < v_h - 3.0


def test_mntc_no_approved_list_rejected():
    p = _player("mntc1", mntc=10, approved=[])
    t1 = _team("AAA", [p])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    package = normalize_trade_package(
        {
            "BBB": [{"type": "player", "id": "mntc1", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        team_by_id={"AAA": t1, "BBB": t2},
    )
    rules = validate_trade_rules(package, league, {"AAA": t1, "BBB": t2}, context={"season_year": 2025})
    assert not rules["ok"]
    assert any("modified no-trade clause requires approved destination" in r.lower() for r in rules["blocking_reasons"])


def test_mntc_destination_not_approved_rejected():
    p = _player("mntc2", mntc=10, approved=["TOR"])
    t1 = _team("AAA", [p])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    package = normalize_trade_package(
        {
            "BBB": [{"type": "player", "id": "mntc2", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        team_by_id={"AAA": t1, "BBB": t2},
    )
    rules = validate_trade_rules(package, league, {"AAA": t1, "BBB": t2}, context={"season_year": 2025})
    assert not rules["ok"]


def test_mntc_approved_destination_allowed():
    p = _player("mntc3", mntc=10, approved=["BBB"])
    t1 = _team("AAA", [p])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    package = normalize_trade_package(
        {
            "BBB": [{"type": "player", "id": "mntc3", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        team_by_id={"AAA": t1, "BBB": t2},
    )
    rules = validate_trade_rules(package, league, {"AAA": t1, "BBB": t2}, context={"season_year": 2025})
    clause_blocks = [r for r in rules["blocking_reasons"] if "modified no-trade" in r.lower()]
    assert not clause_blocks


def test_nmc_still_rejected():
    p = _player("nmc1", nmc=True)
    t1 = _team("AAA", [p])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    rules = validate_trade_rules(
        normalize_trade_package(
            {
                "BBB": [{"type": "player", "id": "nmc1", "team": "AAA"}],
                "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
            },
            team_by_id={"AAA": t1, "BBB": t2},
        ),
        league,
        {"AAA": t1, "BBB": t2},
        context={"season_year": 2025},
    )
    assert not rules["ok"]
    assert any("nmc" in r.lower() or "no-movement" in r.lower() for r in rules["blocking_reasons"])


def test_ntc_still_rejected():
    p = _player("ntc1", ntc=True)
    t1 = _team("AAA", [p])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    rules = validate_trade_rules(
        normalize_trade_package(
            {
                "BBB": [{"type": "player", "id": "ntc1", "team": "AAA"}],
                "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
            },
            team_by_id={"AAA": t1, "BBB": t2},
        ),
        league,
        {"AAA": t1, "BBB": t2},
        context={"season_year": 2025},
    )
    assert not rules["ok"]


def test_recently_acquired_player_blocked():
    from app.sim_engine.trades.trade_rules import TRADE_ACQUISITION_COOLDOWN_DAYS

    p = _player("flip")
    p.acquired_via_trade = True
    p.last_acquired_day = 100
    t1 = _team("AAA", [p])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    rules = validate_trade_rules(
        normalize_trade_package(
            {
                "BBB": [{"type": "player", "id": "flip", "team": "AAA"}],
                "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
            },
            team_by_id={"AAA": t1, "BBB": t2},
        ),
        league,
        {"AAA": t1, "BBB": t2},
        context={"season_year": 2025, "calendar_cursor": 100 + TRADE_ACQUISITION_COOLDOWN_DAYS - 1},
    )
    assert not rules["ok"]
    assert any("recently acquired" in r.lower() for r in rules["blocking_reasons"])


def test_player_tradeable_after_cooldown():
    from app.sim_engine.trades.trade_rules import TRADE_ACQUISITION_COOLDOWN_DAYS

    p = _player("flip2")
    p.acquired_via_trade = True
    p.last_acquired_day = 50
    t1 = _team("AAA", [p])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    rules = validate_trade_rules(
        normalize_trade_package(
            {
                "BBB": [{"type": "player", "id": "flip2", "team": "AAA"}],
                "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
            },
            team_by_id={"AAA": t1, "BBB": t2},
        ),
        league,
        {"AAA": t1, "BBB": t2},
        context={"season_year": 2025, "calendar_cursor": 50 + TRADE_ACQUISITION_COOLDOWN_DAYS},
    )
    cooldown_blocks = [r for r in rules["blocking_reasons"] if "recently acquired" in r.lower()]
    assert not cooldown_blocks


def test_evaluation_returns_contract_slot_impact():
    t1 = _team("AAA", [_player("p1", cap_hit=2.0)])
    t2 = _team("BBB", [_player("p2", cap_hit=2.0)])
    league = _league([t1, t2])
    ev = evaluate_trade_package(
        {
            "BBB": [{"type": "player", "id": "p1", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p2", "team": "BBB"}],
        },
        league=league,
        team_by_id={"AAA": t1, "BBB": t2},
        context={"season_year": 2025, "calendar_cursor": 100, "regular_season_last_index": 192},
        user_team_id="AAA",
    )
    assert "contract_slot_impact" in ev
    assert "AAA" in ev["contract_slot_impact"]
    assert "after" in ev["contract_slot_impact"]["AAA"]


def test_weak_ambient_cpu_trade_rejected():
    t1 = _team("AAA", [_player("a1", cap_hit=8.0)])
    t2 = _team("BBB", [_player("b1", cap_hit=1.5)])
    t1.gm_window = "contender"
    t2.gm_window = "rebuild"
    league = _league([t1, t2])
    ev = evaluate_trade_package(
        {
            "BBB": [{"type": "player", "id": "a1", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "b1", "team": "BBB"}],
        },
        league=league,
        team_by_id={"AAA": t1, "BBB": t2},
        context={
            "season_year": 2025,
            "calendar_cursor": 100,
            "regular_season_last_index": 192,
            "cpu_ambient_trade": True,
        },
    )
    assert ev["can_execute"] is True
    assert ev["accepted"] is False


def test_fair_ambient_cpu_swap_can_pass():
    t1 = _team("AAA", [_player("a1", cap_hit=4.0)])
    t2 = _team("BBB", [_player("b1", cap_hit=4.0)])
    league = _league([t1, t2])
    ev = evaluate_trade_package(
        {
            "BBB": [{"type": "player", "id": "a1", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "b1", "team": "BBB"}],
        },
        league=league,
        team_by_id={"AAA": t1, "BBB": t2},
        context={
            "season_year": 2025,
            "calendar_cursor": 100,
            "regular_season_last_index": 192,
            "cpu_ambient_trade": True,
        },
    )
    assert ev["can_execute"] is True


def test_user_trade_execute_integration():
    from app.sim_engine.trades.trade_executor import execute_validated_trade
    from app.sim_engine.trades.trade_pick_registry import audit_pick_registry_integrity, canonical_pick_id

    p1 = _player("exec1", cap_hit=3.5)
    p2 = _player("exec2", cap_hit=3.5)
    t1 = _team("AAA", [p1])
    t2 = _team("BBB", [p2])
    league = _league([t1, t2])
    team_by_id = {"AAA": t1, "BBB": t2}
    pick_id = canonical_pick_id(2025, 3, "AAA")
    ctx = {"season_year": 2025, "calendar_cursor": 100, "regular_season_last_index": 192}

    evaluation = evaluate_trade_package(
        {
            "BBB": [
                {"type": "player", "id": "exec1", "team": "AAA"},
                {"type": "pick", "id": pick_id, "team": "AAA"},
            ],
            "AAA": [{"type": "player", "id": "exec2", "team": "BBB"}],
        },
        league=league,
        team_by_id=team_by_id,
        context=ctx,
        user_team_id="AAA",
    )
    assert evaluation["can_execute"] is True

    result = execute_validated_trade(
        evaluation,
        league=league,
        team_by_id=team_by_id,
        context=ctx,
        user_team_id="AAA",
    )
    assert result.get("trade_id")
    assert any(getattr(p, "id", "") == "exec1" for p in t2.roster)
    assert any(getattr(p, "id", "") == "exec2" for p in t1.roster)
    assert getattr(p1, "contract", None) is not None
    assert validate_pick_ownership(league, pick_id, "BBB")
    audit = audit_pick_registry_integrity(league)
    assert audit.get("ok", True) is not False
    all_player_ids = []
    for tm in (t1, t2):
        all_player_ids.extend(str(getattr(p, "id", "")) for p in tm.roster)
    assert len(all_player_ids) == len(set(all_player_ids))


def test_pick_value_bottom_five_first_gt_contender():
    from app.sim_engine.trades.trade_value import evaluate_pick_asset_value

    bottom = SimpleNamespace(team_id="BOT", id="BOT", gp=60, pts=50, w=20, l=35, otl=5, gm_window="rebuild", roster=[])
    top = SimpleNamespace(team_id="TOP", id="TOP", gp=60, pts=82, w=38, l=18, otl=4, gm_window="contender", roster=[])
    league = SimpleNamespace()
    ctx = {"season_year": 2025, "team_by_id": {"BOT": bottom, "TOP": top}}
    bottom_pick = {"pick_id": "2025-round1-BOT", "year": 2025, "round": 1, "original_team_id": "BOT"}
    top_pick = {"pick_id": "2025-round1-TOP", "year": 2025, "round": 1, "original_team_id": "TOP"}
    v_bot = float(evaluate_pick_asset_value(bottom_pick, bottom, bottom, league, context=ctx)["total"])
    v_top = float(evaluate_pick_asset_value(top_pick, top, top, league, context=ctx)["total"])
    assert v_bot > v_top + 5.0


def test_pick_value_protected_lt_unprotected():
    from app.sim_engine.trades.trade_value import evaluate_pick_asset_value

    team = SimpleNamespace(team_id="T", id="T", gp=60, pts=55, w=24, l=30, otl=6, gm_window="rebuild", roster=[])
    ctx = {"season_year": 2025, "team_by_id": {"T": team}}
    base = {"year": 2025, "round": 1, "original_team_id": "T"}
    v_open = float(evaluate_pick_asset_value({**base, "pick_id": "2025-round1-T"}, team, team, SimpleNamespace(), context=ctx)["total"])
    v_prot = float(evaluate_pick_asset_value({**base, "pick_id": "2025-round1-T-p", "protection": "top-10"}, team, team, SimpleNamespace(), context=ctx)["total"])
    assert v_prot < v_open


def test_pick_value_future_lt_current():
    from app.sim_engine.trades.trade_value import evaluate_pick_asset_value

    team = SimpleNamespace(team_id="T", id="T", gp=60, pts=55, w=24, l=30, otl=6, gm_window="rebuild", roster=[])
    ctx = {"season_year": 2025, "team_by_id": {"T": team}}
    base = {"round": 1, "original_team_id": "T"}
    v_now = float(evaluate_pick_asset_value({**base, "pick_id": "2025-round1-T", "year": 2025}, team, team, SimpleNamespace(), context=ctx)["total"])
    v_future = float(evaluate_pick_asset_value({**base, "pick_id": "2026-round1-T", "year": 2026}, team, team, SimpleNamespace(), context=ctx)["total"])
    assert v_future < v_now


def test_pick_value_later_rounds_lt_first():
    from app.sim_engine.trades.trade_value import evaluate_pick_asset_value

    team = SimpleNamespace(team_id="T", id="T", gp=60, pts=55, w=24, l=30, otl=6, roster=[])
    ctx = {"season_year": 2025, "team_by_id": {"T": team}}
    v1 = float(evaluate_pick_asset_value({"pick_id": "2025-round1-T", "year": 2025, "round": 1, "original_team_id": "T"}, team, team, SimpleNamespace(), context=ctx)["total"])
    v3 = float(evaluate_pick_asset_value({"pick_id": "2025-round3-T", "year": 2025, "round": 3, "original_team_id": "T"}, team, team, SimpleNamespace(), context=ctx)["total"])
    assert v3 < v1 - 10.0


def _injured_goalie(pid: str, ovr_val: float = 0.90, games: int = 18):
    ident = SimpleNamespace(position=SimpleNamespace(value="G"))
    return SimpleNamespace(
        id=pid,
        identity=ident,
        ovr=lambda: ovr_val,
        retired=False,
        age=29,
        _world_injury_games_remaining=games,
        _world_injury_tier="major",
    )


def test_compute_team_playoff_outlook_fields_and_ordering():
    from app.sim_engine.league.standings import StandingsTable
    from services.franchise_sim import compute_team_playoff_outlook

    top = _team("TOP", [_player("star")])
    top.gm_window = "contender"
    bottom = _team("BOT", [_player("scrub")])
    bottom.gm_window = "rebuild"

    standings = StandingsTable([top, bottom])
    standings.records["TOP"].gp = 60
    standings.records["TOP"].wins = 38
    standings.records["TOP"].losses = 16
    standings.records["TOP"].otl = 6
    standings.records["TOP"].points = 82
    standings.records["BOT"].gp = 60
    standings.records["BOT"].wins = 20
    standings.records["BOT"].losses = 35
    standings.records["BOT"].otl = 5
    standings.records["BOT"].points = 45

    session = SimpleNamespace(user_team_id="TOP", standings=standings, game_results=[])

    top_out = compute_team_playoff_outlook(session, top)
    bot_out = compute_team_playoff_outlook(session, bottom)

    assert 0 <= top_out["playoff_odds"] <= 99
    assert top_out["playoff_pct"] == top_out["playoff_odds"]
    assert top_out["outlook_label"] == top_out["contention_label"]
    assert top_out["playoff_odds"] > bot_out["playoff_odds"]
    assert top_out["health_adjusted_rating"] > 0
    assert "standings_context" in top_out


def test_compute_team_playoff_outlook_injury_reduces_rating():
    from services.franchise_sim import compute_team_playoff_outlook

    healthy_team = _team("T", [_player("c1"), _injured_goalie("g1", games=0)])
    healthy_team.gm_window = "contender"
    healthy_team.roster[1]._world_injury_games_remaining = 0
    healthy_team.roster[1]._world_injury_tier = ""

    injured_team = _team("T", [_player("c1"), _injured_goalie("g1", games=20)])
    injured_team.gm_window = "contender"

    session = SimpleNamespace(user_team_id="T", standings=None, game_results=[])
    healthy = compute_team_playoff_outlook(session, healthy_team)
    injured = compute_team_playoff_outlook(session, injured_team)

    assert injured["injury_impact"] > 0
    assert injured["health_adjusted_rating"] < healthy["health_adjusted_rating"]


def test_trade_assets_payload_includes_playoff_outlook():
    from services.trade_service import build_trade_assets_payload

    t1 = _team("AAA", [_player("p1")])
    t1.gm_window = "contender"
    session = SimpleNamespace(
        session_id="test-playoff-outlook",
        user_team_id="AAA",
        team_by_id={"AAA": t1},
        standings=None,
        game_results=[],
        sim=SimpleNamespace(league=_league([t1]), rng=None),
        season_calendar_year=2025,
        calendar_cursor=100,
        nhl_calendar=[],
        nhl_regular_season_last_index=192,
        transcendent_tank_pressure={},
        transcendent_draft_prospect_id=None,
    )

    payload = build_trade_assets_payload(session)
    team_blob = payload["teams"]["AAA"]
    assert "playoff_odds" in team_blob
    assert "outlook_label" in team_blob
    assert "health_adjusted_rating" in team_blob
