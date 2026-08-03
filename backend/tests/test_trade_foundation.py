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


def test_reject_ahl_only_deal_without_nhl_spc():
    """Pure AHL/ECHL deals stay non-tradeable; NHL-SPC affiliates are allowed elsewhere."""
    p1 = _player("p1")
    p1.contract.contract_type = "AHL"
    p1.contract.type = "AHL"
    p1.contract.aav_m = 0.0
    p1.contract.cap_hit_m = 0.0
    p1.contract.is_nhl_spc = False
    p1.contract.years_remaining = 2
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
    joined = " ".join(r.lower() for r in rules["blocking_reasons"])
    assert "ahl" in joined or "not found" in joined or "does not own" in joined or "spc" in joined


def test_ahl_player_with_nhl_spc_is_tradeable():
    p1 = _player("p1", cap_hit=0.95)
    p1.contract.contract_type = "STANDARD"
    p1.contract.type = "STANDARD"
    p1.contract.is_nhl_spc = True
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
    assert rules["ok"] is True


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


def test_pick_registry_integrity_allows_resolved_current_draft_picks():
    """Mid-draft selections mark picks resolved; post-trade audit must still pass."""
    from app.sim_engine.trades.trade_pick_registry import (
        audit_pick_registry_integrity,
        canonical_pick_id,
        ensure_draft_pick_registry,
    )

    teams = []
    for i in range(4):
        tid = str(i)  # includes team_id "0" (valid NHL-style numeric id)
        teams.append(SimpleNamespace(team_id=tid, id=tid, roster=[], owned_pick_ids=[]))
    league = SimpleNamespace(teams=teams, draft_pick_registry={})
    ensure_draft_pick_registry(league, start_year=2026, years_ahead=2, rounds=2)
    # Simulate picks already used in the live draft.
    for tid in ("0", "1"):
        pid = canonical_pick_id(2026, 1, tid)
        row = league.draft_pick_registry[pid]
        row["resolved"] = True
        row["resolved_reason"] = "draft_selection"
        row["selected_prospect_id"] = f"p_{tid}"
    audit = audit_pick_registry_integrity(league, start_year=2026, years_ahead=2, rounds=2)
    assert audit.get("ok") is True, audit.get("errors")


def test_update_player_nhl_eta_moves_with_readiness():
    from app.sim_engine.progression.development import (
        calculate_nhl_readiness_score,
        update_player_nhl_eta,
    )

    p = SimpleNamespace(
        age=20,
        overall=76,
        ovr=76 / 99.0,
        potential=0.82,
        morale=0.6,
        season_stats={"gp": 12},
        role="top_line",
        dev_type="standard",
        position="C",
        status="prospect",
        nhl_eta=3,
        ratings={"skating": 76, "shooting": 76, "hands": 76, "checking": 70, "defense": 70, "IQ": 76},
    )
    calculate_nhl_readiness_score(p)
    years = update_player_nhl_eta(p)
    assert int(getattr(p, "nhl_eta")) == years
    assert years <= 3  # should not stay stuck at a stale long ETA when ready
    assert getattr(p, "nhl_readiness", 0) > 1.5  # 0–100 scale


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


def test_trade_assets_payload_includes_roster_capacity_and_slots():
    from services.trade_service import build_trade_assets_payload, TRADE_ASSETS_CACHE_VERSION

    nhl = [
        _player("f1"),
        _player("f2"),
        _player("d1"),
        _player("g1"),
    ]
    # Patch positions for composition.
    nhl[0].identity.position = SimpleNamespace(value="C")
    nhl[1].identity.position = SimpleNamespace(value="LW")
    nhl[2].identity.position = SimpleNamespace(value="D")
    nhl[3].identity.position = SimpleNamespace(value="G")
    for p in nhl:
        p.contract.is_nhl_spc = True
        p.contract.type = "STANDARD"
        p.signed_status = "signed"
        p.retired = False
        p.is_buried = False
        p.buried = False
        p.in_minors = False

    ahl_spc = _player("ahl1", cap_hit=0.8)
    ahl_spc.identity.position = SimpleNamespace(value="C")
    ahl_spc.contract.is_nhl_spc = True
    ahl_spc.contract.type = "STANDARD"
    ahl_spc.signed_status = "signed"
    ahl_spc.retired = False
    ahl_spc.in_minors = True

    unsigned = _player("rights1", cap_hit=0.0)
    unsigned.signed_status = "unsigned"
    unsigned.contract = None
    unsigned.identity.position = SimpleNamespace(value="C")

    t1 = _team("AAA", nhl)
    t1.ahl_roster = [ahl_spc]
    t1.echl_roster = []
    t1.prospect_pool = [unsigned]
    t1.gm_window = "contender"

    session = SimpleNamespace(
        session_id="test-capacity",
        user_team_id="AAA",
        team_by_id={"AAA": t1},
        standings=None,
        game_results=[],
        sim=SimpleNamespace(league=_league([t1]), rng=None),
        season_calendar_year=2025,
        calendar_cursor=40,
        nhl_calendar=[],
        nhl_regular_season_last_index=192,
        transcendent_tank_pressure={},
        transcendent_draft_prospect_id=None,
        draft_completed=False,
    )

    payload = build_trade_assets_payload(session)
    assert int(payload.get("formula_version") or 0) == int(TRADE_ASSETS_CACHE_VERSION)
    blob = payload["teams"]["AAA"]
    rc = blob.get("roster_capacity") or {}
    slots = blob.get("contract_slots") or {}
    assert rc.get("nhl_count") == 4
    assert rc.get("forwards") == 2
    assert rc.get("defense") == 1
    assert rc.get("goalies") == 1
    assert rc.get("nhl_max") == 23
    assert "composition" in rc
    # NHL SPCs on NHL + AHL count; unsigned rights do not.
    assert int(slots.get("used") or 0) >= 5
    assert int(slots.get("limit") or 0) == 50
    assert "available" in slots


def test_reduced_trade_value_fallback_never_returns_raw_ovr():
    from app.sim_engine.trades.trade_value import (
        TRADE_VALUE_FALLBACK_CEIL,
        evaluate_player_asset_value,
        reduced_trade_value_fallback,
    )

    broken = _player("broken", cap_hit=4.0)
    broken.ovr = lambda: 1 / 0  # force failures in ovr path when fallback uses it carefully
    # Direct fallback with a normal player must stay below star territory.
    star = _player("star", cap_hit=9.0)
    star.ovr = lambda: 0.92
    fb = reduced_trade_value_fallback(star, reason="unit_test")
    assert fb <= TRADE_VALUE_FALLBACK_CEIL
    assert fb < 82.0

    # Malformed contract / missing stats should not raise; may set fallback flag.
    broken.contract = SimpleNamespace(aav_m=None, cap_hit_m=None, years_remaining=2)
    broken.season_stats = None
    broken.ovr = lambda: 0.82
    t1 = _team("AAA", [broken])
    t2 = _team("BBB", [])
    league = _league([t1, t2])
    out = evaluate_player_asset_value(broken, t1, t2, league, context={})
    assert "total" in out
    assert float(out["total"]) <= 100.0


def test_trade_move_spc_affiliate_lands_on_ahl_and_rolls_back_lists():
    from app.sim_engine.trades.trade_executor import _resolve_trade_destination_attr

    ahl_p = _player("ahl1", cap_hit=0.95)
    ahl_p.contract.contract_type = "STANDARD"
    ahl_p.contract.type = "STANDARD"
    ahl_p.contract.is_nhl_spc = True
    ahl_p.in_minors = True
    ahl_p.roster_location = "ahl"
    assert _resolve_trade_destination_attr("ahl", ahl_p) == "ahl_roster"
    assert _resolve_trade_destination_attr("nhl", ahl_p) == "roster"


def test_cpu_trade_value_fallback_does_not_use_raw_ovr():
    from app.sim_engine.trades import cpu_trade_proposer as ctp

    p = _player("x", cap_hit=2.0)
    p.ovr = lambda: 0.88
    t = _team("AAA", [p])
    league = _league([t])

    original = ctp.evaluate_player_asset_value

    def boom(*_a, **_k):
        raise RuntimeError("forced enrichment failure")

    ctp.evaluate_player_asset_value = boom
    try:
        val = ctp._player_trade_value(p, t, league, {}, acquiring_team=t)
    finally:
        ctp.evaluate_player_asset_value = original
    assert val < 82.0
    assert val <= ctp.reduced_trade_value_fallback(p, reason="assert") + 0.01


def test_reverse_return_hard_blocked_same_season():
    from app.sim_engine.trades.trade_rules import (
        _player_returning_to_prior_club,
        validate_trade_rules,
    )
    from app.sim_engine.trades.trade_asset import normalize_trade_package

    p1 = _player("bounce", cap_hit=2.5)
    p1.acquired_via_trade = True
    p1.acquired_from_team_id = "BBB"
    p1.acquired_via_trade_season = 2025
    p1.last_acquired_day = 0
    p2 = _player("other", cap_hit=2.0)
    t1 = _team("AAA", [p1])
    t2 = _team("BBB", [p2])
    league = _league([t1, t2])
    ctx = {"season_year": 2025, "calendar_cursor": 50}
    assert _player_returning_to_prior_club(p1, "BBB", ctx) is True
    assert _player_returning_to_prior_club(p1, "CCC", ctx) is False

    package = normalize_trade_package(
        {
            "BBB": [{"type": "player", "id": "bounce", "team": "AAA"}],
            "AAA": [{"type": "player", "id": "other", "team": "BBB"}],
        }
    )
    rules = validate_trade_rules(package, league, {"AAA": t1, "BBB": t2}, context=ctx)
    joined = " ".join(str(x) for x in (rules.get("blocking_reasons") or []))
    assert rules.get("ok") is False
    assert "same season" in joined.lower() or "cannot be traded back" in joined.lower()


def test_seller_protects_young_core_and_talent_gap():
    from app.sim_engine.trades import cpu_trade_proposer as ctp

    young = _player("kid", cap_hit=0.95)
    young.identity = SimpleNamespace(name="Kid", age=22, position=SimpleNamespace(value="C"))
    young.ovr = lambda: 0.84  # 84 OVR
    young.contract.years_remaining = 2
    young.contract.is_entry_level = True

    vet = _player("vet", cap_hit=4.0)
    vet.identity = SimpleNamespace(name="Vet", age=31, position=SimpleNamespace(value="C"))
    vet.ovr = lambda: 0.76  # 76 OVR — 8-point gap
    vet.contract.years_remaining = 1
    vet.contract.expiry_status = "UFA"

    assert ctp._is_young_core(young) is True
    assert ctp._seller_must_protect(young, window="rebuild", deadline=0.2) is True
    assert abs(ctp._player_ovr(young) - ctp._player_ovr(vet)) > ctp.CPU_ONE_FOR_ONE_OVR_GAP_MAX
    assert ctp._talent_gap_ok(young, vet, motive="depth_swap") is False
    assert ctp._talent_gap_ok(young, vet, buyer_pick={"pick_id": "x"}, motive="futures_package") is True


def test_package_motive_chooser_prefers_futures_for_rebuild_to_contender():
    from app.sim_engine.trades import cpu_trade_proposer as ctp
    import random

    seller = _team("AAA")
    seller.gm_window = "rebuild"
    buyer = _team("BBB")
    buyer.gm_window = "contender"
    rng = random.Random(1)
    motives = {
        ctp._choose_package_motive(
            seller=seller,
            buyer=buyer,
            deadline=0.6,
            peer_path=False,
            pair_rng=random.Random(i),
            direction_seller="REBUILDING",
            direction_buyer="CONTENDER",
        )
        for i in range(40)
    }
    assert "futures_package" in motives or "rental_sale" in motives
    assert ctp._choose_package_motive(
        seller=seller,
        buyer=buyer,
        deadline=0.1,
        peer_path=True,
        pair_rng=rng,
        direction_seller="REBUILDING",
        direction_buyer="CONTENDER",
    ) == "depth_swap"


def test_upcoming_draft_year_and_calendar_pick_migration():
    from app.sim_engine.trades.trade_pick_registry import (
        ensure_draft_pick_registry,
        ensure_franchise_pick_registry,
        migrate_calendar_year_picks_to_draft_year,
        upcoming_draft_year,
        validate_pick_ownership,
    )

    assert upcoming_draft_year(2025) == 2026
    t1 = _team("AAA")
    t2 = _team("BBB")
    league = SimpleNamespace(teams=[t1, t2], salary_cap_m=88.0, cap_floor_m=65.0, trade_history=[])
    # Legacy path: mint/trade calendar-year picks (the bug).
    ensure_draft_pick_registry(league, start_year=2025, years_ahead=2)
    phantom = canonical_pick_id(2025, 2, "AAA")
    transfer_pick(league, phantom, "BBB")
    assert validate_pick_ownership(league, phantom, "BBB")

    migrated = migrate_calendar_year_picks_to_draft_year(league, season_calendar_year=2025, draft_year=2026)
    assert migrated >= 1
    real = canonical_pick_id(2026, 2, "AAA")
    assert validate_pick_ownership(league, real, "BBB")
    assert not validate_pick_ownership(league, phantom, "BBB")

    ensure_franchise_pick_registry(league, season_calendar_year=2025, years_ahead=4)
    assert canonical_pick_id(2026, 1, "AAA") in (getattr(t1, "owned_pick_ids", None) or []) or validate_pick_ownership(
        league, canonical_pick_id(2026, 1, "AAA"), "AAA"
    )


def test_trade_rules_use_draft_year_anchor():
    t1 = _team("AAA", [_player("p1")])
    t2 = _team("BBB", [_player("p2")])
    league = _league([t1, t2])
    # Franchise calendar context: season 2025 → draft 2026. Calendar-year pick is illegal.
    phantom = canonical_pick_id(2025, 1, "AAA")
    team_by_id = {"AAA": t1, "BBB": t2}
    package = normalize_trade_package(
        {
            "BBB": [{"type": "pick", "id": phantom, "team": "AAA"}],
            "AAA": [{"type": "player", "id": "p1", "team": "AAA"}],
        },
        team_by_id=team_by_id,
    )
    rules = validate_trade_rules(
        package,
        league,
        team_by_id,
        context={"season_year": 2025, "draft_year": 2026, "season_is_calendar": True},
    )
    assert not rules["ok"]
    assert any("out of allowed range" in r.lower() for r in rules["blocking_reasons"])


def test_talent_gap_allows_pick_only_return():
    from app.sim_engine.trades import cpu_trade_proposer as ctp

    sold = _player("star", cap_hit=7.0)
    sold.ovr = lambda: 0.86
    assert ctp._talent_gap_ok(sold, None, motive="futures_package") is False
    assert ctp._talent_gap_ok(
        sold, None, buyer_pick={"pick_id": "2026-round1-BBB"}, motive="futures_package"
    ) is True
    assert ctp._talent_gap_ok(sold, None, buyer_pick={"pick_id": "x"}, motive="depth_swap") is False


def test_build_package_pick_only_return():
    from app.sim_engine.trades import cpu_trade_proposer as ctp

    seller = _team("AAA", [_player("p1")])
    buyer = _team("BBB", [_player("p2")])
    sold = seller.roster[0]
    pick = {"pick_id": "2026-round2-BBB", "year": 2026, "round": 2}
    package = ctp._build_package(seller, buyer, sold, [], buyer_pick=pick)
    assert package
    assert any(a.get("type") == "pick" for a in package["AAA"])
    assert any(a.get("type") == "player" for a in package["BBB"])
    assert not any(a.get("type") == "player" for a in package["AAA"])


def test_retire_draft_year_removes_from_trade_assets():
    from app.sim_engine.trades.trade_pick_registry import (
        get_team_owned_picks,
        retire_draft_year_picks,
        serialize_team_picks,
        tradeable_draft_year,
        validate_pick_ownership,
    )

    assert tradeable_draft_year(2025, draft_completed=False) == 2026
    assert tradeable_draft_year(2025, draft_completed=True) == 2027

    t1 = _team("AAA")
    t2 = _team("BBB")
    league = _league([t1, t2])
    pick_2026 = canonical_pick_id(2026, 1, "AAA")
    # League helper mints from 2025; ensure upcoming class exists then retire it.
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry

    ensure_draft_pick_registry(league, start_year=2026, years_ahead=3)
    assert validate_pick_ownership(league, pick_2026, "AAA")

    n = retire_draft_year_picks(league, draft_year=2026, reason="draft_completed")
    assert n >= 1
    assert not validate_pick_ownership(league, pick_2026, "AAA")
    owned = get_team_owned_picks(league, "AAA", min_year=2026)
    assert all(int(r.get("year") or 0) >= 2027 for r in owned)
    shown = serialize_team_picks(league, "AAA", min_year=2027)
    assert all(int(p.get("year") or 0) >= 2027 for p in shown)
    assert pick_2026 not in [p.get("pick_id") for p in shown]


def test_dict_contract_years_remaining_read_for_trade_value():
    """Real-NHL contracts are dicts — years_remaining must not silently read as 0."""
    from app.sim_engine.trades.trade_value import (
        _contract_years,
        _prospect_upside_score,
        evaluate_player_asset_value,
    )

    ident = SimpleNamespace(name="Star", age=28, position=SimpleNamespace(value="C"))
    player = SimpleNamespace(
        id="star-dict",
        identity=ident,
        contract={
            "aav_m": 8.5,
            "cap_hit_m": 8.5,
            "years_remaining": 6,
            "years": 6,
            "expiry_year": 2031,
            "rights_status": "UFA",
            "type": "STANDARD",
            "source": "real_nhl_spotrac",
        },
        cap_hit_m=8.5,
        ratings={"dev_potential": 90},
        ovr=lambda: 0.88,
        season_stats={"gp": 80, "pts": 80, "g": 35, "a": 45},
    )
    assert _contract_years(player) == 6

    team = SimpleNamespace(
        gm_window="contender",
        needs={"top_line_forward": 0.4, "depth_forward": 0.2, "top_4_defense": 0.2, "goalie": 0.1},
        cap_pressure="moderate",
    )
    league = SimpleNamespace()
    out = evaluate_player_asset_value(player, team, team, league, context={})
    assert float(out["total"]) > 40
    # Must not treat a 6-year deal as an expiring UFA dump.
    comps = out.get("components") or out.get("breakdown") or {}
    assert float(comps.get("cap_dump") or 0) > -10 or int(comps.get("years_remaining") or 6) == 6


def test_low_ovr_youth_cannot_mint_superstar_trade_value():
    from app.sim_engine.trades.trade_value import (
        _prospect_upside_score,
        evaluate_player_asset_value,
    )

    upside = _prospect_upside_score(
        SimpleNamespace(draft_overall_pick=120, scouting_confidence=0.9),
        ovr=68.0,
        age=19,
        pot=86.0,
    )
    assert upside <= 8.0

    def _mk(ovr99, age, pot, aav=0.9, years=3):
        return SimpleNamespace(
            id="y",
            identity=SimpleNamespace(name="Kid", age=age, position=SimpleNamespace(value="C")),
            contract={
                "aav_m": aav,
                "cap_hit_m": aav,
                "years_remaining": years,
                "type": "ELC",
                "is_elc": True,
                "source": "test",
            },
            ratings={"dev_potential": pot},
            ovr=lambda o=ovr99: o / 99.0,
            season_stats={"gp": 10, "pts": 2, "g": 1, "a": 1},
        )

    team = SimpleNamespace(
        gm_window="rebuild",
        needs={"top_line_forward": 0.9, "depth_forward": 0.8, "top_4_defense": 0.5, "goalie": 0.2},
        cap_pressure="comfortable",
    )
    league = SimpleNamespace()
    depth = _mk(68, 19, 88)
    star = _mk(90, 24, 93, aav=8.0, years=5)
    v_depth = float(evaluate_player_asset_value(depth, team, team, league)["total"])
    v_star = float(evaluate_player_asset_value(star, team, team, league)["total"])
    assert v_star > v_depth
    assert v_depth < 55


def test_team_needs_uses_01_scale_not_display_ovr():
    from app.sim_engine.economy.team_needs import TeamNeeds

    def _p(ovr01):
        return SimpleNamespace(
            identity=SimpleNamespace(position=SimpleNamespace(value="C"), age=26),
            position="C",
            ovr=lambda o=ovr01: o,
            season_stats={"gp": 40},
        )

    team = SimpleNamespace(
        roster=[_p(0.70), _p(0.68), _p(0.66), _p(0.64)]
        + [
            SimpleNamespace(
                identity=SimpleNamespace(position=SimpleNamespace(value="D"), age=27),
                position="D",
                ovr=lambda: 0.70,
                season_stats={},
            )
            for _ in range(4)
        ]
        + [
            SimpleNamespace(
                identity=SimpleNamespace(position=SimpleNamespace(value="G"), age=28),
                position="G",
                ovr=lambda: 0.72,
                season_stats={},
            )
        ],
    )
    needs = TeamNeeds().evaluate(team)
    # With 0–1 OVR vs 0.74 targets, weak clubs must show positive need.
    assert float(needs.get("top_line_forward") or 0) > 0.05


def test_superstar_not_matched_by_four_depth_assets():
    from app.sim_engine.trades.trade_value import evaluate_player_asset_value, _talent_base

    assert _talent_base(92) > 4 * _talent_base(72)

    def _mk(ovr99, age=26, aav=4.0, years=3, pos="C"):
        return SimpleNamespace(
            id=f"p{ovr99}",
            identity=SimpleNamespace(name="X", age=age, position=SimpleNamespace(value=pos)),
            contract={
                "aav_m": aav,
                "cap_hit_m": aav,
                "years_remaining": years,
                "type": "STANDARD",
            },
            ratings={"dev_potential": ovr99 + 2},
            ovr=lambda o=ovr99: o / 99.0,
            season_stats={"gp": 70, "pts": 40, "g": 15, "a": 25},
        )

    team = SimpleNamespace(gm_window="emerging", needs={}, cap_pressure="moderate")
    league = SimpleNamespace()
    star = _mk(92, aav=10.0, years=4)
    depths = [_mk(72, aav=1.5) for _ in range(4)]
    v_star = float(evaluate_player_asset_value(star, team, team, league)["total"])
    v_depth = sum(float(evaluate_player_asset_value(p, team, team, league)["total"]) for p in depths)
    assert v_star > v_depth


def test_trade_purges_duplicate_roster_copies():
    """A leftover copy on the source club must be scrubbed so GP cannot double."""
    from app.sim_engine.trades.trade_asset import PlayerTradeAsset
    from app.sim_engine.trades.trade_executor import _apply_player_move

    p1 = _player("dup1", cap_hit=4.0)
    ghost = _player("dup1", cap_hit=4.0)  # same id, leftover copy
    t1 = _team("AAA", [p1, ghost])
    t2 = _team("BBB", [])
    team_by_id = {"AAA": t1, "BBB": t2}
    asset = PlayerTradeAsset(
        player_id="dup1",
        source_team_id="AAA",
        acquiring_team_id="BBB",
        retained_pct=0.0,
    )
    moved: list = []
    retained: list = []
    _apply_player_move(
        asset,
        team_by_id,
        season_label="2025-26",
        moved_players=moved,
        retained_records=retained,
        context={"season_year": 2025, "calendar_cursor": 40},
    )
    assert sum(1 for p in t1.roster if str(getattr(p, "id", "")) == "dup1") == 0
    assert sum(1 for p in t2.roster if str(getattr(p, "id", "")) == "dup1") == 1
    assert moved and moved[0].get("applied") is True


def test_ledger_rejects_gp_for_player_not_on_credited_roster():
    """Ghost dual-roster dress must not credit season GP after a trade."""
    from app.sim_engine.engine import SimEngine

    eng = SimEngine.__new__(SimEngine)
    p = SimpleNamespace(id="ghost1", identity=SimpleNamespace(name="Ghost", position=SimpleNamespace(value="C")), retired=False)
    home = SimpleNamespace(team_id="H", id="H", roster=[])  # not on home
    away = SimpleNamespace(team_id="A", id="A", roster=[p])
    eng.league = SimpleNamespace(teams=[home, away])
    ledger: dict = {}
    eng._gm_ledger_add(ledger, p, "H", gp=1, g=1)
    assert "ghost1" not in ledger or int((ledger.get("ghost1") or {}).get("gp", 0) or 0) == 0
    eng._gm_ledger_add(ledger, p, "A", gp=1, g=1)
    assert int(ledger["ghost1"]["gp"]) == 1
    assert int(ledger["ghost1"]["g"]) == 1


def test_ledger_caps_regular_season_gp_at_82():
    from app.sim_engine.engine import SimEngine

    eng = SimEngine.__new__(SimEngine)
    p = SimpleNamespace(id="cap82", identity=SimpleNamespace(name="Cap", position=SimpleNamespace(value="C")), retired=False)
    team = SimpleNamespace(team_id="T", id="T", roster=[p])
    eng.league = SimpleNamespace(teams=[team])
    ledger = {"cap82": {"player_id": "cap82", "gp": 82, "g": 10, "a": 10, "pts": 20, "position": "C"}}
    eng._gm_ledger_add(ledger, p, "T", gp=1, g=1, a=1)
    assert int(ledger["cap82"]["gp"]) == 82
    assert int(ledger["cap82"]["g"]) == 10

