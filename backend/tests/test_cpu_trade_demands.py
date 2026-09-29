"""CPU trade demands: losing/character pressure, deadline window, and CPU demand resolution."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.sim_engine.franchise.trade_stability_engine import (  # noqa: E402
    PlayerConcernSnapshot,
    _temperament_pressure,
    _winning_pressure,
    count_significant_pressures,
    winning_satisfaction_from_points_pct,
)
from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry  # noqa: E402


def test_winning_satisfaction_spans_real_standings():
    assert winning_satisfaction_from_points_pct(0.500) == pytest.approx(60.0)
    assert winning_satisfaction_from_points_pct(0.350) == pytest.approx(30.0)
    assert winning_satisfaction_from_points_pct(0.650) == pytest.approx(90.0)
    # Old curve (35 + pct*65) left a last-place club at ~58 — above the 62 floor's reach.
    assert winning_satisfaction_from_points_pct(0.380) < 45.0


def test_losing_is_significant_pressure_for_competitors():
    ws = winning_satisfaction_from_points_pct(0.360)
    competitor = PlayerConcernSnapshot(winning_satisfaction=ws, competitiveness=85, character=74)
    assert _winning_pressure(competitor) >= 7.5
    winner = PlayerConcernSnapshot(
        winning_satisfaction=winning_satisfaction_from_points_pct(0.620), competitiveness=85, character=74
    )
    assert _winning_pressure(winner) == 0.0


def test_low_character_generates_its_own_pressure():
    losing = winning_satisfaction_from_points_pct(0.380)
    bad_apple = PlayerConcernSnapshot(character=50, winning_satisfaction=losing, role_satisfaction=45)
    assert _temperament_pressure(bad_apple) >= 7.5
    assert count_significant_pressures({"temperament": _temperament_pressure(bad_apple)}) == 1
    # Same player on a winner is quieter; a pro is silent regardless.
    content = PlayerConcernSnapshot(character=50, winning_satisfaction=80, role_satisfaction=70)
    assert 0.0 < _temperament_pressure(content) < _temperament_pressure(bad_apple)
    pro = PlayerConcernSnapshot(character=82, winning_satisfaction=losing, role_satisfaction=45)
    assert _temperament_pressure(pro) == 0.0


def _session_on(iso: str, phase: str = "regular"):
    return SimpleNamespace(phase=phase, season_phase="", calendar_cursor=0, nhl_calendar=[{"iso": iso}])


def test_deadline_window_open_in_october_through_march_tenth():
    from services.trade_demand_engine import get_trade_deadline_context

    for iso in ("2026-10-15", "2026-12-31", "2027-01-20", "2027-03-10"):
        ctx = get_trade_deadline_context(_session_on(iso))
        assert ctx["past_deadline"] is False, iso
        assert ctx["new_demands_allowed"] is True, iso
    assert get_trade_deadline_context(_session_on("2026-10-15"))["days_to_deadline"] == 146
    assert get_trade_deadline_context(_session_on("2027-03-11"))["past_deadline"] is True


def test_demand_trade_chance_rises_with_time_and_deadline():
    from app.sim_engine.trades.cpu_trade_proposer import demand_trade_chance

    fresh = demand_trade_chance(0, 0.0, disruptor=False)
    stale = demand_trade_chance(25, 0.0, disruptor=False)
    late = demand_trade_chance(25, 0.8, disruptor=False)
    assert fresh < stale < late <= 0.92
    assert demand_trade_chance(0, 0.0, disruptor=True) > fresh


# --- CPU proposer / evaluator ------------------------------------------------


def _player(pid: str, *, ovr: float = 0.80, age: int = 27, pos: str = "C", ntc: bool = False):
    contract = SimpleNamespace(
        cap_hit_m=3.0, aav_m=3.0, years_remaining=2, no_trade_clause=ntc, no_move_clause=False,
        modified_no_trade_teams=0, approved_trade_teams=[], clauses=None,
    )
    ident = SimpleNamespace(name=f"Player {pid}", age=age, position=SimpleNamespace(value=pos))
    return SimpleNamespace(
        id=pid, name=f"Player {pid}", identity=ident, contract=contract, cap_hit_m=3.0,
        ovr=lambda o=ovr: o, season_stats={"gp": 40, "pts": 30, "g": 12, "a": 18},
    )


def _team(tid: str, players, window: str = "emerging"):
    return SimpleNamespace(
        team_id=tid, id=tid, abbr=tid, roster=list(players), owned_pick_ids=[], needs={},
        gm_window=window, window=window, cap_pressure="moderate", cap_pressure_tier="moderate",
        retained_salary_records=[],
    )


def _league(teams):
    lg = SimpleNamespace(teams=teams, salary_cap_m=88.0, cap_floor_m=65.0, trade_history=[])
    ensure_draft_pick_registry(lg, start_year=2025, years_ahead=4)
    return lg


def _flag_demand(player, team_id: str, *, opened_day: int, dests=()):
    player._trade_demand_active = True
    player._trade_demand_opened_day = opened_day
    player._trade_demand_team_id = team_id
    player._trade_demand_destinations = list(dests)
    player._crisis_trade_stage = 1


def test_collect_cpu_trade_demands_oldest_first_and_skips_stale_owner():
    from app.sim_engine.trades.cpu_trade_proposer import collect_cpu_trade_demands

    a, b, c = _player("a"), _player("b"), _player("c")
    _flag_demand(a, "AAA", opened_day=90)
    _flag_demand(b, "AAA", opened_day=60)
    _flag_demand(c, "OLD", opened_day=10)  # flag left over from a previous club
    rows = collect_cpu_trade_demands([_team("AAA", [a, b, c])], calendar_cursor=100)
    assert [r[1].id for r in rows] == ["b", "a"]
    assert rows[0][2] == 40


def test_evaluator_seller_accepts_discount_for_demanding_player():
    from app.sim_engine.trades.trade_evaluator import evaluate_trade_package

    star = _player("star", ovr=0.76)
    depth = _player("depth", ovr=0.72)
    t1 = _team("AAA", [star], window="contender")
    t2 = _team("BBB", [depth], window="contender")
    league = _league([t1, t2])
    package = {
        "BBB": [{"type": "player", "id": "star", "team": "AAA"}],
        "AAA": [{"type": "player", "id": "depth", "team": "BBB"}],
    }
    base_ctx = {"season_year": 2025, "calendar_cursor": 100, "regular_season_last_index": 192, "cpu_ambient_trade": True}
    plain = evaluate_trade_package(package, league=league, team_by_id={"AAA": t1, "BBB": t2}, context=base_ctx)
    demand_ctx = dict(base_ctx, cpu_demand_trade={"seller_team_id": "AAA", "player_id": "star", "stage": 2})
    forced = evaluate_trade_package(package, league=league, team_by_id={"AAA": t1, "BBB": t2}, context=demand_ctx)
    # A modest value loss the club would normally refuse...
    assert plain["interest_level"]["AAA"] < 0.46
    assert plain["accepted"] is False
    # ...is taken when the player has forced the issue.
    assert forced["interest_level"]["AAA"] >= 0.46
    assert forced["accepted"] is True


def test_cpu_proposer_moves_demanding_player():
    from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades

    moved = 0
    for seed in range(12):
        unhappy = _player("unhappy", ovr=0.80, age=27)
        _flag_demand(unhappy, "AAA", opened_day=40, dests=["BBB", "CCC"])
        seller = _team("AAA", [unhappy, _player("a2", ovr=0.74)], window="emerging")
        b = _team("BBB", [_player("b1", ovr=0.79), _player("b2", ovr=0.73)], window="contender")
        c = _team("CCC", [_player("c1", ovr=0.78), _player("c2", ovr=0.71)], window="contender")
        league = _league([seller, b, c])
        league.rng = types.SimpleNamespace(randint=lambda lo, hi, s=seed: lo + s)
        trades = propose_and_execute_cpu_trades(
            league, max_executions=1, calendar_cursor=100, regular_season_last_index=192,
        )
        demand_trades = [t for t in trades if t.get("package_motive") == "demand_resolution"]
        if demand_trades:
            moved += 1
            t = demand_trades[0]
            assert t["demand_player_id"] == "unhappy"
            assert t["trade_category"] == "trade_demand"
            assert t["from_team_id"] == "AAA"
            assert all(p.id != "unhappy" for p in seller.roster)
    assert moved >= 3, f"demand trades executed in only {moved}/12 seeds"


def test_stability_catchup_from_saved_state_matches_daily_updates(monkeypatch):
    """Batch sims check weekly; replaying from the saved score must equal daily checks."""
    import app.sim_engine.franchise.trade_stability_engine as tse

    fixed = {
        "player_id": "p",
        "trade_stability_score": 25.0,
        "escalation_level": 3,
        "pressures": {"winning": 12.0, "temperament": 9.0},
        "character": 55,
        "mental": 70,
        "readiness_penalties": {},
    }
    monkeypatch.setattr(tse, "compute_instant_stability", lambda *a, **k: dict(fixed))

    def run(check_days):
        session = SimpleNamespace()
        player = SimpleNamespace(id="p")
        tse.ensure_trade_stability_state(session)["p"] = {
            "trade_stability_score": 88.0, "escalation_level": 0, "last_calendar_day": 100,
            "last_level_change_day": 100,
        }
        row = None
        for day in check_days:
            row = tse.apply_daily_stability_update(session, player, None, day)
        return row

    daily = run(range(101, 141))
    weekly = run([107, 114, 121, 128, 135, 140])
    assert weekly["trade_stability_score"] == pytest.approx(daily["trade_stability_score"], abs=0.25)  # daily path rounds each step
    assert weekly["escalation_level"] == daily["escalation_level"]
    assert weekly["stability_concern_days"] == daily["stability_concern_days"] == 40
    assert daily["escalation_level"] >= 2
