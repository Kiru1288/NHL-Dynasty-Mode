"""Needs-driven CPU trade market: real deadline, AHL-only freeze, team assessment, matching."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry  # noqa: E402

_N = [0]


def _p(pos: str, ovr: float, *, age: int = 28, years: int = 2, shoots: str = "L", cap: float = 2.0):
    _N[0] += 1
    pid = f"p{_N[0]}"
    contract = SimpleNamespace(
        cap_hit_m=cap, aav_m=cap, years_remaining=years, no_trade_clause=False, no_move_clause=False,
        modified_no_trade_teams=0, approved_trade_teams=[], clauses=None,
    )
    ident = SimpleNamespace(name=f"Player {pid}", age=age, position=SimpleNamespace(value=pos), shoots=shoots)
    return SimpleNamespace(
        id=pid, name=f"{pos} {pid}", identity=ident, contract=contract, cap_hit_m=cap,
        ovr=lambda o=ovr / 100.0: o, season_stats={"gp": 40, "pts": 20, "g": 8, "a": 12},
    )


def _roster(base: float, **over):
    """Dressed lineup (+spares): 4 C, 8 W, 3 LD, 3 RD, 2 G at ``base`` OVR, with overrides."""
    r = []
    r += [_p("C", over.get("c1", base + 4)), _p("C", over.get("c2", base + 2)), _p("C", base - 2), _p("C", base - 4)]
    r += [_p("LW", base + 3 - i) for i in range(4)] + [_p("RW", base + 2 - i) for i in range(4)]
    r += [_p("D", base + 2 - i, shoots="L") for i in range(3)] + [_p("D", base + 2 - i, shoots="R") for i in range(3)]
    r += [_p("G", over.get("g1", base + 3)), _p("G", base - 4)]
    return r


def _team(tid: str, roster, window: str = "emerging", ahl=None):
    return SimpleNamespace(
        team_id=tid, id=tid, abbr=tid, roster=list(roster), ahl_roster=list(ahl or []), owned_pick_ids=[],
        needs={}, gm_window=window, window=window, cap_pressure="moderate", cap_pressure_tier="moderate",
        retained_salary_records=[], prospect_pool=[],
    )


def _league(teams, standings=None):
    lg = SimpleNamespace(
        teams=teams, salary_cap_m=88.0, cap_floor_m=65.0, trade_history=[], games_per_team=82,
        _cpu_standings_snapshot=standings or {},
    )
    ensure_draft_pick_registry(lg, start_year=2025, years_ahead=4)
    return lg


# --- deadline -----------------------------------------------------------------


def test_deadline_phase_uses_real_day_and_peaks_on_deadline():
    from app.sim_engine.trades.trade_deadline import deadline_phase, days_to_deadline, is_post_deadline

    lg = SimpleNamespace(_trade_deadline_day_idx=176)
    assert deadline_phase(lg, 100, 215) == 0.0  # January: nothing yet
    week_out = deadline_phase(lg, 169, 215)
    assert deadline_phase(lg, 176, 215) == 1.0
    assert 0.6 < week_out < 1.0
    # Convex: the last week carries more of the ramp than the first week.
    assert deadline_phase(lg, 176, 215) - week_out > deadline_phase(lg, 148, 215) - deadline_phase(lg, 141, 215)
    assert days_to_deadline(lg, 176, 215) == 0 and not is_post_deadline(lg, 176, 215)
    assert is_post_deadline(lg, 177, 215)


def test_franchise_publishes_calendar_deadline():
    from services.franchise_sim import publish_trade_deadline_state

    cal = [{"iso": f"2027-02-{d:02d}", "tags": ["trade_deadline"] if d >= 25 else []} for d in range(20, 29)]
    cal += [{"iso": f"2027-03-{d:02d}", "tags": ["trade_deadline"] if d <= 10 else []} for d in range(1, 16)]
    league = SimpleNamespace()
    deadline_idx = next(i for i, d in enumerate(cal) if d["iso"] == "2027-03-10")
    session = SimpleNamespace(sim=SimpleNamespace(league=league), nhl_calendar=cal, calendar_cursor=deadline_idx,
                              phase="regular", nhl_regular_season_last_index=215)
    st = publish_trade_deadline_state(session)
    assert st["deadline_day_idx"] == deadline_idx and st["days_to_deadline"] == 0 and st["freeze_active"] is False
    session.calendar_cursor = deadline_idx + 1
    assert publish_trade_deadline_state(session)["freeze_active"] is True
    session.phase = "offseason"
    assert publish_trade_deadline_state(session)["freeze_active"] is False


def test_post_deadline_only_ahl_players_can_move():
    from app.sim_engine.trades.trade_evaluator import evaluate_trade_package

    nhl_a, nhl_b = _p("C", 75), _p("C", 75)
    ahl_a, ahl_b = _p("D", 62, age=24), _p("D", 62, age=24)
    a = _team("AAA", [nhl_a], ahl=[ahl_a])
    b = _team("BBB", [nhl_b], ahl=[ahl_b])
    lg = _league([a, b])
    tbi = {"AAA": a, "BBB": b}
    ctx = {"season_year": 2025, "calendar_cursor": 180, "regular_season_last_index": 215, "trade_deadline_passed": True}

    nhl_deal = {"BBB": [{"type": "player", "id": nhl_a.id, "team": "AAA"}], "AAA": [{"type": "player", "id": nhl_b.id, "team": "BBB"}]}
    ev = evaluate_trade_package(nhl_deal, league=lg, team_by_id=tbi, context=ctx, user_team_id="AAA")
    assert ev["can_execute"] is False
    assert any("trade deadline" in r.lower() for r in ev["rejection_reasons"])

    pick_deal = {"BBB": [{"type": "player", "id": ahl_a.id, "team": "AAA"}], "AAA": [{"type": "pick", "id": "2026-round7-BBB", "team": "BBB"}]}
    assert evaluate_trade_package(pick_deal, league=lg, team_by_id=tbi, context=ctx, user_team_id="AAA")["can_execute"] is False

    ahl_deal = {"BBB": [{"type": "player", "id": ahl_a.id, "team": "AAA"}], "AAA": [{"type": "player", "id": ahl_b.id, "team": "BBB"}]}
    ev = evaluate_trade_package(ahl_deal, league=lg, team_by_id=tbi, context=ctx, user_team_id="AAA")
    assert not any("trade deadline" in r.lower() for r in ev["rejection_reasons"])


def test_cpu_proposer_stops_after_deadline():
    from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades

    lg = _league([_team("AAA", _roster(76)), _team("BBB", _roster(74))])
    lg._trade_deadline_day_idx = 170
    lg._franchise_phase = "regular"
    assert propose_and_execute_cpu_trades(lg, max_executions=3, calendar_cursor=171, regular_season_last_index=215) == []


# --- assessment ---------------------------------------------------------------


def _standings(rows):
    """rows: tid → (pts_pct, gp)."""
    return {tid: {"gp": gp, "pts": int(pct * 2 * gp), "pts_pct": pct} for tid, (pct, gp) in rows.items()}


def _eight_team_league(**teams_override):
    ids = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    pct = dict(zip(ids, (0.70, 0.64, 0.58, 0.54, 0.50, 0.46, 0.40, 0.33)))
    teams = [teams_override.get(t) or _team(t, _roster(76)) for t in ids]
    return _league(teams, _standings({t: (pct[t], 60) for t in ids})), teams


def test_status_follows_standings_and_needs_are_relative():
    from app.sim_engine.trades.team_assessment import STATUS_CONTENDER, STATUS_TANK, assess_league

    weak_c = _team("T1", _roster(76, c1=70, c2=68))  # top team with a hole at centre
    lg, _ = _eight_team_league(T1=weak_c)
    a = assess_league(lg, calendar_cursor=150, deadline_phase=0.5, force=True)
    assert a["T1"].status == STATUS_CONTENDER
    assert a["T8"].status == STATUS_TANK
    assert a["T1"].needs["C_TOP2"] >= 0.8
    assert a["T2"].needs["C_TOP2"] < 0.35  # league-average centres = no real hole


def test_tank_team_lists_expiring_veterans_as_surplus():
    from app.sim_engine.trades.team_assessment import assess_league

    tank_roster = _roster(76)
    rental = _p("C", 81, age=31, years=1)
    tank_roster.append(rental)
    lg, _ = _eight_team_league(T8=_team("T8", tank_roster))
    a = assess_league(lg, calendar_cursor=150, deadline_phase=0.5, force=True)
    reasons = {s.reason for s in a["T8"].surplus if s.player is rental}
    assert reasons & {"expiring_veteran", "depth"}


def test_panic_buyer_flag_and_deadline_premium():
    from app.sim_engine.trades.needs_matcher import buyer_premium
    from app.sim_engine.trades.team_assessment import assess_league

    lg, _ = _eight_team_league()
    lg.cpu_franchise_profiles = {"T5": {"team_direction": "CONTENDER"}}  # expected contender, now just below the line
    a = assess_league(lg, calendar_cursor=170, deadline_phase=0.9, days_to_deadline=2, force=True)
    assert a["T5"].panic_buyer is True
    calm = buyer_premium(a["T1"], deadline_phase=0.1, days_to_deadline=40, rental=False)
    panic = buyer_premium(a["T5"], deadline_phase=0.9, days_to_deadline=0, rental=True)
    assert panic > calm and panic >= 0.35


# --- matching / execution ---------------------------------------------------


def test_contender_hole_matches_tank_rental_and_executes():
    from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades

    executed = []
    for seed in range(10):
        contender = _team("T1", _roster(76, c1=70, c2=68), window="contender")
        tank_roster = _roster(72)
        rental = _p("C", 82, age=31, years=1)
        tank_roster.append(rental)
        tank = _team("T8", tank_roster, window="rebuild")
        lg, _ = _eight_team_league(T1=contender, T8=tank)
        lg._trade_deadline_day_idx = 176
        lg.rng = random.Random(seed)
        out = propose_and_execute_cpu_trades(lg, max_executions=2, calendar_cursor=172, regular_season_last_index=215)
        executed.extend(t for t in out if rental.name in t["outgoing"])
    assert executed, "contender never landed the tank team's expiring centre"
    t = executed[0]
    assert t["to_team_id"] == "T1" and t["from_team_id"] == "T8"
    assert t["package_motive"] in ("tank_selloff", "rental_purchase", "panic_buy")
    assert any("Round" in x for x in t["incoming"])  # paid in futures


def test_balanced_league_makes_no_trades_for_the_sake_of_it():
    from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades

    ids = ["T1", "T2", "T3", "T4"]
    teams = [_team(t, _roster(75)) for t in ids]
    lg = _league(teams, _standings({t: (0.5, 30) for t in ids}))
    lg._trade_deadline_day_idx = 176
    total = 0
    for seed in range(5):
        lg.rng = random.Random(seed)
        total += len(propose_and_execute_cpu_trades(lg, max_executions=3, calendar_cursor=90, regular_season_last_index=215))
    assert total == 0
