"""Trade-demand calibration: depth players know their place, positions resolve, clubs don't
revolt en masse, other clubs' frustration stays off the feed, and CPU deal targets vary
between saves that start from the same rosters."""

from __future__ import annotations

import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.sim_engine.entities.player import Position  # noqa: E402

_N = [0]


def _skater(pos: str, ovr: float, *, gp: int = 30, toi_min: float = 0.0, character: int = 80):
    _N[0] += 1
    pid = f"dc{_N[0]}"
    p = types.SimpleNamespace(
        id=pid,
        player_id=pid,
        name=f"{pos} {pid}",
        retired=False,
        position=Position(pos),
        identity=types.SimpleNamespace(name=f"{pos} {pid}", age=27, position=Position(pos)),
        character=character,
        psych=types.SimpleNamespace(role_satisfaction=0.5, ice_time_satisfaction=0.5),
        season_stats={"gp": gp, "pts": 4, "toi_sec": int(gp * toi_min * 60)},
        contract=types.SimpleNamespace(clause="", years_remaining=2, term=2),
    )
    p.ovr = lambda o=ovr: o
    return p


def _team(tid: str, roster):
    return types.SimpleNamespace(team_id=tid, id=tid, abbr=tid, roster=list(roster))


def _session(teams, user_tid: str = "USR"):
    return types.SimpleNamespace(
        user_team_id=user_tid,
        trade_demands={},
        trade_stability_state={},
        pending_ui_popups=[],
        storyline_events=[],
        notifications=[],
        agent_relationships={},
        sim=types.SimpleNamespace(league=types.SimpleNamespace(teams=teams), rng=random.Random(3)),
        calendar_cursor=60,
    )


def _club(tid: str, *, extra_f_ovr: float = 70.0, extra_toi: float = 7.0):
    """12 regular forwards (80→69), 6 D, 2 G, plus one 13th forward."""
    fwd = [_skater("C", 80 - i, toi_min=18 - i * 0.6) for i in range(12)]
    d = [_skater("D", 82 - i, toi_min=22 - i) for i in range(6)]
    g = [_skater("G", 85, toi_min=60), _skater("G", 76, toi_min=60)]
    extra = _skater("LW", extra_f_ovr, gp=8, toi_min=extra_toi)
    return _team(tid, fwd + d + g + [extra]), extra, d, g


def test_position_enum_resolves_defence_and_goalie():
    from app.sim_engine.franchise.trade_stability_engine import _pos_code, deserved_depth_standing

    team, _extra, d, g = _club("AAA")
    assert str(d[0].position) == "Position.D"  # the trap: str() is not the code
    assert _pos_code(d[0]) == "D"
    assert _pos_code(g[0]) == "G"
    assert deserved_depth_standing(d[0], team)["group"] == "D"
    assert deserved_depth_standing(d[5], team)["rank"] == 3


def test_thirteenth_forward_well_below_regulars_is_not_role_frustrated():
    from app.sim_engine.franchise.trade_stability_engine import (
        deserved_depth_standing,
        infer_role_satisfaction_from_deployment,
    )

    team, extra, _d, _g = _club("AAA", extra_f_ovr=64.0, extra_toi=7.0)
    session = _session([team])
    standing = deserved_depth_standing(extra, team)
    assert standing["extra"] and not standing["bubble"]
    sat = infer_role_satisfaction_from_deployment(extra, team, session)
    assert sat is not None and sat >= 58.0


def test_bubble_extra_close_to_last_regular_still_has_a_gripe():
    from app.sim_engine.franchise.trade_stability_engine import (
        deserved_depth_standing,
        infer_role_satisfaction_from_deployment,
    )

    team, extra, _d, _g = _club("AAA", extra_f_ovr=68.0, extra_toi=6.0)
    session = _session([team])
    assert deserved_depth_standing(extra, team)["bubble"]
    sat_bubble = infer_role_satisfaction_from_deployment(extra, team, session)
    assert sat_bubble < 58.0


def test_team_wide_losing_alone_is_not_a_formal_demand():
    from app.sim_engine.franchise.trade_stability_engine import formal_demand_eligible

    row = {
        "trade_stability_score": 30.0,
        "stability_concern_days": 40,
        "pressures": {"winning": 16.0, "organizational": 9.0, "role": 2.0},
    }
    assert not formal_demand_eligible(row)
    row["pressures"]["role"] = 9.0
    assert formal_demand_eligible(row)


def test_one_open_demand_per_club_per_pass(monkeypatch):
    from services import trade_demand_engine as tde

    cpu_team, *_ = _club("CPU")
    session = _session([cpu_team])
    angry = {
        "trade_stability_score": 25.0,
        "escalation_level": 3,
        "stability_concern_days": 60,
        "pressures": {"role": 12.0, "management": 10.0},
        "character": 80,
    }
    monkeypatch.setattr(tde, "apply_daily_stability_update", lambda *a, **k: dict(angry))
    monkeypatch.setattr(tde, "assign_league_agents", lambda *a, **k: None)
    monkeypatch.setattr(tde, "get_trade_deadline_context", lambda *_a, **_k: {
        "new_demands_allowed": True, "crisis_timer_ticks": True, "past_deadline": False,
    })
    opened_for = []

    def _fake_open(session, player, team, **kw):
        opened_for.append(tde._team_key(team))
        row = {"status": "open", "team_id": tde._team_key(team), "opened_day": kw.get("calendar_idx")}
        tde.ensure_trade_demands(session)[tde._player_id(player)] = row
        return row

    monkeypatch.setattr(tde, "open_trade_demand", _fake_open)
    tde.process_trade_demand_day(session, 60)
    assert opened_for.count("CPU") == 1


def test_other_club_frustration_is_not_a_notification():
    from services import trade_demand_engine as tde

    cpu_team, extra, *_ = _club("CPU")
    user_team, u_extra, *_ = _club("USR")
    session = _session([cpu_team, user_team])
    row = {"trade_stability_score": 50.0, "escalation_level": 2, "pressures": {"role": 9.0}}
    tde._maybe_enqueue_stability_warning(session, extra, cpu_team, row, 60, "", random.Random(1))
    assert session.notifications == [] and session.pending_ui_popups == []
    tde._maybe_enqueue_stability_warning(session, u_extra, user_team, row, 60, "", random.Random(1))
    assert len(session.notifications) == 1 and len(session.pending_ui_popups) == 1


def test_gm_taste_is_stable_within_a_save_and_differs_between_saves():
    from app.sim_engine.trades.needs_matcher import GM_TASTE_SPREAD, gm_taste

    players = [_skater("C", 75) for _ in range(40)]
    save_a = types.SimpleNamespace(rng=random.Random(1))
    save_b = types.SimpleNamespace(rng=random.Random(2))
    a1 = [gm_taste(save_a, "T1", p) for p in players]
    a2 = [gm_taste(save_a, "T1", p) for p in players]
    b = [gm_taste(save_b, "T1", p) for p in players]
    assert a1 == a2
    assert a1 != b
    assert all(-GM_TASTE_SPREAD <= v <= GM_TASTE_SPREAD for v in a1 + b)
    ranked_a = sorted(range(40), key=lambda i: -a1[i])[:5]
    ranked_b = sorted(range(40), key=lambda i: -b[i])[:5]
    assert ranked_a != ranked_b
