"""Aging, line rank, and the one displayed overall."""

from __future__ import annotations

import random
from types import SimpleNamespace

from app.sim_engine.engine import (
    _aging_v3_player_modifiers,
    _career_aging_decline_try_v3,
    apply_league_ovr_soft_regression_if_needed,
    regression_check,
)
from app.sim_engine.franchise.storyline_conduct import (
    engine_ovr99,
    format_ovr_modifier_badge,
    get_base_ovr_display,
)
from app.sim_engine.franchise.storyline_engine import record_lineup_save_decisions
from app.sim_engine.systems.chemistry import usage_satisfaction_score
from services.franchise_sim import _apply_storyline_choice_action


def _skater(age: int, ovr99: float, **extra):
    player = SimpleNamespace(
        age=age,
        ratings={"off_shooting": ovr99},
        retired=False,
        psych=SimpleNamespace(morale=0.6, confidence=0.6, role_satisfaction=0.7),
        rating_history=[],
        position="C",
        **extra,
    )
    player._ovr = float(ovr99) / 100.0

    def ovr():
        return player._ovr

    player.ovr = ovr
    return player


def test_engine_overall_is_the_number_on_the_card():
    player = _skater(31, 86)
    assert get_base_ovr_display(player) == engine_ovr99(player) == 85 or get_base_ovr_display(player) == 86
    # 86/100 * 99 rounds to 85. The card uses that engine scale, not a ratings blend.
    assert get_base_ovr_display(player) == engine_ovr99(player)


def test_modifier_is_a_badge_not_the_overall():
    player = _skater(30, 84)
    player._franchise_ovr_modifiers = [
        {"source": "conduct", "cause_type": "CONDUCT_INCIDENT", "amount": -3, "reason": "suspension", "resolved": False}
    ]
    assert get_base_ovr_display(player) == engine_ovr99(player)
    assert format_ovr_modifier_badge(player) == "−3 conduct"


def test_elite_decline_is_smaller_than_depth():
    elite_losses = []
    depth_losses = []
    for seed in range(40):
        rng = random.Random(seed)
        elite = _skater(32, 91)
        depth = _skater(32, 78)
        elite_loss = _career_aging_decline_try_v3(elite, rng, None)
        depth_loss = _career_aging_decline_try_v3(depth, rng, None)
        if elite_loss:
            elite_losses.append(elite_loss)
            assert 1.0 <= elite_loss <= 1.8 * 1.30 + 1e-6
            assert "Age 32 decline" in elite._pending_aging_reason
        if depth_loss:
            depth_losses.append(depth_loss)
            assert 1.5 <= depth_loss <= 3.0 + 1e-6
    assert len(elite_losses) >= 8
    assert len(depth_losses) >= 8
    assert sum(elite_losses) / len(elite_losses) < sum(depth_losses) / len(depth_losses)


def test_modifiers_stay_inside_the_band_and_name_the_reason():
    player = _skater(33, 80)
    player.ratings = {"off_shooting": 80, "dev_work_ethic": 88, "phy_stamina": 86}
    player._trade_demand_active = True
    player.psych.morale = 0.2
    player.season_stats = {"gp": 80, "toi_sec": 80 * 20 * 60}
    mult, reasons = _aging_v3_player_modifiers(player)
    assert 0.75 <= mult <= 1.30
    assert "work ethic" in reasons
    assert "conditioning" in reasons
    assert "low morale" in reasons
    assert "trade demand" in reasons
    assert "heavy minutes" in reasons
    noted = None
    for seed in range(30):
        noted = _career_aging_decline_try_v3(player, random.Random(seed), None)
        if noted:
            break
    assert noted
    assert player._pending_aging_reason.startswith("Age 33 decline")
    assert "low morale" in player._pending_aging_reason
    fresh = _skater(33, 80)
    fresh.psych.morale = 0.2
    note = regression_check(fresh, random.Random(4))
    if note:
        assert fresh.rating_history[-1]["reason"].startswith("Age 33 decline")
        assert "low morale" in fresh.rating_history[-1]["reason"]


def test_soft_regression_records_the_opening_average_and_does_not_hit_85s():
    players = [_skater(30, 86) for _ in range(8)]
    before = [dict(p.ratings) for p in players]
    team = SimpleNamespace(roster=players)
    apply_league_ovr_soft_regression_if_needed([team], random.Random(1), avg_trigger=78.0)
    assert abs(team._league_ovr_open_baseline - 86.0) < 0.2
    assert [dict(p.ratings) for p in players] == before
    apply_league_ovr_soft_regression_if_needed([team], random.Random(2), avg_trigger=78.0)
    assert [dict(p.ratings) for p in players] == before


def test_reduce_ice_drops_one_line_and_keeps_zero_as_line_one():
    star = SimpleNamespace(id="a", _deployed_line_rank=0, retired=False, name="Star")
    other = SimpleNamespace(id="d", retired=False, name="Other")
    team = SimpleNamespace(roster=[star, other])
    session = SimpleNamespace(
        user_team_id="T",
        team_by_id={"T": team},
        lines={
            "even_strength": {
                "lines": {
                    "forwards": [
                        {"id": "f1", "slots": {"C": "a", "LW": "b", "RW": "c"}},
                        {"id": "f2", "slots": {"C": "d", "LW": "e", "RW": "f"}},
                    ],
                    "defense": [],
                }
            }
        },
    )
    out = _apply_storyline_choice_action(session, {"meta": {"player_id": "a"}}, {"id": "reduce_toi"})
    assert out["ice_time_moved"] == 1
    assert star._deployed_line_rank == 1
    slots = session.lines["even_strength"]["lines"]["forwards"]
    assert slots[1]["slots"]["C"] == "a"
    assert slots[0]["slots"]["C"] == "d"


def test_defence_drop_uses_the_pair_field():
    dman = SimpleNamespace(id="a", _deployed_pair_rank=0, _deployed_line_rank=None, retired=False, name="D")
    other = SimpleNamespace(id="b", retired=False, name="Other")
    team = SimpleNamespace(roster=[dman, other])
    session = SimpleNamespace(
        user_team_id="T",
        team_by_id={"T": team},
        lines={
            "even_strength": {
                "lines": {
                    "forwards": [],
                    "defense": [
                        {"id": "d1", "slots": {"LD": "a", "RD": "x"}},
                        {"id": "d2", "slots": {"LD": "b", "RD": "y"}},
                    ],
                }
            }
        },
    )
    _apply_storyline_choice_action(session, {"meta": {"player_id": "a"}}, {"id": "reduce_toi"})
    assert dman._deployed_pair_rank == 1
    assert dman._deployed_line_rank is None


def test_first_line_save_without_a_game_does_not_invent_a_drop():
    skater = SimpleNamespace(id="a", name="A", retired=False, position="C", ratings={})
    team = SimpleNamespace(roster=[skater])
    session = SimpleNamespace(user_team_id="T", team_by_id={"T": team})
    lines = {"forwards": [{"id": "f3", "slots": {"C": "a", "LW": "b", "RW": "c"}}], "defense": [], "goalies": []}
    stories = record_lineup_save_decisions(session, new_lines=lines, previous_lines=None)
    assert stories == []
    assert skater._deployed_line_rank == 2


def test_one_line_for_one_game_does_not_complain():
    player = _skater(28, 90)
    player._deployed_line_rank = 1
    player._deserved_line_idx = 0
    player._line_gap_streak = 1
    player.psych.role_satisfaction = 0.8
    quiet = usage_satisfaction_score(player)
    player._line_gap_streak = 3
    loud = usage_satisfaction_score(player)
    assert quiet > loud


def test_five_season_three_seed_aging_does_not_run_away():
    """Young players still gain. Veterans decline. The fixed-78 pass does not stack on top."""
    shifts = []
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        roster = []
        for i in range(180):
            age = 19 + (i % 20)
            roster.append(_skater(age, 76 + rng.random() * 12))
        team = SimpleNamespace(roster=roster)
        start = sum(p.ovr() * 100 for p in roster) / len(roster)
        for _season in range(5):
            apply_league_ovr_soft_regression_if_needed([team], rng)
            for player in roster:
                if player.age < 27 and player.ovr() * 100 < 90:
                    player._ovr += 1.6 / 100.0
                elif player.age < 29 and player.ovr() * 100 < 88:
                    player._ovr += 0.35 / 100.0
                loss = _career_aging_decline_try_v3(player, rng, None)
                if loss:
                    player._ovr = max(0.45, player._ovr - loss / 100.0)
                player.age += 1
        end = sum(p.ovr() * 100 for p in roster) / len(roster)
        shifts.append(round(end - start, 2))
    assert max(shifts) < 6.0, shifts
    assert min(shifts) > -12.0, shifts
