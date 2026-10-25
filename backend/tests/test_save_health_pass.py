"""Save-health checks for the long franchise bug list."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.sim_engine.engine import SimEngine


def test_scoring_uses_the_era_goals_cap():
    src = inspect.getsource(SimEngine._simulate_game_strength_ex)
    assert "_era_combined_gpg_cap" in src
    assert "3.70" in src


def test_story_modifier_cannot_stack_past_four():
    from app.sim_engine.franchise.storyline_conduct import apply_temporary_ovr_modifier

    player = SimpleNamespace(overall=92, ratings={"overall": 92})
    apply_temporary_ovr_modifier(player, source="story", amount=-3, reason="slump", storyline_id="a")
    apply_temporary_ovr_modifier(player, source="story", amount=-8, reason="worse", storyline_id="b")
    total = sum(int(m.get("amount") or 0) for m in player._franchise_ovr_modifiers if not m.get("resolved"))
    assert total >= -4


def test_stuck_ceiling_reopens_for_a_young_high_potential_player():
    from app.sim_engine.progression.development import assign_prospect_realization

    player = SimpleNamespace(
        age=21,
        overall=78,
        ovr=78,
        potential=90,
        ratings={},
        _dev_realization=0.15,
        _dev_real_ceiling=79.0,
    )
    assign_prospect_realization(player, __import__("random").Random(1))
    assert float(player._dev_real_ceiling) >= 86.0


def test_ahl_short_roster_is_reported():
    from services.roster_compliance import ahl_roster_shortfalls

    team = SimpleNamespace(id="T", roster=[])
    missing = ahl_roster_shortfalls(team)
    assert len(missing) == 3


def test_under_27_cap_stamp_stays_rfa():
    from services.contract_economy import resolved_rights_status

    player = SimpleNamespace(age=22, contract={"rights": "UFA", "ufa": True})
    assert resolved_rights_status(player) == "RFA"


def test_free_agency_waits_seven_days():
    from services.franchise_offseason import OWN_FA_MORATORIUM_DAYS

    assert OWN_FA_MORATORIUM_DAYS == 7


def test_high_character_kid_does_not_demand_a_trade():
    from app.sim_engine.franchise.trade_stability_engine import formal_demand_eligible

    assert formal_demand_eligible(
        {"trade_stability_score": 6, "character": 88, "age": 20, "personal_signal": True}
    ) is False


def test_high_potential_prospect_is_protected_from_a_pick_swap():
    from app.sim_engine.trades.cpu_trade_proposer import _seller_must_protect

    kid = SimpleNamespace(age=19, ovr=72, potential=88, drafted=True)
    assert _seller_must_protect(kid, window="contend", deadline=0.2) is True


def test_auto_scout_hides_the_exact_overall():
    from services.auto_scout import scout_report

    player = SimpleNamespace(
        overall=84,
        potential=92,
        position="LW",
        ratings={"shot": 90, "skating": 62, "hockey_iq": 88, "defense": 55},
    )
    report = scout_report(player, drafts_away=0)
    assert "84" not in report["summary"]
    assert report["tags"]
    assert report["confidence"] == "High"


def test_draft_age_player_is_signed_by_the_ai():
    from services.draft_rights_engine import should_cpu_auto_sign_elc

    def kid(**extra):
        base = dict(
            age=19,
            signed_status="unsigned",
            entry_level_contract_eligible=True,
            potential=82,
            overall=64,
            development_path="OHL",
            rights_status="",
        )
        base.update(extra)
        return SimpleNamespace(**base)

    team = SimpleNamespace(roster=[])
    ordinary, _reason = should_cpu_auto_sign_elc(kid(), team, season_year=2028)
    assert ordinary is False
    high_pot, pot_reason = should_cpu_auto_sign_elc(kid(potential=89), team, season_year=2028)
    assert high_pot is True and pot_reason == "draft_age_control"
    ready, ready_reason = should_cpu_auto_sign_elc(kid(overall=80), team, season_year=2028)
    assert ready is True and ready_reason == "prospect_readiness"
    almost, _almost = should_cpu_auto_sign_elc(kid(overall=76, potential=84), team, season_year=2028)
    assert almost is False
    college, _college_reason = should_cpu_auto_sign_elc(
        kid(potential=92, development_path="NCAA"), team, season_year=2028
    )
    assert college is False


def test_scout_confidence_rises_by_june():
    from services.auto_scout import attach_auto_scout

    player = SimpleNamespace(overall=80, potential=88, position="C", ratings={"shot": 86, "skating": 80, "hockey_iq": 70})
    fall = {"calendar_month": 10}
    june = {"calendar_month": 6}
    attach_auto_scout(player, fall)
    attach_auto_scout(player, june)
    assert fall["scout_confidence"] == "Low"
    assert june["scout_confidence"] == "High"
    assert fall["ovr_revealed"] is False
    assert june["ovr_range"]["high"] - june["ovr_range"]["low"] < fall["ovr_range"]["high"] - fall["ovr_range"]["low"]


def test_puckgpt_names_who_played_on_a_date():
    from services.sim_history import answer_save_query

    session = SimpleNamespace(
        season_calendar_year=2045,
        sim_history=[],
        social_posts=[],
        universe_players={},
        team_by_id={},
        user_team_id="ott",
        box_score_book={
            "order": ["2029"],
            "seasons": {
                "2029": {
                    "people": {
                        "p1": ["Auston Matthews", "C"],
                        "p2": ["Brady Tkachuk", "LW"],
                    },
                    "games": {
                        "g1": {
                            "id": "g1",
                            "iso": "2029-02-15",
                            "hn": "Toronto",
                            "an": "Ottawa",
                            "hg": 4,
                            "ag": 2,
                            "lines": {"p1": [], "p2": []},
                        }
                    },
                }
            },
        },
    )
    found = answer_save_query(session, "who played on feb 15 2029")
    blob = found["answer"] + " ".join(h.get("summary", "") for h in found["hits"])
    assert "Auston Matthews" in blob
    assert "Brady Tkachuk" in blob


def test_puckgpt_ranks_social_voices():
    from services.sim_history import answer_save_query

    session = SimpleNamespace(
        season_calendar_year=2032,
        sim_history=[],
        box_score_book={"seasons": {}, "order": []},
        team_by_id={},
        user_team_id="ott",
        universe_players={},
        social_posts=[
            {"player_name": "Nice Guy", "text": "Love this group.", "sentiment": 0.9, "calendar_iso": "2032-01-02", "kind": "player"},
            {"player_name": "Hothead", "text": "This team is a joke.", "sentiment": -0.9, "calendar_iso": "2032-01-03", "kind": "player"},
        ],
    )
    worst = answer_save_query(session, "which players said the worst things on social media")
    assert worst["hits"][0]["summary"].startswith("Hothead")


def test_puckgpt_finds_a_logged_score():
    from services.sim_history import answer_save_query, record_final_score

    session = SimpleNamespace(season_calendar_year=2029, sim_history=[])
    record_final_score(session, {"home_name": "Ottawa", "away_name": "Toronto", "home_goals": 4, "away_goals": 1})
    found = answer_save_query(session, "Ottawa Toronto")
    assert "Ottawa" in found["answer"] or any("Ottawa" in str(h.get("summary")) for h in found["hits"])
