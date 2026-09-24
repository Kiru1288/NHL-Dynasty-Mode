"""
Storyline data integrity: missing data must stay missing (never invented), cap hit must be real,
and prospect / popup stories must carry the facts behind them.

Run: cd backend && python -m pytest tests/test_storyline_data_integrity.py -q
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "SimEngine", ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.sim_engine.franchise import social_copy_engine as sc  # noqa: E402
from app.sim_engine.franchise import storyline_engine as se  # noqa: E402
from app.sim_engine.franchise.social_templates import filter_templates, render_template  # noqa: E402


# ----------------------------------------------------------------------------- cap hit / contract
def test_cap_hit_reads_every_contract_shape():
    assert se._cap_hit_m(SimpleNamespace(contract={"cap_hit_m": 6.5})) == 6.5
    assert se._cap_hit_m(SimpleNamespace(contract={"aav": 4_500_000})) > 0  # dict + raw dollars
    assert se._cap_hit_m(SimpleNamespace(contract=SimpleNamespace(aav_m=3.2))) == 3.2
    assert se._cap_hit_m(SimpleNamespace(cap_hit_m=8.0)) == 8.0  # player-level field
    assert se._cap_hit_m(SimpleNamespace(contract=None)) == 0.0


def test_contract_years_left_is_none_when_unknown_not_99():
    assert se._contract_years_left(SimpleNamespace(contract={"years_remaining": 2})) == 2
    assert se._contract_years_left(SimpleNamespace(contract=SimpleNamespace(years=1))) == 1
    assert se._contract_years_left(SimpleNamespace(contract=None)) is None
    assert se._contract_years_left(None) is None


# ----------------------------------------------------------------------------- nothing invented
FABRICATED = (".900", "2.80", ".905", "Unknown player", "the club", "undisclosed", "$0M")


def test_evidence_context_invents_nothing():
    ctx = sc.build_evidence_context({"type": "goalie_meltdown", "headline": "x", "evidence": {}})
    assert set(ctx) == {"heat"}
    for bad in ("save_pct", "gaa", "ppg", "overall", "cap_hit", "age", "team", "name", "team_record"):
        assert bad not in ctx


def test_evidence_context_keeps_real_values_including_zero():
    ctx = sc.build_evidence_context(
        {"player_name": "A B", "team_name": "Buffalo", "evidence": {"goals": 0, "games_played": 12, "points": 3}}
    )
    assert ctx["goals"] == 0 and ctx["games_played"] == 12 and ctx["name"] == "A B"
    assert "cap_hit" not in ctx


def test_dynasty_context_has_no_random_facts():
    sess = SimpleNamespace(team_by_id={}, user_team_id="T1", universe_locker_rooms={}, universe_players={})
    a = sc.enrich_dynasty_context(sess, {"team_id": "T1", "player_name": "A B"}, {"name": "A B"}, random.Random(1))
    b = sc.enrich_dynasty_context(sess, {"team_id": "T1", "player_name": "A B"}, {"name": "A B"}, random.Random(2))
    assert a == b  # no randomness left
    for key in ("years_remaining", "prior_overall", "draft_pick", "rival_cap_hit", "rival_term", "cap_space", "morale", "salary_cap"):
        assert key not in a


def test_unresolved_placeholder_is_not_published():
    ctx = {"name": "A B"}
    assert render_template({"text": "{name} has {points} points"}, ctx) is None
    assert render_template({"text": "{name} is here"}, ctx) == "A B is here"
    # 0 is a real value, None / "" are missing
    assert render_template({"text": "{n} pts", "requires": ["n"]}, {"n": 0}) == "0 pts"
    assert render_template({"text": "{n} pts", "requires": ["n"]}, {"n": None}) is None
    assert filter_templates([{"text": "x", "requires": ["k"]}], ctx={"k": ""}) == []


def test_publish_gate_catches_the_old_invented_values():
    for text in (
        "Unknown player has a big night",
        "cap hit $0M for the winger",
        "he is None in the standings",
        "Rank -.",
        "Buffalo (—) rolling",
    ):
        assert sc._looks_like_broken_social_text(text), text
    assert not sc._looks_like_broken_social_text("Alex Tuch: 18 points in 20 GP for Buffalo")


def test_reporter_posts_never_contain_invented_values_or_mislabeled_stats():
    no_ev = {"headline": "Buffalo goalie struggling badly", "player_name": "", "team_name": "", "evidence": {}}
    real = {
        "headline": "h",
        "player_name": "Alex Tuch",
        "team_name": "Buffalo",
        "evidence": {"games_played": 20, "points": 18, "ppg": 0.9, "team_record": "10-8-2"},
    }
    reporter = {"id": "x", "outlet": "Wire"}
    for i in range(200):
        t1 = sc.compose_reporter_post(no_ev, reporter, random.Random(i))
        t2 = sc.compose_reporter_post(real, reporter, random.Random(i))
        for t in (t1, t2):
            assert not any(bad in t for bad in FABRICATED), t
            assert not re.search(r"\{\w+\}", t), t
            assert not re.search(r"cap hit|save pct|league rank", t), t  # stats we never supplied
        assert "Buffalo goalie struggling badly" in t1  # falls back to the real headline
        assert "Alex Tuch" in t2 or "Tuch, Alex" in t2 or "h" in t2


def test_lookup_does_not_crash_on_real_players_and_finds_real_data():
    from app.sim_engine.entities.player import Position
    from app.sim_engine.league_hierarchy_bootstrap import _spawn_player

    rng = random.Random(3)
    player = _spawn_player(
        rng, pos=Position.C, ovr_lo=0.6, ovr_hi=0.7, age_lo=24, age_hi=28,
        used_names=set(), league_players=[], pool_context="nhl",
    )
    team = SimpleNamespace(id="T1", name="Sabres", roster=[player])
    sess = SimpleNamespace(
        player_season_stats={}, team_by_id={"T1": team}, user_team_id="T1", standings=None, game_results=[],
    )
    out = sc._lookup_session_evidence(sess, {"player_id": str(player.id), "team_id": "T1"})  # used to raise TypeError
    assert out["overall"] > 0 and out["age"] > 0 and out["name"]


# ----------------------------------------------------------------------------- stories carry their facts
def test_evidence_becomes_trigger_reason_rows():
    rows = se._evidence_reason_lines(
        {"games_played": 12, "points_per_game": 0.4166, "team_record": "5-6-1", "streak": 3, "cap_hit": 7.25, "junk": 1}
    )
    labels = {r["label"]: r["value"] for r in rows}
    assert labels["Games played"] == "12"
    assert labels["Points per game"] == "0.42"
    assert labels["Cap hit"] == "$7.25M"
    assert labels["Streak"] == "3 straight"
    assert "junk" not in " ".join(labels)
    assert se._evidence_reason_lines({}) == [] and se._evidence_reason_lines(None) == []


def test_built_storyline_always_shows_its_numbers():
    row = se._build_storyline(
        rng=random.Random(1), session=SimpleNamespace(), stable_key="k|1", stype="rookie_breakout",
        category="development", severity="minor", priority="MEDIUM", tone="positive",
        headline="h", description="d", short_summary="s", cause="c",
        team_id="T1", team_name="Sabres", evidence={"games_played": 12, "points": 14}, effects={},
    )
    codes = {r["code"] for r in row["trigger_reasons"]}
    assert {"ev_games_played", "ev_points"} <= codes


def test_story_popup_keeps_evidence_and_identity():
    sess = SimpleNamespace(user_team_id="T1", pending_ui_popups=[], calendar_cursor=5)
    sl = {
        "storyline_id": "sl_1", "team_id": "T1", "headline": "Slump", "summary": "s", "player_id": "p1",
        "player_name": "A B", "player_position": "LW", "player_overall": 84.0, "team_name": "Sabres",
        "evidence": {"games_played": 12, "points": 4}, "cause": "Pace below expected.",
        "trigger_reasons": [{"code": "ev_points", "label": "Points", "value": "4"}],
    }
    se._u_enqueue_story_impact_popup(sess, sl, {"impact_lines": ["x"], "overall_delta": -2})
    pop = sess.pending_ui_popups[-1]
    assert pop["evidence"] == {"games_played": 12, "points": 4}
    assert pop["player_position"] == "LW" and pop["player_overall"] == 84.0
    assert pop["trigger_reasons"] and pop["cause"] == "Pace below expected."


# ----------------------------------------------------------------------------- prospect stories
def _prospect(pid, name, gp=30, pts=40):
    ident = SimpleNamespace(name=name, age=18, position=SimpleNamespace(value="C"))
    return SimpleNamespace(
        id=pid, name=name, identity=ident, age=18,
        _prospect_season_stats={"gp": gp, "goals": pts // 2, "assists": pts - pts // 2, "points": pts},
    )


def test_prospect_stock_stories_name_the_player_and_are_not_user_team_attributed():
    riser, faller = _prospect("pr1", "Ivan Rise"), _prospect("pr2", "Sam Fall", pts=10)
    league = SimpleNamespace(
        teams=[], development_leagues=[
            {"league_code": "CHL_OHL", "league_name": "OHL", "teams": [{"name": "Erie Otters", "players": [riser, faller]}]}
        ],
    )
    sess = SimpleNamespace(
        user_team_id="T1", season_calendar_year=2025, calendar_cursor=40, nhl_calendar=[{"iso": "2025-12-01"}] * 60,
        team_by_id={}, player_season_stats={}, sim=SimpleNamespace(league=league), universe_players={},
        storyline_events=[], _storyline_fired={}, standings=None, game_results=[],
        draft_rank_prev={"pr1": 5, "pr2": 30, "ghost": 3},
        draft_preseason_rank={"pr1": 20, "pr2": 12, "ghost": 20},
        strength_map={}, pending_decisions=[], _standings_rank_by_team={}, _standings_rank_rev=0, _stats_revision=0,
    )
    out = se.run_data_storyline_pass(sess, calendar_idx=40, day_meta={"iso": "2025-12-01"}, rng=random.Random(2))
    stories = {s["type"]: s for s in out["storylines"]}
    assert "prospect_rising" in stories and "prospect_falling" in stories
    up, down = stories["prospect_rising"], stories["prospect_falling"]
    assert "Ivan Rise" in up["headline"] and "No. 5" in up["headline"]
    assert "Sam Fall" in down["headline"]
    assert up["player_id"] == "pr1" and up["player_position"] == "C"
    assert up["team_id"] == "" and up["team_name"] == "Erie Otters"  # not the user's club
    assert up["cause_type"] == "PROSPECT_RISING" and down["cause_type"] == "PROSPECT_FALLING"
    assert up["evidence"]["points"] == 40 and up["evidence"]["league"] == "OHL"
    assert "ghost" not in " ".join(s["headline"] for s in out["storylines"])  # unnameable prospect not published
