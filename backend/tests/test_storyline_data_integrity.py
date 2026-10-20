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


# ----------------------------------------------------------------------------- effects applied once
def _effects_session():
    league = SimpleNamespace(teams=[], development_leagues=[])
    return SimpleNamespace(
        user_team_id="T1", season_calendar_year=2025, calendar_cursor=40, nhl_calendar=[{"iso": "2025-12-01"}] * 60,
        team_by_id={}, player_season_stats={}, sim=SimpleNamespace(league=league), universe_players={},
        storyline_events=[], _storyline_fired={}, standings=None, game_results=[],
        strength_map={}, pending_decisions=[], _standings_rank_by_team={}, _standings_rank_rev=0, _stats_revision=0,
    )


def test_data_pass_applies_effects_only_to_published_non_cause_stories(monkeypatch):
    applied = []
    monkeypatch.setattr(se, "_apply_storyline_effects", lambda s, t, p, fx: applied.append(fx))
    user_cause = {"headline": "u", "team_id": "T1", "cause_type": "PLAYER_LOW_PRODUCTION", "effects": {"team_morale": -3}}
    league_kept = {"headline": "k", "team_id": "T9", "cause_type": "WINNING_STREAK", "effects": {"team_morale": 1}}
    # The daily cap trims everything else; only these two publish.
    monkeypatch.setattr(se, "select_daily_data_stories", lambda gen, uid, league_cap=7: [user_cause, league_kept])
    se.run_data_storyline_pass(_effects_session(), calendar_idx=40, day_meta={"iso": "2025-12-01"}, rng=random.Random(1))
    # User cause story is left to _emit_cause_storyline; only the kept league story applies here.
    assert applied == [{"team_morale": 1}]


def test_trimmed_data_stories_apply_no_effects(monkeypatch):
    applied = []
    monkeypatch.setattr(se, "_apply_storyline_effects", lambda s, t, p, fx: applied.append(fx))
    monkeypatch.setattr(se, "select_daily_data_stories", lambda gen, uid, league_cap=7: [])
    se.run_data_storyline_pass(_effects_session(), calendar_idx=40, day_meta={"iso": "2025-12-01"}, rng=random.Random(1))
    assert applied == []


def test_cause_path_predicate_matches_routing():
    assert se._data_story_uses_cause_path({"team_id": "T1", "cause_type": "LOSING_STREAK"}, "T1")
    assert not se._data_story_uses_cause_path({"team_id": "T1", "cause_type": ""}, "T1")
    assert not se._data_story_uses_cause_path({"team_id": "T2", "cause_type": "LOSING_STREAK"}, "T1")
    assert not se._data_story_uses_cause_path({"team_id": "", "cause_type": "X"}, "")


# ----------------------------------------------------------------------------- promises end when the player leaves
def _promise_session(roster_ids):
    roster = [SimpleNamespace(id=pid, name=pid) for pid in roster_ids]
    team = SimpleNamespace(roster=roster, ahl_roster=[])
    return SimpleNamespace(
        user_team_id="T1", team_by_id={"T1": team}, season_calendar_year=2025, calendar_cursor=50,
        nhl_calendar=[{"iso": "2025-12-10"}] * 80,
        universe_players={"p1": {"player_id": "p1", "team_id": "T2", "state": {"gm_trust": 50}}},
        universe_promises=[{"id": "pr1", "player_id": "p1", "type": "role_opportunity", "status": "active",
                            "games_remaining": 3, "progress": 0, "required_progress": 5, "description": "More ice"}],
    )


def test_promise_voids_without_penalty_when_player_left_the_org():
    sess = _promise_session(["p9"])  # p1 traded away
    counts = se._u_tick_promises(sess)
    promise = sess.universe_promises[0]
    assert promise["status"] == "void" and counts["void"] == 1
    assert sess.universe_players["p1"]["state"]["gm_trust"] == 50  # no broken-promise penalty
    assert sess.universe_players["p1"]["open_promise_types"] == []


def test_promise_stays_active_while_player_is_on_the_club():
    sess = _promise_session(["p1"])
    se._u_tick_promises(sess)
    assert sess.universe_promises[0]["status"] == "active"


# ----------------------------------------------------------------------------- bulk sims advance timers by elapsed days
def test_readiness_modifiers_tick_by_elapsed_days():
    sess = SimpleNamespace(universe_readiness_modifiers={
        "p1": [{"id": "a", "days_remaining": 5}, {"id": "b", "days_remaining": 12}],
    })
    expired = se._u_tick_readiness_modifiers(sess, 7)  # one bulk pass covering a week
    assert expired == 1
    assert [r["days_remaining"] for r in sess.universe_readiness_modifiers["p1"]] == [5]
    se._u_tick_readiness_modifiers(sess)  # normal single day still steps by one
    assert sess.universe_readiness_modifiers["p1"][0]["days_remaining"] == 4


def test_availability_days_tick_by_elapsed_days():
    sess = SimpleNamespace(universe_player_availability={
        "p1": {"player_id": "p1", "status": "investigative_leave", "days_remaining": 6, "games_remaining": 0, "active": True},
        "p2": {"player_id": "p2", "status": "suspended", "days_remaining": 0, "games_remaining": 5, "active": True},
    })
    cleared = se._u_tick_player_availability_days(sess, 7)
    assert cleared == 1 and sess.universe_player_availability["p1"]["status"] == "available"
    assert sess.universe_player_availability["p2"]["active"]  # game-counted bans don't age by days


# ----------------------------------------------------------------------------- P1 fixes
def _life_session(vol=50.0):
    player = SimpleNamespace(id="p1", name="Life Guy")
    team = SimpleNamespace(roster=[player], ahl_roster=[])
    entity = {"player_id": "p1", "player_name": "Life Guy", "personality": {"volatility": vol},
              "life": {"last_minor_event_day": 0}, "state": {}}
    return SimpleNamespace(user_team_id="T1", team_by_id={"T1": team}, universe_players={"p1": entity},
                           calendar_cursor=40, nhl_calendar=[{"iso": "2025-12-01"}] * 60, season_calendar_year=2025)


class _FixedRng(random.Random):
    def __init__(self, value):
        super().__init__(0)
        self._v = value

    def random(self):
        return self._v


def test_life_event_odds_compound_over_a_bulk_span(monkeypatch):
    from app.sim_engine.franchise import storyline_procedural as sp

    rolled = []
    monkeypatch.setattr(sp, "evaluate_life_event_rules", lambda e, r, positive: rolled.append(1))
    # Daily chance for vol 50 on the user club is 0.07 × 1.25 = 8.75%; a 0.30 roll misses it…
    se._u_generate_minor_life_events(_life_session(), _FixedRng(0.30), days=1)
    assert rolled == []
    # …but 7 days compound to 1 − 0.9125^7 ≈ 47%, so the same roll now passes.
    se._u_generate_minor_life_events(_life_session(), _FixedRng(0.30), days=7)
    assert rolled == [1]


def test_storyline_agent_follows_the_contract_engine_agent():
    player = SimpleNamespace(id="p1", name="A", agent_profile={"id": "walsh"})
    sess = SimpleNamespace(team_by_id={"T1": SimpleNamespace(roster=[player], ahl_roster=[])},
                           agent_relationships={"p1": {"agent_id": "carter", "trust": 0.6}})
    agent = se._agent_for_player(sess, "p1", random.Random(1))
    assert agent["id"] == "walsh"
    assert sess.agent_relationships["p1"] == {"agent_id": "walsh", "trust": 0.6}  # trust kept, id repaired


def test_contract_satisfaction_uses_term_and_pay():
    sess = SimpleNamespace(sim=None)
    ent = {"personality": {"money_focus": 45, "ambition": 50}}

    def p(years, aav):
        return SimpleNamespace(id="x", contract={"years_remaining": years, "cap_hit_m": aav}, ovr=86, overall=86, age=27, position="C")

    long_fair = se._u_contract_satisfaction(sess, p(4, 7.0), ent, 2025)
    short_fair = se._u_contract_satisfaction(sess, p(1, 7.0), ent, 2025)
    long_cheap = se._u_contract_satisfaction(sess, p(4, 1.0), ent, 2025)
    assert long_fair["satisfaction"] > short_fair["satisfaction"]
    assert long_cheap["satisfaction"] < long_fair["satisfaction"]  # underpaid is no longer "satisfied"
    assert long_cheap["market_gap"] < 0


def test_small_samples_do_not_clear_the_evidence_gate():
    # OVR 86 forward, 6 GP: expected ≈ 3.2 points. One point is noise, zero is a story.
    exp_pts = se._expected_points_per_game(86, "C", 27, False) * 6
    assert se._points_z(1, exp_pts) > se.POINTS_Z_MINOR
    assert se._points_z(0, exp_pts) <= se.POINTS_Z_MINOR
    # .880 on 90 shots vs an expected .897 is well inside one standard error.
    assert se._save_pct_z(0.880, 0.897, 90) > se.SAVE_Z_MELTDOWN
    assert se._save_pct_z(0.850, 0.897, 600) <= se.SAVE_Z_MELTDOWN


def test_winning_promise_needs_a_real_core_upgrade(monkeypatch):
    promise = {"player_id": "p1", "type": "winning_commitment", "baseline": {"core_ovr": 80.0, "roster_ids": []}}
    monkeypatch.setattr(se, "_u_promise_snapshot", lambda s, pid: {"core_ovr": 80.3, "roster_ids": ["new80"]})
    se._u_promise_progress_tick(SimpleNamespace(), promise)
    assert int(promise.get("progress") or 0) == 0  # a lateral 80-OVR add no longer counts
    monkeypatch.setattr(se, "_u_promise_snapshot", lambda s, pid: {"core_ovr": 80.8, "roster_ids": []})
    se._u_promise_progress_tick(SimpleNamespace(), promise)
    assert promise["progress"] == 1


def test_team_core_ovr_uses_dressed_core():
    roster = [SimpleNamespace(id=f"f{i}", position="C", ovr=80) for i in range(14)]
    roster += [SimpleNamespace(id=f"d{i}", position="D", ovr=78) for i in range(7)]
    roster += [SimpleNamespace(id="g1", position="G", ovr=85), SimpleNamespace(id="g2", position="G", ovr=70)]
    core = se._team_core_ovr(SimpleNamespace(roster=roster))
    assert core == round((12 * 80 + 6 * 78 + 85) / 19, 2)


def test_conduct_incidents_target_risk_not_random_players():
    low = se._u_conduct_risk({"personality": {"character": 30, "volatility": 80}, "state": {"media_stress": 60}}, None)
    high = se._u_conduct_risk({"personality": {"character": 85, "volatility": 30}, "state": {"media_stress": 20}}, None)
    assert low > high * 10 and high >= 0.02


def test_first_private_trade_rejection_is_softer_than_repeat_exposure():
    def player():
        return SimpleNamespace(id="p1", character=50)

    sess = SimpleNamespace(universe_players={})
    quiet = se._trade_fallout_magnitude(sess, player(), 1)
    loud_player = player()
    se._ensure_player_storyline_state(loud_player)["trade_rumor_heat"] = 60
    loud = se._trade_fallout_magnitude(sess, loud_player, 1)
    assert quiet[2] <= 1 and quiet[4] <= 14  # OVR modifier and recovery days
    assert loud[2] >= quiet[2] and loud[1] > quiet[1]
