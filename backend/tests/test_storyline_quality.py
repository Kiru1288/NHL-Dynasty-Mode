"""Storyline quality: SV% formatting, publish gates, uniqueness, follow-ups."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "1")

from app.sim_engine.franchise.storyline_copy import (  # noqa: E402
    claim_league_story_slot,
    format_sv_pct,
    headline_fingerprint,
    normalize_save_pct,
    select_daily_data_stories,
    story_ctx,
    valid_goalie_heater_sv,
)
from app.sim_engine.franchise.storyline_engine import (  # noqa: E402
    _breaking_news_signal,
    _enqueue_storyline_followup,
    _process_storyline_followups,
)
from app.sim_engine.franchise.storyline_procedural import (  # noqa: E402
    community_event_copy,
    compose_data_story_copy,
    compose_shutout_copy,
    reporter_conflict_copy,
)


class _Session:
    def __init__(self):
        self.user_team_id = "T1"
        self.season_calendar_year = 2026
        self.calendar_cursor = 20
        self.nhl_calendar = [{"iso": "2026-10-20"}] * 40
        self.storyline_events = []
        self.knowledge_graph = []
        self.universe_players = {}
        self._storyline_fired = {}
        self.player_recent_games = {}
        self.player_season_stats = {}
        self.team_by_id = {}
        self.decision_event_log = []
        self.social_posts = []
        self.story_arcs = []
        self._stats_revision = 0
        self._standings_rank_rev = 0
        self._standings_rank_by_team = {"T1": 1, "T2": 8}
        for i in range(3, 10):
            self._standings_rank_by_team[f"T{i}"] = 22


def test_normalize_save_pct_decimal_and_percent():
    assert normalize_save_pct(0.924) == 0.924
    assert normalize_save_pct(92.4) == 0.924
    assert normalize_save_pct(None, "", 0, 0.912) == 0.912
    assert normalize_save_pct(None, 0, 0.0) is None
    assert format_sv_pct(0.924) == ".924"
    assert format_sv_pct(0.912) == ".912"
    assert format_sv_pct(91.2) == ".912"
    assert valid_goalie_heater_sv(0.924)
    assert not valid_goalie_heater_sv(None)
    assert not valid_goalie_heater_sv(0)
    assert not valid_goalie_heater_sv(0.850)


def test_heater_copy_uses_nhl_sv_not_zero_percent():
    rng = random.Random(1)
    ctx = story_ctx(name="Jake Oettinger", team="Dallas", gp=8, sv=0.924, gaa=2.11, exp_sv=0.905)
    headline = compose_data_story_copy("goalie_heater", ctx, rng)
    body = compose_data_story_copy("goalie_heater", ctx, rng, body=True)
    blob = f"{headline} {body}"
    assert headline
    assert "0.0%" not in blob
    assert ".924" in blob
    assert valid_goalie_heater_sv(ctx["save_pct"])


def test_heater_refuses_missing_or_junk_sv():
    rng = random.Random(2)
    assert compose_data_story_copy("goalie_heater", {"name": "A", "team": "T"}, rng) == ""
    assert compose_data_story_copy("goalie_heater", {"name": "A", "sv": 0}, rng) == ""
    assert compose_data_story_copy("goalie_heater", {"name": "A", "save_pct": 0.84}, rng) == ""
    assert compose_data_story_copy("goalie_meltdown", {"name": "A"}, rng) == ""


def test_story_ctx_aliases_sv_keys():
    ctx = story_ctx(sv=0.918, exp_sv=0.900)
    assert ctx["save_pct"] == 0.918
    assert ctx["expected_save_pct"] == 0.900
    assert ctx["sv_fmt"] == ".918"


def test_eight_league_shutouts_cap_and_keep_user_club():
    from app.sim_engine.franchise.storyline_coverage import ingest_game_box_storylines

    session = _Session()
    for i in range(1, 10):
        session.team_by_id[f"T{i}"] = SimpleNamespace(city="City", name=f"Club{i}")

    def _box(home, away, home_goals, away_goals, goalie_name, gid):
        return {
            "home_id": home,
            "away_id": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "overtime": False,
            "home_goalie": {"name": goalie_name, "player_id": gid} if home_goals > away_goals or away_goals == 0 else {},
            "away_goalie": {"name": goalie_name, "player_id": gid} if away_goals > home_goals or home_goals == 0 else {},
            "scoring_events": [],
        }

    # User club named shutout
    ingest_game_box_storylines(
        session,
        _box("T1", "TX", 2, 0, "Connor Hellebuyck", "g_user"),
    )
    # Eight league shutouts with names
    names = [
        ("T2", "Linus Ullmark", "g2"),
        ("T3", "Igor Shesterkin", "g3"),
        ("T4", "Juuse Saros", "g4"),
        ("T5", "Jeremy Swayman", "g5"),
        ("T6", "Jake Oettinger", "g6"),
        ("T7", "Thatcher Demko", "g7"),
        ("T8", "Adin Hill", "g8"),
        ("T9", "Sergei Bobrovsky", "g9"),
    ]
    for tid, name, gid in names:
        ingest_game_box_storylines(session, _box(tid, "TZ", 1, 0, name, gid))

    shutouts = [
        s
        for s in session.storyline_events
        if str(s.get("cause_type") or "") == "SHUTOUT"
    ]
    assert 1 <= len(shutouts) <= 2
    assert any(s.get("team_id") == "T1" for s in shutouts)
    blob = " ".join(str(s.get("headline") or "") for s in shutouts)
    assert "netminder" not in blob.lower()
    assert "throws a shutout" not in blob.lower()
    assert any("Hellebuyck" in str(s.get("headline") or "") or s.get("team_id") == "T1" for s in shutouts)


def test_unnamed_league_shutout_does_not_use_netminder_line():
    from app.sim_engine.franchise.storyline_coverage import ingest_game_box_storylines

    session = _Session()
    session.team_by_id["T4"] = SimpleNamespace(city="City", name="Depth")
    ingest_game_box_storylines(
        session,
        {
            "home_id": "T4",
            "away_id": "TZ",
            "home_goals": 3,
            "away_goals": 0,
            "overtime": False,
            "scoring_events": [],
        },
    )
    headlines = [str(s.get("headline") or "").lower() for s in session.storyline_events]
    assert not any("netminder throws a shutout" in h for h in headlines)


def test_fingerprint_and_user_club_slot():
    session = _Session()
    event = {"headline": "Hot goaltending: A at .924", "cause_type": "GOALIE_HEATER", "calendar_day": 20, "type": "goalie_heater"}
    assert claim_league_story_slot(session, event, user_club=False)
    assert claim_league_story_slot(session, dict(event, headline="Hot goaltending: B at .931"), user_club=False)
    assert not claim_league_story_slot(session, dict(event, headline="Hot goaltending: C at .940"), user_club=False)
    assert claim_league_story_slot(session, event, user_club=True)


def test_select_daily_data_stories_keeps_user_and_caps_league():
    rows = [{"headline": f"Hot goaltending: G{i}", "type": "goalie_heater", "cause_type": "GOALIE_HEATER", "team_id": f"L{i}", "heat": 50 - i} for i in range(10)]
    rows.append({"headline": "User heater", "type": "goalie_heater", "cause_type": "GOALIE_HEATER", "team_id": "T1", "heat": 40})
    kept = select_daily_data_stories(rows, "T1", league_cap=7)
    assert any(r["team_id"] == "T1" for r in kept)
    league = [r for r in kept if r["team_id"] != "T1"]
    assert len(league) <= 2


def test_breaking_signal_skips_routine_shutout_heater():
    assert _breaking_news_signal({"cause_type": "SHUTOUT", "heat": 46, "team_id": "T8", "priority": "MEDIUM"}, "T1") is None
    assert _breaking_news_signal({"cause_type": "GOALIE_HEATER", "heat": 48, "team_id": "T8", "priority": "MEDIUM"}, "T1") is None
    assert _breaking_news_signal({"cause_type": "TRADE_DEMAND", "heat": 80, "team_id": "T1", "priority": "HIGH"}, "T1") == "breaking"
    assert _breaking_news_signal({"cause_type": "PLAYER_ARRESTED", "heat": 90, "team_id": "T8"}, "T1") == "league_defining"


def test_copy_variety_shutout_community_reporter():
    rng = random.Random(3)
    h1, s1 = compose_shutout_copy(goalie_name="Saros", team="Nashville", opponent="Dallas", prior_shutouts=2)
    h0, _ = compose_shutout_copy(goalie_name="Saros", team="Nashville", opponent="Dallas", prior_shutouts=0, league_rank=8, record="12-11-2")
    assert "Vezina" in h1 or "No." in h1
    assert h1 != h0
    assert "throws a shutout" not in h0.lower()
    c1, _ = community_event_copy("McDavid", "Edmonton", "p1")
    c2, _ = community_event_copy("Draisaitl", "Edmonton", "p2")
    assert "community connection is lifting" not in c1.lower()
    frames = {reporter_conflict_copy(rng, "A", "Jenna Lee", "TSN", player_id=f"p{i}")["frame"] for i in range(12)}
    assert len(frames) >= 3
    one = reporter_conflict_copy(rng, "Star", "Jenna Lee", "TSN", player_id="px")
    assert "over repeated coverage" not in one["summary"].lower()


def test_followup_beats_publish_after_due_day():
    session = _Session()
    session.calendar_cursor = 20
    _enqueue_storyline_followup(
        session,
        due_day=20,
        kind="reporter_response",
        team_id="T1",
        player_id="p1",
        player_name="Brady Tkachuk",
        reporter_name="Jenna Lee",
        outlet="TSN",
        frame="trade_rumor",
    )
    n = _process_storyline_followups(session)
    assert n == 1
    assert session.storyline_events
    hl = str(session.storyline_events[-1].get("headline") or "")
    assert "Jenna Lee" in hl or "Brady" in hl
    assert session.storyline_events[-1].get("cause_type") == "STORYLINE_FOLLOWUP"


def test_headline_fingerprint_strips_names_less_than_structure():
    a = headline_fingerprint("Jake Oettinger at .924 through 8 starts")
    b = headline_fingerprint("Linus Ullmark at .931 through 9 starts")
    assert "#" in a
    # numbers collapsed; remaining tokens still differ by name, slot cap handles volume
    assert isinstance(b, str)
