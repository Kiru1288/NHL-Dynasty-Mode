"""Unit tests for published storyline coverage: traits, boxes, insiders payload."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "1")

from app.sim_engine.franchise.storyline_coverage import (  # noqa: E402
    coverage_payload_fields,
    ingest_game_box_storylines,
    personality_from_player,
)
from app.sim_engine.franchise.storyline_engine import _u_personality  # noqa: E402


class _Traits:
    loyalty = 0.9
    ego = 0.2
    volatility = 0.15
    competitiveness = 0.8
    leadership = 0.7
    ambition = 0.85
    coachability = 0.75
    media_comfort = 0.6
    introversion = 0.2
    family_priority = 0.7
    work_ethic = 0.8
    mental_toughness = 0.65
    money_focus = 0.3
    clutch_tendency = 0.55
    patience = 0.6


class _Psych:
    morale = 0.7
    role_satisfaction = 0.3
    ice_time_satisfaction = 0.25
    coach_trust = 0.4
    trust_in_management = 0.35
    trust_in_teammates = 0.8
    locker_room_fit = 0.2
    media_stress = 0.4
    confidence_level = 0.55
    contract_pressure = 0.8


class _Ident:
    name = "Alex Example"
    age = 26
    birth_city = "Winnipeg"
    birth_country = "Canada"
    draft_year = 2018
    draft_round = 1
    draft_pick = 8


class _Player:
    id = "p_test_1"
    name = "Alex Example"
    traits = _Traits()
    psych = _Psych()
    identity = _Ident()
    position = "C"
    overall = 82


class _Session:
    def __init__(self):
        self.user_team_id = "T1"
        self.season_calendar_year = 2026
        self.calendar_cursor = 20
        self.nhl_calendar = [{"iso": "2026-10-20"}] * 40
        self.storyline_events = []
        self.knowledge_graph = []
        self.universe_players = {
            "p_test_1": {
                "player_id": "p_test_1",
                "player_name": "Alex Example",
                "team_id": "T1",
                "active_roster": True,
                "overall": 82,
                "position": "C",
                "identity": {"name": "Alex Example", "birth_city": "Winnipeg"},
                "trusts": {"coach": 40, "gm": 35, "teammates": 80, "room": 20},
                "top_concerns": [{"id": "role", "label": "Role and ice time", "pressure": 62}],
                "memories": [{"summary": "Sat a healthy scratch"}],
                "personality_tags": ["Demanding"],
                "reputation_tags": ["Demanding"],
                "niche_abilities": [{"label": "Spark Plug"}],
            }
        }
        self._storyline_fired = {}
        self.player_recent_games = {}
        self.player_season_stats = {}
        self.team_by_id = {}
        self.decision_event_log = []
        self.social_posts = []


def test_personality_reads_real_traits_not_hash():
    player = _Player()
    first = personality_from_player(player)
    second = _u_personality(player, "p_test_1")
    assert first["loyalty"] == second["loyalty"] == 90.0
    assert first["ego"] == 20.0
    assert first["sociability"] == 80.0
    assert first["ambition"] == 85.0
    hashed = _u_personality(player, "totally-different-id")
    assert hashed["loyalty"] == first["loyalty"]


def test_hat_trick_from_scoring_events():
    session = _Session()
    box = {
        "home_id": "T1",
        "away_id": "T2",
        "home_goals": 4,
        "away_goals": 1,
        "overtime": False,
        "scoring_events": [
            {"scorer_id": "p_test_1", "scorer": "Alex Example", "for_team_id": "T1"},
            {"scorer_id": "p_test_1", "scorer": "Alex Example", "for_team_id": "T1"},
            {"scorer_id": "p_test_1", "scorer": "Alex Example", "for_team_id": "T1"},
            {"scorer_id": "p_other", "scorer": "Other", "for_team_id": "T1"},
        ],
    }
    emitted = ingest_game_box_storylines(session, box)
    assert emitted >= 1
    headlines = [str(s.get("headline") or "") for s in session.storyline_events]
    assert any("hat trick" in h.lower() for h in headlines)
    graph = session.knowledge_graph
    assert any(row.get("headline") for row in graph)
    log = session.player_recent_games.get("p_test_1") or []
    assert log and int(log[-1]["g"]) >= 3


def test_coverage_payload_has_dossiers_and_graph():
    session = _Session()
    session.knowledge_graph = [
        {
            "headline": "Camp restless",
            "knowledge_type": "claim",
            "public_knowledge_level": "rumour",
            "player_name": "Alex Example",
        }
    ]
    payload = coverage_payload_fields(session)
    assert payload["player_dossiers"]
    dossier = payload["player_dossiers"][0]
    assert dossier["player_name"] == "Alex Example"
    assert "wants" in dossier and "trusts" in dossier and "remembers" in dossier
    assert payload["insider_items"]
    assert payload["beat_writers"]
