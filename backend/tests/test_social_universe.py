"""Tests for IceHole reddit engagement and burner risk."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "SimEngine"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# storyline_engine imports storyline_copy; stub when absent in test env.
try:
    import app.sim_engine.franchise.storyline_copy  # noqa: F401
except ModuleNotFoundError:
    import types

    stub = types.ModuleType("app.sim_engine.franchise.storyline_copy")
    stub.classify_story_lane = lambda *a, **k: "default"
    stub.lane_flags = lambda *a, **k: {}
    stub.pick_line = lambda *a, **k: ""
    stub.story_ctx = lambda *a, **k: {}
    sys.modules["app.sim_engine.franchise.storyline_copy"] = stub


class _TeamState:
    def __init__(self) -> None:
        self.organizational_pressure = 0.5
        self.team_morale = 0.5

    def clamp(self) -> None:
        self.organizational_pressure = max(0.0, min(1.0, self.organizational_pressure))
        self.team_morale = max(0.0, min(1.0, self.team_morale))


class _FakeTeam:
    def __init__(self, name: str = "Canucks") -> None:
        self.name = name
        self.city = "Vancouver"
        self.state = _TeamState()


class _FakeSession:
    def __init__(self) -> None:
        self.user_team_id = "van"
        self.team_by_id = {"van": _FakeTeam()}
        self.reddit_engagement_pulse = []
        self.active_cause_storylines = []
        self.gm_burner_account = {
            "handle": "",
            "created_day": 0,
            "posts": [],
            "suspicion_score": 0.0,
            "exposed": False,
        }
        self.gm_burner_investigation = {}
        self.storyline_events = []
        self.social_posts = []
        self.reddit_threads = []
        self.calendar_idx = 100
        self.season_calendar_year = 2025


class SocialUniverseTests(unittest.TestCase):
    def test_high_ratio_positive_thread_moves_engagement(self):
        from app.sim_engine.franchise.storyline_engine import _apply_reddit_sentiment_to_engagement

        session = _FakeSession()
        before = session.team_by_id["van"].state.organizational_pressure
        thread = {
            "subreddit": "r/Canucks",
            "upvote_ratio": 0.92,
            "sentiment_score": 0.8,
        }
        _apply_reddit_sentiment_to_engagement(session, thread)
        after = session.team_by_id["van"].state.organizational_pressure
        self.assertLess(after, before)
        self.assertTrue(session.reddit_engagement_pulse)

    def test_low_ratio_controversial_thread_barely_moves_engagement(self):
        from app.sim_engine.franchise.storyline_engine import _apply_reddit_sentiment_to_engagement

        session = _FakeSession()
        before = session.team_by_id["van"].state.organizational_pressure
        thread = {
            "subreddit": "r/Canucks",
            "upvote_ratio": 0.58,
            "sentiment_score": -0.9,
        }
        _apply_reddit_sentiment_to_engagement(session, thread)
        after = session.team_by_id["van"].state.organizational_pressure
        self.assertLess(abs(after - before), 0.07)

    def test_compute_burner_risk_market_and_words(self):
        from app.sim_engine.franchise.burner_engine import compute_burner_risk

        session = _FakeSession()
        low = compute_burner_risk(session, "Team needs more effort tonight", "arizona")
        high = compute_burner_risk(session, "Fire the coach and trade everyone we are tanking", "montreal")
        self.assertGreaterEqual(low, 6)
        self.assertLessEqual(low, 94)
        self.assertGreater(high, low)
        self.assertGreaterEqual(high, 60)

    def test_contextual_risk_spikes_on_active_storyline(self):
        from app.sim_engine.franchise.burner_engine import compute_burner_risk

        session = _FakeSession()
        session.active_cause_storylines = [{"headline": "Trade rumors swirl around star", "player_id": "p1"}]
        generic = compute_burner_risk(session, "Good effort from the group", "default")
        targeted = compute_burner_risk(session, "Trade shop fire coach tank", "default")
        self.assertGreater(targeted, generic)


if __name__ == "__main__":
    unittest.main()
