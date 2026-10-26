"""PuckGPT edge-case dataset, run as real conversations on a real-NHL save.

Each group in tests/data/puckgpt_edge_cases.json and puckgpt_analytics.json is one conversation: the prime
turns run first, then every case must land on one of its expected intents.
Slow (starts a franchise), so it is marked like the other live-save tests.
"""

import json
from pathlib import Path

import pytest

_DIR = Path(__file__).parent / "data"
GROUPS = []
for _name in ("puckgpt_edge_cases.json", "puckgpt_analytics.json"):
    GROUPS += json.loads((_DIR / _name).read_text(encoding="utf-8"))["groups"]


@pytest.fixture(scope="module")
def session():
    from services.franchise_sim import start_franchise

    try:
        return start_franchise(team_query="Toronto", head_coach_name="Test", coach_archetype="balanced",
                               player_universe="real_nhl", seed=7)
    except Exception as exc:  # no network and no cached rosters
        pytest.skip(f"real-NHL save unavailable: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize("group", GROUPS, ids=[g["name"] for g in GROUPS])
def test_conversation_group(session, group):
    from services.puckgpt import engine

    engine._CTX.clear()
    for q in group["prime"]:
        engine.ask(session, q)
    misses = []
    for case in group["cases"]:
        out = engine.ask(session, case["q"])
        intent = (out.get("interpretation") or {}).get("intent")
        if intent not in case["expect"]:
            misses.append((case["q"], intent, case["expect"]))
        assert out.get("answer"), case["q"]
    assert not misses, misses
