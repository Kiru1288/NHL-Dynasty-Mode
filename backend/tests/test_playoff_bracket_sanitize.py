"""First-round sanitization and playoff→offseason continue handoff."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SimEngine"))
sys.path.insert(0, str(ROOT / "backend"))

from services.franchise_playoffs import sanitize_first_round_matchups  # noqa: E402


def test_sanitize_drops_later_rounds_and_duplicate_teams():
    rows = sanitize_first_round_matchups(
        [
            {"round_index": 1, "conference": "Western Conference", "team_high_id": "COL", "team_low_id": "NSH"},
            {"round_index": 1, "conference": "East", "team_high_id": "COL", "team_low_id": "NYR"},
            {"round_index": 2, "conference": "West", "team_high_id": "COL", "team_low_id": "LAK"},
            {"round_index": 1, "conference": "East", "team_high_id": "BOS", "team_low_id": "WSH"},
        ]
    )
    teams = [r["team_high_id"] for r in rows] + [r["team_low_id"] for r in rows]
    assert teams.count("COL") == 1
    assert all(r["round_index"] == 1 for r in rows)
    assert rows[0]["conference"] == "West"
    assert {r["conference"] for r in rows} <= {"West", "East", "League"}
    assert any(r["team_high_id"] == "BOS" for r in rows)


def test_start_live_does_not_rebuild_existing_bracket():
    from services.franchise_playoffs import start_live_playoffs

    live = {"started": True, "completed": True, "champion_id": "UTA", "series": [{"series_id": "keep"}]}
    session = SimpleNamespace(
        playoff_live=live,
        playoffs_simulated=False,
        phase="playoffs",
        season_phase="playoffs",
        champion_id="UTA",
        user_team_id="WSH",
        playoff_payload={"series": [{"round_index": 2, "team_high_id": "COL", "team_low_id": "BOS"}]},
    )
    out = start_live_playoffs(session)
    assert out is live
    assert out["series"][0]["series_id"] == "keep"
