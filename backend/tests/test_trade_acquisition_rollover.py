"""Regression: season rollover must not leave acquisition cooldowns permanently locked."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.trades.trade_rules import _player_recently_acquired  # noqa: E402


def test_recent_acquire_expires_across_season_rollover():
    p = SimpleNamespace(
        acquired_via_trade=True,
        last_acquired_day=180,
        last_acquired_date="2026-03-01",
        acquired_via_trade_season=2025,
    )
    # New season: cursor reset to 0, year bumped — must NOT still look "recent".
    assert _player_recently_acquired(
        p, {"season_year": 2026, "calendar_cursor": 0, "calendar_iso": "2026-09-15"}
    ) is False


def test_recent_acquire_still_blocks_same_season():
    p = SimpleNamespace(
        acquired_via_trade=True,
        last_acquired_day=100,
        last_acquired_date="2025-12-01",
        acquired_via_trade_season=2025,
    )
    assert _player_recently_acquired(
        p, {"season_year": 2025, "calendar_cursor": 102, "calendar_iso": "2025-12-03"}
    ) is True
    assert _player_recently_acquired(
        p, {"season_year": 2025, "calendar_cursor": 120, "calendar_iso": "2026-01-01"}
    ) is False
