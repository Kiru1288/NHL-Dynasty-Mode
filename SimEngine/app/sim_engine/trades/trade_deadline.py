"""
NHL trade deadline — one source of truth for the engine, CPU proposer and trade rules.

The deadline is a real calendar day (Mar 10; the calendar tags Feb 25–Mar 10 as the
deadline window). Franchise mode publishes the calendar index of that day on the league
(``_trade_deadline_day_idx``); standalone engine runs fall back to the old fraction-of-
season estimate.

After the deadline and through the playoffs, only AHL players may be traded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Days before the deadline that GM urgency starts building.
DEADLINE_RAMP_DAYS = 35
#: Final stretch where the market runs hot.
DEADLINE_WEEK_DAYS = 7

#: Phases where the deadline freeze applies (regular season after Mar 10, playoffs).
_FREEZE_PHASES = frozenset({"regular", "regular_season", "in_season", "playoffs", "postseason"})

POST_DEADLINE_BLOCK_REASON = "After the trade deadline only AHL players can be traded until the season ends."


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def deadline_day_index(league: Any, regular_season_last_index: int) -> int:
    """Calendar index of deadline day (inclusive — trades still allowed that day)."""
    published = getattr(league, "_trade_deadline_day_idx", None)
    if published is not None:
        return _safe_int(published, 0)
    max_d = max(40, int(regular_season_last_index or 192))
    md = max(40, int(max(120, max_d) * 0.56))
    return int(md + max(20.0, float(max_d) * 0.2))


def days_to_deadline(league: Any, day: int, regular_season_last_index: int) -> int:
    """Days remaining until deadline day; 0 on deadline day, negative after."""
    return deadline_day_index(league, regular_season_last_index) - int(day)


def deadline_phase(league: Any, day: int, regular_season_last_index: int) -> float:
    """0 → 1 urgency over the final ``DEADLINE_RAMP_DAYS``; 1.0 on deadline day.

    Convex so the last week carries most of the pressure (real deadline behaviour).
    """
    left = days_to_deadline(league, day, regular_season_last_index)
    if left < 0:
        return 1.0
    linear = max(0.0, min(1.0, 1.0 - float(left) / float(DEADLINE_RAMP_DAYS)))
    return round(linear ** 1.6, 4)


def is_post_deadline(league: Any, day: int, regular_season_last_index: int) -> bool:
    return days_to_deadline(league, day, regular_season_last_index) < 0


def post_deadline_freeze_active(context: Optional[Dict[str, Any]]) -> bool:
    """True when a trade package may only contain AHL players."""
    return bool((context or {}).get("trade_deadline_passed"))


def freeze_applies_to_phase(phase: str) -> bool:
    return str(phase or "").strip().lower() in _FREEZE_PHASES
