# app/sim_engine/world/calendar.py
"""Schedule-derived stress: back-to-backs, rest gaps, simplified travel load."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

# GameSlot-like: .day, .home_id, .away_id
def build_team_play_days(schedule: List[Any]) -> Dict[str, Set[int]]:
    days: Dict[str, Set[int]] = defaultdict(set)
    for g in schedule:
        if getattr(g, "is_playoff", False):
            continue
        d = int(getattr(g, "day", 0))
        hid = str(getattr(g, "home_id", ""))
        aid = str(getattr(g, "away_id", ""))
        days[hid].add(d)
        days[aid].add(d)
    return dict(days)


def is_back_to_back(play_days: Set[int], day: int) -> bool:
    return (day - 1) in play_days


def rest_days_before(play_days: Set[int], day: int) -> int:
    """Consecutive off days immediately before this game day (0 if played yesterday)."""
    if (day - 1) in play_days:
        return 0
    k = 0
    t = day - 1
    while t not in play_days and t > 0:
        k += 1
        t -= 1
        if k > 14:
            break
    return k


def away_games_last_window(schedule: List[Any], team_id: str, day: int, window: int = 7) -> int:
    """Count away games in (day-window, day) for travel stress proxy."""
    tid = str(team_id)
    lo = max(1, day - window)
    n = 0
    for g in schedule:
        if getattr(g, "is_playoff", False):
            continue
        d = int(getattr(g, "day", 0))
        if lo < d < day and getattr(g, "away_id", None) == tid:
            n += 1
    return n


def travel_fatigue_bonus(team_id: str, day: int, schedule: List[Any]) -> float:
    """0..~0.15 extra fatigue factor from recent road load."""
    away = away_games_last_window(schedule, team_id, day, 7)
    return min(0.12, 0.025 * max(0, away - 2))


def summarize_schedule_stress(
    schedule: List[Any], team_ids: List[str]
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Per team: count of back-to-back second games; average rest-before-game (sample)."""
    play = build_team_play_days(schedule)
    b2b_count: Dict[str, int] = {t: 0 for t in team_ids}
    rest_accum: Dict[str, List[int]] = {t: [] for t in team_ids}
    for g in schedule:
        if getattr(g, "is_playoff", False):
            continue
        d = int(getattr(g, "day", 0))
        for tid in (str(getattr(g, "home_id", "")), str(getattr(g, "away_id", ""))):
            if tid not in play:
                continue
            if is_back_to_back(play[tid], d):
                b2b_count[tid] = b2b_count.get(tid, 0) + 1
            rest_accum.setdefault(tid, []).append(rest_days_before(play[tid], d))
    rest_avg = {
        t: (sum(rest_accum[t]) / len(rest_accum[t]) if rest_accum[t] else 0.0) for t in team_ids
    }
    return b2b_count, rest_avg
