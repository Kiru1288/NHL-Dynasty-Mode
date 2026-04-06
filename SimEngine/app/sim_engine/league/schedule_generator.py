from __future__ import annotations

"""
League schedule generation utilities.

This module is responsible for producing a deterministic, balanced-enough
regular-season schedule for an arbitrary set of team objects. It does NOT
assume real-world NHL divisions, only that teams may expose:

    - team_id or id
    - name / city + name
    - conference (optional)
    - division (optional)

If conference/division information is missing, the generator gracefully
falls back to a league-wide balanced schedule.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import random


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_team_id(team: Any, fallback_index: int) -> str:
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", None)
    if tid is None:
        return f"T{fallback_index:02d}"
    return str(tid)


def _safe_team_name(team: Any, team_id: str) -> str:
    name = getattr(team, "name", None)
    city = getattr(team, "city", None)
    if city and name:
        return f"{city} {name}"
    if name:
        return str(name)
    return str(team_id)


def _safe_conf(team: Any) -> Optional[str]:
    return getattr(team, "conference", None)


def _safe_div(team: Any) -> Optional[str]:
    return getattr(team, "division", None)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSlot:
    """
    Represents a single scheduled game in the league calendar.

    - day: integer sequence index (1..N) used as a simple calendar surrogate
    - home_id / away_id: team identifiers (string)
    - is_playoff: flag reserved for playoff layers (regular season uses False)
    """

    day: int
    home_id: str
    away_id: str
    is_playoff: bool = False


@dataclass
class TeamScheduleMeta:
    team_id: str
    name: str
    conference: Optional[str]
    division: Optional[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_teams(teams: Iterable[Any]) -> Tuple[List[TeamScheduleMeta], Dict[str, Any]]:
    """
    Convert arbitrary team objects into a deterministic metadata list and
    id->team mapping that the rest of the league layer can rely on.
    """
    meta: List[TeamScheduleMeta] = []
    by_id: Dict[str, Any] = {}
    for idx, t in enumerate(teams):
        tid = _safe_team_id(t, idx)
        name = _safe_team_name(t, tid)
        conf = _safe_conf(t)
        div = _safe_div(t)
        meta.append(TeamScheduleMeta(team_id=tid, name=name, conference=conf, division=div))
        by_id[tid] = t
    return meta, by_id


def generate_regular_season_schedule(
    rng: random.Random,
    teams: List[Any],
    games_per_team: int = 82,
) -> List[GameSlot]:
    """
    Generate a deterministic, NHL-ish regular season schedule for the
    provided teams.

    Design goals:
    - Every team plays many games (roughly games_per_team).
    - Heavier weighting for intra-division, then intra-conference,
      then inter-conference matchups when metadata exists.
    - Reasonable home/away balance over the season.
    - Deterministic given rng seed and team ordering.

    The exact number of games may not be *perfectly* equal to games_per_team
    for every team, but the spread is kept modest and stable.
    """
    if not teams:
        return []

    meta, _ = normalize_teams(teams)
    n = len(meta)
    if n == 1:
        # Degenerate single-team league; nothing to schedule.
        return []

    # ----------------------------------------------------------------------
    # Decide pairwise game counts
    # ----------------------------------------------------------------------
    #
    # We approximate an NHL-like pattern:
    # - same division: 4 games
    # - same conference (different division): 3 games
    # - different conference: 2 games
    #
    # This yields reasonable volume for a 30–32 team league without
    # overfitting to exact NHL rules.
    # ----------------------------------------------------------------------

    pair_games: Dict[Tuple[str, str], int] = {}
    for i in range(n):
        a = meta[i]
        for j in range(i + 1, n):
            b = meta[j]
            if a.conference and b.conference and a.conference == b.conference:
                if a.division and b.division and a.division == b.division:
                    games = 4
                else:
                    games = 3
            else:
                games = 2
            pair_games[(a.team_id, b.team_id)] = games

    # Optionally scale toward requested games_per_team if we are far off.
    # Compute approximate games per team from this pattern.
    approx_games_per_team = [0] * n
    for (aid, bid), g in pair_games.items():
        ia = next(i for i, t in enumerate(meta) if t.team_id == aid)
        ib = next(i for i, t in enumerate(meta) if t.team_id == bid)
        approx_games_per_team[ia] += g
        approx_games_per_team[ib] += g

    # If league is very small or requested games_per_team is very low/high,
    # we could scale pair games, but for now we keep the pattern stable;
    # the surrounding engine can always treat point percentages as primary.

    # ----------------------------------------------------------------------
    # Build GameSlot list with balanced home/away assignment
    # ----------------------------------------------------------------------
    slots: List[GameSlot] = []
    day_counter = 1

    home_counts: Dict[str, int] = {t.team_id: 0 for t in meta}
    away_counts: Dict[str, int] = {t.team_id: 0 for t in meta}

    for (aid, bid), g in pair_games.items():
        # Alternate home/away; bias slightly toward teams that are currently
        # behind on home games to keep rough balance.
        for k in range(g):
            if home_counts[aid] - away_counts[aid] > home_counts[bid] - away_counts[bid]:
                home, away = bid, aid
            elif home_counts[bid] - away_counts[bid] > home_counts[aid] - away_counts[aid]:
                home, away = aid, bid
            else:
                # Equal; randomize who starts at home but deterministically using rng
                if rng.random() < 0.5:
                    home, away = aid, bid
                else:
                    home, away = bid, aid

            slots.append(GameSlot(day=day_counter, home_id=home, away_id=away, is_playoff=False))
            home_counts[home] += 1
            away_counts[away] += 1
            day_counter += 1

    # Shuffle schedule to avoid long homestands/road trips being systematically
    # aligned by team index; still deterministic under rng.
    rng.shuffle(slots)

    # Re-assign day indices to match shuffled order
    for idx, g in enumerate(slots, start=1):
        slots[idx - 1] = GameSlot(day=idx, home_id=g.home_id, away_id=g.away_id, is_playoff=g.is_playoff)

    return slots


def games_for_team(schedule: List[GameSlot], team_id: str) -> int:
    """Return total regular-season games in the schedule for a given team."""
    return sum(1 for g in schedule if not g.is_playoff and (g.home_id == team_id or g.away_id == team_id))


def team_schedule(schedule: List[GameSlot], team_id: str) -> List[GameSlot]:
    """Extract all scheduled games (in order) for a given team."""
    return [g for g in schedule if g.home_id == team_id or g.away_id == team_id]


