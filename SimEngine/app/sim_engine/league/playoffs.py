from __future__ import annotations

"""
Playoff bracket + series simulation.

This module consumes a StandingsTable and basic team strength map to
produce:
    - playoff field
    - best-of-seven series results
    - a Stanley Cup champion

It deliberately stays at the series level (game-level scoring can be
abstracted by the engine when needed).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random

from .standings import StandingsTable, TeamStandingRecord


@dataclass
class PlayoffSeries:
    round_index: int  # 1 = first round, etc.
    conference: Optional[str]
    seed_high: int
    seed_low: int
    team_high_id: str
    team_low_id: str
    wins_high: int = 0
    wins_low: int = 0
    best_of: int = 7
    upset: bool = False

    def winner_id(self) -> str:
        return self.team_high_id if self.wins_high > self.wins_low else self.team_low_id

    def loser_id(self) -> str:
        return self.team_low_id if self.wins_high > self.wins_low else self.team_high_id

    def series_score(self) -> str:
        return f"{self.wins_high}-{self.wins_low}"


@dataclass
class PlayoffResult:
    champion_id: str
    finalist_ids: List[str]
    series_list: List[PlayoffSeries]


def _build_conference_bracket(
    standings: StandingsTable,
    conference: str,
    seeds: List[TeamStandingRecord],
) -> List[PlayoffSeries]:
    """
    Seed within a conference:
        1 vs 8, 2 vs 7, 3 vs 6, 4 vs 5
    """
    series: List[PlayoffSeries] = []
    pairs = [(0, 7), (1, 6), (2, 5), (3, 4)]
    for high_idx, low_idx in pairs:
        if high_idx >= len(seeds) or low_idx >= len(seeds):
            continue
        hi = seeds[high_idx]
        lo = seeds[low_idx]
        series.append(
            PlayoffSeries(
                round_index=1,
                conference=conference,
                seed_high=high_idx + 1,
                seed_low=low_idx + 1,
                team_high_id=hi.team_id,
                team_low_id=lo.team_id,
            )
        )
    return series


def _build_league_bracket(
    standings: StandingsTable,
    seeds: List[TeamStandingRecord],
) -> List[PlayoffSeries]:
    """
    Fallback when no conferences are available:
        1 vs 16, 2 vs 15, ..., 8 vs 9
    """
    series: List[PlayoffSeries] = []
    n = min(16, len(seeds))
    for i in range(n // 2):
        hi = seeds[i]
        lo = seeds[n - 1 - i]
        series.append(
            PlayoffSeries(
                round_index=1,
                conference=None,
                seed_high=i + 1,
                seed_low=n - i,
                team_high_id=hi.team_id,
                team_low_id=lo.team_id,
            )
        )
    return series


def _series_win_probability(
    strength_high: float,
    strength_low: float,
) -> float:
    """
    Approximate per-game win probability for the higher seed, based on
    normalized strength values (0..1 range). Slightly favors the higher
    strength but leaves plenty of room for upsets.
    """
    # Map strength difference into [-0.25, 0.25] then shift around 0.5.
    diff = max(-0.30, min(0.30, strength_high - strength_low))
    return max(0.40, min(0.75, 0.5 + diff * 0.8))


def _simulate_series(
    rng: random.Random,
    series: PlayoffSeries,
    strength_map: Dict[str, float],
) -> PlayoffSeries:
    p_high = _series_win_probability(
        strength_map.get(series.team_high_id, 0.5),
        strength_map.get(series.team_low_id, 0.5),
    )
    wins_high = 0
    wins_low = 0
    needed = (series.best_of // 2) + 1
    while wins_high < needed and wins_low < needed:
        if rng.random() < p_high:
            wins_high += 1
        else:
            wins_low += 1
    series.wins_high = wins_high
    series.wins_low = wins_low
    # upset if lower seed wins and there is a clear seed gap
    if series.winner_id() == series.team_low_id and series.seed_high + 1 < series.seed_low:
        series.upset = True
    return series


def simulate_playoffs(
    rng: random.Random,
    standings: StandingsTable,
    teams: List[Any],
    strength_map: Dict[str, float],
) -> Optional[PlayoffResult]:
    """
    Build a playoff field and simulate all rounds until a champion is
    crowned. Returns None if there are not enough teams to form a field.
    """
    all_records = standings.league_table()
    if len(all_records) < 2:
        return None

    seeds_by_conf = standings.playoff_seeds_by_conference(per_conf=8)
    series_all: List[PlayoffSeries] = []

    # Round 1
    if "ALL" in seeds_by_conf:
        first_round = _build_league_bracket(standings, seeds_by_conf["ALL"])
    else:
        first_round: List[PlayoffSeries] = []
        for conf, seeds in sorted(seeds_by_conf.items()):
            first_round.extend(_build_conference_bracket(standings, conf, seeds))

    if not first_round:
        return None

    # Helper for advancing winners within a group of series
    def advance_round(prev_round: List[PlayoffSeries], round_index: int) -> List[PlayoffSeries]:
        # Group by conference (None = league-wide)
        grouped: Dict[Optional[str], List[PlayoffSeries]] = {}
        for s in prev_round:
            grouped.setdefault(s.conference, []).append(s)

        out: List[PlayoffSeries] = []
        for conf, bucket in grouped.items():
            # Sort by seed_high so matchups are stable for next round
            bucket = sorted(bucket, key=lambda s: s.seed_high)
            winners = [s.winner_id() for s in bucket]
            # Pair winners 1 vs 4, 2 vs 3 (or 1 vs N, etc. for arbitrary size)
            for i in range(len(winners) // 2):
                hi = winners[i]
                lo = winners[-(i + 1)]
                out.append(
                    PlayoffSeries(
                        round_index=round_index,
                        conference=conf,
                        seed_high=i + 1,
                        seed_low=len(winners) - i,
                        team_high_id=hi,
                        team_low_id=lo,
                    )
                )
        return out

    # Simulate all rounds
    current_round = [ _simulate_series(rng, s, strength_map) for s in first_round ]
    series_all.extend(current_round)

    round_index = 2
    while True:
        next_round = advance_round(current_round, round_index)
        if not next_round:
            break
        current_round = [ _simulate_series(rng, s, strength_map) for s in next_round ]
        series_all.extend(current_round)
        round_index += 1
        # Safety: stop if only one team remains
        total_winners = {s.winner_id() for s in current_round}
        if len(total_winners) == 1:
            break

    # Determine champion and finalists
    last_series = series_all[-1]
    champion = last_series.winner_id()
    finalists = [last_series.team_high_id, last_series.team_low_id]

    return PlayoffResult(
        champion_id=champion,
        finalist_ids=finalists,
        series_list=series_all,
    )


