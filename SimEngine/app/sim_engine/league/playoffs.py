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
import math
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


# --- Playoff realism tuning constants -----------------------------------
# Logistic maps strength gaps onto per-game odds:
#   ~0.10 gap (typical 1 vs 8) -> ~59% favorite
#   ~0.40 gap (70-win vs 30-win) -> ~82-88% favorite
# Hard floor/ceiling still leave a path for upsets, but not coin-flips
# when the regular season was a mismatch.
_SERIES_LOGISTIC_K = 3.8
_SERIES_P_MIN = 0.22
_SERIES_P_MAX = 0.88
# Home-ice swing per game (games 1, 2, 5, 7 for the higher seed).
_HOME_ICE_GAME_EDGE = 0.038


def playoff_game_win_probability(
    strength_high: float,
    strength_low: float,
) -> float:
    """Per-game win probability for the higher-strength club (0..1)."""
    try:
        hi = float(strength_high)
        lo = float(strength_low)
    except (TypeError, ValueError):
        return 0.5
    diff = max(-0.55, min(0.55, hi - lo))
    p = 1.0 / (1.0 + math.exp(-_SERIES_LOGISTIC_K * diff))
    return max(_SERIES_P_MIN, min(_SERIES_P_MAX, p))


def _series_win_probability(
    strength_high: float,
    strength_low: float,
) -> float:
    return playoff_game_win_probability(strength_high, strength_low)


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
        # 2-2-1-1-1 format: higher seed hosts games 1, 2, 5, 7 (0-indexed 0,1,4,6).
        game_idx = wins_high + wins_low
        home_high = game_idx in (0, 1, 4, 6)
        p_game = p_high + (_HOME_ICE_GAME_EDGE if home_high else -_HOME_ICE_GAME_EDGE)
        p_game = max(_SERIES_P_MIN, min(_SERIES_P_MAX, p_game))
        if rng.random() < p_game:
            wins_high += 1
        else:
            wins_low += 1
    series.wins_high = wins_high
    series.wins_low = wins_low
    # upset if lower seed wins and there is a clear seed gap
    if series.winner_id() == series.team_low_id and series.seed_high + 1 < series.seed_low:
        series.upset = True
    return series


def _player_ovr_01(p: Any) -> Optional[float]:
    try:
        fn = getattr(p, "ovr", None)
        o = float(fn() if callable(fn) else fn)
    except Exception:
        return None
    if o > 1.5:
        o /= 99.0
    return max(0.0, min(1.0, o))


def _augment_strength_with_results(
    teams: List[Any],
    standings: StandingsTable,
    strength_map: Dict[str, float],
) -> Dict[str, float]:
    """Blend real regular-season results, goaltending, and top-end talent into the
    playoff strength signal.

    Regular-season points percentage and goal differential come straight from the
    game ledger standings, so Presidents'-Trophy-calibre teams carry their record
    into the bracket. Goalie quality and top-six talent add roster texture.
    Randomness still lives in the per-game series simulation.
    """
    out = dict(strength_map or {})
    goalie_by_tid: Dict[str, float] = {}
    top6_by_tid: Dict[str, float] = {}
    xg_share_by_tid: Dict[str, float] = {}
    for t in teams or []:
        tid = getattr(t, "team_id", None)
        if tid is None:
            tid = getattr(t, "id", None)
        if tid is None:
            continue
        tid = str(tid)
        roster = [p for p in (getattr(t, "roster", None) or []) if not getattr(p, "retired", False)]
        g_vals: List[float] = []
        sk_vals: List[float] = []
        for p in roster:
            o = _player_ovr_01(p)
            if o is None:
                continue
            pos = str(getattr(getattr(p, "identity", None), "position", "") or getattr(p, "position", "") or "")
            if pos.upper().rstrip().endswith("G"):
                g_vals.append(o)
            else:
                sk_vals.append(o)
        if g_vals:
            goalie_by_tid[tid] = max(g_vals)
        if sk_vals:
            sk_vals.sort(reverse=True)
            top = sk_vals[:6]
            top6_by_tid[tid] = sum(top) / len(top)
        xgf = getattr(t, "season_xgf", None)
        xga = getattr(t, "season_xga", None)
        try:
            if xgf is not None and xga is not None:
                tot = float(xgf) + float(xga)
                if tot >= 15.0:
                    xg_share_by_tid[tid] = float(xgf) / tot
        except (TypeError, ValueError):
            pass

    g_mean = (sum(goalie_by_tid.values()) / len(goalie_by_tid)) if goalie_by_tid else 0.0
    s_mean = (sum(top6_by_tid.values()) / len(top6_by_tid)) if top6_by_tid else 0.0

    for rec in standings.league_table():
        tid = str(rec.team_id)
        gp = max(1, int(getattr(rec, "gp", 0) or 0))
        ppct = float(getattr(rec, "points", 0) or 0) / (2.0 * gp)
        gd_pg = (float(getattr(rec, "gf", 0) or 0) - float(getattr(rec, "ga", 0) or 0)) / gp
        adj = float(out.get(tid, 0.5))
        adj += 0.62 * (ppct - 0.5)
        adj += 0.038 * max(-2.0, min(2.0, gd_pg))
        if tid in xg_share_by_tid:
            # Season xG can contradict a lucky/unlucky record.
            adj += 0.90 * (xg_share_by_tid[tid] - 0.5)
        if tid in goalie_by_tid and g_mean > 0:
            adj += 0.20 * (goalie_by_tid[tid] - g_mean)
        if tid in top6_by_tid and s_mean > 0:
            adj += 0.16 * (top6_by_tid[tid] - s_mean)
        out[tid] = max(0.05, min(0.99, adj))
    return out


def _sort_team_ids_by_standings(standings: StandingsTable, team_ids: List[str]) -> List[str]:
    recs = [standings.records[tid] for tid in team_ids if tid in standings.records]
    recs.sort(key=standings._sort_key, reverse=True)
    return [r.team_id for r in recs]


def _simulate_nhl_conference_bracket(
    rng: random.Random,
    standings: StandingsTable,
    conference: str,
    strength_map: Dict[str, float],
) -> Tuple[List[PlayoffSeries], str]:
    """
    Simulate rounds 1–3 for one conference. Returns (all series, conference champion team_id).
    """
    series_all: List[PlayoffSeries] = []
    r1 = standings.nhl_conference_first_round_series(conference)
    if len(r1) != 4:
        return series_all, ""

    current = [_simulate_series(rng, s, strength_map) for s in r1]
    series_all.extend(current)
    wids = _sort_team_ids_by_standings(standings, [s.winner_id() for s in current])
    if len(wids) < 4:
        return series_all, wids[0] if wids else ""

    r2: List[PlayoffSeries] = [
        PlayoffSeries(
            round_index=2,
            conference=conference,
            seed_high=1,
            seed_low=4,
            team_high_id=wids[0],
            team_low_id=wids[3],
        ),
        PlayoffSeries(
            round_index=2,
            conference=conference,
            seed_high=2,
            seed_low=3,
            team_high_id=wids[1],
            team_low_id=wids[2],
        ),
    ]
    cur2 = [_simulate_series(rng, s, strength_map) for s in r2]
    series_all.extend(cur2)
    w2 = _sort_team_ids_by_standings(standings, [s.winner_id() for s in cur2])
    if len(w2) < 2:
        return series_all, w2[0] if w2 else ""

    cf = PlayoffSeries(
        round_index=3,
        conference=conference,
        seed_high=1,
        seed_low=2,
        team_high_id=w2[0],
        team_low_id=w2[1],
    )
    cur3 = _simulate_series(rng, cf, strength_map)
    series_all.append(cur3)
    return series_all, cur3.winner_id()


def _simulate_nhl_full_playoffs(
    rng: random.Random,
    standings: StandingsTable,
    teams: List[Any],
    strength_map: Dict[str, float],
) -> Optional[PlayoffResult]:
    """Stanley Cup playoffs using division + wild-card R1 when standings support it."""
    if not standings.uses_nhl_playoff_pairings():
        return None

    series_all: List[PlayoffSeries] = []
    conf_champs: Dict[str, str] = {}
    for conf in sorted(standings._by_conf.keys()):
        block, champ = _simulate_nhl_conference_bracket(rng, standings, conf, strength_map)
        series_all.extend(block)
        if champ:
            conf_champs[conf] = champ

    champs = list(conf_champs.values())
    if len(champs) >= 2:
        ordered = _sort_team_ids_by_standings(standings, champs)
        fin = PlayoffSeries(
            round_index=4,
            conference=None,
            seed_high=1,
            seed_low=2,
            team_high_id=ordered[0],
            team_low_id=ordered[1],
        )
        curf = _simulate_series(rng, fin, strength_map)
        series_all.append(curf)
        champion = curf.winner_id()
        finalists = [curf.team_high_id, curf.team_low_id]
        return PlayoffResult(champion_id=champion, finalist_ids=finalists, series_list=series_all)

    if len(champs) == 1:
        champion = champs[0]
        cf_list = [s for s in series_all if getattr(s, "round_index", 0) == 3]
        if cf_list:
            last_cf = cf_list[-1]
            finalists = [last_cf.team_high_id, last_cf.team_low_id]
        else:
            finalists = [champion, champion]
        return PlayoffResult(champion_id=champion, finalist_ids=finalists, series_list=series_all)

    return None


def build_playoff_first_round(
    standings: StandingsTable,
) -> Tuple[List[PlayoffSeries], List[TeamStandingRecord]]:
    """
    Build round-1 pairings and the playoff field without simulating series.

    Uses NHL division + wild-card pairings when standings support them;
    otherwise falls back to conference 1v8 seeding or league-wide 1v16.
    """
    playoff_teams: List[TeamStandingRecord] = []
    seen: set[str] = set()

    def _track(records: List[TeamStandingRecord]) -> None:
        for rec in records:
            tid = str(rec.team_id)
            if tid in seen:
                continue
            seen.add(tid)
            playoff_teams.append(rec)

    if standings.uses_nhl_playoff_pairings():
        first_round: List[PlayoffSeries] = []
        for conf in sorted(standings._by_conf.keys()):
            r1 = standings.nhl_conference_first_round_series(conf)
            first_round.extend(r1)
            eight = standings.nhl_conference_playoff_eight(conf)
            if eight:
                _track(eight)
        return first_round, playoff_teams

    seeds_by_conf = standings.playoff_seeds_by_conference(per_conf=8)
    first_round = []
    if "ALL" in seeds_by_conf:
        seeds = seeds_by_conf["ALL"]
        first_round = _build_league_bracket(standings, seeds)
        _track(list(seeds[:16]))
    else:
        for conf, seeds in sorted(seeds_by_conf.items()):
            first_round.extend(_build_conference_bracket(standings, conf, seeds))
            _track(list(seeds))

    return first_round, playoff_teams


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

    # Fold real ledger results (pts%, goal diff), goaltending, and top-end
    # talent into the playoff strength signal before any series is simulated.
    strength_map = _augment_strength_with_results(teams, standings, strength_map)

    nhl_res = _simulate_nhl_full_playoffs(rng, standings, teams, strength_map)
    if nhl_res is not None:
        return nhl_res

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


