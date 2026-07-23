from __future__ import annotations

"""
League schedule generation utilities.

This module is responsible for producing a deterministic, NHL-ish regular-season
schedule for an arbitrary set of team objects.

Core fix:
The old version assigned every game to a unique abstract day:

    Game 1 -> day 1
    Game 2 -> day 2
    Game 3 -> day 3

That is not how NHL scheduling works and it causes the calendar mapper to behave
terribly. This version creates matchups first, then packs them into NHL-style
league slates:

    Day 1 -> multiple games
    Day 2 -> multiple games
    Day 3 -> lighter slate
    Saturday/Tuesday/Thursday -> heavier slate

It also tries to protect each team from:
- too many games in too few days
- huge dead stretches
- insane monthly dumps
- endless home/away imbalance
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import random


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_id_str(value: Any) -> str:
    """Stringify an id; 0 is valid and must not collapse to empty."""
    if value is None:
        return ""
    return str(value)


def _safe_team_id(team: Any, fallback_index: int) -> str:
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", None)
    if tid is None:
        return f"T{fallback_index:02d}"
    return str(tid)


def _safe_slot_team_id(slot: Any, attr: str) -> str:
    return _safe_id_str(getattr(slot, attr, None))


def _safe_team_name(team: Any, team_id: str) -> str:
    name = getattr(team, "name", None)
    city = getattr(team, "city", None)

    if city and name:
        return f"{city} {name}"
    if name:
        return str(name)

    return str(team_id)


def _safe_conf(team: Any) -> Optional[str]:
    value = getattr(team, "conference", None)
    return str(value) if value is not None and str(value).strip() else None


def _safe_div(team: Any) -> Optional[str]:
    value = getattr(team, "division", None)
    return str(value) if value is not None and str(value).strip() else None


def _clamp_int(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = int(fallback)

    return max(int(low), min(int(high), v))


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSlot:
    """
    Represents a single scheduled game in the league calendar.

    - day: abstract slate day, not unique game number.
      Multiple games can and should share the same day.
    - home_id / away_id: team identifiers.
    - is_playoff: regular season uses False.
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


@dataclass(frozen=True)
class _UnslottedGame:
    home_id: str
    away_id: str
    matchup_key: Tuple[str, str]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_teams(teams: Iterable[Any]) -> Tuple[List[TeamScheduleMeta], Dict[str, Any]]:
    """
    Convert arbitrary team objects into a deterministic metadata list and
    id->team mapping.
    """
    meta: List[TeamScheduleMeta] = []
    by_id: Dict[str, Any] = {}

    for idx, team in enumerate(teams):
        tid = _safe_team_id(team, idx)
        name = _safe_team_name(team, tid)
        conf = _safe_conf(team)
        div = _safe_div(team)

        meta.append(
            TeamScheduleMeta(
                team_id=tid,
                name=name,
                conference=conf,
                division=div,
            )
        )
        by_id[tid] = team

    return meta, by_id


# ---------------------------------------------------------------------------
# Matchup construction
# ---------------------------------------------------------------------------


def _has_complete_alignment(meta: List[TeamScheduleMeta]) -> bool:
    """
    True if every team has usable conference/division metadata.
    """
    if not meta:
        return False

    return all(t.conference and t.division for t in meta)


def _base_pair_games(a: TeamScheduleMeta, b: TeamScheduleMeta, complete_alignment: bool) -> int:
    """
    NHL-ish pair frequency.

    If conference/division metadata is complete:
    - same division: 4
    - same conference, different division: 3
    - different conference: 2

    If metadata is missing/incomplete:
    - start with 2 games per opponent and let the balancing pass add games
      until each team approaches games_per_team.
    """
    if not complete_alignment:
        return 2

    if a.conference == b.conference:
        if a.division == b.division:
            return 4
        return 3

    return 2


def _pair_key(aid: str, bid: str) -> Tuple[str, str]:
    return tuple(sorted((str(aid), str(bid))))


def _make_game_for_pair(
    rng: random.Random,
    aid: str,
    bid: str,
    home_counts: Dict[str, int],
    away_counts: Dict[str, int],
) -> _UnslottedGame:
    """
    Create one unslotted game for a pair and choose home/away while trying to
    keep home/away balance.
    """
    a_home_diff = home_counts.get(aid, 0) - away_counts.get(aid, 0)
    b_home_diff = home_counts.get(bid, 0) - away_counts.get(bid, 0)

    if a_home_diff > b_home_diff:
        home, away = bid, aid
    elif b_home_diff > a_home_diff:
        home, away = aid, bid
    else:
        if rng.random() < 0.5:
            home, away = aid, bid
        else:
            home, away = bid, aid

    home_counts[home] = home_counts.get(home, 0) + 1
    away_counts[away] = away_counts.get(away, 0) + 1

    return _UnslottedGame(
        home_id=home,
        away_id=away,
        matchup_key=_pair_key(aid, bid),
    )


def _build_pair_game_counts(
    rng: random.Random,
    meta: List[TeamScheduleMeta],
    games_per_team: int,
) -> Dict[Tuple[str, str], int]:
    """
    Build pair game counts and then add/remove games to get each team close to
    games_per_team.

    This is intentionally generic because franchise mode may not always have
    a real NHL 32-team alignment.
    """
    n = len(meta)
    complete_alignment = _has_complete_alignment(meta)

    pair_counts: Dict[Tuple[str, str], int] = {}

    for i in range(n):
        a = meta[i]
        for j in range(i + 1, n):
            b = meta[j]
            pair_counts[_pair_key(a.team_id, b.team_id)] = _base_pair_games(
                a,
                b,
                complete_alignment,
            )

    team_counts: Dict[str, int] = {t.team_id: 0 for t in meta}

    def recalc_team_counts() -> None:
        for tid in team_counts:
            team_counts[tid] = 0

        for (aid, bid), count in pair_counts.items():
            team_counts[aid] += count
            team_counts[bid] += count

    recalc_team_counts()

    # Add games until teams approach games_per_team.
    # Prefer rivalry/division/conference pairings when metadata exists.
    max_iterations = max(1000, games_per_team * max(2, n) * 8)
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        low_teams = [
            t
            for t in meta
            if team_counts.get(t.team_id, 0) < games_per_team
        ]

        if not low_teams:
            break

        low_teams.sort(key=lambda t: team_counts.get(t.team_id, 0))
        anchor = low_teams[0]

        candidates: List[Tuple[int, TeamScheduleMeta]] = []

        for other in meta:
            if other.team_id == anchor.team_id:
                continue

            other_count = team_counts.get(other.team_id, 0)

            # Do not keep feeding a team that is already heavily over target.
            if other_count >= games_per_team + 2:
                continue

            score = 0

            if anchor.conference and other.conference and anchor.conference == other.conference:
                score += 4

                if anchor.division and other.division and anchor.division == other.division:
                    score += 4

            # Prefer underfilled teams.
            score += max(0, games_per_team - other_count)

            # Avoid absurdly high pair repetition.
            existing = pair_counts.get(_pair_key(anchor.team_id, other.team_id), 0)
            score -= max(0, existing - 4) * 3

            candidates.append((score, other))

        if not candidates:
            break

        candidates.sort(key=lambda row: row[0], reverse=True)

        # Randomize among the top few so the schedule is deterministic but not
        # identical-looking every year if the RNG seed changes.
        top_score = candidates[0][0]
        top = [other for score, other in candidates if score >= top_score - 2]
        opponent = rng.choice(top)

        key = _pair_key(anchor.team_id, opponent.team_id)
        pair_counts[key] = pair_counts.get(key, 0) + 1

        team_counts[anchor.team_id] += 1
        team_counts[opponent.team_id] += 1

    # If a small/custom league overshoots because base pair counts were too high,
    # trim cautiously from overfilled teams.
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        over_teams = [
            t
            for t in meta
            if team_counts.get(t.team_id, 0) > games_per_team
        ]

        if not over_teams:
            break

        over_teams.sort(key=lambda t: team_counts.get(t.team_id, 0), reverse=True)
        anchor = over_teams[0]

        removable: List[Tuple[int, Tuple[str, str]]] = []

        for key, count in pair_counts.items():
            if anchor.team_id not in key:
                continue

            if count <= 1:
                continue

            aid, bid = key
            other_id = bid if aid == anchor.team_id else aid

            if team_counts.get(other_id, 0) <= games_per_team - 1:
                continue

            # Prefer trimming pairs with the highest repetition.
            removable.append((count, key))

        if not removable:
            break

        removable.sort(reverse=True)
        _, key = removable[0]
        aid, bid = key

        pair_counts[key] -= 1
        team_counts[aid] -= 1
        team_counts[bid] -= 1

    return pair_counts


def _build_unslotted_games(
    rng: random.Random,
    meta: List[TeamScheduleMeta],
    games_per_team: int,
) -> List[_UnslottedGame]:
    pair_counts = _build_pair_game_counts(rng, meta, games_per_team)

    home_counts: Dict[str, int] = {t.team_id: 0 for t in meta}
    away_counts: Dict[str, int] = {t.team_id: 0 for t in meta}

    games: List[_UnslottedGame] = []

    # Shuffle pair order to avoid the schedule being grouped by team/pair.
    pair_items = list(pair_counts.items())
    rng.shuffle(pair_items)

    for (aid, bid), count in pair_items:
        for _ in range(max(0, int(count))):
            games.append(
                _make_game_for_pair(
                    rng,
                    aid,
                    bid,
                    home_counts,
                    away_counts,
                )
            )

    rng.shuffle(games)
    return games


def _team_game_totals(games: List[_UnslottedGame]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for game in games:
        counts[game.home_id] += 1
        counts[game.away_id] += 1
    return counts


def _balance_unslotted_games_to_target(
    rng: random.Random,
    meta: List[TeamScheduleMeta],
    games: List[_UnslottedGame],
    games_per_team: int,
) -> List[_UnslottedGame]:
    """
    Add or remove unslotted games so every team lands on games_per_team exactly.
    """
    games_per_team = max(1, int(games_per_team))
    team_ids = [t.team_id for t in meta]
    out = list(games)
    max_iterations = max(500, games_per_team * max(2, len(meta)) * 6)

    for _ in range(max_iterations):
        counts = _team_game_totals(out)
        under = [tid for tid in team_ids if counts.get(tid, 0) < games_per_team]
        over = [tid for tid in team_ids if counts.get(tid, 0) > games_per_team]

        if not under and not over:
            break

        if under:
            anchor = min(under, key=lambda tid: counts.get(tid, 0))
            candidates = [
                other
                for other in team_ids
                if other != anchor and counts.get(other, 0) < games_per_team + 1
            ]
            if not candidates:
                break
            opponent = rng.choice(candidates)
            out.append(
                _make_game_for_pair(
                    rng,
                    anchor,
                    opponent,
                    defaultdict(int),
                    defaultdict(int),
                )
            )
            continue

        anchor = max(over, key=lambda tid: counts.get(tid, 0))
        removable_idx = -1

        for idx, game in enumerate(out):
            if anchor not in (game.home_id, game.away_id):
                continue

            other_id = game.away_id if game.home_id == anchor else game.home_id
            if counts.get(other_id, 0) <= games_per_team:
                continue

            removable_idx = idx
            break

        if removable_idx < 0:
            for idx, game in enumerate(out):
                if anchor in (game.home_id, game.away_id):
                    removable_idx = idx
                    break

        if removable_idx < 0:
            break

        out.pop(removable_idx)

    return out


# ---------------------------------------------------------------------------
# Slate construction
# ---------------------------------------------------------------------------


def _target_slate_count(games_per_team: int) -> int:
    """
    Abstract regular-season slate count.

    This should roughly represent playable NHL calendar nights from October
    through April, not number of games.

    For 82 games/team, around 176-184 abstract slate days works well because:
    - teams play around 2-4 games per week
    - the league has heavy and light nights
    - many games share the same abstract day
    """
    games_per_team = max(1, int(games_per_team))

    if games_per_team >= 70:
        return 180

    # Scale down for shorter franchise modes.
    return max(28, int(round(180 * (games_per_team / 82.0))))


def _slate_capacity_pattern(num_slate_days: int, num_teams: int) -> List[int]:
    """
    Create NHL-like league slate capacities.

    Pattern repeats weekly:
    - Monday: light
    - Tuesday: heavy
    - Wednesday: light
    - Thursday: heavy
    - Friday: medium
    - Saturday: monster slate
    - Sunday: medium

    Capacity means maximum games on that abstract day.
    """
    num_slate_days = max(1, int(num_slate_days))
    max_games_per_day = max(1, num_teams // 2)

    # For a 32-team league, this becomes roughly:
    # Mon 4, Tue 11, Wed 4, Thu 11, Fri 7, Sat 15, Sun 7.
    raw_week = [
        0.25,  # Monday
        0.72,  # Tuesday
        0.25,  # Wednesday
        0.72,  # Thursday
        0.48,  # Friday
        0.95,  # Saturday
        0.48,  # Sunday
    ]

    capacities: List[int] = []

    for i in range(num_slate_days):
        ratio = raw_week[i % 7]
        cap = int(round(max_games_per_day * ratio))
        cap = max(1, min(max_games_per_day, cap))
        capacities.append(cap)

    return capacities


def _month_bucket_for_day(day: int, total_days: int) -> int:
    """
    Approximate month bucket from abstract day.

    Buckets:
    10, 11, 12, 1, 2, 3, 4

    This keeps the schedule rhythm similar to an NHL season even before the
    real calendar mapper turns abstract days into actual dates.
    """
    if total_days <= 1:
        return 10

    pct = (day - 1) / max(1, total_days - 1)

    if pct < 0.13:
        return 10
    if pct < 0.285:
        return 11
    if pct < 0.445:
        return 12
    if pct < 0.605:
        return 1
    if pct < 0.725:
        return 2
    if pct < 0.895:
        return 3
    return 4


def _monthly_game_cap(games_per_team: int, month_bucket: int) -> int:
    """
    Approximate max games/team per abstract month bucket.

    This prevents nonsense like 20 user-team games in one month.
    """
    scale = max(0.25, games_per_team / 82.0)

    base_caps = {
        10: 12,
        11: 14,
        12: 14,
        1: 14,
        2: 11,
        3: 16,
        4: 9,
    }

    return max(3, int(round(base_caps.get(month_bucket, 14) * scale)))


def _recent_games_count(team_days: List[int], day: int, window: int) -> int:
    start = int(day) - int(window) + 1
    return sum(1 for d in team_days if start <= d <= day)


def _last_game_gap(team_days: List[int], day: int) -> Optional[int]:
    if not team_days:
        return None

    return int(day) - max(team_days)


def _would_violate_cadence(
    game: _UnslottedGame,
    day: int,
    team_days: Dict[str, List[int]],
    team_month_counts: Dict[str, Dict[int, int]],
    month_bucket: int,
    games_per_team: int,
) -> bool:
    """
    Strict enough to stop broken schedules, soft enough to still place games.
    """
    for tid in (game.home_id, game.away_id):
        days = team_days.get(tid, [])

        # Never allow 3 games in 3 nights in the abstract layer.
        if _recent_games_count(days, day, 3) >= 2:
            return True

        # Avoid 4 games in 6 nights.
        if _recent_games_count(days, day, 6) >= 3:
            return True

        # Avoid 6 games in 10 nights.
        if _recent_games_count(days, day, 10) >= 5:
            return True

        # Monthly cap.
        month_count = team_month_counts.get(tid, {}).get(month_bucket, 0)
        if month_count >= _monthly_game_cap(games_per_team, month_bucket):
            return True

    return False


def _game_fit_score(
    game: _UnslottedGame,
    day: int,
    team_days: Dict[str, List[int]],
    team_counts: Dict[str, int],
    team_month_counts: Dict[str, Dict[int, int]],
    month_bucket: int,
    games_per_team: int,
) -> float:
    """
    Higher score = better game to place on this day.
    """
    score = 0.0

    for tid in (game.home_id, game.away_id):
        days = team_days.get(tid, [])
        total_played = team_counts.get(tid, 0)

        gap = _last_game_gap(days, day)

        if gap is None:
            # Teams with no games yet should start getting games early.
            score += 10.0
        else:
            if gap == 0:
                score -= 999.0
            elif gap == 1:
                score -= 16.0
            elif gap == 2:
                score += 1.0
            elif gap == 3:
                score += 5.0
            elif gap == 4:
                score += 7.0
            elif gap <= 7:
                score += 4.0
            else:
                # Reward ending long gaps.
                score += min(14.0, gap * 1.4)

        # Teams below expected pace get priority.
        expected_pace = (day / max(1, _target_slate_count(games_per_team))) * games_per_team
        score += max(-8.0, min(8.0, expected_pace - total_played))

        # Avoid monthly overload.
        month_count = team_month_counts.get(tid, {}).get(month_bucket, 0)
        cap = _monthly_game_cap(games_per_team, month_bucket)
        score += max(-10.0, cap - month_count)

    return score


def _place_game(
    game: _UnslottedGame,
    day: int,
    placed: Dict[int, List[_UnslottedGame]],
    team_days: Dict[str, List[int]],
    team_counts: Dict[str, int],
    team_month_counts: Dict[str, Dict[int, int]],
    total_slate_days: int,
) -> None:
    placed.setdefault(day, []).append(game)

    month_bucket = _month_bucket_for_day(day, total_slate_days)

    for tid in (game.home_id, game.away_id):
        team_days.setdefault(tid, []).append(day)
        team_days[tid].sort()

        team_counts[tid] = team_counts.get(tid, 0) + 1

        team_month_counts.setdefault(tid, {})
        team_month_counts[tid][month_bucket] = team_month_counts[tid].get(month_bucket, 0) + 1


def _pack_games_into_slates(
    rng: random.Random,
    games: List[_UnslottedGame],
    team_ids: List[str],
    games_per_team: int,
) -> List[GameSlot]:
    """
    Pack unslotted games into abstract NHL slate days.

    This is the key replacement for the broken day_counter logic.
    """
    if not games:
        return []

    slate_count = _target_slate_count(games_per_team)
    capacities = _slate_capacity_pattern(slate_count, len(team_ids))

    remaining = list(games)
    rng.shuffle(remaining)

    placed: Dict[int, List[_UnslottedGame]] = {day: [] for day in range(1, slate_count + 1)}
    team_days: Dict[str, List[int]] = {tid: [] for tid in team_ids}
    team_counts: Dict[str, int] = {tid: 0 for tid in team_ids}
    team_month_counts: Dict[str, Dict[int, int]] = {tid: {} for tid in team_ids}

    # Multiple passes with gradually relaxed rules.
    # Pass 1: strict cadence.
    # Pass 2: slightly less strict.
    # Pass 3: place leftovers wherever possible.
    for day in range(1, slate_count + 1):
        cap = capacities[day - 1]

        while len(placed[day]) < cap and remaining:
            month_bucket = _month_bucket_for_day(day, slate_count)

            candidates: List[Tuple[float, int, _UnslottedGame]] = []

            scan_limit = min(len(remaining), 180)

            for idx in range(scan_limit):
                game = remaining[idx]

                teams_already_today = {
                    tid
                    for g in placed[day]
                    for tid in (g.home_id, g.away_id)
                }

                if game.home_id in teams_already_today or game.away_id in teams_already_today:
                    continue

                if _would_violate_cadence(
                    game,
                    day,
                    team_days,
                    team_month_counts,
                    month_bucket,
                    games_per_team,
                ):
                    continue

                score = _game_fit_score(
                    game,
                    day,
                    team_days,
                    team_counts,
                    team_month_counts,
                    month_bucket,
                    games_per_team,
                )

                candidates.append((score, idx, game))

            if not candidates:
                break

            candidates.sort(key=lambda row: row[0], reverse=True)

            # Pick from top candidates for deterministic variety.
            top = candidates[: min(8, len(candidates))]
            _, chosen_idx, chosen_game = rng.choice(top)

            remaining.pop(chosen_idx)

            _place_game(
                chosen_game,
                day,
                placed,
                team_days,
                team_counts,
                team_month_counts,
                slate_count,
            )

    # Relaxed pass for leftovers.
    if remaining:
        for day in range(1, slate_count + 1):
            cap = capacities[day - 1]

            while len(placed[day]) < cap and remaining:
                month_bucket = _month_bucket_for_day(day, slate_count)

                candidates = []

                for idx, game in enumerate(remaining[:220]):
                    teams_already_today = {
                        tid
                        for g in placed[day]
                        for tid in (g.home_id, g.away_id)
                    }

                    if game.home_id in teams_already_today or game.away_id in teams_already_today:
                        continue

                    # Relaxed: only block the truly cursed stuff.
                    bad = False

                    for tid in (game.home_id, game.away_id):
                        days = team_days.get(tid, [])

                        if _recent_games_count(days, day, 2) >= 1:
                            bad = True
                            break

                        if _recent_games_count(days, day, 5) >= 3:
                            bad = True
                            break

                    if bad:
                        continue

                    score = _game_fit_score(
                        game,
                        day,
                        team_days,
                        team_counts,
                        team_month_counts,
                        month_bucket,
                        games_per_team,
                    )

                    candidates.append((score, idx, game))

                if not candidates:
                    break

                candidates.sort(key=lambda row: row[0], reverse=True)
                _, chosen_idx, chosen_game = candidates[0]

                remaining.pop(chosen_idx)

                _place_game(
                    chosen_game,
                    day,
                    placed,
                    team_days,
                    team_counts,
                    team_month_counts,
                    slate_count,
                )

    # Emergency pass.
    # This should rarely be needed, but it guarantees we do not drop games.
    if remaining:
        for game in list(remaining):
            best_day = 1
            best_score = -10**9

            for day in range(1, slate_count + 1):
                if len(placed[day]) >= max(1, len(team_ids) // 2):
                    continue

                teams_already_today = {
                    tid
                    for g in placed[day]
                    for tid in (g.home_id, g.away_id)
                }

                if game.home_id in teams_already_today or game.away_id in teams_already_today:
                    continue

                month_bucket = _month_bucket_for_day(day, slate_count)

                score = _game_fit_score(
                    game,
                    day,
                    team_days,
                    team_counts,
                    team_month_counts,
                    month_bucket,
                    games_per_team,
                )

                if score > best_score:
                    best_score = score
                    best_day = day

            _place_game(
                game,
                best_day,
                placed,
                team_days,
                team_counts,
                team_month_counts,
                slate_count,
            )

        remaining.clear()

    # Convert to public GameSlot list.
    output: List[GameSlot] = []

    for day in range(1, slate_count + 1):
        day_games = placed.get(day, [])

        # Shuffle game order inside the slate.
        rng.shuffle(day_games)

        for game in day_games:
            output.append(
                GameSlot(
                    day=day,
                    home_id=game.home_id,
                    away_id=game.away_id,
                    is_playoff=False,
                )
            )

    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_regular_season_schedule(
    rng: random.Random,
    teams: List[Any],
    games_per_team: int = 82,
) -> List[GameSlot]:
    """
    Generate a deterministic, NHL-ish regular-season schedule.

    Fixes from the old version:
    - Multiple games now share the same abstract day.
    - Abstract days represent league slate days, not individual game numbers.
    - The schedule has heavier Tuesdays/Thursdays/Saturdays.
    - Each team is protected from ridiculous cadence as much as possible.
    - games_per_team is now actively targeted instead of mostly ignored.

    Note:
    Final real dates are still assigned later by the calendar mapper.
    This function only creates sane abstract slate days.
    """
    if not teams:
        return []

    games_per_team = _clamp_int(games_per_team, 1, 120, 82)

    meta, _ = normalize_teams(teams)
    n = len(meta)

    if n <= 1:
        return []

    # A team cannot play more unique opponents than the league allows without
    # repeats. Repeats are allowed, but for tiny leagues very high game counts
    # can still be silly. We keep the requested value, because franchise mode
    # may intentionally use weird settings.
    unslotted_games = _build_unslotted_games(
        rng,
        meta,
        games_per_team,
    )
    unslotted_games = _balance_unslotted_games_to_target(
        rng,
        meta,
        unslotted_games,
        games_per_team,
    )

    team_ids = [t.team_id for t in meta]

    schedule = _pack_games_into_slates(
        rng,
        unslotted_games,
        team_ids,
        games_per_team,
    )

    # Stable sort by day, then home/away IDs.
    schedule.sort(key=lambda g: (g.day, g.home_id, g.away_id))

    for tid in team_ids:
        if games_for_team(schedule, tid) != games_per_team:
            raise RuntimeError(
                f"Schedule generation failed GP target for team {tid}: "
                f"expected {games_per_team}, found {games_for_team(schedule, tid)}"
            )

    by_day_games: Dict[int, List[GameSlot]] = defaultdict(list)
    for game in schedule:
        by_day_games[int(game.day)].append(game)

    for day, day_games in by_day_games.items():
        teams_today = {
            tid
            for game in day_games
            for tid in (game.home_id, game.away_id)
        }
        if len(teams_today) != len(day_games) * 2:
            raise RuntimeError(
                f"Abstract schedule double-booked a team on slate day {day}"
            )

    return schedule


def games_for_team(schedule: List[GameSlot], team_id: str) -> int:
    """
    Return total regular-season games in the schedule for a given team.
    """
    team_id = str(team_id)

    return sum(
        1
        for game in schedule
        if not game.is_playoff
        and (game.home_id == team_id or game.away_id == team_id)
    )


def team_schedule(schedule: List[GameSlot], team_id: str) -> List[GameSlot]:
    """
    Extract all scheduled games for a given team, ordered by abstract slate day.
    """
    team_id = str(team_id)

    return [
        game
        for game in sorted(schedule, key=lambda g: (g.day, g.home_id, g.away_id))
        if game.home_id == team_id or game.away_id == team_id
    ]


# ---------------------------------------------------------------------------
# Optional diagnostics
# ---------------------------------------------------------------------------


def schedule_diagnostics(schedule: List[GameSlot]) -> Dict[str, Any]:
    """
    Debug helper for checking if the abstract schedule looks sane.
    Safe to call from tests or reports.
    """
    if not schedule:
        return {
            "games": 0,
            "slate_days": 0,
            "teams": 0,
            "games_per_team": {},
            "games_per_day_top": [],
            "worst_team_cadence": {},
        }

    teams = sorted(
        {
            tid
            for game in schedule
            for tid in (game.home_id, game.away_id)
        }
    )

    games_per_team_map: Dict[str, int] = {tid: 0 for tid in teams}
    team_days: Dict[str, List[int]] = {tid: [] for tid in teams}
    games_per_day: Dict[int, int] = {}

    for game in schedule:
        games_per_day[game.day] = games_per_day.get(game.day, 0) + 1

        for tid in (game.home_id, game.away_id):
            games_per_team_map[tid] += 1
            team_days[tid].append(game.day)

    worst: Dict[str, Any] = {}

    for tid, days in team_days.items():
        days = sorted(days)

        max_games_in_6 = 0
        max_games_in_10 = 0
        max_gap = 0

        for day in days:
            max_games_in_6 = max(
                max_games_in_6,
                sum(1 for d in days if day - 5 <= d <= day),
            )
            max_games_in_10 = max(
                max_games_in_10,
                sum(1 for d in days if day - 9 <= d <= day),
            )

        for a, b in zip(days, days[1:]):
            max_gap = max(max_gap, b - a)

        worst[tid] = {
            "games": len(days),
            "max_games_in_6_days": max_games_in_6,
            "max_games_in_10_days": max_games_in_10,
            "max_gap_between_games": max_gap,
        }

    busiest_days = sorted(
        [
            {"day": day, "games": count}
            for day, count in games_per_day.items()
        ],
        key=lambda row: row["games"],
        reverse=True,
    )[:15]

    return {
        "games": len(schedule),
        "slate_days": len(games_per_day),
        "teams": len(teams),
        "games_per_team": games_per_team_map,
        "games_per_day_top": busiest_days,
        "worst_team_cadence": worst,
    }