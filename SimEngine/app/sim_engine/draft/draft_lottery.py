from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ==================================================
# DATA MODELS
# ==================================================

@dataclass(frozen=True)
class LotteryTeam:
    team_id: str
    points: int


@dataclass
class LotteryResult:
    pick_order: List[str]
    lottery_winners: List[str]


# ==================================================
# NHL DRAFT LOTTERY ODDS (16 non-playoff teams, worst → best)
# ==================================================

ODDS_PCT: List[float] = [
    18.5, 13.5, 11.5, 9.5, 8.5, 7.5, 6.5, 6.0,
    5.0, 3.5, 3.0, 2.5, 2.0, 1.5, 0.5, 0.5,
]
COMBINATIONS: List[int] = [int(round(pct * 10)) for pct in ODDS_PCT]
MAX_JUMP = 10

# Fraction weights used by weighted draws (same curve as ODDS_PCT).
NHL_LOTTERY_WEIGHTS_16: List[float] = [pct / 100.0 for pct in ODDS_PCT]


def generate_lottery_odds(num_teams: int) -> List[float]:
    """
    Worst team gets highest odds.
    NHL table for 16 teams; otherwise linear weighting, normalized.
    """
    if num_teams == 16:
        return list(NHL_LOTTERY_WEIGHTS_16)
    weights = list(range(num_teams, 0, -1))
    total = sum(weights)
    return [w / total for w in weights]


def _weighted_draw(
    teams: Sequence[LotteryTeam],
    weights: Sequence[float],
    rng: random.Random,
) -> LotteryTeam:
    if not teams:
        raise ValueError("weighted_draw requires at least one team")
    if len(teams) == 1:
        return teams[0]
    total = float(sum(weights))
    if total <= 0:
        return teams[0]
    roll = rng.random() * total
    cumulative = 0.0
    for team, weight in zip(teams, weights):
        cumulative += float(weight)
        if roll <= cumulative:
            return team
    return teams[-1]


def _assert_max_jump(orig_rank: dict[str, int], final_order: Sequence[str]) -> None:
    for pick_num, tid in enumerate(final_order, start=1):
        rank = int(orig_rank[str(tid)])
        if pick_num < rank - MAX_JUMP:
            raise ValueError(
                f"Lottery violated max jump: team {tid} original rank {rank} won pick {pick_num}"
            )


# ==================================================
# PUBLIC API
# ==================================================

def run_draft_lottery(
    *,
    teams: List[LotteryTeam],
    seed: Optional[int] = None,
) -> LotteryResult:
    """
    Runs NHL-style draft lottery for picks #1 and #2.

    - Pick #1: weighted draw among bottom-11 teams (ranks 1–11).
    - Pick #2: weighted draw among remaining teams with original rank <= 12.
    - Picks 3–16: remaining teams in original-rank order (worst first).
    """

    if len(teams) < 2:
        raise ValueError("Draft lottery requires at least two teams.")

    rng = random.Random(seed)
    ordered = list(teams)  # WORST → BEST
    n = len(ordered)
    odds = generate_lottery_odds(n)
    orig_rank = {str(t.team_id): i + 1 for i, t in enumerate(ordered)}

    if n == 16:
        # Pick #1 — only ranks 1..11 eligible (12+ would need an 11+ spot jump).
        pool1 = ordered[:11]
        weights1 = ODDS_PCT[:11]
        winner1 = _weighted_draw(pool1, weights1, rng)

        remaining = [t for t in ordered if t.team_id != winner1.team_id]
        pool2 = [t for t in remaining if orig_rank[str(t.team_id)] <= 12]
        weights2 = [ODDS_PCT[orig_rank[str(t.team_id)] - 1] for t in pool2]
        winner2 = _weighted_draw(pool2, weights2, rng)

        winners = [winner1, winner2]
        winner_ids = {str(w.team_id) for w in winners}
        rest = [t for t in ordered if str(t.team_id) not in winner_ids]
        rest.sort(key=lambda t: orig_rank[str(t.team_id)])
        final_order = [str(winner1.team_id), str(winner2.team_id)] + [str(t.team_id) for t in rest]
        _assert_max_jump(orig_rank, final_order)
        return LotteryResult(
            pick_order=final_order,
            lottery_winners=[str(winner1.team_id), str(winner2.team_id)],
        )

    # Non-16-team fallback: legacy two-draw with max-jump guard.
    winners: list[LotteryTeam] = []
    working = list(ordered)
    working_odds = list(odds)
    pick_index = 0
    while len(winners) < 2 and working:
        winner = _weighted_draw(working, working_odds, rng)
        original_index = ordered.index(winner)
        if (original_index - pick_index) <= MAX_JUMP:
            winners.append(winner)
            pick_index += 1
        idx = working.index(winner)
        working.pop(idx)
        working_odds.pop(idx)
        total = sum(working_odds)
        if total > 0:
            working_odds = [o / total for o in working_odds]

    final_order: list[str] = [str(w.team_id) for w in winners]
    for t in ordered:
        if str(t.team_id) not in final_order:
            final_order.append(str(t.team_id))
    _assert_max_jump(orig_rank, final_order)
    return LotteryResult(
        pick_order=final_order,
        lottery_winners=[str(w.team_id) for w in winners[:2]],
    )
