from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


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

    - Two weighted draws over all 16 non-playoff clubs (NHL odds table).
    - A draw winner moves up at most 10 spots; if it can't reach the drawn pick, it
      moves up exactly 10 and the drawn pick falls to the worst remaining club.
    - Everyone else keeps original order (worst first).
    """

    if len(teams) < 2:
        raise ValueError("Draft lottery requires at least two teams.")

    rng = random.Random(seed)
    ordered = list(teams)  # WORST → BEST
    n = len(ordered)
    odds = generate_lottery_odds(n)
    orig_rank = {str(t.team_id): i + 1 for i, t in enumerate(ordered)}

    if n == 16:
        # Current NHL format: two draws over all 16 clubs by their odds. A club may move
        # up at most 10 spots; a winner outside that range jumps exactly 10 places and the
        # drawn pick goes to the worst remaining club instead.
        assigned: dict[int, str] = {}
        winners: list[str] = []
        remaining = list(ordered)
        for pick_no in (1, 2):
            weights = [ODDS_PCT[orig_rank[str(t.team_id)] - 1] for t in remaining]
            won = _weighted_draw(remaining, weights, rng)
            wid = str(won.team_id)
            winners.append(wid)
            rank = orig_rank[wid]
            if rank - MAX_JUMP <= pick_no:
                assigned[pick_no] = wid
            else:
                assigned[rank - MAX_JUMP] = wid
            remaining = [t for t in remaining if str(t.team_id) != wid]
        rest = [str(t.team_id) for t in ordered if str(t.team_id) not in assigned.values()]
        final_order: list[str] = []
        for slot in range(1, n + 1):
            if slot in assigned:
                final_order.append(assigned[slot])
            else:
                final_order.append(rest.pop(0))
        _assert_max_jump(orig_rank, final_order)
        return LotteryResult(pick_order=final_order, lottery_winners=winners)

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
