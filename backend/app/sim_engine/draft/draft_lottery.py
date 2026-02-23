from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


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
# ODDS GENERATION (NO HARDCODED TABLE)
# ==================================================

def generate_lottery_odds(num_teams: int) -> List[float]:
    """
    Worst team gets highest odds.
    Linear weighting, normalized.
    """
    weights = list(range(num_teams, 0, -1))
    total = sum(weights)
    return [w / total for w in weights]


# ==================================================
# CORE LOTTERY LOGIC
# ==================================================

def _draw_winner(
    teams: List[LotteryTeam],
    odds: List[float],
    rng: random.Random,
) -> LotteryTeam:
    roll = rng.random()
    cumulative = 0.0

    for team, weight in zip(teams, odds):
        cumulative += weight
        if roll <= cumulative:
            return team

    return teams[-1]  # safety fallback


def _can_move_up(original_index: int, new_pick_index: int) -> bool:
    """
    NHL rule: max move-up is 10 spots
    """
    return (original_index - new_pick_index) <= 10


# ==================================================
# PUBLIC API
# ==================================================

def run_draft_lottery(
    *,
    teams: List[LotteryTeam],
    seed: Optional[int] = None,
) -> LotteryResult:
    """
    Runs NHL-style draft lottery.
    Only determines picks #1 and #2.
    """

    if len(teams) < 2:
        raise ValueError("Draft lottery requires at least two teams.")

    rng = random.Random(seed)

    ordered = list(teams)  # WORST → BEST
    odds = generate_lottery_odds(len(ordered))

    winners: list[LotteryTeam] = []

    pick_index = 0
    while len(winners) < 2:
        winner = _draw_winner(ordered, odds, rng)
        original_index = teams.index(winner)

        if _can_move_up(original_index, pick_index):
            winners.append(winner)
            idx = ordered.index(winner)
            ordered.pop(idx)
            odds.pop(idx)

            total = sum(odds)
            odds = [o / total for o in odds]
            pick_index += 1
        else:
            # Remove temporarily and redraw
            idx = ordered.index(winner)
            ordered.pop(idx)
            odds.pop(idx)

            total = sum(odds)
            odds = [o / total for o in odds]

    # --------------------------------------------------
    # Build final draft order
    # --------------------------------------------------

    final_order: list[str] = []

    for w in winners:
        final_order.append(w.team_id)

    for t in teams:
        if t.team_id not in final_order:
            final_order.append(t.team_id)

    return LotteryResult(
        pick_order=final_order,
        lottery_winners=[w.team_id for w in winners],
    )
