#!/usr/bin/env python3
"""Dev regression: social copy must not collapse to identical template strings."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "SimEngine"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from app.sim_engine.franchise.social_copy_engine import compose_reporter_post  # noqa: E402

FIXED_STORYLINE = {
    "headline": "Star forward trade chatter intensifies",
    "player_name": "Alex Mercer",
    "team_name": "Vancouver",
    "team_id": "van",
    "player_id": "p1",
    "narrative_angle": "trade_market",
    "knowledge_type": "claim",
    "heat": 62,
    "evidence": {
        "points": 47,
        "goals": 19,
        "assists": 28,
        "games_played": 54,
        "ppg": 0.87,
        "overall": 86.5,
        "cap_hit": 7.25,
        "team_record": "28-22-4",
        "expected_points": 52,
    },
}

REPORTER = {
    "id": "ellison",
    "name": "Mark Ellison",
    "outlet": "NorthStar Hockey",
}


def _longest_shared_word_run(a: str, b: str) -> int:
    wa, wb = a.lower().split(), b.lower().split()
    best = 0
    for i in range(len(wa)):
        for j in range(len(wb)):
            k = 0
            while i + k < len(wa) and j + k < len(wb) and wa[i + k] == wb[j + k]:
                k += 1
            best = max(best, k)
    return best


def main() -> int:
    posts = []
    for i in range(200):
        posts.append(compose_reporter_post(FIXED_STORYLINE, REPORTER, random.Random(i * 9973 + i * i + 17), None))
    unique = len(set(posts))
    if unique != 200:
        print(f"FAIL: expected 200 unique posts, got {unique}")
        return 1
    pairs = 0
    low_overlap = 0
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            pairs += 1
            if _longest_shared_word_run(posts[i], posts[j]) < 6:
                low_overlap += 1
    ratio = low_overlap / pairs if pairs else 0
    if ratio < 0.80:
        print(f"FAIL: only {ratio:.1%} of pairs share fewer than 6 consecutive words (need 80%)")
        return 1
    print(f"OK: 200 unique reporter posts; {ratio:.1%} pairs under 6-word overlap threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
