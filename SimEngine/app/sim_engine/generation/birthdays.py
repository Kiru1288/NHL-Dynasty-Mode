"""Birthdays for generated players.

Hockey birthdays skew early in the year (relative age effect): more January to
March births, fewer in the autumn. The weights are per month.
"""

from __future__ import annotations

import calendar
import random
from typing import Any, Tuple

_MONTH_WEIGHTS = (11, 10, 10, 9, 9, 8, 8, 7, 6, 6, 5, 5)


def random_birthday(rng: Any, birth_year: int) -> Tuple[int, int]:
    """(month, day) for a player born in ``birth_year``."""
    if not isinstance(rng, random.Random):
        rng = random.Random(rng)
    month = rng.choices(range(1, 13), weights=_MONTH_WEIGHTS, k=1)[0]
    days = calendar.monthrange(int(birth_year) if int(birth_year) > 0 else 2001, month)[1]
    return month, rng.randint(1, days)


def stamp_birthday(identity: Any, rng: Any) -> None:
    """Give an identity a real birthday unless it already has one."""
    if identity is None:
        return
    if int(getattr(identity, "birth_month", 0) or 0) and int(getattr(identity, "birth_day", 0) or 0):
        return
    month, day = random_birthday(rng, int(getattr(identity, "birth_year", 0) or 0))
    try:
        identity.birth_month = month
        identity.birth_day = day
    except Exception:
        pass
