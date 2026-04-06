"""
Trait generator.

Produces:
- trait tags (strings)
- numeric modifiers that other systems can read

All deterministic given `rng`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass(frozen=True)
class TraitProfile:
    tags: List[str]
    mods: Dict[str, float]


TRAIT_LIBRARY: List[Tuple[str, float, Dict[str, float]]] = [
    ("leader", 0.12, {"leadership": 0.15, "morale_baseline": 0.05}),
    ("locker_room_cancer", 0.04, {"morale_baseline": -0.10, "volatility": 0.10}),
    ("clutch", 0.08, {"clutch_factor": 0.18, "consistency": 0.04}),
    ("injury_prone", 0.06, {"injury_risk": 0.18, "durability": -0.08}),
    ("ironman", 0.05, {"injury_risk": -0.10, "durability": 0.10, "recovery_rate": 0.05}),
    ("coach_favorite", 0.07, {"coachability": 0.10, "discipline": 0.06}),
    ("fan_favorite", 0.06, {"morale_baseline": 0.06, "media_pressure": 0.05}),
    ("late_bloomer", 0.07, {"development_rate": 0.08, "volatility": 0.06}),
    ("high_compete", 0.10, {"competitiveness": 0.12, "work_ethic": 0.08}),
    ("lazy_practice", 0.05, {"work_ethic": -0.14, "development_rate": -0.06}),
    ("big_game_player", 0.07, {"clutch_factor": 0.10, "media_pressure": 0.04}),
    ("playoff_choker", 0.05, {"clutch_factor": -0.12, "volatility": 0.06}),
]


def generate_traits(rng) -> Dict[str, Any]:
    """
    Returns a dict for integration simplicity:
    {
      "tags": [...],
      "mods": {...}
    }
    """
    tags: List[str] = []
    mods: Dict[str, float] = {}

    # baseline chance for 1-3 traits
    for tag, p, m in TRAIT_LIBRARY:
        if rng.random() < p:
            tags.append(tag)
            for k, v in m.items():
                mods[k] = float(mods.get(k, 0.0) + float(v))

    # upbringing-driven hard traits (rare)
    if rng.random() < 0.10:
        tags.append("tough_upbringing")
        mods["resilience"] = float(mods.get("resilience", 0.0) + 0.10)
        mods["work_ethic"] = float(mods.get("work_ethic", 0.0) + 0.06)

    # medical edge cases (very rare)
    if rng.random() < 0.02:
        tags.append("chronic_condition")
        mods["injury_risk"] = float(mods.get("injury_risk", 0.0) + 0.12)

    # clamp some common modifiers to sensible ranges
    for k in ("morale_baseline", "injury_risk", "volatility", "development_rate", "clutch_factor"):
        if k in mods:
            mods[k] = float(max(-0.35, min(0.35, mods[k])))

    return {"tags": tags, "mods": mods}

