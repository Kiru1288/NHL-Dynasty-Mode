"""Position-aware height/weight generation and body-to-ratings tradeoffs for prospects."""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

from app.sim_engine.entities.player import Position, clamp_rating


def _pos_key(position: Any) -> str:
    p = str(getattr(position, "value", position) or "").upper()
    if p == "G":
        return "G"
    if p in ("D", "LD", "RD", "LHD", "RHD"):
        return "D"
    if p in ("LW", "RW", "W", "C"):
        return "F"
    return "F"


def generate_position_height_cm(
    rng: random.Random,
    position: Any,
    *,
    archetype: str = "",
) -> int:
    """Realistic NHL-style height by position (cm)."""
    pos = _pos_key(position)
    arch = str(archetype or "").lower()

    if pos == "G":
        if rng.random() < 0.02:
            return rng.randint(183, 184)
        if rng.random() < 0.06:
            return rng.randint(200, 204)
        return rng.randint(185, 199)

    if pos == "D":
        if "mobile" in arch or "offensive" in arch:
            if rng.random() < 0.07:
                return rng.randint(181, 182)
            return rng.randint(183, 194)
        if "physical" in arch or "shutdown" in arch:
            return rng.randint(188, 198)
        roll = rng.random()
        if roll < 0.015:
            return rng.randint(178, 179)
        if roll < 0.88:
            return rng.randint(183, 196)
        return rng.randint(197, 201)

    # Forwards
    if "power" in arch:
        return rng.randint(188, 198)
    roll = rng.random()
    # Extremely small forwards are very rare.
    if roll < 0.006:
        return rng.randint(170, 175)
    if roll < 0.14:
        return rng.randint(176, 177)
    if roll < 0.88:
        return rng.randint(178, 191)
    return rng.randint(192, 198)


def generate_realistic_weight_kg(
    height_cm: int,
    position: Any,
    *,
    archetype: str = "",
    age: int = 18,
) -> int:
    """Weight derived from height, position, archetype, and age, anchored to real NHL builds.

    Uses ~0.85 kg/cm slopes off position-specific anchors (an NHL forward at 6'0"/183cm is
    ~194 lb, a D at 6'2"/188cm ~205 lb) plus a HEIGHT-SCALED floor so a tall prospect can
    never fall into toy-body territory (no more 6'5"/150 lb)."""
    pos = _pos_key(position)
    arch = str(archetype or "").lower()
    h = int(height_cm)

    if pos == "G":
        base = 84.0 + (h - 188) * 0.80
    elif pos == "D":
        base = 86.0 + (h - 183) * 0.85
        if "physical" in arch or "shutdown" in arch:
            base += 4
        if "mobile" in arch or "offensive" in arch:
            base -= 2
    else:
        base = 82.0 + (h - 180) * 0.85
        if "power" in arch:
            base += 5
        if "mobile" in arch or "speed" in arch or "playmak" in arch:
            base -= 2

    # Deterministic per-frame jitter so builds vary without breaking reproducibility.
    jitter = random.Random(h * 31 + len(arch) + int(pos == "D") * 3 + int(pos == "G") * 7).uniform(-4.5, 5.5)
    w = base + jitter

    if age <= 17:
        w -= 4.0
    elif age == 18:
        w -= 1.0
    elif age >= 20:
        w += 2.0

    # Height-scaled bounds: leanest believable build still tracks the frame.
    lean_floor = base - 9.0
    heavy_cap = base + 13.0
    # Absolute BMI-style lower bound anchored to height (a 198cm frame can't be ~68kg).
    abs_floor = (h - 100) * 0.80 + (6.0 if pos != "F" else 2.0)
    lo = max(lean_floor, abs_floor)
    hi = max(lo + 4.0, heavy_cap)
    return int(max(lo, min(hi, round(w))))


def apply_body_tradeoffs_to_ratings(player: Any, rng: Optional[random.Random] = None) -> None:
    """Adjust ratings for size — skill can offset penalties for small elite players."""
    if rng is None:
        rng = random.Random(abs(hash(str(getattr(player, "id", "")))) & 0xFFFFFFFF)

    ident = getattr(player, "identity", None)
    if ident is None:
        return
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict):
        return

    pos = _pos_key(getattr(ident, "position", "C"))
    h = int(getattr(ident, "height_cm", 0) or 0)
    w = int(getattr(ident, "weight_kg", 0) or 0)
    if h <= 0:
        return

    # Skating keys use the skg_ prefix (sk_ was a dead prefix that never matched).
    sk_keys = [k for k in ratings if k.startswith("skg_") or "skating" in str(k).lower()]
    phy_keys = [k for k in ratings if k.startswith("phy_") or "physical" in k or "strength" in k]
    agi_keys = [k for k in ratings if "agility" in k or "acceleration" in k]

    def bump(keys: list, delta: int) -> None:
        if not keys:
            return
        k = rng.choice(keys)
        ratings[k] = clamp_rating(int(ratings.get(k, 50)) + delta)

    if pos == "F":
        if h < 176:
            bump(sk_keys + agi_keys, rng.randint(0, 2))
            bump(phy_keys, -rng.randint(2, 4))
            if h < 174 or w < 70:
                bump(phy_keys, -rng.randint(1, 3))
                bump(sk_keys, -rng.randint(0, 2))
        elif h >= 193:
            bump(phy_keys, rng.randint(1, 2))
            if w > 100:
                bump(agi_keys, -rng.randint(0, 2))
    elif pos == "D":
        if h < 181:
            bump(sk_keys + agi_keys, rng.randint(0, 1))
            bump(phy_keys, -rng.randint(2, 4))
            if h < 180 or w < 74:
                bump(phy_keys, -rng.randint(1, 2))
        elif h >= 198:
            bump(phy_keys, rng.randint(1, 3))
            bump(agi_keys, -rng.randint(0, 2))
    elif pos == "G":
        if h < 184:
            bump(agi_keys, -rng.randint(1, 3))
            bump(phy_keys, -rng.randint(1, 2))
        elif h >= 200:
            bump(phy_keys, rng.randint(1, 2))
            bump(agi_keys, -rng.randint(0, 1))
