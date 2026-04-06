from __future__ import annotations

"""
Draft class generator (prospects) using the full human generation system.

Outputs a list of dict "player profiles" with structured pipeline tiers so each class
has realistic franchise / elite / top / middle / depth counts (weak vs stacked years).

This is runner/engine friendly and avoids assuming a specific Player constructor.
"""

import hashlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.generation.name_generator import generate_human_identity
from app.sim_engine.generation.attribute_generator import (
    Archetype,
    generate_backstory,
    generate_attributes,
    generate_health,
    compute_development_rate,
)
from app.sim_engine.generation.trait_generator import generate_traits


def _stable_id_from_seed(seed: int, salt: str) -> str:
    b = f"{seed}:{salt}".encode("utf-8")
    return "GEN_" + hashlib.sha1(b).hexdigest()[:14].upper()


def _choose_position(rng) -> str:
    # NHL-ish distribution: F (C/W) ~ 60%, D ~ 30%, G ~ 10%
    pos = rng.choices(["C", "W", "D", "G"], weights=[0.26, 0.34, 0.30, 0.10], k=1)[0]
    return pos


def _choose_archetype(rng, position: str) -> Archetype:
    if position == "G":
        return Archetype.GOALIE
    if position == "D":
        return rng.choices(
            [Archetype.OFFENSIVE_DEFENSEMAN, Archetype.DEFENSIVE_DEFENSEMAN, Archetype.SHUTDOWN_DEFENSEMAN, Archetype.PUCK_MOVING_DEFENSEMAN],
            weights=[0.22, 0.34, 0.22, 0.22],
            k=1,
        )[0]
    # forward
    return rng.choices(
        [Archetype.SNIPER, Archetype.PLAYMAKER, Archetype.POWER_FORWARD, Archetype.TWO_WAY_FORWARD, Archetype.GRINDER, Archetype.HYBRID_FORWARD],
        weights=[0.18, 0.18, 0.14, 0.24, 0.12, 0.14],
        k=1,
    )[0]


def _talent_tier(rng) -> Tuple[str, float]:
    """
    Returns (tier_name, talent_scalar).
    talent_scalar is the base "skill level" for attribute generation.
    """
    roll = rng.random()
    if roll < 0.03:
        return "franchise", rng.uniform(0.78, 0.90)
    if roll < 0.13:
        return "elite", rng.uniform(0.70, 0.82)
    if roll < 0.55:
        return "normal", rng.uniform(0.58, 0.72)
    if roll < 0.85:
        return "project", rng.uniform(0.50, 0.64)
    return "longshot", rng.uniform(0.42, 0.58)


def _pipeline_talent_scalar(pipeline_tier: str, rng) -> Tuple[str, float]:
    """Map pipeline slot to (attribute-gen tier label, talent scalar)."""
    pt = str(pipeline_tier or "middle").lower()
    if pt == "franchise":
        return "franchise", rng.uniform(0.87, 0.95)
    if pt == "elite":
        return "elite", rng.uniform(0.78, 0.91)
    if pt == "top":
        return "normal", rng.uniform(0.70, 0.86)
    if pt == "middle":
        return "normal", rng.uniform(0.60, 0.77)
    return "project", rng.uniform(0.48, 0.68)


def _potential_ceiling_for_pipeline(pipeline_tier: str, rng, *, bust: bool) -> float:
    pt = str(pipeline_tier or "middle").lower()
    if pt == "franchise":
        ce = rng.uniform(0.92, 0.97)
    elif pt == "elite":
        ce = rng.uniform(0.86, 0.93)
    elif pt == "top":
        ce = rng.uniform(0.80, 0.88)
    elif pt == "middle":
        ce = rng.uniform(0.75, 0.82)
    else:
        ce = rng.uniform(0.62, 0.78)
    if bust:
        ce -= rng.uniform(0.045, 0.13)
    return max(0.52, min(0.99, ce))


def _build_skater_pipeline_slots(rng: random.Random, sk: int, *, elite_boost: int = 0) -> Tuple[List[str], str]:
    """Return shuffled list of pipeline tier tags (length sk) and quality label."""
    if sk <= 0:
        return [], "empty"
    q = rng.random()
    if q < 0.11:
        qual = "weak"
    elif q < 0.67:
        qual = "normal"
    elif q < 0.90:
        qual = "strong"
    else:
        qual = "legendary"
    scale = max(0.42, min(1.35, sk / 210.0))

    def rn(a: int, b: int) -> int:
        return max(0, int(round(rng.randint(a, b) * scale)))

    if qual == "weak":
        fc, el, tp, md = rn(0, 1), rn(4, 7), rn(12, 20), rn(36, 52)
    elif qual == "normal":
        fc, el, tp, md = rn(1, 2), rn(6, 10), rn(16, 25), rn(40, 58)
    elif qual == "strong":
        fc, el, tp, md = rn(2, 3), rn(8, 11), rn(18, 28), rn(44, 62)
    else:
        fc, el, tp, md = rn(2, 3), rn(9, 13), rn(22, 32), rn(48, 66)

    el += max(0, int(elite_boost))
    slots: List[str] = ["franchise"] * fc + ["elite"] * el + ["top"] * tp + ["middle"] * md
    short = sk - len(slots)
    if short > 0:
        slots.extend(["depth"] * short)
    while len(slots) > sk:
        removed = False
        for lab in ("depth", "middle", "top", "elite"):
            if lab in slots:
                slots.remove(lab)
                removed = True
                break
        if not removed:
            if "franchise" in slots and slots.count("franchise") > 1:
                slots.remove("franchise")
            elif slots:
                slots.pop()
            else:
                break
    rng.shuffle(slots)
    return slots, qual


def generate_player_profile(
    rng,
    *,
    age: int,
    forced_nationality: Optional[str] = None,
    seed_hint: Optional[int] = None,
) -> Dict[str, Any]:
    ident = generate_human_identity(rng, nationality=forced_nationality)
    pos = _choose_position(rng)
    arche = _choose_archetype(rng, pos)
    tier, talent = _talent_tier(rng)

    backstory = generate_backstory(rng, nationality=ident.nationality, hometown=ident.hometown)
    traits = generate_traits(rng)
    attrs = generate_attributes(rng, archetype=arche, backstory=backstory, talent=talent)
    health = generate_health(rng, archetype=arche, backstory=backstory)
    dev_rate = compute_development_rate(backstory=backstory, traits=traits, talent=talent)

    # boom/bust markers
    boom_bust = rng.random()
    if tier in ("project", "longshot") and boom_bust < 0.18:
        traits["tags"].append("boom_bust")
        traits["mods"]["volatility"] = float(traits["mods"].get("volatility", 0.0) + 0.12)
        dev_rate = min(0.95, dev_rate + 0.04)

    # late bloomer
    if rng.random() < 0.12:
        if "late_bloomer" not in traits["tags"]:
            traits["tags"].append("late_bloomer")
        traits["mods"]["development_rate"] = float(traits["mods"].get("development_rate", 0.0) + 0.06)
        dev_rate = min(0.95, dev_rate + 0.03)

    seed_val = int(seed_hint if seed_hint is not None else rng.randrange(1, 2_000_000_000))
    pid = _stable_id_from_seed(seed_val, ident.full_name)

    return {
        "id": pid,
        "name": ident.full_name,
        "nationality": ident.nationality,
        "hometown": ident.hometown,
        "nickname": ident.nickname,
        "pronunciation": ident.pronunciation,
        "age": int(age),
        "position": pos,
        "archetype": arche.value,
        "talent_tier": tier,
        "attributes": attrs,
        "traits": traits,
        "backstory": {
            "family": backstory.family,
            "childhood": backstory.childhood,
            "psychology": backstory.psychology,
            "education": backstory.education,
            "development_path": backstory.development_path,
            "adversity": backstory.adversity,
            "social": backstory.social,
            "impact": asdict(backstory.impact),
        },
        "health": health,
        "development_rate": float(dev_rate),
    }


def generate_player_profile_for_pipeline_slot(
    rng,
    *,
    age: int,
    position: str,
    pipeline_tier: str,
    seed_hint: Optional[int] = None,
    forced_nationality: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate one profile for a fixed position and structured pipeline tier (skater or goalie)."""
    pt_in = str(pipeline_tier or "middle").lower()
    pt = pt_in
    steal = pt == "depth" and rng.random() < 0.036
    if steal:
        pt = "top"
    bust = False
    if pt in ("franchise", "elite", "top", "middle") and rng.random() < rng.uniform(0.10, 0.22):
        bust = True

    ident = generate_human_identity(rng, nationality=forced_nationality)
    arche = _choose_archetype(rng, position)
    tier, talent = _pipeline_talent_scalar(pt, rng)
    if bust:
        talent *= rng.uniform(0.86, 0.96)

    backstory = generate_backstory(rng, nationality=ident.nationality, hometown=ident.hometown)
    traits = generate_traits(rng)
    attrs = generate_attributes(rng, archetype=arche, backstory=backstory, talent=talent)
    health = generate_health(rng, archetype=arche, backstory=backstory)
    dev_rate = compute_development_rate(backstory=backstory, traits=traits, talent=talent)
    if bust:
        dev_rate = max(0.22, dev_rate * rng.uniform(0.82, 0.94))

    if tier in ("project", "longshot") and rng.random() < 0.16:
        traits["tags"].append("boom_bust")
        traits["mods"]["volatility"] = float(traits["mods"].get("volatility", 0.0) + 0.10)
        dev_rate = min(0.95, dev_rate + 0.035)

    if rng.random() < 0.11:
        if "late_bloomer" not in traits["tags"]:
            traits["tags"].append("late_bloomer")
        traits["mods"]["development_rate"] = float(traits["mods"].get("development_rate", 0.0) + 0.05)
        dev_rate = min(0.95, dev_rate + 0.025)

    seed_val = int(seed_hint if seed_hint is not None else rng.randrange(1, 2_000_000_000))
    pid = _stable_id_from_seed(seed_val, ident.full_name)
    ceiling = _potential_ceiling_for_pipeline(pt, rng, bust=bust)

    return {
        "id": pid,
        "name": ident.full_name,
        "nationality": ident.nationality,
        "hometown": ident.hometown,
        "nickname": ident.nickname,
        "pronunciation": ident.pronunciation,
        "age": int(age),
        "position": position,
        "archetype": arche.value,
        "talent_tier": tier,
        "pipeline_tier": pt,
        "potential_ceiling_0_1": float(ceiling),
        "pipeline_bust": bool(bust),
        "pipeline_steal": bool(steal),
        "attributes": attrs,
        "traits": traits,
        "backstory": {
            "family": backstory.family,
            "childhood": backstory.childhood,
            "psychology": backstory.psychology,
            "education": backstory.education,
            "development_path": backstory.development_path,
            "adversity": backstory.adversity,
            "social": backstory.social,
            "impact": asdict(backstory.impact),
        },
        "health": health,
        "development_rate": float(dev_rate),
    }


def generate_player_profile_forced_position(rng, *, age: int, position: str, **kwargs: Any) -> Dict[str, Any]:
    """Generate a profile with a specific position (e.g. G for goalies)."""
    pipeline = kwargs.get("pipeline_tier")
    if pipeline is not None:
        return generate_player_profile_for_pipeline_slot(
            rng,
            age=age,
            position=position,
            pipeline_tier=str(pipeline),
            seed_hint=kwargs.get("seed_hint"),
            forced_nationality=kwargs.get("forced_nationality"),
        )
    ident = generate_human_identity(rng, nationality=kwargs.get("forced_nationality"))
    arche = _choose_archetype(rng, position)
    tier, talent = _talent_tier(rng)
    backstory = generate_backstory(rng, nationality=ident.nationality, hometown=ident.hometown)
    traits = generate_traits(rng)
    attrs = generate_attributes(rng, archetype=arche, backstory=backstory, talent=talent)
    health = generate_health(rng, archetype=arche, backstory=backstory)
    dev_rate = compute_development_rate(backstory=backstory, traits=traits, talent=talent)
    seed_val = int(kwargs.get("seed_hint") or rng.randrange(1, 2_000_000_000))
    pid = _stable_id_from_seed(seed_val, ident.full_name)
    return {
        "id": pid,
        "name": ident.full_name,
        "nationality": ident.nationality,
        "hometown": ident.hometown,
        "nickname": ident.nickname,
        "pronunciation": ident.pronunciation,
        "age": int(age),
        "position": position,
        "archetype": arche.value,
        "talent_tier": tier,
        "attributes": attrs,
        "traits": traits,
        "backstory": {
            "family": backstory.family,
            "childhood": backstory.childhood,
            "psychology": backstory.psychology,
            "education": backstory.education,
            "development_path": backstory.development_path,
            "adversity": backstory.adversity,
            "social": backstory.social,
            "impact": asdict(backstory.impact),
        },
        "health": health,
        "development_rate": float(dev_rate),
    }


def generate_draft_class(
    rng,
    size: int = 217,
    *,
    year: int = 2025,
    goalie_count: int = 7,
    pipeline_boost_elite: int = 0,
) -> List[Dict[str, Any]]:
    """
    Draft class of prospects (age 18–20). Structured pipeline tiers per class (weak/normal/strong/legendary).
    Ensures exactly goalie_count goalies; rest are skaters (C/W/D).
    """
    if goalie_count > size:
        goalie_count = max(0, size // 10)
    sk = max(0, size - goalie_count)
    slots, qual = _build_skater_pipeline_slots(rng, sk, elite_boost=max(0, int(pipeline_boost_elite)))
    players: List[Dict[str, Any]] = []
    for i in range(goalie_count):
        age = rng.choices([18, 19, 20], weights=[0.64, 0.28, 0.08], k=1)[0]
        g_tier = rng.choices(["elite", "top", "middle", "middle"], weights=[0.18, 0.28, 0.42, 0.12], k=1)[0]
        prof = generate_player_profile_forced_position(
            rng,
            age=age,
            position="G",
            seed_hint=year * 10000 + 9000 + i,
            pipeline_tier=g_tier,
        )
        prof["_draft_class_quality"] = qual
        players.append(prof)
    for i, slot in enumerate(slots):
        age = rng.choices([18, 19, 20], weights=[0.64, 0.28, 0.08], k=1)[0]
        pos = rng.choices(["C", "W", "D"], weights=[0.26, 0.34, 0.40], k=1)[0]
        prof = generate_player_profile_for_pipeline_slot(
            rng,
            age=age,
            position=pos,
            pipeline_tier=slot,
            seed_hint=year * 1000 + i,
        )
        prof["_draft_class_quality"] = qual
        players.append(prof)
    return players
