"""
Archetypes, backstory, attributes, health, and development calculations.

This module intentionally holds multiple generators because the repository
currently only contains a subset of originally planned generation files.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _gauss01(rng, mu: float, sigma: float) -> float:
    # bounded gaussian in [0,1]
    return _clamp(rng.gauss(mu, sigma), 0.0, 1.0)


class Archetype(str, Enum):
    SNIPER = "sniper"
    PLAYMAKER = "playmaker"
    POWER_FORWARD = "power_forward"
    TWO_WAY_FORWARD = "two_way_forward"
    GRINDER = "grinder"
    ENFORCER = "enforcer"
    OFFENSIVE_DEFENSEMAN = "offensive_defenseman"
    DEFENSIVE_DEFENSEMAN = "defensive_defenseman"
    SHUTDOWN_DEFENSEMAN = "shutdown_defenseman"
    PUCK_MOVING_DEFENSEMAN = "puck_moving_defenseman"
    HYBRID_FORWARD = "hybrid_forward"
    GOALIE = "goalie"


ARCHETYPE_WEIGHTS: Dict[Archetype, Dict[str, float]] = {
    Archetype.SNIPER: {"shot_power": 0.18, "shot_accuracy": 0.22, "creativity": 0.10, "puck_control": 0.10, "skating": 0.10},
    Archetype.PLAYMAKER: {"playmaking": 0.22, "vision": 0.18, "decision_making": 0.12, "puck_control": 0.10, "agility": 0.08},
    Archetype.POWER_FORWARD: {"strength": 0.18, "checking": 0.16, "balance": 0.14, "endurance": 0.10, "shot_power": 0.08},
    Archetype.TWO_WAY_FORWARD: {"positioning": 0.14, "stick_check": 0.10, "hockey_iq": 0.12, "playmaking": 0.08, "skating": 0.08},
    Archetype.GRINDER: {"endurance": 0.16, "checking": 0.12, "discipline": 0.10, "competitiveness": 0.12, "strength": 0.08},
    Archetype.ENFORCER: {"strength": 0.22, "checking": 0.18, "balance": 0.10, "discipline": -0.08, "hockey_iq": -0.06},
    Archetype.OFFENSIVE_DEFENSEMAN: {"puck_control": 0.12, "playmaking": 0.14, "creativity": 0.12, "shot_power": 0.08, "skating": 0.08},
    Archetype.DEFENSIVE_DEFENSEMAN: {"positioning": 0.18, "shot_blocking": 0.12, "stick_check": 0.12, "decision_making": 0.08, "strength": 0.06},
    Archetype.SHUTDOWN_DEFENSEMAN: {"positioning": 0.22, "stick_check": 0.14, "shot_blocking": 0.12, "balance": 0.06, "discipline": 0.06},
    Archetype.PUCK_MOVING_DEFENSEMAN: {"skating": 0.14, "agility": 0.10, "playmaking": 0.12, "vision": 0.10, "puck_control": 0.10},
    Archetype.HYBRID_FORWARD: {"skating": 0.10, "shooting": 0.10, "playmaking": 0.10, "defense": 0.08, "competitiveness": 0.08},
    Archetype.GOALIE: {"hockey_iq": 0.10, "decision_making": 0.10, "competitiveness": 0.10, "balance": 0.08},
}


# ---------------------------------------------------------------------------
# Backstory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackstoryImpact:
    development_rate: float
    morale_baseline: float
    injury_risk: float
    volatility: float
    work_ethic: float
    leadership: float
    consistency: float
    clutch_factor: float


@dataclass(frozen=True)
class PlayerBackstory:
    family: Dict[str, Any]
    childhood: Dict[str, Any]
    psychology: Dict[str, Any]
    education: Dict[str, Any]
    development_path: Dict[str, Any]
    adversity: Dict[str, Any]
    social: Dict[str, Any]
    impact: BackstoryImpact


def generate_backstory(rng, *, nationality: str, hometown: str) -> PlayerBackstory:
    # Family layer
    income_class = rng.choices(
        ["low_income", "working_class", "middle_class", "upper_middle", "wealthy"],
        weights=[0.10, 0.38, 0.35, 0.13, 0.04],
        k=1,
    )[0]
    stability = rng.choices(["unstable", "mixed", "stable"], weights=[0.14, 0.30, 0.56], k=1)[0]
    immigrant = rng.random() < (0.10 if nationality in ("Canada", "USA", "Germany", "France", "UK", "Australia", "New Zealand") else 0.05)
    siblings = int(rng.choices([0, 1, 2, 3, 4], weights=[0.10, 0.30, 0.34, 0.18, 0.08], k=1)[0])
    parent_jobs = rng.sample(
        ["teacher", "nurse", "mechanic", "engineer", "small_business_owner", "truck_driver", "police", "military", "accountant", "construction"],
        k=2,
    )

    # Childhood access
    environment = rng.choices(["urban", "suburban", "rural"], weights=[0.32, 0.50, 0.18], k=1)[0]
    access_to_ice = rng.choices(["limited", "moderate", "excellent"], weights=[0.22, 0.48, 0.30], k=1)[0]
    first_exposure = rng.choices(["family", "school", "local_club", "street/pond", "tv/hero"], weights=[0.28, 0.12, 0.34, 0.18, 0.08], k=1)[0]
    other_sports = rng.sample(["soccer", "basketball", "baseball", "lacrosse", "rugby", "track", "martial_arts", "swimming", "none"], k=2)

    # Psychology
    motivation = rng.choices(["legacy", "joy", "escape", "money", "family_pride"], weights=[0.18, 0.28, 0.18, 0.12, 0.24], k=1)[0]
    resilience = _gauss01(rng, 0.55, 0.18)
    confidence = _gauss01(rng, 0.52, 0.18)
    coachability = _gauss01(rng, 0.55, 0.16)
    discipline = _gauss01(rng, 0.52, 0.16)

    # Education & path
    school_type = rng.choices(["public", "private", "sports_academy"], weights=[0.72, 0.18, 0.10], k=1)[0]
    academic = _gauss01(rng, 0.52, 0.18)
    path = rng.choices(["junior", "college", "european_pro", "mixed"], weights=[0.55, 0.20, 0.18, 0.07], k=1)[0]

    # Development path
    dev_path = rng.choices(["pond_hockey", "elite_academy", "local_club", "late_bloomer"], weights=[0.26, 0.18, 0.44, 0.12], k=1)[0]

    # Adversity
    adversity_flags = {
        "youth_injury": rng.random() < 0.14,
        "family_tragedy": rng.random() < 0.05,
        "financial_hardship": income_class in ("low_income", "working_class") and rng.random() < 0.22,
        "discrimination": nationality in ("China", "Japan", "South Korea", "India", "Nigeria", "Kenya", "Philippines", "Mexico", "Brazil", "Argentina") and rng.random() < 0.10,
        "coaching_abuse": rng.random() < 0.03,
        "relocation": rng.random() < 0.09,
    }

    media_pressure = _gauss01(rng, 0.45, 0.22)
    market_size = rng.choices(["small", "medium", "large"], weights=[0.45, 0.40, 0.15], k=1)[0]

    # --- Impacts (what the sim uses) ---
    dev = 0.50
    morale = 0.50
    inj = 0.12
    vol = 0.50
    work = 0.50
    lead = 0.50
    cons = 0.50
    clutch = 0.50

    if dev_path == "pond_hockey":
        dev += 0.05
        vol += 0.06
        clutch += 0.03
    if dev_path == "elite_academy":
        dev += 0.07
        cons += 0.06
        vol -= 0.04
    if "track" in other_sports or "martial_arts" in other_sports or "swimming" in other_sports:
        dev += 0.03
        inj -= 0.02
        cons += 0.03
    if adversity_flags["financial_hardship"]:
        work += 0.06
        morale -= 0.03
        clutch += 0.03
    if adversity_flags["family_tragedy"]:
        resilience = _clamp(resilience + 0.10)
        vol = _clamp(vol + 0.08)
        morale -= 0.05
    if adversity_flags["youth_injury"]:
        inj += 0.06
        cons -= 0.03
    if adversity_flags["coaching_abuse"]:
        coachability = _clamp(coachability - 0.08)
        vol = _clamp(vol + 0.10)
        morale -= 0.05
    if stability == "stable":
        morale += 0.05
        cons += 0.03
    if stability == "unstable":
        vol += 0.08
        morale -= 0.06
    if immigrant:
        work += 0.03
        lead += 0.02

    # income access affects development resources
    if income_class in ("upper_middle", "wealthy"):
        dev += 0.04
        morale += 0.02
    if income_class == "low_income":
        dev -= 0.02
        work += 0.04

    # media pressure increases volatility; big markets can boost clutch slightly
    vol += (media_pressure - 0.5) * 0.10
    if market_size == "large":
        clutch += 0.03
        morale -= 0.01

    impact = BackstoryImpact(
        development_rate=_clamp(dev, 0.20, 0.90),
        morale_baseline=_clamp(morale, 0.15, 0.85),
        injury_risk=_clamp(inj, 0.02, 0.40),
        volatility=_clamp(vol, 0.05, 0.95),
        work_ethic=_clamp(work, 0.05, 0.95),
        leadership=_clamp(lead, 0.05, 0.95),
        consistency=_clamp(cons, 0.05, 0.95),
        clutch_factor=_clamp(clutch, 0.05, 0.95),
    )

    return PlayerBackstory(
        family={
            "parents_occupations": parent_jobs,
            "income_class": income_class,
            "siblings": siblings,
            "stability": stability,
            "immigrant_status": immigrant,
        },
        childhood={
            "hometown": hometown,
            "environment": environment,
            "access_to_ice": access_to_ice,
            "first_exposure_to_hockey": first_exposure,
            "other_sports": other_sports,
        },
        psychology={
            "motivation_type": motivation,
            "resilience": resilience,
            "confidence": confidence,
            "coachability": coachability,
            "discipline": discipline,
        },
        education={
            "school_type": school_type,
            "academic_ability": academic,
            "path": path,
        },
        development_path={"type": dev_path},
        adversity=adversity_flags,
        social={
            "media_pressure": media_pressure,
            "market_size": market_size,
            "cultural_identity": nationality,
        },
        impact=impact,
    )


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

ATTRIBUTE_KEYS: Tuple[str, ...] = (
    # skating
    "skating","speed","agility","balance",
    # shooting
    "shooting","shot_power","shot_accuracy",
    # offense
    "offense","playmaking","creativity","puck_control",
    # defense
    "defense","stick_check","positioning","shot_blocking",
    # physical
    "physical","strength","checking","endurance",
    # mental
    "mental","hockey_iq","vision","decision_making","competitiveness",
)


def _base_attributes(rng, *, talent: float) -> Dict[str, float]:
    # talent is ~[0.3..0.95]
    attrs: Dict[str, float] = {}
    for k in ATTRIBUTE_KEYS:
        attrs[k] = _clamp(_gauss01(rng, mu=talent, sigma=0.10), 0.05, 0.98)
    return attrs


def _apply_archetype(attrs: Dict[str, float], archetype: Archetype) -> None:
    w = ARCHETYPE_WEIGHTS.get(archetype, {})
    for k, delta in w.items():
        # map composite keys to likely sub-keys
        if k in attrs:
            attrs[k] = _clamp(attrs[k] + delta)
        elif k == "skating":
            attrs["skating"] = _clamp(attrs["skating"] + delta)
            attrs["speed"] = _clamp(attrs["speed"] + delta * 0.6)
            attrs["agility"] = _clamp(attrs["agility"] + delta * 0.6)


def _apply_backstory(attrs: Dict[str, float], backstory: PlayerBackstory, rng) -> None:
    imp = backstory.impact
    dev_path = backstory.development_path.get("type", "")

    # pond hockey => creativity bump, structure penalty
    if dev_path == "pond_hockey":
        attrs["creativity"] = _clamp(attrs["creativity"] + 0.08)
        attrs["decision_making"] = _clamp(attrs["decision_making"] - 0.03)
        attrs["puck_control"] = _clamp(attrs["puck_control"] + 0.05)
    # elite academy => structure and IQ
    if dev_path == "elite_academy":
        attrs["decision_making"] = _clamp(attrs["decision_making"] + 0.06)
        attrs["positioning"] = _clamp(attrs["positioning"] + 0.05)
        attrs["creativity"] = _clamp(attrs["creativity"] - 0.03)

    other_sports = backstory.childhood.get("other_sports", []) or []
    if "track" in other_sports:
        attrs["speed"] = _clamp(attrs["speed"] + 0.05)
        attrs["endurance"] = _clamp(attrs["endurance"] + 0.04)
    if "martial_arts" in other_sports:
        attrs["balance"] = _clamp(attrs["balance"] + 0.05)
        attrs["competitiveness"] = _clamp(attrs["competitiveness"] + 0.03)
    if "soccer" in other_sports:
        attrs["agility"] = _clamp(attrs["agility"] + 0.04)
        attrs["vision"] = _clamp(attrs["vision"] + 0.02)

    # adversity -> resilience/compete + volatility
    if backstory.adversity.get("financial_hardship", False):
        attrs["competitiveness"] = _clamp(attrs["competitiveness"] + 0.03)
    if backstory.adversity.get("youth_injury", False):
        attrs["endurance"] = _clamp(attrs["endurance"] - 0.02)
    if backstory.adversity.get("coaching_abuse", False):
        attrs["decision_making"] = _clamp(attrs["decision_making"] - 0.02)

    # work ethic and consistency influence "mental" composites
    attrs["mental"] = _clamp(attrs["mental"] + (imp.work_ethic - 0.5) * 0.06 + (imp.consistency - 0.5) * 0.04)

    # tiny noise so players don't look same-y
    for k in ("shot_accuracy", "playmaking", "stick_check", "hockey_iq"):
        attrs[k] = _clamp(attrs[k] + rng.uniform(-0.02, 0.02))


def generate_attributes(rng, *, archetype: Archetype, backstory: PlayerBackstory, talent: float) -> Dict[str, float]:
    attrs = _base_attributes(rng, talent=talent)
    _apply_archetype(attrs, archetype)
    _apply_backstory(attrs, backstory, rng)

    # derive some composites from components (keep consistent)
    attrs["skating"] = _clamp((attrs["speed"] + attrs["agility"] + attrs["balance"]) / 3.0)
    attrs["shooting"] = _clamp((attrs["shot_power"] + attrs["shot_accuracy"]) / 2.0)
    attrs["offense"] = _clamp((attrs["playmaking"] + attrs["creativity"] + attrs["puck_control"]) / 3.0)
    attrs["defense"] = _clamp((attrs["stick_check"] + attrs["positioning"] + attrs["shot_blocking"]) / 3.0)
    attrs["physical"] = _clamp((attrs["strength"] + attrs["checking"] + attrs["endurance"] + attrs["balance"]) / 4.0)
    attrs["mental"] = _clamp((attrs["hockey_iq"] + attrs["vision"] + attrs["decision_making"] + attrs["competitiveness"]) / 4.0)
    return attrs


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealthProfile:
    durability: float
    injury_risk: float
    recovery_rate: float
    concussion_risk: float


def generate_health(rng, *, archetype: Archetype, backstory: PlayerBackstory, body_type: Optional[str] = None) -> Dict[str, float]:
    bt = body_type or rng.choices(["lean", "average", "power"], weights=[0.34, 0.46, 0.20], k=1)[0]
    base_dur = _gauss01(rng, 0.55, 0.14)
    base_inj = _gauss01(rng, 0.12, 0.06)
    rec = _gauss01(rng, 0.55, 0.14)
    conc = _gauss01(rng, 0.10, 0.06)

    # body type
    if bt == "lean":
        base_dur += 0.02
        conc -= 0.01
    if bt == "power":
        base_dur += 0.03
        base_inj += 0.02
        conc += 0.02

    # play style: enforcer/power forward takes more contact
    if archetype in (Archetype.ENFORCER, Archetype.POWER_FORWARD, Archetype.GRINDER):
        base_inj += 0.03
        conc += 0.03
    if archetype in (Archetype.SNIPER, Archetype.PLAYMAKER):
        conc -= 0.01

    # backstory adversity / youth injuries
    base_inj += (backstory.impact.injury_risk - 0.12) * 0.50
    if backstory.adversity.get("youth_injury", False):
        rec -= 0.03

    hp = HealthProfile(
        durability=_clamp(base_dur, 0.10, 0.95),
        injury_risk=_clamp(base_inj, 0.02, 0.45),
        recovery_rate=_clamp(rec, 0.10, 0.95),
        concussion_risk=_clamp(conc, 0.01, 0.40),
    )
    return {
        "durability": hp.durability,
        "injury_risk": hp.injury_risk,
        "recovery_rate": hp.recovery_rate,
        "concussion_risk": hp.concussion_risk,
        "body_type": bt,
    }


# ---------------------------------------------------------------------------
# Development rate (single scalar for sim)
# ---------------------------------------------------------------------------

def compute_development_rate(*, backstory: PlayerBackstory, traits: Dict[str, Any], talent: float) -> float:
    base = 0.45 + (talent - 0.55) * 0.55
    base += (backstory.impact.development_rate - 0.5) * 0.35
    mods = (traits or {}).get("mods", {}) or {}
    base += float(mods.get("development_rate", 0.0)) * 0.60
    base += (backstory.impact.work_ethic - 0.5) * 0.12
    return _clamp(base, 0.10, 0.95)

