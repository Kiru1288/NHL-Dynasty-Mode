"""
Team identity — how a club plays and what it hunts for.

A club's identity is not a label someone typed in. It is read from:

* the roster's attribute profile (skating, shooting, playmaking, puck skill,
  physicality, defence, goaltending, age), weighted toward the players who
  actually play, and normalized against the rest of the league;
* its best player — a franchise sniper bends a club toward run-and-gun, a
  franchise shutdown D toward structure;
* the coach's specialization and the club's archetype;
* the drafting staff's taste (size vs skill, from services/draft_team_identity);
* its own history — identity moves slowly (momentum), so one trade does not
  flip a heavy club into a speed club overnight.

The output drives acquisitions quietly: CPU trade valuation, free-agent fit and
draft boards all ask ``player_identity_fit`` how well a player suits the club,
so a run-and-gun team goes after speedsters and finishers while a heavy club
hunts power forwards and shutdown D.

Pure functions only. The franchise layer (backend/services/team_identity_service.py)
handles caching, momentum state and API payloads, and attaches the result to
``team.team_identity`` so SimEngine code (trade evaluator) can read it.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Attribute groups (rating-key prefixes / exact keys)
# ---------------------------------------------------------------------------

GROUPS: Tuple[str, ...] = ("pace", "shooting", "playmaking", "puck", "physical", "defense")

_GROUP_KEYS: Dict[str, Tuple[str, ...]] = {
    "pace": ("skg_",),
    "shooting": (
        "off_finishing", "off_one_timer", "off_puck_placement", "off_wrist_shot", "off_slap_shot",
        "off_shooting_under_pressure", "off_shot_iq", "off_tip_deflection",
    ),
    "playmaking": ("pm_",),
    "puck": ("pc_",),
    "physical": ("phy_physicality", "phy_checking", "phy_strength", "phy_aggression", "def_board_battles"),
    "defense": (
        "def_backchecking_effort", "def_body_positioning", "def_containment_ability", "def_defensive_awareness",
        "def_defensive_iq", "def_defensive_reads", "def_gap_control", "def_interception_skill",
        "def_net_coverage", "def_pk_awareness", "def_pressure_defense", "def_shot_blocking", "def_stick_checking",
    ),
}
_GOALIE_KEYS = ("g_athleticism", "g_positioning", "g_rebound_control_g", "g_reflexes")


def _ratings(player: Any) -> Dict[str, float]:
    r = getattr(player, "ratings", None)
    return r if isinstance(r, dict) else {}


def _pos(player: Any) -> str:
    ident = getattr(player, "identity", None)
    raw = getattr(ident, "position", None) or getattr(player, "position", "") or ""
    raw = str(getattr(raw, "value", raw) or "").upper()
    if raw.startswith("G"):
        return "G"
    if raw.startswith("D") or raw in ("LD", "RD"):
        return "D"
    return "F"


def _age(player: Any) -> int:
    try:
        return int(getattr(getattr(player, "identity", None), "age", None) or getattr(player, "age", 27) or 27)
    except (TypeError, ValueError):
        return 27


def _ovr(player: Any) -> float:
    try:
        fn = getattr(player, "ovr", None)
        v = float(fn() if callable(fn) else fn or 0.0)
        return v * 99.0 if v <= 1.5 else v
    except Exception:
        return 60.0


def _name(player: Any) -> str:
    return str(getattr(getattr(player, "identity", None), "name", None) or getattr(player, "name", "") or "")


def _avg_keys(r: Dict[str, float], keys: Iterable[str]) -> Optional[float]:
    vals: List[float] = []
    for want in keys:
        if want.endswith("_"):
            vals.extend(float(v) for k, v in r.items() if k.startswith(want) and isinstance(v, (int, float)))
        else:
            vals.extend(float(v) for k, v in r.items() if k.startswith(want) and isinstance(v, (int, float)))
    return sum(vals) / len(vals) if vals else None


def player_attribute_profile(player: Any) -> Dict[str, float]:
    """Group averages on the 0-99 rating scale (missing groups fall back to 60)."""
    r = _ratings(player)
    out: Dict[str, float] = {}
    for g, keys in _GROUP_KEYS.items():
        v = _avg_keys(r, keys)
        out[g] = float(v) if v is not None else 60.0
    return out


# ---------------------------------------------------------------------------
# Player style (one vocabulary over the several naming schemes in the codebase)
# ---------------------------------------------------------------------------

STYLE_LABELS: Dict[str, str] = {
    "sniper": "Sniper",
    "playmaker": "Playmaker",
    "power_forward": "Power forward",
    "two_way": "Two-way forward",
    "grinder": "Grinder",
    "speedster": "Speedster",
    "offensive_d": "Offensive D",
    "defensive_d": "Shutdown D",
    "two_way_d": "Two-way D",
    "mobile_d": "Mobile D",
    "goalie": "Goalie",
}


def player_style(player: Any) -> str:
    pos = _pos(player)
    if pos == "G":
        return "goalie"
    raw = " ".join(
        str(getattr(player, a, "") or "") for a in ("playstyle", "archetype", "player_type")
    ).lower()
    if pos == "D":
        if "offensive" in raw or "puck_mov" in raw:
            return "offensive_d"
        if "defensive" in raw or "shutdown" in raw or "enforcer" in raw:
            return "defensive_d"
        if "mobile" in raw:
            return "mobile_d"
        return "two_way_d"
    if "sniper" in raw or "scoring" in raw:
        return "speedster" if "scoring_forward" in raw and "sniper" not in raw else "sniper"
    if "playmak" in raw:
        return "playmaker"
    if "power" in raw:
        return "power_forward"
    if "grinder" in raw or "enforcer" in raw:
        return "grinder"
    # Unlabelled / two-way: let the attributes decide speedsters.
    prof = player_attribute_profile(player)
    m = sum(prof.values()) / len(prof)
    if prof["pace"] - m >= 4.0:
        return "speedster"
    return "two_way"


# ---------------------------------------------------------------------------
# Identity catalog
# ---------------------------------------------------------------------------

IDENTITIES: Dict[str, Dict[str, Any]] = {
    "run_and_gun": {
        "label": "Run & Gun",
        "blurb": "All-out offence: skates, shoots and trades chances.",
        "weights": {"pace": 0.9, "shooting": 1.0, "playmaking": 0.6, "puck": 0.3, "defense": -0.45},
        "star_styles": {"sniper": 1.0, "speedster": 0.8, "offensive_d": 0.6, "playmaker": 0.5},
        "coach": {"offense": 0.5, "offensive": 0.5},
        "targets": ["speedster", "sniper", "offensive_d", "playmaker"],
        "fit_weights": {"pace": 1.0, "shooting": 1.0, "playmaking": 0.5, "defense": -0.2},
    },
    "speed_transition": {
        "label": "Speed & Transition",
        "blurb": "Wins the neutral zone with skating and quick-strike rushes.",
        "weights": {"pace": 1.6, "puck": 0.5, "playmaking": 0.3, "physical": -0.3},
        "star_styles": {"speedster": 1.0, "mobile_d": 0.8, "playmaker": 0.4},
        "coach": {"transition": 0.5, "development": 0.2},
        "targets": ["speedster", "mobile_d", "playmaker"],
        "fit_weights": {"pace": 1.4, "puck": 0.5, "physical": -0.2},
    },
    "heavy_forecheck": {
        "label": "Heavy Forecheck",
        "blurb": "Physical, heavy on the puck, wins along the walls.",
        "weights": {"physical": 1.4, "defense": 0.35, "pace": -0.15},
        "star_styles": {"power_forward": 1.0, "grinder": 0.7, "defensive_d": 0.4},
        "coach": {"physical": 0.5, "grit": 0.5},
        "targets": ["power_forward", "grinder", "defensive_d"],
        "fit_weights": {"physical": 1.4, "defense": 0.3, "shooting": 0.2},
    },
    "defensive_structure": {
        "label": "Defensive Structure",
        "blurb": "Low-event, layered defence and patience.",
        "weights": {"defense": 1.4, "goaltending": 0.45, "pace": -0.2, "shooting": -0.15},
        "star_styles": {"defensive_d": 1.0, "two_way": 0.8, "two_way_d": 0.6, "goalie": 0.5},
        "coach": {"defense": 0.5, "defensive": 0.5, "systems": 0.3},
        "targets": ["defensive_d", "two_way", "two_way_d"],
        "fit_weights": {"defense": 1.4, "physical": 0.2, "pace": -0.1},
    },
    "skill_possession": {
        "label": "Skill & Possession",
        "blurb": "Keeps the puck: passing, puck skill, patience in the offensive zone.",
        "weights": {"playmaking": 1.2, "puck": 1.1, "shooting": 0.2, "physical": -0.3},
        "star_styles": {"playmaker": 1.0, "offensive_d": 0.5, "two_way": 0.3},
        "coach": {"possession": 0.5, "analytics": 0.3, "offense": 0.2},
        "targets": ["playmaker", "offensive_d", "two_way"],
        "fit_weights": {"playmaking": 1.2, "puck": 1.0},
    },
    "goaltending_fortress": {
        "label": "Goaltending Fortress",
        "blurb": "Built from the crease out: elite netminding, low-risk hockey.",
        "weights": {"goaltending": 1.6, "defense": 0.4},
        "star_styles": {"goalie": 1.0},
        "coach": {"goaltending": 0.5},
        "targets": ["defensive_d", "two_way_d", "two_way"],
        "fit_weights": {"defense": 1.0, "physical": 0.2},
    },
}

TRAITS: Dict[str, str] = {
    "youth": "Youth movement",
    "veteran": "Veteran core",
    "deep": "Rolls four lines",
    "top_heavy": "Star-driven",
}


# ---------------------------------------------------------------------------
# Team profile
# ---------------------------------------------------------------------------

def _dressed(team: Any) -> Tuple[List[Any], List[Any], List[Any]]:
    roster = [p for p in (getattr(team, "roster", None) or []) if p is not None and not getattr(p, "retired", False)]
    fw = sorted([p for p in roster if _pos(p) == "F"], key=_ovr, reverse=True)[:12]
    dm = sorted([p for p in roster if _pos(p) == "D"], key=_ovr, reverse=True)[:6]
    gs = sorted([p for p in roster if _pos(p) == "G"], key=_ovr, reverse=True)[:2]
    return fw, dm, gs


def team_raw_profile(team: Any) -> Dict[str, Any]:
    """Usage-weighted attribute means for the lineup that actually plays."""
    fw, dm, gs = _dressed(team)
    skaters = fw + dm
    sums = {g: 0.0 for g in GROUPS}
    wsum = 0.0
    for i, p in enumerate(fw):
        w = (1.0, 1.0, 1.0, 0.85, 0.85, 0.85, 0.65, 0.65, 0.65, 0.45, 0.45, 0.45)[min(i, 11)]
        prof = player_attribute_profile(p)
        for g in GROUPS:
            sums[g] += prof[g] * w
        wsum += w
    for i, p in enumerate(dm):
        w = (0.95, 0.95, 0.8, 0.8, 0.6, 0.6)[min(i, 5)]
        prof = player_attribute_profile(p)
        for g in GROUPS:
            sums[g] += prof[g] * w
        wsum += w
    out: Dict[str, Any] = {g: (sums[g] / wsum if wsum else 60.0) for g in GROUPS}
    g1 = gs[0] if gs else None
    if g1 is not None:
        gv = _avg_keys(_ratings(g1), _GOALIE_KEYS)
        out["goaltending"] = float(gv if gv is not None else _ovr(g1))
    else:
        out["goaltending"] = 55.0
    ages = [_age(p) for p in skaters]
    out["age"] = sum(ages) / len(ages) if ages else 27.0
    ovrs = [_ovr(p) for p in skaters]
    if ovrs:
        top = sorted(ovrs, reverse=True)
        out["top3"] = sum(top[:3]) / min(3, len(top))
        out["depth"] = sum(top[6:12]) / max(1, len(top[6:12])) if len(top) > 6 else out["top3"]
    else:
        out["top3"] = out["depth"] = 60.0
    star = max(skaters + gs, key=_ovr) if (skaters or gs) else None
    out["star"] = star
    return out


def _league_stats(profiles: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for g in GROUPS + ("goaltending", "age"):
        vals = [float(p.get(g, 0.0)) for p in profiles]
        if not vals:
            stats[g] = (0.0, 1.0)
            continue
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
        stats[g] = (mu, sd)
    gaps = [float(p.get("top3", 0)) - float(p.get("depth", 0)) for p in profiles]
    mu = sum(gaps) / len(gaps) if gaps else 0.0
    sd = math.sqrt(sum((v - mu) ** 2 for v in gaps) / len(gaps)) if gaps else 1.0
    stats["star_gap"] = (mu, sd or 1.0)
    return stats


def _pct(z: float) -> int:
    return int(round(100.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))


def _stable_u(*parts: Any) -> float:
    raw = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(raw[:12], 16) / float(1 << 48)


# ---------------------------------------------------------------------------
# Identity scoring
# ---------------------------------------------------------------------------

def score_identities(
    team: Any,
    profile: Dict[str, Any],
    stats: Dict[str, Tuple[float, float]],
    *,
    draft_identity: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """Return (identity scores, profile z-scores, human-readable drivers)."""
    z: Dict[str, float] = {}
    for g in GROUPS + ("goaltending", "age"):
        mu, sd = stats[g]
        z[g] = (float(profile.get(g, mu)) - mu) / (sd or 1.0)
    drivers: List[str] = []
    # Style is SHAPE, not quality: a club good at everything is not "skill" by
    # default. Skater groups are read relative to the club's own average z.
    zbar = sum(z[g] for g in GROUPS) / len(GROUPS)
    shape = {g: (z[g] - zbar) * 1.6 for g in GROUPS}
    shape["goaltending"] = z.get("goaltending", 0.0) * 0.75 - max(0.0, zbar) * 0.25
    scores: Dict[str, float] = {}
    for key, spec in IDENTITIES.items():
        scores[key] = sum(w * shape.get(g, 0.0) for g, w in spec["weights"].items())

    # Franchise player bends the identity (scaled by how far he stands above the room).
    star = profile.get("star")
    if star is not None:
        st = player_style(star)
        margin = max(0.0, _ovr(star) - float(profile.get("top3", 0.0)) + 4.0)
        star_w = max(0.25, min(0.8, 0.25 + margin * 0.06 + max(0.0, _ovr(star) - 85.0) * 0.03))
        for key, spec in IDENTITIES.items():
            aff = float(spec["star_styles"].get(st, 0.0))
            if aff:
                scores[key] += aff * star_w
        drivers.append(f"Built around {_name(star)} ({STYLE_LABELS.get(st, st)})")

    # Coach specialization + club archetype.
    coach = getattr(team, "coach", None)
    spec_txt = " ".join(
        str(x or "").lower()
        for x in (getattr(coach, "specialization", ""), getattr(team, "coach_type", ""), getattr(team, "archetype", ""))
    )
    for key, spec in IDENTITIES.items():
        for word, bump in spec["coach"].items():
            if word in spec_txt:
                scores[key] += bump
    if getattr(coach, "specialization", None):
        drivers.append(f"Coach leans {str(getattr(coach, 'specialization')).lower()}")

    # Drafting staff taste.
    if isinstance(draft_identity, dict):
        size = float(draft_identity.get("size_pref") or 0.0)
        style = float(draft_identity.get("style_pref") or 0.0)
        scores["heavy_forecheck"] += 0.45 * size
        scores["speed_transition"] -= 0.25 * size
        scores["run_and_gun"] += 0.30 * style
        scores["skill_possession"] += 0.30 * style
        scores["defensive_structure"] -= 0.30 * style
        tags = list(draft_identity.get("tags") or [])
        if tags:
            drivers.append("Draft room: " + ", ".join(tags[:2]))

    # Small stable personality so near-identical rosters don't all read the same.
    tid = str(getattr(team, "team_id", "") or getattr(team, "id", ""))
    for key in scores:
        scores[key] += (_stable_u("identity-flavour", tid, key) - 0.5) * 0.3
    return scores, z, drivers


def _secondary_traits(z: Dict[str, float], profile: Dict[str, Any], stats: Dict[str, Tuple[float, float]]) -> List[str]:
    traits: List[str] = []
    if z.get("age", 0.0) <= -0.9:
        traits.append(TRAITS["youth"])
    elif z.get("age", 0.0) >= 0.9:
        traits.append(TRAITS["veteran"])
    mu, sd = stats.get("star_gap", (0.0, 1.0))
    gap_z = ((float(profile.get("top3", 0)) - float(profile.get("depth", 0))) - mu) / (sd or 1.0)
    if gap_z >= 0.9:
        traits.append(TRAITS["top_heavy"])
    elif gap_z <= -0.9:
        traits.append(TRAITS["deep"])
    return traits


def blend_scores(prev: Optional[Dict[str, float]], cur: Dict[str, float], momentum: float = 0.65) -> Dict[str, float]:
    """Identity changes slowly: keep ``momentum`` of last season's read."""
    if not isinstance(prev, dict) or not prev:
        return dict(cur)
    out: Dict[str, float] = {}
    for k, v in cur.items():
        out[k] = momentum * float(prev.get(k, v)) + (1.0 - momentum) * float(v)
    return out


def build_identity(
    team: Any,
    scores: Dict[str, float],
    z: Dict[str, float],
    profile: Dict[str, Any],
    stats: Dict[str, Tuple[float, float]],
    drivers: List[str],
) -> Dict[str, Any]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_val = ranked[0]
    sec_key, sec_val = ranked[1]
    balanced = top_val < 0.55
    primary = (
        {"key": "balanced", "label": "Balanced", "blurb": "No single dominant style; plays to the matchup."}
        if balanced
        else {"key": top_key, "label": IDENTITIES[top_key]["label"], "blurb": IDENTITIES[top_key]["blurb"]}
    )
    secondary = None
    if not balanced and sec_val >= 0.45 and sec_val >= top_val * 0.6:
        secondary = {"key": sec_key, "label": IDENTITIES[sec_key]["label"]}
    traits = _secondary_traits(z, profile, stats)
    strength = max(0.0, min(1.0, (top_val - 0.3) / 1.8))
    # What the front office hunts for: primary targets first, then secondary's.
    target_styles: List[str] = []
    fit_weights: Dict[str, float] = {}
    for key, w in ((top_key, 1.0), (sec_key, 0.5)):
        spec = IDENTITIES[key]
        for s in spec["targets"]:
            if s not in target_styles:
                target_styles.append(s)
        for g, fw in spec["fit_weights"].items():
            fit_weights[g] = fit_weights.get(g, 0.0) + fw * w * (0.4 if balanced else 1.0)
    label_line = primary["label"]
    if secondary:
        label_line += " · " + secondary["label"]
    elif traits:
        label_line += " · " + traits[0]
    star = profile.get("star")
    return {
        "team_id": str(getattr(team, "team_id", "") or getattr(team, "id", "")),
        "primary": primary,
        "secondary": secondary,
        "traits": traits,
        "label": label_line,
        "strength": round(strength, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "profile_pct": {g: _pct(z.get(g, 0.0)) for g in GROUPS + ("goaltending",)},
        "avg_age": round(float(profile.get("age", 27.0)), 1),
        "star": {"name": _name(star), "style": STYLE_LABELS.get(player_style(star), "")} if star is not None else None,
        "target_styles": target_styles[:5],
        "target_labels": [STYLE_LABELS.get(s, s) for s in target_styles[:4]],
        "fit_weights": {g: round(v, 3) for g, v in fit_weights.items()},
        "drivers": drivers[:4],
    }


def compute_league_identities(
    teams: List[Any],
    *,
    prev_scores: Optional[Dict[str, Dict[str, float]]] = None,
    draft_identities: Optional[Dict[str, Dict[str, Any]]] = None,
    momentum: float = 0.65,
) -> Dict[str, Dict[str, Any]]:
    """Identity for every club, normalized against this league."""
    profiles = {str(getattr(t, "team_id", "") or getattr(t, "id", "")): team_raw_profile(t) for t in teams}
    stats = _league_stats(list(profiles.values()))
    out: Dict[str, Dict[str, Any]] = {}
    for t in teams:
        tid = str(getattr(t, "team_id", "") or getattr(t, "id", ""))
        prof = profiles[tid]
        scores, z, drivers = score_identities(t, prof, stats, draft_identity=(draft_identities or {}).get(tid))
        raw = dict(scores)
        scores = blend_scores((prev_scores or {}).get(tid), scores, momentum)
        ident = build_identity(t, scores, z, prof, stats, drivers)
        ident["raw_scores"] = {k: round(v, 3) for k, v in raw.items()}
        out[tid] = ident
    return out


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def player_identity_fit(identity: Optional[Dict[str, Any]], player: Any) -> float:
    """0..1: how well a player suits the club's identity (0.5 = neutral)."""
    if not isinstance(identity, dict) or player is None:
        return 0.5
    st = player_style(player)
    targets = list(identity.get("target_styles") or [])
    strength = float(identity.get("strength") or 0.0)
    style_part = 0.0
    if st in targets:
        style_part = 0.30 - 0.05 * targets.index(st)
    elif st != "goalie":
        style_part = -0.08
    prof = player_attribute_profile(player)
    m = sum(prof.values()) / len(prof)
    weights = identity.get("fit_weights") or {}
    wsum = sum(abs(float(w)) for w in weights.values()) or 1.0
    attr_part = sum(float(w) * (prof.get(g, m) - m) for g, w in weights.items()) / wsum
    attr_part = max(-0.25, min(0.25, attr_part / 24.0))
    fit = 0.5 + (style_part + attr_part) * (0.5 + 0.5 * strength)
    return max(0.0, min(1.0, fit))


def fit_reason(identity: Optional[Dict[str, Any]], player: Any, fit: float) -> Optional[str]:
    if not isinstance(identity, dict):
        return None
    lab = (identity.get("primary") or {}).get("label") or "team"
    if fit >= 0.68:
        return f"Fits the {lab} identity ({STYLE_LABELS.get(player_style(player), 'player')})"
    if fit <= 0.36:
        return f"Poor fit for a {lab} club"
    return None
