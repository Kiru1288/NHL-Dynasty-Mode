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


# Off the player objects so a background thread never grows a player's __dict__ while
# the session is being pickled. Keyed by id() with an identity check; bounded.
_ATTR_PROFILE_MEMO: Dict[int, Tuple[Any, Any, Dict[str, float]]] = {}
_ATTR_PROFILE_MEMO_MAX = 20000


def player_attribute_profile(player: Any) -> Dict[str, float]:
    """Group averages on the 0-99 rating scale (missing groups fall back to 60)."""
    r = _ratings(player)
    # Pure on the ratings dict. League identities re-ran it for every rostered player
    # whenever any roster in the league changed (most sim days).
    try:
        memo_key: Any = tuple(r.items())
        hash(memo_key)
    except TypeError:
        memo_key = None
    if memo_key is not None:
        held = _ATTR_PROFILE_MEMO.get(id(player))
        if held is not None and held[0] is player and held[1] == memo_key:
            return dict(held[2])
    out: Dict[str, float] = {}
    for g, keys in _GROUP_KEYS.items():
        v = _avg_keys(r, keys)
        out[g] = float(v) if v is not None else 60.0
    if memo_key is not None:
        if len(_ATTR_PROFILE_MEMO) >= _ATTR_PROFILE_MEMO_MAX:
            _ATTR_PROFILE_MEMO.clear()
        _ATTR_PROFILE_MEMO[id(player)] = (player, memo_key, dict(out))
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
        "weights": {"goaltending": 1.2, "defense": 0.4},
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

    # Tiny stable tiebreak only (was +/-0.15, which made labels feel random).
    tid = str(getattr(team, "team_id", "") or getattr(team, "id", ""))
    for key in scores:
        scores[key] += (_stable_u("identity-flavour", tid, key) - 0.5) * 0.06
    return scores, z, drivers


# ---------------------------------------------------------------------------
# How the club actually plays (season stats)
# ---------------------------------------------------------------------------

#: metric key -> (label, unit, higher_is "more")
PLAY_METRICS: Dict[str, Tuple[str, str]] = {
    "gf_pg": ("Goals for", "/game"),
    "ga_pg": ("Goals against", "/game"),
    "sf_pg": ("Shots for", "/game"),
    "events_pg": ("Shot attempts (both teams)", "/game"),
    "cf_pct": ("Possession (CF%)", "%"),
    "xgf_pg": ("Expected goals for", "/game"),
    "xga_pg": ("Expected goals against", "/game"),
    "xgf_pct": ("Expected-goal share (xGF%)", "%"),
    "hits_pg": ("Hits", "/game"),
    "pim_pg": ("Penalty minutes", "/game"),
    "sv_pct": ("Save %", ""),
}

#: identity -> {metric: weight on the metric's league z-score}
PLAY_EVIDENCE: Dict[str, Dict[str, float]] = {
    "run_and_gun": {"gf_pg": 0.9, "ga_pg": 0.5, "events_pg": 0.8, "sf_pg": 0.5},
    "speed_transition": {"xgf_pg": 0.8, "events_pg": 0.4, "gf_pg": 0.4, "hits_pg": -0.4},
    "heavy_forecheck": {"hits_pg": 1.3, "pim_pg": 0.35, "cf_pct": 0.3},
    "defensive_structure": {"ga_pg": -0.8, "xga_pg": -0.9, "events_pg": -0.6},
    "skill_possession": {"cf_pct": 1.2, "xgf_pct": 0.7, "hits_pg": -0.35},
    "goaltending_fortress": {"sv_pct": 1.0, "ga_pg": -0.45},
}


def team_play_metrics(row: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Per-game play-style metrics from a team analytics row (None before ~10 games)."""
    def f(*keys: str) -> float:
        for k in keys:
            v = row.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    gp = f("gp", "games_played")
    if gp < 10:
        return None
    cf, ca, xgf, xga = f("cf", "corsi_for"), f("ca", "corsi_against"), f("xgf"), f("xga")
    sa = f("sa", "shots_against")
    ga = f("ga", "goals_against")
    return {
        "gp": gp,
        "gf_pg": f("gf", "goals_for") / gp,
        "ga_pg": ga / gp,
        "sf_pg": f("sf", "sog", "shots_for") / gp,
        "events_pg": (cf + ca) / gp,
        "cf_pct": 100.0 * cf / (cf + ca) if cf + ca > 0 else 50.0,
        "xgf_pg": xgf / gp,
        "xga_pg": xga / gp,
        "xgf_pct": 100.0 * xgf / (xgf + xga) if xgf + xga > 0 else 50.0,
        "hits_pg": f("hits") / gp,
        "pim_pg": f("pim") / gp,
        "sv_pct": (1.0 - ga / sa) if sa > 0 else 0.905,
    }


def _play_layer(
    metrics: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Tuple[float, int]]], Dict[str, Tuple[float, float]]]:
    """League z-scores per metric, plus each value's league rank (1 = most)."""
    stats: Dict[str, Tuple[float, float]] = {}
    zs: Dict[str, Dict[str, float]] = {tid: {} for tid in metrics}
    ranks: Dict[str, Dict[str, Tuple[float, int]]] = {tid: {} for tid in metrics}
    for m in PLAY_METRICS:
        vals = [(tid, float(v[m])) for tid, v in metrics.items()]
        if not vals:
            continue
        mu = sum(x for _, x in vals) / len(vals)
        sd = math.sqrt(sum((x - mu) ** 2 for _, x in vals) / len(vals)) or 1.0
        stats[m] = (mu, sd)
        order = sorted(vals, key=lambda kv: -kv[1])
        for i, (tid, x) in enumerate(order):
            zs[tid][m] = (x - mu) / sd
            ranks[tid][m] = (x, i + 1)
    return zs, ranks, stats


def _fmt_metric(m: str, v: float) -> str:
    if m == "sv_pct":
        return f"{v:.3f}"
    if PLAY_METRICS[m][1] == "%":
        return f"{v:.1f}%"
    return f"{v:.1f}"


def _why_rows(key: str, tid: str, zs: Dict[str, Dict[str, float]], ranks: Dict[str, Dict[str, Tuple[float, int]]],
              stats: Dict[str, Tuple[float, float]], n: int) -> List[Dict[str, Any]]:
    spec = PLAY_EVIDENCE.get(key) or {}
    contrib = sorted(((m, w * zs.get(tid, {}).get(m, 0.0)) for m, w in spec.items()), key=lambda kv: -kv[1])
    out: List[Dict[str, Any]] = []
    for m, c in contrib[:3]:
        if m not in ranks.get(tid, {}):
            continue
        v, rank = ranks[tid][m]
        label, unit = PLAY_METRICS[m]
        low = float(spec.get(m, 0.0)) < 0
        r = (n + 1 - rank) if low else rank
        sfx = "th" if 10 <= r % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
        out.append({
            "rank_text": f"{r}{sfx} {'fewest' if low else 'most'} of {n}",
            "metric": m,
            "label": label,
            "value": round(v, 3),
            "display": _fmt_metric(m, v) + ("" if unit == "%" or not unit else unit),
            "league_avg": _fmt_metric(m, stats[m][0]),
            "rank": rank,
            "of": n,
            "supports": c > 0.15,
        })
    return out


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
    play_rows: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Identity for every club, normalized against this league.

    ``play_rows`` (team_id -> season analytics row) adds how the club actually plays:
    shot volume, possession, xG, hits, goals for/against, save %. Its weight grows with
    games played (full weight from ~40 GP), so by midseason the label is mostly earned
    on the ice, and every identity carries the metrics behind it in ``why``.
    """
    profiles = {str(getattr(t, "team_id", "") or getattr(t, "id", "")): team_raw_profile(t) for t in teams}
    stats = _league_stats(list(profiles.values()))
    metrics: Dict[str, Dict[str, float]] = {}
    for tid, row in (play_rows or {}).items():
        mm = team_play_metrics(row) if isinstance(row, dict) else None
        if mm is not None:
            metrics[str(tid)] = mm
    use_play = len(metrics) >= max(4, int(0.75 * len(profiles)))
    zs, ranks, pstats = _play_layer(metrics) if use_play else ({}, {}, {})
    out: Dict[str, Dict[str, Any]] = {}
    for t in teams:
        tid = str(getattr(t, "team_id", "") or getattr(t, "id", ""))
        prof = profiles[tid]
        scores, z, drivers = score_identities(t, prof, stats, draft_identity=(draft_identities or {}).get(tid))
        play_w = 0.0
        if use_play and tid in metrics:
            play_w = min(1.0, metrics[tid]["gp"] / 40.0) * 1.3
            for key, spec in PLAY_EVIDENCE.items():
                ev = sum(w * zs[tid].get(m, 0.0) for m, w in spec.items()) / (sum(abs(w) for w in spec.values()) or 1.0)
                scores[key] = scores.get(key, 0.0) * (1.0 - 0.45 * min(1.0, play_w)) + play_w * ev * 1.4
        raw = dict(scores)
        scores = blend_scores((prev_scores or {}).get(tid), scores, momentum)
        ident = build_identity(t, scores, z, prof, stats, drivers)
        ident["raw_scores"] = {k: round(v, 3) for k, v in raw.items()}
        pk = (ident.get("primary") or {}).get("key")
        why: Dict[str, Any] = {"roster": list(drivers[:3]), "on_ice": [], "games": 0, "basis": "roster"}
        if play_w > 0 and pk:
            if pk == "balanced":
                why["on_ice"] = [
                    {"metric": m, "label": PLAY_METRICS[m][0],
                     "display": _fmt_metric(m, ranks[tid][m][0]) + ("" if PLAY_METRICS[m][1] in ("%", "") else PLAY_METRICS[m][1]),
                     "league_avg": _fmt_metric(m, pstats[m][0]), "rank": ranks[tid][m][1], "of": len(metrics),
                     "supports": True}
                    for m in ("cf_pct", "events_pg", "hits_pg") if m in ranks[tid]
                ]
            else:
                why["on_ice"] = _why_rows(pk, tid, zs, ranks, pstats, len(metrics))
            why["games"] = int(metrics[tid]["gp"])
            why["basis"] = "on_ice" if play_w >= 0.65 else "roster_and_on_ice"
        else:
            # Roster read only: say which attribute groups drive it.
            why["roster"] += [f"{g.title()} rating: {_pct(z.get(g, 0.0))}th percentile" for g in
                              sorted(GROUPS, key=lambda g: -abs(z.get(g, 0.0)))[:2]]
        ident["why"] = why
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
