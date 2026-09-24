# app/sim_engine/generation/prospect_league_scoring.py
"""
Junior / prospect league scoring environments.

Junior hockey is intentionally more inflated than NHL or pro-Europe leagues.
High CHL/QMJHL point totals do NOT automatically imply elite NHL translation —
overagers, character-risk scorers, and boom/bust profiles can post huge junior
numbers while carrying translation risk.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Tuple

# League profiles: lower difficulty = easier scoring environment.
LEAGUE_SCORING_PROFILES: Dict[str, Dict[str, Any]] = {
    "CHL": {
        "difficulty": 0.62,
        "scoring_multiplier": 1.35,
        "elite_ppg_target": (1.62, 2.15),
        "star_ppg_target": (1.22, 1.70),
        "average_ppg_target": (0.62, 1.00),
        "overager_bonus": 1.18,
        "boom_bust_bonus": 1.15,
        "character_concern_bonus": 1.10,
        "defensive_translation_penalty": 0.90,
        "gp_range": (58, 68),
        "difficulty_label": "Easy",
        "environment_label": "High-scoring junior league",
    },
    "OHL": {
        "difficulty": 0.62,
        "scoring_multiplier": 1.37,
        "elite_ppg_target": (1.65, 2.20),
        "star_ppg_target": (1.25, 1.75),
        "average_ppg_target": (0.65, 1.05),
        "overager_bonus": 1.18,
        "boom_bust_bonus": 1.15,
        "character_concern_bonus": 1.10,
        "defensive_translation_penalty": 0.90,
        "gp_range": (58, 68),
        "difficulty_label": "Easy",
        "environment_label": "High-scoring junior league",
    },
    "WHL": {
        "difficulty": 0.66,
        "scoring_multiplier": 1.30,
        "elite_ppg_target": (1.55, 2.10),
        "star_ppg_target": (1.18, 1.68),
        "average_ppg_target": (0.60, 0.98),
        "overager_bonus": 1.16,
        "boom_bust_bonus": 1.12,
        "character_concern_bonus": 1.08,
        "defensive_translation_penalty": 0.91,
        "gp_range": (58, 68),
        "difficulty_label": "Easy",
        "environment_label": "High-scoring junior league",
    },
    "QMJHL": {
        "difficulty": 0.58,
        "scoring_multiplier": 1.38,
        "elite_ppg_target": (1.70, 2.25),
        "star_ppg_target": (1.28, 1.80),
        "average_ppg_target": (0.68, 1.08),
        "overager_bonus": 1.20,
        "boom_bust_bonus": 1.18,
        "character_concern_bonus": 1.12,
        "defensive_translation_penalty": 0.88,
        "gp_range": (58, 68),
        "difficulty_label": "Easy",
        "environment_label": "High-scoring junior league",
    },
    "USHL": {
        "difficulty": 0.72,
        "scoring_multiplier": 1.15,
        "elite_ppg_target": (1.20, 1.65),
        "star_ppg_target": (0.95, 1.35),
        "average_ppg_target": (0.45, 0.78),
        "overager_bonus": 1.10,
        "boom_bust_bonus": 1.08,
        "character_concern_bonus": 1.05,
        "defensive_translation_penalty": 0.93,
        "gp_range": (52, 62),
        "difficulty_label": "Moderate",
        "environment_label": "Junior inflated",
    },
    "NCAA": {
        "difficulty": 0.82,
        "scoring_multiplier": 0.90,
        "elite_ppg_target": (0.95, 1.35),
        "star_ppg_target": (0.75, 1.10),
        "average_ppg_target": (0.35, 0.62),
        "overager_bonus": 1.06,
        "boom_bust_bonus": 1.05,
        "character_concern_bonus": 1.04,
        "defensive_translation_penalty": 0.95,
        "gp_range": (32, 42),
        "difficulty_label": "Moderate",
        "environment_label": "Lower-scoring college league",
    },
    "SHL": {
        "difficulty": 0.92,
        "scoring_multiplier": 0.70,
        "elite_ppg_target": (0.55, 0.95),
        "star_ppg_target": (0.42, 0.72),
        "average_ppg_target": (0.18, 0.38),
        "overager_bonus": 1.04,
        "boom_bust_bonus": 1.03,
        "character_concern_bonus": 1.02,
        "defensive_translation_penalty": 0.98,
        "gp_range": (40, 52),
        "difficulty_label": "Hard",
        "environment_label": "Hard pro league",
    },
    "LIIGA": {
        "difficulty": 0.88,
        "scoring_multiplier": 0.78,
        "elite_ppg_target": (0.65, 1.05),
        "star_ppg_target": (0.48, 0.78),
        "average_ppg_target": (0.22, 0.42),
        "overager_bonus": 1.05,
        "boom_bust_bonus": 1.04,
        "character_concern_bonus": 1.03,
        "defensive_translation_penalty": 0.97,
        "gp_range": (42, 54),
        "difficulty_label": "Hard",
        "environment_label": "Hard pro league",
    },
    "AHL": {
        "difficulty": 0.86,
        "scoring_multiplier": 0.85,
        "elite_ppg_target": (0.75, 1.15),
        "star_ppg_target": (0.58, 0.88),
        "average_ppg_target": (0.28, 0.52),
        "overager_bonus": 1.03,
        "boom_bust_bonus": 1.02,
        "character_concern_bonus": 1.02,
        "defensive_translation_penalty": 0.96,
        "gp_range": (58, 72),
        "difficulty_label": "Pro",
        "environment_label": "Pro development league",
    },
    "ECHL": {
        "difficulty": 0.72,
        "scoring_multiplier": 0.98,
        "elite_ppg_target": (0.85, 1.25),
        "star_ppg_target": (0.62, 0.95),
        "average_ppg_target": (0.32, 0.58),
        "overager_bonus": 1.05,
        "boom_bust_bonus": 1.04,
        "character_concern_bonus": 1.03,
        "defensive_translation_penalty": 0.94,
        "gp_range": (55, 68),
        "difficulty_label": "Pro",
        "environment_label": "Secondary pro development league",
    },
    "EUROPE_JUNIOR": {
        "difficulty": 0.84,
        "scoring_multiplier": 0.82,
        "elite_ppg_target": (0.70, 1.10),
        "star_ppg_target": (0.50, 0.82),
        "average_ppg_target": (0.24, 0.45),
        "overager_bonus": 1.06,
        "boom_bust_bonus": 1.05,
        "character_concern_bonus": 1.04,
        "defensive_translation_penalty": 0.96,
        "gp_range": (40, 52),
        "difficulty_label": "Hard",
        "environment_label": "European junior ladder",
    },
    "JUNIOR": {
        "difficulty": 0.68,
        "scoring_multiplier": 1.38,
        "elite_ppg_target": (1.48, 2.02),
        "star_ppg_target": (1.15, 1.58),
        "average_ppg_target": (0.54, 0.90),
        "overager_bonus": 1.12,
        "boom_bust_bonus": 1.10,
        "character_concern_bonus": 1.08,
        "defensive_translation_penalty": 0.92,
        "gp_range": (54, 64),
        "difficulty_label": "Moderate",
        "environment_label": "Junior inflated",
    },
}

LEAGUE_ALIASES: Dict[str, str] = {
    "CHL": "CHL",
    "OHL": "OHL",
    "CHL_OHL": "OHL",
    "WHL": "WHL",
    "CHL_WHL": "WHL",
    "QMJHL": "QMJHL",
    "CHL_QMJHL": "QMJHL",
    "QMJHL/Q": "QMJHL",
    "CANADIAN HOCKEY LEAGUE": "CHL",
    "CANADIAN HOCKEY LEAGUE — OHL CLUSTER": "OHL",
    "CANADIAN HOCKEY LEAGUE — WHL CLUSTER": "WHL",
    "CANADIAN HOCKEY LEAGUE — QMJHL CLUSTER": "QMJHL",
    "MAJOR JUNIOR": "CHL",
    "CANADIAN JUNIOR": "CHL",
    "NCAA": "NCAA",
    "COLLEGE": "NCAA",
    "US COLLEGE": "NCAA",
    "NCAA DIVISION I CLUSTER": "NCAA",
    "USHL": "USHL",
    "USA JUNIOR": "USHL",
    "UNITED STATES HOCKEY LEAGUE": "USHL",
    "SHL": "SHL",
    "SWEDEN": "SHL",
    "EU_J_SHL": "SHL",
    "SWEDEN J20 / JUNIOR LADDER": "SHL",
    "LIIGA": "LIIGA",
    "FINLAND": "LIIGA",
    "EU_J_LIIGA": "LIIGA",
    "FINLAND U20 JUNIOR LADDER": "LIIGA",
    "AHL": "AHL",
    "ECHL": "ECHL",
    "ECHL PRO": "ECHL",
    "EAST COAST": "ECHL",
    "EUROPE": "EUROPE_JUNIOR",
    "JUNIOR": "JUNIOR",
    "EU_J_DEL": "EUROPE_JUNIOR",
    "EU_J_SWISS": "EUROPE_JUNIOR",
    "EU_J_CZ": "EUROPE_JUNIOR",
    "EU_J_SK": "EUROPE_JUNIOR",
    "EU_J_KHL_JR": "EUROPE_JUNIOR",
    "EU_J_NOR": "EUROPE_JUNIOR",
    "EU_J_DEN": "EUROPE_JUNIOR",
    "EU_J_AUT": "EUROPE_JUNIOR",
}

# Calendar pacing: fraction of regular season complete by calendar month (0..1).
_BASE_MONTH_FRAC: Dict[int, float] = {
    7: 0.0,
    8: 0.01,
    9: 0.05,
    10: 0.16,
    11: 0.28,
    12: 0.42,
    1: 0.56,
    2: 0.72,
    3: 0.86,
    4: 0.97,
    5: 1.0,
    6: 1.0,
}

# Shift season curve earlier/later per league (NCAA starts later, EU earlier).
_LEAGUE_FRAC_OFFSET: Dict[str, float] = {
    "NCAA": -0.08,
    "USHL": -0.02,
    "SHL": 0.06,
    "LIIGA": 0.05,
    "EUROPE_JUNIOR": 0.04,
    "AHL": 0.0,
    "ECHL": 0.0,
    "JUNIOR": 0.0,
    "CHL": 0.0,
    "OHL": 0.0,
    "WHL": 0.0,
    "QMJHL": 0.0,
}



# Where a typical roster of each league sits on the offensive-composite axis.
# PPG is mapped from (composite - center) so a league's own talent level, not the
# absolute composite, decides where a player lands inside that league's bands.
_LEAGUE_TALENT_CENTER: Dict[str, float] = {
    "AHL": 0.605,
    "ECHL": 0.535,
    "NCAA": 0.450,
    "USHL": 0.415,
    "CHL": 0.430,
    "OHL": 0.430,
    "WHL": 0.430,
    "QMJHL": 0.430,
    "JUNIOR": 0.430,
    "EUROPE_JUNIOR": 0.430,
    "SHL": 0.430,
    "LIIGA": 0.430,
}
_TALENT_SIGMA = 0.05

# Baseline goalie environment per league: (save pct, shots against per game).
_GOALIE_ENV: Dict[str, Tuple[float, float]] = {
    "CHL": (0.893, 30.0),
    "OHL": (0.893, 30.0),
    "WHL": (0.893, 30.0),
    "QMJHL": (0.891, 30.5),
    "JUNIOR": (0.893, 30.0),
    "USHL": (0.895, 29.0),
    "NCAA": (0.903, 27.5),
    "SHL": (0.905, 26.0),
    "LIIGA": (0.905, 26.0),
    "EUROPE_JUNIOR": (0.902, 26.5),
    "AHL": (0.903, 29.0),
    "ECHL": (0.900, 30.0),
}

# Injury hazard per game played (was per *update*, which made injury rates depend
# on how often the calendar was advanced).
_INJURY_HAZARD_PER_GAME = 0.006


def development_league_stat_max_age(code: Any) -> int:
    """Oldest player kept on the daily development-league stat sync for this league."""
    return 24 if normalize_prospect_league_key(code) == "NCAA" else 20


def _stable_rng(prospect: Any, tag: str, *parts: Any) -> random.Random:
    """Deterministic RNG per (player, tag, parts) — draws that must not change between calls."""
    ident = getattr(prospect, "identity", None)
    pid = getattr(prospect, "id", None) or getattr(ident, "name", None) or ""
    seed = getattr(prospect, "rng_seed", None)
    if not pid and seed is None:
        pid = id(prospect)
    raw = "|".join(str(x) for x in (pid, seed, tag) + tuple(parts))
    return random.Random(int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:12], 16))


def _season_year_of(prospect: Any) -> int:
    return _safe_int(getattr(prospect, "_prospect_season_year", 0), 0)


def _seed_year(prospect: Any, fallback: Optional[int] = None) -> int:
    """Year used to seed this season's per-player draws.

    Pinned when the season is initialised so the projection and the live PPG target use the
    *same* usage/luck draw (players spawned before the calendar year is known would otherwise
    project with one draw and play with another, faking over/under-performance).
    """
    pinned = getattr(prospect, "_prospect_ppg_seed_year", None)
    if pinned is not None:
        return _safe_int(pinned, 0)
    if fallback is not None:
        return int(fallback)
    return _season_year_of(prospect)


def _poisson(lam: float, rng: random.Random) -> int:
    lam = max(0.0, float(lam))
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        threshold = math.exp(-lam)
        prod = 1.0
        k = 0
        while prod > threshold:
            k += 1
            prod *= rng.random()
        return max(0, k - 1)
    return max(0, int(round(lam + rng.gauss(0.0, math.sqrt(lam)))))


def _binomial(n: int, p: float, rng: random.Random) -> int:
    n = max(0, int(n))
    p = _clamp(p, 0.0, 1.0)
    if n == 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    return sum(1 for _ in range(n) if rng.random() < p)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def normalize_prospect_league_key(league: Any) -> str:
    """Map varied league codes/names to a scoring profile key."""
    raw = str(league or "").strip()
    if not raw:
        return "JUNIOR"
    up = raw.upper().replace("—", "-").replace("–", "-")
    if up in LEAGUE_ALIASES:
        return LEAGUE_ALIASES[up]
    compact = up.replace("_", " ").replace("-", " ")
    if compact in LEAGUE_ALIASES:
        return LEAGUE_ALIASES[compact]
    for alias, key in LEAGUE_ALIASES.items():
        if alias in up or up in alias:
            return key
    if up.startswith("CHL"):
        if "QMJ" in up or "QUEBEC" in up:
            return "QMJHL"
        if "WHL" in up or "WESTERN" in up:
            return "WHL"
        if "OHL" in up or "ONTARIO" in up:
            return "OHL"
        return "CHL"
    if up.startswith("EU_J"):
        return "EUROPE_JUNIOR"
    return "JUNIOR"


def get_league_scoring_profile(league: Any) -> Dict[str, Any]:
    key = normalize_prospect_league_key(league)
    prof = dict(LEAGUE_SCORING_PROFILES.get(key) or LEAGUE_SCORING_PROFILES["JUNIOR"])
    prof["profile_key"] = key
    return prof


def _player_position(prospect: Any) -> str:
    pos = getattr(prospect, "position", None)
    if pos is not None:
        v = getattr(pos, "value", None)
        if v is not None:
            return str(v).upper()
        return str(pos).upper()
    ident = getattr(prospect, "identity", None)
    if ident is not None:
        ip = getattr(ident, "position", None)
        if ip is not None:
            return str(getattr(ip, "value", ip)).upper()
    return "F"


def _is_goalie(prospect: Any) -> bool:
    return _player_position(prospect) in ("G", "GOALIE", "GOALTENDER")


def _is_defense(prospect: Any) -> bool:
    return _player_position(prospect) in ("D", "LD", "RD", "DEF", "DEFENSE")


def _player_age(prospect: Any) -> int:
    ident = getattr(prospect, "identity", None)
    if ident is not None:
        return _safe_int(getattr(ident, "age", 18), 18)
    return _safe_int(getattr(prospect, "age", 18), 18)


def invalidate_prospect_analytics_cache(prospect: Any) -> None:
    """Clear cached immutable analytics after ratings or identity change."""
    for attr in (
        "_prospect_cached_ovr_0_1",
        "_prospect_cached_playstyle_bucket",
        "_prospect_cached_offensive_talent",
        "_prospect_cached_defensive_analytics",
    ):
        try:
            delattr(prospect, attr)
        except Exception:
            pass


def _player_ovr_0_1(prospect: Any) -> float:
    cached = getattr(prospect, "_prospect_cached_ovr_0_1", None)
    if cached is not None:
        return float(cached)
    ovr_fn = getattr(prospect, "ovr", None)
    if callable(ovr_fn):
        try:
            v = float(ovr_fn())
        except Exception:
            v = 0.5
    else:
        v = _safe_float(ovr_fn, 0.5)
    if v > 1.5:
        v /= 100.0
    v = _clamp(v, 0.15, 0.99)
    try:
        setattr(prospect, "_prospect_cached_ovr_0_1", v)
    except Exception:
        pass
    return v


def _draft_mid(prospect: Any) -> float:  # HIDDEN-TRUTH: draft/board code only — never feed into stats/analytics
    dr = getattr(prospect, "draft_value_range", None)
    if dr and len(dr) >= 2:
        try:
            return _clamp((float(dr[0]) + float(dr[1])) / 2.0, 0.2, 0.99)
        except (TypeError, ValueError):
            pass
    pot = getattr(prospect, "potential", None)
    if pot is not None:
        pv = _safe_float(pot, 0.55)
        if pv > 1.5:
            pv /= 100.0
        return _clamp(pv, 0.2, 0.99)
    return _clamp(_player_ovr_0_1(prospect) * 1.08, 0.2, 0.99)


def _playstyle_bucket(prospect: Any) -> str:
    cached = getattr(prospect, "_prospect_cached_playstyle_bucket", None)
    if isinstance(cached, str) and cached:
        return cached
    chem = getattr(prospect, "chemistry_profile", None)
    chem_style = ""
    if isinstance(chem, dict):
        chem_style = str(chem.get("playstyle") or "")
    parts = [
        chem_style,
        str(getattr(prospect, "playstyle", "") or ""),
        str(getattr(prospect, "archetype", "") or ""),
        str(getattr(prospect, "player_type", "") or ""),
    ]
    blob = " ".join(parts).lower().replace("_", " ").replace("-", " ")
    if _is_goalie(prospect):
        bucket = "goalie"
    elif _is_defense(prospect):
        if "offensive" in blob or "quarterback" in blob or "puck mover" in blob:
            bucket = "offensive_defenseman"
        elif "shutdown" in blob or "stay at home" in blob or "defensive" in blob:
            bucket = "defensive_defenseman"
        else:
            bucket = "two_way_defenseman"
    elif any(x in blob for x in ("sniper", "shooter", "goal scorer")):
        bucket = "sniper"
    elif any(x in blob for x in ("playmaker", "play maker", "passer")):
        bucket = "playmaker"
    elif any(x in blob for x in ("power", "power forward")):
        bucket = "power_forward"
    elif any(x in blob for x in ("grinder", "checker", "energy")):
        bucket = "grinder"
    elif any(x in blob for x in ("two way", "two_way", "two way", "200 foot")):
        bucket = "two_way"
    elif "shutdown" in blob and not _is_defense(prospect):
        bucket = "two_way"
    else:
        bucket = "scoring_forward"
    try:
        setattr(prospect, "_prospect_cached_playstyle_bucket", bucket)
    except Exception:
        pass
    return bucket


def _offensive_talent_score(prospect: Any) -> float:
    """0–1 *public* offensive talent estimate: current OVR + offensive ratings + style only.

    Deliberately excludes potential, pipeline tier, steal/bust flags and dev type so the
    analytics and stock movement built on it cannot leak hidden ceiling information.
    """
    cached = getattr(prospect, "_prospect_cached_offensive_talent", None)
    if cached is not None:
        return float(cached)
    ovr = _player_ovr_0_1(prospect)
    score = 0.0

    off_avg: Optional[float] = None
    ratings = getattr(prospect, "ratings", None)
    if isinstance(ratings, dict) and ratings:
        off_vals: List[float] = []
        for k, v in ratings.items():
            kl = str(k).lower()
            if any(x in kl for x in ("shot", "shoot", "pass", "off", "puck", "skill", "handling", "accuracy")):
                off_vals.append(_safe_float(v, 50.0) / 99.0)
        if off_vals:
            off_avg = sum(off_vals) / len(off_vals)
    if off_avg is None:
        score += ovr * 0.90
    else:
        score += ovr * 0.50 + off_avg * 0.40

    style = _playstyle_bucket(prospect)
    if style in ("sniper", "playmaker", "power_forward", "scoring_forward", "offensive_defenseman"):
        score += 0.05
    elif style in ("grinder", "defensive_defenseman"):
        score -= 0.06

    score = _clamp(score, 0.12, 0.98)
    try:
        setattr(prospect, "_prospect_cached_offensive_talent", score)
    except Exception:
        pass
    return score


def _has_character_concerns(prospect: Any) -> bool:
    if getattr(prospect, "pipeline_bust", False):
        return True
    dev = str(getattr(prospect, "dev_type", "") or "").lower()
    if dev in ("bust", "volatile"):
        return True
    arch = str(getattr(prospect, "_dev_archetype", "") or "").upper()
    if arch in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE"):
        return True
    psych = getattr(prospect, "psychology", None)
    if psych is not None:
        coach = _safe_float(getattr(psych, "coachability", 0.5), 0.5)
        anx = _safe_float(getattr(psych, "anxiety", 0.35), 0.35)
        if coach < 0.38 or anx > 0.62:
            return True
    traits = getattr(prospect, "traits", None)
    if isinstance(traits, dict):
        tags = [str(t).lower() for t in (traits.get("tags") or [])]
        if any(t in tags for t in ("boom_bust", "volatile", "selfish", "attitude", "character_concern")):
            return True
    chem = getattr(prospect, "chemistry_profile", None)
    if isinstance(chem, dict):
        pers = str(chem.get("personality", "") or "").lower()
        if pers in ("volatile", "mercurial", "selfish"):
            return True
    return False


def _is_boom_bust(prospect: Any) -> bool:
    curve = str(getattr(prospect, "_pipeline_dev_curve", "") or getattr(prospect, "_dev_curve_hint", "") or "").lower()
    if curve == "boom_bust":
        return True
    arch = str(getattr(prospect, "_dev_archetype", "") or "").upper()
    if arch in ("HIGH_VARIANCE", "ELITE_CEILING_VOLATILE"):
        return True
    traits = getattr(prospect, "traits", None)
    if isinstance(traits, dict):
        tags = [str(t).lower() for t in (traits.get("tags") or [])]
        return "boom_bust" in tags
    return False


def _default_games_played(profile: Dict[str, Any], age: int, rng: random.Random) -> int:
    lo, hi = profile.get("gp_range", (54, 64))
    gp = rng.randint(int(lo), int(hi))
    if age <= 17:
        gp = int(gp * rng.uniform(0.82, 0.92))
    elif age >= 20:
        gp = int(gp * rng.uniform(0.96, 1.02))
    return max(28, min(72, gp))


def _parse_calendar_iso(calendar_iso: Any) -> Tuple[int, int]:
    raw = str(calendar_iso or "").strip()
    if not raw:
        return 9, 15
    try:
        d = date.fromisoformat(raw[:10])
        return int(d.month), int(d.day)
    except (TypeError, ValueError):
        return 9, 15


def _league_season_fraction(league: Any, month: int, day: int) -> float:
    """Return 0..1 — how much of the prospect league regular season has elapsed."""
    profile = get_league_scoring_profile(league)
    key = str(profile.get("profile_key") or "JUNIOR")
    offset = _safe_float(_LEAGUE_FRAC_OFFSET.get(key, 0.0), 0.0)

    m = int(month)
    if m not in _BASE_MONTH_FRAC:
        if m < 7:
            base = 1.0
        else:
            base = 0.0
    else:
        base = _BASE_MONTH_FRAC[m]
        next_m = m + 1 if m < 12 else 1
        next_base = _BASE_MONTH_FRAC.get(next_m, base)
        day_frac = _clamp((int(day) - 1) / 31.0, 0.0, 1.0)
        base = base + (next_base - base) * day_frac

    frac = _clamp(base + offset, 0.0, 1.0)
    if m == 7:
        frac = 0.0
    return frac


def expected_games_for_date(league: Any, projected_gp: int, calendar_iso: Any, start_frac: float = 0.0) -> int:
    """How many GP should have occurred by calendar_iso for this league.

    ``start_frac`` is the fraction of the season already gone when this stint began, so a
    player who arrives mid-season is not credited games played before he got there.
    """
    month, day = _parse_calendar_iso(calendar_iso)
    frac = _league_season_fraction(league, month, day)
    start = _clamp(start_frac, 0.0, 1.0)
    cap = max(0, int(round(int(projected_gp) * (1.0 - start))))
    gp = int(round(int(projected_gp) * max(0.0, frac - start)))
    return max(0, min(cap, gp))


def _stint(prospect: Any) -> Optional[Dict[str, Any]]:
    st = getattr(prospect, "_prospect_stint", None)
    return st if isinstance(st, dict) else None


def _stint_team_label(prospect: Any) -> str:
    asg = getattr(prospect, "_franchise_assignment", None)
    if isinstance(asg, dict) and asg.get("club"):
        return str(asg.get("club"))
    return str(getattr(getattr(prospect, "context", None), "current_team_id", "") or "")


def begin_prospect_stint(
    prospect: Any,
    league: Any,
    *,
    start_frac: float = 0.0,
    iso: str = "",
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Open a stat stint: one league + team for one continuous stretch of one season.

    A stat line belongs to the stint, not the player. ``start_frac`` is how much of the
    season had passed when the player arrived — games before that are never credited.
    """
    st = {
        "league_key": normalize_prospect_league_key(league),
        "team": _stint_team_label(prospect),
        "start_frac": float(_clamp(start_frac, 0.0, 1.0)),
        "start_iso": str(iso or "")[:10],
        "last_live_iso": str(iso or "")[:10],
        "season_year": int(season_year) if season_year is not None else _season_year_of(prospect),
        "open": True,
    }
    try:
        setattr(prospect, "_prospect_stint", st)
    except Exception:
        pass
    return st


def archive_prospect_stint(prospect: Any, *, reason: str = "left_level") -> bool:
    """Close the open stint and file its line in ``prospect_stat_history`` (if any games)."""
    st = _stint(prospect)
    if st is None or not st.get("open", True):
        return False
    st["open"] = False
    actual = getattr(prospect, "_prospect_season_stats", None)
    if not isinstance(actual, dict):
        return False
    gp = _safe_int(actual.get("gp"), 0)
    if gp <= 0:
        return False
    row: Dict[str, Any] = {
        "season_year": st.get("season_year"),
        "league": st.get("league_key"),
        "team": st.get("team"),
        "reason": reason,
        "start_iso": st.get("start_iso"),
        "gp": gp,
        "synthetic": True,
    }
    if _is_goalie(prospect):
        for k in ("wins", "losses", "ot_losses", "save_pct", "gaa", "shutouts", "shots_against", "goals_against"):
            if actual.get(k) is not None:
                row[k] = actual.get(k)
    else:
        for k in ("goals", "assists", "points", "ppg", "pim"):
            if actual.get(k) is not None:
                row[k] = actual.get(k)
    hist = list(getattr(prospect, "prospect_stat_history", None) or [])
    hist.append(row)
    try:
        setattr(prospect, "prospect_stat_history", hist[-24:])
    except Exception:
        pass
    return True


def end_prospect_stint(prospect: Any, reason: str = "left_level") -> bool:
    return archive_prospect_stint(prospect, reason=reason)


def _empty_actual_stat_line() -> Dict[str, Any]:
    return {
        "gp": 0,
        "games_played": 0,
        "goals": 0,
        "assists": 0,
        "points": 0,
        "ppg": 0.0,
        "points_per_game": 0.0,
        "pim": 0,
        "wins": 0,
        "losses": 0,
        "ot_losses": 0,
        "save_pct": 0.0,
        "gaa": 0.0,
        "shutouts": 0,
        "stat_source": "calendar_sim",
    }


def _recalc_skater_line_from_totals(stats: Dict[str, Any], style: str = "", rng: Optional[random.Random] = None) -> None:
    """Refresh per-game rates from the *accumulated* line. Never re-splits goals/assists."""
    gp = max(0, _safe_int(stats.get("gp"), 0))
    pts = max(0, _safe_int(stats.get("points"), 0))
    goals = max(0, _safe_int(stats.get("goals"), 0))
    if goals > pts:
        goals = pts
    stats["goals"] = goals
    stats["assists"] = max(0, pts - goals)
    if gp <= 0:
        stats["ppg"] = 0.0
        stats["points_per_game"] = 0.0
        return
    ppg = round(pts / gp, 3)
    stats["ppg"] = ppg
    stats["points_per_game"] = ppg
    stats["goals_per_game"] = round(stats["goals"] / gp, 3)
    stats["assists_per_game"] = round(stats["assists"] / gp, 3)


def _sample_prospect_game_points(
    target_ppg: float,
    rng: random.Random,
    *,
    overdispersion: float = 0.0,
    hot_game: bool = False,
    elite_tail: bool = False,
) -> int:
    """Single-game point draw (kept for callers/tests). Season sim uses _sample_points_block."""
    lam = max(0.0, float(target_ppg))
    if hot_game:
        lam *= rng.uniform(1.14, 1.38)
    elif elite_tail and rng.random() < 0.022:
        lam *= rng.uniform(1.22, 1.48)
    elif overdispersion > 0.01:
        lam *= rng.uniform(1.0 - overdispersion * 0.30, 1.0 + overdispersion * 0.45)
    return _poisson(lam, rng)


def _sample_points_block(
    target_ppg: float,
    n: int,
    rng: random.Random,
    *,
    vol: float,
    offensive: float,
    boom_bust: bool,
    concerns: bool,
) -> int:
    """Points over ``n`` games. Same distribution whether n is 1 (daily) or 60 (bulk sim).

    Every per-game effect is applied as an *expected count* over the block (binomial number
    of hot games, scoreless games, tail games) so E[points] == target_ppg * n regardless of
    how the calendar was chunked, and variance scales like a sum of n independent games.
    """
    n = int(n)
    if n <= 0 or target_ppg <= 0.0:
        return 0
    p_hot = (0.035 if offensive >= 0.80 else 0.0) + (0.012 if boom_bust else 0.0)
    hot = _binomial(n, p_hot, rng) if p_hot > 0 else 0
    tail = _binomial(n, 0.022, rng) if offensive >= 0.76 else 0
    zeros = _binomial(n, 0.04, rng) if concerns else 0
    lam = target_ppg * max(0, n - zeros)
    lam += target_ppg * hot * 0.26  # E[hot multiplier - 1] = mean(1.14, 1.38) - 1
    lam += target_ppg * tail * 0.35  # E[tail multiplier - 1] = mean(1.22, 1.48) - 1
    if vol > 0.01:
        lo, hi = 1.0 - 0.30 * vol, 1.0 + 0.45 * vol
        mean_u = 0.5 * (lo + hi)
        lam *= max(0.2, 1.0 + (rng.uniform(lo, hi) - mean_u) / math.sqrt(n))
    return _poisson(lam, rng)


def _prospect_role_multiplier(prospect: Any, league: Any) -> float:
    """Deployment opportunity from explicit role flags and age only.

    Talent and league scale are already in the PPG curve; multiplying them in again here
    double-counted both. PP1 / top-line flags are honoured when a roster system sets them.
    """
    age = _player_age(prospect)
    mult = 1.0
    if age <= 17:
        mult *= 0.94
    elif age == 18:
        mult *= 0.98
    elif age >= 20:
        mult *= 1.04
    if getattr(prospect, "pp1_usage", False) or str(getattr(prospect, "pp_role", "") or "").upper() == "PP1":
        mult *= 1.08
    if str(getattr(prospect, "line_role", "") or "").lower() in ("top", "top_line", "first"):
        mult *= 1.06
    return _clamp(mult, 0.55, 1.12)


def _volatility_factor(prospect: Any) -> float:
    vol = 0.14
    if _is_boom_bust(prospect):
        vol += 0.18
    if _has_character_concerns(prospect):
        vol += 0.10
    if getattr(prospect, "pipeline_bust", False):
        vol += 0.08
    return _clamp(vol, 0.10, 0.42)


def _simulate_skater_games(
    prospect: Any,
    league: Any,
    delta_gp: int,
    rng: random.Random,
    stats: Dict[str, Any],
    target_ppg: float,
) -> None:
    """Advance a skater's running line by ``delta_gp`` games.

    One code path for daily and bulk advances: the block sampler gives the same
    expected total and variance however the games are chunked, and goals/assists
    accumulate (only newly scored points are split), so totals never go backwards.
    """
    n = int(delta_gp)
    if n <= 0:
        return
    style = _playstyle_bucket(prospect)
    vol = _volatility_factor(prospect)
    skills = _prospect_event_skills(prospect)
    offensive = (
        skills["volume"] * 0.18
        + skills["finishing"] * 0.24
        + skills["playmaking"] * 0.32
        + skills["process"] * 0.26
    )
    pts = _sample_points_block(
        target_ppg,
        n,
        rng,
        vol=vol,
        offensive=offensive,
        boom_bust=_is_boom_bust(prospect),
        concerns=_has_character_concerns(prospect),
    )
    _add_points(stats, pts, _season_goal_share(prospect, style), rng)

    # Spread the block's points over its games so recent-form / weekly heat stay realistic.
    games = [0] * n
    for _ in range(pts):
        games[rng.randrange(n)] += 1
    recent = (list(stats.get("_recent_game_points") or []) + games)[-8:]

    if style in ("grinder", "power_forward"):
        pim_mean, pim_sd = 2.0, 1.4
    else:
        pim_mean, pim_sd = 0.35, 0.6
    stats["pim"] = _safe_int(stats.get("pim"), 0) + max(
        0, int(round(rng.gauss(n * pim_mean, math.sqrt(n) * pim_sd)))
    )
    stats["gp"] = _safe_int(stats.get("gp"), 0) + n
    stats["games_played"] = stats["gp"]
    stats["_recent_game_points"] = recent
    _recalc_skater_line_from_totals(stats)


def _goalie_ability(prospect: Any) -> float:
    """0–1 goalie quality from goalie ratings + current OVR (skater tools are irrelevant)."""
    ovr = _player_ovr_0_1(prospect)
    ratings = getattr(prospect, "ratings", None)
    if isinstance(ratings, dict) and ratings:
        vals = [_safe_float(v, 50.0) / 99.0 for k, v in ratings.items() if str(k).lower().startswith("g_")]
        if vals:
            return _clamp(0.5 * ovr + 0.5 * (sum(vals) / len(vals)), 0.15, 0.95)
    return _clamp(ovr, 0.15, 0.95)


def _simulate_goalie_games(
    prospect: Any,
    league: Any,
    delta_gp: int,
    rng: random.Random,
    stats: Dict[str, Any],
    projected: Dict[str, Any],
) -> None:
    """Advance a goalie by ``delta_gp`` games using shots against and goals against.

    SV% and GAA are *derived* from accumulated shots/goals, so they stay consistent with
    each other and with the season total (the old model averaged the last 12 noise draws).
    """
    n = int(delta_gp)
    if n <= 0:
        return
    vol = _volatility_factor(prospect)
    proj_gp = max(1, _safe_int(projected.get("gp"), 40))
    sv_true = _clamp(_safe_float(projected.get("save_pct"), 0.895), 0.84, 0.95)
    sa_pg = max(15.0, _safe_float(projected.get("shots_against_per_game"), 29.0))
    win_rate = _clamp(_safe_int(projected.get("wins"), int(proj_gp * 0.5)) / float(proj_gp), 0.25, 0.75)

    sa = max(n, int(round(rng.gauss(n * sa_pg, math.sqrt(n) * 5.5))))
    lam = sa * (1.0 - sv_true)
    if vol > 0.01:
        lo, hi = 1.0 - 0.15 * vol, 1.0 + 0.20 * vol
        lam *= max(0.3, 1.0 + (rng.uniform(lo, hi) - 0.5 * (lo + hi)) / math.sqrt(n))
    ga = min(sa, _poisson(lam, rng))

    stats["shots_against"] = _safe_int(stats.get("shots_against"), 0) + sa
    stats["goals_against"] = _safe_int(stats.get("goals_against"), 0) + ga
    stats["saves"] = stats["shots_against"] - stats["goals_against"]
    gp_total = _safe_int(stats.get("gp"), 0) + n
    stats["gp"] = gp_total
    stats["games_played"] = gp_total
    if stats["shots_against"] > 0:
        sv_now = 1.0 - stats["goals_against"] / float(stats["shots_against"])
        stats["save_pct"] = round(sv_now, 3)
        stats["savePct"] = stats["save_pct"]
    stats["gaa"] = round(stats["goals_against"] / float(max(1, gp_total)), 2)

    sv_block = 1.0 - ga / float(max(1, sa))
    p_win = _clamp(win_rate + (sv_block - sv_true) * 3.0, 0.15, 0.85)
    w_new = _binomial(n, p_win, rng)
    otl_new = _binomial(n - w_new, 0.12, rng)
    stats["wins"] = _safe_int(stats.get("wins"), 0) + w_new
    stats["ot_losses"] = _safe_int(stats.get("ot_losses"), 0) + otl_new
    stats["losses"] = _safe_int(stats.get("losses"), 0) + (n - w_new - otl_new)
    ga_pg = sa_pg * (1.0 - sv_true)
    stats["shutouts"] = _safe_int(stats.get("shutouts"), 0) + _binomial(n, math.exp(-ga_pg), rng)
    stats["ppg"] = 0.0
    stats["points_per_game"] = 0.0


def _defensive_analytics_score(prospect: Any) -> float:
    """0–1 two-way / defensive process estimate."""
    cached = getattr(prospect, "_prospect_cached_defensive_analytics", None)
    if cached is not None:
        return float(cached)
    ratings = getattr(prospect, "ratings", None)
    if not isinstance(ratings, dict) or not ratings:
        score = 0.45
    else:
        vals: List[float] = []
        for k, v in ratings.items():
            kl = str(k).lower()
            if any(x in kl for x in ("def", "stick", "check", "position", "aware", "gap", "transition", "skating")):
                vals.append(_safe_float(v, 50.0) / 99.0)
        if not vals:
            style = _playstyle_bucket(prospect)
            if style in ("defensive_defenseman", "two_way", "grinder"):
                score = 0.62
            elif style in ("offensive_defenseman", "power_forward"):
                score = 0.48
            else:
                score = 0.42
        else:
            score = _clamp(sum(vals) / len(vals), 0.15, 0.95)
    try:
        setattr(prospect, "_prospect_cached_defensive_analytics", score)
    except Exception:
        pass
    return score


def _production_vs_projection(actual: Dict[str, Any], projected: Dict[str, Any], prospect: Any) -> float:
    """Normalized production signal roughly -1..+1."""
    gp = _safe_int(actual.get("gp"), 0)
    if gp <= 0:
        return 0.0
    if _is_goalie(prospect):
        proj_sv = _safe_float(projected.get("save_pct"), 0.905)
        act_sv = _safe_float(actual.get("save_pct"), proj_sv)
        return _clamp((act_sv - proj_sv) / 0.035, -1.0, 1.0)
    proj_ppg = _safe_float(projected.get("ppg", projected.get("points_per_game")), 0.55)
    actual_ppg = _safe_float(actual.get("ppg", actual.get("points_per_game")), 0.0)
    if proj_ppg <= 0.01:
        return 0.0
    ratio = (actual_ppg - proj_ppg) / max(0.08, proj_ppg)
    return _clamp(ratio, -1.0, 1.0)


def _analytics_process_score(prospect: Any, actual: Dict[str, Any], projected: Dict[str, Any]) -> float:
    """Normalized analytics / process signal roughly -1..+1."""
    offense = _offensive_talent_score(prospect)
    defense = _defensive_analytics_score(prospect)
    style = _playstyle_bucket(prospect)
    pos = str(getattr(getattr(prospect, "identity", None), "position", "") or "").upper()

    base = (offense - 0.45) * 1.35
    if pos == "D" or style in ("defensive_defenseman", "two_way", "grinder"):
        base = base * 0.55 + (defense - 0.45) * 1.25
    elif pos == "C" and style == "two_way":
        base = base * 0.72 + (defense - 0.45) * 0.85

    recent: List[int] = list(actual.get("_recent_game_points") or [])
    if len(recent) >= 3:
        avg3 = sum(recent[-3:]) / 3.0
        target = _safe_float(projected.get("ppg", projected.get("points_per_game")), 0.55)
        if target > 0.01:
            base += _clamp((avg3 - target) / max(0.12, target), -0.35, 0.35) * 0.45

    return _clamp(base, -1.0, 1.0)


def _stock_label_from_delta(signed: int) -> Tuple[str, str]:
    if signed >= 26:
        return "Rocketing", "Rising"
    if signed >= 13:
        return "Surging", "Rising"
    if signed >= 5:
        return "Rising", "Rising"
    if signed >= 3:
        return "Trending Up", "Rising"
    if signed <= -26:
        return "Crashing", "Falling"
    if signed <= -13:
        return "Falling", "Falling"
    if signed <= -5:
        return "Slipping", "Falling"
    if signed <= -3:
        return "Slipping", "Falling"
    return "Holding", "Holding"


def _stock_reason_from_signals(
    prospect: Any,
    signed: int,
    production: float,
    analytics: float,
) -> str:
    if signed >= 13 and production > 0.12 and analytics > 0.12:
        return "Production spike matches strong tracking data."
    if signed >= 6 and production > 0.12 and analytics <= 0.0:
        return "Points are up, process remains weak."
    if signed >= 6 and production <= 0.0 and analytics > 0.12:
        return "Elite chance creation despite quiet scoring."
    if signed >= 6 and analytics > 0.15 and _playstyle_bucket(prospect) in ("defensive_defenseman", "two_way", "grinder"):
        return "Strong transition profile driving rise."
    if signed <= -13 and production < -0.12 and analytics < -0.12:
        return "Poor offense and weak defensive impact."
    if signed <= -6 and production < -0.12:
        return "Production slump dragging public stock down."
    if signed <= -6 and analytics < -0.12:
        return "Underlying process no longer supports the hype."
    if signed >= 6:
        return "Scouts are moving him up the board."
    if signed <= -6:
        return "Draft buzz cooling after recent tape."
    return "Holding steady on the public board."


def _iso_week_key(calendar_iso: Any) -> str:
    """ISO year-week key for weekly draft-stock snapshots."""
    try:
        from datetime import datetime

        dt = datetime.strptime(str(calendar_iso or "")[:10], "%Y-%m-%d")
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    except Exception:
        return str(calendar_iso or "")[:7]


def _sync_prospect_week_baseline(prospect: Any, actual: Dict[str, Any], week_key: str) -> None:
    stored = getattr(prospect, "_prospect_week_baseline", None)
    if isinstance(stored, dict) and str(stored.get("week") or "") == week_key:
        return
    try:
        setattr(
            prospect,
            "_prospect_week_baseline",
            {
                "week": week_key,
                "gp": _safe_int(actual.get("gp"), 0),
                "points": _safe_int(actual.get("points"), 0),
                "goals": _safe_int(actual.get("goals"), 0),
                "assists": _safe_int(actual.get("assists"), 0),
            },
        )
    except Exception:
        pass


def _week_stat_delta(prospect: Any, actual: Dict[str, Any]) -> Dict[str, int]:
    baseline = getattr(prospect, "_prospect_week_baseline", None) or {}
    return {
        "gp": max(0, _safe_int(actual.get("gp"), 0) - _safe_int(baseline.get("gp"), 0)),
        "points": max(0, _safe_int(actual.get("points"), 0) - _safe_int(baseline.get("points"), 0)),
        "goals": max(0, _safe_int(actual.get("goals"), 0) - _safe_int(baseline.get("goals"), 0)),
        "assists": max(0, _safe_int(actual.get("assists"), 0) - _safe_int(baseline.get("assists"), 0)),
    }


def _weekly_stock_label(signed: int) -> tuple:
    """Labels for the weekly heat scale (−8…+8), not the season ±26 scale."""
    if signed >= 5:
        return "Rocketing", "Rising"
    if signed >= 3:
        return "Rising", "Rising"
    if signed >= 1:
        return "Trending Up", "Rising"
    if signed <= -5:
        return "Crashing", "Falling"
    if signed <= -3:
        return "Falling", "Falling"
    if signed <= -1:
        return "Slipping", "Falling"
    return "Holding", "Holding"


def _compact_stock_reason(
    prospect: Any,
    signed: int,
    *,
    production: float,
    analytics: float,
    week_gp: int,
    week_pts: int,
    week_ppg: float,
    is_goalie: bool = False,
    sv: float = 0.0,
    gaa: float = 0.0,
) -> str:
    """Short stat-backed reason — no scout-market fluff."""
    if week_gp <= 0:
        return "No games this week."
    if is_goalie:
        if signed >= 3:
            return f"SV% {sv:.3f} · GAA {gaa:.2f} · {week_gp} GP"
        if signed <= -3:
            return f"Cold starts · SV% {sv:.3f} · GAA {gaa:.2f}"
        return f"{week_gp} GP · SV% {sv:.3f}"
    if signed >= 3 and production > 0.1:
        return f"{week_pts}P in {week_gp} GP · {week_ppg:.2f} PPG"
    if signed >= 2 and analytics > 0.1 and production <= 0.05:
        return f"Process up · {week_ppg:.2f} PPG · {week_gp} GP"
    if signed <= -3 and production < -0.1:
        return f"Cold week · {week_pts}P in {week_gp} GP"
    if signed <= -2 and analytics < -0.1:
        return f"Process down · {week_ppg:.2f} PPG"
    if abs(signed) <= 1:
        return f"{week_pts}P / {week_gp} GP · holding"
    style = _playstyle_bucket(prospect)
    if style in ("defensive_defenseman", "two_way_defenseman", "two_way", "grinder") and analytics > 0.08:
        return f"D impact · {week_ppg:.2f} PPG · {week_gp} GP"
    return f"{week_pts}P in {week_gp} GP · {week_ppg:.2f} PPG"


def _compute_weekly_goalie_stock_fields(
    prospect: Any,
    week_stats: Dict[str, int],
    projected: Dict[str, Any],
    actual: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Goalie weekly heat from SV%/GAA vs projected line — not skater PPG."""
    week_gp = _safe_int(week_stats.get("gp"), 0)
    if week_gp <= 0:
        return {
            "weekly_stock_delta": 0,
            "weekly_stock_label": "Holding",
            "weekly_stock_trend": "Holding",
            "weekly_stock_reason": "No games this week.",
            "weekly_production_score": 0.0,
            "weekly_analytics_score": 0.0,
            "week_gp": 0,
            "week_points": 0,
            "stock_mode": "weekly_heat",
            "stock_unit": "heat",
        }
    actual = actual if isinstance(actual, dict) else {}
    sv = _safe_float(actual.get("save_pct"), _safe_float(projected.get("save_pct"), 0.905))
    gaa = _safe_float(actual.get("gaa"), _safe_float(projected.get("gaa"), 2.85))
    proj_sv = _safe_float(projected.get("save_pct"), 0.905)
    proj_gaa = _safe_float(projected.get("gaa"), 2.85)
    # Soft weekly bar — don't demand season-peak every week.
    bar_sv = max(0.880, proj_sv - 0.008)
    bar_gaa = proj_gaa + 0.12
    sv_gap = (sv - bar_sv) / 0.025
    gaa_gap = (bar_gaa - gaa) / 0.35
    production = _clamp(0.55 * sv_gap + 0.45 * gaa_gap, -1.0, 1.0)
    analytics = _clamp(sv_gap * 0.4, -0.6, 0.6)
    base = production * 5.5 + analytics * 2.5
    signed = int(round(_clamp(base, -8, 8)))
    if week_gp <= 2:
        signed = int(_clamp(signed, -2, 2))
    label, trend = _weekly_stock_label(signed)
    reason = _compact_stock_reason(
        prospect, signed, production=production, analytics=analytics,
        week_gp=week_gp, week_pts=0, week_ppg=0.0, is_goalie=True, sv=sv, gaa=gaa,
    )
    return {
        "weekly_stock_delta": signed,
        "weekly_stock_label": label,
        "weekly_stock_trend": trend,
        "weekly_stock_reason": reason,
        "weekly_production_score": round(production, 3),
        "weekly_analytics_score": round(analytics, 3),
        "week_gp": week_gp,
        "week_points": 0,
        "stock_mode": "weekly_heat",
        "stock_unit": "heat",
    }


def _compute_weekly_stock_fields(
    prospect: Any,
    week_stats: Dict[str, int],
    projected: Dict[str, Any],
    league: Any,
    *,
    actual: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Non-cumulative weekly stock from this ISO week's games only."""
    if _is_goalie(prospect):
        return _compute_weekly_goalie_stock_fields(prospect, week_stats, projected, actual=actual)

    week_gp = _safe_int(week_stats.get("gp"), 0)
    week_pts = _safe_int(week_stats.get("points"), 0)
    season_stats = getattr(prospect, "_prospect_season_stats", None)
    recent: List[int] = []
    if isinstance(season_stats, dict):
        recent = list(season_stats.get("_recent_game_points") or [])

    # Do NOT invent a week from prior-week recent form — only true week_delta counts.
    if week_gp <= 0:
        return {
            "weekly_stock_delta": 0,
            "weekly_stock_label": "Holding",
            "weekly_stock_trend": "Holding",
            "weekly_stock_reason": "No games this week.",
            "weekly_production_score": 0.0,
            "weekly_analytics_score": 0.0,
            "week_gp": 0,
            "week_points": 0,
            "stock_mode": "weekly_heat",
            "stock_unit": "heat",
        }

    week_ppg = week_pts / max(1, week_gp)
    proj_ppg = _safe_float(projected.get("ppg", projected.get("points_per_game")), 0.55)
    # Soft weekly bar: elite kids aren't punished for merely matching season pace.
    weekly_bar = max(0.32, proj_ppg * 0.88)
    profile = get_league_scoring_profile(league)
    league_diff = _safe_float(profile.get("difficulty"), 0.65)
    # Harder leagues: same PPG gap counts for more.
    league_mult = 1.0 + max(0.0, league_diff - 0.55) * 0.55

    if weekly_bar <= 0.01:
        production = 0.0
    else:
        production = _clamp((week_ppg - weekly_bar) / max(0.10, weekly_bar), -1.0, 1.0)

    analytics = 0.0
    if len(recent) >= week_gp:
        slice_pts = recent[-week_gp:]
        avg_week = sum(slice_pts) / max(1, len(slice_pts))
        if weekly_bar > 0.01:
            analytics = _clamp((avg_week - weekly_bar) / max(0.14, weekly_bar), -0.5, 0.5)
    style = _playstyle_bucket(prospect)
    process = _analytics_process_score(
        prospect,
        {"gp": week_gp, "points": week_pts, "ppg": week_ppg, "_recent_game_points": recent[-week_gp:] if recent else []},
        projected,
    )
    # Shutdown / two-way profiles get more credit for process weeks.
    process_w = 0.55 if style in ("defensive_defenseman", "two_way_defenseman", "two_way", "grinder", "shutdown_center") else 0.35
    analytics = _clamp(analytics + process * process_w, -1.0, 1.0)

    if production > 0.14 and analytics > 0.08:
        base = 4.2 + (production + analytics) * 5.2
    elif production > 0.08:
        base = 2.0 + production * 6.2
    elif analytics > 0.10 and production >= -0.05:
        base = 1.5 + analytics * 5.2
    elif production < -0.14 and analytics < -0.08:
        base = -4.2 + (production + analytics) * 5.2
    elif production < -0.08:
        base = -2.2 + production * 5.5
    elif analytics < -0.10:
        base = -1.6 + analytics * 4.8
    else:
        base = production * 3.8 + analytics * 2.8

    vol_mult = 1.0 + _volatility_factor(prospect) * 0.30
    signed = int(round(_clamp(base * vol_mult * league_mult, -8, 8)))

    # Character concerns: small downside bias on soft/negative weeks only.
    if bool(getattr(prospect, "character_concerns", False)) and signed <= 1:
        signed = int(_clamp(signed - 1, -8, 8))

    # Tiny samples cannot rocket or crash.
    if week_gp <= 2:
        signed = int(_clamp(signed, -2, 2))
    elif week_gp <= 3:
        signed = int(_clamp(signed, -4, 4))

    # No forced ±1 — true zero weeks stay Holding.
    label, trend = _weekly_stock_label(signed)
    reason = _compact_stock_reason(
        prospect, signed, production=production, analytics=analytics,
        week_gp=week_gp, week_pts=week_pts, week_ppg=week_ppg,
    )

    return {
        "weekly_stock_delta": signed,
        "weekly_stock_label": label,
        "weekly_stock_trend": trend,
        "weekly_stock_reason": reason,
        "weekly_production_score": round(production, 3),
        "weekly_analytics_score": round(analytics, 3),
        "week_gp": week_gp,
        "week_points": week_pts,
        "stock_mode": "weekly_heat",
        "stock_unit": "heat",
    }


def _compute_stock_fields(
    prospect: Any,
    actual: Dict[str, Any],
    projected: Dict[str, Any],
    league: Any,
) -> Dict[str, Any]:
    gp = _safe_int(actual.get("gp"), 0)
    if gp <= 0:
        return {
            "stock_delta": 0,
            "stock_trend": "Holding",
            "stock_label": "Holding",
            "stock_reason": "Waiting on meaningful sample size.",
            "production_score": 0.0,
            "analytics_score": 0.0,
        }

    production = _production_vs_projection(actual, projected, prospect)
    analytics = _analytics_process_score(prospect, actual, projected)

    if production > 0.12 and analytics > 0.12:
        base = 18 + (production + analytics) * 14
    elif production > 0.12 and analytics <= 0.0:
        base = 2 + production * 10
    elif production <= 0.0 and analytics > 0.12:
        base = 6 + analytics * 14
    elif production > 0.0 and analytics > 0.0:
        base = 4 + (production + analytics) * 8
    elif production < -0.12 and analytics < -0.12:
        base = -18 + (production + analytics) * 16
    elif production < -0.12:
        base = -10 + production * 14
    elif analytics < -0.12:
        base = -8 + analytics * 12
    else:
        base = (production * 6) + (analytics * 5)

    if gp < 5:
        sample_weight = 0.35
    elif gp < 10:
        sample_weight = 0.65
    else:
        sample_weight = 1.0
    if gp >= 10:
        sample_weight = min(1.25, sample_weight * 1.15)

    vol_mult = 1.0 + _volatility_factor(prospect)
    if _has_character_concerns(prospect) or _is_boom_bust(prospect):
        vol_mult *= 1.25

    signed = int(round(base * sample_weight * vol_mult))
    signed = int(_clamp(signed, -45, 45))

    label, trend = _stock_label_from_delta(signed)
    if _is_boom_bust(prospect) and abs(signed) >= 8:
        label = "Volatile"
        trend = "Volatile"
    elif _has_character_concerns(prospect) and signed < 0:
        label = "Scout Split"
        trend = "Falling"

    return {
        "stock_delta": signed,
        "stock_trend": trend,
        "stock_label": label,
        "stock_reason": _stock_reason_from_signals(prospect, signed, production, analytics),
        "production_score": round(production, 3),
        "analytics_score": round(analytics, 3),
    }


def initialize_prospect_season(
    prospect: Any,
    league: Any,
    rng: Optional[random.Random] = None,
    *,
    season_year: Optional[int] = None,
    calendar_iso: Optional[str] = None,
    force: bool = False,
    preserve_actual: bool = False,
    stint_start_frac: float = 0.0,
    stint_iso: str = "",
) -> Dict[str, Any]:
    """
    Build the full-season projection and a zeroed actual line for a new stint.
    Does NOT simulate the season upfront. Any previous open stint is filed to history.
    """
    if not force:
        proj = getattr(prospect, "_prospect_projected_stats", None)
        actual = getattr(prospect, "_prospect_season_stats", None)
        if isinstance(proj, dict) and proj.get("gp") and isinstance(actual, dict):
            return dict(actual)

    if not isinstance(rng, random.Random):
        seed = getattr(prospect, "rng_seed", None) or getattr(prospect, "seed", None) or id(prospect)
        rng = random.Random(int(seed) & 0xFFFFFFFF)

    key = normalize_prospect_league_key(league)
    if not preserve_actual:
        archive_prospect_stint(prospect, reason="new_stint")
    if season_year is not None:
        try:
            setattr(prospect, "_prospect_season_year", int(season_year))
        except Exception:
            pass
    sy = int(season_year) if season_year is not None else _season_year_of(prospect)
    try:
        setattr(prospect, "_prospect_ppg_seed_year", sy)
    except Exception:
        pass

    profile = get_league_scoring_profile(league)
    seeded = _stable_rng(prospect, "projection", sy, key)
    proj_gp = _default_games_played(profile, _player_age(prospect), seeded)
    projected = generate_prospect_scoring_line(prospect, league, games_played=proj_gp, rng=seeded)
    projected["stat_source"] = "season_projection"
    projected["projected_gp"] = int(projected.get("gp") or proj_gp)

    prev_actual = getattr(prospect, "_prospect_season_stats", None)
    if preserve_actual and isinstance(prev_actual, dict):
        actual = dict(prev_actual)
    else:
        actual = _empty_actual_stat_line()

    target_ppg = _safe_float(projected.get("ppg", projected.get("points_per_game")), 0.0)
    if not _is_goalie(prospect):
        target_ppg = calculate_prospect_ppg_scale(prospect, league, season_year=sy)

    try:
        setattr(prospect, "_prospect_projected_stats", dict(projected))
        setattr(prospect, "_prospect_expected_ppg", float(target_ppg))
        setattr(prospect, "_prospect_season_stats", dict(actual))
        if not preserve_actual:
            setattr(prospect, "_prospect_last_stat_update_iso", "")
            setattr(prospect, "_prospect_injury_games_remaining", 0)
    except Exception:
        pass
    if not preserve_actual or _stint(prospect) is None:
        begin_prospect_stint(
            prospect, league, start_frac=stint_start_frac, iso=stint_iso, season_year=sy
        )

    if calendar_iso:
        advance_prospect_stats_to_date(
            prospect,
            league,
            calendar_iso,
            rng=rng,
            season_year=season_year,
        )
        actual = getattr(prospect, "_prospect_season_stats", actual) or actual

    return dict(actual) if isinstance(actual, dict) else actual


def _prospect_injury_games(prospect: Any, delta_gp: int, rng: random.Random) -> int:
    """Return GP lost to injury in this advance window.

    Hazard is per *game*, so the expected number of injuries is the same whether the
    calendar advances daily or in one bulk step. ``remaining`` carries an injury that
    outlasts the window into the next one (no game is missed twice).
    """
    if delta_gp <= 0:
        return 0
    remaining = _safe_int(getattr(prospect, "_prospect_injury_games_remaining", 0), 0)
    if remaining > 0:
        missed = min(delta_gp, remaining)
        left = max(0, remaining - missed)
        try:
            setattr(prospect, "_prospect_injury_games_remaining", left)
            if left <= 0:
                setattr(prospect, "prospect_injured", False)
                setattr(prospect, "injured", False)
                setattr(prospect, "injury_status", None)
                setattr(prospect, "injury_note", None)
            else:
                setattr(prospect, "prospect_injured", True)
                setattr(prospect, "injury_status", f"Out {left} GP")
                setattr(prospect, "injury_note", f"Missed {missed} GP this week — {left} remaining")
        except Exception:
            pass
        return missed
    hazard = _INJURY_HAZARD_PER_GAME
    traits = getattr(prospect, "traits", None)
    if traits is not None:
        mod = getattr(traits, "injury_risk_mod", None) if not isinstance(traits, dict) else traits.get("injury_risk_mod")
        if mod is not None:
            hazard += _safe_float(mod, 0.0) * (_INJURY_HAZARD_PER_GAME / 0.014)
    hazard = _clamp(hazard, 0.0, 0.05)
    if rng.random() >= 1.0 - (1.0 - hazard) ** int(delta_gp):
        return 0
    total = rng.randint(1, 8)
    missed = min(int(delta_gp), total)
    left = total - missed
    note = f"Injury — expected to miss {total} GP"
    try:
        setattr(prospect, "_prospect_injury_games_remaining", left)
        setattr(prospect, "prospect_injured", left > 0)
        setattr(prospect, "injured", left > 0)
        setattr(prospect, "injury_status", note if left > 0 else None)
        setattr(prospect, "injury_note", note)
        setattr(prospect, "weekly_stock_reason", note)
    except Exception:
        pass
    return missed


def _apply_stock_to_actual(prospect: Any, actual: Dict[str, Any], weekly_stock: Dict[str, Any]) -> None:
    actual.update(weekly_stock)
    actual["stock_delta"] = weekly_stock.get("weekly_stock_delta", 0)
    actual["stock_label"] = weekly_stock.get("weekly_stock_label", "Holding")
    actual["stock_trend"] = weekly_stock.get("weekly_stock_trend", "Holding")
    actual["stock_reason"] = weekly_stock.get("weekly_stock_reason", "")
    try:
        setattr(prospect, "stock_delta", actual["stock_delta"])
        setattr(prospect, "weekly_stock_delta", actual["stock_delta"])
        setattr(prospect, "stock_label", actual["stock_label"])
        setattr(prospect, "stock_trend", actual["stock_trend"])
        setattr(prospect, "weekly_stock_reason", actual["stock_reason"])
    except Exception:
        pass


def advance_prospect_stats_to_date(
    prospect: Any,
    league: Any,
    calendar_iso: Any,
    rng: Optional[random.Random] = None,
    *,
    season_year: Optional[int] = None,
    expected_gp_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulate only the games owed to this stint between the last update and calendar_iso.

    The result depends on the calendar date reached, not on how many updates it took.
    """
    if not isinstance(rng, random.Random):
        seed = getattr(prospect, "rng_seed", None) or getattr(prospect, "seed", None) or id(prospect)
        rng = random.Random(int(seed) & 0xFFFFFFFF)

    key = normalize_prospect_league_key(league)
    stored_year = getattr(prospect, "_prospect_season_year", None)
    # Reset when the calendar year advances — including the first tick after a
    # rollover where `_prospect_season_year` was never stamped (None). Without
    # that branch, last season's GP/points leak into September of the new year.
    needs_year_reset = False
    if season_year is not None:
        if stored_year is None:
            actual0 = getattr(prospect, "_prospect_season_stats", None)
            gp0 = 0
            if isinstance(actual0, dict):
                try:
                    gp0 = int(actual0.get("gp") or 0)
                except (TypeError, ValueError):
                    gp0 = 0
            needs_year_reset = gp0 > 0
        elif int(season_year) != int(stored_year):
            needs_year_reset = True
    if needs_year_reset:
        initialize_prospect_season(
            prospect,
            league,
            rng=rng,
            season_year=season_year,
            calendar_iso=None,
            force=True,
        )

    projected = getattr(prospect, "_prospect_projected_stats", None)
    if not isinstance(projected, dict) or not projected.get("gp"):
        initialize_prospect_season(
            prospect,
            league,
            rng=rng,
            season_year=season_year,
            calendar_iso=None,
        )
        projected = getattr(prospect, "_prospect_projected_stats", None) or {}

    target_iso = str(calendar_iso or "")[:10]
    month, day = _parse_calendar_iso(target_iso)
    frac_now = _league_season_fraction(league, month, day)

    # ---- stint bookkeeping: a new league / a returning player starts a fresh stint with no backfill
    stint = _stint(prospect)
    if stint is None:
        stint = begin_prospect_stint(
            prospect, league, start_frac=0.0, iso=target_iso, season_year=season_year
        )
    elif (not stint.get("open", True)) or str(stint.get("league_key")) != key:
        initialize_prospect_season(
            prospect,
            league,
            rng=rng,
            season_year=season_year,
            calendar_iso=None,
            force=True,
            stint_start_frac=frac_now,
            stint_iso=target_iso,
        )
        projected = getattr(prospect, "_prospect_projected_stats", None) or projected
        stint = _stint(prospect) or stint
    start_frac = _safe_float(stint.get("start_frac"), 0.0)

    if season_year is not None:
        try:
            setattr(prospect, "_prospect_season_year", int(season_year))
        except Exception:
            pass

    proj_gp = max(1, _safe_int(projected.get("gp"), 60))
    last_iso = str(getattr(prospect, "_prospect_last_stat_update_iso", "") or "")[:10]

    # Catch prior-season lines that kept high GP after a calendar rollover
    # when `_prospect_season_year` was already stamped to the new year.
    try:
        expected_probe = expected_games_for_date(league, proj_gp, target_iso, start_frac) if target_iso else 0
        actual_probe = getattr(prospect, "_prospect_season_stats", None)
        cur_probe = _safe_int(actual_probe.get("gp"), 0) if isinstance(actual_probe, dict) else 0
        if target_iso and cur_probe > int(expected_probe) + 8:
            initialize_prospect_season(
                prospect,
                league,
                rng=rng,
                season_year=season_year,
                calendar_iso=None,
                force=True,
                stint_start_frac=start_frac,
                stint_iso=str(stint.get("start_iso") or target_iso),
            )
            projected = getattr(prospect, "_prospect_projected_stats", None) or projected
            proj_gp = max(1, _safe_int(projected.get("gp"), 60))
            last_iso = ""
    except Exception:
        pass

    if last_iso == target_iso and target_iso:
        cached = getattr(prospect, "_prospect_season_stats", None)
        actual = dict(cached) if isinstance(cached, dict) else _empty_actual_stat_line()
        # Refresh weekly stock even on a same-day cache hit so the board isn't stuck
        # at +0 after stock-logic changes or a quiet mid-week baseline reset.
        try:
            week_key = _iso_week_key(target_iso)
            _sync_prospect_week_baseline(prospect, actual, week_key)
            week_delta = _week_stat_delta(prospect, actual)
            weekly_stock = _compute_weekly_stock_fields(prospect, week_delta, projected, league, actual=actual)
            _apply_stock_to_actual(prospect, actual, weekly_stock)
            setattr(prospect, "_prospect_season_stats", dict(actual))
        except Exception:
            pass
        return actual

    if expected_gp_override is None:
        expected_gp = expected_games_for_date(league, proj_gp, target_iso, start_frac)
    else:
        expected_gp = int(max(0, min(proj_gp, int(expected_gp_override))))
    actual = dict(getattr(prospect, "_prospect_season_stats", None) or _empty_actual_stat_line())
    cur_gp = _safe_int(actual.get("gp"), 0)
    missed_total = _safe_int(actual.get("gp_missed"), 0)
    # Games lost to injury are gone for good — they must not be re-simulated later.
    delta_gp = max(0, expected_gp - cur_gp - missed_total)

    week_key = _iso_week_key(target_iso)
    prior_stock_week = str(getattr(prospect, "_prospect_last_stock_week_key", "") or "")

    if delta_gp <= 0:
        if prior_stock_week == week_key:
            try:
                setattr(prospect, "_prospect_last_stat_update_iso", target_iso)
                setattr(prospect, "_prospect_games_simulated_to_date", expected_gp)
            except Exception:
                pass
            cached = getattr(prospect, "_prospect_season_stats", None)
            return dict(cached) if isinstance(cached, dict) else _empty_actual_stat_line()

        _sync_prospect_week_baseline(prospect, actual, week_key)
        week_delta = _week_stat_delta(prospect, actual)
        weekly_stock = _compute_weekly_stock_fields(prospect, week_delta, projected, league, actual=actual)
        cached = dict(getattr(prospect, "_prospect_season_stats", None) or actual)
        _apply_stock_to_actual(prospect, cached, weekly_stock)
        try:
            setattr(prospect, "_prospect_season_stats", dict(cached))
            setattr(prospect, "_prospect_last_stat_update_iso", target_iso)
            setattr(prospect, "_prospect_games_simulated_to_date", expected_gp)
            setattr(prospect, "_prospect_last_stock_week_key", week_key)
        except Exception:
            pass
        return dict(cached)

    _sync_prospect_week_baseline(prospect, actual, week_key)

    injury_gp = _prospect_injury_games(prospect, delta_gp, rng)
    if injury_gp:
        actual["gp_missed"] = missed_total + injury_gp
    playable_gp = max(0, delta_gp - injury_gp)
    if _is_goalie(prospect):
        _simulate_goalie_games(prospect, league, playable_gp, rng, actual, projected)
    else:
        target_ppg = calculate_prospect_ppg_scale(prospect, league, season_year=season_year)
        try:
            setattr(prospect, "_prospect_expected_ppg", float(target_ppg))
        except Exception:
            pass
        _simulate_skater_games(prospect, league, playable_gp, rng, actual, target_ppg)

    actual["stat_source"] = "calendar_sim"
    week_delta = _week_stat_delta(prospect, actual)
    weekly_stock = _compute_weekly_stock_fields(prospect, week_delta, projected, league, actual=actual)
    season_stock = _compute_stock_fields(prospect, actual, projected, league)
    recent = list(actual.get("_recent_game_points") or [])
    recent_form = {
        "last_5_gp": min(5, len(recent)),
        "last_5_points": sum(recent[-5:]) if recent else 0,
        "week_gp": weekly_stock.get("week_gp", 0),
        "week_points": weekly_stock.get("week_points", 0),
    }

    ctx = attach_prospect_production_context(prospect, league, actual, rng)
    actual.update(ctx)
    actual.update(season_stock)
    # Display stock is weekly (non-cumulative); season signal stays internal for ranking.
    _apply_stock_to_actual(prospect, actual, weekly_stock)
    actual["recent_form"] = recent_form
    actual["season_stock_delta"] = season_stock.get("stock_delta", 0)

    try:
        setattr(prospect, "_prospect_season_stats", dict(actual))
        setattr(prospect, "_prospect_last_stat_update_iso", target_iso)
        setattr(prospect, "_prospect_games_simulated_to_date", expected_gp)
        setattr(prospect, "_prospect_last_stock_week_key", week_key)
    except Exception:
        pass

    return dict(actual)


def _live_minor_levels(league: Any) -> Dict[int, Tuple[Any, str]]:
    """Players currently on an AHL / ECHL roster list, keyed by object id."""
    live: Dict[int, Tuple[Any, str]] = {}
    for tm in getattr(league, "teams", None) or []:
        for p in getattr(tm, "ahl_roster", None) or []:
            if not getattr(p, "retired", False):
                live[id(p)] = (p, "AHL")
        for p in getattr(tm, "echl_roster", None) or []:
            if not getattr(p, "retired", False):
                live[id(p)] = (p, "ECHL")
    return live


def _still_at_junior(p: Any) -> bool:
    return str(getattr(p, "roster_location", "") or "").lower() not in ("nhl", "ahl", "echl")


def advance_all_development_league_stats(
    sim: Any,
    calendar_iso: Any,
    *,
    season_year: Optional[int] = None,
    rng: Optional[random.Random] = None,
    prospect_rows: Optional[List[Tuple[Any, str]]] = None,
) -> int:
    """Advance junior / NCAA / Europe / AHL / ECHL stat lines to calendar_iso.

    AHL and ECHL membership is read from the live roster lists every run (a cached row list
    goes stale the moment someone is called up). A player who leaves a level closes his stint;
    one who returns — or who missed a sync while elsewhere — opens a new stint at the current
    date instead of being back-filled with games he was not there for.
    """
    league = getattr(sim, "league", None)
    if league is None:
        return 0
    if not isinstance(rng, random.Random):
        rng = getattr(sim, "rng", None)
    if not isinstance(rng, random.Random):
        rng = random.Random()

    live = _live_minor_levels(league)

    rows: List[Tuple[Any, str]] = []
    if isinstance(prospect_rows, list):
        for p, code in prospect_rows:
            if code in ("AHL", "ECHL"):
                if id(p) not in live:
                    end_prospect_stint(p, reason="left_level")
                continue
            if getattr(p, "retired", False):
                continue
            if not _still_at_junior(p):
                end_prospect_stint(p, reason="left_level")
                continue
            rows.append((p, code))
    else:
        for block in getattr(league, "development_leagues", None) or []:
            code = str(block.get("league_code") or "")
            max_age = development_league_stat_max_age(code)
            for tm in block.get("teams") or []:
                for p in tm.get("players") or []:
                    if getattr(p, "retired", False) or not _still_at_junior(p):
                        continue
                    ident = getattr(p, "identity", None)
                    age = int(getattr(ident, "age", 99) or 99) if ident else 99
                    if age <= max_age:
                        rows.append((p, code))
    rows.extend(live.values())

    iso = str(calendar_iso or "")[:10]
    sy = int(season_year) if season_year is not None else 0
    state = getattr(league, "_prospect_stat_sync_state", None)
    if not isinstance(state, dict):
        state = {}
    prev_iso = str(state.get("iso") or "") if int(state.get("season") or 0) == sy else ""
    first_run = not prev_iso

    month, day = _parse_calendar_iso(iso)
    frac_by_code: Dict[str, float] = {}
    updated = 0
    for p, code in rows:
        try:
            ckey = str(code or "")
            st = _stint(p)
            if (
                st is not None
                and st.get("open", True)
                and prev_iso
                and str(st.get("last_live_iso") or "") < prev_iso
                and _safe_int(st.get("season_year"), 0) == sy
            ):
                # Missed a sync while somewhere else (e.g. an NHL call-up): that stretch is not his.
                end_prospect_stint(p, reason="absent")

            projected = getattr(p, "_prospect_projected_stats", None)
            if not isinstance(projected, dict) or not projected.get("gp"):
                if ckey not in frac_by_code:
                    frac_by_code[ckey] = _league_season_fraction(ckey, month, day)
                initialize_prospect_season(
                    p,
                    code,
                    rng=rng,
                    season_year=season_year,
                    calendar_iso=None,
                    # First sync of a season: everyone has been here since day one.
                    stint_start_frac=0.0 if first_run else frac_by_code[ckey],
                    stint_iso=iso,
                )

            advance_prospect_stats_to_date(p, code, iso, rng=rng, season_year=season_year)
            st = _stint(p)
            if st is not None:
                st["last_live_iso"] = iso
            updated += 1
        except Exception:
            pass
    try:
        setattr(league, "_prospect_stat_sync_state", {"iso": iso, "season": sy})
    except Exception:
        pass
    return updated


def _prospect_event_skills(prospect: Any) -> Dict[str, float]:
    """Latent event-skill dimensions shared with NHL event simulation philosophy."""
    ratings = getattr(prospect, "ratings", None) or {}
    style = _playstyle_bucket(prospect)
    # The all-ratings fallback below is identical for every skill (same ratings, same
    # default), so compute it at most once per call.
    fallback_vals: Dict[float, List[float]] = {}

    def _avg(keys: List[str], default: float = 68.0) -> float:
        vals = [_safe_float(ratings.get(k), default) for k in keys if k in ratings]
        if not vals:
            # Alias map for common prospect rating key variants.
            aliases = {
                "shooting": ("shooting_accuracy", "shot_accuracy", "wrist_shot"),
                "passing": ("passing_accuracy", "pass_accuracy"),
                "puck_control": ("puck_handling", "puck_skills", "hands"),
                "wrist_shot_accuracy": ("shooting_accuracy", "shot_accuracy"),
                "slap_shot_accuracy": ("shooting_accuracy", "shot_accuracy"),
                "offensive_awareness": ("hockey_iq", "vision", "awareness"),
                "composure": ("poise", "clutch"),
                "speed": ("skating", "acceleration"),
            }
            for k in keys:
                for alt in aliases.get(k, ()):
                    if alt in ratings:
                        vals.append(_safe_float(ratings.get(alt), default))
                        break
        if not vals:
            if default not in fallback_vals:
                fallback_vals[default] = [
                    _safe_float(v, default) for v in ratings.values() if isinstance(v, (int, float))
                ]
            vals = fallback_vals[default]
        return (sum(vals) / max(1, len(vals))) / 99.0 if vals else 0.55

    volume = _avg(["shooting", "shot_power", "wrist_shot_accuracy", "offensive_awareness"])
    finishing = _avg(["wrist_shot_accuracy", "slap_shot_accuracy", "shot_power", "composure"])
    playmaking = _avg(["passing", "puck_control", "offensive_awareness"])
    process = _avg(["puck_control", "speed", "offensive_awareness", "defensive_awareness"])
    defense = _avg(["defensive_awareness", "shot_blocking", "stick_checking", "strength"])

    # Blend overall so draft-board talent (not only sparse rating dicts) drives PPG bands.
    # Player.ovr is a *method*; float(method) used to fail silently and pin this to 0.50.
    ovr_n = _clamp(_player_ovr_0_1(prospect), 0.20, 0.95)
    volume = _clamp(volume * 0.88 + ovr_n * 0.12, 0.15, 0.98)
    finishing = _clamp(finishing * 0.86 + ovr_n * 0.14, 0.15, 0.98)
    playmaking = _clamp(playmaking * 0.88 + ovr_n * 0.12, 0.15, 0.98)
    process = _clamp(process * 0.90 + ovr_n * 0.10, 0.15, 0.98)

    if style in ("sniper", "scoring_forward"):
        volume *= 1.10
        finishing *= 1.12
        playmaking *= 0.92
    elif style == "playmaker":
        playmaking *= 1.14
        volume *= 0.92
    elif style in ("defensive_defenseman", "shutdown"):
        defense *= 1.12
        volume *= 0.82
    elif style == "offensive_defenseman":
        playmaking *= 1.08
        volume *= 0.95

    return {
        "volume": _clamp(volume, 0.15, 0.98),
        "finishing": _clamp(finishing, 0.15, 0.98),
        "playmaking": _clamp(playmaking, 0.15, 0.98),
        "process": _clamp(process, 0.15, 0.98),
        "defense": _clamp(defense, 0.15, 0.98),
    }


def _ppg_curve(profile: Dict[str, Any], z: float) -> float:
    """League PPG for a forward ``z`` talent-sigmas from the league's typical roster.

    Continuous through the league's average -> star -> elite bands. (The old model bucketed
    players into three tiers by absolute composite, and no generated player ever reached the
    star or elite tiers, so ~90% of a league drew from one uniform band regardless of talent.)
    """
    avg_lo, avg_hi = profile.get("average_ppg_target", (0.45, 0.85))
    star_lo, star_hi = profile.get("star_ppg_target", (1.0, 1.45))
    elite_lo, elite_hi = profile.get("elite_ppg_target", (1.35, 1.85))
    pts = (
        (-3.0, avg_lo * 0.50),
        (-1.5, avg_lo * 0.72),
        (0.0, avg_lo + 0.35 * (avg_hi - avg_lo)),
        (1.5, avg_hi),
        (3.0, 0.5 * (avg_hi + star_lo)),
        (4.5, star_lo),
        (7.0, star_hi),
        (9.0, elite_lo),
        (11.0, elite_hi),
    )
    if z <= pts[0][0]:
        return float(pts[0][1])
    if z >= pts[-1][0]:
        return float(pts[-1][1])
    for (z0, v0), (z1, v1) in zip(pts, pts[1:]):
        if z0 <= z <= z1:
            return float(v0 + (v1 - v0) * (z - z0) / (z1 - z0))
    return float(pts[-1][1])


def calculate_prospect_ppg_scale(
    prospect: Any,
    league: Any,
    rng: Optional[random.Random] = None,
    *,
    season_year: Optional[int] = None,
) -> float:
    """
    Expected PPG after league, talent, usage, age, archetype and risk.

    Deterministic per (player, season, league): every random draw comes from a stable seeded
    stream, so calling this again — daily, weekly or once — returns the same target unless the
    player's ratings actually changed. (It used to redraw on every call and keep any draw that
    was 4%+ higher, so the target drifted up with the number of calendar updates.)
    ``rng`` is accepted for backwards compatibility and ignored.
    """
    if _is_goalie(prospect):
        return 0.0

    profile = get_league_scoring_profile(league)
    key = str(profile.get("profile_key") or "JUNIOR")
    sy = _seed_year(prospect, season_year)
    srng = _stable_rng(prospect, "ppg", sy, key)

    skills = _prospect_event_skills(prospect)
    offensive = (
        skills["volume"] * 0.18
        + skills["finishing"] * 0.24
        + skills["playmaking"] * 0.32
        + skills["process"] * 0.26
    )
    age = _player_age(prospect)
    style = _playstyle_bucket(prospect)
    defense = _is_defense(prospect)

    z = (offensive - _LEAGUE_TALENT_CENTER.get(key, 0.43)) / _TALENT_SIGMA
    base = _ppg_curve(profile, z)

    # Deployment: first-line minutes vs. fourth-line minutes matter as much as talent.
    # Pro leagues talent-compress (pool floors), so deployment is what separates a top-line AHL
    # scorer from a fourth-liner; give it a wider spread there.
    usage_sigma = 0.31 if key in ("AHL", "ECHL") else 0.26
    usage = math.exp(srng.gauss(0.0, usage_sigma) + 0.06 * _clamp(z, -2.0, 3.0) - 0.5 * usage_sigma ** 2 * 0.5)
    base *= _clamp(usage, 0.55, 1.75)

    if defense:
        base *= _safe_float(profile.get("defensive_translation_penalty"), 0.90)
        if style == "offensive_defenseman":
            base *= srng.uniform(0.90, 1.05)
        elif style == "defensive_defenseman":
            base *= srng.uniform(0.32, 0.45)
        else:
            base *= srng.uniform(0.58, 0.75)
    elif style == "grinder":
        base *= srng.uniform(0.52, 0.72)
    elif style in ("sniper", "playmaker", "power_forward", "scoring_forward"):
        style_mult = 0.92 + offensive * 0.22
        if style == "playmaker":
            style_mult *= 1.04
        elif style == "sniper":
            style_mult *= 0.98
        base *= style_mult

    if age >= 20:
        base *= _safe_float(profile.get("overager_bonus"), 1.06) * srng.uniform(1.00, 1.05)
    elif age == 19:
        base *= srng.uniform(0.96, 1.04)
    elif age <= 17:
        base *= srng.uniform(0.92, 0.98)

    if _is_boom_bust(prospect):
        base *= srng.uniform(0.88, 1.12)

    if _has_character_concerns(prospect):
        base *= srng.uniform(0.86, 1.04)

    base *= _prospect_role_multiplier(prospect, league)

    avg_lo = _safe_float((profile.get("average_ppg_target") or (0.45, 0.85))[0], 0.45)
    elite_hi = _safe_float((profile.get("elite_ppg_target") or (1.35, 1.85))[1], 1.85)
    star_lo = _safe_float((profile.get("star_ppg_target") or (1.0, 1.45))[0], 1.0)
    avg_hi = _safe_float((profile.get("average_ppg_target") or (0.45, 0.85))[1], 0.85)
    floor = max(0.06, avg_lo * 0.28) if defense else max(0.10, avg_lo * 0.50)
    ceiling = elite_hi * 1.02
    if key in ("QMJHL", "OHL", "CHL", "WHL"):
        if offensive >= 0.80:
            ceiling = max(ceiling, 2.40)
        elif offensive >= 0.70:
            ceiling = max(ceiling, 2.10)
        else:
            ceiling = max(ceiling, 1.85)
    if defense:
        if style == "offensive_defenseman":
            ceiling = min(ceiling, elite_hi * 0.68)
        elif style == "defensive_defenseman":
            ceiling = min(ceiling, avg_hi * 0.55)
        else:
            ceiling = min(ceiling, star_lo * 0.78)
    return _clamp(base, floor, ceiling)


def _draw_goal_share(style: str, rng: random.Random) -> float:
    if style == "sniper":
        share = rng.uniform(0.48, 0.62)
    elif style == "playmaker":
        share = rng.uniform(0.22, 0.38)
    elif style == "power_forward":
        share = rng.uniform(0.42, 0.55)
    elif style == "grinder":
        share = rng.uniform(0.35, 0.50)
    elif style == "offensive_defenseman":
        share = rng.uniform(0.18, 0.32)
    elif style == "defensive_defenseman":
        share = rng.uniform(0.28, 0.42)
    elif style == "two_way_defenseman":
        share = rng.uniform(0.24, 0.38)
    else:
        share = rng.uniform(0.34, 0.48)
    if _is_boom_bust_style(style, rng):
        share = rng.uniform(0.15, 0.70)
    return share


def _season_goal_share(prospect: Any, style: str) -> float:
    """A player's share of his points that are goals — fixed for the season, never re-rolled."""
    return _draw_goal_share(
        style, _stable_rng(prospect, "goal_share", _seed_year(prospect))
    )


def _split_goals_assists(points: int, style: str, rng: random.Random) -> Tuple[int, int]:
    """One-shot split of a *projected* total. Live stat lines accumulate via _add_points."""
    if points <= 0:
        return 0, 0
    goals = int(round(points * _draw_goal_share(style, rng)))
    goals = max(0, min(points, goals))
    return goals, max(0, points - goals)


def _add_points(stats: Dict[str, Any], new_points: int, share: float, rng: random.Random) -> None:
    """Add newly scored points to the running line, splitting only the *new* points."""
    if new_points <= 0:
        return
    g_new = _binomial(new_points, share, rng)
    stats["goals"] = _safe_int(stats.get("goals"), 0) + g_new
    stats["assists"] = _safe_int(stats.get("assists"), 0) + (new_points - g_new)
    stats["points"] = _safe_int(stats.get("points"), 0) + new_points


def _is_boom_bust_style(style: str, rng: random.Random) -> bool:
    return style in ("scoring_forward", "sniper", "playmaker") and rng.random() < 0.14


def generate_goalie_prospect_line(prospect: Any, league: Any, games_played: int, rng: random.Random) -> Dict[str, Any]:
    """Projected goalie season. ``games_played`` is the *team's* schedule; the goalie's own
    GP is his share of it (starters play more than backups) — not a full season each.

    Built from goalie ratings and the league's goalie environment only; the old version keyed
    off the skater-oriented offensive talent score, which included hidden potential.
    """
    profile = get_league_scoring_profile(league)
    key = str(profile.get("profile_key") or "JUNIOR")
    srng = _stable_rng(prospect, "goalie", _seed_year(prospect), key)
    ability = _goalie_ability(prospect)
    z = (ability - _LEAGUE_TALENT_CENTER.get(key, 0.43)) / 0.06
    sv_base, sa_base = _GOALIE_ENV.get(key, (0.895, 29.0))
    save_pct = _clamp(sv_base + 0.010 * _clamp(z, -3.0, 5.0) + srng.gauss(0.0, 0.006), 0.850, 0.945)
    sa_pg = sa_base * srng.uniform(0.92, 1.08)
    share = _clamp(srng.gauss(0.42 + 0.05 * _clamp(z, -2.0, 3.0), 0.20), 0.10, 0.88)
    gp = max(8, int(round(int(games_played) * share)))
    win_rate = _clamp(0.47 + 0.02 * _clamp(z, -3.0, 4.0) + srng.gauss(0.0, 0.05), 0.30, 0.68)
    wins = int(round(gp * win_rate))
    non_wins = gp - wins
    otl = int(round(non_wins * 0.12))
    gaa = sa_pg * (1.0 - save_pct)
    return {
        "gp": gp,
        "games_played": gp,
        "goals": 0,
        "assists": 0,
        "points": 0,
        "wins": wins,
        "losses": max(0, non_wins - otl),
        "ot_losses": otl,
        "save_pct": round(save_pct, 3),
        "savePct": round(save_pct, 3),
        "gaa": round(gaa, 2),
        "shutouts": int(round(gp * math.exp(-gaa))),
        "shots_against_per_game": round(sa_pg, 1),
        "ppg": 0.0,
        "points_per_game": 0.0,
    }


def calculate_league_adjusted_ppg(prospect: Any, league: Any, base_score: float) -> float:
    """Scale a base offensive score into league-adjusted PPG."""
    profile = get_league_scoring_profile(league)
    mult = _safe_float(profile.get("scoring_multiplier"), 1.0)
    diff = _safe_float(profile.get("difficulty"), 0.7)
    adjusted = _safe_float(base_score, 0.5) * mult * (1.08 - diff * 0.12)
    elite_lo, elite_hi = profile.get("elite_ppg_target", (1.35, 1.85))
    return _clamp(adjusted, elite_lo * 0.25, elite_hi * 1.08)


def generate_prospect_scoring_line(
    prospect: Any,
    league: Any,
    games_played: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Generate GP/G/A/PTS/PPG for a prospect in a given league environment."""
    if not isinstance(rng, random.Random):
        seed = getattr(prospect, "rng_seed", None) or getattr(prospect, "seed", None)
        rng = random.Random(int(seed) if seed is not None else None)

    profile = get_league_scoring_profile(league)
    gp = int(games_played) if games_played is not None else _default_games_played(profile, _player_age(prospect), rng)

    if _is_goalie(prospect):
        return generate_goalie_prospect_line(prospect, league, gp, rng)

    style = _playstyle_bucket(prospect)
    ppg = calculate_prospect_ppg_scale(prospect, league, rng)
    if _is_defense(prospect):
        if style == "offensive_defenseman":
            max_ppg = min(1.62, profile["elite_ppg_target"][1] * 0.78)
        elif style == "defensive_defenseman":
            max_ppg = 0.62
        else:
            max_ppg = 1.15
    elif profile["profile_key"] in ("QMJHL", "OHL", "CHL", "WHL"):
        max_ppg = 2.68
    else:
        max_ppg = profile["elite_ppg_target"][1] * 1.05
    points = int(round(gp * ppg))
    points = max(0, min(int(gp * max_ppg), points))
    ppg = points / max(1, gp)
    goals, assists = _split_goals_assists(points, style, rng)

    return {
        "gp": gp,
        "games_played": gp,
        "goals": goals,
        "assists": assists,
        "points": points,
        "ppg": round(ppg, 3),
        "points_per_game": round(ppg, 3),
        "goals_per_game": round(goals / max(1, gp), 3),
        "assists_per_game": round(assists / max(1, gp), 3),
        "pim": rng.randint(8, 78) if style in ("grinder", "power_forward") else rng.randint(2, 42),
    }


def _translation_risk(prospect: Any, profile: Dict[str, Any], ppg: float, age: int) -> str:
    risk = 0.0
    if age >= 20:
        risk += 0.28
    elif age == 19:
        risk += 0.10
    if _has_character_concerns(prospect):
        risk += 0.18
    if _is_boom_bust(prospect):
        risk += 0.14
    if profile["profile_key"] in ("CHL", "OHL", "WHL", "QMJHL"):
        risk += 0.12
    if profile["profile_key"] in ("SHL", "LIIGA"):
        risk -= 0.12
    if ppg >= 1.55 and age >= 19:
        risk += 0.10
    if risk >= 0.42:
        return "High"
    if risk >= 0.22:
        return "Medium"
    return "Low"


def _production_context_label(prospect: Any, profile: Dict[str, Any], ppg: float, age: int) -> str:
    if age >= 20 and ppg >= 1.25:
        return "Overager scoring"
    if _has_character_concerns(prospect) and ppg >= 1.15:
        return "Risky scorer"
    if _is_boom_bust(prospect):
        return "Boom/Bust"
    if ppg >= profile.get("elite_ppg_target", (1.5, 2.0))[0]:
        return "Elite junior production"
    if profile["profile_key"] in ("CHL", "OHL", "WHL", "QMJHL"):
        return "Junior inflated"
    if profile["difficulty"] >= 0.85:
        return "Hard pro league"
    if profile["profile_key"] == "NCAA":
        return "Low-scoring league"
    return str(profile.get("environment_label") or "Junior inflated")


def attach_prospect_production_context(
    prospect: Any,
    league: Any,
    stat_line: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Attach scouting context labels — high scoring ≠ automatic NHL translation."""
    profile = get_league_scoring_profile(league)
    ppg = _safe_float(stat_line.get("ppg", stat_line.get("points_per_game")), 0.0)
    age = _player_age(prospect)
    translation = _translation_risk(prospect, profile, ppg, age)
    context = _production_context_label(prospect, profile, ppg, age)
    adj = ppg * (1.0 - profile["difficulty"] * 0.32)
    if age >= 20:
        adj *= 0.82
    if _has_character_concerns(prospect):
        adj *= 0.88

    out = {
        "production_context": context,
        "translation_risk": translation,
        "scoring_environment": str(profile.get("environment_label") or ""),
        "league_difficulty": str(profile.get("difficulty_label") or ""),
        "league_scoring_profile": profile.get("profile_key"),
        "production_adjusted_score": round(_clamp(adj, 0.0, 2.35), 3),
    }
    try:
        setattr(prospect, "production_context", context)
        setattr(prospect, "translation_risk", translation)
        setattr(prospect, "scoring_environment", out["scoring_environment"])
        setattr(prospect, "league_difficulty", out["league_difficulty"])
        setattr(prospect, "production_adjusted_score", out["production_adjusted_score"])
        setattr(prospect, "_prospect_season_stats", dict(stat_line))
    except Exception:
        pass
    return out


def ensure_prospect_season_stats(
    prospect: Any,
    league: Any,
    rng: Optional[random.Random] = None,
    *,
    force: bool = False,
    calendar_iso: Optional[str] = None,
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Return current actual season stats (calendar-simulated), never full-season totals."""
    if force or not isinstance(getattr(prospect, "_prospect_projected_stats", None), dict):
        initialize_prospect_season(
            prospect,
            league,
            rng=rng,
            season_year=season_year,
            calendar_iso=calendar_iso,
            force=force,
        )
    elif calendar_iso:
        advance_prospect_stats_to_date(
            prospect,
            league,
            calendar_iso,
            rng=rng,
            season_year=season_year,
        )

    cached = getattr(prospect, "_prospect_season_stats", None)
    if isinstance(cached, dict):
        return dict(cached)

    if not isinstance(rng, random.Random):
        seed = getattr(prospect, "rng_seed", None) or getattr(prospect, "seed", None) or id(prospect)
        rng = random.Random(int(seed) & 0xFFFFFFFF)
    return initialize_prospect_season(prospect, league, rng=rng, season_year=season_year, calendar_iso=calendar_iso)


def _defensive_talent_score(prospect: Any) -> float:
    """0–1 defensive talent estimate from current ratings and playstyle (no potential input)."""
    ovr = _player_ovr_0_1(prospect)
    score = ovr * 0.36

    ratings = getattr(prospect, "ratings", None)
    if isinstance(ratings, dict) and ratings:
        def_vals: List[float] = []
        iq_vals: List[float] = []
        phys_vals: List[float] = []
        skate_vals: List[float] = []
        for k, v in ratings.items():
            kl = str(k).lower()
            if any(x in kl for x in ("def", "stick", "gap", "block", "poke", "positioning")):
                def_vals.append(_safe_float(v, 50.0) / 99.0)
            if any(x in kl for x in ("iq", "sense", "awareness", "read")):
                iq_vals.append(_safe_float(v, 50.0) / 99.0)
            if any(x in kl for x in ("strength", "physical", "reach", "body")):
                phys_vals.append(_safe_float(v, 50.0) / 99.0)
            if "skat" in kl:
                skate_vals.append(_safe_float(v, 50.0) / 99.0)
        if def_vals:
            score += sum(def_vals) / len(def_vals) * 0.30
        if iq_vals:
            score += sum(iq_vals) / len(iq_vals) * 0.10
        if phys_vals:
            score += sum(phys_vals) / len(phys_vals) * 0.08
        if skate_vals:
            score += sum(skate_vals) / len(skate_vals) * 0.06

    chem = getattr(prospect, "chemistry_profile", None)
    if isinstance(chem, dict):
        buy_in = _safe_float(chem.get("defensive_buy_in"), 0.5)
        score += buy_in * 0.14

    style = _playstyle_bucket(prospect)
    if style in ("defensive_defenseman", "two_way_defenseman", "two_way", "grinder"):
        score += 0.10
    elif style in ("sniper", "scoring_forward", "offensive_defenseman", "playmaker"):
        score -= 0.04
    if _is_defense(prospect):
        score += 0.05

    return _clamp(score, 0.10, 0.98)


def _primary_assist_share(style: str, is_d: bool) -> float:
    if style == "playmaker":
        return 0.76
    if style == "sniper":
        return 0.44
    if style == "power_forward":
        return 0.52
    if style == "grinder":
        return 0.48
    if style == "offensive_defenseman":
        return 0.58
    if style == "defensive_defenseman":
        return 0.40
    if style == "two_way_defenseman":
        return 0.50
    if style == "two_way":
        return 0.54
    if is_d:
        return 0.46
    return 0.56


def _estimate_prospect_shots(
    gp: int,
    goals: int,
    assists: int,
    style: str,
    off_talent: float,
    is_d: bool,
) -> int:
    if gp <= 0:
        return 0
    gpg = goals / float(gp)
    apg = assists / float(gp)
    rate = 1.55 + off_talent * 2.35 + gpg * 1.25 + apg * 0.35
    if style == "sniper":
        rate += 1.45
    elif style == "playmaker":
        rate += 0.35
    elif style == "power_forward":
        rate += 0.75
    elif style == "offensive_defenseman":
        rate += 0.55
    elif style == "defensive_defenseman":
        rate -= 0.45
    elif style == "grinder":
        rate -= 0.20
    elif style == "two_way":
        rate += 0.15
    if is_d:
        rate *= 0.84
    shots = int(round(rate * gp))
    return max(goals, max(1, shots))


def _estimate_prospect_plus_minus(
    gp: int,
    goals: int,
    assists: int,
    off_talent: float,
    def_talent: float,
    style: str,
    ppg: float,
) -> int:
    if gp <= 0:
        return 0
    prod_pg = (goals * 0.55 + assists * 0.35) / float(gp)
    base = (prod_pg - 0.48) * gp * 0.42
    base += (def_talent - 0.50) * gp * 0.62
    base += (off_talent - 0.50) * gp * 0.18
    if style in ("defensive_defenseman", "two_way_defenseman", "two_way", "grinder"):
        base += gp * 0.14
    elif style in ("sniper", "scoring_forward") and def_talent < 0.46:
        base -= gp * 0.10
    if ppg >= 1.35 and def_talent >= 0.52:
        base += gp * 0.08
    return int(round(_clamp(base, -gp * 0.95, gp * 0.88)))


def derive_prospect_analytics(
    prospect: Any,
    league: Any,
    stat_line: Mapping[str, Any],
    *,
    draft_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Deterministic prospect analytics derived from season stats, ratings, role,
    league context, and draft profile. Stored on API payloads as `analytics`.
    """
    profile = get_league_scoring_profile(league)
    gp = max(0, _safe_int(stat_line.get("gp", stat_line.get("games_played")), 0))
    goals = max(0, _safe_int(stat_line.get("goals"), 0))
    assists = max(0, _safe_int(stat_line.get("assists"), 0))
    points = goals + assists if stat_line.get("points") is None else max(0, _safe_int(stat_line.get("points"), 0))
    ppg = _safe_float(stat_line.get("ppg", stat_line.get("points_per_game")), 0.0)
    if gp > 0 and ppg <= 0:
        ppg = points / float(gp)

    if _is_goalie(prospect):
        sv = _safe_float(stat_line.get("save_pct"), 0.905)
        gaa = _safe_float(stat_line.get("gaa"), 2.85)
        wins = _safe_int(stat_line.get("wins"), 0)
        if gp <= 0:
            return {
                "gsax": None,
                "quality_starts": None,
            }
        shots_against = _safe_float(stat_line.get("shots_against"), 0.0) or gp * 29.0
        gsax = round((sv - 0.905) * shots_against, 2)
        qs_rate = _clamp(0.42 + (sv - 0.885) * 3.2 - max(0.0, gaa - 2.75) * 0.12, 0.18, 0.82)
        qs = int(round(gp * qs_rate))
        qs = max(wins // 2, min(gp, qs))
        return {
            "gsax": gsax,
            "quality_starts": qs,
        }

    style = _playstyle_bucket(prospect)
    is_d = _is_defense(prospect)
    off_talent = _offensive_talent_score(prospect)
    def_talent = _defensive_talent_score(prospect)
    league_diff = _safe_float(profile.get("difficulty"), 0.65)
    prod_adj = _safe_float(stat_line.get("production_adjusted_score"), ppg * (1.0 - league_diff * 0.28))
    # Public signal only: the published draft rank, else current ability. (Was hidden potential.)
    public_ability = _clamp(_player_ovr_0_1(prospect) * 1.08, 0.2, 0.99)
    rank_signal = 1.0 - _clamp((float(draft_rank or 120) - 1.0) / 119.0, 0.0, 1.0) if draft_rank else public_ability

    shots_raw = stat_line.get("shots", stat_line.get("sog", stat_line.get("shots_on_goal")))
    shots = _safe_int(shots_raw, 0) if shots_raw not in (None, "") else 0
    if shots <= 0 and gp > 0:
        shots = _estimate_prospect_shots(gp, goals, assists, style, off_talent, is_d)

    plus_raw = stat_line.get("plus_minus", stat_line.get("plusMinus"))
    if plus_raw not in (None, ""):
        plus_minus = _safe_int(plus_raw, 0)
    elif gp > 0:
        plus_minus = _estimate_prospect_plus_minus(gp, goals, assists, off_talent, def_talent, style, ppg)
    else:
        plus_minus = None

    shooting_pct: Optional[float] = None
    if shots > 0 and gp > 0:
        shooting_pct = round((goals / float(shots)) * 100.0, 1)
        if is_d:
            shooting_pct = round(_clamp(shooting_pct, 3.5, 14.5), 1)
        elif style == "sniper":
            shooting_pct = round(_clamp(shooting_pct, 8.0, 19.5), 1)
        else:
            shooting_pct = round(_clamp(shooting_pct, 5.5, 17.0), 1)

    shot_rate: Optional[float] = None
    if gp > 0 and shots > 0:
        shot_rate = round(shots / float(gp), 2)

    prim_raw = stat_line.get("primary_points", stat_line.get("primary_pts"))
    if prim_raw not in (None, ""):
        primary_points = max(0, _safe_int(prim_raw, 0))
    elif gp > 0 and points > 0:
        share = _primary_assist_share(style, is_d)
        primary_points = int(round(goals + assists * share))
        primary_points = max(goals, min(points, primary_points))
    else:
        primary_points = None

    gp_factor = _clamp(gp / 42.0, 0.0, 1.0) if gp > 0 else 0.0
    offense_drive = (ppg - 0.42) * 5.2 + off_talent * 4.8 + (shot_rate or 0.0) * 0.55
    defense_drive = (def_talent - 0.48) * 9.5 + (plus_minus or 0) / max(gp, 1) * 2.8
    scorer_def_gap = max(0.0, off_talent - def_talent - 0.12)
    # Per-prospect jitter so similar late picks don't print identical possession cookies.
    try:
        pid = str(getattr(getattr(prospect, "identity", None), "name", None) or getattr(prospect, "id", None) or id(prospect))
        jitter = ((sum(ord(c) for c in pid) % 97) / 97.0 - 0.5) * 8.5
    except Exception:
        jitter = 0.0
    poss_base = 50.0 + offense_drive * 0.95 + defense_drive * 0.72 - scorer_def_gap * 7.5 + jitter
    poss_base += (prod_adj - 0.55) * 2.4
    if style in ("two_way", "two_way_defenseman", "grinder"):
        poss_base += 1.6
    # League context: harder leagues pull possession toward 50 (tougher to dominate).
    poss_base += (0.55 - league_diff) * 3.5
    xgf_pct = round(_clamp(poss_base, 38.5, 61.8), 1)
    cf_pct = round(_clamp(xgf_pct + (def_talent - 0.50) * 3.8 + (off_talent - 0.50) * 1.4 + jitter * 0.35, 37.5, 62.5), 1)
    ff_pct = round(_clamp((xgf_pct * 0.55 + cf_pct * 0.45) + (off_talent - def_talent) * 1.8, 37.8, 62.0), 1)

    defensive_impact = round(
        _clamp(
            (def_talent - 0.42) * 5.8
            + (plus_minus or 0) / max(gp, 1) * 0.42 * gp_factor
            + (cf_pct - 50.0) * 0.06
            + (1.2 if style in ("defensive_defenseman", "two_way_defenseman", "two_way", "grinder") else 0.0),
            -2.5,
            6.5,
        ),
        1,
    )

    league_q = league_diff * 10.0
    usage = 0.55 + rank_signal * 0.28 + off_talent * 0.12 + def_talent * 0.08
    if style in ("two_way", "two_way_defenseman", "grinder", "defensive_defenseman"):
        usage += 0.06
    quality_of_competition = round(_clamp(league_q * 0.42 + usage * 4.6 + ppg * 0.85, 2.8, 9.4), 1)

    scoring_mult = _safe_float(profile.get("scoring_multiplier"), 1.0)
    team_env = _clamp(5.2 + scoring_mult * 1.35 - league_diff * 3.2 + off_talent * 1.4, 2.5, 8.8)
    quality_of_teammates = round(team_env, 1)

    off_war = (
        (ppg - 0.40) * 1.45 * gp_factor
        + ((shot_rate or 0.0) - 2.05) * 0.16 * gp_factor
        + ((shooting_pct or 9.5) / 100.0 - 0.095) * 2.2 * gp_factor
        + (off_talent - 0.48) * 1.15
        + goals / max(gp, 1) * 0.22 * gp_factor
    )
    if style in ("grinder", "defensive_defenseman") and ppg < 0.55:
        off_war *= 0.72
    # Overproduction vs current tools — late-round gem signal (no true-potential leak).
    expected_ppg = (0.18 + off_talent * 0.85) if is_d else (0.22 + off_talent * 1.15)
    surplus = ppg - expected_ppg
    if surplus > 0:
        off_war += surplus * 1.85 * max(0.35, gp_factor)
    off_war = round(_clamp(off_war, -1.2, 1.35), 2)

    def_war = (
        (def_talent - 0.46) * 1.25
        + defensive_impact * 0.14
        + (plus_minus or 0) / max(gp, 1) * 0.28 * gp_factor
        + (cf_pct - 50.0) * 0.018
        + (0.18 if is_d and style in ("defensive_defenseman", "two_way_defenseman") else 0.0)
    )
    if style in ("two_way",) and def_talent >= 0.58:
        def_war += 0.12
    def_war = round(_clamp(def_war, -0.8, 0.95), 2)

    # Ability-weighted WAR: junior scale — top draft-age producers land ~1.6–2.2,
    # not NHL starter territory. Diminishing returns keep elites separated.
    ability_war = ((off_talent + def_talent) / 2.0 - 0.50) * 0.95 * max(0.4, gp_factor)
    raw_war = off_war + def_war + ability_war
    # Soft cap with headroom so only a handful of true outliers touch the ceiling.
    if raw_war > 1.85:
        raw_war = 1.85 + (raw_war - 1.85) * 0.42
    war = round(_clamp(raw_war, -1.5, 2.35), 2)

    # Publish earlier so WAR is usable for gem hunting before the 15 GP wall.
    sample_ready = gp >= 5

    out: Dict[str, Any] = {
        "xgf_pct": xgf_pct if sample_ready else None,
        "cf_pct": cf_pct if sample_ready else None,
        "ff_pct": ff_pct if sample_ready else None,
        "war": war if sample_ready else None,
        "offensive_war": off_war if sample_ready else None,
        "defensive_war": def_war if sample_ready else None,
        "shooting_pct": shooting_pct,
        "plus_minus": plus_minus,
        "primary_points": primary_points,
        "shot_rate": shot_rate,
        "defensive_impact": defensive_impact if sample_ready else None,
        "quality_of_competition": quality_of_competition if sample_ready else None,
        "quality_of_teammates": quality_of_teammates if sample_ready else None,
        "shots": shots if gp > 0 else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def prospect_stats_for_api(
    prospect: Any,
    league: Any,
    rng: Optional[random.Random] = None,
    *,
    calendar_iso: Optional[str] = None,
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Flatten actual + projected stats for franchise API payloads."""
    actual = ensure_prospect_season_stats(
        prospect,
        league,
        rng=rng,
        calendar_iso=calendar_iso,
        season_year=season_year,
    )
    projected = dict(getattr(prospect, "_prospect_projected_stats", None) or {})

    gp = _safe_int(actual.get("gp"), 0)
    pts = _safe_int(actual.get("points"), 0)
    actual_ppg = round(pts / gp, 3) if gp > 0 else None

    proj_gp = _safe_int(projected.get("gp"), 0)
    proj_pts = _safe_int(projected.get("points"), 0)
    proj_ppg = _safe_float(projected.get("ppg", projected.get("points_per_game")), 0.0)

    actual_block = {
        "gp": gp,
        "games_played": gp,
        "goals": _safe_int(actual.get("goals"), 0),
        "assists": _safe_int(actual.get("assists"), 0),
        "points": pts,
        "ppg": actual_ppg,
        "points_per_game": actual_ppg,
        "pim": _safe_int(actual.get("pim"), 0),
        "wins": _safe_int(actual.get("wins"), 0),
        "losses": _safe_int(actual.get("losses"), 0),
        "ot_losses": _safe_int(actual.get("ot_losses"), 0),
        "save_pct": actual.get("save_pct"),
        "gaa": actual.get("gaa"),
        "shutouts": _safe_int(actual.get("shutouts"), 0),
    }
    projected_block = {
        "projected_gp": proj_gp,
        "projected_goals": _safe_int(projected.get("goals"), 0),
        "projected_assists": _safe_int(projected.get("assists"), 0),
        "projected_points": proj_pts,
        "projected_ppg": round(proj_ppg, 3) if proj_ppg else None,
        "projected_wins": _safe_int(projected.get("wins"), 0),
        "projected_save_pct": projected.get("save_pct"),
        "projected_gaa": projected.get("gaa"),
        "projected_shutouts": _safe_int(projected.get("shutouts"), 0),
    }

    out: Dict[str, Any] = {
        "actual_stats": actual_block,
        "projected_stats": projected_block,
        # Honest UI labeling: before any games are played the visible line is a
        # projection; once GP > 0 it is the season-to-date sample.
        "stats_mode": "current" if gp > 0 else "projected",
        "recent_form": dict(actual.get("recent_form") or {}),
        "stock_delta": actual.get("stock_delta"),
        "stock_label": actual.get("stock_label"),
        "stock_trend": actual.get("stock_trend"),
        "stock_reason": actual.get("stock_reason"),
        "weekly_stock_delta": actual.get("weekly_stock_delta"),
        "weekly_stock_label": actual.get("weekly_stock_label"),
        "weekly_stock_reason": actual.get("weekly_stock_reason"),
        "weekly_production_score": actual.get("weekly_production_score"),
        "weekly_analytics_score": actual.get("weekly_analytics_score"),
        "week_gp": actual.get("week_gp"),
        "week_points": actual.get("week_points"),
        "last_prospect_stat_update_date": getattr(prospect, "_prospect_last_stat_update_iso", ""),
        "prospect_games_simulated_to_date": getattr(prospect, "_prospect_games_simulated_to_date", gp),
    }

    for k, v in actual.items():
        if k.startswith("_"):
            continue
        if v is not None and k not in out:
            out[k] = v

    for k, v in actual_block.items():
        if v is not None:
            out[k] = v

    for k, v in projected_block.items():
        if v is not None:
            out[k] = v

    out["stat_history"] = [dict(r) for r in (getattr(prospect, "prospect_stat_history", None) or []) if isinstance(r, dict)]
    st_now = _stint(prospect)
    if st_now:
        out["stint_league"] = st_now.get("league_key")
        out["stint_start_iso"] = st_now.get("start_iso")
    analytics = derive_prospect_analytics(prospect, league, out)
    out["analytics"] = analytics
    for field in ("shots", "plus_minus", "primary_points", "shooting_pct", "shot_rate"):
        if field in analytics and analytics[field] is not None and out.get(field) in (None, "", 0):
            if field == "plus_minus" and out.get("plus_minus") not in (None, ""):
                continue
            out[field] = analytics[field]
    for field in ("xgf_pct", "cf_pct", "ff_pct", "war", "offensive_war", "defensive_war", "defensive_impact",
                  "quality_of_competition", "quality_of_teammates", "gsax", "quality_starts"):
        if field in analytics and analytics[field] is not None:
            out[field] = analytics[field]

    return {k: v for k, v in out.items() if v is not None}


def normalize_league_leader_board(
    rows: List[Dict[str, Any]],
    league: Any,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Light pass: if CHL/QMJHL top forward is still below target, bump top offensive rows slightly.
    Does not flatten the whole league — only nudges leaders for realism.
    """
    if not rows:
        return rows
    if not isinstance(rng, random.Random):
        rng = random.Random()
    key = normalize_prospect_league_key(league)
    if key not in ("CHL", "OHL", "WHL", "QMJHL"):
        return rows

    profile = get_league_scoring_profile(league)
    elite_lo = profile["elite_ppg_target"][0]
    skaters = [r for r in rows if str(r.get("position", "F")).upper() not in ("G", "GOALIE")]
    if not skaters:
        return rows
    top = max(skaters, key=lambda r: _safe_float(r.get("ppg", r.get("points_per_game")), 0))
    top_ppg = _safe_float(top.get("ppg", top.get("points_per_game")), 0)
    if top_ppg >= elite_lo:
        return rows

    target_ppg = rng.uniform(elite_lo, profile["elite_ppg_target"][1])
    gp = max(1, _safe_int(top.get("gp"), 60))
    new_pts = int(round(gp * target_ppg))
    top["points"] = new_pts
    top["ppg"] = round(new_pts / gp, 3)
    top["points_per_game"] = top["ppg"]
    # preserve goal share if possible
    g = _safe_int(top.get("goals"), int(new_pts * 0.42))
    top["goals"] = min(new_pts, g)
    top["assists"] = max(0, new_pts - top["goals"])
    return rows
