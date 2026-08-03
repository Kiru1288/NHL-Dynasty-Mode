"""Draft ranking modifiers, scouting intel, league/team cleaning, transcendent talent.

CANONICAL PATH for live franchise draft board:
- Generation: SimEngine/app/sim_engine/league_hierarchy_bootstrap.py
- Ranking/ETA/sanity: this module (draft_ranking_logic.py)
- Board assembly: backend/services/franchise_sim.py :: build_draft_class_rankings()
- Profiles: backend/services/draft_prospect_profile.py (imports ETA from here)

Do not duplicate ranking/ETA logic in SimEngine franchise monolith mirrors.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


def _stable_unit(*parts: Any) -> float:
    """Deterministic float in [0,1) from arbitrary parts.

    Uses MD5 (stable across process launches) instead of Python's built-in
    hash(), whose salt changes every interpreter start and would make the same
    franchise produce different consensus / storyline results after a restart.
    """
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _stable_int(modulo: int, *parts: Any) -> int:
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode()).hexdigest(), 16) % max(1, int(modulo))

TRANSCENDENT_CLASS_PROB = 0.0001
TRANSCENDENT_FORCE_DEBUG = False

GOALIE_CLASS_TIERS: List[Tuple[str, float, float]] = [
    ("weak", 0.25, -1.5),
    ("normal", 0.55, 0.0),
    ("strong", 0.15, 1.2),
    ("elite", 0.045, 2.5),
    ("generational", 0.005, 4.5),
]

_LEAGUE_PARENT_MAP = {
    "CHL_OHL": ("CHL", "OHL"),
    "CHL_WHL": ("CHL", "WHL"),
    "CHL_QMJHL": ("CHL", "QMJHL"),
    "OHL": ("CHL", "OHL"),
    "WHL": ("CHL", "WHL"),
    "QMJHL": ("CHL", "QMJHL"),
    "NCAA": ("NCAA", None),
    "USHL": ("USHL", None),
    "EU_J_SHL": (None, "J20 Nationell"),
    "EU_J_LIIGA": (None, "U20 SM-sarja"),
    "EU_J_DEL": (None, "DEL"),
    "EU_J_SWISS": (None, "NL"),
    "EU_J_CZ": (None, "Czech Extraliga"),
    "EU_J_SK": (None, "Slovak Extraliga"),
    "EU_J_KHL_JR": (None, "MHL"),
    "EU_J_NOR": (None, "Norway"),
    "EU_J_DEN": (None, "Denmark"),
    "EU_J_AUT": (None, "Austria"),
}

_LEAGUE_CODE_FRAGMENTS = re.compile(
    r"\b(CHL|EU_J|NCAA|USHL|AHL|WHL|OHL|QMJHL|SHL|LIIGA|DEL|SK|CZ)\b",
    re.I,
)

TRANSCENDENT_BACKSTORIES: List[Dict[str, Any]] = [
    {
        "key": "backyard_rink_kid",
        "title": "Backyard Rink Kid",
        "summary": "Creativity forged on backyard ice.",
        "traits": ["creativity", "drive", "pressure"],
        "full_text": "Family flooded the backyard every winter. Endless 1-on-1 games built hands, deception, and cold-weather toughness.",
    },
    {
        "key": "small_town_superstar",
        "title": "Small-Town Superstar",
        "summary": "The kid everyone knew by twelve.",
        "traits": ["pressure", "leadership", "drive"],
        "full_text": "Entire town packed the rink before he was a teenager. Carries expectation like a sweater.",
    },
    {
        "key": "late_bloomer",
        "title": "Late Bloomer",
        "summary": "Cut early, grew late, never forgot.",
        "traits": ["drive", "motor", "resilience"],
        "full_text": "Released from elite programs for size, then hit a growth spurt with a chip on his shoulder.",
    },
    {
        "key": "hockey_family_legacy",
        "title": "Hockey Family Legacy",
        "summary": "Raised around elite locker rooms.",
        "traits": ["hockey_iq", "poise", "pressure"],
        "full_text": "Grew up around pro habits, video sessions, and the unspoken standard of a hockey surname.",
    },
    {
        "key": "outdoor_pond_grinder",
        "title": "Outdoor Pond Grinder",
        "summary": "Rough ice built tough habits.",
        "traits": ["motor", "compete", "resilience"],
        "full_text": "Learned on choppy outdoor ice with bad skates and colder mornings than most kids ever see.",
    },
    {
        "key": "multi_sport_athlete",
        "title": "Multi-Sport Athlete",
        "summary": "Elite instincts from all-sport youth.",
        "traits": ["skating", "creativity", "athleticism"],
        "full_text": "Starred in multiple sports before committing fully to hockey, bringing unusual coordination and vision.",
    },
    {
        "key": "immigrant_family_dream",
        "title": "Immigrant Family Dream",
        "summary": "Family sacrifice fuels every shift.",
        "traits": ["drive", "gratitude", "pressure"],
        "full_text": "Family moved for opportunity. Hockey was expensive, so every shift carries gratitude and urgency.",
    },
    {
        "key": "undersized_skill_wizard",
        "title": "Undersized Skill Wizard",
        "summary": "Survived on IQ, hands, speed.",
        "traits": ["hands", "iq", "creativity"],
        "full_text": "Always the smallest player on the sheet, survived with deception, pace, and impossible hands.",
    },
    {
        "key": "captain_since_childhood",
        "title": "Captain Since Childhood",
        "summary": "Natural leader since street hockey.",
        "traits": ["leadership", "poise", "drive"],
        "full_text": "Organized neighborhood games, calmed teammates after losses, and demanded accountability early.",
    },
    {
        "key": "troublemaker_turned_competitor",
        "title": "Troublemaker Turned Competitor",
        "summary": "Fire redirected into winning.",
        "traits": ["compete", "edge", "motor"],
        "full_text": "Early discipline issues and hot temper were channeled into an obsessive hatred of losing.",
    },
]

ORIGIN_STORY_BY_KEY: Dict[str, Dict[str, Any]] = {str(s["key"]): dict(s) for s in TRANSCENDENT_BACKSTORIES}


def intel_label(confidence: float) -> str:
    c = float(confidence)
    if c <= 35:
        return "Unknown"
    if c <= 55:
        return "Limited"
    if c <= 75:
        return "Solid"
    if c <= 90:
        return "Strong"
    return "Locked"


def normalize_league_code(code: str, league_display: str = "") -> Dict[str, Optional[str]]:
    c = str(code or "").strip().upper()
    label = str(league_display or "").strip()
    try:
        from app.sim_engine.generation.prospect_league_teams import league_display_name

        clean = league_display_name(c)
    except Exception:
        clean = ""
    if not clean:
        parent, sub = _LEAGUE_PARENT_MAP.get(c, (None, None))
        if parent and sub:
            clean = sub
        elif parent:
            clean = parent
        elif sub:
            clean = sub
        elif label:
            if "Pro Jr" in label or re.search(r"\bjr\b", label, re.I):
                clean = label.split("/")[-1].strip()
            else:
                clean = label.split("/")[0].strip()[:24]
        else:
            clean = "Junior"
    clean = re.sub(r"^Pro\s*Jr\s*/\s*", "", clean, flags=re.I).strip()
    if " / " in clean and not clean.startswith("CHL"):
        clean = clean.split(" / ")[-1].strip()
    parent, sub = _LEAGUE_PARENT_MAP.get(c, (None, None))
    if sub and clean.lower() == sub.lower():
        parent = None
    return {"league_parent": parent, "league_sub": sub, "league_display": clean}


def fix_prospect_league_team_row(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from app.sim_engine.generation.prospect_league_teams import apply_prospect_league_team_fix

        return apply_prospect_league_team_fix(row)
    except Exception:
        return row


def clean_team_name(team_name: str, league_code: str = "", league_display: str = "") -> str:
    try:
        from app.sim_engine.generation.prospect_league_teams import normalize_team_display

        resolved = normalize_team_display(team_name, league_code)
        if resolved:
            return resolved
    except Exception:
        pass
    raw = re.sub(r"\s+", " ", str(team_name or "").strip())
    if not raw:
        return ""
    raw = re.sub(r"\s+\d+$", "", raw)
    raw = re.sub(r"\s+EU\s*J(?:\s+[A-Z]{1,6})?\s*", " ", raw, flags=re.I)
    for marker in (" CHL ", " NCAA ", " USHL ", " AHL ", " EU_J "):
        idx = raw.find(marker)
        if idx > 0:
            raw = raw[:idx].strip()
    parts = raw.split()
    league_bits = set()
    lp = normalize_league_code(league_code, league_display)
    for bit in (lp.get("league_parent"), lp.get("league_sub"), league_code):
        if bit:
            league_bits.add(str(bit).upper().replace("_", " "))
    cleaned: List[str] = []
    for p in parts:
        pu = p.upper()
        if p.isdigit():
            continue
        if pu in ("EU", "J", "CHL", "NCAA", "USHL", "AHL", "WHL", "OHL", "QMJHL", "SK", "CZ", "SHL"):
            break
        if pu in league_bits or _LEAGUE_CODE_FRAGMENTS.fullmatch(pu):
            break
        if pu.replace("_", " ") in league_bits:
            break
        cleaned.append(p)
    if cleaned:
        return " ".join(cleaned[:3])
    for p in parts:
        if not p.isdigit() and p.upper() not in ("EU", "J", "SK"):
            return p
    return raw.split()[0] if parts else raw


def roll_goalie_class_strength(rng: Any) -> Tuple[str, float]:
    roll = float(getattr(rng, "random", lambda: 0.5)())
    acc = 0.0
    for label, weight, boost in GOALIE_CLASS_TIERS:
        acc += weight
        if roll <= acc:
            return label, boost
    return "normal", 0.0


def size_score_modifier(row: Dict[str, Any]) -> float:
    """Draft score adjustment for height/production translation."""
    pos = str(row.get("position") or "").upper()
    h = int(row.get("height_cm") or 0)
    prod = float(row.get("production_adjusted_score") or 0)
    ppg = float(row.get("ppg") or row.get("points_per_game") or 0)
    role = infer_prospect_role(row)
    pot = float(row.get("potential_score") or row.get("true_potential_score") or 0)

    if pos == "G":
        if 0 < h < 185:
            return -4.0 if float(row.get("potential_score") or 0) < 86 else -1.5
        return 0.0

    if pos in ("D", "LD", "RD", "LHD", "RHD"):
        if h > 0 and h < 183:
            if prod >= 1.0 or ppg >= 0.75:
                return -0.5
            if role in ("defensive_defenseman", "two_way_defenseman") and pot >= 76:
                return -0.5
            return -3.5
        return 0.0

    if h > 0 and h < 176:
        if prod >= 1.35 or ppg >= 1.25:
            return 0.5
        if prod >= 1.0 or ppg >= 0.95:
            return -1.5
        if role in ("two_way_center", "shutdown_center", "two_way_forward") and pot >= 78:
            return -1.0
        return -5.0
    return 0.0


# Role-aware junior production expectations (CHL-normalized PPG baselines).
_ROLE_EXPECTED_PPG: Dict[str, float] = {
    "sniper": 0.98,
    "playmaker": 0.94,
    "scoring_forward": 0.78,
    "power_forward": 0.72,
    "two_way_forward": 0.58,
    "two_way_center": 0.62,
    "shutdown_center": 0.48,
    "grinder": 0.38,
    "offensive_defenseman": 0.68,
    "two_way_defenseman": 0.42,
    "defensive_defenseman": 0.28,
    "goalie": 0.0,
}

_ROLE_PRODUCTION_WEIGHT: Dict[str, float] = {
    "goalie": 0.0,
    "sniper": 1.0,
    "playmaker": 1.0,
    "scoring_forward": 0.92,
    "power_forward": 0.85,
    "two_way_forward": 0.62,
    "two_way_center": 0.66,
    "shutdown_center": 0.48,
    "grinder": 0.38,
    "offensive_defenseman": 0.90,
    "two_way_defenseman": 0.52,
    "defensive_defenseman": 0.30,
}

_LEAGUE_PPG_SCALE: Dict[str, float] = {
    "CHL": 1.0,
    "OHL": 1.02,
    "WHL": 0.96,
    "QMJHL": 1.06,
    "USHL": 0.74,
    "NCAA": 0.56,
    "SHL": 0.40,
    "LIIGA": 0.44,
    "EUROPE_JUNIOR": 0.46,
    "JUNIOR": 0.88,
    "AHL": 0.52,
}


# Rating-key → bucket classification is identical for players that share a ratings
# schema. Cache by frozenset(keys) so draft-board scans don't re-tokenize strings
# six times per prospect (~3s → tens of ms on a full junior pool).
_RATING_KEY_BUCKET_CACHE: Dict[frozenset, Dict[str, List[str]]] = {}


def _rating_key_buckets(ratings: Mapping[str, Any]) -> Dict[str, List[str]]:
    keyset = frozenset(ratings.keys())
    cached = _RATING_KEY_BUCKET_CACHE.get(keyset)
    if cached is not None:
        return cached
    buckets: Dict[str, List[str]] = {
        "def": [],
        "skate": [],
        "iq": [],
        "phys": [],
        "shoot": [],
        "pass": [],
    }
    for k in keyset:
        kl = str(k).lower()
        if any(x in kl for x in ("def", "stick", "gap", "block", "poke")):
            buckets["def"].append(k)
        if "skat" in kl:
            buckets["skate"].append(k)
        if any(x in kl for x in ("iq", "sense", "awareness")):
            buckets["iq"].append(k)
        if any(x in kl for x in ("strength", "physical", "reach", "body")):
            buckets["phys"].append(k)
        if any(x in kl for x in ("shoot", "shot", "finish", "release", "wrist", "one_timer", "snap")):
            buckets["shoot"].append(k)
        if any(x in kl for x in ("pass", "playmak", "vision", "distribut", "puck_skill", "hands")):
            buckets["pass"].append(k)
    # Bound cache growth for long-running servers with many ad-hoc schemas.
    if len(_RATING_KEY_BUCKET_CACHE) > 64:
        _RATING_KEY_BUCKET_CACHE.clear()
    _RATING_KEY_BUCKET_CACHE[keyset] = buckets
    return buckets


def enrich_prospect_row_from_player(player: Any, row: Dict[str, Any]) -> None:
    """Attach playstyle, archetype, and rating summaries for role-aware ranking."""
    chem = getattr(player, "chemistry_profile", None)
    if isinstance(chem, dict):
        row.setdefault("playstyle", chem.get("playstyle"))
        if chem.get("defensive_buy_in") is not None:
            row.setdefault("defensive_buy_in", float(chem.get("defensive_buy_in")))
    arch = getattr(player, "archetype", None)
    if arch is not None:
        row.setdefault("archetype", str(getattr(arch, "value", arch) or ""))
    tier = str(getattr(player, "pipeline_tier", "") or "").strip()
    if tier:
        row.setdefault("pipeline_tier", tier)

    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict) or not ratings:
        return

    def _avg_keys(keys: List[str]) -> float:
        vals = [float(ratings[k]) for k in keys if ratings.get(k) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    buckets = _rating_key_buckets(ratings)
    def_keys = buckets["def"]
    skate_keys = buckets["skate"]
    iq_keys = buckets["iq"]
    phys_keys = buckets["phys"]
    shoot_keys = buckets["shoot"]
    pass_keys = buckets["pass"]

    if def_keys:
        row.setdefault("def_rating", round(_avg_keys(def_keys), 1))
    if skate_keys:
        row.setdefault("skating_rating", round(_avg_keys(skate_keys), 1))
    if iq_keys:
        row.setdefault("iq_rating", round(_avg_keys(iq_keys), 1))
    if phys_keys:
        row.setdefault("physical_rating", round(_avg_keys(phys_keys), 1))
    pot = ratings.get("dev_potential")
    if pot is not None:
        row.setdefault("dev_potential", float(pot))

    # Emit the six named attributes the draft UI reads (skating/shooting/passing/defense/
    # physical/hockey_iq). Without these the frontend silently falls back to a flat 60 stub,
    # which is what made the "attribute snapshot" look fake. Values are the prospect's real
    # ratings; the UI still fuzzes them into ranges based on scouting confidence.
    def _emit(ui_key: str, keys: List[str]) -> None:
        if keys:
            v = _avg_keys(keys)
            if v > 0:
                row.setdefault(ui_key, int(round(v)))

    _emit("skating", skate_keys)
    _emit("shooting", shoot_keys)
    _emit("passing", pass_keys)
    _emit("defense", def_keys)
    _emit("physical", phys_keys)
    _emit("hockey_iq", iq_keys)


def infer_prospect_role(row: Mapping[str, Any]) -> str:
    pos = str(row.get("position") or "").upper()
    if pos == "G":
        return "goalie"
    blob = " ".join(
        str(row.get(k) or "") for k in ("playstyle", "archetype", "player_type", "_dev_archetype")
    ).lower().replace("_", " ")
    if pos in ("D", "LD", "RD", "LHD", "RHD"):
        if "offensive" in blob or "puck mover" in blob or "quarterback" in blob:
            return "offensive_defenseman"
        if "shutdown" in blob or "defensive" in blob or "stay at home" in blob:
            return "defensive_defenseman"
        return "two_way_defenseman"
    if pos == "C":
        if "shutdown" in blob:
            return "shutdown_center"
        if "two" in blob or "200" in blob:
            return "two_way_center"
    if "sniper" in blob or "shooter" in blob:
        return "sniper"
    if "playmaker" in blob or "passer" in blob:
        return "playmaker"
    if "power" in blob:
        return "power_forward"
    if "grinder" in blob or "checker" in blob or "energy" in blob:
        return "grinder"
    if "two" in blob:
        return "two_way_forward" if pos != "C" else "two_way_center"
    ppg = float(row.get("ppg") or row.get("points_per_game") or 0)
    ovr = float(row.get("true_ovr") or 0)
    if pos == "C" and ppg < 0.58 and ovr >= 70:
        return "shutdown_center" if ppg < 0.48 else "two_way_center"
    if pos in ("D", "LD", "RD", "LHD", "RHD") and ppg < 0.42 and ovr >= 70:
        return "defensive_defenseman"
    return "scoring_forward"


def _league_ppg_scale(row: Mapping[str, Any]) -> float:
    key = str(row.get("league_scoring_profile") or row.get("league_code") or "JUNIOR").upper()
    if key in _LEAGUE_PPG_SCALE:
        return _LEAGUE_PPG_SCALE[key]
    if key.startswith("CHL"):
        return 1.0
    if key.startswith("EU_J"):
        return _LEAGUE_PPG_SCALE["EUROPE_JUNIOR"]
    return _LEAGUE_PPG_SCALE.get("JUNIOR", 0.88)


def _role_expected_ppg(row: Mapping[str, Any], role: str) -> float:
    base = _ROLE_EXPECTED_PPG.get(role, 0.65)
    scale = _league_ppg_scale(row)
    ovr = float(row.get("true_ovr") or 0)
    pot = _true_pot(dict(row))
    talent = max(0.55, min(1.18, (ovr / 99.0) * 0.65 + (pot / 99.0) * 0.45))
    age = int(row.get("age") or 18)
    age_mult = 1.06 if age >= 20 else (1.02 if age == 19 else (0.94 if age <= 17 else 1.0))
    return base * scale * talent * age_mult


def _ppg_to_production_score(ppg: float, row: Mapping[str, Any]) -> float:
    diff = 0.62
    # Read the SAME source fields as _league_ppg_scale so a row with league_code
    # but no league_scoring_profile is normalized consistently by both functions.
    key = str(row.get("league_scoring_profile") or row.get("league_code") or "").upper()
    if key in ("QMJHL", "OHL", "CHL", "WHL") or key.startswith("CHL"):
        diff = 0.62
    elif key == "NCAA":
        diff = 0.82
    elif key in ("SHL", "LIIGA", "EUROPE_JUNIOR", "DEL") or key.startswith("EU_"):
        diff = 0.88
    adj = ppg * (1.0 - diff * 0.35)
    if int(row.get("age") or 18) >= 20:
        adj *= 0.82
    return max(0.0, adj)


def compute_role_adjusted_production(row: Dict[str, Any]) -> float:
    """Normalize junior scoring by playstyle/position expectations."""
    role = infer_prospect_role(row)
    if role == "goalie":
        return 0.0

    raw = float(row.get("production_adjusted_score") or 0)
    ppg = float(row.get("ppg") or row.get("points_per_game") or 0)
    if raw <= 0 and ppg > 0:
        raw = _ppg_to_production_score(ppg, row)

    expected_ppg = _role_expected_ppg(row, role)
    expected_prod = _ppg_to_production_score(expected_ppg, row)
    if expected_prod <= 0.04:
        return round(raw, 3)

    ratio = raw / expected_prod
    weight = _ROLE_PRODUCTION_WEIGHT.get(role, 0.75)
    if ratio >= 1.40:
        adj = raw * (1.0 + min(0.45, (ratio - 1.0) * 0.30 * weight))
    elif ratio >= 1.0:
        adj = raw * (0.88 + (ratio - 1.0) * 0.22 * weight)
    elif ratio >= 0.72:
        adj = raw * (0.78 + (ratio - 0.72) * 0.55)
    else:
        shortfall = min(1.0, (0.72 - ratio) / 0.72)
        adj = raw * max(0.30, 1.0 - shortfall * weight * 0.62)

    return round(min(2.4, max(0.0, adj)), 3)


def compute_defensive_projection_bonus(row: Dict[str, Any]) -> float:
    """Reward shutdown D / two-way C profiles when tools and projection support it."""
    role = infer_prospect_role(row)
    if role not in ("defensive_defenseman", "two_way_defenseman", "shutdown_center", "two_way_center"):
        return 0.0

    pot = _true_pot(row)
    ovr = float(row.get("true_ovr") or 0)
    if pot < 72 or ovr < 66:
        return 0.0

    def_r = float(row.get("def_rating") or 0)
    skate = float(row.get("skating_rating") or 0)
    iq = float(row.get("iq_rating") or 0)
    phys = float(row.get("physical_rating") or 0)
    h = int(row.get("height_cm") or 0)
    buy_in = float(row.get("defensive_buy_in") or 55)

    tool_score = 0.0
    if def_r > 0:
        tool_score += max(0.0, def_r - 68.0) * 0.09
    if skate > 0:
        tool_score += max(0.0, skate - 70.0) * 0.07
    if iq > 0:
        tool_score += max(0.0, iq - 68.0) * 0.06
    if phys > 0:
        tool_score += max(0.0, phys - 70.0) * 0.05
    if h >= 193:
        tool_score += 0.9
    elif h >= 188:
        tool_score += 0.45
    tool_score += max(0.0, buy_in - 58.0) * 0.05
    tool_score += max(0.0, pot - 74.0) * 0.13
    tool_score += max(0.0, ovr - 70.0) * 0.11

    ppg = float(row.get("ppg") or row.get("points_per_game") or 0)
    if role == "defensive_defenseman":
        mult = 1.18 if ppg <= 0.55 else 0.88
    elif role == "shutdown_center":
        mult = 1.12 if ppg <= 0.75 else 0.90
    else:
        mult = 0.96

    bonus = tool_score * mult
    max_bonus = 7.5 if pot >= 82 else (5.5 if pot >= 76 else 3.5)
    return min(max_bonus, bonus)


def build_draft_rank_reason_codes(row: Dict[str, Any]) -> List[str]:
    """Scouting reason codes explaining why a prospect ranks where they do."""
    codes: List[str] = []
    role = infer_prospect_role(row)
    ppg = float(row.get("ppg") or row.get("points_per_game") or 0)
    prod = float(row.get("production_adjusted_score") or 0)
    role_adj = float(row.get("role_adjusted_production") or compute_role_adjusted_production(row))
    pot = _true_pot(row)
    ovr = float(row.get("true_ovr") or 0)
    def_bonus = float(row.get("defensive_projection_bonus") or compute_defensive_projection_bonus(row))

    if role != "goalie":
        if ppg >= 1.75 or prod >= 1.65:
            codes.append("elite_junior_production")
        elif ppg >= 1.25 or prod >= 1.25:
            codes.append("high_junior_production")
        if role_adj > prod * 1.08 and prod > 0:
            codes.append("role_adjusted_production")

    if role == "defensive_defenseman" and def_bonus >= 3.0:
        codes.append("defensive_defenseman_projection")
    elif role == "shutdown_center" and def_bonus >= 3.0:
        codes.append("shutdown_center_projection")
    elif role in ("two_way_center", "two_way_defenseman") and def_bonus >= 2.5:
        codes.append("two_way_center_value" if "center" in role else "defensive_role_context")

    if role in ("defensive_defenseman", "two_way_defenseman") and ppg < 0.48 and pot >= 76 and ovr >= 70:
        codes.append("low_scoring_toolsy_defenseman")

    expected = _role_expected_ppg(row, role)
    if role in ("sniper", "playmaker", "scoring_forward", "offensive_defenseman", "power_forward"):
        if ppg > 0 and ppg < expected * 0.62 and ovr >= 68:
            codes.append("offensive_role_underproduction")
        if ppg > 0 and ppg < expected * 0.48:
            codes.append("production_concern")

    if role in ("defensive_defenseman", "shutdown_center", "grinder") and ppg < 0.55:
        codes.append("defensive_role_context")

    return codes[:6]


def compute_consensus_potential_evaluation(row: Dict[str, Any]) -> float:
    """League/market-facing ceiling estimate from observable signals only.

    Does NOT read hidden truth: no true_potential_score, and no is_transcendent /
    transcendent_talent / generational_goalie flags. The public market must earn
    its ceiling read from production, tools, size, age and league — a truly
    special player rises because his observable evidence is elite, not because a
    backend flag revealed the truth. This preserves realistic public mistakes.
    """
    ovr = float(row.get("true_ovr") or 0)
    prod = float(row.get("production_adjusted_score") or 0)
    ppg = float(row.get("ppg") or row.get("points_per_game") or 0)
    age = int(row.get("age") or 18)
    h = int(row.get("height_cm") or 0)
    code = str(row.get("league_code") or "").upper()
    pos = str(row.get("position") or "").upper()

    prod_lift = min(16.0, prod * 9.5 + ppg * 4.0)
    production_ceiling = ovr + prod_lift

    tool_lift = 0.0
    def_r = float(row.get("def_rating") or 0)
    skate = float(row.get("skating_rating") or 0)
    iq = float(row.get("iq_rating") or 0)
    if def_r > 0:
        tool_lift += max(0.0, def_r - 68.0) * 0.22
    if skate > 0:
        tool_lift += max(0.0, skate - 70.0) * 0.18
    if iq > 0:
        tool_lift += max(0.0, iq - 68.0) * 0.16
    tools_ceiling = ovr + tool_lift

    if pos == "G":
        # No generational_goalie truth shortcut — goalie ceiling is read from
        # current ability + observable production/tools like everyone else.
        return round(max(50.0, min(92.0, ovr * 0.62 + production_ceiling * 0.38 + tool_lift)), 1)

    market = production_ceiling * 0.58 + tools_ceiling * 0.42

    if prod >= 1.35 or ppg >= 1.15:
        market += 2.5
    elif prod < 0.32 and ppg < 0.42:
        market -= 3.5
    if code.startswith("EU_J") and code not in ("EU_J_SHL", "EU_J_LIIGA", "EU_J_DEL"):
        market -= 2.8
    elif code in ("CHL_OHL", "CHL_WHL", "CHL_QMJHL", "NCAA", "USHL"):
        market += 1.2
    if 0 < h < 178 and pos not in ("G", "D", "LD", "RD", "LHD", "RHD"):
        market -= 2.6
    elif h >= 193:
        market += 1.4
    if skate > 0 and skate < 62:
        market -= 3.2
    elif skate >= 80:
        market += 1.6
    # Age affects UPSIDE (ceiling), not readiness. An overager is closer to his
    # peak, so his projectable ceiling is lower, not higher. (Readiness is handled
    # separately in ETA; ranking age penalty in compute_enhanced_draft_score.)
    if age >= 20:
        market -= 1.8
    elif age <= 17:
        market += 1.2
    if row.get("character_concerns"):
        market -= 2.4
    if row.get("is_bust_risk"):
        market -= 1.6

    key = str(row.get("key") or row.get("name") or "")
    hsh = _stable_int(1000, key, row.get("draft_year") or "", row.get("franchise_seed") or "")
    market += ((hsh % 17) - 8) * 0.45

    if prod >= 1.45 and age >= 19:
        market = max(market, ovr + min(14.0, prod * 5.5))

    return round(max(50.0, min(97.0, market)), 1)


def compute_enhanced_draft_score(row: Dict[str, Any]) -> float:
    ovr = float(row.get("true_ovr") or 0)
    pot = float(
        row.get("consensus_potential_score")
        or row.get("potential_score")
        or ovr
    )
    role = infer_prospect_role(row)

    if role == "goalie":
        prod_adj = 0.0
    else:
        role_adj = compute_role_adjusted_production(row)
        raw_prod = float(row.get("production_adjusted_score") or 0)
        weight = _ROLE_PRODUCTION_WEIGHT.get(role, 0.75)
        prod_adj = role_adj * 0.74 + raw_prod * 0.26 * weight
        row["prospect_role"] = role
        row["role_adjusted_production"] = role_adj

    def_bonus = compute_defensive_projection_bonus(row)
    row["defensive_projection_bonus"] = round(def_bonus, 2)

    score = ovr * 0.48 + pot * 0.36 + prod_adj * 5.5 + def_bonus
    age = int(row.get("age") or 18)
    if age <= 18:
        score += 2.0
    elif age >= 20:
        score -= 2.5
    if row.get("is_gem"):
        score += 1.5
    if row.get("is_transcendent") or row.get("transcendent_talent"):
        score += 25.0
    score += size_score_modifier(row)
    if row.get("size_risk_flag"):
        score -= 2.0
    return score


def compute_prospect_outcome_band(row: Dict[str, Any]) -> Dict[str, Any]:
    """Derive an anti-correlated floor/ceiling band for a prospect.

    Design goal: a high ceiling should NOT come with a high floor, and a modest ceiling
    should come with a high (reliable) floor. This creates a genuine risk/reward tradeoff so
    a safe prospect is a legitimate pick versus a boom/bust swing — especially later in the
    draft. The mechanism: outcome VOLATILITY rises with ceiling height, the projection gap
    (ceiling - current ability) and youth, and falls for older/known/steady prospects. The
    floor is then the ceiling minus a spread that widens with that volatility, so risky
    high-ceiling players fall far below their ceiling while safe players sit just under it."""
    ceiling = float(row.get("potential_score") or row.get("true_potential_score") or 0)
    current = float(row.get("true_ovr") or 0)
    if ceiling <= 0:
        return {}
    gap = max(0.0, ceiling - current)
    age = int(row.get("age") or 18)
    boom = bool(row.get("is_bust_risk") or row.get("boom_bust"))
    gem = bool(row.get("is_gem"))

    # Outcome volatility (0..1): higher ceilings are inherently riskier bets.
    vol = 0.10
    vol += 0.014 * max(0.0, ceiling - 72.0)
    vol += 0.016 * gap
    if boom:
        vol += 0.20
    if age <= 17:
        vol += 0.08
    elif age >= 20:
        vol -= 0.14
    if gem:
        vol -= 0.06
    # A stable per-prospect jitter so identical inputs don't produce identical bands.
    vol += (_stable_unit("outcome_vol", str(row.get("key") or row.get("id") or "")) - 0.5) * 0.10
    vol = max(0.03, min(0.95, vol))

    rank = 0
    try:
        rank = int(row.get("rank") or 0)
    except (TypeError, ValueError):
        rank = 0
    conf = 0.0
    try:
        conf = float(row.get("scouting_confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0

    natural_floor = max(38.0, min(ceiling - 2.0, ceiling - (3.0 + vol * 18.0 + 0.08 * gap)))

    # Volatility PROPENSITY (0..1): only prospects that stack real risk signals become
    # true boom/bust — so most of the class lands in a moderate "Balanced" band rather
    # than everyone being a coin flip. A stable per-prospect jitter breaks ties.
    u = _stable_unit("boom_designate", str(row.get("key") or row.get("id") or ""))
    prop = 0.0
    if boom:
        prop += 0.55
    if row.get("character_concerns"):
        prop += 0.12
    if age <= 17:
        prop += 0.13
    elif age == 18:
        prop += 0.05
    elif age >= 20:
        prop -= 0.18
    prop += 0.18 * max(0.0, min(1.0, (gap - 14.0) / 16.0))  # only genuinely big gaps add risk
    if conf < 58:
        prop += 0.11
    elif conf >= 78:
        prop -= 0.16
    if gem:
        prop -= 0.30
    if gap < 9:
        prop -= 0.25
    prop += (u - 0.5) * 0.28
    high_variance = prop >= 0.60

    # Safe profiles: gems / clean, well-scouted, tight-gap prospects.
    is_safe = not high_variance and (
        gem or (not boom and not row.get("character_concerns") and conf >= 66 and gap < 12)
    )

    # Every pick from round 1 through 7 is drafted with the expectation that the
    # prospect will play NHL games, so every draftable-range prospect carries a
    # rank-scaled "projects to the NHL" floor: ~63 for the #1 pick, tapering toward
    # ~50 near the end of the ~7-round board (~pick 224). This guarantees drafted
    # prospects read as future NHLers instead of dead-on-arrival busts, while the
    # ceiling gap still supplies the risk/reward of who becomes a star vs. a role player.
    draft_nhl_floor = 0.0
    if rank and rank <= 224:
        draft_nhl_floor = 63.0 - (rank - 1) * (63.0 - 50.0) / 223.0

    if high_variance:
        # True boom/bust — real downside, but a drafted swing is still expected to make
        # the NHL, so it keeps a softened floor well below the safe-pick guarantee.
        boom_floor = ceiling - (15.0 + vol * 20.0)
        floor = max(42.0, min(natural_floor, boom_floor))
        if draft_nhl_floor:
            floor = max(floor, min(ceiling - 3.0, draft_nhl_floor - 8.0))
    elif is_safe:
        floor = max(natural_floor, ceiling - 9.0)
        if draft_nhl_floor:
            floor = max(floor, min(ceiling - 1.0, draft_nhl_floor + 2.0))
    else:
        # Balanced majority — a moderate NHL floor that still carries real downside
        # (neither a locked safe pick nor a coin flip).
        floor = ceiling - (10.0 + 5.0 * max(0.0, min(1.0, (gap - 10.0) / 20.0)))
        floor = min(natural_floor if natural_floor < floor else floor, floor)
        if draft_nhl_floor:
            floor = max(floor, min(ceiling - 2.0, draft_nhl_floor))

    floor = min(floor, ceiling - 1.0)
    band_width = ceiling - floor

    if boom or band_width >= 22.0:
        label = "Boom/Bust"
    elif band_width <= 9.0:
        label = "Safe Floor"
    else:
        label = "Balanced"

    return {
        "floor_score": round(floor, 1),
        "ceiling_score": round(ceiling, 1),
        "outcome_volatility": round(vol, 3),
        "outcome_band": label,
        "outcome_band_width": round(band_width, 1),
    }


def compute_ceiling_visibility(rank: Any, scout_overlay_pct: Any = 0.0) -> Dict[str, Any]:
    """How readable a prospect's CEILING is, driven by draft position.

    Design goal: an early-round pick is under a national spotlight, so its ceiling is
    obvious. The deeper you go, the less attention a prospect gets, so the read on his
    ceiling fades — first to a vague range ("fogged"), then vanishes entirely ("hidden")
    leaving only the reliable floor. It is then up to the user to project the ceiling from
    production, draft-year analytics, age, size and raw attributes.

    ``scout_overlay_pct`` is the user's OWN dedicated scouting progress on this prospect
    (0 when they haven't scouted him). Only that effort — never the ambient/public
    confidence a prospect gets just from playing games — can re-open a late prospect's
    ceiling. That way heavy late-round scouting is rewarded without every mid-round kid
    with a games-played sample auto-revealing his ceiling.

    Returns ``visibility`` (0..1), a ``state`` of ``clear``/``fogged``/``hidden`` and a
    convenience ``ceiling_hidden`` boolean.
    """
    try:
        r = int(rank or 0)
    except (TypeError, ValueError):
        r = 0
    if r <= 0:
        r = 999
    try:
        overlay = float(scout_overlay_pct or 0.0)
    except (TypeError, ValueError):
        overlay = 0.0

    # Public attention decays with draft position. Non-linear so all of round 1 stays
    # clear and the fade bites through the middle rounds (≈clear R1, fogged R2-3, hidden R4+).
    attention = max(0.0, min(1.0, (160.0 - r) / 150.0)) ** 1.3
    # Only the user's dedicated scouting effort re-opens a late prospect's ceiling.
    scout = max(0.0, min(1.0, (overlay - 20.0) / 70.0))
    vis = max(attention, attention * 0.35 + scout * 0.85)
    vis = max(0.0, min(1.0, vis))

    if vis >= 0.62:
        state = "clear"
    elif vis >= 0.32:
        state = "fogged"
    else:
        state = "hidden"
    return {
        "visibility": round(vis, 3),
        "state": state,
        "ceiling_hidden": state == "hidden",
    }


def build_potential_intel(
    center_pot: float,
    confidence: float,
    *,
    overlay_pct: Optional[float] = None,
    include_true: bool = False,
    seed_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Scout-facing ceiling estimate built around a persistent estimated centre.

    ``center_pot`` should be the observable/consensus ceiling — NOT hidden true
    potential. When ``seed_key`` is provided the evaluation miss is deterministic
    per prospect (and per team, if the key includes a team id), so two prospects
    at the same confidence do NOT receive the identical bias. Higher confidence
    narrows the uncertainty window around that persistent centre; it does not just
    re-reveal a hidden true value.

    By default this does NOT return hidden true potential (client-safe).
    Pass include_true=True only for server-side audits / internal tooling.
    """
    conf = float(overlay_pct if overlay_pct is not None else confidence)
    conf = max(0.0, min(100.0, conf))
    # Gradual fog: higher confidence narrows band and reduces bias, never zeros uncertainty.
    gap = max(2.5, (100.0 - conf) * 0.28)
    # Persistent evaluation miss — magnitude scales with fog.
    bias_span = max(0.0, (100.0 - conf) * 0.055)
    if seed_key is not None:
        # Deterministic per-prospect/team miss (stable across process restarts).
        bias = (_stable_unit(seed_key, "pot_bias") - 0.5) * 2.0 * bias_span
    else:
        # Legacy fallback: bias derived from confidence alone.
        bias = ((int(conf * 10) % 17) - 8) * (bias_span / 8.0)

    # The estimated centre is offset from the input centre by the persistent bias
    # and only converges toward it as confidence approaches certainty.
    convergence = min(1.0, conf / 100.0)
    est_center = float(center_pot) + bias * (1.0 - convergence * 0.5)
    true_pot = est_center

    low = max(50.0, float(true_pot) - gap + bias * 0.35)
    high = min(99.0, float(true_pot) + gap * 0.55 + bias * 0.25)
    if high < low:
        low, high = high, low
    # Even at high confidence keep a small uncertainty window.
    if conf >= 90:
        low = max(50.0, min(low, float(true_pot) - 1.5))
        high = min(99.0, max(high, float(true_pot) + 1.2))
    elif conf >= 75:
        low = max(50.0, min(low, float(true_pot) - 3.5))
        high = min(99.0, max(high, float(true_pot) + 2.5))

    visible = round((low + high) / 2.0 + bias * 0.15, 1)
    visible = max(50.0, min(97.0, visible))
    out: Dict[str, Any] = {
        # User-visible / estimated fields (legacy keys preserved for frontend).
        "potential_score": visible,  # estimated expected ceiling (NOT true)
        "expected_ceiling_estimate": visible,
        "ceiling_range": {
            "low": round(low, 1),
            "high": round(high, 1),
            "confidence": round(conf, 1),
        },
        "potential_range": {
            "low": round(low, 1),
            "high": round(high, 1),
            "confidence": round(conf, 1),
        },
        "scouting_confidence": round(conf, 1),
        "uncertainty": round(max(1.0, high - low), 1),
        "intel_label": intel_label(conf),
    }
    if include_true:
        # Server-only — never attach on public draft-board serialization.
        out["true_potential_score"] = round(float(true_pot), 1)
    return out


def scouting_confidence_for_entry(
    row: Dict[str, Any],
    session: Any,
    *,
    base_conf: float,
) -> float:
    """Merge base intel with real scouting_state overlay when present."""
    pid = str(row.get("key") or "")
    overlay_pct: Optional[float] = None
    scouting = getattr(session, "scouting_state", None) or {}
    prospects = scouting.get("prospects") if isinstance(scouting, dict) else {}
    if pid and isinstance(prospects, dict):
        ov = prospects.get(pid)
        if isinstance(ov, dict) and ov.get("scouted_percentage") is not None:
            overlay_pct = float(ov["scouted_percentage"])
    if overlay_pct is not None:
        return max(0.0, min(100.0, overlay_pct))
    return float(base_conf)


MIN_BOARD_GOALIES = 12
TARGET_BOARD_GOALIES = 16
MAX_BOARD_ENTRIES = 320
VISIBILITY_INJECT_FROM_RANK = 96

# Soft caps — talent can still earn early slots; surplus only demotes weaker goalies.
GOALIE_TOP32_CAPS: Dict[str, int] = {
    "weak": 1,
    "normal": 2,
    "strong": 3,
    "elite": 4,
    "generational": 5,
}

GOALIE_TOP10_CAPS: Dict[str, int] = {
    "weak": 0,
    "normal": 1,
    "strong": 1,
    "elite": 2,
    "generational": 2,
}

# Floor only applies to non-special goalies after score ranking — much less punitive.
GOALIE_MIN_RANK_BY_CLASS: Dict[str, int] = {
    "weak": 40,
    "normal": 16,
    "strong": 10,
    "elite": 6,
    "generational": 3,
}

GOALIE_CLASS_BOOST: Dict[str, float] = {
    "weak": -1.5,
    "normal": 0.0,
    "strong": 1.2,
    "elite": 2.5,
    "generational": 4.5,
}


def _is_goalie_row(row: Dict[str, Any]) -> bool:
    return str(row.get("position") or "").upper() == "G"


def _true_pot(row: Dict[str, Any]) -> float:
    return float(row.get("true_potential_score") or row.get("potential_score") or 0)


def _public_pot(row: Dict[str, Any]) -> float:
    """Observable/consensus ceiling used to construct the PUBLIC board.

    Deliberately never reads true_potential_score so the public consensus can make
    realistic mistakes (miss a hidden gem, overrate a bust). True-potential is for
    audits only, not for building the public board.
    """
    return float(row.get("consensus_potential_score") or row.get("potential_score") or 0)


def _effective_draft_score(row: Dict[str, Any]) -> float:
    return float(row.get("_score", 0)) - float(row.get("_sanity_penalty", 0))


def _short_elite_exception(row: Dict[str, Any]) -> bool:
    if row.get("is_transcendent") or row.get("transcendent_talent"):
        return True
    if str(row.get("backstory_key") or "") == "undersized_skill_wizard":
        return True
    prod = float(row.get("production_adjusted_score") or 0)
    ppg = float(row.get("ppg") or row.get("points_per_game") or 0)
    if prod >= 1.35 or ppg >= 1.25:
        return True
    ovr = float(row.get("true_ovr") or 0)
    pot = _true_pot(row)
    if ovr >= 74 and pot >= 80:
        return True
    return False


def _goalie_is_special(row: Dict[str, Any], goalie_class_strength: str = "normal") -> bool:
    """Only genuine FRANCHISE-potential goalies may be first-rounders. Everyone else is
    scattered into Rounds 2-7 regardless of how strong they look.

    Deliberately keyed on the rare explicit franchise markers (generational / transcendent /
    franchise pipeline tier) rather than a raw potential threshold, because goalie potential
    scores run high and a numeric gate would let ordinary goalies flood Round 1."""
    if row.get("is_transcendent") or row.get("transcendent_talent"):
        return True
    if row.get("generational_goalie"):
        return True
    if str(row.get("pipeline_tier") or "").lower() in ("franchise", "transcendent"):
        return True
    return False


def enforce_goalie_scatter_final(
    board: List[Dict[str, Any]],
    *,
    goalie_class_strength: str = "normal",
) -> List[Dict[str, Any]]:
    """Final, positional guarantee that non-franchise goalies are never Round 1 (top 32) and
    are spread across later rounds. Runs AFTER every score/potential pass so nothing can
    re-promote a high-potential goalie back into the first round."""
    if not board:
        return board
    gclass = str(goalie_class_strength or "normal").lower()

    def _movable(r: Dict[str, Any]) -> bool:
        return _is_goalie_row(r) and not _goalie_is_special(r, gclass)

    movers = [r for r in board if _movable(r)]
    if not movers:
        return board

    keep = [r for r in board if not _movable(r)]
    n = len(board)
    # Deterministic target index (0-based, >=32) by talent tier + stable jitter.
    movers.sort(key=lambda r: (_goalie_scatter_rank(r, n) - 1, str(r.get("key") or "")))

    placed = list(keep)
    for r in movers:
        idx = max(32, min(len(placed), _goalie_scatter_rank(r, n) - 1))
        # Nudge past an immediately adjacent goalie so they don't clump at one slot.
        while idx < len(placed) and _is_goalie_row(placed[idx]):
            idx += 1
        placed.insert(min(idx, len(placed)), r)
        r["goalie_rank_reason"] = "goalie_scattered_non_elite"

    board[:] = placed[:n]
    return board


def _goalie_scatter_rank(row: Dict[str, Any], n: int) -> int:
    """Deterministic board rank (1-based, >=33) for a non-franchise goalie.

    Bars them from Round 1 and SPREADS them across the entire back half of the draft
    (Rounds 2-7) rather than clustering them just outside Round 1. A stronger goalie can
    reach a slightly earlier slot, but the spread stays wide so goalies don't bunch up.
    Uses true OVR for the mild talent lean because goalie potential scores run compressed-
    high (which is exactly why a potential-based gate let them flood the top)."""
    key = str(row.get("key") or row.get("id") or row.get("name") or "")
    ovr = float(row.get("true_ovr") or 0)
    talent = max(0.0, min(1.0, (ovr - 60.0) / 22.0))  # ~60 OVR -> 0, ~82+ -> 1

    span_lo = 33
    span_hi = min(int(n), 224) if n else 224
    span_hi = max(span_lo + 8, span_hi)

    # Earliest reachable slot scales with talent: an elite non-franchise goalie can reach the
    # top of Round 2; a fringe goalie starts deeper. The stable jitter then spreads them out.
    reach = span_lo + int((1.0 - talent) * 48)  # 33..81
    reach = min(reach, span_hi - 8)
    u = _stable_unit("goalie_scatter", key)
    target = reach + int(u * (span_hi - reach))
    return max(span_lo, min(span_hi, target))


def _nhl_ready_low_ceiling_exception(row: Dict[str, Any]) -> bool:
    ovr = float(row.get("true_ovr") or 0)
    prod = float(row.get("production_adjusted_score") or 0)
    pot = _true_pot(row)
    if _is_goalie_row(row) and (row.get("generational_goalie") or pot >= 88):
        return True
    if ovr >= 74 and prod >= 1.0:
        return True
    if ovr >= 72 and pot >= 58:
        return True
    return False


def _ranking_violation_reason(row: Dict[str, Any], rank: int) -> Optional[str]:
    pot = _true_pot(row)
    ovr = float(row.get("true_ovr") or 0)
    pos = str(row.get("position") or "").upper()
    prod = float(row.get("production_adjusted_score") or 0)
    h = int(row.get("height_cm") or 0)

    if rank <= 10 and pos not in ("G", "D", "LD", "RD", "LHD", "RHD"):
        if 0 < h < 176 and not _short_elite_exception(row):
            return "short_top10_demoted"

    if rank <= 5 and pot < 78 and ovr < 74:
        return "low_ceiling_demoted"
    if rank <= 10 and pot < 75 and ovr < 70:
        return "low_ceiling_demoted"
    if rank <= 20 and pot < 70 and ovr < 68:
        return "low_ceiling_demoted"
    if rank <= 32 and pot < 60:
        return "low_ceiling_demoted"
    if rank <= 32 and pot < 64:
        if _nhl_ready_low_ceiling_exception(row):
            return None
        return "low_ceiling_demoted"
    if rank <= 32 and pot < 66 and ovr < 65:
        return "low_ceiling_demoted"
    if rank <= 40 and pot <= 60:
        if _nhl_ready_low_ceiling_exception(row):
            return None
        return "low_ceiling_demoted"
    if rank <= 64 and pot <= 52:
        return "low_ceiling_demoted"
    return None


def _demote_row_to_min_rank(
    prospects: List[Dict[str, Any]],
    index: int,
    min_rank: int,
    *,
    reason: str,
    penalty_floor: float = 40.0,
) -> None:
    """Move a row to at least ``min_rank`` (1-based) on the board."""
    if index < 0 or index >= len(prospects):
        return
    row = prospects.pop(index)
    row["ranking_reason"] = reason
    row["ranking_flag"] = reason
    target_idx = min(len(prospects), max(0, int(min_rank) - 1))
    if target_idx > 0:
        anchor = _effective_draft_score(prospects[target_idx - 1])
    elif prospects:
        anchor = _effective_draft_score(prospects[-1])
    else:
        anchor = 0.0
    needed = float(row.get("_score", 0)) - anchor + 2.5
    row["_sanity_penalty"] = max(float(row.get("_sanity_penalty", 0)), penalty_floor, needed)
    prospects.insert(target_idx, row)


def apply_hard_ranking_floor_pass(prospects: List[Dict[str, Any]]) -> None:
    """Hard demotion pass — violators in top 64 are moved down the board."""
    if not prospects:
        return

    prospects.sort(key=lambda r: -_effective_draft_score(r))
    guard = 0
    max_moves = max(256, len(prospects) * 2)
    while guard < max_moves:
        guard += 1
        moved = False
        scan_limit = min(96, len(prospects))
        for i in range(scan_limit):
            row = prospects[i]
            rank = i + 1
            reason = _ranking_violation_reason(row, rank)
            if not reason:
                continue
            pot = _true_pot(row)
            if rank <= 32 and pot < 60:
                target = 68
                floor_penalty = 72.0
            elif rank <= 32 and pot < 64:
                target = 48
                floor_penalty = 58.0
            elif rank <= 20 and pot < 70:
                target = 28
                floor_penalty = 52.0
            elif rank <= 10:
                target = 18
                floor_penalty = 48.0
            else:
                target = 40
                floor_penalty = 44.0
            _demote_row_to_min_rank(
                prospects, i, target, reason=reason, penalty_floor=floor_penalty
            )
            moved = True
            break
        if not moved:
            break

    prospects.sort(key=lambda r: -_effective_draft_score(r))


def apply_potential_band_enforcement(
    prospects: List[Dict[str, Any]],
    *,
    band_size: int = 32,
) -> None:
    """Reserve the top band for prospects that pass potential floor checks."""
    if not prospects:
        return

    prospects.sort(key=lambda r: -_effective_draft_score(r))
    # Gate on OBSERVABLE consensus potential, not hidden true potential, so the
    # public board is allowed to be wrong about a prospect's real ceiling.
    eligible = [
        r for r in prospects
        if _public_pot(r) >= 60 or _nhl_ready_low_ceiling_exception(r)
    ]
    eligible.sort(key=lambda r: -_effective_draft_score(r))

    head: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    for row in eligible:
        if len(head) >= band_size:
            deferred.append(row)
            continue
        rank = len(head) + 1
        reason = _ranking_violation_reason(row, rank)
        if reason:
            row["ranking_reason"] = reason
            row["ranking_flag"] = reason
            deferred.append(row)
        else:
            head.append(row)

    guard = 0
    while len(head) < band_size and deferred and guard < band_size * 4:
        guard += 1
        deferred.sort(key=lambda r: (-_public_pot(r), -_effective_draft_score(r)))
        progress = False
        remaining: List[Dict[str, Any]] = []
        for row in deferred:
            if len(head) >= band_size:
                remaining.append(row)
                continue
            rank = len(head) + 1
            reason = _ranking_violation_reason(row, rank)
            if reason:
                row["ranking_reason"] = reason
                row["ranking_flag"] = reason
                remaining.append(row)
            else:
                head.append(row)
                progress = True
        if not progress:
            break
        deferred = remaining

    used = {id(r) for r in head}
    still_out: List[Dict[str, Any]] = []
    for row in prospects:
        if id(row) in used:
            continue
        if _public_pot(row) < 60 and not _nhl_ready_low_ceiling_exception(row):
            row["ranking_reason"] = "low_ceiling_demoted"
            row["ranking_flag"] = "low_ceiling_demoted"
        still_out.append(row)
    still_out.sort(key=lambda r: -_effective_draft_score(r))
    prospects[:] = head + still_out


def apply_ranking_sanity_pass(prospects: List[Dict[str, Any]]) -> None:
    """Canonical ranking guardrail — delegates to hard floor pass."""
    apply_hard_ranking_floor_pass(prospects)


def apply_goalie_class_rank_caps(
    prospects: List[Dict[str, Any]],
    *,
    goalie_class_strength: str = "normal",
    preserve_order: bool = False,
) -> None:
    """Bar non-franchise goalies from Round 1 and scatter them across Rounds 2-7.

    Only franchise-potential goalies (``_goalie_is_special``) may rank in the top 32; every
    other goalie is demoted to a deterministic per-prospect scatter rank (>=33), no matter
    how strong its raw score is. Franchise goalies rank purely on merit."""
    if not prospects:
        return

    gclass = str(goalie_class_strength or "normal").lower()
    n = len(prospects)

    guard = 0
    while guard < 128:
        guard += 1
        if not preserve_order:
            prospects.sort(key=lambda r: -_effective_draft_score(r))
        moved = False

        for i in range(min(64, len(prospects))):
            row = prospects[i]
            if not _is_goalie_row(row):
                continue
            if _goalie_is_special(row, gclass):
                continue
            rank = i + 1
            target = _goalie_scatter_rank(row, n)
            if rank < target:
                _demote_row_to_min_rank(
                    prospects,
                    i,
                    target,
                    reason="goalie_scattered_non_elite",
                    penalty_floor=55.0,
                )
                row["goalie_rank_reason"] = "goalie_scattered_non_elite"
                moved = True
                break

        if not moved:
            break

    if not preserve_order:
        prospects.sort(key=lambda r: -_effective_draft_score(r))


def compose_live_draft_board(
    prospects: List[Dict[str, Any]],
    *,
    max_entries: int = MAX_BOARD_ENTRIES,
    min_goalies: int = MIN_BOARD_GOALIES,
    target_goalies: int = TARGET_BOARD_GOALIES,
    visibility_inject_from_rank: int = VISIBILITY_INJECT_FROM_RANK,
) -> List[Dict[str, Any]]:
    """Earned ranking first; inject omitted goalies into the board tail for visibility."""
    if not prospects:
        return []

    for row in prospects:
        if not _is_goalie_row(row):
            continue
        if row.get("generational_goalie"):
            row.setdefault("goalie_rank_reason", "generational_goalie_exception")
        elif row.get("goalie_rank_reason") in ("goalie_class_rank_cap", "goalie_visibility_injected"):
            pass
        elif _goalie_is_special(row):
            row.setdefault("goalie_rank_reason", "elite_goalie_exception")
        else:
            row.setdefault("goalie_rank_reason", "goalie_score_earned")

    board = list(prospects[:max_entries])
    board_keys = {str(r.get("key")) for r in board if r.get("key")}

    goalies_on = sum(1 for r in board if _is_goalie_row(r))
    omitted_preview = [
        r for r in prospects
        if _is_goalie_row(r) and str(r.get("key")) not in board_keys
    ]
    need = max(0, min_goalies - goalies_on)
    need = min(len(omitted_preview), max(need, target_goalies - goalies_on))
    if need > 0:
        omitted = omitted_preview
        omitted.sort(key=lambda r: -_effective_draft_score(r))
        tail_start = max(0, int(visibility_inject_from_rank) - 1)

        for goalie in omitted[:need]:
            goalie["goalie_rank_reason"] = "goalie_visibility_injected"
            goalie["_visibility_injected"] = True
            tail_indices = [
                i for i in range(tail_start, len(board))
                if not _is_goalie_row(board[i])
            ]
            if not tail_indices:
                tail_indices = [
                    i for i in range(len(board) - 1, tail_start - 1, -1)
                    if not _is_goalie_row(board[i])
                ]
            if not tail_indices:
                break
            victim_idx = min(tail_indices, key=lambda i: _effective_draft_score(board[i]))
            old_key = str(board[victim_idx].get("key") or "")
            if old_key in board_keys:
                board_keys.discard(old_key)
            board[victim_idx] = goalie
            board_keys.add(str(goalie.get("key") or ""))

    tail_start = max(0, int(visibility_inject_from_rank) - 1)
    if any(r.get("_visibility_injected") for r in board[tail_start:]):
        tail = board[tail_start:]
        tail.sort(key=lambda r: -_effective_draft_score(r))
        board[tail_start:] = tail
    return board[:max_entries]


def collect_goalie_pipeline_stats(
    league: Any,
    *,
    age_max: int = 20,
    pos_fn: Any = None,
) -> Dict[str, int]:
    """Debug counters for goalie loss across the dev-league pipeline."""
    stats = {
        "total_dev_players": 0,
        "total_dev_goalies": 0,
        "draft_eligible_dev_goalies": 0,
        "goalies_with_ratings": 0,
        "goalies_with_potential": 0,
    }
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                stats["total_dev_players"] += 1
                pos = str(pos_fn(p) if pos_fn else getattr(getattr(p, "identity", None), "position", "") or "")
                if pos.upper() != "G":
                    continue
                stats["total_dev_goalies"] += 1
                ident = getattr(p, "identity", None)
                age = int(getattr(ident, "age", 99) or 99) if ident else 99
                if age > age_max:
                    continue
                stats["draft_eligible_dev_goalies"] += 1
                ratings = getattr(p, "ratings", None) or {}
                if ratings:
                    stats["goalies_with_ratings"] += 1
                pot = ratings.get("dev_potential") if isinstance(ratings, dict) else None
                if pot and float(pot) > 0:
                    stats["goalies_with_potential"] += 1
    return stats


def backfill_draft_eligible_goalies(league: Any, rng: Any, needed: int) -> int:
    """Spawn real goalie Player objects into dev leagues when the pool is thin."""
    if needed <= 0 or league is None:
        return 0

    from app.sim_engine.entities.player import Position
    from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _spawn_player

    dev = list(getattr(league, "development_leagues", None) or [])
    if not dev:
        return 0

    league_players = list(getattr(league, "players", None) or [])
    used_names: set = set()
    for p in league_players:
        ident = getattr(p, "identity", None)
        nm = str(getattr(ident, "name", "") or "")
        if nm:
            used_names.add(nm)

    # Collect every (block, team) target across ALL development leagues so goalies
    # are distributed instead of piling onto one junior club.
    targets: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for block in dev:
        for team in (block.get("teams") or []):
            targets.append((block, team))
    if not targets:
        return 0

    def _goalie_count(team: Dict[str, Any]) -> int:
        return sum(
            1 for p in (team.get("players") or [])
            if str(getattr(getattr(p, "identity", None), "position", "")).endswith("G")
        )

    def _tier_and_pot() -> Tuple[str, int]:
        # Normal class-strength distribution — most are depth/mid, few are top.
        roll = int(rng.randint(1, 100))
        if roll <= 6:
            return "top", int(rng.randint(80, 88))
        if roll <= 25:
            return "high", int(rng.randint(72, 80))
        if roll <= 60:
            return "mid", int(rng.randint(64, 73))
        return "depth", int(rng.randint(55, 66))

    created = 0
    for _ in range(needed):
        # Always fill the currently-thinnest team so goalies spread evenly.
        block, target_team = min(targets, key=lambda bt: _goalie_count(bt[1]))
        code = str(block.get("league_code") or "CHL_OHL")
        club = str(target_team.get("name") or "Backfill")
        tid = str(target_team.get("team_id") or "")
        roster = list(target_team.get("players") or [])

        p = _spawn_player(
            rng,
            pos=Position.G,
            ovr_lo=0.34,
            ovr_hi=0.52,
            age_lo=17,
            age_hi=19,
            used_names=used_names,
            league_players=league_players,
            pool_context="junior",
        )
        tier, pot = _tier_and_pot()
        p.ratings["dev_potential"] = pot
        setattr(p, "pipeline_tier", tier)
        p.context.current_team_id = tid
        _set_assignment(p, level="junior", league_code=code, club=club)
        try:
            from app.sim_engine.generation.prospect_league_scoring import initialize_prospect_season

            initialize_prospect_season(p, code, rng=rng)
        except Exception:
            pass
        roster.append(p)
        target_team["players"] = roster
        created += 1

    league.development_leagues = dev
    league.players = league_players
    return created


def calculate_prospect_eta(
    row: Mapping[str, Any],
    *,
    final_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """Canonical ETA helper — used by draft board rows and prospect profiles."""
    rank = int(final_rank if final_rank is not None else row.get("rank") or 99)
    ovr = float(row.get("true_ovr") or 0)
    pos = str(row.get("position") or "").upper()
    pot = _true_pot(row)
    age = int(row.get("age") or 18)
    is_goalie = pos == "G"
    is_transcendent = bool(row.get("is_transcendent") or row.get("transcendent_talent"))
    code = str(row.get("league_code") or "").upper()

    if ovr <= 0:
        return {"label": "Unknown", "years": 4, "confidence": 40.0}

    # Underagers are never NHL-ready no matter how inflated current ability looks.
    if age <= 16:
        return {"label": "3Y", "years": 3, "confidence": 58.0}

    # NHL-readiness is driven by CURRENT ability. Elite draft position and high
    # ceiling raise confidence and modestly shorten timelines, but never declare a
    # raw prospect NHL-ready on rank/potential alone.
    if is_transcendent and ovr >= 68 and not is_goalie and age >= 18:
        return {"label": "Now", "years": 0, "confidence": 90.0}

    if not is_goalie and rank <= 10:
        # True NHL-ready ability can arrive immediately even as a top pick —
        # but not before draft-eligible age 18.
        if ovr >= 76 and age >= 18:
            return {"label": "Now", "years": 0, "confidence": 88.0}
        if ovr >= 70:
            return {"label": "1Y", "years": 1, "confidence": 82.0}
        if ovr >= 64 and pot >= 76:
            return {"label": "1Y", "years": 1, "confidence": 78.0}
        if ovr >= 60:
            return {"label": "2Y", "years": 2, "confidence": 76.0}
        if ovr >= 54:
            return {"label": "3Y", "years": 3, "confidence": 70.0}
        # High pick, but current ability is raw — long-term project.
        return {"label": "4Y+", "years": 4, "confidence": 62.0}

    if not is_goalie and rank <= 32:
        if ovr >= 70:
            return {"label": "1Y", "years": 1, "confidence": 76.0}
        if ovr >= 64:
            return {"label": "2Y", "years": 2, "confidence": 72.0}
        if ovr >= 58:
            return {"label": "3Y", "years": 3, "confidence": 68.0}
        return {"label": "4Y+", "years": 4, "confidence": 62.0}

    readiness = ovr
    if is_goalie:
        readiness -= 6.0
        if row.get("generational_goalie") or pot >= 90:
            readiness += 8.0
        elif pot >= 84:
            readiness += 4.0
        if rank <= 15 and pot >= 88:
            return {"label": "1Y", "years": 1, "confidence": 74.0}
        if rank <= 32 and pot >= 84:
            return {"label": "2Y", "years": 2, "confidence": 70.0}

    if code.startswith("EU_") and "EU_J" not in code:
        readiness += 3.0
    if age >= 20:
        readiness += 2.0
    elif age <= 18 and rank <= 32:
        readiness -= 1.0

    if readiness >= 76 and age >= 18:
        years, label = 0, "Now"
    elif readiness >= 76:
        years, label = 1, "1Y"
    elif readiness >= 70:
        years, label = 1, "1Y"
    elif readiness >= 64:
        years, label = 2, "2Y"
    elif readiness >= 58:
        years, label = 3, "3Y"
    else:
        years, label = 4, "4Y+"

    conf = 55.0
    if rank <= 10:
        conf = 80.0
    elif rank <= 32:
        conf = 70.0
    return {"label": label, "years": years, "confidence": round(conf, 1)}


def apply_goalie_ranking_adjustments(
    prospects: List[Dict[str, Any]],
    *,
    goalie_class_boost: float,
    generational_cut: float,
    goalie_class_strength: str = "normal",
) -> None:
    gclass = str(goalie_class_strength or "normal").lower()
    # generational_cut is the enhanced-draft-SCORE threshold for the generational
    # tier (a score percentile computed by the caller). A goalie whose own draft
    # score clears it qualifies for the top-band exception even without the
    # generational_goalie flag.
    gen_cut = float(generational_cut) if generational_cut else None
    for row in prospects:
        row["_goalie_penalized"] = False
        if not _is_goalie_row(row):
            continue
        pot = _true_pot(row)
        score = float(row.get("_score", 0))
        clears_gen_cut = gen_cut is not None and score >= gen_cut
        if row.get("generational_goalie") or clears_gen_cut:
            row["goalie_rank_reason"] = "generational_goalie_exception"
            row["_score"] = score + min(3.5, max(0.0, goalie_class_boost) + 1.5)
            continue
        if (
            gclass in ("elite", "generational")
            and pot >= 90
            and (row.get("pipeline_tier") in ("elite", "franchise") or row.get("generational_goalie"))
        ):
            row["goalie_rank_reason"] = "elite_goalie_exception"
            row["_score"] = score + min(2.0, max(0.0, goalie_class_boost) + 0.5)
            continue
        if pot >= 86 and gclass == "strong" and goalie_class_boost >= 1.0 and row.get("pipeline_tier") in ("elite", "top"):
            row["_score"] = score + 0.5
            continue
        # Goalie-specific pipeline: mild positional scarcity adjustment only.
        # Strong goalies must be able to land R1–R3 from talent, not artificial demotion.
        ovr = float(row.get("true_ovr") or 0)
        if pot >= 84 or ovr >= 72:
            row["goalie_rank_reason"] = "goalie_score_earned"
            row["_score"] = score + min(1.5, max(0.0, goalie_class_boost) * 0.35)
            continue
        if pot >= 78 or ovr >= 68:
            row["goalie_rank_reason"] = "goalie_score_earned"
            # Tiny scarcity nudge — not a late-round exile.
            row["_score"] = score - max(0.4, 1.2 - max(0.0, goalie_class_boost) * 0.25)
            row["_goalie_penalized"] = True
            continue
        base_penalty = {
            "weak": 2.4,
            "normal": 1.8,
            "strong": 1.2,
            "elite": 0.6,
            "generational": 0.3,
        }.get(gclass, 1.8)
        penalty = base_penalty - min(0.8, max(0.0, goalie_class_boost) * 0.15)
        row["_score"] = score - max(0.2, penalty)
        row["_goalie_penalized"] = True
        row["goalie_rank_reason"] = "goalie_scarcity_nudge"


def pick_transcendent_backstory(rng: Any, *, key: str = "") -> Dict[str, Any]:
    if key and key in ORIGIN_STORY_BY_KEY:
        story = dict(ORIGIN_STORY_BY_KEY[key])
    else:
        idx = int(getattr(rng, "randrange", lambda a, b: 0)(0, len(TRANSCENDENT_BACKSTORIES)))
        story = dict(TRANSCENDENT_BACKSTORIES[idx % len(TRANSCENDENT_BACKSTORIES)])
    block = {
        "key": story["key"],
        "title": story["title"],
        "summary": story["summary"],
        "traits": list(story.get("traits") or []),
        "full_text": story.get("full_text") or story["summary"],
    }
    return {"origin_story": block, **story}


def transcendent_profile_flags() -> Dict[str, Any]:
    return {
        "is_transcendent": True,
        "transcendent_talent": True,
        "aura_tier": "gold",
        "draft_hype_tier": "mythic",
        "tank_target": True,
        "storyline_priority": "legendary",
        "special_fx": {
            "halo": True,
            "screen_shake": True,
            "music": "boss",
            "intensity": 100,
        },
    }


def log_draft_class_audit(prospects: List[Dict[str, Any]], *, label: str = "") -> Dict[str, Any]:
    goalies = [p for p in prospects if str(p.get("position") or "").upper() == "G"]
    top_g = min((i + 1 for i, p in enumerate(prospects) if str(p.get("position") or "").upper() == "G"), default=0)
    transcendent = [p for p in prospects if p.get("is_transcendent")]
    top32_pot = [float(p.get("true_potential_score") or p.get("potential_score") or 0) for p in prospects[:32]]
    short_top10 = [
        p for i, p in enumerate(prospects[:10])
        if int(p.get("height_cm") or 0) in range(170, 176)
        and str(p.get("position") or "").upper() not in ("G", "D", "LD", "RD")
    ]
    audit = {
        "label": label,
        "total": len(prospects),
        "goalie_count": len(goalies),
        "top_goalie_rank": top_g,
        "transcendent_count": len(transcendent),
        "top32_pot_min": min(top32_pot) if top32_pot else 0,
        "top32_pot_avg": round(sum(top32_pot) / max(1, len(top32_pot)), 1),
        "top32_low_pot": [
            {"rank": i + 1, "pot": float(p.get("potential_score") or 0), "name": p.get("name")}
            for i, p in enumerate(prospects[:32])
            if float(p.get("potential_score") or 0) < 65
        ],
        "short_top10": len(short_top10),
    }
    logger.info("DRAFT_AUDIT %s", audit)
    return audit


def compute_tank_pressure_for_team(
    team: Any,
    *,
    transcendent_present: bool,
    owns_own_first: bool = True,
    pick_ownership_reason: str = "owns_own_first",
    owns_protected_first: bool = False,
) -> Dict[str, Any]:
    base = {
        "tank_pressure": 0,
        "tank_mode": "none",
        "owns_own_first": bool(owns_own_first),
        "pick_ownership_reason": str(pick_ownership_reason or "unknown"),
    }
    if not transcendent_present:
        return base

    status = str(getattr(team, "team_status", "") or getattr(team, "strategy", "") or "").lower()
    if "playoff" in status or "contend" in status:
        base["tank_pressure"] = min(28, 12)
        return base

    pressure = 0
    if "rebuild" in status or "tank" in status:
        pressure = 78
    elif "middling" in status or "collapse" in status:
        pressure = 52
    else:
        pressure = 40

    pressure = min(95, pressure + 12)

    if not owns_own_first:
        pressure = min(pressure, 68)
    elif owns_protected_first:
        pressure = min(pressure, 82)

    mode = "none"
    if pressure >= 90 and owns_own_first and not owns_protected_first:
        mode = "hard_tank"
    elif pressure >= 70:
        mode = "seller"
    elif pressure >= 50:
        mode = "soft_sell"
    elif pressure >= 30:
        mode = "soft_sell"

    if mode == "hard_tank" and not owns_own_first:
        mode = "seller"
        pressure = min(pressure, 88)

    base["tank_pressure"] = int(pressure)
    base["tank_mode"] = mode if pressure >= 30 else "none"
    return base


def transcendent_storyline_event(prospect_row: Dict[str, Any]) -> Dict[str, Any]:
    name = str(prospect_row.get("name") or "Unknown")
    origin = prospect_row.get("origin_story") if isinstance(prospect_row.get("origin_story"), dict) else {}
    headlines = [
        "The Tank War Begins",
        "A Once-in-a-Lifetime Prospect Has Arrived",
        f"Scouts Call {name} the Best Kid in a Generation",
        "The Lottery Just Became the Season",
    ]
    h = _stable_int(len(headlines), prospect_row.get("key") or name, prospect_row.get("draft_year") or "")
    return {
        "type": "TRANSCENDENT_DRAFT_PROSPECT",
        "severity": "legendary",
        "priority": "LEGENDARY",
        "prospect_id": str(prospect_row.get("key") or ""),
        "headline": headlines[h],
        "players": [name],
        "origin_story": origin,
        "effects": {
            "tanking_pressure": 95,
            "trade_market_heat": 80,
            "lottery_hype": 100,
        },
    }


def validate_rating_scale_value(value: Any) -> Dict[str, Any]:
    """Flag malformed rating values without rewriting datasets."""
    try:
        from app.sim_engine.entities.player import normalize_rating, display_rating

        n = normalize_rating(value)
        return {"ok": True, "normalized_01": n, "display_99": display_rating(n), "raw": value}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "raw": value}


def validate_prospect_rating_distributions(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic distribution diagnostics on current fixture/board rows (no sim)."""
    flags: List[str] = []
    ovrs: List[float] = []
    expected: List[float] = []
    maximum: List[float] = []
    gaps: List[float] = []
    goalie_n = 0
    for row in entries or []:
        try:
            o = float(row.get("current_ovr_estimate") or row.get("true_ovr") or 0)
            e = float(row.get("expected_ceiling_estimate") or row.get("potential_score") or 0)
            m = float(row.get("maximum_ceiling_estimate") or e)
        except (TypeError, ValueError):
            flags.append(f"malformed_row:{row.get('key')}")
            continue
        if str(row.get("position") or "").upper() == "G":
            goalie_n += 1
        if o > 0:
            ovrs.append(o)
        if e > 0:
            expected.append(e)
        if m > 0:
            maximum.append(m)
        if e > 0 and o > 0:
            gaps.append(e - o)
        if e > 0 and m > 0 and e > m + 0.5:
            flags.append(f"expected_gt_maximum:{row.get('key')}")
        if o > 0 and m > 0 and o > m + 1.0 and not row.get("decline_state"):
            flags.append(f"above_maximum_ceiling:{row.get('key')}")
        if "true_potential_score" in row:
            flags.append(f"true_potential_leak:{row.get('key')}")

    def _avg(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 2) if xs else None

    if ovrs and _avg(ovrs) and float(_avg(ovrs) or 0) > 78:
        flags.append("prospect_ovr_mean_high")
    if expected and _avg(expected) and float(_avg(expected) or 0) > 88:
        flags.append("expected_ceiling_mean_high")
    if gaps and _avg(gaps) and float(_avg(gaps) or 0) > 18:
        flags.append("avg_gap_very_large")

    return {
        "n": len(entries or []),
        "goalie_n": goalie_n,
        "ovr_mean": _avg(ovrs),
        "expected_ceiling_mean": _avg(expected),
        "maximum_ceiling_mean": _avg(maximum),
        "gap_mean": _avg(gaps),
        "flags": flags[:80],
        "ok": len(flags) == 0,
    }


# Any of these keys (or a leading underscore) leaks hidden truth / engine state.
PUBLIC_FORBIDDEN_KEYS = frozenset({
    "true_potential_score",
    "true_potential",
    "true_ovr",
    "dev_potential",
    "true_center_score",
    "true_shot_score",
    "base_score",
    "sanity_penalty",
    "is_transcendent",
    "transcendent_talent",
    "generational_goalie",
    "generational_cut",
    "pipeline_tier",
    "hidden_tier",
    "consensus_seed",
})


def public_entry_omits_true_potential(entry: Mapping[str, Any]) -> bool:
    """True when a public draft-board row exposes NO hidden-truth / internal field.

    A single forbidden-key check is insufficient — a whitelist-style audit rejects
    any leading-underscore scratch field and every known private-truth key.
    """
    for k in dict(entry or {}).keys():
        if isinstance(k, str) and (k.startswith("_") or k in PUBLIC_FORBIDDEN_KEYS):
            return False
    return True
