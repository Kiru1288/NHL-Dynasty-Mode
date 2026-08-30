from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import random


PERSONALITIES = (
    "leader",
    "quiet_professional",
    "glue_guy",
    "high_ego_star",
    "intense_competitor",
    "low_maintenance",
    "mentor",
    "streaky_confidence",
    "young_skilled",
    "veteran_stabilizer",
    "balanced",
)

PLAYSTYLES = (
    "sniper",
    "playmaker",
    "power_forward",
    "two_way",
    "grinder",
    "puck_mover",
    "shutdown",
    "offensive_defenseman",
    "defensive_defenseman",
    "hybrid_goalie",
    "butterfly_goalie",
    "balanced",
)

_DEFAULT_PROFILE = {
    "personality": "balanced",
    "playstyle": "two_way",
    "leadership": 50,
    "ego": 50,
    "work_ethic": 50,
    "coachability": 50,
    "adaptability": 50,
    "temperament": 50,
    "competitiveness": 50,
    "loyalty": 50,
    "defensive_buy_in": 50,
    "pressure_response": 50,
    "room_presence": 50,
}

_ARCHETYPE_PLAYSTYLE_MAP = {
    "playmaker": "playmaker",
    "sniper": "sniper",
    "power_forward": "power_forward",
    "power": "power_forward",
    "two_way": "two_way",
    "two_way_f": "two_way",
    "two_way_d": "two_way",
    "grinder": "grinder",
    "enforcer": "grinder",
    "puck_mover": "puck_mover",
    "shutdown": "shutdown",
    "stay_at_home": "defensive_defenseman",
    "defensive_d": "defensive_defenseman",
    "defensive_defenseman": "defensive_defenseman",
    "offensive_d": "offensive_defenseman",
    "offensive_defenseman": "offensive_defenseman",
    "hybrid_g": "hybrid_goalie",
    "hybrid_goalie": "hybrid_goalie",
    "butterfly_g": "butterfly_goalie",
    "butterfly_goalie": "butterfly_goalie",
}

_PERSONALITY_SYNERGY_POINTS = {
    ("leader", "young_skilled"): 10,
    ("leader", "streaky_confidence"): 8,
    ("glue_guy", "high_ego_star"): 8,
    ("glue_guy", "intense_competitor"): 7,
    ("high_ego_star", "glue_guy"): 8,
    ("high_ego_star", "high_ego_star"): -14,
    ("high_ego_star", "intense_competitor"): -8,
    ("mentor", "young_skilled"): 9,
    ("veteran_stabilizer", "young_skilled"): 8,
    ("young_skilled", "leader"): 10,
    ("young_skilled", "mentor"): 9,
}

_PLAYSTYLE_SYNERGY_POINTS = {
    ("playmaker", "sniper"): 14,
    ("sniper", "playmaker"): 14,
    ("playmaker", "power_forward"): 6,
    ("power_forward", "playmaker"): 6,
    ("sniper", "power_forward"): 5,
    ("power_forward", "sniper"): 5,
    ("puck_mover", "shutdown"): 12,
    ("shutdown", "puck_mover"): 12,
    ("offensive_defenseman", "defensive_defenseman"): 10,
    ("defensive_defenseman", "offensive_defenseman"): 10,
    ("two_way", "sniper"): 4,
    ("two_way", "playmaker"): 4,
    ("sniper", "sniper"): -3,
    ("playmaker", "playmaker"): -2,
    ("offensive_defenseman", "offensive_defenseman"): -4,
    ("puck_mover", "puck_mover"): -3,
}


def _chemistry_profile_is_materialized(raw: Any) -> bool:
    return isinstance(raw, dict) and raw.get("_materialized") is True


def _chemistry_profile_is_stub(player: Any, raw: Any) -> bool:
    if not isinstance(raw, dict) or not raw:
        return True
    if raw.get("_materialized") is not True:
        return True
    current = str(raw.get("playstyle") or "")
    inferred = _infer_playstyle(player, _player_chemistry_rng(player))
    distinctive = {
        "playmaker",
        "sniper",
        "power_forward",
        "grinder",
        "offensive_defenseman",
        "defensive_defenseman",
        "puck_mover",
        "shutdown",
        "hybrid_goalie",
        "butterfly_goalie",
    }
    return inferred in distinctive and current in ("two_way", "balanced", "")


def _player_chemistry_rng(player: Any) -> random.Random:
    seed = _canonical_player_id(player) or _player_name(player)
    return random.Random(seed)


def _personality_synergy_points(personality_a: str, personality_b: str) -> float:
    a = str(personality_a or "balanced")
    b = str(personality_b or "balanced")
    return float(
        _PERSONALITY_SYNERGY_POINTS.get((a, b), 0.0)
        + _PERSONALITY_SYNERGY_POINTS.get((b, a), 0.0)
    )


def _playstyle_synergy_points(playstyle_a: str, playstyle_b: str) -> float:
    a = str(playstyle_a or "balanced")
    b = str(playstyle_b or "balanced")
    return float(
        _PLAYSTYLE_SYNERGY_POINTS.get((a, b), 0.0)
        + _PLAYSTYLE_SYNERGY_POINTS.get((b, a), 0.0)
    )


def materialize_roster_chemistry_profiles(team: Any) -> int:
    """Ensure every roster player has an inferred, persisted chemistry profile."""
    count = 0
    for p in _team_players(team):
        before = _chemistry_profile_is_stub(p, getattr(p, "chemistry_profile", None))
        ensure_player_chemistry_profile(p, _player_chemistry_rng(p))
        if before:
            count += 1
    return count


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else float(value)


def _to_num(v: Any, default: float = 50.0) -> float:
    try:
        n = float(v)
        if n != n:  # nan
            return default
        return n
    except Exception:
        return default


def _canonical_player_id(player: Any) -> str:
    raw = (
        getattr(player, "id", None)
        or getattr(player, "player_id", None)
        or getattr(player, "_ledger_player_id", None)
        or getattr(player, "external_player_id", None)
        or getattr(player, "uid", None)
        or ""
    )
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.isdigit():
        return f"NHL_{s}"
    return s


def _canonical_player_id_from_string(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    if s.isdigit():
        return f"NHL_{s}"
    return s


def _player_id(player: Any) -> str:
    return _canonical_player_id(player)


def _pair_index_key(ida: Any, idb: Any) -> str:
    a = _canonical_player_id_from_string(ida)
    b = _canonical_player_id_from_string(idb)
    if not a or not b:
        return ""
    left, right = sorted([a, b])
    return f"{left}|{right}"


def build_pair_index(pair_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in pair_rows or []:
        if not isinstance(row, dict):
            continue
        key = _pair_index_key(row.get("player_a_id"), row.get("player_b_id"))
        if key:
            out[key] = row
    return out


def _collect_pair_rows_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in list(report.get("lines") or []) + list(report.get("pairs") or []):
        if not isinstance(unit, dict):
            continue
        for row in list(unit.get("pair_links") or []):
            if isinstance(row, dict):
                rows.append(row)
    for row in list(report.get("deployed_pair_links") or []):
        if isinstance(row, dict):
            rows.append(row)
    for row in list(report.get("top_connections") or []):
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    if ident is not None and getattr(ident, "name", None):
        return str(ident.name)
    return str(getattr(player, "name", None) or "Player")


def _player_position(player: Any) -> str:
    ident = getattr(player, "identity", None)
    pos = getattr(ident, "position", None) if ident is not None else getattr(player, "position", None)
    if hasattr(pos, "value"):
        return str(pos.value).upper()
    return str(pos or "").upper()


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    try:
        if ident is not None and getattr(ident, "age", None) is not None:
            return int(ident.age)
        return int(getattr(player, "age", 24) or 24)
    except Exception:
        return 24


def safe_get_psych(player: Any) -> Dict[str, float]:
    psych = getattr(player, "psych", None)
    if psych is None:
        return {"morale": 0.5, "confidence": 0.5, "role_satisfaction": 0.5}

    conf = getattr(psych, "confidence", None)
    if conf is None:
        conf = getattr(psych, "confidence_level", 0.5)

    return {
        "morale": clamp(_to_num(getattr(psych, "morale", 0.5), 0.5), 0.0, 1.0),
        "confidence": clamp(_to_num(conf, 0.5), 0.0, 1.0),
        "role_satisfaction": clamp(_to_num(getattr(psych, "role_satisfaction", 0.5), 0.5), 0.0, 1.0),
    }


def _infer_personality(player: Any, rng: random.Random) -> str:
    age = _player_age(player)
    pos = _player_position(player)
    traits = getattr(player, "traits", None)
    leadership = _to_num(getattr(traits, "leadership", 0.5), 0.5)
    ego = _to_num(getattr(traits, "ego", 0.5), 0.5)
    work_ethic = _to_num(getattr(traits, "work_ethic", 0.5), 0.5)
    volatility = _to_num(getattr(traits, "volatility", 0.5), 0.5)

    if age >= 33 and leadership > 0.58:
        return rng.choice(["mentor", "veteran_stabilizer", "leader"])
    if age <= 21:
        return rng.choice(["young_skilled", "streaky_confidence", "balanced"])
    if ego > 0.7:
        return "high_ego_star"
    if leadership > 0.68 and work_ethic > 0.6:
        return "leader"
    if volatility > 0.62:
        return "intense_competitor"
    if pos == "G":
        return rng.choice(["quiet_professional", "balanced", "low_maintenance"])
    return rng.choice(["glue_guy", "quiet_professional", "balanced", "low_maintenance"])


def _infer_playstyle(player: Any, rng: random.Random) -> str:
    arch = str(
        getattr(player, "archetype", "")
        or getattr(player, "_generated_profile", "")
        or getattr(player, "player_type", "")
        or ""
    ).lower().replace(" ", "_").replace("-", "_")
    pos = _player_position(player)
    if arch in _ARCHETYPE_PLAYSTYLE_MAP:
        return _ARCHETYPE_PLAYSTYLE_MAP[arch]
    if pos == "G":
        if "hybrid" in arch:
            return "hybrid_goalie"
        if "butterfly" in arch:
            return "butterfly_goalie"
        return "balanced"
    if "sniper" in arch:
        return "sniper"
    if "playmaker" in arch:
        return "playmaker"
    if "power" in arch:
        return "power_forward"
    if "grinder" in arch or "enforcer" in arch:
        return "grinder"
    if "offensive_d" in arch or "offensive_defenseman" in arch or arch == "offensive_d":
        return "offensive_defenseman"
    if (
        "defensive_d" in arch
        or "shutdown" in arch
        or "stay_at_home" in arch
        or arch == "defensive_d"
    ):
        return "defensive_defenseman"
    if "two_way" in arch or "two_way_f" in arch:
        return "two_way"
    if pos == "D":
        return rng.choice(["shutdown", "puck_mover", "defensive_defenseman", "two_way"])
    return rng.choice(["two_way", "grinder", "playmaker", "balanced"])


def get_player_chemistry_profile(player: Any) -> Dict[str, Any]:
    prof = getattr(player, "chemistry_profile", None)
    if isinstance(prof, dict):
        out = dict(_DEFAULT_PROFILE)
        out.update(prof)
        for k in _DEFAULT_PROFILE:
            if k in ("personality", "playstyle"):
                continue
            out[k] = int(round(clamp(_to_num(out.get(k), 50.0), 0.0, 100.0)))
        if out.get("personality") not in PERSONALITIES:
            out["personality"] = "balanced"
        if out.get("playstyle") not in PLAYSTYLES:
            out["playstyle"] = "balanced"
        return out
    return dict(_DEFAULT_PROFILE)


def ensure_player_chemistry_profile(player: Any, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    if player is None:
        return dict(_DEFAULT_PROFILE)
    raw_prof = getattr(player, "chemistry_profile", None)
    if not _chemistry_profile_is_stub(player, raw_prof):
        existing = get_player_chemistry_profile(player)
        if getattr(player, "chemistry_relationships", None) is None:
            setattr(player, "chemistry_relationships", {})
        if getattr(player, "chemistry_history", None) is None:
            setattr(player, "chemistry_history", [])
        return existing

    rng = rng if isinstance(rng, random.Random) else _player_chemistry_rng(player)

    traits = getattr(player, "traits", None)
    psych = safe_get_psych(player)

    def s(v: float) -> int:
        return int(round(clamp(v, 0.0, 100.0)))

    leadership = s((_to_num(getattr(traits, "leadership", 0.5), 0.5) * 100.0) + rng.uniform(-8, 8))
    ego = s((_to_num(getattr(traits, "ego", 0.5), 0.5) * 100.0) + rng.uniform(-10, 10))
    work_ethic = s((_to_num(getattr(traits, "work_ethic", 0.5), 0.5) * 100.0) + rng.uniform(-8, 8))
    coachability = s((_to_num(getattr(traits, "coachability", 0.5), 0.5) * 100.0) + rng.uniform(-8, 8))
    adaptability = s((_to_num(getattr(traits, "adaptability", 0.5), 0.5) * 100.0) + rng.uniform(-10, 10))
    competitiveness = s((_to_num(getattr(traits, "competitiveness", 0.5), 0.5) * 100.0) + rng.uniform(-8, 8))
    loyalty = s((_to_num(getattr(traits, "loyalty", 0.5), 0.5) * 100.0) + rng.uniform(-10, 10))
    pressure_response = s((psych["confidence"] * 100.0) + rng.uniform(-10, 10))
    temperament = s((100.0 - _to_num(getattr(traits, "volatility", 0.5), 0.5) * 100.0) + rng.uniform(-10, 10))
    room_presence = s(0.55 * leadership + 0.30 * competitiveness + 0.15 * loyalty + rng.uniform(-7, 7))
    defensive_buy_in = s(0.45 * coachability + 0.35 * work_ethic + 0.20 * temperament + rng.uniform(-8, 8))

    prof = {
        "personality": _infer_personality(player, rng),
        "playstyle": _infer_playstyle(player, rng),
        "leadership": leadership,
        "ego": ego,
        "work_ethic": work_ethic,
        "coachability": coachability,
        "adaptability": adaptability,
        "temperament": temperament,
        "competitiveness": competitiveness,
        "loyalty": loyalty,
        "defensive_buy_in": defensive_buy_in,
        "pressure_response": pressure_response,
        "room_presence": room_presence,
        "_materialized": True,
    }
    setattr(player, "chemistry_profile", prof)
    if getattr(player, "chemistry_relationships", None) is None:
        setattr(player, "chemistry_relationships", {})
    if getattr(player, "chemistry_history", None) is None:
        setattr(player, "chemistry_history", [])
    return prof


def personality_compatibility(player_a: Any, player_b: Any) -> float:
    pa = ensure_player_chemistry_profile(player_a)
    pb = ensure_player_chemistry_profile(player_b)
    comp = 0.58
    combos = {
        ("leader", "young_skilled"): 0.16,
        ("mentor", "young_skilled"): 0.18,
        ("glue_guy", "high_ego_star"): 0.08,
        ("veteran_stabilizer", "streaky_confidence"): 0.10,
    }
    comp += combos.get((pa["personality"], pb["personality"]), 0.0)
    comp += combos.get((pb["personality"], pa["personality"]), 0.0)
    if pa["personality"] == "high_ego_star" and pb["personality"] == "high_ego_star":
        comp -= 0.20
    comp -= abs(pa["ego"] - pb["ego"]) / 220.0
    comp += (min(pa["temperament"], pb["temperament"]) - 50.0) / 260.0
    return clamp(comp, 0.0, 1.0)


def playstyle_compatibility(player_a: Any, player_b: Any) -> float:
    sa = ensure_player_chemistry_profile(player_a).get("playstyle", "balanced")
    sb = ensure_player_chemistry_profile(player_b).get("playstyle", "balanced")
    if sa == sb:
        return 0.62
    bonuses = {
        ("playmaker", "sniper"): 0.22,
        ("sniper", "playmaker"): 0.22,
        ("two_way", "sniper"): 0.10,
        ("two_way", "playmaker"): 0.10,
        ("shutdown", "puck_mover"): 0.20,
        ("defensive_defenseman", "offensive_defenseman"): 0.22,
    }
    penalties = {
        ("offensive_defenseman", "offensive_defenseman"): -0.15,
    }
    base = 0.54 + bonuses.get((sa, sb), 0.0) + penalties.get((sa, sb), 0.0)
    if sa in ("hybrid_goalie", "butterfly_goalie") or sb in ("hybrid_goalie", "butterfly_goalie"):
        base = 0.50
    return clamp(base, 0.0, 1.0)


def position_fit_score(player: Any, slot: str = "") -> float:
    """0–100 positional/slot fit for EV, PP, or PK contexts."""
    pos = _player_position(player)
    slot_u = str(slot or "").upper()
    if not slot_u:
        return 72.0
    if slot_u in ("STARTER", "BACKUP", "THIRD"):
        return 90.0 if pos == "G" else 25.0
    if slot_u in ("LD", "RD", "D1", "D2"):
        if pos in ("D", "LD", "RD"):
            if slot_u == "LD" and pos == "RD":
                return 62.0
            if slot_u == "RD" and pos == "LD":
                return 62.0
            return 88.0
        return 35.0
    if slot_u == "C":
        return 92.0 if pos == "C" else (70.0 if pos in ("LW", "RW", "W", "F") else 30.0)
    if slot_u in ("LW", "RW"):
        if pos == slot_u:
            return 90.0
        if pos in ("C", "LW", "RW", "W", "F"):
            return 74.0
        return 32.0
    if slot_u in ("F1", "F2", "F"):
        return 82.0 if pos in ("C", "LW", "RW", "W", "F") else 40.0
    return 70.0


def coach_system_fit_score(player: Any, team: Any = None) -> float:
    prof = ensure_player_chemistry_profile(player)
    buy_in = _to_num(prof.get("defensive_buy_in", 50), 50.0)
    coachability = _to_num(prof.get("coachability", 50), 50.0)
    adaptability = _to_num(prof.get("adaptability", 50), 50.0)
    psych = safe_get_psych(player)
    coach = getattr(team, "coach", None) if team is not None else None
    style = str(getattr(coach, "system", None) or getattr(coach, "style", None) or "").lower()
    playstyle = str(prof.get("playstyle", "balanced"))
    bonus = 0.0
    if "defense" in style or "shutdown" in style:
        bonus += 6.0 if playstyle in ("shutdown", "two_way", "defensive_defenseman", "grinder") else -3.0
    elif "offense" in style or "attack" in style:
        bonus += 6.0 if playstyle in ("sniper", "playmaker", "offensive_defenseman", "puck_mover") else -2.0
    score = 0.34 * buy_in + 0.28 * coachability + 0.22 * adaptability + 0.16 * (psych["role_satisfaction"] * 100.0) + bonus
    return clamp(score, 0.0, 100.0)


def usage_satisfaction_score(player: Any) -> float:
    psych = safe_get_psych(player)
    role = psych["role_satisfaction"] * 100.0
    conf = psych["confidence"] * 100.0
    morale = psych["morale"] * 100.0
    deployed = getattr(player, "_gm_game_line_idx", None)
    if deployed is None:
        deployed = getattr(player, "_deployed_line_rank", None)
    expect = 1.5
    try:
        ovr_f = getattr(player, "ovr", None)
        ov = float(ovr_f() if callable(ovr_f) else (ovr_f or 0.7))
        if ov > 1.5:
            ov = ov / 99.0
        expect = 0.6 if ov >= 0.86 else 1.4 if ov >= 0.80 else 2.2 if ov >= 0.74 else 3.0
    except Exception:
        pass
    if deployed is not None:
        gap = abs(float(deployed) - float(expect))
        role = clamp(role - gap * 8.0, 0.0, 100.0)
    return clamp(0.45 * role + 0.30 * morale + 0.25 * conf, 0.0, 100.0)


def calculate_pair_chemistry(player_a: Any, player_b: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = context or {}
    pa = ensure_player_chemistry_profile(player_a, context.get("rng"))
    pb = ensure_player_chemistry_profile(player_b, context.get("rng"))
    psych_a = safe_get_psych(player_a)
    psych_b = safe_get_psych(player_b)
    rel_a = dict(getattr(player_a, "chemistry_relationships", None) or {})
    rel_b = dict(getattr(player_b, "chemistry_relationships", None) or {})
    ida = _player_id(player_a)
    idb = _player_id(player_b)
    familiarity = 0.0
    fam_count = 0
    if idb in rel_a:
        familiarity += _to_num(rel_a.get(idb), 0.0)
        fam_count += 1
    if ida in rel_b:
        familiarity += _to_num(rel_b.get(ida), 0.0)
        fam_count += 1
    familiarity = familiarity / fam_count if fam_count else 50.0

    p_comp = personality_compatibility(player_a, player_b)
    s_comp = playstyle_compatibility(player_a, player_b)
    mood = (psych_a["morale"] + psych_b["morale"] + psych_a["confidence"] + psych_b["confidence"]) / 4.0
    tension_pen = max(0.0, (pa["ego"] + pb["ego"] - 120.0) / 220.0)
    team = context.get("team")
    pos_fit = (
        position_fit_score(player_a, str(context.get("slot_a") or ""))
        + position_fit_score(player_b, str(context.get("slot_b") or ""))
    ) / 2.0
    coach_fit = (coach_system_fit_score(player_a, team) + coach_system_fit_score(player_b, team)) / 2.0
    usage = (usage_satisfaction_score(player_a) + usage_satisfaction_score(player_b)) / 2.0
    role_balance = clamp(100.0 - abs(_to_num(pa.get("ego"), 50) - _to_num(pb.get("ego"), 50)) * 0.55, 0.0, 100.0)
    morale = mood * 100.0

    score01 = (
        0.22 * p_comp
        + 0.20 * s_comp
        + 0.12 * (mood)
        + 0.12 * (familiarity / 100.0)
        + 0.12 * (pos_fit / 100.0)
        + 0.10 * (coach_fit / 100.0)
        + 0.07 * (usage / 100.0)
        + 0.05 * (role_balance / 100.0)
        - 0.12 * tension_pen
    )
    synergy_bonus = _playstyle_synergy_points(pa.get("playstyle"), pb.get("playstyle"))
    synergy_bonus += _personality_synergy_points(pa.get("personality"), pb.get("personality"))
    if _to_num(pa.get("ego"), 50.0) > 72.0 and _to_num(pb.get("ego"), 50.0) > 72.0:
        synergy_bonus -= 12.0
    if familiarity >= 80.0:
        synergy_bonus += min(10.0, (familiarity - 75.0) * 0.45)
    elif familiarity <= 35.0:
        synergy_bonus -= min(12.0, (40.0 - familiarity) * 0.5)
    score = int(round(clamp(score01 * 100.0 + synergy_bonus, 0.0, 100.0)))
    scheme = {
        "position_fit": int(round(pos_fit)),
        "linemate_compatibility": int(round(clamp(0.5 * p_comp + 0.5 * s_comp, 0.0, 1.0) * 100.0)),
        "role_balance": int(round(role_balance)),
        "coach_system_fit": int(round(coach_fit)),
        "familiarity": int(round(clamp(familiarity, 0, 100))),
        "morale": int(round(clamp(morale, 0, 100))),
        "usage_satisfaction": int(round(usage)),
    }
    return {
        "player_a_id": ida,
        "player_b_id": idb,
        "player_a_name": _player_name(player_a),
        "player_b_name": _player_name(player_b),
        "chemistry": score,
        "label": chemistry_label(score),
        "familiarity": scheme["familiarity"],
        "scheme_fit": scheme,
    }


def chemistry_label(score: float) -> str:
    s = clamp(_to_num(score, 50.0), 0.0, 100.0)
    if s < 30:
        return "Broken"
    if s < 45:
        return "Awkward"
    if s < 60:
        return "Neutral"
    if s < 75:
        return "Connected"
    if s < 90:
        return "Strong"
    return "Elite"


def chemistry_trend_label(delta: float) -> str:
    d = _to_num(delta, 0.0)
    if d > 1.5:
        return "↑ improving"
    if d < -1.5:
        return "↓ slipping"
    return "→ stable"


def _line_identity(players: List[Any], score: int) -> Tuple[str, str]:
    styles = [ensure_player_chemistry_profile(p).get("playstyle", "balanced") for p in players]
    if "playmaker" in styles and "sniper" in styles:
        ident = "Creative line with finishing support"
    elif "shutdown" in styles or "defensive_defenseman" in styles:
        ident = "Defensive-first reliability unit"
    elif "power_forward" in styles:
        ident = "Heavy forecheck and net-front pressure"
    else:
        ident = "Balanced transition group"
    risk = "Role frustration can disrupt flow" if score < 52 else "Can be exposed if confidence drops" if score < 68 else "Stable fit with manageable volatility"
    return ident, risk


def calculate_forward_line_chemistry(players: List[Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = dict(context or {})
    line = [p for p in (players or []) if p is not None][:3]
    if len(line) < 2:
        empty_scheme = {
            "position_fit": 50,
            "linemate_compatibility": 50,
            "role_balance": 50,
            "coach_system_fit": 50,
            "familiarity": 50,
            "morale": 50,
            "usage_satisfaction": 50,
        }
        return {
            "chemistry": 50,
            "label": chemistry_label(50),
            "players": [],
            "factors": [],
            "concerns": [],
            "trend": 0,
            "scheme_fit": empty_scheme,
        }
    slots = ["LW", "C", "RW"]
    pairs: List[Dict[str, Any]] = []
    for i in range(len(line)):
        for j in range(i + 1, len(line)):
            pairs.append(
                calculate_pair_chemistry(
                    line[i],
                    line[j],
                    context={
                        **context,
                        "slot_a": slots[i] if i < len(slots) else "",
                        "slot_b": slots[j] if j < len(slots) else "",
                    },
                )
            )
    base = sum(p["chemistry"] for p in pairs) / max(1, len(pairs))
    styles = [ensure_player_chemistry_profile(p).get("playstyle", "balanced") for p in line]
    personalities = [ensure_player_chemistry_profile(p).get("personality", "balanced") for p in line]
    if "playmaker" in styles and "sniper" in styles:
        base += 4.0
    if personalities.count("high_ego_star") >= 2:
        base -= 4.0
    defensive_buy_in = sum(ensure_player_chemistry_profile(p).get("defensive_buy_in", 50) for p in line) / len(line)
    if defensive_buy_in < 44:
        base -= 3.0
    score = int(round(clamp(base, 0.0, 100.0)))
    ident, risk = _line_identity(line, score)
    factors = []
    concerns = []
    if "playmaker" in styles and "sniper" in styles:
        factors.append("Strong playmaker/sniper fit")
    if defensive_buy_in >= 62:
        factors.append("Good defensive buy-in")
    if score >= 74:
        factors.append("Confidence trending upward")
    if defensive_buy_in < 44:
        concerns.append("Low defensive buy-in")
    if personalities.count("high_ego_star") >= 2:
        concerns.append("Multiple high-ego personalities")
    if score < 50:
        concerns.append("Role frustration risk")
    scheme_keys = (
        "position_fit",
        "linemate_compatibility",
        "role_balance",
        "coach_system_fit",
        "familiarity",
        "morale",
        "usage_satisfaction",
    )
    scheme = {k: int(round(sum(p.get("scheme_fit", {}).get(k, 50) for p in pairs) / max(1, len(pairs)))) for k in scheme_keys}
    return {
        "chemistry": score,
        "label": chemistry_label(score),
        "identity": ident,
        "risk": risk,
        "trend": int(round(score - 55)),
        "players": [{"id": _player_id(p), "name": _player_name(p), "position": _player_position(p)} for p in line],
        "pair_links": pairs,
        "factors": factors,
        "concerns": concerns,
        "scheme_fit": scheme,
    }


def calculate_defense_pair_chemistry(players: List[Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = dict(context or {})
    pair = [p for p in (players or []) if p is not None][:2]
    if len(pair) < 2:
        return {
            "chemistry": 50,
            "label": chemistry_label(50),
            "players": [],
            "factors": [],
            "concerns": [],
            "trend": 0,
            "scheme_fit": {
                "position_fit": 50,
                "linemate_compatibility": 50,
                "role_balance": 50,
                "coach_system_fit": 50,
                "familiarity": 50,
                "morale": 50,
                "usage_satisfaction": 50,
            },
        }
    p = calculate_pair_chemistry(
        pair[0],
        pair[1],
        context={**context, "slot_a": "LD", "slot_b": "RD"},
    )
    sa = ensure_player_chemistry_profile(pair[0]).get("playstyle", "balanced")
    sb = ensure_player_chemistry_profile(pair[1]).get("playstyle", "balanced")
    score = float(p["chemistry"])
    if {sa, sb} == {"shutdown", "puck_mover"} or {sa, sb} == {"defensive_defenseman", "offensive_defenseman"}:
        score += 6.0
    if sa == sb and sa in ("offensive_defenseman", "puck_mover"):
        score -= 5.0
    score = int(round(clamp(score, 0.0, 100.0)))
    factors = []
    concerns = []
    if score >= 70:
        factors.append("Strong pair role balance")
    if score < 50:
        concerns.append("Coverage communication risk")
    return {
        "chemistry": score,
        "label": chemistry_label(score),
        "identity": "Blue-line complement pair" if score >= 60 else "Mismatch pairing",
        "risk": "May struggle under forecheck pressure" if score < 60 else "Mostly stable under pressure",
        "trend": int(round(score - 55)),
        "players": [{"id": _player_id(x), "name": _player_name(x), "position": _player_position(x)} for x in pair],
        "pair_links": [{**p, "chemistry": score}],
        "factors": factors,
        "concerns": concerns,
        "scheme_fit": dict(p.get("scheme_fit") or {}),
    }


def calculate_goalie_room_fit(goalie: Any, team: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    gp = ensure_player_chemistry_profile(goalie, (context or {}).get("rng"))
    psych = safe_get_psych(goalie)
    room = calculate_team_room_chemistry(team, session=(context or {}).get("session"))
    score = (
        0.34 * gp.get("pressure_response", 50)
        + 0.22 * gp.get("temperament", 50)
        + 0.18 * gp.get("defensive_buy_in", 50)
        + 0.14 * (psych["confidence"] * 100.0)
        + 0.12 * room.get("buy_in", 50)
    )
    score = int(round(clamp(score, 0.0, 100.0)))
    return {
        "player_id": _player_id(goalie),
        "name": _player_name(goalie),
        "chemistry": score,
        "label": chemistry_label(score),
        "confidence": int(round(psych["confidence"] * 100)),
        "pressure_response": int(gp.get("pressure_response", 50)),
    }


def _team_players(team: Any) -> List[Any]:
    return [p for p in (getattr(team, "roster", None) or []) if p is not None and not getattr(p, "retired", False)]


def calculate_team_room_chemistry(team: Any, session: Optional[Any] = None) -> Dict[str, Any]:
    players = _team_players(team)
    if not players:
        return {
            "overall": 50, "morale": 50, "confidence": 50, "role_satisfaction": 50,
            "leadership": 50, "tension": 25, "buy_in": 50, "coach_trust": 50,
            "chaos_resistance": 50, "label": chemistry_label(50),
        }
    rng = getattr(getattr(session, "sim", None), "rng", None) if session is not None else None
    profs = [ensure_player_chemistry_profile(p, rng) for p in players]
    psychs = [safe_get_psych(p) for p in players]
    morale = sum(x["morale"] for x in psychs) / len(psychs) * 100.0
    confidence = sum(x["confidence"] for x in psychs) / len(psychs) * 100.0
    role_sat = sum(x["role_satisfaction"] for x in psychs) / len(psychs) * 100.0
    leadership = sum(p["leadership"] * 0.6 + p["room_presence"] * 0.4 for p in profs) / len(profs)
    ego = sum(p["ego"] for p in profs) / len(profs)
    work_ethic = sum(p["work_ethic"] for p in profs) / len(profs)
    coachability = sum(p["coachability"] for p in profs) / len(profs)
    temperament = sum(p["temperament"] for p in profs) / len(profs)
    loyalty = sum(p["loyalty"] for p in profs) / len(profs)
    pressure = sum(p["pressure_response"] for p in profs) / len(profs)
    veterans = sum(1 for p in players if _player_age(p) >= 30)
    veteran_ratio = veterans / max(1, len(players))
    chaos_index = _to_num(getattr(session, "chaos_index", 0.35), 0.35) if session is not None else 0.35

    tension = 0.32 * ego + 0.22 * (100.0 - role_sat) + 0.24 * (chaos_index * 100.0) + 0.22 * (100.0 - morale)
    buy_in = 0.35 * work_ethic + 0.30 * coachability + 0.18 * morale + 0.17 * role_sat - 0.12 * max(0.0, ego - 58.0)
    coach_trust = 0.45 * coachability + 0.30 * loyalty + 0.25 * confidence
    chaos_res = 0.40 * leadership + 0.24 * temperament + 0.18 * pressure + 0.18 * (veteran_ratio * 100.0) - 0.12 * tension
    overall = (
        0.15 * morale
        + 0.13 * confidence
        + 0.12 * role_sat
        + 0.14 * leadership
        + 0.14 * buy_in
        + 0.12 * coach_trust
        + 0.12 * chaos_res
        - 0.08 * tension
    )
    out = {
        "overall": int(round(clamp(overall, 0, 100))),
        "morale": int(round(clamp(morale, 0, 100))),
        "confidence": int(round(clamp(confidence, 0, 100))),
        "role_satisfaction": int(round(clamp(role_sat, 0, 100))),
        "leadership": int(round(clamp(leadership, 0, 100))),
        "tension": int(round(clamp(tension, 0, 100))),
        "buy_in": int(round(clamp(buy_in, 0, 100))),
        "coach_trust": int(round(clamp(coach_trust, 0, 100))),
        "chaos_resistance": int(round(clamp(chaos_res, 0, 100))),
    }
    out["label"] = chemistry_label(out["overall"])
    return out


def _project_lines_from_roster(team: Any) -> Dict[str, List[List[Any]]]:
    players = _team_players(team)
    f = [p for p in players if _player_position(p) in ("C", "LW", "RW", "W", "F")]
    d = [p for p in players if _player_position(p) in ("D", "LD", "RD")]
    g = [p for p in players if _player_position(p) == "G"]
    f_lines = [f[i:i + 3] for i in range(0, min(len(f), 12), 3)]
    d_pairs = [d[i:i + 2] for i in range(0, min(len(d), 6), 2)]
    return {"forward_lines": f_lines, "defense_pairs": d_pairs, "goalies": g[:2]}


def _session_even_strength_payload(session: Optional[Any]) -> Optional[Dict[str, Any]]:
    if session is None:
        return None
    lines_root = getattr(session, "lines", None)
    if not isinstance(lines_root, dict):
        return None
    even = lines_root.get("even_strength")
    if not isinstance(even, dict):
        return None
    payload = even.get("lines") if isinstance(even.get("lines"), dict) else even
    if isinstance(payload, dict) and (payload.get("forwards") or payload.get("defense") or payload.get("goalies")):
        return payload
    return None


def _resolve_line_units_from_payload(team: Any, payload: Dict[str, Any]) -> Dict[str, List[List[Any]]]:
    by_id: Dict[str, Any] = {}
    for p in _team_players(team):
        cid = _canonical_player_id(p)
        if not cid:
            continue
        by_id[cid] = p
        bare = cid[4:] if cid.startswith("NHL_") else cid
        if bare:
            by_id.setdefault(bare, p)

    def _lookup_player(pid: Any) -> Any:
        key = str(pid or "").strip()
        if not key:
            return None
        if key in by_id:
            return by_id[key]
        canon = _canonical_player_id_from_string(key)
        return by_id.get(canon)

    forward_lines: List[List[Any]] = []
    for line in list(payload.get("forwards") or [])[:4]:
        if not isinstance(line, dict):
            continue
        slots = line.get("slots") or {}
        trio = []
        for slot in ("LW", "C", "RW"):
            p = _lookup_player(slots.get(slot))
            if p is not None:
                trio.append(p)
        if trio:
            forward_lines.append(trio)
    defense_pairs: List[List[Any]] = []
    for pair in list(payload.get("defense") or [])[:3]:
        if not isinstance(pair, dict):
            continue
        slots = pair.get("slots") or {}
        duo = []
        for slot in ("LD", "RD"):
            p = _lookup_player(slots.get(slot))
            if p is not None:
                duo.append(p)
        if duo:
            defense_pairs.append(duo)
    goalies: List[Any] = []
    for gline in list(payload.get("goalies") or [])[:1]:
        if not isinstance(gline, dict):
            continue
        slots = gline.get("slots") or {}
        for key in ("Starter", "Backup", "Third"):
            p = _lookup_player(slots.get(key))
            if p is not None and p not in goalies:
                goalies.append(p)
    if not goalies:
        goalies = [p for p in _team_players(team) if _player_position(p) == "G"][:2]
    return {"forward_lines": forward_lines, "defense_pairs": defense_pairs, "goalies": goalies[:2]}


def _project_lines_for_team(team: Any, session: Optional[Any] = None) -> Dict[str, List[List[Any]]]:
    """Prefer user session.lines; fall back to roster-order projection."""
    payload = _session_even_strength_payload(session)
    if payload:
        resolved = _resolve_line_units_from_payload(team, payload)
        if resolved["forward_lines"] or resolved["defense_pairs"]:
            return resolved
    return _project_lines_from_roster(team)


def calculate_team_chemistry_report(team: Any, session: Optional[Any] = None) -> Dict[str, Any]:
    room = calculate_team_room_chemistry(team, session=session)
    lines_src = _project_lines_for_team(team, session=session)
    ctx = {"session": session, "rng": getattr(getattr(session, "sim", None), "rng", None), "team": team}
    f_reports = []
    for i, line in enumerate(lines_src["forward_lines"], start=1):
        r = calculate_forward_line_chemistry(line, context=ctx)
        r["slot"] = f"F{i}"
        r["type"] = "forward"
        r["source"] = "session.lines" if _session_even_strength_payload(session) else "roster_projection"
        f_reports.append(r)
    d_reports = []
    for i, pair in enumerate(lines_src["defense_pairs"], start=1):
        r = calculate_defense_pair_chemistry(pair, context=ctx)
        r["slot"] = f"D{i}"
        r["type"] = "defense"
        r["source"] = "session.lines" if _session_even_strength_payload(session) else "roster_projection"
        d_reports.append(r)
    goalies = [
        calculate_goalie_room_fit(g, team, context=ctx)
        for g in lines_src["goalies"]
    ]

    top_connections: List[Dict[str, Any]] = []
    # Prefer connections among deployed lines when available.
    deployed_players: List[Any] = []
    for group in lines_src["forward_lines"] + lines_src["defense_pairs"]:
        for p in group:
            if p not in deployed_players:
                deployed_players.append(p)
    roster = deployed_players[:18] or _team_players(team)[:18]
    for i in range(len(roster)):
        for j in range(i + 1, len(roster)):
            top_connections.append(calculate_pair_chemistry(roster[i], roster[j], context=ctx))
    top_connections.sort(key=lambda x: -int(x.get("chemistry", 0)))
    top_connections = top_connections[:24]

    deployed_pair_links: List[Dict[str, Any]] = []
    fwd_slots = ["LW", "C", "RW"]
    for group in lines_src["forward_lines"]:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                deployed_pair_links.append(
                    calculate_pair_chemistry(
                        group[i],
                        group[j],
                        context={
                            **ctx,
                            "slot_a": fwd_slots[i] if i < len(fwd_slots) else "",
                            "slot_b": fwd_slots[j] if j < len(fwd_slots) else "",
                        },
                    )
                )
    for group in lines_src["defense_pairs"]:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                deployed_pair_links.append(
                    calculate_pair_chemistry(
                        group[i],
                        group[j],
                        context={**ctx, "slot_a": "LD", "slot_b": "RD"},
                    )
                )

    concerns: List[str] = []
    if room["tension"] >= 56:
        concerns.append("Room tension is elevated.")
    if room["role_satisfaction"] < 50:
        concerns.append("Role satisfaction is slipping.")
    if room["leadership"] < 50:
        concerns.append("Leadership group lacks stability.")
    if room["buy_in"] < 52:
        concerns.append("System buy-in is trending low.")
    if not _session_even_strength_payload(session):
        concerns.append("No saved even-strength lines — chemistry projected from roster order.")

    report = {
        "room": room,
        "lines": f_reports,
        "pairs": d_reports,
        "goalies": goalies,
        "top_connections": top_connections,
        "deployed_pair_links": deployed_pair_links,
        "concerns": concerns,
        "line_source": "session.lines" if _session_even_strength_payload(session) else "roster_projection",
    }
    report["pair_index"] = build_pair_index(_collect_pair_rows_from_report(report))
    return report


def apply_daily_chemistry_tick(team: Any, session: Optional[Any] = None, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    rng = rng if isinstance(rng, random.Random) else random.Random()
    roster = _team_players(team)
    if not roster:
        return {"updated": 0}
    room_before = calculate_team_room_chemistry(team, session=session)
    for p in roster:
        ensure_player_chemistry_profile(p, rng)
        rel = dict(getattr(p, "chemistry_relationships", None) or {})
        setattr(p, "chemistry_relationships", rel)
    # familiarity tick among likely line neighbors (user lines when available)
    projected = _project_lines_for_team(team, session=session)
    updated = 0
    for group in projected["forward_lines"] + projected["defense_pairs"]:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ida, idb = _player_id(a), _player_id(b)
                if not ida or not idb:
                    continue
                ra = dict(getattr(a, "chemistry_relationships", None) or {})
                rb = dict(getattr(b, "chemistry_relationships", None) or {})
                ra[idb] = clamp(_to_num(ra.get(idb), 50.0) + rng.uniform(0.08, 0.55), 0.0, 100.0)
                rb[ida] = clamp(_to_num(rb.get(ida), 50.0) + rng.uniform(0.08, 0.55), 0.0, 100.0)
                setattr(a, "chemistry_relationships", ra)
                setattr(b, "chemistry_relationships", rb)
                updated += 2

    chaos = _to_num(getattr(session, "chaos_index", 0.35), 0.35) if session is not None else 0.35
    pressure = max(0.0, chaos - (room_before["chaos_resistance"] / 100.0))
    for p in roster:
        psych = getattr(p, "psych", None)
        if psych is None:
            continue
        try:
            m = _to_num(getattr(psych, "morale", 0.5), 0.5)
            c = _to_num(getattr(psych, "confidence", _to_num(getattr(psych, "confidence_level", 0.5), 0.5)), 0.5)
            rs = _to_num(getattr(psych, "role_satisfaction", 0.5), 0.5)
            leader_mod = ensure_player_chemistry_profile(p, rng).get("leadership", 50) / 100.0
            scratched = bool(getattr(p, "_recently_scratched", False))
            if scratched:
                m = clamp(m - 0.008, 0.0, 1.0)
                rs = clamp(rs - 0.01, 0.0, 1.0)
            m = clamp(m - pressure * 0.0025 + leader_mod * 0.0007, 0.0, 1.0)
            c = clamp(c - pressure * 0.0020 + leader_mod * 0.0005, 0.0, 1.0)
            rs = clamp(rs - pressure * 0.0018 + leader_mod * 0.0004, 0.0, 1.0)
            setattr(psych, "morale", m)
            if hasattr(psych, "confidence"):
                setattr(psych, "confidence", c)
            else:
                setattr(psych, "confidence_level", c)
            setattr(psych, "role_satisfaction", rs)
        except Exception:
            continue

    room_after = calculate_team_room_chemistry(team, session=session)
    setattr(team, "_chemistry_cache", room_after)
    # Keep world chemistry as a derived team-level summary of room pulse.
    try:
        setattr(team, "_world_chemistry", clamp(room_after.get("overall", 50) / 100.0, 0.08, 0.96))
    except Exception:
        pass
    return {
        "updated": updated,
        "room_delta": round(room_after["overall"] - room_before["overall"], 2),
        "line_source": "session.lines" if _session_even_strength_payload(session) else "roster_projection",
    }


def apply_storyline_chemistry_effect(session: Any, decision: Dict[str, Any], choice: Dict[str, Any], target: Any = None) -> Dict[str, Any]:
    effects = dict((choice or {}).get("effects") or {})
    if not effects:
        return {"chemistry_applied": False}
    team = None
    try:
        team = session.team_by_id.get(str(getattr(session, "user_team_id", "")))
    except Exception:
        team = None
    out: Dict[str, Any] = {"chemistry_applied": False}

    room_delta = _to_num(effects.get("room_delta", effects.get("chemistry_delta", effects.get("chemistry", 0))), 0.0) / 100.0
    tension_delta = _to_num(effects.get("room_tension_delta", 0), 0.0) / 100.0
    trust_delta = _to_num(effects.get("trust_delta", effects.get("respect_delta", 0)), 0.0) / 100.0
    familiarity_delta = _to_num(effects.get("familiarity_delta", effects.get("line_chemistry_delta", 0)), 0.0)
    coach_trust_delta = _to_num(effects.get("coach_trust_delta", 0), 0.0) / 100.0
    buy_in_delta = _to_num(effects.get("buy_in_delta", 0), 0.0) / 100.0
    leadership_delta = _to_num(effects.get("leadership_delta", 0), 0.0)

    if target is not None:
        prof = ensure_player_chemistry_profile(target)
        prof["leadership"] = int(round(clamp(_to_num(prof.get("leadership"), 50.0) + leadership_delta, 0.0, 100.0)))
        prof["coachability"] = int(round(clamp(_to_num(prof.get("coachability"), 50.0) + buy_in_delta * 100.0, 0.0, 100.0)))
        prof["room_presence"] = int(round(clamp(_to_num(prof.get("room_presence"), 50.0) + trust_delta * 100.0, 0.0, 100.0)))
        setattr(target, "chemistry_profile", prof)
        out["target_player_affected"] = 1
        out["chemistry_applied"] = True

    if team is not None and any(abs(x) > 0 for x in (room_delta, tension_delta, trust_delta, coach_trust_delta, buy_in_delta)):
        for p in _team_players(team):
            psych = getattr(p, "psych", None)
            if psych is None:
                continue
            try:
                setattr(psych, "morale", clamp(_to_num(getattr(psych, "morale", 0.5), 0.5) + room_delta * 0.35 - tension_delta * 0.4, 0.0, 1.0))
                cval = _to_num(getattr(psych, "confidence", _to_num(getattr(psych, "confidence_level", 0.5), 0.5)), 0.5)
                cval = clamp(cval + trust_delta * 0.32, 0.0, 1.0)
                if hasattr(psych, "confidence"):
                    setattr(psych, "confidence", cval)
                else:
                    setattr(psych, "confidence_level", cval)
                setattr(psych, "role_satisfaction", clamp(_to_num(getattr(psych, "role_satisfaction", 0.5), 0.5) + buy_in_delta * 0.28, 0.0, 1.0))
                if hasattr(psych, "coach_trust"):
                    setattr(psych, "coach_trust", clamp(_to_num(getattr(psych, "coach_trust", 0.5), 0.5) + coach_trust_delta * 0.35, 0.0, 1.0))
            except Exception:
                continue
        out["chemistry_applied"] = True
        out["room_tension_delta"] = round(tension_delta, 4)
        out["room_delta"] = round(room_delta, 4)

    if target is not None and familiarity_delta:
        rel = dict(getattr(target, "chemistry_relationships", None) or {})
        rel["_room"] = clamp(_to_num(rel.get("_room"), 50.0) + familiarity_delta, 0.0, 100.0)
        setattr(target, "chemistry_relationships", rel)
        out["familiarity_delta"] = round(familiarity_delta, 2)
        out["chemistry_applied"] = True

    return out


def build_public_chemistry_report(session: Any) -> Dict[str, Any]:
    user_team_id = str(getattr(session, "user_team_id", "") or "")
    team = session.team_by_id.get(user_team_id) if getattr(session, "team_by_id", None) else None
    if team is None:
        return {
            "ok": True,
            "team_id": user_team_id,
            "team_name": "Unknown Team",
            "as_of_date": "",
            "chaos_index": float(_to_num(getattr(session, "chaos_index", 0.35), 0.35)),
            "room": calculate_team_room_chemistry(None, session=None),
            "lines": [],
            "pairs": [],
            "goalies": [],
            "top_connections": [],
            "pair_index": {},
            "concerns": ["Chemistry report unavailable until team roster initializes."],
            "storyline_pressure": [],
        }
    materialized = materialize_roster_chemistry_profiles(team)
    if materialized and session is not None:
        try:
            session._cached_state_roster_rows = None
            session._cached_chemistry_report = None
        except Exception:
            pass
    rep = calculate_team_chemistry_report(team, session=session)
    storylines = list(getattr(session, "storyline_events", None) or [])[-40:]
    pressure = []
    for ev in reversed(storylines[-8:]):
        if not isinstance(ev, dict):
            continue
        tone = str(ev.get("tone") or ev.get("type") or "").lower()
        txt = str(ev.get("headline") or ev.get("details") or ev.get("summary") or "").strip()
        if not txt:
            continue
        if any(x in tone for x in ("wacky", "injury", "trade", "storyline")):
            pressure.append({"text": txt, "impact": "pressure"})
        else:
            pressure.append({"text": txt, "impact": "stability"})
    room = rep["room"]
    if room.get("overall", 50) >= 72:
        narrative = "The room is stable with good buy-in and leadership carry."
    elif room.get("tension", 20) >= 55:
        narrative = "Room tension is building and could hurt consistency."
    else:
        narrative = "Chemistry is mixed; line-level fit will matter."

    return {
        "ok": True,
        "team_id": user_team_id,
        "team_name": f"{getattr(team, 'city', '')} {getattr(team, 'name', '')}".strip() or str(user_team_id),
        "as_of_date": str((getattr(session, "nhl_today", None) or {}).get("iso") or ""),
        "chaos_index": round(float(_to_num(getattr(session, "chaos_index", 0.35), 0.35)), 3),
        "room": room,
        "lines": rep["lines"],
        "pairs": rep["pairs"],
        "goalies": rep["goalies"],
        "top_connections": rep["top_connections"],
        "deployed_pair_links": rep.get("deployed_pair_links") or [],
        "pair_index": rep.get("pair_index") or {},
        "concerns": rep["concerns"],
        "storyline_pressure": pressure,
        "narrative": narrative,
        "line_source": rep.get("line_source") or "roster_projection",
    }

