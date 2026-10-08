"""NHL Scouting Combine engine: attribute-driven testing, medicals and interviews.

Every number this module produces is derived from the prospect's own sim state:

* Measurements: ``identity.height_cm`` / ``identity.weight_kg`` read with the
  small error of a stadiometer and scale; wingspan anchored on that height.
* Fitness tests (standing long jump, VO2 max, Wingate, pro agility, bench, grip,
  pull-ups, Y-balance): a weighted composite of the ratings that drive the
  quality being tested (explosiveness, endurance, strength, agility, balance),
  plus physical maturity (age) and body mass where it matters. The composite is
  standardized within the class and measured with test-specific noise
  (validity < 1), then expressed in the real units and ranges the NHL combine
  publishes.
* Medicals: the live prospect injury state, the ``HealthState`` record and the
  durability ratings.
* Interviews: personality traits plus character/mental ratings, read by each
  club with an error that shrinks as that club's scouting quality rises.

Noise is deterministic per (session, draft year, prospect, test) so results
are reproducible and persist on the prospect (``player.draft_combine``).
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

COMBINE_SCHEMA_VERSION = 2

# Private-meeting slots the user's club gets at the combine.
USER_INTERVIEW_SLOTS = 10
USER_DINNER_SLOTS = 3

# --------------------------------------------------------------------------- tests
# mean / sd / lo / hi are NHL-combine-scale norms for draft-eligible prospects.
# validity = correlation between the driving attribute composite and the
# measured result (1 - validity^2 of the variance is test-day noise).
COMBINE_TESTS: List[Dict[str, Any]] = [
    {
        "id": "standing_long_jump", "label": "Standing Long Jump", "short": "Long Jump",
        "unit": "in", "better": "high", "decimals": 1,
        "mean": 104.0, "sd": 5.2, "lo": 86.0, "hi": 124.0, "validity": 0.82,
        "group": "power", "weight": 1.0,
        "drivers": [("skg_explosiveness", 0.40), ("skg_acceleration", 0.25), ("phy_strength", 0.20), ("skg_speed", 0.15)],
        "goalie_drivers": [("g_athleticism", 0.35)],
        "age": 0.12, "mass": -0.10,
        "measures": "Lower-body explosiveness",
    },
    {
        "id": "vo2_max", "label": "VO2 Max", "short": "VO2 Max",
        "unit": "ml/kg/min", "better": "high", "decimals": 1,
        "mean": 53.5, "sd": 3.8, "lo": 42.0, "hi": 66.0, "validity": 0.80,
        "group": "engine", "weight": 0.8,
        "drivers": [("phy_endurance", 0.35), ("phy_stamina", 0.35), ("phy_recovery_rate", 0.15), ("trait:work_ethic", 0.15)],
        "age": 0.05, "mass": -0.15,
        "measures": "Aerobic capacity",
    },
    {
        "id": "wingate_peak", "label": "Wingate Peak Power", "short": "Peak Power",
        "unit": "W/kg", "better": "high", "decimals": 2,
        "mean": 14.4, "sd": 1.05, "lo": 11.2, "hi": 18.6, "validity": 0.80,
        "group": "power", "weight": 1.0,
        "drivers": [("skg_explosiveness", 0.35), ("phy_strength", 0.30), ("skg_acceleration", 0.20), ("skg_speed", 0.15)],
        "goalie_drivers": [("g_athleticism", 0.30)],
        "age": 0.15, "mass": 0.0,
        "measures": "Anaerobic peak power",
    },
    {
        "id": "wingate_fatigue", "label": "Wingate Fatigue Index", "short": "Fatigue",
        "unit": "%", "better": "low", "decimals": 1,
        "mean": 46.0, "sd": 6.0, "lo": 28.0, "hi": 66.0, "validity": 0.75,
        "group": "engine", "weight": 0.6,
        "drivers": [("phy_stamina", 0.40), ("phy_recovery_rate", 0.35), ("phy_endurance", 0.25)],
        "age": 0.05, "mass": 0.0,
        "measures": "Power lost over 30 seconds",
    },
    {
        "id": "pro_agility", "label": "Pro Agility (5-10-5)", "short": "Agility",
        "unit": "s", "better": "low", "decimals": 2,
        "mean": 4.47, "sd": 0.11, "lo": 4.10, "hi": 4.90, "validity": 0.80,
        "group": "agility", "weight": 1.0,
        "drivers": [("skg_agility", 0.35), ("skg_pivot_speed", 0.25), ("skg_edge_work", 0.20), ("phy_balance", 0.20)],
        "goalie_drivers": [("g_athleticism", 0.35)],
        "age": 0.05, "mass": -0.10,
        "measures": "Change of direction",
    },
    {
        "id": "bench_press", "label": "Bench Press (150 lb)", "short": "Bench",
        "unit": "reps", "better": "high", "decimals": 0,
        "mean": 8.0, "sd": 3.3, "lo": 0.0, "hi": 24.0, "validity": 0.78,
        "group": "strength", "weight": 0.6,
        "drivers": [("phy_strength", 0.55), ("phy_physicality", 0.25), ("phy_checking", 0.20)],
        "age": 0.25, "mass": 0.30,
        "measures": "Upper-body strength",
    },
    {
        "id": "grip_strength", "label": "Grip Strength", "short": "Grip",
        "unit": "kg", "better": "high", "decimals": 1,
        "mean": 58.0, "sd": 6.0, "lo": 40.0, "hi": 82.0, "validity": 0.75,
        "group": "strength", "weight": 0.5,
        "drivers": [("phy_strength", 0.45), ("def_board_battles", 0.20), ("pc_puck_protection", 0.20), ("phy_checking", 0.15)],
        "age": 0.20, "mass": 0.25,
        "measures": "Hand and forearm strength",
    },
    {
        "id": "pull_ups", "label": "Pull-Ups", "short": "Pull-Ups",
        "unit": "reps", "better": "high", "decimals": 0,
        "mean": 10.0, "sd": 3.8, "lo": 0.0, "hi": 26.0, "validity": 0.75,
        "group": "strength", "weight": 0.6,
        "drivers": [("phy_strength", 0.45), ("phy_stamina", 0.25), ("trait:work_ethic", 0.15), ("phy_endurance", 0.15)],
        "age": 0.15, "mass": -0.30,
        "measures": "Relative strength",
    },
    {
        "id": "y_balance", "label": "Y-Balance", "short": "Y-Balance",
        "unit": "%", "better": "high", "decimals": 1,
        "mean": 98.0, "sd": 4.3, "lo": 84.0, "hi": 113.0, "validity": 0.75,
        "group": "agility", "weight": 0.6,
        "drivers": [("phy_balance", 0.40), ("skg_balance_skating", 0.35), ("skg_edge_work", 0.25)],
        "goalie_drivers": [("g_athleticism", 0.25)],
        "age": 0.0, "mass": -0.05,
        "measures": "Single-leg reach and stability",
    },
]
TEST_BY_ID = {t["id"]: t for t in COMBINE_TESTS}

MEASUREMENTS: List[Dict[str, Any]] = [
    {"id": "height", "label": "Height", "unit": "in", "decimals": 2, "better": "high"},
    {"id": "weight", "label": "Weight", "unit": "lb", "decimals": 0, "better": "high"},
    {"id": "wingspan", "label": "Wingspan", "unit": "in", "decimals": 2, "better": "high"},
]

# Interview dimensions: what clubs actually probe at combine interviews.
INTERVIEW_DIMENSIONS: List[Dict[str, Any]] = [
    {
        "id": "drive", "label": "Compete & drive", "weight": 0.24,
        "drivers": [("trait:competitiveness", 0.35), ("trait:work_ethic", 0.35), ("dev_work_ethic", 0.30)],
    },
    {
        "id": "coachability", "label": "Coachability", "weight": 0.20,
        "drivers": [("trait:coachability", 0.45), ("dev_coachability", 0.35), ("dev_learning_ability", 0.20)],
    },
    {
        "id": "maturity", "label": "Maturity & composure", "weight": 0.22,
        "drivers": [("trait:mental_toughness", 0.30), ("per_emotional_stability", 0.25), ("iqm_composure", 0.25), ("trait_inv:volatility", 0.20)],
    },
    {
        "id": "leadership", "label": "Leadership & presence", "weight": 0.17,
        "drivers": [("trait:leadership", 0.40), ("per_leadership", 0.35), ("per_media_handling", 0.15), ("trait:media_comfort", 0.10)],
    },
    {
        "id": "hockey_sense", "label": "Whiteboard / hockey sense", "weight": 0.17,
        "drivers": [("iqm_hockey_iq", 0.40), ("iqm_game_sense", 0.30), ("pm_decision_making", 0.30)],
    },
]

_INTERVIEW_NOTES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "drive": (
        ("Walked through his off-season training block in detail. Real hunger.",
         "Talked about the next level like a job he intends to take."),
        ("Energy dropped when we pushed on his own game.",
         "Vague on what he does away from the rink to get better."),
    ),
    "coachability": (
        ("Took the film critique well and came back with follow-ups.",
         "Owned the mistakes we showed him and explained the fix."),
        ("Bristled when we cut up his defensive reads.",
         "Deflected the tape critique onto linemates."),
    ),
    "maturity": (
        ("Calm under the tough questions. Older than his age in the room.",
         "Didn't flinch when we went after his weak spots."),
        ("Got rattled when the questions turned uncomfortable.",
         "Emotional swings showed up as the interview went on."),
    ),
    "leadership": (
        ("Comfortable running the room. Natural communicator.",
         "Teammates and coaches come up constantly in his answers."),
        ("Quiet, short answers. Hard to get a read on him.",
         "Doesn't see himself as a voice in the room yet."),
    ),
    "hockey_sense": (
        ("Sharp on the whiteboard. Saw the play two passes ahead.",
         "Diagrammed our breakout options without prompting."),
        ("Struggled with the whiteboard breakout scenarios.",
         "Needed the systems questions repeated."),
    ),
}


# --------------------------------------------------------------------------- utils
def _seeded_rng(*parts: Any) -> random.Random:
    digest = hashlib.md5(":".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 1.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
    return m, math.sqrt(var) if var > 1e-9 else 1.0


def _standardize(values: Mapping[str, float]) -> Dict[str, float]:
    m, sd = _mean_sd(list(values.values()))
    return {k: (float(v) - m) / sd for k, v in values.items()}


def _trait(player: Any, key: str) -> Optional[float]:
    for src in (getattr(player, "traits", None), getattr(player, "psychology", None)):
        if src is None:
            continue
        raw = src.get(key) if isinstance(src, dict) else getattr(src, key, None)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        return v * 100.0 if v <= 1.0 else v
    return None


def _rating(player: Any, key: str) -> Optional[float]:
    ratings = getattr(player, "ratings", None)
    if not isinstance(ratings, dict):
        return None
    raw = ratings.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _driver_value(player: Any, key: str) -> Optional[float]:
    if key.startswith("trait_inv:"):
        v = _trait(player, key.split(":", 1)[1])
        return None if v is None else 100.0 - v
    if key.startswith("trait:"):
        return _trait(player, key.split(":", 1)[1])
    return _rating(player, key)


def _is_goalie(entry: Mapping[str, Any]) -> bool:
    return str(entry.get("position") or "").upper() == "G"


def percentile_rank(value: float, population: Sequence[float], better: str = "high") -> int:
    """Share of the class this result beats (ties count half), 0-100."""
    pop = [float(x) for x in population]
    if not pop:
        return 50
    v = float(value)
    if better == "low":
        below = sum(1 for x in pop if x > v)
    else:
        below = sum(1 for x in pop if x < v)
    ties = sum(1 for x in pop if x == v) - 1
    pct = (below + 0.5 * max(0, ties)) / max(1, len(pop) - 1) * 100.0 if len(pop) > 1 else 50.0
    return int(round(_clamp(pct, 0.0, 100.0)))


def combine_label(score: float) -> str:
    """Label for the 60±10 athletic index (old combine_score semantics)."""
    s = float(score)
    if s >= 78:
        return "Elite"
    if s >= 70:
        return "Strong"
    if s >= 55:
        return "Average"
    if s >= 47:
        return "Below Average"
    return "Poor"


def interview_label(z: float) -> str:
    if z >= 1.1:
        return "Elite"
    if z >= 0.45:
        return "Strong"
    if z >= -0.45:
        return "Average"
    if z >= -1.1:
        return "Below Average"
    return "Poor"


def interview_grade(z: float) -> str:
    if z >= 1.4:
        return "A"
    if z >= 0.8:
        return "A-"
    if z >= 0.35:
        return "B+"
    if z >= -0.1:
        return "B"
    if z >= -0.55:
        return "B-"
    if z >= -1.0:
        return "C+"
    if z >= -1.5:
        return "C"
    return "D"


def read_sigma_for_quality(scouting_quality: float) -> float:
    """Interview read error (in class SD units) for a club's scouting quality."""
    q = float(scouting_quality or 60.0)
    return round(_clamp(0.30 + max(0.0, 88.0 - q) * 0.018, 0.30, 1.05), 3)


def read_confidence_label(sigma: float) -> str:
    if sigma <= 0.35:
        return "High"
    if sigma <= 0.6:
        return "Medium"
    return "Low"


def knowledge_floor_for_quality(scouting_quality: float) -> int:
    """Scouting-file depth the combine gives on physical/athletic tools."""
    q = float(scouting_quality or 60.0)
    return int(round(_clamp(58.0 + (q - 55.0) * 0.6, 55.0, 85.0)))


def format_feet_inches(inches: float) -> str:
    total = float(inches)
    feet = int(total // 12)
    rem = total - feet * 12
    if abs(rem - round(rem)) < 1e-6:
        return f"{feet}'{int(round(rem))}\""
    return f"{feet}'{rem:.2f}".rstrip("0").rstrip(".") + "\""


def format_test_value(test_id: str, value: Any) -> str:
    if value is None:
        return "—"
    if test_id in ("height", "wingspan"):
        return format_feet_inches(float(value))
    spec = TEST_BY_ID.get(test_id)
    if test_id == "weight":
        return f"{int(round(float(value)))} lb"
    if not spec:
        return str(value)
    dec = int(spec.get("decimals") or 0)
    num = f"{float(value):.{dec}f}" if dec else str(int(round(float(value))))
    unit = spec.get("unit") or ""
    return f"{num} {unit}".strip() if unit not in ("%",) else f"{num}%"


# --------------------------------------------------------------------------- index
def build_prospect_player_index(session: Any) -> Dict[str, Any]:
    """Map prospect id -> live Player for every development-league player."""
    league = getattr(getattr(session, "sim", None), "league", None)
    index: Dict[str, Any] = {}
    for block in getattr(league, "development_leagues", None) or []:
        if not isinstance(block, dict):
            continue
        for tm in block.get("teams") or []:
            if not isinstance(tm, dict):
                continue
            for p in tm.get("players") or []:
                pid = str(getattr(p, "id", "") or "")
                if pid and pid not in index:
                    index[pid] = p
    return index


# --------------------------------------------------------------------------- invites
def select_combine_invites(entries: List[Mapping[str, Any]], *, cap: int = 100) -> List[str]:
    """NHL Central Scouting style invite list (deterministic, data-driven).

    Top of the public board, the best-ranked goalies (they test on a separate
    track and would otherwise be squeezed out), and late risers whose stock
    climbed hard over the final weeks.
    """
    if not entries:
        return []
    ranked = sorted(entries, key=lambda e: int(e.get("rank") or 999))
    invited: List[str] = []
    seen: set = set()

    def _add(e: Mapping[str, Any]) -> None:
        k = str(e.get("key") or "")
        if k and k not in seen:
            seen.add(k)
            invited.append(k)

    core = max(1, min(len(ranked), cap - 14))
    for e in ranked[:core]:
        _add(e)
    goalies = [e for e in ranked if _is_goalie(e)]
    for e in goalies[:10]:
        if int(e.get("rank") or 999) <= 160:
            _add(e)
    risers = [
        e for e in ranked[core:core + 80]
        if float(e.get("stock_delta") or 0) >= 6 or float(e.get("weekly_stock_delta") or 0) >= 4
    ]
    risers.sort(key=lambda e: -float(e.get("stock_delta") or 0))
    for e in risers:
        if len(invited) >= cap:
            break
        _add(e)
    order = {str(e.get("key")): int(e.get("rank") or 999) for e in ranked}
    invited = invited[:cap]
    invited.sort(key=lambda k: order.get(k, 999))
    return invited


# --------------------------------------------------------------------------- testing
def _driver_matrix(
    players: Mapping[str, Any],
    keys: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    """Class z-scores per driver key (missing values sit at the class mean)."""
    out: Dict[str, Dict[str, float]] = {}
    for key in set(keys):
        raw: Dict[str, float] = {}
        for pid, p in players.items():
            v = _driver_value(p, key)
            if v is not None:
                raw[pid] = v
        if len(raw) < 3:
            out[key] = {pid: 0.0 for pid in players}
            continue
        z = _standardize(raw)
        out[key] = {pid: z.get(pid, 0.0) for pid in players}
    return out


def _height_cm(player: Any, entry: Mapping[str, Any]) -> Optional[float]:
    try:
        from app.sim_engine.entities.player import clamp_height_cm_for_position

        ident = getattr(player, "identity", None)
        raw = getattr(ident, "height_cm", 0) if ident is not None else 0
        hcm = clamp_height_cm_for_position(raw, entry.get("position"))
        if hcm:
            return float(hcm)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    if entry.get("height_cm"):
        try:
            return float(entry["height_cm"])
        except (TypeError, ValueError):
            return None
    return None


def _weight_kg(player: Any, entry: Mapping[str, Any]) -> Optional[float]:
    ident = getattr(player, "identity", None)
    wkg = getattr(ident, "weight_kg", 0) if ident is not None else 0
    try:
        if wkg and float(wkg) > 0:
            return float(wkg)
    except (TypeError, ValueError):
        pass
    if entry.get("weight"):
        try:
            return float(entry["weight"]) / 2.20462
        except (TypeError, ValueError):
            return None
    return None


def _medical_eval(player: Any, durability_z: float) -> Dict[str, Any]:
    health = getattr(player, "health", None)
    history = list(getattr(health, "injury_history", None) or []) if health is not None else []
    chronic = [str(c) for c in (getattr(health, "chronic_flags", None) or [])] if health is not None else []
    days_out = int(getattr(health, "days_injured_career", 0) or 0) if health is not None else 0
    games_left = int(getattr(player, "_prospect_injury_games_remaining", 0) or 0)
    status_raw = getattr(health, "injury_status", None) if health is not None else None
    status = str(getattr(status_raw, "value", status_raw) or "healthy").lower()
    injured_now = bool(getattr(player, "prospect_injured", False)) or games_left > 0 or status not in ("healthy", "", "none")
    note_now = str(getattr(player, "injury_note", "") or getattr(player, "injury_status", "") or "")
    if note_now.lower() in ("none", "healthy"):
        note_now = ""

    score = 0.0
    reasons: List[str] = []
    if injured_now:
        score += 1.5 + (1.0 if games_left >= 4 else 0.0)
        reasons.append(note_now or "Carrying an in-season injury into the combine")
    if chronic:
        score += 2.0
        reasons.append(f"Chronic: {', '.join(chronic[:2])}")
    if history:
        score += min(2.0, float(len(history)))
        reasons.append(f"{len(history)} prior injur{'y' if len(history) == 1 else 'ies'} on file")
    if days_out >= 30:
        score += 0.5
    if durability_z <= -1.5:
        score += 1.0
        reasons.append("Durability markers in the bottom of the class")
    elif durability_z <= -1.0 and (history or injured_now):
        score += 0.5

    if score >= 3.0:
        level = "High"
    elif score >= 1.0:
        level = "Moderate"
    else:
        level = "Low"
    note = "Cleared by the medical staff — no restrictions." if level == "Low" else "; ".join(reasons[:2])
    return {
        "level": level,
        "flag": level != "Low",
        "injured_now": injured_now,
        "fitness_cleared": not injured_now,
        "note": note,
        "durability_z": round(durability_z, 2),
    }


def run_combine_testing(
    session_key: str,
    draft_year: int,
    invite_entries: List[Mapping[str, Any]],
    players: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Measure every invitee. Returns pid -> public result block."""
    pids = [str(e.get("key")) for e in invite_entries if str(e.get("key")) in players]
    entry_by = {str(e.get("key")): e for e in invite_entries}
    pl = {pid: players[pid] for pid in pids}
    results: Dict[str, Dict[str, Any]] = {}
    if not pl:
        return results

    driver_keys: List[str] = ["phy_durability", "phy_injury_resistance", "st_injury_proneness_inv",
                              "def_stick_checking", "def_interception_skill"]
    for t in COMBINE_TESTS:
        driver_keys.extend(k for k, _ in t["drivers"])
        driver_keys.extend(k for k, _ in t.get("goalie_drivers") or [])
    zmat = _driver_matrix(pl, driver_keys)

    ages = {pid: float(entry_by[pid].get("age") or getattr(getattr(pl[pid], "identity", None), "age", 18) or 18) for pid in pids}
    age_mean = sum(ages.values()) / len(ages)
    mass_raw = {pid: _weight_kg(pl[pid], entry_by[pid]) for pid in pids}
    mass_known = {k: v for k, v in mass_raw.items() if v}
    mass_z = _standardize(mass_known) if len(mass_known) >= 3 else {}

    durability = {
        pid: (zmat["phy_durability"][pid] + zmat["phy_injury_resistance"][pid] + zmat["st_injury_proneness_inv"][pid]) / 3.0
        for pid in pids
    }
    dur_z = _standardize(durability)
    medical = {pid: _medical_eval(pl[pid], dur_z.get(pid, 0.0)) for pid in pids}

    # --- body measurements
    for pid in pids:
        p, e = pl[pid], entry_by[pid]
        rng = _seeded_rng(session_key, draft_year, pid, "body")
        hcm = _height_cm(p, e)
        wkg = mass_raw.get(pid)
        meas: Dict[str, Any] = {}
        if hcm:
            h_in = hcm / 2.54 + rng.gauss(0.0, 0.2)
            meas["height"] = round(round(h_in * 4) / 4, 2)
            reach_z = (zmat["def_stick_checking"][pid] + zmat["def_interception_skill"][pid]) / 2.0
            ape = _clamp(1.5 + 0.5 * reach_z + rng.gauss(0.0, 1.15), -2.5, 6.0)
            meas["wingspan"] = round(round((h_in + ape) * 4) / 4, 2)
            meas["ape_index"] = round(meas["wingspan"] - meas["height"], 2)
        if wkg:
            meas["weight"] = int(round(wkg * 2.20462 + rng.gauss(0.0, 0.8)))
        results[pid] = {
            "prospect_id": pid,
            "draft_year": int(draft_year),
            "schema": COMBINE_SCHEMA_VERSION,
            "measurements": meas,
            "tests": {},
            "medical": medical[pid],
            "tested": bool(medical[pid]["fitness_cleared"]),
        }

    tested = [pid for pid in pids if results[pid]["tested"]]
    if not tested:
        return results

    # --- fitness tests
    for t in COMBINE_TESTS:
        raw: Dict[str, float] = {}
        for pid in tested:
            drivers = list(t["drivers"])
            if _is_goalie(entry_by[pid]) and t.get("goalie_drivers"):
                drivers = [(k, w * 0.7) for k, w in drivers] + list(t["goalie_drivers"])
            val = sum(w * zmat[k][pid] for k, w in drivers)
            val += float(t.get("age") or 0.0) * (ages[pid] - age_mean)
            val += float(t.get("mass") or 0.0) * mass_z.get(pid, 0.0)
            raw[pid] = val
        z_true = _standardize(raw)
        validity = float(t["validity"])
        resid = math.sqrt(max(0.0, 1.0 - validity ** 2))
        sign = -1.0 if t["better"] == "low" else 1.0
        values: Dict[str, float] = {}
        for pid in tested:
            rng = _seeded_rng(session_key, draft_year, pid, t["id"])
            z_meas = validity * z_true[pid] + resid * rng.gauss(0.0, 1.0)
            v = _clamp(float(t["mean"]) + sign * float(t["sd"]) * z_meas, float(t["lo"]), float(t["hi"]))
            dec = int(t["decimals"])
            values[pid] = round(v, dec) if dec else float(int(round(v)))
        pop = list(values.values())
        for pid in tested:
            results[pid]["tests"][t["id"]] = {
                "value": values[pid] if t["decimals"] else int(values[pid]),
                "pct": percentile_rank(values[pid], pop, t["better"]),
            }

    # --- measurement percentiles (whole class, everyone is measured)
    for m in MEASUREMENTS:
        pop = [results[pid]["measurements"][m["id"]] for pid in pids if m["id"] in results[pid]["measurements"]]
        for pid in pids:
            meas = results[pid]["measurements"]
            if m["id"] in meas:
                meas[f"{m['id']}_pct"] = percentile_rank(meas[m["id"]], pop, "high")

    # --- athletic index: weighted mean of signed measured z, re-standardized
    comp_raw: Dict[str, float] = {}
    for pid in tested:
        num = 0.0
        den = 0.0
        for t in COMBINE_TESTS:
            row = results[pid]["tests"].get(t["id"])
            if not row:
                continue
            z = (float(row["value"]) - float(t["mean"])) / float(t["sd"])
            if t["better"] == "low":
                z = -z
            num += float(t["weight"]) * z
            den += float(t["weight"])
        comp_raw[pid] = num / den if den else 0.0
    comp_z = _standardize(comp_raw)
    comp_pop = list(comp_z.values())
    order = sorted(tested, key=lambda k: -comp_z[k])
    for i, pid in enumerate(order, start=1):
        score = round(_clamp(60.0 + 10.0 * comp_z[pid], 30.0, 95.0), 1)
        results[pid]["athletic_z"] = round(comp_z[pid], 3)
        results[pid]["combine_score"] = score
        results[pid]["combine_label"] = combine_label(score)
        results[pid]["athletic_pct"] = percentile_rank(comp_z[pid], comp_pop, "high")
        results[pid]["athletic_rank"] = i
    for pid in pids:
        results[pid]["tested_count"] = len(tested)
    return results


# --------------------------------------------------------------------------- interviews
def interview_truth(
    invite_entries: List[Mapping[str, Any]],
    players: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """True interview profile per invitee (class z per dimension + red flags)."""
    pids = [str(e.get("key")) for e in invite_entries if str(e.get("key")) in players]
    entry_by = {str(e.get("key")): e for e in invite_entries}
    pl = {pid: players[pid] for pid in pids}
    keys: List[str] = ["per_professionalism"]
    for d in INTERVIEW_DIMENSIONS:
        keys.extend(k for k, _ in d["drivers"])
    zmat = _driver_matrix(pl, keys)
    dims: Dict[str, Dict[str, float]] = {}
    for d in INTERVIEW_DIMENSIONS:
        raw = {pid: sum(w * zmat[k][pid] for k, w in d["drivers"]) for pid in pids}
        dims[d["id"]] = _standardize(raw) if len(raw) >= 3 else {pid: 0.0 for pid in pids}
    out: Dict[str, Dict[str, Any]] = {}
    for pid in pids:
        p = pl[pid]
        flags: List[str] = []
        penalty = 0.0
        if bool(entry_by[pid].get("character_concerns")):
            flags.append("character")
            penalty += 0.8
        ego = _trait(p, "ego")
        vol = _trait(p, "volatility")
        if ego is not None and ego >= 85:
            flags.append("ego")
            penalty += 0.3
        if vol is not None and vol >= 85:
            flags.append("volatility")
            penalty += 0.3
        if zmat["per_professionalism"][pid] <= -1.5:
            flags.append("professionalism")
            penalty += 0.3
        total = sum(d["weight"] * dims[d["id"]][pid] for d in INTERVIEW_DIMENSIONS) - penalty
        out[pid] = {
            "dims": {d["id"]: round(dims[d["id"]][pid], 3) for d in INTERVIEW_DIMENSIONS},
            "red_flags": flags,
            "raw_total": total,
        }
    if out:
        m, sd = _mean_sd([v["raw_total"] for v in out.values()]) if len(out) >= 3 else (0.0, 1.0)
        for pid in out:
            out[pid]["total_z"] = round((out[pid]["raw_total"] - m) / sd, 3)
            # Observations are mapped onto the same class scale as the truth.
            out[pid]["scale"] = (round(m, 5), round(sd, 5))
    return out


def observe_interview(
    truth: Mapping[str, Any],
    *,
    sigma: float,
    red_flag_detection: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """One club's read of a prospect in the room."""
    dims_true = truth.get("dims") or {}
    obs_dims: Dict[str, float] = {}
    for d in INTERVIEW_DIMENSIONS:
        obs_dims[d["id"]] = float(dims_true.get(d["id"], 0.0)) + rng.gauss(0.0, sigma)
    flags_seen: List[str] = []
    for flag in truth.get("red_flags") or []:
        p_detect = _clamp(0.35 + float(red_flag_detection or 0.5) * 0.6 - sigma * 0.25, 0.15, 0.95)
        if rng.random() < p_detect:
            flags_seen.append(flag)
    total = sum(d["weight"] * obs_dims[d["id"]] for d in INTERVIEW_DIMENSIONS)
    total -= 0.8 * ("character" in flags_seen) + 0.3 * sum(1 for f in flags_seen if f != "character")
    # Map the observed weighted sum back onto class-z scale (dimension weights sum to 1).
    total_z = total / 0.62
    return {"dims": obs_dims, "flags": flags_seen, "z": total_z, "sigma": sigma}


def combine_observations(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Precision-weighted merge of two interview reads of the same prospect."""
    sa = max(0.05, float(a.get("sigma") or 1.0))
    sb = max(0.05, float(b.get("sigma") or 1.0))
    wa, wb = 1.0 / sa ** 2, 1.0 / sb ** 2
    dims = {
        k: (float((a.get("dims") or {}).get(k, 0.0)) * wa + float((b.get("dims") or {}).get(k, 0.0)) * wb) / (wa + wb)
        for k in set((a.get("dims") or {}).keys()) | set((b.get("dims") or {}).keys())
    }
    z = (float(a.get("z") or 0.0) * wa + float(b.get("z") or 0.0) * wb) / (wa + wb)
    flags = sorted(set(a.get("flags") or []) | set(b.get("flags") or []))
    return {"dims": dims, "flags": flags, "z": z, "sigma": round(1.0 / math.sqrt(wa + wb), 3)}


def interview_notes(obs: Mapping[str, Any], pid: str, salt: str = "") -> List[str]:
    dims = obs.get("dims") or {}
    if not dims:
        return []
    ordered = sorted(dims.items(), key=lambda kv: kv[1])
    notes: List[str] = []
    hi_k, hi_v = ordered[-1]
    lo_k, lo_v = ordered[0]
    pick = int(hashlib.md5(f"{pid}:{salt}".encode()).hexdigest()[:4], 16) % 2
    if hi_v >= 0.6 and hi_k in _INTERVIEW_NOTES:
        notes.append(_INTERVIEW_NOTES[hi_k][0][pick])
    if lo_v <= -0.6 and lo_k in _INTERVIEW_NOTES:
        notes.append(_INTERVIEW_NOTES[lo_k][1][pick])
    flags = obs.get("flags") or []
    if "character" in flags:
        notes.append("Background calls raised character questions we could not put to rest.")
    elif "ego" in flags:
        notes.append("Ego showed in how he talked about teammates.")
    elif "volatility" in flags:
        notes.append("Temper came up with two of his former coaches.")
    elif "professionalism" in flags:
        notes.append("Showed up late and underprepared for the session.")
    if not notes:
        notes.append("Solid, even interview — nothing that moves the needle either way.")
    return notes[:3]


def interview_read_block(obs: Mapping[str, Any], pid: str, *, salt: str = "") -> Dict[str, Any]:
    z = float(obs.get("z") or 0.0)
    sigma = float(obs.get("sigma") or 1.0)
    return {
        "z": round(z, 3),
        "score": int(round(_clamp(60.0 + 12.0 * z, 20.0, 98.0))),
        "label": interview_label(z),
        "grade": interview_grade(z),
        "confidence": read_confidence_label(sigma),
        "sigma": round(sigma, 3),
        "dims": {k: round(float(v), 2) for k, v in (obs.get("dims") or {}).items()},
        "flags": list(obs.get("flags") or []),
        "notes": interview_notes(obs, pid, salt),
    }


# --------------------------------------------------------------------------- stock
def public_stock_boost(result: Mapping[str, Any], consensus_z: float, *, goalie: bool) -> Tuple[float, str]:
    """Draft-score points the combine adds to the public consensus (modest).

    One class SD of athletic testing is worth about one score point (a few
    spots in the middle of round one, less at the top where gaps are wide);
    interviews half that; medical flags subtract.
    """
    reasons: List[Tuple[float, str]] = []
    boost = 0.0
    if result.get("tested") and result.get("athletic_z") is not None:
        az = float(result.get("athletic_z") or 0.0)
        ath = _clamp(az, -2.2, 2.2) * (0.5 if goalie else 1.0)
        boost += ath
        if az >= 1.0:
            reasons.append((ath, "Tested near the top of the class"))
        elif az <= -1.0:
            reasons.append((ath, "Underwhelming fitness testing"))
    iz = _clamp(float(consensus_z or 0.0), -2.0, 2.0) * 0.5
    boost += iz
    if consensus_z >= 1.0:
        reasons.append((iz, "Clubs came away impressed in interviews"))
    elif consensus_z <= -1.0:
        reasons.append((iz, "Interviews raised questions around the league"))
    med = (result.get("medical") or {}).get("level")
    if med == "High":
        boost -= 2.0
        reasons.append((-2.0, "Medical red flag"))
    elif med == "Moderate":
        boost -= 0.6
        reasons.append((-0.6, "Medical follow-up required"))
    boost = round(_clamp(boost, -3.5, 3.0), 2)
    reason = max(reasons, key=lambda r: abs(r[0]))[1] if reasons else "Met expectations"
    return boost, reason


# --------------------------------------------------------------------------- teams
def team_impression(
    *,
    team_id: str,
    pid: str,
    entry: Mapping[str, Any],
    result: Mapping[str, Any],
    obs: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    """One club's combine file on a prospect (feeds its internal draft board)."""
    goalie = _is_goalie(entry)
    combine_trust = float(profile.get("combine_trust") or 0.5)
    interview_trust = float(profile.get("interview_trust") or 0.5)
    red_flag = float(profile.get("red_flag_detection") or 0.5)
    risk_tol = float(profile.get("risk_tolerance") or 0.5)
    strict = float(profile.get("do_not_draft_strictness") or 0.5)

    az = float(result.get("athletic_z") or 0.0) if result.get("tested") else 0.0
    ath_term = _clamp(az, -2.2, 2.2) * 1.2 * combine_trust * (0.5 if goalie else 1.0)
    int_z = float(obs.get("z") or 0.0)
    int_term = _clamp(int_z, -2.5, 2.5) * 1.6 * interview_trust
    med_level = (result.get("medical") or {}).get("level") or "Low"
    med_term = 0.0
    if med_level == "Moderate":
        med_term = -0.6 * (0.6 + red_flag * 0.8)
    elif med_level == "High":
        med_term = -1.5 * (0.6 + red_flag * 0.8)
    board_delta = ath_term + int_term + med_term

    flags = obs.get("flags") or []
    risk_delta = med_term * 0.6
    do_not_draft = False
    if med_level == "High" and strict > 0.7:
        do_not_draft = True
    if "character" in flags and (risk_tol < 0.35 or (int_z <= -1.6 and strict > 0.6)):
        do_not_draft = True
    if "character" in flags:
        risk_delta -= 4.0 * strict

    pub_rank = int(entry.get("rank") or 999)
    scout_favorite = bool(int_z >= 1.2 and az >= 0.4) or board_delta >= 3.2
    sleeper_tag = bool(pub_rank > 45 and board_delta >= 2.2 and float(profile.get("sleeper_detection") or 0) > 0.55)
    concern_tag = bool(board_delta <= -2.6 or do_not_draft)
    if do_not_draft:
        scout_note = "Internal concern — staff recommends passing."
    elif scout_favorite:
        scout_note = "Staff came out of the combine pounding the table for him."
    elif sleeper_tag:
        scout_note = "Combine file says he is better than his public ranking."
    elif concern_tag:
        scout_note = "Combine raised more questions than it answered."
    else:
        scout_note = ""
    return {
        "team_id": str(team_id),
        "prospect_id": pid,
        "interview_impression": interview_label(int_z),
        "interview_z": round(int_z, 3),
        "interview_sigma": round(float(obs.get("sigma") or 1.0), 3),
        "interview_flags": list(flags),
        "combine_impression": result.get("combine_label") or ("Did not test" if not result.get("tested") else "Average"),
        "medical_impression": med_level,
        "private_meeting_impression": "Not held",
        "scout_note": scout_note,
        "gm_note": "",
        "board_delta": round(board_delta, 2),
        "risk_delta": round(risk_delta, 2),
        "confidence_delta": round(max(0.0, 1.0 - float(obs.get("sigma") or 1.0)) * 4.0, 2),
        "do_not_draft": do_not_draft,
        "scout_favorite": scout_favorite,
        "sleeper_tag": sleeper_tag,
        "concern_tag": concern_tag,
    }


def public_block(result: Mapping[str, Any], *, consensus: Mapping[str, Any], stock: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact public combine record stored on the player (dossier/board)."""
    tests = {
        tid: {"value": row.get("value"), "pct": row.get("pct")}
        for tid, row in (result.get("tests") or {}).items()
    }
    meas = dict(result.get("measurements") or {})
    med = dict(result.get("medical") or {})
    return {
        "schema": COMBINE_SCHEMA_VERSION,
        "draft_year": result.get("draft_year"),
        "attended": True,
        "tested": bool(result.get("tested")),
        "measurements": meas,
        "tests": tests,
        "combine_score": result.get("combine_score"),
        "combine_label": result.get("combine_label") or ("Did not test" if not result.get("tested") else None),
        "athletic_pct": result.get("athletic_pct"),
        "athletic_rank": result.get("athletic_rank"),
        "tested_count": result.get("tested_count"),
        "medical": {"level": med.get("level"), "note": med.get("note"), "cleared": bool(med.get("fitness_cleared"))},
        "interview_consensus": {
            "label": consensus.get("label"),
            "grade": consensus.get("grade"),
            "score": consensus.get("score"),
        },
        "stock": dict(stock or {}),
    }


def test_catalog() -> List[Dict[str, Any]]:
    out = []
    for m in MEASUREMENTS:
        out.append({
            "id": m["id"], "label": m["label"], "short": m["label"], "unit": m["unit"],
            "better": m["better"], "decimals": m["decimals"], "kind": "measurement",
        })
    for t in COMBINE_TESTS:
        out.append({
            "id": t["id"], "label": t["label"], "short": t["short"], "unit": t["unit"],
            "better": t["better"], "decimals": t["decimals"], "kind": "fitness",
            "group": t["group"], "measures": t["measures"],
            "norm_mean": t["mean"], "norm_sd": t["sd"],
        })
    return out
