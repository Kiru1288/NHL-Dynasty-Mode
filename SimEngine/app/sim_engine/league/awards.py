"""
Canonical league awards registry and computation.

Official winners, Award Watch official races, eligibility, normalization,
deterministic ballot simulation, and payload assembly all live here.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .playoffs import PlayoffResult
from .standings import StandingsTable, TeamStandingRecord

BALLOT_POINTS = [5.0, 3.0, 1.8, 1.0, 0.6]
VOTER_COUNT = 130
SUBJECTIVE_TROPHY_PUBLIC_CASE = {
    "hart": "Most valuable all-around season.",
    "norris": "Premier defenseman season.",
    "selke": "Elite defensive forward season.",
    "vezina": "Best goaltender season.",
    "calder": "Top first-year NHL player.",
    "lady_byng": "Production with exceptional discipline.",
    "ted_lindsay": "Players' view of most outstanding player.",
    "conn_smythe": "Most valuable playoff performer.",
}

VOTER_ARCHETYPES = (
    "production",
    "two_way",
    "team_success",
    "analytics",
    "workload",
    "traditional",
)


def normalize_percentage(value: Any) -> float:
    """Return a fraction in [0, 1]. Values > 1 are treated as percent points."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v > 1.0:
        v = v / 100.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _seed_int(season_seed: Any, *parts: Any) -> int:
    raw = "|".join(str(p) for p in (season_seed, *parts) if p is not None)
    if not raw:
        raw = "awards-default"
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:12], 16)


def _rng(season_seed: Any, *parts: Any) -> random.Random:
    return random.Random(_seed_int(season_seed, *parts))


def validate_required_award_fields(row: Mapping[str, Any], fields: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for key in fields:
        if key not in row or row.get(key) is None:
            missing.append(str(key))
            continue
        val = row.get(key)
        if isinstance(val, str) and not val.strip():
            missing.append(str(key))
    return missing


def percentile_rank(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.5
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    # Midrank percentile
    below = sum(1 for v in ordered if v < value)
    equal = sum(1 for v in ordered if v == value)
    return (below + 0.5 * equal) / float(n)


def robust_z(values: Sequence[float], value: float, *, clamp: float = 3.0) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = ordered[len(ordered) // 2]
    abs_dev = sorted(abs(v - mid) for v in ordered)
    mad = abs_dev[len(abs_dev) // 2] or 1e-6
    z = 0.6745 * (float(value) - mid) / mad
    return max(-clamp, min(clamp, z))


def _norm01_from_z(z: float) -> float:
    return max(0.0, min(1.0, 0.5 + z / 6.0))


def normalize_pool_metric(pool: Sequence[Mapping[str, Any]], getter: Callable[[Mapping[str, Any]], float]) -> Dict[str, float]:
    vals = [getter(r) for r in pool]
    out: Dict[str, float] = {}
    for row in pool:
        pid = _pid(row)
        out[pid] = percentile_rank(vals, getter(row))
    return out


def _pid(row: Mapping[str, Any]) -> str:
    return str(row.get("player_id") or row.get("id") or row.get("entity_id") or "")


def _tid(row: Mapping[str, Any]) -> str:
    return str(row.get("team_id") or "")


def _pts(row: Mapping[str, Any]) -> int:
    return _safe_int(row.get("pts"), _safe_int(row.get("g")) + _safe_int(row.get("a")))


def _goals(row: Mapping[str, Any]) -> int:
    return _safe_int(row.get("g"), _safe_int(row.get("goals")))


def _pos(row: Mapping[str, Any]) -> str:
    return str(row.get("position") or row.get("pos") or "").upper()


def _gp(row: Mapping[str, Any]) -> int:
    return _safe_int(row.get("gp"), _safe_int(row.get("games_played")))


def _is_goalie(row: Mapping[str, Any]) -> bool:
    return _pos(row) == "G"


def _is_defense(row: Mapping[str, Any]) -> bool:
    return _pos(row) == "D"


def _is_forward(row: Mapping[str, Any]) -> bool:
    return not _is_goalie(row) and not _is_defense(row)


def _stat_scope(row: Mapping[str, Any]) -> str:
    return str(row.get("stat_scope") or row.get("scope") or "regular_season").strip().lower()


def filter_regular_season_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    allowed = {"", "regular", "regular_season", "rs", "none"}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        scope = _stat_scope(row)
        if scope in allowed or scope == "none":
            out.append(dict(row))
        elif scope in {"playoff", "playoffs", "preseason", "exhibition", "international", "allstar", "all_star"}:
            continue
        else:
            # Unknown scopes treated conservatively as non-regular except missing (already covered).
            continue
    return out


def filter_playoff_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        scope = _stat_scope(row)
        if scope in {"playoff", "playoffs", "postseason"}:
            out.append(dict(row))
    return out


def _season_games_threshold(season_length: int, share: float, minimum: int = 1) -> int:
    return max(minimum, int(math.ceil(float(season_length) * float(share))))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _defn(
    award_id: str,
    name: str,
    *,
    recipient_type: str,
    category: str,
    display_metric: str,
    official: bool = True,
    watch_enabled: bool = True,
    ceremony_enabled: bool = True,
    eligibility: str = "",
    score: str = "",
    tiebreakers: Optional[List[str]] = None,
    required_fields: Optional[List[str]] = None,
    finalist_count: int = 3,
    supports_shared_winners: bool = False,
    public_status: str = "official",
    watch_type: str = "official_live_race",
) -> Dict[str, Any]:
    return {
        "award_id": award_id,
        "name": name,
        "recipient_type": recipient_type,
        "category": category,
        "official": official,
        "watch_enabled": watch_enabled,
        "ceremony_enabled": ceremony_enabled,
        "eligibility": eligibility or award_id,
        "score": score or award_id,
        "tiebreakers": list(tiebreakers or []),
        "required_fields": list(required_fields or []),
        "finalist_count": int(finalist_count),
        "supports_shared_winners": bool(supports_shared_winners),
        "display_metric": display_metric,
        "public_status": public_status,
        "watch_type": watch_type,
    }


AWARD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "presidents": _defn(
        "presidents",
        "Presidents' Trophy",
        recipient_type="team",
        category="team_result",
        display_metric="PTS",
        watch_type="official_live_race",
        eligibility="presidents",
        score="presidents",
        tiebreakers=["points", "regulation_wins", "wins", "goal_diff"],
    ),
    "stanley": _defn(
        "stanley",
        "Stanley Cup",
        recipient_type="team",
        category="playoff",
        display_metric="Champion",
        watch_enabled=False,
        watch_type="official_live_race",
    ),
    "conference_champions": _defn(
        "conference_champions",
        "Conference Champions",
        recipient_type="team",
        category="playoff",
        display_metric="Champion",
        watch_enabled=False,
        ceremony_enabled=False,
        supports_shared_winners=True,
    ),
    "art_ross": _defn(
        "art_ross",
        "Art Ross Trophy",
        recipient_type="player",
        category="stat_race",
        display_metric="PTS",
        supports_shared_winners=True,
        eligibility="art_ross",
        score="art_ross",
        tiebreakers=["points", "goals", "ppg", "shared"],
    ),
    "rocket": _defn(
        "rocket",
        "Rocket Richard Trophy",
        recipient_type="player",
        category="stat_race",
        display_metric="G",
        supports_shared_winners=True,
        eligibility="rocket",
        score="rocket",
        tiebreakers=["goals", "fewer_gp", "points", "shared"],
    ),
    "hart": _defn(
        "hart",
        "Hart Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="hart",
        score="hart",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "norris": _defn(
        "norris",
        "James Norris Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="norris",
        score="norris",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "selke": _defn(
        "selke",
        "Frank J. Selke Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="selke",
        score="selke",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "calder": _defn(
        "calder",
        "Calder Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="calder",
        score="calder",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "vezina": _defn(
        "vezina",
        "Vezina Trophy",
        recipient_type="goalie",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        required_fields=["gp"],
        eligibility="vezina",
        score="vezina",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "conn_smythe": _defn(
        "conn_smythe",
        "Conn Smythe Trophy",
        recipient_type="player",
        category="playoff",
        display_metric="Playoff ballot points",
        watch_type="official_projected_ballot",
        eligibility="conn_smythe",
        score="conn_smythe",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "jennings": _defn(
        "jennings",
        "William M. Jennings Trophy",
        recipient_type="multiple",
        category="team_result",
        display_metric="Team GA",
        supports_shared_winners=True,
        eligibility="jennings",
        score="jennings",
        tiebreakers=["team_ga", "shared"],
    ),
    "lady_byng": _defn(
        "lady_byng",
        "Lady Byng Memorial Trophy",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        ceremony_enabled=False,
        eligibility="lady_byng",
        score="lady_byng",
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "ted_lindsay": _defn(
        "ted_lindsay",
        "Ted Lindsay Award",
        recipient_type="player",
        category="ballot",
        display_metric="Ballot points",
        watch_type="official_projected_ballot",
        ceremony_enabled=False,
        eligibility="ted_lindsay",
        score="ted_lindsay",
        required_fields=["gp"],
        tiebreakers=["ballot_points", "first_place_votes", "canonical_score"],
    ),
    "masterton": _defn(
        "masterton",
        "Bill Masterton Memorial Trophy",
        recipient_type="player",
        category="selection",
        display_metric="Selection",
        watch_enabled=False,
        watch_type="watch_only",
        ceremony_enabled=False,
        eligibility="masterton",
        score="masterton",
        required_fields=["injury_games_missed", "games_returned"],
    ),
    "messier": _defn(
        "messier",
        "Mark Messier Leadership Award",
        recipient_type="player",
        category="selection",
        display_metric="Selection",
        watch_enabled=False,
        watch_type="watch_only",
        ceremony_enabled=False,
        eligibility="messier",
        score="messier",
        required_fields=["is_captain", "leadership_score"],
    ),
    "jack_adams": _defn(
        "jack_adams",
        "Jack Adams Award",
        recipient_type="coach",
        category="ballot",
        display_metric="Ballot points",
        watch_enabled=False,
        watch_type="watch_only",
        ceremony_enabled=False,
        eligibility="jack_adams",
        score="jack_adams",
        required_fields=["coach_id", "expected_points", "actual_points"],
    ),
    "all_star_1": _defn(
        "all_star_1",
        "First NHL All-Star Team",
        recipient_type="multiple",
        category="selection",
        display_metric="Selection",
        supports_shared_winners=True,
        watch_enabled=False,
        ceremony_enabled=False,
    ),
    "all_star_2": _defn(
        "all_star_2",
        "Second NHL All-Star Team",
        recipient_type="multiple",
        category="selection",
        display_metric="Selection",
        supports_shared_winners=True,
        watch_enabled=False,
        ceremony_enabled=False,
    ),
}

NAME_TO_ID = {v["name"]: k for k, v in AWARD_REGISTRY.items()}
# Legacy short names used in ceremony catalogs.
NAME_TO_ID.update(
    {
        "Norris Trophy": "norris",
        "Selke Trophy": "selke",
        "Maurice Richard Trophy": "rocket",
        "James Norris Memorial Trophy": "norris",
        "Frank J. Selke Trophy": "selke",
    }
)


@dataclass
class Award:
    name: str
    winner_team_id: Optional[str] = None
    winner_name: Optional[str] = None
    winner_player_id: Optional[str] = None
    winner_team_name: Optional[str] = None
    finalists: List[Any] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    winner_stats: Optional[Dict[str, Any]] = None
    rationale: str = ""
    # Extended canonical fields
    award_id: str = ""
    official: bool = True
    status: str = "complete"
    category: str = ""
    recipient_type: str = "player"
    winners: List[Dict[str, Any]] = field(default_factory=list)
    shared: bool = False
    full_results: List[Dict[str, Any]] = field(default_factory=list)
    display_metric: str = ""
    calculation_quality: str = "full"
    fallback_reason: Optional[str] = None
    public_rationale: str = ""
    eligibility_summary: str = ""
    stat_scope: str = "regular_season"
    season: Any = None
    unavailable_reason: Optional[str] = None
    voting: Optional[Dict[str, Any]] = None
    result: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Identity / Calder
# ---------------------------------------------------------------------------

def _player_from_rosters(teams: Optional[Sequence[Any]], pid: str) -> Any:
    if not teams or not pid:
        return None
    for t in teams:
        for p in getattr(t, "roster", None) or []:
            if str(getattr(p, "id", "") or "") == str(pid):
                return p
    return None


def _player_age(row: Mapping[str, Any], teams: Optional[Sequence[Any]] = None) -> Optional[int]:
    for key in ("age", "player_age", "season_age"):
        if row.get(key) is not None:
            return _safe_int(row.get(key), 0) or None
    p = _player_from_rosters(teams, _pid(row))
    if p is None:
        return None
    try:
        return int(getattr(p, "age", getattr(getattr(p, "identity", None), "age", None)))
    except Exception:
        return None


def calder_eligibility(
    row: Mapping[str, Any],
    *,
    teams: Optional[Sequence[Any]] = None,
    history: Optional[Mapping[str, Any]] = None,
    season_length: int = 82,
) -> Dict[str, Any]:
    """
    Canonical Calder eligibility.

    Uses season-history style fields when present:
      previous_nhl_gp, prior_nhl_seasons, nhl_gp_before, seasons_played, is_rookie/rookie
    Age is only one part of the rule (must be <= 25 for first NHL season under NHL-style caps).
    """
    pid = _pid(row)
    hist = dict(history or {})
    if pid and isinstance(hist.get(pid), Mapping):
        hist = {**hist, **dict(hist.get(pid) or {})}

    gp = _gp(row)
    min_gp = _season_games_threshold(season_length, 0.30, minimum=20)
    age = _player_age(row, teams)

    prior_gp = None
    for key in ("previous_nhl_gp", "prior_nhl_gp", "nhl_gp_before", "career_nhl_gp_before_season"):
        if row.get(key) is not None or hist.get(key) is not None:
            prior_gp = _safe_int(row.get(key, hist.get(key)), 0)
            break

    prior_seasons = None
    for key in ("prior_nhl_seasons", "nhl_seasons_before", "seasons_played", "pro_seasons"):
        if row.get(key) is not None or hist.get(key) is not None:
            prior_seasons = _safe_int(row.get(key, hist.get(key)), 0)
            break

    flagged_rookie = None
    if "is_rookie" in row or "rookie" in row or "is_rookie" in hist or "rookie" in hist:
        flagged_rookie = bool(row.get("is_rookie", row.get("rookie", hist.get("is_rookie", hist.get("rookie")))))

    confidence = "full"
    reasons: List[str] = []

    if prior_gp is None and prior_seasons is None and flagged_rookie is None:
        confidence = "fallback"
        # Conservative: only treat very young first-year profiles as rookies.
        if age is None:
            eligible = False
            reasons.append("Missing rookied history and age; conservative deny.")
        elif age <= 22 and gp >= min_gp:
            eligible = True
            reasons.append("Fallback: age<=22 with meaningful GP and no prior history fields.")
        else:
            eligible = False
            reasons.append("Fallback: insufficient evidence of first NHL season.")
        return {
            "eligible": eligible,
            "confidence": confidence,
            "eligibility_confidence": confidence,
            "details": {
                "gp": gp,
                "min_gp": min_gp,
                "age": age,
                "prior_nhl_gp": prior_gp,
                "prior_nhl_seasons": prior_seasons,
                "is_rookie_flag": flagged_rookie,
                "reasons": reasons,
            },
        }

    # NHL-style thresholds (documented approximation using available fields).
    prior_gp_ok = True if prior_gp is None else prior_gp < 25
    seasons_ok = True if prior_seasons is None else prior_seasons <= 0
    age_ok = True if age is None else age <= 25
    gp_ok = gp >= min_gp

    if flagged_rookie is True:
        eligible = gp_ok and age_ok and prior_gp_ok
        reasons.append("Trusted is_rookie/rookie flag with participation and age checks.")
    elif flagged_rookie is False:
        eligible = False
        reasons.append("is_rookie/rookie flag is false.")
    else:
        eligible = gp_ok and age_ok and prior_gp_ok and seasons_ok
        reasons.append("History-derived first-year checks.")

    if prior_gp is None or prior_seasons is None:
        confidence = "fallback" if confidence == "full" and flagged_rookie is None else confidence

    return {
        "eligible": bool(eligible),
        "confidence": confidence,
        "eligibility_confidence": confidence,
        "details": {
            "gp": gp,
            "min_gp": min_gp,
            "age": age,
            "prior_nhl_gp": prior_gp,
            "prior_nhl_seasons": prior_seasons,
            "is_rookie_flag": flagged_rookie,
            "reasons": reasons,
        },
    }


# ---------------------------------------------------------------------------
# Team context / snapshots
# ---------------------------------------------------------------------------

def _team_name_from_id(teams: Dict[str, Any], tid: str, default: Optional[str] = None) -> str:
    t = teams.get(str(tid))
    if t is None:
        return default or str(tid)
    name = getattr(t, "name", None)
    city = getattr(t, "city", None)
    if city and name:
        return f"{city} {name}"
    if name:
        return str(name)
    return default or str(tid)


def _standing_stats(rec: TeamStandingRecord) -> Dict[str, Any]:
    gp = max(1, _safe_int(getattr(rec, "games_played", None), _safe_int(getattr(rec, "gp", None), 0)) or (_safe_int(rec.wins) + _safe_int(rec.losses) + _safe_int(rec.otl)))
    pts = int(getattr(rec, "points", 0) or 0)
    return {
        "points": pts,
        "pts": pts,
        "wins": int(getattr(rec, "wins", 0) or 0),
        "w": int(getattr(rec, "wins", 0) or 0),
        "losses": int(getattr(rec, "losses", 0) or 0),
        "l": int(getattr(rec, "losses", 0) or 0),
        "otl": int(getattr(rec, "otl", 0) or 0),
        "goals_for": int(getattr(rec, "gf", 0) or 0),
        "goals_against": int(getattr(rec, "ga", 0) or 0),
        "goal_diff": int(rec.goal_diff()),
        "record": f"{int(rec.wins)}-{int(rec.losses)}-{int(rec.otl)}",
        "points_pct": float(pts) / float(2 * gp) if gp else 0.0,
        "gp": gp,
    }


def build_team_context(standings: StandingsTable) -> Dict[str, Dict[str, Any]]:
    tbl = list(standings.league_table() or [])
    if not tbl:
        return {}
    pts_pcts = []
    gds = []
    for rec in tbl:
        st = _standing_stats(rec)
        pts_pcts.append(st["points_pct"])
        gds.append(float(st["goal_diff"]))
    playoff_cut = 16 if len(tbl) >= 16 else max(1, len(tbl) // 2)
    out: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(tbl):
        st = _standing_stats(rec)
        tid = str(rec.team_id)
        out[tid] = {
            **st,
            "league_rank": i + 1,
            "points_pct_norm": percentile_rank(pts_pcts, st["points_pct"]),
            "goal_diff_norm": percentile_rank(gds, float(st["goal_diff"])),
            "playoff_qualified": i < playoff_cut,
            "distance_from_cutoff": float(playoff_cut - (i + 1)),
            "standing_percentile": 1.0 - (float(i) / float(max(1, len(tbl) - 1))),
        }
    return out


def snapshot_row(row: Mapping[str, Any], *, teams: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    snap = dict(row)
    snap["entity_id"] = _pid(row)
    snap["player_id"] = _pid(row)
    snap["team_id"] = _tid(row)
    snap["position"] = _pos(row)
    snap["age"] = _player_age(row, teams)
    snap["gp"] = _gp(row)
    snap["is_rookie"] = bool(row.get("is_rookie", row.get("rookie", False)))
    snap["previous_nhl_gp"] = row.get("previous_nhl_gp", row.get("prior_nhl_gp"))
    snap["is_captain"] = bool(row.get("is_captain", row.get("captain", False)))
    return snap


# ---------------------------------------------------------------------------
# Goalie workload / EN handling
# ---------------------------------------------------------------------------

def goalie_sv_pct(row: Mapping[str, Any]) -> float:
    if row.get("sv_pct") is not None:
        return normalize_percentage(row.get("sv_pct"))
    sa = _safe_int(row.get("shots_against"), 0)
    en = _safe_int(row.get("empty_net_goals"), _safe_int(row.get("en_goals"), 0))
    ga = max(0, _safe_int(row.get("ga"), _safe_int(row.get("goals_against"))) - en)
    saves = _safe_int(row.get("saves"), 0)
    if saves <= 0 and sa > 0:
        saves = max(0, sa - ga)
    denom = saves + ga
    return float(saves) / float(denom) if denom > 0 else 0.0


def goalie_workload_ok(row: Mapping[str, Any], *, season_length: int = 82) -> Tuple[bool, Dict[str, Any]]:
    gp = _gp(row)
    gs = _safe_int(row.get("games_started"), _safe_int(row.get("gs"), gp))
    minutes = _safe_float(row.get("minutes"), _safe_float(row.get("toi_minutes"), _safe_float(row.get("toi"), 0.0)))
    if minutes <= 0 and gp > 0:
        # Derive approximate minutes only from starts when available.
        minutes = float(gs) * 60.0
    shots = _safe_int(row.get("shots_against"), 0)
    team_minutes = _safe_float(row.get("team_goalie_minutes"), float(season_length) * 60.0)
    share = minutes / team_minutes if team_minutes > 0 else 0.0
    min_gs = _season_games_threshold(season_length, 0.30, minimum=20)
    min_minutes = float(min_gs) * 50.0
    ok = (gs >= min_gs) or (minutes >= min_minutes and shots >= min_gs * 22)
    # Pure relief: high appearances, low starts/minutes → not eligible
    if gp >= min_gs and gs < max(8, min_gs // 3) and minutes < min_minutes:
        ok = False
    return ok, {
        "gp": gp,
        "games_started": gs,
        "minutes": minutes,
        "shots_against": shots,
        "team_minutes_share": share,
        "min_gs": min_gs,
        "eligible": ok,
    }


# ---------------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------------

def _component_bundle(
    pool: Sequence[Mapping[str, Any]],
    specs: Mapping[str, Callable[[Mapping[str, Any]], float]],
    weights: Mapping[str, float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], str, Optional[str]]:
    """Normalize each component across the pool, then weighted sum → canonical score."""
    norms: Dict[str, Dict[str, float]] = {k: normalize_pool_metric(pool, fn) for k, fn in specs.items()}
    quality = "full"
    reason = None
    # If all raw values are zero for a heavy component, mark documented_fallback
    scores: Dict[str, float] = {}
    for row in pool:
        pid = _pid(row)
        parts = {k: norms[k].get(pid, 0.0) for k in specs}
        total_w = sum(float(weights.get(k, 0.0)) for k in specs) or 1.0
        score = sum(parts[k] * float(weights.get(k, 0.0)) for k in specs) / total_w
        scores[pid] = score
        # stash components on mutable row copy handled by caller
        row.setdefault("_components", {})
        row["_components"] = parts  # type: ignore[index]
    return { _pid(r): dict(r.get("_components") or {}) for r in pool }, scores, quality, reason


def hart_components_for_row(row: Mapping[str, Any], team_ctx: Mapping[str, Any]) -> Dict[str, float]:
    pts = float(_pts(row))
    gp = max(1, _gp(row))
    production = pts / gp
    two_way = _safe_float(row.get("defense_score"), 0.0) * 0.5 + normalize_percentage(row.get("xgf_pct")) * 50.0
    individual = _safe_float(row.get("war"), _safe_float(row.get("impact_score"), pts * 0.02))
    team = (
        _safe_float(team_ctx.get("points_pct_norm"), 0.5) * 0.55
        + _safe_float(team_ctx.get("goal_diff_norm"), 0.5) * 0.25
        + (0.2 if team_ctx.get("playoff_qualified") else 0.0)
    )
    availability = min(1.0, float(gp) / 70.0)
    return {
        "production_component": production,
        "two_way_component": two_way,
        "individual_value_component": individual,
        "team_context_component": team,
        "availability_component": availability,
    }


def hart_ballot_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None, rank_by_tid: Optional[Mapping[str, int]] = None) -> float:
    """Backward-compatible Hart score using normalized group blend on a singleton+context basis."""
    ctx = dict((team_context_by_tid or {}).get(_tid(row), {}))
    if not ctx and rank_by_tid is not None:
        # Minor fallback only — ordinal converted to percentile-ish soft signal.
        rk = int(rank_by_tid.get(_tid(row), 16) or 16)
        ctx = {"points_pct_norm": max(0.0, 1.0 - (rk - 1) / 31.0), "goal_diff_norm": 0.5, "playoff_qualified": rk <= 16}
    parts = hart_components_for_row(row, ctx)
    # Relative weights after per-group use — production once, no separate goals double-count.
    return (
        parts["production_component"] * 0.34
        + parts["two_way_component"] * 0.18
        + parts["individual_value_component"] * 0.22
        + parts["team_context_component"] * 0.16
        + parts["availability_component"] * 0.10
    )


def norris_ballot_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    ctx = dict((team_context_by_tid or {}).get(_tid(row), {}))
    gp = max(1, _gp(row))
    offensive = min(1.5, float(_pts(row)) / gp) * 0.55 + _safe_float(row.get("offense_score"), 0.0) * 0.01
    # Cap PP influence via even-strength preference when present.
    es_pts = _safe_float(row.get("ev_points"), _safe_float(row.get("es_points"), float(_pts(row)) * 0.7))
    offensive = 0.65 * (es_pts / gp) + 0.35 * offensive
    defensive = _safe_float(row.get("defense_score"), 50.0)
    possession = _safe_float(row.get("possession_score"), normalize_percentage(row.get("xgf_pct")) * 100.0)
    usage = _safe_float(row.get("toi_per_game"), _safe_float(row.get("usage_score"), 20.0))
    discipline = _safe_float(row.get("discipline_score"), 50.0)
    availability = min(1.0, float(gp) / 70.0) * 100.0
    team = _safe_float(ctx.get("points_pct_norm"), 0.5) * 100.0
    return (
        offensive * 18.0
        + defensive * 0.28
        + possession * 0.22
        + usage * 0.12
        + discipline * 0.08
        + availability * 0.06
        + team * 0.06
    )


def selke_ballot_score(row: Mapping[str, Any]) -> float:
    fo_taken = _safe_int(row.get("fo_taken"), _safe_int(row.get("faceoffs_taken"), 0))
    fo_pct = normalize_percentage(row.get("faceoff_pct")) if fo_taken >= 200 else 0.0
    ev_toi = _safe_float(row.get("ev_toi"), _safe_float(row.get("es_toi"), _safe_float(row.get("toi"), 0.0)))
    pk_toi = _safe_float(row.get("pk_toi"), 0.0)
    pk_share = _safe_float(row.get("pk_toi_share"), 0.0)
    if pk_toi <= 0 and pk_share > 0:
        pk_toi = pk_share
    xgf = normalize_percentage(row.get("xgf_pct")) if ev_toi > 0 or _gp(row) >= 40 else 0.5
    takeaways = _safe_float(row.get("takeaways_per_60"), 0.0)
    if _safe_int(row.get("takeaways"), 0) < 10 and takeaways > 0 and _gp(row) < 40:
        takeaways *= 0.5
    defense = _safe_float(row.get("defense_score"), 50.0)
    pk_result = _safe_float(row.get("pk_xga_per_60"), _safe_float(row.get("pk_ga_per_60"), 0.0))
    pk_value = max(0.0, 3.0 - pk_result) if pk_toi > 0 else 0.0
    return (
        defense * 0.40
        + xgf * 25.0
        + fo_pct * 12.0
        + min(1.0, pk_toi / 120.0) * 8.0
        + pk_value * 4.0
        + takeaways * 3.0
    )


def vezina_ballot_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    ctx = dict((team_context_by_tid or {}).get(_tid(row), {}))
    sv = goalie_sv_pct(row)
    hd = normalize_percentage(row.get("high_danger_save_pct")) if row.get("high_danger_save_pct") is not None else sv
    gsax = _safe_float(row.get("gsax"), 0.0)
    workload = _safe_float(row.get("games_started"), _safe_float(row.get("gs"), float(_gp(row))))
    steal = _safe_float(row.get("steal_rate"), _safe_float(row.get("quality_start_pct"), 0.0))
    consistency = _safe_float(row.get("quality_start_pct"), sv)
    # Soft team defence adjustment: harder workload (worse team GD) gets slight lift.
    team_def = 1.0 - _safe_float(ctx.get("goal_diff_norm"), 0.5)
    return (
        gsax * 2.2
        + sv * 40.0
        + hd * 18.0
        + (workload / 70.0) * 12.0
        + normalize_percentage(steal) * 10.0
        + normalize_percentage(consistency) * 8.0
        + team_def * 6.0
        + min(1.0, float(_gp(row)) / 60.0) * 5.0
    )


def lady_byng_score(row: Mapping[str, Any]) -> float:
    gp = max(1, _gp(row))
    pim = _safe_float(row.get("pim"), 0.0)
    return (float(_pts(row)) / gp) * 40.0 + _safe_float(row.get("discipline_score"), 50.0) * 0.4 - (pim / gp) * 6.0


def ted_lindsay_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    # Player-focused: minimize team context vs Hart.
    pts = float(_pts(row))
    gp = max(1, _gp(row))
    return (
        (pts / gp) * 38.0
        + _safe_float(row.get("impact_score"), pts * 0.4) * 0.35
        + _safe_float(row.get("war"), 0.0) * 3.0
        + normalize_percentage(row.get("xgf_pct")) * 10.0
        + min(1.0, float(gp) / 70.0) * 8.0
    )


def calder_position_score(row: Mapping[str, Any], team_context_by_tid: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    if _is_goalie(row):
        return vezina_ballot_score(row, team_context_by_tid) * 0.85
    if _is_defense(row):
        return norris_ballot_score(row, team_context_by_tid) * 0.9
    return (
        hart_ballot_score(row, team_context_by_tid) * 0.55
        + selke_ballot_score(row) * 0.25
        + (float(_pts(row)) / max(1, _gp(row))) * 10.0
    )


def conn_smythe_score(row: Mapping[str, Any], *, champion_id: Optional[str] = None) -> float:
    gp = max(1, _gp(row))
    base = (float(_pts(row)) / gp) * 30.0 + float(_goals(row)) * 1.2
    if _is_goalie(row):
        base = goalie_sv_pct(row) * 55.0 + _safe_float(row.get("gsax"), 0.0) * 2.0 + float(gp) * 1.5
    elif _is_defense(row):
        base += _safe_float(row.get("defense_score"), 40.0) * 0.2
    else:
        base += _safe_float(row.get("defense_score"), 30.0) * 0.12
    base += _safe_float(row.get("elimination_game_points"), 0.0) * 2.5
    base += _safe_float(row.get("clutch_score"), 0.0) * 0.2
    base += _safe_float(row.get("final_round_points"), 0.0) * 2.0
    if champion_id and str(_tid(row)) == str(champion_id):
        base *= 1.18
    return base


# ---------------------------------------------------------------------------
# Ballot simulation
# ---------------------------------------------------------------------------

def simulate_award_ballots(
    scored_rows: Sequence[Tuple[float, Dict[str, Any]]],
    *,
    award_id: str,
    season_seed: Any,
    voter_count: int = VOTER_COUNT,
) -> Dict[str, Any]:
    """
    Deterministic individual ballots across voter archetypes.
    scored_rows: (canonical_score, row) already sorted optional.
    """
    if not scored_rows:
        return {
            "candidates": [],
            "voter_count": 0,
            "margin": 0.0,
            "seed": _seed_int(season_seed, award_id),
        }

    ordered = sorted(scored_rows, key=lambda p: p[0], reverse=True)
    rng = _rng(season_seed, "ballot", award_id)
    tallies: Dict[str, Dict[str, Any]] = {}
    for score, row in ordered:
        pid = _pid(row)
        tallies[pid] = {
            "row": row,
            "canonical_score": float(score),
            "ballot_points": 0.0,
            "first_place_votes": 0,
            "component_scores": dict(row.get("_components") or row.get("component_scores") or {}),
        }

    archetype_bias = {
        "production": 0.12,
        "two_way": 0.08,
        "team_success": 0.10,
        "analytics": 0.09,
        "workload": 0.07,
        "traditional": 0.06,
    }

    for v in range(int(voter_count)):
        arch = VOTER_ARCHETYPES[v % len(VOTER_ARCHETYPES)]
        ranked = []
        for score, row in ordered:
            pid = _pid(row)
            noise = rng.uniform(-0.045, 0.045)
            pref = 0.0
            comps = tallies[pid]["component_scores"]
            if arch == "production":
                pref = _safe_float(comps.get("production_component"), float(score))
            elif arch == "two_way":
                pref = _safe_float(
                    comps.get("two_way_component", comps.get("defensive_value")),
                    float(score),
                )
            elif arch == "team_success":
                pref = _safe_float(comps.get("team_context_component"), float(score))
            elif arch == "analytics":
                pref = _safe_float(
                    comps.get("individual_value_component", comps.get("goals_saved_above_expected")),
                    float(score),
                )
            elif arch == "workload":
                pref = _safe_float(
                    comps.get("availability_component", comps.get("workload")),
                    float(score),
                )
            else:
                pref = float(score)
            adj = float(score) + float(pref) * float(archetype_bias[arch]) + noise * max(0.15, abs(float(score)))
            ranked.append((adj, pid))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for place, (_adj, pid) in enumerate(ranked[: len(BALLOT_POINTS)]):
            pts = BALLOT_POINTS[place]
            tallies[pid]["ballot_points"] += pts
            if place == 0:
                tallies[pid]["first_place_votes"] += 1

    finished = sorted(
        tallies.values(),
        key=lambda c: (c["ballot_points"], c["first_place_votes"], c["canonical_score"]),
        reverse=True,
    )
    for i, c in enumerate(finished):
        c["finish"] = i + 1
    margin = 0.0
    if len(finished) >= 2:
        margin = float(finished[0]["ballot_points"] - finished[1]["ballot_points"])
    return {
        "candidates": finished,
        "voter_count": int(voter_count),
        "margin": margin,
        "seed": _seed_int(season_seed, award_id),
    }


def _candidate_from_tally(
    tally: Mapping[str, Any],
    team_map: Mapping[str, Any],
    *,
    display_metric: str,
    display_value: Any = None,
) -> Dict[str, Any]:
    row = dict(tally.get("row") or {})
    tid = _tid(row)
    finish = _safe_int(tally.get("finish"), 0)
    return {
        "entity_id": _pid(row),
        "player_id": _pid(row),
        "name": str(row.get("name") or ""),
        "team_id": tid,
        "team_name": _team_name_from_id(dict(team_map), tid, str(row.get("team_name") or "")),
        "position": _pos(row),
        "finish": finish,
        "rank": finish,
        "canonical_score": float(tally.get("canonical_score") or 0.0),
        "ballot_points": float(tally.get("ballot_points") or 0.0) if tally.get("ballot_points") is not None else None,
        "first_place_votes": int(tally.get("first_place_votes") or 0) if tally.get("first_place_votes") is not None else None,
        "votes": int(round(float(tally.get("ballot_points") or tally.get("display_value") or 0.0))),
        "display_value": display_value if display_value is not None else tally.get("display_value"),
        "display_metric": display_metric,
        "component_scores": dict(tally.get("component_scores") or {}),
        "eligibility": dict(row.get("_eligibility") or {}),
        "is_winner": finish == 1,
        "points": _pts(row),
        "goals": _goals(row),
        "assists": _safe_int(row.get("a")),
        "gp": _gp(row),
    }


def _rationale_from_components(name: str, comps: Mapping[str, Any], quality: str, fallback_reason: Optional[str]) -> str:
    if not comps:
        base = f"{name} earned the award on the final scoreboard."
    else:
        ranked = sorted(((k, _safe_float(v)) for k, v in comps.items()), key=lambda kv: kv[1], reverse=True)
        top = [k.replace("_", " ").replace(" component", "") for k, _ in ranked[:2]]
        if len(top) == 1:
            base = f"Driven by {top[0]}."
        else:
            base = f"Led by {top[0]} and {top[1]}."
    if quality == "documented_fallback":
        base += f" Calculated with documented fallback ({fallback_reason or 'limited inputs'})."
    elif quality == "unavailable":
        base = fallback_reason or "Required season data was unavailable."
    return base


# ---------------------------------------------------------------------------
# Award builders
# ---------------------------------------------------------------------------

def _unavailable_award(defn: Mapping[str, Any], *, reason: str, season: Any = None) -> Award:
    aid = str(defn["award_id"])
    return Award(
        name=str(defn["name"]),
        award_id=aid,
        official=bool(defn.get("official", True)),
        status="unavailable",
        category=str(defn.get("category") or ""),
        recipient_type=str(defn.get("recipient_type") or "player"),
        display_metric=str(defn.get("display_metric") or ""),
        calculation_quality="unavailable",
        fallback_reason=reason,
        unavailable_reason=reason,
        public_rationale=reason,
        rationale=reason,
        eligibility_summary=reason,
        season=season,
        finalists=[],
        candidates=[],
        winners=[],
        full_results=[],
        result={
            "award_id": aid,
            "name": defn["name"],
            "official": True,
            "status": "unavailable",
            "category": defn.get("category"),
            "recipient_type": defn.get("recipient_type"),
            "winner": None,
            "winners": [],
            "shared": False,
            "finalists": [],
            "full_results": [],
            "display_metric": defn.get("display_metric"),
            "calculation_quality": "unavailable",
            "fallback_reason": reason,
            "public_rationale": reason,
            "unavailable_reason": reason,
            "eligibility_summary": reason,
            "stat_scope": "regular_season",
            "season": season,
        },
    )


def _finalize_player_award(
    defn: Mapping[str, Any],
    full_results: List[Dict[str, Any]],
    *,
    season: Any,
    quality: str,
    fallback_reason: Optional[str],
    eligibility_summary: str,
    stat_scope: str,
    voting: Optional[Dict[str, Any]] = None,
    shared_override: Optional[bool] = None,
) -> Award:
    if not full_results:
        return _unavailable_award(defn, reason="No eligible candidates.", season=season)

    winners = [full_results[0]]
    shared = False
    if defn.get("supports_shared_winners") and len(full_results) > 1:
        a0 = full_results[0]
        a1 = full_results[1]
        if shared_override is True or (
            a0.get("display_value") is not None
            and a0.get("display_value") == a1.get("display_value")
            and a0.get("ballot_points") in (None, a1.get("ballot_points"))
        ):
            # collect equals
            key_fields = ("display_value", "ballot_points", "canonical_score")
            winners = [full_results[0]]
            for cand in full_results[1:]:
                if all(cand.get(k) == full_results[0].get(k) for k in key_fields if full_results[0].get(k) is not None):
                    winners.append(cand)
                else:
                    break
            shared = len(winners) > 1

    for i, cand in enumerate(full_results):
        cand["finish"] = i + 1
        cand["rank"] = i + 1
        cand["is_winner"] = any(_pid(cand) == _pid(w) or cand.get("entity_id") == w.get("entity_id") for w in winners)

    top = winners[0]
    comps = dict(top.get("component_scores") or {})
    rationale = _rationale_from_components(str(defn["name"]), comps, quality, fallback_reason)
    finalists = full_results[: max(1, int(defn.get("finalist_count") or 3))]
    result = {
        "award_id": defn["award_id"],
        "name": defn["name"],
        "official": True,
        "status": "complete",
        "category": defn.get("category"),
        "recipient_type": defn.get("recipient_type"),
        "winner": top,
        "winners": winners,
        "shared": shared,
        "finalists": finalists,
        "full_results": full_results,
        "display_metric": defn.get("display_metric"),
        "calculation_quality": quality,
        "fallback_reason": fallback_reason,
        "public_rationale": rationale,
        "eligibility_summary": eligibility_summary,
        "stat_scope": stat_scope,
        "season": season,
        "voting": voting,
    }
    return Award(
        name=str(defn["name"]),
        award_id=str(defn["award_id"]),
        official=True,
        status="complete",
        category=str(defn.get("category") or ""),
        recipient_type=str(defn.get("recipient_type") or "player"),
        winner_team_id=str(top.get("team_id") or ""),
        winner_name=str(top.get("name") or ""),
        winner_player_id=str(top.get("player_id") or top.get("entity_id") or ""),
        winner_team_name=str(top.get("team_name") or ""),
        finalists=finalists,
        candidates=full_results[:5],
        winners=winners,
        shared=shared,
        full_results=full_results,
        display_metric=str(defn.get("display_metric") or ""),
        calculation_quality=quality,
        fallback_reason=fallback_reason,
        public_rationale=rationale,
        rationale=rationale,
        eligibility_summary=eligibility_summary,
        stat_scope=stat_scope,
        season=season,
        voting=voting,
        winner_stats={
            "points": top.get("points"),
            "goals": top.get("goals"),
            "gp": top.get("gp"),
            "ballot_points": top.get("ballot_points"),
            "first_place_votes": top.get("first_place_votes"),
        },
        result=result,
    )


def _score_pool_with_quality(
    pool: List[Dict[str, Any]],
    score_fn: Callable[[Dict[str, Any]], float],
    required_fields: Sequence[str],
    fallback_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
) -> Tuple[List[Tuple[float, Dict[str, Any]]], str, Optional[str]]:
    if not pool:
        return [], "unavailable", "No eligible candidates."
    missing_counts = 0
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in pool:
        missing = validate_required_award_fields(row, required_fields)
        optional_analytics = [f for f in ("war", "impact_score", "xgf_pct", "gsax", "defense_score") if f in required_fields or True]
        # Only fail hard if NONE of analytics-ish fields exist when required includes them
        hard_missing = [f for f in missing if f in {"gp"}]
        if hard_missing:
            continue
        analytics_present = any(row.get(f) is not None for f in ("war", "impact_score", "gsax", "defense_score", "analytics_rating"))
        quality_row = "full" if analytics_present else "documented_fallback"
        if quality_row == "documented_fallback":
            missing_counts += 1
            s = float(fallback_fn(row)) if fallback_fn else float(score_fn(row))
        else:
            s = float(score_fn(row))
        row["_calc_quality"] = quality_row
        scored.append((s, row))
    if not scored:
        return [], "unavailable", "Candidates missing required participation fields."
    quality = "documented_fallback" if missing_counts == len(scored) else ("documented_fallback" if missing_counts else "full")
    reason = "Missing advanced analytics on one or more candidates; used documented participation fallback." if quality != "full" else None
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored, quality, reason


def _run_ballot_award(
    defn: Mapping[str, Any],
    pool: List[Dict[str, Any]],
    score_fn: Callable[[Dict[str, Any]], float],
    *,
    team_map: Dict[str, Any],
    season_seed: Any,
    season: Any,
    eligibility_summary: str,
    required_fields: Sequence[str],
    fallback_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    stat_scope: str = "regular_season",
) -> Award:
    scored, quality, reason = _score_pool_with_quality(pool, score_fn, required_fields, fallback_fn)
    if quality == "unavailable":
        return _unavailable_award(defn, reason=reason or "Unavailable", season=season)
    # Attach components for rationale/voters when present
    for score, row in scored:
        row.setdefault("_components", row.get("component_scores") or {"canonical": score})
        row["component_scores"] = row["_components"]
    ballot = simulate_award_ballots(scored, award_id=str(defn["award_id"]), season_seed=season_seed)
    full: List[Dict[str, Any]] = []
    for tally in ballot["candidates"]:
        cand = _candidate_from_tally(tally, team_map, display_metric=str(defn["display_metric"]))
        cand["display_value"] = cand.get("ballot_points")
        full.append(cand)
    return _finalize_player_award(
        defn,
        full,
        season=season,
        quality=quality,
        fallback_reason=reason,
        eligibility_summary=eligibility_summary,
        stat_scope=stat_scope,
        voting={
            "voter_count": ballot["voter_count"],
            "margin": ballot["margin"],
            "seed": ballot["seed"],
            "ballot_points_curve": list(BALLOT_POINTS),
        },
    )


def _run_stat_race(
    defn: Mapping[str, Any],
    pool: List[Dict[str, Any]],
    primary: Callable[[Dict[str, Any]], float],
    tiebreak_keys: Sequence[Callable[[Dict[str, Any]], float]],
    *,
    team_map: Dict[str, Any],
    season: Any,
    eligibility_summary: str,
) -> Award:
    if not pool:
        return _unavailable_award(defn, reason="No eligible candidates.", season=season)

    def sort_key(r: Dict[str, Any]) -> Tuple:
        return tuple(fn(r) for fn in (primary, *tiebreak_keys))

    ordered = sorted(pool, key=sort_key, reverse=True)
    full: List[Dict[str, Any]] = []
    for i, row in enumerate(ordered):
        val = primary(row)
        full.append(
            {
                "entity_id": _pid(row),
                "player_id": _pid(row),
                "name": str(row.get("name") or ""),
                "team_id": _tid(row),
                "team_name": _team_name_from_id(team_map, _tid(row), str(row.get("team_name") or "")),
                "position": _pos(row),
                "finish": i + 1,
                "rank": i + 1,
                "canonical_score": float(val),
                "ballot_points": None,
                "first_place_votes": None,
                "votes": int(round(float(val))),
                "display_value": val,
                "display_metric": defn["display_metric"],
                "component_scores": {"race_value": float(val)},
                "eligibility": dict(row.get("_eligibility") or {}),
                "is_winner": i == 0,
                "points": _pts(row),
                "goals": _goals(row),
                "assists": _safe_int(row.get("a")),
                "gp": _gp(row),
            }
        )
    # Shared winners if exact primary matches after tiebreak equivalence
    shared = False
    if defn.get("supports_shared_winners") and len(full) > 1:
        if all(sort_key(ordered[0])[j] == sort_key(ordered[1])[j] for j in range(len(tiebreak_keys) + 1)):
            shared = True
    return _finalize_player_award(
        defn,
        full,
        season=season,
        quality="full",
        fallback_reason=None,
        eligibility_summary=eligibility_summary,
        stat_scope="regular_season",
        shared_override=shared,
    )


# ---------------------------------------------------------------------------
# Eligibility pools
# ---------------------------------------------------------------------------

def eligible_art_ross(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= 1]


def eligible_rocket(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= 1]


def eligible_hart(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.45, minimum=30)
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= need]


def eligible_norris(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.45, minimum=30)
    out = []
    for r in rows:
        if not _is_defense(r) or _gp(r) < need:
            continue
        toi = _safe_float(r.get("toi_per_game"), _safe_float(r.get("toi"), 0.0))
        if toi and toi < 15.0:
            continue
        out.append(r)
    return out


def eligible_selke(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.45, minimum=30)
    out = []
    for r in rows:
        if not _is_forward(r) or _gp(r) < need:
            continue
        ev = _safe_float(r.get("ev_toi"), _safe_float(r.get("toi"), 1.0))
        if ev <= 0:
            continue
        out.append(r)
    return out


def eligible_lady_byng(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    need = _season_games_threshold(season_length, 0.40, minimum=28)
    return [r for r in rows if not _is_goalie(r) and _gp(r) >= need and _pts(r) > 0]


def eligible_vezina(rows: Sequence[Dict[str, Any]], season_length: int) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if not _is_goalie(r):
            continue
        ok, details = goalie_workload_ok(r, season_length=season_length)
        r = dict(r)
        r["_eligibility"] = details
        if ok:
            out.append(r)
    return out


def eligible_calder(
    rows: Sequence[Dict[str, Any]],
    *,
    teams: Optional[Sequence[Any]],
    history_by_player: Optional[Mapping[str, Any]],
    season_length: int,
) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        elig = calder_eligibility(
            r,
            teams=teams,
            history=(history_by_player or {}).get(_pid(r)) if history_by_player else history_by_player,
            season_length=season_length,
        )
        rr = dict(r)
        rr["_eligibility"] = elig
        rr["eligibility_confidence"] = elig.get("confidence")
        if elig.get("eligible"):
            out.append(rr)
    return out


# ---------------------------------------------------------------------------
# Conference champions / Jennings / team awards
# ---------------------------------------------------------------------------

def extract_conference_champions(
    playoff_result: Optional[PlayoffResult],
    team_map: Dict[str, Any],
    *,
    season: Any,
) -> List[Dict[str, Any]]:
    if playoff_result is None:
        return []
    series = list(getattr(playoff_result, "series_list", None) or [])
    if not series:
        return []
    # Highest round_index with a non-null conference before the final (conference None).
    conf_series = [s for s in series if getattr(s, "conference", None)]
    if not conf_series:
        # Fallback: treat Cup finalists as East/West unknown mirrors only if conferences missing.
        return []
    max_round = max(int(getattr(s, "round_index", 0) or 0) for s in conf_series)
    finals = [s for s in conf_series if int(getattr(s, "round_index", 0) or 0) == max_round]
    champs: List[Dict[str, Any]] = []
    for s in finals:
        wid = str(s.winner_id())
        lid = str(s.loser_id())
        conf = str(getattr(s, "conference", "") or "")
        champs.append(
            {
                "conference": conf,
                "team_id": wid,
                "team_name": _team_name_from_id(team_map, wid),
                "final_opponent_id": lid,
                "final_opponent_name": _team_name_from_id(team_map, lid),
                "series_result": s.series_score() if hasattr(s, "series_score") else "",
                "season": season,
                "name": _team_name_from_id(team_map, wid),
                "entity_id": wid,
                "is_winner": True,
            }
        )
    return champs


def compute_jennings(
    standings: StandingsTable,
    goalie_rows: Sequence[Dict[str, Any]],
    team_map: Dict[str, Any],
    *,
    season: Any,
    season_length: int,
) -> Award:
    defn = AWARD_REGISTRY["jennings"]
    tbl = list(standings.league_table() or [])
    if not tbl:
        return _unavailable_award(defn, reason="Standings unavailable for Jennings.", season=season)
    best = sorted(tbl, key=lambda r: (int(getattr(r, "ga", 0) or 0), -int(getattr(r, "points", 0) or 0)))
    team_rec = best[0]
    team_ga = int(getattr(team_rec, "ga", 0) or 0)
    tid = str(team_rec.team_id)
    # Qualifying goalies: >= 25% of team games started/played
    team_gp = max(1, _safe_int(getattr(team_rec, "gp", None), season_length))
    min_apps = max(1, int(math.ceil(team_gp * 0.25)))
    recipients = []
    for g in goalie_rows:
        if str(_tid(g)) != tid:
            continue
        ok, details = goalie_workload_ok(g, season_length=season_length)
        apps = max(_safe_int(g.get("games_started"), 0), _gp(g))
        if apps >= min_apps or ok:
            recipients.append(
                {
                    "entity_id": _pid(g),
                    "player_id": _pid(g),
                    "name": str(g.get("name") or ""),
                    "team_id": tid,
                    "team_name": _team_name_from_id(team_map, tid),
                    "position": "G",
                    "finish": 1,
                    "rank": 1,
                    "canonical_score": float(team_ga),
                    "display_value": team_ga,
                    "display_metric": "Team GA",
                    "component_scores": {"team_goals_against": float(team_ga)},
                    "eligibility": details,
                    "is_winner": True,
                    "qualification_details": {"min_apps": min_apps, "apps": apps},
                    "gp": _gp(g),
                    "points": 0,
                    "goals": 0,
                    "assists": 0,
                    "votes": team_ga,
                }
            )
    if not recipients:
        # Team award with no qualifying goalie still records team winner.
        recipients = [
            {
                "entity_id": tid,
                "name": _team_name_from_id(team_map, tid, team_rec.name),
                "team_id": tid,
                "team_name": _team_name_from_id(team_map, tid, team_rec.name),
                "position": None,
                "finish": 1,
                "canonical_score": float(team_ga),
                "display_value": team_ga,
                "display_metric": "Team GA",
                "is_winner": True,
                "votes": team_ga,
            }
        ]
    award = _finalize_player_award(
        defn,
        recipients,
        season=season,
        quality="full",
        fallback_reason=None,
        eligibility_summary=f"Fewest team goals against ({team_ga}); goalie threshold {min_apps} apps.",
        stat_scope="regular_season",
        shared_override=len(recipients) > 1,
    )
    award.winner_team_id = tid
    award.winner_team_name = _team_name_from_id(team_map, tid, team_rec.name)
    award.winner_name = ", ".join(r["name"] for r in recipients)
    award.winner_stats = {"goals_against": team_ga, "team_goals_against": team_ga}
    if award.result is not None:
        award.result["winner_team"] = {
            "team_id": tid,
            "team_name": award.winner_team_name,
            "team_goals_against": team_ga,
        }
        award.result["recipients"] = recipients
        award.result["team_goals_against"] = team_ga
        award.result["qualification_details"] = {"min_apps": min_apps}
    return award


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute_awards(
    standings: StandingsTable,
    playoff_result: Optional[PlayoffResult],
    teams: List[Any],
    player_season_stats: Optional[List[Dict[str, Any]]] = None,
    *,
    playoff_player_stats: Optional[List[Dict[str, Any]]] = None,
    season_seed: Any = None,
    season_year: Any = None,
    season_length: int = 82,
    history_by_player: Optional[Dict[str, Any]] = None,
) -> Dict[str, Award]:
    awards: Dict[str, Award] = {}
    team_map: Dict[str, Any] = {}
    for t in teams or []:
        tid = getattr(t, "team_id", None)
        if tid is None:
            tid = getattr(t, "id", None)
        if tid is not None:
            team_map[str(tid)] = t

    season = season_year
    team_ctx = build_team_context(standings)
    tbl = list(standings.league_table() or [])

    # Presidents'
    prez = standings.presidents_trophy_winner()
    if prez is not None:
        defn = AWARD_REGISTRY["presidents"]
        name = _team_name_from_id(team_map, str(prez.team_id), prez.name)
        full = []
        for i, rec in enumerate(tbl):
            st = _standing_stats(rec)
            full.append(
                {
                    "entity_id": str(rec.team_id),
                    "name": _team_name_from_id(team_map, str(rec.team_id), rec.name),
                    "team_id": str(rec.team_id),
                    "team_name": _team_name_from_id(team_map, str(rec.team_id), rec.name),
                    "finish": i + 1,
                    "rank": i + 1,
                    "canonical_score": float(st["points"]),
                    "display_value": st["points"],
                    "display_metric": "PTS",
                    "votes": st["points"],
                    "is_winner": i == 0,
                    **st,
                }
            )
        awards[defn["name"]] = _finalize_player_award(
            defn,
            full,
            season=season,
            quality="full",
            fallback_reason=None,
            eligibility_summary="League standings points race.",
            stat_scope="regular_season",
        )
        awards[defn["name"]].winner_team_id = str(prez.team_id)
        awards[defn["name"]].winner_team_name = name
        awards[defn["name"]].winner_name = name
        awards[defn["name"]].winner_stats = _standing_stats(prez)
        awards[defn["name"]].rationale = (
            f"Best regular-season record ({prez.points} pts, {prez.wins}-{prez.losses}-{prez.otl}, "
            f"GD {prez.goal_diff():+d})."
        )
        awards[defn["name"]].public_rationale = awards[defn["name"]].rationale

    # Stanley Cup
    if playoff_result is not None:
        defn = AWARD_REGISTRY["stanley"]
        champ_id = str(playoff_result.champion_id)
        champ_name = _team_name_from_id(team_map, champ_id)
        candidates = []
        for tid in list(playoff_result.finalist_ids or []):
            rec = next((r for r in tbl if str(r.team_id) == str(tid)), None)
            st = _standing_stats(rec) if rec is not None else {}
            candidates.append(
                {
                    "entity_id": str(tid),
                    "name": _team_name_from_id(team_map, str(tid)),
                    "team_id": str(tid),
                    "team_name": _team_name_from_id(team_map, str(tid)),
                    "finish": 1 if str(tid) == champ_id else 2,
                    "canonical_score": 1.0 if str(tid) == champ_id else 0.0,
                    "display_value": "Champion" if str(tid) == champ_id else "Finalist",
                    "display_metric": "Champion",
                    "is_winner": str(tid) == champ_id,
                    "votes": 1 if str(tid) == champ_id else 0,
                    **st,
                }
            )
        candidates.sort(key=lambda c: (not c["is_winner"], c.get("finish", 99)))
        award = _finalize_player_award(
            defn,
            candidates or [
                {
                    "entity_id": champ_id,
                    "name": champ_name,
                    "team_id": champ_id,
                    "team_name": champ_name,
                    "finish": 1,
                    "is_winner": True,
                    "display_value": "Champion",
                    "display_metric": "Champion",
                    "votes": 1,
                    "canonical_score": 1.0,
                }
            ],
            season=season,
            quality="full",
            fallback_reason=None,
            eligibility_summary="Stanley Cup playoff champion.",
            stat_scope="playoffs",
        )
        award.winner_team_id = champ_id
        award.winner_name = champ_name
        award.winner_team_name = champ_name
        award.rationale = "Won the Stanley Cup after navigating the playoff bracket."
        award.public_rationale = award.rationale
        awards[defn["name"]] = award

        # Conference champions
        confs = extract_conference_champions(playoff_result, team_map, season=season)
        cdef = AWARD_REGISTRY["conference_champions"]
        if confs:
            for i, c in enumerate(confs):
                c["finish"] = i + 1
                c["display_value"] = c.get("conference") or "Conference"
                c["display_metric"] = "Champion"
                c["canonical_score"] = 1.0
                c["votes"] = 1
            awards[cdef["name"]] = _finalize_player_award(
                cdef,
                confs,
                season=season,
                quality="full",
                fallback_reason=None,
                eligibility_summary="Conference playoff champions.",
                stat_scope="playoffs",
                shared_override=True,
            )
        else:
            awards[cdef["name"]] = _unavailable_award(
                cdef,
                reason="Conference metadata unavailable on playoff bracket.",
                season=season,
            )

    raw_rows = [dict(r) for r in (player_season_stats or []) if isinstance(r, Mapping)]
    # Assert mixed scopes do not contaminate regular-season awards.
    rows = filter_regular_season_rows(raw_rows)
    # If callers passed only bare rows without scope, filter keeps them (missing → regular).
    if not rows and raw_rows:
        # Rows may have been wrongly tagged; only accept if none were playoff-tagged.
        if not any(_stat_scope(r) in {"playoff", "playoffs"} for r in raw_rows):
            rows = [dict(r) for r in raw_rows]

    snap_rows = [snapshot_row(r, teams=teams) for r in rows]
    skaters = [r for r in snap_rows if not _is_goalie(r)]
    goalies = [r for r in snap_rows if _is_goalie(r)]

    # Art Ross
    awards[AWARD_REGISTRY["art_ross"]["name"]] = _run_stat_race(
        AWARD_REGISTRY["art_ross"],
        eligible_art_ross(skaters, season_length),
        lambda r: float(_pts(r)),
        [lambda r: float(_goals(r)), lambda r: float(_pts(r)) / max(1, _gp(r)), lambda r: -float(_gp(r))],
        team_map=team_map,
        season=season,
        eligibility_summary="Regular-season skater points race.",
    )

    # Rocket
    awards[AWARD_REGISTRY["rocket"]["name"]] = _run_stat_race(
        AWARD_REGISTRY["rocket"],
        eligible_rocket(skaters, season_length),
        lambda r: float(_goals(r)),
        [lambda r: -float(_gp(r)), lambda r: float(_pts(r))],
        team_map=team_map,
        season=season,
        eligibility_summary="Regular-season goals race.",
    )

    # Hart
    awards[AWARD_REGISTRY["hart"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["hart"],
        eligible_hart(skaters, season_length),
        lambda r: hart_ballot_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary=f"Meaningful skater participation (>= {_season_games_threshold(season_length, 0.45, 30)} GP).",
        required_fields=["gp"],
        fallback_fn=lambda r: float(_pts(r)) / max(1, _gp(r)) * 40.0 + min(1.0, _gp(r) / 70.0) * 10.0,
    )

    # Norris
    awards[AWARD_REGISTRY["norris"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["norris"],
        eligible_norris(skaters, season_length),
        lambda r: norris_ballot_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Defencemen with meaningful GP/TOI.",
        required_fields=["gp"],
        fallback_fn=lambda r: float(_pts(r)) / max(1, _gp(r)) * 25.0,
    )

    # Selke
    awards[AWARD_REGISTRY["selke"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["selke"],
        eligible_selke(skaters, season_length),
        selke_ballot_score,
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Forwards with meaningful GP and even-strength usage.",
        required_fields=["gp"],
        fallback_fn=lambda r: _safe_float(r.get("defense_score"), float(_pts(r)) * 0.2),
    )

    # Calder
    calder_pool = eligible_calder(
        snap_rows,
        teams=teams,
        history_by_player=history_by_player,
        season_length=season_length,
    )
    awards[AWARD_REGISTRY["calder"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["calder"],
        calder_pool,
        lambda r: calder_position_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Canonical first-year NHL eligibility (not age-only).",
        required_fields=["gp"],
        fallback_fn=lambda r: float(_pts(r)) / max(1, _gp(r)) * 30.0,
    )

    # Vezina
    awards[AWARD_REGISTRY["vezina"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["vezina"],
        eligible_vezina(goalies, season_length),
        lambda r: vezina_ballot_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Starter-level goalie workload (starts/minutes/shots).",
        required_fields=["gp"],
        fallback_fn=lambda r: goalie_sv_pct(r) * 50.0 + float(_gp(r)),
    )

    # Lady Byng
    awards[AWARD_REGISTRY["lady_byng"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["lady_byng"],
        eligible_lady_byng(skaters, season_length),
        lady_byng_score,
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Meaningful games with offensive contribution and discipline.",
        required_fields=["gp"],
        fallback_fn=lady_byng_score,
    )

    # Ted Lindsay
    awards[AWARD_REGISTRY["ted_lindsay"]["name"]] = _run_ballot_award(
        AWARD_REGISTRY["ted_lindsay"],
        eligible_hart(skaters, season_length),
        lambda r: ted_lindsay_score(r, team_ctx),
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Player-focused outstanding season (distinct from Hart team weighting).",
        required_fields=["gp"],
        fallback_fn=lambda r: float(_pts(r)) / max(1, _gp(r)) * 35.0,
    )

    # Jennings
    awards[AWARD_REGISTRY["jennings"]["name"]] = compute_jennings(
        standings, goalies, team_map, season=season, season_length=season_length
    )

    # Conn Smythe
    po_rows = filter_playoff_rows(list(playoff_player_stats or []))
    if not po_rows and playoff_player_stats:
        # If explicitly provided without scope, treat as playoff.
        po_rows = [dict(r) for r in playoff_player_stats if isinstance(r, Mapping)]
    if playoff_result is None:
        awards[AWARD_REGISTRY["conn_smythe"]["name"]] = Award(
            name=AWARD_REGISTRY["conn_smythe"]["name"],
            award_id="conn_smythe",
            status="pending",
            official=True,
            category="playoff",
            recipient_type="player",
            display_metric="Playoff ballot points",
            calculation_quality="unavailable",
            unavailable_reason="Conn Smythe requires a completed Cup Final.",
            rationale="Conn Smythe requires a completed Cup Final.",
            public_rationale="Conn Smythe requires a completed Cup Final.",
            season=season,
            result={
                "award_id": "conn_smythe",
                "name": AWARD_REGISTRY["conn_smythe"]["name"],
                "status": "pending",
                "official": True,
                "winner": None,
                "winners": [],
                "shared": False,
                "finalists": [],
                "full_results": [],
                "display_metric": "Playoff ballot points",
                "calculation_quality": "unavailable",
                "public_rationale": "Conn Smythe requires a completed Cup Final.",
                "stat_scope": "playoffs",
                "season": season,
            },
        )
    elif not po_rows:
        awards[AWARD_REGISTRY["conn_smythe"]["name"]] = _unavailable_award(
            AWARD_REGISTRY["conn_smythe"],
            reason="Playoff player statistics unavailable.",
            season=season,
        )
    else:
        champ = str(getattr(playoff_result, "champion_id", "") or "")
        need = 1
        pool = [snapshot_row(r, teams=teams) for r in po_rows if _gp(r) >= need]
        awards[AWARD_REGISTRY["conn_smythe"]["name"]] = _run_ballot_award(
            AWARD_REGISTRY["conn_smythe"],
            pool,
            lambda r: conn_smythe_score(r, champion_id=champ),
            team_map=team_map,
            season_seed=season_seed,
            season=season,
            eligibility_summary="Playoff-only participation after Cup Final.",
            required_fields=["gp"],
            fallback_fn=lambda r: float(_pts(r)) + float(_goals(r)),
            stat_scope="playoffs",
        )

    # Masterton / Messier / Jack Adams / All-Star — unavailable unless data exists
    awards[AWARD_REGISTRY["masterton"]["name"]] = _try_masterton(skaters, team_map, season, season_seed)
    awards[AWARD_REGISTRY["messier"]["name"]] = _try_messier(skaters, team_map, season, season_seed)
    awards[AWARD_REGISTRY["jack_adams"]["name"]] = _try_jack_adams(teams, team_ctx, team_map, season, season_seed)
    a1, a2 = _try_all_star_teams(skaters, goalies, team_map, season, season_seed, team_ctx)
    awards[AWARD_REGISTRY["all_star_1"]["name"]] = a1
    awards[AWARD_REGISTRY["all_star_2"]["name"]] = a2

    return awards


def _try_masterton(rows, team_map, season, season_seed) -> Award:
    defn = AWARD_REGISTRY["masterton"]
    pool = []
    for r in rows:
        missing = validate_required_award_fields(r, defn["required_fields"])
        if missing:
            continue
        pool.append(r)
    if not pool:
        return _unavailable_award(
            defn,
            reason="Required injury/recovery fields unavailable (injury_games_missed, games_returned).",
            season=season,
        )
    return _run_ballot_award(
        defn,
        pool,
        lambda r: _safe_float(r.get("injury_games_missed")) * 0.5
        + _safe_float(r.get("games_returned")) * 1.2
        + float(_pts(r)) / max(1, _gp(r)) * 10.0,
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Adversity/return performance using injury fields.",
        required_fields=defn["required_fields"],
    )


def _try_messier(rows, team_map, season, season_seed) -> Award:
    defn = AWARD_REGISTRY["messier"]
    pool = [r for r in rows if not validate_required_award_fields(r, defn["required_fields"])]
    if not pool:
        return _unavailable_award(
            defn,
            reason="Required leadership fields unavailable (is_captain, leadership_score).",
            season=season,
        )
    return _run_ballot_award(
        defn,
        pool,
        lambda r: _safe_float(r.get("leadership_score")) * 1.2
        + (8.0 if r.get("is_captain") else 0.0)
        + _safe_float(r.get("morale"), 0.0) * 0.05
        + float(_pts(r)) / max(1, _gp(r)) * 5.0,
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Leadership/captaincy/morale fields present.",
        required_fields=defn["required_fields"],
    )


def _try_jack_adams(teams, team_ctx, team_map, season, season_seed) -> Award:
    defn = AWARD_REGISTRY["jack_adams"]
    coach_rows = []
    for t in teams or []:
        coach = getattr(t, "coach", None) or getattr(t, "head_coach", None)
        tid = getattr(t, "team_id", getattr(t, "id", None))
        ctx = team_ctx.get(str(tid), {})
        row = {
            "player_id": str(getattr(coach, "id", "") or getattr(t, "coach_id", "") or ""),
            "name": str(getattr(coach, "name", "") or getattr(t, "coach_name", "") or ""),
            "team_id": str(tid or ""),
            "coach_id": str(getattr(coach, "id", "") or getattr(t, "coach_id", "") or ""),
            "expected_points": getattr(t, "expected_points", None),
            "actual_points": ctx.get("points"),
            "gp": 82,
            "position": "Coach",
        }
        if row["coach_id"] and row["expected_points"] is not None and row["actual_points"] is not None:
            coach_rows.append(row)
    if not coach_rows:
        return _unavailable_award(
            defn,
            reason="Required coach analytics unavailable (coach_id, expected_points, actual_points).",
            season=season,
        )
    return _run_ballot_award(
        defn,
        coach_rows,
        lambda r: float(_safe_float(r.get("actual_points")) - _safe_float(r.get("expected_points")))
        + _safe_float(r.get("actual_points")) * 0.05,
        team_map=team_map,
        season_seed=season_seed,
        season=season,
        eligibility_summary="Coach over-performance vs roster expectation.",
        required_fields=defn["required_fields"],
    )


def _try_all_star_teams(skaters, goalies, team_map, season, season_seed, team_ctx) -> Tuple[Award, Award]:
    def pick(pos_pool, n, score_fn):
        ranked = sorted(pos_pool, key=score_fn, reverse=True)
        return ranked[:n]

    centers = [r for r in skaters if _pos(r) in {"C", "CENTER"}] or [r for r in skaters if _is_forward(r)]
    wings = [r for r in skaters if _pos(r) in {"L", "R", "LW", "RW", "W"}] or [r for r in skaters if _is_forward(r)]
    defs = [r for r in skaters if _is_defense(r)]
    if not (centers and wings and defs and goalies):
        reason = "Insufficient positional season data for All-Star Teams."
        return (
            _unavailable_award(AWARD_REGISTRY["all_star_1"], reason=reason, season=season),
            _unavailable_award(AWARD_REGISTRY["all_star_2"], reason=reason, season=season),
        )

    def skor(r):
        return hart_ballot_score(r, team_ctx) if not _is_defense(r) else norris_ballot_score(r, team_ctx)

    first = pick(centers, 1, skor) + pick(wings, 2, skor) + pick(defs, 2, skor) + pick(goalies, 1, lambda r: vezina_ballot_score(r, team_ctx))
    used = {_pid(r) for r in first}
    rest_c = [r for r in centers if _pid(r) not in used]
    rest_w = [r for r in wings if _pid(r) not in used]
    rest_d = [r for r in defs if _pid(r) not in used]
    rest_g = [r for r in goalies if _pid(r) not in used]
    second = pick(rest_c, 1, skor) + pick(rest_w, 2, skor) + pick(rest_d, 2, skor) + pick(rest_g, 1, lambda r: vezina_ballot_score(r, team_ctx))

    def to_award(defn, rows):
        full = []
        for i, r in enumerate(rows):
            full.append(
                {
                    "entity_id": _pid(r),
                    "player_id": _pid(r),
                    "name": str(r.get("name") or ""),
                    "team_id": _tid(r),
                    "team_name": _team_name_from_id(team_map, _tid(r)),
                    "position": _pos(r),
                    "finish": i + 1,
                    "canonical_score": float(skor(r) if not _is_goalie(r) else vezina_ballot_score(r, team_ctx)),
                    "display_value": "Selection",
                    "display_metric": "Selection",
                    "is_winner": True,
                    "votes": 1,
                    "points": _pts(r),
                    "goals": _goals(r),
                    "gp": _gp(r),
                    "component_scores": {},
                    "eligibility": {},
                }
            )
        return _finalize_player_award(
            defn,
            full,
            season=season,
            quality="full",
            fallback_reason=None,
            eligibility_summary="Position-aware season-end All-Star selections.",
            stat_scope="regular_season",
            shared_override=True,
        )

    return to_award(AWARD_REGISTRY["all_star_1"], first), to_award(AWARD_REGISTRY["all_star_2"], second)


def serialize_award(award: Award) -> Dict[str, Any]:
    base = {
        "name": award.name,
        "award_id": award.award_id or NAME_TO_ID.get(award.name, ""),
        "winner_name": award.winner_name,
        "winner_team_id": award.winner_team_id,
        "winner_player_id": award.winner_player_id,
        "winner_team_name": award.winner_team_name,
        "finalists": list(award.finalists or []),
        "candidates": list(award.candidates or []),
        "winner_stats": dict(award.winner_stats or {}),
        "rationale": award.rationale or award.public_rationale,
        "public_rationale": award.public_rationale or award.rationale,
        "winners": list(award.winners or []),
        "shared": bool(award.shared),
        "full_results": list(award.full_results or []),
        "status": award.status,
        "official": award.official,
        "category": award.category,
        "recipient_type": award.recipient_type,
        "display_metric": award.display_metric,
        "calculation_quality": award.calculation_quality,
        "fallback_reason": award.fallback_reason,
        "unavailable_reason": award.unavailable_reason,
        "eligibility_summary": award.eligibility_summary,
        "stat_scope": award.stat_scope,
        "season": award.season,
        "voting": award.voting,
    }
    if award.result:
        base.update({k: v for k, v in award.result.items() if k not in base or base.get(k) in (None, "", [], {})})
        base["result"] = award.result
    return base


def build_awards_payload(
    awards: Mapping[str, Award],
    *,
    season: Any = None,
    season_seed: Any = None,
    season_length: int = 82,
) -> Dict[str, Any]:
    awards_dict = {k: serialize_award(v) for k, v in awards.items()}
    official_results = []
    full_ballots = {}
    team_achievements = []
    all_star_teams: Dict[str, Any] = {}
    reveal = []
    for name, aw in awards.items():
        ser = awards_dict[name]
        aid = ser.get("award_id") or NAME_TO_ID.get(name, "")
        official_results.append(ser)
        if ser.get("full_results"):
            full_ballots[aid or name] = ser["full_results"]
        if aid in {"presidents", "stanley", "conference_champions", "jennings"}:
            team_achievements.append(ser)
        if aid == "all_star_1":
            all_star_teams["first"] = ser
        if aid == "all_star_2":
            all_star_teams["second"] = ser
        if ser.get("status") == "complete" and ser.get("ceremony_enabled", AWARD_REGISTRY.get(aid, {}).get("ceremony_enabled", True)):
            if aid and AWARD_REGISTRY.get(aid, {}).get("ceremony_enabled", True):
                reveal.append(aid)

    # Stable reveal order — lean Awards Night (core hardware only).
    order = [
        "calder",
        "selke",
        "jennings",
        "vezina",
        "norris",
        "rocket",
        "art_ross",
        "hart",
        "conn_smythe",
        "presidents",
        "stanley",
    ]
    reveal_order = [a for a in order if a in reveal] + [a for a in reveal if a not in order]

    return {
        "season": season,
        "status": "complete",
        "official_results": official_results,
        "full_ballots": full_ballots,
        "team_achievements": team_achievements,
        "all_star_teams": all_star_teams,
        "ceremony": {
            "reveal_order": reveal_order,
            "catalog": {k: {"award_id": v["award_id"], "name": v["name"], "display_metric": v["display_metric"], "ceremony_enabled": v["ceremony_enabled"], "official": v["official"]} for k, v in AWARD_REGISTRY.items()},
        },
        "metadata": {
            "computed_at_stage": "post_cup_pre_offseason_mutation",
            "seed": season_seed,
            "season_length": season_length,
            "registry_version": 2,
        },
        # Legacy
        "awards": awards_dict,
        "items": list(awards_dict.values()),
    }


def apply_career_award_history(
    teams: Sequence[Any],
    awards: Mapping[str, Award],
    season: Any,
    *,
    result_id: str,
) -> int:
    """Idempotently append award history onto player objects. Returns writes count."""
    writes = 0
    for _name, award in awards.items():
        if getattr(award, "status", "complete") != "complete":
            continue
        recipients = list(award.winners or [])
        if not recipients and award.winner_player_id:
            recipients = [
                {
                    "player_id": award.winner_player_id,
                    "entity_id": award.winner_player_id,
                    "name": award.winner_name,
                    "team_id": award.winner_team_id,
                    "team_name": award.winner_team_name,
                    "finish": 1,
                    "ballot_points": None,
                    "first_place_votes": None,
                }
            ]
        for rec in recipients:
            pid = str(rec.get("player_id") or rec.get("entity_id") or "")
            if not pid:
                continue
            player = _player_from_rosters(teams, pid)
            if player is None:
                continue
            entry = {
                "award_result_id": f"{result_id}:{award.award_id or NAME_TO_ID.get(award.name, award.name)}:{pid}",
                "award_id": award.award_id or NAME_TO_ID.get(award.name, ""),
                "award_name": award.name,
                "season": season,
                "team_id": rec.get("team_id") or award.winner_team_id,
                "team_name": rec.get("team_name") or award.winner_team_name,
                "position": rec.get("position"),
                "ballot_points": rec.get("ballot_points"),
                "first_place_votes": rec.get("first_place_votes"),
                "finish": rec.get("finish") or 1,
                "winning_stats": dict(award.winner_stats or {}),
                "calculation_quality": award.calculation_quality,
            }
            history = list(getattr(player, "career_awards", None) or [])
            if any(isinstance(h, dict) and h.get("award_result_id") == entry["award_result_id"] for h in history):
                continue
            # Also support string awards_won list for compat without duplicating same season+award
            won = list(getattr(player, "awards_won", None) or [])
            token = f"{entry['award_id']}:{season}"
            if token not in won:
                won.append(token)
                try:
                    player.awards_won = won
                except Exception:
                    pass
            history.append(entry)
            try:
                player.career_awards = history
            except Exception:
                pass
            writes += 1
    return writes


def compute_official_watch_lists(
    rows: Iterable[Mapping[str, Any]],
    *,
    standings: Any = None,
    rank_by_tid: Optional[Mapping[str, int]] = None,
    limit: int = 10,
    season_length: int = 82,
    history_by_player: Optional[Mapping[str, Any]] = None,
    teams: Optional[Sequence[Any]] = None,
    season_seed: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Official Award Watch lists using the same scoring as compute_awards."""
    team_ctx: Dict[str, Dict[str, Any]] = {}
    if standings is not None and hasattr(standings, "league_table"):
        try:
            team_ctx = build_team_context(standings)
        except Exception:
            team_ctx = {}
    if not team_ctx and rank_by_tid:
        for tid, rk in rank_by_tid.items():
            team_ctx[str(tid)] = {
                "points_pct_norm": max(0.0, 1.0 - (int(rk) - 1) / 31.0),
                "goal_diff_norm": 0.5,
                "playoff_qualified": int(rk) <= 16,
            }

    snap = [snapshot_row(r, teams=teams) for r in filter_regular_season_rows(list(rows))]
    if not snap:
        snap = [snapshot_row(r, teams=teams) for r in rows if isinstance(r, Mapping)]

    skaters = [r for r in snap if not _is_goalie(r)]
    goalies = [r for r in snap if _is_goalie(r)]
    out: Dict[str, List[Dict[str, Any]]] = {}

    def pack(award_id: str, pool: List[Dict[str, Any]], score_fn, metric_key: str) -> List[Dict[str, Any]]:
        defn = AWARD_REGISTRY[award_id]
        ranked = sorted(pool, key=lambda r: float(score_fn(r)), reverse=True)[: max(1, int(limit))]
        rows_out = []
        for i, r in enumerate(ranked):
            score = float(score_fn(r))
            rows_out.append(
                {
                    "player_id": _pid(r),
                    "name": r.get("name"),
                    "team_id": _tid(r),
                    "position": _pos(r),
                    "gp": _gp(r),
                    "pts": _pts(r),
                    "g": _goals(r),
                    "award_score": score,
                    "award_name": defn["name"],
                    "award_trophy_key": award_id,
                    "official": True,
                    "watch_type": defn.get("watch_type"),
                    "ceremony_enabled": defn.get("ceremony_enabled"),
                    "display_metric": defn.get("display_metric"),
                    "display_value": score if metric_key == "score" else r.get(metric_key),
                    "calculation_quality": "documented_fallback"
                    if not any(r.get(f) is not None for f in ("war", "impact_score", "gsax", "defense_score"))
                    and award_id in {"hart", "norris", "selke", "vezina", "calder"}
                    else "full",
                    "eligibility_confidence": (r.get("_eligibility") or {}).get("confidence"),
                    "rank": i + 1,
                }
            )
        return rows_out

    out["art_ross"] = pack("art_ross", eligible_art_ross(skaters, season_length), _pts, "pts")
    out["rocket"] = pack("rocket", eligible_rocket(skaters, season_length), _goals, "g")
    out["hart"] = pack("hart", eligible_hart(skaters, season_length), lambda r: hart_ballot_score(r, team_ctx), "score")
    out["norris"] = pack("norris", eligible_norris(skaters, season_length), lambda r: norris_ballot_score(r, team_ctx), "score")
    out["selke"] = pack("selke", eligible_selke(skaters, season_length), selke_ballot_score, "score")
    out["calder"] = pack(
        "calder",
        eligible_calder(snap, teams=teams, history_by_player=history_by_player, season_length=season_length),
        lambda r: calder_position_score(r, team_ctx),
        "score",
    )
    out["vezina"] = pack("vezina", eligible_vezina(goalies, season_length), lambda r: vezina_ballot_score(r, team_ctx), "score")
    out["lady_byng"] = pack("lady_byng", eligible_lady_byng(skaters, season_length), lady_byng_score, "score")
    out["ted_lindsay"] = pack(
        "ted_lindsay",
        eligible_hart(skaters, season_length),
        lambda r: ted_lindsay_score(r, team_ctx),
        "score",
    )
    out["conn_smythe"] = []  # requires playoff scope; filled when playoff rows provided by caller
    out["jennings"] = []
    if standings is not None and hasattr(standings, "league_table"):
        try:
            tbl = list(standings.league_table() or [])
            for i, rec in enumerate(sorted(tbl, key=lambda r: int(getattr(r, "ga", 0) or 0))[:limit]):
                out["jennings"].append(
                    {
                        "team_id": str(rec.team_id),
                        "name": getattr(rec, "name", str(rec.team_id)),
                        "ga": int(getattr(rec, "ga", 0) or 0),
                        "award_score": -float(getattr(rec, "ga", 0) or 0),
                        "award_name": AWARD_REGISTRY["jennings"]["name"],
                        "award_trophy_key": "jennings",
                        "official": True,
                        "watch_type": "official_live_race",
                        "display_metric": "Team GA",
                        "display_value": int(getattr(rec, "ga", 0) or 0),
                        "rank": i + 1,
                        "calculation_quality": "full",
                    }
                )
        except Exception:
            pass
    return out
