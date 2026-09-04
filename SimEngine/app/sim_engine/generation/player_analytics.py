"""
player_analytics.py

Backend analytics engine for NHL Franchise Mode.

Purpose:
- Take real game-ledger stat rows and enrich them with derived analytics.
- Keep game facts separate from formulas.
- Make the stats screen, player screen, storylines, award watches, and team reports smarter.

This file SHOULD calculate:
- points = goals + assists
- goalie save %, GAA, saves, quality-start style values
- per-game stats
- per-60 stats
- shooting %, faceoff %, CF%, FF%, xGF%, GF%, PDO
- shot quality, finishing, regression, defensive impact, special teams impact
- goalie GSAx, GSAA proxy, high-danger save metrics if available
- role/archetype labels
- analytics rating
- award/watch scores
- team aggregate analytics

This file SHOULD NOT:
- choose who scored
- choose who assisted
- simulate games
- overwrite the game ledger
- invent core counting stats when the ledger has none

Source of truth:
- Raw counting stats come from the actual schedule/game ledger.
- This file only normalizes and derives analytics from those raw values.

Recommended backend usage:
- Import enrich_player_rows() for player stats.
- Import build_stats_central_player_payload() for StatsCentral payloads.
- Import build_full_analytics_payload() when you want player + team analytics together.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import math

from app.sim_engine.gameplay.game_analytics_ledger import season_xgf_pct_from_row


# ============================================================
# CONSTANTS
# ============================================================

SECONDS_PER_GAME = 60 * 60
MINUTES_PER_GAME = 60.0
NHL_SEASON_GAMES = 82
MAX_STANDINGS_POINTS = NHL_SEASON_GAMES * 2

DEFAULT_FORWARD_TOI_PER_GAME_SEC = 17 * 60
DEFAULT_DEFENSE_TOI_PER_GAME_SEC = 22 * 60
DEFAULT_GOALIE_TOI_PER_GAME_SEC = 60 * 60

SKATER_POSITIONS = {"C", "LW", "RW", "F", "D"}
FORWARD_POSITIONS = {"C", "LW", "RW", "F"}
DEFENSE_POSITIONS = {"D"}
GOALIE_POSITIONS = {"G", "GOALIE"}

QUALITY_START_SV_PCT = 0.915
BAD_START_SV_PCT = 0.850

LEAGUE_AVG_SV_PCT = 0.898
LEAGUE_AVG_SH_PCT = 0.110
LEAGUE_AVG_HD_SV_PCT = 0.820
LEAGUE_AVG_MD_SV_PCT = 0.900
LEAGUE_AVG_LD_SV_PCT = 0.970

GOALS_PER_WIN = 6.0

IMPACT_ELITE = 76.0
IMPACT_STAR = 66.0
IMPACT_CORE = 54.0
IMPACT_DEPTH = 42.0

REPLACEMENT_POINTS_PER_60 = 0.85
REPLACEMENT_XG_PER_60 = 0.55
REPLACEMENT_XGA_PER_60 = 2.55
REPLACEMENT_FACE_OFF_PCT = 0.475

CAP_SAFE_DEFAULT = 0.0


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        n = float(value)
        if math.isnan(n) or math.isinf(n):
            return float(default)
        return n
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, bool):
            return 1 if value else 0
        n = float(value)
        if math.isnan(n) or math.isinf(n):
            return int(default)
        return int(round(n))
    except Exception:
        return int(default)


def clamp(value: Any, lo: float, hi: float) -> float:
    return max(lo, min(hi, safe_float(value, lo)))


def pct(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    den = safe_float(denominator, 0.0)
    if den <= 0:
        return float(default)
    return safe_float(numerator, 0.0) / den


def pct100(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    return pct(numerator, denominator, default=default) * 100.0


def per_game(value: Any, games_played: Any, default: float = 0.0) -> float:
    gp = safe_float(games_played, 0.0)
    if gp <= 0:
        return float(default)
    return safe_float(value, 0.0) / gp


def per_60(value: Any, toi_sec: Any, default: float = 0.0) -> float:
    sec = safe_float(toi_sec, 0.0)
    if sec <= 0:
        return float(default)
    return safe_float(value, 0.0) * 3600.0 / sec


def per_82(value: Any, games_played: Any, default: float = 0.0) -> float:
    gp = safe_float(games_played, 0.0)
    if gp <= 0:
        return float(default)
    return safe_float(value, 0.0) / gp * NHL_SEASON_GAMES


def round_to(value: Any, places: int = 3) -> float:
    return round(safe_float(value, 0.0), int(places))


def first_present(row: Mapping[str, Any], keys: Sequence[str], default: Any = 0) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return default


def normalize_position(pos: Any) -> str:
    raw = str(pos or "").strip().upper()

    if raw in {"LEFT WING", "LWING", "LEFTWING"}:
        return "LW"
    if raw in {"RIGHT WING", "RWING", "RIGHTWING"}:
        return "RW"
    if raw in {"CENTER", "CENTRE"}:
        return "C"
    if raw in {"DEFENSE", "DEFENCE", "DEFENSEMAN", "DEFENCEMAN"}:
        return "D"
    if raw in {"GOALIE", "GOALTENDER", "NETMINDER"}:
        return "G"

    return raw or "F"


def is_goalie_row(row: Mapping[str, Any]) -> bool:
    pos = normalize_position(first_present(row, ["position", "pos"], ""))
    return pos in GOALIE_POSITIONS


def is_defenseman_row(row: Mapping[str, Any]) -> bool:
    pos = normalize_position(first_present(row, ["position", "pos"], ""))
    return pos in DEFENSE_POSITIONS


def is_forward_row(row: Mapping[str, Any]) -> bool:
    pos = normalize_position(first_present(row, ["position", "pos"], "F"))
    return pos in FORWARD_POSITIONS


def is_skater_row(row: Mapping[str, Any]) -> bool:
    return not is_goalie_row(row)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "y", "rookie"}


def scale_to_100(value: Any, good: float, elite: float, floor: float = 0.0, ceiling: float = 115.0) -> float:
    """
    Generic metric scaler.
    good = roughly average/good.
    elite = elite value.
    """
    v = safe_float(value, 0.0)
    if elite == good:
        return 50.0
    score = ((v - floor) / max(0.0001, elite - floor)) * 100.0
    return clamp(score, 0.0, ceiling)


def inverse_scale_to_100(value: Any, bad: float, elite: float, ceiling: float = 115.0) -> float:
    """
    Lower value is better.
    Example GAA: bad=3.80, elite=2.10.
    """
    v = safe_float(value, bad)
    score = ((bad - v) / max(0.0001, bad - elite)) * 100.0
    return clamp(score, 0.0, ceiling)


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    return pct(numerator, denominator, default=default)


def has_any(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(k in row and row.get(k) is not None for k in keys)


def clean_round_fields(row: MutableMapping[str, Any]) -> None:
    """
    Keep frontend JSON stable and readable.
    Do not round core counting stats.
    """
    integer_like = {
        "gp", "games_played", "g", "goals", "a", "assists", "pts", "points",
        "sog", "shots", "hits", "hit", "blk", "blocks", "pim",
        "wins", "w", "losses", "l", "otl", "sa", "saves", "ga", "so",
    }

    for key, value in list(row.items()):
        if key in integer_like:
            continue
        if isinstance(value, float):
            row[key] = round(value, 4)


# ============================================================
# IDENTITY NORMALIZATION
# ============================================================

def normalize_player_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    pos = normalize_position(first_present(row, ["position", "pos"], "F"))
    player_id = str(first_present(row, ["player_id", "id", "pid"], "") or "")
    name = str(first_present(row, ["name", "player_name", "full_name"], "Unknown Player") or "Unknown Player")
    team_id = str(first_present(row, ["team_id", "team"], "") or "")

    age = safe_int(first_present(row, ["age"], 0), 0)
    handedness = str(first_present(row, ["handedness", "shoots", "catches"], "") or "")

    identity = {
        "player_id": player_id,
        "id": player_id,
        "name": name,
        "player_name": name,
        "team_id": team_id,
        "team": team_id,
        "position": pos,
        "pos": pos,
        "age": age,
        "handedness": handedness,
        "rookie": boolish(first_present(row, ["rookie", "is_rookie"], False)),
        "is_rookie": boolish(first_present(row, ["rookie", "is_rookie"], False)),
        "stat_source": str(first_present(row, ["stat_source"], "game_ledger") or "game_ledger"),
    }
    for key in ("team_name", "team_abbrev", "team_abbr", "team_city", "team_logo_src"):
        val = first_present(row, [key], None)
        if val is not None and str(val).strip():
            identity[key] = str(val).strip()
    return identity


# ============================================================
# RAW SKATER NORMALIZATION
# ============================================================

def normalize_skater_counting_stats(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize skater counting stats from a game-ledger row.

    Does not invent goals/assists.
    If the ledger says zero, we keep zero.
    """
    gp = safe_int(first_present(row, ["gp", "games_played", "games"], 0))
    g = safe_int(first_present(row, ["g", "goals"], 0))
    a = safe_int(first_present(row, ["a", "assists"], 0))

    has_primary_split = first_present(row, ["primary_assists", "primary_a", "a1"], None) is not None
    has_secondary_split = first_present(row, ["secondary_assists", "secondary_a", "a2"], None) is not None
    if has_primary_split or has_secondary_split:
        primary_assists = safe_int(first_present(row, ["primary_assists", "primary_a", "a1"], 0))
        secondary_assists = safe_int(first_present(row, ["secondary_assists", "secondary_a", "a2"], max(0, a - primary_assists)))
    else:
        primary_assists = int(round(a * 0.64))
        secondary_assists = max(0, a - primary_assists)

    pts = g + a
    primary_points = g + primary_assists
    secondary_points = secondary_assists

    sog = safe_int(first_present(row, ["sog", "shots", "shots_on_goal"], 0))
    missed_shots = safe_int(first_present(row, ["missed_shots", "miss"], 0))
    blocked_attempts_for = safe_int(first_present(row, ["blocked_attempts_for", "blocked_shot_attempts_for", "bsf"], 0))
    total_shot_attempts = safe_int(first_present(row, ["total_shots", "shot_attempts"], sog + missed_shots + blocked_attempts_for))

    pim = safe_int(first_present(row, ["pim", "pims", "penalty_minutes"], 0))
    penalties_taken = safe_int(first_present(row, ["penalties_taken", "pen_taken"], 0))
    penalties_drawn = safe_int(first_present(row, ["penalties_drawn", "pen_drawn"], 0))
    minor_penalties = safe_int(first_present(row, ["minor_penalties", "minors"], 0))
    major_penalties = safe_int(first_present(row, ["major_penalties", "majors"], 0))
    misconducts = safe_int(first_present(row, ["misconducts"], 0))
    offensive_zone_penalties = safe_int(first_present(row, ["offensive_zone_penalties", "oz_penalties"], 0))

    hit = safe_int(first_present(row, ["hit", "hits"], 0))
    blk = safe_int(first_present(row, ["blk", "blocks", "blocked_shots"], 0))
    takeaways = safe_int(first_present(row, ["tak", "takeaways", "tk"], 0))
    giveaways = safe_int(first_present(row, ["giv", "giveaways", "gv"], 0))
    interceptions = safe_int(first_present(row, ["interceptions", "ints"], 0))
    stick_checks = safe_int(first_present(row, ["stick_checks"], 0))
    failed_clears = safe_int(first_present(row, ["failed_clears"], 0))
    successful_exits = safe_int(first_present(row, ["successful_exits", "controlled_exits", "zone_exits"], 0))
    failed_exits = safe_int(first_present(row, ["failed_exits"], 0))

    ppg = safe_int(first_present(row, ["ppg", "power_play_goals"], 0))
    ppa = safe_int(first_present(row, ["ppa", "power_play_assists"], 0))
    shg = safe_int(first_present(row, ["shg", "short_handed_goals"], 0))
    sha = safe_int(first_present(row, ["sha", "short_handed_assists"], 0))
    gwg = safe_int(first_present(row, ["gwg", "game_winning_goals"], 0))
    otg = safe_int(first_present(row, ["otg", "overtime_goals"], 0))
    shootout_winners = safe_int(first_present(row, ["shootout_winners", "sow"], 0))

    fow = safe_int(first_present(row, ["fow", "faceoff_wins"], 0))
    fol = safe_int(first_present(row, ["fol", "faceoff_losses"], 0))
    oz_fow = safe_int(first_present(row, ["oz_fow", "offensive_zone_faceoff_wins"], 0))
    oz_fot = safe_int(first_present(row, ["oz_fot", "offensive_zone_faceoffs"], 0))
    dz_fow = safe_int(first_present(row, ["dz_fow", "defensive_zone_faceoff_wins"], 0))
    dz_fot = safe_int(first_present(row, ["dz_fot", "defensive_zone_faceoffs"], 0))
    nz_fow = safe_int(first_present(row, ["nz_fow", "neutral_zone_faceoff_wins"], 0))
    nz_fot = safe_int(first_present(row, ["nz_fot", "neutral_zone_faceoffs"], 0))

    plus_minus = safe_int(first_present(row, ["plus_minus", "+/-", "pm"], 0))

    toi_sec = safe_float(first_present(row, ["toi_sec", "time_on_ice_sec"], 0.0), 0.0)
    if toi_sec <= 0:
        toi_min = safe_float(first_present(row, ["toi", "toi_min", "time_on_ice"], 0.0), 0.0)
        if toi_min > 0:
            # Values <= 45 with games played are almost always ATOI (min/game), not
            # season totals — multiplying only by 60 understates WAR to ~0.
            gp_for_toi = safe_int(first_present(row, ["gp", "games_played"], 0), 0)
            if toi_min <= 45.0 and gp_for_toi > 0:
                toi_sec = toi_min * 60.0 * float(gp_for_toi)
            else:
                toi_sec = toi_min * 60.0

    ev_toi_sec = safe_float(first_present(row, ["ev_toi_sec", "even_strength_toi_sec"], 0.0), 0.0)
    pp_toi_sec = safe_float(first_present(row, ["pp_toi_sec", "power_play_toi_sec"], 0.0), 0.0)
    pk_toi_sec = safe_float(first_present(row, ["pk_toi_sec", "penalty_kill_toi_sec"], 0.0), 0.0)

    if ev_toi_sec <= 0 and toi_sec > 0:
        ev_toi_sec = max(0.0, toi_sec - pp_toi_sec - pk_toi_sec)

    cf = safe_float(first_present(row, ["cf", "corsi_for", "shot_attempts_for"], 0.0), 0.0)
    ca = safe_float(first_present(row, ["ca", "corsi_against", "shot_attempts_against"], 0.0), 0.0)
    ff = safe_float(first_present(row, ["ff", "fenwick_for"], 0.0), 0.0)
    fa = safe_float(first_present(row, ["fa", "fenwick_against"], 0.0), 0.0)

    # Only invent "for" counts when against counts also exist. Otherwise CF%/FF%
    # collapse to 100% (common after light/bulk sims that never write CA/FA).
    if cf <= 0 and ca > 0 and total_shot_attempts > 0:
        cf = float(total_shot_attempts)

    if ff <= 0 and fa > 0 and (sog + missed_shots) > 0:
        ff = float(sog + missed_shots)

    xgf = safe_float(first_present(row, ["xgf", "expected_goals_for", "on_ice_xgf"], 0.0), 0.0)
    xga = safe_float(first_present(row, ["xga", "expected_goals_against", "on_ice_xga"], 0.0), 0.0)
    ixg = safe_float(first_present(row, ["ixg", "individual_xg", "individual_expected_goals", "xg"], 0.0), 0.0)
    xa = safe_float(first_present(row, ["xa", "expected_assists"], 0.0), 0.0)

    # Repair triangular light-path inflation: season iXG/xA were re-added from
    # cumulative SOG/G/A every game (~N²/2), producing ~3000 iXG and WAR ~200.
    if ixg > 0 and sog > 0 and (ixg > sog * 0.45 or (g > 0 and ixg > g * 6.0)):
        ixg = max(float(g) * 0.85, float(sog) * LEAGUE_AVG_SH_PCT)
    elif ixg > 0 and sog <= 0 and g > 0 and ixg > g * 6.0:
        ixg = float(g) * 0.90
    if xa > 0 and a > 0 and xa > a * 2.5:
        xa = float(a) * 0.70
    elif xa > 0 and a <= 0 and xa > 5.0:
        xa = 0.0
    # Display/analytics: whole expected goals / assists (no long decimals).
    if ixg > 0:
        ixg = float(int(round(ixg)))
    if xa > 0:
        xa = float(int(round(xa)))

    gf_on = safe_float(first_present(row, ["gf_on", "on_ice_gf", "goals_for_on_ice"], 0.0), 0.0)
    ga_on = safe_float(first_present(row, ["ga_on", "on_ice_ga", "goals_against_on_ice"], 0.0), 0.0)
    # NHL ledger historically wrote gf_on/ga_on without plus_minus; derive when missing.
    if plus_minus == 0 and (gf_on > 0 or ga_on > 0):
        plus_minus = int(round(gf_on - ga_on))

    on_ice_shots_for = safe_float(first_present(row, ["on_ice_shots_for", "shots_for_on_ice", "sf_on"], 0.0), 0.0)
    on_ice_shots_against = safe_float(first_present(row, ["on_ice_shots_against", "shots_against_on_ice", "sa_on"], 0.0), 0.0)

    if on_ice_shots_for <= 0:
        on_ice_shots_for = safe_float(first_present(row, ["sf", "shots_for"], 0.0), 0.0)
    if on_ice_shots_against <= 0:
        on_ice_shots_against = safe_float(first_present(row, ["sa", "shots_against"], 0.0), 0.0)

    scf = safe_float(first_present(row, ["scf", "scoring_chances_for"], 0.0), 0.0)
    sca = safe_float(first_present(row, ["sca", "scoring_chances_against"], 0.0), 0.0)
    hdcf = safe_float(first_present(row, ["hdcf", "high_danger_chances_for"], 0.0), 0.0)
    hdca = safe_float(first_present(row, ["hdca", "high_danger_chances_against"], 0.0), 0.0)
    hdgf = safe_float(first_present(row, ["hdgf", "high_danger_goals_for"], 0.0), 0.0)
    hdga = safe_float(first_present(row, ["hdga", "high_danger_goals_against"], 0.0), 0.0)

    controlled_entries = safe_int(first_present(row, ["controlled_entries"], 0))
    dump_ins = safe_int(first_present(row, ["dump_ins"], 0))
    failed_entries = safe_int(first_present(row, ["failed_entries"], 0))
    entry_attempts = safe_int(first_present(row, ["entry_attempts"], controlled_entries + dump_ins + failed_entries))
    exit_attempts = safe_int(first_present(row, ["exit_attempts"], successful_exits + failed_exits))

    shot_assists = safe_int(first_present(row, ["shot_assists"], 0))
    scoring_chance_assists = safe_int(first_present(row, ["scoring_chance_assists"], 0))
    high_danger_passes = safe_int(first_present(row, ["high_danger_passes"], 0))
    pass_completion_pct = safe_float(first_present(row, ["pass_completion_pct"], 0.0), 0.0)

    net_front_chances = safe_int(first_present(row, ["net_front_chances"], 0))
    net_front_battles_won = safe_int(first_present(row, ["net_front_battles_won"], 0))
    net_front_battles_total = safe_int(first_present(row, ["net_front_battles_total"], 0))
    board_battles_won = safe_int(first_present(row, ["board_battles_won"], 0))
    board_battles_total = safe_int(first_present(row, ["board_battles_total"], 0))
    missed_hits = safe_int(first_present(row, ["missed_hits"], 0))

    offensive_zone_starts = safe_int(first_present(row, ["offensive_zone_starts", "oz_starts"], 0))
    defensive_zone_starts = safe_int(first_present(row, ["defensive_zone_starts", "dz_starts"], 0))
    neutral_zone_starts = safe_int(first_present(row, ["neutral_zone_starts", "nz_starts"], 0))

    quality_of_competition = safe_float(first_present(row, ["quality_of_competition", "qoc"], 0.0), 0.0)
    quality_of_teammates = safe_float(first_present(row, ["quality_of_teammates", "qot"], 0.0), 0.0)

    team_cf_pct_off = safe_float(first_present(row, ["team_cf_pct_without_player", "team_cf_pct_off"], 0.0), 0.0)
    team_xgf_pct_off = safe_float(first_present(row, ["team_xgf_pct_without_player", "team_xgf_pct_off"], 0.0), 0.0)
    team_gf_pct_off = safe_float(first_present(row, ["team_gf_pct_without_player", "team_gf_pct_off"], 0.0), 0.0)

    clutch_goals = safe_int(first_present(row, ["clutch_goals"], 0))
    clutch_assists = safe_int(first_present(row, ["clutch_assists"], 0))
    clutch_points = safe_int(first_present(row, ["clutch_points"], clutch_goals + clutch_assists))
    clutch_toi_sec = safe_float(first_present(row, ["clutch_toi_sec"], 0.0), 0.0)
    comeback_points = safe_int(first_present(row, ["comeback_points"], 0))

    games_missed = safe_int(first_present(row, ["games_missed"], 0))
    stamina_rating = safe_float(first_present(row, ["stamina", "stamina_rating"], 50.0), 50.0)
    injury_resistance_rating = safe_float(first_present(row, ["injury_resistance", "durability"], 50.0), 50.0)
    previous_injuries = safe_int(first_present(row, ["previous_injuries", "injury_count"], 0))

    cap_hit = safe_float(first_present(row, ["cap_hit", "cap_hit_millions", "salary"], CAP_SAFE_DEFAULT), CAP_SAFE_DEFAULT)
    overall = safe_float(first_present(row, ["overall", "ovr"], 0.0), 0.0)
    potential = safe_float(first_present(row, ["potential"], 0.0), 0.0)

    return {
        "gp": gp,
        "games_played": gp,

        "g": g,
        "goals": g,
        "a": a,
        "assists": a,
        "primary_assists": primary_assists,
        "secondary_assists": secondary_assists,
        "pts": pts,
        "points": pts,
        "primary_points": primary_points,
        "secondary_points": secondary_points,

        "sog": sog,
        "shots": sog,
        "shots_on_goal": sog,
        "missed_shots": missed_shots,
        "blocked_attempts_for": blocked_attempts_for,
        "total_shot_attempts": total_shot_attempts,

        "pim": pim,
        "penalty_minutes": pim,
        "penalties_taken": penalties_taken,
        "penalties_drawn": penalties_drawn,
        "minor_penalties": minor_penalties,
        "major_penalties": major_penalties,
        "misconducts": misconducts,
        "offensive_zone_penalties": offensive_zone_penalties,

        "hit": hit,
        "hits": hit,
        "blk": blk,
        "blocks": blk,
        "blocked_shots": blk,
        "tak": takeaways,
        "takeaways": takeaways,
        "giv": giveaways,
        "giveaways": giveaways,
        "interceptions": interceptions,
        "stick_checks": stick_checks,
        "failed_clears": failed_clears,
        "successful_exits": successful_exits,
        "failed_exits": failed_exits,

        "ppg": ppg,
        "power_play_goals": ppg,
        "ppa": ppa,
        "power_play_assists": ppa,
        "pp_points": ppg + ppa,
        "shg": shg,
        "short_handed_goals": shg,
        "sha": sha,
        "short_handed_assists": sha,
        "sh_points": shg + sha,
        "gwg": gwg,
        "game_winning_goals": gwg,
        "otg": otg,
        "overtime_goals": otg,
        "shootout_winners": shootout_winners,

        "fow": fow,
        "faceoff_wins": fow,
        "fol": fol,
        "faceoff_losses": fol,
        "oz_fow": oz_fow,
        "oz_fot": oz_fot,
        "dz_fow": dz_fow,
        "dz_fot": dz_fot,
        "nz_fow": nz_fow,
        "nz_fot": nz_fot,

        "plus_minus": plus_minus,

        "toi_sec": toi_sec,
        "toi": toi_sec / 60.0 if toi_sec > 0 else 0.0,
        "toi_min": toi_sec / 60.0 if toi_sec > 0 else 0.0,
        "ev_toi_sec": ev_toi_sec,
        "pp_toi_sec": pp_toi_sec,
        "pk_toi_sec": pk_toi_sec,

        "cf": cf,
        "corsi_for": cf,
        "ca": ca,
        "corsi_against": ca,
        "ff": ff,
        "fenwick_for": ff,
        "fa": fa,
        "fenwick_against": fa,

        "xgf": xgf,
        "expected_goals_for": xgf,
        "xga": xga,
        "expected_goals_against": xga,
        "ixg": ixg,
        "individual_xg": ixg,
        "xa": xa,
        "expected_assists": xa,

        "gf_on": gf_on,
        "on_ice_gf": gf_on,
        "ga_on": ga_on,
        "on_ice_ga": ga_on,

        "on_ice_shots_for": on_ice_shots_for,
        "on_ice_shots_against": on_ice_shots_against,

        "scf": scf,
        "scoring_chances_for": scf,
        "sca": sca,
        "scoring_chances_against": sca,
        "hdcf": hdcf,
        "high_danger_chances_for": hdcf,
        "hdca": hdca,
        "high_danger_chances_against": hdca,
        "hdgf": hdgf,
        "high_danger_goals_for": hdgf,
        "hdga": hdga,
        "high_danger_goals_against": hdga,

        "controlled_entries": controlled_entries,
        "dump_ins": dump_ins,
        "failed_entries": failed_entries,
        "entry_attempts": entry_attempts,
        "controlled_exits": successful_exits,
        "exit_attempts": exit_attempts,

        "shot_assists": shot_assists,
        "scoring_chance_assists": scoring_chance_assists,
        "high_danger_passes": high_danger_passes,
        "pass_completion_pct": pass_completion_pct,

        "net_front_chances": net_front_chances,
        "net_front_battles_won": net_front_battles_won,
        "net_front_battles_total": net_front_battles_total,
        "board_battles_won": board_battles_won,
        "board_battles_total": board_battles_total,
        "missed_hits": missed_hits,

        "offensive_zone_starts": offensive_zone_starts,
        "defensive_zone_starts": defensive_zone_starts,
        "neutral_zone_starts": neutral_zone_starts,
        "quality_of_competition": quality_of_competition,
        "quality_of_teammates": quality_of_teammates,
        "team_cf_pct_off": team_cf_pct_off,
        "team_xgf_pct_off": team_xgf_pct_off,
        "team_gf_pct_off": team_gf_pct_off,

        "clutch_goals": clutch_goals,
        "clutch_assists": clutch_assists,
        "clutch_points": clutch_points,
        "clutch_toi_sec": clutch_toi_sec,
        "comeback_points": comeback_points,

        "games_missed": games_missed,
        "stamina_rating": stamina_rating,
        "injury_resistance_rating": injury_resistance_rating,
        "previous_injuries": previous_injuries,

        "cap_hit": cap_hit,
        "overall": overall,
        "ovr": overall,
        "potential": potential,
    }


# ============================================================
# RAW GOALIE NORMALIZATION
# ============================================================

def normalize_goalie_counting_stats(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize goalie counting stats from a game-ledger row.

    Goalie points are forced to 0 unless you intentionally add goalie scoring later.
    """
    gp = safe_int(first_present(row, ["gp", "games_played", "games"], 0))
    starts = safe_int(first_present(row, ["starts", "gs"], gp))
    wins = safe_int(first_present(row, ["w", "wins"], 0))
    losses = safe_int(first_present(row, ["l", "losses"], 0))
    otl = safe_int(first_present(row, ["otl", "ot_losses"], 0))
    decisions = wins + losses + otl

    ga = safe_int(first_present(row, ["ga", "goals_against"], 0))
    sa = safe_int(first_present(row, ["sa", "shots_against"], 0))
    saves = safe_int(first_present(row, ["saves"], 0))

    if saves <= 0 and sa > 0:
        saves = max(0, sa - ga)

    if sa <= 0 and saves > 0:
        sa = saves + ga

    shutouts = safe_int(first_present(row, ["so", "shutouts"], 0))

    toi_sec = safe_float(first_present(row, ["toi_sec", "time_on_ice_sec"], 0.0), 0.0)
    if toi_sec <= 0:
        toi_min = safe_float(first_present(row, ["toi", "toi_min", "time_on_ice"], 0.0), 0.0)
        if toi_min > 0:
            toi_sec = toi_min * 60.0

    if toi_sec <= 0 and gp > 0:
        toi_sec = gp * DEFAULT_GOALIE_TOI_PER_GAME_SEC
    # Light path used to omit per-game toi while still adding GA — GAA exploded (e.g. 30.00).
    elif gp > 0 and toi_sec < float(gp) * 1800.0:
        toi_sec = gp * DEFAULT_GOALIE_TOI_PER_GAME_SEC

    xga = safe_float(first_present(row, ["goalie_xga", "xga", "expected_goals_against"], 0.0), 0.0)
    xga_valid = xga > 0 and sa > 0

    hdsa = safe_int(first_present(row, ["hdsa", "high_danger_shots_against"], 0))
    hdga = safe_int(first_present(row, ["hdga", "high_danger_goals_against"], 0))
    hdsaves = safe_int(first_present(row, ["hdsaves", "high_danger_saves"], max(0, hdsa - hdga) if hdsa > 0 else 0))

    mdsa = safe_int(first_present(row, ["mdsa", "medium_danger_shots_against"], 0))
    mdga = safe_int(first_present(row, ["mdga", "medium_danger_goals_against"], 0))
    mdsaves = safe_int(first_present(row, ["mdsaves", "medium_danger_saves"], max(0, mdsa - mdga) if mdsa > 0 else 0))

    ldsa = safe_int(first_present(row, ["ldsa", "low_danger_shots_against"], 0))
    ldga = safe_int(first_present(row, ["ldga", "low_danger_goals_against"], 0))
    ldsaves = safe_int(first_present(row, ["ldsaves", "low_danger_saves"], max(0, ldsa - ldga) if ldsa > 0 else 0))

    rebounds_allowed = safe_int(first_present(row, ["rebounds_allowed"], 0))
    quality_starts = safe_int(first_present(row, ["quality_starts"], 0))
    bad_starts = safe_int(first_present(row, ["bad_starts"], 0))
    steal_games = safe_int(first_present(row, ["steal_games", "goalie_steals"], 0))

    clutch_saves = safe_int(first_present(row, ["clutch_saves"], 0))
    clutch_shots_against = safe_int(first_present(row, ["clutch_shots_against"], 0))

    sv_pct = pct(saves, sa, default=0.0)
    gaa = safe_float(ga * 3600.0 / toi_sec, 0.0) if toi_sec > 0 else 0.0

    if quality_starts <= 0 and starts > 0:
        # Soft map: .875 SV% → ~0% QS, .930 → ~75% QS (covers below-.915 starters).
        qs_rate = max(0.0, min(0.75, (sv_pct - 0.875) / 0.055))
        quality_starts = int(round(starts * qs_rate)) if qs_rate > 0 else 0

    if bad_starts <= 0 and starts > 0:
        bad_rate = max(0.0, min(0.45, (0.900 - sv_pct) / 0.040)) if sv_pct < 0.900 else 0.0
        bad_starts = int(round(starts * bad_rate)) if bad_rate > 0 else 0

    return {
        "gp": gp,
        "games_played": gp,
        "starts": starts,
        "w": wins,
        "wins": wins,
        "l": losses,
        "losses": losses,
        "otl": otl,
        "ot_losses": otl,
        "decisions": decisions,

        "ga": ga,
        "goals_against": ga,
        "sa": sa,
        "shots_against": sa,
        "saves": saves,
        "so": shutouts,
        "shutouts": shutouts,

        "sv_pct": sv_pct,
        "save_pct": sv_pct,
        "sv%": sv_pct,
        "gaa": gaa,

        "toi_sec": toi_sec,
        "toi": toi_sec / 60.0 if toi_sec > 0 else 0.0,
        "toi_min": toi_sec / 60.0 if toi_sec > 0 else 0.0,

        "xga": xga,
        "expected_goals_against": xga,
        "xga_valid": xga_valid,

        "hdsa": hdsa,
        "high_danger_shots_against": hdsa,
        "hdga": hdga,
        "high_danger_goals_against": hdga,
        "hdsaves": hdsaves,
        "high_danger_saves": hdsaves,

        "mdsa": mdsa,
        "medium_danger_shots_against": mdsa,
        "mdga": mdga,
        "medium_danger_goals_against": mdga,
        "mdsaves": mdsaves,
        "medium_danger_saves": mdsaves,

        "ldsa": ldsa,
        "low_danger_shots_against": ldsa,
        "ldga": ldga,
        "low_danger_goals_against": ldga,
        "ldsaves": ldsaves,
        "low_danger_saves": ldsaves,

        "rebounds_allowed": rebounds_allowed,
        "quality_starts": quality_starts,
        "bad_starts": bad_starts,
        "steal_games": steal_games,
        "goalie_steals": steal_games,

        "clutch_saves": clutch_saves,
        "clutch_shots_against": clutch_shots_against,

        "g": 0,
        "goals": 0,
        "a": 0,
        "assists": 0,
        "pts": 0,
        "points": 0,
    }


# ============================================================
# SKATER RATE ANALYTICS
# ============================================================

def calculate_skater_rates(row: Mapping[str, Any]) -> Dict[str, Any]:
    gp = safe_int(first_present(row, ["gp", "games_played"], 0))
    g = safe_int(first_present(row, ["g", "goals"], 0))
    a = safe_int(first_present(row, ["a", "assists"], 0))
    has_primary_split = first_present(row, ["primary_assists"], None) is not None
    has_secondary_split = first_present(row, ["secondary_assists"], None) is not None
    if has_primary_split or has_secondary_split:
        primary_assists = safe_int(first_present(row, ["primary_assists"], 0))
        secondary_assists = safe_int(first_present(row, ["secondary_assists"], max(0, a - primary_assists)))
    else:
        primary_assists = int(round(a * 0.64))
        secondary_assists = max(0, a - primary_assists)
    pts = g + a

    sog = safe_int(first_present(row, ["sog", "shots"], 0))
    missed_shots = safe_int(first_present(row, ["missed_shots"], 0))
    hit = safe_int(first_present(row, ["hit", "hits"], 0))
    blk = safe_int(first_present(row, ["blk", "blocks"], 0))
    pim = safe_int(first_present(row, ["pim", "penalty_minutes"], 0))
    takeaways = safe_int(first_present(row, ["tak", "takeaways"], 0))
    giveaways = safe_int(first_present(row, ["giv", "giveaways"], 0))
    penalties_taken = safe_int(first_present(row, ["penalties_taken"], 0))
    penalties_drawn = safe_int(first_present(row, ["penalties_drawn"], 0))

    toi_sec = safe_float(first_present(row, ["toi_sec"], 0.0), 0.0)
    pp_toi_sec = safe_float(first_present(row, ["pp_toi_sec"], 0.0), 0.0)
    pk_toi_sec = safe_float(first_present(row, ["pk_toi_sec"], 0.0), 0.0)

    cf = safe_float(first_present(row, ["cf", "corsi_for"], 0.0), 0.0)
    ca = safe_float(first_present(row, ["ca", "corsi_against"], 0.0), 0.0)
    ff = safe_float(first_present(row, ["ff", "fenwick_for"], 0.0), 0.0)
    fa = safe_float(first_present(row, ["fa", "fenwick_against"], 0.0), 0.0)
    xgf = safe_float(first_present(row, ["xgf", "expected_goals_for"], 0.0), 0.0)
    xga_raw = first_present(row, ["xga", "expected_goals_against"], None)
    xga = safe_float(xga_raw, 0.0)
    xga_valid = xga_raw is not None and xga > 0
    ixg = safe_float(first_present(row, ["ixg", "individual_xg"], 0.0), 0.0)
    xa = safe_float(first_present(row, ["xa", "expected_assists"], 0.0), 0.0)

    gf_on = safe_float(first_present(row, ["gf_on", "on_ice_gf"], 0.0), 0.0)
    ga_on = safe_float(first_present(row, ["ga_on", "on_ice_ga"], 0.0), 0.0)
    on_ice_shots_for = safe_float(first_present(row, ["on_ice_shots_for"], 0.0), 0.0)
    on_ice_shots_against = safe_float(first_present(row, ["on_ice_shots_against"], 0.0), 0.0)

    scf = safe_float(first_present(row, ["scf", "scoring_chances_for"], 0.0), 0.0)
    sca = safe_float(first_present(row, ["sca", "scoring_chances_against"], 0.0), 0.0)
    hdcf = safe_float(first_present(row, ["hdcf", "high_danger_chances_for"], 0.0), 0.0)
    hdca = safe_float(first_present(row, ["hdca", "high_danger_chances_against"], 0.0), 0.0)
    hdgf = safe_float(first_present(row, ["hdgf", "high_danger_goals_for"], 0.0), 0.0)
    hdga = safe_float(first_present(row, ["hdga", "high_danger_goals_against"], 0.0), 0.0)

    fow = safe_int(first_present(row, ["fow", "faceoff_wins"], 0))
    fol = safe_int(first_present(row, ["fol", "faceoff_losses"], 0))
    oz_fow = safe_int(first_present(row, ["oz_fow"], 0))
    oz_fot = safe_int(first_present(row, ["oz_fot"], 0))
    dz_fow = safe_int(first_present(row, ["dz_fow"], 0))
    dz_fot = safe_int(first_present(row, ["dz_fot"], 0))
    nz_fow = safe_int(first_present(row, ["nz_fow"], 0))
    nz_fot = safe_int(first_present(row, ["nz_fot"], 0))

    pp_points = safe_int(first_present(row, ["pp_points"], 0))
    sh_points = safe_int(first_present(row, ["sh_points"], 0))

    shot_assists = safe_int(first_present(row, ["shot_assists"], 0))
    scoring_chance_assists = safe_int(first_present(row, ["scoring_chance_assists"], 0))
    high_danger_passes = safe_int(first_present(row, ["high_danger_passes"], 0))

    successful_exits = safe_int(first_present(row, ["successful_exits", "controlled_exits"], 0))
    failed_exits = safe_int(first_present(row, ["failed_exits"], 0))
    controlled_entries = safe_int(first_present(row, ["controlled_entries"], 0))
    dump_ins = safe_int(first_present(row, ["dump_ins"], 0))
    failed_entries = safe_int(first_present(row, ["failed_entries"], 0))
    entry_attempts = safe_int(first_present(row, ["entry_attempts"], controlled_entries + dump_ins + failed_entries))
    exit_attempts = safe_int(first_present(row, ["exit_attempts"], successful_exits + failed_exits))

    offensive_zone_starts = safe_int(first_present(row, ["offensive_zone_starts"], 0))
    defensive_zone_starts = safe_int(first_present(row, ["defensive_zone_starts"], 0))

    clutch_points = safe_int(first_present(row, ["clutch_points"], 0))
    clutch_toi_sec = safe_float(first_present(row, ["clutch_toi_sec"], 0.0), 0.0)

    sh_pct = pct(g, sog, default=0.0)
    expected_sh_pct = pct(ixg, sog, default=0.0)
    shooting_pct_above_expected = sh_pct - expected_sh_pct if sog > 0 and ixg > 0 else 0.0

    # Share metrics require both sides. Never emit 100% from one-sided samples.
    cf_pct = pct(cf, cf + ca, default=0.0) if cf > 0 and ca > 0 else None
    ff_pct = pct(ff, ff + fa, default=0.0) if ff > 0 and fa > 0 else None
    xgf_pct = season_xgf_pct_from_row(row)
    gf_pct = pct(gf_on, gf_on + ga_on, default=0.0) if gf_on > 0 and ga_on > 0 else None

    scf_pct = pct(scf, scf + sca, default=0.0) if scf > 0 and sca > 0 else None
    hdcf_pct = pct(hdcf, hdcf + hdca, default=0.0) if hdcf > 0 and hdca > 0 else None
    hdgf_pct = pct(hdgf, hdgf + hdga, default=0.0) if hdgf > 0 and hdga > 0 else None

    fo_pct = pct(fow, fow + fol, default=0.0)
    oz_fo_pct = pct(oz_fow, oz_fot, default=0.0)
    dz_fo_pct = pct(dz_fow, dz_fot, default=0.0)
    nz_fo_pct = pct(nz_fow, nz_fot, default=0.0)

    on_ice_sh_pct = pct(gf_on, on_ice_shots_for, default=0.0)
    on_ice_sv_pct = 1.0 - pct(ga_on, on_ice_shots_against, default=0.0) if on_ice_shots_against > 0 else 0.0
    pdo_valid = on_ice_shots_for > 0 and on_ice_shots_against > 0
    pdo = round((on_ice_sh_pct + on_ice_sv_pct) * 100.0, 1) if pdo_valid else None

    total_points = max(1, pts)
    primary_point_pct = pct(g + primary_assists, total_points, default=0.0)

    oz_dz_total = offensive_zone_starts + defensive_zone_starts
    offensive_zone_start_pct = pct(offensive_zone_starts, oz_dz_total, default=0.0)
    defensive_zone_start_pct = pct(defensive_zone_starts, oz_dz_total, default=0.0)

    defensive_actions = (
        hit
        + blk
        + takeaways
        + safe_int(first_present(row, ["interceptions"], 0))
        + safe_int(first_present(row, ["stick_checks"], 0))
    )
    defensive_mistakes = (
        giveaways
        + failed_exits
        + safe_int(first_present(row, ["failed_clears"], 0))
        + penalties_taken
    )

    penalty_differential = penalties_drawn - penalties_taken
    pim_differential = safe_float(first_present(row, ["pim_drawn"], 0.0), 0.0) - pim

    individual_shot_quality = pct(ixg, sog, default=0.0)
    average_shot_quality_for = pct(xgf, cf, default=0.0)
    average_shot_quality_against = pct(xga, ca, default=0.0)
    shot_quality_differential = average_shot_quality_for - average_shot_quality_against

    finishing = float(int(round(g - ixg))) if ixg > 0 else 0.0
    finishing_per_60 = per_60(finishing, toi_sec)
    finishing_efficiency = pct(g, ixg, default=0.0) if ixg > 0 else 0.0

    assists_above_expected = float(int(round(a - xa))) if xa > 0 else 0.0
    playmaking_efficiency = pct(a, xa, default=0.0) if xa > 0 else 0.0

    transition_success_rate = pct(
        controlled_entries + successful_exits,
        entry_attempts + exit_attempts,
        default=0.0,
    )

    transition_impact = controlled_entries + successful_exits - failed_entries - failed_exits
    transition_impact_per_60 = per_60(transition_impact, toi_sec)

    return {
        "g_per_game": per_game(g, gp),
        "goals_per_game": per_game(g, gp),
        "a_per_game": per_game(a, gp),
        "assists_per_game": per_game(a, gp),
        "pts_per_game": per_game(pts, gp),
        "points_per_game": per_game(pts, gp),
        "p_per_game": per_game(pts, gp),

        "goals_per_82": per_82(g, gp),
        "assists_per_82": per_82(a, gp),
        "points_per_82": per_82(pts, gp),

        "sog_per_game": per_game(sog, gp),
        "shots_per_game": per_game(sog, gp),
        "hit_per_game": per_game(hit, gp),
        "hits_per_game": per_game(hit, gp),
        "blk_per_game": per_game(blk, gp),
        "blocks_per_game": per_game(blk, gp),
        "pim_per_game": per_game(pim, gp),

        "g_per_60": per_60(g, toi_sec),
        "goals_per_60": per_60(g, toi_sec),
        "a_per_60": per_60(a, toi_sec),
        "assists_per_60": per_60(a, toi_sec),
        "primary_assists_per_60": per_60(primary_assists, toi_sec),
        "secondary_assists_per_60": per_60(secondary_assists, toi_sec),
        "pts_per_60": per_60(pts, toi_sec),
        "points_per_60": per_60(pts, toi_sec),
        "p_per_60": per_60(pts, toi_sec),

        "sog_per_60": per_60(sog, toi_sec),
        "shots_per_60": per_60(sog, toi_sec),
        "missed_shots_per_60": per_60(missed_shots, toi_sec),
        "hit_per_60": per_60(hit, toi_sec),
        "hits_per_60": per_60(hit, toi_sec),
        "blk_per_60": per_60(blk, toi_sec),
        "blocks_per_60": per_60(blk, toi_sec),
        "pim_per_60": per_60(pim, toi_sec),
        "takeaways_per_60": per_60(takeaways, toi_sec),
        "giveaways_per_60": per_60(giveaways, toi_sec),

        "pp_points_per_60": per_60(pp_points, pp_toi_sec if pp_toi_sec > 0 else toi_sec),
        "sh_points_per_60": per_60(sh_points, pk_toi_sec if pk_toi_sec > 0 else toi_sec),
        "pk_toi_share": pct(pk_toi_sec, toi_sec, default=0.0),
        "pp_toi_share": pct(pp_toi_sec, toi_sec, default=0.0),

        "shooting_pct": sh_pct,
        "sh_pct": sh_pct,
        "expected_shooting_pct": expected_sh_pct,
        "shooting_pct_above_expected": shooting_pct_above_expected,

        "fo_pct": fo_pct,
        "faceoff_pct": fo_pct,
        "offensive_zone_faceoff_pct": oz_fo_pct,
        "defensive_zone_faceoff_pct": dz_fo_pct,
        "neutral_zone_faceoff_pct": nz_fo_pct,

        "cf_pct": cf_pct,
        "corsi_pct": cf_pct,
        "corsi_for_pct": cf_pct,
        "cf_percentage": cf_pct,
        "ff_pct": ff_pct,
        "fenwick_pct": ff_pct,
        "fenwick_for_pct": ff_pct,
        "xgf_pct": xgf_pct,
        "expected_goals_pct": xgf_pct,
        "expected_goals_for_pct": xgf_pct,
        "gf_pct": gf_pct,
        "goal_share": gf_pct,
        "on_ice_gf_pct": gf_pct,

        "scf_pct": scf_pct,
        "scoring_chance_pct": scf_pct,
        "hdcf_pct": hdcf_pct,
        "high_danger_chance_pct": hdcf_pct,
        "hdgf_pct": hdgf_pct,
        "high_danger_goal_share": hdgf_pct,

        "xg_diff": xgf - xga,
        "expected_goal_differential": xgf - xga,
        "xg_diff_per_60": per_60(xgf - xga, toi_sec),
        "corsi_diff": cf - ca,
        "corsi_differential": cf - ca,
        "fenwick_diff": ff - fa,
        "fenwick_differential": ff - fa,
        "goal_differential_on_ice": gf_on - ga_on,
        "goal_differential_per_60": per_60(gf_on - ga_on, toi_sec),

        "ixg_per_60": per_60(ixg, toi_sec),
        "individual_xg_per_60": per_60(ixg, toi_sec),
        "xgf_per_60": per_60(xgf, toi_sec),
        "xga_per_60": per_60(xga, toi_sec),
        "xa_per_60": per_60(xa, toi_sec),
        "expected_assists_per_60": per_60(xa, toi_sec),

        "goals_above_expected": finishing,
        "finishing": finishing,
        "finishing_per_60": finishing_per_60,
        "finishing_efficiency": finishing_efficiency,
        "assists_above_expected": assists_above_expected,
        "playmaking_efficiency": playmaking_efficiency,

        "primary_point_pct": primary_point_pct,
        "primary_point_percentage": primary_point_pct,

        "pdo": pdo,
        "pdo_valid": pdo_valid,
        "on_ice_sh_pct": on_ice_sh_pct,
        "on_ice_sv_pct": on_ice_sv_pct,

        "average_shot_quality": average_shot_quality_for,
        "average_shot_quality_for": average_shot_quality_for,
        "average_shot_quality_against": average_shot_quality_against,
        "shot_quality_differential": shot_quality_differential,
        "individual_shot_quality": individual_shot_quality,

        "shot_assists_per_60": per_60(shot_assists, toi_sec),
        "scoring_chance_assists_per_60": per_60(scoring_chance_assists, toi_sec),
        "high_danger_passes_per_60": per_60(high_danger_passes, toi_sec),

        "controlled_entry_rate": pct(controlled_entries, entry_attempts, default=0.0),
        "dump_in_rate": pct(dump_ins, entry_attempts, default=0.0),
        "failed_entry_rate": pct(failed_entries, entry_attempts, default=0.0),
        "controlled_exit_rate": pct(successful_exits, exit_attempts, default=0.0),
        "failed_exit_rate": pct(failed_exits, exit_attempts, default=0.0),
        "transition_success_rate": transition_success_rate,
        "transition_impact": transition_impact,
        "transition_impact_per_60": transition_impact_per_60,

        "defensive_actions": defensive_actions,
        "defensive_actions_per_60": per_60(defensive_actions, toi_sec),
        "defensive_mistakes": defensive_mistakes,
        "defensive_mistakes_per_60": per_60(defensive_mistakes, toi_sec),
        "defensive_impact_raw": defensive_actions - defensive_mistakes,
        "defensive_impact_per_60": per_60(defensive_actions - defensive_mistakes, toi_sec),

        "shot_attempts_against_per_60": per_60(ca, toi_sec),
        "shots_against_per_60": per_60(on_ice_shots_against, toi_sec),
        "high_danger_chances_against_per_60": per_60(hdca, toi_sec),

        "penalty_differential": penalty_differential,
        "penalty_differential_per_60": per_60(penalty_differential, toi_sec),
        "pim_differential": pim_differential,
        "pim_differential_per_60": per_60(pim_differential, toi_sec),
        "penalties_taken_per_60": per_60(penalties_taken, toi_sec),
        "penalties_drawn_per_60": per_60(penalties_drawn, toi_sec),

        "offensive_zone_start_pct": offensive_zone_start_pct,
        "defensive_zone_start_pct": defensive_zone_start_pct,
        "deployment_difficulty": defensive_zone_start_pct + safe_float(first_present(row, ["quality_of_competition"], 0.0), 0.0) - offensive_zone_start_pct,

        "clutch_points_per_60": per_60(clutch_points, clutch_toi_sec if clutch_toi_sec > 0 else toi_sec),
    }


# ============================================================
# SKATER COMPOSITE SCORES
# ============================================================

def calculate_skater_component_scores(row: Mapping[str, Any]) -> Dict[str, Any]:
    pos = normalize_position(first_present(row, ["position", "pos"], "F"))
    is_d = pos == "D"

    gp = safe_int(first_present(row, ["gp", "games_played"], 0))
    pts_per_game = safe_float(first_present(row, ["pts_per_game", "points_per_game"], 0.0), 0.0)
    g60 = safe_float(first_present(row, ["g_per_60", "goals_per_60"], 0.0), 0.0)
    p60 = safe_float(first_present(row, ["pts_per_60", "points_per_60"], 0.0), 0.0)
    shots60 = safe_float(first_present(row, ["shots_per_60", "sog_per_60"], 0.0), 0.0)
    primary_a60 = safe_float(first_present(row, ["primary_assists_per_60"], 0.0), 0.0)
    secondary_a60 = safe_float(first_present(row, ["secondary_assists_per_60"], 0.0), 0.0)
    ixg60 = safe_float(first_present(row, ["ixg_per_60", "individual_xg_per_60"], 0.0), 0.0)
    xa60 = safe_float(first_present(row, ["xa_per_60", "expected_assists_per_60"], 0.0), 0.0)
    finishing60 = safe_float(first_present(row, ["finishing_per_60"], 0.0), 0.0)

    cf_pct_raw = first_present(row, ["cf_pct", "corsi_pct"], None)
    xgf_pct_raw = first_present(row, ["xgf_pct"], None)
    gf_pct_raw = first_present(row, ["gf_pct", "goal_share"], None)
    ff_pct_raw = first_present(row, ["ff_pct"], None)
    cf_pct = safe_float(cf_pct_raw, 0.50) if cf_pct_raw is not None else 0.50
    xgf_pct = safe_float(xgf_pct_raw, 0.50) if xgf_pct_raw is not None else 0.50
    ff_pct = safe_float(ff_pct_raw, 0.50) if ff_pct_raw is not None else 0.50
    gf_on_cs = safe_float(first_present(row, ["gf_on", "on_ice_gf"], 0.0), 0.0)
    ga_on_cs = safe_float(first_present(row, ["ga_on", "on_ice_ga"], 0.0), 0.0)
    if gf_on_cs + ga_on_cs > 0:
        gf_pct = gf_on_cs / (gf_on_cs + ga_on_cs)
    elif gf_pct_raw is not None:
        gf_pct = safe_float(gf_pct_raw, 0.50)
    else:
        gf_pct = 0.50

    tak60 = safe_float(first_present(row, ["takeaways_per_60"], 0.0), 0.0)
    blk60 = safe_float(first_present(row, ["blocks_per_60", "blk_per_60"], 0.0), 0.0)
    hits60 = safe_float(first_present(row, ["hits_per_60", "hit_per_60"], 0.0), 0.0)
    giv60 = safe_float(first_present(row, ["giveaways_per_60"], 0.0), 0.0)
    penalties_taken60 = safe_float(first_present(row, ["penalties_taken_per_60"], 0.0), 0.0)
    xga60 = safe_float(first_present(row, ["xga_per_60"], 0.0), 0.0)
    hdca60 = safe_float(first_present(row, ["high_danger_chances_against_per_60"], 0.0), 0.0)
    defensive_actions60 = safe_float(first_present(row, ["defensive_actions_per_60"], 0.0), 0.0)
    defensive_mistakes60 = safe_float(first_present(row, ["defensive_mistakes_per_60"], 0.0), 0.0)

    pp_points60 = safe_float(first_present(row, ["pp_points_per_60"], 0.0), 0.0)
    pk_toi_share = safe_float(first_present(row, ["pk_toi_share"], 0.0), 0.0)
    sh_points60 = safe_float(first_present(row, ["sh_points_per_60"], 0.0), 0.0)

    penalty_diff60 = safe_float(first_present(row, ["penalty_differential_per_60"], 0.0), 0.0)
    clutch_points60 = safe_float(first_present(row, ["clutch_points_per_60"], 0.0), 0.0)
    gwg = safe_float(first_present(row, ["gwg", "game_winning_goals"], 0.0), 0.0)
    otg = safe_float(first_present(row, ["otg", "overtime_goals"], 0.0), 0.0)
    shootout_winners = safe_float(first_present(row, ["shootout_winners"], 0.0), 0.0)

    qoc = safe_float(first_present(row, ["quality_of_competition"], 0.0), 0.0)
    dz_start_pct = safe_float(first_present(row, ["defensive_zone_start_pct"], 0.0), 0.0)
    deployment_difficulty = safe_float(first_present(row, ["deployment_difficulty"], 0.0), 0.0)
    toi_sec = safe_float(first_present(row, ["toi_sec"], 0.0), 0.0)
    toi_per_game = per_game(toi_sec / 60.0, gp)

    pdo_value = safe_float(first_present(row, ["pdo"], 0.0), 0.0)
    pdo_decimal = (pdo_value / 100.0) if abs(pdo_value) > 10 else pdo_value
    pdo_valid = bool(first_present(row, ["pdo_valid"], pdo_value > 0))
    goals_above_expected = safe_float(first_present(row, ["goals_above_expected"], 0.0), 0.0)

    # Raw formula-based scores before normalization.
    offensive_raw = (
        g60 * 20.0
        + primary_a60 * 14.0
        + secondary_a60 * 7.0
        + shots60 * 1.5
        + ixg60 * 16.0
        + xa60 * 12.0
        + finishing60 * 8.0
        + pts_per_game * 10.0
    )

    defensive_raw = (
        tak60 * (5.0 if is_d else 6.0)
        + blk60 * (4.0 if is_d else 3.0)
        + hits60 * (2.0 if is_d else 1.5)
        + safe_float(first_present(row, ["controlled_exit_rate"], 0.0), 0.0) * 12.0
        - giv60 * 5.0
        - penalties_taken60 * 4.0
        - xga60 * (14.0 if is_d else 10.0)
        - hdca60 * (3.0 if is_d else 2.0)
    )

    possession_raw = (
        (cf_pct - 0.50) * 100.0
        + (ff_pct - 0.50) * 80.0
        + (xgf_pct - 0.50) * 140.0
        + (gf_pct - 0.50) * 80.0
    )

    efficiency_raw = (
        safe_float(first_present(row, ["shooting_pct"], 0.0), 0.0) * 45.0
        + finishing60 * 8.0
        + p60 * 10.0
        + safe_float(first_present(row, ["primary_point_pct"], 0.0), 0.0) * 12.0
    )

    special_raw = (
        pp_points60 * 6.0
        + pk_toi_share * 10.0
        + sh_points60 * 8.0
    )

    discipline_raw = (
        penalty_diff60 * 8.0
        + max(0.0, 2.0 - penalties_taken60) * 5.0
    )

    clutch_raw = (
        clutch_points60 * 4.0
        + gwg * 0.5
        + otg * 1.0
        + shootout_winners * 0.5
    )

    usage_raw = (
        qoc * 5.0
        + dz_start_pct * 5.0
        + deployment_difficulty * 4.0
        + toi_per_game * 0.8
    )

    transition_raw = safe_float(first_present(row, ["transition_impact_per_60"], 0.0), 0.0) * 8.0
    physical_raw = hits60 * 3.0 + blk60 * 1.5 + safe_float(first_present(row, ["battle_win_pct"], 0.0), 0.0) * 20.0

    defensive_action_raw = defensive_actions60 * 4.0 - defensive_mistakes60 * 5.0

    regression_penalty = (
        (max(0.0, pdo_decimal - 1.02) * 150.0 if pdo_valid else 0.0)
        + max(0.0, safe_float(first_present(row, ["finishing_per_60"], 0.0), 0.0) - 1.0) * 2.0
        + max(0.0, goals_above_expected - 10.0) * 0.35
    )

    games_missed = safe_float(first_present(row, ["games_missed"], 0.0), 0.0)
    injury_risk_penalty = pct(games_missed, gp + games_missed, default=0.0) * 20.0 if gp + games_missed > 0 else 0.0

    # Convert raw score bands into digestible 0-100 component scores.
    offense_score = clamp(offensive_raw * (2.6 if not is_d else 2.9), 0.0, 100.0)
    defense_score = clamp(50.0 + defensive_raw * 1.8 + defensive_action_raw * 0.3, 0.0, 100.0)
    possession_score = clamp(50.0 + possession_raw * 1.2, 0.0, 100.0)
    efficiency_score = clamp(efficiency_raw * 2.2, 0.0, 100.0)
    special_teams_score = clamp(special_raw * 2.7, 0.0, 100.0)
    discipline_score = clamp(50.0 + discipline_raw * 2.0, 0.0, 100.0)
    clutch_score = clamp(clutch_raw * 4.0, 0.0, 100.0)
    usage_score = clamp(usage_raw * 2.8, 0.0, 100.0)
    transition_score = clamp(50.0 + transition_raw * 2.0, 0.0, 100.0)
    physical_score = clamp(physical_raw * 2.0, 0.0, 100.0)

    if gp <= 0:
        offense_score = defense_score = possession_score = efficiency_score = 0.0
        special_teams_score = discipline_score = clutch_score = usage_score = 0.0
        transition_score = physical_score = 0.0

    return {
        "offense_score": round_to(offense_score, 2),
        "offensive_score": round_to(offense_score, 2),
        "defense_score": round_to(defense_score, 2),
        "defensive_score": round_to(defense_score, 2),
        "possession_score": round_to(possession_score, 2),
        "efficiency_score": round_to(efficiency_score, 2),
        "special_teams_score": round_to(special_teams_score, 2),
        "discipline_score": round_to(discipline_score, 2),
        "clutch_score": round_to(clutch_score, 2),
        "usage_score": round_to(usage_score, 2),
        "usage_difficulty_score": round_to(usage_score, 2),
        "transition_score": round_to(transition_score, 2),
        "physical_score": round_to(physical_score, 2),
        "regression_penalty": round_to(regression_penalty, 2),
        "injury_risk_penalty": round_to(injury_risk_penalty, 2),
    }


def calculate_skater_analytics_rating(row: Mapping[str, Any]) -> Dict[str, Any]:
    pos = normalize_position(first_present(row, ["position", "pos"], "F"))
    is_d = pos == "D"

    offense = safe_float(first_present(row, ["offense_score", "offensive_score"], 0.0), 0.0)
    defense = safe_float(first_present(row, ["defense_score", "defensive_score"], 0.0), 0.0)
    possession = safe_float(first_present(row, ["possession_score"], 0.0), 0.0)
    efficiency = safe_float(first_present(row, ["efficiency_score"], 0.0), 0.0)
    special = safe_float(first_present(row, ["special_teams_score"], 0.0), 0.0)
    discipline = safe_float(first_present(row, ["discipline_score"], 0.0), 0.0)
    clutch = safe_float(first_present(row, ["clutch_score"], 0.0), 0.0)
    usage = safe_float(first_present(row, ["usage_score", "usage_difficulty_score"], 0.0), 0.0)
    transition = safe_float(first_present(row, ["transition_score"], 0.0), 0.0)
    physical = safe_float(first_present(row, ["physical_score"], 0.0), 0.0)
    regression_penalty = safe_float(first_present(row, ["regression_penalty"], 0.0), 0.0)
    injury_risk_penalty = safe_float(first_present(row, ["injury_risk_penalty"], 0.0), 0.0)

    if is_d:
        rating = (
            defense * 0.30
            + possession * 0.22
            + offense * 0.18
            + transition * 0.12
            + special * 0.08
            + discipline * 0.05
            + physical * 0.03
            + usage * 0.02
        )
    elif pos in FORWARD_POSITIONS:
        rating = (
            offense * 0.38
            + possession * 0.18
            + defense * 0.14
            + efficiency * 0.12
            + special * 0.08
            + discipline * 0.04
            + clutch * 0.04
            + usage * 0.02
        )
    else:
        rating = (
            offense * 0.30
            + defense * 0.22
            + possession * 0.18
            + efficiency * 0.10
            + special * 0.08
            + discipline * 0.05
            + clutch * 0.04
            + usage * 0.03
        )

    rating = rating - regression_penalty - injury_risk_penalty
    rating = clamp(rating, 0.0, 100.0)

    impact_score = rating

    return {
        "analytics_rating": round_to(rating, 2),
        "player_analytics_rating": round_to(rating, 2),
        "impact_score": round_to(impact_score, 2),
        "impact": round_to(impact_score, 2),
        "impact_tier": impact_tier(impact_score),
    }


# ============================================================
# SKATER ADVANCED VALUE
# ============================================================

def calculate_skater_value_metrics(row: Mapping[str, Any]) -> Dict[str, Any]:
    gp = safe_int(first_present(row, ["gp", "games_played"], 0))
    pts = safe_int(first_present(row, ["pts", "points"], 0))
    g = safe_int(first_present(row, ["g", "goals"], 0))
    # Same TOI resolution as normalize_skater_counting_stats — season rows sometimes
    # only carry `toi` / `toi_min` (minutes) without `toi_sec`.
    toi_sec = safe_float(first_present(row, ["toi_sec", "time_on_ice_sec"], 0.0), 0.0)
    if toi_sec <= 0:
        toi_min_raw = safe_float(first_present(row, ["toi", "toi_min", "time_on_ice"], 0.0), 0.0)
        # ATOI-style values (e.g. 18.5) are per-game; season minutes are much larger.
        if toi_min_raw > 0:
            if toi_min_raw <= 45.0 and gp > 0:
                toi_sec = toi_min_raw * 60.0 * float(gp)
            else:
                toi_sec = toi_min_raw * 60.0
    toi_min = toi_sec / 60.0 if toi_sec > 0 else 0.0

    p60 = safe_float(first_present(row, ["pts_per_60", "points_per_60"], 0.0), 0.0)
    ixg60 = safe_float(first_present(row, ["ixg_per_60", "individual_xg_per_60"], 0.0), 0.0)
    xa60 = safe_float(first_present(row, ["xa_per_60", "expected_assists_per_60"], 0.0), 0.0)
    xga60 = safe_float(first_present(row, ["xga_per_60"], 0.0), 0.0)
    penalty_diff60 = safe_float(first_present(row, ["penalty_differential_per_60"], 0.0), 0.0)
    cf_count = safe_float(first_present(row, ["cf", "corsi_for"], 0.0), 0.0)
    ca = safe_float(first_present(row, ["ca", "corsi_against"], 0.0), 0.0)
    xgf_count = safe_float(first_present(row, ["xgf", "expected_goals_for", "on_ice_xgf"], 0.0), 0.0)
    xga = safe_float(first_present(row, ["xga", "expected_goals_against", "on_ice_xga"], 0.0), 0.0)
    cf_raw = first_present(row, ["cf_pct", "corsi_pct"], None)
    xgf_raw = first_present(row, ["xgf_pct"], None)
    gf_raw = first_present(row, ["gf_pct", "goal_share"], None)
    if cf_count > 0 and ca > 0:
        cf_pct = cf_count / (cf_count + ca)
        cf_raw = cf_pct
    else:
        cf_pct = safe_float(cf_raw, 0.0) if cf_raw is not None else 0.0
    if xgf_count > 0 and xga > 0:
        xgf_pct = xgf_count / (xgf_count + xga)
        xgf_raw = xgf_pct
    else:
        xgf_pct = safe_float(xgf_raw, 0.0) if xgf_raw is not None else 0.0
    # Prefer live on-ice goals when present so stored/stale gf_pct can't disagree
    # with the GF-GA sublabel (e.g. 60.7% over 75-73).
    gf_on_v = safe_float(first_present(row, ["gf_on", "on_ice_gf"], 0.0), 0.0)
    ga_on_v = safe_float(first_present(row, ["ga_on", "on_ice_ga"], 0.0), 0.0)
    if gf_on_v + ga_on_v > 0:
        gf_pct = gf_on_v / (gf_on_v + ga_on_v)
        gf_raw = gf_pct
    else:
        gf_pct = safe_float(gf_raw, 0.0) if gf_raw is not None else 0.0
    possession_sample_valid = ca > 0 and xga > 0 and cf_raw is not None and xgf_raw is not None
    special_points = safe_float(
        first_present(
            row,
            ["special_team_points"],
            safe_int(first_present(row, ["ppg"], 0)) + safe_int(first_present(row, ["ppa"], 0)) + safe_int(first_present(row, ["shg"], 0)) + safe_int(first_present(row, ["sha"], 0)),
        ),
        0.0,
    )
    special_points60 = per_60(special_points, toi_sec)

    fow = safe_float(first_present(row, ["fow", "faceoff_wins"], 0.0), 0.0)
    fol = safe_float(first_present(row, ["fol", "faceoff_losses"], 0.0), 0.0)
    fo_pct = safe_float(first_present(row, ["fo_pct", "faceoff_pct"], 0.0), 0.0)
    faceoffs_taken = fow + fol

    offensive_gar = (p60 - REPLACEMENT_POINTS_PER_60) * (toi_min / 60.0) * 0.28
    ixg_sample = safe_float(first_present(row, ["ixg", "individual_xg", "individual_expected_goals"], 0.0), 0.0)
    if ixg_sample > 0:
        offensive_gar += (ixg60 - REPLACEMENT_XG_PER_60) * (toi_min / 60.0) * 0.32
    else:
        # Light boxes may omit ixg — don't punish production-only rows.
        offensive_gar += (p60 - REPLACEMENT_POINTS_PER_60) * (toi_min / 60.0) * 0.12

    # Missing against-stats must not look like elite suppression (xga60==0 → huge WAR).
    defensive_gar = (REPLACEMENT_XGA_PER_60 - xga60) * (toi_min / 60.0) * 0.55 if possession_sample_valid else 0.0
    penalty_gar = penalty_diff60 * (toi_min / 60.0) * 0.12
    faceoff_gar = (fo_pct - REPLACEMENT_FACE_OFF_PCT) * faceoffs_taken * 0.015 if faceoffs_taken > 0 else 0.0
    if possession_sample_valid:
        possession_gar = (
            max(-0.18, cf_pct - 0.475) * 0.70
            + max(-0.18, xgf_pct - 0.475) * 1.10
            + max(-0.18, (gf_pct if gf_raw is not None else 0.475) - 0.475) * 0.40
        ) * (toi_min / 60.0)
    else:
        possession_gar = 0.0
    xa_sample = safe_float(first_present(row, ["xa", "expected_assists"], 0.0), 0.0)
    playmaking_gar = max(-0.25, xa60 - 0.35) * (toi_min / 60.0) * 0.18 if xa_sample > 0 else 0.0
    special_teams_gar = max(-0.20, special_points60 - 0.25) * (toi_min / 60.0) * 0.12

    gar = offensive_gar + defensive_gar + penalty_gar + faceoff_gar
    total_impact_gar = gar + possession_gar + playmaking_gar + special_teams_gar
    base_war = gar / GOALS_PER_WIN
    war = total_impact_gar / GOALS_PER_WIN
    analytics_gp = safe_int(first_present(row, ["analytics_gp", "gp"], 0))
    # Early-season sample: require ice time + possession, not a full 20-game slate.
    # Awards / qualified boards can still require larger samples client-side.
    war_valid = analytics_gp >= 3 and toi_min >= 45 and possession_sample_valid

    cap_hit = safe_float(first_present(row, ["cap_hit"], 0.0), 0.0)

    cost_per_point = cap_hit / pts if cap_hit > 0 and pts > 0 else 0.0
    cost_per_goal = cap_hit / g if cap_hit > 0 and g > 0 else 0.0
    cost_per_war = cap_hit / war if cap_hit > 0 and war > 0 else 0.0

    production_score = safe_float(first_present(row, ["offense_score"], 0.0), 0.0)
    two_way_score = (
        safe_float(first_present(row, ["defense_score"], 0.0), 0.0)
        + safe_float(first_present(row, ["possession_score"], 0.0), 0.0)
    ) / 2.0

    age = safe_int(first_present(row, ["age"], 0), 0)
    if age <= 0:
        age_value = 50.0
    elif age <= 23:
        age_value = 75.0
    elif age <= 28:
        age_value = 65.0
    elif age <= 32:
        age_value = 50.0
    else:
        age_value = max(10.0, 50.0 - (age - 32) * 4.0)

    contract_value = 50.0
    if cap_hit > 0:
        contract_value = clamp(85.0 - cap_hit * 6.0 + war * 6.0, 0.0, 100.0)

    injury_penalty = safe_float(first_present(row, ["injury_risk_penalty"], 0.0), 0.0)

    ovr_raw = safe_float(first_present(row, ["overall", "ovr"], 0.0), 0.0)
    pot_raw = safe_float(first_present(row, ["potential"], 0.0), 0.0)
    if age <= 22 and pot_raw > ovr_raw + 4.0:
        ovr_weight, pot_weight = 0.22, 0.28
    else:
        ovr_weight, pot_weight = 0.30, 0.20

    value_score = clamp(
        production_score * 0.35
        + two_way_score * 0.25
        + age_value * 0.15
        + contract_value * 0.20
        - injury_penalty * 0.50,
        0.0,
        100.0,
    )

    trade_value = (
        safe_float(first_present(row, ["overall", "ovr"], 0.0), 0.0) * ovr_weight
        + safe_float(first_present(row, ["potential"], 0.0), 0.0) * pot_weight
        + war * 12.0
        + age_value * 0.25
        + contract_value * 0.15
        - injury_penalty
    )
    try:
        from services.contract_economy import compute_market_value_from_row

        market_m = float(compute_market_value_from_row(row))
        trade_value = market_m * 14.0 + war * 8.0 + age_value * 0.20 - injury_penalty * 0.5
    except Exception:
        pass

    return {
        "offensive_gar": round_to(offensive_gar, 3),
        "defensive_gar": round_to(defensive_gar, 3),
        "penalty_gar": round_to(penalty_gar, 3),
        "faceoff_gar": round_to(faceoff_gar, 3),
        "possession_gar": round_to(possession_gar, 3),
        "playmaking_gar": round_to(playmaking_gar, 3),
        "special_teams_gar": round_to(special_teams_gar, 3),
        "gar": round_to(gar, 3),
        "base_war": round_to(base_war, 3),
        "war": round_to(war, 1),
        "watr": round_to(war, 1),
        "WATR": round_to(war, 1),
        "total_impact": round_to(war, 1),
        "total_impact_gar": round_to(total_impact_gar, 3),
        "war_formula": "WAR=(offensive_gar+defensive_gar+penalty_gar+faceoff_gar+possession_gar+playmaking_gar+special_teams_gar)/GOALS_PER_WIN",
        "war_valid": war_valid,
        "cost_per_point": round_to(cost_per_point, 3),
        "cost_per_goal": round_to(cost_per_goal, 3),
        "cost_per_war": round_to(cost_per_war, 3),
        "age_value": round_to(age_value, 2),
        "contract_value": round_to(contract_value, 2),
        "value_score": round_to(value_score, 2),
        "trade_value_score": round_to(trade_value, 2),
    }


# ============================================================
# ARCHETYPES / ROLES
# ============================================================

def impact_tier(score: Any) -> str:
    s = safe_float(score, 0.0)
    if s >= IMPACT_ELITE:
        return "Elite Driver"
    if s >= IMPACT_STAR:
        return "Star / First-Line Impact"
    if s >= IMPACT_CORE:
        return "Core Contributor"
    if s >= IMPACT_DEPTH:
        return "Depth Contributor"
    if s > 0:
        return "Replacement / Specialist"
    return "No Data"


def calculate_archetype_scores(row: Mapping[str, Any]) -> Dict[str, Any]:
    pos = normalize_position(first_present(row, ["position", "pos"], "F"))

    g60 = safe_float(first_present(row, ["goals_per_60", "g_per_60"], 0.0), 0.0)
    p60 = safe_float(first_present(row, ["points_per_60", "pts_per_60"], 0.0), 0.0)
    shots60 = safe_float(first_present(row, ["shots_per_60"], 0.0), 0.0)
    sh_pct = safe_float(first_present(row, ["shooting_pct"], 0.0), 0.0)
    ixg60 = safe_float(first_present(row, ["individual_xg_per_60", "ixg_per_60"], 0.0), 0.0)

    primary_a60 = safe_float(first_present(row, ["primary_assists_per_60"], 0.0), 0.0)
    xa60 = safe_float(first_present(row, ["expected_assists_per_60", "xa_per_60"], 0.0), 0.0)
    shot_assists60 = safe_float(first_present(row, ["shot_assists_per_60"], 0.0), 0.0)
    pass_completion_pct = safe_float(first_present(row, ["pass_completion_pct"], 0.0), 0.0)

    defensive_impact60 = safe_float(first_present(row, ["defensive_impact_per_60"], 0.0), 0.0)
    xgf_pct = safe_float(first_present(row, ["xgf_pct"], 0.0), 0.0)
    tak60 = safe_float(first_present(row, ["takeaways_per_60"], 0.0), 0.0)
    giv60 = safe_float(first_present(row, ["giveaways_per_60"], 0.0), 0.0)

    hits60 = safe_float(first_present(row, ["hits_per_60"], 0.0), 0.0)
    net_front_chances60 = per_60(first_present(row, ["net_front_chances"], 0), first_present(row, ["toi_sec"], 0.0))
    battle_win_pct = safe_float(first_present(row, ["battle_win_pct"], 0.0), 0.0)

    pp_points60 = safe_float(first_present(row, ["pp_points_per_60"], 0.0), 0.0)
    controlled_exits60 = per_60(first_present(row, ["controlled_exits", "successful_exits"], 0), first_present(row, ["toi_sec"], 0.0))
    blk60 = safe_float(first_present(row, ["blocks_per_60"], 0.0), 0.0)
    xga60 = safe_float(first_present(row, ["xga_per_60"], 0.0), 0.0)
    hdca60 = safe_float(first_present(row, ["high_danger_chances_against_per_60"], 0.0), 0.0)

    sniper = (
        g60 * 30.0
        + shots60 * 3.0
        + sh_pct * 100.0
        + ixg60 * 15.0
    )

    playmaker = (
        primary_a60 * 25.0
        + xa60 * 20.0
        + shot_assists60 * 5.0
        + pass_completion_pct * 20.0
    )

    two_way = (
        defensive_impact60 * 8.0
        + p60 * 8.0
        + xgf_pct * 30.0
        + tak60 * 6.0
        - giv60 * 5.0
    )

    power_forward = (
        hits60 * 4.0
        + g60 * 20.0
        + net_front_chances60 * 8.0
        + battle_win_pct * 20.0
    )

    offensive_defenseman = (
        p60 * 18.0
        + shots60 * 3.0
        + pp_points60 * 10.0
        + controlled_exits60 * 6.0
    )

    shutdown_defenseman = (
        blk60 * 4.0
        + hits60 * 3.0
        + tak60 * 6.0
        + max(0.0, REPLACEMENT_XGA_PER_60 - xga60) * 20.0
        + max(0.0, 8.0 - hdca60) * 15.0
    )

    transition_driver = safe_float(first_present(row, ["transition_score"], 0.0), 0.0)
    pest = (
        safe_float(first_present(row, ["penalties_drawn_per_60"], 0.0), 0.0) * 12.0
        + safe_float(first_present(row, ["penalty_differential_per_60"], 0.0), 0.0) * 10.0
        + hits60 * 2.0
        + safe_float(first_present(row, ["clutch_score"], 0.0), 0.0) * 0.20
    )

    scores = {
        "sniper_score": clamp(sniper, 0.0, 100.0),
        "playmaker_score": clamp(playmaker, 0.0, 100.0),
        "two_way_score": clamp(two_way, 0.0, 100.0),
        "power_forward_score": clamp(power_forward, 0.0, 100.0),
        "offensive_defenseman_score": clamp(offensive_defenseman, 0.0, 100.0),
        "shutdown_defenseman_score": clamp(shutdown_defenseman, 0.0, 100.0),
        "transition_driver_score": clamp(transition_driver, 0.0, 100.0),
        "pest_score": clamp(pest, 0.0, 100.0),
    }

    if pos == "D":
        candidates = {
            "Offensive Defenseman": scores["offensive_defenseman_score"],
            "Shutdown Defenseman": scores["shutdown_defenseman_score"],
            "Two-Way Defenseman": (scores["offensive_defenseman_score"] + scores["shutdown_defenseman_score"]) / 2.0,
            "Transition Defenseman": scores["transition_driver_score"],
        }
    else:
        candidates = {
            "Sniper": scores["sniper_score"],
            "Playmaker": scores["playmaker_score"],
            "Two-Way Forward": scores["two_way_score"],
            "Power Forward": scores["power_forward_score"],
            "Transition/Rush Forward": scores["transition_driver_score"],
            "Pest/Agitator": scores["pest_score"],
        }

    archetype = max(candidates.items(), key=lambda item: item[1])[0] if candidates else "Depth Player"

    return {
        **{k: round_to(v, 2) for k, v in scores.items()},
        "archetype": archetype,
        "player_type": archetype,
    }


def skater_role_label(row: Mapping[str, Any], impact_score: Optional[float] = None) -> str:
    pos = normalize_position(first_present(row, ["position", "pos"], "F"))
    gp = safe_int(first_present(row, ["gp", "games_played"], 0))
    g = safe_int(first_present(row, ["g", "goals"], 0))
    a = safe_int(first_present(row, ["a", "assists"], 0))
    pts = g + a
    sog = safe_int(first_present(row, ["sog", "shots"], 0))
    hit = safe_int(first_present(row, ["hit", "hits"], 0))
    blk = safe_int(first_present(row, ["blk", "blocks"], 0))
    impact = safe_float(impact_score, safe_float(first_present(row, ["impact_score", "analytics_rating"], 0.0), 0.0))

    ppgame = per_game(pts, gp)
    gpgame = per_game(g, gp)
    shotgame = per_game(sog, gp)
    physical = per_game(hit + blk, gp)
    cf_pct = safe_float(first_present(row, ["cf_pct"], 0.0), 0.0)
    xgf_pct = safe_float(first_present(row, ["xgf_pct"], 0.0), 0.0)

    if impact >= IMPACT_ELITE:
        return "Franchise Driver"
    if ppgame >= 1.05 and gp >= 5:
        return "Elite Producer"
    if gpgame >= 0.55 and gp >= 5:
        return "Goal Scorer"
    if a > g * 1.6 and ppgame >= 0.55:
        return "Playmaker"
    if pos == "D" and ppgame >= 0.55:
        return "Offensive Defenseman"
    if pos == "D" and physical >= 3.0 and xgf_pct >= 0.50:
        return "Shutdown Defenseman"
    if cf_pct >= 0.55 and xgf_pct >= 0.55:
        return "Possession Driver"
    if physical >= 3.5:
        return "Physical Two-Way Player"
    if shotgame >= 3.0:
        return "Volume Shooter"
    if impact >= IMPACT_CORE:
        return "Core Contributor"
    if impact >= IMPACT_DEPTH:
        return "Depth Contributor"
    return "Depth / Replacement"


# ============================================================
# GOALIE RATES / ANALYTICS
# ============================================================

def calculate_goalie_rates(row: Mapping[str, Any]) -> Dict[str, Any]:
    gp = safe_int(first_present(row, ["gp", "games_played"], 0))
    starts = safe_int(first_present(row, ["starts"], gp))
    wins = safe_int(first_present(row, ["w", "wins"], 0))
    losses = safe_int(first_present(row, ["l", "losses"], 0))
    otl = safe_int(first_present(row, ["otl", "ot_losses"], 0))
    decisions = wins + losses + otl

    ga = safe_int(first_present(row, ["ga", "goals_against"], 0))
    sa = safe_int(first_present(row, ["sa", "shots_against"], 0))
    saves = safe_int(first_present(row, ["saves"], 0))
    shutouts = safe_int(first_present(row, ["so", "shutouts"], 0))
    toi_sec = safe_float(first_present(row, ["toi_sec"], 0.0), 0.0)
    if toi_sec <= 0 and gp > 0:
        toi_sec = gp * DEFAULT_GOALIE_TOI_PER_GAME_SEC
    elif gp > 0 and toi_sec < float(gp) * 1800.0:
        toi_sec = gp * DEFAULT_GOALIE_TOI_PER_GAME_SEC

    xga_raw = first_present(row, ["goalie_xga", "xga", "expected_goals_against"], None)
    xga = safe_float(xga_raw, 0.0)
    # xga==0 is a missing sample (light sims), not "expected zero goals".
    xga_valid = xga_raw is not None and xga > 0 and sa > 0

    hdsa = safe_int(first_present(row, ["hdsa", "high_danger_shots_against"], 0))
    hdsaves = safe_int(first_present(row, ["hdsaves", "high_danger_saves"], 0))
    mdsa = safe_int(first_present(row, ["mdsa", "medium_danger_shots_against"], 0))
    mdsaves = safe_int(first_present(row, ["mdsaves", "medium_danger_saves"], 0))
    ldsa = safe_int(first_present(row, ["ldsa", "low_danger_shots_against"], 0))
    ldsaves = safe_int(first_present(row, ["ldsaves", "low_danger_saves"], 0))

    rebounds_allowed = safe_int(first_present(row, ["rebounds_allowed"], 0))
    quality_starts = safe_int(first_present(row, ["quality_starts"], 0))
    bad_starts = safe_int(first_present(row, ["bad_starts"], 0))
    steal_games = safe_int(first_present(row, ["steal_games", "goalie_steals"], 0))

    sv_pct = pct(saves, sa, default=0.0)
    gaa = (ga * 3600.0 / toi_sec) if toi_sec > 0 else 0.0
    win_pct = pct(wins, decisions, default=0.0)

    gsax = (xga - ga) if xga_valid else None
    gsax_valid = bool(xga_valid)

    expected_ga_avg = sa * (1.0 - LEAGUE_AVG_SV_PCT)
    gsaa = expected_ga_avg - ga if sa > 0 else 0.0

    hd_sv_pct = pct(hdsaves, hdsa, default=0.0)
    md_sv_pct = pct(mdsaves, mdsa, default=0.0)
    ld_sv_pct = pct(ldsaves, ldsa, default=0.0)

    rebound_rate = pct(rebounds_allowed, saves, default=0.0)
    rebound_control_score = (1.0 - rebound_rate) * 100.0 if saves > 0 else 0.0

    quality_start_pct = pct(quality_starts, starts, default=0.0)
    bad_start_rate = pct(bad_starts, starts, default=0.0)
    steal_rate = pct(steal_games, starts, default=0.0)

    workload = per_60(sa, toi_sec)
    clutch_save_pct = pct(first_present(row, ["clutch_saves"], 0), first_present(row, ["clutch_shots_against"], 0), default=0.0)

    return {
        "sv_pct": sv_pct,
        "save_pct": sv_pct,
        "gaa": gaa,
        "win_pct": win_pct,
        "goals_against_per_game": per_game(ga, gp),
        "saves_per_game": per_game(saves, gp),
        "shots_against_per_game": per_game(sa, gp),
        "shutouts_per_game": per_game(shutouts, gp),
        "shutout_rate": pct(shutouts, starts, default=0.0),

        "gsax": gsax,
        "goals_saved_above_expected": gsax,
        "gsax_valid": gsax_valid,
        "gsax_per_60": per_60(gsax, toi_sec) if gsax is not None else None,
        "gsax_per_game": per_game(gsax, gp) if gsax is not None else None,

        "gsaa": gsaa,
        "goals_saved_above_average": gsaa,
        "gsaa_per_60": per_60(gsaa, toi_sec),
        "gsaa_per_game": per_game(gsaa, gp),

        "high_danger_save_pct": hd_sv_pct,
        "medium_danger_save_pct": md_sv_pct,
        "low_danger_save_pct": ld_sv_pct,

        "rebound_rate": rebound_rate,
        "rebound_control_score": rebound_control_score,

        "quality_start_pct": quality_start_pct,
        "quality_start_percentage": quality_start_pct,
        "bad_start_rate": bad_start_rate,
        "steal_rate": steal_rate,
        "goalie_steal_rate": steal_rate,

        "workload": workload,
        "shots_against_per_60": workload,
        "clutch_save_pct": clutch_save_pct,
    }


def calculate_goalie_component_scores(row: Mapping[str, Any]) -> Dict[str, Any]:
    sv_pct = safe_float(first_present(row, ["sv_pct", "save_pct"], 0.0), 0.0)
    gaa = safe_float(first_present(row, ["gaa"], 0.0), 0.0)
    gsax60 = safe_float(first_present(row, ["gsax_per_60"], 0.0), 0.0)
    gsaa60 = safe_float(first_present(row, ["gsaa_per_60"], 0.0), 0.0)
    hd_sv_pct = safe_float(first_present(row, ["high_danger_save_pct"], 0.0), 0.0)
    qs_pct = safe_float(first_present(row, ["quality_start_pct", "quality_start_percentage"], 0.0), 0.0)
    rebound_control = safe_float(first_present(row, ["rebound_control_score"], 0.0), 0.0)
    workload = safe_float(first_present(row, ["workload", "shots_against_per_60"], 0.0), 0.0)
    clutch_save_pct = safe_float(first_present(row, ["clutch_save_pct"], 0.0), 0.0)
    bad_start_rate = safe_float(first_present(row, ["bad_start_rate"], 0.0), 0.0)

    save_percentage_score = clamp((sv_pct - 0.880) / 0.050 * 100.0, 0.0, 115.0)
    gaa_score = inverse_scale_to_100(gaa, bad=3.80, elite=2.10, ceiling=115.0) if gaa > 0 else 0.0
    gsax_score = clamp(50.0 + gsax60 * 20.0, 0.0, 115.0)
    gsaa_score = clamp(50.0 + gsaa60 * 20.0, 0.0, 115.0)
    high_danger_save_score = clamp((hd_sv_pct - 0.730) / 0.160 * 100.0, 0.0, 115.0) if hd_sv_pct > 0 else 50.0
    quality_start_score = clamp(qs_pct * 100.0, 0.0, 100.0)
    rebound_score = clamp(rebound_control, 0.0, 100.0)
    workload_score = clamp(workload / 34.0 * 100.0, 0.0, 115.0)
    clutch_score = clamp((clutch_save_pct - 0.860) / 0.080 * 100.0, 0.0, 115.0) if clutch_save_pct > 0 else 50.0
    bad_start_penalty = bad_start_rate * 30.0
    gp = safe_int(first_present(row, ["gp", "games_played"], 0))
    starts = safe_int(first_present(row, ["starts"], gp))
    gsax = first_present(row, ["gsax", "goals_saved_above_expected"], None)
    save_value_goals = safe_float(gsax, safe_float(first_present(row, ["gsaa", "goals_saved_above_average"], 0.0), 0.0))
    quality_start_goals = (qs_pct - 0.50) * max(0, starts) * 0.18
    bad_start_goals = bad_start_rate * max(0, starts) * 0.22
    goalie_total_impact_goals = save_value_goals + quality_start_goals - bad_start_goals
    goalie_watr = goalie_total_impact_goals / GOALS_PER_WIN

    goalie_rating = (
        gsax_score * 0.35
        + save_percentage_score * 0.22
        + high_danger_save_score * 0.18
        + quality_start_score * 0.10
        + rebound_score * 0.08
        + workload_score * 0.05
        + clutch_score * 0.02
        - bad_start_penalty
    )

    goalie_rating = clamp(goalie_rating, 0.0, 100.0)

    return {
        "save_percentage_score": round_to(save_percentage_score, 2),
        "save_score": round_to(save_percentage_score, 2),
        "gaa_score": round_to(gaa_score, 2),
        "gsax_score": round_to(gsax_score, 2),
        "gsaa_score": round_to(gsaa_score, 2),
        "high_danger_save_score": round_to(high_danger_save_score, 2),
        "quality_start_score": round_to(quality_start_score, 2),
        "rebound_score": round_to(rebound_score, 2),
        "workload_score": round_to(workload_score, 2),
        "clutch_score": round_to(clutch_score, 2),
        "bad_start_penalty": round_to(bad_start_penalty, 2),

        "goalie_analytics_rating": round_to(goalie_rating, 2),
        "analytics_rating": round_to(goalie_rating, 2),
        "impact_score": round_to(goalie_rating, 2),
        "impact": round_to(goalie_rating, 2),
        "goalie_impact": round_to(goalie_rating, 2),
        "war": round_to(goalie_watr, 3),
        "watr": round_to(goalie_watr, 3),
        "WATR": round_to(goalie_watr, 3),
        "total_impact": round_to(goalie_watr, 3),
        "total_impact_gar": round_to(goalie_total_impact_goals, 3),
        "war_formula": "Goalie WAR=(save_value_goals+quality_start_goals-bad_start_goals)/GOALS_PER_WIN",
        "impact_tier": goalie_tier(goalie_rating),
    }


def goalie_tier(score: Any) -> str:
    s = safe_float(score, 0.0)
    if s >= 78:
        return "Vezina-Level"
    if s >= 68:
        return "Elite Starter"
    if s >= 56:
        return "Reliable Starter"
    if s >= 44:
        return "Tandem / Backup"
    if s > 0:
        return "Replacement Goalie"
    return "No Data"


def goalie_role_label(row: Mapping[str, Any], impact_score: Optional[float] = None) -> str:
    gp = safe_int(first_present(row, ["gp", "games_played"], 0))
    starts = safe_int(first_present(row, ["starts"], gp))
    sv_pct = safe_float(first_present(row, ["sv_pct", "save_pct"], 0.0), 0.0)
    gsax = safe_float(first_present(row, ["gsax", "goals_saved_above_expected"], 0.0), 0.0)
    impact = safe_float(impact_score, safe_float(first_present(row, ["impact_score"], 0.0), 0.0))

    if starts >= 45 and impact >= 68:
        return "Elite Starter"
    if starts >= 38 and impact >= 56:
        return "Starter"
    if starts >= 22:
        return "Tandem Goalie"
    if gp > 0 and sv_pct >= 0.920 and gsax > 0:
        return "Hot Hand"
    if gp > 0:
        return "Backup Goalie"
    return "No Data"


# ============================================================
# ENRICH ONE PLAYER
# ============================================================

def calculate_regression_signal(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Backend regression-watch labels for Stats Central."""
    gp = safe_int(row.get("gp"), 0)
    finishing = safe_float(row.get("finishing"), 0.0)
    ixg = safe_float(row.get("ixg"), 0.0)
    xgf_pct = safe_float(row.get("xgf_pct"), 0.5)
    pdo = row.get("pdo")
    pdo_valid = bool(row.get("pdo_valid"))
    pts = safe_int(row.get("pts"), 0)
    signal = "NEUTRAL"
    strength = 0.0
    reason = "Within expected range"
    if gp < 15:
        return {"regression_signal": "SMALL_SAMPLE", "regression_strength": 0.0, "regression_reason": "Limited games played"}
    fin_per_game = finishing / max(1.0, gp)
    if ixg > 5 and fin_per_game > 0.18:
        signal = "SKILL_SUPPORTED"
        strength = min(1.0, fin_per_game / 0.35)
        reason = "Elite finishing supports goal rate"
    elif finishing > 8 and fin_per_game > 0.12:
        signal = "HOT_UNSUSTAINABLE"
        strength = min(1.0, finishing / 15.0)
        reason = "Conversion well above skill baseline"
    elif finishing < -8 and xgf_pct > 0.52:
        signal = "COLD_UNLUCKY"
        strength = min(1.0, abs(finishing) / 12.0)
        reason = "Strong process, cold finishing"
    elif xgf_pct < 0.47 and pts / max(1, gp) > 0.75:
        signal = "PROCESS_WARNING"
        strength = 0.6
        reason = "Poor xGF despite strong results"
    elif pdo_valid and pdo is not None and safe_float(pdo) > 1.03:
        signal = "PDO_ELEVATED"
        strength = min(1.0, (safe_float(pdo) - 1.0) * 12.0)
        reason = "PDO elevated in sample"
    return {"regression_signal": signal, "regression_strength": round(strength, 3), "regression_reason": reason}


def enrich_skater_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out.update(normalize_player_identity(row))
    out.update(normalize_skater_counting_stats(row))
    out.update(calculate_skater_rates(out))

    # Battle metrics after base rates.
    board_won = safe_float(first_present(out, ["board_battles_won"], 0.0), 0.0)
    board_total = safe_float(first_present(out, ["board_battles_total"], 0.0), 0.0)
    net_won = safe_float(first_present(out, ["net_front_battles_won"], 0.0), 0.0)
    net_total = safe_float(first_present(out, ["net_front_battles_total"], 0.0), 0.0)
    battle_win_pct = pct(board_won + net_won, board_total + net_total, default=0.0)
    physical_efficiency = (
        safe_float(out.get("hits"), 0.0)
        + board_won
        + net_won
        - safe_float(out.get("missed_hits"), 0.0)
        - safe_float(out.get("penalties_taken"), 0.0)
    )

    out["battle_win_pct"] = battle_win_pct
    out["physical_efficiency"] = physical_efficiency
    out["physical_efficiency_per_60"] = per_60(physical_efficiency, out.get("toi_sec"))

    out.update(calculate_skater_component_scores(out))
    out.update(calculate_skater_analytics_rating(out))
    out.update(calculate_archetype_scores(out))
    out.update(calculate_skater_value_metrics(out))
    out.update(calculate_regression_signal(out))

    out["pts"] = safe_int(out.get("g")) + safe_int(out.get("a"))
    out["points"] = out["pts"]
    out["primary_points"] = safe_int(out.get("g")) + safe_int(out.get("primary_assists"))
    out["special_team_points"] = safe_int(out.get("ppg")) + safe_int(out.get("ppa")) + safe_int(out.get("shg")) + safe_int(out.get("sha"))
    out["role_label"] = skater_role_label(out, out.get("impact_score"))

    out.setdefault("league_rank_pts", None)
    out.setdefault("team_rank_pts", None)
    out.setdefault("league_rank_impact", None)
    out.setdefault("team_rank_impact", None)

    clean_round_fields(out)
    return out


def enrich_goalie_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out.update(normalize_player_identity(row))
    out["position"] = "G"
    out["pos"] = "G"

    out.update(normalize_goalie_counting_stats(row))
    out.update(calculate_goalie_rates(out))
    out.update(calculate_goalie_component_scores(out))

    out["role_label"] = goalie_role_label(out, out.get("impact_score"))
    out["archetype"] = out["role_label"]
    out["player_type"] = out["role_label"]

    out.setdefault("league_rank_sv_pct", None)
    out.setdefault("team_rank_sv_pct", None)
    out.setdefault("league_rank_goalie_impact", None)
    out.setdefault("team_rank_goalie_impact", None)

    clean_round_fields(out)
    return out


def enrich_player_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for one row.
    Takes a raw game-ledger row and returns a frontend/backend analytics row.
    """
    if is_goalie_row(row):
        return enrich_goalie_row(row)
    return enrich_skater_row(row)


def enrich_player_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich many rows and add league/team ranks.
    """
    enriched: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        try:
            enriched.append(enrich_player_row(r))
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "enrich_player_row failed for %s",
                r.get("player_id") or r.get("name"),
            )
            fallback = dict(r)
            fallback.setdefault("war", None)
            enriched.append(fallback)

    skaters = [r for r in enriched if not is_goalie_row(r)]
    goalies = [r for r in enriched if is_goalie_row(r)]

    apply_skater_ranks(skaters)
    apply_goalie_ranks(goalies)

    return skaters + goalies


# ============================================================
# RANKS
# ============================================================

def _assign_numeric_ranks(
    rows: List[MutableMapping[str, Any]],
    key: str,
    out_key: str,
    *,
    reverse: bool = True,
    default: float = 0.0,
) -> None:
    eligible = []
    for row in rows:
        if row.get(key) is None:
            row[out_key] = None
            continue
        eligible.append(row)
    ordered = sorted(eligible, key=lambda r: safe_float(r.get(key), default), reverse=reverse)
    last_value: Optional[float] = None
    last_rank = 0
    for idx, row in enumerate(ordered, start=1):
        value = safe_float(row.get(key), default)
        if last_value is None or abs(value - last_value) > 1e-9:
            last_rank = idx
            last_value = value
        row[out_key] = last_rank


def apply_skater_ranks(rows: List[MutableMapping[str, Any]]) -> None:
    def rank_by(key: str, out_key: str, reverse: bool = True) -> None:
        _assign_numeric_ranks(rows, key, out_key, reverse=reverse, default=0.0)

    rank_by("pts", "league_rank_pts")
    rank_by("g", "league_rank_goals")
    rank_by("a", "league_rank_assists")
    rank_by("impact_score", "league_rank_impact")
    rank_by("analytics_rating", "league_rank_analytics_rating")
    rank_by("pts_per_game", "league_rank_pts_per_game")
    rank_by("pts_per_60", "league_rank_pts_per_60")
    rank_by("war", "league_rank_war")
    rank_by("watr", "league_rank_watr")
    rank_by("value_score", "league_rank_value")
    rank_by("cf_pct", "league_rank_cf_pct")
    rank_by("xgf_pct", "league_rank_xgf_pct")

    teams: Dict[str, List[MutableMapping[str, Any]]] = {}
    for row in rows:
        teams.setdefault(str(row.get("team_id") or ""), []).append(row)

    for _team_id, team_rows in teams.items():
        for key, out_key, reverse in [
            ("pts", "team_rank_pts", True),
            ("g", "team_rank_goals", True),
            ("a", "team_rank_assists", True),
            ("impact_score", "team_rank_impact", True),
            ("analytics_rating", "team_rank_analytics_rating", True),
            ("pts_per_game", "team_rank_pts_per_game", True),
            ("war", "team_rank_war", True),
            ("watr", "team_rank_watr", True),
            ("value_score", "team_rank_value", True),
        ]:
            _assign_numeric_ranks(team_rows, key, out_key, reverse=reverse, default=0.0)


def apply_goalie_ranks(rows: List[MutableMapping[str, Any]]) -> None:
    eligible = [r for r in rows if safe_int(r.get("gp"), 0) > 0]

    for key, out_key, reverse, default in [
        ("sv_pct", "league_rank_sv_pct", True, 0.0),
        ("gaa", "league_rank_gaa", False, 99.0),
        ("impact_score", "league_rank_goalie_impact", True, 0.0),
        ("watr", "league_rank_goalie_watr", True, 0.0),
        ("gsax", "league_rank_gsax", True, 0.0),
        ("quality_start_pct", "league_rank_quality_start_pct", True, 0.0),
    ]:
        _assign_numeric_ranks(eligible, key, out_key, reverse=reverse, default=default)

    teams: Dict[str, List[MutableMapping[str, Any]]] = {}
    for row in rows:
        teams.setdefault(str(row.get("team_id") or ""), []).append(row)

    for _team_id, team_rows in teams.items():
        team_eligible = [r for r in team_rows if safe_int(r.get("gp"), 0) > 0]

        for key, out_key, reverse, default in [
            ("sv_pct", "team_rank_sv_pct", True, 0.0),
            ("gaa", "team_rank_gaa", False, 99.0),
            ("impact_score", "team_rank_goalie_impact", True, 0.0),
            ("watr", "team_rank_goalie_watr", True, 0.0),
            ("gsax", "team_rank_gsax", True, 0.0),
        ]:
            _assign_numeric_ranks(team_eligible, key, out_key, reverse=reverse, default=default)


# ============================================================
# TEAM ANALYTICS
# ============================================================

def _weighted_mean_xgf_pct(
    skaters: Sequence[Mapping[str, Any]],
    fallback_xgf: float,
    fallback_xga: float,
) -> float:
    if not skaters:
        return pct(fallback_xgf, fallback_xgf + fallback_xga, default=0.0)
    weights = [
        max(1, safe_int(r.get("xgf_pct_gp"), safe_int(r.get("gp"), 0)))
        for r in skaters
    ]
    wsum = sum(weights)
    if wsum <= 0:
        return pct(fallback_xgf, fallback_xgf + fallback_xga, default=0.0)
    return sum(season_xgf_pct_from_row(r) * w for r, w in zip(skaters, weights)) / float(wsum)


def aggregate_team_from_player_rows(
    player_rows: Iterable[Mapping[str, Any]],
    *,
    team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build team analytics from enriched or raw player rows.

    Best used as a player-derived snapshot, while actual team W/L/GF/GA should
    still come from game results if available.
    """
    rows = [enrich_player_row(r) for r in player_rows if isinstance(r, Mapping)]
    if team_id is not None:
        rows = [r for r in rows if str(r.get("team_id") or "") == str(team_id)]

    skaters = [r for r in rows if not is_goalie_row(r)]
    goalies = [r for r in rows if is_goalie_row(r)]

    gf = sum(safe_int(r.get("g"), 0) for r in skaters)
    assists = sum(safe_int(r.get("a"), 0) for r in skaters)
    pts = sum(safe_int(r.get("pts"), 0) for r in skaters)
    sog = sum(safe_int(r.get("sog"), 0) for r in skaters)
    hits = sum(safe_int(r.get("hit"), 0) for r in skaters)
    blocks = sum(safe_int(r.get("blk"), 0) for r in skaters)
    pim = sum(safe_int(r.get("pim"), 0) for r in skaters)

    ppg = sum(safe_int(r.get("ppg"), 0) for r in skaters)
    ppa = sum(safe_int(r.get("ppa"), 0) for r in skaters)
    shg = sum(safe_int(r.get("shg"), 0) for r in skaters)
    sha = sum(safe_int(r.get("sha"), 0) for r in skaters)

    cf = sum(safe_float(r.get("cf"), 0.0) for r in skaters)
    ca = sum(safe_float(r.get("ca"), 0.0) for r in skaters)
    ff = sum(safe_float(r.get("ff"), 0.0) for r in skaters)
    fa = sum(safe_float(r.get("fa"), 0.0) for r in skaters)
    xgf = sum(safe_float(r.get("xgf"), 0.0) for r in skaters)
    xga_skaters = sum(safe_float(r.get("xga"), 0.0) for r in skaters)

    ga_goalies = sum(safe_int(first_present(r, ["ga", "goals_against"], 0)) for r in goalies)
    sa_goalies = sum(safe_int(first_present(r, ["sa", "shots_against", "goalie_shots_against"], 0)) for r in goalies)
    saves_goalies = sum(
        safe_int(
            first_present(
                r,
                ["saves"],
                max(0, safe_int(first_present(r, ["sa", "shots_against"], 0)) - safe_int(first_present(r, ["ga"], 0))),
            )
        )
        for r in goalies
    )

    team_sh_pct = pct(gf, sog, default=0.0)
    team_sv_pct = pct(saves_goalies, sa_goalies, default=0.0)
    pdo = round((team_sh_pct + team_sv_pct) * 100.0, 1) if sog > 0 and sa_goalies > 0 else None

    return {
        "team_id": str(team_id or (rows[0].get("team_id") if rows else "")),
        "skaters": len(skaters),
        "goalies": len(goalies),
        "gf_player_sum": gf,
        "goals_player_sum": gf,
        "assists_player_sum": assists,
        "points_player_sum": pts,
        "sog": sog,
        "shots": sog,
        "hits": hits,
        "blocks": blocks,
        "pim": pim,

        "ppg": ppg,
        "ppa": ppa,
        "pp_points": ppg + ppa,
        "shg": shg,
        "sha": sha,
        "sh_points": shg + sha,

        "ga_goalie_sum": ga_goalies,
        "shots_against_goalie_sum": sa_goalies,
        "saves_goalie_sum": saves_goalies,

        "team_sh_pct": team_sh_pct,
        "sh_pct": team_sh_pct,
        "team_sv_pct": team_sv_pct,
        "sv_pct": team_sv_pct,
        "pdo": pdo,
        "pdo_valid": pdo is not None,

        "cf": cf,
        "ca": ca,
        "cf_pct": pct(cf, cf + ca, default=0.0) if cf + ca > 0 else None,
        "cf_pct_valid": bool(cf + ca > 0),
        "ff": ff,
        "fa": fa,
        "ff_pct": pct(ff, ff + fa, default=0.0) if ff + fa > 0 else None,
        "ff_pct_valid": bool(ff + fa > 0),
        "xgf": xgf,
        "xga": xga_skaters,
        "xgf_pct": pct(xgf, xgf + xga_skaters, default=0.0) if xgf + xga_skaters > 0 else None,
        "xgf_pct_valid": bool(xgf + xga_skaters > 0),

        "top_scorer": top_row(skaters, "pts"),
        "top_goal_scorer": top_row(skaters, "g"),
        "top_impact_skater": top_row(skaters, "war"),
        "top_goalie": top_row(goalies, "war"),
    }


def enrich_team_game_result_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Enrich a team aggregate row from actual game boxes.

    Expected possible fields:
    gf, ga, sf, sa, wins, losses, otl, points, ppg, ppo, ppga, opp_ppo
    """
    out = dict(row)

    gf = safe_int(first_present(out, ["gf", "goals_for"], 0))
    ga = safe_int(first_present(out, ["ga", "goals_against"], 0))
    sf = safe_int(first_present(out, ["sf", "shots_for", "sog_for"], 0))
    sa = safe_int(first_present(out, ["sa", "shots_against"], 0))

    wins = safe_int(first_present(out, ["wins", "w"], 0))
    losses = safe_int(first_present(out, ["losses", "l"], 0))
    otl = safe_int(first_present(out, ["otl", "ot_losses"], 0))
    gp = safe_int(first_present(out, ["gp", "games_played"], wins + losses + otl))
    points = safe_int(first_present(out, ["points", "pts"], wins * 2 + otl))

    ppg = safe_int(first_present(out, ["ppg", "power_play_goals"], 0))
    ppo = safe_int(first_present(out, ["ppo", "power_play_opportunities"], 0))
    ppga = safe_int(first_present(out, ["ppga", "power_play_goals_against"], 0))
    opp_ppo = safe_int(first_present(out, ["opp_ppo", "opp_power_play_opportunities", "times_shorthanded"], 0))

    cf = safe_float(first_present(out, ["cf", "corsi_for"], 0.0), 0.0)
    ca = safe_float(first_present(out, ["ca", "corsi_against"], 0.0), 0.0)
    ff = safe_float(first_present(out, ["ff", "fenwick_for"], 0.0), 0.0)
    fa = safe_float(first_present(out, ["fa", "fenwick_against"], 0.0), 0.0)
    xgf = safe_float(first_present(out, ["xgf", "expected_goals_for"], 0.0), 0.0)
    xga = safe_float(first_present(out, ["xga", "expected_goals_against"], 0.0), 0.0)
    xgf_pct_gp = safe_int(first_present(out, ["xgf_pct_gp"], 0), 0)
    xgf_pct_sum = safe_float(first_present(out, ["xgf_pct_sum"], 0.0), 0.0)
    team_xgf_pct = (xgf / (xgf + xga)) if (xgf + xga) > 0 else None

    cf_pct = (cf / (cf + ca)) if (cf + ca) > 0 else None
    ff_pct = (ff / (ff + fa)) if (ff + fa) > 0 else None
    sh_pct = pct(gf, sf, default=0.0) if sf > 0 else None
    # Prefer goalie-ledger SV% when present — (sa - standings_ga)/sa invents .96
    # team SV% that doesn't match any actual goalie.
    goalie_sa = safe_int(first_present(out, ["shots_against_goalie_sum"], 0))
    goalie_saves = safe_int(first_present(out, ["saves_goalie_sum"], 0))
    if goalie_sa > 0 and goalie_saves >= 0:
        sv_pct = pct(goalie_saves, goalie_sa, default=0.0)
        sa = goalie_sa
    elif out.get("sv_pct") is not None and safe_float(out.get("sv_pct"), 0.0) > 0:
        sv_pct = safe_float(out.get("sv_pct"), 0.0)
        if sv_pct > 1.5:
            sv_pct = sv_pct / 100.0
    else:
        sv_pct = pct(sa - ga, sa, default=0.0) if sa > 0 else None
    pp_pct = pct(ppg, ppo, default=0.0) if ppo > 0 else None
    pk_pct = (1.0 - pct(ppga, opp_ppo, default=0.0)) if opp_ppo > 0 else None

    goal_diff = gf - ga
    xg_diff = xgf - xga

    pythagorean_win_pct = pct(gf * gf, (gf * gf) + (ga * ga), default=0.0)
    point_pct = pct(points, max(1, gp * 2), default=0.0)
    playoff_pace = point_pct * MAX_STANDINGS_POINTS
    projected_points = playoff_pace

    out.update(
        {
            "gf": gf,
            "goals_for": gf,
            "ga": ga,
            "goals_against": ga,
            "sf": sf,
            "shots_for": sf,
            "sa": sa,
            "shots_against": sa,

            "wins": wins,
            "w": wins,
            "losses": losses,
            "l": losses,
            "otl": otl,
            "gp": gp,
            "games_played": gp,
            "points": points,
            "pts": points,

            "point_pct": point_pct,
            "points_percentage": point_pct,
            "win_pct": pct(wins, gp, default=0.0),
            "pythagorean_win_pct": pythagorean_win_pct,

            "goal_diff": goal_diff,
            "gd": goal_diff,
            "goal_differential": goal_diff,
            "goal_differential_per_game": per_game(goal_diff, gp),

            "gf_per_game": per_game(gf, gp),
            "ga_per_game": per_game(ga, gp),
            "sf_per_game": per_game(sf, gp),
            "sa_per_game": per_game(sa, gp),

            "xgf": xgf,
            "xga": xga,
            "xgf_pct": team_xgf_pct,
            "xgf_pct_gp": xgf_pct_gp,
            "xgf_pct_sum": xgf_pct_sum,
            "expected_goal_differential": xg_diff,
            "expected_goal_differential_per_game": per_game(xg_diff, gp),

            "cf": cf,
            "ca": ca,
            "cf_pct": cf_pct,
            "cf_pct_valid": cf_pct is not None,
            "ff": ff,
            "fa": fa,
            "ff_pct": ff_pct,
            "ff_pct_valid": ff_pct is not None,

            "sh_pct": sh_pct,
            "shooting_pct": sh_pct,
            "sv_pct": sv_pct,
            "save_pct": sv_pct,
            "pdo": round((sh_pct + sv_pct) * 100.0, 1) if sh_pct is not None and sv_pct is not None else None,
            "pdo_valid": sh_pct is not None and sv_pct is not None,

            "ppg": ppg,
            "ppo": ppo,
            "pp_pct": pp_pct,
            "power_play_pct": pp_pct,
            "ppga": ppga,
            "opp_ppo": opp_ppo,
            "pk_pct": pk_pct,
            "penalty_kill_pct": pk_pct,

            "playoff_pace": playoff_pace,
            "projected_points": projected_points,
        }
    )

    clean_round_fields(out)
    return out


def enrich_team_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    enriched = [enrich_team_game_result_row(r) for r in rows if isinstance(r, Mapping)]

    for key, rank_key, reverse, default in [
        ("points", "league_rank_points", True, 0.0),
        ("point_pct", "league_rank_point_pct", True, 0.0),
        ("gf", "league_rank_gf", True, 0.0),
        ("ga", "league_rank_ga", False, 99.0),
        ("goal_diff", "league_rank_goal_diff", True, 0.0),
        ("xgf_pct", "league_rank_xgf_pct", True, 0.0),
        ("cf_pct", "league_rank_cf_pct", True, 0.0),
        ("ff_pct", "league_rank_ff_pct", True, 0.0),
        ("pp_pct", "league_rank_pp", True, 0.0),
        ("pk_pct", "league_rank_pk", True, 0.0),
        ("pdo", "league_rank_pdo", True, 100.0),
    ]:
        _assign_numeric_ranks(enriched, key, rank_key, reverse=reverse, default=default)

    return enriched


# ============================================================
# LEADERS / AWARDS
# ============================================================

def top_row(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    best = max(rows, key=lambda r: safe_float(r.get(key), 0.0))
    return dict(best)


def leaders_by(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    *,
    limit: int = 20,
    reverse: bool = True,
    goalies: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    enriched = [enrich_player_row(r) for r in rows if isinstance(r, Mapping)]

    if goalies is True:
        enriched = [r for r in enriched if is_goalie_row(r)]
    elif goalies is False:
        enriched = [r for r in enriched if not is_goalie_row(r)]

    ordered = sorted(enriched, key=lambda r: safe_float(r.get(key), 0.0), reverse=reverse)
    return ordered[: max(0, int(limit))]


def build_leaderboards(rows: Iterable[Mapping[str, Any]], *, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    enriched = enrich_player_rows(rows)

    return {
        "points": leaders_by(enriched, "pts", limit=limit, goalies=False),
        "goals": leaders_by(enriched, "g", limit=limit, goalies=False),
        "assists": leaders_by(enriched, "a", limit=limit, goalies=False),
        "shots": leaders_by(enriched, "sog", limit=limit, goalies=False),
        "hits": leaders_by(enriched, "hit", limit=limit, goalies=False),
        "blocks": leaders_by(enriched, "blk", limit=limit, goalies=False),

        "analytics_rating": leaders_by(enriched, "analytics_rating", limit=limit, goalies=False),
        "impact": leaders_by(enriched, "war", limit=limit, goalies=False),
        "watr": leaders_by(enriched, "war", limit=limit, goalies=False),
        "war": leaders_by(enriched, "war", limit=limit, goalies=False),
        "value": leaders_by(enriched, "value_score", limit=limit, goalies=False),

        "points_per_game": leaders_by(enriched, "pts_per_game", limit=limit, goalies=False),
        "points_per_60": leaders_by(enriched, "pts_per_60", limit=limit, goalies=False),
        "goals_per_60": leaders_by(enriched, "goals_per_60", limit=limit, goalies=False),
        "power_play_points": leaders_by(enriched, "pp_points", limit=limit, goalies=False),
        "corsi_pct": leaders_by(enriched, "cf_pct", limit=limit, goalies=False),
        "expected_goals_pct": leaders_by(enriched, "xgf_pct", limit=limit, goalies=False),
        "defensive_impact": leaders_by(enriched, "defense_score", limit=limit, goalies=False),

        "goalie_sv_pct": leaders_by(enriched, "sv_pct", limit=limit, goalies=True),
        "goalie_gaa": leaders_by(enriched, "gaa", limit=limit, reverse=False, goalies=True),
        "goalie_wins": leaders_by(enriched, "wins", limit=limit, goalies=True),
        "goalie_gsax": leaders_by(enriched, "gsax", limit=limit, goalies=True),
        "goalie_impact": leaders_by(enriched, "war", limit=limit, goalies=True),
        "goalie_watr": leaders_by(enriched, "war", limit=limit, goalies=True),
    }


def _award_candidate(
    row: Mapping[str, Any],
    score: float,
    formula: str,
    award_name: str,
    *,
    subjective: bool = False,
    trophy_key: str = "",
) -> Dict[str, Any]:
    out = dict(row)
    if subjective:
        out["award_ballot_score"] = round_to(score, 4)
        out["award_subjective"] = True
        out["award_trophy_key"] = trophy_key or award_name
        out["award_rationale"] = formula
        out["award_name"] = award_name
        return out

    out["award_score"] = round_to(score, 2)
    out["award_formula"] = formula
    out["award_name"] = award_name
    return out


# ---------------------------------------------------------------------------
# Subjective NHL trophies (secret ballot — never expose ballot math to UI)
# ---------------------------------------------------------------------------

SUBJECTIVE_TROPHY_KEYS = frozenset(
    {"hart", "norris", "selke", "vezina", "calder", "lady_byng", "ted_lindsay", "conn_smythe"}
)

SUBJECTIVE_TROPHY_PUBLIC_CASE = {
    "hart": "Most valuable all-around season: impact, production, and team success.",
    "norris": "Premier defenseman: two-way impact, usage, and territorial control.",
    "selke": "Elite defensive forward: shutdown play, special teams, and faceoff value.",
    "vezina": "Best goaltender: high-danger saves, consistency, and steal value.",
    "calder": "Top first-year NHL player by position-aware ballot.",
    "lady_byng": "Production with exceptional discipline.",
    "ted_lindsay": "Players' view of most outstanding player.",
    "conn_smythe": "Most valuable playoff performer.",
}


def _awards_mod():
    from app.sim_engine.league import awards as _awards

    return _awards


def standings_rank_map(
    standings_table: Any = None,
    rank_by_tid: Optional[Mapping[str, int]] = None,
) -> Dict[str, int]:
    if rank_by_tid is not None:
        return {str(k): int(v) for k, v in rank_by_tid.items()}

    out: Dict[str, int] = {}
    if standings_table is None:
        return out

    try:
        for i, rec in enumerate(standings_table.league_table()):
            out[str(getattr(rec, "team_id", i))] = int(i)
    except Exception:
        pass
    return out


def hart_ballot_score(row: Mapping[str, Any], rank_by_tid: Mapping[str, int]) -> float:
    return float(_awards_mod().hart_ballot_score(row, rank_by_tid=rank_by_tid))


def norris_ballot_score(row: Mapping[str, Any]) -> float:
    return float(_awards_mod().norris_ballot_score(row))


def selke_ballot_score(row: Mapping[str, Any]) -> float:
    return float(_awards_mod().selke_ballot_score(row))


def vezina_ballot_score(row: Mapping[str, Any]) -> float:
    return float(_awards_mod().vezina_ballot_score(row))


def _subjective_watch_list(
    pool: Sequence[Mapping[str, Any]],
    score_fn: Any,
    *,
    trophy_key: str,
    award_name: str,
    public_case: str,
    min_gp: int = 1,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    eligible = [r for r in pool if safe_int(first_present(r, ["gp", "games_played"]), 0) >= int(min_gp)]
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in eligible:
        ballot = float(score_fn(row))
        scored.append(
            (
                ballot,
                _award_candidate(
                    row,
                    ballot,
                    public_case,
                    award_name,
                    subjective=True,
                    trophy_key=trophy_key,
                ),
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[: max(1, int(limit))]]


def sanitize_awards_watch_for_public(
    award_map: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Strip secret ballot fields before API / frontend payloads."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, rows in award_map.items():
        key_s = str(key or "")
        is_subjective = key_s in SUBJECTIVE_TROPHY_KEYS
        cleaned: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            pub = dict(row)
            pub.pop("award_ballot_score", None)
            if is_subjective or boolish(pub.get("award_subjective")):
                pub.pop("award_score", None)
                pub.pop("award_formula", None)
                pub["award_subjective"] = True
                if not pub.get("award_rationale"):
                    pub["award_rationale"] = SUBJECTIVE_TROPHY_PUBLIC_CASE.get(
                        str(pub.get("award_trophy_key") or key_s),
                        "Voter panel composite case.",
                    )
            cleaned.append(pub)
        out[key_s] = cleaned
    return out


def award_watch_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 10,
    rank_by_tid: Optional[Mapping[str, int]] = None,
    standings_table: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Formula-driven award watch lists.

    Custom franchise boards are intentionally sparse — a handful of named
    flavors beside the official NHL races, not a 35-trophy flood.
    """
    # Keep only these custom boards in Award Watch (plus official NHL keys below).
    core_custom = {
        "alexander_ovechkin_best_goal_scorer",
        "joe_thornton_best_pure_playmaker",
        "patrice_bergeron_best_defensive_forward",
        "bobby_orr_best_offensive_defenseman",
        "nicklas_lidstrom_best_complete_defenseman",
        "teemu_selanne_best_rookie_scorer",
        "dominik_hasek_best_goalie_advanced_analytics",
        "martin_brodeur_best_workhorse_goalie",
    }
    official_keys = {
        "art_ross",
        "rocket",
        "hart",
        "norris",
        "selke",
        "calder",
        "vezina",
        "lady_byng",
        "ted_lindsay",
        "jennings",
        "conn_smythe",
    }
    enriched = enrich_player_rows(rows)
    skaters = [r for r in enriched if not is_goalie_row(r)]
    forwards = [r for r in skaters if not is_defenseman_row(r)]
    defensemen = [r for r in skaters if is_defenseman_row(r)]
    goalies = [r for r in enriched if is_goalie_row(r)]
    rookies = [r for r in skaters if boolish(r.get("rookie") or r.get("is_rookie"))]
    ranks = standings_rank_map(standings_table, rank_by_tid)

    selke_pool = [
        r
        for r in forwards
        if str(r.get("position", "")).upper() not in {"D", "G"}
    ]

    def sort_awards(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(items, key=lambda r: safe_float(r.get("award_score"), 0.0), reverse=True)[:limit]

    award_map: Dict[str, List[Dict[str, Any]]] = {}

    award_map["wayne_gretzky_best_overall_offensive_player"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("offense_score")) * 0.45
            + safe_float(r.get("pts_per_game")) * 25.0
            + safe_float(r.get("xgf_pct")) * 18.0
            + safe_float(r.get("war")) * 4.0,
            "Offense + P/GP + xGF% + WAR",
            "The Wayne Gretzky Award",
        )
        for r in skaters
    ])

    award_map["mario_lemieux_most_dominant_per_game"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("pts_per_game")) * 45.0
            + safe_float(r.get("points_per_60")) * 14.0
            + safe_float(r.get("impact_score")) * 0.35,
            "P/GP + P/60 + impact",
            "The Mario Lemieux Award",
        )
        for r in skaters
    ])

    award_map["alexander_ovechkin_best_goal_scorer"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("g")) * 1.5
            + safe_float(r.get("goals_per_60")) * 22.0
            + safe_float(r.get("shots_per_60")) * 2.0
            + safe_float(r.get("ixg_per_60")) * 12.0,
            "Goals + G/60 + shots + individual xG",
            "The Alexander Ovechkin Award",
        )
        for r in skaters
    ])

    award_map["sidney_crosby_best_complete_forward"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("offense_score")) * 0.30
            + safe_float(r.get("defense_score")) * 0.25
            + safe_float(r.get("possession_score")) * 0.20
            + safe_float(r.get("faceoff_pct")) * 15.0
            + safe_float(r.get("clutch_score")) * 0.10,
            "Offense + defense + possession + faceoffs + clutch",
            "The Sidney Crosby Award",
        )
        for r in forwards
    ])

    award_map["connor_mcdavid_best_transition_rush_player"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("transition_score")) * 0.60
            + safe_float(r.get("controlled_entry_rate")) * 25.0
            + safe_float(r.get("goals_per_60")) * 8.0,
            "Transition impact + controlled entries + rush scoring",
            "The Connor McDavid Award",
        )
        for r in skaters
    ])

    award_map["pavel_datsyuk_best_two_way_skill_forward"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("two_way_score")) * 0.40
            + safe_float(r.get("takeaways_per_60")) * 10.0
            + safe_float(r.get("defense_score")) * 0.25
            + safe_float(r.get("playmaker_score")) * 0.20,
            "Two-way + takeaways + defense + skill",
            "The Pavel Datsyuk Award",
        )
        for r in forwards
    ])

    award_map["patrice_bergeron_best_defensive_forward"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("defense_score")) * 0.45
            + safe_float(r.get("xgf_pct")) * 22.0
            + safe_float(r.get("faceoff_pct")) * 18.0
            + safe_float(r.get("pk_toi_share")) * 15.0,
            "Defensive score + xGF% + faceoffs + PK usage",
            "The Patrice Bergeron Award",
        )
        for r in forwards
    ])

    award_map["anze_kopitar_shutdown_playmaking_center"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("defense_score")) * 0.35
            + safe_float(r.get("playmaker_score")) * 0.25
            + safe_float(r.get("usage_score")) * 0.20
            + safe_float(r.get("faceoff_pct")) * 15.0,
            "Shutdown usage + playmaking + faceoffs",
            "The Anze Kopitar Award",
        )
        for r in forwards
    ])

    award_map["mark_stone_best_takeaway_winger"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("takeaways_per_60")) * 18.0
            + safe_float(r.get("defense_score")) * 0.35
            + safe_float(r.get("penalty_differential_per_60")) * 8.0,
            "Takeaways + defensive impact + discipline differential",
            "The Mark Stone Award",
        )
        for r in forwards
    ])

    award_map["jaromir_jagr_best_puck_protection"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("possession_score")) * 0.35
            + safe_float(r.get("battle_win_pct")) * 30.0
            + safe_float(r.get("giveaways_per_60")) * -6.0
            + safe_float(r.get("points_per_60")) * 12.0,
            "Possession + battle wins + low giveaways + production",
            "The Jaromir Jagr Award",
        )
        for r in skaters
    ])

    award_map["peter_forsberg_best_power_skill_forward"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("power_forward_score")) * 0.35
            + safe_float(r.get("playmaker_score")) * 0.25
            + safe_float(r.get("offense_score")) * 0.25
            + safe_float(r.get("hits_per_60")) * 3.0,
            "Power + skill + playmaking + physicality",
            "The Peter Forsberg Award",
        )
        for r in forwards
    ])

    award_map["eric_lindros_most_physically_dominant_forward"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("power_forward_score")) * 0.50
            + safe_float(r.get("hits_per_60")) * 5.0
            + safe_float(r.get("net_front_chances")) * 0.5,
            "Physical dominance + power-forward scoring",
            "The Eric Lindros Award",
        )
        for r in forwards
    ])

    award_map["joe_thornton_best_pure_playmaker"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("primary_assists_per_60")) * 24.0
            + safe_float(r.get("expected_assists_per_60")) * 18.0
            + safe_float(r.get("shot_assists_per_60")) * 6.0
            + safe_float(r.get("playmaker_score")) * 0.25,
            "Primary assists + xA + shot assists",
            "The Joe Thornton Award",
        )
        for r in skaters
    ])

    award_map["henrik_sedin_best_passing_vision"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("high_danger_passes_per_60")) * 10.0
            + safe_float(r.get("expected_assists_per_60")) * 22.0
            + safe_float(r.get("playmaker_score")) * 0.35,
            "High-danger passing + expected assists + playmaking",
            "The Henrik Sedin Award",
        )
        for r in skaters
    ])

    award_map["brett_hull_best_shooting_efficiency"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("goals_above_expected")) * 3.0
            + safe_float(r.get("shooting_pct")) * 100.0
            + safe_float(r.get("finishing_efficiency")) * 10.0,
            "Goals above expected + shooting percentage + finishing efficiency",
            "The Brett Hull Award",
        )
        for r in skaters
    ])

    award_map["steven_stamkos_best_power_play_shooter"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("ppg")) * 3.0
            + safe_float(r.get("pp_points_per_60")) * 15.0
            + safe_float(r.get("goals_per_60")) * 8.0,
            "PP goals + PP points/60 + shooting",
            "The Steven Stamkos Award",
        )
        for r in skaters
    ])

    award_map["joe_pavelski_best_net_front_player"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("net_front_chances")) * 1.2
            + safe_float(r.get("goals_per_60")) * 12.0
            + safe_float(r.get("individual_xg_per_60")) * 12.0
            + safe_float(r.get("battle_win_pct")) * 15.0,
            "Net-front chances + goals + xG + battle wins",
            "The Joe Pavelski Award",
        )
        for r in forwards
    ])

    award_map["brad_marchand_best_pest_agitator"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("pest_score")) * 0.55
            + safe_float(r.get("penalty_differential_per_60")) * 12.0
            + safe_float(r.get("clutch_score")) * 0.15,
            "Penalty differential + pest impact + clutch scoring",
            "The Brad Marchand Award",
        )
        for r in forwards
    ])

    award_map["teemu_selanne_best_rookie_scorer"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("g")) * 1.5
            + safe_float(r.get("pts")) * 0.8
            + safe_float(r.get("goals_per_60")) * 12.0,
            "Rookie goals + points + goals/60",
            "The Teemu Selanne Award",
        )
        for r in rookies
    ])

    award_map["pavel_bure_most_explosive_goal_threat"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("goals_per_60")) * 25.0
            + safe_float(r.get("transition_score")) * 0.30
            + safe_float(r.get("controlled_entry_rate")) * 20.0,
            "Explosive G/60 + transition + controlled entries",
            "The Pavel Bure Award",
        )
        for r in skaters
    ])

    award_map["bobby_orr_best_offensive_defenseman"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("offensive_defenseman_score")) * 0.45
            + safe_float(r.get("points_per_60")) * 18.0
            + safe_float(r.get("transition_score")) * 0.25,
            "Defenseman offense + points/60 + transition",
            "The Bobby Orr Award",
        )
        for r in defensemen
    ])

    award_map["nicklas_lidstrom_best_complete_defenseman"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("defense_score")) * 0.30
            + safe_float(r.get("possession_score")) * 0.25
            + safe_float(r.get("offense_score")) * 0.20
            + safe_float(r.get("discipline_score")) * 0.15
            + safe_float(r.get("usage_score")) * 0.10,
            "Defense + offense + possession + discipline + usage",
            "The Nicklas Lidstrom Award",
        )
        for r in defensemen
    ])

    award_map["ray_bourque_best_high_volume_defenseman"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("toi")) * 0.08
            + safe_float(r.get("shots_per_60")) * 6.0
            + safe_float(r.get("points_per_60")) * 16.0
            + safe_float(r.get("blocks_per_60")) * 4.0,
            "TOI + shots + points + blocks",
            "The Ray Bourque Award",
        )
        for r in defensemen
    ])

    award_map["zdeno_chara_best_shutdown_physical_defenseman"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("shutdown_defenseman_score")) * 0.45
            + safe_float(r.get("hits_per_60")) * 5.0
            + safe_float(r.get("blocks_per_60")) * 4.0
            + safe_float(r.get("xga_per_60")) * -6.0,
            "Shutdown defense + hits + blocks + xGA suppression",
            "The Zdeno Chara Award",
        )
        for r in defensemen
    ])

    award_map["scott_stevens_hardest_hitter"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("hits_per_60")) * 10.0
            + safe_float(r.get("physical_score")) * 0.35,
            "Hits/60 + physical impact",
            "The Scott Stevens Award",
        )
        for r in skaters
    ])

    award_map["chris_pronger_nastiest_defensive_impact"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("shutdown_defenseman_score")) * 0.35
            + safe_float(r.get("physical_score")) * 0.25
            + safe_float(r.get("defense_score")) * 0.25
            - safe_float(r.get("pim_per_60")) * 2.0,
            "Shutdown + physicality + defensive impact with nastiness",
            "The Chris Pronger Award",
        )
        for r in defensemen
    ])

    award_map["erik_karlsson_best_transition_defenseman"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("transition_score")) * 0.50
            + safe_float(r.get("controlled_exit_rate")) * 25.0
            + safe_float(r.get("points_per_60")) * 14.0,
            "Transition defense + exits + offensive creation",
            "The Erik Karlsson Award",
        )
        for r in defensemen
    ])

    award_map["cale_makar_best_modern_offensive_defense_impact"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("analytics_rating")) * 0.30
            + safe_float(r.get("offense_score")) * 0.25
            + safe_float(r.get("transition_score")) * 0.25
            + safe_float(r.get("xgf_pct")) * 20.0,
            "Modern D impact: offense + transition + xGF%",
            "The Cale Makar Award",
        )
        for r in defensemen
    ])

    award_map["shea_weber_best_point_shot_defenseman"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("shots_per_60")) * 7.0
            + safe_float(r.get("ppg")) * 3.0
            + safe_float(r.get("goals_per_60")) * 12.0,
            "Shot volume + PP goals + D scoring",
            "The Shea Weber Award",
        )
        for r in defensemen
    ])

    award_map["kris_letang_best_high_risk_high_reward_defenseman"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("offense_score")) * 0.35
            + safe_float(r.get("transition_score")) * 0.25
            + safe_float(r.get("giveaways_per_60")) * 6.0
            + safe_float(r.get("points_per_60")) * 14.0,
            "Offense + transition + chaos",
            "The Kris Letang Award",
        )
        for r in defensemen
    ])

    award_map["dominik_hasek_best_goalie_advanced_analytics"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("goalie_analytics_rating")) * 0.45
            + safe_float(r.get("gsax")) * 2.0
            + safe_float(r.get("high_danger_save_pct")) * 35.0
            + safe_float(r.get("steal_rate")) * 20.0,
            "Goalie analytics + GSAx + high-danger saves + steals",
            "The Dominik Hasek Award",
        )
        for r in goalies
    ])

    award_map["martin_brodeur_best_workhorse_goalie"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("wins")) * 1.5
            + safe_float(r.get("starts")) * 0.8
            + safe_float(r.get("workload_score")) * 0.35
            + safe_float(r.get("quality_start_pct")) * 25.0,
            "Wins + starts + workload + quality starts",
            "The Martin Brodeur Award",
        )
        for r in goalies
    ])

    award_map["patrick_roy_best_clutch_playoff_goalie"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("clutch_score")) * 0.55
            + safe_float(r.get("clutch_save_pct")) * 30.0
            + safe_float(r.get("gsax_per_game")) * 10.0,
            "Clutch saves + clutch save percentage + GSAx/game",
            "The Patrick Roy Award",
        )
        for r in goalies
    ])

    award_map["henrik_lundqvist_best_goalie_carrying_weak_team"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("gsax")) * 2.2
            + safe_float(r.get("shots_against_per_game")) * 1.4
            + safe_float(r.get("sv_pct")) * 45.0
            - safe_float(r.get("wins")) * 0.3,
            "GSAx + workload + save percentage with limited win support",
            "The Henrik Lundqvist Award",
        )
        for r in goalies
    ])

    award_map["carey_price_best_technical_goalie"] = sort_awards([
        _award_candidate(
            r,
            safe_float(r.get("save_percentage_score")) * 0.30
            + safe_float(r.get("rebound_score")) * 0.25
            + safe_float(r.get("high_danger_save_score")) * 0.25
            + safe_float(r.get("quality_start_score")) * 0.20,
            "Save mechanics + rebound control + high-danger saves + consistency",
            "The Carey Price Award",
        )
        for r in goalies
    ])

    # Official NHL trophies — same formulas as compute_awards / Awards Night.
    try:
        official = _awards_mod().compute_official_watch_lists(
            enriched,
            standings=standings_table,
            rank_by_tid=ranks,
            limit=limit,
        )
        for key, watch_rows in official.items():
            tagged = []
            for wr in watch_rows:
                row = dict(wr)
                row.setdefault("official", True)
                row.setdefault("ceremony_enabled", True)
                tagged.append(row)
            award_map[key] = tagged
    except Exception:
        award_map["art_ross"] = leaders_by(skaters, "pts", limit=limit, goalies=False)
        for r in award_map["art_ross"]:
            r["official"] = True
            r["watch_type"] = "official_live_race"
            r["display_metric"] = "PTS"

    # Label custom named boards distinctly from official trophies.
    for key, watch_rows in list(award_map.items()):
        if key in {
            "art_ross",
            "rocket",
            "hart",
            "norris",
            "selke",
            "calder",
            "vezina",
            "lady_byng",
            "ted_lindsay",
            "jennings",
            "conn_smythe",
        }:
            continue
        for r in watch_rows:
            if isinstance(r, dict):
                r.setdefault("official", False)
                r.setdefault("watch_type", "custom_franchise_award")
                r.setdefault("ceremony_enabled", False)

    # Massive cut: drop the long-tail named boards; keep a short core set.
    award_map = {
        k: v
        for k, v in award_map.items()
        if k in official_keys or k in core_custom
    }

    return sanitize_awards_watch_for_public(award_map)


# ============================================================
# STREAK / REGRESSION / DURABILITY HELPERS
# ============================================================

def calculate_hot_cold_streak(
    season_row: Mapping[str, Any],
    recent_row: Optional[Mapping[str, Any]] = None,
    *,
    recent_games: int = 5,
) -> Dict[str, Any]:
    """
    Optional helper if you later pass recent last-N-game rows.

    If recent_row is None, returns neutral streak values.
    """
    if not recent_row:
        return {
            "recent_games": 0,
            "recent_points_per_game": 0.0,
            "season_points_per_game": safe_float(first_present(season_row, ["pts_per_game", "points_per_game"], 0.0), 0.0),
            "hot_streak_score": 0.0,
            "finishing_streak": 0.0,
            "streak_label": "Neutral",
        }

    recent_gp = safe_int(first_present(recent_row, ["gp", "games_played", "recent_games"], recent_games))
    recent_pts = safe_int(first_present(recent_row, ["pts", "points"], 0))
    recent_goals = safe_int(first_present(recent_row, ["g", "goals"], 0))
    recent_shots = safe_int(first_present(recent_row, ["sog", "shots"], 0))

    recent_ppg = per_game(recent_pts, recent_gp)
    season_ppg = safe_float(first_present(season_row, ["pts_per_game", "points_per_game"], 0.0), 0.0)

    recent_sh_pct = pct(recent_goals, recent_shots, default=0.0)
    season_sh_pct = safe_float(first_present(season_row, ["shooting_pct", "sh_pct"], 0.0), 0.0)

    hot_score = recent_ppg - season_ppg
    finishing_streak = recent_sh_pct - season_sh_pct

    if hot_score >= 0.35:
        label = "Hot"
    elif hot_score <= -0.35:
        label = "Cold"
    else:
        label = "Neutral"

    return {
        "recent_games": recent_gp,
        "recent_points_per_game": round_to(recent_ppg, 3),
        "season_points_per_game": round_to(season_ppg, 3),
        "hot_streak_score": round_to(hot_score, 3),
        "finishing_streak": round_to(finishing_streak, 3),
        "streak_label": label,
    }


def calculate_durability_analytics(row: Mapping[str, Any]) -> Dict[str, Any]:
    gp = safe_float(first_present(row, ["gp", "games_played"], 0.0), 0.0)
    games_missed = safe_float(first_present(row, ["games_missed"], 0.0), 0.0)
    team_games = max(gp + games_missed, gp, 1.0)

    availability = pct(gp, team_games, default=0.0)
    games_missed_rate = pct(games_missed, team_games, default=0.0)

    stamina_rating = safe_float(first_present(row, ["stamina_rating", "stamina"], 50.0), 50.0)
    injury_resistance_rating = safe_float(first_present(row, ["injury_resistance_rating", "durability"], 50.0), 50.0)
    previous_injuries = safe_float(first_present(row, ["previous_injuries", "injury_count"], 0.0), 0.0)
    age = safe_float(first_present(row, ["age"], 27.0), 27.0)
    avg_toi = per_game(safe_float(first_present(row, ["toi_sec"], 0.0), 0.0) / 60.0, gp)

    pos = normalize_position(first_present(row, ["position", "pos"], "F"))
    expected_toi = 22.0 if pos == "D" else 17.0
    workload_risk = pct(avg_toi, expected_toi, default=0.0)

    age_risk = max(0.0, age - 31.0) * 0.8

    durability_score = clamp(
        availability * 70.0
        + stamina_rating * 0.20
        + injury_resistance_rating * 0.10,
        0.0,
        100.0,
    )

    injury_risk = clamp(
        previous_injuries * 8.0
        + age_risk
        + max(0.0, workload_risk - 1.0) * 20.0
        + games_missed_rate * 30.0
        + max(0.0, 50.0 - injury_resistance_rating) * 0.20,
        0.0,
        100.0,
    )

    return {
        "availability": round_to(availability, 3),
        "games_missed_rate": round_to(games_missed_rate, 3),
        "durability_score": round_to(durability_score, 2),
        "injury_risk": round_to(injury_risk, 2),
        "injury_risk_score": round_to(injury_risk, 2),
        "workload_risk": round_to(workload_risk, 3),
    }


# ============================================================
# PAYLOAD HELPERS
# ============================================================

def build_stats_central_player_payload(
    player_rows: Iterable[Mapping[str, Any]],
    *,
    user_team_id: Optional[str] = None,
    leader_limit: int = 100,
) -> Dict[str, Any]:
    """
    Build a frontend-ready stats_central player section.

    You can call this from backend/services/franchise_sim.py once you have
    session.player_season_stats values.
    """
    enriched = enrich_player_rows(player_rows)

    skaters = [r for r in enriched if not is_goalie_row(r)]
    goalies = [r for r in enriched if is_goalie_row(r)]

    skaters_sorted = sorted(
        skaters,
        key=lambda r: (
            safe_float(r.get("pts"), 0.0),
            safe_float(r.get("g"), 0.0),
            safe_float(r.get("analytics_rating"), 0.0),
            safe_float(r.get("impact_score"), 0.0),
        ),
        reverse=True,
    )

    analytics_sorted = sorted(
        skaters,
        key=lambda r: (
            safe_float(r.get("analytics_rating"), 0.0),
            safe_float(r.get("war"), 0.0),
            safe_float(r.get("impact_score"), 0.0),
        ),
        reverse=True,
    )

    goalies_sorted = sorted(
        goalies,
        key=lambda r: (
            safe_float(r.get("impact_score"), 0.0),
            safe_float(r.get("sv_pct"), 0.0),
            safe_float(r.get("gsax"), 0.0),
            -safe_float(r.get("gaa"), 99.0),
        ),
        reverse=True,
    )

    if user_team_id is not None:
        user_skaters = [r for r in skaters_sorted if str(r.get("team_id") or "") == str(user_team_id)]
        user_goalies = [r for r in goalies_sorted if str(r.get("team_id") or "") == str(user_team_id)]
    else:
        user_skaters = []
        user_goalies = []

    return {
        "league_leaders": skaters_sorted[: max(1, int(leader_limit))],
        "league_analytics_leaders": analytics_sorted[: max(1, int(leader_limit))],
        "league_goalies": goalies_sorted[: max(1, int(leader_limit))],

        "user_team_skaters": user_skaters,
        "user_team_goalies": user_goalies,

        "skaters": skaters_sorted,
        "goalies": goalies_sorted,

        "leaders": build_leaderboards(enriched, limit=10),
        "awards_watch": award_watch_scores(enriched, limit=10),
    }


def build_full_analytics_payload(
    player_rows: Iterable[Mapping[str, Any]],
    *,
    team_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    user_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-call helper for backend payloads.
    """
    player_payload = build_stats_central_player_payload(
        player_rows,
        user_team_id=user_team_id,
        leader_limit=100,
    )

    team_payload: List[Dict[str, Any]] = []
    if team_rows is not None:
        team_payload = enrich_team_rows(team_rows)

    user_team_stats = None
    if user_team_id is not None:
        for row in team_payload:
            if str(row.get("team_id") or row.get("id") or "") == str(user_team_id):
                user_team_stats = row
                break

    if user_team_stats is None and user_team_id is not None:
        all_players = player_payload["skaters"] + player_payload["goalies"]
        user_team_stats = aggregate_team_from_player_rows(all_players, team_id=str(user_team_id))

    return {
        **player_payload,
        "league_team_stats": team_payload,
        "team_team_stats": user_team_stats or {},
        "user_team_stats": user_team_stats or {},
    }


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

def enrich_stat_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return enrich_player_row(row)


def enrich_stat_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return enrich_player_rows(rows)


def calculate_impact_score(row: Mapping[str, Any]) -> float:
    enriched = enrich_player_row(row)
    return safe_float(enriched.get("impact_score"), 0.0)


def calculate_points(goals: Any, assists: Any) -> int:
    return safe_int(goals, 0) + safe_int(assists, 0)


def calculate_goalie_sv_pct(saves: Any, shots_against: Any) -> float:
    return pct(saves, shots_against, default=0.0)


def calculate_goalie_gaa(goals_against: Any, toi_sec: Any) -> float:
    sec = safe_float(toi_sec, 0.0)
    if sec <= 0:
        return 0.0
    return safe_float(goals_against, 0.0) * 3600.0 / sec


def calculate_corsi_pct(cf: Any, ca: Any) -> float:
    return pct(cf, safe_float(cf) + safe_float(ca), default=0.0)


def calculate_fenwick_pct(ff: Any, fa: Any) -> float:
    return pct(ff, safe_float(ff) + safe_float(fa), default=0.0)


def calculate_xgf_pct(xgf: Any, xga: Any) -> float:
    """Single-game or cumulative ratio; prefer season_xgf_pct_from_row() for season rows."""
    return pct(xgf, safe_float(xgf) + safe_float(xga), default=0.0)


def calculate_pdo(sh_pct_value: Any, sv_pct_value: Any, *, scaled: bool = True) -> float:
    raw = safe_float(sh_pct_value, 0.0) + safe_float(sv_pct_value, 0.0)
    return raw * 100.0 if scaled else raw


def calculate_pythagorean_win_pct(goals_for: Any, goals_against: Any) -> float:
    gf = safe_float(goals_for, 0.0)
    ga = safe_float(goals_against, 0.0)
    return pct(gf * gf, gf * gf + ga * ga, default=0.0)


def calculate_gsax(expected_goals_against: Any, goals_against: Any) -> float:
    return safe_float(expected_goals_against, 0.0) - safe_float(goals_against, 0.0)


__all__ = [
    "safe_float",
    "safe_int",
    "clamp",
    "pct",
    "pct100",
    "per_game",
    "per_60",
    "per_82",

    "calculate_points",
    "calculate_goalie_sv_pct",
    "calculate_goalie_gaa",
    "calculate_corsi_pct",
    "calculate_fenwick_pct",
    "calculate_xgf_pct",
    "calculate_pdo",
    "calculate_pythagorean_win_pct",
    "calculate_gsax",
    "calculate_impact_score",

    "normalize_player_identity",
    "normalize_skater_counting_stats",
    "normalize_goalie_counting_stats",

    "calculate_skater_rates",
    "calculate_skater_component_scores",
    "calculate_skater_analytics_rating",
    "calculate_skater_value_metrics",
    "calculate_archetype_scores",

    "calculate_goalie_rates",
    "calculate_goalie_component_scores",

    "calculate_hot_cold_streak",
    "calculate_durability_analytics",

    "enrich_skater_row",
    "enrich_goalie_row",
    "enrich_player_row",
    "enrich_player_rows",
    "enrich_stat_row",
    "enrich_stat_rows",

    "aggregate_team_from_player_rows",
    "enrich_team_game_result_row",
    "enrich_team_rows",

    "leaders_by",
    "build_leaderboards",
    "award_watch_scores",
    "sanitize_awards_watch_for_public",
    "SUBJECTIVE_TROPHY_KEYS",
    "hart_ballot_score",
    "norris_ballot_score",
    "selke_ballot_score",
    "vezina_ballot_score",

    "build_stats_central_player_payload",
    "build_full_analytics_payload",
]