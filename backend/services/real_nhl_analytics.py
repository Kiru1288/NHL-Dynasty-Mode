"""
Advanced analytics for Real NHL ratings (Option R3).

Primary free source: MoneyPuck season-summary CSVs (NHL playerId keys).

Skaters: 5v5 xGF%, Corsi, gameScore — support for D / two-way identity.
Goalies: all-situations xGoals vs goals → GSAx (goals saved above expected),
plus high-danger GSAx — the main team-context signal beyond raw SV%/GAA.

Natural Stat Trick / hockey-statistics.com are not hard dependencies here.
Credit: MoneyPuck.com (unofficial fan use).
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

USER_AGENT = "NHLFranchiseMode/0.2 (unofficial-fan-project; local-sim; MoneyPuck-credit)"
MONEYPUCK_SKATERS = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv"
)
MONEYPUCK_GOALIES = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/goalies.csv"
)
HTTP_TIMEOUT_S = 60


def _http_get_text(url: str, *, timeout: float = HTTP_TIMEOUT_S) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) if row.get(key) not in (None, "") else default)
    except Exception:
        return float(default)


def parse_moneypuck_skaters_csv(text: str) -> Dict[int, Dict[str, Any]]:
    """
    Prefer 5-on-5 rows; fall back to all-situations.
    Returns nhl_player_id → analytics dict.
    """
    reader = csv.DictReader(io.StringIO(text))
    five: Dict[int, Dict[str, Any]] = {}
    all_sit: Dict[int, Dict[str, Any]] = {}
    for row in reader:
        try:
            pid = int(float(row.get("playerId") or 0))
        except Exception:
            continue
        if pid <= 0:
            continue
        situ = str(row.get("situation") or "").lower()
        packed = {
            "playerId": pid,
            "name": row.get("name"),
            "team": row.get("team"),
            "position": row.get("position"),
            "situation": situ,
            "games_played": _f(row, "games_played"),
            "icetime": _f(row, "icetime"),
            "gameScore": _f(row, "gameScore"),
            "onIce_xGoalsPercentage": _f(row, "onIce_xGoalsPercentage"),
            "offIce_xGoalsPercentage": _f(row, "offIce_xGoalsPercentage"),
            "onIce_corsiPercentage": _f(row, "onIce_corsiPercentage"),
            "offIce_corsiPercentage": _f(row, "offIce_corsiPercentage"),
            "OnIce_F_xGoals": _f(row, "OnIce_F_xGoals"),
            "OnIce_A_xGoals": _f(row, "OnIce_A_xGoals"),
            "I_F_xGoals": _f(row, "I_F_xGoals"),
        }
        if situ == "5on5":
            five[pid] = packed
        elif situ == "all":
            all_sit[pid] = packed
    out = dict(all_sit)
    out.update(five)
    return out


def parse_moneypuck_goalies_csv(text: str) -> Dict[int, Dict[str, Any]]:
    """
    Prefer all-situations rows (full GSAx sample); fall back to 5on5.
    GSAx = xGoals − goals (MoneyPuck goals column is goals against).
    """
    reader = csv.DictReader(io.StringIO(text))
    five: Dict[int, Dict[str, Any]] = {}
    all_sit: Dict[int, Dict[str, Any]] = {}
    for row in reader:
        try:
            pid = int(float(row.get("playerId") or 0))
        except Exception:
            continue
        if pid <= 0:
            continue
        situ = str(row.get("situation") or "").lower()
        xg = _f(row, "xGoals")
        ga = _f(row, "goals")
        hd_xg = _f(row, "highDangerxGoals")
        hd_ga = _f(row, "highDangerGoals")
        packed = {
            "playerId": pid,
            "name": row.get("name"),
            "team": row.get("team"),
            "position": "G",
            "situation": situ,
            "games_played": _f(row, "games_played"),
            "icetime": _f(row, "icetime"),
            "xGoals": xg,
            "goals": ga,
            "gsax": xg - ga,
            "highDangerxGoals": hd_xg,
            "highDangerGoals": hd_ga,
            "hd_gsax": hd_xg - hd_ga,
            "ongoal": _f(row, "ongoal"),
            "kind": "goalie",
        }
        if situ == "all":
            all_sit[pid] = packed
        elif situ == "5on5":
            five[pid] = packed
    out = dict(five)
    out.update(all_sit)
    return out


def fetch_moneypuck_skater_analytics(season_start_year: int) -> Dict[int, Dict[str, Any]]:
    """Load MoneyPuck skater summary for a season start year (2025 → 2025–26)."""
    year = int(season_start_year)
    url = MONEYPUCK_SKATERS.format(year=year)
    try:
        text = _http_get_text(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return {}
    try:
        return parse_moneypuck_skaters_csv(text)
    except Exception:
        return {}


def fetch_moneypuck_goalie_analytics(season_start_year: int) -> Dict[int, Dict[str, Any]]:
    """Load MoneyPuck goalie summary (GSAx) for a season start year."""
    year = int(season_start_year)
    url = MONEYPUCK_GOALIES.format(year=year)
    try:
        text = _http_get_text(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return {}
    try:
        return parse_moneypuck_goalies_csv(text)
    except Exception:
        return {}


def _prefer_merge(
    primary: Dict[int, Dict[str, Any]],
    secondary: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    out = dict(secondary)
    for pid, row in primary.items():
        if float(row.get("games_played") or 0) >= 8 or pid not in out:
            out[pid] = row
        elif float((out.get(pid) or {}).get("games_played") or 0) < float(row.get("games_played") or 0):
            out[pid] = row
    return out


def fetch_moneypuck_skater_analytics_prefer(
    primary_year: int,
    secondary_year: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Merge primary over secondary so newer season wins on collisions."""
    secondary = fetch_moneypuck_skater_analytics(int(secondary_year)) if secondary_year else {}
    primary = fetch_moneypuck_skater_analytics(int(primary_year))
    return _prefer_merge(primary, secondary)


def fetch_moneypuck_goalie_analytics_prefer(
    primary_year: int,
    secondary_year: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    secondary = fetch_moneypuck_goalie_analytics(int(secondary_year)) if secondary_year else {}
    primary = fetch_moneypuck_goalie_analytics(int(primary_year))
    return _prefer_merge(primary, secondary)


def analytics_impact_score(analytics: Optional[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Map MoneyPuck possession/xG into a 0–1 impact score.
    High xGF% + positive relative xGF% → strong shutdown / two-way signal.
    """
    if not analytics:
        return 0.50, "no_mp"

    gp = float(analytics.get("games_played") or 0)
    xgf = float(analytics.get("onIce_xGoalsPercentage") or 0.5)
    xgf_off = float(analytics.get("offIce_xGoalsPercentage") or 0.5)
    cf = float(analytics.get("onIce_corsiPercentage") or 0.5)
    gs = float(analytics.get("gameScore") or 0.0)
    toi_sec = float(analytics.get("icetime") or 0.0)
    toi_min_pg = (toi_sec / 60.0) / max(gp, 1.0) if gp > 0 else 0.0

    xgf_score = max(0.0, min(1.0, (xgf - 0.42) / 0.20))
    rel = xgf - xgf_off
    rel_score = max(0.0, min(1.0, (rel + 0.02) / 0.12))
    cf_score = max(0.0, min(1.0, (cf - 0.42) / 0.20))
    gs_pg = gs / max(gp, 1.0)
    gs_score = max(0.0, min(1.0, (gs_pg - 0.15) / 1.05))
    usage = max(0.0, min(1.0, (toi_min_pg - 12.0) / 12.0))

    raw = (
        xgf_score * 0.42
        + rel_score * 0.22
        + cf_score * 0.16
        + gs_score * 0.12
        + usage * 0.08
    )
    sample = max(0.0, min(1.0, gp / 50.0))
    score = 0.50 + (raw - 0.50) * (0.35 + 0.65 * sample)
    note = (
        f"mp_xgf={xgf:.3f}_rel={rel:+.3f}_cf={cf:.3f}_gspg={gs_pg:.2f}_gp={int(gp)}"
    )
    return max(0.0, min(1.0, score)), note


def goalie_analytics_impact_score(analytics: Optional[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Map MoneyPuck GSAx into 0–1 goalie impact (team-context quality).
    Positive GSAx = saved more than expected given shot quality faced.
    """
    if not analytics:
        return 0.50, "no_mp_g"

    gp = float(analytics.get("games_played") or 0)
    if analytics.get("gsax") is not None:
        gsax = float(analytics.get("gsax") or 0)
    else:
        gsax = float(analytics.get("xGoals") or 0) - float(analytics.get("goals") or 0)
    if analytics.get("hd_gsax") is not None:
        hd = float(analytics.get("hd_gsax") or 0)
    else:
        hd = float(analytics.get("highDangerxGoals") or 0) - float(analytics.get("highDangerGoals") or 0)
    gsax_pg = gsax / max(gp, 1.0)
    hd_pg = hd / max(gp, 1.0)
    # ~-0.20/g → 0, ~0.55/g (Hellebuyck) → 1
    gsax_score = max(0.0, min(1.0, (gsax_pg + 0.20) / 0.75))
    hd_score = max(0.0, min(1.0, (hd_pg + 0.08) / 0.45))
    raw = gsax_score * 0.72 + hd_score * 0.28
    sample = max(0.0, min(1.0, gp / 48.0))
    score = 0.50 + (raw - 0.50) * (0.40 + 0.60 * sample)
    note = f"mp_gsax={gsax:+.1f}_pg={gsax_pg:+.2f}_hd={hd:+.1f}_gp={int(gp)}"
    return max(0.0, min(1.0, score)), note
