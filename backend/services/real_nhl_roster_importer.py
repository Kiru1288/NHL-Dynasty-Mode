"""
Live NHL roster importer (Option B) + usage-aware ratings (R2) + MoneyPuck
analytics (R3) + curated star overrides (R4).

Fetches current NHL.com rosters and recent regular-season stats, maps each player
into a SimEngine Player. Target OVR uses boxcars + ice-time role + PP/EV mix +
team usage rank + MoneyPuck 5v5 xG/Corsi (especially for defensive D). R4 JSON
pins elites.

Also attaches:
- height / weight from NHL roster (+ landing confirmation)
- draft year / round / overall / drafting club from player landing
- AAV / years remaining / UFA-RFA from Spotrac multi-year tables

AHL / juniors / UFAs stay generated via bootstrap_full_league_hierarchy after this runs.
"""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

USER_AGENT = "NHLFranchiseMode/0.2 (unofficial-fan-project; local-sim)"
WEB_API = "https://api-web.nhle.com/v1"
STATS_API = "https://api.nhle.com/stats/rest/en"
HTTP_TIMEOUT_S = 45
R4_OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "data" / "real_nhl_r4_overrides.json"
_PLAYER_LANDING_CACHE: Dict[int, Dict[str, Any]] = {}


def _real_nhl_fast_import_enabled() -> bool:
    """Fast path: dynasty txt ratings + roster API only (~30s start vs ~2min full scrape)."""
    raw = os.environ.get("NHL_REAL_IMPORT_FAST", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _team_abbr(team: Any) -> str:
    return str(
        getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or ""
    ).upper()


def _fetch_team_roster_bundle(
    team: Any,
    roster_season: int,
    stats_primary_id: int,
) -> Tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    abbr = _team_abbr(team)
    if not abbr:
        return "", None, [], f"missing abbr for {getattr(team, 'city', '?')}"
    try:
        payload, rows, _note = _fetch_merged_team_roster(abbr, roster_season, stats_primary_id)
        return abbr, payload, rows, None
    except RealNhlImportError as e:
        return abbr, None, [], f"{abbr}: {e.message}"


class RealNhlImportError(Exception):
    def __init__(self, message: str, *, code: str = "REAL_NHL_ROSTER_IMPORT_FAILED"):
        super().__init__(message)
        self.message = message
        self.code = code


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _http_get_json(url: str, *, timeout: float = HTTP_TIMEOUT_S) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RealNhlImportError(
            f"NHL API HTTP {e.code} for {url}",
            code="REAL_NHL_HTTP_ERROR",
        ) from e
    except urllib.error.URLError as e:
        raise RealNhlImportError(
            f"NHL API unreachable ({e.reason}). Check network or retry.",
            code="REAL_NHL_NETWORK_ERROR",
        ) from e
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RealNhlImportError(
            "NHL API returned invalid JSON.",
            code="REAL_NHL_BAD_JSON",
        ) from e


def _season_id(start_year: int) -> int:
    y = int(start_year)
    return y * 10000 + (y + 1)


def _localized_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("default") or next(iter(value.values()), "") or "").strip()
    return str(value or "").strip()


# ---------------------------------------------------------------------------
# Stats fetch + R2 usage-aware target OVR
# ---------------------------------------------------------------------------


def _fetch_skater_report(season_id: int, report: str) -> Dict[int, Dict[str, Any]]:
    exp = urllib.parse.quote(f"seasonId={int(season_id)} and gameTypeId=2")
    url = f"{STATS_API}/skater/{report}?cayenneExp={exp}&limit=-1"
    payload = _http_get_json(url)
    out: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("data") or []:
        try:
            pid = int(row.get("playerId"))
        except Exception:
            continue
        out[pid] = row
    return out


def _fetch_skater_summary(season_id: int) -> Dict[int, Dict[str, Any]]:
    """Summary boxcars + scoringpergame usage extras (hits/blocks/TOI)."""
    summary = _fetch_skater_report(season_id, "summary")
    try:
        scoring = _fetch_skater_report(season_id, "scoringpergame")
    except Exception:
        scoring = {}
    for pid, row in summary.items():
        extra = scoring.get(pid) or {}
        if extra.get("hits") is not None:
            row["hits"] = extra.get("hits")
        if extra.get("blockedShots") is not None:
            row["blockedShots"] = extra.get("blockedShots")
        if extra.get("hitsPerGame") is not None:
            row["hitsPerGame"] = extra.get("hitsPerGame")
        if extra.get("blocksPerGame") is not None:
            row["blocksPerGame"] = extra.get("blocksPerGame")
        # Prefer per-game TOI from summary; scoringpergame TOI is season total seconds.
        if not row.get("timeOnIcePerGame") and extra.get("timeOnIce") and row.get("gamesPlayed"):
            try:
                gp = float(row.get("gamesPlayed") or 0)
                if gp > 0:
                    row["timeOnIcePerGame"] = float(extra["timeOnIce"]) / gp
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
    _attach_team_toi_ranks(summary)
    return summary


def _fetch_goalie_summary(season_id: int) -> Dict[int, Dict[str, Any]]:
    exp = urllib.parse.quote(f"seasonId={int(season_id)} and gameTypeId=2")
    url = f"{STATS_API}/goalie/summary?cayenneExp={exp}&limit=-1"
    payload = _http_get_json(url)
    out: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("data") or []:
        try:
            pid = int(row.get("playerId"))
        except Exception:
            continue
        out[pid] = row
    _attach_team_goalie_workload_ranks(out)
    return out


def _primary_team_abbrev(raw: Any) -> str:
    text = str(raw or "").upper().replace(" ", "")
    if not text:
        return ""
    # "TOR,BOS" / "TOR, BOS" → last club (most recent in NHL feed)
    parts = [p for p in text.replace(";", ",").split(",") if p]
    return parts[-1] if parts else ""


def _toi_minutes_per_game(stats: Dict[str, Any]) -> float:
    toi_sec = float(stats.get("timeOnIcePerGame") or 0)
    if toi_sec <= 0:
        return 0.0
    toi_min = toi_sec / 60.0 if toi_sec > 45 else toi_sec
    if toi_min > 45:
        toi_min = toi_sec / 60.0
    return float(toi_min)


def _attach_team_toi_ranks(stats_by_id: Dict[int, Dict[str, Any]]) -> None:
    """Add team_toi_rank_pct (0–1, higher = more ice) among skaters on the same club."""
    by_team: Dict[str, List[Tuple[int, float]]] = {}
    for pid, row in stats_by_id.items():
        abbr = _primary_team_abbrev(row.get("teamAbbrevs"))
        if not abbr:
            continue
        toi = _toi_minutes_per_game(row)
        if toi <= 0:
            continue
        by_team.setdefault(abbr, []).append((pid, toi))
    for abbr, pairs in by_team.items():
        pairs.sort(key=lambda x: x[1])
        n = len(pairs)
        if n <= 1:
            for pid, _ in pairs:
                stats_by_id[pid]["team_toi_rank_pct"] = 0.5
            continue
        for i, (pid, _) in enumerate(pairs):
            stats_by_id[pid]["team_toi_rank_pct"] = i / (n - 1)


def _attach_team_goalie_workload_ranks(stats_by_id: Dict[int, Dict[str, Any]]) -> None:
    by_team: Dict[str, List[Tuple[int, float]]] = {}
    for pid, row in stats_by_id.items():
        abbr = _primary_team_abbrev(row.get("teamAbbrevs"))
        if not abbr:
            continue
        gs = float(row.get("gamesStarted") or row.get("gamesPlayed") or 0)
        by_team.setdefault(abbr, []).append((pid, gs))
    for abbr, pairs in by_team.items():
        pairs.sort(key=lambda x: x[1])
        n = len(pairs)
        for i, (pid, _) in enumerate(pairs):
            stats_by_id[pid]["team_start_rank_pct"] = (i / (n - 1)) if n > 1 else 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def infer_skater_profile(stats: Optional[Dict[str, Any]], *, position_code: str) -> str:
    """Pick a role-shaped profile from usage mix (R2)."""
    pos = str(position_code or "C").upper()
    is_d = pos == "D"
    if not stats:
        return "two_way_d" if is_d else "two_way"

    gp = max(float(stats.get("gamesPlayed") or 1), 1.0)
    goals = float(stats.get("goals") or 0)
    assists = float(stats.get("assists") or 0)
    pts = float(stats.get("points") or 0) or (goals + assists)
    hits = float(stats.get("hits") or 0)
    blocks = float(stats.get("blockedShots") or 0)
    pp = float(stats.get("ppPoints") or 0)
    hpg = hits / gp
    bpg = blocks / gp
    g_share = goals / max(pts, 1.0)
    a_share = assists / max(pts, 1.0)
    pp_share = pp / max(pts, 1.0)
    xgf = stats.get("mp_onIce_xGoalsPercentage")
    try:
        xgf_f = float(xgf) if xgf is not None else None
    except Exception:
        xgf_f = None

    if is_d:
        if pts / gp >= 0.45 or pp_share >= 0.35:
            return "offensive_d"
        # Strong possession with modest scoring → true defensive / two-way D
        if xgf_f is not None and xgf_f >= 0.55 and pts / gp < 0.40:
            return "defensive_d"
        if hpg >= 2.2 and bpg < 1.2:
            return "enforcer_d"
        if bpg >= 1.6 or (hpg + bpg) >= 3.0:
            return "defensive_d"
        return "two_way_d"

    if hpg >= 2.0 and g_share >= 0.35:
        return "power_forward"
    if hpg >= 2.4 and pts / gp < 0.55:
        return "grinder"
    if g_share >= 0.42 and float(stats.get("shootingPct") or 0) >= 0.12:
        return "sniper"
    if a_share >= 0.62 or (a_share >= 0.55 and pp_share >= 0.30):
        return "playmaker"
    return "two_way"


def target_ovr_from_skater_stats(
    stats: Optional[Dict[str, Any]],
    *,
    position_code: str,
    age: int,
    analytics: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """R2 + optional MoneyPuck R3 analytics → target OVR (0–1)."""
    from services.real_nhl_analytics import analytics_impact_score

    pos = str(position_code or "C").upper()
    is_d = pos == "D"
    is_c = pos in ("C",)

    # Allow analytics fields pre-merged onto the NHL stats row.
    if analytics is None and stats:
        if stats.get("mp_onIce_xGoalsPercentage") is not None or stats.get("mp_gameScore") is not None:
            analytics = {
                "games_played": stats.get("mp_games_played", stats.get("gamesPlayed")),
                "icetime": stats.get("mp_icetime"),
                "gameScore": stats.get("mp_gameScore"),
                "onIce_xGoalsPercentage": stats.get("mp_onIce_xGoalsPercentage"),
                "offIce_xGoalsPercentage": stats.get("mp_offIce_xGoalsPercentage"),
                "onIce_corsiPercentage": stats.get("mp_onIce_corsiPercentage"),
                "offIce_corsiPercentage": stats.get("mp_offIce_corsiPercentage"),
            }

    if not stats:
        if age <= 21:
            return 0.66, "no_stats_young"
        if age >= 33:
            return 0.64, "no_stats_veteran"
        return 0.68, "no_stats"

    gp = float(stats.get("gamesPlayed") or 0)
    pts = float(stats.get("points") or 0)
    goals = float(stats.get("goals") or 0)
    assists = float(stats.get("assists") or 0)
    ppg = float(stats.get("pointsPerGame") or 0) or (pts / max(gp, 1.0))
    ev_pts = float(stats.get("evPoints") or 0)
    pp_pts = float(stats.get("ppPoints") or 0)
    toi_min = _toi_minutes_per_game(stats)
    sh_pct = float(stats.get("shootingPct") or 0) or 0.0
    if sh_pct > 1.0:
        sh_pct /= 100.0
    fo_win = float(stats.get("faceoffWinPct") or 0) or 0.0
    if fo_win > 1.0:
        fo_win /= 100.0
    team_toi_pct = float(stats.get("team_toi_rank_pct") or 0.5)
    hits = float(stats.get("hits") or 0)
    blocks = float(stats.get("blockedShots") or 0)
    anal_score, anal_note = analytics_impact_score(analytics)
    has_anal = analytics is not None and float(analytics.get("games_played") or 0) >= 10

    if is_d:
        # D need more points for the same prod score (prevents 80-pt D = McDavid).
        prod = _clamp((ppg - 0.08) / 0.85, 0.0, 1.0)
        usage = _clamp((toi_min - 12.0) / 14.0, 0.0, 1.0)
        # Top-pair / PP QB signal
        role = _clamp(team_toi_pct, 0.0, 1.0) * 0.55 + _clamp(pp_pts / max(gp, 1.0) / 0.40, 0.0, 1.0) * 0.45
        defense_work = _clamp(((hits + blocks) / max(gp, 1.0) - 1.0) / 3.0, 0.0, 1.0)
        finishing = defense_work * 0.12
        # EV production for D matters (not just PP inflation)
        if gp >= 20 and pts > 0:
            ev_share = ev_pts / max(pts, 1.0)
            finishing += _clamp((ev_share - 0.45) / 0.40, -0.08, 0.10)
        two_way = 0.0
    else:
        # Offensive identity: 1.00 PPG is clearly top-line; 1.40+ is superstar.
        prod = _clamp((ppg - 0.10) / 1.20, 0.0, 1.0)
        usage = _clamp((toi_min - 10.0) / 12.0, 0.0, 1.0)
        role = _clamp(team_toi_pct, 0.0, 1.0) * 0.55 + _clamp(pp_pts / max(gp, 1.0) / 0.45, 0.0, 1.0) * 0.45
        finishing = 0.0
        if gp >= 20:
            gpg = goals / max(gp, 1.0)
            finishing += _clamp((gpg - 0.05) / 0.45, 0.0, 1.0) * 0.14
            finishing += _clamp((sh_pct - 0.06) / 0.12, 0.0, 1.0) * 0.04
            apg = assists / max(gp, 1.0)
            finishing += _clamp((apg - 0.15) / 0.70, 0.0, 1.0) * 0.08
        # Two-way / defensive F value (Selke/PK) — real, but must not outrank scorers.
        sh_pts = float(stats.get("shPoints") or 0)
        sh_goals = float(stats.get("shGoals") or 0)
        plus_minus = float(stats.get("plusMinus") or 0)
        two_way = 0.0
        if is_c and gp >= 40 and fo_win > 0:
            # Small FO bump only — was over-promoting Hischier over Stützle.
            two_way += _clamp((fo_win - 0.48) / 0.14, -0.02, 0.035)
        two_way += _clamp(((hits + blocks) / max(gp, 1.0) - 0.8) / 2.8, 0.0, 1.0) * 0.05
        two_way += _clamp((sh_pts + sh_goals * 0.5) / 7.0, 0.0, 1.0) * 0.05
        if gp >= 50:
            two_way += _clamp(plus_minus / 30.0, -0.02, 0.03)

    sample = _clamp(gp / 70.0, 0.0, 1.0)
    # Incomplete seasons cannot mint elite targets from hot PPG alone.
    elite_sample = _clamp(gp / 65.0, 0.0, 1.0)
    if is_d:
        # Usage-first D baseline (Sanderson / #1-pair types live on minutes, not PPG).
        counting = prod * 0.34 + usage * 0.28 + role * 0.24 + finishing
        if has_anal:
            blended = counting * 0.72 + anal_score * 0.28
            # Neutral/weak MoneyPuck must not crush high-minute top-pair D.
            if toi_min >= 19.0 and team_toi_pct >= 0.55:
                blended = max(blended, counting * 0.94)
        else:
            blended = counting
    elif has_anal:
        # Scorers: production leads. Low-event defensive F: analytics + two-way carry more.
        if ppg >= 0.90:
            blended = (
                prod * 0.50
                + usage * 0.16
                + role * 0.12
                + anal_score * 0.08
                + finishing
                + two_way * 0.40
            )
        elif ppg >= 0.55:
            blended = (
                prod * 0.38
                + usage * 0.16
                + role * 0.12
                + anal_score * 0.18
                + finishing
                + two_way
            )
        else:
            blended = (
                prod * 0.22
                + usage * 0.18
                + role * 0.12
                + anal_score * 0.30
                + finishing
                + two_way
            )
    else:
        blended = prod * 0.48 + usage * 0.22 + role * 0.18 + finishing + two_way
    prior = 0.69 if (usage >= 0.45 or team_toi_pct >= 0.45) else 0.66
    skill = _lerp(prior, blended, 0.40 + 0.60 * sample)

    if age <= 20:
        skill -= 0.03
    elif age <= 22 and (is_d or ppg < 0.85):
        # Don't ding producing young stars (Stützle tier) just for age.
        skill -= 0.01
    elif age <= 23 and (is_d or ppg < 0.80):
        skill -= 0.01
    elif 27 <= age <= 30:
        skill += 0.01
    elif age >= 35:
        skill -= 0.04
    elif age >= 32:
        skill -= 0.02

    # Cap PP specialists who don't play real 5v5 minutes
    if gp >= 30 and toi_min > 0 and toi_min < 13.5 and pp_pts >= max(8.0, pts * 0.45):
        skill -= 0.04

    target = _lerp(0.62, 0.945, _clamp(skill, 0.0, 1.0))
    if is_d:
        # Soft ceiling for blueliners; analytics unicorns can clear ~93 without 80 pts.
        if has_anal and anal_score >= 0.80 and team_toi_pct >= 0.55:
            d_cap = 0.925
        elif ppg >= 0.95 and team_toi_pct >= 0.85:
            d_cap = 0.93
        else:
            d_cap = 0.91
        target = min(target, d_cap)
        # Floor lift: elite possession D shouldn't sit in the low 70s on points alone.
        if has_anal and anal_score >= 0.72 and gp >= 40:
            floor = _lerp(0.78, 0.86, _clamp((anal_score - 0.72) / 0.28, 0.0, 1.0))
            if team_toi_pct >= 0.45:
                target = max(target, floor)
        # Usage floors: modern #1 / top-pair D (Sanderson tier) belong in the high 80s.
        if gp >= 55 and toi_min >= 21.5 and team_toi_pct >= 0.70:
            usage_floor = _lerp(
                0.875,
                0.905,
                _clamp((toi_min - 21.5) / 4.0, 0.0, 1.0) * 0.45
                + _clamp((ppg - 0.35) / 0.45, 0.0, 1.0) * 0.55,
            )
            target = max(target, usage_floor)
        elif gp >= 55 and toi_min >= 19.5 and team_toi_pct >= 0.60:
            target = max(target, 0.86)
        elif gp >= 50 and toi_min >= 17.5 and team_toi_pct >= 0.50:
            target = max(target, 0.83)
    else:
        # Scorer floors: Stützle / Batherson / top-line identity.
        if gp >= 50 and toi_min >= 16.0 and ppg >= 1.05:
            target = max(target, _lerp(0.895, 0.925, _clamp((ppg - 1.05) / 0.40, 0.0, 1.0)))
        elif gp >= 50 and toi_min >= 16.0 and ppg >= 0.90:
            target = max(target, _lerp(0.890, 0.918, _clamp((ppg - 0.90) / 0.35, 0.0, 1.0)))
        elif gp >= 50 and toi_min >= 15.0 and ppg >= 0.75:
            target = max(target, _lerp(0.865, 0.890, _clamp((ppg - 0.75) / 0.30, 0.0, 1.0)))
        elif gp >= 55 and team_toi_pct >= 0.60 and toi_min >= 14.5:
            target = max(target, 0.82)

        # Defensive / two-way F floors (Selke types) — lift undervalued checkers.
        tw_signal = two_way + (0.12 if (has_anal and anal_score >= 0.68) else 0.0)
        if gp >= 55 and toi_min >= 16.0 and team_toi_pct >= 0.60 and tw_signal >= 0.08:
            tw_floor = _lerp(0.84, 0.875, _clamp((tw_signal - 0.08) / 0.12, 0.0, 1.0))
            if has_anal and anal_score >= 0.72:
                tw_floor = max(tw_floor, 0.855)
            target = max(target, tw_floor)
        elif gp >= 60 and toi_min >= 14.5 and ppg < 0.55 and (
            tw_signal >= 0.06 or (has_anal and anal_score >= 0.60)
        ):
            # True defensive specialist / 3C shutdown — not a 74 OVR brick.
            lift = anal_score if has_anal else 0.55
            target = max(target, _lerp(0.785, 0.835, _clamp((lift - 0.55) / 0.30, 0.0, 1.0)))

        # Soft caps: FO/xG two-way identity cannot leapfrog higher-PPG scorers.
        if ppg < 0.55:
            target = min(target, 0.845)
        elif ppg < 0.75:
            target = min(target, 0.875)
        elif ppg < 0.90:
            target = min(target, 0.892)
        elif ppg < 0.95:
            # Hischier-band two-way (~0.90–0.94 PPG) stays under ~0.95+ scorers.
            target = min(target, 0.892)
        elif ppg < 1.05:
            target = min(target, 0.918)
        # Mid-sample regression: 40–64 GP hot streaks stay below full-season stars.
        if gp < 65 and ppg >= 0.90:
            target = min(target, _lerp(0.84, target, elite_sample))
        if gp < 55 and ppg >= 1.00:
            target = min(target, _lerp(0.82, 0.90, elite_sample))
    # Regular NHLers with real minutes: don't print sub-74 role players.
    if gp >= 45 and toi_min >= 12.0:
        target = max(target, 0.74)
    if gp >= 55 and team_toi_pct >= 0.50 and toi_min >= 13.5:
        target = max(target, 0.76)
    if gp < 10:
        target = min(target, 0.74)
    note = (
        f"r2_gp={int(gp)}_ppg={ppg:.2f}_toi={toi_min:.1f}"
        f"_role={team_toi_pct:.2f}_pp={int(pp_pts)}"
    )
    if has_anal:
        note = f"r3|{note}|{anal_note}"
    return _clamp(target, 0.58, 0.96), note


def target_ovr_from_goalie_stats(
    stats: Optional[Dict[str, Any]],
    *,
    age: int,
    analytics: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Goalie OVR from NHL counting stats + MoneyPuck GSAx.

    Guided by NHL Network-style tiers: true elites (Hellebuyck/Vasy/Shesterkin)
    in the low-90s; proven starters mid/high-80s; workhorse tandem mid-80s;
    backups stay separated (no 81 pile-up). GSAx captures quality vs team shot
    context so a good goalie on a soft D isn't punished for GAA alone.
    """
    from services.real_nhl_analytics import goalie_analytics_impact_score

    if not stats:
        if age <= 23:
            return 0.66, "no_stats_young_g"
        return 0.69, "no_stats_g"

    if analytics is None and stats:
        if stats.get("mp_gsax") is not None or stats.get("mp_xGoals") is not None:
            analytics = {
                "games_played": stats.get("mp_games_played", stats.get("gamesPlayed")),
                "gsax": stats.get("mp_gsax"),
                "xGoals": stats.get("mp_xGoals"),
                "goals": stats.get("mp_goals"),
                "hd_gsax": stats.get("mp_hd_gsax"),
                "highDangerxGoals": stats.get("mp_highDangerxGoals"),
                "highDangerGoals": stats.get("mp_highDangerGoals"),
            }

    gp = float(stats.get("gamesPlayed") or 0)
    gs = float(stats.get("gamesStarted") or gp)
    sv = float(stats.get("savePct") or 0)
    if sv > 1.5:
        sv /= 1000.0 if sv > 10 else 100.0
    gaa = float(stats.get("goalsAgainstAverage") or 3.0)
    wins = float(stats.get("wins") or 0)
    losses = float(stats.get("losses") or 0)
    otl = float(stats.get("otLosses") or 0)
    sa = float(stats.get("shotsAgainst") or 0)
    shutouts = float(stats.get("shutouts") or 0)
    start_pct = float(stats.get("team_start_rank_pct") or (0.85 if gs >= 40 else 0.4))
    anal_score, anal_note = goalie_analytics_impact_score(analytics)
    has_anal = analytics is not None and float(analytics.get("games_played") or 0) >= 10
    gsax_total = 0.0
    if analytics is not None:
        if analytics.get("gsax") is not None:
            gsax_total = float(analytics.get("gsax") or 0)
        else:
            gsax_total = float(analytics.get("xGoals") or 0) - float(analytics.get("goals") or 0)

    sv_score = _clamp((sv - 0.885) / 0.045, 0.0, 1.0)
    # GAA is team-influenced — keep it light vs SV%/GSAx.
    gaa_score = _clamp((3.35 - gaa) / 1.45, 0.0, 1.0)
    workload = _clamp(gs / 58.0, 0.0, 1.0)
    volume = _clamp((sa / max(gp, 1.0) - 22.0) / 16.0, 0.0, 1.0) if gp else 0.0
    decisions = max(wins + losses + otl, gs, 1.0)
    win_pct = wins / decisions
    win_pct_score = _clamp((win_pct - 0.32) / 0.42, 0.0, 1.0)
    win_volume = _clamp((wins - 12.0) / 32.0, 0.0, 1.0)
    so_score = _clamp(shutouts / 7.0, 0.0, 1.0)
    starter = _clamp(start_pct, 0.0, 1.0)

    sample = _clamp(gp / 45.0, 0.0, 1.0)
    if has_anal:
        blended = (
            anal_score * 0.40
            + sv_score * 0.20
            + workload * 0.12
            + win_pct_score * 0.07
            + win_volume * 0.07
            + gaa_score * 0.05
            + volume * 0.05
            + so_score * 0.04
        )
    else:
        blended = (
            sv_score * 0.40
            + gaa_score * 0.14
            + workload * 0.16
            + starter * 0.08
            + win_pct_score * 0.10
            + win_volume * 0.07
            + volume * 0.05
        )
    skill = _lerp(0.70, blended, 0.40 + 0.60 * sample)

    if age <= 22:
        skill -= 0.02
    elif age >= 36:
        skill -= 0.035
    elif age >= 34:
        skill -= 0.015

    # Ceiling: Vezina/Hart tier can touch ~93; stay under skater superstars.
    if has_anal and anal_score >= 0.88 and gs >= 50 and sv >= 0.918:
        hi = 0.938
    elif has_anal and anal_score >= 0.78 and gs >= 45:
        hi = 0.928
    elif sv >= 0.920 and gs >= 45:
        hi = 0.922
    else:
        hi = 0.910
    target = _lerp(0.67, hi, _clamp(skill, 0.0, 1.0))

    gsax_pg = gsax_total / max(gp, 1.0) if gp else 0.0
    # Graduated floors — GSAx-first for team-context quality; spread the mid tier.
    if gs >= 55 and ((has_anal and anal_score >= 0.85) or (sv >= 0.920 and wins >= 38)):
        target = max(target, 0.905)
    elif has_anal and gp >= 50 and gsax_pg >= 0.45:
        target = max(target, 0.90)
    elif has_anal and gp >= 40 and gsax_total >= 20.0:
        target = max(target, 0.875)
    elif has_anal and gp >= 45 and gsax_total >= 12.0:
        target = max(target, 0.86)
    elif gs >= 48 and ((has_anal and anal_score >= 0.70) or sv >= 0.912):
        target = max(target, 0.85)
    elif gs >= 40 and ((has_anal and gsax_total >= 8.0) or sv >= 0.908):
        target = max(target, 0.835)
    elif gs >= 30 and sv >= 0.900:
        target = max(target, 0.80)
    elif gs >= 20:
        target = max(target, 0.76)

    # Small-sample high SV%/GSAx (Stolarz-type) can't leapfrog workhorse elites.
    if gp < 40:
        target = min(target, 0.875)
    if gp < 30:
        target = min(target, 0.855)
    # Negative GSAx workhorse — don't pin as elite on wins alone, but keep starter band.
    if has_anal and gs >= 45 and gsax_total <= -5.0:
        target = min(target, 0.85)
        target = max(target, 0.80)
    if gp < 8:
        target = min(target, 0.72)

    note = (
        f"r2_g_gp={int(gp)}_gs={int(gs)}_sv={sv:.3f}_w={int(wins)}-"
        f"{int(losses)}-{int(otl)}_start={start_pct:.2f}"
    )
    if has_anal:
        note = f"r3g|{note}|{anal_note}"
    return _clamp(target, 0.58, 0.94), note



def pick_stats_row(
    player_id: int,
    *,
    is_goalie: bool,
    primary: Dict[int, Dict[str, Any]],
    secondary: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Prefer a meaningful sample; avoid rating stars from injury-shortened seasons."""
    a = primary.get(player_id)
    b = secondary.get(player_id)
    if a and b:
        gpa = float(a.get("gamesPlayed") or 0)
        gpb = float(b.get("gamesPlayed") or 0)
        # If the preferred season is injury-shortened and the other is a near-full
        # season, use the fuller sample (Heiskanen / Monahan-class cases).
        if gpa < 55 and gpb >= max(55.0, gpa + 10.0):
            return b
        if gpb < 55 and gpa >= max(55.0, gpb + 10.0):
            return a
        if gpa >= 20 or (gpa >= gpb and gpa >= 5):
            return a
        if gpb > gpa:
            return b
        return a or b
    return a or b


# ---------------------------------------------------------------------------
# R4 overrides + landing (draft / body) enrichment
# ---------------------------------------------------------------------------


def load_r4_overrides(path: Optional[Path] = None) -> Dict[int, Dict[str, Any]]:
    p = path or R4_OVERRIDES_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in (raw or {}).items():
        if str(k).startswith("_") or not isinstance(v, dict):
            continue
        try:
            out[int(k)] = v
        except Exception:
            continue
    return out


def apply_r4_target(
    *,
    nhl_id: int,
    target_ovr: float,
    profile: str,
    overrides: Dict[int, Dict[str, Any]],
) -> Tuple[float, str, Optional[Dict[str, Any]]]:
    """Pin or floor target OVR / profile from the R4 pack."""
    ov = overrides.get(int(nhl_id)) or {}
    if not ov:
        return float(target_ovr), profile, None
    pinned = ov.get("target_ovr")
    floored = ov.get("min_ovr")
    out = float(target_ovr)
    if pinned is not None:
        out = float(pinned)
    elif floored is not None:
        out = max(out, float(floored))
    forced = str(ov.get("profile") or profile or "").strip() or profile
    return _clamp(out, 0.58, 0.99), forced, ov


def _fetch_player_landing(nhl_id: int) -> Optional[Dict[str, Any]]:
    try:
        return _http_get_json(f"{WEB_API}/player/{int(nhl_id)}/landing", timeout=20)
    except Exception:
        return None


def fetch_landings_by_id(
    player_ids: List[int],
    *,
    max_workers: int = 16,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    # Preserve order while de-duplicating merged/current/prior roster IDs.
    ids = list(dict.fromkeys(int(i) for i in player_ids if i))
    if not ids:
        return out

    for pid in ids:
        cached = _PLAYER_LANDING_CACHE.get(pid)
        if cached:
            out[pid] = cached
    missing_ids = [pid for pid in ids if pid not in out]
    if not missing_ids:
        return out

    def _one(pid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        return pid, _fetch_player_landing(pid)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, pid) for pid in missing_ids]
        for fut in as_completed(futs):
            try:
                pid, payload = fut.result()
            except Exception:
                continue
            if payload:
                out[pid] = payload
                _PLAYER_LANDING_CACHE[pid] = payload
    return out


def _team_id_for_abbr(teams: List[Any], abbr: str) -> Optional[str]:
    want = str(abbr or "").upper()
    if not want:
        return None
    for tm in teams or []:
        a = str(
            getattr(tm, "abbreviation", None)
            or getattr(tm, "abbr", None)
            or ""
        ).upper()
        if a == want:
            tid = getattr(tm, "team_id", None) or getattr(tm, "id", None)
            return str(tid) if tid is not None else None
    return None


def _apply_draft_and_body(
    player: Any,
    *,
    roster_row: Dict[str, Any],
    landing: Optional[Dict[str, Any]],
    teams: List[Any],
) -> None:
    ident = getattr(player, "identity", None)
    # Height / weight — prefer landing, else roster.
    h_in = None
    w_lb = None
    src = landing or {}
    if src.get("heightInInches"):
        h_in = float(src["heightInInches"])
    elif roster_row.get("heightInInches"):
        h_in = float(roster_row["heightInInches"])
    if src.get("weightInPounds"):
        w_lb = float(src["weightInPounds"])
    elif roster_row.get("weightInPounds"):
        w_lb = float(roster_row["weightInPounds"])
    if h_in and ident is not None:
        try:
            ident.height_cm = int(round(h_in * 2.54))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    if w_lb and ident is not None:
        try:
            ident.weight_kg = float(round(w_lb * 0.453592, 1))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    if h_in:
        setattr(player, "height_in", int(round(h_in)))
    if w_lb:
        setattr(player, "weight_lb", int(round(w_lb)))

    draft = (landing or {}).get("draftDetails") if isinstance(landing, dict) else None
    if not isinstance(draft, dict) or not draft.get("year"):
        setattr(player, "undrafted", True)
        setattr(player, "drafted", False)
        return

    try:
        dy = int(draft.get("year"))
    except Exception:
        setattr(player, "undrafted", True)
        setattr(player, "drafted", False)
        return
    try:
        dr = int(draft.get("round") or 0) or None
    except Exception:
        dr = None
    try:
        overall = int(draft.get("overallPick") or 0) or None
    except Exception:
        overall = None
    try:
        pick_in_round = int(draft.get("pickInRound") or 0) or None
    except Exception:
        pick_in_round = None
    draft_abbr = str(draft.get("teamAbbrev") or "").upper() or None

    if ident is not None:
        try:
            ident.draft_year = dy
            if dr is not None:
                ident.draft_round = int(dr)
            # IdentityBio.draft_pick is treated as overall by the roster serializer.
            if overall is not None:
                ident.draft_pick = int(overall)
            elif pick_in_round is not None:
                ident.draft_pick = int(pick_in_round)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    setattr(player, "drafted", True)
    setattr(player, "undrafted", False)
    setattr(player, "draft_year", dy)
    if dr is not None:
        setattr(player, "draft_round", int(dr))
    if overall is not None:
        setattr(player, "draft_overall_pick", int(overall))
    if pick_in_round is not None:
        setattr(player, "draft_pick_in_round", int(pick_in_round))
    if draft_abbr:
        setattr(player, "draft_team_abbr", draft_abbr)
        tid = _team_id_for_abbr(teams, draft_abbr)
        if tid:
            setattr(player, "draft_team_id", tid)
            setattr(player, "drafted_by", tid)
            setattr(player, "drafted_by_team_id", tid)


def _localized_field(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("default") or next(iter(value.values()), "") or "")
    return str(value or "")


def _season_int_to_label(season_int: Any) -> str:
    s = str(season_int or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[6:8]}"
    return s


def _attach_career_stats(player: Any, *, landing: Optional[Dict[str, Any]], is_goalie: bool) -> None:
    """Attach player.career_stats = {"seasons": [...]} from the NHL landing
    endpoint's seasonTotals (falls back to yearByYear naming used by older
    payload shapes). Only prior NHL regular-season rows are kept; the current
    in-progress franchise season is tracked separately via session sim stats.
    """
    if not isinstance(landing, dict):
        return
    raw_seasons = (
        landing.get("seasonTotals")
        or landing.get("yearByYearTotals")
        or landing.get("yearByYear")
    )
    if not isinstance(raw_seasons, list) or not raw_seasons:
        return

    def _int(entry: Dict[str, Any], *keys: str) -> int:
        for key in keys:
            val = entry.get(key)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    continue
        return 0

    def _flt(entry: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            val = entry.get(key)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    continue
        return None

    seasons_out: List[Dict[str, Any]] = []
    for entry in raw_seasons:
        if not isinstance(entry, dict):
            continue
        league = str(entry.get("leagueAbbrev") or entry.get("league") or "").upper()
        if league and league != "NHL":
            continue
        game_type = entry.get("gameTypeId")
        try:
            if game_type is not None and int(game_type) != 2:
                continue
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        gp = _int(entry, "gamesPlayed", "gp")
        if gp <= 0:
            continue
        team_name = _localized_field(entry.get("teamName")) or _localized_field(entry.get("teamCommonName"))
        team_abbrev = _localized_field(entry.get("teamAbbrevs") or entry.get("teamAbbrev"))
        row: Dict[str, Any] = {
            "season": _season_int_to_label(entry.get("season")),
            "season_key": entry.get("season"),
            "team": team_name or team_abbrev or "—",
            "team_abbrev": team_abbrev or None,
            "league": league or "NHL",
            "gp": gp,
        }
        if is_goalie:
            row.update(
                {
                    "wins": _int(entry, "wins"),
                    "losses": _int(entry, "losses"),
                    "otl": _int(entry, "otLosses", "otl"),
                    "sv_pct": _flt(entry, "savePctg", "sv_pct"),
                    "gaa": _flt(entry, "goalsAgainstAvg", "gaa"),
                    "shutouts": _int(entry, "shutouts"),
                }
            )
        else:
            goals = _int(entry, "goals", "g")
            assists = _int(entry, "assists", "a")
            row.update(
                {
                    "g": goals,
                    "a": assists,
                    "pts": _int(entry, "points", "pts") or (goals + assists),
                    "plus_minus": _int(entry, "plusMinus", "plus_minus"),
                    "pim": _int(entry, "penaltyMinutes", "pim"),
                }
            )
        seasons_out.append(row)

    if not seasons_out:
        return
    # NHL API returns most-recent-first; store chronologically (oldest → newest).
    seasons_out.sort(key=lambda r: str(r.get("season_key") or r.get("season") or ""))
    for row in seasons_out:
        row.pop("season_key", None)

    existing = getattr(player, "career_stats", None)
    if not isinstance(existing, dict):
        existing = {}
    else:
        existing = dict(existing)
    existing["seasons"] = seasons_out
    setattr(player, "career_stats", existing)


def _ensure_estimated_contract(player: Any, *, season_year: int, rng: Any = None) -> None:
    """Rostered NHL players with no matched Spotrac deal still have a contract in real life.

    Without one they were dumped into free agency at the first summer. Give them an
    estimated multi-year deal (market AAV, age-based term) so only genuinely expiring
    players hit the market.
    """
    try:
        from services.contract_economy import (
            apply_contract_to_player,
            compute_market_value,
            has_active_contract,
        )

        if has_active_contract(player):
            return
        age = int(getattr(player, "age", 0) or getattr(getattr(player, "identity", None), "age", 0) or 27)
        r = (rng.random() if rng is not None else 0.5)
        if age <= 21:
            years, rights = 2 + int(r * 2), "RFA"
        elif age <= 25:
            years, rights = 1 + int(r * 3), "RFA"
        elif age <= 30:
            years, rights = 2 + int(r * 4), "UFA"
        elif age <= 34:
            years, rights = 1 + int(r * 3), "UFA"
        else:
            years, rights = 1 + int(r * 2), "UFA"
        try:
            aav = float(compute_market_value(player) or 0.0)
        except Exception:
            aav = 0.0
        aav = round(max(0.775, min(13.0, aav or 0.775)), 3)
        apply_contract_to_player(
            player,
            {
                "aav_m": aav,
                "cap_hit_m": aav,
                "years": years,
                "years_remaining": years,
                "expiry_year": int(season_year) + years,
                "contract_type": "STANDARD",
                "rights_status": rights,
                "source": "estimated",
                "is_nhl_spc": True,
            },
            int(season_year),
        )
        setattr(player, "contract_source", "estimated")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _clamp_estimated_contract_to_minor_level(player: Any, *, season_year: int) -> bool:
    """An unmatched farmhand gets a two-way deal near the league minimum.

    The market-value estimate is fine for an NHL regular we failed to match, but for
    an AHL goalie it invented $3M+ deals (Merilainen) that were then charged to the
    parent club as buried cap.
    """
    c = getattr(player, "contract", None)
    src = str((c.get("source") if isinstance(c, dict) else getattr(c, "source", "")) or "")
    if src != "estimated" and str(getattr(player, "contract_source", "") or "") != "estimated":
        return False
    try:
        from app.sim_engine.economy.cap_engine import nhl_minimum_salary_millions
        from services.contract_economy import apply_contract_to_player

        floor = float(nhl_minimum_salary_millions(int(season_year)))
        cur = float((c.get("aav_m") if isinstance(c, dict) else getattr(c, "aav_m", 0)) or 0.0)
        aav = round(min(max(cur, floor), floor + 0.15), 3)
        yrs = int((c.get("years_remaining") if isinstance(c, dict) else getattr(c, "years_remaining", 1)) or 1)
        yrs = max(1, min(yrs, 2))
        rights = str((c.get("rights_status") if isinstance(c, dict) else getattr(c, "rights_status", "")) or "UFA")
        apply_contract_to_player(
            player,
            {
                "aav_m": aav,
                "cap_hit_m": aav,
                "years": yrs,
                "years_remaining": yrs,
                "expiry_year": int(season_year) + yrs,
                "contract_type": "STANDARD",
                "two_way": True,
                "minor_salary_m": 0.1,
                "rights_status": rights,
                "source": "estimated",
                "is_nhl_spc": True,
            },
            int(season_year),
        )
        return True
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
        return False


def _apply_real_contract(
    player: Any,
    *,
    contract: Dict[str, Any],
    season_year: int,
    override: Optional[Dict[str, Any]] = None,
) -> None:
    from services.contract_economy import apply_contract_to_player

    payload = dict(contract or {})
    if isinstance(override, dict) and isinstance(override.get("contract"), dict):
        payload.update(override["contract"])
        payload["source"] = str(payload.get("source") or "real_nhl_r4")
    yrs = int(payload.get("years_remaining") or payload.get("years") or 0)
    if yrs <= 0 or float(payload.get("aav_m") or payload.get("cap_hit_m") or 0) <= 0:
        return
    from services.contract_economy import _sync_term_from_season_cap_hits

    payload["effective_season"] = int(season_year)
    _sync_term_from_season_cap_hits(payload, int(season_year))
    yrs = int(payload.get("years_remaining") or payload.get("years") or yrs)
    if not payload.get("expiry_year"):
        payload["expiry_year"] = int(season_year) + yrs
    ext = payload.get("pending_extension")
    if isinstance(ext, dict) and float(ext.get("aav_m") or 0) > 0:
        ext = dict(ext)
        ext["expiry_year"] = int(payload.get("expiry_year") or (int(season_year) + yrs)) + int(ext.get("years") or 0)
        ext["rights_status"] = "UFA"
        payload["pending_extension"] = ext
    payload.setdefault("contract_type", "STANDARD")
    payload.setdefault("rights_status", "UFA")
    payload.setdefault("source", "real_nhl_spotrac")
    payload["is_nhl_spc"] = True
    apply_contract_to_player(player, payload, int(season_year))
    setattr(player, "real_nhl_contract", True)
    setattr(player, "contract_source", payload.get("source"))
    try:
        player.cap_hit_m = float(payload.get("cap_hit_m") or payload.get("aav_m") or 0)
        player.aav_m = float(payload.get("aav_m") or payload.get("cap_hit_m") or 0)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


# ---------------------------------------------------------------------------
# Player construction
# ---------------------------------------------------------------------------


def _position_from_code(code: str) -> Any:
    from app.sim_engine.entities.player import Position

    c = str(code or "C").upper()
    if c in ("L", "LW"):
        return Position.LW
    if c in ("R", "RW"):
        return Position.RW
    if c == "D":
        return Position.D
    if c == "G":
        return Position.G
    return Position.C


def _age_from_birth(birth_date: str, as_of: date) -> int:
    try:
        y, m, d = [int(x) for x in str(birth_date)[:10].split("-")]
        years = as_of.year - y
        if (as_of.month, as_of.day) < (m, d):
            years -= 1
        return max(17, min(45, years))
    except Exception:
        return 25


def _country_label(code: str) -> str:
    mapping = {
        "CAN": "Canada",
        "USA": "USA",
        "SWE": "Sweden",
        "FIN": "Finland",
        "RUS": "Russia",
        "CZE": "Czechia",
        "SVK": "Slovakia",
        "CHE": "Switzerland",
        "SUI": "Switzerland",
        "DEU": "Germany",
        "GER": "Germany",
        "DNK": "Denmark",
        "DEN": "Denmark",
        "NOR": "Norway",
        "LVA": "Latvia",
        "LAT": "Latvia",
        "BLR": "Belarus",
        "AUT": "Austria",
        "SVN": "Slovenia",
        "GBR": "United Kingdom",
        "CHN": "China",
        "JPN": "Japan",
        "KOR": "South Korea",
        "IND": "India",
        "NGA": "Nigeria",
        "KEN": "Kenya",
        "PHL": "Philippines",
        "MEX": "Mexico",
        "BRA": "Brazil",
        "ARG": "Argentina",
        "ZAF": "South Africa",
        "RSA": "South Africa",
        "GHA": "Ghana",
        "JAM": "Jamaica",
        "VNM": "Vietnam",
        "IDN": "Indonesia",
        "PAK": "Pakistan",
        "EGY": "Egypt",
        "MAR": "Morocco",
        "COL": "Colombia",
        "PER": "Peru",
        "CHL": "Chile",
        "THA": "Thailand",
        "TWN": "Taiwan",
        "HKG": "Hong Kong",
    }
    c = str(code or "").upper()
    return mapping.get(c, c or "Canada")


def align_attribute_ovr_to_target(player: Any, target_ovr: float, *, rounds: int = 40) -> float:
    """
    Same rule as generated players: displayed OVR is compute_ovr(ratings…),
    not a stamped overall. Nudge attribute card until weighted OVR matches target.
    """
    from app.sim_engine.engine import _nudge_player_ovr_toward
    from app.sim_engine.entities.player import persist_recomputed_ovr

    tgt = _clamp(float(target_ovr), 0.58, 0.99)
    final = persist_recomputed_ovr(player)
    for _ in range(max(1, int(rounds))):
        try:
            cur = float(player.ovr()) if callable(getattr(player, "ovr", None)) else float(final)
        except Exception:
            cur = float(final)
        gap = abs(cur - tgt)
        if gap <= 0.0035:
            break
        # Large gaps need stronger nudges — the engine clamps each scale step.
        strength = 1.55 if gap >= 0.04 else 1.15 if gap >= 0.015 else 0.92
        _nudge_player_ovr_toward(player, tgt, strength=strength)
        final = persist_recomputed_ovr(player)
    return float(final)


def _merge_analytics_into_stats(
    stats: Optional[Dict[str, Any]],
    analytics: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not stats and not analytics:
        return stats
    out = dict(stats or {})
    if not analytics:
        return out
    out["mp_games_played"] = analytics.get("games_played")
    out["mp_icetime"] = analytics.get("icetime")
    out["mp_gameScore"] = analytics.get("gameScore")
    out["mp_onIce_xGoalsPercentage"] = analytics.get("onIce_xGoalsPercentage")
    out["mp_offIce_xGoalsPercentage"] = analytics.get("offIce_xGoalsPercentage")
    out["mp_onIce_corsiPercentage"] = analytics.get("onIce_corsiPercentage")
    out["mp_offIce_corsiPercentage"] = analytics.get("offIce_corsiPercentage")
    # Goalie MoneyPuck fields
    if analytics.get("gsax") is not None or analytics.get("kind") == "goalie":
        out["mp_gsax"] = analytics.get("gsax")
        out["mp_xGoals"] = analytics.get("xGoals")
        out["mp_goals"] = analytics.get("goals")
        out["mp_hd_gsax"] = analytics.get("hd_gsax")
        out["mp_highDangerxGoals"] = analytics.get("highDangerxGoals")
        out["mp_highDangerGoals"] = analytics.get("highDangerGoals")
    return out


def _build_player_from_roster_row(
    row: Dict[str, Any],
    *,
    team: Any,
    league: Any,
    rng: random.Random,
    season_year: int,
    as_of: date,
    skater_stats_a: Dict[int, Dict[str, Any]],
    skater_stats_b: Dict[int, Dict[str, Any]],
    goalie_stats_a: Dict[int, Dict[str, Any]],
    goalie_stats_b: Dict[int, Dict[str, Any]],
    r4_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
    landing: Optional[Dict[str, Any]] = None,
    contract: Optional[Dict[str, Any]] = None,
    all_teams: Optional[List[Any]] = None,
    analytics_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    goalie_analytics_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    dynasty_registry: Optional[Any] = None,
    align_rounds: int = 40,
    skip_ledger_finalize: bool = False,
) -> Any:
    from app.sim_engine.engine import (
        build_role_shaped_ratings,
        finalize_created_player_for_game_ledger,
        pop_generation_profile,
    )
    from app.sim_engine.entities.player import (
        BackstoryType,
        BackstoryUpbringing,
        DevResources,
        IdentityBio,
        Player,
        PressureLevel,
        Shoots,
        SupportLevel,
        UpbringingType,
        archetype_from_generation_profile,
        assign_skater_archetype,
        persist_recomputed_ovr,
    )

    nhl_id = int(row.get("id") or 0)
    first = _localized_name(row.get("firstName"))
    last = _localized_name(row.get("lastName"))
    full_name = f"{first} {last}".strip() or f"NHL Player {nhl_id}"
    pos_code = str(row.get("positionCode") or "C").upper()
    pos = _position_from_code(pos_code)
    is_goalie = pos_code == "G"

    birth = str(row.get("birthDate") or "")[:10]
    if not birth and isinstance(landing, dict):
        birth = str(landing.get("birthDate") or "")[:10]
    age = _age_from_birth(birth, as_of)
    birth_year = as_of.year - age
    try:
        if birth:
            birth_year = int(birth.split("-")[0])
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    dynasty_entry = None
    r4 = None
    team_abbr = str(
        getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or ""
    ).upper()
    if dynasty_registry is not None:
        try:
            dynasty_entry = dynasty_registry.match_player(
                team_abbr, "nhl", full_name, last_name=last
            )
        except Exception:
            dynasty_entry = None

    if dynasty_entry is None:
        if is_goalie:
            stats = pick_stats_row(
                nhl_id, is_goalie=True, primary=goalie_stats_a, secondary=goalie_stats_b
            )
            g_anal = (goalie_analytics_by_id or {}).get(nhl_id)
            stats = _merge_analytics_into_stats(stats, g_anal)
            target_ovr, rating_note = target_ovr_from_goalie_stats(
                stats, age=age, analytics=g_anal
            )
            forced_profile = "balanced_g"
            if stats:
                gs = float(stats.get("gamesStarted") or 0)
                sa_pg = float(stats.get("shotsAgainst") or 0) / max(float(stats.get("gamesPlayed") or 1), 1.0)
                if sa_pg >= 32:
                    forced_profile = "hybrid_g"
                elif float(stats.get("savePct") or 0) >= 0.915 and gs >= 30:
                    forced_profile = "butterfly_g"
        else:
            stats = pick_stats_row(
                nhl_id, is_goalie=False, primary=skater_stats_a, secondary=skater_stats_b
            )
            anal = (analytics_by_id or {}).get(nhl_id)
            stats = _merge_analytics_into_stats(stats, anal)
            target_ovr, rating_note = target_ovr_from_skater_stats(
                stats, position_code=pos_code, age=age, analytics=anal
            )
            forced_profile = infer_skater_profile(stats, position_code=pos_code)

        r4 = None
        target_ovr, forced_profile, r4 = apply_r4_target(
            nhl_id=nhl_id,
            target_ovr=target_ovr,
            profile=forced_profile,
            overrides=r4_overrides or {},
        )
        if r4:
            rating_note = f"r4|{rating_note}"
    else:
        target_ovr = float(dynasty_entry.chapters.get("overall", 75)) / 99.0
        rating_note = "dynasty_txt"
        forced_profile = "balanced_g" if is_goalie else "two_way"

    ratings = build_role_shaped_ratings(
        position=pos,
        target_ovr=target_ovr,
        rng=rng,
        profile=forced_profile,
    )
    gen_profile = pop_generation_profile(ratings)
    synced_arch = archetype_from_generation_profile(gen_profile, pos)
    if not synced_arch:
        synced_arch = assign_skater_archetype(pos, rng)

    shoots_raw = str(
        row.get("shootsCatches")
        or (landing or {}).get("shootsCatches")
        or "L"
    ).upper()[:1]
    shoots = Shoots.R if shoots_raw == "R" else Shoots.L

    h_cm = int(row.get("heightInCentimeters") or 0) or int(
        round(float(row.get("heightInInches") or 72) * 2.54)
    )
    w_kg = float(row.get("weightInKilograms") or 0) or (
        float(row.get("weightInPounds") or 200) * 0.453592
    )
    if isinstance(landing, dict):
        if landing.get("heightInCentimeters"):
            h_cm = int(landing["heightInCentimeters"])
        elif landing.get("heightInInches"):
            h_cm = int(round(float(landing["heightInInches"]) * 2.54))
        if landing.get("weightInKilograms"):
            w_kg = float(landing["weightInKilograms"])
        elif landing.get("weightInPounds"):
            w_kg = float(landing["weightInPounds"]) * 0.453592

    # Draft placeholders — replaced by landing enrichment below.
    identity = IdentityBio(
        name=full_name,
        age=age,
        birth_year=birth_year,
        birth_country=_country_label(
            str(row.get("birthCountry") or (landing or {}).get("birthCountry") or "")
        ),
        birth_city=_localized_name(
            row.get("birthCity") or (landing or {}).get("birthCity")
        )
        or "Unknown",
        height_cm=int(h_cm),
        weight_kg=float(w_kg),
        position=pos,
        shoots=shoots,
        draft_year=0,
        draft_round=0,
        draft_pick=0,
    )
    backstory = BackstoryUpbringing(
        backstory=BackstoryType.PRODIGY if target_ovr >= 0.88 else BackstoryType.GRINDER,
        upbringing=UpbringingType.STABLE_MIDDLE_CLASS,
        family_support=SupportLevel.MEDIUM,
        early_pressure=PressureLevel.MODERATE,
        dev_resources=DevResources.LOCAL,
    )
    player = Player(
        identity=identity,
        backstory=backstory,
        ratings=ratings,
        rng_seed=int(nhl_id) or rng.randint(1, 2_000_000_000),
        archetype=synced_arch,
        pool_context="nhl",
    )
    if gen_profile:
        try:
            setattr(player, "_generated_profile", gen_profile)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    try:
        persist_recomputed_ovr(player)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    if dynasty_entry is not None:
        try:
            from services.dynasty_ratings_parser import apply_dynasty_entry_to_player

            apply_dynasty_entry_to_player(
                player, dynasty_entry, seed=nhl_id, align_rounds=align_rounds
            )
            target_ovr = float(player.ovr()) if callable(getattr(player, "ovr", None)) else target_ovr
            rating_note = str(getattr(player, "real_nhl_rating_note", rating_note) or rating_note)
        except Exception:
            try:
                align_attribute_ovr_to_target(player, float(target_ovr), rounds=align_rounds)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
    else:
        try:
            align_attribute_ovr_to_target(player, float(target_ovr), rounds=align_rounds)
        except Exception:
            try:
                persist_recomputed_ovr(player)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)

    setattr(player, "nhl_player_id", nhl_id)
    setattr(player, "external_player_id", str(nhl_id))
    setattr(player, "real_nhl_import", True)
    setattr(player, "real_nhl_rating_note", rating_note)
    setattr(player, "real_nhl_target_ovr", round(float(target_ovr), 4))
    if birth:
        try:
            setattr(player, "birth_date", birth)
            parts = [int(x) for x in birth.split("-")[:3]]
            if len(parts) == 3:
                setattr(identity, "birth_month", parts[1])
                setattr(identity, "birth_day", parts[2])
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    if dynasty_entry is None and r4:
        setattr(player, "real_nhl_r4", True)
    from app.sim_engine.generation.player_headshots import valid_nhl_headshot_url

    # The landing endpoint is the canonical source. Roster data remains a
    # compatibility fallback for seasons where NHL also includes the field.
    headshot = valid_nhl_headshot_url(
        (landing or {}).get("headshot") or row.get("headshot")
    )
    if headshot:
        setattr(player, "nhl_headshot_url", headshot)
        setattr(player, "portrait_url", headshot)
        setattr(player, "portrait_source", "nhl")
    sweater = row.get("sweaterNumber")
    if sweater is None and isinstance(landing, dict):
        sweater = landing.get("sweaterNumber")
    if sweater is not None:
        try:
            setattr(player, "sweater_number", int(sweater))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    _apply_draft_and_body(
        player,
        roster_row=row,
        landing=landing,
        teams=all_teams or [],
    )
    if not skip_ledger_finalize:
        try:
            from services.player_bio_parser import apply_player_bio_by_name, load_player_bio_registry

            bio_registry = load_player_bio_registry(as_of=date(int(season_year), 9, 15))
            apply_player_bio_by_name(player, bio_registry, as_of_year=season_year)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    if landing:
        _attach_career_stats(player, landing=landing, is_goalie=is_goalie)

    if contract or (r4 and isinstance(r4.get("contract"), dict)):
        _apply_real_contract(
            player,
            contract=contract or {},
            season_year=season_year,
            override=r4,
        )

    if not skip_ledger_finalize:
        finalize_created_player_for_game_ledger(
            player,
            league=league,
            team=team,
            rng=rng,
            source="real_nhl_import",
            season_year=int(season_year),
        )
    _ensure_estimated_contract(player, season_year=int(season_year), rng=rng)

    # Prefer stable NHL id when ledger assigned a random one.
    try:
        if nhl_id:
            setattr(player, "id", f"NHL_{nhl_id}")
            setattr(player, "_ledger_player_id", f"NHL_{nhl_id}")
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    # Persist rating-season boxcars for trade production / audit only.
    # Do NOT write prior-year counting stats into live player.season_stats —
    # that container is reserved for the simulated season ledger sync and a
    # flat import line previously stacked under current-season GP/points.
    try:
        if stats:
            setattr(player, "real_nhl_import_stats", dict(stats))
            # Keep season_stats empty (or year-keyed only after sim sync).
            if not isinstance(getattr(player, "season_stats", None), dict):
                setattr(player, "season_stats", {})
            elif "gp" in (getattr(player, "season_stats", None) or {}):
                # Clear accidental flat prior-year seed.
                setattr(player, "season_stats", {})
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    # Re-align after ledger init — potential / character hooks must not leave
    # stored OVR far below the stats-derived target.
    try:
        align_attribute_ovr_to_target(player, float(target_ovr), rounds=40)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    return player


def _fetch_team_roster(abbr: str, season_key: Any) -> Dict[str, Any]:
    token = "current" if str(season_key).lower() == "current" else str(int(season_key))
    url = f"{WEB_API}/roster/{abbr}/{token}"
    return _http_get_json(url)


def _roster_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in ("forwards", "defensemen", "goalies"):
        for p in payload.get(key) or []:
            if isinstance(p, dict):
                rows.append(p)
    return rows


def _roster_player_id(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("id") or 0)
    except Exception:
        return 0


def _merge_roster_rows(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union roster rows by NHL id. Primary (current season) wins on conflict.

    NHL.com mid-season / early-season roster endpoints often omit LTIR, IR, and
    some veterans. Supplementing from the prior completed season prevents stars
    from silently vanishing from the imported league.
    """
    by_id: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    for row in list(secondary or []) + list(primary or []):
        pid = _roster_player_id(row)
        if pid <= 0:
            continue
        if pid not in by_id:
            order.append(pid)
        by_id[pid] = row
    # Prefer primary ordering, then any secondary-only players.
    primary_ids = [_roster_player_id(r) for r in (primary or []) if _roster_player_id(r) > 0]
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for pid in primary_ids + order:
        if pid in seen or pid not in by_id:
            continue
        seen.add(pid)
        out.append(by_id[pid])
    return out


def _fetch_merged_team_roster(abbr: str, roster_season: int, prior_season: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Prefer live /current when importing this year's opening roster (July UFA signings)."""
    note = "season"
    current_payload: Dict[str, Any] = {}
    current_rows: List[Dict[str, Any]] = []
    try:
        from services.nhl_season_calendar import current_nhl_season_start_year

        live_season = _season_id(current_nhl_season_start_year())
    except Exception:
        live_season = 0
    if live_season and int(roster_season) == int(live_season):
        try:
            current_payload = _fetch_team_roster(abbr, "current")
            current_rows = _roster_rows(current_payload)
        except RealNhlImportError:
            current_payload = {}
            current_rows = []

    season_payload: Dict[str, Any] = {}
    season_rows: List[Dict[str, Any]] = []
    try:
        season_payload = _fetch_team_roster(abbr, roster_season)
        season_rows = _roster_rows(season_payload)
    except RealNhlImportError:
        season_payload = {}
        season_rows = []

    primary_payload = current_payload or season_payload
    primary_rows = _merge_roster_rows(current_rows, season_rows) if (current_rows and season_rows) else (current_rows or season_rows)
    if current_rows and season_rows and current_rows != season_rows:
        note = f"merged_live_{len(current_rows)}_season_{len(season_rows)}"
    elif current_rows:
        note = "current"
    elif season_rows:
        note = "season"

    prior_rows: List[Dict[str, Any]] = []
    prior_payload: Dict[str, Any] = {}
    try:
        prior_payload = _fetch_team_roster(abbr, prior_season)
        prior_rows = _roster_rows(prior_payload)
    except Exception:
        prior_rows = []

    prior_ids = {_roster_player_id(row) for row in prior_rows if _roster_player_id(row) > 0}

    def _mark_prior(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Anyone who dressed on last year's NHL roster is not a Calder rookie."""
        for row in rows:
            if _roster_player_id(row) in prior_ids or row.get("_prior_only_candidate"):
                row["_on_prior_season_roster"] = True
        return rows

    if not primary_rows and prior_rows:
        return prior_payload or primary_payload, _mark_prior(prior_rows), "prior_only"
    if len(primary_rows) < 22 and prior_rows:
        merged = _merge_roster_rows(primary_rows, prior_rows)
        note = f"merged_current_{len(primary_rows)}_prior_{len(prior_rows)}_out_{len(merged)}"
        return primary_payload or prior_payload, _mark_prior(merged), note
    if primary_rows:
        # NHL.com's live roster drops injured (IR/LTIR) players even when they are
        # under contract (e.g. Artem Zub, OTT 2026-27). Carry prior-season players
        # forward as candidates; build_real_nhl_league_players keeps one only if
        # he is not on another club's live roster and still holds a contract with
        # this club (filters out departed UFAs, retirees, and traded players).
        primary_ids = {_roster_player_id(r) for r in primary_rows}
        carry: List[Dict[str, Any]] = []
        for row in prior_rows:
            pid = _roster_player_id(row)
            if pid > 0 and pid not in primary_ids:
                tagged = dict(row)
                tagged["_prior_only_candidate"] = True
                carry.append(tagged)
        if carry:
            note = f"{note}_plus_{len(carry)}_prior_candidates"
        return primary_payload, _mark_prior(list(primary_rows) + carry), note
    raise RealNhlImportError(f"{abbr}: no roster for {roster_season} or {prior_season}")


# ---------------------------------------------------------------------------
# Opening-night 23-man NHL roster (overflow → AHL)
# ---------------------------------------------------------------------------

NHL_OPENING_ROSTER_MAX = 23
NHL_OPENING_MIN_FORWARDS = 12
NHL_OPENING_MIN_DEFENSE = 6
NHL_OPENING_MIN_GOALIES = 2


def _player_sort_ovr(player: Any) -> float:
    try:
        fn = getattr(player, "ovr", None)
        v = float(fn() if callable(fn) else fn or 0.0)
    except Exception:
        v = float(getattr(player, "real_nhl_target_ovr", 0) or 0)
    return v * 99.0 if v <= 1.5 else v


def _player_pos_bucket(player: Any) -> str:
    try:
        from services.roster_compliance import position_bucket

        return position_bucket(player)
    except Exception:
        ident = getattr(player, "identity", None)
        pos = getattr(ident, "position", None) if ident is not None else getattr(player, "position", None)
        code = str(getattr(pos, "value", pos) or "").upper()
        if code in ("C", "LW", "RW", "W", "F", "L", "R"):
            return "F"
        if code in ("D", "LD", "RD"):
            return "D"
        if code == "G":
            return "G"
        return "OTHER"


def select_opening_nhl_roster(
    players: List[Any],
    *,
    limit: int = NHL_OPENING_ROSTER_MAX,
    min_forwards: int = NHL_OPENING_MIN_FORWARDS,
    min_defense: int = NHL_OPENING_MIN_DEFENSE,
    min_goalies: int = NHL_OPENING_MIN_GOALIES,
) -> Tuple[List[Any], List[Any]]:
    """
    Pick a 23-man NHL opening roster with NHL-like position floors, send the rest down.

    Returns (nhl_keepers, ahl_overflow).
    """
    pool = list(players or [])
    if len(pool) <= int(limit):
        return pool, []

    # House-rule Brady is forced to 55 OVR — still keep him on the NHL club so
    # Ottawa's real $8.2M hit (and roster identity) are not deleted by trim.
    protected: List[Any] = []
    try:
        from services.brady_tkachuk_chaos import is_brady_tkachuk

        protected = [p for p in pool if is_brady_tkachuk(p)]
    except Exception:
        protected = []

    by_bucket: Dict[str, List[Any]] = {"F": [], "D": [], "G": [], "OTHER": []}
    for p in pool:
        by_bucket.setdefault(_player_pos_bucket(p), []).append(p)
    for key in by_bucket:
        by_bucket[key].sort(key=_player_sort_ovr, reverse=True)

    selected: List[Any] = []
    selected_ids: set = set()

    def _take(bucket: str, n: int) -> None:
        if n <= 0:
            return
        for p in by_bucket.get(bucket) or []:
            if len(selected) >= int(limit):
                return
            pid = id(p)
            if pid in selected_ids:
                continue
            selected.append(p)
            selected_ids.add(pid)
            n -= 1
            if n <= 0:
                return

    for p in protected:
        if len(selected) >= int(limit):
            break
        if id(p) in selected_ids:
            continue
        selected.append(p)
        selected_ids.add(id(p))

    _take("G", int(min_goalies))
    _take("D", int(min_defense))
    _take("F", int(min_forwards))

    leftover = sorted(
        [p for p in pool if id(p) not in selected_ids],
        key=_player_sort_ovr,
        reverse=True,
    )
    for p in leftover:
        if len(selected) >= int(limit):
            break
        selected.append(p)
        selected_ids.add(id(p))

    overflow = [p for p in pool if id(p) not in selected_ids]
    overflow.sort(key=_player_sort_ovr, reverse=True)
    return selected, overflow


def _assign_player_to_ahl(player: Any, team: Any) -> None:
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    try:
        player.in_minors = True
        player.is_buried = True
        player.buried = True
        player.roster_location = "ahl"
        player.organizational_status = "minors"
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        ctx = getattr(player, "context", None)
        if ctx is not None:
            ctx.current_team_id = f"AHL_{tid}" if tid else "AHL"
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _team_label

        _set_assignment(player, org_nhl_team_id=tid, level="ahl", club=_team_label(team))
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _assign_player_to_nhl(player: Any, team: Any) -> None:
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    try:
        player.in_minors = False
        player.is_buried = False
        player.buried = False
        player.roster_location = "nhl"
        player.organizational_status = "nhl"
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        ctx = getattr(player, "context", None)
        if ctx is not None and tid:
            ctx.current_team_id = tid
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _fetch_org_prospects(abbr: str) -> Dict[str, Any]:
    """NHL API organization prospects (forwards / defensemen / goalies), incl. AHL kids."""
    try:
        data = _http_get_json(f"{WEB_API}/prospects/{str(abbr).upper()}", timeout=12)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _player_is_goalie_obj(p: Any) -> bool:
    pos = getattr(getattr(p, "identity", None), "position", None)
    val = str(getattr(pos, "value", pos) or getattr(p, "position", "") or "").upper()
    return val in ("G", "GOALIE")


def import_real_minor_goalies(
    team: Any,
    abbr: str,
    *,
    league: Any,
    rng: random.Random,
    season_year: int,
    as_of: date,
    dynasty_registry: Any,
    align_rounds: int = 40,
    skip_ledger_finalize: bool = False,
    use_api: bool = True,
    min_ahl_goalies: int = 2,
    league_names: Optional[set] = None,
    contracts_by_team: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    """Stock each AHL affiliate with its real goalies.

    The dynasty ratings file lists every club's AHL (and ECHL) goalies, but nothing ever
    created them, so affiliates ran on generated netminders — and could end up with none.
    Real goalies come from the ratings file, enriched with NHL API prospect data
    (id → headshot, birth date, handedness) when the API answers. Generated AHL goalies
    make way for them; every affiliate ends with at least ``min_ahl_goalies``.
    """
    if not hasattr(team, "ahl_roster") or team.ahl_roster is None:
        team.ahl_roster = []
    added = 0
    entries: List[Any] = []
    if dynasty_registry is not None:
        for lvl in ("ahl", "echl"):
            try:
                entries.extend(
                    e for e in dynasty_registry.entries_for_team(abbr, lvl) if getattr(e, "is_goalie", False)
                )
            except Exception:
                continue
    api_rows: Dict[str, Dict[str, Any]] = {}
    if use_api:
        payload = _fetch_org_prospects(abbr)
        for row in list(payload.get("goalies") or []):
            nm = f"{_localized_name(row.get('firstName'))} {_localized_name(row.get('lastName'))}".strip()
            if nm:
                api_rows[_norm_simple(nm)] = row
    # League-wide, not just this org: the ratings file still lists some goalies under
    # their old affiliate, which put the same person on two clubs (Poirier NJD+DAL,
    # Stevenson WSH AHL + WPG).
    existing_names = league_names if league_names is not None else set()
    for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
        for p in list(getattr(team, attr, None) or []):
            nm = str(getattr(getattr(p, "identity", None), "name", "") or getattr(p, "name", "") or "")
            if nm:
                existing_names.add(_norm_simple(nm))
    contracts_by_team = contracts_by_team or {}
    own_contracts = contracts_by_team.get(str(abbr or "").upper()) or {}

    def _contract_elsewhere(key: str) -> bool:
        if key in own_contracts:
            return False
        return any(key in (m or {}) for a, m in contracts_by_team.items() if a != str(abbr or "").upper())

    def _own_contract(key: str) -> Optional[Dict[str, Any]]:
        hit = own_contracts.get(key)
        if isinstance(hit, list):
            hit = hit[0] if len(hit) == 1 else None
        return dict(hit) if isinstance(hit, dict) else None

    real_new: List[Any] = []
    for e in entries:
        raw = str(getattr(e, "raw_name", "") or "").strip()
        if " " not in raw:
            continue  # last-name-only rows can't build a real person
        key = _norm_simple(raw)
        if key in existing_names or any(_same_person_name(raw, existing) for existing in existing_names):
            continue
        if _contract_elsewhere(key):
            continue  # Spotrac has him under another organization
        first, last = raw.split(" ", 1)
        row = dict(api_rows.get(key) or {})
        if not row:
            age_guess = 23 + int(rng.random() * 5)
            row = {
                "id": 0,
                "firstName": {"default": first},
                "lastName": {"default": last},
                "positionCode": "G",
                "birthDate": f"{int(season_year) - age_guess}-0{1 + int(rng.random() * 8)}-15",
                "shootsCatches": "L",
                "heightInInches": 74,
                "weightInPounds": 190,
            }
        row["positionCode"] = "G"
        try:
            player = _build_player_from_roster_row(
                row,
                team=team,
                league=league,
                rng=rng,
                season_year=season_year,
                as_of=as_of,
                skater_stats_a={},
                skater_stats_b={},
                goalie_stats_a={},
                goalie_stats_b={},
                dynasty_registry=dynasty_registry,
                align_rounds=align_rounds,
                skip_ledger_finalize=skip_ledger_finalize,
                contract=_own_contract(key),
            )
        except Exception:
            continue
        try:
            from services.dynasty_ratings_parser import apply_dynasty_entry_to_player

            apply_dynasty_entry_to_player(player, e, seed=int(rng.random() * 1e9), align_rounds=align_rounds)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        _clamp_estimated_contract_to_minor_level(player, season_year=int(season_year))
        try:
            from services.contract_economy import apply_contract_to_player, has_active_contract

            if not has_active_contract(player):
                yrs = 1 + int(rng.random() * 2)
                apply_contract_to_player(
                    player,
                    {
                        "aav_m": 0.8,
                        "cap_hit_m": 0.8,
                        "years": yrs,
                        "years_remaining": yrs,
                        "expiry_year": int(season_year) + yrs,
                        "contract_type": "STANDARD",
                        "two_way": True,
                        "minor_salary_m": 0.1,
                        "rights_status": "UFA",
                        "source": "estimated",
                        "is_nhl_spc": True,
                    },
                    int(season_year),
                )
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        _assign_player_to_ahl(player, team)
        setattr(player, "real_minor_import", True)
        real_new.append(player)
        existing_names.add(key)
    # Ratings file thin for this org? Pull pro-age goalies straight from the NHL API.
    if len(real_new) < min_ahl_goalies and api_rows:
        for key, row in api_rows.items():
            if len(real_new) >= min_ahl_goalies or key in existing_names or any(
                _same_person_name(key, existing) for existing in existing_names
            ):
                continue
            if _contract_elsewhere(key):
                continue
            try:
                by = int(str(row.get("birthDate") or "0")[:4] or 0)
            except ValueError:
                by = 0
            if by and int(season_year) - by < 20:
                continue  # still a junior/college kid, not an AHL goalie
            try:
                player = _build_player_from_roster_row(
                    dict(row, positionCode="G"),
                    team=team, league=league, rng=rng, season_year=season_year, as_of=as_of,
                    skater_stats_a={}, skater_stats_b={}, goalie_stats_a={}, goalie_stats_b={},
                    dynasty_registry=dynasty_registry, align_rounds=align_rounds,
                    skip_ledger_finalize=skip_ledger_finalize,
                    contract=_own_contract(key),
                )
            except Exception:
                continue
            _ensure_estimated_contract(player, season_year=int(season_year), rng=rng)
            _clamp_estimated_contract_to_minor_level(player, season_year=int(season_year))
            _assign_player_to_ahl(player, team)
            setattr(player, "real_minor_import", True)
            real_new.append(player)
            existing_names.add(key)
    if real_new:
        # Generated AHL goalies give up their crease to the real ones.
        keep_gen = max(0, min_ahl_goalies - len(real_new))
        new_ahl: List[Any] = []
        gen_kept = 0
        for p in list(team.ahl_roster):
            if _player_is_goalie_obj(p) and not getattr(p, "real_minor_import", False) and not getattr(p, "real_nhl_contract", False):
                if gen_kept < keep_gen:
                    gen_kept += 1
                    new_ahl.append(p)
                else:
                    try:
                        if p in (getattr(league, "players", None) or []):
                            league.players.remove(p)
                    except Exception:
                        _swallowed_log.debug("suppressed exception", exc_info=True)
                continue
            new_ahl.append(p)
        # Best three real goalies play in the AHL; any extra go to the ECHL affiliate.
        real_new.sort(key=lambda q: -_player_sort_ovr(q))
        for i, p in enumerate(real_new):
            if i < 3:
                new_ahl.append(p)
            else:
                if not hasattr(team, "echl_roster") or team.echl_roster is None:
                    team.echl_roster = []
                try:
                    p.roster_location = "echl"
                    from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _team_label

                    _set_assignment(p, org_nhl_team_id=str(getattr(team, "team_id", "") or ""), level="echl", club=_team_label(team))
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
                team.echl_roster.append(p)
            try:
                if p not in (getattr(league, "players", None) or []):
                    league.players.append(p)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            added += 1
        team.ahl_roster = new_ahl
    return added


def _same_person_name(a: str, b: str) -> bool:
    """True when two labels are the same player (accents, 'Last, First', or 'L. Last')."""
    na = _norm_simple(a).split()
    nb = _norm_simple(b).split()
    if not na or not nb:
        return False
    if na == nb or sorted(na) == sorted(nb):
        return True
    if na[-1] == nb[-1] and na[0][:1] == nb[0][:1]:
        if na[0].startswith(nb[0]) or nb[0].startswith(na[0]) or min(len(na[0]), len(nb[0])) <= 2:
            return True
    return False


def _drop_org_name_duplicates(team: Any) -> int:
    """Drop a minors copy when the same person is already on the NHL roster."""
    nhl_names = []
    for p in list(getattr(team, "roster", None) or []):
        ident = getattr(p, "identity", None)
        nm = str(getattr(ident, "name", "") or getattr(p, "name", "") or "")
        if nm:
            nhl_names.append(nm)
    if not nhl_names:
        return 0
    removed = 0
    for attr in ("ahl_roster", "echl_roster", "prospect_pool"):
        kept = []
        for p in list(getattr(team, attr, None) or []):
            ident = getattr(p, "identity", None)
            nm = str(getattr(ident, "name", "") or getattr(p, "name", "") or "")
            if nm and any(_same_person_name(nm, n) for n in nhl_names):
                removed += 1
                continue
            kept.append(p)
        try:
            setattr(team, attr, kept)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return removed


def _norm_simple(name: str) -> str:
    try:
        from services.real_nhl_contracts import normalize_player_name

        return normalize_player_name(name)
    except Exception:
        return " ".join(str(name or "").lower().split())


def trim_team_roster_to_nhl_limit(
    team: Any,
    *,
    limit: int = NHL_OPENING_ROSTER_MAX,
    season_year: Optional[int] = None,
) -> Dict[str, int]:
    """Keep <=23 on NHL roster; move the rest to team.ahl_roster."""
    roster = list(getattr(team, "roster", None) or [])
    keepers, overflow = select_opening_nhl_roster(roster, limit=limit)
    if not hasattr(team, "ahl_roster") or team.ahl_roster is None:
        team.ahl_roster = []

    for p in keepers:
        _assign_player_to_nhl(p, team)
    team.roster = list(keepers)

    moved = 0
    for p in overflow:
        _assign_player_to_ahl(p, team)
        if season_year is not None:
            _clamp_estimated_contract_to_minor_level(p, season_year=int(season_year))
        # Avoid duplicates if already queued.
        if p not in team.ahl_roster:
            team.ahl_roster.append(p)
            moved += 1
    return {"nhl": len(team.roster), "sent_to_ahl": moved}


def _player_has_nmc(player: Any) -> bool:
    try:
        from services.brady_tkachuk_chaos import is_brady_tkachuk

        if is_brady_tkachuk(player):
            return True
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        return bool(c.get("no_move_clause") or c.get("nmc") or c.get("no_movement_clause"))
    return bool(
        getattr(c, "no_move_clause", False)
        or getattr(c, "nmc", False)
        or getattr(player, "no_move_clause", False)
    )


def _demote_savings_millions(player: Any, season_year: int) -> float:
    """Cap dollars freed by assigning this player to AHL (NHL bury rules)."""
    from app.sim_engine.economy.cap_engine import (
        nhl_bury_threshold_millions,
        player_cap_hit_millions,
    )

    hit = float(player_cap_hit_millions(player) or 0.0)
    if hit <= 0:
        return 0.0
    return min(hit, nhl_bury_threshold_millions(season_year))


def enforce_opening_night_cap_compliance(
    league: Any,
    season_year: int,
    *,
    min_active: int = 20,
) -> Dict[str, Any]:
    """
    Opening-night style compliance: active roster AAV + AHL bury residuals
    must fit under the Upper Limit. Demote movable players until compliant
    (never below min_active; never force NMC holders down).
    """
    from app.sim_engine.economy.cap_engine import (
        apply_nhl_salary_cap_for_season,
        calculate_team_cap_snapshot,
        player_cap_hit_millions,
    )

    apply_nhl_salary_cap_for_season(league, int(season_year))
    report = {"teams_fixed": 0, "players_demoted": 0, "still_over": []}

    for team in list(getattr(league, "teams", None) or []):
        trim_team_roster_to_nhl_limit(team)
        demoted_here = 0
        for _ in range(40):
            snap = calculate_team_cap_snapshot(
                team,
                league,
                season_label=f"{int(season_year)}-{(int(season_year) + 1) % 100:02d}",
            )
            space = float(snap.get("usableCapSpace") or 0.0)
            if space >= -0.01:
                break
            active = [
                p
                for p in (getattr(team, "roster", None) or [])
                if not getattr(p, "retired", False)
                and not getattr(p, "in_minors", False)
                and not getattr(p, "is_buried", False)
            ]
            if len(active) <= int(min_active):
                break
            movable = [p for p in active if not _player_has_nmc(p)]
            if not movable:
                break

            # Prefer demoting low-OVR / low-savings-efficiency players that free the most room.
            def _score(p: Any) -> Tuple[float, float]:
                save = _demote_savings_millions(p, int(season_year))
                ovr = _player_sort_ovr(p)
                # Higher score = demote sooner: big save, low ovr.
                return (-save, ovr)

            movable.sort(key=_score)
            victim = movable[0]
            # If demotion saves nothing (no contract), skip to next.
            if _demote_savings_millions(victim, int(season_year)) <= 0.01 and player_cap_hit_millions(victim) <= 0:
                # Still demote unsigned depth to free a roster spot for later — rare.
                pass
            try:
                team.roster = [p for p in team.roster if p is not victim]
            except Exception:
                break
            _assign_player_to_ahl(victim, team)
            if not hasattr(team, "ahl_roster") or team.ahl_roster is None:
                team.ahl_roster = []
            if victim not in team.ahl_roster:
                team.ahl_roster.append(victim)
            demoted_here += 1
            report["players_demoted"] += 1

        snap = calculate_team_cap_snapshot(
            team,
            league,
            season_label=f"{int(season_year)}-{(int(season_year) + 1) % 100:02d}",
        )
        if demoted_here:
            report["teams_fixed"] += 1
        if float(snap.get("usableCapSpace") or 0.0) < -0.01:
            abbr = str(getattr(team, "abbreviation", None) or getattr(team, "abbr", "") or "?")
            over_by = round(-float(snap.get("usableCapSpace") or 0.0), 3)
            report["still_over"].append(
                {
                    "team": abbr,
                    "over_by_m": over_by,
                    "total_m": snap.get("totalCapHit"),
                    "upper_m": snap.get("upperLimit"),
                }
            )
            # Every real club opens the season cap-compliant, mostly via LTIR. The NHL.com
            # import carries no injury data, so that relief was missing and these clubs were
            # forced into cap-dump trades on day one. Grant a season-scoped, team-level
            # LTIR allowance equal to the unresolved overage (no invented player injuries).
            existing = float(getattr(team, "ltir_pool_m", 0.0) or 0.0)
            try:
                team.ltir_pool_m = round(existing + over_by + 0.05, 3)
                team.ltir_pool_season = int(season_year)
                team.ltir_pool_source = "opening_day_real_nhl_ltir_allowance"
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            report.setdefault("ltir_allowances", []).append({"team": abbr, "allowance_m": round(over_by + 0.05, 3)})
    try:
        import logging as _logging

        _log = _logging.getLogger(__name__)
        for row in report.get("still_over") or []:
            _log.info(
                "opening-night cap: %s over by $%.2fM (total $%sM vs upper $%sM) → LTIR allowance granted",
                row["team"], row["over_by_m"], row["total_m"], row["upper_m"],
            )
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return report


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_real_nhl_league_players(
    *,
    teams: List[Any],
    league: Any,
    rng: random.Random,
    season_year: int,
) -> Dict[str, Any]:
    """
    Replace empty NHL team.rosters with live NHL.com players + R2/R4 ratings.

    season_year is the calendar start year (2025 → 2025–26 roster).
    Stats prefer the prior completed season, then blend/fallback to current.
    """
    if not teams:
        raise RealNhlImportError("No NHL teams available for import.")

    sy = int(season_year)
    roster_season = _season_id(sy)
    # Primary stats: last completed season. Secondary: current (may be thin).
    stats_primary_id = _season_id(sy - 1)
    stats_secondary_id = roster_season
    as_of = date(sy, 9, 15)
    r4_overrides = load_r4_overrides()
    dynasty_registry = None
    dynasty_meta: Dict[str, Any] = {}
    _dynasty_unmatched: List[Any] = []
    fast_import = _real_nhl_fast_import_enabled()
    try:
        from services.dynasty_ratings_parser import load_dynasty_ratings_registry

        dynasty_registry = load_dynasty_ratings_registry()
        dynasty_meta = dict(getattr(dynasty_registry, "parse_stats", {}) or {})
    except Exception as e:
        dynasty_meta = {"load_error": str(e)}

    analytics_by_id: Dict[int, Dict[str, Any]] = {}
    goalie_analytics_by_id: Dict[int, Dict[str, Any]] = {}
    if not fast_import:
        try:
            from services.real_nhl_analytics import (
                fetch_moneypuck_goalie_analytics_prefer,
                fetch_moneypuck_skater_analytics_prefer,
            )

            analytics_by_id = fetch_moneypuck_skater_analytics_prefer(sy - 1, sy)
            goalie_analytics_by_id = fetch_moneypuck_goalie_analytics_prefer(sy - 1, sy)
        except Exception:
            analytics_by_id = {}
            goalie_analytics_by_id = {}

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sk_fut = pool.submit(_fetch_skater_summary, stats_primary_id)
            gl_fut = pool.submit(_fetch_goalie_summary, stats_primary_id)
            skater_a = sk_fut.result()
            goalie_a = gl_fut.result()
    except RealNhlImportError:
        raise
    except Exception as e:
        raise RealNhlImportError(str(e), code="REAL_NHL_STATS_FETCH_FAILED") from e

    skater_b: Dict[int, Dict[str, Any]] = {}
    goalie_b: Dict[int, Dict[str, Any]] = {}
    if not fast_import:
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                sk_fut = pool.submit(_fetch_skater_summary, stats_secondary_id)
                gl_fut = pool.submit(_fetch_goalie_summary, stats_secondary_id)
                skater_b = sk_fut.result()
                goalie_b = gl_fut.result()
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    if not hasattr(league, "players") or league.players is None:
        league.players = []
    league.players = [p for p in league.players if not getattr(p, "real_nhl_import", False)]

    roster_payloads: Dict[str, Dict[str, Any]] = {}
    roster_rows_by_abbr: Dict[str, List[Dict[str, Any]]] = {}
    roster_failures: List[str] = []
    all_rows: List[Tuple[str, Dict[str, Any]]] = []

    roster_workers = min(20, max(8, len(teams)))
    with ThreadPoolExecutor(max_workers=roster_workers) as pool:
        futs = [
            pool.submit(_fetch_team_roster_bundle, team, roster_season, stats_primary_id)
            for team in teams
        ]
        for fut in as_completed(futs):
            abbr, payload, rows, err = fut.result()
            if err:
                roster_failures.append(err)
                continue
            if not abbr or payload is None:
                continue
            if len(rows) < 12:
                roster_failures.append(f"{abbr}: thin roster ({len(rows)})")
            roster_payloads[abbr] = payload
            roster_rows_by_abbr[abbr] = rows
            for row in rows:
                all_rows.append((abbr, row))

    # Live-roster ids league-wide; a prior-season candidate who now appears on
    # any club's live roster was moved and must not be duplicated.
    live_roster_ids: set = set()
    for _, row in all_rows:
        if not row.get("_prior_only_candidate"):
            pid_live = _roster_player_id(row)
            if pid_live > 0:
                live_roster_ids.add(pid_live)

    nhl_ids = []
    for _, row in all_rows:
        try:
            nhl_ids.append(int(row.get("id") or 0))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    landings: Dict[int, Dict[str, Any]] = {}
    if not fast_import:
        landings = fetch_landings_by_id(nhl_ids, max_workers=24)

    from services.real_nhl_contracts import (
        fetch_league_contracts_by_team,
        match_contract_for_player,
    )

    team_abbrs = list(roster_payloads.keys())
    contracts_by_team: Dict[str, Dict[str, Dict[str, Any]]] = {}
    contract_failures: List[str] = []
    dead_cap_by_team: Dict[str, Dict[str, Any]] = {}
    # Spotrac contracts are required for realistic AAV/terms even in fast import.
    contracts_by_team, contract_failures, dead_cap_by_team = fetch_league_contracts_by_team(
        team_abbrs, sy, max_workers=16
    )

    align_rounds = 8 if fast_import else 40
    skip_ledger_finalize = bool(fast_import)

    imported = 0
    per_team: Dict[str, int] = {}
    failures: List[str] = list(roster_failures) + list(contract_failures)
    r4_applied = 0
    dynasty_applied = 0
    dynasty_missing = 0
    contracts_applied = 0
    drafts_applied = 0
    sent_to_ahl = 0
    restored_missing: List[str] = []

    try:
        from services.real_nhl_contracts import normalize_player_name as _norm_name
    except Exception:  # pragma: no cover
        _norm_name = lambda s: str(s or "").strip().lower()  # noqa: E731

    minor_goalie_queue: List[Tuple[Any, str]] = []
    for team in teams:
        abbr = str(
            getattr(team, "abbreviation", None)
            or getattr(team, "abbr", None)
            or ""
        ).upper()
        if not abbr or abbr not in roster_payloads:
            continue
        rows = roster_rows_by_abbr.get(abbr) or _roster_rows(roster_payloads[abbr])

        if not hasattr(team, "roster") or team.roster is None:
            team.roster = []
        team.roster.clear()
        if hasattr(team, "scratches") and team.scratches is not None:
            team.scratches.clear()

        count = 0
        seen_ids: set = set()
        for row in rows:
            try:
                pid = int(row.get("id") or 0)
            except Exception:
                pid = 0
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            first = _localized_name(row.get("firstName"))
            last = _localized_name(row.get("lastName"))
            full_name = f"{first} {last}".strip()
            pos_code = str(row.get("positionCode") or "").upper()
            is_restored = bool(row.get("_prior_only_candidate"))
            if is_restored:
                if pid and pid in live_roster_ids:
                    continue  # now on another club's live roster
                # Exact-name contract with THIS club for this season (no last-name
                # fallback, to avoid matching a different player with the same surname).
                if _norm_name(full_name) not in (contracts_by_team.get(abbr) or {}):
                    continue
                landing_chk = landings.get(pid) if pid else None
                if isinstance(landing_chk, dict):
                    if landing_chk.get("isActive") is False:
                        continue
                    cur_abbr = str(landing_chk.get("currentTeamAbbrev") or "").upper()
                    if cur_abbr and cur_abbr != abbr:
                        continue
            contract = match_contract_for_player(
                full_name,
                abbr,
                contracts_by_team,
                position_code=pos_code,
            )
            landing = landings.get(pid) if pid else None
            try:
                player = _build_player_from_roster_row(
                    row,
                    team=team,
                    league=league,
                    rng=rng,
                    season_year=sy,
                    as_of=as_of,
                    skater_stats_a=skater_a,
                    skater_stats_b=skater_b,
                    goalie_stats_a=goalie_a,
                    goalie_stats_b=goalie_b,
                    r4_overrides=r4_overrides,
                    landing=landing,
                    contract=contract,
                    all_teams=teams,
                    analytics_by_id=analytics_by_id,
                    goalie_analytics_by_id=goalie_analytics_by_id,
                    dynasty_registry=dynasty_registry,
                    align_rounds=align_rounds,
                    skip_ledger_finalize=skip_ledger_finalize,
                )
            except Exception as e:
                failures.append(f"{abbr} player build failed: {e}")
                continue
            if getattr(player, "dynasty_ratings_import", False):
                dynasty_applied += 1
            elif dynasty_registry is not None:
                dynasty_missing += 1
                _dynasty_unmatched.append((player, full_name, getattr(player, "nhl_id", None) or getattr(player, "nhl_player_id", None)))
            if getattr(player, "real_nhl_r4", False):
                r4_applied += 1
            if getattr(player, "real_nhl_contract", False):
                contracts_applied += 1
            if getattr(player, "drafted", False) or getattr(player, "undrafted", False):
                drafts_applied += 1
            if is_restored:
                setattr(player, "restored_from_prior_roster", True)
                restored_missing.append(f"{abbr}:{full_name}")
            if row.get("_on_prior_season_roster") or is_restored:
                setattr(player, "prior_nhl_roster", True)
            team.roster.append(player)
            league.players.append(player)
            count += 1
            imported += 1

        trim_info = trim_team_roster_to_nhl_limit(team, season_year=sy)
        per_team[abbr] = int(trim_info.get("nhl") or len(team.roster))
        sent_to_ahl += int(trim_info.get("sent_to_ahl") or 0)
        minor_goalie_queue.append((team, abbr))

        # Spotrac dead money (buyouts / retained) — previously never imported, so
        # clubs looked ~$5M too loose vs CapFriendly/Spotrac once AAVs were fixed.
        dead = dead_cap_by_team.get(abbr) or {}
        try:
            team.buyout_cap_hits = list(dead.get("buyouts") or [])
            team.buyouts = list(dead.get("buyouts") or [])
            team.retained_salary_records = list(dead.get("retained") or [])
            team.retained_salary = list(dead.get("retained") or [])
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

        # Competitive score for AI / standings priors
        try:
            if hasattr(team, "state") and hasattr(team.state, "competitive_score"):
                ovrs = sorted(
                    (
                        float(p.ovr())
                        for p in team.roster
                        if callable(getattr(p, "ovr", None))
                        and not getattr(p, "in_minors", False)
                    ),
                    reverse=True,
                )[:12]
                team.state.competitive_score = sum(ovrs) / len(ovrs) if ovrs else 0.5
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    # Minor goalies go in after every NHL roster exists, checked against the whole
    # league, so a goalie the ratings file still lists under an old affiliate is not
    # created a second time.
    league_names: set = set()
    for t in teams:
        for attr in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
            for p in list(getattr(t, attr, None) or []):
                nm = str(getattr(getattr(p, "identity", None), "name", "") or getattr(p, "name", "") or "")
                if nm:
                    league_names.add(_norm_simple(nm))
    for team, abbr in minor_goalie_queue:
        try:
            import_real_minor_goalies(
                team,
                abbr,
                league=league,
                rng=rng,
                season_year=sy,
                as_of=as_of,
                dynasty_registry=dynasty_registry,
                align_rounds=align_rounds,
                skip_ledger_finalize=skip_ledger_finalize,
                use_api=not fast_import,
                league_names=league_names,
                contracts_by_team=contracts_by_team,
            )
        except Exception as e:  # noqa: BLE001
            failures.append(f"{abbr} AHL goalie import failed: {e}")
        try:
            _drop_org_name_duplicates(team)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    if imported < 400:
        detail = "; ".join(failures[:6]) if failures else "unknown"
        raise RealNhlImportError(
            f"Real NHL import incomplete ({imported} players). {detail}",
            code="REAL_NHL_INCOMPLETE_IMPORT",
        )

    # Second pass: players the txt lists under another club (trades / stale headers).
    if dynasty_registry is not None and _dynasty_unmatched:
        try:
            from services.dynasty_ratings_parser import apply_dynasty_entry_to_player, normalize_player_name

            claimed = set()
            used_keys = {
                str(getattr(p, "_dynasty_entry_key", "") or "")
                for t in teams for p in (getattr(t, "roster", None) or [])
                if getattr(p, "dynasty_ratings_import", False)
            }
            for e in dynasty_registry.entries:
                if f"{e.team_abbr}|{e.level}|{e.raw_name}" in used_keys:
                    claimed.add(id(e))
            last_counts: Dict[str, int] = {}
            for _p, nm, _i in _dynasty_unmatched:
                lk = normalize_player_name(str(nm).split()[-1] if nm else "")
                last_counts[lk] = last_counts.get(lk, 0) + 1
            for p, nm, nid in _dynasty_unmatched:
                entry = dynasty_registry.match_anywhere(str(nm), claimed=claimed, unmatched_last_counts=last_counts)
                if entry is None:
                    continue
                apply_dynasty_entry_to_player(p, entry, seed=nid, align_rounds=40)
                claimed.add(id(entry))
                dynasty_applied += 1
                dynasty_missing = max(0, dynasty_missing - 1)
        except Exception as _dyn_err:
            failures.append(f"dynasty second pass failed: {_dyn_err}")

    if not fast_import:
        try:
            from app.sim_engine.entities.player import enforce_league_ovr_distribution_from_league

            enforce_league_ovr_distribution_from_league(league, rng=rng, target_90_plus=0)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    brady_meta: Dict[str, Any] = {}
    if not fast_import:
        try:
            from services.brady_tkachuk_chaos import apply_brady_chaos_to_league

            brady_meta = apply_brady_chaos_to_league(teams)
        except Exception as e:
            brady_meta = {"ok": False, "error": str(e)}

    try:
        setattr(league, "real_nhl_spotrac_contracts", contracts_by_team)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        from services.contract_economy import repair_league_contract_terms

        repair_league_contract_terms(league, sy, contracts_by_team)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    try:
        setattr(league, "real_nhl_import_meta", {
            "season_year": sy,
            "fast_import": bool(fast_import),
            "roster_season_id": roster_season,
            "stats_primary_season_id": stats_primary_id,
            "imported_players": imported,
            "per_team": per_team,
            "failures": failures[:20],
            "rating_model": "dynasty_txt" if dynasty_registry else "R2_R3_R4",
            "dynasty_ratings": {
                **dynasty_meta,
                "nhl_applied": dynasty_applied,
                "nhl_missing": dynasty_missing,
            },
            "r4_applied": r4_applied,
            "contracts_applied": contracts_applied,
            "drafts_applied": drafts_applied,
            "landings_fetched": len(landings),
            "moneypuck_players": len(analytics_by_id),
            "moneypuck_goalies": len(goalie_analytics_by_id),
            "sent_to_ahl": sent_to_ahl,
            "restored_missing_from_live_roster": restored_missing,
            "nhl_roster_max": NHL_OPENING_ROSTER_MAX,
            "brady_tkachuk_chaos": brady_meta,
        })
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    return {
        "ok": True,
        "imported": imported,
        "per_team": per_team,
        "failures": failures,
        "rating_model": "dynasty_txt" if dynasty_registry else "R2_R3_R4",
        "dynasty_ratings": {
            **dynasty_meta,
            "nhl_applied": dynasty_applied,
            "nhl_missing": dynasty_missing,
        },
        "r4_applied": r4_applied,
        "contracts_applied": contracts_applied,
        "drafts_applied": drafts_applied,
        "moneypuck_players": len(analytics_by_id),
        "moneypuck_goalies": len(goalie_analytics_by_id),
        "sent_to_ahl": sent_to_ahl,
        "nhl_roster_max": NHL_OPENING_ROSTER_MAX,
        "brady_tkachuk_chaos": brady_meta,
    }
