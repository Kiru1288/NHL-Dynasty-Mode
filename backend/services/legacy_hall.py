"""Franchise Identity + Legacy Hall: the save's permanent history and each club's identity.

Everything lives in one pickle-safe dict, ``session.legacy`` (see ``ensure_legacy``):

* ``franchises[tid]``   permanent record keyed by the stable team id. Branding is a list
  of identity *versions*; a rebrand appends a version, it never touches the team id,
  rosters, standings, contracts or history. Every finalized season row stores the
  identity version the club wore that season, so history keeps its old name.
* ``seasons[year]``     one row per finalized season: standings, champion, finalist,
  playoff series, awards, draft class, coverage flags. Written once per season at the
  rollover boundary (``archive_legacy_season``), idempotent by year.
* ``players[pid]``      compact career records: one line per regular season the save
  simulated (real-NHL imports keep their imported lines, flagged ``source="import"``).
  Career totals and leaderboards are derived on read, never stored twice.
* ``trades``            compact copies of completed trades (``league.trade_history``),
  keeping the original valuation the trade engine recorded at the time.
* ``captaincy[tid]``    every captaincy tenure, appended, never overwritten.
* ``h2h["a|b"]``        cumulative regular-season and playoff meetings (rivalries).
* ``events``            dedup'd historical events (championships, milestones, rebrands,
  captaincy changes, inductions). The id is deterministic, so retries never duplicate.
* ``hof`` / ``retired_numbers`` / ``philosophy``.

Nothing here invents history. Seasons before the save started tracking are "unknown",
not zero; a stat with no source is omitted. Expensive work happens only at the season
boundary; API reads are targeted (one franchise, one player, one season, one page).
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

LEGACY_VERSION = 1
# Bump when a derived archive's mapping changes; ensure_legacy then runs repair_legacy once.
LEGACY_SCHEMA = 2

# Known club colours (primary, secondary, accent, neutral). Colours only, no marks:
# crests are built in the studio from original primitives.
_CLUB_COLORS: Dict[str, Tuple[str, str, str, str]] = {
    "ANA": ("#F47A38", "#000000", "#B9975B", "#FFFFFF"), "BOS": ("#FFB81C", "#000000", "#FFFFFF", "#1A1A1A"),
    "BUF": ("#003087", "#FFB81C", "#FFFFFF", "#ADAFAA"), "CGY": ("#C8102E", "#F1BE48", "#111111", "#FFFFFF"),
    "CAR": ("#CE1126", "#000000", "#A4A9AD", "#FFFFFF"), "CHI": ("#CF0A2C", "#000000", "#FFD100", "#FFFFFF"),
    "COL": ("#6F263D", "#236192", "#A2AAAD", "#FFFFFF"), "CBJ": ("#002654", "#CE1126", "#A4A9AD", "#FFFFFF"),
    "DAL": ("#006847", "#8F8F8C", "#111111", "#FFFFFF"), "DET": ("#CE1126", "#FFFFFF", "#8A0E1C", "#F2F2F2"),
    "EDM": ("#041E42", "#FF4C00", "#FFFFFF", "#D9D9D9"), "FLA": ("#041E42", "#C8102E", "#B9975B", "#FFFFFF"),
    "LAK": ("#111111", "#A2AAAD", "#FFFFFF", "#572A84"), "MIN": ("#154734", "#A6192E", "#EAAA00", "#DDCBA4"),
    "MTL": ("#AF1E2D", "#192168", "#FFFFFF", "#F2F2F2"), "NSH": ("#FFB81C", "#041E42", "#FFFFFF", "#1A1A1A"),
    "NJD": ("#CE1126", "#000000", "#FFFFFF", "#F2F2F2"), "NYI": ("#00539B", "#F47D30", "#FFFFFF", "#F2F2F2"),
    "NYR": ("#0038A8", "#CE1126", "#FFFFFF", "#F2F2F2"), "OTT": ("#C52032", "#000000", "#C2912C", "#FFFFFF"),
    "PHI": ("#F74902", "#000000", "#FFFFFF", "#F2F2F2"), "PIT": ("#000000", "#FCB514", "#FFFFFF", "#CFC493"),
    "SJS": ("#006D75", "#000000", "#EA7200", "#FFFFFF"), "SEA": ("#001628", "#99D9D9", "#E9072B", "#FFFFFF"),
    "STL": ("#002F87", "#FCB514", "#041E42", "#FFFFFF"), "TBL": ("#002868", "#FFFFFF", "#8DA9C4", "#F2F2F2"),
    "TOR": ("#00205B", "#FFFFFF", "#8DA9C4", "#F2F2F2"), "UTA": ("#6CACE4", "#010101", "#FFFFFF", "#F2F2F2"),
    "VAN": ("#00205B", "#00843D", "#041C2C", "#FFFFFF"), "VGK": ("#B4975A", "#333F42", "#C8102E", "#FFFFFF"),
    "WSH": ("#041E42", "#C8102E", "#FFFFFF", "#F2F2F2"), "WPG": ("#041E42", "#004C97", "#AC162C", "#FFFFFF"),
}
_DEFAULT_COLORS = ("#1B2A55", "#C98A3C", "#FFFFFF", "#E8E4DA")

PHILOSOPHIES = {
    "draft_develop": "Draft and develop",
    "trade_established": "Trade for established talent",
    "homegrown": "Prioritize homegrown players",
    "win_now": "Pursue immediate contention",
    "patient_build": "Build patiently",
    "veteran_leadership": "Invest in veteran leadership",
    "continuity": "Favor long-term roster continuity",
    "flexibility": "Prioritize flexibility and future assets",
}

CREST_SHAPES = ("shield", "circle", "oval", "hexagon", "diamond", "banner", "roundel")
CREST_SYMBOLS = ("none", "star", "crown", "puck", "sticks", "maple", "bolt", "peak", "wave", "wolf", "bear", "bird", "anchor")
JERSEY_STRIPES = ("none", "single", "double", "triple", "heritage", "split")
NUMBER_FONTS = ("block", "athletic", "serif", "rounded", "condensed")
NAMEPLATES = ("arched", "straight", "vertical-arch")
COLLARS = ("v-neck", "lace", "crew", "laced-v")

# Milestones checked only at season finalization, only for players who played that season.
_CAREER_MILESTONES = (("gp", (500, 1000, 1500)), ("pts", (500, 1000, 1500)), ("g", (300, 500, 700)), ("a", (500, 1000)))
_GOALIE_MILESTONES = (("w", (200, 300, 400, 500)), ("so", (50, 75, 100)))
# Hall of Fame: explicit, documented thresholds over the career this save can see.
HOF_RULES = {"skater_pts": 900, "skater_gp": 800, "goalie_w": 330, "major_awards": 2}

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
_NAME = re.compile(r"^[A-Za-z0-9À-ÿ .'&-]{2,32}$")
_ABBR = re.compile(r"^[A-Z]{2,4}$")


# --------------------------------------------------------------------------- helpers
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _i(v: Any) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return 0


def _tid(team: Any) -> str:
    return str(getattr(team, "team_id", "") or getattr(team, "id", "") or "")


def _pid(p: Any) -> str:
    return str(getattr(p, "id", "") or getattr(p, "player_id", "") or "")


def _teams(session: Any) -> List[Any]:
    return list(getattr(getattr(getattr(session, "sim", None), "league", None), "teams", None) or [])


def _team(session: Any, tid: str) -> Any:
    return (getattr(session, "team_by_id", None) or {}).get(str(tid))


def _abbr(team: Any) -> str:
    return str(getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or _tid(team)).upper()


def _pname(p: Any) -> str:
    ident = getattr(p, "identity", None)
    n = getattr(ident, "name", None) or getattr(p, "name", None)
    if not n:
        fn, ln = getattr(ident, "first_name", "") or "", getattr(ident, "last_name", "") or ""
        n = f"{fn} {ln}".strip()
    return str(n or _pid(p))


def _ppos(p: Any) -> str:
    pos = getattr(p, "position", None) or getattr(getattr(p, "identity", None), "position", None)
    return str(getattr(pos, "value", pos) or "").upper()


def _rating(p: Any, *keys: str) -> Optional[float]:
    r = getattr(p, "ratings", None)
    if isinstance(r, dict):
        for k in keys:
            if k in r and r[k] is not None:
                try:
                    return float(r[k])
                except (TypeError, ValueError):
                    continue
    return None


def _event_id(*parts: Any) -> str:
    return "ev_" + hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _pair(a: str, b: str) -> str:
    return "|".join(sorted((str(a), str(b))))


def _season_label(y: int) -> str:
    return f"{y}-{str(y + 1)[-2:]}"


# --------------------------------------------------------------------------- store
def ensure_legacy(session: Any) -> Dict[str, Any]:
    """Create or migrate ``session.legacy``. Safe to call repeatedly (pure defaults)."""
    lg = getattr(session, "legacy", None)
    if not isinstance(lg, dict):
        lg = {}
    lg.setdefault("version", LEGACY_VERSION)
    for key in ("franchises", "seasons", "players", "trades", "captaincy", "h2h", "events", "hof",
                "retired_numbers", "philosophy", "presets"):
        if not isinstance(lg.get(key), dict):
            lg[key] = {}
    cov = lg.get("coverage") if isinstance(lg.get("coverage"), dict) else {}
    cov.setdefault("first_tracked_season", None)
    cov.setdefault("backfilled_seasons", [])
    cov.setdefault("notes", [])
    lg["coverage"] = cov
    session.legacy = lg
    for team in _teams(session):
        _ensure_franchise(session, lg, team)
    if not lg.get("_backfill_done"):
        _backfill_from_session_history(session, lg)
        lg["_backfill_done"] = True
    if int(lg.get("schema") or 1) < LEGACY_SCHEMA:
        repair_legacy(session, lg)
        lg["schema"] = LEGACY_SCHEMA
    return lg


def repair_legacy(session: Any, lg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rebuild derived archives from their sources where the sources still exist: trade
    copies (league.trade_history) and draft classes (draft_results_archive). Simulated
    season rows, careers, events and identity versions are never touched."""
    lg = lg if lg is not None else ensure_legacy(session)
    before = len(lg["trades"])
    lg["trades"] = {}
    _archive_trades(session, lg)
    redrafted = []
    for k, row in lg["seasons"].items():
        if row.get("source") != "simulated":
            continue
        picks = _draft_class(session, int(k))
        if picks and any(p.get("draft_year") for p in picks):
            row["draft"] = picks
            redrafted.append(int(k))
            for pick in picks:
                pr = lg["players"].get(pick.get("player_id") or "")
                if isinstance(pr, dict):
                    pr["draft"] = {kk: pick.get(kk) for kk in ("season", "draft_year", "round", "overall", "team_id")}
    lg.pop("_cache", None)
    return {"trades_before": before, "trades_after": len(lg["trades"]), "draft_seasons_rebuilt": sorted(redrafted)}


def _default_identity(team: Any) -> Dict[str, Any]:
    abbr = _abbr(team)
    p, s, a, n = _CLUB_COLORS.get(abbr, _DEFAULT_COLORS)
    city = str(getattr(team, "city", "") or "")
    raw = str(getattr(team, "name", "") or getattr(team, "team_name", "") or abbr)
    if city and raw.startswith(city):
        name, nickname = raw, raw[len(city):].strip() or raw
    else:
        name, nickname = (f"{city} {raw}".strip() if city else raw), raw
    initials = "".join(w[0] for w in (nickname or abbr).split()[:2]).upper() or abbr[:2]
    colors = {"primary": p, "secondary": s, "accent": a, "neutral": n}
    return {
        "name": name, "city": city, "nickname": nickname, "abbreviation": abbr,
        "colors": colors,
        "crest": default_crest(initials, colors),
        "use_custom_crest": False,
        "uniforms": {
            "home": default_uniform("home", colors),
            "away": default_uniform("away", colors),
            "alternate": None,
        },
        "arena": {"ice_logo": "crest", "boards_trim": "primary"},
    }


def default_crest(initials: str, colors: Dict[str, str]) -> Dict[str, Any]:
    return {
        "shape": "shield",
        "layers": [
            {"id": "base", "kind": "shape", "shape": "shield", "fill": colors["primary"], "stroke": colors["secondary"],
             "stroke_width": 6, "x": 0, "y": 0, "scale": 1.0, "rotation": 0, "visible": True},
            {"id": "inner", "kind": "shape", "shape": "shield", "fill": "none", "stroke": colors["accent"],
             "stroke_width": 3, "x": 0, "y": 0, "scale": 0.82, "rotation": 0, "visible": True},
            {"id": "mark", "kind": "text", "text": initials[:3], "fill": colors["neutral"], "stroke": colors["secondary"],
             "stroke_width": 2, "font": "block", "x": 0, "y": 6, "scale": 1.0, "rotation": 0, "visible": True},
        ],
    }


def default_uniform(kind: str, colors: Dict[str, str]) -> Dict[str, Any]:
    home = kind != "away"
    body = colors["primary"] if home else colors["neutral"]
    trim = colors["secondary"] if home else colors["primary"]
    return {
        "body": body, "sleeve": body, "shoulder": trim, "stripe_color": trim, "stripe_accent": colors["accent"],
        "stripe": "double", "waist_stripe": True, "collar": "v-neck", "collar_color": trim,
        "number_font": "block", "number_color": colors["neutral"] if home else colors["primary"],
        "number_outline": colors["secondary"] if home else colors["secondary"],
        "nameplate": "arched", "name_color": colors["neutral"] if home else colors["primary"],
        "crest_scale": 1.0, "crest_y": 0.0, "fabric_roughness": 0.78, "captain_letter": True,
    }


def _ensure_franchise(session: Any, lg: Dict[str, Any], team: Any) -> Dict[str, Any]:
    tid = _tid(team)
    fr = lg["franchises"].get(tid)
    if isinstance(fr, dict):
        return fr
    year = int(getattr(session, "season_calendar_year", 0) or 0)
    ident = _default_identity(team)
    fr = {
        "franchise_id": tid,
        "original_identity": {k: ident[k] for k in ("name", "city", "nickname", "abbreviation")},
        "first_season_in_save": year or None,
        "identity_versions": [{"version": 1, "effective_season": year or None, "identity": ident, "reason": "Original identity",
                               "source": "initial", "changed": [], "at": _now_iso()}],
        "current_version": 1,
        "relocations": [],
    }
    lg["franchises"][tid] = fr
    return fr


def current_identity(session: Any, tid: str) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    fr = lg["franchises"].get(str(tid)) or {}
    vers = fr.get("identity_versions") or []
    cur = int(fr.get("current_version") or 1)
    for v in vers:
        if int(v.get("version") or 0) == cur:
            return v.get("identity") or {}
    return (vers[-1].get("identity") if vers else {}) or {}


def identity_version_for(lg: Dict[str, Any], tid: str, version: Optional[int]) -> Dict[str, Any]:
    fr = lg["franchises"].get(str(tid)) or {}
    for v in fr.get("identity_versions") or []:
        if version is None or int(v.get("version") or 0) == int(version):
            return v.get("identity") or {}
    return {}


def _record_event(lg: Dict[str, Any], ev_type: str, season: Optional[int], key: Iterable[Any], *,
                  team_ids: Iterable[str] = (), player_ids: Iterable[str] = (), payload: Optional[Dict[str, Any]] = None,
                  iso: Optional[str] = None) -> Tuple[str, bool]:
    """Idempotent: the id is derived from (type, key), so a retry returns the same event."""
    eid = _event_id(ev_type, *key)
    if eid in lg["events"]:
        return eid, False
    tids = [str(t) for t in team_ids if t]
    # Names as they were when it happened: a later rebrand must not rewrite this record.
    names = {}
    for t in tids:
        fr = lg["franchises"].get(t) or {}
        names[t] = (identity_version_for(lg, t, fr.get("current_version")) or {}).get("name")
    lg["events"][eid] = {
        "id": eid, "type": ev_type, "season": season, "iso": iso,
        "team_ids": tids, "player_ids": [str(p) for p in player_ids if p],
        "payload": dict(payload or {}), "team_names_then": names, "recorded_at": _now_iso(),
    }
    return eid, True


# --------------------------------------------------------------------------- season finalization
_LEGACY_STAGES = ("regular", "cup", "awards", "draft", "retirements", "rollover")


def _legacy_row(lg: Dict[str, Any], sy: int) -> Dict[str, Any]:
    key = str(sy)
    row = lg["seasons"].get(key)
    if not isinstance(row, dict):
        row = {
            "season": sy, "label": _season_label(sy), "complete": False, "source": "simulated",
            "stages": {}, "coverage": {},
        }
        lg["seasons"][key] = row
    row.setdefault("stages", {})
    row.setdefault("coverage", {})
    return row


def archive_legacy_stage(session: Any, stage: str, season_year: Optional[int] = None) -> Dict[str, Any]:
    """Add one slice to this season's Legacy row. Writing the same stage twice is a no-op.

    ``complete`` is set only by the rollover stage. Earlier stages still show in the Hall.
    """
    import logging
    lg = ensure_legacy(session)
    sy = int(season_year if season_year is not None else getattr(session, "season_calendar_year", 0) or 0)
    row = _legacy_row(lg, sy)
    if row.get("complete"):
        return row
    if stage != "rollover" and (row.get("stages") or {}).get(stage):
        return row
    try:
        if stage == "regular":
            _stage_regular(session, lg, row, sy)
        elif stage == "cup":
            _stage_cup(session, lg, row, sy)
        elif stage == "awards":
            _stage_awards(session, lg, row, sy)
        elif stage == "draft":
            _stage_draft(session, lg, row, sy)
        elif stage == "retirements":
            _stage_retirements(session, lg, row, sy)
        elif stage == "rollover":
            _stage_rollover(session, lg, row, sy)
        else:
            return row
        if stage == "rollover":
            row["complete"] = True
            row["finalized_at"] = _now_iso()
        else:
            row["stages"][stage] = True
        lg.pop("_cache", None)
    except Exception:
        logging.getLogger(__name__).warning("legacy stage %s failed for %s", stage, sy, exc_info=True)
    return row


def archive_legacy_season(session: Any, season_year: int) -> Dict[str, Any]:
    """Fill any stage that has not run, then mark the season complete. Idempotent once complete."""
    lg = ensure_legacy(session)
    sy = int(season_year)
    if (lg["seasons"].get(str(sy)) or {}).get("complete"):
        return lg["seasons"][str(sy)]
    for stage in _LEGACY_STAGES:
        archive_legacy_stage(session, stage, sy)
    return lg["seasons"].get(str(sy)) or _legacy_row(lg, sy)


def _stage_regular(session: Any, lg: Dict[str, Any], row: Dict[str, Any], sy: int) -> None:
    recs = getattr(getattr(session, "standings", None), "records", None) or {}
    standings: Dict[str, Dict[str, Any]] = {}
    for tid, rr in (recs.items() if isinstance(recs, dict) else []):
        tid = str(tid)
        w, l, o = _i(getattr(rr, "wins", 0)), _i(getattr(rr, "losses", 0)), _i(getattr(rr, "otl", 0))
        fr = lg["franchises"].get(tid) or {}
        standings[tid] = {"w": w, "l": l, "otl": o, "gp": w + l + o, "pts": _i(getattr(rr, "points", 2 * w + o)),
                          "gf": _i(getattr(rr, "gf", 0)), "ga": _i(getattr(rr, "ga", 0)),
                          "identity_version": int(fr.get("current_version") or 1)}
    for rank, tid in enumerate(sorted(standings, key=lambda t: (-standings[t]["pts"], -(standings[t]["gf"] - standings[t]["ga"]), t)), 1):
        standings[tid]["league_rank"] = rank
    row["standings"] = standings
    row["coverage"]["standings"] = bool(standings)
    n_lines = _archive_player_lines(session, lg, sy, scope="regular")
    row["coverage"]["player_lines"] = n_lines > 0
    _archive_h2h(session, lg, [], None, sy, which="games")
    _import_career_lines(session, lg)
    _season_milestones(session, lg, sy)
    if lg["coverage"].get("first_tracked_season") is None:
        lg["coverage"]["first_tracked_season"] = sy


def _stage_cup(session: Any, lg: Dict[str, Any], row: Dict[str, Any], sy: int) -> None:
    key = str(sy)
    series = _playoff_series(session)
    champ = str(getattr(session, "champion_id", None) or getattr(session, "stanley_cup_winner", None) or "") or None
    row["champion_id"] = champ
    finalist = None
    if series:
        last_round = max(_i(s.get("round_index")) for s in series)
        final = [s for s in series if _i(s.get("round_index")) == last_round]
        if final:
            f = final[0]
            ids = [str(f.get("team_high_id") or ""), str(f.get("team_low_id") or "")]
            finalist = next((t for t in ids if t and t != champ), None)
    row["finalist_id"] = finalist
    row["playoff_series"] = series
    playoff_teams = sorted({t for s in series for t in (str(s.get("team_high_id") or ""), str(s.get("team_low_id") or "")) if t})
    row["playoff_teams"] = playoff_teams
    standings = row.get("standings") or {}
    for tid in standings:
        standings[tid]["made_playoffs"] = (tid in playoff_teams) if playoff_teams else None
        standings[tid]["playoff_round_reached"] = _round_reached(series, tid) if series else None
    row["coverage"]["playoffs"] = bool(series)
    n_po = _archive_player_lines(session, lg, sy, scope="playoffs")
    row["coverage"]["playoff_lines"] = n_po > 0
    _credit_championships(lg, sy, champ)
    _archive_h2h(session, lg, series, champ, sy, which="series")
    _season_milestones(session, lg, sy)
    if champ:
        _record_event(lg, "championship", sy, (sy, champ), team_ids=[champ], payload={"finalist_id": finalist})
        first = not any((s.get("champion_id") == champ) for k, s in lg["seasons"].items() if k != key and (s.get("complete") or (s.get("stages") or {}).get("cup")))
        if first:
            _record_event(lg, "first_championship_in_save", sy, (champ,), team_ids=[champ])
    for tid in playoff_teams:
        if not any(tid in (s.get("playoff_teams") or []) for k, s in lg["seasons"].items() if k != key and (s.get("complete") or (s.get("stages") or {}).get("cup"))):
            _record_event(lg, "first_playoffs_in_save", sy, (tid,), team_ids=[tid])


def _stage_awards(session: Any, lg: Dict[str, Any], row: Dict[str, Any], sy: int) -> None:
    awards = _awards(session)
    row["awards"] = awards
    row["coverage"]["awards"] = bool(awards)
    for a in awards:
        pr = lg["players"].get(a.get("player_id") or "")
        if pr is not None and not any(x.get("season") == sy and x.get("award") == a["award"] and x.get("place") in (None, 1, "1") for x in pr["awards"]):
            pr["awards"].append({"season": sy, "award": a["award"], "place": 1})
        for fin in a.get("finalists") or []:
            if not isinstance(fin, dict):
                continue
            fpid = str(fin.get("player_id") or "")
            place = fin.get("place") or fin.get("rank")
            fpr = lg["players"].get(fpid)
            if fpr is None or not place:
                continue
            if not any(x.get("season") == sy and x.get("award") == a["award"] and x.get("place") == place for x in fpr["awards"]):
                fpr["awards"].append({"season": sy, "award": a["award"], "place": place})
    _archive_award_placings(session, lg, sy)


def _stage_draft(session: Any, lg: Dict[str, Any], row: Dict[str, Any], sy: int) -> None:
    row["draft"] = _draft_class(session, sy)
    row["coverage"]["draft"] = bool(row["draft"])
    for pick in row["draft"]:
        pr = lg["players"].get(pick.get("player_id") or "")
        if isinstance(pr, dict) and not pr.get("draft"):
            pr["draft"] = {k: pick.get(k) for k in ("season", "draft_year", "round", "overall", "team_id")}


def _stage_retirements(session: Any, lg: Dict[str, Any], row: Dict[str, Any], sy: int) -> None:
    _archive_retirements(session, lg, sy)
    row["coverage"]["retirements"] = True


def _stage_rollover(session: Any, lg: Dict[str, Any], row: Dict[str, Any], sy: int) -> None:
    _import_career_lines(session, lg)
    _archive_trades(session, lg)
    reconcile_captaincy(session, season=sy, appoint_cpu=True)
    _storyline_hooks(session, lg, sy, row.get("champion_id"))



def _playoff_series(session: Any) -> List[Dict[str, Any]]:
    src: List[Any] = []
    for holder in (getattr(session, "playoff_live", None), getattr(session, "playoff_payload", None)):
        if isinstance(holder, dict):
            src = list(holder.get("series") or holder.get("series_list") or [])
            if src:
                break
    out = []
    for s in src:
        if not isinstance(s, dict):
            continue
        hi, lo = str(s.get("team_high_id") or ""), str(s.get("team_low_id") or "")
        if not hi or not lo:
            continue
        out.append({"round_index": _i(s.get("round_index")) or 1, "team_high_id": hi, "team_low_id": lo,
                    "wins_high": _i(s.get("wins_high")), "wins_low": _i(s.get("wins_low")),
                    "winner_id": str(s.get("winner_id") or "") or None, "conference": s.get("conference")})
    return out


def _round_reached(series: List[Dict[str, Any]], tid: str) -> int:
    rounds = [s["round_index"] for s in series if tid in (s["team_high_id"], s["team_low_id"])]
    return max(rounds) if rounds else 0


def _awards(session: Any) -> List[Dict[str, Any]]:
    aw = getattr(session, "awards_payload", None) or {}
    src = aw.get("official_results") or aw.get("awards") or aw.get("trophies") or []
    out = []
    for a in list(src.values() if isinstance(src, dict) else src):
        if not isinstance(a, dict):
            continue
        pid = a.get("winner_id") or a.get("player_id")
        name = a.get("award_name") or a.get("name") or a.get("award_id") or a.get("id")
        if not name:
            continue
        finalists = []
        for fin in list(a.get("finalists") or a.get("runners_up") or a.get("nominees") or []):
            if isinstance(fin, dict) and (fin.get("player_id") or fin.get("winner_id")):
                finalists.append({
                    "player_id": str(fin.get("player_id") or fin.get("winner_id")),
                    "player_name": fin.get("player_name") or fin.get("name"),
                    "place": fin.get("place") or fin.get("finish") or fin.get("rank"),
                    "team_id": fin.get("team_id"),
                })
        out.append({"award": str(name), "award_id": a.get("award_id") or a.get("id"), "player_id": str(pid) if pid else None,
                    "player_name": a.get("winner_name") or a.get("player_name"), "team_id": a.get("team_id"),
                    "finalists": finalists})
    return out


_SKATER_KEYS = ("gp", "g", "a", "pts", "plus_minus", "pim", "sog", "toi_sec", "ppg", "shg", "gwg", "hit", "blk", "ixg", "fow", "fol")
_GOALIE_KEYS = ("gp", "w", "l", "otl", "so", "ga", "shots_against", "saves", "toi_sec")
_FLOAT_KEYS = ("sv_pct", "gsax", "gaa")


def _blank_player(pid: str) -> Dict[str, Any]:
    return {"player_id": pid, "seasons": [], "teams": [], "championships": [],
            "awards": [], "draft": None, "retired_season": None}


def _stat_line(r: Dict[str, Any], sy: int, team_id: str, scope: str, *, split: bool = False, aggregate: bool = False) -> Dict[str, Any]:
    pos = str(r.get("position") or "").upper()
    goalie = pos.startswith("G")
    line: Dict[str, Any] = {"season": sy, "team_id": str(team_id or ""), "source": "simulated", "scope": scope}
    if split:
        line["split"] = True
    if aggregate:
        line["aggregate"] = True
    for k in (_GOALIE_KEYS if goalie else _SKATER_KEYS):
        if k == "shots_against":
            if "shots_against" in r or "goalie_shots_against" in r:
                line[k] = _i(r.get("shots_against") or r.get("goalie_shots_against"))
        elif k == "ga":
            if "ga" in r or "goalie_ga" in r:
                line[k] = _i(r.get("ga") or r.get("goalie_ga"))
        elif k == "pts":
            line[k] = _i(r.get("pts")) or _i(r.get("g")) + _i(r.get("a"))
        elif k == "plus_minus":
            if "plus_minus" in r or "pm" in r:
                line[k] = _i(r.get("plus_minus") or r.get("pm"))
        elif k == "hit":
            if "hit" in r or "hits" in r:
                line[k] = _i(r.get("hit") if "hit" in r else r.get("hits"))
        elif k == "blk":
            if "blk" in r or "blocks" in r:
                line[k] = _i(r.get("blk") if "blk" in r else r.get("blocks"))
        elif k in r:
            line[k] = _i(r.get(k))
    for k in _FLOAT_KEYS:
        if r.get(k) not in (None, ""):
            try:
                line[k] = round(float(r.get(k)), 3)
            except (TypeError, ValueError):
                pass
    return line


def _team_stat_slices(r: Dict[str, Any]) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Per-team counting stats when a player changed clubs. None if he stayed put."""
    by = r.get("by_team") if isinstance(r.get("by_team"), dict) else {}
    if not by:
        return None
    prior: Dict[str, int] = {}
    slices: List[Tuple[str, Dict[str, Any]]] = []
    for tid, sl in by.items():
        if not isinstance(sl, dict):
            continue
        slices.append((str(tid), sl))
        for k, v in sl.items():
            prior[k] = prior.get(k, 0) + _i(v)
    cur = str(r.get("team_id") or "")
    rem: Dict[str, Any] = {}
    keys = set(_SKATER_KEYS) | set(_GOALIE_KEYS)
    for k in keys:
        if k in r or k in prior:
            rem[k] = _i(r.get(k)) - int(prior.get(k) or 0)
    if cur and _i(rem.get("gp")) > 0:
        slices.append((cur, {**r, **rem, "position": r.get("position")}))
    return slices or None


def _archive_player_lines(session: Any, lg: Dict[str, Any], sy: int, scope: str = "regular") -> int:
    if scope == "playoffs":
        pss = getattr(session, "playoff_player_stats", None) or {}
    else:
        pss = getattr(session, "player_season_stats", None) or {}
    n = 0
    for pid, r in pss.items():
        if not isinstance(r, dict) or _i(r.get("gp")) <= 0:
            continue
        pid = str(r.get("player_id") or pid)
        pos = str(r.get("position") or "").upper()
        slices = _team_stat_slices(r) if scope == "regular" else None
        lines: List[Dict[str, Any]] = []
        if slices and len(slices) > 1:
            for tid, sl in slices:
                src = sl if "position" in sl else {**r, **sl}
                lines.append(_stat_line(src, sy, tid, scope, split=True))
            lines.append(_stat_line(r, sy, "", scope, aggregate=True))
        else:
            lines.append(_stat_line(r, sy, str(r.get("team_id") or ""), scope))
        pr = lg["players"].setdefault(pid, _blank_player(pid))
        pr["name"] = r.get("name") or pr.get("name")
        pr["position"] = pos or pr.get("position")
        pr["seasons"] = [s for s in pr["seasons"] if not (
            s.get("season") == sy and s.get("source") == "simulated" and (s.get("scope") or "regular") == scope
        )]
        for line in lines:
            if line.get("team_id"):
                t = lg["franchises"].get(line["team_id"]) or {}
                line["identity_version"] = int(t.get("current_version") or 1)
            pr["seasons"].append(line)
            if line["team_id"] and line["team_id"] not in pr["teams"]:
                pr["teams"].append(line["team_id"])
            n += 1
    return n


def _credit_championships(lg: Dict[str, Any], sy: int, champ: Optional[str]) -> None:
    """Cup credit for anyone who played for the champion, regular season or playoffs."""
    if not champ:
        return
    for pr in lg["players"].values():
        played = any(
            s.get("season") == sy and str(s.get("team_id") or "") == champ and _i(s.get("gp")) > 0 and not s.get("aggregate")
            for s in (pr.get("seasons") or [])
        )
        if played and sy not in (pr.get("championships") or []):
            pr.setdefault("championships", []).append(sy)


def _import_career_lines(session: Any, lg: Dict[str, Any]) -> None:
    """Merge real-NHL seasons the Hall does not have yet. Safe to call again after a landing-page fetch.

    Franchise opening season and later stay out: those belong to the sim.
    """
    try:
        from services.franchise_sim import _iter_league_players_for_aging
        players = _iter_league_players_for_aging(getattr(session.sim, "league", None))
    except Exception:
        players = []
    try:
        from services.season_carryover import _franchise_first_season
        cutoff = int(_franchise_first_season(session, int(getattr(session, "season_calendar_year", 0) or 0)) or 0)
    except Exception:
        cutoff = 0
    try:
        from services.real_nhl_roster_importer import drop_nhl_import_from_franchise_open, _season_start_year_from_label
    except Exception:
        drop_nhl_import_from_franchise_open = None
        _season_start_year_from_label = None
    for p in players:
        cs = getattr(p, "career_stats", None)
        rows = [s for s in (cs.get("seasons") if isinstance(cs, dict) else []) or [] if isinstance(s, dict) and s.get("source") != "sim"]
        if drop_nhl_import_from_franchise_open is not None and cutoff > 0:
            rows = drop_nhl_import_from_franchise_open(rows, cutoff)
        elif cutoff > 0 and _season_start_year_from_label is not None:
            rows = [s for s in rows if _season_start_year_from_label(s.get("season")) < cutoff]
        if not rows:
            continue
        pid = _pid(p)
        pr = lg["players"].setdefault(pid, _blank_player(pid))
        pr.setdefault("name", _pname(p))
        pr.setdefault("position", _ppos(p))
        have = {
            (str(s.get("season")), str(s.get("team") or ""), str(s.get("league") or ""))
            for s in (pr.get("imported_seasons") or [])
        }
        added = []
        for s in rows[-30:]:
            key = (str(s.get("season")), str(s.get("team") or s.get("team_abbrev") or ""), str(s.get("league") or ""))
            if key in have:
                continue
            have.add(key)
            added.append({"season": str(s.get("season")), "team": s.get("team") or s.get("team_abbrev"),
                          "league": s.get("league"), "gp": s.get("gp"), "g": s.get("g"), "a": s.get("a"),
                          "pts": s.get("pts"), "plus_minus": s.get("plus_minus"), "pim": s.get("pim"),
                          "w": s.get("wins") or s.get("w"), "so": s.get("shutouts") or s.get("so"),
                          "sv_pct": s.get("sv_pct"), "source": "import"})
        if added:
            pr.setdefault("imported_seasons", []).extend(added)


def _archive_award_placings(session: Any, lg: Dict[str, Any], sy: int) -> None:
    """Finalists and other placings live on player_award_history, not only the winner list."""
    label = _season_label(sy)
    try:
        from services.franchise_sim import _iter_league_players_for_aging
        players = _iter_league_players_for_aging(getattr(session.sim, "league", None))
    except Exception:
        return
    for p in players:
        hist = getattr(p, "player_award_history", None)
        if not isinstance(hist, dict):
            continue
        for f in hist.get("finishes") or []:
            if not isinstance(f, dict):
                continue
            season = str(f.get("season") or "")
            if season not in (str(sy), label):
                continue
            place = f.get("finish") or f.get("place")
            award = f.get("award_name") or f.get("award_id")
            if not award or place in (None, ""):
                continue
            pid = _pid(p)
            pr = lg["players"].setdefault(pid, _blank_player(pid))
            pr.setdefault("name", _pname(p))
            if any(x.get("season") == sy and x.get("award") == award and x.get("place") == place for x in pr.get("awards") or []):
                continue
            pr.setdefault("awards", []).append({"season": sy, "award": str(award), "place": place})


def _draft_class(session: Any, sy: int) -> List[Dict[str, Any]]:
    """The entry draft held at the end of season ``sy`` (draft year sy+1). Source of truth is
    ``draft_results_archive`` (one row per pick); the live draft_state is a fallback."""
    rows = [r for r in list(getattr(session, "draft_results_archive", None) or [])
            if isinstance(r, dict) and _i(r.get("draft_year")) == sy + 1]
    if not rows:
        rows = [r for r in list((getattr(session, "draft_state", None) or {}).get("draft_results") or []) if isinstance(r, dict)]
    idx = _player_index(session) if rows else {}
    out = []
    for r in rows:
        pid = r.get("prospect_id") or r.get("player_id") or r.get("id")
        p = idx.get(str(pid)) if pid else None
        out.append({"season": sy, "draft_year": _i(r.get("draft_year")) or sy + 1,
                    "overall": _i(r.get("overall_pick") or r.get("overall")) or None,
                    "round": _i(r.get("round")) or None, "team_id": str(r.get("team_id") or r.get("drafting_team_id") or ""),
                    "player_id": str(pid) if pid else None,
                    "name": r.get("player_name") or r.get("name") or (_pname(p) if p is not None else None),
                    "position": r.get("position") or (_ppos(p) if p is not None else None)})
    out.sort(key=lambda x: x["overall"] or 9999)
    return out


def _archive_trades(session: Any, lg: Dict[str, Any]) -> None:
    hist = list(getattr(getattr(session.sim, "league", None), "trade_history", None) or [])
    idx = None
    for t in hist:
        if not isinstance(t, dict) or not t.get("accepted", True):
            continue
        tid = str(t.get("trade_id") or "")
        if not tid or tid in lg["trades"]:
            continue
        if idx is None:
            idx = _player_index(session)
        moved = []
        for mp in list(t.get("moved_players") or [])[:12]:
            if not isinstance(mp, dict):
                moved.append({"player_id": str(mp)})
                continue
            pid = str(mp.get("asset_id") or mp.get("player_id") or "")
            p = idx.get(pid)
            ident = getattr(p, "identity", None) if p is not None else None
            moved.append({"player_id": pid, "name": mp.get("player_name") or mp.get("name") or (_pname(p) if p is not None else None),
                          "from_team_id": str(mp.get("source_team_id") or mp.get("from_team_id") or ""),
                          "to_team_id": str(mp.get("acquiring_team_id") or mp.get("to_team_id") or ""),
                          # age as of the archive pass (season end), the closest the record gets to trade day
                          "age": getattr(ident, "age", None) if ident is not None else mp.get("age"),
                          "position": _ppos(p) if p is not None else mp.get("position")})
        picks = [{"season": pk.get("year") or pk.get("season"), "round": pk.get("round"), "original_team_id": pk.get("original_team_id"),
                  "from_team_id": str(pk.get("source_team_id") or pk.get("from_team_id") or ""),
                  "to_team_id": str(pk.get("acquiring_team_id") or pk.get("to_team_id") or ""), "label": pk.get("display_name")}
                 for pk in list(t.get("moved_picks") or [])[:12] if isinstance(pk, dict)]
        iso = str(t.get("calendar_iso") or "")
        month = int(iso[5:7]) if len(iso) >= 7 and iso[5:7].isdigit() else 0
        phase = str(t.get("phase") or t.get("season_phase") or "").lower()
        season = _i(t.get("season_year")) or None
        # A July trade is still stamped with the season that just ended. It belongs to the next year.
        if season and month in (7, 8) and phase not in ("regular", "playoffs", "preseason"):
            season += 1
        lg["trades"][tid] = {
            "trade_id": tid, "season": season, "iso": t.get("calendar_iso"),
            "teams": [str(x) for x in (t.get("participating_teams") or [])], "players": moved, "picks": picks,
            "headline": t.get("headline"), "user_involved": bool(t.get("user_involved")),
            # what the trade engine thought at the time; never recomputed
            "original_valuation": dict(t.get("value_scores") or {}), "fairness_gap": t.get("fairness_gap"),
            "identity_versions": {str(x): int((lg["franchises"].get(str(x)) or {}).get("current_version") or 1)
                                  for x in (t.get("participating_teams") or [])},
        }


def _archive_retirements(session: Any, lg: Dict[str, Any], sy: int) -> None:
    for r in list(getattr(session, "retired_players_archive", None) or []):
        if not isinstance(r, dict) or _i(r.get("retirement_year")) != sy:
            continue
        pid = str(r.get("player_id") or r.get("id") or "")
        if not pid:
            continue
        pr = lg["players"].setdefault(pid, {"player_id": pid, "seasons": [], "teams": [], "championships": [],
                                            "awards": [], "draft": None, "retired_season": None})
        pr["name"] = pr.get("name") or r.get("name")
        pr["position"] = pr.get("position") or r.get("position")
        if pr.get("retired_season") is None:
            pr["retired_season"] = sy
            pr["retirement"] = {"type": r.get("retirement_type"), "final_team": r.get("final_team"), "age": r.get("age")}
            _record_event(lg, "retirement", sy, (pid,), player_ids=[pid], team_ids=[str(r.get("team_id") or "")],
                          payload={"name": pr["name"], "final_team": r.get("final_team")})
        _evaluate_hof(lg, pid, sy)


def _archive_h2h(session: Any, lg: Dict[str, Any], series: List[Dict[str, Any]], champ: Optional[str], sy: int, which: str = "both") -> None:
    if which in ("both", "games"):
        for g in list(getattr(session, "game_results", None) or []):
            if str(g.get("stat_scope") or "regular_season") != "regular_season":
                continue
            h, a = str(g.get("home_id") or ""), str(g.get("away_id") or "")
            if not h or not a:
                continue
            hs, as_ = _i(g.get("home_score", g.get("home_goals"))), _i(g.get("away_score", g.get("away_goals")))
            rec = lg["h2h"].setdefault(_pair(h, a), {"gp": 0, "wins": {}, "one_goal": 0, "series": 0, "series_wins": {},
                                                      "seven_game": 0, "finals": 0, "seasons": []})
            rec["gp"] += 1
            win = h if hs > as_ else a
            rec["wins"][win] = rec["wins"].get(win, 0) + 1
            if abs(hs - as_) == 1:
                rec["one_goal"] += 1
            if sy not in rec["seasons"]:
                rec["seasons"].append(sy)
    if which not in ("both", "series"):
        return
    last_round = max((s["round_index"] for s in series), default=0)
    for s in series:
        rec = lg["h2h"].setdefault(_pair(s["team_high_id"], s["team_low_id"]), {"gp": 0, "wins": {}, "one_goal": 0, "series": 0,
                                                                              "series_wins": {}, "seven_game": 0, "finals": 0, "seasons": []})
        mark = f"{sy}:{s['round_index']}"
        done = rec.setdefault("series_keys", [])
        if mark in done:
            continue
        done.append(mark)
        rec["series"] += 1
        if s.get("winner_id"):
            rec["series_wins"][s["winner_id"]] = rec["series_wins"].get(s["winner_id"], 0) + 1
        if s["wins_high"] + s["wins_low"] >= 7:
            rec["seven_game"] += 1
        if s["round_index"] == last_round and last_round >= 3:
            rec["finals"] += 1


def _player_index(session: Any) -> Dict[str, Any]:
    try:
        from services.franchise_sim import _iter_league_players_for_aging
        return {_pid(p): p for p in _iter_league_players_for_aging(getattr(session.sim, "league", None))}
    except Exception:
        return {}


def _counting_lines(pr: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Season totals. Team splits are kept for franchise pages and are not added again here."""
    return [s for s in pr.get("seasons") or [] if s.get("source") == "simulated" and not s.get("split")]


def _season_milestones(session: Any, lg: Dict[str, Any], sy: int) -> None:
    """Players who played this season, against career totals that include the pre-save baseline."""
    for pid, pr in lg["players"].items():
        lines = _counting_lines(pr)
        this = [s for s in lines if s.get("season") == sy]
        if not this:
            continue
        imported = list(pr.get("imported_seasons") or [])
        prior = [s for s in lines if s.get("season") != sy]
        tot = _career_totals(prior + this, imported)
        prev = _career_totals(prior, imported)
        goalie = str(pr.get("position") or "").startswith("G")
        scope = "includes pre-save career" if imported else "in-save career"
        for stat, marks in (_GOALIE_MILESTONES if goalie else _CAREER_MILESTONES):
            for m in marks:
                if prev.get(stat, 0) < m <= tot.get(stat, 0):
                    _record_event(lg, "career_milestone", sy, (pid, stat, m), player_ids=[pid],
                                  team_ids=[this[-1].get("team_id")],
                                  payload={"stat": stat, "threshold": m, "name": pr.get("name"), "scope": scope})
    # franchise single-season points record: one pass for prior bests, one for this season.
    if sum(1 for s in lg["seasons"].values() if s.get("complete")) < 1:
        return  # needs at least one earlier tracked season to be a "record"
    best_prior: Dict[str, int] = {}
    current: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for pid, pr, s in _all_lines(lg):
        if str(pr.get("position") or "").startswith("G"):
            continue
        if s.get("aggregate"):
            continue
        t = s.get("team_id") or ""
        if not t:
            continue
        if s.get("season") == sy:
            current.append((pid, pr, s))
        else:
            best_prior[t] = max(best_prior.get(t, 0), _i(s.get("pts")))
    top: Dict[str, Tuple[str, Dict[str, Any], Dict[str, Any]]] = {}
    for pid, pr, s in current:
        t = s.get("team_id") or ""
        if t not in top or _i(s.get("pts")) > _i(top[t][2].get("pts")):
            top[t] = (pid, pr, s)
    for t, (pid, pr, s) in top.items():
        prev = best_prior.get(t, 0)
        if prev > 0 and _i(s.get("pts")) > prev:
            _record_event(lg, "franchise_record", sy, ("season_pts", t, sy), player_ids=[pid], team_ids=[t],
                          payload={"record": "Most points in a season (in-save)", "value": _i(s.get("pts")),
                                   "previous": prev, "name": pr.get("name")})


def _storyline_hooks(session: Any, lg: Dict[str, Any], sy: int, champ: Optional[str]) -> None:
    """Hand factual history to the existing storyline engine. No effects, just the record;
    the engine's own dedupe applies."""
    try:
        from services.franchise_sim import _record_storyline
    except Exception:
        return
    uid = str(getattr(session, "user_team_id", "") or "")
    for ev in list(lg["events"].values()):
        if ev.get("season") != sy or ev.get("_storyline_sent"):
            continue
        if uid not in ev.get("team_ids", []):
            continue
        headline = _event_headline(session, lg, ev)
        if not headline:
            continue
        ev["_storyline_sent"] = True
        _record_storyline(session, {"type": "legacy", "category": "legacy", "headline": headline,
                                    "team_id": uid, "legacy_event_id": ev["id"], "season_year": sy,
                                    "evidence": [f"legacy:{ev['type']}"], "effects": {}})


def _team_name_at(session: Any, lg: Dict[str, Any], tid: str, version: Optional[int] = None) -> str:
    ident = identity_version_for(lg, tid, version) if version else current_identity(session, tid)
    return str(ident.get("name") or _abbr(_team(session, tid)) if tid else "")


def _event_headline(session: Any, lg: Dict[str, Any], ev: Dict[str, Any]) -> str:
    p = ev.get("payload") or {}
    t = ev.get("type")
    tid0 = (ev.get("team_ids") or [""])[0]
    team = ((ev.get("team_names_then") or {}).get(tid0) or "") if tid0 else ""
    if tid0 and not team:
        # events recorded before the name snapshot existed: the season row knows the version worn
        ver = ((lg["seasons"].get(str(ev.get("season"))) or {}).get("standings", {}).get(tid0) or {}).get("identity_version")
        team = _team_name_at(session, lg, tid0, ver)
    if t == "championship":
        return f"{team} win the {_season_label(int(ev['season']))} championship"
    if t == "first_championship_in_save":
        return f"{team} capture their first championship of this era"
    if t == "first_playoffs_in_save":
        return f"{team} make the playoffs for the first time in this save"
    if t == "career_milestone":
        names = {"gp": "games", "pts": "points", "g": "goals", "a": "assists", "w": "wins", "so": "shutouts"}
        return f"{p.get('name')} reaches {p.get('threshold')} career {names.get(p.get('stat'), p.get('stat'))}"
    if t == "franchise_record":
        return f"{p.get('name')} sets a franchise mark: {p.get('value')} points"
    if t == "retirement":
        return f"{p.get('name')} retires"
    if t == "hall_of_fame":
        return f"{p.get('name')} inducted into the Hall of Fame"
    if t == "rebrand":
        return f"{p.get('to_name')} unveil a new identity"
    if t == "captaincy_start":
        return f"{p.get('name')} named captain of {team}"
    if t == "captaincy_end":
        return f"{p.get('name')}'s captaincy ends ({p.get('reason')})"
    if t == "number_retired":
        return f"{team} retire #{p.get('number')} for {p.get('name')}"
    return ""


# --------------------------------------------------------------------------- backfill (old saves)
def _backfill_from_session_history(session: Any, lg: Dict[str, Any]) -> None:
    """Older saves: rebuild only what ``season_history`` actually recorded (standings,
    champion, awards, draft). Player lines come from ``player_season_archive`` (last six
    seasons). Everything else stays unknown and the season is flagged partial."""
    arch = getattr(session, "player_season_archive", None) or {}
    for h in list(getattr(session, "season_history", None) or []):
        if not isinstance(h, dict):
            continue
        sy = _i(h.get("season_year"))
        if not sy or str(sy) in lg["seasons"]:
            continue
        teams = h.get("teams") if isinstance(h.get("teams"), dict) else {}
        standings = {str(t): {"w": _i(r.get("w")), "l": _i(r.get("l")), "otl": _i(r.get("otl")), "pts": _i(r.get("pts")),
                              "gf": _i(r.get("gf")), "ga": _i(r.get("ga")), "gp": _i(r.get("w")) + _i(r.get("l")) + _i(r.get("otl")),
                              "league_rank": r.get("league_rank"), "identity_version": 1, "made_playoffs": None,
                              "playoff_round_reached": None} for t, r in teams.items() if isinstance(r, dict)}
        lines = arch.get(str(sy)) or {}
        for pid, ln in lines.items():
            if not isinstance(ln, dict):
                continue
            pr = lg["players"].setdefault(str(pid), {"player_id": str(pid), "seasons": [], "teams": [], "championships": [],
                                                     "awards": [], "draft": None, "retired_season": None})
            pr["name"] = pr.get("name") or ln.get("name")
            pr["position"] = pr.get("position") or ln.get("position")
            if not any(s.get("season") == sy for s in pr["seasons"]):
                pr["seasons"].append({"season": sy, "team_id": str(ln.get("team_id") or ""), "source": "simulated", "scope": "regular",
                                      "gp": _i(ln.get("gp")), "g": _i(ln.get("g")), "a": _i(ln.get("a")), "pts": _i(ln.get("pts")),
                                      "plus_minus": _i(ln.get("plus_minus")), "pim": _i(ln.get("pim")), "sog": _i(ln.get("sog")),
                                      "w": _i(ln.get("wins")), "l": _i(ln.get("losses")), "so": _i(ln.get("shutouts")),
                                      "backfilled": True})
        lg["seasons"][str(sy)] = {
            "season": sy, "label": _season_label(sy), "complete": True, "source": "backfill",
            "standings": standings, "champion_id": h.get("champion_id"), "finalist_id": None, "playoff_series": [],
            "playoff_teams": [], "awards": [{"award": a.get("name") or a.get("award_id") or a.get("id"),
                                             "player_id": a.get("winner_id") or a.get("player_id"),
                                             "player_name": a.get("winner_name") or a.get("player_name")}
                                            for a in (h.get("awards") or []) if isinstance(a, dict)],
            "draft": [], "coverage": {"standings": bool(standings), "playoffs": False, "player_lines": bool(lines), "partial": True},
        }
        lg["coverage"]["backfilled_seasons"].append(sy)
        if lg["coverage"]["first_tracked_season"] is None or sy < lg["coverage"]["first_tracked_season"]:
            lg["coverage"]["first_tracked_season"] = sy
    if lg["coverage"]["backfilled_seasons"]:
        lg["coverage"]["notes"].append("Seasons finished before Legacy Hall existed were rebuilt from the save's season "
                                       "history: standings, champions and awards are exact; playoff brackets, trades, "
                                       "captaincy and head-to-head records for those seasons are unknown.")


# --------------------------------------------------------------------------- careers & records
def _career_totals(lines: List[Dict[str, Any]], imported: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    keys = ("gp", "g", "a", "pts", "plus_minus", "pim", "sog", "w", "l", "otl", "so", "ga", "shots_against", "hit", "blk", "ixg")
    for s in lines:
        for k in keys:
            if k in s:
                out[k] = out.get(k, 0) + _i(s.get(k))
    for s in imported or []:
        if not isinstance(s, dict):
            continue
        for k in keys:
            if k == "w":
                raw = s.get("w") if s.get("w") not in (None, "") else s.get("wins")
            elif k == "so":
                raw = s.get("so") if s.get("so") not in (None, "") else s.get("shutouts")
            else:
                raw = s.get(k)
            if raw not in (None, ""):
                out[k] = out.get(k, 0) + _i(raw)
    return out


def player_dossier(session: Any, pid: str, scope: str = "all") -> Dict[str, Any]:
    lg = ensure_legacy(session)
    _import_career_lines(session, lg)
    pr = lg["players"].get(str(pid))
    live = _player_index(session).get(str(pid))
    if pr is None and live is None:
        return {"found": False, "player_id": pid}
    pr = pr or _blank_player(str(pid))
    sim_lines = [s for s in pr.get("seasons") or [] if s.get("source") == "simulated"]
    scope = scope if scope in ("all", "regular", "playoffs", "combined") else "all"

    def _keep(s: Dict[str, Any]) -> bool:
        sc = s.get("scope") or "regular"
        if scope == "regular":
            return sc == "regular" and not s.get("split")
        if scope == "playoffs":
            return sc == "playoffs" and not s.get("split")
        if scope == "combined":
            return not s.get("split")
        return True

    shown = [s for s in sim_lines if _keep(s)]
    counted = [s for s in sim_lines if not s.get("split") and (scope == "all" or _keep(s) or (scope == "combined"))]
    if scope == "regular":
        counted = [s for s in sim_lines if (s.get("scope") or "regular") == "regular" and not s.get("split")]
    elif scope == "playoffs":
        counted = [s for s in sim_lines if s.get("scope") == "playoffs" and not s.get("split")]
    else:
        counted = [s for s in sim_lines if not s.get("split")]
    imported = list(pr.get("imported_seasons") or [])
    by_team: Dict[str, Dict[str, int]] = {}
    for s in sim_lines:
        if s.get("aggregate") or not s.get("team_id"):
            continue
        if scope == "regular" and (s.get("scope") or "regular") != "regular":
            continue
        if scope == "playoffs" and s.get("scope") != "playoffs":
            continue
        t = by_team.setdefault(s.get("team_id") or "", {"seasons": 0})
        t["seasons"] += 1
        for k, v in _career_totals([s]).items():
            t[k] = t.get(k, 0) + v
    caps = [dict(t, team_id=tid) for tid, tenures in lg["captaincy"].items() for t in tenures if t.get("player_id") == str(pid)]
    trades = [t for t in lg["trades"].values() if any(str(m.get("player_id")) == str(pid) for m in t.get("players") or [])]
    events = sorted([e for e in lg["events"].values() if str(pid) in e.get("player_ids", [])], key=lambda e: (e.get("season") or 0))
    team_names = {tid: _team_name_at(session, lg, tid) for tid in by_team if tid}
    has_playoffs = any(s.get("scope") == "playoffs" for s in sim_lines)
    career = _career_totals(counted, imported if scope in ("all", "combined") else None)
    return {
        "found": True, "player_id": str(pid), "name": pr.get("name") or (_pname(live) if live else pid),
        "position": pr.get("position") or (_ppos(live) if live else None), "active": live is not None and not pr.get("retired_season"),
        "retired_season": pr.get("retired_season"), "retirement": pr.get("retirement"),
        "draft": pr.get("draft"), "teams": [{"team_id": t, "name": team_names.get(t, t), **v} for t, v in by_team.items() if t],
        "seasons": shown, "imported_seasons": imported,
        "career_in_save": career, "includes_pre_save_career": bool(imported) and scope in ("all", "combined"),
        "championships": pr.get("championships") or [],
        "awards": pr.get("awards") or [], "captaincy": caps, "trades": trades, "events": events,
        "hall_of_fame": lg["hof"].get(str(pid)), "scope": scope,
        "coverage": {"regular_season": True, "playoffs": has_playoffs,
                     "note": "Career totals add the pre-save NHL line to seasons this save has played. "
                             "A traded player has one line per team plus a season total. "
                             "Playoff lines are stored separately."},
    }


def _all_lines(lg: Dict[str, Any], team_id: Optional[str] = None) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    for pid, pr in lg["players"].items():
        for s in pr.get("seasons") or []:
            if s.get("source") != "simulated":
                continue
            if team_id and s.get("team_id") != team_id:
                continue
            yield pid, pr, s


def leaders(session: Any, *, stat: str = "pts", scope: str = "career", team_id: Optional[str] = None,
            position: Optional[str] = None, limit: int = 25, offset: int = 0) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    stat = stat if stat in ("gp", "g", "a", "pts", "plus_minus", "pim", "sog", "w", "so", "hit", "blk") else "pts"
    scope = scope if scope in ("career", "season", "regular", "playoffs", "combined") else "career"
    cache = lg.setdefault("_cache", {})
    ck = f"leaders|{stat}|{scope}|{team_id}|{position}"
    rows = cache.get(ck)
    if rows is None:
        rows = []

        def _keep(s: Dict[str, Any]) -> bool:
            sc = s.get("scope") or "regular"
            if scope in ("season", "regular") and sc != "regular":
                return False
            if scope == "playoffs" and sc != "playoffs":
                return False
            if scope == "career":
                if team_id:
                    return not s.get("aggregate")
                return not s.get("split")
            if team_id:
                return not s.get("aggregate")
            return not s.get("split")

        listing = scope in ("season", "regular", "playoffs")
        if listing:
            for pid, pr, s in _all_lines(lg, team_id):
                if not _keep(s):
                    continue
                if position and not str(pr.get("position") or "").startswith(position):
                    continue
                if stat in s:
                    rows.append({"player_id": pid, "name": pr.get("name"), "position": pr.get("position"), "season": s["season"],
                                 "team_id": s.get("team_id"), "scope": s.get("scope") or "regular",
                                 "value": _i(s.get(stat)), "gp": _i(s.get("gp"))})
        else:
            agg: Dict[str, Dict[str, Any]] = {}
            for pid, pr, s in _all_lines(lg, team_id):
                if not _keep(s):
                    continue
                if position and not str(pr.get("position") or "").startswith(position):
                    continue
                if stat not in s:
                    continue
                a = agg.setdefault(pid, {"player_id": pid, "name": pr.get("name"), "position": pr.get("position"), "value": 0, "gp": 0,
                                         "seasons": 0, "retired_season": pr.get("retired_season")})
                a["value"] += _i(s.get(stat))
                a["gp"] += _i(s.get("gp"))
                a["seasons"] += 1
            if scope == "career" and not team_id:
                for pid, pr in lg["players"].items():
                    if position and not str(pr.get("position") or "").startswith(position):
                        continue
                    extra = _career_totals([], pr.get("imported_seasons") or [])
                    if not extra.get(stat):
                        continue
                    a = agg.setdefault(pid, {"player_id": pid, "name": pr.get("name"), "position": pr.get("position"), "value": 0, "gp": 0,
                                             "seasons": 0, "retired_season": pr.get("retired_season")})
                    a["value"] += _i(extra.get(stat))
                    a["gp"] += _i(extra.get("gp"))
                    a["includes_pre_save"] = True
            rows = list(agg.values())
        rows.sort(key=lambda r: (-r["value"], r.get("gp", 0), str(r.get("name"))))
        cache[ck] = rows
    return {"stat": stat, "scope": scope, "team_id": team_id, "total": len(rows), "rows": rows[offset: offset + max(1, min(100, limit))],
            "coverage_note": _coverage_note(lg)}


def _coverage_note(lg: Dict[str, Any]) -> str:
    first = lg["coverage"].get("first_tracked_season")
    if first is None:
        return "No completed seasons yet. Records start when the first season is finalized."
    return f"Records since {_season_label(int(first))}, including seasons still in the offseason. Career leaders include a player's pre-save NHL totals once those seasons have been fetched."


# --------------------------------------------------------------------------- franchise summary
def franchise_summary(session: Any, tid: str) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    tid = str(tid)
    seasons = sorted([s for s in lg["seasons"].values() if (s.get("complete") or s.get("stages") or s.get("standings")) and tid in (s.get("standings") or {})], key=lambda s: s["season"])
    rows = []
    for s in seasons:
        st = s["standings"][tid]
        gp = max(1, _i(st.get("gp")))
        rows.append({"season": s["season"], "label": s["label"], "w": st["w"], "l": st["l"], "otl": st["otl"], "pts": st["pts"],
                     "pts_pct": round(st["pts"] / (2 * gp), 3), "gf": st["gf"], "ga": st["ga"], "league_rank": st.get("league_rank"),
                     "made_playoffs": st.get("made_playoffs"), "round": st.get("playoff_round_reached"),
                     "champion": s.get("champion_id") == tid, "finalist": s.get("finalist_id") == tid,
                     "identity_version": st.get("identity_version"), "source": s.get("source")})
    champs = [r["season"] for r in rows if r["champion"]]
    known_po = [r for r in rows if r["made_playoffs"] is not None]
    streak = best_streak = 0
    for r in known_po:
        streak = streak + 1 if r["made_playoffs"] else 0
        best_streak = max(best_streak, streak)
    drought = 0
    for r in reversed(known_po):
        if r["made_playoffs"]:
            break
        drought += 1
    best = max(rows, key=lambda r: (r["pts_pct"], r["gf"] - r["ga"]), default=None)
    worst = min(rows, key=lambda r: (r["pts_pct"], r["gf"] - r["ga"]), default=None)
    caps = list(lg["captaincy"].get(tid) or [])
    longest_cap = max(caps, key=lambda c: _tenure_seasons(session, c), default=None)
    games: Dict[str, Dict[str, Any]] = {}
    for pid, pr, s in _all_lines(lg, tid):
        g = games.setdefault(pid, {"player_id": pid, "name": pr.get("name"), "gp": 0, "pts": 0, "seasons": 0,
                                   "homegrown": bool(pr.get("draft") and str(pr["draft"].get("team_id")) == tid)})
        g["gp"] += _i(s.get("gp"))
        g["pts"] += _i(s.get("pts"))
        g["seasons"] += 1
    most_games = max(games.values(), key=lambda g: g["gp"], default=None)
    homegrown = max([g for g in games.values() if g["homegrown"]], key=lambda g: g["pts"], default=None)
    live_ids = {_pid(p) for p in (getattr(_team(session, tid), "roster", None) or [])}
    longest_active = max([g for g in games.values() if g["player_id"] in live_ids], key=lambda g: (g["seasons"], g["gp"]), default=None)
    events = sorted([e for e in lg["events"].values() if tid in e.get("team_ids", [])], key=lambda e: (e.get("season") or 0, e["id"]))
    return {
        "team_id": tid, "identity": current_identity(session, tid),
        "first_season_in_save": (lg["franchises"].get(tid) or {}).get("first_season_in_save"),
        "seasons": rows, "championships": champs, "finals": [r["season"] for r in rows if r["finalist"]],
        "playoff_appearances": sum(1 for r in known_po if r["made_playoffs"]), "playoff_seasons_known": len(known_po),
        "longest_playoff_streak": best_streak if known_po else None, "current_playoff_drought": drought if known_po else None,
        "best_season": best, "worst_season": worst,
        "most_games": most_games, "top_homegrown": homegrown, "longest_serving_active": longest_active,
        "longest_captain": dict(longest_cap, seasons=_tenure_seasons(session, longest_cap)) if longest_cap else None,
        "recent_events": [dict(e, headline=_event_headline(session, lg, e)) for e in events[-12:]][::-1],
        "coverage_note": _coverage_note(lg),
    }


def _tenure_seasons(session: Any, cap: Optional[Dict[str, Any]]) -> int:
    if not cap:
        return 0
    end = cap.get("end_season") or int(getattr(session, "season_calendar_year", 0) or 0)
    return max(1, int(end) - int(cap.get("start_season") or end) + 1)


# --------------------------------------------------------------------------- eras
def franchise_eras(session: Any, tid: str) -> Dict[str, Any]:
    """Tier each season (contender / playoff / bubble / rebuild) from points% and playoff
    result, then merge runs; a single off season inside a run does not start a new era.
    Captaincy changes, championships and rebrands are listed as overlapping markers."""
    summ = franchise_summary(session, tid)
    rows = summ["seasons"]

    def tier(r: Dict[str, Any]) -> str:
        if r["champion"] or (r["pts_pct"] >= 0.62 and (r["round"] or 0) >= 2):
            return "contender"
        if r["made_playoffs"] or r["pts_pct"] >= 0.56:
            return "playoff"
        if r["pts_pct"] >= 0.47:
            return "bubble"
        return "rebuild"

    tiers = [tier(r) for r in rows]
    for i in range(1, len(tiers) - 1):  # smooth single-season blips
        if tiers[i - 1] == tiers[i + 1] != tiers[i]:
            tiers[i] = tiers[i - 1]
    eras: List[Dict[str, Any]] = []
    for r, t in zip(rows, tiers):
        if eras and eras[-1]["tier"] == t:
            eras[-1]["seasons"].append(r["season"])
        else:
            eras.append({"tier": t, "seasons": [r["season"]]})
    lg = ensure_legacy(session)
    names = {"contender": "Contending era", "playoff": "Playoff era", "bubble": "Bubble years", "rebuild": "Rebuild"}
    for e in eras:
        ss = e["seasons"]
        e["label"] = f"{names[e['tier']]} {_season_label(ss[0])}" + (f" to {_season_label(ss[-1])}" if len(ss) > 1 else "")
        sel = [r for r in rows if r["season"] in ss]
        e["evidence"] = {"seasons": len(ss), "avg_pts_pct": round(sum(r["pts_pct"] for r in sel) / len(sel), 3),
                         "playoffs": sum(1 for r in sel if r["made_playoffs"]), "championships": [r["season"] for r in sel if r["champion"]]}
        e["markers"] = [dict(ev, headline=_event_headline(session, lg, ev)) for ev in lg["events"].values()
                        if tid in ev.get("team_ids", []) and ev.get("season") in ss
                        and ev["type"] in ("championship", "captaincy_start", "captaincy_end", "rebrand")]
    return {"team_id": tid, "eras": eras, "rule": "Season tier from points% and playoff round; single-season blips merged."}


# --------------------------------------------------------------------------- rivalries
def rivalries(session: Any, tid: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """Rivalry score (0-100), bounded and interpretable:
    playoff series x12 + seven-game series x8 + final meetings x15 + one-goal regular
    games x0.6, capped at 100. A pair needs at least one playoff series or 12 one-goal
    games to count as a rivalry at all, so a frequent opponent alone is not one."""
    lg = ensure_legacy(session)
    out = []
    for key, r in lg["h2h"].items():
        a, b = key.split("|")
        if tid and str(tid) not in (a, b):
            continue
        if r.get("series", 0) < 1 and r.get("one_goal", 0) < 12:
            continue
        score = min(100.0, 12 * r.get("series", 0) + 8 * r.get("seven_game", 0) + 15 * r.get("finals", 0) + 0.6 * r.get("one_goal", 0))
        out.append({"teams": [a, b], "names": [_team_name_at(session, lg, a), _team_name_at(session, lg, b)], "score": round(score, 1),
                    "inputs": {k: r.get(k, 0) for k in ("gp", "one_goal", "series", "seven_game", "finals")},
                    "wins": r.get("wins", {}), "series_wins": r.get("series_wins", {}), "seasons": len(r.get("seasons") or [])})
    out.sort(key=lambda x: -x["score"])
    return {"rivalries": out[:limit], "formula": "12/playoff series + 8/seven-game series + 15/final + 0.6/one-goal game, max 100"}


# --------------------------------------------------------------------------- drafts & trades
def draft_history(session: Any, *, season: Optional[int] = None, team_id: Optional[str] = None) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    classes = []
    for k in sorted(lg["seasons"], key=int):
        s = lg["seasons"][k]
        if season and int(k) != int(season):
            continue
        picks = []
        for pk in s.get("draft") or []:
            if team_id and str(pk.get("team_id")) != str(team_id):
                continue
            pr = lg["players"].get(pk.get("player_id") or "") or {}
            lines = [x for x in pr.get("seasons") or [] if x.get("source") == "simulated"]
            tot = _career_totals(lines)
            picks.append({**pk, "nhl_seasons": len(lines), "nhl_gp": tot.get("gp", 0), "nhl_pts": tot.get("pts", 0),
                          "nhl_w": tot.get("w", 0), "retired_season": pr.get("retired_season"),
                          "seasons_since_draft": int(getattr(session, "season_calendar_year", 0) or 0) - int(pk.get("draft_year") or int(k) + 1)})
        if picks:
            n = max(1, len(picks))
            dy = next((p.get("draft_year") for p in picks if p.get("draft_year")), int(k) + 1)
            classes.append({"season": int(k), "draft_year": dy, "label": f"{dy}", "picks": picks,
                            "class_gp": sum(p["nhl_gp"] for p in picks), "class_pts": sum(p["nhl_pts"] for p in picks),
                            "nhl_regulars": sum(1 for p in picks if p["nhl_gp"] >= 200), "size": n})
    return {"classes": classes, "rule": "Outcomes are NHL games and points this save simulated after the draft. A pick with "
                                        "fewer than four seasons since the draft is too early to judge."}


def trade_history(session: Any, *, team_id: Optional[str] = None, limit: int = 30, offset: int = 0) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    _archive_trades(session, lg)  # cheap: only appends unseen trade ids
    rows = sorted(lg["trades"].values(), key=lambda t: (t.get("season") or 0, str(t.get("iso") or "")), reverse=True)
    if team_id:
        rows = [t for t in rows if str(team_id) in t.get("teams", [])]
    page = rows[offset: offset + max(1, min(100, limit))]
    out = []
    for t in page:
        sy = t.get("season") or 0
        players = []
        for m in t.get("players") or []:
            pr = lg["players"].get(str(m.get("player_id") or "")) or {}
            lines = [x for x in pr.get("seasons") or [] if x.get("source") == "simulated"]
            before = _career_totals([x for x in lines if x.get("season", 0) < sy])
            after_lines = [x for x in lines if x.get("season", 0) >= sy and x.get("team_id") == m.get("to_team_id")]
            players.append({**m, "name": m.get("name") or pr.get("name"), "before": before, "after_with_new_team": _career_totals(after_lines),
                            "seasons_after": len(after_lines)})
        out.append({**t, "players": players,
                    "team_names_then": {x: _team_name_at(session, lg, x, (t.get("identity_versions") or {}).get(x)) for x in t.get("teams", [])},
                    "championships_after": {x: [s["season"] for s in lg["seasons"].values() if s.get("champion_id") == x and s["season"] >= sy]
                                            for x in t.get("teams", [])}})
    return {"total": len(rows), "trades": out,
            "note": "Original valuation is what the trade engine recorded on the day. Outcomes are what happened after; "
                    "championships listed after a trade are not a claim the trade caused them."}


# --------------------------------------------------------------------------- captaincy
def leadership_evidence(p: Any, tid: str, lg: Dict[str, Any]) -> Dict[str, Any]:
    ev: Dict[str, Any] = {}
    for key, label in (("per_leadership", "Leadership"), ("per_professionalism", "Professionalism"),
                       ("per_team_chemistry", "Team chemistry"), ("per_emotional_stability", "Composure")):
        v = _rating(p, key)
        if v is not None:
            ev[label] = round(v)
    tr = getattr(p, "traits", None)
    if tr is not None and getattr(tr, "leadership", None) is not None:
        ev["Leadership trait"] = round(float(getattr(tr, "leadership")) * 100)
    pr = lg["players"].get(_pid(p)) or {}
    tenure = [s for s in pr.get("seasons") or [] if s.get("team_id") == tid and s.get("source") == "simulated"]
    ev["Seasons with club (in save)"] = len(tenure)
    ident = getattr(p, "identity", None)
    age = getattr(ident, "age", None) or getattr(p, "age", None)
    if age:
        ev["Age"] = int(age)
    return ev


def _leadership_score(ev: Dict[str, Any]) -> Optional[float]:
    parts = [(ev.get("Leadership"), 0.40), (ev.get("Professionalism"), 0.20), (ev.get("Team chemistry"), 0.15),
             (ev.get("Composure"), 0.10), (ev.get("Leadership trait"), 0.15)]
    have = [(v, w) for v, w in parts if v is not None]
    if not have:
        return None
    base = sum(v * w for v, w in have) / sum(w for _, w in have)
    base += min(6, 1.5 * ev.get("Seasons with club (in save)", 0))
    age = ev.get("Age")
    if age and age < 23:
        base -= 4
    return round(base, 1)


def captain_candidates(session: Any, tid: str, limit: int = 8) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    team = _team(session, tid)
    out = []
    for p in list(getattr(team, "roster", None) or []):
        if getattr(p, "retired", False):
            continue
        ev = leadership_evidence(p, str(tid), lg)
        out.append({"player_id": _pid(p), "name": _pname(p), "position": _ppos(p), "evidence": ev,
                    "score": _leadership_score(ev), "is_captain": bool(getattr(p, "is_captain", False)),
                    "captaincy": getattr(p, "captaincy", None)})
    out.sort(key=lambda c: -(c["score"] if c["score"] is not None else -1))
    return {"candidates": out[:limit],
            "method": "Weighted personality ratings the sim tracks (leadership 40%, professionalism 20%, chemistry 15%, "
                      "composure 10%, leadership trait 15%) plus up to +6 for seasons with the club; -4 under age 23. "
                      "Overall rating is not an input."}


def current_captains(session: Any, tid: str) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    tenures = lg["captaincy"].get(str(tid)) or []
    active = [t for t in tenures if not t.get("end_season") and not t.get("end_iso")]
    cap = next((t for t in active if t.get("role") == "C"), None)
    alts = [t for t in active if t.get("role") == "A"]
    return {"captain": cap, "alternates": alts, "history": [t for t in tenures if t.get("role") == "C"]}


def _set_flags(p: Any, role: Optional[str]) -> None:
    try:
        setattr(p, "is_captain", role == "C")
        setattr(p, "captaincy", role)
    except Exception:
        pass


def appoint_captaincy(session: Any, tid: str, captain_id: Optional[str], alternate_ids: List[str], *,
                      reason: str = "Appointed by GM", source: str = "user") -> Dict[str, Any]:
    lg = ensure_legacy(session)
    tid = str(tid)
    team = _team(session, tid)
    if team is None:
        raise ValueError("Unknown team")
    roster = {_pid(p): p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)}
    alternate_ids = [a for a in dict.fromkeys(str(x) for x in alternate_ids or []) if a and a != str(captain_id or "")]
    if captain_id and str(captain_id) not in roster:
        raise ValueError("Captain must be on the NHL roster")
    if len(alternate_ids) > 3 or any(a not in roster for a in alternate_ids):
        raise ValueError("Up to three alternates, all on the NHL roster")
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    iso = _today_iso(session)
    tenures = lg["captaincy"].setdefault(tid, [])
    want = {"C": [str(captain_id)] if captain_id else [], "A": alternate_ids}
    for t in tenures:
        if t.get("end_season") or t.get("end_iso"):
            continue
        if t["player_id"] not in want.get(t["role"], []):
            t["end_season"], t["end_iso"], t["end_reason"] = sy, iso, "Replaced" if t["role"] == "C" else "Changed"
            p = roster.get(t["player_id"])
            if p is not None:
                _set_flags(p, None)
            if t["role"] == "C":
                _record_event(lg, "captaincy_end", sy, (tid, t["player_id"], t["start_iso"]), team_ids=[tid], player_ids=[t["player_id"]],
                              payload={"name": t.get("name"), "reason": "replaced"}, iso=iso)
    for role, ids in want.items():
        for pid in ids:
            if any(t["player_id"] == pid and t["role"] == role and not t.get("end_season") for t in tenures):
                _set_flags(roster[pid], role)
                continue
            tenures.append({"player_id": pid, "name": _pname(roster[pid]), "role": role, "start_season": sy, "start_iso": iso,
                            "end_season": None, "end_iso": None, "end_reason": None, "reason": reason, "source": source})
            _set_flags(roster[pid], role)
            if role == "C":
                _record_event(lg, "captaincy_start", sy, (tid, pid, iso), team_ids=[tid], player_ids=[pid],
                              payload={"name": _pname(roster[pid]), "source": source}, iso=iso)
    return current_captains(session, tid)


def _today_iso(session: Any) -> Optional[str]:
    cal = getattr(session, "nhl_calendar", None) or []
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    if 0 <= cur < len(cal) and isinstance(cal[cur], dict):
        return cal[cur].get("iso")
    return None


def reconcile_captaincy(session: Any, *, season: Optional[int] = None, appoint_cpu: bool = False) -> None:
    """End tenures whose player left (trade, retirement, release) and, for CPU clubs only,
    name a new captain from the evidence. The user's club is never auto-appointed."""
    lg = ensure_legacy(session)
    sy = int(season or getattr(session, "season_calendar_year", 0) or 0)
    iso = _today_iso(session)
    uid = str(getattr(session, "user_team_id", "") or "")
    idx = None
    for team in _teams(session):
        tid = _tid(team)
        roster_ids = {_pid(p) for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)}
        for t in lg["captaincy"].get(tid) or []:
            if t.get("end_season") or t.get("end_iso") or t["player_id"] in roster_ids:
                continue
            if idx is None:
                idx = _player_index(session)
            p = idx.get(t["player_id"])
            reason = "Retired" if p is None or getattr(p, "retired", False) else "Left the club"
            t["end_season"], t["end_iso"], t["end_reason"] = sy, iso, reason
            if t["role"] == "C":
                _record_event(lg, "captaincy_end", sy, (tid, t["player_id"], t["start_iso"]), team_ids=[tid], player_ids=[t["player_id"]],
                              payload={"name": t.get("name"), "reason": reason.lower()}, iso=iso)
        if appoint_cpu and tid != uid and not current_captains(session, tid)["captain"]:
            cands = [c for c in captain_candidates(session, tid, limit=4)["candidates"] if c["score"] is not None]
            if cands:
                try:
                    appoint_captaincy(session, tid, cands[0]["player_id"], [c["player_id"] for c in cands[1:3]],
                                      reason="Named by the club (leadership evidence)", source="cpu")
                except ValueError:
                    pass


# --------------------------------------------------------------------------- identity / rebrand
def validate_identity(ident: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errs: List[str] = []
    out: Dict[str, Any] = {}
    for k in ("name", "city", "nickname"):
        v = str(ident.get(k) or "").strip()
        if k != "city" and not _NAME.match(v):
            errs.append(f"{k}: 2-32 letters, numbers, spaces or . ' & -")
        if k == "city" and v and not _NAME.match(v):
            errs.append("city: 2-32 letters, numbers, spaces or . ' & -")
        out[k] = v
    ab = str(ident.get("abbreviation") or "").strip().upper()
    if not _ABBR.match(ab):
        errs.append("abbreviation: 2-4 capital letters")
    out["abbreviation"] = ab
    colors = ident.get("colors") or {}
    out["colors"] = {}
    for k in ("primary", "secondary", "accent", "neutral"):
        c = str(colors.get(k) or "")
        if not _HEX.match(c):
            errs.append(f"colors.{k}: #RRGGBB")
        out["colors"][k] = c.upper()
    if not errs and _contrast(out["colors"]["primary"], out["colors"]["secondary"]) < 1.6:
        errs.append("primary and secondary colours are too close to read apart (contrast under 1.6:1)")
    crest = ident.get("crest") or {}
    layers = crest.get("layers") if isinstance(crest.get("layers"), list) else []
    if not 1 <= len(layers) <= 16:
        errs.append("crest: 1 to 16 layers")
    clean_layers = []
    for i, L in enumerate(layers[:16]):
        if not isinstance(L, dict):
            errs.append(f"crest layer {i}: not an object")
            continue
        kind = L.get("kind")
        if kind not in ("shape", "symbol", "text", "stripe"):
            errs.append(f"crest layer {i}: kind must be shape, symbol, text or stripe")
        cl = {"id": str(L.get("id") or f"l{i}")[:24], "kind": kind, "visible": bool(L.get("visible", True))}
        if kind == "shape":
            cl["shape"] = L.get("shape") if L.get("shape") in CREST_SHAPES else "shield"
        if kind == "symbol":
            cl["symbol"] = L.get("symbol") if L.get("symbol") in CREST_SYMBOLS else "star"
        if kind == "text":
            cl["text"] = re.sub(r"[^A-Za-z0-9 .&'-]", "", str(L.get("text") or ""))[:6]
            cl["font"] = L.get("font") if L.get("font") in NUMBER_FONTS else "block"
        for ck in ("fill", "stroke"):
            v = str(L.get(ck) or "none")
            if v != "none" and not _HEX.match(v):
                errs.append(f"crest layer {i}: {ck} must be #RRGGBB or none")
            cl[ck] = v.upper() if v != "none" else "none"
        for nk, lo, hi in (("x", -60, 60), ("y", -60, 60), ("scale", 0.1, 2.5), ("rotation", -180, 180), ("stroke_width", 0, 14)):
            try:
                cl[nk] = max(lo, min(hi, float(L.get(nk, 0 if nk != "scale" else 1))))
            except (TypeError, ValueError):
                errs.append(f"crest layer {i}: {nk} must be a number")
        clean_layers.append(cl)
    out["crest"] = {"shape": crest.get("shape") if crest.get("shape") in CREST_SHAPES else "shield", "layers": clean_layers}
    out["use_custom_crest"] = bool(ident.get("use_custom_crest", True))
    unis = ident.get("uniforms") or {}
    out["uniforms"] = {}
    for kind in ("home", "away", "alternate"):
        u = unis.get(kind)
        if u is None and kind == "alternate":
            out["uniforms"][kind] = None
            continue
        if not isinstance(u, dict):
            errs.append(f"uniforms.{kind}: required")
            continue
        cu: Dict[str, Any] = {}
        for ck in ("body", "sleeve", "shoulder", "stripe_color", "stripe_accent", "collar_color", "number_color", "number_outline", "name_color"):
            v = str(u.get(ck) or "")
            if not _HEX.match(v):
                errs.append(f"uniforms.{kind}.{ck}: #RRGGBB")
            cu[ck] = v.upper()
        cu["stripe"] = u.get("stripe") if u.get("stripe") in JERSEY_STRIPES else "double"
        cu["collar"] = u.get("collar") if u.get("collar") in COLLARS else "v-neck"
        cu["number_font"] = u.get("number_font") if u.get("number_font") in NUMBER_FONTS else "block"
        cu["nameplate"] = u.get("nameplate") if u.get("nameplate") in NAMEPLATES else "arched"
        cu["waist_stripe"] = bool(u.get("waist_stripe", True))
        cu["captain_letter"] = bool(u.get("captain_letter", True))
        for nk, lo, hi, d in (("crest_scale", 0.5, 1.5, 1.0), ("crest_y", -0.3, 0.3, 0.0), ("fabric_roughness", 0.3, 1.0, 0.78)):
            try:
                cu[nk] = max(lo, min(hi, float(u.get(nk, d))))
            except (TypeError, ValueError):
                cu[nk] = d
        if not errs and _contrast(cu["number_color"], cu["body"]) < 2.0:
            errs.append(f"uniforms.{kind}: numbers need more contrast against the jersey (2:1 minimum)")
        out["uniforms"][kind] = cu
    arena = ident.get("arena") or {}
    out["arena"] = {"ice_logo": arena.get("ice_logo") if arena.get("ice_logo") in ("crest", "wordmark", "none") else "crest",
                    "boards_trim": arena.get("boards_trim") if arena.get("boards_trim") in ("primary", "secondary", "accent") else "primary"}
    return out, errs


def _lum(hexc: str) -> float:
    c = [int(hexc[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def rebrand(session: Any, tid: str, proposed: Dict[str, Any], *, reason: str = "", request_id: Optional[str] = None) -> Dict[str, Any]:
    """Atomic: validate everything first, then append one identity version. Nothing on the
    team object, roster, standings or history changes. A retried request_id is a no-op."""
    lg = ensure_legacy(session)
    tid = str(tid)
    fr = lg["franchises"].get(tid)
    if fr is None:
        raise ValueError("Unknown franchise")
    clean, errs = validate_identity(proposed)
    if errs:
        return {"ok": False, "errors": errs}
    if request_id and any(v.get("request_id") == request_id for v in fr["identity_versions"]):
        return {"ok": True, "duplicate": True, "identity": current_identity(session, tid), "version": fr["current_version"]}
    prev = current_identity(session, tid)
    changed = sorted(k for k in clean if clean.get(k) != prev.get(k))
    if not changed:
        return {"ok": True, "unchanged": True, "identity": prev, "version": fr["current_version"]}
    ver = max(int(v.get("version") or 0) for v in fr["identity_versions"]) + 1
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    fr["identity_versions"].append({"version": ver, "effective_season": sy, "identity": clean, "changed": changed,
                                    "reason": str(reason or "Rebrand")[:160], "source": "user", "at": _now_iso(),
                                    "request_id": request_id})
    fr["current_version"] = ver
    eid, _ = _record_event(lg, "rebrand", sy, (tid, ver), team_ids=[tid],
                           payload={"from_name": prev.get("name"), "to_name": clean.get("name"), "changed": changed, "version": ver},
                           iso=_today_iso(session))
    lg.pop("_cache", None)
    try:
        from services.franchise_sim import _record_storyline
        if tid == str(getattr(session, "user_team_id", "")):
            _record_storyline(session, {"type": "legacy", "category": "legacy", "team_id": tid, "season_year": sy,
                                        "headline": f"{clean['name']} unveil a new identity", "legacy_event_id": eid,
                                        "evidence": ["legacy:rebrand"], "effects": {}})
    except Exception:
        log.debug("rebrand storyline skipped", exc_info=True)
    return {"ok": True, "identity": clean, "version": ver, "changed": changed}


# --------------------------------------------------------------------------- philosophy
def set_philosophy(session: Any, tid: str, key: str) -> Dict[str, Any]:
    if key not in PHILOSOPHIES:
        raise ValueError(f"philosophy must be one of {', '.join(PHILOSOPHIES)}")
    lg = ensure_legacy(session)
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    rec = lg["philosophy"].setdefault(str(tid), {"declared": None, "history": []})
    if rec["declared"] != key:
        rec["history"].append({"key": key, "season": sy, "at": _now_iso()})
        rec["declared"] = key
    return philosophy_report(session, tid)


def philosophy_report(session: Any, tid: str) -> Dict[str, Any]:
    """Declared (what the GM says), observed (what the club does, from evidence)."""
    lg = ensure_legacy(session)
    tid = str(tid)
    team = _team(session, tid)
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    _archive_trades(session, lg)
    trades = [t for t in lg["trades"].values() if tid in t.get("teams", [])]
    picks_out = picks_in = young_out = young_in = vets_in = 0
    for t in trades:
        for pk in t.get("picks") or []:
            if str(pk.get("from_team_id")) == tid:
                picks_out += 1
            if str(pk.get("to_team_id")) == tid:
                picks_in += 1
        for m in t.get("players") or []:
            age = _i(m.get("age"))
            if str(m.get("from_team_id")) == tid and age and age <= 22:
                young_out += 1
            if str(m.get("to_team_id")) == tid and age and age <= 22:
                young_in += 1
            if str(m.get("to_team_id")) == tid and age >= 30:
                vets_in += 1
    # Drafted-by comes from the in-save draft record, else the player's own drafted_by_team_id.
    homegrown = sum(1 for p in roster if str((((lg["players"].get(_pid(p)) or {}).get("draft")) or {}).get("team_id")
                                             or getattr(p, "drafted_by_team_id", "") or "") == tid)
    seasons = sorted(int(k) for k, s in lg["seasons"].items() if s.get("complete"))
    continuity = None
    if seasons:
        last = seasons[-1]
        prev_ids = {pid for pid, pr in lg["players"].items() if any(s.get("season") == last and s.get("team_id") == tid for s in pr.get("seasons") or [])}
        if prev_ids:
            continuity = round(sum(1 for p in roster if _pid(p) in prev_ids) / max(1, len(roster)), 2)
    ages = [int(getattr(getattr(p, "identity", None), "age", 0) or getattr(p, "age", 0) or 0) for p in roster]
    ages = [a for a in ages if a]
    evidence = {
        "trades_made": len(trades), "picks_traded_away": picks_out, "picks_acquired": picks_in,
        "young_players_traded_away": young_out, "young_players_acquired": young_in, "veterans_acquired": vets_in,
        "roster_drafted_by_club": homegrown, "roster_returning_share": continuity,
        "roster_avg_age": round(sum(ages) / len(ages), 1) if ages else None,
    }
    tags = []
    if picks_out + young_out >= 4 and picks_out + young_out > 2 * (picks_in + young_in):
        tags.append(("win_now", "Trades futures for the present", f"{picks_out} picks and {young_out} young players out vs {picks_in + young_in} in"))
    if picks_in + young_in >= 4 and picks_in + young_in > 2 * (picks_out + young_out):
        tags.append(("flexibility", "Accumulates future assets", f"{picks_in} picks and {young_in} young players in"))
    if continuity is not None and continuity >= 0.75:
        tags.append(("continuity", "Keeps its core together", f"{int(continuity * 100)}% of the roster returned"))
    if vets_in >= 3:
        tags.append(("veteran_leadership", "Adds veterans", f"{vets_in} players aged 30+ acquired by trade"))
    if homegrown >= 8:
        tags.append(("homegrown", "Built from its own draft", f"{homegrown} roster players were drafted by the club"))
    rec = lg["philosophy"].get(tid) or {}
    declared = rec.get("declared")
    conflict = None
    opposite = {"draft_develop": "win_now", "homegrown": "win_now", "patient_build": "win_now", "win_now": "flexibility",
                "continuity": "win_now", "flexibility": "win_now"}
    if declared and opposite.get(declared) in [t[0] for t in tags]:
        conflict = f"Declared '{PHILOSOPHIES[declared]}', but the record shows '{dict((t[0], t[1]) for t in tags)[opposite[declared]]}'."
    return {"team_id": tid, "declared": declared, "declared_label": PHILOSOPHIES.get(declared or ""), "history": rec.get("history", []),
            "observed": [{"key": k, "label": l, "evidence": e} for k, l, e in tags], "evidence": evidence, "conflict": conflict,
            "options": PHILOSOPHIES,
            "effect_note": "Philosophy is descriptive: it does not change ratings or force trades. CPU clubs are unaffected."}


# --------------------------------------------------------------------------- HOF & retired numbers
def _evaluate_hof(lg: Dict[str, Any], pid: str, sy: int) -> None:
    pr = lg["players"].get(pid) or {}
    if pid in lg["hof"]:
        return
    tot = _career_totals([s for s in pr.get("seasons") or [] if s.get("source") == "simulated"])
    majors = len(pr.get("awards") or [])
    goalie = str(pr.get("position") or "").startswith("G")
    reasons = []
    if goalie and tot.get("w", 0) >= HOF_RULES["goalie_w"]:
        reasons.append(f"{tot['w']} wins")
    if not goalie and tot.get("pts", 0) >= HOF_RULES["skater_pts"] and tot.get("gp", 0) >= HOF_RULES["skater_gp"]:
        reasons.append(f"{tot['pts']} points in {tot['gp']} games")
    if majors >= HOF_RULES["major_awards"]:
        reasons.append(f"{majors} major awards")
    if reasons:
        lg["hof"][pid] = {"player_id": pid, "name": pr.get("name"), "inducted_season": sy, "reasons": reasons, "source": "rule"}
        _record_event(lg, "hall_of_fame", sy, (pid,), player_ids=[pid], payload={"name": pr.get("name"), "reasons": reasons})


def retire_number(session: Any, tid: str, pid: str, number: int) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    tid, pid = str(tid), str(pid)
    if not 0 <= int(number) <= 99:
        raise ValueError("number must be 0-99")
    pr = lg["players"].get(pid)
    if pr is None:
        raise ValueError("Player has no history in this save")
    played = [s for s in pr.get("seasons") or [] if s.get("team_id") == tid]
    if not played:
        raise ValueError("Player never played for this franchise in this save")
    book = lg["retired_numbers"].setdefault(tid, {})
    if str(number) in book:
        raise ValueError(f"#{number} is already retired by this franchise")
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    book[str(number)] = {"player_id": pid, "name": pr.get("name"), "season": sy,
                         "identity_version": int((lg["franchises"].get(tid) or {}).get("current_version") or 1),
                         "seasons_with_club": len(played)}
    _record_event(lg, "number_retired", sy, (tid, number), team_ids=[tid], player_ids=[pid], payload={"number": number, "name": pr.get("name")})
    return {"retired_numbers": book}


# --------------------------------------------------------------------------- league / season views
def season_view(session: Any, season: int) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    s = lg["seasons"].get(str(season))
    if not s:
        return {"found": False, "season": season, "coverage_note": _coverage_note(lg)}
    names = {t: _team_name_at(session, lg, t, r.get("identity_version")) for t, r in (s.get("standings") or {}).items()}
    top = {}
    for stat in ("pts", "g", "w"):
        rows = [(pid, pr, ln) for pid, pr, ln in _all_lines(lg) if ln.get("season") == int(season) and stat in ln]
        rows.sort(key=lambda r: -_i(r[2].get(stat)))
        top[stat] = [{"player_id": pid, "name": pr.get("name"), "team_id": ln.get("team_id"), "value": _i(ln.get(stat))} for pid, pr, ln in rows[:5]]
    return {"found": True, **{k: v for k, v in s.items() if k != "draft"}, "team_names": names, "leaders": top,
            "draft_size": len(s.get("draft") or [])}


def league_overview(session: Any) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    seasons = sorted((s for s in lg["seasons"].values() if s.get("complete") or (s.get("stages") or {}).get("cup") or s.get("champion_id")), key=lambda s: s["season"])
    champs: Dict[str, List[int]] = {}
    for s in seasons:
        if s.get("champion_id"):
            champs.setdefault(s["champion_id"], []).append(s["season"])
    dynasties = []
    for tid, yrs in champs.items():
        yrs = sorted(yrs)
        for i in range(len(yrs)):
            window = [y for y in yrs if yrs[i] <= y < yrs[i] + 5]
            if len(window) >= 3:
                dynasties.append({"team_id": tid, "name": _team_name_at(session, lg, tid), "titles": window})
                break
    return {
        "seasons": [{"season": s["season"], "label": s["label"], "champion_id": s.get("champion_id"),
                     "champion_name": _team_name_at(session, lg, s["champion_id"], (s.get("standings", {}).get(s["champion_id"]) or {}).get("identity_version")) if s.get("champion_id") else None,
                     "finalist_id": s.get("finalist_id"), "source": s.get("source")} for s in seasons],
        "titles_by_team": sorted([{"team_id": t, "name": _team_name_at(session, lg, t), "titles": y} for t, y in champs.items()], key=lambda r: -len(r["titles"])),
        "dynasties": dynasties, "hall_of_fame": sorted(lg["hof"].values(), key=lambda h: h.get("inducted_season") or 0),
        "coverage": lg["coverage"], "coverage_note": _coverage_note(lg),
    }


def timeline(session: Any, *, team_id: Optional[str] = None, ev_type: Optional[str] = None, limit: int = 40, offset: int = 0) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    evs = [e for e in lg["events"].values() if (not team_id or str(team_id) in e.get("team_ids", [])) and (not ev_type or e["type"] == ev_type)]
    evs.sort(key=lambda e: (e.get("season") or 0, e.get("iso") or "", e["id"]), reverse=True)
    page = evs[offset: offset + max(1, min(100, limit))]
    return {"total": len(evs), "events": [dict({k: v for k, v in e.items() if not k.startswith("_")}, headline=_event_headline(session, lg, e)) for e in page]}


# --------------------------------------------------------------------------- trivia
def trivia(session: Any, tid: str, n: int = 6) -> Dict[str, Any]:
    """Facts computed from stored history only; each item carries its source and ids.
    Selection is deterministic per (team, season): stable across renders, rotates yearly."""
    lg = ensure_legacy(session)
    tid = str(tid)
    summ = franchise_summary(session, tid)
    name = summ["identity"].get("name") or tid
    poss = name + ("'" if name.endswith("s") else "'s")
    facts: List[Dict[str, Any]] = []

    def add(key: str, kind: str, text: str, source: str, *, season: Any = None, players: Iterable[str] = (), link: Optional[Dict[str, Any]] = None) -> None:
        facts.append({"id": f"tv_{tid}_{key}", "kind": kind, "text": text, "source": source, "season": season,
                      "player_ids": list(players), "team_ids": [tid], "link": link})

    if summ["best_season"]:
        b = summ["best_season"]
        add("best_season", "A Record That Still Stands", f"{poss} best season in this save is {b['label']}: {b['w']}-{b['l']}-{b['otl']}, {b['pts']} points.",
            "franchise season standings", season=b["season"], link={"type": "season", "season": b["season"]})
    if summ["most_games"]:
        m = summ["most_games"]
        add("most_games", "Franchise Ironman", f"{m['name']} has played {m['gp']} games for {name}, the most in this save.",
            "player season lines", players=[m["player_id"]], link={"type": "player", "player_id": m["player_id"]})
    if summ["top_homegrown"]:
        h = summ["top_homegrown"]
        add("homegrown", "Homegrown", f"{h['name']} is {poss} most productive player drafted by the club: {h['pts']} points.",
            "draft records + season lines", players=[h["player_id"]], link={"type": "player", "player_id": h["player_id"]})
    if summ["longest_captain"]:
        c = summ["longest_captain"]
        add("captain", "Captaincy Legacy", f"{c['name']} has the longest captaincy on record: {c['seasons']} season(s).",
            "captaincy history", players=[c["player_id"]], link={"type": "player", "player_id": c["player_id"]})
    riv = rivalries(session, tid, limit=1)["rivalries"]
    if riv:
        r = riv[0]
        other = r["names"][1] if r["teams"][0] == tid else r["names"][0]
        add("rival", "Rivalry", f"{poss} fiercest rival is {other}: {r['inputs']['series']} playoff series, {r['inputs']['one_goal']} one-goal games.",
            "head-to-head ledger", link={"type": "rivalry", "teams": r["teams"]})
    if summ.get("longest_playoff_streak"):
        add("po_streak", "Playoff Streak", f"Longest playoff streak: {summ['longest_playoff_streak']} straight season(s).", "franchise season standings")
    if summ.get("current_playoff_drought"):
        add("drought", "Drought", f"{name} have missed the playoffs {summ['current_playoff_drought']} straight season(s).", "franchise season standings")
    late = None
    for c in draft_history(session, team_id=tid)["classes"]:
        for pk in c["picks"]:
            if (pk.get("round") or 0) >= 2 and pk["nhl_pts"] > (late["nhl_pts"] if late else 0):
                late = pk
    if late and late["nhl_pts"] > 0:
        add("late_pick", "Draft Class Spotlight", f"Best pick after round one: {late['name']} (round {late['round']}, {late['season']}), {late['nhl_pts']} NHL points.",
            "draft records + season lines", season=late["season"], players=[late["player_id"]], link={"type": "player", "player_id": late["player_id"]})
    rows = summ["seasons"]
    if len(rows) >= 2:
        jumps = [(rows[i]["pts"] - rows[i - 1]["pts"], rows[i]) for i in range(1, len(rows))]
        d, r = max(jumps, key=lambda x: x[0])
        if d > 0:
            add("jump", "The Turning Point", f"Biggest one-year improvement: +{d} points in {r['label']}.", "franchise season standings", season=r["season"])
    if rows and len(rows) >= 2:
        first, last = rows[0], rows[-1]
        add("year_one", "What Changed Since Year One?", f"{first['label']}: {first['pts']} points. {last['label']}: {last['pts']} points.", "franchise season standings")
    tr = trade_history(session, team_id=tid, limit=50)["trades"]
    big = max(tr, key=lambda t: len(t.get("players") or []) + len(t.get("picks") or []), default=None)
    if big and (len(big.get("players") or []) + len(big.get("picks") or [])) >= 3:
        add("big_trade", "A Trade Revisited", f"Largest trade on record: {big.get('headline') or str(len(big['players'])) + ' players, ' + str(len(big['picks'])) + ' picks'}.",
            "trade archive", season=big.get("season"), link={"type": "trade", "trade_id": big["trade_id"]})
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    seed = int(hashlib.sha1(f"{tid}|{sy}".encode()).hexdigest()[:8], 16)
    facts.sort(key=lambda f: int(hashlib.sha1(f"{seed}|{f['id']}".encode()).hexdigest()[:8], 16))
    return {"team_id": tid, "facts": facts[:n], "total_available": len(facts),
            "note": "Every fact is computed from this save's stored history. Facts with no supporting data are omitted."}


# --------------------------------------------------------------------------- lean state hook
def lean_identity_block(session: Any) -> Optional[Dict[str, Any]]:
    """Tiny block for /api/franchise/state: the user club's current presentation only."""
    lg = getattr(session, "legacy", None)
    if not isinstance(lg, dict):
        return None
    uid = str(getattr(session, "user_team_id", "") or "")
    fr = (lg.get("franchises") or {}).get(uid)
    if not fr:
        return None
    ident = current_identity(session, uid)
    return {"team_id": uid, "version": fr.get("current_version"), "name": ident.get("name"), "abbreviation": ident.get("abbreviation"),
            "colors": ident.get("colors"), "rebranded": int(fr.get("current_version") or 1) > 1}


def identity_payload(session: Any, tid: str) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    tid = str(tid)
    reconcile_captaincy(session)
    fr = lg["franchises"].get(tid) or {}
    summ = franchise_summary(session, tid)
    return {
        "team_id": tid, "is_user_team": tid == str(getattr(session, "user_team_id", "")),
        "identity": current_identity(session, tid), "current_version": fr.get("current_version"),
        "versions": [{k: v for k, v in ver.items() if k != "request_id"} for ver in fr.get("identity_versions") or []],
        "original_identity": fr.get("original_identity"), "first_season_in_save": fr.get("first_season_in_save"),
        "captains": current_captains(session, tid), "philosophy": philosophy_report(session, tid),
        "summary": {k: summ[k] for k in ("championships", "finals", "playoff_appearances", "playoff_seasons_known", "longest_playoff_streak",
                                         "current_playoff_drought", "best_season", "most_games", "top_homegrown", "longest_serving_active",
                                         "longest_captain", "recent_events", "coverage_note")},
        "eras": franchise_eras(session, tid)["eras"][-4:], "retired_numbers": lg["retired_numbers"].get(tid, {}),
        "presets": list((lg["presets"].get(tid) or {}).values()),
        "options": {"crest_shapes": CREST_SHAPES, "crest_symbols": CREST_SYMBOLS, "stripes": JERSEY_STRIPES,
                    "number_fonts": NUMBER_FONTS, "nameplates": NAMEPLATES, "collars": COLLARS},
    }


def save_preset(session: Any, tid: str, name: str, uniform: Dict[str, Any]) -> Dict[str, Any]:
    lg = ensure_legacy(session)
    probe = dict(current_identity(session, tid))
    probe["uniforms"] = {"home": uniform, "away": uniform, "alternate": None}
    clean, errs = validate_identity(probe)
    errs = [e for e in errs if e.startswith("uniforms")]
    if errs:
        return {"ok": False, "errors": errs}
    nm = re.sub(r"[^A-Za-z0-9 .'-]", "", str(name or "Preset"))[:28] or "Preset"
    book = lg["presets"].setdefault(str(tid), {})
    if len(book) >= 24 and nm not in book:
        return {"ok": False, "errors": ["preset limit is 24"]}
    book[nm] = {"name": nm, "uniform": clean["uniforms"]["home"], "at": _now_iso()}
    return {"ok": True, "presets": list(book.values())}
