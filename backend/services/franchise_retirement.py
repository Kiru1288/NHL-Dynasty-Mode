"""
Franchise-mode NHL retirement pass — separate from OVR/development progression.

Runs at Final Skate (offseason retirements stage). Development Camp handles progression later.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from services.franchise_session import FranchiseSession
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

# NHL yearly retirement targets
NHL_RETIREMENT_FLOOR = 10
NHL_RETIREMENT_SOFT_CAP = 35
NHL_RETIREMENT_HARD_CAP = 35

# Depth / non-NHL pools
DEPTH_RETIREMENT_FLOOR = 25
DEPTH_RETIREMENT_SOFT_CAP = 80
DEPTH_RETIREMENT_HARD_CAP = 80

RETIREMENT_STATUSES = (
    "safe",
    "monitoring",
    "considering",
    "likely_retiring",
    "confirmed",
    "returning_for_one_more_year",
)

RETIREMENT_TYPES = (
    "age_retirement",
    "injury_retirement",
    "decline_retirement",
    "cup_win_walkaway",
    "no_contract_retirement",
    "role_frustration_retirement",
    "morale_retirement",
    "depth_retirement",
    "legend_farewell",
    "goalie_longevity_retirement",
    "family_retirement",
    "free_agent_walkaway",
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _player_id(player: Any) -> str:
    return str(getattr(player, "player_id", getattr(player, "id", "")) or "")


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    if ident is not None and hasattr(ident, "age"):
        try:
            return int(getattr(ident, "age", 0) or 0)
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(player, "age", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(getattr(ident, "name", getattr(player, "name", "Player")) or "Player")


def _player_position(player: Any) -> str:
    pos = getattr(player, "position", None)
    if pos is not None and hasattr(pos, "value"):
        return str(pos.value or "")
    ident = getattr(player, "identity", None)
    if ident is not None:
        p2 = getattr(ident, "position", None)
        if p2 is not None and hasattr(p2, "value"):
            return str(p2.value or "")
        return str(p2 or "")
    return str(pos or "")


def _is_goalie(player: Any) -> bool:
    p = _player_position(player).upper()
    return p in ("G", "GOALIE", "GOALTENDER")


def _player_ovr_norm(player: Any) -> float:
    ovr_fn = getattr(player, "ovr", None)
    try:
        ov = float(ovr_fn() if callable(ovr_fn) else ovr_fn)
    except Exception:
        ov = float(getattr(player, "overall", 0) or 0)
    if ov > 1.5:
        ov = ov / 99.0
    return _clamp(ov, 0.0, 1.0)


def _player_ovr_display(player: Any) -> float:
    ov = _player_ovr_norm(player)
    return round(ov * 99.0, 1) if ov <= 1.0 else round(ov, 1)


def _injury_wear(player: Any) -> float:
    health = getattr(player, "health", None)
    wear = float(getattr(health, "wear_and_tear", 0.0) or 0.0) if health else 0.0
    hist = getattr(health, "injury_history", None) if health else None
    if isinstance(hist, list) and hist:
        # Injury count is only a proxy: a pile of minor knocks must never read as a
        # career-ending injury (that retired healthy mid-20s stars).
        wear = max(wear, min(0.6, 0.05 * len(hist)))
    return _clamp(wear)


def _true_career_ending(player: Any) -> bool:
    """Real career-ending health state (the tracked wear value, not the injury-count proxy)."""
    health = getattr(player, "health", None)
    try:
        wear = float(getattr(health, "wear_and_tear", 0.0) or 0.0) if health else 0.0
    except (TypeError, ValueError):
        wear = 0.0
    return wear >= 0.92 or bool(getattr(player, "career_ending_injury", False))


def _player_morale(player: Any) -> float:
    psych = getattr(player, "psych", None)
    return float(getattr(psych, "morale", getattr(player, "morale", 0.6)) or 0.6)


def _psych_value(player: Any, key: str, default: float = 0.5) -> float:
    psych = getattr(player, "psych", None)
    try:
        v = getattr(psych, key, None) if psych is not None else None
        return float(default if v is None else v)
    except (TypeError, ValueError):
        return float(default)


def _true_wear(player: Any) -> float:
    health = getattr(player, "health", None)
    try:
        return _clamp(float(getattr(health, "wear_and_tear", 0.0) or 0.0)) if health else 0.0
    except (TypeError, ValueError):
        return 0.0


def _injury_count(player: Any) -> int:
    health = getattr(player, "health", None)
    hist = getattr(health, "injury_history", None) if health else None
    return len(hist) if isinstance(hist, list) else 0


def _rating_peak(player: Any) -> Tuple[Optional[int], Optional[int]]:
    """Highest OVR on the rating history (this save) as (ovr, season)."""
    best: Optional[Tuple[int, Optional[int]]] = None
    for pt in getattr(player, "rating_history", None) or []:
        if not isinstance(pt, dict):
            continue
        ov = _num(pt.get("ovr"))
        if ov is None:
            continue
        season = pt.get("season")
        try:
            season = int(season) if season is not None else None
        except (TypeError, ValueError):
            season = None
        if best is None or ov > best[0]:
            best = (int(round(ov)), season)
    pre = _num(getattr(player, "_pre_retirement_ovr", None))
    if pre is not None and (best is None or pre > best[0]):
        best = (int(round(pre)), None)
    return best if best is not None else (None, None)


def _award_list(player: Any) -> List[Dict[str, Any]]:
    """Trophies: real NHL ones (cached at import, never fetched here) + ones won in the save."""
    out: List[Dict[str, Any]] = []
    seen = set()
    if getattr(player, "nhl_awards_fetched", False):
        for a in getattr(player, "nhl_awards", None) or []:
            if isinstance(a, dict) and a.get("award_name"):
                key = (str(a.get("award_name")), str(a.get("season") or ""))
                if key not in seen:
                    seen.add(key)
                    out.append({"award_name": str(a.get("award_name")), "season": a.get("season"), "source": "nhl"})
    for a in getattr(player, "career_awards", None) or getattr(player, "awards_won", None) or []:
        if isinstance(a, dict):
            name = a.get("award_name") or a.get("name") or a.get("award_id")
            if not name:
                continue
            key = (str(name), str(a.get("season") or ""))
            if key not in seen:
                seen.add(key)
                out.append({"award_name": str(name), "award_id": a.get("award_id"), "season": a.get("season"), "source": "save"})
        elif isinstance(a, str) and a.strip():
            out.append({"award_name": a.strip(), "season": None, "source": "save"})
    return out


def _cup_wins(player: Any) -> int:
    n = int(getattr(player, "cup_wins", 0) or 0)
    try:
        counted = sum(
            1 for a in _award_list(player)
            if "stanley cup" in str(a.get("award_name") or "").lower() or str(a.get("award_id") or "") == "stanley"
        )
    except Exception:
        counted = 0
    return max(n, counted)


def _season_stats(session: FranchiseSession, player: Any) -> Dict[str, Any]:
    pid = _player_id(player)
    if not pid:
        return {}
    raw = (getattr(session, "player_season_stats", None) or {}).get(pid) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _num(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool) or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _season_label(year: int) -> str:
    return f"{int(year)}-{(int(year) + 1) % 100:02d}"


def _season_start_year(label: Any) -> int:
    s = str(label or "").strip()
    try:
        return int(s[:4])
    except (TypeError, ValueError):
        return 0


def _team_abbrev(team: Any) -> str:
    if team is None:
        return ""
    for attr in ("abbr", "abbreviation", "code", "short_name"):
        v = getattr(team, attr, None)
        if v is not None and str(v).strip():
            s = "".join(c for c in str(v).upper() if c.isalnum())
            if 2 <= len(s) <= 4:
                return s
    return ""


def _nhl_career_summary(player: Any, session: FranchiseSession, team: Any = None) -> Dict[str, Any]:
    """Real NHL career totals: imported NHL seasons + seasons simmed in this save.

    The old reader looked for top-level gp/g/a keys that career_stats never has, so every
    retiree showed only the final season ("82 GP"). Seasons with a club split carry a TOT
    row; we use it alone so traded seasons are not counted twice. The season that just
    finished is folded in from the live ledger when it has not been archived yet.
    """
    goalie = _is_goalie(player)
    cs = getattr(player, "career_stats", None)
    seasons = cs.get("seasons") if isinstance(cs, dict) else None
    rows = [
        s for s in (seasons if isinstance(seasons, list) else [])
        if isinstance(s, dict) and str(s.get("league") or "NHL").strip().upper() in ("", "NHL")
    ]
    by_season: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for s in rows:
        lbl = str(s.get("season") or "")
        if lbl not in by_season:
            by_season[lbl] = []
            order.append(lbl)
        by_season[lbl].append(s)

    tot = {"gp": 0, "g": 0, "a": 0, "pts": 0, "pm": 0, "pim": 0, "w": 0, "l": 0, "otl": 0, "so": 0}
    sv_w = 0.0
    sv_gp = 0
    gaa_w = 0.0
    gaa_gp = 0
    teams: List[str] = []
    season_lines: List[Dict[str, Any]] = []

    def _add(line: Dict[str, Any], label: str, team_label: str) -> None:
        nonlocal sv_w, sv_gp, gaa_w, gaa_gp
        gp = int(_num(line.get("gp")) or 0)
        if gp <= 0:
            return
        tot["gp"] += gp
        if goalie:
            tot["w"] += int(_num(line.get("wins", line.get("w"))) or 0)
            tot["l"] += int(_num(line.get("losses", line.get("l"))) or 0)
            tot["otl"] += int(_num(line.get("otl")) or 0)
            tot["so"] += int(_num(line.get("shutouts", line.get("so"))) or 0)
            sv = _num(line.get("sv_pct", line.get("svPct", line.get("save_pct"))))
            if sv is not None and sv > 0:
                sv = sv / 100.0 if sv > 1.5 else sv
                sv_w += sv * gp
                sv_gp += gp
            gaa = _num(line.get("gaa"))
            if gaa is not None and gaa > 0:
                gaa_w += gaa * gp
                gaa_gp += gp
        else:
            g = int(_num(line.get("g", line.get("goals"))) or 0)
            a = int(_num(line.get("a", line.get("assists"))) or 0)
            tot["g"] += g
            tot["a"] += a
            tot["pts"] += int(_num(line.get("pts", line.get("points"))) or (g + a))
            tot["pm"] += int(_num(line.get("plus_minus", line.get("plusMinus", line.get("pm")))) or 0)
            tot["pim"] += int(_num(line.get("pim")) or 0)
        season_lines.append({"season": label, "team": team_label, "gp": gp})

    for lbl in order:
        group = by_season[lbl]
        tot_rows = [s for s in group if str(s.get("team") or s.get("team_abbrev") or "").upper() == "TOT"]
        clubs = [s for s in group if s not in tot_rows]
        for s in clubs:
            t = str(s.get("team_abbrev") or s.get("team") or "").strip()
            if t and t != "—" and t not in teams:
                teams.append(t)
        if tot_rows:
            _add(tot_rows[0], lbl, "/".join(str(s.get("team_abbrev") or s.get("team") or "") for s in clubs) or "TOT")
        else:
            for s in clubs:
                _add(s, lbl, str(s.get("team_abbrev") or s.get("team") or ""))

    # The just-finished season, if the rollover archive has not written it yet.
    try:
        cur_year = int(getattr(session, "season_calendar_year", 0) or 0)
    except (TypeError, ValueError):
        cur_year = 0
    cur_lbl = _season_label(cur_year) if cur_year else ""
    ss = _season_stats(session, player)
    if cur_lbl and cur_lbl not in by_season and int(_num(ss.get("gp")) or 0) > 0:
        abbr = _team_abbrev(team)
        line = dict(ss)
        if goalie:
            sa = _num(ss.get("sa", ss.get("shots_against")))
            saves = _num(ss.get("saves", ss.get("sv")))
            if sa and saves is not None:
                line["sv_pct"] = saves / sa
            toi = _num(ss.get("toi_sec"))
            ga = _num(ss.get("ga"))
            gp_ = int(_num(ss.get("gp")) or 0)
            if ga is not None and gp_ > 0:
                toi = toi if toi and toi >= gp_ * 1800 else gp_ * 3600
                line["gaa"] = ga * 3600.0 / toi
        _add(line, cur_lbl, abbr)
        if abbr and abbr not in teams:
            teams.append(abbr)
        order.append(cur_lbl)

    labelled = [l for l in order if l]
    starts = [y for y in (_season_start_year(l) for l in labelled) if y > 0]
    out: Dict[str, Any] = {
        "gp": tot["gp"],
        "seasons": len(labelled),
        "first_season": _season_label(min(starts)) if starts else None,
        "last_season": _season_label(max(starts)) if starts else None,
        "teams": teams,
    }
    if goalie:
        out.update({
            "w": tot["w"], "l": tot["l"], "otl": tot["otl"], "so": tot["so"],
            "sv_pct": round(sv_w / sv_gp, 3) if sv_gp else None,
            "gaa": round(gaa_w / gaa_gp, 2) if gaa_gp else None,
        })
    else:
        out.update({"g": tot["g"], "a": tot["a"], "pts": tot["pts"], "pm": tot["pm"], "pim": tot["pim"]})
    # Best season by points (skaters) or wins (goalies) for the tribute line.
    best = None
    for lbl in labelled:
        group = by_season.get(lbl) or []
        cand = [s for s in group if str(s.get("team") or "").upper() == "TOT"] or group
        for s in cand:
            score = _num(s.get("wins") if goalie else s.get("pts", s.get("points"))) or 0
            if best is None or score > best[0]:
                best = (score, lbl, s)
    if best is not None and best[0] > 0:
        s = best[2]
        bl: Dict[str, Any] = {"season": best[1], "team": str(s.get("team_abbrev") or s.get("team") or ""), "gp": int(_num(s.get("gp")) or 0)}
        if goalie:
            bl.update({"w": int(best[0]), "sv_pct": _num(s.get("sv_pct")), "gaa": _num(s.get("gaa"))})
        else:
            bl.update({"g": int(_num(s.get("g")) or 0), "a": int(_num(s.get("a")) or 0), "pts": int(best[0])})
        out["best_season"] = bl
    return out


def _career_stats(player: Any, session: FranchiseSession, team: Any = None) -> Dict[str, Any]:
    ss = _season_stats(session, player)
    career = getattr(player, "career_stats", None) or getattr(player, "stats", None) or {}
    if not isinstance(career, dict):
        career = {}
    summary = _nhl_career_summary(player, session, team)
    if int(summary.get("gp") or 0) > 0:
        return {
            "games_played": int(summary.get("gp") or 0),
            "goals": int(summary.get("g") or 0),
            "assists": int(summary.get("a") or 0),
            "points": int(summary.get("pts") or 0),
            "goalie_wins": int(summary.get("w") or 0),
            "shutouts": int(summary.get("so") or 0),
            "seasons_played": max(1, int(summary.get("seasons") or 0)),
            "summary": summary,
        }

    def _as_int(val: Any, default: int = 0) -> int:
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            try:
                return int(float(val.strip()))
            except Exception:
                return default
        return default

    seasons_raw = getattr(player, "seasons_played", None)
    if seasons_raw is None:
        seasons_raw = career.get("seasons_played", career.get("seasons_count", None))
    # career["seasons"] is often a list of season rows — never pass that to int().
    if isinstance(seasons_raw, list):
        seasons_played = max(1, len(seasons_raw))
    else:
        seasons_played = _as_int(seasons_raw, 0)
        if seasons_played <= 0:
            seasons_played = max(1, _player_age(player) - 18)

    gp = _as_int(career.get("gp", career.get("games", ss.get("gp", 0))), 0)
    g = _as_int(career.get("g", career.get("goals", ss.get("g", 0))), 0)
    a = _as_int(career.get("a", career.get("assists", ss.get("a", 0))), 0)
    pts = _as_int(career.get("pts", career.get("points", g + a)), g + a)
    return {
        "games_played": gp,
        "goals": g,
        "assists": a,
        "points": pts,
        "goalie_wins": _as_int(career.get("w", ss.get("w", 0)), 0),
        "shutouts": _as_int(career.get("so", ss.get("so", 0)), 0),
        "seasons_played": seasons_played,
        "summary": None,
    }


def _attr_dict(val: Any) -> Dict[str, Any]:
    """Coerce dict-like or dataclass attrs without dict(obj) on non-iterables."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return dict(val)
    if hasattr(val, "__dict__"):
        return dict(val.__dict__)
    return {}


def _life_pressure_dict(player: Any) -> Dict[str, float]:
    return {str(k): float(v) for k, v in _attr_dict(getattr(player, "life_pressure", None)).items()}


def _draft_info_dict(player: Any) -> Dict[str, Any]:
    raw = getattr(player, "draft_info", None) or getattr(player, "draft", None)
    if raw is not None:
        return _attr_dict(raw)
    ident = getattr(player, "identity", None)
    if ident is None:
        return {}
    out: Dict[str, Any] = {}
    for key in ("draft_year", "draft_round", "draft_pick"):
        if hasattr(ident, key):
            out[key] = getattr(ident, key)
    return out


def _build_retirement_adapter(player: Any) -> Any:
    """Map franchise player objects into RetirementEngine-friendly shape."""
    psych = getattr(player, "psych", None)
    traits = getattr(player, "traits", None)
    personality = getattr(player, "personality", None) or traits
    if personality is None and psych is not None:
        personality = psych

    return SimpleNamespace(
        age=float(_player_age(player)),
        morale=_player_morale(player),
        injury_wear=_injury_wear(player),
        durability=float(getattr(getattr(player, "health", None), "durability", 0.6) or 0.6),
        personality=personality,
        life_pressure=_life_pressure_dict(player),
    )


def _build_retirement_context(
    session: FranchiseSession,
    player: Any,
    team: Any,
    *,
    unsigned: bool = False,
) -> Dict[str, Any]:
    from services.franchise_sim import _contract_years_remaining

    ovr = _player_ovr_norm(player)
    yrs = _contract_years_remaining(player)
    wear = _injury_wear(player)
    morale = _player_morale(player)
    champ = str(getattr(session, "champion_id", "") or getattr(session, "stanley_cup_winner", "") or "")
    tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
    won_cup_this_year = bool(champ and tid and champ == tid)

    # Mental fatigue used to be 1 - morale (+ wear): with the default 0.5 morale every
    # veteran started at 0.5 "fatigue", so the engine labelled ordinary age retirements
    # "burnout". Use the tracked psych fatigue plus only genuinely low morale.
    tracked_fatigue = _psych_value(player, "mental_fatigue", 0.0)
    pressure_fatigue = _psych_value(player, "pressure_fatigue", 0.0)
    mental_fatigue = _clamp(
        max(tracked_fatigue, pressure_fatigue * 0.8) + max(0.0, 0.42 - morale) * 1.2 + wear * 0.2
    )

    return {
        "ovr": ovr,
        "no_offers": unsigned or yrs <= 0,
        "career_ending_injury": wear >= 0.92,
        "recent_major_injury": wear >= 0.65,
        "healthy_scratches": int(getattr(player, "healthy_scratches", 0) or 0),
        "mental_fatigue": mental_fatigue,
        "usage_heavy": float(getattr(player, "toi_share", 0.0) or 0.0),
        "won_cup_this_year": won_cup_this_year,
        "contract_years_remaining": yrs,
    }


def _evaluate_with_engine(session: FranchiseSession, player: Any, team: Any, rng: Any) -> Tuple[Any, float]:
    """Returns (RetirementDecision|None, fallback_chance)."""
    try:
        from app.sim_engine.ai.retirement_engine import RetirementEngine
    except Exception:
        return None, 0.0

    seed = int(__import__("hashlib").sha256(("%s|%s" % (_player_id(player), int(session.season_calendar_year))).encode()).hexdigest()[:12], 16) % (2**31)
    engine = RetirementEngine(seed=seed)
    engine.rng = rng
    adapter = _build_retirement_adapter(player)
    ctx = _build_retirement_context(session, player, team)
    try:
        decision = engine.evaluate_player(adapter, ctx)
        return decision, float(getattr(decision, "retire_chance", 0.0) or 0.0)
    except Exception:
        return None, 0.0


def _fallback_retire_chance(player: Any, rng: Any) -> bool:
    try:
        from app.sim_engine.progression.retirement import should_player_retire

        return bool(should_player_retire(player, rng))
    except Exception:
        age = _player_age(player)
        if age < 35:
            return False
        base = {35: 0.08, 37: 0.25, 39: 0.55, 41: 0.85}.get(
            next((k for k in (41, 39, 37, 35) if age >= k), 35), 0.08
        )
        return rng.random() < base


def _compute_retirement_risk(
    player: Any,
    team: Any,
    session: FranchiseSession,
    engine_chance: float,
    factors: Optional[Any] = None,
) -> int:
    age = _player_age(player)
    ovr = _player_ovr_norm(player)
    wear = _injury_wear(player)
    morale = _player_morale(player)
    from services.franchise_sim import _contract_years_remaining

    yrs = _contract_years_remaining(player)
    cups = _cup_wins(player)

    risk = engine_chance * 100.0
    if age >= 41:
        risk = max(risk, 82.0)
    elif age >= 39:
        risk = max(risk, 62.0)
    elif age >= 37:
        risk = max(risk, 38.0)
    elif age >= 35:
        risk = max(risk, 18.0)

    if ovr < 0.55:
        risk += 14.0
    elif ovr < 0.65:
        risk += 6.0
    if wear > 0.5:
        risk += wear * 18.0
    if morale < 0.42:
        risk += (0.42 - morale) * 40.0
    if yrs <= 0 and age >= 32:
        risk += 16.0
    if cups >= 1 and age >= 36:
        risk += 4.0
    if _is_goalie(player) and age >= 36:
        risk -= 8.0

    if factors is not None:
        try:
            risk += float(getattr(factors, "injury_pressure", 0) or 0) * 12.0
            risk += float(getattr(factors, "morale_pressure", 0) or 0) * 10.0
            risk -= float(getattr(factors, "legacy_resistance", 0) or 0) * 8.0
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    return int(_clamp(risk, 0.0, 100.0) * 1.0)


def _risk_to_status(risk: int, *, confirmed: bool = False, returning: bool = False) -> str:
    if returning:
        return "returning_for_one_more_year"
    if confirmed:
        return "confirmed"
    if risk >= 78:
        return "likely_retiring"
    if risk >= 52:
        return "considering"
    if risk >= 28:
        return "monitoring"
    return "safe"


def _retirement_evidence(
    player: Any,
    session: FranchiseSession,
    team: Any,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """Real signals a retirement reason may lean on (no randomness)."""
    from services.franchise_sim import _contract_years_remaining

    ss = _season_stats(session, player)
    peak, peak_season = _rating_peak(player)
    ovr_disp = _player_ovr_display(player)
    life = _life_pressure_dict(player)
    summary = _nhl_career_summary(player, session, team)
    return {
        "age": _player_age(player),
        "ovr": _player_ovr_norm(player),
        "ovr_disp": ovr_disp,
        "peak": peak,
        "peak_season": peak_season,
        "drop": (float(peak) - ovr_disp) if peak is not None else 0.0,
        "morale": _player_morale(player),
        "fatigue": _psych_value(player, "mental_fatigue", 0.0),
        "role_sat": min(_psych_value(player, "role_satisfaction", 0.5), _psych_value(player, "ice_time_satisfaction", 0.5)),
        "life_family": float(life.get("family", 0.0) or 0.0),
        "life_psych": float(life.get("psychological", 0.0) or 0.0),
        "wear": _true_wear(player),
        "injuries": _injury_count(player),
        "yrs": _contract_years_remaining(player) if getattr(player, "contract", None) is not None else 0,
        "final_gp": int(_num(ss.get("gp")) or 0),
        "scratches": int(getattr(player, "healthy_scratches", 0) or 0),
        "cups": _cup_wins(player),
        "won_cup": bool(ctx.get("won_cup_this_year")),
        "career_gp": int(summary.get("gp") or 0),
        "career_pts": int(summary.get("pts") or 0),
        "seasons": int(summary.get("seasons") or 0),
        "goalie": _is_goalie(player),
        "career_ending": _true_career_ending(player),
    }


def _sustained_burnout(ev: Dict[str, Any]) -> bool:
    """Mental-health retirement needs real, sustained evidence — never the engine label alone.

    Low morale *and* tracked fatigue/psych strain, a veteran age, and no elite player or
    long remaining deal unless the evidence is overwhelming.
    """
    morale = float(ev["morale"])
    strain = max(float(ev["fatigue"]), float(ev["life_psych"]))
    if ev["age"] < 33 or morale >= 0.30 or strain < 0.55:
        return False
    overwhelming = morale < 0.18 and strain >= 0.75
    if ev["ovr"] >= 0.80 and not overwhelming:
        return False
    if ev["yrs"] >= 2 and not overwhelming:
        return False
    return True


def _classify_retirement_type(
    player: Any,
    session: FranchiseSession,
    team: Any,
    *,
    primary_reason: str,
    risk: int,
    ctx: Dict[str, Any],
) -> str:
    ev = _retirement_evidence(player, session, team, ctx)
    age = ev["age"]
    ovr = ev["ovr"]
    pr = str(primary_reason or "").lower()

    # Injuries: the real wear value or a flagged career-ender (not the injury-count proxy).
    if ev["career_ending"] or ev["wear"] >= 0.75 or ("injur" in pr and ev["wear"] >= 0.55):
        return "injury_retirement"
    if ev["won_cup"] and age >= 34:
        return "cup_win_walkaway"
    if age >= 37 and (ovr >= 0.84 or ev["career_pts"] >= 900 or ev["cups"] >= 2):
        return "legend_farewell"
    if ev["goalie"] and age >= 37 and ovr >= 0.72:
        return "goalie_longevity_retirement"
    if _sustained_burnout(ev):
        return "morale_retirement"
    if age >= 33 and (ev["life_family"] >= 0.55 or ("family" in pr and ev["life_family"] >= 0.40)):
        return "family_retirement"
    if ev["yrs"] <= 0 and age >= 33:
        return "no_contract_retirement" if ovr < 0.76 else "free_agent_walkaway"
    if age >= 33 and ev["final_gp"] and (ev["final_gp"] < 40 or ev["scratches"] >= 15) and ovr < 0.80:
        return "role_frustration_retirement"
    if age >= 33 and (ev["drop"] >= 4 or (ovr < 0.58 and age >= 34)):
        return "decline_retirement"
    if ovr < 0.62 and age >= 33:
        return "depth_retirement"
    return "age_retirement"


def _pick(options: List[str], player: Any, salt: str) -> str:
    """Deterministic variety: same player + season always reads the same."""
    import hashlib

    key = "%s|%s" % (_player_id(player), salt)
    idx = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % max(1, len(options))
    return options[idx]


def _readable_retirement_reason(retirement_type: str, player: Any, ctx: Dict[str, Any]) -> str:
    age = _player_age(player)
    yrs = int(ctx.get("contract_years_remaining") or 0)
    mapping = {
        "age_retirement": [f"Calls it a career at {age}", "Ready to step away from the game", f"Hangs up the skates at {age}"],
        "injury_retirement": ["Body can no longer handle the grind", "Injuries force him out", "Medical advice ends his career"],
        "decline_retirement": ["Game has slipped with age", "Declining play at the end of the road", "Steps away as his game fades"],
        "cup_win_walkaway": ["Goes out a Stanley Cup champion", "Retires on top after the Cup", "Leaves with a ring as his final act"],
        "no_contract_retirement": ["No NHL offers as a free agent", "Unsigned and out of options", "Market dried up in free agency"],
        "free_agent_walkaway": ["Turns down offers to retire", "Chooses retirement over a new contract", "Walks away instead of re-signing"],
        "role_frustration_retirement": ["Lost his spot in the lineup", "Shrinking role pushes him out", "Not ready to be a press-box regular"],
        "morale_retirement": ["Worn down mentally after a hard stretch", "Lost his love of the grind", "Steps away for his mental health"],
        "family_retirement": ["Putting family first", "Heading home to his family", "Family comes first now"],
        "depth_retirement": ["Lost his NHL roster foothold", "Squeezed out of an NHL job", "End of the road as a depth player"],
        "legend_farewell": ["A legend takes his final bow", "Farewell to one of the greats", "An icon hangs them up"],
        "goalie_longevity_retirement": ["Veteran goalie ends long run", "Steps out of the crease for good", "Final save of a long career"],
    }
    opts = mapping.get(retirement_type) or mapping["age_retirement"]
    text = _pick(opts, player, retirement_type)
    if retirement_type == "free_agent_walkaway" and yrs > 0:
        text = "Walks away from his contract"
    return text


def _retirement_detail(retirement_type: str, ev: Dict[str, Any]) -> str:
    """One factual sentence built only from the evidence."""
    bits: List[str] = []
    age = ev["age"]
    if ev["career_gp"]:
        bits.append(f"{ev['career_gp']:,} NHL games over {ev['seasons']} season{'s' if ev['seasons'] != 1 else ''}")
    rt = retirement_type
    if rt == "injury_retirement":
        lead = f"{ev['injuries']} injuries on record and heavy wear" if ev["injuries"] else "Heavy physical wear"
    elif rt == "cup_win_walkaway":
        lead = f"Won the Cup this spring at {age}"
    elif rt == "legend_farewell":
        lead = f"Still a {int(round(ev['ovr_disp']))} OVR at {age}"
    elif rt == "decline_retirement":
        lead = (f"Down from a {ev['peak']} peak to {int(round(ev['ovr_disp']))} OVR" if ev["peak"] and ev["drop"] >= 1
                else f"{int(round(ev['ovr_disp']))} OVR at {age}")
    elif rt == "no_contract_retirement":
        lead = f"Unsigned at {age} with a {int(round(ev['ovr_disp']))} OVR"
    elif rt == "free_agent_walkaway":
        lead = f"Free agent at {age}, still a {int(round(ev['ovr_disp']))} OVR"
    elif rt == "role_frustration_retirement":
        lead = f"Played {ev['final_gp']} games last season" + (f" with {ev['scratches']} scratches" if ev["scratches"] else "")
    elif rt == "morale_retirement":
        lead = f"Morale {int(round(ev['morale'] * 100))}/100 with sustained fatigue"
    elif rt == "family_retirement":
        lead = f"Family pressure weighed on him at {age}"
    elif rt == "goalie_longevity_retirement":
        lead = f"Still a {int(round(ev['ovr_disp']))} OVR goalie at {age}"
    elif rt == "depth_retirement":
        lead = f"{int(round(ev['ovr_disp']))} OVR at {age}"
    else:
        lead = f"Final season at {int(round(ev['ovr_disp']))} OVR, age {age}"
    out = lead + (" · " + bits[0] if bits else "")
    if ev["cups"] and rt != "cup_win_walkaway":
        out += f" · {ev['cups']}x Stanley Cup champion"
    return out


def _legacy_tier(ovr: float, age: int, cups: int, pts: int) -> str:
    if ovr >= 0.88 and (pts >= 700 or cups >= 2):
        return "franchise_legend"
    if ovr >= 0.84 or pts >= 550 or cups >= 2:
        return "star"
    if ovr >= 0.78 or pts >= 350:
        return "veteran"
    if age >= 36:
        return "journeyman"
    return "depth"


def _hall_of_fame_score(ovr: float, pts: int, cups: int, gp: int, awards: int) -> int:
    score = int(ovr * 42) + int(pts * 0.08) + cups * 18 + min(gp, 1200) * 0.02 + awards * 12
    return int(_clamp(score, 0.0, 100.0) * 1.0)


def _jersey_retirement_score(ovr: float, gp: int, cups: int, tenure: int, captain: bool) -> int:
    score = int(ovr * 35) + min(gp, 900) * 0.03 + cups * 22 + tenure * 4 + (15 if captain else 0)
    return int(_clamp(score, 0.0, 100.0) * 1.0)


def _captaincy_status(player: Any, team: Any) -> str:
    pid = _player_id(player)
    if not pid:
        return ""
    cap = str(getattr(team, "captain_id", getattr(team, "captain", "")) or "")
    alt = list(getattr(team, "alternate_captains", None) or getattr(team, "alternates", None) or [])
    if cap and cap == pid:
        return "captain"
    if pid in [str(x) for x in alt]:
        return "alternate"
    if getattr(player, "is_captain", False) or getattr(player, "captain", False):
        return "captain"
    return ""


def _news_headline(name: str, team_name: str, retirement_type: str, legacy: str) -> str:
    if retirement_type == "legend_farewell":
        return f"{name} calls it a career — {team_name} icon retires"
    if retirement_type == "cup_win_walkaway":
        return f"{name} retires on top after Cup run with {team_name}"
    if retirement_type == "injury_retirement":
        return f"{name} medically retires after injury battles"
    if legacy == "franchise_legend":
        return f"League mourns loss of {name} to retirement"
    return f"{name} announces retirement from {team_name}"


def _elite_cutoff(teams: List[Any]) -> float:
    ovs: List[float] = []
    for team in teams:
        for p in getattr(team, "roster", None) or []:
            if getattr(p, "retired", False):
                continue
            ovs.append(_player_ovr_norm(p))
    ovs.sort()
    if len(ovs) >= 8:
        return ovs[max(0, int(len(ovs) * 0.90) - 1)]
    return 0.88


def _is_locked_retirement(risk: int, age: int, ovr: float, wear: float) -> bool:
    if age >= 41 and ovr < 0.90:
        return True
    if age >= 43:
        return True
    if risk >= 88:
        return True
    if wear >= 0.92 and age >= 30:
        return True
    if age >= 39 and ovr < 0.52:
        return True
    return False


def _is_user_borderline(session: FranchiseSession, tid: str, risk: int, age: int, locked: bool) -> bool:
    if str(tid) != str(session.user_team_id):
        return False
    if locked:
        return False
    return age >= 33 and 40 <= risk < 85


def _serialize_retirement_row(
    session: FranchiseSession,
    player: Any,
    team: Any,
    *,
    retirement_type: str,
    retirement_reason: str,
    retirement_status: str,
    retirement_risk: int,
    confirmed: bool,
) -> Dict[str, Any]:
    from services.franchise_sim import (
        _contract_years_remaining,
        _display_team,
        player_cap_hit_millions,
    )

    tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
    career = _career_stats(player, session, team)
    ovr = _player_ovr_norm(player)
    ovr_disp = _player_ovr_display(player)
    prev = float(getattr(player, "_pre_retirement_ovr", ovr_disp) or ovr_disp)
    cups = _cup_wins(player)
    cap = round(float(player_cap_hit_millions(player) or 0), 3)
    captain = _captaincy_status(player, team)
    legacy = _legacy_tier(ovr, _player_age(player), cups, career["points"])
    awards = list(getattr(player, "awards", None) or [])
    hof = _hall_of_fame_score(ovr, career["points"], cups, career["games_played"], len(awards))
    jersey = _jersey_retirement_score(
        ovr, career["games_played"], cups, career["seasons_played"], captain == "captain"
    )
    team_name = _display_team(team) if team else tid

    row: Dict[str, Any] = {
        "player_id": _player_id(player),
        "name": _player_name(player),
        "team_id": tid,
        "team_name": team_name,
        "age": _player_age(player),
        "position": _player_position(player),
        "overall": ovr_disp,
        "previous_overall": round(prev, 1),
        "seasons_played": career["seasons_played"],
        "games_played": career["games_played"],
        "goals": career["goals"],
        "assists": career["assists"],
        "points": career["points"],
        "goalie_wins": career["goalie_wins"],
        "shutouts": career["shutouts"],
        "cups_won": cups,
        "awards": awards,
        "captaincy_status": captain,
        "retirement_reason": retirement_reason,
        "retirement_type": retirement_type,
        "retirement_status": retirement_status,
        "retirement_risk": retirement_risk,
        "legacy_tier": legacy,
        "hall_of_fame_score": hof,
        "jersey_retirement_score": jersey,
        "cap_hit_removed": cap if confirmed else 0.0,
        "contract_years_remaining": _contract_years_remaining(player),
        "final_team": team_name,
        "draft_info": _draft_info_dict(player),
        "career_earnings": float(getattr(player, "career_earnings", 0) or 0),
        "news_headline": _news_headline(_player_name(player), team_name, retirement_type, legacy),
        "confirmed": confirmed,
    }
    try:
        row.update(_tribute_fields(session, player, team, career, retirement_type, cap))
    except Exception:
        _swallowed_log.debug("retirement tribute fields failed", exc_info=True)
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row

        row = merge_headshot_into_row(row, player)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return row


def _tribute_fields(
    session: FranchiseSession,
    player: Any,
    team: Any,
    career: Dict[str, Any],
    retirement_type: str,
    cap: float,
) -> Dict[str, Any]:
    """Extra, real-data fields for the Retirements tribute screen (all optional on the wire)."""
    from services.franchise_sim import _contract_years_remaining

    pid = _player_id(player)
    out: Dict[str, Any] = {
        "team_abbrev": _team_abbrev(team) or None,
        "is_user_team": bool(team is not None and str(getattr(team, "team_id", getattr(team, "id", "")) or "") == str(getattr(session, "user_team_id", "") or "")),
    }
    summary = career.get("summary") if isinstance(career, dict) else None
    if isinstance(summary, dict) and int(summary.get("gp") or 0) > 0:
        out["career_totals"] = {k: v for k, v in summary.items() if k != "best_season"}
        if summary.get("best_season"):
            out["best_season"] = summary["best_season"]
        out["career_source"] = "nhl"
    else:
        out["career_source"] = "estimate"

    # Final season (the one that just ended) — regular season and playoffs.
    try:
        from services.franchise_sim import _compact_season_stats_for_player

        fs = _compact_season_stats_for_player(session, pid) or {}
        if int(fs.get("gp") or 0) > 0:
            keep = ("gp", "g", "a", "pts", "plusMinus", "pim", "wins", "losses", "otl", "svPct", "gaa", "shutouts", "war")
            out["final_season"] = {k: fs.get(k) for k in keep if fs.get(k) is not None}
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    po = (getattr(session, "playoff_player_stats", None) or {}).get(pid)
    if isinstance(po, dict) and int(_num(po.get("gp")) or 0) > 0:
        out["final_playoffs"] = {
            k: po.get(k) for k in ("gp", "g", "a", "pts", "w", "l", "so") if po.get(k) is not None
        }

    peak, peak_season = _rating_peak(player)
    if peak is not None:
        out["peak_overall"] = peak
        if peak_season:
            out["peak_season"] = _season_label(int(peak_season))

    trophies = _award_list(player)
    if trophies:
        out["trophies"] = trophies[:24]

    try:
        from services.award_history import _draft_payload_from_player

        d = _draft_payload_from_player(player)
        if d.get("drafted") or d.get("undrafted"):
            dtid = d.get("drafted_by_team_id") or d.get("draft_team_id")
            dteam = (getattr(session, "team_by_id", None) or {}).get(str(dtid)) if dtid else None
            if dteam is not None and not d.get("draft_team_abbr"):
                d["draft_team_abbr"] = _team_abbrev(dteam) or None
            out["draft"] = d
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    contract = getattr(player, "contract", None)
    if contract is not None and cap > 0:
        yrs = int(_contract_years_remaining(player) or 0)
        out["last_contract"] = {
            "aav_millions": round(cap, 3),
            "years_remaining": yrs,
            "walked_away_millions": round(cap * yrs, 3) if yrs > 0 else 0.0,
        }

    try:
        ctx = _build_retirement_context(session, player, team)
        ev = _retirement_evidence(player, session, team, ctx)
        out["retirement_detail"] = _retirement_detail(retirement_type, ev)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return out


def _archive_retired_player(session: FranchiseSession, player: Any, row: Dict[str, Any]) -> None:
    if not hasattr(session, "retired_players_archive") or session.retired_players_archive is None:
        session.retired_players_archive = []
    archive = {
        **row,
        "retirement_year": int(session.season_calendar_year),
        "retirement_season": f"{session.season_calendar_year}-{int(session.season_calendar_year) + 1}",
        "teams_played_for": list(getattr(player, "teams_played_for", None) or [row.get("team_id")]),
        "identity_snapshot": {
            "name": row.get("name"),
            "age": row.get("age"),
            "position": row.get("position"),
            "nationality": str(getattr(getattr(player, "identity", None), "birth_country", "") or ""),
        },
    }
    session.retired_players_archive.append(archive)

    hist = list(getattr(session, "season_history", None) or [])
    found = False
    for entry in hist:
        if isinstance(entry, dict) and int(entry.get("year", -1)) == int(session.season_calendar_year):
            rets = list(entry.get("retirements") or [])
            rets.append(archive)
            entry["retirements"] = rets
            found = True
            break
    if not found:
        hist.append({"year": int(session.season_calendar_year), "retirements": [archive]})
    session.season_history = hist

    tid = str(row.get("team_id") or "")
    team = session.team_by_id.get(tid)
    if team is not None:
        alumni = list(getattr(team, "retired_alumni", None) or [])
        alumni.append(archive)
        try:
            team.retired_alumni = alumni
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)


def _apply_team_retirement_effects(session: FranchiseSession, team: Any, player: Any, row: Dict[str, Any]) -> None:
    from services.franchise_sim import player_cap_hit_millions

    pos = _player_position(player) or "F"
    needs = _attr_dict(getattr(team, "needs", None))
    label = f"Replace retiring {pos}"
    needs[pos] = int(needs.get(pos, 0) or 0) + 1
    needs["offseason_need"] = label
    try:
        team.needs = needs
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    cap_freed = list(getattr(team, "_retirement_cap_freed", None) or [])
    cap_freed.append(float(row.get("cap_hit_removed") or player_cap_hit_millions(player) or 0))
    try:
        team._retirement_cap_freed = cap_freed
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    pid = _player_id(player)
    cap_id = str(getattr(team, "captain_id", getattr(team, "captain", "")) or "")
    if cap_id and cap_id == pid:
        try:
            team.captain_id = ""
            team.captain = ""
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    alts = list(getattr(team, "alternate_captains", None) or getattr(team, "alternates", None) or [])
    if pid in [str(x) for x in alts]:
        alts = [x for x in alts if str(x) != pid]
        try:
            team.alternate_captains = alts
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)


def _sweater_number(player: Any) -> Any:
    ident = getattr(player, "identity", None)
    for src in (player, ident):
        if src is None:
            continue
        for key in ("sweater_number", "jersey_number", "sweater", "jersey", "number"):
            val = getattr(src, key, None)
            if val not in (None, "", 0):
                return val
    return None


def _apply_retirement_cap_charge(session: FranchiseSession, team: Any, player: Any, row: Dict[str, Any]) -> None:
    """NHL CBA: a retired player's deal comes off the books, unless it is a 35+
    contract (signed at 35 or older). Then the full cap hit stays for every
    season left on it. The old version charged a buyout to everyone, starting
    with the season that had already been played."""
    if team is None:
        return
    try:
        # contract_economy's reader handles dict contracts; franchise_sim's returns 1.
        from services.contract_economy import _contract_years_remaining, player_cap_hit_millions
    except Exception:
        return
    years = _contract_years_remaining(player)
    hit = float(player_cap_hit_millions(player) or 0)
    row["cap_hit_removed"] = hit
    if years <= 0 or hit <= 0:
        return
    season_year = int(getattr(session, "season_calendar_year", 0) or 0)
    phase = str(getattr(session, "phase", "") or "").lower()
    in_season = phase in ("regular", "preseason", "playoffs", "playoff_ready")
    # Before the salary-cap stage burns a year, years_remaining still counts the
    # season that just ended.
    burned = bool(getattr(team, "_contract_year_burned", False))
    future_years = years if (in_season or burned) else years - 1
    first_season = season_year if in_season else season_year + 1
    c = getattr(player, "contract", None)
    c = c if isinstance(c, dict) else {}
    try:
        start = int(c.get("effective_season") or c.get("start_year") or 0)
    except (TypeError, ValueError):
        start = 0
    if not start:
        try:
            start = int(c.get("expiry_year") or 0) - int(c.get("years") or years)
        except (TypeError, ValueError):
            start = 0
    ident = getattr(player, "identity", None)
    try:
        age_now = int(getattr(ident, "age", None) or getattr(player, "age", 0) or 0)
    except (TypeError, ValueError):
        age_now = 0
    age_at_signing = age_now - max(0, season_year - start) if start else age_now
    row["thirty_five_plus"] = bool(age_at_signing >= 35)
    if age_at_signing < 35 or future_years <= 0:
        row["buyout_annual_m"] = 0.0
        row["buyout_years"] = 0
        # CBA cap-advantage recapture: a long front-loaded deal that ends by
        # retirement charges the banked advantage over the seasons it had left.
        try:
            from services.contract_economy import cap_recapture_amount_m

            recapture = float(cap_recapture_amount_m(player) or 0.0) if future_years > 0 else 0.0
        except Exception:
            recapture = 0.0
        if recapture > 0.01:
            per = round(recapture / future_years, 3)
            buyouts = list(getattr(team, "buyout_cap_hits", None) or [])
            for i in range(int(future_years)):
                yr = first_season + i
                buyouts.append({
                    "season": f"{yr}-{(yr + 1) % 100:02d}",
                    "amount_m": per,
                    "player_id": _player_id(player),
                    "player_name": _player_name(player),
                    "source": "recapture",
                })
            try:
                team.buyout_cap_hits = buyouts
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            row["recapture_annual_m"] = per
            row["recapture_years"] = int(future_years)
        return
    buyouts = list(getattr(team, "buyout_cap_hits", None) or [])
    for i in range(int(future_years)):
        yr = first_season + i
        buyouts.append({
            "season": f"{yr}-{(yr + 1) % 100:02d}",
            "amount_m": round(hit, 3),
            "player_id": _player_id(player),
            "player_name": _player_name(player),
            "source": "retirement_35_plus",
        })
    try:
        team.buyout_cap_hits = buyouts
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    row["cap_hit_removed"] = 0.0
    row["buyout_annual_m"] = round(hit, 3)
    row["buyout_years"] = int(future_years)
    contract = getattr(player, "contract", None)
    if contract is not None:
        for attr in ("years_remaining", "term_remaining", "remaining_years"):
            if hasattr(contract, attr):
                try:
                    setattr(contract, attr, 0)
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
        for attr in ("aav_m", "cap_hit_m", "aav"):
            if hasattr(contract, attr):
                try:
                    setattr(contract, attr, 0)
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)


def _record_retirement_honors(session: FranchiseSession, team: Any, player: Any, row: Dict[str, Any]) -> None:
    hof = int(row.get("hall_of_fame_score") or 0)
    jersey = int(row.get("jersey_retirement_score") or 0)
    year = int(getattr(session, "season_calendar_year", 0) or 0)
    pid = str(row.get("player_id") or "")
    if hof >= 78:
        row["hall_of_fame_candidate"] = True
        watch = list(getattr(session, "hall_of_fame_watch", None) or [])
        if not any(isinstance(item, dict) and str(item.get("player_id") or "") == pid for item in watch):
            watch.append({"player_id": pid, "name": row.get("name"), "score": hof, "year": year})
        try:
            session.hall_of_fame_watch = watch
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    if jersey < 72 or team is None:
        return
    number = _sweater_number(player)
    if number in (None, ""):
        return
    book = list(getattr(team, "retired_numbers", None) or [])
    if not any(isinstance(item, dict) and str(item.get("player_id") or "") == pid for item in book):
        book.append({"number": number, "player_id": pid, "name": row.get("name"), "year": year})
    try:
        team.retired_numbers = book
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    row["jersey_retired"] = True
    row["retired_number"] = number


def _confirm_retirement(
    session: FranchiseSession,
    player: Any,
    team: Any,
    row: Dict[str, Any],
) -> None:

    _apply_retirement_cap_charge(session, team, player, row)
    _record_retirement_honors(session, team, player, row)
    player.retired = True
    try:
        player.retirement_reason = row.get("retirement_reason")
        player.retirement_type = row.get("retirement_type")
        player.retirement_year = int(session.season_calendar_year)
        player.retirement_status = "confirmed"
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)

    league = getattr(session.sim, "league", None)
    if league is not None:
        pool = list(getattr(league, "retired_players", None) or [])
        if player not in pool:
            pool.append(player)
        try:
            league.retired_players = pool
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    _archive_retired_player(session, player, row)
    _apply_team_retirement_effects(session, team, player, row)


def _enqueue_borderline_decision(session: FranchiseSession, row: Dict[str, Any]) -> None:
    pid = str(row.get("player_id") or "")
    dec_id = f"retire_{pid}_{int(session.season_calendar_year)}"
    if not hasattr(session, "pending_decisions") or session.pending_decisions is None:
        session.pending_decisions = []

    existing = {str(d.get("id") or "") for d in session.pending_decisions if isinstance(d, dict)}
    if dec_id in existing:
        return

    session.pending_decisions.append(
        {
            "id": dec_id,
            "kind": "retirement_decision",
            "type": "retirement_decision",
            "title": f"Retirement — {row.get('name')}",
            "summary": f"{row.get('name')} ({row.get('age')}) is considering retirement ({row.get('retirement_risk')}% risk).",
            "player_id": pid,
            "player_name": row.get("name"),
            "meta": {
                "player_id": pid,
                "player_name": row.get("name"),
                "team_id": row.get("team_id"),
                "retirement_risk": row.get("retirement_risk"),
            },
            "options": [
                {"id": "let_retire", "label": "Let him retire", "effect_summary": "Player retires; cap space opens."},
                {"id": "one_more_year", "label": "Ask for one more year", "effect_summary": "Player returns; morale bump; risk drops."},
                {"id": "one_year_deal", "label": "Offer one-year extension", "effect_summary": "Cap commitment; strong stay signal."},
                {"id": "reduced_role", "label": "Promise reduced workload", "effect_summary": "Lower fatigue risk; slight morale boost."},
                {"id": "leadership_role", "label": "Promise leadership role", "effect_summary": "Captaincy consideration; morale boost."},
                {"id": "contender_push", "label": "Promise contender push", "effect_summary": "Win-now pitch; may backfire if team misses playoffs."},
            ],
        }
    )


def _find_player_in_org(team: Any, player_id: str) -> Any:
    if team is None:
        return None
    for bucket in ("roster", "ahl_roster", "echl_roster", "prospect_pool"):
        for player in list(getattr(team, bucket, None) or []):
            if _player_id(player) == str(player_id):
                return player
    return None


def _extend_contract_one_year(player: Any) -> int:
    contract = getattr(player, "contract", None)
    if contract is None:
        contract = SimpleNamespace(years_remaining=0, aav_m=0.775, cap_hit_m=0.775)
        try:
            player.contract = contract
        except Exception:
            return 0
    years = 0
    for attr in ("years_remaining", "term_remaining", "remaining_years"):
        if hasattr(contract, attr):
            try:
                years = max(years, int(getattr(contract, attr) or 0))
            except (TypeError, ValueError):
                years = 0
            break
    years += 1
    for attr in ("years_remaining", "term_remaining", "remaining_years"):
        if hasattr(contract, attr):
            try:
                setattr(contract, attr, years)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
    hit = 0.0
    for attr in ("aav_m", "cap_hit_m", "aav"):
        if hasattr(contract, attr):
            try:
                hit = float(getattr(contract, attr) or 0)
            except (TypeError, ValueError):
                hit = 0.0
            if hit > 0:
                break
    if hit <= 0:
        for attr in ("aav_m", "cap_hit_m"):
            try:
                setattr(contract, attr, 0.775)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
    return years


def _demote_saved_line(session: FranchiseSession, player_id: str) -> bool:
    lines = getattr(session, "lines", None)
    if not isinstance(lines, dict):
        return False
    block = lines.get("even_strength") if isinstance(lines.get("even_strength"), dict) else lines
    inner = block.get("lines") if isinstance(block, dict) and isinstance(block.get("lines"), dict) else block
    if not isinstance(inner, dict):
        return False
    moved = False
    for group in ("forwards", "defense", "goalies"):
        units = list(inner.get(group) or [])
        for index, unit in enumerate(units):
            if not isinstance(unit, dict):
                continue
            slots = dict(unit.get("slots") or {})
            slot_name = next((key for key, val in slots.items() if str(val) == str(player_id)), None)
            if not slot_name:
                continue
            if index + 1 < len(units) and isinstance(units[index + 1], dict):
                below = units[index + 1]
                below_slots = dict(below.get("slots") or {})
                slots[slot_name] = below_slots.get(slot_name) or ""
                below_slots[slot_name] = str(player_id)
                unit["slots"] = slots
                below["slots"] = below_slots
            moved = True
            break
        if moved:
            break
    for special in ("power_play", "penalty_kill"):
        units = lines.get(special)
        if isinstance(units, dict):
            units = units.get("lines")
        if not isinstance(units, list):
            continue
        for unit in units:
            if not isinstance(unit, dict):
                continue
            slots = dict(unit.get("slots") or {})
            for key, val in list(slots.items()):
                if str(val) == str(player_id):
                    slots[key] = ""
                    moved = True
            unit["slots"] = slots
    return moved


def _grant_leadership(team: Any, player: Any) -> str:
    pid = _player_id(player)
    captain = str(getattr(team, "captain_id", "") or getattr(team, "captain", "") or "")
    alternates = [str(item) for item in (getattr(team, "alternate_captains", None) or [])]
    if not captain:
        try:
            team.captain_id = pid
            team.captain = pid
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return "captain"
    if pid not in alternates and len(alternates) < 2:
        alternates.append(pid)
        try:
            team.alternate_captains = alternates
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return "alternate"
    return "named"


def apply_retirement_decision(session: FranchiseSession, decision: Dict[str, Any], choice_id: str) -> Dict[str, Any]:
    """Resolve a user retirement_decision from apply_decision."""
    meta = dict(decision.get("meta") or {})
    pid = str(meta.get("player_id") or decision.get("player_id") or "")
    cid = str(choice_id or "")
    user_team = session.team_by_id.get(str(session.user_team_id))
    player = _find_player_in_org(user_team, pid)

    effects: Dict[str, Any] = {"choice": cid}
    if player is None:
        return effects

    if cid == "let_retire":
        ctx = _build_retirement_context(session, player, user_team)
        rtype = _classify_retirement_type(player, session, user_team, primary_reason="voluntary", risk=90, ctx=ctx)
        reason = _readable_retirement_reason(rtype, player, ctx)
        row = _serialize_retirement_row(
            session,
            player,
            user_team,
            retirement_type=rtype,
            retirement_reason=reason,
            retirement_status="confirmed",
            retirement_risk=int(meta.get("retirement_risk") or 80),
            confirmed=True,
        )
        _confirm_retirement(session, player, user_team, row)
        _merge_into_retirements_payload(session, row, section="team")
        from services.franchise_sim import _strip_retired_from_nhl_rosters

        league = getattr(session.sim, "league", None)
        _strip_retired_from_nhl_rosters(list(getattr(league, "teams", None) or []))
        if user_team is not None:
            for bucket in ("ahl_roster", "echl_roster", "prospect_pool"):
                rows = list(getattr(user_team, bucket, None) or [])
                kept = [person for person in rows if person is not player and _player_id(person) != pid]
                if len(kept) != len(rows):
                    try:
                        setattr(user_team, bucket, kept)
                    except Exception:
                        _swallowed_log.debug("suppressed exception", exc_info=True)
        effects["retired"] = True
        effects["buyout_annual_m"] = row.get("buyout_annual_m")
        effects["cap_hit_removed"] = row.get("cap_hit_removed")
    else:
        year = int(getattr(session, "season_calendar_year", 0) or 0)
        try:
            player.retirement_status = "returning_for_one_more_year"
            player._retirement_return_season = year
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        psych = getattr(player, "psych", None)
        if psych is not None and hasattr(psych, "morale"):
            try:
                psych.morale = _clamp(_player_morale(player) + 0.06, 0.0, 1.0)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
        if cid == "one_year_deal":
            effects["contract_years"] = _extend_contract_one_year(player)
        elif cid == "reduced_role":
            health = getattr(player, "health", None)
            if health is not None and hasattr(health, "wear_and_tear"):
                try:
                    health.wear_and_tear = max(0.0, float(health.wear_and_tear or 0) - 0.08)
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)
            effects["line_demoted"] = _demote_saved_line(session, pid)
            try:
                player._retirement_role = "reduced_minutes"
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
        elif cid == "leadership_role":
            effects["leadership"] = _grant_leadership(user_team, player)
        elif cid == "contender_push":
            try:
                from app.sim_engine.franchise.storyline_engine import create_universe_promise

                promise = create_universe_promise(
                    session,
                    {
                        "type": "winning_commitment",
                        "player_id": pid,
                        "due_games": 20,
                        "description": "Show him the club is still trying to win.",
                    },
                    pid,
                    f"retire_{pid}",
                )
                effects["promise_id"] = (promise or {}).get("id")
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            try:
                player._retirement_promise = "contender_push"
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
        effects["retired"] = False
        effects["returns_through_season"] = year

    return effects


def _merge_into_retirements_payload(session: FranchiseSession, row: Dict[str, Any], *, section: str) -> None:
    payload = session.retirements_payload
    if not isinstance(payload, dict):
        payload = {"all": safe_list(payload)}
        session.retirements_payload = payload
    for key in ("all", section, "league", "legends", "team", "considering", "depth"):
        if key not in payload:
            payload[key] = []
    payload["all"].append(row)
    payload[section].append(row)
    if row.get("legacy_tier") in ("franchise_legend", "star"):
        payload["legends"].append(row)
    if str(row.get("team_id")) == str(session.user_team_id):
        if row not in payload["team"]:
            payload["team"].append(row)
    else:
        if row not in payload["league"]:
            payload["league"].append(row)


def safe_list(val: Any) -> List[Any]:
    if isinstance(val, list):
        return list(val)
    if isinstance(val, dict) and "all" in val:
        return list(val.get("all") or [])
    return []


def _run_depth_retirement_pass(session: FranchiseSession, rng: Any) -> List[Dict[str, Any]]:
    """Retire older low-OVR players from FA / minors pools."""
    from services.franchise_sim import _purge_retired_from_extra_pools

    league = getattr(session.sim, "league", None)
    if league is None:
        return []

    pool: List[Tuple[Any, Any, str]] = []
    for p in getattr(league, "free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append((p, None, "UFA"))
    for p in getattr(league, "overseas_free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append((p, None, "Overseas"))
    for tm in getattr(league, "teams", None) or []:
        for p in getattr(tm, "ahl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append((p, tm, "AHL"))
        for p in getattr(tm, "echl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append((p, tm, "ECHL"))

    candidates: List[Tuple[float, Any, Any, str]] = []
    for p, team, label in pool:
        age = _player_age(p)
        if age < 30:
            continue
        ovr = _player_ovr_norm(p)
        prob = 0.0
        if age >= 38:
            prob = 0.35
        elif age >= 35:
            prob = 0.18
        elif age >= 33:
            prob = 0.08
        if ovr < 0.55:
            prob += 0.12
        elif ovr < 0.62:
            prob += 0.06
        prob = _clamp(prob, 0.0, 0.55)
        candidates.append((prob, p, team, label))

    projected = sum(c[0] for c in candidates)
    if projected > DEPTH_RETIREMENT_SOFT_CAP:
        factor = DEPTH_RETIREMENT_SOFT_CAP / max(projected, 1.0)
        candidates = [(c[0] * factor, c[1], c[2], c[3]) for c in candidates]

    retired_rows: List[Dict[str, Any]] = []
    for prob, p, team, label in candidates:
        if prob <= 0:
            continue
        if rng.random() >= prob:
            continue
        p.retired = True
        if team is not None:
            team_id = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
            team_name = str(getattr(team, "name", "") or team_id)
            if label in ("AHL", "ECHL"):
                team_name = f"{team_name} ({label})"
        else:
            team_id = str(getattr(p, "last_team_id", "") or getattr(p, "former_team_id", "") or "")
            team_name = "Free agent" if label == "UFA" else "Overseas" if label == "Overseas" else label
            if not team_id:
                team_id = "FA" if label == "UFA" else "OV"
        row = {
            "player_id": _player_id(p),
            "name": _player_name(p),
            "team_id": team_id,
            "team_name": team_name,
            "age": _player_age(p),
            "position": _player_position(p),
            "overall": _player_ovr_display(p),
            "retirement_type": "depth_retirement",
            "retirement_reason": "Stepping away from pro hockey",
            "retirement_status": "confirmed",
            "retirement_risk": int(prob * 100),
            "confirmed": True,
            "news_headline": f"{_player_name(p)} retires from {team_name}",
            "team_abbrev": _team_abbrev(team) or None,
            "level": label,
            "is_user_team": bool(team is not None and str(team_id) == str(getattr(session, "user_team_id", "") or "")),
        }
        try:
            summary = _nhl_career_summary(p, session, None)
            if int(summary.get("gp") or 0) > 0:
                row["career_totals"] = {k: v for k, v in summary.items() if k != "best_season"}
                row["career_source"] = "nhl"
            from app.sim_engine.generation.player_headshots import merge_headshot_into_row

            row = merge_headshot_into_row(row, p)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        retired_rows.append(row)
        _purge_retired_from_extra_pools(session, p)
        if len(retired_rows) >= DEPTH_RETIREMENT_HARD_CAP:
            break

    return retired_rows


def run_franchise_retirement_pass(session: FranchiseSession) -> Dict[str, Any]:
    """
    Retirement-only offseason pass for NHL rosters.
    Idempotent when session.retirements_processed is True.
    """
    if getattr(session, "retirements_processed", False) and session.retirements_payload:
        existing = session.retirements_payload
        if isinstance(existing, dict):
            return existing
        return {"all": safe_list(existing)}

    from services.franchise_sim import _franchise_nhl_age_and_phase_tick, _strip_retired_from_nhl_rosters, player_cap_hit_millions

    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    rng = sim.rng

    if not teams:
        payload = _empty_payload()
        session.retirements_payload = payload
        session.retirements_processed = True
        return payload

    # Age NHL players once for the new season year (no OVR progression here).
    _franchise_nhl_age_and_phase_tick(session, teams)

    elite_cut = _elite_cutoff(teams)
    candidates: List[Dict[str, Any]] = []

    for team in teams:
        tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
        for player in list(getattr(team, "roster", None) or []):
            if getattr(player, "retired", False):
                continue
            if str(getattr(player, "retirement_status", "") or "") == "returning_for_one_more_year":
                granted = getattr(player, "_retirement_return_season", None)
                year = int(getattr(session, "season_calendar_year", 0) or 0)
                try:
                    granted_year = int(granted) if granted is not None else None
                except (TypeError, ValueError):
                    granted_year = None
                if granted_year is not None and granted_year >= year:
                    continue
                try:
                    player.retirement_status = ""
                except Exception:
                    _swallowed_log.debug("suppressed exception", exc_info=True)

            age = _player_age(player)
            ovr = _player_ovr_norm(player)
            wear = _injury_wear(player)
            ctx = _build_retirement_context(session, player, team)

            decision, engine_chance = _evaluate_with_engine(session, player, team, rng)
            factors = getattr(decision, "factors", None) if decision is not None else None
            risk = _compute_retirement_risk(player, team, session, engine_chance, factors)

            retire_chance = max(float(engine_chance), float(risk) / 100.0)
            if _is_goalie(player):
                retire_chance *= 0.72
            if age >= 41 and ovr < 0.90:
                retire_chance = max(retire_chance, 0.82)
            elif age >= 41:
                retire_chance = max(retire_chance, 0.55)
            risk = int(round(_clamp(retire_chance, 0.0, 1.0) * 100))

            # Age gate: nobody under 32 retires without a real career-ending injury, and
            # healthy 32-33s only rarely do. Unknown age (0) counts as young.
            if age < 32 and not _true_career_ending(player):
                retire_chance = 0.0
                risk = min(risk, 5)
            elif age < 34 and not _true_career_ending(player):
                cap_chance = 0.03 if ovr >= 0.62 else 0.08
                if retire_chance > cap_chance:
                    retire_chance = cap_chance
                    risk = int(round(cap_chance * 100))

            healthy_star = ovr >= elite_cut and age < 35 and wear < 0.85 and not ctx.get("career_ending_injury")
            if healthy_star:
                retire_chance = 0.0
                risk = min(risk, 24)
            elif not _true_career_ending(player) and _true_wear(player) < 0.6 and age < 39:
                # Players rarely walk away while still elite or with real money left on a deal.
                yrs_left = int(ctx.get("contract_years_remaining") or 0)
                if ovr >= elite_cut and age < 37:
                    retire_chance *= 0.45
                elif ovr >= 0.80 and yrs_left >= 2:
                    retire_chance *= 0.6
                elif yrs_left >= 3:
                    retire_chance *= 0.7
                risk = int(round(_clamp(retire_chance, 0.0, 1.0) * 100))

            primary = str(getattr(decision, "primary_reason", "age") if decision else "age")
            locked = _is_locked_retirement(risk, age, ovr, wear) and (age >= 34 or _true_career_ending(player))
            borderline = _is_user_borderline(session, tid, risk, age, locked)

            candidates.append(
                {
                    "player": player,
                    "team": team,
                    "tid": tid,
                    "age": age,
                    "ovr": ovr,
                    "risk": risk,
                    "chance": retire_chance,
                    "retire": False,
                    "locked": locked,
                    "borderline": borderline,
                    "primary": primary,
                    "ctx": ctx,
                }
            )

    # League-wide probability scaling (target floor/soft cap)
    active = [c for c in candidates if c["chance"] > 0 and not c["borderline"]]
    projected = sum(float(c["chance"]) for c in active)
    if projected > NHL_RETIREMENT_SOFT_CAP:
        factor = NHL_RETIREMENT_SOFT_CAP / projected
        for c in active:
            c["chance"] *= factor
            c["risk"] = int(round(float(c["chance"]) * 100))
    elif projected < NHL_RETIREMENT_FLOOR and active:
        older = [c for c in active if int(c["age"]) >= 33]
        low = sorted(older, key=lambda x: (-x["age"], x["ovr"]))[: max(1, NHL_RETIREMENT_FLOOR - int(projected))]
        for c in low:
            c["chance"] = min(0.65, float(c["chance"]) + 0.04)
            c["risk"] = int(round(float(c["chance"]) * 100))

    # One roll, using the same chance the card will show.
    for c in active:
        if c["locked"]:
            c["retire"] = True
            continue
        if float(c["chance"]) > 0:
            c["retire"] = rng.random() < float(c["chance"])

    # Locked retirements stay. The cap only drops voluntary ones.
    winners = [c for c in candidates if c["retire"]]
    if len(winners) > NHL_RETIREMENT_HARD_CAP:
        locked_winners = [c for c in winners if c["locked"]]
        voluntary = [c for c in winners if not c["locked"]]
        voluntary.sort(key=lambda x: (-x["age"], x["ovr"]))
        room = NHL_RETIREMENT_HARD_CAP - len(locked_winners)
        keep = {id(c["player"]) for c in locked_winners}
        if room > 0:
            keep.update(id(c["player"]) for c in voluntary[:room])
        for c in candidates:
            if c["retire"] and id(c["player"]) not in keep:
                c["retire"] = False

    payload: Dict[str, Any] = {
        "all": [],
        "legends": [],
        "team": [],
        "league": [],
        "considering": [],
        "depth": [],
        "summary": {
            "nhl_count": 0,
            "cap_freed_user": 0.0,
            "roster_needs": [],
            "headlines": [],
        },
    }

    user_tid = str(session.user_team_id)

    for c in candidates:
        # Only retirees and borderline (your club) rows are shown; skip building the rest.
        if not c["retire"] and not c["borderline"]:
            continue
        player = c["player"]
        team = c["team"]
        rtype = _classify_retirement_type(
            player, session, team, primary_reason=c["primary"], risk=c["risk"], ctx=c["ctx"]
        )
        reason = _readable_retirement_reason(rtype, player, c["ctx"])
        status = _risk_to_status(c["risk"], confirmed=c["retire"], returning=False)

        row = _serialize_retirement_row(
            session,
            player,
            team,
            retirement_type=rtype,
            retirement_reason=reason,
            retirement_status=status,
            retirement_risk=c["risk"],
            confirmed=c["retire"],
        )

        if c["borderline"]:
            row["retirement_status"] = _risk_to_status(c["risk"], confirmed=False)
            row["confirmed"] = False
            payload["considering"].append(row)
            if str(c["tid"]) == user_tid:
                payload["team"].append(row)
            _enqueue_borderline_decision(session, row)
            continue

        if c["retire"]:
            row["retirement_status"] = "confirmed"
            row["confirmed"] = True
            row["cap_hit_removed"] = round(float(player_cap_hit_millions(player) or 0), 3)
            _confirm_retirement(session, player, team, row)
            payload["all"].append(row)
            payload["summary"]["headlines"].append(row.get("news_headline", ""))
            if row.get("legacy_tier") in ("franchise_legend", "star"):
                payload["legends"].append(row)
            if str(c["tid"]) == user_tid:
                payload["team"].append(row)
                payload["summary"]["cap_freed_user"] += float(row.get("cap_hit_removed") or 0)
                needs = _attr_dict(getattr(team, "needs", None))
                need_lbl = needs.get("offseason_need")
                if need_lbl:
                    payload["summary"]["roster_needs"].append(need_lbl)
            else:
                payload["league"].append(row)

    # Already-retired carryover (reload safety)
    for team in teams:
        tid = str(getattr(team, "team_id", getattr(team, "id", "")) or "")
        for player in list(getattr(team, "roster", None) or []):
            if not getattr(player, "retired", False):
                continue
            # Deaths are not retirements; they are announced by their own event.
            if getattr(player, "deceased", False) or getattr(player, "_universe_deceased", False):
                continue
            pid = _player_id(player)
            if any(str(r.get("player_id")) == pid for r in payload["all"]):
                continue
            ctx = _build_retirement_context(session, player, team)
            rtype = str(getattr(player, "retirement_type", "age_retirement") or "age_retirement")
            row = _serialize_retirement_row(
                session,
                player,
                team,
                retirement_type=rtype,
                retirement_reason=str(getattr(player, "retirement_reason", "") or "Stepping away"),
                retirement_status="confirmed",
                retirement_risk=95,
                confirmed=True,
            )
            payload["all"].append(row)

    depth_rows = _run_depth_retirement_pass(session, rng)
    payload["depth"] = depth_rows
    payload["all"].extend(depth_rows)

    payload["summary"]["nhl_count"] = len([r for r in payload["all"] if r.get("confirmed")])
    payload["summary"]["cap_freed_user"] = round(float(payload["summary"]["cap_freed_user"]), 3)

    _strip_retired_from_nhl_rosters(teams)

    session.retirements_payload = payload
    session.retirements_processed = True
    removed = payload["summary"]["nhl_count"]
    session.timeline.append(f"OFFSEASON: Final Skate — {removed} NHL retirement(s) processed.")

    return payload


def _empty_payload() -> Dict[str, Any]:
    return {
        "all": [],
        "legends": [],
        "team": [],
        "league": [],
        "considering": [],
        "depth": [],
        "summary": {"nhl_count": 0, "cap_freed_user": 0.0, "roster_needs": [], "headlines": []},
    }


def get_retirements_list(payload: Any) -> List[Dict[str, Any]]:
    """Backward-compatible flat list for legacy UI."""
    if isinstance(payload, dict):
        return list(payload.get("all") or [])
    if isinstance(payload, list):
        return payload
    return []
