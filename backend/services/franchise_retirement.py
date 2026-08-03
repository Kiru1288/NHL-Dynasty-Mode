"""
Franchise-mode NHL retirement pass — separate from OVR/development progression.

Runs at Final Skate (offseason retirements stage). Development Camp handles progression later.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from services.franchise_session import FranchiseSession

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
        wear = max(wear, min(1.0, 0.08 * len(hist)))
    return _clamp(wear)


def _player_morale(player: Any) -> float:
    psych = getattr(player, "psych", None)
    return float(getattr(psych, "morale", getattr(player, "morale", 0.6)) or 0.6)


def _cup_wins(player: Any) -> int:
    return int(getattr(player, "cup_wins", 0) or 0)


def _season_stats(session: FranchiseSession, player: Any) -> Dict[str, Any]:
    pid = _player_id(player)
    if not pid:
        return {}
    raw = (getattr(session, "player_season_stats", None) or {}).get(pid) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _career_stats(player: Any, session: FranchiseSession) -> Dict[str, Any]:
    ss = _season_stats(session, player)
    career = getattr(player, "career_stats", None) or getattr(player, "stats", None) or {}
    if not isinstance(career, dict):
        career = {}

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

    return {
        "ovr": ovr,
        "no_offers": unsigned or yrs <= 0,
        "career_ending_injury": wear >= 0.92,
        "recent_major_injury": wear >= 0.65,
        "healthy_scratches": int(getattr(player, "healthy_scratches", 0) or 0),
        "mental_fatigue": _clamp(1.0 - morale + wear * 0.35),
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

    seed = hash((_player_id(player), int(session.season_calendar_year))) % (2**31)
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
            pass

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


def _classify_retirement_type(
    player: Any,
    session: FranchiseSession,
    team: Any,
    *,
    primary_reason: str,
    risk: int,
    ctx: Dict[str, Any],
) -> str:
    age = _player_age(player)
    ovr = _player_ovr_norm(player)
    wear = _injury_wear(player)
    morale = _player_morale(player)
    from services.franchise_sim import _contract_years_remaining

    yrs = _contract_years_remaining(player)
    pr = str(primary_reason or "").lower()

    if _is_goalie(player) and age >= 37 and ovr >= 0.72:
        return "goalie_longevity_retirement"
    if ctx.get("won_cup_this_year") and age >= 34:
        return "cup_win_walkaway"
    if wear >= 0.75 or "injur" in pr:
        return "injury_retirement"
    if yrs <= 0 and age >= 33:
        return "no_contract_retirement"
    if morale < 0.38 or "burnout" in pr or "morale" in pr:
        return "morale_retirement"
    if ovr < 0.58 and age >= 34:
        return "decline_retirement"
    if ovr < 0.62 and age >= 33:
        return "depth_retirement"
    if "voluntary" in pr or "family" in pr or "identity" in pr:
        return "role_frustration_retirement"
    if age >= 38 and ovr >= 0.84:
        return "legend_farewell"
    return "age_retirement"


def _readable_retirement_reason(retirement_type: str, player: Any, ctx: Dict[str, Any]) -> str:
    mapping = {
        "age_retirement": "Stepping away after long career",
        "injury_retirement": "Recurring injuries",
        "decline_retirement": "Age and declining role",
        "cup_win_walkaway": "Retiring after championship run",
        "no_contract_retirement": "No NHL contract interest",
        "role_frustration_retirement": "Reduced ice time and morale decline",
        "morale_retirement": "Mental fatigue and burnout",
        "depth_retirement": "Lost NHL roster foothold",
        "legend_farewell": "Legend hangs up the skates",
        "goalie_longevity_retirement": "Veteran goalie ends lengthy career",
    }
    return mapping.get(retirement_type, mapping["age_retirement"])


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
    career = _career_stats(player, session)
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
    return row


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
            pass


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
        pass

    cap_freed = list(getattr(team, "_retirement_cap_freed", None) or [])
    cap_freed.append(float(row.get("cap_hit_removed") or player_cap_hit_millions(player) or 0))
    try:
        team._retirement_cap_freed = cap_freed
    except Exception:
        pass

    pid = _player_id(player)
    cap_id = str(getattr(team, "captain_id", getattr(team, "captain", "")) or "")
    if cap_id and cap_id == pid:
        try:
            team.captain_id = ""
            team.captain = ""
        except Exception:
            pass
    alts = list(getattr(team, "alternate_captains", None) or getattr(team, "alternates", None) or [])
    if pid in [str(x) for x in alts]:
        alts = [x for x in alts if str(x) != pid]
        try:
            team.alternate_captains = alts
        except Exception:
            pass


def _confirm_retirement(
    session: FranchiseSession,
    player: Any,
    team: Any,
    row: Dict[str, Any],
) -> None:
    from services.franchise_sim import _strip_retired_from_nhl_rosters

    player.retired = True
    try:
        player.retirement_reason = row.get("retirement_reason")
        player.retirement_type = row.get("retirement_type")
        player.retirement_year = int(session.season_calendar_year)
        player.retirement_status = "confirmed"
    except Exception:
        pass

    league = getattr(session.sim, "league", None)
    if league is not None:
        pool = list(getattr(league, "retired_players", None) or [])
        if player not in pool:
            pool.append(player)
        try:
            league.retired_players = pool
        except Exception:
            pass

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


def apply_retirement_decision(session: FranchiseSession, decision: Dict[str, Any], choice_id: str) -> Dict[str, Any]:
    """Resolve a user retirement_decision from apply_decision."""
    meta = dict(decision.get("meta") or {})
    pid = str(meta.get("player_id") or decision.get("player_id") or "")
    cid = str(choice_id or "")
    user_team = session.team_by_id.get(str(session.user_team_id))
    player = None
    if user_team is not None:
        for p in getattr(user_team, "roster", None) or []:
            if _player_id(p) == pid:
                player = p
                break

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
        effects["retired"] = True
    else:
        try:
            player.retirement_status = "returning_for_one_more_year"
        except Exception:
            pass
        psych = getattr(player, "psych", None)
        if psych is not None and hasattr(psych, "morale"):
            try:
                psych.morale = _clamp(_player_morale(player) + 0.06, 0.0, 1.0)
            except Exception:
                pass
        if cid == "one_year_deal":
            effects["contract_expectation"] = "one_year_extension"
        elif cid == "reduced_role":
            effects["role_expectation"] = "reduced_minutes"
        elif cid == "leadership_role":
            effects["role_expectation"] = "leadership"
        elif cid == "contender_push":
            effects["promise"] = "contender_push"
            try:
                player._retirement_promise = "contender_push"
            except Exception:
                pass
        effects["retired"] = False

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

    pool: List[Tuple[Any, str]] = []
    for p in getattr(league, "free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append((p, "UFA"))
    for p in getattr(league, "overseas_free_agents", None) or []:
        if not getattr(p, "retired", False):
            pool.append((p, "Overseas"))
    for tm in getattr(league, "teams", None) or []:
        for p in getattr(tm, "ahl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append((p, "AHL"))
        for p in getattr(tm, "echl_roster", None) or []:
            if not getattr(p, "retired", False):
                pool.append((p, "ECHL"))

    candidates: List[Tuple[float, Any, str]] = []
    for p, label in pool:
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
        candidates.append((prob, p, label))

    projected = sum(c[0] for c in candidates)
    if projected > DEPTH_RETIREMENT_SOFT_CAP:
        factor = DEPTH_RETIREMENT_SOFT_CAP / max(projected, 1.0)
        candidates = [(c[0] * factor, c[1], c[2]) for c in candidates]

    retired_rows: List[Dict[str, Any]] = []
    for prob, p, label in candidates:
        if prob <= 0:
            continue
        if rng.random() >= prob:
            continue
        p.retired = True
        row = {
            "player_id": _player_id(p),
            "name": _player_name(p),
            "team_id": label,
            "team_name": label,
            "age": _player_age(p),
            "position": _player_position(p),
            "overall": _player_ovr_display(p),
            "retirement_type": "depth_retirement",
            "retirement_reason": "Stepping away from pro hockey",
            "retirement_status": "confirmed",
            "retirement_risk": int(prob * 100),
            "confirmed": True,
            "news_headline": f"{_player_name(p)} retires from {label}",
        }
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

    from services.franchise_sim import (
        _display_team,
        _franchise_nhl_age_and_phase_tick,
        _strip_retired_from_nhl_rosters,
        player_cap_hit_millions,
    )

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
                continue

            age = _player_age(player)
            ovr = _player_ovr_norm(player)
            wear = _injury_wear(player)
            ctx = _build_retirement_context(session, player, team)

            decision, engine_chance = _evaluate_with_engine(session, player, team, rng)
            factors = getattr(decision, "factors", None) if decision is not None else None
            risk = _compute_retirement_risk(player, team, session, engine_chance, factors)

            retire_chance = engine_chance
            if _is_goalie(player):
                retire_chance *= 0.72
            if age >= 41 and ovr < 0.90:
                retire_chance = max(retire_chance, 0.82)
            elif age >= 41:
                retire_chance = max(retire_chance, 0.55)

            if ovr >= elite_cut and age < 35 and wear < 0.85:
                retire_chance = 0.0
                risk = min(risk, 24)

            primary = str(getattr(decision, "primary_reason", "age") if decision else "age")
            locked = _is_locked_retirement(risk, age, ovr, wear)
            borderline = _is_user_borderline(session, tid, risk, age, locked)

            engine_retired = bool(getattr(decision, "retired", False)) if decision is not None else False
            if decision is None:
                engine_retired = _fallback_retire_chance(player, rng)

            if retire_chance > 0 and not engine_retired and decision is not None:
                engine_retired = rng.random() < retire_chance
            elif decision is None and not engine_retired:
                pass
            elif not engine_retired and retire_chance > 0:
                engine_retired = rng.random() < retire_chance

            candidates.append(
                {
                    "player": player,
                    "team": team,
                    "tid": tid,
                    "age": age,
                    "ovr": ovr,
                    "risk": risk,
                    "chance": retire_chance,
                    "retire": engine_retired and not borderline,
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
    elif projected < NHL_RETIREMENT_FLOOR and active:
        low = sorted(active, key=lambda x: (-x["age"], x["ovr"]))[: max(1, NHL_RETIREMENT_FLOOR - int(projected))]
        for c in low:
            c["chance"] = min(0.65, float(c["chance"]) + 0.04)

    # Re-roll with scaled chances for non-borderline
    for c in active:
        if c["borderline"]:
            continue
        if c["locked"]:
            c["retire"] = True
        elif float(c["chance"]) > 0:
            c["retire"] = rng.random() < float(c["chance"])

    # Trim if hard cap exceeded
    winners = [c for c in candidates if c["retire"]]
    if len(winners) > NHL_RETIREMENT_HARD_CAP:
        winners.sort(key=lambda x: (-x["age"], x["ovr"]))
        keep = set(id(c["player"]) for c in winners[:NHL_RETIREMENT_HARD_CAP])
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
