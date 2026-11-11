"""Transcendent draft tank pressure — compute, persist, and drive AI behavior."""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from services.draft_ranking_logic import compute_tank_pressure_for_team

logger = logging.getLogger(__name__)

TANK_STORYLINE_TYPES = {
    "TEAM_ENTERS_TANK_MODE": "Front office pivots to lottery positioning",
    "TEAM_SHOPS_VETERANS": "Club listening on veteran rentals",
    "TEAM_SELLS_RENTAL": "Expiring veteran on the market",
    "TEAM_PLAYS_YOUTH": "Coaching staff leaning on young lineup",
    "TEAM_SCRATCHES_VETERAN": "Veteran scratched for youth look",
    "TEAM_ABORTS_PLAYOFF_PUSH": "Club abandons short-term push",
}

DEV_LEAGUE_SPAWN_VERSION = 2


def _team_id(team: Any) -> str:
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", "")
    return "" if tid is None else str(tid)


def _team_status_for_tank(team: Any, standings: Any = None) -> str:
    window = str(getattr(team, "gm_window", "") or getattr(team, "window", "") or "").lower()
    if "rebuild" in window or "retool" in window:
        return "rebuilding"
    if "tank" in window:
        return "tanking"
    if "contend" in window or "playoff" in window or "win_now" in window:
        return "playoff_contender"
    st = str(getattr(team, "status", "") or "").lower()
    if "rebuild" in st or "tank" in st:
        return "tanking"
    if "contend" in st or "playoff" in st:
        return "playoff_contender"
    if standings is not None:
        tid = _team_id(team)
        rec = getattr(standings, "records", {}).get(tid)
        if rec is not None:
            gp = int(getattr(rec, "gp", 0) or 0)
            if gp >= 20:
                pts = int(getattr(rec, "points", 0) or 0)
                ppg = pts / max(1, gp)
                if ppg < 0.92:
                    return "tanking"
                if ppg < 1.02:
                    return "middling"
    return "middling"


def refresh_transcendent_tank_pressure(
    session: Any,
    sim: Any,
    *,
    transcendent_present: bool,
    draft_year: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Recompute per-team tank pressure using real pick ownership."""
    from app.sim_engine.trades.trade_pick_registry import ensure_draft_pick_registry, team_owns_own_first

    league = getattr(sim, "league", None)
    if league is None:
        return {}
    season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
    dy = int(draft_year or season_year + 1)
    ensure_draft_pick_registry(league, start_year=dy, years_ahead=4)

    standings = getattr(session, "standings", None)
    table = measure_strength_table(session, sim)
    out: Dict[str, Dict[str, Any]] = {}
    for tm in getattr(league, "teams", None) or []:
        tid = _team_id(tm)
        if not tid:
            continue
        own = team_owns_own_first(league, tid, draft_year=dy)
        strength = dict(table.get(tid) or {})
        setattr(tm, "_tank_strength", strength)
        if strength.get("is_contender"):
            setattr(tm, "team_status", "playoff_contender")
        elif int(strength.get("strength_rank") or 0) >= 22 or float(strength.get("playoff_gap_pts") or 0) >= 10:
            setattr(tm, "team_status", "tanking")
        else:
            setattr(tm, "team_status", "rebuilding")
        payload = compute_tank_pressure_for_team(
            tm,
            transcendent_present=transcendent_present,
            owns_own_first=bool(own.get("owns_own_first")),
            pick_ownership_reason=str(own.get("pick_ownership_reason") or "unknown"),
            owns_protected_first=bool(own.get("owns_protected_first")),
        )
        out[tid] = payload
        setattr(tm, "_franchise_tank_mode", payload.get("tank_mode"))
        setattr(tm, "_franchise_tank_pressure", int(payload.get("tank_pressure") or 0))
        setattr(tm, "_franchise_lottery_pick_intent", payload.get("lottery_pick_intent"))
    session.transcendent_tank_pressure = out
    setattr(league, "transcendent_tank_pressure", out)
    setattr(league, "transcendent_active", bool(transcendent_present))
    return out


def get_tank_payload(session: Any, team_id: str) -> Dict[str, Any]:
    m = getattr(session, "transcendent_tank_pressure", None) or {}
    if isinstance(m, dict):
        row = m.get(str(team_id))
        if isinstance(row, dict):
            return row
    return {"tank_pressure": 0, "tank_mode": "none"}


def record_tank_storyline(session: Any, team_id: str, event_type: str, headline: str) -> None:
    from services.franchise_sim import _record_storyline

    _record_storyline(
        session,
        {
            "type": event_type,
            "team_id": str(team_id),
            "team": str(team_id),
            "headline": headline,
            "priority": "HIGH",
            "effects": {"tank_behavior": event_type},
        },
    )


def _player_age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    if ident is not None:
        return int(getattr(ident, "age", 0) or 0)
    return int(getattr(p, "age", 0) or 0)


def _record_points(standings: Any, team_id: str) -> Tuple[int, int]:
    rec = None
    if standings is not None:
        try:
            if hasattr(standings, "find_record"):
                rec = standings.find_record(str(team_id))
        except Exception:
            rec = None
        if rec is None:
            rec = (getattr(standings, "records", None) or {}).get(str(team_id))
    if rec is None:
        return 0, 0
    gp = int(getattr(rec, "gp", 0) or 0)
    pts = int(getattr(rec, "pts", getattr(rec, "points", 0)) or 0)
    return gp, pts


def _roster_ovr(team: Any) -> float:
    try:
        from app.sim_engine.trades.team_assessment import player_ovr
    except Exception:
        player_ovr = None  # type: ignore
    vals: List[float] = []
    for p in list(getattr(team, "roster", None) or []):
        if getattr(p, "retired", False):
            continue
        if player_ovr is not None:
            vals.append(float(player_ovr(p)))
        else:
            raw = getattr(p, "ovr", 0)
            try:
                v = float(raw() if callable(raw) else raw or 0)
            except Exception:
                v = 0.0
            vals.append(v * 99.0 if 0 < v <= 1.5 else v)
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    vals.sort(reverse=True)
    top = vals[:12]
    return sum(top) / len(top)


def measure_strength_table(session: Any, sim: Any) -> Dict[str, Dict[str, Any]]:
    """Rank clubs by roster OVR and points pace. Logging and the contender test both use this.

    A club is in the true top group only when its combined rank is inside the top
    eight and it is not more than four projected points behind the playoff line.
    """
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    standings = getattr(session, "standings", None)
    games = int(getattr(session, "games_per_team_schedule", None) or getattr(league, "games_per_team", 82) or 82)
    rows: List[Dict[str, Any]] = []
    for tm in teams:
        tid = _team_id(tm)
        if not tid:
            continue
        gp, pts = _record_points(standings, tid)
        pace = (pts / gp) if gp > 0 else 0.0
        projected = pace * games if gp >= 10 else 0.0
        rows.append({
            "team_id": tid,
            "ovr": round(_roster_ovr(tm), 2),
            "gp": gp,
            "pts": pts,
            "pace": round(pace, 4),
            "projected_pts": round(projected, 1),
        })
    by_ovr = sorted(rows, key=lambda r: (-float(r["ovr"]), r["team_id"]))
    for i, r in enumerate(by_ovr, start=1):
        r["ovr_rank"] = i
    paced = [r for r in rows if int(r["gp"]) >= 10]
    by_pace = sorted(paced, key=lambda r: (-float(r["projected_pts"]), r["team_id"]))
    for i, r in enumerate(by_pace, start=1):
        r["pace_rank"] = i
    playoff_line = 0.0
    if len(by_pace) >= 16:
        playoff_line = float(by_pace[15]["projected_pts"])
    elif by_pace:
        playoff_line = float(by_pace[-1]["projected_pts"])
    n = max(1, len(rows))
    for r in rows:
        ovr_rank = int(r["ovr_rank"])
        if int(r["gp"]) >= 10 and r.get("pace_rank"):
            combined = (ovr_rank + int(r["pace_rank"])) / 2.0
            gap = playoff_line - float(r["projected_pts"])
        else:
            combined = float(ovr_rank)
            gap = 0.0
        r["strength_rank"] = int(round(combined))
        r["playoff_line_pts"] = round(playoff_line, 1)
        r["playoff_gap_pts"] = round(gap, 1)
        # Top group: combined rank in the top quarter, and not adrift of the playoff line.
        top_cut = max(6, min(8, n // 4))
        r["is_contender"] = bool(r["strength_rank"] <= top_cut and gap <= 4.0)
        r["label_status"] = _team_status_for_tank(
            next((tm for tm in teams if _team_id(tm) == r["team_id"]), None),
            standings,
        )
    return {str(r["team_id"]): r for r in rows}


def _note_tank_action(session: Any, team_id: str, action: str) -> None:
    bucket = getattr(session, "_tank_week_actions", None)
    if not isinstance(bucket, dict):
        bucket = {}
        setattr(session, "_tank_week_actions", bucket)
    bucket.setdefault(str(team_id), [])
    if action not in bucket[str(team_id)]:
        bucket[str(team_id)].append(action)


def flush_tank_week_log(session: Any, sim: Any, *, week: int, reason: str) -> None:
    """One line per club per week: label, strength, pressure, mode, pick, actions."""
    table = measure_strength_table(session, sim)
    actions = getattr(session, "_tank_week_actions", None) or {}
    tank_map = getattr(session, "transcendent_tank_pressure", None) or {}
    rows = []
    for tid, strength in table.items():
        payload = tank_map.get(tid) if isinstance(tank_map, dict) else {}
        payload = payload if isinstance(payload, dict) else {}
        taken = list(actions.get(tid) or [])
        row = {
            "week": int(week),
            "reason": reason,
            "team_id": tid,
            "label_status": strength.get("label_status"),
            "strength_rank": strength.get("strength_rank"),
            "ovr_rank": strength.get("ovr_rank"),
            "pace_rank": strength.get("pace_rank"),
            "is_contender": bool(strength.get("is_contender")),
            "playoff_gap_pts": strength.get("playoff_gap_pts"),
            "projected_pts": strength.get("projected_pts"),
            "pressure": int(payload.get("tank_pressure") or 0),
            "mode": str(payload.get("tank_mode") or "none"),
            "owns_own_first": bool(payload.get("owns_own_first", True)),
            "pick_ownership_reason": str(payload.get("pick_ownership_reason") or ""),
            "actions": taken,
        }
        rows.append(row)
        logger.info(
            "TANK week=%s team=%s status=%s strength_rank=%s contender=%s pressure=%s mode=%s owns_first=%s actions=%s",
            week,
            tid,
            row["label_status"],
            row["strength_rank"],
            row["is_contender"],
            row["pressure"],
            row["mode"],
            row["owns_own_first"],
            ",".join(taken) or "-",
        )
    log = getattr(session, "tank_debug_log", None)
    if not isinstance(log, list):
        log = []
        session.tank_debug_log = log
    log.extend(rows)
    setattr(session, "_tank_week_actions", {})


def clear_tank_scratches(teams: List[Any]) -> None:
    """Drop intentional tank scratches. Season rollover calls this."""
    for tm in teams or []:
        setattr(tm, "_tank_scratched_ids", set())


def maybe_refresh_tank_pressure(session: Any, sim: Any, *, reason: str) -> None:
    """Refresh when the week turns, the deadline arrives, or the roster/standings move."""
    if not getattr(session, "transcendent_draft_prospect_id", None):
        return
    league = getattr(sim, "league", None)
    if league is None:
        return
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    week = cursor // 7
    parts = []
    standings = getattr(session, "standings", None)
    for tm in getattr(league, "teams", None) or []:
        tid = _team_id(tm)
        if not tid:
            continue
        _gp, pts = _record_points(standings, tid)
        ids = tuple(sorted(str(getattr(p, "id", "") or "") for p in (getattr(tm, "roster", None) or [])))
        parts.append((tid, int(pts) // 4, hash(ids) & 0xFFFF))
    sig = tuple(parts)
    deadline_idx = getattr(league, "_trade_deadline_day_idx", None)
    on_deadline = deadline_idx is not None and cursor == int(deadline_idx)
    prev = getattr(session, "_tank_refresh_stamp", None)
    if prev == (week, sig) and not on_deadline and reason not in ("deadline", "roster"):
        return
    refresh_transcendent_tank_pressure(session, sim, transcendent_present=True)
    setattr(session, "_tank_refresh_stamp", (week, sig))


def _cap_hit(player: Any) -> float:
    try:
        from app.sim_engine.trades.team_assessment import cap_hit_m

        return float(cap_hit_m(player))
    except Exception:
        c = getattr(player, "contract", None)
        for obj in (c, player):
            if obj is None:
                continue
            for key in ("cap_hit_m", "aav_m", "aav"):
                raw = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
                if raw:
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        continue
        return 0.0


def _run_tank_week_moves(
    session: Any,
    sim: Any,
    teams: List[Any],
    news_out: List[Dict[str, Any]],
) -> None:
    """One pass a week: scratch expensive veterans, sign a cheap body, and put sellers on the market."""
    league = getattr(sim, "league", None)
    if league is None:
        return
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    if int(getattr(session, "_tank_roster_week", -1)) == cursor // 7:
        return
    setattr(session, "_tank_roster_week", cursor // 7)

    user_tid = str(getattr(session, "user_team_id", "") or "")
    before: Dict[str, Dict[str, float]] = {}
    for tm in teams:
        tid = _team_id(tm)
        if tid == user_tid:
            continue
        strength = getattr(tm, "_tank_strength", None) or {}
        mode = str(getattr(tm, "_franchise_tank_mode", "none") or "none")
        if strength.get("is_contender") or mode in ("none", ""):
            continue
        roster = [p for p in (getattr(tm, "roster", None) or []) if not getattr(p, "retired", False)]
        roster_ids = {str(getattr(p, "id", "") or "") for p in roster}
        scratched = set(str(x) for x in (getattr(tm, "_tank_scratched_ids", None) or [])) & roster_ids
        ovrs = sorted((_roster_ovr(tm),), reverse=True)
        median = ovrs[0] if ovrs else 75.0
        # Median of the skaters, not the top-12 average, so "below median" means the bottom half.
        raw_ovrs = []
        try:
            from app.sim_engine.trades.team_assessment import player_ovr

            raw_ovrs = sorted(float(player_ovr(p)) for p in roster if float(player_ovr(p)) > 0)
        except Exception:
            raw_ovrs = []
        if raw_ovrs:
            median = raw_ovrs[len(raw_ovrs) // 2]
        pricey = [
            p for p in roster
            if _player_age(p) >= 29 and _cap_hit(p) >= 3.5 and str(getattr(p, "id", "") or "") not in scratched
        ]
        pricey.sort(key=lambda p: (_cap_hit(p), _player_age(p)), reverse=True)
        if len(roster) >= 18:
            for vet in pricey[:2]:
                scratched.add(str(getattr(vet, "id", "") or ""))
            if pricey:
                _note_tank_action(session, tid, "scratch")
                record_tank_storyline(
                    session, tid, "TEAM_SCRATCHES_VETERAN",
                    f"{getattr(tm, 'name', tid)} scratches a veteran to dress cheaper bodies",
                )
                news_out.append({"type": "scratch", "headline": "Veteran scratched in lottery push", "team": tid, "priority": "LOW"})
        setattr(tm, "_tank_scratched_ids", scratched)
        before[tid] = {str(getattr(p, "id", "") or ""): _cap_hit(p) for p in roster}

        intent = str(getattr(tm, "_franchise_lottery_pick_intent", "") or "")
        if intent == "buy":
            _note_tank_action(session, tid, "bid")
            record_tank_storyline(
                session, tid, "TEAM_SHOPS_VETERANS",
                f"{getattr(tm, 'name', tid)} is bidding for a lottery pick",
            )

        if len(roster) >= 23:
            continue
        fas = list(getattr(league, "free_agents", None) or [])
        bums = []
        try:
            from app.sim_engine.trades.team_assessment import player_ovr

            for p in fas:
                if getattr(p, "retired", False):
                    continue
                ovr = float(player_ovr(p))
                if _player_age(p) < 27 or ovr <= 0 or ovr >= median:
                    continue
                bums.append(p)
            bums.sort(key=lambda p: (_cap_hit(p), player_ovr(p)))
        except Exception:
            bums = []
        if not bums:
            continue
        try:
            from services.contract_economy import LEAGUE_MINIMUM_AAV_M, sign_player_to_team

            season_year = int(getattr(session, "season_calendar_year", 2025) or 2025)
            result = sign_player_to_team(
                bums[0],
                tm,
                league,
                season_year,
                {"aav_m": float(LEAGUE_MINIMUM_AAV_M), "years": 1, "context": "ufa", "force": True},
            )
            if isinstance(result, dict) and result.get("ok"):
                fa_left = [p for p in fas if p is not bums[0]]
                try:
                    league.free_agents = fa_left
                except Exception:
                    pass
                _note_tank_action(session, tid, "sign")
                news_out.append({
                    "type": "signing",
                    "headline": f"{getattr(tm, 'name', tid)} signs a replacement-level body",
                    "team": tid,
                    "priority": "LOW",
                })
        except Exception:
            logger.debug("tank replacement signing failed", exc_info=True)

    try:
        from app.sim_engine.trades.cpu_trade_proposer import propose_and_execute_cpu_trades

        max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 0) or 0))
        propose_and_execute_cpu_trades(
            league,
            max_executions=6,
            calendar_cursor=cursor,
            regular_season_last_index=max_d,
        )
    except Exception:
        logger.debug("tank weekly market failed", exc_info=True)

    for tm in teams:
        tid = _team_id(tm)
        if tid not in before:
            continue
        after = {str(getattr(p, "id", "") or "") for p in (getattr(tm, "roster", None) or [])}
        for pid, hit in before[tid].items():
            if pid and pid not in after:
                _note_tank_action(session, tid, "dump" if hit >= 5.0 else "sell")
                record_tank_storyline(
                    session, tid, "TEAM_SELLS_RENTAL",
                    f"{getattr(tm, 'name', tid)} moved a veteran for futures",
                )


def apply_tank_daily_behavior(
    session: Any,
    sim: Any,
    teams: List[Any],
    rng: random.Random,
    news_out: List[Dict[str, Any]],
    counters: Dict[str, int],
) -> None:
    """Daily CPU reactions when a transcendent prospect exists."""
    if not getattr(session, "transcendent_draft_prospect_id", None):
        return
    year = int(getattr(session, "season_calendar_year", 0) or 0)
    if getattr(session, "_tank_scratch_season", None) not in (None, year):
        clear_tank_scratches(teams)
        _note_tank_action(session, "*", "scratch_clear")
    setattr(session, "_tank_scratch_season", year)
    maybe_refresh_tank_pressure(session, sim, reason="week")
    cursor = int(getattr(session, "calendar_cursor", 0) or 0)
    week = cursor // 7
    if int(getattr(session, "_tank_log_week", -1)) != week:
        _run_tank_week_moves(session, sim, teams, news_out)
        flush_tank_week_log(session, sim, week=week, reason="week")
        setattr(session, "_tank_log_week", week)

    for tm in teams:
        tid = _team_id(tm)
        row = get_tank_payload(session, tid)
        pressure = int(row.get("tank_pressure") or 0)
        mode = str(row.get("tank_mode") or "none")
        if pressure < 30 or mode == "none":
            continue
        strength = getattr(tm, "_tank_strength", None) or {}
        if strength.get("is_contender"):
            continue
        if not strength and str(_team_status_for_tank(tm, getattr(session, "standings", None))) == "playoff_contender":
            continue

        setattr(tm, "_franchise_tank_mode", mode)
        setattr(tm, "_franchise_tank_pressure", pressure)

        roster = list(getattr(tm, "roster", None) or [])
        veterans = [p for p in roster if _player_age(p) >= 30 and not getattr(p, "retired", False)]
        youth = [p for p in roster if _player_age(p) <= 22 and not getattr(p, "retired", False)]

        if pressure >= 50 and veterans and rng.random() < 0.08:
            record_tank_storyline(session, tid, "TEAM_SHOPS_VETERANS", f"{getattr(tm, 'name', tid)} listening on veteran pieces")
            news_out.append({"type": "tank", "headline": f"{getattr(tm, 'name', tid)} shopping veterans", "team": tid, "priority": "MEDIUM"})
            _note_tank_action(session, tid, "story_shop")

        if pressure >= 70 and veterans and rng.random() < 0.06:
            record_tank_storyline(session, tid, "TEAM_SELLS_RENTAL", f"{getattr(tm, 'name', tid)} open to moving a rental")
            counters["tank_sell_signals"] = int(counters.get("tank_sell_signals", 0)) + 1
            _note_tank_action(session, tid, "story_sell")

        if pressure >= 50 and youth and rng.random() < 0.10:
            record_tank_storyline(session, tid, "TEAM_PLAYS_YOUTH", f"{getattr(tm, 'name', tid)} giving youth a longer look")
            counters["tank_youth_plays"] = int(counters.get("tank_youth_plays", 0)) + 1
            _note_tank_action(session, tid, "story_youth")

        if mode == "hard_tank" and veterans and rng.random() < 0.05:
            vet = max(veterans, key=lambda p: _player_age(p))
            scratched = getattr(tm, "_tank_scratched_ids", None) or set()
            scratched = set(scratched)
            scratched.add(str(getattr(vet, "id", "")))
            setattr(tm, "_tank_scratched_ids", scratched)
            record_tank_storyline(session, tid, "TEAM_SCRATCHES_VETERAN", f"{getattr(tm, 'name', tid)} scratches veteran for youth push")
            news_out.append({"type": "scratch", "headline": "Veteran scratched in youth push", "team": tid, "priority": "LOW"})
            _note_tank_action(session, tid, "scratch")

        if pressure >= 90 and mode == "hard_tank" and rng.random() < 0.04:
            record_tank_storyline(session, tid, "TEAM_ENTERS_TANK_MODE", f"{getattr(tm, 'name', tid)} enters full lottery mode")
            counters["hard_tank_teams"] = int(counters.get("hard_tank_teams", 0)) + 1
            _note_tank_action(session, tid, "story_hard")

        if pressure >= 60 and _team_status_for_tank(tm) == "middling" and rng.random() < 0.05:
            record_tank_storyline(session, tid, "TEAM_ABORTS_PLAYOFF_PUSH", f"{getattr(tm, 'name', tid)} abandons playoff push")
            _note_tank_action(session, tid, "story_abort")


def tank_trade_interest_adjustment(team_id: str, context: Optional[Dict[str, Any]]) -> Tuple[float, List[str]]:
    """Boost seller interest when shopping veterans under transcendent tank pressure."""
    ctx = context or {}
    m = ctx.get("tank_pressure_by_team") or {}
    row = m.get(str(team_id)) or {}
    pressure = int(row.get("tank_pressure") or 0)
    mode = str(row.get("tank_mode") or "none")
    bonus = 0.0
    notes: List[str] = []
    if pressure >= 30:
        bonus += 0.04
    if pressure >= 50:
        bonus += 0.12
        notes.append("Transcendent lottery race — open to moving veterans")
    if pressure >= 70:
        bonus += 0.10
    if mode == "hard_tank" and row.get("owns_own_first"):
        bonus += 0.08
        notes.append("Hard tank mode — prioritizing future assets")
    if str(row.get("lottery_pick_intent") or "") == "buy":
        bonus += 0.12
        notes.append("Bidding for a lottery pick")
    return min(0.40, bonus), notes


def check_dev_league_generation_version(league: Any) -> Dict[str, Any]:
    """Warn when dev-league players predate body/team normalization."""
    stale = 0
    total = 0
    bad_names = 0
    for block in getattr(league, "development_leagues", None) or []:
        for tm in block.get("teams") or []:
            tname = str(tm.get("name") or "")
            if "EU_J" in tname.upper() or " CHL " in tname.upper():
                bad_names += 1
            for p in tm.get("players") or []:
                total += 1
                if int(getattr(p, "_spawn_version", 0) or 0) < DEV_LEAGUE_SPAWN_VERSION:
                    stale += 1
    needs_rebootstrap = stale > 0 or bad_names > 0
    return {
        "needs_rebootstrap": needs_rebootstrap,
        "stale_player_count": stale,
        "dev_player_total": total,
        "bad_team_name_count": bad_names,
        "expected_spawn_version": DEV_LEAGUE_SPAWN_VERSION,
        "warning": (
            "Development league was generated with an older prospect template. "
            "Start a new franchise or run dev rebootstrap for normalized bodies/team names."
            if needs_rebootstrap
            else None
        ),
    }


def rebootstrap_development_leagues(session: Any) -> Dict[str, Any]:
    """Dev-only: rebuild junior/college dev leagues in-place."""
    sim = getattr(session, "sim", None)
    league = getattr(sim, "league", None)
    if league is None or sim is None:
        return {"ok": False, "error": "no league"}
    rng = getattr(sim, "rng", None) or random.Random()
    from app.sim_engine.league_hierarchy_bootstrap import bootstrap_full_league_hierarchy

    old_dev = list(getattr(league, "development_leagues", None) or [])
    old_ids = set()
    for block in old_dev:
        for tm in block.get("teams") or []:
            for p in tm.get("players") or []:
                old_ids.add(id(p))
    league.players = [p for p in (getattr(league, "players", None) or []) if id(p) not in old_ids]
    bootstrap_full_league_hierarchy(league, rng)
    check = check_dev_league_generation_version(league)
    return {"ok": True, "rebootstrap": check}
