"""
Interactive Stanley Cup playoffs for franchise mode.

Maintains a live bracket with game-by-game series progress instead of
atomically simulating the entire postseason on enter.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.franchise_session import FranchiseSession
from services.json_safe import json_safe

try:
    from app.sim_engine.league.playoffs import playoff_game_win_probability
except Exception:  # pragma: no cover
    playoff_game_win_probability = None  # type: ignore


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rng_float(*parts: Any) -> float:
    raw = ":".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _needed_wins(best_of: int = 7) -> int:
    return (int(best_of) // 2) + 1


def _series_complete(row: Dict[str, Any]) -> bool:
    needed = _needed_wins(int(row.get("best_of") or 7))
    return int(row.get("wins_high") or 0) >= needed or int(row.get("wins_low") or 0) >= needed


def _winner_id(row: Dict[str, Any]) -> Optional[str]:
    if not _series_complete(row):
        return None
    wh = int(row.get("wins_high") or 0)
    wl = int(row.get("wins_low") or 0)
    return str(row.get("team_high_id") if wh > wl else row.get("team_low_id"))


def _loser_id(row: Dict[str, Any]) -> Optional[str]:
    w = _winner_id(row)
    if not w:
        return None
    hi = str(row.get("team_high_id") or "")
    lo = str(row.get("team_low_id") or "")
    return lo if w == hi else hi


def _home_is_high(game_number: int) -> bool:
    # 2-2-1-1-1: high seed hosts games 1,2,5,7
    return int(game_number) in (1, 2, 5, 7)


def _normalize_playoff_conference(conf: Any) -> str:
    text = str(conf or "").strip().lower()
    if "west" in text:
        return "West"
    if "east" in text:
        return "East"
    return str(conf).strip() if conf else "League"


def sanitize_first_round_matchups(matchups: Any) -> List[Dict[str, Any]]:
    """Keep only round-1 pairings and one series per team (no conference duplicates)."""
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in list(matchups or []):
        if not isinstance(raw, dict):
            continue
        try:
            rnd = int(raw.get("round_index") or 1)
        except (TypeError, ValueError):
            rnd = 1
        if rnd not in (0, 1):
            continue
        hi = str(raw.get("team_high_id") or raw.get("home_id") or "").strip()
        lo = str(raw.get("team_low_id") or raw.get("away_id") or "").strip()
        if not hi or not lo or hi == lo:
            continue
        if hi in seen or lo in seen:
            continue
        seen.add(hi)
        seen.add(lo)
        row = dict(raw)
        row["team_high_id"] = hi
        row["team_low_id"] = lo
        row["home_id"] = hi
        row["away_id"] = lo
        row["round_index"] = 1
        row["conference"] = _normalize_playoff_conference(raw.get("conference"))
        rows.append(row)
    return rows


def ensure_playoff_live(session: FranchiseSession) -> Dict[str, Any]:
    live = getattr(session, "playoff_live", None)
    if isinstance(live, dict) and live.get("started"):
        _ensure_calendar(live)
        if not isinstance(live.get("playoff_strength_map"), dict):
            _cache_playoff_strength_map(session, live)
        return live
    return start_live_playoffs(session)


def _ensure_calendar(live: Dict[str, Any]) -> None:
    """Backfill track + tip-off dates for older playoff saves."""
    if "playoff_day" not in live:
        live["playoff_day"] = 0
    active = [s for s in (live.get("series") or []) if s.get("status") == "active"]
    for s in active:
        if not s.get("schedule_track"):
            s["schedule_track"] = "mwf"
        if s.get("series_start_day") is None or s.get("scheduled_day") is None:
            start = int(s.get("series_start_day") if s.get("series_start_day") is not None else 0)
            _assign_series_schedule(s, track="mwf", start_day=start)
    # Both conferences tip on the same nights within a round.
    _sync_same_night_slate(live)


def _sync_same_night_slate(live: Dict[str, Any]) -> None:
    """
    Keep East + West (and all slots) on the same tip nights per round.

    Older saves staggered MWF/TTS by slot which made Sim Day look like
    one conference then the other.
    """
    day = int(live.get("playoff_day") or 0)
    by_round: Dict[int, List[Dict[str, Any]]] = {}
    for s in live.get("series") or []:
        if s.get("status") != "active":
            continue
        if not s.get("team_high_id") or not s.get("team_low_id"):
            continue
        rnd = int(s.get("round_index") or 0)
        by_round.setdefault(rnd, []).append(s)
    for rows in by_round.values():
        if not rows:
            continue
        tips = [int(r.get("scheduled_day") if r.get("scheduled_day") is not None else day) for r in rows]
        # If any series is due tonight (or overdue), pull the whole round onto today.
        if any(t <= day for t in tips):
            target = day
        else:
            target = min(tips)
        for r in rows:
            r["schedule_track"] = "mwf" if int(r.get("round_index") or 0) < 4 else "scf"
            next_g = max(1, int(r.get("next_game") or 1))
            if next_g <= 1:
                _assign_series_schedule(r, track=str(r["schedule_track"]), start_day=target)
            else:
                # Re-anchor remaining travel calendar so the next tip is `target`.
                r["series_start_day"] = int(target)
                # Walk remaining games from tonight using travel rests.
                dates = list(r.get("schedule_dates") or [])
                while len(dates) < 7:
                    dates.append(target)
                day_cursor = target
                for g in range(next_g, 8):
                    if g == next_g:
                        dates[g - 1] = day_cursor
                    else:
                        day_cursor = day_cursor + _rest_days_before_game(r, g) + 1
                        dates[g - 1] = day_cursor
                r["schedule_dates"] = dates[:7]
                r["scheduled_day"] = target



def _home_team_for_game(row: Dict[str, Any], game_number: int) -> str:
    hi = str(row.get("team_high_id") or "")
    lo = str(row.get("team_low_id") or "")
    return hi if _home_is_high(game_number) else lo


def _rest_days_before_game(row: Dict[str, Any], game_number: int) -> int:
    """
    Off-days AFTER the previous game before this tip-off.
    Travel (home changes) → 2 off days. Same rink → 1 off day.
    Stanley Cup Final → always 1 off day (compressed).
    """
    if int(game_number) <= 1:
        return 0
    if int(row.get("round_index") or 0) >= 4:
        return 1
    prev_home = _home_team_for_game(row, int(game_number) - 1)
    cur_home = _home_team_for_game(row, int(game_number))
    return 2 if prev_home != cur_home else 1


# Relative tip offsets from series_start_day (includes travel spacing).
# mwf ≈ Mon/Wed/Fri openers; tts ≈ Tue/Thu/Sun openers; scf = every-other-day Cup.
_TRACK_OFFSETS = {
    "mwf": (0, 2, 4, 7, 9, 12, 14),
    "tts": (1, 3, 6, 8, 11, 13, 15),
    "scf": (0, 2, 4, 6, 8, 10, 12),
}


def _series_game_day(row: Dict[str, Any], game_number: int) -> int:
    """Absolute playoff calendar day for this series' game N."""
    g = max(1, min(7, int(game_number)))
    start = int(row.get("series_start_day") or 0)
    if int(row.get("round_index") or 0) >= 4:
        offsets = _TRACK_OFFSETS["scf"]
    else:
        track = str(row.get("schedule_track") or "mwf").lower()
        offsets = _TRACK_OFFSETS.get(track) or _TRACK_OFFSETS["mwf"]
    # Prefer travel-aware cumulative schedule when home teams are known.
    if row.get("team_high_id") and row.get("team_low_id") and int(row.get("round_index") or 0) < 4:
        day = start
        for n in range(2, g + 1):
            day += _rest_days_before_game(row, n) + 1
        return day
    return start + int(offsets[g - 1])


def _assign_series_schedule(row: Dict[str, Any], *, track: str, start_day: int) -> None:
    row["schedule_track"] = track
    row["series_start_day"] = int(start_day)
    next_g = max(1, int(row.get("next_game") or 1))
    row["scheduled_day"] = _series_game_day(row, next_g)
    row["schedule_dates"] = [_series_game_day(row, g) for g in range(1, 8)]


def _schedule_next_game(live: Dict[str, Any], row: Dict[str, Any], *, rest_days: int = 1) -> None:
    """Recompute next tip from series travel calendar."""
    del rest_days
    if row.get("status") != "active":
        return
    if row.get("series_start_day") is None:
        day = int(live.get("playoff_day") or 0)
        _assign_series_schedule(row, track=str(row.get("schedule_track") or "mwf"), start_day=day)
        return
    next_g = max(1, int(row.get("next_game") or 1))
    row["scheduled_day"] = _series_game_day(row, next_g)
    row["schedule_dates"] = [_series_game_day(row, g) for g in range(1, 8)]


def _series_due_today(live: Dict[str, Any], row: Dict[str, Any]) -> bool:
    if row.get("status") != "active":
        return False
    return int(row.get("scheduled_day") or 0) == int(live.get("playoff_day") or 0)


def _due_active_series(live: Dict[str, Any], *, cpu_only: bool = False) -> List[Dict[str, Any]]:
    due = []
    for s in live.get("series") or []:
        if not _series_due_today(live, s):
            continue
        if cpu_only and s.get("is_user_series"):
            continue
        due.append(s)
    due.sort(
        key=lambda s: (
            0 if s.get("is_user_series") else 1,
            int(s.get("round_index") or 0),
            int(s.get("bracket_slot") or 0),
            str(s.get("series_id") or ""),
        )
    )
    return due


def _user_series_blocking(live: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for s in live.get("series") or []:
        if s.get("is_user_series") and _series_due_today(live, s):
            return s
    return None


def _slim_series(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "series_id": row.get("series_id"),
        "round_index": row.get("round_index"),
        "conference": row.get("conference"),
        "bracket_slot": row.get("bracket_slot"),
        "status": row.get("status"),
        "wins_high": row.get("wins_high"),
        "wins_low": row.get("wins_low"),
        "team_high_id": row.get("team_high_id"),
        "team_low_id": row.get("team_low_id"),
        "seed_high": row.get("seed_high"),
        "seed_low": row.get("seed_low"),
        "winner_id": row.get("winner_id"),
        "next_game": row.get("next_game"),
        "scheduled_day": row.get("scheduled_day"),
        "series_start_day": row.get("series_start_day"),
        "schedule_track": row.get("schedule_track"),
        "schedule_dates": list(row.get("schedule_dates") or [])[:7],
        "is_user_series": bool(row.get("is_user_series")),
        "loser_id": row.get("loser_id"),
        "series_score": row.get("series_score"),
        "game_log": list(row.get("game_log") or []),
    }


def slim_live_for_client(live: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(live, dict):
        return live
    return {
        "started": live.get("started"),
        "completed": live.get("completed"),
        "intro_seen": live.get("intro_seen"),
        "current_round": live.get("current_round"),
        "playoff_day": live.get("playoff_day"),
        "season_year": live.get("season_year"),
        "champion_id": live.get("champion_id"),
        "finalist_ids": list(live.get("finalist_ids") or []),
        "user_team_id": live.get("user_team_id"),
        "updated_at": live.get("updated_at"),
        "series": [_slim_series(s) for s in (live.get("series") or [])],
    }


def _play_series_game(
    session: FranchiseSession,
    live: Dict[str, Any],
    row: Dict[str, Any],
    *,
    respect_schedule: bool = True,
) -> Dict[str, Any]:
    if respect_schedule and not _series_due_today(live, row):
        raise ValueError(
            f"That series tips on day {int(row.get('scheduled_day') or 0) + 1}; "
            f"today is day {int(live.get('playoff_day') or 0) + 1}"
        )
    game = _simulate_one_game(session, row)
    if row.get("status") == "active":
        _schedule_next_game(live, row)
    return game


def start_live_playoffs(session: FranchiseSession) -> Dict[str, Any]:
    """Enter interactive playoffs from playoff_ready (does not crown a champion yet)."""
    existing = getattr(session, "playoff_live", None)
    if isinstance(existing, dict) and existing.get("started"):
        session.phase = "playoffs" if not session.playoffs_simulated else session.phase
        session.season_phase = session.phase
        return existing
    if session.playoffs_simulated and str(getattr(session, "phase", "")) in (
        "post_cup",
        "offseason",
        "preseason",
        "complete",
    ):
        return existing or {
            "started": True,
            "completed": True,
            "champion_id": session.champion_id,
        }

    payload = dict(getattr(session, "playoff_payload", None) or {})
    if not payload.get("first_round") and not payload.get("first_round_matchups"):
        from services.franchise_offseason import _build_playoff_payload

        payload = _build_playoff_payload(session)
        session.playoff_payload = payload

    # Never fall back to payload["series"] — after live sync that list includes R2/R3/Cup.
    r1 = sanitize_first_round_matchups(
        payload.get("first_round") or payload.get("first_round_matchups") or payload.get("matchups")
    )
    user_id = str(getattr(session, "user_team_id", "") or "")
    series_rows: List[Dict[str, Any]] = []

    # Split by conference for bracket slots
    by_conf: Dict[str, List[Dict[str, Any]]] = {}
    for i, m in enumerate(r1):
        conf = _normalize_playoff_conference(m.get("conference"))
        by_conf.setdefault(conf, []).append(dict(m))

    conf_names = sorted(by_conf.keys())
    for conf in conf_names:
        bucket = sorted(by_conf[conf], key=lambda s: int(s.get("seed_high") or 99))
        for slot, m in enumerate(bucket):
            hi = str(m.get("team_high_id") or m.get("home_id") or "")
            lo = str(m.get("team_low_id") or m.get("away_id") or "")
            row = {
                "series_id": f"{conf}-R1-{slot}",
                "round_index": 1,
                "conference": conf if conf != "League" else None,
                "bracket_slot": slot,
                "seed_high": int(m.get("seed_high") or 0),
                "seed_low": int(m.get("seed_low") or 0),
                "team_high_id": hi,
                "team_low_id": lo,
                "home_id": hi,
                "away_id": lo,
                "wins_high": 0,
                "wins_low": 0,
                "best_of": 7,
                "status": "active" if hi and lo else "pending",
                "game_log": [],
                "is_user_series": user_id in (hi, lo),
                "next_game": 1,
                "scheduled_day": 0,
            }
            series_rows.append(row)

    # Pre-create empty later-round slots
    for conf in conf_names:
        conf_key = conf if conf != "League" else None
        n_r1 = len([s for s in series_rows if s.get("conference") == conf_key and s["round_index"] == 1])
        # Round 2
        for slot in range(max(1, n_r1 // 2)):
            series_rows.append({
                "series_id": f"{conf}-R2-{slot}",
                "round_index": 2,
                "conference": conf_key,
                "bracket_slot": slot,
                "seed_high": slot + 1,
                "seed_low": n_r1 - slot,
                "team_high_id": "",
                "team_low_id": "",
                "wins_high": 0,
                "wins_low": 0,
                "best_of": 7,
                "status": "pending",
                "game_log": [],
                "is_user_series": False,
                "next_game": 1,
            })
        # Conference final
        series_rows.append({
            "series_id": f"{conf}-R3-0",
            "round_index": 3,
            "conference": conf_key,
            "bracket_slot": 0,
            "seed_high": 1,
            "seed_low": 2,
            "team_high_id": "",
            "team_low_id": "",
            "wins_high": 0,
            "wins_low": 0,
            "best_of": 7,
            "status": "pending",
            "game_log": [],
            "is_user_series": False,
            "next_game": 1,
        })

    # Stanley Cup Final slot
    series_rows.append({
        "series_id": "CUP-R4-0",
        "round_index": 4,
        "conference": None,
        "bracket_slot": 0,
        "seed_high": 1,
        "seed_low": 2,
        "team_high_id": "",
        "team_low_id": "",
        "wins_high": 0,
        "wins_low": 0,
        "best_of": 7,
        "status": "pending",
        "game_log": [],
        "is_user_series": False,
        "next_game": 1,
    })

    # Both conferences tip on the same nights (no East/West day stagger).
    active_openers = [s for s in series_rows if s.get("status") == "active"]
    for s in active_openers:
        _assign_series_schedule(s, track="mwf", start_day=0)

    live = {
        "started": True,
        "completed": False,
        "intro_seen": False,
        "current_round": 1,
        "playoff_day": 0,
        "season_year": int(getattr(session, "season_calendar_year", 2025) or 2025),
        "series": series_rows,
        "champion_id": None,
        "finalist_ids": [],
        "user_team_id": user_id,
        "updated_at": _now_iso(),
    }
    _cache_playoff_strength_map(session, live)
    session.playoff_live = live
    session.phase = "playoffs"
    session.season_phase = "playoffs"
    session.next_important_event = "playoffs"
    session.playoffs_generated = True
    _sync_payload_from_live(session, live)
    return live


def _sync_payload_from_live(session: FranchiseSession, live: Dict[str, Any]) -> None:
    payload = dict(getattr(session, "playoff_payload", None) or {})
    series = list(live.get("series") or [])
    r1 = [s for s in series if int(s.get("round_index") or 0) == 1]
    payload["series_list"] = series
    payload["series"] = series
    payload["first_round"] = r1
    payload["first_round_matchups"] = r1
    payload["matchups"] = r1
    payload["live"] = True
    payload["current_round"] = live.get("current_round")
    payload["playoff_day"] = live.get("playoff_day")
    payload["champion_id"] = live.get("champion_id")
    payload["finalist_ids"] = list(live.get("finalist_ids") or [])
    payload["completed"] = bool(live.get("completed"))
    session.playoff_payload = payload


def _strength_for(session: FranchiseSession, team_id: str) -> float:
    sm = getattr(session, "strength_map", None) or {}
    try:
        return float(sm.get(str(team_id), 0.5))
    except Exception:
        return 0.5


def _xg_share_for(session: FranchiseSession, team_id: str) -> Optional[float]:
    """Season xGF share from regular-season box scores, if enough sample exists."""
    tid = str(team_id or "")
    if not tid:
        return None
    xgf = 0.0
    xga = 0.0
    n = 0
    for game in list(getattr(session, "game_results", None) or []):
        if not isinstance(game, dict):
            continue
        if str(game.get("stat_scope") or "regular_season") != "regular_season":
            continue
        hid = str(game.get("home_id") or game.get("home_team_id") or "")
        aid = str(game.get("away_id") or game.get("away_team_id") or "")
        try:
            hx = float(game.get("home_xgf") or game.get("home_xg") or 0)
            ax = float(game.get("away_xgf") or game.get("away_xg") or 0)
        except (TypeError, ValueError):
            continue
        if hx <= 0 and ax <= 0:
            continue
        if hid == tid:
            xgf += hx
            xga += ax
            n += 1
        elif aid == tid:
            xgf += ax
            xga += hx
            n += 1
    tot = xgf + xga
    if n < 10 or tot < 20.0:
        return None
    return xgf / tot


def _effective_playoff_strength(
    session: FranchiseSession,
    team_id: str,
    *,
    use_cache: bool = True,
) -> float:
    """Roster strength + regular-season record + xG, cached on the live bracket."""
    tid = str(team_id or "")
    if use_cache:
        live = getattr(session, "playoff_live", None)
        if isinstance(live, dict):
            cached = live.get("playoff_strength_map")
            if isinstance(cached, dict) and tid in cached:
                try:
                    return float(cached[tid])
                except (TypeError, ValueError):
                    pass

    base = _strength_for(session, tid)
    standings = getattr(session, "standings", None)
    rec = None
    if standings is not None:
        recs = getattr(standings, "records", None) or {}
        rec = recs.get(tid) or recs.get(team_id)
    if rec is not None:
        gp = max(1, int(getattr(rec, "gp", 0) or 0))
        pts = float(getattr(rec, "points", 0) or 0)
        gf = float(getattr(rec, "gf", 0) or 0)
        ga = float(getattr(rec, "ga", 0) or 0)
        base += 0.62 * ((pts / (2.0 * gp)) - 0.5)
        base += 0.038 * max(-2.0, min(2.0, (gf - ga) / gp))
    xg = _xg_share_for(session, tid)
    if xg is not None:
        base += 0.90 * (xg - 0.5)
    return max(0.05, min(0.99, base))


def _cache_playoff_strength_map(session: FranchiseSession, live: Dict[str, Any]) -> None:
    ids = []
    for row in list(live.get("series") or []):
        for key in ("team_high_id", "team_low_id"):
            tid = str(row.get(key) or "")
            if tid:
                ids.append(tid)
    smap = {tid: _effective_playoff_strength(session, tid, use_cache=False) for tid in ids}
    live["playoff_strength_map"] = smap


def _live_game_p_high(session: FranchiseSession, hi: str, lo: str) -> float:
    s_hi = _effective_playoff_strength(session, hi)
    s_lo = _effective_playoff_strength(session, lo)
    if callable(playoff_game_win_probability):
        return float(playoff_game_win_probability(s_hi, s_lo))
    diff = max(-0.55, min(0.55, s_hi - s_lo))
    return max(0.22, min(0.88, 0.5 + diff * 1.15))


def _simulate_one_game(
    session: FranchiseSession,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    if _series_complete(row) or row.get("status") == "pending":
        raise ValueError("Series is not active")
    hi = str(row.get("team_high_id") or "")
    lo = str(row.get("team_low_id") or "")
    if not hi or not lo:
        raise ValueError("Series teams not set")

    game_no = len(row.get("game_log") or []) + 1
    home_high = _home_is_high(game_no)
    home_id = hi if home_high else lo
    away_id = lo if home_high else hi

    p_high = _live_game_p_high(session, hi, lo)
    p_home = p_high if home_high else (1.0 - p_high)
    p_home = p_home + 0.035 if home_high else p_home - 0.035
    p_home = max(0.16, min(0.90, p_home))

    roll = _rng_float(
        session.session_id,
        row.get("series_id"),
        game_no,
        getattr(session, "season_calendar_year", 0),
        int(row.get("wins_high") or 0),
        int(row.get("wins_low") or 0),
    )
    home_wins = roll < p_home
    # Score generation
    base = 2 + int(_rng_float(session.session_id, row.get("series_id"), game_no, "s") * 3)
    margin = 1 + int(_rng_float(session.session_id, row.get("series_id"), game_no, "m") * 3)
    if home_wins:
        hs, as_ = base + margin, base
    else:
        hs, as_ = base, base + margin
    ot = False
    if abs(hs - as_) == 1 and _rng_float(session.session_id, row.get("series_id"), game_no, "ot") > 0.72:
        ot = True
        if home_wins:
            hs, as_ = max(hs, as_ + 1), as_
        else:
            as_, hs = max(as_, hs + 1), hs

    winner_id = home_id if home_wins else away_id
    if winner_id == hi:
        row["wins_high"] = int(row.get("wins_high") or 0) + 1
    else:
        row["wins_low"] = int(row.get("wins_low") or 0) + 1

    day = 0
    live = getattr(session, "playoff_live", None)
    if isinstance(live, dict):
        day = int(live.get("playoff_day") or 0)

    entry = {
        "game": game_no,
        "home_id": home_id,
        "away_id": away_id,
        "home_score": int(hs),
        "away_score": int(as_),
        "ot": ot,
        "winner_id": winner_id,
        "played_at": _now_iso(),
        "playoff_day": day,
    }
    log = list(row.get("game_log") or [])
    log.append(entry)
    row["game_log"] = log
    row["next_game"] = game_no + 1
    row["status"] = "complete" if _series_complete(row) else "active"
    if row["status"] == "complete":
        row["series_score"] = f"{row.get('wins_high')}-{row.get('wins_low')}"
        row["winner_id"] = _winner_id(row)
        row["loser_id"] = _loser_id(row)
    return entry


def _sort_winners_by_standings(session: FranchiseSession, team_ids: List[str]) -> List[str]:
    standings = getattr(session, "standings", None)
    if standings is None:
        return list(team_ids)
    try:
        scored: List[Tuple[int, int, int, str]] = []
        for tid in team_ids:
            rec = getattr(standings, "records", {}).get(tid) or getattr(standings, "records", {}).get(str(tid))
            if rec is None:
                return list(team_ids)
            pts = int(getattr(rec, "points", 0) or 0)
            wins = int(getattr(rec, "wins", 0) or 0)
            gd_attr = getattr(rec, "goal_diff", None)
            try:
                gd = int(gd_attr() if callable(gd_attr) else (gd_attr or 0))
            except Exception:
                gd = int(getattr(rec, "gf", 0) or 0) - int(getattr(rec, "ga", 0) or 0)
            scored.append((pts, wins, gd, str(tid)))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [t[3] for t in scored]
    except Exception:
        return list(team_ids)


def _try_advance_bracket(session: FranchiseSession, live: Dict[str, Any]) -> None:
    series = list(live.get("series") or [])
    user_id = str(live.get("user_team_id") or session.user_team_id or "")

    # Conferences present in R1
    confs = sorted({
        str(s.get("conference") or "League")
        for s in series
        if int(s.get("round_index") or 0) == 1
    })

    for conf in confs:
        conf_key = None if conf == "League" else conf
        r1 = [
            s for s in series
            if int(s.get("round_index") or 0) == 1
            and (s.get("conference") == conf_key or (conf == "League" and not s.get("conference")))
        ]
        if not r1 or not all(_series_complete(s) for s in r1):
            continue
        winners = _sort_winners_by_standings(session, [_winner_id(s) for s in r1 if _winner_id(s)])
        r2 = [
            s for s in series
            if int(s.get("round_index") or 0) == 2
            and (s.get("conference") == conf_key or (conf == "League" and not s.get("conference")))
        ]
        r2 = sorted(r2, key=lambda s: int(s.get("bracket_slot") or 0))
        # Pair 1v4, 2v3
        pairs = []
        if len(winners) >= 4:
            pairs = [(winners[0], winners[3]), (winners[1], winners[2])]
        elif len(winners) >= 2:
            pairs = [(winners[0], winners[1])]
        for i, (a, b) in enumerate(pairs):
            if i >= len(r2):
                break
            slot = r2[i]
            if slot.get("status") != "pending" and slot.get("team_high_id"):
                continue
            # Higher seed by standings is high
            ordered = _sort_winners_by_standings(session, [a, b])
            slot["team_high_id"] = ordered[0]
            slot["team_low_id"] = ordered[1]
            slot["home_id"] = ordered[0]
            slot["away_id"] = ordered[1]
            slot["status"] = "active"
            slot["is_user_series"] = user_id in (ordered[0], ordered[1])
            slot["wins_high"] = 0
            slot["wins_low"] = 0
            slot["game_log"] = []
            slot["next_game"] = 1
            day0 = int(live.get("playoff_day") or 0)
            _assign_series_schedule(slot, track="mwf", start_day=day0)

        # Conference final
        r2_active = [
            s for s in series
            if int(s.get("round_index") or 0) == 2
            and (s.get("conference") == conf_key or (conf == "League" and not s.get("conference")))
        ]
        if r2_active and all(_series_complete(s) for s in r2_active):
            w2 = _sort_winners_by_standings(session, [_winner_id(s) for s in r2_active if _winner_id(s)])
            cf = next(
                (
                    s for s in series
                    if int(s.get("round_index") or 0) == 3
                    and (s.get("conference") == conf_key or (conf == "League" and not s.get("conference")))
                ),
                None,
            )
            if cf and len(w2) >= 2 and (cf.get("status") == "pending" or not cf.get("team_high_id")):
                cf["team_high_id"] = w2[0]
                cf["team_low_id"] = w2[1]
                cf["home_id"] = w2[0]
                cf["away_id"] = w2[1]
                cf["status"] = "active"
                cf["is_user_series"] = user_id in (w2[0], w2[1])
                cf["wins_high"] = 0
                cf["wins_low"] = 0
                cf["game_log"] = []
                cf["next_game"] = 1
                day0 = int(live.get("playoff_day") or 0)
                _assign_series_schedule(cf, track="mwf", start_day=day0)

    # Cup Final
    conf_finals = [s for s in series if int(s.get("round_index") or 0) == 3]
    if conf_finals and all(_series_complete(s) for s in conf_finals):
        champs = _sort_winners_by_standings(session, [_winner_id(s) for s in conf_finals if _winner_id(s)])
        cup = next((s for s in series if int(s.get("round_index") or 0) == 4), None)
        if cup and len(champs) >= 2 and (cup.get("status") == "pending" or not cup.get("team_high_id")):
            cup["team_high_id"] = champs[0]
            cup["team_low_id"] = champs[1]
            cup["home_id"] = champs[0]
            cup["away_id"] = champs[1]
            cup["status"] = "active"
            cup["is_user_series"] = user_id in (champs[0], champs[1])
            cup["wins_high"] = 0
            cup["wins_low"] = 0
            cup["game_log"] = []
            cup["next_game"] = 1
            live["finalist_ids"] = [champs[0], champs[1]]
            day0 = int(live.get("playoff_day") or 0)
            _assign_series_schedule(cup, track="mwf", start_day=day0)

    cup = next((s for s in series if int(s.get("round_index") or 0) == 4), None)
    if cup and _series_complete(cup):
        live["champion_id"] = _winner_id(cup)
        live["finalist_ids"] = [str(cup.get("team_high_id")), str(cup.get("team_low_id"))]
        live["completed"] = True

    # Current round = lowest unfinished active/pending with teams
    active_rounds = [
        int(s.get("round_index") or 0)
        for s in series
        if s.get("status") == "active" or (s.get("status") == "pending" and s.get("team_high_id"))
    ]
    if active_rounds:
        live["current_round"] = min(active_rounds)
    elif live.get("completed"):
        live["current_round"] = 4

    live["series"] = series
    live["updated_at"] = _now_iso()


def _find_series(live: Dict[str, Any], series_id: str) -> Dict[str, Any]:
    for s in live.get("series") or []:
        if str(s.get("series_id")) == str(series_id):
            return s
    raise ValueError(f"Unknown series: {series_id}")


def get_playoff_hub_payload(session: FranchiseSession) -> Dict[str, Any]:
    live = ensure_playoff_live(session) if str(getattr(session, "phase", "")) in (
        "playoffs",
        "playoff_ready",
    ) else (getattr(session, "playoff_live", None) or {})
    if not live and str(getattr(session, "phase", "")) == "playoff_ready":
        # preview from static payload without mutating to live until enter
        payload = dict(getattr(session, "playoff_payload", None) or {})
        return {
            "mode": "ready",
            "payload": payload,
            "live": None,
        }
    return {
        "mode": "live" if live.get("started") else "ready",
        "live": slim_live_for_client(live) if live else None,
        "payload": {
            "playoff_day": (live or {}).get("playoff_day"),
            "current_round": (live or {}).get("current_round"),
            "champion_id": (live or {}).get("champion_id") or session.champion_id,
            "completed": bool((live or {}).get("completed")),
        },
        "champion_id": (live or {}).get("champion_id") or session.champion_id,
        "phase": str(getattr(session, "phase", "")),
    }


def mark_intro_seen(session: FranchiseSession) -> Dict[str, Any]:
    live = ensure_playoff_live(session)
    live["intro_seen"] = True
    session.playoff_live = live
    return {"ok": True, "intro_seen": True}


def _maybe_finish(session: FranchiseSession, live: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not live.get("completed") and not live.get("champion_id"):
        return None
    try:
        return finish_live_playoffs(session)
    except Exception as exc:  # noqa: BLE001 — never fail the client on Cup night
        champ = str(live.get("champion_id") or session.champion_id or "")
        session.playoffs_simulated = True
        session.playoffs_done = True
        session.phase = "post_cup"
        session.season_phase = "post_cup"
        session.champion_id = champ or session.champion_id
        session.stanley_cup_winner = session.champion_id
        session.next_important_event = "awards"
        return {
            "status": "post_cup",
            "season_phase": "post_cup",
            "champion_id": session.champion_id,
            "finish_error": str(exc),
        }


def handle_playoff_action(
    session: FranchiseSession,
    action: str,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Actions:
      enter | sim_game | play_user_game | sim_user_game | sim_series |
      sim_round | sim_cpu_games | advance_day | sim_rest | finish
    """
    from services.franchise_sim import invalidate_session_payload_caches

    body = body or {}
    action = str(action or "").strip().lower().replace("-", "_")

    if action in ("enter", "start"):
        live = start_live_playoffs(session)
        invalidate_session_payload_caches(session, "playoffs_enter")
        return {
            "ok": True,
            "action": "enter",
            "playoff_day": live.get("playoff_day"),
            "playoff": get_playoff_hub_payload(session),
        }

    if action == "mark_intro_seen":
        return mark_intro_seen(session)

    # Already crowned — idempotent no-op for sim actions.
    if getattr(session, "playoffs_simulated", False) and str(getattr(session, "phase", "")) in (
        "post_cup",
        "offseason",
        "preseason",
        "complete",
    ):
        return {
            "ok": True,
            "action": action,
            "already_complete": True,
            "champion_id": session.champion_id,
            "playoff": get_playoff_hub_payload(session),
        }

    live = ensure_playoff_live(session)
    # Refresh user series flags (team_id type mismatches can drop them).
    uid = str(live.get("user_team_id") or getattr(session, "user_team_id", "") or "")
    for s in live.get("series") or []:
        hi = str(s.get("team_high_id") or "")
        lo = str(s.get("team_low_id") or "")
        s["is_user_series"] = bool(uid) and uid in (hi, lo)

    series_id = str(body.get("series_id") or body.get("seriesId") or "")

    if action in ("sim_game", "play_user_game", "sim_user_game"):
        if not series_id:
            user_due = _user_series_blocking(live)
            user_active = [
                s for s in live.get("series") or []
                if s.get("is_user_series") and s.get("status") == "active"
            ]
            if user_due:
                series_id = str(user_due.get("series_id"))
            elif len(user_active) == 1:
                series_id = str(user_active[0].get("series_id"))
            else:
                raise ValueError("series_id required")
        row = _find_series(live, series_id)
        if action in ("play_user_game", "sim_user_game") and not row.get("is_user_series"):
            raise ValueError("Not your series")
        # User may always play/sim their own series even if slightly off calendar
        # (missed tip while CPU day advanced). Auto-align to today.
        respect = True
        if row.get("is_user_series"):
            row["scheduled_day"] = int(live.get("playoff_day") or 0)
            respect = False
        elif not _series_due_today(live, row):
            raise ValueError(
                f"That series tips on day {int(row.get('scheduled_day') or 0) + 1}; "
                f"today is day {int(live.get('playoff_day') or 0) + 1}"
            )
        game = _play_series_game(session, live, row, respect_schedule=respect)
        _try_advance_bracket(session, live)
        session.playoff_live = live
        _sync_payload_from_live(session, live)
        finish = _maybe_finish(session, live)
        invalidate_session_payload_caches(session, "playoffs_sim")
        return {
            "ok": True,
            "action": action,
            "game": game,
            "series": _slim_series(row),
            "playoff_day": live.get("playoff_day"),
            "finish": finish,
            "playoff": get_playoff_hub_payload(session),
        }

    if action == "sim_series":
        _sync_same_night_slate(live)
        if not series_id:
            # Prefer user series, then selected-night series, then any active.
            pick = _user_series_blocking(live)
            if pick is None:
                due = _due_active_series(live, cpu_only=False)
                pick = due[0] if due else None
            if pick is None:
                pick = next(
                    (s for s in (live.get("series") or []) if s.get("status") == "active" and s.get("team_high_id")),
                    None,
                )
            if pick is None:
                raise ValueError("No active series to sim")
            series_id = str(pick.get("series_id") or "")
        row = _find_series(live, series_id)
        if row.get("status") != "active":
            raise ValueError("That series is not active")
        if not row.get("team_high_id") or not row.get("team_low_id"):
            raise ValueError("Series teams not set yet")
        games = []
        safety = 0
        while row.get("status") == "active" and safety < 7:
            safety += 1
            # Explicit Sim Series ignores calendar — finish the series now.
            games.append(_play_series_game(session, live, row, respect_schedule=False))
        _try_advance_bracket(session, live)
        _sync_same_night_slate(live)
        session.playoff_live = live
        _sync_payload_from_live(session, live)
        finish = _maybe_finish(session, live)
        invalidate_session_payload_caches(session, "playoffs_sim")
        return {
            "ok": True,
            "action": action,
            "games": games,
            "series": _slim_series(row),
            "playoff_day": live.get("playoff_day"),
            "finish": finish,
            "playoff": get_playoff_hub_payload(session),
        }

    if action == "sim_cpu_games":
        # Play every CPU tip-off scheduled for TODAY. Never skip past a user game night.
        _sync_same_night_slate(live)
        blocker = _user_series_blocking(live)
        if blocker:
            return {
                "ok": False,
                "action": action,
                "blocked": True,
                "reason": "Your series tips tonight — Play/Sim your game first.",
                "series": _slim_series(blocker),
                "playoff_day": live.get("playoff_day"),
                "playoff": get_playoff_hub_payload(session),
            }
        games = []
        for row in list(_due_active_series(live, cpu_only=True)):
            games.append({
                "series_id": row.get("series_id"),
                "game": _play_series_game(session, live, row, respect_schedule=True),
            })
        # Advance calendar one day after clearing tonight's CPU slate (or empty night).
        live["playoff_day"] = int(live.get("playoff_day") or 0) + 1
        _try_advance_bracket(session, live)
        session.playoff_live = live
        _sync_payload_from_live(session, live)
        finish = _maybe_finish(session, live)
        invalidate_session_payload_caches(session, "playoffs_sim")
        return {
            "ok": True,
            "action": action,
            "games": games,
            "playoff_day": live.get("playoff_day"),
            "finish": finish,
            "playoff": get_playoff_hub_payload(session),
        }

    if action == "advance_day":
        # Play every series scheduled tonight (both conferences), then advance one day.
        _sync_same_night_slate(live)
        games = []
        due = _due_active_series(live, cpu_only=False)
        note = None
        if not due:
            note = "Off day — no tips tonight"
        for row in list(due):
            games.append({
                "series_id": row.get("series_id"),
                "game": _play_series_game(session, live, row, respect_schedule=True),
            })
        live["playoff_day"] = int(live.get("playoff_day") or 0) + 1
        _try_advance_bracket(session, live)
        _sync_same_night_slate(live)
        session.playoff_live = live
        _sync_payload_from_live(session, live)
        finish = _maybe_finish(session, live)
        invalidate_session_payload_caches(session, "playoffs_sim")
        return {
            "ok": True,
            "action": action,
            "games": games,
            "note": note,
            "playoff_day": live.get("playoff_day"),
            "finish": finish,
            "playoff": get_playoff_hub_payload(session),
        }

    if action == "sim_round":
        games = []
        current = int(live.get("current_round") or 1)
        safety = 0
        while safety < 120:
            safety += 1
            active_round = [
                s for s in (live.get("series") or [])
                if s.get("status") == "active" and int(s.get("round_index") or 0) == current
            ]
            if not active_round:
                break
            # Advance calendar until round is complete.
            due = _due_active_series(live, cpu_only=False)
            if due:
                for row in list(due):
                    if int(row.get("round_index") or 0) != current:
                        continue
                    games.append({
                        "series_id": row.get("series_id"),
                        "game": _play_series_game(session, live, row, respect_schedule=True),
                    })
            live["playoff_day"] = int(live.get("playoff_day") or 0) + 1
            _try_advance_bracket(session, live)
        session.playoff_live = live
        _sync_payload_from_live(session, live)
        finish = _maybe_finish(session, live)
        invalidate_session_payload_caches(session, "playoffs_sim")
        return {
            "ok": True,
            "action": action,
            "games_played": len(games),
            "finish": finish,
            "playoff": get_playoff_hub_payload(session),
        }

    if action in ("sim_rest", "finish", "complete"):
        safety = 0
        while not live.get("completed") and safety < 400:
            safety += 1
            due = _due_active_series(live, cpu_only=False)
            progressed = False
            for row in list(due):
                _play_series_game(session, live, row, respect_schedule=True)
                progressed = True
            if not progressed:
                active = [s for s in (live.get("series") or []) if s.get("status") == "active"]
                if not active:
                    _try_advance_bracket(session, live)
                    if not any(s.get("status") == "active" for s in live.get("series") or []):
                        if live.get("completed"):
                            break
                        # force-sim any active that somehow isn't dated
                        stuck = [s for s in (live.get("series") or []) if s.get("status") == "active"]
                        if not stuck:
                            break
                        for row in stuck:
                            _simulate_one_game(session, row)
                            if row.get("status") == "active":
                                _schedule_next_game(live, row)
                live["playoff_day"] = int(live.get("playoff_day") or 0) + 1
            else:
                live["playoff_day"] = int(live.get("playoff_day") or 0) + 1
            _try_advance_bracket(session, live)
        session.playoff_live = live
        _sync_payload_from_live(session, live)
        finish = _maybe_finish(session, live)
        invalidate_session_payload_caches(session, "playoffs_complete")
        return {
            "ok": True,
            "action": action,
            "finish": finish,
            "champion_id": live.get("champion_id") or session.champion_id,
            "playoff": get_playoff_hub_payload(session),
        }

    raise ValueError(f"Unknown playoff action: {action}")


def finish_live_playoffs(session: FranchiseSession) -> Dict[str, Any]:
    """Crown champion from live state and run awards / post_cup transition."""
    from services.franchise_offseason import complete_playoffs_from_live_result

    live = getattr(session, "playoff_live", None) or {}
    if not live.get("champion_id"):
        cup = next((s for s in live.get("series") or [] if int(s.get("round_index") or 0) == 4), None)
        if cup and _series_complete(cup):
            live["champion_id"] = _winner_id(cup)
            live["finalist_ids"] = [str(cup.get("team_high_id")), str(cup.get("team_low_id"))]
            live["completed"] = True
            session.playoff_live = live

    if not live.get("champion_id"):
        raise ValueError("Playoffs not complete — no champion yet")

    if session.playoffs_simulated and str(getattr(session, "phase", "")) in (
        "post_cup",
        "offseason",
        "preseason",
        "complete",
    ):
        return {
            "status": "post_cup",
            "season_phase": str(getattr(session, "phase", "post_cup")),
            "champion_id": session.champion_id,
            "already_done": True,
        }

    return complete_playoffs_from_live_result(session, live)
