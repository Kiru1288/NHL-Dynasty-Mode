"""Player-initiated trade demands driven by Trade Stability Score + real-time crisis timer."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise.player_agent_engine import (
    agent_crisis_initial_days,
    agent_crisis_initial_seconds,
    agent_destination_shrink_factor,
    agent_leak_probability,
    agent_public_view,
    assign_league_agents,
    ensure_player_agent,
    get_agent_gm_relationship,
)
from app.sim_engine.franchise.trade_stability_engine import CRISIS_DEADLINE_MAX, apply_daily_stability_update, apply_trade_hub_exposure, clear_demand_temporary_modifiers, crisis_distressed_asset_cost, crisis_stage_from_remaining, crisis_trade_value_multiplier, ensure_player_storyline_state, formal_demand_eligible, _player_character_0_100, primary_complaint_from_pressures, update_player_stability
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

#: Depth players (70+) can now sour too — was 74, which excluded most bottom-six.
DEMAND_MIN_OVR = 70
#: CPU demands age in sim days (the real-time crisis clock only runs for the user's club).
CPU_DEMAND_STAGE2_DAYS = 12
CPU_DEMAND_STAGE3_DAYS = 26
#: Same player cannot open another formal demand until this many sim days after the last closed.
DEMAND_REOPEN_COOLDOWN_DAYS = 50
#: Cap new formal demands per league pass (bulk catch-up used to open dozens at once).
MAX_NEW_FORMAL_DEMANDS_PER_PASS = 3
#: A club carries at most this many open formal demands at once — real rooms rarely have two.
MAX_OPEN_DEMANDS_PER_TEAM = 1
#: …and opens at most this many over a rolling season window.
MAX_DEMANDS_PER_TEAM_WINDOW = 2
DEMAND_TEAM_WINDOW_DAYS = 200
CALENDAR_DAYS_PER_SEASON = 400
DEPTH_DEMAND_MAX_OVR = 78.0
CPU_DEMAND_STAGE3_STALE_DAYS = 21
DEMAND_OPEN_BUDGET_PER_DAY = 0.45
_ORG_POOL_ATTRS = ("roster", "ahl_roster", "echl_roster")

CANADA_ABBRS = frozenset({"TOR", "MTL", "OTT", "VAN", "CGY", "EDM", "WPG"})
SMALL_MARKET_ABBRS = frozenset(
    {"BUF", "OTT", "CBJ", "CGY", "WPG", "ARI", "UTA", "SEA", "NSH", "MIN", "CAR", "FLA"}
)


def _save_seed(session: Any) -> int:
    for key in ("save_seed", "franchise_seed", "rng_seed"):
        raw = getattr(session, key, None)
        if raw is not None:
            try:
                return int(raw) & 0xFFFFFFFF
            except (TypeError, ValueError):
                continue
    fid = str(getattr(session, "franchise_id", "") or getattr(session, "id", "") or "")
    if fid:
        import zlib

        return zlib.crc32(fid.encode()) & 0xFFFFFFFF
    return 0


def absolute_calendar_day(session: Any, calendar_idx: int) -> int:
    """Monotonic day index for cooldowns (survives season calendar_idx resets)."""
    season = int(getattr(session, "season_calendar_year", 0) or getattr(session, "season", 0) or 0)
    return season * CALENDAR_DAYS_PER_SEASON + int(calendar_idx)


def _calendar_day_meta(session: Any) -> Dict[str, Any]:
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    cal = getattr(session, "nhl_calendar", None) or []
    if 0 <= cur < len(cal) and isinstance(cal[cur], dict):
        return dict(cal[cur])
    return {}


def get_trade_deadline_context(session: Any) -> Dict[str, Any]:
    """Trade-deadline window vs when formal demands may open (includes summer)."""
    phase = str(getattr(session, "phase", "") or getattr(session, "season_phase", "") or "").lower()
    day = _calendar_day_meta(session)
    iso = str(day.get("iso") or day.get("date") or "")
    tags = tuple(day.get("tags") or ())
    in_window = "trade_deadline" in tags
    in_season = phase in ("", "regular", "regular_season", "in_season")
    in_playoffs = phase in ("playoffs",)
    in_summer = phase in ("offseason", "preseason", "post_cup")
    past_in_season_deadline = False
    deadline_iso = ""

    if iso and len(iso) >= 10:
        try:
            y, m, d = int(iso[0:4]), int(iso[5:7]), int(iso[8:10])
            deadline_year = y + 1 if m >= 7 else y
            deadline_iso = f"{deadline_year}-03-10"
            if in_season or in_playoffs:
                if (m == 3 and d > 10) or (in_playoffs and m >= 3):
                    past_in_season_deadline = True
                elif 4 <= m <= 9 and in_season:
                    past_in_season_deadline = True
            if m == 2 and d >= 25:
                in_window = True
            elif m == 3 and d <= 10:
                in_window = True
        except (TypeError, ValueError):
            pass

    days_to_deadline = None
    if iso and deadline_iso and in_season and not past_in_season_deadline:
        try:
            from datetime import date

            cur_d = date(int(iso[0:4]), int(iso[5:7]), int(iso[8:10]))
            dl = date(int(deadline_iso[0:4]), int(deadline_iso[5:7]), int(deadline_iso[8:10]))
            days_to_deadline = max(0, (dl - cur_d).days)
        except (TypeError, ValueError):
            days_to_deadline = None

    new_demands_allowed = in_summer or (in_season and not past_in_season_deadline)
    trades_frozen = past_in_season_deadline or in_playoffs

    return {
        "in_window": bool(in_window),
        "past_deadline": bool(trades_frozen),
        "past_in_season_deadline": bool(past_in_season_deadline),
        "deadline_iso": deadline_iso or "03-10",
        "days_to_deadline": days_to_deadline,
        "crisis_timer_ticks": in_season and not past_in_season_deadline,
        "new_demands_allowed": bool(new_demands_allowed),
        "summer_demand_window": bool(in_summer),
        "demand_urgency_mult": 1.18 if in_window and in_season and not past_in_season_deadline else 1.0,
    }


def _resume_summer_pending_demands(session: Any, calendar_idx: int, iso_date: str = "") -> int:
    ctx = get_trade_deadline_context(session)
    if not ctx.get("summer_demand_window"):
        return 0
    book = ensure_trade_demands(session)
    resumed = 0
    abs_day = absolute_calendar_day(session, calendar_idx)
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "summer_pending":
            continue
        row["status"] = "open"
        row["summer_resumed"] = True
        crisis = row.get("crisis")
        if isinstance(crisis, dict):
            crisis["timer_frozen"] = False
            crisis["frozen_reason"] = ""
            crisis["remaining_days"] = float(crisis.get("remaining_days") or crisis.get("initial_days") or 18)
            crisis["last_abs_day"] = abs_day
            row["crisis"] = crisis
        resumed += 1
        user_tid = str(getattr(session, "user_team_id", "") or "")
        if str(row.get("team_id") or "") == user_tid:
            _push_notification(
                session,
                {
                    "id": f"{row.get('demand_id')}:summer_resume",
                    "type": "trade_demand_summer_resume",
                    "headline": f"Summer trade window — {row.get('player_name')} demand active again",
                    "body": "The offseason trade market is open. His trade request carries over from before the deadline.",
                    "team_id": user_tid,
                    "player_id": pid,
                },
            )
    return resumed


def _close_crises_for_trade_deadline(session: Any) -> int:
    """After Mar 10 — park open demands until the summer trade window."""
    ctx = get_trade_deadline_context(session)
    if not ctx.get("past_in_season_deadline") or ctx.get("summer_demand_window"):
        return 0
    book = ensure_trade_demands(session)
    closed = 0
    now = time.time()
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        crisis = row.get("crisis")
        if isinstance(crisis, dict):
            crisis["timer_frozen"] = True
            crisis["frozen_reason"] = "trade_deadline_passed"
            crisis["last_sync_unix"] = now
            row["crisis"] = crisis
        row["status"] = "summer_pending"
        row["resolved"] = False
        row["deadline_closed"] = True
        row["resolution"] = "trade_deadline"
        closed += 1
        user_tid = str(getattr(session, "user_team_id", "") or "")
        if str(row.get("team_id") or "") == user_tid:
            _push_notification(
                session,
                {
                    "id": f"{row.get('demand_id')}:deadline",
                    "type": "trade_demand_deadline_closed",
                    "headline": f"Trade deadline passed — {row.get('player_name')} demand frozen",
                    "body": (
                        "The NHL trade deadline has passed. The crisis timer is paused until the "
                        "offseason. You may revisit this player in the summer trade market."
                    ),
                    "team_id": user_tid,
                    "player_id": pid,
                },
            )
    return closed


def _push_notification(session: Any, note: Dict[str, Any]) -> None:
    """Append (newest last) like every other notification writer.

    The client payload reads ``session.notifications[-56:]``; inserting at index 0 and
    keeping ``[:120]`` hid every demand notice once the inbox passed 56 rows and also
    deleted the newest notices other systems had appended.
    """
    note = dict(note)
    note.setdefault("title", note.get("headline") or "")
    note.setdefault("text", note.get("body") or note.get("headline") or "")
    note.setdefault("source", "trade_demand")
    notes = list(getattr(session, "notifications", None) or [])
    notes.append(note)
    session.notifications = notes[-180:]

REASON_COPY = {
    "role": {
        "variants": [
            {
                "headline": "{name} demands a trade over {phrase}",
                "body": "{agent} says {name} is tired of {phrase} and has formally requested a trade{detail}.",
            },
            {
                "headline": "{name} wants out — unhappy with {phrase}",
                "body": "{name} told {agent} he does not see a path on this club anymore{detail}.",
            },
            {
                "headline": "Trade request: {name}",
                "body": "Through {agent}, {name} has asked out over {phrase}.",
            },
        ],
    },
    "losing": {
        "headline": "{name} wants out — tired of losing",
        "body": "{name} has asked for a trade after another stretch of losses. The locker room knows.",
    },
    "management": {
        "headline": "{name} has lost faith in management",
        "body": "{name} no longer trusts the front office and has requested a trade through his agent.",
    },
    "trade_exposure": {
        "headline": "{name} wants out after repeated trade talks",
        "body": "{name} is fed up with being shopped and has formally requested a move.",
    },
    "locker_room_disruptor": {
        "headline": "LOCKER-ROOM DISRUPTOR — {name} torpedoes the room",
        "body": (
            "{name} has gone nuclear. Unwilling to be reasoned with, he is trying to force a "
            "move even if it burns bridges and torpedoes his own value."
        ),
    },
    "relocation": {
        "headline": "{name} wants out after the relocation",
        "body": "{name} doesn't want to uproot his family for the move and has asked for a trade through his agent.",
    },
    "general": {
        "headline": "{name} has formally requested a trade",
        "body": "{name} has delivered a trade request through his agent. The relationship is at a breaking point.",
    },
    "contract": {
        "headline": "{name} demands a trade over his contract",
        "body": "{name} is unhappy with his deal and has asked his agent to find a trade.",
    },
    "development": {
        "headline": "{name} wants out over his development",
        "body": "{name} does not see a path here and has formally requested a trade.",
    },
    "belonging": {
        "headline": "{name} wants out — he does not feel he belongs",
        "body": "{name} has asked for a trade. He no longer feels like part of the room.",
    },
    "personal": {
        "headline": "{name} has asked for a trade for personal reasons",
        "body": "{name} wants a change of scenery for his family and has requested a trade through his agent.",
    },
    "organizational": {
        "headline": "{name} wants out over the direction of the club",
        "body": "{name} does not believe in where the organization is headed and has requested a trade.",
    },
    "canada": {
        "headline": "{name} wants a trade back to Canada",
        "body": "{name} has asked his agent to find a deal that brings him home.",
    },
    "small_market": {
        "headline": "{name} wants out of the spotlight",
        "body": "{name} has asked for a trade to a quieter market.",
    },
    # Goofy demands (rare; see maybe_open_silly_trade_demand).
    "silly_weather": {
        "headline": "{name} demands a trade because he hates the weather",
        "body": "{name}'s agent confirms the request is 'about the clouds, mostly'. He has been seen checking forecasts in other cities between periods.",
    },
    "silly_hot_dogs": {
        "headline": "{name} demands a trade over the arena hot dogs",
        "body": "{name} called the concourse hot dogs 'a crime against sausages' and has formally asked to be moved to a club with better food.",
    },
    "silly_jersey": {
        "headline": "{name} wants out — he can't stand the jersey colour",
        "body": "{name} says the team colours 'wash him out' on TV. His agent has a mood board of preferred destinations.",
    },
    "silly_goal_horn": {
        "headline": "{name} requests a trade over the goal horn",
        "body": "{name} claims the home goal horn is 'a half-step flat' and he can't celebrate properly. The trade request is real.",
    },
}

SILLY_DEMAND_REASONS = ("silly_weather", "silly_hot_dogs", "silly_jersey", "silly_goal_horn")
#: Weekly chance one random 70+ player league-wide files a goofy trade request.
SILLY_DEMAND_WEEKLY_CHANCE = 0.06
#: CPU demands at or above this OVR become a league-news notification.
CPU_DEMAND_NEWS_MIN_OVR = 78


def maybe_open_silly_trade_demand(session: Any, calendar_idx: int, *, iso_date: str = "") -> Optional[Dict[str, Any]]:
    """Weekly, rare goofy headline — no crisis timer or trade-value cut."""
    if not getattr(session, "goofy_events_enabled", True):
        return None
    import random
    import zlib

    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return None
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    seed = zlib.crc32(f"goofy_demand|{_save_seed(session)}|{season}|{int(calendar_idx) // 7}".encode())
    rng = random.Random(seed)
    if rng.random() >= SILLY_DEMAND_WEEKLY_CHANCE:
        return None
    if not get_trade_deadline_context(session).get("new_demands_allowed"):
        return None
    abs_day = absolute_calendar_day(session, int(calendar_idx))
    teams = [t for t in list(_get(league, "teams", None) or []) if _get(t, "roster", None)]
    if not teams:
        return None
    user_tid = str(getattr(session, "user_team_id", "") or "")
    team_open, team_recent = _team_demand_load(session, int(calendar_idx))
    if rng.random() >= 0.25:
        teams = [t for t in teams if _team_key(t) != user_tid] or teams
    team = rng.choice(teams)
    tkey = _team_key(team)
    if team_open.get(tkey, 0) >= MAX_OPEN_DEMANDS_PER_TEAM:
        return None
    if team_recent.get(tkey, 0) >= MAX_DEMANDS_PER_TEAM_WINDOW:
        return None
    pool = [
        p for p in list(_get(team, "roster", None) or [])
        if not _get(p, "retired", False) and _player_id(p)
        and 70 <= _player_ovr(p) < 84 and not active_demand_for_player(session, _player_id(p))
    ]
    if not pool:
        return None
    player = rng.choice(pool)
    pid = _player_id(player)
    pst = ensure_player_storyline_state(player)
    last_closed = int(pst.get("last_trade_demand_closed_day") or 0)
    if last_closed and 0 <= (abs_day - last_closed) < DEMAND_REOPEN_COOLDOWN_DAYS:
        return None
    reason = rng.choice(SILLY_DEMAND_REASONS)
    copy = _pick_demand_copy(session, player, team, reason, [reason], rng)
    popup_id = f"goofy_demand:{pid}:{calendar_idx}"
    pending = list(getattr(session, "pending_ui_popups", None) or [])
    if any(isinstance(p, dict) and p.get("id") == popup_id for p in pending):
        return None
    pending.append(
        {
            "id": popup_id,
            "kind": "storyline",
            "presentation_type": "goofy_trade_demand",
            "theme": "info",
            "headline": copy["headline"],
            "body": copy["body"],
            "player_id": pid,
            "player_name": _player_name(player),
            "team_id": tkey,
            "actions": [
                {"id": "laugh_off", "label": "Laugh it off"},
                {"id": "meeting", "label": "Call a meeting"},
            ],
            "goofy": True,
            "reason_code": reason,
        }
    )
    session.pending_ui_popups = pending[-80:]
    return {"goofy": True, "player_id": pid, "reason_code": reason, "status": "goofy_popup"}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _player_id(player: Any) -> str:
    return str(_get(player, "id", "") or _get(player, "player_id", "") or "")


def invalidate_trade_demand_player_index(session: Any) -> None:
    try:
        session._trade_demand_player_index = None
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _build_trade_demand_player_index(session: Any) -> Dict[str, Any]:
    cached = getattr(session, "_trade_demand_player_index", None)
    if isinstance(cached, dict) and cached.get("_version") == getattr(session, "_roster_revision", 0):
        return cached
    league = getattr(getattr(session, "sim", None), "league", None)
    out: Dict[str, Any] = {"_version": getattr(session, "_roster_revision", 0)}
    if league is None:
        try:
            session._trade_demand_player_index = out
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        return out
    for team in list(_get(league, "teams", None) or []):
        for attr in _ORG_POOL_ATTRS:
            for p in list(_get(team, attr, None) or []):
                if p is None or _get(p, "retired", False):
                    continue
                pid = _player_id(p)
                if pid:
                    out[pid] = (p, team)
    try:
        session._trade_demand_player_index = out
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return out


def _player_name(player: Any) -> str:
    ident = _get(player, "identity", None)
    return str(_get(ident, "name", "") or _get(player, "name", "") or "Player")


def _player_ovr(player: Any) -> float:
    raw = _get(player, "ovr", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = 70.0
    try:
        o = float(raw if raw is not None else _get(player, "overall", 70) or 70)
    except Exception:
        o = 70.0
    if o <= 1.5:
        o *= 99.0
    return o


def _team_abbr(team: Any) -> str:
    return str(_get(team, "abbr", "") or _get(team, "team_id", "") or _get(team, "id", "") or "").upper()


def ensure_trade_demands(session: Any) -> Dict[str, Any]:
    book = getattr(session, "trade_demands", None)
    if not isinstance(book, dict):
        book = {}
        session.trade_demands = book
    return book


def active_demand_for_player(session: Any, player_id: str) -> Optional[Dict[str, Any]]:
    book = ensure_trade_demands(session)
    row = book.get(str(player_id))
    if isinstance(row, dict) and str(row.get("status") or "") == "open":
        return row
    return None


def _rank_mntc_clubs(player: Any, teams: List[Any], clubs: List[Any], rng: Any) -> List[Any]:
    """Order (team_id, abbr) pairs by how much this player wants to go there."""
    try:
        from app.sim_engine.trades.trade_rules import _market_size, _team_strength_proxy
    except Exception:
        _market_size = None
        _team_strength_proxy = None
    try:
        age = int(_get(player, "age", 0) or 0)
    except (TypeError, ValueError):
        age = 0
    bc = _birth_country(player) if player is not None else ""
    canadian = bc in ("canada", "can", "ca")
    family = age >= 29  # settled veterans weigh home / family more
    by_key = {}
    for t in teams:
        by_key[str(_get(t, "team_id", "") or _get(t, "id", "") or "")] = t
        by_key[_team_abbr(t)] = t
    scored = []
    for tid, ab in clubs:
        t = by_key.get(tid) or by_key.get(ab)
        strength = 0.5
        market = "medium"
        if t is not None and _team_strength_proxy is not None:
            try:
                strength = float(_team_strength_proxy(t))
                market = _market_size(t)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
        score = strength * (1.6 if age >= 27 else 1.0)
        if market == "large":
            score += 0.15 if age < 29 else -0.05
        elif market == "small" and family:
            score += 0.1
        if family:
            in_canada = ab in CANADA_ABBRS
            score += 0.35 if in_canada == canadian else -0.1
        score += rng.uniform(-0.12, 0.12)
        scored.append((score, (tid, ab)))
    scored.sort(key=lambda r: -r[0])
    return [c for _, c in scored]


def seed_mntc_destinations(player: Any, league: Any, *, list_size: int = 8, rng: Any = None) -> List[str]:
    contract = _get(player, "contract", None)
    if contract is None:
        return []
    existing = (
        list(_get(contract, "approved_trade_teams", None) or [])
        or list(_get(contract, "approved_trade_team_ids", None) or [])
        or list(_get(contract, "approved_destinations", None) or [])
    )
    if existing:
        return [str(x).upper() for x in existing if x]
    clause = str(
        _get(contract, "clause", "")
        or _get(contract, "clause_type", "")
        or _get(contract, "trade_clause", "")
        or ""
    ).upper()
    mode = str(_get(contract, "ntc_mode", "") or "").upper()
    listed = int(_get(contract, "modified_no_trade_teams", 0) or 0)
    is_modified = (
        "MNTC" in clause
        or "M-NTC" in clause
        or "MODIFIED" in clause
        or mode in ("MODIFIED", "MNTC", "M-NTC")
        or listed > 0
    )
    if not is_modified:
        size = int(_get(contract, "trade_list_size", 0) or 0)
        if size <= 0:
            return []
    else:
        size = listed or int(_get(contract, "trade_list_size", 0) or 0) or list_size or 10
    size = max(1, min(32, size or list_size))
    teams = list(_get(league, "teams", None) or [])
    clubs = []
    for t in teams:
        tid = str(_get(t, "team_id", "") or _get(t, "id", "") or "")
        ab = _team_abbr(t)
        if tid or ab:
            clubs.append((tid, ab))
    if not clubs:
        return []
    # The player picks the clubs he'd accept: contenders, a market that suits him,
    # and (for veterans with families) his home country. Small seeded jitter only.
    import random

    stable = random.Random(int(__import__("hashlib").sha256(("mntc|" + str(_get(player, "id", ""))).encode()).hexdigest()[:12], 16) & 0xFFFFFFFF)
    picked = _rank_mntc_clubs(player, teams, clubs, stable)[: min(size, len(clubs))]
    chosen = []
    for tid, ab in picked:
        for key in (tid, ab):
            if key and key not in chosen:
                chosen.append(key)
    try:
        if isinstance(contract, dict):
            contract["approved_trade_teams"] = chosen
            contract["approved_destinations"] = chosen
            contract["ntc_teams"] = chosen
        else:
            setattr(contract, "approved_trade_teams", chosen)
            setattr(contract, "approved_destinations", chosen)
            setattr(contract, "ntc_teams", chosen)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return chosen


def _birth_country(player: Any) -> str:
    ident = _get(player, "identity", None)
    raw = (
        _get(ident, "birth_country", "")
        or _get(player, "nationality", "")
        or _get(player, "birth_country", "")
        or ""
    )
    return str(raw).strip().lower()


def _team_market_size(team: Any) -> str:
    market = _get(team, "market", None)
    return str(_get(market, "market_size", "") or _get(team, "market_size", "") or "").strip().lower()


def _home_attachment_score(player: Any, rng: Any = None) -> float:
    """0–1 willingness to insist on a Canada / home-market move (not birthplace alone)."""
    traits = _get(player, "traits", None)
    family = 0.5
    loyalty = 0.5
    if traits is not None:
        try:
            family = float(_get(traits, "family_priority", _get(traits, "loyalty", 0.5)) or 0.5)
            loyalty = float(_get(traits, "loyalty", 0.5) or 0.5)
        except (TypeError, ValueError):
            pass
    if family > 1.5:
        family /= 100.0
    if loyalty > 1.5:
        loyalty /= 100.0
    attach = 0.35 * family + 0.20 * loyalty
    bc = _birth_country(player)
    if bc in ("canada", "can", "ca"):
        attach += 0.12
    try:
        age = int(_get(_get(player, "identity", None), "age", _get(player, "age", 26)) or 26)
    except (TypeError, ValueError):
        age = 26
    if age >= 30:
        attach += 0.08
    if rng is not None and hasattr(rng, "uniform"):
        attach += rng.uniform(-0.06, 0.06)
    return max(0.0, min(1.0, attach))


def _destination_lean(player: Any, team: Any, top: str, rng: Any = None) -> str:
    """Belonging or family grievances can point the wish list at a real market."""
    if player is None or team is None:
        return ""
    home = _team_abbr(team)
    if top in ("belonging", "personal", "broken_promise"):
        if _home_attachment_score(player, rng) >= 0.62 and _birth_country(player) in ("canada", "can", "ca"):
            if home not in CANADA_ABBRS:
                return "canada"
    if top in ("personal", "broken_promise") and _team_market_size(team) == "large":
        if _home_attachment_score(player, rng) >= 0.55:
            return "small_market"
    return ""


def _preferred_destinations(
    player: Any,
    team: Any,
    league: Any,
    reason: str,
    *,
    rng: Any,
    list_size: int = 12,
) -> List[str]:
    seed_mntc_destinations(player, league, list_size=list_size, rng=rng)
    teams = list(_get(league, "teams", None) or [])
    home = _team_abbr(team)
    want_canada = reason == "canada"
    want_small = reason == "small_market"
    try:
        from app.sim_engine.trades.trade_rules import _market_size, _team_strength_proxy
    except Exception:
        _market_size = None
        _team_strength_proxy = None

    strengths: List[Tuple[float, str]] = []
    for t in teams:
        ab = _team_abbr(t)
        if not ab or ab == home:
            continue
        if want_canada and ab not in CANADA_ABBRS:
            continue
        if want_small and ab not in SMALL_MARKET_ABBRS:
            continue
        w = 1.0
        strength = 0.5
        market = "medium"
        if _team_strength_proxy is not None:
            try:
                strength = float(_team_strength_proxy(t))
            except Exception:
                strength = 0.5
        if _market_size is not None:
            try:
                market = str(_market_size(t) or "medium")
            except Exception:
                market = "medium"
        if reason in ("losing", "organizational"):
            w += strength * 1.8
        elif reason == "role":
            w += 0.6 + strength * 0.9
        elif reason == "contract":
            cap = float(_get(t, "cap_space", _get(t, "cap_room", 8.0)) or 8.0)
            w += max(0.2, min(2.5, cap / 6.0))
        elif reason in ("personal", "belonging", "canada"):
            w += 0.8 if market == "large" else 1.1
            if ab in CANADA_ABBRS and _birth_country(player) in ("canada", "can", "ca"):
                w += 0.9
        elif reason in ("management", "trade_exposure"):
            w += 0.5 + strength * 0.4
        else:
            w += 0.35 + strength * 0.5
        strengths.append((max(0.05, w), ab))

    if not strengths:
        strengths = [
            (1.0, ab)
            for ab in (_team_abbr(t) for t in teams)
            if ab and ab != home and (not want_canada or ab in CANADA_ABBRS)
        ]
    n = max(6, min(list_size, 10, len(strengths)))
    out: List[str] = []
    pool = list(strengths)
    for _ in range(min(n, len(pool))):
        total = sum(w for w, _ in pool)
        if total <= 0:
            break
        pick = rng.random() * total if rng is not None else total * 0.5
        acc = 0.0
        chosen_i = 0
        for i, (w, ab) in enumerate(pool):
            acc += w
            if pick <= acc:
                chosen_i = i
                out.append(ab)
                pool.pop(i)
                break
        else:
            out.append(pool.pop(0)[1])
    return out


def _snapshot_value(player: Any, team: Any, league: Any, *, crisis_stage: int = 1) -> float:
    """Trade value before the crisis discount.

    The stage multiplier is applied by ``_apply_crisis_value_state``. Distressed
    pricing is keyword-only and only runs when the crisis timer expires.
    """
    del crisis_stage
    try:
        from app.sim_engine.trades.trade_value import evaluate_player_asset_value

        row = evaluate_player_asset_value(player, team, team, league, context={}) or {}
        raw = row.get("trade_value")
        if raw is None:
            raw = row.get("total") or 0.0
        return max(-20.0, float(raw))
    except Exception:
        return max(5.0, _player_ovr(player) * 0.7)


def _apply_crisis_value_state(
    player: Any,
    *,
    crisis_stage: int,
    base_before: float,
    timer_expired: bool = False,
) -> Tuple[float, float]:
    mult = crisis_trade_value_multiplier(crisis_stage)
    distressed = crisis_distressed_asset_cost(base_before, timer_expired=timer_expired)
    effective_mult = mult
    if timer_expired:
        effective_mult = min(effective_mult, 0.12)
    try:
        setattr(player, "_trade_demand_active", True)
        setattr(player, "_crisis_trade_value_mult", mult)
        setattr(player, "_crisis_distressed_asset", distressed)
        setattr(player, "_crisis_trade_stage", 4 if timer_expired else crisis_stage)
        setattr(player, "_systemic_trade_value_mult", mult)
        if crisis_stage >= 3 and timer_expired:
            pst = ensure_player_storyline_state(player)
            char = int(_player_character_0_100(player))
            pst["character"] = char
            if char < 65:
                setattr(player, "locker_room_disruptor", True)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    after = max(-20.0, base_before * mult - distressed)
    return mult, after


def _pressure_to_reason_code(pressure_key: str, *, character: int, player: Any, team: Any, rng: Any) -> str:
    top = str(pressure_key or "")
    if top == "trade_exposure":
        return "trade_exposure"
    if top in ("role", "performance"):
        return "role"
    if top in ("management", "coach"):
        return "management"
    if top == "winning":
        return "losing"
    if top == "contract":
        return "contract"
    if top == "development":
        return "development"
    if top == "organizational":
        return "organizational"
    if top in ("belonging", "personal", "broken_promise"):
        lean = _destination_lean(player, team, top, rng=rng)
        if lean:
            return lean
        return "belonging" if top == "belonging" else "personal"
    if top == "temperament":
        return "locker_room_disruptor" if character < 62 else "general"
    return "general"


def _reason_from_pressures(
    pressures: Dict[str, float],
    *,
    character: int,
    player: Any = None,
    team: Any = None,
    rng: Any = None,
) -> Tuple[str, List[str]]:
    if not pressures:
        return "general", ["general"]
    ranked = sorted(pressures.items(), key=lambda kv: -float(kv[1] or 0))
    codes: List[str] = []
    top_val = float(ranked[0][1] or 0)
    for key, val in ranked[:3]:
        if codes and float(val or 0) < top_val * 0.7:
            break
        code = _pressure_to_reason_code(key, character=character, player=player, team=team, rng=rng)
        if code not in codes:
            codes.append(code)
    primary = codes[0] if codes else "general"
    return primary, codes


def _start_crisis_timer(
    session: Any,
    player: Any,
    *,
    character: int,
    previous_demands: int,
    calendar_idx: int,
) -> Dict[str, Any]:
    agent = ensure_player_agent(player, session)
    gm_rel = get_agent_gm_relationship(session, str(agent.get("id") or ""))
    initial_days = agent_crisis_initial_days(
        character=character,
        agent=agent,
        gm_rel=gm_rel,
        previous_demands=previous_demands,
    )
    legacy_initial = int(
        agent_crisis_initial_seconds(
            character=character,
            agent=agent,
            gm_rel=gm_rel,
            previous_demands=previous_demands,
        )
    )
    abs_day = absolute_calendar_day(session, calendar_idx)
    now = time.time()
    return {
        "initial_days": initial_days,
        "remaining_days": float(initial_days),
        "last_abs_day": abs_day,
        "deadline_seconds": legacy_initial,
        "initial_seconds": legacy_initial,
        "remaining_seconds": legacy_initial,
        "last_sync_unix": now,
        "crisis_stage": 1,
        "started_at_unix": now,
        "timer_mode": "sim_days",
    }


def _hydrate_crisis_timer(crisis: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(crisis or {})
    if c.get("timer_mode") != "sim_days" and c.get("remaining_seconds") is not None:
        try:
            sec = float(c.get("remaining_seconds") or 0)
            days = max(1.0, sec / 86400.0)
            c["remaining_days"] = days
            c["initial_days"] = int(c.get("initial_days") or max(14, round(days)))
            c["timer_mode"] = "sim_days"
        except (TypeError, ValueError):
            c["remaining_days"] = 18.0
            c["initial_days"] = 18
            c["timer_mode"] = "sim_days"
    if c.get("remaining_days") is None:
        c["remaining_days"] = float(c.get("initial_days") or 18)
    if c.get("initial_days") is None:
        c["initial_days"] = int(round(float(c.get("remaining_days") or 18)))
    return c


def _advance_user_crisis_timers(session: Any, calendar_idx: int) -> None:
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if not user_tid:
        return
    abs_day = absolute_calendar_day(session, calendar_idx)
    book = ensure_trade_demands(session)
    league = getattr(getattr(session, "sim", None), "league", None)
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        if str(row.get("team_id") or "") != user_tid:
            continue
        crisis = _hydrate_crisis_timer(row.get("crisis") or {})
        if crisis.get("timer_frozen"):
            row["crisis"] = crisis
            continue
        last_abs = int(crisis.get("last_abs_day") or abs_day)
        delta = max(0, abs_day - last_abs)
        if delta > 0:
            crisis["remaining_days"] = max(0.0, float(crisis.get("remaining_days") or 0) - float(delta))
            crisis["last_abs_day"] = abs_day
        initial_days = max(1, int(crisis.get("initial_days") or 18))
        legacy_initial = int(crisis.get("initial_seconds") or crisis.get("deadline_seconds") or 360)
        remaining = float(crisis.get("remaining_days") or 0)
        ratio = remaining / float(initial_days) if initial_days else 0.0
        crisis["remaining_seconds"] = max(0.0, legacy_initial * ratio)
        expired = remaining <= 0
        stage = 4 if expired else crisis_stage_from_remaining(legacy_initial, int(round(crisis["remaining_seconds"])))
        prev_stage = int(row.get("crisis_stage") or crisis.get("crisis_stage") or 1)
        crisis["crisis_stage"] = stage
        row["crisis_stage"] = stage
        row["crisis"] = crisis
        player = _find_player(session, pid)
        team = _find_team_for_player(session, pid)
        if player is not None and stage > prev_stage:
            base_before = float(row.get("value_before") or _snapshot_value(player, team, league, crisis_stage=1))
            _, after = _apply_crisis_value_state(
                player,
                crisis_stage=stage,
                base_before=base_before,
                timer_expired=expired,
            )
            row["value_after"] = round(after, 1)
            row["value_delta"] = round(after - base_before, 1)
            _apply_crisis_stage_effects(session, row, player, team, league, stage)
            row["ntc_waiver_snapshot"] = _build_ntc_waiver_snapshot(session, row, player, team, league)
        if expired:
            _finalize_crisis_expiry(session, row)


def _finalize_crisis_expiry(session: Any, row: Dict[str, Any]) -> None:
    pid = str(row.get("player_id") or "")
    row["status"] = "crisis_expired"
    row["resolved"] = False
    row["crisis_expired"] = True
    player = _find_player(session, pid)
    if player is not None:
        pst = ensure_player_storyline_state(player)
        pst["last_trade_demand_closed_day"] = absolute_calendar_day(
            session, int(getattr(session, "calendar_cursor", 0) or 0)
        )
        pst["crisis_expired_season"] = int(getattr(session, "season_calendar_year", 0) or 0)
        try:
            psych = getattr(player, "psych", None)
            if psych is not None:
                m = float(getattr(psych, "morale", 0.5) or 0.5)
                setattr(psych, "morale", max(0.15, m - 0.08))
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    _enqueue_crisis_expired(session, row)
    _archive_closed_demand(session, row, resolution="crisis_expired")
    book = ensure_trade_demands(session)
    book.pop(pid, None)


def sync_trade_demand_crises(
    session: Any,
    *,
    elapsed_hint: Optional[float] = None,
    tick_timers: bool = True,
) -> None:
    """Refresh crisis display state. Timers advance on sim days in process_trade_demand_day."""
    del elapsed_hint, tick_timers
    _close_crises_for_trade_deadline(session)
    book = ensure_trade_demands(session)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        if user_tid and str(row.get("team_id") or "") != user_tid:
            continue
        crisis = _hydrate_crisis_timer(row.get("crisis") or {})
        remaining_days = float(crisis.get("remaining_days") or 0)
        crisis["remaining_seconds"] = int(remaining_days * 86400)
        crisis["timer_active"] = False
        row["crisis"] = crisis


def _clause_label(player: Any) -> str:
    contract = _get(player, "contract", None)
    if contract is None:
        return "None"
    clause = str(
        _get(contract, "clause", "")
        or _get(contract, "clause_type", "")
        or _get(contract, "trade_clause", "")
        or ""
    ).upper()
    if "NMC" in clause or "NO MOVE" in clause:
        return "NMC"
    if "MNTC" in clause or "M-NTC" in clause or "MODIFIED" in clause:
        return "M-NTC"
    if "NTC" in clause:
        return "NTC"
    return "None"


def evaluate_trade_demand_ntc_waiver(
    session: Any,
    player: Any,
    source_team: Any,
    destination_team: Any,
    *,
    demand_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """NTC/M-NTC willingness during an active trade-demand crisis."""
    from app.sim_engine.franchise.trade_stability_engine import _player_character_0_100, _player_mental_0_100
    from app.sim_engine.trades.trade_rules import evaluate_ntc_waiver_request

    ctx = {
        "calendar_cursor": int(getattr(session, "calendar_cursor", 0) or 0),
        "season_year": int(getattr(session, "season_calendar_year", 2025) or 2025),
        "trade_demand_crisis": True,
    }
    base = evaluate_ntc_waiver_request(
        player,
        source_team=source_team,
        destination_team=destination_team,
        context=ctx,
    )
    if not base.get("can_request"):
        return dict(base)

    character = float(_player_character_0_100(player))
    mental = float(_player_mental_0_100(player))
    row = demand_row or {}
    crisis_stage = int(row.get("crisis_stage") or 1)
    agent = ensure_player_agent(player, session)

    chance = float(base.get("accept_chance") or 0.38)
    # Wants out — more willing to waive for a fresh start
    if row.get("status") == "open" or row.get("formal"):
        chance += 0.10 + crisis_stage * 0.04
    # A low-character player who has already demanded out will waive to leave.
    # A high-character player protects the clause. The old ladder did the opposite,
    # and the steepest low-character step never ran because `< 65` matched first.
    if character < 58:
        chance += 0.22
    elif character < 65:
        chance += 0.14
    elif character >= 85:
        chance -= 0.12
    elif character >= 77:
        chance -= 0.06
    if mental >= 88:
        chance += 0.04
    elif mental < 62:
        chance -= 0.06
    if str(agent.get("style") or "") == "disruptor" and crisis_stage >= 2 and character < 70:
        chance += 0.18
    if str(agent.get("style") or "") == "discreet" and character >= 80:
        chance -= 0.08

    chance = max(0.04, min(0.92, chance))
    roll = float(base.get("roll") or 0.5)
    accepted = roll < chance
    out = dict(base)
    out["accept_chance"] = round(chance, 3)
    out["accepted"] = bool(accepted)
    out["crisis_stage"] = crisis_stage
    out["character"] = int(character)
    out["mental"] = int(mental)
    out["agent_style"] = agent.get("style")
    return out


def _build_ntc_waiver_snapshot(
    session: Any,
    demand_row: Dict[str, Any],
    player: Any,
    source_team: Any,
    league: Any,
) -> Dict[str, Any]:
    """Sample waiver outcomes for approved/blocked destinations at current crisis stage."""
    if league is None or source_team is None:
        return {}
    clause = _clause_label(player)
    if clause == "None":
        return {"clause": clause, "waivers_required": False}

    approved = list(demand_row.get("preferred_destinations") or [])
    teams = list(_get(league, "teams", None) or [])
    team_by_abbr = {}
    for tm in teams:
        ab = _team_abbr(tm)
        if ab:
            team_by_abbr[ab] = tm

    blocked_abbrs = [ab for ab in team_by_abbr if ab not in approved and ab != _team_abbr(source_team)][:4]
    samples = []
    for label, abbrs in (("approved", approved[:3]), ("blocked", blocked_abbrs[:3])):
        for ab in abbrs:
            dest = team_by_abbr.get(ab)
            if dest is None:
                continue
            result = evaluate_trade_demand_ntc_waiver(
                session,
                player,
                source_team,
                dest,
                demand_row=demand_row,
            )
            samples.append(
                {
                    "bucket": label,
                    "destination": ab,
                    "accepted": bool(result.get("accepted")),
                    "accept_chance": result.get("accept_chance"),
                    "reason_code": result.get("reason_code"),
                    "reason": result.get("reason"),
                    "clause_label": result.get("clause_label") or clause,
                }
            )
    return {
        "clause": clause,
        "waivers_required": clause in ("NTC", "M-NTC", "NMC"),
        "crisis_stage": int(demand_row.get("crisis_stage") or 1),
        "samples": samples,
    }


def _apply_crisis_stage_effects(
    session: Any,
    row: Dict[str, Any],
    player: Any,
    team: Any,
    league: Any,
    stage: int,
) -> None:
    import random

    agent = ensure_player_agent(player, session)
    gm_rel = get_agent_gm_relationship(session, str(agent.get("id") or ""))
    rng = getattr(getattr(session, "sim", None), "rng", None) or random.Random()

    shrink = agent_destination_shrink_factor(agent, stage)
    full = list(row.get("preferred_destinations_full") or row.get("preferred_destinations") or [])
    if full:
        keep = max(1, int(round(len(full) * shrink)))
        if keep < len(full):
            dests = list(rng.sample(full, keep)) if hasattr(rng, "sample") else full[:keep]
            row["preferred_destinations"] = dests
            row["destination_count"] = len(dests)

    leak_p = agent_leak_probability(agent, crisis_stage=stage, gm_rel=gm_rel)
    if stage >= 2 and float(rng.random()) < leak_p:
        row["leaked"] = True
        row["public_demand"] = stage >= 3

    pst = ensure_player_storyline_state(player)
    pst["gm_trust"] = max(0.05, float(pst.get("gm_trust", 0.72)) - 0.02 * stage)


def _find_player(session: Any, player_id: str) -> Any:
    pid = str(player_id)
    hit = _build_trade_demand_player_index(session).get(pid)
    if hit and isinstance(hit, tuple):
        return hit[0]
    return None


def _find_team_for_player(session: Any, player_id: str) -> Any:
    pid = str(player_id)
    hit = _build_trade_demand_player_index(session).get(pid)
    if hit and isinstance(hit, tuple):
        return hit[1]
    return None


def _archive_closed_demand(session: Any, row: Dict[str, Any], *, resolution: str) -> None:
    if not isinstance(row, dict):
        return
    hist = list(getattr(session, "trade_demands_history", None) or [])
    hist.append(
        {
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "team_id": row.get("team_id"),
            "season": int(getattr(session, "season_calendar_year", 0) or 0),
            "reason_code": row.get("reason_code"),
            "resolution": resolution,
            "opened_abs_day": row.get("opened_abs_day"),
            "closed_abs_day": absolute_calendar_day(session, int(getattr(session, "calendar_cursor", 0) or 0)),
        }
    )
    try:
        session.trade_demands_history = hist[-500:]
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def _reason_phrase_for_copy(reason_codes: List[str]) -> str:
    labels = {
        "role": "role and deployment",
        "contract": "his contract",
        "management": "management",
        "losing": "losing",
        "trade_exposure": "trade talk",
        "development": "development",
        "belonging": "fitting in",
        "personal": "personal reasons",
        "organizational": "the direction of the club",
    }
    codes = [c for c in (reason_codes or []) if c and not str(c).startswith("silly_")]
    if len(codes) >= 2:
        a = labels.get(codes[0], codes[0].replace("_", " "))
        b = labels.get(codes[1], codes[1].replace("_", " "))
        return f"{a} and {b}"
    if codes:
        return labels.get(codes[0], codes[0].replace("_", " "))
    return "the situation"


def _pick_demand_copy(
    session: Any,
    player: Any,
    team: Any,
    reason: str,
    reason_codes: List[str],
    rng: Any,
) -> Dict[str, str]:
    import zlib

    base = REASON_COPY.get(reason) or REASON_COPY["general"]
    variants = list(base.get("variants") or []) if isinstance(base, dict) else []
    if not variants and isinstance(base, dict) and base.get("headline"):
        variants = [{"headline": base["headline"], "body": base["body"]}]
    pid = _player_id(player)
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    seed = zlib.crc32(f"tdcopy|{_save_seed(session)}|{pid}|{season}|{reason}".encode()) & 0xFFFFFFFF
    pick = variants[seed % len(variants)] if variants else base
    name = _player_name(player)
    agent = ensure_player_agent(player, session)
    agent_name = str(agent.get("name") or "His agent")
    phrase = _reason_phrase_for_copy(reason_codes)
    years = _get(_get(player, "contract", None), "years_remaining", None) or _get(
        _get(player, "contract", None), "years_left", None
    )
    detail = ""
    if years is not None:
        detail = f" ({int(years)} years left on his deal)"
    headline = str(pick.get("headline") or base.get("headline", "")).format(name=name, phrase=phrase, agent=agent_name)
    body = str(pick.get("body") or base.get("body", "")).format(
        name=name, phrase=phrase, agent=agent_name, detail=detail
    )
    return {"headline": headline, "body": body}


def open_trade_demand(
    session: Any,
    player: Any,
    team: Any,
    *,
    reason: str,
    calendar_idx: int,
    iso_date: str = "",
    rng: Any = None,
    stability_row: Optional[Dict[str, Any]] = None,
    force_formal: bool = False,
) -> Dict[str, Any]:
    """Create/overwrite an open formal demand with real-time crisis timer."""
    import random

    r = rng if rng is not None else random.Random()
    league = getattr(getattr(session, "sim", None), "league", None)
    pid = _player_id(player)
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    pst = ensure_player_storyline_state(player)

    if stability_row is None:
        stability_row = update_player_stability(session, player, team)

    pressures = dict(stability_row.get("pressures") or {})
    character = int(stability_row.get("character") or pst.get("character") or 74)
    reason_codes: List[str]
    if not reason or reason == "auto":
        reason, reason_codes = _reason_from_pressures(
            pressures, character=character, player=player, team=team, rng=r
        )
    else:
        reason_codes = [reason]

    ovr = _player_ovr(player)
    depth_request = ovr < DEPTH_DEMAND_MAX_OVR and not force_formal

    disruptor = reason == "locker_room_disruptor" or int(stability_row.get("escalation_level") or 0) >= 4
    escalation = int(stability_row.get("escalation_level") or 3)
    if not force_formal and escalation < 3:
        return {"status": "warning", "escalation_level": escalation, "player_id": pid}

    deadline_ctx = get_trade_deadline_context(session)
    if not deadline_ctx.get("new_demands_allowed"):
        return {
            "status": "blocked",
            "escalation_level": escalation,
            "player_id": pid,
            "reason": "trade_deadline_passed",
        }
    if not force_formal and not formal_demand_eligible(stability_row):
        return {"status": "blocked", "escalation_level": escalation, "player_id": pid, "reason": "insufficient_signals"}

    previous_demands = int(pst.get("career_trade_demand_count") or 0)
    crisis = None
    if not depth_request:
        crisis = _start_crisis_timer(
            session, player, character=character, previous_demands=previous_demands, calendar_idx=int(calendar_idx)
        )

    agent_view = agent_public_view(player, session)
    before = _snapshot_value(player, team, league, crisis_stage=1)
    if depth_request:
        after = before * 0.97
    else:
        _, after = _apply_crisis_value_state(player, crisis_stage=1, base_before=before)
    if disruptor:
        try:
            setattr(player, "locker_room_disruptor", True)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    dests = _preferred_destinations(player, team, league, reason, rng=r, list_size=12)
    copy = _pick_demand_copy(session, player, team, reason, reason_codes, r)
    name = _player_name(player)
    demand_id = f"demand:{pid}:{calendar_idx}:{uuid.uuid4().hex[:6]}"
    complaint = _reason_phrase_for_copy(reason_codes)
    abs_opened = absolute_calendar_day(session, int(calendar_idx))

    pst["career_trade_demand_count"] = previous_demands + 1
    pst["season_trade_demand_count"] = int(pst.get("season_trade_demand_count") or 0) + 1
    pst["previous_trade_demand_severity"] = escalation
    pst["previous_trade_demand_team"] = tid
    pst["previous_trade_demand_reason"] = reason

    row = {
        "demand_id": demand_id,
        "player_id": pid,
        "player_name": name,
        "team_id": tid,
        "status": "open",
        "formal": True,
        "reason_code": reason,
        "reason_codes": list(reason_codes),
        "primary_complaint": complaint,
        "depth_request": depth_request,
        "opened_abs_day": abs_opened,
        "cause_type": "TRADE_DEMAND",
        "cause_event_id": demand_id,
        "opened_day": int(calendar_idx),
        "opened_iso": str(iso_date or ""),
        "trade_stability_score": stability_row.get("trade_stability_score"),
        "escalation_level": escalation,
        "value_before": round(float(before), 1),
        "value_after": round(float(after), 1),
        "value_delta": round(float(after) - float(before), 1),
        "preferred_destinations": dests,
        "preferred_destinations_full": list(dests),
        "destination_count": len(dests),
        "disruptor": disruptor,
        "dossier_label": "Locker-room disruptor" if disruptor else "Trade demand",
        "headline": copy["headline"].format(name=name),
        "body": copy["body"].format(name=name),
        "ovr": round(_player_ovr(player)),
        "agent": agent_view,
        "crisis": crisis,
        "crisis_stage": 1,
        "career_trade_demand_count": pst["career_trade_demand_count"],
        "season_trade_demand_count": pst["season_trade_demand_count"],
    }
    row["ntc_waiver_snapshot"] = _build_ntc_waiver_snapshot(session, row, player, team, league)
    book = ensure_trade_demands(session)
    book[pid] = row
    # League-side tags so the CPU proposer (which only sees the league) can act on it.
    try:
        setattr(player, "_trade_demand_opened_day", int(calendar_idx))
        setattr(player, "_trade_demand_destinations", list(dests))
        setattr(player, "_trade_demand_team_id", tid)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        _enqueue_demand_surfaces(session, row, team)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    if tid != str(getattr(session, "user_team_id", "") or ""):
        # CPU clubs usually move a good player within a week of his demand, so the
        # buying window opens now, not at stage 2.
        _maybe_offer_trade_opportunity(session, row, player, team, 1, float(before), float(after))
    return row


def clear_trade_demand(session: Any, player_id: str, *, resolution: str = "cleared") -> None:
    book = ensure_trade_demands(session)
    pid = str(player_id)
    row = book.get(pid)
    closed_abs = absolute_calendar_day(session, int(getattr(session, "calendar_cursor", 0) or 0))
    if isinstance(row, dict):
        row["status"] = resolution
        row["resolved"] = True
        row["closed_day"] = int(getattr(session, "calendar_cursor", 0) or 0)
        row["closed_abs_day"] = closed_abs
        if resolution == "traded":
            row["crisis"] = None
        _archive_closed_demand(session, row, resolution=resolution)
        book.pop(pid, None)
    player = _find_player(session, pid)
    if player is not None:
        clear_demand_temporary_modifiers(player)
        pst = ensure_player_storyline_state(player)
        pst["last_trade_demand_closed_day"] = closed_abs


def _team_key(team: Any) -> str:
    return str(_get(team, "team_id", "") or _get(team, "id", "") or "")


def _team_demand_load(session: Any, calendar_idx: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Per-club (open demands now, demands opened in the recent window)."""
    open_by: Dict[str, int] = {}
    recent_by: Dict[str, int] = {}
    abs_day = absolute_calendar_day(session, calendar_idx)
    rows = list(ensure_trade_demands(session).values())
    rows.extend(list(getattr(session, "trade_demands_history", None) or [])[-120:])
    for row in rows:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("team_id") or "")
        if not tid:
            continue
        if str(row.get("status") or "") == "open":
            open_by[tid] = open_by.get(tid, 0) + 1
        opened_abs = row.get("opened_abs_day")
        if opened_abs is None:
            opened_abs = absolute_calendar_day(session, int(row.get("opened_day") or 0))
        age = int(abs_day) - int(opened_abs or 0)
        if 0 <= age < DEMAND_TEAM_WINDOW_DAYS:
            recent_by[tid] = recent_by.get(tid, 0) + 1
    return open_by, recent_by


def _age_cpu_demands(session: Any, calendar_idx: int) -> Dict[str, int]:
    """Advance CPU-club demands by sim days and close ones already resolved by a trade.

    The user's crisis runs on a real-time clock; CPU clubs have nobody watching that
    clock, so their demands used to sit at stage 1 until the deadline froze them.
    """
    book = ensure_trade_demands(session)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    league = getattr(getattr(session, "sim", None), "league", None)
    out = {"aged": 0, "resolved": 0}
    for pid, row in list(book.items()):
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        tid = str(row.get("team_id") or "")
        if user_tid and tid == user_tid:
            continue
        player = _find_player(session, pid)
        cur_team = _find_team_for_player(session, pid)
        if player is None or cur_team is None:
            row["status"] = "lapsed"
            row["resolved"] = True
            row["resolution"] = "left_nhl_roster"
            out["resolved"] += 1
            continue
        if _team_key(cur_team) != tid:
            # Moved by a CPU-CPU trade (those bypass trade_service's clear hook).
            clear_trade_demand(session, pid, resolution="traded")
            out["resolved"] += 1
            continue

        opened = int(row.get("opened_day") or calendar_idx)
        days_open = max(0, int(calendar_idx) - opened)
        row["days_open"] = days_open
        if getattr(player, "_trade_demand_opened_day", None) is None:
            try:
                setattr(player, "_trade_demand_opened_day", opened)
                setattr(player, "_trade_demand_destinations", list(row.get("preferred_destinations") or []))
                setattr(player, "_trade_demand_team_id", tid)
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)

        if days_open >= CPU_DEMAND_STAGE3_DAYS + CPU_DEMAND_STAGE3_STALE_DAYS:
            row["status"] = "resolved"
            row["resolved"] = True
            row["resolution"] = "stays_reluctantly"
            pst = ensure_player_storyline_state(player)
            pst["last_trade_demand_closed_day"] = absolute_calendar_day(session, int(calendar_idx))
            clear_demand_temporary_modifiers(player)
            _archive_closed_demand(session, row, resolution="stays_reluctantly")
            book.pop(pid, None)
            out["resolved"] += 1
            continue
        if days_open >= CPU_DEMAND_STAGE3_DAYS:
            stage = 3
        elif days_open >= CPU_DEMAND_STAGE2_DAYS:
            stage = 2
        else:
            stage = 1
        prev_stage = int(row.get("crisis_stage") or 1)
        if stage == prev_stage:
            continue
        base_before = float(row.get("value_before") or _snapshot_value(player, cur_team, league, crisis_stage=1))
        _, after = _apply_crisis_value_state(player, crisis_stage=stage, base_before=base_before)
        row["crisis_stage"] = stage
        row["value_after"] = round(after, 1)
        row["value_delta"] = round(after - base_before, 1)
        crisis = row.get("crisis")
        if isinstance(crisis, dict):
            crisis["crisis_stage"] = stage
        if stage > prev_stage:
            _apply_crisis_stage_effects(session, row, player, cur_team, league, stage)
            try:
                setattr(player, "_trade_demand_destinations", list(row.get("preferred_destinations") or []))
            except Exception:
                _swallowed_log.debug("suppressed exception", exc_info=True)
            _maybe_offer_trade_opportunity(session, row, player, cur_team, stage, base_before, after)
        out["aged"] += 1
    return out


# A CPU player whose demand has dragged his value down is a real buying chance. Only
# players worth acting on (a lineup upgrade) raise it, once per stage, and it pauses
# week / month sims so the window is not simmed past.
OPPORTUNITY_MIN_OVR = 78.0
OPPORTUNITY_MIN_DROP = 0.05


def _maybe_offer_trade_opportunity(
    session: Any, row: Dict[str, Any], player: Any, team: Any, stage: int, before: float, after: float
) -> None:
    try:
        if stage < 1 or before <= 0:
            return
        drop = (float(before) - float(after)) / float(before)
        if drop < OPPORTUNITY_MIN_DROP or _player_ovr(player) < OPPORTUNITY_MIN_OVR:
            return
        pid = _player_id(player)
        key = f"trade_opp_{pid}_{stage}"
        pending = list(getattr(session, "pending_ui_popups", None) or [])
        if any(isinstance(p, dict) and p.get("id") == key for p in pending):
            return
        name = _player_name(player)
        abbr = _team_abbr(team)
        reason = str(row.get("reason") or row.get("primary_reason") or "").replace("_", " ")
        contract = getattr(player, "contract", None)
        years = _get(contract, "years_remaining", None) or _get(contract, "years_left", None)
        terms = []
        try:
            from app.sim_engine.economy.cap_engine import player_cap_hit_millions  # noqa: WPS433

            aav_m = float(player_cap_hit_millions(player) or 0.0)
            if aav_m > 0:
                terms.append(f"${aav_m:.2f}M cap hit")
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        if years:
            terms.append(f"{years} yr{'s' if str(years) != '1' else ''} left")
        body = (
            f"{name} ({int(round(_player_ovr(player)))} OVR) wants out of {abbr}"
            + (f" over {reason}" if reason and reason != "general" else "")
            + (
                f". Now that he has asked out, his price is down about {int(round(drop * 100))}% and his club "
                "usually moves him within a week or two."
                if stage <= 1
                else f". His value has fallen about {int(round(drop * 100))}% while the demand drags on, "
                "so the asking price is below what he is worth on the ice."
            )
            + (f" {', '.join(terms)}." if terms else "")
        )
        popup = {
            "id": key,
            "kind": "storyline",
            "presentation_type": "trade_opportunity",
            "theme": "info",
            "source_label": "Pro scouting",
            "headline": f"Buying chance: {name} is available below value",
            "body": body,
            "player_id": pid,
            "player_name": name,
            "team_id": str(_get(team, "team_id", "") or _get(team, "id", "") or ""),
            "value_drop_pct": round(drop * 100, 1),
            "actions": [{"id": "tradehub", "label": "Open Trade Desk"}],
            "priority": "HIGH",
        }
        pending.append(popup)
        session.pending_ui_popups = pending[-80:]
        _push_notification(session, {"headline": popup["headline"], "body": body, "player_id": pid, "kind": "trade_opportunity"})
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def process_trade_demand_day(session: Any, calendar_idx: int, day_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update stability for all players; open formal demands when warranted."""
    import random

    rng = getattr(getattr(session, "sim", None), "rng", None) or random.Random(int(calendar_idx) + 17)
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"opened": 0}

    invalidate_trade_demand_player_index(session)
    assign_league_agents(session)
    iso = ""
    try:
        iso = str((day_meta or {}).get("iso") or (day_meta or {}).get("date") or "")
    except Exception:
        iso = ""
    _resume_summer_pending_demands(session, int(calendar_idx), iso_date=iso)
    _close_crises_for_trade_deadline(session)
    _advance_user_crisis_timers(session, int(calendar_idx))
    sync_trade_demand_crises(session, tick_timers=False)
    cpu_aging = _age_cpu_demands(session, int(calendar_idx))
    deadline_ctx = get_trade_deadline_context(session)
    ensure_trade_demands(session)

    opened: List[Dict[str, Any]] = []
    abs_day = absolute_calendar_day(session, int(calendar_idx))
    if getattr(session, "_trade_demand_open_budget", None) is None:
        budget = 1.0
    else:
        budget = float(getattr(session, "_trade_demand_open_budget", 0.0) or 0.0) + DEMAND_OPEN_BUDGET_PER_DAY
    setattr(session, "_trade_demand_open_budget", budget)
    warnings: List[Dict[str, Any]] = []
    demand_candidates: List[Tuple[float, Any, Any, Dict[str, Any], str]] = []
    import zlib

    try:
        from services.league_governance import trade_demand_multiplier

        demand_mult = float(trade_demand_multiplier(session))
    except Exception:
        demand_mult = 1.0
    seen_pids: set = set()

    # One league-wide storyline sync for the whole pass (per-player syncs were O(N^2)).
    try:
        from app.sim_engine.franchise.storyline_engine import _u_sync_player_entities

        session._stability_entities_pass_cache = _u_sync_player_entities(session)
    except Exception:
        session._stability_entities_pass_cache = None
    try:
        teams = list(_get(league, "teams", None) or [])
        for team in teams:
            for player in list(_get(team, "roster", None) or []):
                if _get(player, "retired", False):
                    continue
                pid = _player_id(player)
                if not pid:
                    continue
                if active_demand_for_player(session, pid):
                    continue
                ovr = _player_ovr(player)
                if ovr < DEMAND_MIN_OVR:
                    continue
                pst = ensure_player_storyline_state(player)
                last_closed = int(pst.get("last_trade_demand_closed_day") or 0)
                if last_closed and 0 <= (abs_day - last_closed) < DEMAND_REOPEN_COOLDOWN_DAYS:
                    continue

                stability_row = apply_daily_stability_update(session, player, team, int(calendar_idx))
                escalation = int(stability_row.get("escalation_level") or 0)
                score = float(stability_row.get("trade_stability_score") or 100.0)

                if escalation in (1, 2):
                    _maybe_enqueue_stability_warning(session, player, team, stability_row, calendar_idx, iso, rng)
                elif escalation == 0:
                    _maybe_enqueue_stability_concern_hint(
                        session, player, team, stability_row, calendar_idx, iso, rng
                    )
                # Roster chip + dossier label follow the player's CURRENT state (they
                # used to stick forever once set, so every player ended up "Frustrated").
                _sync_stability_surface(session, player, team, stability_row)
                seen_pids.add(pid)

                # Board of Governors trade-request rules: >1 lets some level-2 players
                # go formal; <1 talks some level-3 players down.
                gate_u = (zlib.crc32(f"tdgate|{pid}|{calendar_idx}".encode()) & 0xFFFFFFFF) / 0xFFFFFFFF
                formal_level = escalation >= 3 and gate_u < min(1.0, demand_mult)
                if not formal_level and escalation == 2 and demand_mult > 1.0:
                    formal_level = gate_u < (demand_mult - 1.0) * 0.5
                if (
                    formal_level
                    and formal_demand_eligible(stability_row)
                    and deadline_ctx.get("new_demands_allowed")
                ):
                    reason, _codes = _reason_from_pressures(
                        dict(stability_row.get("pressures") or {}),
                        character=int(stability_row.get("character") or 74),
                        player=player,
                        team=team,
                        rng=rng,
                    )
                    priority = float(score)
                    demand_candidates.append((priority, player, team, stability_row, reason))
                elif escalation >= 1 and score < 68:
                    warnings.append({"player_id": pid, "escalation_level": escalation, "score": score})

    finally:
        session._stability_entities_pass_cache = None

    # Drop flags for players no longer evaluated (traded away, below the OVR floor, retired)
    # unless they carry an open formal demand.
    flags = getattr(session, "trade_stability_roster_flags", None)
    if isinstance(flags, dict) and flags:
        session.trade_stability_roster_flags = {
            k: v for k, v in flags.items() if k in seen_pids or active_demand_for_player(session, k)
        }

    demand_candidates.sort(key=lambda t: t[0])
    team_open, team_recent = _team_demand_load(session, int(calendar_idx))
    opened_this_pass = 0
    for _prio, player, team, stability_row, reason in demand_candidates:
        if budget < 1.0:
            break
        if opened_this_pass >= max(1, int(round(MAX_NEW_FORMAL_DEMANDS_PER_PASS * demand_mult))):
            break
        tkey = _team_key(team)
        if team_open.get(tkey, 0) >= MAX_OPEN_DEMANDS_PER_TEAM:
            continue
        if team_recent.get(tkey, 0) >= MAX_DEMANDS_PER_TEAM_WINDOW:
            continue
        row = open_trade_demand(
            session,
            player,
            team,
            reason=reason,
            calendar_idx=int(calendar_idx),
            iso_date=iso,
            rng=rng,
            stability_row=stability_row,
            force_formal=False,
        )
        if row.get("status") == "open":
            opened.append(row)
            opened_this_pass += 1
            budget = max(0.0, budget - 1.0)
            setattr(session, "_trade_demand_open_budget", budget)
            team_open[tkey] = team_open.get(tkey, 0) + 1
            team_recent[tkey] = team_recent.get(tkey, 0) + 1

    return {
        "opened": len(opened),
        "demands": opened,
        "warnings": len(warnings),
        "cpu_demands_aged": cpu_aging["aged"],
        "cpu_demands_resolved": cpu_aging["resolved"],
    }


def build_trade_demand_crisis_payload(session: Any, *, tick_timers: bool = False) -> Optional[Dict[str, Any]]:
    """Active user-team crisis for fullscreen overlay."""
    sync_trade_demand_crises(session, tick_timers=tick_timers)
    deadline_ctx = get_trade_deadline_context(session)
    user_tid = str(getattr(session, "user_team_id", "") or "")
    book = ensure_trade_demands(session)
    for row in book.values():
        if not isinstance(row, dict) or str(row.get("status")) != "open":
            continue
        if str(row.get("team_id") or "") != user_tid:
            continue
        crisis = row.get("crisis")
        if not isinstance(crisis, dict):
            continue
        crisis = _hydrate_crisis_timer(crisis)
        remaining_days = float(crisis.get("remaining_days") or 0)
        initial_days = int(crisis.get("initial_days") or 18)
        remaining = remaining_days * 86400.0
        initial = initial_days * 86400
        stage = int(crisis.get("crisis_stage") or 1)
        dests = list(row.get("preferred_destinations") or [])
        return {
            "demand_id": row.get("demand_id"),
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "team_id": row.get("team_id"),
            "headline": row.get("headline"),
            "body": row.get("body"),
            "primary_complaint": row.get("primary_complaint"),
            "escalation_level": row.get("escalation_level"),
            "trade_stability_score": row.get("trade_stability_score"),
            "remaining_seconds": round(remaining, 1),
            "remaining_days": round(remaining_days, 2),
            "initial_seconds": initial,
            "initial_days": initial_days,
            "crisis_stage": stage,
            "deadline_seconds": initial,
            "timer_active": bool(crisis.get("timer_active")),
            "timer_frozen": bool(crisis.get("timer_frozen")) or not deadline_ctx.get("crisis_timer_ticks"),
            "trade_deadline": deadline_ctx,
            "value_before": row.get("value_before"),
            "value_after": row.get("value_after"),
            "preferred_destinations": dests,
            "destination_count": len(dests),
            "agent": row.get("agent"),
            "leaked": bool(row.get("leaked")),
            "public_demand": bool(row.get("public_demand")),
            "disruptor": bool(row.get("disruptor")),
        }
    return None


_STABILITY_TAGS = {
    0: "Monitor — early concern",
    1: "Growing frustration",
    2: "Disconnecting from organization",
    3: "Unhappy — trade demand risk",
    4: "Locker-room crisis",
}


def _sync_stability_surface(session: Any, player: Any, team: Any, stability_row: Dict[str, Any]) -> None:
    """Make the roster flag and dossier label match today's stability — set, update or clear."""
    from app.sim_engine.franchise.trade_stability_engine import count_significant_pressures

    pid = _player_id(player)
    level = int(stability_row.get("escalation_level") or 0)
    score = float(stability_row.get("trade_stability_score") or 100.0)
    pressures = dict(stability_row.get("pressures") or {})
    character = int(stability_row.get("character") or 74)
    if level >= 1:
        tag: Optional[str] = _STABILITY_TAGS.get(level, _STABILITY_TAGS[4])
    elif character >= 58 and 45.0 < score < 72.0 and count_significant_pressures(pressures) >= 1:
        tag = _STABILITY_TAGS[0]
    else:
        tag = None

    old_labels = list(getattr(player, "dossier_labels", None) or [])
    stale = set(_STABILITY_TAGS.values())
    new_labels = [lb for lb in old_labels if lb not in stale]
    if tag:
        new_labels.append(tag)
    if new_labels != old_labels:
        try:
            setattr(player, "dossier_labels", new_labels[-8:])
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    flags = getattr(session, "trade_stability_roster_flags", None)
    if not isinstance(flags, dict):
        flags = {}
        session.trade_stability_roster_flags = flags
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if tag is None or tid != user_tid:
        flags.pop(pid, None)
        return
    flags[pid] = {
        "player_id": pid,
        "player_name": _player_name(player),
        "escalation_level": level,
        "score": score,
        "label": tag,
        "top_pressure": max(pressures.items(), key=lambda kv: kv[1], default=("role", 0))[0],
    }


_COMPLAINT_FIX = {
    "role": "more ice time or a move up the lineup",
    "management": "a sit-down, and keeping your word",
    "winning": "a better roster around him; he wants to win",
    "contract": "an extension or a talk about his deal",
    "development": "a bigger role to grow into",
    "belonging": "support from the leadership group",
    "trade_exposure": "reassurance that he's not on the block",
    "broken_promise": "delivering on the promise you made",
    "performance": "usage that matches his production",
    "organizational": "a clear plan for where the team is going",
    "temperament": "giving him a role he accepts, or a winning run to calm him",
}


def _stability_warning_body(session: Any, player: Any, name: str, agent: Dict[str, Any],
                            stability_row: Dict[str, Any]) -> str:
    """Why he's unhappy, with his numbers, how close he is to a demand, and what helps.

    The old text was "agent has reached out regarding his satisfaction (stability 61)"
    with no reason at all."""
    from app.sim_engine.franchise.trade_stability_engine import (  # noqa: WPS433
        STABILITY_APATHY_MIN,
        primary_complaint_from_pressures,
    )

    pressures = {k: float(v or 0) for k, v in dict(stability_row.get("pressures") or {}).items()}
    ranked = sorted(pressures.items(), key=lambda kv: -kv[1])
    top = ranked[0][0] if ranked else ""
    complaint = primary_complaint_from_pressures(pressures)
    second = ""
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * 0.7:
        second = primary_complaint_from_pressures({ranked[1][0]: 1.0})
    agent_name = str(agent.get("name") or "His agent")
    last = name.split(" ")[-1] if name else "He"
    parts = [f"{agent_name} called about {last}. His main concern is {complaint}"
             + (f", followed by {second}" if second else "") + "."]
    row = (getattr(session, "player_season_stats", None) or {}).get(_player_id(player)) or {}
    gp = int(row.get("gp") or 0)
    if gp > 0 and str(row.get("position") or "").upper() != "G":
        toi = int(row.get("toi_sec") or 0) / gp
        g, a = int(row.get("g") or 0), int(row.get("a") or 0)
        parts.append(f"This season: {g}-{a}-{g + a} in {gp} games, {int(toi // 60)}:{int(toi % 60):02d} a night.")
    try:
        score = float(stability_row.get("trade_stability_score") or 0)
        parts.append(f"Stability {score:.0f}/100; a trade request becomes possible below {STABILITY_APATHY_MIN}.")
    except (TypeError, ValueError):
        pass
    fix = _COMPLAINT_FIX.get(top)
    if fix:
        parts.append(f"What would help: {fix}.")
    parts.append("No formal trade demand yet.")
    return " ".join(parts)


def _maybe_enqueue_stability_warning(
    session: Any,
    player: Any,
    team: Any,
    stability_row: Dict[str, Any],
    calendar_idx: int,
    iso_date: str,
    rng: Any,
) -> None:
    """Surface L1–L2 angst/apathy to user + league notifications."""
    pid = _player_id(player)
    pst = ensure_player_storyline_state(player)
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    warn_key = f"stability_warn:{pid}:{season}:{int(stability_row.get('escalation_level') or 0)}"
    if warn_key in set(pst.get("stability_warnings_sent") or []):
        return
    sent = list(pst.get("stability_warnings_sent") or [])
    sent.append(warn_key)
    pst["stability_warnings_sent"] = sent[-24:]

    level = int(stability_row.get("escalation_level") or 0)
    labels = {1: "Growing frustration", 2: "Disconnecting from organization"}
    name = _player_name(player)
    agent = agent_public_view(player, session)
    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    is_user = tid == user_tid

    labels_dossier = list(getattr(player, "dossier_labels", None) or [])
    tag = labels.get(level, "Trade concern")
    if tag not in labels_dossier:
        labels_dossier.append(tag)
        try:
            setattr(player, "dossier_labels", labels_dossier)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    body = _stability_warning_body(session, player, name, agent, stability_row)
    popup = {
        "id": warn_key,
        "kind": "storyline",
        "presentation_type": "trade_stability_warning",
        "theme": "warn" if level == 1 else "danger",
        "source_label": f"Agent — {agent.get('name', 'Representative')}",
        "headline": f"{name} — {tag}",
        "body": body,
        "player_id": pid,
        "player_name": name,
        "team_id": tid,
        "trade_stability": {
            "score": stability_row.get("trade_stability_score"),
            "escalation_level": level,
            "agent": agent,
            "primary_complaint": primary_complaint_from_pressures(dict(stability_row.get("pressures") or {})),
        },
        "priority": "HIGH" if is_user else "MID",
    }
    if not is_user:
        # Other clubs' simmering players stay off the feed (the dossier label above still
        # shows it); only a formal demand is league news.
        return

    pending = list(getattr(session, "pending_ui_popups", None) or [])
    pending.append(popup)
    session.pending_ui_popups = pending[-80:]

    _push_notification(
        session,
        {
            "id": warn_key,
            "type": "trade_stability_warning",
            "headline": popup["headline"],
            "body": popup["body"],
            # The feed reads title/text like every other notification.
            "title": popup["headline"],
            "text": popup["body"],
            "priority": popup["priority"],
            "source": "trade_stability",
            "team_id": tid,
            "player_id": pid,
        },
    )

    roster_flags = dict(getattr(session, "trade_stability_roster_flags", None) or {})
    roster_flags[pid] = {
        "player_id": pid,
        "player_name": name,
        "escalation_level": level,
        "score": float(stability_row.get("trade_stability_score") or 0),
        "label": tag,
        "top_pressure": max(
            (stability_row.get("pressures") or {}).items(),
            key=lambda kv: kv[1],
            default=("role", 0),
        )[0],
    }
    session.trade_stability_roster_flags = roster_flags


def _maybe_enqueue_stability_concern_hint(
    session: Any,
    player: Any,
    team: Any,
    stability_row: Dict[str, Any],
    calendar_idx: int,
    iso_date: str,
    rng: Any,
) -> None:
    """Early 'something feels off' signals for non-volatile players (L0, score slipping)."""
    from app.sim_engine.franchise.trade_stability_engine import count_significant_pressures

    pid = _player_id(player)
    character = int(stability_row.get("character") or 74)
    if character < 58:
        return

    score = float(stability_row.get("trade_stability_score") or 100.0)
    if score >= 72 or score <= 45:
        return

    pressures = dict(stability_row.get("pressures") or {})
    if count_significant_pressures(pressures) < 1:
        return

    pst = ensure_player_storyline_state(player)
    last_hint = int(pst.get("last_stability_hint_abs_day") or 0)
    if last_hint and (absolute_calendar_day(session, calendar_idx) - last_hint) < 28:
        return
    top = max(pressures.items(), key=lambda kv: kv[1])[0] if pressures else "role"
    if str(pst.get("last_stability_hint_complaint") or "") == str(top) and last_hint:
        return
    pst["last_stability_hint_abs_day"] = absolute_calendar_day(session, calendar_idx)
    pst["last_stability_hint_complaint"] = str(top)
    hints = list(pst.get("stability_hints_sent") or [])
    warn_key = f"stability_hint:{pid}:{int(calendar_idx // 7)}"
    hints.append(warn_key)
    pst["stability_hints_sent"] = hints[-20:]

    tid = str(_get(team, "team_id", "") or _get(team, "id", "") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if tid != user_tid:
        return

    name = _player_name(player)
    agent = agent_public_view(player, session)
    top = max(pressures.items(), key=lambda kv: kv[1])[0] if pressures else "role"
    complaint = primary_complaint_from_pressures(pressures)

    labels_dossier = list(getattr(player, "dossier_labels", None) or [])
    tag = "Monitor — early concern"
    if tag not in labels_dossier:
        labels_dossier.append(tag)
        try:
            setattr(player, "dossier_labels", labels_dossier[-8:])
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)

    popup = {
        "id": warn_key,
        "kind": "storyline",
        "presentation_type": "trade_stability_hint",
        "theme": "info",
        "source_label": f"Agent — {agent.get('name', 'Representative')}",
        "headline": f"{name} — agent checking in",
        "body": (
            f"{agent.get('name', 'The agent')} called about {name}'s {complaint}. "
            f"No formal demand — but the situation bears watching (stability {score:.0f})."
        ),
        "player_id": pid,
        "player_name": name,
        "team_id": tid,
        "trade_stability": {
            "score": score,
            "escalation_level": 0,
            "top_pressure": top,
            "agent": agent,
        },
        "priority": "MID",
    }
    pending = list(getattr(session, "pending_ui_popups", None) or [])
    pending.append(popup)
    session.pending_ui_popups = pending[-80:]

    roster_flags = dict(getattr(session, "trade_stability_roster_flags", None) or {})
    roster_flags[pid] = {
        "player_id": pid,
        "player_name": name,
        "escalation_level": 0,
        "score": score,
        "label": tag,
        "top_pressure": top,
    }
    session.trade_stability_roster_flags = roster_flags


def _enqueue_crisis_expired(session: Any, row: Dict[str, Any]) -> None:
    tid = str(row.get("team_id") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    if tid != user_tid:
        return
    _push_notification(
        session,
        {
            "id": f"{row.get('demand_id')}:expired",
            "type": "trade_demand_crisis_expired",
            "headline": f"Crisis expired — {row.get('player_name')} is now a distressed asset",
            "body": "You may need to attach assets just to move him.",
            "priority": "HIGH",
            "player_id": row.get("player_id"),
            "team_id": tid,
        },
    )


def _enqueue_demand_surfaces(session: Any, row: Dict[str, Any], team: Any) -> None:
    tid = str(row.get("team_id") or "")
    user_tid = str(getattr(session, "user_team_id", "") or "")
    is_user = tid == user_tid
    dests = list(row.get("preferred_destinations") or [])
    dest_label = (
        "All 32 clubs"
        if len(dests) >= 30
        else (", ".join(dests[:8]) + ("…" if len(dests) > 8 else ""))
        if dests
        else "Open market"
    )
    crisis = row.get("crisis") or {}
    popup = {
        "id": str(row.get("demand_id")),
        "kind": "storyline",
        "presentation_type": "trade_demand_crisis" if row.get("formal") else (
            "trade_demand_disruptor" if row.get("disruptor") else "trade_demand"
        ),
        "theme": "danger",
        "source_label": "Trade Demand Crisis",
        "headline": row.get("headline"),
        "body": row.get("body"),
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_id": tid,
        "cause_type": "TRADE_DEMAND",
        "cause_event_id": row.get("cause_event_id"),
        "trade_demand": {
            "reason_code": row.get("reason_code"),
            "primary_complaint": row.get("primary_complaint"),
            "value_before": row.get("value_before"),
            "value_after": row.get("value_after"),
            "value_delta": row.get("value_delta"),
            "preferred_destinations": dests,
            "destination_label": dest_label,
            "disruptor": bool(row.get("disruptor")),
            "dossier_label": row.get("dossier_label"),
            "agent": row.get("agent"),
            "crisis_stage": row.get("crisis_stage"),
            "remaining_seconds": crisis.get("remaining_seconds"),
            "initial_seconds": crisis.get("initial_seconds"),
            "escalation_level": row.get("escalation_level"),
            "formal_crisis": True,
        },
        "priority": "CRITICAL" if is_user else "HIGH",
    }
    if is_user:
        pending = list(getattr(session, "pending_ui_popups", None) or [])
        pending.append(popup)
        session.pending_ui_popups = pending[-80:]

    try:
        from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433

        _record_storyline(
            session,
            {
                "id": row.get("demand_id"),
                "storyline_id": str(row.get("demand_id") or ""),
                "type": "trade_demand",
                "category": "trade",
                "headline": row.get("headline"),
                "summary": row.get("body"),
                "body": row.get("body"),
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name"),
                "team_id": tid,
                "tone": "negative",
                "cause_type": "TRADE_DEMAND",
                "priority": "CRITICAL" if is_user else "HIGH",
                "trade_demand": popup["trade_demand"],
                "calendar_iso": row.get("opened_iso") or "",
                "calendar_day": row.get("opened_day"),
                # Silly-reason demands also belong in the Rink Oddities section.
                "goofy": str(row.get("reason_code") or "").startswith("silly_") or None,
                "goofy_kind": "silly_trade_demand" if str(row.get("reason_code") or "").startswith("silly_") else None,
            },
        )
    except Exception:
        events = list(getattr(session, "storyline_events", None) or [])
        events.append(
            {
                "id": row.get("demand_id"),
                "type": "trade_demand",
                "headline": row.get("headline"),
                "body": row.get("body"),
                "player_id": row.get("player_id"),
                "team_id": tid,
                "tone": "negative",
                "cause_type": "TRADE_DEMAND",
                "trade_demand": popup["trade_demand"],
            }
        )
        session.storyline_events = events[-200:]

    remaining = crisis.get("remaining_seconds")
    timer_label = f"{int(float(remaining or 0) // 60)}:{int(float(remaining or 0) % 60):02d}" if remaining else "6:00"
    if is_user:
        _push_notification(
            session,
            {
                "id": row.get("demand_id"),
                "type": "trade_demand",
                "headline": row.get("headline"),
                "body": f"CRISIS {timer_label} · TV {row.get('value_before')} → {row.get('value_after')} · {dest_label}",
                "priority": "CRITICAL",
                "date": row.get("opened_iso") or "",
                "team_id": tid,
                "player_id": row.get("player_id"),
            },
        )
    elif float(row.get("ovr") or 0) >= CPU_DEMAND_NEWS_MIN_OVR:
        # Notable CPU players asking out is league news, not an inbox crisis.
        team_name = str(_get(team, "name", "") or _get(team, "abbr", "") or tid)
        _push_notification(
            session,
            {
                "id": f"{row.get('demand_id')}:news",
                "type": "league_news",
                "headline": f"League: {row.get('player_name')} asks out of {team_name}",
                "body": f"{row.get('player_name')} ({row.get('ovr')} OVR) has requested a trade · {dest_label}",
                "priority": "MEDIUM",
                "source": "league_news",
                "date": row.get("opened_iso") or "",
                "team_id": tid,
                "player_id": row.get("player_id"),
            },
        )


def clear_demands_on_trade(session: Any, player_ids: List[str]) -> None:
    for pid in player_ids:
        if not pid:
            continue
        clear_trade_demand(session, pid, resolution="traded")
        player = _find_player(session, pid)
        if player is not None:
            clear_demand_temporary_modifiers(player)


# Re-export for storyline trade hub hook
__all__ = [
    "apply_trade_hub_exposure",
    "build_trade_demand_crisis_payload",
    "sync_trade_demand_crises",
    "process_trade_demand_day",
    "open_trade_demand",
    "clear_demands_on_trade",
]
